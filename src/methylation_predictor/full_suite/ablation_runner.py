"""Controlled architecture-ablation runner for mixed TCGA sources.

This deliberately lives beside, rather than inside, the production E2-E4
trainer so architecture experiments cannot silently change historical runs.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from methylation_predictor.full_coverage_sampler import build_epoch_schedule

from .trainer import MixedSuiteRunner


def checkpoint_metric_value(validation: dict[str, float | int], metric: str) -> tuple[float, bool]:
    """Return ``(value, higher_is_better)`` for supported dev metrics."""
    directions = {
        "mse": False,
        "mae": False,
        "mas_pcc": True,
        "mac_pcc": True,
        "skill_vs_prior": True,
    }
    if metric not in directions:
        raise ValueError(
            "architecture-ablation checkpoint_metric must be one of "
            f"{sorted(directions)}; got {metric!r}"
        )
    value = float(validation[metric])
    if not np.isfinite(value):
        value = -float("inf") if directions[metric] else float("inf")
    return value, directions[metric]


def interleave_full_coverage_schedules(schedules) -> list[tuple[int, int]]:
    """Consume every source-local schedule once, interleaved by progress."""
    entries: list[tuple[float, int, int]] = []
    for source_index, schedule in enumerate(schedules):
        n = len(schedule)
        entries.extend(((step + 0.5) / n, source_index, step) for step in range(n))
    entries.sort(key=lambda x: (x[0], x[1]))
    return [(source_index, step) for _, source_index, step in entries]


class ArchitectureAblationRunner(MixedSuiteRunner):
    """MixedSuiteRunner with deterministic full coverage and metric selection."""

    def _train_stage(self, stage: str, epochs: int, *, select_best: bool):
        stage_dir = self.output / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        done = stage_dir / ".done"
        best_path = stage_dir / "best.pt"
        latest = stage_dir / "latest.pt"
        if done.is_file() and best_path.is_file():
            state = torch.load(best_path, map_location=self.device, weights_only=False)
            model = self._new_model()
            model.load_state_dict(state["model_state"])
            return model, int(state["epoch"])

        model = self._new_model()
        opt_kwargs = dict(
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        if self.cfg.training.fused_adamw:
            opt_kwargs["fused"] = True
        try:
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        except (TypeError, RuntimeError):
            opt_kwargs.pop("fused", None)
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)

        pools = self._pools(stage)
        seen_cpg = {p.name: np.zeros(len(p.cpg_idx), dtype=bool) for p in pools}
        seen_rows = {p.name: np.zeros(len(p.row_positions), dtype=bool) for p in pools}
        metric_name = self.cfg.training.checkpoint_metric
        _, higher_is_better = checkpoint_metric_value(
            {"mse": 0.0, "mae": 0.0, "mas_pcc": 0.0, "mac_pcc": 0.0, "skill_vs_prior": 0.0},
            metric_name,
        )
        best_metric = -float("inf") if higher_is_better else float("inf")
        best_epoch = 0
        bad = 0
        history = []
        start_epoch = 1

        if latest.is_file():
            state = torch.load(latest, map_location=self.device, weights_only=False)
            previous_metric = state.get("best_metric_name")
            if previous_metric is not None and previous_metric != metric_name:
                raise RuntimeError(
                    f"cannot resume {stage_dir}: checkpoint selected {previous_metric!r}, "
                    f"current config requests {metric_name!r}"
                )
            model.load_state_dict(state["model_state"])
            opt.load_state_dict(state["optimizer_state"])
            start_epoch = int(state["epoch"]) + 1
            best_metric = float(state.get("best_metric", best_metric))
            best_epoch = int(state.get("best_epoch", 0))
            bad = int(state.get("bad", 0))
            if "rng_state" in state:
                self.rng.bit_generator.state = state["rng_state"]["numpy_generator"]
                random.setstate(state["rng_state"]["python"])
                torch.set_rng_state(state["rng_state"]["torch_cpu"].to(dtype=torch.uint8, device="cpu"))
                if state["rng_state"].get("torch_cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(
                        [x.to(dtype=torch.uint8, device="cpu") for x in state["rng_state"]["torch_cuda"]]
                    )
            history_path = stage_dir / "history.json"
            if history_path.is_file():
                history = json.loads(history_path.read_text())

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            started = time.time()
            observed = 0
            source_steps = {p.name: 0 for p in pools}
            losses = []
            epoch_seen_cpg = {p.name: np.zeros(len(p.cpg_idx), dtype=bool) for p in pools}
            epoch_seen_rows = {p.name: np.zeros(len(p.row_positions), dtype=bool) for p in pools}

            schedules = [
                build_epoch_schedule(
                    np.arange(len(p.cpg_idx), dtype=np.int64),
                    np.arange(len(p.row_positions), dtype=np.int64),
                    self.block_cpgs[p.name],
                    self.block_rows[p.name],
                    epoch,
                    self.seed + 1009 * source_index,
                )
                for source_index, p in enumerate(pools)
            ]
            plan = interleave_full_coverage_schedules(schedules)

            for source_index, local_step in plan:
                pool = pools[source_index]
                row_slots, cpg_slots = schedules[source_index][local_step]
                sample_ids = pool.sample_idx[row_slots]
                cpg_ids = pool.cpg_idx[cpg_slots]
                beta = self._block(pool, row_slots, cpg_slots)
                if not np.isfinite(beta).any():
                    continue
                opt.zero_grad(set_to_none=True)
                loss, pieces = self._tensor_forward(model, sample_ids, cpg_ids, beta)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss: {pieces}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.gradient_clip_norm)
                opt.step()

                observed += int(pieces["observed"])
                losses.append(float(pieces["loss"]))
                source_steps[pool.name] += 1
                seen_rows[pool.name][row_slots] = True
                seen_cpg[pool.name][cpg_slots] = True
                epoch_seen_rows[pool.name][row_slots] = True
                epoch_seen_cpg[pool.name][cpg_slots] = True

            validation = None
            metric_value = None
            if select_best:
                validation = self._evaluate_array(model, self.dev_s, self.dev_c)
                metric_value, higher_is_better = checkpoint_metric_value(validation, metric_name)
                improved = (
                    metric_value > best_metric + self.cfg.training.min_delta
                    if higher_is_better
                    else metric_value < best_metric - self.cfg.training.min_delta
                )
                if improved:
                    best_metric = metric_value
                    best_epoch = epoch
                    bad = 0
                    self._save_checkpoint(best_path, model, epoch, validation)
                else:
                    bad += 1
            else:
                best_epoch = epoch
                self._save_checkpoint(best_path, model, epoch, None)

            row = {
                "epoch": epoch,
                "seconds": time.time() - started,
                "observed": observed,
                "train_loss": float(np.mean(losses)),
                "source_steps": source_steps,
                "coverage_cpg_epoch": {k: float(v.mean()) for k, v in epoch_seen_cpg.items()},
                "coverage_rows_epoch": {k: float(v.mean()) for k, v in epoch_seen_rows.items()},
                "coverage_cpg_cumulative": {k: float(v.mean()) for k, v in seen_cpg.items()},
                "coverage_rows_cumulative": {k: float(v.mean()) for k, v in seen_rows.items()},
                "checkpoint_metric": metric_name,
                "checkpoint_metric_value": metric_value,
                "validation": validation,
            }
            history.append(row)
            (stage_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

            payload = {
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "best_metric_name": metric_name,
                "best_epoch": best_epoch,
                "bad": bad,
                "rng_state": {
                    "numpy_generator": self.rng.bit_generator.state,
                    "python": random.getstate(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            }
            tmp = latest.with_suffix(".tmp.pt")
            torch.save(payload, tmp)
            os.replace(tmp, latest)
            print(f"[{self.protocol.name}:{stage}:ablation] {row}", flush=True)

            if select_best and epoch >= self.cfg.training.min_epochs and bad >= self.cfg.training.patience:
                break

        if select_best:
            if not best_path.is_file():
                raise RuntimeError("development stage produced no best checkpoint")
            state = torch.load(best_path, map_location=self.device, weights_only=False)
            model.load_state_dict(state["model_state"])
            best_epoch = int(state["epoch"])
        done.write_text("ok\n")
        return model, best_epoch

    def run(self):
        result = super().run()
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["training_schedule"] = "deterministic_full_coverage_per_source"
        manifest["checkpoint_metric"] = self.cfg.training.checkpoint_metric
        manifest["architecture_ablation_runner"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        result["training_schedule"] = "deterministic_full_coverage_per_source"
        result["checkpoint_metric"] = self.cfg.training.checkpoint_metric
        headline = self.output / "evaluation" / "headline.json"
        headline.write_text(json.dumps(result, indent=2) + "\n")
        return result
