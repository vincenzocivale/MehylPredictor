"""Ablation-only single-stage joint trainer.

Tests whether the canonical two-stage architecture -- CpGStatisticsPredictor
(mu/sigma head) trained first and frozen, exported to a static feature cache,
then RNAMethylationPredictor trained second against that frozen prior (see
``rna_training/trainer.py::ScopedRNATrainer``) -- is actually necessary, or
whether training both heads together in a single stage is competitive.

Unlike ``ScopedRNATrainer``, ``prior``/``sigma`` are *not* read from the
precomputed ``LocusFeatureCache.prior``/``.sigma`` arrays. They come from a
live, autograd-tracked forward pass of a ``CpGStatisticsPredictor`` held by
this trainer, run on the same NTv3 embedding tensor already loaded for the
step (``LocusFeatureCache.embeddings``). Both models are optimized jointly
against a combined loss: the usual RNA-side ``residual_loss`` plus an
auxiliary CpG-statistics supervision term (mu_logit/log_sigma MSE against the
same targets ``CpGStatisticsTrainer`` uses) so mu/sigma keep their meaning as
locus statistics rather than becoming a free parameter of the RNA branch.
``cpg_warm_start_checkpoint=None`` gives the end-to-end-from-scratch variant;
setting it to a trained stage-1 checkpoint (with a low ``cpg_learning_rate``)
gives the joint-fine-tuning variant.

Deliberately isolated from ``ScopedRNATrainer``/``MethylProphetTrainer`` (see
CLAUDE.md: don't route generic-pipeline changes through the benchmark
trainer) -- this is a standalone, throwaway ablation trainer, Array-source
only, intended for chr1. It is not wired into ``scripts/train.py``'s stable
CLI; run it via ``scripts/experiments/train_joint_ablation.py``. Delete both
once the joint-training ablation concludes.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..cpg_statistics.model import CpGStatisticsModelConfig, CpGStatisticsPredictor
from ..losses import residual_loss
from ..models import RNAMethylationPredictor
from ..optim import build_lr_scheduler
from ..run_store import RunStore, write_json
from ..scopes import scope_protocol
from ..storage import LocusFeatureCache, RNACache, SortedIndex
from ..tcga_canonical import TCGACanonicalBundle
from .config import load_rna_recipe
from .matched_chr1_data import load_matched_chr1_protocol_and_sources
from .metrics import ArrayMomentMetrics
from .schedule import SourceSchedule, interleave
from .splits import blocked_cpg_split, stratified_sample_split
from .trainer import TrainingPool, loss_config_for_source


class JointRNAMethylationTrainer:
    def __init__(
        self,
        *,
        canonical_root: str | Path,
        scope: str,
        recipe_path: str | Path,
        feature_cache: str | Path,
        rna_cache: str | Path,
        registry: str | Path,
        cpg_targets_dir: str | Path,
        output_root: str | Path,
        matched_chr1_root: str | Path | None = None,
        cpg_warm_start_checkpoint: str | Path | None = None,
        cpg_learning_rate: float = 2e-4,
        joint_loss_weight: float = 1.0,
        cpg_mu_weight: float = 1.0,
        cpg_sigma_weight: float = 1.0,
        mode: str = "final",
        run_id: str | None = None,
        overrides: dict | None = None,
    ):
        if mode not in {"development", "final"}:
            raise ValueError("mode must be development or final")
        self.mode = mode
        self.scope = scope
        self.root = Path(canonical_root)
        self.registry = Path(registry)
        self.recipe = load_rna_recipe(recipe_path)
        for key, value in (overrides or {}).items():
            if not hasattr(self.recipe.training, key):
                raise ValueError(f"unknown TrainingConfig override {key!r}")
            setattr(self.recipe.training, key, value)
        cfg = self.recipe.training
        self.seed = int(cfg.seed)
        self.epochs = int(cfg.epochs)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("joint RNA training requires CUDA")
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32

        self.matched_chr1_root = Path(matched_chr1_root) if matched_chr1_root else None
        if self.matched_chr1_root is not None:
            if scope != "chr1":
                raise ValueError("matched_chr1_root is only valid for scope='chr1'")
            self.bundle = None
            self.protocol, self._sources = load_matched_chr1_protocol_and_sources(
                self.matched_chr1_root, self.root,
            )
        else:
            self.bundle = TCGACanonicalBundle.from_root(self.root)
            self.protocol = scope_protocol(scope, self.bundle, canonical_root=self.root)
            self._sources = self.bundle.sources
        self.features = LocusFeatureCache(feature_cache)
        self.rna = RNACache(rna_cache)
        required = np.unique(np.concatenate([
            self.protocol.array_train_cpg_idx, self.protocol.array_val_cpg_idx,
            *self.protocol.auxiliary_cpg_idx.values(),
        ]))
        self.features.index.positions_of(required)

        self.model = RNAMethylationPredictor(25_017, 1536, self.recipe.model, epsilon=1e-4).to(self.device)

        cpg_targets_dir = Path(cpg_targets_dir)
        self.cpg_target_ids = np.load(cpg_targets_dir / "cpg_idx.npy")
        self.cpg_target_mu = np.load(cpg_targets_dir / "target_mu.npy")
        self.cpg_target_sigma = np.load(cpg_targets_dir / "target_sigma.npy")
        self.cpg_target_index = SortedIndex(self.cpg_target_ids, "cpg_statistics targets")
        self.cpg_mu_weight = float(cpg_mu_weight)
        self.cpg_sigma_weight = float(cpg_sigma_weight)
        self.joint_loss_weight = float(joint_loss_weight)

        self.cpg_model_cfg = CpGStatisticsModelConfig()
        self.cpg_model = CpGStatisticsPredictor(self.cpg_model_cfg).to(self.device)
        self.cpg_warm_start_checkpoint = str(cpg_warm_start_checkpoint) if cpg_warm_start_checkpoint else None
        if self.cpg_warm_start_checkpoint:
            state = torch.load(self.cpg_warm_start_checkpoint, map_location=self.device, weights_only=False)
            self.cpg_model_cfg = CpGStatisticsModelConfig(**state["model_config"])
            self.cpg_model = CpGStatisticsPredictor(self.cpg_model_cfg).to(self.device)
            self.cpg_model.load_state_dict(state["model_state"])
        self.cpg_learning_rate = float(cpg_learning_rate)
        # cpg_learning_rate=0.0 reproduces the canonical two-stage *frozen*-prior
        # baseline on this trainer's array-only pool infrastructure (used because
        # the generic ScopedRNATrainer cannot run at chr1: rna_feature_cache/chr1
        # only covers the official Array CpGs, not the full tcga_mix_chr1
        # multi-technology universe ScopedRNATrainer.__init__ requires regardless
        # of which pools are actually used -- see fusion_mechanism_2026_08's note
        # in results/reference/ablations.yaml). Requires cpg_warm_start_checkpoint
        # (a trained stage-1 checkpoint) to mean anything as a baseline.
        self.freeze_cpg_model = self.cpg_learning_rate == 0.0
        if self.freeze_cpg_model:
            if not self.cpg_warm_start_checkpoint:
                raise ValueError("cpg_learning_rate=0.0 (frozen baseline) requires cpg_warm_start_checkpoint")
            self.cpg_model.requires_grad_(False)
            self.cpg_model.eval()

        self.inner_views = None
        self.pools = self._build_pools()
        self.store = RunStore.create(
            output_root, model="rna_methylation_joint", train_scope=scope, seed=self.seed,
            learning_rate=cfg.learning_rate, scheduler=cfg.scheduler, epochs=self.epochs, run_id=run_id,
        )
        self.store.save_resolved_config({
            **self.recipe.raw,
            "joint": {
                "cpg_warm_start_checkpoint": self.cpg_warm_start_checkpoint,
                "cpg_learning_rate": self.cpg_learning_rate,
                "joint_loss_weight": self.joint_loss_weight,
                "cpg_mu_weight": self.cpg_mu_weight,
                "cpg_sigma_weight": self.cpg_sigma_weight,
            },
        })

    def close(self) -> None:
        if self.bundle is not None:
            self.bundle.close()
        else:
            for source in self._sources.values():
                source.close()

    def _build_pools(self) -> list[TrainingPool]:
        p = self.protocol
        if self.mode == "development":
            frac = float(self.recipe.raw.get("development", {}).get("fraction", 0.1))
            block_bp = int(self.recipe.raw.get("development", {}).get("block_bp", 5_000_000))
            train_s, val_s = stratified_sample_split(
                canonical_root=self.root, sample_ids=p.array_train_sample_idx, val_fraction=frac, seed=self.seed,
            )
            train_c, val_c = blocked_cpg_split(
                registry=self.registry, cpg_ids=p.array_train_cpg_idx, val_fraction=frac, seed=self.seed, block_bp=block_bp,
            )
            self.inner_views = {
                "train_cpg_x_val_sample": (val_s, train_c),
                "val_cpg_x_train_sample": (train_s, val_c),
                "val_cpg_x_val_sample": (val_s, val_c),
            }
            array_rows = self._sources["array"].rows_of_samples(train_s)
            pools = [TrainingPool("array", array_rows, train_s, train_c)]
            forbidden = val_s
        else:
            array_rows = self._sources["array"].rows_of_samples(p.array_train_sample_idx)
            pools = [TrainingPool("array", array_rows, p.array_train_sample_idx, p.array_train_cpg_idx)]
            forbidden = p.array_val_sample_idx if self.recipe.exclude_official_val_from_auxiliary else np.empty(0, np.int64)

        # Mirrors ScopedRNATrainer._build_pools (rna_training/trainer.py) -- same
        # auxiliary-technology pool construction, duplicated rather than shared
        # per this module's isolation note.
        for name in ("epic", "wgbs"):
            if name not in p.sources:
                continue
            source = self._sources[name]
            rows = np.arange(source.n_rows, dtype=np.int64)
            if len(forbidden):
                rows = rows[~np.isin(source.sample_idx[rows], forbidden)]
            cpg = np.asarray(p.auxiliary_cpg_idx.get(name, []), np.int64)
            if self.mode == "development":
                cpg = cpg[np.isin(cpg, pools[0].cpg_idx)]
            if len(rows) and len(cpg):
                pools.append(TrainingPool(name, rows, source.sample_idx[rows], cpg))
        return pools

    def _autocast(self):
        if not self.recipe.training.amp:
            return nullcontext()
        requested = self.recipe.training.amp_dtype.lower()
        dtype = torch.bfloat16 if requested == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _schedules(self, epoch: int):
        schedules = []
        for i, pool in enumerate(self.pools):
            batch = self.recipe.batching[pool.name]
            schedules.append(SourceSchedule(
                len(pool.row_positions), len(pool.cpg_idx), int(batch["sample_size"]), int(batch["cpg_size"]),
                epoch, self.seed + 1009 * i, self.recipe.schedule_policy,
            ))
        return schedules, interleave(schedules, seed=self.seed, epoch=epoch)

    def _read_block(self, pool: TrainingPool, row_slots, cpg_slots):
        rows = pool.row_positions[row_slots]
        cpg = pool.cpg_idx[cpg_slots]
        source = self._sources[pool.name]
        beta = source.block(rows, cpg)
        return pool.sample_idx[row_slots], cpg, beta

    def _cpg_loss(self, cpg_ids: np.ndarray, mu_all: torch.Tensor, log_sigma_all: torch.Tensor) -> torch.Tensor:
        """Ensemble-mean-per-head supervision of mu/sigma against the same targets
        CpGStatisticsTrainer._loss uses -- keeps mu/sigma anchored to locus
        statistics instead of drifting to whatever helps the RNA branch."""
        present = self.cpg_target_index.contains(cpg_ids)
        if not present.any():
            return mu_all.sum() * 0.0
        rows = self.cpg_target_index.positions_of(cpg_ids[present])
        eps = 1e-4
        target_mu_logit = torch.logit(
            torch.from_numpy(self.cpg_target_mu[rows].astype(np.float32)).to(mu_all.device).clamp(eps, 1 - eps)
        )
        target_log_sigma = torch.log(
            torch.from_numpy(self.cpg_target_sigma[rows].astype(np.float32)).to(mu_all.device).clamp_min(1e-6)
        )
        present_t = torch.from_numpy(present).to(mu_all.device)
        mu_loss = F.mse_loss(mu_all[:, present_t], target_mu_logit.unsqueeze(0).expand(mu_all.shape[0], -1))
        sigma_loss = F.mse_loss(
            log_sigma_all[:, present_t], target_log_sigma.unsqueeze(0).expand(log_sigma_all.shape[0], -1)
        )
        return self.cpg_mu_weight * mu_loss + self.cpg_sigma_weight * sigma_loss

    def _step(self, pool: TrainingPool, sample_ids, cpg_ids, beta_np):
        finite = np.isfinite(beta_np)
        if not finite.any():
            return None
        rna_x = torch.from_numpy(self.rna.rows(sample_ids, dtype=np.float16)).to(self.device).float()
        emb_np, _, _ = self.features.get(cpg_ids, embedding_dtype=np.float16)
        emb = torch.from_numpy(emb_np).to(self.device).float()
        beta = torch.from_numpy(beta_np).to(self.device)
        with self._autocast():
            with torch.no_grad() if self.freeze_cpg_model else nullcontext():
                mu_all, log_sigma_all = self.cpg_model.forward_raw_all(emb)
            mu_logit = mu_all.mean(dim=0)
            log_sigma = log_sigma_all.mean(dim=0)
            prior = torch.sigmoid(mu_logit)
            sigma = torch.exp(log_sigma).clamp_min(self.cpg_model_cfg.sigma_floor)
            out = self.model(rna_x, emb, prior, sigma=sigma)
            loss_cfg = loss_config_for_source(self.recipe.loss, pool.name, self.recipe.structured_loss_sources)
            rna_loss, pieces = residual_loss(out, beta, prior, loss_cfg, epsilon=1e-4, sigma=sigma)
            rna_loss = rna_loss * (float(finite.sum()) / max(float(finite.size), 1.0))
            if self.freeze_cpg_model:
                cpg_loss = mu_all.sum() * 0.0
            else:
                cpg_loss = self._cpg_loss(cpg_ids, mu_all, log_sigma_all)
            total = rna_loss + self.joint_loss_weight * cpg_loss
        pieces = {**pieces, "cpg_loss": float(cpg_loss.detach().cpu()), "total_loss": float(total.detach().cpu())}
        return total, pieces

    @torch.no_grad()
    def evaluate_view(self, sample_ids, cpg_ids, *, sample_chunk=128, cpg_chunk=2048):
        self.model.eval()
        self.cpg_model.eval()
        source = self._sources["array"]
        rows = source.rows_of_samples(sample_ids)
        metrics = ArrayMomentMetrics(len(sample_ids), len(cpg_ids))
        for s0 in range(0, len(sample_ids), sample_chunk):
            s1 = min(s0 + sample_chunk, len(sample_ids))
            local_s = sample_ids[s0:s1]
            rna_x = torch.from_numpy(self.rna.rows(local_s)).to(self.device)
            for c0 in range(0, len(cpg_ids), cpg_chunk):
                c1 = min(c0 + cpg_chunk, len(cpg_ids))
                local_c = cpg_ids[c0:c1]
                emb_np, _, _ = self.features.get(local_c)
                emb = torch.from_numpy(emb_np).to(self.device)
                with self._autocast():
                    cpg_out = self.cpg_model(emb)
                    pred = self.model(rna_x, emb, cpg_out["mu"], sigma=cpg_out["sigma"])["beta"]
                target = source.block(rows[s0:s1], local_c)
                metrics.add(s0, c0, target, pred.float().cpu().numpy(), cpg_out["mu"].float().cpu().numpy())
        self.model.train()
        if not self.freeze_cpg_model:
            self.cpg_model.train()
        return metrics.finalize()

    def evaluate_development(self):
        if self.inner_views is None:
            raise RuntimeError("not a development run")
        return {name: self.evaluate_view(s, c) for name, (s, c) in self.inner_views.items()}

    def _save_checkpoint(self, path, optimizer, scheduler, epoch, history):
        payload = {
            "schema_version": 1, "model": "rna_methylation_joint", "scope": self.scope, "mode": self.mode,
            "epoch": epoch, "epochs_planned": self.epochs, "architecture": "joint_variance_normalized_residual",
            "model_state": self.model.state_dict(), "cpg_model_state": self.cpg_model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "model_config": asdict(self.recipe.model), "cpg_model_config": asdict(self.cpg_model_cfg),
            "loss_config": asdict(self.recipe.loss), "training": asdict(self.recipe.training), "history": history,
        }
        import os
        tmp = Path(str(path) + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def run(self) -> dict[str, object]:
        cfg = self.recipe.training
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": cfg.learning_rate},
                {"params": self.cpg_model.parameters(), "lr": self.cpg_learning_rate},
            ],
            weight_decay=cfg.weight_decay,
        )
        schedules0, plan0 = self._schedules(1)
        steps_per_epoch = len(plan0)
        horizon = int(cfg.scheduler_horizon_epochs or self.epochs)
        scheduler = build_lr_scheduler(
            optimizer, name=cfg.scheduler, total_steps=max(1, horizon * steps_per_epoch),
            warmup_steps=int(round(cfg.warmup_epochs * steps_per_epoch)), min_lr_ratio=cfg.min_lr_ratio,
        )
        history = []
        latest = self.store.checkpoint("last.pt")
        best = self.store.checkpoint("best.pt")
        best_score = -np.inf
        best_epoch = 0
        started_all = time.time()
        for epoch in range(1, self.epochs + 1):
            started = time.time()
            self.model.train()
            if not self.freeze_cpg_model:
                self.cpg_model.train()
            schedules, plan = self._schedules(epoch)
            losses = []
            optimizer_steps = 0
            for source_i, local_step in plan:
                pool = self.pools[source_i]
                row_slots, cpg_slots = schedules[source_i][local_step]
                sample_ids, cpg_ids, beta_np = self._read_block(pool, row_slots, cpg_slots)
                result = self._step(pool, sample_ids, cpg_ids, beta_np)
                if result is None:
                    continue
                loss, pieces = result
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self.model.parameters(), *self.cpg_model.parameters()], cfg.gradient_clip_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
            row = {
                "epoch": epoch, "seconds": time.time() - started, "optimizer_steps": optimizer_steps,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "lr": optimizer.param_groups[0]["lr"], "cpg_lr": optimizer.param_groups[1]["lr"],
            }
            if self.mode == "development":
                dev = self.evaluate_development()
                row["development"] = dev
                score = float(dev["val_cpg_x_val_sample"]["mas_pcc"])
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_epoch = epoch
                    self._save_checkpoint(best, optimizer, scheduler, epoch, [*history, row])
            history.append(row)
            self._save_checkpoint(latest, optimizer, scheduler, epoch, history)
            write_json(self.store.training_file("history.json"), history)
            print(f"[rna-joint:{self.scope}:{self.mode}:{epoch}/{self.epochs}] loss={row['loss']:.6g}", flush=True)
        if self.mode == "final":
            self._save_checkpoint(best, optimizer, scheduler, self.epochs, history)
            best_epoch = self.epochs
        summary = {
            "scope": self.scope, "mode": self.mode, "best_epoch": best_epoch,
            "best_inner_double_ood_mas_pcc": None if self.mode == "final" else best_score,
            "epochs": self.epochs, "elapsed_seconds": time.time() - started_all, "run_dir": str(self.store.path),
            "cpg_warm_start_checkpoint": self.cpg_warm_start_checkpoint,
        }
        write_json(self.store.training_file("summary.json"), summary)
        return summary
