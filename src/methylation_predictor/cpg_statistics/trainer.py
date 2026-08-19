"""Development-safe training for CpGStatisticsPredictor."""
from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .model import CpGStatisticsModelConfig, CpGStatisticsPredictor
from ..optim import build_lr_scheduler
from ..run_store import RunStore, write_json
from ..storage import SortedIndex, read_h5_rows


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def statistics_metrics(mu_true, sigma_true, mu_pred, sigma_pred) -> dict[str, float]:
    eps = 1e-4
    mu_true = np.asarray(mu_true, np.float64); mu_pred = np.asarray(mu_pred, np.float64)
    sigma_true = np.asarray(sigma_true, np.float64); sigma_pred = np.asarray(sigma_pred, np.float64)
    sse = np.square(mu_pred - mu_true).sum()
    baseline = np.square(mu_true - mu_true.mean()).sum()
    return {
        "mu_beta_mse": float(np.mean(np.square(mu_pred - mu_true))),
        "mu_mae": float(np.mean(np.abs(mu_pred - mu_true))),
        "mu_pcc": _pearson(mu_true, mu_pred),
        "mu_r2": float(1.0 - sse / baseline) if baseline > 0 else float("nan"),
        "mu_logit_mse": float(np.mean(np.square(np.log(np.clip(mu_pred, eps, 1-eps)/(1-np.clip(mu_pred, eps, 1-eps))) - np.log(np.clip(mu_true, eps, 1-eps)/(1-np.clip(mu_true, eps, 1-eps)))))),
        "sigma_mse": float(np.mean(np.square(sigma_pred - sigma_true))),
        "sigma_mae": float(np.mean(np.abs(sigma_pred - sigma_true))),
        "sigma_pcc": _pearson(sigma_true, sigma_pred),
        "log_sigma_mse": float(np.mean(np.square(np.log(np.maximum(sigma_pred, 1e-6)) - np.log(np.maximum(sigma_true, 1e-6))))),
    }


def genomic_dev_mask(cpg_ids: np.ndarray, registry: str | Path, *, fraction: float, seed: int, block_bp: int) -> np.ndarray:
    """Deterministic whole-genomic-block dev split inside official train CpGs."""
    frame = pd.read_parquet(registry)
    chrom_col = next((c for c in ("chrom", "chr", "chromosome") if c in frame.columns), None)
    pos_col = next((c for c in ("pos", "position", "start") if c in frame.columns), None)
    if chrom_col is None or pos_col is None or "cpg_idx" not in frame.columns:
        raise ValueError("registry needs cpg_idx plus chromosome and position columns for blocked development split")
    indexed = frame.set_index("cpg_idx").loc[cpg_ids]
    chrom = indexed[chrom_col].astype(str).to_numpy()
    pos = indexed[pos_col].to_numpy(np.int64)
    blocks = np.asarray([f"{c}:{int(p)//block_bp}" for c, p in zip(chrom, pos)], dtype=object)
    unique = np.unique(blocks)
    # Stable, independent of Python's randomized hash seed.
    import hashlib
    score = np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}:{x}".encode()).digest()[:8], "big") / 2**64
        for x in unique
    ])
    dev_blocks = set(unique[score < fraction].tolist())
    mask = np.asarray([x in dev_blocks for x in blocks], dtype=bool)
    if not mask.any() or mask.all():
        raise RuntimeError("blocked development split is empty on one side; adjust fraction")
    return mask


class CpGStatisticsTrainer:
    def __init__(
        self,
        *,
        targets_dir: str | Path,
        embeddings_h5: str | Path,
        registry: str | Path,
        recipe_path: str | Path,
        scope: str,
        output_root: str | Path,
        run_id: str | None = None,
        nested_run_store: bool = True,
    ):
        raw = yaml.safe_load(Path(recipe_path).read_text()) or {}
        self.raw = raw
        model_raw = raw.get("model", {})
        if "ensemble_seeds" in model_raw:
            model_raw = {**model_raw, "ensemble_seeds": tuple(model_raw["ensemble_seeds"])}
        self.model_cfg = CpGStatisticsModelConfig(**model_raw)
        self.training = dict(raw.get("training", {}))
        self.scope = scope
        self.seed = int(self.training.get("seed", 17))
        self.epochs = int(self.training.get("epochs", 40))
        self.lr = float(self.training.get("learning_rate", 1e-3))
        self.weight_decay = float(self.training.get("weight_decay", 1e-4))
        self.batch_size = int(self.training.get("batch_cpgs", 4096))
        self.mu_weight = float(self.training.get("mu_weight", 1.0))
        self.sigma_weight = float(self.training.get("sigma_weight", 1.0))
        self.scheduler_name = str(self.training.get("scheduler", "constant"))
        self.warmup_epochs = float(self.training.get("warmup_epochs", 0.0))
        self.min_lr_ratio = float(self.training.get("min_lr_ratio", 0.1))
        self.selection_metric = str(self.training.get("selection_metric", "joint_loss"))
        self.amp = bool(self.training.get("amp", True))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)

        root = Path(targets_dir)
        self.ids = np.load(root / "cpg_idx.npy")
        self.mu = np.load(root / "target_mu.npy")
        self.sigma = np.load(root / "target_sigma.npy")
        train_mask = np.load(root / "official_train_mask.npy").astype(bool)
        self.test_mask = np.load(root / "official_val_mask.npy").astype(bool)
        if not (len(self.ids) == len(self.mu) == len(self.sigma) == len(train_mask) == len(self.test_mask)):
            raise ValueError("statistics target arrays are misaligned")
        if np.any(train_mask & self.test_mask):
            raise ValueError("official train/test CpG masks overlap")

        import h5py
        with h5py.File(embeddings_h5, "r") as h:
            emb_ids = np.asarray(h["cpg_idx"][...], np.int64)
            idx = SortedIndex(emb_ids, "NTv3 embedding atlas")
            rows = idx.positions_of(self.ids)
            self.embeddings = read_h5_rows(h["embeddings"], rows, dtype=np.float32)
        if self.embeddings.shape != (len(self.ids), self.model_cfg.input_dim):
            raise ValueError(f"unexpected embedding matrix {self.embeddings.shape}")

        train_positions = np.flatnonzero(train_mask)
        dev_local = genomic_dev_mask(
            self.ids[train_positions], registry,
            fraction=float(self.training.get("development_fraction", 0.1)),
            seed=self.seed,
            block_bp=int(self.training.get("development_block_bp", 5_000_000)),
        )
        self.dev_idx = train_positions[dev_local]
        self.train_idx = train_positions[~dev_local]
        self.test_idx = np.flatnonzero(self.test_mask)
        self.model = CpGStatisticsPredictor(self.model_cfg).to(self.device)
        self.store = RunStore.create(
            output_root, model="cpg_statistics", train_scope=scope, seed=self.seed,
            learning_rate=self.lr, scheduler=self.scheduler_name, epochs=self.epochs,
            run_id=run_id, nested=nested_run_store,
        )
        self.store.save_resolved_config(raw)
        self.wandb_run = None
        tracking = dict(raw.get("tracking", {}))
        if tracking.get("backend", "none") == "wandb" and tracking.get("mode", "online") != "disabled":
            import wandb
            kwargs = {
                "project": tracking.get("project", "MethylPredicotr-CpGMeanPredictor"),
                "entity": tracking.get("entity"),
                "group": tracking.get("group") or f"cpg-statistics-{scope}",
                "name": tracking.get("name") or self.store.run_id,
                "job_type": tracking.get("job_type", "train"),
                "mode": tracking.get("mode", "online"),
                "dir": str(self.store.path),
                "tags": [*tracking.get("tags", []), f"scope-{scope}", "cpg-statistics"],
            }
            self.wandb_run = wandb.init(**{k: v for k, v in kwargs.items() if v is not None})
        target_manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").is_file() else {}
        self.store.save_metadata({
            "dataset_contract": target_manifest,
            "training": self.training,
            "model_config": asdict(self.model_cfg),
            "split": {"train_cpgs": int(len(self.train_idx)), "dev_cpgs": int(len(self.dev_idx)), "official_test_cpgs": int(len(self.test_idx))},
            "wandb": None if self.wandb_run is None else {"project": self.wandb_run.project, "run_id": self.wandb_run.id},
        })

    def _loss(self, idx: np.ndarray) -> tuple[torch.Tensor, dict[str, float]]:
        x = torch.from_numpy(self.embeddings[idx]).to(self.device)
        mu = torch.from_numpy(self.mu[idx]).to(self.device)
        sigma = torch.from_numpy(self.sigma[idx]).to(self.device)
        eps = 1e-4
        target_mu_logit = torch.logit(mu.clamp(eps, 1-eps))
        target_log_sigma = torch.log(sigma.clamp_min(1e-6))
        mu_all, sigma_all = self.model.forward_raw_all(x)
        mu_loss = torch.mean((mu_all - target_mu_logit.unsqueeze(0)) ** 2)
        sigma_loss = torch.mean((sigma_all - target_log_sigma.unsqueeze(0)) ** 2)
        loss = self.mu_weight * mu_loss + self.sigma_weight * sigma_loss
        return loss, {"mu_logit_mse": float(mu_loss.detach()), "log_sigma_mse": float(sigma_loss.detach())}

    @torch.no_grad()
    def evaluate_indices(self, idx: np.ndarray) -> dict[str, float]:
        self.model.eval()
        preds_mu, preds_sigma = [], []
        for start in range(0, len(idx), self.batch_size):
            local = idx[start:start+self.batch_size]
            x = torch.from_numpy(self.embeddings[local]).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp and self.device.type == "cuda"):
                out = self.model(x)
            preds_mu.append(out["mu"].float().cpu().numpy())
            preds_sigma.append(out["sigma"].float().cpu().numpy())
        return statistics_metrics(self.mu[idx], self.sigma[idx], np.concatenate(preds_mu), np.concatenate(preds_sigma))

    def _selection_value(self, metrics: dict[str, float]) -> float:
        if self.selection_metric == "joint_loss":
            return self.mu_weight * metrics["mu_logit_mse"] + self.sigma_weight * metrics["log_sigma_mse"]
        if self.selection_metric not in metrics:
            raise ValueError(f"unknown selection metric {self.selection_metric!r}")
        return float(metrics[self.selection_metric])

    def run(self) -> dict[str, object]:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps_per_epoch = max(1, int(np.ceil(len(self.train_idx) / self.batch_size)))
        horizon_epochs = int(self.training.get("scheduler_horizon_epochs") or self.epochs)
        scheduler = build_lr_scheduler(
            optimizer, name=self.scheduler_name,
            total_steps=max(1, horizon_epochs * steps_per_epoch),
            warmup_steps=int(round(self.warmup_epochs * steps_per_epoch)),
            min_lr_ratio=self.min_lr_ratio,
        )
        best_value = float("inf"); best_epoch = 0; history = []
        best_path = self.store.checkpoint("best.pt")
        last_path = self.store.checkpoint("last.pt")
        rng = np.random.default_rng(self.seed)
        started_all = time.time()
        for epoch in range(1, self.epochs + 1):
            started = time.time(); self.model.train(); order = rng.permutation(self.train_idx)
            epoch_loss = []
            for start in range(0, len(order), self.batch_size):
                local = order[start:start+self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp and self.device.type == "cuda"):
                    loss, _ = self._loss(local)
                loss.backward(); optimizer.step(); scheduler.step(); epoch_loss.append(float(loss.detach()))
            dev = self.evaluate_indices(self.dev_idx)
            value = self._selection_value(dev)
            row = {"epoch": epoch, "train_loss": float(np.mean(epoch_loss)), "lr": optimizer.param_groups[0]["lr"], "seconds": time.time()-started, **{f"dev_{k}": v for k,v in dev.items()}}
            history.append(row)
            state = {
                "schema_version": 1, "model": "cpg_statistics", "scope": self.scope,
                "epoch": epoch, "epochs_planned": self.epochs,
                "model_state": self.model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "model_config": asdict(self.model_cfg),
                "training": self.training,
            }
            torch.save(state, last_path)
            if value < best_value:
                best_value = value; best_epoch = epoch; torch.save(state, best_path)
            if self.wandb_run is not None:
                self.wandb_run.log({"epoch": epoch, "train/loss": row["train_loss"], "train/lr": row["lr"], **{f"development/{k}": v for k, v in dev.items()}}, step=epoch)
            print(f"[cpg-statistics:{self.scope}:{epoch}/{self.epochs}] dev={value:.6g}", flush=True)
        pd.DataFrame(history).to_csv(self.store.training_file("history.csv"), index=False)
        state = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        test = self.evaluate_indices(self.test_idx)
        summary = {"best_epoch": best_epoch, "selection_metric": self.selection_metric, "best_development_value": best_value, "official_heldout": test, "elapsed_seconds": time.time()-started_all}
        write_json(self.store.training_file("summary.json"), summary)
        write_json(self.store.evaluation_dir(self.scope) / "metrics.json", {"scope": self.scope, **test})
        self.store.write_evaluation_manifest(eval_scope=self.scope, checkpoint=best_path, dataset_contract="official held-out CpGs; static labels use training-patient methylation only")
        if self.wandb_run is not None:
            self.wandb_run.summary.update({f"official_heldout/{k}": v for k, v in test.items()})
            self.wandb_run.summary.update({"best_epoch": best_epoch, "best_development_value": best_value})
            self.wandb_run.finish(); self.wandb_run = None
        return {"run_dir": str(self.store.path), **summary}
