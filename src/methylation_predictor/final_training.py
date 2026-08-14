"""One-stage paper training for the selected RNA256 residual model.

This module intentionally contains no development/refit cycle.  Architecture and
training budget are frozen before launch; all official training data are used from
epoch 1 and the held-out Array views are opened only after the final epoch.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time

import pyarrow  # noqa: F401 -- must precede torch/h5py on the target server
import h5py
import numpy as np
import torch

from .config import load_config
from .full_suite.cache import RNACache
from .full_suite.feature_store import SortedIndex
from .losses import residual_loss
from .models import RNA2DNAmModel, VarianceNormalizedResidualModel
from .table5_protocol import (
    ARRAY_VIEW_EXPECTED_OBSERVED,
    SOURCE_EXPECTED_OBSERVED,
    TABLE5_EXPECTED,
    TABLE5_PROTOCOL_NAME,
    TABLE5_PUBLISHED_METHYLPROPHET,
    Table5Protocol,
    published_delta,
)
from .tcga_canonical import TCGACanonicalBundle


FINAL_ARCHITECTURE = "rna256_residual_prior_product_no_gate_no_anchor"


@dataclass
class CartesianSourceSchedule:
    """All row-block x CpG-block combinations for one source, exactly once.

    The block grid is a lossless partition of the source pair matrix.  Shuffling
    changes only optimizer order: every CpG-sample pair slot is still visited
    once per epoch, matching the MethylProphet Table-5 training universe rather
    than merely touching each sample and CpG once.
    """

    n_rows: int
    n_cpgs: int
    row_block_size: int
    cpg_block_size: int
    epoch: int
    seed: int

    def __post_init__(self):
        if min(self.n_rows, self.n_cpgs, self.row_block_size, self.cpg_block_size) < 1:
            raise ValueError("Cartesian schedule dimensions and block sizes must be positive")
        self.row_blocks = [
            np.arange(start, min(start + self.row_block_size, self.n_rows), dtype=np.int64)
            for start in range(0, self.n_rows, self.row_block_size)
        ]
        self.cpg_blocks = [
            np.arange(start, min(start + self.cpg_block_size, self.n_cpgs), dtype=np.int64)
            for start in range(0, self.n_cpgs, self.cpg_block_size)
        ]
        grid = [(r, c) for r in range(len(self.row_blocks)) for c in range(len(self.cpg_blocks))]
        rng = np.random.default_rng([self.seed, self.epoch, 17])
        order = rng.permutation(len(grid))
        self.grid = [grid[int(i)] for i in order]

    def __len__(self) -> int:
        return len(self.grid)

    def __getitem__(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        r, c = self.grid[step]
        return self.row_blocks[r], self.cpg_blocks[c]

    def coverage_report(self) -> dict[str, int | float]:
        return {
            "row_blocks": len(self.row_blocks),
            "cpg_blocks": len(self.cpg_blocks),
            "steps": len(self),
            "pair_slots": int(self.n_rows * self.n_cpgs),
            "pair_slot_coverage_fraction": 1.0,
        }


def interleave_cartesian_schedules(schedules: list[CartesianSourceSchedule], seed: int, epoch: int) -> list[tuple[int, int]]:
    plan = [
        (source_index, step)
        for source_index, schedule in enumerate(schedules)
        for step in range(len(schedule))
    ]
    rng = np.random.default_rng([seed, epoch, 991])
    order = rng.permutation(len(plan))
    return [plan[int(i)] for i in order]


def pair_weight_scale(observed_in_batch: int, expected_total: int, steps_per_epoch: int) -> float:
    """Scale a mean batch loss so each finite CpG-sample pair has equal epoch weight."""
    if observed_in_batch < 0 or expected_total < 1 or steps_per_epoch < 1:
        raise ValueError("invalid pair-weight inputs")
    mean_observed_per_step = expected_total / steps_per_epoch
    return float(observed_in_batch / mean_observed_per_step)


def resolve_final_epochs(
    confirm_best_epoch: int,
    confirm_steps_per_epoch: int,
    final_steps_per_epoch: int,
) -> int:
    """Preserve the architecture-confirm optimizer-update budget after mixing data."""
    if confirm_best_epoch < 1 or confirm_steps_per_epoch < 1 or final_steps_per_epoch < 1:
        raise ValueError("epoch and step counts must be positive")
    return max(
        1,
        int(round(confirm_best_epoch * confirm_steps_per_epoch / final_steps_per_epoch)),
    )


class FinalFeatureCache:
    """Compact immutable NTv3 embedding + prior cache for the Table-5 universe."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.ids = np.load(root / "cpg_idx.npy", mmap_mode="r")
        self.embeddings = np.load(root / "embeddings.f16.npy", mmap_mode="r")
        self.prior = np.load(root / "prior.npy", mmap_mode="r")
        sigma_path = root / "sigma.npy"
        # sigma.npy only exists once scripts/prepare_final_tcga_mix_chr1.py has
        # built the V1 (variance-normalized-residual) prior cache; fall back to
        # all-ones so callers that never read the third element of .get() (any
        # model that doesn't take sigma) are unaffected.
        self.sigma = np.load(sigma_path, mmap_mode="r") if sigma_path.is_file() else np.ones_like(self.prior)
        if self.embeddings.shape != (len(self.ids), 1536):
            raise RuntimeError(f"unexpected final embedding cache shape: {self.embeddings.shape}")
        if self.prior.shape != (len(self.ids),):
            raise RuntimeError("final prior cache does not align with cpg_idx")
        if self.sigma.shape != (len(self.ids),):
            raise RuntimeError("final sigma cache does not align with cpg_idx")
        self.index = SortedIndex(self.ids, "final Table-5 feature cache")

    def get(
        self, cpg_idx: np.ndarray, embedding_dtype: np.dtype = np.float32
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # embedding_dtype defaults to float32 (backward compatible). The
        # stored cache is already float16; hot training loops that upload
        # straight to GPU can pass embedding_dtype=np.float16 to skip the
        # CPU-side upcast and do it on-GPU instead (near-free there).
        rows = self.index.positions_of(np.asarray(cpg_idx, dtype=np.int64))
        return (
            np.asarray(self.embeddings[rows], dtype=embedding_dtype),
            np.asarray(self.prior[rows], dtype=np.float32),
            np.asarray(self.sigma[rows], dtype=np.float32),
        )


class ExactCompactSource:
    """Table-5 compact source arranged in protocol order for direct 2-D slices."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.h5 = h5py.File(self.path, "r")
        if self.h5.attrs.get("protocol", "") != TABLE5_PROTOCOL_NAME:
            raise RuntimeError(f"compact source is not Table-5 exact: {self.path}")
        self.beta = self.h5["beta"]
        self.sample_idx = np.asarray(self.h5["sample_idx"][...], np.int64)
        self.cpg_idx = np.asarray(self.h5["cpg_idx"][...], np.int64)
        self.sample_index = SortedIndex(self.sample_idx, f"{self.path.name} samples")
        self.cpg_index = SortedIndex(self.cpg_idx, f"{self.path.name} CpGs")

    def rows_of_samples(self, sample_ids: np.ndarray) -> np.ndarray:
        return self.sample_index.positions_of(np.asarray(sample_ids, np.int64))

    def block(self, row_positions: np.ndarray, cpg_ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_positions, np.int64)
        cols = self.cpg_index.positions_of(np.asarray(cpg_ids, np.int64))
        row_contiguous = len(rows) == 0 or (np.all(np.diff(rows) == 1))
        col_contiguous = len(cols) == 0 or (np.all(np.diff(cols) == 1))
        if len(rows) and len(cols) and row_contiguous and col_contiguous:
            return np.asarray(self.beta[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], np.float32)
        # Fallback is used only for unusual external calls; paper training/eval
        # uses the contiguous protocol layout prepared by prepare_final_tcga_mix_chr1.py.
        block = np.empty((len(rows), len(cols)), np.float32)
        col_order = np.argsort(cols, kind="mergesort")
        sorted_cols = cols[col_order]
        col_inverse = np.empty_like(col_order)
        col_inverse[col_order] = np.arange(len(col_order))
        for i, row in enumerate(rows.tolist()):
            # h5py requires monotonically increasing fancy indices.  Restore
            # the caller's exact CpG order after the physical read.
            block[i] = np.asarray(self.beta[row, sorted_cols], np.float32)[col_inverse]
        return block

    def close(self) -> None:
        self.h5.close()


@dataclass
class TrainingPool:
    name: str
    row_positions: np.ndarray
    sample_idx: np.ndarray
    cpg_idx: np.ndarray


class ArrayMomentMetrics:
    """Exact streaming MSE/MAE/MAS-PCC/MAC-PCC moments for an Array view."""

    def __init__(self, n_samples: int, n_cpgs: int):
        self.n = 0
        self.sse = self.sae = self.prior_sse = 0.0
        self.sn = np.zeros(n_samples); self.st = np.zeros(n_samples); self.sp = np.zeros(n_samples)
        self.stt = np.zeros(n_samples); self.spp = np.zeros(n_samples); self.stp = np.zeros(n_samples)
        self.ssse = np.zeros(n_samples); self.spse = np.zeros(n_samples)
        self.cn = np.zeros(n_cpgs); self.ct = np.zeros(n_cpgs); self.cp = np.zeros(n_cpgs)
        self.ctt = np.zeros(n_cpgs); self.cpp = np.zeros(n_cpgs); self.ctp = np.zeros(n_cpgs)
        self.csse = np.zeros(n_cpgs); self.cpse = np.zeros(n_cpgs)

    def add(self, s0: int, c0: int, target: np.ndarray, pred: np.ndarray, prior: np.ndarray) -> None:
        valid = np.isfinite(target) & np.isfinite(pred)
        t = np.where(valid, target, 0.0).astype(np.float64)
        p = np.where(valid, pred, 0.0).astype(np.float64)
        pr = np.broadcast_to(np.asarray(prior, np.float64)[None, :], target.shape)
        e = np.where(valid, p - t, 0.0); pe = np.where(valid, pr - t, 0.0)
        ns, nc = target.shape; ss = slice(s0, s0 + ns); cs = slice(c0, c0 + nc)
        self.n += int(valid.sum()); self.sse += float((e * e).sum()); self.sae += float(np.abs(e).sum())
        self.prior_sse += float((pe * pe).sum())
        self.sn[ss] += valid.sum(1); self.st[ss] += t.sum(1); self.sp[ss] += p.sum(1)
        self.stt[ss] += (t*t).sum(1); self.spp[ss] += (p*p).sum(1); self.stp[ss] += (t*p).sum(1)
        self.ssse[ss] += (e*e).sum(1); self.spse[ss] += (pe*pe).sum(1)
        self.cn[cs] += valid.sum(0); self.ct[cs] += t.sum(0); self.cp[cs] += p.sum(0)
        self.ctt[cs] += (t*t).sum(0); self.cpp[cs] += (p*p).sum(0); self.ctp[cs] += (t*p).sum(0)
        self.csse[cs] += (e*e).sum(0); self.cpse[cs] += (pe*pe).sum(0)

    @staticmethod
    def _corr(n, sx, sy, sxx, syy, sxy):
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = sxy - sx * sy / n
            vx = sxx - sx * sx / n; vy = syy - sy * sy / n
            den = vx * vy
            return np.where((n >= 2) & (den > 0), cov / np.sqrt(np.maximum(den, 1e-30)), np.nan)

    def finalize(self) -> dict[str, float | int]:
        rs = self._corr(self.sn, self.st, self.sp, self.stt, self.spp, self.stp)
        rc = self._corr(self.cn, self.ct, self.cp, self.ctt, self.cpp, self.ctp)
        with np.errstate(divide="ignore", invalid="ignore"):
            sm = self.ssse / self.sn; sp = self.spse / self.sn
            cm = self.csse / self.cn; cp = self.cpse / self.cn
        mse = self.sse / self.n; pmse = self.prior_sse / self.n
        return {
            "rows": int(self.n), "mse": float(mse), "mae": float(self.sae / self.n),
            "prior_mse": float(pmse), "skill_vs_prior": float(1.0 - mse / pmse),
            "mas_pcc": float(np.nanmedian(rc)), "mac_pcc": float(np.nanmedian(rs)),
            "sample_win_fraction": float(np.nanmean(sm < sp)),
            "cpg_win_fraction": float(np.nanmean(cm < cp)),
        }

    def finalize_per_cpg(self) -> dict[str, np.ndarray]:
        """Per-CpG breakdown of the same streaming moments ``finalize()``
        reduces to medians -- used for stratified analyses (genomic context,
        inter-sample variability) rather than the headline scalar report."""
        rc = self._corr(self.cn, self.ct, self.cp, self.ctt, self.cpp, self.ctp)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = self.csse / self.cn; cp_ = self.cpse / self.cn
            true_mean = self.ct / self.cn
            true_var = self.ctt / self.cn - true_mean ** 2
        return {
            "n": self.cn, "pearson": rc, "mse": cm, "prior_mse": cp_,
            "skill_vs_prior": 1.0 - cm / cp_,
            "true_mean": true_mean, "true_var": true_var,
        }


class FinalTCGAMixTrainer:
    """Single-stage deterministic full-coverage trainer for the paper model."""

    def __init__(
        self,
        *,
        canonical_root: str | Path,
        config_path: str | Path,
        protocol_root: str | Path,
        feature_cache: str | Path,
        rna_cache: str | Path,
        array_cache: str | Path,
        epic_cache: str | Path,
        output_dir: str | Path,
        epochs: int,
        seed: int = 17,
        block_rows: dict[str, int] | None = None,
        block_cpgs: dict[str, int] | None = None,
    ):
        if epochs < 1:
            raise ValueError("epochs must be positive")
        self.root = Path(canonical_root)
        self.cfg = load_config(config_path)
        self.output = Path(output_dir); self.output.mkdir(parents=True, exist_ok=True)
        self.epochs = int(epochs); self.seed = int(seed)
        self.block_rows = block_rows or {"array": 128, "epic": 128, "wgbs": 32}
        self.block_cpgs = block_cpgs or {"array": 2048, "epic": 4096, "wgbs": 16384}
        self.bundle = TCGACanonicalBundle.from_root(self.root)
        self.protocol = Table5Protocol.load(protocol_root)
        if self.protocol.provenance.get("status") != "exact_table5_ready":
            raise RuntimeError(
                "Table-5 protocol has not passed the exact finite-pair preflight; "
                "run scripts/prepare_final_tcga_mix_chr1.py first"
            )
        self.features = FinalFeatureCache(feature_cache)
        self.rna = RNACache(rna_cache)
        self.compact = {
            "array": ExactCompactSource(array_cache),
            "epic": ExactCompactSource(epic_cache),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("final paper training requires CUDA")
        torch.manual_seed(self.seed); np.random.seed(self.seed); random.seed(self.seed)
        torch.set_float32_matmul_precision(self.cfg.training.matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = self.cfg.training.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.cfg.training.allow_tf32
        self.variance_normalized = bool(self.cfg.model.variance_normalized_residual)
        # Distinct label so a variance-normalized (V1) run can never silently
        # resume from -- or be resumed into -- a frozen-architecture checkpoint.
        self.architecture_label = (
            FINAL_ARCHITECTURE + "_v1_variance_normalized_residual"
            if self.variance_normalized else FINAL_ARCHITECTURE
        )
        if self.variance_normalized:
            self.model = VarianceNormalizedResidualModel(
                25_017, 1536, self.cfg.model, epsilon=self.cfg.data.clip_beta_epsilon
            ).to(self.device)
        else:
            self.model = RNA2DNAmModel(
                25_017, 1536, self.cfg.model, epsilon=self.cfg.data.clip_beta_epsilon
            ).to(self.device)
        self.pools = self._build_pools()
        self.tracker = None

    def close(self) -> None:
        if self.tracker is not None:
            try:
                self.tracker.finish(status="finished")
            except Exception:
                pass
        for cache in self.compact.values():
            cache.close()
        self.bundle.close()

    def _build_pools(self) -> list[TrainingPool]:
        array = self.compact["array"]
        epic = self.compact["epic"]
        # Exact cache layout: Array train rows/CpGs first, validation second.
        n_array_s = len(self.protocol.array_train_sample_idx)
        n_array_c = len(self.protocol.array_train_cpg_idx)
        if not np.array_equal(array.sample_idx[:n_array_s], self.protocol.array_train_sample_idx):
            raise RuntimeError("Array Table-5 compact cache train sample order mismatch")
        if not np.array_equal(array.cpg_idx[:n_array_c], self.protocol.array_train_cpg_idx):
            raise RuntimeError("Array Table-5 compact cache train CpG order mismatch")
        if not np.array_equal(epic.cpg_idx, self.protocol.epic_train_cpg_idx):
            raise RuntimeError("EPIC Table-5 compact cache CpG order mismatch")
        pools = [
            TrainingPool(
                "array", np.arange(n_array_s, dtype=np.int64),
                self.protocol.array_train_sample_idx, self.protocol.array_train_cpg_idx,
            ),
            TrainingPool(
                "epic", np.arange(len(epic.sample_idx), dtype=np.int64),
                epic.sample_idx, self.protocol.epic_train_cpg_idx,
            ),
        ]
        wgbs = self.bundle.sources["wgbs"]
        pools.append(TrainingPool(
            "wgbs", np.arange(wgbs.n_rows, dtype=np.int64),
            np.asarray(wgbs.sample_idx, np.int64), self.protocol.wgbs_train_cpg_idx,
        ))
        required = np.unique(np.concatenate([p.cpg_idx for p in pools]))
        self.features.index.positions_of(required)
        return pools

    def _autocast(self):
        if not self.cfg.training.amp:
            return nullcontext()
        requested = self.cfg.training.amp_dtype.lower()
        if requested == "bfloat16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif requested in {"bfloat16", "float16"}:
            dtype = torch.float16
        else:
            raise ValueError("training.amp_dtype must be bfloat16 or float16")
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _block(self, pool: TrainingPool, row_slots: np.ndarray, cpg_slots: np.ndarray) -> np.ndarray:
        rows = pool.row_positions[row_slots]
        cpg = pool.cpg_idx[cpg_slots]
        if pool.name in self.compact:
            return self.compact[pool.name].block(rows, cpg)
        return self.bundle.sources[pool.name].block(rows, cpg)

    def _prepare_step(self, schedules, source_index: int, local_step: int):
        """CPU-side data prep for one training step: HDF5 block read + RNA/
        embedding lookup.  Runs on a background thread so it overlaps with the
        GPU work of the *previous* step instead of blocking it -- this is the
        dominant cost (the GPU sits idle ~35-50% of the time in the fully
        synchronous version, per profiling; the actual per-step compute is
        small relative to Python/HDF5/numpy overhead for this model size).
        Embeddings/RNA are read at their native float16 storage dtype; the
        float32 upcast happens on-GPU after transfer (near-free there)
        instead of costing a CPU-side numpy pass on every step.
        """
        pool = self.pools[source_index]
        row_slots, cpg_slots = schedules[source_index][local_step]
        sample_ids = pool.sample_idx[row_slots]
        cpg_ids = pool.cpg_idx[cpg_slots]
        beta_np = self._block(pool, row_slots, cpg_slots)
        has_signal = bool(np.isfinite(beta_np).any())
        rna_np = emb_np = prior_np = sigma_np = None
        if has_signal:
            rna_np = self.rna.rows(sample_ids, dtype=np.float16)
            emb_np, prior_np, sigma_np = self.features.get(cpg_ids, embedding_dtype=np.float16)
        return pool, beta_np, has_signal, rna_np, emb_np, prior_np, sigma_np

    def _schedules(self, epoch: int):
        schedules = [
            CartesianSourceSchedule(
                n_rows=len(p.row_positions),
                n_cpgs=len(p.cpg_idx),
                row_block_size=self.block_rows[p.name],
                cpg_block_size=self.block_cpgs[p.name],
                epoch=epoch,
                seed=self.seed + 1009 * source_index,
            )
            for source_index, p in enumerate(self.pools)
        ]
        counts = {p.name: len(schedule) for p, schedule in zip(self.pools, schedules)}
        plan = interleave_cartesian_schedules(schedules, self.seed, epoch)
        return schedules, plan, counts

    def schedule_summary(self) -> dict[str, object]:
        schedules, plan, counts = self._schedules(1)
        return {
            "schedule": "complete Cartesian sample-block x CpG-block coverage",
            "steps_per_source": counts,
            "steps_per_epoch": len(plan),
            "expected_finite_pairs_per_source": SOURCE_EXPECTED_OBSERVED,
            "expected_finite_pairs_per_epoch": TABLE5_EXPECTED["total_train_observed"],
            "coverage": {p.name: schedule.coverage_report() for p, schedule in zip(self.pools, schedules)},
        }

    def _init_tracker(self):
        from .tracking import create_tracker
        resume_id = None
        run_id_path = self.output / "wandb_run_id.txt"
        if run_id_path.is_file():
            resume_id = run_id_path.read_text().strip() or None
        # Distinguish epoch-budget variants of the same configured run name in
        # the W&B UI (e.g. a 4-epoch smoke run vs. a 25-epoch full run) without
        # requiring a per-run config edit.
        if self.cfg.tracking.name:
            self.cfg.tracking.name = f"{self.cfg.tracking.name}-e{self.epochs}"
        self.cfg.tracking.tags = [*self.cfg.tracking.tags, f"epochs-{self.epochs}"]
        self.tracker = create_tracker(self.cfg, self.model, self.output, resume_id=resume_id)
        if getattr(self.tracker, "enabled", False) and hasattr(self.tracker, "run"):
            run_id_path.write_text(str(self.tracker.run.id) + "\n")

    def _save(self, path: Path, optimizer, epoch: int, history: list[dict], scaler) -> None:
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch),
            "epochs_planned": self.epochs,
            "architecture": self.architecture_label,
            "protocol": TABLE5_PROTOCOL_NAME,
            "seed": self.seed,
            "history": history,
            "rng_state": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            },
        }
        tmp = path.with_suffix(".tmp.pt"); torch.save(payload, tmp); os.replace(tmp, path)

    def train(self) -> Path:
        done = self.output / ".train_done"
        final_path = self.output / "final.pt"
        if done.is_file() and final_path.is_file():
            state = torch.load(final_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state"], strict=True)
            return final_path

        opt_kwargs = dict(lr=self.cfg.training.learning_rate, weight_decay=self.cfg.training.weight_decay)
        if self.cfg.training.fused_adamw:
            opt_kwargs["fused"] = True
        try:
            optimizer = torch.optim.AdamW(self.model.parameters(), **opt_kwargs)
        except (TypeError, RuntimeError):
            opt_kwargs.pop("fused", None); optimizer = torch.optim.AdamW(self.model.parameters(), **opt_kwargs)

        use_fp16_scaler = self.cfg.training.amp and (
            self.cfg.training.amp_dtype.lower() == "float16" or not torch.cuda.is_bf16_supported()
        )
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)
        start_epoch = 1; history: list[dict] = []
        latest = self.output / "latest.pt"
        if latest.is_file():
            state = torch.load(latest, map_location=self.device, weights_only=False)
            if state.get("architecture") != self.architecture_label:
                raise RuntimeError("latest.pt belongs to a different architecture")
            if int(state.get("epochs_planned", self.epochs)) != self.epochs:
                raise RuntimeError("cannot resume with a different fixed epoch budget")
            self.model.load_state_dict(state["model_state"], strict=True)
            optimizer.load_state_dict(state["optimizer_state"])
            if state.get("scaler_state") is not None:
                scaler.load_state_dict(state["scaler_state"])
            start_epoch = int(state["epoch"]) + 1; history = list(state.get("history", []))
            random.setstate(state["rng_state"]["python"]); np.random.set_state(state["rng_state"]["numpy"])
            torch.set_rng_state(state["rng_state"]["torch_cpu"].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in state["rng_state"]["torch_cuda"]])

        self._init_tracker()
        schedule_info = self.schedule_summary()
        manifest = {
            "architecture": self.architecture_label,
            "architecture_selection": {
                "confirm_mas_pcc": 0.5163, "confirm_mse": 0.02058,
                "confirm_skill_vs_prior": 0.237,
                "selection_protocol": "exact Array-chr1 architecture study, seed 17",
            },
            "training": "single_stage_fixed_budget_complete_table5_pair_coverage_no_heldout_selection",
            "protocol": TABLE5_PROTOCOL_NAME,
            "seed": self.seed, "epochs": self.epochs,
            "schedule": schedule_info,
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        global_step = sum(
            int(x.get("optimizer_steps", sum(x["source_steps"].values())))
            for x in history
        ) if history else 0
        # Single background worker that prepares step i+1's CPU-side data
        # (HDF5 block read + RNA/embedding lookup) while the main thread runs
        # the GPU forward/backward for step i.  See _prepare_step for why.
        prefetch = ThreadPoolExecutor(max_workers=1)
        try:
            for epoch in range(start_epoch, self.epochs + 1):
                self.model.train(); torch.cuda.reset_peak_memory_stats(self.device)
                started = time.time(); schedules, plan, counts = self._schedules(epoch)
                source_steps = {p.name: 0 for p in self.pools}
                source_raw_loss = {p.name: [] for p in self.pools}
                source_locus_pearson_loss = {p.name: [] for p in self.pools}
                source_observed = {p.name: 0 for p in self.pools}
                weighted_losses = []; grad_norms = []; optimizer_steps = 0

                pending = prefetch.submit(self._prepare_step, schedules, *plan[0])
                for step_idx in range(len(plan)):
                    pool, beta_np, has_signal, rna_np, emb_np, prior_np, sigma_np = pending.result()
                    if step_idx + 1 < len(plan):
                        pending = prefetch.submit(self._prepare_step, schedules, *plan[step_idx + 1])
                    # Source schedule coverage counts physical Cartesian blocks, not
                    # optimizer updates.  Count the block even if it contains no
                    # observed methylation values; the finite-pair audit below is
                    # the authoritative data-exposure invariant.
                    source_steps[pool.name] += 1
                    if not has_signal:
                        continue
                    rna = torch.from_numpy(rna_np).to(self.device).float()
                    emb = torch.from_numpy(emb_np).to(self.device).float()
                    prior = torch.from_numpy(prior_np).to(self.device)
                    beta = torch.from_numpy(beta_np).to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    with self._autocast():
                        if self.variance_normalized:
                            sigma = torch.from_numpy(sigma_np).to(self.device)
                            outputs = self.model(rna, emb, prior, sigma=sigma)
                        else:
                            sigma = None
                            outputs = self.model(rna, emb, prior)
                        raw_loss, pieces = residual_loss(
                            outputs, beta, prior, self.cfg.loss,
                            epsilon=self.cfg.data.clip_beta_epsilon,
                            sigma=sigma,
                        )
                        scale = pair_weight_scale(
                            int(pieces["observed"]),
                            TABLE5_EXPECTED["total_train_observed"],
                            len(plan),
                        )
                        loss = raw_loss * scale
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite final training loss: {pieces}")
                    if scaler.is_enabled():
                        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip_norm)
                        scaler.step(optimizer); scaler.update()
                    else:
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip_norm)
                        optimizer.step()

                    global_step += 1; optimizer_steps += 1
                    source_raw_loss[pool.name].append(float(raw_loss.detach().cpu()))
                    source_locus_pearson_loss[pool.name].append(float(pieces["locus_pearson_loss"]))
                    source_observed[pool.name] += int(pieces["observed"])
                    weighted_losses.append(float(loss.detach().cpu())); grad_norms.append(float(grad.detach().cpu()))
                    if getattr(self.tracker, "enabled", False) and global_step % self.cfg.tracking.log_every_steps == 0:
                        # Namespaced per-source so each source gets its own clean
                        # curve instead of one line jumping between array/epic/wgbs
                        # loss scales every few steps.  Pure bookkeeping values
                        # (pair_weight_scale, observed_pairs) stay in history.json
                        # / stdout rather than the live dashboard.
                        self.tracker.log({
                            "global_step": global_step,
                            "train/epoch": epoch,
                            f"train/{pool.name}/loss_raw": float(raw_loss.detach().cpu()),
                            f"train/{pool.name}/loss_weighted": float(loss.detach().cpu()),
                            f"train/{pool.name}/locus_pearson_loss": float(pieces["locus_pearson_loss"]),
                            f"train/{pool.name}/grad_norm": float(grad.detach().cpu()),
                            "train/lr": optimizer.param_groups[0]["lr"],
                        }, step=global_step)

                # Fail closed: every source-local full-coverage schedule must be consumed.
                if source_steps != counts:
                    raise RuntimeError(f"incomplete Cartesian Table-5 epoch: actual={source_steps}, expected={counts}")
                if source_observed != SOURCE_EXPECTED_OBSERVED:
                    raise RuntimeError(
                        f"Table-5 pair coverage mismatch: actual={source_observed}, expected={SOURCE_EXPECTED_OBSERVED}"
                    )
                if sum(source_observed.values()) != TABLE5_EXPECTED["total_train_observed"]:
                    raise RuntimeError("Table-5 total training pair coverage mismatch")
                row = {
                    "epoch": epoch, "seconds": time.time() - started,
                    "steps": len(plan), "optimizer_steps": optimizer_steps, "source_steps": source_steps,
                    "source_loss_raw": {k: float(np.mean(v)) for k, v in source_raw_loss.items()},
                    "source_locus_pearson_loss": {k: float(np.mean(v)) for k, v in source_locus_pearson_loss.items()},
                    "source_observed": source_observed,
                    "loss_weighted": float(np.mean(weighted_losses)),
                    "grad_norm_mean": float(np.mean(grad_norms)),
                    "gpu_max_memory_gb": float(torch.cuda.max_memory_allocated(self.device) / 2**30),
                }
                history.append(row)
                (self.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                self._save(latest, optimizer, epoch, history, scaler)
                print(f"[final:{epoch}/{self.epochs}] {row}", flush=True)
                if getattr(self.tracker, "enabled", False):
                    flat = {
                        "epoch": epoch,
                        "epoch/loss_weighted": row["loss_weighted"],
                        "epoch/seconds": row["seconds"],
                        "epoch/gpu_max_memory_gb": row["gpu_max_memory_gb"],
                        "epoch/grad_norm_mean": row["grad_norm_mean"],
                    }
                    for name in source_steps:
                        flat[f"epoch/{name}/loss_raw"] = row["source_loss_raw"][name]
                        flat[f"epoch/{name}/locus_pearson_loss"] = row["source_locus_pearson_loss"][name]
                        flat[f"epoch/{name}/steps"] = source_steps[name]
                    self.tracker.log(flat, step=global_step)
        finally:
            prefetch.shutdown(wait=False)

        self._save(final_path, optimizer, self.epochs, history, scaler)
        done.write_text("ok\n")
        return final_path

    @torch.no_grad()
    def _evaluate_array(self, sample_ids: np.ndarray, cpg_ids: np.ndarray) -> dict[str, float | int]:
        self.model.eval(); compact = self.compact["array"]
        rows = compact.rows_of_samples(sample_ids)
        metrics = ArrayMomentMetrics(len(sample_ids), len(cpg_ids))
        sc, cc = 128, 2048; started = time.time()
        for s0 in range(0, len(sample_ids), sc):
            s1 = min(s0 + sc, len(sample_ids)); local_ids = sample_ids[s0:s1]; local_rows = rows[s0:s1]
            rna = torch.from_numpy(self.rna.rows(local_ids)).to(self.device)
            for c0 in range(0, len(cpg_ids), cc):
                c1 = min(c0 + cc, len(cpg_ids)); local_c = cpg_ids[c0:c1]
                emb_np, prior_np, sigma_np = self.features.get(local_c)
                emb = torch.from_numpy(emb_np).to(self.device); prior = torch.from_numpy(prior_np).to(self.device)
                with self._autocast():
                    if self.variance_normalized:
                        sigma = torch.from_numpy(sigma_np).to(self.device)
                        pred = self.model(rna, emb, prior, sigma=sigma)["beta"]
                    else:
                        pred = self.model(rna, emb, prior)["beta"]
                target = compact.block(local_rows, local_c)
                metrics.add(s0, c0, target, pred.float().cpu().numpy(), prior_np)
        result = metrics.finalize(); result["samples"] = int(len(sample_ids)); result["cpgs"] = int(len(cpg_ids))
        result["seconds"] = time.time() - started
        return result

    def evaluate(self) -> dict[str, object]:
        views = self.protocol.evaluation_views()
        ours = {name: self._evaluate_array(s, c) for name, (s, c) in views.items()}
        for name, metrics in ours.items():
            expected_rows = ARRAY_VIEW_EXPECTED_OBSERVED[name]
            if int(metrics["rows"]) != expected_rows:
                raise RuntimeError(
                    f"Table-5 evaluation {name} has {metrics['rows']:,} finite pairs, expected {expected_rows:,}"
                )
        result = {
            "architecture": self.architecture_label,
            "protocol": TABLE5_PROTOCOL_NAME,
            "ours": ours,
            "methylprophet_table5_published": TABLE5_PUBLISHED_METHYLPROPHET,
            "delta_ours_minus_methylprophet_published": {
                name: published_delta(metrics, name) for name, metrics in ours.items()
            },
        }
        out = self.output / "evaluation"; out.mkdir(exist_ok=True)
        (out / "headline.json").write_text(json.dumps(result, indent=2) + "\n")
        if getattr(self.tracker, "enabled", False):
            summary = {}
            for view, metrics in ours.items():
                for key in ("mse", "mae", "mas_pcc", "mac_pcc", "skill_vs_prior"):
                    summary[f"final/{view}/{key}"] = metrics[key]
                delta = result["delta_ours_minus_methylprophet_published"][view]
                for key, value in delta.items():
                    summary[f"table5_delta/{view}/{key}"] = value
            self.tracker.set_summary(summary)
        return result

    def run(self) -> dict[str, object]:
        self.train()
        result = self.evaluate()
        (self.output / ".done").write_text("ok\n")
        return result
