from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow  # noqa: F401 -- load before h5py on target host
import h5py
import numpy as np
import pandas as pd
import torch

from methylation_predictor.config import load_config
from methylation_predictor.full_coverage_sampler import build_epoch_schedule
from methylation_predictor.losses import residual_loss
from methylation_predictor.models import RNA2DNAmModel

from .cache import CompactSource, RNACache
from .feature_store import CompositeFeatureStore, SortedIndex


@dataclass
class Pool:
    name: str
    row_positions: np.ndarray
    sample_idx: np.ndarray
    cpg_idx: np.ndarray
    weight: float


@dataclass
class SuiteProtocol:
    name: str
    array_train_sample_idx: np.ndarray
    array_val_sample_idx: np.ndarray
    array_train_cpg_idx: np.ndarray
    array_val_cpg_idx: np.ndarray
    auxiliary_cpg_idx: dict[str, np.ndarray]
    sources: tuple[str, ...]


def _nested_sample_split(ids: np.ndarray, labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit, dev = [], []
    for label in sorted(np.unique(labels).tolist()):
        group = np.asarray(ids[labels == label], dtype=np.int64)
        order = rng.permutation(len(group))
        n_dev = 0 if len(group) <= 1 else min(len(group) - 1, max(1, int(round(fraction * len(group)))))
        dev.append(group[order[:n_dev]])
        fit.append(group[order[n_dev:]])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(dev))


def _nested_cpg_split(ids: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.asarray(ids, dtype=np.int64)
    order = rng.permutation(len(ids))
    n_dev = min(len(ids) - 1, max(1, int(round(fraction * len(ids)))))
    return np.sort(ids[order[n_dev:]]), np.sort(ids[order[:n_dev]])


def _load_suite_protocol(
    canonical_root: str | Path,
    protocol_name: str,
    genomewide_cpg_split: str | Path | None = None,
) -> SuiteProtocol:
    from methylation_predictor.tcga_canonical import TCGACanonicalBundle, load_protocol

    root = Path(canonical_root)
    with TCGACanonicalBundle.from_root(root) as bundle:
        if protocol_name != "array_genomewide":
            p = load_protocol(protocol_name, bundle)
            return SuiteProtocol(
                name=protocol_name,
                array_train_sample_idx=np.asarray(p.array_train_sample_idx, np.int64),
                array_val_sample_idx=np.asarray(p.array_val_sample_idx, np.int64),
                array_train_cpg_idx=np.asarray(p.array_train_cpg_idx, np.int64),
                array_val_cpg_idx=np.asarray(p.array_val_cpg_idx, np.int64),
                auxiliary_cpg_idx={k: np.asarray(v, np.int64) for k, v in p.auxiliary_cpg_idx.items()},
                sources=tuple(p.sources),
            )
        if genomewide_cpg_split is None:
            raise ValueError("array_genomewide requires --genomewide-cpg-split")
        split = pd.read_parquet(genomewide_cpg_split, columns=["cpg_idx", "split"])
        train_c = split.loc[split.split.eq("train"), "cpg_idx"].to_numpy(np.int64)
        val_c = split.loc[~split.split.eq("train"), "cpg_idx"].to_numpy(np.int64)
        # The legacy manifest's validation+test union is the official held-out
        # MethylProphet CpG pool.  Pin the known genome-wide counts.
        if len(train_c) != 326_906 or len(val_c) != 81_493:
            raise ValueError(f"unexpected genome-wide Array split counts: train={len(train_c)}, heldout={len(val_c)}")
        arr = bundle.sources["array"]
        sample_train = arr.sample_idx[arr.sample_split == "train"]
        sample_val = arr.sample_idx[arr.sample_split == "val"]
        if len(sample_train) != 8260 or len(sample_val) != 918:
            raise ValueError("unexpected canonical Array sample split counts")
        return SuiteProtocol(
            name=protocol_name,
            array_train_sample_idx=np.asarray(sample_train, np.int64),
            array_val_sample_idx=np.asarray(sample_val, np.int64),
            array_train_cpg_idx=np.sort(train_c),
            array_val_cpg_idx=np.sort(val_c),
            auxiliary_cpg_idx={},
            sources=("array",),
        )


def _rna_labels(root: Path, ids: np.ndarray) -> np.ndarray:
    with h5py.File(root / "tcga_rna_official_full.h5", "r") as h:
        all_ids = np.asarray(h["sample_idx"][...], np.int64)
        values = h["tissue_name"][...] if "tissue_name" in h else np.full(len(all_ids), b"UNKNOWN")
    labels = np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)
    index = SortedIndex(all_ids, "RNA labels")
    return labels[index.positions_of(ids)]


def _weights(policy: str, source_names: tuple[str, ...], row_counts: dict[str, int]) -> dict[str, float]:
    if policy == "equal_source":
        return {x: 1.0 for x in source_names}
    if policy == "array_heavy":
        if set(source_names) == {"array", "epic", "wgbs"}:
            return {"array": 0.50, "epic": 0.25, "wgbs": 0.25}
        return {x: 1.0 for x in source_names}
    if policy == "proportional_to_measurements":
        return {x: float(row_counts[x]) for x in source_names}
    raise ValueError("source policy must be equal_source, array_heavy, or proportional_to_measurements")


def _read_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, np.int64)
    unique, inv = np.unique(rows, return_inverse=True)
    return np.asarray(dataset[unique, :], np.float32)[inv]


class ArrayMomentMetrics:
    def __init__(self, n_samples: int, n_cpgs: int):
        self.n_samples = n_samples
        self.n_cpgs = n_cpgs
        self.n = 0
        self.sse = 0.0
        self.sae = 0.0
        self.prior_sse = 0.0
        self.sn = np.zeros(n_samples, np.float64); self.st = np.zeros(n_samples); self.sp = np.zeros(n_samples)
        self.stt = np.zeros(n_samples); self.spp = np.zeros(n_samples); self.stp = np.zeros(n_samples)
        self.ssse = np.zeros(n_samples); self.spse = np.zeros(n_samples)
        self.cn = np.zeros(n_cpgs, np.float64); self.ct = np.zeros(n_cpgs); self.cp = np.zeros(n_cpgs)
        self.ctt = np.zeros(n_cpgs); self.cpp = np.zeros(n_cpgs); self.ctp = np.zeros(n_cpgs)
        self.csse = np.zeros(n_cpgs); self.cpse = np.zeros(n_cpgs)

    def add(self, s0: int, c0: int, target: np.ndarray, pred: np.ndarray, prior: np.ndarray) -> None:
        valid = np.isfinite(target) & np.isfinite(pred)
        t = np.where(valid, target, 0.0).astype(np.float64)
        p = np.where(valid, pred, 0.0).astype(np.float64)
        pr = np.broadcast_to(np.asarray(prior, np.float64)[None, :], target.shape)
        e = np.where(valid, p - t, 0.0); pe = np.where(valid, pr - t, 0.0)
        ns, nc = target.shape
        ss = slice(s0, s0 + ns); cs = slice(c0, c0 + nc)
        self.n += int(valid.sum()); self.sse += float((e * e).sum()); self.sae += float(np.abs(e).sum()); self.prior_sse += float((pe * pe).sum())
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
            "prior_mse": float(pmse), "skill_vs_prior": float(1 - mse / pmse),
            "mas_pcc": float(np.nanmedian(rc)), "mac_pcc": float(np.nanmedian(rs)),
            "sample_win_fraction": float(np.nanmean(sm < sp)), "cpg_win_fraction": float(np.nanmean(cm < cp)),
        }


class MixedSuiteRunner:
    def __init__(
        self,
        *,
        canonical_root: str | Path,
        protocol_name: str,
        base_config: str | Path,
        base_feature_cache: str | Path,
        expanded_embedding_dir: str | Path | None,
        expanded_feature_dir: str | Path | None,
        rna_cache_dir: str | Path,
        output_dir: str | Path,
        source_policy: str = "equal_source",
        holdout_policy: str = "mp_matched",
        genomewide_cpg_split: str | Path | None = None,
        array_cache: str | Path | None = None,
        epic_cache: str | Path | None = None,
        seed: int = 17,
        mixed_steps_per_epoch: int = 128,
        dev_fraction: float = 0.10,
        block_rows: dict[str, int] | None = None,
        block_cpgs: dict[str, int] | None = None,
    ):
        from methylation_predictor.tcga_canonical import TCGACanonicalBundle

        if holdout_policy not in {"mp_matched", "strict_global"}:
            raise ValueError("holdout_policy must be mp_matched or strict_global")
        self.root = Path(canonical_root)
        self.protocol = _load_suite_protocol(self.root, protocol_name, genomewide_cpg_split)
        self.cfg = load_config(base_config)
        self.features = CompositeFeatureStore.from_cache(
            base_feature_cache, expanded_embedding_dir, expanded_feature_dir
        )
        self.rna = RNACache(rna_cache_dir)
        self.output = Path(output_dir); self.output.mkdir(parents=True, exist_ok=True)
        self.source_policy = source_policy; self.holdout_policy = holdout_policy; self.seed = seed
        self.mixed_steps_per_epoch = mixed_steps_per_epoch; self.dev_fraction = dev_fraction
        self.block_rows = block_rows or {"array": 128, "epic": 128, "wgbs": 32}
        self.block_cpgs = block_cpgs or {"array": 2048, "epic": 4096, "wgbs": 16384}
        self.bundle = TCGACanonicalBundle.from_root(self.root)
        self.compact = {
            "array": CompactSource(array_cache) if array_cache and Path(array_cache).is_file() else None,
            "epic": CompactSource(epic_cache) if epic_cache and Path(epic_cache).is_file() else None,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("full E2-E4 suite refuses to train on CPU")
        torch.set_float32_matmul_precision(self.cfg.training.matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = self.cfg.training.allow_tf32
        self.rng = np.random.default_rng(seed)
        self.labels = _rna_labels(self.root, self.protocol.array_train_sample_idx)
        self.dev_fit_s, self.dev_s = _nested_sample_split(self.protocol.array_train_sample_idx, self.labels, dev_fraction, seed)
        self.dev_fit_c, self.dev_c = _nested_cpg_split(self.protocol.array_train_cpg_idx, dev_fraction, seed)
        required = np.unique(np.concatenate([
            self.protocol.array_train_cpg_idx, self.protocol.array_val_cpg_idx,
            *self.protocol.auxiliary_cpg_idx.values(),
        ]) if self.protocol.auxiliary_cpg_idx else np.concatenate([self.protocol.array_train_cpg_idx, self.protocol.array_val_cpg_idx]))
        missing = required[~self.features.has(required)]
        if len(missing):
            raise RuntimeError(f"feature expansion incomplete: {len(missing)} required CpGs missing; examples={missing[:10].tolist()}")

    def close(self):
        for c in self.compact.values():
            if c is not None: c.close()
        self.bundle.close()

    def _new_model(self) -> RNA2DNAmModel:
        torch.manual_seed(self.seed); np.random.seed(self.seed); random.seed(self.seed)
        return RNA2DNAmModel(25017, 1536, self.cfg.model, epsilon=self.cfg.data.clip_beta_epsilon).to(self.device)

    def _pool(self, name: str, stage: str) -> Pool:
        src = self.bundle.sources[name]
        if name == "array":
            samples = self.dev_fit_s if stage == "development" else self.protocol.array_train_sample_idx
            rows = src.rows_of_samples(samples)
            cpgs = self.dev_fit_c if stage == "development" else self.protocol.array_train_cpg_idx
        else:
            rows = np.arange(src.n_rows, dtype=np.int64)
            sample_ids = src.sample_idx
            cpgs = np.asarray(self.protocol.auxiliary_cpg_idx[name], np.int64)
            # Nested dev is always globally held out across all sources so best_epoch
            # selection cannot see the dev patients/CpGs via auxiliary measurements.
            if stage == "development":
                keep_rows = ~np.isin(sample_ids, self.dev_s)
                rows = rows[keep_rows]
                cpgs = cpgs[~np.isin(cpgs, self.dev_c)]
            if self.holdout_policy == "strict_global":
                keep_rows = ~np.isin(src.sample_idx[rows], self.protocol.array_val_sample_idx)
                rows = rows[keep_rows]
                cpgs = cpgs[~np.isin(cpgs, self.protocol.array_val_cpg_idx)]
            samples = src.sample_idx[rows]
        return Pool(name, np.asarray(rows, np.int64), np.asarray(samples, np.int64), np.asarray(cpgs, np.int64), 1.0)

    def _pools(self, stage: str) -> list[Pool]:
        pools = [self._pool(name, stage) for name in self.protocol.sources]
        row_counts = {p.name: len(p.row_positions) for p in pools}
        weights = _weights(self.source_policy, tuple(p.name for p in pools), row_counts)
        for p in pools: p.weight = weights[p.name]
        return pools

    def _block(self, pool: Pool, row_slots: np.ndarray, cpg_slots: np.ndarray) -> np.ndarray:
        rows = pool.row_positions[row_slots]; cpg = pool.cpg_idx[cpg_slots]
        cache = self.compact.get(pool.name)
        if cache is not None:
            return cache.block(rows, cpg)
        return self.bundle.sources[pool.name].block(rows, cpg)

    def _sample_block(self, pool: Pool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nr = min(self.block_rows[pool.name], len(pool.row_positions)); nc = min(self.block_cpgs[pool.name], len(pool.cpg_idx))
        row_slots = self.rng.choice(len(pool.row_positions), nr, replace=False)
        cpg_slots = self.rng.choice(len(pool.cpg_idx), nc, replace=False)
        beta = self._block(pool, row_slots, cpg_slots)
        return row_slots, cpg_slots, pool.sample_idx[row_slots], pool.cpg_idx[cpg_slots], beta

    def _tensor_forward(self, model, sample_ids: np.ndarray, cpg_ids: np.ndarray, beta_np: np.ndarray):
        rna = torch.from_numpy(self.rna.rows(sample_ids)).to(self.device)
        emb_np, prior_np, var_np = self.features.get(cpg_ids)
        emb = torch.from_numpy(emb_np).to(self.device); prior = torch.from_numpy(prior_np).to(self.device)
        var = torch.from_numpy(var_np).to(self.device); beta = torch.from_numpy(beta_np).to(self.device)
        ref = torch.zeros((1, 25017), dtype=rna.dtype, device=self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.cfg.training.amp):
            out = model(rna, emb, prior, var, reference_rna=ref)
            loss, pieces = residual_loss(out, beta, prior, self.cfg.loss, epsilon=self.cfg.data.clip_beta_epsilon)
        return loss, pieces

    def _evaluate_array(self, model, sample_ids: np.ndarray, cpg_ids: np.ndarray, sample_chunk: int = 128, cpg_chunk: int = 2048) -> dict:
        model.eval(); arr = self.bundle.sources["array"]
        rows = arr.rows_of_samples(sample_ids); source_cols = arr._cpg_index.positions_of(cpg_ids)
        compact = self.compact.get("array")
        if compact is not None:
            compact_cols = compact.cpg_index.positions_of(cpg_ids)
        metrics = ArrayMomentMetrics(len(sample_ids), len(cpg_ids)); started = time.time()
        with torch.inference_mode():
            for s0 in range(0, len(sample_ids), sample_chunk):
                s1 = min(s0 + sample_chunk, len(sample_ids)); local_ids = sample_ids[s0:s1]
                local_rows = rows[s0:s1]
                # Critical I/O optimization: one full row read per sample chunk,
                # then all CpG chunks are sliced from host RAM.
                if compact is not None:
                    full_target = _read_rows(compact.beta, local_rows)
                    cols_for_view = compact_cols
                else:
                    full_target = _read_rows(arr.h5["beta"], local_rows)
                    cols_for_view = source_cols
                rna = torch.from_numpy(self.rna.rows(local_ids)).to(self.device)
                ref = torch.zeros((1, 25017), dtype=rna.dtype, device=self.device)
                for c0 in range(0, len(cpg_ids), cpg_chunk):
                    c1 = min(c0 + cpg_chunk, len(cpg_ids)); local_c = cpg_ids[c0:c1]
                    emb_np, prior_np, var_np = self.features.get(local_c)
                    emb = torch.from_numpy(emb_np).to(self.device); prior = torch.from_numpy(prior_np).to(self.device); var = torch.from_numpy(var_np).to(self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.cfg.training.amp):
                        pred = model(rna, emb, prior, var, reference_rna=ref)["beta"]
                    target = full_target[:, cols_for_view[c0:c1]]
                    metrics.add(s0, c0, target, pred.float().cpu().numpy(), prior_np)
        result = metrics.finalize(); result["samples"] = int(len(sample_ids)); result["cpgs"] = int(len(cpg_ids)); result["seconds"] = time.time() - started
        return result

    def _save_checkpoint(self, path: Path, model, epoch: int, validation: dict | None, optimizer=None) -> None:
        payload = {"model_state": model.state_dict(), "epoch": epoch, "validation_metrics": validation,
                   "protocol": self.protocol.name, "source_policy": self.source_policy, "holdout_policy": self.holdout_policy}
        if optimizer is not None: payload["optimizer_state"] = optimizer.state_dict()
        tmp = path.with_suffix(".tmp.pt"); torch.save(payload, tmp); os.replace(tmp, path)

    def _train_stage(self, stage: str, epochs: int, *, select_best: bool) -> tuple[RNA2DNAmModel, int]:
        stage_dir = self.output / stage; stage_dir.mkdir(parents=True, exist_ok=True)
        done = stage_dir / ".done"; best_path = stage_dir / "best.pt"; latest = stage_dir / "latest.pt"
        if done.is_file() and best_path.is_file():
            state = torch.load(best_path, map_location=self.device, weights_only=False)
            model = self._new_model(); model.load_state_dict(state["model_state"])
            return model, int(state["epoch"])

        model = self._new_model()
        opt_kwargs = dict(lr=self.cfg.training.learning_rate, weight_decay=self.cfg.training.weight_decay)
        if self.cfg.training.fused_adamw:
            opt_kwargs["fused"] = True
        try:
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        except (TypeError, RuntimeError):
            opt_kwargs.pop("fused", None)
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        pools = self._pools(stage); probs = np.asarray([p.weight for p in pools], np.float64); probs /= probs.sum()
        seen_cpg = {p.name: np.zeros(len(p.cpg_idx), dtype=bool) for p in pools}; seen_rows = {p.name: np.zeros(len(p.row_positions), dtype=bool) for p in pools}
        best_mse = float("inf"); best_epoch = 0; bad = 0; history = []
        start_epoch = 1
        if latest.is_file():
            state = torch.load(latest, map_location=self.device, weights_only=False)
            model.load_state_dict(state["model_state"]); opt.load_state_dict(state["optimizer_state"])
            start_epoch = int(state["epoch"]) + 1; best_mse = float(state.get("best_mse", best_mse)); best_epoch = int(state.get("best_epoch", 0)); bad = int(state.get("bad", 0))
            if "rng_state" in state:
                self.rng.bit_generator.state = state["rng_state"]["numpy_generator"]
                random.setstate(state["rng_state"]["python"])
                torch.set_rng_state(state["rng_state"]["torch_cpu"].to(dtype=torch.uint8, device="cpu"))
                if state["rng_state"].get("torch_cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([x.to(dtype=torch.uint8, device="cpu") for x in state["rng_state"]["torch_cuda"]])
            if (stage_dir / "history.json").is_file(): history = json.loads((stage_dir / "history.json").read_text())

        for epoch in range(start_epoch, epochs + 1):
            model.train(); started = time.time(); observed = 0; source_steps = {p.name: 0 for p in pools}; losses = []
            # Array-only E4 reuses the deterministic bidimensional full-coverage
            # schedule; mixed E2/E3 use source-first stochastic blocks.
            if len(pools) == 1 and pools[0].name == "array" and self.protocol.name == "array_genomewide":
                p = pools[0]
                schedule = build_epoch_schedule(p.cpg_idx, p.sample_idx, self.block_cpgs["array"], self.block_rows["array"], epoch, self.seed)
                steps = schedule.steps_per_epoch
                schedule_row_index = SortedIndex(p.sample_idx, "array train sample pool")
                schedule_cpg_index = SortedIndex(p.cpg_idx, "array train CpG pool")
            else:
                schedule = None; steps = self.mixed_steps_per_epoch
                schedule_row_index = schedule_cpg_index = None

            for step in range(steps):
                if schedule is None:
                    pi = int(self.rng.choice(len(pools), p=probs)); pool = pools[pi]
                    row_slots, cpg_slots, sample_ids, cpg_ids, beta = self._sample_block(pool)
                else:
                    pool = pools[0]; sample_ids, cpg_ids = schedule[step]
                    assert schedule_row_index is not None and schedule_cpg_index is not None
                    row_slots = schedule_row_index.positions_of(sample_ids); cpg_slots = schedule_cpg_index.positions_of(cpg_ids)
                    beta = self._block(pool, row_slots, cpg_slots)
                if not np.isfinite(beta).any(): continue
                opt.zero_grad(set_to_none=True)
                loss, pieces = self._tensor_forward(model, sample_ids, cpg_ids, beta)
                if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss: {pieces}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.gradient_clip_norm); opt.step()
                observed += int(pieces["observed"]); losses.append(float(pieces["loss"])); source_steps[pool.name] += 1
                seen_rows[pool.name][row_slots] = True; seen_cpg[pool.name][cpg_slots] = True

            validation = None
            if select_best:
                validation = self._evaluate_array(model, self.dev_s, self.dev_c)
                mse = float(validation["mse"])
                if mse < best_mse - self.cfg.training.min_delta:
                    best_mse = mse; best_epoch = epoch; bad = 0; self._save_checkpoint(best_path, model, epoch, validation)
                else:
                    bad += 1
            else:
                best_epoch = epoch; self._save_checkpoint(best_path, model, epoch, None)

            row = {"epoch": epoch, "seconds": time.time() - started, "observed": observed,
                   "train_loss": float(np.mean(losses)), "source_steps": source_steps,
                   "coverage_cpg": {k: float(v.mean()) for k, v in seen_cpg.items()},
                   "coverage_rows": {k: float(v.mean()) for k, v in seen_rows.items()},
                   "validation": validation}
            history.append(row); (stage_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            payload = {"model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "epoch": epoch,
                       "best_mse": best_mse, "best_epoch": best_epoch, "bad": bad,
                       "rng_state": {
                           "numpy_generator": self.rng.bit_generator.state,
                           "python": random.getstate(),
                           "torch_cpu": torch.get_rng_state(),
                           "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                       }}
            tmp = latest.with_suffix(".tmp.pt"); torch.save(payload, tmp); os.replace(tmp, latest)
            print(f"[{self.protocol.name}:{stage}] {row}", flush=True)
            if select_best and epoch >= self.cfg.training.min_epochs and bad >= self.cfg.training.patience:
                break

        if select_best:
            if not best_path.is_file(): raise RuntimeError("development stage produced no best checkpoint")
            state = torch.load(best_path, map_location=self.device, weights_only=False); model.load_state_dict(state["model_state"]); best_epoch = int(state["epoch"])
        done.write_text("ok\n")
        return model, best_epoch

    def run(self) -> dict[str, object]:
        manifest = {
            "protocol": self.protocol.name, "source_policy": self.source_policy, "holdout_policy": self.holdout_policy,
            "seed": self.seed, "sources": list(self.protocol.sources), "block_rows": self.block_rows, "block_cpgs": self.block_cpgs,
            "mixed_steps_per_epoch": self.mixed_steps_per_epoch,
            "split_counts": {"train_samples": len(self.protocol.array_train_sample_idx), "val_samples": len(self.protocol.array_val_sample_idx),
                             "train_cpgs": len(self.protocol.array_train_cpg_idx), "val_cpgs": len(self.protocol.array_val_cpg_idx),
                             "nested_fit_samples": len(self.dev_fit_s), "nested_dev_samples": len(self.dev_s),
                             "nested_fit_cpgs": len(self.dev_fit_c), "nested_dev_cpgs": len(self.dev_c)},
            "evaluation_semantics": (
                "Array-heldout evaluation; auxiliary EPIC/WGBS overlap preserved to match canonical MP training pools"
                if self.holdout_policy == "mp_matched" else
                "strict global OOD: official Array heldout sample/CpG ids excluded from every auxiliary source"
            ),
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        _, best_epoch = self._train_stage("development", self.cfg.training.epochs, select_best=True)
        final_model, _ = self._train_stage("final_refit", best_epoch, select_best=False)

        views = {
            "train_cpg_x_val_sample": (self.protocol.array_val_sample_idx, self.protocol.array_train_cpg_idx),
            "val_cpg_x_train_sample": (self.protocol.array_train_sample_idx, self.protocol.array_val_cpg_idx),
            "val_cpg_x_val_sample": (self.protocol.array_val_sample_idx, self.protocol.array_val_cpg_idx),
        }
        metrics = {name: self._evaluate_array(final_model, s, c) for name, (s, c) in views.items()}
        result = {"protocol": self.protocol.name, "best_epoch": best_epoch, "source_policy": self.source_policy,
                  "holdout_policy": self.holdout_policy, "views": metrics}
        eval_dir = self.output / "evaluation"; eval_dir.mkdir(exist_ok=True)
        (eval_dir / "headline.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
