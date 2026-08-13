#!/usr/bin/env python3
"""Rebuild a leakage-safe frozen genomic prior for the canonical TCGA Array universe.

The historical ``locus_features.parquet`` is no longer available.  This command
builds a *new, explicitly versioned* ``genomic_prior_v2`` from the immutable
canonical methylation bundle and the consolidated NTv3 atlas without rerunning
NTv3.

Protocol
--------
Targets are computed using official Array TRAIN samples only.  For CpG i:

    prior_i   = E[beta_i]
    within_i  = E_g[Var(beta_i | cancer_type=g)]
    between_i = Var_g(E[beta_i | cancer_type=g])

where every expectation is weighted by the number of finite observations at
that CpG.  Thus total population variance decomposes into within + between up
to floating-point error.

A 3-seed MLP ensemble (the repository's current ProbeEnsemble) maps frozen
1536-D NTv3 embeddings to [logit(prior), log(within/between variance)].
Official train CpGs receive five-fold OOF predictions; official held-out CpGs
receive predictions from a probe fitted only on all official train CpGs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

# pyarrow before h5py on the target host (libstdc++ ordering).
import pyarrow  # noqa: F401
import h5py
import numpy as np
import pandas as pd
import torch

from methylation_predictor.full_suite.probe import ProbeEnsemble

EXPECTED_ARRAY_CPGS = 408_399
EXPECTED_EMBED_DIM = 1536
EXPECTED_TRAIN_SAMPLES = 8_260
EXPECTED_TRAIN_CPGS = 326_906
EXPECTED_HELDOUT_CPGS = 81_493
TARGET_NAMES = ("prior_logit", "log_var_between", "log_var_within")
DEFAULT_SEEDS = (17, 29, 43)


def _decode(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind not in {"S", "O"}:
        return values.astype(str)
    return np.asarray([
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in values.tolist()
    ], dtype=object)


def _sha256_ids(values: np.ndarray) -> str:
    values = np.sort(np.asarray(values, dtype=np.int64))
    h = hashlib.sha256()
    h.update(values.tobytes(order="C"))
    return h.hexdigest()


def _positions(ids: np.ndarray, query: np.ndarray, name: str) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    order = np.argsort(ids, kind="mergesort")
    sorted_ids = ids[order]
    if len(sorted_ids) and np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError(f"{name} contains duplicate ids")
    p = np.searchsorted(sorted_ids, query)
    clipped = np.minimum(p, max(len(sorted_ids) - 1, 0))
    ok = (p < len(sorted_ids)) & (sorted_ids[clipped] == query)
    if not np.all(ok):
        missing = query[~ok]
        raise KeyError(f"{name} missing {len(missing)} ids; examples={missing[:10].tolist()}")
    return order[p]


def variance_components_from_group_moments(
    counts: np.ndarray,
    sums: np.ndarray,
    sumsq: np.ndarray,
    *,
    variance_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return global mean, between variance, within variance for each CpG.

    ``counts/sums/sumsq`` have shape [groups, CpGs] and contain finite-only
    population moments.  Group weights are CpG-specific observation counts.
    """
    counts = np.asarray(counts, dtype=np.float64)
    sums = np.asarray(sums, dtype=np.float64)
    sumsq = np.asarray(sumsq, dtype=np.float64)
    if counts.shape != sums.shape or counts.shape != sumsq.shape or counts.ndim != 2:
        raise ValueError("counts/sums/sumsq must share [groups, cpgs] shape")

    total_n = counts.sum(axis=0)
    total_sum = sums.sum(axis=0)
    total_sumsq = sumsq.sum(axis=0)
    if np.any(total_n <= 0):
        raise RuntimeError(f"{int((total_n <= 0).sum())} CpGs have no finite official-train observations")

    mean = total_sum / total_n
    safe_counts = np.maximum(counts, 1.0)
    group_mean = sums / safe_counts
    group_mean = np.where(counts > 0, group_mean, 0.0)

    within_ss = np.maximum(sumsq - sums * sums / safe_counts, 0.0)
    within_ss = np.where(counts > 0, within_ss, 0.0).sum(axis=0)
    between_ss = (counts * (group_mean - mean[None, :]) ** 2).sum(axis=0)

    within_var = np.maximum(within_ss / total_n, variance_floor)
    between_var = np.maximum(between_ss / total_n, variance_floor)
    total_var = np.maximum(total_sumsq / total_n - mean * mean, 0.0)
    reconstruction = within_ss / total_n + between_ss / total_n
    error = np.abs(total_var - reconstruction)
    report = {
        "decomposition_max_abs_error": float(np.max(error)),
        "decomposition_mean_abs_error": float(np.mean(error)),
        "minimum_observations_per_cpg": float(np.min(total_n)),
        "median_observations_per_cpg": float(np.median(total_n)),
        "maximum_observations_per_cpg": float(np.max(total_n)),
    }
    return mean.astype(np.float32), between_var.astype(np.float32), within_var.astype(np.float32), report


def _canonical_paths(root: Path) -> dict[str, Path]:
    return {
        "rna": root / "rna" / "tcga_rna_official_full.h5",
        "array": root / "methylation" / "tcga_array_official_full.h5",
        "protocol": root / "protocols" / "array_genomewide",
        "atlas": root / "cpg" / "ntv3" / "ntv3_cpg_atlas_v1.h5",
    }


def _load_protocol(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = _canonical_paths(root)["protocol"]
    train_samples = np.load(p / "array_train_sample_idx.npy").astype(np.int64)
    train_cpg = np.load(p / "array_train_cpg_idx.npy").astype(np.int64)
    heldout_cpg = np.load(p / "array_val_cpg_idx.npy").astype(np.int64)
    if len(train_samples) != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError(f"expected {EXPECTED_TRAIN_SAMPLES} train samples, got {len(train_samples)}")
    if len(train_cpg) != EXPECTED_TRAIN_CPGS or len(heldout_cpg) != EXPECTED_HELDOUT_CPGS:
        raise RuntimeError(
            f"unexpected Array genome-wide split: train={len(train_cpg)}, heldout={len(heldout_cpg)}"
        )
    if np.intersect1d(train_cpg, heldout_cpg).size:
        raise RuntimeError("Array genome-wide train/heldout CpG split overlaps")
    return train_samples, train_cpg, heldout_cpg


def _prepare_targets(root: Path, out: Path, row_chunk: int, variance_floor: float) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    done = out / "targets.done"
    required_outputs = [
        out / "array_cpg_idx.npy",
        out / "target_prior.npy",
        out / "target_log_var_between.npy",
        out / "target_log_var_within.npy",
        out / "fold_id.npy",
        out / "targets.json",
    ]
    if done.is_file() and all(p.is_file() for p in required_outputs):
        return json.loads((out / "targets.json").read_text())

    paths = _canonical_paths(root)
    train_samples, train_cpg, heldout_cpg = _load_protocol(root)

    # Resolve cancer type for each official train sample from canonical RNA metadata.
    with h5py.File(paths["rna"], "r") as h:
        rna_ids = np.asarray(h["sample_idx"][...], dtype=np.int64)
        if "tissue_name" not in h:
            raise RuntimeError("canonical RNA must contain tissue_name to define cancer-type variance components")
        labels = _decode(h["tissue_name"][...])
    rna_rows = _positions(rna_ids, train_samples, "RNA sample_idx")
    train_labels = labels[rna_rows].astype(str)
    cancer_names, cancer_codes = np.unique(train_labels, return_inverse=True)

    started = time.time()
    with h5py.File(paths["array"], "r") as h:
        beta = h["beta"]
        cpg_ids = np.asarray(h["cpg_idx"][...], dtype=np.int64)
        sample_ids = np.asarray(h["sample_idx"][...], dtype=np.int64)
        if beta.shape != (9_178, EXPECTED_ARRAY_CPGS):
            raise RuntimeError(f"unexpected canonical Array shape: {beta.shape}")
        if len(cpg_ids) != EXPECTED_ARRAY_CPGS:
            raise RuntimeError("canonical Array cpg_idx length mismatch")
        sample_rows = _positions(sample_ids, train_samples, "Array sample_idx")

        # Fancy HDF5 indexing is monotonic; keep labels aligned after sorting physical rows.
        order = np.argsort(sample_rows)
        rows = sample_rows[order]
        codes = cancer_codes[order]
        n_groups = len(cancer_names)
        counts = np.zeros((n_groups, EXPECTED_ARRAY_CPGS), dtype=np.uint16)
        sums = np.zeros((n_groups, EXPECTED_ARRAY_CPGS), dtype=np.float64)
        sumsq = np.zeros((n_groups, EXPECTED_ARRAY_CPGS), dtype=np.float64)

        for start in range(0, len(rows), row_chunk):
            stop = min(start + row_chunk, len(rows))
            block_rows = rows[start:stop]
            block_codes = codes[start:stop]
            block = np.asarray(beta[block_rows, :], dtype=np.float32)
            for g in np.unique(block_codes):
                x = block[block_codes == g]
                finite = np.isfinite(x)
                safe = np.where(finite, x, 0.0).astype(np.float64, copy=False)
                counts[g] += finite.sum(axis=0, dtype=np.uint16)
                sums[g] += safe.sum(axis=0, dtype=np.float64)
                sumsq[g] += np.square(safe, dtype=np.float64).sum(axis=0)
            if start == 0 or stop == len(rows) or (start // row_chunk) % 10 == 0:
                elapsed = time.time() - started
                frac = stop / len(rows)
                eta = elapsed * (1.0 / frac - 1.0) if frac else float("nan")
                print(
                    f"[targets] samples {stop:,}/{len(rows):,} ({100*frac:.1f}%) "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )

    prior, between, within, decomposition = variance_components_from_group_moments(
        counts, sums, sumsq, variance_floor=variance_floor
    )
    eps = 1e-4
    prior = np.clip(prior, eps, 1.0 - eps).astype(np.float32)
    log_between = np.log(np.maximum(between, variance_floor)).astype(np.float32)
    log_within = np.log(np.maximum(within, variance_floor)).astype(np.float32)
    if not all(np.isfinite(x).all() for x in (prior, log_between, log_within)):
        raise FloatingPointError("non-finite genomic-prior targets")

    # Deterministic 5-fold assignment only for official TRAIN CpGs. Held-out loci are -1.
    cpg_pos = _positions(cpg_ids, train_cpg, "Array cpg_idx")
    fold_id = np.full(EXPECTED_ARRAY_CPGS, -1, dtype=np.int8)
    rng = np.random.default_rng(20260812)
    shuffled = cpg_pos[rng.permutation(len(cpg_pos))]
    for fold, positions in enumerate(np.array_split(shuffled, 5)):
        fold_id[positions] = fold
    if np.any(fold_id[cpg_pos] < 0):
        raise RuntimeError("failed to assign every official train CpG to an OOF fold")

    np.save(out / "array_cpg_idx.npy", cpg_ids)
    np.save(out / "target_prior.npy", prior)
    np.save(out / "target_log_var_between.npy", log_between)
    np.save(out / "target_log_var_within.npy", log_within)
    np.save(out / "fold_id.npy", fold_id)
    np.save(out / "train_cpg_idx.npy", train_cpg)
    np.save(out / "heldout_cpg_idx.npy", heldout_cpg)

    report = {
        "version": "genomic_prior_v2",
        "status": "targets_built",
        "target_definition": {
            "samples": "official Array train samples only",
            "prior": "finite-observation weighted population mean beta per CpG",
            "within_variance": "sum_g within_group_SS_g / N_i",
            "between_variance": "sum_g n_gi*(mean_gi-global_mean_i)^2 / N_i",
            "variance_floor": variance_floor,
            "log_transform": "natural log for between/within; logit applied only when fitting prior probe",
        },
        "counts": {
            "train_samples": int(len(train_samples)),
            "cancer_types": int(len(cancer_names)),
            "array_cpgs": int(len(cpg_ids)),
            "train_cpgs": int(len(train_cpg)),
            "heldout_cpgs": int(len(heldout_cpg)),
        },
        "cancer_types": cancer_names.tolist(),
        "sha256_ids": {
            "train_samples": _sha256_ids(train_samples),
            "train_cpgs": _sha256_ids(train_cpg),
            "heldout_cpgs": _sha256_ids(heldout_cpg),
            "array_cpgs": _sha256_ids(cpg_ids),
        },
        "variance_decomposition": decomposition,
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (out / "targets.json").write_text(json.dumps(report, indent=2) + "\n")
    done.write_text("ok\n")
    return report


def _prepare_embeddings(root: Path, out: Path, chunk_rows: int) -> dict[str, object]:
    emb_path = out / "array_ntv3_embeddings.f16.npy"
    done = out / "embeddings.done"
    if done.is_file() and emb_path.is_file():
        x = np.load(emb_path, mmap_mode="r")
        if x.shape != (EXPECTED_ARRAY_CPGS, EXPECTED_EMBED_DIM):
            raise RuntimeError(f"cached Array embedding shape mismatch: {x.shape}")
        return {"status": "cached", "shape": list(x.shape)}

    paths = _canonical_paths(root)
    array_ids = np.load(out / "array_cpg_idx.npy", mmap_mode="r")
    started = time.time()
    with h5py.File(paths["atlas"], "r") as h:
        atlas_ids = np.asarray(h["cpg_idx"][...], dtype=np.int64)
        source = h["embedding"]
        if source.shape[1] != EXPECTED_EMBED_DIM:
            raise RuntimeError(f"unexpected NTv3 atlas embedding shape: {source.shape}")
        atlas_rows = _positions(atlas_ids, np.asarray(array_ids), "NTv3 atlas cpg_idx")
        mmap = np.lib.format.open_memmap(
            emb_path, mode="w+", dtype=np.float16, shape=(EXPECTED_ARRAY_CPGS, EXPECTED_EMBED_DIM)
        )
        for start in range(0, len(atlas_rows), chunk_rows):
            stop = min(start + chunk_rows, len(atlas_rows))
            physical = atlas_rows[start:stop]
            read_order = np.argsort(physical)
            sorted_rows = physical[read_order]
            block = np.asarray(source[sorted_rows, :], dtype=np.float16)
            inverse = np.empty_like(read_order)
            inverse[read_order] = np.arange(len(read_order))
            mmap[start:stop] = block[inverse]
            if start == 0 or stop == len(atlas_rows) or (start // chunk_rows) % 25 == 0:
                print(f"[embeddings] {stop:,}/{len(atlas_rows):,}", flush=True)
        del mmap
    done.write_text("ok\n")
    return {
        "status": "built",
        "shape": [EXPECTED_ARRAY_CPGS, EXPECTED_EMBED_DIM],
        "elapsed_hours": (time.time() - started) / 3600.0,
    }


def _target_matrix(out: Path) -> np.ndarray:
    p = np.load(out / "target_prior.npy", mmap_mode="r")
    between = np.load(out / "target_log_var_between.npy", mmap_mode="r")
    within = np.load(out / "target_log_var_within.npy", mmap_mode="r")
    logit = np.log(p / (1.0 - p)).astype(np.float32)
    return np.column_stack([logit, between, within]).astype(np.float32)


def _stats(x: np.ndarray, rows: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    sx = np.zeros(x.shape[1], dtype=np.float64)
    sxx = np.zeros(x.shape[1], dtype=np.float64)
    for start in range(0, len(rows), chunk):
        block = np.asarray(x[rows[start:start + chunk]], dtype=np.float32)
        sx += block.sum(axis=0, dtype=np.float64)
        sxx += np.square(block, dtype=np.float64).sum(axis=0)
    mean = sx / len(rows)
    var = np.maximum(sxx / len(rows) - mean * mean, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _inner_split(rows: np.ndarray, ids: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Stable hash-like split based on sorted IDs + deterministic RNG; target fold never enters selection.
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(ids[rows], kind="mergesort")
    ordered = rows[order]
    rng = np.random.default_rng(seed)
    shuffled = ordered[rng.permutation(len(ordered))]
    n_val = max(1000, int(round(fraction * len(shuffled))))
    n_val = min(n_val, len(shuffled) - 1000)
    return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])


def _train_probe(
    embeddings: np.ndarray,
    target: np.ndarray,
    fit_rows: np.ndarray,
    predict_rows: np.ndarray,
    *,
    device: str,
    max_epochs: int,
    patience: int,
    block_rows: int,
    inner_fraction: float,
    scope_seed: int,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ids = np.load(Path(embeddings.filename).parent / "array_cpg_idx.npy", mmap_mode="r")
    inner_train, inner_val = _inner_split(fit_rows, ids, inner_fraction, scope_seed)
    x_mean, x_std = _stats(embeddings, inner_train, block_rows)
    y_train = target[inner_train]
    y_mean = y_train.mean(axis=0).astype(np.float32)
    y_std = np.maximum(y_train.std(axis=0), 1e-6).astype(np.float32)

    model = ProbeEnsemble(list(seeds), dim=embeddings.shape[1], dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    amp = device.startswith("cuda")
    xm = torch.from_numpy(x_mean).to(device); xs = torch.from_numpy(x_std).to(device)
    ym = torch.from_numpy(y_mean).to(device); ys = torch.from_numpy(y_std).to(device)
    blocks = [inner_train[s:s + block_rows] for s in range(0, len(inner_train), block_rows)]
    rng = np.random.default_rng(scope_seed)
    best = float("inf"); best_epoch = 0; bad = 0; history = []
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train(); losses = []
        for bi in rng.permutation(len(blocks)):
            rows = blocks[int(bi)]
            x = torch.from_numpy(np.asarray(embeddings[rows], dtype=np.float32)).to(device)
            y = torch.from_numpy(target[rows]).to(device)
            x = (x - xm) / xs; y = (y - ym) / ys
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                all_pred = model.forward_all(x)
                loss = torch.stack([torch.nn.functional.mse_loss(p, y) for p in all_pred]).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval(); sse = np.zeros(3, np.float64); n = 0
        with torch.inference_mode():
            for start in range(0, len(inner_val), block_rows):
                rows = inner_val[start:start + block_rows]
                x = torch.from_numpy(np.asarray(embeddings[rows], dtype=np.float32)).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    pred_z = model((x - xm) / xs)
                pred = pred_z.float().cpu().numpy() * y_std + y_mean
                sse += np.square(pred - target[rows], dtype=np.float64).sum(axis=0); n += len(rows)
        channel = sse / n; val = float(channel.mean())
        row = {"epoch": epoch, "train_mse_z": float(np.mean(losses)), "validation_mse": val,
               **{f"validation_mse_{k}": float(v) for k, v in zip(TARGET_NAMES, channel)}}
        history.append(row); print(f"[probe-select] {row}", flush=True)
        if val < best - 1e-5:
            best = val; best_epoch = epoch; bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_epoch <= 0:
        raise RuntimeError("probe selection produced no checkpoint")

    # Deterministic refit on every admissible fit locus for exactly the selected epoch count.
    x_mean_f, x_std_f = _stats(embeddings, fit_rows, block_rows)
    yf = target[fit_rows]
    y_mean_f = yf.mean(axis=0).astype(np.float32)
    y_std_f = np.maximum(yf.std(axis=0), 1e-6).astype(np.float32)
    model = ProbeEnsemble(list(seeds), dim=embeddings.shape[1], dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xm = torch.from_numpy(x_mean_f).to(device); xs = torch.from_numpy(x_std_f).to(device)
    ym = torch.from_numpy(y_mean_f).to(device); ys = torch.from_numpy(y_std_f).to(device)
    blocks = [fit_rows[s:s + block_rows] for s in range(0, len(fit_rows), block_rows)]
    rng = np.random.default_rng(scope_seed)
    for epoch in range(1, best_epoch + 1):
        model.train()
        for bi in rng.permutation(len(blocks)):
            rows = blocks[int(bi)]
            x = torch.from_numpy(np.asarray(embeddings[rows], dtype=np.float32)).to(device)
            y = torch.from_numpy(target[rows]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                all_pred = model.forward_all((x - xm) / xs)
                yz = (y - ym) / ys
                loss = torch.stack([torch.nn.functional.mse_loss(p, yz) for p in all_pred]).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        print(f"[probe-refit] epoch {epoch}/{best_epoch}", flush=True)

    predictions = np.empty((len(predict_rows), 3), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(predict_rows), block_rows):
            stop = min(start + block_rows, len(predict_rows)); rows = predict_rows[start:stop]
            x = torch.from_numpy(np.asarray(embeddings[rows], dtype=np.float32)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                pred_z = model((x - xm) / xs)
            predictions[start:stop] = pred_z.float().cpu().numpy() * y_std_f + y_mean_f

    state = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "seeds": list(seeds), "dim": int(embeddings.shape[1]), "dropout": 0.1,
        "x_mean": x_mean_f, "x_std": x_std_f, "y_mean": y_mean_f, "y_std": y_std_f,
        "best_epoch": int(best_epoch), "selection_validation_mse": float(best),
        "target_names": list(TARGET_NAMES),
    }
    metrics = {
        "best_epoch": int(best_epoch), "selection_validation_mse": float(best),
        "fit_rows": int(len(fit_rows)), "predict_rows": int(len(predict_rows)),
        "inner_train_rows": int(len(inner_train)), "inner_validation_rows": int(len(inner_val)),
        "elapsed_hours": (time.time() - started) / 3600.0, "history": history,
    }
    return predictions, state, metrics


def _fit_scope(out: Path, scope: str, fold: int | None, device: str, max_epochs: int, patience: int,
               block_rows: int, inner_fraction: float) -> dict[str, object]:
    embeddings = np.load(out / "array_ntv3_embeddings.f16.npy", mmap_mode="r")
    target = _target_matrix(out)
    ids = np.load(out / "array_cpg_idx.npy", mmap_mode="r")
    fold_id = np.load(out / "fold_id.npy", mmap_mode="r")
    train_ids = np.load(out / "train_cpg_idx.npy", mmap_mode="r")
    heldout_ids = np.load(out / "heldout_cpg_idx.npy", mmap_mode="r")
    train_rows = _positions(ids, train_ids, "Array feature cpg_idx")

    if scope == "fold":
        assert fold is not None and 0 <= fold < 5
        predict_rows = np.flatnonzero(fold_id == fold).astype(np.int64)
        fit_rows = np.flatnonzero((fold_id >= 0) & (fold_id != fold)).astype(np.int64)
        name = f"fold_{fold}"
        scope_seed = 20260812 + fold
    else:
        fit_rows = train_rows
        predict_rows = _positions(ids, heldout_ids, "Array heldout cpg_idx")
        name = "full_fit"
        scope_seed = 20260899

    run_dir = out / "probes" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / ".done"
    pred_path = run_dir / "predictions.npz"
    if done.is_file() and pred_path.is_file():
        return json.loads((run_dir / "metrics.json").read_text())

    prediction, state, metrics = _train_probe(
        embeddings, target, fit_rows, predict_rows, device=device, max_epochs=max_epochs,
        patience=patience, block_rows=block_rows, inner_fraction=inner_fraction, scope_seed=scope_seed,
    )
    np.savez_compressed(pred_path, cpg_idx=np.asarray(ids[predict_rows], np.int64), prediction=prediction)
    torch.save(state, run_dir / "probe.pt")

    truth = target[predict_rows]
    err = prediction.astype(np.float64) - truth.astype(np.float64)
    mse = np.mean(err * err, axis=0)
    metrics.update({f"prediction_mse_{k}": float(v) for k, v in zip(TARGET_NAMES, mse)})
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    done.write_text("ok\n")
    return metrics


def _assemble(root: Path, out: Path) -> dict[str, object]:
    ids = np.load(out / "array_cpg_idx.npy", mmap_mode="r")
    target = _target_matrix(out)
    pred = np.full((len(ids), 3), np.nan, dtype=np.float32)
    filled = np.zeros(len(ids), dtype=np.int8)

    def insert(path: Path):
        z = np.load(path)
        local_ids = np.asarray(z["cpg_idx"], np.int64)
        local_pred = np.asarray(z["prediction"], np.float32)
        rows = _positions(ids, local_ids, "assembled Array cpg_idx")
        if np.any(filled[rows]):
            raise RuntimeError(f"duplicate predictions while assembling {path}")
        pred[rows] = local_pred; filled[rows] = 1

    for fold in range(5):
        insert(out / "probes" / f"fold_{fold}" / "predictions.npz")
    insert(out / "probes" / "full_fit" / "predictions.npz")
    if not np.all(filled == 1) or not np.isfinite(pred).all():
        raise RuntimeError(f"assembled feature store incomplete: filled={int(filled.sum())}/{len(filled)}")

    prior = 1.0 / (1.0 + np.exp(-np.clip(pred[:, 0], -35.0, 35.0)))
    prior = np.clip(prior, 1e-4, 1.0 - 1e-4).astype(np.float32)
    frame = pd.DataFrame({
        "cpg_idx": np.asarray(ids, np.int64),
        "pred_ntv3_prior": prior,
        "pred_log_var_between": pred[:, 1].astype(np.float32),
        "pred_log_var_within": pred[:, 2].astype(np.float32),
    })
    path = out / "locus_features.parquet"
    frame.to_parquet(path, index=False, compression="zstd")

    fold_id = np.load(out / "fold_id.npy", mmap_mode="r")
    oof_rows = np.flatnonzero(fold_id >= 0)
    heldout_rows = np.flatnonzero(fold_id < 0)
    report = {"version": "genomic_prior_v2", "rows": int(len(frame)), "locus_features": str(path)}
    for label, rows in (("oof_train", oof_rows), ("heldout_fullfit", heldout_rows)):
        err = pred[rows].astype(np.float64) - target[rows].astype(np.float64)
        mse = np.mean(err * err, axis=0)
        report[label] = {f"mse_{k}": float(v) for k, v in zip(TARGET_NAMES, mse)}
    report["provenance"] = {
        "canonical_root": str(root),
        "atlas": str(_canonical_paths(root)["atlas"]),
        "probe": "methylation_predictor.full_suite.probe.ProbeEnsemble",
        "probe_seeds": list(DEFAULT_SEEDS),
        "oof_folds": 5,
        "train_samples_only_for_targets": True,
        "train_cpg_oof": True,
        "heldout_cpg_used_for_probe_fit": False,
    }
    (out / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / ".done").write_text("ok\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-root", default=os.environ.get(
        "TCGA_CANONICAL_ROOT", "/raid/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1"
    ))
    p.add_argument("--output", default="/raid/DATASETS/MethylPredictionData/derived/genomic_prior_v2/array_genomewide")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("prepare")
    q.add_argument("--row-chunk", type=int, default=32)
    q.add_argument("--embedding-chunk", type=int, default=2048)
    q.add_argument("--variance-floor", type=float, default=1e-8)

    q = sub.add_parser("fit-fold")
    q.add_argument("--fold", type=int, required=True, choices=range(5))
    q.add_argument("--device", default="cuda")
    q.add_argument("--epochs", type=int, default=40)
    q.add_argument("--patience", type=int, default=6)
    q.add_argument("--block-rows", type=int, default=4096)
    q.add_argument("--inner-fraction", type=float, default=0.10)

    q = sub.add_parser("fit-full")
    q.add_argument("--device", default="cuda")
    q.add_argument("--epochs", type=int, default=40)
    q.add_argument("--patience", type=int, default=6)
    q.add_argument("--block-rows", type=int, default=4096)
    q.add_argument("--inner-fraction", type=float, default=0.10)

    sub.add_parser("assemble")
    args = p.parse_args()
    root = Path(args.canonical_root).resolve(); out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare":
        result = {
            "targets": _prepare_targets(root, out, args.row_chunk, args.variance_floor),
            "embeddings": _prepare_embeddings(root, out, args.embedding_chunk),
        }
    elif args.command == "fit-fold":
        result = _fit_scope(out, "fold", args.fold, args.device, args.epochs, args.patience, args.block_rows, args.inner_fraction)
    elif args.command == "fit-full":
        result = _fit_scope(out, "full", None, args.device, args.epochs, args.patience, args.block_rows, args.inner_fraction)
    else:
        result = _assemble(root, out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
