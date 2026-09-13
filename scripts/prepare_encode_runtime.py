#!/usr/bin/env python3
"""Build the ENCODE study's rna_cache / cpg_targets_dir / prior_cache.

Mirrors methylprophet_table5_tcga_chr1's rna/features caches, but scoped to
the ENCODE canonical bundle (datasets/methylprophet_encode_v1) built by
scripts/prepare_encode_canonical.py and the train/val split already computed
there. Writes under <output-root>/rna, <output-root>/cpg_targets,
<output-root>/prior.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py
import numpy as np

from methylation_predictor.benchmark.methylprophet.cache import prepare_rna_cache


def build_cpg_mean(methylation_h5: Path, train_sample_idx: np.ndarray, out_dir: Path, *, chunk: int = 200_000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(methylation_h5, "r") as h:
        X = h["X"]  # samples x cpgs
        sample_idx = np.asarray(h["sample_idx"][...], dtype=np.int64)
        rows = np.searchsorted(sample_idx, np.sort(train_sample_idx))
        assert np.array_equal(sample_idx[rows], np.sort(train_sample_idx))
        n_cpg = X.shape[1]
        mean = np.empty(n_cpg, dtype=np.float32)
        for start in range(0, n_cpg, chunk):
            stop = min(start + chunk, n_cpg)
            block = np.asarray(X[rows, start:stop], dtype=np.float32)
            with np.errstate(invalid="ignore"):
                mean[start:stop] = np.nanmean(block, axis=0)
        cpg_idx = np.arange(n_cpg, dtype=np.int64)
        finite = np.isfinite(mean)
    np.save(out_dir / "cpg_idx.npy", cpg_idx[finite])
    return {"n_cpg": int(n_cpg), "n_finite": int(finite.sum())}, mean[finite]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encode-root", required=True)
    p.add_argument("--output-root", required=True)
    args = p.parse_args()

    root = Path(args.encode_root)
    out = Path(args.output_root)
    train_sample_idx = np.load(root / "cpg" / "splits" / "train_sample_idx.npy")

    rna_result = prepare_rna_cache(
        canonical_root=root,
        output_dir=out / "rna",
        train_sample_idx=train_sample_idx,
        rna_filename="encode_rna_official_full.h5",
    )
    print(json.dumps({"step": "rna_cache", **rna_result}))

    stats, mean = build_cpg_mean(root / "methylation" / "encode_wgbs_full.h5", train_sample_idx, out / "cpg_targets")
    np.save(out / "cpg_targets" / "target_mu.npy", mean.astype(np.float32))
    (out / "prior").mkdir(parents=True, exist_ok=True)
    np.save(out / "prior" / "cpg_idx.npy", np.load(out / "cpg_targets" / "cpg_idx.npy"))
    np.save(out / "prior" / "prior.npy", mean.astype(np.float32))
    print(json.dumps({"step": "cpg_targets_and_prior", **stats}))


if __name__ == "__main__":
    main()
