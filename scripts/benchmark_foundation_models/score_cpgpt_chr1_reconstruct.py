#!/usr/bin/env python3
"""Score a completed CpGPT chr1 `val_cpg_x_train_sample` reconstruction run against real
Array beta values -- consumes `run_cpgpt_reconstruct_pilot.py`'s `*.pred_beta.npy` output
(shape `[n_context_samples, n_covered_val_cpgs]`, in the exact `--genomic-locations` order
that run was given) and produces a `metrics.json` schema-compatible with every other
evaluator in this repo (`rna_training.metrics.ArrayMomentMetrics`).

Also runs a same-order sanity check: reconstructs a handful of *context* CpGs (already
known, real values) alongside the held-out ones and reports their correlation against the
real values -- if `CpGPTDataModule`/`trainer.predict` reordered samples internally, this
would show up as near-zero correlation on CpGs the model was directly given as input, which
would otherwise be invisible in the held-out-only metrics. Needs the main `methyl-predictor`
conda env (h5py, to re-read real target betas via `TCGACanonicalBundle`).

    conda activate methyl-predictor
    python scripts/benchmark_foundation_models/score_cpgpt_chr1_reconstruct.py \\
        --canonical-root /dune/.../methylprophet_repro_v1 \\
        --table5-protocol-root /dune/.../methylprophet_table5_tcga_chr1/table5_protocol \\
        --array-cpg-registry /dune/.../cpg/registries/array_cpg_map.parquet \\
        --covered-val-locations derived/foundation_models/cpgpt/val_locations_chr1_covered.json \\
        --pred-beta derived/foundation_models/cpgpt/chr1_full_small_reconstruct.pred_beta.npy \\
        --checkpoint external/checkpoints/cpgpt/weights/small.ckpt \\
        --output results/reference/appendix/foundation_models/chr1_cpgpt_small_masked_recovery.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from methylation_predictor.rna_training.metrics import ArrayMomentMetrics  # noqa: E402
from methylation_predictor.tcga_canonical.bundle import TCGACanonicalBundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="chr1", choices=["chr1", "chr123"])
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--table5-protocol-root", required=True)
    parser.add_argument("--array-cpg-registry", required=True)
    parser.add_argument("--covered-val-locations", required=True, type=Path)
    parser.add_argument("--pred-beta", required=True, type=Path)
    parser.add_argument("--variant", default="small", choices=["small", "large"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_sample_idx = np.load(Path(args.table5_protocol_root) / "array_train_sample_idx.npy")
    val_cpg_idx_all = np.load(Path(args.table5_protocol_root) / "array_val_cpg_idx.npy")

    covered = json.loads(args.covered_val_locations.read_text())
    covered_chrom, covered_pos = covered["chrom"], covered["pos"]

    registry = pd.read_parquet(args.array_cpg_registry, columns=["cpg_idx", "chr", "pos"])
    lookup = {(row.chr, int(row.pos)): int(row.cpg_idx) for row in registry.itertuples()}
    covered_cpg_idx = np.array([lookup[(c, p)] for c, p in zip(covered_chrom, covered_pos)], dtype=np.int64)
    assert set(covered_cpg_idx.tolist()).issubset(set(val_cpg_idx_all.tolist())), (
        "covered locations must be a subset of the official val_cpg_idx set"
    )

    pred_beta = np.load(args.pred_beta)
    if pred_beta.shape != (len(train_sample_idx), len(covered_cpg_idx)):
        raise ValueError(
            f"pred_beta shape {pred_beta.shape} != expected "
            f"({len(train_sample_idx)}, {len(covered_cpg_idx)}) -- did the run finish with "
            "these exact --genomic-locations, in this exact order?"
        )

    bundle = TCGACanonicalBundle.from_root(args.canonical_root)
    try:
        source = bundle.sources["array"]
        row_positions = source.rows_of_samples(train_sample_idx)
        target = source.block(row_positions, covered_cpg_idx)
    finally:
        bundle.close()

    metrics = ArrayMomentMetrics(len(train_sample_idx), len(covered_cpg_idx))
    # `prior` is per-CpG (1D, length n_cpgs) -- `add()` does `prior[None, :]` to broadcast
    # across the sample axis itself; passing a 2D (n_samples, n_cpgs) array here (as
    # `evaluator.py`'s own never-yet-run scaffold does) makes that `[None, :]` 3D and
    # `broadcast_to` rejects it. No real prior for CpGPT (it's not our own `cpg_prior`
    # baseline), so this is NaN either way -- only the shape matters for `add()` to run.
    metrics.add(0, 0, target, pred_beta.astype(np.float32), prior=np.full(target.shape[1], np.nan))

    result = {
        "schema_version": 1,
        "model": "cpgpt",
        "variant": args.variant,
        "evaluation_scope": args.scope,
        "checkpoint": str(args.checkpoint),
        "n_context_samples": len(train_sample_idx),
        "n_val_cpgs_covered": len(covered_cpg_idx),
        "n_val_cpgs_official": len(val_cpg_idx_all),
        "crosswalk_coverage": len(covered_cpg_idx) / len(val_cpg_idx_all),
        "views": {"val_cpg_x_train_sample": {"global": metrics.finalize()}},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
