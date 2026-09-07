#!/usr/bin/env python3
"""Score a DeepCpG per-CpG prediction against real Array beta values.

DeepCpG's DNA-only submodel is **patient-agnostic** (see `deepcpg_adapter.py`'s module
docstring) -- its prediction is the same for every sample at a given CpG. This script
broadcasts that single per-CpG value across every context sample before scoring, matching
this repo's own `CpG Prior` baseline framing rather than CpGPT/MethylGPT's per-patient
masked recovery. Because there is zero true prediction variance across samples for a fixed
CpG, `mas_pcc` (per-CpG correlation across the sample axis) is numerically degenerate --
`ArrayMomentMetrics._corr`'s `vx*vy>0` guard is meant to catch a zero-variance array (giving
NaN), but the variance here is computed as a difference of two near-equal sums
(`cpp - cp*cp/n`), so floating-point rounding can leave a tiny nonzero residual instead of an
exact zero; observed in practice (2026-09-07) as `mas_pcc` landing near 0.0 rather than
cleanly NaN. **Either way, `mas_pcc` carries no real signal for a patient-agnostic model and
should not be read** -- `mac_pcc` (per-sample correlation across the CpG axis) and MSE/MAE
are the numbers that actually mean something here.

Needs the main `methyl-predictor` conda env (h5py, to re-read real target betas via
`TCGACanonicalBundle`) -- run after `run_deepcpg_predict.py` (which needs `deepcpg-env`).

    conda activate methyl-predictor
    python scripts/benchmark_foundation_models/score_deepcpg_reconstruct.py \\
        --scope chr1 \\
        --canonical-root /dune/.../methylprophet_repro_v1 \\
        --table5-protocol-root /dune/.../table5_protocol \\
        --locations-json derived/foundation_models/deepcpg/val_locations_chr1_all.json \\
        --pred-per-cpg derived/foundation_models/deepcpg/chr1_hou2016_hcc_dna_pred.npy \\
        --variant hou2016_hcc_dna \\
        --output results/reference/appendix/foundation_models/chr1_deepcpg_hou2016_hcc_dna_masked_recovery.json
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", required=True, choices=["chr1", "chr123"])
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--table5-protocol-root", required=True)
    parser.add_argument("--array-cpg-registry", required=True)
    parser.add_argument("--locations-json", required=True, type=Path)
    parser.add_argument("--pred-per-cpg", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_sample_idx = np.load(Path(args.table5_protocol_root) / "array_train_sample_idx.npy")
    val_cpg_idx = np.load(Path(args.table5_protocol_root) / "array_val_cpg_idx.npy")

    locations = json.loads(args.locations_json.read_text())
    chroms, positions = locations["chrom"], locations["pos"]

    registry = pd.read_parquet(args.array_cpg_registry, columns=["cpg_idx", "chr", "pos"])
    lookup = {(row.chr, int(row.pos)): int(row.cpg_idx) for row in registry.itertuples()}
    location_cpg_idx = np.array([lookup[(c, p)] for c, p in zip(chroms, positions)], dtype=np.int64)
    assert set(location_cpg_idx.tolist()) == set(val_cpg_idx.tolist()), (
        "locations-json must be exactly the scope's official val_cpg_idx set"
    )

    pred_per_cpg = np.load(args.pred_per_cpg).astype(np.float32)
    if pred_per_cpg.shape != (len(location_cpg_idx),):
        raise ValueError(f"pred_per_cpg shape {pred_per_cpg.shape} != ({len(location_cpg_idx)},)")

    bundle = TCGACanonicalBundle.from_root(args.canonical_root)
    try:
        source = bundle.sources["array"]
        row_positions = source.rows_of_samples(train_sample_idx)
        target = source.block(row_positions, location_cpg_idx)
    finally:
        bundle.close()

    pred_broadcast = np.broadcast_to(pred_per_cpg[None, :], target.shape)

    metrics = ArrayMomentMetrics(len(train_sample_idx), len(location_cpg_idx))
    metrics.add(0, 0, target, pred_broadcast, prior=np.full(target.shape[1], np.nan))

    result = {
        "schema_version": 1,
        "model": "deepcpg",
        "variant": args.variant,
        "patient_agnostic": True,
        "evaluation_scope": args.scope,
        "checkpoint": str(args.checkpoint),
        "n_context_samples": len(train_sample_idx),
        "n_val_cpgs": len(location_cpg_idx),
        "views": {"val_cpg_x_train_sample": {"global": metrics.finalize()}},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
