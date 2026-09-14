#!/usr/bin/env python3
"""Run the released DeepCpG DNA-only model on validation CpGs.

DeepCpG's DNA-only submodel is patient agnostic: each validation CpG is scored
from its hg38 sequence window and the score is replicated for every training
patient so that the output has the same long format as the other FM runners.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS = REPO / "external/checkpoints/deepcpg"
DEFAULT_FASTA = Path("/data2/home/vcivale/projects/methylation/backup/MethylPredictionData/reference/hg38/hg38.fa")
VARIANTS = ("hou2016_hcc_dna", "hou2016_hepg2_dna")


def _paths(scope: str):
    from run_official_views import _scope_paths

    h5, registry, split_dir, key, _ = _scope_paths(scope)
    return h5, registry, split_dir, key


def _load_targets(scope: str, registry_path: Path, split_dir: Path) -> pd.DataFrame:
    idx_name = "val_cpg_idx.npy" if scope == "encode" else "array_val_cpg_idx.npy"
    val = set(np.load(split_dir / idx_name).astype(np.int64).tolist())
    reg = pd.read_parquet(registry_path, columns=["cpg_idx", "chr", "pos"])
    reg = reg[reg["cpg_idx"].isin(val)].copy()
    reg["cpg_id"] = reg["chr"].astype(str) + ":" + reg["pos"].astype(str)
    return reg.sort_values("cpg_idx").reset_index(drop=True)


def run(scope: str, variant: str, fasta: Path, checkpoints: Path, output_root: Path) -> Path:
    from methylation_predictor.benchmark.foundation_models.adapters.deepcpg_adapter import (
        DNA_WINDOW_LEN,
        build_random_model,
        load_checkpoint_into,
        one_hot_dna_windows,
        resolve_resources,
    )

    h5_path, registry_path, split_dir, dataset_key = _paths(scope)
    targets = _load_targets(scope, registry_path, split_dir)
    sample_idx_name = "train_sample_idx.npy" if scope == "encode" else "array_train_sample_idx.npy"
    train_samples = np.load(split_dir / sample_idx_name).astype(np.int64)
    with h5py.File(h5_path, "r") as h5:
        sample_axis = np.asarray(h5["sample_idx"])
        cpg_axis = np.asarray(h5["cpg_idx"]) if "cpg_idx" in h5 else np.arange(h5[dataset_key].shape[1])
        row_by_sample = {int(x): i for i, x in enumerate(sample_axis)}
        col_by_cpg = {int(x): i for i, x in enumerate(cpg_axis)}
        rows = np.asarray([row_by_sample[int(x)] for x in train_samples], dtype=np.int64)
        cols = np.asarray([col_by_cpg[int(x)] for x in targets["cpg_idx"]], dtype=np.int64)
        row_order = np.argsort(rows)
        col_order = np.argsort(cols)
        matrix = np.asarray(h5[dataset_key][rows[row_order], :][:, cols[col_order]], dtype=np.float32)
        y_true = matrix[np.argsort(row_order)][:, np.argsort(col_order)]
        sample_names = np.asarray(h5["sample_name"])[rows].astype(str)

    resources = resolve_resources(checkpoints, variant)
    model = build_random_model(resources.arch_path)
    load_checkpoint_into(model, resources.weights_path)
    windows = one_hot_dna_windows(fasta, targets["chr"].astype(str), targets["pos"].astype(int), DNA_WINDOW_LEN)
    raw = model.predict(windows, batch_size=128, verbose=0)
    if isinstance(raw, list):
        # The released DNA heads are cell-specific (25 for HCC, 6 for HepG2).
        # This benchmark is patient-agnostic, so use the released head mean as
        # the single transferable CpG score before replicating it across ours.
        pred_cpg = np.mean(np.column_stack([np.asarray(x).reshape(len(targets), -1)[:, 0] for x in raw]), axis=1)
    else:
        pred_cpg = np.asarray(raw).reshape(len(targets), -1).mean(axis=1)
    pred_cpg = np.clip(pred_cpg.astype(np.float32), 0.0, 1.0)

    out = output_root / scope
    out.mkdir(parents=True, exist_ok=True)
    prediction = pd.DataFrame({
        "sample_id": np.repeat(sample_names, len(targets)),
        "cpg_id": np.tile(targets["cpg_id"].to_numpy(), len(sample_names)),
        "y_true": y_true.reshape(-1),
        "y_pred": np.tile(pred_cpg, len(sample_names)),
        "masked": True,
        "observed": np.isfinite(y_true.reshape(-1)),
    })
    stem = f"{scope}_deepcpg_{variant}"
    prediction_path = out / f"{stem}_predictions.parquet"
    prediction.to_parquet(prediction_path, index=False)
    metadata = {
        "scope": scope, "model": "deepcpg", "variant": variant,
        "view": "val_cpg_x_train_sample", "patient_agnostic": True,
        "checkpoint": str(resources.weights_path), "architecture": str(resources.arch_path),
        "fasta": str(fasta), "n_training_samples": len(sample_names),
        "n_validation_cpgs": len(targets), "dna_window_len": DNA_WINDOW_LEN,
    }
    (out / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2))
    return prediction_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["chr1", "chr123", "encode"], required=True)
    parser.add_argument("--variant", choices=[*VARIANTS, "both"], default="both")
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output-root", type=Path, default=REPO / "local_methyl_data/runs/foundation_models")
    args = parser.parse_args()
    variants = VARIANTS if args.variant == "both" else (args.variant,)
    for variant in variants:
        run(args.scope, variant, args.fasta, args.checkpoints, args.output_root)


if __name__ == "__main__":
    main()
