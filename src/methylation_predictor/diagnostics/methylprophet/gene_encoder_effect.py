#!/usr/bin/env python3
"""Evaluate controlled gene-encoder intervention predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


KEYS = ["cpg_idx", "sample_idx"]
REQUIRED = set(KEYS + ["condition", "pred_methyl", "gt_methyl"])


def _metrics(frame: pd.DataFrame, fixed_condition: str = "fixed") -> dict[str, float | int]:
    factual = frame[frame.condition == "factual"].set_index(KEYS).sort_index()
    fixed = frame[frame.condition == fixed_condition].set_index(KEYS).sort_index()
    if factual.index.has_duplicates or fixed.index.has_duplicates:
        raise ValueError("Each condition must contain one row per (cpg_idx, sample_idx)")
    if not factual.index.equals(fixed.index):
        raise ValueError("Factual and fixed conditions do not cover identical keys")
    if not np.allclose(factual.gt_methyl.to_numpy(), fixed.gt_methyl.to_numpy(), rtol=0, atol=0):
        raise ValueError("Ground truth differs across controlled conditions")
    y = factual.gt_methyl.to_numpy(dtype=float)
    factual_pred = factual.pred_methyl.to_numpy(dtype=float)
    fixed_pred = fixed.pred_methyl.to_numpy(dtype=float)
    delta = factual_pred - fixed_pred
    target = y - fixed_pred
    factual_sse = float(np.dot(y - factual_pred, y - factual_pred))
    fixed_sse = float(np.dot(target, target))
    corr = float(np.corrcoef(delta, target)[0, 1]) if np.std(delta) and np.std(target) else float("nan")
    return {
        "n_pairs": len(frame) // 2,
        "delta_gene_rms": float(np.sqrt(np.mean(delta * delta))),
        "delta_gene_mean": float(delta.mean()),
        "delta_gene_std": float(delta.std(ddof=0)),
        "delta_gene_target_correlation": corr,
        "mse_factual": factual_sse / len(y),
        "mse_fixed": fixed_sse / len(y),
        "incremental_skill_vs_fixed": 1.0 - factual_sse / fixed_sse if fixed_sse else float("nan"),
    }


def run(args: argparse.Namespace) -> None:
    dataset = ds.dataset(args.predictions, format="parquet")
    missing = REQUIRED - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Missing intervention columns: {sorted(missing)}")
    frame = dataset.to_table(columns=sorted(REQUIRED)).to_pandas().dropna()
    required_conditions = {"factual", args.fixed_condition}
    if not required_conditions <= set(frame.condition.unique()):
        raise ValueError(f"Required conditions are absent: {sorted(required_conditions)}")
    result = _metrics(frame[frame.condition.isin(required_conditions)].copy(), args.fixed_condition)
    result["fixed_condition"] = args.fixed_condition
    result["conditions_present"] = sorted(map(str, frame.condition.unique()))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--predictions", required=True, help="Parquet directory with condition-labelled predictions")
    result.add_argument("--output", required=True)
    result.add_argument("--fixed-condition", default="fixed")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
