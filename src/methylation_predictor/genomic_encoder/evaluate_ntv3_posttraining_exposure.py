#!/usr/bin/env python3
"""Stratify frozen-NTv3 prior errors by Borzoi post-training exposure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _metrics(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    y, p = frame.mean_train.to_numpy(float), frame[column].to_numpy(float)
    return {"n_cpg": int(len(frame)), "mse": float(np.mean((p - y) ** 2)),
            "mae": float(np.mean(np.abs(p - y))),
            "pearson": float(np.corrcoef(y, p)[0, 1]) if len(frame) > 1 else float("nan"),
            "spearman": float(spearmanr(y, p).statistic) if len(frame) > 1 else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locus-predictions", required=True)
    p.add_argument("--fold-manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    pred = pd.read_parquet(a.locus_predictions)
    fold = pd.read_parquet(a.fold_manifest, columns=["cpg_idx", "posttraining_exposure"])
    if pred.cpg_idx.duplicated().any() or fold.cpg_idx.duplicated().any():
        raise ValueError("inputs must have one row per CpG")
    joined = pred.merge(fold, on="cpg_idx", how="left", validate="one_to_one")
    if joined.posttraining_exposure.isna().any():
        raise ValueError("fold manifest does not cover all predicted CpGs")
    result = {"design": "beta-space locus-prior error, stratified by official Borzoi interval exposure",
              "rows": {str(k): int(v) for k, v in joined.posttraining_exposure.value_counts().items()}, "models": {}}
    for column in [x for x in joined if x.startswith("pred_")]:
        per_split = {}
        for split in ("validation", "test"):
            subset = joined[joined.split == split]
            per_split[split] = {str(label): _metrics(part, column) for label, part in subset.groupby("posttraining_exposure")}
        result["models"][column.removeprefix("pred_")] = per_split
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
