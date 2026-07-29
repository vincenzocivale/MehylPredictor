#!/usr/bin/env python3
"""Evaluate checkpoint predictions with gene expression fixed to the train mean."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import pyarrow.dataset as ds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-glob", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--prior", required=True, help="cpg_idx,mean_train")
    p.add_argument("--evaluation-stats", required=True, help="group-2 sufficient statistics")
    p.add_argument("--released-predictions", help="Optional released rows for factual checkpoint verification")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    files = sorted(glob.glob(args.predictions_glob))
    if not files:
        raise FileNotFoundError(args.predictions_glob)
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if raw[["cpg_idx", "sample_idx", "condition", "pred_methyl", "gt_methyl"]].isna().any().any():
        raise ValueError("Mean-RNA inference contains NaN in required columns")
    fixed = raw[raw.condition == "fixed_train_mean"].copy()
    factual = raw[raw.condition == "factual"].copy()
    expected = set(pd.read_parquet(args.test_manifest).cpg_idx.astype(int))
    for name, frame in (("fixed", fixed), ("factual", factual)):
        if frame.cpg_idx.duplicated().any():
            raise ValueError(f"{name} has duplicate CpGs across MDS workers")
        if set(frame.cpg_idx.astype(int)) != expected:
            raise ValueError(f"{name} covers {len(set(frame.cpg_idx))} CpGs, expected {len(expected)}")
    prior = pd.read_parquet(args.prior, columns=["cpg_idx", "mean_train"])
    stats = pd.read_parquet(args.evaluation_stats, columns=["cpg_idx", "n", "gt_mean", "gt_within_ss"])
    frame = fixed[["cpg_idx", "pred_methyl"]].merge(prior, on="cpg_idx", validate="one_to_one").merge(stats, on="cpg_idx", validate="one_to_one")
    y, pred = frame.mean_train.to_numpy(float), frame.pred_methyl.to_numpy(float)
    direct_mse = float(np.mean((pred - y) ** 2)); direct_mae = float(np.mean(np.abs(pred - y)))
    total_sse = float(np.sum(frame.n * (pred - frame.gt_mean) ** 2 + frame.gt_within_ss))
    result = {
        "command": " ".join(sys.argv),
        "design": "MethylProphet checkpoint with every gene-expression input fixed to the mean over the released group-1 training samples; one group-2 MDS row per CpG",
        "n_cpg": len(frame), "n_factual_controls": len(factual), "raw_rows": len(raw),
        "duplicate_handling": "rejected duplicate CpGs independently for factual and fixed conditions",
        "nan_handling": "rejected NaN in required inference columns", "direct_prior": {
            "mse": direct_mse, "mae": direct_mae,
            "pearson": float(pearsonr(y, pred)[0]), "spearman": float(spearmanr(y, pred)[0]),
        },
        "group2_static_prediction_mse": total_sse / float(frame.n.sum()),
        "source_worker_files": files,
    }
    if args.released_predictions:
        dataset = ds.dataset(args.released_predictions, format="parquet")
        target = factual[["cpg_idx", "sample_idx", "pred_methyl"]].rename(columns={"pred_methyl": "recomputed"})
        released = dataset.to_table(filter=(ds.field("group_idx") == 2) & ds.field("cpg_idx").isin(sorted(expected)),
                                   columns=["cpg_idx", "sample_idx", "pred_methyl"]).to_pandas()
        check = target.merge(released, on=["cpg_idx", "sample_idx"], validate="one_to_one")
        if len(check) != len(target):
            raise ValueError("A factual MDS row is absent from released group-2 predictions")
        error = check.recomputed.to_numpy(float) - check.pred_methyl.to_numpy(float)
        result["factual_checkpoint_verification"] = {"n_rows": len(check), "mae": float(np.abs(error).mean()),
                                                       "max_abs_error": float(np.abs(error).max())}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    fixed.to_parquet(out.with_name("methylprophet_mean_rna_fixed_per_cpg.parquet"), index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
