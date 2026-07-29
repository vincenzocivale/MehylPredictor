#!/usr/bin/env python3
"""Post-hoc CpG-variability-tertile skill breakdown from saved panel predictions.

Reads a `predictions_<panel>.npz` written by `rna_branch train` (requires
`evaluation.save_predictions: true`) plus the run's `locus_features.parquet`,
bins CpGs into terciles of predicted total variability
(exp(pred_log_var_between) + exp(pred_log_var_within)), and reports
skill-vs-prior per tercile. Read-only; writes one JSON report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_tertile_report(predictions_npz: Path, locus_features: Path) -> dict:
    data = np.load(predictions_npz)
    target, prediction, prior = data["target"], data["prediction"], data["prior"]
    cpg_idx = data["cpg_idx"].astype(str)

    features = pd.read_parquet(locus_features)
    features["cpg_idx"] = features["cpg_idx"].astype(str)
    variability_proxy = (
        np.exp(features["pred_log_var_between"]) + np.exp(features["pred_log_var_within"])
    )
    proxy_by_cpg = dict(zip(features["cpg_idx"], variability_proxy))
    missing = [c for c in cpg_idx if c not in proxy_by_cpg]
    if missing:
        raise ValueError(f"{len(missing)} CpGs in predictions are missing from locus_features")
    proxy = np.asarray([proxy_by_cpg[c] for c in cpg_idx])

    valid = np.isfinite(target) & np.isfinite(prediction)
    cpg_model_mse = np.full(len(cpg_idx), np.nan)
    cpg_prior_mse = np.full(len(cpg_idx), np.nan)
    for j in range(len(cpg_idx)):
        column_valid = valid[:, j]
        if not column_valid.any():
            continue
        cpg_model_mse[j] = float(np.mean((target[column_valid, j] - prediction[column_valid, j]) ** 2))
        cpg_prior_mse[j] = float(np.mean((target[column_valid, j] - prior[j]) ** 2))

    order = np.argsort(proxy)
    tertile_edges = np.array_split(order, 3)
    tertile_report = {}
    for name, indices in zip(("low", "mid", "high"), tertile_edges):
        model_mse = np.nanmean(cpg_model_mse[indices])
        prior_mse = np.nanmean(cpg_prior_mse[indices])
        skill = 1.0 - model_mse / prior_mse if prior_mse > 0 else float("nan")
        tertile_report[name] = {
            "n_cpgs": int(len(indices)),
            "variability_proxy_range": [float(proxy[indices].min()), float(proxy[indices].max())],
            "model_mse": float(model_mse),
            "prior_mse": float(prior_mse),
            "skill_vs_prior": float(skill),
        }
    return {"source": str(predictions_npz), "tertiles": tertile_report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--locus-features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compute_tertile_report(Path(args.predictions), Path(args.locus_features))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
