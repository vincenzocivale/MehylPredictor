#!/usr/bin/env python3
"""Leakage-free downstream test: static locus prior + MP dynamic residual.

Uses per-CpG sufficient statistics from released predictions.  Alpha is fit on
the validation loci only; test-locus methylation targets are never used to
select alpha or any static-model parameter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sse(frame: pd.DataFrame, prior: np.ndarray, alpha: float) -> np.ndarray:
    # Dynamic signal is p_s - E_train[p_s].  Available group summaries provide
    # p/y means and centred cross-products, so raw rows are not needed.
    offset = prior + alpha * (frame.pred_mean - frame.pred_mean_train) - frame.gt_mean
    return (frame.n * offset * offset + alpha * alpha * frame.pred_within_ss + frame.gt_within_ss
            - 2 * alpha * frame.residual_cross_product).to_numpy(float)


def _alpha(frame: pd.DataFrame, prior: np.ndarray) -> float:
    t = (frame.pred_mean - frame.pred_mean_train).to_numpy(float)
    base = prior - frame.gt_mean.to_numpy(float)
    numerator = float(frame.residual_cross_product.sum() - np.dot(frame.n.to_numpy(float) * base, t))
    denominator = float(frame.pred_within_ss.sum() + np.dot(frame.n.to_numpy(float), t * t))
    return numerator / denominator if denominator else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locus-predictions", required=True, help="runner locus_predictions.parquet")
    p.add_argument("--reference-stats", required=True, help="released group-1 summary with pred_mean_train")
    p.add_argument("--evaluation-stats", required=True, help="released group-2 summary")
    p.add_argument("--mean-rna-prior", help="optional one-row-per-CpG fixed-training-mean RNA MP predictions")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    pred = pd.read_parquet(args.locus_predictions)
    # Diagnostic upper control: the empirical train-only locus prior has no
    # sequence-estimation error, but uses exactly the same dynamic component.
    pred["pred_empirical_prior"] = pred["mean_train"]
    if args.mean_rna_prior:
        mean_rna = pd.read_parquet(args.mean_rna_prior, columns=["cpg_idx", "pred_methyl"])
        if mean_rna.cpg_idx.duplicated().any():
            raise ValueError("--mean-rna-prior must contain exactly one row per CpG")
        pred = pred.merge(mean_rna.rename(columns={"pred_methyl": "pred_methylprophet_mean_rna"}), on="cpg_idx", how="left", validate="one_to_one")
        if pred.pred_methylprophet_mean_rna.isna().any():
            raise ValueError("--mean-rna-prior does not cover every validation/test locus")
    model_columns = [x for x in pred if x.startswith("pred_")]
    reference = pd.read_parquet(args.reference_stats, columns=["cpg_idx", "pred_mean"])
    reference = reference.rename(columns={"pred_mean": "pred_mean_train"})
    evaluation = pd.read_parquet(args.evaluation_stats)
    frame = pred.merge(reference, on="cpg_idx", validate="one_to_one").merge(evaluation, on="cpg_idx", validate="one_to_one")
    result = {"design": {"dynamic_signal": "MethylProphet prediction minus group-1 prediction mean per CpG",
                           "alpha": "closed-form least squares on validation loci only",
                           "evaluation": "released group-2 sufficient statistics, held-out test loci only"},
              "models": {}}
    for column in model_columns:
        val, test = frame[frame.split == "validation"].copy(), frame[frame.split == "test"].copy()
        alpha = _alpha(val, val[column].to_numpy(float))
        total = float(test.n.sum())
        prior_sse = _sse(test, test[column].to_numpy(float), 0.0)
        hybrid_sse = _sse(test, test[column].to_numpy(float), alpha)
        mp_sse = _sse(test, test.pred_mean_train.to_numpy(float), 1.0)
        result["models"][column.removeprefix("pred_")] = {
            "alpha_validation": alpha, "test_cpg": int(len(test)), "test_rows": int(total),
            "prior_mse": float(prior_sse.sum() / total), "hybrid_mse": float(hybrid_sse.sum() / total),
            "methylprophet_mse": float(mp_sse.sum() / total),
            "hybrid_total_skill_vs_prior": float(1 - hybrid_sse.sum() / prior_sse.sum()),
            "hybrid_total_skill_vs_methylprophet": float(1 - hybrid_sse.sum() / mp_sse.sum()),
        }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
