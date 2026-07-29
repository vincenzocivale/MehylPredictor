"""CpG-cluster bootstrap CIs for the exact released-row decompositions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(frame: pd.DataFrame, prior: np.ndarray | None = None) -> dict[str, float]:
    n = frame.n.to_numpy(float)
    mean = frame.gt_mean.to_numpy(float)
    pred = frame.pred_mean.to_numpy(float)
    total_n = n.sum()
    grand = np.dot(n, mean) / total_n
    ss_locus = np.dot(n, (mean - grand) ** 2)
    ss_within = frame.gt_within_ss.sum()
    dynamic_sse = (frame.gt_within_ss + frame.pred_within_ss - 2 * frame.residual_cross_product).sum()
    static_sse = np.dot(n, (pred - mean) ** 2)
    out = {
        "f_locus": float(ss_locus / (ss_locus + ss_within)),
        "f_within": float(ss_within / (ss_locus + ss_within)),
        "static_mse": float(static_sse / total_n),
        "dynamic_mse": float(dynamic_sse / total_n),
        "dynamic_skill": float(1 - dynamic_sse / ss_within),
    }
    if prior is not None:
        prior_sse = (frame.gt_within_ss.to_numpy(float) + n * (prior - mean) ** 2).sum()
        model_sse = static_sse + dynamic_sse
        out["prior_mse"] = float(prior_sse / total_n)
        out["total_skill"] = float(1 - model_sse / prior_sse)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--by-cpg", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prior", help="Optional parquet: cpg_idx, mean_train")
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=9176)
    a = p.parse_args()
    frame = pd.read_parquet(a.by_cpg)
    required = {"cpg_idx", "n", "gt_mean", "pred_mean", "gt_within_ss", "pred_within_ss", "residual_cross_product"}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing columns {sorted(missing)}")
    prior_values = None
    if a.prior:
        prior = pd.read_parquet(a.prior)[["cpg_idx", "mean_train"]]
        frame = frame.merge(prior, on="cpg_idx", how="left", validate="one_to_one")
        if frame.mean_train.isna().any():
            raise ValueError("prior does not cover every CpG")
        prior_values = frame.mean_train.to_numpy(float)
    point = _metrics(frame, prior_values)
    rng = np.random.default_rng(a.seed)
    values = {key: np.empty(a.replicates) for key in point}
    n_cpg = len(frame)
    # A cluster resample treats CpGs as the independent units and preserves all
    # patient rows within each selected locus.
    for rep in range(a.replicates):
        idx = rng.integers(0, n_cpg, n_cpg)
        sample = frame.iloc[idx]
        pri = prior_values[idx] if prior_values is not None else None
        result = _metrics(sample, pri)
        for key, value in result.items():
            values[key][rep] = value
    output = {
        "estimand": "row-weighted metrics; percentile CI from CpG-cluster bootstrap",
        "n_cpg": n_cpg,
        "replicates": a.replicates,
        "seed": a.seed,
        "point": point,
        "ci_95": {key: [float(np.quantile(x, .025)), float(np.quantile(x, .975))] for key, x in values.items()},
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
