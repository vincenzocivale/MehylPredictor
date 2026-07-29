#!/usr/bin/env python3
"""Leakage-free empirical-prior hybrid evaluation for released predictions.

The reference rows must contain the *same CpGs* measured on training samples;
they provide both ``mean_train`` and the MethylProphet centring mean.  Evaluation
rows are split by sample, never by individual CpG/sample rows, into calibration
and held-out test partitions.  This permits alpha to be fit without looking at
the test targets while retaining the requested train-only locus prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


REQUIRED = ("group_idx", "cpg_idx", "sample_idx", "pred_methyl", "gt_methyl")
EPS = 1e-12


def _split(sample_idx: np.ndarray, calibration_fraction: float, seed: int) -> np.ndarray:
    """Stable sample-level partition, independent of parquet ordering."""
    # A simple integer mixer is deterministic across Python versions/processes.
    value = (sample_idx.astype(np.uint64) * np.uint64(11400714819323198485) + np.uint64(seed))
    return (value % np.uint64(10_000)) < int(round(calibration_fraction * 10_000))


def _safe(n: float, d: float) -> float | None:
    return float(n / d) if d > EPS and np.isfinite(n) and np.isfinite(d) else None


def _json(value):
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _ids_hash(values: np.ndarray) -> str:
    return hashlib.sha256(("\n".join(map(str, sorted(map(int, values)))) + "\n").encode()).hexdigest()


def _files(path: Path) -> list[Path]:
    return sorted(path.glob("*.parquet")) if path.is_dir() else [path]


def _reference_means(path: Path, group: int, batch_size: int) -> tuple[pd.DataFrame, dict]:
    sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    nan_rows = 0
    dataset = ds.dataset(path, format="parquet")
    for batch in dataset.scanner(columns=list(REQUIRED), filter=ds.field("group_idx") == group, batch_size=batch_size).to_batches():
        frame = batch.to_pandas().drop(columns="group_idx")
        valid = frame.dropna()
        nan_rows += len(frame) - len(valid)
        for cpg, part in valid.groupby("cpg_idx", sort=False):
            y = part.gt_methyl.to_numpy(float); p = part.pred_methyl.to_numpy(float)
            out = sums[int(cpg)]
            out[0] += float(y.sum()); out[1] += float(p.sum()); out[2] += len(part)
    if not sums:
        raise ValueError(f"No valid reference rows for group {group}")
    rows = [(cpg, v[0] / v[2], v[1] / v[2], v[2]) for cpg, v in sums.items()]
    prior = pd.DataFrame(rows, columns=["cpg_idx", "mean_train", "pred_mean_train", "reference_n"]).sort_values("cpg_idx")
    return prior, {"reference_rows": int(prior.reference_n.sum()), "reference_nan_rows": nan_rows}


@dataclass
class Stats:
    n: int = 0
    dropped_nan_rows: int = 0
    sse: dict | None = None
    sae: dict | None = None
    cpg: dict | None = None
    sample: dict | None = None

    def __post_init__(self):
        names = ("prior", "methylprophet", "hybrid_alpha_1", "hybrid_calibrated")
        self.sse = {x: 0.0 for x in names}; self.sae = {x: 0.0 for x in names}
        self.cpg = {x: defaultdict(lambda: np.zeros(6, dtype=float)) for x in names}
        self.sample = {x: defaultdict(lambda: np.zeros(6, dtype=float)) for x in names}
        self.dynamic_cross = self.dynamic_pred_ss = self.dynamic_truth_ss = 0.0

    def add(self, cpg: np.ndarray, sample: np.ndarray, y: np.ndarray, p: np.ndarray, prior: np.ndarray, alpha: float):
        pred_center = np.fromiter((self.pred_mean[int(x)] for x in cpg), dtype=float, count=len(cpg))
        predictions = {
            "prior": prior,
            "methylprophet": p,
            "hybrid_alpha_1": prior + (p - pred_center),
            "hybrid_calibrated": prior + alpha * (p - pred_center),
        }
        self.n += len(y)
        for name, pred in predictions.items():
            err = pred - y
            self.sse[name] += float(np.dot(err, err)); self.sae[name] += float(np.abs(err).sum())
            for key in np.unique(cpg):
                mask = cpg == key; a = self.cpg[name][int(key)]
                a += (mask.sum(), y[mask].sum(), np.dot(y[mask], y[mask]), pred[mask].sum(), np.dot(pred[mask], pred[mask]), np.dot(y[mask], pred[mask]))
            for key in np.unique(sample):
                mask = sample == key; a = self.sample[name][int(key)]
                a += (mask.sum(), y[mask].sum(), pred[mask].sum(), np.dot(y[mask], y[mask]), np.dot(pred[mask], pred[mask]), np.dot(y[mask], pred[mask]))

    def metrics(self) -> dict:
        output = {"n_rows": self.n, "dropped_nan_rows": self.dropped_nan_rows, "models": {}}
        # The cpg accumulator has n, sum_y, sum_y2, sum_p, sum_p2, sum_yp.
        for name in self.sse:
            cpg_values = np.vstack(list(self.cpg[name].values()))
            n, sy, sy2, sp, sp2, syp = cpg_values.T
            ss_y = float(np.sum(sy2 - sy * sy / n)); ss_p = float(np.sum(sp2 - sp * sp / n))
            covariance = float(np.sum(syp - sy * sp / n))
            # SSE after within-CpG centring, computed from sums and total SSE.
            static = float(np.sum((sp - sy) ** 2 / n))
            dynamic_sse = self.sse[name] - static
            samples = np.vstack(list(self.sample[name].values()))
            sn, sy_s, sp_s, sy2_s, sp2_s, syp_s = samples.T
            den = np.sqrt((sn * sy2_s - sy_s * sy_s) * (sn * sp2_s - sp_s * sp_s))
            pcc = np.divide(sn * syp_s - sy_s * sp_s, den, out=np.full_like(den, np.nan), where=den > EPS)
            pcc = pcc[np.isfinite(pcc) & (sn > 1)]
            output["models"][name] = {
                "mse": self.sse[name] / self.n,
                "mae": self.sae[name] / self.n,
                "total_skill_vs_prior": 1.0 - self.sse[name] / self.sse["prior"],
                "dynamic_skill": _safe(ss_y - dynamic_sse, ss_y),
                "mas_pcc": float(np.median(pcc)) if len(pcc) else None,
                "residual_correlation_pooled": _safe(covariance, math.sqrt(max(ss_y, 0.0) * max(ss_p, 0.0))),
                "variance_ratio_truth_over_prediction_within_cpg": _safe(ss_y, ss_p) if ss_p > EPS * max(ss_y, 1.0) else None,
                "n_samples_with_pcc": int(len(pcc)),
            }
        # Correlation of the intended dynamic signal and truth residual from the prior.
        output["dynamic_residual_correlation"] = _safe(
            self.dynamic_cross, math.sqrt(self.dynamic_pred_ss * self.dynamic_truth_ss)
        )
        return output


def _evaluate(path: Path, group: int, prior: pd.DataFrame, fraction: float, seed: int, alpha: float | None, batch_size: int):
    lookup = prior.set_index("cpg_idx")
    calibration, test = Stats(), Stats()
    calibration.pred_mean = test.pred_mean = lookup.pred_mean_train.to_dict()
    for state in (calibration, test):
        state.dynamic_cross = state.dynamic_pred_ss = state.dynamic_truth_ss = 0.0
    dataset = ds.dataset(path, format="parquet")
    missing_cpg = set()
    for batch in dataset.scanner(columns=list(REQUIRED), filter=ds.field("group_idx") == group, batch_size=batch_size).to_batches():
        frame = batch.to_pandas().drop(columns="group_idx")
        valid = frame.dropna()
        bad = len(frame) - len(valid)
        if valid.empty:
            calibration.dropped_nan_rows += bad; test.dropped_nan_rows += bad; continue
        cpg = valid.cpg_idx.to_numpy(np.int64); sample = valid.sample_idx.to_numpy(np.int64)
        covered = np.isin(cpg, lookup.index.to_numpy(np.int64))
        missing_cpg.update(map(int, cpg[~covered]))
        cpg, sample, valid = cpg[covered], sample[covered], valid.iloc[np.flatnonzero(covered)]
        y, p = valid.gt_methyl.to_numpy(float), valid.pred_methyl.to_numpy(float)
        mu = lookup.loc[cpg, "mean_train"].to_numpy(float)
        d = p - lookup.loc[cpg, "pred_mean_train"].to_numpy(float)
        is_cal = _split(sample, fraction, seed)
        for state, mask in ((calibration, is_cal), (test, ~is_cal)):
            state.dropped_nan_rows += bad
            if mask.any():
                state.dynamic_cross += float(np.dot(d[mask], y[mask] - mu[mask]))
                state.dynamic_pred_ss += float(np.dot(d[mask], d[mask]))
                state.dynamic_truth_ss += float(np.dot(y[mask] - mu[mask], y[mask] - mu[mask]))
                if alpha is not None:
                    state.add(cpg[mask], sample[mask], y[mask], p[mask], mu[mask], alpha)
    if missing_cpg:
        raise ValueError(f"Reference prior misses {len(missing_cpg)} evaluation CpGs")
    return calibration, test


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-predictions", required=True)
    p.add_argument("--reference-group", type=int, required=True)
    p.add_argument("--evaluation-predictions", required=True)
    p.add_argument("--evaluation-group", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--calibration-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--batch-size", type=int, default=262_144)
    args = p.parse_args()
    if not 0 < args.calibration_fraction < 1:
        raise ValueError("--calibration-fraction must be strictly between zero and one")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    prior, reference_info = _reference_means(Path(args.reference_predictions), args.reference_group, args.batch_size)
    # First evaluation pass estimates alpha exclusively on calibration samples.
    calibration, _ = _evaluate(Path(args.evaluation_predictions), args.evaluation_group, prior, args.calibration_fraction, args.seed, None, args.batch_size)
    alpha = calibration.dynamic_cross / calibration.dynamic_pred_ss if calibration.dynamic_pred_ss > EPS else 0.0
    calibration, test = _evaluate(Path(args.evaluation_predictions), args.evaluation_group, prior, args.calibration_fraction, args.seed, alpha, args.batch_size)
    report = {
        "command": " ".join(sys.argv), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {"prior_and_prediction_centering": "reference group only (training samples)", "calibration_test_split": "deterministic sample_idx split of evaluation group", "calibration_fraction": args.calibration_fraction, "seed": args.seed},
        "reference": {**reference_info, "n_cpg": len(prior), "cpg_manifest_sha256": _ids_hash(prior.cpg_idx.to_numpy())},
        "alpha_star_calibration": alpha,
        "calibration_diagnostics": {"n_rows": calibration.n, "dynamic_residual_correlation": _safe(calibration.dynamic_cross, math.sqrt(calibration.dynamic_pred_ss * calibration.dynamic_truth_ss)), "dynamic_prediction_ss": calibration.dynamic_pred_ss, "dynamic_truth_residual_ss": calibration.dynamic_truth_ss},
        "test": test.metrics(),
    }
    prior.to_parquet(out / "training_reference_prior_and_prediction_center.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(_json(report), indent=2) + "\n")


if __name__ == "__main__":
    main()
