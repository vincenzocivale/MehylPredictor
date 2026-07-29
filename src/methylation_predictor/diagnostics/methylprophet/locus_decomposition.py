#!/usr/bin/env python3
"""Out-of-core tests of locus dominance in released MethylProphet predictions.

The input is intentionally limited to released evaluation rows.  It never
estimates a training prior from validation rows: a total-skill calculation is
emitted only when an explicit training-prior table is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


REQUIRED_COLUMNS = ("group_idx", "cpg_idx", "sample_idx", "pred_methyl", "gt_methyl")
EPS = 1e-12


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _sha256_ids(values: np.ndarray) -> str:
    payload = "\n".join(map(str, np.sort(values.astype(np.int64)))).encode() + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > EPS else float("nan")


@dataclass
class GroupLayout:
    cpg_ids: np.ndarray
    sample_ids: np.ndarray


@dataclass
class Moments:
    layout: GroupLayout
    n: np.ndarray
    sum_y: np.ndarray
    sum_y2: np.ndarray
    sum_p: np.ndarray
    sum_p2: np.ndarray
    sum_py: np.ndarray
    sse: float = 0.0
    sae: float = 0.0
    observed_rows: int = 0
    dropped_nan_rows: int = 0
    duplicate_rows: int = 0
    prior_sse: float = 0.0
    prior_sae: float = 0.0
    prior_rows: int = 0

    @classmethod
    def empty(cls, layout: GroupLayout) -> "Moments":
        size = len(layout.cpg_ids)
        return cls(
            layout=layout,
            n=np.zeros(size, dtype=np.int64),
            sum_y=np.zeros(size, dtype=np.float64),
            sum_y2=np.zeros(size, dtype=np.float64),
            sum_p=np.zeros(size, dtype=np.float64),
            sum_p2=np.zeros(size, dtype=np.float64),
            sum_py=np.zeros(size, dtype=np.float64),
        )


def discover_layout(dataset: ds.Dataset, groups: Iterable[int], batch_size: int) -> dict[int, GroupLayout]:
    """First streaming pass: enumerate IDs, so aggregate arrays stay compact."""
    wanted = set(groups)
    cpg_ids: dict[int, set[int]] = {g: set() for g in wanted}
    sample_ids: dict[int, set[int]] = {g: set() for g in wanted}
    scanner = dataset.scanner(columns=["group_idx", "cpg_idx", "sample_idx"], batch_size=batch_size)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        for group, part in frame.groupby("group_idx", sort=False):
            group = int(group)
            if group in wanted:
                cpg_ids[group].update(part["cpg_idx"].astype("int64").unique())
                sample_ids[group].update(part["sample_idx"].astype("int64").unique())
    return {
        group: GroupLayout(
            cpg_ids=np.array(sorted(cpg_ids[group]), dtype=np.int64),
            sample_ids=np.array(sorted(sample_ids[group]), dtype=np.int64),
        )
        for group in wanted
    }


def _row_indices(values: np.ndarray, universe: np.ndarray, label: str) -> np.ndarray:
    result = np.searchsorted(universe, values)
    if (result >= len(universe)).any() or not np.array_equal(universe[result], values):
        raise ValueError(f"Unexpected {label} outside layout")
    return result


def accumulate(
    dataset: ds.Dataset,
    layouts: dict[int, GroupLayout],
    batch_size: int,
    prior: pd.DataFrame | None = None,
) -> dict[int, Moments]:
    """Second streaming pass.  A bitmap detects duplicate released prediction keys."""
    moments = {group: Moments.empty(layout) for group, layout in layouts.items()}
    # Each released TCGA group has at most ~56M potential pairs, so this is bounded
    # (one byte/pair) and avoids silently applying a different duplicate policy.
    seen = {
        group: np.zeros((len(layout.cpg_ids), len(layout.sample_ids)), dtype=bool)
        for group, layout in layouts.items()
    }
    prior_means: dict[int, np.ndarray] = {}
    if prior is not None:
        required = {"cpg_idx", "mean_train"}
        if not required.issubset(prior.columns):
            raise ValueError(f"Prior must include {sorted(required)}")
        if prior["cpg_idx"].duplicated().any():
            raise ValueError("Training prior contains duplicate cpg_idx values")
        lookup = pd.Series(prior["mean_train"].to_numpy(dtype=float), index=prior["cpg_idx"].to_numpy(dtype=np.int64))
        for group, layout in layouts.items():
            prior_means[group] = lookup.reindex(layout.cpg_ids).to_numpy(dtype=float)
    scanner = dataset.scanner(columns=list(REQUIRED_COLUMNS), batch_size=batch_size)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        for group, raw in frame.groupby("group_idx", sort=False):
            group = int(group)
            if group not in moments:
                continue
            state = moments[group]
            valid = raw.dropna(subset=list(REQUIRED_COLUMNS))
            state.dropped_nan_rows += len(raw) - len(valid)
            if valid.empty:
                continue
            cpg = valid["cpg_idx"].to_numpy(dtype=np.int64)
            sample = valid["sample_idx"].to_numpy(dtype=np.int64)
            cpg_pos = _row_indices(cpg, state.layout.cpg_ids, "cpg_idx")
            sample_pos = _row_indices(sample, state.layout.sample_ids, "sample_idx")
            duplicate = seen[group][cpg_pos, sample_pos]
            if duplicate.any():
                state.duplicate_rows += int(duplicate.sum())
                raise ValueError(
                    f"Found {state.duplicate_rows} duplicate released keys in group {group}; "
                    "partition and compare values before applying keep-first semantics."
                )
            seen[group][cpg_pos, sample_pos] = True
            y = valid["gt_methyl"].to_numpy(dtype=np.float64)
            p = valid["pred_methyl"].to_numpy(dtype=np.float64)
            np.add.at(state.n, cpg_pos, 1)
            np.add.at(state.sum_y, cpg_pos, y)
            np.add.at(state.sum_y2, cpg_pos, y * y)
            np.add.at(state.sum_p, cpg_pos, p)
            np.add.at(state.sum_p2, cpg_pos, p * p)
            np.add.at(state.sum_py, cpg_pos, p * y)
            error = p - y
            state.sse += float(np.dot(error, error))
            state.sae += float(np.abs(error).sum())
            if prior is not None:
                baseline = prior_means[group][cpg_pos]
                covered = np.isfinite(baseline)
                if covered.any():
                    baseline_error = y[covered] - baseline[covered]
                    state.prior_sse += float(np.dot(baseline_error, baseline_error))
                    state.prior_sae += float(np.abs(baseline_error).sum())
                    state.prior_rows += int(covered.sum())
            state.observed_rows += len(valid)
    return moments


def summarize(state: Moments, prior: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    """Compute exact row-weighted ANOVA and prediction-error decompositions."""
    keep = state.n > 0
    n = state.n[keep].astype(np.float64)
    ysum, y2sum = state.sum_y[keep], state.sum_y2[keep]
    psum, p2sum, pysum = state.sum_p[keep], state.sum_p2[keep], state.sum_py[keep]
    cpg = state.layout.cpg_ids[keep]
    total_n = float(n.sum())
    ybar, pbar = ysum / n, psum / n
    grand_y = ysum.sum() / total_n
    ss_total = float(y2sum.sum() - total_n * grand_y * grand_y)
    ss_locus = float(np.sum(ysum * ysum / n) - total_n * grand_y * grand_y)
    ss_within = float(y2sum.sum() - np.sum(ysum * ysum / n))
    ss_pred_within = float(p2sum.sum() - np.sum(psum * psum / n))
    covariance_within = float(pysum.sum() - np.sum(psum * ysum / n))
    static_sse = float(np.sum((psum - ysum) ** 2 / n))
    dynamic_sse = float(state.sse - static_sse)
    closure = float(state.sse - static_sse - dynamic_sse)
    residual_corr = _safe_ratio(covariance_within, math.sqrt(max(ss_pred_within, 0.0) * max(ss_within, 0.0)))
    dynamic_skill = 1.0 - _safe_ratio(dynamic_sse, ss_within)
    profile = pd.DataFrame(
        {
            "cpg_idx": cpg,
            "n": n.astype(np.int64),
            "gt_mean": ybar,
            "pred_mean": pbar,
            "gt_within_ss": y2sum - ysum * ysum / n,
            "pred_within_ss": p2sum - psum * psum / n,
            "residual_cross_product": pysum - psum * ysum / n,
        }
    )
    output = {
        "n_rows": int(total_n),
        "n_cpg": int(len(profile)),
        "n_samples": int(len(state.layout.sample_ids)),
        "dropped_nan_rows": state.dropped_nan_rows,
        "duplicate_rows": state.duplicate_rows,
        "mse_model": state.sse / total_n,
        "mae_model": state.sae / total_n,
        "ss_total": ss_total,
        "ss_locus": ss_locus,
        "ss_within": ss_within,
        "f_locus": _safe_ratio(ss_locus, ss_total),
        "f_within": _safe_ratio(ss_within, ss_total),
        "prediction_within_variance": ss_pred_within / total_n,
        "residual_correlation_pooled": residual_corr,
        "static_mse": static_sse / total_n,
        "dynamic_mse": dynamic_sse / total_n,
        "mse_decomposition_closure": closure,
        "dynamic_skill": dynamic_skill,
        "mean_prediction_variation_per_cpg": float(np.mean(profile["pred_within_ss"] / profile["n"])),
    }
    if prior is not None:
        required = {"cpg_idx", "mean_train"}
        if not required.issubset(prior.columns):
            raise ValueError(f"Prior must include {sorted(required)}")
        joined = profile.merge(prior[["cpg_idx", "mean_train"]], on="cpg_idx", how="left", validate="one_to_one")
        covered = joined["mean_train"].notna()
        if not covered.all():
            output["total_skill_status"] = "not_computable: missing training means"
            output["prior_missing_cpg"] = int((~covered).sum())
        else:
            prior_sse = float(np.sum(joined["n"] * (joined["gt_mean"] - joined["mean_train"]) ** 2 + joined["gt_within_ss"]))
            if state.prior_rows not in (0, int(total_n)):
                raise AssertionError("Training-prior coverage changed during streaming")
            if state.prior_rows and not np.isclose(prior_sse, state.prior_sse, rtol=0.0, atol=1e-8):
                raise AssertionError("Training-prior SSE did not match streaming calculation")
            output["prior_mse"] = prior_sse / total_n
            output["total_skill"] = 1.0 - state.sse / prior_sse
            if state.prior_rows:
                output["prior_mae"] = state.prior_sae / total_n
                output["mae_model_minus_prior"] = state.sae / total_n - state.prior_sae / total_n
                output["total_mae_skill"] = 1.0 - state.sae / state.prior_sae
            output["total_skill_status"] = "computed_from_explicit_training_prior"
    return output, profile


def file_manifest(prediction_dir: Path, group_mapping: Path | None) -> dict:
    files = sorted(prediction_dir.glob("*.parquet"))
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_files": [{"path": str(p.resolve()), "size_bytes": p.stat().st_size} for p in files],
        "group_mapping": str(group_mapping.resolve()) if group_mapping else None,
    }


def run(args: argparse.Namespace) -> None:
    prediction_dir = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(prediction_dir, format="parquet")
    missing = set(REQUIRED_COLUMNS) - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Prediction dataset missing columns: {sorted(missing)}")
    groups = [int(x) for x in args.groups.split(",")]
    prior = pd.read_parquet(args.training_prior) if args.training_prior else None
    layouts = discover_layout(dataset, groups, args.batch_size)
    states = accumulate(dataset, layouts, args.batch_size, prior)
    mapping = json.loads(Path(args.group_mapping).read_text()) if args.group_mapping else {}
    report = {
        "command": " ".join(sys.argv),
        "manifest": file_manifest(prediction_dir, Path(args.group_mapping) if args.group_mapping else None),
        "groups": {},
        "limitations": [
            "Released prediction rows establish observational patient-dependent output, not causal gene-encoder attribution.",
            "Cancer-conditioned decomposition requires verified sample_idx-to-cancer_type metadata and is intentionally not inferred from prediction rows.",
        ],
    }
    for group, state in states.items():
        metrics, profile = summarize(state, prior)
        scenario = mapping.get(str(group), str(group))
        metrics["scenario"] = Path(scenario).name.replace(".parquet", "")
        metrics["cpg_manifest_sha256"] = _sha256_ids(state.layout.cpg_ids)
        metrics["sample_manifest_sha256"] = _sha256_ids(state.layout.sample_ids)
        report["groups"][str(group)] = {key: _json_number(value) if isinstance(value, float) else value for key, value in metrics.items()}
        profile.to_parquet(output_dir / f"by_cpg_group={group}.parquet", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(report["manifest"], indent=2) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("analyze", nargs="?", help="subcommand retained for an explicit CLI")
    result.add_argument("--predictions", required=True, help="Directory containing released parquet shards")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--groups", default="0,1,2", help="Comma-separated group_idx values")
    result.add_argument("--group-mapping")
    result.add_argument("--training-prior", help="Parquet with cpg_idx and mean_train; never inferred from validation")
    result.add_argument("--batch-size", type=int, default=262_144)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
