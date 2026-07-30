#!/usr/bin/env python3
"""Aggregate signal-grid run metrics.json files into one comparison table (mean +/- std across seeds)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

METRIC_KEYS = [
    "mse", "mae", "prior_mse", "skill_vs_prior", "dynamic_skill",
    "dynamic_pearson", "dynamic_spearman",
    "patient_dynamic_pearson_median", "patient_dynamic_spearman_median",
    "locus_dynamic_pearson_median", "locus_dynamic_spearman_median",
    "within_cancer_skill",
    "within_cancer_pearson", "within_cancer_spearman",
    "dynamic_calibration_alpha", "dynamic_amplitude_ratio",
    "sample_win_fraction", "cpg_win_fraction",
    "macro_cancer_mse", "macro_cancer_skill_vs_prior",
]

SEED_SUFFIX = re.compile(r"_seed(\d+)$")


def load_runs(screening_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(screening_dir.iterdir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        data = json.loads(metrics_path.read_text())
        name = run_dir.name
        match = SEED_SUFFIX.search(name)
        seed = int(match.group(1)) if match else None
        family = name[: match.start()] if match else name
        for panel, metrics in data["panels"].items():
            row = {"family": family, "run": name, "seed": seed, "panel": panel}
            for key in METRIC_KEYS:
                row[key] = metrics.get(key)
            for tertile, tertile_metrics in metrics.get("per_variability_tertile", {}).items():
                for key in ("mse", "skill_vs_prior", "dynamic_pearson",
                            "patient_dynamic_pearson_median", "locus_dynamic_pearson_median"):
                    row[f"variability_{tertile}_{key}"] = tertile_metrics.get(key)
            row["encoder_kind"] = data.get("encoder_kind")
            row["num_parameters"] = data.get("num_parameters")
            row["num_encoder_parameters"] = data.get("num_encoder_parameters")
            row["num_interaction_parameters"] = data.get("num_interaction_parameters")
            row["num_gate_parameters"] = data.get("num_gate_parameters")
            row["elapsed_seconds"] = data.get("elapsed_seconds")
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    extra = [
        "num_parameters", "num_encoder_parameters", "num_interaction_parameters",
        "num_gate_parameters", "elapsed_seconds",
    ]
    tertile = [column for column in df.columns if column.startswith("variability_")]
    grouped = df.groupby(["family", "panel"])[METRIC_KEYS + tertile + extra]
    mean = grouped.mean()
    std = grouped.std()
    n = grouped.size().rename("n_seeds")
    summary = mean.join(std, lsuffix="_mean", rsuffix="_std").join(n)
    return summary.reset_index()


PAIRED_METRICS = [
    "mse", "skill_vs_prior", "dynamic_skill", "within_cancer_skill",
    "patient_dynamic_pearson_median", "locus_dynamic_pearson_median",
]


def paired_comparison(
    df: pd.DataFrame, baseline_family: str, metrics: list[str] = PAIRED_METRICS
) -> pd.DataFrame:
    """Per-seed diffs (family - baseline_family) matched on seed, plus a paired t-test.

    With typically 3 seeds the t-test has weak power; treat the p-value as
    descriptive, not as a standalone go/no-go gate -- compare diff_mean against
    diff_std (seed-to-seed variability) as the primary signal.
    """
    baseline = df[df["family"] == baseline_family]
    rows = []
    for (family, panel), group in df.groupby(["family", "panel"]):
        if family == baseline_family:
            continue
        base_panel = baseline[baseline["panel"] == panel]
        merged = group.merge(base_panel, on="seed", suffixes=("", "_baseline"))
        if merged.empty:
            continue
        row: dict[str, object] = {
            "family": family,
            "panel": panel,
            "baseline_family": baseline_family,
            "n_paired_seeds": len(merged),
        }
        for metric in metrics:
            diffs = merged[metric] - merged[f"{metric}_baseline"]
            row[f"{metric}_diff_mean"] = float(diffs.mean())
            row[f"{metric}_diff_std"] = float(diffs.std()) if len(diffs) > 1 else float("nan")
            if len(diffs) >= 2 and diffs.std() > 0:
                _, p_value = ttest_rel(merged[metric], merged[f"{metric}_baseline"])
                row[f"{metric}_paired_p"] = float(p_value)
            else:
                row[f"{metric}_paired_p"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--baseline-family",
        default=None,
        help="if set, also write <output>.paired.csv with per-seed diffs vs this family",
    )
    args = parser.parse_args()
    df = load_runs(Path(args.screening_dir))
    df.to_csv(Path(args.output).with_suffix(".raw.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(args.output, index=False)
    print(summary.to_string())
    if args.baseline_family:
        paired = paired_comparison(df, args.baseline_family)
        paired_path = Path(args.output).with_suffix("").with_suffix(".paired.csv")
        paired.to_csv(paired_path, index=False)
        print(paired.to_string())


if __name__ == "__main__":
    main()
