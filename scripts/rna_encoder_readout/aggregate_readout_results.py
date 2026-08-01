#!/usr/bin/env python3
"""Aggregate RNA readout manifests into a validation-only selection table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--baseline-run",
        action="append",
        default=[],
        help="run name used for paired validation deltas; may be repeated",
    )
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.root).rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        row = {
            "run_name": manifest.get("run_name", path.parent.name),
            "best_epoch": manifest.get("best_epoch"),
            "input_target_gene_overlap": manifest.get("input_target_gene_overlap"),
            "manifest": str(path),
            "model_kind": manifest.get("model_kind"),
            "num_queries": manifest.get("num_queries"),
            "initial_readout_vs_mean_max_abs": manifest.get("initial_readout_vs_mean_max_abs"),
            "trainable_parameters": manifest.get("trainable_parameters"),
            "pooler_parameters": manifest.get("pooler_parameters"),
            "decoder_parameters": manifest.get("decoder_parameters"),
        }
        for split, metrics in manifest.get("metrics", {}).items():
            for key, value in metrics.items():
                row[f"{split}_{key}"] = value
        for key, value in manifest.get("diagnostics", {}).items():
            if isinstance(value, dict):
                for statistic, number in value.items():
                    row[f"diagnostic_{key}_{statistic}"] = number
        rows.append(row)
    if not rows:
        raise SystemExit("no manifest.json files found")
    frame = pd.DataFrame(rows)
    for baseline_name in args.baseline_run:
        matches = frame[frame["run_name"] == baseline_name]
        if len(matches) != 1:
            raise SystemExit(f"baseline {baseline_name!r} matched {len(matches)} runs")
        baseline = matches.iloc[0]
        safe = baseline_name.replace("/", "_").replace(" ", "_")
        for metric in ("validation_within_r2", "validation_total_r2", "validation_effective_rank"):
            if metric in frame and pd.notna(baseline.get(metric)):
                frame[f"delta_{safe}_{metric.removeprefix('validation_')}"] = frame[metric] - float(baseline[metric])
    if "validation_within_r2" in frame:
        frame = frame.sort_values(
            ["validation_within_r2", "validation_total_r2"], ascending=[False, False]
        )
    if args.baseline_run:
        delta_columns = [column for column in frame if column.startswith("delta_") and column.endswith("within_r2")]
        total_columns = [column for column in frame if column.startswith("delta_") and column.endswith("total_r2")]
        if delta_columns:
            frame["passes_within_gain_0p02"] = np.logical_and.reduce(
                [(frame[column] >= 0.02).fillna(False).to_numpy() for column in delta_columns]
            )
        if total_columns:
            frame["passes_total_loss_0p01"] = np.logical_and.reduce(
                [(frame[column] >= -0.01).fillna(False).to_numpy() for column in total_columns]
            )
        if "passes_within_gain_0p02" in frame and "passes_total_loss_0p01" in frame:
            frame["rna_only_advances"] = (
                frame["passes_within_gain_0p02"]
                & frame["passes_total_loss_0p01"]
                & (frame["input_target_gene_overlap"].fillna(1) == 0)
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
