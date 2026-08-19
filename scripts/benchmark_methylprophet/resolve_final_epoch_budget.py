#!/usr/bin/env python3
"""Freeze the one-stage Table-5 epoch budget from the completed confirm run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow  # noqa: F401 -- target-host import ordering
import pandas as pd
import yaml

from methylation_predictor.benchmark.methylprophet.trainer import resolve_final_epochs
from methylation_predictor.benchmark.methylprophet.protocol import Table5Protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm-run", required=True)
    p.add_argument("--table5-protocol", required=True)
    p.add_argument("--json-output")
    p.add_argument("--epochs-only", action="store_true")
    args = p.parse_args()

    confirm = Path(args.confirm_run)
    metrics_path = confirm / "development" / "metrics.json"
    config_path = confirm / "dev_config.yaml"
    if not metrics_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"confirm run must contain development/metrics.json and dev_config.yaml: {confirm}"
        )
    metrics = json.loads(metrics_path.read_text())
    best_epoch = int(metrics["best_epoch"])
    raw = yaml.safe_load(config_path.read_text())
    tr = raw["training"]
    sample_manifest = pd.read_parquet(raw["data"]["sample_metadata"]["path"], columns=["split"])
    cpg_manifest = pd.read_parquet(raw["data"]["cpg_splits"]["path"], columns=["split"])
    train_sample_label = tr.get("train_sample_split", "train")
    train_cpg_label = tr.get("train_cpg_split", "train")
    n_confirm_samples = int(sample_manifest["split"].eq(train_sample_label).sum())
    n_confirm_cpgs = int(cpg_manifest["split"].eq(train_cpg_label).sum())
    confirm_steps = max(
        math.ceil(n_confirm_samples / int(tr["sample_batch_size"])),
        math.ceil(n_confirm_cpgs / int(tr["cpg_batch_size"])),
    )

    protocol = Table5Protocol.load(args.table5_protocol)
    counts = {
        "array": (len(protocol.array_train_sample_idx), len(protocol.array_train_cpg_idx), 128, 2048),
        "epic": (1_706, len(protocol.epic_train_cpg_idx), 128, 4096),
        "wgbs": (32, len(protocol.wgbs_train_cpg_idx), 32, 16384),
    }
    # The final trainer covers the complete Cartesian block grid, not max(row,cpg)
    # marginal coverage. Thus every CpG-sample pair slot is visited once/epoch.
    final_source_steps = {
        name: math.ceil(n_rows / row_bs) * math.ceil(n_cpg / cpg_bs)
        for name, (n_rows, n_cpg, row_bs, cpg_bs) in counts.items()
    }
    final_steps = sum(final_source_steps.values())
    epochs = resolve_final_epochs(best_epoch, confirm_steps, final_steps)
    report = {
        "confirm_best_epoch": best_epoch,
        "confirm_train_samples": n_confirm_samples,
        "confirm_train_cpgs": n_confirm_cpgs,
        "confirm_steps_per_epoch": confirm_steps,
        "confirm_optimizer_updates": best_epoch * confirm_steps,
        "final_schedule": "complete Table-5 Cartesian pair coverage",
        "final_steps_per_source": final_source_steps,
        "final_steps_per_epoch": final_steps,
        "resolved_final_epochs": epochs,
        "resolved_optimizer_updates": epochs * final_steps,
        "policy": "nearest integer Table-5 pair-complete epoch preserving confirm-run optimizer-update budget",
    }
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(report, indent=2) + "\n")
    if args.epochs_only:
        print(epochs)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
