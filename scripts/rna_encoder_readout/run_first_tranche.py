#!/usr/bin/env python3
"""Run the baseline, P0 and P1 RNA-only readout tranche end to end."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="configs/rna_encoder_readout/bulkrnabert_layers1_3.yaml")
    parser.add_argument("--config-root", default="artifacts/rna_encoder_readout/configs")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    base_path = Path(args.base)
    payload = yaml.safe_load(base_path.read_text())
    baseline_run = payload["run_name"]
    baseline_dir = Path(payload["output_dir"]) / baseline_run
    checkpoint = baseline_dir / "best.pt"

    _run([sys.executable, "-m", "methylation_predictor.rna_encoder_readout", "validate", "--config", str(base_path)])
    if not checkpoint.exists():
        _run([sys.executable, "-m", "methylation_predictor.rna_encoder_readout", "train", "--config", str(base_path)])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    for stage in ("p0", "p1"):
        directory = Path(args.config_root) / stage
        _run([
            sys.executable,
            "scripts/rna_encoder_readout/make_readout_configs.py",
            "--base", str(base_path),
            "--stage", stage,
            "--seed", str(args.seed),
            "--warm-start-checkpoint", str(checkpoint),
            "--output-dir", str(directory),
        ])
        _run([
            sys.executable,
            "scripts/rna_encoder_readout/run_readout_screen.py",
            "--config-dir", str(directory),
            "--stop-on-error",
        ])

    _run([
        sys.executable,
        "scripts/rna_encoder_readout/aggregate_readout_results.py",
        "--root", str(payload["output_dir"]),
        "--output", "artifacts/rna_encoder_readout/readout_screen.csv",
        "--baseline-run", "p0_mean_layer2",
        "--baseline-run", "p0_mean_resume_layer2",
    ])


if __name__ == "__main__":
    main()
