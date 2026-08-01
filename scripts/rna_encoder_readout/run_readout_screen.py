#!/usr/bin/env python3
"""Validate and train a directory of RNA readout YAML configurations."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--pattern", default="*.yaml")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    configs = sorted(Path(args.config_dir).glob(args.pattern))
    if not configs:
        raise SystemExit("no configurations found")
    failures = []
    for config in configs:
        print(f"\n=== {config.name} ===", flush=True)
        commands = [
            [sys.executable, "-m", "methylation_predictor.rna_encoder_readout", "validate", "--config", str(config)],
            [sys.executable, "-m", "methylation_predictor.rna_encoder_readout", "train", "--config", str(config)],
        ]
        for command in commands:
            result = subprocess.run(command, check=False)
            if result.returncode:
                failures.append((config.name, command[-3], result.returncode))
                if args.stop_on_error:
                    raise SystemExit(result.returncode)
                break
    if failures:
        print("failures:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
