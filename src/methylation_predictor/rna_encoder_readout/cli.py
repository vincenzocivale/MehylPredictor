"""CLI for frozen RNA-encoder readout optimisation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .trainer import train, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "train"):
        child = sub.add_parser(command)
        child.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(Path(args.config))
    result = validate(config) if args.command == "validate" else train(config)
    print(json.dumps(result, indent=2, sort_keys=True))
