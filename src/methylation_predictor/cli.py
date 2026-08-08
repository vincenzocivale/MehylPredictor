"""Command-line interface for canonical RNA-to-DNAm training."""
from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import load_bundle, summarize_bundle
from .trainer import ExperimentRunner
from .utils import json_safe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate and summarize aligned training inputs"
    )
    validate.add_argument("--config", required=True)

    train = subparsers.add_parser("train", help="train the canonical model")
    train.add_argument("--config", required=True)

    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)

    if args.command == "validate":
        bundle = load_bundle(config.data, seed=config.training.seed)
        try:
            result = summarize_bundle(bundle)
        finally:
            bundle.close()
    elif args.command == "train":
        runner = ExperimentRunner(config)
        try:
            result = runner.train()
        finally:
            runner.close()
    else:  # pragma: no cover
        raise AssertionError(args.command)

    print(json.dumps(json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
