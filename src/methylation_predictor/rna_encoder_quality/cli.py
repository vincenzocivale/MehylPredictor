"""Command-line interface for methylation-independent RNA encoder audits."""
from __future__ import annotations

import argparse
import json

from .config import load_config
from .runner import run_quality_audit, validate_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate_inputs(load_config(args.config))
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        output = run_quality_audit(args.config)
        print(f"RNA encoder quality report written to {output}")


if __name__ == "__main__":
    main()
