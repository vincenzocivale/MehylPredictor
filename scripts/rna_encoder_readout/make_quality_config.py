#!/usr/bin/env python3
"""Create an rna_encoder_quality config for a trained patient-level readout."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def _parse_view(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("view must be NAME=PATH")
    return name, path


def _parse_perturbation(value: str) -> tuple[str, str, str | None]:
    fields = value.split("=", 2)
    if len(fields) < 2:
        raise argparse.ArgumentTypeError("perturbation must be NAME=PATH or NAME=PATH=CONTROL")
    return fields[0], fields[1], fields[2] if len(fields) == 3 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-quality-config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stability-view", action="append", default=[], type=_parse_view)
    parser.add_argument("--perturbation-view", action="append", default=[], type=_parse_perturbation)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.base_quality_config).read_text())
    config["run_name"] = args.run_name
    config["output_dir"] = args.output_dir
    config["embeddings"] = {
        "path": args.embeddings,
        "row_ids_key": "sample_idx",
        "layers": {"readout": "embeddings"},
    }
    config.pop("token_embeddings", None)
    config["stability_views"] = [
        {"name": name, "path": path, "row_ids_key": "sample_idx", "layers": {"readout": "embeddings"}}
        for name, path in args.stability_view
    ]
    config["perturbation_views"] = [
        {
            "name": name,
            "path": path,
            "row_ids_key": "sample_idx",
            "layers": {"readout": "embeddings"},
            **({"control": control} if control else {}),
        }
        for name, path, control in args.perturbation_view
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(output)


if __name__ == "__main__":
    main()
