"""Materialize a compact YAML ablation grid into independent run configs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def materialize(grid_path: str | Path, output_dir: str | Path) -> list[Path]:
    grid_path = Path(grid_path)
    raw = yaml.safe_load(grid_path.read_text())
    base_path = (grid_path.parent / raw["base_config"]).resolve()
    base = yaml.safe_load(base_path.read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, run in enumerate(raw["runs"]):
        name = run["name"]
        override = run.get("override", {})
        config = _deep_merge(base, override)
        config["run_name"] = name
        root = Path(config.get("output_dir", "artifacts/rna_branch/screening"))
        config["output_dir"] = str(root.parent / name if root.name == "default" else root / name)
        path = output_dir / f"{index:02d}_{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths.append(path)
    (output_dir / "manifest.txt").write_text("\n".join(str(path) for path in paths) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for path in materialize(args.grid, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
