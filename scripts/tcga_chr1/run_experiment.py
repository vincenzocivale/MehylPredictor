#!/usr/bin/env python3
"""Run one controlled TCGA chromosome-1 development experiment from a compact
YAML spec (reference or one-factor ablation).

The public experiment identity is TCGA chr1. The prepared-data cache directory
may still contain ``table5`` in its path because it predates this naming
cleanup; no cache rebuild is required.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pyarrow  # noqa: F401 -- target-host import ordering
import yaml

from methylation_predictor.benchmark.table5.trainer import Table5Trainer

DEFAULT_CANONICAL_ROOT = Path(
    "/raid/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/raid/DATASETS/MethylPredictionData/experiments/MethylPredictor/tcga_chr1"
)
PREPARED_ROOT_CANDIDATES = (
    Path("/raid/DATASETS/MethylPredictionData/derived/tcga_chr1"),
    Path("/raid/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1"),
)


def _deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_prepared_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit)
        if not root.is_dir():
            raise FileNotFoundError(f"prepared TCGA chr1 root does not exist: {root}")
        return root
    for candidate in PREPARED_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "No prepared TCGA chr1 cache found. Checked: "
        + ", ".join(str(p) for p in PREPARED_ROOT_CANDIDATES)
    )


def _load_experiment(path: Path) -> tuple[dict, dict]:
    spec = yaml.safe_load(path.read_text()) or {}
    experiment_id = str(spec.get("experiment_id", "")).strip()
    if not experiment_id or any(ch in experiment_id for ch in "/\\ "):
        raise ValueError("experiment_id must be a non-empty path-safe identifier")
    base_path = (path.parent / spec.get("base_config", "../reference.yaml")).resolve()
    base = yaml.safe_load(base_path.read_text()) or {}
    resolved = _deep_update(base, {"loss": spec.get("loss_overrides", {})})
    return spec, resolved


def _canonical_yaml(payload: dict) -> str:
    return yaml.safe_dump(payload, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment YAML under configs/tcga_chr1/experiments")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=17, help="Development seed; keep 17 until final robustness runs")
    parser.add_argument("--canonical-root", default=str(DEFAULT_CANONICAL_ROOT))
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    experiment_path = Path(args.experiment).resolve()
    spec, resolved = _load_experiment(experiment_path)
    experiment_id = spec["experiment_id"]
    prepared_root = _resolve_prepared_root(args.prepared_root)
    run_dir = Path(args.output_root) / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved["run_name"] = f"tcga-chr1-{experiment_id}"
    resolved["output_dir"] = str(run_dir)
    resolved.setdefault("training", {})["seed"] = args.seed
    tracking = resolved.setdefault("tracking", {})
    tracking["name"] = f"tcga-chr1-{experiment_id}"
    tracking["group"] = "tcga-chr1-development"
    tags = list(tracking.get("tags", []))
    for tag in ("tcga-chr1", experiment_id, f"epochs-{args.epochs}"):
        if tag not in tags:
            tags.append(tag)
    tracking["tags"] = tags

    resolved_text = _canonical_yaml(resolved)
    resolved_path = run_dir / "config.resolved.yaml"
    if resolved_path.exists() and resolved_path.read_text() != resolved_text:
        raise RuntimeError(
            f"{resolved_path} already exists with different content. "
            "Use a new experiment_id instead of silently changing an existing run."
        )
    resolved_path.write_text(resolved_text)

    config_sha256 = hashlib.sha256(resolved_text.encode()).hexdigest()
    layout = spec.get("layout", {})
    block_rows = layout.get("block_rows")
    block_cpgs = layout.get("block_cpgs")
    structured_sources = set(spec.get("structured_loss_sources", ["array", "epic", "wgbs"]))

    experiment_manifest = {
        "benchmark": "tcga_chr1",
        "experiment_id": experiment_id,
        "description": spec.get("description", ""),
        "seed": args.seed,
        "epochs": args.epochs,
        "config_sha256": config_sha256,
        "experiment_spec": str(experiment_path),
        "prepared_root": str(prepared_root),
        "structured_loss_sources": sorted(structured_sources),
        "block_rows": block_rows,
        "block_cpgs": block_cpgs,
    }
    (run_dir / "experiment.json").write_text(
        json.dumps(experiment_manifest, indent=2) + "\n"
    )

    trainer = Table5Trainer(
        canonical_root=args.canonical_root,
        config_path=resolved_path,
        protocol_root=prepared_root / "table5_protocol",
        feature_cache=prepared_root / "features",
        rna_cache=prepared_root / "rna",
        array_cache=prepared_root / "methylation" / "array_table5_chr1.h5",
        epic_cache=prepared_root / "methylation" / "epic_table5_chr1.h5",
        output_dir=run_dir,
        epochs=args.epochs,
        seed=args.seed,
        block_rows=block_rows,
        block_cpgs=block_cpgs,
        structured_loss_sources=structured_sources,
    )
    try:
        print(json.dumps({
            "experiment": experiment_manifest,
            "schedule": trainer.schedule_summary(),
        }, indent=2), flush=True)
        result = trainer.run()
        print(json.dumps(result, indent=2), flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
