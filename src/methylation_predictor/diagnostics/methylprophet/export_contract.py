"""Canonical, validated export from diagnostics to genomic-encoder evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from methylation_predictor.diagnostics.methylprophet.upstream import assert_clean

REQUIRED = {
    "cpg_train_prior.parquet": {"cpg_idx", "chromosome", "position", "mean_train", "within_cpg_variance", "between_cancer_variance", "within_cancer_variance", "n_train"},
    "mp_dynamic_component.parquet": {"cpg_idx", "sample_idx", "target_beta", "mp_prediction", "mp_train_prediction_mean", "mp_dynamic_component", "split"},
    "mp_mean_rna_prior.parquet": {"cpg_idx", "mp_mean_rna_prediction"},
    "sample_metadata.parquet": {"sample_idx"},
    "cpg_split_manifest.parquet": {"cpg_idx"},
}


def _copy_checked(source: Path, destination: Path, required: set[str]) -> int:
    frame = pd.read_parquet(source)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} lacks required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{source} is empty")
    frame.to_parquet(destination, index=False)
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpg-train-prior", type=Path, required=True)
    parser.add_argument("--mp-dynamic-component", type=Path, required=True)
    parser.add_argument("--mp-mean-rna-prior", type=Path, required=True)
    parser.add_argument("--sample-metadata", type=Path, required=True)
    parser.add_argument("--cpg-split-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-metrics", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "cpg_train_prior.parquet": args.cpg_train_prior,
        "mp_dynamic_component.parquet": args.mp_dynamic_component,
        "mp_mean_rna_prior.parquet": args.mp_mean_rna_prior,
        "sample_metadata.parquet": args.sample_metadata,
        "cpg_split_manifest.parquet": args.cpg_split_manifest,
    }
    rows = {name: _copy_checked(path, args.output_dir / name, REQUIRED[name]) for name, path in inputs.items()}
    metrics = json.loads(args.diagnostic_metrics.read_text())
    (args.output_dir / "diagnostic_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    manifest = {"upstream_commit": assert_clean(), "rows": rows, "contract_version": 1}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
