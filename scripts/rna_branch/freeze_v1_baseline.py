#!/usr/bin/env python3
"""Freeze a completed rna_branch matrix-training run as an immutable baseline.

Augments the run's existing ``manifest.json`` (already written by
``ExperimentRunner``/``runtime_manifest``, and already rich: git commit,
input file hashes, environment, seed, best_epoch) with the two pieces the
existing manifest does not carry: exact per-split sample/CpG identity
(as sha256 digests of the sorted ID arrays, same convention as
``stage_d_matched.py::_digest``) and the fitted RNA z-score normalization
statistics. Also copies the split-membership parquet files into the run
directory so the run is self-contained even if the shared inputs directory
changes later.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from methylation_predictor.diagnostics.methylprophet.stage_d_matched import _digest
from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.data import load_bundle
from methylation_predictor.rna_branch.utils import json_safe


def _split_digests(ids: np.ndarray, splits: np.ndarray) -> dict[str, str]:
    return {
        split: _digest(np.sort(ids[splits == split].astype(str)))
        for split in sorted(set(splits.tolist()))
    }


def freeze(checkpoint_dir: Path, tag: str) -> Path:
    config = load_config(checkpoint_dir / "config.yaml")
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())

    bundle = load_bundle(config.data, seed=config.training.seed)

    frozen_inputs = checkpoint_dir / "frozen_inputs"
    frozen_inputs.mkdir(parents=True, exist_ok=True)
    sample_metadata_dst = frozen_inputs / "sample_metadata.parquet"
    cpg_splits_dst = frozen_inputs / "cpg_split_manifest.parquet"
    shutil.copy2(config.data.sample_metadata.path, sample_metadata_dst)
    shutil.copy2(config.data.cpg_splits.path, cpg_splits_dst)

    split_identity = {
        "sample_idx": _split_digests(bundle.samples.ids, bundle.samples.splits),
        "cpg_idx": _split_digests(bundle.loci.ids, bundle.loci.splits),
        "digest_method": "stage_d_matched._digest: sha256 of sorted, str-cast, newline-joined IDs",
    }

    rna_norm_path = None
    standardizer = bundle.rna_transformer.standardizer if bundle.rna_transformer is not None else None
    if standardizer is not None:
        gene_ids = bundle.rna_store.col_ids
        if gene_ids is None:
            raise ValueError("RNA matrix has no col_ids_key; cannot attribute normalization stats to genes")
        stats = pd.DataFrame({
            "gene_id": np.asarray(gene_ids).astype(str),
            "mean": standardizer.mean.astype(float),
            "std": standardizer.scale.astype(float),
        })
        rna_norm_path = checkpoint_dir / "rna_normalization_stats.parquet"
        stats.to_parquet(rna_norm_path, index=False)

    manifest_v1 = dict(manifest)
    manifest_v1["immutable_tag"] = tag
    manifest_v1["split_identity"] = split_identity
    manifest_v1["frozen_inputs"] = {
        "sample_metadata": str(sample_metadata_dst.relative_to(checkpoint_dir)),
        "cpg_split_manifest": str(cpg_splits_dst.relative_to(checkpoint_dir)),
    }
    manifest_v1["rna_normalization_stats"] = (
        str(rna_norm_path.relative_to(checkpoint_dir)) if rna_norm_path is not None else None
    )

    output_path = checkpoint_dir / "manifest_v1.json"
    output_path.write_text(json.dumps(json_safe(manifest_v1), indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tag", default="V1-bilinear-seed17")
    args = parser.parse_args()
    output_path = freeze(args.checkpoint_dir, args.tag)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
