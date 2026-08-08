#!/usr/bin/env python3
"""Carve a nested development/held-out split strictly inside the official
`train` label of the already-prepared genome-wide training inputs, for the
2-stage protocol's development-training stage. Never touches the
existing `validation`/`test` labels, which together are the OFFICIAL
MethylProphet val_cpg/val_sample pool (verified: validation+test CpG counts
sum to exactly 81,493, the official val_cpg count; same pattern on the
sample axis) -- those rows pass through completely unchanged.

Relabels 10% of `train` rows (seed 17, stratified by chromosome for CpGs /
cancer_type for samples) to a new `dev_heldout` label; the remaining 90%
keep the literal label `train`. This "relabel in place" design (rather than
adding a parallel split column) matters because `data.py`'s
`load_bundle()` hardcodes the literal string `"train"` when fitting the RNA
z-score standardizer (`train_rows = np.flatnonzero(sample_splits ==
"train")`) -- pointing the development config at these new files means the
standardizer, reference RNA vector, and variability-tertile thresholds are
automatically and correctly fit on `dev_fit` (=90% train) only, with zero
changes needed to trainer.py/data.py/config.py beyond what full-coverage
training already required.

Caveat documented, not "fixed": `dev_heldout` CpGs' `pred_ntv3_prior`/
variability values in locus_features.parquet were fit via a 5-fold
out-of-fold ensemble over the ORIGINAL full train_cpg pool (which included
these rows). This is standard OOF-ensemble practice (each CpG's own
prediction still comes from a fold that excluded it) -- not full leakage --
but it is a narrower guarantee than "this value was fit with zero knowledge
of this CpG's held-out status in the development stage." Re-fitting the OOF
ensemble for every candidate nested split would be substantial extra
engineering for a second-order effect; this script does not attempt it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEV_HELDOUT_FRACTION = 0.10
DEV_SPLIT_SEED = 17

DEFAULT_INPUT_DIR = Path("/raid/DATASETS/MethylPredictionData")
DEFAULT_FEATURES_PATH = Path("/raid/DATASETS/MethylPredictionData/genome_wide_features.parquet")
DEFAULT_OUTPUT_DIR = Path("/raid/DATASETS/MethylPredictionData/rna_branch_inputs_dev_seed17")


def _assert_disjoint(frame: pd.DataFrame, key: str, split: str) -> None:
    """Assert every value of ``key`` maps to exactly one ``split`` label."""
    counts = frame.groupby(key, dropna=False)[split].nunique()
    if (counts > 1).any():
        raise ValueError(f"{key} occurs in multiple split partitions")


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _stratified_heldout_mask(strata: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Deterministic per-stratum holdout selection, mirroring the
    ceil-rounded, fixed-order-RNG convention already used by
    `data.py::training_sample_pool`/`training_cpg_pool` for nested
    subsampling. Returns a boolean mask (True = held out) over `strata`."""
    n = len(strata)
    heldout = np.zeros(n, dtype=bool)
    rng = np.random.default_rng(seed)
    for stratum in sorted(pd.unique(strata).tolist()):
        group_idx = np.flatnonzero(strata == stratum)
        order = rng.permutation(len(group_idx))
        k = max(1, int(np.ceil(fraction * len(group_idx))))
        heldout[group_idx[order[:k]]] = True
    return heldout


def build_cpg_dev_split(cpg_split_manifest: pd.DataFrame, chromosome_lookup: pd.DataFrame,
                         fraction: float, seed: int) -> tuple[pd.DataFrame, dict]:
    merged = cpg_split_manifest.merge(chromosome_lookup, on="cpg_idx", how="left", validate="one_to_one")
    if merged["chromosome"].isna().any():
        missing = int(merged["chromosome"].isna().sum())
        raise ValueError(f"{missing} cpg_idx values have no chromosome in the lookup table")

    train_mask = merged["split"] == "train"
    heldout_mask = np.zeros(len(merged), dtype=bool)
    heldout_mask[train_mask.to_numpy()] = _stratified_heldout_mask(
        merged.loc[train_mask, "chromosome"].to_numpy(), fraction, seed
    )

    out = cpg_split_manifest.copy()
    out.loc[heldout_mask, "split"] = "dev_heldout"

    before = merged.loc[train_mask].groupby("chromosome").size().to_dict()
    after_heldout = merged.loc[train_mask & heldout_mask].groupby("chromosome").size().to_dict()
    stats = {
        "fraction": fraction,
        "seed": seed,
        "stratify_by": "chromosome",
        "train_before": int(train_mask.sum()),
        "dev_heldout": int(heldout_mask.sum()),
        "train_after": int(train_mask.sum() - heldout_mask.sum()),
        "per_chromosome_train_before": before,
        "per_chromosome_dev_heldout": after_heldout,
    }
    return out, stats


def build_sample_dev_split(sample_metadata: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, dict]:
    train_mask = sample_metadata["split"] == "train"
    heldout_mask = np.zeros(len(sample_metadata), dtype=bool)
    heldout_mask[train_mask.to_numpy()] = _stratified_heldout_mask(
        sample_metadata.loc[train_mask, "cancer_type"].to_numpy(), fraction, seed
    )

    out = sample_metadata.copy()
    out.loc[heldout_mask, "split"] = "dev_heldout"

    before = sample_metadata.loc[train_mask].groupby("cancer_type").size().to_dict()
    after_heldout = sample_metadata.loc[train_mask & heldout_mask].groupby("cancer_type").size().to_dict()
    stats = {
        "fraction": fraction,
        "seed": seed,
        "stratify_by": "cancer_type",
        "train_before": int(train_mask.sum()),
        "dev_heldout": int(heldout_mask.sum()),
        "train_after": int(train_mask.sum() - heldout_mask.sum()),
        "per_cancer_type_train_before": before,
        "per_cancer_type_dev_heldout": after_heldout,
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fraction", type=float, default=DEV_HELDOUT_FRACTION)
    parser.add_argument("--seed", type=int, default=DEV_SPLIT_SEED)
    args = parser.parse_args()

    cpg_split_path = args.input_dir / "cpg_split_manifest.parquet"
    sample_metadata_path = args.input_dir / "sample_metadata.parquet"

    cpg_split_manifest = pd.read_parquet(cpg_split_path)
    sample_metadata = pd.read_parquet(sample_metadata_path)
    chromosome_lookup = pd.read_parquet(args.features_path, columns=["cpg_idx", "chromosome"])

    cpg_dev, cpg_stats = build_cpg_dev_split(cpg_split_manifest, chromosome_lookup, args.fraction, args.seed)
    sample_dev, sample_stats = build_sample_dev_split(sample_metadata, args.fraction, args.seed)

    _assert_disjoint(cpg_dev.rename(columns={"cpg_idx": "key"}), "key", "split")
    _assert_disjoint(sample_dev.rename(columns={"sample_idx": "key"}), "key", "split")
    assert set(cpg_dev["split"].unique()) <= {"train", "dev_heldout", "validation", "test"}
    assert set(sample_dev["split"].unique()) <= {"train", "dev_heldout", "validation", "test"}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpg_dev_path = args.output_dir / "cpg_split_manifest_dev.parquet"
    sample_dev_path = args.output_dir / "sample_metadata_dev.parquet"
    cpg_dev.to_parquet(cpg_dev_path, index=False)
    sample_dev.to_parquet(sample_dev_path, index=False)

    manifest = {
        "fraction": args.fraction,
        "seed": args.seed,
        "source_files": {
            "cpg_split_manifest": {"path": str(cpg_split_path), "sha256": _sha256(cpg_split_path)},
            "sample_metadata": {"path": str(sample_metadata_path), "sha256": _sha256(sample_metadata_path)},
            "genome_wide_features": {"path": str(args.features_path), "sha256": _sha256(args.features_path)},
        },
        "output_files": {
            "cpg_split_manifest_dev": str(cpg_dev_path),
            "sample_metadata_dev": str(sample_dev_path),
        },
        "cpg_axis": cpg_stats,
        "sample_axis": sample_stats,
        "oof_prior_caveat": (
            "dev_heldout CpGs' pred_ntv3_prior/pred_log_var_{between,within} values were fit "
            "via 5-fold out-of-fold MLP ensemble over the ORIGINAL full train_cpg pool (which "
            "included these rows before this relabeling) -- standard OOF practice (each CpG's "
            "own prediction still comes from a fold that excluded it), not full leakage, but not "
            "re-fit against this nested split either. Documented, not corrected -- see module "
            "docstring."
        ),
        "downstream_usage": (
            "The development-training config points data.cpg_splits/data.sample_metadata "
            "at these _dev files, with training.train_{cpg,sample}_split=train (=dev_fit, 90%) "
            "and training.validation_{cpg,sample}_split=dev_heldout (10%). The final-refit stage "
            "uses the ORIGINAL, unmodified cpg_split_manifest.parquet/sample_metadata.parquet "
            "(train=100% of official train_cpg/train_sample) and never reads these _dev files."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
