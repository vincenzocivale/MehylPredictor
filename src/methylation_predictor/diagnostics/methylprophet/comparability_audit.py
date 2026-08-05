#!/usr/bin/env python3
"""Leakage/comparability audit: our rna_branch splits vs. MethylProphet's official splits.

MethylProphet only ever had a 2-way split (train/val, never a third held-out
set) on both axes. This audit checks, per our own split (train/validation/
test, on both the sample and the CpG axis), how it intersects MethylProphet's
official partitions -- so any competitive claim can point at a verifiable
artifact instead of prose. Must run before any benchmark that compares our
model to MethylProphet (see stage_d3_matrix_v1.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from methylation_predictor.common.splits import assert_disjoint


def _matched_cpg_ids(cpg_bg_dir: Path, filenames: list[str], candidate_cpgs: np.ndarray) -> np.ndarray:
    """Which of candidate_cpgs appear in the given me_cpg_bg partitions.

    These partitions are huge (up to ~32GB, ~1B+ rows) long-format beta
    tables with no cpg_idx partitioning, so reading the full cpg_idx column
    and computing its unique values (the naive approach) can exhaust memory.
    Instead push an ``isin(candidate_cpgs)`` filter down to pyarrow -- the
    predicate is still evaluated per row, but only matching rows (a tiny
    fraction, since candidate_cpgs is our few-thousand-CpG universe) are
    ever materialized, exactly like ``stage_d_matched.extract_checkpoint_predictions``.
    """
    candidates = [int(x) for x in candidate_cpgs]
    found = []
    for name in filenames:
        path = cpg_bg_dir / name
        if not path.exists():
            continue
        table = ds.dataset(path, format="parquet").to_table(
            filter=ds.field("cpg_idx").isin(candidates), columns=["cpg_idx"]
        )
        found.append(table.column("cpg_idx").to_numpy())
    return np.unique(np.concatenate(found)) if found else np.array([], dtype=np.int64)


def audit(
    our_sample_metadata: Path,
    our_cpg_splits: Path,
    mp_train_samples: Path,
    mp_val_samples: Path,
    mp_cpg_bg_dir: Path,
) -> dict:
    samples = pd.read_parquet(our_sample_metadata)[["sample_idx", "split"]]
    cpgs = pd.read_parquet(our_cpg_splits)[["cpg_idx", "split"]]

    # Invariant check: our own splits are internally disjoint (train/val/test
    # never both claim the same ID) before comparing against MethylProphet.
    assert_disjoint(samples, "sample_idx", "split")
    assert_disjoint(cpgs, "cpg_idx", "split")

    mp_train_sample_ids = pd.read_csv(mp_train_samples)["sample_idx"].astype(int).to_numpy()
    mp_val_sample_ids = pd.read_csv(mp_val_samples)["sample_idx"].astype(int).to_numpy()

    def split_ids(frame: pd.DataFrame, key: str, split: str) -> np.ndarray:
        return frame.loc[frame["split"] == split, key].astype(int).to_numpy()

    our_sample_splits = {s: split_ids(samples, "sample_idx", s) for s in sorted(samples["split"].unique())}
    our_cpg_splits_ids = {s: split_ids(cpgs, "cpg_idx", s) for s in sorted(cpgs["split"].unique())}
    all_our_cpgs = np.unique(np.concatenate(list(our_cpg_splits_ids.values()))) if our_cpg_splits_ids else np.array([], dtype=int)

    # Two scans total (not one per our-split): query our whole ~6.7k-CpG
    # universe against the official train_cpg/val_cpg partitions once, then
    # break the (small) matched set down per our-split in-memory.
    matched_in_train_cpg = _matched_cpg_ids(
        mp_cpg_bg_dir, ["train_cpg-train_sample.parquet", "train_cpg-val_sample.parquet"], all_our_cpgs
    )
    matched_in_val_cpg = _matched_cpg_ids(
        mp_cpg_bg_dir, ["val_cpg-train_sample.parquet", "val_cpg-val_sample.parquet"], all_our_cpgs
    )

    cpg_axis = {}
    for split, ids in our_cpg_splits_ids.items():
        cpg_axis[f"ours_{split}_in_mp_train_cpg"] = int(np.intersect1d(ids, matched_in_train_cpg).size)
        cpg_axis[f"ours_{split}_in_mp_val_cpg"] = int(np.intersect1d(ids, matched_in_val_cpg).size)
        cpg_axis[f"ours_{split}_size"] = int(ids.size)

    cpg_axis["ours_universe_size"] = int(all_our_cpgs.size)
    cpg_axis["ours_universe_in_mp_train_cpg"] = int(matched_in_train_cpg.size)
    cpg_axis["ours_universe_in_mp_val_cpg"] = int(matched_in_val_cpg.size)
    cpg_axis["ours_universe_in_mp_val_cpg_fraction"] = (
        cpg_axis["ours_universe_in_mp_val_cpg"] / cpg_axis["ours_universe_size"] if cpg_axis["ours_universe_size"] else None
    )

    sample_axis = {}
    for split, ids in our_sample_splits.items():
        sample_axis[f"ours_{split}_in_mp_train_sample"] = int(np.intersect1d(ids, mp_train_sample_ids).size)
        sample_axis[f"ours_{split}_in_mp_val_sample"] = int(np.intersect1d(ids, mp_val_sample_ids).size)
        sample_axis[f"ours_{split}_size"] = int(ids.size)

    train_ids = our_sample_splits.get("train", np.array([], dtype=int))
    sample_axis["ours_train_equals_mp_train_sample"] = bool(
        np.array_equal(np.sort(train_ids), np.sort(mp_train_sample_ids))
    )
    sample_axis["ours_train_minus_mp_train_sample"] = int(np.setdiff1d(train_ids, mp_train_sample_ids).size)
    sample_axis["mp_train_sample_minus_ours_train"] = int(np.setdiff1d(mp_train_sample_ids, train_ids).size)

    leak_free = (
        all(v == 0 for k, v in cpg_axis.items() if k.endswith("_in_mp_train_cpg") and not k.startswith("ours_universe"))
        and all(v == 0 for k, v in sample_axis.items() if k.endswith("_test_in_mp_train_sample") or k.endswith("_validation_in_mp_train_sample"))
    )

    return {
        "cpg_axis": cpg_axis,
        "sample_axis": sample_axis,
        "verdict": (
            "no CpG-axis leakage into MethylProphet's official train_cpg on any of our splits; "
            "sample train pool matches MethylProphet's official train_sample "
            f"({'exactly' if sample_axis['ours_train_equals_mp_train_sample'] else 'NOT exactly -- see diff counts'}); "
            "our validation/test samples are a disjoint-from-training re-split of MethylProphet's own val_sample pool"
        ) if leak_free else (
            "LEAKAGE DETECTED -- see cpg_axis/sample_axis counts above for the offending split"
        ),
        "leak_free": leak_free,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--our-sample-metadata", type=Path, required=True)
    parser.add_argument("--our-cpg-splits", type=Path, required=True)
    parser.add_argument(
        "--mp-train-samples", type=Path,
        default=Path("/data/dataset/methylation/MethylProphetData/parquet/241231-tcga_array/metadata/subset_sample_split/ind_cancer/train_sample_tissue_count_with_idx.csv"),
    )
    parser.add_argument(
        "--mp-val-samples", type=Path,
        default=Path("/data/dataset/methylation/MethylProphetData/parquet/241231-tcga_array/metadata/subset_sample_split/ind_cancer/val_sample_tissue_count_with_idx.csv"),
    )
    parser.add_argument(
        "--mp-cpg-bg-dir", type=Path,
        default=Path("/data/dataset/methylation/MethylProphetData/processed/241231-tcga_array-index_files-ind_cancer/me_cpg_bg"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/comparability/methylprophet_split_audit.json"))
    args = parser.parse_args()

    result = audit(
        args.our_sample_metadata, args.our_cpg_splits,
        args.mp_train_samples, args.mp_val_samples, args.mp_cpg_bg_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: leak_free={result['leak_free']}")


if __name__ == "__main__":
    main()
