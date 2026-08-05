#!/usr/bin/env python3
"""Genome-wide, vectorized replacement for the chr1-only prior source
(empirical_hybrid.py, derived from MethylProphet's public chr1-only
reproducibility bundle) and for ntv3_variability.py::build_targets' CSV+rg+
per-row-Python loop over a 67GB matrix.

Reads me.parquet (TCGA-array beta matrix, columnar, all 22 chromosomes)
directly and computes, for every requested CpG, mean_train plus the same
three variance decompositions build_targets computes (total/between_cancer/
within_cancer), fully vectorized with numpy instead of a per-CpG Python loop.
At chr1 scale (6,742 CpGs) build_targets' CSV approach matched ~1.6% of the
file's rows; at genome scale (408,399 CpGs) it would match ~100% of rows
through the same single-threaded per-row path -- this script avoids that by
never streaming row-by-row at all.

Output schema is a superset of both downstream contracts:
- cpg_idx/mean_train satisfy static_features.py's --prior contract.
- n/gt_within_ss satisfy static_features.py's --within-stats contract
  (within_cpg_variance = gt_within_ss / n).
- cpg_idx/n_train_nonmissing/n_eligible/n_cancer_types/total_variance/
  between_cancer_variance/within_cancer_variance exactly match
  ntv3_variability.py::build_targets' return frame, so this file can be
  passed straight to `variability-probe --targets` unchanged.

Deviation from build_targets' strict "every requested CpG must be retained"
assertion: at 60x the CpG count, isolated per-probe missingness/eligibility
failures are expected data reality, not necessarily a bug. This script warns
on any drop and only raises if drops exceed 5% (a systematic-bug guard),
rather than requiring exact coverage.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from methylation_predictor.genomic_encoder.static_prior import _hash_u64


def assign_methylprophet_split(
    frame: pd.DataFrame, train_split_path: Path, val_split_path: Path, validation_fraction: float, seed: int
) -> pd.Series:
    """train = MethylProphet's real official train_cpg (verified bit-exact
    against the current chr1 val_cpg universe, see docs/tcga_matrix_model.md);
    validation/test = a stable-hash bisection of official val_cpg -- the only
    cell neither this model nor MethylProphet's checkpoint ever trained on.
    Reuses static_prior._hash_u64 (the same primitive _partition uses) so the
    bisection is deterministic and independent of row order."""
    train_chr_pos = set(pd.read_parquet(train_split_path).chr_pos)
    val_chr_pos = set(pd.read_parquet(val_split_path).chr_pos)
    unknown = set(frame.chr_pos) - train_chr_pos - val_chr_pos
    if unknown:
        raise ValueError(f"{len(unknown)} CpGs are in neither official train_cpg nor val_cpg")
    bucket = _hash_u64(frame.chr_pos.astype(str), seed) % np.uint64(100)
    threshold = np.uint64(round(validation_fraction * 100))
    is_train = frame.chr_pos.isin(train_chr_pos)
    return pd.Series(
        np.where(is_train, "train", np.where(bucket < threshold, "validation", "test")),
        index=frame.index,
    )


def cpg_universe(cpg_map_path: Path, chromosomes: list[str] | None) -> pd.DataFrame:
    """All array-platform CpGs (group_idx == 0 in MethylProphet's cpg_chr_pos_df.parquet),
    optionally restricted to a chromosome subset. cpg_idx = row position in this table --
    the exact numbering used throughout the pipeline (locus embeddings, me_cpg_bg beta
    tables, released predictions), verified bit-exact against the current chr1 universe."""
    big_map = pd.read_parquet(cpg_map_path).reset_index(drop=True)
    big_map["cpg_idx"] = big_map.index
    universe = big_map.loc[big_map.group_idx == 0, ["cpg_idx", "chr", "pos", "chr_pos"]].rename(
        columns={"chr": "chromosome", "pos": "position"}
    )
    if chromosomes:
        universe = universe[universe.chromosome.isin(chromosomes)]
    return universe.reset_index(drop=True)


def _locked_train_samples(train_manifest: Path, sample_metadata: Path) -> pd.DataFrame:
    train_ids = set(pd.read_parquet(train_manifest).sample_idx.astype(int))
    md = pd.read_parquet(sample_metadata).set_index("sample_idx")
    return md.loc[md.index.intersection(train_ids)]


def build(
    universe: pd.DataFrame,
    me_parquet: Path,
    train_manifest: Path,
    sample_metadata: Path,
    minimum: int,
) -> pd.DataFrame:
    md = _locked_train_samples(train_manifest, sample_metadata)
    dataset = ds.dataset(me_parquet, format="parquet")
    available = set(dataset.schema.names)
    names = [n for n in md.sample_name if n in available]
    if len(names) != len(md):
        raise ValueError(f"{len(md) - len(names)} locked train samples missing from {me_parquet}")

    print(f"[targets] reading {me_parquet} for {len(universe)} CpGs x {len(names)} train samples", flush=True)
    wanted_chr_pos = list(universe.chr_pos)
    table = dataset.to_table(columns=["Unnamed: 0", *names], filter=ds.field("Unnamed: 0").isin(wanted_chr_pos))
    frame = table.to_pandas().rename(columns={"Unnamed: 0": "chr_pos"})
    if len(frame) != len(universe):
        raise ValueError(f"matrix rows {len(frame)} != requested CpGs {len(universe)} (duplicate or missing chr_pos)")
    frame = universe.merge(frame, on="chr_pos", validate="one_to_one")

    X = frame[names].to_numpy(dtype=np.float64)  # [n_cpg, n_sample]
    cancer = md.set_index("sample_name").loc[names, "cancer_type"].to_numpy(object)
    print(f"[targets] matrix loaded: {X.shape}, computing statistics", flush=True)

    finite = np.isfinite(X)
    n_train_nonmissing = finite.sum(axis=1)
    safe_n = np.maximum(n_train_nonmissing, 1)
    mean_train = np.where(finite, X, 0).sum(axis=1) / safe_n
    total_variance = np.where(finite, (X - mean_train[:, None]) ** 2, 0).sum(axis=1) / safe_n

    groups = pd.unique(cancer)
    group_index = {g: i for i, g in enumerate(groups)}
    sample_group_idx = np.array([group_index[c] for c in cancer])
    counts = np.zeros((len(frame), len(groups)))
    sums = np.zeros((len(frame), len(groups)))
    for g in range(len(groups)):
        mask = (sample_group_idx == g)[None, :] & finite
        counts[:, g] = mask.sum(axis=1)
        sums[:, g] = np.where(mask, X, 0).sum(axis=1)
    eligible = counts >= minimum
    n_eligible = eligible.sum(axis=1)
    group_means = sums / np.maximum(counts, 1)

    eligible_counts = np.where(eligible, counts, 0)
    total_eligible_n = eligible_counts.sum(axis=1)
    safe_eligible_n = np.maximum(total_eligible_n, 1)
    grand_mean = (eligible_counts * group_means).sum(axis=1) / safe_eligible_n
    between_cancer_variance = (eligible_counts * (group_means - grand_mean[:, None]) ** 2).sum(axis=1) / safe_eligible_n

    eligible_sample_mask = eligible[:, sample_group_idx] & finite  # [n_cpg, n_sample]
    assigned_group_mean = group_means[:, sample_group_idx]
    within_ss = np.where(eligible_sample_mask, (X - assigned_group_mean) ** 2, 0).sum(axis=1)
    within_cancer_variance = within_ss / safe_eligible_n

    valid = n_eligible >= 2
    result = pd.DataFrame({
        "cpg_idx": frame.cpg_idx.to_numpy(),
        "chromosome": frame.chromosome.to_numpy(),
        "position": frame.position.to_numpy(),
        "mean_train": mean_train,
        "n_train_nonmissing": n_train_nonmissing.astype(int),
        # Matches build_targets' exact column semantics (verified against
        # artifacts/genomic_encoder/ntv3_variability_final/.../train_only_variance_targets.parquet):
        # n_eligible = total sample count summed across eligible groups (n.sum()),
        # n_cancer_types = count of eligible groups (len(eligible)) -- NOT swapped.
        "n_eligible": total_eligible_n.astype(int),
        "n_cancer_types": n_eligible.astype(int),
        "total_variance": total_variance,
        "between_cancer_variance": between_cancer_variance,
        "within_cancer_variance": within_cancer_variance,
        "n": total_eligible_n.astype(int),
        "gt_within_ss": within_ss,
    })[valid].reset_index(drop=True)

    dropped = len(frame) - len(result)
    if dropped:
        fraction = dropped / len(frame)
        print(f"[targets] dropped {dropped}/{len(frame)} CpGs ({fraction:.4%}) with <2 eligible cancer types (min {minimum} samples/type)", flush=True)
        if fraction > 0.05:
            raise ValueError(f"dropped fraction {fraction:.4%} exceeds 5% -- likely a bug, not expected attrition")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpg-map", required=True, help="third_party/.../cpg_chr_pos_df.parquet (23M rows, group_idx column)")
    p.add_argument("--me-parquet", required=True, help="TCGA-array wide beta matrix (me.parquet)")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--sample-metadata", required=True)
    p.add_argument("--chromosomes", help="comma-separated chr list, e.g. chr21,chr22; default = all")
    p.add_argument("--minimum-per-cancer-type", type=int, default=10)
    p.add_argument("--official-train-cpg", help="metadata/cpg_split/index_files/train.parquet -- adds a split column if given")
    p.add_argument("--official-val-cpg", help="metadata/cpg_split/index_files/val.parquet -- adds a split column if given")
    p.add_argument("--validation-fraction", type=float, default=0.5, help="fraction of official val_cpg assigned to validation (rest -> test)")
    p.add_argument("--split-seed", type=int, default=20260728)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    chromosomes = args.chromosomes.split(",") if args.chromosomes else None
    universe = cpg_universe(Path(args.cpg_map), chromosomes)
    result = build(
        universe, Path(args.me_parquet), Path(args.train_manifest), Path(args.sample_metadata),
        args.minimum_per_cancer_type,
    )
    if args.official_train_cpg and args.official_val_cpg:
        split = assign_methylprophet_split(
            universe, Path(args.official_train_cpg), Path(args.official_val_cpg), args.validation_fraction, args.split_seed
        )
        result = result.merge(
            pd.DataFrame({"cpg_idx": universe.cpg_idx, "split": split}), on="cpg_idx", how="left", validate="one_to_one"
        )
        if result.split.isna().any():
            raise ValueError("some retained CpGs did not receive a split label")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chromosomes": chromosomes or "all",
        "n_cpg_requested": len(universe),
        "n_cpg_retained": len(result),
        "minimum_per_cancer_type": args.minimum_per_cancer_type,
        "target_definitions": {
            "total": "population variance across all nonmissing locked train samples",
            "between": "sample-count-weighted variance of eligible cancer-type means",
            "within": "sample-count-weighted mean within-cancer population variance",
        },
        "output": str(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
