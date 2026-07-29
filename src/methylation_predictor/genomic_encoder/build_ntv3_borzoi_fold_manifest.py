#!/usr/bin/env python3
"""Intersect Experiment-4 hg38 CpGs with the official Borzoi human intervals."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methylation_predictor.genomic_encoder.ntv3_prior_common import sha256


def _assign(loci: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    required = {"chrom", "start", "end", "fold", "split"}
    if missing := required - set(intervals): raise ValueError(f"interval table lacks {sorted(missing)}")
    result = []
    for chrom, part in loci.groupby("chromosome", sort=False):
        table = intervals[intervals.chrom == chrom].sort_values("start").reset_index(drop=True)
        starts, ends = table.start.to_numpy(np.int64), table.end.to_numpy(np.int64)
        for row in part.itertuples(index=False):
            pos0 = int(row.position) - 1; candidates = np.flatnonzero((starts <= pos0) & (pos0 < ends))
            if len(candidates) == 0:
                result.append((row.cpg_idx, None, None, 0, "not_in_borzoi_interval", "not_covered"))
            else:
                labels = table.iloc[candidates][["fold", "split"]].drop_duplicates()
                # A locus can lie in overlapping 524-kb contigs from different
                # folds. For post-training exposure it is train-exposed if *any*
                # overlapping source contig is train; otherwise it is held out.
                folds = ",".join(sorted(labels.fold.astype(str)))
                splits = ",".join(sorted(labels.split.astype(str)))
                exposure = "train_exposed" if "train" in set(labels.split) else "held_out_from_train"
                result.append((row.cpg_idx, folds, splits, len(candidates), "assigned", exposure))
    return pd.DataFrame(result, columns=["cpg_idx", "borzoi_folds", "borzoi_splits", "n_overlapping_intervals", "assignment_status", "posttraining_exposure"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--intervals", required=True); p.add_argument("--output", required=True)
    p.add_argument("--source-revision", required=True, help="immutable source commit/revision of interval artifact")
    a = p.parse_args(); source, intervals_path, out = Path(a.input), Path(a.intervals), Path(a.output)
    loci = pd.read_parquet(source, columns=["cpg_idx", "chromosome", "position"])
    if loci.cpg_idx.duplicated().any(): raise ValueError("input CpG IDs are not unique")
    intervals = pd.read_csv(intervals_path, sep="\t")
    assigned = _assign(loci, intervals)
    out.parent.mkdir(parents=True, exist_ok=True); assigned.to_parquet(out, index=False)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv), "genome_build": "hg38",
                "coordinate_convention": "input CpG C is 1-based; intervals are BED-like 0-based half-open", "input": str(source), "input_sha256": sha256(source),
                "intervals": str(intervals_path), "intervals_sha256": sha256(intervals_path), "source_revision": a.source_revision, "output": str(out),
                "n_cpg": len(assigned), "assignment_counts": {str(k): int(v) for k, v in assigned.assignment_status.value_counts().items()},
                "exposure_counts": {str(k): int(v) for k, v in assigned.posttraining_exposure.value_counts(dropna=False).items()}}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
