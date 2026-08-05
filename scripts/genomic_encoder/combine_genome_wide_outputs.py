#!/usr/bin/env python3
"""Fase 5 prep: combine Fase 4's 22 per-chromosome targets/features/embeddings
into single genome-wide files for prior-probe/variability-probe.

This only concatenates already-computed small per-chromosome parquet/npz
files -- no me.parquet re-read, so none of the memory risk that forced
Fase 4's per-chromosome design applies here."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch", type=Path, default=Path("/data/dataset/methylation/genomic_encoder_genome_wide_scratch"))
    p.add_argument("--representation", default="centre", help="which pooled vector to keep in the combined embeddings npz")
    args = p.parse_args()
    scratch = args.scratch

    print("[combine] targets...", flush=True)
    targets = pd.concat(
        [pd.read_parquet(scratch / "targets_per_chromosome" / f"targets_{c}.parquet") for c in CHROMOSOMES],
        ignore_index=True,
    )
    targets.to_parquet(scratch / "genome_wide_targets.parquet", index=False)
    print(f"  {len(targets)} rows", flush=True)

    print("[combine] features...", flush=True)
    features = pd.concat(
        [pd.read_parquet(scratch / f"features_{c}.parquet") for c in CHROMOSOMES],
        ignore_index=True,
    )
    features.to_parquet(scratch / "genome_wide_features.parquet", index=False)
    print(f"  {len(features)} rows, split counts: {features.split.value_counts().to_dict()}", flush=True)

    print(f"[combine] embeddings ({args.representation})...", flush=True)
    cpg_idx, position, vectors = [], [], []
    for c in CHROMOSOMES:
        with np.load(scratch / f"embeddings_{c}" / "NTv3_650M_post_L32768_forward.npz") as d:
            cpg_idx.append(d["cpg_idx"])
            position.append(d["position"])
            vectors.append(d[args.representation])
    combined = {
        "cpg_idx": np.concatenate(cpg_idx),
        "position": np.concatenate(position),
        args.representation: np.concatenate(vectors),
    }
    if len(set(combined["cpg_idx"].tolist())) != len(combined["cpg_idx"]):
        raise ValueError("duplicate cpg_idx across chromosome embedding files")
    np.savez_compressed(scratch / f"genome_wide_embeddings_{args.representation}.npz", **combined)
    print(f"  {len(combined['cpg_idx'])} CpGs, vector shape {combined[args.representation].shape}", flush=True)

    if set(targets.cpg_idx) != set(combined["cpg_idx"].tolist()):
        raise ValueError("targets and embeddings cover different CpG sets")
    if set(features.cpg_idx) != set(combined["cpg_idx"].tolist()):
        raise ValueError("features and embeddings cover different CpG sets")
    print("[combine] done, all three sources cover the same CpG set", flush=True)


if __name__ == "__main__":
    main()
