#!/usr/bin/env python3
"""Merge NTv3 gene-embedding shards into an RNA-column-aligned NPZ contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from methylation_predictor.genomic_encoder.ntv3_prior_common import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest).sort_values("rna_col").reset_index(drop=True)
    if manifest.rna_col.tolist() != list(range(len(manifest))):
        raise ValueError("manifest rna_col must be contiguous and zero-based")

    collected: dict[int, tuple[str, np.ndarray]] = {}
    embedding_dim: int | None = None
    for raw_path in args.shards:
        path = Path(raw_path)
        with np.load(path, allow_pickle=False) as shard:
            embeddings = np.asarray(shard["embeddings"], dtype=np.float32)
            gene_ids = shard["gene_ids"].astype(str)
            columns = shard["rna_col"].astype(np.int64)
        if not (len(embeddings) == len(gene_ids) == len(columns)):
            raise ValueError(f"inconsistent shard lengths: {path}")
        embedding_dim = embedding_dim or embeddings.shape[1]
        if embeddings.shape[1] != embedding_dim:
            raise ValueError("shards have inconsistent embedding dimensions")
        for column, gene_id, embedding in zip(columns.tolist(), gene_ids.tolist(), embeddings):
            if column in collected:
                raise ValueError(f"duplicate rna_col {column} across shards")
            collected[column] = (gene_id, embedding)

    if embedding_dim is None:
        raise ValueError("no shard embeddings loaded")
    result = np.zeros((len(manifest), embedding_dim), dtype=np.float32)
    for row in manifest.itertuples(index=False):
        if not row.matched:
            continue
        if row.rna_col not in collected:
            raise ValueError(f"matched gene {row.gene_idx!r} is missing from shards")
        gene_id, embedding = collected[row.rna_col]
        if gene_id != str(row.gene_idx):
            raise ValueError(f"gene ID mismatch at RNA column {row.rna_col}: {gene_id!r} != {row.gene_idx!r}")
        result[row.rna_col] = embedding
    if len(collected) != int(manifest.matched.sum()):
        raise ValueError("shards contain rows not represented as matched in the manifest")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=result,
        gene_ids=manifest.gene_idx.astype(str).to_numpy(),
        matched=manifest.matched.to_numpy(bool),
    )
    metadata = {
        "manifest": args.manifest,
        "manifest_sha256": sha256(Path(args.manifest)),
        "shards": [{"path": path, "sha256": sha256(Path(path))} for path in args.shards],
        "genes": len(manifest),
        "matched": int(manifest.matched.sum()),
        "coverage": float(manifest.matched.mean()),
        "embedding_shape": list(result.shape),
        "unmatched_policy": "zero shared unknown embedding; RNA columns retained",
        "output_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metadata)


if __name__ == "__main__":
    main()
