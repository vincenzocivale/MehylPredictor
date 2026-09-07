#!/usr/bin/env python3
"""Build the .arrow input `CpGPTDataSaver` (external/CpGPT) needs for chr1's
`val_cpg_x_train_sample` masked-recovery view: real Array beta values for the 8,260
train-pool samples at the 33,885 chr1 train CpGs (the "known" context each sample gives
CpGPT), column-named by CpGPT's own genomic-location convention (`"chrom:pos"`, 0-based,
unprefixed -- see `crosswalk.IlluminaCrosswalk`).

Deliberately does NOT include the 6,742 held-out val CpGs -- those are what
`trainer.predict(predict_mode="reconstruct", genomic_locations=...)` (external/CpGPT's own
API, see `external/CpGPT/tutorials/quick_setup.ipynb` section 5.3/5.4) is asked to
*reconstruct*, not part of the known input.

Needs only this repo's own `methyl-predictor` conda env (h5py/numpy/pandas/pyarrow) --
reads through `TCGACanonicalBundle`, which `external/cpgpt-env` cannot do directly (no
h5py installed there as of 2026-09-07). The next pipeline step (`CpGPTDataSaver.
process_files`, mapping these locations to DNA embeddings and CpGPT's own on-disk
mmap format) runs from the *output* of this script, under `external/cpgpt-env`'s python,
and needs no further access to the canonical bundle.

    conda activate methyl-predictor
    python scripts/benchmark_foundation_models/prepare_cpgpt_chr1_context.py \\
        --canonical-root /dune/.../methylprophet_repro_v1 \\
        --table5-protocol-root /dune/.../methylprophet_table5_tcga_chr1/table5_protocol \\
        --cpgpt-human-dependencies external/checkpoints/cpgpt_human_dependencies \\
        --output derived/foundation_models/cpgpt/chr1_train_context.arrow
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from methylation_predictor.benchmark.foundation_models.crosswalk import IlluminaCrosswalk  # noqa: E402
from methylation_predictor.tcga_canonical.bundle import TCGACanonicalBundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--table5-protocol-root", required=True)
    parser.add_argument("--array-cpg-registry", required=True)
    parser.add_argument("--cpgpt-human-dependencies", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-chunk", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_cpg_idx = np.load(Path(args.table5_protocol_root) / "array_train_cpg_idx.npy")
    train_sample_idx = np.load(Path(args.table5_protocol_root) / "array_train_sample_idx.npy")

    registry = pd.read_parquet(args.array_cpg_registry, columns=["cpg_idx", "chr", "pos"]).set_index("cpg_idx")
    rows = registry.loc[train_cpg_idx]
    positions = list(zip(rows["chr"].tolist(), rows["pos"].astype(int).tolist()))

    crosswalk = IlluminaCrosswalk.load(args.cpgpt_human_dependencies)
    coverage = crosswalk.coverage(positions)
    print(f"train CpG DNA-embedding coverage: {coverage['n_dna_embedding_hits']}/{coverage['n_query']} "
          f"({coverage['dna_embedding_coverage']:.4%})")

    covered_mask = np.array(
        [crosswalk.dna_embedding_row(chrom, pos) is not None for chrom, pos in positions]
    )
    kept_cpg_idx = train_cpg_idx[covered_mask]
    kept_positions = [p for p, keep in zip(positions, covered_mask) if keep]
    location_columns = [crosswalk.to_location_key(chrom, pos) for chrom, pos in kept_positions]
    if len(set(location_columns)) != len(location_columns):
        raise ValueError("Duplicate location keys among kept train CpGs -- registry/crosswalk mismatch.")

    bundle = TCGACanonicalBundle.from_root(args.canonical_root)
    try:
        source = bundle.sources["array"]
        n_samples, n_cpgs = len(train_sample_idx), len(kept_cpg_idx)
        beta = np.empty((n_samples, n_cpgs), dtype=np.float32)
        for s0 in range(0, n_samples, args.sample_chunk):
            sample_chunk_idx = train_sample_idx[s0 : s0 + args.sample_chunk]
            row_positions = source.rows_of_samples(sample_chunk_idx)
            beta[s0 : s0 + len(sample_chunk_idx)] = source.block(row_positions, kept_cpg_idx)
            print(f"  read samples {s0}..{s0 + len(sample_chunk_idx)}/{n_samples}", end="\r")
        print()
    finally:
        bundle.close()

    df = pd.DataFrame(beta, columns=location_columns)
    df.insert(0, "sample_idx", [str(int(s)) for s in train_sample_idx])
    df = df.set_index("sample_idx")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_feather(args.output)
    print(f"wrote {df.shape[0]} samples x {df.shape[1]} CpGs -> {args.output}")


if __name__ == "__main__":
    main()
