#!/usr/bin/env python3
"""CLI for extending the frozen NTv3 CpG embedding atlas.

Restored/recovered pipeline -- see `src/methylation_predictor/ntv3_atlas.py` module
docstring for provenance and the exact locked extraction contract.

Typical flow for a new CpG universe (e.g. ENCODE, or a future TCGA scope expansion):

    prepare-universe-from-positions --chrom-position ... --base-embeddings atlas.h5 --output universe.h5
    extract-worker --universe universe.h5 --fasta hg38.fa --output shards/ --rank 0 --world-size 1
    merge --universe universe.h5 --shards shards/ --output merged/

`merged/expanded_cpg_idx.npy` + `expanded_embeddings.f16.npy` then need appending to the
atlas HDF5 (`cpg_idx`/`chrom_code`/`position`/`embedding`/`origin`) -- not automated here,
since that mutates the frozen, version-controlled-by-reference atlas file and should be a
reviewed, explicit step.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import pyarrow  # noqa: F401 -- load before h5py/torch on the target server

from methylation_predictor.ntv3_atlas import (
    extract_ntv3_worker,
    merge_ntv3_shards,
    prepare_missing_universe_from_positions,
    prepare_missing_universe_tcga_protocol,
)


def _print(value):
    print(json.dumps(value, indent=2, default=str), flush=True)


def cmd_prepare_universe_tcga(a):
    _print(prepare_missing_universe_tcga_protocol(a.canonical_root, a.base_embeddings, a.output, protocol=a.protocol, shard_size=a.shard_size))


def cmd_prepare_universe_positions(a):
    frame = pd.read_parquet(a.chrom_position) if a.chrom_position.endswith(".parquet") else pd.read_csv(a.chrom_position, sep=None, engine="python")
    if not {"chrom", "position"}.issubset(frame.columns):
        raise ValueError(f"--chrom-position must have chrom,position columns, got {list(frame.columns)}")
    _print(prepare_missing_universe_from_positions(
        frame["chrom"].to_numpy(), frame["position"].to_numpy(),
        a.base_embeddings, a.output, id_offset=a.id_offset, shard_size=a.shard_size, source_label=a.source_label,
    ))


def cmd_extract(a):
    _print(extract_ntv3_worker(a.universe, a.fasta, a.output, rank=a.rank, world_size=a.world_size,
                               checkpoint=a.checkpoint, length=32768, batch_size=a.batch_size,
                               device=a.device, bf16=not a.no_bf16, storage_dtype=a.storage_dtype,
                               compile_model=a.compile, compile_mode=a.compile_mode))


def cmd_merge(a):
    _print(merge_ntv3_shards(a.universe, a.shards, a.output, storage_dtype=a.storage_dtype))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("prepare-universe-tcga", help="missing EPIC/WGBS aux CpGs for a tcga_mix_* protocol")
    q.add_argument("--canonical-root", required=True); q.add_argument("--base-embeddings", required=True); q.add_argument("--output", required=True)
    q.add_argument("--protocol", default="tcga_mix_chr123"); q.add_argument("--shard-size", type=int, default=25_000)
    q.set_defaults(func=cmd_prepare_universe_tcga)

    q = sub.add_parser("prepare-universe-from-positions", help="missing CpGs from an arbitrary external chrom/position list (e.g. ENCODE)")
    q.add_argument("--chrom-position", required=True, help="CSV/TSV/parquet with chrom,position columns")
    q.add_argument("--base-embeddings", required=True); q.add_argument("--output", required=True)
    q.add_argument("--id-offset", type=int, default=None); q.add_argument("--shard-size", type=int, default=25_000)
    q.add_argument("--source-label", default="external")
    q.set_defaults(func=cmd_prepare_universe_positions)

    q = sub.add_parser("extract-worker")
    q.add_argument("--universe", required=True); q.add_argument("--fasta", required=True); q.add_argument("--output", required=True)
    q.add_argument("--rank", type=int, required=True); q.add_argument("--world-size", type=int, required=True)
    q.add_argument("--checkpoint", default="InstaDeepAI/NTv3_650M_post"); q.add_argument("--batch-size", type=int, default=4)
    q.add_argument("--device", default="cuda"); q.add_argument("--no-bf16", action="store_true")
    q.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    q.add_argument("--compile", action="store_true",
                   help="wrap the model in torch.compile -- measured ~2.4x throughput on this codebase's GPU "
                        "(10.7->~26 cpg/s), costs ~2-3min one-time recompilation per worker process. Default "
                        "mode avoids CUDA graphs (see ntv3_atlas.extract_ntv3_worker docstring for why "
                        "plain max-autotune crashed a real run)")
    q.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    q.set_defaults(func=cmd_extract)

    q = sub.add_parser("merge")
    q.add_argument("--universe", required=True); q.add_argument("--shards", required=True); q.add_argument("--output", required=True)
    q.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    q.set_defaults(func=cmd_merge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
