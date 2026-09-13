#!/usr/bin/env python3
"""Run chromosome-chunked TCGA RNA methylation inference from the canonical store."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from methylation_predictor.rna_training.inference import infer_tcga_genomewide


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--recipe", default="configs/models/main.yaml")
    p.add_argument("--locus-store", required=True)
    p.add_argument("--rna-cache", required=True)
    p.add_argument("--output", required=True, help="output directory for per-chromosome matrices")
    p.add_argument("--sample-ids", help=".npy array of canonical RNA sample IDs")
    p.add_argument("--sample-id", action="append", type=int, default=[])
    p.add_argument("--chromosomes", nargs="+", default=["all"])
    p.add_argument("--source", choices=["array", "epic", "wgbs"], help="restrict loci to one TCGA measurement technology")
    p.add_argument("--sample-chunk", type=int, default=32)
    p.add_argument("--locus-chunk", type=int, default=2048)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.sample_ids:
        ids = np.load(args.sample_ids)
        if args.sample_id:
            ids = np.concatenate((np.asarray(ids, dtype=np.int64), np.asarray(args.sample_id, dtype=np.int64)))
    elif args.sample_id:
        ids = np.asarray(args.sample_id, dtype=np.int64)
    else:
        p.error("provide --sample-ids or at least one --sample-id")
    result = infer_tcga_genomewide(checkpoint=args.checkpoint, recipe=args.recipe, locus_store=args.locus_store, rna_cache=args.rna_cache, output=args.output, sample_ids=ids, chromosomes=tuple(args.chromosomes), source=args.source, sample_chunk=args.sample_chunk, locus_chunk=args.locus_chunk, device=args.device, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
