#!/usr/bin/env python3
"""Canonical inference entrypoint for paper-facing models."""
from __future__ import annotations
import argparse, json
import numpy as np

from methylation_predictor.rna_training.inference import infer_tcga_genomewide


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["rna_methylation"], required=True)
    p.add_argument("--dataset", choices=["tcga"], required=True)
    p.add_argument("--scope", choices=["genomewide"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--recipe", default="configs/models/main.yaml")
    p.add_argument("--canonical-root", help="accepted for CLI symmetry; not read by locus inference")
    p.add_argument("--rna-cache", required=True)
    p.add_argument("--locus-store", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-ids", required=True)
    p.add_argument("--sample-chunk", type=int, default=32)
    p.add_argument("--cpg-chunk", type=int, default=4096)
    p.add_argument("--chromosomes", nargs="+", default=["all"])
    p.add_argument("--source", choices=["array", "epic", "wgbs"])
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true", help="reuse complete compatible chromosome outputs")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    ids = np.load(args.sample_ids)
    result = infer_tcga_genomewide(
        checkpoint=args.checkpoint, recipe=args.recipe, locus_store=args.locus_store,
        rna_cache=args.rna_cache, output=args.output, sample_ids=ids,
        chromosomes=tuple(args.chromosomes), sample_chunk=args.sample_chunk,
        source=args.source, locus_chunk=args.cpg_chunk, device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
