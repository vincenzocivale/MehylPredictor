#!/usr/bin/env python3
"""Explain one sample's RNA-conditioned methylation predictions via Integrated Gradients.

Ranks RNA genes by how much they push a frozen rna_methylation checkpoint's
prediction away from the CpG-only prior, for one sample against a set of CpG
loci. See docs/EXPLAINABILITY.md for the method note.
"""
from __future__ import annotations
import argparse, json


def _parse_cpg_idx(args) -> "list[int]":
    import numpy as np
    if args.cpg_idx:
        return [int(x) for x in args.cpg_idx.split(",")]
    if args.cpg_idx_file:
        return np.load(args.cpg_idx_file).astype(int).tolist() if args.cpg_idx_file.endswith(".npy") else [int(x) for x in open(args.cpg_idx_file)]
    raise SystemExit("one of --cpg-idx or --cpg-idx-file is required")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--canonical-root", required=True)
    p.add_argument("--feature-cache", required=True)
    p.add_argument("--rna-cache", required=True)
    p.add_argument("--sample-idx", type=int, required=True, help="TCGA sample_idx to explain")
    p.add_argument("--cpg-idx", help="comma-separated global CpG ids")
    p.add_argument("--cpg-idx-file", help="alternative to --cpg-idx: .npy or one-id-per-line text file")
    p.add_argument("--auto-top-loci", type=int, default=None,
                    help="narrow --cpg-idx to this many loci by |raw_delta| before running IG (see SampleGeneExplainer.explain_sample)")
    p.add_argument("--steps", type=int, default=50, help="Expected Gradients step count per baseline")
    p.add_argument("--n-baselines", type=int, default=8, help="number of random background RNA samples to average over")
    p.add_argument("--baseline-sample-idx", help="comma-separated sample_idx to use as background instead of a random draw")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the random background sample draw")
    p.add_argument("--top-k", type=int, default=25, help="genes to report")
    p.add_argument("--output", help="write the ranked gene table as CSV here (also printed as JSON)")
    args = p.parse_args()

    from methylation_predictor.explainability.rna_gene_attribution import SampleGeneExplainer

    cpg_idx = _parse_cpg_idx(args)
    explainer = SampleGeneExplainer(
        canonical_root=args.canonical_root, checkpoint=args.checkpoint,
        feature_cache=args.feature_cache, rna_cache=args.rna_cache,
    )
    try:
        baseline_sample_idx = [int(x) for x in args.baseline_sample_idx.split(",")] if args.baseline_sample_idx else None
        attribution, explained_cpg_idx = explainer.explain_sample(
            args.sample_idx, cpg_idx, steps=args.steps, auto_top_loci=args.auto_top_loci,
            n_baselines=args.n_baselines, baseline_sample_idx=baseline_sample_idx, seed=args.seed,
        )
        table = explainer.summarize_genes(attribution, top_k=args.top_k)
        print(json.dumps({
            "sample_idx": args.sample_idx,
            "n_loci_explained": len(explained_cpg_idx),
            "max_convergence_gap": float(attribution.convergence_gap.max()),
            "top_genes": table.to_dict(orient="records"),
        }, indent=2))
        if args.output:
            table.to_csv(args.output, index=False)
    finally:
        explainer.close()


if __name__ == "__main__":
    main()
