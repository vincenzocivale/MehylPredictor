#!/usr/bin/env python3
"""Data-preparation entrypoint for CpGStatisticsPredictor and RNAMethylationPredictor.

--model cpg_statistics: build multi-technology CpG mean/sigma labels for
chr1, chr123 or genome-wide (former scripts/prepare_statistics.py).

--model rna_methylation: export a trained CpG-statistics checkpoint's
predicted mu/sigma + NTv3 embeddings into the cache the RNA model consumes
(former scripts/export_statistics_cache.py) - i.e. the second-stage step that
must run after a cpg_statistics prepare+train cycle.
"""
from __future__ import annotations
import argparse, json
from methylation_predictor.cpg_statistics.targets import build_statistics_targets
from methylation_predictor.cpg_statistics.export import export_feature_cache


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["cpg_statistics", "rna_methylation"], required=True)
    p.add_argument("--output", required=True)
    # --model cpg_statistics
    p.add_argument("--canonical-root", help="[cpg_statistics] canonical TCGA bundle root")
    p.add_argument("--registry", help="[cpg_statistics] path to array_cpg_map.parquet")
    p.add_argument("--scope", choices=["chr1", "chr123", "genomewide"], help="[cpg_statistics] genomic scope")
    p.add_argument("--policy", choices=["sample_weighted", "technology_balanced"], default="sample_weighted", help="[cpg_statistics]")
    p.add_argument("--aux-sample-policy", choices=["exclude_array_validation", "all_auxiliary"], default="exclude_array_validation", help="[cpg_statistics]")
    p.add_argument("--sources", nargs="+", default=["array", "epic", "wgbs"], help="[cpg_statistics]")
    p.add_argument("--cpg-chunk", type=int, default=2048, help="[cpg_statistics]")
    # --model rna_methylation
    p.add_argument("--checkpoint", help="[rna_methylation] trained cpg_statistics checkpoint")
    p.add_argument("--targets", help="[rna_methylation] cpg_statistics targets dir (output of --model cpg_statistics)")
    p.add_argument("--embeddings", help="[rna_methylation] NTv3 embeddings HDF5")
    p.add_argument("--predicted-for-train", action="store_true", help="[rna_methylation] use model predictions instead of empirical leakage-safe statistics on official train CpGs")
    args = p.parse_args()

    if args.model == "cpg_statistics":
        missing = [name for name in ("canonical_root", "registry", "scope") if getattr(args, name) is None]
        if missing:
            p.error(f"--model cpg_statistics requires: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        result = build_statistics_targets(
            canonical_root=args.canonical_root,
            registry=args.registry,
            scope=args.scope,
            output=args.output,
            policy=args.policy,
            aux_sample_policy=args.aux_sample_policy,
            sources=tuple(args.sources),
            cpg_chunk=args.cpg_chunk,
        )
    else:
        missing = [name for name in ("checkpoint", "targets", "embeddings") if getattr(args, name) is None]
        if missing:
            p.error(f"--model rna_methylation requires: {', '.join('--' + m for m in missing)}")
        result = export_feature_cache(
            checkpoint=args.checkpoint,
            targets_dir=args.targets,
            embeddings_h5=args.embeddings,
            output=args.output,
            empirical_for_train=not args.predicted_for_train,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
