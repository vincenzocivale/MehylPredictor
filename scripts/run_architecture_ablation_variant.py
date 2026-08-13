#!/usr/bin/env python3
"""Train one full-coverage E2 architecture-ablation variant."""
from __future__ import annotations

import argparse
import json

import pyarrow  # noqa: F401 -- load before h5py/torch on the target host

from methylation_predictor.full_suite.ablation_runner import ArchitectureAblationRunner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-root", required=True)
    p.add_argument("--base-config", required=True)
    p.add_argument("--base-cache", required=True)
    p.add_argument("--expanded-embeddings", required=True)
    p.add_argument("--expanded-features", required=True)
    p.add_argument("--rna-cache", required=True)
    p.add_argument("--array-cache", required=True)
    p.add_argument("--epic-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--holdout-policy", choices=("mp_matched", "strict_global"), default="mp_matched")
    p.add_argument("--array-rows", type=int, default=128)
    p.add_argument("--array-cpgs", type=int, default=2048)
    p.add_argument("--epic-rows", type=int, default=128)
    p.add_argument("--epic-cpgs", type=int, default=4096)
    p.add_argument("--wgbs-rows", type=int, default=32)
    p.add_argument("--wgbs-cpgs", type=int, default=16384)
    args = p.parse_args()

    runner = ArchitectureAblationRunner(
        canonical_root=args.canonical_root,
        protocol_name="tcga_mix_chr1",
        base_config=args.base_config,
        base_feature_cache=args.base_cache,
        expanded_embedding_dir=args.expanded_embeddings,
        expanded_feature_dir=args.expanded_features,
        rna_cache_dir=args.rna_cache,
        output_dir=args.output,
        source_policy="equal_source",
        holdout_policy=args.holdout_policy,
        array_cache=args.array_cache,
        epic_cache=args.epic_cache,
        seed=args.seed,
        mixed_steps_per_epoch=128,
        block_rows={"array": args.array_rows, "epic": args.epic_rows, "wgbs": args.wgbs_rows},
        block_cpgs={"array": args.array_cpgs, "epic": args.epic_cpgs, "wgbs": args.wgbs_cpgs},
    )
    try:
        print(json.dumps(runner.run(), indent=2, default=str), flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
