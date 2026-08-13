#!/usr/bin/env python3
"""Train/evaluate the frozen paper architecture on the exact TCGA Table-5 data."""
from __future__ import annotations

import argparse
import json

import pyarrow  # noqa: F401 -- target-host import ordering

from methylation_predictor.final_training import FinalTCGAMixTrainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--derived-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    root = args.derived_root
    trainer = FinalTCGAMixTrainer(
        canonical_root=args.canonical_root,
        config_path=args.config,
        protocol_root=f"{root}/table5_protocol",
        feature_cache=f"{root}/features",
        rna_cache=f"{root}/rna",
        array_cache=f"{root}/methylation/array_table5_chr1.h5",
        epic_cache=f"{root}/methylation/epic_table5_chr1.h5",
        output_dir=args.output,
        epochs=args.epochs,
        seed=args.seed,
    )
    try:
        print(json.dumps({"schedule": trainer.schedule_summary()}, indent=2), flush=True)
        result = trainer.run()
        print(json.dumps(result, indent=2), flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
