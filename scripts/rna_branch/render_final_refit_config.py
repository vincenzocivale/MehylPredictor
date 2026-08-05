#!/usr/bin/env python3
"""Render the Stage 3B (final refit) config from the Stage 3A (development)
config plus its measured `best_epoch`, via programmatic YAML edit (not sed)
to avoid fragile text surgery.

Per the 2-stage protocol: same architecture, hyperparameters, preprocessing,
optimizer, schedule, seed, and epoch definition as 3A -- the ONLY changes are
(1) data.sample_metadata/cpg_splits point back at the ORIGINAL, unmodified
manifests (100% of official train_cpg/train_sample, not the 90% dev_fit
pool), (2) epochs/min_epochs are fixed to best_epoch and
checkpoint_selection=final so early stopping is structurally unreachable
(every validation epoch overwrites best.pt unconditionally -- see
trainer.py's `improved` logic), (3) validation_{sample,cpg}_split point at
`train` itself as an in-sample-only sanity curve (never official
validation/test), with validation_every set high enough that it only fires
once, at the final epoch (`epoch == config.training.epochs` already forces
this in trainer.py regardless of validation_every).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ORIGINAL_SAMPLE_METADATA = "/data/dataset/methylation/genomic_encoder_genome_wide_scratch/rna_branch_inputs/sample_metadata.parquet"
ORIGINAL_CPG_SPLITS = "/data/dataset/methylation/genomic_encoder_genome_wide_scratch/rna_branch_inputs/cpg_split_manifest.parquet"


def render(dev_config_path: Path, best_epoch: int, output_path: Path) -> Path:
    raw = yaml.safe_load(dev_config_path.read_text())

    raw["run_name"] = "concat-genomewide-fullcoverage-seed17-v1"
    raw["output_dir"] = "artifacts/rna_branch/concat_genomewide_fullcoverage_seed17_v1/final_refit"

    raw["data"]["sample_metadata"]["path"] = ORIGINAL_SAMPLE_METADATA
    raw["data"]["cpg_splits"]["path"] = ORIGINAL_CPG_SPLITS

    raw["training"]["epochs"] = int(best_epoch)
    raw["training"]["min_epochs"] = int(best_epoch)
    raw["training"]["patience"] = int(best_epoch)  # belt-and-suspenders; checkpoint_selection=final already
    raw["training"]["checkpoint_selection"] = "final"
    raw["training"]["validation_sample_split"] = "train"
    raw["training"]["validation_cpg_split"] = "train"
    # Only fires on the final epoch anyway (trainer.py: `epoch % validation_every
    # == 0 or epoch == epochs`); set high so no in-sample validation forward
    # pass is wasted on intermediate epochs.
    raw["training"]["validation_every"] = 1_000_000

    raw["evaluation"]["panels"] = {}

    raw["tracking"]["name"] = "concat-genomewide-fullcoverage-seed17-final"
    tags = raw["tracking"].get("tags", [])
    raw["tracking"]["tags"] = [t for t in tags if t != "stage-3a-dev"] + ["stage-3b-final-refit"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-config", type=Path, required=True)
    parser.add_argument("--best-epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = render(args.dev_config, args.best_epoch, args.output)
    print(json.dumps({"output": str(output_path), "best_epoch": args.best_epoch}, indent=2))


if __name__ == "__main__":
    main()
