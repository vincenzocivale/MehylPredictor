#!/usr/bin/env python3
"""Predict methylation at a set of (chrom, pos) loci with DeepCpG's DNA-only submodel --
patient-agnostic (DNA sequence only, no per-sample input), see `deepcpg_adapter.py`'s
module docstring for why only the two DNA-only variants are used and why CpG-neighbor/Joint
modules are out of scope.

Each released checkpoint has multiple output heads, one per single cell from its original
scRRBS training dataset (`hou2016_hcc_dna`: 25 heads `cpg/CaNN`; `hou2016_hepg2_dna`: 6 heads
`cpg/HepG2N`) -- there is no single "bulk" output. **Methodological choice, stated explicitly
here rather than left implicit**: this script reports the mean predicted probability across
all of a checkpoint's output heads as the bulk-comparable prediction, the natural analogue of
bulk methylation itself being an average over many cells in a tissue sample. This is not
something the DeepCpG authors specify for bulk use -- it is this pipeline's own choice for
making a patient-agnostic, per-cell model produce one number comparable to bulk beta.

Must run under `external/deepcpg-env` (conda, python=3.7/tensorflow==1.13.1/keras==1.2.2 --
legacy stack, see module docstrings elsewhere in this package for why). Needs only numpy +
pyfaidx + keras/tensorflow -- no h5py/pandas dependency on this repo's own canonical bundle,
so genomic locations are passed in as a small JSON file built beforehand in the main env.

    conda run -n deepcpg-env python scripts/benchmark_foundation_models/run_deepcpg_predict.py \\
        --variant hou2016_hcc_dna \\
        --checkpoints-dir external/checkpoints/deepcpg \\
        --fasta-path /dune/.../reference/hg38/hg38.fa \\
        --locations-json derived/foundation_models/cpgpt/val_locations_chr1_covered.json \\
        --output derived/foundation_models/deepcpg/chr1_hou2016_hcc_dna_pred.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=["hou2016_hcc_dna", "hou2016_hepg2_dna"])
    parser.add_argument("--checkpoints-dir", default=None, help="Defaults to external/checkpoints/deepcpg.")
    parser.add_argument("--fasta-path", required=True)
    parser.add_argument(
        "--locations-json",
        required=True,
        help="JSON with 'chrom' and 'pos' parallel lists (this repo's 1-based, 'chr'-prefixed "
        "convention -- the same shape crosswalk.py writes, e.g. val_locations_<scope>_covered.json).",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else REPO_ROOT / "external" / "checkpoints" / "deepcpg"

    from methylation_predictor.benchmark.foundation_models.adapters import deepcpg_adapter as adapter

    resources = adapter.resolve_resources(checkpoints_dir, args.variant)
    model = adapter.build_random_model(resources.arch_path)
    adapter.load_checkpoint_into(model, resources.weights_path)
    print(f"Loaded DeepCpG '{args.variant}' checkpoint OK -- {len(model.output_names)} output heads.")

    locations = json.loads(Path(args.locations_json).read_text())
    chroms, positions = locations["chrom"], locations["pos"]
    n = len(chroms)

    mean_pred = np.empty(n, dtype=np.float32)
    for b0 in range(0, n, args.batch_size):
        chunk_chrom = chroms[b0 : b0 + args.batch_size]
        chunk_pos = positions[b0 : b0 + args.batch_size]
        windows = np.stack(
            [adapter.one_hot_dna_window(args.fasta_path, c, p) for c, p in zip(chunk_chrom, chunk_pos)]
        )
        out = model.predict(windows, verbose=0)
        stacked = np.stack(out, axis=-1) if isinstance(out, list) else out
        mean_pred[b0 : b0 + len(chunk_chrom)] = stacked.mean(axis=-1).reshape(-1)
        print(f"  predicted {b0 + len(chunk_chrom)}/{n}", end="\r")
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, mean_pred)
    print(f"wrote {n} mean-over-heads predictions -> {args.output} "
          f"(range [{mean_pred.min():.4f}, {mean_pred.max():.4f}])")


if __name__ == "__main__":
    main()
