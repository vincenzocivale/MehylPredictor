#!/usr/bin/env python3
"""Create gene-count-matched random module controls for module-aware pooling."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ with weights, gene_ids, module_ids")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    source = np.load(args.input, allow_pickle=True)
    weights = np.asarray(source["weights"], dtype=np.float32)
    gene_ids = np.asarray(source["gene_ids"]).astype(str)
    module_ids = np.asarray(source["module_ids"]).astype(str)
    transposed = False
    if weights.shape == (len(module_ids), len(gene_ids)):
        weights = weights.T
        transposed = True
    elif weights.shape != (len(gene_ids), len(module_ids)):
        raise ValueError("weights shape does not match gene_ids/module_ids")
    rng = np.random.default_rng(args.seed)
    random_weights = np.zeros_like(weights)
    for module in range(weights.shape[1]):
        members = np.flatnonzero(weights[:, module] != 0)
        if not len(members):
            continue
        chosen = rng.choice(len(gene_ids), size=len(members), replace=False)
        values = weights[members, module]
        rng.shuffle(values)
        random_weights[chosen, module] = values
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        weights=random_weights.T if transposed else random_weights,
        gene_ids=gene_ids,
        module_ids=np.asarray([f"random_matched::{value}" for value in module_ids]),
        source_module_ids=module_ids,
        seed=np.asarray(args.seed),
    )
    print(output)


if __name__ == "__main__":
    main()
