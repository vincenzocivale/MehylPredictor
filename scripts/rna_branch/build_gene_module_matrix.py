#!/usr/bin/env python3
"""Build an RNA-aligned pathway/regulon/module matrix for model.encoder.kind=module_tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.data import MatrixStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--membership", required=True, help="TSV: module_id, gene_id, optional weight")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-genes", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    store = MatrixStore(config.data.rna)
    try:
        if store.col_ids is None:
            raise ValueError("RNA matrix requires col_ids_key")
        gene_ids = store.col_ids.astype(str)
    finally:
        store.close()
    gene_to_index = {gene: index for index, gene in enumerate(gene_ids)}
    frame = pd.read_csv(args.membership, sep="\t")
    if not {"module_id", "gene_id"}.issubset(frame.columns):
        raise ValueError("membership TSV requires module_id and gene_id columns")
    if "weight" not in frame:
        frame["weight"] = 1.0
    frame["module_id"] = frame["module_id"].astype(str)
    frame["gene_id"] = frame["gene_id"].astype(str)
    frame = frame[frame["gene_id"].isin(gene_to_index)].copy()
    counts = frame.groupby("module_id")["gene_id"].nunique()
    modules = sorted(counts[counts >= args.min_genes].index.tolist())
    module_to_index = {module: index for index, module in enumerate(modules)}
    weights = np.zeros((len(modules), len(gene_ids)), dtype=np.float32)
    for row in frame.itertuples(index=False):
        if row.module_id in module_to_index:
            weights[module_to_index[row.module_id], gene_to_index[row.gene_id]] += float(row.weight)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, weights=weights, module_ids=np.asarray(modules), gene_ids=gene_ids)
    report = {
        "modules": len(modules),
        "genes": len(gene_ids),
        "nonzero_memberships": int(np.count_nonzero(weights)),
        "min_genes": args.min_genes,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
