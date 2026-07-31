#!/usr/bin/env python3
"""Audit BulkRNABert gene-ID overlap against the current (filtered) and broader (unfiltered)
RNA gene sets, categorize any genes still missing, and flag duplicate stable Ensembl IDs in
the broader source (needed by the TPM-space dedup policy in ``build_bulkrnabert_gene_source.py``).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.data import MatrixStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RNA-branch YAML pointing at the current (filtered) RNA matrix")
    parser.add_argument("--official-repo", required=True, help="checkout of instadeepai/multiomics-open-research")
    parser.add_argument("--broader-parquet", required=True, help="unfiltered genes-as-rows gene_expr.parquet")
    parser.add_argument("--broader-gene-id-column", default="Unnamed: 0")
    parser.add_argument("--output", required=True)
    return parser


def _normalise_gene_id(value: object) -> str:
    fields = [field.strip() for field in str(value).split(";") if field.strip()]
    ensembl = next((field for field in fields if field.upper().startswith("ENSG")), None)
    if ensembl is not None:
        return ensembl.split(".", 1)[0]
    value = fields[0] if fields else str(value)
    return value.split(".", 1)[0] if value.upper().startswith("ENSG") else value


def overlap_report(common_genes: list[str], source_ids: list[str]) -> dict:
    normalised = {_normalise_gene_id(gene) for gene in source_ids}
    matched = [gene for gene in common_genes if gene in normalised]
    missing = [gene for gene in common_genes if gene not in normalised]
    return {
        "checkpoint_genes": len(common_genes),
        "matched_genes": len(matched),
        "overlap": len(matched) / max(len(common_genes), 1),
        "missing_genes": missing,
    }


def categorize_missing(missing: list[str], source_ids: list[str]) -> dict:
    """Split genes missing after exact versionless-ID match into ID-mapping failures
    (a source row's raw label contains the missing ID as a substring, e.g. a symbol/casing
    mismatch upstream of ``_normalise_gene_id``) vs. genuinely absent from the source."""
    raw_by_source = [str(value) for value in source_ids]
    id_mapping_failures = []
    genuinely_absent = []
    for gene in missing:
        if any(gene in raw for raw in raw_by_source):
            id_mapping_failures.append(gene)
        else:
            genuinely_absent.append(gene)
    return {
        "id_mapping_failures": id_mapping_failures,
        "genuinely_absent": genuinely_absent,
    }


def duplicate_stable_ids(source_ids: list[str]) -> dict:
    counts = Counter(_normalise_gene_id(gene) for gene in source_ids)
    duplicates = {gene: count for gene, count in counts.items() if count > 1}
    return {
        "n_duplicate_stable_ids": len(duplicates),
        "duplicate_stable_ids": sorted(duplicates.keys()),
        "max_duplicate_count": max(duplicates.values()) if duplicates else 0,
    }


def main() -> None:
    args = _parser().parse_args()
    repo = Path(args.official_repo).resolve()
    common_genes_path = repo / "data" / "bulkrnabert" / "common_gene_id.txt"
    common_genes = [line.strip() for line in common_genes_path.read_text().splitlines() if line.strip()]

    config = load_config(args.config)
    store = MatrixStore(config.data.rna)
    try:
        if store.col_ids is None:
            raise ValueError("RNA matrix requires gene IDs in col_ids_key")
        current_ids = store.col_ids.astype(str).tolist()
    finally:
        store.close()

    broader_frame = pd.read_parquet(args.broader_parquet)
    broader_ids = broader_frame[args.broader_gene_id_column].astype(str).tolist()

    current = overlap_report(common_genes, current_ids)
    broader = overlap_report(common_genes, broader_ids)
    categorized = categorize_missing(broader["missing_genes"], broader_ids)
    duplicates = duplicate_stable_ids(broader_ids)

    report = {
        "overlap_current": {k: v for k, v in current.items() if k != "missing_genes"},
        "overlap_broader": {k: v for k, v in broader.items() if k != "missing_genes"},
        "missing_genes_categorized": categorized,
        "duplicate_genes_resolved": duplicates,
        "samples_expected": int(broader_frame.shape[1] - 1),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
