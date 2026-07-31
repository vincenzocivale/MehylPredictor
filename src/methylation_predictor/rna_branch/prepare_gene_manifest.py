#!/usr/bin/env python3
"""Align RNA columns to hg38 genes and create a strand-aware TSS manifest.

The output has exactly one row per RNA column.  Unmatched genes are retained with
``matched=False`` so a later merge can assign the shared zero/unknown embedding
without changing the RNA input dimension or the F2 baseline contract.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MatrixConfig
from .data import MatrixStore
from methylation_predictor.genomic_encoder.ntv3_prior_common import sha256

_ATTR = re.compile(r'([A-Za-z0-9_]+)\s+"([^"]*)"')


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")


def _strip_version(value: str) -> str:
    return value.split(".", 1)[0]


def read_gtf_genes(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with _open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attrs = dict(_ATTR.findall(fields[8]))
            gene_id = attrs.get("gene_id")
            if not gene_id:
                continue
            chrom = fields[0]
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            start, end = int(fields[3]), int(fields[4])
            strand = fields[6]
            if strand not in {"+", "-"}:
                continue
            rows.append(
                {
                    "gene_id_gtf": gene_id,
                    "gene_id_normalized": _strip_version(gene_id),
                    "gene_name": attrs.get("gene_name", ""),
                    "chromosome": chrom,
                    "tss": start if strand == "+" else end,
                    "strand": strand,
                    "gene_biotype": attrs.get("gene_type", attrs.get("gene_biotype", "")),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no gene records found in {path}")
    if frame.gene_id_gtf.duplicated().any():
        raise ValueError("GTF contains duplicate gene_id records")
    return frame


def _unique_lookup(frame: pd.DataFrame, column: str) -> dict[str, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(frame[column].fillna("").astype(str)):
        if value:
            grouped[value].append(index)
    return {key: indices[0] for key, indices in grouped.items() if len(indices) == 1}


def _rna_gene_id_candidates(raw: str) -> list[str]:
    """Return ordered identifiers from one RNA feature label.

    TCGA RNA matrices commonly label columns as ``SYMBOL;ENSG...<version>``.
    Retaining both parts lets the stable Ensembl identifier take precedence while
    still supporting symbol-only matrices.  Empty fields are ignored and repeated
    tokens are considered once.
    """
    values = [raw.strip()]
    values.extend(part.strip() for part in raw.split(";"))
    return list(dict.fromkeys(value for value in values if value))


def build_manifest(rna_gene_ids: np.ndarray, genes: pd.DataFrame) -> pd.DataFrame:
    exact = _unique_lookup(genes, "gene_id_gtf")
    normalized = _unique_lookup(genes, "gene_id_normalized")
    names = _unique_lookup(genes, "gene_name")
    records: list[dict[str, object]] = []
    for rna_col, raw in enumerate(rna_gene_ids.astype(str).tolist()):
        identifiers = _rna_gene_id_candidates(raw)
        candidates = (
            [exact.get(value) for value in identifiers]
            + [normalized.get(_strip_version(value)) for value in identifiers]
            + [names.get(value) for value in identifiers]
        )
        index = next((candidate for candidate in candidates if candidate is not None), None)
        if index is None:
            records.append(
                {
                    "rna_col": rna_col,
                    "gene_idx": raw,
                    "matched": False,
                    "gene_id_gtf": "",
                    "gene_name": "",
                    "chromosome": "",
                    "tss": -1,
                    "strand": "",
                    "gene_biotype": "",
                }
            )
        else:
            row = genes.iloc[index]
            records.append(
                {
                    "rna_col": rna_col,
                    "gene_idx": raw,
                    "matched": True,
                    "gene_id_gtf": row.gene_id_gtf,
                    "gene_name": row.gene_name,
                    "chromosome": row.chromosome,
                    "tss": int(row.tss),
                    "strand": row.strand,
                    "gene_biotype": row.gene_biotype,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna", required=True)
    parser.add_argument("--values-key", default="X")
    parser.add_argument("--row-ids-key", default="sample_idx")
    parser.add_argument("--gene-ids-key", default="gene_ids")
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    args = parser.parse_args()

    store = MatrixStore(
        MatrixConfig(args.rna, args.values_key, args.row_ids_key, args.gene_ids_key)
    )
    try:
        if store.col_ids is None:
            raise ValueError("RNA matrix requires gene IDs in col_ids_key")
        rna_gene_ids = store.col_ids.copy()
    finally:
        store.close()

    manifest = build_manifest(rna_gene_ids, read_gtf_genes(Path(args.gtf)))
    coverage = float(manifest.matched.mean())
    if coverage < args.min_coverage:
        missing = manifest.loc[~manifest.matched, "gene_idx"].head(20).tolist()
        raise ValueError(
            f"GTF coverage {coverage:.3%} is below {args.min_coverage:.3%}; "
            f"first unmatched genes={missing}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output, index=False)
    metadata = {
        "rna": args.rna,
        "rna_sha256": sha256(Path(args.rna)),
        "gtf": args.gtf,
        "gtf_sha256": sha256(Path(args.gtf)),
        "genes": len(manifest),
        "matched": int(manifest.matched.sum()),
        "coverage": coverage,
        "matching_order": [
            "exact gene_id (including semicolon-delimited RNA label fields)",
            "version-stripped gene_id",
            "unique gene_name",
        ],
        "output": str(output),
        "output_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metadata)


if __name__ == "__main__":
    main()
