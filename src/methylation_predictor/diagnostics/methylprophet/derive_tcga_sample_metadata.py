#!/usr/bin/env python3
"""Derive the upstream TCGA sample_idx/cancer mapping from raw source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def sha256_ids(values: pd.Series) -> str:
    return hashlib.sha256(("\n".join(map(str, sorted(values))) + "\n").encode()).hexdigest()


def derive(gene_csv: Path, project_csv: Path) -> pd.DataFrame:
    # Reading only nrows=0 avoids loading the multi-GB gene-expression matrix.
    gene_columns = pd.read_csv(gene_csv, nrows=0).columns.tolist()
    if not gene_columns or gene_columns[0] != "Unnamed: 0":
        raise ValueError("Expected the raw gene-expression index column 'Unnamed: 0'")
    mapping: dict[str, str] = {}
    with project_csv.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or row == ["Sample", "Cancer"]:
                continue
            if len(row) < 2:
                raise ValueError(f"Malformed project.csv row: {row}")
            cancer = row[-1].strip()
            for sample in row[:-1]:
                # Matches upstream convert_cancer_type_to_parquet: keep first duplicate.
                mapping.setdefault(sample, cancer)
    sample_names = sorted(sample for sample in gene_columns[1:] if sample in mapping)
    if len(sample_names) != len(set(sample_names)):
        raise ValueError("Duplicate gene-expression sample names")
    result = pd.DataFrame({"sample_idx": range(len(sample_names)), "sample_name": sample_names})
    result["cancer_type"] = result["sample_name"].map(mapping)
    if result["cancer_type"].isna().any():
        raise AssertionError("Unknown cancer type after filtering")
    return result


def run(args: argparse.Namespace) -> None:
    result = derive(Path(args.gene_csv), Path(args.project_csv))
    if args.released_sample_ids:
        released = pq.read_table(args.released_sample_ids).to_pandas()["sample_idx"]
        unknown = released[~released.isin(result.sample_idx)]
        if not unknown.empty:
            raise ValueError(f"{len(unknown)} released sample_idx values are absent from raw mapping")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    summary = {
        "gene_csv": str(Path(args.gene_csv).resolve()),
        "project_csv": str(Path(args.project_csv).resolve()),
        "n_samples": len(result),
        "sample_idx_min": int(result.sample_idx.min()),
        "sample_idx_max": int(result.sample_idx.max()),
        "n_cancer_types": int(result.cancer_type.nunique()),
        "sample_idx_sha256": sha256_ids(result.sample_idx),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-csv", required=True)
    parser.add_argument("--project-csv", required=True)
    parser.add_argument("--released-sample-ids", help="Parquet manifest used to validate sample_idx coverage")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
