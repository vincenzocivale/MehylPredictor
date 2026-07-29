#!/usr/bin/env python3
"""Exact row-weighted TCGA locus/cancer/patient variance decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from methylation_predictor.diagnostics.methylprophet.locus_decomposition import discover_layout


def run(args: argparse.Namespace) -> None:
    dataset = ds.dataset(args.predictions, format="parquet")
    group = int(args.group)
    layout = discover_layout(dataset, [group], args.batch_size)[group]
    metadata = pd.read_parquet(args.sample_metadata)
    required = {"sample_idx", "cancer_type"}
    if not required.issubset(metadata.columns) or metadata.sample_idx.duplicated().any():
        raise ValueError("Sample metadata must have unique sample_idx and cancer_type")
    cancer_types = np.array(sorted(metadata.cancer_type.astype(str).unique()))
    cancer_code = pd.Series(
        pd.Categorical(metadata.cancer_type.astype(str), categories=cancer_types).codes,
        index=metadata.sample_idx.astype(np.int64),
    )
    n_cpg, n_cancer = len(layout.cpg_ids), len(cancer_types)
    n = np.zeros(n_cpg * n_cancer, dtype=np.int64)
    sy = np.zeros(n_cpg * n_cancer, dtype=np.float64)
    sy2 = np.zeros(n_cpg * n_cancer, dtype=np.float64)
    seen = np.zeros((n_cpg, len(layout.sample_ids)), dtype=bool)
    dropped_nan = duplicate_rows = observed = 0
    scanner = dataset.scanner(columns=["group_idx", "cpg_idx", "sample_idx", "gt_methyl"], batch_size=args.batch_size)
    for batch in scanner.to_batches():
        raw = batch.to_pandas()
        raw = raw[raw.group_idx == group]
        if raw.empty:
            continue
        valid = raw.dropna()
        dropped_nan += len(raw) - len(valid)
        cpg = valid.cpg_idx.to_numpy(dtype=np.int64)
        sample = valid.sample_idx.to_numpy(dtype=np.int64)
        cp = np.searchsorted(layout.cpg_ids, cpg)
        sp = np.searchsorted(layout.sample_ids, sample)
        if (cp >= n_cpg).any() or (sp >= len(layout.sample_ids)).any():
            raise ValueError("Unexpected released ID")
        duplicated = seen[cp, sp]
        duplicate_rows += int(duplicated.sum())
        if duplicated.any():
            raise ValueError("Duplicate released key; cannot silently choose an observation")
        seen[cp, sp] = True
        ca = cancer_code.reindex(sample).to_numpy()
        if pd.isna(ca).any():
            raise ValueError("Released sample_idx missing from cancer metadata")
        flat = cp * n_cancer + ca.astype(np.int64)
        y = valid.gt_methyl.to_numpy(dtype=np.float64)
        np.add.at(n, flat, 1)
        np.add.at(sy, flat, y)
        np.add.at(sy2, flat, y * y)
        observed += len(y)
    positive = n > 0
    total_n = float(n.sum())
    total_y, total_y2 = sy.sum(), sy2.sum()
    grand = total_y / total_n
    ss_total = float(total_y2 - total_n * grand * grand)
    cpg_n = n.reshape(n_cpg, n_cancer).sum(axis=1)
    cpg_sy = sy.reshape(n_cpg, n_cancer).sum(axis=1)
    ss_locus = float(np.sum(cpg_sy * cpg_sy / cpg_n) - total_n * grand * grand)
    cell_mean_term = float(np.sum(sy[positive] * sy[positive] / n[positive]))
    ss_cancer_given_locus = cell_mean_term - float(np.sum(cpg_sy * cpg_sy / cpg_n))
    ss_within_cancer = float(total_y2 - cell_mean_term)
    result = {
        "group_idx": group,
        "n_rows": int(total_n),
        "n_cpg": n_cpg,
        "n_samples": len(layout.sample_ids),
        "n_cancer_types_present": int((n.reshape(n_cpg, n_cancer).sum(axis=0) > 0).sum()),
        "dropped_nan_rows": dropped_nan,
        "duplicate_rows": duplicate_rows,
        "ss_total": ss_total,
        "ss_locus": ss_locus,
        "ss_cancer_given_locus": ss_cancer_given_locus,
        "ss_within_cancer": ss_within_cancer,
        "f_locus": ss_locus / ss_total,
        "f_cancer_given_locus": ss_cancer_given_locus / ss_total,
        "f_within_cancer": ss_within_cancer / ss_total,
        "closure": ss_total - ss_locus - ss_cancer_given_locus - ss_within_cancer,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--sample-metadata", required=True)
    parser.add_argument("--group", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=262_144)
    run(parser.parse_args())
