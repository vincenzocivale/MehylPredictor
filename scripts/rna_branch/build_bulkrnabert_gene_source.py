#!/usr/bin/env python3
"""Build a BulkRNABert-alignment-only RNA gene matrix from the broader, unfiltered TCGA
gene-expression source, instead of the mean/std-filtered 21,792-gene matrix used by every
other RNA-branch experiment (F2/C0/E2/Stage B/C/F/T). The broader source has ~100% overlap
with BulkRNABert's 19,062-gene checkpoint list vs ~78% for the filtered matrix.

This matrix is used only to align genes for BulkRNABert extraction; it must not be wired
into the production RNA branch or any other experiment. Values stay raw log2(TPM+1) --
no z-score or train-only transform is applied here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from methylation_predictor.rna_branch.prepare_inputs import MP_DATA, STR_DTYPE, _str_array, build_sample_universe

BROADER_GENE_EXPR = MP_DATA / "parquet/241231-tcga_array/gene_expr.parquet"
GENE_ID_COLUMN = "Unnamed: 0"


def _normalise_gene_id(value: object) -> str:
    fields = [field.strip() for field in str(value).split(";") if field.strip()]
    ensembl = next((field for field in fields if field.upper().startswith("ENSG")), None)
    if ensembl is not None:
        return ensembl.split(".", 1)[0]
    value = fields[0] if fields else str(value)
    return value.split(".", 1)[0] if value.upper().startswith("ENSG") else value


def dedup_genes_tpm_space(
    raw_ids: np.ndarray, values_log2p1: np.ndarray, atol: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, dict]:
    """``values_log2p1`` is [genes, samples] log2(TPM+1). Collapses rows sharing the same
    versionless Ensembl ID: identical duplicate rows keep one copy; differing duplicate
    rows are aggregated in TPM space (sum of ``2**x - 1``) before re-applying
    ``log2(x + 1)`` -- summing the log-scale values directly would not reconstruct the
    correct combined transcript abundance and would break the TPM-sum-to-1e6 identity.
    """
    stable_ids = np.asarray([_normalise_gene_id(v) for v in raw_ids], dtype=object)
    tpm = np.maximum(np.exp2(np.clip(values_log2p1, 0.0, 30.0)) - 1.0, 0.0).astype(np.float64)

    groups: dict[str, list[int]] = {}
    for index, gene in enumerate(stable_ids):
        groups.setdefault(gene, []).append(index)

    ordered_genes = sorted(groups)
    resolved_tpm = np.empty((len(ordered_genes), tpm.shape[1]), dtype=np.float64)
    kept_identical: list[str] = []
    summed_tpm_space: list[str] = []
    for out_index, gene in enumerate(ordered_genes):
        indices = groups[gene]
        if len(indices) == 1:
            resolved_tpm[out_index] = tpm[indices[0]]
            continue
        rows = tpm[indices]
        if np.allclose(rows, rows[0], atol=atol):
            resolved_tpm[out_index] = rows[0]
            kept_identical.append(gene)
        else:
            resolved_tpm[out_index] = rows.sum(axis=0)
            summed_tpm_space.append(gene)

    resolved_log2p1 = np.log2(resolved_tpm + 1.0).astype(np.float32)
    policy = {
        "duplicate_stable_ids": sorted(kept_identical + summed_tpm_space),
        "resolved_by_keep_identical_row": sorted(kept_identical),
        "resolved_by_tpm_space_sum": sorted(summed_tpm_space),
    }
    return resolved_log2p1, np.asarray(ordered_genes, dtype=object), policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--broader-parquet", default=str(BROADER_GENE_EXPR))
    parser.add_argument("--gene-id-column", default=GENE_ID_COLUMN)
    parser.add_argument(
        "--official-repo",
        help="optional checkout of instadeepai/multiomics-open-research; if given, the "
        "sidecar reports overlap against common_gene_id.txt for this matrix",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    sample_universe = build_sample_universe()
    names = sample_universe.sample_name.tolist()

    frame = pd.read_parquet(args.broader_parquet)
    missing_samples = sorted(set(names) - set(frame.columns))
    if missing_samples:
        raise ValueError(
            f"{len(missing_samples)} samples missing from broader gene expression matrix: {missing_samples[:5]}"
        )
    frame = frame[[args.gene_id_column] + names]

    raw_ids = frame[args.gene_id_column].astype(str).to_numpy()
    values_log2p1 = frame[names].to_numpy(dtype=np.float64)  # genes x samples
    resolved_log2p1, gene_ids, dedup_policy = dedup_genes_tpm_space(raw_ids, values_log2p1)

    sample_idx = sample_universe.sample_idx.to_numpy(dtype=np.int64)
    X = resolved_log2p1.T.astype(np.float32)  # samples x genes

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.create_dataset("X", data=X)
        handle.create_dataset("sample_idx", data=_str_array(sample_idx), dtype=STR_DTYPE)
        handle.create_dataset("gene_ids", data=_str_array(gene_ids), dtype=STR_DTYPE)

    sidecar: dict[str, object] = {
        "duplicate_stable_ids": dedup_policy["duplicate_stable_ids"],
        "duplicate_policy": {
            "resolved_by_keep_identical_row": dedup_policy["resolved_by_keep_identical_row"],
            "resolved_by_tpm_space_sum": dedup_policy["resolved_by_tpm_space_sum"],
        },
        "samples_expected": int(len(names)),
        "samples_exported": int(X.shape[0]),
        "missing_samples": len(missing_samples),
        "genes_exported": int(X.shape[1]),
        "source_parquet": str(args.broader_parquet),
        "gene_source": "unfiltered_array_platform",
    }
    if args.official_repo:
        common_genes_path = Path(args.official_repo).resolve() / "data" / "bulkrnabert" / "common_gene_id.txt"
        common_genes = [line.strip() for line in common_genes_path.read_text().splitlines() if line.strip()]
        gene_id_set = set(gene_ids.tolist())
        matched = sum(1 for gene in common_genes if gene in gene_id_set)
        sidecar["checkpoint_genes"] = len(common_genes)
        sidecar["matched_genes"] = matched

    output.with_suffix(output.suffix + ".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    print(json.dumps(sidecar, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
