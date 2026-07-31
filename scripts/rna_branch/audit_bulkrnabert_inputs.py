#!/usr/bin/env python3
"""Audit the BulkRNABert input-scale assumption and exact tokenization equivalence.

Two independent checks, both required before trusting a BulkRNABert extraction:

1. Scale check: confirms the RNA matrix fed to ``--input-scale log2p1`` really is
   ``log2(TPM+1)`` by verifying ``sum(2**x - 1) == 1_000_000`` per sample (the
   defining arithmetic identity of TPM; FPKM/other units have no such constraint).
2. Exact tokenization check: runs both the production pipeline (as implemented in
   ``extract_bulkrnabert_torch.py``) and the official ``preprocess_omic`` +
   ``BinnedOmicTokenizer`` from the InstaDeep checkout on the same reconstructed
   TPM values, and requires zero token mismatches -- a correlation/similarity
   threshold is not sufficient evidence of pipeline equivalence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.data import MatrixStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RNA-branch YAML used to locate the RNA matrix")
    parser.add_argument("--official-repo", required=True, help="checkout of instadeepai/multiomics-open-research")
    parser.add_argument("--model-name", default="bulk_rna_bert_gtex_encode")
    parser.add_argument("--input-scale", choices=("raw_tpm", "log2p1"), default="log2p1")
    parser.add_argument("--n-scale-samples", type=int, default=200, help="samples used for the TPM-sum check")
    parser.add_argument(
        "--scale-check-parquet",
        help=(
            "optional path to a genes-as-rows, samples-as-columns parquet (e.g. the unfiltered "
            "gene_expr.parquet) used only for the TPM-sum identity check. The sum(2^x-1)==1e6 "
            "identity holds only over the (near-)complete transcriptome: checking it against a "
            "gene-filtered matrix (such as the config's possibly-filtered RNA matrix) will always "
            "read a sum well below 1e6 and is not evidence against the log2(TPM+1) hypothesis."
        ),
    )
    parser.add_argument("--scale-check-gene-id-column", default="Unnamed: 0")
    parser.add_argument("--n-token-samples", type=int, default=20, help="samples used for the exact-token check")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    return parser


def _normalise_gene_id(value: object) -> str:
    fields = [field.strip() for field in str(value).split(";") if field.strip()]
    ensembl = next((field for field in fields if field.upper().startswith("ENSG")), None)
    if ensembl is not None:
        return ensembl.split(".", 1)[0]
    value = fields[0] if fields else str(value)
    return value.split(".", 1)[0] if value.upper().startswith("ENSG") else value


def to_raw_tpm(values: np.ndarray, scale: str) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if scale == "raw_tpm":
        return np.maximum(values, 0.0)
    if scale == "log2p1":
        return np.maximum(np.exp2(np.clip(values, 0.0, 30.0)) - 1.0, 0.0)
    raise ValueError(scale)


def align_to_common_genes(
    values: np.ndarray, source_ids: Iterable[object], common_genes: list[str]
) -> tuple[np.ndarray, float]:
    source = {_normalise_gene_id(gene): index for index, gene in enumerate(source_ids)}
    aligned = np.zeros((values.shape[0], len(common_genes)), dtype=np.float32)
    matched = 0
    for target_index, gene in enumerate(common_genes):
        source_index = source.get(gene)
        if source_index is not None:
            aligned[:, target_index] = values[:, source_index]
            matched += 1
    return aligned, matched / max(len(common_genes), 1)


def load_scale_check_source(path: str, gene_id_column: str, n_samples: int, seed: int) -> np.ndarray:
    """Load a genes-as-rows parquet and return a [samples, genes] log2(TPM+1) array.

    All raw rows are summed as-is (duplicate gene IDs included): the TPM identity is a
    property of the original per-transcript quantification, independent of any later
    gene-ID deduplication or filtering choice made downstream of this source file.
    """
    frame = pd.read_parquet(path)
    frame = frame.drop(columns=[gene_id_column], errors="ignore")
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, frame.shape[1])
    columns = rng.choice(frame.columns.to_numpy(), size=n_samples, replace=False)
    return frame[columns].to_numpy(dtype=np.float32).T


def scale_report(tpm: np.ndarray, clip_source: np.ndarray) -> dict:
    sums = tpm.sum(axis=1).astype(np.float64)
    clipped = (clip_source < 0.0) | (clip_source > 30.0)
    return {
        "n_samples_checked": int(len(sums)),
        "tpm_sum_min": float(sums.min()),
        "tpm_sum_max": float(sums.max()),
        "tpm_sum_median": float(np.median(sums)),
        "tpm_sum_mean_abs_error_from_1e6": float(np.mean(np.abs(sums - 1_000_000.0))),
        "verified_log2_tpm_plus_one": bool(np.max(np.abs(sums - 1_000_000.0)) < 1.0),
        "clip_fraction": float(np.mean(clipped)),
    }


def custom_tokens(tpm_aligned: np.ndarray, checkpoint_config: dict) -> tuple[np.ndarray, np.ndarray]:
    values = np.log10(tpm_aligned + 1.0)
    values /= float(checkpoint_config["normalization_factor"])
    token_ids = np.digitize(values, np.linspace(0.0, 1.0, int(checkpoint_config["n_expressions_bins"])))
    token_ids = token_ids.astype(np.int64)
    token_ids[values == 0.0] = 0
    return token_ids, values


def official_tokens(tpm_aligned: np.ndarray, checkpoint_dir: Path) -> np.ndarray:
    from multiomics_open_research.bulk_rna_bert.config import BulkRNABertConfig  # type: ignore
    from multiomics_open_research.common.preprocess import preprocess_omic  # type: ignore
    from multiomics_open_research.common.tokenizer import BinnedOmicTokenizer  # type: ignore

    config = BulkRNABertConfig.parse_file(checkpoint_dir / "config.json")
    tokenizer = BinnedOmicTokenizer(
        n_expressions_bins=config.n_expressions_bins,
        use_max_normalization=config.use_max_normalization,
        normalization_factor=config.normalization_factor,
        prepend_cls_token=False,
    )
    frame = pd.DataFrame(tpm_aligned)
    processed = preprocess_omic(frame, config)
    return tokenizer.batch_tokenize(processed.copy())


def token_histogram_report(token_ids: np.ndarray, n_bins: int) -> dict:
    counts = np.bincount(token_ids.ravel(), minlength=n_bins)
    total = token_ids.size
    return {
        "n_bins": n_bins,
        "fraction_token_0": float(counts[0] / total),
        "fraction_token_max": float(counts[-1] / total),
        "histogram": counts.tolist(),
    }


def main() -> None:
    args = _parser().parse_args()
    repo = Path(args.official_repo).resolve()
    checkpoint_dir = repo / "checkpoints" / args.model_name
    checkpoint_config = json.loads((checkpoint_dir / "config.json").read_text())
    common_genes_path = repo / "data" / "bulkrnabert" / "common_gene_id.txt"
    common_genes = [line.strip() for line in common_genes_path.read_text().splitlines() if line.strip()]

    config = load_config(args.config)
    store = MatrixStore(config.data.rna)
    try:
        if store.col_ids is None:
            raise ValueError("RNA matrix requires gene IDs in col_ids_key")
        source_genes = store.col_ids.astype(str)
        align_rows = np.sort(
            np.random.default_rng(args.seed).choice(
                store.shape[0], size=min(args.n_token_samples, store.shape[0]), replace=False
            )
        )
        raw_for_alignment = store.rows(align_rows)
    finally:
        store.close()

    if args.scale_check_parquet:
        scale_source = load_scale_check_source(
            args.scale_check_parquet, args.scale_check_gene_id_column, args.n_scale_samples, args.seed
        )
    else:
        scale_source = raw_for_alignment

    scale_tpm = to_raw_tpm(scale_source, args.input_scale)
    clip_source = scale_source if args.input_scale == "log2p1" else np.zeros_like(scale_source)
    scale = scale_report(scale_tpm, clip_source)
    scale["source"] = "scale_check_parquet" if args.scale_check_parquet else "config_rna_matrix"
    if not args.scale_check_parquet:
        scale["warning"] = (
            "checked against the config's RNA matrix, which may be gene-filtered; the "
            "sum(2^x-1)==1e6 identity only holds over the complete transcriptome -- pass "
            "--scale-check-parquet with an unfiltered source for a meaningful pass/fail result"
        )

    tpm = to_raw_tpm(raw_for_alignment, args.input_scale)
    aligned, overlap = align_to_common_genes(tpm, source_genes, common_genes)
    tokens_custom, _ = custom_tokens(aligned, checkpoint_config)
    histogram = token_histogram_report(tokens_custom, int(checkpoint_config["n_expressions_bins"]))
    nonzero_per_sample = (aligned != 0.0).sum(axis=1)

    n_token = min(args.n_token_samples, aligned.shape[0])
    token_check_rows = np.arange(n_token)
    sys.path.insert(0, str(repo))
    tokens_official = official_tokens(aligned[token_check_rows], checkpoint_dir)
    tokens_custom_subset = tokens_custom[token_check_rows]
    mismatch_mask = tokens_official != tokens_custom_subset
    token_mismatch_count = int(mismatch_mask.sum())

    report = {
        "input_scale": args.input_scale,
        "gene_overlap": overlap,
        "scale_check": scale,
        "token_histogram": histogram,
        "nonzero_genes_per_sample": {
            "min": int(nonzero_per_sample.min()),
            "median": float(np.median(nonzero_per_sample)),
            "max": int(nonzero_per_sample.max()),
        },
        "exact_token_check": {
            "n_samples_checked": int(n_token),
            "token_mismatch_count": token_mismatch_count,
            "total_tokens_checked": int(tokens_official.size),
            "passed": token_mismatch_count == 0,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["scale_check"]["verified_log2_tpm_plus_one"]:
        raise SystemExit("scale check failed: sum(2^x-1) is not within tolerance of 1,000,000")
    if not report["exact_token_check"]["passed"]:
        raise SystemExit(f"exact token check failed: {token_mismatch_count} mismatches")


if __name__ == "__main__":
    main()
