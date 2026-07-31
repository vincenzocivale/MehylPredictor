#!/usr/bin/env python3
"""Materialize frozen BulkRNABert sample embeddings for RNA-branch experiments.

The clean primary comparison uses ``bulk_rna_bert_gtex_encode``.  Models whose
pretraining corpus includes TCGA are blocked by default because the methylation
benchmark samples are TCGA patients and the transfer estimate would otherwise be
contaminated by pretraining overlap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.data import MatrixStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RNA-branch YAML used to locate the RNA matrix")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("official_jax", "huggingface"), default="official_jax")
    parser.add_argument(
        "--model-name",
        default="bulk_rna_bert_gtex_encode",
        choices=("bulk_rna_bert_tcga", "bulk_rna_bert_gtex_encode", "bulk_rna_bert_gtex_encode_tcga"),
    )
    parser.add_argument("--official-repo", help="checkout of instadeepai/multiomics-open-research")
    parser.add_argument("--checkpoint-directory", default="checkpoints/bulkrnabert")
    parser.add_argument("--input-scale", choices=("raw_tpm", "log2p1", "log10p1"), default="log2p1")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
        help="inference activation precision; bfloat16 is required for dense attention on 12 GB GPUs",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-gene-overlap", type=float, default=0.80)
    parser.add_argument("--allow-tcga-pretraining-overlap", action="store_true")
    return parser


def _normalise_gene_id(value: object) -> str:
    fields = [field.strip() for field in str(value).split(";") if field.strip()]
    # Production RNA labels are ``SYMBOL;ENSG.version``. BulkRNABert's
    # common-gene manifest uses versionless Ensembl IDs, so prefer that field.
    ensembl = next((field for field in fields if field.upper().startswith("ENSG")), None)
    if ensembl is not None:
        return ensembl.split(".", 1)[0]
    value = fields[0] if fields else str(value)
    return value.split(".", 1)[0] if value.upper().startswith("ENSG") else value


def _to_raw_tpm(values: np.ndarray, scale: str) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if scale == "raw_tpm":
        return np.maximum(values, 0.0)
    if scale == "log2p1":
        return np.maximum(np.exp2(np.clip(values, 0.0, 30.0)) - 1.0, 0.0)
    if scale == "log10p1":
        return np.maximum(np.power(10.0, np.clip(values, 0.0, 12.0)) - 1.0, 0.0)
    raise ValueError(scale)


def _align(values: np.ndarray, source_ids: Iterable[object], target_ids: Iterable[object]) -> tuple[np.ndarray, float]:
    source = {_normalise_gene_id(gene): index for index, gene in enumerate(source_ids)}
    target = [_normalise_gene_id(gene) for gene in target_ids]
    aligned = np.zeros((values.shape[0], len(target)), dtype=np.float32)
    matched = 0
    for target_index, gene in enumerate(target):
        source_index = source.get(gene)
        if source_index is not None:
            aligned[:, target_index] = values[:, source_index]
            matched += 1
    return aligned, matched / max(len(target), 1)


def _write(output: Path, values: np.ndarray, sample_ids: np.ndarray, metadata: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output, "w") as handle:
        handle.create_dataset("embeddings", data=np.asarray(values, dtype=np.float32))
        handle.create_dataset("sample_idx", data=np.asarray(sample_ids, dtype=object), dtype=strings)
        handle.create_dataset(
            "feature_ids",
            data=np.asarray([f"feature_{i}" for i in range(values.shape[1])], dtype=object),
            dtype=strings,
        )
        for key, value in metadata.items():
            handle.attrs[key] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _load_source(config_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = load_config(config_path)
    store = MatrixStore(config.data.rna)
    try:
        if store.col_ids is None:
            raise ValueError("RNA matrix requires gene IDs in col_ids_key")
        rows = []
        for start in range(0, store.shape[0], 128):
            rows.append(store.rows(np.arange(start, min(start + 128, store.shape[0]))))
        return np.concatenate(rows), store.row_ids.astype(str), store.col_ids.astype(str)
    finally:
        store.close()


def _official_jax(args: argparse.Namespace, raw_tpm: np.ndarray, sample_ids: np.ndarray, source_genes: np.ndarray):
    if not args.official_repo:
        raise ValueError("--official-repo is required for backend=official_jax")
    repo = Path(args.official_repo).resolve()
    sys.path.insert(0, str(repo))
    import haiku as hk  # type: ignore
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    from multiomics_open_research.bulk_rna_bert.pretrained import (  # type: ignore
        get_bulkrnabert_pretrained_model,
    )
    from multiomics_open_research.common.preprocess import preprocess_omic  # type: ignore

    common_genes_path = repo / "data" / "bulkrnabert" / "common_gene_id.txt"
    if not common_genes_path.is_file():
        raise FileNotFoundError(common_genes_path)
    common_genes = [line.strip() for line in common_genes_path.read_text().splitlines() if line.strip()]
    aligned, overlap = _align(raw_tpm, source_genes, common_genes)
    if overlap < args.min_gene_overlap:
        raise ValueError(f"BulkRNABert gene overlap {overlap:.3f} < {args.min_gene_overlap:.3f}")

    compute_dtype = jnp.bfloat16 if args.compute_dtype == "bfloat16" else jnp.float32
    parameters, forward_fn, tokenizer, model_config = get_bulkrnabert_pretrained_model(
        model_name=args.model_name,
        compute_dtype=compute_dtype,
        param_dtype=jnp.float32,
        output_dtype=compute_dtype,
        embeddings_layers_to_save=(args.layer,),
        checkpoint_directory=args.checkpoint_directory,
    )
    transformed = hk.transform(forward_fn)
    outputs = []
    for start in range(0, len(aligned), args.batch_size):
        batch = aligned[start : start + args.batch_size]
        frame = pd.DataFrame(batch, columns=common_genes)
        frame.insert(0, "identifier", sample_ids[start : start + len(batch)])
        processed = preprocess_omic(frame, model_config)
        token_ids = jnp.asarray(tokenizer.batch_tokenize(processed), dtype=jnp.int32)
        attention_mask = jnp.ones(
            (token_ids.shape[0], 1, token_ids.shape[1], token_ids.shape[1]), dtype=jnp.bool_
        )
        result = transformed.apply(parameters, jax.random.PRNGKey(0), token_ids, attention_mask)
        outputs.append(np.asarray(result[f"embeddings_{args.layer}"]).mean(axis=1, dtype=jnp.float32))
    return np.concatenate(outputs), overlap


def _huggingface(args: argparse.Namespace, raw_tpm: np.ndarray, source_genes: np.ndarray):
    from huggingface_hub import hf_hub_download  # type: ignore
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer  # type: ignore

    repo_id = "InstaDeepAI/BulkRNABert"
    sample_path = hf_hub_download(repo_id=repo_id, filename="data/tcga_sample.csv", repo_type="model")
    model_genes = pd.read_csv(sample_path, nrows=1).drop(columns=["identifier"]).columns.tolist()
    aligned, overlap = _align(raw_tpm, source_genes, model_genes)
    if overlap < args.min_gene_overlap:
        raise ValueError(f"BulkRNABert gene overlap {overlap:.3f} < {args.min_gene_overlap:.3f}")
    values = np.log10(1.0 + aligned)

    model_config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    model_config.embeddings_layers_to_save = (args.layer,)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(repo_id, config=model_config, trust_remote_code=True).to(args.device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), args.batch_size):
            token_ids = tokenizer.batch_encode_plus(
                values[start : start + args.batch_size], return_tensors="pt"
            )["input_ids"].to(args.device)
            result = model(token_ids)[f"embeddings_{args.layer}"].mean(dim=1)
            outputs.append(result.float().cpu().numpy())
    return np.concatenate(outputs), overlap


def main() -> None:
    args = _parser().parse_args()
    includes_tcga = args.backend == "huggingface" or args.model_name in {
        "bulk_rna_bert_tcga", "bulk_rna_bert_gtex_encode_tcga"
    }
    if includes_tcga and not args.allow_tcga_pretraining_overlap:
        raise SystemExit(
            "Refusing a TCGA-pretrained BulkRNABert checkpoint on the TCGA methylation benchmark. "
            "Use bulk_rna_bert_gtex_encode for the primary clean comparison, or pass "
            "--allow-tcga-pretraining-overlap only for a labelled sensitivity analysis."
        )
    values, sample_ids, source_genes = _load_source(args.config)
    raw_tpm = _to_raw_tpm(values, args.input_scale)
    if args.backend == "official_jax":
        embeddings, overlap = _official_jax(args, raw_tpm, sample_ids, source_genes)
    else:
        embeddings, overlap = _huggingface(args, raw_tpm, source_genes)
    metadata = {
        "encoder": "BulkRNABert",
        "backend": args.backend,
        "model_name": args.model_name if args.backend == "official_jax" else "InstaDeepAI/BulkRNABert",
        "pretraining_includes_tcga": bool(includes_tcga),
        "layer": args.layer,
        "pooling": "mean_gene_tokens",
        "input_scale": args.input_scale,
        "compute_dtype": args.compute_dtype,
        "gene_overlap": overlap,
        "n_samples": int(len(sample_ids)),
        "embedding_dim": int(embeddings.shape[1]),
    }
    _write(Path(args.output), embeddings, sample_ids, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
