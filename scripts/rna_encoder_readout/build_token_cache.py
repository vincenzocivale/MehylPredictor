#!/usr/bin/env python3
"""Build a full-patient, selected-gene token cache from frozen BulkRNABert.

BulkRNABert always receives all 19,062 checkpoint genes. Gene selection occurs
only after the complete forward pass, so the frozen encoder context is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ], dtype=object)


def _normalise_gene_id(value: object) -> str:
    fields = [field.strip() for field in str(value).split(";") if field.strip()]
    ensembl = next((field for field in fields if field.upper().startswith("ENSG")), None)
    if ensembl is not None:
        return ensembl.split(".", 1)[0]
    value = fields[0] if fields else str(value)
    return value.split(".", 1)[0] if value.upper().startswith("ENSG") else value


def _sha256_strings(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_extractor(path: Path):
    spec = importlib.util.spec_from_file_location("bulkrnabert_extractor_for_cache", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import extractor from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _select_genes(
    expression: np.ndarray,
    train_rows: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = expression[train_rows]
    variance = np.var(train, axis=0)
    prevalence = np.mean(train > 0, axis=0)
    residual = train.copy()
    # Within-cancer residual is computed by the caller after supplying labels;
    # here it is initialized to total variance and overwritten in main.
    eligible = np.flatnonzero(prevalence > 0.01)
    if len(eligible) < count:
        eligible = np.arange(expression.shape[1])
    ordered = eligible[np.argsort(variance[eligible], kind="stable")]
    quartiles = np.array_split(ordered, 4)
    quotas = [count // 4, count // 4, count // 4, count // 4]
    quotas[-1] += count - sum(quotas)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for group, quota in zip(quartiles, quotas, strict=True):
        if len(group) <= quota:
            selected.append(group)
        else:
            selected.append(np.sort(rng.choice(group, size=quota, replace=False)))
    result = np.concatenate(selected)
    if len(result) < count:
        remaining = np.setdiff1d(np.arange(expression.shape[1]), result)
        result = np.concatenate([result, remaining[: count - len(result)]])
    result = np.sort(result[:count])
    return result, variance, prevalence, residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-h5", required=True, help="full-gene TCGA log2(TPM+1) HDF5")
    parser.add_argument("--metadata", required=True, help="CSV/parquet with sample_idx, split and cancer_type")
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--extractor", default="scripts/rna_branch/extract_bulkrnabert_torch.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="2", help="comma-separated layer numbers; 0 is pre-Transformer")
    parser.add_argument("--gene-count", type=int, default=4096)
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--selection-from-cache", help="reuse checkpoint_gene_indices from an existing cache")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--values-key", default="X")
    parser.add_argument("--sample-ids-key", default="sample_idx")
    parser.add_argument("--gene-ids-key", default="gene_ids")
    parser.add_argument("--metadata-id-column", default="sample_idx")
    parser.add_argument("--metadata-split-column", default="split")
    parser.add_argument("--metadata-cancer-column", default="cancer_type")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory-efficient BulkRNABert token extraction")
    layers = sorted({int(value) for value in args.layers.split(",") if value.strip()})
    if not layers or min(layers) < 0 or max(layers) > 4:
        raise ValueError("--layers must be a non-empty subset of 0,1,2,3,4")
    if args.gene_count < 1 or args.batch_size < 1:
        raise ValueError("gene-count and batch-size must be positive")

    repo = Path(args.official_repo)
    common = np.asarray((repo / "data/bulkrnabert/common_gene_id.txt").read_text().splitlines(), dtype=object)
    with h5py.File(args.rna_h5, "r") as handle:
        sample_ids = _decode(np.asarray(handle[args.sample_ids_key]))
        source_gene_ids = _decode(np.asarray(handle[args.gene_ids_key]))
        lookup = {_normalise_gene_id(gene): index for index, gene in enumerate(source_gene_ids)}
        indices = np.asarray([lookup.get(str(gene), -1) for gene in common], dtype=np.int64)
        if np.any(indices < 0):
            raise ValueError(f"full-gene source still misses {int(np.sum(indices < 0))}/19062 BulkRNABert genes")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("multiple BulkRNABert genes map to the same source column")
        positions = np.argsort(indices)
        selected_sorted = np.asarray(handle[args.values_key][:, indices[positions]], dtype=np.float32)
        aligned_log2p1 = np.empty((len(sample_ids), len(common)), dtype=np.float32)
        aligned_log2p1[:, positions] = selected_sorted
    metadata_path = Path(args.metadata)
    metadata = pd.read_parquet(metadata_path) if metadata_path.suffix == ".parquet" else pd.read_csv(metadata_path)
    required = {args.metadata_id_column, args.metadata_split_column, args.metadata_cancer_column}
    if not required.issubset(metadata.columns):
        raise ValueError(f"metadata is missing columns {sorted(required - set(metadata.columns))}")
    metadata = metadata.copy()
    metadata[args.metadata_id_column] = metadata[args.metadata_id_column].astype(str)
    metadata = metadata.drop_duplicates(args.metadata_id_column).set_index(args.metadata_id_column)
    missing = [str(sample) for sample in sample_ids if str(sample) not in metadata.index]
    if missing:
        raise ValueError(f"metadata misses {len(missing)} RNA samples; first={missing[:5]}")
    splits = metadata.loc[np.asarray(sample_ids).astype(str), args.metadata_split_column].astype(str).to_numpy()
    splits = np.asarray(["validation" if value.lower() in {"val", "valid", "validation"} else value.lower() for value in splits], dtype=object)
    cancers = metadata.loc[np.asarray(sample_ids).astype(str), args.metadata_cancer_column].astype(str).to_numpy(dtype=object)

    aligned_tpm = np.maximum(np.exp2(np.clip(aligned_log2p1, 0.0, 30.0)) - 1.0, 0.0)
    checkpoint = repo / "checkpoints/bulk_rna_bert_gtex_encode"
    extractor = _load_extractor(Path(args.extractor))
    model = extractor.BulkRNABertTorch(
        checkpoint / "params.joblib", checkpoint / "config.json", torch.device("cuda"), torch.bfloat16
    )
    normalized = np.log10(aligned_tpm + 1.0) / float(model.config["normalization_factor"])
    token_ids = np.digitize(
        normalized, np.linspace(0.0, 1.0, int(model.config["n_expressions_bins"]))
    ).astype(np.int64)
    token_ids[normalized == 0.0] = 0

    train_rows = np.flatnonzero(splits == "train")
    if not len(train_rows):
        raise ValueError("metadata contains no train samples")
    if args.selection_from_cache:
        with h5py.File(args.selection_from_cache, "r") as selection_handle:
            if "checkpoint_gene_indices" not in selection_handle:
                raise KeyError("selection cache lacks checkpoint_gene_indices")
            selected = np.asarray(selection_handle["checkpoint_gene_indices"], dtype=np.int64)
        if args.gene_count != len(selected):
            print(
                f"selection-from-cache overrides --gene-count: {args.gene_count} -> {len(selected)}",
                flush=True,
            )
        variance = np.var(aligned_log2p1[train_rows], axis=0)
        prevalence = np.mean(aligned_log2p1[train_rows] > 0, axis=0)
    else:
        selected, variance, prevalence, _ = _select_genes(
            aligned_log2p1, train_rows, min(args.gene_count, len(common)), args.selection_seed
        )
    residual_train = aligned_log2p1[train_rows].copy()
    train_cancers = cancers[train_rows]
    for cancer in np.unique(train_cancers):
        mask = train_cancers == cancer
        residual_train[mask] -= residual_train[mask].mean(axis=0, keepdims=True)
    within_variance = np.var(residual_train, axis=0)

    if args.max_samples is not None:
        keep = slice(0, args.max_samples)
        sample_ids, splits, cancers = sample_ids[keep], splits[keep], cancers[keep]
        aligned_log2p1, normalized, token_ids = aligned_log2p1[keep], normalized[keep], token_ids[keep]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype("utf-8")
    storage_dtype = np.float16 if args.dtype == "float16" else np.float32
    started = time.time()
    with h5py.File(output, "w") as handle:
        token_datasets = {
            layer: handle.create_dataset(
                f"tokens_layer{layer}",
                shape=(len(sample_ids), len(selected), int(model.config["embed_dim"])),
                dtype=storage_dtype,
                chunks=(1, min(256, len(selected)), int(model.config["embed_dim"])),
            )
            for layer in layers
        }
        full_mean_datasets = {
            layer: handle.create_dataset(
                f"full_mean_layer{layer}",
                shape=(len(sample_ids), int(model.config["embed_dim"])),
                dtype=np.float32,
            )
            for layer in layers
        }
        handle.create_dataset("expression_log2p1", data=aligned_log2p1[:, selected].astype(np.float32))
        handle.create_dataset("normalized_expression", data=normalized[:, selected].astype(np.float32))
        handle.create_dataset("token_ids", data=token_ids[:, selected].astype(np.int16))
        handle.create_dataset("sample_idx", data=np.asarray(sample_ids, dtype=object), dtype=strings)
        handle.create_dataset("split", data=np.asarray(splits, dtype=object), dtype=strings)
        handle.create_dataset("cancer_type", data=np.asarray(cancers, dtype=object), dtype=strings)
        handle.create_dataset("gene_ids", data=np.asarray(common[selected], dtype=object), dtype=strings)
        handle.create_dataset("checkpoint_gene_indices", data=selected.astype(np.int32))
        handle.create_dataset("gene_variance_log2p1_train", data=variance[selected].astype(np.float32))
        handle.create_dataset("gene_within_variance_log2p1_train", data=within_variance[selected].astype(np.float32))
        handle.create_dataset("gene_expression_prevalence_train", data=prevalence[selected].astype(np.float32))

        with torch.inference_mode(), torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=False
        ):
            for start in range(0, len(sample_ids), args.batch_size):
                stop = min(start + args.batch_size, len(sample_ids))
                batch = torch.as_tensor(token_ids[start:stop], device="cuda")
                layer0 = model.input_embedding(batch) if 0 in layers else None
                _, hidden = model(batch, return_hidden_states=True)
                if hidden is None:
                    raise RuntimeError("extractor did not return hidden states")
                for layer in layers:
                    values = layer0 if layer == 0 else hidden[layer - 1]
                    full_mean_datasets[layer][start:stop] = values.mean(dim=1).float().cpu().numpy()
                    token_datasets[layer][start:stop] = (
                        values[:, selected, :].float().cpu().numpy().astype(storage_dtype, copy=False)
                    )
                if start == 0 or (start // args.batch_size + 1) % 32 == 0:
                    print(f"processed={stop}/{len(sample_ids)}", flush=True)

    sidecar = {
        "encoder": "BulkRNABert",
        "checkpoint": "bulk_rna_bert_gtex_encode",
        "encoder_frozen": True,
        "layers": layers,
        "n_samples": int(len(sample_ids)),
        "n_input_checkpoint_genes": int(len(common)),
        "n_cached_genes": int(len(selected)),
        "selection": "reused cache indices" if args.selection_from_cache else "train-only expression-variance quartiles",
        "selection_from_cache": args.selection_from_cache,
        "selection_seed": args.selection_seed,
        "selection_checksum": _sha256_strings(common[selected]),
        "dtype": args.dtype,
        "elapsed_seconds": time.time() - started,
        "methylation_inputs_loaded": False,
        "full_mean_anchor_genes": int(len(common)),
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(json.dumps(sidecar, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
