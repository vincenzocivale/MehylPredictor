#!/usr/bin/env python3
"""Memory-efficient PyTorch inference for the official BulkRNABert JAX weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.data import MatrixStore


def _normalise_gene_id(value: object) -> str:
    fields = [field.strip() for field in str(value).split(";") if field.strip()]
    ensembl = next((field for field in fields if field.upper().startswith("ENSG")), None)
    if ensembl is not None:
        return ensembl.split(".", 1)[0]
    value = fields[0] if fields else str(value)
    return value.split(".", 1)[0] if value.upper().startswith("ENSG") else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BulkRNABertTorch:
    """Inference-only translation of the upstream Haiku pre-layer-norm encoder.

    PyTorch SDPA dispatches to a fused memory-efficient CUDA kernel, which is
    algebraically the same attention operation but does not materialize the
    [heads, genes, genes] attention matrix.
    """

    def __init__(self, params_path: Path, config_path: Path, device: torch.device, dtype: torch.dtype):
        self.config = json.loads(config_path.read_text())
        self.device, self.dtype = device, dtype
        params = joblib.load(params_path)
        self.params = {
            scope: {name: torch.as_tensor(value, device=device, dtype=dtype) for name, value in values.items()}
            for scope, values in params.items()
        }

    def _p(self, scope: str, name: str) -> torch.Tensor:
        return self.params[scope][name]

    @staticmethod
    def _linear(x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weights.transpose(0, 1), bias)

    @staticmethod
    def _layer_norm(x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), scale, offset, eps=1e-5)

    def input_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Expression-bin plus gene-identity embedding before contextualization."""
        x = self._p("bulk_bert/~/expression_embedding", "embeddings")[token_ids]
        gene = self._p("bulk_bert/~/gene_embedding", "embeddings")
        gene = self._linear(gene, self._p("bulk_bert/~/linear", "w"), self._p("bulk_bert/~/linear", "b"))
        return x + gene.unsqueeze(0)

    def __call__(
        self, token_ids: torch.Tensor, return_hidden_states: bool = False
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Returns the mean-pooled hidden state after each transformer layer (1-indexed,
        one entry per layer) instead of only the final layer -- needed both to extract
        multiple candidate layers for the R5b screen and to compare intermediate states
        against the JAX reference during parity checks. When ``return_hidden_states`` is
        set, also returns the full per-token hidden state at each layer (only intended
        for small parity-check batches, not the full production extraction sweep)."""
        c = self.config
        x = self.input_embedding(token_ids)
        heads, key_size = c["num_attention_heads"], c["key_size"]
        pooled_layers: list[torch.Tensor] = []
        hidden_state_layers: list[torch.Tensor] | None = [] if return_hidden_states else None
        for layer_idx in range(c["num_layers"]):
            scope = f"bulk_bert/self_attention_block_{layer_idx}/~/"
            residual = x
            x_norm = self._layer_norm(
                x, self._p(scope + "self_attention_layer_norm", "scale"), self._p(scope + "self_attention_layer_norm", "offset")
            )
            q = self._linear(x_norm, self._p(scope + "self_attention/query", "w"), self._p(scope + "self_attention/query", "b"))
            k = self._linear(x_norm, self._p(scope + "self_attention/key", "w"), self._p(scope + "self_attention/key", "b"))
            v = self._linear(x_norm, self._p(scope + "self_attention/value", "w"), self._p(scope + "self_attention/value", "b"))
            q = q.view(len(x), -1, heads, key_size).transpose(1, 2)
            k = k.view(len(x), -1, heads, key_size).transpose(1, 2)
            v = v.view(len(x), -1, heads, key_size).transpose(1, 2)
            attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            attention = attention.transpose(1, 2).reshape(len(x), -1, heads * key_size)
            x = residual + self._linear(
                attention, self._p(scope + "self_attention/mha_output", "w"), self._p(scope + "self_attention/mha_output", "b")
            )
            residual = x
            x_norm = self._layer_norm(x, self._p(scope + "final_layer_norm", "scale"), self._p(scope + "final_layer_norm", "offset"))
            mlp = self._linear(x_norm, self._p(scope + "fc1", "w"), self._p(scope + "fc1", "b"))
            mlp = F.gelu(mlp, approximate="none")
            x = residual + self._linear(mlp, self._p(scope + "fc2", "w"), self._p(scope + "fc2", "b"))
            pooled_layers.append(x.mean(dim=1).float())
            if hidden_state_layers is not None:
                hidden_state_layers.append(x.float())
        return pooled_layers, hidden_state_layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-scale", choices=("raw_tpm", "log2p1"), default="log2p1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-gene-overlap", type=float, default=0.999)
    parser.add_argument("--max-samples", type=int, help="test-only cap; omitted processes every aligned patient")
    parser.add_argument(
        "--token-sample-output",
        help="optional HDF5 containing a deterministic sample × gene subset of contextual token states",
    )
    parser.add_argument("--token-sample-count", type=int, default=1024)
    parser.add_argument("--token-gene-count", type=int, default=128)
    parser.add_argument("--token-sample-seed", type=int, default=17)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory-efficient BulkRNABert extraction")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    config = load_config(args.config)
    store = MatrixStore(config.data.rna)
    try:
        if store.col_ids is None:
            raise ValueError("RNA matrix requires col_ids_key")
        rows = [store.rows(np.arange(start, min(start + 128, store.shape[0]))) for start in range(0, store.shape[0], 128)]
        source, sample_ids, source_ids = np.concatenate(rows), store.row_ids.astype(str), store.col_ids.astype(str)
    finally:
        store.close()
    if args.input_scale == "log2p1":
        source = np.maximum(np.exp2(np.clip(source, 0.0, 30.0)) - 1.0, 0.0)
    else:
        source = np.maximum(source, 0.0)

    repo = Path(args.official_repo)
    common = (repo / "data/bulkrnabert/common_gene_id.txt").read_text().splitlines()
    lookup = {_normalise_gene_id(gene): index for index, gene in enumerate(source_ids)}
    indices = np.asarray([lookup.get(gene, -1) for gene in common], dtype=np.int64)
    overlap = float((indices >= 0).mean())
    if overlap < args.min_gene_overlap:
        raise ValueError(f"BulkRNABert gene overlap {overlap:.6f} < {args.min_gene_overlap:.6f}")
    aligned = np.zeros((len(source), len(common)), dtype=np.float32)
    valid = indices >= 0
    aligned[:, valid] = source[:, indices[valid]]
    if args.max_samples is not None:
        aligned, sample_ids = aligned[:args.max_samples], sample_ids[:args.max_samples]

    checkpoint = repo / "checkpoints/bulk_rna_bert_gtex_encode"
    model = BulkRNABertTorch(checkpoint / "params.joblib", checkpoint / "config.json", torch.device("cuda"), torch.bfloat16)
    values = np.log10(aligned + 1.0)
    values /= float(model.config["normalization_factor"])
    token_ids = np.digitize(values, np.linspace(0.0, 1.0, int(model.config["n_expressions_bins"]))).astype(np.int64)
    token_ids[values == 0.0] = 0
    num_layers = int(model.config["num_layers"])
    input_outputs: list[np.ndarray] = []
    outputs: list[list[np.ndarray]] = [[] for _ in range(num_layers)]

    token_sample_indices = np.empty(0, dtype=np.int64)
    token_gene_indices = np.empty(0, dtype=np.int64)
    token_input_chunks: list[np.ndarray] = []
    token_hidden_chunks: list[list[np.ndarray]] = [[] for _ in range(num_layers)]
    token_chunk_sample_ids: list[np.ndarray] = []
    token_sample_splits = np.empty(0, dtype=object)
    token_gene_variance = np.empty(0, dtype=np.float32)
    if args.token_sample_output:
        if args.token_sample_count < 3 or args.token_gene_count < 1:
            raise ValueError("token sample requires at least 3 patients and 1 gene")
        metadata_path = Path(config.data.sample_metadata.path)
        metadata = pd.read_parquet(metadata_path) if metadata_path.suffix == ".parquet" else pd.read_csv(metadata_path)
        id_column = config.data.sample_metadata.id_column
        split_column = config.data.split_column
        if not {id_column, split_column}.issubset(metadata.columns):
            raise ValueError(f"sample metadata must contain {id_column!r} and {split_column!r}")
        split_lookup = dict(zip(metadata[id_column].astype(str), metadata[split_column].astype(str), strict=False))
        missing_metadata = [str(value) for value in sample_ids if str(value) not in split_lookup]
        if missing_metadata:
            raise ValueError(f"sample metadata misses {len(missing_metadata)} RNA patients")
        split_labels = np.asarray([split_lookup[str(value)] for value in sample_ids], dtype=object)
        required_splits = (
            config.training.train_sample_split,
            config.training.validation_sample_split,
            "test",
        )
        rng = np.random.default_rng(args.token_sample_seed)
        sample_count = min(args.token_sample_count, len(sample_ids))
        selected: list[int] = []
        # Guarantee representation of every evaluation split, then fill the remaining
        # quota proportionally by seeded sampling without replacement.
        for split_name in required_splits:
            candidates = np.flatnonzero(split_labels == split_name)
            if not len(candidates):
                raise ValueError(f"metadata contains no {split_name!r} patients for token audit")
            selected.append(int(rng.choice(candidates)))
        remaining = np.setdiff1d(np.arange(len(sample_ids)), np.asarray(selected), assume_unique=False)
        if sample_count > len(selected):
            selected.extend(rng.choice(remaining, size=sample_count - len(selected), replace=False).tolist())
        token_sample_indices = np.sort(np.asarray(selected, dtype=np.int64))
        token_sample_splits = split_labels[token_sample_indices]

        # Cover the full train-only variance spectrum; validation/test expression
        # never determines which genes are inspected.
        train_rows = np.flatnonzero(split_labels == config.training.train_sample_split)
        log_expression_train = np.log2(aligned[train_rows] + 1.0)
        all_gene_variance = np.var(log_expression_train, axis=0)
        variance_order = np.argsort(all_gene_variance, kind="stable")
        positions = np.linspace(0, len(variance_order) - 1, min(args.token_gene_count, len(variance_order)), dtype=int)
        token_gene_indices = np.sort(variance_order[positions])
        token_gene_variance = all_gene_variance[token_gene_indices].astype(np.float32)
    started = time.time()
    with torch.inference_mode(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
        for start in range(0, len(token_ids), args.batch_size):
            batch = torch.as_tensor(token_ids[start:start + args.batch_size], device="cuda")
            batch_indices = np.arange(start, min(start + args.batch_size, len(token_ids)))
            selected_mask = np.isin(batch_indices, token_sample_indices, assume_unique=True)
            input_hidden = model.input_embedding(batch)
            input_outputs.append(input_hidden.mean(dim=1).float().cpu().numpy())
            pooled_layers, hidden_layers = model(batch, return_hidden_states=bool(selected_mask.any()))
            for layer_idx, pooled in enumerate(pooled_layers):
                outputs[layer_idx].append(pooled.cpu().numpy())
            if selected_mask.any():
                if hidden_layers is None:
                    raise RuntimeError("token hidden states were requested but not returned")
                local_rows = np.flatnonzero(selected_mask)
                token_chunk_sample_ids.append(batch_indices[local_rows])
                token_input_chunks.append(
                    input_hidden[local_rows][:, token_gene_indices, :].float().cpu().numpy().astype(np.float32, copy=False)
                )
                for layer_idx, hidden in enumerate(hidden_layers):
                    selected = hidden[local_rows][:, token_gene_indices, :]
                    token_hidden_chunks[layer_idx].append(selected.cpu().numpy().astype(np.float32, copy=False))
            if start == 0 or (start // args.batch_size + 1) % 32 == 0:
                print(f"processed={min(start + args.batch_size, len(token_ids))}/{len(token_ids)}", flush=True)
    embeddings_layer0 = np.concatenate(input_outputs).astype(np.float32, copy=False)
    embeddings_by_layer = [np.concatenate(layer_chunks).astype(np.float32, copy=False) for layer_chunks in outputs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output, "w") as handle:
        handle.create_dataset("embeddings_layer0", data=embeddings_layer0)
        for layer_idx, embeddings in enumerate(embeddings_by_layer, start=1):
            handle.create_dataset(f"embeddings_layer{layer_idx}", data=embeddings)
        handle.create_dataset("embeddings", data=embeddings_by_layer[-1])
        handle.create_dataset("sample_idx", data=np.asarray(sample_ids, dtype=object), dtype=strings)
        handle.create_dataset(
            "feature_ids",
            data=np.asarray([f"feature_{i}" for i in range(embeddings_by_layer[-1].shape[1])], dtype=object),
            dtype=strings,
        )
    if args.token_sample_output:
        observed = np.concatenate(token_chunk_sample_ids)
        order = np.argsort(observed)
        if not np.array_equal(observed[order], token_sample_indices):
            raise RuntimeError("failed to capture the requested token sample exactly once")
        token_output = Path(args.token_sample_output)
        token_output.parent.mkdir(parents=True, exist_ok=True)
        strings = h5py.string_dtype(encoding="utf-8")
        with h5py.File(token_output, "w") as handle:
            handle.create_dataset("tokens_layer0", data=np.concatenate(token_input_chunks, axis=0)[order], compression="gzip", compression_opts=1)
            for layer_idx, chunks in enumerate(token_hidden_chunks, start=1):
                values_layer = np.concatenate(chunks, axis=0)[order]
                handle.create_dataset(f"tokens_layer{layer_idx}", data=values_layer, compression="gzip", compression_opts=1)
            handle.create_dataset("sample_idx", data=np.asarray(sample_ids[token_sample_indices], dtype=object), dtype=strings)
            handle.create_dataset("sample_split", data=np.asarray(token_sample_splits, dtype=object), dtype=strings)
            handle.create_dataset("gene_ids", data=np.asarray(np.asarray(common)[token_gene_indices], dtype=object), dtype=strings)
            handle.create_dataset(
                "expression_log2p1",
                data=np.log2(aligned[token_sample_indices][:, token_gene_indices] + 1.0).astype(np.float32),
            )
            handle.create_dataset("token_ids", data=token_ids[token_sample_indices][:, token_gene_indices].astype(np.int16))
            handle.create_dataset("gene_variance_log2p1_train", data=token_gene_variance)

    metadata = {
        "encoder": "BulkRNABert", "backend": "torch_sdpa_flash", "model_name": "bulk_rna_bert_gtex_encode",
        "pretraining_includes_tcga": False, "input_scale": args.input_scale, "compute_dtype": "bfloat16",
        "gene_overlap": overlap, "n_samples": int(len(sample_ids)), "embedding_dim": int(embeddings_by_layer[-1].shape[1]),
        "layers_saved": list(range(0, num_layers + 1)),
        "official_repo": str(repo), "official_repo_checkpoint_sha256": _sha256(checkpoint / "params.joblib"),
        "elapsed_seconds": time.time() - started,
        "token_sample_output": args.token_sample_output,
        "token_sample_count": int(len(token_sample_indices)),
        "token_gene_count": int(len(token_gene_indices)),
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
