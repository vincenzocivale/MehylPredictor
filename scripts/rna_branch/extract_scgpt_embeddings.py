#!/usr/bin/env python3
"""Treat each bulk RNA profile as a pseudo-cell and materialize frozen scGPT embeddings.

This is deliberately an *evaluation adapter*, not a claim that single-cell pretraining is
well matched to bulk RNA.  The output is consumed through data.pretrained_rna so the same
replace/concat probes can be compared against BulkRNABert and the supervised linear encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.data import MatrixStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True, help="downloaded official scGPT checkpoint directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gene-map", help="optional TSV with columns source_id and gene_symbol")
    parser.add_argument("--input-scale", choices=("raw", "log2p1", "log10p1"), default="log2p1")
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-fast-transformer", action="store_true")
    return parser


def _to_nonnegative(values: np.ndarray, scale: str) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if scale == "raw":
        return np.maximum(values, 0.0)
    if scale == "log2p1":
        return np.maximum(np.exp2(np.clip(values, 0.0, 30.0)) - 1.0, 0.0)
    return np.maximum(np.power(10.0, np.clip(values, 0.0, 12.0)) - 1.0, 0.0)


def _write(path: Path, embeddings: np.ndarray, sample_ids: np.ndarray, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("embeddings", data=np.asarray(embeddings, dtype=np.float32))
        handle.create_dataset("sample_idx", data=np.asarray(sample_ids, dtype=object), dtype=strings)
        handle.create_dataset(
            "feature_ids",
            data=np.asarray([f"feature_{i}" for i in range(embeddings.shape[1])], dtype=object),
            dtype=strings,
        )
        for key, value in metadata.items():
            handle.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def main() -> None:
    args = _parser().parse_args()
    import anndata as ad  # type: ignore
    import scgpt as scg  # type: ignore

    config = load_config(args.config)
    store = MatrixStore(config.data.rna)
    try:
        if store.col_ids is None:
            raise ValueError("RNA matrix requires gene IDs in col_ids_key")
        chunks = [
            store.rows(np.arange(start, min(start + 128, store.shape[0])))
            for start in range(0, store.shape[0], 128)
        ]
        values = _to_nonnegative(np.concatenate(chunks), args.input_scale)
        sample_ids = store.row_ids.astype(str)
        genes = store.col_ids.astype(str)
    finally:
        store.close()

    if args.gene_map:
        mapping = pd.read_csv(args.gene_map, sep="\t")
        required = {"source_id", "gene_symbol"}
        if not required.issubset(mapping.columns):
            raise ValueError(f"gene map requires columns {sorted(required)}")
        lookup = dict(zip(mapping["source_id"].astype(str), mapping["gene_symbol"].astype(str)))
        keep = np.asarray([gene in lookup for gene in genes])
        genes = np.asarray([lookup[gene] for gene in genes[keep]], dtype=object)
        values = values[:, keep]
    if len(set(genes.tolist())) != len(genes):
        raise ValueError("scGPT gene symbols must be unique; deduplicate the optional gene map first")

    adata = ad.AnnData(X=values)
    adata.obs_names = sample_ids
    adata.var_names = genes
    adata.var["feature_name"] = genes
    embedded = scg.tasks.embed_data(
        adata,
        Path(args.model_dir),
        gene_col="feature_name",
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        use_fast_transformer=not args.disable_fast_transformer,
        return_new_adata=True,
    )
    embeddings = np.asarray(embedded.X, dtype=np.float32)
    metadata = {
        "encoder": "scGPT",
        "model_dir": str(Path(args.model_dir).resolve()),
        "pooling": "official_cls_cell_embedding",
        "bulk_as_pseudocell": True,
        "input_scale": args.input_scale,
        "max_length": args.max_length,
        "n_input_genes": int(len(genes)),
        "n_samples": int(len(sample_ids)),
        "embedding_dim": int(embeddings.shape[1]),
    }
    _write(Path(args.output), embeddings, sample_ids, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
