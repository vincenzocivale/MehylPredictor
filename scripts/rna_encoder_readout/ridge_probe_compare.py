#!/usr/bin/env python3
"""Ridge-probe comparison of frozen RNA-only readout embeddings.

Decouples pooling-architecture quality from gradient-decoder training budget
and from warm-start optimisation budget: every candidate embedding is scored
with the same closed-form Ridge regression used by the rna_encoder_quality
audit, against the same disjoint target-gene set used by the P0/P1 readout
objective (train-only stratified-variance selection, excluding the cached
input genes). No SGD decoder is trained here.

Candidates are either:
  - full_mean_layer{N}   -> the cache's official full-context (19,062-gene)
                            mean-pooled embedding for layer N;
  - NAME=/path/embeddings.h5 -> a readout run's exported embeddings.h5
                            (mean_resume_layer2, concat_means, scalar_mix, ...);
  - NAME=/path/embeddings.h5=WEIGHT_MODE for the fixed weighted poolers,
                            computed on the fly from tokens_layer{N} without
                            any gradient training (deterministic pooling);
  - NAME=pca256 | NAME=randproj256 -> matched-target linear baselines fit on
                            the same 4096 cached input genes (raw expression,
                            train-standardized), giving the missing upper
                            bound that the older, differently-selected-target
                            quality audit's 0.574/0.329 numbers cannot supply.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from methylation_predictor.rna_encoder_quality.probes import (
    fit_ridge_probe,
    residualize_using_training_means,
    training_group_means,
)
from methylation_predictor.rna_encoder_readout.io import (
    _normalise_gene_id,
    _stratified_target_indices_from_stats,
    _stream_train_gene_stats,
)


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ], dtype=object)


def _normalise_split(value: str) -> str:
    text = value.strip().lower()
    if text in {"val", "valid", "validation"}:
        return "validation"
    if text in {"train", "training"}:
        return "train"
    if text in {"test", "testing"}:
        return "test"
    return text


def _load_cache_context(cache_path: str) -> dict:
    with h5py.File(cache_path, "r") as handle:
        sample_ids = _decode(np.asarray(handle["sample_idx"]))
        splits = np.asarray([_normalise_split(s) for s in _decode(np.asarray(handle["split"]))], dtype=object)
        cancer_types = _decode(np.asarray(handle["cancer_type"]))
        gene_ids = _decode(np.asarray(handle["gene_ids"]))
        full_mean = {
            int(key.removeprefix("full_mean_layer")): np.asarray(handle[key], dtype=np.float32)
            for key in handle
            if key.startswith("full_mean_layer")
        }
        weighted_source = {
            "total_variance": np.asarray(handle["gene_variance_log2p1_train"], dtype=np.float32),
            "within_variance": np.asarray(handle["gene_within_variance_log2p1_train"], dtype=np.float32),
        }
        weighted_source["inverse_variance"] = 1.0 / np.maximum(weighted_source["total_variance"], 1e-8)
        input_expression = np.asarray(handle["expression_log2p1"], dtype=np.float32)
    return dict(
        sample_ids=sample_ids, splits=splits, cancer_types=cancer_types,
        gene_ids=gene_ids, full_mean=full_mean, weighted_source=weighted_source,
        input_expression=input_expression,
    )


def _standardize_from_train(values: np.ndarray, train: np.ndarray) -> np.ndarray:
    mean = values[train].mean(axis=0)
    scale = values[train].std(axis=0)
    scale[scale < 1e-8] = 1.0
    return ((values - mean) / scale).astype(np.float32)


def _pca_embedding(expression: np.ndarray, train: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    standardized = _standardize_from_train(expression, train)
    n_components = min(n_components, int(train.sum()) - 1, standardized.shape[1])
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    model.fit(standardized[train])
    return model.transform(standardized).astype(np.float32)


def _random_projection_embedding(expression: np.ndarray, train: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    standardized = _standardize_from_train(expression, train)
    n_components = min(n_components, standardized.shape[1])
    projector = GaussianRandomProjection(n_components=n_components, random_state=seed)
    projector.fit(standardized[train])
    return projector.transform(standardized).astype(np.float32)


def _load_full_checkpoint_expression(rna_path: str, official_repo: str, target_sample_ids: np.ndarray) -> np.ndarray:
    """Raw log2(TPM+1) for all 19,062 BulkRNABert checkpoint genes, aligned
    the same way build_token_cache.py aligns them, but without subsampling."""
    repo = Path(official_repo)
    common = np.asarray((repo / "data/bulkrnabert/common_gene_id.txt").read_text().splitlines(), dtype=object)
    with h5py.File(rna_path, "r") as handle:
        rna_sample_ids = _decode(np.asarray(handle["sample_idx"]))
        source_gene_ids = _decode(np.asarray(handle["gene_ids"]))
        lookup = {_normalise_gene_id(gene): index for index, gene in enumerate(source_gene_ids)}
        indices = np.asarray([lookup.get(str(gene), -1) for gene in common], dtype=np.int64)
        if np.any(indices < 0):
            raise ValueError(f"RNA source misses {int(np.sum(indices < 0))}/19062 BulkRNABert genes")
        positions = np.argsort(indices)
        selected_sorted = np.asarray(handle["X"][:, indices[positions]], dtype=np.float32)
        aligned = np.empty((len(rna_sample_ids), len(common)), dtype=np.float32)
        aligned[:, positions] = selected_sorted
    sample_lookup = {str(value): index for index, value in enumerate(rna_sample_ids)}
    rows = np.asarray([sample_lookup[str(value)] for value in target_sample_ids], dtype=np.int64)
    return aligned[rows]


def _select_targets(
    rna_path: str, sample_ids: np.ndarray, splits: np.ndarray, excluded_gene_ids: np.ndarray,
    target_gene_count: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(rna_path, "r") as handle:
        rna_sample_ids = _decode(np.asarray(handle["sample_idx"]))
        rna_gene_ids = _decode(np.asarray(handle["gene_ids"]))
        lookup = {str(value): index for index, value in enumerate(rna_sample_ids)}
        aligned_rows = np.asarray([lookup[str(value)] for value in sample_ids], dtype=np.int64)
        train_rows = np.sort(aligned_rows[splits == "train"])
        dataset = handle["X"]
        variance, prevalence = _stream_train_gene_stats(dataset, train_rows)
        excluded = {_normalise_gene_id(gene) for gene in excluded_gene_ids}
        target_indices = _stratified_target_indices_from_stats(
            variance, prevalence, rna_gene_ids, excluded, target_gene_count, seed,
        )
        selected_all_rows = np.asarray(dataset[:, target_indices], dtype=np.float32)
        target = selected_all_rows[aligned_rows]
    return target, rna_gene_ids[target_indices]


def _load_embedding_h5(path: str, sample_ids: np.ndarray, key: str = "embeddings") -> np.ndarray:
    with h5py.File(path, "r") as handle:
        embedding_samples = _decode(np.asarray(handle["sample_idx"]))
        embedding = np.asarray(handle[key], dtype=np.float32)
    lookup = {str(value): index for index, value in enumerate(embedding_samples)}
    rows = np.asarray([lookup[str(value)] for value in sample_ids], dtype=np.int64)
    return embedding[rows]


def _weighted_mean_embedding(
    cache_path: str, sample_ids: np.ndarray, layer: int, weights: np.ndarray, chunk_size: int = 256,
) -> np.ndarray:
    weights = np.maximum(weights, 0.0)
    weights = (weights / weights.sum()).astype(np.float32)
    with h5py.File(cache_path, "r") as handle:
        cache_samples = _decode(np.asarray(handle["sample_idx"]))
        lookup = {str(value): index for index, value in enumerate(cache_samples)}
        rows = np.asarray([lookup[str(value)] for value in sample_ids], dtype=np.int64)
        tokens = handle[f"tokens_layer{layer}"]
        embedding = np.empty((len(rows), tokens.shape[-1]), dtype=np.float32)
        order = np.argsort(rows)
        sorted_rows = rows[order]
        for start in range(0, len(sorted_rows), chunk_size):
            stop = min(start + chunk_size, len(sorted_rows))
            batch = np.asarray(tokens[sorted_rows[start:stop]], dtype=np.float32)
            embedding[order[start:stop]] = np.einsum("g,bgd->bd", weights, batch)
    return embedding


def _parse_candidate(
    spec: str, cache_path: str, context: dict, sample_ids: np.ndarray, train_mask: np.ndarray, seed: int,
    rna_path: str | None = None, official_repo: str | None = None,
) -> tuple[str, np.ndarray]:
    name, _, source = spec.partition("=")
    if not source:
        raise argparse.ArgumentTypeError(f"candidate must be NAME=SOURCE, got {spec!r}")
    if source == "pca256":
        return name, _pca_embedding(context["input_expression"], train_mask, 256, seed)
    if source == "randproj256":
        return name, _random_projection_embedding(context["input_expression"], train_mask, 256, seed)
    if source == "pca19062":
        if not rna_path or not official_repo:
            raise argparse.ArgumentTypeError("pca19062 requires --rna-h5 and --official-repo")
        expression_19062 = _load_full_checkpoint_expression(rna_path, official_repo, sample_ids)
        return name, _pca_embedding(expression_19062, train_mask, 256, seed)
    if source.startswith("full_mean_layer"):
        layer = int(source.removeprefix("full_mean_layer"))
        return name, context["full_mean"][layer]
    if "=" in source:
        path, weight_mode = source.split("=", 1)
        layer = int(Path(path).stem.split("layer")[-1]) if "layer" in Path(path).stem else 2
        return name, _weighted_mean_embedding(cache_path, sample_ids, layer, context["weighted_source"][weight_mode])
    return name, _load_embedding_h5(source, sample_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="token cache with full_mean_layer{N} and split/cancer metadata")
    parser.add_argument("--rna-h5", required=True)
    parser.add_argument("--target-gene-count", type=int, default=4096)
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument(
        "--candidate", action="append", required=True, dest="candidates",
        help="NAME=full_mean_layer{N} | NAME=path/to/embeddings.h5 | NAME=path/to/cache=WEIGHT_MODE",
    )
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1.0,10.0,100.0")
    parser.add_argument("--official-repo", help="required for NAME=pca19062 candidates")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    context = _load_cache_context(args.cache)
    sample_ids, splits, cancer_types = context["sample_ids"], context["splits"], context["cancer_types"]
    target, target_gene_ids = _select_targets(
        args.rna_h5, sample_ids, splits, context["gene_ids"], args.target_gene_count, args.selection_seed,
    )
    train, validation, test = splits == "train", splits == "validation", splits == "test"
    global_mean, group_means = training_group_means(target[train], cancer_types[train])
    within = residualize_using_training_means(target, cancer_types, global_mean, group_means)
    alphas = tuple(float(value) for value in args.ridge_alphas.split(","))

    rows = []
    for spec in args.candidates:
        name, embedding = _parse_candidate(
            spec, args.cache, context, sample_ids, train, args.selection_seed,
            rna_path=args.rna_h5, official_repo=args.official_repo,
        )
        for target_name, target_values in (("total_rna", target), ("within_cancer_rna", within)):
            result = fit_ridge_probe(
                embedding[train], embedding[validation], embedding[test],
                target_values[train], target_values[validation], target_values[test],
                alphas,
            )
            rows.append({"candidate": name, "target": target_name, **asdict(result)})
        print(f"done: {name}", flush=True)

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"\nn_target_genes={len(target_gene_ids)}")


if __name__ == "__main__":
    main()
