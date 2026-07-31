"""Leakage-safe loading and alignment for RNA encoder quality audits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

from .config import EmbeddingSpec, MatrixSpec, MetadataSpec, QualityConfig, StabilityViewSpec, TokenEmbeddingSpec


def decode_ids(values: np.ndarray) -> np.ndarray:
    decoded: list[str] = []
    for value in np.asarray(values).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        decoded.append(str(value))
    return np.asarray(decoded, dtype=object)


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(source, sep="\t" if suffix in {".tsv", ".txt"} else ",")
    raise ValueError(f"unsupported metadata format: {source}")


def unique_index(ids: Iterable[object], name: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, value in enumerate(ids):
        key = str(value)
        if key in result:
            raise ValueError(f"duplicate {name}: {key}")
        result[key] = index
    return result


def h5_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return np.empty((0, dataset.shape[1]), dtype=np.float32)
    unique, inverse = np.unique(rows, return_inverse=True)
    return np.asarray(dataset[unique, :], dtype=np.float32)[inverse]


@dataclass(slots=True)
class AlignedData:
    sample_ids: np.ndarray
    cancer_types: np.ndarray
    splits: np.ndarray
    split_indices: dict[str, np.ndarray]
    rna_rows: np.ndarray
    embedding_rows: np.ndarray
    gene_ids: np.ndarray
    selected_gene_indices: np.ndarray
    selected_gene_ids: np.ndarray
    rna_selected: np.ndarray
    embeddings: dict[str, np.ndarray]


def _normalise_split(values: np.ndarray, metadata: MetadataSpec) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = np.asarray(values, dtype=object).astype(str)
    normalised = np.full(len(raw), "other", dtype=object)
    normalised[np.isin(raw, np.asarray(metadata.train_values, dtype=str))] = "train"
    normalised[np.isin(raw, np.asarray(metadata.validation_values, dtype=str))] = "validation"
    normalised[np.isin(raw, np.asarray(metadata.test_values, dtype=str))] = "test"
    unknown = np.unique(raw[normalised == "other"])
    if len(unknown):
        raise ValueError(f"metadata contains unmapped split values: {unknown.tolist()}")
    indices = {name: np.flatnonzero(normalised == name) for name in ("train", "validation", "test")}
    for name, rows in indices.items():
        if not len(rows):
            raise ValueError(f"metadata has no samples assigned to the {name!r} split")
    return normalised, indices


def _scan_train_variance(
    matrix: h5py.Dataset,
    rows: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    total = np.zeros(matrix.shape[1], dtype=np.float64)
    total_sq = np.zeros(matrix.shape[1], dtype=np.float64)
    count = np.zeros(matrix.shape[1], dtype=np.int64)
    for start in range(0, len(rows), chunk_size):
        chunk = h5_rows(matrix, rows[start : start + chunk_size]).astype(np.float64)
        finite = np.isfinite(chunk)
        total += np.where(finite, chunk, 0.0).sum(axis=0)
        total_sq += np.where(finite, chunk * chunk, 0.0).sum(axis=0)
        count += finite.sum(axis=0)
    safe = np.maximum(count, 1)
    mean = total / safe
    variance = np.maximum(total_sq / safe - mean * mean, 0.0)
    variance[count < 2] = 0.0
    return variance


def _load_selected_columns(
    matrix: h5py.Dataset,
    rows: np.ndarray,
    columns: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    output = np.empty((len(rows), len(columns)), dtype=np.float32)
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        output[start:stop] = h5_rows(matrix, rows[start:stop])[:, columns]
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def load_aligned_data(config: QualityConfig) -> AlignedData:
    metadata = read_table(config.metadata.path).copy()
    required = {config.metadata.id_column, config.metadata.cancer_column, config.metadata.split_column}
    missing_columns = sorted(required - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"metadata is missing columns: {missing_columns}")
    metadata[config.metadata.id_column] = metadata[config.metadata.id_column].astype(str)
    if metadata[config.metadata.id_column].duplicated().any():
        raise ValueError("metadata contains duplicate sample IDs")

    with h5py.File(config.rna.path, "r") as rna_handle, h5py.File(config.embeddings.path, "r") as embedding_handle:
        for key in (config.rna.values_key, config.rna.row_ids_key, config.rna.col_ids_key):
            if key not in rna_handle:
                raise KeyError(f"RNA HDF5 does not contain {key!r}")
        for key in (config.embeddings.row_ids_key, *config.embeddings.layers.values()):
            if key not in embedding_handle:
                raise KeyError(f"embedding HDF5 does not contain {key!r}")
        rna_ids = decode_ids(rna_handle[config.rna.row_ids_key][...])
        embedding_ids = decode_ids(embedding_handle[config.embeddings.row_ids_key][...])
        gene_ids = decode_ids(rna_handle[config.rna.col_ids_key][...])
        rna_map = unique_index(rna_ids, "RNA sample ID")
        embedding_map = unique_index(embedding_ids, "embedding sample ID")

        requested = metadata[config.metadata.id_column].tolist()
        missing = [sample_id for sample_id in requested if sample_id not in rna_map or sample_id not in embedding_map]
        if missing and not config.analysis.allow_partial_overlap:
            raise ValueError(f"{len(missing)} metadata samples are missing from RNA or embeddings; examples={missing[:5]}")
        if missing:
            metadata = metadata[~metadata[config.metadata.id_column].isin(missing)].copy()
        sample_ids = metadata[config.metadata.id_column].to_numpy(dtype=object)
        cancer_types = metadata[config.metadata.cancer_column].astype(str).to_numpy(dtype=object)
        splits, split_indices = _normalise_split(
            metadata[config.metadata.split_column].astype(str).to_numpy(), config.metadata
        )
        rna_rows = np.asarray([rna_map[str(value)] for value in sample_ids], dtype=np.int64)
        embedding_rows = np.asarray([embedding_map[str(value)] for value in sample_ids], dtype=np.int64)

        train_matrix_rows = rna_rows[split_indices["train"]]
        variances = _scan_train_variance(
            rna_handle[config.rna.values_key], train_matrix_rows, config.analysis.chunk_size
        )
        number = min(config.analysis.reconstruction_gene_count, len(variances))
        selected_gene_indices = np.argsort(variances, kind="stable")[-number:]
        selected_gene_indices = selected_gene_indices[np.argsort(variances[selected_gene_indices])[::-1]]
        rna_selected = _load_selected_columns(
            rna_handle[config.rna.values_key], rna_rows, selected_gene_indices, config.analysis.chunk_size
        )
        embeddings = {
            name: h5_rows(embedding_handle[key], embedding_rows)
            for name, key in config.embeddings.layers.items()
        }

    for name, values in embeddings.items():
        if values.ndim != 2 or len(values) != len(sample_ids):
            raise ValueError(f"invalid shape for embedding layer {name}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"embedding layer {name} contains non-finite values")
    return AlignedData(
        sample_ids=sample_ids,
        cancer_types=cancer_types,
        splits=splits,
        split_indices=split_indices,
        rna_rows=rna_rows,
        embedding_rows=embedding_rows,
        gene_ids=gene_ids,
        selected_gene_indices=selected_gene_indices,
        selected_gene_ids=gene_ids[selected_gene_indices],
        rna_selected=rna_selected,
        embeddings=embeddings,
    )


def load_stability_view(
    spec: StabilityViewSpec,
    base_sample_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    with h5py.File(spec.path, "r") as handle:
        if spec.row_ids_key not in handle:
            raise KeyError(f"stability view {spec.name!r} lacks {spec.row_ids_key!r}")
        mapping = unique_index(decode_ids(handle[spec.row_ids_key][...]), f"{spec.name} sample ID")
        missing = [str(value) for value in base_sample_ids if str(value) not in mapping]
        if missing:
            raise ValueError(f"stability view {spec.name!r} misses {len(missing)} samples")
        rows = np.asarray([mapping[str(value)] for value in base_sample_ids], dtype=np.int64)
        output: dict[str, np.ndarray] = {}
        for name, key in spec.layers.items():
            if key not in handle:
                raise KeyError(f"stability view {spec.name!r} lacks layer dataset {key!r}")
            output[name] = h5_rows(handle[key], rows)
        return output


@dataclass(slots=True)
class TokenData:
    sample_ids: np.ndarray
    gene_ids: np.ndarray
    expression: np.ndarray
    token_ids: np.ndarray
    layers: dict[str, np.ndarray]
    pooled_layers: dict[str, np.ndarray]


def load_token_data(spec: TokenEmbeddingSpec) -> TokenData:
    with h5py.File(spec.path, "r") as handle:
        required = {spec.sample_ids_key, spec.gene_ids_key, spec.expression_key, spec.token_ids_key, *spec.layers.values()}
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise KeyError(f"token HDF5 is missing datasets: {missing}")
        sample_ids = decode_ids(handle[spec.sample_ids_key][...])
        gene_ids = decode_ids(handle[spec.gene_ids_key][...])
        unique_index(sample_ids, "token sample ID")
        unique_index(gene_ids, "token gene ID")
        expression = np.asarray(handle[spec.expression_key][...], dtype=np.float32)
        token_ids = np.asarray(handle[spec.token_ids_key][...], dtype=np.float32)
        layers = {name: np.asarray(handle[key][...], dtype=np.float32) for name, key in spec.layers.items()}
    expected = (len(sample_ids), len(gene_ids))
    if expression.shape != expected or token_ids.shape != expected:
        raise ValueError("token expression/token-ID arrays do not match sample × gene axes")
    for name, values in layers.items():
        if values.ndim != 3 or values.shape[:2] != expected:
            raise ValueError(f"token layer {name} has invalid shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"token layer {name} contains non-finite values")
    return TokenData(sample_ids, gene_ids, expression, token_ids, layers, {})
