"""Configuration schema for RNA-encoder quality audits.

The schema deliberately contains no methylation matrix, genomic embedding, CpG
identifier, methylation loss, or downstream decoder.  This keeps the evaluation
focused on information represented by the transcriptomic encoder itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class MatrixSpec:
    path: str
    values_key: str
    row_ids_key: str
    col_ids_key: str | None = None


@dataclass(slots=True)
class EmbeddingSpec:
    path: str
    row_ids_key: str = "sample_idx"
    layers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MetadataSpec:
    path: str
    id_column: str = "sample_idx"
    cancer_column: str = "cancer_type"
    split_column: str = "split"
    train_values: tuple[str, ...] = ("train",)
    validation_values: tuple[str, ...] = ("validation", "val")
    test_values: tuple[str, ...] = ("test",)


@dataclass(slots=True)
class TokenEmbeddingSpec:
    path: str
    sample_ids_key: str = "sample_idx"
    gene_ids_key: str = "gene_ids"
    expression_key: str = "expression_log2p1"
    token_ids_key: str = "token_ids"
    layers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class StabilityViewSpec:
    name: str
    path: str
    row_ids_key: str = "sample_idx"
    layers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PerturbationViewSpec:
    name: str
    path: str
    row_ids_key: str = "sample_idx"
    layers: dict[str, str] = field(default_factory=dict)
    control: str | None = None


@dataclass(slots=True)
class AnalysisSpec:
    seed: int = 17
    chunk_size: int = 128
    reconstruction_gene_count: int = 4096
    pca_dimensions: tuple[int, ...] = (64, 256)
    random_projection_dim: int = 256
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    knn_k: int = 15
    pair_sample_count: int = 50_000
    geometry_splits: tuple[str, ...] = ("train", "validation", "test", "all")
    token_ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    token_global_pc_count: int = 8
    allow_partial_overlap: bool = False


@dataclass(slots=True)
class QualityConfig:
    run_name: str
    output_dir: str
    embeddings: EmbeddingSpec
    rna: MatrixSpec
    metadata: MetadataSpec
    analysis: AnalysisSpec = field(default_factory=AnalysisSpec)
    token_embeddings: TokenEmbeddingSpec | None = None
    stability_views: tuple[StabilityViewSpec, ...] = ()
    perturbation_views: tuple[PerturbationViewSpec, ...] = ()


def _tuple(value: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    if value is None:
        return default
    if isinstance(value, (str, bytes)):
        return (value,)
    return tuple(value)


def _matrix(data: dict[str, Any]) -> MatrixSpec:
    return MatrixSpec(
        path=str(data["path"]),
        values_key=str(data["values_key"]),
        row_ids_key=str(data["row_ids_key"]),
        col_ids_key=None if data.get("col_ids_key") is None else str(data["col_ids_key"]),
    )


def _embedding(data: dict[str, Any]) -> EmbeddingSpec:
    layers = {str(name): str(key) for name, key in dict(data.get("layers", {})).items()}
    return EmbeddingSpec(str(data["path"]), str(data.get("row_ids_key", "sample_idx")), layers)


def _metadata(data: dict[str, Any]) -> MetadataSpec:
    return MetadataSpec(
        path=str(data["path"]),
        id_column=str(data.get("id_column", "sample_idx")),
        cancer_column=str(data.get("cancer_column", "cancer_type")),
        split_column=str(data.get("split_column", "split")),
        train_values=_tuple(data.get("train_values"), ("train",)),
        validation_values=_tuple(data.get("validation_values"), ("validation", "val")),
        test_values=_tuple(data.get("test_values"), ("test",)),
    )


def _token(data: dict[str, Any] | None) -> TokenEmbeddingSpec | None:
    if data is None:
        return None
    return TokenEmbeddingSpec(
        path=str(data["path"]),
        sample_ids_key=str(data.get("sample_ids_key", "sample_idx")),
        gene_ids_key=str(data.get("gene_ids_key", "gene_ids")),
        expression_key=str(data.get("expression_key", "expression_log2p1")),
        token_ids_key=str(data.get("token_ids_key", "token_ids")),
        layers={str(name): str(key) for name, key in dict(data.get("layers", {})).items()},
    )


def _analysis(data: dict[str, Any] | None) -> AnalysisSpec:
    data = data or {}
    return AnalysisSpec(
        seed=int(data.get("seed", 17)),
        chunk_size=int(data.get("chunk_size", 128)),
        reconstruction_gene_count=int(data.get("reconstruction_gene_count", 4096)),
        pca_dimensions=tuple(int(x) for x in _tuple(data.get("pca_dimensions"), (64, 256))),
        random_projection_dim=int(data.get("random_projection_dim", 256)),
        ridge_alphas=tuple(float(x) for x in _tuple(data.get("ridge_alphas"), (0.01, 0.1, 1.0, 10.0, 100.0))),
        knn_k=int(data.get("knn_k", 15)),
        pair_sample_count=int(data.get("pair_sample_count", 50_000)),
        geometry_splits=tuple(str(x) for x in _tuple(data.get("geometry_splits"), ("train", "validation", "test", "all"))),
        token_ridge_alphas=tuple(float(x) for x in _tuple(data.get("token_ridge_alphas"), (0.01, 0.1, 1.0, 10.0))),
        token_global_pc_count=int(data.get("token_global_pc_count", 8)),
        allow_partial_overlap=bool(data.get("allow_partial_overlap", False)),
    )


def load_config(path: str | Path) -> QualityConfig:
    source = Path(path)
    payload = yaml.safe_load(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a YAML mapping: {source}")
    views = tuple(
        StabilityViewSpec(
            name=str(item["name"]),
            path=str(item["path"]),
            row_ids_key=str(item.get("row_ids_key", "sample_idx")),
            layers={str(name): str(key) for name, key in dict(item.get("layers", {})).items()},
        )
        for item in payload.get("stability_views", [])
    )
    perturbations = tuple(
        PerturbationViewSpec(
            name=str(item["name"]),
            path=str(item["path"]),
            row_ids_key=str(item.get("row_ids_key", "sample_idx")),
            layers={str(name): str(key) for name, key in dict(item.get("layers", {})).items()},
            control=None if item.get("control") is None else str(item["control"]),
        )
        for item in payload.get("perturbation_views", [])
    )
    config = QualityConfig(
        run_name=str(payload["run_name"]),
        output_dir=str(payload["output_dir"]),
        embeddings=_embedding(dict(payload["embeddings"])),
        rna=_matrix(dict(payload["rna"])),
        metadata=_metadata(dict(payload["metadata"])),
        analysis=_analysis(payload.get("analysis")),
        token_embeddings=_token(payload.get("token_embeddings")),
        stability_views=views,
        perturbation_views=perturbations,
    )
    validate_config(config)
    return config


def validate_config(config: QualityConfig) -> None:
    if not config.embeddings.layers:
        raise ValueError("embeddings.layers cannot be empty")
    if config.rna.col_ids_key is None:
        raise ValueError("rna.col_ids_key is required for train-only gene selection")
    if config.analysis.reconstruction_gene_count < 1:
        raise ValueError("analysis.reconstruction_gene_count must be positive")
    if config.analysis.chunk_size < 1:
        raise ValueError("analysis.chunk_size must be positive")
    if config.analysis.knn_k < 1:
        raise ValueError("analysis.knn_k must be positive")
    if not config.analysis.ridge_alphas or min(config.analysis.ridge_alphas) < 0:
        raise ValueError("analysis.ridge_alphas must contain non-negative values")
    if config.token_embeddings is not None and not config.token_embeddings.layers:
        raise ValueError("token_embeddings.layers cannot be empty")
    for view in config.stability_views:
        if not view.layers:
            raise ValueError(f"stability view {view.name!r} has no layers")
    perturbation_names = {view.name for view in config.perturbation_views}
    if len(perturbation_names) != len(config.perturbation_views):
        raise ValueError("perturbation view names must be unique")
    for view in config.perturbation_views:
        if not view.layers:
            raise ValueError(f"perturbation view {view.name!r} has no layers")
        if view.control is not None and view.control not in perturbation_names:
            raise ValueError(f"perturbation view {view.name!r} references unknown control {view.control!r}")
