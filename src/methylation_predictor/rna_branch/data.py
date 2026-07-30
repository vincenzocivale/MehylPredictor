"""Data backends and leakage-safe alignment for RNA-branch experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import DataConfig, MatrixConfig, TableConfig


def _decode_ids(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        decoded = []
        for value in values.tolist():
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            decoded.append(str(value))
        return np.asarray(decoded, dtype=object)
    return values.astype(str)


def read_table(config: TableConfig) -> pd.DataFrame:
    path = Path(config.path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t" if suffix in {".tsv", ".txt"} else ",")
    raise ValueError(f"unsupported table format: {path}")


class MatrixStore:
    """Read-only 2-D matrix with stable row/column identifiers."""

    def __init__(self, config: MatrixConfig):
        self.config = config
        self.path = Path(config.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._kind = self.path.suffix.lower()
        self._npz = None
        self._h5 = None
        self._array = None

        if self._kind == ".npz":
            self._npz = np.load(self.path, allow_pickle=False, mmap_mode="r")
            self._array = self._npz[config.values_key]
            self.row_ids = _decode_ids(self._npz[config.row_ids_key])
            self.col_ids = (
                _decode_ids(self._npz[config.col_ids_key]) if config.col_ids_key else None
            )
        elif self._kind == ".npy":
            if not config.row_ids_key:
                raise ValueError(".npy matrices require row_ids_key pointing to a companion .npy file")
            self._array = np.load(self.path, mmap_mode="r")
            self.row_ids = _decode_ids(np.load(config.row_ids_key, allow_pickle=False))
            self.col_ids = (
                _decode_ids(np.load(config.col_ids_key, allow_pickle=False))
                if config.col_ids_key
                else None
            )
        elif self._kind in {".h5", ".hdf5"}:
            try:
                import h5py
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError("h5py is required for HDF5 inputs") from exc
            self._h5 = h5py.File(self.path, "r")
            self._array = self._h5[config.values_key]
            self.row_ids = _decode_ids(self._h5[config.row_ids_key][...])
            self.col_ids = (
                _decode_ids(self._h5[config.col_ids_key][...]) if config.col_ids_key else None
            )
        else:
            raise ValueError(f"unsupported matrix format: {self.path}")

        if self._array.ndim != 2:
            raise ValueError(f"expected a two-dimensional matrix in {self.path}")
        if len(self.row_ids) != self._array.shape[0]:
            raise ValueError(f"row IDs do not match matrix rows in {self.path}")
        if self.col_ids is not None and len(self.col_ids) != self._array.shape[1]:
            raise ValueError(f"column IDs do not match matrix columns in {self.path}")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._array.shape)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def _read_rows_native(self, row_indices: np.ndarray) -> np.ndarray:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if not len(row_indices):
            return np.empty((0, self.shape[1]), dtype=np.float32)
        unique_rows, inverse_unique = np.unique(row_indices, return_inverse=True)
        values = np.asarray(self._array[unique_rows, :])
        return values[inverse_unique]

    def rows(self, row_indices: Sequence[int], col_indices: Sequence[int] | None = None) -> np.ndarray:
        values = self._read_rows_native(np.asarray(row_indices, dtype=np.int64))
        if col_indices is not None:
            values = values[:, np.asarray(col_indices, dtype=np.int64)]
        return np.asarray(values, dtype=np.float32)

    def block(self, row_indices: Sequence[int], col_indices: Sequence[int]) -> np.ndarray:
        return self.rows(row_indices, col_indices)


@dataclass(slots=True)
class RNAStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    @classmethod
    def fit(
        cls,
        store: MatrixStore,
        rows: np.ndarray,
        chunk_size: int = 128,
    ) -> "RNAStandardizer":
        rows = np.asarray(rows, dtype=np.int64)
        if not len(rows):
            raise ValueError("cannot fit RNA standardizer without training samples")
        total = np.zeros(store.shape[1], dtype=np.float64)
        total_sq = np.zeros(store.shape[1], dtype=np.float64)
        count = 0
        for start in range(0, len(rows), chunk_size):
            chunk = store.rows(rows[start : start + chunk_size]).astype(np.float64)
            finite = np.isfinite(chunk)
            total += np.where(finite, chunk, 0.0).sum(axis=0)
            total_sq += np.where(finite, chunk * chunk, 0.0).sum(axis=0)
            count += finite.sum(axis=0)
        safe_count = np.maximum(count, 1)
        mean = total / safe_count
        variance = np.maximum(total_sq / safe_count - mean * mean, 0.0)
        scale = np.sqrt(variance)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))


@dataclass(slots=True)
class LocusData:
    ids: np.ndarray
    embeddings: np.ndarray
    prior: np.ndarray
    variability: np.ndarray
    splits: np.ndarray


@dataclass(slots=True)
class SampleData:
    ids: np.ndarray
    cancer_types: np.ndarray
    splits: np.ndarray
    rna_rows: np.ndarray
    methylation_rows: np.ndarray


@dataclass(slots=True)
class DataBundle:
    config: DataConfig
    rna_store: MatrixStore
    methylation_store: MatrixStore
    samples: SampleData
    loci: LocusData
    methylation_cols: np.ndarray
    standardizer: RNAStandardizer | None
    control_row_map: np.ndarray
    cancer_type_names: np.ndarray
    cancer_type_codes: np.ndarray

    @property
    def rna_input_dim(self) -> int:
        if self.config.rna_control == "cancer_type_only":
            return len(self.cancer_type_names)
        return self.rna_store.shape[1]

    @property
    def locus_dim(self) -> int:
        return self.loci.embeddings.shape[1]

    def sample_indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.samples.splits == split)

    def cpg_indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.loci.splits == split)

    def training_sample_pool(self, split: str) -> np.ndarray:
        """Sample indices for SGD sampling, nested-subsampled per `train_sample_fraction`.

        Stratified by cancer_type, one shared RNG consumed in a fixed sorted
        group order regardless of fraction -- the same fraction_seed always
        produces the identical per-group permutation, so a smaller fraction's
        selection is a strict prefix of every larger fraction's (true nesting).
        """
        indices = self.sample_indices(split)
        fraction = self.config.train_sample_fraction
        if fraction >= 1.0:
            return indices
        cancer_types = self.samples.cancer_types[indices]
        rng = np.random.default_rng(self.config.train_sample_fraction_seed)
        selected = []
        for cancer_type in sorted(np.unique(cancer_types).tolist()):
            group = indices[cancer_types == cancer_type]
            order = rng.permutation(len(group))
            k = max(1, int(np.ceil(fraction * len(group))))
            selected.append(group[order[:k]])
        return np.sort(np.concatenate(selected))

    def training_cpg_pool(self, split: str) -> np.ndarray:
        """CpG indices for SGD sampling, nested-subsampled per `train_cpg_fraction`.

        Stratified by variability tertile (computed fresh over just this split,
        independent of any trainer-side tertile state) using the same
        fixed-order-RNG nesting trick as `training_sample_pool`.
        """
        indices = self.cpg_indices(split)
        fraction = self.config.train_cpg_fraction
        if fraction >= 1.0:
            return indices
        proxy = np.exp(self.loci.variability[indices, 0]) + np.exp(self.loci.variability[indices, 1])
        thresholds = np.quantile(proxy, [1.0 / 3.0, 2.0 / 3.0])
        tertiles = np.digitize(proxy, thresholds, right=True)
        rng = np.random.default_rng(self.config.train_cpg_fraction_seed)
        selected = []
        for tertile in (0, 1, 2):
            group = indices[tertiles == tertile]
            if not len(group):
                continue
            order = rng.permutation(len(group))
            k = max(1, int(np.ceil(fraction * len(group))))
            selected.append(group[order[:k]])
        return np.sort(np.concatenate(selected))

    def rna(self, sample_indices: Sequence[int]) -> np.ndarray:
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        if self.config.rna_control == "cancer_type_only":
            values = np.zeros((len(sample_indices), len(self.cancer_type_names)), dtype=np.float32)
            values[np.arange(len(sample_indices)), self.cancer_type_codes[sample_indices]] = 1.0
            return values
        if self.config.rna_control == "mean":
            return np.zeros((len(sample_indices), self.rna_store.shape[1]), dtype=np.float32)
        mapped = self.control_row_map[sample_indices]
        values = self.rna_store.rows(self.samples.rna_rows[mapped])
        if self.standardizer is not None:
            values = self.standardizer.transform(values)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values.astype(np.float32, copy=False)

    def beta(self, sample_indices: Sequence[int], cpg_indices: Sequence[int]) -> np.ndarray:
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        cpg_indices = np.asarray(cpg_indices, dtype=np.int64)
        rows = self.samples.methylation_rows[sample_indices]
        cols = self.methylation_cols[cpg_indices]
        return self.methylation_store.block(rows, cols)

    def close(self) -> None:
        self.rna_store.close()
        self.methylation_store.close()


def _unique_mapping(ids: np.ndarray, name: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, value in enumerate(ids.tolist()):
        key = str(value)
        if key in mapping:
            raise ValueError(f"duplicate {name}: {key}")
        mapping[key] = index
    return mapping


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"missing columns in {source}: {missing}")


def _control_permutation(
    control: str,
    cancer_types: np.ndarray,
    splits: np.ndarray,
    seed: int,
) -> np.ndarray:
    n = len(cancer_types)
    result = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if control in {"real", "mean", "cancer_type_only"}:
        return result
    if control == "shuffle_global":
        for split in np.unique(splits):
            indices = np.flatnonzero(splits == split)
            result[indices] = rng.permutation(indices)
        return result
    if control == "shuffle_within_cancer":
        for split in np.unique(splits):
            for cancer_type in np.unique(cancer_types):
                indices = np.flatnonzero((splits == split) & (cancer_types == cancer_type))
                if len(indices):
                    result[indices] = rng.permutation(indices)
        return result
    raise ValueError(f"unknown rna_control: {control}")


def load_bundle(config: DataConfig, seed: int = 17) -> DataBundle:
    rna_store = MatrixStore(config.rna)
    methylation_store = MatrixStore(config.methylation)
    embedding_store = MatrixStore(config.locus_embeddings)
    try:
        if methylation_store.col_ids is None:
            raise ValueError("methylation matrix requires col_ids_key")
        sample_meta = read_table(config.sample_metadata).copy()
        cpg_splits = read_table(config.cpg_splits).copy()
        locus_features = read_table(config.locus_features).copy()

        _require_columns(
            sample_meta,
            [config.sample_id_column, config.cancer_type_column, config.split_column],
            config.sample_metadata.path,
        )
        _require_columns(cpg_splits, [config.cpg_id_column, config.split_column], config.cpg_splits.path)
        _require_columns(
            locus_features,
            [
                config.cpg_id_column,
                config.prior_column,
                config.variability_between_column,
                config.variability_within_column,
            ],
            config.locus_features.path,
        )

        sample_meta[config.sample_id_column] = sample_meta[config.sample_id_column].astype(str)
        cpg_splits[config.cpg_id_column] = cpg_splits[config.cpg_id_column].astype(str)
        locus_features[config.cpg_id_column] = locus_features[config.cpg_id_column].astype(str)

        if sample_meta[config.sample_id_column].duplicated().any():
            raise ValueError("sample metadata contains duplicate sample IDs")
        if cpg_splits[config.cpg_id_column].duplicated().any():
            raise ValueError("CpG split manifest contains duplicate CpG IDs")
        if locus_features[config.cpg_id_column].duplicated().any():
            raise ValueError("locus feature table contains duplicate CpG IDs")

        rna_map = _unique_mapping(rna_store.row_ids, "RNA sample ID")
        beta_sample_map = _unique_mapping(methylation_store.row_ids, "methylation sample ID")
        beta_cpg_map = _unique_mapping(methylation_store.col_ids, "methylation CpG ID")
        embedding_map = _unique_mapping(embedding_store.row_ids, "embedding CpG ID")

        sample_rows = []
        missing_samples = []
        for sample_id in sample_meta[config.sample_id_column].tolist():
            if sample_id not in rna_map or sample_id not in beta_sample_map:
                missing_samples.append(sample_id)
                continue
            sample_rows.append((sample_id, rna_map[sample_id], beta_sample_map[sample_id]))
        if missing_samples and not config.allow_partial_overlap:
            raise ValueError(f"{len(missing_samples)} metadata samples are missing from RNA or methylation matrices")
        if missing_samples:
            sample_meta = sample_meta[~sample_meta[config.sample_id_column].isin(missing_samples)].copy()
        sample_lookup = {sample_id: (rna_row, beta_row) for sample_id, rna_row, beta_row in sample_rows}

        merged_loci = cpg_splits.merge(
            locus_features,
            on=config.cpg_id_column,
            how="inner" if config.allow_partial_overlap else "left",
            suffixes=("_split", "_features"),
            validate="one_to_one",
        )
        split_column = f"{config.split_column}_split" if f"{config.split_column}_split" in merged_loci else config.split_column
        missing_loci = [
            cpg_id
            for cpg_id in merged_loci[config.cpg_id_column].tolist()
            if cpg_id not in beta_cpg_map or cpg_id not in embedding_map
        ]
        required_values_missing = merged_loci[
            [config.prior_column, config.variability_between_column, config.variability_within_column]
        ].isna().any(axis=1)
        if (missing_loci or required_values_missing.any()) and not config.allow_partial_overlap:
            raise ValueError(
                f"locus contract incomplete: {len(missing_loci)} missing matrices, "
                f"{int(required_values_missing.sum())} missing feature rows"
            )
        if config.allow_partial_overlap:
            valid_ids = set(beta_cpg_map) & set(embedding_map)
            merged_loci = merged_loci[
                merged_loci[config.cpg_id_column].isin(valid_ids) & ~required_values_missing
            ].copy()

        sample_ids = sample_meta[config.sample_id_column].to_numpy(object)
        cancer_types = sample_meta[config.cancer_type_column].astype(str).to_numpy(object)
        sample_splits = sample_meta[config.split_column].astype(str).to_numpy(object)
        rna_rows = np.asarray([sample_lookup[x][0] for x in sample_ids], dtype=np.int64)
        beta_rows = np.asarray([sample_lookup[x][1] for x in sample_ids], dtype=np.int64)

        cpg_ids = merged_loci[config.cpg_id_column].astype(str).to_numpy(object)
        beta_cols = np.asarray([beta_cpg_map[x] for x in cpg_ids], dtype=np.int64)
        embedding_rows = np.asarray([embedding_map[x] for x in cpg_ids], dtype=np.int64)
        embeddings = embedding_store.rows(embedding_rows)
        prior = merged_loci[config.prior_column].to_numpy(np.float32)
        if np.any(~np.isfinite(prior)) or np.any((prior <= 0) | (prior >= 1)):
            epsilon = config.clip_beta_epsilon
            prior = np.clip(np.nan_to_num(prior, nan=0.5), epsilon, 1.0 - epsilon)
        variability = merged_loci[
            [config.variability_between_column, config.variability_within_column]
        ].to_numpy(np.float32)
        variability = np.nan_to_num(variability, nan=0.0, posinf=0.0, neginf=0.0)
        locus_splits = merged_loci[split_column].astype(str).to_numpy(object)

        names, codes = np.unique(cancer_types, return_inverse=True)
        sample_data = SampleData(sample_ids, cancer_types, sample_splits, rna_rows, beta_rows)
        locus_data = LocusData(cpg_ids, embeddings, prior, variability, locus_splits)
        train_rows = np.flatnonzero(sample_splits == "train")
        standardizer = None
        if config.standardize_rna and config.rna_control not in {"cancer_type_only", "mean"}:
            standardizer = RNAStandardizer.fit(rna_store, rna_rows[train_rows])
        control_map = _control_permutation(config.rna_control, cancer_types, sample_splits, seed)

        return DataBundle(
            config=config,
            rna_store=rna_store,
            methylation_store=methylation_store,
            samples=sample_data,
            loci=locus_data,
            methylation_cols=beta_cols,
            standardizer=standardizer,
            control_row_map=control_map,
            cancer_type_names=names,
            cancer_type_codes=codes.astype(np.int64),
        )
    except Exception:
        rna_store.close()
        methylation_store.close()
        raise
    finally:
        embedding_store.close()


def summarize_bundle(bundle: DataBundle) -> dict[str, object]:
    return {
        "rna_shape": list(bundle.rna_store.shape),
        "methylation_shape": list(bundle.methylation_store.shape),
        "aligned_samples": int(len(bundle.samples.ids)),
        "aligned_loci": int(len(bundle.loci.ids)),
        "rna_input_dim": int(bundle.rna_input_dim),
        "locus_embedding_dim": int(bundle.locus_dim),
        "sample_split_counts": {
            str(split): int((bundle.samples.splits == split).sum())
            for split in np.unique(bundle.samples.splits)
        },
        "cpg_split_counts": {
            str(split): int((bundle.loci.splits == split).sum())
            for split in np.unique(bundle.loci.splits)
        },
        "cancer_type_counts": {
            str(value): int((bundle.samples.cancer_types == value).sum())
            for value in np.unique(bundle.samples.cancer_types)
        },
        "rna_control": bundle.config.rna_control,
        "standardized": bundle.standardizer is not None,
    }
