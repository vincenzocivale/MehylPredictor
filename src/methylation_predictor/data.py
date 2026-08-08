"""Data backends and leakage-safe alignment for RNA-branch experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
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


def _stable_row_seed(base_seed: int, row_id: str) -> int:
    digest = hashlib.blake2b(str(row_id).encode("utf-8"), digest_size=8).digest()
    return (base_seed + int.from_bytes(digest, "little")) % (2**32)


def _rank_rows(values: np.ndarray) -> np.ndarray:
    """Within-sample average ranks in [0, 1] without gene-order tie leakage."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.shape[1] <= 1:
        return np.zeros_like(values)
    result = np.empty_like(values, dtype=np.float32)
    denominator = float(values.shape[1] - 1)
    for row_index, row in enumerate(values):
        _, inverse, counts = np.unique(row, return_inverse=True, return_counts=True)
        starts = np.cumsum(counts) - counts
        average_ranks = starts + (counts - 1) / 2.0
        result[row_index] = average_ranks[inverse] / denominator
    return result


def _methylprophet_quantize_rows(
    values: np.ndarray,
    row_ids: Sequence[str],
    num_bins: int,
    seed: int,
) -> np.ndarray:
    """Reproduce MethylProphet's per-sample non-zero quantile binning deterministically.

    The upstream implementation randomly resolves repeated quantile boundaries.  Here the
    same rule is seeded from the sample ID, making preprocessing invariant to batch order.
    """
    if num_bins < 3:
        raise ValueError("rna_quantile_bins must be >= 3")
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    result = np.zeros_like(values, dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, num_bins - 1)
    for index, (row, row_id) in enumerate(zip(values, row_ids)):
        mask = row != 0
        non_zero = row[mask]
        if not len(non_zero):
            continue
        boundaries = np.quantile(non_zero, quantiles)
        left = np.digitize(non_zero, boundaries, right=False)
        right = np.digitize(non_zero, boundaries, right=True)
        rng = np.random.default_rng(_stable_row_seed(seed, str(row_id)))
        assigned = np.ceil(left + (right - left) * rng.random(len(non_zero)))
        maximum = max(float(assigned.max(initial=1.0)), 1.0)
        result[index, mask] = assigned.astype(np.float32) / maximum
    return result


@dataclass(slots=True)
class RNAInputTransformer:
    kind: str
    standardizer: RNAStandardizer | None
    reference: np.ndarray
    quantile_bins: int = 51
    seed: int = 17

    @property
    def output_multiplier(self) -> int:
        return 2 if self.kind in {"continuous_rank", "continuous_binary"} else 1

    def transform(self, values: np.ndarray, row_ids: Sequence[str]) -> np.ndarray:
        raw = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        continuous = self.standardizer.transform(raw) if self.standardizer is not None else raw
        if self.kind in {"none", "zscore"}:
            return continuous.astype(np.float32, copy=False)
        if self.kind == "rank":
            return _rank_rows(raw)
        if self.kind == "methylprophet_quantile":
            return _methylprophet_quantize_rows(raw, row_ids, self.quantile_bins, self.seed)
        if self.kind == "continuous_rank":
            return np.concatenate([continuous, _rank_rows(raw)], axis=1).astype(np.float32, copy=False)
        if self.kind == "continuous_binary":
            return np.concatenate([continuous, (raw != 0).astype(np.float32)], axis=1)
        raise ValueError(f"unknown rna_transform: {self.kind}")

    @classmethod
    def fit(
        cls,
        store: MatrixStore,
        rows: np.ndarray,
        row_ids: Sequence[str],
        kind: str,
        quantile_bins: int,
        seed: int,
        chunk_size: int = 128,
    ) -> "RNAInputTransformer":
        if kind not in {"none", "zscore", "methylprophet_quantile", "rank", "continuous_rank", "continuous_binary"}:
            raise ValueError(f"unknown rna_transform: {kind}")
        rows = np.asarray(rows, dtype=np.int64)
        row_ids = np.asarray(row_ids, dtype=object)
        standardizer = RNAStandardizer.fit(store, rows) if kind in {"zscore", "continuous_rank", "continuous_binary"} else None
        width = store.shape[1] * (2 if kind in {"continuous_rank", "continuous_binary"} else 1)
        # For z-score the train mean is exactly the intended anchor.  Other transforms
        # need their train-only transformed mean, otherwise anchor_to_mean_rna silently
        # anchors to an all-zero/unexpressed pseudo-sample.
        if kind == "zscore":
            reference = np.zeros(width, dtype=np.float32)
            return cls(kind, standardizer, reference, quantile_bins, seed)
        provisional = cls(kind, standardizer, np.zeros(width, dtype=np.float32), quantile_bins, seed)
        total = np.zeros(width, dtype=np.float64)
        count = 0
        for start in range(0, len(rows), chunk_size):
            stop = start + chunk_size
            chunk = store.rows(rows[start:stop])
            transformed = provisional.transform(chunk, row_ids[start:stop])
            total += transformed.astype(np.float64).sum(axis=0)
            count += len(transformed)
        if count == 0:
            raise ValueError("cannot fit RNA transform without training samples")
        provisional.reference = (total / count).astype(np.float32)
        return provisional


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
    rna_transformer: RNAInputTransformer | None
    pretrained_store: MatrixStore | None
    pretrained_rows: np.ndarray | None
    pretrained_standardizer: RNAStandardizer | None
    pretrained_reference: np.ndarray | None
    control_row_map: np.ndarray
    pretrained_control_row_map: np.ndarray
    cancer_type_names: np.ndarray
    cancer_type_codes: np.ndarray
    gene_ids: np.ndarray | None = None
    gene_embeddings: np.ndarray | None = None

    @property
    def raw_rna_dim(self) -> int:
        """Width of the raw-RNA half of the input, before any pretrained embedding is
        concatenated -- i.e. the exact input width the canonical model was trained on.
        Used by ``pretrained_embedding`` encoders to split the concatenated input vector."""
        if self.config.rna_control == "cancer_type_only":
            return len(self.cancer_type_names)
        raw_dim = self.rna_store.shape[1]
        if self.rna_transformer is not None:
            raw_dim *= self.rna_transformer.output_multiplier
        return raw_dim

    @property
    def rna_input_dim(self) -> int:
        raw_dim = self.raw_rna_dim
        pretrained_dim = self.pretrained_store.shape[1] if self.pretrained_store is not None else 0
        if self.config.pretrained_mode == "replace":
            return pretrained_dim
        if self.config.pretrained_mode == "concat":
            return raw_dim + pretrained_dim
        return raw_dim

    @property
    def locus_dim(self) -> int:
        return self.loci.embeddings.shape[1]

    @property
    def gene_embedding_dim(self) -> int | None:
        if self.gene_embeddings is None:
            return None
        return int(self.gene_embeddings.shape[1])

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

    def reference_rna(self) -> np.ndarray:
        if self.config.rna_control == "cancer_type_only":
            return np.zeros(len(self.cancer_type_names), dtype=np.float32)
        raw_reference = (
            self.rna_transformer.reference
            if self.rna_transformer is not None
            else np.zeros(self.rna_store.shape[1], dtype=np.float32)
        )
        pre_reference = self.pretrained_reference
        if self.config.pretrained_mode == "replace":
            if pre_reference is None:
                raise ValueError("pretrained_mode=replace requires pretrained_rna")
            return pre_reference.astype(np.float32, copy=False)
        if self.config.pretrained_mode == "concat":
            if pre_reference is None:
                raise ValueError("pretrained_mode=concat requires pretrained_rna")
            return np.concatenate([raw_reference, pre_reference]).astype(np.float32, copy=False)
        return raw_reference.astype(np.float32, copy=False)

    def rna(self, sample_indices: Sequence[int]) -> np.ndarray:
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        if self.config.rna_control == "cancer_type_only":
            values = np.zeros((len(sample_indices), len(self.cancer_type_names)), dtype=np.float32)
            values[np.arange(len(sample_indices)), self.cancer_type_codes[sample_indices]] = 1.0
            return values
        if self.config.rna_control == "mean":
            return np.repeat(self.reference_rna()[None, :], len(sample_indices), axis=0)
        mapped = self.control_row_map[sample_indices]
        raw = self.rna_store.rows(self.samples.rna_rows[mapped])
        row_ids = self.samples.ids[mapped].astype(str)
        values = self.rna_transformer.transform(raw, row_ids) if self.rna_transformer is not None else raw
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        if self.pretrained_store is None or self.config.pretrained_mode == "none":
            return values
        assert self.pretrained_rows is not None
        pretrained_mapped = self.pretrained_control_row_map[sample_indices]
        pretrained = self.pretrained_store.rows(self.pretrained_rows[pretrained_mapped])
        if self.pretrained_standardizer is not None:
            pretrained = self.pretrained_standardizer.transform(pretrained)
        pretrained = np.nan_to_num(pretrained, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        if self.config.pretrained_mode == "replace":
            return pretrained
        if self.config.pretrained_mode == "concat":
            return np.concatenate([values, pretrained], axis=1)
        raise ValueError(f"unknown pretrained_mode: {self.config.pretrained_mode}")

    def beta(self, sample_indices: Sequence[int], cpg_indices: Sequence[int]) -> np.ndarray:
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        cpg_indices = np.asarray(cpg_indices, dtype=np.int64)
        rows = self.samples.methylation_rows[sample_indices]
        cols = self.methylation_cols[cpg_indices]
        return self.methylation_store.block(rows, cols)

    def close(self) -> None:
        self.rna_store.close()
        if self.pretrained_store is not None:
            self.pretrained_store.close()
        self.methylation_store.close()


def _unique_mapping(ids: np.ndarray, name: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, value in enumerate(ids.tolist()):
        key = str(value)
        if key in mapping:
            raise ValueError(f"duplicate {name}: {key}")
        mapping[key] = index
    return mapping


def align_gene_embeddings(
    rna_gene_ids: np.ndarray, gene_embedding_store: MatrixStore
) -> np.ndarray:
    """Return gene embeddings in exact RNA-column order, rejecting partial contracts."""
    gene_ids = _decode_ids(rna_gene_ids)
    mapping = _unique_mapping(gene_embedding_store.row_ids, "gene embedding ID")
    missing = [str(gene_id) for gene_id in gene_ids.tolist() if str(gene_id) not in mapping]
    if missing:
        raise ValueError(
            f"gene embedding contract incomplete: {len(missing)} RNA genes are missing; "
            f"examples={missing[:5]}"
        )
    rows = np.asarray([mapping[str(gene_id)] for gene_id in gene_ids.tolist()], dtype=np.int64)
    aligned = gene_embedding_store.rows(rows)
    if aligned.shape[0] != len(gene_ids):
        raise ValueError("aligned gene embeddings do not match RNA feature count")
    if not np.isfinite(aligned).all():
        raise ValueError("gene embedding matrix contains NaN or infinite values")
    return aligned


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
    if config.pretrained_mode not in {"none", "replace", "concat"}:
        raise ValueError(f"unknown pretrained_mode: {config.pretrained_mode}")
    if config.pretrained_mode != "none" and config.pretrained_rna is None:
        raise ValueError(f"pretrained_mode={config.pretrained_mode} requires data.pretrained_rna")
    rna_store = MatrixStore(config.rna)
    pretrained_store = (
        MatrixStore(config.pretrained_rna)
        if config.pretrained_mode != "none" and config.pretrained_rna is not None else None
    )
    methylation_store = MatrixStore(config.methylation)
    embedding_store = MatrixStore(config.locus_embeddings)
    gene_embedding_store = MatrixStore(config.gene_embeddings) if config.gene_embeddings is not None else None
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
        pretrained_map = (
            _unique_mapping(pretrained_store.row_ids, "pretrained RNA sample ID")
            if pretrained_store is not None else None
        )
        beta_sample_map = _unique_mapping(methylation_store.row_ids, "methylation sample ID")
        beta_cpg_map = _unique_mapping(methylation_store.col_ids, "methylation CpG ID")
        embedding_map = _unique_mapping(embedding_store.row_ids, "embedding CpG ID")

        gene_ids: np.ndarray | None = None
        aligned_gene_embeddings: np.ndarray | None = None
        if gene_embedding_store is not None:
            if rna_store.col_ids is None:
                raise ValueError("gene embeddings require RNA col_ids_key for exact gene alignment")
            gene_ids = _decode_ids(rna_store.col_ids)
            aligned_gene_embeddings = align_gene_embeddings(gene_ids, gene_embedding_store)

        sample_rows = []
        missing_samples = []
        for sample_id in sample_meta[config.sample_id_column].tolist():
            missing_pretrained = pretrained_map is not None and sample_id not in pretrained_map
            if sample_id not in rna_map or sample_id not in beta_sample_map or missing_pretrained:
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
        transform_kind = config.rna_transform or ("zscore" if config.standardize_rna else "none")
        rna_transformer = None
        if config.rna_control not in {"cancer_type_only"}:
            rna_transformer = RNAInputTransformer.fit(
                rna_store,
                rna_rows[train_rows],
                sample_ids[train_rows].astype(str),
                kind=transform_kind,
                quantile_bins=config.rna_quantile_bins,
                seed=config.rna_transform_seed,
            )
        pretrained_rows = None
        pretrained_standardizer = None
        pretrained_reference = None
        if pretrained_store is not None:
            assert pretrained_map is not None
            pretrained_rows = np.asarray([pretrained_map[x] for x in sample_ids], dtype=np.int64)
            if config.standardize_pretrained_rna:
                pretrained_standardizer = RNAStandardizer.fit(pretrained_store, pretrained_rows[train_rows])
                pretrained_reference = np.zeros(pretrained_store.shape[1], dtype=np.float32)
            else:
                total = np.zeros(pretrained_store.shape[1], dtype=np.float64)
                for start in range(0, len(train_rows), 128):
                    total += pretrained_store.rows(pretrained_rows[train_rows[start:start + 128]]).sum(axis=0)
                pretrained_reference = (total / max(len(train_rows), 1)).astype(np.float32)
        control_map = _control_permutation(config.rna_control, cancer_types, sample_splits, seed)
        pretrained_control_map = _control_permutation(
            config.pretrained_control,
            cancer_types,
            sample_splits,
            config.pretrained_control_seed,
        )

        return DataBundle(
            config=config,
            rna_store=rna_store,
            methylation_store=methylation_store,
            samples=sample_data,
            loci=locus_data,
            methylation_cols=beta_cols,
            rna_transformer=rna_transformer,
            pretrained_store=pretrained_store,
            pretrained_rows=pretrained_rows,
            pretrained_standardizer=pretrained_standardizer,
            pretrained_reference=pretrained_reference,
            control_row_map=control_map,
            pretrained_control_row_map=pretrained_control_map,
            cancer_type_names=names,
            cancer_type_codes=codes.astype(np.int64),
            gene_ids=gene_ids,
            gene_embeddings=aligned_gene_embeddings,
        )
    except Exception:
        rna_store.close()
        if pretrained_store is not None:
            pretrained_store.close()
        methylation_store.close()
        raise
    finally:
        embedding_store.close()
        if gene_embedding_store is not None:
            gene_embedding_store.close()


def summarize_bundle(bundle: DataBundle) -> dict[str, object]:
    return {
        "rna_shape": list(bundle.rna_store.shape),
        "methylation_shape": list(bundle.methylation_store.shape),
        "aligned_samples": int(len(bundle.samples.ids)),
        "aligned_loci": int(len(bundle.loci.ids)),
        "rna_input_dim": int(bundle.rna_input_dim),
        "locus_embedding_dim": int(bundle.locus_dim),
        "gene_embedding_shape": (list(bundle.gene_embeddings.shape) if bundle.gene_embeddings is not None else None),
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
        "rna_transform": bundle.rna_transformer.kind if bundle.rna_transformer is not None else None,
        "pretrained_mode": bundle.config.pretrained_mode,
        "pretrained_control": bundle.config.pretrained_control,
        "pretrained_shape": list(bundle.pretrained_store.shape) if bundle.pretrained_store is not None else None,
        "standardized": bool(
            bundle.rna_transformer is not None and bundle.rna_transformer.standardizer is not None
        ),
    }
