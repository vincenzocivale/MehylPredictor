"""I/O and split-safe target construction for RNA-only readout training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ReadoutConfig


def _decode(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O", "U"}:
        return np.asarray([
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values
        ], dtype=object)
    return values.astype(str)




def _normalise_gene_id(value: object) -> str:
    fields = [field.strip() for field in str(value).split(";") if field.strip()]
    ensembl = next((field for field in fields if field.upper().startswith("ENSG")), None)
    if ensembl is not None:
        return ensembl.split(".", 1)[0]
    value = fields[0] if fields else str(value)
    return value.split(".", 1)[0] if value.upper().startswith("ENSG") else value

def _normalise_split(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"val", "valid", "validation"}:
        return "validation"
    if text in {"train", "training"}:
        return "train"
    if text in {"test", "testing"}:
        return "test"
    return text


@dataclass
class TargetBundle:
    sample_ids: np.ndarray
    splits: np.ndarray
    cancer_types: np.ndarray
    target_gene_ids: np.ndarray
    target_indices: np.ndarray
    total_targets: np.ndarray
    within_targets: np.ndarray
    input_gene_ids: np.ndarray
    input_variance: np.ndarray
    input_within_variance: np.ndarray
    target_train_mean: np.ndarray
    target_train_std: np.ndarray
    within_train_std: np.ndarray
    cancer_train_means: dict[str, np.ndarray]


@dataclass
class CacheMetadata:
    sample_ids: np.ndarray
    gene_ids: np.ndarray
    splits: np.ndarray
    cancer_types: np.ndarray
    expression: np.ndarray
    normalized_expression: np.ndarray | None
    token_ids: np.ndarray | None
    input_variance: np.ndarray
    input_within_variance: np.ndarray
    token_dim: int


class H5TokenDataset(Dataset):
    """Lazy HDF5 token dataset safe with DataLoader worker re-opening."""

    def __init__(
        self,
        cache_path: str,
        indices: np.ndarray,
        layers: list[int],
        key_template: str,
        total_targets: np.ndarray,
        within_targets: np.ndarray,
        expression_key: str,
        normalized_expression_key: str,
        token_ids_key: str,
        augmentation_path: str | None = None,
    ) -> None:
        self.cache_path = str(cache_path)
        self.augmentation_path = str(augmentation_path) if augmentation_path else None
        self.indices = np.asarray(indices, dtype=np.int64)
        self.layers = list(layers)
        self.key_template = key_template
        self.total_targets = np.asarray(total_targets[self.indices], dtype=np.float32)
        self.within_targets = np.asarray(within_targets[self.indices], dtype=np.float32)
        self.expression_key = expression_key
        self.normalized_expression_key = normalized_expression_key
        self.token_ids_key = token_ids_key
        self._handle: h5py.File | None = None
        self._aug_handle: h5py.File | None = None

    def _open(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.cache_path, "r")
        return self._handle

    def _open_aug(self) -> h5py.File | None:
        if self.augmentation_path is None:
            return None
        if self._aug_handle is None:
            self._aug_handle = h5py.File(self.augmentation_path, "r")
        return self._aug_handle

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        source_index = int(self.indices[item])
        handle = self._open()
        batch: dict[str, torch.Tensor] = {
            f"tokens_layer{layer}": torch.from_numpy(
                np.asarray(handle[self.key_template.format(layer=layer)][source_index], dtype=np.float32)
            )
            for layer in self.layers
        }
        for layer in self.layers:
            batch[f"full_mean_layer{layer}"] = torch.from_numpy(
                np.asarray(handle[f"full_mean_layer{layer}"][source_index], dtype=np.float32)
            )
        batch["expression"] = torch.from_numpy(
            np.asarray(handle[self.expression_key][source_index], dtype=np.float32)
        )
        if self.normalized_expression_key in handle:
            batch["normalized_expression"] = torch.from_numpy(
                np.asarray(handle[self.normalized_expression_key][source_index], dtype=np.float32)
            )
        if self.token_ids_key in handle:
            batch["token_ids"] = torch.from_numpy(
                np.asarray(handle[self.token_ids_key][source_index], dtype=np.int64)
            )
        aug = self._open_aug()
        if aug is not None:
            for layer in self.layers:
                batch[f"aug_tokens_layer{layer}"] = torch.from_numpy(
                    np.asarray(aug[self.key_template.format(layer=layer)][source_index], dtype=np.float32)
                )
        batch["target_total"] = torch.from_numpy(self.total_targets[item])
        batch["target_within"] = torch.from_numpy(self.within_targets[item])
        batch["source_index"] = torch.tensor(source_index, dtype=torch.long)
        return batch

    def __del__(self) -> None:
        for handle in (self._handle, self._aug_handle):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass


def _read_cache_metadata(config: ReadoutConfig) -> CacheMetadata:
    cache = config.token_cache
    with h5py.File(cache.path, "r") as handle:
        required = [cache.sample_ids_key, cache.gene_ids_key, cache.expression_key, cache.split_key, cache.cancer_key]
        missing = [key for key in required if key not in handle]
        for layer in cache.layers:
            key = cache.tokens_key_template.format(layer=layer)
            if key not in handle:
                missing.append(key)
            full_mean_key = f"full_mean_layer{layer}"
            if full_mean_key not in handle:
                missing.append(full_mean_key)
        if missing:
            raise KeyError(f"token cache is missing keys: {sorted(set(missing))}")
        sample_ids = _decode(np.asarray(handle[cache.sample_ids_key]))
        gene_ids = _decode(np.asarray(handle[cache.gene_ids_key]))
        splits = np.asarray([_normalise_split(v) for v in _decode(np.asarray(handle[cache.split_key]))], dtype=object)
        cancers = _decode(np.asarray(handle[cache.cancer_key]))
        expression = np.asarray(handle[cache.expression_key], dtype=np.float32)
        normalized = (
            np.asarray(handle[cache.normalized_expression_key], dtype=np.float32)
            if cache.normalized_expression_key in handle else None
        )
        token_ids = np.asarray(handle[cache.token_ids_key], dtype=np.int64) if cache.token_ids_key in handle else None
        first = handle[cache.tokens_key_template.format(layer=cache.layers[0])]
        token_dim = int(first.shape[-1])
        if first.shape[:2] != expression.shape:
            raise ValueError(f"token/expression shapes disagree: {first.shape} vs {expression.shape}")
        if len(sample_ids) != expression.shape[0] or len(gene_ids) != expression.shape[1]:
            raise ValueError("token cache identifiers do not match matrix dimensions")
        input_variance = (
            np.asarray(handle["gene_variance_log2p1_train"], dtype=np.float32)
            if "gene_variance_log2p1_train" in handle else np.var(expression[splits == "train"], axis=0)
        )
        if "gene_within_variance_log2p1_train" in handle:
            input_within_variance = np.asarray(handle["gene_within_variance_log2p1_train"], dtype=np.float32)
        else:
            residual = expression.copy()
            train = splits == "train"
            for cancer in np.unique(cancers[train]):
                mask = train & (cancers == cancer)
                residual[mask] -= residual[mask].mean(axis=0, keepdims=True)
            input_within_variance = np.var(residual[train], axis=0)
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample IDs in token cache")
    if cache.augmentation_path:
        with h5py.File(cache.augmentation_path, "r") as aug:
            for key in (cache.sample_ids_key, cache.gene_ids_key):
                if key not in aug:
                    raise KeyError(f"augmentation cache is missing {key!r}")
            aug_samples = _decode(np.asarray(aug[cache.sample_ids_key]))
            aug_genes = _decode(np.asarray(aug[cache.gene_ids_key]))
            if not np.array_equal(aug_samples.astype(str), sample_ids.astype(str)):
                raise ValueError("augmentation cache sample order differs from base cache")
            if not np.array_equal(aug_genes.astype(str), gene_ids.astype(str)):
                raise ValueError("augmentation cache gene order differs from base cache")
            for layer in cache.layers:
                key = cache.tokens_key_template.format(layer=layer)
                if key not in aug or aug[key].shape != first.shape:
                    raise ValueError(f"augmentation cache layer {key!r} is missing or shape-incompatible")
    return CacheMetadata(
        sample_ids=sample_ids,
        gene_ids=gene_ids,
        splits=splits,
        cancer_types=cancers,
        expression=expression,
        normalized_expression=normalized,
        token_ids=token_ids,
        input_variance=input_variance,
        input_within_variance=input_within_variance,
        token_dim=token_dim,
    )


def _stratified_target_indices_from_stats(
    variance: np.ndarray,
    prevalence: np.ndarray,
    gene_ids: np.ndarray,
    excluded: set[str],
    count: int,
    seed: int,
) -> np.ndarray:
    eligible = np.asarray([_normalise_gene_id(gene) not in excluded for gene in gene_ids]) & (prevalence > 0.01)
    candidates = np.flatnonzero(eligible)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} target genes remain after excluding input genes; requested {count}")
    order = candidates[np.argsort(variance[candidates])]
    bins = np.array_split(order, 4)
    quotas = [count // 8, count // 8, count // 4, count // 2]
    quotas[-1] += count - sum(quotas)
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for bin_index, (group, quota) in enumerate(zip(bins, quotas, strict=True)):
        if len(group) <= quota:
            chosen.append(group)
        elif bin_index == len(bins) - 1:
            chosen.append(group[-quota:])
        else:
            chosen.append(np.sort(rng.choice(group, size=quota, replace=False)))
    result = np.concatenate(chosen)
    if len(result) < count:
        remaining = np.setdiff1d(candidates, result, assume_unique=False)
        fill = remaining[np.argsort(variance[remaining])[-(count - len(result)):]]
        result = np.concatenate([result, fill])
    return np.sort(np.asarray(result[:count], dtype=np.int64))


def _stream_train_gene_stats(
    dataset: h5py.Dataset,
    train_source_rows: np.ndarray,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    variance = np.empty(dataset.shape[1], dtype=np.float32)
    prevalence = np.empty(dataset.shape[1], dtype=np.float32)
    rows = np.sort(np.asarray(train_source_rows, dtype=np.int64))
    for start in range(0, dataset.shape[1], chunk_size):
        stop = min(start + chunk_size, dataset.shape[1])
        values = np.asarray(dataset[rows, start:stop], dtype=np.float32)
        variance[start:stop] = np.var(values, axis=0)
        prevalence[start:stop] = np.mean(values > 0, axis=0)
    return variance, prevalence


def prepare_targets(config: ReadoutConfig, metadata: CacheMetadata) -> TargetBundle:
    train = metadata.splits == "train"
    if not train.any() or not (metadata.splits == "validation").any() or not (metadata.splits == "test").any():
        raise ValueError("cache must contain train, validation and test samples")
    with h5py.File(config.rna.path, "r") as handle:
        for key in (config.rna.values_key, config.rna.row_ids_key, config.rna.col_ids_key):
            if key not in handle:
                raise KeyError(f"RNA HDF5 missing key {key!r}")
        dataset = handle[config.rna.values_key]
        rna_sample_ids = _decode(np.asarray(handle[config.rna.row_ids_key]))
        rna_gene_ids = _decode(np.asarray(handle[config.rna.col_ids_key]))
        lookup = {str(sample): i for i, sample in enumerate(rna_sample_ids)}
        missing = [sample for sample in metadata.sample_ids if str(sample) not in lookup]
        if missing:
            raise ValueError(f"RNA matrix is missing {len(missing)} cache samples; first={missing[:5]}")
        aligned_rows = np.asarray([lookup[str(sample)] for sample in metadata.sample_ids], dtype=np.int64)
        variance, prevalence = _stream_train_gene_stats(dataset, aligned_rows[train])
        excluded = (
            set()
            if config.objective.input_target_overlap_allowed
            else {_normalise_gene_id(gene) for gene in metadata.gene_ids}
        )
        target_indices = _stratified_target_indices_from_stats(
            variance,
            prevalence,
            rna_gene_ids,
            excluded,
            config.objective.target_gene_count,
            config.objective.selection_seed,
        )
        # One fancy axis at a time keeps h5py access valid and limits memory to
        # n_samples × target_gene_count rather than the full 60k-gene matrix.
        selected_all_rows = np.asarray(dataset[:, target_indices], dtype=np.float32)
        target = selected_all_rows[aligned_rows]
    target_train_mean = target[train].mean(axis=0)
    target_train_std = np.maximum(target[train].std(axis=0), 1e-6)
    total = (target - target_train_mean) / target_train_std
    cancer_train_means: dict[str, np.ndarray] = {}
    residual = np.empty_like(target)
    global_train_mean = target_train_mean
    for cancer in np.unique(metadata.cancer_types):
        train_mask = train & (metadata.cancer_types == cancer)
        cancer_train_means[str(cancer)] = target[train_mask].mean(axis=0) if train_mask.any() else global_train_mean
        residual[metadata.cancer_types == cancer] = target[metadata.cancer_types == cancer] - cancer_train_means[str(cancer)]
    within_train_std = np.maximum(residual[train].std(axis=0), 1e-6)
    within = residual / within_train_std
    return TargetBundle(
        sample_ids=metadata.sample_ids,
        splits=metadata.splits,
        cancer_types=metadata.cancer_types,
        target_gene_ids=rna_gene_ids[target_indices],
        target_indices=target_indices,
        total_targets=total.astype(np.float32),
        within_targets=within.astype(np.float32),
        input_gene_ids=metadata.gene_ids,
        input_variance=metadata.input_variance,
        input_within_variance=metadata.input_within_variance,
        target_train_mean=target_train_mean.astype(np.float32),
        target_train_std=target_train_std.astype(np.float32),
        within_train_std=within_train_std.astype(np.float32),
        cancer_train_means=cancer_train_means,
    )


def load_data(config: ReadoutConfig) -> tuple[CacheMetadata, TargetBundle, dict[str, H5TokenDataset]]:
    metadata = _read_cache_metadata(config)
    if metadata.token_dim != config.model.output_dim:
        raise ValueError(
            f"cache token dim={metadata.token_dim} but model.output_dim={config.model.output_dim}; "
            "the current residual poolers require equal dimensions"
        )
    targets = prepare_targets(config, metadata)
    dataset_layers = (
        list(config.token_cache.layers)
        if config.model.kind in {"scalar_mix", "concat_layer_means"}
        else [config.model.primary_layer]
    )
    datasets: dict[str, H5TokenDataset] = {}
    for split in ("train", "validation", "test"):
        indices = np.flatnonzero(metadata.splits == split)
        datasets[split] = H5TokenDataset(
            cache_path=config.token_cache.path,
            indices=indices,
            layers=dataset_layers,
            key_template=config.token_cache.tokens_key_template,
            total_targets=targets.total_targets,
            within_targets=targets.within_targets,
            expression_key=config.token_cache.expression_key,
            normalized_expression_key=config.token_cache.normalized_expression_key,
            token_ids_key=config.token_cache.token_ids_key,
            augmentation_path=config.token_cache.augmentation_path,
        )
    return metadata, targets, datasets


def iter_h5_batches(path: str, key: str, batch_size: int = 32) -> Iterator[tuple[int, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        dataset = handle[key]
        for start in range(0, len(dataset), batch_size):
            yield start, np.asarray(dataset[start:start + batch_size])
