"""Leakage-safe IncrementalPCA preprocessing for RNA baselines."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from sklearn.decomposition import IncrementalPCA

from .config import RunConfig
from .data import load_bundle
from .utils import write_json


def fit_pca(
    config: RunConfig,
    output_h5: str | Path,
    n_components: int = 256,
    batch_size: int = 512,
) -> dict[str, object]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("h5py is required to write the PCA matrix") from exc

    bundle = load_bundle(config.data, seed=config.training.seed)
    try:
        if config.data.rna_control != "real":
            raise ValueError("PCA must be fitted with data.rna_control=real")
        train_indices = bundle.sample_indices(config.training.train_sample_split)
        if len(train_indices) < n_components:
            raise ValueError("training samples must be at least n_components")
        effective_batch = max(batch_size, n_components)
        pca = IncrementalPCA(n_components=n_components, batch_size=effective_batch)

        for start in range(0, len(train_indices), effective_batch):
            rows = train_indices[start : start + effective_batch]
            if len(rows) < n_components and start > 0:
                # IncrementalPCA cannot partial_fit a final undersized batch.
                rows = train_indices[-effective_batch:]
            pca.partial_fit(bundle.rna(rows))
            if start + effective_batch >= len(train_indices):
                break

        output_h5 = Path(output_h5)
        output_h5.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype("utf-8")
        with h5py.File(output_h5, "w") as handle:
            dataset = handle.create_dataset(
                "X",
                shape=(len(bundle.samples.ids), n_components),
                dtype="float32",
                chunks=(min(effective_batch, len(bundle.samples.ids)), n_components),
            )
            for start in range(0, len(bundle.samples.ids), effective_batch):
                rows = np.arange(start, min(start + effective_batch, len(bundle.samples.ids)))
                dataset[start : start + len(rows)] = pca.transform(bundle.rna(rows)).astype(np.float32)
            handle.create_dataset(
                "sample_idx",
                data=np.asarray([str(v) for v in bundle.samples.ids], dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "component_ids",
                data=np.asarray([f"PC{i + 1}" for i in range(n_components)], dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset("components", data=pca.components_.astype(np.float32), compression="gzip")
            handle.create_dataset("explained_variance_ratio", data=pca.explained_variance_ratio_)
            if bundle.standardizer is not None:
                handle.create_dataset("training_mean", data=bundle.standardizer.mean, compression="gzip")
                handle.create_dataset("training_scale", data=bundle.standardizer.scale, compression="gzip")

        result = {
            "output": str(output_h5),
            "n_components": n_components,
            "training_samples": int(len(train_indices)),
            "all_samples": int(len(bundle.samples.ids)),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "fit_split": config.training.train_sample_split,
        }
        write_json(output_h5.with_suffix(".json"), result)
        return result
    finally:
        bundle.close()
