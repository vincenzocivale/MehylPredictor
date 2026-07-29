"""Frozen Gaussian random projection: a non-adaptive control for PCA compression.

Unlike `fit_pca`, the projection matrix does not depend on training data at all
(Johnson-Lindenstrauss random projection) -- only the RNA standardizer (already
train-only) is data-dependent. Provided purely as a same-dimensionality control
to isolate whether PCA's learned structure matters versus arbitrary linear
compression of the same rank.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import RunConfig
from .data import load_bundle
from .utils import write_json


def fit_random_projection(
    config: RunConfig,
    output_h5: str | Path,
    n_components: int = 256,
    seed: int = 20260730,
    batch_size: int = 512,
) -> dict[str, object]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("h5py is required to write the random-projection matrix") from exc

    bundle = load_bundle(config.data, seed=config.training.seed)
    try:
        if config.data.rna_control != "real":
            raise ValueError("random projection must be fitted with data.rna_control=real")
        input_dim = bundle.rna_store.shape[1]
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((input_dim, n_components)).astype(np.float32) / np.sqrt(n_components)

        output_h5 = Path(output_h5)
        output_h5.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype("utf-8")
        n_samples = len(bundle.samples.ids)
        with h5py.File(output_h5, "w") as handle:
            dataset = handle.create_dataset(
                "X",
                shape=(n_samples, n_components),
                dtype="float32",
                chunks=(min(batch_size, n_samples), n_components),
            )
            for start in range(0, n_samples, batch_size):
                rows = np.arange(start, min(start + batch_size, n_samples))
                dataset[start : start + len(rows)] = bundle.rna(rows) @ projection
            handle.create_dataset(
                "sample_idx",
                data=np.asarray([str(v) for v in bundle.samples.ids], dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "component_ids",
                data=np.asarray([f"RP{i + 1}" for i in range(n_components)], dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset("projection_matrix", data=projection, compression="gzip")
            handle.create_dataset("projection_seed", data=seed)

        result = {
            "output": str(output_h5),
            "n_components": n_components,
            "input_dim": int(input_dim),
            "all_samples": n_samples,
            "projection_seed": seed,
        }
        write_json(output_h5.with_suffix(".json"), result)
        return result
    finally:
        bundle.close()
