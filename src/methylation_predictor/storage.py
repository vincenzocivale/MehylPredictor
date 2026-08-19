"""Small mmap-backed caches used by the canonical training/evaluation paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class SortedIndex:
    """Compact int64 ID -> row-position map using sort/searchsorted."""

    def __init__(self, ids: np.ndarray, name: str = "ids") -> None:
        self.ids = np.asarray(ids, dtype=np.int64)
        self.name = name
        self.order = np.argsort(self.ids, kind="mergesort")
        self.sorted_ids = self.ids[self.order]
        if len(self.sorted_ids) and np.any(self.sorted_ids[1:] == self.sorted_ids[:-1]):
            raise ValueError(f"{name} contains duplicate ids")

    def contains(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.int64)
        pos = np.searchsorted(self.sorted_ids, query)
        ok = pos < len(self.sorted_ids)
        if len(self.sorted_ids):
            clipped = np.minimum(pos, len(self.sorted_ids) - 1)
            ok &= self.sorted_ids[clipped] == query
        return ok

    def positions_of(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.int64)
        if not len(self.sorted_ids):
            raise KeyError(f"{self.name} is empty")
        pos = np.searchsorted(self.sorted_ids, query)
        clipped = np.minimum(pos, len(self.sorted_ids) - 1)
        ok = (pos < len(self.sorted_ids)) & (self.sorted_ids[clipped] == query)
        if not np.all(ok):
            missing = query[~ok]
            raise KeyError(f"{self.name} missing {len(missing)} ids; examples={missing[:10].tolist()}")
        return self.order[pos]


def read_h5_rows(dataset, rows: np.ndarray, *, dtype=np.float32) -> np.ndarray:
    """Read unordered/duplicated HDF5 rows and restore caller order."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return np.empty((0, dataset.shape[1]), dtype=dtype)
    unique, inverse = np.unique(rows, return_inverse=True)
    return np.asarray(dataset[unique, :], dtype=dtype)[inverse]


class RNACache:
    """Frozen z-scored canonical RNA cache."""

    def __init__(self, cache_dir: str | Path):
        root = Path(cache_dir)
        self.values = np.load(root / "rna_zscore.f16.npy", mmap_mode="r")
        self.ids = np.load(root / "rna_sample_idx.npy", mmap_mode="r")
        self.index = SortedIndex(self.ids, "RNA cache")

    def rows(self, sample_idx: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
        rows = self.index.positions_of(np.asarray(sample_idx, np.int64))
        return np.asarray(self.values[rows], dtype=dtype)


class LocusFeatureCache:
    """Frozen CpG embeddings plus prior mean/sigma used by the RNA model.

    Contract:
      * ``cpg_idx.npy`` global IDs
      * ``embeddings.f16.npy`` [CpGs, 1536]
      * ``prior.npy`` beta-space mean in (0,1)
      * ``sigma.npy`` logit-space residual scale
    """

    def __init__(self, root: str | Path):
        root = Path(root)
        self.root = root
        self.ids = np.load(root / "cpg_idx.npy", mmap_mode="r")
        self.embeddings = np.load(root / "embeddings.f16.npy", mmap_mode="r")
        self.prior = np.load(root / "prior.npy", mmap_mode="r")
        self.sigma = np.load(root / "sigma.npy", mmap_mode="r")
        if self.embeddings.shape != (len(self.ids), 1536):
            raise ValueError(f"unexpected locus embedding shape {self.embeddings.shape}")
        if self.prior.shape != (len(self.ids),) or self.sigma.shape != (len(self.ids),):
            raise ValueError("prior/sigma do not align with cpg_idx")
        if not np.isfinite(self.prior).all() or not np.isfinite(self.sigma).all():
            raise ValueError("feature cache contains non-finite prior/sigma")
        if np.any(self.prior <= 0) or np.any(self.prior >= 1) or np.any(self.sigma <= 0):
            raise ValueError("feature cache requires prior in (0,1) and sigma > 0")
        self.index = SortedIndex(self.ids, "locus feature cache")

    def get(
        self,
        cpg_idx: np.ndarray,
        *,
        embedding_dtype: np.dtype = np.float32,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self.index.positions_of(np.asarray(cpg_idx, np.int64))
        return (
            np.asarray(self.embeddings[rows], dtype=embedding_dtype),
            np.asarray(self.prior[rows], dtype=np.float32),
            np.asarray(self.sigma[rows], dtype=np.float32),
        )
