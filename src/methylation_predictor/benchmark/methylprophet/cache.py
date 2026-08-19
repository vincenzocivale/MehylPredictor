from __future__ import annotations

from pathlib import Path

import pyarrow  # noqa: F401 -- must precede h5py on target server
import h5py
import numpy as np

from .feature_store import SortedIndex


def prepare_rna_cache(
    canonical_root: str | Path,
    output_dir: str | Path,
    *,
    train_sample_idx: np.ndarray,
    chunk_rows: int = 64,
) -> dict[str, object]:
    """Fit z-score on official Array-train samples and mmap all canonical RNA rows."""
    root = Path(canonical_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    values_path = out / "rna_zscore.f16.npy"
    ids_path = out / "rna_sample_idx.npy"
    stats_path = out / "rna_stats.npz"
    done = out / ".done"
    if done.is_file() and all(p.is_file() for p in (values_path, ids_path, stats_path)):
        x = np.load(values_path, mmap_mode="r")
        return {"status": "cached", "shape": list(x.shape)}

    path = root / "rna" / "tcga_rna_official_full.h5"
    with h5py.File(path, "r") as h:
        X = h["X"]
        ids = np.asarray(h["sample_idx"][...], dtype=np.int64)
        idx = SortedIndex(ids, "canonical RNA sample_idx")
        train_pos = idx.positions_of(np.asarray(train_sample_idx, dtype=np.int64))
        train_pos.sort()
        n_genes = X.shape[1]
        sx = np.zeros(n_genes, dtype=np.float64)
        sxx = np.zeros(n_genes, dtype=np.float64)
        n = 0
        for start in range(0, len(train_pos), chunk_rows):
            rows = train_pos[start:start + chunk_rows]
            block = np.asarray(X[rows, :], dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError("canonical RNA contains non-finite values in official train rows")
            sx += block.sum(axis=0, dtype=np.float64)
            sxx += np.square(block, dtype=np.float64).sum(axis=0)
            n += len(block)
        mean = sx / n
        var = np.maximum(sxx / n - mean * mean, 1e-8)
        std = np.sqrt(var)

        mmap = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float16, shape=X.shape)
        for start in range(0, X.shape[0], chunk_rows):
            stop = min(start + chunk_rows, X.shape[0])
            block = np.asarray(X[start:stop], dtype=np.float32)
            mmap[start:stop] = ((block - mean) / std).astype(np.float16)
        del mmap
    np.save(ids_path, ids)
    np.savez(stats_path, mean=mean.astype(np.float32), std=std.astype(np.float32), train_sample_idx=np.asarray(train_sample_idx, np.int64))
    done.write_text("ok\n")
    return {"status": "built", "shape": [int(len(ids)), int(n_genes)], "train_rows": int(n)}


class RNACache:
    def __init__(self, cache_dir: str | Path):
        root = Path(cache_dir)
        self.values = np.load(root / "rna_zscore.f16.npy", mmap_mode="r")
        self.ids = np.load(root / "rna_sample_idx.npy", mmap_mode="r")
        self.index = SortedIndex(self.ids, "RNA cache")

    def rows(self, sample_idx: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
        p = self.index.positions_of(np.asarray(sample_idx, dtype=np.int64))
        # dtype defaults to float32 (backward compatible with every existing
        # caller). Hot training loops that immediately upload to GPU can pass
        # dtype=np.float16 to skip this CPU-side cast entirely -- the stored
        # cache is already float16, so that call is a pure memory copy, and
        # the eventual float32 upcast is done on-GPU (near-free) instead.
        return np.asarray(self.values[p], dtype=dtype)
