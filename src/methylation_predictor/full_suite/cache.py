from __future__ import annotations

import json
import os
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


def build_compact_source_cache(
    canonical_root: str | Path,
    protocol: str,
    source_name: str,
    output_h5: str | Path,
    *,
    row_chunk: int = 16,
) -> dict[str, object]:
    """Materialize only protocol CpG columns for row-chunked Array/EPIC sources.

    WGBS is already column-major (32 x 8192 chunks) and should not be cached.
    The cache is read-only derived data outside the canonical root.
    """
    if source_name not in {"array", "epic"}:
        raise ValueError("compact cache is intended only for array/epic")
    from methylation_predictor.tcga_canonical import TCGACanonicalBundle, load_protocol

    output = Path(output_h5)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        with h5py.File(output, "r") as h:
            return {"status": "cached", "shape": list(h["beta"].shape)}

    with TCGACanonicalBundle.from_root(canonical_root) as bundle:
        p = load_protocol(protocol, bundle)
        src = bundle.sources[source_name]
        if source_name == "array":
            cpg_ids = np.unique(np.concatenate([p.array_train_cpg_idx, p.array_val_cpg_idx])).astype(np.int64)
        else:
            cpg_ids = np.asarray(p.auxiliary_cpg_idx[source_name], dtype=np.int64)
        col_pos = src._cpg_index.positions_of(cpg_ids)  # internal by design; ids stay global at API boundary
        order = np.argsort(col_pos)
        sorted_cols = col_pos[order]
        sorted_ids = cpg_ids[order]
        beta = src.h5["beta"]

        tmp = output.with_suffix(".tmp.h5")
        with h5py.File(tmp, "w") as h:
            ds_out = h.create_dataset(
                "beta",
                shape=(src.n_rows, len(sorted_ids)),
                dtype="f4",
                chunks=(1, min(8192, len(sorted_ids))),
            )
            for start in range(0, src.n_rows, row_chunk):
                stop = min(start + row_chunk, src.n_rows)
                # Array/EPIC are row-chunked; one full-row read here is exactly
                # what we want to pay once, rather than every SGD step.
                full = np.asarray(beta[start:stop, :], dtype=np.float32)
                ds_out[start:stop, :] = full[:, sorted_cols]
            h.create_dataset("cpg_idx", data=sorted_ids, dtype="i8")
            h.create_dataset("sample_idx", data=src.sample_idx, dtype="i8")
            h.create_dataset("measurement_idx", data=src.measurement_idx, dtype="i8")
            h.attrs["source"] = source_name
            h.attrs["protocol"] = protocol
        os.replace(tmp, output)
    return {"status": "built", "shape": [int(src.n_rows), int(len(sorted_ids))]}


class RNACache:
    def __init__(self, cache_dir: str | Path):
        root = Path(cache_dir)
        self.values = np.load(root / "rna_zscore.f16.npy", mmap_mode="r")
        self.ids = np.load(root / "rna_sample_idx.npy", mmap_mode="r")
        self.index = SortedIndex(self.ids, "RNA cache")

    def rows(self, sample_idx: np.ndarray) -> np.ndarray:
        p = self.index.positions_of(np.asarray(sample_idx, dtype=np.int64))
        return np.asarray(self.values[p], dtype=np.float32)


class CompactSource:
    """Read protocol-specific Array/EPIC cache with global cpg_idx semantics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.h5 = h5py.File(self.path, "r")
        self.beta = self.h5["beta"]
        self.cpg_idx = np.asarray(self.h5["cpg_idx"][...], dtype=np.int64)
        self.sample_idx = np.asarray(self.h5["sample_idx"][...], dtype=np.int64)
        self.measurement_idx = np.asarray(self.h5["measurement_idx"][...], dtype=np.int64)
        self.cpg_index = SortedIndex(self.cpg_idx, f"{self.path.name} CpGs")

    def block(self, row_positions: np.ndarray, cpg_idx: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_positions, dtype=np.int64)
        cols = self.cpg_index.positions_of(np.asarray(cpg_idx, dtype=np.int64))
        # Cache width is <=173k.  Full cached rows are intentionally read once;
        # this is 4-10x smaller than source-width Array/EPIC reads.
        unique_rows, inverse = np.unique(rows, return_inverse=True)
        full = np.asarray(self.beta[unique_rows, :], dtype=np.float32)[inverse]
        return full[:, cols]

    def close(self) -> None:
        self.h5.close()
