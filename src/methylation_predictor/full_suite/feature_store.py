from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import pyarrow as _pyarrow  # noqa: F401 -- must precede h5py on target server when installed
except ImportError:  # lightweight helper tests may not install the full data stack
    _pyarrow = None
import h5py
import numpy as np
import pandas as pd


class SortedIndex:
    """Compact int64 id -> row-position map using sort/searchsorted."""

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
        pos = np.searchsorted(self.sorted_ids, query)
        if len(self.sorted_ids) == 0:
            raise KeyError(f"{self.name} is empty")
        clipped = np.minimum(pos, len(self.sorted_ids) - 1)
        ok = (pos < len(self.sorted_ids)) & (self.sorted_ids[clipped] == query)
        if not np.all(ok):
            missing = query[~ok]
            raise KeyError(f"{self.name} missing {len(missing)} ids; examples={missing[:10].tolist()}")
        return self.order[pos]


def _load_base_feature_arrays(base_ids: np.ndarray, feature_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cols = ["cpg_idx", "pred_ntv3_prior", "pred_log_var_between", "pred_log_var_within"]
    frame = pd.read_parquet(feature_path, columns=cols)
    if frame.cpg_idx.duplicated().any():
        raise ValueError(f"duplicate cpg_idx in {feature_path}")
    f_ids = frame.cpg_idx.to_numpy(np.int64)
    idx = SortedIndex(f_ids, "base locus features")
    rows = idx.positions_of(base_ids)
    prior = frame.pred_ntv3_prior.to_numpy(np.float32)[rows]
    variability = frame[["pred_log_var_between", "pred_log_var_within"]].to_numpy(np.float32)[rows]
    if not np.isfinite(prior).all() or not np.isfinite(variability).all():
        raise ValueError("base locus feature table contains non-finite values")
    return prior, variability


def prepare_base_cache(
    embeddings_h5: str | Path,
    features_parquet: str | Path,
    output_dir: str | Path,
    *,
    storage_dtype: str = "float16",
    chunk_rows: int = 2048,
) -> dict[str, object]:
    """Convert the existing 408k frozen feature universe to mmap-friendly arrays.

    Random HDF5 row access is a measurable training bottleneck.  This is a pure
    representation cache: ids and numeric values are unchanged (embeddings are
    optionally stored as float16 and converted to float32 by the trainer).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "base_cpg_idx.npy"
    emb_path = out / f"base_embeddings.{ 'f16' if storage_dtype == 'float16' else 'f32' }.npy"
    prior_path = out / "base_prior.npy"
    var_path = out / "base_variability.npy"
    done = out / ".done"
    if done.is_file() and all(p.is_file() for p in (ids_path, emb_path, prior_path, var_path)):
        ids = np.load(ids_path, mmap_mode="r")
        emb = np.load(emb_path, mmap_mode="r")
        return {"status": "cached", "rows": int(len(ids)), "embedding_shape": list(emb.shape)}

    for stale in out.glob("base_embeddings.*.npy"):
        if stale != emb_path:
            stale.unlink(missing_ok=True)
    with h5py.File(embeddings_h5, "r") as h:
        ids = np.asarray(h["cpg_idx"][...], dtype=np.int64)
        source = h["embeddings"]
        if source.ndim != 2:
            raise ValueError("locus embedding dataset must be 2-D")
        source_shape = tuple(source.shape)
        dtype = np.float16 if storage_dtype == "float16" else np.float32
        mmap = np.lib.format.open_memmap(emb_path, mode="w+", dtype=dtype, shape=source_shape)
        for start in range(0, len(ids), chunk_rows):
            stop = min(start + chunk_rows, len(ids))
            mmap[start:stop] = np.asarray(source[start:stop], dtype=dtype)
        del mmap

    prior, variability = _load_base_feature_arrays(ids, Path(features_parquet))
    np.save(ids_path, ids)
    np.save(prior_path, prior)
    np.save(var_path, variability)
    done.write_text("ok\n")
    return {"status": "built", "rows": int(len(ids)), "embedding_shape": [int(len(ids)), int(source_shape[1])]}


@dataclass
class CompositeFeatureStore:
    """Frozen CpG inputs from exact base arrays plus optional NTv3 expansion."""

    base_ids: np.ndarray
    base_embeddings: np.ndarray
    base_prior: np.ndarray
    base_variability: np.ndarray
    expanded_ids: np.ndarray | None = None
    expanded_embeddings: np.ndarray | None = None
    expanded_prior: np.ndarray | None = None
    expanded_variability: np.ndarray | None = None

    @classmethod
    def from_cache(
        cls,
        base_cache_dir: str | Path,
        expanded_embedding_dir: str | Path | None = None,
        expanded_feature_dir: str | Path | None = None,
    ) -> "CompositeFeatureStore":
        base = Path(base_cache_dir)
        emb_candidates = list(base.glob("base_embeddings.*.npy"))
        if len(emb_candidates) != 1:
            raise FileNotFoundError(f"expected one base_embeddings.*.npy under {base}, found {emb_candidates}")
        kwargs = dict(
            base_ids=np.load(base / "base_cpg_idx.npy", mmap_mode="r"),
            base_embeddings=np.load(emb_candidates[0], mmap_mode="r"),
            base_prior=np.load(base / "base_prior.npy", mmap_mode="r"),
            base_variability=np.load(base / "base_variability.npy", mmap_mode="r"),
        )
        if (expanded_embedding_dir is None) != (expanded_feature_dir is None):
            raise ValueError("expanded_embedding_dir and expanded_feature_dir must be provided together")
        if expanded_embedding_dir is not None:
            emb_dir = Path(expanded_embedding_dir)
            feat_dir = Path(expanded_feature_dir)
            emb_ids = np.load(emb_dir / "expanded_cpg_idx.npy", mmap_mode="r")
            feat_ids = np.load(feat_dir / "expanded_cpg_idx.npy", mmap_mode="r")
            if not np.array_equal(emb_ids, feat_ids):
                raise RuntimeError("expanded NTv3 ids and expanded prior/variability ids differ")
            emb_candidates = list(emb_dir.glob("expanded_embeddings.*.npy"))
            if len(emb_candidates) != 1:
                raise FileNotFoundError(
                    f"expected one expanded_embeddings.*.npy under {emb_dir}, found {emb_candidates}"
                )
            kwargs.update(
                expanded_ids=emb_ids,
                expanded_embeddings=np.load(emb_candidates[0], mmap_mode="r"),
                expanded_prior=np.load(feat_dir / "expanded_prior.npy", mmap_mode="r"),
                expanded_variability=np.load(feat_dir / "expanded_variability.npy", mmap_mode="r"),
            )
        obj = cls(**kwargs)
        obj._base_index = SortedIndex(obj.base_ids, "base CpG features")
        obj._expanded_index = (
            SortedIndex(obj.expanded_ids, "expanded CpG features") if obj.expanded_ids is not None else None
        )
        if obj.base_embeddings.shape[1] != 1536:
            raise ValueError(f"expected 1536-D base embeddings, got {obj.base_embeddings.shape}")
        if obj.expanded_embeddings is not None and obj.expanded_embeddings.shape[1] != 1536:
            raise ValueError(f"expected 1536-D expanded embeddings, got {obj.expanded_embeddings.shape}")
        return obj

    def has(self, cpg_idx: np.ndarray) -> np.ndarray:
        q = np.asarray(cpg_idx, dtype=np.int64)
        ok = self._base_index.contains(q)
        if self._expanded_index is not None:
            ok |= self._expanded_index.contains(q)
        return ok

    def get(self, cpg_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.asarray(cpg_idx, dtype=np.int64)
        n = len(q)
        embeddings = np.empty((n, self.base_embeddings.shape[1]), dtype=np.float32)
        prior = np.empty(n, dtype=np.float32)
        variability = np.empty((n, 2), dtype=np.float32)

        base_mask = self._base_index.contains(q)
        if base_mask.any():
            p = self._base_index.positions_of(q[base_mask])
            embeddings[base_mask] = np.asarray(self.base_embeddings[p], dtype=np.float32)
            prior[base_mask] = np.asarray(self.base_prior[p], dtype=np.float32)
            variability[base_mask] = np.asarray(self.base_variability[p], dtype=np.float32)

        missing_mask = ~base_mask
        if missing_mask.any():
            if self._expanded_index is None:
                missing = q[missing_mask]
                raise KeyError(f"no expanded feature store; missing {len(missing)} CpGs, examples={missing[:10].tolist()}")
            p = self._expanded_index.positions_of(q[missing_mask])
            embeddings[missing_mask] = np.asarray(self.expanded_embeddings[p], dtype=np.float32)
            prior[missing_mask] = np.asarray(self.expanded_prior[p], dtype=np.float32)
            variability[missing_mask] = np.asarray(self.expanded_variability[p], dtype=np.float32)

        if not np.isfinite(embeddings).all() or not np.isfinite(prior).all() or not np.isfinite(variability).all():
            raise FloatingPointError("non-finite values returned from CompositeFeatureStore")
        return embeddings, prior, variability
