"""Small mmap-backed caches used by the canonical training/evaluation paths."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

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
        self.regulatory_manifest = None
        self.regulatory_provenance = None
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("feature_type") == "regulatory_pca":
                if manifest.get("status") != "complete":
                    raise ValueError("regulatory feature cache is incomplete")
                self.regulatory_manifest = manifest
                self.regulatory_provenance = {
                    "path": str(root.resolve()),
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                }
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

    def validate_training_split(self, training_cpg_ids):
        if self.regulatory_manifest is None:
            return
        fingerprint = hashlib.sha256(np.sort(training_cpg_ids).astype(np.int64).tobytes()).hexdigest()
        if fingerprint != self.regulatory_manifest.get("train_cpg_sha256"):
            raise ValueError("regulatory feature cache was fitted on a different training CpG split")

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


class FunctionalLocusCache:
    """Mmap-backed sparse functional atlas aligned to global CpG IDs."""

    N_TRACKS = 4165
    DENSE_DIM = 23

    def __init__(self, functional_atlas_root: str | Path, annotation_cache_root: str | Path,
                 bigwig_cache_root: str | Path | None = None):
        self.functional_atlas_root = Path(functional_atlas_root)
        self.annotation_cache_root = Path(annotation_cache_root)
        self.bigwig_cache_root = Path(bigwig_cache_root) if bigwig_cache_root is not None else None
        self.ids = np.load(self.functional_atlas_root / "cpg_idx.npy", mmap_mode="r")
        annotation_ids = np.load(self.annotation_cache_root / "cpg_idx.npy", mmap_mode="r")
        if not np.array_equal(self.ids, annotation_ids):
            raise ValueError("functional atlas and annotation cache cpg_idx axes do not match exactly")

        self.annotation = np.load(self.annotation_cache_root / "annotation_core.f32.npy", mmap_mode="r")
        self.breadth = np.load(self.functional_atlas_root / "breadth_features.f32.npy", mmap_mode="r")
        self.bigwig = None
        if self.bigwig_cache_root is not None:
            manifest = json.loads((self.bigwig_cache_root / "manifest.json").read_text())
            if manifest.get("status") != "complete" or manifest.get("dimensions") != 45:
                raise ValueError("BigWig cache is incomplete or has unexpected dimensions")
            self.bigwig = np.load(self.bigwig_cache_root / "pca23.f32.npy", mmap_mode="r")
        csr_root = self.functional_atlas_root / "all_primary_overlap_csr"
        self.indices = np.load(csr_root / "indices.npy", mmap_mode="r")
        self.indptr = np.load(csr_root / "indptr.npy", mmap_mode="r")
        shape_payload = json.loads((csr_root / "shape.json").read_text())
        shape = tuple(shape_payload.get("shape", ())) if isinstance(shape_payload, dict) else tuple(shape_payload)
        expected_shape = (len(self.ids), self.N_TRACKS)
        if shape != expected_shape:
            raise ValueError(f"unexpected functional CSR shape {shape}; expected {expected_shape}")
        if self.annotation.shape != (len(self.ids), 18):
            raise ValueError(f"unexpected annotation_core shape {self.annotation.shape}")
        if self.breadth.shape != (len(self.ids), 5):
            raise ValueError(f"unexpected breadth_features shape {self.breadth.shape}")
        if self.indptr.shape != (len(self.ids) + 1,):
            raise ValueError("functional CSR indptr does not align with cpg_idx")
        if self.bigwig is not None and self.bigwig.shape != (len(self.ids), 23):
            raise ValueError("BigWig PCA cache does not align with cpg_idx")
        self.DENSE_DIM = 46 if self.bigwig is not None else self.DENSE_DIM
        if int(self.indptr[-1]) != len(self.indices):
            raise ValueError("functional CSR indices/indptr are inconsistent")
        if len(self.indices) and (np.min(self.indices) < 0 or np.max(self.indices) >= self.N_TRACKS):
            raise ValueError("functional CSR contains out-of-range track indices")
        self.index = SortedIndex(self.ids, "functional locus cache")

    def get(self, cpg_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Retrieve arbitrary/repeated loci without densifying the track matrix."""
        rows = self.index.positions_of(np.asarray(cpg_ids, dtype=np.int64))
        starts = np.asarray(self.indptr[rows], dtype=np.int64)
        counts = np.asarray(self.indptr[rows + 1], dtype=np.int64) - starts
        offsets = np.empty(len(rows) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        if offsets[-1]:
            repeated_starts = np.repeat(starts, counts)
            within_rows = np.arange(int(offsets[-1]), dtype=np.int64) - np.repeat(offsets[:-1], counts)
            track_indices = np.asarray(self.indices[repeated_starts + within_rows], dtype=np.int64)
        else:
            track_indices = np.empty(0, dtype=np.int64)
        dense_parts = [np.asarray(self.annotation[rows], dtype=np.float32), np.asarray(self.breadth[rows], dtype=np.float32)]
        if self.bigwig is not None:
            dense_parts.append(np.asarray(self.bigwig[rows], dtype=np.float32))
        dense = np.concatenate(dense_parts, axis=1)
        return {"track_indices": track_indices, "offsets": offsets, "dense": dense}
