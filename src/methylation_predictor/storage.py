"""Small mmap-backed caches used by the canonical training/evaluation paths."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def _load_tcga_locus_aliases(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the three parquet aliases without constructing a giant DataFrame."""
    frames = []
    for tech in ("array", "epic", "wgbs"):
        frame = pd.read_parquet(root / "aliases" / f"tcga_{tech}.parquet", columns=["cpg_idx", "locus_key"])
        frames.append((frame.cpg_idx.to_numpy(np.int64), frame.locus_key.to_numpy(np.uint64)))
    ids = np.concatenate([x[0] for x in frames])
    keys = np.concatenate([x[1] for x in frames])
    order = np.argsort(ids, kind="mergesort")
    ids, keys = ids[order], keys[order]
    duplicate = ids[1:] == ids[:-1]
    if np.any(duplicate & (keys[1:] != keys[:-1])):
        raise ValueError("TCGA aliases disagree on cpg_idx coordinate")
    keep = np.r_[True, ~duplicate]
    return ids[keep], keys[keep]


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

class LocusPriorCache:
    """Minimal locus prior used only for evaluation metrics.

    Contract:
      * ``cpg_idx.npy`` global CpG IDs
      * ``prior.npy`` beta-space locus prior in (0, 1)

    A historical feature-cache directory is valid because it contains these
    two files, but RNA training/evaluation never opens ``embeddings.f16.npy``
    or ``sigma.npy`` through this class.
    """

    def __init__(self, root: str | Path):
        root = Path(root)
        self.root = root
        self.ids = np.load(root / "cpg_idx.npy", mmap_mode="r")
        self.prior = np.load(root / "prior.npy", mmap_mode="r")
        if self.prior.shape != (len(self.ids),):
            raise ValueError("prior.npy does not align with cpg_idx.npy")
        if not np.isfinite(self.prior).all():
            raise ValueError("prior cache contains non-finite values")
        if np.any(self.prior <= 0) or np.any(self.prior >= 1):
            raise ValueError("prior cache requires prior values in (0, 1)")
        self.index = SortedIndex(self.ids, "locus prior cache")

    def get(self, cpg_idx: np.ndarray) -> np.ndarray:
        rows = self.index.positions_of(np.asarray(cpg_idx, np.int64))
        return np.asarray(self.prior[rows], dtype=np.float32)


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

    def __init__(self, functional_atlas_root: str | Path | None = None,
                 annotation_cache_root: str | Path | None = None,
                 bigwig_cache_root: str | Path | None = None,
                 *, locus_store: str | Path | None = None,
                 cpg_registry: str | Path | None = None):
        if locus_store is not None:
            if functional_atlas_root is not None or annotation_cache_root is not None or bigwig_cache_root is not None:
                raise ValueError("locus_store cannot be combined with legacy cache roots")
            annotation_cache_root = locus_store
        if annotation_cache_root is None:
            raise ValueError("annotation_cache_root or locus_store is required")
        self.functional_atlas_root = Path(functional_atlas_root) if functional_atlas_root is not None else None
        self.annotation_cache_root = Path(annotation_cache_root)
        self.bigwig_cache_root = Path(bigwig_cache_root) if bigwig_cache_root is not None else None
        self.new_full_store = False
        if (self.annotation_cache_root / "manifest.json").exists():
            root_manifest = json.loads((self.annotation_cache_root / "manifest.json").read_text())
            if root_manifest.get("regulatory_status") == "complete":
                if root_manifest.get("status") != "complete":
                    raise ValueError("locus feature store is incomplete")
                from .locus_features.store import LocusFeatureStore
                gate = json.loads((self.annotation_cache_root / "validation_regulatory_chr1.json").read_text())
                if gate.get("status") != "complete":
                    raise ValueError("regulatory chr1 regression gate is incomplete")
                source_manifest = json.loads((self.annotation_cache_root / "regulatory" / "manifest.json").read_text())
                if source_manifest.get("status") != "complete" or source_manifest.get("verified") != self.N_TRACKS:
                    raise ValueError("frozen regulatory source files are not completely verified")
                if root_manifest.get("regulatory_track_contract_sha256") != source_manifest.get("track_contract_sha256"):
                    raise ValueError("regulatory source contract differs from store manifest")
                if cpg_registry is None:
                    self.ids, self.locus_keys = _load_tcga_locus_aliases(self.annotation_cache_root)
                else:
                    from .locus_features.ids import locus_key
                    frame = pd.read_parquet(cpg_registry, columns=["cpg_idx", "chr", "pos"])
                    self.ids = frame["cpg_idx"].to_numpy(np.int64)
                    self.locus_keys = np.asarray(
                        [locus_key(str(chrom), int(pos)) for chrom, pos in zip(frame["chr"], frame["pos"])],
                        dtype=np.uint64,
                    )
                    order = np.argsort(self.ids, kind="mergesort")
                    self.ids, self.locus_keys = self.ids[order], self.locus_keys[order]
                self.index = SortedIndex(self.ids, "genome-wide functional locus cache")
                self.locus_store = LocusFeatureStore(self.annotation_cache_root)
                self.DENSE_DIM = 23
                self.new_full_store = True
                return
        if self.functional_atlas_root is None:
            raise ValueError("legacy functional cache requires functional_atlas_root; pass a complete locus store for the canonical backend")
        self.ids = np.load(self.functional_atlas_root / "cpg_idx.npy", mmap_mode="r")
        self.locus_store = None
        self.locus_keys = None
        if (self.annotation_cache_root / "features").is_dir():
            from .locus_features.store import LocusFeatureStore
            root_manifest = json.loads((self.annotation_cache_root / "manifest.json").read_text())
            gate = json.loads((self.annotation_cache_root / "validation_chr1.json").read_text())
            if root_manifest.get("status") != "complete" or gate.get("status") != "complete":
                raise ValueError("locus feature store or chr1 regression gate is incomplete")
            self.locus_store = LocusFeatureStore(self.annotation_cache_root)
            alias_ids, alias_keys = _load_tcga_locus_aliases(self.annotation_cache_root)
            missing = np.setdiff1d(self.ids, alias_ids, assume_unique=False)
            if len(missing):
                raise KeyError(f"locus store lacks {len(missing)} functional atlas aliases")
            positions = np.searchsorted(alias_ids, np.asarray(self.ids, dtype=np.int64))
            self.locus_keys = alias_keys[positions]
            self.annotation = None
        else:
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
        if self.annotation is not None and self.annotation.shape != (len(self.ids), 18):
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

    def source_provenance(self) -> dict[str, str]:
        if self.new_full_store:
            return {"locus_store": str(self.annotation_cache_root.resolve())}
        return {
            "functional_atlas": str(self.functional_atlas_root.resolve()),
            "annotation_cache": str(self.annotation_cache_root.resolve()),
        }

    def get(self, cpg_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Retrieve arbitrary/repeated loci without densifying the track matrix."""
        rows = self.index.positions_of(np.asarray(cpg_ids, dtype=np.int64))
        if self.new_full_store:
            from .locus_features.regulatory import unpack_embedding_bags
            annotation, breadth, packed = self.locus_store.lookup_full(self.locus_keys[rows])
            track_indices, offsets = unpack_embedding_bags(packed)
            return {"track_indices": track_indices, "offsets": offsets,
                    "dense": np.concatenate((annotation, breadth), axis=1)}
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
        annotations = (self.locus_store.lookup(self.locus_keys[rows]) if self.locus_store is not None
                       else np.asarray(self.annotation[rows], dtype=np.float32))
        dense_parts = [annotations, np.asarray(self.breadth[rows], dtype=np.float32)]
        if self.bigwig is not None:
            dense_parts.append(np.asarray(self.bigwig[rows], dtype=np.float32))
        dense = np.concatenate(dense_parts, axis=1)
        return {"track_indices": track_indices, "offsets": offsets, "dense": dense}


def open_functional_locus_cache(
    *,
    locus_store: str | Path | None = None,
    functional_atlas: str | Path | None = None,
    annotation_cache: str | Path | None = None,
    bigwig_cache: str | Path | None = None,
    cpg_registry: str | Path | None = None,
) -> FunctionalLocusCache:
    """Open exactly one functional-locus backend.

    The canonical store and the frozen legacy pair are deliberately mutually
    exclusive so callers cannot silently combine feature namespaces.
    """
    if locus_store is not None:
        if any(value is not None for value in (functional_atlas, annotation_cache, bigwig_cache)):
            raise ValueError("--locus-store cannot be combined with legacy functional/annotation/BigWig caches")
        return FunctionalLocusCache(locus_store=locus_store, cpg_registry=cpg_registry)
    if functional_atlas is None or annotation_cache is None:
        raise ValueError("provide --locus-store or both legacy functional-atlas and annotation-cache paths")
    return FunctionalLocusCache(functional_atlas, annotation_cache, bigwig_cache)


class GenomicFMLocusCache:
    """E04 comparator: a dense sequence-model (genomic-FM) locus embedding,
    duck-type compatible with ``FunctionalLocusCache.get()``'s output shape
    (``track_indices``/``offsets``/``dense``) so it can feed
    ``modeling.genomic_fm.GenomicFMLocusPredictor`` (a subclass of the frozen
    final architecture with ``N_TRACKS=1``/``DENSE_DIM=embedding_dim``)
    through the same batch-construction path as the functional cache,
    without any change to the trainer's core data flow.

    ``track_indices``/``offsets`` are always empty (the degenerate
    ``EmbeddingBag(1, WIDTH)`` in ``GenomicFMLocusPredictor`` is never
    populated) -- the embedding is carried entirely through ``dense``.

    Expects an HDF5 file with ``cpg_idx`` (int64, [N]) and ``embedding``
    (float16/float32, [N, embedding_dim]) datasets -- the format produced by
    ``scripts/build_ntv3_atlas.py`` (see e.g.
    ``derived/ntv3_pre_chr1_atlas/chr1_ntv3_pretrain_atlas_v1.h5``).
    """

    def __init__(self, embedding_path: str | Path):
        import h5py

        self.embedding_path = Path(embedding_path)
        self._h5 = h5py.File(self.embedding_path, "r")
        self.ids = np.asarray(self._h5["cpg_idx"][...], dtype=np.int64)
        self.embedding_dim = int(self._h5["embedding"].shape[1])
        if self._h5["embedding"].shape[0] != len(self.ids):
            raise ValueError(
                f"{self.embedding_path}: cpg_idx/embedding row count mismatch"
            )
        self.index = SortedIndex(self.ids, "genomic-FM locus cache")
        self.model = self._h5.attrs.get("model")

    def source_provenance(self) -> dict[str, str]:
        return {
            "genomic_fm_cache": str(self.embedding_path.resolve()),
            "genomic_fm_model": str(self.model) if self.model is not None else "",
        }

    def get(self, cpg_ids: np.ndarray) -> dict[str, np.ndarray]:
        rows = self.index.positions_of(np.asarray(cpg_ids, dtype=np.int64))
        dense = np.asarray(self._h5["embedding"][...][rows], dtype=np.float32)
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        return {
            "track_indices": np.empty(0, dtype=np.int64),
            "offsets": offsets,
            "dense": dense,
        }


class MaskedFunctionalLocusCache:
    """E03 comparator: wraps a full ``FunctionalLocusCache``/genome-wide
    locus store and zeroes out parts of its output to realize the paper
    plan's reduced functional-representation arms (sec. E03), without
    changing the frozen model architecture (input shape is unchanged --
    only which entries are nonzero changes) or duplicating the underlying
    store.

    ``minimal``: no sparse regulatory tracks, only the bare CpG-island
    context (dense indices 0-3) and TSS distance (index 17) dense
    annotations; genomic-region/cCRE class and all breadth/track signal
    zeroed.
    ``basic_context``: no sparse regulatory tracks; all 18 static reference
    annotations (dense indices 0-17: CpG-island context, genomic region,
    cCRE class, TSS distance) kept, breadth (18-22) zeroed.
    ``full_functional``: passthrough, identical to the wrapped cache.

    Dense column indices follow ``resources/locus_features/annotation_core_v1.json``.
    """

    FEATURE_SETS = ("minimal", "basic_context", "full_functional")
    _MINIMAL_DENSE_IDX = (0, 1, 2, 3, 17)
    _BASIC_CONTEXT_DENSE_IDX = tuple(range(0, 18))

    def __init__(self, wrapped, *, feature_set: str):
        if feature_set not in self.FEATURE_SETS:
            raise ValueError(
                f"unknown feature_set {feature_set!r}; expected one of {self.FEATURE_SETS}"
            )
        self.wrapped = wrapped
        self.feature_set = feature_set
        self.index = wrapped.index

    def source_provenance(self) -> dict[str, str]:
        provenance = dict(self.wrapped.source_provenance())
        provenance["feature_set"] = self.feature_set
        return provenance

    def get(self, cpg_ids: np.ndarray) -> dict[str, np.ndarray]:
        payload = self.wrapped.get(cpg_ids)
        if self.feature_set == "full_functional":
            return payload
        dense = np.zeros_like(payload["dense"])
        keep_idx = (
            self._MINIMAL_DENSE_IDX
            if self.feature_set == "minimal"
            else self._BASIC_CONTEXT_DENSE_IDX
        )
        dense[:, list(keep_idx)] = payload["dense"][:, list(keep_idx)]
        n_rows = len(np.asarray(cpg_ids))
        return {
            "track_indices": np.empty(0, dtype=np.int64),
            "offsets": np.zeros(n_rows + 1, dtype=np.int64),
            "dense": dense,
        }
