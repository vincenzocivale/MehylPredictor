"""Shared helper for reading the official MethylProphet-matched chr1 split/data
(``derived/methylprophet_table5_tcga_chr1``) directly, bypassing scope_protocol()'s
own ``tcga_mix_chr1`` reconstruction and the full genome-wide canonical bundle.

Used by more than one ablation-only trainer (``joint_trainer.py``,
``locus_cls_trainer.py``) -- factored out here to avoid duplicating it. Still
part of the same "ablation-only, isolated" family as those trainers (see
CLAUDE.md's note on not routing generic-pipeline changes through the frozen
MethylProphet benchmark trainer) -- this reads matched_chr1's *data*, it does
not touch ``MethylProphetTrainer`` itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from ..tcga_canonical.bundle import SOURCE_FILES, MethylationSource, h5_cache_kwargs
from ..tcga_canonical.ids import GroupIndex, UniqueIndex


def _axis_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, np.int64).tobytes()).hexdigest()


def _file_digest(path: Path, chunk_bytes: int = 8 * 1024**2) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(slots=True)
class MatchedChr1Protocol:
    """Minimal stand-in for tcga_canonical.protocol.Protocol, built directly
    from matched_chr1's own table5_protocol/ arrays instead of scope_protocol()'s
    reconstruction -- only the fields the ablation trainers actually read."""

    array_train_sample_idx: np.ndarray
    array_val_sample_idx: np.ndarray
    array_train_cpg_idx: np.ndarray
    array_val_cpg_idx: np.ndarray
    auxiliary_cpg_idx: dict[str, np.ndarray] = field(default_factory=dict)
    sources: tuple[str, ...] = ("array", "epic", "wgbs")


def open_methylation_source(name: str, path: Path, *, hdf5_cache_mb: int = 256) -> MethylationSource:
    """Build a MethylationSource from any {beta,cpg_idx,sample_idx[,measurement_idx,
    sample_split]}-contract HDF5 file -- matched_chr1's pre-extracted array/epic
    files omit measurement_idx/sample_split (unused by these trainers), so those
    are synthesized/left None rather than required."""
    h5f = h5py.File(path, "r", **h5_cache_kwargs(hdf5_cache_mb))
    sample_idx = np.asarray(h5f["sample_idx"][...], dtype=np.int64)
    measurement_idx = (
        np.asarray(h5f["measurement_idx"][...], dtype=np.int64)
        if "measurement_idx" in h5f else np.arange(len(sample_idx), dtype=np.int64)
    )
    sample_split = None
    if "sample_split" in h5f:
        raw = h5f["sample_split"][...]
        sample_split = np.asarray(
            [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in raw], dtype=object,
        )
    return MethylationSource(
        name=name, path=path, h5=h5f, sample_idx=sample_idx, measurement_idx=measurement_idx,
        sample_split=sample_split,
        _cpg_index=UniqueIndex(h5f["cpg_idx"][...], name=f"{name} cpg_idx"),
        _measurement_index=UniqueIndex(measurement_idx, name=f"{name} measurement_idx"),
        _sample_groups=GroupIndex(sample_idx, name=f"{name} sample_idx"),
    )


def load_matched_chr1_protocol_and_sources(
    matched_chr1_root: Path, canonical_root: Path, *, hdf5_cache_mb: int = 256,
    sources: tuple[str, ...] = ("array", "epic", "wgbs"),
) -> tuple[MatchedChr1Protocol, dict[str, MethylationSource]]:
    """Official MethylProphet-matched chr1 split + pre-extracted, well-chunked
    data files -- see docs/BENCHMARK_METHYLPROPHET.md and
    derived/methylprophet_table5_tcga_chr1/table5_protocol/protocol.json.
    Bypasses scope_protocol()'s own tcga_mix_chr1 reconstruction: the Array
    train/val split matches exactly (independently verified, see that doc),
    but the auxiliary EPIC/WGBS CpG sets differ slightly (matched_chr1 applies
    MethylProphet's own "central 1000bp window has no N bases" hg38 filter,
    e.g. 102 fewer WGBS CpGs than this repo's own reconstruction) -- using
    matched_chr1's own arrays directly removes that gap entirely rather than
    approximating it.

    ``sources``: which training sources to open (Table 7 / paper section B.6's source
    ablation -- Array / Array+WGBS / Array+EPIC / Array+EPIC+WGBS). "array" is always
    implicitly required (the official split's own base); it is opened unconditionally
    regardless of whether it's listed. Default is unchanged from before this parameter
    existed (all three) -- existing callers/recipes are unaffected."""
    proto_dir = matched_chr1_root / "table5_protocol"
    meth_dir = matched_chr1_root / "methylation"
    wanted = {"array", *sources}
    protocol = MatchedChr1Protocol(
        array_train_sample_idx=np.load(proto_dir / "array_train_sample_idx.npy"),
        array_val_sample_idx=np.load(proto_dir / "array_val_sample_idx.npy"),
        array_train_cpg_idx=np.load(proto_dir / "array_train_cpg_idx.npy"),
        array_val_cpg_idx=np.load(proto_dir / "array_val_cpg_idx.npy"),
        auxiliary_cpg_idx={
            name: np.load(proto_dir / f"{name}_train_cpg_idx.npy")
            for name in ("epic", "wgbs") if name in wanted
        },
        sources=tuple(sorted(wanted, key=("array", "epic", "wgbs").index)),
    )
    opened: dict[str, MethylationSource] = {
        "array": open_methylation_source("array", meth_dir / "array_table5_chr1.h5", hdf5_cache_mb=hdf5_cache_mb),
    }
    if "epic" in wanted:
        opened["epic"] = open_methylation_source("epic", meth_dir / "epic_table5_chr1.h5", hdf5_cache_mb=hdf5_cache_mb)
    if "wgbs" in wanted:
        # No pre-extracted WGBS file exists (WGBS only has 32 rows, so
        # per-chr1 extraction saves little) -- read the full genome-wide
        # file directly, filtered to matched_chr1's own wgbs_train_cpg_idx.
        opened["wgbs"] = open_methylation_source("wgbs", canonical_root / SOURCE_FILES["wgbs"], hdf5_cache_mb=hdf5_cache_mb)
    return protocol, opened


def load_compact_scope_sources(
    compact_root: Path, protocol, *, hdf5_cache_mb: int = 256,
) -> dict[str, MethylationSource]:
    """Open scope-specific caches and verify manifest, axes, and content hash.

    Legacy manifests without file hashes remain readable, but still receive
    exact axis validation. Re-running the cache builder upgrades them to the
    content-hashed schema without rebuilding the matrices.
    """
    manifest_path = compact_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"compact cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("scope") != "chr123":
        raise ValueError(f"compact cache has unexpected scope: {manifest.get('scope')!r}")
    records = manifest.get("sources") or {}
    for name in protocol.sources:
        path = compact_root / f"{name}.h5"
        record = records.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"compact cache manifest has no {name!r} source record")
        expected_bytes = record.get("bytes")
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            raise ValueError(f"{name} compact cache size does not match manifest")
        expected_file_hash = record.get("file_sha256")
        if expected_file_hash is not None and _file_digest(path) != expected_file_hash:
            raise ValueError(f"{name} compact cache SHA-256 does not match manifest")
    sources = {
        name: open_methylation_source(name, compact_root / f"{name}.h5", hdf5_cache_mb=hdf5_cache_mb)
        for name in protocol.sources
    }
    required = {
        "array": np.unique(np.concatenate([protocol.array_train_cpg_idx, protocol.array_val_cpg_idx])),
        **{name: np.asarray(ids, np.int64) for name, ids in protocol.auxiliary_cpg_idx.items()},
    }
    try:
        expected_samples = {
            "array": np.concatenate([protocol.array_train_sample_idx, protocol.array_val_sample_idx]),
            **{name: sources[name].sample_idx for name in protocol.sources if name != "array"},
        }
        for name, ids in required.items():
            if name in sources:
                sources[name].cpg_positions(ids)
                record = records[name]
                if _axis_digest(sources[name].h5["cpg_idx"][...]) != record.get("cpgs_sha256"):
                    raise ValueError(f"{name} compact CpG axis does not match manifest")
                if _axis_digest(sources[name].sample_idx) != record.get("samples_sha256"):
                    raise ValueError(f"{name} compact sample axis does not match manifest")
                if name == "array" and not np.array_equal(sources[name].sample_idx, expected_samples[name]):
                    raise ValueError("array compact sample axis does not match protocol order")
        sources["array"].rows_of_samples(
            np.concatenate([protocol.array_train_sample_idx, protocol.array_val_sample_idx])
        )
    except Exception:
        for source in sources.values():
            source.close()
        raise
    return sources
