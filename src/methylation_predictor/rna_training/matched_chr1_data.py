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
from pathlib import Path

import h5py
import numpy as np

from ..tcga_canonical.bundle import SOURCE_FILES, MethylationSource, _H5_CACHE_KWARGS
from ..tcga_canonical.ids import GroupIndex, UniqueIndex


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


def open_methylation_source(name: str, path: Path) -> MethylationSource:
    """Build a MethylationSource from any {beta,cpg_idx,sample_idx[,measurement_idx,
    sample_split]}-contract HDF5 file -- matched_chr1's pre-extracted array/epic
    files omit measurement_idx/sample_split (unused by these trainers), so those
    are synthesized/left None rather than required."""
    h5f = h5py.File(path, "r", **_H5_CACHE_KWARGS)
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
    matched_chr1_root: Path, canonical_root: Path,
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
    approximating it."""
    proto_dir = matched_chr1_root / "table5_protocol"
    meth_dir = matched_chr1_root / "methylation"
    protocol = MatchedChr1Protocol(
        array_train_sample_idx=np.load(proto_dir / "array_train_sample_idx.npy"),
        array_val_sample_idx=np.load(proto_dir / "array_val_sample_idx.npy"),
        array_train_cpg_idx=np.load(proto_dir / "array_train_cpg_idx.npy"),
        array_val_cpg_idx=np.load(proto_dir / "array_val_cpg_idx.npy"),
        auxiliary_cpg_idx={
            "epic": np.load(proto_dir / "epic_train_cpg_idx.npy"),
            "wgbs": np.load(proto_dir / "wgbs_train_cpg_idx.npy"),
        },
    )
    sources = {
        "array": open_methylation_source("array", meth_dir / "array_table5_chr1.h5"),
        "epic": open_methylation_source("epic", meth_dir / "epic_table5_chr1.h5"),
        # No pre-extracted WGBS file exists (WGBS only has 32 rows, so
        # per-chr1 extraction saves little) -- read the full genome-wide
        # file directly, filtered to matched_chr1's own wgbs_train_cpg_idx.
        "wgbs": open_methylation_source("wgbs", canonical_root / SOURCE_FILES["wgbs"]),
    }
    return protocol, sources
