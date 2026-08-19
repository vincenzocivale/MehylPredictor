"""Genomic scope definitions shared by training, tuning and evaluation.

A scope changes the CpG universe only; it never changes the model architecture.
The official Array train/validation split is always inherited from the frozen
canonical TCGA bundle.  This keeps chr1/chr123 matched evaluations comparable to
MethylProphet while allowing the same software path to scale genome-wide.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .tcga_canonical.protocol import Protocol, load_protocol

ScopeName = Literal["chr1", "chr123", "genomewide"]
SCOPE_NAMES: tuple[ScopeName, ...] = ("chr1", "chr123", "genomewide")


@dataclass(frozen=True, slots=True)
class GenomicScope:
    name: ScopeName
    chromosomes: tuple[str, ...] | None
    matched_methylprophet: bool


SCOPES = {
    "chr1": GenomicScope("chr1", ("chr1",), True),
    "chr123": GenomicScope("chr123", ("chr1", "chr2", "chr3"), True),
    "genomewide": GenomicScope("genomewide", None, False),
}


def resolve_scope(name: str) -> GenomicScope:
    try:
        return SCOPES[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown genomic scope {name!r}; expected one of {SCOPE_NAMES}") from exc


def _normalize_chromosome(value: object) -> str:
    text = str(value)
    return text if text.startswith("chr") else f"chr{text}"


def chromosome_lookup(registry: str | Path, cpg_ids: np.ndarray) -> np.ndarray:
    """Return chromosome labels aligned to global ``cpg_idx`` IDs."""
    frame = pd.read_parquet(registry)
    chrom_col = next((x for x in ("chrom", "chr", "chromosome") if x in frame.columns), None)
    if chrom_col is None or "cpg_idx" not in frame.columns:
        raise ValueError("CpG registry must contain cpg_idx and chrom/chr/chromosome")
    if frame.cpg_idx.duplicated().any():
        raise ValueError("CpG registry contains duplicate cpg_idx")
    indexed = frame.set_index("cpg_idx")
    missing = np.setdiff1d(np.asarray(cpg_ids, np.int64), indexed.index.to_numpy(np.int64))
    if len(missing):
        raise KeyError(f"CpG registry is missing {len(missing)} ids; examples={missing[:10].tolist()}")
    return np.asarray([_normalize_chromosome(x) for x in indexed.loc[cpg_ids, chrom_col].tolist()], dtype=object)


def filter_cpg_ids(cpg_ids: np.ndarray, scope: str, registry: str | Path) -> np.ndarray:
    scope_obj = resolve_scope(scope)
    cpg_ids = np.asarray(cpg_ids, dtype=np.int64)
    if scope_obj.chromosomes is None:
        return cpg_ids.copy()
    chrom = chromosome_lookup(registry, cpg_ids)
    return cpg_ids[np.isin(chrom, scope_obj.chromosomes)]


def scope_protocol(scope: str, bundle, *, canonical_root: str | Path) -> Protocol:
    """Resolve the mixed-source training/evaluation protocol for a scope.

    ``chr1`` and ``chr123`` use the on-disk official protocol contracts.  The
    genome-wide contract starts from the official Array genome-wide split and
    adds EPIC/WGBS only at CpGs belonging to the official Array TRAIN universe
    and physically present in the auxiliary technology.  Held-out Array CpGs
    are therefore never promoted into the training pool merely because another
    technology measured them.
    """
    scope_obj = resolve_scope(scope)
    if scope_obj.name == "chr1":
        return load_protocol("tcga_mix_chr1", bundle, root=canonical_root)
    if scope_obj.name == "chr123":
        return load_protocol("tcga_mix_chr123", bundle, root=canonical_root)

    base = load_protocol("array_genomewide", bundle, root=canonical_root)
    auxiliary: dict[str, np.ndarray] = {}
    for source_name in ("epic", "wgbs"):
        source = bundle.sources[source_name]
        mask = source.has_cpg(base.array_train_cpg_idx)
        auxiliary[source_name] = np.asarray(base.array_train_cpg_idx[mask], np.int64)
        if not len(auxiliary[source_name]):
            raise RuntimeError(f"no genome-wide training CpGs overlap source {source_name!r}")
    return Protocol(
        name="tcga_mix_genomewide",
        bundle=bundle,
        metadata={
            **dict(base.metadata),
            "derived_by": "methylation_predictor.scopes.scope_protocol",
            "training_sources": ["array", "epic", "wgbs"],
            "auxiliary_rule": "official Array train CpGs intersect source coverage",
        },
        sources=("array", "epic", "wgbs"),
        chromosomes=tuple(),
        array_train_sample_idx=base.array_train_sample_idx,
        array_val_sample_idx=base.array_val_sample_idx,
        array_train_cpg_idx=base.array_train_cpg_idx,
        array_val_cpg_idx=base.array_val_cpg_idx,
        auxiliary_cpg_idx=auxiliary,
    )


def evaluation_protocol(scope: str, bundle, *, canonical_root: str | Path) -> Protocol:
    """Resolve an Array evaluation contract independently of training scope."""
    return scope_protocol(scope, bundle, canonical_root=canonical_root)
