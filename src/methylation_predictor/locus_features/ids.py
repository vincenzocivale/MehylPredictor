"""hg38 CpG identities. Coordinates are 1-based positions of the C base."""

from __future__ import annotations

from collections.abc import Mapping

_CHROMS = {f"chr{i}": i for i in range(1, 23)} | {"chrX": 23, "chrY": 24}
_CODES = {v: k for k, v in _CHROMS.items()}


def canonical_locus(chrom: str, position: int, *, lengths: Mapping[str, int] | None = None) -> tuple[str, int]:
    """Validate a reference coordinate without silently changing its convention."""
    if chrom not in _CHROMS:
        raise ValueError(f"invalid canonical hg38 chromosome: {chrom!r}")
    if isinstance(position, bool) or not isinstance(position, int) or not 1 <= position < 2**32:
        raise ValueError(f"invalid 1-based CpG position: {position!r}")
    if lengths is not None:
        if chrom not in lengths or position >= lengths[chrom]:
            raise ValueError(f"CpG at {chrom}:{position} exceeds reference bounds")
    return chrom, position


def locus_key(chrom: str, position: int, *, lengths: Mapping[str, int] | None = None) -> int:
    chrom, position = canonical_locus(chrom, position, lengths=lengths)
    return (_CHROMS[chrom] << 32) | position


def decode_locus_key(key: int) -> tuple[str, int]:
    if isinstance(key, bool) or not isinstance(key, int) or key < 0:
        raise ValueError(f"invalid locus key: {key!r}")
    chrom = _CODES.get(key >> 32)
    if chrom is None:
        raise ValueError(f"invalid chromosome code in locus key: {key!r}")
    return canonical_locus(chrom, key & 0xFFFFFFFF)


def tcga_cpg_idx_to_locus_key(cpg_idx: int, registry: Mapping[int, tuple[str, int]]) -> int:
    """Resolve a TCGA ID through an explicit canonical registry; missing IDs fail closed."""
    try:
        chrom, position = registry[cpg_idx]
    except KeyError as exc:
        raise KeyError(f"cpg_idx {cpg_idx} absent from canonical registry") from exc
    return locus_key(chrom, position)
