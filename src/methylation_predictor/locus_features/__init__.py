"""Canonical hg38 locus identities and static feature storage."""

from .ids import canonical_locus, decode_locus_key, locus_key, tcga_cpg_idx_to_locus_key

__all__ = ["canonical_locus", "decode_locus_key", "locus_key", "tcga_cpg_idx_to_locus_key"]
