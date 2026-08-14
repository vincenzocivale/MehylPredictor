"""Reusable preprocessing utilities retained by the production pipeline.

The historical E2/E3/E4 and architecture-ablation trainers were removed after
architecture selection.  This package now only hosts stable cache/index/probe
utilities needed by genomic_prior_v2 and final tcga_mix_chr1 preprocessing.
"""

from .feature_store import SortedIndex

__all__ = ["SortedIndex"]
