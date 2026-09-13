"""E04 comparator: genomic-FM (sequence-derived) locus representation.

Subclasses the frozen final architecture (``EfficientSingleAttentionPredictor``)
and only overrides ``N_TRACKS``/``DENSE_DIM`` so the locus encoder front-end
consumes a dense sequence-model embedding (e.g. NTv3-pre, 1536-d -- see
``derived/ntv3_pre_chr1_atlas``) instead of the 4165-sparse-track + 23-dense
functional annotation representation. RNA branch, retrieval attention,
mean-proxy head, loss and final regressor are all inherited byte-for-byte
unchanged -- per the paper plan's E04 methodology ("Regola sperimentale":
change only the locus representation, keep everything else fixed).

``track_embedding`` becomes a degenerate, never-populated
``nn.EmbeddingBag(1, WIDTH)`` (``storage.GenomicFMLocusCache`` always returns
empty ``track_indices``/``offsets``), so its output is always zero and
``dense_encoder`` carries the entire locus signal.
"""
from __future__ import annotations

from .final import EfficientSingleAttentionPredictor


class GenomicFMLocusPredictor(EfficientSingleAttentionPredictor):
    N_TRACKS = 1
    DENSE_DIM = 1536
