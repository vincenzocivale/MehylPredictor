"""Selectable CpG-query modules for the Perceiver predictor.

Each module maps the trainer's locus payload (``track_indices``, ``offsets``,
``dense``) to ``h_c`` of shape (n_loci, width). Two winning representations of
the CpGRepresentationBenchmark are supported:

* ``GenericFunctionalQuery``: generic reference annotations (ENCODE tracks +
  23 dense annotations), optionally restricted to a track subset such as the
  benchmark's raw Histone+DNase baseline (2,492 tracks).
* ``LatentQuery``: precomputed latent embedding of a model trained to predict
  functional tracks (e.g. Enformer trunk, 3072-d), carried in ``dense``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def track_keep_mask(contract_path: str, blocks: list[str], n_tracks: int = 4165) -> np.ndarray:
    """Boolean mask over track indices from a ``track_contract.tsv`` (functional_block column)."""
    import pandas as pd

    frame = pd.read_csv(contract_path, sep="\t", usecols=["track_index", "functional_block"])
    if len(frame) != n_tracks or not frame.track_index.is_monotonic_increasing:
        raise ValueError("track contract must list the frozen tracks in column order")
    return frame.functional_block.isin(blocks).to_numpy()  # column = row position


class GenericFunctionalQuery(nn.Module):
    def __init__(self, width: int = 256, n_tracks: int = 4165, dense_dim: int = 23,
                 keep_mask: np.ndarray | None = None):
        super().__init__()
        self.track_embedding = nn.EmbeddingBag(n_tracks, width, mode="mean", include_last_offset=True)
        self.dense_encoder = nn.Sequential(nn.LayerNorm(dense_dim), nn.Linear(dense_dim, width), nn.GELU())
        self.norm = nn.LayerNorm(width)
        self.register_buffer("keep", None if keep_mask is None else torch.as_tensor(keep_mask, dtype=torch.bool))

    def forward(self, track_indices, offsets, dense):
        if self.keep is not None:
            n = offsets.numel() - 1
            seg = torch.repeat_interleave(torch.arange(n, device=offsets.device), offsets[1:] - offsets[:-1])
            k = self.keep[track_indices]
            track_indices = track_indices[k]
            counts = torch.bincount(seg[k], minlength=n)
            offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        return self.norm(self.track_embedding(track_indices, offsets) + self.dense_encoder(dense))


class LatentQuery(nn.Module):
    def __init__(self, dim: int, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, track_indices, offsets, dense):
        return self.norm(self.net(dense))
