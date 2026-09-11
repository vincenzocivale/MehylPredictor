"""Sparse projection that preserves binary regulatory features and their scale."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_projection_initializer(encoder, path, training_cpg_ids, atlas_root):
    """Reject a covariance initializer fitted on a different split or atlas."""
    projection = torch.load(path, map_location="cpu", weights_only=False)
    ids_hash = hashlib.sha256(np.sort(training_cpg_ids).astype(np.int64).tobytes()).hexdigest()
    if projection.get("train_cpg_sha256") != ids_hash:
        raise ValueError("regulatory initializer must use exactly this training CpG split")
    if projection.get("source_atlas") != str(Path(atlas_root).resolve()):
        raise ValueError("regulatory initializer uses a different functional atlas")
    if not torch.equal(encoder.frequency.cpu(), projection["state_dict"]["frequency"]):
        raise ValueError("regulatory initializer frequencies disagree with the current atlas")
    encoder.load_state_dict(projection["state_dict"])


class StandardizedTrackEmbedding(nn.Module):
    """Centered, variance-scaled binary projection without a dense atlas.

    Mean pooling divides every active track by the number of active tracks at
    that locus. This instead uses fixed per-track normalization, estimated
    exclusively on the training CpGs (variance floor 1e-3 prevents excessive
    rare-track amplification). Constant training tracks are ignored.
    The buffers travel with the checkpoint, so inference never re-fits them.
    """

    def __init__(self, n_tracks: int, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_tracks, width))
        nn.init.normal_(self.weight, std=n_tracks ** -0.5)
        self.register_buffer("frequency", torch.zeros(n_tracks))
        self.register_buffer("inverse_scale", torch.zeros(n_tracks))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, cache, training_cpg_ids: np.ndarray):
        ids = np.asarray(training_cpg_ids, np.int64)
        if not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("normalization requires nonempty unique training CpGs")
        count = np.zeros(len(self.frequency), dtype=np.float64)
        for start in range(0, len(ids), 1024):
            batch = cache.get(ids[start:start + 1024])
            count += np.bincount(batch["track_indices"], minlength=len(count))
        self.set_frequency(torch.from_numpy(count / len(ids)))

    @torch.no_grad()
    def set_frequency(self, frequency: torch.Tensor):
        p = frequency.to(device=self.frequency.device, dtype=self.frequency.dtype)
        if p.shape != self.frequency.shape or not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("track frequencies must be finite probabilities")
        variance = p * (1 - p)
        scale = torch.where(variance > 0, variance.clamp_min(1e-3).rsqrt(), 0)
        self.frequency.copy_(p)
        self.inverse_scale.copy_(scale)
        self.fitted.fill_(True)

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor):
        if not self.fitted:
            raise RuntimeError("fit regulatory normalization on training CpGs before inference")
        # Keep weights and per-sample weights in the same dtype under AMP.
        weights = self.inverse_scale[indices].to(self.weight.dtype)
        projected = F.embedding_bag(indices, self.weight, offsets, mode="sum",
                                    per_sample_weights=weights, include_last_offset=True)
        # Empty bags represent observed absence, i.e. the negative centering
        # vector, rather than an unrelated zero embedding.
        center = (self.frequency * self.inverse_scale).to(self.weight.dtype) @ self.weight
        return projected - center
