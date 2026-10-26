"""Two-stage AlphaGenome learning and explicit transfer of RNA responses.

The frozen mean is a snapshot of the complete mean function, including its
feature encoders. Freezing just the final head would not freeze that function.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .alphagenome import AlphaGenomeLocalPredictor


class _LocusMeanSnapshot(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.annotation_dim = source.annotation_dim
        self.shallow_mean = source.shallow_mean
        for name in ['track_embedding', 'dense_encoder', 'locus_norm', 'ag_encoder', 'mean_head']:
            setattr(self, name, copy.deepcopy(getattr(source, name)))
        self.functional_ffn = (nn.ModuleList() if source.shallow_mean
                               else copy.deepcopy(source.functional_ffn))
        self.requires_grad_(False)
        self.eval()

    def forward(self, track_indices, offsets, dense):
        ann, ag = dense[:, :self.annotation_dim], dense[:, self.annotation_dim:]
        h = self.locus_norm(self.track_embedding(track_indices, offsets) + self.dense_encoder(ann))
        h = (h + self.ag_encoder(ag)) / math.sqrt(2)
        for block in self.functional_ffn:
            h = block(h)
        return self.mean_head(h).squeeze(-1)


class AlphaGenomeTwoStagePredictor(AlphaGenomeLocalPredictor):
    """Keep a fixed early mean while learning patient-specific responses."""
    def __init__(self, input_dim, *, frozen_mean=False, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.mean_snapshot = _LocusMeanSnapshot(self) if frozen_mean else None
        self.model_spec.update(variant='alphagenome_two_stage_chr1_v3', frozen_mean=frozen_mean)

    def freeze_mean(self):
        self.mean_snapshot = _LocusMeanSnapshot(self)
        self.mean_head.requires_grad_(False)
        self.model_spec['frozen_mean'] = True

    def train(self, mode=True):
        super().train(mode)
        if self.mean_snapshot is not None:
            self.mean_snapshot.eval()
        return self

    def encode_cpg(self, *, track_indices, offsets, dense, cpg_ids=None, calibrate=None):
        b, a = super().encode_cpg(track_indices=track_indices, offsets=offsets, dense=dense,
                                  cpg_ids=cpg_ids, calibrate=False)
        if self.mean_snapshot is not None:
            b = self.mean_snapshot(track_indices, offsets, dense)
        if (not self.training if calibrate is None else calibrate):
            b = self.calibrated_intercept(b, a, dense[:, self.annotation_dim:])
        return b, a


def transfer_response(bias, factors, neighbor_factors, reference_u, blend):
    """Blend responses, preserving the original training-reference beta mean.

    A fixed reference population makes this independent of the query patients
    and their batching. The mean is preserved on this population only.
    """
    if not 0 <= blend <= 1:
        raise ValueError('blend must be in [0, 1]')
    if blend == 0:
        return bias, factors
    if neighbor_factors.ndim != 3 or neighbor_factors.shape[0] != len(factors) or neighbor_factors.shape[2] != factors.shape[1]:
        raise ValueError('neighbor_factors must have shape [loci, neighbors, rank]')
    if reference_u.ndim != 2 or reference_u.shape[1] != factors.shape[1] or not len(reference_u):
        raise ValueError('A fixed nonempty training RNA reference is required')
    with torch.autocast(device_type=factors.device.type, enabled=False):
        b, a, ref = bias.float(), factors.float(), reference_u.float()
        transferred = (1-blend)*a + blend*neighbor_factors.float().mean(1)
        scale = math.sqrt(a.shape[1])
        desired = (b[None] + ref @ a.T / scale).sigmoid().mean(0)
        response = ref @ transferred.T / scale
        bound = response.abs().amax(0) + b.abs() + 20
        lo, hi = -bound, bound
        for _ in range(24):
            mid = (lo+hi)/2
            lower = (mid[None]+response).sigmoid().mean(0) < desired
            lo, hi = torch.where(lower, mid, lo), torch.where(lower, hi, mid)
    return (lo+hi)/2, transferred
