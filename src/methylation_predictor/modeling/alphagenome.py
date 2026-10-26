"""Factorized AlphaGenome model with a calibrated locus mean.

The local-response graph is a training regularizer, not an inference input.
Calibration uses a ridge mean prior and reference RNA factors from training;
no methylation values or statistics of a query locus are required at inference.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class _ResidualFFN(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * width, width), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class _RNAEncoder(nn.Module):
    def __init__(self, genes, width):
        super().__init__()
        self.norm = nn.LayerNorm(genes)
        self.bottleneck = nn.Linear(genes, width)
        self.bottleneck_norm = nn.LayerNorm(width)

    def shared(self, x):
        return self.bottleneck_norm(F.gelu(self.bottleneck(self.norm(x))))


class AlphaGenomeLocalPredictor(nn.Module):
    """RNA x static-CpG predictor with separately cacheable factors."""

    def __init__(self, input_dim: int, *, width=256, rank=32, n_tracks=4165,
                 annotation_dim=23, ag_dim=1536, depth=8, dropout=.1,
                 shallow_mean=True, mean_blend=0., reference_count=512):
        super().__init__()
        if not 0 <= mean_blend <= 1:
            raise ValueError('mean_blend must be in [0, 1]')
        self.rank = int(rank)
        self.shallow_mean = bool(shallow_mean)
        self.mean_blend = float(mean_blend)
        self.ag_dim, self.annotation_dim = int(ag_dim), int(annotation_dim)
        self.model_spec = dict(input_dim=input_dim, width=width, rank=rank, n_tracks=n_tracks,
                               annotation_dim=annotation_dim, ag_dim=ag_dim, depth=depth,
                               dropout=dropout, shallow_mean=shallow_mean, mean_blend=mean_blend,
                               reference_count=reference_count)
        self.rna_encoder = _RNAEncoder(input_dim, width)
        self.track_embedding = nn.EmbeddingBag(n_tracks, width, mode='mean', include_last_offset=True)
        self.dense_encoder = nn.Sequential(nn.LayerNorm(annotation_dim), nn.Linear(annotation_dim, width), nn.GELU())
        self.locus_norm = nn.LayerNorm(width)
        self.functional_ffn = nn.ModuleList([_ResidualFFN(width, dropout) for _ in range(depth)])
        self.mean_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))
        self.cpg_factor = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, rank))
        self.rna_factor = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, rank))
        self.ag_encoder = nn.Sequential(nn.LayerNorm(ag_dim), nn.Linear(ag_dim, width), nn.GELU(), nn.LayerNorm(width))
        for name, value in [('prior_mean', torch.zeros(ag_dim)), ('prior_scale', torch.ones(ag_dim)),
                            ('prior_coef', torch.zeros(ag_dim)), ('prior_intercept', torch.tensor(0.)),
                            ('reference_u', torch.zeros(reference_count, rank)),
                            ('calibration_ready', torch.tensor(False))]:
            self.register_buffer(name, value)

    def cpg_query(self, track_indices, offsets, dense):
        annotations, ag = dense[:, :self.annotation_dim], dense[:, self.annotation_dim:]
        functional = self.locus_norm(self.track_embedding(track_indices, offsets) + self.dense_encoder(annotations))
        return (functional + self.ag_encoder(ag)) / math.sqrt(2)

    def encode_rna(self, rna):
        return self.rna_factor(self.rna_encoder.shared(rna))

    def mean_prior(self, alphagenome):
        with torch.autocast(device_type=alphagenome.device.type, enabled=False):
            pred = ((alphagenome.float() - self.prior_mean) / self.prior_scale) @ self.prior_coef + self.prior_intercept
        return pred.clamp(.005, .995)

    def calibrated_intercept(self, raw_bias, factors, alphagenome, blend=None):
        """Match a blended population mean by a monotone scalar solve per CpG."""
        blend = self.mean_blend if blend is None else float(blend)
        if blend == 0:
            return raw_bias
        if not bool(self.calibration_ready):
            raise RuntimeError('Mean calibration requires fitted prior and training reference factors')
        with torch.autocast(device_type=factors.device.type, enabled=False):
            response = self.reference_u.float() @ factors.float().T / math.sqrt(self.rank)
            raw_mean = (raw_bias.float()[None] + response).sigmoid().mean(0)
            desired = (1-blend)*raw_mean + blend*self.mean_prior(alphagenome)
            bound = response.abs().amax(0) + 20
            lo, hi = -bound, bound
            for _ in range(24):
                mid = (lo+hi)/2
                lower = (mid[None]+response).sigmoid().mean(0) < desired
                lo, hi = torch.where(lower, mid, lo), torch.where(lower, hi, mid)
        return (lo+hi)/2

    def encode_cpg(self, *, track_indices, offsets, dense, cpg_ids=None, calibrate=None):
        if dense.ndim != 2 or dense.shape[1] != self.annotation_dim+self.ag_dim:
            raise ValueError('dense must concatenate annotations and AlphaGenome in the documented order')
        shallow = self.cpg_query(track_indices, offsets, dense)
        deep = shallow
        for block in self.functional_ffn:
            deep = block(deep)
        bias = self.mean_head(shallow if self.shallow_mean else deep).squeeze(-1)
        factors = self.cpg_factor(deep)
        use_calibration = not self.training if calibrate is None else calibrate
        if use_calibration:
            bias = self.calibrated_intercept(bias, factors, dense[:, self.annotation_dim:])
        return bias, factors

    def forward(self, rna, *, track_indices, offsets, dense, cpg_ids=None, return_internals=False):
        b, a = self.encode_cpg(track_indices=track_indices, offsets=offsets, dense=dense, cpg_ids=cpg_ids)
        u = self.encode_rna(rna)
        logit = b[None] + u @ a.T / math.sqrt(self.rank)
        result = {'beta': logit.sigmoid(), 'logit': logit, 'mu_hat': b.sigmoid()}
        if return_internals:
            result.update(cpg_factors=a, rna_factors=u)
        return result


def local_response_penalty(patient_factors, anchor_factors, neighbor_factors):
    """Smooth centered RNA responses, invariant to factor changes of basis."""
    centered = patient_factors.float() - patient_factors.float().mean(0, keepdim=True)
    response_difference = centered @ (anchor_factors.float()-neighbor_factors.float()).T
    return response_difference.square().mean() / patient_factors.shape[1]


class AlphaGenomePredictorFromCheckpoint:
    """Portable inference adapter: RNA in canonical X units + static features."""

    def __init__(self, checkpoint: str | Path, device='cpu'):
        from .factory import build_alphagenome_predictor
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state.get('schema') not in {'alphagenome_local_chr1_v2', 'alphagenome_two_stage_chr1_v3',
                                      'alphagenome_hierarchical_chr1_v3'}:
            raise ValueError('Unsupported checkpoint schema')
        self.device = torch.device(device)
        self.model = build_alphagenome_predictor(**state['model_spec']).to(self.device).eval()
        self.model.load_state_dict(state['model_state'], strict=True)
        self.gene_ids = list(state['gene_ids'])
        self.normalization = {k:torch.as_tensor(v, device=self.device) for k,v in state['normalization'].items()}

    @torch.inference_mode()
    def __call__(self, raw_rna, gene_ids, track_indices, offsets, functional_dense, alphagenome, *, cpg_ids=None):
        if list(gene_ids) != self.gene_ids:
            raise ValueError('Gene identifiers/order differ from training')
        def to(x, dtype):
            return torch.as_tensor(x, device=self.device, dtype=dtype)
        raw = to(raw_rna, torch.float32)
        dense = to(functional_dense, torch.float32).clone()
        ag = to(alphagenome, torch.float32)
        idx, off = to(track_indices, torch.long), to(offsets, torch.long)
        if raw.ndim != 2 or raw.shape[1] != len(self.gene_ids):
            raise ValueError('Invalid RNA shape')
        if dense.ndim != 2 or dense.shape[1] != self.model.annotation_dim or ag.shape != (len(dense),self.model.ag_dim):
            raise ValueError('Invalid CpG feature shape')
        if off.ndim != 1 or len(off) != len(dense)+1 or off[0] != 0 or off[-1] != len(idx) or (off[1:]<off[:-1]).any():
            raise ValueError('Invalid CSR offsets')
        if idx.ndim != 1 or ((idx<0)|(idx>=self.model.track_embedding.num_embeddings)).any():
            raise ValueError('Invalid regulatory track index')
        if not all(torch.isfinite(x).all() for x in [raw,dense,ag]):
            raise ValueError('Nonfinite inputs')
        if cpg_ids is not None:
            cpg_ids = torch.as_tensor(cpg_ids, device=self.device)
            if cpg_ids.dtype not in (torch.int32, torch.int64) or cpg_ids.shape != (len(dense),) or (cpg_ids<0).any():
                raise ValueError('cpg_ids must be a nonnegative integer vector aligned with CpG features')
        norm = self.normalization
        rna = (raw-norm['rna_mean'])/norm['rna_std']
        dense[:,17] = (dense[:,17]-norm['dense17_mean'])/norm['dense17_std']
        return self.model(rna, track_indices=idx, offsets=off, dense=torch.cat([dense,ag],1), cpg_ids=cpg_ids)['beta']
