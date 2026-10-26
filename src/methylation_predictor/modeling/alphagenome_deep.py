"""AlphaGenome factorised predictor with a deep pairwise decoder (raw predictions).

    logit(p, c) = b_c + u_p . a_c / sqrt(rank) + g(p, c)

``b_c``, ``a_c`` and ``u_p`` are the frozen-protocol factorised model, unchanged and
initialised exactly as in ``AlphaGenomeLocalPredictor``. ``g`` is a ``DeepDecoder``
that reads RNA tokens, generated from the same 256-d RNA bottleneck, with the CpG's
shallow query and deep state (both contain the AlphaGenome embedding). Its last
layer starts at zero, so training begins at the factorised model and ``g`` grows only
where it lowers the loss. Calibration is not implemented for ``g``: predictions are raw.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .alphagenome import AlphaGenomeLocalPredictor
from .deep_decoder import DeepDecoder

VARIANT = "alphagenome_deep_decoder_chr1_v4"


class AlphaGenomeDeepDecoderPredictor(AlphaGenomeLocalPredictor):
    def __init__(self, input_dim: int, *, decoder: dict | None = None, n_tokens: int = 64,
                 patient_chunk: int = 512, cpg_chunk: int = 256, **kwargs):
        super().__init__(input_dim, **kwargs)
        width = self.model_spec["width"]
        self.n_tokens, self.patient_chunk, self.cpg_chunk = int(n_tokens), int(patient_chunk), int(cpg_chunk)
        self.token_proj = nn.Linear(width, self.n_tokens * width)
        self.token_norm = nn.LayerNorm(width)
        self.decoder = DeepDecoder(query_dim=width, token_dim=width, zero_output=True, **(decoder or {}))
        self.model_spec.update(variant=VARIANT, decoder=dict(decoder or {}), n_tokens=self.n_tokens,
                               patient_chunk=self.patient_chunk, cpg_chunk=self.cpg_chunk)

    def pair_residual(self, s: torch.Tensor, shallow: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        """g for every (patient, CpG); chunked, and recomputed in backward when training."""
        if not len(shallow):
            return s.new_zeros(len(s), 0)
        tokens = self.token_norm(self.token_proj(s).view(len(s), self.n_tokens, -1))
        recompute = self.training and torch.is_grad_enabled()
        rows = []
        for p0 in range(0, len(s), self.patient_chunk):
            tok = tokens[p0:p0 + self.patient_chunk]
            cols = []
            for c0 in range(0, len(shallow), self.cpg_chunk):
                args = (shallow[c0:c0 + self.cpg_chunk], deep[c0:c0 + self.cpg_chunk], tok)
                cols.append(checkpoint(self.decoder, *args, use_reentrant=False) if recompute else self.decoder(*args))
            rows.append(torch.cat(cols, dim=1))
        return torch.cat(rows, dim=0)

    def forward(self, rna, *, track_indices, offsets, dense, cpg_ids=None, return_internals=False,
                residual_index=None):
        """``residual_index``: unique CpG rows that receive ``g`` (training anchors); default all."""
        shallow = self.cpg_query(track_indices, offsets, dense)
        deep = shallow
        for block in self.functional_ffn:
            deep = block(deep)
        bias = self.mean_head(shallow if self.shallow_mean else deep).squeeze(-1)
        a = self.cpg_factor(deep)
        s = self.rna_encoder.shared(rna)
        u = self.rna_factor(s)
        logit = bias[None] + u @ a.T / self.rank ** .5
        if residual_index is None:
            g = self.pair_residual(s, shallow, deep)
            logit = logit + g.to(logit.dtype)
        else:
            g = self.pair_residual(s, shallow[residual_index], deep[residual_index])
            logit = logit.index_add(1, residual_index, g.to(logit.dtype))
        result = {"beta": logit.sigmoid(), "logit": logit, "mu_hat": bias.sigmoid()}
        if return_internals:
            result.update(cpg_factors=a, rna_factors=u, residual=g)
        return result
