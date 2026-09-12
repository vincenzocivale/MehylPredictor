"""Depth-vs-residual-vs-FFN-capacity ablation, disentangling J0 and J1.

J0 (``SingleRetrievalPredictor``) and J1 (``IterativeRetrievalPredictor``)
differ on several confounded dimensions at once: retrieval (cross-attention
against RNA) depth (1 vs 4 blocks), whether each block's cross-attention
output is added residually to the running locus state (never vs always), the
number of FFN blocks (J0 has ZERO -- its cross-attention output goes straight
to the final concat; J1 has 4, one per retrieval block), and parameter count.
An early, accidentally-built version of J1 (2 blocks, only the second
residual) tracked J0's training loss almost exactly, which is suggestive but
not conclusive on its own -- this module isolates the axes cleanly:

  ``ablation_depth1_residual``    -- n_blocks=1, attn_residual=True
      Same attention depth as J0 (1 cross-attention call), but with the
      residual J1 has (and, like J1, one FFN block after it -- J0 has none).
      Isolates whether the residual ALONE (no extra RNA-attention depth,
      though WITH the extra FFN capacity J1 also carries) already produces
      J1's gain.

  ``ablation_depth4_noresidual``  -- n_blocks=4, attn_residual=False
      Same depth as J1 (4 cross-attention + 4 FFN blocks), but every block's
      attention output REPLACES the running state instead of being added to
      it (matching the accidental first J1 attempt's mechanism, applied
      uniformly to every block instead of only the first). Isolates whether
      depth ALONE (no residual) produces any gain over J0.

  ``ablation_attn1_ffn4_residual`` -- n_blocks=1, attn_residual=True,
      n_ffn_only_blocks=3
      Only ONE real RNA cross-attention call (same as J0/cell A -- the
      expensive, memory-heavy operation), followed by 3 EXTRA residual
      FFN-only blocks that never touch RNA again, so the TOTAL block count
      (4) and FFN count (4) match J1's exactly. Isolates whether J1's gain
      comes from repeatedly re-querying RNA (attention depth) or simply from
      more generic residual non-linear capacity/depth (which is far cheaper
      to compute than repeated cross-attention against the RNA program
      tokens) -- the question of whether "servono più blocchi ma non pesanti
      come la cross attention" explains the gain instead of attention depth
      itself.

Everything else (functional encoder, mean head, final regressor, loss,
training protocol) is identical to J0/J1 -- see their docstrings in
``reference.py``.

``EfficientSingleAttentionPredictor`` (2026-09-11, J4) is a follow-up
candidate, not a diagnostic cell: once the ``ablation_depth1_residual``
result above supports "the residual matters, repeating cross-attention
doesn't", it tests whether J1's gain survives paying the expensive
cross-attention op only ONCE instead of 4x. It runs a single
locus<-RNA cross-attention with J1's residual add, then refines that
state through ``n_ffn_blocks`` (default 4, matching J1's total FFN depth)
residual FFN blocks with no further attention -- same residual-block
depth as J1, 4x fewer cross-attention evaluations (the dominant cost at
genome-wide CpG counts).

VRAM scaling with ``n_blocks`` (2026-09-11, one RTX PRO 5000, isolated
process, the real array-pool batch 640x640 which is the binding case --
it is never gradient-checkpointed, unlike epic/wgbs whose cpg_size exceeds
``LOCUS_CHUNK``): peak memory is ~linear, roughly 3.0 + 4.3*n_blocks GB
(measured n_blocks=4 -> 20.17GB, n_blocks=6 -> 28.77GB; n_blocks=8 OOM'd
when an unrelated ~17GB was already in use by another process on the same
GPU). Re-verify empirically before trusting this extrapolated past
n_blocks=6 -- see ``ablation_depth8_residual``/``_depth10_residual``/
``_depth12_residual`` in ``LocusCLSJointTrainer.ablation_variants``, meant
to be smoke-tested in increasing order once the GPU is free.

``GatedResidualPredictor`` (2026-09-11, J6) replaces J1's unconditional
per-block residual add (``state = state + attn(...)``) with a Flamingo-style
("gated cross-attention", DeepMind 2022) learned scalar gate per block:
``state = state + tanh(alpha_i) * attn(...)``, ``alpha_i`` a scalar
parameter initialized to 0 (so at init the block starts as pure identity,
matching ReZero's init trick -- no depth-specific mechanism like
mHC/RealFormer/AttnRes, this is a per-block gate that works identically at
n_blocks=1). Costs one extra scalar per block; everything else (encoder,
mean head, final regressor, chunking) is identical to
``DepthResidualAblationPredictor``. Diagnostic for free: if the trained
gates saturate near +-1, the plain (unconditional) residual was already
optimal for this block; if they land well short of that, the model
benefits from being able to partially close it.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .retrieval import BatchedLocusToRNAAttention, FeedForwardResidual
from .rna import ProgramTokenEncoder, consume_legacy_attention_initialization


class DepthResidualAblationPredictor(nn.Module):
    """Configurable-depth, optionally-residual RNA-retrieval stack."""

    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256
    LOCUS_CHUNK = 1024

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        n_blocks: int,
        attn_residual: bool,
        n_ffn_only_blocks: int = 0,
        final_regressor_dropout: float = 0.15,
        use_mean_proxy: bool = True,
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        if n_ffn_only_blocks < 0:
            raise ValueError("n_ffn_only_blocks must be >= 0")
        self.n_blocks = int(n_blocks)
        self.attn_residual = bool(attn_residual)
        self.n_ffn_only_blocks = int(n_ffn_only_blocks)
        enc = config.encoder
        if enc.kind != "locus_attention" or enc.program_dim != self.WIDTH:
            raise ValueError(
                "DepthResidualAblationPredictor requires locus_attention RNA "
                "tokens with program_dim=256"
            )

        self.rna_encoder = ProgramTokenEncoder(
            input_dim=input_dim,
            n_programs=enc.n_programs,
            program_dim=enc.program_dim,
            bottleneck_dim=enc.latent_dim,
            layer_norm=enc.layer_norm,
        )
        consume_legacy_attention_initialization(self.WIDTH)

        self.track_embedding = nn.EmbeddingBag(
            self.N_TRACKS, self.WIDTH, mode="mean", include_last_offset=True,
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(self.DENSE_DIM), nn.Linear(self.DENSE_DIM, self.WIDTH), nn.GELU(),
        )
        self.locus_norm = nn.LayerNorm(self.WIDTH)
        mean_head = nn.Sequential(
            nn.LayerNorm(self.WIDTH), nn.Linear(self.WIDTH, 128), nn.GELU(), nn.Linear(128, 1),
        )
        self.mean_head = mean_head if use_mean_proxy else None
        self.retrieval_attn = nn.ModuleList([
            BatchedLocusToRNAAttention(self.WIDTH, enc.n_heads, enc.dropout) for _ in range(self.n_blocks)
        ])
        self.retrieval_ffn = nn.ModuleList([
            FeedForwardResidual(self.WIDTH, enc.dropout) for _ in range(self.n_blocks)
        ])
        # Extra FFN-only blocks: no attention, never touch RNA again -- pure
        # non-linear capacity/depth on the already RNA-conditioned state, at
        # a fraction of cross-attention's compute/memory cost. Always
        # residual (FeedForwardResidual bakes its own residual in
        # regardless of `attn_residual`, which only governs the attention
        # step above).
        self.ffn_only_blocks = nn.ModuleList([
            FeedForwardResidual(self.WIDTH, enc.dropout) for _ in range(self.n_ffn_only_blocks)
        ])
        self.h_c_concat_norm = nn.LayerNorm(self.WIDTH)
        self.r_pc_concat_norm = nn.LayerNorm(self.WIDTH)
        self.final_regressor_dropout = float(final_regressor_dropout)
        self.final_regressor = nn.Sequential(
            nn.Linear(2 * self.WIDTH, self.WIDTH), nn.GELU(), nn.Dropout(self.final_regressor_dropout),
            nn.Linear(self.WIDTH, 128), nn.GELU(), nn.Dropout(self.final_regressor_dropout),
            nn.Linear(128, 1),
        )

    @property
    def requires_cpg_positions(self) -> bool:
        return False

    def _retrieve_and_predict(self, h_c_chunk: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        state = h_c_chunk[None, :, :].expand(batch, -1, -1)
        for attn, ffn in zip(self.retrieval_attn, self.retrieval_ffn):
            a = attn(state, tokens)
            state = state + a if self.attn_residual else a
            state = ffn(state)  # FeedForwardResidual: always has its own internal residual
        for ffn_only in self.ffn_only_blocks:
            state = ffn_only(state)  # no RNA re-query, just extra residual FFN capacity

        h_for_concat = self.h_c_concat_norm(h_c_chunk)
        r_for_concat = self.r_pc_concat_norm(state)
        z_pc = torch.cat(
            [h_for_concat[None, :, :].expand(batch, -1, -1), r_for_concat], dim=-1,
        )
        return self.final_regressor(z_pc).squeeze(-1)

    def forward(
        self,
        rna: torch.Tensor,
        cpg_embedding: torch.Tensor | None = None,
        cpg_positions: torch.Tensor | None = None,
        *,
        functional_track_indices: torch.Tensor,
        functional_offsets: torch.Tensor,
        functional_dense: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del cpg_embedding, cpg_positions
        tokens = self.rna_encoder(rna)

        peak = self.track_embedding(functional_track_indices, functional_offsets)
        dense = self.dense_encoder(functional_dense)
        h_c = self.locus_norm(peak + dense)
        mu_hat = (
            None if self.mean_head is None
            else torch.sigmoid(self.mean_head(h_c).squeeze(-1))
        )

        n_loci = h_c.shape[0]
        if n_loci <= self.LOCUS_CHUNK:
            logit = self._retrieve_and_predict(h_c, tokens)
        else:
            chunks = []
            for start in range(0, n_loci, self.LOCUS_CHUNK):
                chunk = h_c[start:start + self.LOCUS_CHUNK]
                if self.training and torch.is_grad_enabled():
                    chunks.append(
                        checkpoint(self._retrieve_and_predict, chunk, tokens, use_reentrant=False)
                    )
                else:
                    chunks.append(self._retrieve_and_predict(chunk, tokens))
            logit = torch.cat(chunks, dim=1)

        beta = torch.sigmoid(logit)
        return {
            "beta": beta,
            "delta_logit": logit,
            "raw_delta": logit,
            "prediction_logit": logit,
            "mu_logit": None,
            "mu_hat": mu_hat,
            "residual_logit": logit,
            "h_cpg": h_c,
        }


class EfficientSingleAttentionPredictor(nn.Module):
    """J4: single cross-attention + residual, then several FFN-only residual blocks.

    Candidate answer to "is the repeated cross-attention itself necessary,
    or just the residual-block depth it happens to come with" -- see the
    module docstring. Structurally identical to
    ``DepthResidualAblationPredictor(n_blocks=1, attn_residual=True)`` up to
    and including the residual add, then keeps refining `state` through
    ``n_ffn_blocks`` independent ``FeedForwardResidual`` blocks with no
    further attention, instead of stopping after one.

    J9b (2026-09-12) asks the same depth question about the OTHER branch:
    J7/J8 showed FFN depth helps the RNA-conditioned retrieval state
    (``state``) -- does it also help the functional-annotation branch
    (``h_c``, from ``track_embedding``+``dense_encoder``), which today is
    just two shallow encoders summed and normalized? ``n_functional_ffn_blocks``
    (default 0 = J4/J7/J8 behavior, unchanged) runs ``h_c`` through that many
    extra ``FeedForwardResidual`` blocks to get ``h_c_deep``, used for
    ``mean_head`` and the final concat.

    Deliberately NOT fed into the cross-attention query: the query
    (``state``'s initial value) stays the shallow, un-deepened ``h_c`` --
    identical to J7's retrieval mechanism -- so a J9b gain can only be
    attributed to "a richer functional representation improves the
    downstream heads", not to "a different query changed what gets
    retrieved from RNA" (a second, separate hypothesis). ``deep_query=True``
    is the deliberate follow-up (J9c) that lets ``h_c_deep`` drive the query
    too, once J9b establishes there is a gain worth attributing.
    """

    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256
    LOCUS_CHUNK = 1024

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        n_ffn_blocks: int = 4,
        n_functional_ffn_blocks: int = 0,
        deep_query: bool = False,
        final_regressor_dropout: float = 0.15,
        use_mean_proxy: bool = True,
    ):
        super().__init__()
        if n_ffn_blocks < 1:
            raise ValueError("n_ffn_blocks must be >= 1")
        if n_functional_ffn_blocks < 0:
            raise ValueError("n_functional_ffn_blocks must be >= 0")
        if deep_query and n_functional_ffn_blocks < 1:
            raise ValueError("deep_query requires n_functional_ffn_blocks >= 1")
        self.n_ffn_blocks = int(n_ffn_blocks)
        self.n_functional_ffn_blocks = int(n_functional_ffn_blocks)
        self.deep_query = bool(deep_query)
        enc = config.encoder
        if enc.kind != "locus_attention" or enc.program_dim != self.WIDTH:
            raise ValueError(
                "EfficientSingleAttentionPredictor requires locus_attention "
                "RNA tokens with program_dim=256"
            )

        self.rna_encoder = ProgramTokenEncoder(
            input_dim=input_dim,
            n_programs=enc.n_programs,
            program_dim=enc.program_dim,
            bottleneck_dim=enc.latent_dim,
            layer_norm=enc.layer_norm,
        )
        consume_legacy_attention_initialization(self.WIDTH)

        self.track_embedding = nn.EmbeddingBag(
            self.N_TRACKS, self.WIDTH, mode="mean", include_last_offset=True,
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(self.DENSE_DIM), nn.Linear(self.DENSE_DIM, self.WIDTH), nn.GELU(),
        )
        self.locus_norm = nn.LayerNorm(self.WIDTH)
        mean_head = nn.Sequential(
            nn.LayerNorm(self.WIDTH), nn.Linear(self.WIDTH, 128), nn.GELU(), nn.Linear(128, 1),
        )
        self.mean_head = mean_head if use_mean_proxy else None
        # Cross-attention runs exactly once -- this is the expensive op the
        # depth-4 candidates pay for 4x; everything after it is cheap FFN.
        self.retrieval_attn = BatchedLocusToRNAAttention(self.WIDTH, enc.n_heads, enc.dropout)
        self.retrieval_ffn = nn.ModuleList([
            FeedForwardResidual(self.WIDTH, enc.dropout) for _ in range(self.n_ffn_blocks)
        ])
        # J9b: extra non-linear capacity on the functional-annotation branch
        # alone (h_c -> h_c_deep), never touching the RNA side. Cheap: h_c is
        # [n_loci, WIDTH], not replicated per RNA sample like `state` is.
        self.functional_ffn = nn.ModuleList([
            FeedForwardResidual(self.WIDTH, enc.dropout) for _ in range(self.n_functional_ffn_blocks)
        ])
        self.h_c_concat_norm = nn.LayerNorm(self.WIDTH)
        self.r_pc_concat_norm = nn.LayerNorm(self.WIDTH)
        self.final_regressor_dropout = float(final_regressor_dropout)
        self.final_regressor = nn.Sequential(
            nn.Linear(2 * self.WIDTH, self.WIDTH), nn.GELU(), nn.Dropout(self.final_regressor_dropout),
            nn.Linear(self.WIDTH, 128), nn.GELU(), nn.Dropout(self.final_regressor_dropout),
            nn.Linear(128, 1),
        )

    @property
    def requires_cpg_positions(self) -> bool:
        return False

    def _retrieve_and_predict(
        self, h_c_query_chunk: torch.Tensor, h_c_concat_chunk: torch.Tensor, tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        state = h_c_query_chunk[None, :, :].expand(batch, -1, -1)
        a = self.retrieval_attn(state, tokens)
        state = state + a
        for ffn in self.retrieval_ffn:
            state = ffn(state)  # FeedForwardResidual: always has its own internal residual

        h_for_concat = self.h_c_concat_norm(h_c_concat_chunk)
        r_for_concat = self.r_pc_concat_norm(state)
        z_pc = torch.cat(
            [h_for_concat[None, :, :].expand(batch, -1, -1), r_for_concat], dim=-1,
        )
        return self.final_regressor(z_pc).squeeze(-1)

    def forward(
        self,
        rna: torch.Tensor,
        cpg_embedding: torch.Tensor | None = None,
        cpg_positions: torch.Tensor | None = None,
        *,
        functional_track_indices: torch.Tensor,
        functional_offsets: torch.Tensor,
        functional_dense: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del cpg_embedding, cpg_positions
        tokens = self.rna_encoder(rna)

        peak = self.track_embedding(functional_track_indices, functional_offsets)
        dense = self.dense_encoder(functional_dense)
        h_c = self.locus_norm(peak + dense)
        # J9b: extra FFN capacity on the functional-annotation branch alone.
        # h_c_deep feeds mean_head and the final concat; h_c_query (the
        # cross-attention query) stays the shallow h_c unless deep_query=True
        # (J9c) -- see the class docstring for why these are kept separate.
        h_c_deep = h_c
        for ffn in self.functional_ffn:
            h_c_deep = ffn(h_c_deep)
        h_c_query = h_c_deep if self.deep_query else h_c
        mu_hat = (
            None if self.mean_head is None
            else torch.sigmoid(self.mean_head(h_c_deep).squeeze(-1))
        )

        n_loci = h_c.shape[0]
        if n_loci <= self.LOCUS_CHUNK:
            logit = self._retrieve_and_predict(h_c_query, h_c_deep, tokens)
        else:
            chunks = []
            for start in range(0, n_loci, self.LOCUS_CHUNK):
                query_chunk = h_c_query[start:start + self.LOCUS_CHUNK]
                concat_chunk = h_c_deep[start:start + self.LOCUS_CHUNK]
                if self.training and torch.is_grad_enabled():
                    chunks.append(
                        checkpoint(
                            self._retrieve_and_predict, query_chunk, concat_chunk, tokens,
                            use_reentrant=False,
                        )
                    )
                else:
                    chunks.append(self._retrieve_and_predict(query_chunk, concat_chunk, tokens))
            logit = torch.cat(chunks, dim=1)

        beta = torch.sigmoid(logit)
        return {
            "beta": beta,
            "delta_logit": logit,
            "raw_delta": logit,
            "prediction_logit": logit,
            "mu_logit": None,
            "mu_hat": mu_hat,
            "residual_logit": logit,
            "h_cpg": h_c,
        }


class GatedResidualPredictor(nn.Module):
    """J6: J1-style per-block residual, but the add is gated by a learned
    scalar (Flamingo/"gated cross-attention" style) instead of unconditional.

    ``state = state + tanh(alpha_i) * attn(state, tokens)`` per block, with
    ``alpha_i`` a free scalar parameter per block initialized to 0 (block
    starts as identity, same trick as ReZero). Works identically at any
    ``n_blocks`` including 1 -- unlike mHC/RealFormer/AttnRes this is not a
    depth-specific mechanism, see the module docstring.
    """

    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256
    LOCUS_CHUNK = 1024

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        n_blocks: int = 1,
        final_regressor_dropout: float = 0.15,
        use_mean_proxy: bool = True,
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        self.n_blocks = int(n_blocks)
        enc = config.encoder
        if enc.kind != "locus_attention" or enc.program_dim != self.WIDTH:
            raise ValueError(
                "GatedResidualPredictor requires locus_attention RNA tokens "
                "with program_dim=256"
            )

        self.rna_encoder = ProgramTokenEncoder(
            input_dim=input_dim,
            n_programs=enc.n_programs,
            program_dim=enc.program_dim,
            bottleneck_dim=enc.latent_dim,
            layer_norm=enc.layer_norm,
        )
        consume_legacy_attention_initialization(self.WIDTH)

        self.track_embedding = nn.EmbeddingBag(
            self.N_TRACKS, self.WIDTH, mode="mean", include_last_offset=True,
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(self.DENSE_DIM), nn.Linear(self.DENSE_DIM, self.WIDTH), nn.GELU(),
        )
        self.locus_norm = nn.LayerNorm(self.WIDTH)
        mean_head = nn.Sequential(
            nn.LayerNorm(self.WIDTH), nn.Linear(self.WIDTH, 128), nn.GELU(), nn.Linear(128, 1),
        )
        self.mean_head = mean_head if use_mean_proxy else None
        self.retrieval_attn = nn.ModuleList([
            BatchedLocusToRNAAttention(self.WIDTH, enc.n_heads, enc.dropout) for _ in range(self.n_blocks)
        ])
        self.retrieval_ffn = nn.ModuleList([
            FeedForwardResidual(self.WIDTH, enc.dropout) for _ in range(self.n_blocks)
        ])
        # One scalar per block, init 0 -> tanh(0)=0 -> block starts as
        # identity (ReZero-style init), same param count regardless of depth.
        self.residual_gate = nn.Parameter(torch.zeros(self.n_blocks))
        self.h_c_concat_norm = nn.LayerNorm(self.WIDTH)
        self.r_pc_concat_norm = nn.LayerNorm(self.WIDTH)
        self.final_regressor_dropout = float(final_regressor_dropout)
        self.final_regressor = nn.Sequential(
            nn.Linear(2 * self.WIDTH, self.WIDTH), nn.GELU(), nn.Dropout(self.final_regressor_dropout),
            nn.Linear(self.WIDTH, 128), nn.GELU(), nn.Dropout(self.final_regressor_dropout),
            nn.Linear(128, 1),
        )

    @property
    def requires_cpg_positions(self) -> bool:
        return False

    def _retrieve_and_predict(self, h_c_chunk: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        state = h_c_chunk[None, :, :].expand(batch, -1, -1)
        for i, (attn, ffn) in enumerate(zip(self.retrieval_attn, self.retrieval_ffn)):
            a = attn(state, tokens)
            gate = torch.tanh(self.residual_gate[i])
            state = state + gate * a
            state = ffn(state)  # FeedForwardResidual: always has its own internal residual

        h_for_concat = self.h_c_concat_norm(h_c_chunk)
        r_for_concat = self.r_pc_concat_norm(state)
        z_pc = torch.cat(
            [h_for_concat[None, :, :].expand(batch, -1, -1), r_for_concat], dim=-1,
        )
        return self.final_regressor(z_pc).squeeze(-1)

    def forward(
        self,
        rna: torch.Tensor,
        cpg_embedding: torch.Tensor | None = None,
        cpg_positions: torch.Tensor | None = None,
        *,
        functional_track_indices: torch.Tensor,
        functional_offsets: torch.Tensor,
        functional_dense: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del cpg_embedding, cpg_positions
        tokens = self.rna_encoder(rna)

        peak = self.track_embedding(functional_track_indices, functional_offsets)
        dense = self.dense_encoder(functional_dense)
        h_c = self.locus_norm(peak + dense)
        mu_hat = (
            None if self.mean_head is None
            else torch.sigmoid(self.mean_head(h_c).squeeze(-1))
        )

        n_loci = h_c.shape[0]
        if n_loci <= self.LOCUS_CHUNK:
            logit = self._retrieve_and_predict(h_c, tokens)
        else:
            chunks = []
            for start in range(0, n_loci, self.LOCUS_CHUNK):
                chunk = h_c[start:start + self.LOCUS_CHUNK]
                if self.training and torch.is_grad_enabled():
                    chunks.append(
                        checkpoint(self._retrieve_and_predict, chunk, tokens, use_reentrant=False)
                    )
                else:
                    chunks.append(self._retrieve_and_predict(chunk, tokens))
            logit = torch.cat(chunks, dim=1)

        beta = torch.sigmoid(logit)
        return {
            "beta": beta,
            "delta_logit": logit,
            "raw_delta": logit,
            "prediction_logit": logit,
            "mu_logit": None,
            "mu_hat": mu_hat,
            "residual_logit": logit,
            "h_cpg": h_c,
        }
