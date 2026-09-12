"""Final paper-facing MethylPredictor architecture."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .retrieval import BatchedLocusToRNAAttention, FeedForwardResidual
from .rna import ProgramTokenEncoder, consume_legacy_attention_initialization


class EfficientSingleAttentionPredictor(nn.Module):
    """Single RNA retrieval with deep functional/retrieval/head FFN refinement."""

    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256
    LOCUS_CHUNK = 1024

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        n_ffn_blocks: int = 8,
        n_functional_ffn_blocks: int = 8,
        deep_query: bool = False,
        n_head_ffn_blocks: int = 2,
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
        if n_head_ffn_blocks < 0:
            raise ValueError("n_head_ffn_blocks must be >= 0")

        self.n_ffn_blocks = int(n_ffn_blocks)
        self.n_functional_ffn_blocks = int(n_functional_ffn_blocks)
        self.deep_query = bool(deep_query)
        self.n_head_ffn_blocks = int(n_head_ffn_blocks)

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
            self.N_TRACKS,
            self.WIDTH,
            mode="mean",
            include_last_offset=True,
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(self.DENSE_DIM),
            nn.Linear(self.DENSE_DIM, self.WIDTH),
            nn.GELU(),
        )
        self.locus_norm = nn.LayerNorm(self.WIDTH)

        self.functional_ffn = nn.ModuleList(
            [
                FeedForwardResidual(self.WIDTH, enc.dropout)
                for _ in range(self.n_functional_ffn_blocks)
            ]
        )

        mean_head = nn.Sequential(
            nn.LayerNorm(self.WIDTH),
            nn.Linear(self.WIDTH, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.mean_head = mean_head if use_mean_proxy else None

        self.retrieval_attn = BatchedLocusToRNAAttention(
            self.WIDTH,
            enc.n_heads,
            enc.dropout,
        )
        self.retrieval_ffn = nn.ModuleList(
            [
                FeedForwardResidual(self.WIDTH, enc.dropout)
                for _ in range(self.n_ffn_blocks)
            ]
        )

        self.h_c_concat_norm = nn.LayerNorm(self.WIDTH)
        self.r_pc_concat_norm = nn.LayerNorm(self.WIDTH)
        self.final_regressor_dropout = float(final_regressor_dropout)

        self.final_regressor = nn.Sequential(
            nn.Linear(2 * self.WIDTH, self.WIDTH),
            nn.GELU(),
            nn.Dropout(self.final_regressor_dropout),
            nn.Linear(self.WIDTH, 128),
            nn.GELU(),
            nn.Dropout(self.final_regressor_dropout),
            nn.Linear(128, 1),
        )
        self.head_ffn = nn.ModuleList(
            [
                FeedForwardResidual(
                    self.WIDTH,
                    self.final_regressor_dropout,
                )
                for _ in range(self.n_head_ffn_blocks)
            ]
        )

    @property
    def requires_cpg_positions(self) -> bool:
        return False

    def _retrieve_and_predict(
        self,
        h_c_query_chunk: torch.Tensor,
        h_c_concat_chunk: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch = tokens.shape[0]

        state = h_c_query_chunk[None, :, :].expand(batch, -1, -1)
        state = state + self.retrieval_attn(state, tokens)

        for ffn in self.retrieval_ffn:
            state = ffn(state)

        h_for_concat = self.h_c_concat_norm(h_c_concat_chunk)
        r_for_concat = self.r_pc_concat_norm(state)
        z_pc = torch.cat(
            [
                h_for_concat[None, :, :].expand(batch, -1, -1),
                r_for_concat,
            ],
            dim=-1,
        )

        head = self.final_regressor[0](z_pc)
        head = self.final_regressor[1](head)
        head = self.final_regressor[2](head)

        for ffn in self.head_ffn:
            head = ffn(head)

        head = self.final_regressor[3](head)
        head = self.final_regressor[4](head)
        head = self.final_regressor[5](head)
        return self.final_regressor[6](head).squeeze(-1)

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

        peak = self.track_embedding(
            functional_track_indices,
            functional_offsets,
        )
        dense = self.dense_encoder(functional_dense)
        h_c = self.locus_norm(peak + dense)

        h_c_deep = h_c
        for ffn in self.functional_ffn:
            h_c_deep = ffn(h_c_deep)

        h_c_query = h_c_deep if self.deep_query else h_c

        mu_hat = (
            None
            if self.mean_head is None
            else torch.sigmoid(self.mean_head(h_c_deep).squeeze(-1))
        )

        n_loci = h_c.shape[0]
        if n_loci <= self.LOCUS_CHUNK:
            logit = self._retrieve_and_predict(
                h_c_query,
                h_c_deep,
                tokens,
            )
        else:
            chunks = []
            for start in range(0, n_loci, self.LOCUS_CHUNK):
                query_chunk = h_c_query[
                    start : start + self.LOCUS_CHUNK
                ]
                concat_chunk = h_c_deep[
                    start : start + self.LOCUS_CHUNK
                ]

                if self.training and torch.is_grad_enabled():
                    chunks.append(
                        checkpoint(
                            self._retrieve_and_predict,
                            query_chunk,
                            concat_chunk,
                            tokens,
                            use_reentrant=False,
                        )
                    )
                else:
                    chunks.append(
                        self._retrieve_and_predict(
                            query_chunk,
                            concat_chunk,
                            tokens,
                        )
                    )

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
            "h_cpg": h_c_deep,
        }
