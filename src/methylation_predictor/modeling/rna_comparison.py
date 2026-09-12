"""Functional-matched RNA encoder comparison model."""
from __future__ import annotations

from dataclasses import replace

from ..config import ModelConfig
from .final import EfficientSingleAttentionPredictor
from .rna_comparators import build_comparison_program_encoder


class RNAEncoderComparisonPredictor(EfficientSingleAttentionPredictor):
    """Final architecture with only the upstream RNA encoder replaced."""

    REFERENCE_RNA_INPUT_DIM = 25017

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        final_regressor_dropout: float = 0.15,
        use_mean_proxy: bool = True,
    ):
        comparison_encoder = config.encoder

        reference_encoder = replace(
            comparison_encoder,
            kind="locus_attention",
        )
        reference_config = replace(
            config,
            encoder=reference_encoder,
        )

        super().__init__(
            self.REFERENCE_RNA_INPUT_DIM,
            reference_config,
            n_ffn_blocks=8,
            n_functional_ffn_blocks=8,
            deep_query=False,
            n_head_ffn_blocks=2,
            final_regressor_dropout=final_regressor_dropout,
            use_mean_proxy=use_mean_proxy,
        )

        if not (
            comparison_encoder.kind == "locus_attention"
            and input_dim == self.REFERENCE_RNA_INPUT_DIM
        ):
            self.rna_encoder = build_comparison_program_encoder(
                comparison_encoder,
                input_dim,
            )

        self.comparison_encoder_kind = comparison_encoder.kind
        self.comparison_encoder_source = (
            comparison_encoder.frozen_embedding_source
        )
