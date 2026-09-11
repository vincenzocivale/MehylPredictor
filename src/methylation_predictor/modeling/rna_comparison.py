"""Functional-matched RNA encoder comparison model."""
from __future__ import annotations

from dataclasses import replace

from ..config import ModelConfig
from .reference import SingleRetrievalPredictor
from .rna_comparators import build_comparison_program_encoder


class RNAEncoderComparisonPredictor(SingleRetrievalPredictor):
    """J0 with only the upstream patient RNA encoder replaced.

    The functional CpG encoder, retrieval attention, mean-proxy head and final
    beta regressor are constructed by ``SingleRetrievalPredictor`` and retain
    its exact same-seed initialization.

    The canonical RNA axis used by ``main.yaml`` has 25,017 columns.  We build
    that reference J0 first even for frozen-embedding arms, then replace only
    ``rna_encoder``.  This decouples downstream initialization from comparator
    input width/capacity.
    """

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
            final_regressor_dropout=final_regressor_dropout,
            use_mean_proxy=use_mean_proxy,
        )

        # For the reference arm on the canonical 25,017-gene cache, the
        # encoder created by super() already is exactly the production J0
        # ProgramTokenEncoder, so retain it bit-for-bit.
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
