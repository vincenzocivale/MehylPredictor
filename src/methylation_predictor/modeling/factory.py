"""Construction registry for the live functional RNA-methylation models.

The trainer should not know experiment-specific architecture details. This
module is the single mapping from recipe selector -> predictor constructor.
"""
from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .ablation import (
    DepthResidualAblationPredictor,
    EfficientSingleAttentionPredictor,
    FunctionalGeneFFNFusionPredictor,
    GatedResidualPredictor,
)
from .baselines import BASELINE_VARIANTS, FunctionalBaselinePredictor
from .reference import IterativeRetrievalPredictor, SingleRetrievalPredictor
from .rna_comparison import RNAEncoderComparisonPredictor


DEPTH_RESIDUAL_VARIANTS = {
    "ablation_depth1_residual": {"n_blocks": 1, "attn_residual": True},
    "ablation_depth4_noresidual": {"n_blocks": 4, "attn_residual": False},
    "ablation_depth8_residual": {"n_blocks": 8, "attn_residual": True},
    "ablation_depth10_residual": {"n_blocks": 10, "attn_residual": True},
    "ablation_depth12_residual": {"n_blocks": 12, "attn_residual": True},
}

EFFICIENT_VARIANTS = {
    "efficient_single_attn_residual_ffn": {"n_ffn_blocks": 4},
    "efficient_single_attn_8ffn_residual": {"n_ffn_blocks": 8},
    "efficient_single_attn_8ffn_residual_functional8": {
        "n_ffn_blocks": 8,
        "n_functional_ffn_blocks": 8,
        "deep_query": False,
    },
    "efficient_single_attn_4ffn_residual_functional4": {
        "n_ffn_blocks": 4,
        "n_functional_ffn_blocks": 4,
        "deep_query": False,
    },
}

GATED_VARIANTS = {
    "ablation_depth1_gated_residual": {"n_blocks": 1},
}

FFN_FUSION_VARIANTS = {
    "ffn_fusion_concat": {"fusion_mode": "concat"},
    "ffn_fusion_film": {"fusion_mode": "film"},
    "ffn_fusion_two_stream_residual": {
        "fusion_mode": "two_stream_residual",
    },
    "ffn_fusion_two_stream_residual_8_8": {
        "fusion_mode": "two_stream_residual",
        "n_functional_ffn_blocks": 8,
        "n_gene_expr_ffn_blocks": 8,
        "fusion_dropout": 0.1,
    },
    "ffn_fusion_two_stream_residual_4_4": {
        "fusion_mode": "two_stream_residual",
        "n_functional_ffn_blocks": 4,
        "n_gene_expr_ffn_blocks": 4,
        "fusion_dropout": 0.1,
    },
}

REFERENCE_VARIANTS = {
    "mas_concat_v3_purecontext",
    "mas_concat_v4_iterative",
}

RNA_COMPARISON_VARIANT = "functional_rna_encoder_comparison"

SUPPORTED_FUNCTIONAL_VARIANTS = frozenset(
    {
        *REFERENCE_VARIANTS,
        RNA_COMPARISON_VARIANT,
        *BASELINE_VARIANTS,
        *DEPTH_RESIDUAL_VARIANTS,
        *EFFICIENT_VARIANTS,
        *GATED_VARIANTS,
        *FFN_FUSION_VARIANTS,
    }
)


def architecture_label(variant: str, config: ModelConfig) -> str:
    if variant in BASELINE_VARIANTS:
        return variant
    if variant == RNA_COMPARISON_VARIANT:
        source = config.encoder.frozen_embedding_source
        suffix = f"-{source}" if source else f"-{config.encoder.kind}"
        return RNA_COMPARISON_VARIANT + suffix
    return f"functional_concat_{variant}"


def build_functional_predictor(
    *,
    variant: str,
    input_dim: int,
    config: ModelConfig,
    final_regressor_dropout: float,
    use_mean_proxy: bool,
) -> tuple[nn.Module, str]:
    if variant not in SUPPORTED_FUNCTIONAL_VARIANTS:
        raise ValueError(
            f"unsupported functional model variant {variant!r}; expected one "
            f"of {sorted(SUPPORTED_FUNCTIONAL_VARIANTS)}"
        )

    common = {
        "final_regressor_dropout": final_regressor_dropout,
        "use_mean_proxy": use_mean_proxy,
    }

    if variant in BASELINE_VARIANTS:
        model = FunctionalBaselinePredictor(
            input_dim, config, variant=variant, **common
        )
    elif variant == RNA_COMPARISON_VARIANT:
        model = RNAEncoderComparisonPredictor(input_dim, config, **common)
    elif variant in DEPTH_RESIDUAL_VARIANTS:
        model = DepthResidualAblationPredictor(
            input_dim,
            config,
            **common,
            **DEPTH_RESIDUAL_VARIANTS[variant],
        )
    elif variant in EFFICIENT_VARIANTS:
        model = EfficientSingleAttentionPredictor(
            input_dim,
            config,
            **common,
            **EFFICIENT_VARIANTS[variant],
        )
    elif variant in GATED_VARIANTS:
        model = GatedResidualPredictor(
            input_dim,
            config,
            **common,
            **GATED_VARIANTS[variant],
        )
    elif variant in FFN_FUSION_VARIANTS:
        model = FunctionalGeneFFNFusionPredictor(
            input_dim,
            config,
            **common,
            **FFN_FUSION_VARIANTS[variant],
        )
    elif variant == "mas_concat_v4_iterative":
        model = IterativeRetrievalPredictor(input_dim, config, **common)
    else:
        model = SingleRetrievalPredictor(input_dim, config, **common)

    return model, architecture_label(variant, config)
