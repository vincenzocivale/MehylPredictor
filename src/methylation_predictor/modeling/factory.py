"""Construction registry for retained paper-facing RNA methylation models."""
from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .baselines import BASELINE_VARIANTS, FunctionalBaselinePredictor
from .final import EfficientSingleAttentionPredictor
from .genomic_fm import GenomicFMLocusPredictor
from .rna_comparison import RNAEncoderComparisonPredictor


FINAL_VARIANT = "efficient_single_attn_8ffn_residual_functional8_head2"
FINAL_MODEL_KWARGS = {
    "n_ffn_blocks": 8,
    "n_functional_ffn_blocks": 8,
    "deep_query": False,
    "n_head_ffn_blocks": 2,
}

RNA_COMPARISON_VARIANT = "functional_rna_encoder_comparison"

# E04 genomic-FM baseline comparators (paper plan sec. E04). Each name maps
# to a chr1-only frozen embedding cache under METHYL_DATA_ROOT; NTv3-post is
# deliberately never added here -- excluded from the paper regardless of
# scope, see CLAUDE.md and MethylPredictor_Paper_Experiments_and_Codebase_Plan.md.
GENOMIC_FM_VARIANTS = {
    "genomic_fm_ntv3_pre",
}

SUPPORTED_FUNCTIONAL_VARIANTS = frozenset(
    {
        FINAL_VARIANT,
        RNA_COMPARISON_VARIANT,
        *BASELINE_VARIANTS,
        *GENOMIC_FM_VARIANTS,
    }
)


def architecture_label(variant: str, config: ModelConfig) -> str:
    if variant in BASELINE_VARIANTS:
        return variant
    if variant == RNA_COMPARISON_VARIANT:
        source = config.encoder.frozen_embedding_source
        suffix = f"-{source}" if source else f"-{config.encoder.kind}"
        return RNA_COMPARISON_VARIANT + suffix
    if variant == FINAL_VARIANT:
        return "methylpredictor_final"
    if variant in GENOMIC_FM_VARIANTS:
        return variant
    raise ValueError(f"unsupported functional model variant {variant!r}")


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
            input_dim,
            config,
            variant=variant,
            **common,
        )
    elif variant == RNA_COMPARISON_VARIANT:
        model = RNAEncoderComparisonPredictor(
            input_dim,
            config,
            **common,
        )
    elif variant in GENOMIC_FM_VARIANTS:
        model = GenomicFMLocusPredictor(
            input_dim,
            config,
            **common,
            **FINAL_MODEL_KWARGS,
        )
    else:
        model = EfficientSingleAttentionPredictor(
            input_dim,
            config,
            **common,
            **FINAL_MODEL_KWARGS,
        )

    return model, architecture_label(variant, config)
