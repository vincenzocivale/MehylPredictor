from __future__ import annotations

import pytest

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import (
    EfficientSingleAttentionPredictor,
    FunctionalBaselinePredictor,
    RNAEncoderComparisonPredictor,
)
from methylation_predictor.modeling.factory import (
    FINAL_MODEL_KWARGS,
    FINAL_VARIANT,
    RNA_COMPARISON_VARIANT,
    SUPPORTED_FUNCTIONAL_VARIANTS,
    build_functional_predictor,
)


def _config() -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(
            kind="locus_attention",
            latent_dim=16,
            n_programs=6,
            program_dim=256,
            n_heads=4,
            dropout=0.0,
        ),
    )


def test_registry_contains_only_retained_model_families():
    assert FINAL_VARIANT in SUPPORTED_FUNCTIONAL_VARIANTS
    assert RNA_COMPARISON_VARIANT in SUPPORTED_FUNCTIONAL_VARIANTS

    retired_fragments = (
        "ablation_",
        "mas_concat_",
        "ffn_fusion_",
        "gated",
        "iterative",
    )
    assert not any(
        any(fragment in variant for fragment in retired_fragments)
        for variant in SUPPORTED_FUNCTIONAL_VARIANTS
    )


def test_final_factory_contract():
    assert FINAL_MODEL_KWARGS == {
        "n_ffn_blocks": 8,
        "n_functional_ffn_blocks": 8,
        "deep_query": False,
        "n_head_ffn_blocks": 2,
    }

    model, label = build_functional_predictor(
        variant=FINAL_VARIANT,
        input_dim=48,
        config=_config(),
        final_regressor_dropout=0.15,
        use_mean_proxy=True,
    )
    assert isinstance(model, EfficientSingleAttentionPredictor)
    assert label == "methylpredictor_final"
    assert len(model.retrieval_ffn) == 8
    assert len(model.functional_ffn) == 8
    assert len(model.head_ffn) == 2
    assert model.deep_query is False


@pytest.mark.parametrize(
    "variant",
    [
        "functional_baseline_global_shift",
        "functional_baseline_mlp",
        "functional_baseline_bilinear",
    ],
)
def test_retained_baselines_build(variant):
    model, _ = build_functional_predictor(
        variant=variant,
        input_dim=48,
        config=_config(),
        final_regressor_dropout=0.15,
        use_mean_proxy=True,
    )
    assert isinstance(model, FunctionalBaselinePredictor)


def test_rna_comparison_family_builds():
    model, _ = build_functional_predictor(
        variant=RNA_COMPARISON_VARIANT,
        input_dim=48,
        config=_config(),
        final_regressor_dropout=0.15,
        use_mean_proxy=True,
    )
    assert isinstance(model, RNAEncoderComparisonPredictor)


def test_unknown_variant_fails_before_model_construction():
    with pytest.raises(ValueError, match="unsupported functional model"):
        build_functional_predictor(
            variant="not-a-real-variant",
            input_dim=48,
            config=_config(),
            final_regressor_dropout=0.15,
            use_mean_proxy=True,
        )
