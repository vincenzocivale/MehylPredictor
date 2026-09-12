from __future__ import annotations

from pathlib import Path

import pytest

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import (
    DepthResidualAblationPredictor,
    EfficientSingleAttentionPredictor,
    FunctionalGeneFFNFusionPredictor,
    GatedResidualPredictor,
    IterativeRetrievalPredictor,
    SingleRetrievalPredictor,
)
from methylation_predictor.modeling.factory import (
    DEPTH_RESIDUAL_VARIANTS,
    EFFICIENT_VARIANTS,
    FFN_FUSION_VARIANTS,
    GATED_VARIANTS,
    SUPPORTED_FUNCTIONAL_VARIANTS,
    build_functional_predictor,
)
from methylation_predictor.rna_training.config import load_rna_recipe


ROOT = Path(__file__).resolve().parents[1]


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


def test_every_functional_fusion_recipe_selector_is_registered():
    for path in sorted(
        (ROOT / "configs/models/functional_fusion").glob("*.yaml")
    ):
        if path.name == "base.yaml":
            continue
        recipe = load_rna_recipe(path)
        assert (
            recipe.model.functional_fusion_variant
            in SUPPORTED_FUNCTIONAL_VARIANTS
        ), path.name


@pytest.mark.parametrize(
    ("variant", "expected_cls"),
    [
        ("mas_concat_v3_purecontext", SingleRetrievalPredictor),
        ("mas_concat_v4_iterative", IterativeRetrievalPredictor),
        ("ablation_depth8_residual", DepthResidualAblationPredictor),
        (
            "efficient_single_attn_8ffn_residual_functional8",
            EfficientSingleAttentionPredictor,
        ),
        ("ablation_depth1_gated_residual", GatedResidualPredictor),
        (
            "ffn_fusion_two_stream_residual_4_4",
            FunctionalGeneFFNFusionPredictor,
        ),
    ],
)
def test_representative_experiment_variants_build(variant, expected_cls):
    model, label = build_functional_predictor(
        variant=variant,
        input_dim=48,
        config=_config(),
        final_regressor_dropout=0.15,
        use_mean_proxy=True,
    )
    assert isinstance(model, expected_cls)
    assert label == f"functional_concat_{variant}"


def test_registry_specs_preserve_current_experiment_settings():
    assert DEPTH_RESIDUAL_VARIANTS["ablation_depth12_residual"] == {
        "n_blocks": 12,
        "attn_residual": True,
    }
    assert EFFICIENT_VARIANTS[
        "efficient_single_attn_4ffn_residual_functional4"
    ] == {
        "n_ffn_blocks": 4,
        "n_functional_ffn_blocks": 4,
        "deep_query": False,
    }
    assert GATED_VARIANTS["ablation_depth1_gated_residual"] == {
        "n_blocks": 1,
    }
    assert FFN_FUSION_VARIANTS[
        "ffn_fusion_two_stream_residual_8_8"
    ] == {
        "fusion_mode": "two_stream_residual",
        "n_functional_ffn_blocks": 8,
        "n_gene_expr_ffn_blocks": 8,
        "fusion_dropout": 0.1,
    }


def test_unknown_variant_fails_before_model_construction():
    with pytest.raises(ValueError, match="unsupported functional model"):
        build_functional_predictor(
            variant="not-a-real-variant",
            input_dim=48,
            config=_config(),
            final_regressor_dropout=0.15,
            use_mean_proxy=True,
        )
