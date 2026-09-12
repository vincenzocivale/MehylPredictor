from __future__ import annotations

from pathlib import Path

import methylation_predictor as mp

from methylation_predictor.modeling import (
    EfficientSingleAttentionPredictor,
)
from methylation_predictor.modeling.factory import FINAL_VARIANT
from methylation_predictor.rna_training.config import load_rna_recipe


ROOT = Path(__file__).resolve().parents[1]


def test_public_api_exports_only_final_predictor():
    assert mp.__all__ == ["EfficientSingleAttentionPredictor"]
    assert (
        mp.EfficientSingleAttentionPredictor
        is EfficientSingleAttentionPredictor
    )


def test_retired_root_models_are_not_exported():
    for name in (
        "SingleRetrievalPredictor",
        "IterativeRetrievalPredictor",
        "FeatureFusionLocusCLSModel",
        "FeatureFusionArchitectureVariantModel",
    ):
        assert name not in mp.__all__
        assert not hasattr(mp, name)


def test_main_recipe_selects_locked_final_model():
    recipe = load_rna_recipe(ROOT / "configs/models/main.yaml")
    assert recipe.model.functional_fusion_variant == FINAL_VARIANT
    assert recipe.model.encoder.kind == "locus_attention"
    assert recipe.model.encoder.n_programs == 64
    assert recipe.model.encoder.program_dim == 256
    assert recipe.loss.beta_mse_weight == 1.0
    assert recipe.loss.locus_pearson_weight == 0.15
    assert recipe.raw["locus_cls"]["aux_weight"] == 0.15
    assert recipe.raw["locus_cls"]["final_regressor_dropout"] == 0.15
