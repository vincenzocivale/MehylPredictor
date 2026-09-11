from __future__ import annotations

from pathlib import Path

import methylation_predictor as mp

from methylation_predictor.modeling import (
    IterativeRetrievalPredictor,
    SingleRetrievalPredictor,
)
from methylation_predictor.rna_training.config import load_rna_recipe


ROOT = Path(__file__).resolve().parents[1]


def test_public_api_exports_only_paper_facing_candidate_classes():
    assert mp.__all__ == [
        "SingleRetrievalPredictor",
        "IterativeRetrievalPredictor",
    ]
    assert mp.SingleRetrievalPredictor is SingleRetrievalPredictor
    assert mp.IterativeRetrievalPredictor is IterativeRetrievalPredictor


def test_historical_root_models_are_removed():
    assert "FeatureFusionLocusCLSModel" not in mp.__all__
    assert "FeatureFusionArchitectureVariantModel" not in mp.__all__
    assert not hasattr(mp, "FeatureFusionLocusCLSModel")
    assert not hasattr(mp, "FeatureFusionArchitectureVariantModel")


def test_main_recipe_is_exactly_the_resolved_j0_reference():
    main = load_rna_recipe(ROOT / "configs/models/main.yaml")
    j0 = load_rna_recipe(
        ROOT / "configs/models/functional_fusion/j0_final.yaml"
    )

    assert main.raw == j0.raw
    assert main.model == j0.model
    assert main.loss == j0.loss
    assert main.training == j0.training
    assert main.batching == j0.batching
    assert main.schedule_policy == j0.schedule_policy
    assert main.structured_loss_sources == j0.structured_loss_sources
    assert (
        main.exclude_official_val_from_auxiliary
        == j0.exclude_official_val_from_auxiliary
    )


def test_main_recipe_selects_single_retrieval_reference():
    recipe = load_rna_recipe(ROOT / "configs/models/main.yaml")

    assert recipe.model.functional_fusion_variant == "mas_concat_v3_purecontext"
    assert recipe.model.encoder.kind == "locus_attention"
    assert recipe.model.encoder.n_programs == 64
    assert recipe.model.encoder.program_dim == 256
    assert recipe.loss.beta_mse_weight == 1.0
    assert recipe.loss.locus_pearson_weight == 0.15
    assert recipe.raw["locus_cls"]["aux_weight"] == 0.15
    assert recipe.raw["locus_cls"]["final_regressor_dropout"] == 0.15
