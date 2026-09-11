from __future__ import annotations

from pathlib import Path

import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import (
    IterativeRetrievalPredictor,
    SingleRetrievalPredictor,
)
from methylation_predictor.models import (
    FunctionalConcatIterativeRNAModel,
    FunctionalConcatMASModel,
)
from methylation_predictor.rna_training.config import load_rna_recipe


ROOT = Path(__file__).resolve().parents[1]


def _config() -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(
            kind="locus_attention",
            latent_dim=8,
            n_programs=6,
            program_dim=256,
            n_heads=4,
            dropout=0.0,
        ),
    )


def _functional_inputs() -> dict[str, torch.Tensor]:
    # Four loci, including one empty sparse bag (offsets[1] == offsets[2]).
    return {
        "functional_track_indices": torch.tensor([1, 2, 5, 9], dtype=torch.int64),
        "functional_offsets": torch.tensor([0, 2, 2, 3, 4], dtype=torch.int64),
        "functional_dense": torch.randn(4, 23),
    }


def _has_grad(module: torch.nn.Module) -> bool:
    return any(
        p.grad is not None and bool(torch.any(p.grad != 0))
        for p in module.parameters()
    )


def test_phase2b_single_retrieval_is_exactly_equivalent_to_legacy_j0():
    cfg = _config()

    torch.manual_seed(123)
    legacy = FunctionalConcatMASModel(
        48,
        cfg,
        separate_concat_norm=True,
        final_regressor_dropout=0.15,
    ).eval()

    torch.manual_seed(123)
    refactored = SingleRetrievalPredictor(
        48,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    legacy_state = legacy.state_dict()
    new_state = refactored.state_dict()
    assert legacy_state.keys() == new_state.keys()
    for key in legacy_state:
        torch.testing.assert_close(legacy_state[key], new_state[key], rtol=0, atol=0)

    torch.manual_seed(999)
    inputs = _functional_inputs()
    rna = torch.randn(3, 48)
    with torch.no_grad():
        old_out = legacy(rna, None, **inputs)
        new_out = refactored(rna, None, **inputs)

    for key in ("beta", "mu_hat", "prediction_logit", "h_cpg"):
        torch.testing.assert_close(old_out[key], new_out[key], rtol=0, atol=0)


def test_phase2b_iterative_is_exactly_equivalent_to_legacy_j1():
    cfg = _config()

    torch.manual_seed(321)
    legacy = FunctionalConcatIterativeRNAModel(
        48,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    torch.manual_seed(321)
    refactored = IterativeRetrievalPredictor(
        48,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    legacy_state = legacy.state_dict()
    new_state = refactored.state_dict()
    assert legacy_state.keys() == new_state.keys()
    for key in legacy_state:
        torch.testing.assert_close(legacy_state[key], new_state[key], rtol=0, atol=0)

    torch.manual_seed(1001)
    inputs = _functional_inputs()
    rna = torch.randn(3, 48)
    with torch.no_grad():
        old_out = legacy(rna, None, **inputs)
        new_out = refactored(rna, None, **inputs)

    for key in ("beta", "mu_hat", "prediction_logit", "h_cpg"):
        torch.testing.assert_close(old_out[key], new_out[key], rtol=0, atol=0)


def test_j0_recipe_contract_is_frozen_for_the_refactor():
    recipe = load_rna_recipe(ROOT / "configs/models/functional_fusion/j0_final.yaml")

    assert recipe.model.functional_fusion_variant == "mas_concat_v3_purecontext"
    assert recipe.model.encoder.kind == "locus_attention"
    assert recipe.model.encoder.n_programs == 64
    assert recipe.model.encoder.program_dim == 256
    assert recipe.model.encoder.n_heads == 4

    assert recipe.loss.beta_mse_weight == 1.0
    assert recipe.loss.locus_pearson_weight == 0.15
    assert recipe.loss.sample_pearson_weight == 0.0
    assert recipe.loss.locus_centered_mse_weight == 0.0

    assert recipe.training.seed == 17
    assert recipe.training.epochs == 80
    assert recipe.training.learning_rate == 2.0e-4
    assert recipe.training.weight_decay == 1.0e-4
    assert recipe.training.scheduler_horizon_epochs == 80

    assert recipe.batching == {
        "array": {"sample_size": 640, "cpg_size": 640},
        "epic": {"sample_size": 160, "cpg_size": 5120},
        "wgbs": {"sample_size": 32, "cpg_size": 20480},
    }

    locus_cfg = recipe.raw["locus_cls"]
    assert locus_cfg["use_mean_branch"] is True
    assert locus_cfg["aux_weight"] == 0.15
    assert locus_cfg["residual_aux_weight"] == 0.0
    assert locus_cfg["final_regressor_dropout"] == 0.15


def test_j1_recipe_differs_from_j0_only_in_candidate_selector_and_metadata():
    j0 = load_rna_recipe(ROOT / "configs/models/functional_fusion/j0_final.yaml")
    j1 = load_rna_recipe(ROOT / "configs/models/functional_fusion/j1_iterative.yaml")

    assert j1.model.functional_fusion_variant == "mas_concat_v4_iterative"

    # Scientific/training controls must remain matched.  Tracking tags/group
    # and the model selector are intentionally excluded from this equality.
    assert j0.model.encoder == j1.model.encoder
    assert j0.loss == j1.loss
    assert j0.training == j1.training
    assert j0.batching == j1.batching
    assert j0.schedule_policy == j1.schedule_policy
    assert j0.structured_loss_sources == j1.structured_loss_sources
    assert (
        j0.exclude_official_val_from_auxiliary
        == j1.exclude_official_val_from_auxiliary
    )

    for key in (
        "use_mean_branch",
        "aux_weight",
        "residual_aux_weight",
        "final_regressor_dropout",
    ):
        assert j0.raw["locus_cls"][key] == j1.raw["locus_cls"][key]


def test_single_retrieval_candidate_semantic_contract():
    torch.manual_seed(17)
    model = SingleRetrievalPredictor(
        48,
        _config(),
        final_regressor_dropout=0.15,
    ).eval()

    inputs = _functional_inputs()
    rna = torch.randn(3, 48)

    with torch.no_grad():
        out_a = model(rna, torch.randn(4, 1536), **inputs)
        out_b = model(rna, torch.randn(4, 7), **inputs)

    assert out_a["beta"].shape == (3, 4)
    assert out_a["mu_hat"].shape == (4,)
    assert torch.equal(out_a["beta"], out_b["beta"])
    assert torch.equal(out_a["mu_hat"], out_b["mu_hat"])

    # The current paper candidate is explicitly functional-only: genomic/FM
    # embeddings supplied through the legacy call signature are ignored.
    assert model.requires_cpg_positions is False


def test_single_retrieval_mean_proxy_gradient_isolation_contract():
    torch.manual_seed(17)
    model = SingleRetrievalPredictor(
        48,
        _config(),
        final_regressor_dropout=0.15,
    ).train()

    out = model(torch.randn(3, 48), None, **_functional_inputs())
    model.zero_grad(set_to_none=True)
    out["mu_hat"].sum().backward()

    assert _has_grad(model.track_embedding)
    assert _has_grad(model.dense_encoder)
    assert _has_grad(model.locus_norm)
    assert _has_grad(model.mean_head)

    assert not _has_grad(model.rna_encoder)
    assert not _has_grad(model.cross_attention)
    assert not _has_grad(model.final_regressor)


def test_single_retrieval_beta_path_does_not_touch_mean_head():
    torch.manual_seed(17)
    model = SingleRetrievalPredictor(
        48,
        _config(),
        final_regressor_dropout=0.15,
    ).train()

    out = model(torch.randn(3, 48), None, **_functional_inputs())
    model.zero_grad(set_to_none=True)
    out["beta"].sum().backward()

    assert _has_grad(model.track_embedding)
    assert _has_grad(model.dense_encoder)
    assert _has_grad(model.rna_encoder)
    assert _has_grad(model.cross_attention)
    assert _has_grad(model.final_regressor)
    assert not _has_grad(model.mean_head)


def test_iterative_candidate_depth_and_chunking_contract():
    torch.manual_seed(17)
    model = IterativeRetrievalPredictor(
        48,
        _config(),
        final_regressor_dropout=0.15,
    ).eval()

    assert model.N_BLOCKS == 4
    assert len(model.retrieval_attn) == 4
    assert len(model.retrieval_ffn) == 4

    inputs = _functional_inputs()
    rna = torch.randn(3, 48)

    model.LOCUS_CHUNK = 2
    with torch.no_grad():
        chunked = model(rna, **inputs)["beta"]

    model.LOCUS_CHUNK = 10_000
    with torch.no_grad():
        unchunked = model(rna, **inputs)["beta"]

    torch.testing.assert_close(chunked, unchunked)


def test_iterative_mean_proxy_gradient_isolation_contract():
    torch.manual_seed(17)
    model = IterativeRetrievalPredictor(
        48,
        _config(),
        final_regressor_dropout=0.15,
    ).train()

    out = model(torch.randn(3, 48), **_functional_inputs())
    model.zero_grad(set_to_none=True)
    out["mu_hat"].sum().backward()

    assert _has_grad(model.track_embedding)
    assert _has_grad(model.dense_encoder)
    assert _has_grad(model.locus_norm)
    assert _has_grad(model.mean_head)

    assert not _has_grad(model.rna_encoder)
    assert not _has_grad(model.retrieval_attn)
    assert not _has_grad(model.retrieval_ffn)
    assert not _has_grad(model.final_regressor)
