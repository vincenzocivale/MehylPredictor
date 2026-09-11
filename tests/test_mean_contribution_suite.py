from __future__ import annotations

import copy
from pathlib import Path

import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import SingleRetrievalPredictor
from methylation_predictor.rna_training.config import load_rna_recipe


ROOT = Path(__file__).resolve().parents[1]


def _raw(path: str) -> dict:
    raw = copy.deepcopy(load_rna_recipe(ROOT / path).raw)
    raw.pop("tracking", None)
    return raw


def test_mean_contribution_recipes_isolate_only_intended_factors():
    full = _raw("configs/models/main.yaml")
    no_aux = _raw(
        "configs/models/mean_contribution/no_mean_supervision.yaml"
    )
    no_branch = _raw(
        "configs/models/mean_contribution/no_mean_branch.yaml"
    )

    assert full["model"]["functional_fusion_variant"] == (
        "mas_concat_v3_purecontext"
    )
    assert full["locus_cls"]["use_mean_branch"] is True
    assert full["locus_cls"]["aux_weight"] == 0.15

    expected = copy.deepcopy(full)
    expected["locus_cls"]["aux_weight"] = 0.0
    assert no_aux == expected

    expected = copy.deepcopy(full)
    expected["locus_cls"]["use_mean_branch"] = False
    expected["locus_cls"]["aux_weight"] = 0.0
    assert no_branch == expected


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


def test_no_mean_branch_preserves_every_shared_initial_parameter():
    torch.manual_seed(17)
    full = SingleRetrievalPredictor(
        48,
        _config(),
        final_regressor_dropout=0.15,
        use_mean_proxy=True,
    )

    torch.manual_seed(17)
    no_branch = SingleRetrievalPredictor(
        48,
        _config(),
        final_regressor_dropout=0.15,
        use_mean_proxy=False,
    )

    assert no_branch.mean_head is None

    full_state = {
        k: v
        for k, v in full.state_dict().items()
        if not k.startswith("mean_head.")
    }
    no_branch_state = no_branch.state_dict()

    assert full_state.keys() == no_branch_state.keys()
    for key in full_state:
        torch.testing.assert_close(
            full_state[key],
            no_branch_state[key],
            rtol=0,
            atol=0,
        )


def test_no_mean_supervision_keeps_exact_full_model_capacity():
    torch.manual_seed(17)
    full = SingleRetrievalPredictor(48, _config())
    torch.manual_seed(17)
    no_supervision = SingleRetrievalPredictor(48, _config())

    assert sum(p.numel() for p in full.parameters()) == sum(
        p.numel() for p in no_supervision.parameters()
    )
    for key, value in full.state_dict().items():
        torch.testing.assert_close(
            value,
            no_supervision.state_dict()[key],
            rtol=0,
            atol=0,
        )
