from __future__ import annotations

from pathlib import Path
import yaml

from methylation_predictor.modeling.factory import EFFICIENT_VARIANTS
from methylation_predictor.rna_training.config import load_rna_recipe

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/models/main.yaml"
SURFACE = ROOT / "configs/experiment_surface.yaml"
FINAL = "efficient_single_attn_8ffn_residual_functional8_head2"


def test_main_selects_locked_final_architecture():
    recipe = load_rna_recipe(MAIN)
    assert recipe.model.functional_fusion_variant == FINAL


def test_final_constructor_contract():
    assert EFFICIENT_VARIANTS[FINAL] == {
        "n_ffn_blocks": 8,
        "n_functional_ffn_blocks": 8,
        "deep_query": False,
        "n_head_ffn_blocks": 2,
    }


def test_final_recipe_is_standalone():
    assert "extends:" not in MAIN.read_text()


def test_architecture_selection_manifest_is_locked_and_audited():
    payload = yaml.safe_load(SURFACE.read_text())
    selection = payload["architecture_selection"]
    assert selection["status"] == "locked"
    assert selection["selected_variant"] == FINAL
    assert selection["research_surface_status"] == "removed"
    records = {row["path"]: row for row in selection["research_surface_removed"]}
    parent = selection["historical_parent_recipe"]
    assert parent in records
    assert len(records[parent]["sha256"]) == 64
    assert not (ROOT / parent).exists()


def test_main_keeps_selected_training_contract():
    recipe = load_rna_recipe(MAIN)
    assert recipe.training.epochs == 80
    assert recipe.training.learning_rate == 2.0e-4
    assert recipe.training.weight_decay == 1.0e-4
    assert recipe.training.scheduler == "cosine_warmup"
    assert recipe.training.warmup_epochs == 1.0
    assert recipe.training.min_lr_ratio == 0.1
    assert recipe.training.gradient_clip_norm == 1.0
    assert recipe.raw["locus_cls"]["use_mean_branch"] is True
    assert recipe.raw["locus_cls"]["aux_weight"] == 0.15
    assert recipe.raw["locus_cls"]["final_regressor_dropout"] == 0.15
    assert recipe.raw["batching"]["array"] == {"sample_size": 320, "cpg_size": 640}
    assert recipe.raw["batching"]["epic"] == {"sample_size": 160, "cpg_size": 5120}
    assert recipe.raw["batching"]["wgbs"] == {"sample_size": 32, "cpg_size": 20480}
