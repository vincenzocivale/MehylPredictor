from __future__ import annotations

from pathlib import Path

import yaml

from methylation_predictor.rna_training.config import load_rna_recipe


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "configs/config_field_audit.yaml"


def _audit() -> dict:
    return yaml.safe_load(AUDIT.read_text())


def test_phase5a_surface_exists_before_config_pruning():
    assert (ROOT / "configs/experiment_surface.yaml").is_file()


def test_live_modeling_does_not_use_compatibility_model_fields():
    audit = _audit()
    fields = audit["model_config"]["compatibility_only"]

    modeling = "\n".join(
        path.read_text()
        for path in (ROOT / "src/methylation_predictor/modeling").glob("*.py")
    )
    trainer = (
        ROOT
        / "src/methylation_predictor/rna_training/"
        "rna_methylation_trainer.py"
    ).read_text()

    for field in fields:
        assert f"config.{field}" not in modeling
        assert f"self.recipe.model.{field}" not in trainer


def test_live_trainer_does_not_consume_legacy_loss_fields():
    audit = _audit()
    legacy = audit["loss_config"]["legacy_not_consumed_by_live_trainer"]
    trainer = (
        ROOT
        / "src/methylation_predictor/rna_training/"
        "rna_methylation_trainer.py"
    ).read_text()

    for field in legacy:
        assert f"loss_cfg.{field}" not in trainer
        assert f"self.recipe.loss.{field}" not in trainer


def test_runtime_only_loss_objectives_are_inactive_in_frozen_recipes():
    surface = yaml.safe_load(
        (ROOT / "configs/experiment_surface.yaml").read_text()
    )

    non_rna = {
        "configs/models/cpg_statistics.yaml",
        "configs/models/baselines/baseline_cpg_prior.yaml",
        "configs/models/functional_fusion/base.yaml",
    }

    recipe_paths = []
    for role in ("paper_facing", "compatibility", "protected_research"):
        recipe_paths.extend(
            record["path"]
            for record in surface["models"].get(role, [])
            if record["path"].startswith("configs/models/")
            and record["path"] not in non_rna
        )

    for relative in recipe_paths:
        recipe = load_rna_recipe(ROOT / relative)

        assert recipe.loss.sample_pearson_weight == 0.0, relative
        assert recipe.loss.locus_centered_mse_weight == 0.0, relative


def test_compat_locus_keys_match_trainer_compat_defaults():
    audit = _audit()
    expected = set(audit["locus_cls"]["compatibility_only"])

    trainer = (
        ROOT
        / "src/methylation_predictor/rna_training/"
        "rna_methylation_trainer.py"
    ).read_text()

    start = trainer.index("_RETIRED_LOCUS_DEFAULTS = {")
    end = trainer.index("\n}\n", start) + 2
    block = trainer[start:end]

    for key in expected:
        assert f'"{key}"' in block


def test_keep_sets_and_remove_sets_are_disjoint():
    audit = _audit()

    for section in ("model_config", "loss_config", "locus_cls"):
        groups = audit[section]
        seen = set()
        for values in groups.values():
            values = set(values)
            assert seen.isdisjoint(values)
            seen.update(values)
