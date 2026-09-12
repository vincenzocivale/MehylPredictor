from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from methylation_predictor.config import LossConfig, ModelConfig
from methylation_predictor.rna_training.config import load_rna_recipe


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "configs/config_field_audit.yaml"


def _audit() -> dict:
    return yaml.safe_load(AUDIT.read_text())


def test_audit_is_marked_pruned():
    audit = _audit()
    assert audit["schema_version"] == 2
    assert audit["status"] == "phase6d2_pruned"


def test_model_config_contains_only_retained_fields():
    audit = _audit()
    actual = {field.name for field in fields(ModelConfig)}
    assert actual == set(audit["model_config"]["keep"])


def test_loss_config_contains_only_retained_fields():
    audit = _audit()
    actual = {field.name for field in fields(LossConfig)}
    assert actual == set(audit["loss_config"]["keep"])


def test_main_recipe_contains_no_removed_fields():
    audit = _audit()
    raw = load_rna_recipe(ROOT / "configs/models/main.yaml").raw

    assert set(audit["model_config"]["removed"]).isdisjoint(raw["model"])
    assert set(audit["loss_config"]["removed"]).isdisjoint(raw["loss"])
    assert set(audit["locus_cls"]["removed"]).isdisjoint(
        raw["locus_cls"]
    )


def test_every_live_rna_recipe_parses_after_prune():
    surface = yaml.safe_load(
        (ROOT / "configs/experiment_surface.yaml").read_text()
    )
    non_rna = {
        "configs/models/cpg_statistics.yaml",
        "configs/models/baselines/baseline_cpg_prior.yaml",
    }

    for record in surface["models"]["paper_facing"]:
        relative = record["path"]
        if relative in non_rna:
            continue
        load_rna_recipe(ROOT / relative)
