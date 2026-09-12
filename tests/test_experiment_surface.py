from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from methylation_predictor.modeling.factory import (
    SUPPORTED_FUNCTIONAL_VARIANTS,
)
from methylation_predictor.rna_training.config import load_rna_recipe


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/experiment_surface.yaml"


def _manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text())


def _paths(section: dict) -> list[str]:
    return [
        record["path"]
        for records in section.values()
        for record in records
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_model_recipe_is_classified_exactly_once():
    manifest = _manifest()
    listed = _paths(manifest["models"])

    assert len(listed) == len(set(listed))

    actual = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "configs/models").rglob("*.yaml")
    }
    assert set(listed) == actual


def test_every_experiment_script_is_classified_exactly_once():
    manifest = _manifest()
    listed = _paths(manifest["experiment_scripts"])

    assert len(listed) == len(set(listed))

    actual = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "scripts/experiments").iterdir()
        if path.is_file() and path.suffix in {".py", ".sh"}
    }
    assert set(listed) == actual


def test_frozen_surface_hashes_match_manifest():
    manifest = _manifest()

    for section_name in ("models", "experiment_scripts"):
        for records in manifest[section_name].values():
            for record in records:
                path = ROOT / record["path"]
                assert path.is_file()
                assert _sha256(path) == record["sha256"]


def test_all_live_rna_recipes_select_registered_variants():
    manifest = _manifest()

    non_rna = {
        "configs/models/cpg_statistics.yaml",
        "configs/models/baselines/baseline_cpg_prior.yaml",
        "configs/models/functional_fusion/base.yaml",
    }

    for role, records in manifest["models"].items():
        for record in records:
            path = record["path"]
            if path in non_rna:
                continue
            recipe = load_rna_recipe(ROOT / path)
            variant = recipe.model.functional_fusion_variant
            assert variant in SUPPORTED_FUNCTIONAL_VARIANTS, (
                role,
                path,
                variant,
            )


def test_main_is_the_only_primary_reference_recipe():
    manifest = _manifest()
    paper = {
        record["path"]
        for record in manifest["models"]["paper_facing"]
    }
    compat = {
        record["path"]
        for record in manifest["models"]["compatibility"]
    }

    assert "configs/models/main.yaml" in paper
    assert "configs/models/functional_fusion/j0_final.yaml" in compat


def test_research_ladder_is_protected_not_paper_facing():
    manifest = _manifest()
    protected = {
        record["path"]
        for record in manifest["models"]["protected_research"]
    }
    paper = {
        record["path"]
        for record in manifest["models"]["paper_facing"]
    }

    assert protected
    assert protected.isdisjoint(paper)
    assert all(
        path.startswith("configs/models/functional_fusion/")
        for path in protected
    )
