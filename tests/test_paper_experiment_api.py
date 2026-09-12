from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paper_experiment.py"
PROFILE = ROOT / "configs" / "data" / "paper_chr1.yaml"
SURFACE = ROOT / "configs" / "experiment_surface.yaml"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "paper_experiment_api",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paper_profile_has_exact_official_views():
    payload = yaml.safe_load(PROFILE.read_text())
    assert payload["schema_version"] == 1
    assert payload["scope"] == "chr1"
    assert payload["official_evaluation"]["headline_view"] == (
        "val_cpg_x_val_sample"
    )
    assert payload["official_evaluation"]["views"] == [
        "train_cpg_x_val_sample",
        "val_cpg_x_train_sample",
        "val_cpg_x_val_sample",
    ]


def test_final_api_keeps_storage_compatibility_behind_profile():
    payload = yaml.safe_load(PROFILE.read_text())
    assert payload["storage"]["rna_key"] == "locus_cls_joint"

    text = SCRIPT.read_text()
    assert '"locus_cls_joint"' not in text
    assert "'locus_cls_joint'" not in text


def test_final_api_only_accepts_frozen_paper_facing_recipes():
    module = _load_script()
    surface = yaml.safe_load(SURFACE.read_text())

    paper = {
        row["path"]
        for row in surface["models"]["paper_facing"]
    }
    protected = {
        row["path"]
        for row in surface["models"]["protected_research"]
    }

    api_paper = set(module.paper_facing_recipe_records())
    assert api_paper == paper
    assert api_paper.isdisjoint(protected)


def test_main_recipe_validates_against_frozen_manifest():
    module = _load_script()
    relative, record = module.validate_paper_recipe(
        ROOT / "configs" / "models" / "main.yaml"
    )
    assert relative == "configs/models/main.yaml"
    assert record["path"] == relative


def test_cpg_prior_is_detected_as_zero_parameter_baseline():
    module = _load_script()
    kind = module.classify_recipe(
        ROOT
        / "configs"
        / "models"
        / "baselines"
        / "baseline_cpg_prior.yaml"
    )
    assert kind == "cpg_prior"


def test_rna_train_command_uses_public_runtime_without_engine(tmp_path):
    module = _load_script()
    profile = {
        "scope": "chr1",
        "paths": {
            "canonical_root": tmp_path / "canonical",
            "prepared_root": tmp_path / "prepared",
            "registry": tmp_path / "registry.parquet",
            "rna_cache": tmp_path / "rna",
            "prior_cache": tmp_path / "prior",
            "cpg_targets_dir": tmp_path / "targets",
            "functional_atlas": tmp_path / "functional",
            "annotation_cache": tmp_path / "annotations",
            "output_root": tmp_path / "experiments",
        },
    }
    cmd = module.build_train_command(
        profile=profile,
        recipe_relative="configs/models/main.yaml",
        run_id="main-seed17",
        seed=17,
        resume=False,
        rna_cache_override=None,
    )
    joined = " ".join(cmd)
    assert "--model rna_methylation" in joined
    assert "--mode final" in joined
    assert "--recipe configs/models/main.yaml" in joined
    assert "--functional-atlas" in cmd
    assert "--annotation-cache" in cmd
    assert "--prior-cache" in cmd
    assert "--engine" not in cmd
    assert "--feature-cache" not in cmd


def test_cpg_prior_evaluation_has_no_checkpoint(tmp_path):
    module = _load_script()
    profile = {
        "scope": "chr1",
        "paths": {
            "canonical_root": tmp_path / "canonical",
            "registry": tmp_path / "registry.parquet",
            "prior_cache": tmp_path / "prior",
        },
    }
    run_dir = tmp_path / "run"
    cmd = module.build_evaluate_command(
        profile=profile,
        model_kind="cpg_prior",
        recipe_relative=(
            "configs/models/baselines/baseline_cpg_prior.yaml"
        ),
        run_dir=run_dir,
        rna_cache_override=None,
    )
    joined = " ".join(cmd)
    assert "--model cpg_prior" in joined
    assert "--feature-cache" in cmd
    assert "--checkpoint" not in cmd
