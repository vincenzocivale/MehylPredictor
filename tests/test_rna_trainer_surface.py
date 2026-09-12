from __future__ import annotations

import inspect
from pathlib import Path

import methylation_predictor.rna_training as rna_training

from methylation_predictor.rna_training import (
    RNAMethylationTrainer,
    evaluate_rna_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_trainer_shim_and_aliases_are_removed():
    assert not (
        ROOT
        / "src/methylation_predictor/rna_training/locus_cls_trainer.py"
    ).exists()
    assert "LocusCLSJointTrainer" not in rna_training.__all__
    assert "evaluate_official_split" not in rna_training.__all__
    assert not hasattr(rna_training, "LocusCLSJointTrainer")
    assert not hasattr(rna_training, "evaluate_official_split")


def test_live_implementation_has_truthful_module_name():
    path = (
        ROOT
        / "src/methylation_predictor/rna_training/"
        "rna_methylation_trainer.py"
    )
    text = path.read_text()
    assert "class RNAMethylationTrainer" in text
    assert "def evaluate_rna_checkpoint" in text
    assert "_retired_locus_compat" not in text
    assert "_RETIRED_LOCUS_DEFAULTS" not in text


def test_functional_mode_is_an_invariant_not_a_runtime_switch():
    trainer_params = inspect.signature(
        RNAMethylationTrainer.__init__
    ).parameters
    eval_params = inspect.signature(
        evaluate_rna_checkpoint
    ).parameters
    assert "functional_only" not in trainer_params
    assert "functional_only" not in eval_params
    assert "functional_atlas" in trainer_params
    assert "annotation_cache" in trainer_params


def test_public_rna_clis_have_no_legacy_engine_switch():
    for relative in ("scripts/train.py", "scripts/evaluate.py"):
        text = (ROOT / relative).read_text()
        assert "--engine" not in text
        assert "--functional-only" not in text


def test_storage_identifier_is_deferred_to_phase6e():
    text = (
        ROOT
        / "src/methylation_predictor/rna_training/"
        "rna_methylation_trainer.py"
    ).read_text()
    assert 'model="locus_cls_joint"' in text
