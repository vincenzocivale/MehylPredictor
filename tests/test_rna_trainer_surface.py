from __future__ import annotations

import inspect
from pathlib import Path

from methylation_predictor.rna_training import (
    RNAMethylationTrainer,
    evaluate_rna_checkpoint,
)
from methylation_predictor.rna_training.locus_cls_trainer import (
    LocusCLSJointTrainer,
    evaluate_official_split,
)

ROOT = Path(__file__).resolve().parents[1]


def test_old_module_is_only_a_compatibility_shim():
    assert LocusCLSJointTrainer is RNAMethylationTrainer
    assert evaluate_official_split is evaluate_rna_checkpoint
    shim = (
        ROOT
        / "src/methylation_predictor/rna_training/locus_cls_trainer.py"
    ).read_text()
    assert "class RNAMethylationTrainer" not in shim
    assert len(shim.splitlines()) < 30


def test_live_implementation_has_truthful_module_name():
    path = (
        ROOT
        / "src/methylation_predictor/rna_training/rna_methylation_trainer.py"
    )
    assert path.is_file()
    text = path.read_text()
    assert "class RNAMethylationTrainer" in text
    assert "def evaluate_rna_checkpoint" in text


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


def test_public_rna_clis_have_no_engine_or_functional_mode_switch():
    for relative in ("scripts/train.py", "scripts/evaluate.py"):
        text = (ROOT / relative).read_text()
        assert "--engine" not in text
        assert "--functional-only" not in text


def test_run_store_identifier_stays_legacy_for_resume_compatibility():
    text = (
        ROOT
        / "src/methylation_predictor/rna_training/rna_methylation_trainer.py"
    ).read_text()
    assert 'model="locus_cls_joint"' in text
    assert '"model": "locus_cls_joint"' in text
