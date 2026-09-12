from __future__ import annotations

import inspect
from pathlib import Path

from methylation_predictor.rna_training import RNAMethylationTrainer
from methylation_predictor.rna_training.locus_cls_trainer import (
    LocusCLSJointTrainer,
    evaluate_official_split,
)

ROOT = Path(__file__).resolve().parents[1]


def test_new_trainer_name_is_public_and_old_name_is_compat_alias():
    assert LocusCLSJointTrainer is RNAMethylationTrainer


def test_functional_only_is_not_a_runtime_switch():
    trainer_params = inspect.signature(
        RNAMethylationTrainer.__init__
    ).parameters
    eval_params = inspect.signature(
        evaluate_official_split
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


def test_active_experiment_scripts_use_no_retired_rna_switches():
    stale = []
    for path in (ROOT / "scripts/experiments").iterdir():
        if not path.is_file() or path.suffix not in {".py", ".sh"}:
            continue
        text = path.read_text()
        if "--engine" in text or "--functional-only" in text:
            stale.append(path.name)
    assert stale == []


def test_run_store_identifier_stays_legacy_for_resume_compatibility():
    text = (
        ROOT
        / "src/methylation_predictor/rna_training/locus_cls_trainer.py"
    ).read_text()
    assert 'model="locus_cls_joint"' in text
    assert '"model": "locus_cls_joint"' in text
