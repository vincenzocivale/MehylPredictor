from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


suite = _load(
    "functional_rna_encoder_suite",
    "scripts/experiments/functional_rna_encoder_suite.py",
)
runner = _load(
    "run_rna_encoder_comparison",
    "scripts/experiments/run_rna_encoder_comparison.py",
)


def test_registry_has_five_fresh_functional_arms():
    assert [arm.name for arm in suite.ARMS] == [
        "ours_program_tokens",
        "bottleneck_mlp",
        "gene_pathway",
        "bulkformer_147m",
        "bulkrnabert",
    ]
    assert len({arm.run_id(17) for arm in suite.ARMS}) == 5
    assert all(
        arm.run_id(17).startswith("functional-rnaenc-")
        for arm in suite.ARMS
    )


def test_runner_cache_mapping_is_explicit(tmp_path):
    paths = {
        "canonical_rna": tmp_path / "canonical",
        "bulkformer_147m": tmp_path / "bulkformer",
        "bulkrnabert": tmp_path / "bulkrnabert",
    }
    by_name = {arm.name: arm for arm in suite.ARMS}

    assert runner.cache_for(
        by_name["ours_program_tokens"], paths
    ) == paths["canonical_rna"]
    assert runner.cache_for(
        by_name["bottleneck_mlp"], paths
    ) == paths["canonical_rna"]
    assert runner.cache_for(
        by_name["gene_pathway"], paths
    ) == paths["canonical_rna"]
    assert runner.cache_for(
        by_name["bulkformer_147m"], paths
    ) == paths["bulkformer_147m"]
    assert runner.cache_for(
        by_name["bulkrnabert"], paths
    ) == paths["bulkrnabert"]


def test_generated_training_command_uses_functional_inputs(tmp_path):
    paths = {
        "canonical": tmp_path / "canonical",
        "prepared": tmp_path / "prepared",
        "prior_cache": tmp_path / "prior",
        "canonical_rna": tmp_path / "rna",
        "bulkformer_147m": tmp_path / "bulkformer",
        "bulkrnabert": tmp_path / "bulkrnabert",
        "registry": tmp_path / "registry.parquet",
        "cpg_targets": tmp_path / "targets",
        "functional_atlas": tmp_path / "functional",
        "annotation_cache": tmp_path / "annotations",
        "output_root": tmp_path / "experiments",
    }
    arm = suite.ARMS[0]
    cmd = runner.train_cmd(
        arm,
        17,
        "final",
        paths,
        resume=False,
        early_stop_patience=6,
    )

    assert "--functional-only" not in cmd
    assert "--engine" not in cmd
    assert "--feature-cache" not in cmd
    assert cmd[cmd.index("--prior-cache") + 1] == str(
        paths["prior_cache"]
    )
    assert cmd[cmd.index("--functional-atlas") + 1] == str(
        paths["functional_atlas"]
    )
    assert cmd[cmd.index("--annotation-cache") + 1] == str(
        paths["annotation_cache"]
    )
    assert cmd[cmd.index("--recipe") + 1] == arm.recipe
    assert cmd[cmd.index("--run-id") + 1] == arm.run_id(17)


def test_generated_eval_command_uses_same_functional_inputs(tmp_path):
    paths = {
        "canonical": tmp_path / "canonical",
        "prepared": tmp_path / "prepared",
        "prior_cache": tmp_path / "prior",
        "canonical_rna": tmp_path / "rna",
        "bulkformer_147m": tmp_path / "bulkformer",
        "bulkrnabert": tmp_path / "bulkrnabert",
        "registry": tmp_path / "registry.parquet",
        "cpg_targets": tmp_path / "targets",
        "functional_atlas": tmp_path / "functional",
        "annotation_cache": tmp_path / "annotations",
        "output_root": tmp_path / "experiments",
    }
    arm = suite.ARMS[3]
    cmd = runner.evaluate_cmd(arm, 29, paths)

    assert "--functional-only" not in cmd
    assert "--engine" not in cmd
    assert "--feature-cache" not in cmd
    assert cmd[cmd.index("--prior-cache") + 1] == str(
        paths["prior_cache"]
    )
    assert cmd[cmd.index("--rna-cache") + 1] == str(
        paths["bulkformer_147m"]
    )
    assert cmd[cmd.index("--functional-atlas") + 1] == str(
        paths["functional_atlas"]
    )
