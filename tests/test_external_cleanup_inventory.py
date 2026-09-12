from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inventory_external_artifacts.py"
POLICY = ROOT / "configs" / "external_cleanup_policy.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "external_cleanup_inventory",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _profile(tmp_path: Path) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "scope": "chr1",
                "paths": {
                    "canonical_root":
                        "datasets/methylprophet_repro_v1",
                    "prepared_root":
                        "derived/methylprophet_table5_tcga_chr1",
                    "registry":
                        "datasets/methylprophet_repro_v1/"
                        "cpg/registries/array_cpg_map.parquet",
                    "rna_cache":
                        "derived/methylprophet_table5_tcga_chr1/rna",
                    "prior_cache":
                        "derived/methylprophet_table5_tcga_chr1/features",
                    "cpg_targets_dir":
                        "derived/cpg_statistics/chr1",
                    "functional_atlas":
                        "derived/functional_atlas",
                    "annotation_cache":
                        "derived/annotations",
                    "output_root": "experiments",
                },
            },
            sort_keys=False,
        )
    )
    return path


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_policy_disables_destructive_actions():
    payload = yaml.safe_load(POLICY.read_text())
    assert payload["schema_version"] == 1
    assert payload["policy"]["destructive_actions_allowed"] is False


def test_scanner_has_no_destructive_filesystem_calls():
    text = SCRIPT.read_text()
    forbidden = [
        ".unlink(",
        "shutil.rmtree(",
        "os.remove(",
        "os.rmdir(",
        ".rename(",
    ]
    assert [x for x in forbidden if x in text] == []


def test_run_is_protected_before_architecture_selection(tmp_path):
    module = _load()
    data = _mkdir(tmp_path / "data")
    _mkdir(
        data
        / "experiments"
        / "runs"
        / "locus_cls_joint"
        / "chr1"
        / "j9c-seed17"
    )

    payload = module.build_inventory(
        data_root=data,
        policy_path=POLICY,
        profile_path=_profile(tmp_path),
        architecture_selected=False,
        fresh_paper_runs_verified=False,
        include_sizes=False,
    )
    rows = {r["path"]: r for r in payload["artifacts"]}
    assert (
        rows[
            "experiments/runs/locus_cls_joint/chr1/j9c-seed17"
        ]["classification"]
        == "PROTECT"
    )


def test_historical_cache_not_candidate_before_gates(tmp_path):
    module = _load()
    data = _mkdir(tmp_path / "data")
    _mkdir(data / "derived" / "ntv3_pre_old_cache")

    payload = module.build_inventory(
        data_root=data,
        policy_path=POLICY,
        profile_path=_profile(tmp_path),
        architecture_selected=False,
        fresh_paper_runs_verified=False,
        include_sizes=False,
    )
    rows = {r["path"]: r for r in payload["artifacts"]}
    assert (
        rows["derived/ntv3_pre_old_cache"]["classification"]
        == "REVIEW"
    )


def test_candidate_requires_both_gates(tmp_path):
    module = _load()
    data = _mkdir(tmp_path / "data")
    _mkdir(data / "derived" / "ntv3_pre_old_cache")

    payload = module.build_inventory(
        data_root=data,
        policy_path=POLICY,
        profile_path=_profile(tmp_path),
        architecture_selected=True,
        fresh_paper_runs_verified=True,
        include_sizes=False,
    )
    rows = {r["path"]: r for r in payload["artifacts"]}
    assert (
        rows["derived/ntv3_pre_old_cache"]["classification"]
        == "DELETE_CANDIDATE"
    )


def test_paper_record_reference_overrides_candidate_pattern(tmp_path):
    module = _load()
    data = _mkdir(tmp_path / "data")
    cache = _mkdir(data / "derived" / "ntv3_pre_final_needed")

    record = (
        data
        / "experiments"
        / "runs"
        / "locus_cls_joint"
        / "chr1"
        / "final-seed17"
        / "paper"
        / "record.json"
    )
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "methylpredictor.paper_run",
                "run_id": "final-seed17",
                "study": "main",
                "arm": "main",
                "runtime": {
                    "run_dir": str(record.parents[1]),
                },
                "data_profile": {
                    "resolved_paths": {
                        "special_cache": str(cache),
                    }
                },
                "evaluation": {
                    "path": str(
                        record.parents[1]
                        / "evaluation"
                        / "chr1"
                        / "metrics.json"
                    )
                },
            }
        )
    )

    payload = module.build_inventory(
        data_root=data,
        policy_path=POLICY,
        profile_path=_profile(tmp_path),
        architecture_selected=True,
        fresh_paper_runs_verified=True,
        include_sizes=False,
    )
    rows = {r["path"]: r for r in payload["artifacts"]}

    assert (
        rows["derived/ntv3_pre_final_needed"]["classification"]
        == "KEEP"
    )
    assert (
        rows[
            "experiments/runs/locus_cls_joint/chr1/final-seed17"
        ]["classification"]
        == "KEEP"
    )


def test_second_gate_requires_first_gate(tmp_path):
    module = _load()
    data = _mkdir(tmp_path / "data")

    with pytest.raises(ValueError):
        module.build_inventory(
            data_root=data,
            policy_path=POLICY,
            profile_path=_profile(tmp_path),
            architecture_selected=False,
            fresh_paper_runs_verified=True,
            include_sizes=False,
        )


def test_hg38_reference_is_explicitly_kept(tmp_path):
    module = _load()
    data = _mkdir(tmp_path / "data")
    _mkdir(data / "reference" / "hg38")

    payload = module.build_inventory(
        data_root=data,
        policy_path=POLICY,
        profile_path=_profile(tmp_path),
        architecture_selected=False,
        fresh_paper_runs_verified=False,
        include_sizes=False,
    )
    rows = {r["path"]: r for r in payload["artifacts"]}
    assert rows["reference/hg38"]["classification"] == "KEEP"


def test_nested_runs_layout_is_reported_as_anomaly(tmp_path):
    module = _load()
    data = _mkdir(tmp_path / "data")
    _mkdir(
        data
        / "experiments"
        / "runs"
        / "runs"
        / "locus_cls_joint"
        / "chr1"
    )

    payload = module.build_inventory(
        data_root=data,
        policy_path=POLICY,
        profile_path=_profile(tmp_path),
        architecture_selected=False,
        fresh_paper_runs_verified=False,
        include_sizes=False,
    )

    assert payload["summary"]["layout_anomaly_count"] == 1
    assert payload["summary"]["layout_anomalies"] == [
        "experiments/runs/runs/locus_cls_joint/chr1"
    ]
