from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan_historical_archive.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("plan_historical_archive", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_run(root: Path, name: str, *, with_summary: bool, with_eval: bool) -> Path:
    run_dir = root / "experiments" / "runs" / "locus_cls_joint" / "chr1" / name
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"run_id": name}))
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"0" * 1024)
    if with_summary:
        (run_dir / "training").mkdir()
        (run_dir / "training" / "summary.json").write_text(json.dumps({"epochs": 80}))
    if with_eval:
        eval_dir = run_dir / "evaluation" / "chr1"
        eval_dir.mkdir(parents=True)
        (eval_dir / "metrics.json").write_text(json.dumps({"mse": 0.01}))
    return run_dir


def test_no_destructive_calls_in_planner_source():
    text = SCRIPT.read_text()
    for forbidden in ("shutil.rmtree", "os.remove", "os.unlink", "Path.unlink", ".rmdir("):
        assert forbidden not in text


def test_archiveable_run_is_classified_correctly(tmp_path):
    module = _load_script()
    _make_run(tmp_path, "complete-run", with_summary=True, with_eval=True)

    runs = module.discover_historical_runs(tmp_path, "locus_cls_joint")
    assert len(runs) == 1
    plan = module.plan_run(runs[0], tmp_path, hash_checkpoints=False)
    assert plan.classification == "ARCHIVEABLE"
    assert plan.estimated_reclaimable_bytes > 0
    assert plan.original_relative_path == "experiments/runs/locus_cls_joint/chr1/complete-run"


def test_incomplete_run_missing_summary_and_eval(tmp_path):
    module = _load_script()
    _make_run(tmp_path, "incomplete-run", with_summary=False, with_eval=False)

    runs = module.discover_historical_runs(tmp_path, "locus_cls_joint")
    plan = module.plan_run(runs[0], tmp_path, hash_checkpoints=False)
    assert plan.classification == "INCOMPLETE"
    assert plan.estimated_reclaimable_bytes == 0


def test_missing_metadata_run(tmp_path):
    module = _load_script()
    run_dir = tmp_path / "experiments" / "runs" / "locus_cls_joint" / "chr1" / "no-metadata"
    run_dir.mkdir(parents=True)

    runs = module.discover_historical_runs(tmp_path, "locus_cls_joint")
    plan = module.plan_run(runs[0], tmp_path, hash_checkpoints=False)
    assert plan.classification == "MISSING_METADATA"


def test_write_archive_is_additive_and_does_not_touch_original(tmp_path):
    module = _load_script()
    run_dir = _make_run(tmp_path, "complete-run", with_summary=True, with_eval=True)
    runs = module.discover_historical_runs(tmp_path, "locus_cls_joint")
    plan = module.plan_run(runs[0], tmp_path, hash_checkpoints=False)

    before = sorted(p.relative_to(tmp_path) for p in run_dir.rglob("*"))
    archived = module.write_archive_copy(plan, tmp_path)
    after = sorted(p.relative_to(tmp_path) for p in run_dir.rglob("*"))

    assert before == after  # original untouched
    assert archived.is_file()
    payload = json.loads(archived.read_text())
    assert payload["classification"] == "ARCHIVEABLE"
