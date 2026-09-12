from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "configs/repository_cleanup_inventory.yaml"


def _tracked() -> set[str]:
    out = subprocess.check_output(
        ["git", "ls-files"],
        cwd=ROOT,
        text=True,
    )
    return {line for line in out.splitlines() if line}


def test_no_refactor_delivery_scripts_are_tracked():
    tracked = _tracked()
    assert not any(
        Path(path).name.startswith("phase")
        and Path(path).suffix == ".py"
        for path in tracked
    )


def test_no_diagnostic_or_eval_scratch_results_are_tracked():
    tracked = _tracked()
    assert not any(
        path.startswith("results/diagnostics/")
        or "/.eval_runs/" in path
        for path in tracked
    )


def test_stale_runtime_docs_are_gone():
    tracked = _tracked()
    assert "docs/BENCHMARKS.md" not in tracked
    assert "docs/data/GENOMIC_FEATURE_STORE.md" not in tracked


def test_removed_doc_names_are_not_linked_from_current_docs():
    needles = {
        "BENCHMARKS.md",
        "GENOMIC_FEATURE_STORE.md",
        "PAPER_EXPERIMENTS.md",
        "CHR123_TRAINING_OPTIMIZATIONS.md",
    }
    offenders = []
    for path in list((ROOT / "docs").rglob("*.md")) + [
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "scripts/README.md",
    ]:
        if not path.is_file():
            continue
        text = path.read_text()
        for needle in needles:
            if needle in text:
                offenders.append((str(path.relative_to(ROOT)), needle))
    assert offenders == []


def test_cleanup_inventory_protects_j_series_until_selection():
    payload = yaml.safe_load(INVENTORY.read_text())
    protected = payload["protect_until_architecture_selection"]

    assert protected["recipes"]
    assert protected["launchers"]

    for relative in protected["recipes"] + protected["launchers"]:
        assert (ROOT / relative).is_file(), relative


def test_final_results_are_explicitly_deferred_not_deleted_now():
    payload = yaml.safe_load(INVENTORY.read_text())
    reset = payload["reset_after_fresh_paper_runs"]

    assert "results/reference/" in reset["tracked_results"]
    assert (ROOT / "results/reference").is_dir()


def test_external_data_cleanup_requires_dependency_scan():
    payload = yaml.safe_load(INVENTORY.read_text())
    policy = payload["external_data_cleanup_after_final_runs"]
    assert "dry-run dependency scanner" in policy["safety_rule"]
