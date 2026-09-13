from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/data/paper_dependencies_v2.yaml"
CONTRACT = ROOT / "configs/data/storage_v2.yaml"
SURFACE = ROOT / "configs/experiment_surface.yaml"
SCRIPT = ROOT / "scripts/inventory_paper_dependencies.py"


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    return payload


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "paper_dependency_inventory",
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


def _touch(root: Path, relative: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _mkdir(root: Path, relative: str):
    path = root / relative
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_dependency_registry_covers_frozen_study_surface():
    registry = _load_yaml(REGISTRY)
    contract = _load_yaml(CONTRACT)

    assert set(registry["studies"]) == set(contract["paper_studies"])

    for study, frozen in contract["paper_studies"].items():
        actual = registry["studies"][study]["arms"]
        assert set(actual) == set(frozen["arms"])
        for arm in actual.values():
            assert arm["seeds"] == frozen["seeds"]


def test_every_dependency_recipe_is_paper_facing():
    registry = _load_yaml(REGISTRY)
    surface = _load_yaml(SURFACE)
    paper = {
        row["path"] for row in surface["models"]["paper_facing"]
    }

    for study in registry["studies"].values():
        for arm in study["arms"].values():
            assert arm["recipe"] in paper


def test_exact_cache_file_contracts_are_frozen():
    registry = _load_yaml(REGISTRY)
    inputs = registry["external_inputs"]

    assert inputs["canonical_rna"]["required_files"] == [
        "rna_zscore.f16.npy",
        "rna_sample_idx.npy",
    ]
    assert inputs["cpg_targets"]["required_files"] == [
        "cpg_idx.npy",
        "target_mu.npy",
    ]
    assert set(inputs["functional_atlas"]["required_files"]) == {
        "cpg_idx.npy",
        "breadth_features.f32.npy",
        "all_primary_overlap_csr/indices.npy",
        "all_primary_overlap_csr/indptr.npy",
        "all_primary_overlap_csr/shape.json",
    }


def test_cpg_prior_stricter_current_contract_is_explicit():
    registry = _load_yaml(REGISTRY)
    prior = registry["external_inputs"]["prior_cache"]

    assert prior["required_files"] == ["cpg_idx.npy", "prior.npy"]
    extra = prior["consumer_extra_files"]["cpg_prior"]["files"]
    assert set(extra) == {"embeddings.f16.npy", "sigma.npy"}


def test_scanner_is_read_only_by_construction():
    source = SCRIPT.read_text()
    forbidden = (
        ".unlink(",
        "shutil.rmtree",
        "os.remove(",
        "os.rmdir(",
        ".rename(",
        "os.replace(",
    )
    for token in forbidden:
        assert token not in source


def test_artifact_classification_uses_frozen_dependencies():
    module = _load_module()
    registry = _load_yaml(REGISTRY)
    contract = _load_yaml(CONTRACT)

    cls, _ = module.classify_artifact(
        "derived/bulkformer_embeddings",
        registry,
        contract,
    )
    assert cls == "REQUIRED_RUNTIME"

    cls, _ = module.classify_artifact(
        "derived/bulkrnabert_embeddings",
        registry,
        contract,
    )
    assert cls == "REQUIRED_RUNTIME"

    cls, _ = module.classify_artifact(
        "reference/hg38",
        registry,
        contract,
    )
    assert cls == "REQUIRED_REPRODUCIBILITY"

    # 2026-09-13: derived/ntv3_expansion (and the rest of the former
    # review_then_delete_if_unreferenced list) was investigated, confirmed
    # unreferenced, and physically deleted -- see configs/external_cleanup_
    # policy.yaml's `deleted_2026_09_13` record. The contract's cleanup list
    # is now empty (nothing left pending review-for-deletion), so an
    # unreferenced/unlisted path classifies as REVIEW, not
    # UNREFERENCED_CLEANUP_TARGET -- exercise that fallback explicitly here
    # instead of asserting against a path that no longer exists on disk.
    cls, _ = module.classify_artifact(
        "derived/ntv3_expansion",
        registry,
        contract,
    )
    assert cls == "REVIEW"


def test_missing_cache_member_is_detected(tmp_path):
    module = _load_module()
    registry = _load_yaml(REGISTRY)
    root = tmp_path / "data"
    cache = _mkdir(
        root,
        "derived/bulkformer_embeddings/tcga_147m",
    )
    _touch(
        root,
        "derived/bulkformer_embeddings/tcga_147m/"
        "rna_sample_idx.npy",
    )

    status = module.inspect_dependency(
        name="bulkformer_rna",
        spec=registry["external_inputs"]["bulkformer_rna"],
        root=root,
    )
    assert status["valid"] is False
    assert status["missing_files"] == ["rna_zscore.f16.npy"]
