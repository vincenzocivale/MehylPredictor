from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/data/storage_v2.yaml"
PROFILE = ROOT / "configs/data/paper_chr1.yaml"

def _load(path):
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict)
    return data

def test_v2_contract_is_frozen_and_portable():
    cfg = _load(CONTRACT)
    assert cfg["schema_version"] == 1
    assert cfg["status"] == "frozen"
    assert cfg["roots"]["data_env"] == "METHYL_DATA_ROOT"
    assert cfg["roots"]["artifact_uri_scheme"] == "methyl-data"
    assert cfg["artifact_uri"]["absolute_paths_in_curated_results"] == "forbidden"

def test_final_storage_namespace():
    cfg = _load(CONTRACT)
    assert cfg["final_storage"]["rna_key"] == "rna_methylation"
    assert cfg["final_storage"]["complete_marker"] == "paper/record.json"

def test_current_chr1_inputs_are_covered():
    cfg = _load(CONTRACT)
    profile = _load(PROFILE)
    expected = {
        **cfg["paper_inputs"]["chr1_runtime"],
        "canonical_root": cfg["paper_inputs"]["canonical"]["canonical_root"],
    }
    for key, value in expected.items():
        if key in profile["paths"]:
            assert profile["paths"][key] == value

def test_comparator_inputs_are_keep_dependencies():
    cfg = _load(CONTRACT)
    keep = set(cfg["cleanup_target"]["keep_derived_roots"])
    assert "derived/bulkformer_embeddings" in keep
    assert "derived/bulkrnabert_embeddings" in keep

def test_frozen_study_surface():
    studies = _load(CONTRACT)["paper_studies"]
    assert set(studies) == {
        "main_model",
        "functional_baselines",
        "mean_proxy",
        "rna_encoder",
        "cpg_prior",
    }
    # Updated 2026-09-13: `main` renamed to `main_model` to match the study
    # label actually recorded at launch time, and seeds aligned to what was
    # actually launched (17, 42, 123) -- see configs/data/storage_v2.yaml's
    # note and docs/EXPERIMENT_LOG.md.
    assert studies["main_model"]["seeds"] == [17, 42, 123]
    assert set(studies["rna_encoder"]["arms"]) == {
        "ours_program_tokens",
        "bottleneck_mlp",
        "gene_pathway",
        "bulkformer_147m",
        "bulkrnabert",
    }

def test_historical_cleanup_requires_inventory():
    policy = _load(CONTRACT)["historical_policy"]
    assert policy["preserve_metadata_before_deletion"] is True
    assert policy["move_existing_runs"] is False
    assert "dependency_inventory" in policy["destructive_cleanup_requires"]
    assert "dry_run_review" in policy["destructive_cleanup_requires"]
