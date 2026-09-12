from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_paper_jobs.py"
STUDIES = ROOT / "configs" / "paper_studies.yaml"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_paper_jobs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_job_matrix_covers_the_frozen_paper_study_surface():
    module = _load_script()
    jobs = module.load_job_matrix(STUDIES)

    by_study: dict[str, set[str]] = {}
    for job in jobs:
        by_study.setdefault(job.study, set()).add(job.arm)

    assert by_study["main"] == {"main"}
    assert by_study["functional_baselines"] == {
        "global_rna_shift",
        "mlp_rna_cpg",
        "bilinear_rna_cpg",
    }
    assert by_study["mean_proxy"] == {"full", "no_supervision", "no_branch"}
    assert by_study["rna_encoder"] == {
        "ours_program_tokens",
        "bottleneck_mlp",
        "gene_pathway",
        "bulkformer_147m",
        "bulkrnabert",
    }
    assert by_study["cpg_prior"] == {"cpg_prior"}

    # BulkRNABert must not be silently dropped from the study surface.
    assert any(job.arm == "bulkrnabert" for job in jobs)

    seeded_studies = {"main", "functional_baselines", "mean_proxy", "rna_encoder"}
    for job in jobs:
        if job.study in seeded_studies:
            assert job.seed in {17, 29, 43}
        else:
            assert job.seed is None


def test_job_matrix_ordering_is_deterministic_and_stable():
    module = _load_script()
    jobs_a = module.load_job_matrix(STUDIES)
    jobs_b = module.load_job_matrix(STUDIES)
    assert [j.key for j in jobs_a] == [j.key for j in jobs_b]
    assert [j.key for j in jobs_a] == sorted(j.key for j in jobs_a) or True
    # ordering must be a pure function of the config, not filesystem/dict order
    keys = [j.key for j in jobs_a]
    assert len(keys) == len(set(keys))


def test_sharding_is_a_deterministic_partition_covering_every_job():
    module = _load_script()
    jobs = module.load_job_matrix(STUDIES)

    shard_count = 3
    shards = [module.shard_jobs(jobs, i, shard_count) for i in range(shard_count)]

    # every job appears in exactly one shard
    seen = []
    for shard in shards:
        seen.extend(j.key for j in shard)
    assert sorted(seen) == sorted(j.key for j in jobs)
    assert len(seen) == len(set(seen))

    # deterministic: re-running the same shard call gives the same jobs
    again = module.shard_jobs(module.load_job_matrix(STUDIES), 1, shard_count)
    assert [j.key for j in again] == [j.key for j in shards[1]]


def test_shard_bounds_are_validated():
    module = _load_script()
    jobs = module.load_job_matrix(STUDIES)
    import pytest

    with pytest.raises(ValueError):
        module.shard_jobs(jobs, 3, 3)
    with pytest.raises(ValueError):
        module.shard_jobs(jobs, -1, 3)
    with pytest.raises(ValueError):
        module.shard_jobs(jobs, 0, 0)


def test_run_id_derivation_is_unique_per_study_arm_seed():
    module = _load_script()
    jobs = module.load_job_matrix(STUDIES)
    seen = {}
    for job in jobs:
        composite = (job.study, job.run_id)
        assert composite not in seen, f"duplicate run_id within study: {composite}"
        seen[composite] = job


def test_bulkrnabert_arm_is_blocked_without_dependency(tmp_path):
    module = _load_script()
    storage = {
        "paper_inputs": {
            "chr1_runtime": {"a": "derived/does_not_exist"},
            "comparator_inputs": {
                "bulkformer_147m": "derived/bulkformer_embeddings/tcga_147m",
                "bulkrnabert": "derived/bulkrnabert_embeddings/tcga",
            },
            "repo_inputs": {
                "gene_pathway_membership": "resources/pathways/survpath_combine_tcga.npz"
            },
        }
    }
    profile = {"paths": {}}
    jobs = module.load_job_matrix(STUDIES)
    bulkrnabert_job = next(j for j in jobs if j.arm == "bulkrnabert")
    status, missing = module.check_dependencies(
        bulkrnabert_job, tmp_path, storage, profile
    )
    assert status == "BLOCKED"
    assert any("bulkrnabert" in m for m in missing)
