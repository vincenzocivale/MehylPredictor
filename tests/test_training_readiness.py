from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_paper_jobs.py"
PROFILE = ROOT / "configs/data/paper_chr1.yaml"
STORAGE = ROOT / "configs/data/storage_v2.yaml"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "training_readiness_runner",
        RUNNER,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_profile_uses_new_storage_namespace():
    profile = yaml.safe_load(PROFILE.read_text())
    storage = yaml.safe_load(STORAGE.read_text())
    assert profile["storage"]["rna_key"] == "rna_methylation"
    assert (
        profile["storage"]["rna_key"]
        == storage["final_storage"]["rna_key"]
    )


def test_frozen_job_matrix_has_37_unique_jobs():
    module = load_runner()
    jobs = module.load_job_matrix()
    assert len(jobs) == 37
    assert len({j.key for j in jobs}) == 37
    assert len({j.run_id for j in jobs}) == 37

    assert all(
        j.run_id.startswith(f"{j.study}__{j.arm}")
        for j in jobs
    )


def test_comparator_cache_routing(tmp_path):
    module = load_runner()
    storage = yaml.safe_load(STORAGE.read_text())
    jobs = {
        (j.study, j.arm): j
        for j in module.load_job_matrix()
    }

    bulk = jobs[("rna_encoder", "bulkformer_147m")]
    bert = jobs[("rna_encoder", "bulkrnabert")]
    ours = jobs[("rna_encoder", "ours_program_tokens")]

    assert module.resolve_rna_cache(
        bulk, storage, tmp_path
    ) == (
        tmp_path
        / "derived/bulkformer_embeddings/tcga_147m"
    ).resolve()

    assert module.resolve_rna_cache(
        bert, storage, tmp_path
    ) == (
        tmp_path
        / "derived/bulkrnabert_embeddings/tcga"
    ).resolve()

    assert module.resolve_rna_cache(
        ours, storage, tmp_path
    ) == (
        tmp_path
        / "derived/methylprophet_table5_tcga_chr1/rna"
    ).resolve()


def test_gene_pathway_dependency_is_repo_relative(tmp_path):
    module = load_runner()
    storage = yaml.safe_load(STORAGE.read_text())

    data_paths = module._data_dependency_paths(
        storage, "gene_pathway_membership"
    )
    repo_paths = module._repo_dependency_paths(
        storage, "gene_pathway_membership"
    )

    assert data_paths == []
    assert len(repo_paths) == 1
    assert repo_paths[0].is_absolute()
    assert str(repo_paths[0]).startswith(str(ROOT))


def test_build_command_passes_frozen_embedding_cache(tmp_path):
    module = load_runner()
    storage = yaml.safe_load(STORAGE.read_text())
    job = next(
        j
        for j in module.load_job_matrix()
        if j.study == "rna_encoder"
        and j.arm == "bulkformer_147m"
        and j.seed == 17
    )

    args = argparse.Namespace(
        profile="configs/data/paper_chr1.yaml",
        data_root=str(tmp_path),
        allow_dirty=False,
    )
    cmd = module.build_command(
        job,
        args,
        storage=storage,
        data_root=tmp_path,
    )

    idx = cmd.index("--rna-cache")
    assert cmd[idx + 1] == str(
        (
            tmp_path
            / "derived/bulkformer_embeddings/tcga_147m"
        ).resolve()
    )
