from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from methylation_predictor.rna_encoder_quality.config import load_config
from methylation_predictor.rna_encoder_quality.runner import run_quality_audit, validate_inputs


def _strings(handle: h5py.File, key: str, values: list[str]) -> None:
    handle.create_dataset(key, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype("utf-8"))


def _fixture(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    cancers = np.repeat(["A", "B", "C"], 30)
    splits = np.tile(np.asarray(["train"] * 20 + ["validation"] * 5 + ["test"] * 5), 3)
    sample_ids = [f"sample_{index:03d}" for index in range(len(cancers))]
    gene_ids = [f"gene_{index:03d}" for index in range(30)]
    latent = rng.normal(size=(len(cancers), 8))
    weights = rng.normal(size=(8, len(gene_ids)))
    cancer_means = {label: rng.normal(scale=2.0, size=len(gene_ids)) for label in np.unique(cancers)}
    rna = latent @ weights + np.stack([cancer_means[label] for label in cancers]) + rng.normal(scale=0.05, size=(len(cancers), len(gene_ids)))
    rna -= rna.min(axis=0, keepdims=True)
    good = latent + rng.normal(scale=0.01, size=latent.shape)
    noise = rng.normal(size=latent.shape)

    rna_path = tmp_path / "rna.h5"
    with h5py.File(rna_path, "w") as handle:
        handle.create_dataset("X", data=rna.astype(np.float32))
        _strings(handle, "sample_idx", sample_ids)
        _strings(handle, "gene_ids", gene_ids)

    embedding_path = tmp_path / "embeddings.h5"
    with h5py.File(embedding_path, "w") as handle:
        handle.create_dataset("embeddings_layer1", data=good.astype(np.float32))
        handle.create_dataset("embeddings_layer2", data=noise.astype(np.float32))
        _strings(handle, "sample_idx", sample_ids)

    stability_path = tmp_path / "stability.h5"
    with h5py.File(stability_path, "w") as handle:
        handle.create_dataset("embeddings_layer1", data=(good + rng.normal(scale=0.001, size=good.shape)).astype(np.float32))
        _strings(handle, "sample_idx", sample_ids)

    perturbation_path = tmp_path / "perturbation.h5"
    perturbation_control_path = tmp_path / "perturbation_control.h5"
    coherent_shift = np.zeros_like(good)
    coherent_shift[:, 0] = 0.25
    with h5py.File(perturbation_path, "w") as handle:
        handle.create_dataset("embeddings_layer1", data=(good + coherent_shift).astype(np.float32))
        _strings(handle, "sample_idx", sample_ids)
    with h5py.File(perturbation_control_path, "w") as handle:
        handle.create_dataset("embeddings_layer1", data=(good + rng.normal(scale=0.05, size=good.shape)).astype(np.float32))
        _strings(handle, "sample_idx", sample_ids)

    token_path = tmp_path / "tokens.h5"
    token_genes = 6
    expression = rna[:, :token_genes].astype(np.float32)
    token_ids = np.digitize(expression, np.quantile(expression, [0.25, 0.5, 0.75])).astype(np.int16)
    token_layer1 = np.zeros((len(cancers), token_genes, 4), dtype=np.float32)
    token_layer1[:, :, 0] = expression
    token_layer1[:, :, 1:] = latent[:, None, :3]
    token_layer2 = rng.normal(size=token_layer1.shape).astype(np.float32)
    with h5py.File(token_path, "w") as handle:
        handle.create_dataset("expression_log2p1", data=expression)
        handle.create_dataset("token_ids", data=token_ids)
        handle.create_dataset("tokens_layer1", data=token_layer1)
        handle.create_dataset("tokens_layer2", data=token_layer2)
        _strings(handle, "sample_idx", sample_ids)
        _strings(handle, "gene_ids", gene_ids[:token_genes])

    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"sample_idx": sample_ids, "cancer_type": cancers, "split": splits}).to_csv(metadata_path, index=False)

    config = {
        "run_name": "synthetic",
        "output_dir": str(tmp_path / "output"),
        "embeddings": {
            "path": str(embedding_path),
            "row_ids_key": "sample_idx",
            "layers": {"layer1": "embeddings_layer1", "layer2": "embeddings_layer2"},
        },
        "rna": {"path": str(rna_path), "values_key": "X", "row_ids_key": "sample_idx", "col_ids_key": "gene_ids"},
        "metadata": {
            "path": str(metadata_path), "id_column": "sample_idx", "cancer_column": "cancer_type", "split_column": "split"
        },
        "analysis": {
            "seed": 17,
            "chunk_size": 16,
            "reconstruction_gene_count": 20,
            "pca_dimensions": [4],
            "random_projection_dim": 4,
            "ridge_alphas": [0.01, 1.0],
            "knn_k": 3,
            "pair_sample_count": 200,
            "geometry_splits": ["test", "all"],
            "token_ridge_alphas": [0.01, 1.0],
            "token_global_pc_count": 3,
        },
        "token_embeddings": {
            "path": str(token_path),
            "sample_ids_key": "sample_idx",
            "gene_ids_key": "gene_ids",
            "expression_key": "expression_log2p1",
            "token_ids_key": "token_ids",
            "layers": {"layer1": "tokens_layer1", "layer2": "tokens_layer2"},
        },
        "stability_views": [{
            "name": "small_noise", "path": str(stability_path), "row_ids_key": "sample_idx",
            "layers": {"layer1": "embeddings_layer1"},
        }],
        "perturbation_views": [
            {
                "name": "random_control", "path": str(perturbation_control_path), "row_ids_key": "sample_idx",
                "layers": {"layer1": "embeddings_layer1"},
            },
            {
                "name": "coherent_program", "path": str(perturbation_path), "row_ids_key": "sample_idx",
                "control": "random_control", "layers": {"layer1": "embeddings_layer1"},
            },
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def test_intrinsic_pipeline_prefers_informative_embedding(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    validation = validate_inputs(load_config(config_path))
    assert validation["n_samples"] == 90
    output = run_quality_audit(config_path)
    for name in (
        "manifest.json", "summary.json", "report.md", "geometry.csv", "cka.csv",
        "reconstruction.csv", "neighborhood.csv", "stability.csv", "perturbation.csv", "token_quality.csv",
    ):
        assert (output / name).is_file()
    reconstruction = pd.read_csv(output / "reconstruction.csv")
    within = reconstruction[reconstruction["target"] == "within_cancer_rna"].set_index("representation")
    assert within.loc["encoder_layer1", "global_r2"] > within.loc["encoder_layer2", "global_r2"] + 0.25
    stability = pd.read_csv(output / "stability.csv")
    assert stability.loc[0, "same_patient_cosine_mean"] > 0.99
    perturbation = pd.read_csv(output / "perturbation.csv")
    coherent = perturbation[perturbation["view"] == "coherent_program"].iloc[0]
    assert coherent["directional_consistency_gain_vs_control"] > 0.5
    token = pd.read_csv(output / "token_quality.csv")
    layer1_gain = token[(token["layer"] == "layer1") & (token["metric"] == "median_context_gain_expression_r2")]
    assert float(layer1_gain.iloc[0]["value"]) > 0


def test_cleanup_is_dry_run_then_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "src").mkdir()
    target = root / "docs/rna_representation_results.md"
    target.parent.mkdir(parents=True)
    target.write_text("obsolete")
    module_path = Path(__file__).parents[1] / "scripts/maintenance/cleanup_obsolete_rna_experiments.py"
    spec = importlib.util.spec_from_file_location("cleanup_obsolete", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dry = module.cleanup(root, apply=False, include_historical_grids=False)
    assert target.is_file() and "docs/rna_representation_results.md" in dry["existing"]
    applied = module.cleanup(root, apply=True, include_historical_grids=False)
    assert not target.exists() and "docs/rna_representation_results.md" in applied["removed"]
    repeated = module.cleanup(root, apply=True, include_historical_grids=False)
    assert repeated["removed"] == []
