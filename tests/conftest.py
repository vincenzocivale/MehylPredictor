from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture()
def synthetic_project(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    n_samples, n_genes, n_cpgs, locus_dim = 24, 16, 12, 8
    sample_ids = np.asarray([f"S{i:03d}" for i in range(n_samples)], dtype=object)
    cpg_ids = np.asarray([f"cg{i:05d}" for i in range(n_cpgs)], dtype=object)
    gene_ids = np.asarray([f"G{i:04d}" for i in range(n_genes)], dtype=object)
    cancer_types = np.asarray(["A", "B", "C"] * 8, dtype=object)
    sample_splits = np.asarray(["train"] * 15 + ["validation"] * 6 + ["test"] * 3, dtype=object)
    cpg_splits = np.asarray(["train"] * 7 + ["validation"] * 3 + ["test"] * 2, dtype=object)

    rna = rng.normal(size=(n_samples, n_genes)).astype(np.float32)
    embeddings = rng.normal(size=(n_cpgs, locus_dim)).astype(np.float32)
    prior_logit = rng.normal(scale=0.8, size=n_cpgs)
    prior = (1 / (1 + np.exp(-prior_logit))).astype(np.float32)
    sample_factor = rna[:, :4] @ rng.normal(size=(4, 3))
    locus_factor = embeddings[:, :4] @ rng.normal(size=(4, 3))
    delta = 0.18 * sample_factor @ locus_factor.T / np.sqrt(3)
    beta = 1 / (1 + np.exp(-(prior_logit[None, :] + delta + rng.normal(scale=0.03, size=(n_samples, n_cpgs)))))
    beta = beta.astype(np.float32)
    beta[2, 3] = np.nan

    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(tmp_path / "rna.h5", "w") as handle:
        handle.create_dataset("X", data=rna)
        handle.create_dataset("sample_idx", data=sample_ids, dtype=string_dtype)
        handle.create_dataset("gene_ids", data=gene_ids, dtype=string_dtype)
    with h5py.File(tmp_path / "beta.h5", "w") as handle:
        handle.create_dataset("beta", data=beta)
        handle.create_dataset("sample_idx", data=sample_ids, dtype=string_dtype)
        handle.create_dataset("cpg_idx", data=cpg_ids, dtype=string_dtype)
    with h5py.File(tmp_path / "embeddings.h5", "w") as handle:
        handle.create_dataset("embeddings", data=embeddings)
        handle.create_dataset("cpg_idx", data=cpg_ids, dtype=string_dtype)
        handle.create_dataset("embedding_dim", data=np.asarray([str(i) for i in range(locus_dim)], dtype=object), dtype=string_dtype)

    pd.DataFrame(
        {"sample_idx": sample_ids, "cancer_type": cancer_types, "split": sample_splits}
    ).to_csv(tmp_path / "sample_metadata.csv", index=False)
    pd.DataFrame({"cpg_idx": cpg_ids, "split": cpg_splits}).to_csv(tmp_path / "cpg_splits.csv", index=False)
    pd.DataFrame(
        {
            "cpg_idx": cpg_ids,
            "pred_ntv3_prior": prior,
            "pred_log_var_between": rng.normal(size=n_cpgs),
            "pred_log_var_within": rng.normal(size=n_cpgs),
        }
    ).to_csv(tmp_path / "locus_features.csv", index=False)

    config = {
        "run_name": "synthetic",
        "output_dir": str(tmp_path / "run"),
        "data": {
            "rna": {"path": str(tmp_path / "rna.h5"), "values_key": "X", "row_ids_key": "sample_idx", "col_ids_key": "gene_ids"},
            "methylation": {"path": str(tmp_path / "beta.h5"), "values_key": "beta", "row_ids_key": "sample_idx", "col_ids_key": "cpg_idx"},
            "locus_embeddings": {"path": str(tmp_path / "embeddings.h5"), "values_key": "embeddings", "row_ids_key": "cpg_idx", "col_ids_key": "embedding_dim"},
            "locus_features": {"path": str(tmp_path / "locus_features.csv"), "id_column": "cpg_idx"},
            "sample_metadata": {"path": str(tmp_path / "sample_metadata.csv"), "id_column": "sample_idx"},
            "cpg_splits": {"path": str(tmp_path / "cpg_splits.csv"), "id_column": "cpg_idx"},
            "rna_control": "real",
            "standardize_rna": True,
        },
        "model": {
            "encoder": {"kind": "linear", "latent_dim": 8, "hidden_dims": [16], "dropout": 0.0},
            "interaction": {"kind": "bilinear", "hidden_dim": 8, "dropout": 0.0},
            "gate": {"kind": "global", "hidden_dim": 8},
            "anchor_to_mean_rna": True,
            "zero_init_residual": True,
        },
        "loss": {"residual_huber_weight": 0.05, "shrinkage_weight": 1e-5},
        "training": {
            "seed": 13,
            "device": "cpu",
            "epochs": 3,
            "steps_per_epoch": 8,
            "sample_batch_size": 8,
            "cpg_batch_size": 6,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "amp": False,
            "patience": 3,
            "validation_max_cpgs": None,
        },
        "evaluation": {
            "sample_chunk_size": 8,
            "cpg_chunk_size": 4,
            "max_cpgs_per_panel": None,
            "save_predictions": False,
            "panels": {
                "double_ood": {"sample_split": "test", "cpg_split": "test"}
            },
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path
