from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.trainer import ExperimentRunner


def _h5(path: Path, key: str, values: np.ndarray, row_key: str, rows: np.ndarray, col_key=None, cols=None):
    with h5py.File(path, "w") as handle:
        handle[key] = values
        handle[row_key] = rows.astype("S")
        if col_key is not None:
            handle[col_key] = cols.astype("S")


def test_matrix_training_and_streaming_export(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    samples = np.asarray([f"s{i}" for i in range(12)])
    genes = np.asarray([f"g{i}" for i in range(10)])
    cpgs = np.asarray([f"c{i}" for i in range(15)])
    _h5(tmp_path / "rna.h5", "X", rng.normal(size=(12, 10)).astype("f4"), "sample_idx", samples, "gene_ids", genes)
    _h5(tmp_path / "beta.h5", "beta", rng.uniform(0.05, 0.95, size=(12, 15)).astype("f4"), "sample_idx", samples, "cpg_idx", cpgs)
    _h5(tmp_path / "locus.h5", "embeddings", rng.normal(size=(15, 8)).astype("f4"), "cpg_idx", cpgs)
    pd.DataFrame({"sample_idx": samples, "cancer_type": ["A", "B"] * 6,
                  "split": ["train"] * 8 + ["validation"] * 2 + ["test"] * 2}).to_csv(tmp_path / "samples.csv", index=False)
    pd.DataFrame({"cpg_idx": cpgs, "split": ["train"] * 9 + ["validation"] * 3 + ["test"] * 3}).to_csv(tmp_path / "splits.csv", index=False)
    pd.DataFrame({"cpg_idx": cpgs, "pred_ntv3_prior": rng.uniform(0.1, 0.9, 15),
                  "pred_log_var_between": np.linspace(-2, 2, 15),
                  "pred_log_var_within": np.linspace(2, -2, 15)}).to_csv(tmp_path / "features.csv", index=False)

    matrix = lambda name, values, rows, cols=None: {
        "path": str(tmp_path / name), "values_key": values, "row_ids_key": rows,
        **({"col_ids_key": cols} if cols else {}),
    }
    raw = {
        "run_name": "smoke", "output_dir": str(tmp_path / "out"),
        "data": {
            "rna": matrix("rna.h5", "X", "sample_idx", "gene_ids"),
            "methylation": matrix("beta.h5", "beta", "sample_idx", "cpg_idx"),
            "locus_embeddings": matrix("locus.h5", "embeddings", "cpg_idx"),
            "locus_features": {"path": str(tmp_path / "features.csv"), "id_column": "cpg_idx"},
            "sample_metadata": {"path": str(tmp_path / "samples.csv"), "id_column": "sample_idx"},
            "cpg_splits": {"path": str(tmp_path / "splits.csv"), "id_column": "cpg_idx"},
        },
        "model": {"encoder": {"kind": "linear", "latent_dim": 5},
                  "interaction": {"kind": "bilinear", "hidden_dim": 7, "dropout": 0.0},
                  "gate": {"kind": "variability", "hidden_dim": 4},
                  "anchor_to_mean_rna": True, "zero_init_residual": True},
        "training": {"device": "cpu", "epochs": 1, "min_epochs": 1, "steps_per_epoch": 1,
                     "sample_batch_size": 4, "cpg_batch_size": 6, "cpg_sampling": "balanced_tertiles",
                     "amp": False, "validation_max_cpgs": 3, "patience": 2},
        "evaluation": {"sample_chunk_size": 2, "cpg_chunk_size": 2, "max_cpgs_per_panel": 3,
                       "save_predictions": False,
                       "panels": {"double_ood": {"sample_split": "test", "cpg_split": "test"}}},
        "tracking": {"backend": "none"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    runner = ExperimentRunner(load_config(config_path))
    try:
        metrics = runner.train()
        uncached = runner.predict_panel(
            "test", "test", keep_predictions=True, use_factorized_cache=False
        )
        cached = runner.predict_panel(
            "test", "test", keep_predictions=True, use_factorized_cache=True
        )
        export = runner.export_panel_h5("test", "test", tmp_path / "pred.h5", include_target=True)
    finally:
        runner.close()

    assert metrics["best_epoch"] == 1
    np.testing.assert_allclose(uncached.prediction, cached.prediction, rtol=1e-6, atol=1e-6)
    assert export["factorized_inference"] is True
    with h5py.File(tmp_path / "pred.h5") as handle:
        assert handle["prediction"].shape == (2, 3)
        assert handle["target"].shape == (2, 3)
