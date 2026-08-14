import numpy as np

import torch

from methylation_predictor.full_suite.feature_store import SortedIndex
from methylation_predictor.full_suite.probe import (
    ProbeEnsemble,
    _stable_validation_mask,
    infer_expanded_features,
)


def test_sorted_index_preserves_query_order():
    idx = SortedIndex(np.array([50, 10, 30], dtype=np.int64))
    assert idx.positions_of(np.array([10, 50, 30])).tolist() == [1, 0, 2]
    assert idx.contains(np.array([10, 11, 30])).tolist() == [True, False, True]


def test_probe_split_is_deterministic_and_nontrivial():
    ids = np.arange(10000, dtype=np.int64)
    a = _stable_validation_mask(ids, 0.1, 17)
    b = _stable_validation_mask(ids, 0.1, 17)
    assert np.array_equal(a, b)
    assert 500 < int(a.sum()) < 1500


def test_probe_ensemble_shape():
    model = ProbeEnsemble([17, 29, 43], dim=8, dropout=0.0)
    x = torch.randn(5, 8)
    assert model.forward_all(x).shape == (3, 5, 3)
    assert model(x).shape == (5, 3)


def test_feature_extension_inference_writes_summary_and_done(tmp_path):
    embedding_dir = tmp_path / "embeddings"
    output_dir = tmp_path / "features"
    embedding_dir.mkdir()

    np.save(embedding_dir / "expanded_cpg_idx.npy", np.array([101, 102, 103], dtype=np.int64))
    np.save(
        embedding_dir / "expanded_embeddings.f32.npy",
        np.arange(12, dtype=np.float32).reshape(3, 4) / 10.0,
    )

    probe = ProbeEnsemble([17], dim=4, dropout=0.0)
    checkpoint = tmp_path / "probe.pt"
    torch.save(
        {
            "model_state": probe.state_dict(),
            "seeds": [17],
            "dim": 4,
            "dropout": 0.0,
            "x_mean": np.zeros(4, dtype=np.float32),
            "x_std": np.ones(4, dtype=np.float32),
            "y_mean": np.zeros(3, dtype=np.float32),
            "y_std": np.ones(3, dtype=np.float32),
        },
        checkpoint,
    )

    result = infer_expanded_features(
        embedding_dir,
        checkpoint,
        output_dir,
        device="cpu",
        batch_rows=2,
    )

    assert result["status"] == "built"
    assert result["rows"] == 3
    assert (output_dir / "expanded_prior.npy").is_file()
    assert (output_dir / "expanded_variability.npy").is_file()
    assert (output_dir / "feature_extension_summary.json").is_file()
    assert (output_dir / "features.done").is_file()
