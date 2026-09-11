from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import SingleRetrievalPredictor
from methylation_predictor.storage import FunctionalLocusCache


def _write_cache(tmp_path, *, mismatch=False):
    atlas = tmp_path / "atlas"
    annotations = tmp_path / "annotations"
    csr = atlas / "all_primary_overlap_csr"
    csr.mkdir(parents=True)
    annotations.mkdir()

    ids = np.array([91, 12, 45, 77], dtype=np.int64)
    np.save(atlas / "cpg_idx.npy", ids)
    np.save(annotations / "cpg_idx.npy", ids + (1 if mismatch else 0))

    core = np.arange(4 * 18, dtype=np.float32).reshape(4, 18)
    breadth = (100 + np.arange(4 * 5, dtype=np.float32)).reshape(4, 5)
    np.save(annotations / "annotation_core.f32.npy", core)
    np.save(atlas / "breadth_features.f32.npy", breadth)

    # rows: [2,7], [], [1], [3,4,8]
    np.save(
        csr / "indices.npy",
        np.array([2, 7, 1, 3, 4, 8], dtype=np.int32),
    )
    np.save(
        csr / "indptr.npy",
        np.array([0, 2, 2, 3, 6], dtype=np.int64),
    )
    np.save(csr / "data.npy", np.ones(6, dtype=np.uint8))
    (csr / "shape.json").write_text(json.dumps([4, 4165]))

    return atlas, annotations, ids, core, breadth


def test_functional_cache_arbitrary_order_and_empty_bags(tmp_path):
    atlas, annotations, ids, core, breadth = _write_cache(tmp_path)
    cache = FunctionalLocusCache(atlas, annotations)

    result = cache.get(
        np.array([ids[3], ids[1], ids[0]], dtype=np.int64)
    )

    assert result["track_indices"].tolist() == [3, 4, 8, 2, 7]
    assert result["offsets"].tolist() == [0, 3, 3, 5]
    np.testing.assert_array_equal(
        result["dense"][:, :18], core[[3, 1, 0]]
    )
    np.testing.assert_array_equal(
        result["dense"][:, 18:], breadth[[3, 1, 0]]
    )

    bag = torch.nn.EmbeddingBag(
        4165,
        4,
        mode="mean",
        include_last_offset=True,
    )
    encoded = bag(
        torch.from_numpy(result["track_indices"]),
        torch.from_numpy(result["offsets"]),
    )
    assert torch.equal(encoded[1], torch.zeros_like(encoded[1]))


def test_functional_cache_rejects_axis_mismatch(tmp_path):
    atlas, annotations, *_ = _write_cache(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="do not match exactly"):
        FunctionalLocusCache(atlas, annotations)


def test_cache_output_feeds_paper_candidate_without_dense_track_matrix(tmp_path):
    atlas, annotations, ids, *_ = _write_cache(tmp_path)
    cache = FunctionalLocusCache(atlas, annotations)

    requested = np.array([ids[3], ids[1], ids[0]], dtype=np.int64)
    functional = cache.get(requested)

    cfg = ModelConfig(
        encoder=EncoderConfig(
            kind="locus_attention",
            latent_dim=8,
            n_programs=6,
            program_dim=256,
            n_heads=4,
            dropout=0.0,
        ),
    )
    model = SingleRetrievalPredictor(
        48,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    with torch.no_grad():
        out = model(
            torch.randn(2, 48),
            functional_track_indices=torch.from_numpy(
                functional["track_indices"]
            ),
            functional_offsets=torch.from_numpy(
                functional["offsets"]
            ),
            functional_dense=torch.from_numpy(
                functional["dense"]
            ),
        )

    assert out["beta"].shape == (2, 3)
    assert torch.isfinite(out["beta"]).all()
