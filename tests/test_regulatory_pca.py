from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from methylation_predictor.regulatory_pca import RegulatoryPCA
from methylation_predictor.storage import LocusFeatureCache


class SmallCache:
    N_TRACKS = 8

    def __init__(self):
        rng = np.random.default_rng(9)
        self.tracks = (rng.random((30, 8)) > 0.5).astype(np.float32)
        self.dense = rng.normal(size=(30, 3)).astype(np.float32)

    def get(self, ids):
        csr = sparse.csr_matrix(self.tracks[ids])
        return {"track_indices": csr.indices, "offsets": csr.indptr, "dense": self.dense[ids]}


def test_sparse_pca_matches_fitted_dense_transform_and_checkpoint(tmp_path):
    cache = SmallCache()
    train = np.arange(20)
    model = RegulatoryPCA.fit(cache, train, width=4)
    dense = np.column_stack([cache.tracks, cache.dense])
    scaler = StandardScaler().fit(dense[train])
    pca = PCA(n_components=4, svd_solver="randomized", iterated_power=3,
              random_state=17).fit(scaler.transform(dense[train]))
    query = np.array([27, 3, 22, 3])
    expected = pca.transform(scaler.transform(dense[query])) / np.sqrt(pca.explained_variance_)
    np.testing.assert_allclose(model.transform(cache.get(query)), expected, rtol=1e-4, atol=1e-5)
    model.save(tmp_path / "projection.npz")
    restored = RegulatoryPCA.load(tmp_path / "projection.npz")
    np.testing.assert_array_equal(restored.transform(cache.get(query)), model.transform(cache.get(query)))


def test_pca_fit_excludes_heldout_features():
    cache = SmallCache()
    train = np.arange(20)
    first = RegulatoryPCA.fit(cache, train, width=4)
    cache.tracks[20:] = 1 - cache.tracks[20:]
    cache.dense[20:] *= 100
    second = RegulatoryPCA.fit(cache, train, width=4)
    np.testing.assert_array_equal(first.track_weights, second.track_weights)
    np.testing.assert_array_equal(first.dense_weights, second.dense_weights)


def test_regulatory_cache_rejects_incomplete_export_and_wrong_training_split(tmp_path):
    manifest = {"feature_type": "regulatory_pca", "status": "building"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        LocusFeatureCache(tmp_path)
    ids = np.array([11, 12], np.int64)
    manifest.update(status="complete", train_cpg_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    np.save(tmp_path / "cpg_idx.npy", ids)
    np.save(tmp_path / "embeddings.f16.npy", np.zeros((2, 1536), np.float16))
    np.save(tmp_path / "prior.npy", np.full(2, 0.5, np.float32))
    np.save(tmp_path / "sigma.npy", np.ones(2, np.float32))
    cache = LocusFeatureCache(tmp_path)
    cache.validate_training_split(ids[::-1])
    with pytest.raises(ValueError, match="different training"):
        cache.validate_training_split(np.array([11, 12, 13]))
    assert cache.regulatory_provenance["path"] == str(tmp_path.resolve())
