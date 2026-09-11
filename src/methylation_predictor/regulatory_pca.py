"""Train-fitted regulatory PCA, with a sparse exact inference path."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def binary_matrix(batch, n_tracks):
    return sparse.csr_matrix((np.ones(len(batch["track_indices"]), np.float32),
                             batch["track_indices"], batch["offsets"]),
                            shape=(len(batch["offsets"]) - 1, n_tracks))


@dataclass
class RegulatoryPCA:
    track_weights: np.ndarray
    dense_weights: np.ndarray
    bias: np.ndarray
    explained_variance_ratio: float

    @classmethod
    def fit(cls, cache, training_ids, *, width=1536, seed=17):
        ids = np.asarray(training_ids, dtype=np.int64)
        if len(ids) != len(np.unique(ids)) or len(ids) <= width:
            raise ValueError("PCA needs unique training loci and more rows than components")
        batch = cache.get(ids)
        binary = binary_matrix(batch, cache.N_TRACKS)
        features = np.column_stack([binary.toarray(), batch["dense"]])
        if not np.isfinite(features).all():
            raise ValueError("regulatory PCA input contains non-finite values")
        scaler = StandardScaler().fit(features)
        normalized = scaler.transform(features)
        pca = PCA(n_components=width, svd_solver="randomized", iterated_power=3,
                  random_state=seed).fit(normalized)
        whitening = np.sqrt(np.maximum(pca.explained_variance_, 1e-6))
        # Compose standardization, PCA centering and whitening into one
        # affine map, then split the track/dense portions for sparse inference.
        components = pca.components_.T / whitening
        weights = components / scaler.scale_[:, None]
        bias = (-scaler.mean_ / scaler.scale_ - pca.mean_) @ components
        return cls(weights[:cache.N_TRACKS].astype(np.float32),
                   weights[cache.N_TRACKS:].astype(np.float32), bias.astype(np.float32),
                   float(pca.explained_variance_ratio_.sum()))

    def transform(self, batch):
        tracks = binary_matrix(batch, self.track_weights.shape[0])
        return np.asarray(tracks @ self.track_weights + batch["dense"] @ self.dense_weights
                          + self.bias, dtype=np.float32)

    def save(self, path):
        np.savez(path, track_weights=self.track_weights, dense_weights=self.dense_weights,
                 bias=self.bias, explained_variance_ratio=self.explained_variance_ratio)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["track_weights"], data["dense_weights"], data["bias"],
                       float(data["explained_variance_ratio"]))
