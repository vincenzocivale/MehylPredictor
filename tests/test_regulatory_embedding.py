from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from methylation_predictor.regulatory_embedding import StandardizedTrackEmbedding, load_projection_initializer
from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.models import FunctionalFusionModel


def test_sparse_projection_equals_dense_standardization_including_backward():
    encoder = StandardizedTrackEmbedding(5, 8)
    encoder.set_frequency(torch.tensor([0.2, 0.5, 0.7, 0., 1.]))
    indices = torch.tensor([0, 2, 1, 2, 3, 4])
    offsets = torch.tensor([0, 2, 2, 6])  # middle row is an observed empty bag
    binary = torch.tensor([[1., 0., 1., 0., 0.], [0., 0., 0., 0., 0.], [0., 1., 1., 1., 1.]])
    sparse_result = encoder(indices, offsets)
    expected = ((binary - encoder.frequency) * encoder.inverse_scale) @ encoder.weight
    torch.testing.assert_close(sparse_result, expected)
    sparse_grad, = torch.autograd.grad(sparse_result.square().sum(), encoder.weight, retain_graph=True)
    dense_grad, = torch.autograd.grad(expected.square().sum(), encoder.weight)
    torch.testing.assert_close(sparse_grad, dense_grad)


def test_fit_uses_only_requested_loci_and_checkpoint_restores_statistics():
    class Cache:
        def get(self, ids):
            assert ids.tolist() == [11, 12]  # held-out locus 13 must not be queried
            return {"track_indices": np.array([0, 1, 1])}
    encoder = StandardizedTrackEmbedding(3, 4)
    encoder.fit(Cache(), np.array([11, 12]))
    torch.testing.assert_close(encoder.frequency, torch.tensor([0.5, 1., 0.]))
    restored = StandardizedTrackEmbedding(3, 4)
    restored.load_state_dict(encoder.state_dict())
    indices, offsets = torch.tensor([0, 2]), torch.tensor([0, 1, 2])
    torch.testing.assert_close(restored(indices, offsets), encoder(indices, offsets))


def test_unfitted_projection_fails_and_constant_tracks_are_ignored():
    encoder = StandardizedTrackEmbedding(3, 4)
    with pytest.raises(RuntimeError, match="training CpGs"):
        encoder(torch.tensor([0]), torch.tensor([0, 1]))
    encoder.set_frequency(torch.tensor([0., 1., 0.5]))
    a = encoder(torch.tensor([0, 1, 2]), torch.tensor([0, 3]))
    b = encoder(torch.tensor([2]), torch.tensor([0, 1]))
    torch.testing.assert_close(a, b)


def test_covariance_initializer_rejects_validation_loci_or_different_atlas(tmp_path):
    encoder = StandardizedTrackEmbedding(3, 4)
    encoder.set_frequency(torch.tensor([0.2, 0.5, 0.7]))
    ids = np.array([11, 12], np.int64)
    path = tmp_path / "projection.pt"
    torch.save({"train_cpg_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                "source_atlas": str(tmp_path.resolve()), "state_dict": encoder.state_dict()}, path)
    # A permutation is the same split; adding any validation locus is not.
    load_projection_initializer(encoder, path, ids[::-1], tmp_path)
    with pytest.raises(ValueError, match="training CpG split"):
        load_projection_initializer(encoder, path, np.array([11, 12, 13]), tmp_path)
    with pytest.raises(ValueError, match="different functional atlas"):
        load_projection_initializer(encoder, path, ids, tmp_path / "different")
    encoder.set_frequency(torch.tensor([0.3, 0.5, 0.7]))
    with pytest.raises(ValueError, match="frequencies disagree"):
        load_projection_initializer(encoder, path, ids, tmp_path)


def test_f7_model_is_finite_and_independent_of_ntv3():
    config = ModelConfig(encoder=EncoderConfig(kind="locus_attention", latent_dim=16,
                         n_programs=4, program_dim=256, n_heads=4, dropout=0.),
                         functional_fusion_variant="f7_standardized")
    model = FunctionalFusionModel(12, config).eval()
    model.track_embedding.set_frequency(torch.full((4165,), 0.1))
    inputs = dict(functional_track_indices=torch.tensor([0, 2, 1]),
                  functional_offsets=torch.tensor([0, 2, 2, 3]),
                  functional_dense=torch.randn(3, 23))
    rna = torch.randn(2, 12)
    a = model(rna, torch.randn(3, 1536), **inputs)["beta"]
    b = model(rna, torch.randn(3, 1536) * 100, **inputs)["beta"]
    torch.testing.assert_close(a, b)
    a.sum().backward()
    assert torch.isfinite(model.track_embedding.weight.grad).all()
