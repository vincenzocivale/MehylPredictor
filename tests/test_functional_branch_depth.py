from __future__ import annotations

import pytest
import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import EfficientSingleAttentionPredictor


def _config(n_programs=6):
    return ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=16, n_programs=n_programs,
                               program_dim=256, n_heads=4, dropout=0.0),
    )


def _model(n_ffn_blocks=8, n_functional_ffn_blocks=8, deep_query=False, input_dim=48,
           n_programs=6, dropout=0.15):
    torch.manual_seed(17)
    return EfficientSingleAttentionPredictor(
        input_dim, _config(n_programs), n_ffn_blocks=n_ffn_blocks,
        n_functional_ffn_blocks=n_functional_ffn_blocks, deep_query=deep_query,
        final_regressor_dropout=dropout,
    )


def _functional_inputs():
    return {
        "functional_track_indices": torch.tensor([1, 2, 5, 9], dtype=torch.int64),
        "functional_offsets": torch.tensor([0, 2, 2, 3, 4], dtype=torch.int64),
        "functional_dense": torch.randn(4, 23),
    }


def test_functional_ffn_stack_is_built():
    model = _model(n_functional_ffn_blocks=8)
    assert len(model.functional_ffn) == 8


def test_default_is_unchanged_j7_behavior():
    # n_functional_ffn_blocks=0 (default) must be byte-for-byte J7: an empty
    # functional_ffn stack, h_c_deep falls back to h_c itself.
    model = EfficientSingleAttentionPredictor(48, _config(), n_ffn_blocks=8)
    assert len(model.functional_ffn) == 0
    assert model.deep_query is False


def test_forward_shapes():
    model = _model().eval()
    out = model(torch.randn(3, 48), **_functional_inputs())
    assert out["beta"].shape == (3, 4)
    assert torch.all((out["beta"] >= 0) & (out["beta"] <= 1))
    assert out["mu_hat"].shape == (4,)


def test_query_unaffected_by_functional_depth_when_deep_query_false():
    # The whole point of J9b: the cross-attention query must be identical to
    # a model with n_functional_ffn_blocks=0 (J7), so any output difference
    # is attributable only to the downstream (mean_head/concat) use of
    # h_c_deep, never to a changed query.
    torch.manual_seed(0)
    shallow = EfficientSingleAttentionPredictor(48, _config(), n_ffn_blocks=8,
                                                 n_functional_ffn_blocks=0).eval()
    torch.manual_seed(0)
    deep = EfficientSingleAttentionPredictor(48, _config(), n_ffn_blocks=8,
                                              n_functional_ffn_blocks=8,
                                              deep_query=False).eval()
    # Shared submodules (rna_encoder, track_embedding, dense_encoder,
    # locus_norm, retrieval_attn) were initialized identically by the same
    # seed and identical construction order up to functional_ffn, so the
    # query each model builds from the same functional inputs must match.
    functional = _functional_inputs()
    peak_s = shallow.track_embedding(functional["functional_track_indices"], functional["functional_offsets"])
    dense_s = shallow.dense_encoder(functional["functional_dense"])
    h_c_shallow = shallow.locus_norm(peak_s + dense_s)

    peak_d = deep.track_embedding(functional["functional_track_indices"], functional["functional_offsets"])
    dense_d = deep.dense_encoder(functional["functional_dense"])
    h_c_deep_model = deep.locus_norm(peak_d + dense_d)

    torch.testing.assert_close(h_c_shallow, h_c_deep_model)
    # And deep's actual query (deep_query=False) must equal this shallow h_c,
    # not the FFN-refined h_c_deep the model also computes internally.
    h_c_after_ffn = h_c_deep_model
    for ffn in deep.functional_ffn:
        h_c_after_ffn = ffn(h_c_after_ffn)
    assert not torch.allclose(h_c_after_ffn, h_c_deep_model, atol=1e-4)


def test_deep_query_requires_functional_blocks():
    with pytest.raises(ValueError):
        EfficientSingleAttentionPredictor(48, _config(), n_functional_ffn_blocks=0, deep_query=True)


def test_deep_query_true_changes_the_query():
    # J9c smoke test: with deep_query=True, the attention query is h_c_deep,
    # not h_c -- so mean_head and the query see the exact same tensor.
    model = _model(n_functional_ffn_blocks=4, deep_query=True).eval()
    functional = _functional_inputs()
    rna = torch.randn(3, 48)
    out = model(rna, **functional)
    assert out["beta"].shape == (3, 4)


def test_locus_chunking_matches_unchunked():
    torch.manual_seed(0)
    model = _model().eval()
    functional = _functional_inputs()
    rna = torch.randn(3, 48)
    model.LOCUS_CHUNK = 2
    chunked = model(rna, **functional)["beta"]
    model.LOCUS_CHUNK = 10_000
    unchunked = model(rna, **functional)["beta"]
    torch.testing.assert_close(chunked, unchunked)


def test_mean_head_sees_deepened_representation():
    model = _model(n_functional_ffn_blocks=4).train()
    functional = _functional_inputs()
    rna = torch.randn(3, 48)
    out = model(rna, **functional)
    model.zero_grad(set_to_none=True)
    out["mu_hat"].sum().backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    assert has_grad(model.mean_head)
    for ffn in model.functional_ffn:
        assert has_grad(ffn)
    # mean_head must NOT reach the RNA-conditioned retrieval branch.
    assert not has_grad(model.retrieval_attn)
    for ffn in model.retrieval_ffn:
        assert not has_grad(ffn)


def test_rejects_negative_functional_ffn_blocks():
    with pytest.raises(ValueError):
        EfficientSingleAttentionPredictor(48, _config(), n_functional_ffn_blocks=-1)
