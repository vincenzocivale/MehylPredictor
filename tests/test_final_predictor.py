from __future__ import annotations

import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import EfficientSingleAttentionPredictor
from methylation_predictor.modeling.retrieval import (
    BatchedLocusToRNAAttention,
)


def _config(n_programs: int = 6) -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(
            kind="locus_attention",
            latent_dim=16,
            n_programs=n_programs,
            program_dim=256,
            n_heads=4,
            dropout=0.0,
        ),
    )


def _model() -> EfficientSingleAttentionPredictor:
    torch.manual_seed(17)
    return EfficientSingleAttentionPredictor(
        48,
        _config(),
        n_ffn_blocks=8,
        n_functional_ffn_blocks=8,
        deep_query=False,
        n_head_ffn_blocks=2,
        final_regressor_dropout=0.15,
    )


def _functional_inputs():
    return {
        "functional_track_indices": torch.tensor(
            [1, 2, 5, 9],
            dtype=torch.int64,
        ),
        "functional_offsets": torch.tensor(
            [0, 2, 2, 3, 4],
            dtype=torch.int64,
        ),
        "functional_dense": torch.randn(4, 23),
    }


def _has_grad(module) -> bool:
    return any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in module.parameters()
    )


def test_final_depth_contract():
    model = _model()
    assert isinstance(
        model.retrieval_attn,
        BatchedLocusToRNAAttention,
    )
    assert len(model.retrieval_ffn) == 8
    assert len(model.functional_ffn) == 8
    assert len(model.head_ffn) == 2
    assert model.deep_query is False


def test_final_forward_shape():
    model = _model().eval()
    out = model(
        torch.randn(3, 48),
        **_functional_inputs(),
    )
    assert out["beta"].shape == (3, 4)
    assert out["mu_hat"].shape == (4,)
    assert torch.all((out["beta"] >= 0) & (out["beta"] <= 1))


def test_mean_proxy_is_gradient_isolated_from_rna_path():
    model = _model().train()
    out = model(
        torch.randn(3, 48),
        **_functional_inputs(),
    )
    model.zero_grad(set_to_none=True)
    out["mu_hat"].sum().backward()

    assert _has_grad(model.track_embedding)
    assert _has_grad(model.dense_encoder)
    assert _has_grad(model.functional_ffn)
    assert _has_grad(model.mean_head)

    assert not _has_grad(model.rna_encoder)
    assert not _has_grad(model.retrieval_attn)
    assert not _has_grad(model.retrieval_ffn)
    assert not _has_grad(model.head_ffn)
    assert not _has_grad(model.final_regressor)


def test_full_backward_reaches_all_final_modules():
    model = _model().train()
    target = torch.rand(3, 4)
    out = model(
        torch.randn(3, 48),
        **_functional_inputs(),
    )

    model.zero_grad(set_to_none=True)
    loss = (
        ((out["beta"] - target) ** 2).mean()
        + 0.15 * ((out["mu_hat"] - target.mean(0)) ** 2).mean()
    )
    loss.backward()

    for module in (
        model.rna_encoder,
        model.track_embedding,
        model.dense_encoder,
        model.functional_ffn,
        model.mean_head,
        model.retrieval_attn,
        model.retrieval_ffn,
        model.head_ffn,
        model.final_regressor,
    ):
        assert _has_grad(module)


def test_chunked_and_unchunked_predictions_match():
    torch.manual_seed(0)
    model = _model().eval()
    functional = _functional_inputs()
    rna = torch.randn(3, 48)

    model.LOCUS_CHUNK = 2
    chunked = model(rna, **functional)["beta"]

    model.LOCUS_CHUNK = 10_000
    unchunked = model(rna, **functional)["beta"]

    torch.testing.assert_close(chunked, unchunked)
