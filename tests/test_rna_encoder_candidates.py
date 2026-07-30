from __future__ import annotations

import numpy as np
import pytest
import torch

from methylation_predictor.rna_branch.config import EncoderConfig
from methylation_predictor.rna_branch.metrics import evaluate_predictions
from methylation_predictor.rna_branch.models import LinearResidualRNAEncoder, build_rna_encoder


@pytest.mark.parametrize(
    ("kind", "extra"),
    [
        ("linear", {}),
        ("mlp", {"hidden_dims": [12]}),
        ("bottleneck_mlp", {"width": 12, "num_blocks": 2, "expansion_factor": 2}),
        ("linear_residual", {"width": 12, "num_blocks": 1}),
        ("gated_residual", {"width": 12, "num_blocks": 1}),
        (
            "perceiver",
            {"token_dim": 8, "num_latents": 2, "num_heads": 2, "num_self_attention_blocks": 1},
        ),
    ],
)
def test_rna_encoder_candidates_return_standard_representation(kind: str, extra: dict[str, object]):
    config = EncoderConfig(kind=kind, latent_dim=6, dropout=0.0, **extra)
    encoder = build_rna_encoder(10, config)
    representation = encoder(torch.randn(4, 10))
    assert representation.global_vector.shape == (4, 6)
    assert torch.isfinite(representation.global_vector).all()
    if kind == "perceiver":
        assert representation.tokens is not None
        assert representation.tokens.shape[:2] == (4, 2)


def test_linear_tokens_encoder_reshapes_into_k_tokens():
    config = EncoderConfig(kind="linear_tokens", token_dim=4, num_latents=3, dropout=0.0)
    encoder = build_rna_encoder(10, config)
    representation = encoder(torch.randn(5, 10))
    assert representation.tokens is not None
    assert representation.tokens.shape == (5, 3, 4)
    assert representation.global_vector.shape == (5, 12)
    assert torch.isfinite(representation.global_vector).all()


@pytest.mark.parametrize("kind", ["linear_residual", "gated_residual"])
def test_residual_encoder_starts_from_exact_linear_path(kind: str):
    config = EncoderConfig(
        kind=kind,
        latent_dim=6,
        width=12,
        num_blocks=1,
        dropout=0.0,
        zero_init_encoder_residual=True,
    )
    encoder = build_rna_encoder(10, config)
    assert isinstance(encoder, LinearResidualRNAEncoder)
    x = torch.randn(4, 10)
    assert torch.equal(encoder.residual(x), torch.zeros(4, 6))
    assert torch.allclose(encoder(x).global_vector, encoder.base(x).global_vector)


def test_unknown_activation_is_rejected():
    config = EncoderConfig(kind="mlp", latent_dim=4, hidden_dims=[8], activation="invalid")
    with pytest.raises(ValueError, match="unknown activation"):
        build_rna_encoder(10, config)


def test_metrics_expose_patient_locus_and_variability_axes():
    prior = np.array([0.2, 0.4, 0.6, 0.8])
    target = np.array(
        [
            [0.1, 0.5, 0.7, 0.9],
            [0.3, 0.3, 0.5, 0.7],
            [0.2, 0.6, 0.4, 0.8],
        ]
    )
    prediction = 0.75 * target + 0.25 * prior[None, :]
    metrics = evaluate_predictions(
        target,
        prediction,
        prior,
        np.array(["A", "A", "B"]),
        cpg_tertiles=np.array([0, 1, 2, 2]),
    )
    for key in (
        "patient_dynamic_pearson_median",
        "patient_dynamic_spearman_median",
        "locus_dynamic_pearson_median",
        "locus_dynamic_spearman_median",
    ):
        assert np.isfinite(metrics[key])
    assert set(metrics["per_variability_tertile"]) == {"low", "mid", "high"}
    assert metrics["per_variability_tertile"]["high"]["cpgs"] == 2
