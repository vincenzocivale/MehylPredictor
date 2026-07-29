from __future__ import annotations

import torch

from methylation_predictor.rna_branch.config import ModelConfig, EncoderConfig, InteractionConfig, GateConfig
from methylation_predictor.rna_branch.models import ResidualMethylationModel


def test_reference_rna_returns_exact_prior():
    config = ModelConfig(
        encoder=EncoderConfig(kind="mlp", latent_dim=8, hidden_dims=[16], dropout=0.0),
        interaction=InteractionConfig(kind="bilinear", hidden_dim=8, dropout=0.0),
        gate=GateConfig(kind="variability", hidden_dim=8),
        anchor_to_mean_rna=True,
        zero_init_residual=False,
    )
    model = ResidualMethylationModel(10, 6, config)
    reference = torch.zeros(1, 10)
    loci = torch.randn(5, 6)
    prior = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    variability = torch.randn(5, 2)
    output = model(reference, loci, prior, variability, reference)
    assert torch.allclose(output["beta"][0], prior, atol=1e-6)
    assert torch.allclose(output["delta_logit"], torch.zeros_like(output["delta_logit"]), atol=1e-7)


def test_perceiver_fourier_cross_attention_shapes():
    config = ModelConfig(
        encoder=EncoderConfig(
            kind="perceiver",
            latent_dim=8,
            token_dim=16,
            num_latents=4,
            num_heads=4,
            num_self_attention_blocks=1,
            value_encoding="fourier",
            fourier_frequencies=3,
            dropout=0.0,
        ),
        interaction=InteractionConfig(
            kind="cross_attention",
            hidden_dim=12,
            num_heads=4,
            dropout=0.0,
        ),
        gate=GateConfig(kind="global", initial_global_scale=1.0),
        anchor_to_mean_rna=True,
        zero_init_residual=False,
    )
    model = ResidualMethylationModel(11, 7, config)
    rna = torch.randn(3, 11)
    reference = torch.zeros(1, 11)
    loci = torch.randn(5, 7)
    prior = torch.linspace(0.1, 0.9, 5)
    variability = torch.randn(5, 2)
    output = model(rna, loci, prior, variability, reference)
    assert output["beta"].shape == (3, 5)
    assert output["delta_logit"].shape == (3, 5)
    assert torch.isfinite(output["beta"]).all()
