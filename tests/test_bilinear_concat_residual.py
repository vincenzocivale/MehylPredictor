import torch

from methylation_predictor.rna_branch.config import (
    EncoderConfig,
    GateConfig,
    InteractionConfig,
    ModelConfig,
)
from methylation_predictor.rna_branch.models import BilinearConcatResidualInteraction, ResidualMethylationModel


def _model(zero_init_residual: bool) -> ResidualMethylationModel:
    config = ModelConfig(
        encoder=EncoderConfig(kind="linear", latent_dim=5, layer_norm=True),
        interaction=InteractionConfig(kind="bilinear_concat_residual", hidden_dim=7, dropout=0.0),
        gate=GateConfig(kind="variability", hidden_dim=4, dropout=0.0),
        anchor_to_mean_rna=False,
        zero_init_residual=zero_init_residual,
    )
    return ResidualMethylationModel(input_dim=11, locus_dim=6, config=config)


def test_zero_init_residual_reproduces_pure_prior() -> None:
    """Matches every other interaction kind's zero_init_residual convention:
    at init the whole interaction output is zero, so beta_hat == prior exactly."""
    torch.manual_seed(3)
    model = _model(zero_init_residual=True).eval()
    rna = torch.randn(4, 11)
    loci = torch.randn(9, 6)
    prior = torch.rand(9) * 0.8 + 0.1
    variability = torch.randn(9, 2)

    output = model(rna, loci, prior, variability)

    torch.testing.assert_close(output["delta_logit"], torch.zeros_like(output["delta_logit"]))
    torch.testing.assert_close(output["beta"], prior.unsqueeze(0).expand_as(output["beta"]), rtol=1e-6, atol=1e-6)


def test_forward_equals_sum_of_bilinear_and_concat_branches() -> None:
    torch.manual_seed(5)
    model = _model(zero_init_residual=False).eval()
    assert isinstance(model.interaction, BilinearConcatResidualInteraction)
    rna = torch.randn(4, 11)
    loci = torch.randn(9, 6)

    rna_repr = model.rna_encoder(rna)
    expected = model.interaction.bilinear(rna_repr, loci) + model.interaction.residual(rna_repr, loci)
    actual = model.interaction(rna_repr, loci)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_gradients_flow_into_both_branches() -> None:
    torch.manual_seed(9)
    model = _model(zero_init_residual=False).train()
    rna = torch.randn(3, 11)
    loci = torch.randn(8, 6)
    prior = torch.rand(8) * 0.8 + 0.1
    variability = torch.randn(8, 2)

    output = model(rna, loci, prior, variability)
    output["beta"].mean().backward()

    assert model.interaction.bilinear.rna_projection[-1].weight.grad is not None
    assert torch.isfinite(model.interaction.bilinear.rna_projection[-1].weight.grad).all()
    assert model.interaction.residual.network[-1].weight.grad is not None
    assert torch.isfinite(model.interaction.residual.network[-1].weight.grad).all()
