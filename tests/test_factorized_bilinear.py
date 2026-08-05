import math

import pytest
import torch

from methylation_predictor.rna_branch.config import (
    EncoderConfig,
    GateConfig,
    InteractionConfig,
    ModelConfig,
)
from methylation_predictor.rna_branch.models import BilinearInteraction, ResidualMethylationModel


def _model(anchor: bool) -> ResidualMethylationModel:
    config = ModelConfig(
        encoder=EncoderConfig(kind="linear", latent_dim=5, layer_norm=True),
        interaction=InteractionConfig(kind="bilinear", hidden_dim=7, dropout=0.0),
        gate=GateConfig(kind="variability", hidden_dim=4, dropout=0.0),
        anchor_to_mean_rna=anchor,
        zero_init_residual=False,
    )
    return ResidualMethylationModel(input_dim=11, locus_dim=6, config=config).eval()


@pytest.mark.parametrize("anchor", [False, True])
def test_factorized_path_matches_explicit_bilinear_formula(anchor: bool) -> None:
    torch.manual_seed(7)
    model = _model(anchor)
    rna = torch.randn(4, 11)
    reference = torch.randn(1, 11)
    loci = torch.randn(9, 6)
    prior = torch.rand(9) * 0.8 + 0.1
    variability = torch.randn(9, 2)

    assert isinstance(model.interaction, BilinearInteraction)
    patient = model.interaction.project_rna(model.rna_encoder(rna))
    if anchor:
        patient = patient - model.interaction.project_rna(model.rna_encoder(reference))
    locus = model.interaction.project_loci(loci)
    raw = patient @ locus.T / math.sqrt(7)
    gate = model.gate(loci, variability)
    expected = torch.sigmoid(torch.logit(prior).unsqueeze(0) + raw * gate.unsqueeze(0))

    direct = model(rna, loci, prior, variability, reference)
    reference_factors = model.encode_reference_factors(reference) if anchor else None
    cached_patient = model.encode_patient_factors(
        rna, reference, reference_factors=reference_factors
    )
    state = model.encode_locus_factors(loci, prior, variability)
    cached = model.predict_from_factors(cached_patient, state)

    torch.testing.assert_close(direct["beta"], expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cached["beta"], expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cached["delta_logit"], direct["delta_logit"])


def test_factorized_training_path_has_gradients() -> None:
    torch.manual_seed(11)
    model = _model(anchor=True).train()
    rna = torch.randn(3, 11)
    reference = torch.randn(1, 11)
    loci = torch.randn(8, 6)
    prior = torch.rand(8) * 0.8 + 0.1
    variability = torch.randn(8, 2)

    output = model(rna, loci, prior, variability, reference)
    output["beta"].mean().backward()

    assert model.interaction.rna_projection[-1].weight.grad is not None
    assert model.interaction.locus_projection[-1].weight.grad is not None
    assert torch.isfinite(model.interaction.locus_projection[-1].weight.grad).all()
