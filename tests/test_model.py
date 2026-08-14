import pytest
import torch

from methylation_predictor.config import ModelConfig
from methylation_predictor.models import RNA2DNAmModel


def _config() -> ModelConfig:
    cfg = ModelConfig()
    cfg.encoder.kind = "linear"
    cfg.encoder.latent_dim = 256
    cfg.encoder.layer_norm = True
    cfg.interaction.kind = "concat"
    cfg.interaction.hidden_dim = 128
    cfg.interaction.dropout = 0.1
    cfg.zero_init_residual = True
    return cfg


def test_defaults_are_the_selected_final_architecture():
    cfg = ModelConfig()
    assert cfg.encoder.kind == "linear"
    assert cfg.encoder.latent_dim == 256
    assert cfg.interaction.kind == "concat"
    assert cfg.zero_init_residual is True


def test_zero_init_starts_exactly_from_prior():
    torch.manual_seed(1)
    model = RNA2DNAmModel(input_dim=17, locus_dim=11, config=_config())
    rna = torch.randn(3, 17)
    loci = torch.randn(5, 11)
    prior = torch.rand(5).clamp(0.05, 0.95)

    out = model(rna, loci, prior)

    assert out["beta"].shape == (3, 5)
    assert torch.allclose(out["beta"], prior.unsqueeze(0).expand(3, -1), atol=1e-6)
    assert torch.allclose(out["delta_logit"], torch.zeros_like(out["delta_logit"]))


def test_rna_latent_is_fixed_to_256():
    model = RNA2DNAmModel(input_dim=17, locus_dim=11, config=_config())
    assert model.rna_encoder.projection.out_features == 256

    cfg = _config()
    cfg.encoder.latent_dim = 64
    with pytest.raises(ValueError, match="256-D RNA latent"):
        RNA2DNAmModel(input_dim=17, locus_dim=11, config=cfg)


def test_product_interaction_is_mandatory():
    model = RNA2DNAmModel(input_dim=13, locus_dim=7, config=_config())
    assert model.interaction.rna_product is not None
    assert model.interaction.locus_product is not None


def test_compatibility_arguments_do_not_change_prediction():
    torch.manual_seed(3)
    model = RNA2DNAmModel(input_dim=13, locus_dim=7, config=_config())
    model.interaction.network[-1].weight.data.normal_()
    model.interaction.network[-1].bias.data.normal_()
    model.eval()

    rna = torch.randn(3, 13)
    loci = torch.randn(4, 7)
    prior = torch.rand(4).clamp(0.05, 0.95)
    a = model(
        rna,
        loci,
        prior,
        variability=torch.randn(4, 2),
        reference_rna=torch.randn(1, 13),
        cancer_codes=torch.tensor([0, 1, 0]),
        cancer_centroids=torch.randn(2, 256),
    )
    b = model(rna, loci, prior)

    assert torch.allclose(a["beta"], b["beta"], atol=1e-6)
    assert torch.allclose(a["delta_logit"], b["delta_logit"], atol=1e-6)
    assert torch.equal(a["gate"], torch.ones_like(a["gate"]))


def test_prediction_is_always_residual_to_frozen_prior():
    torch.manual_seed(4)
    cfg = _config()
    model = RNA2DNAmModel(input_dim=13, locus_dim=7, config=cfg)
    model.interaction.network[-1].weight.data.normal_()
    model.interaction.network[-1].bias.data.normal_()
    model.eval()

    rna = torch.randn(3, 13)
    loci = torch.randn(4, 7)
    prior_a = torch.full((4,), 0.1)
    prior_b = torch.full((4,), 0.9)
    out_a = model(rna, loci, prior_a)
    out_b = model(rna, loci, prior_b)

    assert torch.allclose(out_a["delta_logit"], out_b["delta_logit"], atol=1e-6)
    assert not torch.allclose(out_a["beta"], out_b["beta"])


def test_noncanonical_interaction_fails_closed():
    cfg = _config()
    cfg.interaction.kind = "bilinear"
    with pytest.raises(ValueError, match="interaction.kind='concat'"):
        RNA2DNAmModel(input_dim=5, locus_dim=3, config=cfg)


def test_zero_initialization_is_not_optional():
    cfg = _config()
    cfg.zero_init_residual = False
    with pytest.raises(ValueError, match="zero_init_residual=true"):
        RNA2DNAmModel(input_dim=5, locus_dim=3, config=cfg)
