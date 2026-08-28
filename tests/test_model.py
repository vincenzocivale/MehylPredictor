import pytest
import torch

from methylation_predictor.config import ModelConfig
from methylation_predictor.models import (
    DirectPredictionModel,
    RNA2DNAmModel,
    VarianceNormalizedResidualModel,
)


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


def test_include_product_defaults_true_and_is_opt_in_only():
    cfg = _config()
    assert cfg.interaction.include_product is True
    model = VarianceNormalizedResidualModel(input_dim=13, locus_dim=7, config=cfg)
    assert model.interaction.include_product is True
    assert model.interaction.rna_product is not None
    assert model.interaction.locus_product is not None


def test_no_product_ablation_drops_product_term_and_still_starts_at_prior():
    torch.manual_seed(5)
    cfg = _config()
    cfg.interaction.include_product = False
    model = VarianceNormalizedResidualModel(input_dim=13, locus_dim=7, config=cfg)
    model.eval()

    assert model.interaction.rna_product is None
    assert model.interaction.locus_product is None
    # joint_dim should be rna_dim(256) + locus_dim(7), no product_dim added.
    assert model.interaction.network[1].in_features == 256 + 7

    rna = torch.randn(3, 13)
    loci = torch.randn(4, 7)
    prior = torch.rand(4).clamp(0.05, 0.95)
    sigma = torch.rand(4).clamp(0.05, 1.0)

    out = model(rna, loci, prior, sigma)

    assert out["beta"].shape == (3, 4)
    assert torch.allclose(out["beta"], prior.unsqueeze(0).expand(3, -1), atol=1e-6)


def test_product_only_ablation_drops_raw_rna_and_cpg():
    torch.manual_seed(6)
    cfg = _config()
    cfg.interaction.include_rna = False
    cfg.interaction.include_cpg = False
    cfg.interaction.include_product = True
    model = VarianceNormalizedResidualModel(input_dim=13, locus_dim=7, config=cfg)
    model.eval()

    product_dim = min(256, 7)
    assert model.interaction.network[1].in_features == product_dim

    rna = torch.randn(3, 13)
    loci = torch.randn(4, 7)
    prior = torch.rand(4).clamp(0.05, 0.95)
    sigma = torch.rand(4).clamp(0.05, 1.0)
    out = model(rna, loci, prior, sigma)
    assert out["beta"].shape == (3, 4)


def test_cpg_product_ablation_drops_raw_rna_only():
    torch.manual_seed(7)
    cfg = _config()
    cfg.interaction.include_rna = False
    cfg.interaction.include_cpg = True
    cfg.interaction.include_product = True
    model = VarianceNormalizedResidualModel(input_dim=13, locus_dim=7, config=cfg)
    model.eval()

    product_dim = min(256, 7)
    assert model.interaction.network[1].in_features == 7 + product_dim

    rna = torch.randn(3, 13)
    loci = torch.randn(4, 7)
    prior = torch.rand(4).clamp(0.05, 0.95)
    sigma = torch.rand(4).clamp(0.05, 1.0)
    out = model(rna, loci, prior, sigma)
    assert out["beta"].shape == (3, 4)


def test_at_least_one_interaction_piece_is_required():
    cfg = _config()
    cfg.interaction.include_rna = False
    cfg.interaction.include_cpg = False
    cfg.interaction.include_product = False
    with pytest.raises(ValueError, match="at least one"):
        VarianceNormalizedResidualModel(input_dim=13, locus_dim=7, config=cfg)


def test_wider_rna_latent_is_an_opt_in_scaling_ablation():
    torch.manual_seed(8)
    cfg = _config()
    cfg.encoder.latent_dim = 512
    model = VarianceNormalizedResidualModel(input_dim=13, locus_dim=7, config=cfg)
    model.eval()

    assert model.rna_encoder.projection.out_features == 512
    product_dim = min(512, 7)
    assert model.interaction.network[1].in_features == 512 + 7 + product_dim

    rna = torch.randn(3, 13)
    loci = torch.randn(4, 7)
    prior = torch.rand(4).clamp(0.05, 0.95)
    sigma = torch.rand(4).clamp(0.05, 1.0)
    out = model(rna, loci, prior, sigma)
    assert out["beta"].shape == (3, 4)
    # zero-init still holds regardless of latent width
    assert torch.allclose(out["beta"], prior.unsqueeze(0).expand(3, -1), atol=1e-6)


def test_rna2dnam_still_rejects_nonstandard_latent_dim():
    # RNA2DNAmModel (legacy flat-residual baseline) is untouched by the
    # latent-dim ablation: it must keep failing closed on non-256 widths.
    cfg = _config()
    cfg.encoder.latent_dim = 512
    with pytest.raises(ValueError, match="256-D RNA latent"):
        RNA2DNAmModel(input_dim=5, locus_dim=3, config=cfg)


@pytest.mark.parametrize("kind", ["film", "cross_attention", "bilinear"])
def test_fusion_mechanism_ablation_forward_shapes_and_zero_init(kind):
    torch.manual_seed(9)
    cfg = _config()
    cfg.interaction.kind = kind
    model = VarianceNormalizedResidualModel(input_dim=13, locus_dim=7, config=cfg)
    model.eval()

    rna = torch.randn(3, 13)
    loci = torch.randn(4, 7)
    prior = torch.rand(4).clamp(0.05, 0.95)
    sigma = torch.rand(4).clamp(0.05, 1.0)
    out = model(rna, loci, prior, sigma)

    assert out["beta"].shape == (3, 4)
    # zero_output() still applies regardless of interaction kind: the model
    # starts exactly at the prior for every fusion mechanism.
    assert torch.allclose(out["beta"], prior.unsqueeze(0).expand(3, -1), atol=1e-6)

    # gradients flow through the new interaction module's parameters.
    model.train()
    out = model(rna, loci, prior, sigma)
    out["beta"].sum().backward()
    grads = [p.grad for p in model.interaction.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)


def test_rna2dnam_still_rejects_fusion_mechanism_kinds():
    # RNA2DNAmModel (legacy flat-residual baseline) is not part of the
    # fusion-mechanism ablation: it must keep failing closed.
    cfg = _config()
    cfg.interaction.kind = "film"
    with pytest.raises(ValueError, match="interaction.kind='concat'"):
        RNA2DNAmModel(input_dim=5, locus_dim=3, config=cfg)


def _direct_prediction_config() -> ModelConfig:
    cfg = _config()
    cfg.use_prior_anchor = False
    cfg.zero_init_residual = False
    return cfg


def test_direct_prediction_model_forward_shape_and_ignores_prior():
    torch.manual_seed(10)
    model = DirectPredictionModel(input_dim=13, locus_dim=7, config=_direct_prediction_config())
    model.eval()

    rna = torch.randn(3, 13)
    loci = torch.randn(4, 7)
    prior_a = torch.full((4,), 0.1)
    prior_b = torch.full((4,), 0.9)

    out_a = model(rna, loci, prior_a)
    out_b = model(rna, loci, prior_b)

    assert out_a["beta"].shape == (3, 4)
    # no prior anchor: changing prior does not change the prediction at all.
    assert torch.allclose(out_a["beta"], out_b["beta"], atol=1e-6)
    assert out_a["prior_logit"] is None


def test_direct_prediction_model_requires_no_prior_anchor_flag():
    cfg = _config()  # use_prior_anchor defaults True, zero_init_residual True
    with pytest.raises(ValueError, match="use_prior_anchor=false"):
        DirectPredictionModel(input_dim=5, locus_dim=3, config=cfg)


def test_direct_prediction_model_rejects_zero_init_residual():
    cfg = _config()
    cfg.use_prior_anchor = False
    # zero_init_residual left True: there is no anchor to start "at zero" relative to.
    with pytest.raises(ValueError, match="zero_init_residual=false"):
        DirectPredictionModel(input_dim=5, locus_dim=3, config=cfg)
