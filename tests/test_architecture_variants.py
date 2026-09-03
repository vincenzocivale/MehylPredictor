"""Tests for the architecture-novelty suite (architecture_novelty_2026_09).

Covers the three invariants that make the suite's numbers trustworthy:

1. every arm starts exactly at the prior (``zero_output`` contract), so an arm
   that learns nothing reproduces the CpG-prior floor rather than noise;
2. the all-default configuration of ``ArchitectureVariantModel`` is the
   canonical architecture, making it a valid same-code control; and
3. mHC's residual mapping really is doubly stochastic, which is the whole basis
   of its conservation/stability claim.
"""
from __future__ import annotations

import pytest
import torch

from methylation_predictor.config import (
    AxialConfig,
    EncoderConfig,
    InteractionConfig,
    LossConfig,
    ModelConfig,
    TrunkConfig,
)
from methylation_predictor.losses import residual_loss
from methylation_predictor.models import (
    ArchitectureVariantModel,
    BilinearInteraction,
    CpGTrunk,
    FeatureFusionArchitectureVariantModel,
    FeatureFusionLocusCLSModel,
    ProductInteraction,
    RNARepresentation,
    feature_fusion_variant_label,
    AxialCpGAttention,
    HyperConnectionTrunk,
    RNAMethylationPredictor,
    architecture_variant_label,
    is_architecture_variant,
    sinkhorn_knopp,
)

INPUT_DIM = 64
LOCUS_DIM = 32
N_SAMPLES = 5
N_LOCI = 7


def _model_config(**overrides) -> ModelConfig:
    config = ModelConfig(
        encoder=EncoderConfig(latent_dim=16, hidden_dim=24, n_programs=6, program_dim=8, n_heads=2),
        interaction=InteractionConfig(hidden_dim=12),
        zero_init_residual=True,
        variance_normalized_residual=True,
        use_prior_anchor=True,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _batch(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    rna = torch.randn(N_SAMPLES, INPUT_DIM, generator=generator)
    loci = torch.randn(N_LOCI, LOCUS_DIM, generator=generator)
    prior = torch.rand(N_LOCI, generator=generator) * 0.8 + 0.1
    sigma = torch.rand(N_LOCI, generator=generator) * 0.5 + 0.1
    return rna, loci, prior, sigma


def _build(config: ModelConfig) -> ArchitectureVariantModel:
    return ArchitectureVariantModel(INPUT_DIM, LOCUS_DIM, config, epsilon=1e-4)


# --------------------------------------------------------------------------
# 1. Every arm starts exactly at the prior.
# --------------------------------------------------------------------------

ARM_CONFIGS = {
    "canonical_equivalent": {},
    "encoder_mlp": {"encoder": EncoderConfig(kind="mlp", latent_dim=16, hidden_dim=24)},
    "encoder_program_bottleneck": {"encoder": EncoderConfig(kind="program_bottleneck", latent_dim=16, n_programs=6)},
    "encoder_locus_attention": {
        "encoder": EncoderConfig(kind="locus_attention", latent_dim=16, n_programs=6, program_dim=8, n_heads=2)
    },
    "trunk_plain": {"trunk": TrunkConfig(kind="plain", depth=3, width=12)},
    "trunk_hc": {"trunk": TrunkConfig(kind="hc", depth=3, width=12, n_streams=4)},
    "trunk_mhc": {"trunk": TrunkConfig(kind="mhc", depth=3, width=12, n_streams=4)},
    "trunk_mhc_semantic": {"trunk": TrunkConfig(kind="mhc", depth=3, width=12, n_streams=4, stream_semantics=True)},
    "axial": {"trunk": TrunkConfig(kind="plain", depth=2, width=12), "axial": AxialConfig(enabled=True, n_heads=2, window=4)},
    "beta_head": {"beta_likelihood_head": True},
    "combined": {
        "encoder": EncoderConfig(kind="locus_attention", latent_dim=16, n_programs=6, program_dim=8, n_heads=2),
        "trunk": TrunkConfig(kind="mhc", depth=2, width=12, n_streams=4, stream_semantics=True),
        "axial": AxialConfig(enabled=True, n_heads=2, window=4),
    },
}


@pytest.mark.parametrize("name", sorted(ARM_CONFIGS))
def test_every_arm_starts_at_the_prior(name):
    """raw_delta must be exactly zero at init, so beta_hat == mu on step 0."""
    model = _build(_model_config(**ARM_CONFIGS[name])).eval()
    rna, loci, prior, sigma = _batch()
    with torch.no_grad():
        outputs = model(rna, loci, prior, sigma)
    assert outputs["beta"].shape == (N_SAMPLES, N_LOCI)
    assert torch.allclose(outputs["raw_delta"], torch.zeros_like(outputs["raw_delta"]))
    expected = prior.unsqueeze(0).expand(N_SAMPLES, N_LOCI)
    assert torch.allclose(outputs["beta"], expected, atol=1e-5)


@pytest.mark.parametrize("name", sorted(ARM_CONFIGS))
def test_every_arm_produces_finite_gradients(name):
    model = _build(_model_config(**ARM_CONFIGS[name])).train()
    rna, loci, prior, sigma = _batch(seed=1)
    target = torch.rand(N_SAMPLES, N_LOCI)
    target[0, 0] = float("nan")  # missing observations must not poison the backward pass
    config = LossConfig(locus_pearson_weight=0.15, locus_min_observed_samples=2)
    if ARM_CONFIGS[name].get("beta_likelihood_head"):
        config.beta_nll_weight = 0.1
    outputs = model(rna, loci, prior, sigma)
    loss, pieces = residual_loss(outputs, target, prior, config, sigma=sigma)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert pieces["observed"] == N_SAMPLES * N_LOCI - 1


# --------------------------------------------------------------------------
# 2. The all-default variant is the canonical architecture.
# --------------------------------------------------------------------------

def test_default_variant_matches_canonical_parameter_shapes():
    config = _model_config()
    assert not is_architecture_variant(config)
    variant = _build(config)
    canonical = RNAMethylationPredictor(INPUT_DIM, LOCUS_DIM, config, epsilon=1e-4)
    variant_shapes = sorted(tuple(p.shape) for p in variant.parameters())
    canonical_shapes = sorted(tuple(p.shape) for p in canonical.parameters())
    assert variant_shapes == canonical_shapes
    assert sum(p.numel() for p in variant.parameters()) == sum(p.numel() for p in canonical.parameters())


def test_default_variant_matches_canonical_outputs_under_copied_weights():
    """Same joint input, same depth-1 head, same anchor -- so with the canonical
    model's weights transplanted in, the two must agree numerically."""
    config = _model_config()
    variant = _build(config).eval()
    canonical = RNAMethylationPredictor(INPUT_DIM, LOCUS_DIM, config, epsilon=1e-4).eval()
    # Break the zero-init so the comparison is not trivially 0 == 0.
    torch.nn.init.normal_(canonical.interaction.network[-1].weight, std=0.1)
    torch.nn.init.normal_(canonical.interaction.network[-1].bias, std=0.1)

    variant.rna_encoder.load_state_dict(canonical.rna_encoder.state_dict())
    variant.rna_product.load_state_dict(canonical.interaction.rna_product.state_dict())
    variant.locus_product.load_state_dict(canonical.interaction.locus_product.state_dict())
    variant.feature_stage[0].load_state_dict(canonical.interaction.network[0].state_dict())
    variant.feature_stage[1].load_state_dict(canonical.interaction.network[1].state_dict())
    variant.head.load_state_dict(canonical.interaction.network[-1].state_dict())

    rna, loci, prior, sigma = _batch(seed=2)
    with torch.no_grad():
        assert torch.allclose(
            variant(rna, loci, prior, sigma)["beta"],
            canonical(rna, loci, prior, sigma)["beta"],
            atol=1e-6,
        )


def test_variant_detection_and_labels():
    assert is_architecture_variant(_model_config(trunk=TrunkConfig(kind="plain", depth=2)))
    assert is_architecture_variant(_model_config(encoder=EncoderConfig(kind="mlp")))
    assert is_architecture_variant(_model_config(axial=AxialConfig(enabled=True)))
    assert is_architecture_variant(_model_config(beta_likelihood_head=True))
    plain = architecture_variant_label(_model_config(trunk=TrunkConfig(kind="plain", depth=4, width=128)))
    semantic = architecture_variant_label(
        _model_config(trunk=TrunkConfig(kind="mhc", depth=4, width=128, n_streams=4, stream_semantics=True))
    )
    # Distinct labels are what stops one arm resuming from another's latest.pt.
    assert plain != semantic
    assert "trunk-plaind4w128" in plain and "sem" in semantic


# --------------------------------------------------------------------------
# 3. The manifold constraint.
# --------------------------------------------------------------------------

def test_sinkhorn_projection_is_doubly_stochastic():
    logits = torch.randn(6, 6, generator=torch.Generator().manual_seed(3))
    matrix = sinkhorn_knopp(logits, iterations=50)
    assert (matrix >= 0).all()
    assert torch.allclose(matrix.sum(dim=-1), torch.ones(6), atol=1e-4)
    assert torch.allclose(matrix.sum(dim=-2), torch.ones(6), atol=1e-4)


def test_single_stream_mhc_degenerates_to_the_identity_mapping():
    """mHC section 4.1: at n = 1 the doubly stochastic condition is the scalar 1,
    recovering the ordinary residual connection."""
    trunk = HyperConnectionTrunk(width=8, depth=2, expansion=2, dropout=0.0, n_streams=1, manifold=True)
    for layer in range(2):
        h_res, _, _ = trunk._mappings(layer)
        assert torch.allclose(h_res, torch.ones(1, 1), atol=1e-6)


def test_mhc_composite_gain_is_conserved_but_hc_is_not():
    """The doubly stochastic set is closed under multiplication, so the composite
    mapping across depth stays gain-1; unconstrained HC has no such bound."""
    manifold = HyperConnectionTrunk(width=8, depth=6, expansion=2, dropout=0.0, n_streams=4, manifold=True)
    with torch.no_grad():
        manifold.res_param.normal_(0.0, 1.0)  # arbitrary, still projected onto the polytope
    gain = manifold.composite_gain()
    assert gain["forward_amax_gain"] == pytest.approx(1.0, abs=1e-3)
    assert gain["backward_amax_gain"] == pytest.approx(1.0, abs=1e-3)

    unconstrained = HyperConnectionTrunk(width=8, depth=6, expansion=2, dropout=0.0, n_streams=4, manifold=False)
    with torch.no_grad():
        unconstrained.res_param.normal_(0.0, 1.0)
    assert unconstrained.composite_gain()["forward_amax_gain"] > 1.5


def test_hyper_connection_trunk_starts_near_the_identity_mapping():
    """HC/mHC start as near-independent residual streams, so stream mixing has to
    be learned rather than assumed.

    "Near" is exact: with the default identity_init_scale=4.0 and n=4, Sinkhorn
    turns diag(e^4) + 1 into ~0.948 on the diagonal and ~0.017 off it. That
    residual leakage is deliberate -- a hard identity would give the residual
    mapping no gradient signal to move away from -- so the assertion is diagonal
    dominance plus double stochasticity, not equality with I.
    """
    trunk = HyperConnectionTrunk(width=8, depth=3, expansion=2, dropout=0.0, n_streams=4, manifold=True).eval()
    for layer in range(3):
        h_res, h_pre, h_post = trunk._mappings(layer)
        assert torch.allclose(h_res.sum(dim=-1), torch.ones(4), atol=1e-4)
        assert torch.allclose(h_res.sum(dim=-2), torch.ones(4), atol=1e-4)
        assert (h_res.diagonal() > 0.9).all()
        assert (h_res - torch.diag(h_res.diagonal()) < 0.05).all()
        # H^pre averages the streams and H^post writes the update back unscaled,
        # matching mHC's fixed-mapping defaults (its Table 1).
        assert torch.allclose(h_pre, torch.full((4,), 0.25), atol=1e-5)
        assert torch.allclose(h_post, torch.ones(4), atol=1e-5)


def test_residual_mappings_are_reported_per_depth():
    trunk = HyperConnectionTrunk(width=8, depth=3, expansion=2, dropout=0.0, n_streams=4, manifold=True)
    mappings = trunk.residual_mappings()
    assert len(mappings) == 3
    assert all(len(m) == 4 and len(m[0]) == 4 for m in mappings)


def test_stream_semantics_requires_four_streams_and_a_multi_stream_trunk():
    with pytest.raises(ValueError, match="n_streams must be 4"):
        _build(_model_config(trunk=TrunkConfig(kind="mhc", depth=2, width=12, n_streams=3, stream_semantics=True)))
    with pytest.raises(ValueError, match="multi-stream trunk"):
        _build(_model_config(trunk=TrunkConfig(kind="plain", depth=2, width=12, stream_semantics=True)))
    with pytest.raises(ValueError, match="requires an actual trunk"):
        _build(_model_config(trunk=TrunkConfig(kind="none", stream_semantics=True)))


# --------------------------------------------------------------------------
# Axial attention.
# --------------------------------------------------------------------------

def test_axial_attention_is_identity_at_initialization():
    axial = AxialCpGAttention(width=8, n_heads=2, window=4, n_distance_buckets=8, dropout=0.0).eval()
    hidden = torch.randn(3, 10, 8)
    with torch.no_grad():
        assert torch.allclose(axial(hidden), hidden, atol=1e-6)


@pytest.mark.parametrize("n_loci", [3, 4, 10, 16])
def test_axial_attention_handles_windows_that_do_not_divide_the_axis(n_loci):
    axial = AxialCpGAttention(width=8, n_heads=2, window=4, n_distance_buckets=8, dropout=0.0).eval()
    torch.nn.init.normal_(axial.out.weight, std=0.1)
    hidden = torch.randn(2, n_loci, 8)
    positions = torch.arange(n_loci) * 137
    with torch.no_grad():
        out = axial(hidden, positions=positions)
        out_no_positions = axial(hidden)
    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(out_no_positions).all()


def test_axial_attention_does_not_mix_across_the_sample_axis():
    """Cross-sample attention would be transductive; the arm must stay per-sample."""
    axial = AxialCpGAttention(width=8, n_heads=2, window=4, n_distance_buckets=8, dropout=0.0).eval()
    torch.nn.init.normal_(axial.out.weight, std=0.5)
    torch.nn.init.normal_(axial.qkv.weight, std=0.5)
    hidden = torch.randn(3, 9, 8)
    with torch.no_grad():
        together = axial(hidden)
        alone = axial(hidden[1:2])
    assert torch.allclose(together[1:2], alone, atol=1e-5)


# --------------------------------------------------------------------------
# Beta likelihood head.
# --------------------------------------------------------------------------

def test_beta_nll_rewards_a_sharper_correct_prediction():
    target = torch.full((4, 5), 0.7)
    prior = torch.full((5,), 0.7)
    sigma = torch.full((5,), 0.2)
    config = LossConfig(beta_mse_weight=0.0, beta_nll_weight=1.0, concentration_min=2.0)
    beta = torch.full((4, 5), 0.7)
    loose = {"beta": beta, "delta_logit": torch.zeros(4, 5), "concentration": torch.full((4, 5), 5.0)}
    sharp = {"beta": beta, "delta_logit": torch.zeros(4, 5), "concentration": torch.full((4, 5), 80.0)}
    loose_loss, _ = residual_loss(loose, target, prior, config, sigma=sigma)
    sharp_loss, _ = residual_loss(sharp, target, prior, config, sigma=sigma)
    assert sharp_loss < loose_loss


def test_beta_nll_is_finite_on_boundary_targets():
    """WGBS beta values hit exactly 0 and 1, where the Beta density diverges."""
    target = torch.tensor([[0.0, 1.0, 0.5]])
    prior = torch.tensor([0.5, 0.5, 0.5])
    sigma = torch.tensor([0.2, 0.2, 0.2])
    outputs = {
        "beta": torch.tensor([[0.5, 0.5, 0.5]]),
        "delta_logit": torch.zeros(1, 3),
        "concentration": torch.full((1, 3), 10.0),
    }
    config = LossConfig(beta_mse_weight=0.0, beta_nll_weight=1.0)
    loss, pieces = residual_loss(outputs, target, prior, config, sigma=sigma)
    assert torch.isfinite(loss)
    assert pieces["beta_nll"] > 0.0


def test_beta_nll_is_inert_without_a_concentration_head():
    target = torch.rand(3, 4)
    prior = torch.full((4,), 0.5)
    outputs = {"beta": torch.full((3, 4), 0.5), "delta_logit": torch.zeros(3, 4)}
    config = LossConfig(
        beta_mse_weight=0.0, residual_huber_weight=0.0, shrinkage_weight=0.0, beta_nll_weight=1.0
    )
    loss, pieces = residual_loss(outputs, target, prior, config)
    assert pieces["beta_nll"] == 0.0
    assert float(loss) == 0.0


# --------------------------------------------------------------------------
# Recorded algebraic finding about the 2026-08 fusion suite.
# --------------------------------------------------------------------------

def test_bilinear_at_canonical_rank_is_the_product_term():
    """``BilinearInteraction`` at rank == min(rna_dim, locus_dim) IS the canonical
    ``ProductInteraction``.

    Both compute ``Linear(rna) * Linear(loci)`` concatenated with the raw pieces, so
    with weights transplanted across they agree to the bit. This means
    ``ablations.yaml::fusion_mechanism_2026_08``'s bilinear arm (0.5460, -0.0153 vs
    canonical) measured its hardcoded rank-64 restriction, not a different fusion
    mechanism -- which is why the retest arm sweeps rank *above* 256 instead of
    re-running rank 256.
    """
    rna_dim, locus_dim = 16, 24
    product = ProductInteraction(rna_dim, locus_dim, 8, 0.0).eval()
    bilinear = BilinearInteraction(rna_dim, locus_dim, 8, 0.0, rank=min(rna_dim, locus_dim)).eval()
    bilinear.rna_bilinear.load_state_dict(product.rna_product.state_dict())
    bilinear.locus_bilinear.load_state_dict(product.locus_product.state_dict())
    bilinear.network.load_state_dict(product.network.state_dict())

    assert sum(p.numel() for p in product.parameters()) == sum(p.numel() for p in bilinear.parameters())
    generator = torch.Generator().manual_seed(11)
    rna = RNARepresentation(torch.randn(4, rna_dim, generator=generator))
    loci = torch.randn(6, locus_dim, generator=generator)
    with torch.no_grad():
        assert torch.equal(product(rna, loci), bilinear(rna, loci))


# ==========================================================================
# FeatureFusionArchitectureVariantModel -- the shared-backbone suite
# (architecture_novelty_2026_09, retargeted 2026-09-04 after
# FeatureFusionLocusCLSModel became the primary/reference architecture).
#
# This model family has no prior/sigma anchor, so there is no "starts exactly
# at the prior" invariant to test. Its own established convention
# (FeatureFusionLocusCLSModel's fusion_init_std) is a small NONZERO init; the
# exact, testable invariant is fusion_init_std=0.0 -> every prediction is
# exactly 0.5, whatever the encoder/trunk/axial configuration.
# ==========================================================================

FF_INPUT_DIM = 48
FF_CPG_DIM = 32
FF_SAMPLES = 4
FF_LOCI = 6


def _ff_config(**overrides) -> ModelConfig:
    config = ModelConfig(
        encoder=EncoderConfig(latent_dim=16, hidden_dim=24, n_programs=6, program_dim=8, n_heads=2),
        interaction=InteractionConfig(hidden_dim=12),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _ff_batch(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    rna = torch.randn(FF_SAMPLES, FF_INPUT_DIM, generator=generator)
    loci = torch.randn(FF_LOCI, FF_CPG_DIM, generator=generator)
    return rna, loci


def _ff_build(config: ModelConfig, **kwargs) -> FeatureFusionArchitectureVariantModel:
    kwargs.setdefault("trunk_hidden_dim", 20)
    kwargs.setdefault("bottleneck_dim", 10)
    return FeatureFusionArchitectureVariantModel(FF_INPUT_DIM, FF_CPG_DIM, config, **kwargs)


FF_ARM_CONFIGS = {
    "default_equivalent": {},
    "encoder_mlp": {"encoder": EncoderConfig(kind="mlp", latent_dim=16, hidden_dim=24)},
    "encoder_locus_attention": {
        "encoder": EncoderConfig(kind="locus_attention", latent_dim=16, n_programs=6, program_dim=8, n_heads=2)
    },
    "trunk_plain": {"trunk": TrunkConfig(kind="plain", depth=2, width=10)},
    "trunk_hc": {"trunk": TrunkConfig(kind="hc", depth=2, width=10, n_streams=3)},
    "trunk_mhc": {"trunk": TrunkConfig(kind="mhc", depth=2, width=10, n_streams=3)},
    "trunk_mhc_stream_semantics": {"trunk": TrunkConfig(kind="mhc", depth=2, width=10, n_streams=2, stream_semantics=True)},
    "axial": {"trunk": TrunkConfig(kind="plain", depth=1, width=10), "axial": AxialConfig(enabled=True, n_heads=2, window=3)},
    "beta_head": {"beta_likelihood_head": True},
    "no_mean_branch": {},  # exercised via use_mean_branch=False below
}


@pytest.mark.parametrize("name", sorted(FF_ARM_CONFIGS))
def test_ff_arm_with_zero_fusion_init_predicts_exactly_half(name):
    """fusion_init_std=0.0 must zero the final head's weight AND bias, so every
    prediction is exactly sigmoid(0) = 0.5 regardless of encoder/trunk/axial."""
    kwargs = {"fusion_init_std": 0.0}
    if name == "no_mean_branch":
        kwargs["use_mean_branch"] = False
    model = _ff_build(_ff_config(**FF_ARM_CONFIGS[name]), **kwargs).eval()
    rna, loci = _ff_batch()
    with torch.no_grad():
        beta = model(rna, loci)["beta"]
    assert beta.shape == (FF_SAMPLES, FF_LOCI)
    assert torch.allclose(beta, torch.full_like(beta, 0.5), atol=1e-6)


@pytest.mark.parametrize("name", sorted(FF_ARM_CONFIGS))
def test_ff_arm_produces_finite_gradients_everywhere(name):
    kwargs = {}
    if name == "no_mean_branch":
        kwargs["use_mean_branch"] = False
    model = _ff_build(_ff_config(**FF_ARM_CONFIGS[name]), **kwargs).train()
    rna, loci = _ff_batch(seed=1)
    target = torch.rand(FF_SAMPLES, FF_LOCI)
    target[0, 0] = float("nan")
    loss_cfg = LossConfig(locus_pearson_weight=0.15, locus_min_observed_samples=2)
    if FF_ARM_CONFIGS[name].get("beta_likelihood_head") or name == "beta_head":
        loss_cfg.beta_nll_weight = 0.1
    # The real trainer always passes CpG ordinals when the axial arm is active
    # (LocusCLSJointTrainer._position_kwargs); without them axial.distance_scale
    # is a genuinely inert parameter (the decay term is skip-added only when
    # positions are given), so this only matters for the "axial" arm here.
    position_kwargs = {"cpg_positions": torch.arange(FF_LOCI) * 91} if model.requires_cpg_positions else {}
    outputs = model(rna, loci, **position_kwargs)
    from methylation_predictor.rna_training.locus_cls_trainer import _direct_beta_loss
    loss, pieces = _direct_beta_loss(outputs, target, loss_cfg)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, g in grads if g is None]
    # residual_head/mean_head only receive gradient through the auxiliary
    # residual/mean losses, which locus_cls_trainer applies separately
    # (residual_aux_weight/aux_weight) -- neither is exercised by
    # _direct_beta_loss alone, so both are expected to be gradient-free here.
    missing = [n for n in missing if not (n.startswith("residual_head") or n.startswith("mean_head"))]
    assert not missing, f"no gradient reached: {missing}"
    non_finite = [n for n, g in grads if g is not None and not torch.isfinite(g).all()]
    assert not non_finite, f"non-finite gradient: {non_finite}"
    assert pieces["observed"] == FF_SAMPLES * FF_LOCI - 1


def test_ff_default_matches_reference_parameter_shapes():
    config = _ff_config()
    assert not is_architecture_variant(config)
    variant = _ff_build(config)
    reference = FeatureFusionLocusCLSModel(
        FF_INPUT_DIM, FF_CPG_DIM, config, trunk_hidden_dim=20, bottleneck_dim=10,
    )
    assert sorted(tuple(p.shape) for p in variant.parameters()) == sorted(tuple(p.shape) for p in reference.parameters())
    assert sum(p.numel() for p in variant.parameters()) == sum(p.numel() for p in reference.parameters())


def test_ff_default_matches_reference_outputs_under_copied_weights():
    """At every default, FeatureFusionArchitectureVariantModel must reproduce
    FeatureFusionLocusCLSModel's forward computation exactly -- the same
    same-code-control role ArchitectureVariantModel plays for the retired
    two-stage architecture."""
    config = _ff_config()
    variant = _ff_build(config, fusion_init_std=0.05).eval()
    reference = FeatureFusionLocusCLSModel(
        FF_INPUT_DIM, FF_CPG_DIM, config, trunk_hidden_dim=20, bottleneck_dim=10, fusion_init_std=0.05,
    ).eval()

    variant.rna_encoder.load_state_dict(reference.rna_encoder.state_dict())
    variant.rna_product.load_state_dict(reference.rna_product.state_dict())
    variant.locus_product.load_state_dict(reference.locus_product.state_dict())
    variant.raw_branch.load_state_dict(reference.raw_branch.state_dict())
    variant.residual_head.load_state_dict(reference.residual_head.state_dict())
    variant.trunk_cpg.load_state_dict(reference.trunk.state_dict())
    variant.mean_head.load_state_dict(reference.mean_head.state_dict())
    variant.fusion.load_state_dict(reference.fusion.state_dict())

    rna, loci = _ff_batch(seed=2)
    with torch.no_grad():
        assert torch.allclose(variant(rna, loci)["beta"], reference(rna, loci)["beta"], atol=1e-6)


def test_ff_stream_semantics_validation():
    with pytest.raises(ValueError, match="n_streams must be 2"):
        _ff_build(_ff_config(trunk=TrunkConfig(kind="mhc", depth=1, width=10, n_streams=4, stream_semantics=True)))
    with pytest.raises(ValueError, match="both branches"):
        _ff_build(
            _ff_config(trunk=TrunkConfig(kind="mhc", depth=1, width=10, n_streams=2, stream_semantics=True)),
            use_mean_branch=False,
        )
    with pytest.raises(ValueError, match="multi-stream trunk"):
        _ff_build(_ff_config(trunk=TrunkConfig(kind="plain", depth=1, width=10, stream_semantics=True)))


def test_ff_variant_labels_are_distinct():
    plain = feature_fusion_variant_label(_ff_config(trunk=TrunkConfig(kind="plain", depth=3, width=64)))
    semantic = feature_fusion_variant_label(
        _ff_config(trunk=TrunkConfig(kind="mhc", depth=3, width=64, n_streams=2, stream_semantics=True))
    )
    assert plain != semantic
    assert "sem" in semantic and "sem" not in plain
