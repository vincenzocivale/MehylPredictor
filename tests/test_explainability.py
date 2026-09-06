import numpy as np
import torch

from methylation_predictor.config import ModelConfig
from methylation_predictor.explainability.integrated_gradients import GeneAttribution, integrated_gradients_rna
from methylation_predictor.explainability.rna_gene_attribution import SampleGeneExplainer
from methylation_predictor.models import FeatureFusionLocusCLSModel


def _config() -> ModelConfig:
    cfg = ModelConfig()
    cfg.encoder.kind = "linear"
    cfg.encoder.latent_dim = 8
    cfg.encoder.layer_norm = True
    cfg.interaction.hidden_dim = 16
    cfg.interaction.dropout = 0.0
    return cfg


def _model() -> FeatureFusionLocusCLSModel:
    return FeatureFusionLocusCLSModel(
        input_dim=6, cpg_input_dim=5, config=_config(), trunk_hidden_dim=8, bottleneck_dim=4,
    )


def test_zero_residual_head_has_zero_attribution_everywhere():
    # residual_head is the raw branch's own auxiliary probe (see
    # integrated_gradients.py's module docstring) -- zeroing its weight and
    # bias makes residual_logit (and its gradient wrt rna) identically 0 for
    # any input, regardless of the rest of the model's (randomly initialized)
    # weights.
    torch.manual_seed(0)
    model = _model()
    with torch.no_grad():
        model.residual_head.weight.zero_()
        model.residual_head.bias.zero_()
    rna = torch.randn(6)
    loci = torch.randn(4, 5)
    baselines = torch.randn(3, 6)  # real (nonzero) reference vectors, not the all-zero baseline

    attribution = integrated_gradients_rna(model, rna, loci, baselines=baselines, steps=8)

    assert attribution.attributions.shape == (4, 6)
    assert torch.allclose(attribution.residual_logit, torch.zeros(4), atol=1e-6)
    assert torch.allclose(attribution.attributions, torch.zeros_like(attribution.attributions), atol=1e-6)
    assert torch.allclose(attribution.convergence_gap, torch.zeros(4), atol=1e-6)


def test_completeness_axiom_holds_after_perturbing_weights():
    # Expected Gradients' defining guarantee: sum of per-gene attributions ==
    # residual_logit(input) - mean_baseline(residual_logit(baseline)). Perturb
    # residual_head away from zero so the model is not the trivial all-zero
    # function, then check the gap.
    #
    # This also exercises the reason the module uses real (nonzero)
    # baselines rather than an all-zero one: with encoder.layer_norm=True (the
    # reference default), a zero baseline sits exactly on a LayerNorm
    # singularity and this same check does not converge within any practical
    # step count (see integrated_gradients.py's module docstring) -- real
    # baselines converge cleanly with a modest step count.
    torch.manual_seed(1)
    model = _model()
    with torch.no_grad():
        model.residual_head.weight.normal_(std=0.5)
        model.residual_head.bias.normal_(std=0.1)
    rna = torch.randn(6)
    loci = torch.randn(4, 5)
    baselines = torch.randn(5, 6)

    attribution = integrated_gradients_rna(model, rna, loci, baselines=baselines, steps=256)

    scale = attribution.residual_logit.abs().clamp_min(1e-3)
    assert (attribution.convergence_gap / scale < 5e-2).all()


def test_summarize_genes_ranks_by_mean_abs_and_reports_sign_consistency():
    # Build a GeneAttribution by hand (no model needed) to test the pure
    # aggregation logic in isolation.
    attributions = torch.tensor(
        [
            [1.0, -0.1, 0.0],
            [1.2, 0.1, 0.0],
            [0.8, -0.3, 0.0],
        ]
    )
    attribution = GeneAttribution(
        attributions=attributions,
        residual_logit=torch.zeros(3),
        residual_logit_baseline=torch.zeros(3),
        convergence_gap=torch.zeros(3),
    )

    class _Stub(SampleGeneExplainer):
        def __init__(self):
            self.gene_ids = np.array(["GENE_A", "GENE_B", "GENE_C"])

    table = _Stub().summarize_genes(attribution, top_k=3)

    assert table.iloc[0]["gene_id"] == "GENE_A"  # largest mean_abs_attribution
    assert table.iloc[0]["sign_consistency"] == 1.0  # always positive
    assert table.iloc[-1]["gene_id"] == "GENE_C"  # always exactly zero -> smallest
