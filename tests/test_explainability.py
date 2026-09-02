import numpy as np
import torch

from methylation_predictor.config import ModelConfig
from methylation_predictor.explainability.integrated_gradients import GeneAttribution, integrated_gradients_rna
from methylation_predictor.explainability.rna_gene_attribution import SampleGeneExplainer
from methylation_predictor.models import VarianceNormalizedResidualModel


def _config() -> ModelConfig:
    cfg = ModelConfig()
    cfg.encoder.kind = "linear"
    cfg.encoder.latent_dim = 8
    cfg.encoder.layer_norm = True
    cfg.interaction.kind = "concat"
    cfg.interaction.hidden_dim = 16
    cfg.interaction.dropout = 0.0
    cfg.zero_init_residual = True
    return cfg


def test_zero_init_model_has_zero_attribution_everywhere():
    # zero_init_residual zeros the interaction's final layer, so raw_delta (and
    # its gradient wrt rna) is identically 0 for any input at construction time.
    torch.manual_seed(0)
    model = VarianceNormalizedResidualModel(input_dim=6, locus_dim=5, config=_config())
    rna = torch.randn(6)
    loci = torch.randn(4, 5)
    baselines = torch.randn(3, 6)  # real (nonzero) reference vectors, not the all-zero baseline

    attribution = integrated_gradients_rna(model, rna, loci, baselines=baselines, steps=8)

    assert attribution.attributions.shape == (4, 6)
    assert torch.allclose(attribution.raw_delta, torch.zeros(4), atol=1e-6)
    assert torch.allclose(attribution.attributions, torch.zeros_like(attribution.attributions), atol=1e-6)
    assert torch.allclose(attribution.convergence_gap, torch.zeros(4), atol=1e-6)


def test_completeness_axiom_holds_after_perturbing_weights():
    # Expected Gradients' defining guarantee: sum of per-gene attributions ==
    # raw_delta(input) - mean_baseline(raw_delta(baseline)). Perturb the
    # (otherwise zero-initialized) final layer so the model is not the
    # trivial all-zero function, then check the gap.
    #
    # This also exercises the reason the module uses real (nonzero)
    # baselines rather than an all-zero one: with encoder.layer_norm=True (the
    # canonical default), a zero baseline sits exactly on a LayerNorm
    # singularity and this same check does not converge within any practical
    # step count (see integrated_gradients.py's module docstring) -- real
    # baselines converge cleanly with a modest step count.
    torch.manual_seed(1)
    model = VarianceNormalizedResidualModel(input_dim=6, locus_dim=5, config=_config())
    with torch.no_grad():
        model.interaction.network[-1].weight.normal_(std=0.5)
        model.interaction.network[-1].bias.normal_(std=0.1)
    rna = torch.randn(6)
    loci = torch.randn(4, 5)
    baselines = torch.randn(5, 6)

    attribution = integrated_gradients_rna(model, rna, loci, baselines=baselines, steps=256)

    scale = attribution.raw_delta.abs().clamp_min(1e-3)
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
        raw_delta=torch.zeros(3),
        raw_delta_baseline=torch.zeros(3),
        convergence_gap=torch.zeros(3),
    )

    class _Stub(SampleGeneExplainer):
        def __init__(self):
            self.gene_ids = np.array(["GENE_A", "GENE_B", "GENE_C"])

    table = _Stub().summarize_genes(attribution, top_k=3)

    assert table.iloc[0]["gene_id"] == "GENE_A"  # largest mean_abs_attribution
    assert table.iloc[0]["sign_consistency"] == 1.0  # always positive
    assert table.iloc[-1]["gene_id"] == "GENE_C"  # always exactly zero -> smallest
