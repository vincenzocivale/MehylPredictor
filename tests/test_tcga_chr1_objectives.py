import torch

from methylation_predictor.config import LossConfig
from methylation_predictor.benchmark.table5.trainer import loss_config_for_source
from methylation_predictor.losses import masked_locus_pearson


def test_pearson_target_std_floor_excludes_near_constant_locus():
    target = torch.tensor([
        [0.500, 0.10],
        [0.501, 0.30],
        [0.499, 0.70],
        [0.500, 0.90],
    ])
    prediction = target.clone()
    mask = torch.ones_like(target, dtype=torch.bool)

    corr, valid = masked_locus_pearson(
        prediction, target, mask,
        min_observed_samples=4,
        epsilon=1e-8,
        min_target_std=0.02,
    )

    assert torch.isfinite(corr).all()
    assert valid.tolist() == [False, True]


def test_source_scoping_disables_only_structured_losses():
    cfg = LossConfig(
        beta_mse_weight=1.0,
        standardized_residual_huber_weight=0.1,
        standardized_shrinkage_weight=1e-4,
        locus_pearson_weight=0.15,
        locus_lower_tail_weight=0.05,
        pairwise_difference_weight=0.02,
        centered_mse_weight=0.03,
        amplitude_weight=0.04,
    )

    array_cfg = loss_config_for_source(cfg, "array", {"array"})
    wgbs_cfg = loss_config_for_source(cfg, "wgbs", {"array"})

    assert array_cfg is cfg
    assert wgbs_cfg.beta_mse_weight == 1.0
    assert wgbs_cfg.standardized_residual_huber_weight == 0.1
    assert wgbs_cfg.standardized_shrinkage_weight == 1e-4
    assert wgbs_cfg.locus_pearson_weight == 0.0
    assert wgbs_cfg.locus_lower_tail_weight == 0.0
    assert wgbs_cfg.pairwise_difference_weight == 0.0
    assert wgbs_cfg.centered_mse_weight == 0.0
    assert wgbs_cfg.amplitude_weight == 0.0
