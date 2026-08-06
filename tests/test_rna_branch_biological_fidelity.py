from __future__ import annotations

import numpy as np
import torch

from methylation_predictor.rna_branch.biological_metrics import (
    BiologicalMetricConfig,
    differential_effect_recovery,
    per_locus_biological_metrics,
    per_sample_biological_metrics,
    regional_effect_recovery,
    structural_fidelity,
    summarize_biological_fidelity,
)
from methylation_predictor.rna_branch.config import LossConfig
from methylation_predictor.rna_branch.losses import (
    locus_ccc_loss,
    locus_skill_loss,
    residual_loss,
    scheduled_loss_config,
    within_cancer_dynamic_loss,
)


def _matrix() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.tensor(
        [
            [0.10, 0.80, 0.20],
            [0.20, 0.70, 0.30],
            [0.40, 0.50, 0.60],
            [0.70, 0.30, 0.80],
            [0.80, 0.20, 0.90],
            [0.90, 0.10, 0.70],
            [0.30, 0.60, 0.40],
            [0.60, 0.40, 0.50],
        ]
    )
    prior = torch.tensor([0.50, 0.50, 0.50])
    mask = torch.ones_like(target, dtype=torch.bool)
    codes = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return target, prior, mask, codes


def test_locus_skill_and_ccc_reject_scale_collapsed_solution() -> None:
    target, prior, mask, _ = _matrix()
    centred = target - target.mean(dim=0, keepdim=True)
    collapsed = target.mean(dim=0, keepdim=True) + 0.10 * centred
    config = LossConfig(
        locus_min_observed_samples=4,
        locus_min_target_std=0.01,
        locus_skill_loss_clip=10.0,
    )
    exact_skill, _ = locus_skill_loss(target, target, prior, mask, config)
    collapsed_skill, _ = locus_skill_loss(collapsed, target, prior, mask, config)
    exact_ccc, _ = locus_ccc_loss(target, target, mask, config)
    collapsed_ccc, _ = locus_ccc_loss(collapsed, target, mask, config)
    assert exact_skill.item() == 0.0
    assert exact_ccc.item() < 1e-6
    assert collapsed_skill.item() > exact_skill.item()
    assert collapsed_ccc.item() > 0.5


def test_within_cancer_dynamic_loss_backpropagates() -> None:
    target, _, mask, codes = _matrix()
    prediction = (target + 0.05 * torch.randn_like(target)).requires_grad_(True)
    config = LossConfig(
        locus_min_observed_samples=6,
        locus_min_target_std=0.01,
        within_cancer_min_samples=4,
    )
    loss, loci, groups = within_cancer_dynamic_loss(
        prediction, target, mask, codes, config
    )
    loss.backward()
    assert loci == 3
    assert groups == 2
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_combined_objective_exposes_all_quantitative_terms() -> None:
    target, prior, _, codes = _matrix()
    prediction = (target * 0.9 + 0.05).detach().requires_grad_(True)
    delta = torch.logit(prediction.clamp(1e-4, 1 - 1e-4)) - torch.logit(prior).unsqueeze(0)
    config = LossConfig(
        beta_mse_weight=0.5,
        global_prior_ratio_weight=0.5,
        residual_huber_weight=0.0,
        shrinkage_weight=0.0,
        locus_skill_weight=1.0,
        locus_ccc_weight=0.25,
        within_cancer_dynamic_weight=0.5,
        locus_pearson_weight=0.05,
        locus_min_observed_samples=6,
        locus_min_target_std=0.01,
        within_cancer_min_samples=4,
    )
    loss, pieces = residual_loss(
        {"beta": prediction, "delta_logit": delta},
        target,
        prior,
        config,
        cancer_type_codes=codes,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert pieces["global_prior_ratio_loss"] < 1.0
    assert pieces["valid_skill_loci"] == 3
    assert pieces["valid_ccc_loci"] == 3
    assert pieces["valid_within_cancer_loci"] == 3


def test_structured_objectives_ramp_after_warmup() -> None:
    config = LossConfig(
        locus_skill_weight=1.0,
        locus_ccc_weight=0.25,
        within_cancer_dynamic_weight=0.5,
        objective_warmup_epochs=2,
        objective_ramp_epochs=4,
    )
    assert scheduled_loss_config(config, 2).locus_skill_weight == 0.0
    epoch3 = scheduled_loss_config(config, 3)
    assert epoch3.locus_skill_weight == 0.25
    assert epoch3.beta_mse_weight == config.warmup_beta_mse_weight + 0.25 * (
        config.beta_mse_weight - config.warmup_beta_mse_weight
    )
    assert scheduled_loss_config(config, 4).locus_ccc_weight == 0.125
    assert scheduled_loss_config(config, 6).within_cancer_dynamic_weight == 0.5


def test_biological_summary_distinguishes_direction_from_concordance() -> None:
    target_t, prior_t, _, codes_t = _matrix()
    target = target_t.numpy()
    prior = prior_t.numpy()
    codes = np.array(["A"] * 4 + ["B"] * 4)
    centred = target - target.mean(axis=0, keepdims=True)
    exact = target.copy()
    collapsed = target.mean(axis=0, keepdims=True) + 0.10 * centred
    config = BiologicalMetricConfig(min_observed_samples=6, min_target_std=0.01, max_structure_cpgs=3)

    exact_locus = per_locus_biological_metrics(target, exact, prior, codes, config=config)
    collapsed_locus = per_locus_biological_metrics(target, collapsed, prior, codes, config=config)
    exact_summary = summarize_biological_fidelity(
        exact_locus, per_sample_biological_metrics(target, exact, prior, cancer_types=codes)
    )
    collapsed_summary = summarize_biological_fidelity(
        collapsed_locus, per_sample_biological_metrics(target, collapsed, prior, cancer_types=codes)
    )
    assert collapsed_summary["mas_pcc_variable"] > 0.999
    assert exact_summary["mas_ccc_variable"] > collapsed_summary["mas_ccc_variable"]
    assert exact_summary["mas_dynamic_r2_variable"] > collapsed_summary["mas_dynamic_r2_variable"]
    assert collapsed_summary["median_amplitude_ratio_variable"] < 0.11


def test_structural_and_differential_metrics_are_perfect_for_exact_copy() -> None:
    target_t, _, _, _ = _matrix()
    target = target_t.numpy()
    cancer = np.array(["A"] * 4 + ["B"] * 4)
    config = BiologicalMetricConfig(
        min_observed_samples=6,
        min_target_std=0.01,
        min_cancer_group_samples=4,
        max_structure_cpgs=3,
        knn_k=2,
    )
    structure = structural_fidelity(target, target.copy(), config=config)
    effects = differential_effect_recovery(target, target.copy(), cancer, config=config)
    assert structure["comethylation_edge_pearson"] > 0.999
    assert structure["patient_distance_pearson"] > 0.999
    assert structure["patient_knn_overlap"] == 1.0
    assert effects["aggregate"]["macro_effect_pearson"] > 0.999


def test_locus_eligibility_is_frozen_from_training_reference() -> None:
    target_t, prior_t, _, _ = _matrix()
    target = target_t.numpy()
    prior = prior_t.numpy()
    prediction = target.copy()
    cancer = np.array(["A"] * 4 + ["B"] * 4)
    eligibility_target = target.copy()
    eligibility_target[:, 1] = 0.5  # constant in training: must remain ineligible on test
    config = BiologicalMetricConfig(min_observed_samples=6, min_target_std=0.01)

    locus = per_locus_biological_metrics(
        target,
        prediction,
        prior,
        cancer,
        eligibility_target=eligibility_target,
        config=config,
    )
    assert bool(locus.loc[0, "eligible_variable_cpg"])
    assert not bool(locus.loc[1, "eligible_variable_cpg"])
    assert locus.loc[1, "target_std"] > config.min_target_std
    assert np.isclose(locus.loc[1, "eligibility_target_std"], 0.0)


def test_regional_dmr_effect_recovery_is_perfect_for_exact_copy() -> None:
    target_t, _, _, _ = _matrix()
    target = target_t.numpy()
    cancer = np.array(["A"] * 4 + ["B"] * 4)
    regions = np.array(["R1", "R1", "R2"])
    result = regional_effect_recovery(
        target, target.copy(), cancer, regions, min_cpgs_per_region=1,
        config=BiologicalMetricConfig(min_cancer_group_samples=4),
    )
    assert result["regions_evaluated"] == 2
    assert result["aggregate"]["macro_effect_pearson"] > 0.999
