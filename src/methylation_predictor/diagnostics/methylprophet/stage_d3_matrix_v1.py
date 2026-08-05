#!/usr/bin/env python3
"""Stage D3: matched benchmark of the matrix-factorized rna_branch models against
MethylProphet, extending Stage D1/D2 to the new cacheable-tower training regime.

Reuses Stage D1's verified panel (double-OOD chr1: 414 test patients, 521 test
CpGs), its statistical helpers (alpha calibration, metrics, bootstrap), and its
released-checkpoint verification, unchanged. 3-seed (17/23/41) confirmation of
bilinear (V1) vs concat (F2), the two survivors of the single-seed matched
screen documented in docs/tcga_matrix_model.md "Interaction choice" --
bilinear_concat_residual is dropped here: it showed no validation-MSE
improvement over pure bilinear in that screen (see
memory rna_branch_v1_baseline_and_stage_d3), so it does not earn a 3-seed run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from methylation_predictor.diagnostics.methylprophet.stage_d1 import (
    _assert_panels,
    _released_panel,
    alpha,
    bootstrap,
    calibrated,
    metrics,
    npz,
)


SEEDS = (17, 23, 41)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bilinear-test", nargs=3, type=Path, required=True, help="seed 17 23 41 order")
    p.add_argument("--bilinear-validation", nargs=3, type=Path, required=True)
    p.add_argument("--concat-test", nargs=3, type=Path, required=True, help="seed 17 23 41 order")
    p.add_argument("--concat-validation", nargs=3, type=Path, required=True)
    p.add_argument("--mp-released", type=Path, required=True)
    p.add_argument("--mp-static", type=Path, required=True, help="cpg_idx,pred_methyl; fixed train-mean RNA")
    p.add_argument("--empirical-prior", type=Path, required=True, help="cpg_idx,mean_train,position,within_cpg_variance")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=9176)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"stage_d3_matrix_v1 (3-seed confirmation) start replicates={a.replicates} output={a.output_dir}", file=sys.stderr, flush=True)

    bilinear_test = [npz(x) for x in a.bilinear_test]
    bilinear_val = [npz(x) for x in a.bilinear_validation]
    concat_test = [npz(x) for x in a.concat_test]
    concat_val = [npz(x) for x in a.concat_validation]

    for i, item in enumerate(bilinear_test[1:], 1):
        _assert_panels(bilinear_test[0], item, f"bilinear test seed {SEEDS[i]}")
    for i, item in enumerate(concat_test):
        _assert_panels(bilinear_test[0], item, f"concat test seed {SEEDS[i]}")
    for i, item in enumerate(bilinear_val[1:], 1):
        _assert_panels(bilinear_val[0], item, f"bilinear validation seed {SEEDS[i]}")
    for i, item in enumerate(concat_val):
        _assert_panels(bilinear_val[0], item, f"concat validation seed {SEEDS[i]}")

    target, cancer = bilinear_test[0]["target"].astype(float), bilinear_test[0]["cancer_type"].astype(str)
    samples, cpgs = bilinear_test[0]["sample_idx"].astype(str), bilinear_test[0]["cpg_idx"].astype(str)
    prior_nt = np.broadcast_to(bilinear_test[0]["prior"].astype(float)[None, :], target.shape)

    val_target = bilinear_val[0]["target"].astype(float)
    nt_val = np.broadcast_to(bilinear_val[0]["prior"].astype(float)[None, :], val_target.shape)

    mp_pred = _released_panel(a.mp_released, samples, cpgs, cancer, target)
    mp_val = _released_panel(a.mp_released, bilinear_val[0]["sample_idx"].astype(str), cpgs, bilinear_val[0]["cancer_type"].astype(str), val_target)

    mp_static = pd.read_parquet(a.mp_static)
    mp_static.cpg_idx = mp_static.cpg_idx.astype(str)
    mp_mu = mp_static.set_index("cpg_idx").loc[cpgs, "pred_methyl"].to_numpy(float)
    if not np.isfinite(mp_mu).all():
        raise ValueError("MethylProphet static prior does not cover the test CpGs")
    mp_static_test = np.broadcast_to(mp_mu[None, :], target.shape)
    mp_static_val = np.broadcast_to(mp_mu[None, :], val_target.shape)
    # alpha is fit against MP's own static prior (its dynamic component's actual
    # anchor), then re-applied against a different static anchor (NTv3) in
    # calibrated() below -- matches stage_d1.py's convention exactly.
    alpha_mp = alpha(val_target, mp_val, mp_static_val)

    empirical = pd.read_parquet(a.empirical_prior)
    empirical.cpg_idx = empirical.cpg_idx.astype(str)
    empirical_mu = empirical.set_index("cpg_idx").loc[cpgs, "mean_train"].to_numpy(float)
    if not np.isfinite(empirical_mu).all():
        raise ValueError("empirical prior does not cover the test CpGs")
    empirical_static = np.broadcast_to(empirical_mu[None, :], target.shape)

    bilinear_test_pred = [item["prediction"].astype(float) for item in bilinear_test]
    concat_test_pred = [item["prediction"].astype(float) for item in concat_test]
    alpha_bilinear = [alpha(val_target, item["prediction"].astype(float), nt_val) for item in bilinear_val]
    alpha_concat = [alpha(val_target, item["prediction"].astype(float), nt_val) for item in concat_val]
    bilinear_ensemble = np.mean(bilinear_test_pred, axis=0)
    concat_ensemble = np.mean(concat_test_pred, axis=0)
    alpha_bilinear_ensemble = alpha(val_target, np.mean([x["prediction"].astype(float) for x in bilinear_val], axis=0), nt_val)
    alpha_concat_ensemble = alpha(val_target, np.mean([x["prediction"].astype(float) for x in concat_val], axis=0), nt_val)

    models = {
        "empirical_train_only_prior": empirical_static,
        "ntv3_prior": prior_nt,
        "methylprophet_static_prior": mp_static_test,
        "methylprophet_original": mp_pred,
        "mp_prior_plus_calibrated_mp_dynamic": calibrated(mp_static_test, mp_pred, alpha_mp),
        "ntv3_prior_plus_calibrated_mp_dynamic": calibrated(prior_nt, mp_pred, alpha_mp),
        "bilinear_v1_beta_ensemble": bilinear_ensemble,
        "ntv3_prior_plus_calibrated_bilinear_v1_dynamic_ensemble": calibrated(prior_nt, bilinear_ensemble, alpha_bilinear_ensemble),
        "concat_beta_ensemble": concat_ensemble,
        "ntv3_prior_plus_calibrated_concat_dynamic_ensemble": calibrated(prior_nt, concat_ensemble, alpha_concat_ensemble),
    }
    for seed, pred in zip(SEEDS, bilinear_test_pred):
        models[f"bilinear_v1_seed{seed}"] = pred
    for seed, pred in zip(SEEDS, concat_test_pred):
        models[f"concat_seed{seed}"] = pred

    point = {name: metrics(target, pred, prior_nt[0], cancer) for name, pred in models.items()}

    pos = empirical.set_index("cpg_idx").loc[cpgs, "position"].to_numpy(int)
    labels = pos // 5_000_000
    blocks = [np.flatnonzero(labels == label) for label in np.unique(labels)]

    comparisons = {
        "concat_vs_bilinear_v1_mean_seed_metrics": bootstrap(target, prior_nt[0], cancer, blocks, concat_test_pred, bilinear_ensemble, a.replicates, a.seed + 200),
        "concat_vs_bilinear_v1_beta_ensemble": bootstrap(target, prior_nt[0], cancer, blocks, [concat_ensemble], bilinear_ensemble, a.replicates, a.seed + 201),
        "concat_vs_mp_mean_seed_metrics": bootstrap(target, prior_nt[0], cancer, blocks, concat_test_pred, mp_pred, a.replicates, a.seed + 202),
        "concat_vs_mp_beta_ensemble": bootstrap(target, prior_nt[0], cancer, blocks, [concat_ensemble], mp_pred, a.replicates, a.seed + 203),
        "bilinear_v1_vs_mp_mean_seed_metrics": bootstrap(target, prior_nt[0], cancer, blocks, bilinear_test_pred, mp_pred, a.replicates, a.seed + 204),
        "bilinear_v1_vs_mp_beta_ensemble": bootstrap(target, prior_nt[0], cancer, blocks, [bilinear_ensemble], mp_pred, a.replicates, a.seed + 205),
    }

    variance = empirical.set_index("cpg_idx").loc[cpgs, "within_cpg_variance"].to_numpy(float)
    tertiles = np.empty(len(cpgs), int)
    order = np.argsort(variance)
    for label, idx in enumerate(np.array_split(order, 3)):
        tertiles[idx] = label
    tertile_metrics = {
        name: {f"tertile_{t + 1}": metrics(target[:, tertiles == t], pred[:, tertiles == t], prior_nt[0, tertiles == t], cancer) for t in range(3)}
        for name, pred in models.items()
    }

    result = {
        "claim_scope": "Stage D3 3-seed confirmation: bilinear (V1) vs concat (F2, Stage F/G winner) at matrix scale vs. MethylProphet released checkpoint. bilinear_concat_residual dropped after the single-seed screen (no validation improvement).",
        "contract": {
            "patients": len(samples), "cpgs": len(cpgs), "observed_rows": int(np.isfinite(target).sum()),
            "bootstrap_blocks": len(blocks), "block_size_bp": 5_000_000,
            "seeds": list(SEEDS),
            "calibration": "all alpha values fitted only on validation sample x test-CpG rows",
        },
        "calibration_alpha": {
            "methylprophet_dynamic": alpha_mp,
            **{f"bilinear_v1_seed{s}": x for s, x in zip(SEEDS, alpha_bilinear)},
            "bilinear_v1_ensemble": alpha_bilinear_ensemble,
            **{f"concat_seed{s}": x for s, x in zip(SEEDS, alpha_concat)},
            "concat_ensemble": alpha_concat_ensemble,
        },
        "metrics": point,
        "variability_tertiles": tertile_metrics,
        "paired_bootstrap": comparisons,
        "conventions": {
            "delta_mse": "candidate minus reference; negative favours candidate",
            "delta_skill": "candidate minus reference; positive favours candidate",
            "delta_within_cancer_skill": "candidate minus reference; positive favours candidate",
            "inference": "hierarchical patient x 5-Mb block CI is primary",
            "primary_metric": "beta-space MSE on identical observations; do not compare skill_vs_prior across models anchored to different priors (methylprophet_static_prior vs ntv3_prior)",
        },
    }
    (a.output_dir / "stage_d3_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(
        [{"model": name, **{key: value for key, value in metric.items() if key != "per_cancer"}} for name, metric in point.items()]
    ).to_csv(a.output_dir / "stage_d3_metrics.csv", index=False)
    print("stage_d3_matrix_v1 complete", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
