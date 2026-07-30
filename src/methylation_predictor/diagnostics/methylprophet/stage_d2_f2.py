#!/usr/bin/env python3
"""Stage D2: matched benchmark extending Stage D1 with the Stage-F/G winner F2.

Reuses Stage D1's verified panel (double-OOD chr1: 414 test patients, 521 test
CpGs), its statistical helpers (alpha calibration, quick_metrics, bootstrap),
and its released-checkpoint verification, unchanged. Only adds F2 (seeds
17/23/41, from the RNA-branch overlay's Stage F confirm grid) to the
comparison and computes new paired bootstraps: F2 ensemble vs. MethylProphet
original, and F2 ensemble vs. C0 ensemble.
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c0-test", nargs=3, type=Path, required=True)
    p.add_argument("--c0-validation", nargs=3, type=Path, required=True)
    p.add_argument("--f2-test", nargs=3, type=Path, required=True)
    p.add_argument("--f2-validation", nargs=3, type=Path, required=True)
    p.add_argument("--mp-released", type=Path, required=True)
    p.add_argument("--mp-static", type=Path, required=True, help="cpg_idx,pred_methyl; fixed train-mean RNA")
    p.add_argument("--empirical-prior", type=Path, required=True, help="cpg_idx,mean_train,position,within_cpg_variance")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=9176)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"stage_d2_f2 start replicates={a.replicates} output={a.output_dir}", file=sys.stderr, flush=True)

    c0_test = [npz(x) for x in a.c0_test]
    c0_val = [npz(x) for x in a.c0_validation]
    f2_test = [npz(x) for x in a.f2_test]
    f2_val = [npz(x) for x in a.f2_validation]

    for i, item in enumerate(c0_test[1:], 1):
        _assert_panels(c0_test[0], item, f"C0 test seed {i}")
    for item in f2_test:
        _assert_panels(c0_test[0], item, "F2 test seed")
    for item in c0_val[1:]:
        _assert_panels(c0_val[0], item, "C0 validation seed")
    for item in f2_val:
        _assert_panels(c0_val[0], item, "F2 validation seed")

    target, cancer = c0_test[0]["target"].astype(float), c0_test[0]["cancer_type"].astype(str)
    samples, cpgs = c0_test[0]["sample_idx"].astype(str), c0_test[0]["cpg_idx"].astype(str)
    prior_nt = np.broadcast_to(c0_test[0]["prior"].astype(float)[None, :], target.shape)

    val_target = c0_val[0]["target"].astype(float)
    nt_val = np.broadcast_to(c0_val[0]["prior"].astype(float)[None, :], val_target.shape)

    mp_pred = _released_panel(a.mp_released, samples, cpgs, cancer, target)
    mp_val = _released_panel(a.mp_released, c0_val[0]["sample_idx"].astype(str), cpgs, c0_val[0]["cancer_type"].astype(str), val_target)

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

    c0_test_pred = [item["prediction"].astype(float) for item in c0_test]
    f2_test_pred = [item["prediction"].astype(float) for item in f2_test]
    alpha_c0 = [alpha(val_target, item["prediction"].astype(float), nt_val) for item in c0_val]
    alpha_f2 = [alpha(val_target, item["prediction"].astype(float), nt_val) for item in f2_val]
    c0_ensemble = np.mean(c0_test_pred, axis=0)
    f2_ensemble = np.mean(f2_test_pred, axis=0)
    alpha_c0_ensemble = alpha(val_target, np.mean([x["prediction"].astype(float) for x in c0_val], axis=0), nt_val)
    alpha_f2_ensemble = alpha(val_target, np.mean([x["prediction"].astype(float) for x in f2_val], axis=0), nt_val)

    def blend_beta(a_pred, b_pred):
        return 0.5 * a_pred + 0.5 * b_pred

    models = {
        "empirical_train_only_prior": empirical_static,
        "ntv3_prior": prior_nt,
        "methylprophet_static_prior": mp_static_test,
        "methylprophet_original": mp_pred,
        "mp_prior_plus_calibrated_mp_dynamic": calibrated(mp_static_test, mp_pred, alpha_mp),
        "ntv3_prior_plus_calibrated_mp_dynamic": calibrated(prior_nt, mp_pred, alpha_mp),
        "c0_v0_beta_ensemble": c0_ensemble,
        "ntv3_prior_plus_calibrated_c0_dynamic_ensemble": calibrated(prior_nt, c0_ensemble, alpha_c0_ensemble),
        "f2_beta_ensemble": f2_ensemble,
        "ntv3_prior_plus_calibrated_f2_dynamic_ensemble": calibrated(prior_nt, f2_ensemble, alpha_f2_ensemble),
        "blend_c0_mp_diagnostic_only": blend_beta(c0_ensemble, mp_pred),
        "blend_f2_mp_diagnostic_only": blend_beta(f2_ensemble, mp_pred),
    }
    for seed, pred in zip((17, 29, 43), c0_test_pred):
        models[f"c0_v0_seed{seed}"] = pred
    for seed, pred in zip((17, 23, 41), f2_test_pred):
        models[f"f2_seed{seed}"] = pred

    point = {name: metrics(target, pred, prior_nt[0], cancer) for name, pred in models.items()}

    pos = empirical.set_index("cpg_idx").loc[cpgs, "position"].to_numpy(int)
    labels = pos // 5_000_000
    blocks = [np.flatnonzero(labels == label) for label in np.unique(labels)]

    comparisons = {
        "f2_vs_mp_mean_seed_metrics": bootstrap(target, prior_nt[0], cancer, blocks, f2_test_pred, mp_pred, a.replicates, a.seed + 200),
        "f2_vs_mp_beta_ensemble": bootstrap(target, prior_nt[0], cancer, blocks, [f2_ensemble], mp_pred, a.replicates, a.seed + 201),
        "f2_vs_c0_mean_seed_metrics": bootstrap(target, prior_nt[0], cancer, blocks, f2_test_pred, c0_ensemble, a.replicates, a.seed + 202),
        "f2_vs_c0_beta_ensemble": bootstrap(target, prior_nt[0], cancer, blocks, [f2_ensemble], c0_ensemble, a.replicates, a.seed + 203),
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
        "claim_scope": "Stage D2: extends the Stage D1 released-checkpoint benchmark with F2 (Stage F/G winner); not a matched-training architecture claim",
        "contract": {
            "patients": len(samples), "cpgs": len(cpgs), "observed_rows": int(np.isfinite(target).sum()),
            "bootstrap_blocks": len(blocks), "block_size_bp": 5_000_000,
            "calibration": "all alpha values fitted only on validation sample x test-CpG rows",
        },
        "calibration_alpha": {
            "methylprophet_dynamic": alpha_mp,
            **{f"c0_seed{s}": x for s, x in zip((17, 29, 43), alpha_c0)},
            "c0_ensemble": alpha_c0_ensemble,
            **{f"f2_seed{s}": x for s, x in zip((17, 23, 41), alpha_f2)},
            "f2_ensemble": alpha_f2_ensemble,
        },
        "metrics": point,
        "variability_tertiles": tertile_metrics,
        "paired_bootstrap": comparisons,
        "conventions": {
            "delta_mse": "candidate minus reference; negative favours candidate",
            "delta_skill": "candidate minus reference; positive favours candidate",
            "delta_within_cancer_skill": "candidate minus reference; positive favours candidate",
            "inference": "hierarchical patient x 5-Mb block CI is primary",
        },
    }
    (a.output_dir / "stage_d2_f2_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(
        [{"model": name, **{key: value for key, value in metric.items() if key != "per_cancer"}} for name, metric in point.items()]
    ).to_csv(a.output_dir / "stage_d2_f2_metrics.csv", index=False)
    print("stage_d2_f2 complete", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
