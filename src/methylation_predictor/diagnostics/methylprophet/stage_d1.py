#!/usr/bin/env python3
"""Locked Stage-D1 benchmark: matched MethylProphet versus C0 on chr1.

This intentionally evaluates the released checkpoint, not a retrained
MethylProphet.  It therefore labels its conclusion as a checkpoint benchmark.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from methylation_predictor.rna_branch.metrics import evaluate_predictions


EPS = 1e-4


def logit(x):
    x = np.clip(np.asarray(x, float), EPS, 1 - EPS)
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def npz(path: Path):
    x = np.load(path)
    return {key: x[key] for key in x.files}


def _assert_panels(reference, other, label):
    for key in ("sample_idx", "cpg_idx", "cancer_type"):
        if not np.array_equal(reference[key].astype(str), other[key].astype(str)):
            raise ValueError(f"{label}: {key} differs from C0 seed 17")
    mask = np.isfinite(reference["target"])
    if not np.array_equal(mask, np.isfinite(other["target"])) or not np.allclose(reference["target"][mask], other["target"][mask], rtol=0, atol=1e-6):
        raise ValueError(f"{label}: target or observed-mask differs from C0 seed 17")


def _released_panel(source, samples, cpgs, cancer, expected_target, group=2):
    table = ds.dataset(source, format="parquet").to_table(
        filter=(ds.field("group_idx") == group) & ds.field("sample_idx").isin([int(x) for x in samples]) & ds.field("cpg_idx").isin([int(x) for x in cpgs]),
        columns=["sample_idx", "cpg_idx", "pred_methyl", "gt_methyl"],
    ).to_pandas()
    table.sample_idx = table.sample_idx.astype(str); table.cpg_idx = table.cpg_idx.astype(str)
    index = pd.MultiIndex.from_product([samples.astype(str), cpgs.astype(str)], names=["sample_idx", "cpg_idx"])
    table = table.set_index(["sample_idx", "cpg_idx"]).reindex(index)
    pred, target = table.pred_methyl.to_numpy(float).reshape(expected_target.shape), table.gt_methyl.to_numpy(float).reshape(expected_target.shape)
    mask = np.isfinite(expected_target)
    if not np.isfinite(pred[mask]).all() or not np.isfinite(target[mask]).all() or not np.allclose(target[mask], expected_target[mask], rtol=0, atol=1e-6):
        raise ValueError("released checkpoint rows do not exactly match C0 targets")
    return pred


def alpha(target, prediction, static):
    mask = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(static)
    d = logit(prediction[mask]) - logit(static[mask]); y = logit(target[mask]) - logit(static[mask])
    return float(np.dot(d, y) / np.dot(d, d))


def calibrated(static, dynamic_prediction, a):
    return sigmoid(logit(static) + a * (logit(dynamic_prediction) - logit(static)))


def metrics(target, prediction, prior, cancer):
    return evaluate_predictions(target, prediction, prior, cancer)


def quick_metrics(target, prediction, prior, cancer):
    """Only the three bootstrap endpoints; avoids correlation/per-cancer work.

    (A sort+reduceat vectorization of the within-cancer loop below was tried
    and measured SLOWER than this loop at real genome-wide panel scale
    -- 4.16s vs 2.58s at 812x81493 -- so it was reverted; the loop's actual
    cost is dominated by array-copy volume proportional to n_samples x
    n_cpgs regardless of how it's chunked, and the vectorized version added
    several full-array-sized intermediate allocations (sorted copies,
    np.where materializations, the final np.repeat) that outweighed the
    saved Python-loop overhead. The real, measured win was eliminating
    `delta()`'s redundant repeated `target[np.ix_(rows,cols)]` slicing
    across the reference/candidate quick_metrics calls -- see `bootstrap()`.)
    """
    valid = np.isfinite(target) & np.isfinite(prediction)
    error = float(np.mean((target[valid] - prediction[valid]) ** 2))
    base = np.broadcast_to(prior[None, :], target.shape)
    prior_error = float(np.mean((target[valid] - base[valid]) ** 2))
    within_ss = 0.0; within_error = 0.0
    for label in np.unique(cancer):
        rows = cancer == label; m = valid[rows]
        t, q = target[rows], prediction[rows]
        count = m.sum(axis=0, keepdims=True)
        mt = np.divide(np.where(m, t, 0).sum(axis=0, keepdims=True), count, out=np.zeros_like(count, dtype=float), where=count > 0)
        mq = np.divide(np.where(m, q, 0).sum(axis=0, keepdims=True), count, out=np.zeros_like(count, dtype=float), where=count > 0)
        within_ss += float(np.where(m, (t - mt) ** 2, 0).sum())
        within_error += float(np.where(m, ((t - mt) - (q - mq)) ** 2, 0).sum())
    return {"mse": error, "skill_vs_prior": 1 - error / prior_error,
            "within_cancer_skill": 1 - within_error / within_ss}


def genomic_blocks(chromosome, position, block_bp: int = 5_000_000):
    """Chromosome-aware N-bp genomic bins for the bootstrap's "genomic_blocks"
    resampling unit. `position` alone is chromosome-LOCAL (see
    genomic_encoder/build_genome_wide_targets.py, which renames raw per-
    chromosome chr/pos without computing a genome-wide cumulative
    coordinate) -- combining with `chromosome` is required once more than one
    chromosome is in play, or two CpGs on different chromosomes at the same
    local position silently collide into the same block. No-op for chr1-only
    callers (this repo's existing usages of `pos // block_bp` alone), since a
    single chromosome can't collide with itself."""
    chromosome = np.asarray(chromosome).astype(str)
    label = np.char.add(np.char.add(chromosome, "_"), (np.asarray(position, dtype=np.int64) // block_bp).astype(str))
    _, inverse = np.unique(label, return_inverse=True)
    return [np.flatnonzero(inverse == group) for group in range(inverse.max() + 1)]


def bootstrap(target, prior, cancer, blocks, candidate_predictions, reference, replicates, seed, include_draws=False):
    """Bootstrap a (possibly multi-seed) candidate; aggregate seeds at metric level."""
    rng = np.random.default_rng(seed)
    def delta(rows, cols):
        # target/prior/cancer don't depend on which prediction (reference vs.
        # candidate) is being scored -- slice them once instead of once per
        # quick_metrics call (was a redundant full fancy-index copy of the
        # panel, e.g. ~530MB at genome-wide official-val scale, every time).
        idx = np.ix_(rows, cols)
        target_sub = target[idx]
        prior_sub = prior[cols]
        cancer_sub = cancer[rows]
        r = quick_metrics(target_sub, reference[idx], prior_sub, cancer_sub)
        cs = [quick_metrics(target_sub, p[idx], prior_sub, cancer_sub) for p in candidate_predictions]
        return {"delta_mse": float(np.mean([c["mse"] for c in cs]) - r["mse"]),
                "delta_skill": float(np.mean([c["skill_vs_prior"] for c in cs]) - r["skill_vs_prior"]),
                "delta_within_cancer_skill": float(np.mean([c["within_cancer_skill"] for c in cs]) - r["within_cancer_skill"])}
    point = delta(np.arange(target.shape[0]), np.arange(target.shape[1]))
    out = {"point": point}
    for mode in ("patients", "genomic_blocks", "hierarchical"):
        values = {key: [] for key in point}
        for _ in range(replicates):
            rep = _ + 1
            rows = rng.integers(target.shape[0], size=target.shape[0]) if mode in {"patients", "hierarchical"} else np.arange(target.shape[0])
            selected = rng.integers(len(blocks), size=len(blocks)) if mode in {"genomic_blocks", "hierarchical"} else np.arange(len(blocks))
            cols = np.concatenate([blocks[i] for i in selected])
            value = delta(rows, cols)
            for key in values: values[key].append(value[key])
            if rep % 100 == 0 or rep == replicates:
                print(f"bootstrap seed={seed} mode={mode} replicate={rep}/{replicates}", file=sys.stderr, flush=True)
            if rep % 200 == 0:
                # Brief pacing pause: a hardware-crash pattern was observed
                # coinciding with many hours of sustained near-100% CPU/GPU
                # utilization on this machine (see project memory). This
                # yields the CPU periodically instead of pegging it
                # uninterrupted for the full multi-hour bootstrap -- cheap
                # (<1% wall-clock overhead at 2000 replicates) insurance,
                # not a correctness change.
                time.sleep(1.5)
        out[mode] = {f"{key}_{tail}": float(np.quantile(value, q)) for key, value in values.items() for tail, q in (("ci_low", .025), ("ci_high", .975))}
        if include_draws:
            out[mode]["draws"] = values
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c0-test", nargs=3, type=Path, required=True)
    p.add_argument("--c0-validation", nargs=3, type=Path, required=True)
    p.add_argument("--mp-released", type=Path, required=True)
    p.add_argument("--mp-static", type=Path, required=True, help="cpg_idx,pred_methyl; fixed train-mean RNA")
    p.add_argument("--empirical-prior", type=Path, required=True, help="cpg_idx,mean_train,position")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=9176)
    p.add_argument("--include-bootstrap-draws", action="store_true")
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"stage_d1 start replicates={a.replicates} output={a.output_dir}", file=sys.stderr, flush=True)
    test = [npz(x) for x in a.c0_test]; val = [npz(x) for x in a.c0_validation]
    for i, item in enumerate(test[1:], 1): _assert_panels(test[0], item, f"C0 test seed {i}")
    for i, item in enumerate(val):
        for key in ("cpg_idx",):
            if not np.array_equal(test[0][key].astype(str), item[key].astype(str)): raise ValueError("C0 validation has different CpGs")
    target, cancer = test[0]["target"].astype(float), test[0]["cancer_type"].astype(str)
    samples, cpgs = test[0]["sample_idx"].astype(str), test[0]["cpg_idx"].astype(str)
    prior_nt = np.broadcast_to(test[0]["prior"].astype(float)[None, :], target.shape)
    mp_pred = _released_panel(a.mp_released, samples, cpgs, cancer, target)
    static = pd.read_parquet(a.mp_static); static.cpg_idx = static.cpg_idx.astype(str)
    empirical = pd.read_parquet(a.empirical_prior); empirical.cpg_idx = empirical.cpg_idx.astype(str)
    mp_mu = static.set_index("cpg_idx").loc[cpgs, "pred_methyl"].to_numpy(float)
    empirical_mu = empirical.set_index("cpg_idx").loc[cpgs, "mean_train"].to_numpy(float)
    if not np.isfinite(mp_mu).all() or not np.isfinite(empirical_mu).all(): raise ValueError("static prior does not cover C0 CpGs")
    mp_static = np.broadcast_to(mp_mu[None, :], target.shape); empirical_static = np.broadcast_to(empirical_mu[None, :], target.shape)
    # C0 validation and released MP validation share the exact CpGs; verify
    # targets before fitting each alpha, and never consult test targets here.
    val_samples = val[0]["sample_idx"].astype(str); val_target = val[0]["target"].astype(float)
    for item in val[1:]:
        _assert_panels(val[0], item, "C0 validation seed")
    mp_val = _released_panel(a.mp_released, val_samples, cpgs, val[0]["cancer_type"].astype(str), val_target)
    mp_static_val = np.broadcast_to(mp_mu[None, :], val_target.shape)
    nt_val = np.broadcast_to(val[0]["prior"].astype(float)[None, :], val_target.shape)
    alpha_mp = alpha(val_target, mp_val, mp_static_val)
    alpha_c0 = [alpha(val_target, item["prediction"].astype(float), nt_val) for item in val]
    c0_test = [item["prediction"].astype(float) for item in test]
    c0_ensemble = np.mean(c0_test, axis=0); alpha_c0_ensemble = alpha(val_target, np.mean([x["prediction"].astype(float) for x in val], axis=0), nt_val)
    models = {"empirical_train_only_prior": empirical_static, "methylprophet_static_prior": mp_static,
              "ntv3_prior": prior_nt, "methylprophet_original": mp_pred,
              "mp_prior_plus_calibrated_mp_dynamic": calibrated(mp_static, mp_pred, alpha_mp),
              "ntv3_prior_plus_calibrated_mp_dynamic": calibrated(prior_nt, mp_pred, alpha_mp)}
    for seed, pred, coef in zip((17, 29, 43), c0_test, alpha_c0):
        models[f"c0_v0_seed{seed}"] = pred
        models[f"ntv3_prior_plus_calibrated_c0_dynamic_seed{seed}"] = calibrated(prior_nt, pred, coef)
    models["c0_v0_beta_ensemble"] = c0_ensemble
    models["ntv3_prior_plus_calibrated_c0_dynamic_ensemble"] = calibrated(prior_nt, c0_ensemble, alpha_c0_ensemble)
    point = {name: metrics(target, pred, prior_nt[0], cancer) for name, pred in models.items()}
    # 5 Mb genomic blocks are fixed from coordinates, never estimated from test outcomes.
    pos = empirical.set_index("cpg_idx").loc[cpgs, "position"].to_numpy(int)
    labels = pos // 5_000_000; blocks = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    comparisons = {f"mp_vs_c0_seed{seed}": bootstrap(target, prior_nt[0], cancer, blocks, [pred], mp_pred, a.replicates, a.seed + seed, a.include_bootstrap_draws)
                   for seed, pred in zip((17, 29, 43), c0_test)}
    comparisons["mp_vs_c0_mean_seed_metrics"] = bootstrap(target, prior_nt[0], cancer, blocks, c0_test, mp_pred, a.replicates, a.seed + 100, a.include_bootstrap_draws)
    comparisons["mp_vs_c0_beta_ensemble"] = bootstrap(target, prior_nt[0], cancer, blocks, [c0_ensemble], mp_pred, a.replicates, a.seed + 101, a.include_bootstrap_draws)
    # Variability tertiles are defined once from the train-only empirical variance proxy.
    variance = empirical.set_index("cpg_idx").loc[cpgs, "within_cpg_variance"].to_numpy(float)
    tertiles = np.empty(len(cpgs), int); order = np.argsort(variance)
    for label, idx in enumerate(np.array_split(order, 3)): tertiles[idx] = label
    tertile_metrics = {name: {f"tertile_{t+1}": metrics(target[:, tertiles == t], pred[:, tertiles == t], prior_nt[0, tertiles == t], cancer) for t in range(3)} for name, pred in models.items()}
    result = {"claim_scope": "benchmark of released MethylProphet checkpoint; not a matched-training architecture claim",
              "contract": {"patients": len(samples), "cpgs": len(cpgs), "observed_rows": int(np.isfinite(target).sum()), "bootstrap_blocks": len(blocks), "block_size_bp": 5_000_000,
                           "test_selection": "none; C0 checkpoints selected by validation beta-MSE", "calibration": "all alpha values fitted only on validation sample x test-CpG rows"},
              "calibration_alpha": {"methylprophet_dynamic": alpha_mp, **{f"c0_seed{s}": x for s, x in zip((17,29,43), alpha_c0)}, "c0_ensemble": alpha_c0_ensemble},
              "metrics": point, "variability_tertiles": tertile_metrics, "paired_bootstrap": comparisons,
              "conventions": {"delta_mse": "candidate minus MethylProphet; negative favours C0", "delta_skill": "candidate minus MethylProphet; positive favours C0", "inference": "hierarchical patient x 5-Mb block CI is primary"}}
    (a.output_dir / "stage_d1_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([{ "model": name, **{key: value for key, value in metric.items() if key != "per_cancer"}} for name, metric in point.items()]).to_csv(a.output_dir / "stage_d1_metrics.csv", index=False)
    print("stage_d1 complete", file=sys.stderr, flush=True)


if __name__ == "__main__": main()
