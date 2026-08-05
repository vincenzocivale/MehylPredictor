#!/usr/bin/env python3
"""Stage D3 genome-wide: bilinear/concat ensembles (trained on real MethylProphet
train_cpg x train_sample, genome-wide, 408,399 CpGs) vs. the released MethylProphet
checkpoint, on double_ood (val_cpg x val_sample -- the cell neither model ever
trained on).

Scope limitation (verified on disk, not assumed): MethylProphet's released
predictions only cover chr1 (6,742 CpGs) -- there is no genome-wide released
eval or checkpoint available. This script therefore restricts the official MP
comparison to the intersection of our genome-wide frozen 30k-CpG double_ood
subset with those released chr1 CpGs (measured: 2,436 CpGs, all 414 samples
covered). The genome-wide-vs-prior-only comparison (all 30k CpGs, no MP) was
already reported separately from build_ensembles_and_compare.py's output.

Also drops the calibrated prior+dynamic MP hybrid rows that stage_d3_matrix_v1.py
reports: those need MethylProphet's mean-RNA-fixed static prior, which only
exists on disk for the original 521-CpG Stage D1 panel, not this 2,436-CpG one.
Regenerating it would mean re-running MethylProphet's own inference -- out of
scope here. The primary table (full models vs released MethylProphet vs prior)
does not depend on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from methylation_predictor.diagnostics.methylprophet.stage_d1 import _released_panel, bootstrap, genomic_blocks, metrics


def npz(path: Path) -> dict:
    x = np.load(path, allow_pickle=True)
    return {k: x[k] for k in x.files}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bilinear-panels-dirs", nargs="+", type=Path, required=True, help="seed17 first (reference), then others")
    p.add_argument("--concat-panels-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--mp-released", type=Path, required=True)
    p.add_argument("--empirical-prior", type=Path, required=True, help="genome-wide cpg_idx,mean_train,position,chromosome,within_cancer_variance")
    p.add_argument("--panel", default="double_ood")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=9176)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"stage_d3_genome_wide_v1 start replicates={a.replicates}", file=sys.stderr, flush=True)

    bilinear = [npz(d / f"panel_{a.panel}.npz") for d in a.bilinear_panels_dirs]
    concat = [npz(d / f"panel_{a.panel}.npz") for d in a.concat_panels_dirs]
    ref = bilinear[0]
    for other in bilinear[1:] + concat:
        if not np.array_equal(ref["cpg_idx"], other["cpg_idx"]) or not np.array_equal(ref["sample_idx"], other["sample_idx"]):
            raise ValueError("all seeds/models must share the identical frozen sample/cpg subset")

    full_cpgs = ref["cpg_idx"].astype(str)
    full_samples = ref["sample_idx"].astype(str)

    mp_group2 = ds.dataset(a.mp_released, format="parquet").to_table(
        columns=["cpg_idx"], filter=(ds.field("group_idx") == 2)
    ).to_pandas()
    mp_cpgs = set(mp_group2.cpg_idx.astype(str).unique().tolist())
    keep = np.array([c in mp_cpgs for c in full_cpgs])
    n_keep = int(keep.sum())
    if n_keep == 0:
        raise ValueError("no overlap between the frozen subset and MethylProphet's released CpGs")
    print(f"restricting {len(full_cpgs)} frozen CpGs to {n_keep} released-MP-covered CpGs", file=sys.stderr, flush=True)

    cpgs = full_cpgs[keep]
    samples = full_samples
    cancer = ref["cancer_type"].astype(str)
    target = ref["target"][:, keep].astype(float)
    prior_nt = np.broadcast_to(ref["prior"][keep].astype(float)[None, :], target.shape)

    mp_pred = _released_panel(a.mp_released, samples, cpgs, cancer, target)

    bilinear_preds = [b["prediction"][:, keep].astype(float) for b in bilinear]
    concat_preds = [c["prediction"][:, keep].astype(float) for c in concat]
    bilinear_ensemble = np.mean(bilinear_preds, axis=0)
    concat_ensemble = np.mean(concat_preds, axis=0)

    empirical = pd.read_parquet(a.empirical_prior)
    empirical.cpg_idx = empirical.cpg_idx.astype(str)
    emp_row = empirical.set_index("cpg_idx").loc[cpgs]
    empirical_mu = emp_row["mean_train"].to_numpy(float)
    if not np.isfinite(empirical_mu).all():
        raise ValueError("empirical prior does not cover the restricted CpGs")
    empirical_static = np.broadcast_to(empirical_mu[None, :], target.shape)

    models = {
        "empirical_train_only_prior": empirical_static,
        "ntv3_prior": prior_nt,
        "methylprophet_original": mp_pred,
        "bilinear_v1_ensemble": bilinear_ensemble,
        "concat_ensemble": concat_ensemble,
    }
    for i, pred in enumerate(bilinear_preds):
        models[f"bilinear_seed{i}"] = pred
    for i, pred in enumerate(concat_preds):
        models[f"concat_seed{i}"] = pred

    point = {name: metrics(target, pred, prior_nt[0], cancer) for name, pred in models.items()}

    blocks = genomic_blocks(emp_row["chromosome"].to_numpy(), emp_row["position"].to_numpy(int))

    comparisons = {
        "concat_vs_mp": bootstrap(target, prior_nt[0], cancer, blocks, [concat_ensemble], mp_pred, a.replicates, a.seed + 1),
        "bilinear_vs_mp": bootstrap(target, prior_nt[0], cancer, blocks, [bilinear_ensemble], mp_pred, a.replicates, a.seed + 2),
        "concat_vs_bilinear": bootstrap(target, prior_nt[0], cancer, blocks, [concat_ensemble], bilinear_ensemble, a.replicates, a.seed + 3),
    }

    result = {
        "claim_scope": (
            "Genome-wide-trained bilinear/concat ensembles (real MethylProphet train_cpg x "
            "train_sample, 408,399 CpGs) vs. released MethylProphet checkpoint, restricted to "
            "the chr1 subset MethylProphet's released predictions actually cover -- MP was never "
            "released/evaluated genome-wide, verified on disk (only eval-tcga_mix_chr1-* exists)."
        ),
        "contract": {
            "patients": len(samples), "cpgs_in_frozen_genome_wide_subset": len(full_cpgs),
            "cpgs_overlapping_released_mp": n_keep, "observed_rows": int(np.isfinite(target).sum()),
            "bootstrap_blocks": len(blocks), "block_size_bp": 5_000_000, "panel": a.panel,
        },
        "metrics": point,
        "paired_bootstrap": comparisons,
        "conventions": {
            "delta_mse": "candidate minus reference; negative favours candidate",
            "delta_skill": "candidate minus reference; positive favours candidate",
            "inference": "hierarchical patient x 5-Mb block CI is primary",
            "primary_metric": "beta-space MSE on identical observations",
            "not_included": "calibrated prior+dynamic MP hybrid rows -- needs MP's mean-RNA-fixed "
                             "static prior, only available on disk for the original 521-CpG Stage D1 "
                             "panel, not this 2,436-CpG one",
        },
    }
    (a.output_dir / "stage_d3_genome_wide_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(
        [{"model": name, **{k: v for k, v in m.items() if k not in ("per_cancer", "per_variability_tertile")}} for name, m in point.items()]
    ).to_csv(a.output_dir / "stage_d3_genome_wide_metrics.csv", index=False)
    print("stage_d3_genome_wide_v1 complete", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
