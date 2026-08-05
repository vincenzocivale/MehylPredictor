#!/usr/bin/env python3
"""Stage D5: the rigorous genome-wide final-refit concat checkpoint vs. prior,
vs. the existing frozen bilinear baseline, vs. released MethylProphet -- all on
the FULL official double-OOD cell (val_sample x val_cpg, all 81,493 val_cpg,
no 30k cap), computed directly from checkpoints via ExperimentRunner rather
than pre-dumped capped/frozen panel .npz files.

Two comparisons at two different scopes, both reported (never conflated):
1. concat_vs_prior, concat_vs_bilinear -- on the FULL official val panel.
   `bilinear` here is the EXISTING frozen checkpoint from the earlier
   (partially leaky-early-stopping) genome-wide screen, reused only for a
   cheap forward pass per constraint #2 ("non riaprire lo screen
   bilinear/concat/residual") -- NOT retrained under the new protocol. This
   asymmetry (bilinear = old protocol, concat = new clean protocol) is
   recorded explicitly in the output, not hidden.
2. concat_vs_mp -- restricted to the intersection with MethylProphet's
   released chr1-only predictions (measured exactly here, not assumed --
   see `--mp-released`), since MP was never released/evaluated genome-wide
   (verified on disk: only eval-tcga_mix_chr1-* exists).

Genomic blocks for the hierarchical bootstrap are chromosome-aware
(`stage_d1.genomic_blocks`) -- required at genome-wide scale since `position`
alone is chromosome-local (see stage_d1.genomic_blocks's own docstring for
why `pos // 5_000_000` alone would silently collide loci across
chromosomes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from methylation_predictor.diagnostics.methylprophet.stage_d1 import _released_panel, bootstrap, genomic_blocks, metrics
from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.trainer import ExperimentRunner


def _official_val_panel(runner: ExperimentRunner):
    sample_indices = np.union1d(runner.bundle.sample_indices("validation"), runner.bundle.sample_indices("test"))
    cpg_indices = np.union1d(runner.bundle.cpg_indices("validation"), runner.bundle.cpg_indices("test"))
    panel = runner.predict_panel(
        None, None,
        sample_indices_override=sample_indices, cpg_indices_override=cpg_indices,
        keep_predictions=True,
    )
    return panel


def _load_runner(config_path: Path, checkpoint_path: Path) -> ExperimentRunner:
    config = load_config(config_path)
    runner = ExperimentRunner(config)
    checkpoint = torch.load(checkpoint_path, map_location=runner.device, weights_only=False)
    runner.model.load_state_dict(checkpoint["model_state"])
    runner._refresh_train_centroids()
    return runner


def _build_result(target, samples, cpgs, blocks, point, comparisons, mp_scope) -> dict[str, object]:
    return {
        "claim_scope": (
            "Primary result: concat_genomewide_fullcoverage_seed17_v1 (Stage 3B final refit, trained "
            "strictly on official train_sample x train_cpg with a full-coverage sampler, official val "
            "untouched until this evaluation) vs. NTv3 prior and vs. the existing frozen bilinear "
            "checkpoint (trained under the EARLIER protocol -- see 'bilinear_protocol_caveat' -- reused "
            "only for a cheap forward pass, not retrained), on the FULL official double-OOD cell "
            "(val_sample x val_cpg, all official val_cpg, no 30k cap). "
            "'MethylProphet official released model/predictions' comparison (mp_comparison_chr1_only "
            "below) is SEPARATE and restricted to the chr1 subset MethylProphet's released predictions "
            "actually cover -- MP was never released/evaluated genome-wide (verified on disk)."
        ),
        "bilinear_protocol_caveat": (
            "The bilinear checkpoint compared here was trained earlier under a protocol whose "
            "early-stopping validation split was later found to be part of the official val_cpg/"
            "val_sample pool (not a dev split carved from train). concat_genomewide_fullcoverage_seed17_v1"
            "was trained under the new, corrected 2-stage protocol (nested dev split strictly inside "
            "official train, official val untouched until this evaluation). This is an evaluation-time "
            "comparison on an identical frozen panel, not an apples-to-apples TRAINING comparison."
        ),
        "contract": {
            "patients": len(samples),
            "cpgs": len(cpgs),
            "observed_rows": int(np.isfinite(target).sum()),
            "bootstrap_blocks": len(blocks),
            "block_size_bp": 5_000_000,
        },
        "metrics": point,
        "paired_bootstrap": comparisons,
        "mp_comparison_chr1_only": mp_scope,
        "conventions": {
            "delta_mse": "candidate minus reference; negative favours candidate",
            "delta_skill": "candidate minus reference; positive favours candidate",
            "inference": "hierarchical patient x chromosome-aware 5-Mb block CI is primary; "
                         "patients-only and blocks-only are sensitivity analyses (see each bootstrap block's own keys)",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concat-config", type=Path, required=True)
    p.add_argument("--concat-checkpoint", type=Path, required=True)
    p.add_argument("--bilinear-config", type=Path, required=True)
    p.add_argument("--bilinear-checkpoint", type=Path, required=True)
    p.add_argument("--mp-released", type=Path, required=True)
    p.add_argument("--empirical-prior", type=Path, required=True, help="genome-wide cpg_idx,mean_train,position,chromosome,within_cancer_variance")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=9176)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: this stage's dominant cost (2000-replicate bootstraps)
    # can take hours and has no other checkpointing, and this environment has
    # been observed to interrupt long-running processes unpredictably. Cache
    # the (expensive-ish, ~10-30 GPU-min) computed panels so a restart that
    # only lost the bootstrap doesn't redo the forward passes too, and save
    # the output JSON incrementally after each of the 3 comparisons so a
    # restart resumes at the comparison that was in flight, not from zero.
    output_path = args.output_dir / "hierarchical_bootstrap.json"
    panel_cache_path = args.output_dir / ".stage_d5_panel_cache.npz"

    if panel_cache_path.is_file():
        print(f"stage_d5: reusing cached official-val panels from {panel_cache_path}", file=sys.stderr, flush=True)
        cache = np.load(panel_cache_path, allow_pickle=True)
        target = cache["target"]
        concat_pred = cache["concat_pred"]
        bilinear_pred = cache["bilinear_pred"]
        prior_nt = cache["prior_nt"]
        cancer_types = cache["cancer_types"]
        cpgs = cache["cpgs"]
        samples = cache["samples"]
    else:
        print("stage_d5 start: computing full official val panel for concat + bilinear", file=sys.stderr, flush=True)
        concat_runner = _load_runner(args.concat_config, args.concat_checkpoint)
        try:
            concat_panel = _official_val_panel(concat_runner)
            concat_cpg_ids = concat_runner.bundle.loci.ids[concat_panel.cpg_indices].astype(str)
            concat_sample_ids = concat_runner.bundle.samples.ids[concat_panel.sample_indices].astype(str)
            cancer_types = concat_runner.bundle.samples.cancer_types[concat_panel.sample_indices].astype(str)
            prior_nt = np.broadcast_to(
                concat_runner.bundle.loci.prior[concat_panel.cpg_indices].astype(float)[None, :],
                concat_panel.target.shape,
            )
        finally:
            concat_runner.close()

        bilinear_runner = _load_runner(args.bilinear_config, args.bilinear_checkpoint)
        try:
            bilinear_panel = _official_val_panel(bilinear_runner)
            bilinear_cpg_ids = bilinear_runner.bundle.loci.ids[bilinear_panel.cpg_indices].astype(str)
            bilinear_sample_ids = bilinear_runner.bundle.samples.ids[bilinear_panel.sample_indices].astype(str)
        finally:
            bilinear_runner.close()

        if not np.array_equal(concat_cpg_ids, bilinear_cpg_ids) or not np.array_equal(concat_sample_ids, bilinear_sample_ids):
            raise ValueError(
                "concat and bilinear official-val panels do not share identical sample/CpG identity -- "
                "check that both configs point at the same (unmodified) split manifest files"
            )
        if not np.allclose(
            np.nan_to_num(concat_panel.target), np.nan_to_num(bilinear_panel.target), atol=1e-6
        ) or not np.array_equal(np.isfinite(concat_panel.target), np.isfinite(bilinear_panel.target)):
            raise ValueError("concat and bilinear panels disagree on observed beta values -- data mismatch")

        target = concat_panel.target.astype(float)
        concat_pred = concat_panel.prediction.astype(float)
        bilinear_pred = bilinear_panel.prediction.astype(float)
        cpgs = concat_cpg_ids
        samples = concat_sample_ids
        np.savez_compressed(
            panel_cache_path, target=target, concat_pred=concat_pred, bilinear_pred=bilinear_pred,
            prior_nt=prior_nt, cancer_types=cancer_types, cpgs=cpgs, samples=samples,
        )

    empirical = pd.read_parquet(args.empirical_prior)
    empirical.cpg_idx = empirical.cpg_idx.astype(str)
    emp_row = empirical.set_index("cpg_idx").reindex(cpgs)
    if emp_row["position"].isna().any() or emp_row["chromosome"].isna().any():
        raise ValueError("empirical-prior table is missing chromosome/position for some official-val CpGs")
    blocks = genomic_blocks(emp_row["chromosome"].to_numpy(), emp_row["position"].to_numpy(int))

    point = {
        "ntv3_prior": metrics(target, prior_nt, prior_nt[0], cancer_types),
        "concat_genomewide_fullcoverage_seed17_v1": metrics(target, concat_pred, prior_nt[0], cancer_types),
        "bilinear_v1_frozen_old_protocol": metrics(target, bilinear_pred, prior_nt[0], cancer_types),
    }

    comparisons: dict[str, object] = {}
    mp_scope: dict[str, object] = {}
    if output_path.is_file():
        previous = json.loads(output_path.read_text())
        comparisons = previous.get("paired_bootstrap", {})
        mp_scope = previous.get("mp_comparison_chr1_only", {})

    def _save_partial() -> None:
        partial = _build_result(target, samples, cpgs, blocks, point, comparisons, mp_scope)
        output_path.write_text(json.dumps(partial, indent=2, sort_keys=True, default=str) + "\n")

    print("stage_d5: running concat_vs_prior / concat_vs_bilinear bootstrap on full official-val panel "
          f"({len(samples)} samples x {len(cpgs)} cpgs)", file=sys.stderr, flush=True)
    if "concat_vs_prior" in comparisons:
        print("stage_d5: concat_vs_prior already complete, skipping", file=sys.stderr, flush=True)
    else:
        comparisons["concat_vs_prior"] = bootstrap(target, prior_nt[0], cancer_types, blocks, [concat_pred], prior_nt,
                                                     args.replicates, args.seed + 1)
        _save_partial()
    if "concat_vs_bilinear" in comparisons:
        print("stage_d5: concat_vs_bilinear already complete, skipping", file=sys.stderr, flush=True)
    else:
        comparisons["concat_vs_bilinear"] = bootstrap(target, prior_nt[0], cancer_types, blocks, [concat_pred],
                                                        bilinear_pred, args.replicates, args.seed + 2)
        _save_partial()

    mp_group2 = ds.dataset(args.mp_released, format="parquet").to_table(
        columns=["cpg_idx"], filter=(ds.field("group_idx") == 2)
    ).to_pandas()
    mp_cpgs = set(mp_group2.cpg_idx.astype(str).unique().tolist())
    keep = np.array([c in mp_cpgs for c in cpgs])
    n_keep = int(keep.sum())
    print(f"stage_d5: {len(cpgs)} official-val CpGs intersected with MethylProphet's released "
          f"chr1-only predictions -> {n_keep} common CpGs", file=sys.stderr, flush=True)

    if n_keep == 0:
        mp_scope = {"note": "no overlap between official-val CpGs and MethylProphet's released CpGs"}
    elif "concat_vs_mp" in comparisons:
        print("stage_d5: concat_vs_mp already complete, skipping", file=sys.stderr, flush=True)
    else:
        mp_cpg_subset = cpgs[keep]
        mp_target = target[:, keep]
        mp_prior_row = prior_nt[0, keep]
        mp_pred = _released_panel(args.mp_released, samples, mp_cpg_subset, cancer_types, mp_target)
        mp_concat_pred = concat_pred[:, keep]
        mp_blocks = genomic_blocks(
            emp_row.loc[mp_cpg_subset, "chromosome"].to_numpy(), emp_row.loc[mp_cpg_subset, "position"].to_numpy(int)
        )
        mp_point = {
            "ntv3_prior": metrics(mp_target, np.broadcast_to(mp_prior_row[None, :], mp_target.shape), mp_prior_row, cancer_types),
            "methylprophet_original": metrics(mp_target, mp_pred, mp_prior_row, cancer_types),
            "concat_genomewide_fullcoverage_seed17_v1": metrics(mp_target, mp_concat_pred, mp_prior_row, cancer_types),
        }
        comparisons["concat_vs_mp"] = bootstrap(
            mp_target, mp_prior_row, cancer_types, mp_blocks, [mp_concat_pred], mp_pred, args.replicates, args.seed + 3
        )
        mp_scope = {
            "cpgs_official_val": len(cpgs),
            "cpgs_overlapping_released_mp": n_keep,
            "samples": len(samples),
            "metrics": mp_point,
        }
        _save_partial()

    result = _build_result(target, samples, cpgs, blocks, point, comparisons, mp_scope)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    panel_cache_path.unlink(missing_ok=True)
    pd.DataFrame(
        [{"model": name, **{k: v for k, v in m.items() if k not in ("per_cancer", "per_variability_tertile")}}
         for name, m in point.items()]
    ).to_csv(args.output_dir / "stage_d5_metrics.csv", index=False)
    print("stage_d5_genome_wide_official_val complete", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
