#!/usr/bin/env python3
"""Fase 7: full, uncapped evaluation of the final-refit concat checkpoint on
all 4 generalization panels, using the ORIGINAL (unmodified) official split
manifests -- i.e. this is the one place official val_cpg/val_sample get
touched, and only after the checkpoint is fully frozen.

- double_ood (val_sample x val_cpg, the primary required panel): official val
  = validation ∪ test labels, ALL 81,493 val_cpg, all aligned val_sample, no
  30k cap. Small enough (~66M cells) for the existing non-streaming
  predict_panel/evaluate_predictions path (exact, includes median
  correlations); predictions are additionally streamed to chunked HDF5 (only
  for this panel, per the "if disk allows" instruction) under /data scratch,
  not the repo's tight `/` filesystem.
- in_distribution (train x train), sample_ood (val_sample x train_cpg),
  locus_ood (train_sample x val_cpg): touch the full ~327k-CpG train_cpg
  pool -- too large to materialize in RAM, so these use
  evaluate_panel_streaming (sufficient statistics only, no predictions
  saved, per "non e' necessario salvare ogni predizione" for huge panels).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.trainer import ExperimentRunner

FEATURES_PATH = Path("/data/dataset/methylation/genomic_encoder_genome_wide_scratch/genome_wide_features.parquet")


def _chromosome_lookup() -> dict[str, str]:
    table = pd.read_parquet(FEATURES_PATH, columns=["cpg_idx", "chromosome"])
    return dict(zip(table["cpg_idx"].astype(str), table["chromosome"].astype(str)))


def _timed_panel(fn, device) -> tuple[dict, float, float | None]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else None
    return result, elapsed, peak_vram


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="final-refit config (original, unmodified split manifests)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--double-ood-predictions-h5", type=Path, default=None,
                    help="chunked HDF5 dump of double_ood predictions, if disk allows")
    p.add_argument("--skip-huge-panels", action="store_true",
                    help="skip in_distribution/sample_ood/locus_ood (for a quick double_ood-only run)")
    p.add_argument(
        "--huge-panel-locus-median-correlations", action="store_true",
        help="also compute the exact per-CpG locus_dynamic_{pearson,spearman}_median diagnostic for "
             "the 3 huge panels (default: skipped there -- measured at real genome-wide scale to cost "
             "~7-10 extra minutes per panel for a secondary diagnostic; every other metric, including "
             "for double_ood, remains exact regardless of this flag)",
    )
    args = p.parse_args()

    config = load_config(args.config)
    runner = ExperimentRunner(config)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
        runner.model.load_state_dict(checkpoint["model_state"])
        runner._refresh_train_centroids()

        chromosome_by_cpg_id = _chromosome_lookup()
        # Panel-level resume: this evaluation touches multi-billion-cell panels
        # and can take a long while, so each panel's result is written to
        # `--output` immediately after it completes (not just at the very
        # end) and an already-present panel key is skipped on restart --
        # a crash/interruption partway through only re-does the panel that
        # was in flight, not everything before it.
        result: dict[str, object] = (
            json.loads(args.output.read_text()) if args.output.is_file() else {}
        )
        result["checkpoint"] = str(args.checkpoint)
        result["checkpoint_epoch"] = checkpoint.get("epoch")
        args.output.parent.mkdir(parents=True, exist_ok=True)

        def _save() -> None:
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

        official_val_sample = np.union1d(runner.bundle.sample_indices("validation"), runner.bundle.sample_indices("test"))
        official_val_cpg = np.union1d(runner.bundle.cpg_indices("validation"), runner.bundle.cpg_indices("test"))
        official_train_sample = runner.bundle.sample_indices("train")
        official_train_cpg = runner.bundle.cpg_indices("train")

        if "double_ood" in result:
            print("[full_eval] double_ood: already complete, skipping", flush=True)
        else:
            print(f"[full_eval] double_ood: {len(official_val_sample)} samples x {len(official_val_cpg)} cpgs", flush=True)
            panel, elapsed, peak_vram = _timed_panel(
                lambda: runner.predict_panel(
                    None, None, sample_indices_override=official_val_sample, cpg_indices_override=official_val_cpg,
                    keep_predictions=True,
                ),
                runner.device,
            )
            double_ood_metrics = dict(panel.metrics)
            double_ood_metrics["inference_seconds"] = elapsed
            double_ood_metrics["peak_vram_gb"] = peak_vram
            double_ood_metrics["samples"] = len(official_val_sample)
            double_ood_metrics["cpgs"] = len(official_val_cpg)
            result["double_ood"] = double_ood_metrics

            if args.double_ood_predictions_h5 is not None:
                import h5py
                args.double_ood_predictions_h5.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(args.double_ood_predictions_h5, "w") as handle:
                    handle.create_dataset("prediction", data=panel.prediction, compression="gzip", compression_opts=1)
                    handle.create_dataset("target", data=panel.target, compression="gzip", compression_opts=1)
                    handle.create_dataset("prior", data=runner.bundle.loci.prior[panel.cpg_indices])
                    string_dtype = h5py.string_dtype("utf-8")
                    handle.create_dataset(
                        "sample_idx",
                        data=np.asarray(runner.bundle.samples.ids[panel.sample_indices].astype(str), dtype=object),
                        dtype=string_dtype,
                    )
                    handle.create_dataset(
                        "cpg_idx",
                        data=np.asarray(runner.bundle.loci.ids[panel.cpg_indices].astype(str), dtype=object),
                        dtype=string_dtype,
                    )
                result["double_ood_predictions_h5"] = str(args.double_ood_predictions_h5)
            _save()

        if not args.skip_huge_panels:
            huge_panels = {
                "in_distribution": (official_train_sample, official_train_cpg),
                "sample_ood": (official_val_sample, official_train_cpg),
                "locus_ood": (official_train_sample, official_val_cpg),
            }
            for name, (sample_idx, cpg_idx) in huge_panels.items():
                if name in result:
                    print(f"[full_eval] {name}: already complete, skipping", flush=True)
                    continue
                print(f"[full_eval] {name}: {len(sample_idx)} samples x {len(cpg_idx)} cpgs (streaming)", flush=True)
                metrics, elapsed, peak_vram = _timed_panel(
                    lambda si=sample_idx, ci=cpg_idx: runner.evaluate_panel_streaming(
                        None, None, chromosome_by_cpg_id=chromosome_by_cpg_id,
                        sample_indices_override=si, cpg_indices_override=ci,
                        include_locus_median_correlations=args.huge_panel_locus_median_correlations,
                    ),
                    runner.device,
                )
                metrics["inference_seconds"] = elapsed
                metrics["peak_vram_gb"] = peak_vram
                result[name] = metrics
                _save()
                print(f"[full_eval] {name} done in {elapsed:.1f}s: mse={metrics.get('mse')} "
                      f"skill_vs_prior={metrics.get('skill_vs_prior')}", flush=True)

        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        print(f"[full_eval] wrote {args.output}", flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
