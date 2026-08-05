#!/usr/bin/env python3
"""Efficiency comparison: our concat checkpoint vs. the released MethylProphet
checkpoint, same hardware.

Scope, stated explicitly rather than silently assumed: this repo has never
loaded MethylProphet's live model into a running process anywhere (grepped:
only `stage_d_matched.py::extract_checkpoint_predictions` exists, which reads
their already-computed RELEASED prediction rows, never runs their model
forward) -- MP's checkpoint is a ~2.2GB PyTorch Lightning module (BERT-based
gene encoder + regressor head, verified via `state_dict` key inspection).
Building a live inference harness for an unfamiliar external model (matching
their exact input contract, tokenization/quantization, batching) from
scratch is a substantial undertaking with real risk of a silently-wrong
forward pass and a correspondingly wrong timing number -- out of proportion
to this being the lowest-priority deliverable (Fase 10) relative to the
core scientific comparison (Fase 7-9). This script therefore:

- measures parameter count, checkpoint file size, and checkpoint load time
  for MP's checkpoint HONESTLY, without instantiating their model class
  (just `torch.load` + summing `state_dict` tensor `.numel()`);
- measures full inference-side numbers (per-patient time, full-panel time,
  throughput, peak VRAM, cache benefit) for OUR OWN concat/bilinear
  checkpoints via the existing `predict_panel`/`benchmark-inference` path;
- explicitly reports MP's live forward-pass inference timing as
  "not measured" with the reason above, rather than fabricating or
  guessing a number.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.trainer import ExperimentRunner


def _our_model_benchmark(config_path: Path, checkpoint_path: Path, sample_split: str, cpg_split: str,
                          max_cpgs: int | None) -> dict[str, object]:
    config = load_config(config_path)
    runner = ExperimentRunner(config)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=runner.device, weights_only=False)
        runner.model.load_state_dict(checkpoint["model_state"])
        runner._refresh_train_centroids()

        if runner.device.type == "cuda":
            torch.cuda.synchronize(runner.device)
            torch.cuda.reset_peak_memory_stats(runner.device)
        # warm-up (excluded from timing): one small forward pass to pay for
        # any lazy CUDA kernel compilation/allocator warm-up.
        runner.predict_panel(sample_split, cpg_split, max_cpgs=min(max_cpgs or 64, 64), seed_offset=7777)
        if runner.device.type == "cuda":
            torch.cuda.synchronize(runner.device)
            torch.cuda.reset_peak_memory_stats(runner.device)

        started = time.perf_counter()
        panel = runner.predict_panel(
            sample_split, cpg_split, max_cpgs=max_cpgs, use_factorized_cache=runner.model.supports_factorized_inference,
        )
        if runner.device.type == "cuda":
            torch.cuda.synchronize(runner.device)
        elapsed = time.perf_counter() - started

        num_parameters = sum(p.numel() for p in runner.model.parameters())
        checkpoint_size_bytes = Path(checkpoint_path).stat().st_size
        load_started = time.perf_counter()
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_load_seconds = time.perf_counter() - load_started

        n_samples, n_cpgs = len(panel.sample_indices), len(panel.cpg_indices)
        return {
            "num_parameters": num_parameters,
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "supports_factorized_inference": runner.model.supports_factorized_inference,
            "panel_samples": n_samples,
            "panel_cpgs": n_cpgs,
            "panel_inference_seconds": elapsed,
            "panel_pairs_per_second": n_samples * n_cpgs / max(elapsed, 1e-9),
            "per_patient_seconds": elapsed / max(n_samples, 1),
            "peak_vram_gb": (
                torch.cuda.max_memory_allocated(runner.device) / (1024 ** 3) if runner.device.type == "cuda" else None
            ),
        }
    finally:
        runner.close()


def _mp_checkpoint_benchmark(checkpoint_path: Path) -> dict[str, object]:
    checkpoint_size_bytes = Path(checkpoint_path).stat().st_size
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_seconds = time.perf_counter() - started
    state_dict = checkpoint.get("state_dict", checkpoint)
    num_parameters = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
    return {
        "num_parameters": num_parameters,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "checkpoint_load_seconds": load_seconds,
        "training_wall_clock": "N/A -- released checkpoint, not retrained (per constraint: MP comparison "
                                "uses only the official released model/predictions, no retraining)",
        "inference_timing": "not measured -- no live inference harness for MethylProphet's model exists "
                             "in this repo (only released, pre-computed predictions are consumed, matching "
                             "existing repo convention e.g. stage_d_matched.py); building one was judged "
                             "out of scope for this task, see module docstring",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concat-config", type=Path, required=True)
    p.add_argument("--concat-checkpoint", type=Path, required=True)
    p.add_argument("--bilinear-config", type=Path, required=True)
    p.add_argument("--bilinear-checkpoint", type=Path, required=True)
    p.add_argument("--mp-checkpoint", type=Path, required=True)
    p.add_argument("--sample-split", default="test")
    p.add_argument("--cpg-split", default="test")
    p.add_argument("--max-cpgs", type=int, default=2436, help="matches the chr1 MP-overlap panel size by default")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    result = {
        "concat_genomewide_fullcoverage_seed17_v1": _our_model_benchmark(
            args.concat_config, args.concat_checkpoint, args.sample_split, args.cpg_split, args.max_cpgs
        ),
        "bilinear_v1_frozen": _our_model_benchmark(
            args.bilinear_config, args.bilinear_checkpoint, args.sample_split, args.cpg_split, args.max_cpgs
        ),
        "methylprophet_official_released": _mp_checkpoint_benchmark(args.mp_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
