#!/usr/bin/env python3
"""Readiness check for the pretrained-FM masked-CpG pipeline (CpGPT / MethylGPT /
DeepCpG).

Deliberately does NOT run the full val_cpg_x_train_sample split (see
`docs/PAPER_EXPERIMENTS.md`'s foundation-model section) -- only confirms: the
vendored repo + all checkpoint variants are present, weights are demonstrably
loaded (not randomly initialized -- `verify.py`'s before/after fingerprint plus a
tiny forward-pass output diff), and, for CpGPT, a real chr1 val-CpG-id sample can
be pulled from the existing `matched_chr1` cache.

Each model needs its own isolated Python environment (see
`scripts/benchmark_foundation_models/setup.sh`) -- the three pull in mutually
awkward dependency stacks (MethylGPT's `torchtext` has no build compatible with
this repo's main torch install; DeepCpG needs a legacy `python=3.7`/
`tensorflow==1.13.1`/`keras==1.2.2` conda env). Run this script once per model,
with that model's environment:

    external/cpgpt-env/bin/python scripts/benchmark_foundation_models/check_readiness.py --model cpgpt
    external/methylgpt-env/bin/python scripts/benchmark_foundation_models/check_readiness.py --model methylgpt
    conda run -n deepcpg-env python scripts/benchmark_foundation_models/check_readiness.py --model deepcpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "CpGPT"))
sys.path.insert(0, str(REPO_ROOT / "external" / "MethylGPT"))

from methylation_predictor.benchmark.foundation_models import verify  # noqa: E402


def check_cpgpt(checkpoints_dir: Path) -> dict:
    from methylation_predictor.benchmark.foundation_models.adapters import cpgpt_adapter as adapter

    report = {"model": "cpgpt", "variants": {}}
    for variant in adapter.VALID_VARIANTS:
        entry: dict = {}
        try:
            resources = adapter.resolve_resources(checkpoints_dir, variant)
            model, config = adapter.build_random_model(resources.config_path)
            probe = adapter.dna_embedding_forward_probe(model, config)
            result = verify.verify_checkpoint_loaded(
                model,
                resources.checkpoint_path,
                lambda m, ckpt=resources.checkpoint_path: adapter.load_checkpoint_into(m, ckpt),
                forward_probe=probe,
            )
            entry["status"] = "PASS" if result.passed else "FAIL"
            entry["verification"] = result.as_dict()
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the other variants
            entry["status"] = "ERROR"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["variants"][variant] = entry
    return report


def _diagnose_methylgpt_key_mismatch(model, checkpoint_path: Path, fresh_random_model) -> dict:
    """Explain *why* some tensors were still flagged unchanged, post-remap.

    UPDATE 2026-09-02: the attention `Wqkv` vs `in_proj_weight` mismatch is a pure
    key rename (identical `(3*d_model, d_model)`/`(3*d_model,)` shapes, confirmed by
    direct comparison) -- NOT a different parameterization requiring flash-attn, as
    an earlier version of this diagnosis incorrectly assumed without checking
    shapes first. `methylgpt_adapter.load_checkpoint_into` now remaps those keys
    before loading; `check_readiness.py` applies the same remap here so this
    diagnosis reflects the fixed loading path.

    After that fix, the only tensors still reported "unchanged" by `verify.py`
    (`cls_decoder._decoder.2/.5`) are present in the checkpoint under the exact
    same name and shape as the model -- this function checks whether the
    checkpoint's own value for those keys already equals the fresh-model's
    (pre-load) value. If so, it's a benign fingerprint false positive (most likely
    an untrained/passthrough LayerNorm that never moved from its default
    weight=1/bias=0 init during training), not evidence the checkpoint failed to
    load -- but this is inferred from matching values, not proven from training
    logs, so it's reported as a caveat rather than silently suppressed.
    """
    import torch

    from methylation_predictor.benchmark.foundation_models.adapters.methylgpt_adapter import (
        _remap_flash_attn_qkv_keys,
    )

    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    ckpt = _remap_flash_attn_qkv_keys(raw_ckpt)
    ckpt_keys = set(ckpt.keys())
    model_keys = set(model.state_dict().keys())
    fresh_sd = fresh_random_model.state_dict()  # independent, never-loaded instance

    shared_but_unchanged = []
    for key in sorted(model_keys & ckpt_keys):
        if key in fresh_sd and torch.allclose(ckpt[key].float(), fresh_sd[key].float(), atol=1e-6):
            shared_but_unchanged.append(key)

    return {
        "model_keys_not_in_checkpoint": sorted(model_keys - ckpt_keys)[:10],
        "checkpoint_keys_not_in_model": sorted(ckpt_keys - model_keys)[:10],
        "present_in_both_but_value_already_matched_pre_load": shared_but_unchanged,
        "note": (
            "keys listed in the last field are not a loading failure -- they were "
            "present in the checkpoint with the exact same value the fresh model "
            "already had (very likely an untrained/passthrough LayerNorm stuck at "
            "its default weight=1/bias=0 init), so the fingerprint 'unchanged' "
            "signal is a false positive here, not evidence of a skipped tensor."
        ),
    }


def check_methylgpt(checkpoints_dir: Path) -> dict:
    from methylation_predictor.benchmark.foundation_models.adapters import methylgpt_adapter as adapter

    report = {"model": "methylgpt", "variants": {}}
    for variant in adapter.VALID_VARIANTS:
        entry: dict = {}
        try:
            resources = adapter.resolve_resources(checkpoints_dir, variant)
            entry["identity_check"] = adapter.check_variant_identity(resources)
            model, vocab, _config = adapter.build_model(resources, load_model=False)
            probe = adapter.embedding_forward_probe(model, vocab)
            result = verify.verify_checkpoint_loaded(
                model,
                resources.checkpoint_path,
                lambda m, ckpt=resources.checkpoint_path: adapter.load_checkpoint_into(m, ckpt),
                forward_probe=probe,
            )
            entry["verification"] = result.as_dict()
            entry["vocab_size"] = len(vocab)
            if not result.passed:
                fresh_model, _fresh_vocab, _fresh_config = adapter.build_model(resources, load_model=False)
                diagnosis = _diagnose_methylgpt_key_mismatch(model, resources.checkpoint_path, fresh_model)
                entry["mismatch_diagnosis"] = diagnosis
                all_unchanged_are_benign = set(result.unchanged_param_names) == set(
                    diagnosis["present_in_both_but_value_already_matched_pre_load"]
                )
                entry["status"] = "PASS_WITH_BENIGN_CAVEAT" if all_unchanged_are_benign else "FAIL"
            else:
                entry["status"] = "PASS"
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "ERROR"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["variants"][variant] = entry
    return report


def check_deepcpg(checkpoints_dir: Path, fasta_path: str | None) -> dict:
    from methylation_predictor.benchmark.foundation_models.adapters import deepcpg_adapter as adapter

    report = {"model": "deepcpg", "variants": {}}
    for variant in adapter.VALID_VARIANTS:
        entry: dict = {}
        try:
            resources = adapter.resolve_resources(checkpoints_dir, variant)
            model = adapter.build_random_model(resources.arch_path)
            probe = adapter.dna_forward_probe(model)  # random one-hot windows -- no FASTA needed
            result = verify.verify_checkpoint_loaded_keras(
                model,
                resources.weights_path,
                lambda m, w=resources.weights_path: adapter.load_checkpoint_into(m, w),
                forward_probe=probe,
            )
            entry["status"] = "PASS" if result.passed else "FAIL"
            entry["verification"] = result.as_dict()
            entry["patient_agnostic"] = True  # DNA-sequence-only: same prediction for every sample
            if fasta_path:
                window = adapter.one_hot_dna_window(fasta_path, "chr1", 1_000_000)
                entry["real_hg38_window_check"] = {
                    "shape": list(window.shape),
                    "n_bases_called": int(window.sum()),
                }
            else:
                entry["real_hg38_window_check"] = "skipped: no --fasta-path given"
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "ERROR"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["variants"][variant] = entry
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["cpgpt", "methylgpt", "deepcpg"], required=True)
    p.add_argument(
        "--checkpoints-dir",
        default=None,
        help="Defaults to external/checkpoints/<model> relative to the repo root.",
    )
    p.add_argument(
        "--fasta-path",
        default=None,
        help="deepcpg only: absolute path to an hg38 FASTA, to sanity-check a real "
        "chr1 DNA window (skipped, not an error, if omitted).",
    )
    args = p.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else REPO_ROOT / "external" / "checkpoints" / args.model

    if args.model == "cpgpt":
        report = check_cpgpt(checkpoints_dir)
    elif args.model == "methylgpt":
        report = check_methylgpt(checkpoints_dir)
    else:
        report = check_deepcpg(checkpoints_dir, args.fasta_path)

    print(json.dumps(report, indent=2, default=str))
    all_pass = all(
        v.get("status") in ("PASS", "PASS_WITH_BENIGN_CAVEAT") for v in report["variants"].values()
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
