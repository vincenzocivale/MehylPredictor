#!/usr/bin/env python3
"""Render one controlled architecture-ablation config from the canonical base config."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


VARIANTS = (
    "baseline",
    "no_gate",
    "no_anchor",
    "no_product",
    "rna256",
    "direct_prediction",
    # Single confirmation run stacking the three modifications that matched
    # or beat baseline in the individual sweep (rna256, no_gate, no_anchor).
    # Not part of the sweep itself -- used only to confirm the winning single
    # change (rna256) isn't left worse off by also dropping the gate/anchor.
    "confirm_rna256_no_gate_no_anchor",
)


def render(base: dict, variant: str, *, run_name: str, output_dir: str, checkpoint_metric: str) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    cfg["run_name"] = run_name
    cfg["output_dir"] = output_dir
    cfg.setdefault("model", {})
    cfg["model"].setdefault("encoder", {})
    cfg["model"].setdefault("interaction", {})
    cfg["model"].setdefault("gate", {})

    # Pin the canonical architecture explicitly so a future base-config change
    # cannot silently mutate the ablation family.
    cfg["model"]["encoder"]["kind"] = "linear"
    cfg["model"]["encoder"]["latent_dim"] = 64
    cfg["model"]["interaction"]["kind"] = "concat"
    cfg["model"]["interaction"]["include_product"] = True
    cfg["model"]["gate"]["kind"] = "variability"
    cfg["model"]["anchor_to_mean_rna"] = True
    cfg["model"]["zero_init_residual"] = True
    cfg["model"]["prediction_mode"] = "residual_prior"

    if variant == "baseline":
        pass
    elif variant == "no_gate":
        cfg["model"]["gate"]["kind"] = "none"
    elif variant == "no_anchor":
        cfg["model"]["anchor_to_mean_rna"] = False
    elif variant == "no_product":
        cfg["model"]["interaction"]["include_product"] = False
    elif variant == "rna256":
        cfg["model"]["encoder"]["latent_dim"] = 256
    elif variant == "direct_prediction":
        cfg["model"]["prediction_mode"] = "direct"
        # Residual-specific mechanisms are undefined for a model that predicts
        # the complete methylation logit rather than a correction to a prior.
        cfg["model"]["gate"]["kind"] = "none"
        cfg["model"]["anchor_to_mean_rna"] = False
        cfg["model"]["zero_init_residual"] = False
        # This is the only baseline loss term that would explicitly pull the
        # direct predictor toward the frozen prior. residual_huber stays valid:
        # prior terms cancel between direct_logit-prior_logit and target_delta.
        cfg.setdefault("loss", {})["shrinkage_weight"] = 0.0
    elif variant == "confirm_rna256_no_gate_no_anchor":
        cfg["model"]["encoder"]["latent_dim"] = 256
        cfg["model"]["gate"]["kind"] = "none"
        cfg["model"]["anchor_to_mean_rna"] = False
    else:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    cfg.setdefault("training", {})["checkpoint_metric"] = checkpoint_metric
    # Use the complete nested-dev CpG panel for stable MAS-PCC model selection.
    cfg["training"]["validation_max_cpgs"] = None
    cfg["tracking"] = {
        "backend": "none",
        "project": "MethylationPredictor",
        "group": "architecture-ablations-array-chr1",
        "name": run_name,
        "job_type": "train",
        "tags": ["architecture-ablation", "tcga-array-chr1", variant],
        "mode": "disabled",
        "log_every_steps": 25,
        "watch_model": False,
        "log_checkpoint": False,
    }
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", default="configs/train.yaml")
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--output", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--run-output-dir", required=True)
    p.add_argument(
        "--checkpoint-metric",
        default="mas_pcc",
        choices=("mse", "mae", "mas_pcc", "mac_pcc", "skill_vs_prior"),
    )
    args = p.parse_args()

    rendered = render(
        yaml.safe_load(Path(args.base_config).read_text()),
        args.variant,
        run_name=args.run_name,
        output_dir=args.run_output_dir,
        checkpoint_metric=args.checkpoint_metric,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rendered, sort_keys=False))
    print(path)


if __name__ == "__main__":
    main()
