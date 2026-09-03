"""One-off entry point for the single-stage joint-training ablation.

Not part of the stable CLI (scripts/{prepare,train,tune,evaluate}.py) -- delete
this script and src/methylation_predictor/rna_training/joint_trainer.py once
the ablation concludes. See the "joint training ablation" plan this was
written for and results/reference/ablations.yaml for where results (if
conclusive) get recorded.

Two variants (``--variant``):
  finetune - CpGStatisticsPredictor starts from a trained stage-1 checkpoint
             (``--cpg-checkpoint``) and is fine-tuned jointly with a low
             ``--cpg-lr``.
  scratch  - CpGStatisticsPredictor starts from random init and is trained
             jointly from scratch at ``--cpg-lr`` (defaults to the RNA
             model's own learning_rate from the recipe).

Also supports ``--variant baseline``: the canonical two-stage *frozen*-prior
architecture, to produce a same-engine control number for the other two
variants -- the existing matched_chr1-engine ablation numbers in
ablations.yaml are not numerically comparable to a generic-engine run (see
``fusion_mechanism_2026_08``'s note in that file). This does NOT use the
generic ``ScopedRNATrainer``: at chr1, ``rna_feature_cache/chr1`` only covers
the official Array CpGs, not the full ``tcga_mix_chr1`` multi-technology
universe ``ScopedRNATrainer.__init__`` requires regardless of which pools are
actually used (same note) -- so it is impossible to construct a same-engine,
apples-to-apples baseline that way at chr1. Instead ``baseline`` routes
through ``JointRNAMethylationTrainer`` with ``cpg_learning_rate=0.0`` (frozen
mu/sigma from ``--cpg-checkpoint``, ``requires_grad_(False)``) and
``--joint-loss-weight 0`` -- mathematically identical to the two-stage
frozen-prior architecture, on the same array-only pool infrastructure the
other two variants use, which keeps all three runs on one truly comparable
engine.

Example:
    python scripts/experiments/train_joint_ablation.py --variant finetune \
        --canonical-root /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1 \
        --feature-cache /dune/DATASETS/MethylPredictionData/derived/rna_feature_cache/chr1 \
        --rna-cache /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1/rna \
        --registry /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet \
        --cpg-targets-dir /dune/DATASETS/MethylPredictionData/derived/cpg_statistics/chr1 \
        --cpg-checkpoint /dune/DATASETS/MethylPredictionData/experiments/runs/cpg_statistics/chr1/cpg-statistics-chr1-retrain-2026-08-28/checkpoints/best.pt \
        --recipe configs/models/rna_methylation.yaml \
        --output-root /dune/DATASETS/MethylPredictionData/experiments/runs \
        --mode development --run-id joint-finetune-chr1-smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from methylation_predictor.rna_training.joint_trainer import JointRNAMethylationTrainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["finetune", "scratch", "baseline"])
    ap.add_argument("--canonical-root", required=True)
    ap.add_argument("--scope", default="chr1")
    ap.add_argument("--recipe", default="configs/models/rna_methylation.yaml")
    ap.add_argument("--feature-cache", required=True)
    ap.add_argument("--rna-cache", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument(
        "--matched-chr1-root",
        help="scope=chr1 only: read the official MethylProphet-matched split/data "
        "(derived/methylprophet_table5_tcga_chr1) instead of scope_protocol()'s own "
        "tcga_mix_chr1 reconstruction -- exact split parity + the pre-extracted, "
        "faster-chunked array/epic files matched_chr1 uses.",
    )
    ap.add_argument("--cpg-targets-dir")
    ap.add_argument("--cpg-checkpoint", help="required for --variant finetune")
    ap.add_argument("--cpg-lr", type=float, default=None, help="default: 1e-5 (finetune) / recipe learning_rate (scratch)")
    ap.add_argument("--joint-loss-weight", type=float, default=1.0)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--mode", default="development", choices=["development", "final"])
    ap.add_argument("--run-id")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    overrides = {"epochs": args.epochs} if args.epochs is not None else None

    if args.cpg_targets_dir is None:
        raise SystemExit("--cpg-targets-dir is required for every variant")
    if args.variant in ("finetune", "baseline") and not args.cpg_checkpoint:
        raise SystemExit("--cpg-checkpoint is required for --variant finetune/baseline")

    import yaml
    recipe_lr = float((yaml.safe_load(Path(args.recipe).read_text()) or {}).get("training", {}).get("learning_rate", 5e-5))
    if args.variant == "baseline":
        cpg_lr = 0.0
        joint_loss_weight = 0.0
    else:
        cpg_lr = args.cpg_lr if args.cpg_lr is not None else (1e-5 if args.variant == "finetune" else recipe_lr)
        joint_loss_weight = args.joint_loss_weight

    trainer = JointRNAMethylationTrainer(
        canonical_root=args.canonical_root, scope=args.scope, recipe_path=args.recipe,
        feature_cache=args.feature_cache, rna_cache=args.rna_cache, registry=args.registry,
        cpg_targets_dir=args.cpg_targets_dir, output_root=args.output_root,
        matched_chr1_root=args.matched_chr1_root,
        cpg_warm_start_checkpoint=args.cpg_checkpoint if args.variant in ("finetune", "baseline") else None,
        cpg_learning_rate=cpg_lr, joint_loss_weight=joint_loss_weight,
        mode=args.mode, run_id=args.run_id, overrides=overrides,
    )
    summary = trainer.run()
    trainer.close()
    print(summary)


if __name__ == "__main__":
    main()
