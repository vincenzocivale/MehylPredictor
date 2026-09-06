#!/usr/bin/env python3
"""Single training entrypoint for CpGStatisticsPredictor and the reference
RNA-methylation model.

RNA-methylation training always uses the single-stage shared-backbone engine
(FeatureFusionLocusCLSModel/FeatureFusionArchitectureVariantModel -- see
docs/RNA_METHYLATION.md). ``--engine matched_chr1_shared_backbone`` additionally
consumes the exact MethylProphet-matched chr1 preparation (``--prepared-root``);
``--engine shared_backbone`` is the scope-agnostic form used for chr123/genomewide.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml


def _rna(args):
    from methylation_predictor.rna_training.locus_cls_trainer import LocusCLSJointTrainer
    if args.scope == "chr1" and not args.prepared_root:
        raise ValueError("matched_chr1_shared_backbone requires --prepared-root (matched_chr1 data root)")
    if not args.cpg_targets_dir:
        raise ValueError("shared-backbone training requires --cpg-targets-dir (cpg_statistics targets, for the mean-branch proxy task)")
    recipe_raw = yaml.safe_load(Path(args.recipe).read_text()) or {}
    lc = recipe_raw.get("locus_cls", {})
    overrides = {k: v for k, v in {"learning_rate": args.lr, "epochs": args.epochs, "scheduler": args.scheduler, "seed": args.seed}.items() if v is not None}
    trainer = LocusCLSJointTrainer(
        canonical_root=args.canonical_root, scope=args.scope, recipe_path=args.recipe,
        feature_cache=args.feature_cache, rna_cache=args.rna_cache, registry=args.registry,
        cpg_targets_dir=args.cpg_targets_dir, output_root=args.output_root,
        matched_chr1_root=args.prepared_root,
        trunk_hidden_dim=lc.get("trunk_hidden_dim", 256), bottleneck_dim=lc.get("bottleneck_dim", 64),
        trunk_dropout=lc.get("trunk_dropout", 0.1),
        use_mean_branch=lc.get("use_mean_branch", True), use_fusion_product=lc.get("use_fusion_product", False),
        use_raw_product=lc.get("use_raw_product", True), product_mlp=lc.get("product_mlp", False),
        include_raw_rna=lc.get("include_raw_rna", True), include_raw_cpg=lc.get("include_raw_cpg", True),
        query_source=args.query_source or lc.get("query_source", "ntv3"),
        fusion_init_std=lc.get("fusion_init_std", 0.01), aux_weight=lc.get("aux_weight", 0.15),
        residual_aux_weight=lc.get("residual_aux_weight", 0.0), raw_lr_multiplier=lc.get("raw_lr_multiplier", 1.0),
        mode=args.mode, early_stop_patience=args.early_stop_patience, run_id=args.run_id,
        overrides=overrides or None, development_split_seed=args.development_split_seed, resume=args.resume,
    )
    try:
        return trainer.run()
    finally:
        trainer.close()


def _stats(args):
    if args.resume:
        raise ValueError("--resume is currently supported for RNA runs; start a fresh cpg_statistics run")
    from methylation_predictor.cpg_statistics.trainer import CpGStatisticsTrainer
    # CLI overrides are applied through a temporary resolved recipe next to output root.
    raw=yaml.safe_load(Path(args.recipe).read_text()) or {}; tr=raw.setdefault("training",{});
    for k,v in (("learning_rate",args.lr),("epochs",args.epochs),("scheduler",args.scheduler),("seed",args.seed)):
        if v is not None: tr[k]=v
    tmp=Path(args.output_root)/".resolved_cpg_statistics_recipe.yaml"; tmp.parent.mkdir(parents=True,exist_ok=True); tmp.write_text(yaml.safe_dump(raw,sort_keys=False)); trainer=CpGStatisticsTrainer(targets_dir=args.targets,embeddings_h5=args.embeddings,registry=args.registry,recipe_path=tmp,scope=args.scope,output_root=args.output_root,run_id=args.run_id); return trainer.run()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",choices=["rna_methylation","cpg_statistics"],required=True); p.add_argument("--scope",choices=["chr1","chr123","genomewide"],required=True); p.add_argument("--recipe",required=True); p.add_argument("--output-root",required=True); p.add_argument("--run-id",default=None); p.add_argument("--resume",action="store_true"); p.add_argument("--lr",type=float,default=None); p.add_argument("--epochs",type=int,default=None); p.add_argument("--scheduler",choices=["constant","cosine","cosine_warmup"],default=None); p.add_argument("--seed",type=int,default=None); p.add_argument("--registry",required=True); p.add_argument("--canonical-root"); p.add_argument("--feature-cache"); p.add_argument("--rna-cache"); p.add_argument("--mode",choices=["development","final"],default="final"); p.add_argument("--engine",choices=["matched_chr1_shared_backbone","shared_backbone"],default="shared_backbone"); p.add_argument("--prepared-root"); p.add_argument("--targets"); p.add_argument("--embeddings"); p.add_argument("--cpg-targets-dir",help="cpg_statistics targets dir for the mean-branch proxy task"); p.add_argument("--early-stop-patience",type=int,default=None,help="stop once the training signal hasn't improved for this many epochs"); p.add_argument("--query-source",choices=["ntv3","mean_only","hybrid_detached","hybrid_joint"],default=None,help="locus-attention query representation; experimental, default preserves recipe/reference"); p.add_argument("--development-split-seed",type=int,default=None,help="fix the inner development split independently of the training seed"); args=p.parse_args()
    if args.model=="rna_methylation":
        for name in ("canonical_root","feature_cache","rna_cache"):
            if getattr(args,name) is None: p.error(f"--{name.replace('_','-')} is required for RNA training")
        result=_rna(args)
    else:
        if not args.targets or not args.embeddings: p.error("--targets and --embeddings are required for CpG statistics training")
        result=_stats(args)
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
