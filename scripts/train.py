#!/usr/bin/env python3
"""Training entrypoint for methylation models.

The paper-facing RNA model is selected by ``configs/models/main.yaml`` and uses
a functional CpG representation plus locus-conditioned retrieval from learned
RNA program tokens. Functional locus inputs are mandatory for RNA training.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml


def _rna(args):
    from methylation_predictor.rna_training import RNAMethylationTrainer
    if args.scope == "chr1" and not args.prepared_root:
        raise ValueError("chr1 RNA training requires --prepared-root (matched protocol data root)")
    if not args.cpg_targets_dir:
        raise ValueError("RNA training requires --cpg-targets-dir (targets for the training-only mean-proxy task)")
    from methylation_predictor.rna_training.config import load_rna_recipe
    recipe_raw = load_rna_recipe(args.recipe).raw
    lc = recipe_raw.get("locus_cls", {})
    overrides = {k: v for k, v in {"learning_rate": args.lr, "epochs": args.epochs, "scheduler": args.scheduler, "seed": args.seed}.items() if v is not None}
    if args.training_sources:
        training_sources = tuple(x.strip() for x in args.training_sources.split(",") if x.strip())
    elif recipe_raw.get("training_sources"):
        training_sources = tuple(recipe_raw["training_sources"])
    else:
        training_sources = None
    trainer = RNAMethylationTrainer(
        canonical_root=args.canonical_root, scope=args.scope, recipe_path=args.recipe,
        rna_cache=args.rna_cache, prior_cache=args.prior_cache, registry=args.registry,
        cpg_targets_dir=args.cpg_targets_dir, output_root=args.output_root,
        matched_chr1_root=args.prepared_root,
        use_mean_branch=lc.get("use_mean_branch", True),
        aux_weight=lc.get("aux_weight", 0.15),
        mode=args.mode, early_stop_patience=args.early_stop_patience, run_id=args.run_id,
        overrides=overrides or None, development_split_seed=args.development_split_seed, resume=args.resume,
        training_sources=training_sources,
        functional_atlas=args.functional_atlas, annotation_cache=args.annotation_cache,
        bigwig_cache=args.bigwig_cache,
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
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",choices=["rna_methylation","cpg_statistics"],required=True); p.add_argument("--scope",choices=["chr1","chr123","genomewide"],required=True); p.add_argument("--recipe",required=True); p.add_argument("--output-root",required=True); p.add_argument("--run-id",default=None); p.add_argument("--resume",action="store_true"); p.add_argument("--lr",type=float,default=None); p.add_argument("--epochs",type=int,default=None); p.add_argument("--scheduler",choices=["constant","cosine","cosine_warmup"],default=None); p.add_argument("--seed",type=int,default=None); p.add_argument("--registry",required=True); p.add_argument("--canonical-root"); p.add_argument("--prior-cache",help="optional metric-only locus prior cache (cpg_idx.npy + prior.npy)"); p.add_argument("--rna-cache"); p.add_argument("--functional-atlas"); p.add_argument("--annotation-cache"); p.add_argument("--bigwig-cache",help="optional complete BigWig PCA context cache"); p.add_argument("--mode",choices=["development","final"],default="final"); p.add_argument("--prepared-root"); p.add_argument("--targets"); p.add_argument("--embeddings"); p.add_argument("--cpg-targets-dir",help="cpg_statistics targets dir for the mean-branch proxy task"); p.add_argument("--early-stop-patience",type=int,default=None,help="stop once the training signal hasn't improved for this many epochs"); p.add_argument("--development-split-seed",type=int,default=None,help="fix the inner development split independently of the training seed"); p.add_argument("--training-sources",default=None,help="comma-separated subset of array,epic,wgbs to restrict training to (paper section B.6 source ablation); array is always included; default is all three, unchanged from before this flag existed"); args=p.parse_args()
    if args.model=="rna_methylation":
        for name in ("canonical_root","rna_cache","functional_atlas","annotation_cache"):
            if getattr(args,name) is None: p.error(f"--{name.replace('_','-')} is required for RNA training")
        result=_rna(args)
    else:
        if not args.targets or not args.embeddings: p.error("--targets and --embeddings are required for CpG statistics training")
        result=_stats(args)
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
