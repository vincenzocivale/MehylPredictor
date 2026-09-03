#!/usr/bin/env python3
"""Evaluate either model across any supported genomic scope.

RNA-methylation checkpoints from the reference shared-backbone architecture
(FeatureFusionLocusCLSModel, trained via --engine matched_chr1_shared_backbone
in scripts/train.py) need --engine matched_chr1_shared_backbone here too --
see docs/RNA_METHYLATION.md.
"""
from __future__ import annotations
import argparse, json

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",choices=["rna_methylation","cpg_statistics","cpg_prior"],required=True); p.add_argument("--checkpoint"); p.add_argument("--eval-scope",choices=["chr1","chr123","genomewide"],required=True); p.add_argument("--output",required=True); p.add_argument("--registry",required=True); p.add_argument("--canonical-root"); p.add_argument("--feature-cache"); p.add_argument("--rna-cache"); p.add_argument("--targets"); p.add_argument("--embeddings"); p.add_argument("--sample-chunk",type=int,default=128); p.add_argument("--cpg-chunk",type=int,default=2048); p.add_argument("--engine",choices=["generic","matched_chr1_shared_backbone"],default="generic"); p.add_argument("--prepared-root",help="matched_chr1_shared_backbone only: matched_chr1 data root"); p.add_argument("--cpg-targets-dir",help="matched_chr1_shared_backbone only: cpg_statistics targets dir"); p.add_argument("--recipe",default="configs/models/rna_methylation_shared_backbone.yaml",help="matched_chr1_shared_backbone only"); args=p.parse_args()
    if args.model=="rna_methylation" and args.engine=="matched_chr1_shared_backbone":
        from methylation_predictor.rna_training.locus_cls_trainer import evaluate_official_split
        if args.checkpoint is None: p.error("--checkpoint is required for RNA evaluation")
        for name in ("canonical_root","feature_cache","rna_cache","prepared_root","cpg_targets_dir"):
            if getattr(args,name) is None: p.error(f"--{name.replace('_','-')} is required for matched_chr1_shared_backbone evaluation")
        result=evaluate_official_split(checkpoint=args.checkpoint,canonical_root=args.canonical_root,recipe_path=args.recipe,feature_cache=args.feature_cache,rna_cache=args.rna_cache,registry=args.registry,matched_chr1_root=args.prepared_root,cpg_targets_dir=args.cpg_targets_dir,output=args.output,sample_chunk=args.sample_chunk,cpg_chunk=args.cpg_chunk)
    elif args.model=="rna_methylation":
        from methylation_predictor.rna_training.evaluator import ScopedRNAEvaluator
        if args.checkpoint is None: p.error("--checkpoint is required for RNA evaluation")
        for name in ("canonical_root","feature_cache","rna_cache"):
            if getattr(args,name) is None: p.error(f"--{name.replace('_','-')} is required for RNA evaluation")
        evaluator=ScopedRNAEvaluator(canonical_root=args.canonical_root,checkpoint=args.checkpoint,feature_cache=args.feature_cache,rna_cache=args.rna_cache,registry=args.registry,eval_scope=args.eval_scope,output=args.output,sample_chunk=args.sample_chunk,cpg_chunk=args.cpg_chunk)
        try: result=evaluator.run()
        finally: evaluator.close()
    elif args.model=="cpg_prior":
        # Zero-parameter "CpG Prior" baseline (docs/PAPER_EXPERIMENTS.md): predicts the
        # cached locus prior mu directly, no checkpoint/RNA input needed.
        from methylation_predictor.rna_training.evaluator import CpGPriorEvaluator
        for name in ("canonical_root","feature_cache"):
            if getattr(args,name) is None: p.error(f"--{name.replace('_','-')} is required for cpg_prior evaluation")
        evaluator=CpGPriorEvaluator(canonical_root=args.canonical_root,feature_cache=args.feature_cache,registry=args.registry,eval_scope=args.eval_scope,output=args.output,sample_chunk=args.sample_chunk,cpg_chunk=args.cpg_chunk)
        try: result=evaluator.run()
        finally: evaluator.close()
    else:
        if args.checkpoint is None: p.error("--checkpoint is required for statistics evaluation")
        from methylation_predictor.cpg_statistics.evaluator import evaluate_statistics_checkpoint
        if not args.targets or not args.embeddings: p.error("--targets and --embeddings are required for statistics evaluation")
        result=evaluate_statistics_checkpoint(checkpoint=args.checkpoint,targets_dir=args.targets,embeddings_h5=args.embeddings,registry=args.registry,eval_scope=args.eval_scope,output=args.output)
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
