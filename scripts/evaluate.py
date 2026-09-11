#!/usr/bin/env python3
"""Evaluate methylation models across supported genomic scopes.

RNA evaluation defaults to the paper-facing functional-locus reference recipe
(``configs/models/main.yaml``). Historical recipes can still be supplied
explicitly while benchmark/comparator migration is in progress.
"""
from __future__ import annotations
import argparse, json

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",choices=["rna_methylation","cpg_statistics","cpg_prior"],required=True); p.add_argument("--checkpoint"); p.add_argument("--eval-scope",choices=["chr1","chr123","genomewide"],required=True); p.add_argument("--output",required=True); p.add_argument("--registry",required=True); p.add_argument("--canonical-root"); p.add_argument("--feature-cache"); p.add_argument("--rna-cache"); p.add_argument("--functional-atlas"); p.add_argument("--annotation-cache"); p.add_argument("--functional-only",action="store_true"); p.add_argument("--targets"); p.add_argument("--embeddings"); p.add_argument("--sample-chunk",type=int,default=128); p.add_argument("--cpg-chunk",type=int,default=2048); p.add_argument("--engine",choices=["matched_chr1_shared_backbone","shared_backbone"],default="shared_backbone"); p.add_argument("--prepared-root",help="matched chr1 data root; omit for canonical chr123"); p.add_argument("--cpg-targets-dir",help="shared-backbone cpg_statistics targets dir"); p.add_argument("--recipe",default="configs/models/main.yaml",help="RNA model recipe; defaults to the paper-facing reference configuration"); args=p.parse_args()
    if args.model=="rna_methylation":
        from methylation_predictor.rna_training.locus_cls_trainer import evaluate_official_split
        if args.checkpoint is None: p.error("--checkpoint is required for RNA evaluation")
        for name in ("canonical_root","feature_cache","rna_cache","cpg_targets_dir"):
            if getattr(args,name) is None: p.error(f"--{name.replace('_','-')} is required for RNA evaluation")
        if args.eval_scope == "chr1" and args.prepared_root is None: p.error("--prepared-root is required for matched chr1 evaluation")
        result=evaluate_official_split(checkpoint=args.checkpoint,canonical_root=args.canonical_root,recipe_path=args.recipe,feature_cache=args.feature_cache,rna_cache=args.rna_cache,registry=args.registry,matched_chr1_root=args.prepared_root,cpg_targets_dir=args.cpg_targets_dir,output=args.output,scope=args.eval_scope,sample_chunk=args.sample_chunk,cpg_chunk=args.cpg_chunk,functional_atlas=args.functional_atlas,annotation_cache=args.annotation_cache,functional_only=args.functional_only)
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
