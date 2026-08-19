#!/usr/bin/env python3
"""Evaluate either model across any supported genomic scope."""
from __future__ import annotations
import argparse, json

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",choices=["rna_methylation","cpg_statistics"],required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--eval-scope",choices=["chr1","chr123","genomewide"],required=True); p.add_argument("--output",required=True); p.add_argument("--registry",required=True); p.add_argument("--canonical-root"); p.add_argument("--feature-cache"); p.add_argument("--rna-cache"); p.add_argument("--targets"); p.add_argument("--embeddings"); p.add_argument("--sample-chunk",type=int,default=128); p.add_argument("--cpg-chunk",type=int,default=2048); args=p.parse_args()
    if args.model=="rna_methylation":
        from methylation_predictor.rna_training.evaluator import ScopedRNAEvaluator
        for name in ("canonical_root","feature_cache","rna_cache"):
            if getattr(args,name) is None: p.error(f"--{name.replace('_','-')} is required for RNA evaluation")
        evaluator=ScopedRNAEvaluator(canonical_root=args.canonical_root,checkpoint=args.checkpoint,feature_cache=args.feature_cache,rna_cache=args.rna_cache,registry=args.registry,eval_scope=args.eval_scope,output=args.output,sample_chunk=args.sample_chunk,cpg_chunk=args.cpg_chunk)
        try: result=evaluator.run()
        finally: evaluator.close()
    else:
        from methylation_predictor.cpg_statistics.evaluator import evaluate_statistics_checkpoint
        if not args.targets or not args.embeddings: p.error("--targets and --embeddings are required for statistics evaluation")
        result=evaluate_statistics_checkpoint(checkpoint=args.checkpoint,targets_dir=args.targets,embeddings_h5=args.embeddings,registry=args.registry,eval_scope=args.eval_scope,output=args.output)
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
