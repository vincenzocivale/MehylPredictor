#!/usr/bin/env python3
"""Export predicted mu/sigma + NTv3 embeddings for RNA training/evaluation."""
from __future__ import annotations
import argparse, json
from methylation_predictor.cpg_statistics.export import export_feature_cache

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",required=True); p.add_argument("--targets",required=True); p.add_argument("--embeddings",required=True); p.add_argument("--output",required=True); p.add_argument("--predicted-for-train",action="store_true",help="use model predictions instead of empirical leakage-safe statistics on official train CpGs"); args=p.parse_args(); result=export_feature_cache(checkpoint=args.checkpoint,targets_dir=args.targets,embeddings_h5=args.embeddings,output=args.output,empirical_for_train=not args.predicted_for_train); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
