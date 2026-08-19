#!/usr/bin/env python3
"""Build multi-technology CpG mean/sigma labels for chr1, chr123 or genome-wide."""
from __future__ import annotations
import argparse, json
from methylation_predictor.cpg_statistics.targets import build_statistics_targets

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--canonical-root",required=True); p.add_argument("--registry",required=True); p.add_argument("--scope",choices=["chr1","chr123","genomewide"],required=True); p.add_argument("--output",required=True); p.add_argument("--policy",choices=["sample_weighted","technology_balanced"],default="sample_weighted"); p.add_argument("--aux-sample-policy",choices=["exclude_array_validation","all_auxiliary"],default="exclude_array_validation"); p.add_argument("--sources",nargs="+",default=["array","epic","wgbs"]); p.add_argument("--cpg-chunk",type=int,default=2048); args=p.parse_args(); result=build_statistics_targets(canonical_root=args.canonical_root,registry=args.registry,scope=args.scope,output=args.output,policy=args.policy,aux_sample_policy=args.aux_sample_policy,sources=tuple(args.sources),cpg_chunk=args.cpg_chunk); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
