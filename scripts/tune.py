#!/usr/bin/env python3
"""Leakage-safe LR/scheduler/epoch search for either canonical model."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import time
import yaml


def _parse_floats(text): return [float(x) for x in text.split(",") if x]
def _parse_strings(text): return [x.strip() for x in text.split(",") if x.strip()]


def _rna(args):
    from methylation_predictor.rna_training.tuning import run_search
    return run_search(canonical_root=args.canonical_root,scope=args.scope,recipe_path=args.recipe,feature_cache=args.feature_cache,rna_cache=args.rna_cache,registry=args.registry,output_root=args.output_root,learning_rates=_parse_floats(args.lrs),schedulers=_parse_strings(args.schedulers),max_epochs=args.max_epochs,seed=args.seed,search_id=args.search_id)


def _stats(args):
    from methylation_predictor.cpg_statistics.trainer import CpGStatisticsTrainer
    from methylation_predictor.run_store import SearchStore, write_json, write_yaml
    store=SearchStore.create(args.output_root,model="cpg_statistics",scope=args.scope,search_id=args.search_id); base=yaml.safe_load(Path(args.recipe).read_text()) or {}; candidates=[]
    write_yaml(store.path/"search_config.yaml",{"scope":args.scope,"learning_rates":_parse_floats(args.lrs),"schedulers":_parse_strings(args.schedulers),"max_epochs":args.max_epochs,"seed":args.seed,"selection_metric":base.get("training",{}).get("selection_metric","joint_loss")})
    for lr,scheduler in itertools.product(_parse_floats(args.lrs),_parse_strings(args.schedulers)):
        raw=yaml.safe_load(Path(args.recipe).read_text()) or {}; tr=raw.setdefault("training",{}); tr.update({"learning_rate":lr,"scheduler":scheduler,"epochs":args.max_epochs,"seed":args.seed}); cid=f"lr{lr:.2g}-{scheduler}-e{args.max_epochs}".replace("+",""); recipe=store.path/f"{cid}.yaml"; recipe.write_text(yaml.safe_dump(raw,sort_keys=False)); started=time.time(); trainer=CpGStatisticsTrainer(targets_dir=args.targets,embeddings_h5=args.embeddings,registry=args.registry,recipe_path=recipe,scope=args.scope,output_root=store.path/"runs",run_id=cid,nested_run_store=False); result=trainer.run(); row={"learning_rate":lr,"scheduler":scheduler,"max_epochs":args.max_epochs,"warmup_epochs":float(tr.get("warmup_epochs",0.0)),"min_lr_ratio":float(tr.get("min_lr_ratio",0.1)),"best_epoch":result["best_epoch"],"best_development_value":result["best_development_value"],"selection_metric":result["selection_metric"],"elapsed_seconds":time.time()-started,"run_dir":result["run_dir"]}; candidates.append(row)
    candidates.sort(key=lambda x:x["best_development_value"])
    fields=["learning_rate","scheduler","max_epochs","warmup_epochs","min_lr_ratio","best_epoch","best_development_value","selection_metric","elapsed_seconds","run_dir"]
    with (store.path/"candidates.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(candidates)
    result={"selection_metric":candidates[0]["selection_metric"],"best_candidate":candidates[0],"all_results":candidates}; write_json(store.path/"selected.json",result); return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",choices=["rna_methylation","cpg_statistics"],required=True); p.add_argument("--scope",choices=["chr1","chr123","genomewide"],required=True); p.add_argument("--recipe",required=True); p.add_argument("--registry",required=True); p.add_argument("--output-root",required=True); p.add_argument("--lrs",required=True,help="comma-separated learning rates"); p.add_argument("--schedulers",default="constant",help="comma-separated scheduler names"); p.add_argument("--max-epochs",type=int,required=True); p.add_argument("--seed",type=int,default=17); p.add_argument("--search-id",default=None); p.add_argument("--canonical-root"); p.add_argument("--feature-cache"); p.add_argument("--rna-cache"); p.add_argument("--targets"); p.add_argument("--embeddings"); args=p.parse_args()
    if args.model=="rna_methylation":
        for name in ("canonical_root","feature_cache","rna_cache"):
            if getattr(args,name) is None: p.error(f"--{name.replace('_','-')} is required for RNA tuning")
        result=_rna(args)
    else:
        if not args.targets or not args.embeddings: p.error("--targets and --embeddings are required for CpG statistics tuning")
        result=_stats(args)
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
