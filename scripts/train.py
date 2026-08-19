#!/usr/bin/env python3
"""Single training entrypoint for CpGStatisticsPredictor and RNAMethylationPredictor."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import yaml


def _rna(args):
    from methylation_predictor.rna_training.trainer import ScopedRNATrainer
    overrides={k:v for k,v in {"learning_rate":args.lr,"epochs":args.epochs,"scheduler":args.scheduler,"seed":args.seed}.items() if v is not None}
    if args.engine=="matched_chr1":
        if args.scope!="chr1": raise ValueError("matched_chr1 engine is valid only for --scope chr1")
        if not args.prepared_root: raise ValueError("matched_chr1 requires --prepared-root")
        # Preserve the numerically validated E1 path while sharing the refactored
        # recipe's architecture/loss/training values.
        from methylation_predictor.benchmark.methylprophet.trainer import MethylProphetTrainer
        from methylation_predictor.run_store import RunStore
        recipe=yaml.safe_load(Path(args.recipe).read_text()) or {}; base=yaml.safe_load(Path(args.matched_base_config).read_text()) or {}; base["model"]=copy.deepcopy(recipe["model"]); base["loss"]=copy.deepcopy(recipe["loss"]); base["training"]={**base.get("training",{}),**copy.deepcopy(recipe.get("training",{})),**overrides}; base["tracking"]={**base.get("tracking",{}),**copy.deepcopy(recipe.get("tracking",{}))}
        cfg=base["training"]; store=RunStore.create(args.output_root,model="rna_methylation",train_scope="chr1",seed=int(cfg.get("seed",17)),learning_rate=float(cfg["learning_rate"]),scheduler=str(cfg.get("scheduler","constant")),epochs=int(cfg["epochs"]),run_id=args.run_id,resume=args.resume); resolved=store.path/"config.resolved.yaml"
        if store.is_new:
            resolved.write_text(yaml.safe_dump(base,sort_keys=False)); store.save_metadata({"mode":args.mode,"engine":"matched_chr1","prepared_root":args.prepared_root,"training":cfg})
        else:
            if (yaml.safe_load(resolved.read_text()) or {}) != base: raise RuntimeError("resume requested with a different matched chr1 config")
            if not (store.path/"checkpoints"/"latest.pt").is_file(): raise RuntimeError("resume requested but checkpoints/latest.pt is missing")
        batching=recipe.get("batching",{}); br={k:int(v["sample_size"]) for k,v in batching.items()}; bc={k:int(v["cpg_size"]) for k,v in batching.items()}; prep=Path(args.prepared_root); trainer=MethylProphetTrainer(canonical_root=args.canonical_root,config_path=resolved,protocol_root=prep/"table5_protocol",feature_cache=args.feature_cache,rna_cache=args.rna_cache,array_cache=prep/"methylation"/"array_table5_chr1.h5",epic_cache=prep/"methylation"/"epic_table5_chr1.h5",output_dir=store.path,epochs=int(cfg["epochs"]),seed=int(cfg.get("seed",17)),block_rows=br,block_cpgs=bc,structured_loss_sources=set(recipe.get("structured_loss_sources",["array","epic","wgbs"])))
        try:
            result=trainer.run()
            from methylation_predictor.run_store import write_json
            history_path=store.path/"metrics"/"history.json"
            if history_path.is_file():
                write_json(store.training_file("history.json"),json.loads(history_path.read_text()))
            checkpoints=store.path/"checkpoints"
            for alias,target in (("best.pt","final.pt"),("last.pt","latest.pt")):
                alias_path=checkpoints/alias
                if not alias_path.exists() and (checkpoints/target).exists():
                    alias_path.symlink_to(target)
            write_json(store.training_file("summary.json"),{"engine":"matched_chr1","scope":"chr1","epochs":int(cfg["epochs"]),"evaluation":result})
            eval_dir=store.evaluation_dir("chr1"); write_json(eval_dir/"metrics.json",result); store.write_evaluation_manifest(eval_scope="chr1",checkpoint=checkpoints/"final.pt",dataset_contract="MethylProphet-matched TCGA chr1 official views")
            return {"run_dir":str(store.path),"evaluation":result}
        finally: trainer.close()
    trainer=ScopedRNATrainer(canonical_root=args.canonical_root,scope=args.scope,recipe_path=args.recipe,feature_cache=args.feature_cache,rna_cache=args.rna_cache,registry=args.registry,output_root=args.output_root,mode=args.mode,run_id=args.run_id,overrides=overrides,resume=args.resume)
    try: return trainer.run()
    finally: trainer.close()


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
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",choices=["rna_methylation","cpg_statistics"],required=True); p.add_argument("--scope",choices=["chr1","chr123","genomewide"],required=True); p.add_argument("--recipe",required=True); p.add_argument("--output-root",required=True); p.add_argument("--run-id",default=None); p.add_argument("--resume",action="store_true"); p.add_argument("--lr",type=float,default=None); p.add_argument("--epochs",type=int,default=None); p.add_argument("--scheduler",choices=["constant","cosine","cosine_warmup"],default=None); p.add_argument("--seed",type=int,default=None); p.add_argument("--registry",required=True); p.add_argument("--canonical-root"); p.add_argument("--feature-cache"); p.add_argument("--rna-cache"); p.add_argument("--mode",choices=["development","final"],default="final"); p.add_argument("--engine",choices=["generic","matched_chr1"],default="generic"); p.add_argument("--prepared-root"); p.add_argument("--matched-base-config",default="configs/benchmark_methylprophet/reference.yaml"); p.add_argument("--targets"); p.add_argument("--embeddings"); args=p.parse_args()
    if args.model=="rna_methylation":
        for name in ("canonical_root","feature_cache","rna_cache"):
            if getattr(args,name) is None: p.error(f"--{name.replace('_','-')} is required for RNA training")
        result=_rna(args)
    else:
        if not args.targets or not args.embeddings: p.error("--targets and --embeddings are required for CpG statistics training")
        result=_stats(args)
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
