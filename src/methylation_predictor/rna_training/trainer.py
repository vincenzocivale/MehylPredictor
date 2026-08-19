"""Scope-general mixed-source trainer for the canonical RNA methylation model.

The validated chr1 Table-5-compatible trainer remains available as a strict
compatibility engine.  This module is the general engine used for chr123 and
whole-genome training and can also run chr1 when exact legacy cache provenance
is not required.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from ..losses import residual_loss
from ..models import RNAMethylationPredictor
from ..optim import build_lr_scheduler
from ..run_store import RunStore, write_json
from ..scopes import scope_protocol
from ..storage import LocusFeatureCache, RNACache
from ..tcga_canonical import TCGACanonicalBundle
from .config import RNARecipe, load_rna_recipe
from .metrics import ArrayMomentMetrics
from .schedule import SourceSchedule, interleave
from .splits import blocked_cpg_split, stratified_sample_split

_STRUCTURED_LOSS_FIELDS = (
    "locus_pearson_weight", "locus_lower_tail_weight", "pairwise_difference_weight",
    "global_prior_ratio_weight", "locus_skill_weight", "locus_ccc_weight",
    "within_cancer_dynamic_weight", "centered_mse_weight", "amplitude_weight",
)


def loss_config_for_source(config, source_name: str, enabled: set[str]):
    if source_name in enabled:
        return config
    return replace(config, **{name: 0.0 for name in _STRUCTURED_LOSS_FIELDS})


@dataclass(slots=True)
class TrainingPool:
    name: str
    row_positions: np.ndarray
    sample_idx: np.ndarray
    cpg_idx: np.ndarray


class _Wandb:
    def __init__(self, recipe: RNARecipe, run_store: RunStore, *, scope: str, mode: str, resume_id: str | None = None):
        self.run = None
        if recipe.tracking.backend != "wandb" or recipe.tracking.mode == "disabled":
            return
        import wandb
        kwargs = dict(
            project=recipe.tracking.project,
            entity=recipe.tracking.entity,
            group=recipe.tracking.group or f"rna-{scope}-{mode}",
            name=recipe.tracking.name or run_store.run_id,
            job_type=mode,
            tags=[*recipe.tracking.tags, f"scope-{scope}", "rna-methylation"],
            mode=recipe.tracking.mode,
            dir=str(run_store.path),
            id=resume_id,
            resume="allow" if resume_id else None,
        )
        kwargs = {k:v for k,v in kwargs.items() if v is not None}
        self.run = wandb.init(**kwargs)

    def log(self, payload: dict, step: int | None = None):
        if self.run is not None:
            self.run.log(payload, step=step)

    def finish(self, summary: dict | None = None):
        if self.run is not None:
            if summary:
                self.run.summary.update(summary)
            self.run.finish(); self.run=None


class ScopedRNATrainer:
    def __init__(
        self,
        *,
        canonical_root: str | Path,
        scope: str,
        recipe_path: str | Path,
        feature_cache: str | Path,
        rna_cache: str | Path,
        registry: str | Path,
        output_root: str | Path,
        mode: str = "final",
        run_id: str | None = None,
        overrides: dict | None = None,
        nested_run_store: bool = True,
        resume: bool = False,
    ):
        if mode not in {"development", "final"}:
            raise ValueError("mode must be development or final")
        self.mode=mode; self.scope=scope; self.root=Path(canonical_root); self.registry=Path(registry)
        self.recipe=load_rna_recipe(recipe_path)
        for key,value in (overrides or {}).items():
            if not hasattr(self.recipe.training,key):
                raise ValueError(f"unknown TrainingConfig override {key!r}")
            setattr(self.recipe.training,key,value)
        cfg=self.recipe.training
        self.seed=int(cfg.seed); self.epochs=int(cfg.epochs)
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type!="cuda": raise RuntimeError("RNA training requires CUDA")
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        torch.backends.cuda.matmul.allow_tf32=cfg.allow_tf32; torch.backends.cudnn.allow_tf32=cfg.allow_tf32
        self.bundle=TCGACanonicalBundle.from_root(self.root)
        self.protocol=scope_protocol(scope,self.bundle,canonical_root=self.root)
        self.features=LocusFeatureCache(feature_cache); self.rna=RNACache(rna_cache)
        required=np.unique(np.concatenate([self.protocol.array_train_cpg_idx,self.protocol.array_val_cpg_idx,*self.protocol.auxiliary_cpg_idx.values()]))
        self.features.index.positions_of(required)
        self.model=RNAMethylationPredictor(25_017,1536,self.recipe.model,epsilon=1e-4).to(self.device)
        self.inner_views=None
        self.pools=self._build_pools()
        self.store=RunStore.create(output_root,model="rna_methylation",train_scope=scope,seed=self.seed,learning_rate=cfg.learning_rate,scheduler=cfg.scheduler,epochs=self.epochs,run_id=run_id,nested=nested_run_store,resume=resume)
        resolved_path=self.store.path/"config.resolved.yaml"
        if self.store.is_new:
            self.store.save_resolved_config(self.recipe.raw)
        else:
            import yaml as _yaml
            existing=_yaml.safe_load(resolved_path.read_text()) or {}
            if existing != self.recipe.raw:
                raise RuntimeError("resume requested with a different resolved recipe")
        resume_wandb_id = None
        if not self.store.is_new:
            existing_meta = json.loads((self.store.path / "metadata.json").read_text())
            resume_wandb_id = (existing_meta.get("wandb") or {}).get("run_id")
        self.wandb=_Wandb(self.recipe,self.store,scope=scope,mode=mode,resume_id=resume_wandb_id)
        if self.store.is_new:
            self.store.save_metadata({
                "mode":mode,
                "dataset_contract":"TCGA canonical methylprophet_repro_v1",
                "canonical_root":str(self.root),
                "sources":[p.name for p in self.pools],
                "chromosomes":list(self.protocol.chromosomes) if self.protocol.chromosomes else "genomewide",
                "schedule_policy":self.recipe.schedule_policy,
                "training":asdict(cfg),
                "batching":self.recipe.batching,
                "feature_cache":str(feature_cache), "rna_cache":str(rna_cache),
                "wandb": None if self.wandb.run is None else {"project": self.wandb.run.project, "run_id": self.wandb.run.id},
            })
        elif not self.store.checkpoint("last.pt").is_file():
            raise RuntimeError("resume requested but checkpoints/last.pt is missing")

    def close(self):
        try: self.wandb.finish()
        finally: self.bundle.close()

    def _build_pools(self) -> list[TrainingPool]:
        p=self.protocol
        if self.mode=="development":
            frac=float(self.recipe.raw.get("development",{}).get("fraction",0.1))
            block_bp=int(self.recipe.raw.get("development",{}).get("block_bp",5_000_000))
            train_s,val_s=stratified_sample_split(canonical_root=self.root,sample_ids=p.array_train_sample_idx,val_fraction=frac,seed=self.seed)
            train_c,val_c=blocked_cpg_split(registry=self.registry,cpg_ids=p.array_train_cpg_idx,val_fraction=frac,seed=self.seed,block_bp=block_bp)
            self.inner_views={
                "train_cpg_x_val_sample":(val_s,train_c),
                "val_cpg_x_train_sample":(train_s,val_c),
                "val_cpg_x_val_sample":(val_s,val_c),
            }
            array_rows=self.bundle.sources["array"].rows_of_samples(train_s)
            pools=[TrainingPool("array",array_rows,train_s,train_c)]
            forbidden=val_s
        else:
            array_rows=self.bundle.sources["array"].rows_of_samples(p.array_train_sample_idx)
            pools=[TrainingPool("array",array_rows,p.array_train_sample_idx,p.array_train_cpg_idx)]
            forbidden=p.array_val_sample_idx if self.recipe.exclude_official_val_from_auxiliary else np.empty(0,np.int64)

        for name in ("epic","wgbs"):
            if name not in p.sources: continue
            source=self.bundle.sources[name]
            rows=np.arange(source.n_rows,dtype=np.int64)
            if len(forbidden): rows=rows[~np.isin(source.sample_idx[rows],forbidden)]
            cpg=np.asarray(p.auxiliary_cpg_idx.get(name,[]),np.int64)
            if self.mode=="development":
                cpg=cpg[np.isin(cpg,pools[0].cpg_idx)]
            if len(rows) and len(cpg): pools.append(TrainingPool(name,rows,source.sample_idx[rows],cpg))
        return pools

    def _autocast(self):
        if not self.recipe.training.amp: return nullcontext()
        requested=self.recipe.training.amp_dtype.lower()
        dtype=torch.bfloat16 if requested=="bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda",dtype=dtype)

    def _schedules(self,epoch:int):
        schedules=[]
        for i,pool in enumerate(self.pools):
            batch=self.recipe.batching[pool.name]
            schedules.append(SourceSchedule(len(pool.row_positions),len(pool.cpg_idx),int(batch["sample_size"]),int(batch["cpg_size"]),epoch,self.seed+1009*i,self.recipe.schedule_policy))
        return schedules,interleave(schedules,seed=self.seed,epoch=epoch)

    def _read_block(self,pool:TrainingPool,row_slots,cpg_slots):
        rows=pool.row_positions[row_slots]; cpg=pool.cpg_idx[cpg_slots]
        source=self.bundle.sources[pool.name]
        beta=source.block(rows,cpg)
        return pool.sample_idx[row_slots],cpg,beta

    def _step(self,pool,sample_ids,cpg_ids,beta_np):
        finite=np.isfinite(beta_np)
        if not finite.any(): return None
        rna=torch.from_numpy(self.rna.rows(sample_ids,dtype=np.float16)).to(self.device).float()
        emb_np,prior_np,sigma_np=self.features.get(cpg_ids,embedding_dtype=np.float16)
        emb=torch.from_numpy(emb_np).to(self.device).float(); prior=torch.from_numpy(prior_np).to(self.device); sigma=torch.from_numpy(sigma_np).to(self.device); beta=torch.from_numpy(beta_np).to(self.device)
        with self._autocast():
            out=self.model(rna,emb,prior,sigma=sigma)
            cfg=loss_config_for_source(self.recipe.loss,pool.name,self.recipe.structured_loss_sources)
            loss,pieces=residual_loss(out,beta,prior,cfg,epsilon=1e-4,sigma=sigma)
            # Normalize sparse technologies by the fraction of finite pair slots.
            loss=loss*(float(finite.sum())/max(float(finite.size),1.0))
        return loss,pieces

    @torch.no_grad()
    def evaluate_view(self,sample_ids,cpg_ids,*,sample_chunk=128,cpg_chunk=2048):
        self.model.eval(); source=self.bundle.sources["array"]; rows=source.rows_of_samples(sample_ids); metrics=ArrayMomentMetrics(len(sample_ids),len(cpg_ids))
        for s0 in range(0,len(sample_ids),sample_chunk):
            s1=min(s0+sample_chunk,len(sample_ids)); local_s=sample_ids[s0:s1]
            rna=torch.from_numpy(self.rna.rows(local_s)).to(self.device)
            # MethylationSource.block is selective after the refactor; read each cpg chunk.
            for c0 in range(0,len(cpg_ids),cpg_chunk):
                c1=min(c0+cpg_chunk,len(cpg_ids)); local_c=cpg_ids[c0:c1]
                emb_np,prior_np,sigma_np=self.features.get(local_c)
                emb=torch.from_numpy(emb_np).to(self.device); prior=torch.from_numpy(prior_np).to(self.device); sigma=torch.from_numpy(sigma_np).to(self.device)
                with self._autocast(): pred=self.model(rna,emb,prior,sigma=sigma)["beta"]
                target=source.block(rows[s0:s1],local_c)
                metrics.add(s0,c0,target,pred.float().cpu().numpy(),prior_np)
        return metrics.finalize()

    def evaluate_development(self):
        if self.inner_views is None: raise RuntimeError("not a development run")
        return {name:self.evaluate_view(s,c) for name,(s,c) in self.inner_views.items()}

    def _save_checkpoint(self,path,optimizer,scheduler,scaler,epoch,history):
        payload={"schema_version":2,"model":"rna_methylation","scope":self.scope,"mode":self.mode,"epoch":epoch,"epochs_planned":self.epochs,"architecture":"variance_normalized_residual","model_state":self.model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"scaler_state":scaler.state_dict() if scaler else None,"model_config":asdict(self.recipe.model),"loss_config":asdict(self.recipe.loss),"training":asdict(self.recipe.training),"history":history}
        tmp=Path(str(path)+".tmp"); torch.save(payload,tmp); os.replace(tmp,path)

    def run(self) -> dict[str,object]:
        cfg=self.recipe.training
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
        schedules0,plan0=self._schedules(1); steps_per_epoch=len(plan0)
        horizon=int(cfg.scheduler_horizon_epochs or self.epochs)
        scheduler=build_lr_scheduler(optimizer,name=cfg.scheduler,total_steps=max(1,horizon*steps_per_epoch),warmup_steps=int(round(cfg.warmup_epochs*steps_per_epoch)),min_lr_ratio=cfg.min_lr_ratio)
        use_scaler=cfg.amp and cfg.amp_dtype.lower()=="float16"; scaler=torch.cuda.amp.GradScaler(enabled=use_scaler)
        history=[]; latest=self.store.checkpoint("last.pt"); best=self.store.checkpoint("best.pt"); start_epoch=1; best_score=-np.inf; best_epoch=0; global_step=0
        if latest.is_file():
            state=torch.load(latest,map_location=self.device,weights_only=False)
            if state.get("scope")!=self.scope or state.get("mode")!=self.mode or int(state.get("epochs_planned",-1))!=self.epochs: raise RuntimeError("resume checkpoint contract mismatch")
            self.model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"]); scheduler.load_state_dict(state["scheduler_state"])
            if state.get("scaler_state") is not None: scaler.load_state_dict(state["scaler_state"])
            history=list(state.get("history",[])); start_epoch=int(state["epoch"])+1; global_step=sum(int(x["optimizer_steps"]) for x in history)
            if self.mode=="development" and history:
                scores=[x.get("development",{}).get("val_cpg_x_val_sample",{}).get("mas_pcc",-np.inf) for x in history]; best_score=float(np.nanmax(scores)); best_epoch=int(history[int(np.nanargmax(scores))]["epoch"])
        started_all=time.time()
        for epoch in range(start_epoch,self.epochs+1):
            started=time.time(); self.model.train(); schedules,plan=self._schedules(epoch); source_steps={p.name:0 for p in self.pools}; losses=[]; optimizer_steps=0
            for source_i,local_step in plan:
                pool=self.pools[source_i]; row_slots,cpg_slots=schedules[source_i][local_step]; sample_ids,cpg_ids,beta_np=self._read_block(pool,row_slots,cpg_slots); source_steps[pool.name]+=1
                result=self._step(pool,sample_ids,cpg_ids,beta_np)
                if result is None: continue
                loss,pieces=result; optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    old_scale=scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(self.model.parameters(),cfg.gradient_clip_norm); scaler.step(optimizer); scaler.update(); stepped=scaler.get_scale()>=old_scale
                else:
                    loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(),cfg.gradient_clip_norm); optimizer.step(); stepped=True
                if stepped: scheduler.step()
                global_step+=1; optimizer_steps+=1; losses.append(float(loss.detach()))
                if global_step%max(1,self.recipe.tracking.log_every_steps)==0: self.wandb.log({"train/loss":float(loss.detach()),"train/lr":optimizer.param_groups[0]["lr"],"train/source":pool.name,"train/epoch":epoch},step=global_step)
            row={"epoch":epoch,"seconds":time.time()-started,"optimizer_steps":optimizer_steps,"source_steps":source_steps,"loss":float(np.mean(losses)) if losses else float("nan"),"lr":optimizer.param_groups[0]["lr"]}
            if self.mode=="development":
                dev=self.evaluate_development(); row["development"]=dev; score=float(dev["val_cpg_x_val_sample"]["mas_pcc"])
                self.wandb.log({f"development/{view}/{k}":v for view,m in dev.items() for k,v in m.items() if isinstance(v,(int,float))},step=global_step)
                if np.isfinite(score) and score>best_score:
                    best_score=score; best_epoch=epoch; self._save_checkpoint(best,optimizer,scheduler,scaler,epoch,[*history,row])
            history.append(row); self._save_checkpoint(latest,optimizer,scheduler,scaler,epoch,history); write_json(self.store.training_file("history.json"),history); print(f"[rna:{self.scope}:{self.mode}:{epoch}/{self.epochs}] loss={row['loss']:.6g}",flush=True)
        if self.mode=="final":
            self._save_checkpoint(best,optimizer,scheduler,scaler,self.epochs,history); best_epoch=self.epochs
        summary={"scope":self.scope,"mode":self.mode,"best_epoch":best_epoch,"best_inner_double_ood_mas_pcc":None if self.mode=="final" else best_score,"epochs":self.epochs,"elapsed_seconds":time.time()-started_all,"run_dir":str(self.store.path)}
        write_json(self.store.training_file("summary.json"),summary); self.wandb.finish(summary); return summary
