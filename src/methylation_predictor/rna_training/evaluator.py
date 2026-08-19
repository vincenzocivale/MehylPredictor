"""Evaluate any canonical RNA checkpoint on chr1, chr123 or genome-wide Array views."""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ..config import EncoderConfig, InteractionConfig, ModelConfig
from ..models import RNAMethylationPredictor
from ..run_store import RunStore, sha256_file, write_json
from ..scopes import chromosome_lookup, evaluation_protocol
from ..storage import LocusFeatureCache, RNACache
from ..tcga_canonical import TCGACanonicalBundle
from .metrics import ArrayMomentMetrics


def _model_config(raw: dict) -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(**raw.get("encoder",{})), interaction=InteractionConfig(**raw.get("interaction",{})),
        zero_init_residual=raw.get("zero_init_residual",True), variance_normalized_residual=raw.get("variance_normalized_residual",True),
    )


class ScopedRNAEvaluator:
    def __init__(self,*,canonical_root,checkpoint,feature_cache,rna_cache,registry,eval_scope,output,sample_chunk=128,cpg_chunk=2048):
        self.root=Path(canonical_root); self.checkpoint=Path(checkpoint); self.eval_scope=eval_scope; self.output=Path(output); self.output.mkdir(parents=True,exist_ok=True); self.sample_chunk=sample_chunk; self.cpg_chunk=cpg_chunk
        self.bundle=TCGACanonicalBundle.from_root(self.root); self.protocol=evaluation_protocol(eval_scope,self.bundle,canonical_root=self.root); self.features=LocusFeatureCache(feature_cache); self.rna=RNACache(rna_cache); self.registry=Path(registry)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type!="cuda": raise RuntimeError("evaluation requires CUDA")
        state=torch.load(self.checkpoint,map_location=self.device,weights_only=False); self.train_scope=str(state.get("scope","unknown")); cfg=_model_config(state.get("model_config",{})); self.model=RNAMethylationPredictor(25_017,1536,cfg).to(self.device); self.model.load_state_dict(state["model_state"],strict=True); self.model.eval()

    def close(self): self.bundle.close()

    @torch.no_grad()
    def _view(self,sample_ids,cpg_ids):
        source=self.bundle.sources["array"]
        rows=source.rows_of_samples(sample_ids)
        chrom0=chromosome_lookup(self.registry,cpg_ids)
        unique=sorted(np.unique(chrom0),key=lambda x:int(x.removeprefix("chr")) if x.removeprefix("chr").isdigit() else 10_000)
        # Group columns by chromosome once. The global metrics are invariant to
        # column permutation, while per-chromosome accumulators then receive
        # contiguous local column offsets even when the official cpg_idx order is
        # not genomic.
        order=np.concatenate([np.flatnonzero(chrom0==c) for c in unique])
        cpg_ids=np.asarray(cpg_ids[order],np.int64); chrom=chrom0[order]
        global_m=ArrayMomentMetrics(len(sample_ids),len(cpg_ids))
        per={c:ArrayMomentMetrics(len(sample_ids),int((chrom==c).sum())) for c in unique}
        chrom_start={}; offset=0
        for c in unique:
            chrom_start[c]=offset; offset+=int((chrom==c).sum())
        for s0 in range(0,len(sample_ids),self.sample_chunk):
            s1=min(s0+self.sample_chunk,len(sample_ids)); rna=torch.from_numpy(self.rna.rows(sample_ids[s0:s1])).to(self.device)
            for c0 in range(0,len(cpg_ids),self.cpg_chunk):
                c1=min(c0+self.cpg_chunk,len(cpg_ids)); ids=cpg_ids[c0:c1]; emb_np,prior_np,sigma_np=self.features.get(ids); emb=torch.from_numpy(emb_np).to(self.device); prior=torch.from_numpy(prior_np).to(self.device); sigma=torch.from_numpy(sigma_np).to(self.device)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16): pred=self.model(rna,emb,prior,sigma=sigma)["beta"]
                target=source.block(rows[s0:s1],ids); pred_np=pred.float().cpu().numpy(); global_m.add(s0,c0,target,pred_np,prior_np)
                local_chrom=chrom[c0:c1]
                for c in np.unique(local_chrom):
                    mask=np.flatnonzero(local_chrom==c)
                    first_global=c0+int(mask[0]); local_start=first_global-chrom_start[c]
                    per[c].add(s0,local_start,target[:,mask],pred_np[:,mask],prior_np[mask])
        return global_m.finalize(),{c:{"cpgs":int((chrom==c).sum()),**m.finalize()} for c,m in per.items()}

    def run(self):
        result={"schema_version":1,"training_scope":self.train_scope,"evaluation_scope":self.eval_scope,"checkpoint":str(self.checkpoint),"checkpoint_sha256":sha256_file(self.checkpoint),"views":{}}; rows=[]
        for name,view in self.protocol.evaluation_views().items():
            started=time.time(); glob,per=self._view(view.sample_idx,view.cpg_idx); result["views"][name]={"global":glob,"per_chromosome":per}; print(f"[eval:{self.train_scope}->{self.eval_scope}:{name}] mas_pcc={glob['mas_pcc']:.6f} mse={glob['mse']:.6f} seconds={time.time()-started:.1f}",flush=True)
            for chrom,m in per.items(): rows.append({"view":name,"chromosome":chrom,**m})
        write_json(self.output/"metrics.json",result); pd.DataFrame(rows).to_csv(self.output/"per_chromosome.csv",index=False); write_json(self.output/"manifest.json",{"training_scope":self.train_scope,"evaluation_scope":self.eval_scope,"checkpoint":str(self.checkpoint),"checkpoint_sha256":result["checkpoint_sha256"],"dataset_contract":"TCGA canonical official Array evaluation views"}); return result
