"""Evaluate the zero-parameter CpG Prior baseline on chr1, chr123 or genome-wide
Array views. (The RNA-methylation checkpoint evaluator lives in
``rna_training.locus_cls_trainer.evaluate_official_split`` -- the earlier
scope-general two-stage evaluator that used to live here, ``ScopedRNAEvaluator``,
has been retired alongside the rest of that architecture generation.)
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..run_store import write_json
from ..scopes import chromosome_lookup, evaluation_protocol
from ..storage import LocusFeatureCache
from ..tcga_canonical import TCGACanonicalBundle
from .metrics import ArrayMomentMetrics


class CpGPriorEvaluator:
    """"CpG Prior" baseline: predicts ``mu`` (the cached locus prior) directly,
    with no RNA input and no trainable parameters at all. Mirrors
    ``ScopedRNAEvaluator`` (same protocol/metrics/output shape, so its
    ``metrics.json`` drops into the same downstream tooling), but skips the
    model entirely -- ``pred`` is just ``prior`` broadcast across samples. See
    docs/PAPER_EXPERIMENTS.md.
    """

    def __init__(self,*,canonical_root,feature_cache,registry,eval_scope,output,sample_chunk=128,cpg_chunk=2048):
        self.root=Path(canonical_root); self.eval_scope=eval_scope; self.output=Path(output); self.output.mkdir(parents=True,exist_ok=True); self.sample_chunk=sample_chunk; self.cpg_chunk=cpg_chunk
        self.bundle=TCGACanonicalBundle.from_root(self.root); self.protocol=evaluation_protocol(eval_scope,self.bundle,canonical_root=self.root); self.features=LocusFeatureCache(feature_cache); self.registry=Path(registry)
        self.train_scope="cpg_prior"

    def close(self): self.bundle.close()

    def _view(self,sample_ids,cpg_ids):
        source=self.bundle.sources["array"]
        rows=source.rows_of_samples(sample_ids)
        chrom0=chromosome_lookup(self.registry,cpg_ids)
        unique=sorted(np.unique(chrom0),key=lambda x:int(x.removeprefix("chr")) if x.removeprefix("chr").isdigit() else 10_000)
        order=np.concatenate([np.flatnonzero(chrom0==c) for c in unique])
        cpg_ids=np.asarray(cpg_ids[order],np.int64); chrom=chrom0[order]
        global_m=ArrayMomentMetrics(len(sample_ids),len(cpg_ids))
        per={c:ArrayMomentMetrics(len(sample_ids),int((chrom==c).sum())) for c in unique}
        chrom_start={}; offset=0
        for c in unique:
            chrom_start[c]=offset; offset+=int((chrom==c).sum())
        for s0 in range(0,len(sample_ids),self.sample_chunk):
            s1=min(s0+self.sample_chunk,len(sample_ids))
            for c0 in range(0,len(cpg_ids),self.cpg_chunk):
                c1=min(c0+self.cpg_chunk,len(cpg_ids)); ids=cpg_ids[c0:c1]; _emb_np,prior_np,_sigma_np=self.features.get(ids)
                target=source.block(rows[s0:s1],ids); pred_np=np.broadcast_to(prior_np[None,:],target.shape)
                global_m.add(s0,c0,target,pred_np,prior_np)
                local_chrom=chrom[c0:c1]
                for c in np.unique(local_chrom):
                    mask=np.flatnonzero(local_chrom==c)
                    first_global=c0+int(mask[0]); local_start=first_global-chrom_start[c]
                    per[c].add(s0,local_start,target[:,mask],pred_np[:,mask],prior_np[mask])
        return global_m.finalize(),{c:{"cpgs":int((chrom==c).sum()),**m.finalize()} for c,m in per.items()}

    def run(self):
        result={"schema_version":1,"training_scope":self.train_scope,"evaluation_scope":self.eval_scope,"checkpoint":None,"checkpoint_sha256":None,"views":{}}; rows=[]
        for name,view in self.protocol.evaluation_views().items():
            started=time.time(); glob,per=self._view(view.sample_idx,view.cpg_idx); result["views"][name]={"global":glob,"per_chromosome":per}; print(f"[eval:{self.train_scope}->{self.eval_scope}:{name}] mas_pcc={glob['mas_pcc']:.6f} mse={glob['mse']:.6f} seconds={time.time()-started:.1f}",flush=True)
            for chrom,m in per.items(): rows.append({"view":name,"chromosome":chrom,**m})
        write_json(self.output/"metrics.json",result); pd.DataFrame(rows).to_csv(self.output/"per_chromosome.csv",index=False); write_json(self.output/"manifest.json",{"training_scope":self.train_scope,"evaluation_scope":self.eval_scope,"checkpoint":None,"checkpoint_sha256":None,"dataset_contract":"TCGA canonical official Array evaluation views"}); return result
