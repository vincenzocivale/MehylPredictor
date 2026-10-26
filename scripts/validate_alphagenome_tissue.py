"""Retrain with one tissue excluded versus a size-matched random control."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from verify_alphagenome_capacity import prepare,folds,normalize,dump,CANON
from train_alphagenome_v2 import train,fit_prior,training_means
from search_alphagenome_architecture import factors
from search_alphagenome_hierarchical import fit_known_offsets
from validate_alphagenome_hypotheses import OUT,SEEDS,t,evaluate,event,UNKNOWN


@torch.inference_mode()
def assess(m,d,ts,vs,tc,vc,group):
    seen=np.sort(np.random.default_rng(20261018).choice(tc,min(4096,len(tc)),replace=False))
    u=m.encode_rna(d.rna[t(vs)]).float();ut=m.encode_rna(d.rna[t(ts)]).float()
    result=[]
    for view,cc in [('novel',vc),('known',seen)]:
        b,a=factors(m,d,cc);response=u@a.T/m.rank**.5
        if view=='novel':
            corrected=m.calibrated_intercept(b,a,t(d.raw_ag[cc]),blend=.25)
        else:
            pieces=[]
            for start in range(0,len(cc),512):
                sl=slice(start,start+512)
                pieces.append(fit_known_offsets(b[sl],a[sl],ut,d.block(t(ts),t(cc[sl])),[1.])[1.])
            corrected=b+torch.cat(pieces)
        for arm,bias in [('raw',b),('hierarchical',corrected)]:
            p=(bias[None]+response).sigmoid();y=d.block(t(vs),t(cc))
            for label,ix in [('all',np.arange(len(vs))),('held_tissue',np.flatnonzero(d.tissue[vs]==group)),
                             ('other_tissues',np.flatnonzero(d.tissue[vs]!=group))]:
                metric,_=evaluate(y[t(ix)],p[t(ix)],d.tissue[vs[ix]],detailed=False)
                result.append(dict(view=view,arm=arm,population=label,patients=len(ix),**metric))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(groups='two largest named tissues among development training patients; target independent',
                  modes=['leave_tissue_out','matched_random'],seeds=SEEDS,steps=1000,
                  control='same number of training patients, random removal; identical test patients and loci',
                  evaluation='dev patients x dev genomic blocks plus 4096 fixed known CpGs',
                  caveat='development data; tissue exclusion is fresh for these fits, not a new external cohort',
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=out/'tissue_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:dump(path,protocol)
    torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False
    features=out/'tissue_features';features.mkdir(exist_ok=True)
    d=prepare(torch.device('cuda'),features)
    _,base_ts,vs,tc,vc=folds(d)[-1]
    groups=pd.Series(d.tissue[base_ts][~np.isin(d.tissue[base_ts],UNKNOWN)]).value_counts().index[:2].tolist()
    dump(out/'tissue_groups.json',dict(groups=groups,selection='training tissue counts, no methylation target use'))
    graph=t(np.load(ROOT/'artifacts/alphagenome_v2_2026-10-18/dev/response_graph.npz')['neighbors'])
    with h5py.File(CANON/'rna/tcga_rna_official_full.h5') as f:
        genes=[v.decode() if isinstance(v,bytes) else str(v) for v in f['gene_ids'][:]]
    records=[]
    for group in groups:
        exclude=base_ts[d.tissue[base_ts]!=group]
        matched=np.sort(np.random.default_rng(20261018).choice(base_ts,len(exclude),replace=False))
        for mode,ts in [('leave_tissue_out',exclude),('matched_random',matched)]:
            if mode=='leave_tissue_out':assert not (d.tissue[ts]==group).any()
            dest=out/'tissue'/group/mode;dest.mkdir(parents=True,exist_ok=True)
            norm=normalize(d,ts,tc)
            prior=fit_prior(d,tc,training_means(d,ts,tc),dest)
            for seed in SEEDS:
                final=dest/f'seed{seed}';final.mkdir(exist_ok=True)
                result_path=final/'evaluation.json'
                if result_path.exists():res=json.loads(result_path.read_text())
                else:
                    m,_=train(d,ts,tc,'local01',seed,prior,graph,final,norm,genes,protocol)
                    res=assess(m,d,ts,vs,tc,vc,group)
                    dump(result_path,res);del m;torch.cuda.empty_cache()
                for row in res:records.append(dict(group=group,mode=mode,seed=seed,train_patients=len(ts),**row))
                event(stage='tissue',group=group,mode=mode,seed=seed,held_tissue=[r for r in res if r['population']=='held_tissue'])
    pd.DataFrame(records).to_csv(out/'tissue_metrics.csv',index=False)
    dump(out/'tissue_complete.json',dict(complete=True,groups=groups,fits=len(groups)*2*len(SEEDS),official_targets_used=False))


if __name__=='__main__':main()
