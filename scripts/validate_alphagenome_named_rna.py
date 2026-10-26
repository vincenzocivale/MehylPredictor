"""RNA permutation confirmation restricted to explicitly named tissues."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from verify_alphagenome_capacity import prepare,folds,normalize,dump
from search_alphagenome_hierarchical import load,fit_known_offsets
from search_alphagenome_architecture import factors
from validate_alphagenome_hypotheses import OUT,SEEDS,UNKNOWN,t,evaluate,permutations,event


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args();out=args.out
    dest=out/'named_rna';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return
    dump(dest/'protocol.json',dict(reason='Unknown tissue is not a biological group; exclude it from the within-tissue shuffle MSE confirmation',
                                  sources='frozen v2 with v3 calibration',seeds=SEEDS,shuffles=3,split='dev',
                                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False
    d=prepare(torch.device('cuda'),dest)
    name,ts,vs,tc,vc=folds(d)[-1];normalize(d,ts,tc)
    vs=vs[~np.isin(d.tissue[vs],UNKNOWN)]
    seen=np.sort(np.random.default_rng(20261018).choice(tc,min(4096,len(tc)),replace=False))
    records=[]
    for seed in SEEDS:
        m=load(name,'v2_early',seed)
        u=m.encode_rna(d.rna[t(vs)]).float();ut=m.encode_rna(d.rna[t(ts)]).float()
        for view,cc in [('novel',vc),('known',seen)]:
            b,a=factors(m,d,cc)
            if view=='novel':b=m.calibrated_intercept(b,a,t(d.raw_ag[cc]),blend=.25)
            else:
                offsets=[]
                for start in range(0,len(cc),512):
                    sl=slice(start,start+512)
                    offsets.append(fit_known_offsets(b[sl],a[sl],ut,d.block(t(ts),t(cc[sl])),[1.])[1.])
                b=b+torch.cat(offsets)
            response=u@a.T/m.rank**.5;y=d.block(t(vs),t(cc))
            for rep in [-1,0,1,2]:
                ix=np.arange(len(vs)) if rep==-1 else permutations(d.tissue[vs],20261018+rep)
                metric,per=evaluate(y,(b[None]+response[t(ix)]).sigmoid(),d.tissue[vs])
                records.append(dict(view=view,seed=seed,shuffle=rep,patients=len(vs),**metric))
                np.savez_compressed(dest/f'{view}_seed{seed}_shuffle{rep}.npz',cpg_idx=d.cpg_idx[cc],**per)
        del m;torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(dest/'metrics.csv',index=False)
    dump(dest/'complete.json',dict(complete=True,named_tissue_patients=len(vs),tissues=len(np.unique(d.tissue[vs]))))
    event(stage='named_rna',results=records)


if __name__=='__main__':main()
