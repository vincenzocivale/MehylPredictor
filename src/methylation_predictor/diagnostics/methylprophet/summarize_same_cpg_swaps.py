"""Cluster-bootstrap summary for same-CpG gene-swap experiments."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--per-cpg',required=True); p.add_argument('--output',required=True); p.add_argument('--replicates',type=int,default=5000); p.add_argument('--seed',type=int,default=9176); a=p.parse_args()
    x=pd.read_parquet(a.per_cpg).dropna(subset=['centered_sse','centered_ss','prediction_std'])
    def metrics(z):
        return {'pooled_dynamic_skill':float(1-z.centered_sse.sum()/z.centered_ss.sum()), 'median_dynamic_skill':float(z.dynamic_skill.median()), 'median_prediction_std':float(z.prediction_std.median()), 'fraction_nonzero_effect':float((z.prediction_std>1e-3).mean())}
    point=metrics(x); rng=np.random.default_rng(a.seed); values={k:[] for k in point}
    for _ in range(a.replicates):
        z=x.iloc[rng.integers(0,len(x),len(x))]
        for k,v in metrics(z).items(): values[k].append(v)
    out={'estimand':'same-CpG patient gene-vector swap; CpG-cluster percentile bootstrap','n_cpg':len(x),'replicates':a.replicates,'point':point,'ci_95':{k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in values.items()}}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
