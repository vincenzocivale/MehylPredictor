"""Exact cancer-type-stratified H1/H3 metrics from released prediction rows."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--sample-metadata',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--batch-rows',type=int,default=1_000_000);a=p.parse_args()
 sm=pd.read_parquet(a.sample_metadata); cancer=np.full(sm.sample_idx.max()+1,'UNKNOWN',dtype=object); cancer[sm.sample_idx.to_numpy(int)]=sm.cancer_type.to_numpy(str)
 parts=[]; rows=0
 for b in ds.dataset(a.predictions,format='parquet').scanner(columns=['group_idx','cpg_idx','sample_idx','gt_methyl','pred_methyl'],batch_size=a.batch_rows).to_batches():
  x=b.to_pandas().dropna()
  x['cancer_type']=cancer[x.sample_idx.to_numpy(int)]; x['y2']=x.gt_methyl*x.gt_methyl; x['p2']=x.pred_methyl*x.pred_methyl; x['yp']=x.gt_methyl*x.pred_methyl
  parts.append(x.groupby(['group_idx','cancer_type','cpg_idx'],sort=False)[['gt_methyl','pred_methyl','y2','p2','yp']].sum().assign(n=x.groupby(['group_idx','cancer_type','cpg_idx'],sort=False).size()))
  rows+=len(x)
 stats=pd.concat(parts).groupby(level=[0,1,2]).sum().reset_index()
 out=[]
 for (g,c),z in stats.groupby(['group_idx','cancer_type'],sort=True):
  n=z.n.to_numpy(float); y=z.gt_methyl.to_numpy()/n; pred=z.pred_methyl.to_numpy()/n; total=n.sum(); grand=np.dot(n,y)/total
  ss_locus=np.dot(n,(y-grand)**2); ss_within=(z.y2.to_numpy()-n*y*y).sum(); dyn=((z.y2.to_numpy()-n*y*y)+(z.p2.to_numpy()-n*pred*pred)-2*(z.yp.to_numpy()-n*y*pred)).sum(); static=np.dot(n,(pred-y)**2)
  out.append({'group_idx':int(g),'cancer_type':c,'n_rows':int(total),'n_cpg':int(len(z)),'f_locus':float(ss_locus/(ss_locus+ss_within)),'dynamic_skill':float(1-dyn/ss_within),'static_mse':float(static/total),'dynamic_mse':float(dyn/total)})
 frame=pd.DataFrame(out);Path(a.output).parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(a.output,index=False)
 a.output.with_suffix('.json').write_text(json.dumps({'n_rows':rows,'n_strata':len(frame),'definition':'centering and variance decomposition within each cancer type and split'},indent=2)+'\n')
if __name__=='__main__':main()
