"""Evaluate per-CpG, per-cancer training mean baseline on TCGA chr1."""
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

def read(path):
    d = pd.read_parquet(path, columns=["cpg_idx", "cpg_chr_pos", "sample_idx", "tissue_idx", "methylation"]).dropna()
    return d[d.cpg_chr_pos.str.startswith("chr1_")]

def main():
    a = argparse.ArgumentParser(); a.add_argument("--data_root", required=True); a.add_argument("--output", required=True); args=a.parse_args()
    root=Path(args.data_root)/"processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg"
    train=sorted(glob.glob(str(root/"train_cpg-train_sample.parquet/*.parquet")))
    test=sorted(glob.glob(str(root/"train_cpg-val_sample.parquet/*.parquet")))
    cpg=set(); max_tissue=0
    for i,f in enumerate(train,1):
        d=read(f)
        if d.empty: continue
        cpg.update(d.cpg_idx.unique()); max_tissue=max(max_tissue, int(d.tissue_idx.max()))
        if i%400==0: print(f"index shards: {i}/{len(train)}",flush=True)
    cpg=np.array(sorted(cpg),dtype=np.int64); nt=max_tissue+1
    sums=np.zeros((nt,len(cpg)),np.float64); counts=np.zeros((nt,len(cpg)),np.int32)
    for i,f in enumerate(train,1):
        d=read(f)
        if d.empty: continue
        j=np.searchsorted(cpg,d.cpg_idx.to_numpy()); t=d.tissue_idx.to_numpy()
        for tissue in np.unique(t):
            mask=t==tissue; np.add.at(sums[tissue],j[mask],d.methylation.to_numpy()[mask]); np.add.at(counts[tissue],j[mask],1)
        if i%400==0: print(f"mean shards: {i}/{len(train)}",flush=True)
    means=sums/np.maximum(counts,1)
    max_sample=20000; n=np.zeros(max_sample,np.int64); sx=np.zeros(max_sample); sy=np.zeros(max_sample); sxx=np.zeros(max_sample); syy=np.zeros(max_sample); sxy=np.zeros(max_sample); total=sse=sae=0.0
    for i,f in enumerate(test,1):
        d=read(f)
        if d.empty: continue
        j=np.searchsorted(cpg,d.cpg_idx.to_numpy()); t=d.tissue_idx.to_numpy(); s=d.sample_idx.to_numpy(); y=d.methylation.to_numpy(); x=means[t,j]; valid=counts[t,j]>0; x,y,s=x[valid],y[valid],s[valid]
        np.add.at(n,s,1); np.add.at(sx,s,x); np.add.at(sy,s,y); np.add.at(sxx,s,x*x); np.add.at(syy,s,y*y); np.add.at(sxy,s,x*y)
        e=x-y; total+=len(e); sse+=float(np.dot(e,e)); sae+=float(np.abs(e).sum())
        if i%400==0: print(f"test shards: {i}/{len(test)}",flush=True)
    den=np.sqrt((n*sxx-sx*sx)*(n*syy-sy*sy)); p=(n*sxy-sx*sy)/den; p=p[np.isfinite(p)&(n>1)]
    out={"baseline":"per_cpg_per_tissue_training_mean","chromosome":"chr1","split":"train_cpg-val_sample","pairs":int(total),"mas_pcc":None,"mac_pcc":float(np.median(p)),"mse":sse/total,"mae":sae/total,"samples_with_pcc":int(len(p))}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2),flush=True)
if __name__=="__main__": main()
