"""Confirm fixed cross-fitted risk/regularization rules under distribution shift."""
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from methylation_predictor.storage import FunctionalLocusCache
from methylation_predictor.modeling.alphagenome import AlphaGenomePredictorFromCheckpoint
from verify_alphagenome_capacity import CANON,AG,positions,selected_matrix,dump
from validate_alphagenome_hypotheses import OUT,SEEDS,WEIGHTS,evaluate,t,event
from validate_alphagenome_transfer import decode,patient_targets


def models(out):
    data=[np.load(out/f'fold{k}/adaptive_data.npz') for k in range(3)]
    x=np.concatenate([z['x'] for z in data]);loss=np.concatenate([z['loss'] for z in data])
    def new():return RandomForestRegressor(n_estimators=150,max_depth=5,min_samples_leaf=100,max_features=.8,random_state=20261018,n_jobs=4)
    risk=new().fit(x,loss[:,1]);adaptive=new().fit(x,loss-loss[:,[0]])
    return risk,adaptive


def risk_scores(view,x,truth,n,risk):
    valid=np.isfinite(truth)&(n>0)
    x,truth,n,risk=x[valid],truth[valid],n[valid],risk[valid]
    results=[]
    for label,value in [('learned',risk),('seed_disagreement',x[:,27]),('predicted_sd',x[:,24]),
                         ('prior_midpoint',x[:,25]*(1-x[:,25]))]:
        order=np.argsort(value,kind='stable')
        rho=float(pd.Series(value).corr(pd.Series(truth),method='spearman'))
        auc=float(roc_auc_score(truth>=np.quantile(truth,.9),value))
        for coverage in [.25,.5,.75,1.]:
            ix=order[:max(1,round(len(order)*coverage))]
            results.append(dict(view=view,estimator=label,coverage=coverage,mse=float(np.average(truth[ix],weights=n[ix])),
                                error_prediction_spearman=rho,hard_decile_auc=auc))
    return results


@torch.inference_mode()
def confirmation(out,label,risk,adaptive,cache,ag_ids,ag):
    dest=out/'reliability'/label;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    axes=np.load(out/'transfer'/label/'axes.npz');cc=axes['cpg_idx'];ss=axes['sample_idx']
    rawag=ag[positions(ag_ids,cc)]
    with h5py.File(CANON/'rna/tcga_rna_official_full.h5') as f:
        ix=positions(f['sample_idx'][:],ss);order=np.argsort(ix)
        raw=f['X'][ix[order]][np.argsort(order)].astype(np.float32)
        tissue=decode(f['tissue_name'][:])[ix]
    filename='tcga_array_official_full.h5' if label.startswith('chr') else label.split('_')[0]+'_full.h5'
    with h5py.File(CANON/'methylation'/filename) as f:
        yy=t(patient_targets(f,ss,cc))
    all_bias,all_factors,all_u,extra,means,variances=[],[],[],[],[],[]
    losses=[];per_weights=[];sum_p=square_p=None
    for seed in SEEDS:
        adapter=AlphaGenomePredictorFromCheckpoint(ROOT/f'artifacts/alphagenome_v2_2026-10-18/official/seed{seed}/checkpoint.pt',device='cuda')
        m=adapter.model;norm=adapter.normalization
        u=m.encode_rna((t(raw)-norm['rna_mean'])/norm['rna_std']).float()
        biases={w:[] for w in WEIGHTS};factors=[];priors=[];annotations=[]
        for start in range(0,len(cc),512):
            ids=cc[start:start+512];f=cache.get(ids)
            annotations.append(f['dense'])
            dense=t(f['dense']);dense[:,17]=(dense[:,17]-norm['dense17_mean'])/norm['dense17_std']
            b,a=m.encode_cpg(track_indices=t(f['track_indices']),offsets=t(f['offsets']),dense=torch.cat([dense,t(rawag[start:start+len(ids)])],1),calibrate=False)
            factors.append(a)
            for w in WEIGHTS:biases[w].append(m.calibrated_intercept(b,a,t(rawag[start:start+len(ids)]),blend=w))
            priors.append(m.mean_prior(t(rawag[start:start+len(ids)])))
        a=torch.cat(factors);bs=torch.stack([torch.cat(biases[w]) for w in WEIGHTS],1)
        response=u@a.T/m.rank**.5
        p=(bs[:,1][None]+response).sigmoid()
        means.append(p.mean(0).cpu().numpy());variances.append(p.var(0,unbiased=False).cpu().numpy())
        extra.append(np.c_[p.mean(0).cpu().numpy(),p.std(0,unbiased=False).cpu().numpy(),torch.cat(priors).cpu().numpy(),torch.linalg.vector_norm(a,dim=1).cpu().numpy()])
        sum_p=p.clone() if sum_p is None else sum_p+p
        square_p=p.square() if square_p is None else square_p+p.square()
        ww=[]
        for wi,w in enumerate(WEIGHTS):
            metric,per=evaluate(yy,(bs[:,wi][None]+response).sigmoid(),tissue)
            ww.append(per)
            dump(dest/f'seed{seed}_weight{w:g}.json',metric)
        per_weights.append(ww);losses.append(np.stack([p['mse'] for p in ww],axis=1))
        all_bias.append(bs);all_factors.append(a);all_u.append(u)
        del m,adapter
    disagreement=(square_p/len(SEEDS)-(sum_p/len(SEEDS)).square()).clamp_min(0).mean(0).sqrt().cpu().numpy()
    x=np.c_[np.concatenate(annotations),np.mean(extra,axis=0),disagreement,np.std(means,axis=0),
            np.mean(variances,axis=0),np.linalg.norm(rawag,axis=1)]
    # Features above are all prediction/sequence based. Evaluation labels have
    # no route into the fit or into choices, including where targets are NaN.
    estimated_risk=risk.predict(x);choice=adaptive.predict(x).argmin(1)
    truth=np.mean(losses,axis=0)[:,1];n=per_weights[0][1]['n']
    curves=risk_scores(label,x,truth,n,estimated_risk)
    pd.DataFrame(curves).to_csv(dest/'risk_coverage.csv',index=False)
    np.savez_compressed(dest/'features.npz',x=x,cpg_idx=cc,risk=estimated_risk,choice=choice,true_mse=truth,n=n)
    records=[]
    for seed,bs,a,u in zip(SEEDS,all_bias,all_factors,all_u):
        selected=bs[torch.arange(len(cc),device='cuda'),t(choice)]
        metric,per=evaluate(yy,(selected[None]+u@a.T/a.shape[1]**.5).sigmoid(),tissue)
        row=dict(view=label,seed=seed,**metric);records.append(row)
        np.savez_compressed(dest/f'adaptive_seed{seed}_per_cpg.npz',cpg_idx=cc,**per)
    pd.DataFrame(records).to_csv(dest/'adaptive_metrics.csv',index=False)
    dump(dest/'complete.json',dict(complete=True,weight_counts={str(w):int((choice==i).sum()) for i,w in enumerate(WEIGHTS)}))
    event(stage='reliability',view=label,risk=[r for r in curves if r['coverage']==.5],adaptive=records)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args();out=args.out
    protocol=dict(fit='all three internal held-locus folds; no dev labels',
                  risk='same RF hyperparameters as internal validation',adaptive='same RF differential-loss rule as internal validation',
                  tests=['chr2','chr22','epic_novel','wgbs_novel'],
                  caution='Follow-up confirmation protocol after inspecting base v3 chromosome results; no adaptive/risk hyperparameter retuning',
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=out/'reliability_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:dump(path,protocol)
    torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False
    risk,adaptive=models(out)
    curves=[]
    for split in ['fold0','fold1','fold2','dev']:
        z=np.load(out/split/'adaptive_data.npz');p=pd.read_csv(out/split/'adaptive_predictions.csv.gz')
        curves.extend(risk_scores(split,z['x'],z['loss'][:,1],z['n'],p.risk.to_numpy()))
    pd.DataFrame(curves).to_csv(out/'risk_controls.csv',index=False)
    cache=FunctionalLocusCache(locus_store=ROOT/'local_methyl_data/derived/locus_features_v1')
    with h5py.File(AG) as f:ag_ids=f['cpg_idx'][:];ag=f['embedding'][:].astype(np.float32)
    for label in protocol['tests']:
        confirmation(out,label,risk,adaptive,cache,ag_ids,ag)
    pd.concat([pd.read_csv(p) for p in (out/'reliability').glob('*/risk_coverage.csv')]).to_csv(out/'external_risk_coverage.csv',index=False)
    pd.concat([pd.read_csv(p) for p in (out/'reliability').glob('*/adaptive_metrics.csv')]).to_csv(out/'external_adaptive_metrics.csv',index=False)
    dump(out/'reliability_complete.json',dict(complete=True,fit_on_external_labels=False))


if __name__=='__main__':main()
