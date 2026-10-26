"""Cross-fit a genomic mean prior plus training-locus intercept adaptation.

This follow-up has its own protocol, frozen before its first evaluation. It
uses only internal training/development axes and never official holdout data.
"""
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
from methylation_predictor.modeling.factory import build_alphagenome_predictor
from verify_alphagenome_capacity import prepare,folds,normalize,score,dump,CANON
from search_alphagenome_architecture import PARENT,SEEDS,MODES,factors

SEARCH=ROOT/'artifacts/alphagenome_architecture_2026-10-18'
SOURCES=['v1_early','v2_early']+MODES
NOVEL_BLEND=[0.,.25,.5]
KNOWN_BLEND=[0.,.5,1.]


def source_path(name,source,seed):
    if source in ['v1_early','v2_early']:
        arm='baseline' if source=='v1_early' else 'local01'
        return PARENT/name/arm/f'seed{seed}'/'checkpoint.pt'
    return SEARCH/name/source/f'seed{seed}'/'checkpoint.pt'


def load(name,source,seed):
    st=torch.load(source_path(name,source,seed),map_location='cpu',weights_only=False)
    m=build_alphagenome_predictor(**st['model_spec']).cuda().eval()
    m.load_state_dict(st['model_state']); return m


@torch.inference_mode()
def fit_known_offsets(b,a,u,y,weights):
    """Fit per-locus offsets on observed training pairs only, no held patients."""
    mask=torch.isfinite(y)
    n=mask.sum(0); assert (n>100).all()
    target=torch.where(mask,y,0).sum(0)/n
    response=u@a.T/a.shape[1]**.5
    initial=torch.where(mask,(b[None]+response).sigmoid(),0).sum(0)/n
    result={0.:torch.zeros_like(b)}
    for w in weights:
        if w==0: continue
        desired=(1-w)*initial+w*target
        bound=response.abs().amax(0)+b.abs()+20
        lo,hi=-bound,bound
        for _ in range(24):
            mid=(lo+hi)/2
            mean=torch.where(mask,(mid[None]+response).sigmoid(),0).sum(0)/n
            lower=mean<desired
            lo,hi=torch.where(lower,mid,lo),torch.where(lower,hi,mid)
        result[w]=(lo+hi)/2-b
    return result


def summarize(y,p,tissue):
    metric,per=score(y,p,tissue)
    per['mae']=np.nanmean(np.where(np.isfinite(y),np.abs(y-p),np.nan),0)
    metric['mae']=float(np.average(per['mae'],weights=per['n']))
    metric['rows']=int(per['n'].sum())
    return metric,per


@torch.inference_mode()
def evaluate(d,spec,source,seed,out,novel_weights,known_weights):
    name,ts,vs,tc,vc=spec
    dest=out/name/source/f'seed{seed}'; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'evaluation.json').exists(): return
    m=load(name,source,seed)
    seen=np.sort(np.random.default_rng(20261018).choice(tc,min(4096,len(tc)),replace=False))
    ss=torch.as_tensor(vs,device=d.device); tt=torch.as_tensor(ts,device=d.device)
    u=m.encode_rna(d.rna[ss]).float(); train_u=m.encode_rna(d.rna[tt]).float()
    result={'novel':{},'known':{}}
    for view,cols,weights in [('novel',vc,novel_weights),('known',seen,known_weights)]:
        cc=torch.as_tensor(cols,device=d.device)
        y=d.block(ss,cc).cpu().numpy()
        predictions={w:np.empty(y.shape,np.float32) for w in weights}
        offsets={w:[] for w in weights}
        for j in range(0,len(cols),512):
            c=cc[j:j+512]; f=d.functional(c)
            b,a=m.encode_cpg(**f,calibrate=False)
            if view=='known':
                delta=fit_known_offsets(b.float(),a.float(),train_u,d.block(tt,c),weights)
                bias={w:b+delta[w] for w in weights}
                for w in weights: offsets[w].append(delta[w].cpu().numpy())
            else:
                bias={w:m.calibrated_intercept(b,a,f['dense'][:,23:],blend=w) for w in weights}
            response=u@a.float().T/m.rank**.5
            for w in weights:
                predictions[w][:,j:j+len(c)]=(bias[w][None]+response).sigmoid().cpu().numpy()
        for w in weights:
            metric,per=summarize(y,predictions[w],d.tissue[vs]); result[view][str(w)]=metric
            np.savez_compressed(dest/f'{view}_{w:g}_per_cpg.npz',**per,cpg_idx=d.cpg_idx[cols])
            if view=='known':
                np.savez_compressed(dest/f'known_offsets_{w:g}.npz',cpg_idx=d.cpg_idx[cols],bias_delta=np.concatenate(offsets[w]),
                                    fit_sample_ids=d.sample_idx[ts],fit_cpg_ids=d.cpg_idx[tc])
    dump(dest/'evaluation.json',result)
    print(json.dumps(dict(fold=name,source=source,seed=seed,result=result)),flush=True)
    del m; torch.cuda.empty_cache()


def select(out):
    candidates=[]
    for source in SOURCES:
        docs=[json.loads((out/f'fold{k}'/source/f'seed{s}'/'evaluation.json').read_text()) for k in range(3) for s in SEEDS]
        for nw in NOVEL_BLEND:
            for kw in KNOWN_BLEND:
                row=dict(source=source,novel_blend=nw,known_blend=kw)
                for view,w in [('novel',nw),('known',kw)]:
                    row[view]={metric:float(np.mean([d[view][str(w)][metric] for d in docs])) for metric in docs[0][view][str(w)]}
                candidates.append(row)
    base=next(c for c in candidates if c['source']=='v1_early' and c['novel_blend']==c['known_blend']==0.)
    bq,bk=base['novel'],base['known']
    for c in candidates:
        q,k=c['novel'],c['known']
        c['admissible']=(q['mse']<=bq['mse']*.995 and k['mse']<=bk['mse']*1.01 and
                         q['mae']<=bq['mae'] and k['mae']<=bk['mae']*1.01 and
                         q['mas_pcc']>=bq['mas_pcc']-.002 and q['within_tissue']>=bq['within_tissue']-.002 and
                         k['mas_pcc']>=bk['mas_pcc']-.002 and k['within_tissue']>=bk['within_tissue']-.002 and
                         q['severe']<=bq['severe']+.001)
        c['balanced_mse_ratio']=.5*(q['mse']/bq['mse']+k['mse']/bk['mse'])
    eligible=[c for c in candidates if c['admissible']]
    chosen=min(eligible,key=lambda c:(c['novel']['mse'],c['known']['mse'],c['novel_blend'],c['known_blend'])) if eligible else None
    fallback=min(candidates,key=lambda c:c['balanced_mse_ratio'])
    frozen=dict(selected=chosen,best_balanced=fallback,candidates=candidates,baseline=base,guardrails_passed=bool(eligible),
                selection_data='3 internal folds x 3 seeds',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(out/'frozen_recipe.json',frozen)
    rows=[]
    for c in candidates:
        row={k:c[k] for k in ['source','novel_blend','known_blend','admissible','balanced_mse_ratio']}
        for view in ['novel','known']: row.update({view+'_'+k:v for k,v in c[view].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'selection.csv',index=False)
    print(json.dumps(dict(selected=chosen,best_balanced=fallback)),flush=True)
    return frozen


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/alphagenome_hierarchical_2026-10-18')
    args=ap.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    assert (SEARCH/'complete.json').exists(),'Complete the two-stage search first'
    torch.set_num_threads(4); threadpool_limits(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
    protocol=dict(sources=SOURCES,novel_blend=NOVEL_BLEND,known_blend=KNOWN_BLEND,seeds=SEEDS,
                  hypothesis='Genomic prior for unseen CpGs; empirical training intercepts for observed CpGs',
                  fitting='No additional encoder fitting; monotone per-locus intercept solves, train patient observations only',
                  selection='same strict guardrails as preceding search; no official scoring; dev only after freeze',
                  justification='Earlier v2 prior improved new-locus MSE but harmed known loci; known labels permit separate intercept fitting',
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  preceding_protocol_sha256=hashlib.sha256((SEARCH/'protocol.json').read_bytes()).hexdigest())
    if (out/'protocol.json').exists(): assert json.loads((out/'protocol.json').read_text())==protocol
    else: dump(out/'protocol.json',protocol)
    d=prepare(torch.device('cuda'),out); specs=folds(d)
    for spec in specs[:3]:
        normalize(d,spec[1],spec[3])
        for seed in SEEDS:
            for source in SOURCES: evaluate(d,spec,source,seed,out,NOVEL_BLEND,KNOWN_BLEND)
    frozen=select(out); c=frozen['selected'] or frozen['best_balanced']
    spec=specs[-1]; normalize(d,spec[1],spec[3])
    # Selected source continuation might not have been needed by preceding dev.
    source=c['source']
    if source in MODES and not source_path('dev',source,17).exists():
        from search_alphagenome_architecture import continue_fit
        with h5py.File(CANON/'rna/tcga_rna_official_full.h5','r') as h:
            genes=[g.decode() if isinstance(g,bytes) else str(g) for g in h['gene_ids'][:]]
        norm=normalize(d,spec[1],spec[3])
        for seed in SEEDS: continue_fit(d,spec,source,seed,SEARCH,norm,genes)
    for seed in SEEDS:
        for s in list(dict.fromkeys(['v1_early','v2_early',source])):
            novel=list(dict.fromkeys([0.,c['novel_blend']])) if s==source else [0.]
            known=list(dict.fromkeys([0.,c['known_blend']])) if s==source else [0.]
            evaluate(d,spec,s,seed,out,novel,known)
    dump(out/'complete.json',dict(complete=True,selected=frozen['selected'],best_balanced=frozen['best_balanced'],
                                  guardrails_passed=frozen['guardrails_passed'],official_holdout_used=False))
    print(json.dumps(dict(complete=True,guardrails_passed=frozen['guardrails_passed'])),flush=True)


if __name__=='__main__': main()
