"""Develop, freeze, refit and audit AlphaGenome v2 on Array chr1.

Recipe selection uses only three internal genomic folds. Official targets are
scored only after freeze; their results never change architecture or budget.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, normalize as normalize_rows
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from methylation_predictor.modeling.factory import build_alphagenome_predictor
from methylation_predictor.modeling.alphagenome import local_response_penalty, AlphaGenomePredictorFromCheckpoint
from verify_alphagenome_capacity import (
    prepare, folds, normalize, CapacityModel, Data, LAB, CANON, PREP, AG,
    positions, rows, selected_matrix, cosine_lr, score, dump,
    masked_mse, locus_pcc_loss, group_centered_mse_loss, group_centered_pcc_loss,
)
from analyze_alphagenome_capacity import block_ci

ARMS={'baseline':(False,0.),'mean':(True,0.),'local01':(True,.01),'local10':(True,.1)}
ALPHAS=[0.,.5,1.]


def code_hashes():
    paths=[Path(__file__),ROOT/'src/methylation_predictor/modeling/alphagenome.py',
           ROOT/'src/methylation_predictor/modeling/factory.py',
           ROOT/'scripts/verify_alphagenome_capacity.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def new_model(genes,arm,seed):
    torch.manual_seed(seed)
    reference=CapacityModel(genes,'d8')
    # Match all common initial weights and the dropout RNG state to the v1
    # control. The portable model itself has no dependency on this laboratory.
    with torch.random.fork_rng(devices=[]):
        m=build_alphagenome_predictor(input_dim=genes,shallow_mean=ARMS[arm][0])
    result=m.load_state_dict(reference.state_dict(),strict=False)
    expected={'prior_mean','prior_scale','prior_coef','prior_intercept','reference_u','calibration_ready'}
    assert set(result.missing_keys)==expected and not result.unexpected_keys
    return m.cuda()


def training_means(d,ts,tc):
    s=torch.as_tensor(ts,device=d.device)
    mu=torch.full((len(d.cpg_idx),),torch.nan,device=d.device)
    for c in torch.as_tensor(tc,device=d.device).split(512):
        mu[c]=d.block(s,c).nanmean(0)
    assert torch.isfinite(mu[torch.as_tensor(tc,device=d.device)]).all()
    return mu


def fit_prior(d,tc,mu,dest):
    x=d.raw_ag[tc]
    scaler=StandardScaler().fit(x)
    ridge=Ridge(alpha=100.).fit(scaler.transform(x),mu[torch.as_tensor(tc,device=d.device)].cpu().numpy())
    prior=dict(prior_mean=scaler.mean_.astype(np.float32),prior_scale=scaler.scale_.astype(np.float32),
               prior_coef=np.asarray(ridge.coef_,np.float32),prior_intercept=np.asarray(ridge.intercept_,np.float32))
    np.savez_compressed(dest/'mean_prior.npz',**prior,fit_cpg_idx=d.cpg_idx[tc])
    return prior


def response_graph(d,tc,dest):
    cached=dest/'response_graph.npz'
    if cached.exists():
        z=np.load(cached)
        assert np.array_equal(z['fit_cpg_idx'],d.cpg_idx[tc])
        return torch.as_tensor(z['neighbors'],device=d.device)
    x=d.raw_ag[tc]
    scaler=StandardScaler().fit(x)
    pc=PCA(n_components=64,svd_solver='randomized',random_state=20261018).fit(scaler.transform(x))
    x=normalize_rows(pc.transform(scaler.transform(x))).astype(np.float32)
    xt=torch.as_tensor(x,device=d.device)
    dense=d.raw_dense
    group=dense[tc,:4].argmax(1)*16+dense[tc,4:8].argmax(1)*4
    group+=np.searchsorted(np.quantile(dense[tc,17],[.25,.5,.75]),dense[tc,17])
    pos=d.cpg_pos_bp[torch.as_tensor(tc,device=d.device)]
    table=torch.full((len(d.cpg_idx),5),-1,dtype=torch.long,device=d.device)
    train=torch.as_tensor(tc,device=d.device); fallback=0
    for g in np.unique(group):
        ix=torch.as_tensor(np.flatnonzero(group==g),device=d.device)
        if len(ix)<6:
            pending=ix
        else:
            sim=xt[ix]@xt[ix].T
            sim[(pos[ix,None]-pos[ix][None]).abs()<1_000_000]=-torch.inf
            value,near=sim.topk(5,dim=1)
            good=torch.isfinite(value).all(1)
            table[train[ix[good]]]=train[ix[near[good]]]
            pending=ix[~good]
        if len(pending):
            fallback+=len(pending)
            for piece in pending.split(256):
                sim=xt[piece]@xt.T
                sim[(pos[piece,None]-pos[None]).abs()<1_000_000]=-torch.inf
                value,near=sim.topk(5,dim=1)
                assert torch.isfinite(value).all()
                table[train[piece]]=train[near]
    result=table.cpu().numpy()
    assert np.isin(result[tc],tc).all()
    position=d.cpg_pos_bp.cpu().numpy()
    assert (np.abs(position[tc,None]-position[result[tc]])>=1_000_000).all()
    np.savez_compressed(cached,neighbors=result,fit_cpg_idx=d.cpg_idx[tc])
    dump(dest/'response_graph.json',dict(loci=len(tc),neighbors=5,distance_min_bp=1000000,
                                        matching='context, region, TSS quartile; global fallback if <5 eligible',
                                        fallback_loci=fallback,representation='train-scaled raw AG PCA64 cosine',
                                        uses_methylation=False))
    print(json.dumps({'graph':str(dest),'loci':len(tc),'fallback':fallback}),flush=True)
    return table


@torch.inference_mode()
def set_reference(m,d,ts):
    sample=np.sort(np.random.default_rng(20261018).choice(ts,len(m.reference_u),replace=False))
    m.eval()
    m.reference_u.copy_(m.encode_rna(d.rna[torch.as_tensor(sample,device=d.device)]).float())
    m.calibration_ready.fill_(True)
    return d.sample_idx[sample]


def checkpoint(m,norm,genes,freeze,seed,arm,fit_sample_ids,fit_cpg_ids,reference_ids):
    return dict(schema='alphagenome_local_chr1_v2',model_state=m.state_dict(),
                model_spec=dict(m.model_spec,mean_blend=m.mean_blend),
                normalization={k:torch.as_tensor(v) for k,v in norm.items()},gene_ids=genes,
                freeze=freeze,seed=seed,arm=arm,step=1000,ema_decay=.99,
                fit_sample_ids=fit_sample_ids,fit_cpg_ids=fit_cpg_ids,reference_sample_ids=reference_ids,
                graph_weight=ARMS[arm][1],code_sha256=code_hashes(),
                sources=['array'],reference_genome='hg38',coordinate_convention='1-based cytosine',
                ag_source=str(AG),ag_dim=1536)


def train(d,ts,tc,arm,seed,prior,graph,dest,norm,genes,freeze=None):
    path=dest/'checkpoint.pt'
    if path.exists():
        st=torch.load(path,map_location='cpu',weights_only=False)
        m=build_alphagenome_predictor(**st['model_spec']).cuda().eval()
        m.load_state_dict(st['model_state']); return m,st['reference_sample_ids']
    dest.mkdir(parents=True,exist_ok=True)
    m=new_model(d.rna.shape[1],arm,seed)
    for key,val in prior.items(): getattr(m,key).copy_(torch.as_tensor(val,device=d.device))
    ema=copy.deepcopy(m).requires_grad_(False).eval()
    opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4,fused=True)
    gen=torch.Generator(device='cuda').manual_seed(seed*7919+13)
    ngen=torch.Generator(device='cuda').manual_seed(seed*7919+1301)
    ts_t=torch.as_tensor(ts,device=d.device); tc_t=torch.as_tensor(tc,device=d.device)
    mu=training_means(d,ts,tc); weight=ARMS[arm][1]
    start=time.time(); history=[]
    for step in range(1000):
        m.train()
        for g in opt.param_groups: g['lr']=cosine_lr(step,4000,200,2e-4,.1)
        s=ts_t[torch.randperm(len(ts),generator=gen,device=d.device)[:256]].sort().values
        c=tc_t[torch.randperm(len(tc),generator=gen,device=d.device)[:512]].sort().values
        if weight:
            neighbor=graph[c,torch.randint(5,(len(c),),device=d.device,generator=ngen)]
            joined,inv=torch.unique(torch.cat([c,neighbor]),sorted=True,return_inverse=True)
            anchor,other=inv[:len(c)],inv[len(c):]
        else: joined=c; anchor=torch.arange(len(c),device=d.device)
        y=d.block(s,c); mask=torch.isfinite(y); y=y.nan_to_num()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=m(d.rna[s],**d.functional(joined),return_internals=bool(weight))
        p=result['beta'][:,anchor].float()
        loss=masked_mse(p,y,mask)+.15*locus_pcc_loss(p,y,mask)
        loss=loss+.15*(result['mu_hat'][anchor].float()-mu[c]).square().mean()
        wm=mask & d.known_tissue[s,None]
        loss=loss+group_centered_mse_loss(p,y,wm,d.tissue_id[s])+.15*group_centered_pcc_loss(p,y,wm,d.tissue_id[s])
        reg=p.sum()*0
        if weight:
            reg=local_response_penalty(result['rna_factors'],result['cpg_factors'][anchor],result['cpg_factors'][other])
            loss=loss+weight*reg
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),m.parameters()): ep.lerp_(p,.01)
        if (step+1)%500==0:
            event=dict(arm=arm,seed=seed,step=step+1,loss=float(loss),graph_penalty=float(reg),seconds=time.time()-start)
            history.append(event); print(json.dumps(event),flush=True)
    ref_ids=set_reference(ema,d,ts)
    torch.save(checkpoint(ema,norm,genes,freeze,seed,arm,d.sample_idx[ts],d.cpg_idx[tc],ref_ids),path)
    dump(dest/'training.json',dict(history=history,train_seconds=time.time()-start,
                                   patients=len(ts),cpgs=len(tc),parameters=sum(p.numel() for p in m.parameters())))
    del m,opt
    return ema,ref_ids


@torch.inference_mode()
def predict_alphas(m,d,ss,cc,alphas):
    """Shared FP32 forward for fair calibration comparisons and exact caches."""
    m.eval()
    u=m.encode_rna(d.rna[torch.as_tensor(ss,device=d.device)]).float()
    result={str(a):np.empty((len(ss),len(cc)),np.float32) for a in alphas}
    for j in range(0,len(cc),512):
        c=torch.as_tensor(cc[j:j+512],device=d.device); f=d.functional(c)
        raw,a=m.encode_cpg(**f,calibrate=False)
        bias={alpha:m.calibrated_intercept(raw,a,f['dense'][:,23:],blend=alpha) for alpha in alphas}
        for i in range(0,len(ss),256):
            response=u[i:i+256]@a.T/m.rank**.5
            for alpha,b in bias.items():
                p=(b[None]+response).sigmoid()
                assert torch.isfinite(p).all()
                result[str(alpha)][i:i+len(p),j:j+len(c)]=p.cpu().numpy()
    return result


def validate(m,d,spec,dest,alphas):
    name,ts,vs,tc,vc=spec
    if (dest/'validation.json').exists(): return json.loads((dest/'validation.json').read_text())
    truth=d.block(torch.as_tensor(vs,device=d.device),torch.as_tensor(vc,device=d.device)).cpu().numpy()
    predictions=predict_alphas(m,d,vs,vc,alphas)
    seen=np.sort(np.random.default_rng(20261018).choice(tc,min(1024,len(tc)),replace=False))
    known_y=d.block(torch.as_tensor(vs,device=d.device),torch.as_tensor(seen,device=d.device)).cpu().numpy()
    known=predict_alphas(m,d,vs,seen,alphas)
    records={}
    for alpha in alphas:
        key=str(alpha); summary,per=score(truth,predictions[key],d.tissue[vs])
        km,_=score(known_y,known[key],d.tissue[vs])
        records[key]=dict(double_ood=summary,known_loci_new_patients=km)
        np.savez_compressed(dest/f'alpha{key}_per_cpg.npz',**per,cpg_idx=d.cpg_idx[vc])
    dump(dest/'validation.json',records)
    print(json.dumps({'validation':str(dest),'results':records}),flush=True)
    return records


def select_recipe(out,seeds):
    candidates=[]
    for arm in ARMS:
        for alpha in ALPHAS:
            docs=[json.loads((out/f'fold{k}'/arm/f'seed{s}/validation.json').read_text())[str(alpha)]
                  for k in range(3) for s in seeds]
            metrics={view:{metric:float(np.mean([d[view][metric] for d in docs]))
                           for metric in docs[0][view]} for view in docs[0]}
            candidates.append(dict(arm=arm,mean_blend=alpha,**metrics))
    baseline=next(c for c in candidates if c['arm']=='baseline' and c['mean_blend']==0.)
    b=baseline['double_ood']; kb=baseline['known_loci_new_patients']
    for c in candidates:
        m=c['double_ood']; k=c['known_loci_new_patients']
        c['admissible']=(m['mse']<=b['mse'] and m['mas_pcc']>=b['mas_pcc']-.005 and
                         m['within_tissue']>=b['within_tissue']-.005 and
                         k['mas_pcc']>=kb['mas_pcc']-.01 and k['mse']<=kb['mse']*1.05)
    new=[c for c in candidates if not(c['arm']=='baseline' and c['mean_blend']==0)]
    admissible=[c for c in new if c['admissible']]
    pool=admissible or new
    selected=min(pool,key=lambda c:(c['double_ood']['mse'],-c['double_ood']['mas_pcc'],ARMS[c['arm']][1],c['mean_blend']))
    freeze=dict(schema='alphagenome_local_chr1_v2',selection_data='three_internal_genomic_folds_only',
                selected=selected,baseline=baseline,candidates=candidates,guardrails_passed=bool(admissible),
                steps=1000,scheduler_horizon=4000,seeds=seeds,code_sha256=code_hashes(),
                rule='Min double-OOD MSE, PCC/WT no worse than -0.005; known-locus PCC -0.01 and MSE +5%; no official tuning')
    dump(out/'frozen_recipe.json',freeze)
    print(json.dumps({'freeze':selected,'guardrails_passed':bool(admissible)}),flush=True)
    return freeze


def prepare_official(out):
    d=Data(LAB/'cache/array_chr1.npz','official',torch.device('cuda'))
    d.raw_dense=d.dense.float().cpu().numpy(); d.block_ids=d.cpg_pos_bp.cpu().numpy()//5000000
    d.beta=torch.empty(0,device=d.device)
    with h5py.File(CANON/'rna/tcga_rna_official_full.h5','r') as h:
        d.raw_rna=rows(h,'X',positions(h['sample_idx'][:],d.sample_idx)).astype(np.float32)
    with h5py.File(PREP/'methylation/array_table5_chr1.h5','r') as h:
        y=selected_matrix(h['beta'],positions(h['sample_idx'][:],d.sample_idx),positions(h['cpg_idx'][:],d.cpg_idx))
    d.beta=torch.as_tensor(y,device=d.device)
    with h5py.File(AG,'r') as h:
        d.raw_ag=h['embedding'][:][positions(h['cpg_idx'][:],d.cpg_idx)].astype(np.float32)
    unknown=np.isin(d.tissue,['NA','nan','None','unknown',''])
    ids=d.tissue_id.cpu().numpy()
    ids[unknown]=np.arange(ids.max()+1,ids.max()+1+unknown.sum())
    d.tissue_id=torch.as_tensor(ids,device=d.device)
    d.known_tissue=torch.as_tensor(~unknown,device=d.device)
    return d


def error_frame(d,ss,cc,y,p):
    summary,m=score(y,p,d.tissue[ss])
    error=np.where(np.isfinite(y),np.abs(p-y),np.nan)
    m['mae']=np.nanmean(error,0); m['rmse']=np.sqrt(m['mse'])
    m['abs_error_p90'],m['abs_error_p95']=np.nanquantile(error,[.9,.95],axis=0)
    m['max_abs_error']=np.nanmax(error,0)
    m['error_gt_0p1_fraction']=np.nanmean(np.where(np.isfinite(y),error>.1,np.nan),0)
    summary['mae']=float(np.sum(m['mae']*m['n'])/m['n'].sum())
    summary['rows']=int(m['n'].sum())
    f=pd.DataFrame(m)
    f.insert(0,'cpg_idx',d.cpg_idx[cc]); f.insert(1,'chr','chr1')
    f.insert(2,'pos_1based',d.cpg_pos_bp[torch.as_tensor(cc,device=d.device)].cpu().numpy())
    f['block_5mb']=f.pos_1based//5000000
    dense=d.raw_dense[cc]
    f['context']=np.array(['island','shore','shelf','open_sea'])[dense[:,:4].argmax(1)]
    f['region']=np.array(['promoter','exon','intron','intergenic'])[dense[:,4:8].argmax(1)]
    f['observed_sd_bin']=pd.cut(f['std'],[-1,.05,.15,np.inf],labels=['low','medium','high'])
    f['coordinate']='chr1:'+f.pos_1based.astype(str)
    return summary,f


def summarize_errors(frame):
    def group(part):
        n=part['n'].to_numpy()
        return dict(cpgs=len(part),mse=float(np.average(part['mse'],weights=n)),
                    rmse=float(np.sqrt(np.average(part['mse'],weights=n))),
                    mae=float(np.average(part['mae'],weights=n)),median_pcc=float(part['corr'].median()),
                    error_gt_0p2_fraction=float(np.average(part['severe'],weights=n)),
                    bias_fraction=float(np.sum(part['bias2']*n)/np.sum(part['mse']*n)))
    result={'all':group(frame),'strata':{}}
    for col in ['context','region','observed_sd_bin','block_5mb']:
        result['strata'][col]={str(k):group(v) for k,v in frame.groupby(col,observed=True) if len(v)>=10}
    sorted_frame=frame.sort_values('mse',ascending=False)
    total_sse=np.sum(frame['mse']*frame['n'])
    result['worst_decile_sse_fraction']=float(np.sum(sorted_frame.head(max(1,int(np.ceil(len(frame)*.1))))['mse']*
                                                     sorted_frame.head(max(1,int(np.ceil(len(frame)*.1))))['n'])/total_sse)
    result['worst_20']=sorted_frame.head(20).replace({np.nan:None}).to_dict(orient='records')
    return result


def load_v1(d):
    st=torch.load(LAB/'final_seed17/checkpoint.pt',map_location='cpu',weights_only=False)
    m=build_alphagenome_predictor(input_dim=d.raw_rna.shape[1],shallow_mean=False).cuda().eval()
    result=m.load_state_dict(st['model_state'],strict=False)
    assert not result.unexpected_keys
    return m,st


def apply_normalization(d,norm):
    def cpu(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    norm={k:cpu(v) for k,v in norm.items()}
    d.rna=torch.as_tensor((d.raw_rna-norm['rna_mean'])/norm['rna_std'],device=d.device)
    dense=d.raw_dense.copy(); dense[:,17]=(dense[:,17]-norm['dense17_mean'])/norm['dense17_std']
    d.dense=torch.as_tensor(np.c_[dense,d.raw_ag],device=d.device)


def audit_official(m,d,dest,seed,baseline_frames=None):
    dest.mkdir(parents=True,exist_ok=True)
    views={'new_patients':(d.val_s.cpu().numpy(),d.train_c.cpu().numpy()),
           'new_cpgs':(d.train_s.cpu().numpy(),d.val_c.cpu().numpy()),
           'double_ood':(d.val_s.cpu().numpy(),d.val_c.cpu().numpy())}
    protocol=json.loads((PREP/'table5_protocol/protocol.json').read_text())
    count_keys={'new_patients':'train_cpg_x_val_sample','new_cpgs':'val_cpg_x_train_sample','double_ood':'val_cpg_x_val_sample'}
    result={}; frames={}
    for view,(ss,cc) in views.items():
        y=d.block(torch.as_tensor(ss,device=d.device),torch.as_tensor(cc,device=d.device)).cpu().numpy()
        p=predict_alphas(m,d,ss,cc,[m.mean_blend])[str(m.mean_blend)]
        metrics,frame=error_frame(d,ss,cc,y,p)
        expected=protocol['finite_pair_audit']['evaluation_observed'][count_keys[view]]
        assert metrics['rows']==expected,(view,metrics['rows'],expected)
        result[view]=dict(metrics=metrics,audit=summarize_errors(frame))
        frame.to_csv(dest/f'{view}_per_cpg.csv.gz',index=False)
        frame.sort_values('mse',ascending=False).head(100).to_csv(dest/f'{view}_worst100.csv',index=False)
        if baseline_frames is not None:
            base=baseline_frames[view]
            assert np.array_equal(base.cpg_idx,frame.cpg_idx) and np.array_equal(base.n,frame.n)
            comp=frame.copy()
            for metric in ['mse','mae','bias','corr','severe']:
                comp[f'v1_{metric}']=base[metric].to_numpy()
                comp[f'delta_{metric}']=comp[metric]-comp[f'v1_{metric}']
            comp.to_csv(dest/f'{view}_vs_v1.csv.gz',index=False)
            result[view]['vs_v1']=dict(mse_difference=float(np.average(comp.delta_mse,weights=comp.n)),
                                       fraction_cpg_mse_improved=float(np.mean(comp.delta_mse<0)),
                                       block_uncertainty=block_ci(comp.delta_mse.to_numpy(),comp.n.to_numpy(),comp.block_5mb.to_numpy()))
        frames[view]=frame
        print(json.dumps({'official_audit':str(dest),'view':view,'metrics':metrics}),flush=True)
    dump(dest/'metrics.json',dict(seed=seed,views=result,precision='FP32, cached factors, common chunk sizes'))
    return result,frames


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/alphagenome_v2_2026-10-18')
    args=ap.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); threadpool_limits(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.is_available()
    seeds=[17,18,19]
    protocol=dict(arms=ARMS,mean_blend=ALPHAS,seeds=seeds,steps=1000,batch=[256,512],
                  lr=.0002,weight_decay=.0001,ema=.99,scheduler_horizon=4000,
                  selection='Three internal block folds, min MSE with PCC/WT/known-locus guardrails',
                  graph='raw AG train-only standardized PCA64, 5 neighbors >=1Mb, context/region/TSS matching',
                  mean_prior='Ridge alpha100 on train-patient means at train loci only; clipped .005-.995',
                  mean_calibration='Monotone intercept solve against 512 training RNA reference factors',
                  code_sha256=code_hashes())
    protocol=json.loads(json.dumps(protocol))
    if (out/'protocol.json').exists(): assert json.loads((out/'protocol.json').read_text())==protocol
    else: dump(out/'protocol.json',protocol)
    with h5py.File(CANON/'rna/tcga_rna_official_full.h5','r') as h:
        genes=[g.decode() if isinstance(g,bytes) else str(g) for g in h['gene_ids'][:]]
    d=prepare(torch.device('cuda'),out)
    specs=folds(d)
    for spec in specs[:3]:
        name,ts,vs,tc,vc=spec; dest=out/name; dest.mkdir(exist_ok=True)
        norm=normalize(d,ts,tc); np.savez_compressed(dest/'normalization.npz',**norm)
        prior=fit_prior(d,tc,training_means(d,ts,tc),dest)
        graph=response_graph(d,tc,dest)
        for seed in seeds:
            for arm in ARMS:
                path=dest/arm/f'seed{seed}'
                m,_=train(d,ts,tc,arm,seed,prior,graph,path,norm,genes)
                validate(m,d,spec,path,ALPHAS)
                del m; torch.cuda.empty_cache()
    if (out/'frozen_recipe.json').exists(): freeze=json.loads((out/'frozen_recipe.json').read_text())
    else: freeze=select_recipe(out,seeds)
    selected=freeze['selected']; arm=selected['arm']; alpha=selected['mean_blend']
    # One dev confirmation of the frozen recipe; no changes follow its results.
    spec=specs[-1]; name,ts,vs,tc,vc=spec; dest=out/'dev'; dest.mkdir(exist_ok=True)
    norm=normalize(d,ts,tc); np.savez_compressed(dest/'normalization.npz',**norm)
    prior=fit_prior(d,tc,training_means(d,ts,tc),dest); graph=response_graph(d,tc,dest)
    for seed in seeds:
        for label in dict.fromkeys(['baseline',arm]):
            path=dest/label/f'seed{seed}'
            m,_=train(d,ts,tc,label,seed,prior,graph,path,norm,genes,freeze)
            choices=([0.] if arm!='baseline' else list(dict.fromkeys([0.,alpha]))) if label=='baseline' else [alpha]
            validate(m,d,spec,path,choices)
            del m; torch.cuda.empty_cache()
    del d; torch.cuda.empty_cache()
    d=prepare_official(out)
    ts=d.train_s.cpu().numpy(); tc=d.train_c.cpu().numpy()
    norm=normalize(d,ts,tc); dest=out/'official'; dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'normalization.npz',**norm)
    prior=fit_prior(d,tc,training_means(d,ts,tc),dest); graph=response_graph(d,tc,dest)
    base_path=dest/'baseline_v1'; base_frames={}
    if (base_path/'metrics.json').exists():
        for view in ['new_patients','new_cpgs','double_ood']:
            base_frames[view]=pd.read_csv(base_path/f'{view}_per_cpg.csv.gz')
    else:
        baseline,old=load_v1(d); apply_normalization(d,old['normalization'])
        _,base_frames=audit_official(baseline,d,base_path,17)
        del baseline,old; apply_normalization(d,norm)
    for seed in seeds:
        path=dest/f'seed{seed}'
        m,ref_ids=train(d,ts,tc,arm,seed,prior,graph,path,norm,genes,freeze)
        m.mean_blend=alpha
        torch.save(checkpoint(m,norm,genes,freeze,seed,arm,d.sample_idx[ts],d.cpg_idx[tc],ref_ids),path/'checkpoint.pt')
        if not (path/'metrics.json').exists(): audit_official(m,d,path,seed,base_frames)
        # Reload through the portable, target-free API and check real inputs.
        adapter=AlphaGenomePredictorFromCheckpoint(path/'checkpoint.pt',device='cuda')
        c=d.val_c[:13]; s=d.val_s[:7]; func=d.functional(c)
        value=adapter(d.raw_rna[s.cpu().numpy()],genes,func['track_indices'],func['offsets'],
                      d.raw_dense[c.cpu().numpy()],d.raw_ag[c.cpu().numpy()])
        m.eval()
        with torch.inference_mode(): expected=m(d.rna[s],**func)['beta']
        diff=float((value-expected).abs().max())
        assert diff<1e-6,diff
        dump(path/'inference_check.json',dict(max_abs_error=diff,passed=True,uses_methylation=False))
        del m,adapter; torch.cuda.empty_cache()
    dump(out/'complete.json',dict(complete=True,selected=selected,guardrails_passed=freeze['guardrails_passed'],
                                  primary_checkpoint=str(dest/'seed17/checkpoint.pt'),code_sha256=code_hashes()))
    print(json.dumps({'complete':True,'primary_checkpoint':str(dest/'seed17/checkpoint.pt')}),flush=True)


if __name__=='__main__': main()
