"""Internal architecture search: frozen mean, frozen CpG, and response retrieval.

All continuation arms restart AdamW from the same saved v2 EMA checkpoint.
This is explicitly not an uninterrupted 2,000-step training comparison.
No official holdout targets are selected, scored, or used for model selection.
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
from sklearn.preprocessing import StandardScaler, normalize as unit_rows
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from methylation_predictor.modeling.factory import build_alphagenome_predictor
from methylation_predictor.modeling.alphagenome import local_response_penalty, AlphaGenomePredictorFromCheckpoint
from methylation_predictor.modeling.alphagenome_transfer import transfer_response
from verify_alphagenome_capacity import (prepare, folds, normalize, score, dump, CANON,
    cosine_lr, masked_mse, locus_pcc_loss, group_centered_mse_loss, group_centered_pcc_loss)
from train_alphagenome_v2 import training_means, set_reference

PARENT = ROOT/'artifacts/alphagenome_v2_2026-10-18'
SEEDS = [17,18,19]
MODES = ['joint', 'freeze_mean', 'freeze_cpg']
# Random-matched retrieval is a negative control, excluded from selection.
TRANSFERS = [('none',0.), ('raw',.25), ('raw',.5), ('learned',.25), ('learned',.5), ('random',.5)]


def hashes():
    paths = [Path(__file__), ROOT/'src/methylation_predictor/modeling/alphagenome_transfer.py',
             ROOT/'src/methylation_predictor/modeling/alphagenome.py',
             ROOT/'src/methylation_predictor/modeling/factory.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_parent(path, two_stage=True):
    st = torch.load(path, map_location='cpu', weights_only=False)
    spec = dict(st['model_spec'])
    if two_stage: spec['variant'] = 'alphagenome_two_stage_chr1_v3'
    m = build_alphagenome_predictor(**spec).cuda()
    m.load_state_dict(st['model_state'], strict=True)
    return m, st


@torch.inference_mode()
def factors(m, d, cc):
    m.eval(); bias, response = [], []
    for piece in torch.as_tensor(cc, device=d.device).split(512):
        b,a = m.encode_cpg(**d.functional(piece), calibrate=False)
        bias.append(b.float()); response.append(a.float())
    return torch.cat(bias), torch.cat(response)


def neighbor_table(d, tc, features, path):
    """Fit preprocessing only on training loci; restrict every neighbor to tc."""
    if path.exists():
        z = np.load(path)
        assert np.array_equal(z['fit_cpg_idx'],d.cpg_idx[tc])
        assert np.array_equal(z['query_cpg_idx'],d.cpg_idx)
        return {k:torch.as_tensor(z[k],device=d.device) for k in ['nearest','random']}
    scaler = StandardScaler().fit(features[tc])
    x = scaler.transform(features)
    if x.shape[1] > 64:
        pc = PCA(n_components=64,svd_solver='randomized',random_state=20261018).fit(x[tc])
        x = pc.transform(x)
    x = torch.as_tensor(unit_rows(x).astype(np.float32),device=d.device)
    fit = torch.as_tensor(tc,device=d.device)
    pos = d.cpg_pos_bp
    dense = d.raw_dense
    group = dense[:,:4].argmax(1)*16+dense[:,4:8].argmax(1)*4
    group += np.searchsorted(np.quantile(dense[tc,17],[.25,.5,.75]),dense[:,17])
    tables = {k:torch.empty((len(d.cpg_idx),5),device=d.device,dtype=torch.long) for k in ['nearest','random']}
    gen = torch.Generator(device=d.device).manual_seed(20261018)
    fallback = 0
    for g in np.unique(group):
        query = torch.as_tensor(np.flatnonzero(group==g),device=d.device)
        bank = fit[torch.as_tensor(group[tc]==g,device=d.device)]
        for q in query.split(256):
            if len(bank) < 5:
                bad = q; good = q[:0]
            else:
                eligible = (pos[q,None]-pos[bank][None]).abs() >= 1_000_000
                valid = eligible.sum(1)>=5
                good,bad = q[valid],q[~valid]
                if len(good):
                    elig = eligible[valid]
                    sim = (x[good]@x[bank].T).masked_fill(~elig,-torch.inf)
                    tables['nearest'][good] = bank[sim.topk(5,1).indices]
                    rnd = torch.rand(sim.shape,device=d.device,generator=gen).masked_fill(~elig,-torch.inf)
                    tables['random'][good] = bank[rnd.topk(5,1).indices]
            if len(bad):
                fallback += len(bad)
                elig = (pos[bad,None]-pos[fit][None]).abs() >= 1_000_000
                assert (elig.sum(1)>=5).all()
                sim = (x[bad]@x[fit].T).masked_fill(~elig,-torch.inf)
                tables['nearest'][bad] = fit[sim.topk(5,1).indices]
                rnd = torch.rand(sim.shape,device=d.device,generator=gen).masked_fill(~elig,-torch.inf)
                tables['random'][bad] = fit[rnd.topk(5,1).indices]
    for t in tables.values():
        assert np.isin(t.cpu().numpy(),tc).all()
        assert ((pos[:,None]-pos[t]).abs()>=1_000_000).all()
    np.savez_compressed(path,**{k:v.cpu().numpy() for k,v in tables.items()},
                        fit_cpg_idx=d.cpg_idx[tc],query_cpg_idx=d.cpg_idx,fallback_loci=fallback)
    return tables


def continue_fit(d,spec,mode,seed,out,norm,genes):
    name,ts,vs,tc,vc = spec
    dest = out/name/mode/f'seed{seed}'; dest.mkdir(parents=True,exist_ok=True)
    path = dest/'checkpoint.pt'
    if path.exists():
        st = torch.load(path,map_location='cpu',weights_only=False)
        m = build_alphagenome_predictor(**st['model_spec']).cuda().eval()
        m.load_state_dict(st['model_state']); return m
    source = PARENT/name/'local01'/f'seed{seed}'/'checkpoint.pt'
    m,parent = load_parent(source)
    # Explicit independent seeds for dropout, paired identically across arms.
    torch.manual_seed(seed*7919+2026)
    if mode == 'freeze_mean': m.freeze_mean()
    if mode == 'freeze_cpg':
        for n,p in m.named_parameters():
            p.requires_grad_(n.startswith(('rna_encoder.','rna_factor.')))
    ema = copy.deepcopy(m).requires_grad_(False).eval()
    trainable = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable,lr=2e-4,weight_decay=1e-4,fused=True)
    gen = torch.Generator(device=d.device).manual_seed(seed*7919+13)
    ngen = torch.Generator(device=d.device).manual_seed(seed*7919+1301)
    ts_t,tc_t = torch.as_tensor(ts,device=d.device),torch.as_tensor(tc,device=d.device)
    graph = torch.as_tensor(np.load(PARENT/name/'response_graph.npz')['neighbors'],device=d.device)
    mu = training_means(d,ts,tc)
    if mode == 'freeze_cpg':
        fixed_b,fixed_a = factors(m,d,np.arange(len(d.cpg_idx)))
    start = time.time(); history=[]
    for step in range(1000):
        m.train()
        for g in opt.param_groups: g['lr'] = cosine_lr(step,2000,100,2e-4,.1)
        s = ts_t[torch.randperm(len(ts),device=d.device,generator=gen)[:256]].sort().values
        c = tc_t[torch.randperm(len(tc),device=d.device,generator=gen)[:512]].sort().values
        near = graph[c,torch.randint(5,(len(c),),device=d.device,generator=ngen)]
        joined,inv = torch.unique(torch.cat([c,near]),sorted=True,return_inverse=True)
        anchor,other = inv[:len(c)],inv[len(c):]
        y = d.block(s,c); mask=torch.isfinite(y); y=y.nan_to_num()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if mode == 'freeze_cpg':
                u = m.encode_rna(d.rna[s]); a=fixed_a[joined]; b=fixed_b[joined]
                p = (b[anchor][None]+u@a[anchor].T/m.rank**.5).sigmoid()
                mean = b[anchor].sigmoid()
            else:
                result = m(d.rna[s],**d.functional(joined),return_internals=True)
                p = result['beta'][:,anchor]; mean=result['mu_hat'][anchor]
                u,a = result['rna_factors'],result['cpg_factors']
        p=p.float()
        loss = masked_mse(p,y,mask)+.15*locus_pcc_loss(p,y,mask)+.15*(mean.float()-mu[c]).square().mean()
        wm = mask & d.known_tissue[s,None]
        loss += group_centered_mse_loss(p,y,wm,d.tissue_id[s])+.15*group_centered_pcc_loss(p,y,wm,d.tissue_id[s])
        reg = local_response_penalty(u,a[anchor],a[other]); loss += .01*reg
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.); opt.step()
        with torch.no_grad():
            for ep,p0 in zip(ema.parameters(),m.parameters()): ep.lerp_(p0,.01)
        if (step+1)%500 == 0:
            event=dict(fold=name,mode=mode,seed=seed,step=step+1,loss=float(loss),seconds=time.time()-start)
            print(json.dumps(event),flush=True); history.append(event)
    refs = set_reference(ema,d,ts)
    state = dict(schema='alphagenome_two_stage_chr1_v3',model_spec=ema.model_spec,model_state=ema.state_dict(),
                 normalization=norm,gene_ids=genes,mode=mode,seed=seed,phase1_steps=1000,phase2_steps=1000,
                 phase2_optimizer='AdamW restart from parent EMA',fit_sample_ids=d.sample_idx[ts],fit_cpg_ids=d.cpg_idx[tc],
                 reference_sample_ids=refs,parent_checkpoint=str(source),parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 code_sha256=hashes())
    torch.save(state,path)
    dump(dest/'training.json',dict(history=history,seconds=time.time()-start,trainable_parameters=sum(p.numel() for p in trainable)))
    # Parity of portable checkpoint on actual inputs, using no target values.
    adapter = AlphaGenomePredictorFromCheckpoint(path,device='cuda')
    c=torch.as_tensor(vc[:13],device=d.device); s=vs[:7]; f=d.functional(c)
    with torch.inference_mode():
        expected=ema(d.rna[torch.as_tensor(s,device=d.device)],**f)['beta']
        got=adapter(d.raw_rna[s],genes,f['track_indices'],f['offsets'],d.raw_dense[vc[:13]],d.raw_ag[vc[:13]])
    diff=float((got-expected).abs().max()); assert diff<1e-6
    dump(dest/'inference_check.json',dict(max_abs_error=diff,passed=True))
    return ema


@torch.inference_mode()
def evaluate(m,d,spec,dest,raw_tables,transfers):
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'evaluation.json').exists(): return json.loads((dest/'evaluation.json').read_text())
    name,ts,vs,tc,vc=spec
    b,a = factors(m,d,np.arange(len(d.cpg_idx)))
    # Learned geometry is fitted on training factors only, never query targets.
    if any(space=='learned' for space,blend in transfers):
        learned = neighbor_table(d,tc,a.cpu().numpy(),dest/'learned_neighbors.npz')
    refs=np.sort(np.random.default_rng(20261018).choice(ts,512,replace=False))
    reference=m.encode_rna(d.rna[torch.as_tensor(refs,device=d.device)]).float()
    u=m.encode_rna(d.rna[torch.as_tensor(vs,device=d.device)]).float()
    seen=np.sort(np.random.default_rng(20261018).choice(tc,min(4096,len(tc)),replace=False))
    cc=np.r_[vc,seen]
    cc_t=torch.as_tensor(cc,device=d.device)
    truth=d.block(torch.as_tensor(vs,device=d.device),cc_t).cpu().numpy()
    records={}
    for space,blend in transfers:
        label=f'{space}_{blend:g}'
        prediction=np.empty(truth.shape,np.float32)
        for j in range(0,len(cc),512):
            cols=cc_t[j:j+512]; bias,factor=b[cols],a[cols]
            if blend:
                table=(learned['nearest'] if space=='learned' else
                       raw_tables['random' if space=='random' else 'nearest'])
                bias,factor=transfer_response(bias,factor,a[table[cols]],reference,blend)
            p=(bias[None]+u@factor.T/m.rank**.5).sigmoid()
            prediction[:,j:j+len(cols)]=p.cpu().numpy()
        summary={}
        for view,sl in [('double_ood',slice(0,len(vc))),('known_loci',slice(len(vc),None))]:
            y,p=truth[:,sl],prediction[:,sl]
            metrics,per=score(y,p,d.tissue[vs])
            error=np.where(np.isfinite(y),np.abs(p-y),np.nan)
            per['mae']=np.nanmean(error,0)
            metrics['mae']=float(np.average(per['mae'],weights=per['n']))
            metrics['rows']=int(per['n'].sum())
            summary[view]=metrics
            np.savez_compressed(dest/f'{label}_{view}_per_cpg.npz',**per,cpg_idx=d.cpg_idx[cc[sl]])
        records[label]=summary
    dump(dest/'evaluation.json',records)
    print(json.dumps(dict(evaluation=str(dest),none=records['none_0'])),flush=True)
    return records


def select(out):
    candidates=[]
    modes=['v1_early','v2_early']+MODES
    for mode in modes:
        transfers=[('none',0.)] if mode=='v1_early' else TRANSFERS
        for space,blend in transfers:
            if space=='random': continue
            key=f'{space}_{blend:g}'
            docs=[json.loads((out/f'fold{k}'/mode/f'seed{seed}'/'evaluation.json').read_text())[key]
                  for k in range(3) for seed in SEEDS]
            row=dict(mode=mode,space=space,blend=blend)
            for view in ['double_ood','known_loci']:
                row[view]={metric:float(np.mean([doc[view][metric] for doc in docs])) for metric in docs[0][view]}
            candidates.append(row)
    base=next(c for c in candidates if c['mode']=='v1_early')
    v2=next(c for c in candidates if c['mode']=='v2_early' and c['space']=='none')
    for c in candidates:
        q,k,bq,bk=c['double_ood'],c['known_loci'],base['double_ood'],base['known_loci']
        c['admissible']=(q['mse']<=bq['mse']*.995 and k['mse']<=bk['mse']*1.01 and
                          q['mae']<=bq['mae'] and k['mae']<=bk['mae']*1.01 and
                          q['mas_pcc']>=bq['mas_pcc']-.002 and q['within_tissue']>=bq['within_tissue']-.002 and
                          k['mas_pcc']>=bk['mas_pcc']-.002 and k['within_tissue']>=bk['within_tissue']-.002 and
                          q['severe']<=bq['severe']+.001)
        # Balanced score is descriptive fallback only; it cannot bypass guardrails.
        c['balanced_mse_ratio']=.5*(q['mse']/bq['mse']+k['mse']/bk['mse'])
    eligible=[c for c in candidates if c['admissible']]
    selected=min(eligible,key=lambda c:(c['double_ood']['mse'],c['known_loci']['mse'])) if eligible else None
    fallback=min(candidates,key=lambda c:c['balanced_mse_ratio'])
    frozen=dict(selected=selected,best_balanced=fallback,baseline=base,v2=v2,candidates=candidates,
                guardrails_passed=bool(eligible),selection_data='3 internal genomic folds x 3 seeds only',
                rule='Double MSE >=0.5% better, known MSE/MAE <=+1%; double MAE no worse, PCC/WT >=-0.002, severe <=+0.001',
                code_sha256=hashes())
    dump(out/'frozen_recipe.json',frozen)
    rows=[]
    for c in candidates:
        r={k:c[k] for k in ['mode','space','blend','admissible','balanced_mse_ratio']}
        for view in ['double_ood','known_loci']:
            r.update({view+'_'+k:v for k,v in c[view].items()})
        rows.append(r)
    pd.DataFrame(rows).to_csv(out/'selection.csv',index=False)
    print(json.dumps(dict(selection=selected,best_balanced=fallback)),flush=True)
    return frozen


def run_fold(d,spec,out,genes,confirm=None):
    name,ts,vs,tc,vc=spec
    fold=out/name; fold.mkdir(parents=True,exist_ok=True)
    norm=normalize(d,ts,tc); np.savez_compressed(fold/'normalization.npz',**norm)
    original=np.load(PARENT/name/'normalization.npz')
    for key,value in norm.items(): assert np.array_equal(value,original[key]),key
    raw=neighbor_table(d,tc,d.raw_ag,fold/'raw_neighbors.npz')
    if confirm is None: modes=['v1_early','v2_early']+MODES
    else: modes=list(dict.fromkeys(['v1_early','v2_early',confirm['mode']]))
    for seed in SEEDS:
        for mode in modes:
            dest=fold/mode/f'seed{seed}'
            if (dest/'evaluation.json').exists(): continue
            if mode in ['v1_early','v2_early']:
                arm='baseline' if mode=='v1_early' else 'local01'
                m,_=load_parent(PARENT/name/arm/f'seed{seed}'/'checkpoint.pt')
            else:
                m=continue_fit(d,spec,mode,seed,out,norm,genes)
            choices=([('none',0.)] if mode=='v1_early' else TRANSFERS) if confirm is None else [('none',0.)]
            if confirm is not None and mode==confirm['mode'] and confirm['blend']:
                choices=list(dict.fromkeys(choices+[(confirm['space'],confirm['blend'])]))
            evaluate(m,d,spec,dest,raw,choices)
            del m; torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/alphagenome_architecture_2026-10-18')
    args=ap.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); threadpool_limits(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
    protocol=dict(seeds=SEEDS,modes=MODES,transfers=TRANSFERS,phase1='frozen v2 EMA1000',
                  phase2_steps=1000,optimizer='AdamW restart for every continuation arm',
                  phase2_lr=.0002,weight_decay=.0001,ema=.99,warmup=100,scheduler_horizon=2000,
                  sampling='paired across modes; explicit independent CUDA dropout seed for each replicate',
                  graph_weight=.01,reference_patients=512,known_loci_evaluated=4096,
                  retrieval='5 training neighbors >=1Mb, matched context/region/TSS quartile, global fallback',
                  retrieval_mean='preserve original mean on fixed training RNA references by intercept solve',
                  official_holdout='never selected or scored; development folds already used in earlier work',
                  selection_guardrails='new MSE -0.5%; known MSE/MAE +1%; new MAE <=base; PCC/WT -0.002; severe +0.001',
                  code_sha256=hashes())
    protocol=json.loads(json.dumps(protocol))
    if (out/'protocol.json').exists(): assert json.loads((out/'protocol.json').read_text())==protocol
    else: dump(out/'protocol.json',protocol)
    with h5py.File(CANON/'rna/tcga_rna_official_full.h5','r') as h:
        genes=[x.decode() if isinstance(x,bytes) else str(x) for x in h['gene_ids'][:]]
    d=prepare(torch.device('cuda'),out); specs=folds(d)
    # Explicitly audit selected axes against the official training identities.
    with np.load(PARENT/'features.npz') as z:
        assert np.array_equal(z['cpg_idx'],d.cpg_idx)
        assert np.array_equal(z['train_s'],d.sample_idx[d.base_train_s])
        assert np.array_equal(z['val_s'],d.sample_idx[d.base_val_s])
    dump(out/'input_audit.json',dict(internal_axes_match_v2=True,official_targets_used=False,
                                    samples=len(d.sample_idx),cpgs=len(d.cpg_idx)))
    for spec in specs[:3]: run_fold(d,spec,out,genes)
    frozen=select(out)
    candidate=frozen['selected'] or frozen['best_balanced']
    run_fold(d,specs[-1],out,genes,confirm=candidate)
    dump(out/'complete.json',dict(complete=True,selected=frozen['selected'],best_balanced=frozen['best_balanced'],
                                  official_holdout_used=False,code_sha256=hashes()))
    print(json.dumps({'complete':True,'guardrails_passed':frozen['guardrails_passed']}),flush=True)


if __name__=='__main__': main()
