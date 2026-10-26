"""Measure cold/warm CpG-factor inference and two-model routing overhead."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from verify_alphagenome_capacity import CapacityModel, Data, LAB, CANON, positions, rows
from analyze_alphagenome_capacity import load_per, dump


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts/alphagenome_capacity_2026-10-18')
    args=ap.parse_args(); out=args.out
    torch.set_num_threads(4); threadpool_limits(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    protocol=json.loads((out/'protocol.json').read_text()); seeds=protocol['seeds']
    f=np.load(out/'features.npz'); dev=np.load(out/'dev/evaluation_axes.npz')
    train=np.concatenate([np.load(out/f'fold{k}/evaluation_axes.npz')['cpg_rows'] for k in range(3)])
    target=np.concatenate([np.mean([load_per(out,f'fold{k}','d2',s,1000)['mse']-
                                    load_per(out,f'fold{k}','d8',s,1000)['mse'] for s in seeds],0) for k in range(3)])
    x=np.c_[f['dense'],f['ag']]; te=dev['cpg_rows']
    gate=make_pipeline(StandardScaler(),Ridge(alpha=100.)).fit(x[train],target)
    benefit=gate.predict(x[te]); use=np.zeros(len(te),bool)
    order=np.argsort(-benefit,kind='stable')[:round(.25*len(te))]
    use[order[benefit[order]>0]]=True
    norm=np.load(out/'dev/normalization.npz'); sample=dev['sample_idx'][:256]
    with h5py.File(CANON/'rna/tcga_rna_official_full.h5','r') as h:
        rna=rows(h,'X',positions(h['sample_idx'][:],sample)).astype(np.float32)
    rna=torch.as_tensor((rna-norm['rna_mean'])/norm['rna_std'],device='cuda')
    z=np.load(LAB/'cache/array_chr1.npz',allow_pickle=True)
    original_rows=positions(z['cpg_idx'],f['cpg_idx'])
    ptr=z['track_offsets']; idx=z['track_indices']
    bags=[idx[ptr[c]:ptr[c+1]] for c in original_rows]
    d=object.__new__(Data); d.device=torch.device('cuda')
    d.indptr=torch.as_tensor(np.r_[0,np.cumsum([len(a) for a in bags])],device='cuda')
    d.indices=torch.as_tensor(np.concatenate(bags),device='cuda')
    dense=f['dense'].copy(); dense[:,17]=(dense[:,17]-norm['dense17_mean'])/norm['dense17_std']
    d.dense=torch.as_tensor(np.c_[dense,f['ag']],device='cuda')
    models={}
    for arm in ['d2','d8']:
        m=CapacityModel(rna.shape[1],arm).cuda().eval()
        st=torch.load(out/'dev'/arm/'seed17/step1000.pt',map_location='cpu',weights_only=False)
        m.load_state_dict(st['model']); models[arm]=m
    allcols=torch.as_tensor(te,device='cuda')
    local={'d2':np.flatnonzero(~use),'d8':np.flatnonzero(use)}
    route_cols={arm:torch.as_tensor(te[loc],device='cuda') for arm,loc in local.items()}
    scatter={arm:torch.as_tensor(loc,device='cuda') for arm,loc in local.items()}
    # Bag construction is outside the measured region for every alternative.
    funcs={arm:d.functional(c) for arm,c in route_cols.items() if len(c)}
    func_all=d.functional(allcols)
    def encode(m,func):
        b,a,_,_,_=m.encode_cpg(**func)
        return b,a
    def decode(m,encoded,u=None):
        b,a=encoded
        if u is None: u=m.rna_factor(m.rna_encoder.shared(rna))
        return (b[None]+u@a.T/m.rank**.5).sigmoid()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        cached_full={a:encode(m,func_all) for a,m in models.items()}
        cached_route={a:encode(models[a],func) for a,func in funcs.items()}
        cached_patient={a:m.rna_factor(m.rna_encoder.shared(rna)) for a,m in models.items()}
        def single(arm,warm,patient=False):
            return decode(models[arm],cached_full[arm] if warm else encode(models[arm],func_all),
                          cached_patient[arm] if patient else None)
        def routed(warm,patient=False):
            p=torch.empty((len(sample),len(te)),dtype=torch.bfloat16,device='cuda')
            for arm,func in funcs.items():
                encoded=cached_route[arm] if warm else encode(models[arm],func)
                p[:,scatter[arm]]=decode(models[arm],encoded,cached_patient[arm] if patient else None)
            return p
        reference=single('d2',True)
        reference[:,torch.as_tensor(use,device='cuda')]=single('d8',True)[:,torch.as_tensor(use,device='cuda')]
        max_error=float((routed(True)-reference).abs().max())
        assert max_error<=.008, max_error
        results=[]
        for warm,patient in [(False,False),(True,False),(True,True)]:
            for arm in ['d2','d8','route25']:
                fn=(lambda: routed(warm,patient)) if arm=='route25' else (lambda:single(arm,warm,patient))
                for _ in range(100): fn()
                torch.cuda.synchronize(); timings=[]
                for _ in range(30):
                    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    start.record(); t=time.perf_counter(); fn(); end.record(); torch.cuda.synchronize()
                    timings.append((start.elapsed_time(end),(time.perf_counter()-t)*1000))
                a=np.array(timings)
                results.append({'arm':arm,'cached_cpg_factors':warm,'cached_rna_factors':patient,'gpu_ms_median':float(np.median(a[:,0])),
                                'wall_ms_median':float(np.median(a[:,1])),
                                'wall_ms_iqr':np.quantile(a[:,1],[.25,.75]).tolist()})
    original_tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        reference=decode(models['d2'],encode(models['d2'],func_all))
        selected=torch.as_tensor(use,device='cuda')
        reference[:,selected]=decode(models['d8'],encode(models['d8'],func_all))[:,selected]
        mixed=torch.empty_like(reference)
        for arm,func in funcs.items():
            mixed[:,scatter[arm]]=decode(models[arm],encode(models[arm],func))
        fp32_max_error=float((reference-mixed).abs().max())
        assert fp32_max_error<1e-5,fp32_max_error
    torch.backends.cuda.matmul.allow_tf32=original_tf32
    dump(out/'latency_summary.json',{'gpu':torch.cuda.get_device_name(),'patients':len(sample),'cpgs':len(te),
                                    'route_fraction_deep':float(use.mean()),'max_bf16_prediction_difference':max_error,
                                    'max_strict_fp32_prediction_difference':fp32_max_error,
                                    'warmup_calls_per_condition':100,'timed_calls_per_condition':30,
                                    'results':results,'note':'Resident inputs/weights, shared GPU, no disk I/O or gate fitting/bag assembly. Three conditions: no factors cached, CpG factors cached, both RNA/CpG factors cached. Routing uses two independent RNA encoders when patient factors are not cached and both routes are populated.'})
    mechanisms={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for arm in ['gated','prototype']:
            m=CapacityModel(rna.shape[1],arm).cuda().eval()
            st=torch.load(out/'dev'/arm/'seed17/step1000.pt',map_location='cpu',weights_only=False)
            m.load_state_dict(st['model'])
            h0=m.cpg_query(func_all['track_indices'],func_all['offsets'],func_all['dense'])
            if arm=='gated':
                g=m.gate(h0).sigmoid().float().cpu().numpy().ravel()
                mechanisms[arm]={'mean':float(g.mean()),'quantiles':np.quantile(g,[0,.1,.5,.9,1]).tolist(),
                                  'fraction_below_0p1':float(np.mean(g<.1)),
                                  'note':'Soft capacity gate; the deep branch is always evaluated.'}
            else:
                h=h0
                for block in m.functional_ffn: h=block(h)
                assign=m.assignment(h).softmax(-1).float()
                entropy=-(assign*assign.clamp_min(1e-12).log()).sum(-1)
                aggregate=assign.mean(0)
                mechanisms[arm]={'mean_assignment_entropy':float(entropy.mean()),
                                  'max_entropy':float(np.log(16)),
                                  'effective_prototypes_aggregate':float((-(aggregate*aggregate.clamp_min(1e-12).log()).sum()).exp()),
                                  'mean_max_assignment':float(assign.max(-1).values.mean()),
                                  'note':'Soft prototypes with residual response; factors are not identifiable biological programs.'}
            del m,st
    dump(out/'mechanism_diagnostics.json',mechanisms)
    print(json.dumps({'complete':True,'results':results}),flush=True)


if __name__=='__main__': main()
