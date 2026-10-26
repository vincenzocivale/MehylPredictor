"""Train controlled local-sequence/AG corrections with the RNA backbone frozen."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from methylation_predictor.modeling.factory import build_alphagenome_residual_head
from verify_alphagenome_capacity import prepare,folds,normalize,dump,masked_mse,locus_pcc_loss,group_centered_mse_loss,group_centered_pcc_loss
from search_alphagenome_hierarchical import load
from search_alphagenome_architecture import factors
from validate_alphagenome_hypotheses import OUT,SEEDS,t,evaluate,event

FASTA=ROOT.parent/'CpGRepresentationBenchmark/data/external/reference/hg38.fa'
ARMS=['ag','sequence','hybrid','shuffled_sequence']


def sequences(d,out):
    path=out/'sequence.npz'
    if path.exists():
        z=np.load(path);assert np.array_equal(z['cpg_idx'],d.cpg_idx)
        return z['sequence']
    f=Fasta(str(FASTA),as_raw=True,sequence_always_upper=True,rebuild=False)
    code=np.full(256,4,np.uint8)
    for i,c in enumerate('ACGT'):code[ord(c)]=i
    rows=[];positions=d.cpg_pos_bp.cpu().numpy()
    for pos in positions:
        start=int(pos)-1-500;stop=int(pos)+1+500
        seq='N'*max(0,-start)+str(f['chr1'][max(0,start):min(stop,len(f['chr1']))])+'N'*max(0,stop-len(f['chr1']))
        assert len(seq)==1002
        rows.append(code[np.frombuffer(seq.encode(),dtype=np.uint8)])
    seq=np.stack(rows)
    assert np.all(seq[:,500]==1) and np.all(seq[:,501]==2),'Registry cytosines must match reference CpG dyads'
    np.savez_compressed(path,cpg_idx=d.cpg_idx,sequence=seq)
    f.close()
    return seq


@torch.inference_mode()
def correction(head,ag,seq):
    values=[head(a,s) for a,s in zip(ag.split(512),seq.split(512))]
    return torch.cat([v[0] for v in values]),torch.cat([v[1] for v in values])


def fit_one(d,spec,seed,arm,seq,out):
    name,ts,vs,tc,vc=spec;dest=out/'sequence'/name/f'seed{seed}'/arm;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metrics.json').exists():return json.loads((dest/'metrics.json').read_text())
    m=load(name,'v2_early',seed).requires_grad_(False)
    with torch.inference_mode():
        b,a=factors(m,d,np.arange(len(d.cpg_idx)))
        u=m.encode_rna(d.rna).float()
    # Inference tensors must be cloned outside inference mode before autograd.
    b,a,u=b.clone(),a.clone(),u.clone()
    ag=t(d.raw_ag)
    if arm=='shuffled_sequence':
        order=np.random.default_rng(20261018).permutation(len(seq));ss=seq[t(order)]
    else:ss=seq
    torch.manual_seed(seed*7919+3301)
    head=build_alphagenome_residual_head(mode='sequence' if arm=='shuffled_sequence' else arm).cuda()
    ema=copy.deepcopy(head).requires_grad_(False).eval()
    opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.01,fused=True)
    gen=torch.Generator(device='cuda').manual_seed(seed*7919+13)
    means=torch.full_like(b,torch.nan)
    for cc in t(tc).split(512):means[cc]=d.block(t(ts),cc).nanmean(0)
    train_s,train_c=t(ts),t(tc);start=time.time()
    for step in range(1000):
        head.train()
        s=train_s[torch.randperm(len(ts),generator=gen,device='cuda')[:256]].sort().values
        c=train_c[torch.randperm(len(tc),generator=gen,device='cuda')[:256]].sort().values
        y=d.block(s,c);mask=torch.isfinite(y);y=y.nan_to_num()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            db,da=head(ag[c],ss[c])
            change=db[None].float()+u[s]@da.float().T/m.rank**.5
            p=(b[c][None]+u[s]@a[c].T/m.rank**.5+change).sigmoid()
            wm=mask & d.known_tissue[s,None]
            loss=masked_mse(p,y,mask)+.15*locus_pcc_loss(p,y,mask)
            loss=loss+group_centered_mse_loss(p,y,wm,d.tissue_id[s])+.15*group_centered_pcc_loss(p,y,wm,d.tissue_id[s])
            loss=loss+.15*((b[c]+db.float()).sigmoid()-means[c]).square().mean()+.001*change.square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite residual loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),head.parameters()):ep.lerp_(p,.01)
    with torch.inference_mode():
        db,da=correction(ema,ag[t(vc)],ss[t(vc)])
        bias=m.calibrated_intercept(b[t(vc)]+db,a[t(vc)]+da,ag[t(vc)],blend=.25)
        pred=(bias[None]+u[t(vs)]@(a[t(vc)]+da).T/m.rank**.5).sigmoid()
        metric,per=evaluate(d.block(t(vs),t(vc)),pred,d.tissue[vs])
    result=dict(split=name,seed=seed,arm=arm,parameters=sum(p.numel() for p in head.parameters()),
                training_seconds=time.time()-start,**metric)
    torch.save(dict(model_spec=head.model_spec,model_state=ema.state_dict(),base_source='v2_early',seed=seed,split=name,
                    fit_sample_ids=d.sample_idx[ts],fit_cpg_ids=d.cpg_idx[tc],window=1002,
                    coordinate_convention='hg38 CpG dyad at window indices500/501',
                    sequence_mapping='fixed global permutation' if arm=='shuffled_sequence' else 'aligned'),dest/'checkpoint.pt')
    np.savez_compressed(dest/'per_cpg.npz',cpg_idx=d.cpg_idx[vc],**per)
    dump(dest/'metrics.json',result);event(stage='sequence',**result)
    del m,head,ema,opt,b,a,u;torch.cuda.empty_cache()
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(arms=ARMS,seeds=SEEDS,steps=1000,learning_rate=.0003,weight_decay=.01,change_penalty=.001,
                  frozen='v2 RNA encoder and CpG predictor; only residual head learns',novel_calibration=.25,
                  sequence='1002bp dyad centred, reverse complement averaged',selection='internal folds; all arms reported on dev without retuning',
                  sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__),ROOT/'src/methylation_predictor/modeling/alphagenome_residual.py']})
    target=out/'sequence_protocol.json'
    if target.exists():assert json.loads(target.read_text())==protocol
    else:dump(target,protocol)
    torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False
    features=out/'sequence_features';features.mkdir(exist_ok=True)
    d=prepare(torch.device('cuda'),features)
    seq=t(sequences(d,out))
    results=[]
    for spec in folds(d):
        normalize(d,spec[1],spec[3])
        for seed in SEEDS:
            for arm in ARMS:results.append(fit_one(d,spec,seed,arm,seq,out))
    pd.DataFrame(results).to_csv(out/'sequence_metrics.csv',index=False)
    dump(out/'sequence_complete.json',dict(complete=True,official_targets_used=False))


if __name__=='__main__':main()
