"""Post-hoc CpG error audit of the frozen chr1 screening checkpoints.

Requires the external pi_lab data/model snapshot; does not train a model.
Target-derived strata and oracle constant-mean comparisons are descriptive only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(y, p):
    """Per-column population moments on identical finite target/prediction masks."""
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    n = mask.sum(0)
    denom = np.maximum(n, 1)
    ym = np.where(mask, y, 0).sum(0) / denom
    pm = np.where(mask, p, 0).sum(0) / denom
    yc = np.where(mask, y - ym, 0)
    pc = np.where(mask, p - pm, 0)
    vy, vp = (yc * yc).sum(0) / denom, (pc * pc).sum(0) / denom
    cov = (yc * pc).sum(0) / denom
    mse = np.where(mask, (p-y)**2, 0).sum(0) / denom
    valid = (n >= 3) & (vy > 1e-12) & (vp > 1e-12)
    corr = np.full(y.shape[1], np.nan)
    corr[valid] = cov[valid] / np.sqrt(vy[valid]*vp[valid])
    ratio = np.full(y.shape[1], np.nan)
    np.divide(np.sqrt(vp), np.sqrt(vy), out=ratio, where=vy > 1e-12)
    return dict(n=n, mean=ym, pred_mean=pm, std=np.sqrt(vy),
                pred_std=np.sqrt(vp), std_ratio=ratio, corr=corr,
                mse=mse, bias=pm-ym, bias2=(pm-ym)**2,
                centered_mse=vy+vp-2*cov,
                severe=np.where(mask, np.abs(p-y) > .2, False).sum(0)/denom)


def self_test():
    y = np.array([[0., .2], [.2, np.nan], [.8, .7], [1., .9]])
    p = y*.5+.1
    m = metrics(y, p)
    assert np.allclose(m['mse'], m['bias2']+m['centered_mse'])
    assert np.allclose(m['corr'], 1)
    assert np.allclose(m['std_ratio'], .5)
    assert np.array_equal(m['n'], [4, 3])
    assert np.isnan(metrics(y, np.zeros_like(y))['corr']).all()
    assert np.allclose(metrics(y, y)['mse'], 0)


def summarize(frame):
    n = frame.baseline_n.to_numpy()
    result = {'cpgs':len(frame), 'genomic_5mb_blocks':int(frame.block.nunique())}
    for tag in ['baseline', 'candidate']:
        def col(k): return frame[f'{tag}_{k}'].to_numpy()
        finite = np.isfinite(col('corr'))
        sse = (col('mse')*n).sum()
        result[tag] = {
            'mse':float(np.average(col('mse'), weights=n)),
            'rmse':float(np.sqrt(np.average(col('mse'), weights=n))),
            'median_corr':float(np.nanmedian(col('corr'))),
            'p10_corr':float(np.nanquantile(col('corr'), .1)),
            'fraction_negative_corr':float(np.mean(col('corr')[finite]<0)),
            'median_std_ratio':float(np.nanmedian(col('std_ratio'))),
            'squared_bias_fraction_of_sse':float((col('bias2')*n).sum()/sse),
            'fraction_abs_error_gt_0p2':float(np.average(col('severe'),weights=n)),
            'median_within_corr':float(np.nanmedian(col('within_corr'))),
        }
    result['delta_median_corr'] = result['candidate']['median_corr']-result['baseline']['median_corr']
    result['fraction_cpg_mse_improved'] = float(np.mean(frame.candidate_mse < frame.baseline_mse))
    result['fraction_cpg_corr_improved'] = float(np.mean(frame.candidate_corr > frame.baseline_corr))
    return result


def clean_json(x):
    if isinstance(x, dict): return {k:clean_json(v) for k,v in x.items()}
    if isinstance(x, list): return [clean_json(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lab',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--cheap',type=Path,help='Optional matched FFN2 checkpoint for oracle allocation diagnostic')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seen-cpgs',type=int,default=3972)
    args=ap.parse_args()
    self_test()
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    sys.path.insert(0,str(args.lab))
    from harness import Data
    from model import Predictor
    args.out.mkdir(parents=True,exist_ok=True)
    schema=json.loads((Path(__file__).resolve().parents[1]/'resources/locus_features/annotation_core_v1.json').read_text())
    data=Data(args.lab/'cache/array_chr1.npz','dev',torch.device('cuda'))
    rng=np.random.default_rng(20261017)
    trainc=data.train_c.cpu().numpy()
    seen=np.sort(rng.choice(trainc,size=min(args.seen_cpgs,len(trainc)),replace=False))
    novel=data.val_c.cpu().numpy()
    c_all=np.concatenate([seen,novel])
    samples=data.val_s
    y=data.block(samples,torch.as_tensor(c_all,device=data.device)).cpu().numpy()
    predictions={}
    durations={}
    checkpoints=[('baseline',args.baseline),('candidate',args.candidate)]
    if args.cheap is not None: checkpoints.append(('cheap',args.cheap))
    for tag,path in checkpoints:
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['step']==2000
        model=Predictor(data.rna.shape[1],locus_chunk=512,n_ffn_blocks=2 if tag=='cheap' else 8).to(data.device)
        model.load_state_dict(state['ema']);model.eval()
        pred=np.empty_like(y)
        torch.cuda.synchronize();start=time.time()
        with torch.inference_mode():
            for j in range(0,len(c_all),512):
                c=torch.as_tensor(c_all[j:j+512],device=data.device)
                func=data.functional(c)
                for i in range(0,len(samples),128):
                    s=samples[i:i+128]
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        p=model(data.rna[s].float(),**func)['beta'].float()
                    pred[i:i+len(s),j:j+len(c)]=p.cpu().numpy()
        torch.cuda.synchronize();durations[tag]=time.time()-start
        assert np.isfinite(pred).all()
        predictions[tag]=pred
        print(f'{tag}: inference {durations[tag]:.1f}s, {pred.shape}',flush=True)
        del model,state
        torch.cuda.empty_cache()
    dense=data.dense[c_all].cpu().numpy()
    pos=data.cpg_pos_bp[c_all].cpu().numpy()
    frame=pd.DataFrame({'cpg_idx':data.cpg_idx[c_all],'position':pos,
                        'block':pos//5_000_000,
                        'view':['seen_cpg_new_patient']*len(seen)+['double_ood']*len(novel)})
    for name,start,end in [('context',0,4),('region',4,8),('ccre',8,17)]:
        assert np.allclose(dense[:,start:end].sum(1),1)
        labels=[f['name'].split('__',1)[1] for f in schema['features'][start:end]]
        frame[name]=np.asarray(labels)[dense[:,start:end].argmax(1)]
    counts=(data.indptr[1:]-data.indptr[:-1]).cpu().numpy()[c_all]
    frame['track_count']=counts
    frame['track_bin']=pd.cut(counts,[-1,0,10,50,np.inf],labels=['0','1-10','11-50','>50']).astype(str)
    norm=schema['features'][17]['normalization']
    tss=10**(dense[:,17]*norm['std']+norm['mean'])-1
    frame['tss_bin']=pd.cut(tss,[-1,2000,10000,100000,np.inf],labels=['<=2kb','2-10kb','10-100kb','>100kb']).astype(str)
    tp=np.sort(data.cpg_pos_bp[data.train_c].cpu().numpy())
    j=np.searchsorted(tp,pos)
    distances=np.minimum(abs(pos-tp[np.clip(j,0,len(tp)-1)]),abs(pos-tp[np.clip(j-1,0,len(tp)-1)]))
    frame['train_cpg_distance_bp']=distances
    frame['distance_bin']=pd.cut(distances,[-1,1000,100000,1000000,np.inf],labels=['<=1kb','1-100kb','0.1-1Mb','>1Mb']).astype(str)
    tissues=data.tissue[samples.cpu().numpy()].astype(str)
    eligible=[t for t in np.unique(tissues) if t!='NA' and (tissues==t).sum()>=20]
    known=np.isin(tissues,eligible)
    tissue_tables={}
    for tag,pred in predictions.items():
        m=metrics(y,pred)
        for k,v in m.items():frame[f'{tag}_{k}']=v
        yr=np.full_like(y,np.nan);pr=np.full_like(pred,np.nan)
        for t in eligible:
            take=tissues==t;yy=y[take];pp=pred[take]
            tm=metrics(yy,pp)
            for view,sel in [('seen_cpg_new_patient',slice(0,len(seen))),('double_ood',slice(len(seen),None))]:
                key=f'{view}/{t}';tissue_tables.setdefault(key,{'patients':int(take.sum()),'tissue':t,'view':view})
                ok=tm['n'][sel]>=10
                tissue_tables[key][tag]={'median_corr':float(np.nanmedian(tm['corr'][sel][ok])),
                                         'mse':float(np.average(tm['mse'][sel],weights=tm['n'][sel])),
                                         'eligible_cpgs':int(ok.sum())}
            yr[take]=yy-tm['mean'];pr[take]=pp-tm['pred_mean']
        wm=metrics(yr[known],pr[known]);frame[f'{tag}_within_corr']=wm['corr']
        frame[f'{tag}_within_std_ratio']=wm['std_ratio']
    frame['target_sd_bin']=pd.cut(frame.baseline_std,[-1,.05,.15,np.inf],labels=['<0.05','0.05-0.15','>0.15']).astype(str)
    frame['target_mean_bin']=pd.cut(frame.baseline_mean,[-1,.2,.8,np.inf],labels=['<0.2','0.2-0.8','>0.8']).astype(str)
    frame['delta_mse']=frame.candidate_mse-frame.baseline_mse
    frame['delta_corr']=frame.candidate_corr-frame.baseline_corr
    summary={'protocol':{'date':'2026-10-17','seed':17,'steps':2000,'weights':'EMA0.99',
              'sample_count':len(samples),'seen_cpgs':len(seen),'novel_cpgs':len(novel),
              'within':'pooled group-centered correlation excluding NA, groups>=20; different from mean of tissue medians',
              'target_strata':'posthoc using validation labels, never usable as deployment features',
              'inference_seconds':durations},'overall':{},'strata':{},'tissues':tissue_tables}
    for view,f in frame.groupby('view'):
        summary['overall'][view]=summarize(f)
        for col in ['context','region','ccre','track_bin','tss_bin','distance_bin','target_sd_bin','target_mean_bin','block']:
            for value,g in f.groupby(col):
                if len(g)>=20:summary['strata'][f'{view}/{col}/{value}']=summarize(g)
        sse=f.baseline_mse*f.baseline_n
        summary['overall'][view]['top10pct_cpg_fraction_sse']=float(sse.nlargest(max(1,len(f)//10)).sum()/sse.sum())
        if args.cheap is not None:
            gain=(f.cheap_mse-f.baseline_mse)*f.baseline_n
            order=np.argsort(-gain.to_numpy())
            oracle=[]
            for budget in [0,.1,.25,.5,1.]:
                chosen=order[:int(budget*len(f))]
                chosen=chosen[gain.iloc[chosen].to_numpy()>0]
                oracle.append({'max_expensive_fraction':budget,'used_expensive_fraction':len(chosen)/len(f),
                    'mse':float(((f.cheap_mse*f.baseline_n).sum()-gain.iloc[chosen].sum())/f.baseline_n.sum()),
                    'relative_pair_ffn_macs':.4+.6*len(chosen)/len(f)})
            summary['overall'][view]['oracle_allocation']={'warning':'Uses validation targets to select CpGs; NOT deployable performance. MAC ratio counts only pair FFNs and assumes routing before model execution.',
                'cheap_mse':float(np.average(f.cheap_mse,weights=f.baseline_n)),
                'cheap_median_corr':float(np.nanmedian(f.cheap_corr)),
                'curve':oracle}
    summary['provenance']={tag:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for tag,path in checkpoints}
    frame.to_csv(args.out/'per_cpg.csv.gz',index=False)
    (args.out/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
    print(json.dumps(clean_json(summary['overall']),indent=2),flush=True)


if __name__=='__main__':
    main()
