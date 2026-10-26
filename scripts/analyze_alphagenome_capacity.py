"""Analyze the internal-only capacity pilot without fitting to dev outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_cpg_errors import clean_json


def dump(path,value):
    path.write_text(json.dumps(clean_json(value),indent=2)+'\n')


def block_ci(delta,n,blocks):
    bs=np.unique(blocks); rng=np.random.default_rng(20261018)
    numer=np.array([np.sum(delta[blocks==b]*n[blocks==b]) for b in bs])
    denom=np.array([np.sum(n[blocks==b]) for b in bs])
    draws=rng.integers(0,len(bs),(2000,len(bs)))
    vals=numer[draws].sum(1)/denom[draws].sum(1)
    return {'difference':float(np.sum(delta*n)/n.sum()),
            'ci95_blocks':np.quantile(vals,[.025,.975]).tolist(),'blocks':len(bs)}


def load_per(out,split,arm,seed,step):
    with np.load(out/split/arm/f'seed{seed}'/f'step{step}_per_cpg.npz') as z:
        return {k:z[k] for k in z.files}


def weighted_mse(p):
    return float(np.sum(p['n']*p['mse'])/p['n'].sum())


def aggregate(out,protocol):
    seeds=protocol['seeds']; records=[]
    for split in protocol['splits']:
        for arm in protocol['arms']:
            for step in protocol['steps'] if arm.startswith('d') else [min(protocol['steps'])]:
                docs=[json.loads((out/split/arm/f'seed{s}'/f'step{step}.json').read_text()) for s in seeds]
                rec={'split':split,'arm':arm,'step':step,'seeds':seeds,
                     'parameters':docs[0]['parameters']}
                for key in ['metrics','training_diagnostic','known_locus_new_patient']:
                    rec[key]={m:float(np.mean([d[key][m] for d in docs])) for m in docs[0][key]}
                rec['train_seconds']=float(np.mean([d['train_seconds'] for d in docs]))
                rec['per_seed']=[d['metrics'] for d in docs]
                records.append(rec)
    inner=[r for r in records if r['split'].startswith('fold') and r['arm'].startswith('d')]
    candidates=[]
    for arm in ['d0','d2','d8']:
        for step in protocol['steps']:
            rr=[r for r in inner if r['arm']==arm and r['step']==step]
            candidates.append({'arm':arm,'step':step,'mse':float(np.mean([r['metrics']['mse'] for r in rr]))})
    selected=min(candidates,key=lambda r:(r['mse'],r['step'],int(r['arm'][1:])))
    return {'records':records,'uniform_inner_candidates':candidates,'uniform_selected':selected}


def capacity_stability(out,protocol,features):
    seeds=protocol['seeds']; result=[]
    for split in protocol['splits']:
        axes=np.load(out/split/'evaluation_axes.npz')
        blocks=features['position'][axes['cpg_rows']]//5000000
        for cheap in ['d0','d2']:
            for step in protocol['steps']:
                small=[load_per(out,split,cheap,s,step) for s in seeds]
                large=[load_per(out,split,'d8',s,step) for s in seeds]
                delta=np.stack([a['mse']-b['mse'] for a,b in zip(small,large)])
                pairs=[]
                for i in range(len(seeds)):
                    for j in range(i):
                        pairs.append({'seed_a':seeds[i],'seed_b':seeds[j],
                                      'spearman':float(spearmanr(delta[i],delta[j]).statistic),
                                      'same_sign':float(np.mean((delta[i]>0)==(delta[j]>0)))})
                # Routing selected on other seeds' validation errors is a
                # diagnostic upper bound, never an available inference rule.
                cross=[]
                for k in range(len(seeds)):
                    choose=np.delete(delta,k,0).mean(0)>0
                    p=np.where(choose,large[k]['mse'],small[k]['mse'])
                    cross.append(float(np.average(p,weights=small[k]['n'])))
                result.append({'split':split,'cheap':cheap,'step':step,
                               'fraction_large_better_per_seed':(delta>0).mean(1).tolist(),
                               'seed_pairs':pairs,'cross_seed_oracle_mse':cross,
                               'large_minus_small':block_ci(-delta.mean(0),small[0]['n'],blocks)})
    return result


def gate_inputs(features):
    # No target statistics or locus IDs/positions enter these features.
    ag=features['ag'].astype(np.float32); dense=features['dense'].astype(np.float32)
    return {'annotation':dense,'alphagenome':ag,'hybrid':np.c_[dense,ag]}


def routing(out,protocol,features):
    seeds=protocol['seeds']; step=min(protocol['steps'])
    oo_axes=[np.load(out/f'fold{k}'/'evaluation_axes.npz') for k in range(3)]
    train=np.concatenate([a['cpg_rows'] for a in oo_axes])
    dev=np.load(out/'dev/evaluation_axes.npz'); test=dev['cpg_rows']
    assert len(np.unique(train))==len(train) and not np.intersect1d(train,test).size
    inputs=gate_inputs(features); results=[]
    for cheap in ['d0','d2']:
        target=np.concatenate([
            np.mean([load_per(out,f'fold{k}',cheap,s,step)['mse']-
                     load_per(out,f'fold{k}','d8',s,step)['mse'] for s in seeds],axis=0)
            for k in range(3)])
        test_small=[load_per(out,'dev',cheap,s,step) for s in seeds]
        test_large=[load_per(out,'dev','d8',s,step) for s in seeds]
        test_delta=np.mean([a['mse']-b['mse'] for a,b in zip(test_small,test_large)],axis=0)
        for feature_name,x in inputs.items():
            for learner in (['ridge','trees'] if feature_name=='hybrid' else ['ridge']):
                if learner=='ridge':
                    model=make_pipeline(StandardScaler(),Ridge(alpha=100.))
                else:
                    model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,
                                                       min_samples_leaf=200,l2_regularization=10.,
                                                       learning_rate=.05,random_state=20261018)
                model.fit(x[train],target)
                benefit=model.predict(x[test])
                diagnostics={'spearman_benefit':float(spearmanr(benefit,test_delta).statistic),
                             'r2_benefit':float(r2_score(test_delta,benefit)),
                             'auroc_positive_benefit':float(roc_auc_score(test_delta>0,benefit))}
                choices=[]
                for budget in [.1,.25,.5,1.]:
                    # Exact budget cap; only feature-predicted positive benefit.
                    order=np.argsort(-benefit,kind='stable')
                    use=np.zeros(len(test),bool)
                    selected=order[:round(budget*len(test))]
                    use[selected[benefit[selected]>0]]=True
                    per_seed=[]; diffs=[]
                    for a,b in zip(test_small,test_large):
                        mse=np.where(use,b['mse'],a['mse'])
                        corr=np.where(use,b['corr'],a['corr'])
                        per_seed.append({'mse':float(np.average(mse,weights=a['n'])),
                                         'mas_pcc':float(np.nanmedian(corr))})
                        diffs.append(mse-a['mse'])
                    choices.append({'budget_cap':budget,'actual_fraction_large':float(use.mean()),
                                    'metrics':{k:float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]},
                                    'minus_cheap':block_ci(np.mean(diffs,0),test_small[0]['n'],features['position'][test]//5000000)})
                results.append({'cheap':cheap,'step':step,'features':feature_name,
                                'learner':learner,'diagnostics':diagnostics,'choices':choices})
    return {'fit_cpgs':len(train),'eval_cpgs':len(test),'results':results,
            'note':'Fixed learners, penalties and budgets. No selection using dev targets. Three independent base fits produce training labels; no official holdout.'}


def comparative_ci(out,protocol,features,summary):
    seeds=protocol['seeds']; selected=summary['uniform_selected']
    axes=np.load(out/'dev/evaluation_axes.npz')
    blocks=features['position'][axes['cpg_rows']]//5000000
    base=[load_per(out,'dev',selected['arm'],s,selected['step']) for s in seeds]
    result=[]
    for arm in protocol['arms']:
        step=min(protocol['steps'])
        candidate=[load_per(out,'dev',arm,s,step) for s in seeds]
        delta=np.mean([c['mse']-b['mse'] for c,b in zip(candidate,base)],axis=0)
        result.append({'arm':arm,'step':step,**block_ci(delta,base[0]['n'],blocks),
                       'fraction_cpg_improved':float(np.mean(delta<0))})
    return result


def read_targets(features):
    # Only the internal development axes, both contained in official training.
    lab=ROOT.parent/'chr1_alphagenome_v1_2026-10-17'
    sys.path.insert(0,str(lab))
    from pilot import PREP, positions
    from verify_alphagenome_capacity import selected_matrix
    sample=np.r_[features['train_s'],features['val_s']]
    with h5py.File(PREP/'methylation/array_table5_chr1.h5','r') as h:
        sr=positions(h['sample_idx'][:],sample)
        cr=positions(h['cpg_idx'][:],features['cpg_idx'])
        y=selected_matrix(h['beta'],sr,cr)
    return y


def structure(out,protocol,features):
    """Frozen-representation probes; dev outcomes only report, never tune."""
    from sklearn.metrics import roc_auc_score
    dev_axes=np.load(out/'dev/evaluation_axes.npz')
    norm=np.load(out/'dev/normalization.npz')
    lookup={int(c):i for i,c in enumerate(features['cpg_idx'])}
    tr=np.array([lookup[int(c)] for c in norm['train_cpg_idx']])
    te=dev_axes['cpg_rows']
    y=read_targets(features)
    nt=len(features['train_s'])
    ytr=y[:nt]; yte=y[nt:]
    target_mean=np.nanmean(ytr,0); test_mean=np.nanmean(yte,0)
    target_sd=np.nanstd(ytr,0); test_sd=np.nanstd(yte,0)
    targets={'mean':(target_mean,test_mean),'sd':(target_sd,test_sd)}
    labels={'context':features['dense'][:,:4].argmax(1),
            'region':features['dense'][:,4:8].argmax(1)}
    result=[]
    arrays={'raw_alphagenome':features['ag']}
    for seed in protocol['seeds']:
        z=np.load(out/'dev'/'d8'/f'seed{seed}'/'step1000_representations.npz')
        for name in ['shallow','deep','factors']:
            arrays[f'{name}_seed{seed}']=z[name]
    # All scalers are fit only on permitted training loci. Ridge has fixed
    # alpha=100, no feature- or target-dependent parameter selection.
    for name,x in arrays.items():
        rec={'representation':name,'targets':{}}
        fit_y=np.column_stack([target_mean,target_sd]+[np.eye(4)[labels[k]] for k in ['context','region']])
        model=make_pipeline(StandardScaler(),Ridge(alpha=100.))
        model.fit(x[tr],fit_y[tr]); predicted=model.predict(x[te])
        for j,(target,(a,b)) in enumerate(targets.items()):
            p=predicted[:,j]
            rec['targets'][target]={'r2':float(r2_score(b[te],p)),
                                    'spearman':float(spearmanr(b[te],p).statistic),
                                    'mse':float(mean_squared_error(b[te],p))}
        # CpG context is explicitly present in hybrid inputs. Its decoding
        # downstream is characterization only; it is circular as validation.
        for j,(target,lbl) in enumerate(labels.items()):
            scores=predicted[:,2+4*j:6+4*j]
            rec['targets'][target]={'macro_auroc':float(np.mean([
                roc_auc_score(lbl[te]==k,scores[:,k]) for k in range(4)]))}
        result.append(rec)
        print(json.dumps({'probe':rec}),flush=True)
    neighbors=response_neighbors(out,features,arrays,tr,te,y)
    del y,ytr,yte
    return {'probes':result,'neighbors':neighbors,'fit_cpgs':len(tr),'eval_cpgs':len(te),
            'note':'Means/SD labels use training patients for probe fit and held-out patients for evaluation. Context/region already enter hybrid branches; their decoding is not independent validation.'}


def response_neighbors(out,features,representations,tr,te,y):
    """Can feature neighbors transfer RNA associations beyond coarse context?"""
    from verify_alphagenome_capacity import CANON, positions, rows
    samples=np.r_[features['train_s'],features['val_s']]
    nt=len(features['train_s'])
    with h5py.File(CANON/'rna/tcga_rna_official_full.h5','r') as h:
        rna=rows(h,'X',positions(h['sample_idx'][:],samples)).astype(np.float32)
    # A diagnostic RNA basis, selected and fit exclusively on training patients.
    sd=rna[:nt].std(0); genes=np.argsort(sd)[-1024:]
    scaler=StandardScaler().fit(rna[:nt,genes])
    rx=scaler.transform(rna[:,genes])
    pca=PCA(n_components=16,svd_solver='randomized',random_state=20261018)
    pca.fit(rx[:nt]); pcs=pca.transform(rx).astype(np.float32)
    del rx,rna
    source=np.load(ROOT.parent/'chr1_alphagenome_v1_2026-10-17/cache/array_chr1.npz',allow_pickle=True)
    tissue=source['tissue'][positions(source['sample_idx'],samples)]
    def descriptor(yy,xx,tt):
        cy=np.zeros_like(yy); cx=np.zeros_like(xx); n=np.zeros(yy.shape[1])
        for label in np.unique(tt):
            ix=np.flatnonzero(tt==label)
            if label in ('NA','nan','None','unknown','') or len(ix)<20: continue
            mask=np.isfinite(yy[ix]); count=mask.sum(0)
            mean=np.where(mask,yy[ix],0).sum(0)/np.maximum(count,1)
            cy[ix]=np.where(mask,yy[ix]-mean,0)
            cx[ix]=xx[ix]-xx[ix].mean(0)
            n+=count
        return (cx.T@cy/np.maximum(n,1)).T
    # Scaling of RNA PCs uses training data only, including after centering.
    pcs/=np.maximum(pcs[:nt].std(0),1e-6)
    train_response=descriptor(y[:nt],pcs[:nt],tissue[:nt])
    test_response=descriptor(y[nt:],pcs[nt:],tissue[nt:])
    context=features['dense'][:,:4].argmax(1); region=features['dense'][:,4:8].argmax(1)
    edges=np.quantile(features['dense'][tr,17],[.25,.5,.75])
    tss=np.searchsorted(edges,features['dense'][:,17])
    groups=context*16+region*4+tss
    pos=features['position']; results=[]
    for name in ['raw_alphagenome','shallow_seed17','deep_seed17','factors_seed17']:
        x=representations[name]
        scaled=StandardScaler().fit(x[tr])
        xt=scaled.transform(x[tr]); xv=scaled.transform(x[te])
        if xt.shape[1]>64:
            pca=PCA(n_components=64,svd_solver='randomized',random_state=20261018).fit(xt)
            xt=pca.transform(xt); xv=pca.transform(xv)
        xt=normalize(xt); xv=normalize(xv)
        estimate=np.full((len(te),16),np.nan); control=np.full_like(estimate,np.nan)
        rng=np.random.default_rng(20261018)
        for group in np.unique(groups[te]):
            a=np.flatnonzero(groups[tr]==group); b=np.flatnonzero(groups[te]==group)
            if len(a)<5: continue
            sim=xv[b]@xt[a].T
            eligible=np.abs(pos[te[b],None]-pos[tr[a]][None])>=1_000_000
            sim[~eligible]=-np.inf
            for j,index in enumerate(b):
                pool=np.flatnonzero(eligible[j])
                if len(pool)<5: continue
                near=np.argsort(-sim[j],kind='stable')[:5]
                estimate[index]=train_response[tr[a[near]]].mean(0)
                # Average 20 matched random sets, reducing control Monte Carlo noise.
                draws=np.array([rng.choice(pool,5,replace=False) for _ in range(20)])
                control[index]=train_response[tr[a[draws]]].mean((0,1))
        valid=np.isfinite(estimate).all(1)
        target=test_response[te][valid]
        em=((estimate[valid]-target)**2).mean(1); cm=((control[valid]-target)**2).mean(1)
        results.append({'representation':name,'cpgs':int(valid.sum()),
                        'neighbor_response_mse':float(em.mean()),'matched_random_response_mse':float(cm.mean()),
                        'relative_mse_change':float(em.mean()/cm.mean()-1),
                        'r2_response':float(r2_score(target,estimate[valid],multioutput='variance_weighted')),
                        'minus_matched_random':block_ci(em-cm,np.ones(len(em)),pos[te][valid]//5000000)})
    return {'results':results,'descriptor':'Within-tissue covariance with 16 RNA PCs of 1024 training-variable genes; association, not causality',
            'controls':'5 nearest vs mean of 20 sets of 5 random loci, matched context/region/TSS quartile; all neighbors >=1Mb away; train-patient associations transfer to held-out patients'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/alphagenome_capacity_2026-10-18')
    ap.add_argument('--skip-structure',action='store_true')
    args=ap.parse_args(); out=args.out
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=4)
    assert (out/'training_complete.json').exists()
    protocol=json.loads((out/'protocol.json').read_text())
    with np.load(out/'features.npz') as z: features={k:z[k] for k in z.files}
    summary=aggregate(out,protocol)
    summary['stability']=capacity_stability(out,protocol,features)
    summary['comparative_ci']=comparative_ci(out,protocol,features,summary)
    dump(out/'capacity_summary.json',summary)
    gates=routing(out,protocol,features); dump(out/'routing_summary.json',gates)
    if not args.skip_structure:
        probes=structure(out,protocol,features); dump(out/'structure_summary.json',probes)
    print(json.dumps({'complete':True,'uniform_selected':summary['uniform_selected']}),flush=True)


if __name__=='__main__': main()
