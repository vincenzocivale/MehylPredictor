"""Validate RNA signal, few-shot calibration and adaptive genomic regularization.

Frozen models and internal axes only. New-locus support labels are explicitly
an additional few-shot setting, never inputs to the zero-shot comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from verify_alphagenome_capacity import prepare, folds, normalize, dump
from search_alphagenome_hierarchical import load, fit_known_offsets
from search_alphagenome_architecture import factors

OUT = ROOT / 'artifacts/alphagenome_hypotheses_2026-10-18'
SEEDS = [17, 18, 19]
WEIGHTS = [0., .25, .5]
SUPPORT = [1, 5, 10, 50]
TAUS = [0., 5., 20.]
UNKNOWN = ['NA', 'nan', 'None', 'unknown', '']


def event(**value):
    print(json.dumps(value), flush=True)


def t(x, dtype=None):
    return torch.as_tensor(x, device='cuda', dtype=dtype)


@torch.inference_mode()
def evaluate(y, p, tissues, detailed=True):
    """FP32 predictions, FP64 per-locus moment reductions; same finite pairs."""
    mask = torch.isfinite(y) & torch.isfinite(p)
    n = mask.sum(0).double()
    den = n.clamp_min(1)
    yy = torch.where(mask, y, 0).double()
    pp = torch.where(mask, p, 0).double()
    ym, pm = yy.sum(0)/den, pp.sum(0)/den
    yc, pc = torch.where(mask, yy-ym, 0), torch.where(mask, pp-pm, 0)
    vy, vp, cov = yc.square().sum(0)/den, pc.square().sum(0)/den, (yc*pc).sum(0)/den
    corr = cov / (vy*vp).clamp_min(1e-30).sqrt()
    corr[(n < 3) | (vy < 1e-12) | (vp < 1e-12)] = torch.nan
    mse = ((yy-pp).square().sum(0)/den)
    mae = (yy-pp).abs().sum(0)/den
    severe = (((yy-pp).abs() > .2) & mask).sum(0)/den
    within, wcov, wvy, wvp = [], torch.zeros_like(n), torch.zeros_like(n), torch.zeros_like(n)
    for name in np.unique(tissues):
        ix = np.flatnonzero(tissues == name)
        if name in UNKNOWN or len(ix) < 20:
            continue
        m = mask[ix]; nn = m.sum(0).clamp_min(1)
        a, b = yy[ix], pp[ix]
        a = torch.where(m, a-a.sum(0)/nn, 0)
        b = torch.where(m, b-b.sum(0)/nn, 0)
        va, vb, cv = a.square().sum(0), b.square().sum(0), (a*b).sum(0)
        c = cv/(va*vb).clamp_min(1e-30).sqrt()
        c[(va < 1e-10) | (vb < 1e-10) | (m.sum(0) < 3)] = torch.nan
        within.append(float(c.nanmedian()))
        wcov += cv; wvy += va; wvp += vb
    wc = wcov/(wvy*wvp).clamp_min(1e-30).sqrt()
    wc[(wvy < 1e-10) | (wvp < 1e-10)] = torch.nan
    def avg(x):
        return float((x*n).sum()/n.sum())
    result = dict(mse=avg(mse), mae=avg(mae), mas_pcc=float(corr.nanmedian()),
                  within_tissue=float(np.nanmean(within)) if within else float('nan'),
                  pooled_within_pcc=float(wc.nanmedian()), severe=avg(severe), pairs=int(n.sum()),
                  bias_fraction=avg((pm-ym).square())/max(avg(mse), 1e-30))
    per = dict(n=n, mse=mse, mae=mae, corr=corr, bias=pm-ym, bias2=(pm-ym).square(),
               mean=ym, std=vy.sqrt(), pred_mean=pm, pred_std=vp.sqrt(),
               centered_mse=vy+vp-2*cov, within_corr=wc, severe=severe)
    return result, {k:v.cpu().numpy() for k,v in per.items()} if detailed else None


@torch.inference_mode()
def solve_bias(b, response, y, tau):
    """Support-only beta mean calibration; no support stays at the base bias."""
    mask = torch.isfinite(y)
    n = mask.sum(0)
    observed = torch.where(mask, y, 0).sum(0)/n.clamp_min(1)
    initial = torch.where(mask, (b[None]+response).sigmoid(), 0).sum(0)/n.clamp_min(1)
    w = n/(n+tau).clamp_min(1)
    desired = ((1-w)*initial+w*observed).clamp(.001, .999)
    bound = response.abs().amax(0)+b.abs()+20
    lo, hi = -bound, bound
    for _ in range(24):
        mid = (lo+hi)/2
        mean = torch.where(mask, (mid[None]+response).sigmoid(), 0).sum(0)/n.clamp_min(1)
        lower = mean < desired
        lo, hi = torch.where(lower, mid, lo), torch.where(lower, hi, mid)
    return torch.where(n > 0, (lo+hi)/2, b)


def permutations(tissues, seed):
    rng = np.random.default_rng(seed)
    result = np.arange(len(tissues))
    for name in np.unique(tissues):
        ii = np.flatnonzero(tissues == name)
        result[ii] = rng.permutation(ii)
    return result


def fit_tissue_prior(d, ts, tc):
    """A genomic+tissue baseline fitted only to training patients/loci."""
    tissue = d.tissue[ts]
    yy = d.block(t(ts), t(tc))
    means = [yy.nanmean(0).cpu().numpy()]
    names = ['__global__']
    for name in np.unique(tissue):
        ix = np.flatnonzero(tissue == name)
        if name in UNKNOWN or len(ix) < 20:
            continue
        means.append(yy[t(ix)].nanmean(0).cpu().numpy()); names.append(name)
    means = np.asarray(means)
    means = np.where(np.isfinite(means), means, means[0])
    scaler = StandardScaler().fit(d.raw_ag[tc])
    model = Ridge(alpha=100.).fit(scaler.transform(d.raw_ag[tc]), means.T)
    return names, model.predict(scaler.transform(d.raw_ag)).T.clip(.005, .995).astype(np.float32)


@torch.inference_mode()
def frozen_split(d, spec, out):
    name, ts, vs, tc, vc = spec
    dest = out/name; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        return
    normalize(d, ts, tc)
    known = np.sort(np.random.default_rng(20261018).choice(tc, min(4096, len(tc)), replace=False))
    names, tissue_prior = fit_tissue_prior(d, ts, tc)
    lookup = {v:i for i,v in enumerate(names)}
    row_tissue = np.asarray([lookup.get(x, 0) for x in d.tissue[vs]])
    records, shots, all_features, all_per = [], [], [], {}
    for view, cc in [('novel', vc), ('known', known)]:
        y = d.block(t(vs), t(cc))
        train_y = d.block(t(ts), t(cc))
        ensemble_means, ensemble_variances = [], []
        feature_extra = []
        for seed in SEEDS:
            base_per = {}
            for source in ['v1_early', 'v2_early']:
                m = load(name, source, seed)
                b, a = factors(m, d, cc)
                u = m.encode_rna(d.rna[t(vs)]).float()
                train_u = m.encode_rna(d.rna[t(ts)]).float()
                response = u@a.T/m.rank**.5
                def record(label, pred, **extra):
                    metric, per = evaluate(y, pred, d.tissue[vs])
                    records.append(dict(split=name, view=view, source=source, seed=seed, arm=label, **metric, **extra))
                    np.savez_compressed(dest/f'{view}_{source}_{seed}_{label}.npz', cpg_idx=d.cpg_idx[cc], **per)
                    return metric, per
                raw_metric, raw_per = record('raw', (b[None]+response).sigmoid())
                if view == 'known':
                    offsets = fit_known_offsets(b, a, train_u, train_y, [1.])[1.]
                    bias = b+offsets
                    record('calibrated', (bias[None]+response).sigmoid())
                else:
                    for w in WEIGHTS:
                        bias_w = m.calibrated_intercept(b, a, t(d.raw_ag[cc]), blend=w)
                        if w == 0:
                            metric, per = raw_metric, raw_per
                        else:
                            metric, per = record(f'prior{w:g}', (bias_w[None]+response).sigmoid())
                        if source == 'v2_early':
                            base_per[str(w)] = per
                    bias = m.calibrated_intercept(b, a, t(d.raw_ag[cc]), blend=.25)
                if source == 'v1_early':
                    del m
                    continue
                prediction = (bias[None]+response).sigmoid()
                train_prediction = (bias[None]+train_u@a.T/m.rank**.5).sigmoid()
                record('population_only', train_prediction.mean(0)[None].expand_as(y))
                tissue_pred = train_prediction.mean(0)[None].expand_as(y).clone()
                for tissue in np.unique(d.tissue[vs]):
                    ii = np.flatnonzero(d.tissue[vs] == tissue)
                    jj = np.flatnonzero(d.tissue[ts] == tissue)
                    if tissue not in UNKNOWN and len(jj) >= 20:
                        tissue_pred[t(ii)] = train_prediction[t(jj)].mean(0)
                record('model_tissue_only', tissue_pred)
                for rep in range(3):
                    ix = permutations(d.tissue[vs], 20261018+rep)
                    record(f'rna_shuffle{rep}', (bias[None]+response[t(ix)]).sigmoid())
                if view == 'novel':
                    ensemble_means.append(prediction.mean(0).cpu().numpy())
                    ensemble_variances.append(prediction.var(0, unbiased=False).cpu().numpy())
                    feature_extra.append(np.c_[prediction.mean(0).cpu().numpy(),
                                               prediction.std(0, unbiased=False).cpu().numpy(),
                                               m.mean_prior(t(d.raw_ag[cc])).cpu().numpy(),
                                               torch.linalg.vector_norm(a, dim=1).cpu().numpy()])
                    for w, per in base_per.items():
                        all_per.setdefault(w, []).append(per)
                    # A prediction-only disagreement statistic, no query labels.
                    if seed == SEEDS[0]:
                        saved_prediction = prediction.clone()
                        pred_sum = prediction.clone(); pred_square = prediction.square()
                    else:
                        pred_sum += prediction; pred_square += prediction.square()
                    for rep in range(5):
                        support = np.random.default_rng(9127+rep).permutation(ts)[:max(SUPPORT)]
                        support_y = d.block(t(support), t(cc))
                        support_response = train_u[t(np.searchsorted(ts, support))]@a.T/m.rank**.5
                        assert not np.intersect1d(support, vs).size
                        for k in SUPPORT:
                            for tau in TAUS:
                                new_b = solve_bias(bias, support_response[:k], support_y[:k], tau)
                                metric, _ = evaluate(y, (new_b[None]+response).sigmoid(), d.tissue[vs], detailed=False)
                                shots.append(dict(split=name, seed=seed, repetition=rep, support=k, tau=tau, **metric))
                    metric, _ = evaluate(y, prediction, d.tissue[vs], detailed=False)
                    shots.append(dict(split=name, seed=seed, repetition=-1, support=0, tau=-1., **metric))
                del m, train_prediction
                torch.cuda.empty_cache()
            event(stage='frozen', split=name, view=view, seed=seed)
        if view == 'novel':
            disagreement = (pred_square/len(SEEDS)-(pred_sum/len(SEEDS)).square()).clamp_min(0).mean(0).sqrt().cpu().numpy()
            extra = np.stack(feature_extra).mean(0)
            features = np.c_[d.raw_dense[cc], extra, disagreement,
                             np.std(ensemble_means, axis=0), np.mean(ensemble_variances, axis=0),
                             np.linalg.norm(d.raw_ag[cc], axis=1)]
            labels = {w:np.mean([p['mse'] for p in per], axis=0) for w,per in all_per.items()}
            np.savez_compressed(dest/'adaptive_data.npz', x=features, cpg_idx=d.cpg_idx[cc],
                                position=d.cpg_pos_bp[t(cc)].cpu().numpy(), weights=WEIGHTS,
                                loss=np.stack([labels[str(w)] for w in WEIGHTS], axis=1),
                                n=np.mean([p['n'] for p in all_per['0.0']], axis=0))
            # Predetermined contrasts: two largest named tissue groups in TRAIN.
            counts = pd.Series(d.tissue[ts][~np.isin(d.tissue[ts], UNKNOWN)]).value_counts()
            contrasts = []
            for g1, g2 in [tuple(counts.index[:2])]:
                i, j = np.flatnonzero(d.tissue[vs] == g1), np.flatnonzero(d.tissue[vs] == g2)
                if min(len(i), len(j)) < 20:
                    continue
                truth = (y[t(i)].nanmean(0)-y[t(j)].nanmean(0)).cpu().numpy()
                pred = (saved_prediction[t(i)].mean(0)-saved_prediction[t(j)].mean(0)).cpu().numpy()
                only = tissue_prior[lookup[g1], cc]-tissue_prior[lookup[g2], cc]
                for label, val in [('v3_seed17', pred), ('genomic_tissue', only)]:
                    mask = np.isfinite(truth); large = mask & (np.abs(truth) >= .1)
                    contrasts.append(dict(groups=[g1,g2], arm=label, samples=[len(i),len(j)],
                                          effect_mse=float(np.mean((truth[mask]-val[mask])**2)),
                                          effect_corr=float(np.corrcoef(truth[mask],val[mask])[0,1]),
                                          large_effect_cpgs=int(large.sum()),
                                          direction_accuracy=float(np.mean(np.sign(truth[large]) == np.sign(val[large])))))
            dump(dest/'contrasts.json', contrasts)
        # Controls have no query target-derived parameters.
        baseline_preds = {'genomic_mean': t(tissue_prior[0, cc])[None].expand_as(y),
                          'genomic_tissue': t(tissue_prior[row_tissue][:, cc])}
        if view == 'known':
            empirical = train_y.nanmean(0)
            baseline_preds['empirical_mean'] = empirical[None].expand_as(y)
            empirical_tissue = empirical[None].expand_as(y).clone()
            for tissue in np.unique(d.tissue[vs]):
                ii, jj = np.flatnonzero(d.tissue[vs] == tissue), np.flatnonzero(d.tissue[ts] == tissue)
                if tissue not in UNKNOWN and len(jj) >= 20:
                    empirical_tissue[t(ii)] = train_y[t(jj)].nanmean(0)
            baseline_preds['empirical_tissue'] = empirical_tissue
        for label, pred in baseline_preds.items():
            metric, per = evaluate(y, pred, d.tissue[vs])
            records.append(dict(split=name, view=view, source='control', seed=-1, arm=label, **metric))
            np.savez_compressed(dest/f'{view}_control_{label}.npz', cpg_idx=d.cpg_idx[cc], **per)
    pd.DataFrame(records).to_csv(dest/'rna_signal.csv', index=False)
    pd.DataFrame(shots).to_csv(dest/'few_shot.csv', index=False)
    np.savez_compressed(dest/'axes.npz', train_patients=d.sample_idx[ts], test_patients=d.sample_idx[vs],
                        train_cpgs=d.cpg_idx[tc], test_cpgs=d.cpg_idx[vc], known_cpgs=d.cpg_idx[known])
    dump(dest/'complete.json', dict(complete=True, official_targets_used=False))


def adaptive(out):
    """Other genomic folds provide labels; dev patients are independent."""
    splits = ['fold0', 'fold1', 'fold2', 'dev']
    data = {s:dict(np.load(out/s/'adaptive_data.npz')) for s in splits}
    records, curves = [], []
    for held in splits:
        train = [s for s in splits[:3] if s != held]
        fit = {k:np.concatenate([data[s][k] for s in train]) for k in ['x','loss','cpg_idx']}
        test = data[held]
        assert not np.intersect1d(fit['cpg_idx'], test['cpg_idx']).size
        rf = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=100,
                                   max_features=.8, random_state=20261018, n_jobs=4)
        # Predict differences, avoiding a common CpG difficulty component.
        rf.fit(fit['x'], fit['loss']-fit['loss'][:,[0]])
        estimated = rf.predict(test['x'])
        choice = estimated.argmin(1)
        selected_loss = test['loss'][np.arange(len(choice)), choice]
        risk = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=100,
                                     max_features=.8, random_state=20261018, n_jobs=4)
        risk.fit(fit['x'], fit['loss'][:,1])
        risk_prediction = risk.predict(test['x'])
        truth = test['loss'][:,1]
        row = dict(split=held, adaptive_mse=float(np.average(selected_loss, weights=test['n'])),
                   fixed025_mse=float(np.average(truth, weights=test['n'])),
                   weight_counts={str(w):int((choice == i).sum()) for i,w in enumerate(WEIGHTS)},
                   error_prediction_spearman=float(pd.Series(risk_prediction).corr(pd.Series(truth), method='spearman')),
                   hard_decile_auc=float(roc_auc_score(truth >= np.quantile(truth,.9), risk_prediction)),
                   fit_loci=len(fit['x']), test_loci=len(truth))
        for label, estimate in [('learned',risk_prediction), ('seed_disagreement',test['x'][:,27]),
                                ('oracle_diagnostic',truth)]:
            order = np.argsort(estimate, kind='stable')
            for coverage in [.25,.5,.75,1.]:
                ix = order[:max(1,round(len(order)*coverage))]
                curves.append(dict(split=held, estimator=label, coverage=coverage,
                                   mse=float(np.average(truth[ix],weights=test['n'][ix]))))
        frame = pd.DataFrame(dict(cpg_idx=test['cpg_idx'], risk=risk_prediction, true_mse=truth,
                                  selected_weight=np.asarray(WEIGHTS)[choice], adaptive_mse=selected_loss))
        frame.to_csv(out/held/'adaptive_predictions.csv.gz', index=False)
        # Full metrics are recomposed from held-locus per-seed results, never
        # from target-derived oracle choices.
        for seed in SEEDS:
            per = [np.load(out/held/f'novel_v2_early_{seed}_{"raw" if w == 0 else f"prior{w:g}"}.npz') for w in WEIGHTS]
            n = per[0]['n']; ix=np.arange(len(n))
            row[f'seed{seed}'] = {}
            for key in ['mse','mae','severe']:
                values=np.stack([p[key] for p in per],axis=1)[ix,choice]
                row[f'seed{seed}'][key]=float(np.average(values,weights=n))
            for key in ['corr','within_corr']:
                values=np.stack([p[key] for p in per],axis=1)[ix,choice]
                row[f'seed{seed}'][key]=float(np.nanmedian(values))
        records.append(row)
        event(stage='adaptive', **row)
    dump(out/'adaptive.json', records)
    pd.DataFrame(curves).to_csv(out/'risk_coverage.csv',index=False)
    shots=pd.concat([pd.read_csv(out/s/'few_shot.csv') for s in splits],ignore_index=True)
    internal=shots[(shots.split!='dev') & (shots.support>0)]
    recipe=internal.groupby('tau').mse.mean().idxmin()
    dump(out/'few_shot_recipe.json',dict(tau=float(recipe),selection='mean MSE over 3 internal folds, all k/repetitions/seeds',
                                       setting='uses observed methylation on new loci in support patients only',
                                       dev_untuned=True))
    shots.to_csv(out/'few_shot.csv',index=False)
    pd.concat([pd.read_csv(out/s/'rna_signal.csv') for s in splits]).to_csv(out/'rna_signal.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--stage',choices=['frozen','adaptive','all'],default='all')
    ap.add_argument('--splits',nargs='+',default=['fold0','fold1','fold2','dev'])
    args=ap.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    protocol=dict(seeds=SEEDS,weights=WEIGHTS,support=SUPPORT,taus=TAUS,support_repeats=5,
                  shuffles=3,known_cpgs=4096,selection='internal genomic folds; dev confirmation, no official target use',
                  few_shot='support patients selected from model training patients, novel loci; evaluation patients disjoint',
                  adaptive='leave-one-genomic-fold-out error learning; inner evaluation patients shared across folds; dev patients independent',
                  tissue_baseline='Ridge alpha100 per tissue, training patients and training CpGs only; unknown tissue uses global',
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text()) == protocol, 'Changed protocol: use a fresh output directory'
    else:
        dump(out/'protocol.json',protocol)
    torch.set_num_threads(4); threadpool_limits(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    if args.stage in ['frozen','all']:
        d=prepare(torch.device('cuda'),out)
        for spec in folds(d):
            if spec[0] in args.splits:
                frozen_split(d,spec,out)
    if args.stage in ['adaptive','all']:
        adaptive(out)


if __name__ == '__main__':
    main()
