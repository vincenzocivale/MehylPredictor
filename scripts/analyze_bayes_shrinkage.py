"""Empirical-Bayes locus adaptation vs. a tuned pseudo-count (internal folds only).

Model: residual of the frozen v3 predictor at locus c, r_pc = y_pc - P_pc, has
population offset delta_c ~ N(0, s2p_c) and patient noise var s2n_c. With n
support patients the posterior-mean offset is  n/(n+tau_c) * rbar,
tau_c = s2n_c / s2p_c. Existing code uses a global tau picked from {0,5,20}.

Everything here is cross-fitted: tau for fold f is estimated from the other two
genomic folds (models that never saw those loci), using training patients only.
No dev, official or query label enters an estimate. Decision rules are frozen
in PROTOCOL and written to disk before the first evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_alphagenome_capacity import prepare, folds, normalize, dump
from search_alphagenome_hierarchical import load
from search_alphagenome_architecture import factors
from validate_alphagenome_hypotheses import t, event

OUT = ROOT / 'artifacts/alphagenome_bayes_shrinkage_2026-10-18'
SEEDS = [17, 18, 19]
SHOTS = [1, 2, 5, 10, 20, 50]
REPS = 5
TAUS = [0., .5, 1., 2., 3., 5., 8., 12., 20., 40.]
Z = {'80': 1.2816, '90': 1.6449}
SOURCE, BLEND, MIN_N = 'v2_early', .25, 100
HGB = dict(max_iter=200, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=50, random_state=0)

PROTOCOL = dict(
    question='Does a closed-form empirical-Bayes weight n/(n+tau_c) match or beat the tuned global pseudo-count?',
    data='3 internal genomic folds x seeds 17/18/19 x 5 nested support sets (rng 9127+rep), checkpoints v2_early (= v3 recipe, blend .25); dev/official untouched',
    estimation='tau_hat for fold f from folds != f, same seed, training patients only; global = n-weighted moments, local = HGB on label-free locus features',
    grid=TAUS, shots=SHOTS, hgb=HGB, tau_local_clip=[.05, 500], prior_var_floor=1e-5,
    rules=dict(
        P1='for every k, pooled MSE(theory_global) <= 1.005 * min over tau grid',
        P2='pooled MSE(theory_local) <= 0.98 * min over tau grid at k in {1,5,10}, and better than the best grid tau in >=8/9 fold-seed units at k=5',
        P3='bayes_local 90% coverage in [0.87, 0.93] for k>=5 and closer to nominal than bayes_global at every such k'),
    aggregation='mean over reps, then over the 9 fold-seed units')


def fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@torch.inference_mode()
def fold_tensors(d, name, vc, seed):
    m = load(name, SOURCE, seed)
    b, a = factors(m, d, vc)
    bp = m.calibrated_intercept(b, a, t(d.raw_ag[vc]), blend=BLEND)
    P = (bp[None] + m.encode_rna(d.rna).float() @ a.T / m.rank ** .5).sigmoid()
    extra = torch.stack([m.mean_prior(t(d.raw_ag[vc])), torch.linalg.vector_norm(a, dim=1)], 1)
    del m
    return P, d.block(t(np.arange(len(d.sample_idx))), t(vc)), extra


@torch.inference_mode()
def locus_stats(d, P, Y, extra, ts, vc):
    s = t(ts)
    r = (Y[s] - P[s]).double()
    mask = torch.isfinite(r)
    n = mask.sum(0).double()
    delta = torch.where(mask, r, 0).sum(0) / n.clamp_min(1)
    s2 = torch.where(mask, r - delta, 0).square().sum(0) / (n - 1).clamp_min(1)
    X = torch.cat([P[s].mean(0)[:, None], P[s].std(0)[:, None], extra], 1).cpu().numpy()
    X = np.c_[X, d.raw_dense[vc]]
    return dict(n=n.cpu().numpy(), delta=delta.cpu().numpy(), s2=s2.cpu().numpy(), X=X)


def pooled(parts):
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    keep = out['n'] >= MIN_N
    return {k: v[keep] for k, v in out.items()}


def global_estimates(parts):
    p = pooled(parts)
    w = p['n']
    noise = (w * p['s2']).sum() / w.sum()
    prior = (w * (p['delta'] ** 2 - p['s2'] / p['n'])).sum() / w.sum()
    return float(noise), float(max(prior, PROTOCOL['prior_var_floor']))


def local_estimates(parts, X):
    p = pooled(parts)
    noise = HistGradientBoostingRegressor(**HGB).fit(p['X'], np.log(p['s2'] + 1e-6))
    smear = (p['n'] * p['s2']).sum() / (p['n'] * np.exp(noise.predict(p['X']))).sum()
    prior = HistGradientBoostingRegressor(**HGB).fit(p['X'], p['delta'] ** 2 - p['s2'] / p['n'])
    return (t(np.exp(noise.predict(X)) * smear, torch.float32),
            t(np.clip(prior.predict(X), PROTOCOL['prior_var_floor'], None), torch.float32))


@torch.inference_mode()
def evaluate_fold(name, seed, P, Y, ts, vs, parts, X, records):
    g_noise, g_prior = global_estimates(parts)
    l_noise, l_prior = local_estimates(parts, X)
    g_tau = g_noise / g_prior
    l_tau = (l_noise / l_prior).clamp(*PROTOCOL['tau_local_clip'])
    event(stage='estimates', fold=name, seed=seed, noise=g_noise, prior=g_prior, tau=g_tau,
          local_tau_median=float(l_tau.median()))
    Q = t(vs)
    PQ, YQ = P[Q], Y[Q]
    mq = torch.isfinite(YQ)
    for rep in range(REPS):
        S = t(np.random.default_rng(9127 + rep).permutation(ts)[:max(SHOTS)])
        R = Y[S] - P[S]
        ms = torch.isfinite(R)
        csum, cnt = torch.where(ms, R, 0).cumsum(0), ms.cumsum(0).float()
        for k in [0] + SHOTS:
            n = torch.zeros_like(PQ[0]) if k == 0 else cnt[k - 1]
            rbar = torch.zeros_like(PQ[0]) if k == 0 else csum[k - 1] / n.clamp_min(1)

            def predict(tau):
                w = torch.where(n > 0, n / (n + tau).clamp_min(1e-9), 0.)
                return (PQ + w * rbar).clamp(0, 1)

            def add(rule, tau, noise=None):
                pq = predict(tau)
                row = dict(fold=name, seed=seed, rep=rep, k=k, rule=rule,
                           mse=float(torch.where(mq, pq - YQ, 0).square().sum() / mq.sum()))
                if noise is not None:
                    sd = (noise * (1 + 1 / (n + tau).clamp_min(1e-9))).sqrt()
                    for label, z in Z.items():
                        lo, hi = pq - z * sd, pq + z * sd
                        row[f'cov{label}'] = float((((YQ >= lo) & (YQ <= hi)) & mq).sum() / mq.sum())
                        row[f'width{label}'] = float(((hi.clamp(0, 1) - lo.clamp(0, 1)) * mq).sum() / mq.sum())
                records.append(row)

            for tau in TAUS:
                add(f'tau{tau:g}', tau)
            add('bayes_global', g_tau, g_noise)
            add('bayes_local', l_tau, l_noise)


def summarize(frame):
    unit = frame.groupby(['fold', 'seed', 'k', 'rule'], as_index=False).mean(numeric_only=True)
    mean = unit.groupby(['k', 'rule'], as_index=False).mean(numeric_only=True)
    grid = mean[mean.rule.str.startswith('tau')]
    best = grid.loc[grid.groupby('k').mse.idxmin()].set_index('k')
    pick = lambda rule: mean[mean.rule == rule].set_index('k')
    g, l, f5 = pick('bayes_global'), pick('bayes_local'), pick('tau5')
    table = pd.DataFrame(dict(prior_k0=mean[(mean.k == 0)].mse.iloc[0], best_grid=best.mse, best_tau=best.rule,
                              tau5=f5.mse, bayes_global=g.mse, bayes_local=l.mse,
                              cov90_global=g.cov90, cov90_local=l.cov90)).loc[SHOTS]
    table['global_vs_best'] = table.bayes_global / table.best_grid
    table['local_vs_best'] = table.bayes_local / table.best_grid
    wins = unit[unit.k == 5].pivot_table(index=['fold', 'seed'], columns='rule', values='mse')
    grid_cols = [c for c in wins if c.startswith('tau')]
    wins5 = int((wins.bayes_local < wins[grid_cols].min(1)).sum())
    verdict = dict(
        P1=bool((table.global_vs_best <= 1.005).all()),
        P2=bool((table.local_vs_best.loc[[1, 5, 10]] <= .98).all() and wins5 >= 8),
        P3=bool(((table.cov90_local.loc[[5, 10, 20, 50]] - .9).abs() <= .03).all() and
                ((table.cov90_local - .9).abs() < (table.cov90_global - .9).abs()).loc[[5, 10, 20, 50]].all()),
        local_wins_at_k5=wins5)
    return table, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='fold0 evaluation, seed 17, one rep (pipeline check)')
    args = ap.parse_args()
    global REPS
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[:1] if args.smoke else SEEDS
    REPS = 1 if args.smoke else REPS
    target = OUT / ('smoke' if args.smoke else 'main')
    target.mkdir(exist_ok=True)
    dump(target / 'protocol.json', dict(PROTOCOL, code_sha256=fingerprint(), seeds=seeds, reps=REPS))
    d = prepare(torch.device('cuda'), OUT)
    specs = folds(d)[:3]
    stats = {}
    with threadpool_limits(8):
        for name, ts, vs, tc, vc in specs:
            normalize(d, ts, tc)
            for seed in seeds:
                P, Y, extra = fold_tensors(d, name, vc, seed)
                stats[name, seed] = locus_stats(d, P, Y, extra, ts, vc)
                del P, Y
            event(stage='stats', fold=name)
        records = []
        for name, ts, vs, tc, vc in specs[:1] if args.smoke else specs:
            normalize(d, ts, tc)
            for seed in seeds:
                P, Y, _ = fold_tensors(d, name, vc, seed)
                parts = [stats[g, seed] for g, *_ in specs if g != name]
                evaluate_fold(name, seed, P, Y, ts, vs, parts, stats[name, seed]['X'], records)
                del P, Y
                torch.cuda.empty_cache()
    frame = pd.DataFrame(records)
    frame.to_csv(target / 'results.csv', index=False)
    table, verdict = summarize(frame)
    table.to_csv(target / 'summary.csv')
    dump(target / 'verdict.json', verdict)
    print(table.round(5).to_string())
    print(verdict)


if __name__ == '__main__':
    main()
