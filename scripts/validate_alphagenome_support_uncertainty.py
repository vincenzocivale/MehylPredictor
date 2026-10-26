"""Cross patient/genomic-block uncertainty for matched external adaptations.

Patients are resampled once across overlapping query repetitions; seeds are
averaged losses, not independent observations. Fits and support assignments
remain fixed. Thus intervals describe query generalization conditional on
the observed support fits, not uncertainty of drawing a fresh support set.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from methylation_predictor.storage import FunctionalLocusCache
from verify_alphagenome_capacity import AG, CANON, positions, dump
from validate_alphagenome_hypotheses import solve_bias, t, event
from validate_alphagenome_transfer import patient_targets
from validate_alphagenome_matched_support import (
    OUT, external_pack, patient_split, shrink_mean, residual_prediction, fingerprint,
)
import json

METHODS = ['v3_logit', 'v1_prior_logit', 'v1_prior_residual', 'v3_residual', 'support_mean', 'support_shrunk_mean', 'v3_population']


def crossed_bootstrap(numerator, baseline, counts, seed=20261018, draws=2000):
    """Joint empirical patient/block bootstrap with common paired weights."""
    rng = np.random.default_rng(seed)
    patients, blocks = counts.shape
    pw = rng.multinomial(patients, np.full(patients, 1 / patients), size=draws)
    bw = rng.multinomial(blocks, np.full(blocks, 1 / blocks), size=draws)
    den = ((pw @ counts) * bw).sum(1)
    candidate = ((pw @ numerator) * bw).sum(1) / den
    reference = ((pw @ baseline) * bw).sum(1) / den
    delta = candidate - reference
    return dict(mse_delta=float((numerator-baseline).sum()/counts.sum()),
        mse_delta_percent=float(100*(numerator.sum()/baseline.sum()-1)),
        ci95_crossed_low=float(np.quantile(delta, .025)), ci95_crossed_high=float(np.quantile(delta, .975)),
        ci95_percent_low=float(np.quantile(100*(candidate/reference-1), .025)),
        ci95_percent_high=float(np.quantile(100*(candidate/reference-1), .975)))


@torch.inference_mode()
def run(out, cohort, cache, ag_ids, ag, registry, selected):
    dest = out / 'uncertainty' / cohort; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    assert (out / cohort / 'complete.json').exists()
    axes = np.load(out / cohort / 'axes_rep0.npz'); cc = axes['cpg_idx']
    ss = np.sort(np.r_[axes['support_sample_idx'], axes['query_sample_idx']])
    with h5py.File(CANON / f'methylation/{cohort}_full.h5') as f:
        y = t(patient_targets(f, ss, cc))
    with h5py.File(CANON / 'rna/tcga_rna_official_full.h5') as f:
        ix = positions(f['sample_idx'][:], ss); order = np.argsort(ix)
        raw = f['X'][ix[order]][np.argsort(order)].astype(np.float32)
    blocks = registry.loc[cc, 'pos'].to_numpy() // 5_000_000
    ub, inverse = np.unique(blocks, return_inverse=True); bi = t(inverse)
    shots = [0, 5, 10] + ([50] if cohort == 'epic' else [])
    max_support = 50 if cohort == 'epic' else 10
    sse = {k: {m: np.zeros((len(ss), len(ub)), np.float64) for m in METHODS} for k in shots}
    counts = np.zeros((len(ss), len(ub)), np.float64)
    hashes = {}
    for seed in [17, 18, 19]:
        pack, prior, hh = external_pack(raw, cc, cache, ag[positions(ag_ids, cc)], seed)
        hashes.update(hh)
        for rep in range(5):
            support, query = patient_split(ss, max_support, rep)
            saved = np.load(out / cohort / f'axes_rep{rep}.npz')
            np.testing.assert_array_equal(ss[support], saved['support_sample_idx'])
            np.testing.assert_array_equal(ss[query], saved['query_sample_idx'])
            sy, qy = y[t(support)], y[t(query)]
            mask = torch.isfinite(qy)
            if seed == 17:
                count = torch.zeros((len(query), len(ub)), device='cuda', dtype=torch.float64)
                count.scatter_add_(1, bi[None].expand(len(query), -1), mask.double())
                counts[query] += count.cpu().numpy()
            rs = {m: pack[m]['u'][t(support)] @ pack[m]['a'].T / 32**.5 for m in pack}
            rq = {m: pack[m]['u'][t(query)] @ pack[m]['a'].T / 32**.5 for m in pack}
            for k in shots:
                for method in METHODS:
                    source = 'v1' if method.startswith('v1') else 'v2'
                    b = pack[source]['prior']; tau = selected[method]
                    if method in ['support_mean', 'support_shrunk_mean']:
                        p = shrink_mean(sy[:k], prior, 0. if method == 'support_mean' else tau)[None].expand_as(qy)
                    elif method == 'v3_population':
                        p = shrink_mean(sy[:k], pack['v2']['population'], tau)[None].expand_as(qy)
                    elif method.endswith('residual'):
                        p = residual_prediction(sy[:k], (b[None]+rs[source][:k]).sigmoid(), (b[None]+rq[source]).sigmoid(), tau)
                    else:
                        bias = solve_bias(b, rs[source][:k], sy[:k], tau) if k else b
                        p = (bias[None]+rq[source]).sigmoid()
                    delta = torch.where(mask, p.double()-qy.double(), 0).square()
                    grouped = torch.zeros((len(query), len(ub)), device='cuda', dtype=torch.float64)
                    grouped.scatter_add_(1, bi[None].expand(len(query), -1), delta)
                    sse[k][method][query] += grouped.cpu().numpy()/3
        event(stage='crossed_uncertainty', cohort=cohort, seed=seed)
        del pack
    records = []
    base_metrics = pd.read_csv(out / cohort / 'metrics.csv')
    for k in shots:
        for method in METHODS:
            # Check independently recomputed losses against the completed run.
            # Full-cohort CpG missingness makes mean-of-repetition MSE and
            # pooled-pair MSE differ slightly; compare pooled NPZ reductions.
            numerator, denominator = 0., 0.
            for seed in [17, 18, 19]:
                for rep in range(5):
                    with np.load(out / cohort / f'{method}_seed{seed}_rep{rep}_k{k}.npz') as z:
                        numerator += np.sum(z['mse']*z['n'])/3
                        denominator += np.sum(z['n'])/3
            assert np.isclose(sse[k][method].sum(), numerator, rtol=2e-6, atol=1e-6), (cohort, k, method)
            assert np.isclose(counts.sum(), denominator)
        for method in METHODS[1:]:
            result = crossed_bootstrap(sse[k]['v3_logit'], sse[k][method], counts)
            records.append(dict(view=cohort, support=k, candidate='v3_logit', baseline=method, **result))
        np.savez_compressed(dest / f'patient_block_losses_k{k}.npz', sample_idx=ss, genomic_blocks=ub, counts=counts, **sse[k])
    pd.DataFrame(records).to_csv(dest / 'contrasts.csv', index=False)
    dump(dest / 'complete.json', dict(complete=True, checkpoint_sha256=hashes, paired_recomputation_passed=True,
         uncertainty='Crossed patient and 5Mb genomic block bootstrap; fixed support fits; patients weighted once across repetitions'))


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(); out = args.out
    protocol = dict(methods=METHODS, shots=[0, 5, 10, 50], wgbs_shots=[0, 5, 10], draws=2000,
        block_bp=5_000_000, resampling='Patient weights shared across overlapping query repetitions; independent genomic block weights',
        fixed='Observed support labels, adaptations, query assignments and seeds; losses averaged over seeds',
        code_sha256=fingerprint(__file__))
    path = out / 'uncertainty_protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        dump(path, protocol)
    torch.set_num_threads(4); threadpool_limits(4); torch.backends.cuda.matmul.allow_tf32 = False
    selected = json.loads((out / 'frozen_selection.json').read_text())['selected']
    registry = pd.read_parquet(CANON / 'cpg/registries/array_cpg_map.parquet').set_index('cpg_idx')
    cache = FunctionalLocusCache(locus_store=ROOT / 'local_methyl_data/derived/locus_features_v1')
    with h5py.File(AG) as f:
        ag_ids, ag = f['cpg_idx'][:], f['embedding'][:].astype(np.float32)
    for cohort in ['epic', 'wgbs']:
        run(out, cohort, cache, ag_ids, ag, registry, selected)
    pd.concat([pd.read_csv(out / f'uncertainty/{cohort}/contrasts.csv') for cohort in ['epic', 'wgbs']]).to_csv(out / 'crossed_uncertainty.csv', index=False)


if __name__ == '__main__':
    main()
