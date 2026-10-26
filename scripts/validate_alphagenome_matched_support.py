"""Matched-label controls and frozen external few-shot confirmation.

All adaptation choices are selected on existing internal genomic folds, then
frozen before dev and external scoring. Reusing previously examined cohorts
is explicitly a follow-up confirmation, not a pristine external test.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from methylation_predictor.storage import FunctionalLocusCache
from methylation_predictor.modeling.alphagenome import AlphaGenomePredictorFromCheckpoint
from verify_alphagenome_capacity import prepare, folds, normalize, dump, CANON, AG, positions, LAB
from search_alphagenome_hierarchical import load, source_path
from search_alphagenome_architecture import factors
from validate_alphagenome_hypotheses import solve_bias, evaluate, permutations, t, event, UNKNOWN
from validate_alphagenome_transfer import decode, patient_targets
from train_alphagenome_v2 import load_v1
from types import SimpleNamespace

OUT = ROOT / 'artifacts/alphagenome_matched_support_2026-10-18'
SEEDS = [17, 18, 19]
TAUS = [0., 5., 20.]
SHOTS = [0, 1, 5, 10, 50]
REPS = 5
METHODS = ['support_mean', 'support_shrunk_mean', 'support_tissue_mean',
           'v1_raw_logit', 'v1_prior_logit', 'v1_prior_residual',
           'v2_raw_logit', 'v3_logit', 'v3_residual', 'v3_population', 'v3_shuffle']


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patient_split(ids, max_support, repetition):
    """Nested support, fixed query for every k, whole unique patients only."""
    ids = np.asarray(ids)
    if len(np.unique(ids)) != len(ids) or max_support >= len(ids):
        raise ValueError('Unique patients and a nonempty query set are required')
    order = np.random.default_rng(9127 + repetition).permutation(len(ids))
    return order[:max_support], np.sort(order[max_support:])


def finite_mean(y, fallback):
    mask = torch.isfinite(y)
    n = mask.sum(0)
    mean = torch.where(mask, y, 0).sum(0) / n.clamp_min(1)
    return torch.where(n > 0, mean, fallback), n


def shrink_mean(y, prior, tau):
    observed, n = finite_mean(y, prior)
    weight = n / (n + tau).clamp_min(1)
    return ((1 - weight) * prior + weight * observed).clamp(0, 1)


def residual_prediction(support_y, support_p, query_p, tau):
    residual, n = finite_mean(support_y - support_p, torch.zeros_like(query_p[0]))
    weight = n / (n + tau).clamp_min(1)
    return (query_p + weight * residual).clamp(0, 1)


def tissue_prediction(support_y, prior, support_tissue, query_tissue, tau):
    global_mean = shrink_mean(support_y, prior, tau)
    pred = global_mean[None].expand(len(query_tissue), -1).clone()
    for name in np.unique(query_tissue):
        if name in UNKNOWN:
            continue
        s = np.flatnonzero(support_tissue == name)
        q = np.flatnonzero(query_tissue == name)
        if len(s):
            pred[torch.as_tensor(q, device=pred.device)] = shrink_mean(
                support_y[torch.as_tensor(s, device=pred.device)], global_mean, tau)
    return pred


@torch.inference_mode()
def compare(y, prediction, detailed=False, tissue=None):
    # Same canonical finite pairs and FP64 reductions as the earlier campaign.
    names = np.full(len(y), 'NA') if tissue is None else tissue
    metric, per = evaluate(y, prediction, names, detailed=detailed)
    return metric, per


@torch.inference_mode()
def run_repetition(dest, label, seed, pack, y, tissue, support, query, shots, taus, repetition, prior,
                   selected=None, detailed=False, cpg_ids=None, sample_ids=None):
    """Every method consumes exactly support[:k] and is scored on query."""
    assert not np.intersect1d(support, query).size
    sy, qy = y[t(support)], y[t(query)]
    st, qt = tissue[support], tissue[query]
    records = []
    sp = {name: pack[name]['u'][t(support)] @ pack[name]['a'].T / 32**.5 for name in pack}
    qp = {name: pack[name]['u'][t(query)] @ pack[name]['a'].T / 32**.5 for name in pack}
    axes_file = dest / f'axes_rep{repetition}.npz'
    if not axes_file.exists():
        np.savez_compressed(axes_file, support_sample_idx=sample_ids[support], query_sample_idx=sample_ids[query], cpg_idx=cpg_ids)
    for k in shots:
        for method in METHODS:
            base = 'v1' if method.startswith('v1') else 'v2'
            b = pack[base]['raw'] if '_raw_' in method else pack[base]['prior']
            response_s, response_q = sp[base][:k], qp[base]
            candidates = [0.] if method == 'support_mean' or k == 0 else taus
            if selected is not None:
                candidates = [0. if method == 'support_mean' or k == 0 else selected[method]]
            for tau in candidates:
                if method in ['support_mean', 'support_shrunk_mean']:
                    p = shrink_mean(sy[:k], prior, tau)[None].expand_as(qy)
                elif method == 'support_tissue_mean':
                    p = tissue_prediction(sy[:k], prior, st[:k], qt, tau)
                elif method == 'v3_population':
                    population = pack['v2']['population']
                    p = shrink_mean(sy[:k], population, tau)[None].expand_as(qy)
                elif method.endswith('residual'):
                    p = residual_prediction(sy[:k], (b[None] + response_s).sigmoid(),
                                            (b[None] + response_q).sigmoid(), tau)
                else:
                    new_b = solve_bias(b, response_s, sy[:k], tau) if k else b
                    query_response = response_q
                    if method == 'v3_shuffle':
                        ix = permutations(qt, 20261018 + repetition)
                        # NA is never shuffled across potentially different tissues.
                        ix[np.isin(qt, UNKNOWN)] = np.flatnonzero(np.isin(qt, UNKNOWN))
                        query_response = response_q[t(ix)]
                    p = (new_b[None] + query_response).sigmoid()
                metric, per = compare(qy, p, detailed=detailed, tissue=qt if detailed else None)
                row = dict(view=label, seed=seed, repetition=repetition, support=k, method=method, tau=tau, **metric)
                records.append(row)
                if detailed:
                    target = dest / f'{method}_seed{seed}_rep{repetition}_k{k}.npz'
                    np.savez_compressed(target, cpg_idx=cpg_ids, **per)
    return records


def choose(out):
    frame = pd.concat([pd.read_csv(out / f'fold{k}' / 'metrics.csv') for k in range(3)])
    # Equal weight to folds, seeds, support sizes and repetitions; same rule for all.
    tuning = frame[frame.support > 0].groupby(['method', 'tau'], as_index=False).mse.mean()
    selected = {}
    for method in METHODS:
        rows = tuning[tuning.method == method].sort_values(['mse', 'tau'])
        selected[method] = float(rows.iloc[0].tau)
    dump(out / 'frozen_selection.json', dict(selected=selected, objective='Mean MSE over 3 folds x 3 seeds x 5 support sets x k=1,5,10,50',
         selection_data='existing internal genomic folds only; no dev/external labels', code_sha256=fingerprint(__file__)))
    tuning.to_csv(out / 'tuning.csv', index=False)
    return selected


@torch.inference_mode()
def internal(out, stage):
    if stage == 'dev':
        selected = json.loads((out / 'frozen_selection.json').read_text())['selected']
    else:
        selected = None
    d = prepare(torch.device('cuda'), out)
    specs = folds(d)
    for name, ts, vs, tc, vc in specs[:3] if stage == 'internal' else specs[-1:]:
        dest = out / name; dest.mkdir(exist_ok=True)
        if (dest / 'complete.json').exists():
            continue
        normalize(d, ts, tc)
        assert not np.intersect1d(tc, vc).size and not np.intersect1d(ts, vs).size
        y = d.block(t(np.arange(len(d.sample_idx))), t(vc))
        records = []
        hashes = {}
        for seed in SEEDS:
            pack = {}
            for model, source in [('v1', 'v1_early'), ('v2', 'v2_early')]:
                path = source_path(name, source, seed); hashes[str(path)] = fingerprint(path)
                m = load(name, source, seed)
                b, a = factors(m, d, vc)
                bp = m.calibrated_intercept(b, a, t(d.raw_ag[vc]), blend=.25)
                u = m.encode_rna(d.rna).float()
                pop = (bp[None] + m.reference_u @ a.T / m.rank**.5).sigmoid().mean(0)
                pack[model] = dict(raw=b, prior=bp, a=a, u=u, population=pop)
                prior = m.mean_prior(t(d.raw_ag[vc]))
                del m
            for rep in range(REPS):
                support = np.random.default_rng(9127 + rep).permutation(ts)[:50]
                records += run_repetition(dest, name, seed, pack, y, d.tissue, support, vs,
                    SHOTS, TAUS, rep, prior, selected, detailed=stage == 'dev',
                    cpg_ids=d.cpg_idx[vc], sample_ids=d.sample_idx)
            event(stage=stage, split=name, seed=seed, records=len(records))
        pd.DataFrame(records).to_csv(dest / 'metrics.csv', index=False)
        dump(dest / 'complete.json', dict(complete=True, checkpoint_sha256=hashes,
            fit_cpg_idx=d.cpg_idx[tc].tolist(), query_cpg_idx=d.cpg_idx[vc].tolist(),
            training_patients=d.sample_idx[ts].tolist(), query_patients=d.sample_idx[vs].tolist()))
    del d
    torch.cuda.empty_cache()
    if stage == 'internal':
        choose(out)


@torch.inference_mode()
def external_pack(raw, cc, cache, xag, seed):
    pack, hashes = {}, {}
    for source in ['v1', 'v2']:
        if source == 'v1':
            m, state = load_v1(SimpleNamespace(raw_rna=raw))
            norm = state['normalization']
            path = LAB / 'final_seed17/checkpoint.pt'
        else:
            path = ROOT / f'artifacts/alphagenome_v2_2026-10-18/official/seed{seed}/checkpoint.pt'
            adapter = AlphaGenomePredictorFromCheckpoint(path, device='cuda')
            m, norm = adapter.model, adapter.normalization
        hashes[str(path)] = fingerprint(path)
        norm = {k: t(v) if not torch.is_tensor(v) else v.cuda() for k, v in norm.items()}
        if source == 'v1':
            # Original v1 predates prior/reference buffers. Attach the same
            # training-only genomic prior and reference patients used by v2,
            # encoding those patients through v1's own RNA encoder.
            reference_path = ROOT / f'artifacts/alphagenome_v2_2026-10-18/official/seed{seed}/checkpoint.pt'
            reference = torch.load(reference_path, map_location='cpu', weights_only=False)
            for key in ['prior_mean', 'prior_scale', 'prior_coef', 'prior_intercept']:
                getattr(m, key).copy_(reference['model_state'][key])
            ref_ids = np.asarray(reference['reference_sample_ids'])
            with h5py.File(CANON / 'rna/tcga_rna_official_full.h5') as f:
                ix = positions(f['sample_idx'][:], ref_ids); order = np.argsort(ix)
                ref_raw = f['X'][ix[order]][np.argsort(order)].astype(np.float32)
            m.reference_u.copy_(m.encode_rna((t(ref_raw) - norm['rna_mean']) / norm['rna_std']))
            m.calibration_ready.fill_(True)
        u = m.encode_rna((t(raw) - norm['rna_mean']) / norm['rna_std']).float()
        bb, aa, pp, priors, pops = [], [], [], [], []
        for start in range(0, len(cc), 512):
            ids = cc[start:start + 512]
            f = cache.get(ids); dense = t(f['dense'])
            dense[:, 17] = (dense[:, 17] - norm['dense17_mean']) / norm['dense17_std']
            ag = t(xag[start:start + len(ids)])
            b, a = m.encode_cpg(track_indices=t(f['track_indices']), offsets=t(f['offsets']),
                               dense=torch.cat([dense, ag], 1), calibrate=False)
            bp = m.calibrated_intercept(b, a, ag, blend=.25)
            bb.append(b); aa.append(a); pp.append(bp); priors.append(m.mean_prior(ag))
            pops.append((bp[None] + m.reference_u @ a.T / m.rank**.5).sigmoid().mean(0))
        pack[source] = dict(raw=torch.cat(bb), prior=torch.cat(pp), a=torch.cat(aa), u=u,
                            population=torch.cat(pops))
        prior = torch.cat(priors)
        del m
    return pack, prior, hashes


@torch.inference_mode()
def external(out):
    selected = json.loads((out / 'frozen_selection.json').read_text())['selected']
    cache = FunctionalLocusCache(locus_store=ROOT / 'local_methyl_data/derived/locus_features_v1')
    with h5py.File(AG) as f:
        ag_ids, ag = f['cpg_idx'][:], f['embedding'][:].astype(np.float32)
    with h5py.File(CANON / 'rna/tcga_rna_official_full.h5') as f:
        rna_ids = f['sample_idx'][:]
    with h5py.File(CANON / 'methylation/tcga_array_official_full.h5') as f:
        array_ids = f['sample_idx'][:]
    state = torch.load(ROOT / 'artifacts/alphagenome_hierarchical_2026-10-18/official/seed17/checkpoint.pt', map_location='cpu', weights_only=False)
    fit_ids = np.asarray(state['fit_cpg_ids']); fit_patients = np.asarray(state['fit_sample_ids'])
    registry = pd.read_parquet(CANON / 'cpg/registries/array_cpg_map.parquet')
    novel = np.setdiff1d(registry.loc[registry.chr == 'chr1', 'cpg_idx'], fit_ids)
    for cohort in ['epic', 'wgbs']:
        dest = out / cohort; dest.mkdir(exist_ok=True)
        if (dest / 'complete.json').exists():
            continue
        with h5py.File(CANON / f'methylation/{cohort}_full.h5') as f:
            ss = np.intersect1d(f['sample_idx'][:], rna_ids)
            cc = np.intersect1d(novel, f['cpg_idx'][:])
            y = t(patient_targets(f, ss, cc))
        assert not np.intersect1d(ss, array_ids).size
        assert not np.intersect1d(ss, fit_patients).size
        with h5py.File(CANON / 'rna/tcga_rna_official_full.h5') as f:
            ix = positions(f['sample_idx'][:], ss); order = np.argsort(ix)
            raw = f['X'][ix[order]][np.argsort(order)].astype(np.float32)
            tissue = decode(f['tissue_name'][:])[ix]
        shots = SHOTS if cohort == 'epic' else [0, 1, 5, 10]
        records, hashes = [], {}
        for seed in SEEDS:
            pack, prior, hh = external_pack(raw, cc, cache, ag[positions(ag_ids, cc)], seed)
            hashes.update(hh)
            for rep in range(REPS):
                support, query = patient_split(ss, max(shots), rep)
                records += run_repetition(dest, cohort, seed, pack, y, tissue, support, query,
                    shots, TAUS, rep, prior, selected, detailed=True, cpg_ids=cc, sample_ids=ss)
            event(stage='external', cohort=cohort, seed=seed, records=len(records))
            del pack
        pd.DataFrame(records).to_csv(dest / 'metrics.csv', index=False)
        dump(dest / 'complete.json', dict(complete=True, checkpoint_sha256=hashes,
             unique_patients=len(ss), query_patients=len(ss) - max(shots), support_pool=max(shots),
             novel_cpgs=len(cc), fit_patient_overlap=0, assay_replicates='finite mean within patient'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--stage', choices=['internal', 'dev', 'external', 'all'], default='all')
    args = ap.parse_args(); out = args.out; out.mkdir(parents=True, exist_ok=True)
    protocol = dict(seeds=SEEDS, repetitions=REPS, support=SHOTS, wgbs_support=[0, 1, 5, 10], taus=TAUS,
        methods=METHODS, fitting='Frozen encoders; only support-locus offsets or means are estimated',
        selection='Per-method tau selected using equal-weight mean MSE across internal folds, seeds, k>0 and support sets',
        dev='Support from training patients; fixed disjoint dev query patients',
        external='Support from assay cohort; fixed disjoint query per repetition; nested support; no Array patient overlap',
        v1_external='Only official v1 seed17 exists; repeated across paired v2 seeds, not three independent v1 fits',
        tissue_control='Uses true tissue labels as additional metadata; unknown groups use global mean',
        query_shuffle='Within named tissue only; unknown tissue left unchanged',
        population_control='Training RNA reference population mean, updated using support, no query RNA',
        caveats=['Follow-up after previous dev and external analyses; not a pristine test set',
                 'Support repetitions overlap; neither repetitions nor seed-locus pairs are independent subjects',
                 'WGBS has only 31 unique patients; each test has 21 after holding 10 support patients',
                 'Native MethylProphet ENCODE checkpoint is not a matched Array-only baseline'],
        code_sha256=fingerprint(__file__))
    path = out / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Protocol differs: use a new output directory'
    else:
        dump(path, protocol)
    torch.set_num_threads(4); threadpool_limits(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    for stage in ['internal', 'dev', 'external'] if args.stage == 'all' else [args.stage]:
        if stage in ['internal', 'dev']:
            internal(out, stage)
        else:
            external(out)


if __name__ == '__main__':
    main()
