"""Attribute feature, mean-head, graph and calibration effects using saved fits.

No new fitting of neural networks and no comparison of unmatched feature arms.
The feature study is the original paired seed17 pilot; the graph/head study
uses three existing genomic folds and three seeds with the same training axes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from methylation_predictor.modeling.factory import build_alphagenome_predictor
from verify_alphagenome_capacity import prepare, folds, normalize, dump, LAB
from pilot import Candidate
from validate_alphagenome_hypotheses import evaluate, solve_bias, permutations, UNKNOWN, t, event
from validate_alphagenome_matched_support import OUT, fingerprint, shrink_mean, residual_prediction


@torch.inference_mode()
def feature_study(d, out):
    dest = out / 'feature_attribution'; dest.mkdir(exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    # These original fits used all development training loci, without the
    # later fold-specific embargo. Only compare them to one another.
    ts, vs, tc, vc = d.base_train_s, d.base_val_s, d.base_train_c, d.base_val_c
    normalize(d, ts, tc)
    yy = d.block(t(vs), t(vc)); means = d.block(t(ts), t(tc)).nanmean(0).cpu().numpy()
    scale = StandardScaler().fit(d.raw_ag[tc])
    ridge = Ridge(alpha=100.).fit(scale.transform(d.raw_ag[tc]), means)
    genomic = t(ridge.predict(scale.transform(d.raw_ag[vc])).clip(.005, .995).astype(np.float32))
    reference = np.sort(np.random.default_rng(20261018).choice(ts, 512, replace=False))
    records, hashes = [], {}
    for arm in ['functional', 'alphagenome', 'hybrid', 'hybrid_wt']:
        path = LAB / f'results/{arm}/seed17_step1000.pt'; hashes[str(path)] = fingerprint(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = Candidate(d.rna.shape[1], state['representation'], state['separate_bias']).cuda().eval()
        model.load_state_dict(state['model'])
        bs, aa = [], []
        for piece in t(vc).split(512):
            f = d.functional(piece)
            h = model.cpg_query(f['track_indices'], f['offsets'], f['dense'])
            for layer in model.functional_ffn:
                h = layer(h)
            bs.append(model.mean_head(h).squeeze(-1)); aa.append(model.cpg_factor(h))
        b, a = torch.cat(bs), torch.cat(aa)
        u = model.rna_factor(model.rna_encoder.shared(d.rna))
        rq = u[t(vs)] @ a.T / model.rank**.5
        rr = u[t(reference)] @ a.T / model.rank**.5
        rawmean = (b[None] + rr).sigmoid().mean(0)
        target = .75 * rawmean + .25 * genomic
        # Same mean solve for every representation. This condition explicitly
        # adds an AG prior even to the functional-only encoder.
        bp = solve_bias(b, rr, target[None].expand_as(rr), 0)
        for prior_label, bias in [('raw', b), ('AG_prior025', bp)]:
            for rep in range(5):
                support = np.random.default_rng(9127 + rep).permutation(ts)[:50]
                sy = d.block(t(support), t(vc)); rs = u[t(support)] @ a.T / model.rank**.5
                for k in [0, 1, 5, 10, 50]:
                    new_b = solve_bias(bias, rs[:k], sy[:k], 5.) if k else bias
                    p = (new_b[None] + rq).sigmoid()
                    metric, per = evaluate(yy, p, d.tissue[vs])
                    records.append(dict(study='features', arm=arm, calibration=prior_label, seed=17,
                                        repetition=rep, support=k, **metric))
                    np.savez_compressed(dest / f'{arm}_{prior_label}_rep{rep}_k{k}.npz', cpg_idx=d.cpg_idx[vc], **per)
        del model
        event(stage='feature_attribution', arm=arm, records=len(records))
    pd.DataFrame(records).to_csv(dest / 'metrics.csv', index=False)
    dump(dest / 'complete.json', dict(complete=True, checkpoint_sha256=hashes, seed=17, step=1000,
        caveat='Original matched feature pilot, one seed only; no cross-fold causal attribution',
        extra_input='AG_prior025 supplies an AlphaGenome genomic prior to every feature arm'))


@torch.inference_mode()
def head_graph_study(d, out):
    dest = out / 'head_graph_attribution'; dest.mkdir(exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    records, hashes = [], {}
    parent = ROOT / 'artifacts/alphagenome_v2_2026-10-18'
    for name, ts, vs, tc, vc in folds(d)[:3]:
        normalize(d, ts, tc)
        known = np.sort(np.random.default_rng(20261018).choice(tc, min(4096, len(tc)), replace=False))
        for seed in [17, 18, 19]:
            for arm in ['baseline', 'mean', 'local01', 'local10']:
                path = parent / name / arm / f'seed{seed}/checkpoint.pt'
                hashes[str(path)] = fingerprint(path)
                state = torch.load(path, map_location='cpu', weights_only=False)
                assert np.array_equal(state['fit_sample_ids'], d.sample_idx[ts])
                assert np.array_equal(state['fit_cpg_ids'], d.cpg_idx[tc])
                m = build_alphagenome_predictor(**state['model_spec']).cuda().eval()
                m.load_state_dict(state['model_state'])
                u = m.encode_rna(d.rna[t(vs)]).float()
                for view, cc in [('novel', vc), ('known', known)]:
                    yy = d.block(t(vs), t(cc)); bb, aa = [], []
                    for piece in t(cc).split(512):
                        b, a = m.encode_cpg(**d.functional(piece), calibrate=False)
                        bb.append(b); aa.append(a)
                    b, a = torch.cat(bb), torch.cat(aa)
                    response = u @ a.T / m.rank**.5
                    for blend in [0., .25]:
                        bias = m.calibrated_intercept(b, a, t(d.raw_ag[cc]), blend=blend)
                        metric, per = evaluate(yy, (bias[None] + response).sigmoid(), d.tissue[vs])
                        records.append(dict(study='head_graph', split=name, seed=seed, arm=arm, view=view, blend=blend, **metric))
                        np.savez_compressed(dest / f'{name}_{arm}_seed{seed}_{view}_w{blend:g}.npz', cpg_idx=d.cpg_idx[cc], **per)
                del m
        event(stage='head_graph_attribution', split=name, records=len(records))
    pd.DataFrame(records).to_csv(dest / 'metrics.csv', index=False)
    dump(dest / 'complete.json', dict(complete=True, checkpoint_sha256=hashes,
         contrasts=['mean minus baseline: shallow mean head', 'local01 minus mean: graph penalty .01',
                    'local10 minus mean: graph penalty .1', 'blend .25 minus 0: intercept calibration'],
         caveat='Existing development folds, previously used in model selection; descriptive attribution'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(); out = args.out; out.mkdir(parents=True, exist_ok=True)
    protocol = dict(feature_arms=['functional', 'alphagenome', 'hybrid', 'hybrid_wt'], feature_seed=17,
        feature_steps=1000, graph_arms=['baseline', 'mean', 'local01', 'local10'], graph_seeds=[17, 18, 19],
        graph_folds=3, support_tau=5., choices='Fixed before this re-evaluation; no new neural fits',
        code_sha256=fingerprint(__file__))
    path = out / 'attribution_protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        dump(path, protocol)
    torch.set_num_threads(4); threadpool_limits(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    temp = out / 'attribution_features'; temp.mkdir(exist_ok=True)
    d = prepare(torch.device('cuda'), temp)
    feature_study(d, out); head_graph_study(d, out)


if __name__ == '__main__':
    main()
