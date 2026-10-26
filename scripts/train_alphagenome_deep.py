"""Deep pairwise decoder on the frozen AlphaGenome mixed-source protocol (raw predictions).

Same data cache, splits, mean prior, neighbour graph, batch sequence, loss, optimiser
and EMA as ``train_alphagenome_mixed`` (imported, not modified). The only change is
``AlphaGenomeDeepDecoderPredictor``: the factorised logit gains a zero-initialised
pairwise term from ``DeepDecoder``. That term has no calibration yet, so every score
here is RAW, and the comparator is the raw arm of the existing baseline checkpoints
scored by this same code. Stage ``dev`` uses only the internal development split;
official targets are not read.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import train_alphagenome_mixed as M
from train_alphagenome_mixed import (SOURCES, STEPS, MixedData, positions, t, evaluate, dump, event, cosine_lr,
                                     masked_mse, locus_pcc_loss, group_centered_mse_loss, group_centered_pcc_loss,
                                     local_response_penalty, build_alphagenome_predictor, sha)
from methylation_predictor.modeling.alphagenome_deep import VARIANT

OUT = ROOT / 'artifacts/alphagenome_deep_decoder_2026-10-19'
BASE = M.OUT
POLICY = 'finite_pairs'          # frozen by the baseline's internal selection
DECODER = dict(width=512, n_rounds=4, ffn_per_round=2, n_head_ffn=2, dropout=.15, drop_path=.1,
               cpg_dropout=.1, zero_init=True)
# Fixed before any deep-decoder number exists. Raw baseline vs raw deep, internal dev, seed 17, step 8000.
RULE = dict(authorise_official_refit_if='novel_array.mse(deep) < novel_array.mse(baseline_raw) and '
                                        'known.mse(deep) <= 1.02 * known.mse(baseline_raw)')


def protocol(out):
    value = dict(
        task='Deep pairwise decoder added to the frozen AlphaGenome mixed-source chr1 model; raw predictions only',
        baseline=str(BASE), baseline_recipe=dict(policy=POLICY, steps=8000, seed=17),
        shared_with_baseline=['cache (symlink)', 'dev/mean_prior.npz', 'dev/graph.npy', 'dev/graph_basis.npz',
                              'batch sampler seeds', 'loss', 'AdamW 2e-4 wd 1e-4 clip 1 warmup 200 cosine 8000', 'EMA .99'],
        model=dict(variant=VARIANT, decoder=DECODER, n_tokens=64,
                   formula='logit = b_c + u_p.a_c/sqrt(32) + g(p,c); g zero-initialised; tokens from the 256-d RNA bottleneck'),
        guards=dict(A='cpg_dropout .1', B='drop_path .1 ramped over rounds', C='zero_init rounds 2-4 and output layer',
                    D='dropout .15'),
        training=dict(steps=STEPS, seed=17, anchors_only_residual=True),
        comparator='baseline checkpoints dev/finite_pairs/seed17/step*.pt scored raw by this script',
        decision_rule=RULE, caveats=['decoder hyper-parameters are untuned first choices',
                                     'raw predictions: the baseline headline numbers are calibrated and not comparable'],
        code_sha256=sha(__file__))
    path = out / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Use a fresh output directory after protocol changes'
    else:
        dump(path, value)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'cache').exists():
        (out / 'cache').symlink_to(BASE / 'cache')
    (out / 'dev').mkdir(exist_ok=True)
    for name in ['mean_prior.npz', 'graph.npy', 'graph_basis.npz']:
        if not (out / 'dev' / name).exists():
            shutil.copy2(BASE / 'dev' / name, out / 'dev' / name)


def check_axes(out):
    with np.load(out / 'dev/axes.npz') as a, np.load(BASE / 'dev/axes.npz') as b:
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f'dev axis {key} differs from the baseline')


def new_model(d, seed):
    torch.manual_seed(seed)
    model = build_alphagenome_predictor(variant=VARIANT, input_dim=len(d.genes), shallow_mean=True, decoder=DECODER).cuda()
    for key, value in d.prior.items():
        getattr(model, key).copy_(t(value))
    return model


@torch.inference_mode()
def raw_predictions(model, d, sample_ids, cpg_ids):
    rna = d.rna[t(positions(d.sample_ids, sample_ids))]
    out = np.empty((len(sample_ids), len(cpg_ids)), np.float32)
    for start in range(0, len(cpg_ids), 512):
        ids = cpg_ids[start:start + 512]
        f = d.functional(t(d.cpg_index.positions_of(ids)))
        out[:, start:start + len(ids)] = model(rna, **f)['beta'].float().cpu().numpy()
    return out


@torch.inference_mode()
def dev_score(model, d, dest, step):
    known = np.sort(np.random.default_rng(20261018).choice(d.inner_c, 4096, replace=False))
    rows = positions(d.sources['array'].sample_ids, d.inner_vs)
    tissue = d.tissue[positions(d.sample_ids, d.inner_vs)]
    result = {}
    for label, cc in [('known', known), ('novel_array', d.inner_vc)]:
        cols = positions(d.sources['array'].cpg_ids, cc)
        yy = d.sources['array'].y[t(rows)[:, None], t(cols)[None]]
        metric, per = evaluate(yy, t(raw_predictions(model, d, d.inner_vs, cc)), tissue)
        result[label] = metric
        np.savez_compressed(dest / f'step{step}_{label}_per_cpg.npz', cpg_idx=cc, **per)
    dump(dest / f'step{step}_dev.json', result)
    event(stage='dev_evaluate', step=step, recipe=str(dest), metrics=result)
    return result


def score_baseline(d, out):
    dest = out / 'dev/baseline_raw'; dest.mkdir(exist_ok=True)
    for step in STEPS:
        if (dest / f'step{step}_dev.json').exists():
            continue
        state = torch.load(BASE / f'dev/{POLICY}/seed17/step{step}.pt', map_location='cpu', weights_only=False)
        model = build_alphagenome_predictor(**state['model_spec']).cuda().eval()
        model.load_state_dict(state['model_state'], strict=True)
        dev_score(model, d, dest, step)
        del model
        torch.cuda.empty_cache()


def train(d, seed, max_steps):
    dest = d.out / 'dev' / POLICY / f'seed{seed}'; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    model = new_model(d, seed); ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, fused=True)
    gen = torch.Generator(device='cuda').manual_seed(seed * 7919 + 13)
    rng = np.random.default_rng(seed * 7919 + 13)
    weights = np.array([d.pools[s]['finite'] for s in SOURCES], np.float64); weights /= weights.sum()
    history = []; exposure = {s: dict(updates=0, finite_pairs=0) for s in SOURCES}
    start = time.time()
    for step in range(max_steps):
        model.train()
        for g in optimizer.param_groups:
            g['lr'] = cosine_lr(step, 8000, 200, 2e-4, .1)
        source = SOURCES[rng.choice(3, p=weights)]; data = d.sources[source]; pool = d.pools[source]
        row_slots = torch.randperm(len(pool['rows']), generator=gen, device='cuda')[:256]
        col_slots = torch.randint(len(pool['cols']), (512,), generator=gen, device='cuda')
        rr, cc = pool['rows'][row_slots], pool['cols'][col_slots]
        global_c = data.gpu_ids[cc]
        neighbor = d.graph[global_c, torch.randint(5, (len(cc),), generator=gen, device='cuda')]
        joined, inv = torch.unique(torch.cat([global_c, neighbor]), sorted=True, return_inverse=True)
        anchor, other = inv[:len(cc)], inv[len(cc):]
        yy = data.y[rr[:, None], cc[None]]; mask = torch.isfinite(yy); yy = yy.nan_to_num()
        rna_rows = data.gpu_rna_rows[rr]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            # the pairwise term is only needed on the anchors; neighbours feed the factor penalty
            result = model(d.rna[rna_rows], **d.functional(joined), return_internals=True,
                           residual_index=torch.unique(anchor))
        p = result['beta'][:, anchor].float()
        loss = masked_mse(p, yy, mask) + .15 * locus_pcc_loss(p, yy, mask)
        loss += .15 * (result['mu_hat'][anchor].float() - pool['mu'][col_slots]).square().mean()
        wm = mask & d.known_tissue[rna_rows, None]
        loss += group_centered_mse_loss(p, yy, wm, d.tissue_id[rna_rows])
        loss += .15 * group_centered_pcc_loss(p, yy, wm, d.tissue_id[rna_rows])
        loss += .01 * local_response_penalty(result['rna_factors'], result['cpg_factors'][anchor], result['cpg_factors'][other])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        with torch.no_grad():
            for ep, p0 in zip(ema.parameters(), model.parameters()):
                ep.lerp_(p0, .01)
        exposure[source]['updates'] += 1; exposure[source]['finite_pairs'] += int(mask.sum())
        if (step + 1) % 500 == 0 or step == 0:
            row = dict(split='dev', policy=POLICY, seed=seed, step=step + 1, loss=float(loss), seconds=time.time() - start,
                       residual_rms=float(result['residual'].float().square().mean().sqrt()), exposure=copy.deepcopy(exposure))
            history.append(row); event(stage='train', **row)
        if step + 1 in STEPS or step + 1 == max_steps:
            M.save_model(ema, d, dest, seed, POLICY, step + 1, history)
            dev_score(ema, d, dest, step + 1)
    dump(dest / 'complete.json', dict(complete=True, step=max_steps, seconds=time.time() - start, exposure=exposure,
         policy_probabilities=dict(zip(SOURCES, weights.tolist())), neural_parameters=sum(p.numel() for p in model.parameters())))
    del model, ema, optimizer
    torch.cuda.empty_cache()


def compare(out, max_steps):
    rows = []
    for step in [s for s in STEPS if s <= max_steps]:
        base = json.loads((out / f'dev/baseline_raw/step{step}_dev.json').read_text())
        deep = json.loads((out / f'dev/{POLICY}/seed17/step{step}_dev.json').read_text())
        row = dict(step=step)
        for view in ['known', 'novel_array']:
            for key in ['mse', 'mas_pcc', 'within_tissue']:
                row[f'{view}.{key}.baseline_raw'] = base[view][key]; row[f'{view}.{key}.deep'] = deep[view][key]
            row[f'{view}.mse_ratio'] = deep[view]['mse'] / base[view]['mse']
        rows.append(row)
    last = rows[-1]
    decision = dict(at_step=last['step'], complete_schedule=last['step'] == 8000,
                    novel_improves=last['novel_array.mse_ratio'] < 1, known_not_worse=last['known.mse_ratio'] <= 1.02)
    decision['authorise_official_refit'] = decision['complete_schedule'] and decision['novel_improves'] and decision['known_not_worse']
    dump(out / 'dev_comparison.json', dict(rows=rows, decision=decision, rule=RULE))
    for r in rows:
        print(f"step {r['step']:>5}  known mse ratio {r['known.mse_ratio']:.4f}  novel_array mse ratio {r['novel_array.mse_ratio']:.4f}"
              f"  (novel mas_pcc {r['novel_array.mas_pcc.baseline_raw']:.4f} -> {r['novel_array.mas_pcc.deep']:.4f})", flush=True)
    print(decision, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--stage', choices=['dev'], default='dev')
    ap.add_argument('--max-steps', type=int, default=max(STEPS))
    args = ap.parse_args()
    torch.set_num_threads(4); threadpool_limits(4); torch.backends.cuda.matmul.allow_tf32 = False
    prepare(args.out); protocol(args.out)
    d = MixedData(args.out); d.configure('dev'); check_axes(args.out)
    score_baseline(d, args.out)
    train(d, 17, args.max_steps)
    if args.max_steps == max(STEPS):
        compare(args.out, args.max_steps)


if __name__ == '__main__':
    main()
