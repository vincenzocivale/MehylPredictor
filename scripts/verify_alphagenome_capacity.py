"""Internal-only AlphaGenome capacity/structure pilot; no official scoring.

Reads the frozen v1 laboratory without modifying it. Writes resumable artifacts
under --out. Protocol is frozen before the first fit. See accompanying report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent / 'chr1_alphagenome_v1_2026-10-17'
sys.path.insert(0, str(LAB))
from pilot import Candidate, positions, rows, CANON, PREP, AG
from harness import Data, cosine_lr
from pilot_losses import (masked_mse, locus_pcc_loss,
                          group_centered_mse_loss, group_centered_pcc_loss)
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_cpg_errors import metrics, clean_json


def dump(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2) + '\n')


class CapacityModel(Candidate):
    def __init__(self, genes, arm):
        # Always initialize the common modules in exactly the same order.
        super().__init__(genes, 'hybrid', False)
        self.arm = arm
        depth = int(arm[1:]) if arm.startswith('d') else 8
        self.functional_ffn = nn.ModuleList(list(self.functional_ffn)[:depth])
        if arm == 'gated':
            self.shallow_factor = copy.deepcopy(self.cpg_factor)
            self.gate = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, -2.)
        if arm == 'prototype':
            self.assignment = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 16))
            self.prototypes = nn.Parameter(torch.randn(16, self.rank) * .02)

    def encode_cpg(self, **func):
        h0 = self.cpg_query(func['track_indices'], func['offsets'], func['dense'])
        h = h0
        for block in self.functional_ffn:
            h = block(h)
        a = self.cpg_factor(h)
        penalty = a.sum() * 0.
        if self.arm == 'gated':
            g = self.gate(h0).sigmoid()
            a0 = self.shallow_factor(h0)
            a = a0 + g * (a - a0)
            penalty = .001 * g.mean()
        elif self.arm == 'prototype':
            residual = a
            a = self.assignment(h).softmax(-1) @ self.prototypes + .1 * residual
            penalty = .001 * residual.square().mean()
        return self.mean_head(h).squeeze(-1), a, penalty, h0, h

    def forward(self, rna, **func):
        b, a, penalty, _, _ = self.encode_cpg(**func)
        u = self.rna_factor(self.rna_encoder.shared(rna))
        interaction = u @ a.T / self.rank ** .5
        if self.arm == 'shrink':
            # Penalize the identifiable interaction, not arbitrary factor scale.
            penalty = penalty + .001 * interaction.float().square().mean()
        return {'beta': (b[None] + interaction).sigmoid(),
                'mu_hat': b.sigmoid(), 'penalty': penalty}


def prepare(device, out):
    d = Data(LAB / 'cache/array_chr1.npz', 'dev', device)
    # Limit canonical HDF5 reads to internal development axes; no official
    # validation patient or CpG target is selected or scored.
    ss = np.sort(np.r_[d.train_s.cpu().numpy(), d.val_s.cpu().numpy()])
    cc = np.sort(np.r_[d.train_c.cpu().numpy(), d.val_c.cpu().numpy()])
    si, ci = d.sample_idx[ss], d.cpg_idx[cc]
    d.beta = torch.empty(0, device=device)
    with h5py.File(CANON / 'rna/tcga_rna_official_full.h5', 'r') as h:
        raw = rows(h, 'X', positions(h['sample_idx'][:], si)).astype(np.float32)
    with h5py.File(PREP / 'methylation/array_table5_chr1.h5', 'r') as h:
        sr, cr = positions(h['sample_idx'][:], si), positions(h['cpg_idx'][:], ci)
        # Read selected patients in contiguous column chunks for efficient HDF5
        # I/O, and immediately restrict to internal development columns. The
        # I/O buffer may span other loci; no such target enters any statistic.
        y = selected_matrix(h['beta'], sr, cr)
    with h5py.File(AG, 'r') as h:
        assert not h.attrs['dummy'] and not h.attrs['patient_specific']
        # IDs are unsorted; full sequential feature I/O is faster than tens of
        # thousands of HDF5 point selections. These are target-free features.
        ag = h['embedding'][:][positions(h['cpg_idx'][:], ci)].astype(np.float32)
    # Remap sparse track bags to the compact development locus axis.
    oldptr = d.indptr.cpu().numpy(); oldidx = d.indices.cpu().numpy()
    bags = [oldidx[oldptr[c]:oldptr[c+1]] for c in cc]
    d.indptr = torch.as_tensor(np.r_[0, np.cumsum([len(b) for b in bags])], device=device)
    d.indices = torch.as_tensor(np.concatenate(bags), device=device)
    dense = d.dense[torch.as_tensor(cc, device=device)].float().cpu().numpy()
    pos = d.cpg_pos_bp[torch.as_tensor(cc, device=device)].cpu().numpy()
    d.train_s = torch.as_tensor(np.searchsorted(ss, d.train_s.cpu().numpy()), device=device)
    d.val_s = torch.as_tensor(np.searchsorted(ss, d.val_s.cpu().numpy()), device=device)
    d.train_c = torch.as_tensor(np.searchsorted(cc, d.train_c.cpu().numpy()), device=device)
    d.val_c = torch.as_tensor(np.searchsorted(cc, d.val_c.cpu().numpy()), device=device)
    d.sample_idx, d.cpg_idx = si, ci
    d.tissue = d.tissue[ss]
    names = np.unique(d.tissue)
    tids = np.searchsorted(names, d.tissue)
    unknown = np.isin(d.tissue, ['NA', 'nan', 'None', 'unknown', ''])
    tids[unknown] = np.arange(len(names), len(names) + unknown.sum())
    d.tissue_id = torch.as_tensor(tids, device=device)
    d.known_tissue = torch.as_tensor(~unknown, device=device)
    d.beta = torch.as_tensor(y, dtype=torch.float32, device=device)
    d.cpg_pos_bp = torch.as_tensor(pos, device=device)
    d.raw_rna = raw
    d.raw_dense = dense
    d.raw_ag = ag
    d.block_ids = pos // 5_000_000
    d.base_train_s = d.train_s.cpu().numpy()
    d.base_train_c = d.train_c.cpu().numpy()
    d.base_val_s = d.val_s.cpu().numpy()
    d.base_val_c = d.val_c.cpu().numpy()
    assert np.isfinite(raw).all() and np.isfinite(ag).all()
    np.savez_compressed(out / 'features.npz', cpg_idx=ci, position=pos,
                        ag=ag, dense=dense, train_c=d.base_train_c,
                        val_c=d.base_val_c, train_s=si[d.base_train_s],
                        val_s=si[d.base_val_s])
    return d


def selected_matrix(dataset, sample_rows, cpg_rows):
    order=np.argsort(sample_rows); inv=np.argsort(order)
    result=np.empty((len(sample_rows),len(cpg_rows)),np.float32)
    # HDF5's point selection becomes prohibitively slow for ~34k interleaved
    # columns. One fancy sample axis plus contiguous column slices avoids it.
    for start in range(0,dataset.shape[1],2048):
        local=np.flatnonzero((cpg_rows>=start)&(cpg_rows<start+2048))
        if not len(local): continue
        block=dataset[sample_rows[order],start:start+2048]
        result[:,local]=block[inv][:,cpg_rows[local]-start]
    return result


def folds(d):
    blocks = np.unique(d.block_ids[d.base_train_c])
    order = np.random.default_rng(20261018).permutation(blocks)
    blocksets = [order[k::3] for k in range(3)]
    shuffled = np.random.default_rng(20261018).permutation(d.base_train_s)
    nv = max(1, round(len(shuffled) * .1))
    inner_val = np.sort(shuffled[:nv]); inner_train = np.sort(shuffled[nv:])
    result = []
    for k, bs in enumerate(blocksets):
        held = d.base_train_c[np.isin(d.block_ids[d.base_train_c], bs)]
        result.append((f'fold{k}', inner_train, inner_val,
                       embargo(d, d.base_train_c, bs), held))
    bs = np.unique(d.block_ids[d.base_val_c])
    result.append(('dev', d.base_train_s, d.base_val_s,
                   embargo(d, d.base_train_c, bs), d.base_val_c))
    return result


def embargo(d, candidates, held_blocks):
    pos = d.cpg_pos_bp.cpu().numpy()[candidates]
    keep = np.ones(len(candidates), bool)
    for b in held_blocks:
        keep &= ~((pos >= b * 5_000_000 - 300_000) &
                  (pos < (b + 1) * 5_000_000 + 300_000))
    return candidates[keep]


def normalize(d, train_s, train_c):
    r = d.raw_rna
    mean = r[train_s].mean(0, dtype=np.float64).astype(np.float32)
    sd = r[train_s].std(0, dtype=np.float64).astype(np.float32).clip(1e-6)
    d.rna = torch.as_tensor((r-mean)/sd, device=d.device)
    dense = d.raw_dense.copy()
    mu = dense[train_c, 17].mean(); scale = max(dense[train_c, 17].std(), 1e-6)
    dense[:, 17] = (dense[:, 17]-mu)/scale
    d.dense = torch.as_tensor(np.c_[dense, d.raw_ag], device=d.device)
    return dict(rna_mean=mean, rna_std=sd, dense17_mean=mu, dense17_std=scale)


@torch.inference_mode()
def predict(model, d, ss, cc):
    model.eval()
    result = np.empty((len(ss), len(cc)), np.float32)
    for j in range(0, len(cc), 1024):
        c = torch.as_tensor(cc[j:j+1024], device=d.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            b, a, _, _, _ = model.encode_cpg(**d.functional(c))
        for i in range(0, len(ss), 256):
            s = torch.as_tensor(ss[i:i+256], device=d.device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                u = model.rna_factor(model.rna_encoder.shared(d.rna[s]))
                p = (b[None] + u @ a.T / model.rank ** .5).sigmoid()
            result[i:i+len(s), j:j+len(c)] = p.float().cpu().numpy()
    assert np.isfinite(result).all()
    return result


def score(y, p, tissue):
    m = metrics(y, p)
    n = m['n']; total = n.sum()
    wt = []; centered_y = np.full_like(y, np.nan); centered_p = np.full_like(p, np.nan)
    for name in np.unique(tissue):
        ix = np.flatnonzero(tissue == name)
        if name in ('NA', 'nan', 'None', 'unknown', '') or len(ix) < 20:
            continue
        wt.append(float(np.nanmedian(metrics(y[ix], p[ix])['corr'])))
        mask = np.isfinite(y[ix]); den = np.maximum(mask.sum(0), 1)
        ym = np.where(mask, y[ix], 0).sum(0)/den
        pm = np.where(mask, p[ix], 0).sum(0)/den
        centered_y[ix] = y[ix]-ym
        centered_p[ix] = np.where(mask, p[ix]-pm, np.nan)
    m['within_corr'] = metrics(centered_y, centered_p)['corr']
    summary = {'mse': float(np.sum(m['mse']*n)/total),
               'mas_pcc': float(np.nanmedian(m['corr'])),
               'within_tissue': float(np.nanmean(wt)),
               'bias_fraction': float(np.sum(m['bias2']*n)/np.sum(m['mse']*n)),
               'severe': float(np.sum(m['severe']*n)/total),
               'p10_pcc': float(np.nanquantile(m['corr'], .1))}
    return summary, m


def train_one(d, spec, arm, seed, out, steps):
    name, ts, vs, tc, vc = spec
    dest = out / name / arm / f'seed{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    final = max(steps) if arm.startswith('d') else min(steps)
    if (dest / f'step{final}.json').exists():
        return
    torch.manual_seed(seed)
    model = CapacityModel(d.rna.shape[1], arm).to(d.device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, fused=True)
    gen = torch.Generator(device='cuda').manual_seed(seed*7919+13)
    train_s = torch.as_tensor(ts, device=d.device); train_c = torch.as_tensor(tc, device=d.device)
    mu = torch.full((len(d.cpg_idx),), torch.nan, device=d.device)
    for cols in train_c.split(512):
        mu[cols] = d.block(train_s, cols).nanmean(0)
    truth = d.block(torch.as_tensor(vs, device=d.device), torch.as_tensor(vc, device=d.device)).cpu().numpy()
    np.savez_compressed(out/name/'evaluation_axes.npz', cpg_idx=d.cpg_idx[vc],
                        sample_idx=d.sample_idx[vs], cpg_rows=vc, tissue=d.tissue[vs])
    train_start = time.time(); eval_seconds = 0.
    for step in range(final):
        model.train()
        for group in opt.param_groups:
            group['lr'] = cosine_lr(step, 4000, 200, 2e-4, .1)
        s = train_s[torch.randperm(len(ts), generator=gen, device=d.device)[:256]].sort().values
        c = train_c[torch.randperm(len(tc), generator=gen, device=d.device)[:512]].sort().values
        y = d.block(s, c); mask = torch.isfinite(y); y = y.nan_to_num()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(d.rna[s], **d.functional(c))
        p = pred['beta'].float()
        loss = masked_mse(p, y, mask) + .15*locus_pcc_loss(p, y, mask)
        loss = loss + .15*(pred['mu_hat'].float()-mu[c]).square().mean()
        wm = mask & d.known_tissue[s, None]
        loss = loss + group_centered_mse_loss(p, y, wm, d.tissue_id[s])
        loss = loss + .15*group_centered_pcc_loss(p, y, wm, d.tissue_id[s])
        loss = loss + pred['penalty'].float()
        assert torch.isfinite(loss), (name, arm, seed, step)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        with torch.no_grad():
            for ep, param in zip(ema.parameters(), model.parameters()):
                ep.lerp_(param, .01)
        if (step+1) % 500 == 0:
            print(json.dumps({'split':name,'arm':arm,'seed':seed,'step':step+1,
                              'seconds':round(time.time()-train_start,1)}), flush=True)
        if step+1 in steps:
            t = time.time()
            p = predict(ema, d, vs, vc)
            summary, per = score(truth, p, d.tissue[vs])
            # Only aggregate per-CpG errors are persisted, no patient predictions.
            np.savez_compressed(dest/f'step{step+1}_per_cpg.npz', **per)
            # Deterministic in-sample diagnostics on the same number of loci.
            seen_c = np.sort(np.random.default_rng(20261018).choice(tc, min(1024,len(tc)), replace=False))
            seen_s = np.sort(np.random.default_rng(20261018).choice(ts, min(512,len(ts)), replace=False))
            seen_p = predict(ema, d, seen_s, seen_c)
            seen_y = d.block(torch.as_tensor(seen_s,device=d.device),torch.as_tensor(seen_c,device=d.device)).cpu().numpy()
            train_summary, _ = score(seen_y, seen_p, d.tissue[seen_s])
            # New-patient / known-locus view distinguishes patient from locus OOD.
            known_p = predict(ema, d, vs, seen_c)
            known_y = d.block(torch.as_tensor(vs,device=d.device),torch.as_tensor(seen_c,device=d.device)).cpu().numpy()
            known_summary, _ = score(known_y, known_p, d.tissue[vs])
            if name == 'dev':
                # Embedding probes later use all internal-training loci and the
                # held-out dev loci, with preprocessing fitted on training only.
                hs0, hs, aa = [], [], []
                with torch.inference_mode():
                    for chunk in np.array_split(np.arange(len(d.cpg_idx)), 40):
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            _, a, _, h0, h = ema.encode_cpg(**d.functional(torch.as_tensor(chunk,device=d.device)))
                        hs0.append(h0.float().cpu().numpy()); hs.append(h.float().cpu().numpy()); aa.append(a.float().cpu().numpy())
                np.savez_compressed(dest/f'step{step+1}_representations.npz',
                                    shallow=np.concatenate(hs0), deep=np.concatenate(hs), factors=np.concatenate(aa))
                torch.save({'model':ema.state_dict(),'arm':arm,'seed':seed,'step':step+1}, dest/f'step{step+1}.pt')
            eval_seconds += time.time()-t
            record = dict(split=name,arm=arm,seed=seed,step=step+1,metrics=summary,
                          training_diagnostic=train_summary,known_locus_new_patient=known_summary,
                          parameters=sum(p.numel() for p in model.parameters()),
                          train_seconds=time.time()-train_start-eval_seconds,
                          train_patients=len(ts),train_cpgs=len(tc),eval_patients=len(vs),eval_cpgs=len(vc))
            dump(dest/f'step{step+1}.json', record)
            print(json.dumps(record),flush=True)
    del model, ema, opt
    torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/alphagenome_capacity_2026-10-18')
    ap.add_argument('--seeds',type=int,nargs='+',default=[17,18,19])
    ap.add_argument('--arms',nargs='+',default=['d0','d2','d8','shrink','gated','prototype'])
    ap.add_argument('--steps',type=int,nargs='+',default=[1000,2000])
    ap.add_argument('--splits',nargs='+',default=['fold0','fold1','fold2','dev'])
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    assert torch.cuda.is_available(), 'CUDA required for this pilot'
    protocol=dict(seeds=args.seeds,arms=args.arms,steps=args.steps,splits=args.splits,
                  rank=32,representation='hybrid',folds=3,block_bp=5000000,embargo_bp=300000,
                  seed_folds=20261018,batch=[256,512],ema=.99,lr=.0002,scheduler_horizon=4000,
                  loss='MSE+.15 PCC+.15 mean+WT_MSE+.15 WT_PCC',
                  adaptive_penalty=.001,prototypes=16,prototype_residual_scale=.1,
                  scope='internal development only; no official holdout evaluation',
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=args.out/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())==protocol, 'Protocol changed; use a new output directory'
    else: dump(path,protocol)
    d=prepare(torch.device('cuda'),args.out)
    specs=folds(d)
    for spec in specs:
        if spec[0] not in args.splits: continue
        name,ts,vs,tc,vc=spec
        assert not np.intersect1d(ts,vs).size and not np.intersect1d(tc,vc).size
        (args.out/name).mkdir(exist_ok=True)
        stats=normalize(d,ts,tc)
        np.savez_compressed(args.out/name/'normalization.npz',**stats,
                            train_sample_idx=d.sample_idx[ts],train_cpg_idx=d.cpg_idx[tc])
        print(json.dumps({'split':name,'train_patients':len(ts),'val_patients':len(vs),
                          'train_cpgs':len(tc),'val_cpgs':len(vc)}),flush=True)
        for seed in args.seeds:
            for arm in args.arms:
                train_one(d,spec,arm,seed,args.out,args.steps)
    dump(args.out/'training_complete.json',dict(completed=True,protocol=protocol))


if __name__=='__main__':
    main()
