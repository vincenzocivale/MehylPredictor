"""Refit the AlphaGenome model on MethylProphet Table-5 Array+EPIC+WGBS.

All neural fits start from scratch. A single, uniform 1Mb AlphaGenome cache
covers every eligible locus. Released predictions are matched by actual IDs
and targets; no published test metric participates in recipe selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from methylation_predictor.modeling.factory import build_alphagenome_predictor
from methylation_predictor.modeling.alphagenome import local_response_penalty, AlphaGenomePredictorFromCheckpoint
from methylation_predictor.storage import FunctionalLocusCache, SortedIndex
from verify_alphagenome_capacity import CANON, PREP, positions, dump, cosine_lr
from search_alphagenome_hierarchical import fit_known_offsets
from validate_alphagenome_hypotheses import evaluate, UNKNOWN, t, event
from pilot_losses import masked_mse, locus_pcc_loss, group_centered_mse_loss, group_centered_pcc_loss

OUT = ROOT / 'artifacts/alphagenome_mixed_2026-10-18'
NATIVE = ROOT.parent / 'data/alphagenome_cpg_embeddings/chr1.h5'
PROTOCOL = CANON / 'protocols/tcga_mix_chr1'
FILTERS = PREP / 'table5_protocol'
SOURCES = ['array', 'epic', 'wgbs']
STEPS = [1000, 2000, 4000, 8000]
POLICIES = ['equal_source', 'finite_pairs']
SEEDS = [17, 18, 19]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(x):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in x])


def column_subset(dataset, columns):
    """Read only occupied physical column bands, avoiding enormous masks."""
    columns = np.asarray(columns)
    order = np.argsort(columns); ordered = columns[order]
    result = np.empty((dataset.shape[0], len(columns)), np.float32)
    width = 8192
    for band in np.unique(ordered // width):
        lo, hi = np.searchsorted(ordered, [band*width, (band+1)*width])
        values = dataset[:, int(band*width):int(min((band+1)*width, dataset.shape[1]))]
        result[:, order[lo:hi]] = values[:, ordered[lo:hi]-band*width]
    return result


def prepare_cache(out):
    dest = out / 'cache'; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    with h5py.File(NATIVE) as f:
        ids, pos = f['cpg_idx'][:], f['pos'][:]
        assert int(f.attrs['window']) == 1048576
        assert len(np.unique(ids)) == len(ids)
        np.save(dest / 'cpg_idx.npy', ids); np.save(dest / 'pos.npy', pos)
        ag = np.lib.format.open_memmap(dest / 'ag.f16.npy', mode='w+', dtype=np.float16, shape=f['emb_1bp'].shape)
        for start in range(0, len(ids), 16384):
            ag[start:start+16384] = f['emb_1bp'][start:start+16384]
        ag.flush(); del ag
    cache = FunctionalLocusCache(locus_store=ROOT / 'local_methyl_data/derived/locus_features_v1')
    dense = np.empty((len(ids), 23), np.float32)
    counts = np.empty(len(ids), np.int64); bags = []
    for start in range(0, len(ids), 8192):
        x = cache.get(ids[start:start+8192])
        dense[start:start+len(x['dense'])] = x['dense']
        counts[start:start+len(x['dense'])] = np.diff(x['offsets'])
        bags.append(x['track_indices'].astype(np.int16))
    np.save(dest / 'dense.npy', dense)
    np.save(dest / 'indptr.npy', np.r_[0, np.cumsum(counts)])
    np.save(dest / 'indices.npy', np.concatenate(bags))
    with h5py.File(CANON / 'rna/tcga_rna_official_full.h5') as f:
        np.save(dest / 'rna.npy', f['X'][:].astype(np.float32))
        np.save(dest / 'sample_idx.npy', f['sample_idx'][:])
        np.save(dest / 'tissue.npy', decode(f['tissue_name'][:]))
        np.save(dest / 'gene_ids.npy', decode(f['gene_ids'][:]))
    finite = {}
    for source in SOURCES:
        if source == 'array':
            path = PREP / 'methylation/array_table5_chr1.h5'
        elif source == 'epic':
            path = PREP / 'methylation/epic_table5_chr1.h5'
        else:
            path = CANON / 'methylation/wgbs_full.h5'
        with h5py.File(path) as f:
            source_ids = f['cpg_idx'][:]
            if source == 'wgbs':
                cc = np.load(FILTERS / 'wgbs_train_cpg_idx.npy')
                y = column_subset(f['beta'], positions(source_ids, cc))
            else:
                cc = source_ids
                y = f['beta'][:].astype(np.float32)
            assert not np.setdiff1d(cc, ids).size, f'{source}: incomplete AlphaGenome coverage'
            np.save(dest / f'{source}_y.npy', y)
            np.save(dest / f'{source}_cpg_idx.npy', cc)
            np.save(dest / f'{source}_sample_idx.npy', f['sample_idx'][:])
            if source != 'array':
                expected = np.load(FILTERS / f'{source}_train_cpg_idx.npy')
                np.testing.assert_array_equal(np.sort(cc), np.sort(expected))
            if source == 'array':
                rows = positions(f['sample_idx'][:], np.load(PROTOCOL / 'array_train_sample_idx.npy'))
                cols = positions(cc, np.load(PROTOCOL / 'array_train_cpg_idx.npy'))
                n = int(np.isfinite(y[rows][:, cols]).sum())
            else:
                n = int(np.isfinite(y).sum())
            finite[source] = n
        event(stage='prepare_source', source=source, shape=list(y.shape), training_finite_pairs=n)
    expected = dict(array=275082367, epic=115856100, wgbs=63982272)
    assert finite == expected, (finite, expected)
    dump(dest / 'complete.json', dict(complete=True, finite_pairs=finite, native_attributes=dict(window=1048576, ag_dim=1536),
         excluded_wgbs_N_windows=102, array_cpgs=40627, epic_cpgs=71748, wgbs_cpgs=1999446))


class MixedData:
    def __init__(self, out):
        self.out = out; dest = out / 'cache'
        self.ids = np.load(dest / 'cpg_idx.npy'); self.pos = np.load(dest / 'pos.npy')
        self.cpg_index = SortedIndex(self.ids, 'native AlphaGenome CpG IDs')
        self.raw_ag = np.load(dest / 'ag.f16.npy', mmap_mode='r')
        self.ag = t(np.array(self.raw_ag))
        self.raw_dense = np.load(dest / 'dense.npy')
        self.ptr = t(np.load(dest / 'indptr.npy')); self.idx = t(np.load(dest / 'indices.npy'))
        self.raw_rna = np.load(dest / 'rna.npy', mmap_mode='r')
        self.sample_ids = np.load(dest / 'sample_idx.npy')
        self.tissue = np.load(dest / 'tissue.npy'); self.genes = np.load(dest / 'gene_ids.npy').tolist()
        unique = np.unique(self.tissue); labels = np.searchsorted(unique, self.tissue)
        unknown = np.isin(self.tissue, UNKNOWN)
        labels[unknown] = np.arange(len(unique), len(unique)+unknown.sum())
        self.tissue_id = t(labels); self.known_tissue = t(~unknown)
        self.sources = {}
        for source in SOURCES:
            cc = np.load(dest / f'{source}_cpg_idx.npy'); ss = np.load(dest / f'{source}_sample_idx.npy')
            self.sources[source] = SimpleNamespace(cpg_ids=cc, ids=positions(self.ids, cc), sample_ids=ss,
                rna_rows=positions(self.sample_ids, ss), y=t(np.load(dest / f'{source}_y.npy')),
                gpu_ids=t(positions(self.ids, cc)), gpu_rna_rows=t(positions(self.sample_ids, ss)))
        self.official_s = np.load(PROTOCOL / 'array_train_sample_idx.npy')
        self.official_c = np.load(PROTOCOL / 'array_train_cpg_idx.npy')
        self.test_s = np.load(PROTOCOL / 'array_val_sample_idx.npy')
        self.test_c = np.load(PROTOCOL / 'array_val_cpg_idx.npy')
        # Reuse the existing internal development split, never official targets.
        lab = ROOT.parent / 'chr1_alphagenome_v1_2026-10-17/cache/array_chr1.npz'
        with np.load(lab) as f:
            self.inner_c = np.sort(f['dev_train_c'])
            self.inner_vc = np.sort(f['dev_val_c'])
        # The old derived patient re-split differs from the released split.
        # Generate internal patients only within the actual author train IDs.
        order = np.random.default_rng(20261018).permutation(self.official_s)
        nval = round(.1*len(order))
        self.inner_s, self.inner_vs = np.sort(order[nval:]), np.sort(order[:nval])
        assert not np.intersect1d(self.inner_s, self.test_s).size
        assert not np.intersect1d(self.inner_vs, self.test_s).size
        assert np.isin(self.inner_c, self.official_c).all() and np.isin(self.inner_vc, self.official_c).all()
        for source in ['epic', 'wgbs']:
            assert not np.intersect1d(self.sources[source].sample_ids, self.sources['array'].sample_ids).size

    def functional(self, cc):
        starts = self.ptr[cc]; counts = self.ptr[cc+1]-starts
        offsets = torch.zeros(len(cc)+1, dtype=torch.long, device='cuda'); offsets[1:] = counts.cumsum(0)
        total = int(offsets[-1])
        at = torch.repeat_interleave(starts-offsets[:-1], counts, output_size=total) + torch.arange(total, device='cuda')
        return dict(track_indices=self.idx[at].long(), offsets=offsets,
                    dense=torch.cat([self.dense[cc], self.ag[cc].float()], 1), cpg_ids=t(self.ids[cc.cpu().numpy()]))

    def configure(self, split):
        ss = self.inner_s if split == 'dev' else self.official_s
        cc = self.inner_c if split == 'dev' else self.official_c
        pools = {}
        for name, source in self.sources.items():
            rows = positions(source.sample_ids, ss) if name == 'array' else np.arange(len(source.sample_ids))
            cols = positions(source.cpg_ids, cc) if name == 'array' else np.arange(len(source.cpg_ids))
            mean, finite = [], 0
            for start in range(0, len(cols), 4096):
                piece = t(cols[start:start+4096]); y = source.y[t(rows)[:, None], piece[None]]
                mask = torch.isfinite(y); n = mask.sum(0)
                mean.append(torch.where(mask, y, 0).sum(0)/n.clamp_min(1)); finite += int(n.sum())
            pools[name] = dict(rows=t(rows), cols=t(cols), mu=torch.cat(mean), finite=finite)
        self.pools = pools
        training_patients = np.unique(np.concatenate([self.sources[name].sample_ids[p['rows'].cpu().numpy()] for name, p in pools.items()]))
        assert not np.intersect1d(training_patients, self.test_s).size
        train_rna = positions(self.sample_ids, training_patients)
        raw = np.asarray(self.raw_rna)
        mean = raw[train_rna].mean(0, dtype=np.float64).astype(np.float32)
        std = raw[train_rna].std(0, dtype=np.float64).clip(1e-6).astype(np.float32)
        self.rna = t((raw-mean)/std)
        all_train = np.unique(np.concatenate([self.sources[name].ids[p['cols'].cpu().numpy()] for name, p in pools.items()]))
        dense = self.raw_dense.copy()
        dm, ds = float(dense[all_train, 17].mean()), max(float(dense[all_train, 17].std()), 1e-6)
        dense[:, 17] = (dense[:, 17]-dm)/ds; self.dense = t(dense)
        self.norm = dict(rna_mean=mean, rna_std=std, dense17_mean=dm, dense17_std=ds)
        self.fit_patients = training_patients; self.fit_cpgs = self.ids[all_train]
        self.array_fit_s, self.array_fit_c = ss, cc
        dest = self.out / split; dest.mkdir(exist_ok=True)
        np.savez_compressed(dest / 'axes.npz', fit_sample_ids=training_patients, fit_cpg_ids=self.fit_cpgs,
            array_fit_sample_ids=ss, array_fit_cpg_ids=cc, validation_sample_ids=self.inner_vs if split == 'dev' else self.test_s,
            validation_cpg_ids=self.inner_vc if split == 'dev' else self.test_c)
        # Original v3's genomic prior is learned from Array training means.
        prior_path = dest / 'mean_prior.npz'
        if prior_path.exists():
            with np.load(prior_path) as z:
                self.prior = {key: z[key] for key in ['prior_mean', 'prior_scale', 'prior_coef', 'prior_intercept']}
        else:
            ac = self.sources['array'].ids[positions(self.sources['array'].cpg_ids, cc)]
            x = np.asarray(self.raw_ag[ac], np.float32)
            scaler = StandardScaler().fit(x)
            ridge = Ridge(alpha=100.).fit(scaler.transform(x), pools['array']['mu'].cpu().numpy())
            self.prior = dict(prior_mean=scaler.mean_.astype(np.float32), prior_scale=scaler.scale_.astype(np.float32),
                prior_coef=np.asarray(ridge.coef_, np.float32), prior_intercept=np.asarray(ridge.intercept_, np.float32))
            np.savez_compressed(prior_path, **self.prior, fit_cpg_ids=cc)
        self.graph = self.graph_for_split(split, all_train)
        event(stage='configure', split=split, unique_training_patients=len(training_patients), training_loci=len(all_train),
              finite_pairs={k: v['finite'] for k, v in pools.items()})

    @torch.inference_mode()
    def graph_for_split(self, split, train):
        path = self.out / split / 'graph.npy'
        if path.exists():
            return t(np.load(path)).long()
        rng = np.random.default_rng(20261018)
        ac = self.sources['array'].ids[positions(self.sources['array'].cpg_ids, self.array_fit_c)]
        extra = np.setdiff1d(train, ac)
        bank = np.sort(np.r_[ac, rng.choice(extra, min(len(extra), 65536-len(ac)), replace=False)])
        xb = self.ag[t(bank)].float()
        mu, sd = xb.mean(0), xb.std(0, unbiased=False).clamp_min(1e-6)
        xb = (xb-mu)/sd
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(20261018)
            _, _, projection = torch.pca_lowrank(xb, q=64, center=False, niter=2)
        latent = torch.empty((len(self.ids), 64), device='cuda')
        for start in range(0, len(self.ids), 8192):
            x = (self.ag[start:start+8192].float()-mu)/sd
            latent[start:start+8192] = torch.nn.functional.normalize(x@projection, dim=1)
        dense = self.raw_dense
        group = dense[:, :4].argmax(1)*16+dense[:, 4:8].argmax(1)*4
        group += np.searchsorted(np.quantile(dense[train, 17], [.25, .5, .75]), dense[:, 17])
        graph = torch.full((len(self.ids), 5), -1, dtype=torch.long, device='cuda')
        pos = t(self.pos); bt = t(bank)
        for g in np.unique(group[train]):
            queries = t(train[group[train] == g]); bb = t(bank[group[bank] == g])
            for q in queries.split(1024):
                use = bb if len(bb) >= 5 else bt
                sim = latent[q]@latent[use].T
                sim[(pos[q, None]-pos[use][None]).abs() < 1_000_000] = -torch.inf
                val, ix = sim.topk(5, dim=1); good = torch.isfinite(val).all(1)
                graph[q[good]] = use[ix[good]]
                if not good.all():
                    bad = q[~good]; full = latent[bad]@latent[bt].T
                    full[(pos[bad, None]-pos[bt][None]).abs() < 1_000_000] = -torch.inf
                    value, ix = full.topk(5, dim=1); assert torch.isfinite(value).all()
                    graph[bad] = bt[ix]
        assert (graph[t(train)] >= 0).all()
        np.save(path, graph.cpu().numpy().astype(np.int32))
        np.savez_compressed(self.out / split / 'graph_basis.npz', bank_cpg_ids=self.ids[bank],
            feature_mean=mu.cpu().numpy(), feature_std=sd.cpu().numpy(), projection=projection.cpu().numpy())
        event(stage='graph', split=split, bank=len(bank), queries=len(train))
        return graph


def new_model(d, seed):
    torch.manual_seed(seed)
    model = build_alphagenome_predictor(input_dim=len(d.genes), shallow_mean=True).cuda()
    for key, value in d.prior.items():
        getattr(model, key).copy_(t(value))
    return model


@torch.inference_mode()
def save_model(model, d, dest, seed, policy, step, history):
    # Use the same fixed unique training patients for the population references.
    reference = np.sort(np.random.default_rng(20261018).choice(d.fit_patients, 512, replace=False))
    model.reference_u.copy_(model.encode_rna(d.rna[t(positions(d.sample_ids, reference))]).float())
    model.calibration_ready.fill_(True)
    state = dict(schema='alphagenome_local_chr1_v2', model_spec=model.model_spec, model_state=model.state_dict(),
        normalization={k: torch.as_tensor(v) for k, v in d.norm.items()}, gene_ids=d.genes, seed=seed, step=step,
        sources=SOURCES, source_sampling_policy=policy, fit_sample_ids=d.fit_patients, fit_cpg_ids=d.fit_cpgs,
        array_fit_sample_ids=d.array_fit_s, array_fit_cpg_ids=d.array_fit_c, reference_sample_ids=reference,
        reference_genome='hg38', coordinate_convention='1-based cytosine', ag_source=str(NATIVE), ag_dim=1536,
        ag_window=1048576, ag_dataset='emb_1bp', graph_weight=.01, ema_decay=.99, code_sha256=sha(__file__))
    torch.save(state, dest / f'step{step}.pt')
    dump(dest / 'history.json', history)


@torch.inference_mode()
def predictions(model, d, sample_ids, cpg_ids, calibrated=True):
    u = model.encode_rna(d.rna[t(positions(d.sample_ids, sample_ids))]).float()
    output = np.empty((len(sample_ids), len(cpg_ids)), np.float32)
    array_lookup = {int(c): i for i, c in enumerate(d.array_fit_c)}
    fit_rows = positions(d.sources['array'].sample_ids, d.array_fit_s)
    train_u = model.encode_rna(d.rna[t(positions(d.sample_ids, d.array_fit_s))]).float()
    for start in range(0, len(cpg_ids), 512):
        ids = cpg_ids[start:start+512]; cc = d.cpg_index.positions_of(ids)
        f = d.functional(t(cc)); b, a = model.encode_cpg(**f, calibrate=False)
        b, a = b.float(), a.float()
        if calibrated:
            bias = model.calibrated_intercept(b, a, t(np.asarray(d.raw_ag[cc], np.float32)), blend=.25)
            known = np.array([int(c) in array_lookup for c in ids])
            if known.any():
                loc = np.flatnonzero(known)
                columns = positions(d.sources['array'].cpg_ids, ids[known])
                yy = d.sources['array'].y[t(fit_rows)[:, None], t(columns)[None]]
                delta = fit_known_offsets(b[t(loc)], a[t(loc)], train_u, yy, [1.])[1.]
                bias[t(loc)] = b[t(loc)]+delta
        else:
            bias = b
        output[:, start:start+len(ids)] = (bias[None]+u@a.T/model.rank**.5).sigmoid().cpu().numpy()
    return output


@torch.inference_mode()
def dev_score(model, d, dest, step):
    known = np.sort(np.random.default_rng(20261018).choice(d.inner_c, 4096, replace=False))
    rows = positions(d.sources['array'].sample_ids, d.inner_vs)
    tissue = d.tissue[positions(d.sample_ids, d.inner_vs)]
    result = {}
    for label, cc in [('known', known), ('novel_array', d.inner_vc)]:
        cols = positions(d.sources['array'].cpg_ids, cc)
        yy = d.sources['array'].y[t(rows)[:, None], t(cols)[None]]
        p = predictions(model, d, d.inner_vs, cc)
        metric, per = evaluate(yy, t(p), tissue)
        result[label] = metric
        np.savez_compressed(dest / f'step{step}_{label}_per_cpg.npz', cpg_idx=cc, **per)
    dump(dest / f'step{step}_dev.json', result)
    event(stage='dev_evaluate', step=step, recipe=str(dest), metrics=result)
    return result


def train(d, split, policy, seed, max_steps):
    dest = d.out / split / policy / f'seed{seed}'; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    model = new_model(d, seed); ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, fused=True)
    gen = torch.Generator(device='cuda').manual_seed(seed*7919+13)
    rng = np.random.default_rng(seed*7919+13)
    weights = np.ones(3) if policy == 'equal_source' else np.array([d.pools[s]['finite'] for s in SOURCES], np.float64)
    weights /= weights.sum()
    history = []; exposure = {s: dict(updates=0, finite_pairs=0) for s in SOURCES}
    start = time.time()
    for step in range(max_steps):
        model.train()
        for g in optimizer.param_groups:
            g['lr'] = cosine_lr(step, 8000, 200, 2e-4, .1)
        source = SOURCES[rng.choice(3, p=weights)]; data = d.sources[source]; pool = d.pools[source]
        row_slots = torch.randperm(len(pool['rows']), generator=gen, device='cuda')[:256]
        col_slots = torch.randint(len(pool['cols']), (512,), generator=gen, device='cuda')
        # CpGs drawn independently with replacement; duplicate draws receive
        # their normal sampling weight, no target-dependent resampling.
        rr, cc = pool['rows'][row_slots], pool['cols'][col_slots]
        global_c = data.gpu_ids[cc]
        neighbor = d.graph[global_c, torch.randint(5, (len(cc),), generator=gen, device='cuda')]
        joined, inv = torch.unique(torch.cat([global_c, neighbor]), sorted=True, return_inverse=True)
        anchor, other = inv[:len(cc)], inv[len(cc):]
        yy = data.y[rr[:, None], cc[None]]; mask = torch.isfinite(yy); yy = yy.nan_to_num()
        rna_rows = data.gpu_rna_rows[rr]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(d.rna[rna_rows], **d.functional(joined), return_internals=True)
        p = result['beta'][:, anchor].float()
        loss = masked_mse(p, yy, mask)+.15*locus_pcc_loss(p, yy, mask)
        loss += .15*(result['mu_hat'][anchor].float()-pool['mu'][col_slots]).square().mean()
        wm = mask & d.known_tissue[rna_rows, None]
        loss += group_centered_mse_loss(p, yy, wm, d.tissue_id[rna_rows])
        loss += .15*group_centered_pcc_loss(p, yy, wm, d.tissue_id[rna_rows])
        loss += .01*local_response_penalty(result['rna_factors'], result['cpg_factors'][anchor], result['cpg_factors'][other])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        with torch.no_grad():
            for ep, p0 in zip(ema.parameters(), model.parameters()):
                ep.lerp_(p0, .01)
        exposure[source]['updates'] += 1; exposure[source]['finite_pairs'] += int(mask.sum())
        if (step+1) % 500 == 0:
            row = dict(split=split, policy=policy, seed=seed, step=step+1, loss=float(loss), seconds=time.time()-start,
                       exposure=copy.deepcopy(exposure))
            history.append(row); event(stage='train', **row)
        if step+1 in STEPS or step+1 == max_steps:
            save_model(ema, d, dest, seed, policy, step+1, history)
            if split == 'dev':
                dev_score(ema, d, dest, step+1)
    dump(dest / 'complete.json', dict(complete=True, step=max_steps, seconds=time.time()-start,
         exposure=exposure, policy_probabilities=dict(zip(SOURCES, weights.tolist())), neural_parameters=sum(p.numel() for p in model.parameters())))
    del model, ema, optimizer
    torch.cuda.empty_cache()


def select(out):
    candidates = []
    for policy in POLICIES:
        for step in STEPS:
            path = out / f'dev/{policy}/seed17/step{step}_dev.json'
            result = json.loads(path.read_text())
            candidates.append(dict(policy=policy, step=step, **result))
    # Both views matter, with fixed reference denominators from the earliest
    # finite-pair candidate; no official scores or published model metrics.
    reference = next(x for x in candidates if x['policy'] == 'finite_pairs' and x['step'] == 1000)
    for row in candidates:
        row['balanced_mse_ratio'] = .5*(row['known']['mse']/reference['known']['mse'] + row['novel_array']['mse']/reference['novel_array']['mse'])
    chosen = min(candidates, key=lambda r: (r['balanced_mse_ratio'], r['step'], r['policy']))
    value = dict(selected=chosen, candidates=candidates, selection='Equal-weight relative MSE known and novel-Array CpGs on existing internal dev; no official labels',
                 seed=17, caveat='Internal split was used in earlier research; source-matched selection is not a fresh independent validation')
    dump(out / 'frozen_recipe.json', value)
    return chosen


@torch.inference_mode()
def export_hierarchical(d, recipe, seed):
    """Portable final checkpoint includes training offsets, not runner state."""
    dest = d.out / f"official/{recipe['policy']}/seed{seed}"
    target = dest / 'checkpoint.pt'
    if target.exists():
        return target
    state = torch.load(dest / f"step{recipe['step']}.pt", map_location='cpu', weights_only=False)
    spec = dict(state['model_spec'], variant='alphagenome_hierarchical_chr1_v3',
                n_known=len(d.array_fit_c), mean_blend=.25)
    model = build_alphagenome_predictor(**spec).cuda().eval()
    loaded = model.load_state_dict(state['model_state'], strict=False)
    assert set(loaded.missing_keys) == {'known_cpg_idx', 'known_bias_delta', 'known_ready'}
    assert not loaded.unexpected_keys
    u = model.encode_rna(d.rna[t(positions(d.sample_ids, d.array_fit_s))]).float()
    rr = positions(d.sources['array'].sample_ids, d.array_fit_s)
    deltas = []
    for start in range(0, len(d.array_fit_c), 512):
        ids = d.array_fit_c[start:start+512]
        cc = d.cpg_index.positions_of(ids); cols = positions(d.sources['array'].cpg_ids, ids)
        b, a = model.encode_cpg(**d.functional(t(cc)), calibrate=False)
        yy = d.sources['array'].y[t(rr)[:, None], t(cols)[None]]
        deltas.append(fit_known_offsets(b, a, u, yy, [1.])[1.])
    model.set_known_offsets(t(d.array_fit_c), torch.cat(deltas))
    state.update(schema='alphagenome_hierarchical_chr1_v3', model_spec=model.model_spec,
                 model_state=model.state_dict(), freeze=recipe)
    torch.save(state, target)
    # Check inference through the public adapter against the experiment path.
    cc = d.array_fit_c[:64]; ss = d.test_s[:16]
    manual = predictions(model, d, ss, cc)
    adapter = AlphaGenomePredictorFromCheckpoint(target, device='cuda')
    model = adapter.model
    f = d.functional(t(d.cpg_index.positions_of(cc)))
    b, a = model.encode_cpg(**f)
    q = model.encode_rna(d.rna[t(positions(d.sample_ids, ss))]).float()
    direct = (b[None]+q@a.T/model.rank**.5).sigmoid().cpu().numpy()
    np.testing.assert_allclose(manual, direct, atol=2e-6, rtol=0)
    dump(dest / 'inference_check.json', dict(complete=True, checkpoint_sha256=sha(target),
         portable_calibration_matches_runner=True, max_abs_difference=float(np.max(np.abs(manual-direct)))))
    del model, adapter
    return target


@torch.inference_mode()
def evaluate_official(d, recipe):
    dest = d.out / 'evaluation'; dest.mkdir(exist_ok=True)
    records = []
    views = [('train_cpg-val_sample', d.test_s, d.official_c),
             ('val_cpg-train_sample', d.official_s, d.test_c),
             ('val_cpg-val_sample', d.test_s, d.test_c)]
    expected = dict(train_cpg_x_val_sample=30574946, val_cpg_x_train_sample=55155121, val_cpg_x_val_sample=6129547)
    for seed in SEEDS:
        path = export_hierarchical(d, recipe, seed)
        adapter = AlphaGenomePredictorFromCheckpoint(path, device='cuda'); model = adapter.model
        for label, ss, cc in views:
            rr = positions(d.sources['array'].sample_ids, ss); cols = positions(d.sources['array'].cpg_ids, cc)
            yy = d.sources['array'].y[t(rr)[:, None], t(cols)[None]]
            tissue = d.tissue[positions(d.sample_ids, ss)]
            for calibrated in [False, True]:
                arm = 'v3_calibrated' if calibrated else 'raw'
                p = predictions(model, d, ss, cc, calibrated=calibrated)
                np.save(dest / f'{label}_{arm}_seed{seed}.npy', p)
                metric, per = evaluate(yy, t(p), tissue)
                assert metric['pairs'] == expected[label.replace('-', '_x_')], (label, metric['pairs'])
                records.append(dict(view=label, model=arm, seed=seed, **metric))
                np.savez_compressed(dest / f'{label}_{arm}_seed{seed}_per_cpg.npz', cpg_idx=cc, **per)
                event(stage='official', view=label, arm=arm, seed=seed, metrics=metric)
            np.savez_compressed(dest / f'{label}_axes.npz', sample_idx=ss, cpg_idx=cc, tissue=tissue)
        del model, adapter
    pd.DataFrame(records).to_csv(dest / 'metrics.csv', index=False)
    dump(dest / 'complete.json', dict(complete=True, frozen_recipe_sha256=sha(d.out / 'frozen_recipe.json')))


def protocol(out):
    value = dict(task='Released MethylProphet TCGA chr1, Array+EPIC+WGBS', source_protocol_sha256=sha(PROTOCOL / 'protocol.json'),
        source_axes='Canonical exact released patient/CpG split; earlier derived patient re-split rejected',
        auxiliary_N_filter_protocol_sha256=sha(FILTERS / 'protocol.json'),
        sources=SOURCES, training_finite_pairs=dict(array=275082367, epic=115856100, wgbs=63982272),
        representation=dict(path=str(NATIVE), dataset='emb_1bp', dimension=1536, window=1048576,
            note='Uniform native context for all loci; no mixing with the earlier 524288-bp Array-only embeddings'),
        architecture='Existing shallow mean + 8 FFN + rank32 RNA interaction, graph penalty .01; original v3 calibration',
        graph='5 distant >=1Mb matched neighbors; PCA64 on fixed <=65536-locus training bank including every Array training locus',
        graph_caveat='Approximate bank for scalable 2M-locus training; target-free, saved bank and PCA, training-only loci',
        train_steps=STEPS, sampling_policies=POLICIES, selection_seed=17, refit_seeds=SEEDS,
        batch=dict(patients=256, cpgs=512, wgbs_rows=32, cpg_sampling='independent uniform with replacement'),
        optimizer=dict(name='AdamW', lr=.0002, wd=.0001, clip=1., warmup=200, horizon=8000, ema=.99),
        loss='MSE+.15 CpG PCC+.15 source-specific mean MSE+within-tissue MSE+.15 within-tissue PCC+.01 graph',
        calibration='Ridge100 genomic mean on Array training loci; prior blend .25 for other loci; empirical offsets on Array training loci only',
        normalization='All unique training patients across three sources; all 25017 canonical RNA genes',
        selection='Balanced relative known/novel-Array MSE; new internal patients within true released training, old genomic blocks; fixed before official/reference scoring',
        caveats=['Same eligible data and exact Array split; source revision compatibility for auxiliaries is documented in canonical protocol',
            'Original offline source mixing ratios are not recovered; two explicit policies compared internally',
            'Same available data does not imply identical compute, sampled pairs, RNA preprocessing or pretrained information',
            'Array-held CpGs can occur in auxiliary training, exactly as in published mixed-source setting',
            'Auxiliary WGBS repeated measurement is retained for matching training, not counted as a new patient'],
        code_sha256=sha(__file__))
    path = out / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Use a fresh output directory after protocol changes'
    else:
        dump(path, value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--stage', choices=['prepare', 'select', 'refit', 'evaluate', 'all'], default='all')
    args = ap.parse_args(); out = args.out; out.mkdir(parents=True, exist_ok=True)
    protocol(out)
    torch.set_num_threads(4); threadpool_limits(4); torch.backends.cuda.matmul.allow_tf32 = False
    prepare_cache(out)
    if args.stage == 'prepare':
        return
    d = MixedData(out)
    if args.stage in ['all', 'select']:
        d.configure('dev')
        for policy in POLICIES:
            train(d, 'dev', policy, 17, max(STEPS))
        select(out)
    if args.stage in ['all', 'refit', 'evaluate']:
        recipe = json.loads((out / 'frozen_recipe.json').read_text())['selected']
        d.configure('official')
        if args.stage != 'evaluate':
            for seed in SEEDS:
                train(d, 'official', recipe['policy'], seed, recipe['step'])
        if args.stage in ['all', 'evaluate']:
            evaluate_official(d, recipe)


if __name__ == '__main__':
    main()
