"""Match released MethylProphet predictions by ID and canonical target.

The audit stage checks axes and targets without scoring either model. The
score stage requires a frozen recipe and completed mixed-source neural fits.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from train_alphagenome_mixed import OUT, PROTOCOL, FILTERS, SOURCES, sha, positions, decode
from verify_alphagenome_capacity import CANON, dump
from validate_alphagenome_hypotheses import evaluate, t, event

VIEWS = ['train_cpg-val_sample', 'val_cpg-train_sample', 'val_cpg-val_sample']


def axes(out):
    return [
        (np.load(PROTOCOL/'array_val_sample_idx.npy'), np.load(PROTOCOL/'array_train_cpg_idx.npy')),
        (np.load(PROTOCOL/'array_train_sample_idx.npy'), np.load(PROTOCOL/'array_val_cpg_idx.npy')),
        (np.load(PROTOCOL/'array_val_sample_idx.npy'), np.load(PROTOCOL/'array_val_cpg_idx.npy')),
    ]


def audit(out):
    dest = out/'reference/aligned'; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        return
    cache = out/'cache'
    y = np.load(cache/'array_y.npy', mmap_mode='r')
    samples = np.load(cache/'array_sample_idx.npy'); cpgs = np.load(cache/'array_cpg_idx.npy')
    panels = axes(out); predictions, counts = [], []
    for label, (ss, cc) in zip(VIEWS, panels):
        predictions.append(np.lib.format.open_memmap(dest/f'{label}.npy', mode='w+', dtype=np.float32, shape=(len(ss), len(cc))))
        predictions[-1][:] = np.nan
        counts.append(np.zeros((len(ss), len(cc)), np.uint8))
    max_error = [0., 0., 0.]; repeated_difference = [0., 0., 0.]; records = [0, 0, 0]
    files = sorted((out/'reference/released_predictions/eval_results-test.parquet').glob('*.parquet'))
    assert len(files) == 6
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1_000_000, columns=['group_idx', 'sample_idx', 'cpg_idx', 'gt_methyl', 'pred_methyl']):
            data = batch.to_pydict()
            data = {k: np.asarray(v) for k, v in data.items()}
            assert np.isin(data['group_idx'], [0, 1, 2]).all()
            for group in range(3):
                mask = data['group_idx'] == group
                if not mask.any():
                    continue
                sid, cid = data['sample_idx'][mask], data['cpg_idx'][mask]
                ss, cc = panels[group]
                rr, cols = positions(ss, sid), positions(cc, cid)
                canonical = y[positions(samples, sid), positions(cpgs, cid)]
                gt, p = data['gt_methyl'][mask], data['pred_methyl'][mask]
                assert np.isfinite(canonical).all() and np.isfinite(gt).all() and np.isfinite(p).all()
                max_error[group] = max(max_error[group], float(np.max(np.abs(canonical-gt))))
                assert max_error[group] <= 1e-6, 'Released targets differ from canonical data'
                old = predictions[group][rr, cols]
                repeated = np.isfinite(old)
                if repeated.any():
                    repeated_difference[group] = max(repeated_difference[group], float(np.max(np.abs(old[repeated]-p[repeated]))))
                assert repeated_difference[group] <= 1e-6, 'Conflicting predictions for duplicate released pairs'
                predictions[group][rr, cols] = p
                np.add.at(counts[group], (rr, cols), 1)
                records[group] += int(mask.sum())
        event(stage='reference_audit', file=path.name, processed_rows=sum(records))
    expected = dict(train_cpg_x_val_sample=30574946, val_cpg_x_train_sample=55155121, val_cpg_x_val_sample=6129547)
    result = {}
    for group, (label, (ss, cc)) in enumerate(zip(VIEWS, panels)):
        canonical = y[positions(samples, ss)][:, positions(cpgs, cc)]
        valid = np.isfinite(canonical)
        assert np.array_equal(np.isfinite(predictions[group]), valid), (label, int(np.isfinite(predictions[group]).sum()), int(valid.sum()))
        assert int(valid.sum()) == expected[label.replace('-', '_x_')]
        result[label] = dict(pairs=int(valid.sum()), released_records=records[group], duplicate_records=records[group]-int(valid.sum()),
                             max_target_abs_difference=max_error[group], repeated_prediction_abs_difference=repeated_difference[group])
        predictions[group].flush()
        np.savez_compressed(dest/f'{label}_axes.npz', sample_idx=ss, cpg_idx=cc)
    dump(dest/'complete.json', dict(complete=True, views=result, metrics_scored=False,
        released_source_revision=json.loads((out/'reference/released_predictions/provenance.json').read_text())['revision']))


def paired_crossed(candidate_sse, reference_sse, n, draws=2000):
    rng = np.random.default_rng(20261018)
    p, b = n.shape
    pw = rng.multinomial(p, np.full(p, 1/p), size=draws)
    bw = rng.multinomial(b, np.full(b, 1/b), size=draws)
    den = ((pw@n)*bw).sum(1)
    a = ((pw@candidate_sse)*bw).sum(1)/den
    r = ((pw@reference_sse)*bw).sum(1)/den
    return dict(mse_delta=float((candidate_sse-reference_sse).sum()/n.sum()),
                mse_delta_percent=float(100*(candidate_sse.sum()/reference_sse.sum()-1)),
                ci95_delta_low=float(np.quantile(a-r, .025)), ci95_delta_high=float(np.quantile(a-r, .975)),
                ci95_percent_low=float(np.quantile(100*(a/r-1), .025)), ci95_percent_high=float(np.quantile(100*(a/r-1), .975)))


@torch.inference_mode()
def paper_correlations(y, p, per):
    """Both axes and both aggregations used in the original eval.py."""
    sample=[]
    for start in range(0,len(y),256):
        a,b=y[start:start+256].double(),p[start:start+256].double()
        mask=torch.isfinite(a)&torch.isfinite(b);n=mask.sum(1).clamp_min(1)
        aa=torch.where(mask,a,0);bb=torch.where(mask,b,0)
        ac=torch.where(mask,aa-aa.sum(1)[:,None]/n[:,None],0)
        bc=torch.where(mask,bb-bb.sum(1)[:,None]/n[:,None],0)
        av,bv=ac.square().sum(1),bc.square().sum(1)
        corr=(ac*bc).sum(1)/(av*bv).clamp_min(1e-30).sqrt()
        corr[(n<3)|(av<1e-12)|(bv<1e-12)]=torch.nan
        sample.append(corr.cpu().numpy())
    sample=np.concatenate(sample)
    return dict(mas_pcc=float(np.nanmedian(per['corr'])),cpg_pcc_mean=float(np.nanmean(per['corr'])),sample_pcc_mean=float(np.nanmean(sample)),
                sample_pcc_median=float(np.nanmedian(sample)))


@torch.inference_mode()
def score(out):
    assert (out/'evaluation/complete.json').exists() and (out/'frozen_recipe.json').exists()
    audit(out)
    cache = out/'cache'; dest = out/'comparison'; dest.mkdir(exist_ok=True)
    y = np.load(cache/'array_y.npy', mmap_mode='r')
    samples = np.load(cache/'array_sample_idx.npy'); cpgs = np.load(cache/'array_cpg_idx.npy')
    with __import__('h5py').File(CANON/'rna/tcga_rna_official_full.h5') as f:
        rna_ids, all_tissue = f['sample_idx'][:], decode(f['tissue_name'][:])
    registry = pd.read_parquet(CANON/'cpg/registries/array_cpg_map.parquet').set_index('cpg_idx')
    auxiliary = np.union1d(np.load(FILTERS/'epic_train_cpg_idx.npy'), np.load(FILTERS/'wgbs_train_cpg_idx.npy'))
    records, comparisons, strata, regression = [], [], [], []
    for label, (ss, cc) in zip(VIEWS, axes(out)):
        yy = t(y[positions(samples, ss)][:, positions(cpgs, cc)])
        tissue = all_tissue[positions(rna_ids, ss)]
        reference = t(np.load(out/f'reference/aligned/{label}.npy'))
        metric, ref_per = evaluate(yy, reference, tissue)
        metric.update(paper_correlations(yy,reference,ref_per))
        original=json.loads((out/f'reference/released_predictions/eval/log_dict-{label}.parquet.json').read_text())
        keys={'mse':'mse_loss_per_point','mae':'mae_loss_per_point','mas_pcc':'pcc_by_cpg_id-median',
              'cpg_pcc_mean':'pcc_by_cpg_id-mean','sample_pcc_mean':'pcc_by_sample_id-mean','sample_pcc_median':'pcc_by_sample_id-median'}
        for name,key in keys.items():
            difference=abs(metric[name]-original[key])
            assert difference<2e-5,(label,name,metric[name],original[key])
            regression.append(dict(view=label,metric=name,recomputed=metric[name],published=original[key],absolute_difference=difference))
        records.append(dict(view=label, model='MethylProphet_released', seed=42, **metric))
        np.savez_compressed(dest/f'{label}_MethylProphet_per_cpg.npz', cpg_idx=cc, **ref_per)
        blocks = registry.loc[cc, 'pos'].to_numpy()//5_000_000
        block_ids, inv = np.unique(blocks, return_inverse=True); bi = t(inv)
        mask = torch.isfinite(yy)
        n = torch.zeros((len(ss), len(block_ids)), device='cuda', dtype=torch.float64)
        n.scatter_add_(1, bi[None].expand(len(ss), -1), mask.double())
        ref_error = torch.where(mask, reference.double()-yy.double(), 0).square()
        ref_sse = torch.zeros_like(n); ref_sse.scatter_add_(1, bi[None].expand(len(ss), -1), ref_error)
        learned_per, mean_sse = [], torch.zeros_like(n)
        for seed in [17, 18, 19]:
            p = t(np.load(out/f'evaluation/{label}_v3_calibrated_seed{seed}.npy'))
            metric, per = evaluate(yy, p, tissue)
            metric.update(paper_correlations(yy,p,per))
            records.append(dict(view=label, model='AlphaGenome_mixed_v3', seed=seed, **metric))
            rounded_metric, rounded_per = evaluate(yy, p.to(torch.bfloat16).float(), tissue)
            rounded_metric['mas_pcc']=float(np.nanmedian(rounded_per['corr']))
            records.append(dict(view=label, model='AlphaGenome_mixed_v3_output_BF16', seed=seed, **rounded_metric))
            learned_per.append(per)
            error = torch.where(mask, p.double()-yy.double(), 0).square()
            sse = torch.zeros_like(n); sse.scatter_add_(1, bi[None].expand(len(ss), -1), error)
            mean_sse += sse/3
            result = paired_crossed(sse.cpu().numpy(), ref_sse.cpu().numpy(), n.cpu().numpy())
            comparisons.append(dict(view=label, seed=seed, **result))
            for scope, selected in [('all', np.ones(len(cc), bool)), ('seen_auxiliary', np.isin(cc, auxiliary)),
                                     ('unseen_all_training', ~np.isin(cc, np.union1d(auxiliary, np.load(PROTOCOL/'array_train_cpg_idx.npy'))))]:
                if not selected.any():
                    continue
                nn = per['n'][selected]
                strata.append(dict(view=label, seed=seed, scope=scope, loci=int(selected.sum()),
                    our_mse=float(np.average(per['mse'][selected], weights=nn)),
                    reference_mse=float(np.average(ref_per['mse'][selected], weights=nn)),
                    our_mae=float(np.average(per['mae'][selected], weights=nn)),
                    reference_mae=float(np.average(ref_per['mae'][selected], weights=nn)),
                    our_pcc=float(np.nanmedian(per['corr'][selected])),
                    reference_pcc=float(np.nanmedian(ref_per['corr'][selected]))))
        comparisons.append(dict(view=label, seed='mean_loss_seeds', **paired_crossed(mean_sse.cpu().numpy(), ref_sse.cpu().numpy(), n.cpu().numpy())))
        frame = pd.DataFrame(dict(cpg_idx=cc, pos=registry.loc[cc, 'pos'].to_numpy(), n=ref_per['n'],
            reference_mse=ref_per['mse'], our_mse=np.mean([x['mse'] for x in learned_per], axis=0),
            reference_mae=ref_per['mae'], our_mae=np.mean([x['mae'] for x in learned_per], axis=0),
            reference_pcc=ref_per['corr'], our_pcc=np.nanmean([x['corr'] for x in learned_per], axis=0),
            our_bias2=np.mean([x['bias2'] for x in learned_per], axis=0),
            our_centered_mse=np.mean([x['centered_mse'] for x in learned_per], axis=0),
            seen_auxiliary=np.isin(cc, auxiliary)))
        frame['delta_mse'] = frame.our_mse-frame.reference_mse
        frame.to_csv(dest/f'{label}_paired_cpg_errors.csv.gz', index=False)
        frame.nlargest(100, 'our_mse').to_csv(dest/f'{label}_worst100.csv', index=False)
        event(stage='reference_comparison', view=label, comparison=comparisons[-1])
    pd.DataFrame(records).to_csv(dest/'metrics.csv', index=False)
    pd.DataFrame(regression).to_csv(dest/'published_metric_regression.csv',index=False)
    pd.DataFrame(comparisons).to_csv(dest/'paired_contrasts.csv', index=False)
    pd.DataFrame(strata).to_csv(dest/'training_exposure_strata.csv', index=False)
    dump(dest/'complete.json', dict(complete=True, reference_pair_axes_and_truth_exact=True,
        recipe_sha256=sha(out/'frozen_recipe.json'), uncertainty='Paired patient and 5Mb genomic blocks, fixed fitted models; mean losses across seeds, not ensemble',
        published_predictions='Released BF16 inference values, no query methylation inputs or few-shot adaptation'))


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--stage', choices=['audit', 'score'], default='score')
    args = ap.parse_args(); torch.set_num_threads(4); threadpool_limits(4)
    if args.stage == 'audit':
        audit(args.out)
    else:
        score(args.out)


if __name__ == '__main__':
    main()
