"""Frozen-model confirmation on new chromosomes and independent assay cohorts.

No choices, fitting or calibration use the confirmation target matrices.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import h5py
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from methylation_predictor.storage import FunctionalLocusCache
from methylation_predictor.modeling.alphagenome import AlphaGenomePredictorFromCheckpoint
from train_alphagenome_v2 import load_v1
from verify_alphagenome_capacity import CANON,AG,positions,selected_matrix,dump,LAB
from validate_alphagenome_hypotheses import OUT,evaluate,t,event,UNKNOWN


def decode(x):
    return np.asarray([v.decode() if isinstance(v,bytes) else str(v) for v in x])


def collapse_measurements(y,measurement_patients,query_patients):
    """Average repeated assays per patient, ignoring unobserved measurements."""
    assert len(np.unique(query_patients))==len(query_patients)
    result=[]
    for patient in query_patients:
        values=y[measurement_patients==patient]
        assert len(values)>0
        finite=np.isfinite(values);n=finite.sum(0)
        mean=np.where(finite,values,0).sum(0)/np.maximum(n,1)
        mean[n==0]=np.nan;result.append(mean)
    return np.asarray(result,np.float32)


def patient_targets(f,sample_ids,query_ids):
    ids=f['sample_idx'][:]
    take=np.flatnonzero(np.isin(ids,sample_ids))
    y=selected_matrix(f['beta'],take,positions(f['cpg_idx'][:],query_ids))
    return collapse_measurements(y,ids[take],sample_ids)


def inventory(out):
    result={}
    with h5py.File(CANON/'rna/tcga_rna_official_full.h5') as f:
        rna_ids=f['sample_idx'][:]
    with h5py.File(CANON/'methylation/tcga_array_official_full.h5') as f:
        array_ids=f['sample_idx'][:]
    for source,filename in [('array','tcga_array_official_full.h5'),('epic','epic_full.h5'),('wgbs','wgbs_full.h5')]:
        with h5py.File(CANON/'methylation'/filename) as f:
            ss=f['sample_idx'][:]
            result[source]=dict(shape=list(f['beta'].shape),patients_with_rna=int(np.isin(ss,rna_ids).sum()),
                                unique_patients_with_rna=int(len(np.intersect1d(ss,rna_ids))),
                                patient_overlap_array=int(np.isin(ss,array_ids).sum()),
                                sha256_from_manifest='see canonical manifest; source files are read only')
    dump(out/'inventory.json',result)
    return result


@torch.inference_mode()
def run_view(out,label,filename,sample_ids,query_ids,cache,ag_ids,ag,registry):
    dest=out/'transfer'/label;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    with h5py.File(CANON/'rna/tcga_rna_official_full.h5') as f:
        sr=positions(f['sample_idx'][:],sample_ids)
        raw=f['X'][np.sort(sr)][np.argsort(np.argsort(sr))].astype(np.float32)
        tissue=decode(f['tissue_name'][:])[sr]
        genes=decode(f['gene_ids'][:]).tolist()
    with h5py.File(CANON/'methylation'/filename) as f:
        ids=f['cpg_idx'][:]
        query_ids=np.intersect1d(query_ids,ids)
        y=patient_targets(f,sample_ids,query_ids)
    xag=ag[positions(ag_ids,query_ids)]
    # Every prediction sees exactly the same observed evaluation pairs.
    yy=t(y); records=[]
    axes=dict(sample_idx=sample_ids,cpg_idx=query_ids,tissue=tissue)
    np.savez_compressed(dest/'axes.npz',**axes)
    official=ROOT/'artifacts/alphagenome_hierarchical_2026-10-18/official'
    reference=ROOT/'artifacts/alphagenome_v2_2026-10-18/official'
    models=[('v1',17,None)]+[(label,seed,path) for seed in [17,18,19] for label,path in
                            [('v2',reference/f'seed{seed}/checkpoint.pt'),('v3',official/f'seed{seed}/checkpoint.pt')]]
    for model_name,seed,path in models:
        if path is None:
            m,st=load_v1(SimpleNamespace(raw_rna=raw))
            norm=st['normalization']
        else:
            adapter=AlphaGenomePredictorFromCheckpoint(path,device='cuda')
            m,norm=adapter.model,adapter.normalization
            assert adapter.gene_ids==genes
        def cpu(v):return v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        norm={k:cpu(v) for k,v in norm.items()}
        u=m.encode_rna(t((raw-norm['rna_mean'])/norm['rna_std'])).float()
        prediction=np.empty_like(y)
        for start in range(0,len(query_ids),512):
            cc=query_ids[start:start+512];payload=cache.get(cc)
            dense=payload['dense'].copy()
            dense[:,17]=(dense[:,17]-norm['dense17_mean'])/norm['dense17_std']
            f=dict(track_indices=t(payload['track_indices']),offsets=t(payload['offsets']),
                   dense=t(np.c_[dense,xag[start:start+len(cc)]]),cpg_ids=t(cc))
            b,a=m.encode_cpg(**f)
            prediction[:,start:start+len(cc)]=(b[None]+u@a.T/m.rank**.5).sigmoid().cpu().numpy()
        metric,per=evaluate(yy,t(prediction),tissue)
        row=dict(view=dest.name,model=model_name,seed=seed,cpgs=len(query_ids),patients=len(sample_ids),**metric)
        records.append(row)
        frame=pd.DataFrame(dict(cpg_idx=query_ids,**per)).merge(registry[['cpg_idx','chr','pos']],on='cpg_idx',validate='one_to_one')
        frame.to_csv(dest/f'{model_name}_seed{seed}_per_cpg.csv.gz',index=False)
        event(stage='transfer',**row)
        del m
        if path is not None:del adapter
        torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(dest/'metrics.csv',index=False)
    dump(dest/'complete.json',dict(complete=True,fit_on_evaluation_targets=False,known_tissue_samples=int((~np.isin(tissue,UNKNOWN)).sum())))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(chromosomes=['chr2','chr22'],chromosome_patients='official Array validation patients (new relative to fit)',
                  cohorts=['epic','wgbs'],cohort_loci='all available Array chr1 CpGs with exact global-ID intersection',
                  cohort_patients='all with canonical RNA, no overlap with any Array patient',
                  models=['frozen v1 seed17','frozen v2 and v3 seeds17/18/19'],
                  choices_frozen_before_confirmation=True,coordinate_convention='hg38 1-based cytosine',
                  caveat='AlphaGenome pretraining on reference genome is not excluded; chromosomes are unseen only to our methylation predictor',
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target=out/'transfer_protocol.json'
    if target.exists():
        original=json.loads(target.read_text())
        assert {k:v for k,v in original.items() if k!='code_sha256'}=={k:v for k,v in protocol.items() if k!='code_sha256'}
        if original['code_sha256']!=protocol['code_sha256']:
            dump(out/'transfer_protocol_amendment.json',dict(original_protocol_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 current_code_sha256=protocol['code_sha256'],reason='First WGBS access discovered repeated patient ID2419; average finite replicate beta values per patient. Prior chromosome/EPIC results unchanged.',
                 repeated_measurement_policy='finite mean per CpG within patient, each unique patient has weight one',changes_model_selection=False))
    else:dump(target,protocol)
    torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False
    inventory(out)
    registry=pd.read_parquet(CANON/'cpg/registries/array_cpg_map.parquet')
    cache=FunctionalLocusCache(locus_store=ROOT/'local_methyl_data/derived/locus_features_v1')
    with h5py.File(AG) as f:
        assert not f.attrs['dummy'] and not f.attrs['patient_specific']
        ag_ids=f['cpg_idx'][:];ag=f['embedding'][:].astype(np.float32)
    with h5py.File(CANON/'rna/tcga_rna_official_full.h5') as f:rna_ids=f['sample_idx'][:]
    validation=np.load(CANON/'protocols/array_genomewide/array_val_sample_idx.npy')
    for chrom in protocol['chromosomes']:
        ids=registry.loc[registry.chr==chrom,'cpg_idx'].to_numpy()
        run_view(out,chrom,'tcga_array_official_full.h5',validation,ids,cache,ag_ids,ag,registry)
    chr1=registry.loc[registry.chr=='chr1','cpg_idx'].to_numpy()
    with h5py.File(CANON/'methylation/tcga_array_official_full.h5') as f:array_patients=f['sample_idx'][:]
    st=torch.load(ROOT/'artifacts/alphagenome_hierarchical_2026-10-18/official/seed17/checkpoint.pt',map_location='cpu',weights_only=False)
    fit_ids=np.asarray(st['fit_cpg_ids']);del st
    for source in ['epic','wgbs']:
        filename=source+'_full.h5'
        with h5py.File(CANON/'methylation'/filename) as f:ss=np.intersect1d(f['sample_idx'][:],rna_ids)
        assert not np.intersect1d(ss,array_patients).size
        for label,ids in [('known',np.intersect1d(chr1,fit_ids)),('novel',np.setdiff1d(chr1,fit_ids))]:
            run_view(out,source+'_'+label,filename,ss,ids,cache,ag_ids,ag,registry)
    frames=[pd.read_csv(p) for p in (out/'transfer').glob('*/metrics.csv')]
    pd.concat(frames).to_csv(out/'transfer_metrics.csv',index=False)
    dump(out/'transfer_complete.json',dict(complete=True,protocol_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))


if __name__=='__main__':main()
