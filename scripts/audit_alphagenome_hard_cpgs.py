"""Descriptive, matched audit of persistent CpG errors; no model selection."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
from pyfaidx import Fasta
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from verify_alphagenome_capacity import CANON,PREP,AG,positions,dump
from validate_alphagenome_hypotheses import OUT,t,event

SOURCE='https://zhouserver.research.chop.edu/InfiniumAnnotation/current/HM450/HM450.hg38.manifest.tsv.gz'


def audit(out):
    dest=out/'hard_cpgs';dest.mkdir(exist_ok=True)
    frame=pd.read_csv(ROOT/'docs/experiments/alphagenome_hierarchical_2026-10-18_official_errors.csv.gz')
    frame['hard']=frame.worst_decile_seed_count>=2
    manifest=out/'external/HM450.hg38.manifest.tsv.gz'
    header=pd.read_csv(manifest,sep='\t',nrows=0).columns
    maskcols=[c for c in header if c.startswith('MASK_')]
    cols=['probeID','CpG_chrm','CpG_beg','CpG_end','designType','probeCpGcnt','context35','gene_HGNC']+maskcols
    anno=pd.read_csv(manifest,sep='\t',usecols=cols,low_memory=False)
    anno=anno.rename(columns={'probeID':'illumina_probe_ids'})
    assert anno.illumina_probe_ids.is_unique
    frame=frame.merge(anno,on='illumina_probe_ids',how='left',validate='many_to_one')
    mapped=frame.CpG_beg.notna()
    frame['manifest_coordinate_delta']=frame.pos_1based-(frame.CpG_beg+1)
    for col in maskcols:
        frame[col]=frame[col].astype(str).str.lower().map({'true':1.,'false':0.,'1':1.,'0':0.})
    # Match on observed mean/SD only for this descriptive error audit. These
    # quantities never enter zero-shot inference or candidate selection.
    pairs=[]
    for (context,region),group in frame.groupby(['context','region']):
        hard=group[group.hard];easy=group[~group.hard]
        if len(hard)==0 or len(easy)<5:continue
        scaler=StandardScaler().fit(group[['mean','std']])
        nn=NearestNeighbors(n_neighbors=5).fit(scaler.transform(easy[['mean','std']]))
        distances,near=nn.kneighbors(scaler.transform(hard[['mean','std']]))
        for i,(_,row) in enumerate(hard.iterrows()):
            for j,distance in zip(near[i],distances[i]):
                pairs.append(dict(hard=int(row.cpg_idx),control=int(easy.iloc[j].cpg_idx),distance=float(distance)))
    pairs=pd.DataFrame(pairs);pairs.to_csv(dest/'matched_controls.csv.gz',index=False)
    indexed=frame.set_index('cpg_idx');results=[]
    for col in ['MASK_general','MASK_mapping','MASK_sub30_copy','MASK_snp5_GMAF1p','MASK_typeINextBaseSwitch']:
        hard=frame.loc[frame.hard & mapped,col].dropna();easy=frame.loc[~frame.hard & mapped,col].dropna()
        odds,p=fisher_exact([[hard.sum(),len(hard)-hard.sum()],[easy.sum(),len(easy)-easy.sum()]])
        a=indexed.loc[pairs.hard,col].to_numpy();b=indexed.loc[pairs.control,col].to_numpy()
        valid=np.isfinite(a)&np.isfinite(b)
        results.append(dict(annotation=col,hard_fraction=float(hard.mean()),other_fraction=float(easy.mean()),
                            matched_hard_fraction=float(np.mean(a[valid])),matched_control_fraction=float(np.mean(b[valid])),
                            matched_fraction_difference=float(np.mean(a[valid]-b[valid])),
                            unmatched_odds_ratio=float(odds),unmatched_p_exploratory=float(p),matched_pairs=int(valid.sum())))
    # Verify coordinate convention against actual reference sequence.
    fasta=Fasta(str(ROOT.parent/'CpGRepresentationBenchmark/data/external/reference/hg38.fa'),as_raw=True,sequence_always_upper=True,rebuild=False)
    frame['reference_is_CG']=[str(fasta['chr1'][int(pos)-1:int(pos)+1])=='CG' for pos in frame.pos_1based]
    fasta.close()
    frame.to_csv(dest/'annotated_errors.csv.gz',index=False)
    pd.DataFrame(results).to_csv(dest/'qc_enrichment.csv',index=False)
    # Training-only target means for the nearest-neighbour reference bank.
    fit=np.load(CANON/'protocols/tcga_array_chr1/array_train_cpg_idx.npy')
    patients=np.load(CANON/'protocols/tcga_array_chr1/array_train_sample_idx.npy')
    with h5py.File(PREP/'methylation/array_table5_chr1.h5') as f:
        all_ids=f['cpg_idx'][:];ss=positions(f['sample_idx'][:],patients)
        means=[]
        for start in range(0,len(all_ids),1024):
            means.extend(np.nanmean(f['beta'][np.sort(ss),start:start+1024],axis=0))
        fitmean=np.asarray(means)[positions(all_ids,fit)]
    with h5py.File(AG) as f:
        ids=f['cpg_idx'][:];emb=f['embedding'][:].astype(np.float32)
    query=frame.cpg_idx.to_numpy()
    scaler=StandardScaler().fit(emb[positions(ids,fit)])
    bank=torch.nn.functional.normalize(t(scaler.transform(emb[positions(ids,fit)])),dim=1)
    queries=torch.nn.functional.normalize(t(scaler.transform(emb[positions(ids,query)])),dim=1)
    registry=pd.read_parquet(CANON/'cpg/registries/array_cpg_map.parquet').set_index('cpg_idx')
    bankpos=t(registry.loc[fit,'pos'].to_numpy());qpos=t(frame.pos_1based.to_numpy())
    similarities,neighbors=[],[]
    for start in range(0,len(query),256):
        sim=queries[start:start+256]@bank.T
        sim[(qpos[start:start+256,None]-bankpos[None]).abs()<1_000_000]=-torch.inf
        val,ix=sim.topk(1,dim=1);similarities.extend(val[:,0].cpu().numpy());neighbors.extend(ix[:,0].cpu().numpy())
    frame['neighbor_cpg_idx']=fit[neighbors];frame['ag_cosine']=similarities
    frame['neighbor_training_mean']=fitmean[neighbors]
    frame['neighbor_mean_gap']=(frame['mean']-frame.neighbor_training_mean).abs()
    frame.to_csv(dest/'embedding_collisions.csv.gz',index=False)
    summary=dict(loci=len(frame),hard_loci=int(frame.hard.sum()),manifest_mapped=int(mapped.sum()),
                 coordinate_delta_counts={str(k):int(v) for k,v in frame.loc[mapped,'manifest_coordinate_delta'].value_counts().items()},
                 reference_CG_mismatches=int((~frame.reference_is_CG).sum()),
                 hard_reference_mismatches=int((frame.hard & ~frame.reference_is_CG).sum()),
                 hard_qc_mask_fraction=float(frame.loc[frame.hard,'MASK_general'].mean()),
                 hard_qc_unmasked=int(((frame.MASK_general==0)&frame.hard).sum()),
                 qc=results,median_nearest_ag_cosine=float(frame.ag_cosine.median()),
                 nearest_mean_gap_gt03_fraction=float((frame.neighbor_mean_gap>.3).mean()),
                 hard_nearest_mean_gap_gt03_fraction=float((frame.loc[frame.hard,'neighbor_mean_gap']>.3).mean()),
                 caveat='Post-hoc associations; target mean/SD matching is descriptive only. Neighbours use training CpGs, >=1Mb away; no target-matched neighbour filtering.',
                 manifest=dict(url=SOURCE,sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),documentation='https://zwdzwd.github.io/InfiniumAnnotation/manifest.html'))
    dump(dest/'summary.json',summary);event(stage='hard_cpgs',**summary)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args();torch.set_num_threads(4);threadpool_limits(4)
    audit(args.out)


if __name__=='__main__':main()
