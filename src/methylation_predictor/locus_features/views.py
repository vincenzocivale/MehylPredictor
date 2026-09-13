"""Lightweight scope views into common chromosome shards."""
from __future__ import annotations
from pathlib import Path
import os
import tempfile
import pandas as pd
import numpy as np

from .store import LocusFeatureStore


def make_view(root: str | Path, dataset: str, scope: str, *, replace: bool=False) -> Path:
    root=Path(root)
    if dataset != 'tcga':
        raise ValueError('currently only TCGA aliases are installed; ENCODE coordinates can be added to the catalog')
    chroms={'chr1':['chr1'],'chr123':['chr1','chr2','chr3'],'genomewide':[f'chr{i}' for i in range(1,23)]+['chrX','chrY']}.get(scope)
    if chroms is None:
        raise ValueError(f'unknown scope: {scope}')
    dest=root/'views'/f'{dataset}_{scope}.parquet'
    if dest.exists() and not replace:
        return dest
    aliases=[pd.read_parquet(root/'aliases'/f'tcga_{t}.parquet') for t in ('array','epic','wgbs')]
    frame=pd.concat(aliases,ignore_index=True).drop_duplicates(['cpg_idx','locus_key'])
    conflicts=frame.groupby('cpg_idx',sort=False).locus_key.nunique()
    if (conflicts > 1).any():
        raise ValueError('TCGA aliases disagree on cpg_idx coordinate')
    frame=frame[frame.chrom.isin(chroms)].copy()
    store=LocusFeatureStore(root)
    row=np.empty(len(frame),dtype=np.int64)
    for chrom in frame.chrom.unique():
        mask=(frame.chrom==chrom).to_numpy()
        keys,_=store._shard(chrom)
        target=frame.loc[mask,'locus_key'].to_numpy(dtype=np.uint64)
        ix=np.searchsorted(keys,target)
        if np.any(ix>=len(keys)) or not np.array_equal(keys[ix],target):
            raise KeyError(f'missing {chrom} feature rows')
        row[mask]=ix
    frame['feature_row']=row
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent,suffix='.parquet',delete=False) as tmp:
        name=tmp.name
    try:
        frame.to_parquet(name,index=False)
        os.replace(name,dest)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return dest
