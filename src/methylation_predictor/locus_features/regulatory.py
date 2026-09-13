"""Frozen ENCODE processed-peak overlap engine and breadth encoding."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import requests

from .ids import canonical_locus

N_TRACKS = 4165
BLOCK_COUNTS = {'accessibility': 533, 'histone': 1959, 'ctcf': 201, 'tf_binding': 1472}
CORE_HISTONES = {'H3K27AC','H3K27ME3','H3K36ME3','H3K4ME1','H3K4ME3','H3K9AC','H3K9ME3'}


def load_track_contract(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep='\t')
    if len(frame) != N_TRACKS or frame.functional_block.value_counts().to_dict() != BLOCK_COUNTS:
        raise ValueError('frozen track count or functional groups differ')
    if not frame.track_index.is_monotonic_increasing or frame.track_index.duplicated().any():
        raise ValueError('frozen track ordering invalid')
    if not (frame.assembly == 'GRCh38').all() or not (frame.status == 'released').all():
        raise ValueError('frozen track assembly/status invalid')
    if frame.file_accession.isna().any() or frame.file_accession.duplicated().any():
        raise ValueError('file accessions missing or duplicated')
    if frame.md5sum.isna().any() or frame.download_url.isna().any():
        raise ValueError('source MD5/URL missing')
    return frame


def core_mask(frame: pd.DataFrame) -> np.ndarray:
    mask = frame.functional_block.isin(['accessibility','ctcf']) | ((frame.functional_block == 'histone') & frame.histone_mark_group.isin(CORE_HISTONES))
    if int(mask.sum()) != 2401:
        raise ValueError(f'core subset has {int(mask.sum())} tracks, expected 2401')
    return mask.to_numpy(bool)


def breadth_from_counts(counts: np.ndarray, core_count: np.ndarray) -> np.ndarray:
    if counts.shape[1] != 4 or len(core_count) != len(counts):
        raise ValueError('breadth count shape invalid')
    totals = np.asarray([533,1959,201,1472],dtype=np.float32)
    return np.column_stack((counts.astype(np.float32) / totals, core_count.astype(np.float32) / np.float32(2401))).astype(np.float32)


def prepare_sources(store: str | Path, reference_atlas: str | Path, *, workers: int = 24) -> dict:
    store=Path(store); atlas=Path(reference_atlas)
    source=atlas/'all_primary_columns.tsv'
    frame=load_track_contract(source)
    reg=store/'regulatory'; reg.mkdir(parents=True,exist_ok=True)
    dest=reg/'track_contract.tsv'
    if dest.exists() and dest.read_bytes()!=source.read_bytes():
        raise ValueError('existing track contract differs from frozen chr1 atlas')
    if not dest.exists():
        tmp=reg/'.track_contract.tsv.tmp';tmp.write_bytes(source.read_bytes());os.replace(tmp,dest)
    source_dir=reg/'sources';source_dir.mkdir(exist_ok=True)
    records=frame[['file_accession','download_url','md5sum','file_size']].to_dict('records')
    old_manifest=reg/'manifest.json'
    if old_manifest.exists():
        old=json.loads(old_manifest.read_text())
        if old.get('status')=='complete':
            if old.get('track_contract_sha256')!=hashlib.sha256(dest.read_bytes()).hexdigest():
                raise ValueError('complete source manifest has different frozen contract')
            for record in records:
                path=source_dir/f"{record['file_accession']}.bed.gz"
                if not path.exists() or _md5(path)!=str(record['md5sum']).lower():
                    raise ValueError(f'complete source manifest has missing/mismatched file: {path}')
            return old
    status={'schema_version':1,'status':'building','tracks':N_TRACKS,'track_contract_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'verified':0,'missing':N_TRACKS}
    tmp=reg/'.manifest.json.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');os.replace(tmp,reg/'manifest.json')
    def one(record):
        accession=record['file_accession'];expected=str(record['md5sum']).lower();path=source_dir/f'{accession}.bed.gz'
        if path.exists():
            if _md5(path)!=expected:
                raise ValueError(f'existing peak file MD5 differs: {path}')
            return accession,'reused'
        for attempt in range(3):
            part=source_dir/f'.{accession}.{os.getpid()}.{attempt}.part'
            try:
                digest=hashlib.md5()
                with requests.get(record['download_url'],stream=True,timeout=(20,90)) as response:
                    response.raise_for_status()
                    with open(part,'wb') as fh:
                        for block in response.iter_content(1024*1024):
                            if block:
                                digest.update(block);fh.write(block)
                if digest.hexdigest()!=expected:
                    raise ValueError(f'downloaded peak MD5 differs: {accession}')
                os.replace(part,path)
                return accession,'downloaded'
            except Exception:
                part.unlink(missing_ok=True)
                if attempt==2:raise
                time.sleep(2**attempt)
    counts=Counter();errors=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,record) for record in records]
        for fut in as_completed(futures):
            try:
                _,kind=fut.result();counts[kind]+=1
            except Exception as exc:
                errors.append(str(exc))
    status.update({'status':'complete' if not errors else 'failed','verified':sum(counts.values()),'missing':len(errors),'reused':counts['reused'],'downloaded':counts['downloaded'],'errors':errors[:20]})
    tmp.write_text(json.dumps(status,indent=2)+'\n');os.replace(tmp,reg/'manifest.json')
    if errors:
        raise RuntimeError(f'{len(errors)} peak files failed verification; examples={errors[:5]}')
    return status


def _md5(path: Path) -> str:
    h=hashlib.md5()
    with open(path,'rb') as fh:
        for block in iter(lambda:fh.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _load_intervals(path: Path, chrom: str) -> np.ndarray:
    intervals=[]
    with gzip.open(path,'rt') as fh:
        for line in fh:
            if line.startswith(chrom+'\t'):
                parts=line.split('\t',3)
                if len(parts)>=3:
                    intervals.append((int(parts[1]),int(parts[2])))
    if not intervals:
        return np.empty((0,2),dtype=np.int64)
    return np.asarray(intervals,dtype=np.int64)


def point_overlap(position: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    if not len(intervals):
        return np.zeros(len(position),dtype=bool)
    point=np.asarray(position,dtype=np.int64)-1
    starts=np.sort(intervals[:,0]);ends=np.sort(intervals[:,1])
    return np.searchsorted(starts,point,side='right') > np.searchsorted(ends,point,side='right')


def _point_overlap_sorted(sorted_position: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Interval-centric sweep against a sorted 1-based CpG axis."""
    n=len(sorted_position)
    if not len(intervals):
        return np.zeros(n,dtype=bool)
    starts=np.searchsorted(sorted_position,intervals[:,0]+1,side='left')
    ends=np.searchsorted(sorted_position,intervals[:,1]+1,side='left')
    valid=starts<ends
    diff=np.zeros(n+1,dtype=np.int32)
    np.add.at(diff,starts[valid],1)
    np.add.at(diff,ends[valid],-1)
    return np.cumsum(diff[:-1])>0


def compute_regulatory_overlaps(chrom: str, position: np.ndarray, track_manifest: str | Path, source_dir: str | Path, *, output: str | Path | None = None):
    """Build [N,521] packbits without allocating an N-by-4165 dense matrix."""
    frame=load_track_contract(track_manifest)
    pos=np.asarray(position,dtype=np.int64)
    canonical_locus(chrom,1)
    if pos.ndim!=1 or np.any(pos<1) or np.any(pos>=2**32):
        raise ValueError('positions must be positive 1D 1-based coordinates')
    n=len(pos);byte_width=(N_TRACKS+7)//8
    packed=np.zeros((n,byte_width),dtype=np.uint8)
    order=None if np.all(pos[1:]>=pos[:-1]) else np.argsort(pos,kind='stable')
    sorted_pos=pos if order is None else pos[order]
    block_order=list(BLOCK_COUNTS)
    counts=np.zeros((n,4),dtype=np.uint16)
    core_count=np.zeros(n,dtype=np.uint16)
    core=core_mask(frame)
    source_dir=Path(source_dir)
    for i,row in enumerate(frame.itertuples(index=False)):
        path=source_dir/f'{row.file_accession}.bed.gz'
        if not path.exists():
            raise FileNotFoundError(path)
        sorted_hit=_point_overlap_sorted(sorted_pos,_load_intervals(path,chrom))
        if order is None:
            hit=sorted_hit
        else:
            hit=np.empty(n,dtype=bool);hit[order]=sorted_hit
        packed[hit,i//8] |= np.uint8(1 << (7-i%8))
        counts[hit,block_order.index(row.functional_block)] += 1
        if core[i]:
            core_count[hit] += 1
    if output is not None:
        with open(output,'wb') as fh:
            np.save(fh,packed)
    return packed,breadth_from_counts(counts,core_count)


def unpack_embedding_bags(packed: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    bits=np.unpackbits(np.asarray(packed,dtype=np.uint8),axis=1,bitorder='big')[:,:N_TRACKS]
    row,col=np.nonzero(bits)
    offsets=np.empty(len(bits)+1,dtype=np.int64);offsets[0]=0
    np.cumsum(np.bincount(row,minlength=len(bits)),out=offsets[1:])
    return col.astype(np.int64),offsets


def validate_regulatory_chr1(store: str | Path, reference_atlas: str | Path, *, report: str | Path | None = None) -> dict:
    """Regenerate every frozen chr1 overlap, then compare against the full CSR."""
    store=Path(store);atlas=Path(reference_atlas)
    frame=load_track_contract(store/'regulatory/track_contract.tsv')
    frozen=load_track_contract(atlas/'all_primary_columns.tsv')
    if not frame.equals(frozen):
        raise ValueError('regulatory contract differs from frozen chr1 track order')
    coordinates=pd.read_csv(atlas/'cpg_coordinates.tsv.gz',sep='\t')
    ids=np.load(atlas/'cpg_idx.npy',mmap_mode='r')
    n=len(ids)
    axis_ok=(len(coordinates)==n and np.array_equal(coordinates.cpg_idx.to_numpy(),ids)
             and (coordinates['chr']=='chr1').all()
             and np.array_equal(coordinates.position.to_numpy()-1,coordinates.bed_pos0.to_numpy()))
    if not axis_ok:
        raise ValueError('frozen chr1 CpG coordinate axis is inconsistent')
    temp=store/'regulatory'/'.chr1_validation.packbits.npy'
    try:
        packed,breadth=compute_regulatory_overlaps('chr1',coordinates.position.to_numpy(),store/'regulatory/track_contract.tsv',store/'regulatory/sources',output=temp)
        indptr=np.load(atlas/'all_primary_overlap_csr/indptr.npy',mmap_mode='r')
        indices=np.load(atlas/'all_primary_overlap_csr/indices.npy',mmap_mode='r')
        if len(indptr)!=n+1:
            raise ValueError('frozen CSR row axis differs')
        mismatched_bits=0;mismatched_rows=0;track_mismatch=np.zeros(N_TRACKS,dtype=bool)
        blocks=frame.functional_block.to_numpy()
        group_expected={name:0 for name in BLOCK_COUNTS};group_observed={name:0 for name in BLOCK_COUNTS}
        expected_nnz=0;observed_nnz=0
        examples=[]
        for start in range(0,n,5000):
            end=min(n,start+5000)
            actual=np.unpackbits(packed[start:end],axis=1,bitorder='big')[:,:N_TRACKS].astype(bool)
            expected=np.zeros_like(actual)
            lo=int(indptr[start]);hi=int(indptr[end])
            counts=np.diff(np.asarray(indptr[start:end+1],dtype=np.int64))
            rows=np.repeat(np.arange(end-start),counts)
            expected[rows,np.asarray(indices[lo:hi],dtype=np.int64)]=True
            diff=actual!=expected
            mismatched_bits+=int(diff.sum());mismatched_rows+=int(diff.any(axis=1).sum())
            track_mismatch|=diff.any(axis=0)
            if len(examples)<10:
                rr,cc=np.nonzero(diff)
                for k in range(min(10-len(examples),len(rr))):
                    examples.append({'row':int(start+rr[k]),'cpg_idx':int(ids[start+rr[k]]),'position':int(coordinates.position.iloc[start+rr[k]]),'column_index':int(cc[k]),'track_index':int(frame.track_index.iloc[cc[k]]),'expected':int(expected[rr[k],cc[k]]),'actual':int(actual[rr[k],cc[k]])})
            expected_nnz+=int(expected.sum());observed_nnz+=int(actual.sum())
            for name in BLOCK_COUNTS:
                mask=blocks==name
                group_expected[name]+=int(expected[:,mask].sum())
                group_observed[name]+=int(actual[:,mask].sum())
        old_breadth=np.load(atlas/'breadth_features.f32.npy',mmap_mode='r')
        delta=np.abs(breadth.astype(np.float64)-old_breadth.astype(np.float64))
        breadth_result={'shape_exact':breadth.shape==old_breadth.shape,'exact_matches':int((breadth==old_breadth).sum()),'mismatches':int((delta>0).sum()),'max_abs_error':float(np.max(delta)),'per_feature_mismatches':(delta>0).sum(axis=0).astype(int).tolist()}
        output={'status':'complete' if (mismatched_bits==0 and breadth_result['mismatches']==0 and expected_nnz==252488360 and observed_nnz==252488360 and axis_ok) else 'failed',
                'rows':n,'tracks':N_TRACKS,'cpg_idx_axis_exact':bool(axis_ok),'track_contract_exact':True,
                'expected_nnz':expected_nnz,'observed_nnz':observed_nnz,'mismatched_bits':mismatched_bits,'mismatched_rows':mismatched_rows,'mismatched_tracks':int(track_mismatch.sum()),'examples':examples,
                'per_block':{name:{'tracks':BLOCK_COUNTS[name],'expected_nnz':group_expected[name],'observed_nnz':group_observed[name]} for name in BLOCK_COUNTS},'breadth':breadth_result}
        target=Path(report) if report is not None else store/'validation_regulatory_chr1.json'
        temp_report=target.with_name('.'+target.name+'.tmp');temp_report.write_text(json.dumps(output,indent=2)+'\n');os.replace(temp_report,target)
        return output
    finally:
        temp.unlink(missing_ok=True)


def build_regulatory_shard(store: str | Path, chrom: str, *, replace: bool = False) -> dict:
    store=Path(store)
    gate=json.loads((store/'validation_regulatory_chr1.json').read_text())
    if gate.get('status')!='complete':
        raise ValueError('full chr1 regulatory regression has not passed')
    sources=json.loads((store/'regulatory/manifest.json').read_text())
    if sources.get('status')!='complete' or sources.get('verified')!=N_TRACKS:
        raise ValueError('not all frozen ENCODE peak files have passed MD5')
    shard=store/'features'/chrom
    manifest_path=shard/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='complete':
        raise ValueError(f'annotation shard is incomplete: {chrom}')
    keys=np.load(shard/'locus_key.npy',mmap_mode='r')
    pos=np.load(shard/'position.npy',mmap_mode='r')
    if len(keys)!=len(pos) or not np.all((keys & np.uint64(0xFFFFFFFF))==pos):
        raise ValueError(f'{chrom} locus axis invalid')
    output=shard/'regulatory.packbits.npy';breadth_path=shard/'breadth.f32.npy'
    if manifest.get('regulatory_status')=='complete' and output.exists() and breadth_path.exists() and not replace:
        if manifest.get('regulatory_track_contract_sha256')!=sources['track_contract_sha256'] or manifest.get('regulatory_packbits_bitorder')!='big':
            raise ValueError(f'{chrom} regulatory provenance differs from frozen contract')
        if np.load(output,mmap_mode='r').shape!=(len(keys),(N_TRACKS+7)//8) or np.load(breadth_path,mmap_mode='r').shape!=(len(keys),5):
            raise ValueError(f'{chrom} regulatory array shapes differ')
        return {'chromosome':chrom,'rows':len(keys),'status':'reused'}
    if output.exists() and not replace:
        raise ValueError(f'{chrom} has unverified existing regulatory output')
    partial=shard/'.regulatory.packbits.npy.tmp'
    partial_breadth=shard/'.breadth.f32.npy.tmp'
    try:
        packed,breadth=compute_regulatory_overlaps(chrom,pos,store/'regulatory/track_contract.tsv',store/'regulatory/sources',output=partial)
        with open(partial_breadth,'wb') as fh:
            np.save(fh,breadth)
        del packed
        os.replace(partial,output)
        os.replace(partial_breadth,breadth_path)
        manifest.update({'regulatory_status':'complete','regulatory_schema':'regulatory_peak_overlap_v1','regulatory_track_contract_sha256':sources['track_contract_sha256'],'regulatory_tracks':N_TRACKS,'regulatory_packbits_bitorder':'big','breadth_dimension':5,'regulatory_source_manifest':'regulatory/manifest.json'})
        tmp=shard/'.manifest.json.tmp';tmp.write_text(json.dumps(manifest,indent=2)+'\n');os.replace(tmp,manifest_path)
        return {'chromosome':chrom,'rows':len(keys),'status':'complete','packbits_bytes':output.stat().st_size,'breadth_bytes':breadth_path.stat().st_size}
    finally:
        partial.unlink(missing_ok=True);partial_breadth.unlink(missing_ok=True)
