"""Mmap-backed chromosome shards for the validated annotation contract."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from .annotations import ReferenceAnnotationEngine, encode_annotation_core
from .ids import decode_locus_key
from .ids import locus_key


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as fh:
        for chunk in iter(lambda:fh.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def build_shard(root: str | Path, chrom: str, engine: ReferenceAnnotationEngine, contract: str | Path, *, replace: bool=False):
    root=Path(root); contract=Path(contract)
    gate=root/'validation_chr1.json'
    if not gate.exists() or json.loads(gate.read_text()).get('status')!='complete':
        raise ValueError('full chr1 regression gate has not passed')
    dest=root/'features'/chrom
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if not replace and json.loads((dest/'manifest.json').read_text()).get('status')=='complete':
            manifest=json.loads((dest/'manifest.json').read_text())
            if manifest.get('annotation_schema_sha256')!=sha256(contract):
                raise ValueError(f'existing shard uses different annotation contract: {dest}')
            catalog=pd.read_parquet(root/'catalog'/f'{chrom}.parquet',columns=['locus_key'])
            keys=np.load(dest/'locus_key.npy',mmap_mode='r')
            if len(keys)!=len(catalog) or not np.array_equal(keys,np.sort(catalog.locus_key.to_numpy(dtype=np.uint64))):
                raise ValueError(f'existing shard differs from catalog: {dest}')
            return dest
        if not replace:
            raise ValueError(f'incomplete existing shard: {dest}')
    frame=pd.read_parquet(root/'catalog'/f'{chrom}.parquet')
    frame=frame.sort_values('locus_key')
    positions=frame.position.to_numpy(dtype=np.int64)
    keys=frame.locus_key.to_numpy(dtype=np.uint64)
    if len(np.unique(keys))!=len(keys):
        raise ValueError('duplicate locus keys')
    temp=Path(tempfile.mkdtemp(prefix=f'.{chrom}.',dir=dest.parent))
    try:
        np.save(temp/'locus_key.npy',keys)
        np.save(temp/'position.npy',positions)
        raw=engine.compute_raw_annotations(chrom,positions)
        encoded=encode_annotation_core(raw,contract)
        np.save(temp/'annotation_core.f32.npy',encoded)
        pd.DataFrame({'locus_key':keys,'cpg_island_class':raw.cpg_island_class,'genomic_region':raw.genomic_region,'ccre_class':raw.ccre_class,'dist_tss_abs_log10':raw.dist_tss_abs_log10}).to_parquet(temp/'raw_annotations.parquet',index=False)
        src={p.name:sha256(p) for p in engine.source_root.iterdir() if p.is_file()}
        recovery=Path(__file__).resolve().parents[3]/'annotation_contract_recovery.json'
        frozen=json.loads(recovery.read_text())['source_sha256']
        required={'gencode.v50.annotation.gtf.gz':frozen['gencode'],
                  'cpgIslandExt.hg38.txt.gz':frozen['ucsc_cpg_islands'],
                  'GRCh38-cCREs.Registry-V4.bed':frozen['encode_ccre_v4']}
        if any(src.get(name)!=digest for name,digest in required.items()):
            raise ValueError('static annotation source hash differs from frozen chr1 provenance')
        manifest={'schema_version':1,'reference_genome':'hg38','annotation_schema':'annotation_core_v1','annotation_schema_sha256':sha256(contract),'source_hashes':src,'source_versions':{'cpg_islands':'UCSC hg38 cpgIslandExt','gencode':'v50','ccre':'Registry V4'},'chromosome':chrom,'number_of_loci':len(keys),'feature_dimension':18,'built_at':datetime.now(timezone.utc).isoformat(),'git_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'build_command':sys.argv,'coordinate_convention':'1-based C; BED [position-1,position)','status':'complete'}
        (temp/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(temp,dest)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return dest


class LocusFeatureStore:
    def __init__(self,root: str | Path):
        self.root=Path(root)
        self._shards={}

    def _shard(self,chrom):
        if chrom not in self._shards:
            path=self.root/'features'/chrom
            manifest=json.loads((path/'manifest.json').read_text())
            if manifest.get('status')!='complete' or manifest.get('feature_dimension')!=18:
                raise ValueError(f'incomplete or invalid shard {chrom}')
            contract=Path(__file__).resolve().parents[3]/'resources/locus_features/annotation_core_v1.json'
            if manifest.get('annotation_schema_sha256')!=sha256(contract):
                raise ValueError(f'annotation contract hash differs for shard {chrom}')
            keys=np.load(path/'locus_key.npy',mmap_mode='r')
            values=np.load(path/'annotation_core.f32.npy',mmap_mode='r')
            if values.shape!=(len(keys),18) or not np.all(keys[1:]>keys[:-1]):
                raise ValueError(f'invalid arrays in shard {chrom}')
            self._shards[chrom]=(keys,values)
        return self._shards[chrom]

    def lookup(self,locus_keys):
        query=np.asarray(locus_keys,dtype=np.uint64)
        result=np.empty((len(query),18),dtype=np.float32)
        for chrom in sorted({decode_locus_key(int(k))[0] for k in query}):
            mask=np.array([decode_locus_key(int(k))[0]==chrom for k in query])
            keys,values=self._shard(chrom)
            target=query[mask]
            ix=np.searchsorted(keys,target)
            valid=ix<len(keys)
            valid[valid] &= keys[ix[valid]]==target[valid]
            if not valid.all():
                raise KeyError(f'{chrom}: missing {int((~valid).sum())} loci')
            result[mask]=values[ix]
        return result

    def lookup_or_compute(self, chrom: str, positions, engine: ReferenceAnnotationEngine, contract: str | Path):
        """Reuse installed loci and calculate missing ENCODE/future CpGs identically."""
        pos=np.asarray(positions,dtype=np.int64)
        if pos.ndim != 1:
            raise ValueError('positions must be one-dimensional')
        keys=np.asarray([locus_key(chrom,int(p),lengths=engine.lengths) for p in pos],dtype=np.uint64)
        shard_keys,values=self._shard(chrom)
        ix=np.searchsorted(shard_keys,keys)
        found=ix<len(shard_keys)
        found[found] &= shard_keys[ix[found]]==keys[found]
        out=np.empty((len(keys),18),dtype=np.float32)
        out[found]=values[ix[found]]
        if not found.all():
            unique, inverse=np.unique(pos[~found],return_inverse=True)
            fresh=encode_annotation_core(engine.compute_raw_annotations(chrom,unique),contract)
            out[~found]=fresh[inverse]
        return out

    def lookup_full(self, locus_keys):
        """Return annotation, breadth and packed regulatory bits in caller order."""
        query=np.asarray(locus_keys,dtype=np.uint64)
        annotation=self.lookup(query)
        breadth=np.empty((len(query),5),dtype=np.float32)
        packed=np.empty((len(query),(4165+7)//8),dtype=np.uint8)
        for chrom in sorted({decode_locus_key(int(k))[0] for k in query}):
            mask=np.array([decode_locus_key(int(k))[0]==chrom for k in query])
            keys,_=self._shard(chrom)
            target=query[mask];ix=np.searchsorted(keys,target)
            valid=ix<len(keys)
            valid[valid] &= keys[ix[valid]]==target[valid]
            if not valid.all():
                raise KeyError(f'{chrom}: missing {int((~valid).sum())} loci')
            path=self.root/'features'/chrom
            manifest=json.loads((path/'manifest.json').read_text())
            if manifest.get('regulatory_status')!='complete' or manifest.get('regulatory_tracks')!=4165:
                raise ValueError(f'incomplete regulatory shard {chrom}')
            source=json.loads((self.root/'regulatory/manifest.json').read_text())
            if manifest.get('regulatory_track_contract_sha256')!=source.get('track_contract_sha256') or manifest.get('regulatory_packbits_bitorder')!='big':
                raise ValueError(f'regulatory contract/bit order differs on {chrom}')
            breadth_mem=np.load(path/'breadth.f32.npy',mmap_mode='r')
            packed_mem=np.load(path/'regulatory.packbits.npy',mmap_mode='r')
            if breadth_mem.shape!=(len(keys),5) or packed_mem.shape!=(len(keys),(4165+7)//8):
                raise ValueError(f'regulatory axis differs from locus axis on {chrom}')
            breadth[mask]=breadth_mem[ix]
            packed[mask]=packed_mem[ix]
        return annotation,breadth,packed
