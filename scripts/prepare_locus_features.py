#!/usr/bin/env python3
"""Build and validate the reusable hg38 locus annotation store."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import hashlib
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from methylation_predictor.locus_features.annotations import ReferenceAnnotationEngine
from methylation_predictor.locus_features.catalog import build_catalog
from methylation_predictor.locus_features.store import build_shard
from methylation_predictor.locus_features.validation import validate_chr1
from methylation_predictor.locus_features.views import make_view
from methylation_predictor.locus_features.regulatory import prepare_sources, validate_regulatory_chr1, build_regulatory_shard


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--contract',default=str(Path(__file__).resolve().parents[1]/'resources/locus_features/annotation_core_v1.json'))
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('build-catalog');c.add_argument('--canonical-root',required=True);c.add_argument('--output',required=True);c.add_argument('--replace',action='store_true')
    v=sub.add_parser('validate-annotations-chr1');v.add_argument('--store',required=True);v.add_argument('--reference-cache',required=True);v.add_argument('--sources',required=True);v.add_argument('--fai',required=True)
    b=sub.add_parser('build-annotations');b.add_argument('--store',required=True);b.add_argument('--chromosomes',nargs='+',required=True);b.add_argument('--sources',required=True);b.add_argument('--fai',required=True);b.add_argument('--replace',action='store_true')
    w=sub.add_parser('make-view');w.add_argument('--store',required=True);w.add_argument('--dataset',required=True);w.add_argument('--scope',required=True);w.add_argument('--replace',action='store_true')
    rs=sub.add_parser('prepare-regulatory-sources');rs.add_argument('--store',required=True);rs.add_argument('--reference-atlas',required=True);rs.add_argument('--workers',type=int,default=24)
    rv=sub.add_parser('validate-regulatory-chr1');rv.add_argument('--store',required=True);rv.add_argument('--reference-atlas',required=True)
    rb=sub.add_parser('build-regulatory');rb.add_argument('--store',required=True);rb.add_argument('--chromosomes',nargs='+',required=True);rb.add_argument('--workers',type=int,default=1);rb.add_argument('--replace',action='store_true')
    args=p.parse_args()
    if args.command=='build-catalog':
        print(build_catalog(args.canonical_root,args.output,replace=args.replace))
    elif args.command=='validate-annotations-chr1':
        engine=ReferenceAnnotationEngine(args.sources,args.fai)
        target=Path(args.store)/'validation_chr1.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        result=validate_chr1(engine,args.contract,args.reference_cache,target)
        print(json.dumps({'status':result['status'],'loci':result['loci']}))
        if result['status']!='complete':
            raise SystemExit(1)
    elif args.command=='build-annotations':
        engine=ReferenceAnnotationEngine(args.sources,args.fai)
        chroms=sorted(x.stem for x in (Path(args.store)/'catalog').glob('chr*.parquet')) if args.chromosomes==['all'] else args.chromosomes
        for chrom in chroms:
            print(build_shard(args.store,chrom,engine,args.contract,replace=args.replace))
        if args.chromosomes==['all']:
            root=Path(args.store)
            manifests=[json.loads((root/'features'/c/'manifest.json').read_text()) for c in chroms]
            if any(m.get('status')!='complete' for m in manifests):
                raise ValueError('not all chromosome shards are complete')
            payload={'schema_version':1,'reference_genome':'hg38','annotation_schema':'annotation_core_v1',
                     'annotation_schema_sha256':hashlib.sha256(Path(args.contract).read_bytes()).hexdigest(),
                     'chromosomes':chroms,'number_of_loci':sum(m['number_of_loci'] for m in manifests),
                     'source_versions':manifests[0].get('source_versions'),
                     'source_hashes':manifests[0].get('source_hashes'),
                     'git_commit':manifests[0].get('git_commit'),
                     'build_command':sys.argv,'coordinate_convention':'1-based C; BED [position-1,position)',
                     'built_at':datetime.now(timezone.utc).isoformat(),'status':'complete',
                     'chr1_regression':str(root/'validation_chr1.json')}
            target=root/'manifest.json';temp=root/'.manifest.json.tmp'
            temp.write_text(json.dumps(payload,indent=2)+'\n');os.replace(temp,target)
    elif args.command=='make-view':
        print(make_view(args.store,args.dataset,args.scope,replace=args.replace))
    elif args.command=='prepare-regulatory-sources':
        result=prepare_sources(args.store,args.reference_atlas,workers=args.workers)
        print(json.dumps({k:result[k] for k in ('status','verified','reused','downloaded')}))
    elif args.command=='validate-regulatory-chr1':
        result=validate_regulatory_chr1(args.store,args.reference_atlas)
        print(json.dumps({k:result[k] for k in ('status','rows','tracks','mismatched_bits','expected_nnz','observed_nnz')}))
        if result['status']!='complete':raise SystemExit(1)
    elif args.command=='build-regulatory':
        chroms=sorted(x.stem for x in (Path(args.store)/'catalog').glob('chr*.parquet')) if args.chromosomes==['all'] else args.chromosomes
        if args.workers==1:
            for chrom in chroms:
                print(json.dumps(build_regulatory_shard(args.store,chrom,replace=args.replace)),flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(build_regulatory_shard,args.store,chrom,replace=args.replace):chrom for chrom in chroms}
                for future in as_completed(futures):
                    print(json.dumps(future.result()),flush=True)
        if args.chromosomes==['all']:
            root=Path(args.store);manifest=json.loads((root/'manifest.json').read_text())
            source_manifest=json.loads((root/'regulatory/manifest.json').read_text())
            expected={'regulatory_status':'complete','regulatory_schema':'regulatory_peak_overlap_v1','regulatory_tracks':4165,'breadth_dimension':5,'regulatory_chr1_validation':'validation_regulatory_chr1.json','regulatory_track_contract_sha256':source_manifest['track_contract_sha256'],'regulatory_verified_source_files':source_manifest['verified']}
            if any(manifest.get(k)!=v for k,v in expected.items()):
                manifest.update(expected)
                temp=root/'.manifest.json.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n');os.replace(temp,root/'manifest.json')

if __name__=='__main__':main()
