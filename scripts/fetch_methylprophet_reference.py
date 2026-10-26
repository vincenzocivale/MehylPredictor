"""Download immutable published comparison files with configured HF access."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/alphagenome_mixed_2026-10-18'
DATASET='MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2'
DATASET_REVISION='c3a4e7e7f5cbfd1373aadc155d6bef7e1e6dcd9e'
MODEL='MethylProphet/tcga_mix_chr1-bs_512-c2b2'
MODEL_REVISION='e3bf682757b37e681c292221c5aaf5456e9949f5'
CONFIG_SHA='f0b74f0c9566591d4e8fffd648ecdfd57e080a559c0af2af8844107b3722596d'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    out=ap.parse_args().out/'reference'
    files=[f'eval/log_dict-{view}.parquet.json' for view in ['train_cpg-val_sample','val_cpg-train_sample','val_cpg-val_sample']]
    files += [f'eval_results-test.parquet/{k:06d}.parquet' for k in range(6)]
    files += ['eval_save_idx-test.json','group_idx_name_mapping-test.json','test_dataloader.json']
    record=dict(repo=DATASET,revision=DATASET_REVISION,files=[])
    for name in files:
        p=Path(hf_hub_download(DATASET,name,repo_type='dataset',revision=DATASET_REVISION,local_dir=out/'released_predictions'))
        record['files'].append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        print(json.dumps(dict(downloaded=name)),flush=True)
    (out/'released_predictions/provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    p=Path(hf_hub_download(MODEL,'version_0/config.yaml',revision=MODEL_REVISION,local_dir=out/'model'))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==CONFIG_SHA
    (out/'model_provenance.json').write_text(json.dumps(dict(repo=MODEL,revision=MODEL_REVISION,
        file='version_0/config.yaml',configuration_sha256=CONFIG_SHA),indent=2)+'\n')


if __name__=='__main__':
    main()
