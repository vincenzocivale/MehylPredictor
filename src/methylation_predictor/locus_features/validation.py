"""Full frozen chr1 regression gate for coordinate-generated annotations."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .annotations import ReferenceAnnotationEngine, encode_annotation_core


def validate_chr1(engine: ReferenceAnnotationEngine, contract: str | Path, reference_cache: str | Path, report: str | Path, *, batch_size: int = 200_000) -> dict:
    ref = Path(reference_cache)
    feature_names = [f['name'] for f in json.loads(Path(contract).read_text())['features']]
    ids = np.load(ref / 'cpg_idx.npy', mmap_mode='r')
    expected = np.load(ref / 'annotation_core.f32.npy', mmap_mode='r')
    if expected.shape != (len(ids), 18):
        raise ValueError('frozen reference shape invalid')
    stats = [{'index': j, 'name': feature_names[j], 'exact_matches': 0, 'mismatches': 0, 'max_abs_error': 0., 'sum_abs_error': 0., 'finite_pattern_matches': 0, 'expected_unique': set(), 'generated_unique': set(), 'dtype': 'float32', 'examples': []} for j in range(18)]
    count = 0
    axis_ok = True
    for frame in pd.read_csv(ref / 'aligned_annotations.tsv.gz', sep='\t', usecols=['cpg_idx', 'chrom', 'position'], chunksize=batch_size):
        n = len(frame)
        if not (frame.chrom == 'chr1').all():
            raise ValueError('non-chr1 coordinate in frozen table')
        axis_ok &= np.array_equal(frame.cpg_idx.to_numpy(), ids[count:count+n])
        new = encode_annotation_core(engine.compute_raw_annotations('chr1', frame.position.to_numpy()), contract)
        old = np.asarray(expected[count:count+n])
        for j in range(18):
            a,b = new[:,j],old[:,j]
            finite = np.isfinite(a) == np.isfinite(b)
            exact = a == b
            delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
            bad = ~finite | (delta > (1e-6 if j == 17 else 0.))
            s = stats[j]
            s['exact_matches'] += int(exact.sum())
            s['mismatches'] += int(bad.sum())
            s['finite_pattern_matches'] += int(finite.sum())
            s['max_abs_error'] = max(s['max_abs_error'],float(np.nanmax(delta)))
            s['sum_abs_error'] += float(np.nansum(delta))
            if j < 17:
                s['expected_unique'].update(np.unique(b).tolist())
                s['generated_unique'].update(np.unique(a).tolist())
            if len(s['examples']) < 5:
                for k in np.flatnonzero(bad)[:5-len(s['examples'])]:
                    s['examples'].append({'cpg_idx':int(ids[count+k]),'position':int(frame.position.iloc[k]),'expected':float(b[k]),'actual':float(a[k])})
        count += n
    output = {'status':'complete' if axis_ok and count == len(ids) and all(s['mismatches']==0 for s in stats) else 'failed', 'loci':count,'cpg_idx_axis_exact':bool(axis_ok and count==len(ids)),'tss_tolerance_atol':1e-6,'features':[]}
    for s in stats:
        s['mean_abs_error'] = s.pop('sum_abs_error')/count
        s['expected_unique'] = sorted(s['expected_unique'])
        s['generated_unique'] = sorted(s['generated_unique'])
        output['features'].append(s)
    Path(report).write_text(json.dumps(output,indent=2)+'\n')
    return output
