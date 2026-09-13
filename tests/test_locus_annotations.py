import json
from pathlib import Path
import numpy as np
import pytest

from methylation_predictor.locus_features.annotations import RawAnnotations, encode_annotation_core
from methylation_predictor.locus_features.ids import locus_key
from methylation_predictor.locus_features.store import LocusFeatureStore

CONTRACT=Path(__file__).resolve().parents[1]/'resources/locus_features/annotation_core_v1.json'


def test_encoding_is_exclusive_and_frozen():
    raw=RawAnnotations(np.array(['island','open_sea']),np.array(['exon','intergenic']),np.array(['PLS','none']),np.array([2.,2.],np.float32))
    out=encode_annotation_core(raw,CONTRACT)
    assert out.shape==(2,18) and out.dtype==np.float32
    assert np.all(out[:,:4].sum(axis=1)==1)
    assert np.all(out[:,4:8].sum(axis=1)==1)
    assert np.all(out[:,8:17].sum(axis=1)==1)
    assert out[0,17]==out[1,17]
    norm=json.loads(CONTRACT.read_text())['features'][17]['normalization']
    assert norm['mean']==3.2289013469204666
    assert norm['std']==0.9242045243544466


def test_cross_technology_identity():
    loc=('chr1',100034965)
    assert len({locus_key(*loc) for _ in ('array','epic','wgbs','encode')})==1


def test_store_mmap_and_missing(tmp_path):
    shard=tmp_path/'features'/'chr1';shard.mkdir(parents=True)
    key=locus_key('chr1',100)
    np.save(shard/'locus_key.npy',np.array([key],np.uint64))
    np.save(shard/'annotation_core.f32.npy',np.zeros((1,18),np.float32))
    import hashlib
    (shard/'manifest.json').write_text(json.dumps({'status':'complete','feature_dimension':18,'annotation_schema_sha256':hashlib.sha256(CONTRACT.read_bytes()).hexdigest()}))
    store=LocusFeatureStore(tmp_path)
    assert store.lookup([key]).shape==(1,18)
    assert isinstance(store._shard('chr1')[1],np.memmap)
    with pytest.raises(KeyError):store.lookup([locus_key('chr1',101)])


def test_arbitrary_reference_coordinate_when_data_available():
    import os
    from methylation_predictor.locus_features.annotations import ReferenceAnnotationEngine

    root=os.environ.get('METHYL_DATA_ROOT')
    if not root:
        pytest.skip('METHYL_DATA_ROOT is not set')
    source=Path(root)/'derived/ntv3_probe_targets/sources'
    fai=Path(root)/'reference/hg38/hg38.fa.fai'
    if not source.exists() or not fai.exists():
        pytest.skip('hg38 annotation sources unavailable')
    engine=ReferenceAnnotationEngine(source,fai)
    raw=engine.compute_raw_annotations('chr1',np.array([100034965]))
    assert encode_annotation_core(raw,CONTRACT).shape==(1,18)
    with pytest.raises(ValueError,match='not a reference-forward CpG'):
        engine.compute_raw_annotations('chr1',np.array([100034966]))
