import os
from pathlib import Path
import numpy as np
import pytest
from methylation_predictor.locus_features.regulatory import point_overlap,_point_overlap_sorted,unpack_embedding_bags,breadth_from_counts,load_track_contract,core_mask


def test_half_open_point_overlap():
    intervals=np.array([[9,11],[20,22]],dtype=np.int64)
    # positions are 1-based C; BED points are 9,10,11,19,20,21,22
    out=point_overlap(np.array([10,11,12,20,21,22,23]),intervals)
    assert out.tolist()==[True,True,False,False,True,True,False]
    np.testing.assert_array_equal(_point_overlap_sorted(np.array([10,11,12,20,21,22,23]),intervals),out)


def test_packbits_embedding_bag_order_and_repeats():
    bits=np.zeros((3,4165),np.uint8)
    bits[0,[0,8,4164]]=1
    bits[2,[1,4163]]=1
    packed=np.packbits(bits,axis=1,bitorder='big')
    indices,offsets=unpack_embedding_bags(packed[[2,0,2]])
    assert indices.tolist()==[1,4163,0,8,4164,1,4163]
    assert offsets.tolist()==[0,2,5,7]


def test_breadth_denominators_are_frozen():
    counts=np.array([[533,1959,201,1472],[0,0,0,0]],dtype=np.uint16)
    core=np.array([2401,0],dtype=np.uint16)
    out=breadth_from_counts(counts,core)
    assert out.dtype==np.float32
    np.testing.assert_array_equal(out[0],np.ones(5,np.float32))
    np.testing.assert_array_equal(out[1],np.zeros(5,np.float32))


def test_frozen_track_contract_when_data_available():
    root=os.environ.get('METHYL_DATA_ROOT')
    if not root:
        pytest.skip('METHYL_DATA_ROOT not set')
    path=Path(root)/'derived/ntv3_functional_peak_atlas_chr1_all_sources/all_primary_columns.tsv'
    if not path.exists():
        pytest.skip('frozen chr1 atlas unavailable')
    frame=load_track_contract(path)
    assert int(core_mask(frame).sum())==2401
    assert frame.track_index.is_monotonic_increasing
