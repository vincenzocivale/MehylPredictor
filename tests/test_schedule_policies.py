from methylation_predictor.rna_training.schedule import SourceSchedule
import numpy as np


def test_pair_complete_enumerates_cartesian_grid():
    s=SourceSchedule(10,11,4,5,1,17,"pair_complete")
    assert len(s)==3*3
    assert s.report()["pair_complete"] is True


def test_axis_full_coverage_is_scalable():
    s=SourceSchedule(100,1000,10,100,1,17,"axis_full_coverage")
    assert len(s)==10
    rows=set(); cols=set()
    for i in range(len(s)):
        r,c=s[i]; rows.update(r.tolist()); cols.update(c.tolist())
    assert len(rows)==100 and len(cols)==1000


def test_contiguous_layout_keeps_locality_and_pair_coverage():
    s=SourceSchedule(10,11,4,5,1,17,"pair_complete","contiguous_blocks")
    seen=np.zeros((10,11),dtype=np.int8)
    for i in range(len(s)):
        rows,cols=s[i]
        assert len(rows)==1 or np.all(np.diff(rows)==1)
        assert len(cols)==1 or np.all(np.diff(cols)==1)
        seen[np.ix_(rows,cols)]+=1
    assert np.all(seen==1)
    assert s.report()["layout"]=="contiguous_blocks"


def test_legacy_layout_remains_element_shuffled_and_deterministic():
    a=SourceSchedule(100,100,10,10,3,17,"pair_complete")
    b=SourceSchedule(100,100,10,10,3,17,"pair_complete","legacy_scattered")
    assert all(np.array_equal(a[i][0],b[i][0]) and np.array_equal(a[i][1],b[i][1]) for i in range(len(a)))
    assert any(not np.all(np.diff(block)==1) for block in a.rows)
