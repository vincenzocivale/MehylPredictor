from methylation_predictor.rna_training.schedule import SourceSchedule


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
