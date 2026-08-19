from methylation_predictor.scopes import resolve_scope


def test_scope_contracts():
    assert resolve_scope("chr1").chromosomes == ("chr1",)
    assert resolve_scope("chr123").chromosomes == ("chr1","chr2","chr3")
    assert resolve_scope("genomewide").chromosomes is None
    assert resolve_scope("chr1").matched_methylprophet
    assert not resolve_scope("genomewide").matched_methylprophet
