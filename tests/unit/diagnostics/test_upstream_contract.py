from methylation_predictor.diagnostics.methylprophet.upstream import assert_clean, upstream_root


def test_upstream_is_pinned_and_clean() -> None:
    assert (upstream_root() / "src" / "eval.py").is_file()
    assert assert_clean() == "b24f5af3c7b4d6aa2689950e2ea4e3b2bcc8ddfd"
