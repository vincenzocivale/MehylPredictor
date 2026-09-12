from __future__ import annotations

from pathlib import Path

import pytest

from methylation_predictor.artifact_uri import from_uri, is_uri, to_uri


def test_round_trip(tmp_path):
    root = tmp_path / "data_root"
    target = root / "experiments" / "runs" / "rna_methylation" / "chr1" / "main-seed17"
    target.mkdir(parents=True)

    uri = to_uri(target, data_root=root)
    assert uri == "methyl-data://experiments/runs/rna_methylation/chr1/main-seed17"
    assert is_uri(uri)

    resolved = from_uri(uri, data_root=root)
    assert resolved == target.resolve()


def test_to_uri_rejects_path_outside_root(tmp_path):
    root = tmp_path / "data_root"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "file.pt"
    outside.parent.mkdir()

    with pytest.raises(ValueError):
        to_uri(outside, data_root=root)


def test_from_uri_rejects_malformed_scheme(tmp_path):
    root = tmp_path / "data_root"
    root.mkdir()
    with pytest.raises(ValueError):
        from_uri("s3://bucket/key", data_root=root)
    with pytest.raises(ValueError):
        from_uri("methyl-data:/missing-slash", data_root=root)


def test_from_uri_rejects_path_traversal(tmp_path):
    root = tmp_path / "data_root"
    root.mkdir()
    with pytest.raises(ValueError):
        from_uri("methyl-data://../escape", data_root=root)
    with pytest.raises(ValueError):
        from_uri("methyl-data://a/../../escape", data_root=root)


def test_from_uri_rejects_absolute_relative_part(tmp_path):
    root = tmp_path / "data_root"
    root.mkdir()
    with pytest.raises(ValueError):
        from_uri("methyl-data:///abs/path", data_root=root)


def test_from_uri_normalizes_dot_segments(tmp_path):
    root = tmp_path / "data_root"
    root.mkdir()
    resolved = from_uri("methyl-data://a/./b/c", data_root=root)
    assert resolved == (root / "a" / "b" / "c").resolve()


def test_no_dependency_on_machine_specific_mount(monkeypatch, tmp_path):
    root_a = tmp_path / "mount_a"
    root_b = tmp_path / "mount_b"
    (root_a / "experiments").mkdir(parents=True)
    (root_b / "experiments").mkdir(parents=True)

    uri = to_uri(root_a / "experiments", data_root=root_a)
    resolved_on_b = from_uri(uri, data_root=root_b)
    assert resolved_on_b == (root_b / "experiments").resolve()


def test_data_root_required_without_env(monkeypatch):
    monkeypatch.delenv("METHYL_DATA_ROOT", raising=False)
    with pytest.raises(ValueError):
        to_uri("/tmp/whatever")


def test_is_uri_rejects_non_string_and_other_schemes():
    assert not is_uri(123)  # type: ignore[arg-type]
    assert not is_uri("/absolute/path")
    assert not is_uri("s3://bucket/key")
