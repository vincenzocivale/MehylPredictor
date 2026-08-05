from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "rna_branch" / "merge_encoder_embeddings.py"
    spec = importlib.util.spec_from_file_location("merge_encoder_embeddings", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, ids: list[str], datasets: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample_idx", data=np.asarray(ids, dtype=object), dtype=h5py.string_dtype("utf-8"))
        for name, values in datasets.items():
            handle.create_dataset(name, data=np.asarray(values, dtype=np.float32))


def test_merge_aligns_rows_and_concatenates(tmp_path: Path) -> None:
    module = _load_module()
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write(
        first,
        ["a", "b", "c"],
        {
            "layer0": np.asarray([[1], [2], [3]]),
            "layer1": np.asarray([[10, 11], [20, 21], [30, 31]]),
        },
    )
    _write(second, ["c", "a", "b"], {"mean": np.asarray([[300], [100], [200]])})
    output = tmp_path / "merged.h5"
    config = tmp_path / "merge.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "output": str(output),
                "sources": {
                    "bert": {
                        "path": str(first),
                        "datasets": {"concat": {"keys": ["layer0", "layer1"], "transform": "concat"}},
                    },
                    "former": {"path": str(second), "datasets": {"official": "mean"}},
                },
            }
        )
    )

    assert module.merge(config) == output
    with h5py.File(output, "r") as handle:
        assert handle["sample_idx"].asstr()[...].tolist() == ["a", "b", "c"]
        np.testing.assert_allclose(
            handle["bert__concat"][...],
            np.asarray([[1, 10, 11], [2, 20, 21], [3, 30, 31]], dtype=np.float32),
        )
        np.testing.assert_allclose(handle["former__official"][...].ravel(), [100, 200, 300])


def test_merge_rejects_different_sample_sets(tmp_path: Path) -> None:
    module = _load_module()
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write(first, ["a", "b"], {"x": np.zeros((2, 2))})
    _write(second, ["a", "c"], {"x": np.zeros((2, 2))})
    config = tmp_path / "merge.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "output": str(tmp_path / "out.h5"),
                "sources": {
                    "one": {"path": str(first), "datasets": {"x": "x"}},
                    "two": {"path": str(second), "datasets": {"x": "x"}},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="sample sets differ"):
        module.merge(config)
