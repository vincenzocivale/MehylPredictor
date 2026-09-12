from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import h5py
import numpy as np
import pytest

from methylation_predictor.rna_training.rna_methylation_trainer import (
    ordered_bounded_prefetch,
)
from methylation_predictor.rna_training.matched_chr1_data import load_compact_scope_sources


def _digest(values):
    return hashlib.sha256(np.asarray(values, np.int64).tobytes()).hexdigest()


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compact_fixture(tmp_path):
    samples = np.array([11, 12, 13], dtype=np.int64)
    cpgs = np.array([101, 103, 107, 109], dtype=np.int64)
    path = tmp_path / "array.h5"
    beta = np.arange(12, dtype=np.float32).reshape(3, 4) / 12
    beta[1, 2] = np.nan
    with h5py.File(path, "w") as handle:
        handle.create_dataset("beta", data=beta, chunks=(1, 4))
        handle.create_dataset("sample_idx", data=samples)
        handle.create_dataset("cpg_idx", data=cpgs)
    manifest = {
        "schema_version": 1,
        "scope": "chr123",
        "sources": {
            "array": {
                "bytes": path.stat().st_size,
                "file_sha256": _file_digest(path),
                "samples_sha256": _digest(samples),
                "cpgs_sha256": _digest(cpgs),
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    protocol = SimpleNamespace(
        sources=("array",), array_train_sample_idx=samples[:2], array_val_sample_idx=samples[2:],
        array_train_cpg_idx=cpgs[:3], array_val_cpg_idx=cpgs[3:], auxiliary_cpg_idx={},
    )
    return path, protocol, beta


def test_compact_loader_validates_manifest_and_preserves_values(tmp_path):
    _, protocol, beta = _compact_fixture(tmp_path)
    sources = load_compact_scope_sources(tmp_path, protocol, hdf5_cache_mb=1)
    try:
        np.testing.assert_allclose(
            sources["array"].block(np.arange(3), np.array([109, 101, 107])),
            beta[:, [3, 0, 2]], equal_nan=True,
        )
    finally:
        for source in sources.values():
            source.close()


def test_compact_loader_rejects_content_hash_mismatch(tmp_path):
    path, protocol, _ = _compact_fixture(tmp_path)
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="size does not match manifest"):
        load_compact_scope_sources(tmp_path, protocol, hdf5_cache_mb=1)


def test_ordered_bounded_prefetch_preserves_order_and_worker_bound():
    lock = threading.Lock()
    active = 0
    maximum = 0

    def work(value):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.005 * (4 - value % 4))
        with lock:
            active -= 1
        return value * 2

    with ThreadPoolExecutor(max_workers=3) as executor:
        output = [result for result, _ in ordered_bounded_prefetch(executor, work, range(12), depth=5)]
    assert output == [value * 2 for value in range(12)]
    assert maximum <= 3


def test_ordered_bounded_prefetch_propagates_worker_errors():
    def work(value):
        if value == 2:
            raise RuntimeError("worker failed")
        return value

    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(RuntimeError, match="worker failed"):
            list(ordered_bounded_prefetch(executor, work, range(5), depth=3))
