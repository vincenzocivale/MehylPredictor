"""Regression test for a real bug found 2026-08-28: cpg_statistics/{trainer,evaluator,export}.py
all read the NTv3 atlas embedding matrix via ``h["embeddings"]`` (plural), but the actual
canonical bundle's ``cpg/ntv3/ntv3_cpg_atlas_v1.h5`` stores it under ``embedding`` (singular).
This was never caught because no test exercised the HDF5-reading code path against a real
(or even schema-accurate synthetic) atlas file -- fixed to ``h["embedding"]`` in all three
modules. This test opens the real atlas file directly to pin the key name going forward.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from methylation_predictor.storage import SortedIndex, read_h5_rows


def test_real_ntv3_atlas_uses_singular_embedding_key(bundle_root):
    atlas_path = Path(bundle_root) / "cpg" / "ntv3" / "ntv3_cpg_atlas_v1.h5"
    if not atlas_path.is_file():
        pytest.skip(f"NTv3 atlas not available at {atlas_path}")
    with h5py.File(atlas_path, "r") as h:
        assert "embedding" in h, f"expected key 'embedding' in {atlas_path}; got {list(h.keys())}"
        assert "embeddings" not in h, "atlas schema changed to plural 'embeddings' -- revert the singular-key fix"
        atlas_ids = np.asarray(h["cpg_idx"][...], np.int64)
        sample_ids = atlas_ids[:8]
        rows = SortedIndex(atlas_ids, "NTv3 atlas").positions_of(sample_ids)
        emb = read_h5_rows(h["embedding"], rows, dtype=np.float32)
    assert emb.shape == (8, 1536)
