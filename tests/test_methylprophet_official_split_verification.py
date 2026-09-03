"""Optional, opt-in cross-check of this repo's chr1 Array split against the actual
released MethylProphet evaluation artifact (not just a reconstruction).

Skipped by default -- requires downloading a ~1.3GB gated HuggingFace dataset
(``MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2``) that isn't part of the canonical
bundle. Point ``MP_EVAL_DIR`` at a local copy to run it. See
``results/reference/methylprophet_comparison/chr1_official_split_verification.md``
for the recorded 2026-08-28 result (exact match on all four axes) and the download
command.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from methylation_predictor.tcga_canonical import load_protocol

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PREPARE_PY = _REPO_ROOT / "scripts" / "benchmark_methylprophet" / "prepare.py"


def _load_array_ids_from_mp_eval():
    spec = importlib.util.spec_from_file_location("_mp_prepare", _PREPARE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._array_ids_from_mp_eval


@pytest.fixture(scope="module")
def mp_eval_dir():
    raw = os.environ.get("MP_EVAL_DIR")
    if not raw:
        pytest.skip("MP_EVAL_DIR not set -- see chr1_official_split_verification.md to download it")
    path = Path(raw)
    if not path.is_dir():
        pytest.skip(f"MP_EVAL_DIR does not exist: {path}")
    return path


def test_chr1_split_matches_released_evaluation_artifact_exactly(bundle, bundle_root, mp_eval_dir):
    array_ids_from_mp_eval = _load_array_ids_from_mp_eval()
    train_s, val_s, train_c, val_c, _provenance = array_ids_from_mp_eval(mp_eval_dir)

    protocol = load_protocol("tcga_array_chr1", bundle=bundle, root=bundle_root)

    def same_set(ours, official):
        return set(np.asarray(ours).tolist()) == set(np.asarray(official).tolist())

    assert same_set(protocol.array_train_sample_idx, train_s)
    assert same_set(protocol.array_val_sample_idx, val_s)
    assert same_set(protocol.array_train_cpg_idx, train_c)
    assert same_set(protocol.array_val_cpg_idx, val_c)


@pytest.fixture(scope="module")
def mp_eval_dir_chr123():
    raw = os.environ.get("MP_EVAL_DIR_CHR123")
    if not raw:
        pytest.skip("MP_EVAL_DIR_CHR123 not set -- see chr1_official_split_verification.md to download it")
    path = Path(raw)
    if not path.is_dir():
        pytest.skip(f"MP_EVAL_DIR_CHR123 does not exist: {path}")
    return path


def test_chr123_cpg_axis_matches_released_evaluation_artifact_exactly(bundle, bundle_root, mp_eval_dir_chr123):
    """Verified 2026-09-02: the released chr123 evaluation artifact's CpG axis matches this
    repo's note1-union-note4 reconstruction exactly.

    The sample axis deliberately is NOT asserted here: it does not match either chr1's reused
    split OR our own canonical Array bundle at all (306 of the release's sample_idx values don't
    exist in our 241231-snapshot Array HDF5, and vice versa -- a real content difference, not an
    indexing bug, since the CpG axis extracted from the same rows via the same code path matches
    exactly). protocols/tcga_mix_chr123 keeps the chr1-reused sample split as the only one
    internally consistent with our own data. See chr1_official_split_verification.md's chr123
    section for the full record."""
    array_ids_from_mp_eval = _load_array_ids_from_mp_eval()
    train_s, val_s, train_c, val_c, _provenance = array_ids_from_mp_eval(
        mp_eval_dir_chr123,
        expected_counts={
            "array_train_samples": 8258,
            "array_val_samples": 920,
            "array_train_cpgs": 78211,
            "array_val_cpgs": 14893,
        },
    )

    protocol = load_protocol("tcga_mix_chr123", bundle=bundle, root=bundle_root)

    def same_set(ours, official):
        return set(np.asarray(ours).tolist()) == set(np.asarray(official).tolist())

    assert same_set(protocol.array_train_cpg_idx, train_c)
    assert same_set(protocol.array_val_cpg_idx, val_c)
