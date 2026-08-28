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
