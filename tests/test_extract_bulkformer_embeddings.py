from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "rna_branch" / "extract_bulkformer_embeddings.py"
    spec = importlib.util.spec_from_file_location("extract_bulkformer_embeddings", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gene_id_normalization_prefers_versionless_ensembl() -> None:
    module = _load_module()
    assert module._normalise_gene_id("TP53;ENSG00000141510.18") == "ENSG00000141510"
    assert module._normalise_gene_id("ENSG00000141510.18") == "ENSG00000141510"


def test_log2p1_to_ln1p_is_exact_change_of_base() -> None:
    module = _load_module()
    raw_tpm = np.asarray([[0.0, 1.0, 9.0, 99.0]], dtype=np.float32)
    log2p1 = np.log2(raw_tpm + 1.0)
    converted = module._to_ln1p(log2p1, "log2p1_tpm")
    np.testing.assert_allclose(converted, np.log1p(raw_tpm), rtol=1e-6, atol=1e-6)


def test_negative_log_expression_is_rejected() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="negative"):
        module._to_ln1p(np.asarray([[-0.1]], dtype=np.float32), "log2p1_tpm")
