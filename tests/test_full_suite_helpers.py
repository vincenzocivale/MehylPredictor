import numpy as np

from types import SimpleNamespace

import torch

from methylation_predictor.full_suite.expansion import (
    _base_to_output,
    _build_fast_char_token_lut,
    _centre_embedding,
    _fast_tokenize_sequences,
)
from methylation_predictor.full_suite.feature_store import SortedIndex
from methylation_predictor.full_suite.probe import _stable_validation_mask


def test_sorted_index_preserves_query_order():
    idx = SortedIndex(np.array([50, 10, 30], dtype=np.int64))
    assert idx.positions_of(np.array([10, 50, 30])).tolist() == [1, 0, 2]
    assert idx.contains(np.array([10, 11, 30])).tolist() == [True, False, True]


def test_probe_split_is_deterministic_and_nontrivial():
    ids = np.arange(10000, dtype=np.int64)
    a = _stable_validation_mask(ids, 0.1, 17)
    b = _stable_validation_mask(ids, 0.1, 17)
    assert np.array_equal(a, b)
    assert 500 < int(a.sum()) < 1500


def test_probe_ensemble_shape():
    import torch
    from methylation_predictor.full_suite.probe import ProbeEnsemble

    model = ProbeEnsemble([17, 29, 43], dim=8, dropout=0.0)
    x = torch.randn(5, 8)
    assert model.forward_all(x).shape == (3, 5, 3)
    assert model(x).shape == (5, 3)


class _FakeCharTokenizer:
    mapping = {"A": 11, "C": 12, "G": 13, "T": 14, "N": 15}

    def __call__(self, value, *, add_special_tokens=False, **kwargs):
        assert add_special_tokens is False
        if isinstance(value, str):
            return {"input_ids": [self.mapping[x] for x in value]}
        raise TypeError("test tokenizer only supports scalar strings")


def test_fast_ntv3_character_tokenizer_matches_reference_mapping():
    tokenizer = _FakeCharTokenizer()
    lut = _build_fast_char_token_lut(tokenizer)
    sequences = ["ACGTNACG", "NNNNACGT"]
    observed = _fast_tokenize_sequences(sequences, lut).numpy()
    expected = np.asarray(
        [[tokenizer.mapping[x] for x in sequence] for sequence in sequences],
        dtype=np.int64,
    )
    assert np.array_equal(observed, expected)


def test_centre_embedding_select_before_fp32_matches_historical_order():
    length = 32768
    embedding = torch.randn(2, 128, 16, dtype=torch.bfloat16)
    output = SimpleNamespace(embedding=embedding)
    c, g = length // 2 - 1, length // 2
    ci = _base_to_output(c, length, embedding.shape[1])
    gi = _base_to_output(g, length, embedding.shape[1])
    historical = embedding.float()[:, [ci, gi]].mean(dim=1)
    optimized = _centre_embedding(output, length)
    assert torch.equal(optimized, historical)
