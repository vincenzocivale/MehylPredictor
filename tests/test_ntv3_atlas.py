import numpy as np
import torch
from types import SimpleNamespace

from methylation_predictor.ntv3_atlas import (
    _base_to_output,
    _build_fast_char_token_lut,
    _centre_embedding,
    _fast_tokenize_sequences,
    prepare_missing_universe_from_positions,
)


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


def test_prepare_missing_universe_from_positions_dedups_by_coordinate(tmp_path):
    import h5py

    base = tmp_path / "base.h5"
    with h5py.File(base, "w") as h:
        h.create_dataset("cpg_idx", data=np.array([10, 20, 30], dtype=np.int64))
        h.create_dataset("chrom_code", data=np.array([1, 1, 2], dtype=np.uint8))
        h.create_dataset("position", data=np.array([100, 200, 300], dtype=np.int64))

    # position 200 on chr1 is already covered (by cpg_idx=20); the rest are new.
    chrom = np.array(["chr1", "chr1", "chr1", "chr3"], dtype=object)
    position = np.array([200, 400, 400, 500], dtype=np.int64)  # includes an exact duplicate query row

    out = tmp_path / "universe.h5"
    result = prepare_missing_universe_from_positions(chrom, position, base, out, shard_size=2)

    assert result["status"] == "built"
    assert result["already_covered"] == 1  # the chr1:200 row
    assert result["missing_cpg"] == 2  # chr1:400 (deduped from the two query rows) and chr3:500

    with h5py.File(out, "r") as h:
        ids = np.asarray(h["cpg_idx"][...])
        chrom_code = np.asarray(h["chrom_code"][...])
        pos = np.asarray(h["position"][...])
    assert len(ids) == 2
    assert ids.min() > 30  # new ids minted above the base atlas max
    assert len(set(ids.tolist())) == 2
    assert sorted(zip(chrom_code.tolist(), pos.tolist())) == [(1, 400), (3, 500)]


def test_prepare_missing_universe_from_positions_rejects_unsupported_chromosome(tmp_path):
    import h5py

    base = tmp_path / "base.h5"
    with h5py.File(base, "w") as h:
        h.create_dataset("cpg_idx", data=np.array([1], dtype=np.int64))
        h.create_dataset("chrom_code", data=np.array([1], dtype=np.uint8))
        h.create_dataset("position", data=np.array([1], dtype=np.int64))

    chrom = np.array(["chrX"], dtype=object)
    position = np.array([1], dtype=np.int64)
    try:
        prepare_missing_universe_from_positions(chrom, position, base, tmp_path / "out.h5")
        assert False, "expected ValueError for unsupported chromosome"
    except ValueError as exc:
        assert "chrX" in str(exc)
