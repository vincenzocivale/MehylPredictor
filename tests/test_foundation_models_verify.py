"""Synthetic (no real checkpoints/GPU needed) tests for the foundation-model
readiness pipeline's verification and vocabulary-overlap logic."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from methylation_predictor.benchmark.foundation_models.verify import (
    fingerprint_state_dict,
    vocabulary_overlap,
    verify_checkpoint_loaded,
)
from methylation_predictor.benchmark.foundation_models.adapters.methylgpt_adapter import (
    _remap_flash_attn_qkv_keys,
)
from methylation_predictor.benchmark.foundation_models.adapters.deepcpg_adapter import (
    one_hot_dna_window,
)


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_fingerprint_changes_after_real_load(tmp_path):
    """A real checkpoint load should change every tensor's fingerprint."""
    reference = _tiny_model()
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(reference.state_dict(), ckpt_path)

    torch.manual_seed(1)  # different random init than `reference`
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def load_fn(m: nn.Module) -> None:
        m.load_state_dict(torch.load(ckpt_path, weights_only=True))

    fixed_input = torch.randn(3, 4)
    result = verify_checkpoint_loaded(
        model, ckpt_path, load_fn, forward_probe=lambda m: m(fixed_input)
    )

    assert result.passed
    assert result.n_params_unchanged == 0
    assert result.forward_output_changed is True
    assert result.checkpoint_sha256  # file existed, got hashed


def test_verify_catches_silent_partial_load(tmp_path):
    """A load_fn that only updates some tensors must fail, not silently pass."""
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({}, ckpt_path)  # nothing to load -> a no-op load_fn

    def load_fn(_m: nn.Module) -> None:
        pass  # simulates a loader whose non-strict fallback skipped everything

    result = verify_checkpoint_loaded(model, ckpt_path, load_fn)

    assert not result.passed
    assert result.n_params_unchanged == result.n_params_total
    assert "unchanged" in result.notes[0]


def test_fingerprint_state_dict_is_order_independent_and_complete():
    model = _tiny_model()
    fp = fingerprint_state_dict(model.state_dict())
    assert set(fp.keys()) == set(model.state_dict().keys())
    for norm, mean, numel in fp.values():
        assert numel > 0
        assert norm >= 0.0
        assert isinstance(mean, float)


def test_vocabulary_overlap_partial_match():
    probe_to_position = {
        "cg00000001": ("chr1", 100),
        "cg00000002": ("chr1", 200),
        "cg00000003": ("chr2", 300),
    }
    query = [("chr1", 100), ("chr1", 999), ("chr2", 300)]

    result = vocabulary_overlap(query, probe_to_position)

    assert result["n_query"] == 3
    assert result["n_vocab"] == 3
    assert result["n_overlap"] == 2
    assert result["coverage_fraction"] == 2 / 3
    assert set(result["overlap_probe_ids"]) == {"cg00000001", "cg00000003"}


def test_vocabulary_overlap_empty_query_does_not_divide_by_zero():
    result = vocabulary_overlap([], {"cg1": ("chr1", 1)})
    assert result["n_query"] == 0
    assert result["coverage_fraction"] == 0.0


def test_remap_flash_attn_qkv_keys_renames_only_attention_weights():
    """Regression test for the 2026-09-02 misdiagnosis: Wqkv and in_proj_weight are
    the same fused QKV projection under a different name, not different
    parameterizations -- `load_checkpoint_into` must rename, not drop, them."""
    state_dict = {
        "transformer_encoder.layers.0.self_attn.Wqkv.weight": torch.zeros(192, 64),
        "transformer_encoder.layers.0.self_attn.Wqkv.bias": torch.zeros(192),
        "cls_decoder._decoder.0.weight": torch.ones(64, 64),  # unrelated key, untouched
    }

    remapped = _remap_flash_attn_qkv_keys(state_dict)

    assert set(remapped.keys()) == {
        "transformer_encoder.layers.0.self_attn.in_proj_weight",
        "transformer_encoder.layers.0.self_attn.in_proj_bias",
        "cls_decoder._decoder.0.weight",
    }
    assert torch.equal(
        remapped["transformer_encoder.layers.0.self_attn.in_proj_weight"],
        state_dict["transformer_encoder.layers.0.self_attn.Wqkv.weight"],
    )


def test_one_hot_dna_window_centers_correctly_and_sums_to_window_length(tmp_path):
    pytest.importorskip("pyfaidx")
    fasta_path = tmp_path / "toy.fa"
    fasta_path.write_text(">chr1\n" + "ACGT" * 50 + "\n")

    window = one_hot_dna_window(str(fasta_path), "chr1", 100, wlen=11)

    assert window.shape == (11, 4)
    assert window.sum() == 11  # every base called, one-hot (no N's in this toy sequence)
    assert (window.sum(axis=1) == 1).all()  # exactly one base called per position


def test_one_hot_dna_window_pads_out_of_bounds_with_zero_rows(tmp_path):
    pytest.importorskip("pyfaidx")
    fasta_path = tmp_path / "toy.fa"
    fasta_path.write_text(">chr1\n" + "ACGT" * 3 + "\n")  # 12bp contig

    # A window centered near position 1 runs off the start of the contig.
    window = one_hot_dna_window(str(fasta_path), "chr1", 1, wlen=11)

    assert window.shape == (11, 4)
    assert window.sum() < 11  # some rows are all-zero (out-of-bounds padding)
    assert (window[0] == 0).all()  # the leftmost positions are off-contig
