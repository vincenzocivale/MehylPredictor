from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from methylation_predictor.config import EncoderConfig, InteractionConfig, ModelConfig
from methylation_predictor.models import FeatureFusionArchitectureVariantModel, FunctionalFusionModel
from methylation_predictor.storage import FunctionalLocusCache


def _write_cache(tmp_path, *, mismatch=False):
    atlas = tmp_path / "atlas"
    annotations = tmp_path / "annotations"
    csr = atlas / "all_primary_overlap_csr"
    csr.mkdir(parents=True)
    annotations.mkdir()
    ids = np.array([91, 12, 45, 77], dtype=np.int64)
    np.save(atlas / "cpg_idx.npy", ids)
    np.save(annotations / "cpg_idx.npy", ids + (1 if mismatch else 0))
    core = np.arange(4 * 18, dtype=np.float32).reshape(4, 18)
    breadth = (100 + np.arange(4 * 5, dtype=np.float32)).reshape(4, 5)
    np.save(annotations / "annotation_core.f32.npy", core)
    np.save(atlas / "breadth_features.f32.npy", breadth)
    # rows: [2,7], [], [1], [3,4,8]
    np.save(csr / "indices.npy", np.array([2, 7, 1, 3, 4, 8], dtype=np.int32))
    np.save(csr / "indptr.npy", np.array([0, 2, 2, 3, 6], dtype=np.int64))
    np.save(csr / "data.npy", np.ones(6, dtype=np.uint8))
    (csr / "shape.json").write_text(json.dumps([4, 4165]))
    return atlas, annotations, ids, core, breadth


def test_functional_cache_arbitrary_order_and_empty_bags(tmp_path):
    atlas, annotations, ids, core, breadth = _write_cache(tmp_path)
    cache = FunctionalLocusCache(atlas, annotations)
    result = cache.get(np.array([ids[3], ids[1], ids[0]], dtype=np.int64))
    assert result["track_indices"].tolist() == [3, 4, 8, 2, 7]
    assert result["offsets"].tolist() == [0, 3, 3, 5]
    np.testing.assert_array_equal(result["dense"][:, :18], core[[3, 1, 0]])
    np.testing.assert_array_equal(result["dense"][:, 18:], breadth[[3, 1, 0]])
    bag = torch.nn.EmbeddingBag(4165, 4, mode="mean", include_last_offset=True)
    encoded = bag(torch.from_numpy(result["track_indices"]), torch.from_numpy(result["offsets"]))
    assert torch.equal(encoded[1], torch.zeros_like(encoded[1]))


def test_functional_cache_rejects_axis_mismatch(tmp_path):
    atlas, annotations, *_ = _write_cache(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="do not match exactly"):
        FunctionalLocusCache(atlas, annotations)


def _config():
    return ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=16, n_programs=6, program_dim=8, n_heads=2, dropout=0.0),
        interaction=InteractionConfig(hidden_dim=12, dropout=0.0),
    )


def _models():
    kwargs = dict(trunk_hidden_dim=20, bottleneck_dim=10, use_raw_product=False)
    torch.manual_seed(17)
    control = FeatureFusionArchitectureVariantModel(48, 32, _config(), **kwargs).eval()
    torch.manual_seed(17)
    functional = FeatureFusionArchitectureVariantModel(
        48, 32, _config(), functional_conditioning=True, **kwargs,
    ).eval()
    return control, functional


def _functional_inputs():
    return {
        "functional_track_indices": torch.tensor([1, 2, 5, 9], dtype=torch.int64),
        "functional_offsets": torch.tensor([0, 2, 2, 3, 4], dtype=torch.int64),
        "functional_dense": torch.randn(4, 23),
    }


def test_functional_init_preserves_common_parameters_and_predictions():
    control, functional = _models()
    common = control.state_dict()
    for name, value in common.items():
        assert torch.equal(value, functional.state_dict()[name]), name
    assert torch.count_nonzero(functional.functional_delta.weight) == 0
    rna, loci = torch.randn(3, 48), torch.randn(4, 32)
    with torch.no_grad():
        a = control(rna, loci)["beta"]
        b = functional(rna, loci, **_functional_inputs())["beta"]
    assert torch.equal(a, b)


def test_functional_construction_restores_rng_state():
    kwargs = dict(trunk_hidden_dim=20, bottleneck_dim=10, use_raw_product=False)
    torch.manual_seed(17)
    FeatureFusionArchitectureVariantModel(48, 32, _config(), **kwargs)
    control_state = torch.get_rng_state()
    torch.manual_seed(17)
    FeatureFusionArchitectureVariantModel(48, 32, _config(), functional_conditioning=True, **kwargs)
    assert torch.equal(control_state, torch.get_rng_state())


def test_nonzero_functional_delta_makes_input_effective():
    _, model = _models()
    torch.nn.init.normal_(model.functional_delta.weight, std=0.1)
    rna, loci = torch.randn(3, 48), torch.randn(4, 32)
    inputs = _functional_inputs()
    changed_dense = inputs["functional_dense"].clone()
    changed_dense[:, 0] += 2.0
    changed = {**inputs, "functional_dense": changed_dense}
    with torch.no_grad():
        first = model(rna, loci, **inputs)["beta"]
        second = model(rna, loci, **changed)["beta"]
    assert not torch.allclose(first, second)


def test_functional_only_ignores_genomic_embedding():
    kwargs = dict(trunk_hidden_dim=20, bottleneck_dim=10, use_raw_product=False)
    torch.manual_seed(17)
    model = FeatureFusionArchitectureVariantModel(
        48, 32, _config(), functional_conditioning=True, functional_only=True, **kwargs,
    ).eval()
    rna, loci = torch.randn(3, 48), torch.randn(4, 32)
    inputs = _functional_inputs()
    with torch.no_grad():
        first = model(rna, loci, **inputs)["beta"]
        second = model(rna, loci + 1000.0 * torch.randn_like(loci), **inputs)["beta"]
    assert torch.equal(first, second)


def _ladder_model(variant):
    cfg = ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=8, n_programs=6,
                              program_dim=256, n_heads=4, dropout=0.0),
        functional_fusion_variant=variant,
    )
    return FunctionalFusionModel(48, cfg)


@pytest.mark.parametrize("variant", ["f0_single", "f1_capacity", "f2_iter2", "f3_iter4", "f4_head_gated", "f5_atlas_router"])
def test_functional_fusion_forward_backward_and_no_genomic_dependency(variant):
    model = _ladder_model(variant)
    inputs = _functional_inputs()
    rna = torch.randn(3, 48)
    first = model(rna, torch.randn(4, 1536), **inputs)["beta"]
    second = model(rna, torch.randn(4, 7), **inputs)["beta"]
    assert first.shape == (3, 4)
    assert torch.equal(first, second)
    first.mean().backward()
    assert model.track_embedding.weight.grad is not None


def test_f4_gates_are_exactly_one_at_initialization():
    model = _ladder_model("f4_head_gated").eval()
    with torch.no_grad():
        model(torch.randn(2, 48), None, **_functional_inputs())
    for block in model.blocks:
        assert torch.equal(block.last_gates, torch.ones_like(block.last_gates))


def test_f5_active_track_weights_normalize_and_empty_bag_is_dense_only():
    model = _ladder_model("f5_atlas_router").eval()
    inputs = _functional_inputs()
    with torch.no_grad():
        out = model(torch.randn(2, 48), None, **inputs)["beta"]
    sums = model.last_router_normalization
    assert torch.allclose(sums[:, [0, 2, 3]], torch.ones_like(sums[:, [0, 2, 3]]), atol=1e-6)
    assert torch.equal(sums[:, 1], torch.zeros_like(sums[:, 1]))
    assert torch.isfinite(out).all()


# --- G0-G4 regulatory-fusion ladder -----------------------------------------

from methylation_predictor.models import (  # noqa: E402
    GroupedFunctionalEncoder, RegulatoryHeadCrossAttention,
)


@pytest.mark.parametrize("variant", ["g0_control", "g1_grouped", "g2_regulatory_head",
                                      "g3_gated_residual", "g4_lowrank_fusion"])
def test_g_variant_forward_backward_and_no_genomic_dependency(variant):
    model = _ladder_model(variant)
    inputs = _functional_inputs()
    rna = torch.randn(3, 48)
    first = model(rna, torch.randn(4, 1536), **inputs)["beta"]
    second = model(rna, torch.randn(4, 7), **inputs)["beta"]
    assert first.shape == (3, 4)
    assert torch.equal(first, second)  # genomic embedding input is ignored
    first.mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is None or torch.isfinite(param.grad).all(), name


def test_g0_reproduces_f0_mathematical_behavior():
    # Same weights, same forward math: build both with a shared seed and copy
    # g0's control-path submodules onto an f0 model (different module names,
    # identical computation), then compare outputs exactly.
    torch.manual_seed(3)
    g0 = _ladder_model("g0_control")
    torch.manual_seed(3)
    f0 = _ladder_model("f0_single")
    assert torch.equal(g0.track_embedding.weight, f0.track_embedding.weight)
    assert torch.equal(g0.dense_encoder[1].weight, f0.dense_encoder[1].weight)
    g0.standard_block.load_state_dict(f0.blocks[0].state_dict())
    g0.head.load_state_dict(f0.head.state_dict())
    g0.eval(); f0.eval()
    inputs = _functional_inputs()
    rna = torch.randn(3, 48)
    with torch.no_grad():
        a = g0(rna, None, **inputs)["beta"]
        b = f0(rna, None, **inputs)["beta"]
    assert torch.equal(a, b)


def test_grouped_functional_encoder_returns_expected_shapes():
    torch.manual_seed(0)
    group_id = torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.long)
    enc = GroupedFunctionalEncoder(n_tracks=7, dense_dim=23, group_id=group_id, group_sizes=(3, 2, 2))
    indices = torch.tensor([0, 3, 1, 4, 2, 5], dtype=torch.long)  # locus0: acc x2; locus1: empty; locus2: hist,binding
    offsets = torch.tensor([0, 2, 2, 4, 6], dtype=torch.long)
    dense = torch.randn(4, 23)
    f_c, streams = enc(indices, offsets, dense)
    assert f_c.shape == (4, 256)
    assert len(streams) == 4
    for s in streams:
        assert s.shape == (4, 64)
    assert torch.equal(streams[0][1], torch.zeros(64))  # locus1 accessibility bag is empty


def test_grouped_functional_encoder_arbitrary_cpg_order_is_row_independent():
    # A locus's output must depend only on its own CSR segment + dense row,
    # not on the order loci are presented in -- so permuting the whole batch
    # (segments and dense rows together) permutes the output identically.
    torch.manual_seed(1)
    group_id = torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.long)
    enc = GroupedFunctionalEncoder(n_tracks=7, dense_dim=23, group_id=group_id, group_sizes=(3, 2, 2)).eval()
    indices = torch.tensor([0, 3, 4, 5, 2], dtype=torch.long)
    offsets = torch.tensor([0, 2, 3, 5], dtype=torch.long)  # locus0: acc*2; locus1: hist; locus2: hist,binding
    dense = torch.randn(3, 23)
    with torch.no_grad():
        f_c, _ = enc(indices, offsets, dense)

        perm = [2, 0, 1]
        segments = [(0, 2), (2, 3), (3, 5)]
        perm_indices = torch.cat([indices[segments[i][0]:segments[i][1]] for i in perm])
        perm_counts = [segments[i][1] - segments[i][0] for i in perm]
        perm_offsets = torch.tensor([0, *torch.cumsum(torch.tensor(perm_counts), 0).tolist()], dtype=torch.long)
        f_c_perm, _ = enc(perm_indices, perm_offsets, dense[perm])
    assert torch.allclose(f_c_perm, f_c[perm], atol=1e-6)


def test_g2_maps_exactly_one_stream_per_head_and_gradients_reach_all_streams():
    torch.manual_seed(0)
    attn = RegulatoryHeadCrossAttention(width=8, n_heads=4, dropout=0.0)
    streams = [torch.randn(5, 2, requires_grad=True) for _ in range(4)]
    tokens = torch.randn(3, 6, 8, requires_grad=True)
    out = attn(streams, tokens)
    assert out.shape == (3, 5, 8)
    out.mean().backward()
    for s in streams:
        assert s.grad is not None and torch.isfinite(s.grad).all()
    assert tokens.grad is not None
    # Each head's query projection only ever sees its own stream (constructor
    # invariant, checked structurally): one LayerNorm/Linear pair per head.
    assert len(attn.query_proj) == 4 and len(attn.query_norms) == 4


def test_g3_gate_is_finite_and_in_unit_interval():
    model = _ladder_model("g3_gated_residual").eval()
    with torch.no_grad():
        model(torch.randn(2, 48), None, **_functional_inputs())
    gate = model.gate.last_gate
    assert torch.isfinite(gate).all()
    assert (gate >= 0).all() and (gate <= 1).all()


def test_g3_gate_neutral_at_initialization():
    model = _ladder_model("g3_gated_residual").eval()
    with torch.no_grad():
        model(torch.randn(2, 48), None, **_functional_inputs())
    gate = model.gate.last_gate
    assert torch.allclose(gate, torch.full_like(gate, 0.5))


def test_g4_product_rank_is_exactly_128():
    from methylation_predictor.models import LowRankMultiplicativeFusion
    fusion = LowRankMultiplicativeFusion(width=256, dropout=0.0)
    assert fusion.RANK == 128
    f_c = torch.randn(2, 5, 256)
    a = torch.randn(2, 5, 256)
    u, v = fusion.u_proj(f_c), fusion.v_proj(a)
    m = u * v
    assert m.shape[-1] == 128


def test_g2_mean_proxy_produces_locus_only_mu_logit_gradients_reach_mean_head():
    # Off-ladder mean-branch ablation (docs/MEAN_CONTRIBUTION_EXPERIMENTS.md's
    # causal claim, ported to the functional/regulatory domain): mu_logit must
    # come from f_c alone (locus-only, no batch dependence) and gradients on
    # it must reach mean_head without requiring beta_hat's own loss.
    model = _ladder_model("g2_mean_proxy")
    inputs = _functional_inputs()
    rna = torch.randn(3, 48)
    out = model(rna, None, **inputs)
    assert out["mu_logit"] is not None
    assert out["mu_logit"].shape == (4,)  # one scalar per locus, no batch axis
    out["mu_logit"].sum().backward()
    assert model.mean_head.weight.grad is not None and torch.isfinite(model.mean_head.weight.grad).all()


def test_other_g_variants_have_no_mean_head_and_no_mu_logit():
    for variant in ["g0_control", "g1_grouped", "g2_regulatory_head", "g3_gated_residual", "g4_lowrank_fusion"]:
        model = _ladder_model(variant)
        assert model.mean_head is None
        with torch.no_grad():
            out = model(torch.randn(2, 48), None, **_functional_inputs())
        assert out["mu_logit"] is None


def test_g_variants_no_dense_track_matrix_is_ever_materialized():
    # Structural guard: GroupedFunctionalEncoder must only ever hold
    # EmbeddingBag tables (sparse lookups), never a [n_tracks] dense per-locus
    # matrix parameter/buffer.
    model = _ladder_model("g2_regulatory_head")
    enc = model.functional_encoder
    for bag in enc.bags:
        assert isinstance(bag, torch.nn.EmbeddingBag)
    for name, buf in enc.named_buffers():
        assert buf.dim() <= 1, f"unexpected dense buffer {name} with shape {tuple(buf.shape)}"
