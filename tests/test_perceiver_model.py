from pathlib import Path

import torch

from methylation_predictor.config import ModelConfig, EncoderConfig
from methylation_predictor.modeling import PERCEIVER_VARIANT, build_functional_predictor


def test_perceiver_forward_shapes():
    cfg = ModelConfig(encoder=EncoderConfig(kind="locus_attention", n_programs=8, program_dim=256, n_heads=4))
    model, label = build_functional_predictor(
        variant=PERCEIVER_VARIANT, input_dim=1000, config=cfg, final_regressor_dropout=0.1, use_mean_proxy=True
    )
    n, b = 5, 3
    out = model(
        torch.randn(b, 1000),
        functional_track_indices=torch.randint(0, 4165, (2 * n,)),
        functional_offsets=torch.arange(0, 2 * n + 1, 2),
        functional_dense=torch.randn(n, 23),
    )
    assert label == "methylpredictor_perceiver" and out["beta"].shape == (b, n)
    out["beta"].sum().backward()


def _cfg(**kw):
    return ModelConfig(encoder=EncoderConfig(kind="locus_attention", n_programs=8, program_dim=256, n_heads=4), **kw)


def test_generic_query_track_subset():
    from methylation_predictor.modeling import PERCEIVER_GENERIC_VARIANT
    from methylation_predictor.modeling.cpg_query import GenericFunctionalQuery
    import numpy as np
    keep = np.zeros(4165, bool); keep[:10] = True
    q = GenericFunctionalQuery(keep_mask=keep).eval()
    idx, off = torch.tensor([1, 20, 3, 30, 40]), torch.tensor([0, 2, 5])  # row1 has only dropped tracks
    d = torch.randn(2, 23)
    assert torch.allclose(q(torch.tensor([1]), torch.tensor([0, 1, 1]), d), q(torch.tensor([1, 20]), torch.tensor([0, 2, 2]), d))
    model, label = build_functional_predictor(variant=PERCEIVER_GENERIC_VARIANT, input_dim=100, config=_cfg(),
                                              final_regressor_dropout=0.1, use_mean_proxy=True)
    out = model(torch.randn(2, 100), functional_track_indices=torch.tensor([1, 2, 3]),
                functional_offsets=torch.tensor([0, 1, 3]), functional_dense=d)
    assert out["beta"].shape == (2, 2) and label.endswith("functional_generic")


def test_latent_query_forward_and_requires_dim():
    import pytest
    from methylation_predictor.modeling import PERCEIVER_LATENT_VARIANT
    kw = dict(variant=PERCEIVER_LATENT_VARIANT, input_dim=100, config=_cfg(), final_regressor_dropout=0.1, use_mean_proxy=True)
    with pytest.raises(ValueError):
        build_functional_predictor(**kw)
    model, _ = build_functional_predictor(cpg_dim=48, **kw)
    assert model.DENSE_DIM == 48 and not hasattr(model, "track_embedding")
    out = model(torch.randn(2, 100), functional_track_indices=torch.empty(0, dtype=torch.long),
                functional_offsets=torch.zeros(5, dtype=torch.long), functional_dense=torch.randn(4, 48))
    assert out["beta"].shape == (2, 4)
    out["beta"].sum().backward()


def _deep(**decoder):
    from methylation_predictor.modeling import PERCEIVER_DEEP_VARIANT
    model, label = build_functional_predictor(
        variant=PERCEIVER_DEEP_VARIANT, input_dim=100, config=_cfg(decoder=decoder),
        final_regressor_dropout=0.1, use_mean_proxy=True)
    return model, label


def _locus(n):
    return dict(functional_track_indices=torch.randint(0, 4165, (2 * n,)),
                functional_offsets=torch.arange(0, 2 * n + 1, 2), functional_dense=torch.randn(n, 23))


def test_deep_decoder_trains_every_parameter():
    model, label = _deep(width=512, n_rounds=3, ffn_per_round=2, zero_init=False, drop_path=0.2)
    assert label == "methylpredictor_perceiver_deep_decoder"
    assert not hasattr(model, "retrieval_attn") and not hasattr(model, "final_regressor")
    out = model(torch.randn(3, 100), **_locus(5))
    assert out["beta"].shape == (3, 5)
    out["beta"].sum().backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None and not n.startswith("mean_head")]
    assert not dead, dead


def test_deep_decoder_zero_init_rounds_are_identity_and_chunking_is_exact():
    torch.manual_seed(0)
    model, _ = _deep(width=256, n_rounds=3, locus_chunk=3)
    model.eval()
    rna, loci = torch.randn(2, 100), _locus(7)
    chunked = model(rna, **loci)["beta"]
    model.LOCUS_CHUNK = 1024
    assert torch.allclose(chunked, model(rna, **loci)["beta"], atol=1e-5)
    # rounds 2.. contribute nothing at init: dropping them leaves the prediction unchanged
    del model.decoder.rounds[1:]
    assert torch.allclose(chunked, model(rna, **loci)["beta"], atol=1e-5)


def test_deep_decoder_is_larger_and_recipe_loads(tmp_path):
    from methylation_predictor.rna_training.config import load_rna_recipe
    recipe = load_rna_recipe(Path(__file__).parents[1] / "configs/models/perceiver_deep.yaml")
    base, _ = build_functional_predictor(variant=PERCEIVER_VARIANT, input_dim=100, config=_cfg(),
                                         final_regressor_dropout=0.15, use_mean_proxy=True)
    deep, _ = build_functional_predictor(variant=recipe.model.functional_fusion_variant, input_dim=100,
                                         config=recipe.model, final_regressor_dropout=0.15, use_mean_proxy=True)
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert recipe.model.decoder["n_rounds"] == 4 and count(deep) > 1.4 * count(base)
