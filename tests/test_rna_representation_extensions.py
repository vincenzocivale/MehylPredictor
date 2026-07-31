from __future__ import annotations

import numpy as np
import torch

from methylation_predictor.rna_branch.config import EncoderConfig, InteractionConfig, ModelConfig
from methylation_predictor.rna_branch.data import _methylprophet_quantize_rows, _rank_rows
from methylation_predictor.rna_branch.models import ResidualMethylationModel


def _forward(config: ModelConfig, input_dim: int = 12, locus_dim: int = 7):
    model = ResidualMethylationModel(input_dim, locus_dim, config)
    rna = torch.randn(3, input_dim)
    loci = torch.randn(5, locus_dim)
    prior = torch.full((5,), 0.5)
    variability = torch.zeros(5, 2)
    output = model(rna, loci, prior, variability, torch.zeros(1, input_dim))
    assert output["beta"].shape == (3, 5)
    assert torch.isfinite(output["beta"]).all()
    return model, output


def test_rank_and_methylprophet_quantization_are_deterministic():
    values = np.asarray([[0.0, 4.0, 1.0, 4.0], [3.0, 0.0, 2.0, 1.0]], dtype=np.float32)
    ranks = _rank_rows(values)
    assert ranks.shape == values.shape
    first = _methylprophet_quantize_rows(values, ["a", "b"], 51, 17)
    second = _methylprophet_quantize_rows(values, ["a", "b"], 51, 17)
    np.testing.assert_array_equal(first, second)
    assert first.min() >= 0.0 and first.max() <= 1.0
    assert np.all(first[values == 0] == 0)


def test_cpg_conditioned_experts_forward():
    config = ModelConfig(
        encoder=EncoderConfig(kind="global_experts", latent_dim=8, num_experts=4, expert_dim=6),
        interaction=InteractionConfig(kind="cpg_expert_f2", hidden_dim=16),
    )
    model, _ = _forward(config)
    assert model.interaction.last_weights is not None
    torch.testing.assert_close(model.interaction.last_weights.sum(dim=-1), torch.ones(5))


def test_module_query_forward(tmp_path):
    path = tmp_path / "modules.npz"
    np.savez(path, weights=np.asarray([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32))
    config = ModelConfig(
        encoder=EncoderConfig(
            kind="module_tokens", latent_dim=8, token_dim=6,
            module_weights_path=str(path), module_weights_key="weights",
        ),
        interaction=InteractionConfig(kind="cpg_module_f2", hidden_dim=16),
    )
    _forward(config)


def test_sparse_gene_query_forward():
    config = ModelConfig(
        encoder=EncoderConfig(kind="gene_tokens", latent_dim=8, token_dim=6),
        interaction=InteractionConfig(kind="cpg_gene_topk_f2", hidden_dim=16, token_top_k=4),
    )
    model, _ = _forward(config)
    assert model.interaction.last_weights is not None
    assert model.interaction.last_weights.shape == (5, 4)


def test_f2_warm_start_is_prediction_exact():
    source_config = ModelConfig(
        encoder=EncoderConfig(kind="linear", latent_dim=8),
        interaction=InteractionConfig(kind="concat", hidden_dim=16),
        zero_init_residual=False,
    )
    target_config = ModelConfig(
        encoder=EncoderConfig(kind="global_experts", latent_dim=8, num_experts=4, expert_dim=6),
        interaction=InteractionConfig(kind="cpg_expert_f2", hidden_dim=16),
        zero_init_residual=True,
    )
    source = ResidualMethylationModel(12, 7, source_config)
    target = ResidualMethylationModel(12, 7, target_config)
    target_state = target.state_dict()
    remapped = {}
    for key, value in source.state_dict().items():
        baseline = "interaction.baseline." + key[len("interaction."):] if key.startswith("interaction.") else key
        remapped[baseline if baseline in target_state else key] = value
    target.load_state_dict(remapped, strict=False)
    rna = torch.randn(3, 12)
    loci = torch.randn(5, 7)
    prior = torch.full((5,), 0.5)
    variability = torch.zeros(5, 2)
    reference = torch.zeros(1, 12)
    source.eval()
    target.eval()
    with torch.no_grad():
        expected = source(rna, loci, prior, variability, reference)["beta"]
        observed = target(rna, loci, prior, variability, reference)["beta"]
    torch.testing.assert_close(observed, expected)


def test_f2_warm_start_prefers_shape_compatible_wrapped_interaction_key():
    """The residual expert branch may share a name with F2 at another width."""
    source_config = ModelConfig(
        encoder=EncoderConfig(kind="linear", latent_dim=8),
        interaction=InteractionConfig(kind="concat", hidden_dim=16),
    )
    target_config = ModelConfig(
        encoder=EncoderConfig(kind="global_experts", latent_dim=8, num_experts=4, expert_dim=6),
        interaction=InteractionConfig(kind="cpg_expert_f2", hidden_dim=16),
    )
    source = ResidualMethylationModel(12, 7, source_config)
    target = ResidualMethylationModel(12, 7, target_config)
    source_weight = source.state_dict()["interaction.locus_product.weight"]
    target_state = target.state_dict()
    assert target_state["interaction.locus_product.weight"].shape != source_weight.shape
    assert target_state["interaction.baseline.locus_product.weight"].shape == source_weight.shape


def test_pretrained_concat_and_hybrid_transform_contract(tmp_path):
    import pandas as pd

    from methylation_predictor.rna_branch.config import DataConfig, MatrixConfig, TableConfig
    from methylation_predictor.rna_branch.data import load_bundle

    samples = np.asarray(["s0", "s1", "s2", "s3"])
    genes = np.asarray(["g0", "g1", "g2"])
    cpgs = np.asarray(["c0", "c1"])
    np.savez(tmp_path / "rna.npz", X=np.arange(12, dtype=np.float32).reshape(4, 3), sample_idx=samples, gene_ids=genes)
    np.savez(
        tmp_path / "pre.npz", embeddings=np.arange(8, dtype=np.float32).reshape(4, 2),
        sample_idx=samples, feature_ids=np.asarray(["p0", "p1"]),
    )
    np.savez(
        tmp_path / "beta.npz", beta=np.full((4, 2), 0.5, dtype=np.float32),
        sample_idx=samples, cpg_idx=cpgs,
    )
    np.savez(tmp_path / "locus.npz", embeddings=np.ones((2, 4), dtype=np.float32), cpg_idx=cpgs)
    pd.DataFrame(
        {"sample_idx": samples, "cancer_type": ["A", "A", "B", "B"], "split": ["train", "train", "validation", "test"]}
    ).to_csv(tmp_path / "samples.csv", index=False)
    pd.DataFrame({"cpg_idx": cpgs, "split": ["train", "test"]}).to_csv(tmp_path / "cpg.csv", index=False)
    pd.DataFrame(
        {
            "cpg_idx": cpgs,
            "pred_ntv3_prior": [0.4, 0.6],
            "pred_log_var_between": [0.0, 0.0],
            "pred_log_var_within": [0.0, 0.0],
        }
    ).to_csv(tmp_path / "locus.csv", index=False)
    config = DataConfig(
        rna=MatrixConfig(str(tmp_path / "rna.npz"), "X", "sample_idx", "gene_ids"),
        pretrained_rna=MatrixConfig(str(tmp_path / "pre.npz"), "embeddings", "sample_idx", "feature_ids"),
        pretrained_mode="concat",
        methylation=MatrixConfig(str(tmp_path / "beta.npz"), "beta", "sample_idx", "cpg_idx"),
        locus_embeddings=MatrixConfig(str(tmp_path / "locus.npz"), "embeddings", "cpg_idx"),
        locus_features=TableConfig(str(tmp_path / "locus.csv"), "cpg_idx"),
        sample_metadata=TableConfig(str(tmp_path / "samples.csv"), "sample_idx"),
        cpg_splits=TableConfig(str(tmp_path / "cpg.csv"), "cpg_idx"),
        rna_transform="continuous_rank",
    )
    bundle = load_bundle(config)
    try:
        assert bundle.rna_input_dim == 3 * 2 + 2
        assert bundle.rna([0, 2]).shape == (2, 8)
        assert bundle.reference_rna().shape == (8,)
    finally:
        bundle.close()


def test_pretrained_embedding_interaction_forward_and_zero_init():
    input_dim, raw_rna_dim, locus_dim = 12, 8, 7
    config = ModelConfig(
        encoder=EncoderConfig(kind="pretrained_embedding", latent_dim=8),
        interaction=InteractionConfig(kind="cpg_pretrained_f2", hidden_dim=16),
    )
    model = ResidualMethylationModel(input_dim, locus_dim, config, raw_rna_dim=raw_rna_dim)
    rna = torch.randn(3, input_dim)
    loci = torch.randn(5, locus_dim)
    prior = torch.full((5,), 0.5)
    variability = torch.zeros(5, 2)
    reference = torch.zeros(1, input_dim)

    # zero_init_residual (default True) must zero only the residual branch, not baseline:
    # baseline is still randomly initialized at this point (warm-start overwrites it later
    # in the real training entry point), so raw_delta_logit is driven entirely by baseline.
    output = model(rna, loci, prior, variability, reference)
    assert output["beta"].shape == (3, 5)
    assert torch.isfinite(output["beta"]).all()
    torch.testing.assert_close(
        model.interaction.residual[-1].weight, torch.zeros_like(model.interaction.residual[-1].weight)
    )
    torch.testing.assert_close(model.interaction.residual_scale, torch.tensor(1.0))


def test_pretrained_embedding_encoder_rejects_bad_raw_rna_dim():
    config = ModelConfig(
        encoder=EncoderConfig(kind="pretrained_embedding", latent_dim=8),
        interaction=InteractionConfig(kind="cpg_pretrained_f2", hidden_dim=16),
    )
    try:
        ResidualMethylationModel(12, 7, config, raw_rna_dim=None)
        raise AssertionError("expected ValueError when raw_rna_dim is missing")
    except ValueError:
        pass


def test_pretrained_shuffle_does_not_shuffle_raw_rna(tmp_path):
    import pandas as pd

    from methylation_predictor.rna_branch.config import DataConfig, MatrixConfig, TableConfig
    from methylation_predictor.rna_branch.data import load_bundle

    samples = np.asarray(["s0", "s1", "s2", "s3"])
    np.savez(tmp_path / "rna.npz", X=np.arange(12, dtype=np.float32).reshape(4, 3), sample_idx=samples, gene_ids=np.asarray(["g0", "g1", "g2"]))
    np.savez(tmp_path / "pre.npz", embeddings=np.arange(8, dtype=np.float32).reshape(4, 2), sample_idx=samples, feature_ids=np.asarray(["p0", "p1"]))
    np.savez(tmp_path / "beta.npz", beta=np.full((4, 2), 0.5, dtype=np.float32), sample_idx=samples, cpg_idx=np.asarray(["c0", "c1"]))
    np.savez(tmp_path / "locus.npz", embeddings=np.ones((2, 4), dtype=np.float32), cpg_idx=np.asarray(["c0", "c1"]))
    pd.DataFrame({"sample_idx": samples, "cancer_type": ["A", "A", "B", "B"], "split": ["train"] * 4}).to_csv(tmp_path / "samples.csv", index=False)
    pd.DataFrame({"cpg_idx": ["c0", "c1"], "split": ["train", "test"]}).to_csv(tmp_path / "cpg.csv", index=False)
    pd.DataFrame({"cpg_idx": ["c0", "c1"], "pred_ntv3_prior": [0.4, 0.6], "pred_log_var_between": [0.0, 0.0], "pred_log_var_within": [0.0, 0.0]}).to_csv(tmp_path / "locus.csv", index=False)
    config = DataConfig(
        rna=MatrixConfig(str(tmp_path / "rna.npz"), "X", "sample_idx", "gene_ids"),
        pretrained_rna=MatrixConfig(str(tmp_path / "pre.npz"), "embeddings", "sample_idx", "feature_ids"),
        pretrained_mode="concat", pretrained_control="shuffle_within_cancer", pretrained_control_seed=7,
        methylation=MatrixConfig(str(tmp_path / "beta.npz"), "beta", "sample_idx", "cpg_idx"),
        locus_embeddings=MatrixConfig(str(tmp_path / "locus.npz"), "embeddings", "cpg_idx"),
        locus_features=TableConfig(str(tmp_path / "locus.csv"), "cpg_idx"),
        sample_metadata=TableConfig(str(tmp_path / "samples.csv"), "sample_idx"),
        cpg_splits=TableConfig(str(tmp_path / "cpg.csv"), "cpg_idx"), rna_transform="none",
    )
    bundle = load_bundle(config)
    try:
        values = bundle.rna([0, 1, 2, 3])
        np.testing.assert_array_equal(values[:, :3], np.arange(12, dtype=np.float32).reshape(4, 3))
        assert not np.array_equal(values[:, 3:], np.arange(8, dtype=np.float32).reshape(4, 2))
    finally:
        bundle.close()
