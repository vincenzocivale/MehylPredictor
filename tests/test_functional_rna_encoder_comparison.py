from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import (
    RNAEncoderComparisonPredictor,
    SingleRetrievalPredictor,
)
from methylation_predictor.modeling.rna_comparators import (
    BottleneckMLPProgramEncoder,
    FrozenEmbeddingProgramEncoder,
    GenePathwayProgramEncoder,
)
from methylation_predictor.rna_training.config import load_rna_recipe


ROOT = Path(__file__).resolve().parents[1]


def _functional_inputs():
    return {
        "functional_track_indices": torch.tensor(
            [1, 2, 5, 9], dtype=torch.int64
        ),
        "functional_offsets": torch.tensor(
            [0, 2, 2, 3, 4], dtype=torch.int64
        ),
        "functional_dense": torch.randn(4, 23),
    }


def _reference_config():
    return ModelConfig(
        encoder=EncoderConfig(
            kind="locus_attention",
            latent_dim=8,
            n_programs=6,
            program_dim=256,
            n_heads=4,
            dropout=0.0,
            layer_norm=True,
        )
    )


def _membership(path: Path):
    np.savez_compressed(
        path,
        n_genes=np.asarray([6], dtype=np.int64),
        n_pathways=np.asarray([3], dtype=np.int64),
        gene_idx=np.asarray([0, 1, 1, 2, 3, 4], dtype=np.int64),
        pathway_idx=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        pathway_names=np.asarray(["p0", "p1", "p2"]),
        matched_genes_per_pathway=np.asarray([2, 2, 2], dtype=np.int64),
    )


def test_functional_rna_encoder_recipes_keep_non_encoder_protocol_fixed():
    main = load_rna_recipe(ROOT / "configs/models/main.yaml").raw

    files = (
        "functional_ours_program_tokens.yaml",
        "functional_bottleneck_mlp.yaml",
        "functional_gene_pathway.yaml",
        "functional_bulkformer_147m.yaml",
        "functional_bulkrnabert.yaml",
    )

    for filename in files:
        raw = load_rna_recipe(
            ROOT / "configs/models/rna_encoder_comparison" / filename
        ).raw

        for key in (
            "loss",
            "training",
            "schedule_policy",
            "exclude_official_val_from_auxiliary",
            "structured_loss_sources",
            "locus_cls",
            "batching",
            "development",
        ):
            assert raw[key] == main[key]

        lhs = dict(raw["model"])
        rhs = dict(main["model"])
        assert lhs.pop("functional_fusion_variant") == (
            "functional_rna_encoder_comparison"
        )
        assert rhs.pop("functional_fusion_variant") == (
            "mas_concat_v3_purecontext"
        )
        lhs.pop("encoder")
        rhs.pop("encoder")
        assert lhs == rhs


def test_ours_comparison_arm_is_bit_exact_j0_at_initialization_and_forward():
    cfg = _reference_config()

    torch.manual_seed(17)
    j0 = SingleRetrievalPredictor(
        25017,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    torch.manual_seed(17)
    comp = RNAEncoderComparisonPredictor(
        25017,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    assert j0.state_dict().keys() == comp.state_dict().keys()
    for key, value in j0.state_dict().items():
        torch.testing.assert_close(
            value,
            comp.state_dict()[key],
            rtol=0,
            atol=0,
        )

    rna = torch.randn(2, 25017)
    inputs = _functional_inputs()
    with torch.no_grad():
        a = j0(rna, None, **inputs)
        b = comp(rna, None, **inputs)
    torch.testing.assert_close(a["beta"], b["beta"], rtol=0, atol=0)
    torch.testing.assert_close(
        a["mu_hat"], b["mu_hat"], rtol=0, atol=0
    )


def test_bottleneck_and_frozen_encoders_emit_common_token_contract():
    bottleneck = BottleneckMLPProgramEncoder(
        input_dim=12,
        latent_dim=8,
        hidden_dim=16,
        n_blocks=2,
        mlp_ratio=2,
        dropout=0.0,
        layer_norm=True,
        n_programs=6,
        program_dim=256,
    )
    frozen = FrozenEmbeddingProgramEncoder(
        input_dim=10,
        latent_dim=8,
        layer_norm=True,
        n_programs=6,
        program_dim=256,
    )

    assert bottleneck(torch.randn(3, 12)).shape == (3, 6, 256)
    assert frozen(torch.randn(3, 10)).shape == (3, 6, 256)


def test_gene_pathway_program_encoder_forward_and_gradient(tmp_path):
    membership = tmp_path / "membership.npz"
    _membership(membership)

    model = GenePathwayProgramEncoder(
        input_dim=6,
        latent_dim=8,
        membership_path=str(membership),
        dim1=2,
        dim2=3,
        dropout=0.0,
        layer_norm=False,
        n_programs=6,
        program_dim=256,
    )
    x = torch.randn(4, 6, requires_grad=True)
    tokens = model(x)

    assert tokens.shape == (4, 6, 256)
    assert torch.isfinite(tokens).all()
    tokens.square().mean().backward()
    assert model.edge_weight.grad is not None
    assert torch.isfinite(model.edge_weight.grad).all()


def test_downstream_j0_parameters_do_not_depend_on_comparator_encoder(tmp_path):
    ref_cfg = _reference_config()

    torch.manual_seed(23)
    reference = SingleRetrievalPredictor(
        25017,
        ref_cfg,
        final_regressor_dropout=0.15,
    )
    reference_state = reference.state_dict()

    variants = []

    variants.append(
        (
            12,
            ModelConfig(
                encoder=EncoderConfig(
                    kind="bottleneck_mlp",
                    latent_dim=8,
                    hidden_dim=16,
                    n_blocks=2,
                    mlp_ratio=2,
                    dropout=0.0,
                    layer_norm=True,
                    n_programs=6,
                    program_dim=256,
                    n_heads=4,
                )
            ),
        )
    )

    variants.append(
        (
            10,
            ModelConfig(
                encoder=EncoderConfig(
                    kind="frozen_embedding",
                    latent_dim=8,
                    layer_norm=True,
                    n_programs=6,
                    program_dim=256,
                    n_heads=4,
                    frozen_embedding_source="unit-test",
                )
            ),
        )
    )

    membership = tmp_path / "membership.npz"
    _membership(membership)
    variants.append(
        (
            6,
            ModelConfig(
                encoder=EncoderConfig(
                    kind="gene_pathway",
                    latent_dim=8,
                    dropout=0.0,
                    layer_norm=False,
                    n_programs=6,
                    program_dim=256,
                    n_heads=4,
                    pathway_membership_path=str(membership),
                    pathway_dim1=2,
                    pathway_dim2=3,
                )
            ),
        )
    )

    for input_dim, cfg in variants:
        torch.manual_seed(23)
        comparator = RNAEncoderComparisonPredictor(
            input_dim,
            cfg,
            final_regressor_dropout=0.15,
        )
        state = comparator.state_dict()

        for key, value in reference_state.items():
            if key.startswith("rna_encoder."):
                continue
            torch.testing.assert_close(
                value,
                state[key],
                rtol=0,
                atol=0,
            )
