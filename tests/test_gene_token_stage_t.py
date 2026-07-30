from __future__ import annotations

import numpy as np
import pytest
import torch

from methylation_predictor.rna_branch.config import EncoderConfig, InteractionConfig, ModelConfig
from methylation_predictor.rna_branch.data import MatrixStore, align_gene_embeddings
from methylation_predictor.rna_branch.config import MatrixConfig
from methylation_predictor.rna_branch.models import (
    GeneTokenPerceiverRNAEncoder,
    ResidualMethylationModel,
    build_rna_encoder,
)


def _encoder_config(source: str) -> EncoderConfig:
    return EncoderConfig(
        kind="gene_token_perceiver",
        latent_dim=8,
        token_dim=8,
        num_latents=3,
        num_heads=2,
        num_self_attention_blocks=0,
        dropout=0.0,
        value_encoding="linear",
        gene_token_fusion="film",
        gene_identity_source=source,
    )


def test_gene_embeddings_align_to_rna_column_order(tmp_path):
    path = tmp_path / "genes.npz"
    np.savez(
        path,
        embeddings=np.asarray([[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]], dtype=np.float32),
        gene_ids=np.asarray(["g3", "g1", "g2"]),
    )
    store = MatrixStore(MatrixConfig(str(path), "embeddings", "gene_ids"))
    try:
        aligned = align_gene_embeddings(np.asarray(["g1", "g2", "g3"]), store)
    finally:
        store.close()
    assert aligned.tolist() == [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]


def test_gene_embeddings_reject_partial_alignment(tmp_path):
    path = tmp_path / "genes.npz"
    np.savez(path, embeddings=np.ones((1, 2), np.float32), gene_ids=np.asarray(["g1"]))
    store = MatrixStore(MatrixConfig(str(path), "embeddings", "gene_ids"))
    try:
        with pytest.raises(ValueError, match="RNA genes are missing"):
            align_gene_embeddings(np.asarray(["g1", "g2"]), store)
    finally:
        store.close()


@pytest.mark.parametrize("source", ["learned", "ntv3", "ntv3_permuted"])
def test_gene_token_encoder_shapes_and_finite(source):
    genes, static_dim = 11, 13
    static = torch.randn(genes, static_dim)
    static[4].zero_()  # shared unknown identity for an unmatched gene
    embeddings = None if source == "learned" else static
    encoder = build_rna_encoder(genes, _encoder_config(source), embeddings)
    representation = encoder(torch.randn(5, genes))
    assert representation.global_vector.shape == (5, 8)
    assert representation.tokens is not None
    assert representation.tokens.shape == (5, 3, 8)
    assert torch.isfinite(representation.global_vector).all()
    assert torch.isfinite(representation.tokens).all()


def test_permutation_control_preserves_zero_rows_and_changes_assignment():
    genes = 9
    static = torch.arange(genes * 5, dtype=torch.float32).view(genes, 5) + 1
    static[2].zero_()
    direct = GeneTokenPerceiverRNAEncoder(genes, _encoder_config("ntv3"), static)
    permuted = GeneTokenPerceiverRNAEncoder(genes, _encoder_config("ntv3_permuted"), static)
    assert torch.equal(permuted.fixed_gene_embeddings[2], torch.zeros(5))
    assert not torch.equal(direct.fixed_gene_embeddings, permuted.fixed_gene_embeddings)
    direct_nonzero = direct.fixed_gene_embeddings[torch.linalg.vector_norm(direct.fixed_gene_embeddings, dim=1) > 0]
    permuted_nonzero = permuted.fixed_gene_embeddings[torch.linalg.vector_norm(permuted.fixed_gene_embeddings, dim=1) > 0]
    assert torch.equal(
        direct_nonzero.sort(dim=0).values,
        permuted_nonzero.sort(dim=0).values,
    )


def test_gene_token_model_runs_with_f2_and_exact_mean_anchor():
    genes, locus_dim = 12, 10
    config = ModelConfig(
        encoder=_encoder_config("ntv3"),
        interaction=InteractionConfig(kind="concat", hidden_dim=16, dropout=0.0, num_heads=2),
        anchor_to_mean_rna=True,
        zero_init_residual=False,
    )
    model = ResidualMethylationModel(
        genes,
        locus_dim,
        config,
        gene_embeddings=torch.randn(genes, 7),
    ).eval()
    loci = torch.randn(4, locus_dim)
    prior = torch.full((4,), 0.5)
    variability = torch.zeros(4, 2)
    reference = torch.zeros(1, genes)
    output = model(reference, loci, prior, variability, reference_rna=reference)
    assert torch.allclose(output["delta_logit"], torch.zeros_like(output["delta_logit"]), atol=1e-6)


def test_gene_locus_query_interaction_forward():
    genes, locus_dim = 12, 10
    config = ModelConfig(
        encoder=_encoder_config("ntv3"),
        interaction=InteractionConfig(
            kind="gene_token_cross_attention", hidden_dim=16, dropout=0.0, num_heads=2
        ),
        anchor_to_mean_rna=True,
        zero_init_residual=False,
    )
    model = ResidualMethylationModel(
        genes,
        locus_dim,
        config,
        gene_embeddings=torch.randn(genes, 7),
    )
    output = model(
        torch.randn(3, genes),
        torch.randn(5, locus_dim),
        torch.full((5,), 0.5),
        torch.zeros(5, 2),
        reference_rna=torch.zeros(1, genes),
    )
    assert output["beta"].shape == (3, 5)
    assert torch.isfinite(output["beta"]).all()


def test_ntv3_source_requires_aligned_embeddings():
    with pytest.raises(ValueError, match="requires aligned gene embeddings"):
        build_rna_encoder(10, _encoder_config("ntv3"), None)
