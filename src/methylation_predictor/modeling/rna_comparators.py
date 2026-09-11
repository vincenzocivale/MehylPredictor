"""RNA encoders for the functional, matched RNA-encoder comparison.

Every encoder returns a tensor of shape ``[batch, n_programs, program_dim]``.
This keeps J0's functional locus encoder, locus-conditioned retrieval,
mean-proxy task, and beta regressor fixed while varying only the patient RNA
representation upstream of retrieval.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import EncoderConfig
from .rna import ProgramTokenEncoder


class GlobalToProgramTokens(nn.Module):
    """Lift one patient-global latent into the common program-token space."""

    def __init__(
        self,
        global_dim: int,
        n_programs: int,
        program_dim: int,
    ):
        super().__init__()
        self.global_norm = nn.LayerNorm(global_dim)
        self.program_basis = nn.Parameter(
            torch.empty(n_programs, program_dim, global_dim)
        )
        nn.init.normal_(self.program_basis, std=global_dim ** -0.5)
        self.token_norm = nn.LayerNorm(program_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_norm(x)
        tokens = torch.einsum("br,kdr->bkd", x, self.program_basis)
        return self.token_norm(tokens)


class _BottleneckMLPBlock(nn.Module):
    def __init__(self, width: int, inner_width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, inner_width)
        self.fc2 = nn.Linear(inner_width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(
            self.dropout(
                torch.nn.functional.gelu(self.fc1(self.norm(x)))
            )
        )
        return x + h


class BottleneckMLPProgramEncoder(nn.Module):
    """MethylProphet-style B_6-Wi_1024 encoder -> common token bank."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_blocks: int,
        mlp_ratio: int,
        dropout: float,
        layer_norm: bool,
        n_programs: int,
        program_dim: int,
    ):
        super().__init__()
        self.norm = (
            nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            _BottleneckMLPBlock(
                hidden_dim,
                hidden_dim * mlp_ratio,
                dropout,
            )
            for _ in range(n_blocks)
        )
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.to_tokens = GlobalToProgramTokens(
            latent_dim, n_programs, program_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(self.norm(x))
        for block in self.blocks:
            h = block(h)
        return self.to_tokens(self.output_proj(h))


class FrozenEmbeddingProgramEncoder(nn.Module):
    """Frozen sample embedding + thin adapter -> common token bank."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        layer_norm: bool,
        n_programs: int,
        program_dim: int,
    ):
        super().__init__()
        self.norm = (
            nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        )
        self.projection = nn.Linear(input_dim, latent_dim)
        self.to_tokens = GlobalToProgramTokens(
            latent_dim, n_programs, program_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_tokens(self.projection(self.norm(x)))


class GenePathwayProgramEncoder(nn.Module):
    """SurvPath-style sparse gene->pathway encoder -> common token bank."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        membership_path: str,
        dim1: int,
        dim2: int,
        dropout: float,
        layer_norm: bool,
        n_programs: int,
        program_dim: int,
    ):
        super().__init__()
        if not membership_path:
            raise ValueError(
                "gene_pathway requires encoder.pathway_membership_path"
            )
        path = Path(membership_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"gene-pathway membership file not found: {path}"
            )

        data = np.load(path, allow_pickle=False)
        n_genes = int(np.asarray(data["n_genes"]).reshape(-1)[0])
        n_pathways = int(np.asarray(data["n_pathways"]).reshape(-1)[0])
        gene_idx = np.asarray(data["gene_idx"], dtype=np.int64)
        pathway_idx = np.asarray(data["pathway_idx"], dtype=np.int64)

        if n_genes != input_dim:
            raise ValueError(
                f"pathway membership was built for {n_genes} genes but "
                f"RNA cache width is {input_dim}"
            )
        if (
            gene_idx.ndim != 1
            or pathway_idx.ndim != 1
            or gene_idx.shape != pathway_idx.shape
        ):
            raise ValueError(
                "invalid gene-pathway membership: gene_idx/pathway_idx "
                "must be equal-length 1-D arrays"
            )
        if len(gene_idx) == 0:
            raise ValueError("gene-pathway membership has zero edges")

        self.input_dim = int(input_dim)
        self.n_pathways = int(n_pathways)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.norm = (
            nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        )

        indices = np.stack([pathway_idx, gene_idx], axis=0)
        self.register_buffer(
            "membership_indices",
            torch.from_numpy(indices).long(),
            persistent=True,
        )

        n_edges = int(gene_idx.size)
        per_path = np.bincount(pathway_idx, minlength=n_pathways)
        scale = float(
            1.0 / max(np.sqrt(max(float(per_path.mean()), 1.0)), 1.0)
        )
        self.edge_weight = nn.Parameter(torch.empty(self.dim1, n_edges))
        nn.init.normal_(self.edge_weight, std=scale)
        self.bias1 = nn.Parameter(
            torch.zeros(n_pathways, self.dim1)
        )

        self.weight2 = nn.Parameter(
            torch.empty(n_pathways, self.dim1, self.dim2)
        )
        nn.init.xavier_normal_(self.weight2)
        self.bias2 = nn.Parameter(
            torch.zeros(n_pathways, self.dim2)
        )

        flattened = n_pathways * self.dim2
        hidden = max(latent_dim, flattened // 4)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(flattened),
            nn.Linear(flattened, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )
        self.to_tokens = GlobalToProgramTokens(
            latent_dim, n_programs, program_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x_t = x.float().transpose(0, 1).contiguous()

        first_channels = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for channel in range(self.dim1):
                weight = torch.sparse_coo_tensor(
                    self.membership_indices,
                    self.edge_weight[channel].float(),
                    size=(self.n_pathways, self.input_dim),
                    device=x.device,
                ).coalesce()
                first_channels.append(
                    torch.sparse.mm(weight, x_t).transpose(0, 1)
                )

        h1 = (
            torch.stack(first_channels, dim=-1)
            + self.bias1.float()[None, :, :]
        )
        h2 = (
            torch.einsum(
                "bpi,pij->bpj",
                h1,
                self.weight2.float(),
            )
            + self.bias2.float()[None, :, :]
        )
        latent = self.to_latent(h2.reshape(h2.shape[0], -1))
        return self.to_tokens(latent)


def build_comparison_program_encoder(
    config: EncoderConfig,
    input_dim: int,
) -> nn.Module:
    """Build one RNA encoder under the common program-token contract."""

    common = dict(
        n_programs=config.n_programs,
        program_dim=config.program_dim,
    )

    if config.kind == "locus_attention":
        return ProgramTokenEncoder(
            input_dim=input_dim,
            n_programs=config.n_programs,
            program_dim=config.program_dim,
            bottleneck_dim=config.latent_dim,
            layer_norm=config.layer_norm,
        )

    if config.kind == "bottleneck_mlp":
        return BottleneckMLPProgramEncoder(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            n_blocks=config.n_blocks,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
            **common,
        )

    if config.kind == "gene_pathway":
        return GenePathwayProgramEncoder(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            membership_path=config.pathway_membership_path,
            dim1=config.pathway_dim1,
            dim2=config.pathway_dim2,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
            **common,
        )

    if config.kind == "frozen_embedding":
        return FrozenEmbeddingProgramEncoder(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            layer_norm=config.layer_norm,
            **common,
        )

    raise ValueError(
        "functional RNA-encoder comparison supports "
        "locus_attention, bottleneck_mlp, gene_pathway, "
        f"or frozen_embedding; got {config.kind!r}"
    )
