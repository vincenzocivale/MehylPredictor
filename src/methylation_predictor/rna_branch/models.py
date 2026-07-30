"""RNA encoders, locus interactions, and residual methylation model."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import EncoderConfig, ModelConfig


@dataclass
class RNARepresentation:
    global_vector: torch.Tensor
    tokens: torch.Tensor | None = None


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown activation: {name}")


class LinearRNAEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, layer_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.projection = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        return RNARepresentation(self.projection(self.norm(x)))


class MLPRNAEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: float,
        layer_norm: bool,
        activation: str = "gelu",
        input_dropout: float = 0.0,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims, latent_dim]
        layers: list[nn.Module] = [nn.Dropout(input_dropout)] if input_dropout > 0 else []
        for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(dims) - 2:
                if layer_norm:
                    layers.append(nn.LayerNorm(right))
                layers.extend([_activation(activation), nn.Dropout(dropout)])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        return RNARepresentation(self.network(x))


class BottleneckResidualBlock(nn.Module):
    """Pre-norm residual MLP block matching the MethylProphet bottleneck bias."""

    def __init__(self, width: int, expansion_factor: int, dropout: float, activation: str, layer_norm: bool):
        super().__init__()
        hidden = width * expansion_factor
        self.norm = nn.LayerNorm(width) if layer_norm else nn.Identity()
        self.block = nn.Sequential(
            nn.Linear(width, hidden),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(self.norm(x))


class BottleneckMLPRNAEncoder(nn.Module):
    """Raw RNA -> thin width -> residual bottleneck blocks -> latent vector."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        width: int,
        num_blocks: int,
        expansion_factor: int,
        dropout: float,
        input_dropout: float,
        activation: str,
        layer_norm: bool,
    ):
        super().__init__()
        if width < 1 or num_blocks < 1 or expansion_factor < 1:
            raise ValueError("width, num_blocks and expansion_factor must be positive")
        self.input = nn.Sequential(nn.Dropout(input_dropout), nn.Linear(input_dim, width))
        self.blocks = nn.Sequential(*[
            BottleneckResidualBlock(width, expansion_factor, dropout, activation, layer_norm)
            for _ in range(num_blocks)
        ])
        self.output_norm = nn.LayerNorm(width) if layer_norm else nn.Identity()
        self.output = nn.Linear(width, latent_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        hidden = self.blocks(self.input(x))
        return RNARepresentation(self.output(self.output_norm(hidden)))


class ResidualBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        width: int,
        num_blocks: int,
        dropout: float,
        input_dropout: float,
        activation: str,
        layer_norm: bool,
        gated: bool,
        zero_init: bool,
    ):
        super().__init__()
        if width < 1 or num_blocks < 1:
            raise ValueError("width and num_blocks must be positive")
        self.input_dropout = nn.Dropout(input_dropout)
        self.input_projection = nn.Linear(input_dim, width)
        self.gate_projection = nn.Linear(input_dim, width) if gated else None
        layers: list[nn.Module] = []
        for _ in range(num_blocks):
            layers.extend([
                nn.LayerNorm(width) if layer_norm else nn.Identity(),
                nn.Linear(width, width),
                _activation(activation),
                nn.Dropout(dropout),
            ])
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(width, latent_dim)
        if zero_init:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        hidden = self.input_projection(x)
        if self.gate_projection is not None:
            hidden = hidden * torch.sigmoid(self.gate_projection(x))
        return self.output(self.hidden(hidden))


class LinearResidualRNAEncoder(nn.Module):
    """Strong linear C0 path plus a nonlinear correction initialized at zero."""

    def __init__(self, input_dim: int, config: EncoderConfig, gated: bool = False):
        super().__init__()
        self.base = LinearRNAEncoder(input_dim, config.latent_dim, config.layer_norm)
        self.residual = ResidualBranch(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            width=config.width,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            input_dropout=config.input_dropout,
            activation=config.activation,
            layer_norm=config.layer_norm,
            gated=gated,
            zero_init=config.zero_init_encoder_residual,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(config.residual_scale_init)))

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        base = self.base(x).global_vector
        return RNARepresentation(base + self.residual_scale * self.residual(x))


class FourierValueEncoding(nn.Module):
    def __init__(self, frequencies: int):
        super().__init__()
        if frequencies < 1:
            raise ValueError("fourier_frequencies must be positive")
        bands = torch.pow(2.0, torch.arange(frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("bands", bands, persistent=False)

    @property
    def output_dim(self) -> int:
        return 1 + 2 * len(self.bands)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != 1:
            raise ValueError("FourierValueEncoding expects a singleton value dimension")
        scalar = values.squeeze(-1)
        angles = scalar.unsqueeze(-1) * self.bands
        return torch.cat([scalar.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)


class PerceiverRNAEncoder(nn.Module):
    """Compress gene-value tokens into a small latent array.

    Gene identity is represented by a learned embedding; values are encoded either
    linearly or with Fourier features. The same latent array can be pooled for
    bilinear models or queried directly by a CpG in the cross-attention model.
    """

    def __init__(
        self,
        num_genes: int,
        latent_dim: int,
        token_dim: int,
        num_latents: int,
        num_heads: int,
        self_blocks: int,
        dropout: float,
        value_encoding: str,
        fourier_frequencies: int,
    ):
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        self.gene_embeddings = nn.Embedding(num_genes, token_dim)
        self.value_kind = value_encoding
        if value_encoding == "linear":
            self.value_encoder: nn.Module = nn.Linear(1, token_dim)
        elif value_encoding == "fourier":
            fourier = FourierValueEncoding(fourier_frequencies)
            self.value_encoder = nn.Sequential(fourier, nn.Linear(fourier.output_dim, token_dim))
        else:
            raise ValueError(f"unknown value encoding: {value_encoding}")
        self.latents = nn.Parameter(torch.randn(num_latents, token_dim) / math.sqrt(token_dim))
        self.cross_attention = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm_q = nn.LayerNorm(token_dim)
        self.cross_norm_kv = nn.LayerNorm(token_dim)
        self.cross_ff = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 4 * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * token_dim, token_dim),
        )
        self.self_blocks = nn.ModuleList()
        for _ in range(self_blocks):
            attention = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
            feed_forward = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, 4 * token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * token_dim, token_dim),
            )
            self.self_blocks.append(nn.ModuleList([nn.LayerNorm(token_dim), attention, feed_forward]))
        self.global_projection = nn.Linear(token_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        batch, genes = x.shape
        gene_ids = torch.arange(genes, device=x.device)
        identity = self.gene_embeddings(gene_ids).unsqueeze(0)
        value = self.value_encoder(x.unsqueeze(-1))
        tokens = identity + value
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        attended, _ = self.cross_attention(
            self.cross_norm_q(latents), self.cross_norm_kv(tokens), self.cross_norm_kv(tokens), need_weights=False
        )
        latents = latents + attended
        latents = latents + self.cross_ff(latents)
        for norm, attention, feed_forward in self.self_blocks:
            attended, _ = attention(norm(latents), norm(latents), norm(latents), need_weights=False)
            latents = latents + attended
            latents = latents + feed_forward(latents)
        global_vector = self.global_projection(latents.mean(dim=1))
        return RNARepresentation(global_vector, latents)


class LinearMultiTokenRNAEncoder(nn.Module):
    """Strictly linear encoder reshaped into K tokens of dimension d (no nonlinearity, no attention).

    K*d equals the same total capacity as the other 64-dim encoders (e.g. K=4, d=16); the
    nonlinearity/selection in Stage F's F4 fusion lives entirely in the locus-query cross
    attention, not here.
    """

    def __init__(self, input_dim: int, num_tokens: int, token_dim: int, layer_norm: bool = True):
        super().__init__()
        if num_tokens < 1 or token_dim < 1:
            raise ValueError("num_tokens and token_dim must be positive")
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.projection = nn.Linear(input_dim, num_tokens * token_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        batch = x.shape[0]
        tokens = self.projection(self.norm(x)).view(batch, self.num_tokens, self.token_dim)
        return RNARepresentation(tokens.reshape(batch, -1), tokens)


def build_rna_encoder(input_dim: int, config: EncoderConfig) -> nn.Module:
    if config.kind == "linear":
        return LinearRNAEncoder(input_dim, config.latent_dim, config.layer_norm)
    if config.kind == "mlp":
        return MLPRNAEncoder(
            input_dim, config.latent_dim, config.hidden_dims, config.dropout,
            config.layer_norm, config.activation, config.input_dropout
        )
    if config.kind == "bottleneck_mlp":
        return BottleneckMLPRNAEncoder(
            input_dim=input_dim, latent_dim=config.latent_dim, width=config.width,
            num_blocks=config.num_blocks, expansion_factor=config.expansion_factor,
            dropout=config.dropout, input_dropout=config.input_dropout,
            activation=config.activation, layer_norm=config.layer_norm,
        )
    if config.kind == "linear_residual":
        return LinearResidualRNAEncoder(input_dim, config, gated=False)
    if config.kind == "gated_residual":
        return LinearResidualRNAEncoder(input_dim, config, gated=True)
    if config.kind == "perceiver":
        return PerceiverRNAEncoder(
            num_genes=input_dim, latent_dim=config.latent_dim, token_dim=config.token_dim,
            num_latents=config.num_latents, num_heads=config.num_heads,
            self_blocks=config.num_self_attention_blocks, dropout=config.dropout,
            value_encoding=config.value_encoding, fourier_frequencies=config.fourier_frequencies,
        )
    if config.kind == "linear_tokens":
        return LinearMultiTokenRNAEncoder(input_dim, config.num_latents, config.token_dim, config.layer_norm)
    raise ValueError(f"unknown RNA encoder: {config.kind}")


class Interaction(nn.Module):
    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def zero_output(self) -> None:
        pass


class BilinearInteraction(Interaction):
    def __init__(self, rna_dim: int, locus_dim: int, interaction_dim: int, dropout: float):
        super().__init__()
        self.rna_projection = nn.Sequential(nn.LayerNorm(rna_dim), nn.Dropout(dropout), nn.Linear(rna_dim, interaction_dim))
        self.locus_projection = nn.Sequential(nn.LayerNorm(locus_dim), nn.Dropout(dropout), nn.Linear(locus_dim, interaction_dim))
        self.scale = math.sqrt(interaction_dim)

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        p = self.rna_projection(rna.global_vector)
        q = self.locus_projection(loci)
        return p @ q.transpose(0, 1) / self.scale

    def zero_output(self) -> None:
        nn.init.zeros_(self.locus_projection[-1].weight)
        nn.init.zeros_(self.locus_projection[-1].bias)


class InteractionMLP(Interaction):
    """A small locus-conditioned nonlinear residual over 64-dimensional factors."""
    def __init__(self, rna_dim: int, locus_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        self.rna_projection = nn.Sequential(nn.LayerNorm(rna_dim), nn.Linear(rna_dim, 64))
        self.locus_projection = nn.Sequential(nn.LayerNorm(locus_dim), nn.Linear(locus_dim, 64))
        dims = [192, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for n, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(left, right))
            if n < len(dims) - 2:
                layers.extend([nn.LayerNorm(right), nn.GELU(), nn.Dropout(dropout)])
        self.network = nn.Sequential(*layers)

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        p = self.rna_projection(rna.global_vector)
        q = self.locus_projection(loci)
        batch, n_loci = p.shape[0], q.shape[0]
        p2 = p[:, None, :].expand(batch, n_loci, -1)
        q2 = q[None, :, :].expand(batch, n_loci, -1)
        return self.network(torch.cat([p2, q2, p2 * q2], dim=-1)).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


class ConcatOnlyInteraction(Interaction):
    """Stage G1: concat of independently-projected RNA/locus factors, no explicit product term."""

    def __init__(self, rna_dim: int, locus_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        self.rna_projection = nn.Sequential(nn.LayerNorm(rna_dim), nn.Linear(rna_dim, 64))
        self.locus_projection = nn.Sequential(nn.LayerNorm(locus_dim), nn.Linear(locus_dim, 64))
        dims = [128, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for n, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(left, right))
            if n < len(dims) - 2:
                layers.extend([nn.LayerNorm(right), nn.GELU(), nn.Dropout(dropout)])
        self.network = nn.Sequential(*layers)

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        p = self.rna_projection(rna.global_vector)
        q = self.locus_projection(loci)
        batch, n_loci = p.shape[0], q.shape[0]
        p2 = p[:, None, :].expand(batch, n_loci, -1)
        q2 = q[None, :, :].expand(batch, n_loci, -1)
        return self.network(torch.cat([p2, q2], dim=-1)).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


class ProductOnlyInteraction(Interaction):
    """Stage G2: elementwise product of independently-projected RNA/locus factors, no raw concat."""

    def __init__(self, rna_dim: int, locus_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        self.rna_projection = nn.Sequential(nn.LayerNorm(rna_dim), nn.Linear(rna_dim, 64))
        self.locus_projection = nn.Sequential(nn.LayerNorm(locus_dim), nn.Linear(locus_dim, 64))
        dims = [64, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for n, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(left, right))
            if n < len(dims) - 2:
                layers.extend([nn.LayerNorm(right), nn.GELU(), nn.Dropout(dropout)])
        self.network = nn.Sequential(*layers)

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        p = self.rna_projection(rna.global_vector)
        q = self.locus_projection(loci)
        product = p[:, None, :] * q[None, :, :]
        return self.network(product).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


class ConcatProductLinearInteraction(Interaction):
    """Stage G3: same joint feature map as InteractionMLP (G4), but a single linear decoder."""

    def __init__(self, rna_dim: int, locus_dim: int, dropout: float):
        super().__init__()
        self.rna_projection = nn.Sequential(nn.LayerNorm(rna_dim), nn.Linear(rna_dim, 64))
        self.locus_projection = nn.Sequential(nn.LayerNorm(locus_dim), nn.Linear(locus_dim, 64))
        self.input_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(192, 1)

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        p = self.rna_projection(rna.global_vector)
        q = self.locus_projection(loci)
        batch, n_loci = p.shape[0], q.shape[0]
        p2 = p[:, None, :].expand(batch, n_loci, -1)
        q2 = q[None, :, :].expand(batch, n_loci, -1)
        joint = self.input_dropout(torch.cat([p2, q2, p2 * q2], dim=-1))
        return self.output(joint).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


class MultiHeadBilinearInteraction(Interaction):
    """Locus-gated mixture of independent bilinear programs."""
    def __init__(self, rna_dim: int, locus_dim: int, total_dim: int, heads: int, dropout: float):
        super().__init__()
        if heads < 1:
            raise ValueError("num_heads must be positive")
        self.heads = heads
        self.head_dim = max(1, total_dim // heads)
        self.rna_projection = nn.Sequential(nn.LayerNorm(rna_dim), nn.Dropout(dropout), nn.Linear(rna_dim, heads * self.head_dim))
        self.locus_projection = nn.Sequential(nn.LayerNorm(locus_dim), nn.Dropout(dropout), nn.Linear(locus_dim, heads * self.head_dim))
        self.mixture = nn.Sequential(nn.LayerNorm(locus_dim), nn.Linear(locus_dim, heads))
        self.scale = math.sqrt(self.head_dim)
        self.last_weights: torch.Tensor | None = None

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        p = self.rna_projection(rna.global_vector).view(-1, self.heads, self.head_dim)
        q = self.locus_projection(loci).view(-1, self.heads, self.head_dim)
        values = torch.einsum("bhd,lhd->blh", p, q) / self.scale
        weights = torch.softmax(self.mixture(loci), dim=-1)
        self.last_weights = weights.detach()
        return (values * weights.unsqueeze(0)).sum(dim=-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.locus_projection[-1].weight)
        nn.init.zeros_(self.locus_projection[-1].bias)


class BetweenWithinBilinear(nn.Module):
    """Separate bilinear programs for train-centroid and within-cancer RNA."""
    def __init__(self, rna_dim: int, locus_dim: int, interaction_dim: int, dropout: float):
        super().__init__()
        self.between = BilinearInteraction(rna_dim, locus_dim, interaction_dim, dropout)
        self.within = BilinearInteraction(rna_dim, locus_dim, interaction_dim, dropout)
        self.between_gate = ResidualGate("variability", locus_dim, 64, 0.0, 1.0)
        self.within_gate = ResidualGate("variability", locus_dim, 64, 0.0, 1.0)

    def forward(self, p: torch.Tensor, loci: torch.Tensor, variability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        between = self.between(RNARepresentation(p[0]), loci)
        within = self.within(RNARepresentation(p[1]), loci)
        # Each gate receives locus embedding plus its own variability estimate.
        vb = torch.stack([variability[:, 0], variability[:, 0]], dim=-1)
        vw = torch.stack([variability[:, 1], variability[:, 1]], dim=-1)
        return between * self.between_gate(loci, vb).unsqueeze(0), within * self.within_gate(loci, vw).unsqueeze(0)

    def zero_output(self) -> None:
        self.between.zero_output()
        self.within.zero_output()


class RawConcatInteraction(Interaction):
    """Stage G follow-up (F1 revisited): concat of the raw RNA vector and the raw,
    unprojected locus embedding, no explicit product term. Isolates whether F2's win
    comes from direct MLP access to the full-dimensional locus embedding, independent
    of the elementwise product."""

    def __init__(self, rna_dim: int, locus_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        joint_dim = rna_dim + locus_dim
        self.network = nn.Sequential(
            nn.LayerNorm(joint_dim), nn.Linear(joint_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        batch, n_loci = rna.global_vector.shape[0], loci.shape[0]
        rna_expanded = rna.global_vector[:, None, :].expand(batch, n_loci, -1)
        loci_expanded = loci[None, :, :].expand(batch, n_loci, -1)
        return self.network(torch.cat([rna_expanded, loci_expanded], dim=-1)).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


class ConcatInteraction(Interaction):
    def __init__(self, rna_dim: int, locus_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        joint_dim = rna_dim + locus_dim + min(rna_dim, locus_dim)
        self.rna_product = nn.Linear(rna_dim, min(rna_dim, locus_dim))
        self.locus_product = nn.Linear(locus_dim, min(rna_dim, locus_dim))
        self.network = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        batch, n_loci = rna.global_vector.shape[0], loci.shape[0]
        rna_expanded = rna.global_vector[:, None, :].expand(batch, n_loci, -1)
        loci_expanded = loci[None, :, :].expand(batch, n_loci, -1)
        product = self.rna_product(rna.global_vector)[:, None, :] * self.locus_product(loci)[None, :, :]
        return self.network(torch.cat([rna_expanded, loci_expanded, product], dim=-1)).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


class FiLMInteraction(Interaction):
    def __init__(self, rna_dim: int, locus_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.rna_to_film = nn.Linear(rna_dim, 2 * hidden_dim)
        self.locus_projection = nn.Linear(locus_dim, hidden_dim)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        gamma, shift = self.rna_to_film(rna.global_vector).chunk(2, dim=-1)
        locus = self.locus_projection(loci)
        conditioned = (1.0 + gamma[:, None, :]) * locus[None, :, :] + shift[:, None, :]
        return self.output(conditioned).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)


class FiLMLocusInteraction(Interaction):
    """Locus-conditioned FiLM: the CpG (not the patient) generates the modulation.

    gamma_i, eta_i = f(g_i); h_{s,i} = gamma_i * z_s + eta_i; delta_{s,i} = D(h_{s,i}).
    This is the mirror image of `FiLMInteraction`, which conditions on RNA instead.
    """

    def __init__(self, rna_dim: int, locus_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.locus_to_film = nn.Sequential(nn.LayerNorm(locus_dim), nn.Linear(locus_dim, 2 * rna_dim))
        self.decoder = nn.Sequential(
            nn.LayerNorm(rna_dim), nn.Linear(rna_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        gamma, shift = self.locus_to_film(loci).chunk(2, dim=-1)
        conditioned = gamma[None, :, :] * rna.global_vector[:, None, :] + shift[None, :, :]
        return self.decoder(conditioned).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)


class LinearTokenCrossAttention(Interaction):
    """Locus-query cross attention over a strictly linear patient encoder's K tokens.

    One cross-attention block, no self-attention among tokens, no transformer stack: the
    locus only selects which of the patient's linear factors to use, capacity stays fixed.
    """

    def __init__(self, token_dim: int, locus_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(locus_dim, token_dim)
        self.key = nn.Linear(token_dim, token_dim)
        self.value = nn.Linear(token_dim, token_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(token_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        if rna.tokens is None:
            raise ValueError("linear_token_cross_attention requires the linear_tokens RNA encoder")
        tokens = rna.tokens
        q = self.query(loci)
        k = self.key(tokens)
        v = self.value(tokens)
        scores = torch.einsum("ld,bkd->blk", q, k) / self.scale
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        context = torch.einsum("blk,bkd->bld", weights, v)
        return self.decoder(context).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)


class CrossAttentionInteraction(Interaction):
    def __init__(self, token_dim: int, locus_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("Perceiver token_dim must be divisible by interaction num_heads")
        self.query = nn.Linear(locus_dim, token_dim)
        self.attention = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(token_dim + locus_dim),
            nn.Linear(token_dim + locus_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        if rna.tokens is None:
            raise ValueError("cross_attention interaction requires the Perceiver RNA encoder")
        batch, n_loci = rna.tokens.shape[0], loci.shape[0]
        queries = self.query(loci)[None, :, :].expand(batch, n_loci, -1)
        attended, _ = self.attention(queries, rna.tokens, rna.tokens, need_weights=False)
        locus_expanded = loci[None, :, :].expand(batch, n_loci, -1)
        return self.output(torch.cat([attended, locus_expanded], dim=-1)).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)


class ResidualGate(nn.Module):
    def __init__(self, kind: str, locus_dim: int, hidden_dim: int, dropout: float, initial_scale: float):
        super().__init__()
        self.kind = kind
        if kind == "none":
            self.network = None
        elif kind == "global":
            initial_raw = math.log(math.expm1(max(initial_scale, 1e-4)))
            self.raw_scale = nn.Parameter(torch.tensor(initial_raw, dtype=torch.float32))
            self.network = None
        elif kind == "locus":
            self.network = nn.Sequential(
                nn.LayerNorm(locus_dim), nn.Linear(locus_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
            )
        elif kind == "variability":
            self.network = nn.Sequential(
                nn.LayerNorm(locus_dim + 2),
                nn.Linear(locus_dim + 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"unknown gate kind: {kind}")

    def forward(self, loci: torch.Tensor, variability: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return torch.ones(loci.shape[0], device=loci.device, dtype=loci.dtype)
        if self.kind == "global":
            return F.softplus(self.raw_scale).expand(loci.shape[0])
        features = loci if self.kind == "locus" else torch.cat([loci, variability], dim=-1)
        return torch.sigmoid(self.network(features)).squeeze(-1)


class ResidualMethylationModel(nn.Module):
    """Frozen-prior model: logit(beta_hat) = logit(prior) + gated RNA residual."""

    def __init__(self, input_dim: int, locus_dim: int, config: ModelConfig, epsilon: float = 1e-4):
        super().__init__()
        self.config = config
        self.epsilon = epsilon
        encoder = config.encoder
        self.rna_encoder = build_rna_encoder(input_dim, encoder)

        interaction = config.interaction
        if interaction.kind == "bilinear":
            self.interaction: Interaction = BilinearInteraction(
                encoder.latent_dim, locus_dim, interaction.hidden_dim, interaction.dropout
            )
        elif interaction.kind == "interaction_mlp":
            self.interaction = InteractionMLP(
                encoder.latent_dim, locus_dim, interaction.mlp_hidden_dims, interaction.dropout
            )
        elif interaction.kind == "concat_only":
            self.interaction = ConcatOnlyInteraction(
                encoder.latent_dim, locus_dim, interaction.mlp_hidden_dims, interaction.dropout
            )
        elif interaction.kind == "product_only":
            self.interaction = ProductOnlyInteraction(
                encoder.latent_dim, locus_dim, interaction.mlp_hidden_dims, interaction.dropout
            )
        elif interaction.kind == "concat_product_linear":
            self.interaction = ConcatProductLinearInteraction(encoder.latent_dim, locus_dim, interaction.dropout)
        elif interaction.kind == "multihead_bilinear":
            self.interaction = MultiHeadBilinearInteraction(
                encoder.latent_dim, locus_dim, interaction.hidden_dim, interaction.num_heads, interaction.dropout
            )
        elif interaction.kind == "between_within":
            self.interaction = BetweenWithinBilinear(
                encoder.latent_dim, locus_dim, interaction.hidden_dim, interaction.dropout
            )
        elif interaction.kind == "concat":
            self.interaction = ConcatInteraction(encoder.latent_dim, locus_dim, interaction.hidden_dim, interaction.dropout)
        elif interaction.kind == "raw_concat":
            self.interaction = RawConcatInteraction(encoder.latent_dim, locus_dim, interaction.hidden_dim, interaction.dropout)
        elif interaction.kind == "film":
            self.interaction = FiLMInteraction(encoder.latent_dim, locus_dim, interaction.hidden_dim, interaction.dropout)
        elif interaction.kind == "film_locus":
            self.interaction = FiLMLocusInteraction(encoder.latent_dim, locus_dim, interaction.hidden_dim, interaction.dropout)
        elif interaction.kind == "cross_attention":
            if encoder.kind != "perceiver":
                raise ValueError("cross_attention requires model.encoder.kind=perceiver")
            self.interaction = CrossAttentionInteraction(
                encoder.token_dim, locus_dim, interaction.hidden_dim, interaction.num_heads, interaction.dropout
            )
        elif interaction.kind == "linear_token_cross_attention":
            if encoder.kind != "linear_tokens":
                raise ValueError("linear_token_cross_attention requires model.encoder.kind=linear_tokens")
            self.interaction = LinearTokenCrossAttention(
                encoder.token_dim, locus_dim, interaction.hidden_dim, interaction.dropout
            )
        else:
            raise ValueError(f"unknown interaction: {interaction.kind}")

        gate = config.gate
        self.gate = ResidualGate(gate.kind, locus_dim, gate.hidden_dim, gate.dropout, gate.initial_global_scale)
        if config.zero_init_residual:
            self.interaction.zero_output()

    def forward(
        self,
        rna: torch.Tensor,
        loci: torch.Tensor,
        prior: torch.Tensor,
        variability: torch.Tensor,
        reference_rna: torch.Tensor | None = None,
        cancer_codes: torch.Tensor | None = None,
        cancer_centroids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        representation = self.rna_encoder(rna)
        components: dict[str, torch.Tensor] = {}
        if self.config.interaction.kind == "between_within":
            if cancer_codes is None or cancer_centroids is None:
                raise ValueError("between_within requires training-only cancer centroids and sample cancer codes")
            between = cancer_centroids[cancer_codes]
            within = representation.global_vector - between
            raw_between, raw_within = self.interaction((between, within), loci, variability)
            raw = raw_between + raw_within
            components = {"between_delta_logit": raw_between, "within_delta_logit": raw_within}
        else:
            raw = self.interaction(representation, loci)
        if self.config.anchor_to_mean_rna:
            if reference_rna is None:
                reference_rna = torch.zeros((1, rna.shape[1]), dtype=rna.dtype, device=rna.device)
            if self.config.interaction.kind != "between_within":
                reference = self.rna_encoder(reference_rna)
                reference_raw = self.interaction(reference, loci)
                raw = raw - reference_raw.expand_as(raw)
        gate = self.gate(loci, variability)
        delta_logit = raw if self.config.interaction.kind == "between_within" else raw * gate.unsqueeze(0)
        prior = prior.clamp(self.epsilon, 1.0 - self.epsilon)
        prior_logit = torch.logit(prior)
        beta = torch.sigmoid(prior_logit.unsqueeze(0) + delta_logit)
        return {
            "beta": beta,
            "delta_logit": delta_logit,
            "raw_delta_logit": raw,
            "gate": gate,
            "prior_logit": prior_logit,
            **components,
        }
