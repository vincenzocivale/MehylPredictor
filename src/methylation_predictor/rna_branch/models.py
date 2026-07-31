"""RNA encoders, locus interactions, and residual methylation model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import EncoderConfig, ModelConfig


@dataclass
class RNARepresentation:
    global_vector: torch.Tensor
    tokens: torch.Tensor | None = None
    token_keys: torch.Tensor | None = None


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


class GeneTokenPerceiverRNAEncoder(nn.Module):
    """Build one expression-conditioned token per gene, then pool to patient latents.

    The identity source is the only difference among the Stage-T controls:
    learned IDs (T1), aligned frozen NTv3 gene embeddings (T2), or a fixed
    permutation of the same NTv3 rows (T3).  Expression encoding, latent queries,
    and downstream interaction remain identical.
    """

    def __init__(
        self,
        num_genes: int,
        config: EncoderConfig,
        gene_embeddings: torch.Tensor | None = None,
    ):
        super().__init__()
        if config.token_dim % config.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        if config.num_latents < 1:
            raise ValueError("num_latents must be positive")
        self.num_genes = num_genes
        self.token_dim = config.token_dim
        self.identity_source = config.gene_identity_source
        self.fusion_kind = config.gene_token_fusion

        if self.identity_source == "learned":
            self.learned_gene_embeddings: nn.Module | None = nn.Embedding(num_genes, config.token_dim)
            self.gene_projection: nn.Module = nn.Identity()
            self.register_buffer("fixed_gene_embeddings", None, persistent=False)
        elif self.identity_source in {"ntv3", "ntv3_permuted"}:
            if gene_embeddings is None:
                raise ValueError(f"{self.identity_source} requires aligned gene embeddings")
            if gene_embeddings.ndim != 2 or gene_embeddings.shape[0] != num_genes:
                raise ValueError(
                    f"gene embedding shape {tuple(gene_embeddings.shape)} does not match {num_genes} RNA genes"
                )
            values = gene_embeddings.detach().float().cpu()
            if self.identity_source == "ntv3_permuted":
                # Keep zero rows (unmatched genes) fixed and permute only real
                # NTv3 embeddings, so the control destroys gene assignment without
                # changing annotation coverage.
                generator = torch.Generator(device="cpu")
                generator.manual_seed(config.gene_embedding_permutation_seed)
                present = torch.linalg.vector_norm(values, dim=1) > 0
                present_indices = torch.nonzero(present, as_tuple=False).flatten()
                permutation = present_indices[torch.randperm(len(present_indices), generator=generator)]
                permuted = values.clone()
                permuted[present_indices] = values[permutation]
                values = permuted
            self.learned_gene_embeddings = None
            if config.freeze_gene_embeddings:
                self.register_buffer("fixed_gene_embeddings", values, persistent=False)
            else:
                self.fixed_gene_embeddings = nn.Parameter(values)
            self.gene_projection = nn.Sequential(
                nn.LayerNorm(values.shape[1]),
                nn.Linear(values.shape[1], config.token_dim),
            )
        else:
            raise ValueError(f"unknown gene identity source: {self.identity_source}")

        if config.value_encoding == "linear":
            value_input_dim = 1
            self.value_features: nn.Module = nn.Identity()
        elif config.value_encoding == "fourier":
            fourier = FourierValueEncoding(config.fourier_frequencies)
            value_input_dim = fourier.output_dim
            self.value_features = fourier
        else:
            raise ValueError(f"unknown value encoding: {config.value_encoding}")

        value_output_dim = 2 * config.token_dim if self.fusion_kind == "film" else config.token_dim
        self.value_projection = nn.Linear(value_input_dim, value_output_dim)
        if self.fusion_kind == "concat":
            self.token_fusion: nn.Module | None = nn.Sequential(
                nn.LayerNorm(2 * config.token_dim),
                nn.Linear(2 * config.token_dim, config.token_dim),
                nn.GELU(),
            )
        elif self.fusion_kind in {"add", "film"}:
            self.token_fusion = None
        else:
            raise ValueError(f"unknown gene token fusion: {self.fusion_kind}")

        self.latents = nn.Parameter(
            torch.randn(config.num_latents, config.token_dim) / math.sqrt(config.token_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            config.token_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.cross_norm_q = nn.LayerNorm(config.token_dim)
        self.cross_norm_kv = nn.LayerNorm(config.token_dim)
        self.cross_ff = nn.Sequential(
            nn.LayerNorm(config.token_dim),
            nn.Linear(config.token_dim, 4 * config.token_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.token_dim, config.token_dim),
        )
        self.self_blocks = nn.ModuleList()
        for _ in range(config.num_self_attention_blocks):
            attention = nn.MultiheadAttention(
                config.token_dim, config.num_heads, dropout=config.dropout, batch_first=True
            )
            feed_forward = nn.Sequential(
                nn.LayerNorm(config.token_dim),
                nn.Linear(config.token_dim, 4 * config.token_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(4 * config.token_dim, config.token_dim),
            )
            self.self_blocks.append(nn.ModuleList([nn.LayerNorm(config.token_dim), attention, feed_forward]))
        self.global_projection = nn.Linear(config.token_dim, config.latent_dim)

    def _identity(self, device: torch.device) -> torch.Tensor:
        if self.learned_gene_embeddings is not None:
            ids = torch.arange(self.num_genes, device=device)
            return self.learned_gene_embeddings(ids)
        return self.gene_projection(self.fixed_gene_embeddings.to(device))

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        if x.ndim != 2 or x.shape[1] != self.num_genes:
            raise ValueError(f"expected RNA [batch,{self.num_genes}], received {tuple(x.shape)}")
        identity = self._identity(x.device).unsqueeze(0)
        values = self.value_projection(self.value_features(x.unsqueeze(-1)))
        if self.fusion_kind == "add":
            tokens = identity + values
        elif self.fusion_kind == "film":
            gamma, shift = values.chunk(2, dim=-1)
            tokens = (1.0 + gamma) * identity + shift
        else:
            identity_expanded = identity.expand(x.shape[0], -1, -1)
            tokens = self.token_fusion(torch.cat([identity_expanded, values], dim=-1))

        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            self.cross_norm_q(latents),
            self.cross_norm_kv(tokens),
            self.cross_norm_kv(tokens),
            need_weights=False,
        )
        latents = latents + attended
        latents = latents + self.cross_ff(latents)
        for norm, attention, feed_forward in self.self_blocks:
            attended, _ = attention(norm(latents), norm(latents), norm(latents), need_weights=False)
            latents = latents + attended
            latents = latents + feed_forward(latents)
        return RNARepresentation(self.global_projection(latents.mean(dim=1)), latents)


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


class GlobalExpertRNAEncoder(nn.Module):
    """Strong global linear representation plus independent supervised RNA experts."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_experts: int,
        expert_dim: int,
        layer_norm: bool = True,
    ):
        super().__init__()
        if num_experts < 2 or expert_dim < 1:
            raise ValueError("global_experts requires num_experts >= 2 and expert_dim >= 1")
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        # Names intentionally match LinearRNAEncoder, enabling exact F2 warm-start.
        self.projection = nn.Linear(input_dim, latent_dim)
        self.expert_projection = nn.Linear(input_dim, num_experts * expert_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        normalized = self.norm(x)
        global_vector = self.projection(normalized)
        experts = self.expert_projection(normalized).view(x.shape[0], self.num_experts, self.expert_dim)
        return RNARepresentation(global_vector, experts)


def _load_matrix(path: str, key: str) -> torch.Tensor:
    import numpy as np

    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(file)
    if file.suffix.lower() == ".npy":
        values = np.load(file, allow_pickle=False)
    elif file.suffix.lower() == ".npz":
        archive = np.load(file, allow_pickle=False)
        if key not in archive:
            raise KeyError(f"{key!r} not present in {file}")
        values = archive[key]
    else:
        raise ValueError(f"module/gene matrices must be .npy or .npz: {file}")
    return torch.as_tensor(values, dtype=torch.float32)


class ModuleTokenRNAEncoder(nn.Module):
    """Global linear RNA vector plus pathway/regulon/NMF tokens from a fixed gene-module map."""

    def __init__(self, input_dim: int, config: EncoderConfig):
        super().__init__()
        if not config.module_weights_path:
            raise ValueError("module_tokens requires model.encoder.module_weights_path")
        weights = _load_matrix(config.module_weights_path, config.module_weights_key)
        if weights.ndim != 2 or weights.shape[1] != input_dim:
            raise ValueError(
                f"module weights must have shape [modules, {input_dim}], got {tuple(weights.shape)}"
            )
        denominator = weights.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("module_weights", weights / denominator)
        self.norm = nn.LayerNorm(input_dim) if config.layer_norm else nn.Identity()
        self.projection = nn.Linear(input_dim, config.latent_dim)
        self.module_identity = nn.Embedding(weights.shape[0], config.token_dim)
        self.value_encoder = nn.Linear(1, config.token_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        normalized = self.norm(x)
        global_vector = self.projection(normalized)
        scores = normalized @ self.module_weights.transpose(0, 1)
        identity = self.module_identity.weight.unsqueeze(0)
        tokens = identity + self.value_encoder(scores.unsqueeze(-1))
        return RNARepresentation(global_vector, tokens, self.module_identity.weight)


class GeneTokenRNAEncoder(nn.Module):
    """Global F2 vector plus explicit gene tokens for sparse CpG-to-gene routing."""

    def __init__(self, input_dim: int, config: EncoderConfig):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if config.layer_norm else nn.Identity()
        self.projection = nn.Linear(input_dim, config.latent_dim)
        if config.gene_embedding_path:
            embeddings = _load_matrix(config.gene_embedding_path, config.gene_embedding_key)
            if embeddings.ndim != 2 or embeddings.shape[0] != input_dim:
                raise ValueError(
                    f"gene embeddings must have shape [{input_dim}, dim], got {tuple(embeddings.shape)}"
                )
            if config.freeze_gene_embeddings:
                self.register_buffer("gene_identity_source", embeddings)
            else:
                self.gene_identity_source = nn.Parameter(embeddings)
            self.identity_projection = (
                nn.Identity()
                if embeddings.shape[1] == config.token_dim
                else nn.Linear(embeddings.shape[1], config.token_dim, bias=False)
            )
        else:
            self.gene_identity_source = nn.Parameter(
                torch.randn(input_dim, config.token_dim) / math.sqrt(config.token_dim)
            )
            self.identity_projection = nn.Identity()
        self.value_encoder = nn.Sequential(nn.Linear(1, config.token_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        normalized = self.norm(x)
        global_vector = self.projection(normalized)
        identity = self.identity_projection(self.gene_identity_source)
        # Identity remains available separately as static routing keys. Expression affects values only.
        tokens = identity.unsqueeze(0) + self.value_encoder(normalized.unsqueeze(-1))
        return RNARepresentation(global_vector, tokens, identity)


class PretrainedEmbeddingRNAEncoder(nn.Module):
    """F2-identical linear projection of the raw RNA vector, plus a frozen pretrained
    sample embedding (e.g. BulkRNABert) passed through untouched as ``tokens`` for a
    downstream residual interaction to consume.

    ``raw_rna_dim`` is derived from the data bundle at construction time (not a config
    field) so it can never silently drift from the actual concatenated input layout.
    Names intentionally match ``LinearRNAEncoder`` (``norm``/``projection``), enabling
    exact F2 warm-start of the raw-RNA half via the existing checkpoint remap.
    """

    def __init__(self, input_dim: int, raw_rna_dim: int, latent_dim: int, layer_norm: bool = True):
        super().__init__()
        if not 0 < raw_rna_dim < input_dim:
            raise ValueError(f"raw_rna_dim must be in (0, {input_dim}), got {raw_rna_dim}")
        self.raw_rna_dim = raw_rna_dim
        self.norm = nn.LayerNorm(raw_rna_dim) if layer_norm else nn.Identity()
        self.projection = nn.Linear(raw_rna_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        if x.shape[1] <= self.raw_rna_dim:
            raise ValueError(
                f"expected input wider than raw_rna_dim={self.raw_rna_dim}, received width {x.shape[1]}"
            )
        raw, embedding = x[:, : self.raw_rna_dim], x[:, self.raw_rna_dim :]
        return RNARepresentation(self.projection(self.norm(raw)), embedding)


def build_rna_encoder(
    input_dim: int,
    config: EncoderConfig,
    gene_embeddings: torch.Tensor | None = None,
    raw_rna_dim: int | None = None,
) -> nn.Module:
    if config.kind == "pretrained_embedding":
        if raw_rna_dim is None:
            raise ValueError("pretrained_embedding requires raw_rna_dim")
        return PretrainedEmbeddingRNAEncoder(input_dim, raw_rna_dim, config.latent_dim, config.layer_norm)
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
    if config.kind == "gene_token_perceiver":
        return GeneTokenPerceiverRNAEncoder(input_dim, config, gene_embeddings)
    if config.kind == "global_experts":
        return GlobalExpertRNAEncoder(
            input_dim, config.latent_dim, config.num_experts, config.expert_dim, config.layer_norm
        )
    if config.kind == "module_tokens":
        return ModuleTokenRNAEncoder(input_dim, config)
    if config.kind == "gene_tokens":
        return GeneTokenRNAEncoder(input_dim, config)
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


class GeneLatentCrossAttentionInteraction(Interaction):
    """Query patient gene latents with the CpG while retaining F2's raw-locus/product path."""

    def __init__(self, token_dim: int, locus_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("gene latent token_dim must be divisible by num_heads")
        self.query = nn.Linear(locus_dim, token_dim)
        self.attention = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.locus_product = nn.Linear(locus_dim, token_dim)
        joint_dim = token_dim + locus_dim + token_dim
        self.network = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        if rna.tokens is None:
            raise ValueError("gene_token_cross_attention requires patient gene latents")
        batch, n_loci = rna.tokens.shape[0], loci.shape[0]
        queries = self.query(loci)[None, :, :].expand(batch, n_loci, -1)
        attended, _ = self.attention(queries, rna.tokens, rna.tokens, need_weights=False)
        loci_expanded = loci[None, :, :].expand(batch, n_loci, -1)
        product = attended * self.locus_product(loci)[None, :, :]
        return self.network(torch.cat([attended, loci_expanded, product], dim=-1)).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


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




class CpGExpertF2Interaction(Interaction):
    """F2 baseline plus a CpG-gated mixture of independent RNA projections."""

    def __init__(
        self,
        rna_dim: int,
        expert_dim: int,
        num_experts: int,
        locus_dim: int,
        hidden_dim: int,
        dropout: float,
        temperature: float,
        residual_scale_init: float,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("expert_temperature must be positive")
        self.baseline = ConcatInteraction(rna_dim, locus_dim, hidden_dim, dropout)
        self.num_experts = num_experts
        self.temperature = temperature
        self.expert_gate = nn.Sequential(nn.LayerNorm(locus_dim), nn.Linear(locus_dim, num_experts))
        product_dim = min(expert_dim, locus_dim)
        self.expert_product = nn.Linear(expert_dim, product_dim)
        self.locus_product = nn.Linear(locus_dim, product_dim)
        joint_dim = expert_dim + locus_dim + product_dim
        self.residual = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.last_weights: torch.Tensor | None = None

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        if rna.tokens is None or rna.tokens.shape[1] != self.num_experts:
            raise ValueError("cpg_expert_f2 requires the global_experts RNA encoder")
        baseline = self.baseline(rna, loci)
        weights = torch.softmax(self.expert_gate(loci) / self.temperature, dim=-1)
        self.last_weights = weights.detach()
        context = torch.einsum("lk,bkd->bld", weights, rna.tokens)
        loci_expanded = loci[None, :, :].expand(context.shape[0], -1, -1)
        product = self.expert_product(context) * self.locus_product(loci)[None, :, :]
        residual = self.residual(torch.cat([context, loci_expanded, product], dim=-1)).squeeze(-1)
        return baseline + self.residual_scale * residual

    def zero_output(self) -> None:
        self.baseline.zero_output()
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)


class CpGTokenF2Interaction(Interaction):
    """F2 baseline plus CpG-query attention over pathway/regulon tokens."""

    def __init__(
        self,
        rna_dim: int,
        token_dim: int,
        locus_dim: int,
        hidden_dim: int,
        dropout: float,
        residual_scale_init: float,
    ):
        super().__init__()
        self.baseline = ConcatInteraction(rna_dim, locus_dim, hidden_dim, dropout)
        self.query = nn.Linear(locus_dim, token_dim)
        self.key = nn.Linear(token_dim, token_dim)
        self.value = nn.Linear(token_dim, token_dim)
        self.scale = math.sqrt(token_dim)
        product_dim = min(token_dim, locus_dim)
        self.token_product = nn.Linear(token_dim, product_dim)
        self.locus_product = nn.Linear(locus_dim, product_dim)
        self.residual = nn.Sequential(
            nn.LayerNorm(token_dim + locus_dim + product_dim),
            nn.Linear(token_dim + locus_dim + product_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.last_weights: torch.Tensor | None = None

    def _context(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        if rna.tokens is None:
            raise ValueError("cpg_module_f2 requires tokenized RNA")
        q = self.query(loci)
        k = self.key(rna.tokens)
        v = self.value(rna.tokens)
        scores = torch.einsum("ld,bkd->blk", q, k) / self.scale
        weights = torch.softmax(scores, dim=-1)
        self.last_weights = weights.detach()
        return torch.einsum("blk,bkd->bld", weights, v)

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        baseline = self.baseline(rna, loci)
        context = self._context(rna, loci)
        loci_expanded = loci[None, :, :].expand(context.shape[0], -1, -1)
        product = self.token_product(context) * self.locus_product(loci)[None, :, :]
        residual = self.residual(torch.cat([context, loci_expanded, product], dim=-1)).squeeze(-1)
        return baseline + self.residual_scale * residual

    def zero_output(self) -> None:
        self.baseline.zero_output()
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)


class CpGGeneTopKF2Interaction(CpGTokenF2Interaction):
    """F2 plus sparse, locus-static top-k routing over explicit patient-specific gene tokens."""

    def __init__(self, *args, top_k: int, **kwargs):
        super().__init__(*args, **kwargs)
        if top_k < 1:
            raise ValueError("token_top_k must be positive")
        self.top_k = top_k

    def _context(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        if rna.tokens is None or rna.token_keys is None:
            raise ValueError("cpg_gene_topk_f2 requires gene_tokens with static token_keys")
        q = self.query(loci)
        static_keys = self.key(rna.token_keys)
        routing_scores = q @ static_keys.transpose(0, 1) / self.scale
        k = min(self.top_k, static_keys.shape[0])
        selected_scores, selected = torch.topk(routing_scores, k=k, dim=-1)
        selected_values = self.value(rna.tokens)[:, selected, :]
        weights = torch.softmax(selected_scores, dim=-1)
        self.last_weights = weights.detach()
        return torch.einsum("lk,blkd->bld", weights, selected_values)


class PretrainedEmbeddingF2Interaction(Interaction):
    """F2 baseline (frozen via warm-start) plus a residual adapter over a single frozen
    pretrained sample embedding (e.g. BulkRNABert), gated by locus identity.

    Unlike ``CpGExpertF2Interaction``/``CpGTokenF2Interaction``, ``baseline`` is left at
    its default initialization here rather than zero-initialized in ``zero_output()``:
    the training entry point always warm-starts it from a real F2 checkpoint before any
    forward pass is used for training, and relying on a call-order coincidence
    (zero-init at construction, overwritten later by warm-start) is fragile. Only the
    new residual branch's final layer is zero-init; ``residual_scale`` starts at
    ``token_residual_scale_init`` (default 1.0, not 0) so the residual branch is not
    permanently dead -- zeroing both simultaneously would make every gradient through
    the branch identically zero, since ``residual_scale`` multiplies the (already zero)
    residual output at every layer, not just the last one.
    """

    def __init__(
        self,
        rna_dim: int,
        pretrained_embedding_dim: int,
        locus_dim: int,
        hidden_dim: int,
        dropout: float,
        residual_scale_init: float,
    ):
        super().__init__()
        self.baseline = ConcatInteraction(rna_dim, locus_dim, hidden_dim, dropout)
        product_dim = min(pretrained_embedding_dim, locus_dim)
        self.embedding_product = nn.Linear(pretrained_embedding_dim, product_dim)
        self.locus_product = nn.Linear(locus_dim, product_dim)
        joint_dim = pretrained_embedding_dim + locus_dim + product_dim
        self.residual = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        if rna.tokens is None:
            raise ValueError("cpg_pretrained_f2 requires the pretrained_embedding RNA encoder")
        baseline = self.baseline(rna, loci)
        embedding = rna.tokens
        batch, n_loci = embedding.shape[0], loci.shape[0]
        embedding_expanded = embedding[:, None, :].expand(batch, n_loci, -1)
        loci_expanded = loci[None, :, :].expand(batch, n_loci, -1)
        product = self.embedding_product(embedding)[:, None, :] * self.locus_product(loci)[None, :, :]
        residual = self.residual(torch.cat([embedding_expanded, loci_expanded, product], dim=-1)).squeeze(-1)
        return baseline + self.residual_scale * residual

    def zero_output(self) -> None:
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)


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

    def __init__(
        self,
        input_dim: int,
        locus_dim: int,
        config: ModelConfig,
        epsilon: float = 1e-4,
        gene_embeddings: torch.Tensor | None = None,
        raw_rna_dim: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.epsilon = epsilon
        encoder = config.encoder
        self.rna_encoder = build_rna_encoder(input_dim, encoder, gene_embeddings, raw_rna_dim)

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
        elif interaction.kind == "gene_token_cross_attention":
            if encoder.kind != "gene_token_perceiver":
                raise ValueError("gene_token_cross_attention requires model.encoder.kind=gene_token_perceiver")
            self.interaction = GeneLatentCrossAttentionInteraction(
                encoder.token_dim, locus_dim, interaction.hidden_dim, interaction.num_heads, interaction.dropout
            )
        elif interaction.kind == "cpg_expert_f2":
            if encoder.kind != "global_experts":
                raise ValueError("cpg_expert_f2 requires model.encoder.kind=global_experts")
            self.interaction = CpGExpertF2Interaction(
                encoder.latent_dim, encoder.expert_dim, encoder.num_experts, locus_dim,
                interaction.hidden_dim, interaction.dropout, interaction.expert_temperature,
                interaction.token_residual_scale_init,
            )
        elif interaction.kind == "cpg_module_f2":
            if encoder.kind != "module_tokens":
                raise ValueError("cpg_module_f2 requires model.encoder.kind=module_tokens")
            self.interaction = CpGTokenF2Interaction(
                encoder.latent_dim, encoder.token_dim, locus_dim, interaction.hidden_dim,
                interaction.dropout, interaction.token_residual_scale_init,
            )
        elif interaction.kind == "cpg_gene_topk_f2":
            if encoder.kind != "gene_tokens":
                raise ValueError("cpg_gene_topk_f2 requires model.encoder.kind=gene_tokens")
            self.interaction = CpGGeneTopKF2Interaction(
                encoder.latent_dim, encoder.token_dim, locus_dim, interaction.hidden_dim,
                interaction.dropout, interaction.token_residual_scale_init,
                top_k=interaction.token_top_k,
            )
        elif interaction.kind == "cpg_pretrained_f2":
            if encoder.kind != "pretrained_embedding":
                raise ValueError("cpg_pretrained_f2 requires model.encoder.kind=pretrained_embedding")
            pretrained_embedding_dim = input_dim - self.rna_encoder.raw_rna_dim
            self.interaction = PretrainedEmbeddingF2Interaction(
                encoder.latent_dim, pretrained_embedding_dim, locus_dim, interaction.hidden_dim,
                interaction.dropout, interaction.token_residual_scale_init,
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
