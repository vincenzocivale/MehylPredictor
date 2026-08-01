"""Patient-level readouts over frozen BulkRNABert gene tokens."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


@dataclass
class ReadoutOutput:
    embedding: torch.Tensor
    latent_tokens: torch.Tensor | None = None
    attention_weights: torch.Tensor | None = None


def _zero_last_linear(module: nn.Module) -> None:
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not linears:
        raise ValueError("module has no Linear layer to zero-initialize")
    nn.init.zeros_(linears[-1].weight)
    if linears[-1].bias is not None:
        nn.init.zeros_(linears[-1].bias)


class ContinuousTokenAugment(nn.Module):
    """Optional continuous residual added outside the frozen encoder.

    The projection starts at zero, therefore every readout still reproduces the
    frozen token representation exactly at initialization.
    """

    def __init__(self, mode: str, token_dim: int, frequencies: int = 8) -> None:
        super().__init__()
        self.mode = mode
        self.frequencies = frequencies
        if mode == "none":
            self.projection = None
        elif mode in {"raw", "bin_residual"}:
            self.projection = nn.Linear(1, token_dim, bias=False)
            nn.init.zeros_(self.projection.weight)
        elif mode == "fourier_bin_residual":
            self.projection = nn.Linear(2 * frequencies, token_dim, bias=False)
            nn.init.zeros_(self.projection.weight)
        else:
            raise ValueError(f"unsupported continuous_mode={mode!r}")

    def forward(self, tokens: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.projection is None:
            return tokens
        if self.mode == "raw":
            feature = batch["expression"].unsqueeze(-1)
        else:
            if "normalized_expression" not in batch or "token_ids" not in batch:
                raise KeyError(f"continuous_mode={self.mode} requires normalized_expression and token_ids in cache")
            values = batch["normalized_expression"]
            token_ids = batch["token_ids"].to(values.dtype)
            # BulkRNABert uses np.digitize against 64 edges on [0,1].
            center = torch.clamp((token_ids - 0.5) / 63.0, min=0.0, max=1.0)
            residual = values - center
            if self.mode == "bin_residual":
                feature = residual.unsqueeze(-1)
            else:
                frequencies = 2.0 ** torch.arange(
                    self.frequencies, device=tokens.device, dtype=tokens.dtype
                )
                angles = residual.unsqueeze(-1) * frequencies * torch.pi
                feature = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return tokens + self.projection(feature.to(tokens.dtype))


class BasePooler(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.primary_layer = config.primary_layer
        self.continuous = ContinuousTokenAugment(
            config.continuous_mode, config.output_dim, config.fourier_frequencies
        )

    def tokens(self, batch: dict[str, torch.Tensor], layer: int | None = None) -> torch.Tensor:
        layer = self.primary_layer if layer is None else layer
        tokens = batch[f"tokens_layer{layer}"]
        if layer == self.primary_layer:
            tokens = self.continuous(tokens, batch)
        return tokens

    def baseline(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Anchor on the official full-context mean over all 19,062 checkpoint
        # genes (cached by build_token_cache.py), not the mean of the cached
        # gene subset — otherwise apparent readout gains would be confounded
        # with the gene-subsampling gap instead of measuring the residual.
        key = f"full_mean_layer{self.primary_layer}"
        if key not in batch:
            raise KeyError(f"token cache is missing {key!r}; rebuild it with the full-context mean anchor")
        return batch[key]


class MeanPooler(BasePooler):
    def forward(self, batch: dict[str, torch.Tensor]) -> ReadoutOutput:
        return ReadoutOutput(self.baseline(batch))


class MeanStdPooler(BasePooler):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.residual = nn.Sequential(
            nn.LayerNorm(2 * config.output_dim),
            nn.Linear(2 * config.output_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        _zero_last_linear(self.residual)

    def forward(self, batch: dict[str, torch.Tensor]) -> ReadoutOutput:
        tokens = self.tokens(batch)
        mean = tokens.mean(dim=1)
        std = tokens.std(dim=1, unbiased=False)
        baseline = self.baseline(batch)
        return ReadoutOutput(baseline + self.residual(torch.cat([mean, std], dim=-1)))


class FixedWeightedMeanPooler(BasePooler):
    def __init__(self, config: ModelConfig, weights: np.ndarray) -> None:
        super().__init__(config)
        weights = np.maximum(np.asarray(weights, dtype=np.float32), 0.0)
        weights = weights + np.finfo(np.float32).eps
        weights /= weights.sum()
        self.register_buffer("weights", torch.from_numpy(weights))

    def forward(self, batch: dict[str, torch.Tensor]) -> ReadoutOutput:
        tokens = self.tokens(batch)
        embedding = torch.einsum("g,bgd->bd", self.weights.to(tokens.dtype), tokens)
        return ReadoutOutput(embedding, attention_weights=self.weights.expand(len(tokens), 1, -1))


class LearnedGeneWeightPooler(BasePooler):
    def __init__(self, config: ModelConfig, n_genes: int) -> None:
        super().__init__(config)
        self.logits = nn.Parameter(torch.zeros(n_genes))
        self.residual = nn.Sequential(
            nn.LayerNorm(config.output_dim),
            nn.Linear(config.output_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        _zero_last_linear(self.residual)

    def forward(self, batch: dict[str, torch.Tensor]) -> ReadoutOutput:
        tokens = self.tokens(batch)
        baseline = self.baseline(batch)
        weights = torch.softmax(self.logits, dim=0)
        weighted = torch.einsum("g,bgd->bd", weights, tokens)
        embedding = baseline + self.residual(weighted - baseline)
        return ReadoutOutput(embedding, attention_weights=weights.expand(len(tokens), 1, -1))


class GatedAttentiveStatsPooler(BasePooler):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        d, h = config.output_dim, config.hidden_dim
        self.tanh = nn.Linear(d, h)
        self.sigmoid = nn.Linear(d, h)
        self.score = nn.Linear(h, 1)
        self.residual = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, d)
        )
        _zero_last_linear(self.residual)

    def forward(self, batch: dict[str, torch.Tensor]) -> ReadoutOutput:
        tokens = self.tokens(batch)
        baseline = self.baseline(batch)
        scores = self.score(torch.tanh(self.tanh(tokens)) * torch.sigmoid(self.sigmoid(tokens))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        mean = torch.einsum("bg,bgd->bd", weights, tokens)
        variance = torch.einsum("bg,bgd->bd", weights, (tokens - mean[:, None, :]).square())
        std = torch.sqrt(torch.clamp_min(variance, 1e-8))
        embedding = baseline + self.residual(torch.cat([mean - baseline, std], dim=-1))
        return ReadoutOutput(embedding, latent_tokens=mean[:, None, :], attention_weights=weights[:, None, :])


class PMAPooler(BasePooler):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        token_dim = config.output_dim
        if token_dim % config.num_queries != 0:
            raise ValueError("output_dim must be divisible by num_queries for capacity-matched PMA")
        self.latent_dim = token_dim // config.num_queries
        if self.latent_dim % config.num_heads != 0:
            raise ValueError("PMA latent_dim must be divisible by num_heads")
        self.queries = nn.Parameter(
            torch.randn(config.num_queries, self.latent_dim) / np.sqrt(self.latent_dim)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            kdim=token_dim,
            vdim=token_dim,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.latent_dim)
        # num_queries * latent_dim == output_dim, keeping PMA1/PMA4/PMA8
        # readout width matched at 256 dimensions.
        self.residual = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, token_dim),
        )
        _zero_last_linear(self.residual)

    def attend(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.queries.unsqueeze(0).expand(len(tokens), -1, -1)
        latent, weights = self.attention(
            query, tokens, tokens, need_weights=True, average_attn_weights=False
        )
        latent = self.norm(latent + query)
        # [batch, heads, queries, genes] -> [batch, queries, genes]
        weights = weights.mean(dim=1)
        return latent, weights

    def forward(self, batch: dict[str, torch.Tensor]) -> ReadoutOutput:
        tokens = self.tokens(batch)
        baseline = self.baseline(batch)
        latent, weights = self.attend(tokens)
        embedding = baseline + self.residual(latent.reshape(len(tokens), -1))
        return ReadoutOutput(embedding, latent_tokens=latent, attention_weights=weights)


class PerceiverLitePooler(PMAPooler):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        d = self.latent_dim
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=max(config.hidden_dim, 2 * d),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.latent_encoder = nn.TransformerEncoder(layer, num_layers=config.perceiver_self_attention_layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> ReadoutOutput:
        tokens = self.tokens(batch)
        baseline = self.baseline(batch)
        latent, weights = self.attend(tokens)
        latent = self.latent_encoder(latent)
        embedding = baseline + self.residual(latent.reshape(len(tokens), -1))
        return ReadoutOutput(embedding, latent_tokens=latent, attention_weights=weights)


class ScalarMixPooler(BasePooler):
    def __init__(self, config: ModelConfig, layers: list[int]) -> None:
        super().__init__(config)
        self.layers = layers
        self.layer_logits = nn.Parameter(torch.zeros(len(layers)))
        self.residual = nn.Linear(config.output_dim, config.output_dim, bias=False)
        nn.init.zeros_(self.residual.weight)

    def forward(self, batch: dict[str, torch.Tensor]) -> ReadoutOutput:
        means = torch.stack([self.tokens(batch, layer).mean(dim=1) for layer in self.layers], dim=1)
        baseline = self.baseline(batch)
        weights = torch.softmax(self.layer_logits, dim=0)
        mixed = torch.einsum("l,bld->bd", weights, means)
        return ReadoutOutput(baseline + self.residual(mixed - baseline))


class ConcatLayerMeansPooler(BasePooler):
    def __init__(self, config: ModelConfig, layers: list[int]) -> None:
        super().__init__(config)
        self.layers = layers
        self.residual = nn.Sequential(
            nn.LayerNorm(len(layers) * config.output_dim),
            nn.Linear(len(layers) * config.output_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        _zero_last_linear(self.residual)

    def forward(self, batch: dict[str, torch.Tensor]) -> ReadoutOutput:
        means = [self.tokens(batch, layer).mean(dim=1) for layer in self.layers]
        baseline = self.baseline(batch)
        return ReadoutOutput(baseline + self.residual(torch.cat(means, dim=-1)))


class ModulePMAPooler(PMAPooler):
    def __init__(self, config: ModelConfig, gene_ids: np.ndarray) -> None:
        super().__init__(config)
        path = Path(str(config.module_membership_path))
        data = np.load(path, allow_pickle=True)
        weights = np.asarray(data["weights"], dtype=np.float32)
        module_gene_ids = np.asarray(data["gene_ids"]).astype(str)
        module_ids = np.asarray(data["module_ids"]).astype(str)
        if weights.shape == (len(module_ids), len(module_gene_ids)):
            weights = weights.T
        elif weights.shape != (len(module_gene_ids), len(module_ids)):
            raise ValueError(
                f"module weights shape {weights.shape} is incompatible with "
                f"{len(module_gene_ids)} genes and {len(module_ids)} modules"
            )
        lookup = {gene: i for i, gene in enumerate(module_gene_ids)}
        aligned = np.zeros((len(gene_ids), weights.shape[1]), dtype=np.float32)
        for i, gene in enumerate(np.asarray(gene_ids).astype(str)):
            if gene in lookup:
                aligned[i] = weights[lookup[gene]]
        normalizer = aligned.sum(axis=0, keepdims=True)
        valid = normalizer.squeeze(0) > 0
        if not valid.any():
            raise ValueError("module membership has no overlap with token-cache genes")
        aligned = aligned[:, valid] / np.maximum(normalizer[:, valid], 1e-8)
        self.register_buffer("membership", torch.from_numpy(aligned))
        self.module_ids = module_ids[valid]

    def forward(self, batch: dict[str, torch.Tensor]) -> ReadoutOutput:
        tokens = self.tokens(batch)
        baseline = self.baseline(batch)
        modules = torch.einsum("gm,bgd->bmd", self.membership.to(tokens.dtype), tokens)
        latent, weights = self.attend(modules)
        embedding = baseline + self.residual(latent.reshape(len(tokens), -1))
        return ReadoutOutput(embedding, latent_tokens=latent, attention_weights=weights)


def build_pooler(
    config: ModelConfig,
    layers: list[int],
    gene_ids: np.ndarray,
    input_variance: np.ndarray,
    input_within_variance: np.ndarray,
) -> BasePooler:
    if config.kind == "mean":
        return MeanPooler(config)
    if config.kind == "mean_std":
        return MeanStdPooler(config)
    if config.kind == "variance_weighted":
        if config.weight_mode == "total_variance":
            weights = input_variance
        elif config.weight_mode == "within_variance":
            weights = input_within_variance
        elif config.weight_mode == "inverse_variance":
            weights = 1.0 / np.maximum(input_variance, 1e-8)
        else:
            raise ValueError(f"unknown weight_mode={config.weight_mode!r}")
        return FixedWeightedMeanPooler(config, weights)
    if config.kind == "learned_gene_weights":
        return LearnedGeneWeightPooler(config, len(gene_ids))
    if config.kind == "gated_attentive_stats":
        return GatedAttentiveStatsPooler(config)
    if config.kind == "pma":
        return PMAPooler(config)
    if config.kind == "scalar_mix":
        return ScalarMixPooler(config, layers)
    if config.kind == "concat_layer_means":
        return ConcatLayerMeansPooler(config, layers)
    if config.kind == "perceiver_lite":
        return PerceiverLitePooler(config)
    if config.kind == "module_pma":
        return ModulePMAPooler(config, gene_ids)
    raise ValueError(f"unsupported pooler kind={config.kind!r}")


def attention_diagnostics(output: ReadoutOutput) -> dict[str, torch.Tensor]:
    diagnostics: dict[str, torch.Tensor] = {}
    if output.attention_weights is not None:
        weights = torch.clamp_min(output.attention_weights, 1e-12)
        entropy = -(weights * weights.log()).sum(dim=-1)
        diagnostics["attention_entropy"] = entropy
        diagnostics["effective_genes"] = entropy.exp()
        diagnostics["attention_max"] = weights.max(dim=-1).values
    if output.latent_tokens is not None and output.latent_tokens.shape[1] > 1:
        latent = F.normalize(output.latent_tokens, dim=-1)
        cosine = latent @ latent.transpose(1, 2)
        k = cosine.shape[1]
        mask = ~torch.eye(k, dtype=torch.bool, device=cosine.device)
        diagnostics["latent_pairwise_cosine"] = cosine[:, mask].reshape(len(cosine), -1)
    return diagnostics
