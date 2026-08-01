"""Closed-form Ridge head for the alternating pooler/decoder training loop.

The head has no learnable parameters of its own (no ``nn.Parameter``): its
weight, mean and scale are plain tensors recomputed by a closed-form solve
every refit. Backpropagating a loss computed from ``predict(embedding)``
therefore updates only the pooler that produced ``embedding``, never the
head itself — this is what keeps the pooler and the head strictly alternating
rather than jointly trained by SGD.
"""
from __future__ import annotations

import torch


class RidgeHead:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.x_mean: torch.Tensor | None = None
        self.x_scale: torch.Tensor | None = None
        self.weight: torch.Tensor | None = None

    def fit(self, embeddings: torch.Tensor, targets: torch.Tensor) -> None:
        embeddings = embeddings.detach().to(torch.float64)
        targets = targets.detach().to(torch.float64)
        x_mean = embeddings.mean(dim=0, keepdim=True)
        x_scale = embeddings.std(dim=0, keepdim=True).clamp_min(1e-8)
        centered = (embeddings - x_mean) / x_scale
        dim = centered.shape[1]
        gram = centered.T @ centered + self.alpha * torch.eye(dim, dtype=torch.float64, device=centered.device)
        rhs = centered.T @ targets
        weight = torch.linalg.solve(gram, rhs)
        self.x_mean = x_mean.to(torch.float32)
        self.x_scale = x_scale.to(torch.float32)
        self.weight = weight.to(torch.float32)

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError("RidgeHead.predict called before fit")
        x_mean = self.x_mean.to(embeddings.device, embeddings.dtype)
        x_scale = self.x_scale.to(embeddings.device, embeddings.dtype)
        weight = self.weight.to(embeddings.device, embeddings.dtype)
        centered = (embeddings - x_mean) / x_scale
        return centered @ weight

    def state_dict(self) -> dict[str, object]:
        return {"alpha": self.alpha, "x_mean": self.x_mean, "x_scale": self.x_scale, "weight": self.weight}
