"""Expected Gradients (Erion et al. 2021) attribution of RNA genes onto the
RNA-driven residual ``raw_delta`` (not the final ``beta``).

Why ``raw_delta`` and not ``beta``: ``beta_hat = sigmoid(logit(mu_i) + sigma_i
* raw_delta)`` (``VarianceNormalizedResidualModel``/``RNAMethylationPredictor``,
see models.py) already has a CpG-only component (``mu_i``, the locus prior)
baked in. Attributing ``beta`` would mix "this locus is usually methylated"
with "this sample's RNA pushed it further" into one number. ``raw_delta`` is
exactly the model's RNA+CpG interaction term before the prior is added back
in, so attributing it isolates the RNA effect the model actually learned.

Why Expected Gradients instead of vanilla Integrated Gradients with a
zero baseline: the RNA encoder starts with a ``LayerNorm`` (canonical
default, ``EncoderConfig.layer_norm=True``), and z-scored RNA input has
mean 0 by construction (``RNACache``/``rna_stats.npz``), so the "obvious"
baseline is the all-zero vector -- but ``LayerNorm`` is numerically singular
at exactly zero (its variance term vanishes, so the ``eps`` floor dominates
and the local gradient blows up right at the baseline). Measured directly
against this repo's encoder: a straight-line path from an all-zero baseline
needs on the order of 10^4-10^5 IG steps to converge, because almost the
entire attribution mass sits in an infinitesimally narrow region of the path
right next to the singularity. Averaging Integrated Gradients over several
straight-line paths from *real* background RNA samples (Expected Gradients)
sidesteps the singularity entirely -- none of those paths pass through the
origin -- and converges cleanly with a normal step count (verified: paths
from a random nonzero vector converge within ~50-500 steps for the same
network where a zero baseline had not converged after 5000). This is also a
better-posed question biologically: "how does this sample differ from a
background of real patients" rather than from a hypothetical zero-expression
patient.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _raw_delta(model: nn.Module, rna: torch.Tensor, loci: torch.Tensor) -> torch.Tensor:
    """Paired (not Cartesian) raw_delta: element ``i`` uses ``rna[i]`` with ``loci[i]``.

    Every interaction module in models.py (``ProductInteraction``,
    ``FiLMInteraction``, ``CrossAttentionInteraction``, ``BilinearInteraction``,
    ``GlobalShiftInteraction``) shares the same broadcasting contract:
    ``interaction(rna_repr, loci) -> (batch, n_loci)``. Calling it with
    ``batch == n_loci == len(loci)`` and taking the diagonal gives the paired
    score without an interaction-specific pairwise implementation, so this
    helper -- and everything built on it -- works unmodified across
    ``interaction.kind`` and across ``RNA2DNAmModel`` /
    ``VarianceNormalizedResidualModel`` / ``DirectPredictionModel`` (anything
    exposing ``.rna_encoder`` + ``.interaction``; ``FeatureFusionLocusCLSModel``
    does not and is out of scope).
    """
    representation = model.rna_encoder(rna)
    full = model.interaction(representation, loci)  # (n_loci, n_loci)
    return torch.diagonal(full)


@dataclass
class GeneAttribution:
    """Per-locus, per-gene signed attribution for one sample.

    ``attributions``: (n_loci, rna_dim) -- gene-level contribution to
    ``raw_delta`` at each locus, averaged over background baselines, in the
    RNA input's own units (z-scored gene expression), signed (positive =
    pushes the residual up relative to the background).
    ``raw_delta`` / ``raw_delta_baseline``: (n_loci,) the model's actual
    ``raw_delta`` at the input RNA vector, and averaged over the background
    baselines.
    ``convergence_gap``: (n_loci,) ``|sum(attributions) - (raw_delta -
    raw_delta_baseline)|`` -- the method's own diagnostic (completeness
    axiom, averaged over baselines). Small relative to ``raw_delta``'s scale
    means the step count was sufficient; large means call again with more
    ``steps`` (or check whether the baselines are pathologically close to a
    normalization-layer singularity -- see module docstring).
    """

    attributions: torch.Tensor
    raw_delta: torch.Tensor
    raw_delta_baseline: torch.Tensor
    convergence_gap: torch.Tensor


@torch.enable_grad()
def integrated_gradients_rna(
    model: nn.Module,
    rna: torch.Tensor,
    loci: torch.Tensor,
    *,
    baselines: torch.Tensor,
    steps: int = 50,
) -> GeneAttribution:
    """Expected Gradients of ``raw_delta`` wrt one sample's RNA vector,
    evaluated separately (paired) against each row of ``loci``.

    ``rna``: (rna_dim,) single sample, already z-scored (``RNACache`` convention).
    ``loci``: (n_loci, locus_dim) CpG embeddings to explain this sample against.
    ``baselines``: (n_baselines, rna_dim) real background RNA vectors (e.g. a
    handful of other patients' z-scored RNA rows) -- see module docstring for
    why this must not default to the all-zero vector. Attributions are the
    average of vanilla Integrated Gradients run against each baseline row.

    Cost is O(n_baselines * n_loci^2 * steps) (the diagonal trick above
    materializes the full (n_loci, n_loci) interaction output each step) --
    fine for the handful-to-low-hundreds of loci an explainability query
    targets and a small (~8-16) baseline set, not meant for scoring thousands
    of loci at once.
    """
    was_training = model.training
    model.eval()
    try:
        n_loci = loci.shape[0]
        rna = rna.detach().to(torch.float32)
        baselines = baselines.detach().to(torch.float32)
        if baselines.ndim != 2 or baselines.shape[1] != rna.shape[0]:
            raise ValueError(f"baselines must be (n_baselines, {rna.shape[0]}), got {tuple(baselines.shape)}")

        rna_rep = rna.unsqueeze(0).expand(n_loci, -1)
        with torch.no_grad():
            raw_delta = _raw_delta(model, rna_rep, loci)

        attribution_sum = torch.zeros(n_loci, rna.shape[0], device=rna.device)
        raw_delta_baseline_sum = torch.zeros(n_loci, device=rna.device)
        for baseline in baselines:
            baseline_rep = baseline.unsqueeze(0).expand(n_loci, -1)
            diff = rna_rep - baseline_rep

            grad_sum = torch.zeros_like(rna_rep)
            for step in range(1, steps + 1):
                alpha = step / steps
                scaled = (baseline_rep + alpha * diff).clone().requires_grad_(True)
                out = _raw_delta(model, scaled, loci)
                (grad,) = torch.autograd.grad(out.sum(), scaled)
                grad_sum = grad_sum + grad
            attribution_sum = attribution_sum + diff * (grad_sum / steps)
            with torch.no_grad():
                raw_delta_baseline_sum = raw_delta_baseline_sum + _raw_delta(model, baseline_rep, loci)

        n_baselines = baselines.shape[0]
        attributions = attribution_sum / n_baselines
        raw_delta_baseline = raw_delta_baseline_sum / n_baselines
        convergence_gap = (attributions.sum(-1) - (raw_delta - raw_delta_baseline)).abs()

        return GeneAttribution(
            attributions=attributions.detach(),
            raw_delta=raw_delta.detach(),
            raw_delta_baseline=raw_delta_baseline.detach(),
            convergence_gap=convergence_gap.detach(),
        )
    finally:
        model.train(was_training)
