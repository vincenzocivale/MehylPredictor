"""Experiment 0: singular-spectrum analysis of the NTv3 residual matrix."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .config import RunConfig
from .data import load_bundle
from .utils import write_json


def _logit(values: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(values) - np.log1p(-values)


def run_low_rank(
    config: RunConfig,
    output: str | Path,
    components: list[int],
    within_cancer: bool = False,
    max_samples: int | None = None,
    max_cpgs: int | None = None,
) -> dict[str, object]:
    bundle = load_bundle(config.data, seed=config.training.seed)
    try:
        sample_indices = bundle.sample_indices(config.training.train_sample_split)
        cpg_indices = bundle.cpg_indices(config.training.train_cpg_split)
        rng = np.random.default_rng(config.training.seed)
        if max_samples is not None and len(sample_indices) > max_samples:
            sample_indices = np.sort(rng.choice(sample_indices, max_samples, replace=False))
        if max_cpgs is not None and len(cpg_indices) > max_cpgs:
            cpg_indices = np.sort(rng.choice(cpg_indices, max_cpgs, replace=False))

        beta = bundle.beta(sample_indices, cpg_indices).astype(np.float64)
        prior_logit = _logit(bundle.loci.prior[cpg_indices], config.data.clip_beta_epsilon)
        residual = _logit(beta, config.data.clip_beta_epsilon) - prior_logit[None, :]
        valid = np.isfinite(beta)
        residual[~valid] = np.nan

        if within_cancer:
            cancer_types = bundle.samples.cancer_types[sample_indices]
            for cancer_type in np.unique(cancer_types):
                rows = cancer_types == cancer_type
                residual[rows] -= np.nanmean(residual[rows], axis=0, keepdims=True)
        else:
            residual -= np.nanmean(residual, axis=0, keepdims=True)
        missing_fraction = float(np.mean(~np.isfinite(residual)))
        residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

        max_rank = min(max(components), min(residual.shape) - 1)
        if max_rank < 1:
            raise ValueError("residual matrix is too small for SVD")
        _, singular_values, _ = randomized_svd(
            residual,
            n_components=max_rank,
            random_state=config.training.seed,
            n_iter=5,
        )
        explained = singular_values**2
        total_energy = float(np.sum(residual**2))
        cumulative = np.cumsum(explained) / total_energy if total_energy > 0 else np.zeros_like(explained)
        requested = {
            str(k): float(cumulative[min(k, len(cumulative)) - 1])
            for k in components
            if k >= 1
        }
        result = {
            "space": "logit(beta) - logit(NTv3 prior)",
            "centering": "within_cancer_per_locus" if within_cancer else "per_locus",
            "samples": int(len(sample_indices)),
            "cpgs": int(len(cpg_indices)),
            "missing_fraction_filled_with_zero_after_centering": missing_fraction,
            "explained_variance": requested,
            "singular_values": singular_values.tolist(),
            "sample_split": config.training.train_sample_split,
            "cpg_split": config.training.train_cpg_split,
        }
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
        return result
    finally:
        bundle.close()
