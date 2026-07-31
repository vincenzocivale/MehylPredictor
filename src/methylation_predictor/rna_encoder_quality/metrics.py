"""Intrinsic representation metrics for transcriptomic encoders."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

_EPS = 1e-12


def _center(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) - np.asarray(x, dtype=np.float64).mean(axis=0, keepdims=True)


def covariance_spectrum(x: np.ndarray) -> np.ndarray:
    x = _center(x)
    if len(x) < 2:
        return np.zeros(min(x.shape), dtype=np.float64)
    singular = np.linalg.svd(x, compute_uv=False, full_matrices=False)
    values = (singular**2) / max(len(x) - 1, 1)
    return np.maximum(values, 0.0)


def effective_rank_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = values.sum()
    if total <= _EPS:
        return 0.0
    p = values / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.sum() ** 2 / max(np.square(values).sum(), _EPS))


def stable_rank(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or values[0] <= _EPS:
        return 0.0
    return float(values.sum() / values[0])


def sampled_cosine(x: np.ndarray, pair_count: int, seed: int, centered: bool) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if centered:
        x = _center(x)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    first = rng.integers(0, n, size=pair_count)
    second = rng.integers(0, n, size=pair_count)
    same = first == second
    second[same] = (second[same] + 1) % n
    a, b = x[first], x[second]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    values = np.sum(a * b, axis=1) / np.maximum(denom, _EPS)
    return float(np.mean(values)), float(np.std(values))


def between_within_ratio(x: np.ndarray, cancer: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    cancer = np.asarray(cancer, dtype=object).astype(str)
    grand = x.mean(axis=0)
    between = 0.0
    within = 0.0
    residual = np.empty_like(x)
    for label in np.unique(cancer):
        mask = cancer == label
        group = x[mask]
        mean = group.mean(axis=0)
        between += float(mask.sum()) * float(np.square(mean - grand).sum())
        within += float(np.square(group - mean).sum())
        residual[mask] = group - mean
    total = max(len(x) - 1, 1)
    between /= total
    within /= total
    return float(between / max(within, _EPS)), float(between), float(within)


def residualize_groups(x: np.ndarray, cancer: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    cancer = np.asarray(cancer, dtype=object).astype(str)
    result = np.empty_like(x)
    for label in np.unique(cancer):
        mask = cancer == label
        result[mask] = x[mask] - x[mask].mean(axis=0, keepdims=True)
    return result


def geometry_metrics(x: np.ndarray, cancer: np.ndarray, pair_count: int, seed: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    spectrum = covariance_spectrum(x)
    total = max(spectrum.sum(), _EPS)
    ratio, between, within = between_within_ratio(x, cancer)
    within_spectrum = covariance_spectrum(residualize_groups(x, cancer))
    raw_cos, raw_cos_sd = sampled_cosine(x, pair_count, seed, centered=False)
    centered_cos, centered_cos_sd = sampled_cosine(x, pair_count, seed + 1, centered=True)
    norms = np.linalg.norm(x, axis=1)
    result = {
        "n_samples": float(len(x)),
        "dimension": float(x.shape[1]),
        "effective_rank": effective_rank_entropy(spectrum),
        "participation_ratio": participation_ratio(spectrum),
        "stable_rank": stable_rank(spectrum),
        "within_cancer_effective_rank": effective_rank_entropy(within_spectrum),
        "between_within_ratio": ratio,
        "between_variance": between,
        "within_variance": within,
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "mean_pair_cosine": raw_cos,
        "std_pair_cosine": raw_cos_sd,
        "mean_centered_pair_cosine": centered_cos,
        "std_centered_pair_cosine": centered_cos_sd,
    }
    for count in (1, 5, 10, 20):
        result[f"variance_top{count}"] = float(spectrum[:count].sum() / total)
    return result


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = _center(x)
    y = _center(y)
    cross = x.T @ y
    numerator = float(np.square(cross).sum())
    denominator = np.sqrt(float(np.square(x.T @ x).sum()) * float(np.square(y.T @ y).sum()))
    return float(numerator / max(denominator, _EPS))


def _neighbors(x: np.ndarray, k: int) -> np.ndarray:
    n = len(x)
    if n < 2:
        return np.empty((n, 0), dtype=np.int64)
    k = min(k, n - 1)
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    indices = model.fit(x).kneighbors(return_distance=False)
    return indices[:, 1:]


def knn_jaccard(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    ref = _neighbors(reference, k)
    cand = _neighbors(candidate, k)
    if not ref.size:
        return float("nan")
    scores = []
    for a, b in zip(ref, cand, strict=True):
        union = len(set(a.tolist()) | set(b.tolist()))
        scores.append(len(set(a.tolist()) & set(b.tolist())) / max(union, 1))
    return float(np.mean(scores))


def same_cancer_neighbor_fraction(x: np.ndarray, cancer: np.ndarray, k: int) -> float:
    neighbors = _neighbors(x, k)
    cancer = np.asarray(cancer).astype(str)
    if not neighbors.size:
        return float("nan")
    return float(np.mean(cancer[neighbors] == cancer[:, None]))


def within_cancer_knn_jaccard(reference: np.ndarray, candidate: np.ndarray, cancer: np.ndarray, k: int) -> float:
    cancer = np.asarray(cancer).astype(str)
    weighted = 0.0
    count = 0
    for label in np.unique(cancer):
        mask = np.flatnonzero(cancer == label)
        if len(mask) < 3:
            continue
        score = knn_jaccard(reference[mask], candidate[mask], min(k, len(mask) - 1))
        weighted += score * len(mask)
        count += len(mask)
    return float(weighted / count) if count else float("nan")


def neighborhood_metrics(reference: np.ndarray, candidate: np.ndarray, cancer: np.ndarray, k: int) -> dict[str, float]:
    n = len(reference)
    trust_k = min(k, max(1, (n - 1) // 2))
    result = {
        "knn_jaccard": knn_jaccard(reference, candidate, k),
        "within_cancer_knn_jaccard": within_cancer_knn_jaccard(reference, candidate, cancer, k),
        "same_cancer_neighbor_fraction": same_cancer_neighbor_fraction(candidate, cancer, k),
    }
    if n > trust_k + 1:
        result["trustworthiness"] = float(trustworthiness(reference, candidate, n_neighbors=trust_k))
        result["continuity"] = float(trustworthiness(candidate, reference, n_neighbors=trust_k))
    else:
        result["trustworthiness"] = float("nan")
        result["continuity"] = float("nan")
    return result


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), _EPS)


def stability_metrics(reference: np.ndarray, perturbed: np.ndarray, cancer: np.ndarray, k: int) -> dict[str, float]:
    if reference.shape != perturbed.shape:
        raise ValueError(f"stability views have different shapes: {reference.shape} vs {perturbed.shape}")
    cosine = row_cosine(reference, perturbed)
    distances = np.linalg.norm(reference - perturbed, axis=1)
    # Same-patient retrieval is intentionally measured against all perturbed samples.
    ref_norm = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), _EPS)
    pert_norm = perturbed / np.maximum(np.linalg.norm(perturbed, axis=1, keepdims=True), _EPS)
    similarity = ref_norm @ pert_norm.T
    ranks = np.argsort(-similarity, axis=1)
    exact = float(np.mean(ranks[:, 0] == np.arange(len(reference))))
    reciprocal = []
    for index, row in enumerate(ranks):
        position = int(np.flatnonzero(row == index)[0]) + 1
        reciprocal.append(1.0 / position)
    return {
        "same_patient_cosine_mean": float(cosine.mean()),
        "same_patient_cosine_p05": float(np.quantile(cosine, 0.05)),
        "same_patient_l2_mean": float(distances.mean()),
        "same_patient_top1_retrieval": exact,
        "same_patient_mrr": float(np.mean(reciprocal)),
        "perturbed_knn_jaccard": knn_jaccard(reference, perturbed, k),
        "perturbed_within_cancer_knn_jaccard": within_cancer_knn_jaccard(reference, perturbed, cancer, k),
    }


def _directional_consistency(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    norms = np.linalg.norm(delta, axis=1)
    valid = norms > _EPS
    if valid.sum() < 2:
        return float("nan")
    unit = delta[valid] / norms[valid, None]
    mean_direction = unit.mean(axis=0)
    mean_norm = np.linalg.norm(mean_direction)
    if mean_norm <= _EPS:
        return 0.0
    return float(np.mean(unit @ (mean_direction / mean_norm)))


def perturbation_metrics(reference: np.ndarray, perturbed: np.ndarray, cancer: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    if reference.shape != perturbed.shape:
        raise ValueError(f"perturbation views have different shapes: {reference.shape} vs {perturbed.shape}")
    delta = perturbed - reference
    norms = np.linalg.norm(delta, axis=1)
    cancer = np.asarray(cancer).astype(str)
    weighted = 0.0
    count = 0
    natural_distances: list[float] = []
    for label in np.unique(cancer):
        rows = np.flatnonzero(cancer == label)
        if len(rows) < 2:
            continue
        score = _directional_consistency(delta[rows])
        if np.isfinite(score):
            weighted += score * len(rows)
            count += len(rows)
        # A deterministic derangement avoids self-pairs, which would
        # underestimate the natural within-cancer distance scale.
        partners = np.roll(rows, 1)
        natural_distances.extend(np.linalg.norm(reference[rows] - reference[partners], axis=1).tolist())
    natural_scale = float(np.mean(natural_distances)) if natural_distances else float("nan")
    return {
        "displacement_norm_mean": float(norms.mean()),
        "displacement_norm_median": float(np.median(norms)),
        "directional_consistency": _directional_consistency(delta),
        "within_cancer_directional_consistency": float(weighted / count) if count else float("nan"),
        "same_patient_cosine": float(row_cosine(reference, perturbed).mean()),
        "natural_within_cancer_distance": natural_scale,
        "displacement_to_natural_ratio": float(norms.mean() / max(natural_scale, _EPS)) if np.isfinite(natural_scale) else float("nan"),
    }
