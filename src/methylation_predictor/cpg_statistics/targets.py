"""Leakage-aware multi-technology target construction for CpGStatisticsPredictor."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..scopes import filter_cpg_ids, resolve_scope, scope_protocol

TARGET_POLICIES = ("sample_weighted", "technology_balanced")
AUX_SAMPLE_POLICIES = ("exclude_array_validation", "all_auxiliary")


@dataclass(slots=True)
class SourceMoments:
    count: np.ndarray
    beta_sum: np.ndarray
    logit_sum: np.ndarray
    logit_sumsq: np.ndarray

    @classmethod
    def zeros(cls, n: int) -> "SourceMoments":
        return cls(
            np.zeros(n, np.int64),
            np.zeros(n, np.float64),
            np.zeros(n, np.float64),
            np.zeros(n, np.float64),
        )


def _logit(x: np.ndarray, epsilon: float) -> np.ndarray:
    x = np.clip(x, epsilon, 1.0 - epsilon)
    return np.log(x) - np.log1p(-x)


def _selected_rows(source, *, source_name: str, protocol, aux_sample_policy: str) -> np.ndarray:
    if source_name == "array":
        return source.rows_of_samples(protocol.array_train_sample_idx)
    rows = np.arange(source.n_rows, dtype=np.int64)
    if aux_sample_policy == "all_auxiliary":
        return rows
    if aux_sample_policy != "exclude_array_validation":
        raise ValueError(f"unknown aux sample policy {aux_sample_policy!r}")
    # Auxiliary patients without Array measurements are legitimate training
    # observations.  Only patients explicitly assigned to the frozen Array
    # validation split are excluded if they occur in an auxiliary technology.
    return rows[~np.isin(source.sample_idx, protocol.array_val_sample_idx)]


def _moments_for_source(
    source,
    source_name: str,
    cpg_ids: np.ndarray,
    *,
    protocol,
    aux_sample_policy: str,
    epsilon: float,
    cpg_chunk: int,
) -> SourceMoments:
    moments = SourceMoments.zeros(len(cpg_ids))
    present = source.has_cpg(cpg_ids)
    present_positions = np.flatnonzero(present)
    if not len(present_positions):
        return moments
    rows = _selected_rows(
        source, source_name=source_name, protocol=protocol, aux_sample_policy=aux_sample_policy
    )
    if not len(rows):
        return moments

    for start in range(0, len(present_positions), cpg_chunk):
        local_pos = present_positions[start : start + cpg_chunk]
        local_ids = cpg_ids[local_pos]
        beta = source.block(rows, local_ids)
        finite = np.isfinite(beta)
        safe_beta = np.where(finite, beta, 0.0).astype(np.float64, copy=False)
        logit = np.where(finite, _logit(np.where(finite, beta, 0.5), epsilon), 0.0)
        moments.count[local_pos] = finite.sum(axis=0, dtype=np.int64)
        moments.beta_sum[local_pos] = safe_beta.sum(axis=0, dtype=np.float64)
        moments.logit_sum[local_pos] = logit.sum(axis=0, dtype=np.float64)
        moments.logit_sumsq[local_pos] = np.square(logit, dtype=np.float64).sum(axis=0)
    return moments


def _combine_sample_weighted(
    source_moments: dict[str, SourceMoments], *, epsilon: float, sigma_floor: float
) -> tuple[np.ndarray, np.ndarray]:
    count = sum((x.count for x in source_moments.values()), start=np.zeros_like(next(iter(source_moments.values())).count))
    beta_sum = sum((x.beta_sum for x in source_moments.values()), start=np.zeros_like(next(iter(source_moments.values())).beta_sum))
    logit_sum = sum((x.logit_sum for x in source_moments.values()), start=np.zeros_like(next(iter(source_moments.values())).logit_sum))
    logit_sumsq = sum((x.logit_sumsq for x in source_moments.values()), start=np.zeros_like(next(iter(source_moments.values())).logit_sumsq))
    if np.any(count <= 0):
        raise RuntimeError(f"{int((count <= 0).sum())} CpGs have no finite observation across configured technologies")
    mu = np.clip(beta_sum / count, epsilon, 1.0 - epsilon)
    mean_logit = logit_sum / count
    var_logit = np.maximum(logit_sumsq / count - mean_logit * mean_logit, sigma_floor**2)
    return mu.astype(np.float32), np.sqrt(var_logit).astype(np.float32)


def _combine_technology_balanced(
    source_moments: dict[str, SourceMoments], *, epsilon: float, sigma_floor: float
) -> tuple[np.ndarray, np.ndarray]:
    means_beta = []
    means_logit = []
    second_logit = []
    available = []
    for moments in source_moments.values():
        ok = moments.count > 0
        denom = np.maximum(moments.count, 1)
        means_beta.append(np.where(ok, moments.beta_sum / denom, 0.0))
        means_logit.append(np.where(ok, moments.logit_sum / denom, 0.0))
        second_logit.append(np.where(ok, moments.logit_sumsq / denom, 0.0))
        available.append(ok.astype(np.float64))
    avail = np.stack(available)
    denom = avail.sum(axis=0)
    if np.any(denom <= 0):
        raise RuntimeError(f"{int((denom <= 0).sum())} CpGs have no finite technology")
    mu = (np.stack(means_beta) * avail).sum(axis=0) / denom
    mean_logit = (np.stack(means_logit) * avail).sum(axis=0) / denom
    second = (np.stack(second_logit) * avail).sum(axis=0) / denom
    var = np.maximum(second - mean_logit * mean_logit, sigma_floor**2)
    return np.clip(mu, epsilon, 1.0 - epsilon).astype(np.float32), np.sqrt(var).astype(np.float32)


def build_statistics_targets(
    *,
    canonical_root: str | Path,
    registry: str | Path,
    scope: str,
    output: str | Path,
    policy: str = "sample_weighted",
    aux_sample_policy: str = "exclude_array_validation",
    sources: tuple[str, ...] = ("array", "epic", "wgbs"),
    epsilon: float = 1e-4,
    sigma_floor: float = 0.01,
    cpg_chunk: int = 2048,
) -> dict[str, object]:
    """Create reusable static-statistic labels for one genomic scope.

    Targets for held-out loci are labels only: the predictor is fitted solely on
    official train CpGs.  Methylation from official Array validation *patients*
    is never used.  Auxiliary observations from patients not represented by the
    Array split remain available; auxiliary rows matching explicit Array-val
    patients are excluded by default.
    """
    if policy not in TARGET_POLICIES:
        raise ValueError(f"policy must be one of {TARGET_POLICIES}")
    if aux_sample_policy not in AUX_SAMPLE_POLICIES:
        raise ValueError(f"aux_sample_policy must be one of {AUX_SAMPLE_POLICIES}")
    resolve_scope(scope)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    # Import pyarrow before opening h5py-backed bundle on the target host.
    import pyarrow  # noqa: F401
    from ..tcga_canonical import TCGACanonicalBundle

    with TCGACanonicalBundle.from_root(canonical_root) as bundle:
        protocol = scope_protocol(scope, bundle, canonical_root=canonical_root)
        universe = np.unique(
            np.concatenate([protocol.array_train_cpg_idx, protocol.array_val_cpg_idx])
        ).astype(np.int64)
        universe = filter_cpg_ids(universe, scope, registry)
        train_mask = np.isin(universe, protocol.array_train_cpg_idx)
        val_mask = np.isin(universe, protocol.array_val_cpg_idx)
        if np.any(train_mask & val_mask) or not np.all(train_mask | val_mask):
            raise RuntimeError("scope CpG universe does not partition into official train/validation loci")

        source_moments: dict[str, SourceMoments] = {}
        for source_name in sources:
            if source_name not in bundle.sources:
                raise KeyError(f"canonical bundle has no source {source_name!r}")
            print(f"[cpg-statistics] accumulating {source_name} moments for {len(universe):,} CpGs", flush=True)
            source_moments[source_name] = _moments_for_source(
                bundle.sources[source_name],
                source_name,
                universe,
                protocol=protocol,
                aux_sample_policy=aux_sample_policy,
                epsilon=epsilon,
                cpg_chunk=cpg_chunk,
            )

        if policy == "sample_weighted":
            mu, sigma = _combine_sample_weighted(source_moments, epsilon=epsilon, sigma_floor=sigma_floor)
        else:
            mu, sigma = _combine_technology_balanced(source_moments, epsilon=epsilon, sigma_floor=sigma_floor)

    np.save(out / "cpg_idx.npy", universe)
    np.save(out / "target_mu.npy", mu)
    np.save(out / "target_sigma.npy", sigma)
    np.save(out / "official_train_mask.npy", train_mask)
    np.save(out / "official_val_mask.npy", val_mask)
    counts = {name: moments.count.astype(np.int32) for name, moments in source_moments.items()}
    np.savez_compressed(out / "source_counts.npz", **counts)
    manifest = {
        "schema_version": 1,
        "scope": scope,
        "sources": list(sources),
        "target_policy": policy,
        "aux_sample_policy": aux_sample_policy,
        "epsilon": epsilon,
        "sigma_floor": sigma_floor,
        "cpgs": int(len(universe)),
        "official_train_cpgs": int(train_mask.sum()),
        "official_val_cpgs": int(val_mask.sum()),
        "source_observation_totals": {name: int(m.count.sum()) for name, m in source_moments.items()},
        "definition": {
            "mu": "beta-space mean across configured technology policy",
            "sigma": "std of clipped logit(beta) across configured technology policy",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
