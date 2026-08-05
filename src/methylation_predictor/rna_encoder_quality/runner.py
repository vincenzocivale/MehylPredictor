"""End-to-end intrinsic quality audit for frozen RNA encoder representations."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from .config import QualityConfig, load_config
from .io import AlignedData, load_aligned_data, load_stability_view, load_token_data
from .metrics import (
    geometry_metrics,
    linear_cka,
    neighborhood_metrics,
    perturbation_metrics,
    residualize_groups,
    stability_metrics,
)
from .probes import fit_ridge_probe, residualize_using_training_means, training_group_means
from .token_audit import audit_tokens


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _standardize_from_train(x: np.ndarray, train: np.ndarray) -> np.ndarray:
    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale < 1e-8] = 1.0
    return ((x - mean) / scale).astype(np.float32)


def _representations(data: AlignedData, config: QualityConfig) -> dict[str, np.ndarray]:
    train = data.split_indices["train"]
    rna = _standardize_from_train(data.rna_selected, train)
    output = {f"encoder_{name}": values.astype(np.float32) for name, values in data.embeddings.items()}

    # Probe capacity must not depend on the native embedding width.  Every encoder
    # is therefore also evaluated after a train-only PCA projection to the same
    # dimensionality.  Native representations remain available as a sensitivity
    # analysis, but the matched representation is the preregistered primary one.
    for name, values in data.embeddings.items():
        values = _standardize_from_train(values, train)
        for requested in config.analysis.matched_encoder_dimensions:
            n_components = min(int(requested), len(train) - 1, values.shape[1])
            if n_components < 1:
                continue
            model = PCA(n_components=n_components, svd_solver="randomized", random_state=config.analysis.seed)
            model.fit(values[train])
            output[f"encoder_{name}__pca{n_components}"] = model.transform(values).astype(np.float32)

    for dimension in config.analysis.pca_dimensions:
        n_components = min(int(dimension), len(train) - 1, rna.shape[1])
        if n_components < 1:
            continue
        model = PCA(n_components=n_components, svd_solver="randomized", random_state=config.analysis.seed)
        model.fit(rna[train])
        output[f"pca_{n_components}"] = model.transform(rna).astype(np.float32)
    rp_dim = min(int(config.analysis.random_projection_dim), rna.shape[1])
    projector = GaussianRandomProjection(n_components=rp_dim, random_state=config.analysis.seed)
    projector.fit(rna[train])
    output[f"random_projection_{rp_dim}"] = projector.transform(rna).astype(np.float32)
    return output


def _geometry(data: AlignedData, representations: dict[str, np.ndarray], config: QualityConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split_name in config.analysis.geometry_splits:
        rows = np.arange(len(data.sample_ids)) if split_name == "all" else data.split_indices.get(split_name, np.array([], dtype=int))
        if len(rows) < 2:
            continue
        for representation, values in representations.items():
            metrics = geometry_metrics(
                values[rows], data.cancer_types[rows], config.analysis.pair_sample_count, config.analysis.seed
            )
            records.append({"representation": representation, "split": split_name, **metrics})
    return pd.DataFrame.from_records(records)


def _cka(data: AlignedData, representations: dict[str, np.ndarray]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split_name in ("validation", "test"):
        rows = data.split_indices[split_name]
        names = sorted(representations)
        for residualized in (False, True):
            values = {
                name: residualize_groups(representations[name][rows], data.cancer_types[rows])
                if residualized else representations[name][rows]
                for name in names
            }
            for left_index, left in enumerate(names):
                for right in names[left_index:]:
                    records.append({
                        "split": split_name,
                        "residualized_within_cancer": residualized,
                        "left": left,
                        "right": right,
                        "cka": linear_cka(values[left], values[right]),
                    })
    return pd.DataFrame.from_records(records)


def _probe_targets(data: AlignedData) -> dict[str, np.ndarray]:
    train = data.split_indices["train"]
    global_mean, group_means = training_group_means(data.rna_selected[train], data.cancer_types[train])
    residual = residualize_using_training_means(data.rna_selected, data.cancer_types, global_mean, group_means)
    return {"total_rna": data.rna_selected, "within_cancer_rna": residual}


def _reconstruction(data: AlignedData, representations: dict[str, np.ndarray], config: QualityConfig) -> pd.DataFrame:
    train, validation, test = (data.split_indices[name] for name in ("train", "validation", "test"))
    targets = _probe_targets(data)
    records: list[dict[str, Any]] = []
    for representation, values in representations.items():
        for target_name, target in targets.items():
            result = fit_ridge_probe(
                values[train], values[validation], values[test],
                target[train], target[validation], target[test], config.analysis.ridge_alphas,
                cv_folds=config.analysis.ridge_cv_folds,
                seed=config.analysis.seed,
                cv_groups=data.cancer_types[train],
            )
            records.append({"representation": representation, "target": target_name, **asdict(result)})
    return pd.DataFrame.from_records(records)


def _neighborhood(data: AlignedData, representations: dict[str, np.ndarray], config: QualityConfig) -> pd.DataFrame:
    test = data.split_indices["test"]
    rna = _standardize_from_train(data.rna_selected, data.split_indices["train"])
    n_components = min(256, len(data.split_indices["train"]) - 1, rna.shape[1])
    pca = PCA(n_components=max(1, n_components), svd_solver="randomized", random_state=config.analysis.seed)
    pca.fit(rna[data.split_indices["train"]])
    reference = pca.transform(rna[test])
    records: list[dict[str, Any]] = []
    train = data.split_indices["train"]
    for name, values in representations.items():
        candidate = _standardize_from_train(values, train)[test]
        metrics = neighborhood_metrics(reference, candidate, data.cancer_types[test], config.analysis.knn_k)
        records.append({"representation": name, **metrics})
    return pd.DataFrame.from_records(records)


def _stability(data: AlignedData, config: QualityConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for view in config.stability_views:
        loaded = load_stability_view(view, data.sample_ids)
        for layer, perturbed in loaded.items():
            if layer not in data.embeddings:
                raise ValueError(f"stability view {view.name!r} contains unknown layer {layer!r}")
            metrics = stability_metrics(data.embeddings[layer], perturbed, data.cancer_types, config.analysis.knn_k)
            records.append({"view": view.name, "layer": layer, **metrics})
    return pd.DataFrame.from_records(records)


def _perturbations(data: AlignedData, config: QualityConfig) -> pd.DataFrame:
    loaded = {view.name: load_stability_view(view, data.sample_ids) for view in config.perturbation_views}
    records: list[dict[str, Any]] = []
    for view in config.perturbation_views:
        for layer, perturbed in loaded[view.name].items():
            if layer not in data.embeddings:
                raise ValueError(f"perturbation view {view.name!r} contains unknown layer {layer!r}")
            metrics = perturbation_metrics(data.embeddings[layer], perturbed, data.cancer_types)
            record: dict[str, Any] = {"view": view.name, "layer": layer, "control": view.control, **metrics}
            if view.control is not None:
                if layer not in loaded[view.control]:
                    raise ValueError(f"control {view.control!r} lacks layer {layer!r}")
                control_metrics = perturbation_metrics(
                    data.embeddings[layer], loaded[view.control][layer], data.cancer_types
                )
                record["content_specific_displacement"] = (
                    metrics["displacement_norm_mean"] - control_metrics["displacement_norm_mean"]
                )
                record["displacement_vs_control_ratio"] = metrics["displacement_norm_mean"] / max(
                    control_metrics["displacement_norm_mean"], 1e-12
                )
                record["directional_consistency_gain_vs_control"] = (
                    metrics["directional_consistency"] - control_metrics["directional_consistency"]
                )
            records.append(record)
    return pd.DataFrame.from_records(records)


def _token_quality(data: AlignedData, config: QualityConfig) -> pd.DataFrame:
    if config.token_embeddings is None:
        return pd.DataFrame()
    token_data = load_token_data(config.token_embeddings)
    sample_lookup = {str(value): index for index, value in enumerate(data.sample_ids)}
    missing = [str(value) for value in token_data.sample_ids if str(value) not in sample_lookup]
    if missing:
        raise ValueError(f"token sample includes IDs absent from aligned data: {missing[:5]}")
    rows = np.asarray([sample_lookup[str(value)] for value in token_data.sample_ids], dtype=int)
    for layer in token_data.layers:
        if layer in data.embeddings:
            token_data.pooled_layers[layer] = data.embeddings[layer][rows]
    return audit_tokens(
        token_data, data.sample_ids, data.splits, data.cancer_types,
        config.analysis.token_ridge_alphas, config.analysis.token_global_pc_count,
    )


def _encoder_ranking(
    config: QualityConfig,
    reconstruction: pd.DataFrame,
    neighborhood: pd.DataFrame,
    geometry: pd.DataFrame,
) -> pd.DataFrame:
    suffix = f"__pca{config.analysis.primary_encoder_dimension}"
    primary = reconstruction[
        reconstruction["representation"].str.startswith("encoder_")
        & reconstruction["representation"].str.endswith(suffix)
        & (reconstruction["target"] == "within_cancer_rna")
    ].copy()
    if primary.empty:
        raise ValueError(
            "no encoder representation matches the preregistered dimension "
            f"{config.analysis.primary_encoder_dimension}; check native widths and "
            "analysis.matched_encoder_dimensions"
        )
    primary = primary.rename(columns={
        "train_cv_global_r2": "within_cancer_train_cv_global_r2",
        "train_cv_mean_gene_pearson": "within_cancer_train_cv_mean_gene_pearson",
        "validation_global_r2": "within_cancer_validation_global_r2",
        "validation_mean_gene_pearson": "within_cancer_validation_mean_gene_pearson",
        "global_r2": "within_cancer_test_global_r2",
        "variance_weighted_r2": "within_cancer_test_variance_weighted_r2",
        "median_gene_r2": "within_cancer_test_median_gene_r2",
        "mean_gene_pearson": "within_cancer_test_mean_gene_pearson",
    })
    keep = [
        "representation", "alpha", "train_cv_mse", "validation_mse", "test_mse",
        "within_cancer_train_cv_global_r2", "within_cancer_train_cv_mean_gene_pearson",
        "within_cancer_validation_global_r2", "within_cancer_validation_mean_gene_pearson",
        "within_cancer_test_global_r2", "within_cancer_test_variance_weighted_r2",
        "within_cancer_test_median_gene_r2", "within_cancer_test_mean_gene_pearson",
    ]
    ranking = primary[keep]
    total = reconstruction[
        reconstruction["representation"].isin(ranking["representation"])
        & (reconstruction["target"] == "total_rna")
    ][[
        "representation", "train_cv_global_r2", "validation_global_r2", "global_r2",
        "train_cv_mean_gene_pearson", "validation_mean_gene_pearson", "mean_gene_pearson",
    ]].rename(columns={
        "train_cv_global_r2": "total_rna_train_cv_global_r2",
        "validation_global_r2": "total_rna_validation_global_r2",
        "global_r2": "total_rna_test_global_r2",
        "train_cv_mean_gene_pearson": "total_rna_train_cv_mean_gene_pearson",
        "validation_mean_gene_pearson": "total_rna_validation_mean_gene_pearson",
        "mean_gene_pearson": "total_rna_test_mean_gene_pearson",
    })
    ranking = ranking.merge(total, on="representation", how="left")
    if not neighborhood.empty:
        ranking = ranking.merge(neighborhood, on="representation", how="left")
    test_geometry = (
        geometry[geometry["split"] == "test"].copy()
        if not geometry.empty and "split" in geometry.columns
        else pd.DataFrame()
    )
    if not test_geometry.empty:
        columns = [
            "representation", "effective_rank", "within_cancer_effective_rank",
            "between_within_ratio", "variance_top10",
        ]
        ranking = ranking.merge(test_geometry[columns], on="representation", how="left")
    ranking = ranking.sort_values(
        [
            "within_cancer_train_cv_global_r2",
            "within_cancer_validation_global_r2",
            "within_cancer_train_cv_mean_gene_pearson",
        ],
        ascending=False,
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    validation_order = ranking.sort_values(
        ["within_cancer_validation_global_r2", "within_cancer_validation_mean_gene_pearson"],
        ascending=False,
    )["representation"].tolist()
    cv_order = ranking["representation"].tolist()
    ranking["validation_confirms_cv_winner"] = bool(validation_order and cv_order and validation_order[0] == cv_order[0])
    return ranking


def _summary(
    config: QualityConfig,
    geometry: pd.DataFrame,
    reconstruction: pd.DataFrame,
    neighborhood: pd.DataFrame,
    stability: pd.DataFrame,
    perturbations: pd.DataFrame,
    token_quality: pd.DataFrame,
    ranking: pd.DataFrame,
) -> dict[str, Any]:
    winner = None if ranking.empty else ranking.iloc[0]
    return {
        "run_name": config.run_name,
        "selection_rule": (
            f"highest training-CV within-cancer RNA global R2 after train-only PCA to "
            f"{config.analysis.primary_encoder_dimension} dimensions; validation is confirmation only; "
            "test is exploratory"
        ),
        "best_encoder": None if winner is None else str(winner["representation"]),
        "best_encoder_train_cv_within_cancer_global_r2": None if winner is None else float(
            winner["within_cancer_train_cv_global_r2"]
        ),
        "best_encoder_validation_within_cancer_global_r2": None if winner is None else float(
            winner["within_cancer_validation_global_r2"]
        ),
        "best_encoder_test_within_cancer_global_r2_exploratory": None if winner is None else float(
            winner["within_cancer_test_global_r2"]
        ),
        "validation_confirms_cv_winner": None if winner is None else bool(winner["validation_confirms_cv_winner"]),
        "geometry_rows": int(len(geometry)),
        "reconstruction_rows": int(len(reconstruction)),
        "neighborhood_rows": int(len(neighborhood)),
        "stability_rows": int(len(stability)),
        "perturbation_rows": int(len(perturbations)),
        "token_quality_rows": int(len(token_quality)),
    }


def _report(summary: dict[str, Any], ranking: pd.DataFrame, reconstruction: pd.DataFrame, output: Path) -> None:
    lines = [
        f"# RNA encoder quality — {summary['run_name']}",
        "",
        "This audit is transcriptome-only: it does not load methylation targets, CpG embeddings, an F2 checkpoint, or a methylation regressor.",
        "",
        "## Preregistered decision rule",
        "",
        summary["selection_rule"],
        "",
        "## Primary result",
        "",
        f"- Best frozen encoder representation: `{summary['best_encoder']}`.",
        f"- Training-CV within-cancer global R²: `{summary['best_encoder_train_cv_within_cancer_global_r2']}`.",
        f"- Validation within-cancer global R²: `{summary['best_encoder_validation_within_cancer_global_r2']}`.",
        f"- Validation confirms the CV winner: `{summary['validation_confirms_cv_winner']}`.",
        f"- Test within-cancer global R² (exploratory): `{summary['best_encoder_test_within_cancer_global_r2_exploratory']}`.",
        "",
        "## Matched encoder ranking",
        "",
        "```csv",
        ranking.to_csv(index=False).strip(),
        "```",
        "",
        "## Complete reconstruction table",
        "",
        "```csv",
        reconstruction.sort_values(["target", "global_r2"], ascending=[True, False]).to_csv(index=False).strip(),
        "```",
        "",
        "Native-width and test results are sensitivity analyses. The winner is determined only from training-CV on the matched-dimensionality rows in `encoder_ranking.csv`; validation may confirm or reject transportability but is never used to tune the Ridge penalty.",
    ]
    output.write_text("\n".join(lines) + "\n")

def validate_inputs(config: QualityConfig) -> dict[str, Any]:
    data = load_aligned_data(config)
    result = {
        "n_samples": int(len(data.sample_ids)),
        "n_train": int(len(data.split_indices["train"])),
        "n_validation": int(len(data.split_indices["validation"])),
        "n_test": int(len(data.split_indices["test"])),
        "n_selected_genes": int(len(data.selected_gene_ids)),
        "embedding_layers": {name: list(values.shape) for name, values in data.embeddings.items()},
    }
    if config.token_embeddings is not None:
        token = load_token_data(config.token_embeddings)
        sample_lookup = {str(value): index for index, value in enumerate(data.sample_ids)}
        missing = [str(value) for value in token.sample_ids if str(value) not in sample_lookup]
        if missing:
            raise ValueError(f"token sample includes IDs absent from aligned data: {missing[:5]}")
        token_splits = data.splits[[sample_lookup[str(value)] for value in token.sample_ids]]
        missing_splits = [name for name in ("train", "validation", "test") if not np.any(token_splits == name)]
        if missing_splits:
            raise ValueError(f"token sample lacks required splits: {missing_splits}")
        result["token_sample_shape"] = list(token.expression.shape)
        result["token_split_counts"] = {name: int(np.sum(token_splits == name)) for name in ("train", "validation", "test")}
        result["token_layers"] = {name: list(values.shape) for name, values in token.layers.items()}
    for view in config.stability_views:
        load_stability_view(view, data.sample_ids)
    for view in config.perturbation_views:
        load_stability_view(view, data.sample_ids)
    return result


def run_quality_audit(config_path: str | Path) -> Path:
    config = load_config(config_path)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_aligned_data(config)
    representations = _representations(data, config)
    geometry = _geometry(data, representations, config)
    cka = _cka(data, representations)
    reconstruction = _reconstruction(data, representations, config)
    neighborhood = _neighborhood(data, representations, config)
    stability = _stability(data, config)
    perturbations = _perturbations(data, config)
    token_quality = _token_quality(data, config)

    selected = pd.DataFrame({
        "gene_index": data.selected_gene_indices,
        "gene_id": data.selected_gene_ids.astype(str),
    })
    selected.to_csv(output / "selected_genes.csv", index=False)
    geometry.to_csv(output / "geometry.csv", index=False)
    cka.to_csv(output / "cka.csv", index=False)
    reconstruction.to_csv(output / "reconstruction.csv", index=False)
    neighborhood.to_csv(output / "neighborhood.csv", index=False)
    stability.to_csv(output / "stability.csv", index=False)
    perturbations.to_csv(output / "perturbation.csv", index=False)
    token_quality.to_csv(output / "token_quality.csv", index=False)

    ranking = _encoder_ranking(config, reconstruction, neighborhood, geometry)
    ranking.to_csv(output / "encoder_ranking.csv", index=False)
    summary = _summary(config, geometry, reconstruction, neighborhood, stability, perturbations, token_quality, ranking)
    _write_json(output / "summary.json", summary)
    manifest = {
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "rna_path": config.rna.path,
        "rna_sha256": _sha256(config.rna.path),
        "embedding_path": config.embeddings.path,
        "embedding_sha256": _sha256(config.embeddings.path),
        "metadata_path": config.metadata.path,
        "metadata_sha256": _sha256(config.metadata.path),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "sample_count": int(len(data.sample_ids)),
        "split_counts": {name: int(len(rows)) for name, rows in data.split_indices.items()},
        "selected_gene_count": int(len(data.selected_gene_ids)),
        "representations": {name: list(values.shape) for name, values in representations.items()},
        "methylation_inputs_loaded": False,
        "downstream_regressor_trained": False,
    }
    if config.token_embeddings is not None:
        manifest["token_embedding_path"] = config.token_embeddings.path
        manifest["token_embedding_sha256"] = _sha256(config.token_embeddings.path)
    _write_json(output / "manifest.json", manifest)
    _report(summary, ranking, reconstruction, output / "report.md")
    return output
