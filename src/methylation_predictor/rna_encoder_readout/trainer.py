"""Leakage-safe trainer for RNA-only readout optimisation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from .config import ReadoutConfig
from .evaluation import _effective_rank, evaluate_loader, export_embeddings
from .io import CacheMetadata, TargetBundle, _normalise_gene_id, load_data
from .model import RNAReadoutModel
from .objectives import compute_loss, r2_score_torch
from .poolers import build_pooler
from .ridge_head import RidgeHead


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _load_warm_start(model: RNAReadoutModel, path: str, mode: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    current = model.state_dict()
    if mode == "decoder_only":
        prefixes = ("total_decoder.", "within_decoder.")
        selected = {key: value for key, value in state.items() if key.startswith(prefixes)}
        required = {"total_decoder.weight", "total_decoder.bias", "within_decoder.weight", "within_decoder.bias"}
        absent = required - set(selected)
        if absent:
            raise ValueError(f"warm-start checkpoint is missing decoder keys: {sorted(absent)}")
    elif mode == "compatible":
        selected = {
            key: value for key, value in state.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        if not selected:
            raise ValueError("compatible warm start found no matching parameter shapes")
    elif mode == "full":
        model.load_state_dict(state, strict=True)
        return {"path": path, "mode": mode, "loaded_keys": sorted(state), "skipped_keys": []}
    else:
        raise ValueError(f"unsupported warm_start_mode={mode!r}")
    model.load_state_dict(selected, strict=False)
    skipped = sorted(set(state) - set(selected))
    return {"path": path, "mode": mode, "loaded_keys": sorted(selected), "skipped_keys": skipped}


def _augmented_batch(batch: dict[str, torch.Tensor], layers: list[int]) -> dict[str, torch.Tensor] | None:
    present = [layer for layer in layers if f"aug_tokens_layer{layer}" in batch]
    if not present:
        return None
    augmented = dict(batch)
    for layer in present:
        augmented[f"tokens_layer{layer}"] = batch[f"aug_tokens_layer{layer}"]
    return augmented


def _epoch(
    model: RNAReadoutModel,
    loader: DataLoader,
    config: ReadoutConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "total_mse": 0.0, "within_mse": 0.0, "consistency": 0.0}
    count = 0
    autocast_enabled = bool(config.training.mixed_precision and device.type == "cuda")
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            output, total_hat, within_hat = model(batch)
            augmented_embedding = None
            aug = _augmented_batch(batch, config.token_cache.layers)
            if aug is not None and config.objective.consistency_weight > 0:
                augmented_embedding = model.encode(aug).embedding
            loss_output = compute_loss(
                config.objective,
                output.embedding,
                total_hat,
                within_hat,
                batch,
                augmented_embedding,
            )
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_output.loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
                optimizer.step()
        batch_size = int(batch["target_total"].shape[0])
        count += batch_size
        for name in totals:
            totals[name] += float(getattr(loss_output, name).detach()) * batch_size
    return {name: value / max(count, 1) for name, value in totals.items()}


def _initial_mean_difference(
    model: RNAReadoutModel,
    dataset,
    primary_layer: int,
    device: torch.device,
) -> float:
    example = dataset[0]
    batch = {
        key: (value.unsqueeze(0) if value.ndim > 0 else value[None]).to(device)
        for key, value in example.items()
    }
    with torch.inference_mode():
        embedding = model.encode(batch).embedding
        baseline = batch[f"full_mean_layer{primary_layer}"]
    return float(torch.max(torch.abs(embedding - baseline)).cpu())


def train(config: ReadoutConfig) -> dict[str, Any]:
    if config.training.decoder_mode == "ridge_alternating":
        return _train_ridge_alternating(config)
    return _train_sgd(config)


def _train_sgd(config: ReadoutConfig) -> dict[str, Any]:
    _seed_everything(config.training.seed)
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, targets, datasets = load_data(config)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=(split == "train"),
            num_workers=config.training.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=config.training.num_workers > 0,
            prefetch_factor=4 if config.training.num_workers > 0 else None,
        )
        for split, dataset in datasets.items()
    }
    pooler = build_pooler(
        config.model,
        config.token_cache.layers,
        metadata.gene_ids,
        metadata.input_variance,
        metadata.input_within_variance,
    )
    model = RNAReadoutModel(config, pooler, len(targets.target_gene_ids))
    warm_start = None
    if config.training.warm_start_checkpoint:
        warm_start = _load_warm_start(
            model, config.training.warm_start_checkpoint, config.training.warm_start_mode
        )
    device = _device(config.training.device)
    model.to(device)
    initial_mean_difference = _initial_mean_difference(
        model, datasets["train"], config.model.primary_layer, device
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler_enabled = bool(config.training.mixed_precision and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # PyTorch 2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    history: list[dict[str, Any]] = []
    initial = evaluate_loader(model, loaders["validation"], device)
    initial_objective = (
        config.objective.total_weight * initial["total_mse"]
        + config.objective.within_weight * initial["within_mse"]
    )
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_objective = initial_objective if config.training.seed_initial_checkpoint else float("inf")
    history.append({"epoch": 0, "split": "validation", "objective": initial_objective, **initial})
    print(
        f"[{config.run_name}] epoch=0 objective={initial_objective:.5f} "
        f"total_r2={initial['total_r2']:.4f} within_r2={initial['within_r2']:.4f} (initial)",
        flush=True,
    )
    stale = 0
    for epoch in range(1, config.training.epochs + 1):
        epoch_started = time.time()
        train_metrics = _epoch(model, loaders["train"], config, device, optimizer, scaler)
        validation = evaluate_loader(model, loaders["validation"], device)
        objective = (
            config.objective.total_weight * validation["total_mse"]
            + config.objective.within_weight * validation["within_mse"]
        )
        history.append({"epoch": epoch, "split": "train", **train_metrics})
        history.append({"epoch": epoch, "split": "validation", "objective": objective, **validation})
        improved = objective < best_objective - config.training.min_delta
        if improved:
            best_objective = objective
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"[{config.run_name}] epoch={epoch} objective={objective:.5f} "
            f"total_r2={validation['total_r2']:.4f} within_r2={validation['within_r2']:.4f} "
            f"stale={stale}/{config.training.patience} best_epoch={best_epoch} "
            f"elapsed={time.time() - epoch_started:.1f}s"
            + (" *" if improved else ""),
            flush=True,
        )
        if stale >= config.training.patience:
            break
    model.load_state_dict(best_state)
    split_metrics = {split: evaluate_loader(model, loader, device) for split, loader in loaders.items()}
    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "target_gene_ids": np.asarray(targets.target_gene_ids).astype(str).tolist(),
        "input_gene_ids": np.asarray(metadata.gene_ids).astype(str).tolist(),
    }
    checkpoint_path = output_dir / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    diagnostics = export_embeddings(
        config, model, loaders, metadata, output_dir / "embeddings.h5", device
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame({
        "gene_id": np.asarray(targets.target_gene_ids).astype(str),
        "target_index": targets.target_indices,
        "train_mean": targets.target_train_mean,
        "train_std": targets.target_train_std,
        "within_train_std": targets.within_train_std,
    }).to_csv(output_dir / "selected_target_genes.csv", index=False)
    pd.DataFrame({
        "gene_id": np.asarray(metadata.gene_ids).astype(str),
        "train_variance": metadata.input_variance,
        "within_train_variance": metadata.input_within_variance,
    }).to_csv(output_dir / "selected_input_genes.csv", index=False)
    manifest = {
        "run_name": config.run_name,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "initial_validation": initial,
        "best_validation_objective": best_objective,
        "metrics": split_metrics,
        "diagnostics": diagnostics,
        "warm_start": warm_start,
        "methylation_inputs_loaded": False,
        "downstream_methylation_regressor_trained": False,
        "encoder_frozen": True,
        "model_kind": config.model.kind,
        "num_queries": config.model.num_queries,
        "initial_readout_vs_mean_max_abs": initial_mean_difference,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "pooler_parameters": int(sum(p.numel() for p in model.pooler.parameters())),
        "decoder_parameters": int(
            sum(p.numel() for p in model.total_decoder.parameters())
            + sum(p.numel() for p in model.within_decoder.parameters())
        ),
        "input_target_gene_overlap": int(
            len(
                {_normalise_gene_id(gene) for gene in metadata.gene_ids}
                & {_normalise_gene_id(gene) for gene in targets.target_gene_ids}
            )
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "config.resolved.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    return manifest


def _fit_ridge_heads(
    model: RNAReadoutModel,
    loader: DataLoader,
    device: torch.device,
    alpha_total: float,
    alpha_within: float,
) -> tuple[RidgeHead, RidgeHead]:
    model.eval()
    embeddings, targets_total, targets_within = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            output = model.encode(batch)
            embeddings.append(output.embedding)
            targets_total.append(batch["target_total"])
            targets_within.append(batch["target_within"])
    z = torch.cat(embeddings)
    ridge_total = RidgeHead(alpha_total)
    ridge_within = RidgeHead(alpha_within)
    ridge_total.fit(z, torch.cat(targets_total))
    ridge_within.fit(z, torch.cat(targets_within))
    return ridge_total, ridge_within


def _epoch_ridge(
    model: RNAReadoutModel,
    loader: DataLoader,
    config: ReadoutConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    ridge_total: RidgeHead,
    ridge_within: RidgeHead,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "total_mse": 0.0, "within_mse": 0.0, "consistency": 0.0}
    count = 0
    autocast_enabled = bool(config.training.mixed_precision and device.type == "cuda")
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            output = model.encode(batch)
            prediction_total = ridge_total.predict(output.embedding)
            prediction_within = ridge_within.predict(output.embedding)
            augmented_embedding = None
            aug = _augmented_batch(batch, config.token_cache.layers)
            if aug is not None and config.objective.consistency_weight > 0:
                augmented_embedding = model.encode(aug).embedding
            loss_output = compute_loss(
                config.objective, output.embedding, prediction_total, prediction_within, batch, augmented_embedding,
            )
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_output.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.pooler.parameters(), config.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.pooler.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
        batch_size = int(batch["target_total"].shape[0])
        count += batch_size
        for name in totals:
            totals[name] += float(getattr(loss_output, name).detach()) * batch_size
    return {name: value / max(count, 1) for name, value in totals.items()}


def _evaluate_ridge(
    model: RNAReadoutModel,
    loader: DataLoader,
    device: torch.device,
    ridge_total: RidgeHead,
    ridge_within: RidgeHead,
) -> dict[str, float]:
    model.eval()
    targets_total, targets_within, pred_total, pred_within, embeddings = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            output = model.encode(batch)
            targets_total.append(batch["target_total"].cpu())
            targets_within.append(batch["target_within"].cpu())
            pred_total.append(ridge_total.predict(output.embedding).cpu())
            pred_within.append(ridge_within.predict(output.embedding).cpu())
            embeddings.append(output.embedding.cpu())
    y_total = torch.cat(targets_total)
    y_within = torch.cat(targets_within)
    p_total = torch.cat(pred_total)
    p_within = torch.cat(pred_within)
    z = torch.cat(embeddings).numpy()
    return {
        "total_mse": float(torch.mean((y_total - p_total).square())),
        "within_mse": float(torch.mean((y_within - p_within).square())),
        "total_r2": float(r2_score_torch(y_total, p_total)),
        "within_r2": float(r2_score_torch(y_within, p_within)),
        "effective_rank": _effective_rank(z),
    }


def _train_ridge_alternating(config: ReadoutConfig) -> dict[str, Any]:
    _seed_everything(config.training.seed)
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, targets, datasets = load_data(config)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=(split == "train"),
            num_workers=config.training.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=config.training.num_workers > 0,
            prefetch_factor=4 if config.training.num_workers > 0 else None,
        )
        for split, dataset in datasets.items()
    }
    pooler = build_pooler(
        config.model, config.token_cache.layers, metadata.gene_ids, metadata.input_variance, metadata.input_within_variance,
    )
    model = RNAReadoutModel(config, pooler, len(targets.target_gene_ids))
    warm_start = None
    if config.training.warm_start_checkpoint:
        warm_start = _load_warm_start(
            model, config.training.warm_start_checkpoint, config.training.warm_start_mode
        )
    device = _device(config.training.device)
    model.to(device)
    initial_mean_difference = _initial_mean_difference(
        model, datasets["train"], config.model.primary_layer, device
    )
    # The decoder heads are unused in this mode: only the pooler is trained,
    # a closed-form Ridge head supplies gradients instead of a joint SGD head.
    for parameter in model.total_decoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.within_decoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.pooler.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler_enabled = bool(config.training.mixed_precision and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # PyTorch 2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    ridge_total, ridge_within = _fit_ridge_heads(
        model, loaders["train"], device, config.training.ridge_alpha_total, config.training.ridge_alpha_within,
    )
    history: list[dict[str, Any]] = []
    initial = _evaluate_ridge(model, loaders["validation"], device, ridge_total, ridge_within)
    initial_objective = (
        config.objective.total_weight * initial["total_mse"]
        + config.objective.within_weight * initial["within_mse"]
    )
    best_state = deepcopy(model.state_dict())
    best_ridge = (ridge_total, ridge_within)
    best_epoch = 0
    best_objective = initial_objective if config.training.seed_initial_checkpoint else float("inf")
    history.append({"epoch": 0, "split": "validation", "objective": initial_objective, **initial})
    print(
        f"[{config.run_name}] epoch=0 objective={initial_objective:.5f} "
        f"total_r2={initial['total_r2']:.4f} within_r2={initial['within_r2']:.4f} (initial, ridge)",
        flush=True,
    )
    stale = 0
    for epoch in range(1, config.training.epochs + 1):
        epoch_started = time.time()
        if (epoch - 1) % config.training.ridge_refit_interval == 0:
            ridge_total, ridge_within = _fit_ridge_heads(
                model, loaders["train"], device, config.training.ridge_alpha_total, config.training.ridge_alpha_within,
            )
        train_metrics = _epoch_ridge(model, loaders["train"], config, device, optimizer, scaler, ridge_total, ridge_within)
        validation = _evaluate_ridge(model, loaders["validation"], device, ridge_total, ridge_within)
        objective = (
            config.objective.total_weight * validation["total_mse"]
            + config.objective.within_weight * validation["within_mse"]
        )
        history.append({"epoch": epoch, "split": "train", **train_metrics})
        history.append({"epoch": epoch, "split": "validation", "objective": objective, **validation})
        improved = objective < best_objective - config.training.min_delta
        if improved:
            best_objective = objective
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_ridge = (ridge_total, ridge_within)
            stale = 0
        else:
            stale += 1
        print(
            f"[{config.run_name}] epoch={epoch} objective={objective:.5f} "
            f"total_r2={validation['total_r2']:.4f} within_r2={validation['within_r2']:.4f} "
            f"stale={stale}/{config.training.patience} best_epoch={best_epoch} "
            f"elapsed={time.time() - epoch_started:.1f}s (ridge)"
            + (" *" if improved else ""),
            flush=True,
        )
        if stale >= config.training.patience:
            break
    model.load_state_dict(best_state)
    ridge_total, ridge_within = best_ridge
    split_metrics = {
        split: _evaluate_ridge(model, loader, device, ridge_total, ridge_within)
        for split, loader in loaders.items()
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "ridge_total": ridge_total.state_dict(),
        "ridge_within": ridge_within.state_dict(),
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "target_gene_ids": np.asarray(targets.target_gene_ids).astype(str).tolist(),
        "input_gene_ids": np.asarray(metadata.gene_ids).astype(str).tolist(),
    }
    checkpoint_path = output_dir / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    diagnostics = export_embeddings(
        config, model, loaders, metadata, output_dir / "embeddings.h5", device
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame({
        "gene_id": np.asarray(targets.target_gene_ids).astype(str),
        "target_index": targets.target_indices,
        "train_mean": targets.target_train_mean,
        "train_std": targets.target_train_std,
        "within_train_std": targets.within_train_std,
    }).to_csv(output_dir / "selected_target_genes.csv", index=False)
    pd.DataFrame({
        "gene_id": np.asarray(metadata.gene_ids).astype(str),
        "train_variance": metadata.input_variance,
        "within_train_variance": metadata.input_within_variance,
    }).to_csv(output_dir / "selected_input_genes.csv", index=False)
    manifest = {
        "run_name": config.run_name,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "initial_validation": initial,
        "best_validation_objective": best_objective,
        "metrics": split_metrics,
        "diagnostics": diagnostics,
        "warm_start": warm_start,
        "methylation_inputs_loaded": False,
        "downstream_methylation_regressor_trained": False,
        "encoder_frozen": True,
        "decoder_mode": "ridge_alternating",
        "ridge_alpha_total": config.training.ridge_alpha_total,
        "ridge_alpha_within": config.training.ridge_alpha_within,
        "ridge_refit_interval": config.training.ridge_refit_interval,
        "model_kind": config.model.kind,
        "num_queries": config.model.num_queries,
        "initial_readout_vs_mean_max_abs": initial_mean_difference,
        "trainable_parameters": int(sum(p.numel() for p in model.pooler.parameters() if p.requires_grad)),
        "pooler_parameters": int(sum(p.numel() for p in model.pooler.parameters())),
        "decoder_parameters": 0,
        "input_target_gene_overlap": int(
            len(
                {_normalise_gene_id(gene) for gene in metadata.gene_ids}
                & {_normalise_gene_id(gene) for gene in targets.target_gene_ids}
            )
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "config.resolved.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    return manifest


def validate(config: ReadoutConfig) -> dict[str, Any]:
    metadata, targets, datasets = load_data(config)
    pooler = build_pooler(
        config.model,
        config.token_cache.layers,
        metadata.gene_ids,
        metadata.input_variance,
        metadata.input_within_variance,
    )
    sample = datasets["train"][0]
    batched = {key: value.unsqueeze(0) if value.ndim > 0 else value[None] for key, value in sample.items()}
    with torch.no_grad():
        output = pooler(batched)
    if output.embedding.shape != (1, config.model.output_dim):
        raise AssertionError(f"unexpected embedding shape {tuple(output.embedding.shape)}")
    baseline = batched[f"full_mean_layer{config.model.primary_layer}"]
    initial_mean_difference = float(torch.max(torch.abs(output.embedding - baseline)))
    return {
        "n_samples": int(len(metadata.sample_ids)),
        "n_input_genes": int(len(metadata.gene_ids)),
        "n_target_genes": int(len(targets.target_gene_ids)),
        "token_dim": metadata.token_dim,
        "layers": config.token_cache.layers,
        "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "input_target_gene_overlap": int(
            len(
                {_normalise_gene_id(gene) for gene in metadata.gene_ids}
                & {_normalise_gene_id(gene) for gene in targets.target_gene_ids}
            )
        ),
        "initial_embedding_shape": list(output.embedding.shape),
        "initial_readout_vs_mean_max_abs": initial_mean_difference,
        "methylation_inputs_loaded": False,
    }
