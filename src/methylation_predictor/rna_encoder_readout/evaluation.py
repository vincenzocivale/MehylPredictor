"""Representation diagnostics and HDF5 export for locked readouts."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ReadoutConfig
from .io import CacheMetadata, TargetBundle
from .model import RNAReadoutModel
from .objectives import r2_score_torch
from .poolers import attention_diagnostics


def _effective_rank(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    spectrum = np.square(singular)
    if spectrum.sum() <= 0:
        return 0.0
    probabilities = spectrum / spectrum.sum()
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def evaluate_loader(
    model: RNAReadoutModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals, within, pred_total, pred_within, embeddings = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output, total_hat, within_hat = model(batch)
            totals.append(batch["target_total"].cpu())
            within.append(batch["target_within"].cpu())
            pred_total.append(total_hat.cpu())
            pred_within.append(within_hat.cpu())
            embeddings.append(output.embedding.cpu())
    y_total = torch.cat(totals)
    y_within = torch.cat(within)
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


def export_embeddings(
    config: ReadoutConfig,
    model: RNAReadoutModel,
    loaders: dict[str, DataLoader],
    metadata: CacheMetadata,
    output_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    embeddings_by_index = np.zeros((len(metadata.sample_ids), config.model.output_dim), dtype=np.float32)
    latent_by_index: dict[int, np.ndarray] = {}
    attention_sum: np.ndarray | None = None
    attention_count = 0
    diagnostics_values: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        for loader in loaders.values():
            for batch in loader:
                source_index = batch["source_index"].numpy()
                device_batch = {key: value.to(device) for key, value in batch.items()}
                output = model.encode(device_batch)
                embeddings_by_index[source_index] = output.embedding.cpu().numpy()
                if output.latent_tokens is not None:
                    latent = output.latent_tokens.cpu().numpy()
                    for idx, value in zip(source_index, latent, strict=True):
                        latent_by_index[int(idx)] = value
                if output.attention_weights is not None:
                    weights = output.attention_weights.cpu().numpy()
                    current = weights.sum(axis=0)
                    attention_sum = current if attention_sum is None else attention_sum + current
                    attention_count += len(weights)
                for name, value in attention_diagnostics(output).items():
                    diagnostics_values[name].append(value.detach().cpu().numpy())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype("utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("embeddings", data=embeddings_by_index)
        handle.create_dataset("sample_idx", data=np.asarray(metadata.sample_ids, dtype=object), dtype=strings)
        handle.create_dataset("split", data=np.asarray(metadata.splits, dtype=object), dtype=strings)
        handle.create_dataset("cancer_type", data=np.asarray(metadata.cancer_types, dtype=object), dtype=strings)
        if latent_by_index:
            first = next(iter(latent_by_index.values()))
            latent = np.zeros((len(metadata.sample_ids), *first.shape), dtype=np.float32)
            for index, value in latent_by_index.items():
                latent[index] = value
            handle.create_dataset("latent_tokens", data=latent)
    diagnostics: dict[str, Any] = {}
    if attention_sum is not None:
        mean_attention = attention_sum / max(attention_count, 1)
        rows = []
        for query in range(mean_attention.shape[0]):
            order = np.argsort(mean_attention[query])[::-1]
            for rank, gene_index in enumerate(order[:100], start=1):
                rows.append({
                    "query": query,
                    "rank": rank,
                    "gene_id": str(metadata.gene_ids[gene_index]),
                    "mean_weight": float(mean_attention[query, gene_index]),
                })
        attention_path = output_path.with_name("attention_gene_weights.csv")
        pd.DataFrame(rows).to_csv(attention_path, index=False)
        diagnostics["attention_gene_weights"] = str(attention_path)
    for name, chunks in diagnostics_values.items():
        values = np.concatenate(chunks, axis=0)
        diagnostics[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
        }
    return diagnostics
