from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from methylation_predictor.rna_encoder_readout.config import (
    ModelConfig,
    ObjectiveConfig,
    RNAConfig,
    ReadoutConfig,
    TokenCacheConfig,
    TrainingConfig,
    load_config,
)
from methylation_predictor.rna_encoder_readout.io import load_data
from methylation_predictor.rna_encoder_readout.poolers import build_pooler
from methylation_predictor.rna_encoder_readout.trainer import train, validate


def _write_fixture(tmp_path: Path) -> Path:
    rng = np.random.default_rng(3)
    n_samples, n_input, n_all, dim = 30, 12, 32, 8
    sample_ids = np.asarray([f"S{i:03d}" for i in range(n_samples)], dtype=object)
    splits = np.asarray(["train"] * 20 + ["validation"] * 5 + ["test"] * 5, dtype=object)
    cancers = np.asarray((["A", "B"] * 15), dtype=object)
    input_genes = np.asarray([f"G{i:03d}" for i in range(n_input)], dtype=object)
    all_genes = np.asarray([f"G{i:03d}" for i in range(n_all)], dtype=object)
    expression = rng.normal(size=(n_samples, n_input)).astype(np.float32)
    base = rng.normal(size=(n_samples, n_input, dim)).astype(np.float32)
    layer2 = base + expression[..., None] * 0.15
    layer3 = base * 0.8 + expression[..., None] * 0.2
    strings = h5py.string_dtype("utf-8")
    cache = tmp_path / "tokens.h5"
    with h5py.File(cache, "w") as handle:
        handle.create_dataset("tokens_layer2", data=layer2)
        handle.create_dataset("tokens_layer3", data=layer3)
        handle.create_dataset("full_mean_layer2", data=layer2.mean(axis=1))
        handle.create_dataset("full_mean_layer3", data=layer3.mean(axis=1))
        handle.create_dataset("expression_log2p1", data=expression)
        handle.create_dataset("normalized_expression", data=np.clip(expression / 10 + 0.5, 0, 1))
        handle.create_dataset("token_ids", data=np.clip((expression + 3) * 8, 0, 63).astype(np.int16))
        handle.create_dataset("sample_idx", data=sample_ids, dtype=strings)
        handle.create_dataset("gene_ids", data=input_genes, dtype=strings)
        handle.create_dataset("split", data=splits, dtype=strings)
        handle.create_dataset("cancer_type", data=cancers, dtype=strings)
    rna = tmp_path / "rna.h5"
    values = rng.normal(size=(n_samples, n_all)).astype(np.float32)
    with h5py.File(rna, "w") as handle:
        handle.create_dataset("X", data=values)
        handle.create_dataset("sample_idx", data=sample_ids, dtype=strings)
        handle.create_dataset("gene_ids", data=all_genes, dtype=strings)
    config = {
        "run_name": "fixture",
        "output_dir": str(tmp_path / "out"),
        "token_cache": {"path": str(cache), "layers": [2, 3]},
        "rna": {"path": str(rna)},
        "model": {
            "kind": "mean",
            "primary_layer": 2,
            "output_dim": dim,
            "hidden_dim": 16,
            "num_heads": 2,
            "num_queries": 2,
            "dropout": 0.0,
        },
        "objective": {
            "target_gene_count": 8,
            "total_weight": 1.0,
            "within_weight": 1.0,
            "input_target_overlap_allowed": False,
        },
        "training": {
            "seed": 17,
            "batch_size": 5,
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "patience": 2,
            "min_delta": 0.0,
            "device": "cpu",
            "mixed_precision": False,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _batch(dataset, batch_size: int = 3):
    examples = [dataset[i] for i in range(batch_size)]
    return {key: torch.stack([example[key] for example in examples]) for key in examples[0]}


def test_poolers_start_from_mean_and_target_genes_are_disjoint(tmp_path: Path) -> None:
    config = load_config(_write_fixture(tmp_path))
    metadata, targets, datasets = load_data(config)
    assert not (set(metadata.gene_ids.astype(str)) & set(targets.target_gene_ids.astype(str)))
    batch = _batch(datasets["train"])
    # The mean-config dataset loads only its primary layer; add layer 3 here to
    # exercise the multi-layer poolers directly.
    with h5py.File(config.token_cache.path, "r") as handle:
        train_indices = np.flatnonzero(metadata.splits == "train")[:3]
        batch["tokens_layer3"] = torch.from_numpy(
            np.asarray(handle["tokens_layer3"][train_indices], dtype=np.float32)
        )
    baseline = batch["full_mean_layer2"]
    residual_kinds = [
        "mean", "mean_std", "learned_gene_weights", "gated_attentive_stats",
        "pma", "scalar_mix", "concat_layer_means", "perceiver_lite",
    ]
    for kind in residual_kinds:
        model_config = replace(config.model, kind=kind)
        pooler = build_pooler(
            model_config,
            config.token_cache.layers,
            metadata.gene_ids,
            metadata.input_variance,
            metadata.input_within_variance,
        )
        output = pooler(batch)
        assert output.embedding.shape == baseline.shape
        assert torch.allclose(output.embedding, baseline, atol=1e-6), kind


def test_end_to_end_training_and_warm_start(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = load_config(config_path)
    summary = validate(config)
    assert summary["methylation_inputs_loaded"] is False
    mean_result = train(config)
    assert Path(mean_result["checkpoint"]).exists()
    assert mean_result["input_target_gene_overlap"] == 0

    raw = yaml.safe_load(config_path.read_text())
    raw["run_name"] = "pma"
    raw["model"]["kind"] = "pma"
    raw["training"]["epochs"] = 1
    raw["training"]["warm_start_checkpoint"] = mean_result["checkpoint"]
    pma_path = tmp_path / "pma.yaml"
    pma_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    pma_result = train(load_config(pma_path))
    assert pma_result["warm_start"] is not None
    assert pma_result["initial_validation"]["total_mse"] >= 0


def test_ridge_alternating_training_runs(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["run_name"] = "pma_ridge"
    raw["model"]["kind"] = "pma"
    raw["training"]["epochs"] = 3
    raw["training"]["decoder_mode"] = "ridge_alternating"
    raw["training"]["ridge_refit_interval"] = 1
    raw["training"]["ridge_alpha_total"] = 1.0
    raw["training"]["ridge_alpha_within"] = 1.0
    ridge_path = tmp_path / "pma_ridge.yaml"
    ridge_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    result = train(load_config(ridge_path))
    assert result["decoder_mode"] == "ridge_alternating"
    assert result["decoder_parameters"] == 0
    assert result["trainable_parameters"] == result["pooler_parameters"]
    assert Path(result["checkpoint"]).exists()
    assert result["initial_validation"]["total_mse"] >= 0
    assert (tmp_path / "out" / "pma_ridge" / "embeddings.h5").exists()
