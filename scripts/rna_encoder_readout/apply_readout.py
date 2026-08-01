#!/usr/bin/env python3
"""Apply a locked RNA readout checkpoint to a compatible token cache."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from methylation_predictor.rna_encoder_readout.config import load_config
from methylation_predictor.rna_encoder_readout.evaluation import export_embeddings
from methylation_predictor.rna_encoder_readout.io import load_data
from methylation_predictor.rna_encoder_readout.model import RNAReadoutModel
from methylation_predictor.rna_encoder_readout.poolers import build_pooler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    config = deepcopy(load_config(args.config))
    config.token_cache.path = args.token_cache
    config.token_cache.augmentation_path = None
    if args.batch_size:
        config.training.batch_size = args.batch_size
    metadata, targets, datasets = load_data(config)
    pooler = build_pooler(
        config.model,
        config.token_cache.layers,
        metadata.gene_ids,
        metadata.input_variance,
        metadata.input_within_variance,
    )
    model = RNAReadoutModel(config, pooler, len(targets.target_gene_ids))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loaders = {
        split: DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
        for split, dataset in datasets.items()
    }
    diagnostics = export_embeddings(config, model, loaders, metadata, Path(args.output), device)
    print(diagnostics)


if __name__ == "__main__":
    main()
