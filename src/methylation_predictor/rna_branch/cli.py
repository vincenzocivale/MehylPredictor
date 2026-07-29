"""Command-line interface for RNA-branch experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .config import load_config
from .data import load_bundle, summarize_bundle
from .low_rank import run_low_rank
from .pca import fit_pca
from .random_projection import fit_random_projection
from .trainer import ExperimentRunner
from .utils import json_safe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate and summarize aligned inputs")
    validate.add_argument("--config", required=True)

    train = subparsers.add_parser("train", help="train one RNA residual model")
    train.add_argument("--config", required=True)

    predict = subparsers.add_parser("predict", help="predict a panel from a selected checkpoint")
    predict.add_argument("--config", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--sample-split", required=True)
    predict.add_argument("--cpg-split", required=True)
    predict.add_argument("--output", required=True)

    low_rank = subparsers.add_parser("low-rank", help="analyze the residual singular spectrum")
    low_rank.add_argument("--config", required=True)
    low_rank.add_argument("--output", required=True)
    low_rank.add_argument("--components", default="8,16,32,64,128,256")
    low_rank.add_argument("--within-cancer", action="store_true")
    low_rank.add_argument("--max-samples", type=int)
    low_rank.add_argument("--max-cpgs", type=int)

    pca = subparsers.add_parser("fit-pca", help="fit training-only IncrementalPCA and transform all samples")
    pca.add_argument("--config", required=True)
    pca.add_argument("--output", required=True)
    pca.add_argument("--components", type=int, default=256)
    pca.add_argument("--batch-size", type=int, default=512)

    random_projection = subparsers.add_parser(
        "fit-random-projection", help="apply a frozen Gaussian random projection to all samples"
    )
    random_projection.add_argument("--config", required=True)
    random_projection.add_argument("--output", required=True)
    random_projection.add_argument("--components", type=int, default=256)
    random_projection.add_argument("--seed", type=int, default=20260730)
    random_projection.add_argument("--batch-size", type=int, default=512)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        bundle = load_bundle(config.data, seed=config.training.seed)
        try:
            result = summarize_bundle(bundle)
        finally:
            bundle.close()
    elif args.command == "train":
        runner = ExperimentRunner(config)
        try:
            result = runner.train()
        finally:
            runner.close()
    elif args.command == "predict":
        runner = ExperimentRunner(config)
        try:
            checkpoint = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
            runner.model.load_state_dict(checkpoint["model_state"])
            runner._refresh_train_centroids()
            panel = runner.predict_panel(args.sample_split, args.cpg_split, keep_predictions=True)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                target=panel.target,
                prediction=panel.prediction,
                prior=runner.bundle.loci.prior[panel.cpg_indices],
                sample_idx=runner.bundle.samples.ids[panel.sample_indices].astype(str),
                cpg_idx=runner.bundle.loci.ids[panel.cpg_indices].astype(str),
                cancer_type=runner.bundle.samples.cancer_types[panel.sample_indices].astype(str),
            )
            result = {"checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
                      "sample_split": args.sample_split, "cpg_split": args.cpg_split,
                      "output": str(output), "metrics": panel.metrics}
        finally:
            runner.close()
    elif args.command == "low-rank":
        components = [int(value) for value in args.components.split(",") if value.strip()]
        result = run_low_rank(
            config,
            args.output,
            components,
            within_cancer=args.within_cancer,
            max_samples=args.max_samples,
            max_cpgs=args.max_cpgs,
        )
    elif args.command == "fit-pca":
        result = fit_pca(config, args.output, args.components, args.batch_size)
    elif args.command == "fit-random-projection":
        result = fit_random_projection(config, args.output, args.components, args.seed, args.batch_size)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
