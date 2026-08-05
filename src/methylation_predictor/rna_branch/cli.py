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

    predict_stream = subparsers.add_parser(
        "predict-stream", help="stream a complete prediction matrix to HDF5"
    )
    predict_stream.add_argument("--config", required=True)
    predict_stream.add_argument("--checkpoint", required=True)
    predict_stream.add_argument("--sample-split", required=True)
    predict_stream.add_argument("--cpg-split", required=True)
    predict_stream.add_argument("--output", required=True)
    predict_stream.add_argument("--include-target", action="store_true")

    predict_stream_metrics = subparsers.add_parser(
        "predict-stream-metrics",
        help="evaluate a huge panel via streaming sufficient statistics (no full matrix in RAM, no predictions saved)",
    )
    predict_stream_metrics.add_argument("--config", required=True)
    predict_stream_metrics.add_argument("--checkpoint", required=True)
    predict_stream_metrics.add_argument("--sample-split", required=True)
    predict_stream_metrics.add_argument("--cpg-split", required=True)
    predict_stream_metrics.add_argument("--output", required=True, help="path to write metrics JSON")
    predict_stream_metrics.add_argument(
        "--chromosome-lookup", help="parquet with cpg_idx,chromosome columns, for per_chromosome breakdown"
    )
    predict_stream_metrics.add_argument("--cpg-chunk-size", type=int)
    predict_stream_metrics.add_argument("--skip-locus-median-correlations", action="store_true")

    upload_artifact = subparsers.add_parser(
        "upload-artifact",
        help="post-hoc: upload a completed run's checkpoint+manifest as a W&B artifact "
             "(kept out of the training loop -- see tracking.log_checkpoint=false)",
    )
    upload_artifact.add_argument("--run-dir", required=True)
    upload_artifact.add_argument("--checkpoint-name", default="best.pt")
    upload_artifact.add_argument("--aliases", default="best")

    benchmark = subparsers.add_parser(
        "benchmark-inference", help="compare cached and uncached bilinear panel inference"
    )
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--sample-split", default="test")
    benchmark.add_argument("--cpg-split", default="test")
    benchmark.add_argument("--max-cpgs", type=int, default=512)
    benchmark.add_argument("--cpg-chunk-size", type=int)

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


def _upload_artifact(run_dir: str, checkpoint_name: str, aliases: str) -> dict[str, object]:
    from .config import load_config as _load_config
    from .tracking import WandbTracker
    from .utils import json_safe as _json_safe

    run_path = Path(run_dir)
    run_config = _load_config(run_path / "config.yaml")
    run_id_path = run_path / "wandb_run_id.txt"
    if not run_id_path.is_file():
        raise FileNotFoundError(f"{run_id_path} not found -- was tracking.backend=wandb for this run?")
    resume_id = run_id_path.read_text().strip()

    import wandb

    run = wandb.init(
        project=run_config.tracking.project,
        entity=run_config.tracking.entity,
        id=resume_id,
        resume="must",
    )
    try:
        artifact = wandb.Artifact(name=f"{resume_id}-checkpoint", type="model")
        artifact.add_file(str(run_path / checkpoint_name))
        manifest_path = run_path / "manifest.json"
        if manifest_path.is_file():
            artifact.add_file(str(manifest_path))
        run.log_artifact(artifact, aliases=[a.strip() for a in aliases.split(",") if a.strip()])
    finally:
        wandb.finish()
    return {"run_id": resume_id, "checkpoint": str(run_path / checkpoint_name), "aliases": aliases}


def main() -> None:
    args = _parser().parse_args()
    if args.command == "upload-artifact":
        print(json.dumps(_upload_artifact(args.run_dir, args.checkpoint_name, args.aliases), indent=2, sort_keys=True))
        return
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
    elif args.command == "predict-stream":
        runner = ExperimentRunner(config)
        try:
            checkpoint = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
            runner.model.load_state_dict(checkpoint["model_state"])
            runner._refresh_train_centroids()
            result = runner.export_panel_h5(
                args.sample_split,
                args.cpg_split,
                args.output,
                include_target=args.include_target,
            )
            result["checkpoint"] = str(args.checkpoint)
            result["checkpoint_epoch"] = checkpoint.get("epoch")
        finally:
            runner.close()
    elif args.command == "predict-stream-metrics":
        runner = ExperimentRunner(config)
        try:
            checkpoint = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
            runner.model.load_state_dict(checkpoint["model_state"])
            runner._refresh_train_centroids()
            chromosome_by_cpg_id = None
            if args.chromosome_lookup:
                import pandas as pd
                lookup = pd.read_parquet(args.chromosome_lookup)
                chromosome_by_cpg_id = dict(zip(lookup["cpg_idx"].astype(str), lookup["chromosome"].astype(str)))
            if args.cpg_chunk_size is not None:
                if args.cpg_chunk_size < 1:
                    raise ValueError("--cpg-chunk-size must be positive")
            result = runner.evaluate_panel_streaming(
                args.sample_split,
                args.cpg_split,
                chromosome_by_cpg_id=chromosome_by_cpg_id,
                cpg_chunk_size=args.cpg_chunk_size,
                include_locus_median_correlations=not args.skip_locus_median_correlations,
            )
            result["checkpoint"] = str(args.checkpoint)
            result["checkpoint_epoch"] = checkpoint.get("epoch")
            result["sample_split"] = args.sample_split
            result["cpg_split"] = args.cpg_split
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            from .utils import write_json as _write_json
            _write_json(output, result)
            result["output"] = str(output)
        finally:
            runner.close()
    elif args.command == "benchmark-inference":
        runner = ExperimentRunner(config)
        try:
            checkpoint = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
            runner.model.load_state_dict(checkpoint["model_state"])
            runner._refresh_train_centroids()
            if args.cpg_chunk_size is not None:
                if args.cpg_chunk_size < 1:
                    raise ValueError("--cpg-chunk-size must be positive")
                runner.config.evaluation.cpg_chunk_size = args.cpg_chunk_size
            uncached = runner.predict_panel(
                args.sample_split, args.cpg_split, max_cpgs=args.max_cpgs,
                keep_predictions=True, use_factorized_cache=False, seed_offset=7001,
            )
            cached = runner.predict_panel(
                args.sample_split, args.cpg_split, max_cpgs=args.max_cpgs,
                keep_predictions=True, use_factorized_cache=True, seed_offset=7001,
            )
            if uncached.prediction is None or cached.prediction is None:
                raise RuntimeError("benchmark predictions were not retained")
            uncached_seconds = float(uncached.metrics["inference_seconds"])
            cached_seconds = float(cached.metrics["inference_seconds"])
            result = {
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "sample_split": args.sample_split,
                "cpg_split": args.cpg_split,
                "num_samples": len(cached.sample_indices),
                "num_cpgs": len(cached.cpg_indices),
                "uncached_seconds": uncached_seconds,
                "cached_seconds": cached_seconds,
                "speedup": uncached_seconds / max(cached_seconds, 1e-9),
                "uncached_pairs_per_second": uncached.metrics["inference_pairs_per_second"],
                "cached_pairs_per_second": cached.metrics["inference_pairs_per_second"],
                "max_abs_prediction_difference": float(
                    np.max(np.abs(uncached.prediction - cached.prediction))
                ),
            }
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
