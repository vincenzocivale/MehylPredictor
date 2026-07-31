#!/usr/bin/env python3
"""Generate matched R0/R1/R2/R4/R5 RNA representation experiment YAMLs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def _write(root: Path, name: str, config: dict) -> None:
    config["run_name"] = name
    config["output_dir"] = str(root / name)
    path = root / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def _f2(config: dict) -> None:
    config.setdefault("model", {}).setdefault("encoder", {})
    config["model"]["encoder"].update({"kind": "linear", "latent_dim": 64, "layer_norm": True})
    config["model"].setdefault("interaction", {})
    config["model"]["interaction"].update({"kind": "concat", "hidden_dim": 128, "dropout": 0.1})
    config["model"]["anchor_to_mean_rna"] = True
    config["model"]["zero_init_residual"] = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="known-good Stage F YAML with current data paths")
    parser.add_argument("--output-root", default="artifacts/rna_branch/representation_search")
    parser.add_argument(
        "--f2-checkpoint",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="production F2 best.pt for one seed; repeat for every requested seed",
    )
    parser.add_argument(
        "--module-weights",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named NPZ produced by build_gene_module_matrix.py; repeat for matched controls",
    )
    parser.add_argument("--gene-embeddings", help="optional NTv3/scFM gene embeddings for R4")
    parser.add_argument("--pretrained-rna", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--seeds", default="17,23,41")
    parser.add_argument(
        "--r5-seeds",
        default="17",
        help="R5 seeds; use 17 for the screen and 17,23,41 only for selected modes",
    )
    parser.add_argument(
        "--r5-modes",
        default="replace,concat,shuffled",
        help="comma-separated R5 modes: replace,concat,shuffled",
    )
    parser.add_argument(
        "--r5b-layers",
        default="",
        help="comma-separated BulkRNABert layers (1-4) to screen as nested F2+adapter "
        "configs; leave empty to skip R5b generation",
    )
    parser.add_argument(
        "--r5b-shuffled-layer",
        type=int,
        help="generate only the concat_shuffled control for this single layer (run this "
        "as a separate invocation after --r5b-layers screening has picked a winner, not "
        "together with --r5b-layers)",
    )
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.base).read_text())
    root = Path(args.output_root)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    pretrained = dict(item.split("=", 1) for item in args.pretrained_rna)
    r5_seeds = [int(value) for value in args.r5_seeds.split(",") if value.strip()]
    r5_modes = [value.strip() for value in args.r5_modes.split(",") if value.strip()]
    if set(r5_modes) - {"replace", "concat", "shuffled"}:
        raise ValueError("--r5-modes accepts only replace,concat,shuffled")
    modules = dict(item.split("=", 1) for item in args.module_weights)
    f2_checkpoints = {int(seed): path for seed, path in (item.split("=", 1) for item in args.f2_checkpoint)}
    if f2_checkpoints and set(seeds) != set(f2_checkpoints):
        raise ValueError("--f2-checkpoint must provide exactly one checkpoint for every requested seed")

    # R0: zscore is the existing production F2 reference, so do not retrain it.
    for transform in ("methylprophet_quantile", "rank", "continuous_rank", "continuous_binary"):
        for seed in seeds:
            config = deepcopy(base)
            _f2(config)
            config["data"]["rna_transform"] = transform
            config["data"]["rna_transform_seed"] = seed
            config["training"]["seed"] = seed
            config["training"]["warm_start_checkpoint"] = None
            _write(root, f"r0_{transform}_s{seed}", config)

    if f2_checkpoints:
        # R1: CpG-conditioned mixture of supervised linear RNA subspaces.
        for experts in (4, 8, 16):
            for seed in seeds:
                config = deepcopy(base)
                _f2(config)
                config["data"]["rna_transform"] = "zscore"
                config["model"]["encoder"].update(
                    {"kind": "global_experts", "num_experts": experts, "expert_dim": 32}
                )
                config["model"]["interaction"].update(
                    {"kind": "cpg_expert_f2", "expert_temperature": 1.0, "token_residual_scale_init": 1.0}
                )
                config["training"]["seed"] = seed
                config["training"]["warm_start_checkpoint"] = f2_checkpoints[seed]
                config["training"]["residual_learning_rate"] = config["training"]["learning_rate"]
                _write(root, f"r1_experts{experts}_s{seed}", config)

        # R2: pathway/regulon/NMF tokens queried by the target CpG.
        for name, path in modules.items():
            for seed in seeds:
                config = deepcopy(base)
                _f2(config)
                config["data"]["rna_transform"] = "zscore"
                config["model"]["encoder"].update(
                    {
                        "kind": "module_tokens",
                        "token_dim": 64,
                        "module_weights_path": path,
                        "module_weights_key": "weights",
                    }
                )
                config["model"]["interaction"].update(
                    {"kind": "cpg_module_f2", "token_residual_scale_init": 1.0}
                )
                config["training"]["seed"] = seed
                config["training"]["warm_start_checkpoint"] = f2_checkpoints[seed]
                config["training"]["residual_learning_rate"] = config["training"]["learning_rate"]
                _write(root, f"r2_{name}_module_query_s{seed}", config)

        # R4: final direct gene-query test, sparse top-k routing to control memory.
        if not args.gene_embeddings:
            raise ValueError("--gene-embeddings is required when generating R4 configs")
        for top_k in (64, 128, 256):
            for seed in seeds:
                config = deepcopy(base)
                _f2(config)
                config["data"]["rna_transform"] = "zscore"
                config["model"]["encoder"].update(
                    {
                        "kind": "gene_tokens",
                        "token_dim": 32,
                        "gene_embedding_path": args.gene_embeddings,
                        "gene_embedding_key": "embeddings",
                        "freeze_gene_embeddings": bool(args.gene_embeddings),
                    }
                )
                config["model"]["interaction"].update(
                    {"kind": "cpg_gene_topk_f2", "token_top_k": top_k, "token_residual_scale_init": 1.0}
                )
                config["training"]["seed"] = seed
                config["training"]["warm_start_checkpoint"] = f2_checkpoints[seed]
                config["training"]["residual_learning_rate"] = config["training"]["learning_rate"]
                # Explicit gene tokens are memory-heavy; start conservatively.
                config["training"]["sample_batch_size"] = min(config["training"].get("sample_batch_size", 32), 8)
                config["training"]["cpg_batch_size"] = min(config["training"].get("cpg_batch_size", 256), 128)
                _write(root, f"r4_gene_query_k{top_k}_s{seed}", config)

    # R5: frozen pretrained sample representation alone and concatenated with raw RNA.
    for name, path in pretrained.items():
        for mode in r5_modes:
            for seed in r5_seeds:
                config = deepcopy(base)
                _f2(config)
                config["data"]["rna_transform"] = "zscore"
                config["data"]["pretrained_rna"] = {
                    "path": path,
                    "values_key": "embeddings",
                    "row_ids_key": "sample_idx",
                    "col_ids_key": "feature_ids",
                }
                config["data"]["pretrained_mode"] = "replace" if mode == "shuffled" else mode
                config["data"]["standardize_pretrained_rna"] = True
                config["data"]["pretrained_control"] = "shuffle_within_cancer" if mode == "shuffled" else "real"
                config["data"]["pretrained_control_seed"] = 20260731
                config["training"]["seed"] = seed
                config["training"]["warm_start_checkpoint"] = None
                _write(root, f"r5_{name}_{mode}_s{seed}", config)

    # R5b: genuinely nested comparison -- F2 warm-started and hard-frozen
    # (requires_grad_(False), not a zero learning rate), plus a zero-init residual
    # adapter over one BulkRNABert layer's frozen embedding. Screened one layer at a
    # time via --r5b-layers; the shuffled control is generated separately, only for
    # the winning layer, via --r5b-shuffled-layer (see docs/rna_representation_experiments.md).
    r5b_layers = [int(value) for value in args.r5b_layers.split(",") if value.strip()]
    if r5b_layers or args.r5b_shuffled_layer is not None:
        if not pretrained:
            raise ValueError("--pretrained-rna is required for R5b config generation")
        if not f2_checkpoints:
            raise ValueError("--f2-checkpoint is required for R5b config generation (nested warm start)")
        if set(r5b_layers) - {1, 2, 3, 4}:
            raise ValueError("--r5b-layers accepts only values in 1..4")

        def _r5b_config(path: str, layer: int, seed: int, shuffled: bool) -> dict:
            config = deepcopy(base)
            _f2(config)
            config["data"]["rna_transform"] = "zscore"
            config["data"]["pretrained_rna"] = {
                "path": path,
                "values_key": f"embeddings_layer{layer}",
                "row_ids_key": "sample_idx",
                "col_ids_key": "feature_ids",
            }
            config["data"]["pretrained_mode"] = "concat"
            config["data"]["standardize_pretrained_rna"] = True
            config["data"]["pretrained_control"] = "shuffle_within_cancer" if shuffled else "real"
            config["data"]["pretrained_control_seed"] = 20260731
            config["model"]["encoder"] = {"kind": "pretrained_embedding", "latent_dim": 64, "layer_norm": True}
            config["model"]["interaction"] = {
                "kind": "cpg_pretrained_f2", "hidden_dim": 128, "dropout": 0.1, "token_residual_scale_init": 1.0,
            }
            config["model"]["zero_init_residual"] = True
            config["training"]["seed"] = seed
            config["training"]["warm_start_checkpoint"] = f2_checkpoints[seed]
            config["training"]["freeze_warm_start_params"] = True
            config["training"]["seed_initial_checkpoint"] = True
            config["training"]["residual_learning_rate"] = config["training"]["learning_rate"]
            return config

        for name, path in pretrained.items():
            for layer in r5b_layers:
                for seed in r5_seeds:
                    config = _r5b_config(path, layer, seed, shuffled=False)
                    _write(root, f"r5b_{name}_layer{layer}_s{seed}", config)
            if args.r5b_shuffled_layer is not None:
                for seed in r5_seeds:
                    config = _r5b_config(path, args.r5b_shuffled_layer, seed, shuffled=True)
                    _write(root, f"r5b_{name}_concat_shuffled_layer{args.r5b_shuffled_layer}_s{seed}", config)

    print(f"Wrote configs under {root / 'configs'}")


if __name__ == "__main__":
    main()
