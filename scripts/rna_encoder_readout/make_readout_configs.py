#!/usr/bin/env python3
"""Generate staged RNA-only readout-search configurations."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def _write(root: Path, name: str, config: dict) -> None:
    path = root / f"{name}.yaml"
    payload = deepcopy(config)
    payload["run_name"] = name
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("p0", "p1", "p2", "p3", "p4"), required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warm-start-checkpoint", help="mean-layer2 checkpoint for new architectures")
    parser.add_argument("--winner-kind", help="fallback when no exact winner YAML is available")
    parser.add_argument("--runner-up-kind")
    parser.add_argument("--winner-config", help="preferred: exact winning YAML")
    parser.add_argument("--runner-up-config", help="exact runner-up YAML")
    parser.add_argument("--winner-checkpoint", help="checkpoint produced by --winner-config")
    parser.add_argument("--runner-up-checkpoint", help="checkpoint produced by --runner-up-config")
    parser.add_argument("--module-membership")
    parser.add_argument(
        "--gene-count-cache", action="append", default=[], metavar="COUNT=PATH",
        help="P3 cache mapping, e.g. 2048=artifacts/.../cache_2048.h5",
    )
    parser.add_argument(
        "--decoder-mode", choices=("sgd", "ridge_alternating"), default="sgd",
        help="p1 defaults to ridge_alternating regardless of this flag; kept for p0/p2+ overrides",
    )
    parser.add_argument("--ridge-refit-interval", type=int, default=1)
    parser.add_argument("--ridge-alpha-total", type=float, default=1.0)
    parser.add_argument("--ridge-alpha-within", type=float, default=1.0)
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.base).read_text())
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base.setdefault("training", {})["seed"] = args.seed

    def candidate(
        path: str | None,
        kind: str | None,
        checkpoint: str | None,
    ) -> tuple[str, dict, str | None] | None:
        if path:
            payload = yaml.safe_load(Path(path).read_text())
            return (
                str(payload.get("run_name", Path(path).stem)),
                deepcopy(payload.get("model", {})),
                checkpoint,
            )
        if kind:
            return kind, {"kind": kind, "primary_layer": 2}, checkpoint
        return None

    def emit(
        name: str,
        model_updates: dict,
        objective_updates: dict | None = None,
        checkpoint: str | None = None,
        warm_mode: str = "decoder_only",
        token_cache_path: str | None = None,
        decoder_mode: str | None = None,
    ) -> None:
        config = deepcopy(base)
        config.setdefault("model", {}).update(deepcopy(model_updates))
        if objective_updates:
            config.setdefault("objective", {}).update(objective_updates)
        if token_cache_path:
            config.setdefault("token_cache", {})["path"] = token_cache_path
        training = config.setdefault("training", {})
        training["warm_start_checkpoint"] = checkpoint
        training["warm_start_mode"] = warm_mode
        if decoder_mode == "ridge_alternating":
            training["decoder_mode"] = "ridge_alternating"
            training["ridge_refit_interval"] = args.ridge_refit_interval
            training["ridge_alpha_total"] = args.ridge_alpha_total
            training["ridge_alpha_within"] = args.ridge_alpha_within
        _write(root, name, config)

    baseline_checkpoint = args.warm_start_checkpoint
    if args.stage == "p0" and not baseline_checkpoint:
        raise SystemExit(f"{args.stage} requires --warm-start-checkpoint from mean layer 2")

    if args.stage == "p0":
        emit(
            "p0_mean_resume_layer2",
            {"kind": "mean", "primary_layer": 2},
            checkpoint=baseline_checkpoint,
            warm_mode="decoder_only",
        )
        emit("p0_mean_layer3", {"kind": "mean", "primary_layer": 3})
        emit(
            "p0_scalar_mix_cached_layers",
            {"kind": "scalar_mix", "primary_layer": 2},
            checkpoint=baseline_checkpoint,
        )
        emit(
            "p0_concat_means_cached_layers",
            {"kind": "concat_layer_means", "primary_layer": 2},
            checkpoint=baseline_checkpoint,
        )
        return print(f"wrote configs to {root}")

    if args.stage == "p1":
        # The three fixed variance-weighted poolers were already rejected by
        # the Ridge probe (all below the plain layer-2 mean) and are dropped.
        # Every remaining pooler trains under the Ridge-alternating objective:
        # a decoder warm start is meaningless here (the SGD decoder heads are
        # frozen/unused), so no --warm-start-checkpoint is required for p1.
        # Baseline reproduction at epoch 0 comes entirely from each pooler's
        # zero-initialised residual around full_mean_layer2.
        emit(
            "p1_mean_std", {"kind": "mean_std", "primary_layer": 2},
            decoder_mode="ridge_alternating",
        )
        emit(
            "p1_learned_gene_weights", {"kind": "learned_gene_weights", "primary_layer": 2},
            decoder_mode="ridge_alternating",
        )
        emit(
            "p1_gated_attentive_stats", {"kind": "gated_attentive_stats", "primary_layer": 2},
            decoder_mode="ridge_alternating",
        )
        for queries in (1, 4, 8):
            emit(
                f"p1_pma_q{queries}", {"kind": "pma", "primary_layer": 2, "num_queries": queries},
                decoder_mode="ridge_alternating",
            )
        return print(f"wrote configs to {root}")

    winner = candidate(args.winner_config, args.winner_kind, args.winner_checkpoint)
    runner_up = candidate(args.runner_up_config, args.runner_up_kind, args.runner_up_checkpoint)
    if winner is None:
        raise SystemExit(f"{args.stage} requires --winner-config/--winner-kind")
    if winner[2] is None:
        raise SystemExit(f"{args.stage} requires --winner-checkpoint for staged continuation")

    if args.stage == "p2":
        candidates = [winner] + ([runner_up] if runner_up is not None else [])
        if runner_up is not None and runner_up[2] is None:
            raise SystemExit("runner-up continuation requires --runner-up-checkpoint")
        objectives = {
            "total": {"total_weight": 1.0, "within_weight": 0.0, "consistency_weight": 0.0},
            "within": {"total_weight": 0.0, "within_weight": 1.0, "consistency_weight": 0.0},
            "joint": {"total_weight": 1.0, "within_weight": 1.0, "consistency_weight": 0.0},
        }
        if base.get("token_cache", {}).get("augmentation_path"):
            objectives["joint_consistency"] = {
                "total_weight": 1.0,
                "within_weight": 1.0,
                "consistency_weight": 0.1,
            }
        for candidate_label, model_updates, checkpoint in candidates:
            safe = candidate_label.replace("/", "_").replace(" ", "_")
            for label, objective in objectives.items():
                emit(
                    f"p2_{safe}_{label}",
                    model_updates,
                    objective,
                    checkpoint=checkpoint,
                    warm_mode="compatible",
                )
        return print(f"wrote configs to {root}")

    selected_label, selected_model, selected_checkpoint = winner
    safe = selected_label.replace("/", "_").replace(" ", "_")
    if args.stage == "p3":
        for layer in (2, 3):
            updates = deepcopy(selected_model)
            updates["primary_layer"] = layer
            emit(
                f"p3_{safe}_layer{layer}",
                updates,
                checkpoint=selected_checkpoint,
                warm_mode="compatible",
            )
        emit(
            f"p3_{safe}_scalar_mix",
            {"kind": "scalar_mix", "primary_layer": 2},
            checkpoint=baseline_checkpoint,
            warm_mode="decoder_only",
        )
        for mapping in args.gene_count_cache:
            count_text, separator, cache_path = mapping.partition("=")
            if not separator:
                raise SystemExit(f"invalid --gene-count-cache {mapping!r}; expected COUNT=PATH")
            count = int(count_text)
            updates = deepcopy(selected_model)
            updates["primary_layer"] = 2
            emit(
                f"p3_{safe}_genes{count}",
                updates,
                checkpoint=selected_checkpoint,
                warm_mode="compatible",
                token_cache_path=cache_path,
            )
        return print(f"wrote configs to {root}")

    for continuous in ("none", "raw", "bin_residual", "fourier_bin_residual"):
        updates = deepcopy(selected_model)
        updates["primary_layer"] = 2
        updates["continuous_mode"] = continuous
        emit(
            f"p4_{safe}_{continuous}",
            updates,
            checkpoint=selected_checkpoint,
            warm_mode="compatible",
        )
    if selected_model.get("kind") in {"pma", "gated_attentive_stats"}:
        emit(
            "p4_perceiver_lite",
            {"kind": "perceiver_lite", "primary_layer": 2, "num_queries": 8},
            checkpoint=selected_checkpoint,
            warm_mode="compatible",
        )
    if args.module_membership:
        emit(
            "p4_module_pma",
            {
                "kind": "module_pma",
                "primary_layer": 2,
                "num_queries": 4,
                "module_membership_path": args.module_membership,
            },
            checkpoint=selected_checkpoint,
            warm_mode="compatible",
        )
    print(f"wrote configs to {root}")


if __name__ == "__main__":
    main()
