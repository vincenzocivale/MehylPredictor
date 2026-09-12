"""Leakage-safe hyperparameter searches for the reference RNA model."""
from __future__ import annotations

import csv
import itertools
from pathlib import Path
import time

import yaml

from ..run_store import SearchStore, write_json, write_yaml
from .locus_cls_trainer import LocusCLSJointTrainer


def run_search(
    *, canonical_root, scope, recipe_path, rna_cache, registry, cpg_targets_dir,
    functional_atlas, annotation_cache, output_root, learning_rates, schedulers, max_epochs,
    prior_cache=None, bigwig_cache=None, matched_chr1_root=None, seed=17, search_id=None,
):
    store = SearchStore.create(output_root, model="rna_methylation", scope=scope, search_id=search_id)
    write_yaml(store.path / "search_config.yaml", {
        "scope": scope, "learning_rates": learning_rates, "schedulers": schedulers,
        "max_epochs": max_epochs, "seed": seed, "selection_metric": "inner/val_cpg_x_val_sample/mas_pcc",
    })
    results = []
    for lr, scheduler in itertools.product(learning_rates, schedulers):
        candidate_id = f"lr{lr:.2g}-{scheduler}-e{max_epochs}".replace("+", "")
        started = time.time()
        trainer = LocusCLSJointTrainer(
            canonical_root=canonical_root, scope=scope, recipe_path=recipe_path,
            rna_cache=rna_cache, prior_cache=prior_cache, registry=registry,
            cpg_targets_dir=cpg_targets_dir, matched_chr1_root=matched_chr1_root,
            functional_atlas=functional_atlas, annotation_cache=annotation_cache,
            bigwig_cache=bigwig_cache, functional_only=True,
            output_root=store.path / "runs", mode="development", run_id=candidate_id,
            overrides={"learning_rate": float(lr), "scheduler": scheduler, "epochs": int(max_epochs), "seed": int(seed)},
            track=False,
        )
        try:
            summary = trainer.run()
        finally:
            trainer.close()
        row = {
            "learning_rate": float(lr), "scheduler": scheduler, "max_epochs": int(max_epochs),
            "warmup_epochs": float(trainer.recipe.training.warmup_epochs),
            "min_lr_ratio": float(trainer.recipe.training.min_lr_ratio),
            "best_epoch": int(summary["best_epoch"]),
            "best_inner_double_ood_mas_pcc": float(summary["best_inner_double_ood_mas_pcc"]),
            "elapsed_seconds": time.time() - started, "run_dir": summary["run_dir"],
        }
        results.append(row)
        write_json(store.path / "runs" / f"{candidate_id}.json", row)
    results.sort(key=lambda x: x["best_inner_double_ood_mas_pcc"], reverse=True)
    fields = [
        "learning_rate", "scheduler", "max_epochs", "warmup_epochs", "min_lr_ratio",
        "best_epoch", "best_inner_double_ood_mas_pcc", "elapsed_seconds", "run_dir",
    ]
    with (store.path / "candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    selected = {"selection_metric": "inner_double_ood_mas_pcc", "best_candidate": results[0], "all_results": results}
    write_json(store.path / "selected.json", selected)
    return selected
