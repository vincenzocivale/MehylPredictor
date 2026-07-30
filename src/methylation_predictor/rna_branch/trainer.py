"""Cartesian-batch training and bidimensional OOD evaluation."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .config import RunConfig, save_config
from .data import DataBundle, load_bundle, summarize_bundle
from .losses import residual_loss
from .metrics import evaluate_predictions
from .models import ResidualMethylationModel
from .utils import resolve_device, runtime_manifest, seed_everything, write_json


@dataclass(slots=True)
class PanelResult:
    metrics: dict[str, object]
    sample_indices: np.ndarray
    cpg_indices: np.ndarray
    target: np.ndarray | None = None
    prediction: np.ndarray | None = None


class ExperimentRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        seed_everything(config.training.seed)
        self.rng = np.random.default_rng(config.training.seed)
        self.device = resolve_device(config.training.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bundle: DataBundle = load_bundle(config.data, seed=config.training.seed)
        aligned_gene_embeddings = (
            torch.from_numpy(self.bundle.gene_embeddings)
            if self.bundle.gene_embeddings is not None
            else None
        )
        self.model = ResidualMethylationModel(
            self.bundle.rna_input_dim,
            self.bundle.locus_dim,
            config.model,
            epsilon=config.data.clip_beta_epsilon,
            gene_embeddings=aligned_gene_embeddings,
        ).to(self.device)
        self.reference_rna = torch.zeros(
            (1, self.bundle.rna_input_dim), dtype=torch.float32, device=self.device
        )
        # Fit variability thresholds on train CpGs only, then apply the frozen
        # genomic proxy to every locus. This supports locus-OOD tertile metrics
        # without using held-out beta values.
        train_cpgs = self.bundle.cpg_indices(config.training.train_cpg_split)
        proxy = np.exp(self.bundle.loci.variability[:, 0]) + np.exp(self.bundle.loci.variability[:, 1])
        self.cpg_tertile_thresholds = np.quantile(proxy[train_cpgs], [1.0 / 3.0, 2.0 / 3.0])
        self.cpg_tertiles = np.digitize(proxy, self.cpg_tertile_thresholds, right=True).astype(np.int64)
        self.cancer_centroids: torch.Tensor | None = None

    @torch.no_grad()
    def _refresh_train_centroids(self) -> None:
        """Encode train samples only; validation/test never affect centroids."""
        if self.config.model.interaction.kind != "between_within":
            return
        train = self.bundle.sample_indices(self.config.training.train_sample_split)
        n_types = len(self.bundle.cancer_type_names)
        sums = torch.zeros((n_types, self.config.model.encoder.latent_dim), device=self.device)
        counts = torch.zeros(n_types, device=self.device)
        self.model.eval()
        for start in range(0, len(train), self.config.evaluation.sample_chunk_size):
            indices = train[start:start + self.config.evaluation.sample_chunk_size]
            vectors = self.model.rna_encoder(torch.from_numpy(self.bundle.rna(indices)).to(self.device)).global_vector
            codes = torch.from_numpy(self.bundle.cancer_type_codes[indices]).to(self.device)
            sums.index_add_(0, codes, vectors)
            counts.index_add_(0, codes, torch.ones_like(codes, dtype=torch.float32))
        self.cancer_centroids = sums / counts.clamp_min(1).unsqueeze(1)

    def close(self) -> None:
        self.bundle.close()

    def _input_paths(self) -> dict[str, str]:
        data = self.config.data
        paths = {
            "rna": data.rna.path,
            "methylation": data.methylation.path,
            "locus_embeddings": data.locus_embeddings.path,
            "locus_features": data.locus_features.path,
            "sample_metadata": data.sample_metadata.path,
            "cpg_splits": data.cpg_splits.path,
        }
        if data.gene_embeddings is not None:
            paths["gene_embeddings"] = data.gene_embeddings.path
        return paths

    def _tensor_inputs(
        self, sample_indices: np.ndarray, cpg_indices: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rna = torch.from_numpy(self.bundle.rna(sample_indices)).to(self.device)
        loci = torch.from_numpy(self.bundle.loci.embeddings[cpg_indices]).to(self.device)
        prior = torch.from_numpy(self.bundle.loci.prior[cpg_indices]).to(self.device)
        variability = torch.from_numpy(self.bundle.loci.variability[cpg_indices]).to(self.device)
        beta = torch.from_numpy(self.bundle.beta(sample_indices, cpg_indices)).to(self.device)
        return rna, loci, prior, variability, beta

    def _sample_batch(self, population: np.ndarray, size: int) -> np.ndarray:
        replace = len(population) < size
        return self.rng.choice(population, size=size, replace=replace)

    def _sample_cpg_batch(self, population: np.ndarray, size: int) -> np.ndarray:
        if self.config.training.cpg_sampling != "balanced_tertiles":
            return self._sample_batch(population, size)
        groups = [population[self.cpg_tertiles[population] == label] for label in range(3)]
        if any(len(group) == 0 for group in groups):
            raise ValueError("balanced_tertiles requires all three training tertiles")
        counts = [size // 3 + (idx < size % 3) for idx in range(3)]
        selected = [self._sample_batch(group, count) for group, count in zip(groups, counts)]
        return self.rng.permutation(np.concatenate(selected))

    @torch.no_grad()
    def predict_panel(
        self,
        sample_split: str,
        cpg_split: str,
        max_cpgs: int | None = None,
        keep_predictions: bool = False,
        seed_offset: int = 0,
    ) -> PanelResult:
        self.model.eval()
        sample_indices = self.bundle.sample_indices(sample_split)
        cpg_indices = self.bundle.cpg_indices(cpg_split)
        if not len(sample_indices):
            raise ValueError(f"no samples in split {sample_split!r}")
        if not len(cpg_indices):
            raise ValueError(f"no CpGs in split {cpg_split!r}")
        if max_cpgs is not None and len(cpg_indices) > max_cpgs:
            rng = np.random.default_rng(self.config.training.seed + seed_offset)
            cpg_indices = np.sort(rng.choice(cpg_indices, size=max_cpgs, replace=False))

        target = np.full((len(sample_indices), len(cpg_indices)), np.nan, dtype=np.float32)
        prediction = np.full_like(target, np.nan)
        sample_chunk = self.config.evaluation.sample_chunk_size
        cpg_chunk = self.config.evaluation.cpg_chunk_size

        for cpg_start in range(0, len(cpg_indices), cpg_chunk):
            local_cpg = cpg_indices[cpg_start : cpg_start + cpg_chunk]
            loci = torch.from_numpy(self.bundle.loci.embeddings[local_cpg]).to(self.device)
            prior = torch.from_numpy(self.bundle.loci.prior[local_cpg]).to(self.device)
            variability = torch.from_numpy(self.bundle.loci.variability[local_cpg]).to(self.device)
            for sample_start in range(0, len(sample_indices), sample_chunk):
                local_samples = sample_indices[sample_start : sample_start + sample_chunk]
                rna = torch.from_numpy(self.bundle.rna(local_samples)).to(self.device)
                codes = torch.from_numpy(self.bundle.cancer_type_codes[local_samples]).to(self.device)
                outputs = self.model(rna, loci, prior, variability, self.reference_rna, codes, self.cancer_centroids)
                beta = self.bundle.beta(local_samples, local_cpg)
                target[
                    sample_start : sample_start + len(local_samples),
                    cpg_start : cpg_start + len(local_cpg),
                ] = beta
                prediction[
                    sample_start : sample_start + len(local_samples),
                    cpg_start : cpg_start + len(local_cpg),
                ] = outputs["beta"].float().cpu().numpy()

        metrics = evaluate_predictions(
            target,
            prediction,
            self.bundle.loci.prior[cpg_indices],
            self.bundle.samples.cancer_types[sample_indices],
            cpg_tertiles=self.cpg_tertiles[cpg_indices],
        )
        return PanelResult(
            metrics=metrics,
            sample_indices=sample_indices,
            cpg_indices=cpg_indices,
            target=target if keep_predictions else None,
            prediction=prediction if keep_predictions else None,
        )

    def _apply_warm_start(self, checkpoint_path: str) -> dict[str, list[str]]:
        """Load a checkpoint from a plain (e.g. linear) encoder into a residual encoder.

        `rna_encoder.*` keys are remapped to `rna_encoder.base.*` when the current
        model exposes that submodule; every other key (interaction/gate/...) loads
        unchanged. The residual branch and its scale are intentionally left at
        their own (zero-)init, which is what makes the warm-started model produce
        exactly the source checkpoint's predictions before any training step.
        """
        source_state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)["model_state"]
        target_state = self.model.state_dict()
        remapped = {}
        for key, value in source_state.items():
            base_key = "rna_encoder.base." + key[len("rna_encoder."):] if key.startswith("rna_encoder.") else key
            remapped[base_key if base_key in target_state else key] = value
        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        expected_missing = {
            name for name in target_state
            if name.startswith("rna_encoder.residual") or name == "rna_encoder.residual_scale"
        }
        if unexpected:
            raise ValueError(f"warm start checkpoint has unexpected keys: {unexpected}")
        if set(missing) != expected_missing:
            raise ValueError(f"warm start checkpoint missing unaccounted keys: {set(missing) - expected_missing}")
        return {"missing": missing, "unexpected": unexpected}

    def _save_predictions(self, panel_name: str, result: PanelResult) -> None:
        if result.target is None or result.prediction is None:
            return
        path = self.output_dir / f"predictions_{panel_name}.npz"
        np.savez_compressed(
            path,
            target=result.target,
            prediction=result.prediction,
            prior=self.bundle.loci.prior[result.cpg_indices],
            sample_idx=self.bundle.samples.ids[result.sample_indices].astype(str),
            cpg_idx=self.bundle.loci.ids[result.cpg_indices].astype(str),
            cancer_type=self.bundle.samples.cancer_types[result.sample_indices].astype(str),
            cpg_variability_tertile=self.cpg_tertiles[result.cpg_indices],
        )

    def train(self) -> dict[str, object]:
        config = self.config
        save_config(config, self.output_dir / "config.yaml")
        manifest = runtime_manifest(
            command=["python", "-m", "methylation_predictor.rna_branch.cli", "train"],
            inputs=self._input_paths(),
            seed=config.training.seed,
        )
        manifest["data_summary"] = summarize_bundle(self.bundle)
        write_json(self.output_dir / "manifest.json", manifest)

        train_samples = self.bundle.training_sample_pool(config.training.train_sample_split)
        train_cpgs = self.bundle.training_cpg_pool(config.training.train_cpg_split)
        if not len(train_samples) or not len(train_cpgs):
            raise ValueError("training sample/CpG split is empty")

        if config.training.warm_start_checkpoint:
            warm_start_info = self._apply_warm_start(config.training.warm_start_checkpoint)
            initial_validation = self.predict_panel(
                config.training.validation_sample_split,
                config.training.validation_cpg_split,
                max_cpgs=config.training.validation_max_cpgs,
                seed_offset=101,
            )
            manifest["warm_start_checkpoint"] = config.training.warm_start_checkpoint
            manifest["warm_start_loaded_keys"] = warm_start_info
            manifest["warm_start_initial_validation_mse"] = float(initial_validation.metrics["mse"])
            write_json(self.output_dir / "manifest.json", manifest)

        residual_param_names = {
            name for name, _ in self.model.named_parameters()
            if name.startswith("rna_encoder.residual") or name == "rna_encoder.residual_scale"
        }
        if config.training.residual_learning_rate is not None:
            if not residual_param_names:
                raise ValueError("residual_learning_rate is set but the encoder has no residual branch")
            named_params = dict(self.model.named_parameters())
            param_groups = [
                {
                    "params": [p for n, p in named_params.items() if n not in residual_param_names],
                    "lr": config.training.learning_rate,
                    "name": "backbone",
                },
                {
                    "params": [p for n, p in named_params.items() if n in residual_param_names],
                    "lr": config.training.residual_learning_rate,
                    "name": "residual",
                },
            ]
        else:
            if config.training.freeze_backbone_epochs:
                raise ValueError("freeze_backbone_epochs requires residual_learning_rate to be set")
            param_groups = self.model.parameters()

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        amp_enabled = config.training.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        history: list[dict[str, float | int]] = []
        maximize_checkpoint = config.training.checkpoint_metric in {
            "skill_vs_prior", "dynamic_skill", "within_cancer_skill", "dynamic_pearson", "dynamic_spearman"
        }
        best_metric = -float("inf") if maximize_checkpoint else float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        checkpoint_path = self.output_dir / "best.pt"
        started = time.time()

        try:
            for epoch in range(1, config.training.epochs + 1):
                self._refresh_train_centroids()
                self.model.train()
                running = {"loss": 0.0, "beta_mse": 0.0, "beta_macro_mse": 0.0, "residual_huber": 0.0, "shrinkage": 0.0}
                observed = 0
                valid_steps = 0
                for _ in range(config.training.steps_per_epoch):
                    sample_indices = self._sample_batch(train_samples, config.training.sample_batch_size)
                    cpg_indices = self._sample_cpg_batch(train_cpgs, config.training.cpg_batch_size)
                    rna, loci, prior, variability, beta = self._tensor_inputs(sample_indices, cpg_indices)
                    if not torch.isfinite(beta).any():
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    autocast = (
                        torch.autocast(device_type="cuda", dtype=torch.float16)
                        if amp_enabled
                        else nullcontext()
                    )
                    with autocast:
                        codes = torch.from_numpy(self.bundle.cancer_type_codes[sample_indices]).to(self.device)
                        outputs = self.model(rna, loci, prior, variability, self.reference_rna, codes, self.cancer_centroids)
                        loss, pieces = residual_loss(
                            outputs,
                            beta,
                            prior,
                            config.loss,
                            epsilon=config.data.clip_beta_epsilon,
                            tertile_labels=torch.from_numpy(self.cpg_tertiles[cpg_indices]).to(self.device),
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite loss at epoch {epoch}; pieces={pieces}"
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.training.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    for key in running:
                        running[key] += pieces[key]
                    observed += pieces["observed"]
                    valid_steps += 1

                if not valid_steps:
                    raise RuntimeError("all training batches were empty")
                row: dict[str, float | int] = {
                    "epoch": epoch,
                    "observed": observed,
                    **{f"train_{key}": value / valid_steps for key, value in running.items()},
                }

                do_validation = epoch % config.training.validation_every == 0 or epoch == config.training.epochs
                if do_validation:
                    validation = self.predict_panel(
                        config.training.validation_sample_split,
                        config.training.validation_cpg_split,
                        max_cpgs=config.training.validation_max_cpgs,
                        seed_offset=101,
                    )
                    for key in [
                        "mse", "skill_vs_prior", "dynamic_skill", "within_cancer_skill",
                        "dynamic_pearson", "dynamic_spearman",
                        "patient_dynamic_pearson_median", "locus_dynamic_pearson_median",
                        "dynamic_amplitude_ratio", "dynamic_calibration_alpha",
                    ]:
                        value = validation.metrics.get(key)
                        row[f"validation_{key}"] = float(value) if value is not None else float("nan")
                    current = float(validation.metrics[config.training.checkpoint_metric])
                    if config.training.save_every_epoch:
                        torch.save(
                            {"model_state": self.model.state_dict(), "epoch": epoch,
                             "validation_metrics": validation.metrics, "config": config.as_dict()},
                            self.output_dir / f"epoch_{epoch:03d}.pt",
                        )
                    improved = (
                        current > best_metric + config.training.min_delta
                        if maximize_checkpoint else current < best_metric - config.training.min_delta
                    )
                    if improved:
                        best_metric = current
                        best_epoch = epoch
                        epochs_without_improvement = 0
                        torch.save(
                            {
                                "model_state": self.model.state_dict(),
                                "epoch": epoch,
                                "validation_metrics": validation.metrics,
                                "config": config.as_dict(),
                            },
                            checkpoint_path,
                        )
                    else:
                        epochs_without_improvement += 1
                history.append(row)
                pd.DataFrame(history).to_csv(self.output_dir / "training_history.csv", index=False)
                print(row, flush=True)

                if do_validation and epoch >= config.training.min_epochs and epochs_without_improvement >= config.training.patience:
                    break

            if best_epoch < 0:
                raise RuntimeError("no checkpoint was selected")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state"])

            panel_metrics: dict[str, object] = {}
            for index, (name, panel) in enumerate(config.evaluation.panels.items()):
                result = self.predict_panel(
                    panel["sample_split"],
                    panel["cpg_split"],
                    max_cpgs=config.evaluation.max_cpgs_per_panel,
                    keep_predictions=config.evaluation.save_predictions,
                    seed_offset=1000 + index,
                )
                panel_metrics[name] = result.metrics
                if config.evaluation.save_predictions:
                    self._save_predictions(name, result)

            metrics = {
                "run_name": config.run_name,
                "encoder_kind": config.model.encoder.kind,
                "best_epoch": best_epoch,
                "best_validation_mse": best_metric,
                "elapsed_seconds": time.time() - started,
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "num_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "num_encoder_parameters": sum(p.numel() for p in self.model.rna_encoder.parameters()),
                "num_interaction_parameters": sum(p.numel() for p in self.model.interaction.parameters()),
                "num_gate_parameters": sum(p.numel() for p in self.model.gate.parameters()),
                "cpg_variability_tertile_thresholds": self.cpg_tertile_thresholds.tolist(),
                "train_sample_fraction": config.data.train_sample_fraction,
                "train_cpg_fraction": config.data.train_cpg_fraction,
                "num_train_samples_used": len(train_samples),
                "num_train_cpgs_used": len(train_cpgs),
                "train_beta_mse_last_epoch": history[-1]["train_beta_mse"] if history else None,
                "panels": panel_metrics,
            }
            write_json(self.output_dir / "metrics.json", metrics)
            manifest["status"] = "complete"
            manifest["best_epoch"] = best_epoch
            manifest["elapsed_seconds"] = metrics["elapsed_seconds"]
            write_json(self.output_dir / "manifest.json", manifest)
            return metrics
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = repr(exc)
            manifest["elapsed_seconds"] = time.time() - started
            write_json(self.output_dir / "manifest.json", manifest)
            raise
