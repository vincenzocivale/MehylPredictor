"""Cartesian-batch training and bidimensional OOD evaluation."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import RunConfig, save_config
from .data import DataBundle, load_bundle, summarize_bundle
from .full_coverage_sampler import build_epoch_schedule
from .losses import ramp_loss_config, residual_loss, scheduled_loss_config
from .metrics import evaluate_predictions
from .models import RNA2DNAmModel
from .streaming_metrics import StreamingPanelMetrics
from .tracking import create_tracker
from .utils import atomic_torch_save, resolve_device, runtime_manifest, seed_everything, write_json


def freeze_parameters(model: nn.Module, names: Iterable[str]) -> list[str]:
    """Sets ``requires_grad_(False)`` on exactly the named parameters (typically the
    keys a warm start actually loaded from a prior checkpoint) and returns the list of
    parameter names that were frozen. A true hard freeze rather than a zero learning
    rate: frozen parameters never accumulate a gradient at all, so a large backbone
    gradient can never distort a shared ``clip_grad_norm_`` call's effect on a small
    residual branch's gradient.
    """
    names = set(names)
    frozen = []
    for name, param in model.named_parameters():
        if name in names:
            param.requires_grad_(False)
            frozen.append(name)
    return frozen


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
        if config.training.matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("training.matmul_precision must be highest, high, or medium")
        torch.set_float32_matmul_precision(config.training.matmul_precision)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = config.training.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.training.allow_tf32
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bundle: DataBundle = load_bundle(config.data, seed=config.training.seed)
        aligned_gene_embeddings = (
            torch.from_numpy(self.bundle.gene_embeddings)
            if self.bundle.gene_embeddings is not None
            else None
        )
        raw_rna_dim = self.bundle.raw_rna_dim if config.model.encoder.kind == "pretrained_embedding" else None
        self.model = RNA2DNAmModel(
            self.bundle.rna_input_dim,
            self.bundle.locus_dim,
            config.model,
            epsilon=config.data.clip_beta_epsilon,
            raw_rna_dim=raw_rna_dim,
            gene_embeddings=aligned_gene_embeddings,
        ).to(self.device)
        self.reference_rna = torch.from_numpy(self.bundle.reference_rna()[None, :]).to(self.device)
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
        if data.pretrained_rna is not None:
            paths["pretrained_rna"] = data.pretrained_rna.path
        if self.config.model.encoder.module_weights_path:
            paths["module_weights"] = self.config.model.encoder.module_weights_path
        if self.config.model.encoder.gene_embedding_path:
            paths["gene_token_embeddings"] = self.config.model.encoder.gene_embedding_path
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

    def _resolved_amp_dtype(self) -> torch.dtype | None:
        if not (self.config.training.amp and self.device.type == "cuda"):
            return None
        dtype_name = self.config.training.amp_dtype.lower()
        if dtype_name == "bfloat16":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if dtype_name == "float16":
            return torch.float16
        raise ValueError("training.amp_dtype must be float16 or bfloat16")

    def _autocast_context(self):
        dtype = self._resolved_amp_dtype()
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    @torch.no_grad()
    def _factorized_patient_cache(self, sample_indices: np.ndarray) -> torch.Tensor:
        """Compute each patient tower exactly once and keep the compact factors on CPU."""
        if not self.model.supports_factorized_inference:
            raise RuntimeError("factorized cache requested for a non-factorizable interaction")
        reference_factors = None
        if self.config.model.anchor_to_mean_rna:
            with self._autocast_context():
                reference_factors = self.model.encode_reference_factors(self.reference_rna)
        chunks: list[torch.Tensor] = []
        sample_chunk = self.config.evaluation.sample_chunk_size
        for start in range(0, len(sample_indices), sample_chunk):
            local = sample_indices[start : start + sample_chunk]
            rna = torch.from_numpy(self.bundle.rna(local)).to(self.device)
            with self._autocast_context():
                factors = self.model.encode_patient_factors(
                    rna, self.reference_rna, reference_factors=reference_factors
                )
            chunks.append(factors.float().cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def predict_panel(
        self,
        sample_split: str,
        cpg_split: str,
        max_cpgs: int | None = None,
        keep_predictions: bool = False,
        seed_offset: int = 0,
        use_factorized_cache: bool = True,
        cpg_indices_override: np.ndarray | None = None,
        sample_indices_override: np.ndarray | None = None,
    ) -> PanelResult:
        """cpg_indices_override/sample_indices_override, when given, replace the
        cpg_split/sample_split lookup (and, for CpGs, max_cpgs subsampling)
        entirely -- use this to reuse an exact, previously frozen CpG subset
        (e.g. loaded from another model's panel_*.npz `cpg_idx` field) instead
        of drawing a fresh random subsample, or to evaluate a panel spanning
        the union of several split labels (e.g. official val = validation ∪
        test) that no single `split` string selects on its own."""
        self.model.eval()
        sample_indices = (
            np.asarray(sample_indices_override, dtype=np.int64)
            if sample_indices_override is not None
            else self.bundle.sample_indices(sample_split)
        )
        cpg_indices = (
            np.asarray(cpg_indices_override, dtype=np.int64)
            if cpg_indices_override is not None
            else self.bundle.cpg_indices(cpg_split)
        )
        if not len(sample_indices):
            raise ValueError(f"no samples in split {sample_split!r}")
        if not len(cpg_indices):
            raise ValueError(f"no CpGs in split {cpg_split!r}")
        if cpg_indices_override is None and max_cpgs is not None and len(cpg_indices) > max_cpgs:
            rng = np.random.default_rng(self.config.training.seed + seed_offset)
            cpg_indices = np.sort(rng.choice(cpg_indices, size=max_cpgs, replace=False))

        target = np.full((len(sample_indices), len(cpg_indices)), np.nan, dtype=np.float32)
        prediction = np.full_like(target, np.nan)
        sample_chunk = self.config.evaluation.sample_chunk_size
        cpg_chunk = self.config.evaluation.cpg_chunk_size
        factorized = self.model.supports_factorized_inference and use_factorized_cache
        started = time.perf_counter()
        patient_factors = self._factorized_patient_cache(sample_indices) if factorized else None

        for cpg_start in range(0, len(cpg_indices), cpg_chunk):
            local_cpg = cpg_indices[cpg_start : cpg_start + cpg_chunk]
            loci = torch.from_numpy(self.bundle.loci.embeddings[local_cpg]).to(self.device)
            prior = torch.from_numpy(self.bundle.loci.prior[local_cpg]).to(self.device)
            variability = torch.from_numpy(self.bundle.loci.variability[local_cpg]).to(self.device)
            locus_state = None
            if factorized:
                with self._autocast_context():
                    locus_state = self.model.encode_locus_factors(loci, prior, variability)
            for sample_start in range(0, len(sample_indices), sample_chunk):
                local_samples = sample_indices[sample_start : sample_start + sample_chunk]
                with self._autocast_context():
                    if factorized:
                        assert patient_factors is not None and locus_state is not None
                        local_factors = patient_factors[
                            sample_start : sample_start + len(local_samples)
                        ].to(self.device, dtype=locus_state.factors.dtype)
                        outputs = self.model.predict_from_factors(local_factors, locus_state)
                    else:
                        rna = torch.from_numpy(self.bundle.rna(local_samples)).to(self.device)
                        codes = torch.from_numpy(self.bundle.cancer_type_codes[local_samples]).to(self.device)
                        outputs = self.model(
                            rna, loci, prior, variability, self.reference_rna, codes, self.cancer_centroids
                        )
                beta = self.bundle.beta(local_samples, local_cpg)
                target[
                    sample_start : sample_start + len(local_samples),
                    cpg_start : cpg_start + len(local_cpg),
                ] = beta
                prediction[
                    sample_start : sample_start + len(local_samples),
                    cpg_start : cpg_start + len(local_cpg),
                ] = outputs["beta"].float().cpu().numpy()

        elapsed = time.perf_counter() - started
        metrics = evaluate_predictions(
            target,
            prediction,
            self.bundle.loci.prior[cpg_indices],
            self.bundle.samples.cancer_types[sample_indices],
            cpg_tertiles=self.cpg_tertiles[cpg_indices],
        )
        metrics["inference_seconds"] = elapsed
        metrics["inference_pairs_per_second"] = len(sample_indices) * len(cpg_indices) / max(elapsed, 1e-9)
        metrics["factorized_inference"] = factorized
        return PanelResult(
            metrics=metrics,
            sample_indices=sample_indices,
            cpg_indices=cpg_indices,
            target=target if keep_predictions else None,
            prediction=prediction if keep_predictions else None,
        )

    @torch.no_grad()
    def evaluate_panel_streaming(
        self,
        sample_split: str | None,
        cpg_split: str | None,
        chromosome_by_cpg_id: dict[str, str] | None = None,
        sample_indices_override: np.ndarray | None = None,
        cpg_indices_override: np.ndarray | None = None,
        cpg_chunk_size: int | None = None,
        include_locus_median_correlations: bool = True,
    ) -> dict[str, object]:
        """Same GPU-chunked forward pass as `predict_panel`, but each CpG
        chunk's [n_samples, chunk_cpgs] block is fed straight into a
        `StreamingPanelMetrics` accumulator and discarded -- never
        materializes a full [n_samples, n_cpgs] array in host RAM. Use this
        for panels touching the full genome-wide train_cpg pool
        (`in_distribution`/`sample_ood`/`locus_ood`); the `double_ood` panel
        (official val_sample x val_cpg, ~66M cells) is small enough for the
        existing `predict_panel`/`evaluate_predictions` path."""
        self.model.eval()
        sample_indices = (
            np.asarray(sample_indices_override, dtype=np.int64)
            if sample_indices_override is not None
            else self.bundle.sample_indices(sample_split)
        )
        cpg_indices = (
            np.asarray(cpg_indices_override, dtype=np.int64)
            if cpg_indices_override is not None
            else self.bundle.cpg_indices(cpg_split)
        )
        if not len(sample_indices):
            raise ValueError(f"no samples in split {sample_split!r}")
        if not len(cpg_indices):
            raise ValueError(f"no CpGs in split {cpg_split!r}")

        cancer_types = self.bundle.samples.cancer_types[sample_indices]
        cancer_type_names = sorted(np.unique(cancer_types).tolist())
        if chromosome_by_cpg_id is not None:
            cpg_ids = self.bundle.loci.ids[cpg_indices].astype(str)
            chromosome_codes_all = np.asarray(
                [chromosome_by_cpg_id.get(cpg_id, "unknown") for cpg_id in cpg_ids], dtype=object
            )
        else:
            chromosome_codes_all = np.full(len(cpg_indices), "unknown", dtype=object)
        tertile_codes_all = self.cpg_tertiles[cpg_indices]

        accumulator = StreamingPanelMetrics(
            n_samples=len(sample_indices),
            cancer_type_names=cancer_type_names,
            include_locus_median_correlations=include_locus_median_correlations,
        )

        sample_chunk = self.config.evaluation.sample_chunk_size
        cpg_chunk = cpg_chunk_size or self.config.evaluation.cpg_chunk_size
        factorized = self.model.supports_factorized_inference
        started = time.perf_counter()
        patient_factors = self._factorized_patient_cache(sample_indices) if factorized else None
        # For non-factorized models (e.g. concat), RNA doesn't depend on which
        # CpG chunk is being processed, but the naive per-chunk inner loop
        # re-reads it from HDF5 + re-applies the RNA transform on EVERY CpG
        # chunk (up to ~600x redundant reads at genome-wide scale, measured
        # to cost real wall-clock time). Cache it once, CPU-resident (a few
        # hundred MB at genome-wide sample counts -- cheap), analogous to the
        # factorized path's own `_factorized_patient_cache`.
        rna_cache = None
        codes_cache = None
        if not factorized:
            rna_cache = torch.from_numpy(self.bundle.rna(sample_indices))
            codes_cache = torch.from_numpy(self.bundle.cancer_type_codes[sample_indices])

        n_cpg_chunks = (len(cpg_indices) + cpg_chunk - 1) // cpg_chunk
        progress_every = max(1, n_cpg_chunks // 20)

        for chunk_number, cpg_start in enumerate(range(0, len(cpg_indices), cpg_chunk)):
            local_cpg = cpg_indices[cpg_start : cpg_start + cpg_chunk]
            loci = torch.from_numpy(self.bundle.loci.embeddings[local_cpg]).to(self.device)
            prior = torch.from_numpy(self.bundle.loci.prior[local_cpg]).to(self.device)
            variability = torch.from_numpy(self.bundle.loci.variability[local_cpg]).to(self.device)
            locus_state = None
            if factorized:
                with self._autocast_context():
                    locus_state = self.model.encode_locus_factors(loci, prior, variability)

            # Both target and prediction are fetched/written in small
            # per-sample sub-chunks (matching predict_panel's inner loop) --
            # NOT via a single bundle.beta(sample_indices, ...) call with the
            # full sample axis, since MatrixStore._read_rows_native reads a
            # requested row's FULL width regardless of the col_indices
            # sub-selection; requesting all ~7k samples per CpG chunk would
            # re-read the entire beta matrix once per chunk (~600x at
            # genome-wide scale) instead of once per sample sub-chunk.
            target_block = np.full((len(sample_indices), len(local_cpg)), np.nan, dtype=np.float32)
            prediction_block = np.full_like(target_block, np.nan)
            for sample_start in range(0, len(sample_indices), sample_chunk):
                local_samples = sample_indices[sample_start : sample_start + sample_chunk]
                with self._autocast_context():
                    if factorized:
                        assert patient_factors is not None and locus_state is not None
                        local_factors = patient_factors[
                            sample_start : sample_start + len(local_samples)
                        ].to(self.device, dtype=locus_state.factors.dtype)
                        outputs = self.model.predict_from_factors(local_factors, locus_state)
                    else:
                        assert rna_cache is not None and codes_cache is not None
                        rna = rna_cache[sample_start : sample_start + len(local_samples)].to(self.device)
                        codes = codes_cache[sample_start : sample_start + len(local_samples)].to(self.device)
                        outputs = self.model(
                            rna, loci, prior, variability, self.reference_rna, codes, self.cancer_centroids
                        )
                row_slice = slice(sample_start, sample_start + len(local_samples))
                target_block[row_slice, :] = self.bundle.beta(local_samples, local_cpg)
                prediction_block[row_slice, :] = outputs["beta"].float().cpu().numpy()

            accumulator.add_cpg_chunk(
                target_block,
                prediction_block,
                self.bundle.loci.prior[local_cpg],
                cancer_types,
                chromosome_codes_all[cpg_start : cpg_start + len(local_cpg)],
                tertile_codes=tertile_codes_all[cpg_start : cpg_start + len(local_cpg)],
            )

            if chunk_number % progress_every == 0 or chunk_number == n_cpg_chunks - 1:
                elapsed_so_far = time.perf_counter() - started
                done = chunk_number + 1
                rate = elapsed_so_far / done
                eta = rate * (n_cpg_chunks - done)
                print(
                    f"[evaluate_panel_streaming] chunk {done}/{n_cpg_chunks} "
                    f"({100 * done / n_cpg_chunks:.0f}%), elapsed={elapsed_so_far:.0f}s, "
                    f"eta={eta:.0f}s", flush=True,
                )

        elapsed = time.perf_counter() - started
        metrics = accumulator.finalize()
        metrics["inference_seconds"] = elapsed
        metrics["inference_pairs_per_second"] = len(sample_indices) * len(cpg_indices) / max(elapsed, 1e-9)
        metrics["factorized_inference"] = factorized
        metrics["streaming"] = True
        return metrics

    @torch.no_grad()
    def export_panel_h5(
        self,
        sample_split: str,
        cpg_split: str,
        output_path: str | Path,
        include_target: bool = False,
    ) -> dict[str, object]:
        """Stream a complete panel to HDF5 without holding the N x L output in RAM."""
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - dependency is in requirements
            raise RuntimeError("streaming prediction requires h5py") from exc

        self.model.eval()
        sample_indices = self.bundle.sample_indices(sample_split)
        cpg_indices = self.bundle.cpg_indices(cpg_split)
        if not len(sample_indices) or not len(cpg_indices):
            raise ValueError("sample and CpG splits must both be non-empty")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_chunk = self.config.evaluation.sample_chunk_size
        cpg_chunk = self.config.evaluation.cpg_chunk_size
        factorized = self.model.supports_factorized_inference
        started = time.perf_counter()
        patient_factors = self._factorized_patient_cache(sample_indices) if factorized else None
        string_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(output_path, "w") as handle:
            prediction_ds = handle.create_dataset(
                "prediction", shape=(len(sample_indices), len(cpg_indices)), dtype="f4",
                chunks=(min(sample_chunk, len(sample_indices)), min(cpg_chunk, len(cpg_indices))),
                compression="gzip", compression_opts=1,
            )
            target_ds = None
            if include_target:
                target_ds = handle.create_dataset(
                    "target", shape=prediction_ds.shape, dtype="f4", chunks=prediction_ds.chunks,
                    compression="gzip", compression_opts=1,
                )
            handle.create_dataset(
                "sample_idx",
                data=np.asarray(self.bundle.samples.ids[sample_indices].astype(str), dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "cpg_idx",
                data=np.asarray(self.bundle.loci.ids[cpg_indices].astype(str), dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset("prior", data=self.bundle.loci.prior[cpg_indices].astype(np.float32))
            handle.attrs["sample_split"] = sample_split
            handle.attrs["cpg_split"] = cpg_split
            handle.attrs["factorized_inference"] = factorized

            for cpg_start in range(0, len(cpg_indices), cpg_chunk):
                local_cpg = cpg_indices[cpg_start : cpg_start + cpg_chunk]
                loci = torch.from_numpy(self.bundle.loci.embeddings[local_cpg]).to(self.device)
                prior = torch.from_numpy(self.bundle.loci.prior[local_cpg]).to(self.device)
                variability = torch.from_numpy(self.bundle.loci.variability[local_cpg]).to(self.device)
                locus_state = None
                if factorized:
                    with self._autocast_context():
                        locus_state = self.model.encode_locus_factors(loci, prior, variability)
                for sample_start in range(0, len(sample_indices), sample_chunk):
                    local_samples = sample_indices[sample_start : sample_start + sample_chunk]
                    with self._autocast_context():
                        if factorized:
                            assert patient_factors is not None and locus_state is not None
                            local_factors = patient_factors[
                                sample_start : sample_start + len(local_samples)
                            ].to(self.device, dtype=locus_state.factors.dtype)
                            outputs = self.model.predict_from_factors(local_factors, locus_state)
                        else:
                            rna = torch.from_numpy(self.bundle.rna(local_samples)).to(self.device)
                            codes = torch.from_numpy(self.bundle.cancer_type_codes[local_samples]).to(self.device)
                            outputs = self.model(
                                rna, loci, prior, variability, self.reference_rna, codes, self.cancer_centroids
                            )
                    row_slice = slice(sample_start, sample_start + len(local_samples))
                    col_slice = slice(cpg_start, cpg_start + len(local_cpg))
                    prediction_ds[row_slice, col_slice] = outputs["beta"].float().cpu().numpy()
                    if target_ds is not None:
                        target_ds[row_slice, col_slice] = self.bundle.beta(local_samples, local_cpg)

            elapsed = time.perf_counter() - started
            handle.attrs["elapsed_seconds"] = elapsed
            handle.attrs["pairs_per_second"] = len(sample_indices) * len(cpg_indices) / max(elapsed, 1e-9)

        return {
            "output": str(output_path),
            "sample_split": sample_split,
            "cpg_split": cpg_split,
            "num_samples": len(sample_indices),
            "num_cpgs": len(cpg_indices),
            "factorized_inference": factorized,
            "elapsed_seconds": elapsed,
            "pairs_per_second": len(sample_indices) * len(cpg_indices) / max(elapsed, 1e-9),
            "include_target": include_target,
        }

    def _apply_warm_start(self, checkpoint_path: str) -> dict[str, list[str]]:
        """Warm-start the model from a prior canonical checkpoint.

        Linear encoder keys retain their original names.  When the target interaction wraps
        the source model as ``interaction.baseline``, source interaction keys are remapped
        into that module; newly introduced branches stay at their configured initialization.
        """
        source_state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)["model_state"]
        target_state = self.model.state_dict()
        remapped: dict[str, torch.Tensor] = {}
        for key, value in source_state.items():
            candidates = [key]
            if key.startswith("rna_encoder."):
                candidates.append("rna_encoder.base." + key[len("rna_encoder."):])
            if key.startswith("interaction."):
                # The wrapped source checkpoint is authoritative.  New residual
                # branches can share names (and sometimes dimensions) with it.
                candidates = ["interaction.baseline." + key[len("interaction."):], key]
            # Some newly introduced residual branches intentionally reuse names
            # from the source checkpoint (for example ``locus_product``) at a
            # different width. Prefer the shape-compatible wrapped key over
            # that new branch.
            target_key = next(
                (
                    candidate for candidate in candidates
                    if candidate in target_state and target_state[candidate].shape == value.shape
                ),
                None,
            )
            if target_key is not None:
                remapped[target_key] = value
        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        allowed_missing_prefixes = (
            "rna_encoder.residual",
            "rna_encoder.expert_projection",
            "rna_encoder.module_",
            "rna_encoder.value_encoder",
            "rna_encoder.gene_identity",
            "rna_encoder.gene_identity_source",
            "rna_encoder.identity_projection",
            "interaction.expert_",
            "interaction.query",
            "interaction.key",
            "interaction.value",
            "interaction.token_product",
            "interaction.locus_product",
            "interaction.residual",
            "interaction.locus_product",
            "interaction.embedding_product",
            "interaction.patient_projection",
            "interaction.region_projection",
            "interaction.locus_loading",
            "interaction.regional_decoder",
            "interaction.learned_gate",
        )
        unaccounted = [
            name for name in missing
            if not name.startswith(allowed_missing_prefixes)
            and name not in {"rna_encoder.residual_scale", "interaction.residual_scale"}
        ]
        if unexpected:
            raise ValueError(f"warm start checkpoint has unexpected keys: {unexpected}")
        if unaccounted:
            raise ValueError(f"warm start checkpoint missing unaccounted keys: {unaccounted}")
        return {"missing": list(missing), "unexpected": list(unexpected), "loaded": sorted(remapped.keys())}

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
            command=["python", "-m", "methylation_predictor", "train"],
            inputs=self._input_paths(),
            seed=config.training.seed,
        )
        manifest["data_summary"] = summarize_bundle(self.bundle)
        manifest["runtime_optimizations"] = {
            "factorized_inference": self.model.supports_factorized_inference,
            "amp_requested": config.training.amp,
            "amp_dtype_requested": config.training.amp_dtype,
            "amp_dtype_resolved": (
                str(self._resolved_amp_dtype()).removeprefix("torch.")
                if self._resolved_amp_dtype() is not None else None
            ),
            "allow_tf32": config.training.allow_tf32,
            "matmul_precision": config.training.matmul_precision,
            "fused_adamw_requested": config.training.fused_adamw,
        }
        write_json(self.output_dir / "manifest.json", manifest)

        train_samples = self.bundle.training_sample_pool(config.training.train_sample_split)
        train_cpgs = self.bundle.training_cpg_pool(config.training.train_cpg_split)
        if not len(train_samples) or not len(train_cpgs):
            raise ValueError("training sample/CpG split is empty")

        if config.training.freeze_warm_start_params and not config.training.warm_start_checkpoint:
            raise ValueError("freeze_warm_start_params requires warm_start_checkpoint to be set")
        if config.training.seed_initial_checkpoint and not config.training.warm_start_checkpoint:
            raise ValueError("seed_initial_checkpoint requires warm_start_checkpoint to be set")

        initial_validation = None
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
            if config.training.freeze_warm_start_params:
                manifest["frozen_warm_start_params"] = freeze_parameters(self.model, warm_start_info["loaded"])
            write_json(self.output_dir / "manifest.json", manifest)

        residual_prefixes = (
            "rna_encoder.residual", "rna_encoder.expert_projection", "rna_encoder.module_",
            "rna_encoder.value_encoder", "rna_encoder.gene_identity", "rna_encoder.gene_identity_source",
            "rna_encoder.identity_projection", "interaction.expert_",
            "interaction.query", "interaction.key", "interaction.value", "interaction.token_product",
            "interaction.residual", "interaction.locus_product", "interaction.embedding_product",
        )
        residual_param_names = {
            name for name, _ in self.model.named_parameters()
            if name.startswith(residual_prefixes)
            or name in {"rna_encoder.residual_scale", "interaction.residual_scale"}
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

        optimizer_kwargs = {
            "lr": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
        }
        if config.training.fused_adamw and self.device.type == "cuda":
            optimizer_kwargs["fused"] = True
        try:
            optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
        except (TypeError, RuntimeError):
            optimizer_kwargs.pop("fused", None)
            optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
        amp_enabled = config.training.amp and self.device.type == "cuda"
        scaler_enabled = amp_enabled and self._resolved_amp_dtype() == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        history: list[dict[str, float | int]] = []
        global_step = 0
        maximize_checkpoint = config.training.checkpoint_metric in {
            "skill_vs_prior", "dynamic_skill", "within_cancer_skill",
            "dynamic_pearson", "dynamic_spearman",
            "patient_dynamic_pearson_median", "patient_dynamic_spearman_median",
            "locus_dynamic_pearson_median", "locus_dynamic_spearman_median",
            "mas_pcc", "mas_pcc_variable",
            "mas_skill_vs_prior_variable", "mean_skill_vs_prior_variable",
            "positive_skill_fraction_variable", "mas_ccc_variable",
            "mas_dynamic_r2_variable", "within_cancer_mas_pcc_variable",
            "within_cancer_mas_ccc_variable", "within_cancer_mas_dynamic_r2_variable",
            "mac_pcc_median", "mac_ccc_median",
        }
        checkpoint_path = self.output_dir / "best.pt"
        checkpoint_latest_path = self.output_dir / "checkpoint_latest.pt"
        wandb_run_id_path = self.output_dir / "wandb_run_id.txt"

        resume_state = (
            torch.load(checkpoint_latest_path, map_location=self.device, weights_only=False)
            if checkpoint_latest_path.is_file() else None
        )
        if resume_state is not None:
            self.model.load_state_dict(resume_state["model_state"])
            best_metric = resume_state["best_metric"]
            best_epoch = resume_state["best_epoch"]
            epochs_without_improvement = resume_state["epochs_without_improvement"]
            global_step = resume_state["global_step"]
            start_epoch = resume_state["epoch"] + 1
            history_path = self.output_dir / "training_history.csv"
            if history_path.is_file():
                completed = pd.read_csv(history_path)
                history = completed[completed["epoch"] <= resume_state["epoch"]].to_dict("records")
            print(
                f"[resume] resuming from completed epoch {resume_state['epoch']}; "
                f"starting epoch {start_epoch}", flush=True,
            )
        elif config.training.seed_initial_checkpoint:
            best_metric = float(initial_validation.metrics[config.training.checkpoint_metric])
            best_epoch = 0
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "epoch": 0,
                    "validation_metrics": initial_validation.metrics,
                    "config": config.as_dict(),
                },
                checkpoint_path,
            )
            epochs_without_improvement = 0
            start_epoch = 1
        else:
            best_metric = -float("inf") if maximize_checkpoint else float("inf")
            best_epoch = -1
            epochs_without_improvement = 0
            start_epoch = 1

        started = time.time()
        resume_id = wandb_run_id_path.read_text().strip() if wandb_run_id_path.is_file() else None
        tracker = create_tracker(config, self.model, self.output_dir, resume_id=resume_id)
        if tracker.enabled and not wandb_run_id_path.is_file():
            wandb_run_id_path.write_text(tracker.run.id)

        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer_state"])
            scaler.load_state_dict(resume_state["scaler_state"])
            rng_state = resume_state["rng_state"]
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            self.rng.bit_generator.state = rng_state["sampler_rng"]
            # torch.load(..., map_location=self.device) above moves EVERY tensor in the
            # checkpoint onto that device, including this one -- but torch.set_rng_state
            # requires a CPU ByteTensor specifically. Force it back regardless of
            # map_location's effect.
            torch.set_rng_state(rng_state["torch_cpu"].to(dtype=torch.uint8, device="cpu"))
            if rng_state["torch_cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(
                    [t.to(dtype=torch.uint8, device="cpu") for t in rng_state["torch_cuda"]]
                )

        try:
            for epoch in range(start_epoch, config.training.epochs + 1):
                epoch_started = time.perf_counter()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                self._refresh_train_centroids()
                self.model.train()
                epoch_loss_config = (
                    ramp_loss_config(config.loss, epoch)
                    if config.loss.ramp_epochs > 0
                    else scheduled_loss_config(config.loss, epoch)
                )
                running: dict[str, float] = {}
                observed = 0
                valid_steps = 0

                full_coverage = config.training.cpg_sampling == "full_coverage"
                schedule = (
                    build_epoch_schedule(
                        train_cpgs, train_samples,
                        config.training.cpg_batch_size, config.training.sample_batch_size,
                        epoch, config.training.seed,
                    ) if full_coverage else None
                )
                steps_this_epoch = schedule.steps_per_epoch if schedule is not None else config.training.steps_per_epoch
                if schedule is not None and epoch == start_epoch:
                    print(
                        f"[full_coverage] epoch {epoch}: steps_per_epoch={schedule.steps_per_epoch} "
                        f"(n_cpg_blocks={schedule.n_cpg_blocks}, n_sample_blocks={schedule.n_sample_blocks}); "
                        f"overriding config.training.steps_per_epoch={config.training.steps_per_epoch} "
                        "(full-coverage sampling requires this many blocks to guarantee 100% coverage)",
                        flush=True,
                    )

                for step_in_epoch in range(steps_this_epoch):
                    if schedule is not None:
                        sample_indices, cpg_indices = schedule[step_in_epoch]
                    else:
                        sample_indices = self._sample_batch(train_samples, config.training.sample_batch_size)
                        cpg_indices = self._sample_cpg_batch(train_cpgs, config.training.cpg_batch_size)
                    rna, loci, prior, variability, beta = self._tensor_inputs(sample_indices, cpg_indices)
                    if not torch.isfinite(beta).any():
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    batch_started = time.perf_counter()
                    with self._autocast_context():
                        codes = torch.from_numpy(self.bundle.cancer_type_codes[sample_indices]).to(self.device)
                        outputs = self.model(rna, loci, prior, variability, self.reference_rna, codes, self.cancer_centroids)
                        loss, pieces = residual_loss(
                            outputs,
                            beta,
                            prior,
                            epoch_loss_config,
                            epsilon=config.data.clip_beta_epsilon,
                            tertile_labels=torch.from_numpy(self.cpg_tertiles[cpg_indices]).to(self.device),
                            cancer_type_codes=codes,
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
                    for key, value in pieces.items():
                        if key == "observed":
                            continue
                        running[key] = running.get(key, 0.0) + float(value)
                    observed += int(pieces["observed"])
                    valid_steps += 1
                    global_step += 1
                    if tracker.enabled and global_step % max(1, config.tracking.log_every_steps) == 0:
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        batch_seconds = time.perf_counter() - batch_started
                        tracker.log(
                            {
                                "train/epoch": epoch,
                                "train/step_in_epoch": step_in_epoch + 1,
                                "train/loss": pieces["loss"],
                                "train/beta_mse": pieces["beta_mse"],
                                "train/beta_huber": pieces["beta_huber"],
                                "train/residual_huber": pieces["residual_huber"],
                                "train/shrinkage": pieces["shrinkage"],
                                "train/locus_pearson_loss": pieces["locus_pearson_loss"],
                                "train/locus_lower_tail_loss": pieces["locus_lower_tail_loss"],
                                "train/pairwise_difference_loss": pieces["pairwise_difference_loss"],
                                "train/centered_mse_loss": pieces["centered_mse_loss"],
                                "train/amplitude_loss": pieces["amplitude_loss"],
                                "train/global_prior_ratio_loss": pieces["global_prior_ratio_loss"],
                                "train/locus_skill_loss": pieces["locus_skill_loss"],
                                "train/locus_ccc_loss": pieces["locus_ccc_loss"],
                                "train/within_cancer_dynamic_loss": pieces["within_cancer_dynamic_loss"],
                                "train/valid_skill_loci": pieces["valid_skill_loci"],
                                "train/valid_ccc_loci": pieces["valid_ccc_loci"],
                                "train/valid_within_cancer_loci": pieces["valid_within_cancer_loci"],
                                "train/observed": pieces["observed"],
                                "train/compute_pairs_per_second": pieces["observed"] / max(batch_seconds, 1e-9),
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                            },
                            step=global_step,
                        )

                if not valid_steps:
                    raise RuntimeError("all training batches were empty")
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                epoch_seconds = time.perf_counter() - epoch_started
                row: dict[str, float | int] = {
                    "epoch": epoch,
                    "observed": observed,
                    "epoch_seconds": epoch_seconds,
                    "train_pairs_per_second": observed / max(epoch_seconds, 1e-9),
                    **{f"train_{key}": value / valid_steps for key, value in running.items()},
                }
                if self.device.type == "cuda":
                    row["gpu_peak_memory_gb"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

                if schedule is not None:
                    coverage = schedule.coverage_report()
                    row.update({f"coverage_{key}": value for key, value in coverage.items()})
                    if coverage["cpg_coverage_fraction"] < 1.0 or coverage["sample_coverage_fraction"] < 1.0:
                        raise RuntimeError(
                            f"full_coverage epoch {epoch} did not reach 100% coverage: {coverage}"
                        )

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
                        "patient_dynamic_pearson_median", "locus_dynamic_pearson_median", "mas_pcc",
                        "dynamic_amplitude_ratio", "dynamic_calibration_alpha",
                        "variable_cpgs_evaluated", "mas_skill_vs_prior_variable",
                        "mean_skill_vs_prior_variable", "positive_skill_fraction_variable",
                        "mas_pcc_variable", "mas_ccc_variable", "mas_dynamic_r2_variable",
                        "within_cancer_mas_pcc_variable", "within_cancer_mas_ccc_variable",
                        "within_cancer_mas_dynamic_r2_variable",
                        "median_amplitude_ratio_variable", "mac_pcc_median", "mac_ccc_median",
                    ]:
                        value = validation.metrics.get(key)
                        row[f"validation_{key}"] = float(value) if value is not None else float("nan")
                    tracker.log(
                        {
                            **{
                                f"validation/{key}": value
                                for key, value in validation.metrics.items()
                                if isinstance(value, (int, float, np.number, bool))
                            },
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                    current = float(validation.metrics[config.training.checkpoint_metric])
                    if config.training.save_every_epoch:
                        torch.save(
                            {"model_state": self.model.state_dict(), "epoch": epoch,
                             "validation_metrics": validation.metrics, "config": config.as_dict()},
                            self.output_dir / f"epoch_{epoch:03d}.pt",
                        )
                    global_skill = float(validation.metrics.get("skill_vs_prior", float("nan")))
                    amplitude = float(validation.metrics.get("dynamic_amplitude_ratio", float("nan")))
                    guardrail_ok = True
                    if config.training.checkpoint_min_global_skill is not None:
                        guardrail_ok &= (
                            np.isfinite(global_skill)
                            and global_skill >= config.training.checkpoint_min_global_skill
                        )
                    if config.training.checkpoint_min_amplitude_ratio is not None:
                        guardrail_ok &= (
                            np.isfinite(amplitude)
                            and amplitude >= config.training.checkpoint_min_amplitude_ratio
                        )
                    if config.training.checkpoint_max_amplitude_ratio is not None:
                        guardrail_ok &= (
                            np.isfinite(amplitude)
                            and amplitude <= config.training.checkpoint_max_amplitude_ratio
                        )
                    row["validation_checkpoint_guardrail_ok"] = bool(guardrail_ok)
                    improved_by_metric = (
                        current > best_metric + config.training.min_delta
                        if maximize_checkpoint else current < best_metric - config.training.min_delta
                    )
                    improved = (
                        True if config.training.checkpoint_selection == "final"
                        else guardrail_ok and np.isfinite(current) and improved_by_metric
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
                tracker.log(
                    {
                        **{f"epoch/{key}": value for key, value in row.items() if key != "epoch"},
                        "epoch": epoch,
                    },
                    step=global_step,
                )

                atomic_torch_save(
                    {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "rng_state": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "sampler_rng": self.rng.bit_generator.state,
                            "torch_cpu": torch.get_rng_state(),
                            "torch_cuda": (
                                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                            ),
                        },
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "epochs_without_improvement": epochs_without_improvement,
                        "config": config.as_dict(),
                    },
                    checkpoint_latest_path,
                )

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
                "best_validation_mse": float(
                    checkpoint.get("validation_metrics", {}).get("mse", float("nan"))
                ),
                "best_validation_metric": best_metric,
                "best_validation_metric_name": config.training.checkpoint_metric,
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
            final_scalars = {
                f"final/{panel_name}/{key}": value
                for panel_name, panel_values in panel_metrics.items()
                for key, value in panel_values.items()
                if isinstance(value, (int, float, np.number, bool))
            }
            tracker.log(final_scalars, step=global_step)
            tracker.set_summary({
                **final_scalars,
                "best_epoch": best_epoch,
                "best_validation_metric": best_metric,
                "elapsed_seconds": metrics["elapsed_seconds"],
                "num_parameters": metrics["num_parameters"],
                "factorized_inference": self.model.supports_factorized_inference,
            })
            if config.tracking.log_checkpoint:
                tracker.log_checkpoint(checkpoint_path, aliases=["best", f"epoch-{best_epoch}"])
            tracker.finish("finished")
            return metrics
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = repr(exc)
            manifest["elapsed_seconds"] = time.time() - started
            write_json(self.output_dir / "manifest.json", manifest)
            tracker.finish("failed")
            raise
