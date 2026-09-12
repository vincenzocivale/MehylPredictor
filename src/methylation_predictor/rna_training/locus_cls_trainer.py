"""Trainer for ``models.FeatureFusionLocusCLSModel`` -- the repo's primary/
reference RNA-methylation architecture as of 2026-09-03 (see
docs/RNA_METHYLATION.md): late fusion of two learned representations (a
locus-only "mean" branch, an RNA-conditioned "raw" branch) into a single
prediction head, replacing the earlier two-stage frozen-prior + residual
pipeline (``VarianceNormalizedResidualModel``/``RNAMethylationPredictor``,
still frozen for old-checkpoint compatibility -- see CLAUDE.md's "Model
compatibility note"). Selected via the ``shared_backbone_locus_cls_2026_09``
ablation ladder (``results/reference/ablations.yaml``) -- see that entry and
``models.FeatureFusionLocusCLSModel``'s docstring for the architecture/
selection rationale.

Trained end-to-end in one phase from random initialization -- no separate
pretraining stage, no warm start from another checkpoint (an earlier
two-phase design and "direct"/"anchored" model variants it warm-started were
superseded by this single-phase, single-head architecture; see git history
for that earlier exploration).

Wired into the stable CLI: ``scripts/train.py --model rna_methylation
--engine matched_chr1_shared_backbone`` (currently chr1-only, matched_chr1
data). Deliberately still isolated from the frozen MethylProphet benchmark
trainer (see CLAUDE.md: "the MethylProphet benchmark is isolated, not
central") -- this module owns its own pool/schedule/read_block logic,
mirroring (not sharing) ``JointRNAMethylationTrainer``'s.
"""
from __future__ import annotations

from contextlib import nullcontext
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..losses import locus_correlation_loss, masked_mean, sample_correlation_loss, within_locus_centered_mse_loss
from ..config import TrainingConfig
from ..modeling import (
    DepthResidualAblationPredictor,
    EfficientSingleAttentionPredictor,
    FunctionalBaselinePredictor,
    FunctionalGeneFFNFusionPredictor,
    GatedResidualPredictor,
    IterativeRetrievalPredictor,
    RNAEncoderComparisonPredictor,
    SingleRetrievalPredictor,
)
from ..modeling.baselines import BASELINE_VARIANTS
from ..optim import build_lr_scheduler
from ..run_store import RunStore, write_json
from ..scopes import scope_protocol
from ..storage import FunctionalLocusCache, LocusFeatureCache, RNACache, SortedIndex
from ..tcga_canonical import TCGACanonicalBundle
from .config import load_rna_recipe
from .matched_chr1_data import load_compact_scope_sources, load_matched_chr1_protocol_and_sources
from .metrics import ArrayMomentMetrics
from .schedule import SourceSchedule, interleave
from .splits import blocked_cpg_split, stratified_sample_split
from .trainer import TrainingPool, loss_config_for_source


def ordered_bounded_prefetch(executor, function, items, depth):
    """Yield ``function(item)`` results in input order with bounded look-ahead.

    A replacement is submitted before yielding the completed result, so CPU
    preparation continues while the caller performs the GPU update.  Keeping
    this policy separate from the training loop makes its ordering and bound
    independently regression-testable.
    """
    iterator = iter(items)
    pending = deque()
    for _ in range(depth):
        try:
            item = next(iterator)
        except StopIteration:
            break
        pending.append(executor.submit(function, item))
    while pending:
        wait_started = time.perf_counter()
        result = pending.popleft().result()
        wait_seconds = time.perf_counter() - wait_started
        try:
            item = next(iterator)
        except StopIteration:
            pass
        else:
            pending.append(executor.submit(function, item))
        yield result, wait_seconds


class LocusCLSJointTrainer:
    """Single model, single optimizer, single training phase. Pool/schedule/
    read_block logic mirrors JointRNAMethylationTrainer's (duplicated rather
    than shared, same isolation note as that module)."""

    def __init__(
        self,
        *,
        canonical_root: str | Path,
        scope: str,
        recipe_path: str | Path,
        feature_cache: str | Path,
        rna_cache: str | Path,
        registry: str | Path,
        cpg_targets_dir: str | Path,
        output_root: str | Path,
        matched_chr1_root: str | Path | None = None,
        trunk_hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        trunk_dropout: float = 0.1,
        use_mean_branch: bool = True,
        use_fusion_product: bool = False,
        use_raw_product: bool = True,
        include_raw_rna: bool = True,
        include_raw_cpg: bool = True,
        product_mlp: bool = False,
        fusion_init_std: float = 0.01,
        query_source: str = "ntv3",
        aux_weight: float = 0.15,
        residual_aux_weight: float = 0.15,
        raw_lr_multiplier: float = 1.0,
        mode: str = "final",
        early_stop_patience: int | None = None,
        run_id: str | None = None,
        overrides: dict | None = None,
        development_split_seed: int | None = None,
        track: bool = True,
        resume: bool = False,
        training_sources: tuple[str, ...] | None = None,
        functional_atlas: str | Path | None = None,
        annotation_cache: str | Path | None = None,
        bigwig_cache: str | Path | None = None,
        functional_only: bool = False,
    ):
        # >1.0 gives the raw/RNA branch (raw_branch, fusion, residual_head --
        # everything that only ever gets gradient through the fusion layer,
        # unlike the trunk which also gets a direct auxiliary-loss gradient
        # into h_mean) a higher effective LR to compensate for that branch
        # imbalance -- measured empirically (chr1 pair_complete, 2026-09-02).
        self.raw_lr_multiplier = float(raw_lr_multiplier)
        self.query_source = str(query_source)
        self.development_split_seed = None if development_split_seed is None else int(development_split_seed)
        if mode not in {"development", "final"}:
            raise ValueError("mode must be development or final")
        self.mode = mode
        # Stop once val_cpg_x_val_sample.mas_pcc hasn't improved for this many
        # epochs -- ablation-ladder rungs plateau well before their epoch
        # budget (observed: rung A flat from epoch ~6 through 16+ of 30), and
        # running out the full budget on every rung wastes shared-GPU time for
        # no interpretive benefit. Only meaningful in development mode (final
        # mode has no per-epoch score to compare).
        self.early_stop_patience = int(early_stop_patience) if early_stop_patience else None
        self.scope = scope
        self.root = Path(canonical_root)
        self.registry = Path(registry)
        self.recipe = load_rna_recipe(recipe_path)
        for key, value in (overrides or {}).items():
            if not hasattr(self.recipe.training, key):
                raise ValueError(f"unknown TrainingConfig override {key!r}")
            setattr(self.recipe.training, key, value)
        cfg = self.recipe.training
        self.seed = int(cfg.seed)
        self.epochs = int(cfg.epochs)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("locus-CLS fine-tuning requires CUDA")
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32

        self.training_sources = tuple(training_sources) if training_sources else ("array", "epic", "wgbs")
        self.matched_chr1_root = Path(matched_chr1_root) if matched_chr1_root else None
        if self.matched_chr1_root is not None:
            if scope == "chr1":
                self.bundle = None
                self.protocol, self._sources = load_matched_chr1_protocol_and_sources(
                    self.matched_chr1_root, self.root, hdf5_cache_mb=cfg.hdf5_cache_mb,
                    sources=self.training_sources,
                )
            else:
                # The canonical bundle is needed only to resolve the frozen
                # protocol. Do not retain its four HDF5 raw-chunk caches while
                # the protocol-ordered compact sources are open for training.
                canonical_bundle = TCGACanonicalBundle.from_root(self.root, hdf5_cache_mb=cfg.hdf5_cache_mb)
                try:
                    self.protocol = scope_protocol(scope, canonical_bundle, canonical_root=self.root)
                finally:
                    canonical_bundle.close()
                self.bundle = None
                self._sources = load_compact_scope_sources(
                    self.matched_chr1_root, self.protocol, hdf5_cache_mb=cfg.hdf5_cache_mb,
                )
        else:
            self.bundle = TCGACanonicalBundle.from_root(self.root, hdf5_cache_mb=cfg.hdf5_cache_mb)
            self.protocol = scope_protocol(scope, self.bundle, canonical_root=self.root)
            self._sources = self.bundle.sources
        self.features = LocusFeatureCache(feature_cache)
        self.rna = RNACache(rna_cache)
        required = np.unique(np.concatenate([
            self.protocol.array_train_cpg_idx, self.protocol.array_val_cpg_idx,
            *self.protocol.auxiliary_cpg_idx.values(),
        ]))
        self.features.index.positions_of(required)
        if (functional_atlas is None) != (annotation_cache is None):
            raise ValueError("--functional-atlas and --annotation-cache must be provided together")
        self.functional = (
            FunctionalLocusCache(functional_atlas, annotation_cache, bigwig_cache)
            if functional_atlas is not None else None
        )
        self.functional_only = bool(functional_only)
        if self.functional_only and self.functional is None:
            raise ValueError("--functional-only requires both functional caches")
        self.functional_fusion_variant = self.recipe.model.functional_fusion_variant
        if self.functional_fusion_variant and not self.functional_only:
            raise ValueError("model.functional_fusion_variant requires --functional-only")
        if self.functional is not None:
            required_axes = {
                "array_train": self.protocol.array_train_cpg_idx,
                "array_val": self.protocol.array_val_cpg_idx,
                "epic_train": self.protocol.auxiliary_cpg_idx.get("epic", np.empty(0, np.int64)),
                "wgbs_train": self.protocol.auxiliary_cpg_idx.get("wgbs", np.empty(0, np.int64)),
            }
            for axis_name, axis in required_axes.items():
                try:
                    self.functional.index.positions_of(axis)
                except KeyError as exc:
                    raise ValueError(f"functional cache does not cover {axis_name}") from exc

        self.aux_weight = float(aux_weight)
        self.residual_aux_weight = float(residual_aux_weight)
        # Stashed as attributes (not just forwarded to the model constructor below)
        # so _save_checkpoint can persist them into the checkpoint's own "locus_cls"
        # dict -- evaluate_official_split rebuilds the model from that dict alone,
        # and previously silently fell back to the constructor defaults (256/64)
        # for any checkpoint trained with a non-default trunk_hidden_dim/
        # bottleneck_dim, causing a state_dict shape-mismatch load failure.
        self.trunk_hidden_dim = int(trunk_hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        cpg_targets_dir = Path(cpg_targets_dir)
        self.cpg_target_ids = np.load(cpg_targets_dir / "cpg_idx.npy")
        self.cpg_target_mu = np.load(cpg_targets_dir / "target_mu.npy")
        self.cpg_target_index = SortedIndex(self.cpg_target_ids, "cpg_statistics targets")

        self.use_mean_branch = bool(use_mean_branch)
        self._fusion_init_std = float(fusion_init_std)
        # Paper-facing functional-locus models only. The historical
        # FeatureFusion shared-backbone family has been removed from the repo.
        # ablation_depth1_residual / ablation_depth4_noresidual: the
        # depth-vs-residual ablation (docs/RNA_METHYLATION.md) disentangling
        # J0/J1's two confounded architectural axes -- see
        # modeling/ablation.py's module docstring.
        self.ablation_variants = {
            "ablation_depth1_residual": {"n_blocks": 1, "attn_residual": True},
            "ablation_depth4_noresidual": {"n_blocks": 4, "attn_residual": False},
            # 2026-09-11: how far the classic (attn_residual=True) J1-style
            # depth ladder can go on one RTX PRO 5000 -- see the VRAM-scaling
            # note this same date in modeling/ablation.py's module docstring.
            # Not run yet; smoke-test each before the full 80-epoch launch.
            "ablation_depth8_residual": {"n_blocks": 8, "attn_residual": True},
            "ablation_depth10_residual": {"n_blocks": 10, "attn_residual": True},
            "ablation_depth12_residual": {"n_blocks": 12, "attn_residual": True},
        }
        # efficient_single_attn_residual_ffn: J4, a follow-up candidate (not
        # a diagnostic cell) once ablation_depth1_residual supports "the
        # residual matters, not the repeated attention" -- single
        # cross-attention + residual, then n_ffn_blocks FFN-only residual
        # blocks (default 4, matching J1's total FFN depth). See
        # modeling/ablation.py's module docstring.
        self.efficient_variants = {
            "efficient_single_attn_residual_ffn": {"n_ffn_blocks": 4},
            # J7 (2026-09-12): ablation_depth8_residual (J5) is outperforming
            # J1 (depth 4) -- if it's the FFN/residual depth that matters and
            # not the repeated cross-attention (the expensive op this whole
            # family is testing), 1x attention + 8 cheap FFN blocks should
            # recover most of depth8's gain without paying for 8x
            # cross-attention. See modeling/ablation.py's module docstring.
            "efficient_single_attn_8ffn_residual": {"n_ffn_blocks": 8},
            # J8 (retrieval-only 16-FFN) was implemented and queued
            # 2026-09-12 but never launched, then dropped: J7 (8 FFN) beat
            # J4 (4 FFN) by only ~0.0005 MAS-PCC on the headline view --
            # noise-level -- so pushing retrieval-only FFN depth further
            # wasn't worth the GPU time. See git history before this
            # removal if the retrieval-only axis needs revisiting. J9b's
            # early training-loss signal (functional-branch depth) looked
            # more promising at the time of this call.
            # J9b (2026-09-12): J7/J8 added FFN depth to the RNA-conditioned
            # retrieval branch; this asks the same question about the OTHER
            # branch -- does FFN depth on the functional-annotation branch
            # (h_c, from track_embedding+dense_encoder) also help? Retrieval
            # depth held fixed at J7's 8 (the best cost/benefit point found
            # so far) so a gain here is attributable to the functional branch
            # alone. deep_query stays False -- the cross-attention query is
            # still the shallow h_c, unchanged from J7 -- see
            # EfficientSingleAttentionPredictor's docstring. J9c (query fed
            # by the deepened h_c) is the deliberate follow-up if this wins.
            "efficient_single_attn_8ffn_residual_functional8": {
                "n_ffn_blocks": 8, "n_functional_ffn_blocks": 8, "deep_query": False,
            },
            # J10 (2026-09-12): symmetric-depth control -- 4 FFN blocks on
            # EACH branch (retrieval matching J4's depth exactly, functional
            # matching it too) instead of J9b's asymmetric 8/8 (retrieval
            # held at J7's depth). Answers a different question than J9b:
            # not "does functional depth help on top of J7's already-deep
            # retrieval", but "at matched, moderate depth on both branches,
            # is functional-branch depth still worth it over J4 alone".
            # deep_query stays False, same isolation rationale as J9b.
            "efficient_single_attn_4ffn_residual_functional4": {
                "n_ffn_blocks": 4, "n_functional_ffn_blocks": 4, "deep_query": False,
            },
        }
        # ablation_depth1_gated_residual: J6, Flamingo-style learned scalar
        # gate on J1's per-block residual add instead of an unconditional
        # one (n_blocks=1, matching ablation_depth1_residual's depth). See
        # modeling/ablation.py's module docstring (GatedResidualPredictor).
        self.gated_variants = {
            "ablation_depth1_gated_residual": {"n_blocks": 1},
        }
        # ffn_fusion_*: J9 (2026-09-12), 2 extra FFN residual blocks on EACH
        # branch (functional locus branch + gene-expression/RNA-attention
        # branch) before fusion, testing the final recombination mechanism
        # itself -- concat (baseline) vs. FiLM vs. two-stream residual
        # mixing -- as the sole varying axis. See
        # modeling/ablation.py's FunctionalGeneFFNFusionPredictor docstring.
        self.ffn_fusion_variants = {
            "ffn_fusion_concat": {"fusion_mode": "concat"},
            "ffn_fusion_film": {"fusion_mode": "film"},
            "ffn_fusion_two_stream_residual": {"fusion_mode": "two_stream_residual"},
            # ffn_fusion_two_stream_residual_{8_8,4_4} (2026-09-12): candidate
            # "definitive model" scale-up of the two_stream_residual ladder
            # cell above, once it's judged the winning fusion mechanism --
            # depth raised symmetrically on both branches from the ladder's
            # 2/2 to 8/8 (heavier) and 4/4 (lighter), n_head_ffn_blocks left
            # at the default 2 ("due FFN" before beta prediction, unchanged).
            # fusion_dropout=0.1 (matching this recipe family's usual
            # enc.dropout/final_regressor_dropout scale) on the two
            # cross-injection projections, which the ladder cells leave at
            # the safe (no-op) default -- see FunctionalGeneFFNFusionPredictor's
            # docstring for why the extra branch depth here makes that a
            # sensible place to add regularization.
            "ffn_fusion_two_stream_residual_8_8": {
                "fusion_mode": "two_stream_residual",
                "n_functional_ffn_blocks": 8,
                "n_gene_expr_ffn_blocks": 8,
                "fusion_dropout": 0.1,
            },
            "ffn_fusion_two_stream_residual_4_4": {
                "fusion_mode": "two_stream_residual",
                "n_functional_ffn_blocks": 4,
                "n_gene_expr_ffn_blocks": 4,
                "fusion_dropout": 0.1,
            },
        }
        allowed_variants = {
            "mas_concat_v3_purecontext",
            "mas_concat_v4_iterative",
            "functional_rna_encoder_comparison",
            *BASELINE_VARIANTS,
            *self.ablation_variants,
            *self.efficient_variants,
            *self.gated_variants,
            *self.ffn_fusion_variants,
        }
        if not self.functional_only or self.functional is None:
            raise ValueError(
                "paper-facing RNA training requires --functional-only plus "
                "--functional-atlas and --annotation-cache"
            )
        if self.functional_fusion_variant not in allowed_variants:
            raise ValueError(
                "unsupported paper-facing functional_fusion_variant "
                f"{self.functional_fusion_variant!r}; expected one of "
                f"{sorted(allowed_variants)}"
            )
        if residual_aux_weight != 0.0:
            raise ValueError(
                f"{self.functional_fusion_variant} requires "
                "residual_aux_weight=0"
            )
        if self.aux_weight < 0.0:
            raise ValueError("aux_weight must be non-negative")
        if self.aux_weight != 0.0 and not use_mean_branch:
            raise ValueError(
                f"{self.functional_fusion_variant} cannot use a nonzero "
                "aux_weight when use_mean_branch=false"
            )
        if self.query_source != "ntv3":
            raise ValueError(
                "legacy query_source variants were removed with the "
                "shared-backbone model family"
            )
        if self.raw_lr_multiplier != 1.0:
            raise ValueError(
                "raw_lr_multiplier was a shared-backbone ablation and is "
                "not supported by paper-facing functional models"
            )

        final_regressor_dropout = float(
            self.recipe.raw.get("locus_cls", {}).get(
                "final_regressor_dropout", 0.0
            )
        )

        if self.functional_fusion_variant in BASELINE_VARIANTS:
            self.architecture_label = self.functional_fusion_variant
            self.model = FunctionalBaselinePredictor(
                self.rna.values.shape[1],
                self.recipe.model,
                variant=self.functional_fusion_variant,
                final_regressor_dropout=final_regressor_dropout,
                use_mean_proxy=use_mean_branch,
            ).to(self.device)
        elif (
            self.functional_fusion_variant
            == "functional_rna_encoder_comparison"
        ):
            source = self.recipe.model.encoder.frozen_embedding_source
            suffix = (
                f"-{source}"
                if source
                else f"-{self.recipe.model.encoder.kind}"
            )
            self.architecture_label = (
                "functional_rna_encoder_comparison" + suffix
            )
            self.model = RNAEncoderComparisonPredictor(
                self.rna.values.shape[1],
                self.recipe.model,
                final_regressor_dropout=final_regressor_dropout,
                use_mean_proxy=use_mean_branch,
            ).to(self.device)
        elif self.functional_fusion_variant in self.ablation_variants:
            self.architecture_label = f"functional_concat_{self.functional_fusion_variant}"
            self.model = DepthResidualAblationPredictor(
                self.rna.values.shape[1],
                self.recipe.model,
                final_regressor_dropout=final_regressor_dropout,
                use_mean_proxy=use_mean_branch,
                **self.ablation_variants[self.functional_fusion_variant],
            ).to(self.device)
        elif self.functional_fusion_variant in self.efficient_variants:
            self.architecture_label = f"functional_concat_{self.functional_fusion_variant}"
            self.model = EfficientSingleAttentionPredictor(
                self.rna.values.shape[1],
                self.recipe.model,
                final_regressor_dropout=final_regressor_dropout,
                use_mean_proxy=use_mean_branch,
                **self.efficient_variants[self.functional_fusion_variant],
            ).to(self.device)
        elif self.functional_fusion_variant in self.gated_variants:
            self.architecture_label = f"functional_concat_{self.functional_fusion_variant}"
            self.model = GatedResidualPredictor(
                self.rna.values.shape[1],
                self.recipe.model,
                final_regressor_dropout=final_regressor_dropout,
                use_mean_proxy=use_mean_branch,
                **self.gated_variants[self.functional_fusion_variant],
            ).to(self.device)
        elif self.functional_fusion_variant in self.ffn_fusion_variants:
            self.architecture_label = f"functional_concat_{self.functional_fusion_variant}"
            self.model = FunctionalGeneFFNFusionPredictor(
                self.rna.values.shape[1],
                self.recipe.model,
                final_regressor_dropout=final_regressor_dropout,
                use_mean_proxy=use_mean_branch,
                **self.ffn_fusion_variants[self.functional_fusion_variant],
            ).to(self.device)
        else:
            candidate_cls = (
                IterativeRetrievalPredictor
                if self.functional_fusion_variant
                == "mas_concat_v4_iterative"
                else SingleRetrievalPredictor
            )
            self.architecture_label = (
                f"functional_concat_{self.functional_fusion_variant}"
            )
            self.model = candidate_cls(
                self.rna.values.shape[1],
                self.recipe.model,
                final_regressor_dropout=final_regressor_dropout,
                use_mean_proxy=use_mean_branch,
            ).to(self.device)
        self.train_model = (
            torch.compile(self.model, mode=cfg.compile_mode) if cfg.compile else self.model
        )

        self.inner_views = None
        self.pools = self._build_pools()
        if track and not self.functional_only:
            self.features.validate_training_split(self.pools[0].cpg_idx)
        self.store = RunStore.create(
            output_root, model="locus_cls_joint", train_scope=scope, seed=self.seed,
            learning_rate=cfg.learning_rate, scheduler=cfg.scheduler, epochs=self.epochs, run_id=run_id,
            resume=resume,
        )
        locus_resolved = {
            "use_mean_branch": use_mean_branch, "use_fusion_product": use_fusion_product,
            "use_raw_product": use_raw_product, "product_mlp": product_mlp,
            "include_raw_rna": include_raw_rna, "include_raw_cpg": include_raw_cpg,
            "fusion_init_std": fusion_init_std, "aux_weight": self.aux_weight,
            "residual_aux_weight": self.residual_aux_weight, "raw_lr_multiplier": self.raw_lr_multiplier,
            "trunk_hidden_dim": trunk_hidden_dim, "bottleneck_dim": bottleneck_dim,
        }
        # Preserve exact resolved-config compatibility for every pre-patch run:
        # the default NTv3 query is implicit, just as it was before this patch.
        if self.query_source != "ntv3":
            locus_resolved["query_source"] = self.query_source
        resolved_config = {**self.recipe.raw, "training": asdict(cfg), "locus_cls": locus_resolved}
        if not self.functional_only and self.features.regulatory_provenance is not None:
            resolved_config["regulatory_feature_cache"] = self.features.regulatory_provenance
        if self.functional is not None:
            resolved_config["functional_locus"] = {
                "functional_atlas": str(self.functional.functional_atlas_root.resolve()),
                "annotation_cache": str(self.functional.annotation_cache_root.resolve()),
                "n_tracks": 4165, "dense_dim": 23,
                "encoder_dim": 256 if self.functional_fusion_variant else 64,
                **({"fusion_variant": self.functional_fusion_variant} if self.functional_fusion_variant else {}),
                "residual_policy": (
                    "random_initialized_functional_projection" if self.functional_only
                    else "zero_initialized_additive_projection"
                ),
                "mode": "functional_only" if self.functional_only else "additive_residual",
            }
        # Same preserve-compatibility convention as query_source above: only record
        # training_sources when it's a real restriction (paper section B.6's source
        # ablation), so every run saved before this parameter existed still resumes
        # (its own config.resolved.yaml has no such key, and the default here matches
        # what those runs actually trained with).
        if self.training_sources != ("array", "epic", "wgbs"):
            resolved_config["training_sources"] = list(self.training_sources)
        if self.development_split_seed is not None:
            resolved_config["development"] = {
                **dict(resolved_config.get("development", {})),
                "split_seed": self.development_split_seed,
            }
        if self.store.is_new:
            self.store.save_resolved_config(resolved_config)
        else:
            import yaml as _yaml
            existing = _yaml.safe_load((self.store.path / "config.resolved.yaml").read_text()) or {}
            existing = {**existing, "training": asdict(TrainingConfig(**existing.get("training", {})))}
            if existing != resolved_config:
                raise RuntimeError("resume requested with a different resolved shared-backbone recipe")
            if not self.store.checkpoint("last.pt").is_file():
                raise RuntimeError("resume requested but checkpoints/last.pt is missing")

        # wandb: one run per training call, resumed later by evaluate_official_split
        # so the official-split numbers land as `val/*` in the same run instead of a
        # disconnected second entry -- see that function and _save_checkpoint below,
        # which persists (project, entity, run_id) into the checkpoint for exactly
        # that resume. `track=False` is used by evaluate_official_split's own scratch
        # trainer instance so evaluation-only re-instantiation never opens a second,
        # spurious training run.
        self.wandb_run = None
        tracking = self.recipe.tracking
        resume_wandb_id = None
        if not self.store.is_new:
            existing_meta = json.loads((self.store.path / "metadata.json").read_text())
            resume_wandb_id = (existing_meta.get("wandb") or {}).get("run_id")
        if track and tracking.backend == "wandb" and tracking.mode != "disabled":
            import wandb
            kwargs = {
                "project": tracking.project, "entity": tracking.entity,
                "group": tracking.group or self.architecture_label,
                "name": tracking.name or self.store.run_id, "job_type": tracking.job_type,
                "mode": tracking.mode, "dir": str(self.store.path),
                # wandb caps each tag at 64 chars; architecture_label can exceed that
                # once ablation-variant suffixes (e.g. query-source) are appended, so
                # truncate only the tag copy -- group/config keep the full label.
                "tags": [*tracking.tags, f"scope-{scope}", f"mode-{mode}", self.architecture_label[:64]],
                "config": {
                    "model": asdict(self.recipe.model), "loss": asdict(self.recipe.loss),
                    "training": asdict(self.recipe.training), "schedule_policy": self.recipe.schedule_policy,
                    "locus_cls": {
                        "use_mean_branch": use_mean_branch, "use_fusion_product": use_fusion_product,
                        "use_raw_product": use_raw_product, "product_mlp": product_mlp,
                        "include_raw_rna": include_raw_rna, "include_raw_cpg": include_raw_cpg,
                        "query_source": self.query_source,
                        "fusion_init_std": fusion_init_std, "aux_weight": self.aux_weight,
                        "residual_aux_weight": self.residual_aux_weight, "raw_lr_multiplier": self.raw_lr_multiplier,
                        "development_split_seed": self.development_split_seed,
                    },
                    "scope": scope, "mode": mode, "seed": self.seed, "architecture": self.architecture_label,
                    **({"functional_locus": resolved_config["functional_locus"]} if self.functional is not None else {}),
                },
                "id": resume_wandb_id,
                "resume": "allow" if resume_wandb_id else None,
            }
            self.wandb_run = wandb.init(**{k: v for k, v in kwargs.items() if v is not None})
        if self.store.is_new:
            self.store.save_metadata({
                "architecture": self.architecture_label, "seed": self.seed,
                "training": asdict(cfg),
                **({"functional_locus": resolved_config["functional_locus"]} if self.functional is not None else {}),
                "wandb": None if self.wandb_run is None else {
                    "project": self.wandb_run.project, "entity": self.wandb_run.entity, "run_id": self.wandb_run.id,
                },
            })

    def close(self) -> None:
        if self.bundle is not None:
            self.bundle.close()
            for source in self._sources.values():
                if source not in self.bundle.sources.values():
                    source.close()
        else:
            for source in self._sources.values():
                source.close()

    def _build_pools(self) -> list[TrainingPool]:
        p = self.protocol
        if self.mode == "development":
            frac = float(self.recipe.raw.get("development", {}).get("fraction", 0.1))
            block_bp = int(self.recipe.raw.get("development", {}).get("block_bp", 5_000_000))
            split_seed = self.seed if self.development_split_seed is None else self.development_split_seed
            train_s, val_s = stratified_sample_split(
                canonical_root=self.root, sample_ids=p.array_train_sample_idx, val_fraction=frac, seed=split_seed,
            )
            train_c, val_c = blocked_cpg_split(
                registry=self.registry, cpg_ids=p.array_train_cpg_idx, val_fraction=frac, seed=split_seed, block_bp=block_bp,
            )
            self.inner_views = {
                "train_cpg_x_val_sample": (val_s, train_c),
                "val_cpg_x_train_sample": (train_s, val_c),
                "val_cpg_x_val_sample": (val_s, val_c),
            }
            array_rows = self._sources["array"].rows_of_samples(train_s)
            pools = [TrainingPool("array", array_rows, train_s, train_c)]
            forbidden = val_s
        else:
            array_rows = self._sources["array"].rows_of_samples(p.array_train_sample_idx)
            pools = [TrainingPool("array", array_rows, p.array_train_sample_idx, p.array_train_cpg_idx)]
            forbidden = p.array_val_sample_idx if self.recipe.exclude_official_val_from_auxiliary else np.empty(0, np.int64)

        for name in ("epic", "wgbs"):
            if name not in p.sources:
                continue
            source = self._sources[name]
            rows = np.arange(source.n_rows, dtype=np.int64)
            if len(forbidden):
                rows = rows[~np.isin(source.sample_idx[rows], forbidden)]
            cpg = np.asarray(p.auxiliary_cpg_idx.get(name, []), np.int64)
            if self.mode == "development":
                cpg = cpg[np.isin(cpg, pools[0].cpg_idx)]
            if len(rows) and len(cpg):
                pools.append(TrainingPool(name, rows, source.sample_idx[rows], cpg))
        # ``contiguous_blocks`` refers to slots in each pool. Protocol arrays
        # are split/shuffled ID lists, so those slots are not necessarily
        # contiguous in the underlying HDF5. Sort each axis once by its physical
        # position. Pair-complete still visits the exact same Cartesian pairs
        # and computes identical per-block losses; reads become bounded local
        # rectangles instead of expensive HDF5 fancy selections.
        if self.recipe.training.schedule_layout == "contiguous_blocks":
            local_pools = []
            for pool in pools:
                source = self._sources[pool.name]
                row_order = np.argsort(pool.row_positions, kind="stable")
                cpg_order = np.argsort(source.cpg_positions(pool.cpg_idx), kind="stable")
                local_pools.append(TrainingPool(
                    pool.name,
                    np.asarray(pool.row_positions[row_order], np.int64),
                    np.asarray(pool.sample_idx[row_order], np.int64),
                    np.asarray(pool.cpg_idx[cpg_order], np.int64),
                ))
            pools = local_pools
        return pools

    def _autocast(self):
        if not self.recipe.training.amp:
            return nullcontext()
        requested = self.recipe.training.amp_dtype.lower()
        dtype = torch.bfloat16 if requested == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _schedules(self, epoch: int):
        schedules = []
        for i, pool in enumerate(self.pools):
            batch = self.recipe.batching[pool.name]
            schedules.append(SourceSchedule(
                len(pool.row_positions), len(pool.cpg_idx), int(batch["sample_size"]), int(batch["cpg_size"]),
                epoch, self.seed + 1009 * i, self.recipe.schedule_policy, self.recipe.training.schedule_layout,
            ))
        return schedules, interleave(schedules, seed=self.seed, epoch=epoch)

    def _read_block(self, pool: TrainingPool, row_slots, cpg_slots):
        rows = pool.row_positions[row_slots]
        cpg = pool.cpg_idx[cpg_slots]
        source = self._sources[pool.name]
        beta = source.block(rows, cpg)
        return pool.sample_idx[row_slots], cpg, beta

    @staticmethod
    def _pinned(array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        return tensor.pin_memory() if torch.cuda.is_available() else tensor

    def _prepare_step(self, source_i: int, row_slots, cpg_slots):
        """Read and stage one complete CPU batch in the prefetch thread."""
        started = time.perf_counter()
        pool = self.pools[source_i]
        sample_ids, cpg_ids, beta_np = self._read_block(pool, row_slots, cpg_slots)
        finite_count = int(np.isfinite(beta_np).sum())
        pair_slots = int(beta_np.size)
        if finite_count == 0:
            return source_i, sample_ids, cpg_ids, None, None, None, None, finite_count, pair_slots, time.perf_counter() - started
        rna = self._pinned(self.rna.rows(sample_ids, dtype=np.float16))
        emb_np = (
            np.empty((len(cpg_ids), 0), dtype=np.float16)
            if self.functional_only else self.features.get(cpg_ids, embedding_dtype=np.float16)[0]
        )
        emb = self._pinned(emb_np)
        beta = self._pinned(beta_np)
        functional = None
        if self.functional is not None:
            functional = {key: self._pinned(value) for key, value in self.functional.get(cpg_ids).items()}
        return source_i, sample_ids, cpg_ids, rna, emb, beta, functional, finite_count, pair_slots, time.perf_counter() - started

    def _mean_aux_loss_raw_beta(self, cpg_ids: np.ndarray, mu_hat: torch.Tensor) -> torch.Tensor:
        """Raw-probability-space counterpart of ``_mean_aux_loss`` (which is
        logit-space): ``MSE(mu_hat_c, train_patient_mean_beta_c)`` directly,
        no logit transform -- the H0/H1/H2 sample-wise-Pearson experiment's
        ``L_mean`` term (``FunctionalConcatMASModel.mean_head`` is already
        sigmoid'd, so both sides of this MSE live in [0, 1])."""
        present = self.cpg_target_index.contains(cpg_ids)
        if not present.any():
            return mu_hat.sum() * 0.0
        rows = self.cpg_target_index.positions_of(cpg_ids[present])
        target_mu = torch.from_numpy(self.cpg_target_mu[rows].astype(np.float32)).to(mu_hat.device)
        present_t = torch.from_numpy(present).to(mu_hat.device)
        return F.mse_loss(mu_hat[present_t], target_mu)

    def _mas_concat_loss(self, out: dict, target_beta: torch.Tensor, cpg_ids: np.ndarray, loss_cfg) -> tuple[torch.Tensor, dict]:
        """``FunctionalConcatMASModel``'s own loss (H0/H1/H2 sample-wise-
        Pearson experiment, extended with the within-locus-centered-MSE and
        locus-wise-PCC follow-ups): ``L = L_beta_MSE + lambda_mean*L_mean +
        lambda_mas*L_sample_PCC + lambda_centered*L_within_locus_centered_MSE
        + lambda_MAC*L_locus_PCC``. Every term past beta_mse is weight-gated
        (each new recipe sets only the ones its experiment actually wants
        nonzero); ``L_locus_PCC`` reuses the reference architecture's own
        ``locus_correlation_loss`` (``rho_c = corr_p(beta_hat, beta)`` for
        fixed CpG c -- a locus-only shortcut has zero across-patient variance
        for any c, so this term cannot be improved by ignoring RNA)."""
        mask = torch.isfinite(target_beta)
        prediction = out["beta"]
        safe_target = torch.where(mask, target_beta, torch.zeros_like(target_beta))
        beta_mse = masked_mean((prediction - safe_target) ** 2, mask)
        if self.aux_weight != 0.0:
            if out["mu_hat"] is None:
                raise RuntimeError(
                    "mean-proxy supervision requested but model has no mean head"
                )
            mean_aux_loss = self._mean_aux_loss_raw_beta(
                cpg_ids, out["mu_hat"]
            )
        else:
            mean_aux_loss = prediction.sum() * 0.0
        sample_pcc_loss, mean_rho_p, n_valid_samples = sample_correlation_loss(prediction, safe_target, mask, loss_cfg)
        centered_mse, n_valid_loci = within_locus_centered_mse_loss(prediction, safe_target, mask, loss_cfg)
        locus_pcc_loss, n_valid_pcc_loci = locus_correlation_loss(prediction, safe_target, mask, loss_cfg)
        total = (
            loss_cfg.beta_mse_weight * beta_mse
            + self.aux_weight * mean_aux_loss
            + loss_cfg.sample_pearson_weight * sample_pcc_loss
            + loss_cfg.locus_centered_mse_weight * centered_mse
            + loss_cfg.locus_pearson_weight * locus_pcc_loss
        )
        pieces = {
            "beta_mse": beta_mse.detach(), "beta_mse_weighted": (loss_cfg.beta_mse_weight * beta_mse).detach(),
            "mean_aux_loss": mean_aux_loss.detach(), "mean_aux_loss_weighted": (self.aux_weight * mean_aux_loss).detach(),
            "sample_pcc_loss": sample_pcc_loss.detach(),
            "sample_pcc_loss_weighted": (loss_cfg.sample_pearson_weight * sample_pcc_loss).detach(),
            "locus_pcc_loss": locus_pcc_loss.detach(),
            "locus_pcc_loss_weighted": (loss_cfg.locus_pearson_weight * locus_pcc_loss).detach(),
            "valid_locus_pcc_loci": n_valid_pcc_loci,
            "centered_mse": centered_mse.detach(),
            "centered_mse_weighted": (loss_cfg.locus_centered_mse_weight * centered_mse).detach(),
            "mean_rho_p": mean_rho_p, "valid_correlation_samples": n_valid_samples,
            "valid_centered_loci": n_valid_loci, "observed": mask.sum().detach(),
        }
        return total, pieces

    def _step(
        self,
        pool: TrainingPool,
        sample_ids,
        cpg_ids,
        rna_cpu,
        emb_cpu,
        beta_cpu,
        functional_cpu,
        finite_count,
    ):
        if finite_count == 0:
            return None

        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        compute_start = torch.cuda.Event(enable_timing=True)

        h2d_start.record()
        rna_x = rna_cpu.to(
            self.device, non_blocking=True
        ).float()
        emb = emb_cpu.to(
            self.device, non_blocking=True
        ).float()
        beta = beta_cpu.to(
            self.device, non_blocking=True
        )
        functional_kwargs = self._functional_kwargs(functional_cpu)
        h2d_end.record()
        compute_start.record()

        with self._autocast():
            out = self.train_model(
                rna_x,
                emb,
                **functional_kwargs,
            )
            loss_cfg = loss_config_for_source(
                self.recipe.loss,
                pool.name,
                self.recipe.structured_loss_sources,
            )
            total, pieces = self._mas_concat_loss(
                out,
                beta,
                cpg_ids,
                loss_cfg,
            )

        pieces = {
            **pieces,
            "total_loss": total.detach(),
        }
        return (
            total,
            pieces,
            h2d_start,
            h2d_end,
            compute_start,
        )

    def _functional_kwargs(self, functional_cpu) -> dict[str, torch.Tensor]:
        if functional_cpu is None:
            return {}
        return {
            "functional_track_indices": functional_cpu["track_indices"].to(self.device, non_blocking=True),
            "functional_offsets": functional_cpu["offsets"].to(self.device, non_blocking=True),
            "functional_dense": functional_cpu["dense"].to(self.device, non_blocking=True).float(),
        }

    def _position_kwargs(self, cpg_ids: np.ndarray) -> dict[str, torch.Tensor]:
        """CpG ordinals for the axial arm, empty for every other architecture.

        Cartesian blocks are contiguous slices of a source's CpG index array
        (CartesianSourceSchedule / SourceSchedule), so these ordinals are
        monotone in genomic coordinate within the chromosome -- a proxy for
        base-pair distance, which is all the axial attention's distance decay
        needs.
        """
        if not getattr(self.model, "requires_cpg_positions", False):
            return {}
        return {"cpg_positions": torch.from_numpy(np.asarray(cpg_ids, dtype=np.int64)).to(self.device)}

    @torch.no_grad()
    def evaluate_view(self, sample_ids, cpg_ids, *, sample_chunk=128, cpg_chunk=2048):
        self.model.eval()
        source = self._sources["array"]
        rows = source.rows_of_samples(sample_ids)
        metrics = ArrayMomentMetrics(len(sample_ids), len(cpg_ids))
        for s0 in range(0, len(sample_ids), sample_chunk):
            s1 = min(s0 + sample_chunk, len(sample_ids))
            local_s = sample_ids[s0:s1]
            rna_x = torch.from_numpy(self.rna.rows(local_s)).to(self.device)
            for c0 in range(0, len(cpg_ids), cpg_chunk):
                c1 = min(c0 + cpg_chunk, len(cpg_ids))
                local_c = cpg_ids[c0:c1]
                if self.functional_only:
                    feature_rows = self.features.index.positions_of(local_c)
                    emb_np = np.empty((len(local_c), 0), dtype=np.float32)
                    prior_np = np.asarray(self.features.prior[feature_rows], dtype=np.float32)
                else:
                    emb_np, prior_np, _ = self.features.get(local_c)
                emb = torch.from_numpy(emb_np).to(self.device)
                functional_cpu = None
                if self.functional is not None:
                    functional_cpu = {
                        key: torch.from_numpy(np.ascontiguousarray(value))
                        for key, value in self.functional.get(local_c).items()
                    }
                with self._autocast():
                    pred = self.model(
                        rna_x, emb, **self._position_kwargs(local_c),
                        **self._functional_kwargs(functional_cpu),
                    )["beta"]
                target = source.block(rows[s0:s1], local_c)
                metrics.add(s0, c0, target, pred.float().cpu().numpy(), prior_np)
        self.model.train()
        return metrics.finalize()

    def evaluate_development(self):
        if self.inner_views is None:
            raise RuntimeError("not a development run")
        return {name: self.evaluate_view(s, c) for name, (s, c) in self.inner_views.items()}

    def _save_checkpoint(self, path, optimizer, scheduler, epoch, history):
        payload = {
            "schema_version": 2, "model": "locus_cls_joint", "scope": self.scope, "mode": self.mode,
            "epoch": epoch, "epochs_planned": self.epochs, "architecture": self.architecture_label,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "model_config": asdict(self.recipe.model),
            "loss_config": asdict(self.recipe.loss), "training": asdict(self.recipe.training), "history": history,
            "locus_cls": {
                "use_mean_branch": self.use_mean_branch, "use_fusion_product": getattr(self.model, "use_fusion_product", False),
                "use_raw_product": getattr(self.model, "use_raw_product", True),
                "product_mlp": getattr(self.model, "product_mlp", False),
                "include_raw_rna": getattr(self.model, "include_raw_rna", True),
                "include_raw_cpg": getattr(self.model, "include_raw_cpg", True),
                "query_source": self.query_source,
                "fusion_init_std": self._fusion_init_std, "aux_weight": self.aux_weight,
                "residual_aux_weight": self.residual_aux_weight, "raw_lr_multiplier": self.raw_lr_multiplier,
                "trunk_hidden_dim": self.trunk_hidden_dim, "bottleneck_dim": self.bottleneck_dim,
                **({"functional_locus": {
                    "functional_atlas": str(self.functional.functional_atlas_root.resolve()),
                    "annotation_cache": str(self.functional.annotation_cache_root.resolve()),
                    **({"bigwig_cache": str(self.functional.bigwig_cache_root.resolve())} if self.functional.bigwig_cache_root is not None else {}),
                    "n_tracks": 4165, "dense_dim": self.functional.DENSE_DIM,
                    "encoder_dim": 256 if self.functional_fusion_variant else 64,
                    **({"fusion_variant": self.functional_fusion_variant} if self.functional_fusion_variant else {}),
                    "residual_policy": (
                        "random_initialized_functional_projection" if self.functional_only
                        else "zero_initialized_additive_projection"
                    ),
                    "mode": "functional_only" if self.functional_only else "additive_residual",
                }} if self.functional is not None else {}),
            },
            "wandb": None if self.wandb_run is None else {
                "project": self.wandb_run.project, "entity": self.wandb_run.entity, "run_id": self.wandb_run.id,
            },
            "rng_state": {
                "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
        }
        if not self.functional_only and self.features.regulatory_provenance is not None:
            payload["regulatory_feature_cache"] = self.features.regulatory_provenance
        tmp = Path(str(path) + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def run(self) -> dict[str, object]:
        cfg = self.recipe.training
        opt_kwargs = {"weight_decay": cfg.weight_decay}
        if cfg.fused_adamw:
            opt_kwargs["fused"] = True
        groups = list(self.model.parameters())
        opt_kwargs["lr"] = cfg.learning_rate
        try:
            optimizer = torch.optim.AdamW(groups, **opt_kwargs)
        except (TypeError, RuntimeError):
            opt_kwargs.pop("fused", None)
            optimizer = torch.optim.AdamW(groups, **opt_kwargs)
        schedules0, plan0 = self._schedules(1)
        plan_hasher = hashlib.sha256()
        for source_i, local_step in plan0:
            row_slots, cpg_slots = schedules0[source_i][local_step]
            plan_hasher.update(np.asarray([source_i, local_step], np.int64).tobytes())
            plan_hasher.update(np.asarray(row_slots, np.int64).tobytes())
            plan_hasher.update(np.asarray(cpg_slots, np.int64).tobytes())
        self.first_epoch_plan_sha256 = plan_hasher.hexdigest()
        self.trainable_parameter_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[locus-cls-joint] params={self.trainable_parameter_count} epoch1_plan_sha256={self.first_epoch_plan_sha256}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        steps_per_epoch = len(plan0)
        horizon = int(cfg.scheduler_horizon_epochs or self.epochs)
        scheduler = build_lr_scheduler(
            optimizer, name=cfg.scheduler, total_steps=max(1, horizon * steps_per_epoch),
            warmup_steps=int(round(cfg.warmup_epochs * steps_per_epoch)), min_lr_ratio=cfg.min_lr_ratio,
        )
        history = []
        latest = self.store.checkpoint("last.pt")
        best = self.store.checkpoint("best.pt")
        best_score = -np.inf
        best_epoch = 0
        # mode="final" has no held-out score per epoch (evaluating against the
        # official split mid-training would leak it into model selection,
        # defeating the point of a true final number) -- so its early-stop
        # signal is training-loss plateau instead, not val_cpg_x_val_sample.
        best_loss = np.inf
        best_loss_epoch = 0
        loss_improve_rel = 1e-3  # ignore floating noise smaller than 0.1% relative
        last_epoch = 0
        start_epoch = 1
        if latest.is_file():
            state = torch.load(latest, map_location=self.device, weights_only=False)
            if state.get("scope") != self.scope or state.get("mode") != self.mode:
                raise RuntimeError("resume checkpoint scope/mode mismatch")
            if int(state.get("epochs_planned", -1)) != self.epochs:
                raise RuntimeError("resume checkpoint epoch-budget mismatch")
            self.model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            history = list(state.get("history", []))
            start_epoch = int(state["epoch"]) + 1
            last_epoch = int(state["epoch"])
            rng = state.get("rng_state") or {}
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("torch_cpu") is not None:
                torch.set_rng_state(rng["torch_cpu"].cpu())
            if rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all([x.cpu() for x in rng["torch_cuda"]])
            if self.mode == "development" and history:
                scores = [x.get("development", {}).get("val_cpg_x_val_sample", {}).get("mas_pcc", -np.inf) for x in history]
                best_score = float(np.nanmax(scores)); best_epoch = int(history[int(np.nanargmax(scores))]["epoch"])
            finite_losses = [(float(x["loss"]), int(x["epoch"])) for x in history if np.isfinite(x.get("loss", np.nan))]
            if finite_losses:
                best_loss, best_loss_epoch = min(finite_losses)
        started_all = time.time()
        for epoch in range(start_epoch, self.epochs + 1):
            started = time.time()
            self.model.train()
            schedules, plan = self._schedules(epoch)
            loss_sum = None
            piece_sums: dict[str, torch.Tensor | float] = {}
            piece_counts: dict[str, int] = {}
            optimizer_steps = 0
            data_wait_seconds = 0.0
            cpu_prepare_seconds = 0.0
            cuda_timings = []
            observed_pairs = 0
            pair_slots = 0
            # Prepare blocks concurrently but consume futures in submission order:
            # optimizer order/numerics stay identical while HDF5 decode, NaN scans,
            # feature/RNA copies and pinning can overlap across CPU cores. The deque
            # is strictly bounded, so large WGBS embeddings cannot inflate RAM.
            items = [(source_i, *schedules[source_i][local_step]) for source_i, local_step in plan]
            item_iter = iter(items)

            def prepare(item):
                return self._prepare_step(*item)

            with ThreadPoolExecutor(
                max_workers=cfg.prefetch_workers, thread_name_prefix="block-prefetch",
            ) as executor:
                for block, wait_seconds in ordered_bounded_prefetch(
                    executor, prepare, item_iter, cfg.prefetch_depth,
                ):
                    data_wait_seconds += wait_seconds
                    (
                        source_i, sample_ids, cpg_ids, rna_cpu, emb_cpu, beta_cpu,
                        functional_cpu, finite_count, batch_pair_slots, prepare_seconds,
                    ) = block
                    cpu_prepare_seconds += prepare_seconds
                    observed_pairs += finite_count
                    pair_slots += batch_pair_slots
                    pool = self.pools[source_i]
                    result = self._step(
                        pool, sample_ids, cpg_ids, rna_cpu, emb_cpu, beta_cpu, functional_cpu, finite_count,
                    )
                    if result is None:
                        continue
                    loss, pieces, h2d_start, h2d_end, compute_start = result
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip_norm)
                    optimizer.step()
                    scheduler.step()
                    compute_end = torch.cuda.Event(enable_timing=True)
                    compute_end.record()
                    cuda_timings.append((h2d_start, h2d_end, compute_start, compute_end))
                    optimizer_steps += 1
                    loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
                    for key, value in pieces.items():
                        if isinstance(value, torch.Tensor):
                            piece_sums[key] = value if key not in piece_sums else piece_sums[key] + value
                        else:
                            piece_sums[key] = float(piece_sums.get(key, 0.0)) + float(value)
                        piece_counts[key] = piece_counts.get(key, 0) + 1
            torch.cuda.synchronize(self.device)
            training_seconds = time.time() - started
            h2d_seconds = sum(a.elapsed_time(b) for a, b, _, _ in cuda_timings) / 1000.0
            compute_seconds = sum(c.elapsed_time(d) for _, _, c, d in cuda_timings) / 1000.0
            mean_loss = float((loss_sum / optimizer_steps).cpu()) if optimizer_steps else float("nan")
            row = {
                "epoch": epoch, "seconds": training_seconds, "training_seconds": training_seconds,
                "data_wait_seconds": data_wait_seconds, "cpu_prepare_seconds": cpu_prepare_seconds,
                "h2d_seconds": h2d_seconds, "compute_seconds": compute_seconds,
                "observed_pairs": observed_pairs, "pair_slots": pair_slots,
                "pair_slots_per_second": pair_slots / training_seconds if training_seconds else float("nan"),
                "optimizer_steps": optimizer_steps, "loss": mean_loss,
                "trainable_parameters": self.trainable_parameter_count,
                "epoch1_plan_sha256": self.first_epoch_plan_sha256,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(self.device) if torch.cuda.is_available() else None,
                "lr": optimizer.param_groups[0]["lr"],
            }
            piece_means = {
                k: float((v / piece_counts[k]).cpu()) if isinstance(v, torch.Tensor) else float(v / piece_counts[k])
                for k, v in piece_sums.items() if piece_counts[k]
            }
            # Persisted to history.json (not just W&B) so per-epoch loss-term
            # diagnostics (e.g. the H0/H1/H2 sample-wise-Pearson experiment's
            # beta_mse/mean_aux_loss/sample_pcc_loss raw+weighted, mean_rho_p)
            # survive an offline/no-wandb run.
            row["loss_components"] = piece_means
            validation_started = time.time()
            validated = self.mode == "development" and (epoch % cfg.validation_every == 0 or epoch == self.epochs)
            if validated:
                dev = self.evaluate_development()
                row["development"] = dev
                score = float(dev["val_cpg_x_val_sample"]["mas_pcc"])
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_epoch = epoch
                    self._save_checkpoint(best, optimizer, scheduler, epoch, [*history, row])
            else:
                loss_val = row["loss"]
                if np.isfinite(loss_val) and loss_val < best_loss * (1 - loss_improve_rel):
                    best_loss = loss_val
                    best_loss_epoch = epoch
            row["validation_seconds"] = time.time() - validation_started if validated else 0.0
            diagnostics = self.model.diagnostics() if hasattr(self.model, "diagnostics") else {}
            if diagnostics:
                row["architecture_diagnostics"] = diagnostics
            history.append(row)
            last_epoch = epoch
            checkpoint_started = time.time()
            if epoch % cfg.checkpoint_every == 0 or epoch == self.epochs:
                self._save_checkpoint(latest, optimizer, scheduler, epoch, history)
            row["checkpoint_seconds"] = time.time() - checkpoint_started
            write_json(self.store.training_file("history.json"), history)
            print(f"[locus-cls-joint:{self.scope}:{self.mode}:{epoch}/{self.epochs}] loss={row['loss']:.6g}", flush=True)
            if self.wandb_run is not None:
                wandb_log = {
                    "train/loss": row["loss"], "train/lr": row["lr"], "train/seconds": row["seconds"],
                    "train/optimizer_steps": row["optimizer_steps"],
                    "system/trainable_parameters": row["trainable_parameters"],
                    "system/peak_cuda_memory_bytes": row["peak_cuda_memory_bytes"],
                    "train/pair_slots_per_second": row["pair_slots_per_second"],
                    **{f"train/{k}": v for k, v in piece_means.items()},
                    **{f"diagnostics/{k}": v for k, v in diagnostics.items()},
                }
                if validated:
                    for view_name, view_metrics in row["development"].items():
                        wandb_log.update({f"val/{view_name}/{k}": v for k, v in view_metrics.items()})
                self.wandb_run.log(wandb_log, step=epoch)
            if self.early_stop_patience is not None:
                if self.mode == "development":
                    if not validated:
                        continue
                    stalled = epoch - best_epoch >= self.early_stop_patience
                    ref_msg = f"no val_cpg_x_val_sample.mas_pcc improvement for {self.early_stop_patience} epochs (best={best_score:.4f} @ epoch {best_epoch})"
                else:
                    stalled = epoch - best_loss_epoch >= self.early_stop_patience
                    ref_msg = f"no training-loss improvement (>{loss_improve_rel:.1%} rel.) for {self.early_stop_patience} epochs (best_loss={best_loss:.5g} @ epoch {best_loss_epoch})"
                if stalled:
                    print(f"[locus-cls-joint:{self.scope}:{self.mode}] early stop at epoch {epoch}: {ref_msg}", flush=True)
                    break
        if last_epoch and last_epoch % cfg.checkpoint_every != 0:
            # Early stopping may occur between periodic checkpoints; always
            # leave a resumable snapshot at the actual terminal epoch.
            self._save_checkpoint(latest, optimizer, scheduler, last_epoch, history)
        if self.mode == "final":
            self._save_checkpoint(best, optimizer, scheduler, last_epoch, history)
            best_epoch = last_epoch
        summary = {
            "scope": self.scope, "mode": self.mode, "best_epoch": best_epoch,
            "best_inner_double_ood_mas_pcc": None if self.mode == "final" else best_score,
            "epochs_planned": self.epochs, "epochs_run": last_epoch,
            "elapsed_seconds": time.time() - started_all, "run_dir": str(self.store.path),
            "trainable_parameters": self.trainable_parameter_count,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(self.device) if torch.cuda.is_available() else None,
            "epoch1_plan_sha256": self.first_epoch_plan_sha256,
        }
        write_json(self.store.training_file("summary.json"), summary)
        if self.wandb_run is not None:
            # finish() here so the run reads as complete rather than "crashed" for
            # anyone inspecting it before evaluation runs. evaluate_official_split
            # (a separate CLI call on this engine, see its docstring) resumes this
            # exact run id -- via the (project, entity, run_id) this class persists
            # into the checkpoint above -- and appends the official-split `val/*`
            # numbers to it, so one wandb run still carries the whole story.
            self.wandb_run.summary.update({f"train/{k}": v for k, v in summary.items() if k != "run_dir"})
            self.wandb_run.finish()
        return summary


def evaluate_official_split(
    *,
    checkpoint: str | Path,
    canonical_root: str | Path,
    recipe_path: str | Path,
    feature_cache: str | Path,
    rna_cache: str | Path,
    registry: str | Path,
    matched_chr1_root: str | Path | None,
    cpg_targets_dir: str | Path,
    output: str | Path,
    scope: str = "chr1",
    sample_chunk: int = 128,
    cpg_chunk: int = 2048,
    functional_atlas: str | Path | None = None,
    annotation_cache: str | Path | None = None,
    functional_only: bool = False,
) -> dict:
    """Evaluate a checkpoint on all three TRUE official MethylProphet views.

    The headline ``metrics`` field remains the double-OOD
    ``val_cpg_x_val_sample`` view for backwards compatibility.  The complete
    result is also available under ``views`` and is logged to W&B with one
    namespace per view.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("regulatory_feature_cache") is not None:
        supplied_cache = LocusFeatureCache(feature_cache)
        if supplied_cache.regulatory_provenance != ckpt["regulatory_feature_cache"]:
            raise ValueError("evaluation regulatory feature cache differs from the trained checkpoint")
    lc = ckpt.get("locus_cls") or {}
    checkpoint_functional = lc.get("functional_locus")
    if checkpoint_functional is not None and (functional_atlas is None or annotation_cache is None):
        raise ValueError("functional checkpoint evaluation requires --functional-atlas and --annotation-cache")
    if checkpoint_functional is None and (functional_atlas is not None or annotation_cache is not None):
        raise ValueError("cannot enable functional conditioning when evaluating a non-functional checkpoint")
    if checkpoint_functional is not None:
        checkpoint_functional_only = checkpoint_functional.get("mode") == "functional_only"
        if bool(functional_only) != checkpoint_functional_only:
            raise ValueError("--functional-only does not match the checkpoint metadata")
        supplied = {
            "functional_atlas": str(Path(functional_atlas).resolve()),
            "annotation_cache": str(Path(annotation_cache).resolve()),
        }
        for key, value in supplied.items():
            if value != checkpoint_functional.get(key):
                raise ValueError(f"evaluation {key} does not match the checkpoint metadata")
    # LocusCLSJointTrainer always wants its own run-store scratch dir (distinct
    # from `output`, which here is the single evaluation-summary JSON file the
    # scripts/evaluate.py CLI convention expects). track=False: this is an
    # evaluation-only re-instantiation of the model, not a second training run --
    # it must not open its own wandb run. The training run this checkpoint came
    # from is resumed explicitly below instead, by (project, entity, run_id)
    # persisted into the checkpoint by LocusCLSJointTrainer._save_checkpoint.
    scratch_root = Path(output).parent / ".eval_runs"
    trainer = LocusCLSJointTrainer(
        canonical_root=canonical_root, scope=scope, recipe_path=recipe_path,
        feature_cache=feature_cache, rna_cache=rna_cache, registry=registry,
        cpg_targets_dir=cpg_targets_dir, matched_chr1_root=matched_chr1_root,
        output_root=scratch_root, mode="final", run_id=f"eval-{Path(checkpoint).stem}-{os.getpid()}",
        use_mean_branch=lc.get("use_mean_branch", True), use_fusion_product=lc.get("use_fusion_product", False),
        use_raw_product=lc.get("use_raw_product", True), product_mlp=lc.get("product_mlp", False),
        include_raw_rna=lc.get("include_raw_rna", True), include_raw_cpg=lc.get("include_raw_cpg", True),
        query_source=lc.get("query_source", "ntv3"),
        trunk_hidden_dim=lc.get("trunk_hidden_dim", 256), bottleneck_dim=lc.get("bottleneck_dim", 64),
        fusion_init_std=lc.get("fusion_init_std", 0.01), aux_weight=lc.get("aux_weight", 0.15),
        residual_aux_weight=lc.get("residual_aux_weight", 0.0), raw_lr_multiplier=lc.get("raw_lr_multiplier", 1.0),
        functional_atlas=functional_atlas, annotation_cache=annotation_cache,
        functional_only=functional_only,
        track=False,
    )
    try:
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.model.eval()
        p = trainer.protocol
        views = {
            "train_cpg_x_val_sample": (p.array_val_sample_idx, p.array_train_cpg_idx),
            "val_cpg_x_train_sample": (p.array_train_sample_idx, p.array_val_cpg_idx),
            "val_cpg_x_val_sample": (p.array_val_sample_idx, p.array_val_cpg_idx),
        }
        with torch.no_grad():
            view_results = {
                name: trainer.evaluate_view(sample_ids, cpg_ids, sample_chunk=sample_chunk, cpg_chunk=cpg_chunk)
                for name, (sample_ids, cpg_ids) in views.items()
            }
        result = view_results["val_cpg_x_val_sample"]
        summary = {
            "model": "feature_fusion_locus_cls", "checkpoint": str(checkpoint), "checkpoint_epoch": ckpt.get("epoch"),
            "eval_scope": scope, "view": "val_cpg_x_val_sample", "metrics": result,
            "views": view_results,
        }
        write_json(Path(output), summary)
        wandb_info = ckpt.get("wandb")
        if wandb_info and wandb_info.get("run_id"):
            import wandb
            run = wandb.init(
                project=wandb_info.get("project"), entity=wandb_info.get("entity"),
                id=wandb_info["run_id"], resume="must",
            )
            # Keep each evaluation split as a separate top-level W&B namespace
            # instead of nesting all three under a shared ``val/`` prefix.
            headline = {f"val_cpg_x_val_sample/{k}": v for k, v in result.items()}
            per_view = {
                f"{view_name}/{key}": value
                for view_name, metrics in view_results.items()
                for key, value in metrics.items()
            }
            run.log({**headline, **per_view})
            run.summary.update({**headline, **per_view})
            run.summary.update({"evaluation_view": summary["view"], "evaluation_checkpoint_epoch": ckpt.get("epoch")})
            run.finish()
        return summary
    finally:
        trainer.close()
