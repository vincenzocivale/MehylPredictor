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
from dataclasses import asdict
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..losses import beta_nll_term, locus_correlation_loss, masked_mean
from ..models import (
    FeatureFusionArchitectureVariantModel,
    FeatureFusionLocusCLSModel,
    feature_fusion_variant_label,
    is_architecture_variant,
)
from ..optim import build_lr_scheduler
from ..run_store import RunStore, write_json
from ..scopes import scope_protocol
from ..storage import LocusFeatureCache, RNACache, SortedIndex
from ..tcga_canonical import TCGACanonicalBundle
from .config import load_rna_recipe
from .matched_chr1_data import load_matched_chr1_protocol_and_sources
from .metrics import ArrayMomentMetrics
from .schedule import SourceSchedule, interleave
from .splits import blocked_cpg_split, stratified_sample_split
from .trainer import TrainingPool, loss_config_for_source


def _direct_beta_loss(outputs: dict, target_beta: torch.Tensor, loss_cfg) -> tuple[torch.Tensor, dict]:
    """No-prior beta loss -- FeatureFusionLocusCLSModel predicts beta directly
    (no mu/sigma anchor to derive residual_loss's other terms from), so this
    is just the beta_mse/locus_pearson/beta_nll subset of that objective (same
    subset DirectPredictionModel effectively reduces to). beta_nll_term is a
    no-op whenever loss_cfg.beta_nll_weight is zero or the model produced no
    concentration head, so it is always safe to add unconditionally --
    architecture_novelty_2026_09's beta_likelihood_head arm is the only one
    that ever makes it nonzero."""
    mask = torch.isfinite(target_beta)
    prediction = outputs["beta"]
    safe_target = torch.where(mask, target_beta, torch.zeros_like(target_beta))
    beta_mse = masked_mean((prediction - safe_target) ** 2, mask)
    if loss_cfg.locus_pearson_weight != 0.0:
        pearson_loss, valid_loci = locus_correlation_loss(prediction, safe_target, mask, loss_cfg)
    else:
        pearson_loss = prediction.sum() * 0.0
        valid_loci = 0
    beta_nll = beta_nll_term(outputs, safe_target, mask, loss_cfg)
    total = (
        loss_cfg.beta_mse_weight * beta_mse
        + loss_cfg.locus_pearson_weight * pearson_loss
        + loss_cfg.beta_nll_weight * beta_nll
    )
    return total, {
        "beta_mse": float(beta_mse.detach().cpu()),
        "locus_pearson_loss": float(pearson_loss.detach().cpu()),
        "beta_nll": float(beta_nll.detach().cpu()),
        "valid_correlation_loci": valid_loci,
        "observed": int(mask.sum().detach().cpu()),
    }


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
        fusion_init_std: float = 0.01,
        aux_weight: float = 0.15,
        residual_aux_weight: float = 0.15,
        raw_lr_multiplier: float = 1.0,
        mode: str = "final",
        early_stop_patience: int | None = None,
        run_id: str | None = None,
        overrides: dict | None = None,
        track: bool = True,
    ):
        # >1.0 gives the raw/RNA branch (raw_branch, fusion, residual_head --
        # everything that only ever gets gradient through the fusion layer,
        # unlike the trunk which also gets a direct auxiliary-loss gradient
        # into h_mean) a higher effective LR to compensate for that branch
        # imbalance -- measured empirically (chr1 pair_complete, 2026-09-02).
        self.raw_lr_multiplier = float(raw_lr_multiplier)
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

        self.matched_chr1_root = Path(matched_chr1_root) if matched_chr1_root else None
        if self.matched_chr1_root is not None:
            if scope != "chr1":
                raise ValueError("matched_chr1_root is only valid for scope='chr1'")
            self.bundle = None
            self.protocol, self._sources = load_matched_chr1_protocol_and_sources(
                self.matched_chr1_root, self.root,
            )
        else:
            self.bundle = TCGACanonicalBundle.from_root(self.root)
            self.protocol = scope_protocol(scope, self.bundle, canonical_root=self.root)
            self._sources = self.bundle.sources
        self.features = LocusFeatureCache(feature_cache)
        self.rna = RNACache(rna_cache)
        required = np.unique(np.concatenate([
            self.protocol.array_train_cpg_idx, self.protocol.array_val_cpg_idx,
            *self.protocol.auxiliary_cpg_idx.values(),
        ]))
        self.features.index.positions_of(required)

        self.aux_weight = float(aux_weight)
        self.residual_aux_weight = float(residual_aux_weight)
        cpg_targets_dir = Path(cpg_targets_dir)
        self.cpg_target_ids = np.load(cpg_targets_dir / "cpg_idx.npy")
        self.cpg_target_mu = np.load(cpg_targets_dir / "target_mu.npy")
        self.cpg_target_index = SortedIndex(self.cpg_target_ids, "cpg_statistics targets")

        self.use_mean_branch = bool(use_mean_branch)
        self._fusion_init_std = float(fusion_init_std)
        # An architecture-novelty recipe (a non-default encoder kind, trunk or
        # axial block, or a beta-likelihood head) routes to
        # FeatureFusionArchitectureVariantModel instead of the reference class --
        # see architecture_novelty_2026_09 / docs/RNA_METHYLATION.md. At every
        # default this class is numerically identical to
        # FeatureFusionLocusCLSModel (tests/test_architecture_variants.py), so
        # this dispatch changes nothing for the reference recipe itself.
        self.architecture_variant = is_architecture_variant(self.recipe.model)
        model_cls = FeatureFusionArchitectureVariantModel if self.architecture_variant else FeatureFusionLocusCLSModel
        self.architecture_label = (
            "feature_fusion_locus_cls_arch_" + feature_fusion_variant_label(self.recipe.model)
            if self.architecture_variant
            else "feature_fusion_locus_cls"
        )
        self.model = model_cls(
            25_017, 1536, self.recipe.model,
            trunk_hidden_dim=trunk_hidden_dim, bottleneck_dim=bottleneck_dim, trunk_dropout=trunk_dropout,
            use_mean_branch=use_mean_branch, use_fusion_product=use_fusion_product, fusion_init_std=fusion_init_std,
        ).to(self.device)

        self.inner_views = None
        self.pools = self._build_pools()
        self.store = RunStore.create(
            output_root, model="locus_cls_joint", train_scope=scope, seed=self.seed,
            learning_rate=cfg.learning_rate, scheduler=cfg.scheduler, epochs=self.epochs, run_id=run_id,
        )
        self.store.save_resolved_config({
            **self.recipe.raw,
            "locus_cls": {
                "use_mean_branch": use_mean_branch, "use_fusion_product": use_fusion_product,
                "fusion_init_std": fusion_init_std, "aux_weight": self.aux_weight,
                "residual_aux_weight": self.residual_aux_weight, "raw_lr_multiplier": self.raw_lr_multiplier,
                "trunk_hidden_dim": trunk_hidden_dim, "bottleneck_dim": bottleneck_dim,
            },
        })

        # wandb: one run per training call, resumed later by evaluate_official_split
        # so the official-split numbers land as `val/*` in the same run instead of a
        # disconnected second entry -- see that function and _save_checkpoint below,
        # which persists (project, entity, run_id) into the checkpoint for exactly
        # that resume. `track=False` is used by evaluate_official_split's own scratch
        # trainer instance so evaluation-only re-instantiation never opens a second,
        # spurious training run.
        self.wandb_run = None
        tracking = self.recipe.tracking
        if track and tracking.backend == "wandb" and tracking.mode != "disabled":
            import wandb
            kwargs = {
                "project": tracking.project, "entity": tracking.entity,
                "group": tracking.group or self.architecture_label,
                "name": tracking.name or self.store.run_id, "job_type": tracking.job_type,
                "mode": tracking.mode, "dir": str(self.store.path),
                "tags": [*tracking.tags, f"scope-{scope}", f"mode-{mode}", self.architecture_label],
                "config": {
                    "model": asdict(self.recipe.model), "loss": asdict(self.recipe.loss),
                    "training": asdict(self.recipe.training), "schedule_policy": self.recipe.schedule_policy,
                    "locus_cls": {
                        "use_mean_branch": use_mean_branch, "use_fusion_product": use_fusion_product,
                        "fusion_init_std": fusion_init_std, "aux_weight": self.aux_weight,
                        "residual_aux_weight": self.residual_aux_weight, "raw_lr_multiplier": self.raw_lr_multiplier,
                    },
                    "scope": scope, "mode": mode, "seed": self.seed, "architecture": self.architecture_label,
                },
            }
            self.wandb_run = wandb.init(**{k: v for k, v in kwargs.items() if v is not None})
        self.store.save_metadata({
            "architecture": self.architecture_label, "seed": self.seed,
            "wandb": None if self.wandb_run is None else {
                "project": self.wandb_run.project, "entity": self.wandb_run.entity, "run_id": self.wandb_run.id,
            },
        })

    def close(self) -> None:
        if self.bundle is not None:
            self.bundle.close()
        else:
            for source in self._sources.values():
                source.close()

    def _build_pools(self) -> list[TrainingPool]:
        p = self.protocol
        if self.mode == "development":
            frac = float(self.recipe.raw.get("development", {}).get("fraction", 0.1))
            block_bp = int(self.recipe.raw.get("development", {}).get("block_bp", 5_000_000))
            train_s, val_s = stratified_sample_split(
                canonical_root=self.root, sample_ids=p.array_train_sample_idx, val_fraction=frac, seed=self.seed,
            )
            train_c, val_c = blocked_cpg_split(
                registry=self.registry, cpg_ids=p.array_train_cpg_idx, val_fraction=frac, seed=self.seed, block_bp=block_bp,
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
                epoch, self.seed + 1009 * i, self.recipe.schedule_policy,
            ))
        return schedules, interleave(schedules, seed=self.seed, epoch=epoch)

    def _read_block(self, pool: TrainingPool, row_slots, cpg_slots):
        rows = pool.row_positions[row_slots]
        cpg = pool.cpg_idx[cpg_slots]
        source = self._sources[pool.name]
        beta = source.block(rows, cpg)
        return pool.sample_idx[row_slots], cpg, beta

    def _mean_aux_loss(self, cpg_ids: np.ndarray, mu_logit: torch.Tensor) -> torch.Tensor:
        present = self.cpg_target_index.contains(cpg_ids)
        if not present.any():
            return mu_logit.sum() * 0.0
        rows = self.cpg_target_index.positions_of(cpg_ids[present])
        eps = 1e-4
        target_mu_logit = torch.logit(
            torch.from_numpy(self.cpg_target_mu[rows].astype(np.float32)).to(mu_logit.device).clamp(eps, 1 - eps)
        )
        present_t = torch.from_numpy(present).to(mu_logit.device)
        return F.mse_loss(mu_logit[present_t], target_mu_logit)

    def _residual_aux_loss(self, cpg_ids: np.ndarray, beta: torch.Tensor, residual_logit: torch.Tensor) -> torch.Tensor:
        """Direct sample-locus supervision for h_raw (via residual_head),
        symmetric to _mean_aux_loss's supervision of h_mean -- target is
        logit(target_beta) - logit(target_mu), reusing the same per-locus
        target_mu already loaded for the mean probe (no new data)."""
        present = self.cpg_target_index.contains(cpg_ids)
        if not present.any():
            return residual_logit.sum() * 0.0
        eps = 1e-4
        target_mu_np = np.full(len(cpg_ids), np.nan, dtype=np.float32)
        rows = self.cpg_target_index.positions_of(cpg_ids[present])
        target_mu_np[present] = self.cpg_target_mu[rows].astype(np.float32)
        target_mu = torch.from_numpy(target_mu_np).to(beta.device)
        valid_locus = torch.isfinite(target_mu)
        mask = torch.isfinite(beta) & valid_locus.unsqueeze(0)
        if not bool(mask.any()):
            return residual_logit.sum() * 0.0
        safe_beta = torch.where(mask, beta, torch.full_like(beta, 0.5)).clamp(eps, 1 - eps)
        safe_mu = torch.where(valid_locus, target_mu, torch.full_like(target_mu, 0.5)).clamp(eps, 1 - eps)
        target_residual = torch.logit(safe_beta) - torch.logit(safe_mu).unsqueeze(0)
        # Huber, not MSE: WGBS's per-patient beta is often near 0/1, where
        # logit(beta)-logit(mu) can be huge -- a plain MSE on this target lets
        # rare extreme cells dominate the whole loss (measured: ~20 vs ~0.9 on
        # comparable Array blocks). Matches the canonical model's own
        # residual_huber choice for exactly this reason (losses.py).
        return masked_mean(
            F.huber_loss(residual_logit, target_residual, reduction="none", delta=1.0), mask,
        )

    def _step(self, pool: TrainingPool, sample_ids, cpg_ids, beta_np):
        finite = np.isfinite(beta_np)
        if not finite.any():
            return None
        rna_x = torch.from_numpy(self.rna.rows(sample_ids, dtype=np.float16)).to(self.device).float()
        emb_np, _, _ = self.features.get(cpg_ids, embedding_dtype=np.float16)
        emb = torch.from_numpy(emb_np).to(self.device).float()
        beta = torch.from_numpy(beta_np).to(self.device)
        position_kwargs = self._position_kwargs(cpg_ids)
        with self._autocast():
            out = self.model(rna_x, emb, **position_kwargs)
            loss_cfg = loss_config_for_source(self.recipe.loss, pool.name, self.recipe.structured_loss_sources)
            main_loss, pieces = _direct_beta_loss(out, beta, loss_cfg)
            main_loss = main_loss * (float(finite.sum()) / max(float(finite.size), 1.0))
            total = main_loss
            aux_loss = None
            if self.use_mean_branch and self.aux_weight != 0.0:
                aux_loss = self._mean_aux_loss(cpg_ids, out["mu_logit"])
                total = total + self.aux_weight * aux_loss
            residual_aux_loss = None
            if self.residual_aux_weight != 0.0:
                residual_aux_loss = self._residual_aux_loss(cpg_ids, beta, out["residual_logit"])
                total = total + self.residual_aux_weight * residual_aux_loss
        pieces = {**pieces, "main_loss": float(main_loss.detach().cpu()), "total_loss": float(total.detach().cpu())}
        if aux_loss is not None:
            pieces["aux_loss"] = float(aux_loss.detach().cpu())
        if residual_aux_loss is not None:
            pieces["residual_aux_loss"] = float(residual_aux_loss.detach().cpu())
        return total, pieces

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
                emb_np, prior_np, _ = self.features.get(local_c)
                emb = torch.from_numpy(emb_np).to(self.device)
                with self._autocast():
                    pred = self.model(rna_x, emb, **self._position_kwargs(local_c))["beta"]
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
                "use_mean_branch": self.use_mean_branch, "use_fusion_product": self.model.use_fusion_product,
                "fusion_init_std": self._fusion_init_std, "aux_weight": self.aux_weight,
                "residual_aux_weight": self.residual_aux_weight, "raw_lr_multiplier": self.raw_lr_multiplier,
            },
            "wandb": None if self.wandb_run is None else {
                "project": self.wandb_run.project, "entity": self.wandb_run.entity, "run_id": self.wandb_run.id,
            },
        }
        tmp = Path(str(path) + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def run(self) -> dict[str, object]:
        cfg = self.recipe.training
        if self.raw_lr_multiplier != 1.0:
            raw_params = [
                *self.model.raw_branch.parameters(), *self.model.fusion.parameters(),
                *self.model.rna_product.parameters(), *self.model.locus_product.parameters(),
                *self.model.residual_head.parameters(),
            ]
            raw_param_ids = {id(p) for p in raw_params}
            other_params = [p for p in self.model.parameters() if id(p) not in raw_param_ids]
            optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "lr": cfg.learning_rate},
                    {"params": raw_params, "lr": cfg.learning_rate * self.raw_lr_multiplier},
                ],
                weight_decay=cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        schedules0, plan0 = self._schedules(1)
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
        started_all = time.time()
        for epoch in range(1, self.epochs + 1):
            started = time.time()
            self.model.train()
            schedules, plan = self._schedules(epoch)
            losses = []
            piece_sums: dict[str, list[float]] = {}
            optimizer_steps = 0
            for source_i, local_step in plan:
                pool = self.pools[source_i]
                row_slots, cpg_slots = schedules[source_i][local_step]
                sample_ids, cpg_ids, beta_np = self._read_block(pool, row_slots, cpg_slots)
                result = self._step(pool, sample_ids, cpg_ids, beta_np)
                if result is None:
                    continue
                loss, pieces = result
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
                for key, value in pieces.items():
                    piece_sums.setdefault(key, []).append(value)
            row = {
                "epoch": epoch, "seconds": time.time() - started, "optimizer_steps": optimizer_steps,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "lr": optimizer.param_groups[0]["lr"],
            }
            piece_means = {k: float(np.mean(v)) for k, v in piece_sums.items() if v}
            if self.mode == "development":
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
            diagnostics = self.model.diagnostics() if hasattr(self.model, "diagnostics") else {}
            if diagnostics:
                row["architecture_diagnostics"] = diagnostics
            history.append(row)
            last_epoch = epoch
            self._save_checkpoint(latest, optimizer, scheduler, epoch, history)
            write_json(self.store.training_file("history.json"), history)
            print(f"[locus-cls-joint:{self.scope}:{self.mode}:{epoch}/{self.epochs}] loss={row['loss']:.6g}", flush=True)
            if self.wandb_run is not None:
                wandb_log = {
                    "train/loss": row["loss"], "train/lr": row["lr"], "train/seconds": row["seconds"],
                    "train/optimizer_steps": row["optimizer_steps"],
                    **{f"train/{k}": v for k, v in piece_means.items()},
                    **{f"diagnostics/{k}": v for k, v in diagnostics.items()},
                }
                if self.mode == "development":
                    for view_name, view_metrics in row["development"].items():
                        wandb_log.update({f"val/{view_name}/{k}": v for k, v in view_metrics.items()})
                self.wandb_run.log(wandb_log, step=epoch)
            if self.early_stop_patience is not None:
                if self.mode == "development":
                    stalled = epoch - best_epoch >= self.early_stop_patience
                    ref_msg = f"no val_cpg_x_val_sample.mas_pcc improvement for {self.early_stop_patience} epochs (best={best_score:.4f} @ epoch {best_epoch})"
                else:
                    stalled = epoch - best_loss_epoch >= self.early_stop_patience
                    ref_msg = f"no training-loss improvement (>{loss_improve_rel:.1%} rel.) for {self.early_stop_patience} epochs (best_loss={best_loss:.5g} @ epoch {best_loss_epoch})"
                if stalled:
                    print(f"[locus-cls-joint:{self.scope}:{self.mode}] early stop at epoch {epoch}: {ref_msg}", flush=True)
                    break
        if self.mode == "final":
            self._save_checkpoint(best, optimizer, scheduler, last_epoch, history)
            best_epoch = last_epoch
        summary = {
            "scope": self.scope, "mode": self.mode, "best_epoch": best_epoch,
            "best_inner_double_ood_mas_pcc": None if self.mode == "final" else best_score,
            "epochs_planned": self.epochs, "epochs_run": last_epoch,
            "elapsed_seconds": time.time() - started_all, "run_dir": str(self.store.path),
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
    matched_chr1_root: str | Path,
    cpg_targets_dir: str | Path,
    output: str | Path,
    sample_chunk: int = 128,
    cpg_chunk: int = 2048,
) -> dict:
    """Evaluate a FeatureFusionLocusCLSModel checkpoint on the TRUE official
    MethylProphet split (protocol.array_val_sample_idx x array_val_cpg_idx) --
    the number this is directly comparable to the canonical model's own
    val_cpg_x_val_sample_mas_pcc (see docs/PAPER_EXPERIMENTS.md's methodology
    note on mode=development's inner proxy split vs. this true split)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lc = ckpt.get("locus_cls") or {}
    # LocusCLSJointTrainer always wants its own run-store scratch dir (distinct
    # from `output`, which here is the single evaluation-summary JSON file the
    # scripts/evaluate.py CLI convention expects). track=False: this is an
    # evaluation-only re-instantiation of the model, not a second training run --
    # it must not open its own wandb run. The training run this checkpoint came
    # from is resumed explicitly below instead, by (project, entity, run_id)
    # persisted into the checkpoint by LocusCLSJointTrainer._save_checkpoint.
    scratch_root = Path(output).parent / ".eval_runs"
    trainer = LocusCLSJointTrainer(
        canonical_root=canonical_root, scope="chr1", recipe_path=recipe_path,
        feature_cache=feature_cache, rna_cache=rna_cache, registry=registry,
        cpg_targets_dir=cpg_targets_dir, matched_chr1_root=matched_chr1_root,
        output_root=scratch_root, mode="final", run_id=f"eval-{Path(checkpoint).stem}-{os.getpid()}",
        use_mean_branch=lc.get("use_mean_branch", True), use_fusion_product=lc.get("use_fusion_product", False),
        fusion_init_std=lc.get("fusion_init_std", 0.01), aux_weight=lc.get("aux_weight", 0.15),
        residual_aux_weight=lc.get("residual_aux_weight", 0.0), raw_lr_multiplier=lc.get("raw_lr_multiplier", 1.0),
        track=False,
    )
    try:
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.model.eval()
        p = trainer.protocol
        with torch.no_grad():
            result = trainer.evaluate_view(p.array_val_sample_idx, p.array_val_cpg_idx, sample_chunk=sample_chunk, cpg_chunk=cpg_chunk)
        summary = {
            "model": "feature_fusion_locus_cls", "checkpoint": str(checkpoint), "checkpoint_epoch": ckpt.get("epoch"),
            "eval_scope": "chr1", "view": "official_val_cpg_x_val_sample", "metrics": result,
        }
        write_json(Path(output), summary)
        wandb_info = ckpt.get("wandb")
        if wandb_info and wandb_info.get("run_id"):
            import wandb
            run = wandb.init(
                project=wandb_info.get("project"), entity=wandb_info.get("entity"),
                id=wandb_info["run_id"], resume="must",
            )
            run.log({f"val/official_{k}": v for k, v in result.items()})
            run.summary.update({f"val/official_{k}": v for k, v in result.items()})
            run.summary.update({"val/official_view": summary["view"], "val/official_checkpoint_epoch": ckpt.get("epoch")})
            run.finish()
        return summary
    finally:
        trainer.close()
