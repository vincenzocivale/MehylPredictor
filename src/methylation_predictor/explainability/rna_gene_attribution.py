"""Sample-level gene-importance summaries built on top of ``integrated_gradients_rna``.

Loads a frozen ``rna_methylation`` checkpoint (``FeatureFusionLocusCLSModel``/
``FeatureFusionArchitectureVariantModel``, trained by
``LocusCLSJointTrainer``) plus the same ``LocusFeatureCache``/``RNACache``
pair the trainer/evaluator use, runs Integrated Gradients for one sample
against a set of CpG loci, and aggregates the per-locus gene attributions
into one ranked table: "which genes drive this patient's RNA-conditioned
methylation predictions".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import AxialConfig, EncoderConfig, InteractionConfig, ModelConfig, TrunkConfig
from ..models import (
    FeatureFusionArchitectureVariantModel,
    FeatureFusionLocusCLSModel,
    is_architecture_variant,
)
from ..storage import LocusFeatureCache, RNACache
from ..tcga_canonical import TCGACanonicalBundle
from .integrated_gradients import GeneAttribution, integrated_gradients_rna


def _model_config(raw: dict) -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(**raw.get("encoder", {})),
        interaction=InteractionConfig(**raw.get("interaction", {})),
        trunk=TrunkConfig(**raw.get("trunk", {})),
        axial=AxialConfig(**raw.get("axial", {})),
        beta_likelihood_head=raw.get("beta_likelihood_head", False),
    )


def _build_model_from_checkpoint(state: dict, input_dim: int) -> torch.nn.Module:
    """Reconstruct the reference shared-backbone model from a checkpoint saved
    by ``LocusCLSJointTrainer._save_checkpoint`` -- mirrors the dispatch/kwargs
    ``LocusCLSJointTrainer.__init__`` and ``evaluate_official_split`` use, but
    without needing the full training data pipeline (this is an inference-only
    diagnostic tool, see docs/EXPLAINABILITY.md)."""
    model_cfg = _model_config(state.get("model_config", {}))
    lc = state.get("locus_cls") or {}
    variant = is_architecture_variant(model_cfg)
    model_cls = FeatureFusionArchitectureVariantModel if variant else FeatureFusionLocusCLSModel
    kwargs = dict(
        trunk_hidden_dim=lc.get("trunk_hidden_dim", 256), bottleneck_dim=lc.get("bottleneck_dim", 64),
        trunk_dropout=lc.get("trunk_dropout", 0.1), use_mean_branch=lc.get("use_mean_branch", True),
        use_fusion_product=lc.get("use_fusion_product", False), fusion_init_std=lc.get("fusion_init_std", 0.01),
    )
    if variant:
        kwargs.update(
            use_raw_product=lc.get("use_raw_product", True), product_mlp=lc.get("product_mlp", False),
            include_raw_rna=lc.get("include_raw_rna", True), include_raw_cpg=lc.get("include_raw_cpg", True),
        )
    return model_cls(input_dim, 1536, model_cfg, **kwargs)


class SampleGeneExplainer:
    """Loads once, explains many (sample, loci) queries against the same checkpoint."""

    def __init__(
        self,
        *,
        canonical_root: str | Path,
        checkpoint: str | Path,
        feature_cache: str | Path,
        rna_cache: str | Path,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = Path(checkpoint)
        self.bundle = TCGACanonicalBundle.from_root(canonical_root)
        self.gene_ids = np.asarray(self.bundle.rna.gene_ids)
        self.features = LocusFeatureCache(feature_cache)
        self.rna = RNACache(rna_cache)

        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self.train_scope = str(state.get("scope", "unknown"))
        if len(self.gene_ids) != self.rna.values.shape[1]:
            # Also correctly rejects a "frozen_embedding" checkpoint (e.g.
            # BulkRNABert): its --rna-cache holds a precomputed embedding, not
            # gene expression, so per-gene attribution is not a meaningful
            # question for it.
            raise ValueError(
                f"RNA cache has {self.rna.values.shape[1]} genes but the canonical bundle's "
                f"gene_ids has {len(self.gene_ids)} -- cache/bundle mismatch, or a "
                "frozen_embedding checkpoint (not supported: its input is not gene expression)"
            )
        self.model = _build_model_from_checkpoint(state, self.rna.values.shape[1]).to(self.device)
        self.model.load_state_dict(state["model_state"], strict=True)
        self.model.eval()

    def close(self) -> None:
        self.bundle.close()

    def explain_sample(
        self,
        sample_idx: int,
        cpg_idx,
        *,
        steps: int = 50,
        auto_top_loci: int | None = None,
        n_baselines: int = 8,
        baseline_sample_idx=None,
        seed: int = 0,
    ) -> tuple[GeneAttribution, np.ndarray]:
        """Run Expected Gradients for one sample against ``cpg_idx``.

        ``auto_top_loci``: if set, ``cpg_idx`` is treated as a candidate pool --
        one cheap no-grad forward pass ranks it by ``|residual_logit|`` (how
        strongly the RNA-conditioned raw branch is already driving the
        prediction at that locus for this sample) and only the top
        ``auto_top_loci`` are explained. Use this to avoid hand-picking loci:
        "explain whatever this patient's RNA is already having the biggest
        effect on". Returns the attribution together with the (possibly
        narrowed) ``cpg_idx`` actually explained.

        ``baseline_sample_idx``: explicit background sample_idx values to
        attribute against (real RNA rows -- see integrated_gradients.py's
        module docstring for why these must be real samples, not an
        all-zero vector). Defaults to ``n_baselines`` samples drawn uniformly
        at random (``seed``) from the RNA cache, excluding ``sample_idx``.
        """
        cpg_idx = np.asarray(cpg_idx, dtype=np.int64)
        rna_row = torch.from_numpy(self.rna.rows(np.asarray([sample_idx], dtype=np.int64))[0]).to(self.device)
        emb_np, _prior_np, _sigma_np = self.features.get(cpg_idx)
        loci = torch.from_numpy(emb_np).to(self.device)

        if auto_top_loci is not None and auto_top_loci < len(cpg_idx):
            with torch.no_grad():
                residual_logit_all = self.model(rna_row.unsqueeze(0), loci)["residual_logit"].squeeze(0)  # (n_loci,)
            top = torch.topk(residual_logit_all.abs(), k=auto_top_loci).indices.cpu().numpy()
            top.sort()
            cpg_idx = cpg_idx[top]
            loci = loci[torch.from_numpy(top).to(self.device)]

        if baseline_sample_idx is None:
            rng = np.random.default_rng(seed)
            pool = np.asarray(self.rna.ids)
            pool = pool[pool != sample_idx]
            baseline_sample_idx = rng.choice(pool, size=min(n_baselines, len(pool)), replace=False)
        baselines = torch.from_numpy(self.rna.rows(np.asarray(baseline_sample_idx, dtype=np.int64))).to(self.device)

        attribution = integrated_gradients_rna(self.model, rna_row, loci, baselines=baselines, steps=steps)
        return attribution, cpg_idx

    def summarize_genes(
        self,
        attribution: GeneAttribution,
        *,
        top_k: int = 25,
    ) -> pd.DataFrame:
        """Aggregate per-locus gene attributions into one ranked table.

        For each gene: ``mean_attribution`` (signed, averaged across the
        explained loci -- net direction), ``mean_abs_attribution`` (the
        ranking key -- importance regardless of sign), and
        ``sign_consistency`` (fraction of loci where the sign agrees with
        ``mean_attribution``'s sign -- near 1.0 means the gene pushes the
        same way at every explained locus, near 0.5 means its effect is
        locus-dependent and the mean signed value is not representative).
        """
        attr = attribution.attributions.cpu().numpy()  # (n_loci, n_genes)
        mean_attr = attr.mean(axis=0)
        mean_abs = np.abs(attr).mean(axis=0)
        sign = np.sign(mean_attr)
        sign[sign == 0] = 1.0
        sign_consistency = (np.sign(attr) == sign[None, :]).mean(axis=0)
        df = pd.DataFrame(
            {
                "gene_id": self.gene_ids,
                "mean_attribution": mean_attr,
                "mean_abs_attribution": mean_abs,
                "sign_consistency": sign_consistency,
            }
        )
        return df.sort_values("mean_abs_attribution", ascending=False).head(top_k).reset_index(drop=True)
