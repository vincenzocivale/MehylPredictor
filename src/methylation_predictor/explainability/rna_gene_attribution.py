"""Sample-level gene-importance summaries built on top of ``integrated_gradients_rna``.

Loads a frozen ``rna_methylation`` checkpoint plus the same
``LocusFeatureCache``/``RNACache`` pair the trainer/evaluator use (see
``rna_training/evaluator.py::ScopedRNAEvaluator`` -- this mirrors its
checkpoint/cache loading so the two paths never see different data), runs
Integrated Gradients for one sample against a set of CpG loci, and aggregates
the per-locus gene attributions into one ranked table: "which genes drive
this patient's RNA-conditioned methylation predictions".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import EncoderConfig, InteractionConfig, ModelConfig
from ..models import RNAMethylationPredictor
from ..storage import LocusFeatureCache, RNACache
from ..tcga_canonical import TCGACanonicalBundle
from .integrated_gradients import GeneAttribution, integrated_gradients_rna


def _model_config(raw: dict) -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(**raw.get("encoder", {})),
        interaction=InteractionConfig(**raw.get("interaction", {})),
        zero_init_residual=raw.get("zero_init_residual", True),
        variance_normalized_residual=raw.get("variance_normalized_residual", True),
    )


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
        cfg = _model_config(state.get("model_config", {}))
        if len(self.gene_ids) != self.rna.values.shape[1]:
            raise ValueError(
                f"RNA cache has {self.rna.values.shape[1]} genes but the canonical bundle's "
                f"gene_ids has {len(self.gene_ids)} -- cache/bundle mismatch"
            )
        self.model = RNAMethylationPredictor(len(self.gene_ids), 1536, cfg).to(self.device)
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
        one cheap no-grad forward pass ranks it by ``|raw_delta|`` (how much
        the model already deviates from the CpG-only prior at that locus for
        this sample) and only the top ``auto_top_loci`` are explained. Use
        this to avoid hand-picking loci: "explain whatever this patient's RNA
        is already having the biggest effect on". Returns the attribution
        together with the (possibly narrowed) ``cpg_idx`` actually explained.

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
                representation = self.model.rna_encoder(rna_row.unsqueeze(0))
                raw_delta_all = self.model.interaction(representation, loci).squeeze(0)  # (n_loci,)
            top = torch.topk(raw_delta_all.abs(), k=auto_top_loci).indices.cpu().numpy()
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
