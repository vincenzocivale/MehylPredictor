"""Construction of the RNA methylation model (placeholder until the next model is chosen)."""
from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .final import EfficientSingleAttentionPredictor
from .cpg_query import GenericFunctionalQuery, LatentQuery, track_keep_mask
from .perceiver import PerceiverCpGPredictor
from .deep_decoder import DeepDecoderCpGPredictor
from .alphagenome import AlphaGenomeLocalPredictor
from .alphagenome_transfer import AlphaGenomeTwoStagePredictor
from .alphagenome_hierarchical import AlphaGenomeHierarchicalPredictor
from .alphagenome_deep import AlphaGenomeDeepDecoderPredictor, VARIANT as ALPHAGENOME_DEEP_VARIANT

ALPHAGENOME_LOCAL_VARIANT = "alphagenome_local_chr1_v2"


def build_alphagenome_residual_head(*, mode, **spec):
    """Experimental corrections to a frozen AlphaGenome predictor."""
    from .alphagenome_residual import AlphaGenomeResidualHead
    return AlphaGenomeResidualHead(mode, **spec)


def build_alphagenome_predictor(*, variant=ALPHAGENOME_LOCAL_VARIANT, **spec):
    """Explicit AlphaGenome contract; separate from functional-only trainer inputs."""
    if variant == ALPHAGENOME_DEEP_VARIANT:
        return AlphaGenomeDeepDecoderPredictor(**spec)
    if variant == 'alphagenome_hierarchical_chr1_v3':
        return AlphaGenomeHierarchicalPredictor(**spec)
    if variant == 'alphagenome_two_stage_chr1_v3':
        return AlphaGenomeTwoStagePredictor(**spec)
    if variant != ALPHAGENOME_LOCAL_VARIANT:
        raise ValueError(f"unsupported AlphaGenome variant {variant!r}")
    return AlphaGenomeLocalPredictor(**spec)

FINAL_VARIANT = "efficient_single_attn_8ffn_residual_functional8_head2"
FINAL_MODEL_KWARGS = {
    "n_ffn_blocks": 8,
    "n_functional_ffn_blocks": 8,
    "deep_query": False,
    "n_head_ffn_blocks": 2,
}
PERCEIVER_VARIANT = "perceiver_cpg_query"
# CpG-query variants (CpGRepresentationBenchmark): generic reference annotations
# vs. latent of a model trained to predict functional tracks (--genomic-fm-cache).
PERCEIVER_GENERIC_VARIANT = "perceiver_functional_generic"
PERCEIVER_LATENT_VARIANT = "perceiver_functional_latent"
# Same inherited functional query, deeper post-query decoder (``model.decoder`` options).
PERCEIVER_DEEP_VARIANT = "perceiver_deep_decoder"
SUPPORTED_FUNCTIONAL_VARIANTS = frozenset(
    {FINAL_VARIANT, PERCEIVER_VARIANT, PERCEIVER_GENERIC_VARIANT, PERCEIVER_LATENT_VARIANT,
     PERCEIVER_DEEP_VARIANT}
)


def build_functional_predictor(
    *,
    variant: str,
    input_dim: int,
    config: ModelConfig,
    final_regressor_dropout: float,
    use_mean_proxy: bool,
    cpg_dim: int | None = None,
) -> tuple[nn.Module, str]:
    if variant not in SUPPORTED_FUNCTIONAL_VARIANTS:
        raise ValueError(f"unsupported variant {variant!r}; expected {sorted(SUPPORTED_FUNCTIONAL_VARIANTS)}")
    extra: dict = {}
    if variant == FINAL_VARIANT:
        cls, label = EfficientSingleAttentionPredictor, "methylpredictor_final"
    else:
        cls, label = PerceiverCpGPredictor, "methylpredictor_perceiver"
        width = EfficientSingleAttentionPredictor.WIDTH
        opts = config.cpg_query
        if variant == PERCEIVER_GENERIC_VARIANT:
            mask = None
            if opts.get("track_contract"):  # e.g. histone+accessibility = benchmark raw2492 baseline
                mask = track_keep_mask(opts["track_contract"], list(opts.get("track_blocks", ["histone", "accessibility"])))
            extra["cpg_query"] = GenericFunctionalQuery(width, keep_mask=mask)
        elif variant == PERCEIVER_LATENT_VARIANT:
            if not cpg_dim:
                raise ValueError(f"{variant} needs the latent dimension (cpg_dim); run with --genomic-fm-cache")
            extra.update(cpg_query=LatentQuery(cpg_dim, width), cpg_dim=cpg_dim)
        elif variant == PERCEIVER_DEEP_VARIANT:
            cls = DeepDecoderCpGPredictor
            extra.update(config.decoder)
        if variant != PERCEIVER_VARIANT:
            label += "_" + variant.removeprefix("perceiver_")
    model = cls(
        input_dim, config,
        final_regressor_dropout=final_regressor_dropout,
        use_mean_proxy=use_mean_proxy,
        **extra,
        **FINAL_MODEL_KWARGS,
    )
    return model, label
