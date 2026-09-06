# Explainability: which RNA genes drive a prediction

`src/methylation_predictor/explainability/` + `scripts/explain.py`. Answers "for this
patient, which RNA genes are pushing this CpG's predicted methylation away from what the
CpG embedding alone would predict, and by how much" for a frozen `rna_methylation`
checkpoint (the reference `FeatureFusionLocusCLSModel`/
`FeatureFusionArchitectureVariantModel` shared-backbone family). Read-only diagnostics
over a trained model -- not part of the train/eval loop (same isolation pattern as
`benchmark/methylprophet/`, see CLAUDE.md).

## Method: Expected Gradients on `residual_logit`

The reference architecture's forward pass (see `docs/RNA_METHYLATION.md`) fuses two
branches: a locus-only "mean" branch (`h_mean`, no RNA input) and an RNA-conditioned "raw"
branch (`h_raw`) into `beta_hat`. `h_raw`'s own auxiliary probe head,
`residual_logit = residual_head(h_raw)`, is what gets attributed -- not `beta_hat`.
`beta_hat` conflates two things: "this locus is usually methylated" (the mean branch,
CpG-only) and "this sample's RNA moved it" (the raw branch). Attributing `residual_logit`
isolates the second, RNA-driven part, which is also the only part any RNA feature
attribution *can* explain (the mean branch doesn't depend on RNA at all).

The attribution method is [Integrated Gradients](https://arxiv.org/abs/1703.01365)
(Sundararajan et al. 2017), averaged over several real background samples instead of a
single fixed baseline -- i.e. [Expected
Gradients](https://arxiv.org/abs/1906.10670) (Erion et al. 2021), the same
baseline-averaging IG variant SHAP's `GradientExplainer` uses. For each of `n_baselines`
random other patients' RNA vectors, IG walks the straight line from that baseline to the
query sample's RNA and integrates the gradient of `residual_logit` along the way; the
final attribution is the average across baselines. This satisfies IG's completeness axiom:

```text
sum_genes(attribution) == residual_logit(sample) - mean_baseline(residual_logit(baseline))
```

`scripts/explain.py`'s JSON output reports `max_convergence_gap`, the largest violation of
that identity across the explained loci -- small relative to `residual_logit`'s own scale
means `--steps` was sufficient; large means increase it.

### Why real background samples, not an all-zero baseline

RNA input is z-scored (`RNACache`/`rna_stats.npz`, mean 0 by construction), and the
reference encoder's first layer is `LayerNorm` (`EncoderConfig.layer_norm=True`). The
"obvious" baseline -- the all-zero vector, i.e. "population-average expression" -- sits
exactly on a numerical singularity of `LayerNorm` (its variance term vanishes there, so
the `eps` floor dominates and the local gradient blows up). Measured directly against this
repo's encoder: a straight-line IG path from an all-zero baseline needs on the order of
10^4-10^5 steps to converge, because nearly all the attribution mass sits in an
infinitesimally narrow region of the path right next to the singularity; a handful of
straight lines from *real* RNA samples (which don't pass through the origin) converge
cleanly within 50-500 steps on the same network. This was caught empirically while adding
this feature (see `tests/test_explainability.py`), not assumed up front -- if the encoder
architecture changes to drop `LayerNorm`, a zero baseline would likely be fine again, but
there is no reason to give up the (already more standard) baseline-averaging approach.

## Usage

```bash
python scripts/explain.py \
  --checkpoint /path/to/best.pt \
  --canonical-root /path/to/methylprophet_repro_v1 \
  --feature-cache /path/to/rna_feature_cache/<scope> \
  --rna-cache /path/to/rna_zscore_cache \
  --sample-idx 1234 \
  --cpg-idx-file candidate_cpg_ids.npy \
  --auto-top-loci 20 \
  --top-k 25
```

- `--cpg-idx`/`--cpg-idx-file`: the candidate CpG loci (global ids from the CpG registry).
  With `--auto-top-loci N`, this list is treated as a *pool*: one cheap forward pass ranks
  it by `|residual_logit|` (how strongly the RNA-conditioned raw branch is already driving
  the prediction there for this sample) and only the top `N` are explained with Expected
  Gradients -- "explain whatever this patient's RNA is already having the biggest effect
  on", without hand-picking loci.
- `--n-baselines`/`--baseline-sample-idx`/`--seed`: control the background sample set
  (default: `n_baselines` random other patients, reproducible via `--seed`).
- `--steps`: Expected Gradients steps per baseline (default 50; raise if
  `max_convergence_gap` is large relative to `residual_logit`).

Output is one ranked gene table per query: `mean_attribution` (signed, net direction
across the explained loci), `mean_abs_attribution` (the ranking key), and
`sign_consistency` (fraction of explained loci where the gene's effect has the same sign
as its mean -- near 1.0 means it consistently pushes one way, near 0.5 means its effect is
locus-dependent and the mean is not representative).

Programmatic access is `methylation_predictor.explainability.rna_gene_attribution.SampleGeneExplainer`
(loads the checkpoint/caches once, explains many queries -- reconstructs whichever of
`FeatureFusionLocusCLSModel`/`FeatureFusionArchitectureVariantModel` the checkpoint's
`model_config` selects) and
`methylation_predictor.explainability.integrated_gradients.integrated_gradients_rna` (the
core algorithm -- architecture-agnostic: it calls the model's full `forward(rna,
cpg_embedding)` and reads `residual_logit` back out, so it needs no per-architecture
decomposition). Not supported: checkpoints with `encoder.kind=frozen_embedding` (e.g.
BulkRNABert) -- their RNA input is a precomputed embedding, not gene expression, so
per-gene attribution is not a meaningful question for them.

## Scope

Explains one sample's gene-level contribution to `residual_logit` at chosen loci. It does
not (and is not meant to) explain the CpG-embedding side (the mean branch / NTv3
features), nor provide a genome-wide "important genes overall" ranking across samples --
that would need aggregating this per-sample tool's output over a cohort, which is a
straightforward extension if a paper claim ever needs it, not something built
speculatively here.
