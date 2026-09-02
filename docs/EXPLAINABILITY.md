# Explainability: which RNA genes drive a prediction

`src/methylation_predictor/explainability/` + `scripts/explain.py`. Answers "for this
patient, which RNA genes are pushing this CpG's predicted methylation away from what the
CpG embedding alone would predict, and by how much" for a frozen `rna_methylation`
checkpoint. Read-only diagnostics over a trained model -- not part of the train/eval loop
(same isolation pattern as `benchmark/methylprophet/`, see CLAUDE.md).

## Method: Expected Gradients on `raw_delta`

`RNAMethylationPredictor`'s forward pass (see `docs/RNA_METHYLATION.md`) is:

```text
logit(beta_hat) = logit(mu_i) + sigma_i * raw_delta(rna, cpg_embedding_i)
```

`raw_delta` -- not `beta_hat` -- is what gets attributed. `beta_hat` conflates two things:
"this locus is usually methylated" (`mu_i`, CpG-only) and "this sample's RNA moved it"
(the residual). Attributing `raw_delta` isolates the second, RNA-driven part, which is
also the only part any RNA feature attribution *can* explain (`mu_i`/`sigma_i` don't
depend on RNA at all).

The attribution method is [Integrated Gradients](https://arxiv.org/abs/1703.01365)
(Sundararajan et al. 2017), averaged over several real background samples instead of a
single fixed baseline -- i.e. [Expected
Gradients](https://arxiv.org/abs/1906.10670) (Erion et al. 2021), the same
baseline-averaging IG variant SHAP's `GradientExplainer` uses. For each of `n_baselines`
random other patients' RNA vectors, IG walks the straight line from that baseline to the
query sample's RNA and integrates the gradient of `raw_delta` along the way; the final
attribution is the average across baselines. This satisfies IG's completeness axiom:

```text
sum_genes(attribution) == raw_delta(sample) - mean_baseline(raw_delta(baseline))
```

`scripts/explain.py`'s JSON output reports `max_convergence_gap`, the largest violation of
that identity across the explained loci -- small relative to `raw_delta`'s own scale means
`--steps` was sufficient; large means increase it.

### Why real background samples, not an all-zero baseline

RNA input is z-scored (`RNACache`/`rna_stats.npz`, mean 0 by construction), and the
canonical encoder's first layer is `LayerNorm` (`EncoderConfig.layer_norm=True`). The
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
  it by `|raw_delta|` (how much the model already deviates from the prior there for this
  sample) and only the top `N` are explained with Expected Gradients -- "explain whatever
  this patient's RNA is already having the biggest effect on", without hand-picking loci.
- `--n-baselines`/`--baseline-sample-idx`/`--seed`: control the background sample set
  (default: `n_baselines` random other patients, reproducible via `--seed`).
- `--steps`: Expected Gradients steps per baseline (default 50; raise if
  `max_convergence_gap` is large relative to `raw_delta`).

Output is one ranked gene table per query: `mean_attribution` (signed, net direction
across the explained loci), `mean_abs_attribution` (the ranking key), and
`sign_consistency` (fraction of explained loci where the gene's effect has the same sign
as its mean -- near 1.0 means it consistently pushes one way, near 0.5 means its effect is
locus-dependent and the mean is not representative).

Programmatic access is `methylation_predictor.explainability.rna_gene_attribution.SampleGeneExplainer`
(loads the checkpoint/caches once, explains many queries) and
`methylation_predictor.explainability.integrated_gradients.integrated_gradients_rna` (the
core algorithm, model-agnostic across `interaction.kind` -- see its docstring).

## Scope

Explains one sample's gene-level contribution to `raw_delta` at chosen loci. It does not
(and is not meant to) explain the CpG-embedding side (`mu_i`/`sigma_i`/NTv3 features), nor
provide a genome-wide "important genes overall" ranking across samples -- that would need
aggregating this per-sample tool's output over a cohort, which is a straightforward
extension if a paper claim ever needs it, not something built speculatively here.
