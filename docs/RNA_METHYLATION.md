# RNA methylation model

`RNAMethylationPredictor` (former "V1"): frozen NTv3 CpG embedding + RNA ->
sample-specific methylation. See [`CPG_STATISTICS.md`](CPG_STATISTICS.md) for
the companion `CpGStatisticsPredictor` (mu, sigma) model this one consumes.

```text
RNA (25,017) -> LayerNorm -> Linear(256) -> z_s
CpG -> frozen 1536-D NTv3 embedding e_i
[z_s, e_i, W_R(z_s) ⊙ W_C(e_i)] -> MLP -> raw_delta
logit(beta_hat_{s,i}) = mu_i + sigma_i * raw_delta_{s,i}
```

Reference objective: beta MSE 1.0, standardized residual Huber 0.1,
standardized shrinkage 1e-4, locus Pearson 0.15, sigma floor 0.05.
The flat residual, variability gate, mean-RNA anchor, direct prediction and
no-product branches are historical ablations retained in git history. A 2026-08
re-check (product/product_only/cpg_product/no_product interaction-concat pieces, plus a
256/512/1024 RNA-latent-width sweep; chr1, single seed) confirmed the product term matters
but found no case strong enough to change the canonical architecture -- see
`results/reference/ablations.yaml::interaction_concat_and_latent_dim_2026_08`.
