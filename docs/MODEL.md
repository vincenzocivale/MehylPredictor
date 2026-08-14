# Model

Current reference model:

```text
RNA (25,017) -> LayerNorm -> Linear(256) -> z_s
CpG -> frozen 1536-D NTv3 embedding e_i
[z_s, e_i, W_R(z_s) ⊙ W_C(e_i)] -> MLP -> raw_delta
logit(beta_hat_{s,i}) = mu_i + sigma_i * raw_delta_{s,i}
```

Reference objective: beta MSE 1.0, standardized residual Huber 0.1,
standardized shrinkage 1e-4, locus Pearson 0.15, sigma floor 0.05.
The flat residual, variability gate, mean-RNA anchor, direct prediction and
no-product branches are historical ablations retained in git history.
