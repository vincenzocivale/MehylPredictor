# Architecture

`src/methylation_predictor/models.py` implements exactly one architecture,
`RNA2DNAmModel`, and hard-rejects every other `model.*.kind` value in the
config at construction time (`ValueError`). `config.py`'s dataclasses still
carry extra fields for encoder/interaction/gate variants — perceiver, MoE,
gene-token, bilinear, region-aware, etc. — inherited from the pre-refactor
research code; none of them are wired into `RNA2DNAmModel` any more. Treat
any config field not mentioned below as vestigial.

## Model

```text
RNA_s (21,792 genes)
  -> LayerNorm -> Linear(21792 -> 64)                         [LinearRNAEncoder]
  -> z_s

CpG embedding_i (1536-d, frozen, from an external genomic model)
  + [pred_log_var_between_i, pred_log_var_within_i]
  -> LayerNorm -> Linear -> GELU -> Dropout -> Linear -> sigmoid  [ResidualGate]
  -> gate g_i

[z_s, e_i, proj(z_s) * proj(e_i)]
  -> LayerNorm -> Linear(-> 128) -> GELU -> Dropout(0.1) -> Linear(-> 1)  [ConcatInteraction]
  -> interaction(RNA_s, CpG_i)   (zero-initialized output head)

delta_si = g_i * (interaction(RNA_s, CpG_i) - interaction(mean_RNA, CpG_i))
beta_hat_si = sigmoid(logit(prior_i) + delta_si)
```

- **Encoder** (`model.encoder.kind = "linear"`, `LinearRNAEncoder`):
  `LayerNorm(21792) -> Linear(21792 -> 64)`. `z_s` is the sample's 64-d RNA
  latent.
- **Gate** (`model.gate.kind = "variability"`, `ResidualGate`): a per-CpG
  sigmoid scalar `g_i in [0,1]`, computed from the CpG's frozen embedding
  plus its two variability features (`pred_log_var_between`,
  `pred_log_var_within`). Controls how much the residual branch is allowed
  to move a given CpG away from its prior — CpGs the upstream variability
  model considers static get gated toward zero, high-variability CpGs get
  more room.
- **Interaction** (`model.interaction.kind = "concat"`, `ConcatInteraction`):
  concatenates `z_s`, the CpG embedding, and their projected element-wise
  product, then an MLP (`LayerNorm -> Linear(128) -> GELU -> Dropout(0.1) ->
  Linear(1)`) with a zero-initialized output layer.
- **Mean-RNA anchoring** (`model.anchor_to_mean_rna = true`): the residual is
  the *difference* between `interaction(RNA_s, CpG_i)` and
  `interaction(mean_RNA, CpG_i)`, not the raw interaction output. This
  centers the residual branch so that a "typical" patient reproduces the
  prior almost exactly, and only RNA that deviates from the population mean
  moves the prediction.
- **Zero-initialized residual** (`model.zero_init_residual = true`): the
  interaction head's final linear layer starts at all-zero weights/bias, so
  `delta_si = 0` at initialization and training starts exactly at the frozen
  prior in logit space.

Parameter count of the trained model: **1,859,078** total
(1,438,336 encoder + 319,105 interaction + 101,637 gate).

## The frozen prior and variability features

`prior_i` (`pred_ntv3_prior` in `locus_features.parquet`) and the two
variability features (`pred_log_var_between`, `pred_log_var_within`) are
**not** produced by this repo. They are frozen outputs of an upstream
genomic-embedding pipeline that no longer exists in this repo's working
tree (removed by the training-only refactor; recoverable from git history at
commit `253bd56` if ever needed again):

1. Per-chromosome embeddings extracted from a genomic foundation model
   (NTv3-650M) run over hg38.
2. A Ridge(alpha=10) + 3-seed MLP ensemble (`LayerNorm -> 256 -> 64 -> 1`)
   fitted on those embeddings to predict the prior methylation level and its
   between-/within-cancer-type variance components.
3. For CpGs in the official `train` split (where in-sample predictions would
   leak), values are 5-fold out-of-fold ensemble predictions instead of
   in-sample ones.

This repo only trains the residual head described above on top of those
frozen values — it never re-fits the prior or the variability model.
