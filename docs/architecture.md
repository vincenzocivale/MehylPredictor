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
RNA_s (25,017 genes in the current canonical TCGA path)
  -> LayerNorm -> Linear(25017 -> 64)                         [LinearRNAEncoder]
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

- **Encoder** (`model.encoder.kind = "linear"`, `LinearRNAEncoder`): in the
  current MethylProphet-compatible TCGA path this is
  `LayerNorm(25017) -> Linear(25017 -> 64)`. `z_s` is the sample's 64-d RNA
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

Parameter count for the current 25,017-gene canonical model: **2,071,928**
total (1,651,186 encoder + 319,105 interaction + 101,637 gate). The historical
21,792-gene path has a smaller encoder and is retained only for legacy
checkpoint reproducibility.

## The frozen prior and variability features

`prior_i` (`pred_ntv3_prior` in `locus_features.parquet`) and the two
variability features (`pred_log_var_between`, `pred_log_var_within`) have two
provenance tiers:

1. **Base 408,399 Array loci.** Their frozen values come from the historical
   upstream genomic-embedding/probe pipeline. These values remain immutable
   and are never re-fitted by the current RNA model.
2. **New EPIC/WGBS loci.** `methylation_predictor.full_suite` can extend the
   input contract by extracting the same NTv3 representation and distilling
   the existing frozen Array feature map. The distillation probe is used only
   for loci absent from the base 408,399-locus store; base values stay
   bit-identical.

The historical base-feature generation pipeline used:

1. Per-chromosome embeddings extracted from a genomic foundation model
   (NTv3-650M) run over hg38.
2. A Ridge(alpha=10) + 3-seed MLP ensemble (`LayerNorm -> 256 -> 64 -> 1`)
   fitted on those embeddings to predict the prior methylation level and its
   between-/within-cancer-type variance components.
3. For CpGs in the official `train` split (where in-sample predictions would
   leak), values are 5-fold out-of-fold ensemble predictions instead of
   in-sample ones.

The current RNA-to-DNAm training treats all locus inputs as frozen.
Feature-extension fitting is a separate preprocessing stage and does not update
jointly with `RNA2DNAmModel`. See
[`data/GENOMIC_FEATURE_STORE.md`](data/GENOMIC_FEATURE_STORE.md) and
[`FULL_E2_E4_SUITE.md`](FULL_E2_E4_SUITE.md).
