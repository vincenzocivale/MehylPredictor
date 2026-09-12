# RNA configuration-field audit

Phase 5b inventories the remaining RNA architecture/loss configuration surface
without changing any model, recipe, checkpoint, or experiment script.

The authoritative machine-readable audit is
[`configs/config_field_audit.yaml`](../configs/config_field_audit.yaml).

## Scope

This audit covers:

- `EncoderConfig`;
- RNA-relevant fields of `ModelConfig`;
- `LossConfig`;
- `locus_cls` recipe keys consumed by `RNAMethylationTrainer`.

`TrainingConfig` and `TrackingConfig` are deliberately deferred because they
are shared with training infrastructure and/or the `cpg_statistics` workflow.

## Findings

### Keep: encoder configuration

The encoder fields remain live because the paper-facing RNA-encoder comparison
uses multiple encoder families: program tokens, bottleneck MLP, gene/pathway,
BulkFormer and BulkRNABert.

### Keep: current model selector

The live RNA model surface needs only:

- `model.encoder`;
- `model.functional_fusion_variant`.

The factory dispatches exclusively from these fields.

### Compatibility-only model fields

The following shared-backbone-era `ModelConfig` fields are parsed but do not
participate in any current predictor forward pass:

- `interaction`;
- `trunk`;
- `axial`;
- `beta_likelihood_head`;
- `zero_init_residual`;
- `variance_normalized_residual`;
- `use_prior_anchor`.

They remain temporarily because current frozen recipes still contain some of
them and resume compatibility compares resolved recipe metadata.

### Loss surface

The current paper-facing trainer directly consumes:

- `beta_mse_weight`;
- `locus_pearson_weight`;
- `locus_min_observed_samples`;
- `locus_pearson_epsilon`;
- `locus_pearson_min_target_std`.

The code still supports, but the frozen experiment surface does not currently
select:

- `sample_pearson_weight`;
- `sample_pearson_min_observed_cpgs`;
- `sample_pearson_epsilon`;
- `locus_centered_mse_weight`.

These are candidates for removal once the architecture/experiment decision is
locked.

The residual/shrinkage/Beta-NLL fields are legacy fields from retired model
generations. The live RNA trainer does not consume them.

### `locus_cls` keys

Still live:

- `use_mean_branch`;
- `aux_weight`;
- `final_regressor_dropout`.

Compatibility-only:

- `use_fusion_product`;
- `use_raw_product`;
- `product_mlp`;
- `include_raw_rna`;
- `include_raw_cpg`;
- `fusion_init_std`;
- `query_source`;
- `residual_aux_weight`;
- `raw_lr_multiplier`;
- `trunk_hidden_dim`;
- `bottleneck_dim`;
- `trunk_dropout`.

The compatibility-only keys are retained solely so pre-refactor resolved
configs can still be compared during `--resume`.

## Removal order

Do not delete compatibility fields while protected J-series runs may need
resume.

After the final architecture is selected:

1. promote the selected paper architecture;
2. delete non-retained J-series recipes/launchers;
3. rewrite `main.yaml` and retained paper recipes without compatibility keys;
4. remove compatibility parsing from `rna_training/config.py` and
   `_retired_locus_compat`;
5. remove the dead `ModelConfig`/`LossConfig` fields;
6. run checkpoint/resume regression tests;
7. only then consider renaming the historical `locus_cls_joint` storage key.
