# RNA residual branch experiments

This module evaluates how patient RNA modifies a frozen, locus-specific NTv3 methylation prior. It is deliberately isolated from `third_party/MethylProphet` and from the genomic encoder training code.

## Scientific target

For sample `s` and CpG `i`, the model predicts a residual in logit space:

```text
b_i                 = logit(clip(NTv3_prior_i))
delta_hat_{s,i}     = gate_i * interaction(RNA_s, NTv3_embedding_i)
beta_hat_{s,i}      = sigmoid(b_i + delta_hat_{s,i})
```

When `anchor_to_mean_rna: true`, the interaction computed from the training-mean RNA profile is subtracted exactly. Therefore a mean RNA profile returns the frozen NTv3 prior, and the RNA branch cannot redefine the static baseline.

The genomic backbone, prior head, and variability heads are never updated by this package.

## What is implemented

- Dense HDF5, NPZ, and memory-mapped NPY input backends.
- ID-based alignment across RNA, methylation, CpG embeddings, metadata, and split manifests.
- RNA normalization fitted only on training patients.
- Negative controls that never permute samples across train/validation/test boundaries:
  - `mean`
  - `cancer_type_only`
  - `shuffle_global`
  - `shuffle_within_cancer`
- RNA encoders:
  - linear projection
  - MLP
  - Perceiver-style latent bottleneck
  - linear or Fourier value encoding for Perceiver gene tokens
- RNA/CpG interactions:
  - low-rank bilinear
  - concatenation MLP with multiplicative features
  - FiLM
  - CpG-query cross-attention over Perceiver latents
- Residual gates:
  - none
  - global scale
  - locus-specific
  - locus plus predicted between/within-cancer variability
- Beta-space MSE, auxiliary logit-residual Huber loss, and residual shrinkage.
- Cartesian patient x CpG minibatches.
- Four evaluation panels:
  - in-distribution
  - sample-OOD
  - locus-OOD
  - double-OOD
- Total skill, dynamic skill, within-cancer skill, calibration alpha, amplitude ratio, macro cancer metrics, and sample/CpG win fractions.
- Training-only IncrementalPCA and residual singular-spectrum analysis.
- Config, runtime manifest, history, checkpoint, metrics, and optional predictions for every run.

## Installation

From the repository root:

```bash
pip install -e .
pip install -r requirements-rna.txt
```

For development checks:

```bash
PYTHONPATH=src pytest -q tests/test_rna_branch_*.py
```

## Canonical input contract

The default config is `configs/rna_branch/base.yaml`. Change only the paths and column names needed by your artifacts.

### 1. RNA matrix

Recommended HDF5 layout:

```text
/X           float32 [n_samples, n_genes]
/sample_idx  string  [n_samples]
/gene_ids    string  [n_genes]
```

Input values should be the same normalized/log-transformed expression representation used for the MethylProphet TCGA experiment. The module then performs an additional gene-wise z-score fitted only on training samples when `standardize_rna: true`.

Avoid performing gene selection, PCA, normalization, or imputation on the full cohort before defining sample splits.

### 2. Methylation matrix

Recommended HDF5 layout:

```text
/beta        float32 [n_samples, n_cpgs]
/sample_idx  string  [n_samples]
/cpg_idx     string  [n_cpgs]
```

Missing observations must be encoded as `NaN`. They are masked from training and evaluation.

### 3. Frozen locus embeddings

Recommended HDF5 layout:

```text
/embeddings     float32 [n_cpgs, embedding_dim]
/cpg_idx        string  [n_cpgs]
/embedding_dim  optional identifiers [embedding_dim]
```

Use the selected NTv3-650M-post, 32,768-bp, forward, central C/G-average embedding. The module loads embeddings as immutable arrays.

### 4. Locus feature table

One row per CpG, for example:

```text
cpg_idx
pred_ntv3_prior
pred_log_var_between
pred_log_var_within
```

The prior and variability predictions must be out-of-fold for training CpGs and held-out for validation/test CpGs. Do not provide in-sample predictions from heads trained on the same CpGs.

If the existing prior output uses `pred_mlp_ensemble`, set:

```yaml
prior_column: pred_mlp_ensemble
```

The two variability column names can likewise be changed in the config to match the final exported tables.

### 5. Sample metadata

One row per patient:

```text
sample_idx
cancer_type
split
```

Supported split values are arbitrary strings, but the default training/evaluation config expects `train`, `validation`, and `test`.

A patient must occur in exactly one split. The RNA and methylation sample identifiers must resolve to the same biological sample level; do not mix aliquot IDs with case IDs without an explicit deterministic mapping.

### 6. CpG split manifest

One row per CpG:

```text
cpg_idx
split
```

Use the existing 5-Mb block split. A CpG must occur in exactly one split.

## Fast input validation

```bash
python -m methylation_predictor.rna_branch validate \
  --config configs/rna_branch/base.yaml
```

This checks files, required columns, duplicate IDs, matrix dimensions, sample/CpG overlap, prior validity, and split counts before a GPU run.

Set `allow_partial_overlap: false` for formal experiments. Enable it only during initial debugging.

## Experiment 0: residual rank

Run once with per-locus centering:

```bash
python -m methylation_predictor.rna_branch low-rank \
  --config configs/rna_branch/base.yaml \
  --output artifacts/rna_branch/low_rank/total.json \
  --components 8,16,32,64,128,256
```

Then remove cancer-type means before SVD:

```bash
python -m methylation_predictor.rna_branch low-rank \
  --config configs/rna_branch/base.yaml \
  --output artifacts/rna_branch/low_rank/within_cancer.json \
  --components 8,16,32,64,128,256 \
  --within-cancer
```

Use `--max-samples` and `--max-cpgs` for a quick diagnostic. Formal results should use the full training block or a prespecified, reproducible subset.

Interpretation:

- Rapid singular-value decay supports a compact bilinear patient/locus factorization.
- Slow decay, especially after within-cancer centering, motivates multiple RNA latents and CpG-query cross-attention.

## Optional training-only RNA PCA

```bash
python -m methylation_predictor.rna_branch fit-pca \
  --config configs/rna_branch/base.yaml \
  --output artifacts/rna_branch/inputs/tcga_rna_pca256.h5 \
  --components 256 \
  --batch-size 512
```

The PCA basis is fitted on training patients only and all patients are transformed afterward. To use it, point `data.rna` to the generated HDF5 file with `values_key: X`, `row_ids_key: sample_idx`, and `col_ids_key: component_ids`.

## Materialize an experiment grid

Signal-existence controls:

```bash
python -m methylation_predictor.rna_branch.grid \
  --grid configs/rna_branch/signal_grid.yaml \
  --output-dir artifacts/rna_branch/generated_configs/signal
```

Encoder comparison:

```bash
python -m methylation_predictor.rna_branch.grid \
  --grid configs/rna_branch/encoder_grid.yaml \
  --output-dir artifacts/rna_branch/generated_configs/encoders
```

Interaction and gating comparison:

```bash
python -m methylation_predictor.rna_branch.grid \
  --grid configs/rna_branch/interaction_grid.yaml \
  --output-dir artifacts/rna_branch/generated_configs/interactions
```

Train one materialized configuration:

```bash
python -m methylation_predictor.rna_branch train \
  --config artifacts/rna_branch/generated_configs/signal/02_rna_linear_real.yaml
```

## Recommended experiment order

### Stage A: establish RNA signal

Run `signal_grid.yaml`, preferably with 3-5 seeds.

Advance only if real RNA satisfies all of the following on sample-OOD and double-OOD:

1. lower beta MSE than the frozen NTv3 prior;
2. positive `dynamic_skill`;
3. positive `within_cancer_skill`;
4. better results than `shuffle_within_cancer`;
5. better macro-cancer skill than the prior;
6. no systematic degradation of low-variability CpGs in a later stratified analysis.

The most important comparison is real RNA versus within-cancer shuffled RNA. A gain over `cancer_type_only` estimates information beyond diagnosis.

### Stage B: choose RNA compression

Hold the bilinear interaction and variability gate fixed. Compare:

1. linear encoder;
2. MLP encoder;
3. training-only PCA plus linear/MLP;
4. Perceiver with linear values;
5. Perceiver with Fourier values.

Do not start with the full Perceiver grid. Tokenizing approximately 25,000 genes is memory intensive. First establish signal with linear/MLP models. For the first Perceiver run, use `num_latents: 32`, `token_dim: 128`, one cross-attention block, and two latent self-attention blocks.

### Stage C: choose interaction

Hold the winning RNA encoder fixed. Compare bilinear, concat, FiLM, and cross-attention. Cross-attention is valid only with the Perceiver encoder because it requires latent RNA tokens.

### Stage D: matched benchmark with MethylProphet

This is not another architecture search. Evaluate the frozen NTv3 prior, the
optimized C0, original MethylProphet, and MethylProphet with a globally
validation-calibrated dynamic component on one identical double-OOD panel.
Patients, CpGs, observed-mask, cancer labels, split, metric definitions, and
bootstrap must be identical. A static-MethylProphet-prior plus calibrated
dynamic variant is optional. See `docs/stage_d_matched_evaluation.md` for the
strict input contract and paired bootstrap runner.

For every comparison report delta MSE, delta Skill and delta within-cancer
Skill, with paired 95% CIs from patient, genomic-block, and hierarchical
patient+block bootstrap. Do not infer robustness from aggregate means without
these intervals.

## Output layout

Each run writes:

```text
<output_dir>/
  config.yaml
  manifest.json
  training_history.csv
  best.pt
  metrics.json
  predictions_<panel>.npz  # when enabled
```

`manifest.json` records the seed, runtime environment, input file hashes, data counts, completion status, and error on failed runs.

`metrics.json` contains panel-level metrics. The primary panel is `double_ood`; `sample_ood` is the most direct patient-generalization diagnostic.

## Metric definitions

- `skill_vs_prior = 1 - MSE_model / MSE_NTv3_prior`
- `dynamic_skill`: skill after centering true and predicted residuals per CpG
- `within_cancer_skill`: dynamic skill after centering separately within each cancer type and CpG
- `dynamic_calibration_alpha`: least-squares post-hoc multiplier for predicted centered dynamics
- `dynamic_amplitude_ratio`: SD(predicted dynamics) / SD(true dynamics)

`dynamic_calibration_alpha < 1` or amplitude ratio above 1 indicates over-amplified dynamics, analogous to the behavior observed for MethylProphet.

## Important implementation decisions

- The prior is expressed and combined in logit space, while the primary loss and headline MSE remain in beta space.
- The final residual layer is initialized at zero, so training starts exactly from the NTv3 prior.
- The reference RNA vector is zero after training-only standardization, corresponding to the training mean.
- RNA shuffles are performed independently inside each sample split. Within-cancer shuffling additionally preserves the cancer-type distribution.
- Validation checkpointing uses beta-space MSE by default.
- The module never reads or modifies `third_party/MethylProphet` internals.

## Expected adaptations in this repository

The most likely small integration edits are limited to:

1. actual paths to TCGA RNA and beta matrices;
2. exact ID conventions (`sample_idx`, case ID, aliquot ID);
3. exact output column names of NTv3 prior and variability heads;
4. the location/format of the final NTv3 embedding artifact;
5. cluster module and environment commands in the SLURM script.

Do not change the evaluation logic to make these artifacts fit. Prefer a one-time deterministic conversion into the canonical contract above.
