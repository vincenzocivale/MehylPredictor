# Full chromosome-1 biological-fidelity training and matched test evaluation

## Purpose

This stage addresses the failure mode observed in the Pearson-oriented P1-P4
runs: locus-wise ordering improved, but the patient-dependent methylation
amplitude collapsed. The new objective therefore optimizes quantitative
patient-specific signal rather than correlation alone.

The primary development metric is `mas_skill_vs_prior_variable`: for every
variable CpG, compare the model SSE with the SSE of the same fixed, non-leaky
locus prior, then take the median across CpGs. The key co-metrics are
`mas_ccc_variable` and `within_cancer_mas_dynamic_r2_variable`.

CpG eligibility is fixed without using test-patient targets. It is computed from
training patients at the held-out test CpGs:

- at least 20 observed training patients;
- training-patient target standard deviation at least 0.05.

The training minibatch threshold is stricter (`64` observed samples) because
structured losses estimated from small Cartesian batches are noisy.

## Training objective

The recommended B4 configuration uses:

- global SSE ratio versus the frozen prior plus a small beta-MSE stabilizer;
- mean per-CpG SSE ratio versus the frozen prior;
- Lin CCC to penalize bias and amplitude collapse;
- within-cancer centred dynamic SSE ratio;
- a small locus Pearson auxiliary;
- three MSE-only warm-up epochs and a five-epoch linear ramp.

The existing `full_coverage` sampler is used. Every train CpG and every train
sample is visited at least once per epoch without enumerating the complete
Cartesian product.

Checkpoint selection maximizes complete-validation
`mas_skill_vs_prior_variable`, subject to:

- global skill versus prior >= 0.05;
- dynamic amplitude ratio in [0.40, 1.60].

Every epoch checkpoint is retained so the Pareto frontier can be audited.

## Run the ablation

```bash
python -m pip install -e .
pytest -q \
  tests/test_rna_branch_mas_pcc_losses.py \
  tests/test_rna_branch_biological_fidelity.py \
  tests/test_chr1_biological_fidelity_evaluator.py

python -m methylation_predictor.rna_branch.grid \
  --grid configs/rna_branch/stage_chr1_biological_fidelity_grid.yaml \
  --output-dir artifacts/rna_branch/generated_configs/chr1_biological_fidelity

while read -r CONFIG; do
  python -m methylation_predictor.rna_branch validate --config "$CONFIG"
  python -m methylation_predictor.rna_branch train --config "$CONFIG"
done < artifacts/rna_branch/generated_configs/chr1_biological_fidelity/manifest.txt
```

The grid contains one matched full-coverage MSE control and four mechanistic
ablations. Freeze the selected loss configuration on validation before running
the held-out test evaluator.

## Matched complete test evaluation

The evaluator runs our selected checkpoint on every configured
`test sample x test CpG` cell. For MethylProphet it consumes the official
released prediction rows produced by the released checkpoint. This is the
established repository convention and avoids a silently incorrect reimplementation
of MethylProphet tokenization and inference.

Both the checkpoint and its official prediction artifact are required for
provenance. The evaluator verifies:

- exact sample and CpG alignment;
- no duplicate sample-CpG pairs;
- complete coverage of every observed canonical test cell by default;
- agreement between MethylProphet `gt_methyl` and the canonical target;
- identical target, prior, eligibility mask and metric code for both models.

Example:

```bash
python scripts/rna_branch/evaluate_chr1_biological_fidelity.py \
  --our-config artifacts/rna_branch/chr1_biological_fidelity/b4_combined/config.yaml \
  --our-checkpoint artifacts/rna_branch/chr1_biological_fidelity/b4_combined/best.pt \
  --methylprophet-checkpoint /path/to/tcga_mix_chr1-bs_512-c2b2.ckpt \
  --methylprophet-predictions /path/to/methylprophet_official_predictions.parquet \
  --methylprophet-group-idx <OFFICIAL_TEST_GROUP> \
  --mp-sample-map /path/to/methylprophet_sample_index_to_tcga_id.parquet \
  --mp-sample-map-source-column sample_idx \
  --mp-sample-map-target-column sample_id \
  --mp-cpg-map /path/to/methylprophet_cpg_index_to_coordinate.parquet \
  --mp-cpg-map-source-column cpg_idx \
  --mp-cpg-map-target-column cpg_id \
  --cpg-coordinates /path/to/cpg_chr_pos_df.parquet \
  --dmr-region-annotation /path/to/frozen_cpg_to_region.parquet \
  --dmr-region-id-column cpg_id \
  --dmr-region-column region_id \
  --coordinate-id-column cpg_idx \
  --chromosome-column chr \
  --position-column pos \
  --bootstrap-replicates 2000 \
  --run-downstream \
  --output-dir artifacts/rna_branch/chr1_biological_fidelity/test_comparison
```

Do not use `--allow-partial-overlap` for the paper result. It exists only for
exploratory diagnostics when an incomplete MethylProphet artifact is supplied.

## Post-hoc calibration of the dynamic component

Added 2026-08-06 after B3's amplitude_ratio (~0.57) raised the question of
whether a global rescale could close the gap. `src/methylation_predictor/rna_branch/calibration.py`
implements

```
logit(beta_cal[s, i]) = logit(prior[i]) + alpha * (logit(raw[s, i]) - logit(prior[i]))
```

`alpha` is grid-searched **on the validation panel only, never on test**,
under one of two objectives (`--calibration-objective`):

- `mse` (default, recommended): minimizes global validation MSE. Conservative,
  cannot be gamed by median-of-ratio quirks.
- `median_skill`: maximizes median per-CpG skill-vs-prior on eligible CpGs.
  Kept only as a secondary/diagnostic criterion.

The evaluator fits both, freezes whichever `--calibration-objective` names,
applies it unchanged to the test matrices, and reports **both raw and
calibrated rows** for our model in `headline_metrics.tsv` — the primary
headline stays raw. The same procedure applies to MethylProphet only if
`--methylprophet-validation-predictions` (a released-predictions file covering
our validation split) is supplied; otherwise MethylProphet calibration is
skipped and logged as such in `report["calibration"]["methylprophet"]`.

**Measured on B3 (epoch 28, validation panel, 2026-08-06)**: alpha_mse=0.90
(MSE 0.024064→0.023989, -0.31% relative), alpha_median_skill=0.975 (median
skill 0.213→0.215, +0.002 absolute). Both essentially identity — the
amplitude_ratio<1 gap is *not* a cheap-fix global miscalibration; a naive
rescale toward amplitude_ratio=1 would need alpha~1.75 and the MSE-optimal
search rejects that (regression-to-the-mean: an under-confident-but-correlated
predictor's MSE-optimal linear recalibration need not match its raw std ratio).
See `artifacts/rna_branch/chr1_biological_fidelity/b3_skill_within/b3_locus_skill_plus_within_cancer/calibration.json`
for the full alpha grid.

## Outputs

The evaluator writes:

- `headline_metrics.tsv`;
- `biological_fidelity_report.json`;
- model-specific per-CpG and per-sample Parquet tables;
- `matched_test_matrices.npz` with exact IDs and matrices;
- paired cancer-stratified patient x 5-Mb genomic-block bootstrap intervals.

The report includes:

- median per-CpG skill versus prior;
- MAS-PCC and MAS-CCC on variable CpGs;
- median dynamic R2 overall and within cancer type;
- MSE, MAE, global skill and amplitude ratio;
- MAC-PCC and MAC-CCC;
- co-methylation and patient-neighbourhood preservation;
- one-vs-rest differential methylation effect recovery at CpG level;
- optional region-level DMR effect recovery on a frozen CpG-to-region annotation;
- optional frozen downstream cancer-type utility retention.

The region annotation must be fixed independently of model predictions (for
example, an externally defined regulatory-region set or a DMR partition frozen
before comparison). The differential, regional and downstream analyses are
biological validation layers, not hyperparameter-selection metrics. They must
be run only after the validation choice is frozen.
