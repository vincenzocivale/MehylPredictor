# Stage E2 — RNA encoder screening

## Current authorized scope

Only the **first tranche** is authorized to run right now:

- `A0_linear_c0_seed17` (paired reference, exact C0 hyperparameters);
- `A2_linear_residual_w256` at `lr in {1e-4, 3e-4}`, same seed as A0;
- `A3_gated_residual_w256` at `lr in {1e-4, 3e-4}`, same seed as A0.

Materialize and run only `configs/rna_branch/e2_encoder_first_tranche_grid.yaml`
(5 runs total). The full grids below (`e2_encoder_lr_screen_grid.yaml` with
A1/A4/A5, and `e2_encoder_confirm_grid.yaml` with 3 seeds x 6 families) are the
target *design* for later Stage E2 phases and are checked in as infrastructure,
but must not be materialized/run until the first-tranche report is reviewed and
a decision is made to widen scope. Model selection for the first tranche is
made on validation beta-MSE only; test-set numbers are descriptive.

## Scientific invariant

This stage changes only `model.encoder`. Keep the following identical to C0:

- frozen NTv3 locus embedding and prior;
- bilinear interaction;
- variability gate;
- residual target and loss;
- sample/CpG splits and observed-value mask;
- checkpoint selection by validation beta-MSE.

The candidates all emit a 64-dimensional patient vector. The `a1_mp_b6w512`
model reproduces the *architectural bias* of the MethylProphet bottleneck MLP,
but is trained from scratch and does not load MethylProphet weights.

## Candidate families

| Code | Encoder | Primary question |
|---|---|---|
| A0 | linear C0 | reference |
| A1 | 6-block bottleneck MLP, width 512 | is an MP-like nonlinear encoder sufficient? |
| A2 | linear + residual MLP | can nonlinearity improve C0 without discarding its linear path? |
| A3 | linear + gated residual MLP | are patient-dependent program gates useful? |
| A4 | raw shallow MLP 21792→512→64 | does direct nonlinearity beat the supervised projection? |
| A5 | compact 2-block bottleneck MLP | does a lower-capacity residual MLP generalize better? |

## 0. Installation and unit tests

```bash
python -m pip install -e .
pytest -q tests/test_rna_encoder_candidates.py tests/test_rna_branch_model.py
```

## 1. Smoke test every constructor

```bash
python -m methylation_predictor.rna_branch.grid \
  --grid configs/rna_branch/e2_encoder_smoke_grid.yaml \
  --output-dir artifacts/rna_branch/stage_e2_encoder/configs/smoke

while read -r config; do
  python -m methylation_predictor.rna_branch validate --config "$config"
  python -m methylation_predictor.rna_branch train --config "$config"
done < artifacts/rna_branch/stage_e2_encoder/configs/smoke/manifest.txt
```

Required checks before proceeding:

1. all six runs complete;
2. A0 reproduces the C0 path within expected seed/run variation;
3. residual branches A2/A3 start at an exact linear path;
4. parameter counts in `metrics.json` are plausible;
5. no NaNs appear in training history or predictions.

## 2. Learning-rate screen

The short screen uses one seed and three learning rates per family. It is only an
optimization screen; it must not be used for the final architectural conclusion.

```bash
python -m methylation_predictor.rna_branch.grid \
  --grid configs/rna_branch/e2_encoder_lr_screen_grid.yaml \
  --output-dir artifacts/rna_branch/stage_e2_encoder/configs/lr_screen
```

For a SLURM array of 18 materialized configurations:

```bash
N=$(wc -l < artifacts/rna_branch/stage_e2_encoder/configs/lr_screen/manifest.txt)
sbatch --array=0-$((N-1)) \
  --export=ALL,MANIFEST=artifacts/rna_branch/stage_e2_encoder/configs/lr_screen/manifest.txt \
  jobs/slurm/run_rna_encoder_array.sh
```

Choose the learning rate independently for each family using validation beta-MSE.
Reject a learning rate only for instability or clearly inferior validation behavior;
do not use the chr1 test panel for this choice.

## 3. Three-seed confirmation

Update the six learning rates in `e2_encoder_confirm_grid.yaml` with the Phase-2
winners, then materialize and run the 18 full configurations.

```bash
python -m methylation_predictor.rna_branch.grid \
  --grid configs/rna_branch/e2_encoder_confirm_grid.yaml \
  --output-dir artifacts/rna_branch/stage_e2_encoder/configs/confirm
```

Aggregate:

```bash
python -m methylation_predictor.rna_branch.aggregate_report \
  --screening-dir artifacts/rna_branch/stage_e2_encoder/confirm \
  --output artifacts/rna_branch/stage_e2_encoder/encoder_summary.csv \
  --baseline-family a0_linear
```

## Selection rule

Primary ranking is mean validation beta-MSE across paired seeds. Use the following
as mandatory diagnostics rather than independent optimization targets:

- patient-wise median dynamic Pearson/Spearman;
- locus-wise median dynamic Pearson/Spearman;
- within-cancer skill;
- high-variability-tertile MSE and dynamic correlation;
- calibration alpha and amplitude ratio;
- seed variance, encoder parameter count and runtime.

Keep two finalists, not one:

1. the best stable single-vector encoder by validation MSE;
2. a second encoder that materially improves patient-wise/high-variability signal
   without a major locus-wise penalty.

Only these finalists proceed to the fusion stage. Do not select an encoder from a
single seed or from test MSE. After choosing finalists, rerun them with five seeds
and evaluate on a fresh locus panel/chromosome if available.

## Result — first tranche (2026-07-30)

Gate review: all 5 required checks pass. 6/6 smoke configs completed (0 NaN
cells across any training_history.csv). Unit tests
(`tests/test_rna_encoder_candidates.py`, `tests/test_rna_branch_model.py`,
12 tests) confirm every residual/gated-residual encoder starts at an exact
zero-init linear path. Parameter counts are plausible and monotonic with
declared architecture size. First-tranche training (5 runs, seed 17, full
20-epoch protocol): ranked by validation beta-MSE, `A0_linear_c0` (0.025719)
beats both `A2_linear_residual` (0.025971 / 0.026382 at lr 1e-4 / 3e-4) and
`A3_gated_residual` (0.025823 / 0.026110). No instability, no red flag.
Decision: gate cleared, widen to the full Phase-2 LR screen.

## Result — LR screen (2026-07-30)

18 runs (6 families x 3 LRs, seed 17, 5 epochs, 512-CpG cap — optimization
screen only, not an architecture verdict). Every one of the 6 families picks
**lr=1e-4** as its best validation-MSE learning rate, with no instability at
any LR tested. Full ranking and winners in
`artifacts/rna_branch/stage_e2_encoder/lr_screen_report.csv`.
`e2_encoder_confirm_grid.yaml` updated accordingly (only `a0_linear`'s LR
placeholder needed changing, from 2e-5 to 1e-4; the other five families were
already correctly pre-filled at 1e-4).

## Result — 3-seed confirm (2026-07-30)

18 runs (3 seeds x 6 families, full 20-epoch protocol, lr=1e-4 for all),
0 failures. Aggregate table: `confirm_summary.csv` /
`confirm_summary.raw.csv` / `confirm_summary.paired.csv` (paired vs
`a0_linear`).

**Primary criterion (mean validation beta-MSE across 3 seeds):**

| family | val_mse mean | val_mse std |
|---|---:|---:|
| a3_gated_residual_w256 | 0,025225 | 0,000427 |
| a1_mp_b6w512 | 0,025268 | 0,000398 |
| a0_linear | 0,025284 | 0,000483 |
| a2_linear_residual_w256 | 0,025294 | 0,000483 |
| a4_shallow_mlp_w512 | 0,025361 | 0,000526 |
| a5_bottleneck_b2w512 | 0,025363 | 0,000528 |

All six families fall inside a 0.000138 band — smaller than any single
family's own seed-to-seed standard deviation (~0.0004-0.0005). No family is
distinguishable from any other on the primary criterion; this is noise, not
signal.

**Diagnostics** (double_ood panel, descriptive only per the doc's own rule
that test-panel numbers don't drive selection): paired per-seed diff vs
`a0_linear` on `skill_vs_prior` is not significant for any family
(`a1_mp_b6w512`: +0.0040±0.0116, p=0.61 — diff smaller than its own std;
`a2`/`a3`/`a4` negative and non-significant; `a5_bottleneck_b2w512`:
-0.0064±0.0023, p=0.04, the only nominally significant result and it is
*against* adopting a more complex encoder). High-variability-tertile skill
and patient-wise/locus-wise dynamic Pearson show the same pattern: `a1`
edges out `a0` by margins of 0.001-0.002 on every diagnostic, none of it
material, none of it distinguishable from seed noise. No family shows a
locus-wise penalty either way.

**Cost**: `a1_mp_b6w512` (24.1M params, 13.8x `a0_linear`'s 1.75M, ~200s vs
~134s per run) would be the most expensive candidate to adopt for a
statistically indistinguishable (p=0.61) result.

## Decision

**Keep `A0_linear_c0`** (the existing C0 linear RNA encoder). No candidate
in this screen — residual, gated-residual, shallow MLP, or MP-style deep
bottleneck MLP — clears a real margin over plain linear on the mandated
primary criterion (validation beta-MSE), and none shows a material,
seed-robust gain on any secondary diagnostic either. Per the doc's own
two-finalist rule, a second finalist is only kept if it "materially
improves patient-wise/high-variability signal without a major locus-wise
penalty" — no family meets that bar, so **no second finalist advances**.
Nothing proceeds to the fusion stage from this encoder screen; the 5-seed/
fresh-chromosome re-verification step is skipped because there is no
finalist requiring it.

This is the third independent confirmation in this project (Stage A:
MLP-vs-linear RNA compression; Stage B: PCA/bottleneck/random-projection
RNA compression; Stage E2: encoder depth/gating/bottleneck-MLP complexity)
that added nonlinearity or capacity in the RNA-encoding path does not
measurably improve this system's methylation predictions — the ceiling is
elsewhere, not in RNA-encoder expressiveness.

## Stage E2.5 — is the plain-linear result a data-scale artifact? (2026-07-30)

Stage E2's confirm grid tied `A0_linear` against `A3_gated_residual` almost
exactly (val MSE 0.025284 vs 0.025225, inside noise). Before closing the
question, this stage asks whether that tie is a *data-scale* artifact: does
`A3` (representative nonlinear encoder — picked over A4 as the strongest,
not weakest, nonlinear contender in the E2 confirm) pull ahead of `A0` as
training data grows, implying more data (not more architecture search)
is the missing ingredient?

**Setup.** Two independent scaling axes, everything else identical to the E2
confirm protocol (same frozen prior/embeddings/splits, bilinear interaction,
variability gate, loss, checkpoint selection, lr=1e-4, 20 epochs):

- **E2.5-A (patients)**: nested, cancer-type-stratified subsets of the 7,304
  train patients at 25/50/75/100% (1,838/3,660/5,490/7,304), all train CpGs
  kept.
- **E2.5-B (loci)**: nested, variability-tertile-stratified subsets of the
  5,243 train CpGs at 25/50/100% (1,311/2,622/5,243), all train patients kept.

Nesting is by construction: one fixed-seed RNG consumed in a fixed group
order regardless of fraction, so the 25% pool is a strict subset of 50%,
which is a strict subset of 75%/100% (verified directly: every smaller pool
is a literal subset of every larger one). Preprocessing stays fixed exactly
as specified: the RNA standardizer and the gate's variability-tertile
calibration are always computed on the *full* train pool regardless of
fraction — only which patients/CpGs enter the SGD sampling pool changes.
Code: `DataBundle.training_sample_pool` / `training_cpg_pool` in `data.py`,
gated by new `train_sample_fraction` / `train_cpg_fraction` fields on
`DataConfig` (default 1.0, fully backward-compatible no-op). 3 seeds
(17/23/41) per (fraction, encoder) cell. The 100% point on both curves
reuses the existing E2-confirm `a0_linear`/`a3_gated_residual_w256` runs
rather than re-running them. 30 new runs, 0 failures.

**G(n) = MSE_A3(n) − MSE_A0(n)**, paired per seed, mean±std over 3 seeds
(negative = nonlinear better; full per-seed values in
`artifacts/rna_branch/stage_e2_5/{patient,locus}_scaling_paired.csv`):

| patients used | G(val MSE) | G(test/double_ood MSE) | G(tertile-high skill) |
|---:|---:|---:|---:|
| 1,838 (25%) | +0.000044 ± 0.000034 | +0.000191 ± 0.000132 | −0.0089 |
| 3,660 (50%) | −0.000021 ± 0.000083 | −0.000133 ± 0.000297 | +0.0080 |
| 5,490 (75%) | −0.000088 ± 0.000074 | −0.000183 ± 0.000505 | +0.0098 |
| 7,304 (100%) | −0.000059 ± 0.000076 | +0.000056 ± 0.000073 | −0.0030 |

| CpGs used | G(val MSE) | G(test/double_ood MSE) | G(tertile-high skill) |
|---:|---:|---:|---:|
| 1,311 (25%) | +0.000613 ± 0.000429 | +0.000638 ± 0.000632 | **−0.0397** (all 3 seeds negative: −0.018/−0.077/−0.025) |
| 2,622 (50%) | −0.000048 ± 0.000250 | −0.000219 ± 0.000169 | +0.0093 |
| 5,243 (100%) | −0.000059 ± 0.000076 | +0.000056 ± 0.000073 | −0.0030 |

**Reading against the pre-registered rule.** Neither axis produces the
"promising" signature (G(n) decreasing monotonically toward negative):

- **Patients**: G(val MSE) does not move monotonically — it goes
  +→~0→more-negative→back-to-~0 from 25%→50%75%→100%, and every value's
  magnitude (≤0.00009) is smaller than that value's own seed-to-seed std at
  3 of 4 points. Per-seed signs at 25% are consistently positive (a3 worse:
  +0.000083/+0.000022/+0.000026) and at 75% consistently negative (a3
  better: −0.000132/−0.000129/−0.000003), but the pattern reverts to mixed
  signs at 100% rather than continuing to strengthen. This is "stays
  roughly at zero across sizes," not a growing advantage.
- **Loci**: the one seed-robust effect in the whole experiment is at 25%
  CpGs, and it points the *wrong* way for the "needs more data" story — A3
  is consistently worse there (all 3 seeds agree on both val MSE and
  tertile-high skill direction). That penalty shrinks to noise by 50% and
  100% CpGs, i.e. A3 recovers to parity with more loci, but never crosses
  into an advantage at any tested size. A shrinking penalty that bottoms out
  at zero is not the same as a growing benefit.
- **Train–validation gap** (train_beta_mse vs best_validation_mse, the
  overfitting diagnostic) shrinks with more data as expected, but tracks
  almost identically for A0 and A3 at every fraction on both axes (e.g.
  locus 25%: 0.0122 vs 0.0129; patient 75%: 0.00392 vs 0.00390) — no sign
  that A3's extra capacity is being disproportionately wasted on overfitting
  at small n, which is what you'd expect to see fading away if "data-limited
  encoder" were the right story.
- **Patient-wise dynamic Pearson** shows the same no-trend pattern as
  val/test MSE on both axes — small, sign-flipping deltas, no cluster of
  seeds agreeing on a growing edge for A3.

**Decision**: neither "encoder is data-limited" nor "the bottleneck is the
RNA–locus interaction, more loci fix it" is supported. Both curves land in
the third bucket from the pre-registered rule: *stays roughly zero (or, for
scarce loci, mildly positive/worse) at every tested size* → **do not invest
further in dense RNA encoders**. This is now the fourth independent line of
evidence in this project (Stage A compression, Stage B compression, Stage E2
architecture-at-fixed-data, Stage E2.5 scaling on two independent axes) that
converges on the same conclusion: `A0_linear_c0` remains the selected
encoder, and the system's performance ceiling is not RNA-encoder capacity or
training-set size in the ranges tested here.
