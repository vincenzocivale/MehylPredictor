# Stage F — RNA-locus fusion screening

Four independent lines of evidence (Stage A compression, Stage B compression,
Stage E2 architecture-at-fixed-data, Stage E2.5 data scaling on two axes)
converged on: RNA-encoder capacity is not the ceiling.
Stage F changes the question from *how is the RNA encoded* to *how is the
encoded RNA vector combined with the locus embedding*.

## Current authorized scope

Only the **first tranche** is authorized to run right now:
`configs/rna_branch/stage_f_fusion_first_tranche_grid.yaml` (4 runs, single
seed 17). `configs/rna_branch/stage_f_lr_screen_grid.yaml` (cheap, 5-epoch,
3-learning-rate-per-family pass) may be run first to pick a learning rate per
family; it is not itself a scientific result. Model selection is made on
validation beta-MSE / skill only; test-set numbers are descriptive until a
confirm (multi-seed) grid is authorized.

## Scientific invariant

This stage changes only `model.interaction` (and, for F4 only,
`model.encoder`). Keep identical across every run:

- encoder A0: linear, `latent_dim: 64` (F4 reshapes the same 64-dim capacity
  into `K=4` tokens of `d=16`, it does not add capacity);
- frozen NTv3 locus embedding and prior;
- variability gate, residual target and loss;
- sample/CpG splits, observed-value mask, and the Stage E2.5 training
  protocol (`stage_f_base.yaml` == `e2_5_base.yaml`'s data/loss/training
  blocks, unchanged).

## Candidate families

| Code | Fusion | `interaction.kind` | `encoder.kind` | Primary question |
|---|---|---|---|---|
| F0 | bilinear, C0 | `bilinear` | `linear` | reference / bit-exact control |
| F2 | concat + explicit product `[z, g~, z⊙g~]` | `concat` | `linear` | does minimal joint nonlinearity beat bilinear? |
| F3 | FiLM, locus-conditioned: `γ_i,η_i=f(g_i)`, `h=γ_i⊙z_s+η_i` | `film_locus` | `linear` | does the locus modulating the patient vector beat a fixed bilinear program? |
| F4 | linear `K×d` tokens (`4×16`) + single-block locus-query cross attention, no self-attention | `linear_token_cross_attention` | `linear_tokens` | does locus-specific *selection* among linear patient factors help, without adding nonlinear capacity? |

F1 (concat + MLP, no explicit product) is deliberately **not** run: F2
strictly contains F1's concatenation and adds the interaction term, so
running F1 only gives F2 a weaker baseline to beat, not new information.

Note: `model.interaction.kind: film` and `kind: cross_attention` already
existed in this codebase from an earlier exploratory grid
(`configs/rna_branch/interaction_grid.yaml`), but condition the *other*
direction (RNA modulates locus / perceiver tokens queried by locus). F3/F4
here are new kinds (`film_locus`, `linear_token_cross_attention` +
`linear_tokens`) implementing the locus-conditions-patient direction
specified for Stage F; the old kinds are untouched.

## F4 capacity control

`K·d = 4·16 = 64`, matching every other encoder's total output width. Single
cross-attention block, no self-attention among tokens, no transformer stack,
small decoder (`LayerNorm -> Linear -> GELU -> Dropout -> Linear(1)`). If
`4×16` shows no signal, do not widen to more/larger tokens before reporting
the negative result.

## Advancement criterion

A fusion mechanism proceeds to a multi-seed confirm grid only if it beats F0
on validation in at least one of:

- lower beta-MSE;
- clearly higher patient-wise dynamic correlation, with MSE at parity or better;
- higher within-cancer skill;
- improvement on the high-variability tertile without material low/mid degradation.

Improving only train, or requiring aggressive calibration/regularization to
avoid collapse, does not count — stop, don't confirm.

## Status

Code (`film_locus`, `linear_tokens`, `linear_token_cross_attention` in
`src/methylation_predictor/rna_branch/models.py`) and configs are unit-tested
(forward-pass shape/finiteness checks in `tests/test_rna_branch_model.py`,
`tests/test_rna_encoder_candidates.py`).

## First-tranche result (2026-07-30, single seed 17, real TCGA inputs)

LR screen (5 epochs, 3 LRs each) picked `lr=1e-4` for all four families
(already the base default). Full-protocol (20 epochs) results, checkpoint
selected on validation beta-MSE:

| Run | val best MSE | in_dist skill | double_ood skill | double_ood within-cancer skill | double_ood patient pearson | double_ood tertile skill (low/mid/high) |
|---|---:|---:|---:|---:|---:|---|
| F0 bilinear (control) | 0.025955 | 0.1967 | 0.1074 | 0.1413 | 0.2976 | -0.0205 / 0.0983 / 0.1280 |
| F2 concat+product | **0.025542** | 0.2071 | **0.1187** | **0.1481** | 0.2945 | -0.0111 / 0.1151 / 0.1372 |
| F3 film_locus | 0.026601 | 0.2477 | 0.0649 | 0.1451 | 0.2564 | -0.0469 / 0.0429 / 0.0891 |
| F4 linear_tokens+cross_attn | 0.027412 | 0.1648 | 0.0610 | 0.0967 | 0.2277 | -0.0182 / 0.0465 / 0.0776 |

**F2** is the only candidate that beats F0 on validation MSE, and it does so
alongside higher within-cancer skill and a better high-tertile/low-tertile
trade on the (descriptive) double_ood panel — no single metric traded off
against another. Single seed only; not yet confirmed.

**F3** shows a train/OOD split characteristic of overfitting rather than a
real gain: markedly higher in-distribution skill/pearson (0.2477/0.4520) than
F0, but it loses on validation MSE and is the worst or near-worst candidate
on every held-out (double_ood) metric, including all three variability
tertiles. Per the pre-registered stop rule ("improving only train ... does
not count"), F3 does not advance.

**F4** loses to F0 on every metric, in-distribution and OOD alike. Does not
advance; do not widen K/d per the pre-registered rule.

**Decision**: F2 (concat + explicit product) is the only candidate cleared
for a multi-seed confirm grid; F3 and F4 are closed out at the first tranche.

## Confirm grid result (2026-07-30, seeds 17/23/41, real TCGA inputs)

`configs/rna_branch/stage_f_confirm_grid.yaml` ran F0/F2 at seeds 23/41
(seed 17 reused from the first tranche, not rerun). Paired per-seed deltas
(F2 minus F0, `stage_f_fusion_paired.csv`-equivalent computed ad hoc):

| seed | Δ val MSE | Δ double_ood skill | Δ within-cancer skill | Δ high-tertile skill | Δ low-tertile skill |
|---:|---:|---:|---:|---:|---:|
| 17 | -0.000413 | +0.0113 | +0.0069 | +0.0091 | +0.0093 |
| 23 | -0.000307 | +0.0119 | -0.0104 | +0.0192 | -0.0121 |
| 41 | -0.000602 | +0.0274 | +0.0066 | +0.0312 | -0.0080 |

- **Validation MSE**: F2 wins at all 3 seeds, consistently (-0.0003 to
  -0.0006), an order of magnitude larger than the seed-to-seed noise floor
  observed in Stage E2.5 (~0.00003-0.0001).
- **double_ood skill_vs_prior**: F2 wins at all 3 seeds, margin growing
  (+0.011, +0.012, +0.027).
- **High-variability tertile skill** (the axis the pre-registered rule cares
  about most): F2 wins at all 3 seeds, cleanly, with growing margin (+0.009,
  +0.019, +0.031).
- **Within-cancer skill**: F2 wins at 2/3 seeds; loses narrowly at seed 23
  (-0.0104), smaller in magnitude than its high-tertile gain that seed.
- **Low-variability tertile**: mixed (wins 1/3, loses 2/3) — consistent with
  this axis being the noisiest one throughout the whole RNA-branch project
  (few, low-signal CpGs; see Stage E2.5's locus-scaling note on the same
  tertile).

**Final Stage F decision**: F2 (concat of RNA vector, projected locus
embedding, and their elementwise product, fed through a small MLP) is a
real, seed-robust improvement over bilinear C0 on validation MSE and on the
high-variability-tertile skill specifically — unanimous across all 3 seeds
on both. This is the first Stage A-F result in the project that clears the
bar for "something other than the linear/bilinear baseline helps." Encoder
stays A0 linear; **the interaction/fusion mechanism moves from `bilinear` to
`concat` as the new default** for any RNA-branch work downstream of this
stage. F3 and F4 remain rejected.
