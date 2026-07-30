# Stage G — fusion ablation and refinement

Frozen going into this stage (see `docs/rna_fusion_stage_f.md`): encoder A0
linear 64-dim, frozen NTv3 prior, `concat` (F2) is the new default fusion
mechanism. Stage G asks *why* F2 wins, before building anything more
elaborate on top of it.

## Stage G1 — which component of F2 does the work?

F2's joint feature map is `[z; g̃; z⊙g̃]` fed through a small MLP, where
`g̃ = P_g(g_i)` is the locus embedding projected to the same dimension as the
RNA vector `z`. Note: Stage F's actual `concat` kind concatenates the *raw*
1536-dim locus embedding, not a projected `g̃` — only the product term used a
projected pair. Stage G1 uses a clean, consistent `p = P_rna(z)`,
`q = P_locus(g)` (both →64-dim) throughout, so the four cells below isolate
exactly one design choice each without that confound. G4 in this convention
is the pre-existing (but previously unused-in-Stage-F) `interaction_mlp`
kind.

| Code | joint feature to decoder | decoder | `interaction.kind` |
|---|---|---|---|
| G0 | `p^T W q` (bilinear) | -- | `bilinear` (control, reuses Stage F confirm runs) |
| G1 | `[p; q]` | MLP | `concat_only` |
| G2 | `p⊙q` | MLP | `product_only` |
| G3 | `[p; q; p⊙q]` | linear | `concat_product_linear` |
| G4 | `[p; q; p⊙q]` | MLP | `interaction_mlp` |

This separates four hypotheses: concatenation alone is enough; the explicit
product is required; a linear combination of the expanded features is
enough; real nonlinear refinement after fusion is required.

Single seed 17 for G1-G4 (`configs/rna_branch/stage_g1_fusion_ablation_grid.yaml`);
G0 is not rerun, it reuses `f0_bilinear_c0_seed{17,23,41}` from the Stage F
confirm grid. Same advancement rule as Stage F: a cell only proceeds further
if it beats the *previous* frozen baseline (F2/G4-with-Stage-F's-concat-dims)
on validation, not just G0.

### Status

Code (`ConcatOnlyInteraction`, `ProductOnlyInteraction`,
`ConcatProductLinearInteraction` in `models.py`) and the grid config are in
place, unit-tested (`tests/test_rna_branch_model.py`).

### G1 result (2026-07-30, single seed 17, real TCGA inputs) — negative

| Run | val MSE | interaction params | double_ood skill | within-cancer skill | tertile skill (low/mid/high) |
|---|---:|---:|---:|---:|---|
| G0 bilinear (control) | 0.025955 | ~- | 0.1074 | 0.1413 | -0.0205 / 0.0983 / 0.1280 |
| G1 `[p;q]` MLP | 0.027854 | small | 0.0554 | 0.1266 | -0.0422 / 0.0634 / 0.0646 |
| G2 `p⊙q` MLP | 0.027827 | small | 0.0529 | 0.1031 | -0.0194 / 0.0573 / 0.0603 |
| G3 `[p;q;p⊙q]` linear | 0.026051 | small | 0.0966 | 0.1416 | -0.0192 / 0.0852 / 0.1167 |
| G4 `[p;q;p⊙q]` MLP (`interaction_mlp`) | 0.027592 | small | 0.0684 | 0.1122 | -0.0341 / 0.0639 / 0.0837 |
| F2 (Stage F production `concat`, raw 1536-dim locus) | **0.025542** | large (~316k) | **0.1187** | **0.1481** | -0.0111 / 0.1151 / 0.1372 |

**None of G1-G4 beat G0**, let alone F2. G3 (linear decoder on clean
projected `p,q,p⊙q`) comes closest to G0 but still loses; G4 — the *clean*
version of F2's formula, with locus projected to 64-dim like RNA before
combination — is clearly worse than both G0 and the real F2. This falsifies
the hypothesis that "concat", "explicit product", or "nonlinear decoder" in
isolation explain Stage F's win.

The likely real driver: F2's production `concat` kind feeds the **raw**
1536-dim locus embedding directly into the joint MLP's first layer
(alongside the 64-dim product term), giving that layer roughly `1536x128 ≈
197k` extra weights the whole G-series never had access to (G-series
projects locus to 64-dim *before* any combination). Interaction-module
parameter counts bear this out: F2 ≈ 316k vs. G4 ≈ 131k. Capacity alone
doesn't fully explain the ranking either (G1 has fewer params than G0 and is
much worse), but "direct, uncompressed access to the locus embedding in the
decoder" was never actually tested in isolation here — Stage G1's clean `p,q`
convention removed exactly the ingredient that may matter most.

### F1-revisited result (2026-07-30, seed 17) — resolves the open question

Ran the diagnostic cell: `raw_concat` kind = `[z; loci_raw]` through the same
MLP decoder, **no** product term (`configs/rna_branch/stage_g1b_raw_concat_grid.yaml`).
Its parameter count (1,748,230) lands almost exactly on G0's (1,748,229) —
an accidental capacity-matched comparison to bilinear.

| Run | val MSE | params | double_ood skill | tertile skill (low/mid/high) |
|---|---:|---:|---:|---|
| G0 bilinear (control) | 0.025955 | 1,748,229 | 0.1074 | -0.0205 / 0.0983 / 0.1280 |
| G1 `[p;q]` MLP (projected locus) | 0.027854 | 1,666,694 | 0.0554 | -0.0422 / 0.0634 / 0.0646 |
| G4 `[p;q;p⊙q]` MLP (projected locus) | 0.027592 | 1,674,886 | 0.0684 | -0.0341 / 0.0639 / 0.0837 |
| **F1-revisited** `[z;loci_raw]`, no product | 0.026971 | 1,748,230 | 0.0693 | -0.0211 / 0.0675 / 0.0818 |
| **F2** `[z;loci_raw;product]` (Stage F production) | **0.025542** | 1,859,078 | **0.1187** | -0.0111 / 0.1151 / 0.1372 |

F1-revisited sits between the projected-locus G-series and F2, but even at
**matched capacity with G0 it is worse than bilinear** (0.026971 vs
0.025955). So raw-locus access alone is not sufficient, and it isn't even
neutral — plain concatenation of a huge raw embedding with no product term
actively hurts relative to bilinear.

**Answer**: F2's win is a genuine interaction between two ingredients, not
attributable to either alone:
1. concatenating the **raw**, uncompressed 1536-dim locus embedding (not a
   64-dim projection) into the decoder's input, and
2. the **explicit elementwise product** of *projected* 64-dim RNA/locus
   factors, added alongside that raw concatenation.

Neither ingredient in isolation beats G0 (raw concat alone: 0.026971 worse
than G0; product-only-on-projected-factors, G2: 0.027827 worse than G0).
Only the conjunction (F2 exactly as built in Stage F) clears bilinear. This
is a synergy, not a simple ablatable sum of two independent effects — Stage
G1's original four-cell matrix, by construction, could only ever test one
ingredient at a time and so could not have found this on its own.

**Consequence for Stage G2**: the "winning joint feature map" to carry
forward is F2's own exact construction (`concat` kind, unchanged) — no
simpler G1-G4 variant recovers it, so there is nothing cheaper to swap in.
Stage G2 (refiner capacity: linear vs. 1-layer MLP vs. 2-block residual MLP
vs. gated MLP, `h in {64,128,256}`, `L in {1,2}`) should vary only the
decoder on top of F2's exact `[z; loci_raw; product]` input, not on the
G1-G4 alternatives. Not started.

## Stage G2 — refiner capacity (not started)

Conditional on G1's winning joint feature map: sweep the decoder only
(linear / 1-hidden-layer MLP / 2-block residual MLP / gated MLP), hidden
width `h in {64,128,256}`, depth `L in {1,2}`. No transformer here — F2's
representation is a single joint vector, not a natural token sequence, so a
transformer over arbitrary vector chunks would be an unjustified bias.
Candidate must improve validation across multiple seeds, not just
in-distribution.

## Residual-onto-bilinear protection for stable (low-variability) CpGs (not started)

F2 clearly helps high-variability loci; the low tertile was mixed across the
Stage F confirm seeds. Proposed structure (not yet implemented):
`Δ_new = Δ_C0 + r_θ(z, g)`, `r_θ` zero-initialized and regularized — keeps
the bilinear solution at simple loci, learns the F2-style correction where
it helps, unlike the failed *encoder* residuals (Stage E2's `linear_residual`
/ `gated_residual`), because this correction sits after RNA and locus have
already interacted, exactly where Stage F showed nonlinearity pays off. A
gated variant (`Δ = Δ_C0 + g(v_i) r_θ(z,g)`) is a later option, only if the
ungated residual already shows a real gain.

## Benchmark against MethylProphet (not started)

Once G1 (and G2, if needed) settle on a final fusion, produce a 3-seed F2
ensemble and repeat the matched benchmark already used for C0/A0: F2
ensemble vs. original MethylProphet vs. NTv3+calibrated-MP-dynamic vs. C0
ensemble. Questions: does F2 close the gap to the MP dynamic component; is
the gain mostly patient-wise correlation; does F2 keep C0's locus-wise
correlation edge; is it stable under a patient x block bootstrap. Do not use
chr1 to screen further variants beyond this — it's an exhausted development
panel at this point.

## Generalization beyond chr1 (not started)

Before any conclusive claim, freeze the F2 baseline and evaluate on loci
outside this research's exposure: other chromosomes, fresh genomic blocks,
ideally an external RNA+methylation cohort. Unlike Stage E2.5's
patient/locus scaling (which was trying to "rescue" a complex encoder that
never worked), multi-chromosome training here has a different purpose:
confirming the F2 interaction generalizes across locus diversity, not
re-litigating encoder capacity.
