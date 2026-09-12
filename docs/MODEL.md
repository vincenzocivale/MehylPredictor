# MethylPredictor model

MethylPredictor predicts patient-specific DNA methylation from bulk RNA while
representing each CpG from a **reference functional atlas**, not from a
patient-specific genome and not from a genomic foundation-model embedding.

The reference implementation is
`methylation_predictor.SingleRetrievalPredictor`.  The repository also contains
`IterativeRetrievalPredictor`, a depth ablation/candidate that keeps the same
inputs, RNA tokenization, mean-proxy task, final regressor, loss, and training
protocol while replacing one retrieval operation with four residual
retrieval/FFN blocks.

## Inputs

For locus \(c\), the functional input contains:

- 4,165 sparse binary regulatory-track indicators, represented with
  `EmbeddingBag`;
- 23 dense locus annotations (18 static annotations + 5 regulatory breadth
  features).

For patient \(p\), the RNA input is the canonical normalized bulk-RNA vector.

No patient-specific DNA sequence is required by the paper-facing predictor.

## Functional locus representation

Sparse regulatory tracks and dense annotations are encoded independently and
summed:

\[
h_c =
\operatorname{LN}\left(
    h^{\mathrm{tracks}}_c +
    h^{\mathrm{dense}}_c
\right),
\qquad h_c \in \mathbb{R}^{256}.
\]

The sparse track matrix is never materialized as a dense
`[n_loci, 4165]` tensor during model training.

## RNA program tokens

The transcriptome is first compressed to a shared bottleneck and then mapped
to \(K=64\) learned program tokens:

\[
b_p =
\operatorname{LN}\left(
    \operatorname{GELU}(W_b\,\operatorname{LN}(x_p))
\right),
\]

\[
R_{p,k} =
\operatorname{LN}(B_k b_p),
\qquad
R_p \in \mathbb{R}^{64 \times 256}.
\]

`ProgramTokenEncoder` performs only RNA representation.  Locus-conditioned
retrieval is implemented in a separate module.

## Single-retrieval reference model

The locus representation queries the patient's RNA program tokens:

\[
r_{p,c} =
\operatorname{MHA}
\left(
Q=\operatorname{LN}(h_c),
K=\operatorname{LN}(R_p),
V=\operatorname{LN}(R_p)
\right).
\]

This retrieval is deliberately **non-residual**: `r_pc` contains no implicit
`+ h_c` shortcut.

The final prediction uses the stable locus representation and the retrieved
patient-specific RNA context:

\[
z_{p,c} =
[
\operatorname{LN}(h_c);
\operatorname{LN}(r_{p,c})
],
\]

\[
\hat\beta_{p,c}
=
\sigma(
\operatorname{MLP}_{512\rightarrow256\rightarrow128\rightarrow1}(z_{p,c})
).
\]

The final MLP uses dropout 0.15 in the reference recipe.

## Mean-proxy auxiliary task

A training-only head predicts the mean methylation tendency of each locus:

\[
\hat\mu_c = g_\mu(h_c).
\]

The scalar mean prediction is **not** fed into the beta predictor.  Its purpose
is to shape the functional locus representation.  Because the mean head reads
only \(h_c\), its loss has no computational path into the RNA encoder,
retrieval module, or final beta regressor.

The current reference objective is:

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{beta\ MSE}}
+
0.15\,\mathcal{L}_{\mathrm{locus\ PCC}}
+
0.15\,\mathcal{L}_{\mathrm{mean}}.
\]

## Iterative-retrieval candidate

`IterativeRetrievalPredictor` starts from the locus state broadcast over
patients and applies four identical residual blocks:

\[
s^{(0)}_{p,c}=h_c,
\]

\[
s'^{(\ell)}_{p,c}
=
s^{(\ell)}_{p,c}
+
\operatorname{MHA}
\left(
Q=\operatorname{LN}(s^{(\ell)}_{p,c}),
K=\operatorname{LN}(R_p),
V=\operatorname{LN}(R_p)
\right),
\]

\[
s^{(\ell+1)}_{p,c}
=
s'^{(\ell)}_{p,c}
+
\operatorname{FFN}(\operatorname{LN}(s'^{(\ell)}_{p,c})).
\]

The original stable \(h_c\) and the final iterative state are then concatenated
and passed through the same final regressor.

## Evaluation views

The official RNA-to-methylation evaluation reports three generalization views:

- train CpG x validation sample;
- validation CpG x train sample;
- validation CpG x validation sample (double OOD).

Reported metrics include MAS-PCC, MAC-PCC, MSE, MAE, and skill versus the
locus prior where applicable.

## Implementation map

```text
src/methylation_predictor/modeling/
    reference.py   # SingleRetrievalPredictor, IterativeRetrievalPredictor
    rna.py         # ProgramTokenEncoder
    retrieval.py   # locus-to-RNA attention and iterative FFN primitives

src/methylation_predictor/storage.py
    FunctionalLocusCache

configs/models/main.yaml
    paper-facing reference recipe
```

Historical architecture-search implementations are not part of the
paper-facing model API.

## Status update (2026-09-12): the depth-ablation family is currently ahead of this reference, not settled history

The line above was accurate when written but is now stale and should not be read as a closed
question. `src/methylation_predictor/modeling/ablation.py` holds an active depth-ablation family
(`EfficientSingleAttentionPredictor`, informally "J4"/"J7"/"J8"/"J9b"/"J10" in run IDs and W&B
tags) built directly on `IterativeRetrievalPredictor`'s mechanism, and its leading arm currently
**beats both `SingleRetrievalPredictor` and this doc's own reference number** on chr1's official
double-OOD view (`val_cpg_x_val_sample`, MAS-PCC, seed 17, re-verified 2026-09-12 by reloading each
checkpoint and recomputing all three official views rather than trusting older cached summaries,
one of which turned out to be stale):

| arm | retrieval FFN blocks | functional-branch FFN blocks | MAS-PCC (val_cpg_x_val_sample) |
| --- | --- | --- | --- |
| `cross_attention` reference (`results/reference/ours/01_final_chr1_model.yaml`, prior codebase generation) | n/a | n/a | 0.5838 |
| J4 (`efficient_single_attn_residual_ffn`) | 4 | 0 | 0.5644 |
| J7 (`efficient_single_attn_8ffn_residual`) | 8 | 0 | 0.5641 |
| J10 (`efficient_single_attn_4ffn_residual_functional4`) | 4 | 4 | 0.5814 |
| **J9b (`efficient_single_attn_8ffn_residual_functional8`)** | 8 | **8** | **0.5871** |

Two things this table already tells us, both worth carrying into any future revision of this
document:

- **Retrieval-branch depth alone plateaus early.** J4 and J7 are statistically indistinguishable
  (0.5644 vs 0.5641) despite doubling the FFN-only blocks after the single cross-attention call --
  depth on the RNA-conditioned side stops helping past ~4 blocks at this data scale.
- **The functional-annotation branch (`h_c`, from `track_embedding`+`dense_encoder`) had never been
  given comparable depth until J9b/J10**, and doing so is the single largest lever found in this
  family so far. J10 (moderate, matched 4/4 depth) already beats J4/J7 by ~3% relative despite
  *less* total FFN depth than J7 alone -- confirming the effect is not an artifact of J9b's specific
  8/8 configuration. Going from J10's 4/4 to J9b's 8/8 adds a further, smaller ~1% relative gain,
  i.e. diminishing but still positive returns to depth on both branches together.
  `n_functional_ffn_blocks`/`deep_query` on `EfficientSingleAttentionPredictor` are the relevant
  knobs; see that class's docstring for the isolation rationale (the cross-attention query stays
  the shallow `h_c` unless `deep_query=True`, so the gain above is attributable to richer
  downstream features, not a changed retrieval query).

**This is not yet a promotion decision, only a documentation correction against stale claims.**
Before treating J9b (or any deeper variant) as the new paper-facing reference:

- Every number above is a single seed (17); no confirmation-seed run exists for any arm in this
  table.
- A separate, concurrently-run FFN-fusion ladder (`configs/models/functional_fusion/j9a_ffn_fusion_concat.yaml`
  through `j9e_ffn_fusion_two_stream_residual_4_4.yaml`, a different mechanism -- fusing RNA and
  functional information rather than deepening them separately) exists on this same branch and has
  not been cross-compared against the table above.
- Everything here is chr1 only; per `docs/PAPER_EXPERIMENTS.md`, a paper-facing architecture choice
  needs independent validation on the chr1-3 and ENCODE settings too, not a chr1-only decision
  reused elsewhere.

Do not edit the "reference implementation" language above to name a new winner until those four
points are resolved -- update this section instead as each one closes.
