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
