# Final architecture

`RNA2DNAmModel` now exposes one production architecture only. Architecture
selection was performed on the exact MethylProphet Array-chr1 split with all
variants sharing the same data, seed, optimizer, full-coverage schedule and
checkpoint metric.

## Selection evidence

Primary metric: double-heldout MAS-PCC.

| variant | MAS-PCC | MSE | skill vs prior |
|---|---:|---:|---:|
| **RNA256 + no gate + no anchor (confirm)** | **0.5163** | **0.02058** | **+23.7%** |
| RNA256 | 0.5117 | 0.02073 | +23.1% |
| no gate | 0.5001 | 0.02089 | +22.6% |
| baseline | 0.4960 | 0.02093 | +22.4% |
| no anchor | 0.4951 | 0.02097 | +22.3% |
| direct prediction | 0.4922 | 0.02187 | +18.9% |
| no product | 0.4493 | 0.02197 | +18.6% |

The confirmation establishes three production decisions:

1. enlarge the linear RNA latent from 64 to **256**;
2. remove the variability gate and mean-RNA anchor;
3. retain both the frozen residual prior and the projected RNA×CpG product.

The old ablation switches are intentionally not executable after this point.

## Computation

For sample `s` and CpG `i`:

```text
RNA_s (25,017 genes)
  -> LayerNorm -> Linear(25,017 -> 256)
  -> z_s

CpG_i
  -> frozen NTv3-650M-post embedding e_i (1536-D)

q_si = W_r z_s * W_c e_i

[z_s, e_i, q_si]
  -> LayerNorm
  -> Linear(128)
  -> GELU
  -> Dropout(0.1)
  -> Linear(1)
  -> delta_si

beta_hat_si = sigmoid(logit(prior_i) + delta_si)
```

The last residual linear layer is zero-initialized. Therefore
`delta_si = 0` at initialization and the model starts exactly from the frozen
per-CpG prior.

There is **no** learned gate, no subtraction of an interaction evaluated at a
mean-RNA profile, and no direct complete-logit prediction mode.

## Frozen genomic inputs

The model consumes two genomic quantities:

- the 1536-D NTv3 embedding;
- a frozen scalar methylation prior.

`genomic_prior_v2` is the canonical Array prior. Its targets use only official
Array training samples; official train CpGs receive five-fold OOF predictions,
while held-out Array CpGs are predicted by a full-fit probe trained only on
train CpGs.

For auxiliary EPIC/WGBS loci in `tcga_mix_chr1`, the production preparer applies
that already-saved full-fit NTv3→prior probe. It does not fit a new probe and it
does not rerun NTv3. The previous between-/within-cancer variability features
remain in `genomic_prior_v2` for provenance and historical metrics, but they are
not model inputs anymore.

## Final training policy

The paper-final model is trained in one stage on all official `tcga_mix_chr1`
training data. Each epoch consumes a deterministic full-coverage schedule for
Array, EPIC and WGBS. Source coverage and source objective weight are separated:
with equal-source training, each source contributes one third of the integrated
per-epoch loss even though WGBS requires more batches to cover its CpG pool.

The epoch budget is frozen before training. By default the launcher converts the
completed architecture-confirm best epoch into the nearest mixed-source epoch
count that preserves the optimizer-update budget. `FINAL_EPOCHS` can override
this explicitly. No held-out Array target is inspected during training.

After the fixed final epoch, the exact Array views are evaluated and the released
MethylProphet predictions can be scored on those same cells.
