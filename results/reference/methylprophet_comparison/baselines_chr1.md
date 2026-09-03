# Comparison: simplified baselines vs Ours vs MethylProphet (chr1)

- **MethylProphet**: published Table 5 results (single mixed-source model, chr1) —
  see [`table5_chr1.md`](table5_chr1.md).
- **Ours**: `RNAMethylationPredictor`, chr1, seed 17, LR 5e-5 constant, 80 epochs —
  frozen reference in `results/reference/rna_methylation/chr1.yaml`.
- **Baselines**: four simplified architectures, each trained/tuned independently on this exact
  chr1 setting (`--engine matched_chr1`, except CpG Prior which needs no training), isolating why
  the canonical architecture works — see `docs/PAPER_EXPERIMENTS.md`'s "Baseline models" section
  for what each tests and its recipe.
- **↑** higher is better, **↓** lower is better.

STATUS: **complete** (chr1) — all four baselines trained/evaluated 2026-09-01. chr123/genomewide
follow once those settings are themselves verified/built (see `docs/PAPER_EXPERIMENTS.md`). Run
provenance in `results/reference/PROVENANCE.md`.

| Val Data | Model | MAS-PCC ↑ | MSE ↓ |
|---|---|---:|---:|
| train CpG × val sample | MethylProphet | 0.5455 | 0.0199 |
| train CpG × val sample | **Ours** | **0.6327** | **0.01278** |
| train CpG × val sample | Bilinear RNA–CpG | 0.5780 | 0.01440 |
| train CpG × val sample | MLP RNA–CpG | 0.5651 | 0.01471 |
| train CpG × val sample | Global RNA Shift | 0.3297 | 0.02069 |
| train CpG × val sample | CpG Prior | n/a¹ | 0.02384 |
| val CpG × train sample | MethylProphet | 0.4194 | 0.0266 |
| val CpG × train sample | **Ours** | **0.5984** | **0.01874** |
| val CpG × train sample | MLP RNA–CpG | 0.5613 | 0.01961 |
| val CpG × train sample | Bilinear RNA–CpG | 0.5692 | 0.01985 |
| val CpG × train sample | CpG Prior | n/a¹ | 0.02734 |
| val CpG × train sample | Global RNA Shift | 0.2488 | 0.02547 |
| val CpG × val sample | MethylProphet | 0.3904 | 0.0271 |
| val CpG × val sample | **Ours** | **0.5613** | **0.01941** |
| val CpG × val sample | Bilinear RNA–CpG | 0.5273 | 0.02043 |
| val CpG × val sample | MLP RNA–CpG | 0.5198 | 0.02014 |
| val CpG × val sample | CpG Prior | n/a¹ | 0.02779 |
| val CpG × val sample | Global RNA Shift | 0.2343 | 0.02535 |

¹ CpG Prior predicts a value that is *constant across samples for a given CpG*, so the
within-CpG-across-sample Pearson correlation (MAS-PCC) is mathematically undefined (zero
predictor variance) rather than a genuine "no skill" score — report MSE/MAE for this baseline,
not MAS-PCC. Its across-CpG-per-sample correlation (MAC-PCC, well-defined since `mu` does vary
across CpGs) is 0.925/0.910/0.910 on the three views, for reference.

## Takeaways

Ranked by val-CpG × val-sample MAS-PCC: **Ours (0.5613)** > Bilinear RNA–CpG (0.5273) > MLP
RNA–CpG (0.5198) > MethylProphet (0.3904) > Global RNA Shift (0.2343) > CpG Prior (undefined,
MSE only).

- **Bilinear RNA–CpG** and **MLP RNA–CpG** are both notably strong simplified baselines — within
  ~0.03–0.04 MAS-PCC of the canonical model and clearly ahead of MethylProphet's own published
  numbers on every view. This suggests the canonical architecture's specific choice of fusion
  mechanism (explicit `[rna, cpg, proj(rna)*proj(cpg)]` product term) contributes a real but
  modest margin over either a pure low-rank bilinear interaction or a plain generic MLP fusion —
  consistent with the internal `fusion_mechanism_2026_08` and
  `interaction_concat_and_latent_dim_2026_08` ablations (`results/reference/ablations.yaml`),
  which found the same qualitative ordering on different (non-baseline, non-independently-tuned)
  variants of these same architectures.
- **Global RNA Shift** lands well below both MethylProphet and the canonical model on every
  view — a single per-patient correction with no CpG-specific RNA effect is not competitive.
- **CpG Prior** (no RNA at all) has MSE close to but slightly worse than MethylProphet's own
  numbers, and far worse than the canonical model — most of the canonical model's MSE reduction
  over the static prior comes from the RNA-conditioned residual, not just having a good prior.
