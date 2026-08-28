# Comparison: MethylProphet Table 5 vs Ours

- **MethylProphet**: published Table 5 results (single mixed-source model,
  trained on Array + EPIC + WGBS, chr1)
- **Ours**: `RNAMethylationPredictor`, chr1, seed 17, LR 5e-5 constant, 80
  epochs — frozen reference in `results/reference/rna_methylation/chr1.yaml`
- **↑** higher is better, **↓** lower is better

Same row as Table 7's `T(A+E+W)` (see
[`table7_source_ablation.md`](table7_source_ablation.md)) — MethylProphet's
Table 5 and Table 7's all-sources row report the same trained-on-everything
model, just under two different table names in the paper.

| Val Data | Model | MAS-PCC ↑ | MAC-PCC ↑ | MSE ↓ | MAE ↓ |
|---|---|---:|---:|---:|---:|
| train CpG × val sample | MethylProphet | 0.5455 | 0.9320 | 0.0199 | 0.0882 |
| train CpG × val sample | **Ours** | **0.6327** | — | **0.01278** | — |
| val CpG × train sample | MethylProphet | 0.4194 | 0.9065 | 0.0266 | 0.1000 |
| val CpG × train sample | **Ours** | **0.5984** | — | **0.01874** | — |
| val CpG × val sample | MethylProphet | 0.3904 | 0.9059 | 0.0271 | 0.1011 |
| val CpG × val sample | **Ours** | **0.5613** | — | **0.01941** | — |

MAC-PCC for the frozen chr1 reference is not separately tracked in
`rna_methylation/chr1.yaml`; see `methylprophet_comparison/table7_source_ablation.md`'s
`T(A+E+W)` row (same model/split) for the full MAS-PCC/MAC-PCC/MSE/MAE set.

## Historical architecture progression on this exact benchmark

Kept as the record of *why* the variance-normalized (V1) architecture was
selected over earlier iterations — code for V0/V2/V3 no longer exists in
this repo (git history only), so these rows are not reproducible today:

| run | train CpG × val sample | val CpG × train sample | val CpG × val sample |
|---|---:|---:|---:|
| V0, 4 epochs | 0.5055 / 0.0251 | 0.4705 / 0.0217 | 0.4695 / 0.0217 |
| V0, 25 epochs | 0.5505 / 0.0235 | 0.5295 / 0.0204 | 0.5169 / 0.0207 |
| V3 (prior fix) | 0.5609 / 0.0150 | 0.5276 / 0.0204 | 0.5135 / 0.0207 |
| V2 (+ locus PCC) | 0.5773 / 0.0147 | 0.5647 / 0.0199 | 0.5342 / 0.0204 |
| V1 (+ variance normalization) | 0.5811 / 0.0144 | 0.5708 / 0.0197 | 0.5401 / 0.0201 |
| MethylProphet paper | 0.5455 / 0.0199 | 0.4194 / 0.0266 | 0.3904 / 0.0271 |

Cells are `MAS-PCC / MSE`. V1 was selected as the canonical
`RNAMethylationPredictor` architecture; the current unified-scopes pipeline's
frozen chr1 reference (table above) supersedes this table numerically — it
was re-run end-to-end under the current pipeline.

**Known protocol caveat:** the canonical bundle reproduces an 8,260/918
Array train/validation split; the MethylProphet paper reports 8,258/920
after excluding Array/WGBS patient overlap that this repo's bundle carries
no crosswalk for. See
[`../../../docs/BENCHMARK_METHYLPROPHET.md`](../../../docs/BENCHMARK_METHYLPROPHET.md)
for the full explanation.
