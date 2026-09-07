# A.2 — MethylProphet Table 7: source ablation

**Source: MethylProphet paper (published), not our training.** Tests how much EPIC and WGBS help
beyond TCGA Array alone. Context for the dataset, not a primary paper contribution.

| Training data | Train CpG × Val Sample | Val CpG × Train Sample | Val CpG × Val Sample (double-OOD) |
|---|---:|---:|---:|
| Array | 0.4000 | 0.2769 | 0.2597 |
| Array + WGBS | 0.4705 | 0.8674 | 0.0369 |
| Array + EPIC | 0.5226 | 0.3727 | 0.3451 |
| Array + EPIC + WGBS | 0.5455 | 0.4194 | 0.3904 |

**Ours — CLOSED 2026-09-07, all 4 cells** (double-OOD MAS-PCC, val_cpg × val_sample):

| Training data | Ours | MethylProphet | Δ |
|---|---:|---:|---:|
| Array | 0.5608 | 0.2597 | +0.3011 |
| Array + WGBS | 0.5661 | 0.0369 | +0.5292 |
| Array + EPIC | 0.5847 | 0.3451 | +0.2396 |
| Array + EPIC + WGBS | 0.5838 | 0.3904 | +0.1934 |

Two robustness findings worth their own sentence in the manuscript: (1) Array-only already
reaches 96% of the full-data number for us, vs. 67% for MethylProphet — far less dependent on
EPIC/WGBS; (2) adding WGBS **collapses** MethylProphet's own model (0.2597→0.0369) while ours
improves slightly (0.5608→0.5661) — a real fragility in their numbers we don't share. Full detail,
checkpoints, and the small code change that unblocked this (an optional source-subset parameter,
verified not to affect any existing recipe/run) in
[`../ours/06_source_ablation.yaml`](../ours/06_source_ablation.yaml).

Full MAS-PCC/MAC-PCC/MSE/MAE detail and provenance:
[`../appendix/methylprophet_comparison/table7_source_ablation.md`](../appendix/methylprophet_comparison/table7_source_ablation.md)
(also carries the retired two-stage architecture's historical "Ours" numbers per source — kept
for record, not reproducible by current code, and not the current-architecture source ablation
this file's "Ours" row will eventually be).
