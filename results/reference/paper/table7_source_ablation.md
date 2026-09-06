# A.2 — MethylProphet Table 7: source ablation

**Source: MethylProphet paper (published), not our training.** Tests how much EPIC and WGBS help
beyond TCGA Array alone. Context for the dataset, not a primary paper contribution.

| Training data | Train CpG × Val Sample | Val CpG × Train Sample | Val CpG × Val Sample (double-OOD) |
|---|---:|---:|---:|
| Array | 0.4000 | 0.2769 | 0.2597 |
| Array + WGBS | 0.4705 | 0.8674 | 0.0369 |
| Array + EPIC | 0.5226 | 0.3727 | 0.3451 |
| Array + EPIC + WGBS | 0.5455 | 0.4194 | 0.3904 |

**Ours**: see [`../ours/06_source_ablation.yaml`](../ours/06_source_ablation.yaml) — **not started**,
lowest priority (do only after §B.1–B.5 are closed), feasibility under the current engine
unconfirmed (see that file).

Full MAS-PCC/MAC-PCC/MSE/MAE detail and provenance:
[`../appendix/methylprophet_comparison/table7_source_ablation.md`](../appendix/methylprophet_comparison/table7_source_ablation.md)
(also carries the retired two-stage architecture's historical "Ours" numbers per source — kept
for record, not reproducible by current code, and not the current-architecture source ablation
this file's "Ours" row will eventually be).
