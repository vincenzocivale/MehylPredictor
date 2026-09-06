# A.1 — MethylProphet Table 5: main comparison

**Source: MethylProphet paper (published), not our training.** Single mixed-source model,
trained on Array + EPIC + WGBS, evaluated on chr1.

| Evaluation view | MethylProphet MAS-PCC | MethylProphet MSE |
|---|---:|---:|
| Train CpG × Val Sample (new patients) | 0.5455 | 0.0199 |
| Val CpG × Train Sample (new loci) | 0.4194 | 0.0266 |
| Val CpG × Val Sample (double-OOD, headline) | 0.3904 | 0.0271 |

**Ours**: see [`../ours/01_final_chr1_model.yaml`](../ours/01_final_chr1_model.yaml) — kept in a
separate file (not duplicated here) so this file stays purely "what MethylProphet published,"
never drifts out of sync with a training re-run.

Full MAS-PCC/MAC-PCC/MSE/MAE detail and provenance:
[`../appendix/methylprophet_comparison/table5_chr1.md`](../appendix/methylprophet_comparison/table5_chr1.md)
(also carries the retired two-stage architecture's historical "Ours" numbers, kept for record —
do not cite those as the current model).
