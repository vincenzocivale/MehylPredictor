# A.3 — MethylProphet Table 8: chromosome generalization

**Source: MethylProphet paper (published), not our training.** "The ablation of increasing
training data scale by adding chromosomes for TCGA data."

**Verified 2026-09-06 against the actual paper table (transcribed in full by the user directly
from the paper, including MAC-PCC/MSE/MAE columns not previously on file) — no longer a
from-memory transcription.**

| Train Chr. | Val Chr. | View | MAS-PCC | MAC-PCC | MSE | MAE |
|---|---|---|---:|---:|---:|---:|
| 1 | 1 | Train CpG × Val Sample | 0.5455 | 0.9320 | 0.0199 | 0.0882 |
| 1 | 1 | Val CpG × Train Sample | 0.4194 | 0.9065 | 0.0266 | 0.1000 |
| 1 | 1 | Val CpG × Val Sample (double-OOD) | 0.3904 | 0.9059 | 0.0271 | 0.1011 |
| 1+2+3 | 1 | Train CpG × Val Sample | 0.4928 | 0.9249 | 0.0219 | 0.0915 |
| 1+2+3 | 1 | Val CpG × Train Sample | 0.3760 | 0.8961 | 0.0294 | 0.1047 |
| 1+2+3 | 1 | Val CpG × Val Sample (double-OOD) | 0.3505 | 0.8960 | 0.0298 | 0.1057 |
| 1 | 1+2+3 | Train CpG × Val Sample | 0.3025 | 0.8012 | 0.0535 | 0.1473 |
| 1 | 1+2+3 | Val CpG × Train Sample | 0.2654 | 0.8216 | 0.0492 | 0.1362 |
| 1 | 1+2+3 | Val CpG × Val Sample (double-OOD) | 0.2513 | 0.8230 | 0.0495 | 0.1368 |
| 1+2+3 | 1+2+3 | Train CpG × Val Sample | 0.4872 | 0.9246 | 0.0224 | 0.0919 |
| 1+2+3 | 1+2+3 | Val CpG × Train Sample | 0.3736 | 0.8993 | 0.0290 | 0.1027 |
| 1+2+3 | 1+2+3 | Val CpG × Val Sample (double-OOD) | 0.3460 | 0.8992 | 0.0295 | 0.1037 |

Note: the `1 → 1` (chr1 → chr1) row is the same trained-on-everything model as Table 5 / Table 7's
`T(A+E+W)` row (same numbers) — this repo calls chromosomes 1+2+3 "chr123" throughout.

**Ours**: see [`../ours/05_chromosome_generalization.yaml`](../ours/05_chromosome_generalization.yaml)
— chr1→chr1 confirmed 2026-09-06 (MAS-PCC 0.5838); chr123→chr123, chr123→chr1, chr1→chr123 queued,
running now (see `docs/PAPER_ROADMAP.md` section 5). Deliberately run last, after the architecture
is fully frozen on chr1 (a test of the architecture, not another chance to pick one).
