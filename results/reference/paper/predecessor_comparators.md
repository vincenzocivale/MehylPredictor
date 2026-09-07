# A.5 — Predecessor comparator (Levy/Jurgenson)

**Source: MethylProphet paper (published), not our training.** **CLOSED 2026-09-06** — transcribed
directly from the paper's own table by the user, both settings it reports for this predecessor.

Purpose: avoids a main table that is only "us vs. a single competing model" — adds the method
MethylProphet itself compared against (Levy-Jurgenson et al., 2019b).

## TCGA chr1 (the setting this repo's paper plan targets)

| Model | View | MAS-PCC | MAC-PCC | MSE | MAE |
|---|---|---:|---:|---:|---:|
| Levy-Jurgenson et al. (2019b) | Train CpG × Val Sample | 0.2630 | 0.6325 | 0.0874 | 0.2148 |
| Levy-Jurgenson et al. (2019b) | Val CpG × Train Sample | 0.2203 | 0.6563 | 0.0848 | 0.2048 |
| Levy-Jurgenson et al. (2019b) | Val CpG × Val Sample (double-OOD) | 0.2158 | 0.6562 | 0.0854 | 0.2055 |
| MethylProphet | Train CpG × Val Sample | 0.5455 | 0.9320 | 0.0199 | 0.0882 |
| MethylProphet | Val CpG × Train Sample | 0.4194 | 0.9065 | 0.0266 | 0.1000 |
| MethylProphet | Val CpG × Val Sample (double-OOD) | 0.3904 | 0.9059 | 0.0271 | 0.1011 |
| **Ours** | all three views | *(see [`../ours/01_final_chr1_model.yaml`](../ours/01_final_chr1_model.yaml): 0.7004 / 0.6311 / 0.5838)* | | | |

## ENCODE (a different setting MethylProphet also reports this predecessor on — not part of the
current paper plan, which excludes ENCODE per `docs/PAPER_ROADMAP.md`; kept here as reference so
it isn't re-looked-up later if ENCODE work resumes)

| Model | View | MAS-PCC | MAC-PCC | MSE | MAE |
|---|---|---:|---:|---:|---:|
| Levy-Jurgenson et al. (2019b) | Train CpG × Val Sample | 0.2878 | 0.8355 | 0.0182 | 0.0875 |
| Levy-Jurgenson et al. (2019b) | Val CpG × Train Sample | 0.5453 | 0.7959 | 0.0345 | 0.1250 |
| Levy-Jurgenson et al. (2019b) | Val CpG × Val Sample (double-OOD) | 0.1930 | 0.8037 | 0.0343 | 0.1262 |

Note: the ENCODE row's Val CpG × Train Sample MAS-PCC (0.5453) is higher than its Train CpG × Val
Sample MAS-PCC (0.2878) — non-monotonic across views, unlike the chr1 block or MethylProphet's own
numbers. Confirmed against the paper's own table (not a transcription error) — a real property of
this predecessor's ENCODE result, not this repo's data.
