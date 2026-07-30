# Stage D2 — confronto matched con MethylProphet, esteso a F2

## Esito

Estende Stage D1 (`docs/stage_d1_report.md`) allo stesso identico pannello
double-OOD chr1 (414 pazienti test, 521 CpG test, 213.091 beta osservati),
stesso protocollo (calibrazione alpha solo su validation × CpG-test, bootstrap
percentile 95%, 2.000 repliche, blocchi genomici 5 Mb, gerarchico primario).
Tutti i numeri C0/MethylProphet del Stage D1 sono stati riprodotti bit-esatti
da questo script (`stage_d2_f2.py`, che riusa le funzioni statistiche di
`stage_d1.py` senza duplicarle) prima di aggiungere F2 — vedi verifica in
`calibration_alpha.methylprophet_dynamic = 0.714586948`, identico al report
originale.

**F2 migliora il punto stimato sia su MethylProphet originale sia su C0**, ma
supera la soglia inferenziale gerarchica prespecificata solo nel confronto con
MethylProphet, non establisce un vantaggio netto e definitivo su C0: il CI
gerarchico F2-vs-C0 è il più vicino a escludere lo zero mai osservato in
questo progetto, ma non lo esclude formalmente.

## Endpoint sul double-OOD

| Modello | MSE | MAE | Skill NTv3 | DynamicSkill | Within-cancer Skill | α diagnostico | Ampiezza |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prior empirico train-only | 0,018244 | 0,077380 | 0,333647 | 0 | 0 | 0,003505 | 0 |
| Prior statico MP | 0,033927 | 0,109253 | −0,239170 | 0 | 0 | NA | 0 |
| Prior NTv3 | 0,027379 | 0,105725 | 0 | 0 | 0 | NA | 0 |
| MethylProphet originale | 0,027193 | 0,096087 | 0,006786 | 0,170388 | 0,131661 | 0,657635 | 0,735154 |
| Prior MP + dinamica MP calibrata | 0,027025 | 0,096634 | 0,012903 | 0,226320 | 0,172164 | 0,915264 | 0,522016 |
| Prior NTv3 + dinamica MP calibrata | 0,023357 | 0,091464 | 0,146869 | 0,229570 | 0,175120 | 0,888709 | 0,543413 |
| C0 ensemble (seed 17/29/43) | 0,024091 | 0,092265 | 0,120082 | 0,214939 | 0,161352 | 0,972743 | 0,476793 |
| Prior NTv3 + dinamica C0 calibrata | 0,024062 | 0,092342 | 0,121142 | 0,215089 | 0,161136 | 0,998255 | 0,464588 |
| **F2 ensemble (seed 17/23/41)** | **0,023863** | **0,091454** | **0,128421** | **0,219797** | 0,160714 | 0,925542 | 0,508188 |
| Prior NTv3 + dinamica F2 calibrata | 0,023780 | 0,091642 | 0,131420 | 0,221197 | 0,160375 | 0,988372 | 0,475882 |
| Blend diagnostico C0+MP (media beta) | 0,020737 | 0,086908 | 0,242568 | 0,287124 | 0,208135 | 0,996016 | 0,537987 |
| Blend diagnostico F2+MP (media beta) | 0,020634 | 0,086508 | 0,246349 | 0,291505 | 0,207960 | 0,981127 | 0,550400 |

F2 ensemble è il miglior modello RNA-branch puro su MSE/skill/dynamic skill,
ma resta dietro a "Prior NTv3 + dinamica MP calibrata" (0,023357) sul punto
stimato: **il vantaggio medio di F2 sul checkpoint originale deriva
soprattutto dal prior NTv3**, non da una dinamica RNA dimostrabilmente
superiore a quella di MethylProphet — stessa lettura di Stage D1, non
cambiata da F2.

## Audit dinamico F2 vs C0

`d_{s,i} = σ(b_i+Δ) − σ(b_i)` sullo stesso pannello:

| | MP dinamica calibrata (NTv3) | MethylProphet orig. | C0 ensemble | F2 ensemble |
|---|---:|---:|---:|---:|
| patient-wise pearson (mediana) | 0,4102 | 0,4140 | 0,2885 | 0,2973 |
| locus-wise pearson (mediana) | 0,2898 | 0,2887 | 0,3644 | 0,3553 |
| amplitude ratio | 0,5434 | 0,7352 | 0,4768 | 0,5082 |
| within-cancer skill (terzile high) | 0,1990 | 0,1759 | 0,1731 | 0,1717 |
| dynamic pearson (terzile high) | 0,5148 | 0,5172 | 0,4835 | 0,4912 |

F2 **non** impara una dinamica individuale migliore: la correlazione
patient-wise si muove appena (+0,009 su C0) e resta lontanissima da MP
(0,41). La correlazione locus-wise scende leggermente (−0,009), pur restando
ben sopra MP — F2 non cambia la firma strutturale di RNA-branch (forte su
locus, debole su paziente). L'unico movimento reale è il recupero
dell'ampiezza (+0,031 verso MP) e un piccolo avvicinamento a MP nel terzile
high. Conclusione: F2 è una **migliore modellazione locus-conditioned
dell'errore ad alta varianza**, non una dinamica paziente-specifica
superiore — coerente con la sinergia raw-locus+prodotto di Stage G1.

## Bootstrap appaiato

ΔMSE è candidato meno riferimento (negativo favorisce il candidato); ΔSkill e
ΔWithin-cancer Skill sono candidato meno riferimento (positivo favorisce il
candidato).

### F2 ensemble meno MethylProphet originale

| CI | ΔMSE | ΔSkill | ΔWithin-cancer Skill |
|---|---:|---:|---:|
| Pazienti | [−0,003700, −0,002973] | [0,107253, 0,137130] | [0,015057, 0,041280] |
| Blocchi genomici 5 Mb | [−0,005929, 0,002202] | [−0,076961, 0,221228] | [−0,016100, 0,063508] |
| **Gerarchico (primario)** | **[−0,006052, 0,002707]** | **[−0,091468, 0,227177]** | **[−0,025969, 0,070714]** |

Include zero sotto il criterio gerarchico — identico esito qualitativo a
Stage D1 (C0 vs MP): favorevole a livello puntuale e per bootstrap-pazienti,
**non significativo** sotto il criterio primario prespecificato.

### F2 ensemble meno C0 ensemble

| CI | ΔMSE | ΔSkill | ΔWithin-cancer Skill |
|---|---:|---:|---:|
| Pazienti | [−0,000304, −0,000158] | [0,005829, 0,010948] | [−0,004055, 0,002773] |
| Blocchi genomici 5 Mb | [−0,000417, −0,000012] | [0,000410, 0,015452] | [−0,002706, 0,000737] |
| **Gerarchico (primario)** | **[−0,000437, 0,000020]** | **[−0,000682, 0,016954]** | **[−0,005836, 0,003278]** |

ΔMSE è significativo (esclude lo zero) sia per pazienti sia per blocchi
genomici presi separatamente — **il CI gerarchico più vicino a escludere lo
zero in tutto questo progetto**, ma lo include ancora (per un margine di
0,00002 su un intervallo di 0,00046). Con il criterio prespecificato, F2 non
può essere dichiarato statisticamente superiore a C0 sul pannello chr1.

**Avvertenza sull'aggregazione**: confrontando le metriche seed-per-seed
anziché l'ensemble delle beta (`f2_vs_c0_mean_seed_metrics`), il quadro è più
debole — ΔMSE puntuale è leggermente positivo (F2 seed-medio marginalmente
peggiore, +0,0000582) e il within-cancer skill è **significativamente
peggiore** per F2 sotto tutti e tre i bootstrap (gerarchico: [−0,014893,
−0,005815], esclude lo zero a favore di C0). Il vantaggio di F2 vive
principalmente nell'ensembling delle beta, non uniformemente in ogni singolo
seed.

## Interpretazione e limite del claim

1. F2 supera MethylProphet originale nel punto stimato e per bootstrap
   pazienti, ma non con l'inferenza gerarchica — stessa conclusione di Stage
   D1 per C0, non cambiata.
2. F2 supera C0 ensemble su MSE/skill in modo consistente su bootstrap
   pazienti e blocchi separatamente, con il CI gerarchico più vicino allo
   zero mai osservato, ma senza escluderlo formalmente. Non è un secondo
   claim di superiorità statisticamente stabilito, è un segnale forte ma
   sotto soglia.
3. Il within-cancer skill non migliora con F2 nell'aggregazione seed-per-seed
   (anzi peggiora significativamente); migliora solo nell'ensemble delle
   beta. Va riportato entrambi, non solo la versione favorevole.
4. Il benchmark confronta un checkpoint autore rilasciato e F2 addestrato su
   un protocollo diverso (stesso limite di Stage D1).
5. Il divario dinamico principale (patient-wise correlation) tra RNA-branch e
   MethylProphet non si chiude con F2.

## Artefatti

- `results/rna_branch/stage_d2_f2/stage_d2_f2_results.json`: contratto,
  tutti gli endpoint, terzili di variabilità, draw/CI bootstrap.
- `results/rna_branch/stage_d2_f2/stage_d2_f2_metrics.csv`: tabella completa.
- `src/methylation_predictor/diagnostics/methylprophet/stage_d2_f2.py`: runner
  (riusa `stage_d1.py` per le funzioni statistiche).
