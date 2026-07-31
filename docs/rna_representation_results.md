# Risultati degli esperimenti RNA R0–R4

Data di completamento: 2026-07-31. Tutte le 36 run richieste sono concluse:
R0 (12), R1 (9), R2 (6) e R4 (9), con seed 17, 23 e 41. Il riferimento è
F2 `concat+product` esistente, appaiato per seed. R0 `zscore` non è stato
rieseguito perché coincide con F2.

## Double-OOD, media sui tre seed

| Famiglia | MSE | Skill vs prior | Dynamic skill | Patient Pearson | Locus Pearson | Within-cancer skill | Decisione |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F2 zscore | 0.02414902 | 0.117958 | 0.207578 | 0.290283 | 0.345902 | 0.151586 | riferimento |
| R0 continuous rank | 0.02417912 | 0.116858 | 0.195108 | 0.288478 | 0.339336 | 0.141767 | non avanza |
| R0 continuous binary | 0.02423685 | 0.114750 | 0.193564 | 0.280169 | 0.334467 | 0.139461 | non avanza |
| R0 rank | 0.02426123 | 0.113859 | 0.195588 | 0.278587 | 0.324489 | 0.138775 | non avanza |
| R0 MethylProphet quantile | 0.02428951 | 0.112826 | 0.195554 | 0.277323 | 0.324950 | 0.138810 | non avanza |
| R1 experts 4 | 0.02436946 | 0.109906 | 0.205074 | 0.288553 | 0.355219 | 0.155059 | non avanza |
| R1 experts 8 | 0.02437040 | 0.109872 | 0.205080 | 0.288083 | 0.355909 | 0.155056 | non avanza |
| R1 experts 16 | 0.02437243 | 0.109798 | 0.205011 | 0.288113 | 0.355278 | 0.155029 | non avanza |
| R2 Hallmark | 0.02437586 | 0.109672 | 0.204831 | 0.288175 | 0.354338 | 0.154962 | non avanza |
| R2 Hallmark random matched | 0.02437588 | 0.109671 | 0.204827 | 0.288167 | 0.354323 | 0.154960 | non avanza |
| R4 gene query k=256 | 0.02448771 | 0.105587 | 0.188972 | 0.284777 | 0.333514 | 0.146353 | non avanza |
| R4 gene query k=128 | 0.02448780 | 0.105584 | 0.188972 | 0.284778 | 0.333524 | 0.146353 | non avanza |
| R4 gene query k=64 | 0.02448797 | 0.105577 | 0.188964 | 0.284782 | 0.333558 | 0.146352 | non avanza |

## Confronti appaiati con F2

- R0: `continuous_rank` è il più vicino a F2, ma ha ΔMSE `+0.0000301009`,
  Δdynamic-skill `-0.0124703` e Δpatient-Pearson `-0.0018047`.
- R1: tutti i valori di `K` peggiorano MSE di circa `+0.00022`; migliorano
  locus-Pearson di circa `+0.0093–0.0100`, ma senza migliorare il criterio
  primario double-OOD o la correlazione patient-wise.
- R2: Hallmark e il controllo random matched sono indistinguibili nella MSE
  media (differenza `2.67e-08`) e peggiorano F2 di circa `+0.00022684` MSE.
  Non c'è evidenza che la struttura biologica Hallmark aggiunga segnale oltre
  il controllo matched in questa configurazione.
- R4: i tre valori di `top_k` sono praticamente equivalenti e peggiorano F2
  di circa `+0.000339` MSE e `-0.01861` dynamic-skill.

Conclusione: nessuna rappresentazione R0/R1/R2/R4 soddisfa la regola di
avanzamento rispetto a F2 sui tre seed. F2 zscore resta il modello da usare
come riferimento operativo.

## Artefatti e riproducibilità

- Tabella aggregata: `artifacts/rna_branch/representation_search/representation_summary.csv`.
- Differenze appaiate seed-per-seed: `artifacts/rna_branch/representation_search/representation_summary.paired.csv`.
- Report Hallmark: `artifacts/rna_branch/representation_search/modules/hallmark_alignment_report.json`.
- Fonte Hallmark: MSigDB Human Hallmark v2026.1.Hs,
  `h.all.v2026.1.Hs.symbols.gmt`, SHA-256
  `eecaf6dad908334ae885406ec72bdc0646d8917588ed7c219fac92fc5363f596`.
