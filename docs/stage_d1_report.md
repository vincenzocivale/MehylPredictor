# Stage D1 — confronto matched con MethylProphet

## Esito

Il confronto è un **benchmark matched del checkpoint MethylProphet rilasciato**
`tcga_mix_chr1-bs_512-c2b2`; non è un claim di superiorità architetturale in
training matched. Tutti i modelli sono stati valutati sulle stesse 414 pazienti
test, 521 CpG test e 213.091 beta osservati del pannello chr1 double-OOD. I
controlli hanno verificato identità di `sample_idx`, `cpg_idx`, cancer type,
target e mask; non è stata eliminata alcuna riga osservata dopo la predizione.

L’ensemble C0 migliora il checkpoint MethylProphet nel punto stimato
(ΔMSE = −0,003101879). Tuttavia l'intervallo primario bootstrap gerarchico
pazienti × blocchi genomici 5-Mb è [−0,005893550, 0,002659391] e include zero.
Con il criterio inferenziale prespecificato, **non si può affermare che C0
superi statisticamente il checkpoint sul pannello chr1**. Il bootstrap per
pazienti isolatamente è favorevole, ma non sostituisce quello gerarchico.

## Protocollo bloccato

- CpG/pazienti/mask/cancer type: comuni e verificati prima delle metriche.
- Prior empirico: `mean_train`, stimato solo sui pazienti train.
- Prior statico MethylProphet: inferenza con lo stesso RNA medio train-only per
  tutti i pazienti.
- Prior NTv3: `pred_ntv3_prior` congelato.
- Checkpoint C0: i tre `best.pt` selezionati esclusivamente con validation
  beta-MSE (seed 17, 29, 43); l’ensemble è la media beta post-hoc dei tre.
- α globale: stimato sulle 398 pazienti validation × gli stessi 521 CpG, mai
  sul double-OOD test. α MP = 0,714586948; α C0 = 0,957325112, 0,975544765 e
  0,915461295; α dell’ensemble C0 = 0,971325355.
- Bootstrap percentile 95%, 2.000 repliche: pazienti, blocchi 5-Mb e
  gerarchico pazienti × blocchi. Il terzo è primario.

## Endpoint sul double-OOD

| Modello | MSE | MAE | Skill NTv3 | DynamicSkill | Within-cancer Skill | α diagnostico | Ampiezza |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prior empirico train-only | 0,018243748 | 0,077379615 | 0,333647399 | 0 | 0 | 0,003505 | 0,000000 |
| Prior statico MP | 0,033926648 | 0,109252862 | −0,239170247 | 0 | 0 | NA | 0 |
| Prior NTv3 | 0,027378521 | 0,105724645 | 0 | 0 | 0 | NA | 0 |
| MethylProphet originale | 0,027192729 | 0,096087314 | 0,006786023 | 0,170387765 | 0,131660531 | 0,657635 | 0,735153 |
| Prior MP + dinamica MP calibrata | 0,027025242 | 0,096634212 | 0,012903495 | 0,226319687 | 0,172163838 | 0,915264 | 0,522016 |
| Prior NTv3 + dinamica MP calibrata | 0,023357464 | 0,091464370 | 0,146869036 | 0,229569848 | 0,175119664 | 0,888709 | 0,543413 |
| C0 seed 17 | 0,024149237 | 0,092657518 | 0,117949536 | 0,209255542 | 0,155017149 | 0,962955 | 0,475394 |
| C0 seed 29 | 0,024179431 | 0,092391145 | 0,116846704 | 0,207885007 | 0,154881324 | 0,952903 | 0,479065 |
| C0 seed 43 | 0,024301579 | 0,092418247 | 0,112385238 | 0,214353826 | 0,160522942 | 0,946915 | 0,489709 |
| Ensemble C0 beta | 0,024090850 | 0,092264818 | 0,120082102 | 0,214938864 | 0,161351911 | 0,972743 | 0,476793 |
| Prior NTv3 + dinamica C0 calibrata, ensemble | 0,024061835 | 0,092341507 | 0,121141879 | 0,215088534 | 0,161136233 | 0,998255 | 0,464588 |

Il confronto diagnostico cruciale favorisce la dinamica MP dopo che il prior è
reso identico: NTv3+dynamica MP calibrata ha MSE 0,023357464, rispetto a
0,024061835 per NTv3+dynamica C0 calibrata ensemble. Il vantaggio medio C0 sul
checkpoint originale deriva quindi soprattutto dal prior NTv3, non da una
dimostrazione di dinamica RNA superiore.

## Bootstrap appaiato: C0 meno MethylProphet originale

ΔMSE è C0−MP (negativo favorisce C0); ΔSkill e ΔWithin-cancer Skill sono
C0−MP (positivo favorisce C0).

| Confronto | ΔMSE | CI gerarchico ΔMSE | CI gerarchico ΔSkill | CI gerarchico ΔWithin-cancer Skill |
|---|---:|---:|---:|---:|
| Seed 17 | −0,003043493 | [−0,006061862, 0,003548773] | [−0,120070854, 0,221179197] | [−0,030820103, 0,065522879] |
| Seed 29 | −0,003013299 | [−0,005690569, 0,002580397] | [−0,088291976, 0,210184277] | [−0,028339403, 0,064296868] |
| Seed 43 | −0,002891150 | [−0,005518592, 0,002714066] | [−0,093573609, 0,205562629] | [−0,021687804, 0,068199328] |
| Media metriche seed | −0,002982647 | [−0,005716187, 0,002892544] | [−0,100507369, 0,212729321] | [−0,026212317, 0,068624818] |
| Ensemble beta | −0,003101879 | [−0,005893550, 0,002659391] | [−0,091362365, 0,218500528] | [−0,021113546, 0,069798340] |

Gli intervalli paziente per l’ensemble sono favorevoli: ΔMSE
[−0,003470354, −0,002743187], ΔSkill [0,099083613, 0,128493088] e
ΔWithin-cancer Skill [0,015271621, 0,042786912]. Gli intervalli solo-blocco,
come quelli gerarchici, includono zero; la variabilità tra blocchi è quindi il
fattore che impedisce il claim primario.

## Interpretazione e limite del claim

1. C0 supera MethylProphet originale nel punto stimato e con bootstrap per
   pazienti, ma non con l'inferenza gerarchica prespecificata.
2. La calibrazione migliora chiaramente la dinamica MP. Combinata con NTv3,
   essa supera C0 ensemble nel punto stimato; non supporta il claim “C0 ha il
   miglior ramo RNA”.
3. Il benchmark confronta un checkpoint autore rilasciato e C0 addestrato su
   un protocollo diverso. Per attribuire differenze all’architettura occorre il
   secondo livello: riaddestrare MethylProphet con gli stessi train CpG,
   pazienti, validation, mask e input availability di C0.
4. Le metriche complete per cancer type, win rate, Pearson/Spearman residuali e
   terzili di variabilità sono nel JSON; non sono state usate per selezionare
   alcun modello sul test.

## Artefatti

- `artifacts/stage_d/d1_final/stage_d1_results.json`: contratto, tutti gli
  endpoint, risultati per cancer type e terzili, draw/CI bootstrap.
- `artifacts/stage_d/d1_final/stage_d1_metrics.csv`: tabella completa delle
  metriche per modello.
- `artifacts/stage_d/methylprophet_original.provenance.json`: checkpoint e
  verifica target contro il pannello C0.
