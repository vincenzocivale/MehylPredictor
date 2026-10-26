# Controlli a parità di supporto e attribuzione del metodo — 18 ottobre 2026

Il beneficio dell’adattamento con pochi campioni si conferma rispetto alle medie senza RNA. La superiorità della nuova architettura rispetto a una v1 ugualmente calibrata non si conferma sulle due coorti esterne. Il confronto con il metodo pubblicato a parità di training resta aperto; questi risultati sostengono un lavoro sull’adattamento e sui limiti di trasferimento, ma non bastano per dichiarare v3 metodologicamente superiore.

## Protocollo

Encoder congelati. Ogni metodo riceve gli stessi 0/1/5/10/50 pazienti di supporto e gli stessi target finiti; per WGBS il massimo è 10. Tre seed per i modelli interni e v2/v3 ufficiali; cinque insiemi di supporto annidati. L’unica v1 ufficiale è seed17: la sua ripetizione nei confronti appaiati non equivale a tre fit indipendenti.

La forza di shrinkage di ciascun metodo è selezionata esclusivamente sui tre fold genomici interni, fra 0, 5 e 20, con lo stesso budget di ricerca. Scelte congelate prima di valutare dev ed esterni. Questi dataset erano già stati esaminati in precedenza: è una conferma di follow-up, non un test intatto.

Nel dev, supporto dai pazienti training e test sugli 825 pazienti dev. In EPIC, 50 pazienti riservati al supporto e 1.656 al test per ripetizione. In WGBS, 10 al supporto e 21 al test, dopo aggregazione delle repliche tecniche per paziente. Nessun paziente esterno compare nel training Array. Gli stessi pazienti test sono mantenuti per tutti i valori di k nella stessa ripetizione; le ripetizioni possono sovrapporsi.

`support_mean`: media delle beta osservate, senza RNA; senza osservazioni finite usa il prior genomico. `support_shrunk_mean`: media regolarizzata verso il prior genomico. `support_tissue_mean` usa anche il tessuto osservato; gruppi ignoti ricadono sulla media globale. `v3_population` aggiorna la media predetta sui riferimenti RNA training, senza RNA del paziente query. `residual` corregge il residuo medio nello spazio beta e limita a [0,1]; `logit` adatta l’intercetta prima della sigmoid. v1 e v3 ricevono lo stesso prior genomico training-only e gli stessi pazienti di riferimento, codificati attraverso il proprio encoder.

## Scelte interne congelate

| method | tau |
| --- | --- |
| support_mean | 0.000000 |
| support_shrunk_mean | 5.000000 |
| support_tissue_mean | 5.000000 |
| v1_raw_logit | 5.000000 |
| v1_prior_logit | 5.000000 |
| v1_prior_residual | 5.000000 |
| v2_raw_logit | 5.000000 |
| v3_logit | 5.000000 |
| v3_residual | 5.000000 |
| v3_population | 5.000000 |
| v3_shuffle | 5.000000 |

## Risultati a parità di dati

| view | support | method | mse | mae | mas_pcc |
| --- | --- | --- | --- | --- | --- |
| dev | 0 | support_mean | 0.033423 | 0.132107 | — |
| dev | 0 | support_shrunk_mean | 0.033423 | 0.132107 | — |
| dev | 0 | v1_prior_logit | 0.025577 | 0.105461 | 0.527395 |
| dev | 0 | v1_prior_residual | 0.025577 | 0.105461 | 0.527395 |
| dev | 0 | v3_logit | 0.025059 | 0.106168 | 0.527678 |
| dev | 0 | v3_population | 0.032347 | 0.125036 | — |
| dev | 0 | v3_residual | 0.025059 | 0.106168 | 0.527678 |
| dev | 5 | support_mean | 0.028125 | 0.106789 | — |
| dev | 5 | support_shrunk_mean | 0.027048 | 0.113359 | — |
| dev | 5 | v1_prior_logit | 0.019229 | 0.088925 | 0.531397 |
| dev | 5 | v1_prior_residual | 0.019461 | 0.089653 | 0.527485 |
| dev | 5 | v3_logit | 0.018913 | 0.089303 | 0.531660 |
| dev | 5 | v3_population | 0.026796 | 0.110188 | — |
| dev | 5 | v3_residual | 0.019127 | 0.089834 | 0.527749 |
| dev | 10 | support_mean | 0.025832 | 0.103394 | — |
| dev | 10 | support_shrunk_mean | 0.025635 | 0.108124 | — |
| dev | 10 | v1_prior_logit | 0.017867 | 0.085064 | 0.532697 |
| dev | 10 | v1_prior_residual | 0.018188 | 0.086241 | 0.527754 |
| dev | 10 | v3_logit | 0.017609 | 0.085302 | 0.532626 |
| dev | 10 | v3_population | 0.025527 | 0.106173 | — |
| dev | 10 | v3_residual | 0.017899 | 0.086211 | 0.527901 |
| dev | 50 | support_mean | 0.023839 | 0.099706 | — |
| dev | 50 | support_shrunk_mean | 0.023848 | 0.100913 | — |
| dev | 50 | v1_prior_logit | 0.016355 | 0.079924 | 0.534377 |
| dev | 50 | v1_prior_residual | 0.016685 | 0.081746 | 0.528173 |
| dev | 50 | v3_logit | 0.016150 | 0.079936 | 0.534183 |
| dev | 50 | v3_population | 0.023842 | 0.100497 | — |
| dev | 50 | v3_residual | 0.016441 | 0.081410 | 0.528402 |
| epic | 0 | support_mean | 0.025816 | 0.107412 | — |
| epic | 0 | support_shrunk_mean | 0.025816 | 0.107412 | — |
| epic | 0 | v1_prior_logit | 0.020097 | 0.089781 | 0.346143 |
| epic | 0 | v1_prior_residual | 0.020097 | 0.089781 | 0.346143 |
| epic | 0 | v3_logit | 0.020340 | 0.090825 | 0.339426 |
| epic | 0 | v3_population | 0.024290 | 0.097277 | — |
| epic | 0 | v3_residual | 0.020340 | 0.090825 | 0.339426 |
| epic | 5 | support_mean | 0.020116 | 0.076226 | — |
| epic | 5 | support_shrunk_mean | 0.020098 | 0.086675 | — |
| epic | 5 | v1_prior_logit | 0.015185 | 0.072792 | 0.347156 |
| epic | 5 | v1_prior_residual | 0.015298 | 0.073238 | 0.346516 |
| epic | 5 | v3_logit | 0.015392 | 0.073629 | 0.340217 |
| epic | 5 | v3_population | 0.019714 | 0.081993 | — |
| epic | 5 | v3_residual | 0.015480 | 0.073979 | 0.339504 |
| epic | 10 | support_mean | 0.019135 | 0.074571 | — |
| epic | 10 | support_shrunk_mean | 0.019061 | 0.080926 | — |
| epic | 10 | v1_prior_logit | 0.014334 | 0.068727 | 0.347086 |
| epic | 10 | v1_prior_residual | 0.014432 | 0.069499 | 0.346851 |
| epic | 10 | v3_logit | 0.014550 | 0.069474 | 0.340597 |
| epic | 10 | v3_population | 0.018877 | 0.077976 | — |
| epic | 10 | v3_residual | 0.014610 | 0.069974 | 0.339588 |
| epic | 50 | support_mean | 0.017608 | 0.071233 | — |
| epic | 50 | support_shrunk_mean | 0.017604 | 0.072690 | — |
| epic | 50 | v1_prior_logit | 0.013138 | 0.063317 | 0.347854 |
| epic | 50 | v1_prior_residual | 0.013234 | 0.064784 | 0.347159 |
| epic | 50 | v3_logit | 0.013364 | 0.063971 | 0.340947 |
| epic | 50 | v3_population | 0.017594 | 0.072073 | — |
| epic | 50 | v3_residual | 0.013418 | 0.065009 | 0.340078 |
| wgbs | 0 | support_mean | 0.058374 | 0.177217 | — |
| wgbs | 0 | support_shrunk_mean | 0.058374 | 0.177217 | — |
| wgbs | 0 | v1_prior_logit | 0.044397 | 0.148521 | 0.398100 |
| wgbs | 0 | v1_prior_residual | 0.044397 | 0.148521 | 0.398100 |
| wgbs | 0 | v3_logit | 0.044886 | 0.150345 | 0.395354 |
| wgbs | 0 | v3_population | 0.056444 | 0.169211 | — |
| wgbs | 0 | v3_residual | 0.044886 | 0.150345 | 0.395354 |
| wgbs | 5 | support_mean | 0.049650 | 0.143683 | — |
| wgbs | 5 | support_shrunk_mean | 0.047750 | 0.154022 | — |
| wgbs | 5 | v1_prior_logit | 0.036899 | 0.130854 | 0.396691 |
| wgbs | 5 | v1_prior_residual | 0.037076 | 0.130551 | 0.398088 |
| wgbs | 5 | v3_logit | 0.037250 | 0.132271 | 0.395086 |
| wgbs | 5 | v3_population | 0.047267 | 0.150377 | — |
| wgbs | 5 | v3_residual | 0.037366 | 0.131974 | 0.395491 |
| wgbs | 10 | support_mean | 0.044900 | 0.136826 | — |
| wgbs | 10 | support_shrunk_mean | 0.044418 | 0.145059 | — |
| wgbs | 10 | v1_prior_logit | 0.034792 | 0.124883 | 0.396794 |
| wgbs | 10 | v1_prior_residual | 0.035017 | 0.125490 | 0.398216 |
| wgbs | 10 | v3_logit | 0.035111 | 0.126114 | 0.394962 |
| wgbs | 10 | v3_population | 0.044239 | 0.142759 | — |
| wgbs | 10 | v3_residual | 0.035254 | 0.126561 | 0.395532 |

## Quanto aggiunge v3 alla v1 già calibrata?

| view | support | mse_delta_percent | mae_delta | fraction_cpg_improved | ci95_genomic_blocks_low | ci95_genomic_blocks_high |
| --- | --- | --- | --- | --- | --- | --- |
| dev | 0 | -2.024595 | 0.000707 | 0.577795 | -0.000632 | -0.000346 |
| dev | 5 | -1.640116 | 0.000378 | 0.641239 | -0.000390 | -0.000196 |
| dev | 10 | -1.438815 | 0.000238 | 0.684794 | -0.000311 | -0.000171 |
| dev | 50 | -1.253642 | 0.000012 | 0.691843 | -0.000246 | -0.000139 |
| epic | 0 | 1.209273 | 0.001044 | 0.497006 | -0.000062 | 0.000544 |
| epic | 5 | 1.358145 | 0.000838 | 0.506287 | 0.000114 | 0.000312 |
| epic | 10 | 1.507274 | 0.000746 | 0.524850 | 0.000153 | 0.000300 |
| epic | 50 | 1.722613 | 0.000654 | 0.557784 | 0.000179 | 0.000294 |
| wgbs | 0 | 1.102554 | 0.001824 | 0.491592 | 0.000215 | 0.000763 |
| wgbs | 5 | 0.950799 | 0.001417 | 0.501813 | 0.000211 | 0.000467 |
| wgbs | 10 | 0.916928 | 0.001231 | 0.512034 | 0.000197 | 0.000420 |

Gli intervalli ricampionano blocchi genomici da 5 Mb, condizionatamente ai pazienti e agli insiemi di supporto osservati. Non sono intervalli congiunti pazienti-genoma. In particolare, 21 pazienti WGBS per test restano una limitazione sostanziale. Seed e ripetizioni di supporto non sono trattati come pazienti indipendenti.

## Incertezza congiunta sui pazienti esterni e sui blocchi genomici

Ricampionamento incrociato con 2.000 estrazioni: ogni paziente riceve lo stesso peso in tutte le ripetizioni in cui compare come query, evitando di contarle come osservazioni indipendenti. Le loss sono mediate sui seed, senza costruire un ensemble. Il supporto e gli adattamenti già stimati restano fissi: questi intervalli non includono tutta l’incertezza derivante da un nuovo campione di supporto. Le loss sono state ricomputate indipendentemente e verificate contro i risultati per CpG.

| view | support | mse_delta_percent | ci95_percent_low | ci95_percent_high |
| --- | --- | --- | --- | --- |
| epic | 0 | 1.209273 | -0.261434 | 2.841583 |
| epic | 5 | 1.358145 | 0.735073 | 2.227722 |
| epic | 10 | 1.507274 | 1.011662 | 2.229805 |
| epic | 50 | 1.722613 | 1.267813 | 2.396517 |
| wgbs | 0 | 1.102554 | 0.276358 | 2.136054 |
| wgbs | 5 | 0.950799 | 0.355985 | 1.587535 |
| wgbs | 10 | 0.916928 | 0.306804 | 1.557625 |
## Attribuzione degli input

Confronto originale appaiato, stesso training, budget 1.000 step, seed17. Le prime tre condizioni condividono le loss; `hybrid_wt` aggiunge la loss entro tessuto. Nessuna conclusione multi-seed sugli input. Aggiungere `AG_prior025` introduce informazioni AlphaGenome anche nel modello funzionale.

| arm | support | mse | mae | mas_pcc |
| --- | --- | --- | --- | --- |
| alphagenome | 0 | 0.026318 | 0.106234 | 0.524674 |
| alphagenome | 5 | 0.019445 | 0.089146 | 0.529235 |
| alphagenome | 50 | 0.016365 | 0.079967 | 0.532356 |
| functional | 0 | 0.033191 | 0.123536 | 0.444301 |
| functional | 5 | 0.022521 | 0.099334 | 0.450749 |
| functional | 50 | 0.018183 | 0.085768 | 0.454789 |
| hybrid | 0 | 0.026292 | 0.106350 | 0.521755 |
| hybrid | 5 | 0.019442 | 0.089247 | 0.528487 |
| hybrid | 50 | 0.016394 | 0.080084 | 0.531588 |
| hybrid_wt | 0 | 0.026288 | 0.106099 | 0.525080 |
| hybrid_wt | 5 | 0.019347 | 0.088755 | 0.530819 |
| hybrid_wt | 50 | 0.016310 | 0.079676 | 0.533602 |

## Attribuzione della testa e della regolarizzazione

Tre fold × tre seed, assi di training verificati uguali nei checkpoint. `mean − baseline` isola la testa della media superficiale; `local01 − mean` aggiunge la penalità sui vicini di peso 0,01; `local10` ne aumenta il peso. Le misure sono descrittive su fold già usati nello sviluppo.

| arm | blend | mse | mae | mas_pcc | within_tissue |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.000000 | 0.027531 | 0.107382 | 0.503784 | 0.338144 |
| baseline | 0.250000 | 0.026382 | 0.106021 | 0.504540 | 0.338291 |
| local01 | 0.000000 | 0.026583 | 0.107559 | 0.504307 | 0.338487 |
| local01 | 0.250000 | 0.025696 | 0.106698 | 0.504973 | 0.338609 |
| local10 | 0.000000 | 0.027371 | 0.112800 | 0.503813 | 0.336110 |
| local10 | 0.250000 | 0.026629 | 0.112380 | 0.504022 | 0.336166 |
| mean | 0.000000 | 0.027302 | 0.107879 | 0.502539 | 0.337480 |
| mean | 0.250000 | 0.026342 | 0.106749 | 0.503339 | 0.337617 |

## Confronto con MethylProphet

L’inventario locale trova il checkpoint originale ENCODE (24.337 input RNA), mentre il nostro asse canonico ne contiene 25.017. Il modello TCGA elencato nel MODEL_ZOO è allenato su Array+EPIC+WGBS: usarlo come baseline Array-only o come test esterno indipendente su queste coorti sarebbe scorretto. Non è stata prodotta una falsa comparazione numerica. Per una dichiarazione di superiorità rispetto al metodo pubblicato serve ancora un refit MethylProphet Array-only, con mapping RNA/tokenizzazione nativi verificati e identici assi di training, oltre alla stessa calibrazione sui campioni di supporto. Questa campagna non chiude quel confronto.

## Errori CpG e artefatti

Il CSV `*_cpg_errors.csv.gz` contiene per ogni locus, a k=5 e k=10, MSE/MAE v3, MSE v1, differenza, bias² ed errore centrato, mediati sulle predizioni dei seed e sulle ripetizioni (non un ensemble). `*_worst_cpgs.csv` riporta i 50 loci con maggiore MSE per vista e supporto. Le coordinate e gli identificatori provengono dal registro canonico.

[Figura](experiments/alphagenome_matched_support_2026-10-18.png) · [PDF](experiments/alphagenome_matched_support_2026-10-18.pdf) · [Riepilogo](experiments/alphagenome_matched_support_2026-10-18_summary.csv) · [Contrasti](experiments/alphagenome_matched_support_2026-10-18_contrasts.csv) · [Errori CpG](experiments/alphagenome_matched_support_2026-10-18_cpg_errors.csv.gz) · [CpG peggiori](experiments/alphagenome_matched_support_2026-10-18_worst_cpgs.csv).

Riproduzione: `python scripts/validate_alphagenome_matched_support.py`, `python scripts/validate_alphagenome_attribution.py`, `python scripts/validate_alphagenome_support_uncertainty.py`, quindi `python scripts/report_alphagenome_matched_support.py` con matplotlib disponibile. Protocolli, selezione interna, assi supporto/test, hash dei checkpoint e risultati per CpG in `artifacts/alphagenome_matched_support_2026-10-18/`.


Tre esempi degli errori maggiori per vista con 10 pazienti di supporto; RMSE e MAE sono in unità beta. I nomi Illumina e i geni sono annotazioni descrittive del manifest, senza attribuzione causale.

| view | illumina_probe_ids | chr | pos | gene_HGNC | v3_rmse | v3_mae | delta_mse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev | cg05393828 | chr1 | 22708306 |  | 0.338026 | 0.234028 | 0.001118 |
| dev | cg10400707 | chr1 | 160400050 | VANGL2 | 0.337847 | 0.292087 | 0.000741 |
| dev | cg14832490 | chr1 | 20631268 |  | 0.337417 | 0.220307 | 0.002541 |
| epic | cg15386853 | chr1 | 3390435 | PRDM16 | 0.561508 | 0.432343 | 0.006940 |
| epic | cg15617537 | chr1 | 198637545 | PTPRC | 0.457315 | 0.346803 | 0.002649 |
| epic | cg02030410 | chr1 | 245483623 | KIF26B | 0.457236 | 0.359124 | -0.014641 |
| wgbs | cg12407057 | chr1 | 1430826 | RP4-758J18.10 | 0.382883 | 0.318354 | 0.006190 |
| wgbs | cg01783650 | chr1 | 3394347 | PRDM16 | 0.379600 | 0.333007 | 0.003517 |
| wgbs | cg20810478 | chr1 | 247857330 | TRIM58 | 0.377738 | 0.309773 | -0.004257 |
