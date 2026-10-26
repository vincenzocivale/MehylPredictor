# Validazione delle ipotesi AlphaGenome — 18 ottobre 2026

Campagna completata: tre fold genomici interni, tre seed e conferma dev; **60 nuovi fit** (48 correzioni genomiche e 12 esclusioni di tessuto), controlli RNA, calibrazione con pochi campioni, affidabilità e conferme fuori da chr1. I risultati negativi sono inclusi. Nessun modello è promosso automaticamente.

## 1. Quanto segnale individuale contiene l’RNA?

Conferma limitata ai pazienti con tessuto esplicito: lo shuffle avviene esclusivamente entro il medesimo tessuto. I campioni NA sono esclusi da questa tabella; nel controllo originario erano un gruppo convenzionale e la loro permutazione poteva mescolare tessuti diversi. Media dei seed; per lo shuffle anche delle tre permutazioni.

| view | arm | patients | mse | mae | mas_pcc | within_tissue |
| --- | --- | --- | --- | --- | --- | --- |
| known | RNA corretta | 729 | 0.013493 | 0.071962 | 0.591327 | 0.385609 |
| known | RNA permutata entro tessuto | 729 | 0.019854 | 0.087730 | 0.328384 | -0.011002 |
| novel | RNA corretta | 729 | 0.024853 | 0.105522 | 0.531321 | 0.362465 |
| novel | RNA permutata entro tessuto | 729 | 0.030756 | 0.117625 | 0.293007 | -0.009365 |

Controlli sul dev completo; statistiche empiriche e prior Ridge stimati soltanto sui pazienti/loci training. Il baseline tessuto+genoma usa il tessuto osservato e non deve essere interpretato come RNA-only.

| view | source | arm | mse | mae | mas_pcc |
| --- | --- | --- | --- | --- | --- |
| known | control | empirical_mean | 0.022741 | 0.096982 | — |
| known | control | empirical_tissue | 0.015218 | 0.074417 | 0.502192 |
| known | control | genomic_mean | 0.031041 | 0.127177 | — |
| known | control | genomic_tissue | 0.026695 | 0.117509 | 0.371732 |
| known | v1_early | calibrated | 0.013840 | 0.072605 | 0.588091 |
| known | v1_early | raw | 0.017172 | 0.085222 | 0.584834 |
| known | v2_early | calibrated | 0.013852 | 0.072978 | 0.587821 |
| known | v2_early | raw | 0.017861 | 0.088560 | 0.584016 |
| novel | control | genomic_mean | 0.033423 | 0.132107 | — |
| novel | control | genomic_tissue | 0.029748 | 0.124106 | 0.339727 |
| novel | v1_early | prior0.25 | 0.025577 | 0.105461 | 0.527395 |
| novel | v1_early | raw | 0.026422 | 0.106299 | 0.526757 |
| novel | v2_early | prior0.25 | 0.025059 | 0.106168 | 0.527678 |
| novel | v2_early | raw | 0.025633 | 0.106514 | 0.527197 |

La calibrazione della v1 è competitiva con quella della v2/v3 sulle CpG note: il miglioramento rispetto a v1 non calibrata non identifica un vantaggio specifico della nuova rappresentazione. Lo shuffle e i controlli di tessuto verificano invece l’informazione individuale RNA. Una variazione della correlazione dopo la calibrazione è possibile per la sigmoid, anche a fattori fissi.

## 2. Da zero osservazioni a pochi campioni di supporto

Il modello resta congelato. Per CpG mai usate nel fit, 1/5/10/50 pazienti training forniscono metilazione di supporto; gli altri pazienti, esclusi dal fit e dal supporto, misurano l’errore. Cinque insiemi di supporto annidati, tre seed. Il peso empirico è n/(n+τ), n numero di osservazioni finite per locus. **τ=5** selezionato sui tre fold interni, senza ottimizzazione sul dev. Con nessuna osservazione il prior è mantenuto; livelli desiderati limitati a [0,001;0,999].

| support | mse | mae | mas_pcc | delta_mse_percent |
| --- | --- | --- | --- | --- |
| 0 | 0.025059 | 0.106168 | 0.527678 | +0.00 |
| 1 | 0.022668 | 0.100015 | 0.528989 | -9.54 |
| 5 | 0.018913 | 0.089303 | 0.531660 | -24.53 |
| 10 | 0.017609 | 0.085302 | 0.532626 | -29.73 |
| 50 | 0.016150 | 0.079936 | 0.534183 | -35.55 |

L’adattamento senza shrinkage con un solo campione può peggiorare l’errore. Questo esperimento cambia il problema applicativo: richiede metilazione dei nuovi loci nei campioni di supporto. Non è un miglioramento zero-shot senza metilazione in ingresso. Il supporto qui proviene dalla popolazione Array; non dimostra ancora adattamento few-shot a EPIC/WGBS.

## 3. CpG persistentemente difficili

Audit descrittivo delle 6.742 CpG ufficiali nuove, di cui **660** difficili in almeno due seed. Il manifest pubblico HM450 mappa esattamente 6737 loci alle coordinate hg38; tutte le 6.742 posizioni corrispondono a CG nel riferimento. **660 delle 660 CpG difficili** non sono segnalate da MASK_general; mapping, non unicità della sequenza e SNP comuni nei 5bp terminali non identificano un sottogruppo problematico nel pannello già selezionato. Questo non esclude altri artefatti sperimentali.

Controlli abbinati per contesto, regione, media e SD osservate sono esclusivamente descrittivi. Un grande scarto di media (>0,3 beta) rispetto al vicino AlphaGenome training distante almeno 1Mb riguarda il 8.9% dei loci e il 29.2% di quelli difficili. La similarità coseno mediana del vicino è 0.630: non si tratta di embedding indistinguibili, né il confronto dimostra causalità. Il risultato indica limiti del trasferimento del livello medio.

Fonti: [manifest HM450](https://zhouserver.research.chop.edu/InfiniumAnnotation/current/HM450/HM450.hg38.manifest.tsv.gz), [documentazione delle maschere](https://zwdzwd.github.io/InfiniumAnnotation/manifest.html). SHA256 del manifest: `668a11ea624b3e645aab963dca048bdd4628ae5d297732867c91525c831cf191`. Annotazioni complete, controlli abbinati e vicini sono negli artefatti `hard_cpgs/`.

## 4. Regolarizzazione adattiva e affidabilità

Random forest con iperparametri prefissati, addestrata sugli errori delle altre regioni genomiche. Nessuna CpG valutata è presente nei target di fit della regola. Le feature sono annotazioni e statistiche delle predizioni: nessuna metilazione del locus interrogato. I fold interni condividono pazienti di valutazione, mentre i pazienti dev sono distinti; il dev è quindi il controllo più informativo sul trasferimento.

Sul dev la selezione per locus fra pesi 0/0,25/0,5 passa da MSE 0.025059 a **0.024898** (-0.65%). MAE e correlazioni vanno considerate insieme: minimizzare la MSE non garantisce una dominanza sulle altre metriche. La regola adattiva resta sperimentale.

Affidabilità sul dev, mantenendo il 50% delle CpG ritenute più sicure:

| estimator | mse | error_prediction_spearman | hard_decile_auc |
| --- | --- | --- | --- |
| learned | 0.010827 | 0.814445 | 0.810697 |
| seed_disagreement | 0.011922 | 0.757990 | 0.764930 |
| predicted_sd | 0.011186 | 0.775335 | 0.750002 |
| prior_midpoint | 0.011279 | 0.766812 | 0.796628 |

Conferma della stessa regola, senza rifit su etichette esterne:

| view | estimator | mse | hard_decile_auc |
| --- | --- | --- | --- |
| chr2 | learned | 0.009870 | 0.828808 |
| chr2 | seed_disagreement | 0.011098 | 0.789318 |
| chr2 | predicted_sd | 0.010782 | 0.752348 |
| chr2 | prior_midpoint | 0.010370 | 0.816847 |
| chr22 | learned | 0.007105 | 0.832893 |
| chr22 | seed_disagreement | 0.007726 | 0.800871 |
| chr22 | predicted_sd | 0.007633 | 0.768532 |
| chr22 | prior_midpoint | 0.007439 | 0.813450 |
| epic_novel | learned | 0.005610 | 0.832733 |
| epic_novel | seed_disagreement | 0.005826 | 0.826011 |
| epic_novel | predicted_sd | 0.005600 | 0.801117 |
| epic_novel | prior_midpoint | 0.006857 | 0.830947 |
| wgbs_novel | learned | 0.017464 | 0.774491 |
| wgbs_novel | seed_disagreement | 0.018792 | 0.775578 |
| wgbs_novel | predicted_sd | 0.017709 | 0.734236 |
| wgbs_novel | prior_midpoint | 0.018529 | 0.754941 |

Il baseline distanza della media da 0,5 è importante: selezionare CpG poco variabili è già relativamente facile. Questi risultati descrivono rischio medio per locus nella popolazione RNA interrogata; non sono intervalli predittivi calibrati per il singolo paziente. Si riportano tutti i controlli, compresi quelli che eguagliano o superano la regola appresa sotto shift.

## 5. Sequenza locale, nuovi cromosomi, tessuti e piattaforme

Correzioni addestrate da zero sopra v2 congelata: MLP AlphaGenome, CNN DNA di 1.002bp centrata sul dinucleotide, combinazione, e sequenze permutate come controllo negativo. La CNN media i due orientamenti reverse complement. Si correggono intercetta e fattori RNA con 1.000 step prefissati; il peso del prior sulle nuove CpG resta 0,25. I rami hanno capacità diverse, esplicitate nella tabella; non è un confronto a uguale numero di parametri. Tutti gli encoder RNA originali restano congelati.

| arm | parameters | mse | mae | mas_pcc | within_tissue | delta_mse_percent | novel_guards |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ag | 54401 | 0.025026 | 0.106682 | 0.526127 | 0.361985 | -0.13 | False |
| hybrid | 58489 | 0.025028 | 0.106689 | 0.526194 | 0.361996 | -0.13 | False |
| sequence | 4121 | 0.025054 | 0.106355 | 0.526908 | 0.361862 | -0.02 | False |
| shuffled_sequence | 4121 | 0.025052 | 0.106260 | 0.527069 | 0.361847 | -0.03 | False |

Le sequenze permutate permettono di distinguere il contributo della sequenza da una semplice correzione aggiuntiva. La CNN allineata non supera il controllo con sequenze permutate; nessuno dei rami supera i criteri sulle nuove CpG. Il risultato è negativo per questa implementazione. Questo test riguarda una piccola CNN locale su backbone congelato; non esaurisce finestre più ampie o riaddestramento congiunto. `novel_guards` richiede almeno −0,5% MSE senza peggiorare MAE, e PCC/within-tissue entro −0,002; non comprende i controlli sulle CpG note, non valutate per questi rami.

Conferma su **tutti** i loci Array di chr2 e chr22, mai usati dal nostro predittore chr1, e su pazienti EPIC/WGBS senza sovrapposizione di ID con Array. Confronto primario seed17; v2/v3 disponibili anche per seed18/19. Le fonti Array/EPIC/WGBS provengono dal pacchetto canonico MethylProphet; EPIC include campioni con prefissi C3L/C3N, HCM, HTMCP e TARGET, WGBS campioni TCGA. Il solo ID disgiunto non certifica assenza di ogni possibile dipendenza di studio. EPIC/WGBS non hanno tessuti canonici annotati, quindi la correlazione entro tessuto non è stimabile.

| view | model | patients | cpgs | mse | mae | mas_pcc | delta_mse_percent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chr2 | v1 | 918 | 30288 | 0.025207 | 0.102884 | 0.543058 | +0.00 |
| chr2 | v3 | 918 | 30288 | 0.024452 | 0.104425 | 0.543500 | -3.00 |
| chr22 | v1 | 918 | 7180 | 0.022373 | 0.095562 | 0.540272 | +0.00 |
| chr22 | v3 | 918 | 7180 | 0.021374 | 0.095990 | 0.542731 | -4.46 |
| epic_known | v1 | 1706 | 33885 | 0.020924 | 0.094315 | 0.403739 | +0.00 |
| epic_known | v3 | 1706 | 33885 | 0.017588 | 0.083526 | 0.401368 | -15.94 |
| epic_novel | v1 | 1706 | 3340 | 0.020381 | 0.088264 | 0.345808 | +0.00 |
| epic_novel | v3 | 1706 | 3340 | 0.020135 | 0.090149 | 0.342337 | -1.21 |
| wgbs_known | v1 | 31 | 33885 | 0.040350 | 0.140792 | 0.405674 | +0.00 |
| wgbs_known | v3 | 31 | 33885 | 0.038636 | 0.140666 | 0.403479 | -4.25 |
| wgbs_novel | v1 | 31 | 3033 | 0.044325 | 0.146164 | 0.386819 | +0.00 |
| wgbs_novel | v3 | 31 | 3033 | 0.044546 | 0.149686 | 0.381755 | +0.50 |

Loci noti e nuovi sono definiti rispetto al training del modello chr1. La valutazione EPIC/WGBS è limitata alle CpG comuni al pannello Array chr1: non dimostra copertura dei milioni di loci esclusivi WGBS. AlphaGenome può aver visto queste sequenze nel pretraining di altri task: i cromosomi sono nuovi per il nostro fit della metilazione, non necessariamente per AlphaGenome. I risultati di conferma non hanno modificato le ricette.

WGBS contiene 32 misure e 31 pazienti distinti. Due misure dello stesso paziente sono mediate per CpG, ignorando i valori mancanti, prima delle metriche; il paziente mantiene peso uno. La correzione del lettore è tracciata in `transfer_protocol_amendment.json`, dopo il primo accesso a WGBS; le valutazioni chr2/chr22/EPIC già completate non sono cambiate.

Esclusione completa di BRCA o LGG dal training, contro rimozione casuale dello stesso numero di pazienti; stessi loci e stessi pazienti di test. Tre seed, dodici fit nuovi. Si riporta la popolazione del tessuto escluso, con calibrazione gerarchica.

| group | view | mode | patients | train_patients | mse | mae | mas_pcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TCGA-BRCA | known | leave_tissue_out | 70 | 6804 | 0.015649 | 0.077777 | 0.289090 |
| TCGA-BRCA | known | matched_random | 70 | 6804 | 0.013098 | 0.070935 | 0.346946 |
| TCGA-BRCA | novel | leave_tissue_out | 70 | 6804 | 0.025599 | 0.107696 | 0.274262 |
| TCGA-BRCA | novel | matched_random | 70 | 6804 | 0.024330 | 0.105437 | 0.315277 |
| TCGA-LGG | known | leave_tissue_out | 46 | 7021 | 0.021420 | 0.092383 | 0.356503 |
| TCGA-LGG | known | matched_random | 46 | 7021 | 0.014919 | 0.074361 | 0.405428 |
| TCGA-LGG | novel | leave_tissue_out | 46 | 7021 | 0.030502 | 0.115390 | 0.343844 |
| TCGA-LGG | novel | matched_random | 46 | 7021 | 0.028048 | 0.105643 | 0.383455 |

Differenze BRCA–LGG sul dev: contrasto definito dai due gruppi più numerosi nel training, quindi indipendente dai target di valutazione. Si misurano effetti per CpG, non regioni DMR né effetti causali.

| arm | effect_mse | effect_corr | large_effect_cpgs | direction_accuracy |
| --- | --- | --- | --- | --- |
| v3_seed17 | 0.015919 | 0.644346 | 1490 | 0.884564 |
| genomic_tissue | 0.016759 | 0.622956 | 1490 | 0.869799 |

## Interpretazione e limiti

I controlli permettono di distinguere informazione individuale RNA, calibrazione del livello medio e trasferimento del contesto genomico. Il segnale RNA entro tessuto può ancora includere purezza tumorale, composizione cellulare e batch: non dimostra una risposta intrinseca delle singole cellule. L’adattamento con pochi campioni e la stima del rischio sono risultati da sviluppare; l’aumento indiscriminato della capacità o la sola selezione della MSE non risolvono tutti i compromessi. I confronti calibrati sono necessari per attribuire correttamente i guadagni architetturali.

La v3 trasferisce il vantaggio di MSE a chr2 e chr22, ma non domina sulle altre metriche; sulle CpG nuove WGBS non supera v1. L’esclusione di un tessuto peggiora la risposta anche con lo stesso numero di pazienti training: calibrare il livello medio non risolve il trasferimento dello stato biologico. La regola adattiva del prior migliora lievemente MSE su Array, ma non risolve MAE e non dà un vantaggio affidabile su EPIC/WGBS. Nessuna di queste varianti è una sostituzione universale della ricetta esistente.

Fold e dev sono già stati usati nello sviluppo. L’embargo interno di 300kb non rende completamente disgiunte tutte le finestre AlphaGenome da 524kb; la conferma su cromosomi diversi riduce questa ambiguità per il fit del nostro decoder. Gli intervalli bootstrap in JSON ricampionano blocchi genomici di 5Mb e sono condizionati ai pazienti osservati; non sostituiscono una replica con pazienti/studi indipendenti. Tre seed non rappresentano tre coorti. Le maschere di qualità consultate non coprono tutti i fattori sperimentali. Nessuna interpretazione biologica causale è dimostrata.

## Riproducibilità

Script: `validate_alphagenome_hypotheses.py`, `validate_alphagenome_named_rna.py`, `audit_alphagenome_hard_cpgs.py`, `validate_alphagenome_sequence.py`, `validate_alphagenome_tissue.py`, `validate_alphagenome_transfer.py`, `validate_alphagenome_reliability.py`; rendering `report_alphagenome_hypotheses.py`. Tutti in `scripts/`. Protocolli, assi, checkpoint, per-CpG e hash in `artifacts/alphagenome_hypotheses_2026-10-18/`.

[Figura](experiments/alphagenome_hypotheses_2026-10-18.png) · [PDF](experiments/alphagenome_hypotheses_2026-10-18.pdf) · [Metriche complete](experiments/alphagenome_hypotheses_2026-10-18.json). La suite di test e i controlli finali sono tracciati in `verification.json` negli artefatti.
