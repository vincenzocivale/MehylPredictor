# AlphaGenome Array+EPIC+WGBS vs MethylProphet — 18 ottobre 2026

Confronto del nostro modello riaddestrato da zero sui tre dataset con le predizioni del modello pubblicato, appaiate per paziente, CpG e target. La ricetta è stata congelata sullo sviluppo interno prima di calcolare le metriche ufficiali o quelle di MethylProphet.

## Correzione dello split prima del confronto

L’audit sulle predizioni rilasciate ha rilevato che il precedente protocollo derivato `methylprophet_table5_tcga_chr1/table5_protocol` ricostruiva un diverso split di pazienti: 723 pazienti validation differenti, nonostante gli stessi conteggi 8.260/918. Gli assi canonici `protocols/tcga_mix_chr1` coincidono invece con il rilascio. Il primo fit esplorativo è stato interrotto dopo 500 aggiornamenti, archiviato in `rejected_pre_alignment` ed escluso dalla selezione. Anche prior, grafo e normalizzazione sono stati ricostruiti. I vecchi risultati del progetto conservano il loro significato sui propri split, ma non sono confronti appaiati con questo modello pubblicato.

## Dati e rappresentazione

Array: 8.260 pazienti × 33.885 CpG training; 918 pazienti e 6.742 CpG esclusi dall’Array training. Tutti i 1.706 profili EPIC e le 32 misure WGBS di 31 pazienti sono usati nel training, con 71.748 e 1.999.446 CpG rispettivamente. Il filtro nativo sulle finestre di sequenza esclude 102 loci WGBS contenenti N. Le misure replicate WGBS sono preservate nel training per rispettare il dataset; non diventano pazienti indipendenti negli intervalli di valutazione Array.

Training finito: **275.082.367 Array + 115.856.100 EPIC + 63.982.272 WGBS = 454.920.739 coppie**. Nessun paziente Array validation compare nelle fonti ausiliarie. Tutti i 25.017 geni canonici sono disponibili; il nostro modello usa RNA standardizzata sui soli pazienti training, MethylProphet usa la propria quantizzazione nativa. Il numero di coppie disponibili non equivale al numero di coppie campionate durante l’ottimizzazione.

Cache AlphaGenome uniforme: `../data/alphagenome_cpg_embeddings/chr1.h5`, `emb_1bp`, 1.536 dimensioni, contesto 1.048.576 bp. Copertura completa degli assi di training; ID e coordinate confrontati con i tre registri canonici, zero discrepanze. Nessuna fusione con i vecchi embedding a 524.288 bp. Il modello è riaddestrato da zero. Le informazioni pretrained di AlphaGenome e le annotazioni funzionali sono parte del metodo: stesso dataset di metilazione non significa stesso pretraining o stesso costo totale.

## Selezione e architettura

Testa della media superficiale, otto FFN CpG, rango 32, un encoder RNA e regolarizzazione locale di peso 0,01. Per scalare a due milioni di loci, i cinque vicini distanti almeno 1 Mb sono cercati in un banco fisso di al massimo 65.536 loci training, con tutte le CpG Array training e un campione delle altre, dopo PCA64 training-only e matching delle annotazioni. Il grafo è quindi un’approssimazione esplicita della precedente ricerca su tutti i loci; gli assi di supervisione restano completi.

Due politiche di campionamento delle fonti: uguale peso oppure proporzionale alle coppie finite disponibili. Il rapporto originale di costruzione offline degli shard non è stato ricostruito; non si dichiara identità di distribuzione dei minibatch. Seed17 per la selezione, tre refit indipendenti seed17/18/19. Pazienti interni generati soltanto entro il vero training pubblicato, con vecchi blocchi genomici interni; scelta per media dei rapporti MSE delle viste CpG note e nuove rispetto all’Array. **Scelta: finite_pairs, 8000 step**.

Calibrazione v3: prior Ridge100 sulle medie delle sole CpG Array training, blend 0,25 per gli altri loci; offset empirici esclusivamente sulle CpG Array training usando i pazienti Array training. Nessun target Array escluso dal training entra nella calibrazione. I checkpoint finali includono gli offset e sono ricaricabili dall’adapter pubblico.

Checkpoint principale: `artifacts/alphagenome_mixed_2026-10-18/official/finite_pairs/seed17/checkpoint.pt`; repliche nelle cartelle `seed18` e `seed19`. Richiede feature AlphaGenome native `emb_1bp` a 1 Mb e gli ID CpG globali, oltre alle 23 annotazioni e al contratto delle 4.165 tracce funzionali. La cache Array precedente a 512 kb non è intercambiabile.

| policy | step | known_mse | novel_array_mse | known_mae | novel_array_mae | score |
| --- | --- | --- | --- | --- | --- | --- |
| equal_source | 1000 | 0.015680 | 0.024361 | 0.078172 | 0.105335 | 1.016588 |
| equal_source | 2000 | 0.014631 | 0.022729 | 0.075194 | 0.101459 | 0.948536 |
| equal_source | 4000 | 0.013520 | 0.020904 | 0.071924 | 0.097020 | 0.874458 |
| equal_source | 8000 | 0.012800 | 0.019335 | 0.069699 | 0.092829 | 0.818508 |
| finite_pairs | 1000 | 0.015248 | 0.024243 | 0.076956 | 0.105164 | 1.000000 |
| finite_pairs | 2000 | 0.014027 | 0.022853 | 0.073483 | 0.101747 | 0.931290 |
| finite_pairs | 4000 | 0.012741 | 0.020921 | 0.069544 | 0.096691 | 0.849286 |
| finite_pairs | 8000 | 0.012037 | 0.019539 | 0.067347 | 0.093075 | 0.797662 |

## Risultati appaiati sulle tre viste pubblicate

| view | model | mse | mae | mas_pcc | within_tissue | severe |
| --- | --- | --- | --- | --- | --- | --- |
| train_cpg-val_sample | AlphaGenome_mixed_raw | 0.013034 | 0.074182 | 0.669381 | 0.432285 | 0.085982 |
| train_cpg-val_sample | AlphaGenome_mixed_v3 | 0.012095 | 0.067714 | 0.668663 | 0.432302 | 0.080751 |
| train_cpg-val_sample | AlphaGenome_mixed_v3_output_BF16 | 0.012096 | 0.067719 | 0.668631 | 0.432113 | 0.080754 |
| train_cpg-val_sample | MethylProphet_released | 0.020902 | 0.089509 | 0.541904 | 0.324785 | 0.132808 |
| val_cpg-train_sample | AlphaGenome_mixed_raw | 0.016747 | 0.081832 | 0.612415 | 0.484980 | 0.113923 |
| val_cpg-train_sample | AlphaGenome_mixed_v3 | 0.016498 | 0.082433 | 0.612120 | 0.484794 | 0.112954 |
| val_cpg-train_sample | AlphaGenome_mixed_v3_output_BF16 | 0.016499 | 0.082436 | 0.612103 | 0.484710 | 0.112956 |
| val_cpg-train_sample | MethylProphet_released | 0.027571 | 0.101290 | 0.409791 | 0.262784 | 0.162973 |
| val_cpg-val_sample | AlphaGenome_mixed_raw | 0.017562 | 0.083661 | 0.576797 | 0.352803 | 0.119273 |
| val_cpg-val_sample | AlphaGenome_mixed_v3 | 0.017309 | 0.084280 | 0.577351 | 0.352872 | 0.118523 |
| val_cpg-val_sample | AlphaGenome_mixed_v3_output_BF16 | 0.017310 | 0.084282 | 0.577366 | 0.352698 | 0.118525 |
| val_cpg-val_sample | MethylProphet_released | 0.028263 | 0.102559 | 0.396722 | 0.231986 | 0.166222 |

`mas_pcc` è la mediana delle correlazioni entro CpG attraverso i pazienti; le medie sui tre seed sono medie delle metriche, non un ensemble. MSE e MAE usano gli stessi target finiti per entrambi i metodi. Le predizioni pubblicate sono valori di inferenza BF16; le nostre predizioni vengono calcolate FP32 per la valutazione. `output_BF16` è un controllo dell’arrotondamento finale dei nostri output: non equivale a rifare tutto il forward BF16. `mixed_raw` mostra il modello prima della calibrazione, per non attribuire alla rete un guadagno dovuto agli offset.

Le aggregazioni originali sono riportate anche su entrambi gli assi. Le sei metriche ricalcolate di MethylProphet (MSE, MAE, media/mediana PCC per CpG e per paziente) sono verificate contro i JSON degli autori entro tolleranza assoluta 2e-5.

| view | model | cpg_pcc_mean | mas_pcc | sample_pcc_mean | sample_pcc_median |
| --- | --- | --- | --- | --- | --- |
| train_cpg-val_sample | AlphaGenome_mixed_v3 | 0.596289 | 0.668663 | 0.954378 | 0.958843 |
| train_cpg-val_sample | MethylProphet_released | 0.470791 | 0.541904 | 0.920649 | 0.927351 |
| val_cpg-train_sample | AlphaGenome_mixed_v3 | 0.568783 | 0.612120 | 0.936801 | 0.941538 |
| val_cpg-train_sample | MethylProphet_released | 0.377453 | 0.409791 | 0.894388 | 0.902707 |
| val_cpg-val_sample | AlphaGenome_mixed_v3 | 0.499637 | 0.577351 | 0.934598 | 0.939233 |
| val_cpg-val_sample | MethylProphet_released | 0.363635 | 0.396722 | 0.892621 | 0.900976 |

## Differenze e incertezza

Segno negativo: MSE del nostro modello inferiore a MethylProphet. Bootstrap appaiato con 2.000 ricampionamenti indipendenti di pazienti e blocchi genomici da 5 Mb. Gli intervalli sono condizionati ai modelli stimati e non comprendono tutta la variabilità di training. Le loss dei tre seed sono mediate senza trattarli come pazienti indipendenti.

| view | mse_delta_percent | ci95_percent_low | ci95_percent_high |
| --- | --- | --- | --- |
| train_cpg-val_sample | -42.135094 | -43.758450 | -40.410084 |
| val_cpg-train_sample | -40.160781 | -42.922490 | -37.931602 |
| val_cpg-val_sample | -38.755831 | -41.560038 | -36.328508 |

| view | loci | fraction_improved | median_delta_mse |
| --- | --- | --- | --- |
| train_cpg-val_sample | 33885 | 0.888387 | -0.002831 |
| val_cpg-train_sample | 6742 | 0.668496 | -0.001254 |
| val_cpg-val_sample | 6742 | 0.663750 | -0.001196 |

## Quali CpG sono davvero nuove?

Nel protocollo misto **6.373 delle 6.742 CpG escluse dall’Array sono osservate in EPIC/WGBS; solo 369 sono escluse da tutte le fonti training**. Questo è mantenuto identico per entrambi i metodi. Il pannello Array-held nel suo insieme non viene chiamato zero-shot genomico; il sottoinsieme di 369 loci viene riportato separatamente, con la cautela dovuta alle dimensioni ridotte.

| view | scope | loci | our_mse | reference_mse | delta_mse_percent | our_mae | reference_mae | our_pcc | reference_pcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_cpg-val_sample | all | 33885.000000 | 0.012095 | 0.020902 | -42.135094 | 0.067714 | 0.089509 | 0.668663 | 0.541904 |
| train_cpg-val_sample | seen_auxiliary | 33885.000000 | 0.012095 | 0.020902 | -42.135094 | 0.067714 | 0.089509 | 0.668663 | 0.541904 |
| val_cpg-train_sample | all | 6742.000000 | 0.016498 | 0.027571 | -40.160781 | 0.082433 | 0.101290 | 0.612120 | 0.409791 |
| val_cpg-train_sample | seen_auxiliary | 6373.000000 | 0.016623 | 0.027458 | -39.461639 | 0.082909 | 0.101377 | 0.612677 | 0.410848 |
| val_cpg-train_sample | unseen_all_training | 369.000000 | 0.014359 | 0.029509 | -51.342235 | 0.074262 | 0.099805 | 0.603734 | 0.394595 |
| val_cpg-val_sample | all | 6742.000000 | 0.017309 | 0.028263 | -38.755831 | 0.084280 | 0.102559 | 0.577351 | 0.396722 |
| val_cpg-val_sample | seen_auxiliary | 6373.000000 | 0.017439 | 0.028154 | -38.057152 | 0.084773 | 0.102660 | 0.577966 | 0.397374 |
| val_cpg-val_sample | unseen_all_training | 369.000000 | 0.015072 | 0.030131 | -49.977678 | 0.075816 | 0.100825 | 0.565317 | 0.382512 |

## Controllo con esposizione alle coppie simile al modello pubblicato

La configurazione originale specifica 55.524 aggiornamenti, batch locale 512 e 16 GPU, cioè **454.852.608 esposizioni a coppie**. Prima della selezione primaria e di qualsiasi scoring pubblicato è stata fissata una regola secondaria: scegliere fra i checkpoint già previsti quello con numero medio effettivo di coppie finite campionate più vicino a questo valore. Nessuna metrica test sceglie questo checkpoint.

Controllo: **4000 step, 446,260,833 coppie finite medie** sui tre seed. È un confronto di esposizione approssimata, non di FLOPs, tempo o costo del pretraining. I risultati primari restano quelli della ricetta selezionata internamente; il controllo verifica quanto dipendano da un numero diverso di coppie campionate.

| view | reference_mse | mse | delta_mse_percent | mae | mas_pcc |
| --- | --- | --- | --- | --- | --- |
| train_cpg-val_sample | 0.020902 | 0.012827 | -38.632348 | 0.070005 | 0.645124 |
| val_cpg-train_sample | 0.027571 | 0.017077 | -38.060727 | 0.083991 | 0.602646 |
| val_cpg-val_sample | 0.028263 | 0.017842 | -36.869951 | 0.085705 | 0.568941 |

## Riproduzione e artefatti

Le predizioni ufficiali provengono da [MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2](https://huggingface.co/datasets/MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2); la configurazione del modello da [MethylProphet/tcga_mix_chr1-bs_512-c2b2](https://huggingface.co/MethylProphet/tcga_mix_chr1-bs_512-c2b2). Revisione dataset: `c3a4e7e7f5cbfd1373aadc155d6bef7e1e6dcd9e`. Hash e file originali nel sottoalbero `reference/`. L’audit verifica tutti gli ID, tutti i target e le maschere finite, senza selezionare sulla performance pubblicata.

[Figura](experiments/alphagenome_mixed_2026-10-18.png) · [PDF](experiments/alphagenome_mixed_2026-10-18.pdf) · [Metriche](experiments/alphagenome_mixed_2026-10-18_metrics.csv) · [Contrasti appaiati](experiments/alphagenome_mixed_2026-10-18_paired_contrasts.csv) · [Esposizione delle CpG](experiments/alphagenome_mixed_2026-10-18_training_exposure_strata.csv) · [Errori maggiori](experiments/alphagenome_mixed_2026-10-18_worst_cpgs.csv).

Comandi: `python scripts/fetch_methylprophet_reference.py` (con accesso Hugging Face già configurato per i repository), `python scripts/train_alphagenome_mixed.py --stage all`, `python scripts/compare_alphagenome_methylprophet.py --stage score`, `python scripts/evaluate_alphagenome_mixed_budget.py`, `python scripts/report_alphagenome_mixed.py` (matplotlib disponibile per il report). Protocolli e snapshot in `artifacts/alphagenome_mixed_2026-10-18/`. Il confronto pubblicato qui riguarda il rilascio chr1; non dimostra superiorità su altri cromosomi o su EPIC/WGBS indipendenti, ora usati nel training.

La famiglia architetturale è stata sviluppata in precedenza su altri split di questo stesso dataset Array. Il refit e la selezione di questa campagna escludono i pazienti test corretti, ma il dataset non è una coorte mai esaminata durante la ricerca. Per una conferma indipendente servirà una nuova coorte.
