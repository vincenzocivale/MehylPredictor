# Stato della ricerca AlphaGenome — 18 ottobre 2026

Questo documento è il punto di ingresso per riprendere il lavoro, interpretare i risultati e preparare l’articolo. Riassume campagne distinte: i relativi numeri non vanno combinati come se provenissero dallo stesso esperimento. Per tabelle complete, protocolli e analisi degli errori fare riferimento ai report collegati.

## Risultato attuale e modello di riferimento

L’ultimo confronto completato è il riaddestramento **da zero su Array+EPIC+WGBS, chr1**, con feature AlphaGenome uniformi a **1.048.576 bp**, confrontato con le predizioni rilasciate di MethylProphet sugli stessi ID e target. Tre refit indipendenti, seed 17/18/19; i risultati sono medie delle metriche, non un ensemble.

| Vista Array | MSE MethylProphet | MSE nostro | Riduzione MSE | MAE nostro | Mediana PCC per CpG, riferimento → nostro |
| --- | ---: | ---: | ---: | ---: | ---: |
| CpG training / pazienti nuovi | 0,020902 | 0,012095 | 42,1% | 0,067714 | 0,542 → 0,669 |
| CpG escluse da Array training / pazienti training | 0,027571 | 0,016498 | 40,2% | 0,082433 | 0,410 → 0,612 |
| CpG escluse da Array training / pazienti nuovi | 0,028263 | 0,017309 | 38,8% | 0,084280 | 0,397 → 0,577 |

Nell’ultima vista la MAE del riferimento è 0,102559: riduzione del 17,8%. L’intervallo bootstrap al 95% della riduzione MSE è 36,3–41,6%. Il controllo a esposizione di coppie approssimativamente equivalente mantiene una riduzione MSE del 36,9%.

Questi risultati supportano la competitività sul benchmark chr1. Non dimostrano ancora superiorità su coorti indipendenti, tutti i cromosomi o per ogni CpG. Il contributo dei singoli componenti richiede ablation nello stesso nuovo protocollo.

**Report completo:** [benchmark appaiato](ALPHAGENOME_MIXED_METHYLPROPHET_2026-10-18.md). **Runbook e inferenza:** [istruzioni operative](ALPHAGENOME_REPRODUCIBILITY_2026-10-18.md).

## Correzione dello split e significato delle CpG nuove

Il vecchio protocollo derivato ricostruiva uno split con gli stessi conteggi, ma 723 pazienti validation differenti rispetto al rilascio. Il confronto attuale usa gli assi canonici `protocols/tcga_mix_chr1/array_{train,val}_{sample,cpg}_idx.npy`. Le CpG erano già coincidenti. Il primo pilot sullo split errato è stato fermato a 500 step e archiviato in `rejected_pre_alignment`, senza contribuire alla selezione o ai risultati ufficiali. Normalizzazione, grafo e prior sono stati ricostruiti.

I vecchi report rimangono validi sui rispettivi split; le loro metriche non sono confronti appaiati con il modello pubblicato. Inoltre usavano contesto AlphaGenome a 524.288 bp e prevalentemente training Array-only.

Nel nuovo esperimento:

- Array training: 8.260 pazienti e 33.885 CpG; test: 918 pazienti e 6.742 CpG.
- EPIC: 1.706 pazienti, 71.748 CpG; WGBS: 32 misure di 31 pazienti, 1.999.446 CpG eleggibili. La misura replicata WGBS è preservata.
- Nessun paziente Array test compare nelle fonti ausiliarie. La normalizzazione RNA usa solo i 9.997 pazienti training unici delle tre fonti.
- Delle 6.742 CpG escluse dall’Array training, **6.373 sono viste nelle fonti ausiliarie; soltanto 369 sono assenti da tutte le fonti training**.
- Su queste 369 CpG e pazienti nuovi: MSE 0,015072 contro 0,030131, circa −50%. È un sottoinsieme piccolo; non è stato calcolato un intervallo bootstrap dedicato.
- EPIC e WGBS sono ora fonti training: le analisi precedenti su tali coorti erano esterne soltanto rispetto ai modelli Array-only.

Le coppie finite training sono 275.082.367 Array + 115.856.100 EPIC + 63.982.272 WGBS = **454.920.739**. I 102 loci WGBS esclusi dal filtro sulle finestre contenenti N non vanno reintrodotti usando automaticamente tutti i candidati canonici chr1.

## Architettura e decisioni congelate

Famiglia checkpoint `alphagenome_hierarchical_chr1_v3`, 12.184.801 parametri neurali. Input: RNA canonica di 25.017 geni, embedding AlphaGenome `emb_1bp` di 1.536 dimensioni, 23 annotazioni dense e bag sparse delle 4.165 tracce funzionali.

La rete separa media del locus e risposta individuale: `beta(p,c) = sigmoid(b_c + u_p · a_c / sqrt(32))`. Encoder RNA a larghezza 256 e rango 32; testa della media superficiale e otto blocchi FFN CpG. La regolarizzazione di risposta locale ha peso 0,01: cinque vicini, distanti almeno 1 Mb, selezionati dopo PCA64 training-only e matching del contesto funzionale. Nel nuovo training il banco dei vicini è limitato a 65.536 loci, comprendendo tutte le CpG Array training e un campione delle altre. Questa approssimazione differisce dalla ricerca precedente su tutti i loci.

Obiettivo: MSE, PCC per locus (peso 0,15), media ausiliaria specifica per fonte (0,15), MSE entro tessuto, PCC entro tessuto (0,15), penalità del grafo (0,01). I termini entro tessuto richiedono tessuti nominati; i profili ausiliari con tessuto non disponibile non vi contribuiscono.

AdamW, learning rate 2e-4, weight decay 1e-4, gradient clipping 1, warmup 200, cosine horizon 8.000, rapporto minimo 0,1, EMA 0,99; training BF16 e valutazione FP32. Ogni batch sceglie una fonte e campiona con rimpiazzo fino a 256 righe × 512 CpG.

La selezione interna confronta `equal_source` e `finite_pairs` agli step 1.000/2.000/4.000/8.000 con seed17: 826 pazienti interni esclusi dai veri 8.260 training, e split genomico interno delle sole CpG Array training. Criterio: media dei rapporti MSE nelle viste CpG note e Array-held rispetto a `finite_pairs` a 1.000 step. Scelta: **finite_pairs, 8.000 step**. `equal_source` a 8.000 è leggermente migliore sulle CpG Array-held interne ma peggiore sulle note: non c’è un vincitore di ogni metrica.

La calibrazione usa Ridge alpha100 sulle medie delle sole CpG Array training. Per gli altri loci applica blend 0,25; per i loci Array noti usa offset stimati esclusivamente dai pazienti Array training e riferimenti RNA training fissi. Gli offset sono inclusi nei checkpoint. `fit_cpg_ids` contiene l’unione dei 2.004.067 loci training; `known_cpg_idx` identifica invece solo i 33.885 loci Array con offset specifico.

Il modello non calibrato raggiunge già MSE 0,017562 nella vista doppiamente esclusa. La calibrazione porta a 0,017309 ma peggiora leggermente la MAE: 0,083661 → 0,084280. Non attribuire tutti i guadagni alla calibrazione, né presentarla come miglioramento uniforme.

## Compute: cosa sappiamo e cosa non sappiamo

Il training principale ha campionato mediamente 891,6 milioni di coppie finite. Il riferimento configura circa 454,9 milioni di esposizioni (`55524 × 512 × 2 × 8`). Il supplemento sceglie lo step più vicino per esposizione tra quelli preregistrati, senza metriche test: 4.000 step, media 446,3 milioni. Il vantaggio MSE resta circa 38,6% / 38,1% / 36,9% nelle tre viste.

Questa è equivalenza approssimativa delle esposizioni a coppie, non dei FLOP, del tempo totale, del numero di esempi unici o del pretraining. Non è stato recuperato l’esatto rapporto del sampler originale degli shard. Le informazioni pretrained e il costo di estrazione AlphaGenome fanno parte del metodo.

L’ipotesi iniziale di compute adattivo per CpG **non è stata validata**. Nella vecchia campagna aumentare il training del modello profondo da 1.000 a 2.000 step migliorava le CpG note ma peggiorava quelle nuove; gran parte del peggioramento era bias della media. I gate che cercavano di predire il beneficio della profondità avevano AUROC circa 0,49–0,52 e non giustificavano instradamento dinamico. Questi risultati motivano la separazione media/risposta e controlli di calibrazione, non una rivendicazione di adaptive compute riuscito.

## Evidenze raccolte lungo il percorso

| Domanda | Risultato | Interpretazione consentita |
| --- | --- | --- |
| Gli embedding contengono struttura del locus? | Nei vecchi split, R² cross-locus circa 0,91 per media e 0,71 per deviazione standard | Forte informazione statica, non prova di risposta individuale |
| La RNA aggiunge informazione oltre al tessuto? | Shuffle RNA entro tessuto peggiora le predizioni | Segnale individuale; possono contribuire composizione cellulare, purezza e batch; nessuna prova causale |
| Retrieval, prototipi o continuazioni con componenti congelati risolvono il problema? | Nessun miglioramento stabile sotto i criteri bilanciati | Strade esplorate senza promozione |
| v3 è superiore a v1 calibrato a parità di supporto? | Nessuna superiorità uniforme; v1 calibrato migliore in alcune coorti esterne | Non usare i guadagni di calibrazione come prova isolata della nuova architettura |
| Poche misure di metilazione aiutano nuovi loci? | Shrinkage con tau5 aiuta; una sola misura senza shrinkage può peggiorare | Adattamento con etichette aggiuntive, distinto dallo zero-shot |
| Correzioni locali di sequenza aiutano? | 48 fit CNN/residuali/ibridi senza miglioramento richiesto; CNN non migliore del controllo shuffled | Risultato negativo della configurazione studiata, non inutilità generale della sequenza |
| Generalizza a tessuti esclusi? | Peggioramento su BRCA/LGG; calibrare la media non recupera la risposta | Limite di trasferimento biologico |
| Possiamo segnalare loci rischiosi? | Ranking di rischio con AUROC circa 0,77–0,83 nei vecchi controlli esterni | Ranking utile, non intervalli calibrati sul singolo paziente |
| Le CpG difficili sono tutte errori di coordinate o probe? | Audit non ha identificato una spiegazione unica | Le maschere disponibili non escludono ogni artefatto |

Le vecchie campagne hanno identificato 660 CpG persistentemente difficili su 6.742; questo elenco non definisce automaticamente i fallimenti del nuovo modello. Le associazioni biologiche e il pilot BRCA/LGG sono descrittivi, non un’analisi DMR o una dimostrazione causale.

## Errori del modello attuale e loro localizzazione

Il nuovo modello migliora la MSE media per CpG nell’88,84% dei loci noti, nel 66,85% dei loci Array-held con pazienti training e nel 66,38% con pazienti nuovi. Rimangono quindi molti loci per cui il riferimento è migliore. Nella vista doppiamente esclusa la frazione di errori assoluti beta >0,2 scende da 16,62% a 11,85%.

Le tabelle versionate sono in `docs/experiments/`, prefisso `alphagenome_mixed_2026-10-18`:

- `_cpg_improvement.csv` e `_worst_cpgs.csv`: riepiloghi e loci difficili.
- `_*_paired_cpg_errors.csv.gz`: errori appaiati per locus nelle singole viste.
- `_training_exposure_strata.csv`: separazione loci visti nelle fonti ausiliarie / mai visti.
- `_raw_and_calibrated.csv`, `_matched_exposure.csv`, `_paper_correlations.csv`: controlli del metodo e delle metriche.
- `_paired_contrasts.csv`: confronti con incertezza; figura `.png` e `.pdf` per la sintesi.

Per nuovi approfondimenti partire dagli ID globali, separare errore sulla media e sulla variabilità individuale, stratificare per esposizione training e verificare se il peggioramento è riproducibile nei tre seed. Non selezionare nuove ricette sulle CpG test che risultano difficili e poi riportare lo stesso test come conferma indipendente.

## Correttezza, incertezza e provenienza

Sono state allineate **91.859.614 coppie** rilasciate: zero duplicati e differenza target massima zero. Sono state riprodotte le correlazioni originali su entrambi gli assi; le differenze residue MSE/MAE sono inferiori a 1e-7 per aggregazione FP32 contro FP64. `mas_pcc` nel nuovo confronto è la mediana convenzionale delle correlazioni per CpG; il vecchio helper `torch.nanmedian` usa il valore centrale inferiore con numerosità pari, producendo piccole differenze.

Gli intervalli principali usano 2.000 bootstrap incrociati di pazienti e blocchi genomici di 5 Mb, sulle perdite medie dei tre seed. Condizionano sui modelli addestrati: non descrivono tutta l’incertezza dell’addestramento né la variabilità tra nuove coorti. L’arrotondamento BF16 degli output cambia trascurabilmente il risultato, ma non equivale a un forward interamente BF16.

I tre checkpoint sono stati ricaricati dall’adapter pubblico e confrontati con le matrici di benchmark su 16 pazienti × 64 loci, noti e held-out: differenza massima <8e-7. Suite registrata: 147 test passati e 2 skip per dati opzionali; quattro test helper mirati passati dopo gli aggiornamenti, parzialmente sovrapposti alla suite. Non sommare questi conteggi come test indipendenti.

## Contributo per l’articolo e lavoro ancora utile

Il contributo empirico è ora più solido: confronto appaiato a sorgenti di supervisione allineate, tre seed, controllo di esposizione, audit dei target e analisi separata delle CpG veramente nuove. La decomposizione gerarchica media/risposta, il vicinato funzionale e la calibrazione sono una proposta di metodo da valutare nel loro insieme. I singoli ingredienti non giustificano da soli rivendicazioni di novità; non abbiamo dimostrato adaptive compute, causalità biologica o ottimalità dell’architettura.

Priorità per rafforzare il lavoro:

1. Ablation a parità di split canonico, feature 1Mb, fonti e budget: funzionale-only, AlphaGenome-only, v1/v3, grafo e calibrazione. Le vecchie ablation non isolano i componenti nel nuovo setting.
2. Coorte indipendente mai usata in training e più loci assenti da tutte le fonti; EPIC/WGBS attuali non possono svolgere quel ruolo.
3. Curve di apprendimento e rango/capacità selezionati sul solo sviluppo interno, con costo di estrazione feature e training documentato.
4. Interpretazione riproducibile dei fallimenti e calibrazione di intervalli individuali, distinta dal ranking di rischio per locus.
5. Se si presenta il few-shot, confronto con identiche etichette di supporto anche per il riferimento attuale. Non è ancora completato nel nuovo benchmark.

## Mappa dei report

| Documento | Contenuto e ambito |
| --- | --- |
| [Capacità e struttura](ALPHAGENOME_CAPACITY_STRUCTURE_2026-10-18.md) | Compute, bias della media, struttura degli embedding |
| [Ricerca architetturale](ALPHAGENOME_ARCHITECTURE_SEARCH_2026-10-18.md) | Alternative esplorate e criteri di selezione |
| [v1](TCGA_CHR1_ALPHAGENOME_V1_2026-10-17.md) / [v2](TCGA_CHR1_ALPHAGENOME_V2_2026-10-18.md) | Modelli Array-only e sviluppo precedente |
| [Gerarchico v3](ALPHAGENOME_HIERARCHICAL_2026-10-18.md) | Calibrazione e checkpoint Array-only a 512kb |
| [Validazione ipotesi](ALPHAGENOME_HYPOTHESES_VALIDATION_2026-10-18.md) | 60 nuovi fit, trasferimento, sequence correction, rischio e QC |
| [Supporto appaiato](ALPHAGENOME_MATCHED_SUPPORT_2026-10-18.md) | Controlli v1/v3 con uguali etichette di supporto |
| [Benchmark attuale](ALPHAGENOME_MIXED_METHYLPROPHET_2026-10-18.md) | Nuovo training mixed a 1Mb e confronto rilasciato |
| [Riproducibilità](ALPHAGENOME_REPRODUCIBILITY_2026-10-18.md) | Checkpoint, inferenza, dipendenze, comandi e archivio |
