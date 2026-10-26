# Prima v1 RNA → metilazione TCGA chr1 con AlphaGenome

> **Allineamento del benchmark, 18 ottobre 2026:** l’audit sulle predizioni rilasciate di MethylProphet ha trovato 723 pazienti validation diversi nello split derivato usato da questa campagna. I risultati sotto restano riferiti agli assi qui descritti; il confronto appaiato con il modello pubblicato usa invece lo split canonico verificato su tutti gli ID e target. Vedere il [nuovo benchmark misto](ALPHAGENOME_MIXED_METHYLPROPHET_2026-10-18.md).

## Ricetta congelata e prima v1 addestrata

Il confronto interno dei cinque bracci è completo. La regola prefissata ha
selezionato `hybrid_wt`, EMA 0,99 allo step 1.000. È stato eseguito un refit
da zero, seed 17, sugli assi training ufficiali, seguito da una sola valutazione
ufficiale dopo il freeze. La v1 ha **12.184.801 parametri attivi**.

Ricetta: RNA 25.017→256, fusione AlphaGenome+annotazioni, otto FFN solo lato
CpG, decoder fattorizzato rango 32, intercetta condivisa con la testa ausiliaria
della media, MSE/PCC globali + MSE/PCC within-tissue + supervisione media,
EMA. La variante con intercetta separata non ha fornito un vantaggio utile.

**AlphaGenome solo è praticamente alla pari con il selezionato.** Il beneficio
grande è quello delle nuove feature, non una superiorità dimostrata della
fusione o della combinazione di loss. Questa è una prima v1 congelata, non
una conclusione multi-seed sulle nuove feature o sull'architettura finale.

Obiettivo: congelare una ricetta che predica la metilazione usando RNA del
paziente, informazioni funzionali della CpG e un embedding AlphaGenome già
calcolato. Nessuna metilazione del paziente o statistica osservata della CpG
nuova deve essere necessaria in inferenza. La v1 è inizialmente **Array-only**.
Il confronto con un MethylProphet allenato anche su EPIC/WGBS deve dichiarare
questa differenza; non sarebbe un confronto a parità di dati di training.

La decisione riguarda un primo candidato riproducibile. La combinazione di
modifiche favorevoli negli screening precedenti non è automaticamente migliore:
per questo precede il refit un confronto breve interno con gli input definitivi.

## Input verificati

Cache trovata nel progetto adiacente:
`../CpGRepresentationBenchmark/data/cache/representations/alphagenome_locus.h5`.

- 408.399 CpG × 1.536 componenti float16; embedding del decoder a 1 bp.
- hg38/GRCh38, citosina della CpG in coordinate 1-based, contesto 524.288 bp,
  margine 65.536 bp; sequenza di riferimento, nessun input paziente.
- Metadati `dummy=false`, `patient_specific=false`, fit esterno pretrained.
- Tutte le 40.627 CpG Array chr1 presenti, finite e non nulle. Identificatori,
  cromosoma e posizione coincidono esattamente tra i due registri.
- Gli ID del file non sono ordinati: lookup per ID obbligatorio; la posizione
  di riga non è un identificatore condiviso.
- Nessuna nuova inferenza AlphaGenome è necessaria.

| Asse ufficiale chr1 | CpG richieste | Mancanti nella cache AlphaGenome |
|---|---:|---:|
| Array training | 33.885 | 0 |
| Array validation | 6.742 | 0 |
| EPIC training | 71.748 | 34.523 |
| WGBS training | 1.999.446 | 1.962.528 |

La copertura completa Array rende praticabile la prima v1. Non si riducono
silenziosamente gli assi EPIC/WGBS per aggirare la copertura insufficiente.

## Architettura candidata

1. RNA: tutti i 25.017 geni, normalizzazione per gene calcolata soltanto sui
   pazienti training, LayerNorm per campione, proiezione 256, GELU e LayerNorm.
2. CpG: due rami normalizzati distinti. Il ramo AlphaGenome proietta
   1.536→256; quello funzionale usa le 4.165 tracce sparse e 23 annotazioni.
   Per la fusione iniziale si sommano i rami con fattore `1/sqrt(2)` e si
   mantengono gli otto FFN funzionali testati, calcolati una volta per locus.
3. Interazione: `u_p = f(RNA_p)` e `a_c = g(CpG_c)`, rango 32;
   `beta_hat_pc = sigmoid(b_c + u_p @ a_c / sqrt(32))`.
4. `b_c` è una funzione appresa delle feature CpG. Si verifica anche una testa
   di intercetta indipendente dalla testa ausiliaria della media: la media di
   `sigmoid(b+interazione)` non coincide in generale con `sigmoid(b)`.

I fattori RNA e CpG possono essere calcolati e riutilizzati separatamente in
inferenza. Il costo dell'interazione è proporzionale a `P*C*32`, senza FFN
su ogni coppia. L'effettivo vantaggio end-to-end resta da misurare con I/O,
cache e batch controllati. I vettori non sono automaticamente programmi
biologici identificabili: il prodotto è invariante a cambi di base dei fattori.

La proiezione RNA 512, i Perceiver, routing adattivo, vicini con metilazione e
incrementi del rango restano fuori dalla prima combinazione: non hanno ancora
dimostrato un beneficio robusto nello stesso regime. Si mantiene il gene set
completo e il campionamento casuale dei pazienti; il forte raggruppamento per
tessuto aveva peggiorato i risultati.

## Training e validazione

- Split ufficiali immutati: 8.260 pazienti e 33.885 CpG training; 918 pazienti
  e 6.742 CpG validation. Tre viste ufficiali, riportate separatamente.
- Selezione interna solo nel training ufficiale: 7.435 pazienti × 29.913 CpG
  training, 825 pazienti × 3.972 CpG validation, in cinque blocchi da 5 Mb.
- RNA ricostruito dall'`X` canonico float32 e standardizzato sui soli pazienti
  del training interno. Nel refit le statistiche vanno ricalcolate sugli 8.260
  pazienti training ufficiali e salvate nel checkpoint.
- Target canonici float32 anche in valutazione. La cache dei vecchi screening
  conteneva beta float16; i nuovi numeri non sono confronti perfettamente
  appaiati con quei run. Anche la distanza TSS viene ristandardizzata sui soli
  loci training interni.
- EMA 0,99 per valutazione/checkpoint; AdamW 2e-4, wd 1e-4, gradient clipping 1,
  BF16 solo per forward, loss/statistiche in float32; warmup 200, cosine 4.000.
- Batch casuali 256 pazienti × 512 CpG. Valutazioni prefissate a 1.000, 2.000 e
  4.000 step; il numero di step va scelto internamente e congelato prima del refit.
- Base: `MSE + 0.15*PCC_locus + 0.15*MSE_media_aux`. Il target della media
  ausiliaria usa esclusivamente metilazione dei pazienti e CpG training.
- Variante da verificare: aggiunta `1.0*MSE_within_tissue + 0.15*PCC_within_tissue`.
  Tissue serve soltanto a costruire la loss e le metriche; non è un input del
  predittore. I tessuti ignoti (`NA`) sono esclusi da questi termini.

L'EMA e la loss within-tissue hanno supporto negli screening, ma la loro
combinazione con AlphaGenome deve essere valutata direttamente. Non si impone
varianza predetta uguale alla varianza osservata e non si usano gruppi definiti
dai target validation per scegliere il percorso di inferenza.

## Confronto interno preregistrato

Cinque bracci seed 17: funzionale, AlphaGenome, fusione, fusione+within-tissue,
fusione con intercetta separata. Stessi dati, batch, init delle parti comuni,
EMA e scheduler. Il protocollo è salvato prima dei risultati nel laboratorio.

Per ogni checkpoint si usa il riferimento funzionale allo stesso step.
Un candidato AlphaGenome è ammissibile se:

- MAS-PCC double-OOD almeno pari al riferimento;
- MSE double-OOD non oltre +1% rispetto al riferimento;
- MAS-PCC nuovi pazienti non inferiore di oltre 0,01;
- MAS-PCC within-tissue double-OOD non inferiore di oltre 0,005.

Fra gli ammissibili si sceglie la maggiore MAS-PCC double-OOD; in parità, MSE
minore e poi meno step. Le soglie sono guardrail di sviluppo, non equivalenza
statisticamente dimostrata. Nessun candidato viene adottato se nessuno passa.

Il confronto ufficiale rimane chiuso durante questi esperimenti. Dopo la scelta:
congelare architettura/loss/budget, rifittare sugli assi training ufficiali,
valutare una volta sulle tre viste, quindi riportare risultati senza cambiare
la ricetta in risposta al benchmark. Un singolo refit è una v1; una conclusione
robusta richiede altri seed e altri blocchi/cromosomi.

## Risultati del confronto interno

Double-OOD interno, **stesso budget di 1.000 step**, checkpoint EMA, seed 17:

| Input/modifica | MAS-PCC | MSE | Within-tissue |
|---|---:|---:|---:|
| Annotazioni funzionali | 0,44489 | 0,033197 | 0,31496 |
| AlphaGenome solo | 0,52463 | 0,026320 | 0,35591 |
| AlphaGenome + annotazioni | 0,52224 | 0,026298 | 0,35231 |
| Fusione + loss within-tissue, selezionato | **0,52490** | **0,026289** | **0,36117** |
| Fusione + intercetta separata | 0,52222 | 0,026299 | 0,35226 |

Rispetto alle sole annotazioni allo stesso budget: +0,0800 MAS-PCC e −20,8%
MSE. AlphaGenome solo a 2.000 step ottiene 0,52481 e MSE 0,027846: la
differenza di PCC rispetto al selezionato è 0,000095 e non dimostra un
vantaggio statisticamente affidabile. La scelta segue la regola congelata;
il checkpoint più breve ha anche MSE minore. A 4.000 step tutti i bracci
AlphaGenome perdono PCC rispetto al loro checkpoint iniziale; i dati non
motivano un refit automaticamente prolungato a 80 epoche.

Il refit usa **1.000 aggiornamenti con scheduler ancora a orizzonte 4.000**,
come nel checkpoint selezionato. Sono 131.072.000 coppie campionate prima
del masking, circa 0,47 volte la dimensione della griglia training ufficiale:
non si tratta di 80 epoche, né di una visita esaustiva a ogni coppia.

## Valutazione ufficiale del refit

| Vista ufficiale | Coppie osservate | MAS-PCC | MAC-PCC | MSE | MAE | Within-tissue |
|---|---:|---:|---:|---:|---:|---:|
| Nuovi pazienti, CpG training | 30.563.936 | 0,58998 | 0,93611 | 0,017964 | 0,08760 | 0,39039 |
| Nuove CpG, pazienti training | 55.154.676 | 0,56366 | 0,93046 | 0,019665 | 0,08768 | 0,45476 |
| Nuovi pazienti e nuove CpG | 6.129.992 | **0,53454** | 0,92950 | **0,020031** | 0,08858 | 0,34478 |

Tutti e tre i conteggi delle osservazioni valide coincidono con l'audit del
protocollo MethylProphet. Questi numeri non sono direttamente confrontabili
con lo split interno a blocchi. Non sono ancora affiancati a predizioni di
MethylProphet rivalutate con gli stessi ID e le stesse metriche.

Nel double-OOD ufficiale il 13,44% delle coppie ha errore assoluto >0,2 beta;
il decile inferiore della PCC per CpG è 0,1505, lo 0,53% dei loci ha PCC
negativa e il bias medio per locus rappresenta il 34,59% dell'errore quadratico.
Le open sea hanno MSE 0,02812 contro 0,01359 delle island; le CpG con SD
osservata >0,15 hanno MSE 0,03823 e 26,67% di errori >0,2. Sono diagnostiche
post-hoc: nessuna di queste statistiche viene usata in inferenza o per ritoccare
la ricetta dopo la valutazione ufficiale. Le correlazioni grezze tra categorie
rimangono influenzate dalla diversa variabilità dei target.

## Metriche per decidere se generalizza

Oltre a MAS-PCC e MAC-PCC: MSE/MAE, correlazione within-tissue, frequenza di
errori assoluti beta >0,2, bias per locus, decile peggiore e risultati per
blocco genomico. Le stratificazioni island/shore/open sea, promotori/cCRE e
variabilità osservata servono alla diagnostica, non a fornire input oracolo.
Un aumento di correlazione con MSE peggiore è un compromesso da rendere esplicito.

Per MethylProphet servono gli stessi ID, gli stessi target, la stessa gestione
dei NaN e le medesime definizioni di metriche: una tabella pubblicata con un
training mix diverso non basta per una conclusione a parità di condizioni.

## Artefatti

Laboratorio isolato:
`/data2/home/vcivale/projects/methylation/chr1_alphagenome_v1_2026-10-17/`.
`pilot.py` esegue il confronto interno; `protocol.json` registra audit e regola
di scelta; `normalization_dev.npz` conserva statistiche e ID usati per il fit.
Il modello main della repository non viene sostituito durante lo screening.

Artefatti del refit nella directory `final_seed17/`:

- `checkpoint.pt`: EMA e pesi raw, optimizer, ricetta congelata, statistiche
  RNA/TSS, ordine dei geni, ID dei set di fit, provenienza e hash del codice.
- `official_metrics.json`: tre viste ufficiali e diagnostiche per blocco,
  contesto e variabilità osservata.
- `*_per_cpg.csv.gz`: errori aggregati per CpG, senza predizioni individuali.
- `inference_check.json`: ricaricamento del checkpoint, parità esatta del
  preprocessing su input reali, rifiuto di un ordine errato dei geni.

`predict_checkpoint.py` fornisce `PredictorFromCheckpoint`: accetta RNA nelle
unità canoniche di `X`, ordine esplicito dei geni, tracce/annotazioni e
AlphaGenome. Il controllo di inferenza non legge target di metilazione e
riproduce esattamente il percorso di preprocessing di riferimento.

Il training del refit ha richiesto 20,6 secondi con dati residenti in GPU:
caricamento/preparazione, selezione dei bracci e costo storico di generazione
degli embedding sono esclusi. Il checkpoint pesa circa 197 MB perché include
anche pesi raw e stato dell'optimizer.

Snapshot di metriche, audit, freeze e hash:
[alphagenome_chr1_v1_2026-10-17.json](experiments/alphagenome_chr1_v1_2026-10-17.json).
Codice e checkpoint sono nel laboratorio isolato indicato sopra; questa v1
non è ancora integrata nel trainer di produzione `RNAMethylationTrainer`.
