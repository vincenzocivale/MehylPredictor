# AlphaGenome chr1 v2: training e audit degli errori CpG

> **Allineamento del benchmark, 18 ottobre 2026:** l’audit sulle predizioni rilasciate di MethylProphet ha trovato 723 pazienti validation diversi nello split derivato usato da questa campagna. I risultati sotto restano riferiti agli assi qui descritti; il confronto appaiato con il modello pubblicato usa invece lo split canonico verificato su tutti gli ID e target. Vedere il [nuovo benchmark misto](ALPHAGENOME_MIXED_METHYLPROPHET_2026-10-18.md).

Versione creata e addestrata il 18 ottobre 2026. Selezione sui soli fold interni, refit su tutti gli assi training ufficiali, poi valutazione delle tre viste ufficiali.

## Ricetta selezionata

Braccio **local01**, mean blend **0.0**, 1.000 aggiornamenti, EMA 0,99; seed 17 primario e repliche 18/19. Guardrail interni superati: **True**.

Configurazione attiva: intercetta shallow=True; penalità locale=0.01; blend del prior=0.0.

La calibrazione verso il prior non è attiva nella ricetta selezionata.

Il candidato supera i guardrail interni prefissati.

Il decoder resta fattorizzato a rango 32. La famiglia candidata confronta intercetta shallow/deep, regolarizzazione locale delle risposte RNA (peso 0 / 0,01 / 0,1) e calibrazione della media (blend 0 / 0,5 / 1). La calibrazione risolve un offset affinché la media prevista su 512 fattori RNA training corrisponda alla combinazione fra media del modello e prior Ridge AlphaGenome. Il prior è fittato soltanto sulle medie dei pazienti e loci training; nessuna metilazione della CpG nuova è un input.

I vicini del grafo sono cinque CpG training, distanti almeno 1 Mb, scelte nello spazio AlphaGenome standardizzato/PCA64 e appaiate per contesto, regione e quartile TSS; fallback globale se necessario. La penalità confronta risposte RNA centrate, non intercette o coefficienti dipendenti dalla base. Il grafo serve solo nel training.

## Selezione interna e generalizzazione

Tre seed × tre fold genomici da 5 Mb con embargo 300 kb; pazienti esclusi separatamente. Budget, lambda e blend prefissati prima dei risultati. Si sceglie la minima MSE double-OOD con PCC e within-tissue non inferiori di oltre 0,005 al controllo; sui loci noti PCC non inferiore di oltre 0,01 e MSE entro +5%. Se nessun candidato passa, si addestra e riporta il miglior candidato nuovo dichiarandolo non confermato. Lo split dev successivo non cambia la scelta.

| Braccio | Blend | MSE interna | PCC interna | MSE loci noti | Ammissibile |
|---|---:|---:|---:|---:|---|
| baseline | 0.0 | 0.027531 | 0.50380 | 0.015401 | True |
| baseline | 0.5 | 0.025968 | 0.50484 | 0.016887 | False |
| baseline | 1.0 | 0.027305 | 0.50436 | 0.021422 | False |
| mean | 0.0 | 0.027302 | 0.50256 | 0.015773 | True |
| mean | 0.5 | 0.026043 | 0.50331 | 0.017210 | False |
| mean | 1.0 | 0.027399 | 0.50242 | 0.021320 | False |
| local01 | 0.0 | 0.026583 | 0.50436 | 0.015873 | True |
| local01 | 0.5 | 0.025428 | 0.50524 | 0.017304 | False |
| local01 | 1.0 | 0.026731 | 0.50462 | 0.021226 | False |
| local10 | 0.0 | 0.027371 | 0.50384 | 0.018538 | False |
| local10 | 0.5 | 0.026435 | 0.50423 | 0.019980 | False |
| local10 | 1.0 | 0.027687 | 0.50417 | 0.023572 | False |

Conferma dev a ricetta congelata, media dei tre seed: MSE double-OOD 0.026422→0.025633, PCC 0.52686→0.52724. Questi risultati non modificano la scelta. Il dev a blocchi non è direttamente confrontabile con lo split ufficiale intercalato delle CpG.


## Confronto ufficiale

Stessi ID, target canonici float32, preprocessing salvato separatamente per ciascun checkpoint e forward FP32. Il controllo è il checkpoint v1 seed17 già congelato. Il confronto primario seed17 è appaiato; gli altri seed v2 misurano variabilità del candidato e non sono tre baseline v1 appaiati.

| Vista | Modello | MSE | MAE | MAS-PCC | Within-tissue | Errori >0,2 beta |
|---|---|---:|---:|---:|---:|---:|
| Nuovi pazienti / CpG note | v1 | 0.017964 | 0.08760 | 0.58996 | 0.39051 | 12.35% |
| Nuovi pazienti / CpG note | v2 seed17 | 0.018731 | 0.09121 | 0.59030 | 0.38935 | 13.09% |
| Nuove CpG / pazienti noti | v1 | 0.019663 | 0.08767 | 0.56388 | 0.45497 | 13.12% |
| Nuove CpG / pazienti noti | v2 seed17 | 0.019423 | 0.08862 | 0.56381 | 0.45532 | 13.28% |
| Nuovi pazienti e nuove CpG | v1 | 0.020030 | 0.08858 | 0.53443 | 0.34497 | 13.44% |
| Nuovi pazienti e nuove CpG | v2 seed17 | 0.019772 | 0.08959 | 0.53722 | 0.34185 | 13.60% |

| Seed v2 | MSE double-OOD | PCC double-OOD |
|---:|---:|---:|
| 17 | 0.019772 | 0.53722 |
| 18 | 0.019935 | 0.53775 |
| 19 | 0.019797 | 0.53641 |

Sul confronto primario migliora in MSE il **52.8% delle CpG**. Differenza media MSE v2−v1: -0.000258; intervallo bootstrap per blocchi [-0.0004763028005907807, -4.0206843836765465e-05]. Intervalli descrittivi, senza bootstrap pazienti o correzione multipla.

La differenza MSE si scompone in -0.000163 di bias² e -0.000095 di errore centrato.

## In quali CpG sbaglia e di quanto

Audit di **6,742 CpG nuove** sui pazienti esclusi. Il 10% peggiore dei loci concentra il **35.8% dell’errore quadratico**. Il bias medio del locus rappresenta il **34.2% dell’SSE**. 658 CpG sono nel decile peggiore in almeno due seed su tre.

| Contesto | CpG | RMSE beta | MAE beta | PCC mediana | Errori >0,2 beta |
|---|---:|---:|---:|---:|---:|
| island | 3330 | 0.1161 | 0.0653 | 0.3934 | 8.7% |
| open_sea | 1638 | 0.1669 | 0.1203 | 0.6143 | 19.9% |
| shelf | 459 | 0.1568 | 0.1099 | 0.6175 | 17.2% |
| shore | 1315 | 0.1552 | 0.1063 | 0.5592 | 17.0% |

Queste differenze sono descrittive: contesto e variabilità dei target sono correlati. Lo snapshot riporta anche gli incroci contesto×SD; la SD osservata non viene usata per predire. Sono disponibili anche diagnostiche per distanza dalla CpG training più vicina. Una bassa PCC su CpG quasi costanti può accompagnarsi a un errore assoluto piccolo. La SD predetta inferiore alla SD osservata può riflettere variabilità non prevedibile dall’RNA.

Le peggiori 15 CpG per MSE nel seed primario:

| CpG Illumina | Coordinate hg38, C 1-based | Contesto | RMSE beta | MAE beta | Bias | PCC | Errori >0,2 |
|---|---|---|---:|---:|---:|---:|---:|
| cg13420845 | chr1:155250440 | island | 0.528 | 0.505 | +0.497 | 0.255 | 93.6% |
| cg27181253 | chr1:112509224 | island | 0.515 | 0.494 | -0.475 | 0.342 | 92.8% |
| cg26233753 | chr1:201347316 | open_sea | 0.501 | 0.465 | +0.463 | 0.400 | 90.2% |
| cg03257417 | chr1:110647249 | open_sea | 0.469 | 0.413 | +0.400 | 0.567 | 75.8% |
| cg26896499 | chr1:223362954 | shore | 0.466 | 0.457 | +0.457 | 0.244 | 98.6% |
| cg08465307 | chr1:3910517 | shelf | 0.458 | 0.400 | +0.385 | 0.529 | 75.9% |
| cg11314684 | chr1:243842986 | open_sea | 0.447 | 0.397 | +0.381 | 0.496 | 77.5% |
| cg20611680 | chr1:16227676 | shore | 0.436 | 0.410 | +0.406 | 0.612 | 90.5% |
| cg21829336 | chr1:38257113 | open_sea | 0.433 | 0.408 | +0.407 | 0.512 | 90.5% |
| cg19809791 | chr1:16701337 | island | 0.431 | 0.390 | +0.372 | 0.249 | 82.1% |
| cg13484546 | chr1:15758444 | shore | 0.430 | 0.373 | +0.356 | 0.467 | 72.1% |
| cg05785481 | chr1:38044262 | shore | 0.423 | 0.401 | -0.331 | 0.256 | 91.0% |
| cg14375345 | chr1:155230566 | shelf | 0.421 | 0.381 | +0.380 | 0.367 | 81.6% |
| cg23106844 | chr1:151398833 | shore | 0.421 | 0.391 | +0.391 | 0.468 | 85.8% |
| cg03047965 | chr1:22609412 | open_sea | 0.420 | 0.402 | +0.402 | 0.553 | 94.4% |

Bias positivo significa sovrastima della metilazione. Il CSV completo include media osservata/predetta, SD, RMSE, MAE, bias, PCC, PCC sui residui within-tissue, p90/p95 e massimo errore assoluto, frazione errori >0,1 e >0,2, distanza dalla CpG training più vicina e confronto locus per locus con v1. Il rapporto MSE/varianza osservata è solo un confronto diagnostico con una media-oracolo non disponibile in deployment.

Identificatori Illumina disponibili per 6741/6742 CpG. Join esatto per coordinate hg38, mantenendo tutti gli eventuali probe associati alla stessa coordinata; gli ID del registro restano univoci.

## Artefatti e utilizzo

[Figura PNG](experiments/alphagenome_v2_2026-10-18.png) · [PDF](experiments/alphagenome_v2_2026-10-18.pdf) · [Metriche JSON](experiments/alphagenome_v2_2026-10-18.json) · [Tutte le CpG held-out, CSV.gz](experiments/alphagenome_v2_2026-10-18_errors.csv.gz) · [Peggiori 100, CSV](experiments/alphagenome_v2_2026-10-18_worst100.csv) · [Maggiori peggioramenti rispetto alla v1, CSV](experiments/alphagenome_v2_2026-10-18_regressions100.csv)

Checkpoint primario: `artifacts/alphagenome_v2_2026-10-18/official/seed17/checkpoint.pt`. La directory contiene anche repliche, freeze, normalizzazione, grafo, prior e audit completi delle tre viste.

```python
from methylation_predictor.modeling.alphagenome import AlphaGenomePredictorFromCheckpoint
predictor = AlphaGenomePredictorFromCheckpoint('artifacts/alphagenome_v2_2026-10-18/official/seed17/checkpoint.pt', device='cuda')
beta = predictor(raw_rna, gene_ids, track_indices, offsets, functional_dense, alphagenome)
```

RNA nelle unità canoniche di X, ordine dei geni esplicito ed esatto; annotazioni 23D nella rappresentazione dello store congelato prima della standardizzazione TSS aggiuntiva. Annotazioni e AlphaGenome devono condividere lo stesso ordine CpG, con lookup per ID. Nessun accesso ai target durante inferenza. Il checkpoint è ricaricato e confrontato con input reali nel controllo di inferenza.

Verifiche: suite disponibile eseguita con esito positivo; otto test specifici su calibrazione, invarianza al batching, separazione dei gradienti, regolarizzazione delle risposte, ricaricamento del checkpoint e parità del controllo con la v1. Gli assi di fit e l’ordine dei 25.017 geni della v1 sono stati verificati prima del confronto. I conteggi delle coppie osservate sono confrontati con il protocollo canonico per tutte le viste.

Limiti: Array chr1, tre seed; holdout ufficiale già usato per la v1, quindi confronto successivo al precedente sviluppo e non nuova conferma esterna. Nessun tuning dopo il freeze in risposta alle metriche ufficiali. Queste misure non stabiliscono causalità biologica dei singoli errori. La nuova factory ha un contratto AlphaGenome esplicito; il trainer funzionale storico non viene usato per leggere automaticamente questi input.
