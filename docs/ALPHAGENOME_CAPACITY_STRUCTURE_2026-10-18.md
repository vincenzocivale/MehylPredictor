# AlphaGenome: capacità adattiva e struttura funzionale delle CpG

Verifica interna del 18 ottobre 2026. **Il routing del compute non mostra un
vantaggio affidabile nelle ricette testate. Gli embedding contengono invece
informazione trasferibile sulla media, variabilità e associazioni con l'RNA.**

La distinzione è operativa: prevedere la variabilità di una CpG non equivale a
prevedere quanto quella CpG beneficerà di un modello più profondo.

## Disegno e isolamento

72 fit da zero, seed 17/18/19, sei architetture, quattro split; 108 checkpoint
valutati. Il modello è la v1 AlphaGenome+annotazioni, decoder rango 32, con
inizializzazione identica delle parti comuni. AlphaGenome resta congelato;
proiezione e decoder sono appresi. `d0` mantiene proiezione CpG, testa della
media e fattori: rimuove soltanto gli otto FFN residui lato CpG.

- Tre fold genomici out-of-fold sui 29.913 loci del training dev, blocchi da
  5 Mb assegnati con seed 20261018. In ciascuno: 6.691 pazienti training e
  744 pazienti esclusi, scelti soltanto fra i 7.435 pazienti training dev.
- CpG held-out nei tre fold: 6.782 / 12.787 / 10.344. CpG effettivamente
  utilizzate nel training dopo embargo: 22.579 / 16.401 / 18.972.
- Embargo di 300 kb attorno ai blocchi held-out. Riduce la prossimità genomica,
  ma non garantisce contesti AlphaGenome da 524 kb completamente disgiunti.
- Refit interno su 7.435 pazienti e 29.306 CpG dopo embargo; valutazione dev
  su 825 pazienti e 3.972 nuove CpG, nei cinque blocchi 2/4/32/35/47.
- Tutti gli ID appartengono al training ufficiale: 8.260 pazienti e 33.885
  CpG. Sovrapposizione con pazienti e CpG dell'holdout ufficiale: **zero**.
- RNA canonico e target float32; statistiche RNA/TSS stimate di nuovo su
  ciascun training fold. Batch 256×512, AdamW 2e-4, wd 1e-4, warmup 200,
  scheduler con orizzonte 4.000, EMA 0,99, forward BF16, loss float32.
- Loss della v1: MSE + 0,15 PCC + 0,15 media ausiliaria + MSE within-tissue
  + 0,15 PCC within-tissue. Tissue non è input del predittore.

Profondità 0/2/8 valutate a 1.000 e 2.000 aggiornamenti. Tre ricette aggiuntive,
prefissate e valutate a 1.000: shrinkage dell'interazione con penalità 0,001;
correzione shallow/deep con gate morbido e penalità media del gate 0,001;
16 prototipi soft più residuo di risposta scalato 0,1 e penalizzato a 0,001.
I prototipi e il gate morbido usano comunque gli otto FFN.

Protocollo di training e analisi salvati prima dei rispettivi risultati.
Il riferimento uniforme è scelto per minima MSE media dei tre fold interni,
poi meno step e minore profondità: **d0 a 1.000 step**. Questa scelta riguarda
la MSE sulle nuove CpG; non riproduce la regola multi-metrica della v1.

Il dev finale è lo split già usato negli screening precedenti: questa è
un'analisi interna di sviluppo, non una nuova conferma esterna incontaminata.

## Risultati del confronto

Medie di metriche su tre seed, non ensemble delle predizioni. La MSE è pesata
per coppie osservate; MAS-PCC è la mediana delle correlazioni per CpG.

| Modello, 1.000 step | MSE media fold interni | MSE dev double-OOD | MAS-PCC dev | Within-tissue dev |
|---|---:|---:|---:|---:|
| 0 FFN aggiuntivi | **0,026996** | **0,025954** | 0,52163 | 0,35735 |
| 2 FFN | 0,027301 | 0,026199 | 0,52369 | 0,36103 |
| 8 FFN | 0,027535 | 0,026428 | 0,52683 | 0,36243 |
| 8 FFN + shrinkage | 0,027344 | 0,026290 | **0,52708** | 0,36242 |
| Correzione con gate | 0,027736 | 0,026802 | 0,52520 | 0,36092 |
| Prototipi + residuo | 0,027208 | 0,026496 | 0,51453 | 0,35499 |

Il modello senza FFN aggiuntivi riduce la MSE dev dell'1,79% rispetto a otto
FFN, ma perde circa 0,0052 MAS-PCC e generalizza peggio sui loci noti. Nessuna
ricetta domina su tutte le viste e metriche. Lo shrinkage migliora leggermente
la MSE del completo (−0,52%), ma resta peggiore di d0 sulla MSE delle nuove CpG.
Queste penalità/prototipi fissi non esauriscono le possibili regolarizzazioni.

Differenza MSE d8−d0: +0,000474, intervallo bootstrap per blocchi
[+0,000225; +0,000665]. Solo cinque blocchi: intervalli descrittivi,
senza bootstrap pazienti o correzione per confronti multipli.

### Il training prolungato favorisce le CpG note

| Modello | MSE nuove CpG, 1k → 2k | MSE CpG note / nuovi pazienti, 1k → 2k |
|---|---:|---:|
| d0 | 0,025954 → 0,027235 | 0,019078 → 0,016092 |
| d2 | 0,026199 → 0,027544 | 0,018002 → 0,014952 |
| d8 | 0,026428 → 0,027927 | 0,017115 → 0,013952 |

La vista sui loci noti usa un campione fisso di 1.024 loci training, uguale tra
architetture nello stesso split, e tutti i pazienti esclusi. Non è l'intero
asse ufficiale dei loci noti. Le diagnostiche in-sample usano 512×1.024.

Con d8, 2.000 aggiornamenti peggiorano la MSE delle nuove CpG del 5,67% e
la migliorano sui loci noti del 18,48%. La MAS-PCC double-OOD scende
0,52683→0,52191. Il 65,84% dei loci held-out peggiora in MSE, usando errori
mediati tra seed. Il bias medio per locus spiega il 71,4% dell'aumento di
MSE: +0,001071 bias² e +0,000428 errore centrato.

È evidenza di un problema di generalizzazione ai loci nuovi. La decomposizione
non identifica causalmente la testa responsabile: la media predetta dipende
sia dall'intercetta sia dall'interazione media con l'RNA.

## Il gate non recupera la complementarità

Per ogni CpG training si stima il beneficio `MSE_cheap − MSE_d8` solo da
predizioni di modelli che escludono quel blocco genomico e i pazienti usati per
misurare l'errore. I target del gate sono la media dei tre seed.

Gate fissi: Ridge alpha 100 su annotazioni dense, AlphaGenome o entrambi;
HistGradientBoosting sul concatenato, massimo 100 iterazioni, 7 foglie,
minimo 200 loci per foglia, l2=10. Nessuna scelta di iperparametri sui target dev.
Nessun ID, posizione o statistica target della nuova CpG è input del gate.
Budget massimi prefissati 10/25/50/100%; si usa d8 solo se il beneficio
previsto è positivo. Il budget è un limite, non una quota da riempire.

| Confronto | Feature / gate | AUROC beneficio positivo | Spearman beneficio |
|---|---|---:|---:|
| d0 vs d8 | Annotazioni / Ridge | 0,524 | 0,054 |
| d0 vs d8 | AlphaGenome / Ridge | 0,501 | 0,005 |
| d0 vs d8 | Fusione / Ridge | 0,502 | 0,007 |
| d2 vs d8 | Annotazioni / Ridge | 0,515 | 0,032 |
| d2 vs d8 | AlphaGenome / Ridge | 0,490 | −0,030 |
| d2 vs d8 | Fusione / Ridge | 0,491 | −0,029 |

Con d0/d8, il gate sulle annotazioni usa d8 nel 6,7% dei loci e ottiene MSE
0,025940 contro 0,025954 di d0: differenza −0,000013, intervallo per blocchi
[−0,000047; +0,000025]. Il vantaggio non è convincente. Il gate Ridge sulla
fusione al budget 25% ottiene 0,025989. Con d2/d8, lo stesso gate a 25%
peggiora 0,026199→0,026325. Gli alberi prevedono vantaggi sempre non positivi
sul dev e ricadono nel modello economico.

La scelta per locus è poco stabile tra seed. Sul dev, d2/d8 a 1.000 step:
Spearman 0,079–0,141 e accordo sul segno 56,9–58,2%. Per d0/d8:
0,175–0,227 e 59,0–61,3%. Anche la selezione basata su errori di altri seed
lascia un margine oracolo, ma usa target della CpG held-out e non è deployabile.

La complementarità del singolo run sopravvaluta quindi la quantità di segnale
che possiamo assegnare a una regola funzionale generalizzabile. Questi risultati
non dimostrano l'impossibilità di ogni routing, né riguardano un gate diverso
per ciascuna coppia paziente–CpG o un gate tra checkpoint temporali.

## Struttura presente negli embedding

Probe Ridge alpha 100, scaler sui soli loci training effettivi. Per il fit,
media e SD della metilazione sono calcolate sui pazienti training; per il test,
sui pazienti esclusi. Nessuna statistica osservata della nuova CpG viene usata
come input. Le rappresentazioni apprese sono quelle dei d8 a 1.000 step.

| Rappresentazione | R² media della metilazione | R² deviazione standard |
|---|---:|---:|
| AlphaGenome originale, 1.536D | **0,9072** | 0,7068 |
| Fusione proiettata, 256D, media seed | 0,8986 | 0,6417 |
| Dopo 8 FFN, 256D, media seed | 0,9041 | **0,7126** |
| Coefficienti CpG, 32D, media seed | 0,8905 | 0,6955 |

La proiezione riduce soprattutto la decodificabilità della variabilità; il
ramo profondo recupera buona parte dell'informazione. Non emerge una perdita
catastrofica nell'imbuto a 32 fattori. Il R² della media riguarda differenze
tra loci, non la varianza della metilazione spiegata tra pazienti.

Da AlphaGenome grezzo si recuperano anche contesto CpG e regione genica:
macro-AUROC 0,9356 e 0,8893 in questo probe chr1. La decodifica quasi perfetta
nelle rappresentazioni ibride è circolare: queste etichette sono input diretti.
I numeri non sono direttamente confrontabili con i precedenti probe logistic
regression chromosome-blocked del progetto CpGRepresentationBenchmark.

### I vicini trasferiscono associazioni con l'RNA

Per ogni CpG si costruisce un descrittore di 16 covarianze within-tissue con
PC dell'RNA. I 1.024 geni a maggiore variabilità, scaler e PCA sono scelti e
fittati soltanto sui pazienti training. Le associazioni dei vicini sono stimate
sui pazienti training; il target della CpG nuova è stimato sui pazienti esclusi.
Sono associazioni osservazionali, non effetti causali dell'RNA.

Si confrontano cinque vicini nello spazio delle feature con la media di venti
gruppi di cinque loci casuali. Tutti i vicini distano almeno 1 Mb; controlli
appaiati per island/shore/shelf/open sea, regione genica e quartile distanza
TSS. Standardizzazione e PCA delle feature sono fittate sui soli loci training.
Per dimensioni >64 si usa PCA64, poi similarità coseno. Tutte le 3.972 CpG
dev hanno almeno cinque controlli ammissibili.

| Spazio dei vicini | Δ MSE descrittore vs controlli appaiati | R² descrittore |
|---|---:|---:|
| AlphaGenome originale | −11,18% | 0,106 |
| Fusione proiettata, seed17 | −16,88% | 0,164 |
| Fusione profonda, seed17 | −23,77% | 0,233 |
| Fattori CpG, seed17 | −23,80% | 0,233 |

Le differenze per blocchi sono tutte negative; per i fattori, differenza MSE
−9,56e−6, intervallo [−1,10e−5; −7,53e−6]. Questo sostiene una geometria
funzionale oltre alle categorie grossolane e alla prossimità genomica.
**Non è una riduzione del 24% della MSE beta del predittore.** Il controllo
media venti gruppi casuali, quindi usa uno stimatore più stabile del singolo
gruppo di cinque vicini. La verifica dei vicini appresi usa un solo seed.

I 16 prototipi testati non hanno tradotto questa struttura in un vantaggio
predittivo. Non sono collassati a un'unica classe: nel seed17 l'uso aggregato
equivale a circa 9,15 prototipi. Il gate morbido ha media 0,364 e non è
collassato a zero. La struttura a vicini non implica che 16 prototipi globali
siano la parametrizzazione giusta; i fattori non sono programmi biologici
identificabili senza ulteriori vincoli.

## Compute reale e precisione numerica

Microbenchmark residente su A100 40 GB, 256 pazienti × 3.972 CpG, seed17.
Routing Ridge sulle feature ibride d2/d8, 25% loci al profondo. 100 warmup e
30 misure per condizione. I tempi escludono I/O, fitting del gate e costruzione
delle bag sparse. GPU condivisa: non è un benchmark end-to-end isolato.

| Cache dei fattori | d2, ms | d8, ms | Routing 25%, ms |
|---|---:|---:|---:|
| Nessuna | 3,259 | 3,844 | 4,476 |
| Solo CpG | 0,450 | 0,450 | 0,883 |
| RNA e CpG | 0,075 | 0,074 | 0,178 |

Con i fattori CpG già in cache, d2 e d8 hanno praticamente lo stesso costo;
il routing esegue due encoder RNA. Con entrambe le modalità in cache,
restano prodotti a rango 32 e il costo di assemblare due percorsi. I tempi
esatti e gli IQR sono nello snapshot. Il costo storico di estrazione
AlphaGenome è escluso.

Il routing su due modelli indipendenti non elimina quindi il costo più utile
da ottimizzare nell'inferenza ripetuta: i fattori statici si possono già
precalcolare. Una futura architettura adattiva dovrebbe condividere l'encoder
RNA e dimostrare un vantaggio sul migliore uso delle cache.

Il confronto BF16 fra inferenza su sottogruppi e batch completo mostra una
differenza massima di 0,00586 beta, dipendente dal batching. L'equivalenza
FP32 stretta mostra una differenza massima di 2,98e−7 (tolleranza 1e−5). Le metriche di
routing derivano dalla selezione delle metriche per locus dei forward comuni;
non incorporano il cambiamento numerico di un nuovo batching BF16.

## Indicazioni per il prossimo esperimento

1. **Priorità alla generalizzazione del livello medio e alla durata del fit.**
   Il prolungamento del training migliora le CpG note e peggiora quelle nuove,
   soprattutto nella media per locus. Un test causale utile è congelare presto
   ramo CpG/intercetta e continuare il ramo RNA, misurando tutte le viste.
2. **Condivisione locale delle risposte fra CpG funzionalmente vicine.**
   Il trasferimento delle associazioni oltre 1 Mb supporta regolarizzazione
   locale delle risposte o un banco di coefficienti condivisi, confrontati con
   controlli appaiati. L'esito dei 16 prototipi suggerisce di non imporre subito
   pochi gruppi globali.
3. **Usare media e variabilità previste come prior appresi.**
   La variabilità va prevista dagli input genomici su training/cross-fitting;
   non si può usare la SD osservata delle CpG nuove. Un prior di variabilità
   non autorizza a imporre SD predetta uguale alla SD dei target: esiste rumore
   non prevedibile dall'RNA.
4. **Routing compute a priorità ridotta per questa v1.**
   Il gate non predice il beneficio e le cache riducono il potenziale risparmio.
   Un ulteriore tentativo richiede target di beneficio meno rumorosi, encoder
   condiviso e un vantaggio sulle nuove CpG senza sacrificare le altre viste.

## Artefatti e riproduzione

Snapshot aggregato, senza predizioni individuali:
[JSON](experiments/alphagenome_capacity_2026-10-18.json).
[Figura PNG](experiments/alphagenome_capacity_2026-10-18.png) e
[PDF](experiments/alphagenome_capacity_2026-10-18.pdf).

Artefatti completi nella directory ignorata da git:
`artifacts/alphagenome_capacity_2026-10-18/`. Include protocolli, hash del
codice sorgente congelato, audit ID, normalizzazioni per fold, checkpoint dev,
errori aggregati per CpG e rappresentazioni. Non contiene predizioni RNA/
metilazione individuali esportate. Il laboratorio v1 adiacente viene letto.

```bash
python scripts/verify_alphagenome_capacity.py
python scripts/analyze_alphagenome_capacity.py
python scripts/benchmark_alphagenome_capacity.py
python scripts/diagnose_alphagenome_capacity.py
python scripts/plot_alphagenome_capacity.py
```

Training/analisi richiedono l'ambiente `methyl-predictor`, i dati canonici e
il laboratorio v1; training e benchmark richiedono CUDA. Il rendering della
figura è stato eseguito con il Python base che dispone di matplotlib.

Verifiche automatiche: dieci test su selezione HDF5 con assi non ordinati,
embargo, inizializzazione comune, parità d8/v1, fattorizzazione e gradienti
finiti per tutti i bracci, identità MSE=bias²+errore centrato. Inoltre controlli
runtime di finitezza, disgiunzione dei fold e parità numerica GPU del routing.

Limiti: chr1 Array-only, tre seed, cinque blocchi nel dev finale, nessuna nuova
valutazione ufficiale; i bootstrap riportati misurano solo variabilità tra
blocchi. Nessuna conclusione genome-wide o equivalenza statistica deriva da
questo pilot. La ricetta main non viene sostituita da questi esperimenti.
