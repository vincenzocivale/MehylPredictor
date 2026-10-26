# Screening RNA, interazione CpG e training — 17 ottobre 2026

## Esito e decisioni

Screening concluso su chr1 Array, seed 17, senza nuova inferenza genomica.
Le feature e l'architettura dell'encoder CpG restano quelle del riferimento;
i suoi pesi vengono allenati normalmente. I risultati non costituiscono una
conferma multi-seed né una validazione sul training completo multisorgente.

- La loss within-tissue con batch casuali è il candidato da confermare.
  Con EMA migliora il double-OOD di **0,0041 rispetto al baseline con EMA**.
- EMA aiuta soprattutto durante il training: il vantaggio del baseline passa
  da +0,0077 a 2.000 step a +0,0013 a 4.000 step.
- Il forte raggruppamento dei pazienti per tessuto peggiora le prestazioni,
  anche aggiungendo la loss within-tissue.
- Chiavi statiche, ramo bilineare e RNA centrato per tessuto non mostrano un
  vantaggio double-OOD convincente nelle implementazioni provate.
- Ridurre gli FFN dopo l'attenzione da 8 a 2 mantiene il double-OOD con EMA
  vicino al riferimento, ma peggiora la vista nuovi pazienti e la correlazione
  within-tissue. Candidato di efficienza, non miglioramento generale dimostrato.
- Permutare l'RNA entro tessuto annulla la correlazione within-tissue:
  il modello usa l'abbinamento individuale RNA–metilazione, oltre al tessuto.

I numeri completi per tutte le viste, MSE inclusa, sono conservati in
[experiments/rna_training_2026-10-17.json](experiments/rna_training_2026-10-17.json).
Il file contiene anche percorsi originali e hash SHA-256 dei risultati e del
codice sperimentale per identificarne la versione.

## Protocollo comune

- Dati: cache `pi_lab/cache/array_chr1.npz`, split `dev` con CpG held-out
  in blocchi genomici da 5 Mb, pazienti stratificati per tessuto.
- Train: 7.435 pazienti e 29.913 CpG; validation: 825 pazienti e 3.972 CpG.
- Encoder CpG: 4.165 tracce sparse + 23 annotazioni dense; nessun AlphaGenome.
- Riferimento: copia congelata di `pi_lab/model.py`, larghezza 256,
  64 token RNA, FFN functional/retrieval/head = 8/8/2, circa 21,66 M parametri.
- Batch cartesiani 256 pazienti × 512 CpG; seed 17.
- AdamW, learning rate 2e-4, weight decay 1e-4, gradient clipping 1;
  BF16/TF32, GPU A100 40 GB condivisa, quattro thread CPU per job.
- Loss base: MSE beta + 0,15 PCC per locus + 0,15 mean-proxy.
- Statistiche ausiliarie derivate dal solo blocco train-pazienti × train-CpG;
  target held-out non disponibili ai metodi in training.
- MAS-PCC: mediana, sulle CpG, della correlazione tra pazienti.
  Within-tissue: aggregazione delle metriche nei tessuti con almeno 20 pazienti
  valutati, secondo l'implementazione `pi_lab`.
- Tre viste: nuove persone/CpG note, nuove CpG/persone note, double-OOD.

Training e interazione usano un orizzonte cosine comune di 4.000 step,
warmup 200 step e min-LR ratio 0,1, anche quando il run si ferma a 2.000.
La baseline è allenata fino a 4.000 con checkpoint a 2.000 e 4.000.
I due run RNA a doppio ramo hanno invece orizzonte 2.000 e warmup 100:
si confrontano direttamente tra loro, non con il baseline dell'altra famiglia.

La valutazione finale usa tutte le entità. Le valutazioni intermedie della
baseline limitano a 2.000 i pazienti della vista nuove CpG/persone note.
Per questa vista si usa la rivalutazione completa del checkpoint baseline
2.000 presente nel laboratorio interazione. Per nuove persone e double-OOD,
gli 825 pazienti validation sono già tutti inclusi anche nelle metriche
intermedie. La snapshot distingue queste valutazioni.

## Risultati training: double-OOD a 2.000 step

| Variante | MAS-PCC | Within-tissue | MSE |
|---|---:|---:|---:|
| Baseline | 0,44473 | 0,32492 | 0,03444 |
| Baseline + EMA | 0,45239 | 0,32867 | 0,03339 |
| Loss within-tissue | 0,45086 | 0,33027 | 0,03400 |
| Loss within-tissue + EMA | **0,45650** | **0,33497** | **0,03283** |
| Batch cluster | 0,41411 | 0,31276 | 0,03420 |
| Batch cluster + EMA | 0,41912 | 0,31749 | 0,03319 |
| Batch cluster + loss within-tissue | 0,41565 | 0,31702 | 0,03457 |
| Batch cluster + loss within-tissue + EMA | 0,42184 | 0,32130 | 0,03320 |
| Retrieval FFN 8→2 | 0,44863 | 0,32200 | 0,03427 |
| Retrieval FFN 8→2 + EMA | 0,45254 | 0,32679 | 0,03312 |

La loss within-tissue è aggiunta con peso 0,15: predizioni e target sono
centrati per (tessuto, CpG) nel batch; gruppi con meno di quattro pazienti
sono esclusi. Il successivo calcolo di correlazione aggrega i residui,
non assegna necessariamente peso uniforme ai tessuti.

EMA usa `theta_ema = 0.99 * theta_ema + 0.01 * theta` dopo ogni step,
inizializzata dai pesi iniziali; si valutano sia pesi correnti sia mediati.
Sul baseline a **4.000 step**, double-OOD 0,44738 senza EMA e 0,44872 con
EMA; within-tissue 0,32681 e 0,32820. Non estrapolare il vantaggio intermedio
di EMA al modello convergente.

Il sampler cluster prende una finestra circolare di 256 pazienti da un elenco
ordinato per tessuto, con inizio uniforme. Non duplica pazienti e preserva
esattamente le marginali di inclusione in aspettativa. Nei run raggruppa in
media 2,15 tessuti per batch, contro circa 30,55 del random: è un intervento
forte, diverso dall'idea iniziale di 8 tessuti × 32 pazienti. Le frequenze finite
possono variare e gli stessi vicini ricorrono insieme. Il risultato negativo
non esclude sampler più moderati o batch misti.

La variante FFN2 ha 18.499.490 parametri contro 21.655.970 del baseline.
Nuovi pazienti: 0,49595 contro 0,51599 (senza EMA), quindi il double-OOD simile
non equivale a prestazioni identiche in tutte le viste. I tempi osservati sono
inclusi nella snapshot; il carico concorrente della GPU varia e non permette
di dedurre un fattore di accelerazione controllato.

## Risultati interazione: double-OOD a 2.000 step, senza EMA

| Variante | MAS-PCC | Within-tissue | Delta MAS vs baseline |
|---|---:|---:|---:|
| Baseline | 0,44473 | 0,32492 | — |
| Chiavi statiche | 0,44568 | 0,32246 | +0,00096 |
| Residuo bilineare rank 16 | 0,44088 | 0,32017 | −0,00385 |
| Controllo additivo rank 16 | 0,44500 | 0,32607 | +0,00027 |

- Static: chiavi apprese condivise 64×256, valori ancora derivati dall'RNA.
  Aggiunge 16.384 parametri. Sui nuovi pazienti arriva a 0,52303 contro
  0,51599, ma non migliora chiaramente le CpG nuove né il within-tissue.
- Bilineare: aggiunge al logit un residuo basato sul prodotto di proiezioni
  di `h_c_deep` e della media dei token RNA; rango 16 e uscita inizialmente zero.
  Il controllo additivo usa gli stessi parametri e sostituisce il prodotto
  con una somma scalata. Pesi comuni iniziali e sequenza batch sono preservati.
- Test CPU passati: shape/backward, query CpG identica e predizioni iniziali
  dei due residui esattamente uguali al baseline.

Il risultato riguarda questa specifica forma di pooling e rango: non esclude
interazioni bilineari diverse, FiLM o altri modi di utilizzare gli stessi encoding.

## Risultati RNA: confronto a capacità e scheduler uguali

Due encoder con bottleneck 128+128 producono ciascuno 64 token di larghezza
256; le due banche vengono sommate e divise per sqrt(2). Controllo:
originale/originale; candidato: originale/RNA centrato rispetto alla media
tissutale dei soli pazienti training. Stessa capacità tra i due, ma circa
21,71 M parametri e diversa architettura rispetto al riferimento.

| Variante RNA | MAS-PCC double-OOD | Within-tissue | Nuovi pazienti |
|---|---:|---:|---:|
| Originale/originale | 0,44636 | 0,32186 | 0,49150 |
| Originale/centrato per tessuto | 0,44413 | 0,31559 | 0,48610 |

Nessun beneficio nel test eseguito. Il candidato richiede etichette tissutali
anche in inferenza, mentre il controllo non le usa per costruire l'input.
Tutti i tessuti validation sono presenti nel train; il gruppo `NA` contiene
860 pazienti training e non rappresenta un tessuto biologico. TCGA-OV ha
solo cinque pazienti training: medie di gruppi piccoli sono poco robuste.

### Diagnostica di permutazione dell'RNA

Checkpoint baseline 4.000, 512 CpG casuali fisse per vista e fino a 2.000
pazienti. Si riassegna soltanto l'RNA tra pazienti dello stesso gruppo,
lasciando target e CpG fissi; tre permutazioni con seed 101/102/103,
subset seed 20261017. Nel double-OOD:

- MAS-PCC abbinata: 0,44967; permutata: 0,22179 / 0,20928 / 0,19610.
- Within-tissue abbinata: 0,33104; permutata: 0,00375 / −0,00430 / −0,01155.
- Circa il 96% degli abbinamenti cambia; sono possibili punti fissi.

Il risultato dimostra dipendenza predittiva dall'abbinamento individuale,
non causalità biologica né esclusione di confondenti condivisi. `NA` limita
anche l'interpretazione della permutazione come preservazione del tessuto.
I valori su questo subset non sono confrontabili direttamente con quelli
della valutazione su tutte le CpG.

## Artefatti e riproducibilità

Laboratori originali, esterni alla repo, sotto
`/data2/home/vcivale/projects/methylation/`:

- `training_hypotheses_lab/`: `REPORT.md`, harness, sampler, `run_queue.sh`,
  log, metriche e checkpoint baseline a 2.000/4.000 (con pesi EMA).
- `interaction_hypotheses_lab/`: implementazioni, test CPU, `run_queue.sh`,
  `eval_baseline.py`, `summarize.py`, metriche complete e correlazioni per CpG.
- `rna_hypotheses_lab/`: `run.py`, `queue.sh`, metriche e checkpoint doppio ramo,
  medie tissutali training e diagnostica permutazioni.
- `pi_lab/REPORT.md`: studio precedente su informazione privilegiata,
  retrieval e diagnostiche; i suoi run non sostituiscono il baseline matched.

Gli script richiedono cache ed ambiente CUDA originali. I `run_queue.sh`
originali rilanciano training e possono sovrascrivere output: per una replica
usare directory nuove. La snapshot in repo conserva metriche e provenienza,
non i dati, checkpoint o tutto il codice dei laboratori esterni.

## Conferme da fare

1. Baseline vs loss within-tissue, entrambi con/senza EMA, almeno tre seed e
   stesso budget più lungo; riportare tutte le viste, MSE e within-tissue.
2. Ripetere su un altro split di blocchi genomici. Non selezionare ripetutamente
   sulla stessa validation come se fosse un test indipendente.
3. Confrontare EMA e checkpoint intermedi anche a convergenza; misurare
   eventuali regressioni nella vista nuovi pazienti.
4. Se interessa l'efficienza, confermare FFN2 con pari risorse e testare la sua
   combinazione con loss within-tissue: tale combinazione non è stata provata.
5. Congelamento/rallentamento del ramo CpG e utilizzo diretto della testa
   della media restano ipotesi non testate in questa campagna.

Non integrare automaticamente il candidato nel modello di riferimento:
il vantaggio matched della loss su EMA è piccolo e nessuna variante è stata
validata multi-seed, genome-wide o nel training multisorgente da 80 epoche.
