# Secondo screening RNA e training — 17 ottobre 2026

Stato: tutti i nove run conclusi il 17 ottobre 2026 (ultimo output alle 12:53).
Screening a seed singolo, nessuna conferma su training lungo. Questa campagna segue lo
[screening precedente](EXPERIMENTS_RNA_TRAINING_2026-10-17.md).

## Obiettivo e protocollo

Verificare cinque ipotesi tenendo invariati feature e architettura dell'encoder
CpG, senza inferenza AlphaGenome. I pesi CpG vengono allenati normalmente.
L'esperimento resta isolato dal modello di riferimento della repo.

Protocollo: Array chr1, split dev a blocchi da 5 Mb, seed 17, batch casuali
256 pazienti × 512 CpG, 2.000 step, scheduler cosine con orizzonte 4.000 e
warmup 200; AdamW lr 2e-4, wd 1e-4, clipping 1, BF16/TF32. Loss base:
MSE beta + 0,15 PCC per locus + 0,15 mean-proxy. EMA 0,99 in tutti i run.
Valutazione finale sulle tre viste complete, con pesi correnti ed EMA,
MAS-PCC globale, within-tissue e MSE. Controllare le configurazioni salvate
prima di confrontare metriche tra laboratori o campagne.

## Ipotesi e controlli previsti

1. **Media CpG come livello di partenza della predizione.** Logit finale =
   logit della media predetta (clamp per stabilità) + residuo RNA–CpG.
   Residuo inizializzato a zero. Confrontare percorso della media con gradienti
   e con detach; mantenere la supervisione ausiliaria in entrambi. Un controllo
   del regressor standard inizializzato a zero distingue il cambiamento di
   struttura da quello dell'inizializzazione.
2. **Mean-proxy utile soprattutto all'inizio.** Misurare il coseno tra gradienti
   MSE, PCC e mean-proxy sui parametri CpG condivisi, usando un batch training
   fisso a step 0, 1.000 e 2.000. Diagnostica in eval, con stato RNG preservato.
   Separatamente confrontare peso ausiliario costante con decadimento da 0,15
   dopo 500 step a zero a 2.000. Un coseno negativo su un solo batch non prova
   un conflitto persistente né un vantaggio del decadimento.
3. **Bottleneck RNA troppo stretto.** Aumentare 256→512 prima della generazione
   dei token, lasciando 64 token di larghezza 256 e downstream identico.
   La capacità aumenta: un eventuale vantaggio non dimostra da solo che il
   bottleneck sia l'unica causa. Pesi iniziali non-RNA preservati.
4. **Normalizzazione RNA.** La cache `rna_zscore.f16.npy` è già standardizzata
   per gene: verificare metadati e statistiche prima di trasformarla ancora.
   Confrontare rimozione della LayerNorm per paziente con il baseline. Se
   giustificato dall'audit, confrontare anche ricalcolo z-score sui soli pazienti
   dev-train, senza LayerNorm, con floor esplicito per deviazioni quasi nulle.
   Questo controllo distingue l'effetto della LayerNorm da quello delle
   statistiche ricalcolate, non introduce una nuova misura RNA.
5. **Ampiezza delle variazioni individuali.** Aggiungere MSE dei residui dopo
   centratura masked per (tessuto, CpG) nel batch, escludendo gruppi con meno
   di quattro pazienti. Peso 1,0 in scala beta; confrontare con loss base e
   con il precedente controllo wtPCC a peso 0,15. Secondo braccio: combinazione
   delle due loss within-tissue. Nessuna ottimizzazione dei pesi sulla validation.

### Audit RNA e scelta dei run

L'audit conferma 25.017 geni già standardizzati. Sui 7.435 pazienti dev-train
le medie per gene sono tra −0,0130 e +0,0089, le deviazioni standard tra
0,9705 e 1,0273. Nessun gene supera |media| 0,1 o deviazione fuori [0,9; 1,1].
Il ricalcolo z-score senza LayerNorm non viene quindi lanciato: sarebbe un
controllo quasi ridondante. Restano i due run `noLN` e `bottleneck512`.

I metadati dello z-score originale includono 8.260 pazienti: tutti i 7.435
dev-train e gli 825 dev-validation. Il preprocessing preesistente usa quindi
anche RNA dev-validation, senza usare i loro target di metilazione. È una
limitazione comune alle due campagne rispetto a una pipeline strettamente
induttiva; i confronti qui mantengono la stessa cache congelata. Una conferma
rigorosa dovrà rifittare il preprocessing sul solo dev-train per tutti i bracci.
Le nuove statistiche di audit sono calcolate solo su dev-train e il preflight
verifica che alterare RNA validation non le cambi.

Totale previsto: nove run da 2.000 step (baseline e quattro varianti mean,
due varianti RNA, due loss dei residui), oltre alle diagnostiche.

## Esecuzione e artefatti

Root dei laboratori esterni:
`/data2/home/vcivale/projects/methylation/rna_training_round2_2026-10-17/`.

- `mean_training/`: baseline, mean-anchor, controllo inizializzazione,
  decadimento mean-proxy e diagnostiche gradienti.
- `rna/`: audit cache, bottleneck 512 e normalizzazione.
- `residual_loss/`: MSE centrata e combinazione con wtPCC.

Script, preflight, configurazioni, log, checkpoint e risultati sono salvati nei
rispettivi laboratori; cache precedenti lette senza modificarle. Al massimo
due job sperimentali GPU contemporanei: la coda residual-loss aspetta la fine
di quella RNA. I processi estranei alla campagna non vengono toccati.

Questi sono screening single-seed con scheduler non ancora concluso. La scelta
di un metodo richiede confronto matched, eventuali regressioni nelle altre
viste, conferma multi-seed e training più lungo. Nessuna modifica automatica
della ricetta principale è prevista.

## Risultati conclusivi

Metriche double-OOD su tutte le CpG validation, stesso seed e budget, pesi EMA.
I risultati con pesi correnti e tutte le viste sono nella
[snapshot JSON](experiments/rna_training_round2_2026-10-17.json).

| Variante | MAS-PCC | Within-tissue | MSE | Delta MAS vs baseline EMA |
|---|---:|---:|---:|---:|
| Baseline | 0.45239 | 0.32867 | 0.03339 | +0.00000 |
| Mean-anchor detach | 0.45797 | 0.32708 | 0.03260 | +0.00558 |
| Mean-anchor joint | 0.45263 | 0.32739 | 0.03393 | +0.00023 |
| Decadimento mean-proxy | 0.45266 | 0.32879 | 0.03343 | +0.00027 |
| Controllo uscita zero | 0.44892 | 0.32792 | 0.03404 | -0.00347 |
| MSE within-tissue | 0.45441 | 0.32967 | 0.03301 | +0.00202 |
| MSE + PCC within-tissue | 0.45958 | 0.33625 | 0.03256 | +0.00718 |
| Bottleneck RNA 512 | 0.45327 | 0.33207 | 0.03333 | +0.00088 |
| RNA senza LayerNorm | 0.45284 | 0.32575 | 0.03349 | +0.00045 |
| PCC within-tissue (round1 matched) | 0.45650 | 0.33497 | 0.03283 | +0.00411 |

La combinazione MSE+PCC within-tissue è il candidato migliore del secondo
screening: +0,00718 MAS e +0,00757 within-tissue rispetto al baseline EMA,
ma soltanto +0,00307 MAS rispetto a PCC within-tissue+EMA del round1.
Nuovi pazienti: MAS 0,51121 contro 0,51865 del baseline EMA, mentre
within-tissue sale da 0,35332 a 0,35994. Esiste quindi un compromesso tra viste.

Mean-anchor detach guadagna correlazione globale double-OOD ma non
within-tissue (0,32708 contro 0,32867) e peggiora nuovi pazienti
(0,49649 contro 0,51865). Non è un miglioramento uniforme. Joint, mean-decay,
noLN e bottleneck512 non danno guadagni globali double-OOD convincenti.

Nel baseline il coseno gradienti MSE/mean-proxy sul batch diagnostico
è −0,0127 a init, +0,704 a 1000 e +0,740 a 2000; PCC/mean resta positivo
ma piccolo. Non emerge un conflitto sistematico da questa diagnostica locale.
Il decadimento del mean-proxy è sostanzialmente equivalente al baseline.
Il checkpoint baseline raw ed EMA è stato verificato bitwise identico al
round1: le diagnostiche non hanno alterato il percorso di training.

Prossima conferma suggerita: baseline, wtPCC e wtMSE+wtPCC con/senza EMA,
più seed e training più lungo. I delta piccoli non costituiscono una
stima di significatività; non integrare automaticamente nella ricetta principale.
