# Screening del decoder fattorizzato RNA–CpG

Esperimento economico su Array chr1, split `dev` a blocchi genomici da 5 Mb,
seed 17. Ha testato se la CpG può selezionare una risposta all'RNA attraverso
coefficienti appresi dagli input disponibili, comprimendo il calcolo sulle
coppie paziente–CpG.

## Modello e protocollo

Il baseline produce una query CpG e 64 token RNA, quindi esegue attenzione,
otto FFN di retrieval e due FFN finali per ogni coppia paziente–CpG. Il
candidato rank 32 produce invece fattori paziente `u_p` dal bottleneck RNA e
fattori CpG `a_c` dalla rappresentazione funzionale profonda. Predice
`sigmoid(b_c + u_p · a_c / sqrt(32))`, dove `b_c` è il logit della testa della
media prevista. Il candidato usa soltanto RNA e annotazioni disponibili in
inferenza; non usa statistiche osservate della CpG held-out.

Sono rimasti uguali split, seed, batch 256×512, loss MSE + 0,15 PCC per locus +
0,15 loss della media, AdamW, learning rate 2e-4, warmup 200 e scheduler con
orizzonte 4.000 step. Screening fermato a 2.000 step, senza EMA. Il baseline è
la rivalutazione completa del checkpoint seed 17 allo step 2.000; entrambe le
valutazioni finali includono tutti i pazienti e CpG di ciascuna vista.

## Risultati su tre seed

Tabella: differenza appaiata decoder meno baseline per ogni seed.

| Seed | Δ MAS nuovi pazienti | Δ MAS nuove CpG | Δ MAS double-OOD | Δ MSE double-OOD | Δ within-tissue double-OOD |
|---:|---:|---:|---:|---:|---:|
| 17 | −0,0011 | +0,0091 | +0,0057 | +0,00011 | −0,0013 |
| 18 | −0,0000 | +0,0074 | +0,0067 | +0,00064 | +0,0053 |
| 19 | −0,0068 | +0,0053 | +0,0031 | +0,00011 | −0,0003 |
| **Media** | **−0,0026** | **+0,0073** | **+0,0052** | **+0,00029** | **+0,0012** |

La MAS-PCC migliora in tutti e tre i seed per le nuove CpG e per il double-OOD.
Il guadagno è più netto per le nuove CpG. La MSE double-OOD peggiora in ogni
seed, mentre la MAS-PCC sui nuovi pazienti scende in media. Quindi il decoder
sembra migliorare il profilo di risposta specifico della CpG, ma il vantaggio
non è uniforme su tutte le metriche e viste. La media within-tissue double-OOD
è leggermente positiva, con risultati variabili tra seed.

Il risultato ora è coerente su tre seed, ma resta limitato a chr1 e allo split
dev a cinque blocchi; non misura la variabilità tra cromosomi o tra split.

## Costo osservato e calcolo

Il modello schermato ha 15.982.561 parametri contro 21.655.970 del baseline
(−26% registrati). Audit successivo: 4.194.816 parametri del generatore di token
RNA restavano inizializzati ma inutilizzati nel fattorizzato. I parametri attivi
erano quindi 11.787.745 (−45,6%); il nuovo candidato li rimuove esplicitamente.
Nei seed 18 e 19 il training ha richiesto 358–365 secondi al baseline e
31–34 secondi al candidato, circa 11× meno tempo osservato. Il dato viene da
run sequenziali sulla GPU condivisa e non è ancora un benchmark controllato;
non include i tempi di caricamento, preparazione e valutazione.
Il tempo del baseline non è stato misurato in condizioni contemporanee e non
va usato come benchmark diretto.

Le sole proiezioni lineari dei dieci FFN che operano sulle coppie nel baseline
richiedono circa 5.242.880 MAC per coppia. Il candidato rimpiazza quei FFN con
un prodotto di rango 32; i fattori si calcolano una volta per paziente e CpG e
la matrice di interazione richiede 32 MAC per coppia. Questo è un confronto del
decoder, non un rapporto di accelerazione end-to-end: encoder, FFN lato CpG,
attenzione baseline, gestione memoria e throughput reale vanno misurati in una
valutazione controllata.

## Prossimi controlli

Priorità: verificare il risultato su altri blocchi genomici o cromosomi e con
budget pieno a 4.000 step. Se il guadagno regge, confrontare rank 16/64 e
misurare latenza e picco di memoria end-to-end con gli stessi batch. Il
candidato è una possibile semplificazione del decoder, non ancora una modifica
da adottare nel trainer principale.

Il runner isolato e i risultati grezzi sono in
`/data2/home/vcivale/projects/methylation/factorized_decoder_chr1_2026-10-17/`.
Il baseline grezzo è in
`/data2/home/vcivale/projects/methylation/interaction_hypotheses_lab/results/dev/baseline_s2000_h4000/`.
Il riepilogo appaiato per seed è anche salvato in
[JSON](experiments/factorized_decoder_chr1_2026-10-17.json).
