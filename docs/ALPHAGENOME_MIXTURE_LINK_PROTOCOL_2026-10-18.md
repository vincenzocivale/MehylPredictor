# Protocollo A — link a mistura (bulk = media pesata di profili cellulari) — 18 ottobre 2026

**Stato: protocollo congelato, nessun codice eseguito.** Le soglie sotto sono fissate prima di scrivere lo script e di vedere qualunque risultato del braccio nuovo.

## Domanda

Sostituire il link logit-bilineare del v3, `beta = sigmoid(b_c + u_p·a_c/√r)`, con una media convessa in spazio beta, `beta = Σ_k w_k(p) · sigmoid(b_c + a_ck)` con `w = softmax(u_p)`, riduce il divario di generalizzazione a un tessuto escluso dal training?

Perché ci aspettiamo che possa accadere: la metilazione bulk è una mistura di tipi cellulari. Per un tessuto mai visto, `w` può cadere in una nuova regione del simplesso, e l'output resta limitato dai profili `m_k(c)` appresi. Nel logit-bilineare `u_p` fuori distribuzione estrapola senza vincoli. Il costo atteso è espressività: la media convessa non può uscire dal guscio dei `m_k`.

## Disegno

È un'ablation del solo link. Il braccio nuovo (A1) ha **la stessa architettura, gli stessi pesi iniziali, lo stesso ordine dei dati e la stessa ricetta del v3**; cambiano soltanto `forward` e la calibrazione dell'intercetta.

- **Rete:** stessi moduli del v3 (`rank`=32, quindi K=32 componenti). `rna_factor` → logit dei pesi, `cpg_factor` → `a_ck`, `mean_head` → `b_c`.
- **Calibrazione:** il solve monotono per `b_c` usa il nuovo link (resta monotono in `b_c`, quindi la bisezione vale).
- **Penalità locale (peso 0,01):** stessa forma, applicata ai pesi simplex centrati e moltiplicati per K, così che le voci siano O(1) come in v3. È una scelta di progetto fatta prima di eseguire; non viene tarata.
- **Ricetta:** `local01`, 1000 step, batch 256 × 512, AdamW 2e-4, EMA 0,99, seed 17/18/19. Perdite identiche.
- **Baseline A0:** le 12 fit v3 già completate in `artifacts/alphagenome_hypotheses_2026-10-18/tissue/` (`tissue_metrics.csv`). Stessi pazienti, loci, prior e grafo. Non vengono rifatte.
- **Gruppi:** TCGA-BRCA e TCGA-LGG (i due tessuti più numerosi nel training, scelti senza target). Modi: `leave_tissue_out` e `matched_random` (stesso numero di pazienti rimossi a caso).
- **Costo:** 12 fit × circa 30 s.

## Metriche

Popolazione `held_tissue` (70 pazienti BRCA, 46 LGG), vista `novel` e vista `known` (4096 CpG fisse), braccio `hierarchical` come nei report precedenti. `raw` si riporta ma non decide.

- **Divario** `Gap = MSE(leave_tissue_out) − MSE(matched_random)` sul tessuto escluso, per cella (tessuto × vista), media sui seed.
- Correlazione entro tessuto (`within_tissue`) sul tessuto escluso, come misura della risposta individuale.
- MSE su `matched_random` e su `other_tissues`, come controllo di non regressione.

## Regole di decisione

| Regola | Condizione |
| --- | --- |
| **R1 (primaria)** | Media dei 4 `Gap` di A1 ≤ 0,75 × media dei 4 `Gap` di A0, e `Gap` di A1 < A0 in almeno 3 celle su 4 |
| **R2 (nessuna regressione)** | MSE di A1 su `matched_random` ≤ 1,03 × A0 (media delle 4 celle, popolazione `all`) |
| **R3 (risposta individuale)** | `within_tissue` di A1 in `leave_tissue_out` ≥ A0 in almeno 3 celle su 4 |

- **Se R1 e R2 sono soddisfatte:** passa alla fase 2 (K allineati ai tipi cellulari tramite le tracce AlphaGenome, validazione contro la purezza tumorale).
- **Se R1 fallisce:** A viene archiviata come risultato negativo e non viene ritentata con altri K o altre penalità sugli stessi dati.
- **Se R1 passa ma R2 fallisce:** compromesso espressività/robustezza da riportare, senza promozione.

## Analisi descrittive (non decidono)

- Pesi `w` medi per tessuto; similarità del coseno fra `w` del tessuto escluso e ciascun tessuto del training; entropia di `w`.
- Norma di `u_p` fuori distribuzione nel v3 contro l'output limitato nel mistura.
- Per i pesi ↔ purezza tumorale serve un file esterno (stime di purezza TCGA, per esempio CPE/consensus). **Non è nel repository**; i `sample_name` TCGA permettono l'unione. Da scaricare prima della fase 2.

## Limiti dichiarati

- Con 70 e 46 pazienti, e 3 seed, si rilevano solo effetti grandi: i `Gap` osservati nel v3 sono 0,001–0,0065 di MSE, e l'incertezza sulla vista `novel` è la più alta. Nessun intervallo bootstrap con questi conteggi; si riportano i range per seed.
- Gli stessi pazienti dev e blocchi genomici sono già stati esaminati nella campagna precedente. È una conferma di follow-up, non un test intatto. Ufficiale e benchmark MethylProphet non vengono toccati.
- Le K componenti sono latenti: nessuna affermazione di corrispondenza con tipi cellulari prima della validazione esterna di fase 2.
- Checkpoint Array-only a 512 kb, non il modello misto a 1 Mb.
