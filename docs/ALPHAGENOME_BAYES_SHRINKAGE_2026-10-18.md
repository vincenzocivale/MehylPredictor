# Shrinkage empirico-bayesiana sul livello del locus — 18 ottobre 2026

Esperimento interno sull'idea B (prior da foundation model più aggiornamento bayesiano con poche misure). Solo i tre fold genomici interni; dev e ufficiale non toccati. Codice: `scripts/analyze_bayes_shrinkage.py`. Risultati: `artifacts/alphagenome_bayes_shrinkage_2026-10-18/main/` (`results.csv`, `summary.csv`, `protocol.json`).

## In breve

- Il peso `n/(n+tau)` usato finora è la forma del posterior gaussiano con `tau = σ²_rumore / σ²_prior`. Stimato in forma chiusa dai residui di loci esclusi dal fit, dà **tau ≈ 1,7–1,8**, stabile in 9 unità fold×seed (1,68–1,81).
- Il valore congelato `tau=5` era scelto su una griglia {0, 5, 20} troppo grossolana. Con `tau` teorico l'MSE è inferiore in **9/9 unità** a k=1, 5 e 10: −6,7% (k=1), −4,7% (k=5), −2,2% (k=10) rispetto a `tau=5`.
- Rispetto al migliore di una griglia fine, calcolato sugli stessi dati (quindi favorevole alla griglia), il `tau` teorico è entro 0,04% per k≥5. A k=1 è +0,98%.
- Intervalli gaussiani con `tau` globale: copertura del 90% nominale = 89,4–89,6% per ogni k. All'80% nominale la copertura è 83–84%: la calibrazione è solo approssimata e vale a livello marginale.
- L'adattività per locus (`tau_c` da feature senza etichette) non aiuta: entro ±0,5% dal `tau` globale.

## Regole di decisione congelate prima della run

| Regola | Esito | Dettaglio |
| --- | --- | --- |
| P1: MSE del `tau` teorico ≤ 1,005 × minimo di griglia per ogni k | **Non soddisfatta** | Fallisce solo a k=1 (1,0098); k=2 0,997; k≥5 entro ±0,05% |
| P2: `tau` per locus ≤ 0,98 × minimo di griglia a k∈{1,5,10} e meglio in ≥8/9 unità a k=5 | **Non soddisfatta** | Rapporto 0,995–1,006; meglio in 4/9 unità a k=5 |
| P3: copertura 90% locale in [0,87; 0,93] per k≥5 e più vicina al nominale della globale | **Non soddisfatta** | Locale 91,3–91,4% (nel range) ma la globale (89,4–89,5%) è più vicina |

## Tabella principale (media su 9 unità × 5 ripetizioni)

| k | Prior (k=0) | `tau=5` congelato | Migliore griglia | `tau` teorico | `tau` per locus |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0,0257 | 0,02321 | 0,02144 (`tau`=1) | 0,02165 | 0,02158 |
| 2 | 0,0257 | 0,02184 | 0,02058 (`tau`=2) | 0,02052 | 0,02048 |
| 5 | 0,0257 | 0,01960 | 0,01868 (`tau`=1) | 0,01868 | 0,01868 |
| 10 | 0,0257 | 0,01826 | 0,01786 (`tau`=2) | 0,01785 | 0,01785 |
| 20 | 0,0257 | 0,01746 | 0,01732 (`tau`=2) | 0,01732 | 0,01731 |
| 50 | 0,0257 | 0,01681 | 0,01675 (`tau`=1) | 0,01676 | 0,01676 |

Stime: `σ²_rumore` ≈ 0,016, `σ²_prior` ≈ 0,009 (deviazioni standard circa 0,127 e 0,095).

## Limiti

- Vale solo per l'intercetta del locus, con risposta `a_c` e paziente `u_p` fissi. È la shrinkage empirico-bayesiana classica: il risultato è pratico, non una novità metodologica.
- Checkpoint Array-only a 512 kb su fold interni, non il modello misto a 1 Mb; nessun confronto ancora con lo stesso supporto per MethylProphet.
- Il minimo di griglia è scelto sugli stessi dati e quindi è ottimistico. La griglia è discreta (1, 2, 3 …): la superficie è piatta fra `tau` 1 e 2.
- Copertura solo marginale (media su loci e pazienti); non stratificata per rischio del locus, tessuto o fonte. Larghezze non confrontate con baseline (per esempio conformal).
- Un test pipeline su fold0/seed17/1 ripetizione è stato visto prima della run completa, senza modificare le soglie.
