# Stage B del ramo RNA — scelta della compressione RNA (29 luglio 2026)

Stato: completato. Nessun file in `third_party/MethylProphet` né in
`artifacts/genomic_encoder` è stato toccato. Gli artifact di Stage A
(`artifacts/rna_branch/2026-07-29_first_benchmark/`, incluso `inputs/`) non
sono stati modificati né sovrascritti: Stage B li riusa in sola lettura.
Tutti gli output nuovi vivono sotto `artifacts/rna_branch/stage_b/`.

Riferimento: `docs/rna_branch_first_benchmark.md` (Stage A), che ha stabilito
che il segnale RNA patient-specific reale esiste (`skill_vs_prior≈0,103` su
double-OOD) e che il prossimo passo doveva essere Stage B, non Stage C.

## 1. Cosa è stato tenuto invariato rispetto a Stage A

- Artifact genomici/NTv3 e prior (`inputs/locus_embeddings.h5`,
  `inputs/locus_features.parquet`, colonna `pred_ntv3_prior`).
- Split pazienti (train 7.304 / validation 398 / test 414) e split CpG chr1
  (train 5.243 / validation 978 / test 521) — stessi file parquet di Stage A.
- Interaction module bilineare, gate `variability`, loss (beta MSE +
  residual Huber + shrinkage), sampling (batch pazienti×CpG cartesiano),
  pannelli di valutazione (in-distribution, sample-OOD, locus-OOD, double-OOD),
  metriche e seed (17/29/43).
- Preprocessing di base: standardizzazione RNA fittata solo su train,
  `anchor_to_mean_rna: true`, `zero_init_residual: true`.

L'unica cosa che varia tra le famiglie sperimentali è l'encoder RNA (e, nella
famiglia 5, `training.weight_decay`).

## 2. Modifiche di codice

- **Bug corretto in `pca.py`** (preesistente, non introdotto qui): il dataset
  HDF5 `X` era scritto gzip-chunked, stesso tipo di trappola di performance
  già documentata per le matrici RNA/beta di Stage A (letture a righe casuali
  ~1000× più lente). Rimossa la compressione sul dataset principale.
- **Bug corretto in `pca.py`**: `bundle.samples.ids.astype(str)` produceva un
  array NumPy a larghezza fissa (`<U…`), incompatibile con
  `h5py.string_dtype`; il fit-pca falliva sempre con
  `TypeError: No conversion path for dtype`. Corretto costruendo l'array di
  stringhe come oggetti Python (`np.asarray([str(v) for v in …], dtype=object)`),
  stesso pattern già usato in `prepare_inputs.py`.
- **Nuovo modulo `random_projection.py`** (+ subcomando CLI
  `fit-random-projection`): proiezione casuale gaussiana congelata (stile
  Johnson–Lindenstrauss, seed 20260730, indipendente dai dati di training)
  come controllo a parità di dimensionalità con la PCA. Stesso contratto di
  output di `fit-pca` (`X`/`sample_idx`/`component_ids`, non compresso).
- **`trainer.py`**: aggiunte `num_parameters` e `num_trainable_parameters` a
  `metrics.json` (richiesto per il reporting Stage B).
- **`aggregate_report.py`**: aggiunta `paired_comparison()` (diff per seed
  appaiato rispetto a una famiglia di riferimento + t-test appaiato) dietro
  il flag opzionale `--baseline-family`, non distruttivo per l'uso Stage A
  esistente.

## 3. Famiglie testate (tutte 3 seed 17/29/43, tranne dove indicato)

1. **`baseline_linear`**: encoder lineare pieno (input 21.792 geni,
   `latent_dim=64`), identico alla configurazione vincente `rna_linear_real`
   di Stage A, rieseguito sotto `stage_b/` per il confronto appaiato.
2. **`pca<k>_linear`**, k∈{32,64,128,256,512,1024}: IncrementalPCA
   train-only (`rna_branch fit-pca`, fit solo sui 7.304 pazienti train) →
   proiezione lineare appresa (`encoder.kind=linear`, `latent_dim=k`).
   Varianza spiegata cumulativa: 32→65,4%, 64→71,7%, 128→77,1%, 256→81,6%,
   512→85,4%, 1024→89,6%.
3. **`pca512_mlp` / `pca1024_mlp`**: MLP piccola (`hidden_dims=[128]`,
   `latent_dim=64`) sulle due configurazioni PCA migliori (selezionate su
   `skill_vs_prior` double-OOD: 1024→0,0956, 512→0,0944, entrambe sopra
   256→0,0898).
4. **`bottleneck<k>`**, k∈{16,32,128,256}: bottleneck lineare supervisionato
   end-to-end sull'RNA pieno (nessuna riduzione PCA/random a monte),
   `latent_dim=k`. k=64 coincide esattamente con `baseline_linear` e non è
   duplicato.
5. **`wd<v>`**, v∈{0, 1e-5, 1e-3, 1e-2}: sweep di `weight_decay` sul modello
   lineare pieno. v=1e-4 (il default) non è duplicato: è `baseline_linear`.
6. **`randomproj<k>_linear`**, k∈{32,64,128,256,512,1024}: controllo a
   proiezione casuale congelata, stessa architettura a valle di 2.

PCA, standardizzazione e ogni trasformazione sono fittate esclusivamente sui
pazienti train; oggetti e manifest salvati in
`artifacts/rna_branch/stage_b/features/{pca,randomproj}_<k>.{h5,json}`. Split
pazienti/CpG mai modificati.

## 4. Tabella riassuntiva (media su 3 seed; deviazione standard su
   `skill_vs_prior`; tabella completa in `summary_table.csv`/`.raw.csv`)

### Pannello double_ood (primario)

| famiglia | mse | mae | skill_vs_prior | dynamic_skill | within_cancer_skill | calib. alpha | amp_ratio | macro_cancer_skill | n° parametri | tempo (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_linear | 0,02455 | 0,09294 | **0,1033±0,0027** | 0,1891 | 0,1370 | 0,928 | 0,470 | 0,1087 | 1.748.229 | 127 |
| pca032_linear | 0,02640 | 0,09639 | 0,0357±0,0046 | 0,1097 | 0,0378 | 0,925 | 0,360 | 0,0306 | 306.853 | 38 |
| pca064_linear | 0,02553 | 0,09456 | 0,0677±0,0043 | 0,1552 | 0,0843 | 0,971 | 0,407 | 0,0712 | 314.181 | 39 |
| pca128_linear | 0,02507 | 0,09352 | 0,0845±0,0040 | 0,1765 | 0,1137 | 0,948 | 0,445 | 0,0905 | 334.981 | 60 |
| pca256_linear | 0,02492 | 0,09316 | 0,0898±0,0130 | 0,1888 | 0,1308 | 0,935 | 0,469 | 0,0953 | 401.157 | 57 |
| pca512_linear | 0,02480 | 0,09332 | 0,0944±0,0113 | 0,1847 | 0,1329 | 0,905 | 0,479 | 0,1000 | 631.813 | 68 |
| pca1024_linear | 0,02476 | 0,09294 | 0,0956±0,0051 | 0,1896 | 0,1412 | 0,829 | 0,538 | 0,1031 | 1.486.341 | 84 |
| pca512_mlp | 0,02529 | 0,09406 | 0,0763±0,0025 | 0,1774 | 0,1242 | 0,949 | 0,445 | 0,0810 | 384.069 | 74 |
| pca1024_mlp | 0,02501 | 0,09362 | 0,0864±0,0067 | 0,1842 | 0,1322 | 0,955 | 0,450 | 0,0910 | 449.605 | 87 |
| bottleneck016 | 0,02499 | 0,09359 | 0,0873±0,0055 | 0,1787 | 0,1309 | 0,953 | 0,445 | 0,0934 | 695.925 | 198 |
| bottleneck032 | 0,02475 | 0,09294 | 0,0959±0,0032 | 0,1893 | 0,1378 | 0,928 | 0,471 | 0,1035 | 1.046.693 | 122 |
| bottleneck128 | 0,02486 | 0,09306 | 0,0920±0,0038 | 0,1859 | 0,1374 | 0,881 | 0,494 | 0,0987 | 3.151.301 | 82 |
| bottleneck256 | 0,02485 | 0,09314 | 0,0924±0,0105 | 0,1907 | 0,1403 | 0,911 | 0,484 | 0,0986 | 5.957.445 | 81 |
| randomproj032_linear | 0,02727 | 0,09840 | 0,0039±0,0050 | 0,0611 | 0,0049 | 0,896 | 0,278 | 0,0042 | 306.853 | 53 |
| randomproj064_linear | 0,02671 | 0,09735 | 0,0245±0,0029 | 0,0902 | 0,0144 | 0,910 | 0,332 | 0,0235 | 314.181 | 54 |
| randomproj128_linear | 0,02627 | 0,09593 | 0,0405±0,0048 | 0,1175 | 0,0367 | 0,889 | 0,390 | 0,0429 | 334.981 | 55 |
| randomproj256_linear | 0,02584 | 0,09515 | 0,0562±0,0088 | 0,1379 | 0,0615 | 0,937 | 0,397 | 0,0603 | 401.157 | 57 |
| randomproj512_linear | 0,02569 | 0,09479 | 0,0616±0,0189 | 0,1535 | 0,0893 | 0,894 | 0,443 | 0,0676 | 631.813 | 68 |
| randomproj1024_linear | 0,02524 | 0,09427 | 0,0780±0,0109 | 0,1723 | 0,1145 | 0,952 | 0,441 | 0,0808 | 1.486.341 | 84 |
| wd0 (weight_decay=0) | 0,02455 | 0,09294 | 0,1033±0,0027 | 0,1891 | 0,1370 | 0,928 | 0,470 | 0,1087 | 1.748.229 | 78 |
| wd1e-3 | 0,02455 | 0,09294 | 0,1033±0,0028 | 0,1891 | 0,1370 | 0,928 | 0,470 | 0,1087 | 1.748.229 | 79 |
| wd1e-2 | 0,02455 | 0,09294 | 0,1034±0,0027 | 0,1891 | 0,1370 | 0,928 | 0,470 | 0,1088 | 1.748.229 | 72 |

Pannello `sample_ood` (diagnostica di generalizzazione paziente più diretta):
`baseline_linear` 0,1964±0,0071; `pca1024_linear` 0,2072±0,0053;
`bottleneck256` 0,2027±0,0112; `pca512_linear` 0,1953±0,0245;
`randomproj1024_linear` 0,1917±0,0306 (tabella completa in `summary_table.csv`).
Su questo pannello alcune varianti a rango alto sono nominalmente sopra la
baseline, ma **il pannello primario di decisione resta double_ood**
(paziente *e* CpG mai visti), dove nessuna variante compressa supera la
baseline (vedi §5).

Pearson/Spearman dei residui dinamici (double_ood): baseline 0,436/0,368;
pca1024_linear 0,445/0,386 (leggermente sopra, entro rumore); bottleneck032
0,437/0,371; pca512_mlp 0,422/0,359; randomproj032_linear 0,249/0,203 (molto
più debole). Pattern coerente con `skill_vs_prior`.

## 5. Confronti appaiati vs `baseline_linear` (per seed, double_ood;
   tabella completa in `summary_table.paired.csv`)

| famiglia | diff medio skill_vs_prior | std diff | p (t-test appaiato, n=3) |
|---|---:|---:|---:|
| pca512_linear | -0,0090 | 0,0086 | 0,215 |
| pca1024_linear | -0,0077 | 0,0076 | 0,222 |
| pca512_mlp | -0,0270 | 0,0053 | 0,012 |
| pca1024_mlp | -0,0169 | 0,0083 | 0,071 |
| bottleneck016 | -0,0160 | 0,0068 | 0,055 |
| bottleneck032 | -0,0074 | 0,0007 | 0,003 |
| bottleneck128 | -0,0113 | 0,0045 | 0,050 |
| bottleneck256 | -0,0109 | 0,0091 | 0,173 |
| randomproj1024_linear | -0,0253 | 0,0082 | 0,033 |
| wd1e-2 | +0,00006 | 0,00004 | 0,119 |
| wd1e-3 | +0,00001 | 0,00002 | 0,418 |

Con n=3 il test ha potenza debole (va letto come descrittivo, non come
gate autonomo): il segnale primario è che **il diff medio è negativo per
ogni singola famiglia compressa**, in ogni singolo seed appaiato — nessuna
variante supera mai la baseline, nemmeno una volta su tre. `bottleneck032` è
l'unica differenza piccola *e* stabile tra i seed (std 0,0007) — la
compressione supervisionata a rango 32 è la più vicina alla baseline, ma
resta comunque sistematicamente sotto.

## 6. Sweep weight-decay: nessun effetto misurabile

`wd0`, `wd1e-3`, `wd1e-2` (e il default `wd=1e-4` = `baseline_linear`) danno
metriche **identiche fino alla 3ª–5ª cifra decimale** (`skill_vs_prior`
double_ood: 0,10332–0,10339 su tutto il range 0–100× del default). Causa
diagnosticata: il checkpoint migliore viene selezionato a `best_epoch=2` in
tutte le run (early stopping su validation MSE), quindi il weight decay
decoupled di AdamW (`lr·wd`≈4×10⁻⁸ per step al default) non ha il tempo di
accumulare un effetto misurabile prima che il training si fermi — non è un
bug di esecuzione (verificato: `config.yaml` salvato per ogni run riflette
il `weight_decay` corretto). Conclusione: **la regolarizzazione non è il
fattore limitante nel regime di training attuale**; il collo di bottiglia è
altrove (si veda §8).

## 7. Terzili di variabilità del CpG (double_ood, seed 17;
   `tertile_reports/*.json`)

| famiglia | terzile basso | terzile medio | terzile alto |
|---|---:|---:|---:|
| baseline_linear | +0,0028 | +0,0676 | +0,1237 |
| pca512_linear | -0,0017 | +0,0697 | +0,1211 |
| pca1024_linear | +0,0027 | +0,0652 | +0,1004 |
| pca1024_mlp | +0,0019 | +0,0420 | +0,0973 |
| pca512_mlp | -0,0093 | +0,0739 | +0,0753 |
| bottleneck032 | +0,0056 | +0,0605 | +0,1169 |
| bottleneck256 | +0,0032 | +0,0550 | +0,1126 |

Nessuna famiglia mostra una **sistematica** degradazione dei CpG a bassa
variabilità (i valori sul terzile basso restano vicini a zero per tutte le
configurazioni vicine alla baseline; `pca512_mlp` ha il valore più negativo,
-0,0093, coerente con l'essere la configurazione complessivamente più
debole, non con un pattern di sovra-correzione specifico). Il gate
`variability` continua a fare il suo lavoro in ogni configurazione testata.

## 8. Pathway pooling: non eseguito

Non ci sono annotazioni gene→pathway già disponibili localmente (verificato:
nessun file `.gmt`/MSigDB/Reactome/KEGG sotto
`/data/dataset/methylation/MethylProphetData/` né nel repository). Introdurle
richiederebbe una nuova dipendenza esterna (download di un gene-set
database, mapping gene-symbol↔Ensembl id, gestione versioni) — esattamente
il tipo di "dipendenza fragile" che l'istruzione originale chiedeva di
evitare in questa fase. Skippato per Stage B; da riconsiderare solo se in
futuro un'annotazione pathway diventa un artifact già scaricato e verificato
altrove nel progetto.

## 9. Criteri di accettazione applicati

Un encoder compresso è "genuinamente migliore" solo se soddisfa **tutti**:

1. Migliora `skill_vs_prior` double_ood oltre la variabilità tra seed —
   **nessuna famiglia lo soddisfa**: ogni diff appaiato vs baseline è
   negativo (§5).
2. Mantiene o migliora `within_cancer_skill` — `bottleneck032` (+0,0008) e
   `pca1024_linear` (+0,0043) lo soddisfarebbero da soli, ma falliscono già
   il criterio 1.
3. Nessuna sovra-amplificazione — soddisfatto da tutte le famiglie vicine
   alla baseline (`amp_ratio` 0,44–0,54, tutte sotto 1, nessun pattern
   opposto a MethylProphet).
4. Nessun peggioramento sistematico dei CpG a bassa variabilità —
   soddisfatto (§7).
5. Continua a superare `cancer_type_only` e `shuffle_within_cancer` — tutte
   le famiglie reali (non-random-projection, non-shuffle) restano ben sopra
   entrambi i controlli di Stage A (0,035 e 0,021 rispettivamente su
   double_ood); anche la configurazione compressa più debole testata qui con
   segnale reale (`pca032_linear`, 0,036) è al livello di `cancer_type_only`,
   non sotto.

**Nessuna famiglia soddisfa il criterio 1**, condizione necessaria dichiarata
in partenza. Questo chiude la domanda per tutte e sei le famiglie.

## 10. Decisione

**Mantenere il modello lineare completo** (`encoder.kind=linear`,
`latent_dim=64` su RNA pieno, `weight_decay=1e-4`) come configurazione
Stage B di riferimento. Motivazioni:

- Nessuna PCA (32–1024 componenti), nessuna proiezione casuale, nessun
  bottleneck supervisionato (16–256), nessuna MLP-su-PCA supera la baseline
  su double_ood oltre la variabilità tra seed (§5, §9 criterio 1).
- La compressione supervisionata end-to-end (`bottleneck032`) è la più
  vicina (differenza piccola e stabile, -0,0074±0,0007) ma resta comunque
  sistematicamente sotto la baseline in tutti e 3 i seed — non c'è
  evidenza che comprimere l'RNA a un rango inferiore a 64 aiuti la
  generalizzazione.
- La PCA non struttura mai meglio del semplice accesso diretto ai 21.792
  geni: anche a 1024 componenti (89,6% varianza spiegata) resta sotto la
  baseline; il gap PCA→random-projection a parità di rango (es. rango 1024:
  0,096 vs 0,078) conferma che la struttura PCA aiuta rispetto a una
  proiezione arbitraria, ma non abbastanza da recuperare il gap verso
  l'accesso diretto.
- Lo sweep weight-decay non mostra alcun effetto misurabile nel regime di
  training attuale (early stopping a ~epoca 2): la regolarizzazione non è il
  fattore limitante, quindi non è un percorso di miglioramento nemmeno per
  la baseline stessa.
- Nessun guadagno di parametri/tempo di training giustifica comunque uno
  scambio verso una configurazione più debole: tutte le run (baseline
  incluso) convergono già in 40–200 secondi su GPU singola; non c'è un
  problema di costo computazionale da risolvere con la compressione.

**Non si procede a Stage C** (encoder RNA complessi: Perceiver, Fourier,
FiLM, cross-attention). La domanda posta a Stage B era se la compressione
RNA fosse il fattore limitante rispetto al modello lineare pieno; la
risposta è no — comprimere (in modo lineare, non lineare via MLP, o con
proiezione casuale) non recupera mai le prestazioni del modello lineare
pieno, figuriamoci superarle. Combinato con il risultato di Stage A (la MLP
non batte il lineare sull'RNA pieno), l'evidenza converge: la scelta della
rappresentazione RNA (lineare vs non lineare, compressa vs piena) non è
attualmente il collo di bottiglia del ramo RNA. Investire in architetture
più complesse per l'encoder RNA senza nuova evidenza che la capacità di
rappresentazione sia il limite non è giustificato; un collo di bottiglia più
probabile è altrove nella pipeline (rumore nei target beta-space, capacità
dell'interazione bilineare, o segnale RNA intrinsecamente limitato per i
CpG di chr1 osservati) — non indagato qui, fuori scope di Stage B.

## 11. Artifact

```text
artifacts/rna_branch/stage_b/
  features/{pca,randomproj}_<k>.{h5,json}     # trasformazioni congelate
  generated_configs/<grid>/*.yaml             # 69 config materializzate
  screening/<run_name>/                       # config.yaml, manifest.json,
                                               # metrics.json, training_history.csv,
                                               # best.pt, predictions_*.npz (seed17)
  tertile_reports/*.json
  summary_table.csv / .raw.csv / .paired.csv
  run_stage_b_grid_parallel.sh
```

69 run totali (63 + 6 PCA+MLP), 0 falliti, stesso manifest di input
condiviso (`artifacts/rna_branch/2026-07-29_first_benchmark/inputs/manifest.json`)
per ogni run.
