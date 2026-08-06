# Audit finale artefatti MethylProphet — valutazione matched chr1

Stato conclusivo: **`READY_FOR_MATCHED_TEST`**

Questo audit non ha aperto il test del nostro modello, non ha calcolato
metriche finali e non ha modificato training, loss, checkpoint selezionato
(B3, epoca 28, seed 17) o il protocollo congelato. Non ha toccato alcun file
sotto `outputs/`. Il submodule `third_party/MethylProphet` resta pulito al
commit pinnato `b24f5af3c7b4d6aa2689950e2ea4e3b2bcc8ddfd`.

## 1. Ricerca locale e Hugging Face (`MethylProphet`, `xk-huang`)

Tutti gli artefatti rilevanti sono **già presenti localmente**; non è stata
necessaria alcuna nuova richiesta di rete. La cache Hugging Face conferma
quattro repository scaricati in precedenza:

| repo HF | revision | ruolo |
|---|---|---|
| `MethylProphet/tcga_mix_chr1-bs_512-c2b2` | `e3bf682757b37e681c292221c5aaf5456e9949f5` | checkpoint ufficiale |
| `MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2` | `c3a4e7e7f5cbfd1373aadc155d6bef7e1e6dcd9e` | predizioni rilasciate |
| `MethylProphet/methylprophet-example_data-tcga_mix_chr1` | `ede7ee12e390be984240eb23fec3603038ad994e` | dati esempio (input, non predizioni) |
| `xk-huang/methylprophet-example_data-tcga_mix_chr1` | `ede7ee12e390be984240eb23fec3603038ad994e` | stesso repo, stessa revision, mirror sotto l'altro namespace |

Nota: la cache HF dei primi due repo contiene solo puntatori/ref (12KB e
214MB rispettivamente, molto meno del contenuto reale) — le copie di lavoro
effettive vivono già sotto `artifacts/cache/methylprophet/upstream_outputs/`
(scaricate in una sessione precedente con un meccanismo diverso dal cache
standard `hf_hub_download`, verosimilmente `snapshot_download` con
`local_dir` esplicito). Non è stato necessario ri-scaricare nulla.

Nessun altro checkpoint o file di predizioni rilasciate per
`tcga_mix_chr1-bs_512-c2b2` è stato trovato in nessuno dei due namespace.

## 2. Provenienza ricostruita (Stage D1/D2/D3)

Gli stessi artefatti sono già stati usati e validati in stage precedenti:

- `src/methylation_predictor/diagnostics/methylprophet/stage_d1.py` legge
  esattamente `eval_results-test.parquet` filtrando `group_idx==2`
  (`_released_panel`, default `group=2`).
- `artifacts/cache/methylprophet/upstream_outputs/reproducibility/tcga/AUDIT_REPORT.md`
  contiene un audit di riproducibilità pre-esistente: le PCC ricalcolate con
  `src.eval.compute_pcc_by_group` upstream (mai reimplementato, per la regola
  del confine in `upstream.py`) sono vicine ma non identiche alla tabella
  ICLR (`pass_mas_pcc: False` su tutte e tre le partizioni) — gap
  documentato e intenzionale, non un bug (vedi `docs/methylprophet_diagnosis.md`
  per il precedente).
- `artifacts/cache/methylprophet/upstream_outputs/reproducibility/tcga/split_summary.json`
  fornisce sha256 già calcolati per le quattro tabelle di split
  (`train_cpg` n=33885, `val_cpg` n=6742, `train_sample` n=8260,
  `val_sample` n=918) estratte dallo stesso file di predizioni.
- `docs/stage_d_matched_evaluation.md` documenta già il comando
  `extract-checkpoint-predictions` che consuma lo stesso
  `eval_results-test.parquet` come "evidenza primaria" preferita a una nuova
  inferenza.
- `docs/reproducibility.md` fissa il commit upstream pinnato.

Nessuna incoerenza trovata tra queste fonti; il file candidato C1 (sotto) è
lo stesso già usato, con provenienza tracciata end-to-end.

## 3. Candidati parquet ispezionati

Tabella completa in `artifacts/methylprophet_audit/candidates.tsv`. Sintesi:

### C1 — RACCOMANDATO
`artifacts/cache/methylprophet/upstream_outputs/eval/eval-tcga_mix_chr1-bs_512-c2b2/eval_results-test.parquet`
(6 shard parquet, dataset pyarrow)

- Colonne: `mse_loss_per_point, pred_methyl, gt_methyl, cpg_idx, sample_idx, group_idx`.
- Shape: 91.859.614 righe totali. `group_idx=0` (train_cpg-val_sample):
  30.574.946; `group_idx=1` (val_cpg-train_sample): 55.155.121; `group_idx=2`
  (val_cpg-val_sample, **il pannello doppiamente held-out che serve qui**):
  6.129.547.
- Sulla fetta `group_idx=2`: 918 sample_idx unici (`min=1417, max=10701`),
  6742 cpg_idx unici (`min=8943035, max=10946991` — esattamente il nostro
  intero universo CpG chr1) — **0 coppie (sample, cpg) duplicate**.
- Copertura vs pannello canonico: verificata riga per riga contro il nostro
  bundle (`load_bundle` sul config B3 congelato, split reali, nessun
  campionamento):
  - **test** (414 sample × 521 cpg, 213.091 celle osservate): **100.00%**
    coperte, 0 mancanti, `gt_methyl` vs target canonico **max_abs_error=0.0**
    esatto su tutte le 213.091 righe.
  - **validation** (398 sample × 978 cpg, 383.999 celle osservate):
    **100.00%** coperte, 0 mancanti, stesso controllo **max_abs_error=0.0**
    esatto su tutte le 383.999 righe.
- `pred_methyl`/`gt_methyl` entrambi presenti, finiti, range plausibile
  (`pred_methyl` in [0.0065, 0.992], `gt_methyl` in [0.0, 0.995]).
- Nessun mismatch trovato tra ground truth ufficiale e target canonico.

**Perché lo stesso file serve sia per il test sia per la calibrazione
validation**: le nostre CpG di validation (978) e di test (521) sono
disgiunte per costruzione (`cpg_split_manifest.parquet`), così come i nostri
sample di validation (398) e di test (414). Le righe usate per il fit di
alpha (validation) e quelle usate per l'headline (test) non si sovrappongono
mai, anche se estratte dallo stesso file/gruppo — non c'è leakage.

### Candidati scartati (C2–C4, dettagli in candidates.tsv)

- **C2** `eval-encode_wgbs-bs_512-64xl40s-aws` — dataset ENCODE WGBS, non
  TCGA array chr1: universo sample/tessuto incompatibile con questo
  checkpoint, fuori scope.
- **C3** `third_party/MethylProphet/data/examples/tcga_mix_chr1/` (+ mirror
  HF `MethylProphet`/`xk-huang`) — bundle di input di esempio/smoke (gene
  expression, shard MDS tokenizzati), **non contiene `pred_methyl`/`gt_methyl`**:
  non è un candidato di predizioni.
- **C4** `artifacts/cache/source/tcga_example_data/` — sottoinsieme
  smoke-test locale, non rappresentativo (vedi memoria `repo_data_layout`:
  ~10916 sample vs le 9195 ufficiali, md5 diverso), nessuna riga di
  predizione.
- **C5** (supporto, non candidato di predizioni) — le quattro tabelle di
  split sha256-verificate già estratte da C1 in un audit precedente;
  confermano semanticamente `group_idx` senza bisogno di ricalcolo.

## 4. Set di artefatti raccomandato (uno solo)

Dettagli machine-readable in `artifacts/methylprophet_audit/recommended_artifacts.json`.

| ruolo | percorso | hash |
|---|---|---|
| checkpoint ufficiale | `artifacts/cache/methylprophet/upstream_outputs/ckpts/tcga_mix_chr1-bs_512-c2b2/version_0/finished.ckpt` | sha256 `970b8b1f…3d11` |
| predizioni test (raw, headline) | `.../eval/eval-tcga_mix_chr1-bs_512-c2b2/eval_results-test.parquet`, `group_idx=2` | shard sha256 in JSON |
| predizioni validation (calibrazione, opzionale) | stesso file, stesso `group_idx=2` | idem |
| checkpoint nostro (B3 congelato) | `artifacts/rna_branch/chr1_biological_fidelity/b3_skill_within/b3_locus_skill_plus_within_cancer/best.pt` (epoca 28) | sha256 `15753406…7043` |
| coordinate CpG (bootstrap a blocchi genomici) | `artifacts/methylprophet_audit/cpg_chr_pos_chr1_6742.parquet` (derivato da `.../241231-tcga_mix/cpg_chr_pos_df.parquet`, riga = `cpg_idx`, filtrato al nostro universo di 6742 CpG, tutte confermate `chr1`) | sha256 `da0e6905…7da09` |

**Mapping sample_idx→sample_id e cpg_idx→coordinata**: nessuna mappa di
stringhe è necessaria per l'allineamento. Il campo `.ids` del nostro
`DataBundle` (sia per i sample sia per i CpG) è già la rappresentazione
stringa dello stesso spazio intero `sample_idx`/`cpg_idx` usato dalle righe
rilasciate — verificato empiricamente sopra (copertura 100%, 0 righe
mancanti, senza applicare alcun `--mp-sample-map`/`--mp-cpg-map`). Una
tabella `sample_idx→sample_name` esiste comunque per riferimento opzionale
(`/data/dataset/methylation/MethylProphetData/processed/241231-tcga/sample_tissue_count_with_idx.csv`)
ma non serve per questo comando. La mappa `cpg_idx→chr/pos` **serve solo**
per il bootstrap gerarchico a blocchi genomici da 5 Mb, non per
l'allineamento pred/gt; il file scoperto e verificato è
`.../241231-tcga_mix/cpg_chr_pos_df.parquet` (convenzione riga=cpg_idx, già
documentata in memoria e ri-verificata qui su tutte le 6742 righe).

## 5. Nessuna inferenza ufficiale necessaria

Le predizioni rilasciate sono complete e verificate bit-esatte
(§3, C1) sia per il pannello di test sia per quello di validation — l'item 6
della richiesta (documentare un piano di inferenza) **non si applica**: il
checkpoint scaricabile esiste comunque localmente
(`finished.ckpt`, sha256 sopra) e, se in futuro servisse rieseguirlo,
`third_party/MethylProphet/docs/EVAL.md` documenta il comando ufficiale
(`weight_path=.../finished.ckpt`) — non è stato necessario consultarlo oltre
questo puntatore.

## 6. Comando finale pronto (non eseguito in questo turno)

```bash
conda run -n methil-predictor python scripts/rna_branch/evaluate_chr1_biological_fidelity.py \
  --our-config artifacts/rna_branch/generated_configs/chr1_biological_fidelity/03_b3_locus_skill_plus_within_cancer.yaml \
  --our-checkpoint artifacts/rna_branch/chr1_biological_fidelity/b3_skill_within/b3_locus_skill_plus_within_cancer/best.pt \
  --methylprophet-checkpoint artifacts/cache/methylprophet/upstream_outputs/ckpts/tcga_mix_chr1-bs_512-c2b2/version_0/finished.ckpt \
  --methylprophet-predictions artifacts/cache/methylprophet/upstream_outputs/eval/eval-tcga_mix_chr1-bs_512-c2b2/eval_results-test.parquet \
  --methylprophet-group-idx 2 \
  --methylprophet-validation-predictions artifacts/cache/methylprophet/upstream_outputs/eval/eval-tcga_mix_chr1-bs_512-c2b2/eval_results-test.parquet \
  --methylprophet-validation-group-idx 2 \
  --calibration-objective mse \
  --cpg-coordinates artifacts/methylprophet_audit/cpg_chr_pos_chr1_6742.parquet \
  --coordinate-id-column cpg_idx \
  --chromosome-column chr \
  --position-column pos \
  --genomic-block-size 5000000 \
  --bootstrap-replicates 2000 \
  --output-dir artifacts/rna_branch/chr1_biological_fidelity/test_comparison
```

Note sul comando:

- `--mp-sample-map`/`--mp-cpg-map` omessi deliberatamente (§4): non servono,
  e passarli senza necessità rischierebbe di introdurre un mapping non
  verificato.
- `--allow-partial-overlap` **non** incluso, come da protocollo (copertura
  già confermata 100%, non serve la modalità esplorativa).
- `--dmr-region-annotation` **non** incluso: nessuna annotazione di regione
  congelata indipendentemente è stata identificata in questo audit; se ne
  esiste una da usare, va aggiunta esplicitamente prima di riportare
  l'analisi DMR (resta comunque un livello di validazione biologica
  opzionale, non necessario per l'headline).
- `--run-downstream` **non** incluso di default; aggiungerlo è sicuro
  (richiede solo `scikit-learn`) ma resta un layer opzionale.

## Stato conclusivo

**`READY_FOR_MATCHED_TEST`** — checkpoint ufficiale, predizioni test e
predizioni per la calibrazione validation sono tutti locali, verificati
bit-esatti contro il target canonico, con copertura 100% e zero duplicati.
Nessun mapping è stato inventato: l'allineamento sample/cpg usa lo stesso
spazio di indici già condiviso, verificato empiricamente; la mappa
coordinate-CpG usata per il bootstrap a blocchi è una tabella ufficiale
MethylProphet già presente su disco, con la convenzione riga=cpg_idx
ri-verificata su tutte le 6742 righe del nostro universo. Il comando in §6 è
pronto ma **non è stato eseguito** in questo turno.
