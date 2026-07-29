# Primo benchmark del ramo RNA — 29 luglio 2026

Stato: completato. Nessun file in `third_party/MethylProphet` né in
`artifacts/genomic_encoder` è stato modificato. NTv3, gli embedding genomici,
il prior locus-specific e le teste di variabilità restano frozen; il ramo RNA
predice solo la correzione patient-specific rispetto al prior.

Run directory: `artifacts/rna_branch/2026-07-29_first_benchmark/`.

## 1. Modifiche effettuate

- Installato l'overlay `MethylProphetTest-rna-experiments/` nella repo
  (`src/methylation_predictor/rna_branch/`, `configs/rna_branch/`,
  `docs/rna_branch_experiments.md`, `jobs/slurm/run_rna_grid.sh`,
  `tests/test_rna_branch_*.py`, `requirements-rna.txt`).
- `metrics.py`: aggiunte `dynamic_spearman` e `within_cancer_spearman`
  (Spearman dei residui), assenti nell'overlay ma richieste come metrica
  obbligatoria. Nessun'altra chiave esistente modificata.
- Nuovo modulo `src/methylation_predictor/rna_branch/prepare_inputs.py`:
  conversione one-time e deterministica dai veri artifact TCGA-array
  (`/data/dataset/methylation/MethylProphetData/`) e dagli artifact NTv3
  frozen (`artifacts/genomic_encoder/...`) nel contratto canonico richiesto
  da `docs/rna_branch_experiments.md`. Non importa codice di
  `genomic_encoder`/`diagnostics` (isolamento di dominio), non scrive né
  rilegge nulla sotto `third_party/MethylProphet`.
- Nuovo modulo `aggregate_report.py` (tabella comparativa multi-seed) e
  `tertile_report.py` (breakdown per terzile di variabilità dalle predizioni
  salvate).
- `configs/rna_branch/base.yaml` adattato ai path reali generati.

## 2. Provenienza dei dati (dettaglio in `inputs/manifest.json`)

- **CpG (6.742, chr1)**: universo e split train/validation/test (5.243/978/521)
  identici a quelli già usati per la selezione del prior/variabilità NTv3
  (`artifacts/genomic_encoder/ntv3_prior/split_manifests/`), 5 Mb block,
  seed 20260728. **Scoperta rilevante**: questi 6.742 CpG di chr1 risiedono
  interamente nella partizione `val_cpg` ufficiale di MethylProphet (0 righe
  lette da entrambe le partizioni `train_cpg-*` di `me_cpg_bg`) — coerente con
  la nota "chr1-only exposure" di `docs/decisions/genomic_encoder_selection.md`.
- **Pazienti (8.116)**: split ufficiale MethylProphet `ind_cancer`
  (train=8.260, val=918, senza overlap) da
  `MethylProphetData/.../subset_sample_split/ind_cancer/`, filtrato ai soli
  pazienti con `cancer_type` reale (esclusi 956+106 campioni `NA`, verosimilmente
  tessuto non tumorale/non etichettato). Il pool train ufficiale (7.304 dopo
  filtro) resta intatto; il pool val ufficiale (812 dopo filtro) è stato
  diviso 50/50 in validation/test, stratificato per cancer_type, seed 20260729
  — è l'unica suddivisione **nuova** introdotta, perché MethylProphet non ha
  mai avuto un terzo split di pazienti.
- **RNA**: `gene_expr.filtered.parquet` ufficiale della variante
  `241231-tcga_array-index_files-ind_cancer` (21.792 geni, valori già
  log-trasformati), trasposto samples×genes.
- **Beta**: estratto per filtro predicato da `me_cpg_bg/{train,val}_cpg-{train,val}_sample.parquet`
  (dataset lungo cpg_idx/sample_idx/methylation), densificato
  8.116×6.742, copertura 99,00% (0,9975% mancante, NaN).
- **Prior NTv3** (`pred_ntv3_prior`): per CpG validation/test riusa
  direttamente `pred_mlp_ensemble` già congelato
  (`final_NTv3_650M_post/.../locus_predictions.parquet`); per CpG train (mai
  scritte altrove, per evitare leakage in-sample) rifittati 5-fold
  out-of-fold con la stessa ricetta congelata (Ridge non usato in ensemble,
  MLP `LayerNorm→256→64→1`, 3 seed 17/29/43, 500 epoche, lr 1e-3, wd 1e-4,
  dropout 0,1).
- **Variabilità between/within-cancer**: le teste esistenti non avevano mai
  scritto predizioni per-CpG (solo `metrics.json` aggregati). Rifittate con
  la stessa ricetta sui target già congelati
  (`train_only_variance_targets.parquet`): validation/test per refit diretto
  su train, train per 5-fold OOF. **Parity check**: MSE e Spearman di
  validazione ricalcolati coincidono esattamente con i valori già salvati in
  `ntv3_variability_components/{between,within}_block/metrics.json` (vedi
  `inputs/manifest.json`), confermando che la ricetta è stata riprodotta
  fedelmente.

## 3. Controlli di integrità superati

Tutti in `artifacts/rna_branch/2026-07-29_first_benchmark/integrity_checks.json`
e via `rna_branch validate`:

- ID: differenza simmetrica zero fra RNA/beta/embedding/split/metadata.
- Split pazienti e CpG: overlap zero fra tutte le coppie train/validation/test.
- Orientamento: `validate` conferma `rna_shape=[8116,21792]`,
  `methylation_shape=[8116,6742]`, allineamento 8.116/6.742 al 100%.
- Normalizzazione RNA: `RNAStandardizer.fit` usa solo righe train (verificato
  nel codice `data.py`); PCA (`fit-pca`) rifiuta `rna_control != real` e fitta
  solo su `train_sample_split`.
- Permutation control: `shuffle_global`/`shuffle_within_cancer` permutano solo
  entro ciascuno split (mai attraverso train/validation/test).
- Beta mancanti: NaN espliciti (0,9975%), mascherati sia in loss
  (`losses.py`, `torch.isfinite`) sia in metriche (`metrics.py`).
- Sanity model: `mean_rna_negative_control` produce `skill_vs_prior≈0` su
  tutti i pannelli (ancoraggio alla RNA media esatto), `cancer_type_only`
  produce `within_cancer_skill≈0` esatto (un one-hot non porta segnale
  entro-tipo) — entrambi comportamenti attesi e confermati.
- Bug di performance corretto: gli HDF5 iniziali erano scritti gzip-chunked,
  rendendo l'accesso a righe casuali del trainer ~1000× più lento
  (0,048s → 10s per 10 letture di 32 righe). Risolto rigenerando le matrici
  RNA/beta/embedding senza compressione.

## 4. Esperimenti completati

- Test: `PYTHONPATH=src pytest -q tests/test_rna_branch_*.py` → 7 passed.
- `rna_branch validate --config configs/rna_branch/base.yaml` → OK.
- Low-rank totale e within-cancer, componenti 8/16/32/64/128/256.
- Signal grid: 7 righe (prior, mean, cancer_type_only, rna_linear_real,
  rna_linear_shuffle_global, rna_linear_shuffle_within_cancer, rna_mlp_real),
  3 seed (17/29/43) per ogni configurazione allenabile (`prior` è closed-form,
  nessun training). 18 run totali, **0 falliti**. Stessi split, stesso seed
  di partizione, stessi CpG e pazienti osservati per tutte le configurazioni
  (unico `manifest.json` di input condiviso).
- Valutati tutti e quattro i pannelli (in-distribution, sample-OOD, locus-OOD,
  double-OOD) più la componente within-cancer, per ogni run.
- Breakdown per terzile di variabilità del CpG (post-hoc, da predizioni
  salvate) su `rna_linear_real`, `rna_mlp_real`,
  `rna_linear_shuffle_within_cancer` (pannello double_ood).

Nessuna esecuzione SLURM: non è presente uno scheduler in questo ambiente
(`sbatch` assente); si è eseguito direttamente su GPU locale (RTX 3080 Ti,
libera). Lo script `jobs/slurm/run_rna_grid.sh` fornito dall'overlay resta
il comando pronto per un cluster reale (`sbatch --array=1-18 jobs/slurm/run_rna_grid.sh <manifest>`).

## 5. Spettro low-rank (train: 7.304 pazienti × 5.243 CpG train)

Spazio: `logit(beta) − logit(prior NTv3)`.

| componenti | var. spiegata (totale) | var. spiegata (within-cancer) |
|---:|---:|---:|
| 8   | 0,440 | 0,292 |
| 16  | 0,530 | 0,351 |
| 32  | 0,604 | 0,416 |
| 64  | 0,663 | 0,484 |
| 128 | 0,715 | 0,557 |
| 256 | 0,770 | 0,640 |

Il centraggio within-cancer rimuove una parte sostanziale della "facile"
struttura a basso rango (il tipo di cancro spiega da solo un pezzo notevole
del residuo totale): il decadimento residuo within-cancer è più lento. Non è
un rango bassissimo in nessuno dei due casi — motiva compressioni RNA a più
componenti (o non lineari) se si passa allo Stage B, ma non impone di per sé
architetture complesse.

## 6. Tabella comparativa (media±std su 3 seed; pannelli double_ood e sample_ood)

| famiglia | pannello | skill_vs_prior | dynamic_skill | within_cancer_skill | macro_cancer_skill |
|---|---|---:|---:|---:|---:|
| rna_linear_real | double_ood | 0,1033±0,0027 | 0,1891 | **0,1370** | 0,1087 |
| rna_mlp_real | double_ood | 0,0831±0,0074 | 0,1704 | 0,1246 | 0,0906 |
| cancer_type_only | double_ood | 0,0353±0,0104 | 0,0950 | 0,0000 | 0,0311 |
| rna_linear_shuffle_within_cancer | double_ood | 0,0208±0,0027 | 0,0853 | **-0,0137** | 0,0210 |
| mean_rna_negative_control | double_ood | ≈0 | ≈0 | ≈0 | ≈0 |
| prior_reference | double_ood | 0 (rif.) | 0 | 0 | 0 |
| rna_linear_shuffle_global | double_ood | -0,0318 | ≈0 | -0,0022 | -0,0263 |
| rna_linear_real | sample_ood | 0,1964±0,0071 | 0,2532 | **0,1620** | 0,1910 |
| rna_mlp_real | sample_ood | 0,1774±0,0088 | 0,2314 | 0,1505 | 0,1756 |
| cancer_type_only | sample_ood | 0,1069±0,0175 | 0,1332 | 0,0000 | 0,0955 |
| rna_linear_shuffle_within_cancer | sample_ood | 0,1073±0,0225 | 0,1247 | **-0,0164** | 0,1026 |

Tabella completa (tutti i pannelli, tutte le metriche, dev. standard) in
`summary_table.csv` / `summary_table.raw.csv`. Calibrazione:
`dynamic_calibration_alpha≈0,87–1,00`, `dynamic_amplitude_ratio≈0,47–0,51`
per i modelli RNA reali (vedi §7).

## 7. Segnale individuale oltre il cancer type

Tutti i criteri richiesti sono soddisfatti da `rna_linear_real` (e in misura
lievemente minore da `rna_mlp_real`), su sample-OOD **e** double-OOD, con
deviazione standard piccola su 3 seed (alta riproducibilità):

1. `skill_vs_prior > 0`: sì (0,103 / 0,196).
2. `dynamic_skill > 0`: sì (0,189 / 0,253).
3. `within_cancer_skill > 0`: sì (0,137 / 0,162) — **il one-hot cancer_type
   dà esattamente 0** su questa metrica, quindi il segnale within-cancer è
   inequivocabilmente oltre la sola diagnosi.
4. Vantaggio su `shuffle_within_cancer`: netto — within_cancer_skill reale
   0,137/0,162 contro **-0,014/-0,016** del controllo permutato entro tipo
   (la permutazione preserva la marginale per cancer_type ma distrugge
   l'identità del paziente, e il segnale within-cancer collassa a leggermente
   negativo). Questo è il confronto più diretto e più probante.
5. Vantaggio su `cancer_type_only`: netto sia in skill_vs_prior (0,103 vs
   0,035; 0,196 vs 0,107) sia in macro_cancer_skill (0,109 vs 0,031; 0,191 vs
   0,096).
6. Miglioramento replicato su sample-OOD **e** double-OOD, con 3 seed
   indipendenti e std piccola (es. skill_vs_prior double_ood: 0,1033±0,0027).

## 8. Calibrazione e sovra-amplificazione

Nessun problema di sovra-amplificazione: `dynamic_amplitude_ratio≈0,47–0,51`
per i modelli RNA reali indica dinamiche predette **sotto**-amplificate
rispetto a quelle vere (comportamento conservativo, non il pattern opposto
osservato in MethylProphet). `dynamic_calibration_alpha` è vicino a 1
(0,928–1,00 su sample/double-OOD), quindi la scala è ragionevolmente ben
calibrata pur restando leggermente conservativa.

Il breakdown per terzile di variabilità del CpG (pannello double_ood,
`tertile_reports/`) mostra un comportamento del gate coerente e sicuro:

| terzile variabilità | rna_linear_real skill | rna_mlp_real skill | shuffle_within_cancer skill |
|---|---:|---:|---:|
| bassa  | 0,003 | 0,003 | -0,006 |
| media  | 0,068 | 0,047 |  0,010 |
| alta   | 0,124 | 0,101 |  0,029 |

Il gate `variability` sopprime correttamente la correzione sui CpG stabili
(skill ≈ 0, non negativo) e concentra il guadagno sui CpG davvero variabili —
esattamente il comportamento desiderato, nessuna evidenza di
sovra-correzione su loci stabili.

## 9. Decisione

**Criterio soddisfatto**: `rna_linear_real` mostra simultaneamente tutti e
sei i segnali richiesti (skill>0, dynamic_skill>0, within_cancer_skill>0,
vantaggio su shuffle_within_cancer, vantaggio su cancer_type_only,
miglioramento replicato su sample-OOD e double-OOD), su 3 seed indipendenti.

**Decisione**: procedere. Il segnale RNA individuale esiste ed è oltre la
sola diagnosi. Non è però ancora giustificato saltare direttamente a
Perceiver/Fourier/FiLM/cross-attention: lo spettro low-rank (§5) non è
estremamente basso, e MLP non batte ancora il modello lineare — la mossa
corretta secondo l'ordine raccomandato in
`docs/rna_branch_experiments.md` è lo **Stage B** (confronto compressione
RNA: lineare vs MLP vs PCA training-only, a parità di interazione bilineare e
gate `variability`), non ancora lo Stage C. Le architetture complesse restano
giustificate solo dopo che lo Stage B avrà mostrato che una compressione più
ricca migliora — non solo eguaglia — il modello lineare.
