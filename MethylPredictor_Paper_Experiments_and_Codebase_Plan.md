# MethylPredictor — Piano sperimentale e organizzazione della codebase per il paper

> **Scopo del documento**
>
> Questo file definisce:
> 1. quali esperimenti devono essere eseguiti per il paper;
> 2. quali claim scientifici supporta ciascun esperimento;
> 3. quali esperimenti sono **Main**, **Supplementary** o **Development/Architecture**;
> 4. come modificare la codebase per eseguirli in modo riproducibile;
> 5. come organizzare dataset, feature, checkpoint e risultati;
> 6. come rendere automatico il passaggio `run -> metriche -> tabelle/figure del paper`.
>
> La priorità scientifica del paper è:
>
> \[
> \textbf{Performance}
> >
> \textbf{Functional CpG representation}
> >
> \textbf{Mean-proxy / residual decomposition}
> >
> \textbf{Generalization + mechanistic analysis}
> >
> \textbf{Efficiency}
> >
> \textbf{Architecture ablation}
> \]
>
> **Nota importante:** il modello finale usa una rappresentazione della CpG derivata da **annotazioni genomiche/funzionali patient-independent**.  
> I genomic foundation models sono soltanto baseline di confronto della rappresentazione del locus.  
> **NTv3 post-trained non deve essere incluso né citato nel benchmark del paper.**

---

# 1. Claim scientifici da supportare

## C1 — Performance e generalizzazione

**Claim**

MethylPredictor predice la metilazione a partire dall'RNA meglio di MethylProphet sugli split ufficiali, inclusi:

- nuovi pazienti;
- nuove CpG;
- nuovi pazienti + nuove CpG.

**Esperimenti principali associati**

- E01 — Benchmark ufficiale vs MethylProphet.
- E02 — Confronto con modelli di methylation imputation nel setting consentito.
- E11 — External validation.
- E14 — Robustezza multi-seed e confidence intervals.

---

## C2 — Functional representation della CpG

**Claim**

Una rappresentazione patient-independent costruita da annotazioni genomiche e regolatorie esplicite permette al modello di caratterizzare CpG mai osservate durante il training.

La rappresentazione finale può includere, a seconda della feature release congelata:

- CpG island / shore / shelf / open sea;
- promoter / gene body / intergenic;
- TSS-related features;
- cCRE;
- DNase/accessibility;
- H3K27ac;
- H3K4me1;
- H3K4me3;
- H3K27me3;
- H3K36me3;
- H3K9me3;
- CTCF;
- altre annotazioni incluse nel feature manifest finale.

**Esperimenti principali associati**

- E03 — Basic/reduced annotations vs full functional representation.
- E04 — Functional annotations vs generic genomic FM representations.
- E06 — Capacità delle functional annotations di predire la methylation mean di CpG unseen.
- S01 — Ablation dei gruppi di annotazioni.
- S02 — Annotazioni context-specific vs aggregate.

---

## C3 — Mean-proxy e decomposizione locus + patient residual

Il modello sfrutta la struttura:

\[
\text{prediction}_{p,c}
=
\text{locus component}_{c}
+
\text{patient-specific component}_{p,c}
\]

nella parametrizzazione effettivamente usata dalla versione finale.

Il branch CpG è supervisionato a predire una quantità proxy basata sulla metilazione media del locus sui pazienti di training; il branch RNA modella la componente patient-specific.

**Claim**

Separare una componente stabile locus-specific dalla deviazione patient-specific rende il problema RNA→DNAm più semplice e migliora la generalizzazione.

**Esperimenti principali associati**

- E05 — Analisi della stabilità delle CpG.
- E06 — Direct mean-proxy evaluation.
- E07 — Proxy supervision vs no proxy supervision.
- E08 — Structured/residual prediction vs direct prediction.
- E09 — CpG-only / correct RNA / shuffled RNA.
- E10 — RNA gain vs CpG variability.
- S10 — Oracle mean analysis.

---

## C4 — Il trascrittoma contiene realmente informazione patient-specific utile

**Claim**

Le performance non derivano soltanto dalla forte componente locus-specific.  
L'RNA corretto del paziente fornisce informazione necessaria per spiegare la variazione inter-patient.

**Esperimenti**

- E09 — CpG-only vs correct RNA vs shuffled RNA.
- E10 — Gain dell'RNA stratificato per variabilità CpG.
- A04 — Ulteriori causal controls sul meccanismo di interazione.

---

## C5 — Efficienza

**Claim secondario**

Il vantaggio predittivo non richiede una soluzione computazionalmente più pesante di MethylProphet.

**Esperimenti**

- E12 — Parametri, memoria, training time e throughput.
- S09 — Scaling computazionale dettagliato.

---

## C6 — Architettura

**Claim volutamente secondario**

Il meccanismo locus-conditioned e il blocco finale di interazione forniscono un vantaggio rispetto a fusioni più semplici.

**Esperimenti**

- A01 — Global RNA vs locus-conditioned RNA.
- A02 — Simple fusion vs final fusion.
- A03 — Numero di program token.
- A04 — Attention/interaction controls.
- A05 — Varianti architetturali già esplorate.

Questi esperimenti vanno **alla fine dei Results** o nel Supplementary, salvo risultati eccezionalmente forti.

---

# 2. Protocollo sperimentale da congelare

## 2.1 Split

Usare gli **split ufficiali di MethylProphet ottenuti dalla release Hugging Face**.

Le viste canoniche sono:

| View ID | CpG | Patient | Interpretazione |
|---|---|---|---|
| `id` | Train | Train | In-distribution |
| `sample_ood` | Train | Val | Nuovi pazienti |
| `locus_ood` | Val | Train | Nuove CpG |
| `double_ood` | Val | Val | Nuovi pazienti + nuove CpG |

Le viste scientificamente principali sono:

1. `sample_ood`
2. `locus_ood`
3. `double_ood`

`double_ood` è la headline principale.

---

## 2.2 Metriche canoniche

### Primary metric

- `mas_pcc`

### Secondary metrics

- `mac_pcc`
- `mse`
- `mae`
- `skill_vs_prior`, quando applicabile
- eventuale `ccc` solo come analisi complementare

Ogni evaluator deve restituire **lo stesso schema di metriche** indipendentemente dal modello.

---

## 2.3 Checkpoint selection

Il criterio di selezione deve essere definito una volta e registrato nel manifest.

Esempio:

```yaml
checkpoint_selection:
  metric: dev/mas_pcc
  mode: max
  patience: 6
```

### Regola anti-leakage

Le viste utilizzate come risultato finale del paper non devono essere usate in modo iterativo per:

- scegliere architetture;
- scegliere iperparametri;
- scegliere checkpoint;
- scegliere feature.

Se gli split ufficiali MP sono trattati come benchmark finale, creare un **dev split interno esclusivamente dal training set** per model selection.

Il manifest deve rendere esplicito quale split è:

- `train`
- `dev`
- `paper_eval`

---

## 2.4 Seeds

Convenzione consigliata:

```text
17
29
43
59
```

Uso:

- final model: almeno 3 seed;
- core ablations: idealmente 3 seed;
- expensive architecture development: seed 17;
- eval-only controls: tutti i seed disponibili del modello finale quando economico.

---

# 3. Catalogo esperimenti MAIN

---

## E01 — Official MethylProphet benchmark

### Obiettivo

Confrontare MethylPredictor e MethylProphet sul protocollo ufficiale.

### Arms

```text
E01_MP
E01_MethylPredictor
```

Eventuali semplici baseline possono essere aggiunte come arms separati.

### Views

```text
sample_ood
locus_ood
double_ood
```

### Metriche

```text
MAS-PCC
MAC-PCC
MSE
MAE
```

### Scala

**Genome-wide.**

### Seeds

- MethylPredictor: >= 3
- MP: seguire il protocollo più fedele possibile alla release ufficiale.

### Output paper

- Main Table 1.
- Figure/plot con delta rispetto a MP.
- Headline: `double_ood`.

### Codice richiesto

- loader degli split ufficiali MP;
- evaluator condiviso;
- adapter per checkpoint MP;
- runner capace di valutare entrambi sugli stessi sample/CpG IDs.

---

## E02 — Methylation models / FM benchmark

### Obiettivo

Confrontare con modelli che imputano CpG unseen quando il paziente è già disponibile.

### View

```text
locus_ood = Val CpG x Train Patient
```

### Candidati

```text
DeepCpG
CpGPT
MethylGPT
MethylProphet
MethylPredictor
```

Includere solo modelli per cui il protocollo è riproducibile correttamente.

### Importante

La tabella deve dichiarare la modalità di input:

```text
Uses RNA?
Uses observed patient DNAm?
```

Non presentare il confronto come iso-input quando non lo è.

### Scala

Protocollo comparabile a quello usato da MP.

### Output paper

**Main Table 2.**

---

## E03 — Functional annotation contribution

### Domanda

Quanto contribuisce la rappresentazione funzionale esplicita della CpG?

### Arms minimi

```text
minimal
basic_context
full_functional
```

Definizioni da congelare in feature manifest.

#### `minimal`

Solo informazione minima necessaria per identificare/caratterizzare il locus senza il blocco funzionale completo.

#### `basic_context`

Esempi:

- CpG island class;
- genomic region;
- gene/TSS context;
- static locus annotations.

#### `full_functional`

Feature finali utilizzate dal modello paper.

### Views prioritarie

```text
locus_ood
double_ood
```

### Views secondarie

```text
sample_ood
```

### Claim

Il gain dovrebbe emergere soprattutto quando la CpG è unseen.

### Scala

Preferibilmente genome-wide per `minimal` vs `full_functional`.

Le ablation più fini possono essere chr1.

### Output paper

- Main Figure 3 o Main Table 3.

---

## E04 — Functional annotations vs generic genomic FMs

### Domanda

Le annotazioni funzionali esplicite sono una rappresentazione competitiva/migliore rispetto a rappresentazioni sequence-derived generiche?

### Arms

```text
functional_full
genomic_fm_gena
genomic_fm_ntv3_pre
genomic_fm_<optional_third_model>
```

**Esclusione esplicita:**

```text
DO NOT ADD NTV3 POST
```

### Regola sperimentale

Cambiare **solo la rappresentazione del locus** quanto più possibile.

RNA branch, training protocol, loss, split ed evaluator devono restare invariati.

### Views

```text
locus_ood
double_ood
```

### Output paper

Main experiment.

### Claim consentito

> Functional annotations outperform the evaluated generic genomic sequence representations in this experimental setting.

### Claim da evitare

> Functional annotations are universally superior to genomic foundation models.

---

## E05 — CpG stability analysis

### Tipo

Analysis-only, nessun training.

### Domanda

Quanto della metilazione è stabile e locus-specific?

### Per ogni CpG calcolare

```text
mean_beta
std_beta
variance_beta
iqr_beta
```

sui **training patients**.

### Analisi

1. distribuzione `std_beta`;
2. distribuzione `IQR`;
3. errore di un predictor che usa soltanto `mean_beta`;
4. quantili di CpG variability;
5. eventuale breakdown per functional class.

### Output

- Main Figure 4A.
- Tabella supplementary completa.

### Scopo

Motivare empiricamente la decomposizione:

```text
stable locus component + patient-specific deviation
```

---

## E06 — Direct evaluation of the mean proxy

### Domanda

Le feature funzionali permettono di predire il comportamento medio di una CpG unseen?

### Input

Val CpGs.

### Prediction

```text
mu_hat[c]
```

### Ground truth

Media della CpG calcolata sui target held-out **solo per evaluation**.

Non deve mai entrare nel training.

### Metriche

```text
Pearson(mu_hat, mu_true)
Spearman(mu_hat, mu_true)
MSE
MAE
```

### Output

- scatter `mu_true vs mu_hat`;
- density/error plot;
- metrics JSON;
- per-CpG parquet.

### Paper

Main.

---

## E07 — Mean-proxy supervision ablation

### Domanda

Supervisionare esplicitamente il branch locus con il proxy migliora la predizione finale?

### Arms

```text
full
no_mean_proxy_loss
```

### Vincolo

Le due configurazioni devono essere identiche tranne per la supervisione del proxy.

### Views

```text
locus_ood
double_ood
```

### Metriche

Primary:

```text
MAS-PCC
```

Secondary:

```text
MSE
MAE
MAC-PCC
```

### Paper

Main ablation.

---

## E08 — Structured residual vs direct prediction

### Domanda

La decomposizione locus component + patient-specific correction è utile rispetto a una predizione diretta?

### Arms

```text
direct_prediction
structured_residual
```

### Vincolo

Mantenere quanto più possibile costanti:

- CpG representation;
- RNA encoder;
- parameter budget;
- training schedule.

### Paper

Main ablation.

---

## E09 — Does the model actually use RNA?

### Tipo

In gran parte evaluation-only.

### Conditions

```text
cpg_only
correct_rna
shuffled_rna
```

Opzionale:

```text
mean_rna
```

### Domanda

Il trascrittoma corretto del paziente contiene informazione necessaria oltre al locus prior?

### Aspettativa

```text
correct_rna > shuffled_rna
correct_rna > cpg_only
```

soprattutto per CpG con elevata variabilità tra pazienti.

### Views

```text
sample_ood
double_ood
```

### Paper

Main mechanistic experiment.

---

## E10 — RNA gain vs CpG variability

### Tipo

Evaluation-only.

### Procedura

1. calcolare la variabilità delle CpG sui training patients;
2. suddividere in quantili, ad esempio Q1–Q5;
3. per ogni bin valutare:

```text
CpG-only
Full model
MethylProphet
```

4. calcolare:

```text
delta_rna = Full - CpG-only
delta_vs_mp = Full - MP
```

### Ipotesi

- CpG stabili -> forte contributo del locus component;
- CpG variabili -> maggiore contributo dell'RNA.

### Paper

Main se il trend è forte e coerente; altrimenti Supplementary.

---

## E11 — External validation

### Obiettivo

Testare generalizzazione fuori da TCGA.

### Protocollo

```text
train: TCGA
test: independent matched RNA/DNAm dataset
```

Nessun fine-tuning sul test.

### Requisiti

- mapping CpG esplicito;
- coverage report;
- stessa feature pipeline del locus;
- preprocessing RNA documentato;
- nessun fitting sul dataset esterno.

### Metriche

```text
MAS-PCC
MAC-PCC
MSE
MAE
n_samples
n_cpg
coverage
```

### Paper

Main se disponibile in forma rigorosa.

---

## E12 — Efficiency benchmark

### Confronto minimo

```text
MethylProphet
MethylPredictor
```

### Misure

```text
trainable_params
total_params
peak_gpu_memory_gb
seconds_per_epoch
total_training_gpu_hours
inference_pairs_per_second
cpgs_per_second
```

### Protocollo

Quando possibile:

- stesso hardware;
- stessa precisione;
- stesso tipo di workload;
- warm-up prima del timing;
- più ripetizioni per inference timing.

### Paper

Tabella compatta nel Main.

---

## E13 — Biological / functional stratification

### Tipo

Evaluation-only.

### Strati

Almeno:

```text
CpG island
shore
shelf
open sea
promoter
gene body
intergenic
cCRE classes
accessibility classes
```

in base alle feature effettivamente disponibili.

### Per ogni strato

```text
n_cpg
ours_mas
mp_mas
delta_mas
ours_mse
mp_mse
```

### Paper

Una figura riassuntiva Main; tabella completa Supplementary.

---

## E14 — Statistical robustness

### Multi-seed

Per:

- final model;
- E03 core comparison;
- E07;
- E08, se computazionalmente sostenibile.

### Riportare

```text
mean
std
```

### Confidence intervals

Per differenze vs MP:

- bootstrap a livello CpG per MAS;
- bootstrap a livello patient per MAC.

Evitare di trattare ogni patient×CpG pair come osservazione indipendente per un test di significatività.

### Paper

Main confidence interval + Supplementary seed table.

---

# 4. Supplementary experiments

---

## S01 — Functional annotation group ablation

Due possibili strategie; sceglierne **una**.

### Opzione consigliata: leave-one-group-out

```text
full
full_minus_accessibility
full_minus_active_histones
full_minus_repressive_histones
full_minus_ctcf
full_minus_static_context
```

Vantaggio: misura direttamente quanto perde il full model rimuovendo un gruppo.

### Scala

Prima chr1.

Portare genome-wide solo le ablation informative.

---

## S02 — Context-specific vs aggregate annotations

### Arms

```text
aggregate_only
context_specific
context_specific_plus_aggregate
```

### Domanda

Conta soltanto sapere che un locus è funzionale o conta il pattern di attività attraverso diversi contesti/biosample?

### Esempio aggregate

```text
accessibility_breadth
h3k27ac_breadth
ctcf_breadth
```

---

## S03 — Performance by methylation level

Bin basati sulla methylation mean:

```text
hypomethylated
intermediate
hypermethylated
```

Riportare MAS/MSE/MAE e gain vs MP.

---

## S04 — Performance by cancer type / tissue

Per ogni coorte TCGA con numerosità sufficiente:

```text
n_patients
ours_mas
mp_mas
delta_mas
```

---

## S05 — Performance by chromosome

Per:

```text
chr1 ... chr22
```

Sanity check di generalizzazione genomica.

---

## S06 — Per-CpG wins

Per ogni CpG:

```text
pcc_ours
pcc_mp
delta_pcc
```

Riportare:

- percentuale di loci vinti;
- mediana delta;
- distribuzione;
- enrichment per functional category.

---

## S07 — Per-patient wins

Per ogni paziente:

```text
pcc_ours
pcc_mp
delta_pcc
```

Riportare:

- percentuale di pazienti vinti;
- distribuzione;
- cancer-type breakdown.

---

## S08 — Calibration and error characterization

Produrre:

```text
calibration bins
predicted vs observed
slope
intercept
CCC (optional)
absolute error distribution
```

Stratificare almeno per methylation range.

---

## S09 — Compute scaling

Misurare:

```text
VRAM vs CpG batch size
runtime vs number of CpGs
runtime vs number of patients
RNA encoding cost
locus encoding cost
```

---

## S10 — Oracle mean analysis

### Scopo

Separare:

- errore nella stima del locus component;
- errore nella stima della patient-specific correction.

### Importante

`mu_oracle` usa target held-out ed è esclusivamente diagnostico.

Non presentarlo come modello utilizzabile.

---

# 5. Architecture / development experiments

Questi esperimenti hanno priorità inferiore.

---

## A01 — Global RNA vs locus-conditioned RNA

### Arms

```text
global_rna
locus_conditioned_rna
```

### Vincolo

CpG representation, proxy branch e capacity devono restare comparabili.

### Domanda

È utile che la rappresentazione RNA utilizzata per la predizione dipenda dalla CpG interrogata?

---

## A02 — Fusion block ablation

Confrontare al massimo 2–3 varianti:

```text
simple_concat_or_additive
simple_multiplicative
final_fusion
```

Non creare un architecture zoo nel Main paper.

---

## A03 — Number of RNA program tokens

Esempio:

```text
K=16
K=32
K=64
K=128
```

Misurare:

```text
MAS-PCC
MSE
VRAM
throughput
```

---

## A04 — Interaction / attention causal controls

Evaluation-only dove possibile:

```text
normal
cpg_query_permutation
rna_patient_permutation
mean_rna
uniform_attention
locus_mean_attention
```

I controlli più informativi possono essere richiamati nel Main; il resto va nel Supplementary.

---

## A05 — Existing architecture variants

Raccogliere i run già eseguiti senza necessariamente rilanciarli:

```text
residual_mlp
headwise
soft_router
program_projection
other completed development variants
```

Questa tabella serve a documentare il development, non a costruire il claim principale.

---

## A06 — Temperature / top-k

Raccogliere le sweep già disponibili:

```text
attention_temperature
top_k
```

Solo Supplementary.

---

# 6. Struttura consigliata della codebase

Obiettivo: evitare script monolitici specifici per singolo esperimento.

```text
MehylPredictor/
├── configs/
│   ├── data/
│   │   ├── tcga_mp_official.yaml
│   │   └── external_<dataset>.yaml
│   │
│   ├── features/
│   │   ├── minimal.yaml
│   │   ├── basic_context.yaml
│   │   ├── functional_full_v1.yaml
│   │   ├── functional_minus_accessibility.yaml
│   │   ├── functional_minus_histones.yaml
│   │   ├── genomic_fm_gena.yaml
│   │   └── genomic_fm_ntv3_pre.yaml
│   │
│   ├── models/
│   │   ├── final.yaml
│   │   ├── no_mean_proxy.yaml
│   │   ├── direct_prediction.yaml
│   │   ├── global_rna.yaml
│   │   └── fusion_ablations/
│   │
│   ├── training/
│   │   └── paper_default.yaml
│   │
│   └── experiments/
│       └── paper_v1/
│           ├── E01_official_benchmark.yaml
│           ├── E03_functional_features.yaml
│           ├── E04_genomic_fm.yaml
│           ├── E07_proxy_ablation.yaml
│           ├── E08_residual_ablation.yaml
│           └── ...
│
├── src/methylation_predictor/
│   ├── data/
│   │   ├── split_registry.py
│   │   ├── target_store.py
│   │   └── rna_store.py
│   │
│   ├── features/
│   │   ├── registry.py
│   │   ├── functional_annotations.py
│   │   └── genomic_fm.py
│   │
│   ├── models/
│   │   ├── cpg_encoder.py
│   │   ├── rna_encoder.py
│   │   ├── mean_proxy.py
│   │   ├── interaction.py
│   │   └── predictor.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── strata.py
│   │   └── controls.py
│   │
│   └── experiments/
│       ├── registry.py
│       └── manifest.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   │
│   └── paper/
│       ├── run_experiment.py
│       ├── evaluate_run.py
│       ├── collect_results.py
│       ├── validate_manifests.py
│       ├── make_tables.py
│       └── make_figures.py
│
└── tests/
    ├── test_official_splits.py
    ├── test_feature_registry.py
    ├── test_no_target_leakage.py
    ├── test_evaluator_views.py
    ├── test_metrics.py
    └── test_result_schema.py
```

---

# 7. Principio fondamentale: componenti configurabili

Ogni esperimento deve essere ottenibile modificando **config**, non copiando trainer.

Separare almeno questi assi:

```yaml
data:
  protocol: tcga_mp_official_v1

features:
  kind: functional
  feature_set: functional_full_v1

model:
  cpg_encoder: ...
  rna_encoder: ...
  mean_proxy:
    enabled: true
    supervised: true
  interaction:
    kind: final
  prediction:
    kind: structured_residual

training:
  epochs: 80
  seed: 17

evaluation:
  views:
    - sample_ood
    - locus_ood
    - double_ood
```

E07 dovrebbe richiedere soltanto:

```yaml
model:
  mean_proxy:
    supervised: false
```

E08:

```yaml
model:
  prediction:
    kind: direct
```

E04:

```yaml
features:
  kind: genomic_fm
  feature_set: gena_v1
```

Questo rende le ablation realmente controllate.

---

# 8. Freeze della final architecture

Poiché l'architettura è ancora stata oggetto di modifiche durante il development, **non affidarsi al nome colloquiale del modello**.

La versione paper deve essere identificata da:

```yaml
model_id: methylpredictor_final_v1
recipe_path: configs/models/final.yaml
git_commit: <commit>
config_sha256: <hash>
```

Il manifest deve inoltre registrare:

```text
CpG feature set
RNA feature set
mean-proxy formulation
interaction module
prediction head
losses
dimensions
number of RNA program tokens
```

In questo documento vengono fissati i requisiti funzionali:

1. input CpG = functional annotations;
2. locus branch / mean proxy;
3. RNA branch;
4. patient-specific interaction conditioned on the locus;
5. final methylation prediction.

La formula esatta del blocco finale deve essere presa dalla **recipe finale congelata**, non ricostruita a memoria nella pipeline degli esperimenti.

---

# 9. Organizzazione dei dati

Non duplicare i dataset per ogni esperimento.

Struttura consigliata:

```text
MethylPredictionData/
└── paper_v1/
    ├── manifests/
    │   ├── tcga_mp_official_v1.yaml
    │   ├── rna_v1.yaml
    │   ├── methylation_targets_v1.yaml
    │   ├── functional_full_v1.yaml
    │   └── genomic_fm_features_v1.yaml
    │
    ├── splits/
    │   └── methylprophet_official_v1/
    │       ├── train_samples.parquet
    │       ├── val_samples.parquet
    │       ├── train_cpg.parquet
    │       ├── val_cpg.parquet
    │       └── source.json
    │
    ├── targets/
    │   └── tcga/
    │       └── <canonical target store>
    │
    ├── rna/
    │   └── tcga_rna_v1/
    │
    ├── cpg_features/
    │   ├── functional_full_v1/
    │   │   ├── features.*
    │   │   ├── cpg_ids.parquet
    │   │   ├── feature_names.json
    │   │   └── manifest.json
    │   │
    │   ├── basic_context_v1/
    │   ├── minimal_v1/
    │   │
    │   └── genomic_fm/
    │       ├── gena_v1/
    │       └── ntv3_pre_v1/
    │
    └── external/
        └── <dataset>/
```

## Regola

Le feature set sono **dataset artifacts versionati**.

Un run deve referenziare:

```text
functional_full_v1
```

non un path ad hoc come:

```text
/tmp/features_final_really_final2.npy
```

---

# 10. Feature manifest

Ogni feature set deve avere un manifest.

Esempio:

```json
{
  "feature_set_id": "functional_full_v1",
  "genome_build": "hg38",
  "n_cpg": 0,
  "n_features": 0,
  "groups": {
    "static_context": [],
    "dnase": [],
    "active_histones": [],
    "repressive_histones": [],
    "ctcf": []
  },
  "source_versions": {},
  "created_by": "scripts/features/build_functional_features.py",
  "git_commit": "<commit>",
  "cpg_id_hash": "<sha256>",
  "feature_matrix_hash": "<sha256>"
}
```

Questo è indispensabile per S01 e S02.

---

# 11. Split manifest

Lo split ufficiale deve diventare un artefatto immutabile.

Esempio:

```json
{
  "split_id": "methylprophet_official_v1",
  "source": "official MethylProphet Hugging Face release",
  "genome_build": "hg38",
  "train_samples": {
    "n": 0,
    "sha256": "..."
  },
  "val_samples": {
    "n": 0,
    "sha256": "..."
  },
  "train_cpg": {
    "n": 0,
    "sha256": "..."
  },
  "val_cpg": {
    "n": 0,
    "sha256": "..."
  }
}
```

## Test automatici

Il CI/test suite deve verificare:

```text
train_samples ∩ val_samples = ∅
train_cpg ∩ val_cpg = ∅
all IDs exist in canonical stores
order-independent hashes match expected values
```

---

# 12. Organizzazione dei training run

Non organizzare i checkpoint solo per nome del modello.

Usare:

```text
experiments/
└── paper_v1/
    ├── E01_official_benchmark/
    │   └── methylpredictor/
    │       ├── seed17/
    │       ├── seed29/
    │       └── seed43/
    │
    ├── E03_functional_representation/
    │   ├── minimal/
    │   ├── basic_context/
    │   └── functional_full/
    │
    ├── E04_genomic_fm/
    │   ├── gena/
    │   └── ntv3_pre/
    │
    ├── E07_proxy_ablation/
    │   ├── full/
    │   └── no_proxy/
    │
    └── ...
```

Ogni leaf directory corrisponde a:

```text
experiment + arm + seed
```

---

# 13. Struttura di una singola run

```text
seed17/
├── run_manifest.json
├── config.resolved.yaml
├── metadata.json
├── history.json
│
├── checkpoints/
│   ├── best.pt
│   └── last.pt
│
├── evaluation/
│   ├── summary.json
│   │
│   ├── sample_ood/
│   │   ├── metrics.json
│   │   ├── per_cpg.parquet
│   │   ├── per_patient.parquet
│   │   └── strata.parquet
│   │
│   ├── locus_ood/
│   │   └── ...
│   │
│   └── double_ood/
│       └── ...
│
├── predictions/
│   └── OPTIONAL
│
└── diagnostics/
    ├── mean_proxy.parquet
    ├── attention/
    └── compute.json
```

---

# 14. `run_manifest.json`

È il file più importante della run.

Esempio:

```json
{
  "paper_version": "paper_v1",
  "experiment_id": "E07",
  "arm_id": "no_proxy",
  "run_id": "E07_no_proxy_seed17",
  "seed": 17,

  "git_commit": "<commit>",
  "dirty_repo": false,

  "data_protocol": "tcga_mp_official_v1",
  "data_manifest_hash": "...",
  "feature_set": "functional_full_v1",
  "feature_manifest_hash": "...",

  "model_id": "methylpredictor_final_v1",
  "config_hash": "...",

  "checkpoint_selection": {
    "metric": "dev/mas_pcc",
    "mode": "max"
  },

  "hardware": {
    "gpu": "...",
    "precision": "..."
  }
}
```

---

# 15. Schema unificato delle metriche

Ogni evaluator deve produrre un record compatibile con uno schema unico.

Esempio `metrics.json`:

```json
{
  "experiment_id": "E01",
  "arm_id": "methylpredictor",
  "seed": 17,
  "view": "double_ood",

  "n_samples": 0,
  "n_cpg": 0,
  "n_pairs": 0,

  "metrics": {
    "mas_pcc": 0.0,
    "mac_pcc": 0.0,
    "mse": 0.0,
    "mae": 0.0,
    "skill_vs_prior": 0.0
  }
}
```

Non salvare metriche con naming diverso tra script, per esempio:

```text
mas
MAS
median_sample_corr
mas_pcc
```

Scegliere una convenzione unica:

```text
mas_pcc
mac_pcc
mse
mae
skill_vs_prior
```

---

# 16. Per-CpG results

Schema consigliato:

```text
cpg_id
chrom
position
n_samples
pcc
mse
mae
mean_true
mean_pred
std_true
feature_class_*
```

Serve per:

- S03;
- S05;
- S06;
- E13;
- bootstrap MAS.

---

# 17. Per-patient results

Schema:

```text
sample_id
cohort
n_cpg
pcc
mse
mae
```

Serve per:

- S04;
- S07;
- bootstrap MAC.

---

# 18. Prediction storage

Le matrici complete patient×CpG possono essere enormi.

Non salvarle automaticamente per ogni run.

Config:

```yaml
evaluation:
  save_predictions: false
```

Abilitarlo solo quando serve.

Formato consigliato quando necessario:

```text
Parquet chunked
oppure
Zarr/HDF5
```

con manifest associato.

Per le analisi standard bastano:

- aggregate metrics;
- per-CpG;
- per-patient.

---

# 19. Mean-proxy diagnostic schema

Per E06 salvare:

```text
cpg_id
split
mu_hat
mu_true
error
abs_error
functional_class
variability
```

File:

```text
diagnostics/mean_proxy.parquet
```

Questo deve essere generabile per qualunque checkpoint compatibile.

---

# 20. Result warehouse del paper

I risultati finali non devono essere letti manualmente dalle singole cartelle.

Creare:

```text
paper_results/
└── paper_v1/
    ├── registry.parquet
    ├── metrics_long.parquet
    ├── per_cpg/
    ├── per_patient/
    │
    ├── tables/
    │   ├── table_main_benchmark.csv
    │   ├── table_methylation_models.csv
    │   ├── table_core_ablations.csv
    │   └── table_efficiency.csv
    │
    └── figures/
        ├── fig_main_benchmark.*
        ├── fig_functional_representation.*
        ├── fig_mean_proxy.*
        └── fig_rna_variability.*
```

---

# 21. `metrics_long.parquet`

Una riga per:

```text
experiment
arm
seed
view
metric
```

Schema:

```text
paper_version
experiment_id
arm_id
run_id
seed
view
metric_name
metric_value
n_samples
n_cpg
checkpoint
git_commit
data_protocol
feature_set
model_id
```

Esempio:

```text
paper_v1 | E07 | no_proxy | ... | 17 | double_ood | mas_pcc | 0.55 | ...
paper_v1 | E07 | full     | ... | 17 | double_ood | mas_pcc | 0.58 | ...
```

Questo formato rende banale creare tutte le tabelle.

---

# 22. Experiment registry

Creare un file:

```text
experiments/paper_v1/registry.yaml
```

Esempio:

```yaml
E01:
  title: Official MethylProphet benchmark
  tier: main
  status: planned
  arms:
    - methylprophet
    - methylpredictor
  required_views:
    - sample_ood
    - locus_ood
    - double_ood

E03:
  title: Functional CpG representation
  tier: main
  status: planned
  arms:
    - minimal
    - basic_context
    - functional_full
  required_views:
    - locus_ood
    - double_ood

E07:
  title: Mean proxy supervision
  tier: main
  status: planned
  arms:
    - full
    - no_proxy
```

Stati ammessi:

```text
planned
implemented
running
complete
validated
paper_ready
```

---

# 23. Runner unico degli esperimenti

Interfaccia desiderata:

```bash
python scripts/paper/run_experiment.py \
  --experiment E07 \
  --arm no_proxy \
  --seed 17
```

Il runner:

1. legge `registry.yaml`;
2. carica config base;
3. applica override dell'arm;
4. valida dataset/feature manifest;
5. crea la directory standard;
6. salva `run_manifest.json`;
7. avvia training;
8. salva best checkpoint;
9. opzionalmente lancia evaluation.

---

# 24. Evaluator unico

Interfaccia:

```bash
python scripts/paper/evaluate_run.py \
  --run experiments/paper_v1/E07_proxy_ablation/no_proxy/seed17 \
  --views locus_ood double_ood
```

L'evaluator deve supportare:

```text
standard prediction
CpG-only
RNA shuffle
mean RNA
query permutation
uniform interaction
other eval-only controls
```

senza modificare il checkpoint.

Questo è particolarmente importante per E09 e A04.

---

# 25. Collector

Comando:

```bash
python scripts/paper/collect_results.py \
  --paper-version paper_v1
```

Funzioni:

- trova tutte le run;
- valida i manifest;
- verifica che gli experiment arms richiesti siano presenti;
- raccoglie metriche;
- genera `metrics_long.parquet`;
- segnala run mancanti o incompatibili.

---

# 26. Table generator

```bash
python scripts/paper/make_tables.py \
  --paper-version paper_v1
```

Non copiare numeri manualmente dentro LaTeX.

Generare automaticamente:

```text
CSV
LaTeX
Markdown
```

per ogni tabella.

Idealmente:

```text
table_main_benchmark.tex
table_core_ablations.tex
table_efficiency.tex
```

---

# 27. Figure generator

Stesso principio:

```bash
python scripts/paper/make_figures.py \
  --paper-version paper_v1
```

Le figure devono leggere soltanto dal result warehouse, non dai checkpoint.

Così:

```text
checkpoint -> evaluator -> canonical results -> figure
```

e non:

```text
random notebook -> checkpoint -> numero copiato a mano
```

---

# 28. Test da aggiungere prima dei grandi training

## Data tests

```text
official split hashes
no train/val sample overlap
no train/val CpG overlap
all CpG coordinates hg38
all CpG IDs mapped exactly once
RNA sample IDs aligned
target sample IDs aligned
```

## Feature tests

```text
same CpG ordering after load
feature dimension equals manifest
no NaN/Inf
functional features patient-independent
no target-derived values in feature matrix
```

## Model ablation tests

Per E07:

```text
no_proxy config differs from full only in proxy supervision
```

Per E08:

```text
direct vs structured uses same feature set/RNA split/training schedule
```

Per E04:

```text
only locus representation changes
```

## Evaluator tests

```text
view IDs select expected cartesian products
MAS implementation fixed
MAC implementation fixed
shuffled RNA actually changes patient mapping
shuffling is deterministic per seed
```

---

# 29. Naming convention

Evitare run ID del tipo:

```text
new-final-best-final2-k64
```

Usare:

```text
{experiment_id}_{arm_id}_seed{seed}
```

Esempi:

```text
E01_methylpredictor_seed17
E03_functional_full_seed17
E03_basic_context_seed17
E04_gena_seed17
E04_ntv3_pre_seed17
E07_no_proxy_seed17
E08_direct_seed17
A01_global_rna_seed17
```

L'informazione dettagliata è nel manifest, non nel nome.

---

# 30. Policy sui checkpoint

Per ogni run conservare almeno:

```text
best.pt
last.pt
```

Per run paper-ready:

- non sovrascrivere;
- hashare il best checkpoint;
- registrare l'hash nel manifest/result registry.

Esempio:

```json
{
  "best_checkpoint": "checkpoints/best.pt",
  "best_checkpoint_sha256": "..."
}
```

---

# 31. Development vs paper runs

Separare fisicamente:

```text
experiments/dev/
experiments/paper_v1/
```

Una run entra in `paper_v1` solo quando:

1. config congelata;
2. split ufficiale;
3. feature manifest versionato;
4. commit registrato;
5. naming conforme.

Non spostare manualmente una vecchia run dentro `paper_v1` senza creare un manifest di provenance.

---

# 32. Compatibilità con i run esistenti

I vecchi run possono essere importati tramite:

```text
scripts/paper/import_legacy_run.py
```

Lo script dovrebbe richiedere:

```text
experiment_id
arm_id
seed
original_run_path
data_protocol
feature_set
git_commit, se recuperabile
```

e generare un manifest.

Se qualche elemento non è recuperabile:

```text
provenance_complete: false
```

Quella run può essere utile per development/supplementary ma non dovrebbe diventare automaticamente un numero headline del paper.

---

# 33. Matrice esperimento → necessità di training

| ID | Esperimento | Nuovo training? | Eval-only possibile? |
|---|---|---:|---:|
| E01 | MP benchmark | Final model sì | Evaluation sì |
| E02 | methylation models | dipende dal metodo | sì per checkpoint pubblici |
| E03 | functional representation | **sì** | no |
| E04 | genomic FM representation | **sì** | no |
| E05 | CpG stability | no | **sì** |
| E06 | mean proxy evaluation | no, se head disponibile | **sì** |
| E07 | no proxy loss | **sì** | no |
| E08 | direct vs residual | **sì** | no |
| E09 | RNA controls | no | **sì** |
| E10 | variability analysis | no | **sì** |
| E11 | external validation | no fine-tuning | **sì** |
| E12 | efficiency | no / controlled rerun | benchmark |
| E13 | functional strata | no | **sì** |
| E14 | seeds | **sì** | evaluation |
| S01 | feature groups | sì | no |
| S02 | aggregate/context-specific | sì | no |
| S03–S08 | analyses | no | **sì** |
| S10 | oracle | no | **sì** |
| A01 | global RNA | sì | no |
| A02 | fusion | sì | no |
| A03 | K sweep | sì | no |
| A04 | causal controls | spesso no | **sì** |
| A05 | existing variants | preferire run esistenti | sì |

---

# 34. Priorità di implementazione della codebase

## Phase 1 — Reproducibility infrastructure

Prima di lanciare nuovi run:

- [ ] official split manifest;
- [ ] feature manifest;
- [ ] experiment registry;
- [ ] standardized run manifest;
- [ ] unified evaluator;
- [ ] result collector;
- [ ] tests anti-leakage.

Questa fase evita di dover riorganizzare decine di run in seguito.

---

## Phase 2 — Paper-critical training support

Implementare configurabilità per:

- [ ] functional feature set;
- [ ] generic genomic FM feature set;
- [ ] proxy supervision on/off;
- [ ] structured/direct prediction;
- [ ] seeds.

Questo abilita:

```text
E01
E03
E04
E07
E08
E14
```

che sono i training più importanti.

---

## Phase 3 — Eval-only analyses

Implementare evaluator per:

- [ ] CpG-only;
- [ ] RNA shuffle;
- [ ] mean-proxy extraction;
- [ ] variability bins;
- [ ] functional strata;
- [ ] per-CpG;
- [ ] per-patient;
- [ ] calibration;
- [ ] bootstrap.

Questo abilita molti esperimenti senza nuovo training:

```text
E05
E06
E09
E10
E13
S03-S08
S10
```

---

## Phase 4 — External and efficiency

- [ ] external dataset adapter;
- [ ] coverage report;
- [ ] compute profiler;
- [ ] standardized inference benchmark.

---

## Phase 5 — Architecture

Solo dopo che i claim principali sono coperti:

- [ ] global RNA control;
- [ ] fusion ablation;
- [ ] K sweep;
- [ ] extra interaction controls.

---

# 35. Ordine consigliato dei nuovi training

## P0 — indispensabili

1. `E01_methylpredictor` — final model sugli split ufficiali.
2. `E03_minimal/basic/full`.
3. `E04_gena`.
4. `E04_ntv3_pre`.
5. `E07_no_proxy`.
6. `E08_direct`.
7. seed aggiuntivi del final model.

## P1 — dopo i risultati P0

8. seed aggiuntivi delle core ablations.
9. feature-group ablation più informativa.
10. external validation support.
11. eventuale terzo generic genomic FM.

## P2 — architecture

12. global RNA.
13. fusion baseline.
14. K sweep soltanto se necessario.

---

# 36. Criteri per decidere cosa portare genome-wide

## Sempre genome-wide

```text
E01 final benchmark
final model multi-seed
```

## Fortemente consigliato genome-wide

```text
E03 full vs reduced
E07 proxy ablation
E08 direct vs structured
```

## Può iniziare su chr1

```text
E04 multiple genomic FMs
S01 feature-group ablations
S02 context-specific ablation
A01-A06 architecture experiments
```

Se il risultato è importante e diventa un claim Main, promuovere la comparison chiave genome-wide.

---

# 37. Mapping verso il paper

## Figure 1 — Architecture / Method

Nessun risultato sperimentale.

Mostrare concettualmente:

```text
Functional CpG annotations
        |
        v
 locus representation ----> mean/locus component
        |
        +------ conditions interaction with RNA
                                   |
RNA patient -> RNA representation -+
                                   |
                                   v
                     patient-specific component
                                   |
                                   v
                         final methylation
```

La figura deve essere aggiornata sulla **recipe finale congelata**.

---

## Table 1 — Main benchmark

Source:

```text
E01
```

---

## Table 2 — Methylation models

Source:

```text
E02
```

---

## Figure 2/3 — Functional representation

Source:

```text
E03
E04
```

---

## Figure 4 — Why mean proxy works

Source:

```text
E05
E06
E07
E08
```

Possibili panel:

```text
A CpG variability distribution
B mu_true vs mu_hat
C proxy ablation
D direct vs structured
```

---

## Figure 5 — Patient-specific RNA contribution

Source:

```text
E09
E10
```

---

## Figure 6 — Generalization / interpretation

Source:

```text
E11
E13
```

---

## Efficiency table

Source:

```text
E12
```

---

## Last Results paragraph / Supplement

Source:

```text
A01
A02
A03
A04
A05
A06
```

---

# 38. Definition of done per experiment

Un esperimento non è `complete` perché il training è finito.

È `paper_ready` soltanto se:

- [ ] tutti gli arms richiesti esistono;
- [ ] manifest validi;
- [ ] split corretto;
- [ ] feature versionate;
- [ ] checkpoint hashato;
- [ ] evaluation canonica completa;
- [ ] per-CpG/per-patient output presenti quando richiesti;
- [ ] seeds richiesti completi;
- [ ] collector lo trova;
- [ ] tabella/figura generabile automaticamente;
- [ ] nessun numero deve essere copiato manualmente dal terminale.

---

# 39. Checklist minima prima di scrivere i Results

## Performance

- [ ] E01 complete.
- [ ] E02 complete o giustificatamente limitato ai metodi riproducibili.
- [ ] E14 confidence intervals.

## Functional representation

- [ ] E03 complete.
- [ ] E04 almeno 2 generic genomic FMs.
- [ ] E13 functional stratification.

## Proxy/residual

- [ ] E05 stability.
- [ ] E06 direct proxy evaluation.
- [ ] E07 no-proxy.
- [ ] E08 direct prediction.
- [ ] E09 RNA shuffle.
- [ ] E10 variability-dependent RNA gain.

## Generalization

- [ ] E11 external validation, se tecnicamente possibile con protocollo rigoroso.

## Practical

- [ ] E12 efficiency.

## Architecture

- [ ] A01.
- [ ] A02.
- [ ] altri soltanto se utili.

---

# 40. Priorità scientifica finale

Se le risorse computazionali diventano il collo di bottiglia, non sacrificare gli esperimenti core per ulteriori architecture sweep.

La sequenza di evidenza che il paper deve dimostrare è:

\[
\boxed{\text{1. Ours > MethylProphet on the official protocol}}
\]

\[
\boxed{\text{2. Functional locus information helps unseen CpGs}}
\]

\[
\boxed{\text{3. It is competitive with / better than generic genomic FM representations}}
\]

\[
\boxed{\text{4. The mean proxy captures the stable locus component}}
\]

\[
\boxed{\text{5. RNA explains additional patient-specific variation}}
\]

\[
\boxed{\text{6. Structured residual modeling improves the final task}}
\]

e soltanto dopo:

\[
\boxed{\text{7. The chosen architecture is a good implementation of these ideas}}
\]

Questa gerarchia deve guidare sia il budget computazionale sia la struttura della codebase.
