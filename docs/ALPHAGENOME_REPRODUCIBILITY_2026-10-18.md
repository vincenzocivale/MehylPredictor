# Riproducibilità e uso del modello AlphaGenome mixed — 18 ottobre 2026

Per interpretazione scientifica e limiti vedere [stato della ricerca](ALPHAGENOME_RESEARCH_STATE_2026-10-18.md) e [report del confronto](ALPHAGENOME_MIXED_METHYLPROPHET_2026-10-18.md). I comandi seguenti si eseguono dalla root del repository. Descrivono il layout locale della campagna, non un pacchetto autonomo che scarica e ricostruisce ogni dipendenza.

## Checkpoint da usare

Directory: `artifacts/alphagenome_mixed_2026-10-18/official/finite_pairs/`.

| Seed | File relativo alla directory | SHA256 |
| --- | --- | --- |
| 17 | `seed17/checkpoint.pt` | `c41133ede667b6fa3563b25418be70a12591ef8ebc1e7c927fd5bbf51be7ec77` |
| 18 | `seed18/checkpoint.pt` | `41b1a5df598948508b516e1b2d992a13cbb116358a6ccd3301f5b43bc54a28ea` |
| 19 | `seed19/checkpoint.pt` | `d63badb4c23defd228c09d7f940cd2bd1ae353f386620d29f5fe18fb56ea1a2c` |

Seed17 è il checkpoint principale; seed18/19 sono repliche indipendenti. I file `step4000.pt` servono al controllo di esposizione, mentre `checkpoint.pt` è l’esportazione finale con calibrazione. Il vecchio `artifacts/alphagenome_hierarchical_2026-10-18/official/seed17/checkpoint.pt`, SHA256 `7cac66aefcde376fcb1df0c1141d3f1d3ad7c29e42b37b6069c96b123bb38820`, appartiene alla campagna Array-only a 512kb.

## Contratto di inferenza

```python
from methylation_predictor.modeling.alphagenome import AlphaGenomePredictorFromCheckpoint

predictor = AlphaGenomePredictorFromCheckpoint(
    'artifacts/alphagenome_mixed_2026-10-18/official/finite_pairs/seed17/checkpoint.pt',
    device='cuda',  # anche 'cpu'
)
beta = predictor(
    raw_rna, gene_ids, track_indices, offsets,
    functional_dense, alphagenome, cpg_ids=global_cpg_ids,
)
# beta: tensor [numero_pazienti, numero_CpG]
```

Questo è il contratto dell’API; le variabili devono essere caricate e allineate prima della chiamata:

- `raw_rna`: matrice nelle unità canoniche di `X`, senza applicare nuovamente la standardizzazione; 25.017 geni nell’esatto ordine di `predictor.gene_ids`.
- `functional_dense`: matrice non normalizzata `[CpG, 23]` del locus store. L’adapter applica la normalizzazione salvata.
- `track_indices`, `offsets`: bag CSR delle 4.165 tracce, con offset iniziale zero e finale incluso, lunghezza `numero_CpG + 1`.
- `alphagenome`: `[CpG, 1536]`, dataset **`emb_1bp` del cache nativo a 1.048.576 bp**, nell’ordine dei loci richiesti.
- `global_cpg_ids`: vettore intero degli ID globali MethylProphet allineati a tutte le feature; non indici locali, coordinate o stringhe Illumina. Passarlo per applicare correttamente gli offset dei loci noti.

Il cache precedente a 524.288 bp ha la stessa dimensionalità ma non è intercambiabile. L’adapter valida forme e ordine dei geni, ma non può ricostruire il contesto genomico dai valori del tensore. Verificare provenienza, coordinate e attributi del file. Non servono target di metilazione al momento dell’inferenza. L’esempio concreto di assemblaggio CSR e confronto con output congelati è in `scripts/verify_alphagenome_mixed.py`.

## Dati e dipendenze locali

| Risorsa | Percorso rispetto alla root |
| --- | --- |
| Bundle canonico, read-only | `../data/mhc_experiments_root/datasets/methylprophet_repro_v1` |
| Preparazione compatta e pool ausiliari filtrati | `../data/mhc_experiments_root/derived/methylprophet_table5_tcga_chr1` |
| AlphaGenome nativo chr1 | `../data/alphagenome_cpg_embeddings/chr1.h5` |
| Locus store funzionale | `local_methyl_data/derived/locus_features_v1` |
| FASTA di riferimento per audit | `../CpGRepresentationBenchmark/data/external/reference/hg38.fa` |
| Helper della campagna precedente | `../chr1_alphagenome_v1_2026-10-17` |

Gli script di ricerca importano anche helper delle campagne precedenti: conservare i relativi script e la directory adiacente. L’adapter di inferenza è nel package `src/`; non dipende dal trainer esterno per eseguire il forward. Il cache nativo contiene 2.004.538 loci, `emb_1bp` float16 a 1.536 dimensioni e un distinto `emb_128bp` a 3.072 dimensioni che qui non viene usato. Riferimento GRCh38.d1.vd1, posizione della citosina 1-based, modello AlphaGenome `all_folds`.

Non rigenerare gli split e non scrivere nel bundle canonico. Gli assi Array vengono dal protocollo canonico; del vecchio protocollo derivato si riutilizzano soltanto i pool CpG ausiliari filtrati. Il file di feature nativo è circa 18GB; il cache materializzato della campagna circa 10GB, oltre a checkpoint e matrici di valutazione.

## Ambiente registrato

Python ML locale: `/data2/home/vcivale/miniconda3/envs/methyl-predictor/bin/python`.
Versioni registrate: Python 3.11.16, PyTorch 2.5.1+cu121, CUDA 12.1, NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, h5py 3.16.0. Training su A100 40GB. Lo snapshot completo è `environment.json` nell’archivio della campagna.

Il plotting usa `/data2/home/vcivale/miniconda3/bin/python`, dove è disponibile matplotlib; usare `MPLCONFIGDIR=/tmp/methylpredictor-matplotlib`. I percorsi sono quelli della macchina dell’esperimento, da adattare su altre installazioni.

## Fasi operative

Con ambiente ML attivo e dipendenze locali disponibili:

```bash
python scripts/train_alphagenome_mixed.py --stage prepare
python scripts/train_alphagenome_mixed.py --stage select
python scripts/train_alphagenome_mixed.py --stage refit
python scripts/train_alphagenome_mixed.py --stage evaluate
python scripts/fetch_methylprophet_reference.py
python scripts/compare_alphagenome_methylprophet.py --stage audit
python scripts/compare_alphagenome_methylprophet.py --stage score
```

`--stage all` raggruppa le quattro fasi del trainer. `refit` da solo non sostituisce l’esportazione e valutazione finale. Il fetch richiede accesso Hugging Face già configurato e autorizzazione ai repository gated; non inserire credenziali nei report o nei comandi archiviati.

Per il supplemento e il report, **dopo avere conservato il protocollo preregistrato e gli audit richiesti**:

```bash
python scripts/evaluate_alphagenome_mixed_budget.py
python scripts/verify_alphagenome_mixed.py
MPLCONFIGDIR=/tmp/methylpredictor-matplotlib /data2/home/vcivale/miniconda3/bin/python scripts/report_alphagenome_mixed.py
```

Questi comandi descrivono la sequenza su una campagna predisposta. Non costituiscono una rigenerazione automatica di ogni audit storico: `fairness_supplement_protocol.json`, i witness del congelamento, gli audit delle feature e la provenienza sono stati registrati separatamente. Il verificatore richiede anche gli artefatti di audit e test salvati. Il supplemento e il verificatore usano la directory `OUT` predefinita; non accettano `--out`. Per nuove campagne predisporre esplicitamente questi passaggi, congelare i protocolli prima dello scoring e usare una nuova directory/versione.

I protocolli controllano gli hash del codice. Non forzare il resume dopo modifiche: usare lo snapshot storico per una ripetizione fedele oppure un nuovo output per un nuovo esperimento. Durante la campagna la risoluzione degli ID è stata ottimizzata con un indice ordinato in cache; parità numerica verificata e modifica registrata in `implementation_amendment.json`, con sorgente precedente in `source_pre_lookup_optimization/`.

## Riferimento pubblicato

Predizioni: `MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2`, revisione `c3a4e7e7f5cbfd1373aadc155d6bef7e1e6dcd9e`.
Configurazione: `MethylProphet/tcga_mix_chr1-bs_512-c2b2`, revisione `e3bf682757b37e681c292221c5aaf5456e9949f5`, file `version_0/config.yaml`, SHA256 `f0b74f0c9566591d4e8fffd648ecdfd57e080a559c0af2af8844107b3722596d`.

Si confrontano le predizioni effettivamente rilasciate, senza dover scaricare i pesi del riferimento. I sei parquet e i JSON delle metriche sono conservati con provenienza e hash in `reference/`; gli alias storici e i nomi incompleti dei repository possono non risolvere correttamente.

## Cosa conservare per una consegna o un backup

La directory `artifacts/alphagenome_mixed_2026-10-18/` è esclusa dal versionamento Git: un clone del repository non contiene automaticamente pesi, cache e prove di riproducibilità. Conservare almeno:

| Contenuto | Scopo |
| --- | --- |
| `protocol.json`, `frozen_recipe.json`, `fairness_supplement_protocol.json`, `freeze_witness.json` | Decisioni, selezione e ordine del congelamento |
| `matched_protocol/`, `dev/`, `official/`, `evaluation/` | Assi, sviluppo, refit e predizioni |
| `reference/`, `comparison/`, `budget_matched/` | Riferimento allineato, errori appaiati e controllo di budget |
| `coordinate_audit.json`, `feature_value_audit.json`, `reference_metric_audit.json` | Validità di coordinate, feature, target e metriche |
| `source_snapshot/`, `source_provenance.json`, `source_cache_provenance.json`, `environment.json` | Codice, hash delle dipendenze/dati e ambiente |
| `implementation_amendment.json`, `source_pre_lookup_optimization/`, `rejected_pre_alignment/` | Traccia delle correzioni e del pilot escluso |
| `verification.json`, `pytest.xml`, `pytest_mixed_helpers.xml`, `report_complete.json` | Esiti dei controlli e della generazione report |

I dati sorgente restano dipendenze esterne read-only; i manifest di hash ne consentono il riconoscimento, non li sostituiscono. Per un archivio completo includere anche le feature necessarie all’inferenza e i cache o le sorgenti necessarie alla ripetizione.
