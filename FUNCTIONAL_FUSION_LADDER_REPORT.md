# Functional-only RNA → DNA methylation architecture ladder

Data dello screening: 2026-09-10  
Protocollo: inner development split, Array + EPIC + WGBS  
Seed: 17  
Learning rate: `2e-4`  
Scheduler: `cosine_warmup`, orizzonte 30 epoch, warmup 1 epoch  
Early-stop patience: 8

## Risultati dello screening

| Variant | Epoche | Parametri | Peak VRAM | train CpG × val sample MAS | val CpG × train sample MAS | val CpG × val sample MAS | MAC | MSE | MAE | sec/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **F0** | 20 | 12,777,633 | 12.49 GiB | 0.54259 | **0.49759** | **0.47001** | 0.87834 | 0.03290 | 0.12085 | 79.6 |
| F1 | 16 | 13,829,793 | 21.53 GiB | 0.51557 | 0.48842 | 0.46345 | 0.87763 | 0.03301 | 0.12146 | 119.2 |
| F2 | 20 | 13,568,417 | 21.32 GiB | **0.57335** | 0.49661 | 0.46580 | 0.87675 | 0.03358 | 0.12065 | 74.3 |
| F3 | 16 | 15,149,985 | 38.99 GiB | 0.54176 | 0.48964 | 0.46200 | 0.87711 | 0.03319 | 0.12120 | 139.5 |
| F4 | 16 | 13,571,497 | 22.11 GiB | 0.51873 | 0.48982 | 0.46584 | **0.88000** | **0.03271** | **0.12047** | 76.6 |

Tutti i run hanno usato early stopping:

- F0: migliore epoch 12, stop epoch 20.
- F1: migliore epoch 8, stop epoch 16.
- F2: migliore epoch 12, stop epoch 20.
- F3: migliore epoch 8, stop epoch 16.
- F4: migliore epoch 8, stop epoch 16.

## Conclusione scientifica

F0 è il vincitore secondo la metrica primaria double-OOD:

- `+0.00418` MAS rispetto a F4;
- `+0.00422` MAS rispetto a F2;
- migliore anche su `val CpG × train sample`;
- meno parametri e memoria, con l'inductive bias più semplice.

F2 migliora fortemente `train CpG × val sample` (`0.57335` contro `0.54259` di F0), indicando che il secondo blocco aumenta la capacità di adattamento paziente-locus per loci già osservati. Il vantaggio non si trasferisce però ai CpG non osservati.

F1 e F3 mostrano che aumentare capacità o profondità non produce un miglioramento generalizzabile. F3 è inoltre nettamente svantaggioso per memoria e throughput.

F4 supera F2 sulla metrica primaria di appena `0.000039`, una differenza insufficiente per sostenere un beneficio del gating. Migliora MAC, MSE e MAE, ma peggiora entrambe le viste MAS secondarie. Non costituisce quindi una vittoria indipendente convincente su F2.

## F5 e F6

F5 non è stato classificato:

- il pooling iniziale produceva temporanei incompatibili con batch WGBS reali;
- il fallback memory-safe con segment reduction e activation checkpointing era eccessivamente lento;
- la variante SpMM ha richiesto gestione dedicata del backward FP32/BF16;
- l'ultimo tentativo è stato interrotto prima di produrre un epoch valido.

F6 è stato correttamente saltato: F5 non ha dimostrato un miglioramento e F4 non migliora F2 in maniera scientificamente significativa.

## Raccomandazione

Adottare **F0** come backbone funzionale principale: è il modello più semplice, il migliore sulla metrica primaria e il più efficiente.

I due candidati formalmente meglio classificati sulla metrica primaria sono:

1. F0 — `0.470014`
2. F4 — `0.465835`

Una eventuale conferma scientificamente più informativa potrebbe confrontare F0 e F2, verificando se il forte guadagno di F2 sui loci osservati sia riproducibile senza sacrificare la generalizzazione double-OOD.

## Audit e provenienza

- Hash identico del piano epoch-1: `5b2b9f9a235e36e533dc17e8b6506f00546d3670ba31145dcbaf3c20c5ae4813`
- Git HEAD: `636a7beecf83f3a29e7c7b36f3ab2dc7ae6052f9`
- Base recipe SHA256: `19e22027118e6aba2b5243375294af5c0d6507ee0f7b3125424aaf508389bae7`
- Functional atlas: `/dune/DATASETS/MethylPredictionData/derived/ntv3_functional_peak_atlas_chr1_all_sources`
- Annotation cache: `/dune/DATASETS/MethylPredictionData/derived/ntv3_probe_targets/chr1_annotation_features_all_sources`
- TSV: `/dune/DATASETS/MethylPredictionData/experiments/analysis/functional_fusion_ladder_screening.tsv`
- Markdown generato dal collector: `/dune/DATASETS/MethylPredictionData/experiments/analysis/functional_fusion_ladder_screening.md`

## Run directory

- F0: `/dune/DATASETS/MethylPredictionData/experiments/runs/locus_cls_joint/chr1/functional-fusion-f0-screen-seed17-e30`
- F1: `/dune/DATASETS/MethylPredictionData/experiments/runs/locus_cls_joint/chr1/functional-fusion-f1-screen-seed17-e30`
- F2: `/dune/DATASETS/MethylPredictionData/experiments/runs/locus_cls_joint/chr1/functional-fusion-f2-screen-seed17-e30`
- F3: `/dune/DATASETS/MethylPredictionData/experiments/runs/locus_cls_joint/chr1/functional-fusion-f3-screen-seed17-e30`
- F4: `/dune/DATASETS/MethylPredictionData/experiments/runs/locus_cls_joint/chr1/functional-fusion-f4-screen-seed17-e30`

