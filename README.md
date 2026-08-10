# MethylPredictor

RNA-to-DNAm research code for the current `RNA2DNAmModel`: a patient RNA
representation conditions a residual correction of a frozen per-CpG
methylation prior. New experiments use the MethylProphet-compatible TCGA
canonical data layer: all **25,017 RNA genes**, the official global `cpg_idx`
namespace, and explicit Array/EPIC/WGBS protocols.

The repository intentionally keeps one current model architecture. Historical
training/data paths are retained only for reproducibility and are labelled as
legacy; they must not be mixed silently with the current canonical benchmark.

## Current model

The canonical model uses:

- standardized bulk RNA (`25,017` genes) -> `LayerNorm -> Linear(..., 64)`;
- frozen 1536-D NTv3 locus embeddings;
- frozen per-locus prior plus two variability features;
- a locus-specific variability gate;
- concat/product RNA-CpG interaction;
- mean-RNA anchoring and a zero-initialized residual head.

See [`docs/architecture.md`](docs/architecture.md) for the exact computation.

## Current benchmark status

The exact Array-chr1 benchmark (seed 17) is complete for OURS using the
official MethylProphet split: 8,260/918 train/validation patients and
33,885/6,742 train/held-out CpGs.

| view | MSE | MAE | MAS-PCC | MAC-PCC | skill vs prior |
|---|---:|---:|---:|---:|---:|
| train-CpG x val-sample | 0.022541 | 0.096200 | 0.530027 | 0.918942 | 0.277282 |
| val-CpG x train-sample | 0.020289 | 0.086770 | 0.513238 | 0.928712 | 0.235500 |
| val-CpG x val-sample | 0.020968 | 0.088142 | 0.493754 | 0.926788 | 0.223360 |

The paired MethylProphet prediction-level comparison is currently optional and
may be unavailable when Hugging Face access to the released evaluation dataset
is gated. Do not interpret `cpg_win_fraction`/`sample_win_fraction` from an
OURS-only report as wins against MethylProphet: those fields compare OURS to
the frozen prior.

See [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) for the current
experiment ledger and artifact locations.

## Supported workflows

### Exact Array-chr1 benchmark (current E0)

```bash
bash scripts/run_overnight_current_model_vs_mp.sh
```

This workflow prepares the exact canonical adapter, performs nested-development
selection, final refit on all official train IDs, and evaluates the three
official Array views. Released MethylProphet predictions are used only when
`MP_EVAL_DIR` is explicitly available (or auto-download is explicitly enabled).

### Full canonical suite (E2/E3/E4)

```bash
HG38_FASTA=/path/to/hg38.fa \
  bash scripts/run_full_e2_e4.sh
```

This is the current Array/EPIC/WGBS feature-expansion and training workflow.
See [`docs/FULL_E2_E4_SUITE.md`](docs/FULL_E2_E4_SUITE.md).

Feature preparation can be run without starting E2/E3/E4 by setting:

```bash
RUN_E2=0 RUN_E3=0 RUN_E4=0 \
  HG38_FASTA=/path/to/hg38.fa \
  bash scripts/run_full_e2_e4.sh
```

### Legacy training path

`scripts/train.sh`, `configs/train.yaml`, `docs/data.md`, `docs/training.md`,
and `docs/evaluation.md` describe the older 21,792-gene research path. They are
retained for checkpoint reproducibility, not as the default path for new
MethylProphet-compatible experiments.

## Repository map

- [`docs/architecture.md`](docs/architecture.md) — current model architecture.
- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — current experiment state.
- [`docs/data/TCGA_CANONICAL_DATA.md`](docs/data/TCGA_CANONICAL_DATA.md) — canonical data bundle.
- [`docs/data/METHYLPROPHET_PROTOCOLS.md`](docs/data/METHYLPROPHET_PROTOCOLS.md) — exact/matched protocol semantics.
- [`docs/data/GENOMIC_FEATURE_STORE.md`](docs/data/GENOMIC_FEATURE_STORE.md) — reusable NTv3 feature store.
- [`scripts/README.md`](scripts/README.md) — script/entrypoint map.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-genomics.txt
python -m pip install -e .
```

Generated checkpoints, caches, logs and large genomic features remain outside
git. Canonical source data is treated as read-only.
