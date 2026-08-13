# MethylPredictor

RNA-to-DNAm reconstruction with a frozen genomic prior and patient-specific bulk
RNA conditioning. The production path uses the canonical TCGA bundle and the
MethylProphet-compatible Array/EPIC/WGBS protocols.

## Production architecture

Architecture search is closed. The selected model is:

```text
RNA (25,017 genes)
  -> LayerNorm -> Linear(25,017 -> 256)
  -> z

CpG -> frozen NTv3 embedding e (1536-D)

[z, e, proj(z) * proj(e)]
  -> LayerNorm -> Linear(128) -> GELU -> Dropout(0.1) -> Linear(1)
  -> delta_logit

beta_hat = sigmoid(logit(frozen_prior) + delta_logit)
```

The residual output head is zero-initialized, so optimization starts exactly at
the frozen CpG prior. The previous variability gate, mean-RNA anchor and direct
prediction branch were removed after controlled ablations. See
[`docs/architecture.md`](docs/architecture.md).

The confirming exact Array-chr1 run (`seed=17`) reached **MAS-PCC 0.5163**,
**MSE 0.02058**, and **+23.7% skill vs prior**, improving over RNA256 with the
old gate/anchor retained.

## Paper-final MethylProphet Table-5 benchmark

The TCGA results reported in MethylProphet Table 5 are a **chromosome-1**
Array+EPIC+WGBS experiment. The paper-final path therefore uses a dedicated
`methylprophet_table5_tcga_chr1` protocol rather than the older generic chr1
manifest. Preparation fails closed on this repo's reproducible data
contract: 8,260/918 Array train/validation samples, 33,885/6,742 Array
train/validation CpGs, 71,748 EPIC CpGs, 1,999,446 WGBS CpGs and exactly
454,931,749 finite training pairs. (The paper reports 8,258/920 samples and
454,857,221 pairs after excluding Array/WGBS patient overlap; this bundle
carries no such crosswalk, so preparation reconstructs 8,260/918 instead --
see [`docs/METHYLPROPHET_TABLE5.md`](docs/METHYLPROPHET_TABLE5.md).)

The preparer also reproduces MethylProphet's MDS-specific 1,000-bp hg38
`no-N` filter and builds a Table-5-only OOF genomic prior from the exact Array
training universe. The consolidated NTv3 atlas is reused; NTv3 inference is
never rerun.

Always run the exact-data preflight first:

```bash
HG38_FASTA=/path/to/hg38.fa GPU=0 PREPARE_ONLY=1 \
  bash scripts/run_final_tcga_mix_chr1.sh
```

When the released MethylProphet evaluation artifact is available, set
`MP_EVAL_DIR` so the Array IDs are verified directly against its prediction
rows. The launcher stops on any release-vs-paper discrepancy.

After `Table-5 exact protocol preflight: PASS`, run the one-stage training:

```bash
HG38_FASTA=/path/to/hg38.fa GPU=0 \
nohup bash scripts/run_final_tcga_mix_chr1.sh \
  > table5_train.log 2>&1 &
```

Every final epoch visits the complete source-local CpG x sample block grids and
asserts the exact reproducible finite-pair counts above. This matches the Table-5 data
universe and pair exposure; it does not claim to reproduce MethylProphet's
optimizer or batching. Training scalars are logged to W&B project
**`MethylPredictor`**. See [`docs/METHYLPROPHET_TABLE5.md`](docs/METHYLPROPHET_TABLE5.md).

## Repository map

- `src/methylation_predictor/models.py` — single production architecture.
- `src/methylation_predictor/final_training.py` — one-stage mixed-source trainer.
- `scripts/prepare_final_tcga_mix_chr1.py` — derived cache/preflight builder.
- `scripts/run_final_tcga_mix_chr1.sh` — paper-final entrypoint.
- `scripts/evaluate_current_model_vs_methylprophet.py` — independent exact-view evaluator.
- `scripts/rebuild_genomic_prior_v2.py` — generic genomic-prior builder (not used for Table 5).
- `docs/METHYLPROPHET_TABLE5.md` — exact paper benchmark contract and launch protocol.
- `docs/architecture.md` — architecture-selection evidence and final computation.
- `docs/data/TCGA_CANONICAL_DATA.md` — canonical data bundle.
- `docs/data/METHYLPROPHET_PROTOCOLS.md` — exact evaluation vs matched-source semantics.
- `docs/data/GENOMIC_FEATURE_STORE.md` — NTv3/prior feature provenance.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-genomics.txt
python -m pip install -e .
```

Canonical source data is treated as read-only.
