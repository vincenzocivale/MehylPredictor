# Exact TCGA benchmark corresponding to MethylProphet Table 5

## Scope

The published TCGA experiment behind MethylProphet Table 5 is **chromosome 1**,
not a genome-wide TCGA run.  The paper/release trains on a single mixed MDS
containing Array, EPIC and WGBS observations and evaluates three Array-only
held-out views.

This repository represents that paper benchmark as
`methylprophet_table5_tcga_chr1`.  It reuses the same CpG/pair-count
machinery as the older `tcga_array_chr1` / `tcga_mix_chr1` protocol
manifests, and, in practice, reconstructs the *same* 8,260/918 Array sample
split as those manifests (`Array HDF5`'s own `sample_split` field) rather
than the paper's published 8,258/920.

### Known divergence from the paper's published split

The paper's preprocessing excludes Array samples that overlap the WGBS
source before running the stratified 90/10 split, which is why it lands on
8,258/920 instead of 8,260/918.  This repo's canonical bundle carries no
Array<->WGBS patient crosswalk: checking both `sample_idx` and parsed TCGA
patient barcodes finds **zero** overlap between the 9,178 Array rows and the
32 WGBS measurements, so there is nothing to exclude and the reconstructed
seed=42 `numpy.random.default_rng(42)` stratified split naturally reproduces
the 8,260/918 split already baked into the canonical Array source instead.
All finite-pair counts below are this repo's actual, reproducible output for
that split -- they are **not** the paper's published counts.

If you obtain the released MethylProphet evaluation rows (or the original
`split_sample_tcga.py` + Array<->WGBS crosswalk), pass `MP_EVAL_DIR` to
extract the exact sample/CpG IDs directly instead of reconstructing them;
preparation then requires them to agree with the reconstructed manifests
and fails rather than silently reconciling a release-vs-paper discrepancy.

## Data contract (this repo's reproducible split)

Training data after the same chromosome/sequence filtering used to create the
MethylProphet MDS:

| source | samples / measurements | CpGs | finite training pairs |
|---|---:|---:|---:|
| Array | 8,260 | 33,885 | 275,093,377 |
| EPIC | 1,706 | 71,748 | 115,856,100 |
| WGBS | 32 measurements | 1,999,446 | 63,982,272 |
| **total** | | | **454,931,749** |

Array evaluation:

| view | samples | CpGs | finite targets |
|---|---:|---:|---:|
| train CpG x val sample | 918 | 33,885 | 30,563,936 |
| val CpG x train sample | 8,260 | 6,742 | 55,154,676 |
| val CpG x val sample | 918 | 6,742 | 6,129,992 |

Preparation fails if any one of these counts differs.  (The paper's
published counts -- 8,258/920 samples, 275,018,849 / 30,638,464 /
55,141,308 / 6,143,360 / 454,857,221 pairs -- are kept in
`docs/data/...` and the paper-comparison tables for reference, but are not
what this reconstruction fails closed on; see above.)

## ID reconstruction

### Array samples

The release preprocessing uses a 90/10 split stratified by `tissue_idx`, with
`numpy.random.default_rng(42)`, after excluding the Array samples that overlap
the WGBS source.  `scripts/tcga_chr1/prepare.py` reimplements that
split on the immutable 9,178-row canonical Array source, stratifying by
`tissue_idx` when present or by factorized `tissue_name` otherwise (see
divergence note above for why the result differs from the paper).

For the strongest possible audit, pass `MP_EVAL_DIR` pointing to the released
MethylProphet evaluation rows.  The preparer then extracts the sample/CpG IDs
from the three released groups and requires them to agree with the reconstructed
Table-5 manifests.  A release-vs-paper discrepancy is an error; it is never
silently reconciled.

### CpGs

Array train/validation CpGs are the released 33,885/6,742 chr1 split already
present in the canonical bundle.

The MethylProphet MDS builder applies another filter after selecting chr1: the
central 1,000-bp hg38 sequence must contain no `N`.  The canonical source
matrices intentionally contain loci before this MDS-specific filter.  The
Table-5 preparer therefore reproduces the rule from source coordinates and the
hg38 FASTA.  This yields the published auxiliary pools, including 1,999,446
WGBS CpGs.  The FASTA is used only for this deterministic filter; NTv3 is never
rerun.

## Table-5-only genomic prior

The generic `genomic_prior_v2` is deliberately **not** used in this benchmark:
it was fitted with a different sample split and genome-wide Array labels.  That
would introduce TCGA methylation supervision outside the Table-5 training
universe.

Instead preparation builds `table5_genomic_prior`:

1. prior target = mean beta over the exact 8,260 Array train samples;
2. the 33,885 Array **train** CpGs are served this exact empirical mean directly
   (leakage-safe: it only touches train samples) rather than an NTv3-probe
   approximation of it -- the probe is still fit and 5-fold OOF-scored on this
   same target for auditing (`pred_ntv3_prior_probe_only` in
   `locus_features.parquet`), but the served `pred_ntv3_prior` column uses the
   exact value, not the probe's approximation of it;
3. genomic probe fit scope = the exact 33,885 Array train CpGs on chr1;
4. the 6,742 held-out Array CpGs and auxiliary EPIC/WGBS loci -- where no true
   value is available at train time -- receive predictions from the full-fit
   probe trained only on the 33,885 Array train CpGs;
5. all NTv3 embeddings are copied from `ntv3_cpg_atlas_v1.h5`.

Thus no held-out Array methylation and no off-chr1 TCGA methylation labels enter
model inputs.

## Training exposure

The final trainer uses a complete Cartesian block schedule for each source.
Every source matrix pair slot is visited exactly once per epoch and NaN targets
are excluded from the loss.  At the end of every epoch it requires the finite
observed counts to equal this repo's reproducible 454,931,749 training
records exactly (see the known-divergence note above).

The block schedule is an implementation optimization for the RNA256 model; it
is **not** claimed to reproduce MethylProphet's optimizer, global batch size or
sample-level shuffle.  The matched claim is about the data universe and one-pass
pair exposure, not identical optimization dynamics.

`FINAL_EPOCHS=1` requests one exact pass over the published training records.
Without an override, the launcher translates the already-frozen architecture
confirmation update budget into an integer number of complete Table-5 epochs;
the chosen value is saved before training in `epoch_budget.json`.

## Published reference metrics

| view | MAS-PCC | MAC-PCC | MSE | MAE |
|---|---:|---:|---:|---:|
| train CpG x val sample | 0.5455 | 0.9320 | 0.0199 | 0.0882 |
| val CpG x train sample | 0.4194 | 0.9065 | 0.0266 | 0.1000 |
| val CpG x val sample | 0.3904 | 0.9059 | 0.0271 | 0.1011 |

The final `headline.json` reports OURS, these published values and
`OURS - MethylProphet` deltas for every metric.  MAS-PCC is the median Pearson
correlation across CpGs (correlation across samples for each CpG); MAC-PCC is
the median Pearson correlation across samples (correlation across CpGs for each
sample), matching MethylProphet's evaluation code.

## Required preflight

Do **not** launch the expensive training first.  Prepare and audit the protocol:

```bash
HG38_FASTA=/path/to/hg38.fa \
GPU=0 \
PREPARE_ONLY=1 \
nohup bash scripts/tcga_chr1/run.sh \
  > table5_prepare.log 2>&1 &
```

Recommended when the released evaluation artifact is available:

```bash
HG38_FASTA=/path/to/hg38.fa \
MP_EVAL_DIR=/path/to/eval-tcga_mix_chr1-bs_512-c2b2 \
GPU=0 PREPARE_ONLY=1 \
bash scripts/tcga_chr1/run.sh
```

Training is allowed only after the launcher prints:

```text
Table-5 exact protocol preflight: PASS
```

Then launch the one-stage run:

```bash
HG38_FASTA=/path/to/hg38.fa \
GPU=0 \
nohup bash scripts/tcga_chr1/run.sh \
  > table5_train.log 2>&1 &
```
