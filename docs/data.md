# Data

> **Legacy path notice.** This document describes the original, self-contained
> training pipeline (`21,792`-gene RNA, chr1-only universe, custom
> `7,304`/`398`/`414` sample split derived from the official MethylProphet
> pool). It is retained for legacy checkpoint reproducibility but is **not**
> the default data path for new experiments. For the current model benchmark
> and MethylProphet-compatible training (all `25,017` genes, official
> Array/EPIC/WGBS splits, chr1/chr1-3 protocols), use the canonical data layer:
> [`docs/data/TCGA_CANONICAL_DATA.md`](data/TCGA_CANONICAL_DATA.md)
> and [`docs/data/METHYLPROPHET_PROTOCOLS.md`](data/METHYLPROPHET_PROTOCOLS.md).
> The two paths are independent; nothing below changed.

All training/evaluation inputs live in a single flat directory (no
sub-folders except the generated dev split), currently
`/raid/DATASETS/MethylPredictionData/` on this machine. Paths are hardcoded
absolute paths in `configs/train.yaml` and the `scripts/*.py` files listed
below — there is no environment-variable override; moving the data root
means editing those paths directly.

## Files

| file | contents | referenced by |
|---|---|---|
| `tcga_rna.h5` | RNA expression matrix, `[8116 samples x 21792 genes]` (keys: `X`, `sample_idx`, `gene_ids`) | `configs/train.yaml` (`data.rna`) |
| `tcga_genome_wide_beta.h5` | DNAm beta matrix, `[8116 samples x 408399 CpGs]`, ~98% coverage (keys: `beta`, `sample_idx`, `cpg_idx`) | `configs/train.yaml` (`data.methylation`) |
| `locus_embeddings.h5` | frozen per-CpG embeddings, `[408399 x 1536]` (keys: `embeddings`, `cpg_idx`) | `configs/train.yaml` (`data.locus_embeddings`) |
| `locus_features.parquet` | per-CpG `cpg_idx`, `pred_ntv3_prior`, `pred_log_var_between`, `pred_log_var_within` — frozen, externally fitted (see [`architecture.md`](architecture.md)) | `configs/train.yaml` (`data.locus_features`) |
| `sample_metadata.parquet` | `sample_idx`, `cancer_type` (32 TCGA cancer types), `split` | `configs/train.yaml` (final-refit `data.sample_metadata`), `scripts/build_dev_split.py`, `scripts/render_final_refit_config.py` |
| `cpg_split_manifest.parquet` | `cpg_idx`, `split` | same as above, for `data.cpg_splits` |
| `genome_wide_features.parquet` | per-CpG `cpg_idx`, `chromosome`, `position`, `mean_train`, plus sequence/region features — used **only** to stratify the dev split by chromosome | `scripts/build_dev_split.py` |
| `manifest.json` | provenance record from the original data-prep run (sources, sha256, split counts) | reference only, not read by training |
| `rna_branch_inputs_dev_seed17/` | generated nested dev split (see [`training.md`](training.md)) — `cpg_split_manifest_dev.parquet`, `sample_metadata_dev.parquet`, `manifest.json` | `configs/train.yaml` (development-stage `data.sample_metadata`/`data.cpg_splits`) |

`genome_wide_features.parquet` is the one file with no small, cheap
regeneration path: it is the output of a per-chromosome NTv3-650M genomic
embedding extraction over hg38 (see [`architecture.md`](architecture.md)),
not something this repo can rebuild on its own.

## Splits

Both samples and CpGs carry their own, independent `split` column
(`train`/`validation`/`test`) in `sample_metadata.parquet` /
`cpg_split_manifest.parquet` — a patient-axis split and a CpG-axis split,
combined as needed at evaluation time:

| axis | train | validation | test |
|---|---|---|---|
| samples | 7,304 | 398 | 414 |
| CpGs | 326,906 | 40,804 | 40,689 |

`validation`+`test` together are the official MethylProphet held-out pool on
each axis; `validation`/`test` themselves are a 50/50 stratified split of
that pool (seed `20260729`, stratified by cancer type for samples / by
chromosome for CpGs — see `manifest.json`'s `derived_validation_test_split`).
`sample_metadata.parquet`'s `train` count (7,304) is smaller than the
official MethylProphet `train_sample` count because RNA expression isn't
available for every officially-assigned sample
(`manifest.json`'s `dropped_na_cancer_type: true`) — `train` here means
*aligned* train rows (every row with usable RNA + beta + prior data), not
literally every official train sample ID.

Because both axes are independently split, four evaluation panels are
meaningful:

| panel | sample split | CpG split | meaning |
|---|---|---|---|
| `in_distribution` | train | train | seen samples, seen CpGs |
| `sample_ood` | test | train | unseen patients, seen CpGs |
| `locus_ood` | train | test | seen patients, unseen CpGs |
| `double_ood` | test | test | **the official test set** — unseen on both axes |

## Nested dev split (development stage only)

`scripts/build_dev_split.py` relabels 10% of `train` (seed 17) to
`dev_heldout`, stratified by chromosome for CpGs and by cancer type for
samples, and writes the result to
`rna_branch_inputs_dev_seed17/{cpg_split_manifest_dev,sample_metadata_dev}.parquet`.
This "relabel in place" design matters because `data.py::load_bundle`
hardcodes the literal string `"train"` when fitting the RNA z-score
standardizer — pointing the development config at these `_dev` files means
the standardizer, reference RNA vector, and variability-tertile thresholds
are automatically fit on the 90% dev-fit pool only, never on `dev_heldout`.

Documented caveat, not corrected: `dev_heldout` CpGs' `pred_ntv3_prior`/
variability values were fit via 5-fold out-of-fold ensemble over the
*original* full `train_cpg` pool, which included these rows before this
relabeling. Standard OOF practice (each CpG's own prediction still comes
from a fold that excluded it), not full leakage, but not re-fit against this
nested split either.

The final-refit stage never reads the `_dev` files — it uses the original,
unmodified `sample_metadata.parquet`/`cpg_split_manifest.parquet` (100% of
official `train_cpg`/`train_sample`).
