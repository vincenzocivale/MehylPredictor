# Locked intrinsic comparison: BulkRNABert vs BulkFormer

## Question

Which frozen encoder exposes the strongest patient-specific bulk-RNA signal,
without using methylation performance to select the representation?

This protocol deliberately excludes methylation matrices, CpG embeddings, NTv3,
F2 checkpoints and methylation losses. It extends the existing
`rna_encoder_quality` audit rather than introducing a second evaluation stack.

## Locked candidates

The primary comparison contains exactly one representation per encoder:

1. **BulkRNABert GTEx+ENCODE**: concatenation of the full-gene means from
   `layer0` through `layer4`. This representation was selected before adding
   BulkFormer.
2. **BulkFormer-147M**: `official_mean`, the mean of the released model's
   `dim+3` gene-token output over the official `interested_gene_list.pt`, matching
   the upstream feature-extraction notebook.

The BulkFormer extractor also writes observed-gene and core-only summaries for
engineering diagnostics. They are intentionally absent from the primary YAML.
Adding the best of those variants after reading validation/test would give
BulkFormer multiple attempts and invalidate the matched comparison.

## Primary estimand and decision rule

Both embeddings are standardized and projected by PCA fitted on the training
patients only to **256 dimensions**. The same multi-output Ridge probe then
predicts 4,096 RNA genes selected by train-only variance.

The primary target is RNA residualized by cancer type using means estimated on
the training split. The winner is the encoder with the highest:

`training-CV within-cancer global R²`

The Ridge penalty is selected by deterministic, cancer-stratified five-fold CV
inside the training split. Validation is used only to check whether the winner
transports. Test metrics are written and explicitly labelled exploratory.

Secondary diagnostics are total-RNA reconstruction, within-cancer mean gene
Pearson, RNA-neighborhood preservation, effective rank, within-cancer effective
rank, between/within variance ratio and linear CKA. They explain the result but
do not override the primary ranking.

## Important provenance limitation

The clean BulkRNABert checkpoint is GTEx+ENCODE-only. For BulkFormer, the public
release describes atlas-scale bulk-RNA pretraining, but an accession-level list
that proves exclusion of the exact TCGA patients used here is not part of this
patch. Keep `--pretraining-overlap-status unknown` unless such evidence is
available. A BulkFormer win under unknown overlap is evidence about accessible
embedding information on this cohort, not yet a clean-transfer claim.

## 1. Extract BulkFormer

Create the upstream environment from `BulkFormer/bulkformer.yaml`, download the
current checkpoint and data assets, then run from the MethylProphetTest root:

```bash
PYTHONPATH=src python scripts/rna_branch/extract_bulkformer_embeddings.py \
  --bulkformer-repo /path/to/BulkFormer \
  --checkpoint /path/to/BulkFormer/model/bulkformer_147M.pt \
  --graph /path/to/BulkFormer/data/G_tcga.pt \
  --graph-weights /path/to/BulkFormer/data/G_tcga_weight.pt \
  --gene-embedding /path/to/BulkFormer/data/esm2_feature_concat.pt \
  --gene-info /path/to/BulkFormer/data/bulkformer_gene_info.csv \
  --interested-gene-indices /path/to/BulkFormer/data/interested_gene_list.pt \
  --rna-h5 artifacts/rna_branch/pretrained/inputs/tcga_rna_full_gene.h5 \
  --values-key X \
  --sample-ids-key sample_idx \
  --gene-ids-key gene_ids \
  --input-scale log2p1_tpm \
  --model-scale 147M \
  --batch-size 1 \
  --device cuda \
  --min-gene-overlap 0.95 \
  --pretraining-overlap-status unknown \
  --output artifacts/rna_branch/pretrained/bulkformer_147m_tcga.h5
```

The conversion `log2(TPM+1) -> ln(TPM+1)` is exact (`x * ln(2)`). Missing model
vocabulary genes receive the upstream mask token `-10`; their fraction is passed
to the model and recorded in the sidecar JSON.

## 2. Merge the locked representations

```bash
PYTHONPATH=src python scripts/rna_branch/merge_encoder_embeddings.py \
  --config configs/rna_encoder_quality/bulkrnabert_vs_bulkformer_merge.yaml
```

The merge fails on duplicate IDs, different sample sets, missing datasets,
non-finite embeddings or invalid dimensions. It reorders the second source to
the first source's exact sample axis and writes a manifest.

## 3. Validate and run

```bash
PYTHONPATH=src python -m methylation_predictor.rna_encoder_quality.cli validate \
  --config configs/rna_encoder_quality/bulkrnabert_vs_bulkformer.yaml

PYTHONPATH=src python -m methylation_predictor.rna_encoder_quality.cli run \
  --config configs/rna_encoder_quality/bulkrnabert_vs_bulkformer.yaml
```

Read these outputs first:

- `encoder_ranking.csv`: locked matched-dimensionality ranking;
- `summary.json`: selected encoder and validation-confirmation flag;
- `report.md`: compact result with exploratory test clearly separated;
- `reconstruction.csv`: complete native-width, matched-width and raw-RNA controls;
- `geometry.csv`, `neighborhood.csv`, `cka.csv`: diagnostic structure.

## Interpretation

- **BulkFormer wins CV and validation confirms**: prioritize BulkFormer for the
  next RNA-only architectural stage, while retaining the pretraining-overlap
  caveat.
- **BulkRNABert wins CV and validation confirms**: BulkFormer does not solve the
  patient-specific accessibility deficit under a capacity-matched linear
  readout; proceed with the planned BulkRNABert decode-then-compress test.
- **CV winner flips on validation**: no encoder is locked as superior. Treat the
  difference as unstable and repeat only with a predeclared split/seed protocol,
  not by choosing whichever test result is larger.
- **Both remain far below PCA-256**: the main conclusion is encoder information
  loss/inaccessibility, regardless of the pairwise winner.
