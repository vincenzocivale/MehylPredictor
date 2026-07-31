# RNA encoder quality audit

This pipeline evaluates **what a frozen RNA encoder represents**, independently of the methylation objective. It never loads beta values, CpG embeddings, an NTv3 prior, an F2 checkpoint, or a methylation regressor. The only learned readouts are fixed closed-form Ridge probes trained to reconstruct RNA quantities from frozen embeddings.

The default configuration targets the corrected R5.2 BulkRNABert export, but the HDF5 contracts are generic enough to audit another bulk or single-cell encoder.


## Scope closed by the cleanup

The removed Stage T and R0/R1/R2/R4 execution files represented completed, non-advancing searches. Their operative conclusion is retained here: train-only z-scored raw RNA with the supervised linear F2 representation remained the reference; rank/quantile encodings, CpG-conditioned experts, Hallmark module queries and sparse gene queries did not advance. The corrected BulkRNABert R5.2 provenance and its current extraction/audit utilities are preserved.

The cleanup removes code and instructions for rerunning those closed grids, not the generated artifacts below `artifacts/` and not the core model classes needed to read historical checkpoints.

## What is measured

The audit has six complementary components.

1. **Geometry (`geometry.csv`)**
   - effective rank, participation ratio and stable rank;
   - optional layer 0 control (gene identity + expression-bin embedding before Transformer blocks);
   - variance concentrated in the first 1/5/10/20 components;
   - raw and centered pairwise cosine similarity;
   - between-cancer / within-cancer variance ratio;
   - effective rank after cancer-type residualization.

2. **Information accessibility (`reconstruction.csv`)**
   - total RNA reconstruction;
   - within-cancer RNA reconstruction using cancer means estimated on train only;
   - alpha selected on validation, one final Ridge fit on train+validation, one test evaluation;
   - PCA-64, PCA-256 and a seeded random projection are matched baselines.

   The primary encoder-quality endpoint is test `global_r2` for `within_cancer_rna`. It rewards patient-specific information while removing the trivial benefit of identifying cancer type.

3. **Geometry preservation (`cka.csv`, `neighborhood.csv`)**
   - linear CKA among all encoder layers and linear baselines;
   - CKA before and after within-cancer residualization;
   - k-nearest-neighbour overlap with a train-fitted RNA PCA reference;
   - within-cancer neighbour overlap, trustworthiness and continuity.

4. **Technical stability (`stability.csv`, optional)**
   - same-patient cosine under technical perturbations;
   - same-patient retrieval and mean reciprocal rank;
   - neighbour preservation globally and within cancer type.

5. **Biological sensitivity (`perturbation.csv`, optional)**
   - displacement caused by a coordinated gene-module perturbation;
   - directional consistency across patients and within cancer type;
   - response magnitude relative to natural within-cancer distances;
   - matched-random control to isolate module content from perturbation size.

6. **Gene-token quality (`token_quality.csv`, optional)**
   - effective rank of the sampled-token pooled representation;
   - agreement between pooling over sampled tokens and the full 19,062-gene mean embedding;
   - how well each contextual gene token preserves its own continuous expression;
   - gain over the scalar expression-bin token alone;
   - how much within-cancer global RNA state is accessible from an individual contextual token.

The token audit is designed to distinguish a weak encoder from a weak mean-pooling readout.

## 1. Apply the patch and install

```bash
cd ~/projects/methylation/MethylProphetTest
git apply --check /path/to/methylprophettest_rna_encoder_quality.patch
git apply /path/to/methylprophettest_rna_encoder_quality.patch
python -m pip install -r requirements-rna.txt
python -m pip install -e .
```

No dependency beyond `requirements-rna.txt` is required.

## 2. Remove superseded RNA-search files

The patch includes an idempotent, conservative cleanup utility. It preserves the corrected BulkRNABert R5.2 implementation and provenance.

Review first:

```bash
python scripts/maintenance/cleanup_obsolete_rna_experiments.py
```

Apply the reviewed deletion set:

```bash
python scripts/maintenance/cleanup_obsolete_rna_experiments.py \
  --apply \
  --report artifacts/maintenance/obsolete_rna_cleanup.json
```

Old Stage B/C search grids are not removed by default. Remove them only after confirming they are no longer needed:

```bash
python scripts/maintenance/cleanup_obsolete_rna_experiments.py \
  --apply --include-historical-grids
```

## 3. Validate the existing corrected BulkRNABert export

The checked-in template expects the corrected R5.2 paths:

```bash
python -m methylation_predictor.rna_encoder_quality validate \
  --config configs/rna_encoder_quality/bulkrnabert_tcga.yaml
```

Validation checks:

- exact sample alignment among RNA, embeddings and metadata;
- unique IDs;
- train/validation/test availability;
- all four embedding datasets;
- finite values and dimensions;
- optional token and stability files.

Adjust only the path fields in the YAML when local artifact locations differ.

## 4. Run the core methylation-independent audit

```bash
python -m methylation_predictor.rna_encoder_quality run \
  --config configs/rna_encoder_quality/bulkrnabert_tcga.yaml
```

Outputs are written to:

```text
artifacts/rna_encoder_quality/bulkrnabert_gtex_encode_tcga/
├── manifest.json
├── summary.json
├── report.md
├── selected_genes.csv
├── geometry.csv
├── cka.csv
├── reconstruction.csv
├── neighborhood.csv
├── stability.csv
├── perturbation.csv
└── token_quality.csv
```

`manifest.json` explicitly records `methylation_inputs_loaded: false` and `downstream_regressor_trained: false`, together with input checksums and software versions.

### Runtime knobs

The most expensive operation is multi-output RNA reconstruction. For a first smoke run, use:

```yaml
analysis:
  reconstruction_gene_count: 512
  pca_dimensions: [64]
  pair_sample_count: 5000
```

For the production audit, restore 4,096 genes and both PCA controls. Gene selection uses variance computed only on train patients.

## 5. Enable the token-level audit

The normal R5.2 HDF5 contains only patient-level mean embeddings. Re-extract once with a deterministic subset of patients and genes:

```bash
python scripts/rna_branch/extract_bulkrnabert_torch.py \
  --config configs/rna_branch/stage_f_bulkrnabert_full_gene.yaml \
  --official-repo artifacts/models/multiomics-open-research \
  --input-scale log2p1 \
  --min-gene-overlap 0.999 \
  --output artifacts/rna_branch/pretrained/bulkrnabert_gtex_encode_v2.h5 \
  --token-sample-output artifacts/rna_encoder_quality/inputs/bulkrnabert_token_sample.h5 \
  --token-sample-count 1024 \
  --token-gene-count 128 \
  --token-sample-seed 17
```

The extractor now also writes `embeddings_layer0` and `tokens_layer0`, the pre-Transformer gene+expression embedding. Existing layer-1–4 datasets and the `embeddings` final-layer alias remain unchanged. The additional HDF5 contains only the selected token states, continuous expression, bin IDs, gene IDs and sample IDs.

To include layer 0, uncomment its entries in both `embeddings.layers` and `token_embeddings.layers`. Then uncomment `token_embeddings` in `configs/rna_encoder_quality/bulkrnabert_tcga.yaml`, then rerun `validate` and `run`.

A token sample must contain train, validation and test patients. The default 1,024-patient deterministic sample normally does so; `validate` fails explicitly when one split is absent.

## 6. Evaluate technical stability

Create two RNA-only technical views: multinomial library resampling and 5% gene dropout, both re-normalized to one million TPM.

```bash
python scripts/rna_encoder_quality/make_technical_views.py \
  --quality-config configs/rna_encoder_quality/bulkrnabert_tcga.yaml \
  --extractor-base-config configs/rna_branch/stage_f_bulkrnabert_full_gene.yaml \
  --output-dir artifacts/rna_encoder_quality/technical_views
```

The command writes two HDF5 matrices and two extractor YAML files. Extract frozen embeddings with the existing production extractor:

```bash
python scripts/rna_branch/extract_bulkrnabert_torch.py \
  --config artifacts/rna_encoder_quality/technical_views/extract_multinomial_50pct.yaml \
  --official-repo artifacts/models/multiomics-open-research \
  --input-scale log2p1 --min-gene-overlap 0.999 \
  --output artifacts/rna_encoder_quality/technical_views/multinomial_50pct_embeddings.h5

python scripts/rna_branch/extract_bulkrnabert_torch.py \
  --config artifacts/rna_encoder_quality/technical_views/extract_gene_dropout_5pct.yaml \
  --official-repo artifacts/models/multiomics-open-research \
  --input-scale log2p1 --min-gene-overlap 0.999 \
  --output artifacts/rna_encoder_quality/technical_views/gene_dropout_5pct_embeddings.h5
```

Add the generated embeddings to the quality YAML:

```yaml
stability_views:
  - name: multinomial_50pct
    path: artifacts/rna_encoder_quality/technical_views/multinomial_50pct_embeddings.h5
    row_ids_key: sample_idx
    layers:
      layer1: embeddings_layer1
      layer2: embeddings_layer2
      layer3: embeddings_layer3
      layer4: embeddings_layer4
  - name: gene_dropout_5pct
    path: artifacts/rna_encoder_quality/technical_views/gene_dropout_5pct_embeddings.h5
    row_ids_key: sample_idx
    layers:
      layer1: embeddings_layer1
      layer2: embeddings_layer2
      layer3: embeddings_layer3
      layer4: embeddings_layer4
```

Then rerun the audit.

## 7. Evaluate sensitivity to coordinated biological programs

Create a perturbation of one or more gene modules and an expression-matched random control. The membership TSV must contain `module_id` and `gene_id`.

```bash
python scripts/rna_encoder_quality/make_module_perturbation_views.py \
  --quality-config configs/rna_encoder_quality/bulkrnabert_tcga.yaml \
  --extractor-base-config configs/rna_branch/stage_f_bulkrnabert_full_gene.yaml \
  --membership artifacts/rna_branch/representation_search/modules/hallmark_aligned.tsv \
  --modules HALLMARK_HYPOXIA,HALLMARK_E2F_TARGETS \
  --fold-change 2.0 \
  --output-dir artifacts/rna_encoder_quality/module_perturbations
```

For every generated real and random-control RNA HDF5, run `extract_bulkrnabert_torch.py` exactly as for the technical views. Then register the embedding outputs:

```yaml
perturbation_views:
  - name: hallmark_hypoxia_random
    path: artifacts/rna_encoder_quality/module_perturbations/hallmark_hypoxia_random_embeddings.h5
    row_ids_key: sample_idx
    layers:
      layer1: embeddings_layer1
      layer2: embeddings_layer2
      layer3: embeddings_layer3
      layer4: embeddings_layer4
  - name: hallmark_hypoxia
    path: artifacts/rna_encoder_quality/module_perturbations/hallmark_hypoxia_embeddings.h5
    row_ids_key: sample_idx
    control: hallmark_hypoxia_random
    layers:
      layer1: embeddings_layer1
      layer2: embeddings_layer2
      layer3: embeddings_layer3
      layer4: embeddings_layer4
```

A high real-vs-random displacement ratio and positive directional-consistency gain indicate that the encoder reacts coherently to the biological program rather than merely to a matched amount of expression change.

## How to interpret the result

- **Good total-RNA R², poor within-cancer R²:** the encoder is dominated by tissue/cancer identity.
- **Good token-level context gain, weak pooled reconstruction:** contextual gene states are useful but mean pooling dilutes them; test a transcriptome-only learned pooling head next.
- **Layer 0/early layers equivalent to layers 3–4:** most useful signal comes from tokenization/static gene embeddings rather than Transformer contextualization.
- **Low effective rank and high top-PC concentration:** sample embeddings are anisotropic or partially collapsed.
- **Good reconstruction but poor technical stability:** representation is informative yet too sensitive to sequencing noise; denoising adaptation is justified.
- **Weak or random-like module response:** the sample embedding does not preserve coherent biological perturbation directions even if static reconstruction is acceptable.
- **Poor reconstruction and poor token quality:** domain-adaptive RNA-only pretraining is more defensible than another downstream methylation architecture.
- **Strong within-cancer reconstruction, stable geometry and useful contextual tokens:** the encoder is adequate; improve pooling before changing encoder weights.

Do not use methylation metrics to select layers or audit settings. The only validation choices in this pipeline are RNA reconstruction hyperparameters and fixed audit controls.
