# RNA representation experiments

This extension keeps the frozen NTv3 prior and the production F2 interaction as the reference.
It adds five matched experiment families:

- **R0**: input value encodings (`zscore`, MethylProphet quantiles, within-sample ranks,
  continuous+rank, continuous+binary);
- **R1**: CpG-conditioned mixture of supervised linear RNA experts;
- **R2**: CpG-query over pathway/regulon/NMF module tokens;
- **R4**: sparse direct CpG-query over gene tokens (final high-cost representation test);
- **R5**: frozen pretrained sample embeddings, alone or concatenated with raw RNA.

The existing metric implementation is reused unchanged: double-OOD MSE/skill, within-cancer
skill, patient-wise and locus-wise correlations, amplitude ratio, and variability-tertile metrics.

## 1. Install

```bash
pip install -r requirements-rna.txt
pip install -e .
```

Use a separate environment for the optional pretrained extractors:

```bash
pip install -r requirements-rna-pretrained.txt
```

## 2. Build clean BulkRNABert embeddings

Clone the official InstaDeep repository, then use the checkpoint pretrained only on
GTEx+ENCODE. This is the primary transfer experiment because it does not contain TCGA patients.

```bash
python scripts/rna_branch/extract_bulkrnabert_embeddings.py \
  --config configs/rna_branch/stage_f_base.yaml \
  --backend official_jax \
  --official-repo /path/to/multiomics-open-research \
  --model-name bulk_rna_bert_gtex_encode \
  --input-scale log2p1 \
  --output artifacts/rna_branch/pretrained/bulkrnabert_gtex_encode.h5
```

On GPUs where the reference JAX implementation cannot materialize its dense
sequence-attention tensor, use the memory-efficient PyTorch SDPA translation
with the same frozen official weights, tokenization and pooling:

```bash
python scripts/rna_branch/extract_bulkrnabert_torch.py \
  --config configs/rna_branch/stage_f_base.yaml \
  --official-repo artifacts/models/multiomics-open-research \
  --model-name bulk_rna_bert_gtex_encode \
  --input-scale log2p1 \
  --min-gene-overlap 0.78 \
  --output artifacts/rna_branch/pretrained/bulkrnabert_gtex_encode.h5
```

Record the backend and gene-overlap threshold in the HDF5 sidecar; this path
is a memory-equivalent implementation, not a claim of bitwise identity with
the reference JAX runtime.

The Hugging Face checkpoint is TCGA-pretrained and is blocked by default. It may only be run as an
explicitly labelled overlap sensitivity analysis with `--allow-tcga-pretraining-overlap`; it must not
be used for the primary claim.

### R5.1 protocol

Use only the official `bulk_rna_bert_gtex_encode` checkpoint for the primary
comparison. The screen has exactly three seed-17 conditions: `replace`,
`concat`, and `shuffled`. `shuffled` uses `replace` but permutes **only** the
frozen embedding within cancer type; raw RNA is never permuted by this control.

```bash
python scripts/rna_branch/make_representation_configs.py \
  --base configs/rna_branch/stage_f_base.yaml \
  --pretrained-rna bulkrnabert=artifacts/rna_branch/pretrained/bulkrnabert_gtex_encode.h5 \
  --r5-seeds 17 --r5-modes replace,concat,shuffled
```

Advance only a non-dominated `replace` or `concat` condition by regenerating
that mode for all three seeds, for example `--r5-seeds 17,23,41 --r5-modes concat`.
Do not advance the shuffled control and do not fine-tune the foundation model
in this tranche.

## 3. Build scGPT pseudo-cell embeddings

Download an official scGPT checkpoint (the whole-human checkpoint is the default first test). If the
RNA matrix uses Ensembl IDs, provide a deduplicated mapping to gene symbols.

```bash
python scripts/rna_branch/extract_scgpt_embeddings.py \
  --config configs/rna_branch/stage_f_base.yaml \
  --model-dir /path/to/scGPT_human \
  --gene-map artifacts/rna_branch/ensembl_to_symbol.tsv \
  --input-scale log2p1 \
  --output artifacts/rna_branch/pretrained/scgpt_whole_human.h5
```

Any other pretrained encoder can be evaluated by writing the same HDF5 contract:
`embeddings [samples, features]`, `sample_idx [samples]`, and `feature_ids [features]`.
For layer-wise scFoundation/Tahoe-X1/Geneformer features produced by an existing pipeline:

```bash
python scripts/rna_branch/pack_precomputed_embeddings.py \
  --embeddings artifacts/external/tahoe_layer12.npy \
  --sample-ids artifacts/external/sample_ids.csv \
  --sample-id-column sample_idx \
  --encoder-name Tahoe-X1 \
  --checkpoint 1.3B \
  --layer 12 \
  --pooling mean \
  --output artifacts/rna_branch/pretrained/tahoe_layer12.h5
```

## 4. Build pathway/regulon tokens

The membership file is long-form TSV with `module_id`, `gene_id`, and optional `weight`.

```bash
python scripts/rna_branch/build_gene_module_matrix.py \
  --config configs/rna_branch/stage_f_base.yaml \
  --membership artifacts/rna_branch/modules/hallmark_or_regulons.tsv \
  --output artifacts/rna_branch/modules/hallmark_or_regulons.npz
```

## 5. Generate matched configs

`best.pt` must be the production F2 checkpoint (`encoder=linear`, `interaction=concat`).

```bash
python scripts/rna_branch/make_representation_configs.py \
  --base configs/rna_branch/stage_f_base.yaml \
  --f2-checkpoint 17=artifacts/rna_branch/stage_f_fusion/first_tranche/f2_concat_product_seed17/best.pt \
  --f2-checkpoint 23=artifacts/rna_branch/stage_f_fusion/confirm/f2_concat_product_seed23/best.pt \
  --f2-checkpoint 41=artifacts/rna_branch/stage_f_fusion/confirm/f2_concat_product_seed41/best.pt \
  --module-weights hallmark=artifacts/rna_branch/modules/hallmark.npz \
  --module-weights hallmark_random_matched=artifacts/rna_branch/modules/hallmark_random_matched.npz \
  --gene-embeddings artifacts/rna_branch/stage_t_gene_tokens/inputs/ntv3_gene_embeddings.npz \
  --pretrained-rna bulkrnabert=artifacts/rna_branch/pretrained/bulkrnabert_gtex_encode.h5 \
  --pretrained-rna scgpt=artifacts/rna_branch/pretrained/scgpt_whole_human.h5 \
  --seeds 17,23,41
```

Validate and train one generated run:

```bash
python -m methylation_predictor.rna_branch.cli validate \
  --config artifacts/rna_branch/representation_search/configs/r1_experts8_s17.yaml
python -m methylation_predictor.rna_branch.cli train \
  --config artifacts/rna_branch/representation_search/configs/r1_experts8_s17.yaml
```

## 6. Advancement rule

A representation should advance only when it improves the three-seed double-OOD result and the
patient-wise dynamic correlation without a material loss in locus-wise correlation. A lower MSE
caused only by increased amplitude is insufficient. For R5, compare both `replace` and `concat`:
`replace` tests representation quality; `concat` tests incremental information beyond the supervised
raw-RNA projection.
