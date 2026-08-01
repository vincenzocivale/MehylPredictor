# Frozen RNA Encoder Readout Search

This stage optimizes the **patient-level output of frozen BulkRNABert** without
loading methylation targets, CpG features, NTv3 embeddings, F2 checkpoints, or a
methylation regressor. The frozen encoder always processes all 19,062 checkpoint
genes; only its contextualized output tokens are subsampled and cached.

The search is staged to separate three questions:

1. can layer mixing or simple statistics recover information lost by mean pooling?
2. do learned gene weights, attentive statistics, or PMA recover more
   within-cancer RNA state?
3. do multi-latent, continuous-value, or module-aware refinements add value after
   a pooler has already won?

The primary selection endpoint is validation `within_r2` on a set of RNA target
genes disjoint from the cached input genes. `total_r2`, effective rank, technical
stability, and biological perturbation sensitivity are safeguards. Methylation is
opened only after the readout is locked.

## 1. Installation and tests

```bash
python -m pip install -r requirements-rna.txt
python -m pip install -e .
pytest -q tests/test_rna_encoder_readout.py
```

## 2. Build the frozen token cache

The corrected full-gene RNA matrix and the official GTEx+ENCODE checkpoint from
R5.2 are reused. The command below caches layers 1, 2 and 3 for 4,096 genes.
It requires roughly 48 GiB in float16. To start economically, use `--layers 2,3`
(~32 GiB).

```bash
python scripts/rna_encoder_readout/build_token_cache.py \
  --rna-h5 artifacts/rna_branch/pretrained/inputs/tcga_rna_full_gene.h5 \
  --metadata artifacts/rna_branch/2026-07-29_first_benchmark/inputs/sample_metadata.parquet \
  --official-repo artifacts/models/multiomics-open-research \
  --extractor scripts/rna_branch/extract_bulkrnabert_torch.py \
  --layers 1,2,3 \
  --gene-count 4096 \
  --selection-seed 17 \
  --output artifacts/rna_encoder_readout/cache/bulkrnabert_l1_l2_l3_4096.h5
```

Update `token_cache.path` in:

```text
configs/rna_encoder_readout/bulkrnabert_layers1_3.yaml
```

Validate the base run:

```bash
python -m methylation_predictor.rna_encoder_readout validate \
  --config configs/rna_encoder_readout/bulkrnabert_layers1_3.yaml
```

After the cache path is correct, the entire baseline + P0 + P1 tranche can be launched with:

```bash
python scripts/rna_encoder_readout/run_first_tranche.py \
  --base configs/rna_encoder_readout/bulkrnabert_layers1_3.yaml
```

The manual commands below remain useful for debugging or selective reruns.

The validation report must show:

```text
methylation_inputs_loaded: false
input_target_gene_overlap: 0
```

## 3. P0: mean and layer mixing

Run the layer-2 mean baseline first:

```bash
python -m methylation_predictor.rna_encoder_readout train \
  --config configs/rna_encoder_readout/bulkrnabert_layers1_3.yaml
```

Its checkpoint is normally:

```text
artifacts/rna_encoder_readout/search/p0_mean_layer2/best.pt
```

Generate the P0 configs. Residual readouts warm-start the matched linear decoders
from the mean baseline and reproduce mean layer 2 exactly at epoch 0.

```bash
python scripts/rna_encoder_readout/make_readout_configs.py \
  --base configs/rna_encoder_readout/bulkrnabert_layers1_3.yaml \
  --stage p0 \
  --seed 17 \
  --warm-start-checkpoint artifacts/rna_encoder_readout/search/p0_mean_layer2/best.pt \
  --output-dir artifacts/rna_encoder_readout/configs/p0

python scripts/rna_encoder_readout/run_readout_screen.py \
  --config-dir artifacts/rna_encoder_readout/configs/p0
```

P0 compares:

- the already trained mean layer-2 baseline;
- a `mean_resume` control with the same warm start and additional decoder optimization;
- mean layer 3;
- trainable scalar mixing of all cached layers;
- concatenated layer means projected back to 256 dimensions.

## 4. P1: pooling search on layer 2

```bash
python scripts/rna_encoder_readout/make_readout_configs.py \
  --base configs/rna_encoder_readout/bulkrnabert_layers1_3.yaml \
  --stage p1 \
  --seed 17 \
  --warm-start-checkpoint artifacts/rna_encoder_readout/search/p0_mean_layer2/best.pt \
  --output-dir artifacts/rna_encoder_readout/configs/p1

python scripts/rna_encoder_readout/run_readout_screen.py \
  --config-dir artifacts/rna_encoder_readout/configs/p1

python scripts/rna_encoder_readout/aggregate_readout_results.py \
  --root artifacts/rna_encoder_readout/search \
  --output artifacts/rna_encoder_readout/readout_screen.csv \
  --baseline-run p0_mean_layer2 \
  --baseline-run p0_mean_resume_layer2
```

P1 includes:

- mean + standard deviation;
- fixed total-, within-cancer-, and inverse-variance weighting;
- learned global gene weights;
- gated attentive statistics pooling;
- PMA with 1, 4 and 8 learned queries.

All learned residual readouts output the layer-2 mean exactly at initialization.
The checkpoint from epoch 0 is eligible to win.

## 5. Selection rule after P1

Advance at most two poolers. A candidate should satisfy all of the following on
validation:

1. `within_r2` improves by at least `+0.02` over both mean layer 2 and the `mean_resume` control;
2. `total_r2` decreases by no more than `0.01`;
3. the improvement is not due to input/target gene overlap (`overlap = 0`);
4. PMA queries do not collapse: inspect `diagnostic_latent_pairwise_cosine_*`,
   `diagnostic_effective_genes_*`, and `attention_gene_weights.csv`.

Do not inspect methylation panels at this point.

## 6. P2: objective search

Pass the exact winning YAML files so query count, hidden size, continuous mode,
and every other architectural setting remain locked.

```bash
python scripts/rna_encoder_readout/make_readout_configs.py \
  --base configs/rna_encoder_readout/bulkrnabert_layers1_3.yaml \
  --stage p2 \
  --winner-config artifacts/rna_encoder_readout/configs/p1/p1_pma_q8.yaml \
  --winner-checkpoint artifacts/rna_encoder_readout/search/p1_pma_q8/best.pt \
  --runner-up-config artifacts/rna_encoder_readout/configs/p1/p1_gated_attentive_stats.yaml \
  --runner-up-checkpoint artifacts/rna_encoder_readout/search/p1_gated_attentive_stats/best.pt \
  --seed 17 \
  --warm-start-checkpoint artifacts/rna_encoder_readout/search/p0_mean_layer2/best.pt \
  --output-dir artifacts/rna_encoder_readout/configs/p2
```

This compares RNA-only objectives:

- total RNA reconstruction;
- within-cancer RNA reconstruction;
- joint total + within-cancer reconstruction;
- joint + technical consistency, only when `token_cache.augmentation_path` is set.

### Optional consistency cache

Generate an augmented full-gene RNA view using the existing technical-view
script, then extract tokens with **the same selected genes**:

```bash
python scripts/rna_encoder_readout/build_token_cache.py \
  --rna-h5 artifacts/rna_encoder_quality/technical_views/<VIEW>.h5 \
  --metadata artifacts/rna_branch/2026-07-29_first_benchmark/inputs/sample_metadata.parquet \
  --official-repo artifacts/models/multiomics-open-research \
  --layers 1,2,3 \
  --selection-from-cache artifacts/rna_encoder_readout/cache/bulkrnabert_l1_l2_l3_4096.h5 \
  --output artifacts/rna_encoder_readout/cache/<VIEW>_l1_l2_l3_4096.h5
```

Set that path as `token_cache.augmentation_path` before generating P2 configs.
The loader rejects different sample or gene ordering.

## 7. P3: layer and gene coverage

Build 2,048- and 8,192-gene caches. Reuse the same full 19,062-gene encoder
forward; only the cached output subset changes.

```bash
python scripts/rna_encoder_readout/build_token_cache.py ... \
  --layers 2,3 --gene-count 2048 \
  --output artifacts/rna_encoder_readout/cache/bulkrnabert_l2_l3_2048.h5

python scripts/rna_encoder_readout/build_token_cache.py ... \
  --layers 2,3 --gene-count 8192 \
  --output artifacts/rna_encoder_readout/cache/bulkrnabert_l2_l3_8192.h5
```

Generate P3 configs:

```bash
python scripts/rna_encoder_readout/make_readout_configs.py \
  --base configs/rna_encoder_readout/bulkrnabert_layers1_3.yaml \
  --stage p3 \
  --winner-config artifacts/rna_encoder_readout/configs/p2/<WINNER>.yaml \
  --winner-checkpoint artifacts/rna_encoder_readout/search/<WINNER>/best.pt \
  --warm-start-checkpoint artifacts/rna_encoder_readout/search/p0_mean_layer2/best.pt \
  --gene-count-cache 2048=artifacts/rna_encoder_readout/cache/bulkrnabert_l2_l3_2048.h5 \
  --gene-count-cache 4096=artifacts/rna_encoder_readout/cache/bulkrnabert_l1_l2_l3_4096.h5 \
  --gene-count-cache 8192=artifacts/rna_encoder_readout/cache/bulkrnabert_l2_l3_8192.h5 \
  --output-dir artifacts/rna_encoder_readout/configs/p3
```

Run 8,192 genes only if 4,096 clearly beats 2,048.

## 8. P4: focused refinements

Only the locked P2/P3 winner is refined.

```bash
python scripts/rna_encoder_readout/make_readout_configs.py \
  --base configs/rna_encoder_readout/bulkrnabert_layers1_3.yaml \
  --stage p4 \
  --winner-config artifacts/rna_encoder_readout/configs/p3/<WINNER>.yaml \
  --winner-checkpoint artifacts/rna_encoder_readout/search/<WINNER>/best.pt \
  --warm-start-checkpoint artifacts/rna_encoder_readout/search/p0_mean_layer2/best.pt \
  --module-membership artifacts/rna_branch/representation_search/modules/hallmark_modules.npz \
  --output-dir artifacts/rna_encoder_readout/configs/p4
```

P4 includes:

- continuous raw-value residual;
- residual within the BulkRNABert expression bin;
- Fourier encoding of the within-bin residual;
- Perceiver-lite only as a focused multi-latent extension;
- module-aware PMA when a module matrix is supplied.

Create a matched random module control:

```bash
python scripts/rna_encoder_readout/make_matched_random_modules.py \
  --input artifacts/rna_branch/representation_search/modules/hallmark_modules.npz \
  --output artifacts/rna_encoder_readout/modules/hallmark_random_matched.npz \
  --seed 17
```

Module-aware pooling advances only if it beats this matched random control.

## 9. Three-seed confirmation

Regenerate only the final configuration with seeds 17, 23 and 41. Keep
`objective.selection_seed: 17` fixed so every seed predicts the same disjoint
target genes. Aggregate deltas seed-by-seed rather than relying on a t-test with
three seeds.

The final representation should improve `within_r2` in all three seeds without
materially degrading total-RNA reconstruction or effective rank.

## 10. Re-run the intrinsic quality audit on the winner

Every training run exports:

```text
<run_dir>/embeddings.h5
```

Apply the winner to technical or biological token-cache views:

```bash
python scripts/rna_encoder_readout/apply_readout.py \
  --config <WINNER_CONFIG> \
  --checkpoint <WINNER_RUN>/best.pt \
  --token-cache <VIEW_TOKEN_CACHE>.h5 \
  --output <WINNER_RUN>/<VIEW>_embeddings.h5
```

Create a compatible intrinsic-quality config:

```bash
python scripts/rna_encoder_readout/make_quality_config.py \
  --base-quality-config configs/rna_encoder_quality/bulkrnabert_tcga.yaml \
  --run-name locked_readout_intrinsic \
  --embeddings <WINNER_RUN>/embeddings.h5 \
  --stability-view multinomial=<WINNER_RUN>/multinomial_embeddings.h5 \
  --stability-view dropout=<WINNER_RUN>/dropout_embeddings.h5 \
  --perturbation-view hypoxia_random=<WINNER_RUN>/hypoxia_random_embeddings.h5 \
  --perturbation-view hypoxia=<WINNER_RUN>/hypoxia_embeddings.h5=hypoxia_random \
  --output artifacts/rna_encoder_readout/locked/quality.yaml \
  --output-dir artifacts/rna_encoder_quality/locked_readout

python -m methylation_predictor.rna_encoder_quality run \
  --config artifacts/rna_encoder_readout/locked/quality.yaml
```

The winner must not reduce same-patient technical cosine by more than 0.01,
within-cancer KNN Jaccard by more than 0.05, or systematically erase the
real-vs-random pathway displacement observed at layer 2.

## 11. Lock the RNA-only representation

```bash
python scripts/rna_encoder_readout/lock_readout.py \
  --run-dir <WINNER_RUN> \
  --output-dir artifacts/rna_encoder_readout/locked \
  --decision-note "Selected using RNA-only validation; no methylation metrics inspected"
```

The locked directory includes the checkpoint, embeddings, configuration,
selected input/target genes, SHA-256 hashes, and an explicit statement that
methylation was not used for selection.

## 12. Only after locking: methylation evaluation

Use the existing nested frozen-adapter design from R5.2 to compare:

1. F2;
2. F2 + BulkRNABert mean layer 3;
3. F2 + locked RNA-only readout;
4. F2 + locked readout shuffled within cancer.

When PMA4/PMA8 wins, preserve `latent_tokens` from `embeddings.h5` and optionally
run a single CpG-query experiment over those compact latent programs. Do not
reopen readout selection using methylation results.
