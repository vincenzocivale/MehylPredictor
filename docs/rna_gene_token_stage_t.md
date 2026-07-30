# Stage T — gene-semantic RNA representation with frozen NTv3 identities

## Scientific question

Stages A/B/E2/E2.5 closed the search over **global** RNA compressors. They did not test a representation in which every RNA feature remains attached to a semantic gene identity before patient-level pooling.

For patient `s` and gene `j`, Stage T constructs

```text
identity_j = learned ID                (T1)
           | frozen NTv3 TSS embedding (T2)
           | permuted NTv3 embedding   (T3)

token_sj = expression_film(identity_j, standardized_expression_sj)
```

Sixteen learned latent queries pool the ~21,792 gene tokens. The first tranche then feeds the pooled 64-dimensional patient vector to the **exact production F2 interaction**:

```text
[z_s ; g_i_raw ; z_s * P_g(g_i_raw)] -> small MLP -> delta_si
```

This separation is deliberate. T0/T1/T2/T3 first test the RNA representation while holding the combined RNA+CpG representation fixed. The separate, unauthorized `stage_t_gene_locus_query_grid.yaml` asks the later question of whether a CpG should directly query the gene-derived patient latents.

## Controls

| Code | Gene identity | Purpose |
|---|---|---|
| T0 | none; A0 linear RNA + F2 | production baseline |
| T1 | learned gene IDs | does semantic per-gene tokenization help without genomic content? |
| T2 | aligned frozen NTv3 TSS embeddings | does NTv3 genomic content add value beyond tokenization? |
| T3 | fixed permutation of T2 across matched genes | does correct gene-to-sequence assignment matter? |

Unmatched genes remain in the RNA vector. Their fixed NTv3 row is zero, representing one shared unknown identity. T3 permutes only non-zero rows, so annotation coverage is unchanged.

## NTv3 gene embedding contract

The default extraction uses:

- hg38;
- 32,768-bp window around the TSS;
- reverse complement for genes on the minus strand, so orientation follows transcription;
- mean final NTv3 representation within ±128 bp of the TSS;
- frozen checkpoint and float32 saved vectors;
- exact row alignment to the RNA matrix's `gene_ids`.

### 1. Build the RNA-to-GTF manifest

```bash
python -m methylation_predictor.rna_branch.prepare_gene_manifest \
  --rna artifacts/rna_branch/2026-07-29_first_benchmark/inputs/tcga_rna.h5 \
  --values-key X --row-ids-key sample_idx --gene-ids-key gene_ids \
  --gtf /path/to/gencode.hg38.annotation.gtf.gz \
  --output artifacts/rna_branch/stage_t_gene_tokens/inputs/gene_tss_manifest.parquet
```

Inspect coverage and manually audit at least 20 mappings, including versioned Ensembl IDs, symbols, plus/minus strands and unmatched genes.

### 2. Extract NTv3 embeddings in resumable shards

```bash
NUM_SHARDS=32
sbatch --array=0-$((NUM_SHARDS-1)) \
  --export=ALL,NUM_SHARDS="$NUM_SHARDS",\
MANIFEST=artifacts/rna_branch/stage_t_gene_tokens/inputs/gene_tss_manifest.parquet,\
FASTA=/path/to/hg38.fa,\
CHECKPOINT=/or/hf/name/of/NTv3-650M-post,\
OUTPUT_DIR=artifacts/rna_branch/stage_t_gene_tokens/inputs/shards \
  jobs/slurm/run_ntv3_gene_embedding_array.sh
```

### 3. Merge and validate exact RNA-column alignment

```bash
python -m methylation_predictor.rna_branch.merge_ntv3_gene_embeddings \
  --manifest artifacts/rna_branch/stage_t_gene_tokens/inputs/gene_tss_manifest.parquet \
  --shards artifacts/rna_branch/stage_t_gene_tokens/inputs/shards/*.npz \
  --output artifacts/rna_branch/stage_t_gene_tokens/inputs/ntv3_gene_embeddings.npz
```

Before training, verify:

- output rows equal RNA columns;
- `gene_ids` are exactly identical and in the same order;
- all matched vectors are finite and non-zero;
- unmatched rows alone are zero;
- extraction metadata records checkpoint revision, FASTA hash, window, pooling and strand convention.

## Authorized execution order

1. Unit tests and a tiny synthetic forward/backward smoke test.
2. Run `stage_t_gene_token_lr_screen_grid.yaml`; choose one LR per T1/T2/T3 from validation only.
3. Update the three candidate LRs in `stage_t_gene_token_first_tranche_grid.yaml`.
4. Run only the four seed-17 scientific runs T0/T1/T2/T3.
5. Stop and report. Do not run confirm or locus-query grids without authorization.

T0 must reproduce the existing seed-17 F2 result within the established deterministic tolerance. If it does not, stop before interpreting T1-T3.

## Primary comparisons

- **T1 − T0:** value of gene-semantic tokenization.
- **T2 − T1:** incremental value of NTv3 sequence-derived gene identity.
- **T2 − T3:** necessity of the correct gene↔NTv3 assignment.

Selection remains validation beta-MSE/skill. Supporting metrics are patient-wise dynamic correlation, locus-wise correlation, within-cancer skill and variability-tertile skill. Test numbers are descriptive until a multi-seed confirm is authorized.

A candidate advances only if it improves validation MSE, or clearly improves patient-wise/within-cancer dynamics with MSE at parity, and the effect is not reproduced by T3. If T2 does not beat both T1 and T3, the claim that NTv3 gene content improves the representation is rejected.
