# Genomic feature store

The production pipeline treats genomic representations as immutable inputs.
No final-model training job invokes NTv3.

## Consolidated NTv3 atlas

Canonical location:

```text
/raid/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1/
  cpg/ntv3/ntv3_cpg_atlas_v1.h5
```

The atlas contains 5,723,092 CpGs with 1536-D float16 embeddings and covers the
408,399 genome-wide Array loci plus the additional chr1-3 loci required by the
canonical EPIC/WGBS protocols.

Embedding protocol is frozen to:

- `InstaDeepAI/NTv3_650M_post`;
- hg38;
- 32,768-bp forward window;
- mean of the central C/G representation;
- BF16 inference, FP16 storage.

## `genomic_prior_v2`

Canonical derived location:

```text
/raid/DATASETS/MethylPredictionData/derived/genomic_prior_v2/array_genomewide/
```

The Array feature table contains 408,399 rows. Prior targets use official Array
training samples only. Official train CpGs are five-fold OOF; official held-out
CpGs are predicted by a probe fit only on all train CpGs.

The rebuild persists both `locus_features.parquet` and every probe checkpoint,
including:

```text
probes/full_fit/probe.pt
```

The selected final RNA model consumes only `pred_ntv3_prior` from this table;
the between-/within-cancer variability predictions are retained for provenance
and historical analyses but are no longer model inputs.

## Auxiliary EPIC/WGBS prior

`scripts/benchmark_methylprophet/prepare.py` extracts the required embeddings from
the consolidated atlas and applies the saved `genomic_prior_v2` full-fit probe
to loci absent from the Array store. It does **not** rerun NTv3 and does **not**
fit another genomic probe.

The final protocol-specific cache is derived, disposable and outside git:

```text
/raid/DATASETS/MethylPredictionData/derived/final_tcga_mix_chr1/
```

The immutable canonical bundle and `genomic_prior_v2` remain the sources of
truth; experiment checkpoints/logs live separately under `experiments/`.
