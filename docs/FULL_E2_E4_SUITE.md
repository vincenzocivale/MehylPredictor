# Full E2-E4 experiment suite

This suite extends the **current** `RNA2DNAmModel` without changing its architecture or objective.
It is additive and does not modify the E0 exact Array-chr1 launcher.

## Experiments

- **E2** — `tcga_mix_chr1`: Array + EPIC + WGBS, full canonical chr1 auxiliary CpG pools.
- **E3** — `tcga_mix_chr123`: Array + EPIC + WGBS, full canonical chr1-3 auxiliary CpG pools.
- **E4** — `array_genomewide`: 408,399 Array CpGs genome-wide, with the legacy manifest's
  `train` (326,906) versus `validation ∪ test` (81,493) split, whose union is the official
  MethylProphet held-out CpG pool documented by this repository.

E2/E3 use all 25,017 canonical RNA genes. E4 uses the same RNA/sample axis and the existing
408,399-locus NTv3 feature universe.

## Why feature expansion is necessary

The current model requires, for every CpG:

1. a 1536-D `NTv3_650M_post` locus embedding (hg38, 32,768 bp, forward orientation,
   central C/G mean),
2. `pred_ntv3_prior`,
3. `pred_log_var_between`,
4. `pred_log_var_within`.

The existing files cover exactly the 408,399 Array loci. E3 contains ~5.4M WGBS loci, so
new NTv3 inference is unavoidable for a full mixed-source run.

The original upstream Ridge/MLP probe weights were not retained in the current training-only
repo or under the canonical data root. The suite therefore **does not refit the prior from
methylation labels**, which would change E1's input semantics. Instead it trains a frozen
NTv3-to-feature **distillation probe** against the already-frozen `locus_features.parquet`:

- existing 408,399 loci keep their original feature values bit-for-bit;
- the probe is used only for newly extracted EPIC/WGBS loci;
- its targets are frozen feature outputs, never held-out methylation labels;
- by default it distils the 81,493 Array official-heldout CpGs, whose frozen features use the
  upstream full-fit inference path rather than the OOF path used on train CpGs;
- each seed uses three independent historical-shape probes (`LayerNorm -> 256 -> 64 -> 1`),
  one for prior, between-variance and within-variance;
- validation and provenance are written under the shared versioned methylation-prior store.

This is the least invasive way to extend the current model when the historical upstream
probe checkpoint is unavailable. If the historical checkpoint is recovered later, the
expanded feature store can be regenerated without changing the mixed-source trainer.

## Holdout semantics

The canonical MethylProphet-compatible protocol intentionally includes some Array `val_cpg`
coordinates in EPIC/WGBS training pools. Therefore two policies are explicit:

- `mp_matched` (default): preserve the canonical auxiliary pools exactly. The three reported
  views are **Array-heldout** views, matching the published-model comparison protocol.
- `strict_global`: remove official Array held-out sample/CpG IDs from every auxiliary source.
  This is the correct ablation when the claim is that a locus was unseen in *all* sources.

Nested development selection is stricter in both modes: its internal dev patients and CpGs
are removed from every source before selecting `best_epoch`.

## I/O and GPU optimizations

The existing Array/EPIC HDF5 files are row-chunked. Reading a random 2k-CpG minibatch directly
therefore reads a full source-width row and discards most columns. The suite addresses this
without rewriting canonical data:

- existing 408k NTv3 embeddings are copied once to NumPy memmaps (float32 by default, preserving values exactly);
- new NTv3 embeddings are sharded, resumable, then merged to a float32 memmap by default;
  `NTV3_STORAGE_DTYPE=float16` is available only as an explicit storage/throughput trade-off;
- Array/EPIC chr1-3 compact caches are derived outside the canonical root;
- WGBS remains on its native `(32,8192)` column-major HDF5 layout;
- source-specific Cartesian blocks default to `128x2048` Array, `128x4096` EPIC,
  and `32x16384` WGBS;
- evaluation reads each Array sample row once per view and iterates CpG chunks from host RAM,
  avoiding the previous repeated full-row HDF5 reads.

Canonical source files are never modified.

## Multi-GPU NTv3 extraction

`run_full_e2_e4.sh` starts one persistent NTv3 process per explicitly selected GPU. Each process
loads the model once and owns every Nth shard. Successful shards receive `.done` markers.
A rerun only computes missing shards.

Example for the current server, leaving GPU0/E1 and GPU3 alone:

```bash
export HG38_FASTA=/absolute/path/to/hg38.fa
GPUS=1,2 MAX_GPUS=2 NTV3_STORAGE_DTYPE=float16 \
  nohup bash scripts/run_full_e2_e4.sh \
  > /raid/DATASETS/MethylPredictionData/experiments/current_model/full_e2_e4/seed17/launcher.log 2>&1 &
```

`HG38_FASTA` must be the GRCh38/hg38 reference matching the canonical registries. Every locus
is validated to have `CG` at the expected 32,768-bp window centre before NTv3 inference.
A coordinate/reference mismatch fails closed.

The launcher never automatically uses an unlisted GPU. `MAX_GPUS` can further cap a long
`GPUS=` list.

## Additional dependency install

Keep the CUDA-compatible PyTorch build already used by E1. Install only the extra genomic
packages:

```bash
python -m pip install -r requirements-genomics.txt
python -m pip install -e . --no-deps
```

## Main artifacts

Generated data and experiment outputs are deliberately separated. The default layout is:

```text
/raid/DATASETS/MethylPredictionData/
├── genomic_features/
│   ├── ntv3_650m_post/
│   │   └── hg38_L32768_forward_cpg_center/
│   │       ├── universe/
│   │       ├── shards/
│   │       └── merged/
│   │           ├── expanded_cpg_idx.npy
│   │           └── expanded_embeddings.f16.npy
│   └── methylation_prior/
│       └── ntv3_650m_post/
│           └── official_array_train_distilled_current_features/
│               ├── probe/
│               └── expanded/
│                   ├── expanded_cpg_idx.npy
│                   ├── expanded_prior.npy
│                   └── expanded_variability.npy
├── derived/
│   └── tcga_canonical/
│       ├── base_array_features_float16/
│       ├── rna_official_array_train_zscore/
│       ├── compact_methylation/tcga_mix_chr123/
│       └── methylprophet_released_eval/
└── experiments/
    └── current_model/full_e2_e4/seed17/
        ├── logs/
        └── runs/
            ├── E2_.../
            ├── E3_.../
            └── E4_array_genomewide/
```

`genomic_features/` never contains seed-specific training outputs. NTv3 embeddings are a deterministic
DNA-derived resource and are reused by every future seed/model/loss that uses the same encoder setup.
`methylation_prior/` is versioned separately because its probe depends on the frozen prior-generation
protocol. `derived/` contains regenerable performance caches. Only checkpoints, metrics and experiment
logs belong under `experiments/`.

The roots are independently overridable with `FEATURE_ROOT`, `DERIVED_ROOT`, `EXPERIMENT_ROOT`,
`NTV3_STORE`, `PRIOR_STORE`, and `CACHE_ROOT`.

## Optional runs

Primary matched-source policy:

```bash
SOURCE_POLICY=equal_source HOLDOUT_POLICY=mp_matched ...
```

Alternative source weights already supported by the canonical repository:

```bash
SOURCE_POLICY=array_heavy ...
SOURCE_POLICY=proportional_to_measurements ...
```

Strict locus/sample OOD ablation in addition to the primary run:

```bash
RUN_STRICT_OOD_ABLATION=1 ...
```

Individual experiments can be disabled with `RUN_E2=0`, `RUN_E3=0`, or `RUN_E4=0`.
