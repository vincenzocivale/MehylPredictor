# Genome-wide hg38 locus annotations

The complete static locus representation has three components: 4,165 binary
ENCODE processed-peak overlaps, 18 reference annotation values and five
regulatory breadth values. Together the latter two form the model's 23-dense
dimension input. Every component is CpG-coordinate based and independent of
patients, RNA, methylation labels and measurement technology.

The exact 4,165-track order and selected `ENCFF` files are frozen in
`regulatory/track_contract.tsv`, copied byte-for-byte from the chr1 atlas's
`all_primary_columns.tsv`. Their URLs and MD5s are provenance, never a new
file-selection procedure. There are 533 accessibility, 1,959 histone, 201
CTCF and 1,472 TF-binding tracks. Each bit is one iff the CpG's BED point
overlaps at least one interval in that track's selected processed peak file.
The bit-packed array `features/chrN/regulatory.packbits.npy` stores 4,165
columns as 521 bytes per locus in big-endian bit order within each byte;
column `i` is row `i` of the frozen track contract. No BigWig signal,
peak-score weighting, window aggregation or biosample aggregation is used.

The [breadth contract](../../resources/locus_features/breadth_v1.json)
defines each breadth value as the fraction of overlapping tracks in its
fixed group. The core group has 2,401 tracks: accessibility, CTCF and the
seven frozen histone marks. `features/chrN/breadth.f32.npy` has five columns
in historical order. A full chr1 regression checked every bit against the
frozen CSR and every breadth value against the frozen breadth matrix:
252,488,360 overlap bits, zero mismatches and zero breadth mismatches.
The report is `validation_regulatory_chr1.json` in the master store.

Preparation commands, after inspecting the existing data root and reusing
validated artifacts, are:

```bash
python scripts/prepare_locus_features.py prepare-regulatory-sources \
  --store "$METHYL_DATA_ROOT/derived/locus_features_v1" \
  --reference-atlas "$METHYL_DATA_ROOT/derived/ntv3_functional_peak_atlas_chr1_all_sources"
python scripts/prepare_locus_features.py validate-regulatory-chr1 \
  --store "$METHYL_DATA_ROOT/derived/locus_features_v1" \
  --reference-atlas "$METHYL_DATA_ROOT/derived/ntv3_functional_peak_atlas_chr1_all_sources"
python scripts/prepare_locus_features.py build-regulatory \
  --store "$METHYL_DATA_ROOT/derived/locus_features_v1" \
  --chromosomes all --workers 8
```

`FunctionalLocusCache` reads the common store when its regulatory manifest
and chr1 gate are complete, unpacks requested minibatch rows into
`track_indices` and `offsets`, then concatenates annotation and breadth into
`dense`. The frozen chr1 CSR backend remains supported. TCGA chr1, chr123 and
genomewide views all select rows from this same master store. Future ENCODE
coordinates use the same hg38 `locus_key` and overlap engine; there is no
ENCODE-specific feature schema.

One coordinate, `(chrom, position)`, identifies a CpG across TCGA Array,
EPIC, WGBS, ENCODE and future datasets. `position` is the 1-based C base of
its forward-reference CpG. A BED point is `[position-1, position)`. The engine
checks the hg38 FASTA for `CG`, chromosome bounds and duplicate coordinates.
`locus_key = (chrom_code << 32) | position` is the technology-independent ID.
TCGA `cpg_idx` remains an alias.

The [versioned 18-feature contract](../../resources/locus_features/annotation_core_v1.json)
uses UCSC hg38 CpG islands (four context indicators), GENCODE v50 genes/exons
(four region indicators), ENCODE cCRE Registry V4 (nine class indicators), and
GENCODE v50 gene-record TSSs (one standardized absolute log10 distance).
Island, shore (2 kb), shelf (next 2 kb), open sea have that precedence.
Genomic region precedence is promoter, exon, intron, intergenic. Promoters
are strand-aware: 2 kb upstream and 500 bp downstream of a gene TSS. cCRE
BED overlap uses the frozen class ordering. The raw categories and raw
`dist_tss_abs_log10` are saved separately from `annotation_core.f32.npy`.
The 4,165 regulatory tracks remain in the functional atlas and its five
breadth dimensions remain separate. `FunctionalLocusCache` combines 18 + 5
into the model's unchanged 23-dimensional dense input.

The underlying genomic annotations are reference-genome/static. The
TSS-distance normalization uses preprocessing parameters fitted on the
historical training loci and frozen thereafter; validation, test, ENCODE and
genome-wide inference never contribute to these parameters. The frozen mean
is `3.2289013469204666`, the standard deviation is `0.9242045243544466`.
No new dataset refits them. `target_mu`, `target_sigma`, patient methylation,
mean beta, sample statistics, RNA, cancer type and methylation-fitted priors
remain outside the store.

The master store is `derived/locus_features_v1`. `catalog/chr*.parquet`
contains unique coordinate keys. `aliases/tcga_{array,epic,wgbs}.parquet`
map old IDs to those keys. `features/chr*/` holds mmap-able keys, positions,
18-column float32 arrays, interpretable raw annotations and a complete
manifest with source and contract hashes. `views/tcga_{chr1,chr123,genomewide}.parquet`
contains keys and row indices, without feature copies. An ENCODE dataset can
add coordinates to the same catalog and use the same engine and shards.
The same store supports TCGA chr1/chr123 training, ENCODE genome-wide
training and TCGA genome-wide inference.

Build from the existing canonical registries:

```bash
python scripts/prepare_locus_features.py build-catalog \
  --canonical-root "$METHYL_DATA_ROOT/datasets/methylprophet_repro_v1" \
  --output "$METHYL_DATA_ROOT/derived/locus_features_v1"
python scripts/prepare_locus_features.py validate-annotations-chr1 \
  --store "$METHYL_DATA_ROOT/derived/locus_features_v1" \
  --reference-cache "$METHYL_DATA_ROOT/derived/ntv3_probe_targets/chr1_annotation_features_all_sources" \
  --sources "$METHYL_DATA_ROOT/derived/ntv3_probe_targets/sources" \
  --fai "$METHYL_DATA_ROOT/reference/hg38/hg38.fa.fai"
python scripts/prepare_locus_features.py build-annotations \
  --store "$METHYL_DATA_ROOT/derived/locus_features_v1" \
  --chromosomes all \
  --sources "$METHYL_DATA_ROOT/derived/ntv3_probe_targets/sources" \
  --fai "$METHYL_DATA_ROOT/reference/hg38/hg38.fa.fai"
python scripts/prepare_locus_features.py make-view \
  --store "$METHYL_DATA_ROOT/derived/locus_features_v1" \
  --dataset tcga --scope chr123
```

The full chr1 regression gate covers all 2,004,436 frozen loci. Dimensions
0–16 must match exactly; dimension 17 must differ by no more than `1e-6`.
The recorded run passed, with maximum TSS error `4.76837158203125e-07`.
The frozen cache is never overwritten. New paper-facing training/evaluation
passes the store with `--locus-store`; `FunctionalLocusCache` resolves TCGA
aliases and reads the same chromosome shards. Existing chr1 cache paths remain
supported for legacy checkpoints during migration.

To add a dataset, validate hg38 CpG coordinates, add missing keys to the
catalog, build missing chromosome features with the same engine, and create
an alias or index view. Never create a dataset-specific annotation matrix.
`LocusFeatureStore.lookup_or_compute` can already reuse installed loci and
calculate previously unseen ENCODE-style coordinates with the same engine;
persisting those coordinates requires extending the catalog and explicitly
rebuilding the affected chromosome shard.
