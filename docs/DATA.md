# TCGA canonical data bundle

The MethylProphet-compatible training/evaluation source data lives outside
this repository, at (default, overridable -- see [Path resolution](#path-resolution)):

```
/raid/DATASETS/MethylPredictionData/methylprophet_official/official_training_data
```

Everything under this root is **read-only** from this repository's point of
view: nothing in `src/methylation_predictor/tcga_canonical/` or
`scripts/` writes to it, regenerates it, or drops loci/rows to make counts
match a historical checkpoint. See [Validation provenance](#validation-provenance)
for why, and [`METHYLPROPHET_PROTOCOLS.md`](METHYLPROPHET_PROTOCOLS.md) for
how the released splits are used without being regenerated.

## Artifacts

| artifact | path (relative to root) | shape | keys |
|---|---|---|---|
| RNA | `tcga_rna_official_full.h5` | `(10916, 25017)` | `X`, `gene_ids`, `sample_idx`, `sample_name`, `tissue_idx`, `tissue_name` |
| Array | `methylation/tcga_array_official_full.h5` | `(9178, 408399)` | `beta`, `cpg_idx`, `sample_idx`, `measurement_idx`, `sample_split`, `sample_name`, `raw_sample_name` |
| EPIC | `methylation/epic_full.h5` | `(1706, 740296)` | `beta`, `cpg_idx`, `sample_idx`, `measurement_idx`, `raw_sample_name` |
| WGBS | `methylation/wgbs_full.h5` | `(32, 23047052)` | `beta`, `cpg_idx`, `sample_idx`, `measurement_idx`, `raw_sample_name` |
| Array CpG registry | `registries/array_cpg_map.parquet` | 408,399 rows | `raw_cpg_row`, `cpg_idx`, `chr`, `pos`, `chr_pos`, `group_idx` |
| EPIC CpG registry | `registries/epic_cpg_map.parquet` | 740,296 rows | same columns |
| WGBS CpG registry | `registries/wgbs_cpg_map.parquet` | 23,047,052 rows | same columns |

RNA uses **all 25,017 genes** from `gene_ids`, in that exact column order.
This is the canonical gene set for this repo going forward -- the older
21,792-gene RNA artifact referenced by `docs/data.md` (the pre-existing,
unrelated chr1-only training path) is legacy and must not be reintroduced
here.

## CpG namespace

`cpg_idx` is always the **official MethylProphet global CpG namespace**, as
stored verbatim in each source's own `cpg_idx` HDF5 dataset (and mirrored in
the registries' `cpg_idx` column). This repository does not mint a new CpG
id space. A source's own HDF5 column position (`raw_cpg_row` in the
registries) is an internal detail -- in code it is always called
`source_cpg_col` (see `MethylationSource` in
`src/methylation_predictor/tcga_canonical/bundle.py`) and never surfaces
past the loader.

## Sample mapping

- `sample_idx` is the TCGA patient id, shared across RNA and all three
  methylation sources.
- `measurement_idx` is a source-local row id, **distinct from `sample_idx`**.
  For Array and EPIC these are 1:1 (one measurement per sample). For WGBS,
  32 measurements map to only **31 unique `sample_idx`** -- one patient has
  two WGBS measurements. This duplicate is intentional (a genuine repeated
  measurement in the source data) and must never be deduplicated.
- Every `sample_idx` that appears in any of the three methylation sources
  has a matching RNA row (verified in `tests/test_tcga_canonical_bundle.py`,
  `test_every_source_measurement_resolves_a_real_rna_row`). The loader
  (`RNASource.rows`) raises `KeyError` rather than silently dropping or
  imputing a measurement whose RNA row is missing.

## Protocols available

See [`METHYLPROPHET_PROTOCOLS.md`](METHYLPROPHET_PROTOCOLS.md) for the full
per-protocol split provenance:

- `tcga_array_chr1` -- Array only, chr1, exact released evaluation protocol
- `tcga_array_epic_chr1` -- Array (exact split) + EPIC (auxiliary, no held-out)
- `tcga_array_wgbs_chr1` -- Array (exact split) + WGBS (auxiliary, no held-out)
- `tcga_mix_chr1` -- Array + EPIC + WGBS, chr1 (the released checkpoint's mix)
- `tcga_mix_chr123` -- Array + EPIC + WGBS, chr1-3

## Exact vs. source-revision-compatible

Every protocol falls into one of two provenance tiers, recorded in its own
`protocols/<name>/protocol.json` (`status` field):

- **exact**: IDs read verbatim from a file MethylProphet's release (or an
  exact reconstruction of it) produced -- e.g. `tcga_array_chr1`'s
  8260/918 sample split and 33885/6742 CpG split, and `tcga_mix_chr123`'s
  Array split (`note1_union_note4_exact`). Regression-tested to the exact
  integer.
- **source-revision-compatible**: correct under the *current* (`241231`)
  raw/source pipeline, but not bit-identical to a historical MDS snapshot
  the original checkpoint config names (`241213-tcga-mix-chr1`). EPIC/WGBS
  CpG pools for the mix/chr123 protocols fall here -- see
  `METHYLPROPHET_PROTOCOLS.md` for the exact counts and why they are not
  "fixed" to match.

## Validation provenance

The Array genome-wide artifact (`tcga_array_official_full.h5`,
`(9178, 408399)`) was validated cell-for-cell against the canonical chr1
Array dataset:

- 366,941,981 finite cells compared
- finite-mask mismatch: 0
- MAE / RMSE / max abs error: 0 / 0 / 0

That chr1 block was itself compared against MethylProphet's own official
evaluation output with exactly zero error. On the strength of that chain,
**the Array, EPIC, WGBS, and RNA artifacts listed above are treated as
immutable** -- this repository's data layer only reads them.

## Rules for not regenerating the raw data

1. Never write to any path under the canonical bundle root.
2. Never re-run raw preprocessing to "fix" a count mismatch; if current and
   historical counts disagree (see `METHYLPROPHET_PROTOCOLS.md`'s drift
   section), document the discrepancy, don't paper over it by dropping loci.
3. Treat `protocols/<name>/protocol.json` and the split files next to it
   (`array_train_sample_idx.npy`, `array_train_cpg_idx.{npy,parquet}`, ...)
   as the split of record; the loader (`load_protocol`) only ever reads
   them, never regenerates a random split.
4. `registries/*_cpg_map.parquet` (in particular the 23M-row WGBS registry)
   must never be loaded whole into pandas in training code -- CpG->column
   resolution goes through each source's own (already-loaded) `cpg_idx`
   HDF5 array and `UniqueIndex` (`tcga_canonical/ids.py`), not the
   registries.

## Path resolution

No `/raid/...` path is hardcoded in any Python module under
`src/methylation_predictor/tcga_canonical/`. Resolution order (see
`tcga_canonical/config.py::resolve_bundle_root`):

1. an explicit `root=` argument in code / `--canonical-root` on `scripts/{prepare,train,tune,evaluate}.py`
2. the `TCGA_CANONICAL_ROOT` environment variable
3. `root:` in `configs/data/tcga_canonical.yaml`

## Loader design notes

- HDF5 files stay open and are read lazily, in chunks -- nothing here loads
  the full WGBS matrix, a full registry, or a full source matrix into RAM.
- `sample_idx -> row` and `cpg_idx -> column` maps are vectorized
  (sort + `np.searchsorted`, `tcga_canonical/ids.py::UniqueIndex`), not a
  giant Python dict -- the 23M-row WGBS CpG index would cost roughly a
  gigabyte as a plain dict.
- Array/EPIC are chunked one-row-per-HDF5-chunk, so row-grouped reads are
  cheap; WGBS is chunked with every row in each chunk (32 rows x 8192
  cols), so `MethylationSource.block`/`finite_count` switch to
  column-grouped reads for it automatically (`_column_major`).
