# Exact TCGA benchmark corresponding to MethylProphet Table 5

This document records the exact TCGA chr1 protocol used for the
MethylProphet comparison. Current model/training commands live in
[`WORKFLOWS.md`](WORKFLOWS.md).

## Scope

The published TCGA experiment behind MethylProphet Table 5 is **chromosome 1**,
not a genome-wide TCGA run.  The paper/release trains on a single mixed MDS
containing Array, EPIC and WGBS observations and evaluates three Array-only
held-out views.

This repository represents that paper benchmark as
`methylprophet_table5_tcga_chr1`.  It reuses the same CpG/pair-count
machinery as the older `tcga_array_chr1` / `tcga_mix_chr1` protocol
manifests, which carry the 8,260/918 Array sample split and 33,885/6,742
Array CpG split (`Array HDF5`'s own `sample_split` field).

### Split verified exact against the released evaluation artifact

**2026-08-28: independently verified, not just reconstructed.** Downloaded
the actual released MethylProphet chr1 evaluation artifact
(`MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2` on HuggingFace) and compared
its sample/CpG ID membership set-for-set against this repo's cached
`tcga_array_chr1` protocol. **Exact match on all four axes** (train/val
samples, train/val CpGs) -- not merely matching counts, the actual ID sets
are identical. Full record, checksums, and reproduction command:
[`results/reference/methylprophet_comparison/chr1_official_split_verification.md`](../results/reference/methylprophet_comparison/chr1_official_split_verification.md).
A regression test (`tests/test_methylprophet_official_split_verification.py`,
opt-in via `MP_EVAL_DIR`) re-runs this check against a local copy of the
artifact.

This **retracts an earlier claim** in this document that our split (8,260/918)
diverged from "the paper's published 8,258/920" due to an unreproducible
Array<->WGBS patient-overlap exclusion. That figure came from the paper's
prose, not from the actually-released evaluation data; the real released
rows contain exactly 8,260/918, identical to this repo's split. There is no
known divergence for chr1.

The stratified seed=42 `numpy.random.default_rng(42)` reconstruction in
`scripts/benchmark_methylprophet/prepare.py` (used when `MP_EVAL_DIR` is not
supplied) is kept as the default path since it now has independent
confirmation of producing the correct split; passing `MP_EVAL_DIR` extracts
the exact sample/CpG IDs directly from the released rows instead and is the
stronger audit path when the artifact is available -- preparation requires
the two to agree and fails rather than silently reconciling any future
disagreement.

### chr123: verified 2026-09-02 (CpG axis exact; sample axis not reproducible from our data)

Access to the exhaustive parallel artifact
(`MethylProphet/eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123`)
was approved, downloaded, and checked ID-for-ID the same way as chr1.

- **CpG axis (`note1 ∪ note4`): exact match**, resolving the item that was
  open since 2026-09-01.
- **Sample axis: release IDs don't fully overlap our canonical bundle.**
  The earlier code-reading argument (MethylProphet's split algorithm has no
  chromosome dependency, so chr123 "should" reuse chr1's exact 8,260/918
  split) turned out to be an oversimplification: the release's real chr123
  sample split is **8,258/920**, and 306 of its sample_idx values don't
  exist anywhere in this repo's canonical Array HDF5 at all (release
  sample_idx range extends to 10,915, our bundle's only to 10,702) -- a
  genuine content difference, not an indexing bug (the CpG axis, extracted
  via the same code path from the same rows, matched exactly). Attempting to
  apply the release's sample IDs broke `scripts/prepare.py --model
  cpg_statistics` outright (`KeyError`, missing row). Reverted to the
  chr1-reused 8,260/918 split -- the only one internally consistent with
  this repo's own data, and identical to what the existing
  `derived/cpg_statistics/chr123/` cache and `cpg_statistics/chr123.yaml` /
  `rna_methylation/chr123.yaml` checkpoints already use.

Full record, ID-set diff, and reasoning:
[`results/reference/methylprophet_comparison/chr1_official_split_verification.md`](../results/reference/methylprophet_comparison/chr1_official_split_verification.md)'s
"chr123: verified 2026-09-02" section. A regression test
(`tests/test_methylprophet_official_split_verification.py::test_chr123_cpg_axis_matches_released_evaluation_artifact_exactly`,
opt-in via `MP_EVAL_DIR_CHR123`) re-runs the CpG-axis check.

chr123's CpG-axis provenance is now on the same footing as chr1's (exact
ID-for-ID match). The Array sample axis remains a *reconstruction* (chr1's
algorithm/split, reused) rather than a literal release-ID match -- the
release's own IDs don't fully overlap our canonical bundle's Array universe,
so a literal match isn't currently achievable. No retrain is needed: the
existing 2026-08-31 `cpg_statistics`/`rna_methylation` chr123 checkpoints
already use the (unchanged) chr1-reused split.

## Data contract (this repo's reproducible split)

Training data after the same chromosome/sequence filtering used to create the
MethylProphet MDS:

| source | samples / measurements | CpGs | finite training pairs |
|---|---:|---:|---:|
| Array | 8,260 | 33,885 | 275,093,377 |
| EPIC | 1,706 | 71,748 | 115,856,100 |
| WGBS | 32 measurements | 1,999,446 | 63,982,272 |
| **total** | | | **454,931,749** |

Array evaluation:

| view | samples | CpGs | finite targets |
|---|---:|---:|---:|
| train CpG x val sample | 918 | 33,885 | 30,563,936 |
| val CpG x train sample | 8,260 | 6,742 | 55,154,676 |
| val CpG x val sample | 918 | 6,742 | 6,129,992 |

Preparation fails if any one of these counts differs.  (The paper's
published counts -- 8,258/920 samples, 275,018,849 / 30,638,464 /
55,141,308 / 6,143,360 / 454,857,221 pairs -- are kept in
`docs/data/...` and the paper-comparison tables for reference, but are not
what this reconstruction fails closed on; see above.)

## ID reconstruction

### Array samples

The release preprocessing uses a 90/10 split stratified by `tissue_idx`, with
`numpy.random.default_rng(42)`, after excluding the Array samples that overlap
the WGBS source.  `scripts/benchmark_methylprophet/prepare.py` reimplements that
split on the immutable 9,178-row canonical Array source, stratifying by
`tissue_idx` when present or by factorized `tissue_name` otherwise (see
divergence note above for why the result differs from the paper).

For the strongest possible audit, pass `MP_EVAL_DIR` pointing to the released
MethylProphet evaluation rows.  The preparer then extracts the sample/CpG IDs
from the three released groups and requires them to agree with the reconstructed
Table-5 manifests.  A release-vs-paper discrepancy is an error; it is never
silently reconciled.

### CpGs

Array train/validation CpGs are the released 33,885/6,742 chr1 split already
present in the canonical bundle.

The MethylProphet MDS builder applies another filter after selecting chr1: the
central 1,000-bp hg38 sequence must contain no `N`.  The canonical source
matrices intentionally contain loci before this MDS-specific filter.  The
Table-5 preparer therefore reproduces the rule from source coordinates and the
hg38 FASTA.  This yields the published auxiliary pools, including 1,999,446
WGBS CpGs.  The FASTA is used only for this deterministic filter; NTv3 is never
rerun.

## Historical Table-5 genomic prior

This section documents the genomic-prior construction used by historical benchmark
preparation. The current RNA predictor does not consume the NTv3 embedding or genomic prior
as model features; only the small prior cache may be used for `prior_mse` /
`skill_vs_prior`.

The generic `genomic_prior_v2` was deliberately **not** used in this benchmark:
it was fitted with a different sample split and genome-wide Array labels.  That
would introduce TCGA methylation supervision outside the Table-5 training
universe.

Instead preparation builds `table5_genomic_prior`:

1. prior target = mean beta over the exact 8,260 Array train samples;
2. the 33,885 Array **train** CpGs are served this exact empirical mean directly
   (leakage-safe: it only touches train samples) rather than an NTv3-probe
   approximation of it -- the probe is still fit and 5-fold OOF-scored on this
   same target for auditing (`pred_ntv3_prior_probe_only` in
   `locus_features.parquet`), but the served `pred_ntv3_prior` column uses the
   exact value, not the probe's approximation of it;
3. genomic probe fit scope = the exact 33,885 Array train CpGs on chr1;
4. the 6,742 held-out Array CpGs and auxiliary EPIC/WGBS loci -- where no true
   value is available at train time -- receive predictions from the full-fit
   probe trained only on the 33,885 Array train CpGs;
5. all NTv3 embeddings are copied from `ntv3_cpg_atlas_v1.h5`.

Thus no held-out Array methylation and no off-chr1 TCGA methylation labels enter
model inputs.

## Training exposure

**Historical note**: the detailed exposure discussion below describes the retired
MethylProphet-reproduction trainer. The current RNA model uses
`RNAMethylationTrainer` with the same frozen matched-chr1 data universe plus
the functional atlas/annotation inputs documented in `WORKFLOWS.md`. There is
no selectable shared-backbone engine in the current runtime.

The retired reproduction trainer used a complete Cartesian block schedule for each source.
Every source matrix pair slot is visited exactly once per epoch and NaN targets
are excluded from the loss.  At the end of every epoch it requires the finite
observed counts to equal this repo's reproducible 454,931,749 training
records exactly (see the known-divergence note above).

The block schedule is an implementation optimization for the RNA256 model; it
is **not** claimed to reproduce MethylProphet's optimizer, global batch size or
sample-level shuffle.  The matched claim is about the data universe and one-pass
pair exposure, not identical optimization dynamics.

`FINAL_EPOCHS=1` requests one exact pass over the published training records.
Without an override, the launcher translates the already-frozen architecture
confirmation update budget into an integer number of complete Table-5 epochs;
the chosen value is saved before training in `epoch_budget.json`.

## Published reference metrics

| view | MAS-PCC | MAC-PCC | MSE | MAE |
|---|---:|---:|---:|---:|
| train CpG x val sample | 0.5455 | 0.9320 | 0.0199 | 0.0882 |
| val CpG x train sample | 0.4194 | 0.9065 | 0.0266 | 0.1000 |
| val CpG x val sample | 0.3904 | 0.9059 | 0.0271 | 0.1011 |

The final `headline.json` reports OURS, these published values and
`OURS - MethylProphet` deltas for every metric.  MAS-PCC is the median Pearson
correlation across CpGs (correlation across samples for each CpG); MAC-PCC is
the median Pearson correlation across samples (correlation across CpGs for each
sample), matching MethylProphet's evaluation code.

The head-to-head comparison table (ours vs. this published reference, plus the
historical architecture progression that led to the current model) is kept at
[`results/reference/methylprophet_comparison/table5_chr1.md`](../results/reference/methylprophet_comparison/table5_chr1.md).

## Table 7: training-source comparison (TCGA rows)

**Historical note**: produced by the retired `MethylProphetTrainer` (see the "Training exposure"
note above) — the frozen numbers below remain valid paper-comparison provenance, but are not
reproducible by current code without reintroducing that trainer.

`MethylProphetTrainer` accepted an opt-in `sources: {"array", "epic", "wgbs"}`
subset (default: all three, so the frozen Table-5 path above is unaffected)
plus a pluggable published-reference table for `evaluate()`/`run()`, so the
exact same Array chr1 protocol/split can reproduce MethylProphet paper
**Table 7** ("results of training models on different data sources"), TCGA
rows only (ENCODE rows are a separate, not-yet-done experiment). All three
TCGA rows — T(A), T(A+W), T(A+E) — are complete; results and
`ours`-vs-`published` deltas are in
[`results/reference/methylprophet_comparison/table7_source_ablation.md`](../results/reference/methylprophet_comparison/table7_source_ablation.md).
The published Table 7 reference numbers live in
`TABLE7_PUBLISHED_METHYLPROPHET` (`src/methylation_predictor/benchmark/methylprophet/protocol.py`).
This is a head-to-head paper comparison, not a design ablation — internal
architecture/hyperparameter ablations are tracked separately in
`results/reference/ablations.yaml`.

## Required preflight

The end-to-end `run.sh` launcher (prepare -> `MethylProphetTrainer` -> report) described in
earlier revisions of this section has been retired along with the two-stage architecture. Data
preparation itself is still live and required before any chr1 training:

```bash
python scripts/benchmark_methylprophet/prepare.py \
  --canonical-root "$CANONICAL_ROOT" \
  --atlas "$CANONICAL_ROOT/cpg/ntv3/ntv3_cpg_atlas_v1.h5" \
  --hg38-fasta "$HG38_FASTA" \
  --config configs/benchmark_methylprophet/reference.yaml \
  --output "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --device cuda
```

The preparer still accepts `--atlas` because historical benchmark artifacts and
compatibility outputs are generated by the current preparation path. This must not be
interpreted as an NTv3 dependency of the RNA predictor itself.

Train the current RNA architecture against the matched preparation with
`scripts/train.py --model rna_methylation --scope chr1 --prepared-root ...`
and the functional atlas / annotation inputs documented in `WORKFLOWS.md`.
