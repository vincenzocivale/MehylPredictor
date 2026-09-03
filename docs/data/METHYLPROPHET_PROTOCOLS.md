# MethylProphet protocols: exact vs. matched vs. source-revision drift

This document distinguishes three related but different things people tend
to conflate when comparing a new run to a released MethylProphet checkpoint:
the **exact evaluation protocol**, a **matched training protocol**, and
**source-revision drift** between the `241213` snapshot the released
checkpoint config names and the `241231` snapshot this repo's canonical
bundle was rebuilt from.

## Two-level comparison policy

Comparing a new model to a published MethylProphet checkpoint happens at
two independent levels; only the first is load-bearing for a correctness
claim.

1. **Published-model benchmark (primary).** Evaluate our model and the MP
   checkpoint on exactly the same held-out Array targets --
   `Protocol.evaluation_views()` / `evaluation_finite_counts()`, IDs read
   verbatim from the released split (§1 below). This comparison is rigorous
   regardless of which array:EPIC:WGBS mixing ratio either model trained
   with, because it never touches the mixing question at all.

2. **Matched-source training (secondary).** Train on Array+EPIC+WGBS using
   an explicitly named `source_sampling` policy -- e.g.

   ```yaml
   source_sampling:
     policy: explicit_balanced
     weights:
       array: 0.333333
       epic: 0.333333
       wgbs: 0.333333
   ```

   Never named or labeled `methylprophet_exact`: MethylProphet's own
   internal MDS mixing ratio is not recoverable from the released
   checkpoint config (see the `sampling_method: balanced` section below), so
   nothing here claims to reproduce it. Because the exact ratio is
   genuinely unknown, it becomes a **cheap ablation axis** instead of a
   single arbitrary guess baked into the result:

   | ablation config | policy | weights |
   |---|---|---|
   | `configs/protocols/ablations/tcga_mix_chr1_equal_source.yaml` | `explicit_balanced` | 1/3, 1/3, 1/3 |
   | `configs/protocols/ablations/tcga_mix_chr1_array_heavy.yaml` | `explicit_balanced` | 0.50 / 0.25 / 0.25 |
   | `configs/protocols/ablations/tcga_mix_chr1_proportional_to_measurements.yaml` | `proportional_to_measurements` | pool row count |

   (mirrored for chr1-3 as `tcga_mix_chr123_*`). If the new model wins the
   published-model benchmark (§1) stably across all three, the mixing-ratio
   choice is no longer a plausible objection to the result.

## 1. Exact evaluation protocol

The Array chr1 split is reproduced **bit-exactly** from the released
checkpoint:

| axis | train | val | total |
|---|---|---|---|
| samples | 8,260 | 918 | 9,178 |
| CpGs | 33,885 | 6,742 | 40,627 |

Provenance: `official_training_data/protocols/tcga_array_chr1/`
(`array_train_sample_idx.npy`, `array_val_sample_idx.npy`,
`array_train_cpg_idx.npy`, `array_val_cpg_idx.npy`). These files are read
verbatim by `load_protocol()` -- never regenerated, never re-derived from a
fresh random split. **Independently confirmed 2026-08-28**: compared against
the actual released MethylProphet chr1 evaluation artifact
(`MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2` on HuggingFace) -- exact ID-set
match on all four axes, not just matching counts. See
`results/reference/methylprophet_comparison/chr1_official_split_verification.md`
and `docs/BENCHMARK_METHYLPROPHET.md`'s "Split verified exact" section for the
full record (this also retracts that document's earlier "known divergence
from the paper's published split" claim -- there is none).

The same Array split is reused unchanged by `tcga_array_epic_chr1`,
`tcga_array_wgbs_chr1`, and `tcga_mix_chr1` (verified equal in
`tests/test_tcga_canonical_protocol.py`); it is also, separately, the
sample-axis split for `tcga_mix_chr123` (see §3) because the sample
(patient) split is genome-wide, not chromosome-specific -- it lives on the
Array HDF5's own `sample_split` field, unconditioned on which chromosome is
in view.

The three official Array evaluation panels (`Protocol.evaluation_views()`)
and their exact finite-target counts, regression-tested against the
released `tcga_mix_chr1` checkpoint's split:

| view | samples x CpGs | finite targets |
|---|---|---|
| `train_cpg_x_val_sample` | 918 x 33,885 | 30,574,946 |
| `val_cpg_x_train_sample` | 8,260 x 6,742 | 55,155,121 |
| `val_cpg_x_val_sample` | 918 x 6,742 | 6,129,547 |

There is no official held-out split for EPIC or WGBS -- both are
training-only auxiliary sources in every protocol that includes them; only
Array has an evaluation panel.

## 2. Matched training protocol (chr123)

For chr1-3, the Array split was reconstructed exactly via **`note1 union
note4`** (the two source annotation files identifying, respectively, the
originally-released train pool and the CpGs added by a later note):

| | count |
|---|---|
| total | 93,104 |
| train | 78,211 |
| held-out (val) | 14,893 |

Provenance: `official_training_data/protocols/tcga_mix_chr123/`
(`array_train_cpg_idx.npy`, `array_val_cpg_idx.npy`, `array_train_sample_idx.npy`,
`array_val_sample_idx.npy`; `status:
cpg_axis_exact_verified_sample_axis_reverted_pending_investigation` in `protocol.json`).

**CpG axis verified 2026-09-02** against the exhaustive released evaluation artifact
(`MethylProphet/eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123`), the same method
used for chr1 in §1: **exact match** -- 78,211 train / 14,893 val, identical ID sets to the
released rows. This was the item flagged unverified before this date; it is now resolved, and the
CpG split files are unchanged (already correct).

**Sample axis: investigated, found not reproducible against our own canonical bundle, kept
unchanged.** The earlier assumption in this section (through 2026-09-01) -- that chr123 reuses
chr1's exact 8,260/918 sample split, because MethylProphet's split-generating code
(`split_sample_tcga.py`'s `create_ind_cancer_split`) has no chromosome dependency -- doesn't
survive contact with the actual release: the release's chr123 sample split is 8,258 train / 920
val, and critically, **306 of its sample_idx values don't exist anywhere in this repo's canonical
Array HDF5 at all** (release sample_idx range extends to 10,915; our bundle's only to 10,702) --
a genuine content difference between whatever raw snapshot produced the release and this repo's
`241231` bundle, not an ID-extraction bug (the CpG axis, extracted via the identical code path
from the identical rows, matched exactly). Applying the release's sample IDs to this repo's
protocol broke `scripts/prepare.py --model cpg_statistics` outright (`KeyError: sample_idx value
not found`), confirming they are not valid keys into this repo's data. The sample idx files were
reverted to the chr1-reused 8,260/918 split -- the only one internally consistent with this repo's
own data (backup of the investigation kept at
`tcga_mix_chr123/_pre_release_verification_backup_2026-09-02/`, now identical to the live files
again).

Full record: `results/reference/methylprophet_comparison/chr1_official_split_verification.md`'s
"chr123: verified 2026-09-02" section. `tests/test_methylprophet_official_split_verification.py::
test_chr123_cpg_axis_matches_released_evaluation_artifact_exactly` (opt-in, `MP_EVAL_DIR_CHR123`)
regression-tests the CpG axis; `tests/test_tcga_canonical_protocol.py::
test_chr123_sample_split_matches_chr1` still pins the unchanged chr1-reused sample split.

**No retrain is needed** as a result of this investigation: `derived/cpg_statistics/chr123/` and
`derived/rna_feature_cache/chr123/` were already built from (and still match) the chr1-reused
sample split, and the existing `cpg_statistics/chr123.yaml`/`rna_methylation/chr123.yaml`
checkpoints are unaffected. EPIC/WGBS CpG pools for this protocol still fall under source-revision
drift (§3).

## 3. Source-revision drift: 241213 vs. 241231

The released checkpoint's data loader config names an MDS snapshot,
`241213-tcga-mix-chr1`. This repository's canonical bundle was rebuilt from
raw sources under a *later* snapshot tag, `241231-tcga`. For the Array
source this makes no difference (validated to exact zero error -- see
`docs/data/TCGA_CANONICAL_DATA.md#validation-provenance`), but for EPIC and
WGBS the current `241231` pipeline produces slightly different CpG counts
than the historical `241213`-era numbers documented for the released
checkpoint:

| source | chromosomes | current (`241231`) | documented historical | difference |
|---|---|---|---|---|
| EPIC | chr1 | 71,748 | -- (chr1-only historical count not published) | -- |
| WGBS | chr1 | 1,999,548 | -- | -- |
| EPIC | chr1-3 | 172,723 | 172,722 | +1 |
| WGBS | chr1-3 | 5,396,437 | 5,396,193 | +244 |

Investigated and **not corrected**: the current `241231` preprocessing code
contains no filtering step that would explain removing exactly 1 EPIC CpG
or 244 WGBS CpGs to reproduce the historical numbers, and the historical
figures themselves trace to the `241213` MDS snapshot rather than to a
still-runnable script in this pipeline. Silently dropping loci to force
agreement would introduce a hidden, undocumented filter with no principled
selection rule -- worse than an acknowledged, tested drift. Protocols
touching EPIC/WGBS beyond chr1-Array (`tcga_mix_chr1`, `tcga_mix_chr123`,
`tcga_array_epic_chr1`, `tcga_array_wgbs_chr1`) are therefore marked
**source-revision-compatible**, not exact, for their EPIC/WGBS pools; their
Array component remains exact per §1/§2. `tests/test_tcga_canonical_protocol.py`
pins the *current* (`241231`) counts as the regression baseline going
forward -- if they ever change again, that's a real pipeline change to
investigate, not something to quietly reconcile against `241213`.

## `sampling_method: balanced`

The released `tcga_mix_chr1` checkpoint config sets
`sampling_method: balanced`. Traced to MethylProphet's own source
(`xk-huang/methylprophet`, `src/data/dataset.py`,
`create_methylformer_streaming_dataset`), this is not a MethylProphet-authored
class-balancing routine -- it is passed straight through to the upstream
`mosaicml-streaming` library's `StreamingDataset(sampling_method=...)`. In
that library, `"balanced"` (vs. `"fixed"`) controls what happens when
`epoch_size` is smaller than the underlying, already-mixed shard pool: a
fresh random subset each epoch (`"balanced"`) vs. the same subset reused
every epoch (`"fixed"`). The released configs never override `epoch_size`,
so the flag mainly affects reshuffling behavior, not which
array/EPIC/WGBS mixing ratio gets used -- that ratio was baked in offline,
at MDS-shard build time, via `Stream(proportion=...|repeat=...|choose=...)`
weights not recoverable from the released checkpoint config. See the full
derivation in `src/methylation_predictor/tcga_canonical/sampler.py`'s
module docstring, and the "Two-level comparison policy" section above for
how this repo's `BalancedPairSampler` turns that unknown into a named,
swappable `source_sampling` policy instead of a single hardcoded guess.
