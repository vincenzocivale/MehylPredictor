# chr1 official split: verified against the released MethylProphet evaluation artifact

**Verified 2026-08-28.** Downloaded the publicly released MethylProphet chr1 evaluation
artifact and compared its sample/CpG ID membership set-for-set against this repo's cached
`tcga_array_chr1` protocol (`protocols/tcga_array_chr1/{array_train_sample_idx,array_val_sample_idx,
array_train_cpg_idx,array_val_cpg_idx}.npy` in the canonical bundle).

- **Source**: HuggingFace dataset `MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2`
  (revision/commit `c3a4e7e7f5cbfd1373aadc155d6bef7e1e6dcd9e`, uploaded 2025-02-03).
  Gated dataset; requires accepting the HF terms on the `MethylProphet` org.
- **Method**: `scripts/benchmark_methylprophet/prepare.py::_array_ids_from_mp_eval` reads
  `eval_results-test.parquet/*.parquet` (columns `group_idx`, `sample_idx`, `cpg_idx`) and
  `group_idx_name_mapping-test.json`, and derives the four ID sets (train/val samples,
  train/val CpGs) directly from which evaluation group each row belongs to — no assumptions,
  no re-random-split, IDs read straight off the released rows.
- **Result: exact set match on all four axes**, not just matching counts:

  | axis | ours | official | identical set |
  |---|---:|---:|---|
  | train samples | 8,260 | 8,260 | **yes** |
  | val samples | 918 | 918 | **yes** |
  | train CpGs | 33,885 | 33,885 | **yes** |
  | val CpGs | 6,742 | 6,742 | **yes** |

- **SHA256 of the downloaded parquet shards** (for independent re-verification):
  ```
  c41343af6b0b888d7d1aa80096cd8fc5b9efb8fd2e694da24d76bbf653e941b9  eval_results-test.parquet/000000.parquet
  341ba5d2ecc785ecd60470cc7aec0d4c4202bf5b4d597cfd9d786ed8860055b4  eval_results-test.parquet/000001.parquet
  47857b985b3119073aff26cc463a266e04cea9347c710ae6a00d6c299968b328  eval_results-test.parquet/000002.parquet
  35bf48e2bad8adc4ffeba769a2ff7ee4d4e9a7c96747e82b23dbb90e1ffa1c8f  eval_results-test.parquet/000003.parquet
  ba6d2c96c8b255c5f535ba15483ceb163538c4c689ad5e9da514a0597c941842  eval_results-test.parquet/000004.parquet
  a2346e6f2fb927ab487e6a06cca08e5acd15b91dc332e1aadddc1e190acd17d2  eval_results-test.parquet/000005.parquet
  ```

## Supersedes the earlier "known divergence" caveat

`docs/BENCHMARK_METHYLPROPHET.md` previously stated a 2-sample discrepancy (this repo's
8,260/918 vs. a paper-text figure of 8,258/920), attributed to an unreproducible Array/WGBS
patient-overlap exclusion step. That caveat was based on the **paper's prose**, not the
**actually released evaluation data** — this verification checks against the real released
artifact and finds it identical to this repo's split, counts included. The paper-text
8,258/920 figure appears to not match what MethylProphet's own released evaluation rows
actually contain. The caveat is retracted for chr1; `docs/BENCHMARK_METHYLPROPHET.md` and
`docs/data/METHYLPROPHET_PROTOCOLS.md` updated accordingly.

## chr123: verified 2026-09-02 (CpG axis exact; sample axis not reproducible from our data)

HF access to the gated exhaustive artifact,
`MethylProphet/eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123`, was approved. Same
method as chr1 above (`_array_ids_from_mp_eval`, now called with a chr123 `expected_counts` dict):

- **Source**: HuggingFace dataset
  `MethylProphet/eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123` (13 parquet shards,
  `eval_results-test.parquet/000000..000012.parquet`), downloaded to
  `methylprophet_official/eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123/` under the
  data root.
- **CpG axis: exact match.** The `note1 ∪ note4` reconstruction (78,211 train / 14,893 val)
  matches the released rows **ID-for-ID**, not just by count. This resolves the axis that was the
  real open item since 2026-09-01 — `tcga_mix_chr123`'s CpG split files were already correct and
  are unchanged.
- **Sample axis: release contains different concrete IDs than chr1's split — investigated,
  reconstruction kept.** The release's Array sample split is **8,258 train / 920 val**, not the
  8,260/918 this repo reuses from chr1's own (separately, genuinely) verified split. Initial
  reasoning suggested this was simply a different-but-valid split within the same 9,178-patient
  universe (the earlier 2026-09-01 "strong indirect evidence" section below, which argued from
  MethylProphet's split-generating *code* that the algorithm is chromosome-blind and therefore
  chr123 "should" reuse chr1's exact split — correct about the algorithm, wrong about the
  conclusion). **Trying to apply the release's IDs directly disproved even the weaker
  same-universe assumption**: 306 of the release's sample_idx values do not exist anywhere in
  this repo's canonical Array HDF5 at all (`tcga_array_official_full.h5`, 241231 snapshot) — the
  release's sample_idx range extends to 10,915, our bundle's only to 10,702 — and a symmetric 306
  of our bundle's own sample_idx values are absent from the release's universe. This is a genuine
  **content** difference (some patients present in one dataset, absent from the other), not an
  ID-extraction or indexing bug: the CpG axis, extracted from the exact same rows via the exact
  same code path, matched with zero discrepancy. Concretely, applying `protocols/tcga_mix_chr123`'s
  sample idx files as the release-exact IDs made `scripts/prepare.py --model cpg_statistics
  --scope chr123` crash immediately (`KeyError: 'array sample_idx value not found: 1903'`) when
  looking up the very first Array row — the release's chr123 sample_idx values are simply not
  valid row keys into this repo's canonical bundle.
- **Action taken**: the sample idx files were briefly overwritten with the release's 8,258/920 IDs,
  found broken against this repo's own data as above, and **reverted** to the chr1-reused
  8,260/918 split (backup of the pre-investigation files still kept at
  `protocols/tcga_mix_chr123/_pre_release_verification_backup_2026-09-02/`, now identical to the
  live files again). `protocol.json`'s `verification` block records both the CpG-axis success and
  the sample-axis finding. **No caches or checkpoints changed as a result** — `derived/
  cpg_statistics/chr123/`, `derived/rna_feature_cache/chr123/`, and the `cpg_statistics/chr123.yaml`
  / `rna_methylation/chr123.yaml` reference checkpoints are unaffected; a rebuild was tried on the
  briefly-corrected split, failed for the reason above, and re-run cleanly once reverted, confirming
  the existing caches are already built from (and still match) the correct, internally-consistent
  split.
- **What this means for "verified"**: chr123's CpG-axis provenance is now on the same footing as
  chr1's (exact ID-for-ID match against a released artifact). The Array sample axis remains a
  *reconstruction* (chr1's algorithm/split, reused), not a literal release-ID match — but that
  reconstruction is now the best one available, since the release's own IDs don't fully overlap
  our canonical bundle's Array universe. This is a materially different, weaker kind of gap than
  "unverified": it's not that verification wasn't attempted, it's that the release and this repo's
  canonical bundle disagree on which ~306-of-9178 (~3.3%) patients even exist in the Array source,
  a genuine snapshot/content difference this repo cannot currently resolve (no access to whatever
  raw snapshot produced the release's larger 10,915-max sample_idx universe).

Test coverage: `tests/test_methylprophet_official_split_verification.py::
test_chr123_cpg_axis_matches_released_evaluation_artifact_exactly` (opt-in via
`MP_EVAL_DIR_CHR123`, asserts the CpG axis only, deliberately not the sample axis);
`tests/test_tcga_canonical_protocol.py::test_chr123_sample_split_matches_chr1` still pins the
(unchanged) chr1-reused 8,260/918 counts unconditionally.

## chr123: partial progress (2026-09-01), superseded by the verification above

Two candidate sources were investigated:

1. **`MethylProphet/eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123`**
   (HuggingFace dataset, `xk-huang` org alias) — this is the parallel artifact to chr1's
   `eval-tcga_mix_chr1-bs_512-c2b2`, and looks like the correct exhaustive eval-rows source.
   **Still gated**: this project's HF account is not yet on the authorized list for this
   specific dataset (a 403 `GatedRepoError` is returned even though the checkpoint below, a
   different repo, is now accessible). Requesting access on
   <https://huggingface.co/datasets/MethylProphet/eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123>
   is the next action needed for full verification.
2. **`MethylProphet/tcga-mix-chr123-bs_512-32xl40s-aws`** (checkpoint repo, access approved by
   the author) — downloaded successfully. It bundles its own W&B/eval logs, including
   `eval/version_12/eval_results-val.parquet` + `group_idx_name_mapping-val.json`, in the exact
   same `group_idx`/`sample_idx`/`cpg_idx` schema `_array_ids_from_mp_eval` already parses for
   chr1. However this bundled log is a **periodic training-time validation subsample**, not the
   full held-out split: `ckpt/version_2/config.yaml` shows the validation dataloader points at
   `data/mds_tokenized/241213-tcga-mix-chr123/val_10_shards` (10 shards only), and the bundled
   eval parquet contains only the `train_cpg_x_val_sample` group (199,555 rows) with **220**
   unique CpGs — far short of the expected 14,893 val CpGs. It cannot be used as-is for a full
   CpG-axis ID match.

   It does, however, contain **920 unique val samples** in that subsample — consistent with (and
   likely equal to) the full val-sample count, since sub-sampling by shard only ever *removes*
   samples, never adds them. This is worth flagging: it matches the **paper-text** chr1-3 count
   (920, from `MethylProphet/docs/DATA.md`'s own published table) rather than the **918** this
   repo's chr123 protocol currently assumes by reusing chr1's actually-released sample split
   (verified 918/8,260 above). `docs/data/METHYLPROPHET_PROTOCOLS.md` §2's assumption that the
   chr123 sample axis is identical to chr1's therefore needs re-examination once the full chr123
   eval-rows dataset is accessible — it's not yet contradicted definitively (a 10-shard
   subsample could in principle still land on 920 by chance even off a 918-sample pool if val
   samples are shard-clustered oddly), but it's a concrete signal that it might differ.

3. **`MethylProphet/docs/DATA.md`** itself (their repo's own published statistics table, checked
   2026-09-01) was considered as a possible extraction shortcut, per the author's pointer. It is
   **not usable as one**: `DATA_STATS.md`'s cited script
   (`scripts/tools/eval/save_idx_and_me.py --filter_by_chr chr1,chr2,chr3`) does not create the
   split, it only *filters by chromosome* an already-split parquet
   (`data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg/*.parquet`) that
   this repo's canonical bundle does not contain — producing it would require running
   MethylProphet's full raw-data download + preprocessing pipeline (`DATA.md`'s "Process"
   section: hundreds of GB, hours of compute), which is explicitly out of scope (the user asked
   only to extract IDs from a released artifact, not to reconstruct their pipeline). `DATA.md`'s
   published *statistics table* is still useful as a second, independent source confirming the
   920 val-sample count above — the CpG counts (78,211 train / 14,893 val) match this repo's
   `note1 ∪ note4` reconstruction exactly (already known), and the published 920 val-sample count
   now agrees with the checkpoint's own training-time validation log (point 2 above), rather than
   this repo's 918-sample assumption.

## chr123 sample axis: strong indirect evidence via MethylProphet's own split code (2026-09-01)

The 920-vs-918 question mark above was substantially resolved (not by a released artifact, but
by reading MethylProphet's own split-generating code, cloned locally in `MethylProphet/`):

- **`scripts/tools/data_preprocessing/split_sample_tcga.py`** (`create_ind_cancer_split`) is the
  exact algorithm: group Array samples by `tissue_idx`, seed=42 `np.random.default_rng`,
  `group.sample(frac=0.9, random_state=rng)` per group (singleton-tissue groups go to train
  unconditionally), no chromosome dependency at all — confirming the split is computed **once**,
  genome-wide, for all of chr1/chr123/genomewide alike (already assumed, now seen directly in
  their code).
- This is a **precise match** for `_reconstruct_array_sample_split` in this repo's own
  `scripts/benchmark_methylprophet/prepare.py` (same grouping, same `frac=0.9`, same seed-42
  `Generator`), which is exactly the reconstruction already verified ID-for-ID against the real
  released chr1 artifact above.
- Critically, `scripts/data_preprocessing/241231-tcga_array.sh` runs `split_sample_tcga.py` with
  `--exclude_sample_name_files data/parquet/241231-tcga/duplicated_sample_names_after_merge.txt`
  — a pre-split exclusion step, sourced from
  `scripts/tools/data_preprocessing/merge_gene_expr_parquet.py`, which flags any patient whose
  **RNA sample name appears in more than one TCGA modality's** gene-expression columns (Array,
  EPIC, WGBS) when merged — i.e., any patient with data in more than one TCGA sequencing source.
  This exclusion, not a WGBS-only quirk, is the real candidate explanation for a train/val count
  that differs from the exclusion-free 8,260/918.
- **Checked directly against this repo's own canonical bundle** (`raw_sample_name` and
  `sample_idx` fields on all three TCGA HDF5 sources, `241231` snapshot): **zero overlap** between
  Array and EPIC, Array and WGBS, and EPIC and WGBS, both by internal `sample_idx` and by raw
  source-name string. Applying MethylProphet's own exclusion algorithm to this repo's current data
  therefore yields an **empty exclusion list** — i.e., their code, run on our data, reproduces the
  exact 8,260/918 split already verified against the real chr1 artifact, not 8,258/920.
- **This repo's `tcga_mix_chr123` and `tcga_mix_chr1` protocol files already store the identical
  sample-split set** (`array_train_sample_idx.npy`/`array_val_sample_idx.npy` are the same 8,260
  and 918 IDs in both, checked directly) — consistent with the algorithm being chromosome-blind.

**Interpretation**: the 920 signal (checkpoint validation log + `DATA.md`'s published table) is
best explained by patient-overlap drift between MethylProphet's `241213` raw snapshot (used to
train the actual released checkpoints) and the `241231` snapshot this repo's bundle derives from
— the same kind of drift already documented for EPIC/WGBS CpG counts in
`docs/data/METHYLPROPHET_PROTOCOLS.md` §3 (+1 EPIC CpG, +244 WGBS CpGs between the two snapshots).
A handful of patients gaining or losing a second-modality measurement between snapshots would
change the exclusion list without changing the algorithm. This is a plausible, evidence-consistent
explanation, not a proven one — it still isn't a literal ID-for-ID match against a chr123-specific
released artifact, which remains blocked on the gated dataset below. But it meaningfully upgrades
confidence in this repo's current chr123 sample split (still reused unchanged from chr1) beyond
"assumed" to "independently reconstructed from MethylProphet's own algorithm, on this repo's own
data, with a well-supported account of the residual discrepancy."

For the **CpG axis** (`note1 ∪ note4`), MethylProphet's code shows the *mechanism*
(`scripts/tools/data_preprocessing/split_cpg_by_index_files.py`, a generic include/exclude-file
union tool — consistent with "note1 ∪ note4" being two such static list files) but not the
*content* of `note1`/`note4` themselves, which are pre-existing static artifacts not derivable
from code. No new evidence on this axis beyond what was already documented.

**Net result**: chr123 is still **not formally verified** by a released-artifact ID match — the
CpG axis remains the real open item, blocked on the gated dataset. No files under
`official_training_data/protocols/tcga_mix_chr123/` were changed. Reproduce the chr1 check
(same method, once chr123 dataset access is granted, swap in the chr123 repo name and the
chr123 `expected_counts` dict — the extraction helper now takes an `expected_counts` parameter
for this) with:

```bash
huggingface-cli download --repo-type dataset MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2 \
  --local-dir <dir>
# then, from the repo root, with PYTHONPATH including scripts/benchmark_methylprophet and src:
python -c "
from pathlib import Path
from prepare import _array_ids_from_mp_eval
print(_array_ids_from_mp_eval(Path('<dir>')))
"
```
