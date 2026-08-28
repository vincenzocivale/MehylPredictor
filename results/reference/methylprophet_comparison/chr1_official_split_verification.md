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

## chr123: not verified, access pending

The equivalent released evaluation artifact for chr1-3 (`eval-tcga_mix_chr123-*` or similar)
was not found publicly, and `MethylProphet/tcga-mix-chr123-bs_512-32xl40s-aws` (a model
checkpoint, not an eval-rows dataset) is currently not accessible to this project. The chr123
CpG split (documented elsewhere as `note1 ∪ note4`) therefore remains **unverified** by the
same direct method used here for chr1. Access is still being pursued; until it succeeds,
chr123 should not be presented as a verified MethylProphet-matched comparison scope.
Reproduce this chr1 check with:

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
