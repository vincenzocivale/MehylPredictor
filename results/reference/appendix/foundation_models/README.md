# Foundation-model masked-CpG comparison — appendix, not a paper result

**Not on `docs/PAPER_ROADMAP.md`** (2026-09-06 reduced paper scope: chr1 → chr123, no ENCODE, no
new code). This directory holds a parallel, non-paper track comparing off-the-shelf pretrained
masked-methylation foundation models (CpGPT, MethylGPT, DeepCpG) against this repo's own
`val_cpg_x_train_sample` view (unseen CpGs, seen patients) — see
`docs/PAPER_EXPERIMENTS.md`'s "Foundation-model masked-CpG comparison" section for the full
status, method, and code pointers (`src/methylation_predictor/benchmark/foundation_models/`).

## Files

- `chr1_crosswalk_coverage.json` — real (not readiness-only) coverage check: what fraction of
  chr1's 6,742 `val_cpg_x_train_sample` held-out CpGs are resolvable through the Illumina
  probe-ID / hg38-position crosswalk both CpGPT (DNA-embedding-mmap lookup) and MethylGPT (fixed
  probe vocabulary) need. **99.99% (6741/6742)**, computed 2026-09-07 by
  `scripts/benchmark_foundation_models/check_crosswalk_coverage.py`.
- `chr1_cpgpt_small_masked_recovery.json` — **first real masked-recovery number**, CpGPT
  `small` checkpoint, zero-shot (no training on this repo's data), chr1
  `val_cpg_x_train_sample` (8,260 context samples x 6,741 crosswalk-covered held-out CpGs,
  99.99% of the official 6,742). Produced 2026-09-07 on this host's GPU (NVIDIA RTX PRO 5000
  Blackwell, `external/cpgpt-env-gpu`, `torch==2.13.0+cu130`) via
  `scripts/benchmark_foundation_models/{prepare_cpgpt_chr1_context,
  run_cpgpt_reconstruct_pilot,score_cpgpt_chr1_reconstruct}.py` (see that section of
  `docs/PAPER_EXPERIMENTS.md` for the exact pipeline). **MAS-PCC 0.3924** (MSE 0.0241, MAE
  0.0840, `mac_pcc` 0.9195) vs. this repo's own reference architecture's 0.5627 on the same
  official chr1 split (`results/reference/ours/01_final_chr1_model.yaml`) — a real,
  substantial gap, zero-shot vs. a model trained on this exact task/data. Not a paper-comparison
  number (see "Not on `docs/PAPER_ROADMAP.md`" above) and not a reproduction of any number
  CpGPT's own paper reports (different task/setting, see `docs/PAPER_EXPERIMENTS.md`).
- `chr1_cpgpt_large_masked_recovery.json` — same setup, CpGPT `large` checkpoint: **MAS-PCC
  0.4330** (MSE 0.0176, MAE 0.0685, `mac_pcc` 0.9431). Better than `small` (expected, larger
  pretrained model) but still well short of this repo's own 0.5627.
- `chr123_crosswalk_coverage.json` — same coverage check, chr123 scope: 14892/14893
  (99.99%), computed 2026-09-07.
- `chr123_cpgpt_small_masked_recovery.json` — CpGPT `small`, chr123
  `val_cpg_x_train_sample` (8,260 context samples x 14,892 covered val CpGs across
  chr1+chr2+chr3, out of 14,893 official). **MAS-PCC 0.3883** (MSE 0.0244, MAE 0.0829,
  `mac_pcc` 0.9212) — closely matches chr1's 0.3924, as expected for a position/DNA-sequence
  model whose per-CpG behavior shouldn't depend much on which chromosome a CpG happens to sit
  on.
- `chr123_cpgpt_large_masked_recovery.json` — CpGPT `large`, same chr123 setup: **MAS-PCC
  0.4342** (MSE 0.0174, MAE 0.0680, `mac_pcc` 0.9445) — closely matches chr1's 0.4330, and
  again better than `small` as expected. Took ~93 minutes on GPU (vs. chr1 `large`'s ~13 min)
  due to GPU/memory contention from other users' jobs on this shared host during the run, not
  a problem with the pipeline itself.

**Summary so far (CpGPT, zero-shot, `val_cpg_x_train_sample` MAS-PCC):**

| Scope | small | large | this repo's own reference |
|---|---|---|---|
| chr1 | 0.3924 | 0.4330 | 0.5627 |
| chr123 | 0.3883 | 0.4342 | (chr123 reference pending, see `results/reference/ours/`) |

CpGPT's chr1 and chr123 numbers are close to each other for both variants -- consistent with
a model whose predictions are driven by DNA sequence/position rather than which chromosome a
CpG sits on. `large` consistently beats `small`; both remain well short of this repo's own
task-trained reference on chr1.

- `{chr1,chr123}_deepcpg_{hou2016_hcc_dna,hou2016_hepg2_dna}_masked_recovery.json` (4 files)
  -- DeepCpG's DNA-only submodel, **patient-agnostic** (same prediction for every sample at a
  given CpG, see `deepcpg_adapter.py`), zero-shot, both released human variants, both scopes.
  Each checkpoint has multiple per-cell output heads (`hou2016_hcc_dna`: 25, from HCC scRRBS
  cells; `hou2016_hepg2_dna`: 6, HepG2 cells); the bulk-comparable prediction reported here is
  the **mean across all of a checkpoint's heads** -- a methodological choice made explicit in
  `run_deepcpg_predict.py`'s docstring, not something DeepCpG's own authors specify for bulk
  data. Because the prediction has zero variance across samples for a fixed CpG, `mas_pcc` is
  numerically degenerate (lands near 0.0 due to floating-point cancellation, not a real
  "no-signal" 0 -- see `score_deepcpg_reconstruct.py`'s docstring) and should not be read; MSE/
  MAE/`mac_pcc` are the meaningful numbers.

  | Scope | Variant | MSE | MAE | mac_pcc |
  |---|---|---:|---:|---:|
  | chr1 | hou2016_hcc_dna | 0.2747 | 0.3978 | **-0.470** |
  | chr1 | hou2016_hepg2_dna | 0.2499 | 0.3750 | **-0.488** |
  | chr123 | hou2016_hcc_dna | 0.2736 | 0.3947 | **-0.467** |
  | chr123 | hou2016_hepg2_dna | 0.2486 | 0.3720 | **-0.489** |

  **Negative correlation for every variant/scope combination** -- a legitimate, notable
  result, not a pipeline bug: DeepCpG's DNA-only submodel was trained on single-cell scRRBS
  data (Hou et al. 2016) and, applied zero-shot to bulk TCGA array methylation, doesn't just
  fail to transfer, it anti-correlates. MSE (~0.25-0.27) is also an order of magnitude worse
  than CpGPT's (~0.02-0.03) or this repo's own reference. Report this as-is; it's informative
  precisely because it's a clean negative result under a real, working pipeline (real hg38
  FASTA window, verified-loaded checkpoint weights), not a broken one.

**Note on script provenance (2026-09-07)**: an earlier pass in this same session (predating a
context summarization this conversation went through) had already built and successfully run
a complete DeepCpG pipeline (`prepare_deepcpg_val_positions.py` + an earlier
`run_deepcpg_predict.py` + `score_deepcpg_masked_recovery.py`, driven by a throwaway
`/tmp/deepcpg_all_chain.sh`) producing these exact same four results. Not remembering that
work after the summarization, this pass rebuilt the same pipeline from scratch under
different script names/interfaces (`run_deepcpg_predict.py` rewritten in place,
`score_deepcpg_reconstruct.py` added alongside the old `score_deepcpg_masked_recovery.py`).
Both pipelines' independently-computed numbers matched to full float precision (as expected,
deterministic computation) -- no data-integrity issue -- but the two now-redundant older
scripts (`prepare_deepcpg_val_positions.py`, `score_deepcpg_masked_recovery.py`) were deleted
to avoid two parallel, drifting implementations of the same scoring logic; the surviving
scripts are `run_deepcpg_predict.py` (rewritten interface: `--locations-json` instead of
`--val-positions <npz>`) and `score_deepcpg_reconstruct.py`.
