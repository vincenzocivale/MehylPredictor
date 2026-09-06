# Experiment roadmap

Every pending experiment this repo needs, one row each, with a short **ID** meant to be used
literally as `--run-id` when launching it (so the run directory, checkpoint, and W&B run are all
found by grepping this same string) — new experiments get a fresh ID here; experiments already
in flight keep the run-id they were actually launched with (noted in the "run-id" column when it
differs from the ID). Designed so independent rows can be handed to different machines and run in
parallel: the "depends on" column is the only ordering constraint that matters across machines —
everything else is independent.

All commands assume the placeholders used elsewhere in this repo's docs:
`$CANONICAL_ROOT` (the frozen TCGA bundle), `$DERIVED_ROOT` (derived caches), `$EXPERIMENT_ROOT`
(where runs get written). See `CLAUDE.md`'s "Commands" section for what each resolves to on this
machine.

## Quick reference

| ID | Priority | Status | Depends on | What it produces |
|---|---|---|---|---|
| [`baseline-global-shift`](#baseline-global-shift) | P1 | not started | — | `results/reference/baselines/global_rna_shift/chr1.yaml` |
| [`baseline-bilinear`](#baseline-bilinear) | P1 | not started | — | `results/reference/baselines/bilinear_rna_cpg/chr1.yaml` |
| [`baseline-mlp`](#baseline-mlp) | P1 | not started | — | `results/reference/baselines/mlp_rna_cpg/chr1.yaml` |
| [`seed_variance_reference`](#seed_variance_reference) | **P0** | stalled, resumable | — | chr1 noise floor + the missing 80-epoch convergence number |
| [`p0-row-c`](#p0-row-c) | P1 | interrupted at epoch 21/80, resumable | — | does trunk depth help the reference encoder at all (closes an open question in `docs/RNA_METHYLATION.md`) |
| [`trunk_plain_d4`](#trunk_plain_d4) | P2 | stalled, resumable | — | depth-4 control (does depth alone help the reference architecture) |
| [`chr123-cache`](#chr123-cache) | P3 | not started | — | `$DERIVED_ROOT/methylprophet_compact_chr123/` |
| [`chr123-ref`](#chr123-ref) | P3 | not started | `chr123-cache` | `results/reference/rna_methylation/chr123.yaml` |
| [`enc_mlp`](#not-yet-started-architecture-novelty-arms) | P4 | not started | — | encoder-capacity control |
| [`enc_locus_attention_k64`](#not-yet-started-architecture-novelty-arms) | P4 | not started | — | locus-attention, 64 program tokens |
| [`enc_locus_attention_k128`](#not-yet-started-architecture-novelty-arms) | P4 | not started | — | locus-attention, 128 program tokens |
| [`trunk_plain_d2`](#not-yet-started-architecture-novelty-arms) | P4 | not started | — | depth-2 control |
| [`trunk_plain_d8`](#not-yet-started-architecture-novelty-arms) | P4 | not started | `trunk_plain_d4` | depth-8 control (where does depth saturate) |
| [`axial_cpg_d4`](#not-yet-started-architecture-novelty-arms) | P4 | not started | — | genomic-neighbour (co-methylation) attention |
| [`beta_likelihood_head`](#not-yet-started-architecture-novelty-arms) | P4 | not started | — | bounded heteroscedastic output head |

`collect-arch-results` (not a training run — run any time, re-runnable, safe to call after every
row above that's a `results/reference/ablations/architecture_novelty_2026_09/` arm finishes):

```bash
python scripts/experiments/collect_arch_results.py
```

**Also not on this roadmap, because they're done** (this is a *pending*-work list; see
`docs/RNA_METHYLATION.md`'s "Forward direction" section for the numbers): `enc_bottleneck_mlp`
(evaluated 2026-09-06, official MAS-PCC 0.6631/0.6127/0.5810, -7.6%/-4.0%/-2.2% vs. row B) and
`enc_frozen_embedding_bulkrnabert` (evaluated 2026-09-06 after fixing the eval-time `--rna-cache`
bug noted below in `run_arch_suite.py::_eval_command`, official MAS-PCC 0.5795/0.5321/0.5164,
-19.3%/-16.7%/-13.0% vs. row B, a larger gap than `enc_bottleneck_mlp`). Neither is promoted;
both stay in the harness as measured non-wins.

**Not on this roadmap** (blocked on integration work, not a matter of launching a training run —
see `docs/PAPER_EXPERIMENTS.md`'s settings 3 and "Foundation-model" section): ENCODE atlas
expansion, CpGPT/MethylGPT/DeepCpG baselines. Also not on this roadmap: the Hyper-Connections/mHC
trunk arms (`trunk_hc_n2_d4`, `trunk_mhc_n2_d4`, `trunk_mhc_stream_semantics_d4`,
`trunk_mhc_gated_residual_d4`, `locus_attention_mhc_additive_residual_d4`,
`combined_locus_attention_mhc_semantic`) — measured, found not worth their complexity, and
**removed from the codebase entirely** (code, configs, on-disk runs/checkpoints) — see
`results/reference/ablations/architecture_novelty_2026_09/README.md`'s "What was tried and
removed" section.

---

## P0/P1: highest priority, run these first

### `seed_variance_reference`

**What**: 3-seed, full 80-epoch run of the reference recipe on chr1 — doubles as (a) the
architecture-novelty suite's noise floor (every other arm is judged against `2 x` this seed SD)
and (b) the still-missing full-budget convergence number for the reference architecture itself
(the headline 0.5627 MAS-PCC in `results/reference/rna_methylation/chr1.yaml` is a lower bound,
stopped at epoch 47/80). **The single most important run to finish** — every other
architecture-novelty arm's promotion verdict depends on it.

**Status**: queue shows `training` but no process is running on this host (stalled). Has a
resumable `checkpoints/last.pt`.

**Run-id actually in use**: `arch-architecture_novelty_2026_09-shared-seed_variance_reference-seed17`

**Launch** (auto-resumes from the checkpoint):

```bash
python scripts/experiments/run_arch_suite.py --arms seed_variance_reference --gpu 0
```

**Results land at**: `results/reference/ablations/architecture_novelty_2026_09/README.md`'s
"Current status" section has the full run-directory path; after `collect-arch-results` also at
`results/reference/ablations/architecture_novelty_2026_09/summary.{yaml,md}`.

### `baseline-global-shift`

**What**: "Global RNA Shift" paper-required baseline — a single per-patient correction, no
CpG-specific modeling at all (`use_mean_branch=false, include_raw_cpg=false,
use_raw_product=false`). Tests whether global per-patient differences alone explain any
performance. See `docs/PAPER_EXPERIMENTS.md`'s "Baseline models" section.

**Status**: not started. The old engine's frozen number (0.2343 MAS-PCC) is historical, not
reproducible by current code — this is a fresh run under the current architecture.

```bash
python scripts/train.py --model rna_methylation --scope chr1 \
  --engine matched_chr1_shared_backbone --mode final \
  --recipe configs/models/baselines/baseline_global_rna_shift.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --feature-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/features" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr1" \
  --output-root "$EXPERIMENT_ROOT" --run-id baseline-global-shift

python scripts/evaluate.py --model rna_methylation --engine matched_chr1_shared_backbone \
  --checkpoint "$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/baseline-global-shift/checkpoints/best.pt" \
  --eval-scope chr1 --recipe configs/models/baselines/baseline_global_rna_shift.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --feature-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/features" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr1" \
  --output "$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/baseline-global-shift/evaluation/chr1/metrics.json"
```

**Results land at**: `$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/baseline-global-shift/`. Then
record into `results/reference/baselines/global_rna_shift/chr1.yaml` (keep the old engine's
number alongside as `legacy_two_stage`, don't overwrite — PROJECT_SPECS.md §9.3).

### `baseline-bilinear`

**What**: "Bilinear RNA-CpG" baseline — raw branch sees only the RNA×CpG product term
(`use_mean_branch=false, include_raw_rna=false, include_raw_cpg=false`). Old engine's number:
0.5273 MAS-PCC (historical, not reproducible as-is).

```bash
python scripts/train.py --model rna_methylation --scope chr1 \
  --engine matched_chr1_shared_backbone --mode final \
  --recipe configs/models/baselines/baseline_bilinear_rna_cpg.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --feature-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/features" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr1" \
  --output-root "$EXPERIMENT_ROOT" --run-id baseline-bilinear

python scripts/evaluate.py --model rna_methylation --engine matched_chr1_shared_backbone \
  --checkpoint "$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/baseline-bilinear/checkpoints/best.pt" \
  --eval-scope chr1 --recipe configs/models/baselines/baseline_bilinear_rna_cpg.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --feature-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/features" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr1" \
  --output "$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/baseline-bilinear/evaluation/chr1/metrics.json"
```

**Results land at**: `$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/baseline-bilinear/`, then
`results/reference/baselines/bilinear_rna_cpg/chr1.yaml`.

### `baseline-mlp`

**What**: "MLP RNA-CpG" baseline — raw branch sees concatenated RNA+CpG, no product term
(`use_mean_branch=false, use_raw_product=false`). Old engine's number: 0.5198 MAS-PCC
(historical).

```bash
python scripts/train.py --model rna_methylation --scope chr1 \
  --engine matched_chr1_shared_backbone --mode final \
  --recipe configs/models/baselines/baseline_mlp_rna_cpg.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --feature-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/features" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr1" \
  --output-root "$EXPERIMENT_ROOT" --run-id baseline-mlp

python scripts/evaluate.py --model rna_methylation --engine matched_chr1_shared_backbone \
  --checkpoint "$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/baseline-mlp/checkpoints/best.pt" \
  --eval-scope chr1 --recipe configs/models/baselines/baseline_mlp_rna_cpg.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --feature-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/features" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr1" \
  --output "$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/baseline-mlp/evaluation/chr1/metrics.json"
```

**Results land at**: `$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/baseline-mlp/`, then
`results/reference/baselines/mlp_rna_cpg/chr1.yaml`.

### `p0-row-c`

**What**: trunk-depth control (`configs/models/arch_shared_backbone/p0_row_c_locus_attention_plain_trunk_d4.yaml`
— locus-attention encoder + plain depth-4 trunk). Tests whether adding trunk depth helps the
reference (locus-attention) encoder at all — an open question in `docs/RNA_METHYLATION.md`.

**Status**: interrupted at epoch 21/80, never evaluated. **Not part of the `run_arch_suite.py`
queue** (launched as a one-off; no `Arm` entry in `arch_suite.py`).

**Run-id actually in use**: `p0_row_c-seed17`

```bash
python scripts/train.py --model rna_methylation --scope chr1 \
  --engine matched_chr1_shared_backbone --mode final --resume --run-id p0_row_c-seed17 \
  --recipe configs/models/arch_shared_backbone/p0_row_c_locus_attention_plain_trunk_d4.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --feature-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/features" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr1" \
  --output-root "$EXPERIMENT_ROOT"

python scripts/evaluate.py --model rna_methylation --engine matched_chr1_shared_backbone \
  --checkpoint "$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/p0_row_c-seed17/checkpoints/best.pt" \
  --eval-scope chr1 \
  --recipe configs/models/arch_shared_backbone/p0_row_c_locus_attention_plain_trunk_d4.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --feature-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/features" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr1" \
  --output "$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/p0_row_c-seed17/evaluation/chr1/metrics.json"
```

**Results land at**: `$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/p0_row_c-seed17/`.

---

## P2: already-started architecture-novelty arm, resume when a GPU is free

### `trunk_plain_d4`
Depth-4 control (plain residual trunk) — isolates whether depth alone helps beyond the
reference's single fusion `Linear`.
```bash
python scripts/experiments/run_arch_suite.py --arms trunk_plain_d4 --gpu 0
```
Results land under `results/reference/ablations/architecture_novelty_2026_09/`'s status
table / run directories; see that README for the full arm taxonomy.

---

## P3: chr123 matched reference

Two sequential steps (the second depends on the first) — cannot be parallelized against each
other, but the cache-build step is a one-time prerequisite that unblocks re-running the training
step as many times as needed afterward (retries, seed sweeps, etc.).

### `chr123-cache`

**What**: build the protocol-ordered compact chr123 methylation cache (dense HDF5 slices instead
of sparse fancy-selection reads against the genome-wide bundle) — see
`docs/CHR123_TRAINING_OPTIMIZATIONS.md`.

```bash
python scripts/prepare_chr123_compact.py \
  --canonical-root "$CANONICAL_ROOT" \
  --output "$DERIVED_ROOT/methylprophet_compact_chr123"
```

**Produces**: `$DERIVED_ROOT/methylprophet_compact_chr123/` (~5.1 GiB; restartable, validates
completed source caches before reuse if rerun).

### `chr123-ref`

**What**: the reference architecture's first shared-backbone training run on chr123 (the existing
`results/reference/rna_methylation/chr123.yaml` predates this architecture entirely — see
`docs/PAPER_EXPERIMENTS.md`). Use the `contiguous_blocks`/bounded-prefetch settings from
`docs/CHR123_TRAINING_OPTIMIZATIONS.md` (the reference recipe currently ships the more
conservative chr1-era `prefetch_depth: 2`; bump to `8`/`prefetch_workers: 4` in a copy of the
recipe if validating that first, per PROJECT_SPECS.md §7.1's caveat).

**Depends on**: `chr123-cache`.

```bash
python scripts/train.py --model rna_methylation --scope chr123 --engine shared_backbone \
  --mode final --recipe configs/models/rna_methylation_locus_attention.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_compact_chr123" \
  --feature-cache "$DERIVED_ROOT/rna_feature_cache/chr123" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr123" \
  --output-root "$EXPERIMENT_ROOT" --run-id chr123-ref

python scripts/evaluate.py --model rna_methylation --engine shared_backbone \
  --checkpoint "$EXPERIMENT_ROOT/runs/locus_cls_joint/chr123/chr123-ref/checkpoints/best.pt" \
  --eval-scope chr123 --recipe configs/models/rna_methylation_locus_attention.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_compact_chr123" \
  --feature-cache "$DERIVED_ROOT/rna_feature_cache/chr123" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr123" \
  --output "$EXPERIMENT_ROOT/runs/locus_cls_joint/chr123/chr123-ref/evaluation/chr123/metrics.json"
```

**Results land at**: `$EXPERIMENT_ROOT/runs/locus_cls_joint/chr123/chr123-ref/`, then
`results/reference/rna_methylation/chr123.yaml` (keep the pre-shared-backbone number alongside as
historical, don't overwrite).

---

## P4: not-yet-started architecture-novelty arms

Lower priority — run once P0-P2 above are done, or opportunistically on any machine sitting idle.
All independent of each other; launch each the same way:

```bash
python scripts/experiments/run_arch_suite.py --arms <name> --gpu 0
# or several at once, comma-separated:
python scripts/experiments/run_arch_suite.py --arms enc_mlp,enc_locus_attention_k64,axial_cpg_d4 --gpu 0
```

| ID | What it tests |
|---|---|
| `enc_mlp` | Does the RNA branch just need a nonlinearity + hidden layer (capacity control)? |
| `enc_locus_attention_k64` | Locus-conditioned RNA attention, 64 gene-program tokens |
| `enc_locus_attention_k128` | Same, 128 tokens (capacity variant) |
| `trunk_plain_d2` | Depth-2 control |
| `trunk_plain_d8` | Depth-8 control (run after `trunk_plain_d4` — where does depth saturate) |
| `axial_cpg_d4` | Windowed attention along the CpG axis (genomic-neighbour co-methylation signal) |
| `beta_likelihood_head` | Bounded heteroscedastic Beta likelihood instead of plain MSE |

Each budgeted at roughly 3.5-4.5h training + a shorter evaluation pass on one RTX PRO 5000-class
GPU (measured for this suite, see the ablation README's "Running it" section).

Full arm taxonomy, promotion rule, and rationale: `results/reference/ablations/architecture_novelty_2026_09/README.md`.
