# Paper roadmap — chr1 → chr123, no ENCODE, no new code

Single, priority-ordered launch list for everything still needed to write the paper, under the
explicitly reduced scope agreed 2026-09-06: **exclude ENCODE**, **exclude anything that needs new
code**, get **chr1 fully closed first**, then verify chromosomal generalization on **chr123**.
Source ablation with the current architecture is kept but demoted to lowest priority and its
feasibility under "no new code" is still unconfirmed (see §6).

This document's job specifically: for every pending item, pin down **exactly where its output
must land and in what shape**, so runs launched on different machines merge into
`results/reference/` without collision or manual reconciliation. It supersedes nothing —
`docs/PAPER_EXPERIMENTS.md` remains the authority on *what the three settings are*,
`docs/EXPERIMENT_ROADMAP.md` remains the authority on the *architecture-novelty suite's* own
internal arm taxonomy/launch commands (P0-row / trunk / encoder-capacity arms) — this file is the
one flat, dependency-ordered sequence across *all* of it, in the order the paper narrative needs
it, with schemas made explicit.

**Rule inherited from `CLAUDE.md`/`docs/PAPER_EXPERIMENTS.md`**: never overwrite a `legacy_*` /
historical number when landing a new one; write the new engine's number alongside it, not over
it. Every collector below already follows this — flagged explicitly only where it doesn't yet
exist and has to be created.

## How to read this

- **ID** = literal `--run-id` (or, for doc-only items, an anchor into the paper's reference
  tables). Grep any log/run-dir/W&B project for it.
- **Machine-independent** rows have no cross-row ordering constraint — hand each to a different
  GPU host and run concurrently. Only the "depends on" column creates ordering.
- **Output (raw)** = where the run/eval writes under `$EXPERIMENT_ROOT` (per machine, gitignored).
- **Output (versioned)** = the collected, version-controlled artifact under `results/reference/`
  — same target path regardless of which machine produced the raw run, so `git pull` from any
  machine after collection is the merge step (last collector to run wins on overlapping fields;
  keep collectors idempotent and re-run them after every batch of finished runs, never hand-edit
  the generated files).
- Every `$CANONICAL_ROOT`/`$DERIVED_ROOT`/`$EXPERIMENT_ROOT` placeholder resolves as in
  `CLAUDE.md`'s "Commands" section; on the current host `$DERIVED_ROOT`/`$EXPERIMENT_ROOT` live
  under whatever `METHYL_DATA_ROOT` (or each script's own `--data-root`) resolves to on that
  specific machine — the canonical bundle root itself (`configs/data/tcga_canonical.yaml`) is
  `/raid/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1`, not necessarily mounted
  at that path on every host (confirm per-machine before launching).

## Result hygiene: what to keep, what to prune, on every machine

`results/experiments/` is gitignored and **per-machine** — nothing here syncs on its own, so left
unmanaged it just grows until disk fills. `results/reference/` is the only durable, shared copy of
record (git-tracked); once a run's numbers are in there (with checkpoint sha256 recorded in
`results/reference/PROVENANCE.md`), the raw run directory's only remaining jobs are (a) resuming
training further, (b) re-evaluating under a protocol change, or (c) audit. Prune anything not
serving one of those three, on a schedule, not just once:

- **A run from a retired/removed engine, once its numbers are frozen in `results/reference/` with
  a checkpoint sha256 recorded** → delete the entire raw run directory (checkpoints + wandb +
  logs). It cannot be resumed (the engine no longer exists in this codebase) and cannot be
  re-evaluated (same reason); the sha256 in `PROVENANCE.md` is the only provenance that still
  matters, and it survives the deletion. Precedent: `results/experiments/legacy/methylprophet/`
  (the retired two-stage `MethylProphetTrainer` runs) was removed under exactly this rule
  (2026-09-06, 1.3 GB reclaimed) — do the same on every other machine holding a copy of that
  directory; there is nothing left to extract from it.
- **`checkpoints/best.pt` vs `checkpoints/last.pt` — do NOT assume these are duplicates even when
  their file sizes match.** Checked directly (`md5sum`) on four finished runs on this host
  (`p0_row_b-seed17`, `baseline-mlp`, `chr1-shared-primary-aggressive-optimized-seed17`,
  `arch-...-enc_bottleneck_mlp-seed17`): all four pairs have equal size but **different content**
  in every case — `last.pt` is a genuinely later/different optimizer+model state, not a copy of
  `best.pt`. Never delete one on a same-size assumption. Only prune `last.pt` for a run that is
  (a) fully finished, (b) already collected into `results/reference/`, and (c) not on any pending
  roadmap row as a resume target — and even then, treat it as a judgment call to flag to the team
  before deleting, not an automatic sweep, since it also discards the true final-epoch state.
  `best.pt` is what every `scripts/evaluate.py` command in this doc reads — never delete it.
- **Keep unconditionally, they're tiny (KB, not GB) and are the actual audit trail**:
  `config.resolved.yaml`, `metadata.json`, `training/history.json` (or `.csv`),
  `evaluation/*/metrics.json`. These are what `results/reference/PROVENANCE.md` rows point at —
  losing them breaks traceability even though the multi-hundred-MB checkpoint is what actually
  fills the disk.
- **A resumable/stalled run a roadmap row depends on** (e.g. §1's `seed_variance_reference`,
  `docs/EXPERIMENT_ROADMAP.md`'s `p0-row-c`/`trunk_plain_d4`) → never prune `checkpoints/last.pt`
  on these; that file is the only thing `--resume` has to pick up from.
- **Low-priority, do only when a host is actually full**: each run's `wandb/` subdirectory
  duplicates itself on disk (`wandb/run-<timestamp>-<id>/` and `wandb/latest-run/` are two
  independent copies here, not a symlink — confirmed by directory listing, not just inferred) and
  again logs the same `debug*.log` at the `wandb/` root. Local W&B mirrors are non-canonical (the
  W&B backend already has the authoritative copy) — safe to delete a run's whole local `wandb/`
  directory once you've confirmed that run's metrics are visible on the W&B dashboard, but this is
  a few MB per run, not worth the review effort until disk pressure is real.

**After collecting any batch of finished runs, run the relevant `collect_*.py` script first, then
apply the pruning rules above** — never prune before collecting; a deleted checkpoint that turns
out to have an uncollected number is unrecoverable without a full retrain.

## Priority overview

| Tier | Section | What | Independent of |
|---|---|---|---|
| P0 | [§1](#1-p0-final-chr1-reference-model-converged) | Final chr1 reference model, converged (3 seeds) | everything below |
| P1 | [§2](#2-p1-rna-encoder-comparison-table-official) | RNA-encoder comparison, official chr1 | §3, §4 |
| P1 | [§3](#3-p1-mean-contribution-causal-ablation) | Mean-contribution causal ablation | §2, §4 |
| P2 | [§4](#4-p2-simplified-baselines-official-chr1) | 3 remaining simplified baselines | §2, §3 |
| P3 | [§5](#5-p3-chr1--chr123-generalization) | chr1↔chr123 generalization matrix | §1 (needs its checkpoint) |
| P4 | [§6](#6-p4-nice-to-have-source-ablation-current-architecture) | Source ablation, current architecture | needs a feasibility check first |
| §0 | [§0](#0-p0-methylprophet-published-numbers-to-transcribe-no-compute) | MethylProphet published-number transcription | nothing — do any time, zero compute |

Everything inside a tier is mutually independent and safe to spread across machines; §5's four
cells depend on checkpoints from §1 (chr1 side) and its own `chr123-ref` run (chr123 side), so
launch §5's training halves as soon as those checkpoints exist rather than waiting for the whole
tier to finish.

---

## 0. P0: MethylProphet published-number transcription (no compute)

Not training — read the numbers directly off the MethylProphet paper's own tables and commit them
as literal reference data. Zero GPU cost, do this any time, in parallel with everything else.

| Table | Status | Target file |
|---|---|---|
| Table 5 (main comparison) | **done** | `results/reference/methylprophet_comparison/table5_chr1.md` |
| Table 7 (source ablation) | **done** | `results/reference/methylprophet_comparison/table7_source_ablation.md` |
| Table 8 (chromosome generalization) | recorded from memory/notes, **not yet checked against the actual paper table** | new: `results/reference/methylprophet_comparison/table8_chromosome_generalization.md` |
| Table 9 (gene-pathway RNA ablation) | recorded, only as `reported_context` inside `results/reference/ablations/rna_encoder_comparison_2026_09/summary_development.yaml` — not yet in its own paper-facing file | new: `results/reference/methylprophet_comparison/table9_gene_pathway_ablation.md` |
| Predecessor row (Levy/Jurgenson) | **not recorded anywhere in this repo** — must be read directly off MethylProphet's own main table, never invented | new row inside `table5_chr1.md` |

Action: re-open the actual MethylProphet paper (not this repo's notes) for Table 8/9 and the
predecessor citation, confirm every digit against the table image/text directly, then write:

```markdown
<!-- table8_chromosome_generalization.md -->
# MethylProphet Table 8 — chromosome generalization (published numbers only)

Published-only; "Ours" columns are filled by §5 below once trained. Never edit the
MethylProphet columns by hand except to correct a transcription error, and note the correction.

| Train | Eval | View | MethylProphet MAS-PCC | Ours MAS-PCC |
|---|---|---|---:|---:|
| chr1 | chr1 | train_cpg_x_val_sample | | |
| chr1 | chr1 | val_cpg_x_train_sample | | |
| chr1 | chr1 | val_cpg_x_val_sample | 0.3904 | |
| chr123 | chr1 | val_cpg_x_val_sample | 0.3505 | |
| chr1 | chr123 | val_cpg_x_val_sample | 0.2513 | |
| chr123 | chr123 | val_cpg_x_val_sample | 0.3460 | |
```

Same pattern for `table9_gene_pathway_ablation.md` (MethylProphet's own gene-pathway vs. default
RNA representation — already have the three headline numbers, just needs its own citable file
instead of living only inside an ablation summary) and the predecessor row inside `table5_chr1.md`
(add a `| Levy/Jurgenson | ... |` row above the MethylProphet row, cited from the paper's own
Table 5, not this repo).

---

## 1. P0: Final chr1 reference model, converged

**The priority-zero item.** Everything downstream (encoder table's cross-attention row, chr123
generalization, the paper's headline number) is blocked on this.

This **is** `seed_variance_reference` from `docs/EXPERIMENT_ROADMAP.md` — 3 seeds, full 80-epoch
budget, unmodified `configs/models/rna_methylation_locus_attention.yaml` (which already defaults
to `query_source=ntv3`, `n_programs=64`, `program_dim=256`, `n_heads=4`, mean branch ON,
`use_raw_product=false` — i.e. exactly the config the paper is freezing on). Running it also
retires the `query_representation_2026_09` study's own "one final-protocol confirmation" step
(its winner `q0_ntv3` **is** this recipe's default) — no separate query-representation run needed.

- **Status** (per `docs/EXPERIMENT_ROADMAP.md`, last checked 2026-09-06): queue shows `training`
  but stalled on its host, resumable from `checkpoints/last.pt`.
- **Run-id**: `arch-architecture_novelty_2026_09-shared-seed_variance_reference-seed{17,29,43}`
- **Machine-independent across seeds** — hand each seed to a different GPU host.

```bash
python scripts/experiments/run_arch_suite.py --arms seed_variance_reference --gpu 0
```

**Output (raw)**: `$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/arch-architecture_novelty_2026_09-shared-seed_variance_reference-seed{N}/`.

**Collect** (idempotent, run after each seed finishes, from any machine that can see the run dirs):

```bash
python scripts/experiments/collect_arch_results.py
```

**Output (versioned)**:
`results/reference/ablations/architecture_novelty_2026_09/{summary.yaml,summary.md}` gets the
noise floor (mean/stdev across the 3 seeds) filled in — this unblocks every arm currently marked
`unjudged` in that file (`enc_bottleneck_mlp`, `enc_frozen_embedding_bulkrnabert`; re-run the
collector once more after this to flip their verdicts).

**Gap to close once this converges — `results/reference/rna_methylation/chr1.yaml` does not
currently exist on disk**, despite being referenced throughout `CLAUDE.md`/`docs/*`. This is the
one file every paper table ultimately cites for "Ours" on chr1; recreate it by hand from the
converged run's `evaluation/chr1/metrics.json`, schema matching the existing baseline files
(`results/reference/baselines/*/chr1.yaml`):

```yaml
# results/reference/rna_methylation/chr1.yaml
status: reference
model: rna_methylation
architecture: FeatureFusionArchitectureVariantModel, encoder.kind=locus_attention (n_programs=64,
  program_dim=256, n_heads=4, query_source=ntv3), use_mean_branch=true, use_raw_product=false
training_scope: chr1
seed: 17          # report all 3; mean ± stdev is the paper number, per-seed rows below it
seeds: [17, 29, 43]
training:
  recipe: configs/models/rna_methylation_locus_attention.yaml
  engine: matched_chr1_shared_backbone
  epochs: 80
evaluation:
  chr1:
    train_cpg_x_val_sample: {mas_pcc: null, mac_pcc: null, mse: null, mae: null}
    val_cpg_x_train_sample: {mas_pcc: null, mac_pcc: null, mse: null, mae: null}
    val_cpg_x_val_sample:   {mas_pcc: null, mac_pcc: null, mse: null, mae: null}
  # per-seed breakdown for the noise-floor/error-bar
  by_seed:
    17: {val_cpg_x_val_sample_mas_pcc: null}
    29: {val_cpg_x_val_sample_mas_pcc: null}
    43: {val_cpg_x_val_sample_mas_pcc: null}
benchmark_role: headline "Ours" row, MethylProphet-matched chr1 comparison (Table 5)
legacy_two_stage: {val_cpg_x_val_sample_mas_pcc: 0.5613}   # RNAMethylationPredictor, removed generation — keep, do not overwrite
note: >
  Reference architecture (docs/RNA_METHYLATION.md, CLAUDE.md's "What this repo is"); supersedes
  legacy_two_stage. Fill in from the seed_variance_reference collector once all 3 seeds converge.
```

No script currently writes this file (`collect_arch_results.py` writes to the ablation-study
directory, not here) — this is a manual, one-time transcription step once the 3 seeds are in, not
a new pipeline to build.

---

## 2. P1: RNA-encoder comparison table, official

Cross-attention row = §1's result (same checkpoint, no extra run). Two arms are **already done**
officially; two need a fresh `--mode final` run.

| Encoder | Status | Run-id / recipe |
|---|---|---|
| Cross-attention (reference) | = §1 | — |
| `enc_bottleneck_mlp` | **done** (official MAS-PCC 0.6631/0.6127/0.5810) | `configs/models/arch_shared_backbone/enc_bottleneck_mlp.yaml` |
| `enc_frozen_embedding_bulkrnabert` | **done** (official MAS-PCC 0.5795/0.5321/0.5164) | `configs/models/arch_shared_backbone/enc_frozen_embedding_bulkrnabert.yaml` |
| `gene_pathway` | dev-only (inner double-OOD 0.5072) | `configs/models/rna_encoder_comparison/enc_gene_pathway.yaml` |
| `bulkformer_147m` | dev-only (inner double-OOD 0.4516) | `configs/models/rna_encoder_comparison/enc_frozen_bulkformer_147m.yaml` |

**Machine-independent**: `gene_pathway` and `bulkformer_147m` can run on two different hosts.
`bulkformer_147m` additionally needs its embedding cache present first (checked automatically by
the runner, which prints the exact prep command if missing).

```bash
python scripts/experiments/run_rna_encoder_comparison.py --arms gene_pathway --mode final --gpu 0
python scripts/experiments/run_rna_encoder_comparison.py --arms bulkformer_147m --mode final --gpu 0
```

**Output (raw)**: `$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/rnaenc-{arm}-final-seed17/`.

**Collect** (idempotent):

```bash
python scripts/experiments/collect_rna_encoder_comparison.py --mode final
```

**Output (versioned)**: `results/reference/ablations/rna_encoder_comparison_2026_09/summary_final.{yaml,md}`
(same schema as the existing `summary_development.{yaml,md}`, `mode: final` in the protocol block).

**Gap**: the full 5-row table (cross-attention + 4 comparators) is currently split across two
different collectors/files (`collect_arch_results.py` → `architecture_novelty_2026_09/summary.md`
for the first two rows, `collect_rna_encoder_comparison.py` → `rna_encoder_comparison_2026_09/
summary_final.md` for the last two). Once all four comparator numbers exist, hand-assemble the
single paper table into a new file — this is bookkeeping, not new pipeline code:

```markdown
<!-- results/reference/ablations/rna_encoder_comparison_2026_09/summary_official.md -->
# RNA encoder comparison — official chr1 (paper table)

| Encoder | train_cpg×val_sample | val_cpg×train_sample | val_cpg×val_sample |
|---|---:|---:|---:|
| **Cross-attention (reference)** | | | |
| MethylProphet Bottleneck MLP | 0.6631 | 0.6127 | 0.5810 |
| BulkRNABert frozen | 0.5795 | 0.5321 | 0.5164 |
| Gene-pathway | | | |
| BulkFormer-147M frozen | | | |
```

---

## 3. P1: Mean-contribution causal ablation

**DONE (2026-09-07).** All three arms × 3 seeds (17/29/43) on chr1 completed and collected into
`results/reference/appendix/mean_contribution_2026_09/`; a single-seed (17) chr123 follow-up
(same arms, generic `shared_backbone` engine) completed and collected into
`results/reference/appendix/mean_contribution_2026_09_chr123/`, confirming the effect isn't
chr1-specific. Manuscript-facing verdict recorded in `results/reference/ours/03_mean_contribution.yaml`.
Commands below kept for reference/reproduction only.

Full protocol already written: `docs/MEAN_CONTRIBUTION_EXPERIMENTS.md`. Three arms × 3 seeds
(17/29/43); `full_reference` reuses the already-launched `ref-locus-attn-k64-noproduct-seed{N}`
runs (verify these exist on the training host before assuming the runner will skip training for
them — not visible from this machine).

**Machine-independent across arms** (not across seeds within `no_mean_supervision`/
`no_mean_branch` unless spread further):

```bash
python scripts/experiments/run_mean_contribution.py --arms no_mean_supervision --gpu 0
python scripts/experiments/run_mean_contribution.py --arms no_mean_branch --gpu 1
python scripts/experiments/run_mean_contribution.py --arms full_reference --gpu 2
```

**Output (raw)**: `$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/{arm}-seed{N}/`.

**Collect**:

```bash
python scripts/experiments/collect_mean_contribution.py \
  --dataset-diagnostics "$METHYL_DATA_ROOT/experiments/analysis/mean_contribution_2026_09/dataset_variance.json"
```

**Output (versioned)** — schema already fixed by the protocol doc, reproduced here for reference:

```text
results/reference/appendix/mean_contribution_2026_09/
  summary.yaml                    # aggregate metrics, paired effects, protocol metadata
  summary.md                      # manuscript-facing summary
  paper_table.csv                 # one row per arm × seed × official view
  variance_decile_effects.csv     # MSE/bias effect across CpG variance deciles
  dataset_variance.json           # missing-aware variance decomposition
  runs/{arm}__seed{N}.json         # full per-run provenance (recipe/config/checkpoint SHA-256,
                                   # epoch, seed, all official views, mean-head accuracy,
                                   # h_mean linear-probe, locus-bias, variance-decile metrics)
```

Primary contrast for the paper: `full_reference` vs `no_mean_supervision` (proxy-task
supervision) and `no_mean_supervision` vs `no_mean_branch` (branch capacity alone). Report MSE +
locus-level bias as primary; MAS-PCC is secondary (invariant to a per-CpG constant shift).

---

## 4. P2: Simplified baselines, official chr1

CpG Prior is already done (zero-parameter, no training —
`results/reference/baselines/cpg_prior/chr1.yaml`). The other three need a fresh training run
under the current engine; their existing numbers are from the retired two-stage engine
(kept as historical context, not overwritten).

**Fully machine-independent, 3 separate hosts:**

```bash
# host A
python scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
  --mode final --recipe configs/models/baselines/baseline_global_rna_shift.yaml \
  --canonical-root "$CANONICAL_ROOT" --prepared-root "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --feature-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/features" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr1" \
  --output-root "$EXPERIMENT_ROOT" --run-id baseline-global-shift

# host B — same shape, --recipe configs/models/baselines/baseline_bilinear_rna_cpg.yaml, --run-id baseline-bilinear
# host C — same shape, --recipe configs/models/baselines/baseline_mlp_rna_cpg.yaml, --run-id baseline-mlp
```

Exact train+eval command pairs for all three: `docs/EXPERIMENT_ROADMAP.md`'s P1 section (verbatim,
don't re-derive).

**Output (raw)**: `$EXPERIMENT_ROOT/runs/locus_cls_joint/chr1/baseline-{global-shift,bilinear,mlp}/`.

**Output (versioned)** — no collector script exists for these; write by hand into
`results/reference/baselines/{global_rna_shift,bilinear_rna_cpg,mlp_rna_cpg}/chr1.yaml`, schema
identical to the existing `cpg_prior`/`bilinear_rna_cpg` files in that directory (`status`,
`model`, `architecture`, `training_scope`, `seed`, `training:{...}`, `evaluation.chr1.{train_cpg_x_val_sample,val_cpg_x_train_sample,val_cpg_x_val_sample}:{mas_pcc,mse}`,
`benchmark_role`, `note`) — keep the old engine's number as `legacy_two_stage`, don't overwrite.

---

## 5. P3: chr1 ↔ chr123 generalization

Freeze the architecture from §1 first — this tier is a **test** of the chosen architecture, not
another chance to pick one (per the paper's own defensibility rule: choose everything on chr1,
freeze, then run chr123 untouched).

> **No chr123 checkpoint trained with the current cross-attention architecture exists anywhere
> yet — verified directly against both checkpoints on disk (2026-09-06), not assumed.** Two chr123
> runs currently exist and **neither** is it:
> - `rna-methylation-chr123-retrain-2026-08-31` (`results/experiments/runs/rna_methylation/chr123/`)
>   — the retired two-stage engine, predates shared-backbone entirely.
> - `chr123-shared-primary-aggressive-seed17` (`results/experiments/runs/locus_cls_joint/chr123/`,
>   trained 2026-09-04 on host `hal`) — **`model.encoder.kind: linear`** (confirmed from its
>   `config.resolved.yaml`), i.e. the *previous* reference (`FeatureFusionLocusCLSModel`, promoted
>   2026-09-03, superseded on chr1 2026-09-05 by `locus_attention`). This one is easy to mistake
>   for the current model precisely because it lives under the same `locus_cls_joint` run-store
>   family as every current-architecture run — the run-id alone doesn't tell you which encoder it
>   used, only `config.resolved.yaml`'s `model.encoder.kind` does. **Do not cite this run's numbers
>   as "Ours" for chr123** — if it's ever reported at all, label it explicitly as the
>   previous-reference (linear-encoder) architecture, not the paper's headline model.
>
> `chr123-ref` below is a genuinely new run, not a resume/rename of either of these.

Two sequential prerequisite steps, then four cells that are themselves fully independent of each
other (each is just a training-checkpoint × eval-scope pair, so all four can run on different
machines once their two checkpoints exist):

### 5a. Prerequisite: chr123 cache + chr123 training

```bash
# one-time cache build (~5.1 GiB, restartable)
python scripts/prepare_chr123_compact.py --canonical-root "$CANONICAL_ROOT" \
  --output "$DERIVED_ROOT/methylprophet_compact_chr123"

# chr123-ref training (depends on the cache above)
python scripts/train.py --model rna_methylation --scope chr123 --engine shared_backbone \
  --mode final --recipe configs/models/rna_methylation_locus_attention.yaml \
  --canonical-root "$CANONICAL_ROOT" --prepared-root "$DERIVED_ROOT/methylprophet_compact_chr123" \
  --feature-cache "$DERIVED_ROOT/rna_feature_cache/chr123" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr123" \
  --output-root "$EXPERIMENT_ROOT" --run-id chr123-ref
```

### 5b. Four cells (each independent once both checkpoints exist)

| Train | Eval | Checkpoint | `--eval-scope` |
|---|---|---|---|
| chr1 | chr1 | §1's converged checkpoint | `chr1` |
| chr123 | chr1 | `chr123-ref`'s checkpoint | `chr1` |
| chr1 | chr123 | §1's converged checkpoint | `chr123` |
| chr123 | chr123 | `chr123-ref`'s checkpoint | `chr123` |

```bash
python scripts/evaluate.py --model rna_methylation --engine shared_backbone \
  --checkpoint <checkpoint-from-table-above> --eval-scope <chr1|chr123> \
  --recipe configs/models/rna_methylation_locus_attention.yaml \
  --canonical-root "$CANONICAL_ROOT" --prepared-root <matching prepared-root for the eval scope> \
  --feature-cache <matching feature-cache> --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir <matching cpg-targets-dir> \
  --output "$EXPERIMENT_ROOT/runs/locus_cls_joint/<train-scope>/<run-id>/evaluation/<eval-scope>/metrics.json"
```

(Cross-scope eval — chr1 checkpoint against chr123 data or vice versa — needs whichever engine
matches the *checkpoint's* scope but pointed at the *other* scope's `--prepared-root`/
`--feature-cache`/`--cpg-targets-dir`; confirm `evaluate.py --engine shared_backbone` accepts a
chr1 checkpoint against chr123 inputs and a chr123 checkpoint against chr1 inputs before assuming
this works unmodified — flag as a real risk, not a certainty, since neither direction has been
tried under the current engine.)

**Output (raw)**: four `evaluation/<eval-scope>/metrics.json` files as above.

**Output (versioned)** — new file, mirrors Table 7's existing narrative-table pattern:

```markdown
<!-- results/reference/methylprophet_comparison/table8_chromosome_generalization.md -->
(same file §0 defines — fill the "Ours" column from these four cells once done)
```

Also update `results/reference/rna_methylation/chr123.yaml` (create if the chr123-ref cell is the
first chr123 number under the current architecture; keep the existing 2026-08-31 pre-shared-backbone
number as historical alongside it, same `legacy_two_stage`-style convention as §1).

---

## 6. P4 (nice-to-have): source ablation, current architecture

Lowest priority (user's own ordering) **and its mechanism is unconfirmed**: MethylProphet's own
Table 7 "Ours" column was produced by the now-retired two-stage engine's own source-restriction
path (`benchmark/methylprophet`'s old pipeline); grepping the current
`matched_chr1_shared_backbone`/`shared_backbone` engines and `scripts/train.py` found no
`--sources`/train-time source-subset flag — `locus_cls.structured_loss_sources` in the recipe only
reweights the loss per source, it does not appear to change which sources' *samples* enter
training. **Before launching anything here, confirm (by reading
`rna_training/locus_cls_trainer.py`'s pool-construction code, not by assuming) whether restricting
training to a source subset is possible with existing code and existing prepared-root caches.** If
it isn't, this item needs new code and is out of scope for the current "no new code" reduction —
leave it explicitly deferred rather than force it in.

If/once confirmed feasible, four independent runs (Array / Array+WGBS / Array+EPIC /
Array+EPIC+WGBS — the last **is** §1's own run, reused, not repeated), run-ids
`source-ablation-{array,array-wgbs,array-epic,array-epic-wgbs}`, output versioned into a new
`results/reference/methylprophet_comparison/table7_source_ablation_ours.md` (current-architecture
companion to the existing, frozen `table7_source_ablation.md`) with the same
Train-Data/MAS-PCC/MAC-PCC/MSE/MAE column shape.
