# architecture_novelty_2026_09

Internal architecture ablation. **Not** a paper baseline and **not** a
MethylProphet comparison — see `results/reference/PROVENANCE.md` and
`CLAUDE.md`'s results taxonomy for why those live elsewhere. If an arm here is
ever adopted as the canonical architecture it must then be retrained and tuned
independently in all three MethylProphet-matched settings before appearing in
any "vs MethylProphet" table (`docs/PAPER_EXPERIMENTS.md`).

## 2026-09-04: retargeted to the shared-backbone architecture

This suite originally sat on `VarianceNormalizedResidualModel`
(`RNAMethylationPredictor`, the two-stage frozen-prior + residual model,
0.5613 chr1 MAS-PCC). While it was being built, a separate agent completed
the `shared_backbone_locus_cls_2026_09` ablation ladder and
`FeatureFusionLocusCLSModel` was promoted to the repo's primary/reference
RNA-methylation architecture (`docs/RNA_METHYLATION.md`, CLAUDE.md's "Model
compatibility note") — at the time of that retargeting the older model was
kept for old-checkpoint compatibility, but see the "Update (later refactor)"
note below: a subsequent 2026-09-06 refactor removed that whole generation
(and its `configs/models/arch/` recipes) from the codebase entirely, dropping
old-checkpoint compatibility too. Treat this paragraph as describing the
2026-09-04 state only, not the current one.

Every arm below was rebuilt on top of the new reference architecture.
`scripts/experiments/{arch_suite,run_arch_suite,collect_arch_results}.py`
target the shared-backbone family only; the retired two-stage suite's driver
was stopped after 2 epochs of its noise-floor arm and never continued, and its
own code/configs no longer exist at all (see below), not merely unused.

## Why this study exists

A postdoc reviewer (2026-09-03) judged the architecture too simple for the
paper's novelty claim despite its performance, and suggested improving how the
two branch embeddings are combined — naming *mHC: Manifold-Constrained
Hyper-Connections* (DeepSeek, arXiv:2512.24880) — and the RNA encoding.

The critique holds up against the **new** reference architecture too:

- `FeatureFusionLocusCLSModel`'s RNA branch is still a single
  `Linear(25017 → 256)` (`models.py::LinearRNAEncoder`, unchanged by the
  ladder that selected this architecture).
- The two branch embeddings (`h_mean`, `h_raw`) are combined by **one
  `Linear`** — literally the exact operation the postdoc's suggestion targets,
  now on the architecture that actually has two clearly separated embeddings.
- No fusion-mechanism ablation has ever been run on this architecture (the
  retired `fusion_mechanism_2026_08` suite predates it).

**Later update**: the mHC trunk this suggestion motivated (originally "Stage
2" of this study) was built, run, measured, and **removed from the codebase
entirely** — see "What was tried and removed" near the end of this file. The
RNA-encoding half of the suggestion (Stage 1b below) is the part that
actually moved the number and remains the repo's primary/reference
architecture change.

## Protocol

Every arm: `matched_chr1_shared_backbone` engine, `mode=final`, 80 epochs,
`schedule_policy=pair_complete`, evaluated via
`rna_training/locus_cls_trainer.py::evaluate_official_split` on the official
MethylProphet Table-5 chr1 Array views (view `official_val_cpg_x_val_sample`).

**Two mechanical differences from the retired suite, both consequences of how
this engine works:**

1. **Training and evaluation are separate steps.** `scripts/train.py` only
   trains; the official-split number comes from a second
   `scripts/evaluate.py --engine matched_chr1_shared_backbone` call against
   the checkpoint. `run_arch_suite.py` runs both back to back per unit.
2. **Auto-resume from a checkpoint.** `run_arch_suite.py` passes `--resume`
   to `scripts/train.py` whenever a unit's run directory already exists with
   a valid `checkpoints/last.pt` — see "Current status" below for the exact
   command.

## The reference number is not converged yet

`rung_b_official_final` (`results/reference/ablations.yaml`) — the number that
selected this architecture — was **stopped by user request at epoch 47/80**,
still improving 0.3–0.8%/epoch. **0.5627 / MSE 0.01702 is a lower bound, not
the converged reference.**

`seed_variance_reference` (3 seeds of the unmodified reference recipe, run to
the full 80-epoch budget) is therefore doing double duty: it is both the seed
standard deviation every other arm is judged against, **and** the missing
convergence run for the project's own primary-architecture number. Run it
first; nothing else is interpretable, and the project itself is waiting on it
independently of this suite.

## Promotion rule

**An arm counts as an improvement only if it beats the (converged, once
`seed_variance_reference` completes) reference mean by more than 2 × the seed
standard deviation.**

## Arms

| stage | arm | what it asks |
|---|---|---|
| 0-noise-floor | `seed_variance_reference` | How large is a null delta, and what does rung_b converge to? |
| 1a-encoder-capacity | `enc_mlp` | Does the raw branch's RNA encoder just need a nonlinearity and a hidden layer? |
| 1b-locus-conditioned-rna | `enc_locus_attention_k64` / `_k128` | Does letting each CpG query the transcriptome beat one locus-invariant RNA vector? |
| 1c-trunk-depth | `trunk_plain_d2` / `_d4` / `_d8` | Does the single fusion Linear lose anything to depth, and where does it saturate? |
| 1d-axial-comethylation | `axial_cpg_d4` | Does attending to genomic neighbours along the CpG axis add signal? |
| 1e-output-likelihood | `beta_likelihood_head` | Does a bounded, heteroscedastic likelihood beat MSE on beta values? |

A Hyper-Connections/Manifold-Constrained-HC multi-stream trunk stage (originally "Stage
2", the postdoc's most literal suggestion for "how the two branch embeddings are
combined") and a "Stage 3" composing it with locus-conditioned attention were run,
measured, and removed — see "What was tried and removed" below.

## Current status (2026-09-06) and how to resume

`enc_bottleneck_mlp` and `enc_frozen_embedding_bulkrnabert` have been captured into
`runs/`/`summary.yaml` via `collect_arch_results.py` (2026-09-06) -- see `summary.md` for
their numbers. Both underperform row B (the `locus_attention` reference) and are not
promoted; see `docs/RNA_METHYLATION.md`'s "Forward direction" section for the read on each.
Note the verdict column in `summary.md` currently reads "unjudged" for both: the noise
floor (`seed_variance_reference`) hasn't converged yet, so their deltas are only measured
against the epoch-47/80 lower-bound reference, not the 2×SD promotion rule -- re-run
`collect_arch_results.py` once that arm finishes to get a judged verdict.
Status per remaining arm, from `logs/arch_suite/queue_state_<host>.json` and each run
directory:

| arm | status | what's needed |
|---|---|---|
| `seed_variance_reference` | queue says "training" but stalled (no process running) -- this is also the **missing chr1 convergence run** (rung_b_official_final was stopped at epoch 47/80) | rerun to auto-resume from `checkpoints/last.pt` |
| `trunk_plain_d4` | stalled, same as above | rerun to auto-resume |
| `p0_row_c_locus_attention_plain_trunk_d4` (does trunk depth help the reference encoder at all; run-id `p0_row_c-seed17`) | interrupted at epoch 21/80, never evaluated -- **not part of the `run_arch_suite.py` queue** (no `Arm` entry in `arch_suite.py`; it was launched as a one-off, see `logs/p0_tier1/driver.log`) | resume manually (see command below), then evaluate manually |
| `enc_mlp`, `enc_locus_attention_k64`/`_k128`, `trunk_plain_d2`/`_d8`, `axial_cpg_d4`, `beta_likelihood_head` | not started | run via the queue once the above are done |

The `trunk_hc_n2_d4`/`trunk_mhc_n2_d4`/`trunk_mhc_stream_semantics_d4`/`trunk_mhc_gated_residual_d4`/
`locus_attention_mhc_additive_residual_d4`/`combined_locus_attention_mhc_semantic` arms that
used to appear in this table have been **removed entirely** (code, configs, on-disk runs and
checkpoints) — see "What was tried and removed" below.

**Exact launch/resume commands for every row above (one per arm, meant to be handed to different
machines in parallel) are in [`../../../../docs/EXPERIMENT_ROADMAP.md`](../../../../docs/EXPERIMENT_ROADMAP.md)** --
that's the canonical copy; don't re-derive them here. Once arms finish, capture everything into
provenance (idempotent, safe to rerun any time):

```bash
python scripts/experiments/collect_arch_results.py
```

## Running it

```bash
# CPU preflight: build every arm, check the fusion_init_std=0 -> exactly-0.5
# invariant, print sizes
python scripts/experiments/run_arch_suite.py --dry-run

# GPU preflight: one small forward+backward per arm under real bf16 autocast,
# including the auxiliary mean/residual losses the real trainer applies
python scripts/experiments/run_arch_suite.py --cuda-smoke --gpu 0

# the noise floor / convergence run first — nothing else is interpretable
python scripts/experiments/run_arch_suite.py --stages 0-noise-floor --gpu 0

# then spread the rest across machines that share the output root
python scripts/experiments/run_arch_suite.py --shard 1/3 --gpu 0
python scripts/experiments/run_arch_suite.py --shard 2/3 --gpu 0 --data-root /mnt/m
python scripts/experiments/run_arch_suite.py --shard 3/3 --gpu 1

# merge everything into this directory (idempotent, re-runnable)
python scripts/experiments/collect_arch_results.py \
    --output-root results/experiments \
    --output-root /mnt/m/results/experiments
```

Run ids are deterministic per (arm, seed). A killed unit **is** auto-resumed
the next time its arm is selected (`run_arch_suite.py` passes `--resume` to
`scripts/train.py` whenever the run directory already exists with a valid
`checkpoints/last.pt`) — only a run directory that exists but has no
`last.pt` at all (e.g. killed before the first checkpoint) is reported as
`blocked_incomplete_run_dir` and skipped.

Budget: 13 units (11 arms, `seed_variance_reference` at 3 seeds), ~3.5–4.5 h
training + a shorter evaluation pass each, on one RTX PRO 5000. Measured peak
GPU memory at the real WGBS block size (32×20480, this architecture's
batching) is 15–19 GB (`--cuda-smoke --smoke-samples 32 --smoke-loci 20480`),
comfortably under the `--min-free-gb` default of 20.

## Files here

- `runs/<arm>__seed<N>.json` — one self-describing record per completed run:
  official-split metrics, checkpoint path/epoch, training summary. Generated;
  do not hand-edit.
- `summary.yaml` / `summary.md` — generated rollup with the noise floor and
  per-arm verdict.

When the study concludes, fold the verdict into `results/reference/ablations.yaml`
as a normal study block and add the run rows to `results/reference/PROVENANCE.md`.

## What was tried and removed: Hyper-Connections / mHC

Stage 2 (`trunk_hc_n2_d4`, `trunk_mhc_n2_d4`, `trunk_mhc_stream_semantics_d4`,
`trunk_mhc_gated_residual_d4`, `locus_attention_mhc_additive_residual_d4`) and Stage 3
(`combined_locus_attention_mhc_semantic`) tested a multi-stream Hyper-Connections /
Manifold-Constrained-HC trunk (`models.py::HyperConnectionTrunk`, `TrunkConfig.kind`
`hc`/`mhc`) as the literal answer to the postdoc's "improve how the two branch embeddings
are combined" suggestion. Measured result (chr1, single seed, see
`docs/RNA_METHYLATION.md`'s former "2026-09-05 update" section for the full table before
this removal, recoverable from git history): the mHC flagship beat the reference by only
+0.0057 / +0.0087 / +0.0050 MAS-PCC across the three official views — an order of
magnitude below the RNA-encoder effect (Stage 1b) and within the range a single seed could
produce by chance. A checkpoint-level diagnostic reinforced this independently of any
retraining: the flagship's learned residual mapping's off-diagonal mass (0.0152-0.0156)
was *smaller* than at random initialization (0.0180) — training pushed the cross-stream
exchange the mechanism exists to enable *closer* to the identity, not away from it.

Given that evidence, mHC/HC were judged not worth their added complexity and code surface
and were **removed from the codebase entirely** (`HyperConnectionTrunk`, `sinkhorn_knopp`,
`TrunkConfig.n_streams`/`sinkhorn_iters`/`identity_init_scale`/`stream_semantics`/
`semantic_readout`, the six arm entries and their configs/on-disk runs/checkpoints) rather
than merely demoted to a kept ablation arm — see CLAUDE.md's "Model compatibility note".
`TrunkConfig.kind` now supports only `none`/`plain`. If mHC is ever worth revisiting, the
implementation is recoverable from git history before this removal; re-derive the
measurement above rather than assuming it still holds against whatever the reference
architecture has become by then.

Note (updated 2026-09-05): `FeatureFusionArchitectureVariantModel` (`models.py`) is no
longer throwaway code to delete once this study concludes -- since 2026-09-05 it is the
live primary RNA-methylation model class (`encoder.kind=locus_attention`, see CLAUDE.md
and docs/RNA_METHYLATION.md), and `configs/models/arch_shared_backbone/` is the ongoing
home of the RNA-encoder-comparison harness that reuses this same registry across studies
(docs/RNA_METHYLATION.md's "Forward direction" section) -- neither should be deleted.
**Update (later refactor)**: `ArchitectureVariantModel` (the older two-stage variant class) and
its `configs/models/arch/` recipes have since been **removed entirely**, along with the whole
two-stage frozen-prior + residual architecture generation and its `MethylProphetTrainer`/
`ScopedRNATrainer` engines -- see CLAUDE.md's "Model compatibility note". The Stage 2/3
Hyper-Connections/mHC arms described above have also been removed entirely (code, configs, and
on-disk runs/checkpoints) -- see "What was tried and removed" above. This study's own
queue-runner scaffolding -- `scripts/experiments/{run_arch_suite,collect_arch_results}.py` --
and its now-closed arm entries in `arch_suite.py`/`configs/models/arch_shared_backbone/`
remain candidates for removal once the verdict above is archived, per `CLAUDE.md`'s rule on
one-off experiment code.
