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
compatibility note") — the older model is now kept only for old-checkpoint
compatibility.

Every arm below was rebuilt on top of the new reference architecture. The
original arms and their `configs/models/arch/` recipes stay in the repo,
unmodified, as a **compatibility-only** measurement of the retired
architecture (`scripts/experiments/{arch_suite,run_arch_suite,collect_arch_results}.py`
now target the shared-backbone family; the retired suite's driver was stopped
after 2 epochs of its noise-floor arm — resumable via `--resume`, `latest.pt`
saved — but was not continued, since the new reference architecture makes it
no longer the primary comparison).

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

## Protocol

Every arm: `matched_chr1_shared_backbone` engine, `mode=final`, 80 epochs,
`schedule_policy=pair_complete`, evaluated via
`rna_training/locus_cls_trainer.py::evaluate_official_split` on the official
MethylProphet Table-5 chr1 Array views (view `official_val_cpg_x_val_sample`).

The stage-2 HC/mHC runs launched on 2026-09-03 use the user-selected aggressive
optimizer profile: peak lr 2e-4, `cosine_warmup`, one warm-up epoch, and
`min_lr_ratio=0.1` (final lr 2e-5). Other arms retain the original lr 5e-5
constant protocol unless their recipe records an explicit override.

**Two mechanical differences from the retired suite, both consequences of how
this engine works:**

1. **Training and evaluation are separate steps.** `scripts/train.py` only
   trains; the official-split number comes from a second
   `scripts/evaluate.py --engine matched_chr1_shared_backbone` call against
   the checkpoint. `run_arch_suite.py` runs both back to back per unit.
2. **No resume support.** `LocusCLSJointTrainer`'s `RunStore.create` is never
   called with `resume=True` — a run directory left behind by a killed
   process is a dead end, not something a re-invocation can continue. The
   runner reports a blocked unit and moves on rather than deleting or
   fighting over an existing directory.

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
| 2-hyper-connections | `trunk_hc_n2_d4`, `trunk_mhc_n2_d4`, `trunk_mhc_stream_semantics_d4` | Do two residual streams over the joint help, and does the manifold constraint matter? |
| 3-combination | `combined_locus_attention_mhc_semantic` | Do the two novelty components compose? |

**The flagship arm is `trunk_mhc_stream_semantics_d4`** — instead of the
reference architecture's single `Linear([h_mean, h_raw])`, the mean and raw
branch embeddings themselves *are* the `n_streams=2` of an mHC trunk: they
exchange information under a doubly stochastic (mass-conserving) mixing
matrix for `depth` steps before the final head. This is the most literal
possible answer to "improve how the two branch embeddings are combined" —
asked of the architecture that actually has two named branch embeddings to
combine, rather than of a single internal joint vector. The exchange matrix
is directly plottable per depth
(`models.py::HyperConnectionTrunk.residual_mappings`), an interpretability
figure the flat single-`Linear` fusion cannot produce.

Two mHC/HC pairs exist deliberately: `trunk_hc_n2_d4`/`trunk_mhc_n2_d4` are a
depth control on the *concatenated* joint (matched to the `trunk_plain_dX`
ladder, isolating "did stream mixing help" from "did depth help");
`trunk_mhc_stream_semantics_d4` is the arm where the streams have identity.

Unlike the retired suite, there is **no capacity-matched fusion-mechanism
retest arm** here: the raw branch's product-term construction
(`[rna, cpg, proj(rna)·proj(cpg)]`) is fixed inside
`FeatureFusionLocusCLSModel`/`FeatureFusionArchitectureVariantModel` and does
not go through `InteractionConfig.kind` dispatch the way the retired
architecture's fusion did, so `fusion_mechanism_2026_08`'s FiLM/bilinear/
cross-attention arms have no direct analogue to retest here. The
`trunk_hc`/`trunk_mhc` arms are this architecture's fusion-mechanism axis.

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
    --output-root /dune/DATASETS/MethylPredictionData/experiments \
    --output-root /mnt/m/experiments
```

Run ids are deterministic per (arm, seed). Unlike the retired suite, a killed
unit's run directory **cannot** be resumed on this engine — the runner
reports it as blocked and moves to the next unit.

Budget: 15 units, ~3.5–4.5 h training + a shorter evaluation pass each, on one
RTX PRO 5000. Measured peak GPU memory at the real WGBS block size
(32×20480, this architecture's batching) is 15–19 GB across all 13 arms
(`--cuda-smoke --smoke-samples 32 --smoke-loci 20480`), comfortably under the
`--min-free-gb` default of 20.

## Files here

- `runs/<arm>__seed<N>.json` — one self-describing record per completed run:
  official-split metrics, checkpoint path/epoch, training summary, and the
  mHC signal-propagation diagnostics when the arm has a multi-stream trunk.
  Generated; do not hand-edit.
- `summary.yaml` / `summary.md` — generated rollup with the noise floor and
  per-arm verdict.

When the study concludes, fold the verdict into `results/reference/ablations.yaml`
as a normal study block, add the run rows to `results/reference/PROVENANCE.md`,
and delete the suite's throwaway code
(`scripts/experiments/{arch_suite,run_arch_suite,collect_arch_results}.py`,
`configs/models/arch_shared_backbone/`, `configs/models/arch/` (retired arm),
and both `ArchitectureVariantModel`/`FeatureFusionArchitectureVariantModel` in
`models.py`) per `CLAUDE.md`'s rule on one-off experiment code.
