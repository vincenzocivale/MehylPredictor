# architecture_novelty_2026_09

Internal architecture ablation. **Not** a paper baseline and **not** a
MethylProphet comparison — see `results/reference/PROVENANCE.md` and
`CLAUDE.md`'s results taxonomy for why those live elsewhere. If an arm here is
ever adopted as the canonical architecture it must then be retrained and tuned
independently in all three MethylProphet-matched settings before appearing in
any "vs MethylProphet" table (`docs/PAPER_EXPERIMENTS.md`).

## Why this study exists

A postdoc reviewer (2026-09-03) judged the architecture too simple for the
paper's novelty claim despite its performance, and suggested improving how the
two branch embeddings are combined — naming *mHC: Manifold-Constrained
Hyper-Connections* (DeepSeek, arXiv:2512.24880) — and the RNA encoding.

The critique holds up against the code:

- The RNA branch is **one `Linear(25017 → 256)` after a LayerNorm** — no
  nonlinearity, no hidden layer (`models.py::LinearRNAEncoder`). The only
  encoder ablation ever run was its *width*
  (`ablations.yaml::interaction_concat_and_latent_dim_2026_08`, saturating at
  +0.002).
- `z_s` is **identical for every CpG** of a sample, so all locus specificity has
  to pass through one 256-dim product term.
- The fusion trunk is **one hidden layer**. HC and mHC are macro-design
  mechanisms for deep residual stacks (the mHC paper measures them on 27B
  models, 60 layers); on a depth-1 head they have nothing to act on. Depth is a
  prerequisite here, not an alternative.
- Drop-in fusion swaps are **already exhausted**: FiLM 0.5530, bilinear 0.5460,
  cross-attention 0.5329, all below the canonical 0.5613
  (`ablations.yaml::fusion_mechanism_2026_08`).

## Protocol

Every arm: `matched_chr1` engine, `mode=final`, official MethylProphet Table-5
chr1 Array views, 80 epochs, lr 5e-5 constant — identical to the canonical
reference (**0.5613** MAS-PCC / **0.01941** MSE,
`results/reference/rna_methylation/chr1.yaml`) and to the
`fusion_mechanism_2026_08` arms. There is deliberately **no development-mode
screening tier**: `MethylProphetTrainer` has no refit cycle by design, and a
second metric convention (`inner_double_ood_mas_pcc`) would make this suite's
numbers non-comparable to the very baseline it is judged against.

Headline metric: `val_cpg_x_val_sample.mas_pcc` (double-held-out).

## Promotion rule

**An arm counts as an improvement only if it beats the canonical seed mean by
more than 2 × the seed standard deviation.**

`seed_variance_canonical` (3 seeds of the unmodified canonical recipe) measures
that standard deviation, and must be run first. No previous ablation in this
repo ever measured it — which is why `sigma_normalization_2026_08`'s +0.0007
could only be called "in noise" by assertion rather than by measurement.

## Arms

| stage | arm | what it asks |
|---|---|---|
| 0-noise-floor | `seed_variance_canonical` | How large is a null delta on this protocol? |
| 1a-encoder-capacity | `enc_mlp` | Does the RNA branch just need a nonlinearity and a hidden layer? |
| 1a-encoder-capacity | `enc_program_bottleneck` | Does an interpretable gene-program bottleneck match a plain MLP? |
| 1b-locus-conditioned-rna | `enc_locus_attention_k64` / `_k128` | Does letting each CpG *query* the transcriptome beat one locus-invariant vector? |
| 1c-trunk-depth | `trunk_plain_d2` / `_d4` / `_d8` | Does the depth-1 head lose anything to depth, and where does it saturate? |
| 1d-axial-comethylation | `axial_cpg_d4` | Does attending to genomic neighbours along the CpG axis add signal? |
| 1e-output-likelihood | `beta_likelihood_head` | Does a bounded, heteroscedastic likelihood beat MSE on beta values? |
| 1f-fusion-retest | `fusion_bilinear_rank512`, `fusion_cross_attention_h4` | Were the 2026-08 fusion losses capacity artefacts? |
| 2-hyper-connections | `trunk_hc_n4_d4`, `trunk_mhc_n4_d4`, `trunk_mhc_semantic_d4` | Do multiple residual streams help, and does the manifold constraint matter? |
| 3-combination | `combined_locus_attention_mhc_semantic` | Do the two novelty components compose? |

Two arms carry the suite's novelty claim:

- **`enc_locus_attention_k64`** — the transcriptome becomes a set of gene-program
  tokens and the per-pair RNA vector comes from multi-head softmax cross
  attention with the CpG embedding as query, so the RNA representation itself
  becomes locus-specific. This is *not* a re-run of the losing `cross_attention`
  fusion arm: that arm was a single sigmoid-gated scalar precisely because it had
  no token axis to pool over (`models.py::CrossAttentionInteraction` says so).
  The attention weights are a directly plottable locus → gene-program figure.
- **`trunk_mhc_semantic_d4`** — mHC where the `n` streams are not anonymous
  copies of the residual width but the four **modalities** (RNA, CpG, product,
  prior). Under the doubly stochastic constraint `H^res` is then a
  mass-conserving *cross-modal exchange* operator — a convex combination of
  permutations of the modalities, readable per depth
  (`HyperConnectionTrunk.residual_mappings`). That reading exists only because of
  the manifold constraint; an unconstrained HC matrix has no conservation
  interpretation.

### Recorded finding, zero GPU cost

`BilinearInteraction` at `rank == min(rna_dim, locus_dim) == 256` is
**bit-identical** to the canonical `ProductInteraction` — same parameter count,
zero output difference under transplanted weights
(`tests/test_architecture_variants.py::test_bilinear_at_canonical_rank_is_the_product_term`).
So `fusion_mechanism_2026_08`'s bilinear arm (0.5460, −0.0153) measured a
hardcoded rank-64 restriction, not a different fusion mechanism. The retest arm
therefore sweeps rank *above* 256 rather than re-running rank 256.

## Running it

```bash
# CPU preflight: build every arm, check the zero-init contract, print sizes
python scripts/experiments/run_arch_suite.py --dry-run

# GPU preflight: one small forward+backward per arm under the real bf16 autocast.
# The CPU preflight runs in float32 and cannot catch bf16 einsum behaviour, the
# float32 Sinkhorn island inside a bf16 graph, gradient checkpointing under
# autocast, or the axial masked softmax in low precision. Needs only a couple of
# spare GB -- run it before committing to multi-hour runs.
python scripts/experiments/run_arch_suite.py --cuda-smoke --gpu 0

# the noise floor first — nothing else is interpretable without it
python scripts/experiments/run_arch_suite.py --stages 0-noise-floor --gpu 0

# then spread the rest across machines that share the output root
python scripts/experiments/run_arch_suite.py --shard 1/3 --gpu 0                    # machine A
python scripts/experiments/run_arch_suite.py --shard 2/3 --gpu 0 --data-root /mnt/m # machine B
python scripts/experiments/run_arch_suite.py --shard 3/3 --gpu 1                    # machine C

# merge everything into this directory (idempotent, re-runnable)
python scripts/experiments/collect_arch_results.py \
    --output-root /dune/DATASETS/MethylPredictionData/experiments \
    --output-root /mnt/m/experiments
```

Run ids are deterministic per (arm, seed), and a unit whose run directory already
carries the trainer's `.done` marker is skipped — so shards may overlap, a
machine can join late, and a killed runner can simply be restarted.

Budget: 18 units × roughly 3.5–4.3 h each on one RTX PRO 5000
(measured from `logs/queue_ablation_suite_2026-08*.log`).

Expected peak GPU memory is in line with the canonical run's ~12.2 GB: the
dominant tensor is still the `(batch, n_loci, 2048)` joint, and the new arms add
a few hundred MB (locus attention's `(B, heads, n_loci, K)` scores, the axial
`(B, windows, heads, w, w)` scores, the four-stream trunk state at width 128).
The runner waits for `--min-free-gb` (default 20) before starting a unit, so it
will not start on a machine that has no room.

## Files here

- `runs/<arm>__seed<N>.json` — one self-describing record per completed run:
  metrics for all three views, resolved model/loss config, host, git commit,
  checkpoint sha256, wall clock, peak GPU memory, and the mHC signal-propagation
  diagnostics. Generated; do not hand-edit.
- `summary.yaml` / `summary.md` — generated rollup with the noise floor and the
  per-arm verdict.

When the study concludes, fold the verdict into
`results/reference/ablations.yaml` as a normal study block, add the run rows to
`results/reference/PROVENANCE.md`, and delete the suite's throwaway code
(`scripts/experiments/{arch_suite,run_arch_suite,collect_arch_results}.py`,
`configs/models/arch/`, and the `ArchitectureVariantModel` machinery) per
`CLAUDE.md`'s rule on one-off experiment code.
