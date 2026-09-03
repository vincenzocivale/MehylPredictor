# RNA methylation model

`FeatureFusionLocusCLSModel` is the repo's **primary/reference RNA-methylation
architecture** as of 2026-09-03. See [`CPG_STATISTICS.md`](CPG_STATISTICS.md) for the
companion `CpGStatisticsPredictor` (mu, sigma) model whose `target_mu` output feeds this
model's mean-branch proxy task, and [`EXPLAINABILITY.md`](EXPLAINABILITY.md) for
attributing a trained checkpoint's predictions back to RNA genes (`scripts/explain.py`,
written against the earlier two-stage architecture -- see below).

## Primary architecture: shared-backbone late fusion

A single-stage model: two branches, each producing a learned feature vector, late-fused
into one prediction head -- no separate frozen prior stage, no explicit
`mu + sigma*residual` composition.

```text
CpG -> frozen 1536-D NTv3 embedding e_i
  -> CpGTrunk (LayerNorm -> Linear(256) -> GELU -> Dropout -> Linear(64) -> GELU) -> h_mean_i
  -> mean_head: Linear(h_mean_i) -> mu_logit_i   [auxiliary proxy-task probe only, not used in beta]

RNA (25,017) -> Linear(256) -> z_s
[z_s, e_i, W_R(z_s) ⊙ W_C(e_i)] -> LayerNorm -> Linear(128) -> GELU -> Dropout -> h_raw_{s,i}
  -> residual_head: Linear(h_raw_{s,i}) -> residual_logit_{s,i}   [auxiliary probe only, weight=0 in the reference recipe]

fusion: Linear([h_mean_i, h_raw_{s,i}]) -> beta_logit_{s,i} -> sigmoid -> beta_hat_{s,i}
```

`beta_hat` is bounded in (0,1) by construction (final `sigmoid`), same as the earlier
architecture. `mu_logit`/`residual_logit` are auxiliary training-time probes only --
`mean_head`'s output shapes `h_mean` via a proxy loss but never enters the `beta`
computation directly; `residual_head` is available (`residual_aux_weight` in the recipe)
but weighted to 0 in the reference configuration (see selection rationale below).

Code: `models.py::FeatureFusionLocusCLSModel` (+ `CpGTrunk`), trained via
`rna_training/locus_cls_trainer.py::LocusCLSJointTrainer`. Entry points:

```bash
python scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
  --recipe configs/models/rna_methylation_shared_backbone.yaml --mode final \
  --prepared-root ... --canonical-root ... --feature-cache ... --rna-cache ... \
  --registry ... --cpg-targets-dir ... --output-root ...

python scripts/evaluate.py --model rna_methylation --engine matched_chr1_shared_backbone \
  --checkpoint /path/to/last.pt --eval-scope chr1 \
  --prepared-root ... --canonical-root ... --feature-cache ... --rna-cache ... \
  --registry ... --cpg-targets-dir ... --output ...
```

Currently chr1-only (matched_chr1 data); chr123/genomewide support tracked alongside the
same expansion work as the earlier architecture (`docs/PAPER_EXPERIMENTS.md`).

### Selection: the `shared_backbone_locus_cls_2026_09` ablation ladder

Prompted by a postdoc proposal (2026-09-01) for a more elegant single-stage design: use
a mean-prediction proxy task to learn locus features useful for the full prediction,
rather than the two-stage design's frozen prior + separately-trained residual. A
systematic, one-change-at-a-time ladder (chr1, `pair_complete` schedule, same batching as
the two-stage model's own 0.5613 reference run) tested each added piece of complexity
against the project's own rule -- keep only what's independently justified over the
simplest configuration:

| rung | change vs. previous | proxy-split MAS-PCC* |
| --- | --- | --- |
| A | no mean branch (raw RNA+CpG -> MLP -> beta directly) | 0.5322 |
| **B** | **+ mean branch (`CpGTrunk`+`mean_head`), concat fusion** | **0.5365** |
| C | + symmetric residual probe (`residual_head`, `residual_aux_weight=0.15`) | 0.5372 |
| D | + 4x LR on the raw/fusion branch | 0.5369 |
| E | + zero fusion-layer init (isolated from D) | 0.5370 |
| F | + element-wise product term in the fusion layer | 0.5359 |

\* `mode=development` inner proxy split (carved out of the official training pool, not
the true official held-out split -- see `docs/PAPER_EXPERIMENTS.md`'s methodology note).
Only rung A -> B is a real jump (+0.0043); C-F all land within a 0.0013 band of rung B
and each other -- noise, not signal. **None of C/D/E/F earned a permanent place** in the
reference architecture: the residual probe, LR multiplier, init tweak and fusion product
term are all still implemented (`LocusCLSJointTrainer`/`FeatureFusionLocusCLSModel`
constructor flags) but left off by default. Rung D/E do converge faster (12/9 epochs vs.
rung B/C's 18) without hurting the ceiling -- useful only if training wall-clock is a
real constraint.

**Rung B confirmed on the true official split** (2026-09-03, `mode=final`, full official
training pool, evaluated on `array_val_sample_idx x array_val_cpg_idx`): **MAS-PCC
0.5627**, slightly *above* the two-stage model's 0.5613 (+0.0014) -- and a lower bound,
since that confirmation run was stopped by user request at epoch 47/80 while training
loss was still improving (~0.3-0.8%/epoch). Single-seed, chr1 only, not yet run to
convergence -- a full-budget rerun (or a second seed) is worth doing before citing 0.5627
as this architecture's final number, but the qualitative conclusion (an elegant
single-stage design is competitive with, not worse than, the two-stage pipeline) is
already well-supported. Full numbers/caveats:
`results/reference/ablations.yaml::shared_backbone_locus_cls_2026_09`.

Per-locus analysis (rung B checkpoint, proxy split) found the mean branch predicts the
per-locus mean *better*, not worse, on the CpGs where the model struggles most on the
sample-specific value -- driven by those low-variance loci being easy to predict a mean
for, not by any mean-branch deficiency. This is why C-F (all targeting the
RNA-conditioned/residual branch) failed to move the number: the mean branch was already
doing its job.

### Architecture-novelty suite (`architecture_novelty_2026_09`)

A 2026-09 suite, prompted by a separate postdoc review that judged the architecture too
simple for the paper's novelty claim, targets two things this ladder never touched: the
raw branch's RNA encoder (still a single `Linear(25017 -> 256)`, no nonlinearity) and how
`h_mean`/`h_raw` are combined (still one `Linear`). It adds
`models.py::FeatureFusionArchitectureVariantModel` -- a separate class, so
`FeatureFusionLocusCLSModel`'s parameter set is untouched -- selected whenever a recipe
sets a non-default `model.encoder.kind`, `model.trunk`, `model.axial` or
`model.beta_likelihood_head`. The flagship arm answers the postdoc's suggestion most
literally: with `trunk.stream_semantics=True`, `h_mean` and `h_raw` themselves become the
two streams of a Manifold-Constrained Hyper-Connections (mHC, arXiv:2512.24880) trunk --
exchanging information under a doubly stochastic, mass-conserving mixing matrix for
several depth steps instead of being combined once. Every arm runs on
`matched_chr1_shared_backbone` at `mode=final`, evaluated via `evaluate_official_split`,
so its MAS-PCC is directly comparable to rung B's own number. Its noise-floor arm
(3 seeds, full 80-epoch budget) doubles as the full-budget rerun this section already
flagged as worth doing -- see `results/reference/ablations/architecture_novelty_2026_09/README.md`.

Note: this suite originally targeted the earlier two-stage architecture (see that
section's own note below) -- it was rebuilt on `FeatureFusionLocusCLSModel` on
2026-09-04, once this ladder concluded and the shared-backbone model became primary.

## Previous architecture: two-stage frozen prior + residual

Kept live and fully supported for reproducing/extending existing chr1/chr123/genomewide
checkpoints and the frozen MethylProphet benchmark path (`RNAMethylationPredictor`/
`VarianceNormalizedResidualModel` in `models.py`, `RNA2DNAmModel` for
`MethylProphetTrainer`) -- **not the recommended architecture for new work**, superseded
by the shared-backbone model above. `scripts/train.py --engine matched_chr1` (no
`_shared_backbone` suffix) and `--engine generic` still train it.

```text
RNA (25,017) -> LayerNorm -> Linear(256) -> z_s
CpG -> frozen 1536-D NTv3 embedding e_i
[z_s, e_i, W_R(z_s) ⊙ W_C(e_i)] -> MLP -> raw_delta
logit(beta_hat_{s,i}) = logit(mu_i) + sigma_i * raw_delta_{s,i}
```

`mu_i` is a probability in (0,1) (the sigmoid output of `CpGStatisticsPredictor`), not a
logit -- the sum above happens in logit space, and a final `sigmoid` (see `models.py`
`VarianceNormalizedResidualModel.forward`) maps `beta_hat` back to (0,1). This means
`beta_hat` is bounded in (0,1) by construction regardless of the residual's magnitude; no
separate clamp is needed or applied to the model's output.

Reference objective: beta MSE 1.0, standardized residual Huber 0.1,
standardized shrinkage 1e-4, locus Pearson 0.15, sigma floor 0.05.
The variability gate and mean-RNA anchor branches are historical ablations retained only
in git history. The flat residual (`RNA2DNAmModel`) and a no-prior-anchor direct
prediction (`DirectPredictionModel`) remain live, ablation-only model paths (matched_chr1
engine only). A 2026-08 re-check (product/product_only/cpg_product/no_product
interaction-concat pieces, plus a 256/512/1024 RNA-latent-width sweep; chr1, single seed)
confirmed the product term matters but found no case strong enough to change the
canonical architecture -- see
`results/reference/ablations.yaml::interaction_concat_and_latent_dim_2026_08`. A later
2026-08 suite re-measured sigma-scaling (no measurable effect,
`::sigma_normalization_2026_08`), the prior/mu anchor (a real but modest -0.0065 MAS-PCC
cost if dropped, `::prior_anchor_2026_08`), and three new RNA-CpG fusion mechanisms --
FiLM/gating, cross-attention, low-rank bilinear pooling, via `InteractionConfig.kind` +
`build_interaction()` -- none of which beat the canonical concat+product interaction
(`::fusion_mechanism_2026_08`). No permanent architecture change adopted from any of
these; the two-stage recipe (`configs/models/rna_methylation.yaml`) is unchanged.

**Historical / compatibility-only note** (retired architecture, not the primary target of the
suite as of 2026-09-04 -- see the shared-backbone section above): a 2026-09 suite
(`architecture_novelty_2026_09`) originally went after the two axes none of the above
touched, following a postdoc review that judged the architecture too simple for the paper's
novelty claim: the **RNA encoder** (canonically a single `Linear(25017 -> 256)` with no
nonlinearity -- only its width was ever ablated) and the **depth/topology of the fusion trunk**
(canonically one hidden layer, which is why Hyper-Connections and mHC have nothing to act on
until a real trunk exists). It adds `models.py::ArchitectureVariantModel` -- a separate class, so
`RNAMethylationPredictor`'s parameter set and state-dict keys are untouched -- selected whenever a
recipe sets a non-default `model.encoder.kind`, `model.trunk`, `model.axial` or
`model.beta_likelihood_head`. Every arm keeps the canonical
`logit(mu) + sigma * raw_delta` anchor and runs on the matched_chr1 engine at `mode=final`, so its
MAS-PCC is directly comparable to 0.5613 and to `fusion_mechanism_2026_08`. The suite also
introduces the repo's first **measured noise floor** (three seeds of the unmodified canonical
recipe) and judges every arm against `2 x` that seed SD -- earlier studies called deltas "in
noise" without ever measuring the noise. See
`results/reference/ablations/architecture_novelty_2026_09/README.md` for the arm taxonomy,
protocol and promotion rule, and `configs/models/arch/` for the recipes.

One result from it needs no GPU time: `BilinearInteraction` at `rank == min(rna_dim, locus_dim)`
is bit-identical to the canonical `ProductInteraction`, so `fusion_mechanism_2026_08`'s bilinear
arm (0.5460) measured a hardcoded rank-64 restriction rather than a different fusion mechanism.

Four of these ablation-only pieces are also formalized as **paper baselines** (fresh,
independently-tuned runs per setting, distinct from the ablation-suite numbers above) — see
`docs/PAPER_EXPERIMENTS.md`'s "Baseline models" section: a zero-parameter CpG-Prior floor
(`CpGPriorEvaluator`), a fourth `InteractionConfig.kind` — `global_shift`
(`GlobalShiftInteraction`, ignores the CpG embedding entirely for a single per-patient
correction), the existing `bilinear` kind with `include_rna=include_cpg=false` (bilinear term
only), and the existing `concat` kind with `include_product=false` (plain MLP, no interaction
term).
