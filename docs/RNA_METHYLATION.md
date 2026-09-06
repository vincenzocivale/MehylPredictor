# RNA methylation model

`FeatureFusionArchitectureVariantModel` with `encoder.kind=locus_attention` is the repo's
**primary/reference RNA-methylation architecture** as of 2026-09-05, superseding the
`encoder.kind=linear` reference of 2026-09-03 -- see "2026-09-05 update" below for the
ablation evidence. See [`CPG_STATISTICS.md`](CPG_STATISTICS.md) for the companion
`CpGStatisticsPredictor` (mu, sigma) model whose `target_mu` output feeds this model's
mean-branch proxy task, and [`EXPLAINABILITY.md`](EXPLAINABILITY.md) for attributing a
trained checkpoint's predictions back to RNA genes (`scripts/explain.py`).

## Primary architecture: shared-backbone late fusion

A single-stage model: two branches, each producing a learned feature vector, late-fused
into one prediction head -- no separate frozen prior stage, no explicit
`mu + sigma*residual` composition, and no post-fusion trunk (`model.trunk.kind=none`).

```text
CpG -> frozen 1536-D NTv3 embedding e_l
  -> CpGTrunk (LayerNorm -> Linear(256) -> GELU -> Dropout -> Linear(64) -> GELU) -> h_mean_l
  -> mean_head: Linear(h_mean_l) -> mu_logit_l   [auxiliary proxy-task probe only, not used in beta]

RNA (25,017) -> LayerNorm -> Linear(256) -> GELU -> LayerNorm -> u_s (patient bottleneck)
  -> 64 learned bases -> program tokens T_s (64 x 256)
e_l -> query Q(e_l) --4-head cross-attention over T_s--> r_{s,l} (256, locus-conditioned RNA)

[r_{s,l}, e_l] -> LayerNorm -> Linear(128) -> GELU -> Dropout -> h_raw_{s,l}
  -> residual_head: Linear(h_raw_{s,l}) -> residual_logit_{s,l}   [auxiliary probe only, weight=0 in the reference recipe]

fusion: Linear([h_mean_l, h_raw_{s,l}]) -> beta_logit_{s,l} -> sigmoid -> beta_hat_{s,l}
```

`beta_hat` is bounded in (0,1) by construction (final `sigmoid`), same as the earlier
architecture. `mu_logit`/`residual_logit` are auxiliary training-time probes only --
`mean_head`'s output shapes `h_mean` via a proxy loss but never enters the `beta`
computation directly; `residual_head` is available (`residual_aux_weight` in the recipe)
but weighted to 0 in the reference configuration (see selection rationale below). The RNA
encoder is the one piece of this diagram that is **not** fixed going forward -- see
"Forward direction" below.

Code: `models.py::FeatureFusionArchitectureVariantModel` (`encoder.kind=locus_attention`,
`trunk.kind=none`) + `CpGTrunk`, trained via
`rna_training/locus_cls_trainer.py::LocusCLSJointTrainer`. The older
`models.py::FeatureFusionLocusCLSModel` (`encoder.kind` hardcoded to `linear`) remains live
for old-checkpoint compatibility -- see the model-compatibility note in `CLAUDE.md` -- but
is no longer where new recipes should target. Entry points:

```bash
python scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
  --recipe configs/models/rna_methylation_locus_attention.yaml --mode final \
  --prepared-root ... --canonical-root ... --feature-cache ... --rna-cache ... \
  --registry ... --cpg-targets-dir ... --output-root ...

python scripts/evaluate.py --model rna_methylation --engine matched_chr1_shared_backbone \
  --checkpoint /path/to/last.pt --eval-scope chr1 \
  --prepared-root ... --canonical-root ... --feature-cache ... --rna-cache ... \
  --registry ... --cpg-targets-dir ... --output ...
```

`configs/models/rna_methylation_shared_backbone.yaml` (`encoder.kind=linear`) remains as
the frozen, previous-reference recipe for compatibility/comparison.

The shared-backbone engine also supports canonical chr123 training via
`--engine shared_backbone`; pass its protocol-ordered target cache through
`--prepared-root`. See [`CHR123_TRAINING_OPTIMIZATIONS.md`](CHR123_TRAINING_OPTIMIZATIONS.md)
for the cache contract, preparation command, measured 7.4x epoch speedup and provenance
caveats. The legacy `matched_chr1_shared_backbone` engine name remains supported for the
exact chr1 path.

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
simple for the paper's novelty claim, targets two things the ladder above never touched:
the raw branch's RNA encoder (still a single `Linear(25017 -> 256)`, no nonlinearity) and
whether depth after `h_mean`/`h_raw` are combined helps at all (still one `Linear`). It
adds `models.py::FeatureFusionArchitectureVariantModel` -- a separate class, so
`FeatureFusionLocusCLSModel`'s parameter set is untouched -- selected whenever a recipe
sets a non-default `model.encoder.kind`, `model.trunk`, `model.axial` or
`model.beta_likelihood_head`. Every arm runs on `matched_chr1_shared_backbone` at
`mode=final`, evaluated via `evaluate_official_split`, so its MAS-PCC is directly
comparable to rung B's own number. Its noise-floor arm (3 seeds, full 80-epoch budget)
doubles as the full-budget rerun this section already flagged as worth doing -- see
`results/reference/ablations/architecture_novelty_2026_09/README.md`.

Note: this suite originally targeted the earlier two-stage architecture (see that
section's own note below) -- it was rebuilt on `FeatureFusionLocusCLSModel` on
2026-09-04, once this ladder concluded and the shared-backbone model became primary. A
multi-stream (Hyper-Connections / Manifold-Constrained-HC) trunk kind was also tried as
part of this suite and later **removed from the codebase entirely** -- see the note at the
end of the next section and CLAUDE.md's "Model compatibility note".

### 2026-09-05 update: locus-attention becomes the reference RNA encoder

The architecture-novelty suite's locus-conditioned-RNA arm (`enc_locus_attention_k64` /
"row B" below) beat the then-reference linear-encoder model on every official
view/metric. A chr1/seed-17 isolation ladder, matched on every training hyperparameter
(lr 2e-4, cosine_warmup, 80 epochs, same batching/loss), quantifies exactly this encoder
change against the product term and mean branch:

| arm | encoder | trunk | product term | mean branch | official MAS-PCC (train_cpg×val / val_cpg×train / val_cpg×val) |
| --- | --- | --- | --- | --- | --- |
| A (old reference) | linear | none | yes | yes | 0.6565 / 0.6138 / 0.5760 |
| **row B** | **locus_attention** | none | yes | yes | **0.7179 / 0.6385 / 0.5938** |
| row B, mean removed | locus_attention | none | yes | no | 0.7212 / 0.6301 / 0.5860 |
| row B, product+mean removed | locus_attention | none | no | no | 0.7046 / 0.6232 / 0.5812 |

Conclusions supported by the table above:

- **Locus-conditioned RNA (row B vs A) is a decisive, consistent win** on every view:
  +0.0614 train_cpg×val_sample, +0.0247 val_cpg×train_sample, +0.0178 val_cpg×val_sample
  (the hardest, double-OOD view). This is *the* effect worth building on.
- **The product term and the mean branch are not interchangeable** despite both looking
  like "extra complexity added to row B". Removing only the mean branch collapses MSE
  specifically on the two unseen-CpG views (val_cpg×train_sample: 0.01334 -> 0.01714,
  +28.5%; val_cpg×val_sample: 0.01414 -> 0.01788, +26.5%) while barely moving MAS-PCC --
  the mean branch is the model's only locus-specific signal that does not require having
  seen that CpG's RNA relationship before, so losing it mainly hurts calibration on novel
  loci, not correlation. The product term's effect is the opposite shape: comparing "mean
  removed" against "product+mean removed" shows the product term recovers MSE on
  train_cpg×val_sample (already-seen loci: 0.01191 -> 0.01088, -8.6%) but does essentially
  nothing on either unseen-CpG view (-1.3 to -1.5%) -- its benefit is concentrated on the
  single least-generalization-relevant view. The reference recipe below therefore keeps
  the mean branch and drops the product term (see that recipe's header comment for the
  one caveat: the exact cell "product removed, mean kept" was not itself an isolated run
  before this decision -- only "both kept", "both removed", and "mean removed only" have
  official numbers).

**New reference recipe**: `configs/models/rna_methylation_locus_attention.yaml` --
`encoder.kind=locus_attention`, `use_mean_branch=true`, `use_raw_product=false`,
`trunk.kind=none`. Train/evaluate before citing its numbers (see recipe header).

Open questions (do not overclaim past this point):

- **Does any extra trunk depth help the reference (locus-attention) encoder at all?** Row
  C (`locus_attention` + a parameter-matched **plain** depth-4 trunk) tests this directly.
  Interrupted at epoch 21/80, never evaluated -- see
  `results/reference/ablations/architecture_novelty_2026_09/README.md`'s status section
  and `docs/EXPERIMENT_ROADMAP.md` for the resume command.
- **Single seed, chr1 only, official-split-informed selection.** Every number above is one
  seed; chr123 contains chr1; the architecture decisions here were made looking at chr1's
  official split. None of the multi-seed/fresh-scope requirements this warrants have been
  run for this specific ladder.
- Reproducibility: chr1 runs `p0_row_b_locus_attention_matched`,
  `p0_row_b_no_mean_only`, `p0_row_b_no_product_no_mean` (recipes under
  `configs/models/arch_shared_backbone/`), evaluated via the same
  `evaluate_official_split` path as the flagship arm.

### Forward direction: RNA-encoder comparison harness

With locus-conditioned attention shown to be the dominant lever and the rest of the
architecture (mean branch, concat fusion, direct sigmoid readout) now fixed as a stable
harness, **the repo's RNA-methylation work going forward is a comparison of RNA-encoding
approaches**, not further additions to the fusion/trunk side. `EncoderConfig.kind` +
`build_rna_encoder()` (`models.py`) is the extension point: a new encoder only needs to
produce a `RNARepresentation` (a per-sample global vector, plus optionally per-sample
`program_tokens` for locus-conditioning) to slot into the existing harness and be
comparable to `locus_attention`/`linear`/`mlp`/`program_bottleneck` on the same protocol.
Candidate directions worth an arm, roughly in order of implementation cost (see the
literature survey in this repo's working session notes for detail on each):

- **evaluated 2026-09-06, underperforms row B** -- `encoder.kind=bottleneck_mlp`
  (`configs/models/arch_shared_backbone/enc_bottleneck_mlp.yaml`,
  `models.py::BottleneckMLPEncoder`): reproduces the *architecture* of MethylProphet's
  published RNA branch (bioRxiv 2025.02.05.636730, `github.com/xk-huang/methylprophet`,
  `BottleneckMLP` "B_6-Wi_1024" -- 6 pre-norm residual blocks, width 1024, GELU), trained
  from scratch in place of the current encoder. Deliberately not their preprocessing
  (log-quantize to [0,1]) or their fusion (DistilBERT token concatenation) -- it consumes
  the same frozen z-scored RNA cache and plugs into the same fixed branch architecture as
  every other arm here, isolating the encoder architecture as the only variable. Trained
  and evaluated chr1/seed17, matched to row B's own protocol: official MAS-PCC
  0.6631 / 0.6127 / 0.5810 (train_cpg×val / val_cpg×train / val_cpg×val), **-7.6% / -4.0% /
  -2.2% vs. row B** on every view -- the training-loss curve was already visibly worse
  than row B's from epoch ~10 onward, not just a held-out generalization gap. A
  from-scratch bottleneck MLP does not match locus-conditioned attention at this data
  scale; not promoted.
- **evaluated 2026-09-06, underperforms row B (worse than bottleneck_mlp above)** --
  `encoder.kind=frozen_embedding`
  (`configs/models/arch_shared_backbone/enc_frozen_embedding_bulkrnabert.yaml`,
  `models.py::FrozenEmbeddingEncoder`, `scripts/prepare_bulkrnabert_embeddings.py`): a
  frozen, *not* fine-tuned, embedding from BulkRNABert
  (`github.com/instadeepai/multiomics-open-research`, CC BY-NC-SA 4.0 --
  non-commercial), a small pretrained bulk-RNA-seq transformer, adapted by a single
  trainable Linear projection. The extraction script's model/tokenizer API was verified
  against the live checkpoint, including a real bug fixed in the vendored tokenizer
  (`tokenizer.mask_token_id` crashes with `KeyError: None`; the script reads
  `tokenizer.vocab` directly instead) and a real memory constraint discovered by actually
  triggering it (full self-attention over the ~19,062-gene sequence needs far more memory
  than one batch-size unit's naive estimate, and bfloat16 + explicit CUDA-cache-clearing
  between samples was required to complete a full run without OOM -- see that script's
  docstring). The full 10,916-sample extraction has been run end to end (cache at
  `MethylPredictionData/derived/bulkrnabert_embeddings/tcga`, real coverage: 15,259/19,062
  BulkRNABert genes matched, 80.0%). Trained and evaluated chr1/seed17, matched to row B's
  protocol: official MAS-PCC 0.5795 / 0.5321 / 0.5164, **-19.3% / -16.7% / -13.0% vs. row
  B** -- a substantially larger gap than the from-scratch bottleneck MLP above, consistent
  with an embedding that was never fine-tuned for this task and only adapted by a single
  Linear layer, plus incomplete gene-panel coverage. Not promoted.
  `frozen_embedding_source` records provenance for any future frozen-embedding arm
  (Geneformer, scGPT -- see below), not just this one.
- data-derived gene programs (consensus NMF / topic-model-style modules discovered from
  co-expression, e.g. cNMF, SPECTRA) as the token bank instead of `program_basis`'s fully
  end-to-end-learned bases -- would also finally let the "gene program" language used
  informally today be backed by a stability/enrichment check, which the current learned
  tokens explicitly cannot support (see the locus-attention encoder's own docstring);
- pathway/gene-set-informed projections (MSigDB/KEGG/Reactome) as a biologically
  constrained alternative bottleneck;
- frozen embeddings from other pretrained single-cell/bulk transcriptome foundation
  models (Geneformer's rank-value tokens, scGPT's binned-value tokens), sharing
  `FrozenEmbeddingEncoder` with the BulkRNABert arm above via `frozen_embedding_source`;
- deeper Perceiver-IO-style query/latent designs (the existing cross-attention is already
  in this family; the open question is whether more latent slots, more cross-attention
  rounds, or a learned rather than CpG-derived query change anything).

Any new arm should be evaluated under the same matched-hyperparameter protocol used
above (chr1, seed 17 minimum, `evaluate_official_split`'s three views) against row B's
numbers as the baseline to beat.

## Retired architecture: two-stage frozen prior + residual

An earlier generation (`RNAMethylationPredictor`/`VarianceNormalizedResidualModel`/
`RNA2DNAmModel`/`ArchitectureVariantModel`/`DirectPredictionModel`, trained via the
`MethylProphetTrainer`/`ScopedRNATrainer` engines: `logit(beta_hat) = logit(mu_i) + sigma_i *
raw_delta`, an explicit frozen-prior + variance-normalized-residual composition) has been
**removed entirely**, including old-checkpoint compatibility -- see CLAUDE.md's "Model
compatibility note". Its frozen chr1 numbers (MAS-PCC 0.5613 on `val_cpg_x_val_sample`, the
number the shared-backbone architecture above was originally selected against) remain under
`results/reference/methylprophet_comparison/` and `results/reference/rna_methylation/chr1.yaml`'s
`legacy_two_stage` field as historical provenance; the ablation ladders that were run against it
(`interaction_concat_and_latent_dim_2026_08`, `sigma_normalization_2026_08`,
`prior_anchor_2026_08`, `fusion_mechanism_2026_08`) remain recorded in
`results/reference/ablations.yaml` for the same reason. The code itself is only recoverable from
git history before this removal.

Three things that depended on this generation were **ported** to the current architecture rather
than retired with it:

- **Explainability** (`scripts/explain.py`) now attributes `FeatureFusionLocusCLSModel`/
  `FeatureFusionArchitectureVariantModel`'s `residual_logit` auxiliary probe instead of the old
  `raw_delta` -- see `docs/EXPLAINABILITY.md`.
- **Hyperparameter tuning** (`scripts/tune.py`) now runs `LocusCLSJointTrainer` instead of the
  retired `ScopedRNATrainer`.
- **Three of the four paper-required simplified baselines** (Global RNA Shift, Bilinear RNA-CpG,
  MLP RNA-CpG -- the fourth, CpG Prior, has no model) are now expressed as
  `FeatureFusionArchitectureVariantModel` configurations instead of `InteractionConfig.kind`
  variants on the old engine: `use_mean_branch=False` plus `include_raw_rna`/`include_raw_cpg`/
  `use_raw_product` constructor kwargs select which raw-branch pieces feed the joint input (Global
  RNA Shift: RNA only; Bilinear RNA-CpG: product term only; MLP RNA-CpG: concatenated RNA+CpG, no
  product) -- see `docs/PAPER_EXPERIMENTS.md`'s "Baseline models" section and
  `FeatureFusionArchitectureVariantModel`'s docstring in `models.py`.
