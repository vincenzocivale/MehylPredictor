# Repository scope during the paper refactor

This branch is being reduced from a research-history repository to the public
reproducibility repository for the paper.

The paper-facing repository should retain only code and artifacts required to:

1. build the TCGA/ENCODE inputs and official sample/CpG splits;
2. train the selected functional-locus + RNA methylation predictor;
3. train the mean-proxy ablation and the minimal claim-driven comparators;
4. evaluate the three official sample/CpG generalization views;
5. reproduce MethylProphet/external-model comparisons used in the paper;
6. reproduce efficiency measurements and manuscript tables.

Historical architecture-search ladders, one-off launchers, planning documents,
intermediate diagnostics, and superseded result ledgers belong to Git history or
the original research branch, not to the final submission surface.

## Transitional files intentionally retained

Until the final architecture is selected, these remain on this branch:

- `configs/models/functional_fusion/base.yaml`
- `configs/models/functional_fusion/j0_final.yaml`
- `configs/models/functional_fusion/j1_iterative.yaml`
- `scripts/experiments/wait_and_run_iterative_retrieval.sh`

The mean-proxy experiment harness, RNA-encoder comparison, official baselines,
and benchmark code are also retained until the paper experiment set is frozen.

No core Python model/trainer code is removed in phase 1. Code pruning starts
only after behavior-preserving regression tests are in place.

## Phase 3a: paper-facing public cutover

The public/core interface now points to:

```python
methylation_predictor.SingleRetrievalPredictor
methylation_predictor.IterativeRetrievalPredictor
```

and the standalone reference recipe is:

```text
configs/models/main.yaml
```

`main.yaml` is regression-tested to resolve exactly to the historical
`functional_fusion/j0_final.yaml` recipe during this migration.

The README and CLI documentation now describe the functional-atlas + RNA
retrieval method rather than the superseded NTv3/shared-backbone reference.

The old `FeatureFusionLocusCLSModel` and
`FeatureFusionArchitectureVariantModel` remain lazily reachable only as
temporary compatibility infrastructure because surviving paper baselines and
RNA-encoder comparators still depend on them. They are no longer exported as
the public model API. Phase 3b will migrate the comparators that remain
scientifically justified onto the functional-locus core before deleting the
legacy family.


## Phase 3b1: functional mean-proxy ablation

The mean-contribution experiment now runs directly on the paper-facing
`SingleRetrievalPredictor`, not on the historical shared-backbone model.

The three chr1 arms are:

```text
full_reference       : mean head ON,  aux_weight=0.15
no_mean_supervision  : mean head ON,  aux_weight=0
no_mean_branch       : mean head OFF, aux_weight=0
```

Every other resolved recipe field is identical after removing tracking
metadata. When the mean head is removed, its initialization draws are still
consumed before the module is discarded, so every surviving shared parameter
has bit-identical same-seed initialization relative to the full model.

The migrated experiment uses a new study name and new run IDs; historical
shared-backbone results therefore cannot be silently mistaken for functional
J0 results. The migrated harness is intentionally chr1-only until equivalent
functional-atlas coverage is frozen for broader scopes.


## Phase 3b2: functional-matched internal baselines

Three new internal baselines now share the paper-facing functional locus
encoder, RNA `ProgramTokenEncoder`, mean-proxy task, loss, optimizer, batching,
and chr1 protocol with `main.yaml`.

```text
functional_global_rna_shift
    locus logit + one patient-global RNA logit shift

functional_mlp_rna_cpg
    [functional h_c ; globally pooled RNA tokens] -> MLP

functional_bilinear_rna_cpg
    [functional h_c ; U(h_c) * V(global RNA)] -> MLP
```

These isolate the value of locus-conditioned RNA retrieval instead of
confounding the comparison with a different CpG representation.

The older shared-backbone baseline recipes and their recorded result ledger are
retained unchanged for provenance until the new baselines are run. Fresh run
IDs are registered in `scripts/experiments/functional_baseline_suite.py`.

## Phase 3b3a: functional-matched RNA encoder comparison

RNA encoder comparisons now have a paper-facing implementation that keeps the
functional locus representation and J0 retrieval stack fixed.

Every arm emits the same `[patient, 64, 256]` program-token contract before
the unchanged locus-conditioned retrieval:

```text
ours             canonical RNA -> ProgramTokenEncoder -> 64x256 tokens
BottleneckMLP    canonical RNA -> MP-style global encoder -> token lift
GenePathway      canonical RNA -> sparse pathway encoder -> token lift
BulkFormer       frozen sample embedding -> thin adapter -> token lift
BulkRNABert      frozen sample embedding -> thin adapter -> token lift
```

`RNAEncoderComparisonPredictor` constructs the production J0 downstream stack
with the canonical 25,017-gene reference input before replacing only the RNA
encoder. Therefore the functional CpG encoder, retrieval attention, mean-proxy
head and final regressor retain bit-identical same-seed initialization across
comparison arms.

Historical shared-backbone RNA-encoder configs and recorded result ledgers are
retained unchanged for provenance. Fresh functional-matched results must use
the new `functional_*` recipes and fresh run IDs.

## Phase 3b3b: functional RNA-encoder execution harness

The executable RNA-encoder runner and collector now target only the
functional-matched comparison introduced in phase 3b3a.

The migrated runner:

- is chr1-only;
- passes the frozen functional atlas and annotation cache explicitly;
- always supplies the frozen functional atlas and annotation cache;
- selects canonical RNA, BulkFormer, or BulkRNABert sample caches explicitly;
- uses fresh `functional-rnaenc-*` run IDs;
- supports `--print-commands` for command-level audit before GPU execution.

The collector writes to a new
`functional_rna_encoder_comparison_2026_09` result directory and never mixes
historical shared-backbone metrics with the new functional comparison.

Historical result ledgers and historical recipes remain untouched for
provenance; their old executable runner logic has been retired.

## Phase 4a: explainability removed

Checkpoint explainability is intentionally out of scope for the paper-facing
reproducibility repository. The previous Expected/Integrated-Gradients
implementation targeted the retired shared-backbone `residual_logit` and was
not part of the paper's central claims.

The implementation, CLI, tests, and dedicated documentation were removed
rather than migrated. The public workflow is therefore limited to data
preparation, training, hyperparameter selection where retained, and
evaluation. Historical explainability code remains available through Git
history.

## Phase 4b1: retired model family removed

The historical `models.py` FeatureFusion/shared-backbone implementation has
been deleted. All surviving paper-facing trainable RNA models now live under
`methylation_predictor.modeling`:

```text
SingleRetrievalPredictor
IterativeRetrievalPredictor
FunctionalBaselinePredictor
RNAEncoderComparisonPredictor
```

Architecture-search recipes, superseded shared-backbone reference recipes,
and tests whose only purpose was to exercise the retired model family were
also removed. Historical metrics remain in the version-controlled result
ledgers for provenance, and the deleted implementation remains available in
Git history.

`LocusCLSJointTrainer` now dispatches only to the functional paper-facing
models and uses only the functional objective path. Some obsolete constructor,
checkpoint-metadata, and config fields are intentionally left for phase 4b2
so this large model-family deletion can be regression-tested independently.

## Phase 4b2a: live experiment registry isolated from trainer

The refactor branch accumulated additional functional experiments while the
repository cleanup was in progress (depth/residual, efficient single-attention,
gated residual, functional-branch depth, and FFN-fusion variants).

Those variants are now a protected live experiment surface. Their
selector-to-constructor mapping lives in `modeling/factory.py`; the training
harness no longer embeds J-number-specific architecture dictionaries.

This phase intentionally changes no recipe, checkpoint schema, model math,
optimizer, batching, loss, run ID, or result path. Configuration-schema
pruning is deferred until the active experiment recipes are frozen.

## Phase 4b2b: retired trainer controls removed from the runtime API

Shared-backbone-only locus controls are no longer constructor or CLI
arguments of the paper-facing RNA trainer. Active J0/J1/J4-J10 model
construction remains isolated in `modeling/factory.py`.

For experiment safety, the current recipe files and legacy resolved-config
shape are intentionally unchanged in this phase. A narrow compatibility
parser validates that retired recipe keys remain at their canonical no-op
values and reproduces the same resolved metadata, so in-flight functional
runs remain resumable.

The actual recipe/config-schema deletion is deferred until the active
architecture experiments are frozen.

## Phase 4c: RNA runtime independent of genomic feature embeddings

The RNA workflow no longer consumes the historical 1,536-D CpG embedding cache. Model batches contain RNA, methylation targets, and functional-locus features only.

Prior-relative metrics use the separate `LocusPriorCache` contract:

```text
cpg_idx.npy
prior.npy
```

The old derived feature directory may still be supplied as `--prior-cache` during migration because it contains those two arrays, but the RNA runtime does not open its embedding or sigma files. The standalone `cpg_prior` baseline remains the only supported evaluator that still uses `LocusFeatureCache`.

All active J-series shell launchers, the mean-proxy harness, and the RNA-encoder-comparison harness now pass `--prior-cache`, so the executable experiment surface matches the runtime API.

## Phase 4d1: single RNA trainer and CLI

The live trainer is now `RNAMethylationTrainer`. The historical `LocusCLSJointTrainer` symbol remains only as a temporary direct-import compatibility alias and is no longer part of the package public API.

RNA training and evaluation no longer expose `--engine` or `--functional-only`. Functional locus inputs are mandatory and there is only one paper-facing runtime. Existing run directories and checkpoint metadata intentionally keep the historical `locus_cls_joint` identifier so in-flight runs remain resumable and existing evaluation paths do not move.

## Phase 4d2: trainer module cutover

The live RNA implementation now resides in `rna_training/rna_methylation_trainer.py`. The old `rna_training/locus_cls_trainer.py` file is a small compatibility shim only.

The public evaluator name is `evaluate_rna_checkpoint`; the historical `evaluate_official_split` name remains available only through compatibility imports.

Operational docs and agent guidance now describe the functional-locus runtime that actually exists. The historical `locus_cls_joint` run/checkpoint identifier remains unchanged for resume/evaluation compatibility.

## Phase 5a: experiment surface inventory and freeze

All `configs/models/**/*.yaml` recipes and all current
`scripts/experiments/*.{py,sh}` files are now classified in
`configs/experiment_surface.yaml` as paper-facing, support, compatibility, or
protected research.

J1/J2-J10 remain protected and content-hashed until the final architecture
selection is locked. They are not part of the public paper model surface.
`configs/models/main.yaml` remains the sole primary reference recipe.

No architecture, loss, batching, optimizer, run ID, checkpoint, or result
semantics changed in this phase.

## Phase 5b: configuration-field audit

The remaining RNA architecture/loss config fields are classified in `configs/config_field_audit.yaml`. No fields are removed in this phase because protected J-series runs remain resumable. Compatibility-only ModelConfig, LossConfig and `locus_cls` keys now have an explicit deletion gate tied to final architecture selection.

## Phase 5c: architecture-decision audit collector

`scripts/architecture_decision_audit.py` provides a read-only inventory of J0/J-series run metrics and compute characteristics. It deliberately lives outside `scripts/experiments/`, so adding the analysis utility does not mutate the frozen experiment surface or recipe hashes.

## Phase 5c: repository cleanup inventory

Transient refactor helpers, tracked diagnostic/eval scratch outputs, and two stale runtime documents were removed. Active J-series recipes/launchers remain protected until architecture selection. `results/reference/` is explicitly deferred for a clean rebuild after fresh final paper runs. External data is not deleted in this phase; later deletion requires a dry-run dependency scan against final run manifests.
