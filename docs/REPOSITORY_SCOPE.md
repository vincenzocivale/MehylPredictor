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
