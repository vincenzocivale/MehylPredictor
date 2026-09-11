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
