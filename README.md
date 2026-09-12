# MethylPredictor

MethylPredictor predicts patient-specific DNA methylation from bulk RNA using
**reference functional information at each CpG locus**.

The paper-facing model does not require a genomic foundation-model embedding as
its locus representation.  Instead, each CpG is represented from a sparse
regulatory atlas and static genomic annotations, and that locus representation
retrieves patient-specific information from learned RNA program tokens.

```text
reference functional atlas
        |
        v
  functional locus encoder -----> training-only mean proxy
        |
        | query
        v
RNA --> program tokens --> locus-conditioned retrieval
                              |
                              v
                    [locus ; RNA context]
                              |
                              v
                         beta prediction
```

The public model API is:

```python
from methylation_predictor import (
    SingleRetrievalPredictor,
    IterativeRetrievalPredictor,
)
```

`SingleRetrievalPredictor` is the current reference configuration.
`IterativeRetrievalPredictor` is the matched deeper-retrieval candidate.

See [`docs/MODEL.md`](docs/MODEL.md) for the exact equations and architecture.

## Repository status

This branch is being reduced from the original research-history repository into
the paper reproducibility repository. The core model, internal baselines, and
retained RNA-encoder comparators now all use the paper-facing functional-locus
path. The retired FeatureFusion/shared-backbone model family has been removed;
remaining cleanup is limited to transitional trainer/config/data plumbing.

The historical experiment names remain available through Git history and the
original research branch, not as the primary public interface.

## Installation

```bash
conda activate methyl-predictor
python -m pip install -r requirements.txt
python -m pip install -r requirements-genomics.txt
python -m pip install -e .
```

## Reference configuration

The standalone paper-facing recipe is:

```text
configs/models/main.yaml
```

It is regression-tested to resolve exactly to the current J0 reference recipe
during the migration.

The reference architecture uses:

- 4,165 sparse regulatory tracks;
- 23 dense locus annotations;
- 256-dimensional functional locus representations;
- 64 learned RNA program tokens of width 256;
- 4-head locus-to-RNA cross-attention;
- a training-only mean-proxy head;
- a 512 -> 256 -> 128 -> 1 beta regressor with dropout 0.15.

The reference objective is:

```text
beta MSE + 0.15 * locus-PCC loss + 0.15 * mean-proxy loss
```

## Data inputs

RNA training consumes the canonical TCGA data preparation together with:

```text
--functional-atlas   sparse ENCODE regulatory-track atlas
--annotation-cache   static/breadth CpG annotation cache
--rna-cache          normalized RNA cache
--cpg-targets-dir    training-locus mean targets for the auxiliary mean proxy
```

RNA training no longer loads the historical 1,536-D genomic/FM feature cache.
`--prior-cache` is optional during training and is used only to report
`prior_mse` / `skill_vs_prior`; it contains just `cpg_idx.npy` and `prior.npy`.
A legacy feature-cache directory is accepted as a prior-cache location because
those two files are present there, but its embedding and sigma arrays are never
opened by the RNA workflow.

The sparse functional input is provided by:

```python
methylation_predictor.storage.FunctionalLocusCache
```

which preserves CSR-style track lookup and does not densify the full
locus-by-track matrix.

## Train

Example for the exact matched chr1 protocol:

```bash
python scripts/train.py \
  --model rna_methylation \
  --scope chr1 \
  --recipe configs/models/main.yaml \
  --mode final \
  --canonical-root "$TCGA_CANONICAL_ROOT" \
  --registry "$REGISTRY" \
  --rna-cache "$RNA_CACHE" \
  --prior-cache "$PRIOR_CACHE" \
  --cpg-targets-dir "$CPG_TARGETS" \
  --functional-atlas "$FUNCTIONAL_ATLAS" \
  --annotation-cache "$ANNOTATION_CACHE" \
  --prepared-root "$MATCHED_CHR1_ROOT" \
  --output-root "$RUN_ROOT" \
  --run-id functional-reference-seed17
```

`--prior-cache` is metric-only. During training it may be omitted if
prior-relative development metrics are not needed; official evaluation uses it
to reproduce `prior_mse` and `skill_vs_prior`.

## Evaluate

```bash
python scripts/evaluate.py \
  --model rna_methylation \
  --checkpoint "$CHECKPOINT" \
  --recipe configs/models/main.yaml \
  --eval-scope chr1 \
  --canonical-root "$TCGA_CANONICAL_ROOT" \
  --registry "$REGISTRY" \
  --rna-cache "$RNA_CACHE" \
  --prior-cache "$PRIOR_CACHE" \
  --cpg-targets-dir "$CPG_TARGETS" \
  --functional-atlas "$FUNCTIONAL_ATLAS" \
  --annotation-cache "$ANNOTATION_CACHE" \
  --prepared-root "$MATCHED_CHR1_ROOT" \
  --output "$OUTPUT_JSON"
```

RNA evaluation reports the official sample/CpG generalization views, including
the double-OOD validation-CpG x validation-sample setting.

## Paper-facing layout

```text
src/methylation_predictor/
    modeling/
        reference.py
        retrieval.py
        rna.py
    storage.py
    rna_training/
        rna_methylation_trainer.py

configs/models/
    main.yaml
    functional_fusion/
        j0_final.yaml
        j1_iterative.yaml

docs/
    MODEL.md
    REPOSITORY_SCOPE.md
    REFACTORING_CONTRACT.md
```

Benchmark-specific code is kept isolated under
`methylation_predictor/benchmark/` and `scripts/benchmark_*`.

## Reproducibility refactor

The refactor contract is documented in
[`docs/REFACTORING_CONTRACT.md`](docs/REFACTORING_CONTRACT.md).  The intended
final repository boundary is documented in
[`docs/REPOSITORY_SCOPE.md`](docs/REPOSITORY_SCOPE.md).

Current priorities after the functional/core cutover are:

1. simplify historical trainer/CLI naming and remove redundant functional-mode flags;
2. freeze the active architecture-experiment surface;
3. consolidate data preparation and paper-table reproduction commands;
4. perform the final repository and reproducibility audit.
