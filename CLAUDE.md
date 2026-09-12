# Repository guidance

## Current RNA model

MethylPredictor predicts patient-specific CpG methylation from bulk RNA using
a reference functional representation of each CpG. The RNA model does not
consume an NTv3/genomic-FM embedding at runtime.

Current implementation:

```text
src/methylation_predictor/modeling/
src/methylation_predictor/rna_training/rna_methylation_trainer.py
configs/models/main.yaml
```

Public API:

```python
from methylation_predictor.rna_training import (
    RNAMethylationTrainer,
    evaluate_rna_checkpoint,
)
```

RNA runs require `--functional-atlas`, `--annotation-cache`, `--rna-cache`, and
`--cpg-targets-dir`. `--prior-cache` is metric-only.

Do not reintroduce `--engine`, `--functional-only`, RNA `--feature-cache`,
FeatureFusion classes, or the retired shared-backbone model family.

All live architecture selectors belong in
`methylation_predictor.modeling.factory`; do not put J-series dispatch logic
back into the trainer.

Existing run/checkpoint storage intentionally retains the historical internal
identifier `locus_cls_joint`. Do not rename existing runs while active
experiments may need resume/evaluation.

`rna_training/locus_cls_trainer.py` is only a compatibility shim. New trainer
implementation work belongs in `rna_training/rna_methylation_trainer.py`.

After structural changes run:

```bash
python -m compileall -q src scripts
pytest -q
```

See `docs/MODEL.md`, `docs/WORKFLOWS.md`, and `docs/REPOSITORY_SCOPE.md`.
