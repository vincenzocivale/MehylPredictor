# Repository guidance

## Functional RNA trainer (historical main workflow)

MethylPredictor predicts patient-specific CpG methylation from bulk RNA using
a reference functional representation of each CpG. This functional model does not
consume a genomic-FM embedding; the `perceiver_functional_latent` variant
(`--genomic-fm-cache`, frozen Enformer trunk latent) is the explicit exception.
Perceiver variants live in `modeling/perceiver.py`, `modeling/cpg_query.py` and
`modeling/deep_decoder.py` (deeper post-query decoder, `perceiver_deep_decoder`).

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

See `docs/MODEL.md` and `docs/WORKFLOWS.md`.

## Current AlphaGenome research handoff

Read [research state](docs/ALPHAGENOME_RESEARCH_STATE_2026-10-18.md) and
[reproducibility](docs/ALPHAGENOME_REPRODUCIBILITY_2026-10-18.md) before continuing
the AlphaGenome campaign. Its latest mixed-source model is a separate workflow
from the functional trainer above and consumes native 1Mb AlphaGenome features.
Never substitute the dimensionally identical old 512kb cache. Use canonical
released Array patient axes; the old derived split differs by 723 validation
patients. Report Array-held and all-source-unseen CpGs separately (6,742 vs 369).
EPIC/WGBS are training sources for the mixed model. Preserve frozen protocols,
source snapshots and negative results; do not tune on official benchmark errors.
