# Workflows

The repository exposes two trainable model families:

- `rna_methylation`: patient-specific CpG methylation prediction from bulk RNA
  plus a frozen reference functional representation of each CpG;
- `cpg_statistics`: auxiliary/static CpG-statistics workflow.

Experiment history is not part of the public API.

## RNA methylation

The paper-facing model is described in [`MODEL.md`](MODEL.md):

```text
functional CpG atlas + static annotations -> locus representation h_c
patient RNA -> program tokens
h_c queries program tokens -> patient/locus RNA context
[h_c ; RNA context] -> beta prediction
```

The runtime is `RNAMethylationTrainer`. There is no selectable engine and no
`functional_only` switch. Functional atlas and annotation caches are mandatory.
The genomic/FM feature cache is not a model input.

## Training

```bash
python scripts/train.py   --model rna_methylation   --scope chr1   --recipe configs/models/main.yaml   --mode final   --canonical-root "$TCGA_CANONICAL_ROOT"   --prepared-root "$MATCHED_CHR1_ROOT"   --registry "$REGISTRY"   --rna-cache "$RNA_CACHE"   --prior-cache "$PRIOR_CACHE"   --cpg-targets-dir "$CPG_TARGETS"   --functional-atlas "$FUNCTIONAL_ATLAS"   --annotation-cache "$ANNOTATION_CACHE"   --output-root "$RUN_ROOT"   --run-id reference-seed17
```

`--prior-cache` is metric-only. `--cpg-targets-dir` supplies `target_mu.npy`
for the training-only mean-proxy objective.

## Evaluation

```bash
python scripts/evaluate.py   --model rna_methylation   --checkpoint "$CHECKPOINT"   --recipe configs/models/main.yaml   --eval-scope chr1   --canonical-root "$TCGA_CANONICAL_ROOT"   --prepared-root "$MATCHED_CHR1_ROOT"   --registry "$REGISTRY"   --rna-cache "$RNA_CACHE"   --prior-cache "$PRIOR_CACHE"   --cpg-targets-dir "$CPG_TARGETS"   --functional-atlas "$FUNCTIONAL_ATLAS"   --annotation-cache "$ANNOTATION_CACHE"   --output "$OUTPUT_JSON"
```

The three official views are train-CpG x validation-sample,
validation-CpG x train-sample, and double-OOD validation-CpG x
validation-sample.

## Run storage

Existing runs remain under the historical compatibility key:

```text
<output-root>/runs/locus_cls_joint/<scope>/<run-id>/
```

This string is retained for resume/evaluation compatibility; it is not the
public trainer name.

New code should import:

```python
from methylation_predictor.rna_training import (
    RNAMethylationTrainer,
    evaluate_rna_checkpoint,
)
```
