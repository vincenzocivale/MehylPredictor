# Stato della ricerca AlphaGenome

Il [dossier aggiornato al 18 ottobre 2026](docs/ALPHAGENOME_RESEARCH_STATE_2026-10-18.md) raccoglie risultati, ipotesi validate e scartate, limiti e prossimi controlli. Il [runbook](docs/ALPHAGENOME_REPRODUCIBILITY_2026-10-18.md) documenta checkpoint, contratto di inferenza, ambiente e riproducibilità del confronto mixed-source.

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
path. The retired FeatureFusion/shared-backbone family is no longer part of the
paper-facing API. Some compatibility trainer/config/data plumbing remains temporarily so
active architecture-search checkpoints can still resume and be evaluated.

The historical experiment names remain available through Git history and the
original research branch, not as the primary public interface.

## Installation

Research notes: [RNA, interaction, and training screening (2026-10-17)](docs/EXPERIMENTS_RNA_TRAINING_2026-10-17.md)
records the completed chr1 experiments, matched baselines, limitations, and
candidate follow-up tests; these results do not change the reference recipe.
The [second screening protocol](docs/EXPERIMENTS_RNA_TRAINING_ROUND2_2026-10-17.md)
covers mean/residual prediction, RNA compression/normalization, and residual losses.
The [CpG error audit](docs/CPG_ERROR_AUDIT_2026-10-17.md) stratifies errors by
genomic context and methylation variability, and examines adaptive inference.
The [factorized RNA–CpG decoder screening](docs/FACTORIZED_DECODER_SCREENING_2026-10-17.md)
tests whether CpG annotations can select RNA response factors while reducing
pairwise computation.
The [TCGA chr1 AlphaGenome v1 design](docs/TCGA_CHR1_ALPHAGENOME_V1_2026-10-17.md)
records verified embedding coverage, leakage-controlled development, and the
criteria for freezing the first Array-only candidate.

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
--locus-store        canonical genome-wide 4,165+23 CpG feature store
--functional-atlas / --annotation-cache  legacy frozen fallback
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
New paper-facing runs open the mmap-backed `derived/locus_features_v1` store;
the frozen chr1 CSR pair remains available only for compatibility and regression.

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
  --locus-store "$METHYL_DATA_ROOT/derived/locus_features_v1" \
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
  --locus-store "$METHYL_DATA_ROOT/derived/locus_features_v1" \
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
    WORKFLOWS.md
    DATA.md
```

## AlphaGenome chr1 v2

The versioned AlphaGenome model is `AlphaGenomeLocalPredictor`, constructed by
`modeling.factory.build_alphagenome_predictor`. It combines static AlphaGenome
and functional CpG features with RNA through a rank-32 decoder. Development
compares a shallow mean head, local response regularization and population-mean
calibration using training-only priors.

`scripts/train_alphagenome_v2.py` selects a recipe on internal genomic folds,
refits it on the official training axes and exports per-CpG error audits.
The target-free inference adapter is `AlphaGenomePredictorFromCheckpoint` in
`modeling.alphagenome`; it enforces the saved gene order and preprocessing.
See [training results and CpG errors](docs/TCGA_CHR1_ALPHAGENOME_V2_2026-10-18.md)
for the frozen recipe, checkpoint location, validation limits and error tables.

## AlphaGenome architecture search and hierarchical v3

The follow-up compares two-stage learning, explicit response retrieval and a
hierarchical mean. The selected `alphagenome_hierarchical_chr1_v3` keeps the v2
encoders, calibrates unseen loci toward a genomic mean prior (weight 0.25), and
fits training-locus intercepts using training patients only. It retains one RNA
encoder and a rank-32 decoder. Pass global registry `cpg_ids=` to
`AlphaGenomePredictorFromCheckpoint`; feature-matrix row numbers and Illumina
probe strings are not registry identifiers.

The primary checkpoint of the earlier Array-only 512kb campaign is
`artifacts/alphagenome_hierarchical_2026-10-18/official/seed17/checkpoint.pt`.
[Architecture comparisons](docs/ALPHAGENOME_ARCHITECTURE_SEARCH_2026-10-18.md)
and [hierarchical v3 results](docs/ALPHAGENOME_HIERARCHICAL_2026-10-18.md)
include protocols, three-seed results and complete CpG error tables. Official
MSE improves for both observed and unseen loci, but unseen-locus MAE and
within-tissue correlation do not improve over v1. This is a chr1 research
candidate; the historical functional trainer remains a separate input contract.

Reproduce with `scripts/search_alphagenome_architecture.py`,
`scripts/search_alphagenome_hierarchical.py`, and
`scripts/refit_alphagenome_hierarchical.py`. The refit requires successful
internal and dev guardrails recorded by `scripts/report_alphagenome_hierarchical.py`.
Archived search protocols reject in-place resume after code hashes change;
use a fresh `--out` directory or the archived source snapshots to repeat a search.

## AlphaGenome hypothesis validation

[The validation report](docs/ALPHAGENOME_HYPOTHESES_VALIDATION_2026-10-18.md)
covers RNA permutations within named tissues, calibrated v1 controls, few-shot
new-locus adaptation, persistent-error probe QC, local sequence corrections,
cross-fitted locus risk, and transfer to chr2/chr22, EPIC/WGBS and excluded
tissues. It includes 60 new fits and negative results; none of these experiments
automatically replaces the current model. Few-shot adaptation requires measured
methylation at the query loci in a disjoint support set.

The `scripts/validate_alphagenome_*.py` entry points and
`scripts/audit_alphagenome_hard_cpgs.py` write versioned protocols, checkpoints
and per-CpG results under `artifacts/alphagenome_hypotheses_2026-10-18/`.
Run the frozen hypothesis checks before reliability confirmation, and transfer
before external reliability. The probe audit requires the public HM450 manifest
at `external/HM450.hg38.manifest.tsv.gz` inside the artifact directory; its URL
and SHA256 are recorded in the report. Plotting uses matplotlib through
`scripts/report_alphagenome_hypotheses.py`. Experimental genomic correction
heads are constructed by `modeling.factory.build_alphagenome_residual_head`.

## AlphaGenome controls with matched support labels

[Matched-support validation](docs/ALPHAGENOME_MATCHED_SUPPORT_2026-10-18.md)
compares v1/v3 intercept adaptation, additive residual calibration and mean-only
baselines using identical support patients. It includes disjoint EPIC/WGBS
support/query sets, frozen internal hyperparameter selection, existing matched
feature/head/graph ablations, patient/genomic-block uncertainty, and CpG error
tables. Adaptation helps on external cohorts; the new v3 architecture does not
consistently improve over a similarly calibrated v1. That Array-only campaign
did not compare against published MethylProphet; the mixed-source benchmark
below addresses its native published setting.

Run `scripts/validate_alphagenome_matched_support.py`,
`scripts/validate_alphagenome_attribution.py`,
`scripts/validate_alphagenome_support_uncertainty.py`, and
`scripts/report_alphagenome_matched_support.py` in that order. Versioned
protocols and frozen-model outputs are stored separately under
`artifacts/alphagenome_matched_support_2026-10-18/`.

## Source-matched MethylProphet benchmark

[The mixed-source benchmark](docs/ALPHAGENOME_MIXED_METHYLPROPHET_2026-10-18.md)
refits the AlphaGenome model from scratch on Array+EPIC+WGBS and compares it
with the released MethylProphet predictions on exactly matched patient/CpG
pairs and canonical targets. Its audit corrects a 723-patient discrepancy
between the earlier derived patient split and the actual released split.
Earlier reports retain their original split-specific results.

Across three refits, MSE is 42.1% lower for known CpGs/new patients, 40.2%
lower for new Array CpGs/training patients, and 38.8% lower for new Array
CpGs/new patients. With approximately matched sampled-pair exposure, the last
gain remains 36.9%. Only 369 test CpGs are absent from all three training
sources; those are reported separately. These are chr1 benchmark results.

This experiment uses uniform native 1Mb AlphaGenome embeddings at all loci;
the earlier 512kb Array-only cache is a different feature contract. Source
sampling and training duration are selected internally before official scoring.
The frozen recipe, three-seed checkpoints and ID/target audits are under
`artifacts/alphagenome_mixed_2026-10-18/`. Run
`scripts/train_alphagenome_mixed.py`, then
`scripts/compare_alphagenome_methylprophet.py` and
`scripts/report_alphagenome_mixed.py`.
