# MethylProphetTest — training-only

Minimal repository for training the final RNA-to-DNAm model.

The historical research code (MethylProphet diagnostics, baselines, ablations,
RNA-encoder experiments, biological-fidelity evaluation, bootstrap analyses,
paper reports and generated results) has intentionally been removed. Git
history is the archive.

## Canonical model — `RNA2DNAmModel`

For patient `s` and CpG `i`:

```text
RNA_s
  -> LayerNorm
  -> Linear(21792 -> 64)
  -> z_s

CpG embedding_i (frozen) + variability_i
  -> variability gate g_i

[z_s, e_i, P_rna(z_s) * P_locus(e_i)]
  -> LayerNorm
  -> Linear(... -> 128)
  -> GELU
  -> Dropout(0.1)
  -> Linear(128 -> 1)
  -> raw residual
```

With mean-RNA anchoring:

```text
delta_si = g_i * (
    interaction(RNA_s, CpG_i)
    - interaction(mean_RNA, CpG_i)
)

beta_hat_si = sigmoid(logit(prior_i) + delta_si)
```

The residual head is zero-initialized so training starts exactly from the
frozen CpG prior, in logit space.

## Training

Install the canonical training dependencies and package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the complete training protocol:

```bash
bash scripts/train.sh
```

The runner performs only:

1. data preflight;
2. nested development split creation;
3. development training;
4. best-epoch selection;
5. final refit.

It deliberately does **not** run test evaluation, MethylProphet comparisons,
ablations, bootstraps or report generation.

The canonical config is:

```text
configs/train.yaml
```

`scripts/render_final_refit_config.py` derives the final-refit config from
`configs/train.yaml` plus the development stage's measured `best_epoch`; the
result is written under the run's `artifacts/` directory, not tracked in
`configs/`.

The CLI can also be invoked directly:

```bash
python -m methylation_predictor train --config configs/train.yaml
python -m methylation_predictor validate --config configs/train.yaml
```

Generated checkpoints and runtime artifacts belong under `artifacts/`, which
must remain untracked.
