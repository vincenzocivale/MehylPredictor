# Architecture ablations on exact Array-chr1

Architecture selection is intentionally separated from the final mixed-source
MethylProphet comparison.

All variants are first trained on the exact released `tcga_array_chr1` protocol:

- 8,260 official Array train samples / 918 official held-out samples;
- 33,885 official Array train CpGs / 6,742 official held-out CpGs;
- all 25,017 canonical RNA genes;
- frozen NTv3 CpG embeddings from
  `datasets/methylprophet_repro_v1/cpg/ntv3/ntv3_cpg_atlas_v1.h5`;
- the exact historical Array prior and variability features from
  `locus_features.parquet`.

This stage does **not** use EPIC/WGBS. That is deliberate: architecture
decisions should be cheap and controlled. Only the selected architecture is
promoted to the expensive Array+EPIC+WGBS training used for the final
MethylProphet comparison.

## Variants

| variant | intended change |
|---|---|
| `baseline` | canonical `RNA2DNAmModel` |
| `no_gate` | replace the variability gate with identity |
| `no_anchor` | remove subtraction of the mean-RNA interaction |
| `no_product` | remove projected RNA x CpG product, retain concatenation |
| `rna256` | enlarge the linear RNA latent from 64 to 256 |
| `direct_prediction` | direct methylation-logit regression with residual-specific prior/gate/anchor mechanisms disabled |

`direct_prediction` is a model-family control rather than a one-factor ablation:
once the frozen prior is removed, the residual gate, mean-RNA anchor and
zero-residual initialization no longer have the semantics they have in the
canonical residual model. `shrinkage_weight` is therefore also disabled for
that variant.

## Training protocol

Every variant uses exactly the same seed, nested development split, optimizer,
loss, and deterministic `full_coverage` schedule. The standard trainer derives
the required number of steps per epoch and verifies 100% sample/CpG coverage.

Checkpoint selection uses `mas_pcc` by default and evaluates the complete
nested-development CpG panel (`validation_max_cpgs: null`) to avoid ranking
architectures on a noisy 1,024-CpG subset.

After development selects `best_epoch`, the model is reinitialized and refit for
exactly that number of epochs on all official Array training samples and CpGs.
Only after this fixed refit are the three official Array evaluation views
opened.

## Launch

Required resources:

```text
/raid/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1/
/raid/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1/cpg/ntv3/ntv3_cpg_atlas_v1.h5
/raid/DATASETS/MethylPredictionData/locus_features.parquet
```

At launch, a shared ~125 MB fp16 HDF5 cache containing only the exact 40,627
Array-chr1 CpG embeddings is extracted from the consolidated 5.7M-row atlas.
This is a pure representation cache and is reused by every variant; it avoids
constructing a multi-million-entry Python ID mapping six times.

Run on two GPUs:

```bash
GPUS=0,1 \
  nohup bash scripts/run_architecture_ablations.sh \
  > launcher.log 2>&1 &
```

To run a subset:

```bash
GPUS=0,1 VARIANTS=baseline,no_gate,no_anchor \
  bash scripts/run_architecture_ablations.sh
```

Outputs:

```text
.../architecture_ablations_array_chr1/seed17/
  base_configs/<variant>.yaml
  logs/<variant>.log
  runs/<variant>/development/
  runs/<variant>/final_refit/
  runs/<variant>/evaluation/headline.json
  summary.tsv
```

Rank variants primarily by double-heldout `val_cpg_x_val_sample` MAS-PCC.
Use MSE and skill-vs-prior as guardrails. The winner is then promoted to
`tcga_mix_chr1` (Array+EPIC+WGBS) for the final data-matched comparison with
the released MethylProphet checkpoint.
