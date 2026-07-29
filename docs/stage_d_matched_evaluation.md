# Stage D — confronto matched e bootstrap appaiato

Questo stadio non riusa le metriche MethylProphet precedenti. Quelle righe
rilasciate rappresentano split diversi dal double-OOD chr1 del ramo RNA e
quindi non sono direttamente confrontabili con C0.

Il comando seguente accetta una tabella Parquet long per modello con colonne
`sample_idx`, `cpg_idx`, `cancer_type`, `target`, `prediction`. Prima di
calcolare qualsiasi metrica, verifica identità di pazienti, CpG, cancer type,
target osservati e missing-value mask; rifiuta predizioni mancanti sulle righe
osservate. Il manifest di blocchi deve fornire `cpg_idx,genomic_block`.

```bash
PYTHONPATH=src conda run -n methil_classif python -m methylation_predictor.diagnostics.methylprophet.stage_d_matched evaluate \
  --model ntv3_prior=artifacts/stage_d/ntv3_prior.parquet \
  --model c0_optimized=artifacts/stage_d/c0_optimized.parquet \
  --model methylprophet=artifacts/stage_d/methylprophet.parquet \
  --model methylprophet_global_calibrated=artifacts/stage_d/methylprophet_global_calibrated.parquet \
  --prior-model ntv3_prior --reference-model c0_optimized \
  --blocks artifacts/stage_d/cpg_blocks.parquet \
  --output artifacts/stage_d/matched_bootstrap.json --replicates 2000
```

`delta_mse` è candidato meno riferimento (negativo favorisce il candidato);
`delta_skill` e `delta_within_cancer_skill` sono candidato meno riferimento
(positivo favorisce il candidato). Sono riportati intervalli percentile 95%
appaiati per pazienti, blocchi genomici e bootstrap gerarchico pazienti+blocchi.

Per calibrare MethylProphet non si deve stimare alpha sul double-OOD. Si usa un
file di validation con la colonna `static_prediction` (o un altro anchor
esplicito):

```bash
PYTHONPATH=src conda run -n methil_classif python -m methylation_predictor.diagnostics.methylprophet.stage_d_matched calibrate-global \
  --validation artifacts/stage_d/mp_validation.parquet \
  --output artifacts/stage_d/methylprophet_global_calibrated.parquet
```

Il comando salva alpha e l'origine validation nel JSON adiacente. La variante
opzionale “prior statico MethylProphet + dinamica calibrata” è un ulteriore
file `--model`: deve essere costruito con lo stesso pannello e supererà gli
stessi controlli.

Per esportare C0 o il prior NTv3 dai file `predictions_double_ood.npz` già
salvati dal ramo RNA:

```bash
PYTHONPATH=src conda run -n methil_classif python -m methylation_predictor.diagnostics.methylprophet.stage_d_matched export-rna-npz \
  --input artifacts/rna_branch/stage_c/screening/c0_v0_seed17/predictions_double_ood.npz \
  --output artifacts/stage_d/c0_optimized.parquet
```

Per esportare il baseline NTv3 dallo stesso NPZ, aggiungere
`--prediction-key prior`; il runner controlla inoltre che sia statico per CpG.

Quando sono disponibili le predizioni rilasciate dello stesso checkpoint, il
seguente adattatore estrae il solo pannello C0 **solo dopo** aver verificato i
target riga per riga. È preferibile a una nuova inferenza perché conserva le
predizioni del checkpoint rilasciate come evidenza primaria.

```bash
PYTHONPATH=src conda run -n methil_classif python -m methylation_predictor.diagnostics.methylprophet.stage_d_matched extract-checkpoint-predictions \
  --released artifacts/cache/methylprophet/upstream_outputs/eval/eval-tcga_mix_chr1-bs_512-c2b2/eval_results-test.parquet \
  --panel-npz artifacts/rna_branch/stage_c/screening/c0_v0_seed17/predictions_double_ood.npz \
  --output artifacts/stage_d/methylprophet_original.parquet
```
