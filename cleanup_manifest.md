# RNA branch cleanup manifest

## Conservare e versionare

- `src/methylation_predictor/rna_branch/`, i test `tests/test_rna_branch_*.py`
  e il runner D1 `stage_d1.py`/`stage_d_matched.py`;
- configurazioni base, C0-V0 finale e una griglia rappresentativa per signal,
  PCA, random projection, bottleneck e interaction;
- `docs/rna_branch_*.md`, `docs/stage_d1_report.md` e i risultati compatti in
  `results/rna_branch/`;
- manifest, validation, summary CSV/JSON e intervalli bootstrap compatti.

## Spostare/organizzare

- lo script SLURM RNA in `scripts/rna_branch/run_rna_grid.sh`;
- summary Stage A/B/C/D1 da `artifacts/` a `results/rna_branch/`;
- la configurazione generata selezionata C0-V0 seed 17 a
  `configs/rna_branch/c0_v0_final.yaml`.

## Ignorare ma conservare localmente

- input canonici HDF5, embedding, cache downloaded e artifact originali del
  genomic encoder; sono necessari per rieseguire ma non sono versionabili qui;
- checkpoint, predizioni dense, feature PCA/random projection, manifest di run
  completi e log sotto `artifacts/`.

## Eliminare localmente

- directory di singole run RNA, checkpoint `.pt`, predizioni `.npz`, config
  generate automaticamente, log e output intermedi Stage A/B/C/D1;
- la copia di lavoro `MethylProphetTest-rna-experiments/`, duplicata rispetto al
  codice ora consolidato nel repository.

## Salvaguardie

Non vengono eliminati `third_party/MethylProphet`, gli artifact originali sotto
`artifacts/genomic_encoder/`, né gli input canonici RNA/beta/embedding. Prima
della rimozione, i report e i CSV/JSON piccoli sono copiati in
`results/rna_branch/` e i loro input/provenance restano documentati nei report.
