# Stage T — execution record

## 2026-07-30: RNA-to-gene manifest

The Stage T manifest was built from the production RNA matrix and GENCODE v41:

```bash
PYTHONPATH=src conda run -n methil-predictor python -m \
  methylation_predictor.rna_branch.prepare_gene_manifest \
  --rna artifacts/rna_branch/2026-07-29_first_benchmark/inputs/tcga_rna.h5 \
  --values-key X --row-ids-key sample_idx --gene-ids-key gene_ids \
  --gtf artifacts/genomic_encoder/static_prior/reference/gencode.v41.annotation.gtf.gz \
  --output artifacts/rna_branch/stage_t_gene_tokens/inputs/gene_tss_manifest.parquet
```

Results:

- RNA rows: 21,792; matched: 21,792; unmatched: 0; coverage: 1.0.
- Strand counts: `+` 10,983; `-` 10,809.
- RNA SHA-256: `e51e86829df30aa2e67051e120c15fc7f24e2c949ac258b6cd933fd5ceae6ce5`.
- GTF SHA-256: `b7c4e40968a74f1e2237995dc0d5eb22658f968440e76ae25e202b1355fb71d9`.
- Manifest SHA-256: `dbdfc122b7811bf67c0d9198297c78a8c6ae0916091874630075241593236a5f`.

The RNA labels use `SYMBOL;ENSEMBL_ID.version`.  The manifest builder now
parses each semicolon-delimited field and matches exact Ensembl ID,
version-stripped Ensembl ID, then unique gene symbol.  A regression test covers
this contract.

## Extraction inputs and current gate

The complete hg38 source was converted from
`/data/dataset/methylation/MethylProphetData/parquet/grch38_hg38/hg38.fa.parquet`
to `artifacts/reference/hg38.fa.gz`.  It contains 455 contigs, including every
chromosome required by the manifest, and has SHA-256
`6a9043a5d2914fb451cc7d594f33744db6aafe6f4b285dffb09f4553cf89dfe1`.

The public `InstaDeepAI/NTv3_650M_post` checkpoint was downloaded to
`artifacts/models/NTv3_650M_post/`; `model.safetensors` has SHA-256
`1015e48bf1aeb5c131ea9ecb0ae34989efd315ab3d0fb8bbbc28689fdd751cf8`.
Its configuration and tokenizer resolve offline with `trust_remote_code=True`.

The 32 GPU shards completed and merged into the RNA-aligned `[21792, 1536]`
embedding matrix.  All rows are finite and non-zero; exact RNA gene-ID
alignment passed.

## First tranche T0--T3

The LR screen selected `2e-5` for T1, `3e-4` for T2, and `2e-5` for T3,
using `best_validation_mse` only.  T0 reproduced the Stage F F2 seed-17
reference within the predeclared `1e-6` tolerance: `0.025541976699461448`
previously versus `0.025541976713567758` here (difference
`1.4106309870198785e-11`).

| Run | Best validation MSE | Double-OOD MSE | Patient dynamic Pearson | Within-cancer skill |
| --- | ---: | ---: | ---: | ---: |
| T0 F2 linear | 0.025541976713567758 | 0.024127573122812905 | 0.2944914415623836 | 0.14813751351429238 |
| T1 learned tokens | 0.030273728280127206 | 0.027378176089761497 | 0.02993097774083484 | 0.000002437883715677991 |
| T2 aligned NTv3 tokens | 0.029986932496535078 | 0.028089929911366354 | 0.09777084801106448 | 0.006224881363351975 |
| T3 permuted NTv3 tokens | 0.03039938832414518 | 0.02737070576879231 | 0.05727284789571649 | -0.0009059154821906557 |

T2 improves validation MSE versus T1 by `0.0002867957835921281` and versus
T3 by `0.00041245582761010105`, with corresponding patient-wise and
within-cancer improvements over both controls.  It does not beat T0, so
gene-token representation itself is not competitive with the F2 baseline in
this tranche.  Stop at this point: the confirm and locus-query grids have not
been run.
