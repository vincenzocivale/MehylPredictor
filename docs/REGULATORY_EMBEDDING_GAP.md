# Regulatory embedding quality investigation

Status: in progress. No claim that the NTv3 POST gap has been closed.

## Evidence from the official runs

All rows below are seed 17, 80 epochs, chr1 official evaluation. These existing
test results motivated the investigation; new model selection uses development
data only.

| Locus representation | Seen-CpG/new-sample MAS | New-CpG/new-sample MAS | New-CpG/new-sample MSE |
|---|---:|---:|---:|
| NTv3 PRE | 0.482431 | 0.319165 | 0.049329 |
| PRE + functional | 0.609001 | 0.465432 | 0.021359 |
| Functional only (64-dimensional bottleneck) | 0.665625 | 0.457802 | 0.019424 |
| NTv3 POST | 0.700581 | 0.584740 | 0.014201 |

Authoritative run roots under `/dune/DATASETS/MethylPredictionData/experiments/`:

- `runs/runs/locus_cls_joint/chr1/chr1_locus_attention_ntv3_pre_20260909b`
- `runs/locus_cls_joint/chr1/ntv3-pre-functional-all-fresh-seed17-v2`
- `runs/locus_cls_joint/chr1/functional-only-fresh-seed17`
- `runs/locus_cls_joint/chr1/ntv3-post-fresh-seed17`

Use `evaluation/chr1/metrics.json` for each. `skill_vs_prior` is not directly
comparable because its reference prior differs between the runs.

## Controlled representation diagnostic

`scripts/experiments/audit_regulatory_embedding.py` uses the existing seed-17
5-Mb-block inner split: 29,913 training CpGs and 3,972 validation CpGs. It takes
512 samples from the inner training sample pool, builds a rank-32 methylation
response basis using training CpGs only, and fits ridge predictors of locus
mean and response coefficients. Scaling is fitted on inner training loci.
No official held-out labels enter fitting or scoring.

This tests **unseen loci in seen samples**. It is neither the full RNA model
nor the official/double-OOD benchmark. A low linear-probe score measures
linear accessibility, not a proof of irreversible information loss. F0 was
already selected on this development split, so this remains exploratory.

At the shared ridge alpha of 1,000:

| Representation | Median response correlation | Locus-mean R² |
|---|---:|---:|
| NTv3 POST | 0.526459 | 0.903466 |
| Raw binary tracks + existing dense annotations | 0.474525 | 0.847030 |
| Tracks divided by active-track count + same annotations | 0.427227 | 0.801237 |
| Raw binary tracks + expanded annotations | 0.475885 | 0.847151 |
| Learned F0 static embedding | 0.444371 | 0.836718 |

Full results and axis hashes:
`results/diagnostics/regulatory_embedding_20260910/report.json`.
The low-rank reconstruction using true validation coefficients scores 0.763226;
it is an oracle diagnostic, not a usable prediction method.

The count-normalized representation performs worse than the raw representation
under this matched probe. Expanded annotations alone make little difference.
The remaining raw-track versus POST gap supports investigating missing context
and quantitative signal, in addition to improving the learned projection.

Follow-up probes at the same alpha, axes and target basis:

| Representation | Median response correlation |
|---|---:|
| Raw + CpG-sampled neighborhood context | 0.477026 |
| Raw + 32-kb 5-mer frequencies | 0.463956 |
| Raw + context + 5-mer frequencies | 0.465990 |
| Standardized raw features, random projection to 256D | 0.423029 |
| Standardized raw features, train-fitted PCA to 256D | 0.472929 |
| Standardized raw features, train-fitted PCA to 1536D | 0.483970 |

The context shortcut adds little and 5-mer counts do not help this probe;
neither result tests true quantitative signal profiles or sequence motifs.
Covariance-based compression preserves substantially more linearly accessible
response information than a random projection of the same dimension.
See `results/diagnostics/regulatory_context_20260910/report.json` and
`results/diagnostics/regulatory_compression_20260910/report.json`.

## Implemented candidate: F7

`StandardizedTrackEmbedding` replaces the track pooling in F0 with a sparse
linear projection of centered, variance-scaled binary features. The variance
floor is 0.001; constant training tracks are suppressed. Absolute track-count
information is preserved. Initial weights use standard deviation
`1 / sqrt(n_tracks)` instead of the default unit-variance EmbeddingBag weights.

Frequency estimates are fitted on the actual Array training pool after the
development split. They are persistent checkpoint buffers. Checkpoint resume
and evaluation restore those buffers instead of fitting on validation data.
The RNA encoder, 256-dimensional locus width, fusion, prediction head, loss,
batching, schedule, seed and early-stopping policy match F0.

This is an input-calibration/projection ablation: both normalization and
initialization scale change. It does not isolate those two effects separately.
It uses no NTv3 embeddings or teacher targets in the prediction path.

Recipe: `configs/models/functional_fusion/f7.yaml`.
Training run: `artifacts/regulatory_embedding/runs/locus_cls_joint/chr1/regulatory-f7-standardized-dev-seed17-e30`.
The epoch-1 sampling-plan SHA256 matches F0 exactly:
`5b2b9f9a235e36e533dc17e8b6506f00546d3670ba31145dcbaf3c20c5ae4813`.

Validation: 32 tests passed in `test_regulatory_embedding.py` and
`test_functional_locus.py`, including sparse/dense projection and gradient
equivalence, training-only fitting, checkpoint restoration, empty/constant
tracks, and absence of NTv3 dependency.

## Covariance initialization candidate: F8

`scripts/prepare_regulatory_projection.py` fits a whitened PCA projection of
standardized binary tracks, using only the 29,913 inner-training Array CpGs.
It uses no NTv3 targets and no methylation labels. The 256-component initializer
explains 0.586657 of the variance of these scaled tracks. It is saved in
`artifacts/regulatory_embedding/projections/chr1_dev_pca256.pt`, with a JSON
manifest recording the training-axis and file hashes.

F8 uses the same model implementation as F7, with the projection weights and
frequency buffers initialized by this artifact. The loader verifies the exact
training-CpG set, atlas path and current training frequencies; a final-mode or
different-split initializer is rejected. All weights then remain trainable.
Inference restores checkpoint state and does not require the initializer file.

Recipe: `configs/models/functional_fusion/f8.yaml`.
F8 completed on the development split: best inner double-OOD MAS-PCC
`0.473849` at epoch 3, after 11 epochs. F7 reached `0.473467`, so the delta
(`+0.000382`) is exploratory and not evidence of a meaningful improvement.

Expected F8 run directory:
`artifacts/regulatory_embedding/runs/locus_cls_joint/chr1/regulatory-f8-dev-seed17-e30`.

## BigWig pilot

Seven public GRCh38 BigWig tracks have been downloaded and extracted at all
2,004,436 atlas CpGs (35 local-window summaries). The corrected cache uses an
explicit inner-training-only mask (29,913 CpGs), chromosome-aware windows,
1-based-to-0-based coordinate conversion, and stores per-track robust scalers
and provenance in `artifacts/bigwig_pilot/cache_tiny_corrected`.
The cache still needs semantic review of ENCODE output types and a model
adapter before any BigWig training result can be reported.

## Remaining work

- Compare F7/F8 across seeds and against the controlled F0 reference
  and the NTv3 development reference, using the same views and protocol.
- Select a scientifically consistent BigWig track manifest, add its dense
  feature adapter, and run the matched baseline/BigWig comparison.
- Refine the representation according to measured results, confirm across
  seeds, then select one model before final-protocol training/evaluation.
- Closing the objective requires demonstrated predictive quality, not just
  a passing implementation or an improved auxiliary probe.
