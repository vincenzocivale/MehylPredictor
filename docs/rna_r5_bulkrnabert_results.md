# R5.1 — BulkRNABert frozen pretraining

## Decision

**Stop after the frozen-feature tranche; do not run LoRA or fine-tuning.**  The
BulkRNABert concatenation is statistically and practically indistinguishable
from F2 on the three matched seeds, but does not improve it.  The standalone
embedding (`replace`) is substantially worse.  Thus the checkpoint does not
provide demonstrable incremental, decoder-usable information beyond the raw
RNA projection `Wx_s` on this benchmark.

The seed-17 screen used validation MSE only, before inspecting held-out panels:
`replace` was rejected (0.0274592474 versus F2 0.0255419767); `concat` was
advanced as non-dominated within numerical resolution (0.0255425694, a
difference of `5.93e-7`, or 0.0023%).  No test metric was used to make that
advancement decision.  The shuffled embedding is a negative control and was
not eligible for confirmation.

## Frozen representation provenance

- Encoder: official InstaDeep `bulk_rna_bert_gtex_encode`, pretrained on
  GTEx+ENCODE and declared not to include TCGA.
- Source checkout: `artifacts/models/multiomics-open-research`, commit
  `625e8165abd1dfab19dea7b5e9be7c67571aa5b1`.
- Checkpoint SHA-256:
  `dc5ef813371982881ff970bc284af37c512cdf79889bc2bc3c968f6977a362ef`.
- Export: `artifacts/rna_branch/pretrained/bulkrnabert_gtex_encode.h5`, 8,116
  samples × 256 features; `log2(TPM+1)` input; RNA/checkpoint gene overlap
  0.7827615151.  The HDF5 sidecar records these fields.
- Extraction used a bfloat16 PyTorch SDPA/Flash translation of the frozen
  official weights, tokenization and pooling.  The reference JAX path tried to
  materialize dense attention and exceeded the 12-GB GPU memory.  This export
  is therefore not claimed to be bitwise-identical to JAX, although its
  scientific inputs and frozen parameters are preserved.

## Commands and completed runs

```bash
# extract (actual production path)
PYTHONPATH=src conda run --no-capture-output -n methil-predictor \
  python scripts/rna_branch/extract_bulkrnabert_torch.py \
  --config configs/rna_branch/stage_f_base.yaml \
  --official-repo artifacts/models/multiomics-open-research \
  --model-name bulk_rna_bert_gtex_encode --input-scale log2p1 \
  --min-gene-overlap 0.78 \
  --output artifacts/rna_branch/pretrained/bulkrnabert_gtex_encode.h5

# aggregate completed metrics
PYTHONPATH=src conda run --no-capture-output -n methil-predictor \
  python src/methylation_predictor/rna_branch/aggregate_report.py \
  --screening-dir artifacts/rna_branch/r5_bulkrnabert \
  --baseline-dir artifacts/rna_branch/stage_f_fusion/first_tranche \
  --baseline-dir artifacts/rna_branch/stage_f_fusion/confirm \
  --baseline-family f2_concat_product \
  --output artifacts/rna_branch/r5_bulkrnabert/r5_summary.csv
```

Completed runs: `concat` seeds 17/23/41; `replace` seed 17; `shuffled` seed
17.  In `shuffled`, only the frozen embedding was permuted within cancer type;
the raw RNA branch was left intact.  All inputs were standardized from train
rows only; the foundation model was frozen throughout.

## Double-OOD results

| family | seeds | MSE (mean ± SD) | skill vs prior | patient dynamic Pearson (median) | locus dynamic Pearson (median) |
| --- | ---: | ---: | ---: | ---: | ---: |
| F2 raw-RNA baseline | 3 | 0.024149 ± 0.000027 | 0.117958 | 0.290283 | 0.345902 |
| BulkRNABert concat | 3 | 0.024155 ± 0.000165 | 0.117748 | 0.292607 | 0.344479 |
| BulkRNABert replace | 1 | 0.025918 | 0.053343 | 0.212767 | 0.231883 |
| BulkRNABert shuffled | 1 | 0.026828 | 0.020096 | 0.159880 | 0.139092 |

For the matched three-seed `concat − F2` comparison, the double-OOD changes
are: MSE `+0.000006 ± 0.000150` (paired t-test descriptive `p=0.9533`), skill
`−0.000209`, patient correlation `+0.002324`, and locus correlation
`−0.001422`.  The small patient-correlation increase is not accompanied by a
MSE or skill benefit and has a small locus-correlation cost.

The shuffled control is markedly worse than both `concat` and F2, confirming
that the frozen vectors carry patient-specific signal rather than merely
cancer-type or dimensionality effects.  It does not, however, change the main
conclusion: that signal is already effectively available to the supervised
raw-RNA F2 branch in the frozen-concat design.

Complete per-panel, full-precision exports are
`artifacts/rna_branch/r5_bulkrnabert/r5_summary.csv`,
`r5_summary.paired.csv`, and `r5_summary.raw.csv`.

## Verification

The focused RNA-representation suite passed: `11 passed`.

---

# R5.2 — corrected re-validation (2026-07-31)

## Why R5.1 was revisited

A review of the R5.1 protocol found the `concat` comparison was not actually nested: every
R5 config set `training.warm_start_checkpoint = null`
(`scripts/rna_branch/make_representation_configs.py`), so `concat` trained a freshly
initialized, wider `nn.Linear` over `[raw RNA ; BulkRNABert embedding]` from scratch rather
than adding a residual on top of F2's own trained raw-RNA projection. The gene overlap
(0.7827615151) also came from aligning against the mean/std-filtered 21,792-gene matrix
shared by every other RNA-branch experiment, not the full TCGA gene set. Both were fixed and
the comparison was re-run end to end; this section documents the corrected result, which
**reverses** the R5.1 "stop" conclusion for the frozen-adapter design (does not reopen
LoRA/fine-tuning, which remains out of scope).

## Fixes applied

1. **Gene overlap: 78.3% → 100%.** `scripts/rna_branch/build_bulkrnabert_gene_source.py`
   sources gene alignment from the unfiltered
   `parquet/241231-tcga_array/gene_expr.parquet` (60,616 genes, pre-MethylProphet-filtering)
   instead of the 21,792-gene `tcga_rna.h5`. All 19,062 checkpoint genes matched
   (`artifacts/rna_branch/audits/bulkrnabert_gene_overlap.json`); 44 stable Ensembl IDs that
   appeared as duplicate rows were resolved by summing in TPM space before re-applying
   `log2(x+1)` (`artifacts/rna_branch/pretrained/inputs/tcga_rna_full_gene.h5.json`), not by
   arbitrarily keeping one row.
2. **Input scale independently verified, not just asserted.** `sum(2**x - 1)` over the full
   gene set is `1,000,000 ± 0.04` per sample (`artifacts/rna_branch/audits/bulkrnabert_input_scale.json`),
   confirming the source is genuinely `log2(TPM+1)`. Separately, the production tokenization
   in `extract_bulkrnabert_torch.py` was checked token-for-token against the official
   `preprocess_omic`/`BinnedOmicTokenizer` for 20 real patients (381,240 tokens):
   `token_mismatch_count = 0`. The `np.clip(source, 0, 30)` safety guard affects 0% of real
   values.
3. **JAX/PyTorch forward-pass parity measured, not just claimed.** The hand-translated
   PyTorch attention/LayerNorm/GELU forward pass was compared against the official
   JAX/Haiku forward pass (same token IDs, same checkpoint, CPU float32 for the JAX side
   since its dense attention is ~11.6GB for one sample regardless of batch size) at every
   layer, on both the mean-pooled embedding and a fixed 256-position token-wise subset
   (`artifacts/rna_branch/audits/bulkrnabert_jax_parity.json`). All 4 layers passed
   (cosine ≥ 0.9999999999, relative L2 ≤ 5e-6), far inside the pre-registered FP32
   thresholds (cosine ≥ 0.99999, relative L2 ≤ 1e-4).
4. **Genuinely nested F2 + frozen-adapter architecture.** New encoder
   `PretrainedEmbeddingRNAEncoder` (`kind: pretrained_embedding`) keeps F2's exact
   `norm`/`projection` for the raw-RNA half; new interaction `PretrainedEmbeddingF2Interaction`
   (`kind: cpg_pretrained_f2`) wraps F2's `ConcatInteraction` as `baseline` and adds a
   zero-init residual over the embedding. `training.warm_start_checkpoint` loads the real F2
   checkpoint; `training.freeze_warm_start_params` hard-freezes those exact parameters via
   `requires_grad_(False)` (not a zero learning rate, which would still let F2's gradients
   distort the residual branch's effective gradient through the shared `clip_grad_norm_`
   call); `training.seed_initial_checkpoint` makes epoch 0 (== F2 exactly) an eligible winner
   if the adapter never improves validation. Verified by
   `test_pretrained_f2_warm_start_frozen_and_residual_learns` and
   `test_seed_initial_checkpoint_can_win` (`tests/test_rna_branch_smoke.py`): predictions are
   exact to `<1e-6` immediately after warm start, F2's parameters are bit-identical after an
   optimizer step, and the residual branch's parameters do change.
5. **All 4 layers extracted and screened**, not only the final layer
   (`extract_bulkrnabert_torch.py` now saves `embeddings_layer1..4` in one pass).

## Layer screening (seed 17, validation only)

Every run's epoch-0 validation MSE matched F2's own validation MSE exactly
(`0.02554198`), confirming the nested architecture reproduces F2 at initialization.

| layer | best epoch | validation MSE | validation patient Pearson (median) |
| --- | ---: | ---: | ---: |
| 1 | 1 | 0.02537846 | 0.322883 |
| 2 | 1 | 0.02535170 | 0.322091 |
| 3 | 1 | **0.02533138** | **0.325302** |
| 4 | 1 | 0.02533941 | 0.324205 |

Selection rule (pre-registered): lowest validation MSE; ties within `1e-5` broken by
validation patient Pearson. Layers 3 and 4 are tied (`Δ=8.0e-6`); layer 3 wins the
tie-break on patient Pearson. **Layer 3 was selected.**

## `concat_shuffled` control (layer 3, seed 17)

Run on the same nested/frozen architecture as the real-embedding comparison (not the old
standalone-embedding `replace` control), with only the embedding permuted within cancer
type: validation MSE `0.02540960` (between F2's `0.02554198` and real layer 3's
`0.02533138`), double-OOD MSE `0.024034` and patient Pearson `0.292766` (vs. real layer 3's
`0.023885` / `0.300908`, and F2's `0.024128`(seed17) / `0.294491`(seed17)). Unlike the old
non-nested shuffled control (which was far worse than F2, `0.026828` double-OOD MSE), the
frozen/nested shuffled control still improves modestly over F2. This means part of the
apparent gain comes from the residual branch's extra capacity and direct access to the raw
locus embedding (present even with a shuffled, contentless patient vector), not solely from
BulkRNABert's content — the real-vs-shuffled gap (not the adapter-vs-F2 gap) is the more
honest estimate of BulkRNABert's own incremental contribution, and it is real but small.

## Double-OOD, 3-seed confirmation (layer 3 vs. F2, paired)

| family | seeds | MSE (mean ± SD) | skill vs prior | patient dynamic Pearson (median) | locus dynamic Pearson (median) |
| --- | ---: | ---: | ---: | ---: | ---: |
| F2 raw-RNA baseline | 3 | 0.024149 ± 0.000027 | 0.117958 | 0.290283 | 0.345902 |
| BulkRNABert layer-3 nested adapter | 3 | 0.024000 ± 0.000101 | 0.123390 | 0.296175 | 0.353866 |

Paired (`layer3 − F2`, matched by seed): MSE `−0.000149 ± 0.000082` (descriptive paired
`p=0.087`), skill `+0.005432` (`p=0.087`), **patient Pearson `+0.005892 ± 0.000559`
(`p=0.003`)**, locus Pearson `+0.007964` (`p=0.144`). MSE and patient Pearson both improve
in the same direction on all 3 seeds individually (seed 17/23/41 MSE: `0.023885 <0.024128`,
`0.024044<0.024140`, `0.024072<0.024180`; patient Pearson likewise higher on all 3 seeds).

## Decision

Per the pre-registered rule (accept "stop BulkRNABert" only if `ΔMSE ≥ 0` **and** patient
Pearson does not improve stably across seeds): **neither condition holds** — MSE improves on
all 3 seeds and patient Pearson improves significantly and consistently
(`p=0.003`, low cross-seed SD). **The R5.1 "stop BulkRNABert" conclusion is reversed for the
frozen-adapter design.** The effect is small (MSE `−0.6%` relative, patient Pearson `+0.006`
absolute) and the shuffled control shows part of it is architecture capacity rather than
BulkRNABert content specifically, so this should be read as "a small but real and
statistically supported incremental signal," not a large win. This does not by itself
justify LoRA/fine-tuning (out of scope here); it justifies treating BulkRNABert layer 3 as a
validated, worthwhile default addition to the frozen-feature RNA branch rather than a
dead end.

## Commands (production paths, R5.2)

```bash
# corrected extraction (100% gene overlap, all 4 layers)
PYTHONPATH=src conda run --no-capture-output -n methil-predictor \
  python scripts/rna_branch/build_bulkrnabert_gene_source.py \
  --output artifacts/rna_branch/pretrained/inputs/tcga_rna_full_gene.h5 \
  --official-repo artifacts/models/multiomics-open-research

PYTHONPATH=src conda run --no-capture-output -n methil-predictor \
  python scripts/rna_branch/extract_bulkrnabert_torch.py \
  --config configs/rna_branch/stage_f_bulkrnabert_full_gene.yaml \
  --official-repo artifacts/models/multiomics-open-research \
  --input-scale log2p1 --min-gene-overlap 0.999 \
  --output artifacts/rna_branch/pretrained/bulkrnabert_gtex_encode_v2.h5

# nested config generation + screening + confirmation
PYTHONPATH=src conda run --no-capture-output -n methil-predictor \
  python scripts/rna_branch/make_representation_configs.py \
  --base configs/rna_branch/stage_f_base.yaml \
  --output-root artifacts/rna_branch/r5_bulkrnabert_v2 \
  --seeds 17,23,41 \
  --f2-checkpoint 17=artifacts/rna_branch/stage_f_fusion/first_tranche/f2_concat_product_seed17/best.pt \
  --f2-checkpoint 23=artifacts/rna_branch/stage_f_fusion/confirm/f2_concat_product_seed23/best.pt \
  --f2-checkpoint 41=artifacts/rna_branch/stage_f_fusion/confirm/f2_concat_product_seed41/best.pt \
  --gene-embeddings artifacts/rna_branch/stage_t_gene_tokens/inputs/ntv3_gene_embeddings.npz \
  --pretrained-rna bulkrnabert=artifacts/rna_branch/pretrained/bulkrnabert_gtex_encode_v2.h5 \
  --r5-seeds 17,23,41 --r5-modes concat --r5b-layers 3   # screen: --r5b-layers 1,2,3,4 --r5-seeds 17

# aggregate
PYTHONPATH=src conda run --no-capture-output -n methil-predictor \
  python src/methylation_predictor/rna_branch/aggregate_report.py \
  --screening-dir artifacts/rna_branch/r5_bulkrnabert_v2 \
  --baseline-dir artifacts/rna_branch/stage_f_fusion/first_tranche \
  --baseline-dir artifacts/rna_branch/stage_f_fusion/confirm \
  --baseline-family f2_concat_product \
  --output artifacts/rna_branch/r5_bulkrnabert_v2/r5b_summary.csv
```

Audit artifacts: `artifacts/rna_branch/audits/bulkrnabert_{input_scale,gene_overlap,jax_parity}.json`,
`bulkrnabert_concat_training_dynamics_note.md`. Full per-panel exports:
`artifacts/rna_branch/r5_bulkrnabert_v2/r5b_summary.{csv,paired.csv,raw.csv}`.
