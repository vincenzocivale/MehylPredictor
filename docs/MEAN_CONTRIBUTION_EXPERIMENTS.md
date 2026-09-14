# Mean-contribution experiment — paper protocol

This study isolates what the locus-mean proxy task contributes to the current
RNA→DNA-methylation reference model. It deliberately does **not** change the
RNA encoder, token count, CpG embedding, optimizer, beta loss, data sources,
batching, or official split.

> **2026-09-14 fix.** `analyze_mean_contribution.py`'s claim-3 diagnostics
> previously probed pre-FFN `h_c` (`LayerNorm(peak+dense)`) and ran
> `mean_head` directly on it. The real forward pass never does this --
> `mean_head` only ever sees `h_c_deep`, `h_c` after the 8 functional FFN
> blocks (`src/methylation_predictor/modeling/final.py`). Fixed so the probe
> and the direct mean-head evaluation both use `h_c_deep`; pre-FFN `h_c` is
> kept as a supplementary shallow-vs-deep control, not the main-paper metric.

## Claims and matching experiments

1. **Biological proxy supervision improves final prediction on unseen CpGs.**
   Compare `full_reference` vs `no_mean_supervision`.
2. **The effect is not merely extra branch capacity.**
   Compare `no_mean_supervision` vs `no_mean_branch`.
3. **The proxy makes `h_c_deep` methylation-aware.**
   Fit a linear probe on `h_c_deep` (functional `h_c` after the 8 functional
   FFN blocks -- the exact tensor the mean head and the auxiliary loss consume
   in `EfficientSingleAttentionPredictor.forward`, see `docs/MODEL.md`) using
   train CpGs only; evaluate it on official val CpGs. `analyze_mean_contribution.py`
   also reports a supplementary shallow-`h_c` (pre-FFN) control probe, which is
   never seen by the mean head or the auxiliary loss; that control is not a
   main-paper metric, only a sanity check for how much signal already exists
   before the functional FFN stack.
4. **The mean primarily corrects absolute locus calibration.**
   Report MSE and locus-level bias as primary metrics; MAS-PCC is secondary.
5. **The biological motivation exists in the data.**
   Decompose methylation variance into between-locus mean variance and
   within-locus patient variance on the official training matrix.

## Arms

| arm | mean branch | mean auxiliary loss | RNA encoder | product |
|---|---:|---:|---|---:|
| `full_reference` | yes | 0.15 | locus attention K=64 | no |
| `no_mean_supervision` | yes | 0 | locus attention K=64 | no |
| `no_mean_branch` | no | — | locus attention K=64 | no |

Seeds: **17, 29, 43**.

The full-reference run ids intentionally reuse the already-launched
`ref-locus-attn-k64-noproduct-seed{seed}` runs.

## Launch

```bash
conda activate methyl-predictor
export METHYL_DATA_ROOT=/dune/DATASETS/MethylPredictionData

# Distribute these two arms across free GPUs/machines.
python scripts/experiments/run_mean_contribution.py \
  --arms no_mean_supervision --gpu 0

python scripts/experiments/run_mean_contribution.py \
  --arms no_mean_branch --gpu 1

# Reuse reference training and fill only missing eval/diagnostics.
python scripts/experiments/run_mean_contribution.py \
  --arms full_reference --gpu 2
```

The runner is idempotent: it resumes training when possible and skips
training/evaluation/diagnostics that already exist.

## Dataset variance decomposition

No checkpoint or GPU is required:

```bash
python scripts/experiments/analyze_mean_contribution.py dataset \
  --canonical-root "$METHYL_DATA_ROOT/datasets/methylprophet_repro_v1" \
  --prepared-root "$METHYL_DATA_ROOT/derived/methylprophet_table5_tcga_chr1" \
  --output "$METHYL_DATA_ROOT/experiments/analysis/mean_contribution_2026_09/dataset_variance.json"
```

The primary decomposition is `train_cpg × train_sample`, so the biological
motivation does not inspect the held-out evaluation matrix. A
`val_cpg × train_sample` confirmation is saved separately.

## Collect paper results

After the runs finish:

```bash
python scripts/experiments/collect_mean_contribution.py \
  --dataset-diagnostics \
  "$METHYL_DATA_ROOT/experiments/analysis/mean_contribution_2026_09/dataset_variance.json"
```

Generated, version-controlled outputs:

```text
results/reference/appendix/mean_contribution_2026_09/
  summary.yaml
  summary.md
  paper_table.csv
  variance_decile_effects.csv
  dataset_variance.json
  runs/
    full_reference__seed17.json
    ...
```

Each per-run JSON stores recipe SHA-256, resolved-config SHA-256, checkpoint
SHA-256, checkpoint epoch, seed, run id, all official views, mean-head
accuracy, `h_mean` linear-probe performance, locus-bias metrics, and
variance-decile metrics.

## chr123 single-seed follow-up

The same three arms (`full_reference`, `no_mean_supervision`, `no_mean_branch`)
are also run once, seed 17 only, on chr123 -- a lighter-weight check that the
mean-contribution effect isn't chr1-specific, not a second full statistical
ladder. It reuses the exact same recipes and the generic `shared_backbone`
engine against the chr123 compact cache (`scripts/prepare_chr123_compact.py`,
`derived/methylprophet_compact_chr123`), rather than the chr1-only
`matched_chr1_shared_backbone` engine:

```bash
python scripts/experiments/run_mean_contribution.py --scope chr123 \
  --arms full_reference,no_mean_supervision,no_mean_branch --seeds 17

python scripts/experiments/collect_mean_contribution.py --scope chr123
```

Both `run_mean_contribution.py` and `collect_mean_contribution.py` take
`--scope {chr1,chr123}` (default `chr1`); chr123 run-ids get a `-chr123` suffix
(e.g. `ref-locus-attn-k64-noproduct-seed17-chr123`) to match the convention in
`results/reference/ours/04_baselines.yaml`'s `chr123_arms`. Output lands in a
sibling ledger, not mixed into the chr1 one:

```text
results/reference/appendix/mean_contribution_2026_09_chr123/
  summary.yaml
  summary.md
  paper_table.csv
  variance_decile_effects.csv
  runs/
    full_reference__seed17.json
    ...
```

No dataset-variance decomposition is repeated for chr123 (chr1's already
establishes the between-/within-locus split is real; re-running it isn't part
of this single-seed check). Paired-effect statistics (mean/stdev across seeds)
degenerate to n=1 point estimates here -- read them as a single-seed sanity
check, not as statistically powered evidence, unlike the chr1 ladder.

## Results — claim 3 (chr1, seed 17, preliminary single-seed)

Produced with the fixed `analyze_mean_contribution.py` (see the 2026-09-14 fix
note above) run directly against two already-trained checkpoints, no
retraining: `full_reference` reuses `main-seed17-r3` (aux_weight=0.15, 80/80
epochs); `no_mean_supervision` is `mean_proxy__no_supervision__seed17`
(aux_weight=0.0, 80/80 epochs, `mean_aux_loss=0.0` throughout confirmed from
`training/history.json`). Both diagnosed on host `spyro`; raw output lives at
`dune_data/experiments/runs/rna_methylation/chr1/<run-id>/evaluation/chr1/mean_diagnostics.json`
(gitignored, shared data mount, not tracked in this repo). Logged in
`docs/EXPERIMENT_LOG.md` under the 2026-09-14 `mean_proxy` diag rows.

Held-out official val CpGs, n=6742:

| metric | `full_reference` (λ_μ=0.15) | `no_mean_supervision` (λ_μ=0.0) |
|---|---:|---:|
| `h_c_deep` linear probe — MSE | **0.00380** | 0.00718 |
| `h_c_deep` linear probe — R² | **0.968** | 0.939 |
| `h_c_deep` linear probe — Pearson | **0.985** | 0.970 |
| shallow-`h_c` control probe — MSE | 0.01108 | 0.01107 |
| direct `mean_head(h_c_deep)` — Pearson | 0.993 | n/a (see note) |

Reading:

- **Effect is concentrated in `h_c_deep`, not upstream of it.** The shallow-`h_c`
  control probe is essentially identical between arms (0.01107 vs 0.01108) --
  the pre-FFN functional embedding carries the same signal regardless of the
  proxy, as expected since `track_embedding`/`dense_encoder`/`locus_norm` are
  shared and trained by the beta loss in both arms. The gap only opens up
  after the 8 `functional_ffn` blocks.
- **The proxy roughly halves the `h_c_deep` probe's MSE** (0.00380 vs 0.00718)
  and lifts R² from 0.939 to 0.968, supporting claim 3: the auxiliary loss
  organizes `h_c_deep` to be more methylation-aware than beta-loss gradient
  through the same blocks achieves on its own.
- **`no_mean_supervision`'s `mean_head` is untrained by construction**
  (aux_weight=0.0 means it never receives gradient), so its direct
  `mean_head(h_c_deep)` evaluation is meaningless there (observed
  pearson=-0.25, r2=-0.19) -- excluded from the table above. The linear probe,
  fit fresh on that arm's frozen `h_c_deep`, is the only valid claim-3
  comparison for `no_mean_supervision`.

**Caveat:** this is seed 17 only. The chr1 protocol above calls for seeds
17/29/43 paired statistics; seeds 29 and 43 have not been trained for this
study yet, so treat this as a single-seed directional result, not the
paper-ready statistically powered claim.

## Statistical reporting

Use paired seeds.

- `full_reference` vs `no_mean_supervision` = **proxy-task supervision**.
- `no_mean_supervision` vs `no_mean_branch` = **branch capacity**.
- `full_reference` vs `no_mean_branch` = **total mean contribution**.

For unseen-CpG views, emphasize percentage MSE reduction and percentage
reduction in median absolute locus bias. Keep MAS-PCC visible, but do not use
it as the primary evidence for a CpG-specific mean effect because Pearson
correlation across samples is invariant to a CpG-wise constant shift.
