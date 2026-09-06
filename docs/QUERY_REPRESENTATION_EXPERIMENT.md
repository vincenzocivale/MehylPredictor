# Query representation experiment

## Hypothesis

The current locus-attention encoder queries RNA program tokens directly from the frozen 1536-D
NTv3 CpG embedding. Separately, the model learns `h_mean = CpGTrunk(NTv3)` under an auxiliary
per-locus mean-methylation objective. This experiment tests whether that methylation-aware
representation is a better query for transcriptomic program selection.

## Controlled arms

All arms keep the current final architecture fixed: K=64 RNA program tokens, program_dim=256,
4 heads, RNA bottleneck=256, mean branch enabled, no raw product term, identical optimizer/loss.
Only `query_source` changes:

1. `ntv3`: current `W_e e_c` query.
2. `mean_only`: `W_mu h_mean`.
3. `hybrid_detached`: `W_e e_c + W_mu stopgrad(h_mean)`.
4. `hybrid_joint`: `W_e e_c + W_mu h_mean`.

The hybrid mean projection is zero-initialized so the hybrid arms are exact reference queries at
initialization.

## Protocol

Architecture selection uses `mode=development`, training seed 17 and fixed split seed 17. Keeping
the split seed separate prevents a seed sweep from silently changing the validation set. The
primary selection metric is inner `val_cpg_x_val_sample` MAS-PCC; `val_cpg_x_train_sample`, MSE,
MAE and skill-vs-prior remain secondary diagnostics.

Run all arms sequentially:

```bash
nohup python scripts/experiments/run_query_representation.py --gpu 0 \
  > logs/query_representation_2026_09.driver.log 2>&1 &
```

The runner stops on failure, auto-resumes an interrupted arm when `checkpoints/last.pt` exists,
and automatically collects results when the sequence finishes.

Results are written to `results/reference/ablations/query_representation_2026_09/` with checkpoint
hashes, resolved-config hashes and a strict one-factor-at-a-time audit. Do not evaluate all four
arms on the official held-out split. Select once on development, then retrain/evaluate only the
winner under the final protocol.
