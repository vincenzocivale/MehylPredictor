# Current ablations

Development ablations use one frozen seed (`17`) for controlled comparison.
Multi-seed robustness is intentionally the final experiment after model choice.
See [`TCGA_CHR1_EXPERIMENTS.md`](TCGA_CHR1_EXPERIMENTS.md) for how to launch
these and the full experiment rationale.

## Reference

- Train CpG × Val Sample: MAS-PCC 0.5811, MSE 0.0144
- Val CpG × Train Sample: MAS-PCC 0.5708, MSE 0.0197
- Val CpG × Val Sample: MAS-PCC 0.5401, MSE 0.0201

## `large_sample_pcc`
Array row block 128 -> 512; CpG block 2048 -> 512.

## `tail_aware_pcc`
Mean PCC 0.10 + lower-tail PCC 0.05, lower-tail fraction 0.60,
target-std floor 0.02.

## `array_only_structured`
MSE/standardized-residual objectives remain on all sources; Pearson-family
structured objectives are active only on Array.

Run the three experiments independently from the reference. Combine only
changes that help.
