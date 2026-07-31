# R0–R4 execution

This study uses the Stage-F data and the matched F2 `concat` checkpoints for
seeds 17, 23 and 41.  R0 reuses F2 as the `zscore` reference and trains only
the four additional numerical encodings (12 runs); R1 has 9 runs; R2 has 6
runs (Hallmark and its matched random control); R4 has 9 runs.

Download the official human MSigDB Hallmark GMT, retain its release/version in
the downloaded filename, and run the sequence in the established environment:

```bash
PYTHONPATH=src conda run -n methil-predictor python \
  scripts/rna_branch/run_representation_experiments.py \
  --hallmark-gmt artifacts/rna_branch/modules/msigdb.<release>.Hs.H.all.v<release>.symbols.gmt
```

The launcher checks PyTorch CUDA before generating any run output.  It does
not fall back to CPU.  It writes the source GMT hash, RNA-alignment coverage,
random-control seed, matrix metadata, per-run manifests and the final paired
comparison at `artifacts/rna_branch/representation_search/representation_summary.{csv,paired.csv}`.

The Hallmark alignment is deliberately strict: an ambiguous RNA symbol is not
used; Ensembl versions are matched both with and without their version suffix.
The random control samples genes independently per retained module while
preserving each module's post-alignment size.
