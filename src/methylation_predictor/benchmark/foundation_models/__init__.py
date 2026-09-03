"""Isolated masked-CpG readiness/evaluation pipeline for external pretrained
methylation foundation models (CpGPT, MethylGPT).

Mirrors `benchmark/methylprophet/`'s isolation: self-contained, not routed through
`rna_training/`'s shared trainer/evaluator, and does not add these models'
framework-specific dependencies (torchtext, hydra, lightning, ...) to the core
package's requirements -- see `docs/PAPER_EXPERIMENTS.md`'s foundation-model
section and `scripts/benchmark_foundation_models/setup.sh` for the vendoring/env
setup this module assumes (`external/CpGPT`, `external/MethylGPT`, one isolated
venv per model).
"""
