"""Full val_cpg_x_train_sample masked-recovery evaluators for CpGPT / MethylGPT.

Duck-typed against the same shape as `rna_training.evaluator.CpGPriorEvaluator`
(`__init__`, `close`, `_view`, `run`), reusing `rna_training.metrics.ArrayMomentMetrics`
so the output `metrics.json`/`manifest.json` are schema-compatible with every other
evaluator in this repo.

**Not run in this repo yet** (2026-09-02) -- readiness only, per
`docs/PAPER_EXPERIMENTS.md`'s foundation-model section:
  - CpGPT's `predict()` needs each val-CpG's real DNA-sequence embedding, looked up
    by genomic position from `external/checkpoints/cpgpt_human_dependencies/
    dna_embeddings/.../2001bp_dna_embeddings.mmap` -- that position->row lookup
    (an index into a ~5GB NTv2 embedding mmap, likely keyed by an Ensembl/Illumina
    id per `ensembl_metadata.db`/`illumina_metadata.db` in the same dependency
    bundle) isn't built yet; `_predict_cpgpt` raises `NotImplementedError` until it
    is, rather than guess at a mapping.
  - MethylGPT's released checkpoints only load exactly with flash-attn installed
    (see `scripts/benchmark_foundation_models/check_readiness.py`'s
    `mismatch_diagnosis` -- confirmed 2026-09-02, this CPU dev machine has no
    flash-attn build); `_predict_methylgpt` raises `NotImplementedError` until run
    on a GPU host with flash-attn.

Both classes' `__init__`/`_view`/`run` scaffolding is otherwise complete and
matches the existing evaluator template -- only the per-model prediction step is
pending.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from methylation_predictor.rna_training.metrics import ArrayMomentMetrics
from methylation_predictor.tcga_canonical.bundle import TCGACanonicalBundle
from methylation_predictor.benchmark.foundation_models.protocol import (
    load_train_sample_context_ids_chr1,
    load_val_cpg_positions_chr1,
)


class _MaskedRecoveryEvaluatorBase:
    """Shared `val_cpg_x_train_sample` scaffolding for chr1, before per-model predict."""

    view_name = "val_cpg_x_train_sample"

    def __init__(
        self,
        *,
        canonical_root: str | Path,
        table5_protocol_root: str | Path,
        array_cpg_registry: str | Path,
        checkpoint: str | Path,
        output: str | Path,
        sample_chunk: int = 128,
        cpg_chunk: int = 2048,
    ) -> None:
        self.bundle = TCGACanonicalBundle.from_root(canonical_root)
        self.checkpoint = Path(checkpoint)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.sample_chunk = sample_chunk
        self.cpg_chunk = cpg_chunk

        self.context_sample_idx = load_train_sample_context_ids_chr1(table5_protocol_root)
        self.target_positions = load_val_cpg_positions_chr1(table5_protocol_root, array_cpg_registry)

    def close(self) -> None:
        self.bundle.close()

    def _predict(self, sample_idx: np.ndarray, positions: list[tuple[str, int]]) -> np.ndarray:
        raise NotImplementedError  # overridden per model

    def run(self) -> dict:
        source = self.bundle.sources["array"]
        metrics = ArrayMomentMetrics(len(self.context_sample_idx), len(self.target_positions))

        for s0 in range(0, len(self.context_sample_idx), self.sample_chunk):
            sample_chunk_idx = self.context_sample_idx[s0 : s0 + self.sample_chunk]
            rows = source.rows_of_samples(sample_chunk_idx)
            for c0 in range(0, len(self.target_positions), self.cpg_chunk):
                position_chunk = self.target_positions[c0 : c0 + self.cpg_chunk]
                target = source.block(rows, [pos for _chrom, pos in position_chunk])
                pred = self._predict(sample_chunk_idx, position_chunk)
                metrics.add(s0, c0, target, pred, prior=np.full_like(target, np.nan))

        result = {
            "schema_version": 1,
            "model": self.model_name,
            "evaluation_scope": "chr1",
            "checkpoint": str(self.checkpoint),
            "views": {self.view_name: {"global": metrics.finalize()}},
        }
        (self.output / "metrics.json").write_text(__import__("json").dumps(result, indent=2))
        return result


class CpGPTEvaluator(_MaskedRecoveryEvaluatorBase):
    model_name = "cpgpt"

    def _predict(self, sample_idx: np.ndarray, positions: list[tuple[str, int]]) -> np.ndarray:
        raise NotImplementedError(
            "CpGPT masked-recovery prediction needs a genomic-position -> "
            "dna_embeddings-mmap-row lookup that is not built yet -- see this "
            "module's docstring."
        )


class MethylGPTEvaluator(_MaskedRecoveryEvaluatorBase):
    model_name = "methylgpt"

    def _predict(self, sample_idx: np.ndarray, positions: list[tuple[str, int]]) -> np.ndarray:
        raise NotImplementedError(
            "MethylGPT checkpoints only load exactly with flash-attn installed "
            "(see check_readiness.py's mismatch_diagnosis) -- not available on "
            "this CPU dev machine."
        )
