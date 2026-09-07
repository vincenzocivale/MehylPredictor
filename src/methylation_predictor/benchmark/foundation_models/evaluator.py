"""Full val_cpg_x_train_sample masked-recovery evaluators for CpGPT / MethylGPT.

Duck-typed against the same shape as `rna_training.evaluator.CpGPriorEvaluator`
(`__init__`, `close`, `_view`, `run`), reusing `rna_training.metrics.ArrayMomentMetrics`
so the output `metrics.json`/`manifest.json` are schema-compatible with every other
evaluator in this repo.

**Not run in this repo yet** (last checked 2026-09-07) -- the two blockers recorded
2026-09-02 were the same missing piece and are now resolved (see
`crosswalk.IlluminaCrosswalk`, `docs/PAPER_EXPERIMENTS.md`'s foundation-model section):
CpGPT's genomic-position -> DNA-embedding-mmap-row lookup and MethylGPT's Illumina
probe-ID -> hg38-position crosswalk both come from CpGPT's own `human_dependencies`
bundle, and empirically cover 6741/6742 (99.99%) of chr1's `val_cpg_x_train_sample`
held-out CpGs (`scripts/benchmark_foundation_models/check_crosswalk_coverage.py`,
`results/reference/appendix/foundation_models/chr1_crosswalk_coverage.json`).

**Update 2026-09-07 (later same day): CpGPT's real masked-recovery pipeline is now working
end-to-end**, but NOT through this evaluator class -- it's driven by three standalone
scripts under `scripts/benchmark_foundation_models/`, because the real API is a three-stage
pipeline (`TCGACanonicalBundle` -> `.arrow` -> `CpGPTDataSaver` -> `CpGPTDataModule` ->
`CpGPTTrainer.predict(predict_mode="reconstruct")`), not a per-CpG `_predict()` call this
class's chunked-block loop shape can express directly:
  1. `prepare_cpgpt_chr1_context.py` (main `methyl-predictor` env, needs `h5py`): reads real
     Array beta values for chr1's 8,260 train-pool samples at their 33,885 known train CpGs
     from `TCGACanonicalBundle`, writes a `.arrow` file column-named by CpGPT's own
     `"chrom:pos"` convention (`crosswalk.IlluminaCrosswalk.to_location_key`).
  2 + 3. `run_cpgpt_reconstruct_pilot.py` (despite the name, this is the actual driver --
     kept the pilot name from its original smoke-test purpose; run under
     `external/cpgpt-env`'s python, which needed `pip install requests pyfaidx` beyond what
     `setup.sh` installs): runs `CpGPTDataSaver.process_files()` then
     `CpGPTTrainer().predict(predict_mode="reconstruct", genomic_locations=<val CpGs>,
     species="homo_sapiens")`, converts the M-value output via `cpgpt.model.utils.m_to_beta`.
  4. `score_cpgpt_chr1_reconstruct.py` (main env again): compares predicted beta against
     real target beta via `TCGACanonicalBundle`, produces a `metrics.json` using this same
     module's `ArrayMomentMetrics` convention.

Measured on this host (CPU-only, `external/cpgpt-env`'s torch has no CUDA build): the
`small` checkpoint, `batch_size=32`, full 33,885-CpG context (truncated to
`max_length=10_000` per sample by CpGPT's own `CpGPTDataModule`) reconstructing all 6,741
crosswalk-covered val CpGs took ~78s for a 64-sample pilot -- extrapolates to ~2.8h for the
full 8,260-sample run (launched in background 2026-09-07, see
`derived/foundation_models/cpgpt/chr1_full_small_reconstruct.log`). `external/cpgpt-env`
is still missing `h5py` (not needed for stages 2-3, which only read the already-built
`.arrow` file); MethylGPT and DeepCpG remain unimplemented -- see below.

What's still genuinely missing, per model:
  - MethylGPT: the vocabulary crosswalk is solved, but scoring still needs MethylGPT's
    own tokenizer/embedding-lookup wired to its 49,156-probe vocabulary, and
    `external/methylgpt-env`'s torch build is CPU-only on this host (no CUDA) --
    workable for a coverage-scale run, likely slow for the full context.
  - DeepCpG: not blocked on this crosswalk (DNA-sequence-only, needs an hg38 FASTA
    path instead -- still not provided, see `docs/PAPER_EXPERIMENTS.md`).

`_predict_cpgpt`/`_predict_methylgpt` below still raise `NotImplementedError` -- this class
itself was never wired up; use the standalone scripts above for CpGPT instead.
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
                # `prior` must be 1D (per-CpG, length n_cpgs) -- `add()` does `prior[None, :]`
                # to broadcast across the sample axis; a 2D array here makes that 3D and
                # `broadcast_to` rejects it (hit for real in
                # `score_cpgpt_chr1_reconstruct.py`, 2026-09-07, fixed there and here).
                metrics.add(s0, c0, target, pred, prior=np.full(target.shape[1], np.nan))

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
            "The genomic-position -> DNA-embedding-mmap-row lookup this needed is now "
            "resolved (see crosswalk.IlluminaCrosswalk); still missing is driving CpGPT's "
            "own trainer.predict(predict_mode='reconstruct') data pipeline -- see this "
            "module's docstring."
        )


class MethylGPTEvaluator(_MaskedRecoveryEvaluatorBase):
    model_name = "methylgpt"

    def _predict(self, sample_idx: np.ndarray, positions: list[tuple[str, int]]) -> np.ndarray:
        raise NotImplementedError(
            "Checkpoint loading no longer needs flash-attn (methylgpt_adapter's "
            "Wqkv key remap fixed that, see PAPER_EXPERIMENTS.md) and the probe-ID "
            "crosswalk this needed is now resolved (see crosswalk.IlluminaCrosswalk); "
            "still missing is wiring MethylGPT's own tokenizer/vocabulary-indexed "
            "forward pass -- see this module's docstring."
        )
