"""Deterministic full-CpG-coverage-per-epoch sampler.

Replaces per-step independent random draws (``ExperimentRunner._sample_batch``/
``_sample_cpg_batch``) with a schedule that guarantees every CpG in the training
pool is visited at least once per epoch (and every sample at least once),
without enumerating the full Cartesian product. Each epoch's schedule is a
pure function of ``(seed, epoch, pool sizes, block sizes)`` -- no sampler
state needs to be persisted across a resume, since re-deriving epoch N+1's
schedule from scratch reproduces it exactly.

Block sizes reuse the existing ``training.cpg_batch_size``/``sample_batch_size``
config fields -- no new batch-size hyperparameter is introduced. The number of
steps needed for full coverage (``steps_per_epoch``) is derived, not
configured: it is the minimum number of blocks required to touch every CpG
once, given the fixed block size. This overrides the previously frozen
``training.steps_per_epoch`` -- a "correzione strettamente necessaria" for
full-coverage training (constraint #1 of the training request explicitly
permits this).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class EpochSchedule:
    epoch: int
    cpg_pool: np.ndarray
    sample_pool: np.ndarray
    cpg_block_size: int
    sample_block_size: int
    seed: int

    n_cpg_blocks: int = field(init=False)
    n_sample_blocks: int = field(init=False)
    steps_per_epoch: int = field(init=False)
    rotation_offset: int = field(init=False)
    _cpg_blocks: list[np.ndarray] = field(init=False, repr=False)
    _sample_blocks: list[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cpg_block_size < 1 or self.sample_block_size < 1:
            raise ValueError("block sizes must be positive")
        if not len(self.cpg_pool) or not len(self.sample_pool):
            raise ValueError("cpg_pool and sample_pool must be non-empty")

        self.n_cpg_blocks = int(np.ceil(len(self.cpg_pool) / self.cpg_block_size))
        self.n_sample_blocks = int(np.ceil(len(self.sample_pool) / self.sample_block_size))
        self.steps_per_epoch = max(self.n_cpg_blocks, self.n_sample_blocks)
        self.rotation_offset = self.epoch % self.n_sample_blocks

        cpg_rng = np.random.default_rng([self.seed, self.epoch, 1])
        sample_rng = np.random.default_rng([self.seed, self.epoch, 2])
        cpg_order = cpg_rng.permutation(len(self.cpg_pool))
        sample_order = sample_rng.permutation(len(self.sample_pool))
        self._cpg_blocks = list(np.array_split(cpg_order, self.n_cpg_blocks))
        self._sample_blocks = list(np.array_split(sample_order, self.n_sample_blocks))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _block_indices(self, step: int) -> tuple[int, int]:
        """Two INDEPENDENT cyclic counters, each over its own modulus. Using
        `step % n_cpg_blocks` and `step % n_sample_blocks` directly (rather
        than deriving one index from the other) guarantees every residue of
        BOTH moduli is visited at least once within `steps_per_epoch =
        max(n_cpg_blocks, n_sample_blocks)` steps -- deriving the sample index
        from the cpg index (e.g. `(cpg_block_idx + offset) % n_sample_blocks`)
        would only cycle through `min(n_cpg_blocks, n_sample_blocks)` distinct
        sample residues whenever n_cpg_blocks < n_sample_blocks, silently
        missing sample blocks. `rotation_offset` shifts the sample counter's
        phase so a given epoch's cpg/sample pairing differs from the previous
        epoch's, without affecting the coverage guarantee."""
        cpg_block_idx = step % self.n_cpg_blocks
        sample_block_idx = (step + self.rotation_offset) % self.n_sample_blocks
        return cpg_block_idx, sample_block_idx

    def __getitem__(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        if step < 0 or step >= self.steps_per_epoch:
            raise IndexError(step)
        cpg_block_idx, sample_block_idx = self._block_indices(step)
        sample_indices = self.sample_pool[self._sample_blocks[sample_block_idx]]
        cpg_indices = self.cpg_pool[self._cpg_blocks[cpg_block_idx]]
        return sample_indices, cpg_indices

    def coverage_report(self) -> dict[str, float | int]:
        """Analytic coverage/exposure summary -- does not require actually
        running training, since block membership is fully determined above."""
        cpg_exposure = np.zeros(len(self.cpg_pool), dtype=np.int64)
        sample_exposure = np.zeros(len(self.sample_pool), dtype=np.int64)
        pair_slots = 0
        for step in range(self.steps_per_epoch):
            cpg_block_idx, sample_block_idx = self._block_indices(step)
            cpg_local = self._cpg_blocks[cpg_block_idx]
            sample_local = self._sample_blocks[sample_block_idx]
            cpg_exposure[cpg_local] += 1
            sample_exposure[sample_local] += 1
            pair_slots += len(cpg_local) * len(sample_local)

        unique_cpg_seen = int(np.count_nonzero(cpg_exposure))
        unique_samples_seen = int(np.count_nonzero(sample_exposure))
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "n_cpg_blocks": self.n_cpg_blocks,
            "n_sample_blocks": self.n_sample_blocks,
            "unique_cpg_seen": unique_cpg_seen,
            "cpg_coverage_fraction": unique_cpg_seen / len(self.cpg_pool),
            "unique_samples_seen": unique_samples_seen,
            "sample_coverage_fraction": unique_samples_seen / len(self.sample_pool),
            "pair_slots_processed": int(pair_slots),
            "cpg_exposure_mean": float(cpg_exposure.mean()),
            "cpg_exposure_min": int(cpg_exposure.min()),
            "cpg_exposure_max": int(cpg_exposure.max()),
            "sample_exposure_mean": float(sample_exposure.mean()),
            "sample_exposure_min": int(sample_exposure.min()),
            "sample_exposure_max": int(sample_exposure.max()),
        }


def build_epoch_schedule(
    cpg_pool: np.ndarray,
    sample_pool: np.ndarray,
    cpg_block_size: int,
    sample_block_size: int,
    epoch: int,
    seed: int,
) -> EpochSchedule:
    return EpochSchedule(
        epoch=epoch,
        cpg_pool=np.asarray(cpg_pool),
        sample_pool=np.asarray(sample_pool),
        cpg_block_size=cpg_block_size,
        sample_block_size=sample_block_size,
        seed=seed,
    )
