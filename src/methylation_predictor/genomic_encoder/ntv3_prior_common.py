"""Shared, deterministic primitives for the NTv3 locus-prior experiment."""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_fasta(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return "".join(line.strip() for line in handle if not line.startswith(">" )).upper()


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def centred_window(genome: str, cpg_position_1based: int, length: int) -> str:
    """Return an even-length window with C/G at zero-based L/2-1,L/2."""
    if length <= 1 or length % 2:
        raise ValueError("NTv3 window length must be a positive even number")
    c = cpg_position_1based - 1
    if genome[c:c + 2] != "CG":
        raise ValueError(f"hg38 position {cpg_position_1based} is not a CG dinucleotide")
    start, end = c - (length // 2 - 1), c + (length // 2 + 1)
    left, right = max(0, -start), max(0, end - len(genome))
    value = "N" * left + genome[max(0, start):min(len(genome), end)] + "N" * right
    if len(value) != length or value[length // 2 - 1:length // 2 + 1] != "CG":
        raise AssertionError("centred-window invariant violated")
    return value


def base_to_output(base_index: int, input_length: int, output_length: int) -> int:
    """Map a nucleotide centre to the output bin that contains its centre."""
    if not 0 <= base_index < input_length or output_length < 1:
        raise ValueError("invalid base/output coordinate")
    return min(output_length - 1, int(((base_index + .5) * output_length) // input_length))


def locus_vectors(values: np.ndarray, input_length: int) -> dict[str, np.ndarray]:
    """Centre CpG and local mean vectors for a single base/bin embedding array."""
    if values.ndim != 2:
        raise ValueError("expected [sequence-position, embedding-dimension]")
    n = len(values); c, g = input_length // 2 - 1, input_length // 2
    ci, gi = base_to_output(c, input_length, n), base_to_output(g, input_length, n)
    result = {"centre": values[[ci, gi]].mean(axis=0)}
    for radius in (32, 128, 512):
        left, right = max(0, c - radius), min(input_length - 1, g + radius)
        lo, hi = base_to_output(left, input_length, n), base_to_output(right, input_length, n)
        result[f"pool_{radius}"] = values[lo:hi + 1].mean(axis=0)
    return result
