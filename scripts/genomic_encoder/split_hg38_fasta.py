#!/usr/bin/env python3
"""One-off: split the whole-genome hg38 FASTA into one gzip file per chromosome.

Both static_features.py and ntv3_embeddings.py process one chromosome per
invocation, and ntv3_prior_common.load_fasta() concatenates every record in
a FASTA into a single string with no chromosome-boundary tracking -- so each
invocation must be given a single-chromosome FASTA, never the whole genome
directly. This splits artifacts/reference/hg38.fa.gz (already present, no
download needed) once, reusing that same 22-invocation-per-chromosome design
already established for the chr1-only pipeline.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path


def split(source: Path, output_dir: Path, chromosomes: set[str] | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_name: str | None = None
    current_handle = None
    written = []
    with gzip.open(source, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                name = line[1:].split()[0]
                if current_handle is not None:
                    current_handle.close()
                current_handle = None
                current_name = name
                if chromosomes is None or name in chromosomes:
                    target = output_dir / f"{name}.fa.gz"
                    current_handle = gzip.open(target, "wt")
                    current_handle.write(line)
                    written.append(name)
            elif current_handle is not None:
                current_handle.write(line)
        if current_handle is not None:
            current_handle.close()
    print(f"wrote {len(written)} chromosome FASTA files to {output_dir}: {sorted(written)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("artifacts/reference/hg38.fa.gz"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--chromosomes", help="comma list, e.g. chr1,chr2; default = all autosomes chr1-chr22")
    args = p.parse_args()
    chromosomes = set(args.chromosomes.split(",")) if args.chromosomes else {f"chr{i}" for i in range(1, 23)}
    split(args.source, args.output_dir, chromosomes)


if __name__ == "__main__":
    main()
