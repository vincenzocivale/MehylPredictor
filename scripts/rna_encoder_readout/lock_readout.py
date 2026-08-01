#!/usr/bin/env python3
"""Copy a validated RNA-only readout into a provenance-complete locked export."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decision-note", required=True)
    args = parser.parse_args()
    source = Path(args.run_dir)
    destination = Path(args.output_dir)
    required = [
        "best.pt", "embeddings.h5", "manifest.json", "config.resolved.yaml",
        "selected_input_genes.csv", "selected_target_genes.csv",
    ]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"run directory is incomplete: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in required:
        target = destination / name
        shutil.copy2(source / name, target)
        hashes[name] = _sha256(target)
    for optional in ("attention_gene_weights.csv", "history.csv"):
        if (source / optional).exists():
            shutil.copy2(source / optional, destination / optional)
            hashes[optional] = _sha256(destination / optional)
    lock = {
        "source_run": str(source),
        "decision_note": args.decision_note,
        "files": hashes,
        "selection_used_methylation": False,
        "encoder_frozen": True,
    }
    (destination / "LOCK.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
