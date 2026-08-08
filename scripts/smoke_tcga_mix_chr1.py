#!/usr/bin/env python3
"""Smoke test for the tcga_mix_chr1 data/protocol layer. No training happens here.

    python scripts/smoke_tcga_mix_chr1.py [--root PATH]

Exercises exactly the Definition-of-Done checklist:
  1. open tcga_mix_chr1
  2. sample a small training batch from the mix
  3. load RNA + beta + sample_idx + cpg_idx for that batch
  4. read all three evaluation views (shape + a tiny finite-count sample)
  5. verify the official chr1 counts by assertion
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from methylation_predictor.tcga_canonical import (  # noqa: E402
    TCGACanonicalBundle,
    load_protocol,
    resolve_bundle_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="override the canonical bundle root")
    args = parser.parse_args()

    root = resolve_bundle_root(root=args.root)
    print(f"[1/5] opening bundle at {root}")
    t0 = time.time()
    bundle = TCGACanonicalBundle.from_root(root)
    print(f"      shapes: rna={bundle.rna.shape} array={bundle.sources['array'].shape} "
          f"epic={bundle.sources['epic'].shape} wgbs={bundle.sources['wgbs'].shape} "
          f"({time.time() - t0:.1f}s)")

    print("[2/5] loading protocol tcga_mix_chr1")
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    assert len(protocol.array_train_sample_idx) == 8260
    assert len(protocol.array_val_sample_idx) == 918
    assert len(protocol.array_train_cpg_idx) == 33885
    assert len(protocol.array_val_cpg_idx) == 6742
    print("      official chr1 split counts verified: 8260/918 samples, 33885/6742 CpGs")

    print("[3/5] sampling one small training batch from the mix")
    dataset = protocol.train_dataset(batch_sample_size=16, batch_cpg_size=128, seed=0)
    t0 = time.time()
    batch = dataset.sample_batch()
    print(f"      {len(batch)} finite (sample, cpg) pairs, sources={sorted(set(batch.source.tolist()))}, "
          f"rna shape={batch.rna.values.shape} ({time.time() - t0:.1f}s)")
    import numpy as np
    assert np.isfinite(batch.beta).all(), "sampler must never emit a non-finite target"
    assert np.isfinite(batch.rna.values).all()

    print("[4/5] reading the three official evaluation views")
    views = protocol.evaluation_views()
    for name, view in views.items():
        print(f"      {name}: {len(view.sample_idx)} samples x {len(view.cpg_idx)} CpGs")

    print("[5/5] computing exact finite-target counts for all three views (reads real data, ~1min)")
    counts = protocol.evaluation_finite_counts()
    expected = {
        "train_cpg_x_val_sample": 30_574_946,
        "val_cpg_x_train_sample": 55_155_121,
        "val_cpg_x_val_sample": 6_129_547,
    }
    for name, value in counts.items():
        status = "OK" if value == expected[name] else "MISMATCH"
        print(f"      {name}: {value:,} (expected {expected[name]:,}) [{status}]")
    assert counts == expected, f"evaluation finite counts do not match the released checkpoint: {counts}"

    bundle.close()
    print("\nSMOKE OK -- no training was run, no model was constructed.")


if __name__ == "__main__":
    main()
