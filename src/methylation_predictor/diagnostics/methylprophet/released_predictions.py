#!/usr/bin/env python3
"""Extract a deterministic sample-ID manifest from released prediction rows."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.dataset as ds
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    dataset = ds.dataset(args.predictions, format="parquet")
    ids = set()
    for batch in dataset.scanner(columns=["sample_idx"], filter=ds.field("group_idx") == args.group).to_batches():
        ids.update(batch.column(0).to_pylist())
    values = sorted(map(int, ids))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_idx": values}).to_parquet(out, index=False)
    digest = hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()
    out.with_suffix(".json").write_text(json.dumps({"group": args.group, "n_samples": len(values), "sample_idx_sha256": digest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
