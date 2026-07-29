#!/usr/bin/env python3
"""Audit which requested CpGs are present in each MDS shard without inference."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd
from streaming import StreamingDataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", required=True); p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); wanted = set(pd.read_parquet(a.manifest, columns=["cpg_idx"]).cpg_idx.astype(int))
    result, union = {}, set()
    for path in sorted(glob.glob(a.glob)):
        dataset = StreamingDataset(local=path, shuffle=False)
        found = {int(dataset[i]["cpg_idx"]) for i in range(len(dataset)) if int(dataset[i]["cpg_idx"]) in wanted}
        result[path] = {"matching_cpg": len(found), "total_records": len(dataset)}
        union.update(found)
    output = {"requested_cpg": len(wanted), "covered_cpg": len(union), "missing_cpg": len(wanted - union), "shards": result}
    Path(a.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
