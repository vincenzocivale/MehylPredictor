#!/usr/bin/env python3
"""Lock the two final NTv3 configurations from a validation-only leaderboard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leaderboard", required=True); p.add_argument("--output", required=True)
    p.add_argument("--n-configs", type=int, default=2)
    a = p.parse_args()
    board = json.loads(Path(a.leaderboard).read_text())
    entries = sorted(board["entries"], key=lambda x: x["validation_mse"])
    selected, seen = [], set()
    for entry in entries:
        key = (entry["length"], entry["representation"])
        if key in seen: continue  # orientation is selected inside each locus representation pair
        selected.append({name: entry[name] for name in ("length", "representation", "orientation")})
        seen.add(key)
        if len(selected) == a.n_configs: break
    if len(selected) != a.n_configs:
        raise ValueError(f"leaderboard contains only {len(selected)} distinct length/representation configurations")
    Path(a.output).write_text(json.dumps({"selection_source": str(a.leaderboard), "selection_metric": "validation MSE in beta-space", "configs": selected}, indent=2) + "\n")


if __name__ == "__main__": main()
