#!/usr/bin/env python3
"""Validation-only orchestration for Experiment 4 probe screening/final comparison.

Extraction is deliberately a separate GPU stage. This runner consumes its NPZ
artifacts and never opens test metrics to choose a configuration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--embeddings-dir", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint", default="NTv3_100M_post"); p.add_argument("--split", choices=("random", "block"), default="block"); p.add_argument("--block-size", type=int, default=5_000_000)
    p.add_argument("--lengths", default="1024,8192,32768,131072", help="comma-separated contexts for screen phase")
    p.add_argument("--device", default="cpu", help="probe device; CPU avoids competing with NTv3 extraction")
    p.add_argument("--phase", choices=("screen", "final"), required=True); p.add_argument("--final-configs", help="JSON list of selected {length,representation,orientation}")
    a = p.parse_args(); root, output = Path(a.embeddings_dir), Path(a.output_dir); output.mkdir(parents=True, exist_ok=True)
    if a.phase == "screen": configs = [{"length": l, "representation": r, "orientation": o} for l in (int(x) for x in a.lengths.split(",")) for r in ("centre","transformer","multiscale") for o in ("forward","rc_average")]
    else:
        if not a.final_configs: raise ValueError("--final-configs is required for final phase")
        loaded = json.loads(Path(a.final_configs).read_text())
        configs = loaded["configs"] if isinstance(loaded, dict) and "configs" in loaded else loaded
    entries = []
    for index, cfg in enumerate(configs, start=1):
        prefix = f"{a.checkpoint}_L{cfg['length']}"; forward = root / f"{prefix}_forward.npz"; reverse = root / f"{prefix}_reverse_complement.npz"
        target = output / f"{prefix}_{cfg['representation']}_{cfg['orientation']}"
        cmd = [sys.executable, "-m", "methylation_predictor.genomic_encoder.ntv3_prior", "--input", a.input, "--embeddings", str(forward), "--output-dir", str(target), "--representation", cfg["representation"], "--split", a.split, "--block-size", str(a.block_size), "--device", a.device]
        if cfg["orientation"] == "rc_average": cmd += ["--reverse-embeddings", str(reverse)]
        if a.phase == "screen": cmd += ["--validation-only"]
        print(f"[NTv3 probes] {index}/{len(configs)}: {a.checkpoint} L={cfg['length']} {cfg['representation']} {cfg['orientation']}", flush=True)
        subprocess.run(cmd, check=True); metrics = json.loads((target / "metrics.json").read_text())
        entries.append({**cfg, "output_dir": str(target), "validation_mse": metrics["validation"]["mlp_ensemble"]["mse"]})
    entries.sort(key=lambda x: x["validation_mse"])
    (output / "validation_leaderboard.json").write_text(json.dumps({"selection_metric": "validation MSE, beta-space", "entries": entries}, indent=2) + "\n")


if __name__ == "__main__": main()
