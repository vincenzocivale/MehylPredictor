#!/usr/bin/env python3
"""Real coverage of a scope's `val_cpg_x_train_sample` held-out CpGs against the Illumina
probe-ID / hg38-position crosswalk shared by CpGPT (DNA-embedding lookup) and MethylGPT
(fixed probe vocabulary) -- see `methylation_predictor.benchmark.foundation_models.crosswalk`
for where this crosswalk comes from and the offset convention it encodes.

Needs only this repo's own `methyl-predictor` conda env (numpy/pandas/sqlite3, all stdlib or
already-installed) -- no CpGPT/MethylGPT/DeepCpG environment required, since this only checks
whether the *lookup* resolves, not the model forward pass.

    conda activate methyl-predictor
    python scripts/benchmark_foundation_models/check_crosswalk_coverage.py \\
        --scope chr1 \\
        --table5-protocol-root /dune/.../methylprophet_table5_tcga_chr1/table5_protocol \\
        --array-cpg-registry /dune/.../cpg/registries/array_cpg_map.parquet \\
        --cpgpt-human-dependencies external/checkpoints/cpgpt_human_dependencies \\
        --output results/reference/appendix/foundation_models/chr1_crosswalk_coverage.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from methylation_predictor.benchmark.foundation_models.crosswalk import IlluminaCrosswalk  # noqa: E402
from methylation_predictor.benchmark.foundation_models.protocol import (  # noqa: E402
    load_val_cpg_positions_chr1,
    load_train_sample_context_ids_chr1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, choices=["chr1", "chr123"])
    parser.add_argument("--table5-protocol-root", required=True)
    parser.add_argument("--array-cpg-registry", required=True)
    parser.add_argument("--cpgpt-human-dependencies", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    positions = load_val_cpg_positions_chr1(args.table5_protocol_root, args.array_cpg_registry)
    context_sample_idx = load_train_sample_context_ids_chr1(args.table5_protocol_root)

    crosswalk = IlluminaCrosswalk.load(args.cpgpt_human_dependencies)
    coverage = crosswalk.coverage(positions)

    result = {
        "schema_version": 1,
        "scope": args.scope,
        "view": "val_cpg_x_train_sample",
        "n_context_samples": len(context_sample_idx),
        "crosswalk_source": str(Path(args.cpgpt_human_dependencies).resolve()),
        **coverage,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
