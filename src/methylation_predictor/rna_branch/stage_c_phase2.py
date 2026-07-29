"""Materialize the fixed Stage-C variability sampling/loss grid."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.base).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models = {
        "c0": {},
        "c1_mlp64": {"model": {"interaction": {"kind": "interaction_mlp", "mlp_hidden_dims": [64]}}},
        "c2_k4": {"model": {"interaction": {"kind": "multihead_bilinear", "hidden_dim": 64, "num_heads": 4}}},
    }
    variants = {
        "v0": {},
        "v1": {"training": {"cpg_sampling": "balanced_tertiles"}},
        "v2": {"loss": {"beta_mse_weight": 0.0, "beta_macro_weight": 1.0}},
        "v3": {"training": {"cpg_sampling": "balanced_tertiles"}, "loss": {"beta_mse_weight": 0.5, "beta_macro_weight": 0.5}},
    }
    paths = []
    index = 0
    for family, model_override in models.items():
        for variant, variant_override in variants.items():
            for seed in (17, 29, 43):
                name = f"{family}_{variant}_seed{seed}"
                config = merge(merge(base, model_override), variant_override)
                config["run_name"] = name
                config["output_dir"] = f"artifacts/rna_branch/stage_c/screening/{name}"
                config["training"]["seed"] = seed
                path = output / f"{index:02d}_{name}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                paths.append(path)
                index += 1
    (output / "manifest.txt").write_text("\n".join(map(str, paths)) + "\n")
    print(f"materialized={len(paths)}")


if __name__ == "__main__":
    main()
