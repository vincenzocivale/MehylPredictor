from pathlib import Path
import copy

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text())


def _normalized(raw: dict) -> dict:
    out = copy.deepcopy(raw)
    # Experiment-specific W&B labels are provenance, not a causal factor.
    out.pop("tracking", None)
    return out


def test_mean_contribution_recipes_isolate_only_intended_factors():
    full = _normalized(_load("configs/models/rna_methylation_locus_attention.yaml"))
    no_aux = _normalized(_load("configs/models/mean_contribution/no_mean_supervision.yaml"))
    no_branch = _normalized(_load("configs/models/mean_contribution/no_mean_branch.yaml"))

    assert full["model"]["encoder"]["kind"] == "locus_attention"
    assert full["model"]["encoder"]["n_programs"] == 64
    assert full["locus_cls"]["use_raw_product"] is False
    assert full["training"]["learning_rate"] == 2.0e-4
    assert full["training"]["scheduler"] == "cosine_warmup"

    expected = copy.deepcopy(full)
    expected["locus_cls"]["aux_weight"] = 0.0
    assert no_aux == expected

    expected = copy.deepcopy(full)
    expected["locus_cls"]["use_mean_branch"] = False
    expected["locus_cls"]["aux_weight"] = 0.0
    assert no_branch == expected
