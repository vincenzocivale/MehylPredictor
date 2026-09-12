#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import subprocess
import yaml

ROOT = Path.cwd()
EXPECTED_BRANCH = "refactor/repo-v2-2026-09"


def main() -> int:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    if branch != EXPECTED_BRANCH:
        raise SystemExit(
            f"expected branch {EXPECTED_BRANCH!r}, got {branch!r}"
        )

    surface = ROOT / "configs/experiment_surface.yaml"
    if not surface.is_file():
        raise SystemExit(
            "Phase 5a manifest is missing: commit/apply phase 5a first"
        )

    audit_path = ROOT / "configs/config_field_audit.yaml"
    doc_path = ROOT / "docs/CONFIG_FIELD_AUDIT.md"
    test_path = ROOT / "tests/test_config_field_audit.py"

    for path in (audit_path, doc_path, test_path):
        if path.exists():
            raise SystemExit(f"{path} already exists; refusing overwrite")

    audit = {
        "schema_version": 1,
        "scope": "RNA architecture/loss configuration only",
        "deferred": [
            "TrainingConfig",
            "TrackingConfig",
        ],
        "encoder_config": {
            "keep": [
                "kind",
                "latent_dim",
                "layer_norm",
                "hidden_dim",
                "dropout",
                "n_programs",
                "program_dim",
                "n_heads",
                "n_blocks",
                "mlp_ratio",
                "frozen_embedding_source",
                "pathway_membership_path",
                "pathway_dim1",
                "pathway_dim2",
            ]
        },
        "model_config": {
            "keep": [
                "encoder",
                "functional_fusion_variant",
            ],
            "compatibility_only": [
                "interaction",
                "trunk",
                "axial",
                "beta_likelihood_head",
                "zero_init_residual",
                "variance_normalized_residual",
                "use_prior_anchor",
            ],
        },
        "loss_config": {
            "paper_live": [
                "beta_mse_weight",
                "locus_pearson_weight",
                "locus_min_observed_samples",
                "locus_pearson_epsilon",
                "locus_pearson_min_target_std",
            ],
            "runtime_supported_not_selected_by_frozen_surface": [
                "sample_pearson_weight",
                "sample_pearson_min_observed_cpgs",
                "sample_pearson_epsilon",
                "locus_centered_mse_weight",
            ],
            "legacy_not_consumed_by_live_trainer": [
                "residual_huber_weight",
                "residual_huber_delta",
                "shrinkage_weight",
                "standardized_residual_huber_weight",
                "standardized_residual_huber_delta",
                "standardized_shrinkage_weight",
                "sigma_min",
                "beta_nll_weight",
                "beta_nll_epsilon",
                "concentration_min",
            ],
        },
        "locus_cls": {
            "keep": [
                "use_mean_branch",
                "aux_weight",
                "final_regressor_dropout",
            ],
            "compatibility_only": [
                "use_fusion_product",
                "use_raw_product",
                "product_mlp",
                "include_raw_rna",
                "include_raw_cpg",
                "fusion_init_std",
                "query_source",
                "residual_aux_weight",
                "raw_lr_multiplier",
                "trunk_hidden_dim",
                "bottleneck_dim",
                "trunk_dropout",
            ],
        },
        "next_prune_gate": {
            "required_condition": (
                "final architecture selected and protected J-series surface "
                "released from resume compatibility"
            ),
            "then_remove": [
                "ModelConfig compatibility_only fields/classes",
                "LossConfig legacy fields",
                "runtime-only loss fields if not promoted into retained paper recipes",
                "_retired_locus_compat and compatibility-only locus_cls keys",
            ],
        },
    }

    audit_path.write_text(
        yaml.safe_dump(audit, sort_keys=False, width=100)
    )
    doc_path.write_text('# RNA configuration-field audit\n\nPhase 5b inventories the remaining RNA architecture/loss configuration surface\nwithout changing any model, recipe, checkpoint, or experiment script.\n\nThe authoritative machine-readable audit is\n[`configs/config_field_audit.yaml`](../configs/config_field_audit.yaml).\n\n## Scope\n\nThis audit covers:\n\n- `EncoderConfig`;\n- RNA-relevant fields of `ModelConfig`;\n- `LossConfig`;\n- `locus_cls` recipe keys consumed by `RNAMethylationTrainer`.\n\n`TrainingConfig` and `TrackingConfig` are deliberately deferred because they\nare shared with training infrastructure and/or the `cpg_statistics` workflow.\n\n## Findings\n\n### Keep: encoder configuration\n\nThe encoder fields remain live because the paper-facing RNA-encoder comparison\nuses multiple encoder families: program tokens, bottleneck MLP, gene/pathway,\nBulkFormer and BulkRNABert.\n\n### Keep: current model selector\n\nThe live RNA model surface needs only:\n\n- `model.encoder`;\n- `model.functional_fusion_variant`.\n\nThe factory dispatches exclusively from these fields.\n\n### Compatibility-only model fields\n\nThe following shared-backbone-era `ModelConfig` fields are parsed but do not\nparticipate in any current predictor forward pass:\n\n- `interaction`;\n- `trunk`;\n- `axial`;\n- `beta_likelihood_head`;\n- `zero_init_residual`;\n- `variance_normalized_residual`;\n- `use_prior_anchor`.\n\nThey remain temporarily because current frozen recipes still contain some of\nthem and resume compatibility compares resolved recipe metadata.\n\n### Loss surface\n\nThe current paper-facing trainer directly consumes:\n\n- `beta_mse_weight`;\n- `locus_pearson_weight`;\n- `locus_min_observed_samples`;\n- `locus_pearson_epsilon`;\n- `locus_pearson_min_target_std`.\n\nThe code still supports, but the frozen experiment surface does not currently\nselect:\n\n- `sample_pearson_weight`;\n- `sample_pearson_min_observed_cpgs`;\n- `sample_pearson_epsilon`;\n- `locus_centered_mse_weight`.\n\nThese are candidates for removal once the architecture/experiment decision is\nlocked.\n\nThe residual/shrinkage/Beta-NLL fields are legacy fields from retired model\ngenerations. The live RNA trainer does not consume them.\n\n### `locus_cls` keys\n\nStill live:\n\n- `use_mean_branch`;\n- `aux_weight`;\n- `final_regressor_dropout`.\n\nCompatibility-only:\n\n- `use_fusion_product`;\n- `use_raw_product`;\n- `product_mlp`;\n- `include_raw_rna`;\n- `include_raw_cpg`;\n- `fusion_init_std`;\n- `query_source`;\n- `residual_aux_weight`;\n- `raw_lr_multiplier`;\n- `trunk_hidden_dim`;\n- `bottleneck_dim`;\n- `trunk_dropout`.\n\nThe compatibility-only keys are retained solely so pre-refactor resolved\nconfigs can still be compared during `--resume`.\n\n## Removal order\n\nDo not delete compatibility fields while protected J-series runs may need\nresume.\n\nAfter the final architecture is selected:\n\n1. promote the selected paper architecture;\n2. delete non-retained J-series recipes/launchers;\n3. rewrite `main.yaml` and retained paper recipes without compatibility keys;\n4. remove compatibility parsing from `rna_training/config.py` and\n   `_retired_locus_compat`;\n5. remove the dead `ModelConfig`/`LossConfig` fields;\n6. run checkpoint/resume regression tests;\n7. only then consider renaming the historical `locus_cls_joint` storage key.\n')
    test_path.write_text('from __future__ import annotations\n\nfrom pathlib import Path\n\nimport yaml\n\n\nROOT = Path(__file__).resolve().parents[1]\nAUDIT = ROOT / "configs/config_field_audit.yaml"\n\n\ndef _audit() -> dict:\n    return yaml.safe_load(AUDIT.read_text())\n\n\ndef test_phase5a_surface_exists_before_config_pruning():\n    assert (ROOT / "configs/experiment_surface.yaml").is_file()\n\n\ndef test_live_modeling_does_not_use_compatibility_model_fields():\n    audit = _audit()\n    fields = audit["model_config"]["compatibility_only"]\n\n    modeling = "\\n".join(\n        path.read_text()\n        for path in (ROOT / "src/methylation_predictor/modeling").glob("*.py")\n    )\n    trainer = (\n        ROOT\n        / "src/methylation_predictor/rna_training/"\n        "rna_methylation_trainer.py"\n    ).read_text()\n\n    for field in fields:\n        assert f"config.{field}" not in modeling\n        assert f"self.recipe.model.{field}" not in trainer\n\n\ndef test_live_trainer_does_not_consume_legacy_loss_fields():\n    audit = _audit()\n    legacy = audit["loss_config"]["legacy_not_consumed_by_live_trainer"]\n    trainer = (\n        ROOT\n        / "src/methylation_predictor/rna_training/"\n        "rna_methylation_trainer.py"\n    ).read_text()\n\n    for field in legacy:\n        assert f"loss_cfg.{field}" not in trainer\n        assert f"self.recipe.loss.{field}" not in trainer\n\n\ndef test_runtime_only_loss_fields_are_not_selected_by_frozen_recipes():\n    audit = _audit()\n    fields = set(\n        audit["loss_config"]["runtime_supported_not_selected_by_frozen_surface"]\n    )\n    surface = yaml.safe_load(\n        (ROOT / "configs/experiment_surface.yaml").read_text()\n    )\n\n    recipe_paths = []\n    for role in ("paper_facing", "compatibility", "protected_research"):\n        recipe_paths.extend(\n            record["path"]\n            for record in surface["models"].get(role, [])\n            if record["path"].startswith("configs/models/")\n        )\n\n    selected = set()\n    for relative in recipe_paths:\n        path = ROOT / relative\n        raw = yaml.safe_load(path.read_text()) or {}\n        selected.update((raw.get("loss") or {}).keys())\n\n    assert fields.isdisjoint(selected)\n\n\ndef test_compat_locus_keys_match_trainer_compat_defaults():\n    audit = _audit()\n    expected = set(audit["locus_cls"]["compatibility_only"])\n\n    trainer = (\n        ROOT\n        / "src/methylation_predictor/rna_training/"\n        "rna_methylation_trainer.py"\n    ).read_text()\n\n    start = trainer.index("_RETIRED_LOCUS_DEFAULTS = {")\n    end = trainer.index("\\n}\\n", start) + 2\n    block = trainer[start:end]\n\n    for key in expected:\n        assert f\'"{key}"\' in block\n\n\ndef test_keep_sets_and_remove_sets_are_disjoint():\n    audit = _audit()\n\n    for section in ("model_config", "loss_config", "locus_cls"):\n        groups = audit[section]\n        seen = set()\n        for values in groups.values():\n            values = set(values)\n            assert seen.isdisjoint(values)\n            seen.update(values)\n')

    scope = ROOT / "docs/REPOSITORY_SCOPE.md"
    with scope.open("a") as handle:
        handle.write(
            "\n## Phase 5b: configuration-field audit\n\n"
            "The remaining RNA architecture/loss config fields are classified "
            "in `configs/config_field_audit.yaml`. No fields are removed in "
            "this phase because protected J-series runs remain resumable. "
            "Compatibility-only ModelConfig, LossConfig and `locus_cls` keys "
            "now have an explicit deletion gate tied to final architecture "
            "selection.\n"
        )

    print("Phase 5b audit applied.")
    print()
    print("Created:")
    print("  configs/config_field_audit.yaml")
    print("  docs/CONFIG_FIELD_AUDIT.md")
    print("  tests/test_config_field_audit.py")
    print()
    print("Run:")
    print("  pytest -q tests/test_config_field_audit.py")
    print("  python -m compileall -q src scripts")
    print("  pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
