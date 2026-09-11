#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


ROOT = Path.cwd()
EXPECTED_BRANCH = "refactor/repo-v2-2026-09"


def require_branch() -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    if branch != EXPECTED_BRANCH:
        raise SystemExit(
            f"expected branch {EXPECTED_BRANCH!r}, got {branch!r}"
        )


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected exactly one match, found {count}\\n"
            f"--- needle ---\\n{old[:900]}"
        )
    path.write_text(text.replace(old, new, 1))


def remove_function_block(
    path: Path,
    function_name: str,
    next_function_name: str,
) -> None:
    text = path.read_text()
    start_marker = f"def {function_name}("
    end_marker = f"def {next_function_name}("
    start = text.find(start_marker)
    if start < 0:
        raise SystemExit(f"{path}: missing {start_marker}")
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit(f"{path}: missing following {end_marker}")
    path.write_text(text[:start] + text[end:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    require_branch()

    deletions = [
        ROOT / "src/methylation_predictor/models.py",
        ROOT / "configs/models/arch_shared_backbone",
        ROOT / "configs/models/rna_methylation_shared_backbone.yaml",
        ROOT / "configs/models/rna_methylation_locus_attention.yaml",
        ROOT / "configs/models/baselines/baseline_global_rna_shift.yaml",
        ROOT / "configs/models/baselines/baseline_mlp_rna_cpg.yaml",
        ROOT / "configs/models/baselines/baseline_bilinear_rna_cpg.yaml",
        ROOT / "configs/models/rna_encoder_comparison/enc_gene_pathway.yaml",
        ROOT / "configs/models/rna_encoder_comparison/enc_frozen_bulkformer_147m.yaml",
        ROOT / "tests/test_architecture_variants.py",
        ROOT / "tests/test_query_representation.py",
        ROOT / "tests/test_gene_pathway_encoder.py",
    ]

    if args.dry_run:
        print(
            '''Phase 4b1 will:
  - delete the retired src/methylation_predictor/models.py model family
  - delete arch_shared_backbone architecture-search recipes
  - delete superseded shared-backbone reference recipes
  - delete superseded non-functional baseline/RNA-comparator recipes
  - delete tests that only exercise the retired model family
  ~ remove root-level lazy compatibility exports for FeatureFusion*
  ~ cut LocusCLSJointTrainer model dispatch to paper-facing functional models
  ~ remove the retired direct-beta/residual auxiliary training branch
  ~ make optimizer construction independent of retired raw-branch modules
  ~ update public/refactor contract tests
  ~ update BulkRNABert recipe documentation
  ~ mark the legacy model family as removed in README/repository scope

This phase intentionally leaves some now-dead legacy config/CLI fields in
config.py and the trainer signature. Phase 4b2 will prune those fields after
the model-family deletion is regression-tested.'''
        )
        for p in deletions:
            print(f"  {'exists' if p.exists() else 'MISSING'}  {p}")
        return 0

    missing = [str(p) for p in deletions if not p.exists()]
    if missing:
        raise SystemExit(
            "expected phase4b1 targets missing; refusing a partial delete:\\n  "
            + "\\n  ".join(missing)
        )

    for p in deletions:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

    package_init = ROOT / "src/methylation_predictor/__init__.py"
    package_init.write_text(
        '''# Public API for the paper-facing MethylPredictor models.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .modeling import IterativeRetrievalPredictor, SingleRetrievalPredictor

__all__ = [
    "SingleRetrievalPredictor",
    "IterativeRetrievalPredictor",
]


def __getattr__(name: str):
    if name in __all__:
        from . import modeling

        return getattr(modeling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
'''
    )

    trainer = ROOT / "src/methylation_predictor/rna_training/locus_cls_trainer.py"
    text = trainer.read_text()

    old_import = '''from ..models import (
    FeatureFusionArchitectureVariantModel,
    FeatureFusionLocusCLSModel,
    feature_fusion_variant_label,
    is_architecture_variant,
)
'''
    if text.count(old_import) != 1:
        raise SystemExit(
            "trainer: expected exactly one retired models import block, found "
            f"{text.count(old_import)}"
        )
    text = text.replace(old_import, "", 1)

    old_loss_import = (
        "from ..losses import beta_nll_term, locus_correlation_loss, "
        "masked_mean, sample_correlation_loss, "
        "within_locus_centered_mse_loss"
    )
    new_loss_import = (
        "from ..losses import locus_correlation_loss, masked_mean, "
        "sample_correlation_loss, within_locus_centered_mse_loss"
    )
    if text.count(old_loss_import) != 1:
        raise SystemExit("trainer: loss import no longer matches expected state")
    text = text.replace(old_loss_import, new_loss_import, 1)

    start_marker = "        # An architecture-novelty recipe"
    end_marker = "        self.train_model = ("
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise SystemExit("trainer: could not locate legacy model-dispatch block")

    dispatch = '''        # Paper-facing functional-locus models only. The historical
        # FeatureFusion shared-backbone family has been removed from the repo.
        allowed_variants = {
            "mas_concat_v3_purecontext",
            "mas_concat_v4_iterative",
            "functional_rna_encoder_comparison",
            *BASELINE_VARIANTS,
        }
        if not self.functional_only or self.functional is None:
            raise ValueError(
                "paper-facing RNA training requires --functional-only plus "
                "--functional-atlas and --annotation-cache"
            )
        if self.functional_fusion_variant not in allowed_variants:
            raise ValueError(
                "unsupported paper-facing functional_fusion_variant "
                f"{self.functional_fusion_variant!r}; expected one of "
                f"{sorted(allowed_variants)}"
            )
        if residual_aux_weight != 0.0:
            raise ValueError(
                f"{self.functional_fusion_variant} requires "
                "residual_aux_weight=0"
            )
        if self.aux_weight < 0.0:
            raise ValueError("aux_weight must be non-negative")
        if self.aux_weight != 0.0 and not use_mean_branch:
            raise ValueError(
                f"{self.functional_fusion_variant} cannot use a nonzero "
                "aux_weight when use_mean_branch=false"
            )
        if self.query_source != "ntv3":
            raise ValueError(
                "legacy query_source variants were removed with the "
                "shared-backbone model family"
            )
        if self.raw_lr_multiplier != 1.0:
            raise ValueError(
                "raw_lr_multiplier was a shared-backbone ablation and is "
                "not supported by paper-facing functional models"
            )

        final_regressor_dropout = float(
            self.recipe.raw.get("locus_cls", {}).get(
                "final_regressor_dropout", 0.0
            )
        )

        if self.functional_fusion_variant in BASELINE_VARIANTS:
            self.architecture_label = self.functional_fusion_variant
            self.model = FunctionalBaselinePredictor(
                self.rna.values.shape[1],
                self.recipe.model,
                variant=self.functional_fusion_variant,
                final_regressor_dropout=final_regressor_dropout,
                use_mean_proxy=use_mean_branch,
            ).to(self.device)
        elif (
            self.functional_fusion_variant
            == "functional_rna_encoder_comparison"
        ):
            source = self.recipe.model.encoder.frozen_embedding_source
            suffix = (
                f"-{source}"
                if source
                else f"-{self.recipe.model.encoder.kind}"
            )
            self.architecture_label = (
                "functional_rna_encoder_comparison" + suffix
            )
            self.model = RNAEncoderComparisonPredictor(
                self.rna.values.shape[1],
                self.recipe.model,
                final_regressor_dropout=final_regressor_dropout,
                use_mean_proxy=use_mean_branch,
            ).to(self.device)
        else:
            candidate_cls = (
                IterativeRetrievalPredictor
                if self.functional_fusion_variant
                == "mas_concat_v4_iterative"
                else SingleRetrievalPredictor
            )
            self.architecture_label = (
                f"functional_concat_{self.functional_fusion_variant}"
            )
            self.model = candidate_cls(
                self.rna.values.shape[1],
                self.recipe.model,
                final_regressor_dropout=final_regressor_dropout,
                use_mean_proxy=use_mean_branch,
            ).to(self.device)
'''
    text = text[:start] + dispatch + text[end:]
    trainer.write_text(text)

    text = trainer.read_text()
    top_start = text.find("def _direct_beta_loss(")
    class_start = text.find("class LocusCLSJointTrainer:", top_start)
    if top_start < 0 or class_start < 0:
        raise SystemExit("trainer: could not locate _direct_beta_loss block")
    trainer.write_text(text[:top_start] + text[class_start:])

    text = trainer.read_text()
    mean_start = text.find("    def _mean_aux_loss(")
    step_start = text.find("    def _step(", mean_start)
    if mean_start < 0 or step_start < 0:
        raise SystemExit(
            "trainer: could not locate legacy mean/residual aux loss block"
        )
    trainer.write_text(text[:mean_start] + text[step_start:])

    text = trainer.read_text()
    step_start = text.find("    def _step(")
    functional_start = text.find("    def _functional_kwargs(", step_start)
    if step_start < 0 or functional_start < 0:
        raise SystemExit("trainer: could not locate _step block")

    step_method = '''    def _step(
        self,
        pool: TrainingPool,
        sample_ids,
        cpg_ids,
        rna_cpu,
        emb_cpu,
        beta_cpu,
        functional_cpu,
        finite_count,
    ):
        if finite_count == 0:
            return None

        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        compute_start = torch.cuda.Event(enable_timing=True)

        h2d_start.record()
        rna_x = rna_cpu.to(
            self.device, non_blocking=True
        ).float()
        emb = emb_cpu.to(
            self.device, non_blocking=True
        ).float()
        beta = beta_cpu.to(
            self.device, non_blocking=True
        )
        functional_kwargs = self._functional_kwargs(functional_cpu)
        h2d_end.record()
        compute_start.record()

        with self._autocast():
            out = self.train_model(
                rna_x,
                emb,
                **functional_kwargs,
            )
            loss_cfg = loss_config_for_source(
                self.recipe.loss,
                pool.name,
                self.recipe.structured_loss_sources,
            )
            total, pieces = self._mas_concat_loss(
                out,
                beta,
                cpg_ids,
                loss_cfg,
            )

        pieces = {
            **pieces,
            "total_loss": total.detach(),
        }
        return (
            total,
            pieces,
            h2d_start,
            h2d_end,
            compute_start,
        )

'''
    trainer.write_text(
        text[:step_start] + step_method + text[functional_start:]
    )

    text = trainer.read_text()
    old_optimizer = '''        if self.raw_lr_multiplier != 1.0:
            raw_params = [
                *self.model.raw_branch.parameters(), *self.model.fusion.parameters(),
                *self.model.rna_product.parameters(), *self.model.locus_product.parameters(),
                *self.model.residual_head.parameters(),
            ]
            raw_param_ids = {id(p) for p in raw_params}
            other_params = [p for p in self.model.parameters() if id(p) not in raw_param_ids]
            groups = [
                {"params": other_params, "lr": cfg.learning_rate},
                {"params": raw_params, "lr": cfg.learning_rate * self.raw_lr_multiplier},
            ]
        else:
            groups = list(self.model.parameters())
            opt_kwargs["lr"] = cfg.learning_rate
'''
    new_optimizer = '''        groups = list(self.model.parameters())
        opt_kwargs["lr"] = cfg.learning_rate
'''
    if text.count(old_optimizer) != 1:
        raise SystemExit(
            "trainer: expected legacy optimizer grouping block once, found "
            f"{text.count(old_optimizer)}"
        )
    trainer.write_text(text.replace(old_optimizer, new_optimizer, 1))

    public_test = ROOT / "tests/test_public_api.py"
    text = public_test.read_text()
    old = '''def test_historical_root_models_remain_lazy_compatibility_only():
    assert "FeatureFusionLocusCLSModel" not in mp.__all__
    assert "FeatureFusionArchitectureVariantModel" not in mp.__all__

    assert mp.FeatureFusionLocusCLSModel.__name__ == "FeatureFusionLocusCLSModel"
    assert (
        mp.FeatureFusionArchitectureVariantModel.__name__
        == "FeatureFusionArchitectureVariantModel"
    )


'''
    new = '''def test_historical_root_models_are_removed():
    assert "FeatureFusionLocusCLSModel" not in mp.__all__
    assert "FeatureFusionArchitectureVariantModel" not in mp.__all__
    assert not hasattr(mp, "FeatureFusionLocusCLSModel")
    assert not hasattr(mp, "FeatureFusionArchitectureVariantModel")


'''
    if text.count(old) != 1:
        raise SystemExit("test_public_api.py: compatibility test block changed")
    public_test.write_text(text.replace(old, new, 1))

    contract = ROOT / "tests/test_refactor_candidate_contract.py"
    remove_function_block(
        contract,
        "test_program_token_encoder_matches_legacy_token_generation_exactly",
        "test_candidate_checkpoint_loader_filters_only_known_dead_rna_attention_keys",
    )
    remove_function_block(
        contract,
        "test_historical_candidate_classes_are_no_longer_part_of_models_module",
        "test_j0_recipe_contract_is_frozen_for_the_refactor",
    )

    bulkrna = ROOT / "scripts/prepare_bulkrnabert_embeddings.py"
    replace_once(
        bulkrna,
        "``configs/models/arch_shared_backbone/enc_frozen_embedding_bulkrnabert.yaml``)",
        "``configs/models/rna_encoder_comparison/functional_bulkrnabert.yaml``)",
    )

    readme = ROOT / "README.md"
    replace_once(
        readme,
        '''This branch is being reduced from the original research-history repository into
the paper reproducibility repository.  The current core model and functional
input path are already paper-facing.  Some legacy shared-backbone code remains
temporarily because surviving baselines and RNA-encoder comparators still
depend on it; those comparators will be migrated before the legacy model family
is removed.
''',
        '''This branch is being reduced from the original research-history repository into
the paper reproducibility repository. The core model, internal baselines, and
retained RNA-encoder comparators now all use the paper-facing functional-locus
path. The retired FeatureFusion/shared-backbone model family has been removed;
remaining cleanup is limited to transitional trainer/config/data plumbing.
''',
    )

    replace_once(
        readme,
        '''Current priorities after the public/core cutover are:

1. migrate claim-driven mean-proxy ablations onto the functional model;
2. migrate paper baselines onto the same functional locus representation;
3. migrate any retained RNA-encoder comparisons;
4. remove the remaining shared-backbone architecture-search infrastructure;
5. consolidate data preparation and paper-table reproduction commands.
''',
        '''Current priorities after the functional/core cutover are:

1. prune transitional shared-backbone-era trainer/config fields;
2. remove the legacy genomic feature-cache requirement from functional runs;
3. consolidate data preparation and paper-table reproduction commands;
4. perform the final repository and reproducibility audit.
''',
    )

    scope = ROOT / "docs/REPOSITORY_SCOPE.md"
    with scope.open("a") as handle:
        handle.write(
            '''
## Phase 4b1: retired model family removed

The historical `models.py` FeatureFusion/shared-backbone implementation has
been deleted. All surviving paper-facing trainable RNA models now live under
`methylation_predictor.modeling`:

```text
SingleRetrievalPredictor
IterativeRetrievalPredictor
FunctionalBaselinePredictor
RNAEncoderComparisonPredictor
```

Architecture-search recipes, superseded shared-backbone reference recipes,
and tests whose only purpose was to exercise the retired model family were
also removed. Historical metrics remain in the version-controlled result
ledgers for provenance, and the deleted implementation remains available in
Git history.

`LocusCLSJointTrainer` now dispatches only to the functional paper-facing
models and uses only the functional objective path. Some obsolete constructor,
checkpoint-metadata, and config fields are intentionally left for phase 4b2
so this large model-family deletion can be regression-tested independently.
'''
        )

    forbidden_paths = [
        ROOT / "src/methylation_predictor/models.py",
        ROOT / "configs/models/arch_shared_backbone",
        ROOT / "tests/test_architecture_variants.py",
        ROOT / "tests/test_query_representation.py",
    ]
    present = [str(p) for p in forbidden_paths if p.exists()]
    if present:
        raise SystemExit(
            "phase4b1 deletion guard failed:\\n  " + "\\n  ".join(present)
        )

    stale = []
    for base in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
        for path in base.rglob("*.py"):
            txt = path.read_text()
            if (
                "methylation_predictor.models" in txt
                or "from ..models import" in txt
                or "from .models import" in txt
            ):
                stale.append(str(path))
    if stale:
        raise SystemExit(
            "live imports of deleted models.py remain:\\n  "
            + "\\n  ".join(stale)
        )

    print("Phase 4b1 applied.")
    print()
    print("Run focused tests:")
    print(
        "  pytest -q tests/test_public_api.py "
        "tests/test_refactor_candidate_contract.py "
        "tests/test_functional_baselines.py "
        "tests/test_functional_rna_encoder_comparison.py "
        "tests/test_mean_contribution_suite.py"
    )
    print()
    print("Then:")
    print("  python -m compileall -q src scripts")
    print("  pytest -q")
    print()
    print("Inspect:")
    print("  git status")
    print("  git diff --stat")
    print("  git diff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
