!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected exactly one match, found {count}\n"
            f"--- needle ---\n{old[:700]}"
        )
    path.write_text(text.replace(old, new, 1))


# The destructive part of phase4a already ran before the stale-reference
# guard stopped the script. Refuse to proceed if that assumption is false.
expected_removed = [
    ROOT / "src/methylation_predictor/explainability",
    ROOT / "scripts/explain.py",
    ROOT / "tests/test_explainability.py",
    ROOT / "docs/EXPLAINABILITY.md",
]
still_present = [str(p) for p in expected_removed if p.exists()]
if still_present:
    raise SystemExit(
        "phase4a does not appear to be in the expected partial-applied state; "
        "these targets still exist:\n  " + "\n  ".join(still_present)
    )

# 1. Remove the last live source-code reference.
models = ROOT / "src/methylation_predictor/models.py"
replace_once(
    models,
    """    Side benefit: ``attention`` is returned per forward, so locus -> gene-program
    weights are directly plottable, complementing the Expected-Gradients
    attribution in scripts/explain.py.
""",
    """    Side benefit: ``attention`` is returned per forward, so locus -> gene-program
    weights are directly plottable for diagnostic analysis.
""",
)

# 2. The phase4a documentation block itself listed the deleted file paths,
# which made the stale-reference guard reject its own audit note. Keep the
# scope decision, but do not retain dead-path references.
scope = ROOT / "docs/REPOSITORY_SCOPE.md"
old_block = """
## Phase 4a: explainability removed

Checkpoint explainability is intentionally out of scope for the paper-facing
reproducibility repository. The previous Expected/Integrated-Gradients
implementation targeted the retired shared-backbone `residual_logit` and was
not part of the paper's central claims.

The following were removed rather than migrated:

```text
src/methylation_predictor/explainability/
scripts/explain.py
tests/test_explainability.py
docs/EXPLAINABILITY.md
```

The public workflow is therefore limited to data preparation, training,
hyperparameter selection where retained, and evaluation. Historical
explainability code remains available through Git history.
"""

new_block = """
## Phase 4a: explainability removed

Checkpoint explainability is intentionally out of scope for the paper-facing
reproducibility repository. The previous Expected/Integrated-Gradients
implementation targeted the retired shared-backbone `residual_logit` and was
not part of the paper's central claims.

The implementation, CLI, tests, and dedicated documentation were removed
rather than migrated. The public workflow is therefore limited to data
preparation, training, hyperparameter selection where retained, and
evaluation. Historical explainability code remains available through Git
history.
"""

replace_once(scope, old_block, new_block)

# 3. Final stale-reference guard.
roots = [
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "docs",
    ROOT / "README.md",
    ROOT / "CLAUDE.md",
]
needles = (
    "EXPLAINABILITY.md",
    "scripts/explain.py",
    "methylation_predictor.explainability",
    "SampleGeneExplainer",
    "integrated_gradients_rna",
)

stale = []
for base in roots:
    paths = [base] if base.is_file() else list(base.rglob("*"))
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for needle in needles:
            if needle in text:
                stale.append(f"{path}: {needle}")

if stale:
    raise SystemExit(
        "stale explainability references still remain:\n  "
        + "\n  ".join(stale)
    )

print("Phase 4a cleanup completed.")
print()
print("Run:")
print("  python -m compileall -q src scripts")
print("  pytest -q")
print()
print("Then inspect:")
print("  git status")
print("  git diff --stat")
print("  git diff")
