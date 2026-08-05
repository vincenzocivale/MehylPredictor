#!/usr/bin/env python3
"""Build bilinear/concat prediction ensembles (mean over seeds) on the frozen
30k-CpG subset and compare all models (per-seed + ensemble) on the 4 panels.
Does not touch the official val_cpg x val_sample MethylProphet comparison --
that's a separate step (needs released predictions + Stage-D3-style bootstrap)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from methylation_predictor.rna_branch.metrics import evaluate_predictions

PANELS = ["in_distribution", "sample_ood", "locus_ood", "double_ood"]


def load(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bilinear-panels-dirs", nargs="+", type=Path, required=True, help="one panels dir per bilinear seed")
    p.add_argument("--concat-panels-dirs", nargs="+", type=Path, required=True, help="one panels dir per concat seed")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--panel", choices=PANELS, default=None, help="process only this panel (one process per panel to stay under the harness's process-lifetime limit); omit to do all 4 in one process")
    p.add_argument("--model", choices=["bilinear_ensemble", "concat_ensemble"], default=None,
                    help="process only this ensemble (needed for the 7304-train-sample panels: dynamic_spearman/"
                         "within_cancer_spearman are full-array O(n log n) scipy calls on ~219M elements, "
                         "measured ~108s EACH; two models sequentially exceeds the harness's process-lifetime "
                         "limit even though each alone fits)")
    args = p.parse_args()

    panels = [args.panel] if args.panel else PANELS
    result = {"panels": {}}
    for name in panels:
        bilinear = [load(d / f"panel_{name}.npz") for d in args.bilinear_panels_dirs]
        concat = [load(d / f"panel_{name}.npz") for d in args.concat_panels_dirs]
        ref = bilinear[0]
        for other, label in [(item, f"bilinear[{i}]") for i, item in enumerate(bilinear[1:], 1)] + [
            (item, f"concat[{i}]") for i, item in enumerate(concat)
        ]:
            if not np.array_equal(ref["cpg_idx"], other["cpg_idx"]) or not np.array_equal(ref["sample_idx"], other["sample_idx"]):
                raise ValueError(f"{name}: {label} does not share the frozen sample/cpg identity with the reference")

        target = ref["target"].astype(float)
        prior = ref["prior"].astype(float)
        cancer_type = ref["cancer_type"].astype(str)

        bilinear_preds = [b["prediction"].astype(float) for b in bilinear]
        concat_preds = [c["prediction"].astype(float) for c in concat]
        bilinear_ensemble = np.mean(bilinear_preds, axis=0)
        concat_ensemble = np.mean(concat_preds, axis=0)

        # Only the two NEW ensemble predictions are evaluated here -- per-seed
        # metrics were already computed (and are already on disk) by the
        # individual predict_panels_capped.py/predict_panels_frozen.py runs
        # that produced these npz files.
        models = {"bilinear_ensemble": bilinear_ensemble, "concat_ensemble": concat_ensemble}
        if args.model:
            models = {args.model: models[args.model]}
        # include_median_correlations=False: measured cost breakdown (in_distribution
        # panel, 7304 train samples x 30k frozen cpgs) showed dynamic_spearman alone
        # costs ~108s (full-array O(n log n) rank on ~219M elements); the four
        # patient/locus-axis median-correlation diagnostics add a 7304+30000-iteration
        # Python loop on top of that and reliably pushed even single-model calls past
        # the harness's process-lifetime limit. mse/skill_vs_prior/dynamic_pearson/
        # dynamic_spearman/within_cancer_pearson/within_cancer_spearman are unaffected;
        # only the four *_median fields are dropped for ensemble rows (still available
        # per-seed from the individual predict_panels_*.py runs).
        result["panels"][name] = {
            model_name: evaluate_predictions(
                target, pred, prior, cancer_type,
                include_median_correlations=False, correlation_max_n=2_000_000,
            )
            for model_name, pred in models.items()
        }

    suffix_bits = [b for b in (args.panel, args.model) if b]
    suffix = ("_" + "_".join(suffix_bits)) if suffix_bits else ""
    out_path = args.output.with_name(f"{args.output.stem}{suffix}{args.output.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {out_path}", flush=True)
    for name in panels:
        row = result["panels"][name]
        for model_name, metric in row.items():
            print(f"{name}: {model_name} skill={metric['skill_vs_prior']:.4f}", flush=True)


if __name__ == "__main__":
    main()
