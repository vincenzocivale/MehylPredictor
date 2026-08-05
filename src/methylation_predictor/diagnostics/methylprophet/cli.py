"""CLI for analyses of the read-only MethylProphet dependency."""
from __future__ import annotations

import argparse
from collections.abc import Callable


def _commands() -> dict[str, Callable[[], None]]:
    from . import comparability_audit, empirical_hybrid, export_contract, gene_intervention, locus_decomposition, mean_rna_baseline, released_predictions, stage_d3_matrix_v1, stage_d_matched
    return {
        "released-audit": released_predictions.main,
        "locus-decomposition": lambda: locus_decomposition.run(locus_decomposition.parser().parse_args()),
        "gene-intervention": gene_intervention.main,
        "empirical-hybrid": empirical_hybrid.main,
        "mean-rna-baseline": mean_rna_baseline.main,
        "export": export_contract.main,
        "stage-d-matched": stage_d_matched.main,
        "comparability-audit": comparability_audit.main,
        "stage-d3-matrix-v1": stage_d3_matrix_v1.main,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(_commands()))
    args, remaining = parser.parse_known_args()
    import sys
    sys.argv = [f"{parser.prog} {args.command}", *remaining]
    _commands()[args.command]()


if __name__ == "__main__":
    main()
