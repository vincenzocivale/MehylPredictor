"""CLI for independent genomic-encoder experiments."""
from __future__ import annotations

import argparse
from collections.abc import Callable


def _commands() -> dict[str, Callable[[], None]]:
    from . import downstream_evaluation, ntv3_embeddings, ntv3_prior, ntv3_variability, reference_neighborhood, static_features, static_prior
    return {
        "build-features": static_features.main,
        "static-baselines": static_prior.main,
        "reference-neighborhood": reference_neighborhood.main,
        "extract-ntv3": ntv3_embeddings.main,
        "prior-probe": ntv3_prior.main,
        "variability-probe": ntv3_variability.main,
        "downstream-evaluation": downstream_evaluation.main,
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
