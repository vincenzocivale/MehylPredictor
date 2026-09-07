#!/usr/bin/env python3
"""Pilot, end-to-end smoke test of CpGPT's real masked-recovery API on a tiny slice of
chr1 data -- validates the whole mechanism (arrow -> CpGPTDataSaver -> CpGPTDataModule ->
CpGPTTrainer.predict(predict_mode="reconstruct")) before committing CPU-hours to the full
8,260-sample / 33,885-context-CpG / 6,742-target-CpG run.

Must run under `external/cpgpt-env`'s python (has CpGPT's own hydra/lightning/torch stack;
this repo's own `methylation_predictor` package is reached via sys.path, no install needed).
Does NOT need h5py / TCGACanonicalBundle -- takes its context data from the .arrow file
already built by `prepare_cpgpt_chr1_context.py` (which does need the canonical bundle, run
separately under the main `methyl-predictor` env).

    external/cpgpt-env/bin/python scripts/benchmark_foundation_models/run_cpgpt_reconstruct_pilot.py \\
        --context-arrow derived/foundation_models/cpgpt/pilot_chr1_train_context.arrow \\
        --cpgpt-checkpoints external/checkpoints/cpgpt \\
        --cpgpt-human-dependencies external/checkpoints/cpgpt_human_dependencies \\
        --variant small \\
        --genomic-locations 1:1000037,1:1000039 \\
        --output derived/foundation_models/cpgpt/pilot_reconstruct.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "CpGPT"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-arrow", required=True)
    parser.add_argument("--cpgpt-checkpoints", required=True)
    parser.add_argument("--cpgpt-human-dependencies", required=True)
    parser.add_argument("--variant", default="small", choices=["small", "large"])
    parser.add_argument(
        "--genomic-locations",
        default=None,
        help="Comma-separated CpGPT-convention locations to reconstruct, e.g. '1:1000037,1:1000039'. "
        "Mutually exclusive with --genomic-locations-file; use the file form for large (e.g. "
        "chr123-scale) location lists -- the OS argv length limit rejects a long inline list "
        "(hit for real at chr123's 14,892 locations, 'Argument list too long').",
    )
    parser.add_argument(
        "--genomic-locations-file",
        default=None,
        help="Path to a file with one comma-separated line of locations (or newline-separated).",
    )
    parser.add_argument("--processed-dir", default=None, help="Defaults to a sibling 'processed/' dir.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=10_000)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.genomic_locations) == bool(args.genomic_locations_file):
        raise SystemExit("pass exactly one of --genomic-locations / --genomic-locations-file")
    if args.genomic_locations_file:
        text = Path(args.genomic_locations_file).read_text().strip()
        genomic_locations = [loc for line in text.splitlines() for loc in line.split(",") if loc]
    else:
        genomic_locations = args.genomic_locations.split(",")

    from methylation_predictor.benchmark.foundation_models.adapters import cpgpt_adapter as adapter

    resources = adapter.resolve_resources(args.cpgpt_checkpoints, args.variant)
    model, config = adapter.build_random_model(resources.config_path)
    adapter.load_checkpoint_into(model, resources.checkpoint_path)
    model.eval()
    print(f"Loaded CpGPT '{args.variant}' checkpoint OK.")

    from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
    from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
    from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
    from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
    from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer

    deps_dir = args.cpgpt_human_dependencies
    embedder = DNALLMEmbedder(dependencies_dir=deps_dir)
    prober = IlluminaMethylationProber(dependencies_dir=deps_dir, embedder=embedder)

    processed_dir = args.processed_dir or str(Path(args.context_arrow).with_suffix("")) + "_processed"
    saver = CpGPTDataSaver(data_paths=args.context_arrow, processed_dir=processed_dir)
    saver.process_files(prober, embedder)
    print(f"CpGPTDataSaver.process_files() OK -> {processed_dir}")

    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=deps_dir,
        batch_size=args.batch_size,
        num_workers=0,
        max_length=args.max_length,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False,
    )
    print("CpGPTDataModule built OK.")

    trainer = CpGPTTrainer()
    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        predict_mode="reconstruct",
        genomic_locations=genomic_locations,
        species="homo_sapiens",
        return_keys=["pred_meth"],
    )
    print("trainer.predict(predict_mode='reconstruct') OK.")

    from cpgpt.model.utils import m_to_beta

    pred_m = predictions["pred_meth"].float()
    pred_beta = m_to_beta(pred_m).numpy()

    import numpy as np

    np.save(args.output.with_suffix(".pred_beta.npy"), pred_beta)
    result = {
        "variant": args.variant,
        "genomic_locations": genomic_locations,
        "pred_beta_shape": list(pred_beta.shape),
        "pred_beta_sample": pred_beta[:5].tolist(),
        "pred_beta_npy": str(args.output.with_suffix(".pred_beta.npy")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
