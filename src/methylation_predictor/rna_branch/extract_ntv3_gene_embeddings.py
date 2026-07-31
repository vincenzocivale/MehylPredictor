#!/usr/bin/env python3
"""Extract a shard of strand-aware NTv3 TSS embeddings for the RNA gene manifest."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import model_info

from methylation_predictor.genomic_encoder.ntv3_embeddings import _forward_locus_embeddings, _model
from methylation_predictor.genomic_encoder.ntv3_prior_common import base_to_output, reverse_complement, sha256


def load_fasta_records(path: Path) -> dict[str, str]:
    opener = gzip.open if path.suffix == ".gz" else open
    records: dict[str, list[str]] = {}
    current: str | None = None
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                current = line[1:].split()[0]
                if not current.startswith("chr"):
                    current = f"chr{current}"
                records.setdefault(current, [])
            elif current is not None:
                records[current].append(line.strip().upper())
    return {key: "".join(parts) for key, parts in records.items()}


def tss_window(sequence: str, tss_1based: int, length: int) -> str:
    if length <= 0 or length % 2:
        raise ValueError("window length must be a positive even number")
    centre = tss_1based - 1
    start = centre - length // 2
    end = start + length
    left, right = max(0, -start), max(0, end - len(sequence))
    result = "N" * left + sequence[max(start, 0):min(end, len(sequence))] + "N" * right
    if len(result) != length:
        raise AssertionError("TSS window length invariant violated")
    return result


def pool_tss(output, input_length: int, radius: int) -> np.ndarray:
    values = getattr(output, "embedding", None)
    if values is None:
        raise RuntimeError("NTv3 checkpoint did not expose final embeddings")
    values = values.float()
    left_base = max(0, input_length // 2 - 1 - radius)
    right_base = min(input_length - 1, input_length // 2 + radius)
    left = base_to_output(left_base, input_length, values.shape[1])
    right = base_to_output(right_base, input_length, values.shape[1])
    return values[:, left:right + 1].mean(dim=1).detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--pool-radius", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index/count")
    if args.length % 128:
        raise ValueError("NTv3 length must be divisible by 128")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    manifest = pd.read_parquet(args.manifest).sort_values("rna_col").reset_index(drop=True)
    required = {"rna_col", "gene_idx", "matched", "chromosome", "tss", "strand"}
    if missing := required - set(manifest):
        raise ValueError(f"manifest lacks {sorted(missing)}")
    matched = manifest[manifest.matched].copy()
    shards = np.array_split(np.arange(len(matched)), args.num_shards)
    local = matched.iloc[shards[args.shard_index]].copy().reset_index(drop=True)
    if local.empty:
        raise ValueError("selected shard is empty")

    genome = load_fasta_records(Path(args.fasta))
    sequences = []
    for row in local.itertuples(index=False):
        if row.chromosome not in genome:
            raise ValueError(f"chromosome {row.chromosome!r} is absent from FASTA")
        sequence = tss_window(genome[row.chromosome], int(row.tss), args.length)
        if row.strand == "-":
            sequence = reverse_complement(sequence)
        sequences.append(sequence)

    try:
        resolved_revision = model_info(args.checkpoint, revision=args.revision).sha
    except Exception:
        # Extraction must remain usable with a local or already-cached checkpoint
        # on an offline cluster.  The requested revision and output hash remain
        # recorded even when the Hub cannot be queried.
        resolved_revision = None
    tokenizer, model = _model(args.checkpoint, args.revision, args.device, args.bf16)
    vectors: list[np.ndarray] = []
    total_batches = math.ceil(len(sequences) / args.batch_size)
    started = time.monotonic()
    for begin in range(0, len(sequences), args.batch_size):
        batch = sequences[begin:begin + args.batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(args.device)
        device_type = torch.device(args.device).type
        with torch.inference_mode(), torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=args.bf16
        ):
            output = _forward_locus_embeddings(model, input_ids, ["human"] * len(batch))
            vectors.append(pool_tss(output, args.length, args.pool_radius))
        done = begin // args.batch_size + 1
        if args.progress_every and (done % args.progress_every == 0 or done == total_batches):
            elapsed = time.monotonic() - started
            print(f"[gene-NTv3] shard={args.shard_index} {done}/{total_batches} elapsed={elapsed/60:.1f}m", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.concatenate(vectors, axis=0)
    np.savez_compressed(
        output,
        embeddings=embeddings,
        # NPZ consumers deliberately use allow_pickle=False.  Pandas' default
        # string conversion yields object arrays, so materialize native Unicode
        # arrays instead of serialising Python objects.
        gene_ids=local.gene_idx.astype(str).to_numpy(dtype=str),
        rna_col=local.rna_col.to_numpy(np.int64),
        chromosome=local.chromosome.astype(str).to_numpy(dtype=str),
        tss=local.tss.to_numpy(np.int64),
        strand=local.strand.astype(str).to_numpy(dtype=str),
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest,
        "manifest_sha256": sha256(Path(args.manifest)),
        "fasta": args.fasta,
        "fasta_sha256": sha256(Path(args.fasta)),
        "checkpoint": args.checkpoint,
        "revision_requested": args.revision,
        "revision_resolved": resolved_revision,
        "window_length": args.length,
        "pool_radius": args.pool_radius,
        "orientation": "gene-strand canonical",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "rows": len(local),
        "embedding_shape": list(embeddings.shape),
        "artifact_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
