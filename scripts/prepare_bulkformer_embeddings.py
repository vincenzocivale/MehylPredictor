#!/usr/bin/env python3
"""Extract frozen BulkFormer-147M sample embeddings into RNACache format.

Run this from an environment containing the HF-compatible `bulkformer` package.
The resulting cache is environment-independent and is consumed by the normal
`methyl-predictor` trainer through FrozenEmbeddingEncoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylation_predictor.tcga_canonical.bundle import TCGACanonicalBundle
from methylation_predictor.tcga_canonical.config import resolve_bundle_root

DEFAULT_MODEL = "zhfanrui/BulkFormer-147M"


def parse_ensembl(raw: str) -> str:
    base = str(raw).split(";")[-1].split(".")[0]
    if not (base.startswith("ENSG") and base[4:].isdigit()):
        raise ValueError(f"unexpected canonical gene id: {raw!r}")
    return base


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0, :]
    if value.ndim != 2:
        raise RuntimeError(f"BulkFormer returned {value.shape}; expected [samples, dim]")
    return value.astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-root", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    try:
        from bulkformer import BulkFormerModel, extract_features
        from bulkformer.preprocessing import normalize_data, align_to_vocab
    except ImportError as exc:
        raise RuntimeError("BulkFormer extraction requires the `bulkformer` package/environment") from exc

    root = resolve_bundle_root(args.canonical_root)
    bundle = TCGACanonicalBundle.from_root(root)
    sample_ids = np.asarray(bundle.rna.h5["sample_idx"][...], dtype=np.int64)
    ensembl_ids = [parse_ensembl(g) for g in bundle.rna.gene_ids]

    model = BulkFormerModel.from_pretrained(args.model).eval()
    gene_vocab = list(model.config.gene_ids)
    canonical_set = set(ensembl_ids)
    overlap = sum(g in canonical_set for g in gene_vocab)
    print(f"[bulkformer] gene coverage {overlap}/{len(gene_vocab)} ({overlap/len(gene_vocab):.1%})")

    def extract_chunk(ids: np.ndarray) -> np.ndarray:
        raw = bundle.rna.rows(ids).astype(np.float32, copy=False)
        counts = pd.DataFrame(raw, columns=ensembl_ids)
        normalized = normalize_data(counts)
        aligned, mask_prob = align_to_vocab(normalized, gene_vocab)
        features = extract_features(
            model,
            aligned.values,
            output_feature_type="sample_level",
            aggregate_type="mean",
            mask_prob=mask_prob,
            device=args.device,
            batch_size=min(args.batch_size, len(ids)),
        )
        return as_numpy(features)

    if args.probe_only:
        probe = extract_chunk(sample_ids[: min(2, len(sample_ids))])
        print(f"[bulkformer] probe OK {probe.shape}"); bundle.close(); return

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    values_path = out / "rna_zscore.f16.npy"; sample_path = out / "rna_sample_idx.npy"
    progress_path = out / "progress.json"
    start = 0; mmap = None; embedding_dim = None
    if args.resume and progress_path.is_file() and values_path.is_file() and sample_path.is_file():
        progress = json.loads(progress_path.read_text())
        start = int(progress["next_index"]); embedding_dim = int(progress["embedding_dim"])
        if not np.array_equal(np.load(sample_path), sample_ids):
            raise RuntimeError("existing cache sample_idx does not match canonical bundle")
        mmap = np.lib.format.open_memmap(values_path, mode="r+", dtype=np.float16, shape=(len(sample_ids), embedding_dim))
        print(f"[bulkformer] resume {start}/{len(sample_ids)}")

    for s0 in range(start, len(sample_ids), args.batch_size):
        s1 = min(s0 + args.batch_size, len(sample_ids)); emb = extract_chunk(sample_ids[s0:s1])
        if mmap is None:
            embedding_dim = int(emb.shape[1]); np.save(sample_path, sample_ids.astype(np.int64))
            mmap = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float16, shape=(len(sample_ids), embedding_dim))
        if emb.shape[1] != embedding_dim:
            raise RuntimeError(f"embedding dimension changed {embedding_dim}->{emb.shape[1]}")
        mmap[s0:s1] = emb.astype(np.float16); mmap.flush()
        progress_path.write_text(json.dumps({
            "model": args.model, "next_index": s1, "n_samples": len(sample_ids),
            "embedding_dim": embedding_dim, "canonical_root": str(root),
            "gene_vocab_size": len(gene_vocab), "matched_genes": overlap,
        }, indent=2) + "\n")
        print(f"[bulkformer] {s1}/{len(sample_ids)}", flush=True)

    meta = {
        "source": "BulkFormer frozen sample-level mean-pooled embedding",
        "model": args.model, "n_samples": len(sample_ids), "embedding_dim": int(embedding_dim),
        "gene_vocab_size": len(gene_vocab), "matched_genes": overlap,
        "gene_coverage": overlap / len(gene_vocab), "canonical_root": str(root),
        "note": "rna_zscore.f16.npy contains embeddings, not z-scores; filename is RNACache compatibility only",
    }
    (out / "README.md").write_text(json.dumps(meta, indent=2) + "\n")
    progress_path.write_text(json.dumps({**meta, "next_index": len(sample_ids), "complete": True}, indent=2) + "\n")
    bundle.close(); print(f"[bulkformer] complete -> {out}")


if __name__ == "__main__":
    main()
