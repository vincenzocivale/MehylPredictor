"""Chromosome-chunked inference against the canonical locus feature store."""
from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import hashlib

import numpy as np
import torch

from ..locus_features.regulatory import unpack_embedding_bags
from ..locus_features.store import LocusFeatureStore
from ..storage import RNACache
from .config import load_rna_recipe
from ..modeling.factory import build_functional_predictor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_tcga_genomewide(
    *, checkpoint: str | Path, recipe: str | Path, locus_store: str | Path,
    rna_cache: str | Path, output: str | Path,
    sample_ids: np.ndarray, chromosomes: tuple[str, ...] | list[str] = ("all",),
    source: str | None = None,
    sample_chunk: int = 32, locus_chunk: int = 2048, device: str = "auto",
    overwrite: bool = False,
) -> dict:
    """Write one memory-mappable prediction matrix per requested chromosome.

    Only RNA rows and canonical locus-store arrays are read; no target or legacy
    functional/annotation cache is consulted. Existing complete chromosome files
    are reused unless ``overwrite`` is set.
    """
    if sample_chunk < 1 or locus_chunk < 1:
        raise ValueError("chunk sizes must be positive")
    root, out = Path(locus_store), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if source is not None:
        out = out / source
        out.mkdir(parents=True, exist_ok=True)
    store = LocusFeatureStore(root)
    store_manifest_hash = _sha256_file(root / "manifest.json")
    rna = RNACache(rna_cache)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    if sample_ids.ndim != 1 or len(sample_ids) == 0:
        raise ValueError("sample_ids must be a non-empty one-dimensional array")
    # Fail closed for unknown sample IDs before creating output artifacts.
    rna.index.positions_of(sample_ids)
    if source is not None:
        if source not in {"array", "epic", "wgbs"}:
            raise ValueError("source must be array, epic or wgbs")
        import pandas as pd
        source_alias = pd.read_parquet(root / "aliases" / f"tcga_{source}.parquet", columns=["cpg_idx", "locus_key"])
        source_keys = np.sort(source_alias.locus_key.to_numpy(dtype=np.uint64))
        if len(source_keys) == 0:
            raise ValueError(f"TCGA {source} alias is empty")
    recipe_obj = load_rna_recipe(recipe)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_hash = _sha256_file(Path(checkpoint))
    lc = ckpt.get("locus_cls") or {}
    functional = lc.get("functional_locus") or {}
    if functional.get("n_tracks") != 4165 or functional.get("dense_dim") != 23:
        raise ValueError("checkpoint does not use the 4165+23 functional contract")
    variant = recipe_obj.model.functional_fusion_variant
    if not variant:
        raise ValueError("recipe lacks model.functional_fusion_variant")
    model, _ = build_functional_predictor(
        variant=variant, input_dim=rna.values.shape[1], config=recipe_obj.model,
        final_regressor_dropout=float(recipe_obj.raw.get("locus_cls", {}).get("final_regressor_dropout", 0.0)),
        use_mean_proxy=bool(lc.get("use_mean_branch", True)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    model.to(dev).eval()
    requested = [f"chr{i}" for i in range(1, 23)] if list(chromosomes) == ["all"] else [str(c) for c in chromosomes]
    summaries = []
    for chrom in requested:
        shard = root / "features" / chrom
        keys = np.load(shard / "locus_key.npy", mmap_mode="r")
        positions = np.load(shard / "position.npy", mmap_mode="r")
        if source is not None:
            probe = np.searchsorted(source_keys, keys)
            selected = probe < len(source_keys)
            # searchsorted indices are clipped before dereference.
            probe = np.minimum(probe, len(source_keys) - 1)
            selected &= source_keys[probe] == keys
            keys = np.asarray(keys[selected], dtype=np.uint64)
            positions = np.asarray(positions[selected], dtype=np.int64)
        path = out / f"{chrom}.prediction.f32.npy"
        manifest_path = out / f"{chrom}.manifest.json"
        if manifest_path.exists() and path.exists() and not overwrite:
            old = json.loads(manifest_path.read_text())
            if (old.get("status") == "complete" and old.get("n_loci") == len(keys)
                    and old.get("n_samples") == len(sample_ids)
                    and old.get("sample_ids") == sample_ids.tolist()
                    and old.get("source") == source
                    and old.get("checkpoint_sha256") == checkpoint_hash
                    and old.get("locus_store_manifest_sha256") == store_manifest_hash
                    and old.get("sample_chunk") == sample_chunk
                    and old.get("locus_chunk") == locus_chunk):
                summaries.append(old); continue
            if old.get("status") == "complete":
                raise ValueError(f"existing {chrom} output is incompatible; use --overwrite explicitly")
        tmp_dir = Path(tempfile.mkdtemp(prefix=f".{chrom}.", dir=out))
        try:
            tmp_path = tmp_dir / path.name
            pred = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(len(sample_ids), len(keys)))
            for s0 in range(0, len(sample_ids), sample_chunk):
                sids = sample_ids[s0:s0 + sample_chunk]
                rx = torch.from_numpy(rna.rows(sids)).to(dev).float()
                for c0 in range(0, len(keys), locus_chunk):
                    c1 = min(c0 + locus_chunk, len(keys))
                    annotation, breadth, packed = store.lookup_full(keys[c0:c1])
                    tracks, offsets = unpack_embedding_bags(packed)
                    kwargs = {
                        "functional_track_indices": torch.from_numpy(tracks).to(dev),
                        "functional_offsets": torch.from_numpy(offsets).to(dev),
                        "functional_dense": torch.from_numpy(np.concatenate((annotation, breadth), axis=1)).to(dev).float(),
                    }
                    with torch.no_grad():
                        pred[s0:s0 + len(sids), c0:c1] = model(rx, **kwargs)["beta"].detach().cpu().numpy().astype(np.float32)
            pred.flush(); del pred
            metadata = {"status": "complete", "schema_version": 1, "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": checkpoint_hash, "recipe": str(Path(recipe).resolve()), "locus_store": str(root.resolve()), "locus_store_manifest_sha256": store_manifest_hash, "chromosome": chrom, "source": source, "sample_chunk": sample_chunk, "locus_chunk": locus_chunk, "n_samples": len(sample_ids), "n_loci": len(keys), "sample_ids": sample_ids.tolist(), "coordinate_source": str((shard / "locus_key.npy").resolve()), "source_alias": None if source is None else str((root / "aliases" / f"tcga_{source}.parquet").resolve())}
            os.replace(tmp_path, path)
            tmp_manifest = tmp_dir / manifest_path.name; tmp_manifest.write_text(json.dumps(metadata, indent=2) + "\n"); os.replace(tmp_manifest, manifest_path)
            summaries.append(metadata)
        finally:
            for item in tmp_dir.iterdir(): item.unlink()
            tmp_dir.rmdir()
    return {"status": "complete", "chromosomes": summaries, "output": str(out)}
