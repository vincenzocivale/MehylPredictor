"""Technology aliases and one coordinate-deduplicated TCGA locus catalog."""
from __future__ import annotations

from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd

from .ids import locus_key


def _atomic_parquet(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.parquet', delete=False) as tmp:
        name = tmp.name
    try:
        frame.to_parquet(name, index=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def build_catalog(canonical_root: str | Path, output: str | Path, *, replace: bool = False):
    root = Path(canonical_root) / 'cpg' / 'registries'
    out = Path(output)
    if not root.exists():
        raise FileNotFoundError(root)
    catalog = []
    for tech in ('array','epic','wgbs'):
        dest = out / 'aliases' / f'tcga_{tech}.parquet'
        if dest.exists() and not replace:
            frame = pd.read_parquet(dest)
        else:
            frame = pd.read_parquet(root / f'{tech}_cpg_map.parquet', columns=['cpg_idx','chr','pos'])
            if frame[['cpg_idx']].duplicated().any():
                raise ValueError(f'{tech} contains duplicate cpg_idx')
            frame = frame.rename(columns={'chr':'chrom','pos':'position'})
            codes = frame.chrom.map({f'chr{i}':i for i in range(1,23)} | {'chrX':23,'chrY':24})
            if codes.isna().any() or (frame.position < 1).any() or (frame.position >= 2**32).any():
                raise ValueError(f'{tech} contains invalid hg38 coordinates')
            frame['locus_key'] = (codes.to_numpy(dtype=np.uint64) << np.uint64(32)) | frame.position.to_numpy(dtype=np.uint64)
            if frame.locus_key.duplicated().any():
                raise ValueError(f'{tech} contains duplicate hg38 coordinates')
            _atomic_parquet(frame, dest)
        catalog.append(frame[['locus_key','chrom','position']])
    union = pd.concat(catalog, ignore_index=True)
    conflict = union.groupby('locus_key',sort=False).agg({'chrom':'nunique','position':'nunique'})
    if (conflict > 1).any().any():
        raise ValueError('locus_key collision')
    union = union.drop_duplicates('locus_key').sort_values('locus_key')
    union['chrom_code'] = (union.locus_key.to_numpy(dtype=np.uint64) >> np.uint64(32)).astype(np.uint8)
    for chrom,frame in union.groupby('chrom',sort=False):
        dest = out / 'catalog' / f'{chrom}.parquet'
        if dest.exists() and not replace:
            old = pd.read_parquet(dest)
            if not old.equals(frame.reset_index(drop=True)):
                raise ValueError(f'existing catalog differs: {dest}')
        else:
            _atomic_parquet(frame.reset_index(drop=True), dest)
    return len(union)


def append_coordinates(output: str | Path, coordinates: pd.DataFrame) -> int:
    """Add ENCODE/future hg38 loci to the common catalog, retaining existing shards."""
    root = Path(output)
    required = {'chrom', 'position'}
    if not required.issubset(coordinates.columns):
        raise ValueError('coordinate input requires chrom and position')
    frame = coordinates[['chrom', 'position']].copy()
    frame['locus_key'] = [locus_key(c, int(p)) for c, p in zip(frame.chrom, frame.position)]
    if frame.locus_key.duplicated().any():
        raise ValueError('input contains duplicate hg38 loci')
    frame['chrom_code'] = (frame.locus_key.to_numpy(dtype=np.uint64) >> np.uint64(32)).astype(np.uint8)
    added = 0
    for chrom, group in frame.groupby('chrom'):
        path = root / 'catalog' / f'{chrom}.parquet'
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=group.columns)
        combined = pd.concat([old, group], ignore_index=True).drop_duplicates('locus_key').sort_values('locus_key').reset_index(drop=True)
        delta = len(combined) - len(old)
        if delta and (root / 'features' / chrom).exists():
            raise ValueError(f'{chrom} shard must be rebuilt before adding {delta} new loci')
        if delta:
            _atomic_parquet(combined, path)
            added += delta
    return added
