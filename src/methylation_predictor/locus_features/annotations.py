"""Static hg38 reference annotations with frozen paper-model encoding."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import json

import numpy as np

from .ids import canonical_locus

CONTEXT = ('island', 'shore', 'shelf', 'open_sea')
REGIONS = ('promoter', 'exon', 'intron', 'intergenic')
CCRE = ('none', 'PLS', 'pELS', 'dELS', 'CA', 'TF', 'CA-CTCF', 'CA-H3K4me3', 'CA-TF')


@dataclass
class RawAnnotations:
    cpg_island_class: np.ndarray
    genomic_region: np.ndarray
    ccre_class: np.ndarray
    dist_tss_abs_log10: np.ndarray


def _point_coverage(pos: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    if not intervals:
        return np.zeros(len(pos), dtype=bool)
    ar = np.asarray(intervals, dtype=np.int64)
    return np.searchsorted(np.sort(ar[:, 0]), pos, side='right') > np.searchsorted(np.sort(ar[:, 1]), pos, side='right')


def _nearest_distance(pos: np.ndarray, sites: list[int]) -> np.ndarray:
    if not sites:
        raise ValueError('no GENCODE TSS on chromosome')
    sites = np.unique(np.asarray(sites, np.int64))
    right = np.searchsorted(sites, pos)
    left_site = sites[np.maximum(right - 1, 0)]
    right_site = sites[np.minimum(right, len(sites) - 1)]
    return np.minimum(np.abs(pos - left_site), np.abs(pos - right_site))


class ReferenceAnnotationEngine:
    """Chromosome-local interval engine using the frozen UCSC/GENCODE/cCRE sources."""

    def __init__(self, source_root: str | Path, fai: str | Path):
        self.source_root = Path(source_root)
        self.lengths = {}
        self.fai = {}
        with open(fai) as fh:
            for line in fh:
                name, length, offset, bases_per_line, bytes_per_line = line.split('\t')[:5]
                self.lengths[name] = int(length)
                self.fai[name] = tuple(map(int, (offset, bases_per_line, bytes_per_line)))
        self.reference = np.memmap(Path(fai).with_suffix(''), dtype=np.uint8, mode='r')
        self._cache = {}

    def _load(self, chrom: str):
        if chrom in self._cache:
            return self._cache[chrom]
        islands = []
        with gzip.open(self.source_root / 'cpgIslandExt.hg38.txt.gz', 'rt') as fh:
            for line in fh:
                row = line.rstrip().split('\t')
                if row[1] == chrom:
                    islands.append((int(row[2]), int(row[3])))
        cre = {name: [] for name in CCRE[1:]}
        with open(self.source_root / 'GRCh38-cCREs.Registry-V4.bed') as fh:
            for line in fh:
                row = line.rstrip().split('\t')
                if row[0] == chrom and row[5] in cre:
                    cre[row[5]].append((int(row[1]), int(row[2])))
        genes, exons, promoters, tss = [], [], [], []
        with gzip.open(self.source_root / 'gencode.v50.annotation.gtf.gz', 'rt') as fh:
            for line in fh:
                if not line.startswith(chrom + '\t'):
                    continue
                row = line.rstrip().split('\t')
                if len(row) < 9:
                    continue
                typ, start, end, strand = row[2], int(row[3]), int(row[4]), row[6]
                if typ == 'gene':
                    genes.append((start - 1, end))
                    site = start if strand == '+' else end
                    tss.append(site)
                    promoters.append((max(0, site - 2000 - 1), site + 500) if strand == '+' else (max(0, site - 500 - 1), site + 2000))
                elif typ == 'exon':
                    exons.append((start - 1, end))
        self._cache[chrom] = (islands, cre, genes, exons, promoters, tss)
        return self._cache[chrom]

    def compute_raw_annotations(self, chrom: str, position: np.ndarray) -> RawAnnotations:
        pos = np.asarray(position, dtype=np.int64)
        if pos.ndim != 1 or len(pos) == 0:
            raise ValueError('positions must be a nonempty 1D array')
        for p in pos:
            canonical_locus(chrom, int(p), lengths=self.lengths)
        if len(np.unique(pos)) != len(pos):
            raise ValueError('duplicate canonical CpG coordinates')
        offset, bases_per_line, bytes_per_line = self.fai[chrom]
        zero = pos - 1
        c_at = offset + (zero // bases_per_line) * bytes_per_line + (zero % bases_per_line)
        g_zero = zero + 1
        g_at = offset + (g_zero // bases_per_line) * bytes_per_line + (g_zero % bases_per_line)
        if not np.all(np.isin(self.reference[c_at], (ord('C'), ord('c'))) & np.isin(self.reference[g_at], (ord('G'), ord('g')))):
            raise ValueError(f'coordinate is not a reference-forward CpG on {chrom}')
        islands, cre, genes, exons, promoters, tss = self._load(chrom)
        bed = pos - 1
        context = np.full(len(pos), 'open_sea', dtype='U8')
        # UCSC shore/shelf are 2 kb bands around island boundaries.
        for cls, radius in [('shelf', 4000), ('shore', 2000)]:
            expanded = [(max(0, s-radius), e+radius) for s,e in islands]
            context[_point_coverage(bed, expanded)] = cls
        context[_point_coverage(bed, islands)] = 'island'
        region = np.full(len(pos), 'intergenic', dtype='U10')
        region[_point_coverage(bed, genes)] = 'intron'
        region[_point_coverage(bed, exons)] = 'exon'
        region[_point_coverage(bed, promoters)] = 'promoter'
        cc = np.full(len(pos), 'none', dtype='U12')
        for cls in CCRE[1:]:
            cc[_point_coverage(bed, cre[cls])] = cls
        distance = _nearest_distance(pos, tss)
        return RawAnnotations(context, region, cc, np.log10(distance.astype(np.float64) + 1).astype(np.float32))


def encode_annotation_core(raw: RawAnnotations, contract: dict | str | Path) -> np.ndarray:
    if not isinstance(contract, dict):
        contract = json.loads(Path(contract).read_text())
    names = [f['name'] for f in contract['features']]
    expected = [f'cpg_context__{x}' for x in CONTEXT] + [f'genomic_region__{x}' for x in REGIONS] + [f'ccre_class__{x}' for x in CCRE] + ['dist_tss_abs_log10__z']
    if names != expected or contract['n_features'] != 18:
        raise ValueError('annotation contract order differs from historical 18 features')
    n = len(raw.cpg_island_class)
    if any(len(v) != n for v in (raw.genomic_region, raw.ccre_class, raw.dist_tss_abs_log10)):
        raise ValueError('raw annotation axes differ')
    out = np.empty((n, 18), dtype=np.float32)
    for field, choices, offset in ((raw.cpg_island_class, CONTEXT, 0), (raw.genomic_region, REGIONS, 4), (raw.ccre_class, CCRE, 8)):
        if not np.isin(field, choices).all():
            raise ValueError('unknown annotation category')
        for j, choice in enumerate(choices):
            out[:, offset+j] = field == choice
    norm = contract['features'][17]['normalization']
    out[:, 17] = (np.asarray(raw.dist_tss_abs_log10, np.float64) - norm['mean']) / norm['std']
    return out
