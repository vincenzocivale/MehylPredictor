"""Illumina probe-ID / hg38-position crosswalk, reused from CpGPT's own dependency bundle.

Both foundation-model blockers recorded in `docs/PAPER_EXPERIMENTS.md`'s foundation-model
section as of 2026-09-02 -- CpGPT's genomic-position -> DNA-embedding-mmap-row lookup, and
MethylGPT's missing Illumina probe-ID <-> hg38-position crosswalk -- turn out to be the *same*
missing piece, and it already ships inside CpGPT's own human-dependencies bundle
(`external/checkpoints/cpgpt_human_dependencies/`, downloaded 2026-09-02, see
`external/checkpoints/cpgpt_human_dependencies/README.md`):

- `illumina_metadata.db` -- a `sqlitedict` database with one key per species (`"homo_sapiens"`
  here); its value is a pickled `{probe_id: "chrom:pos"}` dict, built by CpGPT's own
  `IlluminaMethylationProber._process_file` (`external/CpGPT/cpgpt/data/components/
  illumina_methylation_prober.py`) from six Illumina hg38 manifests (EPICv2/EPIC+/EPIC/HM27/
  HM450/MSA) -- this *is* the probe-ID <-> hg38-position crosswalk MethylGPT's fixed
  49,156-probe vocabulary needs, already covering ~1.19M probes across all six arrays.
- `ensembl_metadata.db` -- a `sqlitedict` database whose `"homo_sapiens"` key's pickled value
  nests `[dna_llm][dna_context_len]` -> `{"chrom:pos": mmap_row_index}`, built by CpGPT's own
  `DNALLMEmbedder._process_genomic_locations` (`.../dna_llm_embedder.py`) -- this *is* the
  genomic-position -> DNA-embedding-mmap-row lookup CpGPT's own evaluator needs.

Both dicts use the same location-string convention: `f"{chrom_without_chr_prefix}:{pos}"`,
0-based GRCh38 (Illumina manifest `CpG_beg`, per `illumina_methylation_prober.py` -- read
directly, not re-derived). This repo's own `array_cpg_map.parquet` registry stores 1-based
positions with a `"chr"` prefix (e.g. `("chr1", 100038040)`); **verified empirically 2026-09-07**
(swept a +/-2bp offset against real chr1 `val_cpg_x_train_sample` positions) that the correct
conversion is `illumina_pos = registry_pos - 1` -- at that offset, 6741/6742 (99.99%) of chr1's
held-out CpGs hit both dicts. Off by one in either direction drops the hit rate to <2%, so this
isn't a coincidental partial match; it's the correct convention, encoded once here as
`to_location_key` so no caller has to re-derive it.
"""
from __future__ import annotations

import pickle
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_DNA_LLM = "nucleotide-transformer-v2-500m-multi-species"
DEFAULT_DNA_CONTEXT_LEN = 2001
DEFAULT_SPECIES = "homo_sapiens"
DEFAULT_DNA_EMBEDDING_SIZE = 1024


def _load_pickled_key(db_path: str | Path, key: str) -> dict:
    """Read one `sqlitedict`-style pickled value by key, without the `sqlitedict` dependency.

    `illumina_metadata.db`/`ensembl_metadata.db` are plain SQLite files with a single
    `unnamed(key TEXT, value BLOB)` table (`sqlitedict`'s on-disk format) -- reading them with
    stdlib `sqlite3` + `pickle` avoids pulling in `sqlitedict` (a CpGPT-only dependency, not in
    this repo's own requirements) just to read two keys.
    """
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT value FROM unnamed WHERE key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"key {key!r} not found in {db_path}")
        return pickle.loads(row[0])
    finally:
        con.close()


@dataclass
class IlluminaCrosswalk:
    """probe-id <-> (chrom, pos) and (chrom, pos) -> DNA-embedding-mmap-row, both from CpGPT's
    human-dependencies bundle. See module docstring for provenance and the offset convention.
    """

    position_to_probe: dict[str, str]
    position_to_dna_embedding_row: dict[str, int]
    dna_embeddings_path: Path
    dna_embedding_size: int = DEFAULT_DNA_EMBEDDING_SIZE

    @classmethod
    def load(
        cls,
        cpgpt_human_dependencies_dir: str | Path,
        *,
        species: str = DEFAULT_SPECIES,
        dna_llm: str = DEFAULT_DNA_LLM,
        dna_context_len: int = DEFAULT_DNA_CONTEXT_LEN,
    ) -> "IlluminaCrosswalk":
        deps = Path(cpgpt_human_dependencies_dir)

        illumina = _load_pickled_key(deps / "illumina_metadata.db", species)
        position_to_probe = {location: probe for probe, location in illumina.items()}

        ensembl = _load_pickled_key(deps / "ensembl_metadata.db", species)
        dna_index = dict(ensembl[dna_llm][dna_context_len])

        dna_embeddings_path = (
            deps / "dna_embeddings" / species / dna_llm / f"{dna_context_len}bp_dna_embeddings.mmap"
        )

        return cls(
            position_to_probe=position_to_probe,
            position_to_dna_embedding_row=dna_index,
            dna_embeddings_path=dna_embeddings_path,
        )

    @staticmethod
    def to_location_key(chrom: str, pos: int) -> str:
        """`(chrom, pos)` in this repo's own 1-based, `"chr"`-prefixed convention ->
        the 0-based, unprefixed `"chrom:pos"` key both crosswalk dicts use. See module
        docstring for the empirical verification of the `-1` offset.
        """
        bare_chrom = chrom[3:] if chrom.startswith("chr") else chrom
        return f"{bare_chrom}:{pos - 1}"

    def probe_id(self, chrom: str, pos: int) -> str | None:
        return self.position_to_probe.get(self.to_location_key(chrom, pos))

    def dna_embedding_row(self, chrom: str, pos: int) -> int | None:
        return self.position_to_dna_embedding_row.get(self.to_location_key(chrom, pos))

    def dna_embedding_matrix(self, *, n_rows: int | None = None) -> np.memmap:
        """Memory-map CpGPT's `2001bp_dna_embeddings.mmap` (read-only, ~5GB on disk).

        `n_rows` defaults to the file size implied row count; pass the exact row count from
        `ensembl_metadata.db` (`max(position_to_dna_embedding_row.values()) + 1`) if the file
        might have trailing padding.
        """
        if n_rows is None:
            row_bytes = np.dtype("float32").itemsize * self.dna_embedding_size
            n_rows = self.dna_embeddings_path.stat().st_size // row_bytes
        return np.memmap(
            self.dna_embeddings_path,
            dtype="float32",
            mode="r",
            shape=(n_rows, self.dna_embedding_size),
        )

    def coverage(self, positions: list[tuple[str, int]]) -> dict:
        """Fraction of `positions` (this repo's `(chrom, pos)` convention) covered by each
        crosswalk -- `probe_vocab_coverage` is what limits MethylGPT (fixed vocabulary),
        `dna_embedding_coverage` is what limits CpGPT (needs a precomputed embedding row).
        """
        keys = [self.to_location_key(chrom, pos) for chrom, pos in positions]
        n_probe = sum(k in self.position_to_probe for k in keys)
        n_dna = sum(k in self.position_to_dna_embedding_row for k in keys)
        n = len(keys)
        return {
            "n_query": n,
            "n_probe_vocab_hits": n_probe,
            "probe_vocab_coverage": (n_probe / n) if n else 0.0,
            "n_dna_embedding_hits": n_dna,
            "dna_embedding_coverage": (n_dna / n) if n else 0.0,
        }
