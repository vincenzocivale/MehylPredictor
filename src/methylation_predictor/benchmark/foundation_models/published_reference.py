"""Published-reference metrics for CpGPT / MethylGPT, mirroring the dict-of-dicts +
`published_delta()` pattern already used for MethylProphet's own tables
(`benchmark.methylprophet.protocol.TABLE5_PUBLISHED_METHYLPROPHET`,
`TABLE7_PUBLISHED_METHYLPROPHET`, `published_delta`).

Deliberately left as an empty, structured template rather than pre-filled with
numbers pulled from a search summary: the only lead so far (a MethylProphet-paper
comparison table reporting a CpGPT MAS-PCC around 0.32/0.48 on ENCODE/TCGA) was
seen only as a secondhand web-search snippet, not read directly off the paper's own
table -- filling this in needs the actual PDF table (view definition, which
CpGPT/MethylGPT checkpoint size, TCGA vs TCGA-chr1) confirmed directly, not
transcribed from a summary. Fill in `CPGPT_PUBLISHED` / `METHYLGPT_PUBLISHED` once
that's done, keeping the same `{view: {mas_pcc, mac_pcc, mse, mae}}` shape so
`published_delta()` (reused as-is, no changes needed) works unmodified against
either dict via `benchmark.methylprophet.protocol.published_delta(ours, view,
reference=CPGPT_PUBLISHED)`.
"""
from __future__ import annotations

# TODO(paper-comparison): fill in from the CpGPT paper (biorxiv 2024.10.24.619766)
# and/or MethylProphet's own Table comparing against CpGPT/DeepCPG, once read
# directly off the source table -- see module docstring.
CPGPT_PUBLISHED: dict[str, dict[str, float]] = {}

# TODO(paper-comparison): fill in from the MethylGPT paper (biorxiv 2024.10.30.621013)
# if/when it reports a comparable held-out-CpG imputation view.
METHYLGPT_PUBLISHED: dict[str, dict[str, float]] = {}
