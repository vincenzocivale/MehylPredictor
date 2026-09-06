"""Offline: extract frozen BulkRNABert embeddings for every canonical TCGA RNA sample.

BulkRNABert (github.com/instadeepai/multiomics-open-research,
huggingface.co/InstaDeepAI/BulkRNABert) is a ~6M-parameter transformer
pretrained on bulk RNA-seq over a fixed panel of 19,062 Ensembl gene IDs,
producing a 256-d embedding per sample. This script runs it once, frozen (no
fine-tuning), over every sample in the canonical TCGA RNA bundle and writes
the result in the same on-disk contract ``RNACache`` already reads
(``storage.py``): ``rna_sample_idx.npy`` + ``rna_zscore.f16.npy`` (the latter
name is reused only for ``RNACache`` on-disk-format compatibility -- these
are embeddings, not a z-score) under ``--output``. Point a
``frozen_embedding``-kind recipe's ``--rna-cache`` at that directory (see
``configs/models/arch_shared_backbone/enc_frozen_embedding_bulkrnabert.yaml``)
to train against it -- ``rna_training/locus_cls_trainer.py`` derives
``input_dim`` dynamically from whichever cache is actually opened, so no
other wiring change is needed.

LICENSE: BulkRNABert is CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike).
Do not use its embeddings, or anything derived from them, in a commercial
context without a separate agreement with InstaDeepAI.

Model API and data-alignment logic below were ACTUALLY RUN against the real
checkpoint and a real canonical-bundle copy found on this machine
(2026-09-05) -- not just inferred from reading source, though the full
10,916-sample extraction itself was never run end to end (see the memory
caveat below):

  - ``AutoModel``/``AutoConfig``/``AutoTokenizer.from_pretrained(hf_repo,
    trust_remote_code=True)`` -- loaded and run for real. This repo hosts a
    single checkpoint on ``main`` (no revision/variant selector found); its
    bundled sample data is TCGA-specific (``data/tcga_sample.csv``),
    consistent with this being the TCGA-pretrained variant, but that is an
    inference, not a documented fact -- the GitHub repo's
    ``get_bulkrnabert_pretrained_model`` JAX path additionally exposes
    ``gtex_encode``/``gtex_encode_tcga`` variants not visibly hosted here.
  - Preprocessing is ``log10(1 + raw_count)`` (model card), THEN
    ``tokenizer.batch_encode_plus(log_expression, return_tensors="pt")`` on a
    ``(n_samples, n_genes)`` array in the model's fixed gene order -- the
    tokenizer bins each value into its own bin count internally
    (``np.digitize``), so this script does not reimplement binning itself.
  - ``model.forward(input_ids, attention_mask=None) -> dict`` -- confirmed by
    ``inspect.signature`` on the loaded model: a plain dict, NOT a HF
    ``ModelOutput``. The embedding is only populated when
    ``model.config.embeddings_layers_to_save`` is set (mutated on the loaded
    model instance, mirroring ``ntv3_atlas.py::_load_ntv3_model``'s identical
    pattern for the same publisher's NTv3 checkpoints), under key
    ``f"embeddings_{layer}"``; this script sets it to ``[config.num_layers]``.
  - Gene panel: ``download_gene_panel_from_checkpoint`` downloads and parses
    the checkpoint's own ``data/tcga_sample.csv`` header for real. That header
    has 19,063 columns, the LAST of which is a non-gene ``"identifier"``
    column (a TCGA sample barcode) -- filtered out by Ensembl-ID shape.
  - The canonical bundle's ``gene_ids`` are **not** plain Ensembl IDs --
    spot-checked against all 25,017 rows of a real on-disk bundle copy
    (``.../MethylFM/outputs/canonical_recovery/tcga_rna_official_full.h5``,
    same schema as ``docs/DATA.md``'s file of the same name): every entry is
    ``"{symbol};{ENSG_ID}.{version}"`` (e.g. ``"TSPAN6;ENSG00000000003.15"``),
    no exceptions. ``parse_ensembl_id``/``extract_ensembl_ids`` extract the
    versionless Ensembl ID BulkRNABert's panel keys on; real alignment against
    the downloaded panel matched **15,259 / 19,062 genes (80.0%)**.
  - This checkpoint's ``tokenizer_config.json`` sets ``prepend_cls_token:
    false``, confirmed empirically: ``input_ids.shape[1] == n_genes`` exactly.
    ``build_input_ids`` still auto-detects a 0-or-+1 offset rather than
    hardcoding this, so it stays correct if a different checkpoint sets it.
  - **Real bug found and fixed**: ``tokenizer.mask_token_id`` (the standard HF
    property) crashes on this checkpoint with ``KeyError: None`` -- its custom
    ``_convert_token_to_id`` does ``self.vocab.get(token, self.vocab[self.unk_token])``,
    and Python evaluates that default *eagerly*, so ``self.vocab[None]``
    (``unk_token`` is unset here) raises even though ``"<mask>"`` genuinely is
    in the vocab. ``resolve_mask_token_id`` reads ``tokenizer.vocab`` directly
    instead, bypassing the broken property -- this is a bug in the vendored
    checkpoint's own code, verified by loading it, not a guess.

**Memory caveat, discovered by actually running the forward pass (module has
NOT completed a real end-to-end batch due to this)**: self-attention over a
~19,062-token sequence is O(n^2) -- the attention-weights tensor alone is
``batch_size * num_heads * n_genes^2 * 4 bytes`` ~= ``batch_size * 8 *
19062^2 * 4`` ~= ``batch_size * 11.6 GiB``. A 2-sample probe batch already
triggered ``CUDA OutOfMemoryError`` (tried to allocate 21.66 GiB) on a 40GB
GPU that had only ~11GB free at the time. ``--batch-size`` therefore defaults
to 1 here, not a larger number -- raise it only if you have measured that
much *free* (not total) GPU memory per unit of batch size, and expect the
full 10,916-sample extraction to take a real amount of wall-clock time
either way (untimed in this environment: the probe forward pass itself was
not let finish, see the CPU-vs-GPU tradeoff discussion this script's authors
had before writing this note).

Still a genuine design choice, not delegated to BulkRNABert's own documented
behavior: how the canonical bundle's 25,017 genes not in BulkRNABert's
19,062-gene panel, or the panel's genes absent from the canonical bundle
(19.7% of the panel, per the real alignment above), are handled by the
*pretrained checkpoint itself*. This script represents "not covered by our
data" positions with the tokenizer's ``<mask>`` token -- a reasonable
inference from the vocab having a mask token, not a documented BulkRNABert
usage pattern for missing genes.

Usage (once weights are available)::

    # Sanity-check gene-panel coverage and the model's actual input_ids shape
    # first, without writing a full embedding cache:
    python scripts/prepare_bulkrnabert_embeddings.py \\
        --canonical-root /path/to/tcga_canonical_bundle \\
        --output /path/to/caches/bulkrnabert_tcga --probe-only

    # Then the real extraction (gene panel auto-downloaded from the
    # checkpoint's own data/tcga_sample.csv header unless --gene-panel is
    # given explicitly):
    python scripts/prepare_bulkrnabert_embeddings.py \\
        --canonical-root /path/to/tcga_canonical_bundle \\
        --output /path/to/caches/bulkrnabert_tcga --batch-size 64 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import torch

from methylation_predictor.tcga_canonical.bundle import TCGACanonicalBundle
from methylation_predictor.tcga_canonical.config import resolve_bundle_root

DEFAULT_HF_REPO = "InstaDeepAI/BulkRNABert"
EXPECTED_EMBED_DIM = 256


def parse_ensembl_id(gene_id: str) -> str:
    """Extract the versionless Ensembl gene ID from the canonical bundle's
    ``"{symbol};{ENSG_ID}.{version}"`` format (e.g. ``"TSPAN6;ENSG00000000003.15"``
    -> ``"ENSG00000000003"``) -- verified against all 25,017 rows of a real
    on-disk bundle copy (module docstring), not a guess. Raises if a gene_id
    doesn't match this exact shape, so a future bundle snapshot with a
    genuinely different format fails loudly here rather than silently
    misaligning the panel.
    """
    parts = gene_id.split(";")
    ensembl_with_version = parts[-1]
    base = ensembl_with_version.split(".")[0]
    if not (base.startswith("ENSG") and base[4:].isdigit()):
        raise ValueError(
            f"gene_id {gene_id!r} does not match the expected "
            "'{{symbol}};ENSG{{digits}}.{{version}}' format -- see parse_ensembl_id's docstring. "
            "The canonical bundle's gene_ids format may have changed; update this function."
        )
    return base


def extract_ensembl_ids(gene_ids: np.ndarray) -> np.ndarray:
    return np.asarray([parse_ensembl_id(str(g)) for g in gene_ids])


def load_gene_panel(path: str | Path) -> list[str]:
    """One Ensembl gene ID per line (a manually supplied panel file)."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty gene panel file: {path}")
    return lines


def download_gene_panel_from_checkpoint(hf_repo: str) -> list[str]:
    """Gene order = the header row of the checkpoint's own bundled sample data
    (``data/tcga_sample.csv`` on the Hub) -- avoids depending on the separate
    GitHub repo's ``data/bulkrnabert/common_gene_id.txt`` for the same panel.

    Actually downloaded and inspected 2026-09-05 (not just read about): the
    header has 19,063 columns, the first 19,062 of which are Ensembl gene IDs
    in the model's fixed order (matching ``config.json``'s ``n_genes``) and
    the LAST of which is a non-gene ``"identifier"`` column (a TCGA sample
    barcode per data row, e.g. ``"TCGA-06-2559-01A"``) -- filtered out here by
    Ensembl-ID shape rather than by position, in case a future version of this
    file reorders or adds columns.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=hf_repo, filename="data/tcga_sample.csv")
    with open(path, newline="") as fh:
        header = next(csv.reader(fh))
    genes = [g.strip() for g in header if re.match(r"^ENSG\d+$", g.strip())]
    dropped = len(header) - len(genes)
    if dropped:
        print(f"[download_gene_panel_from_checkpoint] dropped {dropped} non-gene column(s) from the CSV header (e.g. \"identifier\")")
    return genes


def align_to_panel(repo_gene_ids: np.ndarray, panel_gene_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Map the canonical bundle's 25017-gene axis onto BulkRNABert's fixed panel.

    Returns ``(repo_columns, panel_positions)``: ``repo_columns[i]`` is the
    canonical RNA matrix column supplying ``panel_gene_ids[panel_positions[i]]``.
    Panel genes absent from the canonical bundle are left out of both arrays
    and filled with the tokenizer's mask token by ``build_input_ids`` instead
    of a fabricated expression value (module docstring, item 3).
    """
    repo_position = {str(g): i for i, g in enumerate(repo_gene_ids)}
    repo_columns: list[int] = []
    panel_positions: list[int] = []
    for panel_pos, gene in enumerate(panel_gene_ids):
        col = repo_position.get(gene)
        if col is not None:
            repo_columns.append(col)
            panel_positions.append(panel_pos)
    if not repo_columns:
        raise ValueError("zero overlap between canonical bundle gene_ids and the BulkRNABert gene panel")
    coverage = len(repo_columns) / len(panel_gene_ids)
    print(f"[align_to_panel] {len(repo_columns)}/{len(panel_gene_ids)} panel genes matched ({coverage:.1%})")
    return np.asarray(repo_columns, dtype=np.int64), np.asarray(panel_positions, dtype=np.int64)


def load_bulkrnabert(hf_repo: str, device: str, dtype: torch.dtype | None = None):
    """Load model + tokenizer via ``trust_remote_code``, requesting the final
    transformer layer's embedding output. Follows the model card's own
    documented usage order exactly (build ``AutoConfig`` first, set
    ``embeddings_layers_to_save`` on it, pass ``config=`` explicitly into
    ``AutoModel.from_pretrained`` -- an earlier version of this function
    mutated ``model.config`` *after* loading instead, which was verified to
    also work in practice, but this matches the checkpoint's own published
    example precisely rather than relying on that). The forward pass returns
    a plain dict, populated only for layers named in
    ``embeddings_layers_to_save``, mirroring
    ``ntv3_atlas.py::_load_ntv3_model``'s identical pattern for the same
    publisher's NTv3 checkpoints.

    ``dtype``: the checkpoint is published as float32 (``config.json``'s
    ``torch_dtype``). Its hand-written attention (``bulkrnabert.py::self.mha``,
    plain ``torch.einsum``, not a fused/flash kernel) materializes the full
    ``(batch, heads, n_genes, n_genes)`` score matrix explicitly -- casting to
    bfloat16 (matching this repo's own AMP convention elsewhere) roughly
    halves that tensor's memory, discovered necessary 2026-09-05 after a real
    CUDA OOM on the *second* consecutive sample at float32/batch_size=1 on a
    40GB GPU (module docstring's memory caveat under-estimated the true cost:
    memory was not fully released between successive forward calls, not just
    scaling with batch_size).
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    kwargs = {"trust_remote_code": True}
    config = AutoConfig.from_pretrained(hf_repo, **kwargs)
    n_layers = getattr(config, "num_layers", None)
    if n_layers is None:
        raise RuntimeError("loaded config has no num_layers -- cannot determine which layer to request embeddings from")
    config.embeddings_layers_to_save = (n_layers,)
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, **kwargs)
    model = AutoModel.from_pretrained(hf_repo, config=config, **kwargs).to(device).eval()
    if dtype is not None:
        model = model.to(dtype)
    embed_dim = getattr(config, "embed_dim", None)
    if embed_dim is not None and embed_dim != EXPECTED_EMBED_DIM:
        print(f"[load_bulkrnabert] WARNING: config.embed_dim={embed_dim}, expected {EXPECTED_EMBED_DIM}")
    print(f"[load_bulkrnabert] loaded {hf_repo!r} dtype={dtype}, requesting embeddings_{n_layers}")
    return model, tokenizer, n_layers


def resolve_mask_token_id(tokenizer) -> int:
    """``tokenizer.mask_token_id`` (the standard HF property) crashes on this
    checkpoint -- confirmed 2026-09-05 by actually loading it: its custom
    ``_convert_token_to_id`` does ``self.vocab.get(token, self.vocab[self.unk_token])``,
    and Python evaluates that default *eagerly*, so ``self.vocab[None]``
    (``unk_token`` is unset on this tokenizer) raises ``KeyError: None`` even
    though ``"<mask>"`` is genuinely in the vocab. Reads ``tokenizer.vocab``
    directly instead, bypassing the broken property -- this is a bug in the
    vendored checkpoint's own code, not a guess on this repo's part.
    """
    vocab = getattr(tokenizer, "vocab", None)
    mask_token = getattr(tokenizer, "mask_token", None)
    if not vocab or mask_token not in vocab:
        raise RuntimeError(
            "tokenizer has no usable vocab/mask_token -- cannot represent genes missing from the "
            "canonical bundle (module docstring, item 3)"
        )
    return vocab[mask_token]


def build_input_ids(tokenizer, log_expression: np.ndarray, matched_panel_mask: np.ndarray) -> torch.Tensor:
    """``log_expression``: ``(n_samples, n_panel_genes)``, already log10(1+x)
    and in the model's fixed panel order/length, with an arbitrary placeholder
    (0.0) at positions the canonical bundle doesn't cover.
    ``matched_panel_mask``: ``(n_panel_genes,)`` bool, True where a real value
    exists. Missing positions are overwritten with the tokenizer's mask token
    id post-encoding (module docstring, item 3).

    The checkpoint hosted on the Hub as of 2026-09-05 sets
    ``prepend_cls_token: false`` (``tokenizer_config.json``), so
    ``input_ids.shape[1] == n_panel_genes`` with no offset -- but rather than
    hardcode that, this detects a 0 or +1 leading-token offset empirically
    from the tokenizer's actual output, so it stays correct if a different or
    future checkpoint sets ``prepend_cls_token: true`` (module docstring,
    item 2).
    """
    encoded = tokenizer.batch_encode_plus(log_expression, return_tensors="pt")
    input_ids = encoded["input_ids"]
    n_panel_genes = log_expression.shape[1]
    offset = input_ids.shape[1] - n_panel_genes
    if offset not in (0, 1):
        raise RuntimeError(
            f"tokenizer produced input_ids of length {input_ids.shape[1]}, expected {n_panel_genes} "
            f"or {n_panel_genes + 1} (one token per gene, optionally +1 for a leading <cls>) -- "
            "module docstring, item 2. Inspect tokenizer.batch_encode_plus's actual output before "
            "trusting this function's masking offset."
        )
    mask_token_id = resolve_mask_token_id(tokenizer)
    missing_cols = np.where(~matched_panel_mask)[0] + offset
    if len(missing_cols):
        input_ids[:, missing_cols] = mask_token_id
    return input_ids


@torch.no_grad()
def extract_embeddings(
    model,
    tokenizer,
    embedding_layer: int,
    raw_expression: np.ndarray,
    matched_panel_mask: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Batched frozen forward pass -> ``(n_samples, embed_dim)`` mean-pooled embedding.

    Explicitly frees ``result``/``hidden``/``input_ids`` and empties CUDA's
    caching allocator after every batch: a real run (2026-09-05, GPU, fp32,
    batch_size=1) OOM'd on the *second* sample with far more memory "in use"
    than one sample's peak should need, meaning this vendored model's tensors
    were not being released between successive ``model(...)`` calls as
    promptly as assumed -- see ``load_bulkrnabert``'s dtype docstring for the
    other half of this fix (bfloat16).
    """
    n = raw_expression.shape[0]
    log_expression = np.log10(1.0 + np.clip(raw_expression, 0, None))
    outputs = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        input_ids = build_input_ids(tokenizer, log_expression[start:stop], matched_panel_mask).to(device)
        result = model(input_ids=input_ids)
        hidden = result[f"embeddings_{embedding_layer}"]
        pooled = hidden.mean(dim=1).float().cpu().numpy()
        outputs.append(pooled)
        del input_ids, result, hidden
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[extract_embeddings] {stop}/{n} samples", end="\r")
    print()
    return np.concatenate(outputs, axis=0)


def write_cache(output_dir: str | Path, sample_idx: np.ndarray, embeddings: np.ndarray) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "rna_sample_idx.npy", sample_idx.astype(np.int64))
    np.save(out / "rna_zscore.f16.npy", embeddings.astype(np.float16))
    (out / "README.md").write_text(
        "Frozen BulkRNABert embeddings, NOT a z-score -- named `rna_zscore.f16.npy` "
        "only for RNACache (storage.py) on-disk-format compatibility. See "
        "scripts/prepare_bulkrnabert_embeddings.py for provenance.\n"
    )
    print(f"[write_cache] wrote {embeddings.shape} to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--canonical-root", default=None, help="TCGA canonical bundle root (else env/config default)")
    parser.add_argument("--gene-panel", default=None, help="optional: path to a plain gene-id-per-line file; default auto-downloads the checkpoint's own gene order")
    parser.add_argument("--output", required=True, help="output cache directory (RNACache-compatible)")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="O(n^2) attention over ~19,062 gene tokens costs ~11.6 GiB per unit of batch size "
             "just for the attention-weights tensor (see module docstring's memory caveat, found by "
             "actually triggering a CUDA OOM at batch_size=2) -- raise this only after checking your "
             "GPU's actually-free (not total) memory.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"],
        help="the checkpoint is published as float32, but its hand-written attention materializes "
             "the full (batch, heads, n_genes, n_genes) score matrix explicitly -- bfloat16 "
             "(default, matches this repo's own AMP convention) roughly halves that tensor's memory. "
             "Only applied on --device cuda; ignored on cpu (see load_bulkrnabert's docstring).",
    )
    parser.add_argument("--probe-only", action="store_true", help="check gene alignment + input_ids shape on 1 sample (see --batch-size's memory caveat), write nothing")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        # Mitigates the fragmentation PyTorch itself suggested in the real OOM
        # this repo hit (2026-09-05): "reserved but unallocated" memory that
        # couldn't satisfy the next allocation. Must be set before any CUDA
        # context/allocation happens.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    dtype = dtype if args.device.startswith("cuda") else None

    root = resolve_bundle_root(args.canonical_root)
    bundle = TCGACanonicalBundle.from_root(root)
    repo_ensembl_ids = extract_ensembl_ids(bundle.rna.gene_ids)
    panel = load_gene_panel(args.gene_panel) if args.gene_panel else download_gene_panel_from_checkpoint(args.hf_repo)
    repo_columns, panel_positions = align_to_panel(repo_ensembl_ids, panel)
    matched_panel_mask = np.zeros(len(panel), dtype=bool)
    matched_panel_mask[panel_positions] = True

    # All canonical RNA sample_idx -- read via the bundle's own already-open
    # HDF5 handle (bundle.rna.h5), not a second direct h5py.File open, mirroring
    # benchmark/methylprophet/cache.py::prepare_rna_cache's need for the same
    # "every sample_idx in the canonical RNA source" list (tcga_canonical has no
    # public "all ids" accessor beyond the handle itself).
    all_sample_idx = np.asarray(bundle.rna.h5["sample_idx"][...], dtype=np.int64)

    model, tokenizer, embedding_layer = load_bulkrnabert(args.hf_repo, args.device, dtype=dtype)
    n_genes = getattr(model.config, "n_genes", None)
    if n_genes is not None and len(panel) != n_genes:
        # Same invariant the model card's own usage example asserts
        # (`gene_expression_array.shape[1] == config.n_genes`) -- if this
        # ever fails, download_gene_panel_from_checkpoint's "identifier"
        # column filter (or a future checkpoint's panel) needs a look.
        raise RuntimeError(f"gene panel has {len(panel)} genes, checkpoint config declares n_genes={n_genes}")

    def build_panel_row_matrix(sample_idx: np.ndarray) -> np.ndarray:
        raw = bundle.rna.rows(sample_idx)[:, repo_columns]  # (n, len(repo_columns))
        full = np.zeros((raw.shape[0], len(panel)), dtype=np.float32)
        full[:, panel_positions] = raw
        return full

    if args.probe_only:
        probe = build_panel_row_matrix(all_sample_idx[:1])
        log_probe = np.log10(1.0 + np.clip(probe, 0, None))
        input_ids = build_input_ids(tokenizer, log_probe, matched_panel_mask).to(args.device)
        with torch.no_grad():
            result = model(input_ids=input_ids)
        hidden = result[f"embeddings_{embedding_layer}"]
        print(f"[probe-only] forward OK, pooled embedding shape would be {tuple(hidden.mean(dim=1).shape)}")
        return

    full = build_panel_row_matrix(all_sample_idx)
    embeddings = extract_embeddings(
        model, tokenizer, embedding_layer, full, matched_panel_mask,
        device=args.device, batch_size=args.batch_size,
    )
    write_cache(args.output, all_sample_idx, embeddings)


if __name__ == "__main__":
    main()
