#!/usr/bin/env python3
"""Extract frozen BulkFormer sample embeddings from an HDF5 RNA matrix.

The script follows the official BulkFormer preprocessing contract:

* genes are aligned to ``bulkformer_gene_info.csv`` (20,010 Ensembl IDs);
* absent genes are filled with the model mask token ``-10``;
* expression is supplied as natural-log ``log1p(TPM)``;
* the released checkpoint is used frozen;
* ``official_mean`` exactly reproduces the repository's sample-level mean
  aggregation over all gene-token outputs.

Additional observed-gene/core summaries are emitted only as diagnostics.  The
primary benchmark configuration must name one representation per encoder and
must not select among these summaries after looking at validation/test results.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd


_MODEL_SPECS: dict[str, dict[str, int]] = {
    "37M": {"dim": 128, "bins": 0, "gb_repeat": 1, "p_repeat": 1, "bin_head": 12, "full_head": 8},
    "50M": {"dim": 256, "bins": 0, "gb_repeat": 1, "p_repeat": 2, "bin_head": 12, "full_head": 8},
    "93M": {"dim": 512, "bins": 0, "gb_repeat": 1, "p_repeat": 6, "bin_head": 12, "full_head": 8},
    "127M": {"dim": 640, "bins": 0, "gb_repeat": 1, "p_repeat": 8, "bin_head": 12, "full_head": 8},
    "147M": {"dim": 640, "bins": 0, "gb_repeat": 1, "p_repeat": 12, "bin_head": 12, "full_head": 8},
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulkformer-repo", required=True, help="Checkout of KangBoming/BulkFormer")
    parser.add_argument("--checkpoint", required=True, help="Released BulkFormer checkpoint")
    parser.add_argument("--graph", required=True, help="Graph edge-index tensor, e.g. data/G_tcga.pt")
    parser.add_argument("--graph-weights", required=True, help="Graph edge weights")
    parser.add_argument("--gene-embedding", required=True, help="Released ESM2 gene feature tensor")
    parser.add_argument("--gene-info", required=True, help="bulkformer_gene_info.csv")
    parser.add_argument(
        "--interested-gene-indices",
        help="Official interested_gene_list.pt used by the upstream sample-level extraction notebook",
    )
    parser.add_argument("--gene-id-column", default="ensg_id")
    parser.add_argument("--model-scale", choices=tuple(_MODEL_SPECS), default="147M")
    parser.add_argument("--rna-h5", required=True)
    parser.add_argument("--values-key", default="X")
    parser.add_argument("--sample-ids-key", default="sample_idx")
    parser.add_argument("--gene-ids-key", default="gene_ids")
    parser.add_argument(
        "--input-scale",
        choices=("raw_tpm", "log2p1_tpm", "ln1p_tpm", "log10p1_tpm"),
        default="log2p1_tpm",
        help="Scale stored in --rna-h5. BulkFormer receives natural log1p(TPM).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-gene-overlap", type=float, default=0.95)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--pretraining-overlap-status",
        choices=("unknown", "excluded", "present"),
        default="unknown",
        help="Recorded provenance only; use 'excluded' solely with accession-level evidence.",
    )
    return parser


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in np.asarray(values).tolist()],
        dtype=object,
    )


def _normalise_gene_id(value: object) -> str:
    fields = [field.strip() for field in str(value).split(";") if field.strip()]
    ensembl = next((field for field in fields if field.upper().startswith("ENSG")), None)
    chosen = ensembl if ensembl is not None else (fields[0] if fields else str(value))
    return chosen.split(".", 1)[0].upper() if chosen.upper().startswith("ENSG") else chosen


def _unique_gene_index(values: Iterable[object], label: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for index, value in enumerate(values):
        gene = _normalise_gene_id(value)
        if gene in output:
            raise ValueError(f"duplicate normalized {label} gene ID {gene!r}")
        output[gene] = index
    return output


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(torch: Any, path: str | Path, *, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0 compatibility
        return torch.load(path, map_location=map_location)


def _state_dict(checkpoint: Any) -> OrderedDict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("BulkFormer checkpoint is not a state-dict mapping")
    cleaned: OrderedDict[str, Any] = OrderedDict()
    for key, value in checkpoint.items():
        name = str(key)
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def _to_ln1p(values: np.ndarray, scale: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("RNA matrix contains non-finite values")
    if scale == "raw_tpm":
        if np.min(values) < -1e-6:
            raise ValueError("raw TPM contains negative values")
        return np.log1p(np.maximum(values, 0.0)).astype(np.float32)
    if np.min(values) < -1e-5:
        raise ValueError(f"{scale} contains negative values")
    if scale == "log2p1_tpm":
        return (values * np.float32(math.log(2.0))).astype(np.float32)
    if scale == "ln1p_tpm":
        return values
    if scale == "log10p1_tpm":
        return (values * np.float32(math.log(10.0))).astype(np.float32)
    raise ValueError(scale)


def _validate_paths(args: argparse.Namespace) -> Path:
    repo = Path(args.bulkformer_repo).resolve()
    expected = repo / "utils" / "BulkFormer.py"
    if not expected.is_file():
        raise FileNotFoundError(f"invalid BulkFormer checkout; missing {expected}")
    required_paths = [args.checkpoint, args.graph, args.graph_weights, args.gene_embedding, args.gene_info, args.rna_h5]
    if args.interested_gene_indices:
        required_paths.append(args.interested_gene_indices)
    for value in required_paths:
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not 0.0 <= args.min_gene_overlap <= 1.0:
        raise ValueError("--min-gene-overlap must be in [0, 1]")
    return repo


def _load_model(args: argparse.Namespace, repo: Path) -> tuple[Any, Any, dict[str, int]]:
    sys.path.insert(0, str(repo))
    try:
        import torch
        from torch_geometric.typing import SparseTensor
        from utils.BulkFormer import BulkFormer
    except Exception as exc:  # pragma: no cover - depends on external CUDA environment
        raise RuntimeError(
            "BulkFormer dependencies are unavailable. Create the official bulkformer.yaml environment "
            "and install torch-geometric/torch-sparse for the active CUDA build."
        ) from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    graph_index = _torch_load(torch, args.graph)
    graph_weights = _torch_load(torch, args.graph_weights)
    try:
        graph_size = len(graph_index)
    except TypeError as exc:
        raise ValueError("graph file must contain a two-row edge-index object") from exc
    if graph_size < 2:
        raise ValueError("graph file must contain a two-row edge-index object")
    graph = SparseTensor(row=graph_index[1], col=graph_index[0], value=graph_weights).t().to(args.device)
    gene_embedding = _torch_load(torch, args.gene_embedding)
    params = dict(_MODEL_SPECS[args.model_scale])
    params["gene_length"] = 20010
    params["graph"] = graph
    params["gene_emb"] = gene_embedding
    model = BulkFormer(**params).to(args.device)
    incompatible = model.load_state_dict(_state_dict(_torch_load(torch, args.checkpoint)), strict=True)
    if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    model.eval()
    return torch, model, params


def _write_output(
    output: Path,
    sample_ids: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(output, "w") as handle:
        handle.create_dataset("sample_idx", data=np.asarray(sample_ids.astype(str), dtype=object), dtype=string_dtype)
        for name, values in arrays.items():
            handle.create_dataset(name, data=np.asarray(values, dtype=np.float32), compression="gzip", shuffle=True)
        handle.attrs["transcriptome_only"] = True
        handle.attrs["encoder"] = "BulkFormer"
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def extract(args: argparse.Namespace) -> Path:
    repo = _validate_paths(args)
    gene_info = pd.read_csv(args.gene_info)
    if args.gene_id_column not in gene_info.columns:
        raise KeyError(f"{args.gene_info} lacks column {args.gene_id_column!r}")
    model_genes = gene_info[args.gene_id_column].astype(str).to_numpy(dtype=object)
    if len(model_genes) != 20010:
        raise ValueError(f"BulkFormer vocabulary must contain 20,010 genes, found {len(model_genes)}")
    model_gene_index = _unique_gene_index(model_genes, "BulkFormer")
    if len(model_gene_index) != len(model_genes):
        raise ValueError("BulkFormer vocabulary contains duplicate normalized IDs")

    torch, model, params = _load_model(args, repo)
    output = Path(args.output)
    with h5py.File(args.rna_h5, "r") as source:
        required = {args.values_key, args.sample_ids_key, args.gene_ids_key}
        missing = sorted(required - set(source.keys()))
        if missing:
            raise KeyError(f"RNA HDF5 lacks datasets: {missing}")
        matrix = source[args.values_key]
        sample_ids = _decode(source[args.sample_ids_key][...])
        source_genes = _decode(source[args.gene_ids_key][...])
        if matrix.ndim != 2 or matrix.shape != (len(sample_ids), len(source_genes)):
            raise ValueError(
                f"RNA matrix shape {matrix.shape} does not match sample/gene axes "
                f"({len(sample_ids)}, {len(source_genes)})"
            )
        if len(set(sample_ids.astype(str))) != len(sample_ids):
            raise ValueError("RNA sample IDs are not unique")
        source_index = _unique_gene_index(source_genes, "RNA")
        target_to_source = np.full(len(model_genes), -1, dtype=np.int64)
        for target_index, gene in enumerate(model_genes):
            target_to_source[target_index] = source_index.get(_normalise_gene_id(gene), -1)
        observed_targets = np.flatnonzero(target_to_source >= 0)
        observed_sources = target_to_source[observed_targets]
        overlap = len(observed_targets) / len(model_genes)
        if overlap < args.min_gene_overlap:
            raise ValueError(f"BulkFormer gene overlap {overlap:.4f} < {args.min_gene_overlap:.4f}")
        mask_probability = float(1.0 - overlap)
        observed_mask_np = np.zeros(len(model_genes), dtype=np.float32)
        observed_mask_np[observed_targets] = 1.0

        n_samples = len(sample_ids)
        dim = int(params["dim"])
        if args.interested_gene_indices:
            raw_indices = _torch_load(torch, args.interested_gene_indices)
            if hasattr(raw_indices, "detach"):
                raw_indices = raw_indices.detach().cpu().numpy()
            official_indices_np = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
            if not len(official_indices_np):
                raise ValueError("official interested-gene index list is empty")
            if np.min(official_indices_np) < 0 or np.max(official_indices_np) >= len(model_genes):
                raise ValueError("official interested-gene indices fall outside the BulkFormer vocabulary")
            if len(np.unique(official_indices_np)) != len(official_indices_np):
                raise ValueError("official interested-gene index list contains duplicates")
        else:
            official_indices_np = np.arange(len(model_genes), dtype=np.int64)
        official_indices = torch.as_tensor(official_indices_np, device=args.device, dtype=torch.long)

        arrays = {
            # Exact repository aggregation over the supplied interested_gene_list.pt.
            "official_mean": np.empty((n_samples, dim + 3), dtype=np.float32),
            # Diagnostics; deliberately excluded from the preregistered primary config.
            "all_gene_mean": np.empty((n_samples, dim + 3), dtype=np.float32),
            "observed_gene_mean": np.empty((n_samples, dim + 3), dtype=np.float32),
            "observed_core_mean": np.empty((n_samples, dim), dtype=np.float32),
            "observed_core_mean_std": np.empty((n_samples, 2 * dim), dtype=np.float32),
            "global_features": np.empty((n_samples, 3), dtype=np.float32),
        }
        observed_mask = torch.as_tensor(observed_mask_np, device=args.device).view(1, -1, 1)
        use_amp = args.device.startswith("cuda") and not args.disable_amp

        with torch.inference_mode():
            progress = tqdm(
                range(0, n_samples, args.batch_size),
                total=math.ceil(n_samples / args.batch_size),
                desc="BulkFormer inference",
                unit="batch",
            )
            for start in progress:
                stop = min(start + args.batch_size, n_samples)
                rows = np.asarray(matrix[start:stop, :], dtype=np.float32)
                aligned = np.full((stop - start, len(model_genes)), -10.0, dtype=np.float32)
                aligned[:, observed_targets] = _to_ln1p(rows[:, observed_sources], args.input_scale)
                x = torch.as_tensor(aligned, device=args.device, dtype=torch.float32)
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
                    if use_amp
                    else contextlib.nullcontext()
                )
                with amp_context:
                    tokens = model(x, mask_prob=mask_probability, output_expr=False)
                tokens = tokens.float()
                official = tokens.index_select(1, official_indices).mean(dim=1)
                all_gene_mean = tokens.mean(dim=1)
                denominator = observed_mask.sum(dim=1).clamp_min(1.0)
                observed_mean = (tokens * observed_mask).sum(dim=1) / denominator
                core = tokens[..., :dim]
                core_mean = (core * observed_mask).sum(dim=1) / denominator
                centered = (core - core_mean.unsqueeze(1)) * observed_mask
                core_std = torch.sqrt((centered.square().sum(dim=1) / denominator).clamp_min(0.0))
                arrays["official_mean"][start:stop] = official.cpu().numpy()
                arrays["all_gene_mean"][start:stop] = all_gene_mean.cpu().numpy()
                arrays["observed_gene_mean"][start:stop] = observed_mean.cpu().numpy()
                arrays["observed_core_mean"][start:stop] = core_mean.cpu().numpy()
                arrays["observed_core_mean_std"][start:stop] = torch.cat([core_mean, core_std], dim=1).cpu().numpy()
                arrays["global_features"][start:stop] = tokens[:, 0, -3:].cpu().numpy()
                del x, tokens, official, all_gene_mean, observed_mean, core, core_mean, core_std
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    metadata: dict[str, Any] = {
        "encoder": "BulkFormer",
        "model_scale": args.model_scale,
        "model_params": {key: int(value) for key, value in params.items() if key not in {"graph", "gene_emb"}},
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "bulkformer_repo": str(repo),
        "graph": str(Path(args.graph).resolve()),
        "graph_sha256": _sha256(args.graph),
        "graph_weights": str(Path(args.graph_weights).resolve()),
        "graph_weights_sha256": _sha256(args.graph_weights),
        "gene_embedding": str(Path(args.gene_embedding).resolve()),
        "gene_embedding_sha256": _sha256(args.gene_embedding),
        "gene_info": str(Path(args.gene_info).resolve()),
        "gene_info_sha256": _sha256(args.gene_info),
        "interested_gene_indices": None if not args.interested_gene_indices else str(Path(args.interested_gene_indices).resolve()),
        "interested_gene_indices_sha256": None if not args.interested_gene_indices else _sha256(args.interested_gene_indices),
        "interested_gene_count": int(len(official_indices_np)),
        "rna_h5": str(Path(args.rna_h5).resolve()),
        "input_scale": args.input_scale,
        "model_input_scale": "ln1p_tpm",
        "gene_overlap_fraction": overlap,
        "matched_gene_count": int(len(observed_targets)),
        "vocabulary_gene_count": int(len(model_genes)),
        "mask_probability": mask_probability,
        "sample_count": int(len(sample_ids)),
        "datasets": {name: list(values.shape) for name, values in arrays.items()},
        "primary_representation": "official_mean",
        "primary_representation_definition": (
            "mean over gene_emb_output tokens selected by interested_gene_list.pt"
            if args.interested_gene_indices
            else "mean over all gene_emb_output tokens (fallback; no interested list supplied)"
        ),
        "pretraining_overlap_status": args.pretraining_overlap_status,
        "pretraining_overlap_warning": (
            "BulkFormer accession-level pretraining membership must be independently verified before making a "
            "clean-transfer claim on TCGA. The intrinsic comparison remains descriptive when status is unknown."
        ),
        "methylation_inputs_loaded": False,
        "downstream_regressor_trained": False,
    }
    _write_output(output, sample_ids, arrays, metadata)
    return output


def main() -> None:
    args = _parser().parse_args()
    print(extract(args))


if __name__ == "__main__":
    main()
