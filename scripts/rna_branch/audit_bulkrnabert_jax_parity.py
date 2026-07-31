#!/usr/bin/env python3
"""Numeric parity between the PyTorch translation (extract_bulkrnabert_torch.py) and the
official JAX/Haiku BulkRNABert forward pass, for the *same* input token IDs (tokenization
itself is already independently verified exact by audit_bulkrnabert_inputs.py, so this
script isolates the transformer forward-pass translation).

Three modes, run across two environments (the JAX and PyTorch dependencies conflict in a
single env in this repo's setup -- see requirements-rna-pretrained.txt):

  extract-tokens : builds aligned, tokenized input for one sample and writes token_ids.npz
                   (any env with the trainer's dependencies, e.g. methil-predictor).
  extract-pytorch: loads token_ids.npz, runs BulkRNABertTorch, writes pytorch_layers.npz
                   (env with torch+CUDA, e.g. methil-predictor).
  extract-jax    : loads token_ids.npz, runs the official Haiku forward_fn on CPU (the
                   dense attention matrix is ~5.8GB per sample and OOMs a 12GB GPU
                   regardless of batch size), writes jax_layers.npz
                   (env with jax+haiku, e.g. bulkrnabert).
  compare        : reads pytorch_layers.npz and jax_layers.npz, reports per-layer cosine
                   similarity / relative L2 error for both the pooled embedding and a fixed
                   deterministic subset of token-wise hidden states, against explicit
                   thresholds (numpy-only, works in either env).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

N_TOKEN_SUBSET = 256
FP32_COSINE_MIN = 0.99999
FP32_REL_L2_MAX = 1e-4


def _normalise_gene_id(value: object) -> str:
    fields = [field.strip() for field in str(value).split(";") if field.strip()]
    ensembl = next((field for field in fields if field.upper().startswith("ENSG")), None)
    if ensembl is not None:
        return ensembl.split(".", 1)[0]
    value = fields[0] if fields else str(value)
    return value.split(".", 1)[0] if value.upper().startswith("ENSG") else value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=("extract-tokens", "extract-pytorch", "extract-jax", "compare"))
    parser.add_argument("--config", help="RNA-branch YAML (extract-tokens)")
    parser.add_argument("--official-repo", help="checkout of instadeepai/multiomics-open-research")
    parser.add_argument("--model-name", default="bulk_rna_bert_gtex_encode")
    parser.add_argument("--input-scale", choices=("raw_tpm", "log2p1"), default="log2p1")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--tokens", default="artifacts/rna_branch/audits/bulkrnabert_parity_tokens.npz")
    parser.add_argument("--pytorch-output", default="artifacts/rna_branch/audits/bulkrnabert_parity_pytorch.npz")
    parser.add_argument("--jax-output", default="artifacts/rna_branch/audits/bulkrnabert_parity_jax.npz")
    parser.add_argument("--report-output", default="artifacts/rna_branch/audits/bulkrnabert_jax_parity.json")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def extract_tokens(args: argparse.Namespace) -> None:
    from methylation_predictor.rna_branch.config import load_config
    from methylation_predictor.rna_branch.data import MatrixStore

    repo = Path(args.official_repo).resolve()
    checkpoint_config = json.loads((repo / "checkpoints" / args.model_name / "config.json").read_text())
    common_genes = [
        line.strip()
        for line in (repo / "data" / "bulkrnabert" / "common_gene_id.txt").read_text().splitlines()
        if line.strip()
    ]

    config = load_config(args.config)
    store = MatrixStore(config.data.rna)
    try:
        if store.col_ids is None:
            raise ValueError("RNA matrix requires gene IDs in col_ids_key")
        source_genes = store.col_ids.astype(str)
        row = store.rows(np.asarray([args.sample_index]))
        sample_id = str(store.row_ids[args.sample_index])
    finally:
        store.close()

    if args.input_scale == "log2p1":
        tpm = np.maximum(np.exp2(np.clip(row, 0.0, 30.0)) - 1.0, 0.0)
    else:
        tpm = np.maximum(row, 0.0)

    source = {_normalise_gene_id(gene): index for index, gene in enumerate(source_genes)}
    aligned = np.zeros((1, len(common_genes)), dtype=np.float32)
    for target_index, gene in enumerate(common_genes):
        source_index = source.get(gene)
        if source_index is not None:
            aligned[:, target_index] = tpm[:, source_index]

    values = np.log10(aligned + 1.0)
    values /= float(checkpoint_config["normalization_factor"])
    token_ids = np.digitize(values, np.linspace(0.0, 1.0, int(checkpoint_config["n_expressions_bins"])))
    token_ids = token_ids.astype(np.int64)
    token_ids[values == 0.0] = 0

    rng = np.random.default_rng(args.seed)
    n_subset = min(N_TOKEN_SUBSET, token_ids.shape[1])
    subset_positions = np.sort(rng.choice(token_ids.shape[1], size=n_subset, replace=False))

    output = Path(args.tokens)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        token_ids=token_ids,
        subset_positions=subset_positions,
        sample_id=sample_id,
        model_name=args.model_name,
    )
    print(f"wrote {output}: token_ids {token_ids.shape}, subset {n_subset} positions")


def extract_pytorch(args: argparse.Namespace) -> None:
    import torch

    from extract_bulkrnabert_torch import BulkRNABertTorch  # noqa: local script import, see _load_script convention

    payload = np.load(args.tokens, allow_pickle=True)
    token_ids, subset_positions = payload["token_ids"], payload["subset_positions"]

    repo = Path(args.official_repo).resolve()
    checkpoint = repo / "checkpoints" / str(payload["model_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BulkRNABertTorch(checkpoint / "params.joblib", checkpoint / "config.json", device, torch.float32)
    with torch.inference_mode():
        batch = torch.as_tensor(token_ids, device=device)
        pooled_layers, hidden_state_layers = model(batch, return_hidden_states=True)
        pooled = np.stack([layer.cpu().numpy()[0] for layer in pooled_layers])
        subset = np.stack([layer[0, subset_positions].cpu().numpy() for layer in hidden_state_layers])

    output = Path(args.pytorch_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, pooled=pooled, subset=subset, token_ids=token_ids, subset_positions=subset_positions)
    print(f"wrote {output}: pooled {pooled.shape}, subset {subset.shape}")


def extract_jax(args: argparse.Namespace) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import haiku as hk
    import jax
    import jax.numpy as jnp

    from multiomics_open_research.bulk_rna_bert.pretrained import get_bulkrnabert_pretrained_model

    payload = np.load(args.tokens, allow_pickle=True)
    token_ids, subset_positions = payload["token_ids"], payload["subset_positions"]

    repo = Path(args.official_repo).resolve()
    parameters, forward_fn, _, model_config = get_bulkrnabert_pretrained_model(
        model_name=str(payload["model_name"]),
        compute_dtype=jnp.float32,
        param_dtype=jnp.float32,
        output_dtype=jnp.float32,
        embeddings_layers_to_save=(1, 2, 3, 4),
        checkpoint_directory=str(repo / "checkpoints"),
    )
    transformed = hk.transform(forward_fn)
    token_ids_j = jnp.asarray(token_ids, dtype=jnp.int32)
    attention_mask = jnp.ones((1, 1, token_ids_j.shape[1], token_ids_j.shape[1]), dtype=jnp.bool_)
    result = transformed.apply(parameters, jax.random.PRNGKey(0), token_ids_j, attention_mask)

    pooled = np.stack([np.asarray(result[f"embeddings_{layer}"]).mean(axis=1)[0] for layer in (1, 2, 3, 4)])
    subset = np.stack([np.asarray(result[f"embeddings_{layer}"])[0, subset_positions] for layer in (1, 2, 3, 4)])

    output = Path(args.jax_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, pooled=pooled, subset=subset, token_ids=token_ids, subset_positions=subset_positions)
    print(f"wrote {output}: pooled {pooled.shape}, subset {subset.shape}")


def _cosine_and_rel_l2(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rel_l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
    return cosine, rel_l2


def compare(args: argparse.Namespace) -> None:
    pytorch = np.load(args.pytorch_output)
    jax_ref = np.load(args.jax_output)

    if not np.array_equal(pytorch["token_ids"], jax_ref["token_ids"]):
        raise ValueError("token_ids differ between the pytorch and jax extractions -- not a valid parity check")

    per_layer = []
    for layer_idx in range(pytorch["pooled"].shape[0]):
        pooled_cosine, pooled_rel_l2 = _cosine_and_rel_l2(pytorch["pooled"][layer_idx], jax_ref["pooled"][layer_idx])
        subset_cosine, subset_rel_l2 = _cosine_and_rel_l2(pytorch["subset"][layer_idx], jax_ref["subset"][layer_idx])
        per_layer.append(
            {
                "layer": layer_idx + 1,
                "pooled_cosine": pooled_cosine,
                "pooled_relative_l2": pooled_rel_l2,
                "token_subset_cosine": subset_cosine,
                "token_subset_relative_l2": subset_rel_l2,
                "passed": (
                    pooled_cosine >= FP32_COSINE_MIN
                    and pooled_rel_l2 <= FP32_REL_L2_MAX
                    and subset_cosine >= FP32_COSINE_MIN
                    and subset_rel_l2 <= FP32_REL_L2_MAX
                ),
            }
        )

    report = {
        "thresholds": {"fp32_cosine_min": FP32_COSINE_MIN, "fp32_relative_l2_max": FP32_REL_L2_MAX},
        "per_layer": per_layer,
        "all_layers_passed": all(layer["passed"] for layer in per_layer),
    }
    output = Path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_layers_passed"]:
        raise SystemExit("JAX/PyTorch parity check failed on at least one layer")


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "extract-tokens":
        extract_tokens(args)
    elif args.mode == "extract-pytorch":
        extract_pytorch(args)
    elif args.mode == "extract-jax":
        extract_jax(args)
    elif args.mode == "compare":
        compare(args)


if __name__ == "__main__":
    main()
