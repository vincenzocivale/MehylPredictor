"""Run a controlled gene-encoder intervention on TCGA MDS validation records.

This is deliberately separate from the vendored MethylProphet code.  It feeds
the released checkpoint its ordinary locus inputs twice: once with the factual
gene-expression vector and once with a single fixed, training-set mean vector.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from streaming import StreamingDataset
from torch.nn.utils.rnn import pad_sequence


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mds", type=Path, required=True)
    p.add_argument("--gene-expression", type=Path, required=True)
    p.add_argument("--sample-map", type=Path, required=True)
    p.add_argument("--train-samples", type=Path, required=True,
                   help="Parquet manifest with the training sample_idx column.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-records", type=int, default=32)
    p.add_argument("--cpg-manifest", type=Path, help="Optional parquet cpg_idx manifest")
    p.add_argument("--group-idx", type=int, help="Optional released group restriction")
    p.add_argument("--one-per-cpg", action="store_true", help="Keep the first matching row per CpG")
    p.add_argument("--allow-incomplete-cpg-manifest", action="store_true",
                   help="Permit a shard to cover only a subset of --cpg-manifest")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=9176)
    return p.parse_args()


def _quantize(vector: np.ndarray, bins: int = 51) -> np.ndarray:
    """Compatibility shim; the implementation remains in the upstream submodule."""
    from methylation_predictor.diagnostics.methylprophet.upstream import quantize_gene_expression
    return quantize_gene_expression(vector, bins)


def _collate(rows: list[dict]) -> dict[str, torch.Tensor]:
    seq = [torch.tensor(x.pop("tokenized_sequence"), dtype=torch.long) for x in rows]
    cgi = [torch.tensor(x.pop("tokenized_cgi"), dtype=torch.long) for x in rows]
    seq_ids = pad_sequence(seq, batch_first=True)
    cgi_ids = pad_sequence(cgi, batch_first=True)
    return {
        "gene_expr": torch.tensor(np.stack([x["gene_expr"] for x in rows]), dtype=torch.float32),
        "tokenized_sequence_input_ids": seq_ids,
        "tokenized_sequence_attention_mask": (seq_ids != 0).long(),
        "tokenized_cgi_input_ids": cgi_ids,
        "tokenized_cgi_attention_mask": (cgi_ids != 0).long(),
        "chr_idx": torch.tensor([x["chr_idx"] for x in rows], dtype=torch.long),
        "tissue_idx": torch.tensor([x["tissue_idx"] for x in rows], dtype=torch.long),
    }


def main() -> None:
    a = _args()
    from methylation_predictor.diagnostics.methylprophet.upstream import import_upstream
    import_upstream()
    from src.data.data_preprocessor import CGITokenizer  # pylint: disable=import-outside-toplevel
    from src.models.model_factory import create_model_class, create_model_config_class  # pylint: disable=import-outside-toplevel
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg = OmegaConf.load(a.config)
    cfg.model._attn_implementation = "eager"  # flash-attn is not required for numerical inference
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.val_dataset.dna_tokenizer_name, trust_remote_code=True)
    cgi_tokenizer = CGITokenizer()
    raw = StreamingDataset(local=str(a.mds), shuffle=False)
    rows: list[dict] = []
    wanted = set(pd.read_parquet(a.cpg_manifest)["cpg_idx"].astype(int)) if a.cpg_manifest else None
    seen_cpg: set[int] = set()
    for index in range(len(raw)):
        item = raw[index]
        if a.group_idx is not None and int(item["group_idx"]) != a.group_idx:
            continue
        if wanted is not None and int(item["cpg_idx"]) not in wanted:
            continue
        if a.one_per_cpg and int(item["cpg_idx"]) in seen_cpg:
            continue
        # Raw records in MethylProphetData have sequence/cpg_island_tuple, rather than tokens.
        sequence = item["sequence"].upper()
        middle = len(sequence) // 2
        sequence = sequence[middle - 500: middle + 500]
        item["tokenized_sequence"] = tokenizer.encode(sequence, add_special_tokens=False)
        item["tokenized_cgi"] = cgi_tokenizer(item["cpg_island_tuple"])
        rows.append(item)
        seen_cpg.add(int(item["cpg_idx"]))
        # With an explicit CpG manifest, scan the complete shard (or stop only
        # once every requested locus is found). ``n_records`` is a smoke-test
        # cap and must not silently truncate a requested locus set.
        if wanted is not None and a.one_per_cpg and len(seen_cpg) == len(wanted):
            break
        if wanted is None and len(rows) == a.n_records:
            break
    expected = len(wanted) if wanted is not None and a.one_per_cpg else a.n_records
    if len(rows) != expected and not a.allow_incomplete_cpg_manifest:
        raise RuntimeError(f"Requested {expected} rows but MDS supplied {len(rows)}")
    del raw

    # Delay the 2.2-GB checkpoint and expression matrix until after locating
    # the requested MDS records; full shards otherwise exceed the host cache.
    expr = pd.read_parquet(a.gene_expression)
    sample_map = pd.read_csv(a.sample_map).set_index("sample_idx")["sample_name"].to_dict()
    train_ids = pd.read_parquet(a.train_samples)["sample_idx"].astype(int).tolist()
    train_names = [sample_map[i] for i in train_ids]
    fixed_gene = _quantize(expr[train_names].mean(axis=1).to_numpy(dtype=np.float32))
    for item in rows:
        item["gene_expr"] = _quantize(expr[sample_map[int(item["sample_idx"])]].to_numpy(dtype=np.float32))
    del expr

    model_cfg = create_model_config_class(cfg.model.model_config_class)(**OmegaConf.to_container(cfg.model, resolve=True))
    model = create_model_class(cfg.model.model_class)(model_cfg)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    device = torch.device("cuda")
    model.to(device).eval()

    output_rows = []
    with torch.inference_mode():
        for start in range(0, len(rows), a.batch_size):
            selected = rows[start:start + a.batch_size]
            factual_batch = _collate([copy.deepcopy(r) for r in selected])
            fixed_rows = [copy.deepcopy(r) for r in selected]
            for r in fixed_rows:
                r["gene_expr"] = fixed_gene
            fixed_batch = _collate(fixed_rows)
            factual_batch = {k: v.to(device) for k, v in factual_batch.items()}
            fixed_batch = {k: v.to(device) for k, v in fixed_batch.items()}
            pred_factual = model(**factual_batch).output_value.float().cpu().numpy()
            pred_fixed = model(**fixed_batch).output_value.float().cpu().numpy()
            for r, pf, px in zip(selected, pred_factual, pred_fixed):
                output_rows.extend((
                    {"cpg_idx": int(r["cpg_idx"]), "sample_idx": int(r["sample_idx"]), "group_idx": int(r["group_idx"]), "gt_methyl": float(r["methylation"]), "condition": "factual", "pred_methyl": float(np.asarray(pf).item())},
                    {"cpg_idx": int(r["cpg_idx"]), "sample_idx": int(r["sample_idx"]), "group_idx": int(r["group_idx"]), "gt_methyl": float(r["methylation"]), "condition": "fixed_train_mean", "pred_methyl": float(np.asarray(px).item())},
                ))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_parquet(a.output, index=False)
    manifest = {"n_input_rows": len(rows), "n_prediction_rows": len(output_rows), "fixed_condition": "mean raw expression over train samples, then upstream-equivalent per-vector quantization", "device": str(device), "checkpoint": str(a.checkpoint), "mds": str(a.mds), "group_idx": a.group_idx, "requested_cpg": len(wanted) if wanted is not None else None, "covered_cpg": len(seen_cpg), "missing_cpg": len(wanted - seen_cpg) if wanted is not None else None}
    a.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
