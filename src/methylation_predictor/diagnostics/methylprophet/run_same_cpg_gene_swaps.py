"""Same-CpG, many-patient gene-encoder counterfactual experiment.

Each selected validation locus is held fixed while the model is supplied the
gene-expression vector of many *other* patients from the matching release
quadrant.  Ground truth is joined from the released authoritative rows.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from omegaconf import OmegaConf
from streaming import StreamingDataset
from torch.nn.utils.rnn import pad_sequence

from methylation_predictor.diagnostics.methylprophet.gene_intervention import _quantize


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "config", "mds", "gene_expression", "sample_map", "train_samples", "released_predictions", "output"):
        p.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    p.add_argument("--n-loci", type=int, default=8)
    p.add_argument("--n-patients", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=9176)
    return p


def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
    seq = [torch.tensor(r["tokenized_sequence"], dtype=torch.long) for r in rows]
    cgi = [torch.tensor(r["tokenized_cgi"], dtype=torch.long) for r in rows]
    seq_ids, cgi_ids = pad_sequence(seq, batch_first=True), pad_sequence(cgi, batch_first=True)
    return {"gene_expr": torch.tensor(np.stack([r["gene_expr"] for r in rows]), dtype=torch.float32),
            "tokenized_sequence_input_ids": seq_ids, "tokenized_sequence_attention_mask": (seq_ids != 0).long(),
            "tokenized_cgi_input_ids": cgi_ids, "tokenized_cgi_attention_mask": (cgi_ids != 0).long(),
            "chr_idx": torch.tensor([r["chr_idx"] for r in rows]),
            "tissue_idx": torch.tensor([r["tissue_idx"] for r in rows])}


def main() -> None:
    a = parser().parse_args(); np.random.seed(a.seed); torch.manual_seed(a.seed)
    from methylation_predictor.diagnostics.methylprophet.upstream import import_upstream
    import_upstream()
    from src.data.data_preprocessor import CGITokenizer  # pylint: disable=import-outside-toplevel
    from src.models.model_factory import create_model_class, create_model_config_class  # pylint: disable=import-outside-toplevel
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel
    cfg = OmegaConf.load(a.config); cfg.model._attn_implementation = "eager"
    model = create_model_class(cfg.model.model_class)(create_model_config_class(cfg.model.model_config_class)(**OmegaConf.to_container(cfg.model, resolve=True)))
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=False)
    if missing or unexpected: raise RuntimeError(f"checkpoint mismatch: {missing=} {unexpected=}")
    device = torch.device("cuda"); model.to(device).eval()
    sample_map = pd.read_csv(a.sample_map).set_index("sample_idx")["sample_name"].to_dict()
    patient_ids = pd.read_parquet(a.train_samples)["sample_idx"].astype(int).to_numpy()
    # fixed locus rows from group 1, whose sample dimension is the training patients.
    raw = StreamingDataset(local=str(a.mds), shuffle=False)
    token, cgi = AutoTokenizer.from_pretrained(cfg.data.val_dataset.dna_tokenizer_name, trust_remote_code=True), CGITokenizer()
    loci, seen_cpg = [], set()
    for ix in range(len(raw)):
        item = raw[ix]
        if int(item["group_idx"]) != 1 or int(item["cpg_idx"]) in seen_cpg: continue
        seq = item["sequence"].upper(); mid = len(seq)//2
        item["tokenized_sequence"] = token.encode(seq[mid-500:mid+500], add_special_tokens=False)
        item["tokenized_cgi"] = cgi(item["cpg_island_tuple"])
        loci.append(item)
        seen_cpg.add(int(item["cpg_idx"]))
        if len(loci) == a.n_loci: break
    if len(loci) < a.n_loci: raise RuntimeError("insufficient group-1 loci in MDS")
    # Keep only patient IDs which have an authoritative released target for every selected locus.
    release = ds.dataset(a.released_predictions, format="parquet")
    cpgs = [int(x["cpg_idx"]) for x in loci]
    table = release.to_table(filter=(ds.field("group_idx") == 1) & ds.field("cpg_idx").isin(cpgs), columns=["cpg_idx", "sample_idx", "gt_methyl"])
    targets = table.to_pandas().drop_duplicates(["cpg_idx", "sample_idx"])
    complete = set(patient_ids)
    for cpg in cpgs: complete &= set(targets.loc[targets.cpg_idx == cpg, "sample_idx"])
    patients = np.array(sorted(complete))[:a.n_patients]
    if len(patients) < a.n_patients: raise RuntimeError(f"only {len(patients)} complete patients")
    gt = targets.set_index(["cpg_idx", "sample_idx"])["gt_methyl"].to_dict()
    expr = pd.read_parquet(a.gene_expression, columns=[sample_map[int(x)] for x in patients])
    vectors = {int(s): _quantize(expr[sample_map[int(s)]].to_numpy(dtype=np.float32)) for s in patients}
    rows = []
    for locus in loci:
        for patient in patients:
            row = copy.copy(locus); row["gene_expr"] = vectors[int(patient)]; row["sample_idx"] = int(patient)
            rows.append(row)
    output = []
    with torch.inference_mode():
        for begin in range(0, len(rows), a.batch_size):
            selected = rows[begin:begin+a.batch_size]
            batch = {k:v.to(device) for k,v in collate(selected).items()}
            prediction = model(**batch).output_value.float().cpu().numpy().reshape(-1)
            for row, pred in zip(selected, prediction):
                key = (int(row["cpg_idx"]), int(row["sample_idx"]))
                output.append({"cpg_idx": key[0], "sample_idx": key[1], "gt_methyl": float(gt[key]), "pred_methyl": float(pred)})
    frame = pd.DataFrame(output); a.output.parent.mkdir(parents=True, exist_ok=True); frame.to_parquet(a.output, index=False)
    per_cpg = []
    pooled_sse = pooled_ss = 0.0
    for cpg, part in frame.groupby("cpg_idx", sort=True):
        y, pred = part.gt_methyl.to_numpy(), part.pred_methyl.to_numpy()
        yc, pc = y-y.mean(), pred-pred.mean()
        denominator = np.dot(yc,yc)
        sse = np.dot(pc-yc,pc-yc); pooled_sse += sse; pooled_ss += denominator
        per_cpg.append({"cpg_idx": int(cpg), "n_patients":len(part), "target_std":float(y.std(ddof=0)), "prediction_std":float(pred.std(ddof=0)), "residual_correlation":float(np.corrcoef(yc,pc)[0,1]) if y.std() and pred.std() else float("nan"), "dynamic_skill":float(1-sse/denominator) if denominator else float("nan"), "centered_sse":float(sse), "centered_ss":float(denominator)})
    per_cpg_frame = pd.DataFrame(per_cpg)
    per_cpg_frame.to_parquet(a.output.with_name(a.output.stem + "_per_cpg.parquet"), index=False)
    a.output.with_suffix(".json").write_text(json.dumps({"n_loci":len(loci),"n_patients":len(patients),"n_rows":len(frame),"group":1,"design":"same locus, swapped training-patient gene vectors; GT from released group 1","pooled_dynamic_skill":float(1-pooled_sse/pooled_ss),"median_per_cpg_dynamic_skill":float(per_cpg_frame.dynamic_skill.median()),"median_prediction_std":float(per_cpg_frame.prediction_std.median())}, indent=2, allow_nan=False)+"\n")

if __name__ == "__main__": main()
