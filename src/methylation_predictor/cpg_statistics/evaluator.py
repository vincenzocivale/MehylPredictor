"""Evaluate a CpGStatisticsPredictor checkpoint on an arbitrary genomic scope."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from .model import CpGStatisticsModelConfig, CpGStatisticsPredictor
from .trainer import statistics_metrics
from ..run_store import sha256_file, write_json
from ..scopes import filter_cpg_ids
from ..storage import SortedIndex, read_h5_rows


def evaluate_statistics_checkpoint(*,checkpoint,targets_dir,embeddings_h5,registry,eval_scope,output,batch_size=4096):
    root=Path(targets_dir); ids=np.load(root/"cpg_idx.npy"); mu=np.load(root/"target_mu.npy"); sigma=np.load(root/"target_sigma.npy"); heldout=np.load(root/"official_val_mask.npy").astype(bool)
    eval_ids=filter_cpg_ids(ids[heldout],eval_scope,registry); index=SortedIndex(ids,"statistics targets"); rows=index.positions_of(eval_ids)
    with h5py.File(embeddings_h5,"r") as h:
        emb_ids=np.asarray(h["cpg_idx"][...],np.int64); emb_rows=SortedIndex(emb_ids,"NTv3 embeddings").positions_of(eval_ids); emb=read_h5_rows(h["embeddings"],emb_rows,dtype=np.float32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); state=torch.load(checkpoint,map_location=device,weights_only=False); cfg_raw=state.get("model_config",{}); cfg=CpGStatisticsModelConfig(**{**cfg_raw,"ensemble_seeds":tuple(cfg_raw.get("ensemble_seeds",(17,29,43)))})
    model=CpGStatisticsPredictor(cfg).to(device); model.load_state_dict(state["model_state"],strict=True); model.eval(); pred_mu=[]; pred_sigma=[]
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            x=torch.from_numpy(emb[start:start+batch_size]).to(device); out=model(x); pred_mu.append(out["mu"].float().cpu().numpy()); pred_sigma.append(out["sigma"].float().cpu().numpy())
    metrics=statistics_metrics(mu[rows],sigma[rows],np.concatenate(pred_mu),np.concatenate(pred_sigma)); result={"schema_version":1,"training_scope":state.get("scope"),"evaluation_scope":eval_scope,"checkpoint":str(checkpoint),"checkpoint_sha256":sha256_file(checkpoint),"cpgs":int(len(rows)),**metrics}; out=Path(output); out.mkdir(parents=True,exist_ok=True); write_json(out/"metrics.json",result); write_json(out/"manifest.json",{"model":"cpg_statistics","training_scope":state.get("scope"),"evaluation_scope":eval_scope,"checkpoint_sha256":result["checkpoint_sha256"]}); return result
