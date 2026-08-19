"""Export CpGStatisticsPredictor outputs to the feature-cache contract consumed by RNA training."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .model import CpGStatisticsModelConfig, CpGStatisticsPredictor
from ..run_store import sha256_file
from ..storage import SortedIndex, read_h5_rows


def export_feature_cache(*,checkpoint,targets_dir,embeddings_h5,output,batch_size=4096,empirical_for_train=True):
    target_root=Path(targets_dir); ids=np.load(target_root/"cpg_idx.npy"); target_mu=np.load(target_root/"target_mu.npy"); target_sigma=np.load(target_root/"target_sigma.npy"); train_mask=np.load(target_root/"official_train_mask.npy").astype(bool)
    with h5py.File(embeddings_h5,"r") as h:
        atlas_ids=np.asarray(h["cpg_idx"][...],np.int64); rows=SortedIndex(atlas_ids,"NTv3 atlas").positions_of(ids); embeddings=read_h5_rows(h["embeddings"],rows,dtype=np.float16)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); state=torch.load(checkpoint,map_location=device,weights_only=False); cfg_raw=state.get("model_config",{}); cfg=CpGStatisticsModelConfig(**{**cfg_raw,"ensemble_seeds":tuple(cfg_raw.get("ensemble_seeds",(17,29,43)))})
    model=CpGStatisticsPredictor(cfg).to(device); model.load_state_dict(state["model_state"],strict=True); model.eval(); mu=[]; sigma=[]
    with torch.no_grad():
        for start in range(0,len(ids),batch_size):
            x=torch.from_numpy(embeddings[start:start+batch_size].astype(np.float32)).to(device); out=model(x); mu.append(out["mu"].float().cpu().numpy()); sigma.append(out["sigma"].float().cpu().numpy())
    mu=np.concatenate(mu); sigma=np.concatenate(sigma)
    if empirical_for_train:
        mu[train_mask]=target_mu[train_mask]; sigma[train_mask]=target_sigma[train_mask]
    out=Path(output); out.mkdir(parents=True,exist_ok=True); np.save(out/"cpg_idx.npy",ids); np.save(out/"embeddings.f16.npy",embeddings); np.save(out/"prior.npy",mu.astype(np.float32)); np.save(out/"sigma.npy",sigma.astype(np.float32)); manifest={"schema_version":1,"source_checkpoint":str(checkpoint),"source_checkpoint_sha256":sha256_file(checkpoint),"scope":state.get("scope"),"empirical_for_official_train_cpgs":bool(empirical_for_train),"heldout_statistics":"predicted from NTv3 only","cpgs":int(len(ids))}; (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); return manifest
