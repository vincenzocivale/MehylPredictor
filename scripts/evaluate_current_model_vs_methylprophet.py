#!/usr/bin/env python3
"""Memory-safe evaluation of the current RNA2DNAmModel on exact MP chr1 views.

Optionally scans a released MethylProphet evaluation parquet dataset and computes
identically-oriented MSE/MAE/MAS-PCC/MAC-PCC on the same protocol IDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Match tcga_canonical.__init__: pyarrow must load before h5py on this server.
import pyarrow.dataset as ds
import h5py  # noqa: F401
import numpy as np
import torch

from methylation_predictor.config import load_config
from methylation_predictor.trainer import ExperimentRunner
from methylation_predictor.utils import json_safe

EXPECTED_FINITE = {
    "train_cpg_x_val_sample": 30_574_946,
    "val_cpg_x_train_sample": 55_155_121,
    "val_cpg_x_val_sample": 6_129_547,
}
GROUP_IDX = {
    "train_cpg_x_val_sample": 0,
    "val_cpg_x_train_sample": 1,
    "val_cpg_x_val_sample": 2,
}


def _pearson_from_moments(n, sx, sy, sxx, syy, sxy):
    n = np.asarray(n, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        denom = vx * vy
        r = np.where((n >= 2) & (denom > 0), cov / np.sqrt(np.where(denom > 0, denom, 1.0)), np.nan)
    return r


class DenseChunkStats:
    def __init__(self, n_samples: int):
        self.n = 0
        self.sse = 0.0
        self.sae = 0.0
        self.prior_sse = 0.0
        self.sample_n = np.zeros(n_samples, np.float64)
        self.sample_st = np.zeros(n_samples, np.float64)
        self.sample_sp = np.zeros(n_samples, np.float64)
        self.sample_stt = np.zeros(n_samples, np.float64)
        self.sample_spp = np.zeros(n_samples, np.float64)
        self.sample_stp = np.zeros(n_samples, np.float64)
        self.sample_model_sse = np.zeros(n_samples, np.float64)
        self.sample_prior_sse = np.zeros(n_samples, np.float64)
        self.cpg_r: list[np.ndarray] = []
        self.cpg_model_mse: list[np.ndarray] = []
        self.cpg_prior_mse: list[np.ndarray] = []

    def add(self, target: np.ndarray, pred: np.ndarray, prior: np.ndarray):
        target = np.asarray(target, np.float64)
        pred = np.asarray(pred, np.float64)
        pm = np.broadcast_to(np.asarray(prior, np.float64)[None, :], target.shape)
        valid = np.isfinite(target) & np.isfinite(pred) & np.isfinite(pm)
        t = np.where(valid, target, 0.0)
        p = np.where(valid, pred, 0.0)
        pr = np.where(valid, pm, 0.0)
        err = np.where(valid, pred - target, 0.0)
        perr = np.where(valid, pm - target, 0.0)
        self.n += int(valid.sum())
        self.sse += float((err * err).sum())
        self.sae += float(np.abs(err).sum())
        self.prior_sse += float((perr * perr).sum())

        self.sample_n += valid.sum(axis=1)
        self.sample_st += t.sum(axis=1); self.sample_sp += p.sum(axis=1)
        self.sample_stt += (t*t).sum(axis=1); self.sample_spp += (p*p).sum(axis=1); self.sample_stp += (t*p).sum(axis=1)
        self.sample_model_sse += (err*err).sum(axis=1); self.sample_prior_sse += (perr*perr).sum(axis=1)

        cn = valid.sum(axis=0).astype(np.float64)
        st = t.sum(axis=0); sp = p.sum(axis=0)
        stt = (t*t).sum(axis=0); spp = (p*p).sum(axis=0); stp = (t*p).sum(axis=0)
        self.cpg_r.append(_pearson_from_moments(cn, st, sp, stt, spp, stp))
        with np.errstate(invalid="ignore", divide="ignore"):
            self.cpg_model_mse.append(np.where(cn > 0, (err*err).sum(axis=0)/cn, np.nan))
            self.cpg_prior_mse.append(np.where(cn > 0, (perr*perr).sum(axis=0)/cn, np.nan))

    def finalize(self):
        r_s = _pearson_from_moments(self.sample_n, self.sample_st, self.sample_sp, self.sample_stt, self.sample_spp, self.sample_stp)
        r_c = np.concatenate(self.cpg_r) if self.cpg_r else np.array([])
        cm = np.concatenate(self.cpg_model_mse) if self.cpg_model_mse else np.array([])
        cp = np.concatenate(self.cpg_prior_mse) if self.cpg_prior_mse else np.array([])
        mse = self.sse / self.n
        pmse = self.prior_sse / self.n
        return {
            "rows": self.n,
            "mse": mse,
            "mae": self.sae / self.n,
            "prior_mse": pmse,
            "skill_vs_prior": 1.0 - mse / pmse if pmse > 0 else float("nan"),
            "mas_pcc": float(np.nanmedian(r_c)),
            "mac_pcc": float(np.nanmedian(r_s)),
            "sample_win_fraction": float(np.nanmean((self.sample_model_sse/self.sample_n) < (self.sample_prior_sse/self.sample_n))),
            "cpg_win_fraction": float(np.nanmean(cm < cp)),
            "per_cpg_mse": cm,
            "per_cpg_prior_mse": cp,
        }


class IndexedStats:
    def __init__(self, sample_ids: np.ndarray, cpg_ids: np.ndarray, prior: np.ndarray):
        self.sample_ids = np.asarray(sample_ids, np.int64)
        self.cpg_ids = np.asarray(cpg_ids, np.int64)
        self.prior = np.asarray(prior, np.float64)
        self.s_order = np.argsort(self.sample_ids); self.s_sorted = self.sample_ids[self.s_order]
        self.c_order = np.argsort(self.cpg_ids); self.c_sorted = self.cpg_ids[self.c_order]
        ns, nc = len(sample_ids), len(cpg_ids)
        self.n = 0; self.sse = 0.0; self.sae = 0.0; self.prior_sse = 0.0
        self.sn=np.zeros(ns); self.st=np.zeros(ns); self.sp=np.zeros(ns); self.stt=np.zeros(ns); self.spp=np.zeros(ns); self.stp=np.zeros(ns)
        self.ssse=np.zeros(ns); self.spse=np.zeros(ns)
        self.cn=np.zeros(nc); self.ct=np.zeros(nc); self.cp=np.zeros(nc); self.ctt=np.zeros(nc); self.cpp=np.zeros(nc); self.ctp=np.zeros(nc)
        self.csse=np.zeros(nc); self.cpse=np.zeros(nc)

    @staticmethod
    def _acc(dst, idx, weights):
        dst += np.bincount(idx, weights=weights, minlength=len(dst))

    def _map(self, values, sorted_values, order, name):
        values = np.asarray(values, np.int64)
        pos = np.searchsorted(sorted_values, values)
        ok = (pos < len(sorted_values))
        if not np.all(ok):
            raise RuntimeError(f"released MP contains {name} outside protocol")
        if not np.array_equal(sorted_values[pos], values):
            raise RuntimeError(f"released MP contains {name} outside protocol")
        return order[pos]

    def add(self, sample, cpg, target, pred):
        sample=np.asarray(sample,np.int64); cpg=np.asarray(cpg,np.int64)
        target=np.asarray(target,np.float64); pred=np.asarray(pred,np.float64)
        si=self._map(sample,self.s_sorted,self.s_order,"sample_idx"); ci=self._map(cpg,self.c_sorted,self.c_order,"cpg_idx")
        valid=np.isfinite(target)&np.isfinite(pred)
        si=si[valid]; ci=ci[valid]; t=target[valid]; p=pred[valid]; pr=self.prior[ci]
        e=p-t; pe=pr-t
        self.n += len(t); self.sse += float(np.dot(e,e)); self.sae += float(np.abs(e).sum()); self.prior_sse += float(np.dot(pe,pe))
        one=np.ones(len(t))
        for dst,w in ((self.sn,one),(self.st,t),(self.sp,p),(self.stt,t*t),(self.spp,p*p),(self.stp,t*p),(self.ssse,e*e),(self.spse,pe*pe)):
            self._acc(dst,si,w)
        for dst,w in ((self.cn,one),(self.ct,t),(self.cp,p),(self.ctt,t*t),(self.cpp,p*p),(self.ctp,t*p),(self.csse,e*e),(self.cpse,pe*pe)):
            self._acc(dst,ci,w)

    def finalize(self):
        rs=_pearson_from_moments(self.sn,self.st,self.sp,self.stt,self.spp,self.stp)
        rc=_pearson_from_moments(self.cn,self.ct,self.cp,self.ctt,self.cpp,self.ctp)
        mse=self.sse/self.n; pmse=self.prior_sse/self.n
        with np.errstate(divide="ignore",invalid="ignore"):
            cm=self.csse/self.cn; cp=self.cpse/self.cn
            sm=self.ssse/self.sn; sp=self.spse/self.sn
        return {"rows":self.n,"mse":mse,"mae":self.sae/self.n,"prior_mse":pmse,"skill_vs_prior":1-mse/pmse,
                "mas_pcc":float(np.nanmedian(rc)),"mac_pcc":float(np.nanmedian(rs)),
                "sample_win_fraction":float(np.nanmean(sm<sp)),"cpg_win_fraction":float(np.nanmean(cm<cp)),
                "per_cpg_mse":cm,"per_cpg_prior_mse":cp}


def _view_indices(runner: ExperimentRunner, name: str):
    if name == "train_cpg_x_val_sample":
        return runner.bundle.sample_indices("validation"), runner.bundle.cpg_indices("train")
    if name == "val_cpg_x_train_sample":
        return runner.bundle.sample_indices("train"), runner.bundle.cpg_indices("validation")
    if name == "val_cpg_x_val_sample":
        return runner.bundle.sample_indices("validation"), runner.bundle.cpg_indices("validation")
    raise KeyError(name)


@torch.no_grad()
def evaluate_ours(runner: ExperimentRunner, name: str):
    sample_idx, cpg_idx = _view_indices(runner, name)
    stats = DenseChunkStats(len(sample_idx))
    runner.model.eval()
    sc = runner.config.evaluation.sample_chunk_size
    cc = runner.config.evaluation.cpg_chunk_size
    for c0 in range(0, len(cpg_idx), cc):
        local_cpg = cpg_idx[c0:c0+cc]
        loci=torch.from_numpy(runner.bundle.loci.embeddings[local_cpg]).to(runner.device)
        prior_np=runner.bundle.loci.prior[local_cpg]
        prior=torch.from_numpy(prior_np).to(runner.device)
        variability=torch.from_numpy(runner.bundle.loci.variability[local_cpg]).to(runner.device)
        tgt=np.full((len(sample_idx),len(local_cpg)),np.nan,np.float32)
        pred=np.full_like(tgt,np.nan)
        for s0 in range(0,len(sample_idx),sc):
            local_s=sample_idx[s0:s0+sc]
            rna=torch.from_numpy(runner.bundle.rna(local_s)).to(runner.device)
            codes=torch.from_numpy(runner.bundle.cancer_type_codes[local_s]).to(runner.device)
            with runner._autocast_context():
                out=runner.model(rna,loci,prior,variability,runner.reference_rna,codes,runner.cancer_centroids)
            tgt[s0:s0+len(local_s)]=runner.bundle.beta(local_s,local_cpg)
            pred[s0:s0+len(local_s)]=out["beta"].float().cpu().numpy()
        stats.add(tgt,pred,prior_np)
    result=stats.finalize()
    result["samples"]=len(sample_idx); result["cpgs"]=len(cpg_idx)
    return result, sample_idx, cpg_idx


def evaluate_mp(eval_root: Path, group: int, sample_ids: np.ndarray, cpg_ids: np.ndarray, prior: np.ndarray, expected_rows: int):
    eval_root = Path(eval_root)
    if eval_root.is_file():
        parquet_source = str(eval_root)
    else:
        parquet_files = sorted(eval_root.rglob("*.parquet"))
        if not parquet_files:
            raise RuntimeError(f"no parquet files found under released MP evaluation: {eval_root}")
        parquet_source = [str(p) for p in parquet_files]
    dataset=ds.dataset(parquet_source,format="parquet")
    cols=["sample_idx","cpg_idx","pred_methyl","gt_methyl"]
    scanner=dataset.scanner(filter=ds.field("group_idx")==group,columns=cols,batch_size=262144)
    stats=IndexedStats(sample_ids,cpg_ids,prior)
    for batch in scanner.to_batches():
        d=batch.to_pydict()
        stats.add(d["sample_idx"],d["cpg_idx"],d["gt_methyl"],d["pred_methyl"])
    result=stats.finalize()
    if result["rows"] != expected_rows:
        raise RuntimeError(f"MP group {group}: finite rows {result['rows']} != expected {expected_rows}")
    return result


def _public(metrics):
    return {k:v for k,v in metrics.items() if not k.startswith("per_cpg_")}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--mp-eval",default=None,help="local released MethylProphet eval parquet directory (optional)")
    p.add_argument("--mp-label",default="MethylProphet released")
    args=p.parse_args()
    cfg=load_config(args.config)
    runner=ExperimentRunner(cfg)
    try:
        ckpt=torch.load(args.checkpoint,map_location=runner.device,weights_only=False)
        runner.model.load_state_dict(ckpt["model_state"],strict=True)
        views={}
        for name in EXPECTED_FINITE:
            ours,sidx,cidx=evaluate_ours(runner,name)
            if ours["rows"] != EXPECTED_FINITE[name]:
                raise RuntimeError(f"{name}: ours rows {ours['rows']} != {EXPECTED_FINITE[name]}")
            entry={"ours":_public(ours)}
            if args.mp_eval:
                sample_ids=runner.bundle.samples.ids[sidx].astype(np.int64)
                cpg_ids=runner.bundle.loci.ids[cidx].astype(np.int64)
                mp=evaluate_mp(Path(args.mp_eval),GROUP_IDX[name],sample_ids,cpg_ids,runner.bundle.loci.prior[cidx],EXPECTED_FINITE[name])
                entry["methylprophet"]=_public(mp)
                entry["delta_ours_minus_mp"]={
                    k: float(ours[k]-mp[k]) for k in ("mse","mae","mas_pcc","mac_pcc")
                }
                valid=np.isfinite(ours["per_cpg_mse"])&np.isfinite(mp["per_cpg_mse"])
                entry["paired_cpg_win_fraction_ours"] = float(np.mean(ours["per_cpg_mse"][valid] < mp["per_cpg_mse"][valid])) if valid.any() else float("nan")
            views[name]=entry
        report={"checkpoint":args.checkpoint,"checkpoint_epoch":ckpt.get("epoch"),"mp_label":args.mp_label if args.mp_eval else None,"views":views}
        out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(json_safe(report),indent=2,sort_keys=True)+"\n")
        print(json.dumps(json_safe(report),indent=2,sort_keys=True))
    finally:
        runner.close()

if __name__ == "__main__":
    main()
