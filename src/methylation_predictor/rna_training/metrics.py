"""Streaming metrics for Array sample x CpG evaluation panels."""
from __future__ import annotations

import numpy as np


class ArrayMomentMetrics:
    def __init__(self, n_samples: int, n_cpgs: int):
        self.n = 0; self.sse = self.sae = self.prior_sse = 0.0
        self.sn = np.zeros(n_samples); self.st = np.zeros(n_samples); self.sp = np.zeros(n_samples)
        self.stt = np.zeros(n_samples); self.spp = np.zeros(n_samples); self.stp = np.zeros(n_samples)
        self.cn = np.zeros(n_cpgs); self.ct = np.zeros(n_cpgs); self.cp = np.zeros(n_cpgs)
        self.ctt = np.zeros(n_cpgs); self.cpp = np.zeros(n_cpgs); self.ctp = np.zeros(n_cpgs)

    @staticmethod
    def _corr(n, sx, sy, sxx, syy, sxy):
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = sxy - sx * sy / n
            vx = sxx - sx * sx / n; vy = syy - sy * sy / n
            return np.where((n >= 2) & (vx * vy > 0), cov / np.sqrt(np.maximum(vx * vy, 1e-30)), np.nan)

    def add(self, s0: int, c0: int, target: np.ndarray, pred: np.ndarray, prior: np.ndarray) -> None:
        valid = np.isfinite(target) & np.isfinite(pred)
        t = np.where(valid, target, 0.0).astype(np.float64)
        p = np.where(valid, pred, 0.0).astype(np.float64)
        pr = np.broadcast_to(np.asarray(prior, np.float64)[None, :], target.shape)
        e = np.where(valid, p-t, 0.0); pe = np.where(valid, pr-t, 0.0)
        ns,nc = target.shape; ss=slice(s0,s0+ns); cs=slice(c0,c0+nc)
        self.n += int(valid.sum()); self.sse += float((e*e).sum()); self.sae += float(np.abs(e).sum()); self.prior_sse += float((pe*pe).sum())
        self.sn[ss] += valid.sum(1); self.st[ss] += t.sum(1); self.sp[ss] += p.sum(1); self.stt[ss] += (t*t).sum(1); self.spp[ss] += (p*p).sum(1); self.stp[ss] += (t*p).sum(1)
        self.cn[cs] += valid.sum(0); self.ct[cs] += t.sum(0); self.cp[cs] += p.sum(0); self.ctt[cs] += (t*t).sum(0); self.cpp[cs] += (p*p).sum(0); self.ctp[cs] += (t*p).sum(0)

    def finalize(self) -> dict[str, float | int]:
        if not self.n:
            return {"rows": 0, "mse": float("nan"), "mae": float("nan"), "prior_mse": float("nan"), "skill_vs_prior": float("nan"), "mas_pcc": float("nan"), "mac_pcc": float("nan")}
        rs = self._corr(self.sn,self.st,self.sp,self.stt,self.spp,self.stp)
        rc = self._corr(self.cn,self.ct,self.cp,self.ctt,self.cpp,self.ctp)
        mse = self.sse/self.n; prior_mse=self.prior_sse/self.n
        return {"rows": int(self.n), "mse": float(mse), "mae": float(self.sae/self.n), "prior_mse": float(prior_mse), "skill_vs_prior": float(1-mse/prior_mse) if prior_mse>0 else float("nan"), "mas_pcc": float(np.nanmedian(rc)), "mac_pcc": float(np.nanmedian(rs))}
