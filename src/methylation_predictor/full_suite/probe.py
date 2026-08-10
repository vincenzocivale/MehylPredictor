from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


TARGET_NAMES = ("prior_logit", "log_var_between", "log_var_within")


class ScalarProbeNet(nn.Module):
    """Historical probe shape: LayerNorm -> 256 -> 64 -> 1."""

    def __init__(self, dim: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ProbeSeed(nn.Module):
    """Three independent scalar probes: prior, between-var, within-var."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([ScalarProbeNet(dim, dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(x) for head in self.heads], dim=-1)


class ProbeEnsemble(nn.Module):
    def __init__(self, seeds: list[int], dim: int = 1536, dropout: float = 0.1):
        super().__init__()
        nets = []
        for seed in seeds:
            torch.manual_seed(seed)
            nets.append(ProbeSeed(dim, dropout))
        self.nets = nn.ModuleList(nets)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([net(x) for net in self.nets], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all(x).mean(dim=0)


def _stable_validation_mask(ids: np.ndarray, fraction: float = 0.1, seed: int = 20260809) -> np.ndarray:
    # Deterministic id hash without Python's randomized hash().
    x = np.asarray(ids, dtype=np.uint64)
    z = x + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    threshold = int(round(fraction * 10_000))
    return (z % np.uint64(10_000)) < np.uint64(threshold)


def _compute_embedding_stats(embeddings: np.ndarray, rows: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)
    sx = np.zeros(embeddings.shape[1], dtype=np.float64)
    sxx = np.zeros_like(sx)
    for start in range(0, n, chunk):
        x = np.asarray(embeddings[rows[start:start + chunk]], dtype=np.float32)
        sx += x.sum(axis=0, dtype=np.float64)
        sxx += np.square(x, dtype=np.float64).sum(axis=0)
    mean = sx / n
    var = np.maximum(sxx / n - mean * mean, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _target_matrix(prior: np.ndarray, variability: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(np.asarray(prior, np.float32), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return np.column_stack([logit, variability[:, 0], variability[:, 1]]).astype(np.float32)


def fit_feature_extension_probe(
    base_cache_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    seeds: tuple[int, ...] = (17, 29, 43),
    max_epochs: int = 40,
    patience: int = 6,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    block_rows: int = 4096,
    validation_fraction: float = 0.10,
    fit_cpg_ids: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit a frozen NTv3->current-feature distillation probe.

    Base CpGs continue to use their *exact* existing features.  The probe is
    used only on newly extracted NTv3 loci, which keeps E1 unchanged while
    extending the current model's input contract to EPIC/WGBS.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for feature probe but unavailable")
    base = Path(base_cache_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "feature_extension_probe.pt"
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return {"status": "cached", "best_epoch": int(state["best_epoch"]), "validation_mse": state["validation_mse"]}

    ids = np.load(base / "base_cpg_idx.npy", mmap_mode="r")
    emb_path = next(iter(base.glob("base_embeddings.*.npy")))
    embeddings = np.load(emb_path, mmap_mode="r")
    prior = np.load(base / "base_prior.npy", mmap_mode="r")
    variability = np.load(base / "base_variability.npy", mmap_mode="r")
    target = _target_matrix(prior, variability)
    if fit_cpg_ids is None:
        candidate_rows = np.arange(len(ids), dtype=np.int64)
        fit_scope = "all_base_loci"
    else:
        from .feature_store import SortedIndex
        candidate_rows = SortedIndex(ids, "base probe ids").positions_of(np.asarray(fit_cpg_ids, dtype=np.int64))
        candidate_rows = np.sort(candidate_rows)
        fit_scope = "official_heldout_full_fit_loci"
    local_val = _stable_validation_mask(np.asarray(ids[candidate_rows], dtype=np.int64), validation_fraction)
    train_rows = candidate_rows[~local_val]
    val_rows = candidate_rows[local_val]
    if len(val_rows) < 1000 or len(train_rows) < 1000:
        raise RuntimeError("feature-probe train/validation split is unexpectedly small")

    x_mean, x_std = _compute_embedding_stats(embeddings, train_rows)
    y_train = target[train_rows]
    y_mean = y_train.mean(axis=0).astype(np.float32)
    y_std = np.maximum(y_train.std(axis=0), 1e-6).astype(np.float32)

    ensemble = ProbeEnsemble(list(seeds), dim=embeddings.shape[1], dropout=dropout).to(device)
    opt = torch.optim.AdamW(ensemble.parameters(), lr=learning_rate, weight_decay=weight_decay)
    use_amp = device.startswith("cuda")
    best = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    history = []
    train_blocks = [train_rows[start:start + block_rows] for start in range(0, len(train_rows), block_rows)]
    rng = np.random.default_rng(20260809)
    started = time.time()

    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)
    y_mean_t = torch.from_numpy(y_mean).to(device)
    y_std_t = torch.from_numpy(y_std).to(device)

    for epoch in range(1, max_epochs + 1):
        ensemble.train()
        order = rng.permutation(len(train_blocks))
        losses = []
        for bi in order:
            rows = train_blocks[int(bi)]
            x = torch.from_numpy(np.asarray(embeddings[rows], dtype=np.float32)).to(device)
            y = torch.from_numpy(target[rows]).to(device)
            x = (x - x_mean_t) / x_std_t
            y = (y - y_mean_t) / y_std_t
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                all_pred = ensemble.forward_all(x)
                loss = torch.stack([torch.nn.functional.mse_loss(p, y) for p in all_pred]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        ensemble.eval()
        channel_sse = np.zeros(3, dtype=np.float64)
        count = 0
        with torch.inference_mode():
            for start in range(0, len(val_rows), block_rows):
                rows = val_rows[start:start + block_rows]
                x = torch.from_numpy(np.asarray(embeddings[rows], dtype=np.float32)).to(device)
                y = target[rows]
                x = (x - x_mean_t) / x_std_t
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred_z = ensemble(x)
                pred = pred_z.float().cpu().numpy() * y_std + y_mean
                channel_sse += np.square(pred - y, dtype=np.float64).sum(axis=0)
                count += len(rows)
        channel_mse = channel_sse / max(count, 1)
        val_mse = float(channel_mse.mean())
        row = {"epoch": epoch, "train_mse_z": float(np.mean(losses)), "validation_mse": val_mse,
               **{f"validation_mse_{name}": float(v) for name, v in zip(TARGET_NAMES, channel_mse)}}
        history.append(row)
        print(f"[feature-probe] {row}", flush=True)
        if val_mse < best - 1e-5:
            best = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in ensemble.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is None:
        raise RuntimeError("feature probe did not produce a finite validation checkpoint")

    # Refit from the same seeded initialization on the complete full-fit target
    # locus set for exactly the selected epoch count.  This mirrors the model's
    # development/final-refit discipline and avoids throwing away the probe's
    # validation loci in the deployed extension mapping.
    x_mean_final, x_std_final = _compute_embedding_stats(embeddings, candidate_rows)
    y_final = target[candidate_rows]
    y_mean_final = y_final.mean(axis=0).astype(np.float32)
    y_std_final = np.maximum(y_final.std(axis=0), 1e-6).astype(np.float32)
    x_mean_final_t = torch.from_numpy(x_mean_final).to(device)
    x_std_final_t = torch.from_numpy(x_std_final).to(device)
    y_mean_final_t = torch.from_numpy(y_mean_final).to(device)
    y_std_final_t = torch.from_numpy(y_std_final).to(device)

    refit = ProbeEnsemble(list(seeds), dim=embeddings.shape[1], dropout=dropout).to(device)
    refit_opt = torch.optim.AdamW(refit.parameters(), lr=learning_rate, weight_decay=weight_decay)
    all_blocks = [candidate_rows[start:start + block_rows] for start in range(0, len(candidate_rows), block_rows)]
    refit_rng = np.random.default_rng(20260809)
    for epoch in range(1, best_epoch + 1):
        refit.train()
        for bi in refit_rng.permutation(len(all_blocks)):
            rows = all_blocks[int(bi)]
            x = torch.from_numpy(np.asarray(embeddings[rows], dtype=np.float32)).to(device)
            y = torch.from_numpy(target[rows]).to(device)
            x = (x - x_mean_final_t) / x_std_final_t
            y = (y - y_mean_final_t) / y_std_final_t
            refit_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                all_pred = refit.forward_all(x)
                loss = torch.stack([torch.nn.functional.mse_loss(pred, y) for pred in all_pred]).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(refit.parameters(), 1.0); refit_opt.step()
    refit_state = {k: v.detach().cpu().clone() for k, v in refit.state_dict().items()}

    state = {
        "model_state": refit_state,
        "seeds": list(seeds),
        "dim": int(embeddings.shape[1]),
        "dropout": float(dropout),
        "x_mean": x_mean_final,
        "x_std": x_std_final,
        "y_mean": y_mean_final,
        "y_std": y_std_final,
        "target_names": list(TARGET_NAMES),
        "best_epoch": int(best_epoch),
        "validation_mse": float(best),
        "history": history,
        "design": "distill existing frozen locus_features; base rows remain bit-identical; probe used only for new CpGs",
        "fit_scope": fit_scope,
        "fit_rows": int(len(candidate_rows)),
        "development_train_rows": int(len(train_rows)),
        "development_validation_rows": int(len(val_rows)),
    }
    tmp = checkpoint.with_suffix(".tmp.pt")
    torch.save(state, tmp)
    os.replace(tmp, checkpoint)
    (out / "metrics.json").write_text(json.dumps({k: v for k, v in state.items() if k not in {"model_state", "x_mean", "x_std", "y_mean", "y_std"}}, indent=2) + "\n")
    return {"status": "built", "best_epoch": best_epoch, "validation_mse": best, "elapsed_hours": (time.time() - started) / 3600}


def infer_expanded_features(
    embedding_dir: str | Path,
    probe_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    batch_rows: int = 8192,
) -> dict[str, object]:
    emb_dir = Path(embedding_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.load(emb_dir / "expanded_cpg_idx.npy", mmap_mode="r")
    emb_candidates = list(emb_dir.glob("expanded_embeddings.*.npy"))
    if len(emb_candidates) != 1:
        raise FileNotFoundError(
            f"expected one expanded_embeddings.*.npy under {emb_dir}, found {emb_candidates}"
        )
    embeddings = np.load(emb_candidates[0], mmap_mode="r")
    ids_out = out / "expanded_cpg_idx.npy"
    prior_path = out / "expanded_prior.npy"
    var_path = out / "expanded_variability.npy"
    done = out / "features.done"
    if not ids_out.is_file():
        np.save(ids_out, np.asarray(ids, dtype=np.int64))
    else:
        saved_ids = np.load(ids_out, mmap_mode="r")
        if not np.array_equal(saved_ids, ids):
            raise RuntimeError("expanded-feature cpg_idx do not match the NTv3 embedding store")
    if done.is_file() and prior_path.is_file() and var_path.is_file():
        return {"status": "cached", "rows": int(len(ids))}

    state = torch.load(probe_checkpoint, map_location="cpu", weights_only=False)
    model = ProbeEnsemble(state["seeds"], dim=state["dim"], dropout=state["dropout"])
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    x_mean = torch.from_numpy(np.asarray(state["x_mean"], np.float32)).to(device)
    x_std = torch.from_numpy(np.asarray(state["x_std"], np.float32)).to(device)
    y_mean = np.asarray(state["y_mean"], np.float32)
    y_std = np.asarray(state["y_std"], np.float32)

    progress_path = out / "features.progress"
    if prior_path.is_file() and var_path.is_file() and progress_path.is_file():
        prior = np.load(prior_path, mmap_mode="r+")
        variability = np.load(var_path, mmap_mode="r+")
        if prior.shape != (len(ids),) or variability.shape != (len(ids), 2):
            raise RuntimeError("partial expanded-feature arrays have unexpected shapes")
        resume_from = int(progress_path.read_text().strip() or 0)
    else:
        prior = np.lib.format.open_memmap(prior_path, mode="w+", dtype=np.float32, shape=(len(ids),))
        variability = np.lib.format.open_memmap(var_path, mode="w+", dtype=np.float32, shape=(len(ids), 2))
        resume_from = 0
        progress_path.write_text("0\n")
    started = time.time()
    with torch.inference_mode():
        for start in range(resume_from, len(ids), batch_rows):
            stop = min(start + batch_rows, len(ids))
            x = torch.from_numpy(np.asarray(embeddings[start:stop], dtype=np.float32)).to(device)
            x = (x - x_mean) / x_std
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                pred_z = model(x)
            pred = pred_z.float().cpu().numpy() * y_std + y_mean
            p = 1.0 / (1.0 + np.exp(-np.clip(pred[:, 0], -35, 35)))
            p_out = np.clip(p, 1e-4, 1 - 1e-4).astype(np.float32)
            v_out = pred[:, 1:3].astype(np.float32)
            if not np.isfinite(p_out).all() or not np.isfinite(v_out).all():
                raise FloatingPointError(f"non-finite expanded features at rows {start}:{stop}")
            prior[start:stop] = p_out
            variability[start:stop] = v_out
            prior.flush(); variability.flush(); progress_path.write_text(f"{stop}\n")
            if start == 0 or stop == len(ids) or (start // batch_rows) % 50 == 0:
                elapsed = time.time() - started
                frac = stop / len(ids)
                eta = elapsed * (1 / frac - 1) if frac else float("nan")
                print(f"[feature-infer] {stop:,}/{len(ids):,} ({100*frac:.1f}%) elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    summary = {
        "status": "built", "rows": int(len(ids)), "elapsed_hours": (time.time() - started) / 3600,
        "prior_min": float(np.min(prior)), "prior_max": float(np.max(prior)),
        "between_logvar_min": float(np.min(variability[:, 0])), "between_logvar_max": float(np.max(variability[:, 0])),
        "within_logvar_min": float(np.min(variability[:, 1])), "within_logvar_max": float(np.max(variability[:, 1])),
    }
    del prior, variability
    progress_path.unlink(missing_ok=True)
    (out / "feature_extension_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    done.write_text("ok\n")
    return summary
