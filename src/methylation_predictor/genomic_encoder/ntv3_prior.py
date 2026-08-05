#!/usr/bin/env python3
"""Train frozen-NTv3 Ridge/MLP probes without looking at held-out CpG targets."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:  # allow direct module-file execution in development
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methylation_predictor.genomic_encoder.ntv3_prior_common import sha256
from methylation_predictor.genomic_encoder.static_prior import _inverse_target, _metrics, _partition, _target


def _load(path: Path, representation: str, rc_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as forward:
        cpg = forward["cpg_idx"].astype(np.int64)
        def vectors(value):
            if representation == "centre": return value["centre"]
            if representation == "pool_32": return value["pool_32"]
            if representation == "pool_128": return value["pool_128"]
            if representation == "transformer": return value["transformer"]
            if representation == "multiscale": return np.concatenate([value[x] for x in ("centre", "pool_32", "pool_128", "pool_512", "transformer")], axis=1)
            raise ValueError("representation must be centre, pool_32, pool_128, transformer, or multiscale")
        z = vectors(forward)
    if rc_path:
        with np.load(rc_path) as reverse:
            if not np.array_equal(cpg, reverse["cpg_idx"]): raise ValueError("forward/reverse cpg_idx differ")
            z = (z + vectors(reverse)) / 2
    # Multi-scale has 5x the raw width. A fixed, seeded Gaussian projection
    # keeps the learned probe architecture/parameter count identical to the
    # centre and transformer comparisons; it is not fitted to any target.
    if representation == "multiscale":
        out_dim = z.shape[1] // 5
        rng = np.random.default_rng(20260728)
        projection = rng.normal(0, 1 / np.sqrt(z.shape[1]), (z.shape[1], out_dim)).astype(np.float32)
        z = z @ projection
    return cpg, z.astype(np.float32)


class MLP(torch.nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__(); self.net = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(256, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def _fit_mlp(x_train, y_train, x_other, seed, epochs, lr, weight_decay, dropout, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(x_train.shape[1], dropout).to(device); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x = torch.tensor(x_train, device=device); y = torch.tensor(y_train.astype(np.float32), device=device)
    for _ in range(epochs):
        model.train(); opt.zero_grad(); torch.nn.functional.mse_loss(model(x), y).backward(); opt.step()
    model.eval()
    with torch.inference_mode(): return model(torch.tensor(x_other, device=device)).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--embeddings", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--reverse-embeddings"); p.add_argument("--representation", choices=("centre", "pool_32", "pool_128", "transformer", "multiscale"), required=True)
    p.add_argument("--split", choices=("random", "block", "fixed"), required=True); p.add_argument("--block-size", type=int, default=5_000_000); p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--seeds", default="17,29,43"); p.add_argument("--epochs", type=int, default=500); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--validation-only", dest="include_test", action="store_false", default=True,
                   help="do not write held-out test predictions (required during configuration screening)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = p.parse_args()
    source, embed_path = Path(a.input), Path(a.embeddings); frame = pd.read_parquet(source)
    if frame.cpg_idx.duplicated().any() or frame.mean_train.isna().any() or not frame.mean_train.between(0, 1).all(): raise ValueError("invalid one-row-per-CpG target table")
    cpg, z = _load(embed_path, a.representation, Path(a.reverse_embeddings) if a.reverse_embeddings else None)
    joined = frame.merge(pd.DataFrame({"cpg_idx": cpg, "_row": np.arange(len(cpg))}), on="cpg_idx", how="inner", validate="one_to_one")
    if len(joined) != len(frame): raise ValueError("embeddings do not cover every input CpG")
    joined["split"] = _partition(joined, a.split, "cpg_idx", "chromosome", "position", a.block_size, a.seed)
    train, validation, test = (joined[joined.split == x].copy() for x in ("train", "validation", "test"))
    scaler = StandardScaler().fit(z[train._row]); x_train, y_train = scaler.transform(z[train._row]), _target(train.mean_train.to_numpy(float), "logit")
    predictions: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    ridge = Ridge(alpha=10.).fit(x_train, y_train)
    ridge_test = _inverse_target(ridge.predict(scaler.transform(z[test._row])), "logit") if a.include_test else None
    predictions["ridge"] = (_inverse_target(ridge.predict(scaler.transform(z[validation._row])), "logit"), ridge_test)
    for seed in (int(x) for x in a.seeds.split(",")):
        # One training run per seed; predicting validation and test from the
        # same frozen probe prevents needless retraining and preserves the
        # validation-only selection protocol.
        stacked_other = scaler.transform(z[validation._row]) if not a.include_test else np.concatenate([scaler.transform(z[validation._row]), scaler.transform(z[test._row])])
        pred = _fit_mlp(x_train, y_train, stacked_other, seed, a.epochs, a.lr, a.weight_decay, a.dropout, a.device)
        val, tst = pred[:len(validation)], pred[len(validation):] if a.include_test else None
        predictions[f"mlp_seed_{seed}"] = (_inverse_target(val, "logit"), _inverse_target(tst, "logit") if tst is not None else None)
    # The ensemble is computed from models trained independently on exactly the train partition.
    mlp_names = [x for x in predictions if x.startswith("mlp_seed_")]
    predictions["mlp_ensemble"] = (np.mean([predictions[x][0] for x in mlp_names], axis=0), np.mean([predictions[x][1] for x in mlp_names], axis=0) if a.include_test else None)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv), "source": {"path": str(source), "sha256": sha256(source)},
              "embeddings": {"path": str(embed_path), "sha256": sha256(embed_path), "reverse_path": a.reverse_embeddings, "representation": a.representation, "dimension": int(z.shape[1])},
              "design": {"target": "logit(clip(mean_train,1e-6,1-1e-6))", "metrics_space": "beta", "split": a.split, "block_size": a.block_size if a.split == "block" else None, "seed": a.seed, "mlp": "LayerNorm-d:256:64:1", "mlp_seeds": [int(x) for x in a.seeds.split(",")], "held_out_test_written": a.include_test},
              "rows": {x: int(len(y)) for x, y in (("train", train), ("validation", validation), ("test", test))},
              "validation": {name: _metrics(validation.mean_train.to_numpy(float), pred[0], validation.reset_index(drop=True), "cgi_category") for name, pred in predictions.items()}}
    if a.include_test:
        report["test"] = {name: _metrics(test.mean_train.to_numpy(float), pred[1], test.reset_index(drop=True), "cgi_category") for name, pred in predictions.items()}
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.concat(([validation[["cpg_idx", "mean_train", "split"]], test[["cpg_idx", "mean_train", "split"]]] if a.include_test else [validation[["cpg_idx", "mean_train", "split"]]]), ignore_index=True)
    for name, (val, tst) in predictions.items():
        prediction_frame[f"pred_{name}"] = np.concatenate([val, tst]) if a.include_test else val
    prediction_frame.to_parquet(out / "locus_predictions.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__": main()
