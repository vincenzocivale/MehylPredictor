from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean((y_true[valid] - y_pred[valid]) ** 2)) if valid.any() else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid]))) if valid.any() else float("nan")
