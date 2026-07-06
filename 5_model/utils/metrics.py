"""Regression and routing metrics."""

from __future__ import annotations
import numpy as np
from typing import Dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    names: list[str] | None = None) -> Dict[str, float]:
    all_metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
    if names is None:
        return all_metrics
    return {k: v for k, v in all_metrics.items() if k in names}


def routing_stats(gates: np.ndarray, group_names: list[str]) -> Dict[str, float]:
    """Compute routing statistics from gate activations.

    gates: (N, n_groups) binary or continuous gate values.
    """
    stats: Dict[str, float] = {}
    stats["mean_active_groups"] = float(gates.mean(axis=1).sum() / len(gates))

    # Per-group activation rate
    for i, name in enumerate(group_names):
        stats[f"gate_{name}"] = float(gates[:, i].mean())

    stats["sparsity"] = float(1.0 - gates.mean())
    return stats
