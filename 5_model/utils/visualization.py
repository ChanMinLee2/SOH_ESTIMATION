"""Visualization utilities for training and evaluation."""

from __future__ import annotations
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_rmse: List[float],
    val_rmse: List[float],
    save_path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    ax = axes[0]
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, train_rmse, label="train")
    ax.plot(epochs, val_rmse, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (Ah)")
    ax.set_title("Capacity RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_routing_heatmap(
    gate_means: np.ndarray,
    group_names: List[str],
    save_path: pathlib.Path,
    title: str = "Routing Gate Activation (mean over test set)",
) -> None:
    """
    gate_means: (n_groups,) mean gate values.
    group_names: list of 'seg_cat' strings, length n_groups.
    """
    # Parse into segment × category grid
    segs = ["dis_hi", "dis_mid", "dis_lo", "chg_lo", "chg_mid", "chg_hi"]
    cats = ["stat", "diff", "lfp", "morph"]

    grid = np.full((len(segs), len(cats)), np.nan)
    for i, name in enumerate(group_names):
        for r, seg in enumerate(segs):
            for c, cat in enumerate(cats):
                if name == f"{seg}_{cat}":
                    grid[r, c] = gate_means[i]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean gate value")

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_yticks(range(len(segs)))
    ax.set_yticklabels(segs)
    ax.set_title(title)

    for r in range(len(segs)):
        for c in range(len(cats)):
            v = grid[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if v < 0.6 else "white")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    save_path: pathlib.Path,
    dataset_labels: Optional[np.ndarray] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    if dataset_labels is not None:
        unique_ds = np.unique(dataset_labels)
        colors = ["steelblue", "orangered", "seagreen", "purple"]
        for i, ds in enumerate(unique_ds):
            mask = dataset_labels == ds
            ax.scatter(y_true[mask], y_pred[mask], s=8, alpha=0.4,
                       color=colors[i % len(colors)], label=ds)
        ax.legend(markerscale=3)
    else:
        ax.scatter(y_true, y_pred, s=8, alpha=0.3, color="steelblue")

    vmin = min(y_true.min(), y_pred.min()) * 0.98
    vmax = max(y_true.max(), y_pred.max()) * 1.02
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1, alpha=0.6)

    label = "\n".join([
        f"RMSE={metrics.get('rmse', 0):.4f} Ah",
        f"MAE ={metrics.get('mae', 0):.4f} Ah",
        f"R²  ={metrics.get('r2', 0):.4f}",
        f"MAPE={metrics.get('mape', 0):.2f}%",
    ])
    ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7))

    ax.set_xlabel("True capacity (Ah)")
    ax.set_ylabel("Predicted capacity (Ah)")
    ax.set_title("Capacity Estimation")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_schedule(
    temperatures: List[float],
    save_path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(temperatures) + 1), temperatures)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gumbel Temperature")
    ax.set_title("Router Temperature Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
