"""Figure 4 -- Classification / routing performance (E12 + E13).

(a) routing-classifier confusion matrix (lo/mid/hi, row-normalized recall)
(b) per-scenario classification accuracy
(c) capacity RMSE: oracle -> hard -> soft routing
(d) capacity R^2: oracle -> hard -> soft routing

Real data: metrics/metrics.json from a completed p1v4 run.

Run: python docs/figures/fig4_routing_performance.py
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from _style import (INK, SUBINK, FAINT, GRID, SCEN_NAMES, SCEN_POS_COLOR,
                     ORACLE_COLOR, HARD_COLOR, SOFT_COLOR, setup_rcparams,
                     label_panel, PROJECT_ROOT)

setup_rcparams()

RUN_DIR = PROJECT_ROOT / "5_model/experiments/phase1_lab/results/p1v2_runs/0827_1705_p1v2_p1v4_full_seed42"
METRICS_JSON = RUN_DIR / "metrics" / "metrics.json"
CLASS_NAMES = ["lo", "mid", "hi"]
SEQ_CMAP = LinearSegmentedColormap.from_list("cls", ["#F4F1E8", "#274C63"])


def load_metrics():
    return json.load(open(METRICS_JSON, encoding="utf-8"))["test"]


def panel_confusion(ax, cm_counts):
    cm = np.array(cm_counts, dtype=float)
    row_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(row_norm, cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(3):
        for j in range(3):
            v = row_norm[i, j]
            ax.text(j, i - 0.13, f"{v:.3f}", ha="center", va="center", fontsize=8.2,
                    color="white" if v > 0.55 else INK, fontweight="bold")
            ax.text(j, i + 0.20, f"n={int(cm[i, j]):,}", ha="center", va="center", fontsize=5.8,
                    color="white" if v > 0.55 else SUBINK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Routing-classifier confusion matrix (recall)", loc="left", pad=6, fontsize=8.0)
    return im


def panel_scenario_accuracy(ax, per_scenario, overall_acc):
    vals = [per_scenario[s] for s in SCEN_NAMES]
    bars = ax.bar(range(6), vals, color=SCEN_POS_COLOR, width=0.62, zorder=3)
    ax.axhline(overall_acc, color=SUBINK, lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax.text(5.55, overall_acc, f"overall {overall_acc:.3f}", fontsize=6.6, color=SUBINK,
            va="bottom", ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom",
                fontsize=6.6, color=INK)
    ax.set_xticks(range(6))
    ax.set_xticklabels(SCEN_NAMES, rotation=40, ha="right", fontsize=6.8)
    ax.set_ylim(0.90, 1.006)
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Per-scenario routing accuracy", loc="left", pad=6, fontsize=8.0)


def panel_degradation(ax, values, ylabel, title, higher_is_better, fmt="{:.4f}"):
    modes = ["oracle", "hard", "soft"]
    colors = [ORACLE_COLOR, HARD_COLOR, SOFT_COLOR]
    base = values[0]
    span = max(values) - min(values)
    pad = max(span * 4.6, max(values) * 0.03)
    lo = min(values) - pad
    hi = max(values) + pad
    ax.set_ylim(max(0, lo) if higher_is_better else lo, hi)
    offset = (hi - lo) * 0.045

    bars = ax.bar(range(3), values, color=colors, width=0.56, zorder=3)
    for i, (b, v) in enumerate(zip(bars, values)):
        if i == 0:
            label = fmt.format(v)
        else:
            pct = (v - base) / base * 100
            sign = "+" if pct > 0 else ""
            label = f"{fmt.format(v)}\n({sign}{pct:.2f}%)"
        ax.text(b.get_x() + b.get_width() / 2, v + offset, label, ha="center", va="bottom",
                fontsize=7.0, color=INK, fontweight="bold", linespacing=1.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(modes)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=6, fontsize=8.0)


def build_figure():
    m = load_metrics()
    hard_cls = m["hard"]["classification"]

    fig = plt.figure(figsize=(7.6, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.42,
                           top=0.955, bottom=0.09, left=0.11, right=0.965)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_confusion(ax_a, hard_cls["confusion_matrix"])
    panel_scenario_accuracy(ax_b, hard_cls["per_scenario_accuracy"], hard_cls["accuracy"])

    rmse_vals = [m[mode]["capacity"]["rmse"] for mode in ("oracle", "hard", "soft")]
    r2_vals = [m[mode]["capacity"]["r2"] for mode in ("oracle", "hard", "soft")]
    panel_degradation(ax_c, rmse_vals, "Capacity RMSE", "Regression cost of routing errors (RMSE)",
                      higher_is_better=False)
    panel_degradation(ax_d, r2_vals, "Capacity $R^2$", "Regression cost of routing errors ($R^2$)",
                      higher_is_better=True, fmt="{:.4f}")

    for ax in (ax_b, ax_c, ax_d):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=2.8)

    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")):
        label_panel(fig, ax, letter, dx=-0.075 if ax in (ax_a, ax_c) else -0.055, dy=0.012)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig4_routing_performance.png"
    out_pdf = "docs/figures/fig4_routing_performance.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
