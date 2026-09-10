"""Figure 6 -- Final SOH regression performance (G20 + G21 + G22), "main".

(a) true-vs-predicted SOH scatter, all six scenarios overlaid
(b) representative cell capacity-fade curve, true vs predicted (oracle)
(c) error heatmap: RMSE by SOH level bin x scenario

Real data: predictions/test_predictions.csv from a completed p1v4 run.

Run: python docs/figures/fig6_final_regression.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from _style import (INK, SUBINK, FAINT, GRID, SCEN_NAMES, SCEN_POS_COLOR, setup_rcparams,
                     label_panel, PROJECT_ROOT)

setup_rcparams()

RUN_DIR = PROJECT_ROOT / "5_model/experiments/phase1_lab/results/p1v2_runs/0827_1705_p1v2_p1v4_full_seed42"
PRED_CSV = RUN_DIR / "predictions" / "test_predictions.csv"
EXAMPLE_CELL = "b1c0"

SEQ_CMAP = LinearSegmentedColormap.from_list("err", ["#F4F1E8", "#B14A3C"])
RNG = np.random.default_rng(20260910)


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PRED_CSV)
    df["abs_err"] = (df.cap_pred_Ah - df.cap_true_Ah).abs()
    return df


def r2_rmse(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return r2, rmse


def panel_scatter(ax, df: pd.DataFrame, n_per_scen: int = 2500):
    lo, hi = df.soh_true.min(), df.soh_true.max()
    pad = 0.02 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=SUBINK, lw=0.9,
            ls=(0, (5, 3)), zorder=1)

    for name, color in zip(SCEN_NAMES, SCEN_POS_COLOR):
        sub = df[df.seg_name == name]
        idx = RNG.choice(len(sub), size=min(n_per_scen, len(sub)), replace=False)
        s = sub.iloc[idx]
        ax.scatter(s.soh_true, s.soh_pred, s=3.5, color=color, alpha=0.16,
                   linewidths=0, label=name, zorder=2, rasterized=True)

    r2, rmse = r2_rmse(df.soh_true.values, df.soh_pred.values)
    ax.text(0.04, 0.96, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}", transform=ax.transAxes,
            fontsize=7.4, color=INK, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=GRID, lw=0.8))

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("True SOH")
    ax.set_ylabel("Predicted SOH")
    ax.set_title("Test-set accuracy, all scenarios (oracle)", loc="left", pad=6, fontsize=8.0)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=6.2, handlelength=0.9,
                     markerscale=2.2, labelspacing=0.25, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(SUBINK)


def panel_capacity_curve(ax, df: pd.DataFrame, cell_id: str):
    c = df[df.cell_id == cell_id]
    per_cycle_true = c.groupby("cycle").cap_true_Ah.first()
    per_cycle_pred = c.groupby("cycle").cap_pred_Ah.mean()
    per_cycle_std = c.groupby("cycle").cap_pred_Ah.std()

    ax.plot(per_cycle_true.index, per_cycle_true.values, color=INK, lw=1.6,
            label="True", zorder=4)
    ax.plot(per_cycle_pred.index, per_cycle_pred.values, color="#B14A3C", lw=1.3,
            ls=(0, (4.5, 2.2)), label="Predicted (mean over 6 scenarios)", zorder=5)
    ax.fill_between(per_cycle_pred.index, per_cycle_pred.values - per_cycle_std.values,
                     per_cycle_pred.values + per_cycle_std.values, color="#B14A3C", alpha=0.14,
                     lw=0, zorder=3)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title(f"Capacity fade, cell {cell_id} (LFP, oracle)", loc="left", pad=6, fontsize=8.0)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=6.6, handlelength=1.8,
                     borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(SUBINK)


def panel_error_heatmap(ax, df: pd.DataFrame, n_bins: int = 5):
    bins = np.linspace(df.soh_true.min(), df.soh_true.max(), n_bins + 1)
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(n_bins)]
    df = df.copy()
    df["soh_bin"] = pd.cut(df.soh_true, bins=bins, labels=labels, include_lowest=True)
    piv = df.pivot_table(index="soh_bin", columns="seg_name", values="abs_err",
                          aggfunc=lambda x: np.sqrt(np.mean(x ** 2)))
    piv = piv[SCEN_NAMES].loc[labels[::-1]]
    mat = piv.values

    im = ax.imshow(mat, aspect="auto", cmap=SEQ_CMAP)
    ax.set_xticks(range(6))
    ax.set_xticklabels(SCEN_NAMES, rotation=40, ha="right", fontsize=6.8)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(labels[::-1], fontsize=6.8)
    ax.set_ylabel("True SOH bin")
    vmax = np.nanmax(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v*1000:.1f}", ha="center", va="center", fontsize=6.0,
                    color="white" if v > 0.55 * vmax else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("RMSE by SOH level x scenario (mAh)", loc="left", pad=6, fontsize=8.0)
    return im


def build_figure():
    df = load_predictions()

    fig = plt.figure(figsize=(10.2, 3.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.42,
                           top=0.86, bottom=0.185, left=0.065, right=0.975)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_scatter(ax_a, df)
    panel_capacity_curve(ax_b, df, EXAMPLE_CELL)
    panel_error_heatmap(ax_c, df)

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=2.8)

    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")):
        label_panel(fig, ax, letter, dx=-0.045, dy=0.012)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig6_final_regression.png"
    out_pdf = "docs/figures/fig6_final_regression.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
