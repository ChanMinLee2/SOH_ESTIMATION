"""Figure 8 -- Single-dataset performance (MIT-only / HUST-only / TJU-only).

Each of MIT, HUST, TJU trained and evaluated independently (own split, own
gates) -- the cross-chemistry generalization check (hard routing, the
realistic/deployment condition, not oracle).

Layout: 2 rows x 3 columns.
  row (a): true-vs-predicted SOH scatter, hard routing
  row (b): mean |error| (%) heatmap, scenario x observed capacity, hard routing
           (same smoothed-grid technique as
           test.py::_smoothed_error_grid, re-rendered in
           this paper's palette instead of the diagnostic script's jet map)

Real data: predictions/test_predictions_hard.csv from each single-dataset
p1v4 run (2026-09-13: test.py extended to also export
hard/soft mode predictions, not just oracle -- see that file's
_export_for_visualize).

Run: python docs/figures/fig5_single_dataset.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from _style import INK, SUBINK, GRID, SCEN_NAMES, setup_rcparams, label_panel, PROJECT_ROOT

setup_rcparams()

RUNS_DIR = PROJECT_ROOT / "legacy_results/experiments/phase1_lab/results/p1v2_runs"
DATASETS = [
    ("MIT", RUNS_DIR / "0908_0100_p1v2_p1v4_mit_only_seed42"),
    ("HUST", RUNS_DIR / "0908_0358_p1v2_p1v4_hust_only_seed42"),
    ("TJU", RUNS_DIR / "0910_1533_p1v2_p1v4_tju_only_seed42"),
]

# Fig5-only dataset palette (per user request, 2026-09-13) -- deliberately
# distinct hues (orange/blue/green) rather than the paper-wide DATASET_COLOR
# (navy/slate/rust, _style.py) used in Fig1, so this is scoped to this figure.
FIG5_DATASET_COLOR = {"MIT": "#D08A2E", "HUST": "#2F6690", "TJU": "#4C7A3D"}
SEQ_CMAP_BY_DATASET = {
    name: LinearSegmentedColormap.from_list(f"err_{name}", ["#F5F3EE", color])
    for name, color in FIG5_DATASET_COLOR.items()
}


def load_hard(run_dir):
    df = pd.read_csv(run_dir / "predictions" / "test_predictions_hard.csv")
    df["abs_err_pct"] = (df.cap_pred_Ah - df.cap_true_Ah).abs() / df.cap_true_Ah * 100
    return df


def r2_rmse(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot), rmse


def panel_scatter(ax, df, name):
    rng = np.random.default_rng(20260913)
    lo, hi = df.soh_true.min(), df.soh_true.max()
    pad = 0.02 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=SUBINK, lw=0.9,
            ls=(0, (5, 3)), zorder=1)
    idx = rng.choice(len(df), size=min(4500, len(df)), replace=False)
    ss = df.iloc[idx]
    ax.scatter(ss.soh_true, ss.soh_pred, s=3.5, color=FIG5_DATASET_COLOR[name], alpha=0.18,
               linewidths=0, zorder=2, rasterized=True)

    r2, rmse = r2_rmse(df.soh_true.values, df.soh_pred.values)
    ax.text(0.04, 0.96, f"$R^2$={r2:.3f}\nRMSE={rmse:.4f}", transform=ax.transAxes,
            fontsize=6.8, color=INK, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, lw=0.8))
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("True SOH")
    ax.set_ylabel("Predicted SOH")
    ax.set_title(f"{name} -- test accuracy (hard)", loc="left", pad=6, fontsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


def _smoothed_grid(y_idx, x, err, n_x=40, sigma=(0.9, 2.2)):
    x_edges = np.linspace(x.min(), x.max(), n_x + 1)
    y_edges = np.arange(7) - 0.5
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, n_x - 1)
    y_bin = np.clip(y_idx, 0, 5)
    sum_grid = np.zeros((6, n_x))
    cnt_grid = np.zeros((6, n_x))
    np.add.at(sum_grid, (y_bin, x_bin), err)
    np.add.at(cnt_grid, (y_bin, x_bin), 1)
    sum_s = gaussian_filter(sum_grid, sigma=sigma, mode="nearest")
    cnt_s = gaussian_filter(cnt_grid, sigma=sigma, mode="nearest")
    grid = np.divide(sum_s, cnt_s, out=np.full_like(sum_s, np.nan), where=cnt_s > 1e-6)
    return grid, x_edges


def _error_grid(df):
    scen_to_idx = {s: i for i, s in enumerate(SCEN_NAMES)}
    y_idx = df.seg_name.map(scen_to_idx).values
    x = df.cap_true_Ah.values
    err = df.abs_err_pct.values
    return _smoothed_grid(y_idx, x, err)


def panel_error_heatmap(ax, name, grid, x_edges, vmax):
    im = ax.imshow(grid, aspect="auto", cmap=SEQ_CMAP_BY_DATASET[name], vmin=0, vmax=vmax,
                    extent=[x_edges[0], x_edges[-1], 5.5, -0.5], interpolation="bilinear")
    ax.set_yticks(range(6))
    ax.set_yticklabels(SCEN_NAMES, fontsize=6.6)
    ax.set_xlabel("Observed capacity (Ah)")
    ax.set_title(f"{name} -- mean |error| (%), hard", loc="left", pad=6, fontsize=8.0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def build_figure():
    fig = plt.figure(figsize=(11.4, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42,
                           top=0.92, bottom=0.09, left=0.055, right=0.965)

    dfs = [(name, load_hard(run_dir)) for name, run_dir in DATASETS]
    grids = [_error_grid(df) for _, df in dfs]
    # 2026-09-13 (user request): unify the error-% scale across all three
    # datasets instead of a per-panel vmax -- makes MIT/HUST/TJU's error
    # magnitudes directly comparable at a glance (previously MIT~5%/HUST~4.2%/
    # TJU~1.5% each had their own scale, which was truer to each panel's own
    # dynamic range but hid how much noisier MIT/HUST are than TJU overall).
    shared_vmax = max(np.nanpercentile(grid, 97) for grid, _ in grids)

    for j, ((name, _), (grid, x_edges)) in enumerate(zip(dfs, grids)):
        df = dfs[j][1]
        ax_top = fig.add_subplot(gs[0, j])
        ax_bot = fig.add_subplot(gs[1, j])
        panel_scatter(ax_top, df, name)
        im = panel_error_heatmap(ax_bot, name, grid, x_edges, shared_vmax)
        cbar = fig.colorbar(im, ax=ax_bot, fraction=0.055, pad=0.03)
        cbar.ax.tick_params(labelsize=5.8, length=2)
        cbar.outline.set_visible(False)
        if j == 0:
            label_panel(fig, ax_top, "(a)", dx=-0.055, dy=0.014)
            label_panel(fig, ax_bot, "(b)", dx=-0.055, dy=0.014)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig8_single_dataset.png"
    out_pdf = "docs/figures/fig8_single_dataset.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
