"""Figure 5 (PREWRITE_v4 numbering) -- diagnostic plots, LFP (MIT+HUST pooled)
error heatmaps + all-dataset scatter + NCM (TJU) heatmap, 2x2.

2026-10-16: dropped the old MIT+HUST pred-vs-true scatter panel, leaving a
2x2 of (then-)(b)-(e). 2026-10-18: relettered that remaining 2x2 down to
(a)-(d) (no gap at the front any more), and panel (c) (the scatter, former
(d)) now overlays all three datasets -- MIT/HUST/TJU -- in their
_style.DATASET_COLOR colors instead of TJU alone:
  (a) MIT+HUST error heatmap, cycle x capacity (already-rendered PNG, pasted
      via imshow -- same source as before, not regenerated)
  (b) MIT+HUST error heatmap, scenario x capacity (same)
  (c) MIT + HUST + TJU pred-vs-true scatter, natively rendered, one color
      per dataset
  (d) TJU (NCM) error heatmap, scenario x capacity, natively rendered

Source images for (a)/(b):
model_lib/results/p1v2_runs/1009_1342_p1v2_baseline_seed42/figures/
  error_heatmap_capacity_cycle_pct_test_hard.png
  error_heatmap_capacity_scenario_pct_test_hard.png

Panel (c)'s MIT/HUST points come from that same baseline run's
predictions/test_predictions_hard.csv (same source as (a)/(b)'s pasted
PNGs); cell_id naming tells the two apart (MIT: "bXcY", HUST: "N-M" --
same convention fig1_problem_scenario_definition.py uses). (c)/(d) reuse
fig8_single_dataset.py's panel_scatter/panel_error_heatmap logic adapted
here, with the paper-wide _style.DATASET_COLOR palette.

Run: python docs/figures/fig3_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from _style import INK, SUBINK, GRID, SCEN_NAMES, DATASET_COLOR, setup_rcparams, label_panel

setup_rcparams()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "model_lib/results/p1v2_runs/1009_1342_p1v2_baseline_seed42/figures"
LFP_RUN_DIR = PROJECT_ROOT / "model_lib/results/p1v2_runs/1009_1342_p1v2_baseline_seed42"
TJU_RUN_DIR = (PROJECT_ROOT / "legacy_results/experiments/phase1_lab/results/p1v2_runs"
               / "0910_1533_p1v2_p1v4_tju_only_seed42")

PANEL_A_PNG = SRC_DIR / "error_heatmap_capacity_cycle_pct_test_hard.png"
PANEL_B_PNG = SRC_DIR / "error_heatmap_capacity_scenario_pct_test_hard.png"

TJU_CMAP = LinearSegmentedColormap.from_list("err_tju", ["#F5F3EE", DATASET_COLOR["TJU"]])


# ------------------------------------------------------------- panel a/b ---

def _panel_image(ax, png_path: Path):
    img = mpimg.imread(png_path)
    # aspect="auto" (2026-10-22): default imshow preserves the pasted PNG's
    # own native pixel aspect ratio, which doesn't match this 2x2 gridspec
    # cell -- (c)/(d) are drawn natively and fill their cell, so without this
    # (a)/(b) end up letterboxed and visibly smaller than (c)/(d).
    ax.imshow(img, aspect="auto")
    ax.axis("off")


# --------------------------------------------------------------- panel c/d --

def _load_tju_hard() -> pd.DataFrame:
    df = pd.read_csv(TJU_RUN_DIR / "predictions" / "test_predictions_hard.csv")
    df["abs_err_pct"] = (df.cap_pred_Ah - df.cap_true_Ah).abs() / df.cap_true_Ah * 100
    df["dataset"] = "TJU"
    return df


def _load_lfp_hard_by_dataset() -> dict[str, pd.DataFrame]:
    """MIT+HUST pooled predictions (panel a/b's source), split back into
    per-dataset frames by cell_id naming -- MIT: "bXcY", HUST: "N-M" (same
    convention fig1_problem_scenario_definition.py's CELLS dict uses)."""
    df = pd.read_csv(LFP_RUN_DIR / "predictions" / "test_predictions_hard.csv")
    is_mit = df.cell_id.str.match(r"^b\d+c\d+$")
    out = {}
    for name, mask in (("MIT", is_mit), ("HUST", ~is_mit)):
        sub = df.loc[mask].copy()
        sub["dataset"] = name
        out[name] = sub
    return out


def _r2_rmse(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot), rmse


def _panel_scatter_multi(ax, dfs_by_dataset: dict, n_per_dataset: int = 1500):
    """Pred-vs-true scatter overlaying all three datasets, one color each
    (equal per-dataset subsample so HUST's larger row count doesn't visually
    swamp TJU -- raw counts are MIT 215k / HUST 446k / TJU 45k)."""
    rng = np.random.default_rng(20261018)
    all_true = np.concatenate([d.soh_true.values for d in dfs_by_dataset.values()])
    all_pred = np.concatenate([d.soh_pred.values for d in dfs_by_dataset.values()])
    lo, hi = all_true.min(), all_true.max()
    pad = 0.02 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=SUBINK, lw=0.9,
            ls=(0, (5, 3)), zorder=1)

    for name, df in dfs_by_dataset.items():
        idx = rng.choice(len(df), size=min(n_per_dataset, len(df)), replace=False)
        ss = df.iloc[idx]
        ax.scatter(ss.soh_true, ss.soh_pred, s=3.5, color=DATASET_COLOR[name], alpha=0.25,
                   linewidths=0, zorder=2, rasterized=True, label=name)

    r2, rmse = _r2_rmse(all_true, all_pred)
    ax.text(0.04, 0.96, f"$R^2$={r2:.3f}\nRMSE={rmse:.4f}", transform=ax.transAxes,
            fontsize=6.8, color=INK, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, lw=0.8))
    leg = ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none",
                     framealpha=0.85, fontsize=6.8, handletextpad=0.4, borderaxespad=0.3,
                     markerscale=2.2)
    for t in leg.get_texts():
        t.set_color(SUBINK)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("True SOH")
    ax.set_ylabel("Predicted SOH")
    ax.set_title("MIT + HUST + TJU -- test accuracy (hard)", loc="left", pad=6, fontsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


def _smoothed_grid(y_idx, x, err, n_x=40, sigma=(0.9, 2.2)):
    x_edges = np.linspace(x.min(), x.max(), n_x + 1)
    y_bin = np.clip(y_idx, 0, 5)
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, n_x - 1)
    sum_grid = np.zeros((6, n_x))
    cnt_grid = np.zeros((6, n_x))
    np.add.at(sum_grid, (y_bin, x_bin), err)
    np.add.at(cnt_grid, (y_bin, x_bin), 1)
    sum_s = gaussian_filter(sum_grid, sigma=sigma, mode="nearest")
    cnt_s = gaussian_filter(cnt_grid, sigma=sigma, mode="nearest")
    grid = np.divide(sum_s, cnt_s, out=np.full_like(sum_s, np.nan), where=cnt_s > 1e-6)
    return grid, x_edges


def _panel_error_heatmap(ax, fig, df):
    scen_to_idx = {s: i for i, s in enumerate(SCEN_NAMES)}
    y_idx = df.seg_name.map(scen_to_idx).values
    grid, x_edges = _smoothed_grid(y_idx, df.cap_true_Ah.values, df.abs_err_pct.values)
    vmax = np.nanpercentile(grid, 97)
    im = ax.imshow(grid, aspect="auto", cmap=TJU_CMAP, vmin=0, vmax=vmax,
                    extent=[x_edges[0], x_edges[-1], 5.5, -0.5], interpolation="bilinear")
    ax.set_yticks(range(6))
    ax.set_yticklabels(SCEN_NAMES, fontsize=6.6)
    ax.set_xlabel("Observed capacity (Ah)")
    ax.set_title("TJU (NCM) -- mean |error| (%), scenario x capacity", loc="left", pad=6, fontsize=8.0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.03)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    cbar.outline.set_visible(False)


# ------------------------------------------------------------------ main ---

def build_figure():
    df_tju = _load_tju_hard()
    lfp = _load_lfp_hard_by_dataset()
    dfs_all = {"MIT": lfp["MIT"], "HUST": lfp["HUST"], "TJU": df_tju}

    fig = plt.figure(figsize=(9.6, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                           top=0.95, bottom=0.07, left=0.075, right=0.97)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _panel_image(ax_a, PANEL_A_PNG)
    _panel_image(ax_b, PANEL_B_PNG)
    _panel_scatter_multi(ax_c, dfs_all)
    _panel_error_heatmap(ax_d, fig, df_tju)

    label_panel(fig, ax_a, "(a)", dx=-0.035, dy=0.012)
    label_panel(fig, ax_b, "(b)", dx=-0.035, dy=0.012)
    label_panel(fig, ax_c, "(c)", dx=-0.06, dy=0.012)
    label_panel(fig, ax_d, "(d)", dx=-0.06, dy=0.012)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = Path(__file__).resolve().parent / "fig3_diagnostics.png"
    out_pdf = Path(__file__).resolve().parent / "fig3_diagnostics.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
    print("panels: (a)/(b) <- MIT+HUST error heatmaps (pasted), "
          "(c) <- MIT+HUST+TJU scatter (native), (d) <- TJU error heatmap (native)")
