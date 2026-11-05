"""Figure 5 (2026-10-30 layout) -- per-dataset error heatmaps, 3 rows x 2 cols.

One row per single-dataset experiment (MIT / HUST / TJU), each row the same
pair the old fig3_diagnostics (a)/(b) showed for pooled LFP:
  left  : mean |error| (%) interpolated over cycle x observed capacity
  right : mean |error| (%) interpolated over scenario x observed capacity

Rendering follows 9_eval/test.py::_plot_error_heatmaps (sum/count grids
smoothed separately with gaussian_filter, jet colormap, vmax = 99th pct of
raw error) so the panels match the run-folder PNGs, but drawn natively here.

2026-11-01: rows now come from the single-dataset runs (scen Transformer,
union, no warm-start, lambda_L0 2e-4, n2 20%). One colour scale is shared by
all six panels (vmax = 99th pct of the pooled raw error of the three runs) so
the three datasets can be compared directly; one colourbar on the right.

Run: python docs/figures/fig5_per_dataset_heatmaps.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from _style import SCEN_NAMES, setup_rcparams, label_panel

setup_rcparams()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "model_lib/results/p1v2_runs"

PLACEHOLDER = False
# dataset -> (run dir, cell filter applied to that run's test predictions)
RUNS = {
    "MIT":  ("1031_0117_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_mit_seed42", None),
    "HUST": ("1031_0431_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_hust_seed42", None),
    "TJU":  ("1031_0938_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_tju_seed42", None),
}
CHEM = {"MIT": "LFP", "HUST": "LFP", "TJU": "NCM"}


def _load(run: str, cell_filter: str | None) -> pd.DataFrame:
    df = pd.read_csv(RUN_ROOT / run / "predictions" / "test_predictions_hard.csv")
    if cell_filter is not None:
        is_mit = df.cell_id.str.match(r"^b\d+c\d+$")
        df = df.loc[is_mit if cell_filter == "MIT" else ~is_mit]
    df = df.copy()
    df["err_pct"] = (df.cap_pred_Ah - df.cap_true_Ah).abs() / df.cap_true_Ah * 100
    return df


def _smoothed_grid(x, y, err, x_edges, y_edges, sigma):
    n_y, n_x = len(y_edges) - 1, len(x_edges) - 1
    xb = np.clip(np.digitize(x, x_edges) - 1, 0, n_x - 1)
    yb = np.clip(np.digitize(y, y_edges) - 1, 0, n_y - 1)
    s = np.zeros((n_y, n_x))
    c = np.zeros((n_y, n_x))
    np.add.at(s, (yb, xb), err)
    np.add.at(c, (yb, xb), 1)
    s = gaussian_filter(s, sigma=sigma, mode="nearest")
    c = gaussian_filter(c, sigma=sigma, mode="nearest")
    return np.divide(s, c, out=np.full_like(s, np.nan), where=c > 1e-6)


def _draw(ax, grid, x_edges, y_edges, vmax):
    cmap = matplotlib.colormaps["jet"].copy()
    cmap.set_bad("white")
    return ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap=cmap,
                     extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                     vmin=0, vmax=vmax, interpolation="bilinear")


def _row(ax_cyc, ax_scen, df, name, vmax):
    cap = df.cap_true_Ah.values
    err = df.err_pct.values
    y_edges = np.linspace(cap.min(), cap.max(), 181)

    cyc = df.cycle.values.astype(float)
    x_edges = np.linspace(cyc.min(), cyc.max(), 251)
    _draw(ax_cyc, _smoothed_grid(cyc, cap, err, x_edges, y_edges, sigma=3), x_edges, y_edges, vmax)
    ax_cyc.set_xlabel("Cycle")
    ax_cyc.set_ylabel("Observed capacity (Ah)")

    scen_idx = df.seg_name.map({s: i for i, s in enumerate(SCEN_NAMES)}).values
    rng = np.random.default_rng(0)
    xj = scen_idx + rng.uniform(-0.45, 0.45, size=scen_idx.shape)
    x_edges = np.linspace(-0.5, len(SCEN_NAMES) - 0.5, len(SCEN_NAMES) * 40 + 1)
    im = _draw(ax_scen, _smoothed_grid(xj, cap, err, x_edges, y_edges, sigma=(3, 1.5)),
               x_edges, y_edges, vmax)
    ax_scen.set_xticks(range(len(SCEN_NAMES)))
    ax_scen.set_xticklabels(SCEN_NAMES, rotation=30, ha="right", fontsize=7)

    tag = f"{name} ({CHEM[name]})" + ("  [placeholder]" if PLACEHOLDER else "")
    for ax, what in ((ax_cyc, "cycle x capacity"), (ax_scen, "scenario x capacity")):
        ax.set_title(f"{tag} -- {what}", loc="left", pad=5, fontsize=8.0)
        for s in ax.spines.values():
            s.set_visible(False)
    return im


def build_figure():
    fig, axes = plt.subplots(3, 2, figsize=(9.6, 10.2),
                             gridspec_kw=dict(hspace=0.42, wspace=0.18,
                                              top=0.96, bottom=0.06, left=0.08, right=0.90))
    dfs = {name: _load(run, filt) for name, (run, filt) in RUNS.items()}
    vmax = float(np.percentile(np.concatenate([d.err_pct.values for d in dfs.values()]), 99))
    print(f"shared vmax = {vmax:.2f} %")
    letters = iter("abcdef")
    for r, (name, df) in enumerate(dfs.items()):
        im = _row(axes[r, 0], axes[r, 1], df, name, vmax)
        for c in range(2):
            label_panel(fig, axes[r, c], f"({next(letters)})", dx=-0.045, dy=0.010)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.015)
    cbar.set_label("Mean |error| (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_visible(False)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out = Path(__file__).resolve().parent / "fig5_per_dataset_heatmaps"
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"saved: {out}.png / .pdf")
