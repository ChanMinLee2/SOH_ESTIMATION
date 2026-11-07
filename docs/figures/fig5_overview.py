"""Figure 5 (2026-11-04) -- one architecture, both chemistries, any segment position.

Row (a): parity density of true vs estimated SOH for the three single-dataset
runs (hexbin, log counts), with RMSE / R^2 of the hard-routed estimates.
Row (b): one representative test cell per dataset. Measured SOH (black) and
every segment estimate as a dot coloured by scenario, with the residual in
percentage points underneath.

Scenario colours: charge = blue family, discharge = rust family, darker =
lower SOC zone. Dots are coloured by the TRUE scenario label (seg_name);
the routed label is not stored per segment, and routing accuracy is
>= 99.25% for every dataset, so the two colourings are visually identical.

Run: python docs/figures/fig5_overview.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from _style import INK, SUBINK, FAINT, GRID, setup_rcparams, strip_top_right

setup_rcparams(base_size=7.5)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "model_lib/results/p1v2_runs"
OUT = Path(__file__).resolve().parent / "fig5_overview"

# dataset -> (run dir, representative test cell)
# cells: per-cell RMSE close to the dataset median and a long aging range
# (TJU: the longest test cell, 890 cycles, down to 75% SOH)
RUNS = {
    "MIT":  ("1031_0117_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_mit_seed42", "b1c24"),
    "HUST": ("1031_0431_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_hust_seed42", "6-2"),
    "TJU":  ("1031_0938_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_tju_seed42", "CY45-05_1-#24"),
}
CHEM = {"MIT": "LFP", "HUST": "LFP", "TJU": "NCM"}

SCEN_ORDER = ["chg_lo", "chg_mid", "chg_hi", "dis_lo", "dis_mid", "dis_hi"]
SCEN_COLOR = {
    "chg_lo": "#1F3A52", "chg_mid": "#4F7A9A", "chg_hi": "#9DBBD0",
    "dis_lo": "#6E2A1C", "dis_mid": "#B0583F", "dis_hi": "#E0A48C",
}
SCEN_LABEL = {s: s.replace("_", " ") for s in SCEN_ORDER}


def _load(run: str) -> pd.DataFrame:
    df = pd.read_csv(RUN_ROOT / run / "predictions" / "test_predictions_hard.csv")
    df["t"] = df.soh_true * 100
    df["p"] = df.soh_pred * 100
    return df


def _metrics(df: pd.DataFrame) -> tuple[float, float]:
    e = df.p - df.t
    rmse = float(np.sqrt((e ** 2).mean()))
    r2 = 1 - float((e ** 2).sum() / ((df.t - df.t.mean()) ** 2).sum())
    return rmse, r2


def _parity(ax, df, name, lim, norm):
    hb = ax.hexbin(df.t, df.p, gridsize=55, extent=(*lim, *lim), cmap="viridis",
                   norm=norm, mincnt=1, linewidths=0, rasterized=True)
    ax.plot(lim, lim, color=INK, lw=0.7, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    rmse, r2 = _metrics(df)
    ax.text(0.04, 0.96, f"RMSE {rmse:.2f}%\n$R^2$ {r2:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.8, color=INK,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, lw=0.6))
    ax.set_title(f"{name} ({CHEM[name]})", fontsize=9)
    ax.set_xlabel("Measured SOH (%)")
    strip_top_right(ax)
    return hb


def _trajectory(fig, spec, df, name, cell):
    gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, height_ratios=[3.2, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)
    c = df[df.cell_id == cell].sort_values("cycle")
    for s in SCEN_ORDER:
        d = c[c.seg_name == s]
        ax.scatter(d.cycle, d.p, s=0.6, color=SCEN_COLOR[s], alpha=0.55, lw=0, rasterized=True)
        ax_r.scatter(d.cycle, d.p - d.t, s=0.6, color=SCEN_COLOR[s], alpha=0.55, lw=0, rasterized=True)
    true = c.groupby("cycle").t.mean()
    ax.plot(true.index, true.values, color=INK, lw=1.0)
    rmse, _ = _metrics(c)
    if name == "TJU":   # convex fade leaves the upper right empty
        ax.text(0.97, 0.95, f"cell {cell}\nRMSE {rmse:.2f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.6, color=INK)
    else:               # concave fade leaves the lower left empty
        ax.text(0.04, 0.05, f"cell {cell}\nRMSE {rmse:.2f}%", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.6, color=INK)
    ax.set_title(f"{name} ({CHEM[name]})", fontsize=9)
    ax_r.axhline(0, color=INK, lw=0.6)
    lim = np.nanpercentile(np.abs(c.p - c.t), 99.5)
    lim = max(2.0, np.ceil(lim))
    ax_r.set_ylim(-lim, lim)
    ax_r.set_xlabel("Cycle")
    plt.setp(ax.get_xticklabels(), visible=False)
    for a in (ax, ax_r):
        strip_top_right(a)
    return ax, ax_r


def build_figure():
    data = {k: _load(r) for k, (r, _) in RUNS.items()}
    fig = plt.figure(figsize=(7.1, 4.75))
    outer = fig.add_gridspec(2, 3, height_ratios=[1, 1.08], hspace=0.62, wspace=0.42,
                             left=0.075, right=0.90, top=0.93, bottom=0.17)

    norm = LogNorm(vmin=1, vmax=max(
        np.histogram2d(d.t, d.p, bins=55)[0].max() for d in data.values()))
    hb = None
    parity_axes = []
    for j, (name, df) in enumerate(data.items()):
        lo = np.floor(min(df.t.min(), df.p.min()) / 5) * 5
        hi = 105.0
        ax = fig.add_subplot(outer[0, j])
        hb = _parity(ax, df, name, (lo, hi), norm)
        parity_axes.append(ax)
    parity_axes[0].set_ylabel("Estimated SOH (%)")

    traj_axes = []
    for j, (name, df) in enumerate(data.items()):
        ax, ax_r = _trajectory(fig, outer[1, j], df, name, RUNS[name][1])
        traj_axes.append((ax, ax_r))
    traj_axes[0][0].set_ylabel("SOH (%)")
    traj_axes[0][1].set_ylabel("Error (pp)")

    # colourbar for the parity row
    p0 = parity_axes[-1].get_position()
    cax = fig.add_axes([0.915, p0.y0, 0.012, p0.height])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("Segments per bin", fontsize=7.8)
    cb.ax.tick_params(labelsize=7.2)

    # shared scenario legend
    handles = [Line2D([], [], color=INK, lw=1.0, label="Measured")]
    handles += [Line2D([], [], ls="", marker="o", ms=3.6, color=SCEN_COLOR[s], label=SCEN_LABEL[s])
                for s in SCEN_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=7.8,
               bbox_to_anchor=(0.49, 0.015), handletextpad=0.3, columnspacing=1.2)

    # panel labels
    fig.text(0.012, 0.955, "(a)", fontsize=10.9, fontweight="bold", color=INK)
    ty = traj_axes[0][0].get_position().y1
    fig.text(0.012, ty + 0.03, "(b)", fontsize=10.9, fontweight="bold", color=INK)
    return fig


def main():
    fig = build_figure()
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300)
    print("saved", OUT.with_suffix(".png"), OUT.with_suffix(".pdf"))
    for name, (run, cell) in RUNS.items():
        df = _load(run)
        print(name, "dataset RMSE %.2f R2 %.3f" % _metrics(df),
              "| cell", cell, "RMSE %.2f" % _metrics(df[df.cell_id == cell])[0])


if __name__ == "__main__":
    main()
