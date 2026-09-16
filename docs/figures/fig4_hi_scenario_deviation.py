"""Figure 4 -- within-category cross-scenario deviation, most vs. least.

Companion to Figure 3-1 (fig3_hi_design_rationale.py, redundancy heatmap
only since 2026-09-16). This figure asks a different question: scenario-
sensitivity isn't just a category-level property (stat vs. diff vs. lfp vs.
morph) -- it varies a lot WITHIN a category too. So instead of one
"representative" HI per category (picked by strongest SOH-correlation, the
old Fig3 (b)/(c) approach), this picks, independently within EACH of the 4
categories, the member HI with the LARGEST cross-scenario deviation (row 1,
panels b-e) and the member HI with the SMALLEST (row 2, panels f-i).

x = Cycle (one representative cell, HUST 1-7, same cell as Fig6/7/arch/
fig3-1), y = that HI's z-score (each of the 6 scenario columns z-scored
independently against its own population mean/std). The line is the mean
across the 6 scenarios at each cycle; the shaded band is their min-max
spread -- band WIDTH is the visual (and printed, mean band width) measure of
cross-scenario deviation.

Real data: same HI_CACHE/GATE_JSON as fig3_hi_design_rationale.py (imported
from there, not recomputed) -- reuses base_names_and_categories/CAT_*/
load_hi_frame so the two figures stay on the same category vocabulary.

Run: python docs/figures/fig3_hi_scenario_deviation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import INK, SUBINK, SCEN_NAMES, setup_rcparams, label_panel
from fig3_hi_design_rationale import (CAT_ORDER, CAT_LABEL, CAT_COLOR,
                                        base_names_and_categories, load_hi_frame)

setup_rcparams()

REP_CELL = ("HUST", "1-7")   # same representative cell as Fig6/Fig7/fig_arch/fig3-1
MIN_STD = 1e-6               # skip near-constant HI (division-by-~0 in z-score)


def zscored_scenario_columns(df, rep):
    """{scenario: z-scored array, full population} for one HI -- each
    scenario column normalised independently against its OWN mean/std
    across all cells."""
    out = {}
    for s in SCEN_NAMES:
        col = f"{rep}_{s}"
        vals = df[col].values
        std = np.nanstd(vals)
        out[s] = (vals - np.nanmean(vals)) / (std + 1e-12) if std > MIN_STD else np.full_like(vals, np.nan)
    return out


def cell_cycle_band(df, rep, dataset, cell_id):
    """(cycle, mean_z, lo_z, hi_z) for one cell -- mean/min/max of the 6
    z-scored scenario columns at each of that cell's real cycles."""
    z_cols = zscored_scenario_columns(df, rep)
    zdf = pd.DataFrame(z_cols, index=df.index)
    zdf["cycle"] = df["cycle"].values
    mask = (df["dataset"].values == dataset) & (df["cell_id"].values == cell_id)
    zdf = zdf.loc[mask].sort_values("cycle")
    vals = zdf[SCEN_NAMES].values
    valid = ~np.all(np.isnan(vals), axis=1)
    zdf, vals = zdf.loc[valid], vals[valid]
    return (zdf["cycle"].values, np.nanmean(vals, axis=1),
            np.nanmin(vals, axis=1), np.nanmax(vals, axis=1))


def mean_band_width(df, rep):
    _, _, lo, hi = cell_cycle_band(df, rep, *REP_CELL)
    if len(lo) == 0:
        return np.nan
    return float(np.nanmean(hi - lo))


def pick_extremes_within_category(df, order, cats):
    """{cat: (max_dev_HI, min_dev_HI)} -- independently within each category,
    the member with the widest and narrowest mean band width. Members whose
    band width can't be computed (near-constant columns, all-NaN cell data)
    are skipped."""
    out = {}
    for cat in CAT_ORDER:
        members = [n for n, c in zip(order, cats) if c == cat]
        widths = {n: mean_band_width(df, n) for n in members}
        widths = {n: w for n, w in widths.items() if not np.isnan(w)}
        ranked = sorted(widths, key=widths.get, reverse=True)
        out[cat] = (ranked[0], ranked[-1], widths[ranked[0]], widths[ranked[-1]])
    return out


def panel_band(ax, df, cat, rep, width, row_label):
    cycle, mean_z, lo_z, hi_z = cell_cycle_band(df, rep, *REP_CELL)
    c = CAT_COLOR[cat]
    ax.fill_between(cycle, lo_z, hi_z, color=c, alpha=0.22, linewidth=0, zorder=3)
    ax.plot(cycle, mean_z, color=c, lw=1.3, zorder=4)
    ax.set_title(f"{CAT_LABEL[cat]} -- {row_label}\n{rep}", loc="left", fontsize=7.2, color=c, pad=4)
    ax.text(0.03, 0.06, f"mean band width = {width:.2f}", transform=ax.transAxes,
             fontsize=6.0, color=SUBINK, style="italic")
    ax.set_xlabel("Cycle", fontsize=6.8, labelpad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.6, labelsize=5.8)


def build_figure():
    order, cats = base_names_and_categories()
    df = load_hi_frame(order)
    extremes = pick_extremes_within_category(df, order, cats)

    fig = plt.figure(figsize=(15.6, 8.2))
    gs = fig.add_gridspec(2, 4, wspace=0.34, hspace=0.55,
                           top=0.88, bottom=0.08, left=0.045, right=0.99)

    letters_top = iter(["(b)", "(c)", "(d)", "(e)"])
    letters_bot = iter(["(f)", "(g)", "(h)", "(i)"])
    axes_all = []
    for col, cat in enumerate(CAT_ORDER):
        hi_max, hi_min, w_max, w_min = extremes[cat]

        ax_top = fig.add_subplot(gs[0, col])
        panel_band(ax_top, df, cat, hi_max, w_max, "largest deviation")
        label_panel(fig, ax_top, next(letters_top), dx=-0.045, dy=0.014)
        axes_all.append(ax_top)

        ax_bot = fig.add_subplot(gs[1, col])
        panel_band(ax_bot, df, cat, hi_min, w_min, "smallest deviation")
        label_panel(fig, ax_bot, next(letters_bot), dx=-0.045, dy=0.014)
        axes_all.append(ax_bot)

    axes_all[0].set_ylabel("HI (z), mean ± 6-scenario range", fontsize=6.8)
    axes_all[1].set_ylabel("HI (z), mean ± 6-scenario range", fontsize=6.8)

    fig.suptitle("Within each category: the HI with the largest (row 1) vs. smallest (row 2) "
                  "cross-scenario deviation", fontsize=9.0, color=INK, y=0.965)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig4_hi_scenario_deviation.png"
    out_pdf = "docs/figures/fig4_hi_scenario_deviation.pdf"
    fig.savefig(out_png, dpi=500)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
