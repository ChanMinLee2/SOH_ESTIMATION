"""Figure 2 -- HI design rationale (B4 + B6, + B5).

(a) HI category composition (64 raw HI, 4 physically-motivated categories)
(b) redundancy structure: pairwise |r| among the 64 HI, hierarchically
    clustered -- motivates kernel/synergy grouping instead of using all 64
    independently
(c) HI-SOH univariate correlation, category x scenario -- minimum evidence
    that individual HIs actually carry SOH signal before any gate selects them

Real data: 4_hi_analysis/hi_features_qfref_*.pkl (cycle-level HI cache) and
gates/regression_HIs.json (for the canonical 64-name / category ordering
shared with Figure 5).

Run: python docs/figures/fig2_hi_design_rationale.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, dendrogram

from _style import (INK, SUBINK, FAINT, GRID, SCEN_NAMES, setup_rcparams,
                     label_panel, PROJECT_ROOT, DATA_4_HI_ROOT)

setup_rcparams()

HI_CACHE = DATA_4_HI_ROOT.parent / "4_hi_analysis" / (
    "hi_features_qfref_n1-35%_n2-20%_N-2_minpts5_lag-1_noise-3%_ou-200_calib-100_offA-5mA.pkl")
GATE_JSON = (PROJECT_ROOT / "5_model/experiments/phase1_lab/results/p1v2_runs/"
             "0827_1705_p1v2_p1v4_full_seed42/gates/regression_HIs.json")

CAT_ORDER = ["stat", "diff", "lfp", "morph"]
CAT_LABEL = {"stat": "stat", "diff": "diff (dQ/dV)", "lfp": "lfp-specific", "morph": "morphology"}
CAT_DESC = {
    "stat": "voltage/current distribution statistics (mean, entropy, skew, ...)",
    "diff": "differential-voltage / dQ-dV peak & curvature descriptors",
    "lfp": "phase-plateau geometry specific to LFP's two-phase transition",
    "morph": "whole-curve shape similarity (DTW / Frechet) to a reference",
}
CAT_COLOR = {"stat": "#3D6E8C", "diff": "#D9A544", "lfp": "#B14A3C", "morph": "#6B4E8C"}

DIVERGE = LinearSegmentedColormap.from_list("div", ["#3D6E8C", "#F4F1E8", "#B14A3C"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#F4F1E8", "#274C63"])


def base_names_and_categories():
    gate = json.load(open(GATE_JSON))
    names = sorted(set(n.replace("_chg_lo", "") for n in gate["seg_0_names"]))
    order = sorted(names, key=lambda n: CAT_ORDER.index(n.split("_")[0]))
    cats = [n.split("_")[0] for n in order]
    return order, cats


def load_hi_frame(base_names):
    df = pd.read_pickle(HI_CACHE)
    df = df.copy()
    first_cap = df.sort_values("cycle").groupby("cell_id").capacity_Ah.transform("first")
    df["soh"] = df.capacity_Ah / first_cap
    return df


def panel_categories(ax, order, cats):
    counts = {c: cats.count(c) for c in CAT_ORDER}
    ys = np.arange(len(CAT_ORDER))[::-1]
    for y, c in zip(ys, CAT_ORDER):
        ax.barh(y, counts[c], color=CAT_COLOR[c], height=0.55, zorder=3)
        ax.text(counts[c] + 0.6, y, f"n={counts[c]}", va="center", fontsize=7.2, color=INK,
                fontweight="bold")
        ax.text(0.3, y - 0.42, CAT_DESC[c], va="top", ha="left", fontsize=6.4, color=SUBINK,
                style="italic")
    ax.set_yticks(ys)
    ax.set_yticklabels([CAT_LABEL[c] for c in CAT_ORDER], fontsize=8.0, fontweight="bold")
    ax.set_xlim(0, 26)
    ax.set_ylim(-1.05, 3.75)
    ax.set_xlabel("number of raw HI definitions")
    ax.set_title(f"HI category composition (n = {sum(counts.values())})", loc="left",
                pad=6, fontsize=8.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)


def panel_redundancy(ax, df, order, cats, scen="dis_lo"):
    cols = [f"{n}_{scen}" for n in order]
    sub = df[cols].dropna()
    keep = [c for c in cols if sub[c].std() > 1e-12]
    dropped = len(cols) - len(keep)
    sub = sub[keep]
    order_r = [n for n, c in zip(order, cols) if c in keep]
    cats = [n.split("_")[0] for n in order_r]
    corr = sub.corr().values
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.clip(1 - np.abs(corr), 0, None)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(dist[np.triu_indices(len(dist), k=1)], method="average")
    leaf_order = dendrogram(Z, no_plot=True)["leaves"]
    corr_r = corr[np.ix_(leaf_order, leaf_order)]
    cats_r = [cats[i] for i in leaf_order]
    order = order_r

    im = ax.imshow(corr_r, cmap=DIVERGE, vmin=-1, vmax=1)
    for i, c in enumerate(cats_r):
        ax.add_patch(plt.Rectangle((i - 0.5, -2.6), 1, 1.6, facecolor=CAT_COLOR[c],
                                    edgecolor="none", clip_on=False))
        ax.add_patch(plt.Rectangle((-2.6, i - 0.5), 1.6, 1, facecolor=CAT_COLOR[c],
                                    edgecolor="none", clip_on=False))
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLOR[c]) for c in CAT_ORDER]
    ax.legend(handles, [CAT_LABEL[c] for c in CAT_ORDER], loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False, fontsize=6.4,
              handlelength=0.9, columnspacing=1.0, borderaxespad=0)
    ax.set_title(f"Redundancy structure, |Pearson r|  (n = {len(order)}, {scen} segment)",
                loc="left", pad=6, fontsize=8.0)
    return im


def panel_hi_soh_corr(ax, df, order, cats):
    mat = np.zeros((len(order), 6))
    for i, n in enumerate(order):
        for j, s in enumerate(SCEN_NAMES):
            col = f"{n}_{s}"
            mat[i, j] = df[col].corr(df["soh"]) if col in df.columns else np.nan

    im = ax.imshow(mat, aspect="auto", cmap=DIVERGE, vmin=-1, vmax=1, interpolation="none")
    ax.set_xticks(range(6))
    ax.set_xticklabels(SCEN_NAMES, rotation=40, ha="right", fontsize=6.8)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    boundaries = [0]
    cur = cats[0]
    for i, c in enumerate(cats):
        if c != cur:
            boundaries.append(i)
            cur = c
    boundaries.append(len(cats))
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="white", lw=1.4)
    for gi in range(len(boundaries) - 1):
        s, e = boundaries[gi], boundaries[gi + 1]
        gname = cats[s]
        ax.add_patch(plt.Rectangle((-0.62, s - 0.5), 0.22, e - s, facecolor=CAT_COLOR[gname],
                                    edgecolor="none", clip_on=False, transform=ax.transData))
        ax.text(-0.85, (s + e) / 2 - 0.5, CAT_LABEL[gname], ha="right", va="center",
                fontsize=6.6, color=SUBINK)
    ax.set_xlim(-0.5, 5.5)
    ax.set_title(f"HI-SOH correlation (Pearson r, n = {len(order)})", loc="left", pad=6, fontsize=8.0)
    return im


def build_figure():
    order, cats = base_names_and_categories()
    df = load_hi_frame(order)

    fig = plt.figure(figsize=(11.0, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.0, 1.05], wspace=0.55,
                           top=0.86, bottom=0.20, left=0.075, right=0.955)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_categories(ax_a, order, cats)
    im_b = panel_redundancy(ax_b, df, order, cats)
    im_c = panel_hi_soh_corr(ax_c, df, order, cats)

    for im, ax in ((im_b, ax_b), (im_c, ax_c)):
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.ax.tick_params(labelsize=6.2, length=2)
        cbar.outline.set_visible(False)

    for ax, letter, dx in ((ax_a, "(a)", -0.055), (ax_b, "(b)", -0.045), (ax_c, "(c)", -0.075)):
        label_panel(fig, ax, letter, dx=dx, dy=0.014)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig2_hi_design_rationale.png"
    out_pdf = "docs/figures/fig2_hi_design_rationale.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
