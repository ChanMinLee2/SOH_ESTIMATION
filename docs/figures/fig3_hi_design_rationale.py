"""Figure 3-1 -- HI redundancy structure (B4).

(a) redundancy structure: pairwise |r| among the 64 HI, axes in CATEGORY
    order (stat/diff/lfp/morph blocks, no dendrogram reordering) -- open
    markers flag pairs with |r|>=0.95 (the extreme-multicollinearity cases)

2026-09-16: split into two figures -- this file keeps only panel (a); the
per-HI cross-scenario deviation panels moved to
fig3_hi_scenario_deviation.py ("Figure 3-2") as their own 2-row small
multiple. Also exports base_names_and_categories/CAT_*/load_hi_frame, reused
by Fig9, fig_arch, and fig3-2 -- keep these names stable.

Category descriptions themselves now live in Table 1 (main text) rather than
a bar-chart panel; the old network-graph panel moved conceptually to Figure 4
(kernel/synergy-group fusion), since it is really about that story, not HI
design per se -- 2026-09-13 revision, both panels removed here.

Real data: 4_hi_analysis/hi_features_qfref_*.pkl (cycle-level HI cache) and
gates/regression_HIs.json (for the canonical 64-name / category ordering
shared with Figure 5).

Run: python docs/figures/fig3_hi_design_rationale.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from _style import (INK, SUBINK, SCEN_NAMES, setup_rcparams,
                     label_panel, PROJECT_ROOT, DATA_4_HI_ROOT)

setup_rcparams()

HI_CACHE = DATA_4_HI_ROOT.parent / "4_hi_analysis" / (
    "hi_features_qfref_n1-35%_n2-20%_N-2_minpts5_lag-1_noise-3%_ou-200_calib-100_offA-5mA.pkl")
GATE_JSON = (PROJECT_ROOT / "5_model/experiments/phase1_lab/results/p1v2_runs/"
             "0827_1705_p1v2_p1v4_full_seed42/gates/regression_HIs.json")

CAT_ORDER = ["stat", "diff", "lfp", "morph"]
CAT_LABEL = {"stat": "stat", "diff": "diff (dQ/dV)", "lfp": "lfp-specific", "morph": "morphology"}
CAT_COLOR = {"stat": "#3D6E8C", "diff": "#D9A544", "lfp": "#B14A3C", "morph": "#6B4E8C"}

DIVERGE = LinearSegmentedColormap.from_list("div", ["#3D6E8C", "#F4F1E8", "#B14A3C"])

EXTREME_CORR = 0.95


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


# ------------------------------------------------------------- panel (a) ---

def panel_redundancy(ax, df, order, cats, scen="dis_lo"):
    cols = [f"{n}_{scen}" for n in order]
    sub = df[cols].dropna()
    keep = [c for c in cols if sub[c].std() > 1e-12]
    sub = sub[keep]
    order_r = [n for n, c in zip(order, cols) if c in keep]
    cats_r = [n.split("_")[0] for n in order_r]
    corr = sub.corr().values
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    # No dendrogram reordering -- order_r is already category-blocked
    # (base_names_and_categories() sorts by CAT_ORDER), keep that as-is so
    # the axes read directly against the category color bars.
    n = len(order_r)

    im = ax.imshow(corr, cmap=DIVERGE, vmin=-1, vmax=1)
    ys, xs = np.where((np.abs(corr) >= EXTREME_CORR) & ~np.eye(n, dtype=bool))
    ax.scatter(xs, ys, s=7, marker="o", facecolor="none", edgecolor=INK,
               linewidths=0.6, zorder=5)

    for i, c in enumerate(cats_r):
        ax.add_patch(plt.Rectangle((i - 0.5, -2.6), 1, 1.6, facecolor=CAT_COLOR[c],
                                    edgecolor="none", clip_on=False))
        ax.add_patch(plt.Rectangle((-2.6, i - 0.5), 1.6, 1, facecolor=CAT_COLOR[c],
                                    edgecolor="none", clip_on=False))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    handles = ([plt.Rectangle((0, 0), 1, 1, color=CAT_COLOR[c]) for c in CAT_ORDER]
               + [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                              markeredgecolor=INK, markersize=5, markeredgewidth=0.8)])
    ax.legend(handles, [CAT_LABEL[c] for c in CAT_ORDER] + [f"|r|$\\geq${EXTREME_CORR}"],
              loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=5, frameon=False,
              fontsize=6.4, handlelength=0.9, columnspacing=1.0, borderaxespad=0)
    ax.set_title(f"Redundancy, |Pearson r| (n={n}, {scen}, category order)",
                 loc="left", pad=6, fontsize=7.6)
    return im


def build_figure():
    order, cats = base_names_and_categories()
    df = load_hi_frame(order)

    fig, ax_a = plt.subplots(figsize=(5.6, 5.1))
    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.14, right=0.92)
    im_a = panel_redundancy(ax_a, df, order, cats)
    cbar = fig.colorbar(im_a, ax=ax_a, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(labelsize=6.2, length=2)
    cbar.outline.set_visible(False)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig3_hi_design_rationale.png"
    out_pdf = "docs/figures/fig3_hi_design_rationale.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
