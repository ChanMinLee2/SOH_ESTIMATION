"""Figure 3 -- HI design rationale (B4 + B6, + B5).

(a) redundancy structure: pairwise |r| among the 64 HI, axes in CATEGORY
    order (stat/diff/lfp/morph blocks, no dendrogram reordering) -- open
    markers flag pairs with |r|>=0.95 (the extreme-multicollinearity cases)
(b) one representative HI per category (Table 1), z-scored vs SOH, replotted
    for each of the 6 scenarios along a depth axis -- combines "what does a
    real HI look like" and "how much does it vary across scenarios" in one
    3D view

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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from _style import (INK, SUBINK, FAINT, GRID, SCEN_NAMES, setup_rcparams,
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


def pick_representatives(df, order, cats):
    """One HI per category = the member with the strongest mean |corr(HI, SOH)|
    across all 6 scenarios -- a data-driven, defensible pick of "the HI in
    this category that actually carries the most SOH signal"."""
    reps = {}
    for cat in CAT_ORDER:
        members = [n for n, c in zip(order, cats) if c == cat]
        best_n, best_score = None, -1.0
        for n in members:
            corrs = [abs(df[f"{n}_{s}"].corr(df["soh"])) for s in SCEN_NAMES
                     if f"{n}_{s}" in df.columns]
            corrs = [c for c in corrs if not np.isnan(c)]
            score = float(np.mean(corrs)) if corrs else -1.0
            if score > best_score:
                best_n, best_score = n, score
        reps[cat] = best_n
    return reps


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


# ---------------------------------------------------------- panels (b,c) ---

N_BINS = 14
MIN_BIN_N = 50


def _soh_bins(df):
    soh = df["soh"].values
    bins = np.linspace(np.nanpercentile(soh, 1), np.nanpercentile(soh, 99), N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.clip(np.digitize(soh, bins) - 1, 0, N_BINS - 1)
    return centers, bin_idx


def compute_category_surface(df, rep, bin_idx):
    """(6, N_BINS) z-scored-per-scenario-per-SOH-bin surface for one
    representative HI, NaN-gaps patched and extreme cells clipped (see
    inline notes) so plot_surface renders a clean sheet."""
    Z = np.full((6, N_BINS), np.nan)
    for yi, s in enumerate(SCEN_NAMES):
        col = f"{rep}_{s}"
        if col not in df.columns:
            continue
        vals = df[col].values
        z = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-12)
        for b in range(N_BINS):
            m = bin_idx == b
            # sparse bins (few valid, non-NaN samples for this specific
            # scenario column) give wildly noisy means that would poke a
            # spike through the surface -- require a minimum count, else
            # leave a gap to be patched below rather than plot noise.
            if np.sum(~np.isnan(vals[m])) >= MIN_BIN_N:
                Z[yi, b] = np.nanmean(z[m])
    # plot_surface can't render NaN holes cleanly -- patch any gap (only ever
    # at the sparse SOH extremes) by interpolating along the SOH axis.
    for yi in range(6):
        row = pd.Series(Z[yi])
        Z[yi] = row.interpolate(limit_direction="both").values
    # a handful of (scenario, SOH-bin) cells still have a genuine but extreme
    # mean even above MIN_BIN_N (large-but-skewed sample, not a sparsity
    # artifact) -- clip so one cell can't dominate the z-scale.
    return np.clip(Z, -3.0, 2.0)


def rank_by_cross_scenario_deviation(surfaces: dict) -> list[str]:
    """Categories ranked by how much their representative HI's typical level
    (SOH-bin-averaged) shifts across the 6 scenarios -- std of the 6
    per-scenario means, descending (most scenario-sensitive first)."""
    scores = {cat: np.std(np.nanmean(Z, axis=1)) for cat, Z in surfaces.items()}
    return sorted(scores, key=scores.get, reverse=True)


def panel_hi_scenario_3d(ax, centers, surfaces: dict, cats_subset: list[str], reps, title):
    X, Y = np.meshgrid(centers, np.arange(6))
    for cat in cats_subset:
        ax.plot_surface(X, Y, surfaces[cat], color=CAT_COLOR[cat], alpha=0.62,
                         edgecolor="none", linewidth=0, antialiased=True, shade=True, zorder=4)

    ax.set_yticks(range(6))
    ax.set_yticklabels(SCEN_NAMES, fontsize=6.2)
    ax.set_zlim(-3.0, 2.0)
    ax.set_xlabel("SOH", fontsize=7.4, labelpad=1)
    ax.set_zlabel("Representative HI (z)", fontsize=7.0, labelpad=1)
    ax.tick_params(axis="both", labelsize=6.0, pad=0)
    ax.tick_params(axis="z", labelsize=6.0, pad=0)
    ax.invert_xaxis()
    ax.view_init(elev=22, azim=-55)
    ax.set_title(title, loc="left", pad=0, fontsize=8.0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLOR[c], alpha=0.75) for c in cats_subset]
    labels = [f"{CAT_LABEL[c]}: {reps[c]}" for c in cats_subset]
    leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 0.98),
                     frameon=False, fontsize=6.2, handlelength=1.0, labelspacing=0.35)
    for t in leg.get_texts():
        t.set_color(SUBINK)
    ax.xaxis.pane.set_edgecolor(GRID)
    ax.yaxis.pane.set_edgecolor(GRID)
    ax.zaxis.pane.set_edgecolor(GRID)
    ax.xaxis.pane.set_alpha(0.3)
    ax.yaxis.pane.set_alpha(0.3)
    ax.zaxis.pane.set_alpha(0.3)


def build_figure():
    order, cats = base_names_and_categories()
    df = load_hi_frame(order)
    reps = pick_representatives(df, order, cats)
    centers, bin_idx = _soh_bins(df)
    surfaces = {cat: compute_category_surface(df, reps[cat], bin_idx) for cat in CAT_ORDER}
    ranked = rank_by_cross_scenario_deviation(surfaces)
    high_dev, low_dev = ranked[:2], ranked[2:]

    fig = plt.figure(figsize=(15.6, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15], wspace=0.34,
                           top=0.90, bottom=0.14, left=0.045, right=0.985)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    ax_c = fig.add_subplot(gs[0, 2], projection="3d")

    im_a = panel_redundancy(ax_a, df, order, cats)
    panel_hi_scenario_3d(ax_b, centers, surfaces, high_dev, reps,
                         "High cross-scenario deviation")
    panel_hi_scenario_3d(ax_c, centers, surfaces, low_dev, reps,
                         "Low cross-scenario deviation")

    cbar = fig.colorbar(im_a, ax=ax_a, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(labelsize=6.2, length=2)
    cbar.outline.set_visible(False)

    label_panel(fig, ax_a, "(a)", dx=-0.045, dy=0.014)
    fig.text(0.362, 0.94, "(b)", fontsize=10.5, fontweight="bold", color=INK,
             va="bottom", ha="left")
    fig.text(0.685, 0.94, "(c)", fontsize=10.5, fontweight="bold", color=INK,
             va="bottom", ha="left")

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig3_hi_design_rationale.png"
    out_pdf = "docs/figures/fig3_hi_design_rationale.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
