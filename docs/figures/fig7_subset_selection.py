"""Figure 7 -- what "subset learning"
actually selects.

The paper's title is "Scenario-Conditioned Subset Learning", but no existing
figure shows the learned subset itself: Fig3 is about raw-HI redundancy/
scenario-sensitivity (the ingredients), Fig4 is about kernel fusion's value,
and the architecture diagram is a schematic of the mechanism, not a result.
This figure closes that gap with the one thing that makes "subset learning"
concrete: the actual gate_prob (HI x scenario) learned by the canonical v4
run, i.e. the deployed subset -- raw HI (64) AND kernel HI (59).

Real data only -- gates/regression_HIs.json + gates/regression_kernel_HIs.json
from the same canonical run Fig6 uses (0904_1708_..._seed42), no retraining/
checkpoint loading needed (phase1_trainer_v2.py already exports both at the
end of training). Reuses fig3's category ordering/coloring
(base_names_and_categories, CAT_*) so the two figures read against the same
visual vocabulary.

Layout (2026-09-16 revision): one 64+59=123-row heatmap read top to bottom
was too tall to read comfortably, so this splits into 5 small per-category
panels (stat/diff/lfp/morph/kernel), each HI(rows, that category only) x
scenario(columns). Kernel (59 rows) gets its own column; the four raw
categories (64 rows total) stack in the other column so both sides come out
roughly the same height. A thin right-margin strip on each raw panel flags
shared-gate rows (identical prob across all 6 columns, from the hybrid
architecture's shared_gate) vs specific-gate rows -- kernel gates have no
shared option (always fully per-scenario, an architecture asymmetry noted in
Sec. 3.3/Discussion), so panel (e) has no such strip.

Run: python docs/figures/fig9_subset_selection.py   (PNG only, no PDF --
draft not yet placed in the paper).
"""

from __future__ import annotations

import json

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from _style import INK, SUBINK, GRID, setup_rcparams, label_panel, PROJECT_ROOT
from fig3_hi_design_rationale import CAT_ORDER, CAT_LABEL, CAT_COLOR, base_names_and_categories

setup_rcparams()

RUN_DIR = (PROJECT_ROOT / "5_model/experiments/phase1_lab/results/p1v2_runs/"
           "0904_1708_p1v2_p1v4_minpts5_calib100_offA5mA_seed42")
GATE_JSON = RUN_DIR / "gates" / "regression_HIs.json"
KERNEL_GATE_JSON = RUN_DIR / "gates" / "regression_kernel_HIs.json"
INTERACTION_JSON = PROJECT_ROOT / "5_model/experiments/phase1_lab/results/hi_scenario_interaction_k25_full_N2.json"

SCEN_NAMES = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
KERNEL_COLOR = "#555555"   # matches plot_hi_selection_matrix.py's dev-tool convention

# monochrome sequential (low chroma, Elsevier-safe) -- pale ground to LFP navy
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_navy", ["#F4F1E8", "#182E45"])


def load_raw_matrix():
    d = json.loads(GATE_JSON.read_text(encoding="utf-8"))
    order, cats = base_names_and_categories()
    mat = np.zeros((len(order), 6))
    for s, sname in enumerate(SCEN_NAMES):
        idx_to_prob = dict(zip(d[f"seg_{s}_ranked"], d[f"seg_{s}_probs"]))
        idx_to_name = dict(zip(d[f"seg_{s}_ranked"], d[f"seg_{s}_names"]))
        name_to_idx = {v.replace(f"_{sname}", ""): k for k, v in idx_to_name.items()}
        for r, hi in enumerate(order):
            mat[r, s] = idx_to_prob[name_to_idx[hi]]

    # Shared vs. specific is an architecture fact (which gate a HI was routed
    # to at construction time), not something to infer post-hoc from how
    # similar its learned probs look across scenarios -- two independently
    # -trained specific gates can both saturate to ~0 or ~1 for a given HI
    # without being the shared_gate. Read the ground truth from the same
    # interaction json that built the hybrid routing (n_significant=39
    # "specific" HIs, the rest route through shared_gate).
    interaction = json.loads(INTERACTION_JSON.read_text(encoding="utf-8"))
    is_shared = np.array([not interaction["per_hi"][hi]["significant"] for hi in order])
    return mat, order, cats, is_shared


def load_kernel_matrix():
    """(mat[n_kernel,6], names) -- kernel base names, mean-prob descending
    (no architecture-level category to sort by, unlike raw HI).

    Unlike raw HI, a kernel's own name already embeds the scenario its
    *source* raw-HI group came from ("kernel_chg_hi_g0" was built from
    chg_hi-segment raw HI) -- but the kernel feature itself is evaluated
    under all 6 scenario gates equally (confirmed: all 59 names appear
    identically in every seg_s_names list), so no suffix-stripping/
    realignment is needed -- the raw name IS the shared row identity."""
    d = json.loads(KERNEL_GATE_JSON.read_text(encoding="utf-8"))
    base_names = sorted(d["seg_0_names"])
    mat = np.zeros((len(base_names), 6))
    for s, sname in enumerate(SCEN_NAMES):
        name_to_prob = dict(zip(d[f"seg_{s}_names"], d[f"seg_{s}_probs"]))
        for r, kname in enumerate(base_names):
            mat[r, s] = name_to_prob[kname]
    order = np.argsort(-mat.mean(axis=1))
    return mat[order], [base_names[i] for i in order]


def draw_heatmap(ax, mat, row_labels, row_colors, show_xticks, show_shared_strip=None):
    n = len(row_labels)
    im = ax.imshow(mat, aspect="auto", cmap=SEQ_CMAP, vmin=0.0, vmax=1.0)
    ax.set_yticks(range(n))
    fontsize = 5.6 if n <= 25 else max(3.6, 5.6 - (n - 25) * 0.03)
    ax.set_yticklabels(row_labels, fontsize=fontsize)
    for t, c in zip(ax.get_yticklabels(), row_colors):
        t.set_color(c)
    ax.set_xticks(range(6))
    if show_xticks:
        ax.set_xticklabels(SCEN_NAMES, rotation=40, ha="right", fontsize=6.6)
    else:
        ax.set_xticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.axvline(2.5, color="white", lw=1.2, zorder=5)

    if show_shared_strip is not None:
        divider_x = 5.72
        for i in range(n):
            color = INK if show_shared_strip[i] else "#FFFFFF"
            ax.add_patch(plt.Rectangle((divider_x, i - 0.5), 0.42, 1, facecolor=color,
                                        edgecolor=GRID, linewidth=0.2, clip_on=False))
        ax.set_xlim(-0.5, divider_x + 0.42)
    return im


def build_figure():
    raw_mat, raw_order, raw_cats, is_shared = load_raw_matrix()
    ker_mat, ker_names = load_kernel_matrix()
    n_raw_by_cat = {c: raw_cats.count(c) for c in CAT_ORDER}

    fig = plt.figure(figsize=(9.6, 8.6))
    gs_outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.55,
                                  top=0.90, bottom=0.045, left=0.12, right=0.965)
    gs_raw = gs_outer[0, 0].subgridspec(4, 1, height_ratios=[n_raw_by_cat[c] for c in CAT_ORDER],
                                          hspace=0.55)
    ax_kernel = fig.add_subplot(gs_outer[0, 1])

    letters = iter(["(a)", "(b)", "(c)", "(d)"])
    row0 = 0
    im = None
    for i, cat in enumerate(CAT_ORDER):
        ax = fig.add_subplot(gs_raw[i, 0])
        n_cat = n_raw_by_cat[cat]
        sel = slice(row0, row0 + n_cat)
        row0 += n_cat
        im = draw_heatmap(ax, raw_mat[sel], raw_order[sel], [CAT_COLOR[cat]] * n_cat,
                           show_xticks=(i == 3), show_shared_strip=is_shared[sel])
        ax.set_title(f"{CAT_LABEL[cat]} (N={n_cat})", loc="left", fontsize=7.6,
                      color=CAT_COLOR[cat], pad=3)
        label_panel(fig, ax, next(letters), dx=-0.10, dy=0.006, fontsize=9.5)

    im_k = draw_heatmap(ax_kernel, ker_mat, ker_names, [KERNEL_COLOR] * len(ker_names),
                          show_xticks=True, show_shared_strip=None)
    ax_kernel.set_title(f"kernel (N={len(ker_names)}, no shared-gate option)", loc="left",
                          fontsize=7.6, color=KERNEL_COLOR, pad=3)
    label_panel(fig, ax_kernel, "(e)", dx=-0.06, dy=0.006, fontsize=9.5)

    # single shared strip legend (raw panels only) + colorbar, once at figure level
    strip_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor=GRID, linewidth=0.3),
                      plt.Rectangle((0, 0), 1, 1, facecolor="#FFFFFF", edgecolor=GRID, linewidth=0.3)]
    fig.legend(strip_handles, ["shared gate", "specific gate"], loc="upper left",
               bbox_to_anchor=(0.115, 0.975), ncol=2, frameon=False, fontsize=6.6,
               handlelength=1.0, columnspacing=0.8)

    cbar = fig.colorbar(im, ax=fig.axes, fraction=0.018, pad=0.015, location="right",
                         anchor=(0, 0.5), shrink=0.5)
    cbar.set_label("gate probability", fontsize=7.0, color=SUBINK)
    cbar.ax.tick_params(labelsize=6.0, length=2, color=SUBINK)
    cbar.outline.set_visible(False)

    fig.suptitle("Learned subset per scenario, raw + kernel HI (canonical v4 run, gate probability)",
                  fontsize=9.6, y=0.985, x=0.50)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig7_subset_selection.png"
    fig.savefig(out_png, dpi=500)
    print(f"saved: {out_png}")
