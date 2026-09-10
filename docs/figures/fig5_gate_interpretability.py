"""Figure 5 -- Gate interpretability (F15 + F16), the paper's "main" gate figure.

(a) raw HI selection matrix   (b) kernel HI selection matrix
(c) raw Jaccard similarity    (d) kernel Jaccard similarity

Real data: gates/regression_HIs.json + regression_kernel_HIs.json from a
completed p1v4 run. Update RUN_DIR below to point at whichever run should be
the paper's canonical example.

Run: python docs/figures/fig5_gate_interpretability.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from _style import (INK, SUBINK, FAINT, GRID, SCEN_NAMES, SCEN_POS_COLOR, setup_rcparams,
                     label_panel, strip_top_right, PROJECT_ROOT)

setup_rcparams()

RUN_DIR = PROJECT_ROOT / "5_model/experiments/phase1_lab/results/p1v2_runs/0827_1705_p1v2_p1v4_full_seed42"
RAW_JSON = RUN_DIR / "gates" / "regression_HIs.json"
KERNEL_JSON = RUN_DIR / "gates" / "regression_kernel_HIs.json"

CAT_ORDER = ["stat", "diff", "lfp", "morph"]
CAT_LABEL = {"stat": "stat", "diff": "diff (dQ/dV)", "lfp": "lfp-specific", "morph": "morphology"}
CAT_COLOR = {"stat": "#3D6E8C", "diff": "#D9A544", "lfp": "#B14A3C", "morph": "#6B4E8C"}

SEL_CMAP = LinearSegmentedColormap.from_list("sel", ["#F4F1E8", "#274C63"])


def load(path: Path) -> dict:
    return json.load(open(path))


def raw_matrix(gj: dict):
    scen_probs = {}
    for s in range(6):
        seg_name = gj[f"seg_{s}_seg_name"]
        names, probs = gj[f"seg_{s}_names"], gj[f"seg_{s}_probs"]
        scen_probs[seg_name] = {n.replace("_" + seg_name, ""): p for n, p in zip(names, probs)}
    base_names = list(scen_probs[SCEN_NAMES[0]].keys())

    def cat_of(n):
        return n.split("_")[0]

    order = sorted(base_names, key=lambda n: (CAT_ORDER.index(cat_of(n)),
                                               -max(scen_probs[s][n] for s in SCEN_NAMES)))
    mat = np.array([[scen_probs[s][n] for s in SCEN_NAMES] for n in order])
    cats = [cat_of(n) for n in order]
    return order, cats, mat


def kernel_matrix(gj: dict):
    scen_probs = {}
    for s in range(6):
        seg_name = gj[f"seg_{s}_seg_name"]
        names, probs = gj[f"seg_{s}_names"], gj[f"seg_{s}_probs"]
        scen_probs[seg_name] = dict(zip(names, probs))
    all_names = list(scen_probs[SCEN_NAMES[0]].keys())

    def origin_of(n):
        for sc in SCEN_NAMES:
            if f"_{sc}_g" in n:
                return sc
        return "other"

    order = sorted(all_names, key=lambda n: (SCEN_NAMES.index(origin_of(n)),
                                              -max(scen_probs[s][n] for s in SCEN_NAMES)))
    mat = np.array([[scen_probs[s][n] for s in SCEN_NAMES] for n in order])
    origins = [origin_of(n) for n in order]
    return order, origins, mat


def selected_set(gj, seg_idx, thresh=0.5, strip=False):
    seg_name = gj[f"seg_{seg_idx}_seg_name"]
    names, probs = gj[f"seg_{seg_idx}_names"], gj[f"seg_{seg_idx}_probs"]
    sel = [n for n, p in zip(names, probs) if p > thresh]
    if strip:
        sel = [n.replace("_" + seg_name, "") for n in sel]
    return set(sel)


def jaccard_matrix(gj, strip):
    sets = [selected_set(gj, s, strip=strip) for s in range(6)]
    J = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            a, b = sets[i], sets[j]
            J[i, j] = len(a & b) / len(a | b) if (a | b) else 1.0
    return J, [len(s) for s in sets]


def panel_selection(ax, order, groups, mat, group_order, group_label, group_color, n_total_label):
    im = ax.imshow(mat, aspect="auto", cmap=SEL_CMAP, vmin=0, vmax=1, interpolation="none")
    ax.set_xticks(range(6))
    ax.set_xticklabels(SCEN_NAMES, rotation=40, ha="right", fontsize=6.8)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    boundaries = [0]
    cur = groups[0]
    for i, g in enumerate(groups):
        if g != cur:
            boundaries.append(i)
            cur = g
    boundaries.append(len(groups))

    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="white", lw=1.4)
    for gi in range(len(boundaries) - 1):
        s, e = boundaries[gi], boundaries[gi + 1]
        gname = groups[s]
        ax.add_patch(plt.Rectangle((-0.6, s - 0.5), 0.22, e - s, facecolor=group_color[gname],
                                    edgecolor="none", clip_on=False, transform=ax.transData))
        ax.text(-0.85, (s + e) / 2 - 0.5, group_label[gname], ha="right", va="center",
                fontsize=6.6, color=SUBINK, rotation=0)

    ax.set_xlim(-0.5, 5.5)
    ax.text(1.0, 1.012, n_total_label, transform=ax.transAxes, fontsize=6.8,
            color=FAINT, ha="right", va="bottom", style="italic")
    return im


def panel_jaccard(ax, J, sizes, title):
    im = ax.imshow(J, cmap=SEL_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(SCEN_NAMES, rotation=40, ha="right", fontsize=6.8)
    ax.set_yticklabels([f"{s}  (n={n})" for s, n in zip(SCEN_NAMES, sizes)], fontsize=6.8)
    for i in range(6):
        for j in range(6):
            v = J[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if v > 0.55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", pad=6, fontsize=8.0)
    return im


def build_figure():
    raw_gj = load(RAW_JSON)
    ker_gj = load(KERNEL_JSON)
    raw_order, raw_cats, raw_mat = raw_matrix(raw_gj)
    ker_order, ker_origins, ker_mat = kernel_matrix(ker_gj)
    J_raw, sizes_raw = jaccard_matrix(raw_gj, strip=True)
    J_ker, sizes_ker = jaccard_matrix(ker_gj, strip=False)

    fig = plt.figure(figsize=(7.6, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.10, 2.15, 0.95], hspace=0.05,
                           top=0.935, bottom=0.045, left=0.185, right=0.965)
    gs.update(hspace=0.5)
    ax_cbar = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])

    im_a = panel_selection(ax_a, raw_order, raw_cats, raw_mat, CAT_ORDER, CAT_LABEL, CAT_COLOR,
                           f"n = {len(raw_order)} raw HI")
    ax_a.set_title("Raw HI selection probability", loc="left", pad=6, fontsize=8.0)

    origin_label = {s: s for s in SCEN_NAMES}
    origin_color = {s: c for s, c in zip(SCEN_NAMES, SCEN_POS_COLOR)}
    panel_selection(ax_b, ker_order, ker_origins, ker_mat, SCEN_NAMES, origin_label, origin_color,
                    f"n = {len(ker_order)} kernel HI")
    ax_b.set_title("Kernel HI selection probability", loc="left", pad=6, fontsize=8.0)

    panel_jaccard(ax_c, J_raw, sizes_raw, "Cross-scenario Jaccard similarity (raw)")
    panel_jaccard(ax_d, J_ker, sizes_ker, "Cross-scenario Jaccard similarity (kernel)")

    cbar = fig.colorbar(im_a, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("gate selection probability", fontsize=7.4, color=SUBINK, labelpad=4)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.tick_params(labelsize=6.4, length=2)
    cbar.outline.set_visible(False)

    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")):
        label_panel(fig, ax, letter, dx=-0.06 if ax in (ax_a, ax_c) else -0.05, dy=0.010)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig5_gate_interpretability.png"
    out_pdf = "docs/figures/fig5_gate_interpretability.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
