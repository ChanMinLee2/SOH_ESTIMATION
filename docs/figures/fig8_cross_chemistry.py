"""Figure 8 -- Cross-chemistry generalization (I27), supporting-evidence figure.

(a) selected raw-HI count by category, one independently-trained gate per
    chemistry group (MIT-only / HUST-only / CALCE+TJU-only)
(b) cross-chemistry Jaccard similarity of the selected raw-HI sets
(c) final test-set R^2 per chemistry group (oracle)

Real data: three independently-trained p1v4 runs restricted to a single
chemistry group each.

Run: python docs/figures/fig8_cross_chemistry.py
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from _style import INK, SUBINK, FAINT, GRID, setup_rcparams, label_panel, PROJECT_ROOT

setup_rcparams()

RUNS = {
    "MIT\n(LFP)": "0908_0100_p1v2_p1v4_mit_only_seed42",
    "HUST\n(LFP)": "0908_0358_p1v2_p1v4_hust_only_seed42",
    "CALCE+TJU\n(LCO+NCM)": "0908_0752_p1v2_p1v4_calce_tju_seed42",
}
GROUP_COLOR = {"MIT\n(LFP)": "#3D6E8C", "HUST\n(LFP)": "#5F8FA8", "CALCE+TJU\n(LCO+NCM)": "#B14A3C"}
CAT_ORDER = ["stat", "diff", "lfp", "morph"]
CAT_LABEL = {"stat": "stat", "diff": "diff (dQ/dV)", "lfp": "lfp-specific", "morph": "morphology"}
RUN_ROOT = PROJECT_ROOT / "5_model/experiments/phase1_lab/results/p1v2_runs"
SEQ = LinearSegmentedColormap.from_list("seq", ["#F4F1E8", "#274C63"])


def load_gate(run_id):
    return json.load(open(RUN_ROOT / run_id / "gates" / "regression_HIs.json"))


def load_metrics(run_id):
    return json.load(open(RUN_ROOT / run_id / "metrics" / "metrics.json", encoding="utf-8"))


def selected_base_names(gate, thresh=0.5):
    sel = set()
    for s in range(6):
        seg = gate[f"seg_{s}_seg_name"]
        for n, p in zip(gate[f"seg_{s}_names"], gate[f"seg_{s}_probs"]):
            if p > thresh:
                sel.add(n.replace("_" + seg, ""))
    return sel


def category_counts(gate, thresh=0.5):
    counts = {c: 0 for c in CAT_ORDER}
    for s in range(6):
        seg = gate[f"seg_{s}_seg_name"]
        for n, p in zip(gate[f"seg_{s}_names"], gate[f"seg_{s}_probs"]):
            if p > thresh:
                base = n.replace("_" + seg, "")
                counts[base.split("_")[0]] += 1
    return counts


def panel_category_counts(ax, gates):
    x = np.arange(len(CAT_ORDER))
    width = 0.25
    for i, (label, gate) in enumerate(gates.items()):
        counts = category_counts(gate)
        vals = [counts[c] for c in CAT_ORDER]
        bars = ax.bar(x + (i - 1) * width, vals, width=width * 0.92, color=GROUP_COLOR[label],
                       label=label.replace("\n", " "), zorder=3)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.08, str(v), ha="center",
                        va="bottom", fontsize=6.4, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABEL[c] for c in CAT_ORDER], fontsize=7.2)
    ax.set_ylabel("selected count (of up to 6 scenarios)")
    ax.set_ylim(0, max(6.5, ax.get_ylim()[1]))
    ax.set_title("Selected raw HI by category, per chemistry group", loc="left", pad=6, fontsize=8.0)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=6.2, handlelength=1.0,
                     borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(SUBINK)


def panel_jaccard(ax, gates):
    labels = list(gates.keys())
    sets = [selected_base_names(gates[l]) for l in labels]
    n = len(labels)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = sets[i], sets[j]
            J[i, j] = len(a & b) / len(a | b) if (a | b) else 1.0
    im = ax.imshow(J, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [l.replace("\n", " ") for l in labels]
    ax.set_xticklabels(short, rotation=25, ha="right", fontsize=6.8)
    ax.set_yticklabels([f"{s}  (n={len(st)})" for s, st in zip(short, sets)], fontsize=6.8)
    for i in range(n):
        for j in range(n):
            v = J[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.0,
                    color="white" if v > 0.55 else INK, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Cross-chemistry Jaccard similarity\n(selected raw-HI sets)", loc="left",
                pad=6, fontsize=8.0)
    return im


def panel_r2(ax, gates):
    labels = list(gates.keys())
    r2 = [load_metrics(RUNS[l])["test"]["oracle"]["capacity"]["r2"] for l in labels]
    colors = [GROUP_COLOR[l] for l in labels]
    bars = ax.bar(range(len(labels)), r2, color=colors, width=0.55, zorder=3)
    for b, v in zip(bars, r2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.4f}", ha="center", va="bottom",
                fontsize=7.2, color=INK, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=6.8, rotation=15, ha="right")
    ax.set_ylabel("Capacity $R^2$ (oracle, test)")
    ax.set_ylim(0.85, 1.0)
    ax.set_title("Per-chemistry final regression accuracy", loc="left", pad=6, fontsize=8.0)


def build_figure():
    gates = {label: load_gate(run_id) for label, run_id in RUNS.items()}

    fig = plt.figure(figsize=(10.6, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.9], wspace=0.5,
                           top=0.84, bottom=0.20, left=0.07, right=0.97)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_category_counts(ax_a, gates)
    im_b = panel_jaccard(ax_b, gates)
    panel_r2(ax_c, gates)

    cbar = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6.2, length=2)
    cbar.outline.set_visible(False)

    for ax in (ax_a, ax_c):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=2.8)

    for ax, letter, dx in ((ax_a, "(a)", -0.055), (ax_b, "(b)", -0.05), (ax_c, "(c)", -0.06)):
        label_panel(fig, ax, letter, dx=dx, dy=0.014)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig8_cross_chemistry.png"
    out_pdf = "docs/figures/fig8_cross_chemistry.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
