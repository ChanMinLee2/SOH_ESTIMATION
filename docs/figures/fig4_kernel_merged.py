"""
docs/figures/fig4_kernel_merged.py — Figure 4: synergy grouping and kernel fusion, structure and effect.

2026-11-04: merges the schematic new_fig_3.png (former Figure 4) with the quantitative content of
fig7_synergy.py (former Figure 7). Data: _cache/fig7_synergy_cache.json made by fig7_synergy_cache.py
(132 synergy groups of size >= 2 from MIT 1031_0117 / HUST 1031_0431 / TJU 1031_0938).
All |r| values are on test-cell segments except the growth scores (train partial r from the synergy json)
and the train |r| in panel (f).

(a) schematic (new_fig_3.png) with links to the quantitative panels
(b) kernel |r| vs best member |r| with marginal distributions per dataset
(c) dataset x scenario map: median gain K - B, number of groups and gate-selected kernel HIs
(d) best member -> linear fit -> kernel HI ladder (half violins, paired lines, medians)
(e) linear synergy gain (L - B) vs nonlinear gain (K - L)
(f) kernel |r| on training vs test cells
(g) traced example: MIT chg_hi group g8

Run: python docs/figures/fig4_kernel_merged.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache" / "fig7_synergy_cache.json"
SCHEMATIC = HERE / "new_fig_3.png"
OUT = HERE / "fig4_kernel_merged"

W_MM, H_MM = 174.0, 214.0
DATASETS = ["MIT", "HUST", "TJU"]
DS_LABEL = {"MIT": "MIT (LFP)", "HUST": "HUST (LFP)", "TJU": "TJU (NCM)"}
DS_COLOR = {"MIT": "#2b6cb0", "HUST": "#2a9d8f", "TJU": "#c8553d"}
DS_MARKER = {"MIT": "o", "HUST": "s", "TJU": "^"}
SCENS = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
# level colors follow the schematic: light blue survivors, orange dashed linear fit, dark blue kernel HI
C_BEST, C_LIN, C_KER = "#8ec5e6", "#e08a2c", "#164f86"
CAT_COLORS = {"stat": "#4c72b0", "diff": "#dd8452", "lfp": "#55a868", "morph": "#8172b3"}
EXAMPLE = ("MIT", "kernel_chg_hi_g8")
SEL_THR = 0.9

plt.rcParams.update({
    "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8, "legend.fontsize": 5.6, "axes.linewidth": 0.55,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2,
    "ytick.major.size": 2.2, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def ax_mm(fig, x0, y0, w, h):
    """Axes from a top-left corner and size in mm."""
    return fig.add_axes([x0 / W_MM, 1 - (y0 + h) / H_MM, w / W_MM, h / H_MM])


def best(r):
    return max(abs(v) for v in r["member_r_test"])


def lin(r):
    return abs(r["linear_r_test"])


def ker(r):
    return abs(r["kernel_r_test"])


def title(ax, tag, text, x=0.0, y=1.0, **kw):
    ax.text(x, y, f"({tag})", transform=ax.transAxes, fontsize=7.4, fontweight="bold",
            va="bottom", ha="left", **kw)
    ax.text(x + 0.075 * kw.pop("scale", 1.0), y, text, transform=ax.transAxes, fontsize=6.6,
            va="bottom", ha="left", **kw)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e9e9e9", lw=0.4, zorder=0)


def badge(ax, x, y, text, fc="#164f86"):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=6.2, color="white", fontweight="bold",
            ha="center", va="center", zorder=10,
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.35", fc=fc, ec="white", lw=0.6))


# ---------------------------------------------------------------------------------------------
def panel_schematic(fig):
    img = plt.imread(SCHEMATIC)
    h_px, w_px = img.shape[:2]
    w = 109.0
    h = w * h_px / w_px
    ax = ax_mm(fig, 2.0, 7.0, w, h)
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()
    ax.text(0.0, 1.012, "(a)", transform=ax.transAxes, fontsize=7.4, fontweight="bold", va="bottom")
    ax.text(0.045, 1.012, "Synergy grouping and kernel fusion in each scenario", transform=ax.transAxes,
            fontsize=6.6, va="bottom")
    # links from schematic steps to the quantitative panels
    badge(ax, 0.445, 0.318, "→ (g)", fc="#e08a2c")          # group growing
    badge(ax, 0.925, 0.618, "→ (b)(f)", fc="#164f86")      # RBF / landmarks
    badge(ax, 0.925, 0.335, "→ (d)(e)", fc="#164f86")      # ridge output
    badge(ax, 0.925, 0.89, "→ (c)", fc="#555555")          # group selection
    return ax, h


def panel_scatter(fig, rows):
    x0, y0, s = 123.0, 15.5, 40.0
    ax = ax_mm(fig, x0, y0, s, s)
    axt = ax_mm(fig, x0, y0 - 7.5, s, 7.0)
    axr = ax_mm(fig, x0 + s + 0.8, y0, 7.0, s)
    ax.fill_between([0, 1], [0, 1], [1, 1], color="#eef4fa", zorder=0)
    ax.plot([0, 1], [0, 1], color="#8a8a8a", lw=0.6, ls="--", zorder=1)
    for k in (0.2, 0.4):
        ax.plot([0, 1 - k], [k, 1], color="#c9d7e6", lw=0.45, ls=":", zorder=1)
        ax.text(0.98 - k, 0.995, f"+{k:.1f}", fontsize=4.8, color="#7d97b3", ha="right", va="top")
    for r in rows:
        xv, yv = best(r), ker(r)
        if r["status"] != "kept":
            ax.scatter(xv, yv, marker="x", s=9, color="#9a9a9a", lw=0.6, zorder=2)
            continue
        sel = (r["gate_prob"] or 0.0) >= SEL_THR
        sz = {2: 7, 3: 11, 4: 15}[len(r["members"])]
        c = DS_COLOR[r["dataset"]]
        ax.scatter(xv, yv, marker=DS_MARKER[r["dataset"]], s=sz * (1.7 if sel else 1.0),
                   facecolor=c if sel else "white", edgecolor=c, lw=0.55, alpha=0.9, zorder=3)
    ex = next(r for r in rows if (r["dataset"], r["kernel"]) == EXAMPLE)
    ax.scatter(best(ex), ker(ex), s=70, facecolor="none", edgecolor="black", lw=0.8, zorder=5)
    ax.annotate("(g)", xy=(best(ex), ker(ex)), xytext=(0.04, 0.93), fontsize=5.8, fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="black", shrinkB=3))
    n_up = sum(ker(r) > best(r) for r in rows)
    ax.text(0.97, 0.05, f"K > B in {n_up}/{len(rows)}", ha="right", fontsize=5.8, color=C_KER,
            fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Best member |r| with SOH (B)")
    ax.set_ylabel("Kernel HI |r| with SOH (K)", labelpad=1)
    tidy(ax)
    bins = np.linspace(0, 1, 26)
    for ds in DATASETS:
        rs = [r for r in rows if r["dataset"] == ds]
        axt.hist([best(r) for r in rs], bins=bins, histtype="step", color=DS_COLOR[ds], lw=0.8)
        axt.hist([best(r) for r in rs], bins=bins, color=DS_COLOR[ds], alpha=0.12)
        axr.hist([ker(r) for r in rs], bins=bins, histtype="step", color=DS_COLOR[ds], lw=0.8,
                 orientation="horizontal")
        axr.hist([ker(r) for r in rs], bins=bins, color=DS_COLOR[ds], alpha=0.12, orientation="horizontal")
    for a in (axt, axr):
        a.set_axis_off()
    axt.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axt.text(0.0, 1.25, "(b)", transform=axt.transAxes, fontsize=7.4, fontweight="bold", va="bottom")
    axt.text(0.155, 1.25, "Kernel HI against its best member", transform=axt.transAxes, fontsize=6.6,
             va="bottom")
    return ax


def panel_map(fig, rows):
    ax = ax_mm(fig, 131.0, 70.0, 39.5, 18.0)
    gain = np.full((3, 6), np.nan)
    for i, ds in enumerate(DATASETS):
        for j, sc in enumerate(SCENS):
            rs = [r for r in rows if r["dataset"] == ds and r["scenario"] == sc]
            if rs:
                gain[i, j] = np.median([ker(r) - best(r) for r in rs])
            n = len(rs)
            nsel = sum((r["gate_prob"] or 0) >= SEL_THR for r in rs if r["status"] == "kept")
            ndrop = sum(r["status"] != "kept" for r in rs)
            txt = f"{n}" + (f"\n{nsel}★" if nsel else "\n–")
            ax.text(j, i, txt, ha="center", va="center", fontsize=4.9, linespacing=0.95,
                    color="white" if (not np.isnan(gain[i, j]) and gain[i, j] > 0.42) else "black")
            if ndrop:
                ax.text(j + 0.42, i - 0.40, "×" * ndrop, ha="right", va="top", fontsize=4.6, color="#b00020")
    im = ax.imshow(gain, cmap="Blues", vmin=0.1, vmax=0.6, aspect="auto")
    ax.set_xticks(range(6))
    ax.set_xticklabels([s.replace("_", "\n") for s in SCENS], fontsize=5.0, linespacing=0.85)
    ax.set_yticks(range(3))
    ax.set_yticklabels(DATASETS, fontsize=5.6)
    ax.tick_params(length=0, pad=1.2)
    ax.set_xticks(np.arange(-0.5, 6), minor=True)
    ax.set_yticks(np.arange(-0.5, 3), minor=True)
    ax.grid(which="minor", color="white", lw=0.9)
    ax.axvline(2.5, color="black", lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = ax_mm(fig, 131.0, 103.0, 39.5, 1.4)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0.1, 0.35, 0.6])
    cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cax.text(0.5, 1.6, "number: groups   ★: selected kernel HIs   ×: filtered\ncolor: median K − B of the cell",
             transform=cax.transAxes, fontsize=4.8, ha="center", va="bottom", color="#333333", linespacing=1.1)
    ax.text(-0.17, 1.12, "(c)", transform=ax.transAxes, fontsize=7.4, fontweight="bold", va="bottom")
    ax.text(-0.02, 1.12, "Groups per scenario and gain", transform=ax.transAxes, fontsize=6.6, va="bottom")


def half_violin(ax, data, pos, color, side=-1, width=0.42):
    kde_y = np.linspace(0, 1, 200)
    data = np.asarray(data)
    bw = 1.06 * data.std() * len(data) ** (-1 / 5) + 1e-3
    dens = np.exp(-0.5 * ((kde_y[:, None] - data[None, :]) / bw) ** 2).sum(1)
    dens = dens / dens.max() * width
    ax.fill_betweenx(kde_y, pos, pos + side * dens, color=color, alpha=0.55, lw=0, zorder=2)
    ax.plot(pos + side * dens, kde_y, color=color, lw=0.5, zorder=2)


def panel_ladder(fig, rows):
    ax = ax_mm(fig, 13.0, 119.0, 60.0, 51.0)
    kept = [r for r in rows if r["status"] == "kept"]
    gap = 3.7
    for di, ds in enumerate(DATASETS):
        rs = [r for r in kept if r["dataset"] == ds]
        vals = np.array([[best(r), lin(r), ker(r)] for r in rs])
        xs = di * gap + np.arange(3) * 1.05
        for v in vals:
            ax.plot(xs, v, color=DS_COLOR[ds], lw=0.25, alpha=0.28, zorder=1)
        for k, c in enumerate((C_BEST, C_LIN, C_KER)):
            half_violin(ax, vals[:, k], xs[k] - 0.05, c, side=-1, width=0.45)
            ax.scatter(np.full(len(vals), xs[k] + 0.09) + np.random.default_rng(k + di).uniform(-0.05, 0.05,
                       len(vals)), vals[:, k], s=1.6, color=c, edgecolor="none", zorder=3, alpha=0.9)
        med = np.median(vals, axis=0)
        ax.plot(xs, med, color="black", lw=0.8, zorder=4)
        ax.scatter(xs, med, marker="D", s=13, color=[C_BEST, C_LIN, C_KER], edgecolor="black", lw=0.5,
                   zorder=5)
        for k in range(3):
            ax.text(xs[k] + 0.17, med[k], f"{med[k]:.2f}", fontsize=4.9, va="center", zorder=6,
                    color="#222222")
        n_up = int((vals[:, 2] > vals[:, 1]).sum())
        ax.text(xs[1], 1.03, f"K > L  {n_up}/{len(vals)}", ha="center", va="bottom", fontsize=5.3,
                color=C_KER, fontweight="bold")
        ax.text(xs[1], -0.13, DS_LABEL[ds], ha="center", va="top", fontsize=6.0, color=DS_COLOR[ds],
                fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xticks([di * gap + k * 1.05 for di in range(3) for k in range(3)])
    ax.set_xticklabels(["B", "L", "K"] * 3, fontsize=5.6)
    ax.set_xlim(-0.75, 2 * gap + 2.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("|r| with SOH on test cells")
    tidy(ax)
    ax.grid(axis="x", visible=False)
    ax.text(0.0, 1.10, "(d)", transform=ax.transAxes, fontsize=7.4, fontweight="bold", va="bottom")
    ax.text(0.095, 1.10, "Best member (B), linear fit (L) and kernel HI (K)", transform=ax.transAxes,
            fontsize=6.6, va="bottom")


def panel_gain(fig, rows):
    ax = ax_mm(fig, 87.0, 117.0, 35.0, 35.0)
    kept = [r for r in rows if r["status"] == "kept"]
    ax.axhspan(0, 0.8, color="#eef4fa", zorder=0)
    ax.axhline(0, color="#777777", lw=0.5)
    ax.axvline(0, color="#777777", lw=0.5)
    for r in kept:
        sel = (r["gate_prob"] or 0) >= SEL_THR
        c = DS_COLOR[r["dataset"]]
        ax.scatter(lin(r) - best(r), ker(r) - lin(r), marker=DS_MARKER[r["dataset"]], s=10 if sel else 7,
                   facecolor=c if sel else "white", edgecolor=c, lw=0.5, alpha=0.9, zorder=3)
    ex = next(r for r in rows if (r["dataset"], r["kernel"]) == EXAMPLE)
    ax.scatter(lin(ex) - best(ex), ker(ex) - lin(ex), s=55, facecolor="none", edgecolor="black", lw=0.7,
               zorder=5)
    n_up = sum(ker(r) > lin(r) for r in kept)
    ax.text(0.97, 0.95, f"K > L\n{n_up}/{len(kept)}", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.6, color=C_KER, fontweight="bold")
    ax.text(0.97, 0.04, "linear only", transform=ax.transAxes, ha="right", fontsize=5.0, color="#777777")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.12, 0.72)
    ax.set_xlabel("Linear gain L − B")
    ax.set_ylabel("Nonlinear gain K − L", labelpad=1)
    tidy(ax)
    ax.text(-0.22, 1.06, "(e)", transform=ax.transAxes, fontsize=7.4, fontweight="bold", va="bottom")
    ax.text(-0.04, 1.06, "Linear vs nonlinear synergy", transform=ax.transAxes, fontsize=6.6, va="bottom")


def panel_traintest(fig, rows):
    ax = ax_mm(fig, 87.0, 166.0, 35.0, 35.0)
    kept = [r for r in rows if r["status"] == "kept"]
    ax.plot([0, 1], [0, 1], color="#8a8a8a", lw=0.6, ls="--")
    for ds in DATASETS:
        rs = [r for r in kept if r["dataset"] == ds]
        ax.scatter([abs(r["kernel_r_train"]) for r in rs], [ker(r) for r in rs], marker=DS_MARKER[ds],
                   s=7, facecolor="white", edgecolor=DS_COLOR[ds], lw=0.5, alpha=0.9, label=ds)
    tr = np.array([abs(r["kernel_r_train"]) for r in kept])
    te = np.array([ker(r) for r in kept])
    rho = np.corrcoef(tr, te)[0, 1]
    ax.text(0.04, 0.95, f"r = {rho:.2f}\nmedian drop {np.median(tr - te):.2f}", transform=ax.transAxes,
            va="top", fontsize=5.4)
    ax.set_xlim(0.2, 1)
    ax.set_ylim(0.2, 1)
    ax.set_xlabel("Kernel HI |r| on training cells")
    ax.set_ylabel("|r| on test cells", labelpad=1)
    tidy(ax)
    ax.text(-0.22, 1.06, "(f)", transform=ax.transAxes, fontsize=7.4, fontweight="bold", va="bottom")
    ax.text(-0.04, 1.06, "Frozen kernel maps on new cells", transform=ax.transAxes, fontsize=6.6,
            va="bottom")


def _cat(name):
    return name.split("_")[0]


def panel_example(fig, rows):
    ex = next(r for r in rows if (r["dataset"], r["kernel"]) == EXAMPLE)
    ax = ax_mm(fig, 128.0, 112.0, 45.0, 56.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.0, "(g)", fontsize=7.4, fontweight="bold", va="bottom")
    ax.text(0.10, 1.0, "Traced kernel HI (MIT chg hi)", fontsize=6.6, va="bottom")
    ax.text(0.0, 0.955, "growth order with partial r given earlier members", fontsize=5.0, color="#555555",
            va="top")
    n = len(ex["members"])
    ys = np.linspace(0.83, 0.42, n)
    xm, wb = 0.12, 0.60
    for i, (nm, rt) in enumerate(zip(ex["members"], ex["member_r_test"])):
        c = CAT_COLORS[_cat(nm)]
        ax.add_patch(FancyBboxPatch((xm, ys[i] - 0.05), wb, 0.10, boxstyle="round,pad=0.004,rounding_size=0.02",
                                    fc=c, ec="none", alpha=0.16))
        ax.add_patch(FancyBboxPatch((xm, ys[i] - 0.05), 0.016, 0.10, boxstyle="square,pad=0", fc=c, ec="none"))
        ax.text(xm + 0.035, ys[i] + 0.018, nm, fontsize=5.1, family="monospace", va="center")
        # |r| bar inside the box
        ax.add_patch(plt.Rectangle((xm + 0.035, ys[i] - 0.032), 0.5 * abs(rt), 0.018, fc=C_BEST, ec="none"))
        ax.text(xm + 0.045 + 0.5 * abs(rt), ys[i] - 0.023, f"|r| {abs(rt):.2f}", fontsize=4.9, va="center")
        ax.add_patch(FancyArrowPatch((xm + wb + 0.01, ys[i]), (0.86, 0.62), arrowstyle="-|>", mutation_scale=4,
                                     lw=0.55, color="#8a8a8a", connectionstyle="arc3,rad=-0.15"))
    for i in range(1, n):
        ax.add_patch(FancyArrowPatch((xm - 0.01, ys[i - 1] - 0.03), (xm - 0.01, ys[i] + 0.03), arrowstyle="-|>",
                                     mutation_scale=4.5, lw=0.6, color=C_LIN, connectionstyle="arc3,rad=0.6"))
        ax.text(xm - 0.085, (ys[i - 1] + ys[i]) / 2, f"{abs(ex['growth_scores'][i]):.2f}", fontsize=5.0,
                ha="center", va="center", color=C_LIN)
    ax.text(xm - 0.085, ys[0] + 0.065, "seed", fontsize=4.9, ha="center", color="#555555")
    ax.add_patch(FancyBboxPatch((0.80, 0.56), 0.18, 0.12, boxstyle="round,pad=0.006,rounding_size=0.03",
                                fc=C_KER, ec="none"))
    ax.text(0.89, 0.62, "kernel\nHI", color="white", ha="center", va="center", fontsize=5.8, fontweight="bold",
            linespacing=0.9)
    ax.text(0.89, 0.71, f"gate p {ex['gate_prob']:.2f}", ha="center", fontsize=5.0)
    # bar comparison
    bx = ax_mm(fig, 136.0, 150.5, 35.0, 17.5)
    labels = [f"m{i + 1}" for i in range(n)] + ["L", "K"]
    vals = [abs(v) for v in ex["member_r_test"]] + [lin(ex), ker(ex)]
    cols = [CAT_COLORS[_cat(m)] for m in ex["members"]] + [C_LIN, C_KER]
    bx.barh(range(len(vals))[::-1], vals, color=cols, height=0.7, alpha=0.9)
    for k, v in enumerate(vals):
        bx.text(v + 0.02, len(vals) - 1 - k, f"{v:.2f}", va="center", fontsize=4.9)
    bx.set_yticks(range(len(vals))[::-1])
    bx.set_yticklabels(labels, fontsize=5.0)
    bx.set_xlim(0, 1)
    bx.set_xlabel("|r| with SOH on test cells", fontsize=5.4, labelpad=1)
    bx.tick_params(axis="x", labelsize=5.0)
    bx.spines[["top", "right"]].set_visible(False)
    bx.axvline(max(vals[:n]), color=C_BEST, lw=0.6, ls=":")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, label=k) for k, c in CAT_COLORS.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(129.5 / W_MM, 1 - 172.5 / H_MM), ncol=4,
               frameon=False, handlelength=0.8, handletextpad=0.3, columnspacing=0.7, fontsize=5.2,
               title="HI category (g)", title_fontsize=5.4, alignment="left")


def legend_block(fig):
    h = [Line2D([0], [0], marker=DS_MARKER[d], ls="", mfc=DS_COLOR[d], mec=DS_COLOR[d], ms=3.6, label=DS_LABEL[d])
         for d in DATASETS]
    h += [Line2D([0], [0], marker="o", ls="", mfc="white", mec="black", ms=3.6, mew=0.6, label="gate not selected"),
          Line2D([0], [0], marker="o", ls="", mfc="black", mec="black", ms=3.6, label="gate selected (p ≥ 0.9)"),
          Line2D([0], [0], marker="x", ls="", color="#9a9a9a", ms=3.6, label="filtered out"),
          Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555555", ms=2.4, label="2 members"),
          Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555555", ms=4.2, label="4 members")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(10.0 / W_MM, 1 - 199.5 / H_MM), ncol=3,
               frameon=False, handletextpad=0.2, columnspacing=1.0, labelspacing=0.35, fontsize=5.5,
               title="Markers in (b) (e) (f)", title_fontsize=5.6, alignment="left")
    h2 = [plt.Rectangle((0, 0), 1, 1, fc=c, label=l) for c, l in
          ((C_BEST, "B  best single member"), (C_LIN, "L  least-squares fit of members"), (C_KER, "K  kernel HI"))]
    fig.legend(handles=h2, loc="upper left", bbox_to_anchor=(10.0 / W_MM, 1 - 182.0 / H_MM), ncol=1,
               frameon=False, handlelength=0.9, handletextpad=0.35, labelspacing=0.3, fontsize=5.5,
               title="Levels in (d) (g)", title_fontsize=5.6, alignment="left")


def main():
    rows = json.loads(CACHE.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax_a, h_a = panel_schematic(fig)
    ax_b = panel_scatter(fig, rows)
    panel_map(fig, rows)
    panel_ladder(fig, rows)
    panel_gain(fig, rows)
    panel_traintest(fig, rows)
    panel_example(fig, rows)
    legend_block(fig)
    # connector from the ridge output of the schematic to the scatter
    con = ConnectionPatch(xyA=(0.985, 0.43), coordsA=ax_a.transAxes, xyB=(0.0, 0.62), coordsB=ax_b.transAxes,
                          arrowstyle="-|>", mutation_scale=7, lw=0.8, color=C_KER,
                          connectionstyle="arc3,rad=-0.25", zorder=20)
    fig.add_artist(con)
    fig.text(112.5 / W_MM, 1 - 66.0 / H_MM, "fit on train\n|r| on test", fontsize=4.6, color=C_KER,
             ha="left", va="center")
    fig.add_artist(Line2D([4 / W_MM, 170 / W_MM], [1 - 109.0 / H_MM] * 2, color="#bdbdbd", lw=0.5, ls=(0, (3, 2))))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    print("saved", OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
