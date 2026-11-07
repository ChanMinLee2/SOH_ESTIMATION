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

W_MM, H_MM = 174.0, 150.0
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
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.7,
    "ytick.labelsize": 6.7, "legend.fontsize": 6.4, "axes.linewidth": 0.55,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
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
    ax.text(x, y, f"({tag})", transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="left", **kw)
    ax.text(x + 0.075 * kw.pop("scale", 1.0), y, text, transform=ax.transAxes, fontsize=7.6,
            va="bottom", ha="left", **kw)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e9e9e9", lw=0.4, zorder=0)


def badge(ax, x, y, text, fc="#164f86"):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=7.1, color="white", fontweight="bold",
            ha="center", va="center", zorder=10,
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.35", fc=fc, ec="white", lw=0.6))


# ---------------------------------------------------------------------------------------------
def panel_schematic(fig):
    """Vector schematic of stages 2-4 in one scenario (replaces the raster new_fig_3.png)."""
    X0, Y0, WS, HS = 2.0, 6.0, 112.0, 80.0
    ax = ax_mm(fig, X0, Y0, WS, HS)
    ax.set_xlim(0, WS)
    ax.set_ylim(HS, 0)
    ax.set_axis_off()
    sub = lambda x, y, w, h: ax_mm(fig, X0 + x, Y0 + y, w, h)
    rng = np.random.default_rng(3)
    ax.text(0.0, -1.2, "(a)", fontsize=8.5, fontweight="bold", va="bottom")
    ax.text(7.0, -1.2, "Synergy grouping and kernel fusion in each scenario $s$", fontsize=7.6, va="bottom")
    ax.add_patch(FancyBboxPatch((0.4, 0.4), WS - 0.8, HS - 0.8, boxstyle="round,pad=0,rounding_size=1.5",
                                fc="none", ec="#9a9a9a", lw=0.6))
    cols = ((1.8, 54.0, "Synergy grouping (linear)", C_LIN), (57.2, 53.0, "Kernel fusion (nonlinear)", C_KER))
    for x, w, t, c in cols:
        ax.add_patch(FancyBboxPatch((x, 1.8), w, 5.2, boxstyle="round,pad=0,rounding_size=1.0", fc=c, ec="none",
                                    alpha=0.9))
        ax.text(x + w / 2, 4.5, t, ha="center", va="center", fontsize=7.0, color="white", fontweight="bold")
    for y in (31.0, 55.0):
        for x, w, *_ in cols:
            ax.plot([x + 1, x + w - 1], [y, y], color="#c8c8c8", lw=0.5, ls=(0, (2, 1.5)))
    ax.plot([55.9, 55.9], [8.5, HS - 2], color="#d0d0d0", lw=0.5)

    def step(x, y, n, t):
        ax.text(x, y, f"{n}", fontsize=6.6, fontweight="bold", va="top",
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="#555555", lw=0.5))
        ax.text(x + 3.4, y + 0.1, t, fontsize=6.4, va="top")

    # ---- L1: redundancy collapse -----------------------------------------------------------------
    step(3.0, 9.0, 1, r"Collapse redundant HIs with $|\rho_{ij}| \geq \tau_\mathrm{red}$")
    n = 6
    C = np.abs(rng.uniform(-0.6, 0.6, (n, n)))
    C = (C + C.T) / 2
    for i, j in ((0, 2), (3, 4)):
        C[i, j] = C[j, i] = 0.95
    np.fill_diagonal(C, 1.0)
    m = sub(5.0, 16.5, 13.0, 13.0)
    m.imshow(C, cmap="Reds", vmin=0, vmax=1.15)
    for i, j in ((0, 2), (2, 0), (3, 4), (4, 3)):
        m.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="none", ec="black", lw=0.7))
    m.set_xticks([]); m.set_yticks([])
    for sp in m.spines.values():
        sp.set_linewidth(0.4)
    ax.text(11.5, 15.8, r"$|\rho_{ij}|$", ha="center", va="bottom", fontsize=6.2)
    rs = np.array([0.42, 0.18, 0.31, 0.12, 0.27, 0.08])
    v = sub(20.0, 16.5, 2.4, 13.0)
    v.imshow(rs[:, None], cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
    v.set_xticks([]); v.set_yticks([])
    for sp in v.spines.values():
        sp.set_linewidth(0.4)
    ax.text(21.2, 15.8, r"$|\rho_{iy}|$", ha="center", va="bottom", fontsize=6.2)
    ax.annotate("", xy=(29.5, 23.0), xytext=(23.6, 23.0),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#444444", mutation_scale=6))
    sv = [0, 1, 3, 5]
    for k, i in enumerate(sv):
        y = 14.5 + k * 3.2
        ax.add_patch(plt.Rectangle((31.0, y), 2.6, 2.6, fc=plt.cm.Blues(rs[i] / 0.5 * 0.8 + 0.1), ec="#164f86",
                                   lw=0.6))
        ax.text(34.3, y + 1.3, f"HI$_{i + 1}$", fontsize=5.9, va="center")
    for k, (i, host) in enumerate(((2, 0), (4, 3))):
        y = 15.5 + k * 6.0
        ax.add_patch(plt.Rectangle((44.0, y), 2.6, 2.6, fc="white", ec="#164f86", lw=0.6, ls=(0, (1.5, 1))))
        ax.text(47.2, y + 1.3, f"HI$_{i + 1}$", fontsize=5.9, va="center")
        ax.annotate("", xy=(38.6, 14.5 + sv.index(host) * 3.2 + 1.3), xytext=(43.7, y + 1.3),
                    arrowprops=dict(arrowstyle="-|>", lw=0.45, color="#777777", mutation_scale=4,
                                    connectionstyle="arc3,rad=0.25"))
    ax.text(29.0, 29.2, r"survivors HI$_\mathrm{sv}^{(s)}$", fontsize=5.5, va="center")
    ax.text(43.6, 29.2, r"attached HI$_\mathrm{at}^{(s)}$", fontsize=5.5, va="center")

    # ---- L2: group growth --------------------------------------------------------------------------
    step(3.0, 33.0, 2, r"Grow groups by partial correlation with SOH")
    ax.text(4.0, 38.3, r"survivors by $|\rho_{iy}|$", fontsize=5.9, va="center", color="#555555")
    order = [0, 1, 3, 5]
    for k, i in enumerate(order):
        y = 40.2 + k * 3.4
        ax.add_patch(plt.Rectangle((6.0, y), 2.6, 2.6, fc=plt.cm.Blues(rs[i] / 0.5 * 0.8 + 0.1), ec="#164f86",
                                   lw=0.6))
        ax.text(9.3, y + 1.3, f"HI$_{i + 1}$", fontsize=5.9, va="center")
    ax.annotate("", xy=(6.0, 53.5), xytext=(6.0, 41.0),
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color="#9a9a9a", mutation_scale=5))
    ax.text(5.0, 47.0, "rank", fontsize=5.3, rotation=90, ha="center", va="center", color="#777777")
    for gy, mem, lab in ((38.0, [0, 3, 5], "Group 1"), (45.0, [1], "Group 2")):
        ax.add_patch(FancyBboxPatch((25.0, gy), 21.0, 6.0, boxstyle="round,pad=0,rounding_size=1.0", fc="#fbf3ea",
                                    ec=C_LIN, lw=0.6))
        ax.text(46.6, gy + 3.0, lab, fontsize=5.7, va="center", ha="left", color=C_LIN, rotation=0)
        for k, i in enumerate(mem):
            ax.add_patch(plt.Rectangle((26.5 + k * 4.4, gy + 1.7), 2.6, 2.6, fc=plt.cm.Blues(rs[i] / 0.5 * 0.8 + 0.1),
                                       ec="#164f86", lw=0.6))
        for k in range(len(mem), 4):
            ax.add_patch(plt.Rectangle((26.5 + k * 4.4, gy + 1.7), 2.6, 2.6, fc="none", ec="#bdbdbd", lw=0.5,
                                       ls=(0, (1.2, 1))))
    for i_from, gy, ls in ((0, 38.0, "-"), (3, 38.0, "--"), (5, 38.0, "--"), (1, 45.0, "-")):
        y = 40.2 + order.index(i_from) * 3.4 + 1.3
        ax.annotate("", xy=(24.8, gy + 3.0), xytext=(13.5, y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color="#444444", ls=ls, mutation_scale=5))
    ax.text(25.0, 53.0, r"$j^\ast = \arg\max_j\, |\rho(x_j, y \mid \mathbf{x}_G)|$", fontsize=5.9, va="center")

    # ---- L3: partial correlation example --------------------------------------------------------
    step(3.0, 57.0, 3, "Credit the information a candidate adds")
    yv = rng.uniform(0.75, 1.0, 400)
    x1 = 0.35 * (yv - 0.875) / 0.07 + rng.normal(0, 1.0, 400)
    x2 = (yv - 0.875) / 0.07 - 0.9 * x1 + rng.normal(0, 0.8, 400)
    rx = x2 - np.polyval(np.polyfit(x1, x2, 1), x1)
    ry = yv - np.polyval(np.polyfit(x1, yv, 1), x1)
    for k, (xa, ya, lab, xl, yl) in enumerate((
            (x1, yv, f"seed: r = {np.corrcoef(x1, yv)[0, 1]:.2f}", r"$x_1$", "SOH"),
            (rx, ry, f"partial r = {np.corrcoef(rx, ry)[0, 1]:.2f}", r"$x_2 \mid x_1$", r"SOH $\mid x_1$"))):
        s = sub(9.0 + k * 28.0, 62.5, 15.0, 11.5)
        s.scatter(xa, ya, s=0.6, color="#5b8fc7", alpha=0.5, lw=0)
        s.plot(np.sort(xa), np.polyval(np.polyfit(xa, ya, 1), np.sort(xa)), color=C_LIN, lw=0.8)
        s.set_xticks([]); s.set_yticks([])
        s.set_xlabel(xl, fontsize=5.6, labelpad=0.5)
        s.set_ylabel(yl, fontsize=5.6, labelpad=0.5)
        s.set_title(lab, fontsize=5.7, pad=1.5)
        for sp in s.spines.values():
            sp.set_linewidth(0.4)
    ax.annotate("", xy=(35.5, 68.0), xytext=(25.5, 68.0),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#444444", mutation_scale=6))
    ax.text(30.5, 66.6, "+ $x_2$", fontsize=5.6, ha="center", va="bottom")

    # ---- R1: keep groups of two or more ---------------------------------------------------------
    step(58.4, 9.0, 1, r"Fuse groups with $|G| \geq 2$ only")
    for k, nm in enumerate((3, 2, 4)):
        gx = 61.0 + k * 10.0
        ax.add_patch(FancyBboxPatch((gx, 15.0), 8.4, 8.4, boxstyle="round,pad=0,rounding_size=1.0", fc="#eef4fa",
                                    ec=C_KER, lw=0.6))
        for q in range(nm):
            ax.add_patch(plt.Rectangle((gx + 1.0 + (q % 2) * 3.4, 16.0 + (q // 2) * 3.4), 2.6, 2.6,
                                       fc=plt.cm.Blues(0.35 + 0.12 * q), ec="#164f86", lw=0.5))
    ax.text(75.4, 25.2, "kernel candidates", fontsize=5.9, ha="center", va="top", color=C_KER)
    ax.add_patch(FancyBboxPatch((95.5, 15.0), 8.4, 8.4, boxstyle="round,pad=0,rounding_size=1.0", fc="#f3f3f3",
                                ec="#9a9a9a", lw=0.6, ls=(0, (1.5, 1))))
    ax.add_patch(plt.Rectangle((96.5, 16.0), 2.6, 2.6, fc=plt.cm.Blues(0.35), ec="#164f86", lw=0.5))
    ax.text(99.7, 25.2, "singleton:\nraw HI only", fontsize=5.6, ha="center", va="top", color="#777777",
            linespacing=1.0)

    # ---- R2: RBF on landmarks -------------------------------------------------------------------
    step(58.4, 33.0, 2, r"RBF map to $N_\mathrm{lm}$ landmarks (Nyström)")
    e = sub(60.5, 38.5, 20.0, 13.0)
    pts = rng.normal(0, 1, (500, 2)) * [1.0, 0.7]
    lm = pts[rng.choice(500, 26, replace=False)]
    new = np.array([0.35, 0.15])
    e.scatter(pts[:, 0], pts[:, 1], s=0.6, color="#b5b5b5", lw=0)
    e.scatter(lm[:, 0], lm[:, 1], s=5, marker="D", color=C_KER, lw=0)
    near = np.argsort(((lm - new) ** 2).sum(1))[:3]
    for k in near:
        e.plot([new[0], lm[k, 0]], [new[1], lm[k, 1]], color=C_LIN, lw=0.6)
    e.scatter(*new, s=24, marker="*", color=C_LIN, edgecolor="black", lw=0.3, zorder=5)
    e.set_xticks([]); e.set_yticks([])
    e.set_xlabel("member HI space of $G$", fontsize=5.6, labelpad=0.5)
    for sp in e.spines.values():
        sp.set_linewidth(0.4)
    hs = [Line2D([0], [0], marker="o", ls="", ms=1.8, color="#b5b5b5", label="training"),
          Line2D([0], [0], marker="D", ls="", ms=2.4, color=C_KER, label="landmark"),
          Line2D([0], [0], marker="*", ls="", ms=4.5, color=C_LIN, mec="black", mew=0.3, label="new segment")]
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(82.0 / WS, 1 - 38.0 / HS), frameon=False, fontsize=5.6,
              handletextpad=0.2, labelspacing=0.25, borderaxespad=0)
    ax.text(82.4, 49.5, r"$\phi_k = K(\mathbf{x}, \mathbf{z}_k)$", fontsize=6.0, va="center")
    ax.text(82.4, 52.8, r"$K = \exp(-\|\mathbf{x}-\mathbf{x}'\|^2 / 2\sigma_K^2)$", fontsize=5.6, va="center")

    # ---- R3: ridge to SOH -----------------------------------------------------------------------
    step(58.4, 57.0, 3, "Ridge regression to SOH")
    b = sub(61.0, 62.0, 18.0, 13.0)
    k_ = np.arange(14)
    phi = np.exp(-0.5 * ((k_ - 5.5) / 2.2) ** 2) * rng.uniform(0.5, 1.0, 14)
    w = rng.normal(0, 1, 14)
    b.bar(k_, phi * w, color=np.where(w > 0, C_KER, C_LIN), width=0.75)
    b.axhline(0, color="#555555", lw=0.4)
    b.set_xticks([]); b.set_yticks([])
    b.set_xlabel(r"landmark $k$", fontsize=5.6, labelpad=0.5)
    b.set_ylabel(r"$w_k \phi_k$", fontsize=5.6, labelpad=0.5)
    for sp in b.spines.values():
        sp.set_linewidth(0.4)
    ax.annotate("", xy=(87.5, 68.5), xytext=(80.5, 68.5),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#444444", mutation_scale=6))
    ax.text(84.0, 67.6, r"$\Sigma$", fontsize=6.6, ha="center", va="bottom")
    ax.add_patch(FancyBboxPatch((88.0, 65.6), 13.0, 5.8, boxstyle="round,pad=0,rounding_size=1.2", fc=C_KER, ec="none"))
    ax.text(94.5, 68.5, r"$h_\mathrm{ker}(\mathbf{x})$", fontsize=6.4, color="white", ha="center", va="center")
    ax.text(95.0, 75.5, r"$\mathbf{w}^\ast = (\Phi^\top\Phi + \lambda_\mathrm{ridge} I)^{-1}\Phi^\top \mathbf{y}$",
            fontsize=5.8, ha="center", va="center")
    ax.text(94.5, 62.4, r"kept if partial $r \geq 0.10$", fontsize=5.5, ha="center", va="center", color="#555555")

    # links to the quantitative panels
    for x, y, t, c in ((50.5, 77.0, "→ (d)", C_LIN), (103.5, 59.0, "→ (b)(c)", C_KER)):
        ax.text(x, y, t, fontsize=6.4, color="white", fontweight="bold", ha="center", va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.35", fc=c, ec="white", lw=0.6))
    return ax, HS

def panel_scatter(fig, rows):
    x0, y0, s = 125.5, 14.5, 36.0
    ax = ax_mm(fig, x0, y0, s, s)
    axt = ax_mm(fig, x0, y0 - 7.5, s, 7.0)
    axr = ax_mm(fig, x0 + s + 0.8, y0, 7.0, s)
    ax.fill_between([0, 1], [0, 1], [1, 1], color="#eef4fa", zorder=0)
    ax.plot([0, 1], [0, 1], color="#8a8a8a", lw=0.6, ls="--", zorder=1)
    for k in (0.2, 0.4):
        ax.plot([0, 1 - k], [k, 1], color="#c9d7e6", lw=0.45, ls=":", zorder=1)
        ax.text(0.98 - k, 0.995, f"+{k:.1f}", fontsize=5.5, color="#7d97b3", ha="right", va="top")
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
    ax.annotate("(d)", xy=(best(ex), ker(ex)), xytext=(0.04, 0.93), fontsize=6.7, fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="black", shrinkB=3))
    n_up = sum(ker(r) > best(r) for r in rows)
    ax.text(0.97, 0.05, f"K > B in {n_up}/{len(rows)}", ha="right", fontsize=6.7, color=C_KER,
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
    axt.text(0.0, 1.25, "(b)", transform=axt.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
    axt.text(0.155, 1.25, "Kernel HI and its best member", transform=axt.transAxes, fontsize=7.6,
             va="bottom")
    return ax




def half_violin(ax, data, pos, color, side=-1, width=0.42):
    kde_y = np.linspace(0, 1, 200)
    data = np.asarray(data)
    bw = 1.06 * data.std() * len(data) ** (-1 / 5) + 1e-3
    dens = np.exp(-0.5 * ((kde_y[:, None] - data[None, :]) / bw) ** 2).sum(1)
    dens = dens / dens.max() * width
    ax.fill_betweenx(kde_y, pos, pos + side * dens, color=color, alpha=0.55, lw=0, zorder=2)
    ax.plot(pos + side * dens, kde_y, color=color, lw=0.5, zorder=2)


def panel_ladder(fig, rows):
    ax = ax_mm(fig, 13.0, 99.0, 74.0, 38.0)
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
            ax.text(xs[k] + 0.17, med[k], f"{med[k]:.2f}", fontsize=5.6, va="center", zorder=6,
                    color="#222222")
        n_up = int((vals[:, 2] > vals[:, 1]).sum())
        ax.text(xs[1], 1.03, f"K > L  {n_up}/{len(vals)}", ha="center", va="bottom", fontsize=6.1,
                color=C_KER, fontweight="bold")
        ax.text(xs[1], -0.13, DS_LABEL[ds], ha="center", va="top", fontsize=6.9, color=DS_COLOR[ds],
                fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xticks([di * gap + k * 1.05 for di in range(3) for k in range(3)])
    ax.set_xticklabels(["B", "L", "K"] * 3, fontsize=6.4)
    ax.set_xlim(-0.75, 2 * gap + 2.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("|r| with SOH on test cells")
    tidy(ax)
    ax.grid(axis="x", visible=False)
    ax.text(-0.13, 1.12, "(c)", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
    ax.text(-0.04, 1.12, "Best member (B), linear fit (L) and kernel HI (K)", transform=ax.transAxes,
            fontsize=7.6, va="bottom")






def _cat(name):
    return name.split("_")[0]


def panel_example(fig, rows):
    ex = next(r for r in rows if (r["dataset"], r["kernel"]) == EXAMPLE)
    ax = ax_mm(fig, 97.0, 96.5, 50.0, 46.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.0, "(d)", fontsize=8.5, fontweight="bold", va="bottom")
    ax.text(0.105, 1.0, "Traced kernel HI of the MIT high charge scenario", fontsize=7.6, va="bottom")
    ax.text(0.0, 0.955, "growth order with partial r given earlier members", fontsize=5.8, color="#555555",
            va="top")
    n = len(ex["members"])
    ys = np.linspace(0.80, 0.40, n)
    xm, wb = 0.12, 0.60
    for i, (nm, rt) in enumerate(zip(ex["members"], ex["member_r_test"])):
        c = CAT_COLORS[_cat(nm)]
        ax.add_patch(FancyBboxPatch((xm, ys[i] - 0.05), wb, 0.10, boxstyle="round,pad=0.004,rounding_size=0.02",
                                    fc=c, ec="none", alpha=0.16))
        ax.add_patch(FancyBboxPatch((xm, ys[i] - 0.05), 0.016, 0.10, boxstyle="square,pad=0", fc=c, ec="none"))
        ax.text(xm + 0.035, ys[i] + 0.018, nm, fontsize=5.9, family="monospace", va="center")
        # |r| bar inside the box
        ax.add_patch(plt.Rectangle((xm + 0.035, ys[i] - 0.032), 0.5 * abs(rt), 0.018, fc=C_BEST, ec="none"))
        ax.text(xm + 0.045 + 0.5 * abs(rt), ys[i] - 0.023, f"|r| {abs(rt):.2f}", fontsize=5.6, va="center")
        ax.add_patch(FancyArrowPatch((xm + wb + 0.01, ys[i]), (0.86, 0.62), arrowstyle="-|>", mutation_scale=4,
                                     lw=0.55, color="#8a8a8a", connectionstyle="arc3,rad=-0.15"))
    for i in range(1, n):
        ax.add_patch(FancyArrowPatch((xm - 0.01, ys[i - 1] - 0.03), (xm - 0.01, ys[i] + 0.03), arrowstyle="-|>",
                                     mutation_scale=4.5, lw=0.6, color=C_LIN, connectionstyle="arc3,rad=0.6"))
        ax.text(xm - 0.085, (ys[i - 1] + ys[i]) / 2, f"{abs(ex['growth_scores'][i]):.2f}", fontsize=5.8,
                ha="center", va="center", color=C_LIN)
    ax.text(xm - 0.085, ys[0] + 0.065, "seed", fontsize=5.6, ha="center", color="#555555")
    ax.add_patch(FancyBboxPatch((0.80, 0.56), 0.18, 0.12, boxstyle="round,pad=0.006,rounding_size=0.03",
                                fc=C_KER, ec="none"))
    ax.text(0.89, 0.62, "kernel\nHI", color="white", ha="center", va="center", fontsize=6.7, fontweight="bold",
            linespacing=0.9)
    ax.text(0.89, 0.71, f"gate p {ex['gate_prob']:.2f}", ha="center", fontsize=5.8)
    # bar comparison
    bx = ax_mm(fig, 157.0, 106.0, 15.0, 26.0)
    labels = [f"m{i + 1}" for i in range(n)] + ["L", "K"]
    vals = [abs(v) for v in ex["member_r_test"]] + [lin(ex), ker(ex)]
    cols = [CAT_COLORS[_cat(m)] for m in ex["members"]] + [C_LIN, C_KER]
    bx.barh(range(len(vals))[::-1], vals, color=cols, height=0.7, alpha=0.9)
    for k, v in enumerate(vals):
        bx.text(v + 0.02, len(vals) - 1 - k, f"{v:.2f}", va="center", fontsize=5.6)
    bx.set_yticks(range(len(vals))[::-1])
    bx.set_yticklabels(labels, fontsize=5.8)
    bx.set_xlim(0, 1)
    bx.set_xlabel("|r| on test cells", fontsize=6.2, labelpad=1)
    bx.tick_params(axis="x", labelsize=5.8)
    bx.spines[["top", "right"]].set_visible(False)
    bx.axvline(max(vals[:n]), color=C_BEST, lw=0.6, ls=":")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, label=k) for k, c in CAT_COLORS.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(108.0 / W_MM, 1 - 140.5 / H_MM), ncol=4,
               frameon=False, handlelength=0.8, handletextpad=0.3, columnspacing=0.7, fontsize=6,
               title="HI category (d)", title_fontsize=5.4, alignment="left")


def legend_block(fig):
    h = [Line2D([0], [0], marker=DS_MARKER[d], ls="", mfc=DS_COLOR[d], mec=DS_COLOR[d], ms=3.6, label=DS_LABEL[d])
         for d in DATASETS]
    h += [Line2D([0], [0], marker="o", ls="", mfc="white", mec="black", ms=3.6, mew=0.6, label="not selected"),
          Line2D([0], [0], marker="o", ls="", mfc="black", mec="black", ms=3.6, label="selected (p ≥ 0.9)"),
          Line2D([0], [0], marker="x", ls="", color="#9a9a9a", ms=3.6, label="filtered out"),
          Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555555", ms=2.4, label="2 members"),
          Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555555", ms=4.2, label="4 members")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(119.0 / W_MM, 1 - 63.5 / H_MM), ncol=2,
               frameon=False, handletextpad=0.2, columnspacing=1.0, labelspacing=0.35, fontsize=6.3,
               title="Markers in (b)", title_fontsize=5.6, alignment="left")
    h2 = [plt.Rectangle((0, 0), 1, 1, fc=c, label=l) for c, l in
          ((C_BEST, "B  best single member"), (C_LIN, "L  least-squares fit of members"), (C_KER, "K  kernel HI"))]
    fig.legend(handles=h2, loc="upper left", bbox_to_anchor=(10.0 / W_MM, 1 - 144.0 / H_MM), ncol=3,
               frameon=False, handlelength=0.9, handletextpad=0.35, labelspacing=0.3, fontsize=6.3,
               title_fontsize=5.6, alignment="left")


def main():
    rows = json.loads(CACHE.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax_a, h_a = panel_schematic(fig)
    ax_b = panel_scatter(fig, rows)
    panel_ladder(fig, rows)
    panel_example(fig, rows)
    legend_block(fig)
    fig.add_artist(Line2D([4 / W_MM, 170 / W_MM], [1 - 89.0 / H_MM] * 2, color="#bdbdbd", lw=0.5, ls=(0, (3, 2))))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    print("saved", OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
