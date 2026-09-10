"""Figure 3 -- Information expansion via kernel fusion (D9 + D11, +D10 inset).

(a) candidate-feature pool size per scenario: 64 raw + kernel-fused groups added
(b) representative kernel groups: composition + train R^2 (top-3 by R^2)
(c) predictive-power distribution: raw univariate r^2 vs kernel group train R^2

Real data: 5_model/experiments/phase1_lab/results/kernel_group_features_k25_full_N2_kernel_v3.pkl
(synergy-grouped kernel features, Nystroem+Ridge) and the same HI cache used
in Figure 2 for the raw univariate baseline.

Run: python docs/figures/fig3_kernel_expansion.py
"""

from __future__ import annotations

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import (INK, SUBINK, FAINT, GRID, SCEN_NAMES, SCEN_POS_COLOR, setup_rcparams,
                     label_panel, PROJECT_ROOT, DATA_4_HI_ROOT)

warnings.filterwarnings("ignore", category=UserWarning)
setup_rcparams()

KERNEL_PKL = PROJECT_ROOT / "5_model/experiments/phase1_lab/results/kernel_group_features_k25_full_N2_kernel_v3.pkl"
GATE_JSON = (PROJECT_ROOT / "5_model/experiments/phase1_lab/results/p1v2_runs/"
             "0827_1705_p1v2_p1v4_full_seed42/gates/regression_HIs.json")
HI_CACHE = DATA_4_HI_ROOT.parent / "4_hi_analysis" / (
    "hi_features_qfref_n1-35%_n2-20%_N-2_minpts5_lag-1_noise-3%_ou-200_calib-100_offA-5mA.pkl")

RAW_COLOR = "#7C93A3"
KERNEL_COLOR = "#B14A3C"


def load_kernel_features():
    d = pickle.load(open(KERNEL_PKL, "rb"))
    return d["features"]


def load_raw_r2():
    gate = json.load(open(GATE_JSON))
    base_names = sorted(set(n.replace("_chg_lo", "") for n in gate["seg_0_names"]))
    df = pd.read_pickle(HI_CACHE)
    first_cap = df.sort_values("cycle").groupby("cell_id").capacity_Ah.transform("first")
    soh = df.capacity_Ah / first_cap
    r2s = []
    for n in base_names:
        for s in SCEN_NAMES:
            col = f"{n}_{s}"
            if col in df.columns:
                r = df[col].corr(soh)
                if np.isfinite(r):
                    r2s.append(r ** 2)
    return np.array(r2s)


def panel_pool_size(ax, feats):
    from collections import Counter
    counts = Counter(f["scenario"] for f in feats)
    kernel_n = [counts.get(s, 0) for s in SCEN_NAMES]
    raw_n = [64] * 6
    x = np.arange(6)
    ax.bar(x, raw_n, color=RAW_COLOR, width=0.55, label="raw HI (fixed pool)", zorder=3)
    ax.bar(x, kernel_n, bottom=raw_n, color=KERNEL_COLOR, width=0.55,
           label="kernel-fused HI (added)", zorder=3)
    for xi, (r, k) in zip(x, zip(raw_n, kernel_n)):
        ax.text(xi, r + k + 1.5, f"{r+k}", ha="center", va="bottom", fontsize=7.2,
                color=INK, fontweight="bold")
        ax.text(xi, r / 2, f"{r}", ha="center", va="center", fontsize=6.4, color="white")
        ax.text(xi, r + k / 2, f"+{k}", ha="center", va="center", fontsize=6.4, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(SCEN_NAMES, rotation=40, ha="right", fontsize=6.8)
    ax.set_ylabel("candidate feature count")
    ax.set_ylim(0, 88)
    ax.set_title("Candidate pool expansion per scenario", loc="left", pad=6, fontsize=8.0)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=6.4, handlelength=1.0,
                     borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(SUBINK)


def panel_examples(ax, feats):
    top = sorted(feats, key=lambda f: -f["train_r2"])[:3]
    ys = np.arange(len(top))[::-1]
    for y, f in zip(ys, top):
        color = dict(zip(SCEN_NAMES, SCEN_POS_COLOR))[f["scenario"]]
        ax.barh(y, f["train_r2"], color=color, height=0.5, zorder=3)
        ax.text(f["train_r2"] + 0.015, y, f"$R^2$={f['train_r2']:.3f}", va="center",
                fontsize=7.0, color=INK, fontweight="bold")
        members = "\n".join(m.replace(f"_{f['scenario']}", "") for m in f["member_names"])
        ax.text(0.01, y - 0.40, f"[{f['scenario']}]  " + " + ".join(
            m.replace(f"_{f['scenario']}", "") for m in f["member_names"]),
            va="top", ha="left", fontsize=6.0, color=SUBINK, style="italic", wrap=True)
    ax.set_yticks(ys)
    ax.set_yticklabels([f["name"] for f in top], fontsize=7.4)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.15, len(top) - 0.3)
    ax.set_xlabel("train $R^2$ (Nystroem-kernel ridge)")
    ax.set_title("Top-3 kernel groups by predictive power", loc="left", pad=6, fontsize=8.0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)


def panel_boxplot(ax, raw_r2, kernel_r2):
    bp = ax.boxplot([raw_r2, kernel_r2], positions=[0, 1], widths=0.45, patch_artist=True,
                     showfliers=True, flierprops=dict(marker="o", ms=2.5, alpha=0.35,
                                                       markerfacecolor=SUBINK, markeredgewidth=0))
    for patch, color in zip(bp["boxes"], [RAW_COLOR, KERNEL_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(INK)
    for med in bp["medians"]:
        med.set_color("white")
        med.set_linewidth(1.6)
    for whisk in bp["whiskers"] + bp["caps"]:
        whisk.set_color(SUBINK)

    rng = np.random.default_rng(7)
    for i, vals in enumerate([raw_r2, kernel_r2]):
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=3.5,
                   color=(RAW_COLOR if i == 0 else KERNEL_COLOR), alpha=0.25, linewidths=0,
                   zorder=1)

    ax.text(0, np.median(raw_r2), f"  median={np.median(raw_r2):.3f}", fontsize=6.4,
            color=SUBINK, va="bottom", ha="center")
    ax.text(1, np.median(kernel_r2), f"  median={np.median(kernel_r2):.3f}", fontsize=6.4,
            color=SUBINK, va="bottom", ha="center")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"raw HI\n(univariate $r^2$, n={len(raw_r2)})",
                         f"kernel group\n(ridge $R^2$, n={len(kernel_r2)})"], fontsize=7.0)
    ax.set_ylabel("predictive power ($R^2$ vs SOH)")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Kernel fusion strengthens the signal", loc="left", pad=6, fontsize=8.0)


def build_figure():
    feats = load_kernel_features()
    raw_r2 = load_raw_r2()
    kernel_r2 = np.array([f["train_r2"] for f in feats])

    fig = plt.figure(figsize=(11.0, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.9], wspace=0.5,
                           top=0.86, bottom=0.20, left=0.07, right=0.975)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_pool_size(ax_a, feats)
    panel_examples(ax_b, feats)
    panel_boxplot(ax_c, raw_r2, kernel_r2)

    for ax in (ax_a, ax_c):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=2.8)

    for ax, letter, dx in ((ax_a, "(a)", -0.055), (ax_b, "(b)", -0.075), (ax_c, "(c)", -0.06)):
        label_panel(fig, ax, letter, dx=dx, dy=0.014)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig3_kernel_expansion.png"
    out_pdf = "docs/figures/fig3_kernel_expansion.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
