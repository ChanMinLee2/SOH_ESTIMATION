"""
docs/figures/fig7_synergy.py — Figure 7: 시너지 그룹과 커널 융합이 raw HI에 더하는 정보.

2026-11-04: 메인 단일 데이터셋 run 3개(MIT 1031_0117 / HUST 1031_0431 / TJU 1031_0938)의
크기 2 이상 synergy 그룹 132개(=커널 후보)를 대상으로 한다. 데이터는
fig7_synergy_cache.py가 만든 _cache/fig7_synergy_cache.json(학습 환경에서 실행)에서 읽는다.
모든 상관은 test 셀 세그먼트 기준(그룹 성장 단계의 편상관만 train 기준, synergy json 값).

(a) 그룹마다 x = 멤버 raw HI 중 최대 |r(HI, SOH)|, y = 커널 HI의 |r(kernel, SOH)|.
    대각선 위 = 융합이 가장 좋은 단일 멤버보다 SOH를 더 잘 설명. 색 = 시나리오,
    모양 = 데이터셋, 채움 = 게이트 선택(p >= 0.9), x 표시 = raw-conditioned 필터 탈락.
(b) 같은 그룹들의 최대 단일 멤버 -> 멤버 선형결합(최소제곱) -> 커널 HI 사다리(데이터셋별 중앙값과 IQR).
(c) 되짚기 예시: MIT chg_hi 그룹 g8 — 멤버 이름, 개별 |r|, 성장 순서의 편상관, 융합 결과.

실행: python docs/figures/fig7_synergy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache" / "fig7_synergy_cache.json"
OUT = HERE / "fig7_synergy"

DATASETS = ["MIT", "HUST", "TJU"]
DS_LABEL = {"MIT": "MIT (LFP)", "HUST": "HUST (LFP)", "TJU": "TJU (NCM)"}
DS_MARKER = {"MIT": "o", "HUST": "s", "TJU": "^"}
SCEN_COLORS = {
    "chg_lo": "#1f77b4", "chg_mid": "#2ca02c", "chg_hi": "#9467bd",
    "dis_hi": "#d62728", "dis_mid": "#ff7f0e", "dis_lo": "#8c564b",
}
CAT_COLORS = {"stat": "#1f77b4", "diff": "#ff7f0e", "lfp": "#2ca02c", "morph": "#9467bd"}
KERNEL_COLOR = "#444444"
EXAMPLE = ("MIT", "kernel_chg_hi_g8")
SEL_THR = 0.9

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})


def _best_member(r: dict) -> float:
    return max(abs(v) for v in r["member_r_test"])


def panel_scatter(ax, rows: list[dict]) -> None:
    ax.fill_between([0, 1], [0, 1], [1, 1], color="#f3f3f3", zorder=0)
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.7, ls="--", zorder=1)
    ax.text(0.99, 0.355, "shaded area:\nkernel > best member", fontsize=5.8, color="#555555",
            ha="right", va="bottom")
    for r in rows:
        x, y = _best_member(r), abs(r["kernel_r_test"])
        c = SCEN_COLORS[r["scenario"]]
        if r["status"] != "kept":
            ax.scatter(x, y, marker="x", s=12, color="#999999", lw=0.7, zorder=2)
            continue
        sel = (r["gate_prob"] or 0.0) >= SEL_THR
        ax.scatter(x, y, marker=DS_MARKER[r["dataset"]], s=22 if sel else 13,
                   facecolor=c if sel else "white", edgecolor=c, lw=0.7, alpha=0.9, zorder=3)
    ex = next(r for r in rows if (r["dataset"], r["kernel"]) == EXAMPLE)
    ex_xy = (_best_member(ex), abs(ex["kernel_r_test"]))
    ax.scatter(*ex_xy, s=95, facecolor="none", edgecolor="black", lw=0.9, zorder=5)
    ax.annotate("example (c)", xy=ex_xy, xytext=(0.30, 0.985), fontsize=6.0, va="top",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="black", shrinkB=4), zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Best single member |r| with SOH")
    ax.set_ylabel("Kernel HI |r| with SOH")
    ax.set_title("(a) Kernel HI vs best member", loc="left", fontweight="bold")
    handles = [Line2D([0], [0], marker=DS_MARKER[d], ls="", color="black", mfc="black", ms=4, label=d)
               for d in DATASETS]
    handles += [Line2D([0], [0], marker="o", ls="", color="black", mfc="white", ms=4, label="not selected"),
                Line2D([0], [0], marker="x", ls="", color="#999999", ms=4, label="filtered out")]
    ax.legend(handles=handles, loc="lower right", frameon=False, handletextpad=0.2,
              borderaxespad=0.1, labelspacing=0.2)
    ax.spines[["top", "right"]].set_visible(False)


LEVEL_COLORS = ["#9e9e9e", "#5f8fbf", KERNEL_COLOR]
LEVEL_LABELS = ["best single member", "linear fit of members", "kernel HI"]


def panel_ladder(ax, rows: list[dict]) -> None:
    gap = 3.6
    for di, ds in enumerate(DATASETS):
        rs = [r for r in rows if r["dataset"] == ds and r["status"] == "kept"]
        vals = np.array([[_best_member(r), abs(r["linear_r_test"]), abs(r["kernel_r_test"])] for r in rs])
        xs = di * gap + np.arange(3)
        for v in vals:
            ax.plot(xs, v, color="#c8c8c8", lw=0.35, alpha=0.7, zorder=1)
        med = np.median(vals, axis=0)
        q1, q3 = np.percentile(vals, 25, axis=0), np.percentile(vals, 75, axis=0)
        ax.vlines(xs, q1, q3, color=LEVEL_COLORS, lw=2.2, zorder=3)
        ax.plot(xs, med, color="black", lw=0.9, zorder=3)
        ax.scatter(xs, med, marker=DS_MARKER[ds], s=26, color=LEVEL_COLORS, edgecolor="black",
                   lw=0.5, zorder=4)
        n_up = int((vals[:, 2] > vals[:, 1]).sum())
        ax.text(xs[1], 0.995, f"K > L\n{n_up}/{len(vals)}", ha="center", va="top", fontsize=5.9)
    ax.set_xticks([di * gap + 1 for di in range(3)])
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], fontsize=6.3)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 2 * gap + 2.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("|r| with SOH (test cells)")
    ax.set_title("(b) From single HIs to the fused HI", loc="left", fontweight="bold")
    handles = [Line2D([0], [0], marker="s", ls="", mfc=c, mec="black", mew=0.4, ms=4, label=l)
               for c, l in zip(LEVEL_COLORS, ["B  best member", "L  linear fit", "K  kernel HI"])]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.6, fontsize=5.9)
    ax.spines[["top", "right"]].set_visible(False)


def _cat(name: str) -> str:
    return name.split("_")[0]


def panel_example(ax, rows: list[dict]) -> None:
    ex = next(r for r in rows if (r["dataset"], r["kernel"]) == EXAMPLE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(c) Traceable kernel HI (MIT chg hi)", loc="left", fontweight="bold")
    ax.text(0.0, 1.0, "arrows: growth order, numbers: partial r with SOH\ngiven the members before",
            fontsize=5.6, color="#555555", va="top")

    n = len(ex["members"])
    ys = np.linspace(0.76, 0.16, n)
    xm, wbox, xk = 0.09, 0.56, 0.86
    for i, (nm, r_te) in enumerate(zip(ex["members"], ex["member_r_test"])):
        ax.add_patch(FancyBboxPatch((xm, ys[i] - 0.06), wbox, 0.12,
                                    boxstyle="round,pad=0.004,rounding_size=0.02",
                                    fc=CAT_COLORS[_cat(nm)], ec="none", alpha=0.18))
        ax.add_patch(FancyBboxPatch((xm, ys[i] - 0.06), 0.014, 0.12, boxstyle="square,pad=0",
                                    fc=CAT_COLORS[_cat(nm)], ec="none"))
        ax.text(xm + 0.03, ys[i] + 0.02, f"{i + 1}  {nm}", fontsize=5.9, family="monospace", va="center")
        ax.text(xm + 0.03, ys[i] - 0.027, f"   |r| = {abs(r_te):.2f}", fontsize=5.9, va="center",
                color="#333333", family="monospace")
        ax.add_patch(FancyArrowPatch((xm + wbox + 0.005, ys[i]), (xk - 0.10, 0.46), arrowstyle="-|>",
                                     mutation_scale=5, lw=0.6, color="#777777", zorder=1))
    for i in range(1, n):
        ax.add_patch(FancyArrowPatch((xm - 0.008, ys[i - 1] - 0.035), (xm - 0.008, ys[i] + 0.035),
                                     arrowstyle="-|>", mutation_scale=5, lw=0.6, color="black",
                                     connectionstyle="arc3,rad=0.55"))
        ax.text(xm - 0.07, (ys[i - 1] + ys[i]) / 2, f"{abs(ex['growth_scores'][i]):.2f}",
                fontsize=5.9, ha="center", va="center")

    ax.add_patch(FancyBboxPatch((xk - 0.10, 0.39), 0.20, 0.14, boxstyle="round,pad=0.006,rounding_size=0.03",
                                fc=KERNEL_COLOR, ec="none"))
    ax.text(xk, 0.46, "kernel\nHI", color="white", ha="center", va="center", fontsize=6.2,
            fontweight="bold", linespacing=0.95)
    ax.text(xk, 0.36, f"|r| = {abs(ex['kernel_r_test']):.2f}", ha="center", va="top", fontsize=6.8,
            fontweight="bold")
    ax.text(xk, 0.26, f"linear fit\n|r| ={abs(ex['linear_r_test']):.2f}", ha="center", va="top",
            fontsize=5.9, color="#5f8fbf")
    ax.text(xk, 0.56, f"gate p\n={ex['gate_prob']:.2f}", ha="center", va="bottom", fontsize=5.9)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=CAT_COLORS[c], label=c) for c in CAT_COLORS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.06), ncol=4, frameon=False,
              handlelength=0.8, handletextpad=0.3, columnspacing=0.8, fontsize=5.9)


def main() -> None:
    rows = json.loads(CACHE.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(174 / 25.4, 74 / 25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.3], wspace=0.30,
                          left=0.065, right=0.995, bottom=0.17, top=0.82)
    panel_scatter(fig.add_subplot(gs[0]), rows)
    panel_ladder(fig.add_subplot(gs[1]), rows)
    panel_example(fig.add_subplot(gs[2]), rows)
    scen_handles = [Line2D([0], [0], marker="o", ls="", color=c, ms=4, label=s.replace("_", " "))
                    for s, c in SCEN_COLORS.items()]
    fig.legend(handles=scen_handles, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=6, frameon=False,
               handletextpad=0.1, columnspacing=0.9, title="Scenario (a)", title_fontsize=6.2,
               alignment="left")
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    print(f"saved {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
