"""Figure 4 -- (a) ablation RMSE 분포 + (b) 시너지 Level1/2 x 게이트확률 통합 산점도.

2026-10-12: 기존 fig12_ablation.py는 (a)RMSE/(b)MAPE 2패널이었음 -- PREWRITE_v1.md가
이미 "(b)MAPE 패널은 제거, (a)만 사용"으로 결정한 상태였는데, 그 빈 (b) 자리에
fig_synergy_level1_2_gated.py의 통합 산점도(Level1=선형 시너지, Level2=비선형 시너지,
마커=실제 게이트 선택확률)를 넣어 "결과(a, 빼면 나빠짐) -> 근거(b, 왜 좋은지)"가
한 Figure 안에서 바로 이어지게 함.

(a)는 fig12_ablation.py의 RMSE 패널 로직을 그대로 재사용(레거시 phase1_lab 비교 run
그대로 -- 이 ablation 자체를 다른 run으로 다시 돌리는 건 범위 밖, 패널 구성만 변경).

Run: python docs/figures/fig4_ablation_synergy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _style import INK, SUBINK, setup_rcparams, label_panel  # noqa: E402
from fig_synergy_level1_2_gated import build_points, plot_on_ax  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "legacy_results/experiments/phase1_lab/results/p1v2_runs"

CONDITIONS = [
    ("no scen", RUNS_DIR / "0911_0541_p1v2_p1v4_noscen_kernel_zonetile_seed42", "#3A6178"),
    ("v4 (w/o group)", RUNS_DIR / "0911_0026_p1v2_p1v4_label_kernel_noshare_seed42", "#A9863F"),
    ("v4 (w/o kernel)*", RUNS_DIR / "0909_1318_p1v2_p1v4_rawonly_seed42_seed42", "#96473A"),
    ("v4", RUNS_DIR / "0904_1708_p1v2_p1v4_minpts5_calib100_offA5mA_seed42", "#182E45"),
]


def per_cell_metrics(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "predictions" / "test_predictions.csv")
    rows = []
    for cell, g in df.groupby("cell_id"):
        err = g.cap_pred_Ah - g.cap_true_Ah
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rows.append((cell, rmse))
    return pd.DataFrame(rows, columns=["cell_id", "rmse"])


def panel_rmse(ax: plt.Axes, data_by_cond: dict[str, pd.DataFrame]) -> None:
    labels = [c[0] for c in CONDITIONS]
    colors = [c[2] for c in CONDITIONS]
    vals = [data_by_cond[label]["rmse"].values for label in labels]

    bp = ax.boxplot(vals, positions=range(len(labels)), widths=0.5, patch_artist=True,
                     showfliers=True, flierprops=dict(marker="o", ms=2.5, alpha=0.35,
                                                       markerfacecolor=SUBINK, markeredgewidth=0))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
        patch.set_edgecolor(INK)
    for med in bp["medians"]:
        med.set_color("white")
        med.set_linewidth(1.6)
    for whisk in bp["whiskers"] + bp["caps"]:
        whisk.set_color(SUBINK)

    rng = np.random.default_rng(11)
    for i, v in enumerate(vals):
        jitter = rng.normal(0, 0.045, size=len(v))
        ax.scatter(np.full(len(v), i) + jitter, v, s=6, color=colors[i], alpha=0.45,
                   linewidths=0, zorder=1)
        med = np.median(v)
        ax.text(i, med, f"  {med:.4f}", fontsize=6.2, color=SUBINK, va="bottom", ha="center")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.2, rotation=12, ha="right")
    ax.set_ylabel("Per-cell RMSE (Ah)")
    ax.set_title("Ablation: removing synergy/kernel hurts RMSE", loc="left", fontsize=9.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


def build_figure() -> plt.Figure:
    setup_rcparams()
    data_by_cond = {label: per_cell_metrics(d) for label, d, _ in CONDITIONS}
    points = build_points()

    fig = plt.figure(figsize=(10.6, 4.6))
    gs = fig.add_gridspec(1, 2, wspace=0.42, top=0.86, bottom=0.22, left=0.07, right=0.97)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_rmse(ax_a, data_by_cond)
    plot_on_ax(fig, ax_b, points, title_fontsize=8.6)

    fig.text(0.5, 0.015, "*rawonly also lacks the raw-HI shared/specific group split "
             "(interaction_json unset), so it removes kernel AND group at once, not kernel alone.",
             ha="center", va="bottom", fontsize=6.2, color=SUBINK, style="italic")

    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)")):
        label_panel(fig, ax, letter, dx=-0.045, dy=0.016)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = Path(__file__).resolve().parent / "fig4_ablation_synergy.png"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"saved: {out_png}\nsaved: {out_png.with_suffix('.pdf')}")
