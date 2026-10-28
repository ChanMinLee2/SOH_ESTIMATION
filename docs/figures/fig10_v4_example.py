"""Figure 6 (PREWRITE_v4 numbering) -- v4 example, now LFP (HUST 1-7) + NCM
(TJU) side by side (2026-10-16, PREWRITE_v4 요구사항4).

2026-10-12 당시엔 TJU/NCM을 의도적으로 뺐었음(독립 그림이던 구 Figure 10/
fig8_single_dataset.py가 교차화학종 비교를 전담했었기 때문). PREWRITE_v4에서
그 구 Figure 10을 삭제하고 그 역할을 Fig5/Fig6 서브플랏으로 흡수하기로
하면서, 이 그림도 TJU 대표 셀 궤적을 추가함.

Representative cells: HUST 1-7 (LFP, pooled MIT+HUST run) + TJU CY45-05_1-#24
(NCM, TJU-only run -- 그 run의 test split 중 사이클 수가 가장 많은 셀, hard
routing). 두 셀은 서로 다른 체크포인트(화학종별 독립 학습)에서 나온 예측이라
각자 자기 run_dir에서 로드.

Layout: 셀마다 2 rows (charge / discharge) x 2 columns:
  col 1: capacity curve, true vs per-scenario-level predicted
  col 2: relative error (%) per scenario level, vs cycle
(columns 2 and 3 of the original 2x3 diagnostic layout --
scr_evaluator.py::_plot_capacity_curves / capacity_curve_1-7.png -- dropped
the absolute-error-in-Ah column per user feedback, kept shape + %error.)

Real data: predictions/test_predictions_hard.csv from each cell's own run
(2026-09-13: test.py extended to export hard/soft mode predictions, not just
oracle).

Run: python docs/figures/fig10_v4_example.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import INK, SUBINK, setup_rcparams, label_panel, PROJECT_ROOT

setup_rcparams()

_LFP_RUN_DIR = PROJECT_ROOT / "legacy_results/experiments/phase1_lab/results/p1v2_runs/0904_1708_p1v2_p1v4_minpts5_calib100_offA5mA_seed42"
_TJU_RUN_DIR = PROJECT_ROOT / "legacy_results/experiments/phase1_lab/results/p1v2_runs/0910_1533_p1v2_p1v4_tju_only_seed42"

# (run_dir, cell_id, chemistry label) -- 각 셀이 자기 체크포인트의 run_dir을 따로 가짐
EXAMPLE_CELLS = [
    (_LFP_RUN_DIR, "1-7", "LFP"),              # HUST -- confirmed present in this run's test split
    (_TJU_RUN_DIR, "CY45-05_1-#24", "NCM"),    # TJU -- 가장 사이클 수 많은 테스트 셀
]

CHG_LEVELS = [("chg_lo", "lo"), ("chg_mid", "mid"), ("chg_hi", "hi")]
DIS_LEVELS = [("dis_hi", "hi"), ("dis_mid", "mid"), ("dis_lo", "lo")]
LEVEL_COLOR = {"lo": "#3A6178", "mid": "#A9863F", "hi": "#96473A"}
LEVEL_LS = {"lo": "-", "mid": (0, (4, 1.5)), "hi": (0, (1.5, 1.2))}


def load_cell(run_dir, cell):
    df = pd.read_csv(run_dir / "predictions" / "test_predictions_hard.csv")
    return df[df.cell_id == cell]


def _true_line(df, cycles):
    g = df.groupby("cycle").cap_true_Ah.mean()
    return np.array([g.get(c, np.nan) for c in cycles])


def panel_curve(ax, df, seg_levels, cycles, true_line, title):
    ax.plot(cycles, true_line, color=INK, lw=1.5, label="True", zorder=5)
    for seg, lv in seg_levels:
        sub = df[df.seg_name == seg]
        if len(sub) == 0:
            continue
        g = sub.groupby("cycle").cap_pred_Ah.mean()
        line = np.array([g.get(c, np.nan) for c in cycles])
        ax.plot(cycles, line, color=LEVEL_COLOR[lv], lw=1.1, ls=LEVEL_LS[lv],
                label=f"Pred-{lv}", zorder=4)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title(title, loc="left", pad=6, fontsize=8.0)
    leg = ax.legend(loc="best", frameon=False, fontsize=6.4, handlelength=1.6, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(SUBINK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


def panel_rel_error(ax, df, seg_levels, cycles, true_line, title):
    for seg, lv in seg_levels:
        sub = df[df.seg_name == seg]
        if len(sub) == 0:
            continue
        g = sub.groupby("cycle").cap_pred_Ah.mean()
        line = np.array([g.get(c, np.nan) for c in cycles])
        rel = np.abs(line - true_line) / true_line * 100
        ax.plot(cycles, rel, color=LEVEL_COLOR[lv], lw=1.1, ls=LEVEL_LS[lv],
                label=lv, zorder=4)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(title, loc="left", pad=6, fontsize=8.0)
    leg = ax.legend(loc="best", frameon=False, fontsize=6.4, handlelength=1.6, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(SUBINK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


def build_figure():
    n_cells = len(EXAMPLE_CELLS)
    fig = plt.figure(figsize=(8.0, 3.4 * n_cells))
    gs = fig.add_gridspec(2 * n_cells, 2, hspace=0.65, wspace=0.34,
                           top=0.95, bottom=0.06, left=0.10, right=0.97)

    letters = iter(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"])
    for row_pair, (run_dir, cell, chem) in enumerate(EXAMPLE_CELLS):
        df = load_cell(run_dir, cell)
        cycles = np.sort(df.cycle.unique())
        chg_true = _true_line(df[df.seg_name.isin([s for s, _ in CHG_LEVELS])], cycles)
        dis_true = _true_line(df[df.seg_name.isin([s for s, _ in DIS_LEVELS])], cycles)

        r_chg, r_dis = 2 * row_pair, 2 * row_pair + 1
        ax_cc, ax_ce = fig.add_subplot(gs[r_chg, 0]), fig.add_subplot(gs[r_chg, 1])
        ax_dc, ax_de = fig.add_subplot(gs[r_dis, 0]), fig.add_subplot(gs[r_dis, 1])

        cell_label = f"{cell} ({chem})"
        panel_curve(ax_cc, df, CHG_LEVELS, cycles, chg_true, f"Charge -- capacity, cell {cell_label} (hard)")
        panel_rel_error(ax_ce, df, CHG_LEVELS, cycles, chg_true, f"Charge -- relative error (%), cell {cell_label}")
        panel_curve(ax_dc, df, DIS_LEVELS, cycles, dis_true, f"Discharge -- capacity, cell {cell_label} (hard)")
        panel_rel_error(ax_de, df, DIS_LEVELS, cycles, dis_true, f"Discharge -- relative error (%), cell {cell_label}")

        for ax in (ax_cc, ax_ce, ax_dc, ax_de):
            label_panel(fig, ax, next(letters), dx=-0.06, dy=0.014)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig10_v4_example.png"
    out_pdf = "docs/figures/fig10_v4_example.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
