"""Figure 6 -- v4 example on a real cell (MIT+HUST pooled, LFP-only from here on).

From Figure 6 onward the paper's remaining results focus on the pooled
MIT+HUST (LFP) setting -- the canonical v4 configuration -- rather than
cross-chemistry comparison (that lives in Figure 5 now).

Representative cell: HUST 1-7, hard routing (realistic/deployment condition).
Layout: 2 rows (Charge / Discharge) x 2 columns:
  col 1: capacity curve, true vs per-scenario-level predicted
  col 2: relative error (%) per scenario level, vs cycle
(columns 2 and 3 of the original 2x3 diagnostic layout --
scr_evaluator.py::_plot_capacity_curves / capacity_curve_1-7.png -- dropped
the absolute-error-in-Ah column per user feedback, kept shape + %error.)

Real data: predictions/test_predictions_hard.csv from the v4 canonical run
(2026-09-13: test_phase1_checkpoint.py extended to export hard/soft mode
predictions, not just oracle).

Run: python docs/figures/fig6_v4_example.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import INK, SUBINK, setup_rcparams, label_panel, PROJECT_ROOT

setup_rcparams()

RUN_DIR = PROJECT_ROOT / "5_model/experiments/phase1_lab/results/p1v2_runs/0904_1708_p1v2_p1v4_minpts5_calib100_offA5mA_seed42"
EXAMPLE_CELL = "1-7"

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
    df = load_cell(RUN_DIR, EXAMPLE_CELL)
    cycles = np.sort(df.cycle.unique())

    fig = plt.figure(figsize=(8.0, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.34,
                           top=0.92, bottom=0.09, left=0.10, right=0.97)
    ax_cc, ax_ce = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_dc, ax_de = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    chg_true = _true_line(df[df.seg_name.isin([s for s, _ in CHG_LEVELS])], cycles)
    dis_true = _true_line(df[df.seg_name.isin([s for s, _ in DIS_LEVELS])], cycles)

    panel_curve(ax_cc, df, CHG_LEVELS, cycles, chg_true, f"Charge -- capacity, cell {EXAMPLE_CELL} (hard)")
    panel_rel_error(ax_ce, df, CHG_LEVELS, cycles, chg_true, "Charge -- relative error (%)")
    panel_curve(ax_dc, df, DIS_LEVELS, cycles, dis_true, f"Discharge -- capacity, cell {EXAMPLE_CELL} (hard)")
    panel_rel_error(ax_de, df, DIS_LEVELS, cycles, dis_true, "Discharge -- relative error (%)")

    for ax, letter in ((ax_cc, "(a)"), (ax_ce, "(b)"), (ax_dc, "(c)"), (ax_de, "(d)")):
        label_panel(fig, ax, letter, dx=-0.06, dy=0.014)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig6_v4_example.png"
    out_pdf = "docs/figures/fig6_v4_example.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
