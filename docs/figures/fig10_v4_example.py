"""Figure 10 -- v4 example on two real cells (MIT+HUST pooled, LFP-only from here on).

From Figure 6 onward the paper's remaining results focus on the pooled
MIT+HUST (LFP) setting -- the canonical v4 configuration -- rather than
cross-chemistry comparison (that lives in Figure 5 now). TJU/NCM is
deliberately excluded here even though it appears elsewhere in the paper:
this run's checkpoint is MIT+HUST-only (Fig5 already owns the cross-chemistry
comparison with a visualization suited to independently-trained per-chemistry
models -- folding TJU into this pooled-LFP figure would contradict its own
caption/scope).

Representative cells: MIT b1c5, HUST 1-7 (both in this run's held-out test
split), hard routing (realistic/deployment condition). Two cells instead of
one strengthens the "representative behavior" claim (not a single-cell
cherry-pick) while staying inside the figure's stated pooled-MIT+HUST scope.

Layout: 4 rows (MIT charge / MIT discharge / HUST charge / HUST discharge)
x 2 columns:
  col 1: capacity curve, true vs per-scenario-level predicted
  col 2: relative error (%) per scenario level, vs cycle
(columns 2 and 3 of the original 2x3 diagnostic layout --
scr_evaluator.py::_plot_capacity_curves / capacity_curve_1-7.png -- dropped
the absolute-error-in-Ah column per user feedback, kept shape + %error.)

Real data: predictions/test_predictions_hard.csv from the v4 canonical run
(2026-09-13: test.py extended to export hard/soft mode
predictions, not just oracle).

Run: python docs/figures/fig6_v4_example.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import INK, SUBINK, setup_rcparams, label_panel, PROJECT_ROOT

setup_rcparams()

RUN_DIR = PROJECT_ROOT / "legacy_results/experiments/phase1_lab/results/p1v2_runs/0904_1708_p1v2_p1v4_minpts5_calib100_offA5mA_seed42"
EXAMPLE_CELLS = ["b1c5", "1-7"]   # (MIT, HUST) -- both confirmed present in this run's test split

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
    fig = plt.figure(figsize=(8.0, 13.0))
    gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.34,
                           top=0.96, bottom=0.045, left=0.10, right=0.97)

    letters = iter(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"])
    for row_pair, cell in enumerate(EXAMPLE_CELLS):
        df = load_cell(RUN_DIR, cell)
        cycles = np.sort(df.cycle.unique())
        chg_true = _true_line(df[df.seg_name.isin([s for s, _ in CHG_LEVELS])], cycles)
        dis_true = _true_line(df[df.seg_name.isin([s for s, _ in DIS_LEVELS])], cycles)

        r_chg, r_dis = 2 * row_pair, 2 * row_pair + 1
        ax_cc, ax_ce = fig.add_subplot(gs[r_chg, 0]), fig.add_subplot(gs[r_chg, 1])
        ax_dc, ax_de = fig.add_subplot(gs[r_dis, 0]), fig.add_subplot(gs[r_dis, 1])

        panel_curve(ax_cc, df, CHG_LEVELS, cycles, chg_true, f"Charge -- capacity, cell {cell} (hard)")
        panel_rel_error(ax_ce, df, CHG_LEVELS, cycles, chg_true, f"Charge -- relative error (%), cell {cell}")
        panel_curve(ax_dc, df, DIS_LEVELS, cycles, dis_true, f"Discharge -- capacity, cell {cell} (hard)")
        panel_rel_error(ax_de, df, DIS_LEVELS, cycles, dis_true, f"Discharge -- relative error (%), cell {cell}")

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
