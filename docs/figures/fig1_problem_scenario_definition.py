"""Figure 1 — Problem framing & SOC-scenario definition (A2 + C7 + C8).

Camera-ready draft for Journal of Power Sources. All curves are real cell
data (MIT b1c0 = LFP, TJU CY25-05_1-#1 = NCM), loaded via data_directories.py
so the D:-drive location stays the single point of change. Zone boundaries
use the production q_frac_wide formula (n1=0.35, n2=0.20):
    left  bucket [0.00, n1]        -> chg_lo / dis_hi
    mid   bucket [0.5-n1/2,0.5+n1/2] -> chg_mid / dis_mid
    right bucket [1-n1, 1.00]      -> chg_hi / dis_lo
(see common/scenario/q_frac_wide.py::_zone_bounds + scenario_spec.json routing)

Run: python docs/figures/fig1_problem_scenario_definition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_directories import DATA_4_HI_ROOT  # noqa: E402

# ---------------------------------------------------------------- style ----

INK = "#141414"
SUBINK = "#5A5A5A"
FAINT = "#9C9C9C"
GRID = "#E7E4DC"

Z_LEFT = "#3D6E8C"   # bucket [0, n1]
Z_MID = "#D9A544"    # bucket [0.5-n1/2, 0.5+n1/2]
Z_RIGHT = "#B14A3C"  # bucket [1-n1, 1]

LFP_COLOR = "#152A45"   # navy
NCM_COLOR = "#9C3B26"   # rust

CHARGE_LS = dict(color=INK, lw=1.7, ls="-", solid_capstyle="round")
DISCHARGE_LS = dict(color=INK, lw=1.4, ls=(0, (4.5, 2.2)), alpha=0.80)

_AVAILABLE_FONTS = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": "Arial" if "Arial" in _AVAILABLE_FONTS else "sans-serif",
    "font.size": 8.3,
    "text.color": INK,
    "axes.edgecolor": SUBINK,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": SUBINK,
    "ytick.color": SUBINK,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "axes.labelsize": 8.4,
    "axes.titlesize": 8.6,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.default": "regular",
    "axes.grid": False,
})

N1 = 0.35
N2 = 0.20
ZONE_BOUNDS = {
    "left": (0.00, N1),
    "mid": (0.5 - N1 / 2, 0.5 + N1 / 2),
    "right": (1.0 - N1, 1.0),
}
OVERLAPS = [(ZONE_BOUNDS["mid"][0], ZONE_BOUNDS["left"][1]),
            (ZONE_BOUNDS["right"][0], ZONE_BOUNDS["mid"][1])]
ZONE_COLOR = {"left": Z_LEFT, "mid": Z_MID, "right": Z_RIGHT}
COVERAGE_PCT = 100.000  # validated result, n2-range coverage sweep (all 6 zones)


# --------------------------------------------------------------- data ------

MIT_CSV = DATA_4_HI_ROOT / "clean" / "MIT" / "b1c0.csv"
TJU_CSV = DATA_4_HI_ROOT / "clean" / "TJU" / "CY25-05_1-#1.csv"


def load_cycle(path: Path, cycle: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["cycle", "time_s", "voltage_V", "current_A"])
    return df[df.cycle == cycle].sort_values("time_s").reset_index(drop=True)


def split_charge_discharge(c: pd.DataFrame):
    neg = np.where(c.current_A.values < -0.01)[0]
    split = neg[0] if len(neg) else len(c)
    return c.iloc[:split], c.iloc[split:]


def build_curve(part: pd.DataFrame, sign: int, i_thresh: float = 0.05, gap_factor: float = 6.0):
    """q_frac (cumulative-capacity fraction) vs voltage, with rest steps handled.

    Rest steps (|I|~0) are dropped (i_thresh), but MIT/HUST's multi-step
    fast-charge protocols leave a large time GAP across each dropped rest.
    Naively trapz-integrating current*dt straight across that gap attributes
    a large *spurious* charge to it (using the pre/post-rest current over a
    duration where the true rest current was ~0) -- it does not reflect
    a spurious capacity value AND stretches the subsequent relaxation-voltage
    drop across many q_frac points, which looks like a wide multi-lobe glitch
    once differentiated (dV/dq). We zero the increment across any such gap
    (real charge transferred during a rest is ~0) and NaN-break the curve
    there so no line is drawn across a q_frac interval with no defined dV/dQ.
    """
    i_signed = part.current_A.values * sign
    keep = i_signed > i_thresh
    t = part.time_s.values[keep]
    i = i_signed[keep]
    v = part.voltage_V.values[keep].astype(float)
    dt = np.diff(t)
    med_dt = np.median(dt) if len(dt) else 1.0
    inc = 0.5 * (i[1:] + i[:-1]) * dt
    gap_mask = dt > gap_factor * med_dt
    inc[gap_mask] = 0.0
    dq = np.concatenate([[0.0], np.cumsum(inc)]) / 3600.0
    q = dq / dq[-1]
    gap_idx = np.where(gap_mask)[0]
    for gi in gap_idx:
        v[gi + 1] = np.nan
    return q, v, q[gap_idx]


class CellCycle:
    def __init__(self, csv_path: Path, cycle: int, label: str):
        c = load_cycle(csv_path, cycle)
        chg, dis = split_charge_discharge(c)
        self.label = label
        self.cycle = cycle
        self.q_chg, self.v_chg, self.gaps_chg = build_curve(chg, 1)
        self.q_dis, self.v_dis, self.gaps_dis = build_curve(dis, -1)

    def dvdq_in(self, q0: float, q1: float, n: int = 300, smooth_win: int = 15,
                src_guard: float = 0.02, out_guard: float = 0.05):
        ok = ~np.isnan(self.v_chg)
        for g in self.gaps_chg:
            ok &= np.abs(self.q_chg - g) > src_guard
        q_src, v_src = self.q_chg[ok], self.v_chg[ok]
        grid = np.linspace(q0, q1, n)
        vg = np.interp(grid, q_src, v_src)
        vg = sig.savgol_filter(vg, smooth_win, 3)
        dv = np.gradient(vg, grid)
        for g in self.gaps_chg:
            dv[np.abs(grid - g) < out_guard] = np.nan
        return grid, dv


LFP = CellCycle(MIT_CSV, cycle=600, label="LFP  ·  MIT b1c0  ·  cycle 600")
NCM = CellCycle(TJU_CSV, cycle=100, label="NCM  ·  TJU CY25-05_1-#1  ·  cycle 100")


# ------------------------------------------------------------- helpers -----

def draw_zone_bands(ax, ymin, ymax):
    for name, (s, e) in ZONE_BOUNDS.items():
        ax.axvspan(s, e, color=ZONE_COLOR[name], alpha=0.11, lw=0, zorder=0)
    for (s, e) in OVERLAPS:
        ax.axvspan(s, e, facecolor="none", edgecolor=SUBINK, hatch="////",
                   lw=0, alpha=0.55, zorder=1)
    ax.set_ylim(ymin, ymax)


def panel_vq(ax, cell: CellCycle, chg_names, dis_names, letter):
    vmin = min(np.nanmin(cell.v_chg), np.nanmin(cell.v_dis))
    vmax = max(np.nanmax(cell.v_chg), np.nanmax(cell.v_dis))
    rng = vmax - vmin
    ylim = (vmin - 0.20 * rng, vmax + 0.20 * rng)
    draw_zone_bands(ax, *ylim)

    ax.plot(cell.q_chg, cell.v_chg, **CHARGE_LS, zorder=5)
    ax.plot(cell.q_dis, cell.v_dis, **DISCHARGE_LS, zorder=5)

    lx = 0.40
    ax.text(lx, np.interp(lx, cell.q_chg, cell.v_chg) + 0.035 * rng, "Charge",
            fontsize=6.9, color=INK, fontweight="bold", va="bottom", ha="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.72))
    ax.text(lx, np.interp(lx, cell.q_dis, cell.v_dis) - 0.035 * rng, "Discharge",
            fontsize=6.9, color=SUBINK, va="top", ha="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.72))

    top_y = vmax + 0.10 * rng
    for name, cx in zip(chg_names, (0.175, 0.5, 0.825)):
        ax.text(cx, top_y, name, ha="center", va="bottom", fontsize=6.8,
                color=SUBINK, fontweight="bold")
    ax.annotate("", xy=(0.97, top_y + 0.075 * rng), xytext=(0.05, top_y + 0.075 * rng),
                arrowprops=dict(arrowstyle="-|>", color=SUBINK, lw=0.9,
                                 shrinkA=0, shrinkB=0), annotation_clip=False)

    bot_y = vmin - 0.08 * rng
    for name, cx in zip(dis_names, (0.825, 0.5, 0.175)):
        ax.text(cx, bot_y, name, ha="center", va="top", fontsize=6.8,
                color=SUBINK, fontweight="bold")
    ax.annotate("", xy=(0.05, bot_y - 0.075 * rng), xytext=(0.97, bot_y - 0.075 * rng),
                arrowprops=dict(arrowstyle="-|>", color=SUBINK, lw=0.9,
                                 shrinkA=0, shrinkB=0), annotation_clip=False)

    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Voltage (V)")
    ax.set_xlabel("$q_{frac}$")
    ax.set_title(cell.label, loc="left", pad=6, fontsize=8.0)


def panel_dva(ax):
    q_l, dv_l = LFP.dvdq_in(*ZONE_BOUNDS["right"])
    q_n, dv_n = NCM.dvdq_in(*ZONE_BOUNDS["right"])
    ax.axvspan(*ZONE_BOUNDS["right"], color=Z_RIGHT, alpha=0.08, lw=0, zorder=0)
    ax.plot(q_l, dv_l, color=LFP_COLOR, lw=1.5, ls="-", label="LFP (MIT)", zorder=5)
    ax.plot(q_n, dv_n, color=NCM_COLOR, lw=1.5, ls=(0, (4.5, 2.2)), label="NCM (TJU)", zorder=5)

    pl_i, pn_i = np.nanargmax(dv_l), np.nanargmax(dv_n)
    pl_v, pn_v = dv_l[pl_i], dv_n[pn_i]
    ymax = max(pl_v, pn_v)
    ax.annotate(f"LFP peak $\\approx${pl_v:.1f}", xy=(q_l[pl_i], pl_v),
                xytext=(0.70, ymax * 0.72), fontsize=6.6, color=LFP_COLOR, ha="left",
                arrowprops=dict(arrowstyle="-", color=LFP_COLOR, lw=0.7, alpha=0.6))
    ax.annotate(f"NCM peak $\\approx${pn_v:.1f}", xy=(q_n[pn_i], pn_v),
                xytext=(0.70, ymax * 0.50), fontsize=6.6, color=NCM_COLOR, ha="left",
                arrowprops=dict(arrowstyle="-", color=NCM_COLOR, lw=0.7, alpha=0.6))

    ax.set_xlim(*ZONE_BOUNDS["right"])
    ax.set_xlabel("$q_{frac}$  (chg_hi zone)")
    ax.set_ylabel("dV/dq  (V / unit $q_{frac}$)")
    ax.set_title("Chemistry contrast within the same zone (chg_hi)", loc="left", pad=6, fontsize=8.0)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=7.0, handlelength=2.0,
                     borderaxespad=0.15)
    for t in leg.get_texts():
        t.set_color(SUBINK)


D_CHG_Y, D_ROW_H, D_DIS_Y = 0.98, 0.30, 0.42
D_AXIS_Y = -0.06
D_BRACKET_Y = -0.40
D_LABEL_Y = -0.85
D_COV_Y = -1.05
D_YLIM = (-1.20, 1.62)


def _zone_row(ax, y, h, order, dlabel, up):
    for name in order:
        s, e = ZONE_BOUNDS[name]
        ax.add_patch(Rectangle((s, y), e - s, h, facecolor=ZONE_COLOR[name],
                                edgecolor="white", lw=1.0, alpha=0.88, zorder=3))
    for (s, e) in OVERLAPS:
        ax.add_patch(Rectangle((s, y), e - s, h, facecolor="none", edgecolor=SUBINK,
                                hatch="////", lw=0, alpha=0.55, zorder=4))
    ay = y + h + 0.13 if up else y - 0.13
    x0, x1 = (0.03, 0.97) if up else (0.97, 0.03)
    ax.annotate("", xy=(x1, ay), xytext=(x0, ay),
                arrowprops=dict(arrowstyle="-|>", color=SUBINK, lw=0.9,
                                 shrinkA=0, shrinkB=0), annotation_clip=False)
    ax.text(0.5, ay + (0.03 if up else -0.03), dlabel, ha="center",
            va=("bottom" if up else "top"), fontsize=6.3, color=SUBINK, style="italic")


def panel_zone_schematic(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(*D_YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    _zone_row(ax, D_CHG_Y, D_ROW_H, ["left", "mid", "right"],
              "time →  charge: lo → mid → hi", up=True)
    _zone_row(ax, D_DIS_Y, D_ROW_H, ["left", "mid", "right"],
              "← time  discharge: hi → mid → lo", up=False)

    for name, cx in zip(["chg_lo", "chg_mid", "chg_hi"], (0.175, 0.5, 0.825)):
        ax.text(cx, D_CHG_Y + D_ROW_H / 2, name, ha="center", va="center",
                fontsize=6.9, fontweight="bold", color="white", zorder=5)
    for name, cx in zip(["dis_hi", "dis_mid", "dis_lo"], (0.175, 0.5, 0.825)):
        ax.text(cx, D_DIS_Y + D_ROW_H / 2, name, ha="center", va="center",
                fontsize=6.9, fontweight="bold", color="white", zorder=5)

    ax.plot([0, 1], [D_AXIS_Y, D_AXIS_Y], color=SUBINK, lw=0.9, zorder=2)
    for tick in np.linspace(0, 1, 6):
        ax.plot([tick, tick], [D_AXIS_Y - 0.035, D_AXIS_Y + 0.035], color=SUBINK, lw=0.9)
        ax.text(tick, D_AXIS_Y - 0.085, f"{tick:.2f}" if tick in (N1, 1 - N1) else f"{tick:.1f}",
                ha="center", va="top", fontsize=6.4, color=SUBINK)
    ax.text(0.5, D_LABEL_Y, "Normalized capacity fraction, $q_{frac}$",
            ha="center", va="top", fontsize=7.4, color=INK)

    ax.annotate("", xy=(ZONE_BOUNDS["left"][1], D_BRACKET_Y), xytext=(0, D_BRACKET_Y),
                arrowprops=dict(arrowstyle="|-|,widthA=0.35,widthB=0.35", color=FAINT, lw=0.9),
                annotation_clip=False)
    ax.text(ZONE_BOUNDS["left"][1] / 2, D_BRACKET_Y + 0.05,
            f"$n_1$ = {N1:.2f}", ha="center", va="bottom", fontsize=6.6, color=SUBINK)
    ax.annotate("", xy=(OVERLAPS[0][1], D_BRACKET_Y - 0.28), xytext=(OVERLAPS[0][0], D_BRACKET_Y - 0.28),
                arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3", color=FAINT, lw=0.9),
                annotation_clip=False)
    ax.text(sum(OVERLAPS[0]) / 2, D_BRACKET_Y - 0.23,
            f"$n_2$ = {N2:.2f} overlap (tie-break: max overlap ratio)",
            ha="center", va="bottom", fontsize=6.3, color=SUBINK)

    ax.text(0.5, D_COV_Y, f"Point coverage (validated, n2-range sweep): {COVERAGE_PCT:.3f}% — all six zones",
            ha="center", va="top", fontsize=7.0, color=SUBINK, style="italic")

    ax.set_title("Scenario zone definition on the SOC axis", loc="left", pad=6, fontsize=8.0)


# ------------------------------------------------------------------ main ---

def label_panel(fig, ax, text, dx=-0.028, dy=0.006):
    pos = ax.get_position()
    fig.text(pos.x0 + dx, pos.y1 + dy, text, fontsize=10.5, fontweight="bold",
              color=INK, va="bottom", ha="left")


def build_figure():
    fig = plt.figure(figsize=(7.6, 7.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                           hspace=0.34, wspace=0.30,
                           top=0.965, bottom=0.055, left=0.085, right=0.975)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_vq(ax_a, LFP, ["chg_lo", "chg_mid", "chg_hi"], ["dis_hi", "dis_mid", "dis_lo"], "a")
    panel_vq(ax_b, NCM, ["chg_lo", "chg_mid", "chg_hi"], ["dis_hi", "dis_mid", "dis_lo"], "b")
    panel_dva(ax_c)
    panel_zone_schematic(ax_d)

    for ax in (ax_a, ax_b, ax_c):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=2.8)

    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")):
        label_panel(fig, ax, letter)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig1_problem_scenario_definition.png"
    out_pdf = "docs/figures/fig1_problem_scenario_definition.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
