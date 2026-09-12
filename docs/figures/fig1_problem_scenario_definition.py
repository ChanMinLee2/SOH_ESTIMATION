"""Figure 1 -- Problem framing: chemistry-dependent V-Q shape (A1 + A2).

Camera-ready draft for Journal of Power Sources. All curves are real cell
data (MIT b1c0, HUST 1-7 -- both LFP, different manufacturer; TJU
CY25-05_1-#1 -- NCM), loaded via data_directories.py so the D:-drive location
stays the single point of change. Zone boundaries use the production
q_frac_wide formula (n1=0.35, n2=0.20):
    left  bucket [0.00, n1]        -> chg_lo / dis_hi
    mid   bucket [0.5-n1/2,0.5+n1/2] -> chg_mid / dis_mid
    right bucket [1-n1, 1.00]      -> chg_hi / dis_lo
(see common/scenario/q_frac_wide.py::_zone_bounds + scenario_spec.json routing)

The SOC-zone schematic itself (old panel d) now lives in its own figure,
fig2_scenario_spec.py -- this figure stays focused on the physical problem
statement: three real cells, three distinct dV/dq peak shapes, one shared
SOC axis.

Run: python docs/figures/fig1_problem_scenario_definition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_directories import DATA_4_HI_ROOT  # noqa: E402
from _style import (INK, SUBINK, GRID, Z_LEFT, Z_MID, Z_RIGHT, ZONE_COLOR,  # noqa: E402
                     DATASET_COLOR, ZONE_BOUNDS, OVERLAPS, N1,
                     setup_rcparams, label_panel, strip_top_right)

setup_rcparams()

CELLS = {
    "MIT": dict(csv=DATA_4_HI_ROOT / "clean" / "MIT" / "b1c0.csv", cycle=600, chem="LFP"),
    "HUST": dict(csv=DATA_4_HI_ROOT / "clean" / "HUST" / "1-7.csv", cycle=600, chem="LFP"),
    "TJU": dict(csv=DATA_4_HI_ROOT / "clean" / "TJU" / "CY25-05_1-#1.csv", cycle=100, chem="NCM"),
}


# --------------------------------------------------------------- data ------

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
    a spurious charge to it, and stretches the subsequent relaxation-voltage
    drop across many q_frac points -- a wide multi-lobe glitch once
    differentiated (dV/dq). We zero the increment across any such gap (real
    charge transferred during a rest is ~0) and NaN-break the curve there so
    no line is drawn across a q_frac interval with no defined dV/dQ.
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
    def __init__(self, name: str, csv_path: Path, cycle: int, chem: str):
        c = load_cycle(csv_path, cycle)
        chg, dis = split_charge_discharge(c)
        self.name = name
        self.chem = chem
        self.cycle = cycle
        self.color = DATASET_COLOR[name]
        self.q_chg, self.v_chg, self.gaps_chg = build_curve(chg, 1)
        self.q_dis, self.v_dis, self.gaps_dis = build_curve(dis, -1)

    def dvdq(self, q0: float = 0.0, q1: float = 1.0, n: int = 500, smooth_win: int = 21,
              src_guard: float = 0.02, out_guard: float = 0.05):
        ok = ~np.isnan(self.v_chg)
        for g in self.gaps_chg:
            ok &= np.abs(self.q_chg - g) > src_guard
        q_src, v_src = self.q_chg[ok], self.v_chg[ok]
        grid = np.linspace(q0, q1, n)
        vg = np.interp(grid, q_src, v_src)
        vg = sig.savgol_filter(vg, smooth_win, 3)
        dv = np.gradient(vg, grid)
        edge = int(n * 0.02)
        dv[:edge] = np.nan
        dv[-edge:] = np.nan
        for g in self.gaps_chg:
            dv[np.abs(grid - g) < out_guard] = np.nan
        return grid, dv


CELLS = {k: CellCycle(k, v["csv"], v["cycle"], v["chem"]) for k, v in CELLS.items()}
ORDER = ["MIT", "HUST", "TJU"]


# ------------------------------------------------------------- panel (a) ---

def draw_zone_bands(ax, ymin, ymax):
    for name, (s, e) in ZONE_BOUNDS.items():
        ax.axvspan(s, e, color=ZONE_COLOR[name], alpha=0.10, lw=0, zorder=0)
    for (s, e) in OVERLAPS:
        ax.axvspan(s, e, facecolor="none", edgecolor=SUBINK, hatch="////",
                   lw=0, alpha=0.45, zorder=1)
    ax.set_ylim(ymin, ymax)


def panel_vq_overlay(ax):
    vmin = min(min(np.nanmin(c.v_chg), np.nanmin(c.v_dis)) for c in CELLS.values())
    vmax = max(max(np.nanmax(c.v_chg), np.nanmax(c.v_dis)) for c in CELLS.values())
    rng = vmax - vmin
    ylim = (vmin - 0.22 * rng, vmax + 0.20 * rng)
    draw_zone_bands(ax, *ylim)

    for name in ORDER:
        cell = CELLS[name]
        ax.plot(cell.q_chg, cell.v_chg, color=cell.color, lw=1.6, ls="-",
                solid_capstyle="round", zorder=5, label=f"{name} ({cell.chem})")
        ax.plot(cell.q_dis, cell.v_dis, color=cell.color, lw=1.1, ls=(0, (3.4, 1.8)),
                alpha=0.62, zorder=4)

    top_y = vmax + 0.095 * rng
    for label, cx in zip(["chg_lo", "chg_mid", "chg_hi"], (0.175, 0.5, 0.825)):
        ax.text(cx, top_y, label, ha="center", va="bottom", fontsize=6.7,
                color=SUBINK, fontweight="bold")
    ax.annotate("", xy=(0.97, top_y + 0.07 * rng), xytext=(0.05, top_y + 0.07 * rng),
                arrowprops=dict(arrowstyle="-|>", color=SUBINK, lw=0.85,
                                 shrinkA=0, shrinkB=0), annotation_clip=False)

    bot_y = vmin - 0.10 * rng
    for label, cx in zip(["dis_hi", "dis_mid", "dis_lo"], (0.825, 0.5, 0.175)):
        ax.text(cx, bot_y, label, ha="center", va="top", fontsize=6.7,
                color=SUBINK, fontweight="bold")
    ax.annotate("", xy=(0.05, bot_y - 0.07 * rng), xytext=(0.97, bot_y - 0.07 * rng),
                arrowprops=dict(arrowstyle="-|>", color=SUBINK, lw=0.85,
                                 shrinkA=0, shrinkB=0), annotation_clip=False)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.80),
                     frameon=True, facecolor="white", edgecolor="none", framealpha=0.82,
                     fontsize=7.0, handlelength=1.9, borderaxespad=0.25)
    for t in leg.get_texts():
        t.set_color(SUBINK)

    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Voltage (V)")
    ax.set_xlabel("$q_{frac}$")
    ax.set_title("Charge/discharge V-$q_{frac}$ across three real cells", loc="left", pad=6, fontsize=8.0)


# ------------------------------------------------------------- panel (b) ---

Q_START = 0.08  # skip the shared initial-activation transient (all 3 cells spike
                 # here regardless of chemistry -- not the feature of interest)


def panel_dva_overlay(ax):
    for name, (s, e) in ZONE_BOUNDS.items():
        ax.axvspan(s, e, color=ZONE_COLOR[name], alpha=0.07, lw=0, zorder=0)

    peaks = {}
    for name in ORDER:
        cell = CELLS[name]
        q, dv = cell.dvdq()
        keep = q >= Q_START
        q, dv = q[keep], dv[keep]
        ax.plot(q, dv, color=cell.color, lw=1.5, zorder=5, label=f"{name} ({cell.chem})")
        i = np.nanargmax(dv)
        peaks[name] = (q[i], dv[i])

    ymax = max(v for _, v in peaks.values())
    y_targets = {"MIT": 0.94, "HUST": 0.76, "TJU": 0.56}
    for name in ORDER:
        qx, vy = peaks[name]
        cell = CELLS[name]
        ax.annotate(f"{name} peak $\\approx${vy:.1f} @ $q$={qx:.2f}", xy=(qx, vy),
                    xytext=(0.30, ymax * y_targets[name]), fontsize=6.5, color=cell.color,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=cell.color, lw=0.7, alpha=0.6))

    ax.set_xlim(Q_START, 1)
    ax.set_ylim(-0.3, ymax * 1.12)
    ax.set_xlabel(f"$q_{{frac}}$  (initial activation transient, $q<${Q_START:.2f}, omitted)")
    ax.set_ylabel("dV/dq  (V / unit $q_{frac}$)")
    ax.set_title("Three distinct dV/dq peak shapes, one shared SOC axis", loc="left", pad=6, fontsize=8.0)
    leg = ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none",
                     framealpha=0.82, fontsize=7.0, handlelength=1.9, borderaxespad=0.25)
    for t in leg.get_texts():
        t.set_color(SUBINK)


# ------------------------------------------------------------------ main ---

def build_figure():
    fig = plt.figure(figsize=(7.6, 3.9))
    gs = fig.add_gridspec(1, 2, wspace=0.30,
                           top=0.90, bottom=0.155, left=0.085, right=0.975)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_vq_overlay(ax_a)
    panel_dva_overlay(ax_b)

    for ax in (ax_a, ax_b):
        strip_top_right(ax)

    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)")):
        label_panel(fig, ax, letter)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig1_problem_scenario_definition.png"
    out_pdf = "docs/figures/fig1_problem_scenario_definition.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
