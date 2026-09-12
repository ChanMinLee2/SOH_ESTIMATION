"""Figure 2 -- What a "scenario" actually looks like on a real cycle.

Single, self-contained diagnostic figure (real HUST cell 1-1, cycle 2):
  (top)     full-cycle V-t, colored by which of the six SOC-zone scenarios
            (chg_lo/mid/hi, dis_hi/mid/lo) each point falls in
  (bottom)  the same cycle re-plotted as V vs cumulative charge (Q_cum),
            split into discharge (left) and charge (right)

This replaces the earlier 3-panel abstract-schematic + coverage-bar version:
per user feedback (2026-09-13), Figure 2's only job is to make "scenario"
concrete on one real example -- an English-relabeled, _style.py-consistent
re-render of
4_hi_analysis/outputs/seg_diagnose/.../HUST_1-1_cycauto.png
(seg_diagnose.py's own output, which is Korean-labeled and styled
independently of this paper's figure set).

Run: python docs/figures/fig2_scenario_spec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_directories import DATA_4_HI_ROOT  # noqa: E402
from _style import (INK, SUBINK, ZONE_COLOR, ZONE_BOUNDS, setup_rcparams,  # noqa: E402
                     label_panel)

setup_rcparams()

EXAMPLE_CSV = DATA_4_HI_ROOT / "clean" / "HUST" / "1-1.csv"
EXAMPLE_CELL = "HUST 1-1"
EXAMPLE_CYCLE = 2

CHG_ZONE_NAME = {"left": "chg_lo", "mid": "chg_mid", "right": "chg_hi"}
DIS_ZONE_NAME = {"left": "dis_hi", "mid": "dis_mid", "right": "dis_lo"}


def load_cycle(path: Path, cycle: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["cycle", "time_s", "voltage_V", "current_A"])
    return df[df.cycle == cycle].sort_values("time_s").reset_index(drop=True)


def split_charge_discharge(c: pd.DataFrame):
    neg = np.where(c.current_A.values < -0.01)[0]
    split = neg[0] if len(neg) else len(c)
    return c.iloc[:split], c.iloc[split:]


def build_curve(part: pd.DataFrame, sign: int, i_thresh: float = 0.05, gap_factor: float = 6.0):
    """Returns (t, v, q, q_cum_Ah) for one direction -- rest-step gaps zeroed
    + NaN-broken (see fig1's identical helper for the full rationale)."""
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
    q_cum = np.concatenate([[0.0], np.cumsum(inc)]) / 3600.0
    q = q_cum / q_cum[-1]
    gap_idx = np.where(gap_mask)[0]
    for gi in gap_idx:
        v[gi + 1] = np.nan
    return t, v, q, q_cum


def _zone_of(q: np.ndarray) -> np.ndarray:
    """Nearest-zone-center coloring per point (illustrative simplification of
    common/scenario/q_frac_wide.py::_zone_of_segment, which zones whole
    segment windows by max overlap ratio, not single points)."""
    names = np.array(["left", "mid", "right"])
    centers = np.array([sum(ZONE_BOUNDS[n]) / 2 for n in names])
    idx = np.argmin(np.abs(q[:, None] - centers[None, :]), axis=1)
    return names[idx]


def _plot_colored(ax, x, v, zone, lw, ls="-", alpha=1.0, zorder=5):
    for zname in ("left", "mid", "right"):
        m = zone == zname
        if m.any():
            ax.plot(x[m], v[m], color=ZONE_COLOR[zname], lw=lw, ls=ls,
                    alpha=alpha, zorder=zorder, solid_capstyle="round")


def _shade_zones(ax, x, zone):
    """Background axvspan per zone, spanning that zone's [min,max] extent in
    whatever x-coordinate is passed (time or Q_cum) -- makes the colored-line
    convention doubly legible (line color + background tint agree)."""
    for zname in ("left", "mid", "right"):
        m = zone == zname
        if m.any():
            ax.axvspan(x[m].min(), x[m].max(), color=ZONE_COLOR[zname], alpha=0.12,
                       lw=0, zorder=0)


def panel_vt(ax, t_chg, v_chg, zone_chg, t_dis, v_dis, zone_dis, gap_s):
    t_dis_shifted = t_dis + gap_s
    _shade_zones(ax, t_chg, zone_chg)
    _shade_zones(ax, t_dis_shifted, zone_dis)
    _plot_colored(ax, t_chg, v_chg, zone_chg, lw=1.8)
    _plot_colored(ax, t_dis_shifted, v_dis, zone_dis, lw=1.8)

    vmin = min(np.nanmin(v_chg), np.nanmin(v_dis))
    vmax = max(np.nanmax(v_chg), np.nanmax(v_dis))
    rng = vmax - vmin
    ax.set_ylim(vmin - 0.10 * rng, vmax + 0.12 * rng)
    ax.axvline(t_chg[-1] + gap_s / 2, color=SUBINK, lw=0.8, ls=(0, (4, 2)), alpha=0.6)
    ax.text(t_chg[len(t_chg) // 3], vmax + 0.06 * rng, "Charge", fontsize=8.5,
            color=INK, fontweight="bold", ha="center")
    ax.text(t_dis_shifted[len(t_dis) // 3], vmax + 0.06 * rng, "Discharge", fontsize=8.5,
            color=INK, fontweight="bold", ha="center")

    handles = [plt.Line2D([0], [0], color=ZONE_COLOR[z], lw=2.2) for z in ("left", "mid", "right")]
    labels = [f"{CHG_ZONE_NAME[z]} / {DIS_ZONE_NAME[z]}" for z in ("left", "mid", "right")]
    leg = ax.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                     fontsize=7.6, handlelength=1.6, bbox_to_anchor=(0.5, -0.32),
                     columnspacing=1.4, borderaxespad=0.2)
    for lt in leg.get_texts():
        lt.set_color(SUBINK)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Full cycle, {EXAMPLE_CELL}, cycle {EXAMPLE_CYCLE} -- colored by scenario",
                 loc="left", pad=6, fontsize=8.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


def panel_vq(ax, q_cum, v, zone, title, xlabel):
    _shade_zones(ax, q_cum, zone)
    _plot_colored(ax, q_cum, v, zone, lw=2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Voltage (V)")
    ax.set_title(title, loc="left", pad=6, fontsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


def build_figure():
    c = load_cycle(EXAMPLE_CSV, EXAMPLE_CYCLE)
    chg, dis = split_charge_discharge(c)
    t_chg, v_chg, q_chg, qcum_chg = build_curve(chg, 1)
    t_dis, v_dis, q_dis, qcum_dis = build_curve(dis, -1)
    zone_chg = _zone_of(q_chg)
    zone_dis = _zone_of(q_dis)
    gap_s = 0.03 * (t_chg[-1] - t_chg[0])

    fig = plt.figure(figsize=(7.6, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.32,
                           top=0.94, bottom=0.10, left=0.10, right=0.97)
    ax_top = fig.add_subplot(gs[0, :])
    ax_dis = fig.add_subplot(gs[1, 0])
    ax_chg = fig.add_subplot(gs[1, 1])

    panel_vt(ax_top, t_chg, v_chg, zone_chg, t_dis, v_dis, zone_dis, gap_s)
    panel_vq(ax_dis, qcum_dis, v_dis, zone_dis, "Discharge V-Q", "$Q_{cum}$ (Ah)")
    panel_vq(ax_chg, qcum_chg, v_chg, zone_chg, "Charge V-Q", "$Q_{cum}$ (Ah)")

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig2_scenario_spec.png"
    out_pdf = "docs/figures/fig2_scenario_spec.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
