"""Figure 1 -- the three datasets and the six scenarios (2026-11-04).

Replaces fig1_problem_scenario_definition.py. One column per dataset (MIT LFP, HUST LFP, TJU NCM).
(a) Current and voltage of one early-life cycle of a representative cell with the protocol steps.
(b) Charge (solid) and discharge (dashed) voltage against SOC at several aging stages with the three
    SOC zones shaded. Charge SOC = q_frac, discharge SOC = 1 - q_frac (Section 2.2), so the zone bands
    define all six scenarios.
(c) Charge dQ/dV against voltage at the same aging stages (CC part only, Gaussian-smoothed).
(d) SOH of every cell against cycle; TJU colored by ambient temperature; representative cell highlighted.

Data: DATA_4_HI_ROOT/clean/<ds>/<cell>.pkl (cleaned time series) and the per-cycle capacity tables of
      q_frac_ref/<config>/cycle/<ds>/*.pkl (SOH = capacity / first valid cycle capacity).
Run:  python docs/figures/fig1_datasets.py   (LFP_SOH_ESTIMATION env)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))
from data_directories import DATA_4_HI_ROOT  # noqa: E402
from _style import INK, SUBINK, FAINT, DATASET_COLOR, Z_LEFT, Z_MID, Z_RIGHT, setup_rcparams  # noqa: E402

setup_rcparams(6.5)

CYCLE_DIR = (DATA_4_HI_ROOT / "q_frac_ref"
             / "n1-35%_n2-20%_N-2_minpts5_lag-1_noise-3%_ou-200_calib-100_offA-5mA" / "cycle")
DATASETS = ["MIT", "HUST", "TJU"]
CHEM = {"MIT": "LFP", "HUST": "LFP", "TJU": "NCM"}
CELL = {"MIT": ("b1c2", 100), "HUST": ("5-3", 100), "TJU": ("CY45-05_1-#24", 50)}
NOMINAL_AH = {"MIT": 1.1, "HUST": 1.1, "TJU": 3.5}
SOH_STAGES = [1.00, 0.95, 0.90, 0.85, 0.80]
SOH_NORM = Normalize(0.75, 1.0)
CMAP = plt.get_cmap("viridis")
TEMP_COLOR = {"25": "#3A6178", "35": "#A9863F", "45": "#96473A"}
W_ZONE = 0.35
ZONES = [("lo", 0.0, W_ZONE, Z_LEFT), ("mid", 0.5 - W_ZONE / 2, 0.5 + W_ZONE / 2, Z_MID),
         ("hi", 1.0 - W_ZONE, 1.0, Z_RIGHT)]
DQDV_BIN = {"MIT": 0.004, "HUST": 0.004, "TJU": 0.01}
DQDV_XLIM = {"MIT": (3.33, 3.60), "HUST": (3.33, 3.60), "TJU": (3.45, 4.20)}


# ----------------------------------------------------------------- data ----

def load_cells(ds: str) -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(CYCLE_DIR.joinpath(ds).glob("*.pkl")):
        c = pd.read_pickle(p).sort_values("cycle").reset_index(drop=True)
        c["soh"] = c.capacity_Ah / c.capacity_Ah.iloc[0]
        out[p.stem] = c
    return out


def load_series(ds: str, cell: str) -> pd.DataFrame:
    return pd.read_pickle(DATA_4_HI_ROOT / "clean" / ds / f"{cell}.pkl")["cycles"]


def one_cycle(ts: pd.DataFrame, cyc: int) -> pd.DataFrame:
    return ts[ts.cycle == cyc].sort_values("time_s").reset_index(drop=True)


def split(c: pd.DataFrame):
    neg = np.where(c.current_A.values < -0.01)[0]
    k = neg[0] if len(neg) else len(c)
    return c.iloc[:k], c.iloc[k:]


def q_curve(part: pd.DataFrame, sign: int, i_min: float = 0.05):
    """Charge passed (Ah) and voltage over the active part of one step; rests carry no charge."""
    i = part.current_A.values * sign
    keep = i > i_min
    t, i, v = part.time_s.values[keep], i[keep], part.voltage_V.values[keep].astype(float)
    dt = np.diff(t)
    inc = 0.5 * (i[1:] + i[:-1]) * dt / 3600.0
    inc[dt > 6 * np.median(dt)] = 0.0
    return np.concatenate([[0.0], np.cumsum(inc)]), v, i


def steps(t, i, cap, min_len=30.0):
    """Constant-current steps from a run-length of the C-rate rounded to 0.25C."""
    r = np.round(i / cap * 4) / 4
    edges = np.r_[0, np.where(np.diff(r) != 0)[0] + 1, len(r)]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if t[b - 1] - t[a] >= min_len:
            out.append((t[a], t[b - 1], float(np.median(i[a:b]) / cap)))
    return out


def dqdv(part: pd.DataFrame, binw: float):
    q, v, i = q_curve(part, +1)
    vmax = v.max()
    dq = np.diff(q)
    vm = 0.5 * (v[1:] + v[:-1])
    ok = vm < vmax - 0.004                         # drop the constant-voltage hold
    bins = np.arange(np.floor(vm[ok].min() / binw) * binw, vmax + binw, binw)
    h, _ = np.histogram(vm[ok], bins=bins, weights=dq[ok])
    return 0.5 * (bins[1:] + bins[:-1]), gaussian_filter1d(h / binw, 1.5)


def stage_cycles(c: pd.DataFrame):
    picks = []
    for s in SOH_STAGES:
        if c.soh.min() > s + 0.02 and s < 1.0:
            continue
        k = (c.soh - s).abs().idxmin()
        cyc = int(c.cycle[k])
        if cyc not in [p[0] for p in picks]:
            picks.append((cyc, float(c.soh[k])))
    last = (int(c.cycle.iloc[-1]), float(c.soh.iloc[-1]))
    if last[1] < picks[-1][1] - 0.02:
        picks.append(last)
    return picks


# ----------------------------------------------------------------- panels ----

def ax_span(ds):
    return 6.5 if ds != "TJU" else 1.8


def panel_a(ax, ds, ts, cyc_tab, report):
    cell, cyc = CELL[ds]
    c = one_cycle(ts, cyc)
    cap = NOMINAL_AH[ds]
    t = (c.time_s.values - c.time_s.values[0]) / 60.0
    i, v = c.current_A.values, c.voltage_V.values
    ax.axhline(0, color=FAINT, lw=0.4)
    ax.plot(t, i / cap, color=SUBINK, lw=0.8)
    ax.set_ylabel("C-rate", color=SUBINK)
    ax2 = ax.twinx()
    ax2.plot(t, v, color=DATASET_COLOR[ds], lw=1.0)
    ax2.set_ylabel("Voltage (V)", color=DATASET_COLOR[ds])
    ax2.tick_params(axis="y", colors=DATASET_COLOR[ds])
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (min)")
    # annotate CC steps with their C-rate and, for discharge, the SOC at each rate change
    chg, dis = split(c)
    st = steps(t * 60.0, i, cap)
    qd, _, idis = q_curve(dis, -1)
    tdis = (dis.time_s.values[np.abs(dis.current_A.values) > 0.05] - c.time_s.values[0]) / 60.0
    lines = []
    box = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)
    for a, b, cr in st:
        lines.append((a / 60.0, b / 60.0, cr))
        if abs(cr) < 0.4 or (b - a) < 120:
            continue                                 # CV taper and short transitions
        lab = f"{abs(cr):.1f}C".replace(".0C", "C")
        ax2.text((a + b) / 120.0, cr - 0.12 * (1 if cr > 0 else -1) * ax_span(ds), lab, ha="center",
                va="top" if cr > 0 else "bottom", fontsize=5.3, color=SUBINK, bbox=box, zorder=5,
                transform=ax.transData)
    # one CV label over the constant-voltage hold of the charge
    vc_ = chg.voltage_V.values
    tc_ = (chg.time_s.values - c.time_s.values[0]) / 60.0
    hold = np.where(vc_ >= vc_.max() - 0.003)[0]
    if len(hold):
        ax2.text(0.5 * (tc_[hold[0]] + tc_[hold[-1]]), 0.12 * ax_span(ds), "CV", ha="center", va="bottom",
                 fontsize=5.3, color=SUBINK, bbox=box, zorder=5, transform=ax.transData)
    # SOC at discharge rate changes (fraction of the discharge charge)
    soc_marks = []
    rd = np.round(idis / cap * 4) / 4
    ch = np.where(np.diff(rd) != 0)[0] + 1
    ch = [k for k in ch if 0.02 < qd[k] / qd[-1] < 0.98]
    merged = []
    for k in ch:
        if not merged or qd[k] - qd[merged[-1]] > 0.02 * qd[-1]:
            merged.append(k)
    for k in merged:
        soc = 1 - qd[k] / qd[-1]
        soc_marks.append(round(soc, 3))
        ax.axvline(tdis[k], color=FAINT, lw=0.4, ls=":")
    # charge: SOC at the end of the first fast step
    qc, _, ichg = q_curve(chg, +1)
    rc = np.round(ichg / cap * 4) / 4
    chc = [k for k in np.where(np.diff(rc) != 0)[0] + 1 if 0.02 < qc[k] / qc[-1] < 0.98]
    chg_marks = []
    for k in chc:
        if not chg_marks or qc[k] / qc[-1] - chg_marks[-1] > 0.02:
            chg_marks.append(round(qc[k] / qc[-1], 3))
    ax.set_ylim(-6.5, 6.5) if ds != "TJU" else ax.set_ylim(-1.8, 1.8)
    ax2.set_ylim(1.8, 3.8) if ds != "TJU" else ax2.set_ylim(2.3, 4.4)
    ax.set_title(f"{ds} ({CHEM[ds]}), cell {cell}, cycle {cyc}", fontsize=6.5, color=INK, pad=3)
    report[ds] = {"steps": [(round(a / 60, 1), round(b / 60, 1), round(cr, 2)) for a, b, cr in lines],
                  "chg_soc_changes": chg_marks, "dis_soc_changes": soc_marks,
                  "chg_Ah": round(qc[-1], 3), "dis_Ah": round(qd[-1], 3)}


def panel_b(ax, ds, ts, picks):
    for name, a, b, col in ZONES:
        ax.axvspan(a, b, color=col, alpha=0.10, lw=0)
        ax.text((a + b) / 2, 0.03, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=5.8, color=col, fontweight="bold")
    for cyc, soh in picks:
        chg, dis = split(one_cycle(ts, cyc))
        qc, vc, _ = q_curve(chg, +1)
        qd, vd, _ = q_curve(dis, -1)
        col = CMAP(SOH_NORM(soh))
        ax.plot(qc / qc[-1], vc, color=col, lw=0.8)
        ax.plot(1 - qd / qd[-1], vd, color=col, lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim((2.0, 3.75) if ds != "TJU" else (2.5, 4.35))
    ax.set_xlabel("SOC")
    ax.set_ylabel("Voltage (V)")


def panel_c(ax, ds, ts, picks, report):
    peaks = []
    for cyc, soh in picks:
        chg, _ = split(one_cycle(ts, cyc))
        vb, y = dqdv(chg, DQDV_BIN[ds])
        ax.plot(vb, y, color=CMAP(SOH_NORM(soh)), lw=0.8)
        lo, hi = DQDV_XLIM[ds]
        sel = (vb > lo) & (vb < hi - 0.01)
        k = np.argmax(y[sel])
        peaks.append((cyc, round(soh, 3), round(float(vb[sel][k]), 3), round(float(y[sel][k]), 2)))
    ax.set_xlim(*DQDV_XLIM[ds])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(r"dQ/dV (Ah V$^{-1}$)")
    report[ds]["dqdv_peaks"] = peaks


def panel_d(ax, ds, cells, report):
    rep = CELL[ds][0]
    ends = []
    for name, c in cells.items():
        if ds == "TJU":
            col = TEMP_COLOR[name[2:4]]
        else:
            col = "#B5B5B5"
        if name != rep:
            ax.plot(c.cycle, c.soh, color=col, lw=0.35, alpha=0.65)
        ends.append(c.soh.min())
    c = cells[rep]
    ax.plot(c.cycle, c.soh, color=DATASET_COLOR[ds], lw=1.1)
    ax.set_ylim(0.7, 1.03)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH")
    if ds == "TJU":
        for k, (tt, col) in enumerate(TEMP_COLOR.items()):
            ax.plot([], [], color=col, lw=1.0, label=f"{tt} °C")
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), frameon=False, fontsize=5.5, handlelength=1.2)
        temps = {}
        for name, cc in cells.items():
            temps.setdefault(name[2:4], []).append((cc.cycle.iloc[-1], cc.soh.min()))
        report[ds]["temp"] = {k: (len(v), int(np.median([a for a, _ in v])), round(float(np.median([b for _, b in v])), 3))
                              for k, v in temps.items()}
    ax.text(0.98, 0.95, f"{len(cells)} cells", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.8, color=SUBINK)
    report[ds]["soh_min_range"] = (round(min(ends), 3), round(max(ends), 3))
    report[ds]["cycles_range"] = (int(min(cc.cycle.iloc[-1] for cc in cells.values())),
                                  int(max(cc.cycle.iloc[-1] for cc in cells.values())))


# ----------------------------------------------------------------- figure ----

def build():
    fig, axes = plt.subplots(4, 3, figsize=(174 / 25.4, 150 / 25.4),
                             gridspec_kw={"hspace": 0.62, "wspace": 0.52,
                                          "left": 0.075, "right": 0.905, "top": 0.955, "bottom": 0.06})
    report = {}
    for j, ds in enumerate(DATASETS):
        cells = load_cells(ds)
        ts = load_series(ds, CELL[ds][0])
        picks = stage_cycles(cells[CELL[ds][0]])
        panel_a(axes[0, j], ds, ts, cells, report)
        panel_b(axes[1, j], ds, ts, picks)
        panel_c(axes[2, j], ds, ts, picks, report)
        panel_d(axes[3, j], ds, cells, report)
        report[ds]["stages"] = picks
        for r in range(1, 4):
            axes[r, j].spines[["top", "right"]].set_visible(False)
    for r, lab in enumerate("abcd"):
        axes[r, 0].text(-0.36, 1.13, f"({lab})", transform=axes[r, 0].transAxes, fontsize=8,
                        fontweight="bold", va="bottom")
    axes[1, 0].plot([], [], color=SUBINK, lw=0.8, label="Charge")
    axes[1, 0].plot([], [], color=SUBINK, lw=0.8, ls="--", label="Discharge")
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="center", bbox_to_anchor=(0.952, 0.72),
               frameon=False, fontsize=5.5, handlelength=1.6)
    cax = fig.add_axes([0.935, 0.30, 0.012, 0.34])
    cb = fig.colorbar(ScalarMappable(norm=SOH_NORM, cmap=CMAP), cax=cax)
    cb.set_label("SOH of the drawn cycle", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    out = Path(__file__).resolve().parent / "fig1_datasets"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return report


if __name__ == "__main__":
    rep = build()
    for ds, r in rep.items():
        print(ds)
        for k, v in r.items():
            print("  ", k, v)
