"""Figure 8 (2026-11-04) -- estimation reliability by segment position and chemistry.

(a) One row per dataset model (MIT / HUST / TJU single runs). An early-life charge
    and discharge voltage curve of one test cell is drawn against SOC
    (charge SOC = q_frac, discharge SOC = 1 - q_frac). Below the curves the 12
    segment windows sit at their real SOC positions (w_zone 0.35, w_seg 0.20, two
    segments per zone) and are coloured by the test RMSE of their scenario.
    One colour scale is shared by the three rows.
(b) Routing of the scenario classifier, one 3x3 confusion matrix per dataset.
    The matrix in metrics.json is over the classifier position classes, which
    pool the two directions (scenario_spec.json routing: class lo -> chg_hi /
    dis_lo, mid -> chg_mid / dis_mid, hi -> chg_lo / dis_hi). In step terms the
    classes are the start, middle and end of the charge or discharge step, so the
    axes are labelled that way and reordered start -> end.
    The title of each matrix also gives the RMSE under classifier routing against
    routing by the true label.

Run: python docs/figures/fig8_reliability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))   # repo root for data_directories
from data_directories import DATA_4_HI_ROOT  # noqa: E402
from _style import INK, SUBINK, FAINT, DATASET_COLOR, setup_rcparams  # noqa: E402

setup_rcparams(7.0)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "model_lib/results/p1v2_runs"
RUNS = {
    "MIT": "1031_0117_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_mit_seed42",
    "HUST": "1031_0431_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_hust_seed42",
    "TJU": "1031_0938_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_tju_seed42",
}
CHEM = {"MIT": "LFP", "HUST": "LFP", "TJU": "NCM"}
# test cell of each single run (longest-lived with a clean record) and an early-life cycle
CURVE_CELL = {"MIT": ("b1c2", 100), "HUST": ("5-3", 100), "TJU": ("CY45-05_1-#24", 50)}

W_ZONE, W_SEG = 0.35, 0.20
ZONE_START = {"lo": 0.0, "mid": 0.5 - W_ZONE / 2, "hi": 1.0 - W_ZONE}
STEP = W_ZONE - W_SEG                      # spacing of the two segments in a zone
CHG_SCEN = ["chg_lo", "chg_mid", "chg_hi"]
DIS_SCEN = ["dis_lo", "dis_mid", "dis_hi"]  # ordered by SOC
CMAP = plt.get_cmap("YlOrRd")
# classifier class index (lo, mid, hi) -> step position; reorder to start, middle, end
CLASS_ORDER = [2, 1, 0]
CLASS_LABEL = ["Start", "Middle", "End"]


# ----------------------------------------------------------------- data ----

def load_cycle(path: Path, cycle: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["cycle", "time_s", "voltage_V", "current_A"])
    return df[df.cycle == cycle].sort_values("time_s").reset_index(drop=True)


def split_charge_discharge(c: pd.DataFrame):
    neg = np.where(c.current_A.values < -0.01)[0]
    split = neg[0] if len(neg) else len(c)
    return c.iloc[:split], c.iloc[split:]


def build_curve(part: pd.DataFrame, sign: int, i_thresh: float = 0.05, gap_factor: float = 6.0):
    """q_frac vs voltage with rest gaps carrying no charge (same as fig1)."""
    i_signed = part.current_A.values * sign
    keep = i_signed > i_thresh
    t = part.time_s.values[keep]
    i = i_signed[keep]
    v = part.voltage_V.values[keep].astype(float)
    dt = np.diff(t)
    med_dt = np.median(dt) if len(dt) else 1.0
    inc = 0.5 * (i[1:] + i[:-1]) * dt
    inc[dt > gap_factor * med_dt] = 0.0
    q = np.concatenate([[0.0], np.cumsum(inc)])
    return q / q[-1], v


def scenario_rmse(run: str) -> dict:
    df = pd.read_csv(RUN_ROOT / run / "predictions" / "test_predictions_hard.csv",
                     usecols=["seg_name", "soh_true", "soh_pred"])
    e2 = (df.soh_pred - df.soh_true) ** 2
    return (e2.groupby(df.seg_name).mean() ** 0.5 * 100).to_dict()


def load_metrics(run: str) -> dict:
    return json.loads((RUN_ROOT / run / "metrics" / "metrics.json").read_text(encoding="utf-8"))["test"]


# --------------------------------------------------------------- panels ----

def draw_soc_row(ax, ax_bar, name, rmse, norm):
    cell, cyc = CURVE_CELL[name]
    chg, dis = split_charge_discharge(load_cycle(DATA_4_HI_ROOT / "clean" / name / f"{cell}.csv", cyc))
    q_c, v_c = build_curve(chg, 1)
    q_d, v_d = build_curve(dis, -1)
    col = DATASET_COLOR[name]
    ax.plot(q_c, v_c, color=col, lw=1.1, label="Charge")
    ax.plot(1 - q_d, v_d, color=col, lw=1.1, ls=(0, (3, 1.6)), label="Discharge")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Voltage (V)")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.tick_params(labelbottom=False, direction="out", length=2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{name} ({CHEM[name]})", loc="left", fontsize=7.2, fontweight="bold", pad=2)
    ax.set_title(f"test cell {cell}, cycle {cyc}", loc="right", fontsize=6.3, fontweight="normal",
                 color=SUBINK, pad=2)
    for z, x0 in ZONE_START.items():
        ax.axvspan(x0, x0 + W_ZONE, color=FAINT, alpha=0.06 if z != "mid" else 0.12, lw=0)

    # two tracks: charge (top) and discharge (bottom). Each zone is one bar coloured by the
    # scenario RMSE, with its two segment windows marked as thin staggered lines above it.
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 2)
    for track, scens in ((1, CHG_SCEN), (0, DIS_SCEN)):
        for scen, (zone, x0) in zip(scens, ZONE_START.items()):
            c = CMAP(norm(rmse[scen]))
            ax_bar.add_patch(Rectangle((x0 + 0.004, track + 0.06), W_ZONE - 0.008, 0.80,
                                       fc=c, ec="white", lw=0.8))
            for k in range(2):                       # the two 20% segment windows of the zone
                xs = x0 + k * STEP + 0.01
                ax_bar.plot([xs, xs + W_SEG - 0.02], [track + 0.12 + k * 0.08] * 2,
                            color="white", lw=0.7, alpha=0.9, solid_capstyle="butt")
            ax_bar.text(x0 + W_ZONE / 2, track + 0.52, f"{scen}  {rmse[scen]:.2f}", ha="center",
                        va="center", fontsize=6.2,
                        color=INK if norm(rmse[scen]) < 0.6 else "white", fontweight="bold")
    ax_bar.set_yticks([1.5, 0.5])
    ax_bar.set_yticklabels(["Chg", "Dis"], fontsize=6.3)
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.tick_params(axis="x", direction="out", length=2.5)
    for s in ("top", "right", "left"):
        ax_bar.spines[s].set_visible(False)


def draw_confusion(ax, name, met):
    cm = np.asarray(met["hard"]["classification"]["confusion_matrix"], dtype=float)
    cm = cm[np.ix_(CLASS_ORDER, CLASS_ORDER)]
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            n = int(cm[i, j])
            cnt = f"{n / 1000:.1f}k" if n >= 1000 else f"{n}"
            txt = f"{pct[i, j]:.2f}\n{cnt}" if n else "0"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white" if pct[i, j] > 60 else INK)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_LABEL, fontsize=6.3)
    ax.set_yticklabels(CLASS_LABEL, fontsize=6.3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    hard = met["hard"]["capacity"]["rmse"] * 100
    orc = met["oracle"]["capacity"]["rmse"] * 100
    acc = met["hard"]["classification"]["accuracy"] * 100
    ax.set_title(f"{name}  accuracy {acc:.2f}%\nRMSE {hard:.2f}% routed, {orc:.2f}% true label",
                 fontsize=6.6, fontweight="normal", color=INK, pad=3)


def build_figure():
    rmse = {k: scenario_rmse(v) for k, v in RUNS.items()}
    mets = {k: load_metrics(v) for k, v in RUNS.items()}
    allv = [x for d in rmse.values() for x in d.values()]
    norm = Normalize(vmin=np.floor(min(allv) * 10) / 10, vmax=np.ceil(max(allv) * 10) / 10)

    mm = 1 / 25.4
    fig = plt.figure(figsize=(174 * mm, 128 * mm))
    left, right_a, left_b, right = 0.075, 0.655, 0.745, 0.985
    top, bottom = 0.955, 0.17
    row_h = (top - bottom) / 3
    axes_cm = []
    for r, name in enumerate(RUNS):
        y0 = top - (r + 1) * row_h
        ax = fig.add_axes([left, y0 + row_h * 0.42, right_a - left, row_h * 0.52])
        ax_bar = fig.add_axes([left, y0 + row_h * 0.10, right_a - left, row_h * 0.30], sharex=ax)
        draw_soc_row(ax, ax_bar, name, rmse[name], norm)
        if r == 0:
            ax.legend(loc="lower right", fontsize=6.3, frameon=False, handlelength=2.4)
        if r == 2:
            ax_bar.set_xlabel("SOC")
        else:
            ax_bar.tick_params(labelbottom=False)
        axc = fig.add_axes([left_b, y0 + row_h * 0.12, right - left_b, row_h * 0.62])
        draw_confusion(axc, name, mets[name])
        axes_cm.append(axc)
        if r == 0:
            fig.text(left - 0.06, y0 + row_h, "(a)", fontsize=9.5, fontweight="bold", va="top")
            fig.text(left_b - 0.065, y0 + row_h, "(b)", fontsize=9.5, fontweight="bold", va="top")

    axes_cm[-1].set_xlabel("Predicted position in step", fontsize=6.6)
    axes_cm[1].set_ylabel("True position in step", fontsize=6.6)

    cax = fig.add_axes([left + 0.12, 0.075, right_a - left - 0.24, 0.016])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_label("Scenario RMSE (% SOH)", fontsize=6.6, labelpad=1.5)
    cb.ax.tick_params(labelsize=6.3, length=2)
    cb.outline.set_linewidth(0.5)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig8_reliability.png", dpi=300)
    fig.savefig(out / "fig8_reliability.pdf")
    print("saved", out / "fig8_reliability.png")
