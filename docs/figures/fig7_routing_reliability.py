"""Figure 7 (2026-11-04) -- routing accuracy and position-dependent reliability.

One column per dataset model (MIT / HUST / TJU single runs, hard routing).
(a) Routing of the scenario classifier. The 3x3 matrix is over the classifier
    position classes (start, middle, end of the charge or discharge step; the
    direction is known from the current). metrics.json stores the matrix pooled
    over the two directions. The strip on the right gives the accuracy of each
    of the six scenarios (per_scenario_accuracy in metrics.json).
(b) Signed error of every test segment per scenario (half violins, 1-99th
    percentile) with the scenario RMSE above it under classifier routing
    (filled) and under routing by the true label (open). seg_name in the
    prediction files is the TRUE scenario; test_predictions.csv is true-label
    routing and test_predictions_hard.csv is classifier routing.
(c) Early-life charge and discharge voltage curve of one test cell against SOC
    with the 12 segment windows at their real SOC positions; each zone bar is
    coloured by the RMSE of its scenario.
(d) RMSE of each scenario in bands of the true SOH, with the share of test
    segments in each band on top.

Run: python docs/figures/fig7_routing_reliability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))   # repo root for data_directories
from data_directories import DATA_4_HI_ROOT  # noqa: E402
from _style import INK, SUBINK, FAINT, GRID, DATASET_COLOR, Z_LEFT, Z_MID, Z_RIGHT, setup_rcparams  # noqa: E402

setup_rcparams(6.6)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "model_lib/results/p1v2_runs"
RUNS = {
    "MIT": "1031_0117_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_mit_seed42",
    "HUST": "1031_0431_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_hust_seed42",
    "TJU": "1031_0938_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_tju_seed42",
}
CHEM = {"MIT": "LFP", "HUST": "LFP", "TJU": "NCM"}
CURVE_CELL = {"MIT": ("b1c2", 100), "HUST": ("5-3", 100), "TJU": ("CY45-05_1-#24", 50)}

W_ZONE, W_SEG = 0.35, 0.20
ZONE_START = {"lo": 0.0, "mid": 0.5 - W_ZONE / 2, "hi": 1.0 - W_ZONE}
STEP = W_ZONE - W_SEG
CHG_SCEN = ["chg_lo", "chg_mid", "chg_hi"]
DIS_SCEN = ["dis_lo", "dis_mid", "dis_hi"]            # ordered by SOC
SCEN_ORDER = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]   # step order
SCEN_COLOR = dict(zip(SCEN_ORDER, [Z_LEFT, Z_MID, Z_RIGHT, Z_LEFT, Z_MID, Z_RIGHT]))
SCEN_LABEL = {"chg_lo": "C-lo", "chg_mid": "C-mid", "chg_hi": "C-hi",
              "dis_hi": "D-hi", "dis_mid": "D-mid", "dis_lo": "D-lo"}
CLASS_ORDER = [2, 1, 0]                                # classifier class (lo, mid, hi) -> start, middle, end
CLASS_LABEL = ["Start", "Middle", "End"]
SOH_BANDS = [(0.95, 1.10, "≥0.95"), (0.90, 0.95, "0.90–0.95"), (0.85, 0.90, "0.85–0.90"),
             (0.80, 0.85, "0.80–0.85"), (0.0, 0.80, "<0.80")]
CMAP_POS = plt.get_cmap("YlOrRd")
CMAP_BAND = LinearSegmentedColormap.from_list("band", ["#F7F4EC", "#E8C77A", "#C9773F", "#8E2F25", "#4A1410"])
CMAP_CM = LinearSegmentedColormap.from_list("cm", ["#FFFFFF", "#C9D6E0", "#3A6178", "#182E45"])


# ----------------------------------------------------------------- data ----

def load_cycle(path: Path, cycle: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["cycle", "time_s", "voltage_V", "current_A"])
    return df[df.cycle == cycle].sort_values("time_s").reset_index(drop=True)


def split_charge_discharge(c: pd.DataFrame):
    neg = np.where(c.current_A.values < -0.01)[0]
    split = neg[0] if len(neg) else len(c)
    return c.iloc[:split], c.iloc[split:]


def build_curve(part: pd.DataFrame, sign: int, i_thresh: float = 0.05, gap_factor: float = 6.0):
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


def load_run(run: str):
    d = RUN_ROOT / run
    met = json.loads((d / "metrics" / "metrics.json").read_text(encoding="utf-8"))["test"]
    cols = ["cell_id", "seg_name", "soh_true", "soh_pred"]
    hard = pd.read_csv(d / "predictions" / "test_predictions_hard.csv", usecols=cols)
    orc = pd.read_csv(d / "predictions" / "test_predictions.csv", usecols=cols)
    hard["err"] = (hard.soh_pred - hard.soh_true) * 100
    hard["err_true"] = (orc.soh_pred.values - orc.soh_true.values) * 100
    hard["misrouted"] = (hard.soh_pred.values - orc.soh_pred.values) != 0
    return met, hard


def rmse(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


# --------------------------------------------------------------- panels ----

def style_ax(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(direction="out", length=2.2, width=0.5)


def draw_routing(ax, ax_acc, name, met, df):
    cm = np.asarray(met["hard"]["classification"]["confusion_matrix"], dtype=float)
    cm = cm[np.ix_(CLASS_ORDER, CLASS_ORDER)]
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    disp = np.where(np.eye(3, dtype=bool), 100.0, np.clip(pct * 40, 0, 60))   # off-diagonal errors amplified for visibility
    ax.imshow(disp, cmap=CMAP_CM, vmin=0, vmax=100, aspect="equal")
    for i in range(3):
        for j in range(3):
            n = int(cm[i, j])
            if i == j:
                ax.text(j, i - 0.13, f"{pct[i, j]:.2f}%", ha="center", va="center", fontsize=5.4,
                        color="white", fontweight="bold")
                ax.text(j, i + 0.22, f"n={n / 1000:.1f}k", ha="center", va="center", fontsize=4.9,
                        color="#D5DEE6")
            elif n:
                ax.text(j, i, f"{n}", ha="center", va="center", fontsize=5.6, color=INK, fontweight="bold")
            else:
                ax.text(j, i, "0", ha="center", va="center", fontsize=5.6, color=FAINT)
    # the far corners (start <-> end) are never used: mark them
    for (i, j) in ((0, 2), (2, 0)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec="#C9C2B4", lw=0))
    for k in range(4):
        ax.axhline(k - 0.5, color="white", lw=1.0)
        ax.axvline(k - 0.5, color="white", lw=1.0)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_LABEL, fontsize=5.8)
    ax.set_yticklabels(CLASS_LABEL, fontsize=5.8, rotation=90, va="center")
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    acc = met["hard"]["classification"]["accuracy"] * 100
    n_mis = int(df.misrouted.sum())
    ax.set_title(f"accuracy {acc:.2f}%   misrouted {n_mis:,}", fontsize=5.9, color=INK, pad=2.5,
                 fontweight="normal")

    # per-scenario accuracy strip
    acc_s = met["hard"]["classification"]["per_scenario_accuracy"]
    ys = np.arange(len(SCEN_ORDER))[::-1]
    for y, s in zip(ys, SCEN_ORDER):
        a = acc_s[s] * 100
        ax_acc.plot([98.0, a], [y, y], color=GRID, lw=1.6, solid_capstyle="butt", zorder=1)
        ax_acc.scatter([a], [y], s=13, color=SCEN_COLOR[s], edgecolor="white", lw=0.4, zorder=3)
        ax_acc.text(97.93, y, SCEN_LABEL[s], ha="right", va="center", fontsize=5.0, color=SUBINK)
    ax_acc.axvline(100, color=FAINT, lw=0.5, ls=(0, (2, 1.5)))
    ax_acc.axhline(2.5, color=FAINT, lw=0.4)
    ax_acc.set_xlim(98.0, 100.15)
    ax_acc.set_ylim(-0.6, 5.6)
    ax_acc.set_yticks([])
    ax_acc.set_xticks([98, 99, 100])
    ax_acc.tick_params(axis="x", labelsize=5.0, length=1.8, pad=1)
    style_ax(ax_acc, keep=("bottom",))
    ax_acc.set_title("scenario acc. (%)", fontsize=5.3, color=SUBINK, pad=2.5, fontweight="normal")


def draw_errors(ax, ax_r, name, met, df, rmse_hard, rmse_true, best, worst):
    xs = np.arange(len(SCEN_ORDER))
    for x, s in zip(xs, SCEN_ORDER):
        e = df.loc[df.seg_name == s, "err"].values
        lo, hi = np.percentile(e, [1, 99])
        e = e[(e >= lo) & (e <= hi)]
        vp = ax.violinplot([e], positions=[x], widths=0.82, showextrema=False, points=160)
        body = vp["bodies"][0]
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], x, None)            # right half only
        body.set_facecolor(SCEN_COLOR[s])
        body.set_edgecolor("none")
        body.set_alpha(0.55)
        q1, med, q3 = np.percentile(e, [25, 50, 75])
        p5, p95 = np.percentile(e, [5, 95])
        ax.plot([x - 0.06, x - 0.06], [p5, p95], color=SCEN_COLOR[s], lw=0.6)
        ax.add_patch(Rectangle((x - 0.12, q1), 0.12, q3 - q1, fc=SCEN_COLOR[s], ec="none", alpha=0.95))
        ax.plot([x - 0.14, x + 0.02], [med, med], color="white", lw=0.8)
        # misrouted segments of this scenario as dots
        mis = df.loc[(df.seg_name == s) & df.misrouted, "err"].values
        if len(mis):
            rng = np.random.default_rng(0)
            sub = mis if len(mis) <= 120 else rng.choice(mis, 120, replace=False)
            ax.scatter(x - 0.24 + rng.uniform(-0.05, 0.05, len(sub)), np.clip(sub, -7.4, 7.4), s=1.2,
                       color="#B23A2B", lw=0, alpha=0.8, zorder=4)
    ax.axhline(0, color=INK, lw=0.5)
    for y in (-4, -2, 2, 4):
        ax.axhline(y, color=GRID, lw=0.4, zorder=0)
    ax.axvline(2.5, color=FAINT, lw=0.5, ls=(0, (2, 1.5)))
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(-7.5, 7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCEN_ORDER], fontsize=5.4)
    style_ax(ax)
    ax.text(1.0, -7.3, "charge", ha="center", va="bottom", fontsize=5.4, color=SUBINK)
    ax.text(4.0, -7.3, "discharge", ha="center", va="bottom", fontsize=5.4, color=SUBINK)

    # RMSE bars above: classifier routing (filled) and true-label routing (open)
    for x, s in zip(xs, SCEN_ORDER):
        rh, rt = rmse_hard[s], rmse_true[s]
        ax_r.bar(x - 0.17, rh, width=0.32, color=SCEN_COLOR[s], lw=0)
        ax_r.bar(x + 0.17, rt, width=0.32, facecolor="white", edgecolor=SCEN_COLOR[s], lw=0.7, hatch="//////")
        tag = ""
        if s == worst:
            tag = "▲"
        elif s == best:
            tag = "▼"
        ax_r.text(x, max(rh, rt) + 0.12, f"{rh:.2f}{tag}", ha="center", va="bottom", fontsize=5.1,
                  color="#B23A2B" if s == worst else ("#2F6B4F" if s == best else INK),
                  fontweight="bold" if s in (best, worst) else "normal")
    ax_r.set_ylim(0, 3.35)
    ax_r.set_xlim(-0.6, 5.6)
    ax_r.set_yticks([0, 1, 2, 3])
    ax_r.tick_params(axis="y", labelsize=5.2)
    ax_r.tick_params(axis="x", labelbottom=False, length=0)
    ax_r.axvline(2.5, color=FAINT, lw=0.5, ls=(0, (2, 1.5)))
    for y in (1, 2, 3):
        ax_r.axhline(y, color=GRID, lw=0.4, zorder=0)
    style_ax(ax_r, keep=("left",))
    h = met["hard"]["capacity"]["rmse"] * 100
    o = met["oracle"]["capacity"]["rmse"] * 100
    ax_r.set_title(f"RMSE {h:.2f}% routed  |  {o:.2f}% true label  (Δ {h - o:+.2f} pp)",
                   fontsize=5.9, color=INK, pad=2.5, fontweight="normal")


def draw_soc(ax, ax_bar, name, rmse_hard, norm, worst):
    cell, cyc = CURVE_CELL[name]
    chg, dis = split_charge_discharge(load_cycle(DATA_4_HI_ROOT / "clean" / name / f"{cell}.csv", cyc))
    q_c, v_c = build_curve(chg, 1)
    q_d, v_d = build_curve(dis, -1)
    col = DATASET_COLOR[name]
    for z, x0 in ZONE_START.items():
        ax.axvspan(x0, x0 + W_ZONE, color=FAINT, alpha=0.05 if z != "mid" else 0.11, lw=0)
    ax.plot(q_c, v_c, color=col, lw=1.0)
    ax.plot(1 - q_d, v_d, color=col, lw=1.0, ls=(0, (3, 1.6)))
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.4 if CHEM[name] == "NCM" else 0.3))
    ax.tick_params(labelbottom=False)
    style_ax(ax)
    ax.tick_params(axis="y", labelsize=5.4)
    if CHEM[name] == "LFP":
        # flat plateau bracket over the mid zone of the charge curve
        sel = (q_c > 0.25) & (q_c < 0.75)
        vtop = np.nanmax(v_c[sel]) + 0.06
        ax.annotate("", xy=(0.25, vtop), xytext=(0.75, vtop),
                    arrowprops=dict(arrowstyle="|-|", lw=0.5, color=SUBINK, mutation_scale=2.5))
        ax.text(0.5, vtop + 0.025, "flat plateau", ha="center", va="bottom", fontsize=5.1, color=SUBINK)
    else:
        ax.text(0.5, 0.06, "sloped curve, no plateau", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=5.1, color=SUBINK)
    ax.set_title(f"cell {cell}, cycle {cyc}", fontsize=5.4, color=SUBINK, pad=1.5, fontweight="normal",
                 loc="right")

    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 2)
    for track, scens in ((1, CHG_SCEN), (0, DIS_SCEN)):
        for scen, (zone, x0) in zip(scens, ZONE_START.items()):
            c = CMAP_POS(norm(rmse_hard[scen]))
            ax_bar.add_patch(Rectangle((x0 + 0.004, track + 0.08), W_ZONE - 0.008, 0.78, fc=c,
                                       ec="#B23A2B" if scen == worst else "white",
                                       lw=1.1 if scen == worst else 0.8, zorder=2 if scen == worst else 1))
            for k in range(2):
                xs = x0 + k * STEP + 0.012
                ax_bar.plot([xs, xs + W_SEG - 0.024], [track + 0.16 + k * 0.09] * 2, color="white", lw=0.6,
                            alpha=0.9, solid_capstyle="butt", zorder=3)
            ax_bar.text(x0 + W_ZONE / 2, track + 0.55, f"{rmse_hard[scen]:.2f}", ha="center", va="center",
                        fontsize=5.5, color=INK if norm(rmse_hard[scen]) < 0.6 else "white", fontweight="bold",
                        zorder=4)
    ax_bar.set_yticks([1.47, 0.47])
    ax_bar.set_yticklabels(["Chg", "Dis"], fontsize=5.4)
    ax_bar.tick_params(axis="y", length=0, pad=1)
    ax_bar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.tick_params(axis="x", labelsize=5.4, length=2, pad=1)
    for s in ("top", "right", "left"):
        ax_bar.spines[s].set_visible(False)
    ax_bar.spines["bottom"].set_linewidth(0.6)


def draw_bands(ax, name, df, norm):
    mat = np.full((6, len(SOH_BANDS)), np.nan)
    cnt = np.zeros_like(mat)
    for i, s in enumerate(SCEN_ORDER):
        d = df[df.seg_name == s]
        for j, (lo, hi, _) in enumerate(SOH_BANDS):
            e = d.loc[(d.soh_true > lo) & (d.soh_true <= hi), "err"].values
            cnt[i, j] = len(e)
            if len(e):
                mat[i, j] = rmse(e)
    ax.imshow(mat, cmap=CMAP_BAND, norm=norm, aspect="auto")
    for i in range(6):
        for j in range(len(SOH_BANDS)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "–", ha="center", va="center", fontsize=5.2, color=FAINT)
                continue
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=5.0,
                    color="white" if norm(mat[i, j]) > 0.5 else INK)
    for k in range(7):
        ax.axhline(k - 0.5, color="white", lw=0.8)
    for k in range(len(SOH_BANDS) + 1):
        ax.axvline(k - 0.5, color="white", lw=0.8)
    ax.axhline(2.5, color=INK, lw=0.8)
    ax.set_yticks(range(6))
    ax.set_yticklabels([SCEN_LABEL[s] for s in SCEN_ORDER], fontsize=5.4)
    ax.set_xticks(range(len(SOH_BANDS)))
    ax.set_xticklabels([b[2] for b in SOH_BANDS], fontsize=5.0, rotation=0)
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    share = [(df.soh_true > lo).mul(df.soh_true <= hi).mean() * 100 for lo, hi, _ in SOH_BANDS]
    for j, p in enumerate(share):
        ax.text(j, -0.78, f"{p:.0f}%", ha="center", va="bottom", fontsize=4.8, color=SUBINK)
    return mat, cnt


# --------------------------------------------------------------- figure ----

def build_figure():
    data = {k: load_run(v) for k, v in RUNS.items()}
    rmse_hard = {k: df.groupby("seg_name").err.apply(rmse).to_dict() for k, (_, df) in data.items()}
    rmse_true = {k: df.groupby("seg_name").err_true.apply(rmse).to_dict() for k, (_, df) in data.items()}
    norm_pos = Normalize(vmin=0.7, vmax=2.7)
    norm_band = Normalize(vmin=0.0, vmax=6.0)

    mm = 1 / 25.4
    fig = plt.figure(figsize=(174 * mm, 196 * mm))
    L, R = 0.07, 0.985
    gap = 0.045
    cw = (R - L - 2 * gap) / 3
    # row boundaries (figure fraction, top -> bottom)
    rA = (0.800, 0.940)
    rB = (0.545, 0.735)
    rC = (0.300, 0.470)
    rD = (0.045, 0.188)
    out = {}
    for c, name in enumerate(RUNS):
        x0 = L + c * (cw + gap)
        met, df = data[name]
        rh, rt = rmse_hard[name], rmse_true[name]
        worst = max(rh, key=rh.get)
        best = min(rh, key=rh.get)

        # column header
        fig.text(x0 + cw / 2, 0.982, f"{name}  ({CHEM[name]})", ha="center", va="top", fontsize=7.6,
                 fontweight="bold", color=DATASET_COLOR[name])
        fig.add_artist(Line2D([x0, x0 + cw], [0.962, 0.962], color=DATASET_COLOR[name], lw=1.2))

        ax_cm = fig.add_axes([x0 + 0.004, rA[0], cw * 0.52, rA[1] - rA[0]])
        ax_acc = fig.add_axes([x0 + cw * 0.76, rA[0] + 0.012, cw * 0.24, rA[1] - rA[0] - 0.024])
        draw_routing(ax_cm, ax_acc, name, met, df)

        hB = rB[1] - rB[0]
        ax_r = fig.add_axes([x0, rB[0] + hB * 0.66, cw, hB * 0.30])
        ax_e = fig.add_axes([x0, rB[0], cw, hB * 0.62])
        draw_errors(ax_e, ax_r, name, met, df, rh, rt, best, worst)

        hC = rC[1] - rC[0]
        ax_v = fig.add_axes([x0, rC[0] + hC * 0.40, cw, hC * 0.56])
        ax_b = fig.add_axes([x0, rC[0], cw, hC * 0.36], sharex=ax_v)
        draw_soc(ax_v, ax_b, name, rh, norm_pos, worst)

        ax_d = fig.add_axes([x0, rD[0], cw, rD[1] - rD[0]])
        out[name] = draw_bands(ax_d, name, df, norm_band)

        if c == 0:
            ax_cm.set_ylabel("True position", fontsize=5.8, labelpad=1)
            ax_e.set_ylabel("Error (% SOH)", fontsize=5.8, labelpad=1)
            ax_r.set_ylabel("RMSE (%)", fontsize=5.8, labelpad=1)
            ax_v.set_ylabel("Voltage (V)", fontsize=5.8, labelpad=1)
            ax_d.set_ylabel("Scenario", fontsize=5.8, labelpad=1)
        ax_cm.set_xlabel("Predicted position", fontsize=5.8, labelpad=1)
        ax_b.set_xlabel("SOC", fontsize=5.8, labelpad=1)
        ax_d.set_xlabel("True SOH band", fontsize=5.8, labelpad=1)

    # panel letters and row titles
    rows = [(rA, "(a)", "Routing of the scenario classifier"),
            (rB, "(b)", "Error per scenario under classifier and true-label routing"),
            (rC, "(c)", "Scenario RMSE at the SOC position of each zone"),
            (rD, "(d)", "Scenario RMSE by SOH band")]
    for (lo, hi), lab, title in rows:
        top = hi + (0.030 if lab != "(a)" else 0.006)
        fig.text(0.008, top, lab, fontsize=8.5, fontweight="bold", va="center")
        if lab != "(a)":
            fig.text(0.04, top, title, fontsize=6.4, va="center", color=INK, style="italic")

    # legends
    leg = [Rectangle((0, 0), 1, 1, fc=SUBINK, lw=0, label="classifier routing"),
           Rectangle((0, 0), 1, 1, fc="white", ec=SUBINK, hatch="//////", lw=0.6, label="true-label routing"),
           Line2D([], [], marker="o", ls="", ms=2.2, color="#B23A2B", label="misrouted segment"),
           Line2D([], [], marker="$▲$", ls="", ms=3.5, color="#B23A2B", label="worst"),
           Line2D([], [], marker="$▼$", ls="", ms=3.5, color="#2F6B4F", label="best")]
    fig.legend(handles=leg, loc="center right", bbox_to_anchor=(R, rB[1] + 0.030), ncol=5, fontsize=5.3,
               frameon=False, handlelength=1.3, columnspacing=1.0, handletextpad=0.4)
    leg2 = [Line2D([], [], color=SUBINK, lw=1.0, label="charge"),
            Line2D([], [], color=SUBINK, lw=1.0, ls=(0, (3, 1.6)), label="discharge"),
            Rectangle((0, 0), 1, 1, fc="white", ec="#B23A2B", lw=1.0, label="worst scenario")]
    fig.legend(handles=leg2, loc="center right", bbox_to_anchor=(R - 0.30, rC[1] + 0.030), ncol=3,
               fontsize=5.3, frameon=False, handlelength=1.8, columnspacing=1.0)

    cax = fig.add_axes([R - 0.17, rC[1] + 0.026, 0.17, 0.009])
    cb = fig.colorbar(ScalarMappable(norm=norm_pos, cmap=CMAP_POS), cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    fig.text(R - 0.175, rC[1] + 0.0305, "scenario RMSE (%)", ha="right", va="center", fontsize=5.3)

    cax2 = fig.add_axes([R - 0.17, rD[1] + 0.037, 0.17, 0.009])
    cb2 = fig.colorbar(ScalarMappable(norm=norm_band, cmap=CMAP_BAND), cax=cax2, orientation="horizontal")
    cb2.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
    cb2.outline.set_linewidth(0.4)
    fig.text(R - 0.175, rD[1] + 0.0415, "RMSE (%)    numbers on top: share of test segments in each band",
             ha="right", va="center", fontsize=5.3)
    return fig, data, rmse_hard, rmse_true, out


if __name__ == "__main__":
    fig, data, rh, rt, bands = build_figure()
    outdir = Path(__file__).resolve().parent
    fig.savefig(outdir / "fig7_routing_reliability.png", dpi=300)
    fig.savefig(outdir / "fig7_routing_reliability.pdf")
    print("saved", outdir / "fig7_routing_reliability.png")
    for name, (met, df) in data.items():
        mis = df[df.misrouted]
        print(name, "misrouted", len(mis), "of", len(df),
              "| RMSE on misrouted: routed %.2f true %.2f" % (rmse(mis.err), rmse(mis.err_true)) if len(mis) else "")
        print("  hard", {k: round(v, 2) for k, v in rh[name].items()})
        print("  true", {k: round(v, 2) for k, v in rt[name].items()})
        mat, cnt = bands[name]
        print("  bands\n", np.round(mat, 2), "\n", cnt.astype(int))
