"""Figure 8 — window-width sweep of the pooled LFP model (2026-11-04).

Replaces the former fig8_reliability. One setting is varied at a time on the pooled MIT+HUST model
(scenario Transformer, union gates, no warm start, lambda_L0 2.37e-4); only the segment width
w_seg (n2) changes. The figure explains why w_seg = 0.20 is the reference:
  narrow windows hold too little of the curve shape, and wide windows make the segments of one
  zone nearly identical.

(a) SOC axis with the three zones of width w_zone = 0.35 and the segment placement for each width
    (N_samp segments per zone, spacing Delta = (w_zone - w_seg)/(N_samp - 1)); overlap of adjacent
    segments annotated. An early-life MIT and HUST charge curve is drawn behind the axis.
(b) Voltage span captured by a window against its centre on those curves (shape content).
(c) RMSE against width: all, charge, discharge, MIT and HUST, with R^2 on a twin axis.
(d) Scenario x width RMSE heatmap.
(e) Per-cell RMSE of the 40 test cells, paired across widths.
(f) Median absolute error against true SOH for each width (IQR band).
(g) Position dependence d_bar of the 63 raw HIs (stage-1 test) for each width, tau_eff = 0.2.
(h) Pipeline outcome per width: scenario-specific HIs, synergy groups, selected raw and kernel HIs.

To add a run (e.g. w_seg = 0.35) put one entry in RUNS; panel (a) already shows 0.35 as a dashed
row because it is the limiting case Delta = 0 where both segments of a zone coincide.
Run:  python docs/figures/fig8_window_sweep.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))
from data_directories import DATA_4_HI_ROOT  # noqa: E402
from _style import INK, SUBINK, FAINT, GRID, ZONE_COLOR, DATASET_COLOR, setup_rcparams  # noqa: E402

setup_rcparams(6.4)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "model_lib/results/p1v2_runs"
OUT = Path(__file__).resolve().parent

# width -> run id prefix. Add 0.35 here once that run exists.
RUNS = {0.10: "1101_0418", 0.20: "1029_0347", 0.30: "1101_2107"}
REF = 0.20
PENDING = [0.35]                       # drawn only in the schematic
W_COLOR = {0.10: "#7FA3BF", 0.20: "#96473A", 0.30: "#3A6178", 0.35: "#644E70"}
W_ZONE = 0.35
ZONE_START = {"lo": 0.0, "mid": 0.5 - W_ZONE / 2, "hi": 1.0 - W_ZONE}
ZKEY = {"lo": "left", "mid": "mid", "hi": "right"}
SCEN = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
SCEN_LABEL = ["c-lo", "c-mid", "c-hi", "d-hi", "d-mid", "d-lo"]
CURVES = {"MIT": ("b1c2", 100), "HUST": ("5-3", 100)}
TAU_EFF = 0.2


# ----------------------------------------------------------------- data ----

def run_dir(prefix: str) -> Path:
    return Path(glob.glob(str(RUN_ROOT / f"{prefix}_*"))[0])


def jload(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_run(prefix: str) -> dict:
    d = run_dir(prefix)
    spec = jload(d / "scenario_spec.json")["params"]
    pred = pd.read_csv(d / "predictions" / "test_predictions_hard.csv",
                       usecols=["cell_id", "seg_name", "soh_true", "soh_pred"])
    pred["err"] = (pred.soh_pred - pred.soh_true) * 100
    pred["ds"] = np.where(pred.cell_id.str.startswith("b"), "MIT", "HUST")
    met = jload(d / "metrics" / "metrics.json")["test"]["hard"]
    inter = jload(glob.glob(str(d / "hi_scenario_interaction_*.json"))[0])
    groups = jload(glob.glob(str(d / "synergy_groups_*.json"))[0])
    eff = jload(d / "gates" / "regression_effective_HIs.json")
    ker = jload(d / "gates" / "regression_kernel_HIs.json")
    out = {"n_samp": spec["n_samples"], "pred": pred, "met": met}
    out["dbar"] = np.array([v["cell_level_std_r_mean"] for v in inter["per_hi"].values()])
    out["n_spec"] = inter["n_significant"]
    by = lambda js, key: {js[f"seg_{i}_seg_name"]: js[f"seg_{i}_{key}"] for i in range(6)}
    out["groups"] = {s: sum(len(g) >= 2 for g in v) for s, v in by(groups, "groups").items()}
    out["raw_sel"] = {s: len(v) for s, v in by(eff, "names").items()}
    out["ker_sel"] = {s: int(sum(p >= 0.9 for p in v)) for s, v in by(ker, "probs").items()}
    return out


def rmse(e) -> float:
    return float(np.sqrt(np.mean(np.square(e))))


def load_cycle(path: Path, cycle: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["cycle", "time_s", "voltage_V", "current_A"])
    return df[df.cycle == cycle].sort_values("time_s").reset_index(drop=True)


def charge_curve(name: str):
    cell, cyc = CURVES[name]
    c = load_cycle(DATA_4_HI_ROOT / "clean" / name / f"{cell}.csv", cyc)
    neg = np.where(c.current_A.values < -0.01)[0]
    part = c.iloc[: neg[0] if len(neg) else len(c)]
    keep = part.current_A.values > 0.05
    t, i, v = part.time_s.values[keep], part.current_A.values[keep], part.voltage_V.values[keep]
    dt = np.diff(t)
    inc = 0.5 * (i[1:] + i[:-1]) * dt
    inc[dt > 6 * np.median(dt)] = 0.0
    q = np.concatenate([[0.0], np.cumsum(inc)])
    return q / q[-1], v.astype(float)


def segments(w: float, n: int):
    step = (W_ZONE - w) / (n - 1) if n > 1 else 0.0
    return {z: [(a + k * step, a + k * step + w) for k in range(n)] for z, a in ZONE_START.items()}


def span_curve(q, v, w, centres):
    out = []
    for c in centres:
        m = (q >= c - w / 2) & (q <= c + w / 2)
        out.append((v[m].max() - v[m].min()) * 1000 if m.sum() > 2 else np.nan)
    return np.array(out)


# --------------------------------------------------------------- panels ----

def label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.7, fontweight="bold", color=INK,
            ha="right", va="bottom")


def panel_a(ax, axv, R, curves):
    """Segment placement per width (ax) under the early-life charge curves (axv, shared x)."""
    widths = sorted(R) + PENDING
    n_rows = len(widths)
    for a_ in (ax, axv):
        for z, a in ZONE_START.items():
            a_.axvspan(a, a + W_ZONE, color=ZONE_COLOR[ZKEY[z]], alpha=0.08, lw=0)
        for x in (0.325, 0.35, 0.65, 0.675):
            a_.axvline(x, color=FAINT, lw=0.35, ls=":")
    for z, a in ZONE_START.items():
        axv.text(a + W_ZONE / 2, 1.04, f"{z} zone [{a:.3f}, {a + W_ZONE:.3f}]", transform=axv.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=6.4, color=ZONE_COLOR[ZKEY[z]], fontweight="bold")
    for name, (q, v) in curves.items():
        axv.plot(q, v, color=DATASET_COLOR[name], lw=0.9, label=f"{name} charge, cycle {CURVES[name][1]}")
    axv.set_ylim(3.25, 3.65)
    axv.set_ylabel("V (V)")
    axv.legend(loc="lower right", frameon=False, fontsize=6, handlelength=1.3, ncol=2,
               bbox_to_anchor=(1.0, -0.04))
    axv.tick_params(labelbottom=False)
    yt, yl = [], []
    for r, w in enumerate(widths):
        pend = w not in R
        n = R[w]["n_samp"] if not pend else 2
        col = W_COLOR[w]
        step = (W_ZONE - w) / (n - 1)
        ov = max(0.0, w - step) / w
        y0 = n_rows - r - 1
        lane = 0.8 / n
        for z, segs in segments(w, n).items():
            for k, (s0, s1) in enumerate(segs):
                yk = y0 + 0.1 + k * lane
                ax.add_patch(Rectangle((s0, yk + 0.12 * lane), s1 - s0, 0.76 * lane, facecolor=col,
                                       alpha=0.15 if pend else 0.7, edgecolor=col, lw=0.5,
                                       ls="--" if pend else "-"))
                if k < n - 1 and not pend:
                    o0, o1 = segs[k + 1][0], s1
                    if o1 > o0:
                        ax.add_patch(Rectangle((o0, yk + 0.12 * lane), o1 - o0, 1.76 * lane, facecolor="none",
                                               edgecolor=INK, lw=0.3, hatch="//////"))
        ax.axhline(y0, color=GRID, lw=0.5)
        yt.append(y0 + 0.5)
        yl.append(f"{w:.2f}")
        txt = f"N={n}  $\\Delta$={step:.3f}\noverlap {ov * 100:.0f}%" + ("\nnot run" if pend else "")
        ax.text(1.015, y0 + 0.5, txt, va="center", ha="left", fontsize=6.2, linespacing=1.1,
                color=col if not pend else SUBINK, fontweight="bold" if w == REF else "normal",
                transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    axv.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    for t, w in zip(ax.get_yticklabels(), widths):
        t.set_color(W_COLOR[w]); t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel(r"$w_\mathrm{seg}$")
    ax.set_xlabel("SOC along the charge step (hatched: overlap of adjacent segments)")
    label(axv, "(a)", y=1.16)


def panel_b(ax, curves, widths):
    centres = np.linspace(0.05, 0.95, 181)
    for w in widths:
        for name, (q, v) in curves.items():
            sp = span_curve(q, v, w, centres)
            ax.plot(centres, sp, color=W_COLOR[w], lw=1.1 if w == REF else 0.8,
                    ls="-" if name == "MIT" else (0, (3, 1.5)))
    for z, a in ZONE_START.items():
        ax.axvspan(a, a + W_ZONE, color=ZONE_COLOR[ZKEY[z]], alpha=0.06, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(5, 3000)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Segment center (SOC)")
    ax.set_ylabel("Voltage span in segment (mV)")
    ax.grid(True, which="both", color=GRID, lw=0.35)
    h = [Line2D([], [], color=W_COLOR[w], lw=1.2, label=f"{w:.2f}") for w in widths]
    h += [Line2D([], [], color=SUBINK, lw=0.8, label="MIT"),
          Line2D([], [], color=SUBINK, lw=0.8, ls=(0, (3, 1.5)), label="HUST")]
    ax.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=6,
              columnspacing=0.8, handlelength=1.4)
    ax.text(0.5, 7.5, "LFP plateau: 14-49 mV per segment", ha="center", va="center", fontsize=6,
            color=SUBINK)
    label(ax, "(b)")


def panel_c(ax, R):
    ws = sorted(R)
    x = np.array(ws) * 100
    series = {
        "All": lambda p: rmse(p.err),
        "Charge": lambda p: rmse(p.err[p.seg_name.str.startswith("chg")]),
        "Discharge": lambda p: rmse(p.err[p.seg_name.str.startswith("dis")]),
        "MIT cells": lambda p: rmse(p.err[p.ds == "MIT"]),
        "HUST cells": lambda p: rmse(p.err[p.ds == "HUST"]),
    }
    style = {"All": (INK, "-", "o", 1.4), "Charge": ("#6E8B6B", "--", "^", 0.9),
             "Discharge": ("#A9863F", "--", "v", 0.9), "MIT cells": (DATASET_COLOR["MIT"], ":", "s", 0.9),
             "HUST cells": (DATASET_COLOR["HUST"], ":", "D", 0.9)}
    vals = {}
    for k, f in series.items():
        y = [f(R[w]["pred"]) for w in ws]
        vals[k] = y
        c, ls, m, lw = style[k]
        ax.plot(x, y, color=c, ls=ls, marker=m, ms=3 if k != "All" else 4, lw=lw, label=k,
                mfc="white" if k != "All" else c, mew=0.7, zorder=3 if k == "All" else 2)
    for xi, yi in zip(x, vals["All"]):
        ax.text(xi, yi + 0.07, f"{yi:.2f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold")
    ax.axvspan(REF * 100 - 2, REF * 100 + 2, color=W_COLOR[REF], alpha=0.12, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v / 100:.2f}" for v in x])
    ax.set_xlim(6, 34)
    ax.set_xlabel("Segment width $w_\\mathrm{seg}$")
    ax.set_ylabel("RMSE (%)")
    ax.grid(True, axis="y", color=GRID, lw=0.35)
    axr = ax.twinx()
    r2 = [R[w]["met"]["capacity"]["r2"] for w in ws]
    axr.bar(x, r2, width=3.2, color=FAINT, alpha=0.22, zorder=0)
    axr.set_ylim(0.85, 0.97)
    axr.set_ylabel("$R^2$ (bars)", color=SUBINK)
    axr.tick_params(axis="y", colors=SUBINK)
    ax.set_zorder(axr.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=5.8, handlelength=1.6, ncol=1,
              labelspacing=0.25)
    label(ax, "(a)")
    return vals, r2


def panel_d(ax, R, cax):
    ws = sorted(R)
    M = np.array([[rmse(R[w]["pred"].err[R[w]["pred"].seg_name == s]) for s in SCEN] for w in ws])
    cmap = LinearSegmentedColormap.from_list("w", ["#F4EFE6", "#D9B48A", "#96473A", "#3E1F1A"])
    im = ax.imshow(M, cmap=cmap, vmin=0.6, vmax=2.8, aspect="auto")
    for i in range(M.shape[0]):
        best = np.argmin(M[:, :].T[:, i]) if False else None
        for j in range(M.shape[1]):
            colbest = np.argmin(M[:, j]) == i
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.4,
                    color="white" if M[i, j] > 1.9 else INK, fontweight="bold" if colbest else "normal")
            if colbest:
                ax.add_patch(Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False, ec=INK, lw=0.7))
    ax.set_xticks(range(6))
    ax.set_xticklabels(SCEN_LABEL, rotation=0)
    ax.set_yticks(range(len(ws)))
    ax.set_yticklabels([f"{w:.2f}" for w in ws])
    ax.axvline(2.5, color="white", lw=1.2)
    for t, w in zip(ax.get_yticklabels(), ws):
        t.set_color(W_COLOR[w])
        t.set_fontweight("bold")
    ax.tick_params(length=0)
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Scenario RMSE (%)  (box: best width per scenario)", fontsize=6.2)
    cb.ax.tick_params(labelsize=6)
    label(ax, "(b)")
    return M


def panel_e(ax, R):
    ws = sorted(R)
    cells = sorted(R[REF]["pred"].cell_id.unique())
    per = {w: R[w]["pred"].groupby("cell_id").err.apply(rmse) for w in ws}
    ds = R[REF]["pred"].groupby("cell_id").ds.first()
    x = np.arange(len(ws))
    for c in cells:
        y = [per[w].get(c, np.nan) for w in ws]
        ax.plot(x, y, color=DATASET_COLOR[ds[c]], lw=0.35, alpha=0.45, zorder=1)
        ax.scatter(x, y, s=3, color=DATASET_COLOR[ds[c]], alpha=0.6, lw=0, zorder=2)
    parts = ax.violinplot([per[w].values for w in ws], positions=x, widths=0.55, showextrema=False)
    for b, w in zip(parts["bodies"], ws):
        b.set_facecolor(W_COLOR[w]); b.set_alpha(0.22); b.set_edgecolor("none")
    med = [np.median(per[w].values) for w in ws]
    ax.plot(x, med, color=INK, lw=1.2, marker="_", ms=10, mew=1.4, zorder=4)
    for xi, m in zip(x, med):
        ax.text(xi, 0.97, f"{m:.2f}", transform=ax.get_xaxis_transform(), fontsize=6,
                va="top", ha="center", fontweight="bold")
    wins = {w: int((per[w] < per[REF]).sum()) for w in ws if w != REF}
    for w, n in wins.items():
        xi = ws.index(w)
        ax.text(xi, ax.get_ylim()[1] if False else 0.02, f"{n}/{len(cells)} cells\nlower than 0.20",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.8, color=SUBINK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w:.2f}" for w in ws])
    ax.set_ylabel("Per-cell RMSE (%)")
    ax.set_xlim(-0.5, len(ws) - 0.5)
    ax.grid(True, axis="y", color=GRID, lw=0.35)
    ax.legend(handles=[Line2D([], [], color=DATASET_COLOR[k], lw=0.8, marker="o", ms=2.5, label=k)
                       for k in ("MIT", "HUST")] + [Line2D([], [], color=INK, lw=1.2, label="Median")],
              loc="upper right", bbox_to_anchor=(1.0, 0.9), frameon=False, fontsize=5.8, handlelength=1.4)
    label(ax, "(c)")
    return per, med, wins


def panel_f(ax, R):
    bins = np.arange(0.72, 1.035, 0.02)
    cen = 0.5 * (bins[1:] + bins[:-1]) * 100
    for w in sorted(R):
        p = R[w]["pred"]
        idx = np.digitize(p.soh_true.values, bins) - 1
        ae = p.err.abs().values
        q = np.full((len(cen), 3), np.nan)
        for b in range(len(cen)):
            m = idx == b
            if m.sum() > 200:
                q[b] = np.percentile(ae[m], [25, 50, 75])
        ax.fill_between(cen, q[:, 0], q[:, 2], color=W_COLOR[w], alpha=0.13, lw=0)
        ax.plot(cen, q[:, 1], color=W_COLOR[w], lw=1.2 if w == REF else 0.8, marker="o", ms=1.8,
                label=f"{w:.2f}")
    ax.set_xlabel("True SOH (%)")
    ax.set_ylabel("|Error| (%), median and IQR")
    ax.invert_xaxis()
    ax.grid(True, color=GRID, lw=0.35)
    ax.legend(loc="upper left", frameon=False, fontsize=6, handlelength=1.4)
    label(ax, "(d)")


def panel_g(ax, R):
    ws = sorted(R)
    rng = np.random.default_rng(0)
    for i, w in enumerate(ws):
        d = R[w]["dbar"]
        parts = ax.violinplot([d], positions=[i], widths=0.7, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(W_COLOR[w]); b.set_alpha(0.3); b.set_edgecolor(W_COLOR[w]); b.set_lw(0.5)
        jit = rng.uniform(-0.17, 0.17, len(d))
        ax.scatter(i + jit, d, s=2.5, color=np.where(d >= TAU_EFF, W_COLOR[w], FAINT), lw=0, zorder=3)
        ax.text(i, 1.01, f"{R[w]['n_spec']}/63\nspecific", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6, color=W_COLOR[w], fontweight="bold")
    ax.axhline(TAU_EFF, color=INK, lw=0.6, ls="--")
    ax.text(len(ws) - 0.45, TAU_EFF, r"$\tau_\mathrm{eff}$", fontsize=6.4, va="bottom", ha="right")
    ax.set_xticks(range(len(ws)))
    ax.set_xticklabels([f"{w:.2f}" for w in ws])
    ax.set_ylabel(r"Scenario dependence $\bar D$ of each raw HI")
    ax.set_xlim(-0.6, len(ws) - 0.4)
    ax.grid(True, axis="y", color=GRID, lw=0.35)
    label(ax, "(e)", y=1.12)


def panel_h(ax, R):
    ws = sorted(R)
    nS = len(SCEN)
    bw = 0.8 / len(ws)
    for i, w in enumerate(ws):
        x = np.arange(nS) + (i - (len(ws) - 1) / 2) * bw
        raw = [R[w]["raw_sel"][s] for s in SCEN]
        ker = [R[w]["ker_sel"][s] for s in SCEN]
        grp = [R[w]["groups"][s] for s in SCEN]
        ax.bar(x, raw, bw * 0.92, color=W_COLOR[w], alpha=0.75, lw=0)
        ax.bar(x, ker, bw * 0.92, bottom=raw, color=W_COLOR[w], alpha=0.35, hatch="////",
               edgecolor="white", lw=0)
        ax.scatter(x, grp, marker="_", s=14, color=INK, lw=0.9, zorder=4)
    ax.set_xticks(range(nS))
    ax.set_xticklabels(SCEN_LABEL)
    ax.axvline(2.5, color=FAINT, lw=0.5, ls=":")
    ax.set_ylabel("Number of HIs / groups")
    ax.grid(True, axis="y", color=GRID, lw=0.35)
    h = [Rectangle((0, 0), 1, 1, color=W_COLOR[w], alpha=0.75, label=f"{w:.2f}") for w in ws]
    h += [Rectangle((0, 0), 1, 1, facecolor=FAINT, alpha=0.5, hatch="////", edgecolor="white",
                    label="+ selected kernel"),
          Line2D([], [], marker="_", color=INK, ls="none", ms=6, mew=0.9, label="synergy groups")]
    ax.legend(handles=h, loc="upper left", ncol=3, frameon=False, fontsize=5.8, handlelength=1.2,
              columnspacing=0.7)
    ax.set_ylim(0, max(R[w]["raw_sel"][s] + R[w]["ker_sel"][s] for w in ws for s in SCEN) * 1.32)
    label(ax, "(f)")


# ----------------------------------------------------------------- main ----

def main():
    R = {w: load_run(p) for w, p in RUNS.items()}
    mm = 1 / 25.4
    # 2026-11-05: the segment placement and voltage span panels moved to Figure 1(d)(e)
    fig = plt.figure(figsize=(174 * mm, 122 * mm))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.45,
                  left=0.07, right=0.94, top=0.94, bottom=0.075)
    mid = GridSpecFromSubplotSpec(1, 3, gs[0], width_ratios=[1.0, 1.25, 0.95], wspace=0.55)
    bot = GridSpecFromSubplotSpec(1, 3, gs[1], width_ratios=[1.0, 0.8, 1.45], wspace=0.42)

    ax_c = fig.add_subplot(mid[0])
    vals, r2 = panel_c(ax_c, R)
    dgs = GridSpecFromSubplotSpec(2, 1, mid[1], height_ratios=[1, 0.07], hspace=0.55)
    ax_d = fig.add_subplot(dgs[0])
    M = panel_d(ax_d, R, fig.add_subplot(dgs[1]))
    ax_e = fig.add_subplot(mid[2])
    per, med, wins = panel_e(ax_e, R)

    ax_f = fig.add_subplot(bot[0])
    panel_f(ax_f, R)
    ax_g = fig.add_subplot(bot[1])
    panel_g(ax_g, R)
    ax_h = fig.add_subplot(bot[2])
    panel_h(ax_h, R)

    fig.savefig(OUT / "fig8_window_sweep.png", dpi=300)
    fig.savefig(OUT / "fig8_window_sweep.pdf")

    # numbers for the text
    for w in sorted(R):
        p = R[w]["pred"]
        print(f"w={w:.2f} N={R[w]['n_samp']} n={len(p)} RMSE={rmse(p.err):.3f} MAE={p.err.abs().mean():.3f} "
              f"R2={R[w]['met']['capacity']['r2']:.4f} acc={R[w]['met']['classification']['accuracy']:.4f} "
              + " ".join(f"{k}={vals[k][sorted(R).index(w)]:.2f}" for k in vals)
              + f" spec={R[w]['n_spec']} groups={sum(R[w]['groups'].values())} "
              f"raw_sel_mean={np.mean(list(R[w]['raw_sel'].values())):.1f} ker_sel={sum(R[w]['ker_sel'].values())} "
              f"cell_median={med[sorted(R).index(w)]:.2f}")
    print("scenario RMSE\n", pd.DataFrame(M, index=sorted(R), columns=SCEN).round(2))
    print("cells better than 20%:", wins)


if __name__ == "__main__":
    main()
