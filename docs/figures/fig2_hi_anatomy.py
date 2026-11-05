"""Figure 2 -- anatomy of the four HI categories (2026-11-04).

Replaces fig4_hi_scenario_deviation.py as Figure 2. Every number is computed from real segments with the
HI functions of 4_hi_analysis/tools/hi_compute.py (the extraction code itself), and the life trajectories
and the heat map come from the HI cache used to train the models.

Rows (a)-(d), one per category. Each row = two construction panels on real segments of one cell at four
aging stages (colored by SOH, viridis as in Fig. 1) + the life trajectory of four HIs of that category
over the same cell and scenario (z-score within the cell, rolling median).
  (a) stat   HUST 5-3, dis_mid segment SOC [0.325, 0.525]  (multistage discharge -> current changes)
  (b) diff   MIT b1c2, chg_mid segment SOC [0.475, 0.675]
  (c) LFP    MIT b1c2, chg_lo segment SOC [0.15, 0.35]; NCM TJU CY45-05_1-#24 same scenario for |G|
  (d) morph  HUST 5-3, dis_lo segment SOC [0.15, 0.35], reference = first valid cycle of the cell
(e) Cell-level mean |r(HI, SOH)| of all 63 raw HIs in the six scenarios of each dataset, grouped by
    category, with the scenario spread d_bar of the interaction screening (run JSONs 1031_0117 / 0431 / 0938)
    and the specificity decision (d_bar >= 0.2 and CI > 0).

Segment SOC uses q_ref = charge of the same step one cycle earlier (L = 1, no noise) as in Section 2.2.
Run:  python docs/figures/fig2_hi_anatomy.py   (LFP_SOH_ESTIMATION env)
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / "4_hi_analysis" / "tools"))
from data_directories import DATA_4_HI_ROOT  # noqa: E402
from _style import INK, SUBINK, FAINT, SCEN_NAMES, TJU_COLOR, setup_rcparams  # noqa: E402
import fig1_datasets as f1  # noqa: E402
import hi_compute as H  # noqa: E402

setup_rcparams(6.0)
plt.rcParams.update({"axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.labelsize": 5.2,
                     "ytick.labelsize": 5.2, "axes.labelsize": 5.8, "legend.fontsize": 4.8})

OUT = HERE / "fig2_hi_anatomy"
MM = 1 / 25.4
CAT = ["stat", "diff", "lfp", "morph"]
CAT_COLOR = {"stat": "#3D6E8C", "diff": "#C48F2C", "lfp": "#B14A3C", "morph": "#6B4E8C"}
CAT_NAME = {"stat": "Statistical", "diff": "Differential", "lfp": "LFP-specific", "morph": "Morphological"}
SOH_NORM = Normalize(0.78, 1.0)
CMAP = plt.get_cmap("viridis")
CACHE = DATA_4_HI_ROOT.parent / "4_hi_analysis"
CFG = "qfref_n1-35%_n2-20%_N-2_minpts5_lag-1_noise-3%_ou-200_calib-100_offA-5mA"
RUNS = {"MIT": "1031_0117", "HUST": "1031_0431", "TJU": "1031_0938"}
STAGES = {"MIT": ("b1c2", [100, 1462, 1914, 2158]), "HUST": ("5-3", [100, 825, 1549, 2002]),
          "TJU": ("CY45-05_1-#24", [50, 63, 201, 398])}

_TS: dict = {}
_SOH: dict = {}


# ------------------------------------------------------------------ data --

def series(ds):
    if ds not in _TS:
        _TS[ds] = f1.load_series(ds, STAGES[ds][0])
        c = f1.load_cells(ds)[STAGES[ds][0]]
        _SOH[ds] = dict(zip(c.cycle.astype(int), c.soh))
    return _TS[ds]


def soh(ds, cyc):
    series(ds)
    return float(_SOH[ds][cyc])


def segment(ds, cyc, direction, a, b):
    """(t, V, |I|, SOC) of the points of one cycle whose SOC lies in [a, b]."""
    ts = series(ds)
    part = f1.split(f1.one_cycle(ts, cyc))[0 if direction == "chg" else 1]
    sign = 1 if direction == "chg" else -1
    prev = f1.one_cycle(ts, cyc - 1)
    ref = f1.split(prev)[0 if direction == "chg" else 1] if len(prev) else part
    qref = f1.q_curve(ref, sign)[0][-1]
    i = part.current_A.values * sign
    keep = i > 0.05
    t, v, ii = part.time_s.values[keep], part.voltage_V.values[keep].astype(float), i[keep]
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    dt[dt > 6 * np.median(dt[dt > 0])] = 0.0
    q = np.cumsum(ii * dt) / 3600
    s = q / qref if direction == "chg" else 1 - q / qref
    m = (s >= a) & (s <= b)
    return t[m] - t[m][0], v[m], ii[m], s[m]


def window(ds, cyc, direction, a, b):
    t, v, i, s = segment(ds, cyc, direction, a, b)
    return H.W(v, i, t), (t, v, i, s)


def hi_cache(lfp=True):
    name = f"hi_features_{CFG}{'' if lfp else '_ncm'}.pkl"
    df = pd.read_pickle(CACHE / name)
    df = df[df.dataset.isin(["MIT", "HUST", "TJU"])].copy()
    first = df.sort_values("cycle").groupby(["dataset", "cell_id"]).capacity_Ah.transform("first")
    df["soh"] = df.capacity_Ah / first
    return df


def hi_names():
    f = glob.glob(str(ROOT / f"model_lib/results/p1v2_runs/{RUNS['MIT']}*/hi_scenario_interaction*.json"))[0]
    return list(json.load(open(f, encoding="utf-8"))["per_hi"].keys())


def interaction(ds):
    f = glob.glob(str(ROOT / f"model_lib/results/p1v2_runs/{RUNS[ds]}*/hi_scenario_interaction*.json"))[0]
    return json.load(open(f, encoding="utf-8"))["per_hi"]


def cell_level_abs_r(df, names):
    """Mean over cells of |r(HI, SOH)| within each cell, for every HI x scenario column."""
    cols = [f"{n}_{s}" for n in names for s in SCEN_NAMES]
    X = df[cols].astype(float)
    y = df["soh"].astype(float)
    g = df["cell_id"]
    ok = X.notna()
    cnt = ok.groupby(g).transform("sum")
    xm = X.groupby(g).transform("mean")
    ym = pd.DataFrame({c: y for c in cols}).where(ok)
    ymm = ym.groupby(g).transform("mean")
    xd, yd = X - xm, ym - ymm
    sxy = (xd * yd).groupby(g).sum()
    sxx = (xd ** 2).groupby(g).sum()
    syy = (yd ** 2).groupby(g).sum()
    r = sxy / np.sqrt(sxx * syy)
    n = cnt.groupby(g).first()
    r = r.where(n >= 10)
    return r.abs().mean().values.reshape(len(names), len(SCEN_NAMES))


# --------------------------------------------------------------- helpers --

def sty(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(pad=1.2)
    if grid:
        ax.grid(color="#ECE9E1", lw=0.35, zorder=0)


def ccol(ds, cyc):
    return CMAP(SOH_NORM(soh(ds, cyc)))


def tag(ax, text, color, x=0.02, y=0.97, ha="left", va="top", size=4.8):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.35, alpha=0.92), zorder=20)


def formula(ax, text, color, x=0.98, y=0.04, ha="right", va="bottom", size=4.6):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=INK,
            bbox=dict(boxstyle="square,pad=0.22", fc="#FBFAF6", ec=color, lw=0.4), zorder=20)


def dim(ax, x0, x1, y, text, color, dy=0.0, size=4.4, above=True):
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", lw=0.45, color=color, shrinkA=0, shrinkB=0), zorder=15)
    ax.text(0.5 * (x0 + x1), y + dy, text, ha="center", va="bottom" if above else "top",
            fontsize=size, color=color, zorder=15)


def life(ax, df, ds, scen, names, labels, cat):
    """z-scored life trajectory of four HIs of one cell / scenario against SOH."""
    cell = STAGES[ds][0]
    d = df[(df.dataset == ds) & (df.cell_id.astype(str) == cell)].sort_values("cycle")
    base = plt.matplotlib.colors.to_rgb(CAT_COLOR[cat])
    shades = [tuple(1 - (1 - c) * f for c in base) for f in (1.0, 0.78, 0.58, 0.40)]
    lstyle = ["-", "--", "-.", ":"]
    rng = []
    for k, (n, lab) in enumerate(zip(names, labels)):
        x = d["soh"].values
        y = d[f"{n}_{scen}"].astype(float).values
        ok = np.isfinite(y) & np.isfinite(x)
        x, y = x[ok], y[ok]
        z = (y - np.mean(y)) / (np.std(y) + 1e-12)
        r = np.corrcoef(x, z)[0, 1]
        ax.scatter(x, z, s=0.25, color=shades[k], alpha=0.18, lw=0, rasterized=True, zorder=2)
        zs = pd.Series(z).rolling(61, center=True, min_periods=15).median().values
        rng += [np.nanmin(zs), np.nanmax(zs)]
        ax.plot(x, zs, color=shades[k] if k else base, lw=0.9 if k == 0 else 0.75, ls=lstyle[k], zorder=4,
                label=f"{lab}  r={r:+.2f}")
    ax.set_xlim(1.01, d["soh"].min() - 0.005)
    lo, hi = min(rng), max(rng)
    pad = 0.25 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 1.05 * (hi - lo))
    ax.set_xlabel("SOH")
    ax.set_ylabel("HI z-score")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.1, labelspacing=0.15,
              handletextpad=0.4)
    for cyc in STAGES[ds][1]:
        ax.axvline(soh(ds, cyc), color=ccol(ds, cyc), lw=0.5, ls=(0, (1, 1.2)), zorder=1)
    sty(ax)


def soh_legend(ax, ds, loc="lower right", extra=None):
    hs = [plt.Line2D([], [], color=ccol(ds, c), lw=1.0) for c in STAGES[ds][1]]
    ls = [f"{soh(ds, c):.2f}" for c in STAGES[ds][1]]
    if extra:
        hs += extra[0]
        ls += extra[1]
    ax.legend(hs, ls, loc=loc, frameon=False, title="SOH", title_fontsize=4.6, handlelength=1.1,
              borderaxespad=0.1, labelspacing=0.12, handletextpad=0.35)


# --------------------------------------------------------------- row (a) --

def row_stat(axs, df):
    ds, scen, (d, a, b) = "HUST", "dis_mid", ("dis", 0.325, 0.525)
    cyc0, cycA = STAGES[ds][1][0], STAGES[ds][1][-1]
    ax = axs[0]
    axi = ax.twinx()
    for cyc in STAGES[ds][1]:
        w, (t, v, i, s) = window(ds, cyc, d, a, b)
        c = ccol(ds, cyc)
        ax.plot(t, v, color=c, lw=0.85, zorder=4)
        axi.step(t, i, where="post", color=c, lw=0.55, ls=(0, (2.2, 1.2)), alpha=0.9, zorder=3)
    w0, (t0, v0, i0, s0) = window(ds, cyc0, d, a, b)
    vcw = H._HI["stat_v_mean_cw"](w0)
    p10, p90 = np.percentile(v0, [10, 90])
    ax.axhspan(p10, p90, color=CAT_COLOR["stat"], alpha=0.07, lw=0, zorder=1)
    ax.axhline(vcw, color=CAT_COLOR["stat"], lw=0.5, ls="--", zorder=2)
    ax.text(2, vcw, r"$\bar V_{cw}$", fontsize=4.7, color=CAT_COLOR["stat"], ha="left", va="bottom")
    ax.text(2, p10 + 0.004, r"$V_{p10}$" + "\u2013" + r"$V_{p90}$ band", fontsize=4.3, color=CAT_COLOR["stat"],
            ha="left", va="bottom")
    ax.set_xlabel("Time in segment (s)")
    ax.set_ylabel("Voltage (V)")
    axi.set_ylabel("|I| (A)", color=SUBINK, labelpad=1)
    axi.tick_params(labelsize=5.0, pad=1, length=1.8)
    axi.set_ylim(0, 9.5)
    for sp in ("top",):
        axi.spines[sp].set_visible(False)
    ax.set_ylim(v0.min() - 0.12, v0.max() + 0.03)
    sty(ax)
    # marginal V histogram (stat_v_ent)
    hx = ax.inset_axes([1.24, 0.0, 0.15, 1.0])
    for cyc in (cyc0, cycA):
        wv = window(ds, cyc, d, a, b)[1][1]
        cnt, edges = np.histogram(wv, bins=20)
        hx.barh(0.5 * (edges[1:] + edges[:-1]), cnt / cnt.sum(), height=np.diff(edges), color=ccol(ds, cyc),
                alpha=0.55, lw=0)
        hv = H._HI["stat_v_ent"](H.W(wv, np.ones_like(wv), np.arange(len(wv), dtype=float)))
        hx.text(1.0, -0.02 if cyc == cyc0 else -0.11, f"H={hv:.2f}", transform=hx.transAxes, fontsize=4.2,
                color=INK if cyc == cyc0 else ccol(ds, cyc), ha="right", va="top")
    hx.set_ylim(ax.get_ylim())
    hx.axis("off")
    hx.set_title("20-bin\nV hist.", fontsize=4.2, color=SUBINK, pad=1)
    tag(ax, "HUST 5-3 · dis_mid · SOC 0.33–0.53", CAT_COLOR["stat"], x=0.03, y=0.04, va="bottom")

    # V-I scatter with fits + I-q inset
    ax = axs[1]
    txt = []
    for cyc in STAGES[ds][1]:
        w, (t, v, i, s) = window(ds, cyc, d, a, b)
        c = ccol(ds, cyc)
        ax.scatter(i, v, s=3.0, color=c, lw=0.2, edgecolor="white", alpha=0.85, zorder=3)
        k, b0 = np.polyfit(i, v, 1)
        xx = np.linspace(i.min(), i.max(), 10)
        ax.plot(xx, k * xx + b0, color=c, lw=0.6, zorder=4)
        txt.append((c, H._HI["stat_corr_vi"](w), k))
    for n, (c, r, k) in enumerate(txt):
        ax.text(0.03, 0.04 + 0.075 * (3 - n), f"r(V,I)={r:+.3f}  dV/dI={k * 1e3:.0f} m\u03a9",
                transform=ax.transAxes, fontsize=4.1, color=INK if n == 0 else c, ha="left", va="bottom")
    ax.set_xlabel("|I| (A)")
    ax.set_ylabel("Voltage (V)")
    sty(ax)
    ins = ax.inset_axes([0.62, 0.55, 0.36, 0.34])
    for cyc in (cyc0, cycA):
        w, (t, v, i, s) = window(ds, cyc, d, a, b)
        ins.plot(w.q_rel, i, color=ccol(ds, cyc), lw=0.7)
        sl = H._HI["stat_i_q_slope"](w)
        A = np.column_stack([w.q_rel, np.ones(len(w.q_rel))])
        cf = np.linalg.lstsq(A, i, rcond=None)[0]
        ins.plot(w.q_rel, A @ cf, color=ccol(ds, cyc), lw=0.5, ls="--")
        ins.text(w.q_rel[-1], A[-1] @ cf, f" {sl:.1f}", fontsize=4.0, color=ccol(ds, cyc), va="center")
    ins.set_xlabel("q (Ah)", fontsize=4.4, labelpad=0.5)
    ins.set_ylabel("|I| (A)", fontsize=4.4, labelpad=0.5)
    ins.tick_params(labelsize=4.0, length=1.2, pad=0.6)
    ins.set_title("stat_i_q_slope", fontsize=4.3, pad=1, color=CAT_COLOR["stat"])
    ins.patch.set_alpha(0.9)
    formula(ax, r"$r(V,I)=\frac{\mathrm{cov}(V,|I|)}{\sigma_V\sigma_I}$", CAT_COLOR["stat"], y=0.36, x=0.03,
            ha="left")
    life(axs[2], df, ds, scen, ["stat_corr_vi", "stat_i_q_slope", "stat_v_ent", "stat_v_mean_cw"],
         ["corr_vi", "i_q_slope", "v_ent", "v_mean_cw"], "stat")


# --------------------------------------------------------------- row (b) --

def row_diff(axs, df):
    ds, scen, (d, a, b) = "MIT", "chg_mid", ("chg", 0.475, 0.675)
    cyc0, cycA = STAGES[ds][1][0], STAGES[ds][1][-1]
    ts = series(ds)
    ax = axs[0]
    for cyc in STAGES[ds][1]:
        ch = f1.split(f1.one_cycle(ts, cyc))[0]
        vv, dq = f1.dqdv(ch, 0.004)
        ax.plot(vv, dq, color=ccol(ds, cyc), lw=0.7, zorder=3)
    for cyc, yb in ((cyc0, 0.92), (cycA, 0.84)):
        v = segment(ds, cyc, d, a, b)[1]
        ax.axvspan(v.min(), v.max(), color=ccol(ds, cyc), alpha=0.16, lw=0, zorder=1)
        ax.annotate("", xy=(v.min(), yb), xytext=(v.max(), yb), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="|-|", lw=0.45, color=ccol(ds, cyc), mutation_scale=1.6))
    ax.text(0.5 * (v.min() + v.max()) + 0.012, 0.95, "segment\nwindow", transform=ax.get_xaxis_transform(),
            fontsize=4.3, color=SUBINK, ha="left", va="top")
    ax.set_xlim(3.38, 3.58)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("dQ/dV of full charge (Ah V$^{-1}$)")
    sty(ax)
    soh_legend(ax, ds, loc="upper left")
    tag(ax, "MIT b1c2 · chg_mid · SOC 0.48–0.68", CAT_COLOR["diff"], x=0.98, y=0.04, ha="right",
        va="bottom")

    ax = axs[1]
    peaks = {}
    for cyc in STAGES[ds][1]:
        w = window(ds, cyc, d, a, b)[0]
        vm, dq = w.ica
        c = ccol(ds, cyc)
        ax.plot(vm, dq, color=c, lw=0.8, zorder=4, marker="o", ms=1.4, mew=0)
        pk = int(np.argmax(dq))
        peaks[cyc] = (vm[pk], dq[pk], vm, dq, pk)
    vp, hp, vm, dq, pk = peaks[cyc0]
    fw, asym = H._peak_fwhm_asym(dq, pk, vm)
    half = hp / 2
    li = next((j for j in range(pk, -1, -1) if dq[j] <= half), 0)            # same fallback as hi_compute
    ri = next((j for j in range(pk, len(dq)) if dq[j] <= half), len(dq) - 1)
    ax.hlines(half, vm[li], vm[ri], color=CAT_COLOR["diff"], lw=0.4, ls=":", zorder=5)
    dim(ax, vm[li], vp, half * 0.82, "", CAT_COLOR["diff"], above=False)
    dim(ax, vp, vm[ri], half * 0.82, "", CAT_COLOR["diff"], above=False)
    ax.text(vp, half * 0.82 - 0.15, f"l = {1e3 * (vp - vm[li]):.0f} mV | r = {1e3 * (vm[ri] - vp):.0f} mV",
            fontsize=4.3, color=CAT_COLOR["diff"], ha="center", va="top")
    dim(ax, vm[li], vm[ri], half * 1.03, f"FWHM={1e3 * fw:.0f} mV", CAT_COLOR["diff"])
    ax.plot([vp], [hp], marker="v", ms=3.0, color=ccol(ds, cyc0), mec=INK, mew=0.3, zorder=7)
    ax.text(vp, hp * 1.035, r"$H_p$" + f"={hp:.2f}", fontsize=4.6, ha="center", va="bottom", color=INK)
    vpa, hpa = peaks[cycA][:2]
    ax.plot([vpa], [hpa], marker="v", ms=3.0, color=ccol(ds, cycA), mec=INK, mew=0.3, zorder=7)
    ax.add_patch(FancyArrowPatch((vp, hp), (vpa, hpa), arrowstyle="-|>", mutation_scale=4.5, lw=0.55,
                                 color=INK, connectionstyle="arc3,rad=-0.25", zorder=6))
    ax.text(3.503, 1.75, f"ΔV$_p$={1e3 * (vpa - vp):+.0f} mV (LLI)\n"
            f"ΔH$_p$={hpa - hp:+.2f} (LAM)", fontsize=4.3, ha="left", va="center", color=INK)
    ax.set_ylim(0, hp * 1.22)
    ax.set_xlim(3.462, 3.535)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Segment dQ/dV (Ah V$^{-1}$)")
    sty(ax)
    formula(ax, f"asym = l / r = {asym:.2f}", CAT_COLOR["diff"], x=0.02, y=0.04, ha="left")
    ins = ax.inset_axes([0.74, 0.60, 0.25, 0.32])
    for k, cyc in enumerate((cyc0, cycA)):
        w = window(ds, cyc, d, a, b)[0]
        g = np.abs(H._vd(w)[3])
        cnt, edges = np.histogram(g, bins=10)
        ins.bar(np.arange(10) + (k - 0.5) * 0.4, cnt / cnt.sum(), width=0.4, color=ccol(ds, cyc), lw=0)
        ins.text(0.97, 0.95 - 0.2 * k, f"H={H._HI['diff_dvdq_ent'](w):.2f}", transform=ins.transAxes,
                 fontsize=3.9, color=ccol(ds, cyc), ha="right", va="top")
    ins.set_xticks([0, 9])
    ins.set_xticklabels(["min", "max"])
    ins.tick_params(labelsize=3.8, length=1.0, pad=0.5)
    ins.set_yticks([])
    ins.set_title("|G| 10-bin hist.", fontsize=4.0, pad=1, color=CAT_COLOR["diff"])
    life(axs[2], df, ds, scen, ["diff_dqdv_peak_h", "diff_dqdv_peak_w", "diff_dvdq_ent", "diff_dqdv_peak_v"],
         ["dqdv_peak_h", "dqdv_peak_w", "dvdq_ent", "dqdv_peak_v"], "diff")


# --------------------------------------------------------------- row (c) --

def row_lfp(axs, df):
    ds, scen, (d, a, b) = "MIT", "chg_lo", ("chg", 0.15, 0.35)
    cyc0, cycA = STAGES[ds][1][0], STAGES[ds][1][-1]
    ax = axs[0]
    w0 = window(ds, cyc0, d, a, b)[0]
    qm, vs, g, qt = w0.vq
    dq_b = qm[1] - qm[0]
    flat = np.abs(g) < H.THETA_FLAT
    for j in np.where(flat)[0]:
        ax.axvspan(qm[j] - dq_b / 2, qm[j] + dq_b / 2, color=CAT_COLOR["lfp"], alpha=0.13, lw=0, zorder=1)
    for cyc in STAGES[ds][1]:
        w = window(ds, cyc, d, a, b)[0]
        ax.plot(w.q_rel, w.v, color=ccol(ds, cyc), lw=0.35, alpha=0.5, zorder=2)
        q2, v2 = w.vq[0], w.vq[1]
        ax.plot(q2, v2, color=ccol(ds, cyc), lw=0.85, marker="o", ms=1.1, mew=0, zorder=4)
    vlin = np.interp(qm, [qm[0], qm[-1]], [vs[0], vs[-1]])
    ax.plot(qm, vlin, color=INK, lw=0.5, ls="--", zorder=5)
    qmid = qt / 2
    v_mid = np.interp(qmid, qm, vs)
    v_lmid = np.interp(qmid, [qm[0], qm[-1]], [vs[0], vs[-1]])
    ax.annotate("", xy=(qmid, v_mid), xytext=(qmid, v_lmid),
                arrowprops=dict(arrowstyle="<->", lw=0.5, color=CAT_COLOR["lfp"], shrinkA=0, shrinkB=0), zorder=6)
    ax.text(qmid + 0.003, v_lmid - 0.001, f"v_dev_mid\n{1e3 * (v_mid - v_lmid):+.1f} mV",
            fontsize=4.3, color=CAT_COLOR["lfp"], va="top", ha="left")
    ax.text(qm[-1] - 0.002, vlin[-1] - 0.011, r"chord $\tilde V^{lin}$", fontsize=4.3, ha="right", va="top",
            color=INK)
    ax.set_xlabel("q in segment (Ah)")
    ax.set_ylabel("Voltage (V)")
    sty(ax)
    vals = {k: H._HI[k](w0) for k in ("lfp_plateau_frac", "lfp_nonlin_idx", "lfp_v_flatness")}
    formula(ax, f"plateau_frac = {vals['lfp_plateau_frac']:.2f}\nnonlin_idx = {vals['lfp_nonlin_idx']:.3f}\n"
            f"v_flatness = {vals['lfp_v_flatness']:.3f}", CAT_COLOR["lfp"], x=0.98, y=0.04)
    tag(ax, "MIT b1c2 · chg_lo · SOC 0.15–0.35", CAT_COLOR["lfp"], x=0.02, y=0.97)
    ax.text(qm[np.where(flat)[0][len(np.where(flat)[0]) // 2]], ax.get_ylim()[0] + 0.004,
            "|G| < 0.25 V Ah$^{-1}$", fontsize=4.2, color=CAT_COLOR["lfp"], ha="center", va="bottom")

    ax = axs[1]
    pf = []
    for cyc in STAGES[ds][1]:
        w = window(ds, cyc, d, a, b)[0]
        q2, _, g2, qt2 = w.vq
        ax.plot(q2 / qt2, np.abs(g2), color=ccol(ds, cyc), lw=0.8, marker="o", ms=1.1, mew=0, zorder=4)
        pf.append(H._HI['lfp_plateau_frac'](w))
    for cyc in (STAGES["TJU"][1][0], STAGES["TJU"][1][-1]):
        w = window("TJU", cyc, d, a, b)[0]
        q2, _, g2, qt2 = w.vq
        ax.plot(q2 / qt2, np.abs(g2), color=TJU_COLOR, lw=0.7, ls=(0, (2.5, 1.0)) if cyc != 50 else "-",
                alpha=0.9, zorder=3)
    ax.axhline(H.THETA_FLAT, color=CAT_COLOR["lfp"], lw=0.6, ls="--", zorder=5)
    ax.axhspan(1e-3, H.THETA_FLAT, color=CAT_COLOR["lfp"], alpha=0.07, lw=0, zorder=0)
    ax.text(0.03, H.THETA_FLAT * 0.82, r"$\theta_{flat}$ = 0.25 V Ah$^{-1}$" + "\nplateau bins below",
            fontsize=4.4, color=CAT_COLOR["lfp"], transform=ax.get_yaxis_transform(), va="top")
    ax.set_yscale("log")
    ax.set_ylim(0.012, 30)
    ax.set_xlim(0, 1)
    ax.set_xlabel("q / Q$_s$")
    ax.set_ylabel("|dV/dQ| = |G| (V Ah$^{-1}$)")
    sty(ax)
    hs = [plt.Line2D([], [], color=ccol(ds, c), lw=1.0) for c in STAGES[ds][1]]
    hs.append(plt.Line2D([], [], color=TJU_COLOR, lw=0.8))
    ls = [f"{soh(ds, c):.2f}  ({f:.2f})" for c, f in zip(STAGES[ds][1], pf)] + ["NCM TJU"]
    ax.legend(hs, ls, loc="upper right", frameon=False, title="SOH  (plateau_frac)", title_fontsize=4.4,
              handlelength=1.1, borderaxespad=0.1, labelspacing=0.12, handletextpad=0.35)
    life(axs[2], df, ds, scen, ["lfp_plateau_frac", "lfp_nonlin_idx", "lfp_v_dev_mid", "lfp_plateau_q_onset"],
         ["plateau_frac", "nonlin_idx", "v_dev_mid", "plateau_q_onset"], "lfp")


# --------------------------------------------------------------- row (d) --

def dtw_path(a, b, band=H._DTW_BAND):
    n = len(a)
    dmat = np.abs(a[:, None] - b[None, :])
    acc = np.full((n, n), np.inf)
    acc[0, 0] = dmat[0, 0]
    for j in range(1, min(band + 1, n)):
        acc[0, j] = acc[0, j - 1] + dmat[0, j]
    for i in range(1, n):
        acc[i, 0] = acc[i - 1, 0] + dmat[i, 0]
        for j in range(max(1, i - band), min(n, i + band + 1)):
            acc[i, j] = dmat[i, j] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, path = n - 1, n - 1, [(n - 1, n - 1)]
    while i > 0 or j > 0:
        cand = [(acc[i - 1, j - 1], i - 1, j - 1) if i and j else (np.inf, 0, 0),
                (acc[i - 1, j], i - 1, j) if i else (np.inf, 0, 0),
                (acc[i, j - 1], i, j - 1) if j else (np.inf, 0, 0)]
        _, i, j = min(cand)
        path.append((i, j))
    return dmat, acc, path[::-1]


def row_morph(axs, df):
    ds, scen, (d, a, b) = "HUST", "dis_lo", ("dis", 0.15, 0.35)
    ref_cyc, cycA = 2, STAGES[ds][1][-1]
    tr, vr, ir, _ = segment(ds, ref_cyc, d, a, b)
    ta, va, ia, _ = segment(ds, cycA, d, a, b)
    dtr = np.clip(np.diff(tr, prepend=tr[0]), 0, None)
    dta = np.clip(np.diff(ta, prepend=ta[0]), 0, None)
    ref = H._seg_morph_curves(vr, ir, dtr)
    agd = H._seg_morph_curves(va, ia, dta)
    grid = np.linspace(0, 1, H._MORPH_GRID)
    dmat, acc, path = dtw_path(ref[0], agd[0])
    dtw = H._dtw_distance(ref[0], agd[0])
    fre = H._frechet_distance(ref[0], agd[0])
    kf = int(np.argmax(np.abs(ref[0] - agd[0])))

    ax = axs[0]
    off = 0.12
    cr, ca = "#4D4D4D", ccol(ds, cycA)
    for (i, j) in path[::2]:
        ax.plot([grid[i], grid[j]], [ref[0][i] + off, agd[0][j]], color=CAT_COLOR["morph"], lw=0.3, alpha=0.55,
                zorder=2)
    ax.plot(grid, ref[0] + off, color=cr, lw=1.0, marker="o", ms=1.2, mew=0, zorder=4)
    ax.plot(grid, agd[0], color=ca, lw=1.0, marker="o", ms=1.2, mew=0, zorder=4)
    ax.plot(grid, ref[0], color=cr, lw=0.5, ls=":", zorder=3)
    ax.annotate("", xy=(grid[kf], ref[0][kf]), xytext=(grid[kf], agd[0][kf]),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="#B3261E", shrinkA=0, shrinkB=0), zorder=6)
    ax.text(grid[kf] - 0.02, 0.5 * (ref[0][kf] + agd[0][kf]), f"Fr\u00e9chet\n{1e3 * fre:.0f} mV",
            fontsize=4.3, color="#B3261E", va="center", ha="right",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.text(0.02, ref[0][0] + off + 0.01, f"BOL reference (cycle {ref_cyc}, +{off:.2f} V offset)", fontsize=4.3,
            color=cr, va="bottom")
    ax.text(0.98, agd[0][-1] - 0.012, f"cycle {cycA}, SOH {soh(ds, cycA):.2f}", fontsize=4.3, color=ca,
            ha="right", va="top")
    ax.set_xlabel("Normalized time in segment")
    ax.set_ylabel("Voltage (V)")
    sty(ax)
    tag(ax, "HUST 5-3 · dis_lo · SOC 0.15–0.35", CAT_COLOR["morph"], x=0.98, y=0.97, ha="right")
    formula(ax, f"DTW = {1e3 * dtw:.1f} mV  (50-pt grid, band 5)", CAT_COLOR["morph"], x=0.02, y=0.04, ha="left")

    ax = axs[1]
    cm = LinearSegmentedColormap.from_list("m", ["#E6DDF0", "#C2AEDA", CAT_COLOR["morph"], "#2B1D3B"])
    band = np.abs(np.subtract.outer(np.arange(len(grid)), np.arange(len(grid)))) <= H._DTW_BAND
    show = np.where(band, dmat * 1e3, np.nan)
    ax.imshow(dmat * 1e3, origin="lower", cmap="Greys", alpha=0.18, extent=(0, 1, 0, 1), aspect="auto")
    im = ax.imshow(show, origin="lower", cmap=cm, extent=(0, 1, 0, 1), aspect="auto", interpolation="nearest")
    p = np.array(path)
    ax.plot(grid[p[:, 1]], grid[p[:, 0]], color="#F2C14E", lw=0.9, zorder=5)
    ax.plot(grid[p[:, 1]], grid[p[:, 0]], color=INK, lw=0.25, zorder=6)
    ax.plot([grid[kf]], [grid[kf]], marker="s", ms=2.6, mfc="none", mec="#B3261E", mew=0.6, zorder=7)
    ax.set_xlabel("Aged segment index (norm.)")
    ax.set_ylabel("Reference index (norm.)")
    ax.text(0.97, 0.05, "warping path", color="#7A5A12", fontsize=4.3, ha="right", transform=ax.transAxes)
    ax.text(0.03, 0.93, "Sakoe–Chiba band", color=SUBINK, fontsize=4.3, transform=ax.transAxes)
    cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=4.0, length=1.2, pad=0.6)
    cb.outline.set_linewidth(0.3)
    cb.set_label("|V$_{ref}$−V| (mV)", fontsize=4.4, labelpad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    life(axs[2], df, ds, scen, ["morph_vt_dtw", "morph_ve_dtw", "morph_vt_frec", "morph_vq_dtw"],
         ["vt_dtw", "ve_dtw", "vt_frec", "vq_dtw"], "morph")


# --------------------------------------------------------------- panel e --

def panel_e(fig, gs_e, names, rmat, dbar, sig):
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_e, width_ratios=[3.1, 0.13, 1.15], wspace=0.03)
    ax, axc, axb = (fig.add_subplot(sub[0, k]) for k in range(3))
    cats = [n.split("_")[0] for n in names]
    n = len(names)
    cm = LinearSegmentedColormap.from_list("r", ["#FFFFFF", "#E9E4F0", "#9DB4C7", "#3D6E8C", "#1B3347"])
    im = ax.imshow(rmat, cmap=cm, vmin=0, vmax=0.9, aspect="auto", interpolation="nearest")
    for k in (6, 12):
        ax.axvline(k - 0.5, color="white", lw=1.6)
        ax.axvline(k - 0.5, color=SUBINK, lw=0.3)
    edges = [0] + [i for i in range(1, n) if cats[i] != cats[i - 1]] + [n]
    for e in edges[1:-1]:
        for a_ in (ax, axb):
            a_.axhline(e - 0.5, color=INK, lw=0.45)
    ax.set_yticks(range(n))
    ax.set_yticklabels([x.split("_", 1)[1] for x in names], fontsize=3.5)
    ax.tick_params(axis="y", length=0, pad=0.8)
    ax.set_xticks(range(18))
    ax.set_xticklabels([s.replace("_", " ") for s in SCEN_NAMES] * 3, rotation=90, fontsize=3.7)
    ax.tick_params(axis="x", length=0, pad=0.8)
    for k, ds in enumerate(("MIT", "HUST", "TJU")):
        ax.text(6 * k + 2.5, -1.2, f"{ds} ({'NCM' if ds == 'TJU' else 'LFP'})", ha="center", va="bottom",
                fontsize=5.0, fontweight="bold", color=INK)
    # row max marker per dataset
    for k in range(3):
        blk = rmat[:, 6 * k:6 * k + 6]
        jj = np.nanargmax(np.where(np.isfinite(blk), blk, -1), axis=1)
        ax.scatter(6 * k + jj, np.arange(n), s=0.9, color="#F2C14E", lw=0, zorder=5)
    for s in ax.spines.values():
        s.set_visible(False)

    # category strip
    axc.set_xlim(0, 1)
    axc.set_ylim(n - 0.5, -0.5)
    for a0, a1 in zip(edges[:-1], edges[1:]):
        c = cats[a0]
        axc.add_patch(Rectangle((0, a0 - 0.5), 1, a1 - a0, color=CAT_COLOR[c], lw=0))
        axc.text(0.5, 0.5 * (a0 + a1) - 0.5, f"{CAT_NAME[c]} ({a1 - a0})", rotation=90, ha="center",
                 va="center", fontsize=4.5, color="white", fontweight="bold")
    axc.axis("off")

    # d_bar bars
    axb.set_ylim(n - 0.5, -0.5)
    dcol = {"MIT": "#182E45", "HUST": "#5B7C93", "TJU": TJU_COLOR}
    hgt = 0.27
    for k, ds in enumerate(("MIT", "HUST", "TJU")):
        y = np.arange(n) + (k - 1) * hgt
        axb.barh(y, dbar[:, k], height=hgt, color=dcol[ds], lw=0, alpha=0.85, label=ds)
        m = sig[:, k]
        axb.scatter(dbar[m, k] + 0.025, y[m], s=0.7, color=dcol[ds], lw=0, marker="o")
    axb.axvline(0.2, color="#B3261E", lw=0.5, ls="--")
    axb.text(0.205, -1.1, r"$\tau_{eff}$=0.2", fontsize=4.2, color="#B3261E", va="bottom")
    axb.set_xlim(0, 0.75)
    axb.set_yticks([])
    axb.set_xlabel(r"$\bar d$ = mean$_c$ std$_s$ r$_{c,s}$", fontsize=4.8, labelpad=1)
    axb.tick_params(axis="x", labelsize=4.2, length=1.2, pad=0.6)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    hs, ls = axb.get_legend_handles_labels()
    hs.append(plt.Line2D([], [], marker="o", ls="", ms=1.6, color=SUBINK))
    ls.append("specific")
    axb.legend(hs, ls, loc="lower right", frameon=False, fontsize=4.2, handlelength=0.9, borderaxespad=0.1,
               labelspacing=0.15, handletextpad=0.3)

    pe = ax.get_position()
    cax = fig.add_axes([pe.x0 + 0.02, 0.03, 0.15, 0.007])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=4.0, length=1.2, pad=0.6)
    cb.outline.set_linewidth(0.3)
    cb.set_label("cell-level mean |r(HI, SOH)|", fontsize=4.4, labelpad=1)
    fig.text(pe.x0 + 0.02, 0.05, "\u25cf best scenario of the HI in each dataset", fontsize=4.4,
             color="#B08A1E", va="center")
    return ax, axc


# ------------------------------------------------------------------ main --

def main():
    names = hi_names()
    lfp_df, ncm_df = hi_cache(True), hi_cache(False)
    rmat = np.hstack([cell_level_abs_r(lfp_df[lfp_df.dataset == "MIT"], names),
                      cell_level_abs_r(lfp_df[lfp_df.dataset == "HUST"], names),
                      cell_level_abs_r(ncm_df[ncm_df.dataset == "TJU"], names)])
    inter = {ds: interaction(ds) for ds in RUNS}
    dbar = np.array([[inter[ds][n]["cell_level_std_r_mean"] for ds in RUNS] for n in names])
    sig = np.array([[bool(inter[ds][n]["significant"]) for ds in RUNS] for n in names])

    fig = plt.figure(figsize=(174 * MM, 196 * MM))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[2.05, 1.0], wspace=0.13,
                     left=0.055, right=0.99, top=0.945, bottom=0.09)
    left = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0, 0], hspace=0.52)
    rows = []
    for r in range(4):
        g = GridSpecFromSubplotSpec(1, 3, subplot_spec=left[r, 0], width_ratios=[1.18, 1.0, 0.92], wspace=0.62)
        rows.append([fig.add_subplot(g[0, k]) for k in range(3)])
    row_stat(rows[0], lfp_df)
    row_diff(rows[1], lfp_df)
    row_lfp(rows[2], lfp_df)
    row_morph(rows[3], lfp_df)
    ax_e, axc = panel_e(fig, outer[0, 1], names, rmat, dbar, sig)

    # category bands, row titles and panel letters
    for r, cat in enumerate(CAT):
        p0, p2 = rows[r][0].get_position(), rows[r][2].get_position()
        y1 = p0.y1 + 0.019
        fig.patches.append(Rectangle((p0.x0 - 0.045, y1 - 0.0005), p2.x1 - p0.x0 + 0.045, 0.0028,
                                     transform=fig.transFigure, color=CAT_COLOR[cat], lw=0))
        fig.text(p0.x0 - 0.045, y1 + 0.004, f"({'abcd'[r]})", fontsize=7.5, fontweight="bold", va="bottom")
        fig.text(p0.x0 - 0.012, y1 + 0.0045, f"{CAT_NAME[cat]} HIs", fontsize=6.2, fontweight="bold",
                 color=CAT_COLOR[cat], va="bottom")
        sub = {"stat": "distribution and V–I response of the segment",
               "diff": "peaks of dQ/dV and spread of dV/dQ",
               "lfp": "plateau bins, chord deviation and curvature of V–Q",
               "morph": "shape distance to the BOL segment of the same scenario"}[cat]
        fig.text(p0.x0 + 0.115, y1 + 0.0045, sub, fontsize=5.0, color=SUBINK, va="bottom")
        fig.text(p2.x0, y1 + 0.0045, "life trajectory", fontsize=5.0, color=SUBINK, va="bottom",
                 style="italic")
    pe = ax_e.get_position()
    pc = axc.get_position()
    fig.text(pe.x0 - 0.075, pe.y1 + 0.03, "(e)", fontsize=7.5, fontweight="bold", va="bottom")
    fig.text(pe.x0 - 0.04, pe.y1 + 0.0305, "Scenario dependence of all 63 raw HIs", fontsize=6.2,
             fontweight="bold", va="bottom")
    # SOH colorbar for rows (a)-(d)
    cax = fig.add_axes([0.06, 0.03, 0.22, 0.007])
    cb = fig.colorbar(ScalarMappable(SOH_NORM, CMAP), cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=4.6, length=1.2, pad=0.6)
    cb.outline.set_linewidth(0.3)
    cb.set_label("SOH of the segment cycle (a)–(d)", fontsize=5.0, labelpad=1)
    fig.text(0.31, 0.034, "Dotted verticals in the trajectories mark the four plotted cycles.\n"
             "z-scores within cell and scenario, rolling median of 61 cycles.", fontsize=4.6, color=SUBINK,
             va="center", linespacing=1.3)

    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    print("saved", OUT)
    print("specific per dataset:", sig.sum(axis=0), "rmat max", np.nanmax(rmat))


if __name__ == "__main__":
    main()
