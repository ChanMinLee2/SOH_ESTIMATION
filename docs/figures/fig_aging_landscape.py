"""Aging landscape figure (draft, candidate main/graphical-abstract figure).

One real LFP test cell (HUST 1-7, a TEST cell of the canonical run
1009_1342_p1v2_baseline_seed42, so the error panels and the curves come from
the same run) shown as a differential-voltage-analysis (DVA) "waterfall" in 3D:
  x = SOC position q_frac within the charge / discharge,
  y = cycle number,
  z (and colour) = log10|dV/dq_frac| of that cycle's curve.
Phase-transition peaks appear as ridges whose position migrates with ageing --
the visual statement of "the same voltage window carries different degradation
information depending on its position". The three SOC-position zones that
define the scenarios are marked along the front edge in the paper-wide zone
triad. Bottom row: the trained model's hard-routing |relative SOH error| per
scenario along the same cell's life.

One ridge per direction is a test-protocol artefact, not degradation: HUST
cycles step the current mid-charge and mid-discharge, which produces a voltage
step and hence a dV/dq spike. It is labelled as such in the figure.

Rendering: PyVista/VTK (smooth shading, specular lighting, parallel
projection) for the 3D panels; tick labels and annotations are drawn by
matplotlib at the 3D anchors' projected screen positions, so typography
matches the other figures.

Data prep (curves -> q_frac grid) is cached in _cache/landscape_hust17.npz by
_prep_landscape() on first run.

Run (figure env): python docs/figures/fig_aging_landscape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv
import cmocean
from scipy.ndimage import gaussian_filter, uniform_filter1d

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from _style import (INK, SUBINK, GRID, Z_LEFT, Z_MID, Z_RIGHT, N1,  # noqa: E402
                     setup_rcparams, label_panel)

setup_rcparams(base_size=7.6)
CACHE = HERE / "_cache"
CELL = "1-7"
RUN = ROOT / "model_lib/results/p1v2_runs/1009_1342_p1v2_baseline_seed42"
# zone split for display = midpoint of each zone overlap (left/mid and mid/right)
ZONE_EDGES = ((N1 + (0.5 - N1 / 2)) / 2, ((1 - N1) + (0.5 + N1 / 2)) / 2)
ZONE_COLS = (Z_LEFT, Z_MID, Z_RIGHT)
ZONES = {"chg": ("chg_lo", "chg_mid", "chg_hi"), "dis": ("dis_hi", "dis_mid", "dis_lo")}
XS, YS, ZS, ZOFF = 3.2, 3.6, 1.25, 0.18
CMAP = cmocean.cm.thermal


# ----------------------------------------------------------------- data ---

def _curve(part, sign, i_thresh=0.05, gap_factor=6.0):
    i_s = part.current_A.values * sign
    keep = i_s > i_thresh
    t, i, v = part.time_s.values[keep], i_s[keep], part.voltage_V.values[keep].astype(float)
    if len(t) < 20:
        return None
    dt = np.diff(t)
    inc = 0.5 * (i[1:] + i[:-1]) * dt
    inc[dt > gap_factor * np.median(dt)] = 0.0  # no charge credited across rest gaps
    q = np.concatenate([[0.0], np.cumsum(inc)])
    return q / q[-1], v


def _prep_landscape(step: int = 8) -> dict:
    path = CACHE / "landscape_hust17.npz"
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    from data_directories import DATA_4_HI_ROOT
    df = pd.read_csv(DATA_4_HI_ROOT / "clean" / "HUST" / f"{CELL}.csv",
                     usecols=["cycle", "time_s", "voltage_V", "current_A"])
    qg = np.linspace(0, 1, 400)
    out = {"chg": [], "dis": []}
    used = []
    for c in np.sort(df.cycle.unique())[::step]:
        g = df[df.cycle == c].sort_values("time_s")
        neg = np.where(g.current_A.values < -0.01)[0]
        if not len(neg):
            continue
        a, b = _curve(g.iloc[:neg[0]], 1), _curve(g.iloc[neg[0]:], -1)
        if a is None or b is None:
            continue
        for k, (q, v) in (("chg", a), ("dis", b)):
            qq, ix = np.unique(q, return_index=True)
            out[k].append(np.interp(qg, qq, v[ix]))
        used.append(c)
    pr = pd.read_csv(RUN / "predictions" / "test_predictions_hard.csv")
    pr = pr[pr.cell_id == CELL].copy()
    pr["err"] = (pr.soh_pred - pr.soh_true).abs() / pr.soh_true * 100
    err = pr.groupby(["seg_name", "cycle"]).err.mean().unstack(0)
    soh = pr.groupby("cycle").soh_true.mean()
    data = dict(q=qg, cycles=np.array(used), V_chg=np.array(out["chg"]), V_dis=np.array(out["dis"]),
                err_cycles=err.index.values, err_cols=np.array(err.columns), err=err.values,
                soh_cycles=soh.index.values, soh=soh.values)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, **data)
    return data


def _surface(d, k):
    q = d["q"]
    keep = (q >= 0.03) & (q <= 0.97)
    qq = q[keep]
    V = gaussian_filter(d[f"V_{k}"][:, keep], (1.6, 1.4))
    H = gaussian_filter(np.log10(np.abs(np.gradient(V, qq, axis=1)) + 0.02), (1.2, 1.0))
    lo, hi = np.percentile(H, 2), np.percentile(H, 99.7)
    return qq, H, lo, hi


# --------------------------------------------------------------- render ---

def _render(d, k, lo, hi, size=(2400, 1800)):
    qq, H, _, _ = _surface(d, k)
    cyc = d["cycles"].astype(float)
    Hn = np.clip((H - lo) / (hi - lo), 0, 1)
    Q, C = np.meshgrid(qq, cyc)
    X = Q * XS
    Y = (C - cyc.min()) / (cyc.max() - cyc.min()) * YS
    Z = Hn * ZS + ZOFF

    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    grid = pv.StructuredGrid(X, Y, Z)
    grid["h"] = Hn.ravel(order="F")
    pl.add_mesh(grid, scalars="h", cmap=CMAP, clim=(0, 1), smooth_shading=True, specular=0.35,
                specular_power=30, ambient=0.3, diffuse=0.75, show_scalar_bar=False)
    for ci in np.linspace(0, len(cyc) - 1, 12).astype(int):
        pl.add_mesh(pv.lines_from_points(np.c_[X[ci], Y[ci], Z[ci] + 0.006]),
                    color="white", line_width=1.3, opacity=0.7)
    # floor + skirt + zone triad along the front edge
    pl.add_mesh(pv.Plane(center=(XS * 0.5 + 0.03, YS / 2, 0), direction=(0, 0, 1),
                         i_size=XS + 0.2, j_size=YS + 0.15), color="#F1EFEA", ambient=0.9, diffuse=0.2)
    skirt = pv.StructuredGrid(np.stack([X[0], X[0]]), np.stack([Y[0], Y[0]]), np.stack([Z[0], np.zeros_like(Z[0])]))
    pl.add_mesh(skirt, color="#A8A8A8", opacity=0.30)
    edges = (0.03, ZONE_EDGES[0], ZONE_EDGES[1], 0.97)
    for j in range(3):
        bar = pv.Box(bounds=(edges[j] * XS + 0.01, edges[j + 1] * XS - 0.01, -0.30, -0.12, 0.0, 0.035))
        pl.add_mesh(bar, color=ZONE_COLS[j], ambient=0.5, diffuse=0.6)
    for xb in ZONE_EDGES:
        pl.add_mesh(pv.Line((xb * XS, -0.30, 0.004), (xb * XS, YS, 0.004)), color="#3A3A3A",
                    line_width=2.0, opacity=0.6)
    pl.enable_parallel_projection()
    pl.camera_position = [(-3.2, -6.0, 5.0), (1.6, 1.7, 0.55), (0, 0, 1)]
    pl.camera.parallel_scale = 2.45

    # world -> image pixel anchors for matplotlib annotation
    ren = pl.renderer
    W, Hpx = size

    def proj(p):
        ren.SetWorldPoint(*p, 1.0)
        ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint()
        return x, Hpx - y

    anchors = {
        "qticks": [(qv, proj((qv * XS, -0.62, 0.0))) for qv in (0.2, 0.4, 0.6, 0.8)],
        "cticks": [(cv, proj((-0.12, (cv - cyc.min()) / (cyc.max() - cyc.min()) * YS, 0.0)))
                   for cv in (1, 500, 1000, 1500) if cyc.min() <= cv <= cyc.max()],
        "qlabel": proj((XS * 0.5, -1.15, 0.0)),
        "clabel": proj((-0.65, YS * 0.5, 0.0)),
        "zones": [proj(((edges[j] + edges[j + 1]) / 2 * XS, -0.21, 0.04)) for j in range(3)],
        "x_axis": (proj((0.03 * XS, -0.32, 0)), proj((0.97 * XS, -0.32, 0))),
        "y_axis": (proj((-0.04, 0, 0)), proj((-0.04, YS, 0))),
    }
    # protocol current-step ridge: largest |dV/dq| between 0.25 and 0.85 at mid-life
    mid = len(cyc) // 2
    sel = (qq > 0.25) & (qq < 0.85)
    jstep = np.where(sel)[0][np.argmax(H[mid, sel])]
    anchors["step"] = proj((qq[jstep] * XS, Y[mid, 0], Z[mid, jstep] + 0.02))
    img = pl.screenshot(return_img=True)
    pl.close()
    return img, anchors


# --------------------------------------------------------------- figure ---

def _annotate_3d(ax, anc, k):
    for qv, (x, y) in anc["qticks"]:
        ax.text(x, y, f"{qv:.1f}", ha="center", va="top", fontsize=5.8, color=SUBINK)
    for cv, (x, y) in anc["cticks"]:
        ax.text(x, y, f"{cv}", ha="right", va="center", fontsize=5.8, color=SUBINK)
    ax.text(*anc["qlabel"], r"SOC position $q_{\mathrm{frac}}$", ha="center", va="top", fontsize=6.6)
    ax.text(*anc["clabel"], "Cycle", ha="right", va="center", fontsize=6.6)
    for name, (x, y) in zip(ZONES[k], anc["zones"]):
        ax.text(x, y, name, ha="center", va="center", fontsize=5.6, color="white", fontweight="bold")
    sx, sy = anc["step"]
    ax.annotate("current step\n(test protocol)", xy=(sx, sy), xytext=(sx + 330, sy - 170) if k == "chg" else (sx - 420, sy - 40),
                fontsize=5.6, color=SUBINK, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=SUBINK, shrinkA=1, shrinkB=2))


def _err_panel(ax, d, k):
    cols = list(d["err_cols"])
    ec = d["err_cycles"]
    for name, col in zip(ZONES[k], ZONE_COLS):
        a = d["err"][:, cols.index(name)]
        ok = ~np.isnan(a)
        sm = uniform_filter1d(a[ok], 31)
        ax.plot(ec[ok], sm, color=col, lw=1.15, label=name)
    ax.set_xlim(ec.min(), ec.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("|SOH error| (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5)
    ax2 = ax.twinx()
    ax2.plot(d["soh_cycles"], d["soh"], color=GRID, lw=2.6, zorder=0)
    ax2.set_ylim(0.70, 1.02)
    ax2.set_ylabel("True SOH", color=SUBINK, fontsize=6.4)
    ax2.tick_params(colors=SUBINK, labelsize=5.8, length=2.5)
    for s in ("top", "left"):
        ax2.spines[s].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    leg = ax.legend(frameon=False, fontsize=5.8, loc="upper left", ncol=3, handlelength=1.4,
                    columnspacing=0.9, borderaxespad=0.1)
    for t in leg.get_texts():
        t.set_color(INK)


def build_figure():
    d = _prep_landscape()
    fig = plt.figure(figsize=(7.2, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.05, 1.0], left=0.02, right=0.985,
                          bottom=0.085, top=0.955, hspace=0.12, wspace=0.10)
    # one shared colour/height scale for both directions so one colourbar reads both panels
    ranges = [_surface(d, k)[2:] for k in ("chg", "dis")]
    lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
    for col, (k, title) in enumerate((("chg", "Charge"), ("dis", "Discharge"))):
        img, anc = _render(d, k, lo, hi)
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, interpolation="lanczos")
        ax.set_xlim(150, img.shape[1] - 120)
        ax.set_ylim(img.shape[0] - 60, 170)
        ax.axis("off")
        _annotate_3d(ax, anc, k)
        ax.set_title(f"({'ab'[col]})  {title}: DVA landscape, HUST {CELL} (LFP)", loc="left",
                     fontsize=7.8, pad=0)
        axe = fig.add_subplot(gs[1, col])
        inner = axe.get_position()
        axe.set_position([inner.x0 + 0.055, inner.y0, inner.width - 0.10, inner.height])
        _err_panel(axe, d, k)
        axe.set_title(f"({'cd'[col]})  Model error by scenario (hard routing)", loc="left",
                      fontsize=7.8, pad=3)

    cax = fig.add_axes([0.40, 0.395, 0.20, 0.011])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(lo, hi))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(r"Height & colour: $\log_{10}|dV/dq_{\mathrm{frac}}|$ (V)", fontsize=6.2, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.outline.set_visible(False)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    for ext, kw in (("png", dict(dpi=600)), ("pdf", dict(dpi=600))):
        p = HERE / f"fig_aging_landscape.{ext}"
        fig.savefig(p, **kw)
        print("saved:", p)
