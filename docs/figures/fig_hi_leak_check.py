"""fig_hi_leak_check.py — segment-charge leakage check for all 66 raw HIs.

Each segment spans a fixed fraction (w_seg) of the reference capacity, so the
segment charge Q_s (= stat_q_abs) is roughly proportional to cell capacity.
An HI that tracks Q_s is a capacity proxy (leak). For every HI this script
computes |Pearson r| with Q_s and with the measured capacity, separately for
each dataset x scenario, and plots the median (bar) and the min-max range
(whisker) over those groups.

Output: docs/figures/fig_hi_leak_check.png / .pdf and
        docs/figures/_cache/hi_leak_check.csv
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_directories import DATA_4_HI_ROOT  # noqa: E402

AXIS_DIR = "q_frac_ref/n1-35%_n2-20%_N-2_minpts5_lag-1_noise-3%_ou-200_calib-100_offA-5mA"
DATASETS = [("MIT", ""), ("HUST", ""), ("TJU", "_ncm")]
ROWS_PER_CELL = 1500
MIN_N = 200
EXCLUDED = {"stat_q_abs", "stat_energy_seg", "diff_dqdv_area"}
CAT_COLOR = {"stat": "#3A6178", "diff": "#A9863F", "lfp": "#5B8C5A", "morph": "#8A5A9B"}
OUT = Path(__file__).resolve().parent


def load() -> pd.DataFrame:
    frames = []
    for ds, suffix in DATASETS:
        files = sorted(glob.glob(f"{DATA_4_HI_ROOT}/{AXIS_DIR}{suffix}/seg/{ds}/*.pkl"))
        for f in files:
            d = pd.read_pickle(f)
            if len(d) > ROWS_PER_CELL:
                d = d.sample(ROWS_PER_CELL, random_state=0)
            d["ds"] = ds
            frames.append(d)
        print(f"{ds}: {len(files)} cells")
    return pd.concat(frames, ignore_index=True)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    his = [c for c in df.columns if c.split("_")[0] in CAT_COLOR]
    rows = []
    for (ds, sc), g in df.groupby(["ds", "scen"]):
        qs = g["stat_q_abs"].astype(float).to_numpy()
        cap = g["capacity_Ah"].astype(float).to_numpy()
        for h in his:
            x = g[h].astype(float).to_numpy()
            m = np.isfinite(x) & np.isfinite(qs) & np.isfinite(cap)
            if m.sum() < MIN_N or np.std(x[m]) == 0:
                continue
            rows.append((ds, sc, h,
                         abs(np.corrcoef(x[m], qs[m])[0, 1]),
                         abs(np.corrcoef(x[m], cap[m])[0, 1])))
    return pd.DataFrame(rows, columns=["ds", "scen", "hi", "r_q", "r_cap"])


def summarize(r: pd.DataFrame) -> pd.DataFrame:
    s = r.groupby("hi").agg(
        rq_med=("r_q", "median"), rq_min=("r_q", "min"), rq_max=("r_q", "max"),
        rc_med=("r_cap", "median"), groups=("r_q", "size"),
    ).reset_index()
    s["cat"] = s["hi"].str.split("_").str[0]
    s["excluded"] = s["hi"].isin(EXCLUDED)
    return s.sort_values("rq_med", ascending=True).reset_index(drop=True)


def plot(s: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.2, 11.5))
    y = np.arange(len(s))
    colors = [CAT_COLOR[c] for c in s["cat"]]
    bars = ax.barh(y, s["rq_med"], color=colors, height=0.72, zorder=2)
    for b, ex in zip(bars, s["excluded"]):
        if ex:
            b.set_hatch("////")
            b.set_edgecolor("#B22222")
            b.set_linewidth(1.2)
    ax.errorbar(s["rq_med"], y, xerr=[s["rq_med"] - s["rq_min"], s["rq_max"] - s["rq_med"]],
                fmt="none", ecolor="#555555", elinewidth=0.6, capsize=1.5, zorder=3)
    ax.scatter(s["rc_med"], y, marker="D", s=9, color="black", zorder=4,
               label="median |r| with measured capacity")

    kept_max = s.loc[~s["excluded"], "rq_med"].max()
    ex_min = s.loc[s["excluded"], "rq_med"].min()
    ax.axvline(kept_max, color="#555555", ls=":", lw=0.8, zorder=1)
    ax.axvline(ex_min, color="#B22222", ls="--", lw=0.8, zorder=1)
    ax.text(kept_max + 0.01, len(s) - 5.5, f"highest retained\n{kept_max:.2f}", color="#555555", fontsize=6.5)
    ax.text(ex_min - 0.01, len(s) - 7.5, f"lowest excluded\n{ex_min:.2f}", color="#B22222", fontsize=6.5, ha="right")

    labels = [f"{h}  ✕" if ex else h for h, ex in zip(s["hi"], s["excluded"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.3)
    for t, ex in zip(ax.get_yticklabels(), s["excluded"]):
        if ex:
            t.set_color("#B22222")
            t.set_fontweight("bold")
    ax.set_ylim(-0.7, len(s) - 0.3)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("|Pearson r| with segment charge $Q_\\mathrm{s}$ (median over dataset × scenario; whisker = min–max)")
    ax.grid(axis="x", color="#DDDDDD", lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [Patch(color=CAT_COLOR[c], label=n) for c, n in
               [("stat", "stat"), ("diff", "diff"), ("lfp", "LFP-specific"), ("morph", "morphology")]]
    handles.append(Patch(facecolor="white", edgecolor="#B22222", hatch="////", label="excluded (Q_s proxy)"))
    handles.append(plt.Line2D([], [], marker="D", color="black", ls="", ms=3,
                              label="median |r| with capacity"))
    ax.legend(handles=handles, loc="lower right", fontsize=6.5, frameon=False)
    ax.set_title(f"Segment-charge leakage check of all {len(s)} raw HIs "
                 f"(MIT, HUST, TJU; {int(s['groups'].max())} dataset × scenario groups)",
                 loc="left", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_hi_leak_check.{ext}", dpi=300, bbox_inches="tight")
    print("saved", OUT / "fig_hi_leak_check.png")


if __name__ == "__main__":
    df = load()
    r = correlations(df)
    s = summarize(r)
    (OUT / "_cache").mkdir(exist_ok=True)
    s.to_csv(OUT / "_cache" / "hi_leak_check.csv", index=False)
    print(s.sort_values("rq_med", ascending=False).head(8).round(3).to_string(index=False))
    plot(s)
