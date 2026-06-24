"""
hi_segment_viz.py

1. hi_segment_cuts.png  — 실제 V-q_frac 곡선에 세그먼트 경계 표시
2. hi_trend.png         — 주요 HI 값의 용량(SOH) 추이

사용:
  python hi_segment_viz.py
  python hi_segment_viz.py --workers 8 --n-cells 3
"""

import argparse
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from hi_correlation import (
    load_or_extract,
    HI_LABELS, HI_GROUPS,
    SEG_DEFS, CHG_SEG_DEFS,
    MIT_DIR, HUST_DIR,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# ── 한글 폰트 설정 ────────────────────────────────────────────────────────────
for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ── 세그먼트 시각화 설정 ──────────────────────────────────────────────────────
# 방전: q_frac 0→1 = SoC 100→0  (hi / mid / lo 순)
DIS_SEG_COLORS = ["#aed6f1", "#a9dfbf", "#f9e79f"]
DIS_SEG_LABELS = ["SoC 60~100%\n(초반·고전압)", "SoC 30~60%\n(플래토)", "SoC 0~30%\n(후반·저전압)"]

# 충전: q_frac 0→1 = SoC 0→100  (lo / mid / hi 순)
CHG_SEG_COLORS = ["#f9e79f", "#a9dfbf", "#aed6f1"]
CHG_SEG_LABELS = ["SoC 0~30%\n(초반·저전압)", "SoC 30~60%\n(플래토)", "SoC 60~100%\n(후반·CV)"]

SEG_BOUNDS = [0.0, 0.4, 0.7, 1.0]

# HI 추이 플롯에서 보여줄 HI 목록
TREND_HIS = [
    # 방전 Global
    "energy_Wh", "v_mean", "v_end", "v_drop",
    # 방전 SoC 세그먼트 — 평균전압
    "v_mean_s_hi", "v_mean_s_mid", "v_mean_s_lo",
    # 방전 SoC 세그먼트 — 구간 용량
    "q_abs_s_hi", "q_abs_s_mid", "q_abs_s_lo",
    # 충전 Global
    "q_cc_ratio", "chg_v_energy",
    # 충전 SoC 세그먼트 — 평균전압
    "v_mean_chg_s_lo", "v_mean_chg_s_mid", "v_mean_chg_s_hi",
    # 충전 SoC 세그먼트 — 구간 용량
    "q_abs_chg_s_lo", "q_abs_chg_s_mid", "q_abs_chg_s_hi",
    # ICA / DVA
    "ica_peak_h", "chg_ica_peak_h",
]


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _load_cell(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    return raw["meta"], raw["cycles"]


def _pick_cycles(cyc_series, n=3):
    """전체 사이클 중 early / mid / late n개 균등 선택."""
    cycs = sorted(cyc_series.unique())
    if len(cycs) <= n:
        return cycs
    idx = np.linspace(0, len(cycs) - 1, n, dtype=int)
    return [cycs[i] for i in idx]


def _vq_frac(cycle_df, phase):
    """phase 필터 후 (q_frac, voltage) 배열 반환. 데이터 부족 시 None."""
    grp = cycle_df[cycle_df["phase"] == phase]
    if len(grp) < 10:
        return None, None, None
    tc = grp["time_s"].values.astype(float)
    vc = grp["voltage_V"].values.astype(float)
    ic = np.abs(grp["current_A"].values.astype(float))
    dt = np.clip(np.diff(tc, prepend=tc[0]), 0, None)
    q_cum = np.cumsum(ic * dt) / 3600.0
    q_tot = float(q_cum[-1])
    if q_tot < 0.05:
        return None, None, None
    return q_cum / q_tot, vc, q_tot


def _cap_from_group(cycle_df):
    """방전 capacity_Ah 첫 값."""
    dis = cycle_df[cycle_df["phase"] == "discharge"]
    if len(dis) == 0:
        return np.nan
    return float(dis["capacity_Ah"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: 세그먼트 분할 시각화
# ─────────────────────────────────────────────────────────────────────────────

def _draw_seg_panel(ax, cell_df, cell_id, phase, seg_colors, seg_labels, n_cycles):
    """V vs q_frac 패널 그리기."""
    rep = _pick_cycles(cell_df["cycle"], n=n_cycles)
    cmap = matplotlib.colormaps["RdYlGn_r"].resampled(len(rep))

    # 배경 밴드 + 경계선
    for si, (x0, x1) in enumerate(zip(SEG_BOUNDS[:-1], SEG_BOUNDS[1:])):
        ax.axvspan(x0, x1, color=seg_colors[si], alpha=0.30, zorder=0)
        xc = (x0 + x1) / 2
        ax.text(xc, 0.98, seg_labels[si], transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="dimgray",
                fontweight="bold", linespacing=1.3)
    for xb in SEG_BOUNDS[1:-1]:
        ax.axvline(xb, color="gray", lw=0.9, ls="--", zorder=1)

    # 대표 사이클 곡선
    for ci, cyc_num in enumerate(rep):
        cyd = cell_df[cell_df["cycle"] == cyc_num]
        q_frac, vc, q_tot = _vq_frac(cyd, phase)
        if q_frac is None:
            continue
        cap = _cap_from_group(cyd)
        cap_s = f"{cap:.3f} Ah" if np.isfinite(cap) else "?"
        lbl = f"cycle {cyc_num}  ({cap_s})"
        ax.plot(q_frac, vc, color=cmap(ci), lw=1.4, alpha=0.85,
                label=lbl, zorder=2)

    phase_kor = "방전" if phase == "discharge" else "충전"
    ax.set_title(f"{cell_id}  [{phase_kor}]", fontsize=9, fontweight="bold")
    ax.set_xlabel("q_frac  (누적Q / 총Q)", fontsize=8)
    ax.set_ylabel("Voltage (V)", fontsize=8)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7, loc="best", framealpha=0.7)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=7)


def plot_segment_cuts(mit_pkls, hust_pkls, out_path: Path, n_cycles: int = 4):
    # MIT / HUST 각 n_cells개 셀 균등 선택
    selected = []
    for pkls, ds in [(mit_pkls, "MIT"), (hust_pkls, "HUST")]:
        if not pkls:
            continue
        # 셀 2개 선택 (첫번째, 마지막)
        picks = [pkls[0], pkls[-1]] if len(pkls) >= 2 else [pkls[0]]
        for p in picks:
            selected.append((p, ds))

    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.2 * n), squeeze=False)
    fig.suptitle(
        "세그먼트 분할 확인 — V vs q_frac\n"
        "( 배경색: 각 SoC 구간 / 선: early→late 사이클 )",
        fontsize=11, fontweight="bold",
    )

    for ri, (pkl_path, ds) in enumerate(selected):
        meta, cyc_df = _load_cell(pkl_path)
        cell_id = f"{ds}:{meta.get('cell_id', pkl_path.stem)}"

        _draw_seg_panel(axes[ri, 0], cyc_df, cell_id, "discharge",
                        DIS_SEG_COLORS, DIS_SEG_LABELS, n_cycles)
        _draw_seg_panel(axes[ri, 1], cyc_df, cell_id, "charge",
                        CHG_SEG_COLORS, CHG_SEG_LABELS, n_cycles)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: HI 열화 추이
# ─────────────────────────────────────────────────────────────────────────────

DS_COLOR = {"MIT": "#1f77b4", "HUST": "#d55e00"}
DS_CMAP  = {"MIT": "Blues",   "HUST": "Oranges"}


def plot_hi_trend(df: pd.DataFrame, out_path: Path):
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")

    avail = [h for h in TREND_HIS if h in df.columns]
    ncols = 4
    nrows = (len(avail) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.2, nrows * 3.2),
                             squeeze=False)
    axes_flat = axes.flatten()

    fig.suptitle(
        "HI 열화 추이  (x = Capacity Ah,  선 = 셀별 궤적,  색 = MIT/HUST)",
        fontsize=11, fontweight="bold",
    )

    legend_done = False
    for ai, hi_key in enumerate(avail):
        ax = axes_flat[ai]
        for ds, color in DS_COLOR.items():
            sub = df[df["dataset"] == ds][["cell_id", hi_key, "capacity_Ah"]].dropna()
            if len(sub) == 0:
                continue
            # 셀별 궤적
            for _, grp in sub.groupby("cell_id"):
                grp_s = grp.sort_values("capacity_Ah", ascending=False)
                ax.plot(grp_s["capacity_Ah"], grp_s[hi_key],
                        color=color, lw=0.7, alpha=0.25)
            # scatter (위에 얹기)
            ax.scatter(sub["capacity_Ah"], sub[hi_key],
                       color=color, s=1.2, alpha=0.30,
                       label=ds if not legend_done else None)

        ax.set_xlabel("Capacity (Ah)", fontsize=8)
        lbl = HI_LABELS.get(hi_key, hi_key)
        ax.set_ylabel(lbl, fontsize=8)
        ax.set_title(hi_key, fontsize=8.5, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.4)

        if not legend_done:
            ax.legend(fontsize=8, markerscale=4, loc="best")
            legend_done = True

    for ai in range(len(avail), len(axes_flat)):
        axes_flat[ai].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: 세그먼트별 전체 HI 열화 추이 (6 seg × 6 HI 그리드)
# ─────────────────────────────────────────────────────────────────────────────

SEG_GROUP_NAMES = [
    "Dis. SoC 60~100%",
    "Dis. SoC 30~60%",
    "Dis. SoC 0~30%",
    "Chg. SoC 0~30%",
    "Chg. SoC 30~60%",
    "Chg. SoC 60~100%",
]

# 행 구분선 색 (방전 3행 / 충전 3행)
SEG_ROW_COLORS = ["#d6eaf8"] * 3 + ["#fdebd0"] * 3


def _draw_trend_cell(ax, df, hi_key):
    """단일 (segment, HI) 셀 그리기."""
    for ds, color in DS_COLOR.items():
        sub = df[df["dataset"] == ds][["cell_id", hi_key, "capacity_Ah"]].dropna()
        if len(sub) == 0:
            continue
        for _, grp in sub.groupby("cell_id"):
            grp_s = grp.sort_values("capacity_Ah", ascending=False)
            ax.plot(grp_s["capacity_Ah"], grp_s[hi_key],
                    color=color, lw=0.6, alpha=0.20)
        ax.scatter(sub["capacity_Ah"], sub[hi_key],
                   color=color, s=0.8, alpha=0.25)
    ax.tick_params(labelsize=6)
    ax.grid(True, lw=0.3, alpha=0.4)


def plot_segment_hi_trend(df: pd.DataFrame, out_path: Path):
    """6 segments × N HIs 그리드 — 세그먼트별 전체 HI 열화 추이."""
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")

    n_segs = len(SEG_GROUP_NAMES)
    first_keys = HI_GROUPS[SEG_GROUP_NAMES[0]]
    n_his = len(first_keys)

    fig, axes = plt.subplots(
        n_segs, n_his,
        figsize=(n_his * 3.0, n_segs * 2.8),
        squeeze=False,
    )
    fig.suptitle(
        "세그먼트별 HI 열화 추이  (행=SoC 구간, 열=HI 종류,  x=Capacity Ah)\n"
        "■ 파란 계열=MIT   ■ 주황 계열=HUST",
        fontsize=11, fontweight="bold",
    )

    # 열 헤더 (HI 이름) — 첫 번째 세그먼트의 키로 추출
    col_labels = [HI_LABELS.get(k, k) for k in first_keys]

    for ci, lbl in enumerate(col_labels):
        axes[0, ci].set_title(lbl, fontsize=8.5, fontweight="bold", pad=4)

    for ri, gname in enumerate(SEG_GROUP_NAMES):
        hi_keys = HI_GROUPS[gname]

        # 행 배경색 (방전=파랑 / 충전=주황)
        bg = SEG_ROW_COLORS[ri]
        for ci in range(n_his):
            axes[ri, ci].set_facecolor(bg)

        # 행 라벨 (y축 첫 열)
        axes[ri, 0].set_ylabel(
            f"{gname}\n{HI_LABELS.get(hi_keys[0], hi_keys[0])}",
            fontsize=7.5,
        )

        for ci, hi_key in enumerate(hi_keys):
            ax = axes[ri, ci]
            _draw_trend_cell(ax, df, hi_key)
            ax.set_xlabel("Cap (Ah)", fontsize=7)
            if ci > 0:
                ax.set_ylabel(HI_LABELS.get(hi_key, hi_key), fontsize=7)

    # 범례
    handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=ds)
        for ds, c in DS_COLOR.items()
    ]
    fig.legend(handles=handles, loc="upper right",
               fontsize=9, framealpha=0.85, bbox_to_anchor=(1.0, 0.995))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="세그먼트 분할 시각화 + HI 열화 추이")
    parser.add_argument("--workers",  type=int, default=4,
                        help="HI 추출 병렬 워커 수 (기본: 4)")
    parser.add_argument("--n-cycles", type=int, default=4,
                        help="세그먼트 플롯에서 보여줄 대표 사이클 수 (기본: 4)")
    parser.add_argument("--force",    action="store_true",
                        help="캐시 무시하고 HI 재추출")
    args = parser.parse_args()

    mit_pkls  = sorted(MIT_DIR.glob("*.pkl"))
    hust_pkls = sorted(HUST_DIR.glob("*.pkl"))

    hi_plot_dir = PROJECT_ROOT / "hi_plot" / date.today().strftime("%m%d")
    hi_plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: 세그먼트 분할 ──────────────────────────────────────────
    print("=== 세그먼트 분할 시각화 ===")
    plot_segment_cuts(mit_pkls, hust_pkls,
                      hi_plot_dir / "hi_segment_cuts.png",
                      n_cycles=args.n_cycles)

    # ── Figure 2, 3: HI 열화 추이 ────────────────────────────────────────
    print("\n=== HI 로드/추출 ===")
    df = load_or_extract(n_workers=args.workers, force=args.force)
    print(f"  총 사이클: {len(df):,}")

    print("\n=== HI 열화 추이 플롯 ===")
    plot_hi_trend(df, hi_plot_dir / "hi_trend.png")

    print("\n=== 세그먼트별 전체 HI 열화 추이 ===")
    plot_segment_hi_trend(df, hi_plot_dir / "hi_segment_trend.png")
    print("완료!")


if __name__ == "__main__":
    main()
