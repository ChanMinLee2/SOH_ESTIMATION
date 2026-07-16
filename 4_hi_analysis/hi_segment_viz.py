"""
hi_segment_viz.py

출력 파일:
  hi_segment_cuts.png           -V vs q_frac 세그먼트 경계 확인 (qfrac 축 전용)
  hi_trend.png                  -Global HI 15종 열화 추이
  hi_segment_trend_stat.png     -N구간 × 통계 HI 열화 추이 (카테고리 A)  [행=시나리오, 열=HI]
  hi_segment_trend_diff.png     -N구간 × 미분 HI 열화 추이 (카테고리 B)
  hi_segment_trend_lfp.png      -N구간 × LFP HI 열화 추이 (카테고리 C)
  hi_segment_trend_morph.png    -N구간 × 형태학적 거리 HI 열화 추이 (카테고리 D)
  hi_overlay_stat.png           -통계 HI 시나리오 오버레이 (카테고리 A)
  hi_overlay_diff.png           -미분 HI 시나리오 오버레이 (카테고리 B)
  hi_overlay_lfp.png            -LFP HI 시나리오 오버레이 (카테고리 C)
  hi_overlay_morph.png          -형태학적 거리 HI 시나리오 오버레이 (카테고리 D)

사용:
  python hi_segment_viz.py
  python hi_segment_viz.py --seg-axis protocol --workers 8
  python hi_segment_viz.py --seg-axis vwindow --axis-config '{"vwindow": {"n_windows": 4}}'
  python hi_segment_viz.py --seg-axis rcs
"""

import argparse
import io
import json
import pickle
import sys
from datetime import date
from pathlib import Path

# Windows 콘솔 cp949 → UTF-8 강제 적용
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hi_correlation import (
    ALL_SEGS,
    GLOBAL_HI_KEYS,
    HI_GROUPS,
    HUST_DIR,
    MIT_DIR,
    load_or_extract,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# ── 폰트 설정 ──────────────────────────────────────────────────────────────────
for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ── 세그먼트 경계선/색상 (hi_segment_cuts용 — qfrac 전용) ──────────────────────
SEG_BOUNDS     = [0.0, 0.4, 0.7, 1.0]
DIS_SEG_COLORS = ["#aed6f1", "#a9dfbf", "#f9e79f"]
DIS_SEG_LABELS = ["SoC 60~100%\n(초반·고전압)", "SoC 30~60%\n(플래토)", "SoC 0~30%\n(후반·저전압)"]
CHG_SEG_COLORS = ["#f9e79f", "#a9dfbf", "#aed6f1"]
CHG_SEG_LABELS = ["SoC 0~30%\n(초반·저전압)", "SoC 30~60%\n(플래토)", "SoC 60~100%\n(후반·CV)"]

DS_COLOR = {"MIT": "#1f77b4", "HUST": "#d55e00"}

# ── 카테고리 메타 ──────────────────────────────────────────────────────────────
CATEGORIES = [
    ("Stat",  "카테고리 A: 통계 기반 (S01–S20)",         "hi_segment_trend_stat.png"),
    ("Diff",  "카테고리 B: 미분 기반 (D01–D20)",         "hi_segment_trend_diff.png"),
    ("LFP",   "카테고리 C: LFP 특징 기반 (L01–L20)",    "hi_segment_trend_lfp.png"),
    ("Morph", "카테고리 D: 형태학적 거리 (M01–M06)",    "hi_segment_trend_morph.png"),
]
OVERLAY_CATEGORIES = [
    ("Stat",  "카테고리 A: 통계 기반 (S01–S20)",         "hi_overlay_stat.png"),
    ("Diff",  "카테고리 B: 미분 기반 (D01–D20)",         "hi_overlay_diff.png"),
    ("LFP",   "카테고리 C: LFP 특징 기반 (L01–L20)",    "hi_overlay_lfp.png"),
    ("Morph", "카테고리 D: 형태학적 거리 (M01–M06)",    "hi_overlay_morph.png"),
]

# ── 방향별 색상 팔레트 (dis=파랑 계열, chg=주황 계열) ──────────────────────────
_DIS_SHADES = ["#AED6F1", "#5DADE2", "#2471A3", "#1A5276", "#0E3460",
               "#7FB3D3", "#3498DB", "#1F618D"]
_CHG_SHADES = ["#FAD7A0", "#F0A500", "#E67E22", "#CA6F1E", "#7D3C98",
               "#F5CBA7", "#DC7633", "#A04000"]

_CAT_PREFIXES = ("stat_", "diff_", "lfp_", "morph_")


def _plain_label(key: str) -> str:
    """HI 컬럼 이름에서 직관적인 스네이크 표기 라벨 추출.

    stat_v_mean_cw_dis_hi → v_mean_cw
    lfp_inflect_v_chg_lo  → inflect_v
    q_dis                 → q_dis   (글로벌 HI는 그대로)
    """
    for prefix in _CAT_PREFIXES:
        if key.startswith(prefix):
            without_prefix = key[len(prefix):]
            parts = without_prefix.rsplit("_", 2)
            return parts[0] if len(parts) == 3 else without_prefix
    return key


# ─────────────────────────────────────────────────────────────────────────────
# 세그먼트 메타 동적 생성 (축 독립)
# ─────────────────────────────────────────────────────────────────────────────

def _build_seg_meta(hi_groups: dict, category: str) -> tuple:
    """HI_GROUPS에서 활성 세그먼트 목록과 시각화 메타(색상/레이블) 동적 생성.

    qfrac(dis_hi/mid/lo, chg_lo/mid/hi)뿐 아니라 protocol(chg_step0, dis_step0 …),
    vwindow(chg_win0, dis_win0 …), rcs 등 임의 축에 대해 작동.

    Returns
    -------
    seg_order    : list[str]        활성 세그먼트 이름 목록 (dis 먼저, chg 다음)
    seg_row_bg   : list[str]        행 배경색 (dis=청, chg=주황)
    seg_row_label: list[str]        행 레이블 (세그먼트 이름 그대로)
    scen_colors  : dict[str, str]   세그먼트별 색상 hex
    scen_labels  : dict[str, str]   세그먼트별 범례 레이블
    """
    seg_order = list(dict.fromkeys(
        g.split(" — ")[0] for g in hi_groups if f" — {category}" in g
    ))
    if not seg_order:
        return [], [], [], {}, {}

    dis_segs = [s for s in seg_order if s.startswith("dis")]
    chg_segs = [s for s in seg_order if s.startswith("chg")]

    seg_row_bg    = ["#eaf4fb" if s.startswith("dis") else "#fef5eb" for s in seg_order]
    seg_row_label = seg_order[:]

    scen_colors: dict[str, str] = {}
    scen_labels: dict[str, str] = {}
    for i, s in enumerate(dis_segs):
        scen_colors[s] = _DIS_SHADES[i % len(_DIS_SHADES)]
        scen_labels[s] = s
    for i, s in enumerate(chg_segs):
        scen_colors[s] = _CHG_SHADES[i % len(_CHG_SHADES)]
        scen_labels[s] = s

    return seg_order, seg_row_bg, seg_row_label, scen_colors, scen_labels


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _load_cell(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    return raw["meta"], raw["cycles"]


def _pick_cycles(cyc_series, n=4):
    cycs = sorted(cyc_series.unique())
    if len(cycs) <= n:
        return cycs
    idx = np.linspace(0, len(cycs) - 1, n, dtype=int)
    return [cycs[i] for i in idx]


def _vq_frac(cycle_df, phase):
    if phase == "discharge":
        grp = cycle_df[cycle_df["current_A"] < -0.1]
    else:
        grp = cycle_df[cycle_df["current_A"] > 0.1]
    if len(grp) < 10:
        return None, None
    grp = grp.sort_values("time_s")
    tc = grp["time_s"].values.astype(float)
    vc = grp["voltage_V"].values.astype(float)
    ic = np.abs(grp["current_A"].values.astype(float))
    dt = np.clip(np.diff(tc, prepend=tc[0]), 0, None)
    q_cum = np.cumsum(ic * dt) / 3600.0
    q_tot = float(q_cum[-1])
    if q_tot < 0.05:
        return None, None
    return q_cum / q_tot, vc


def _cap_from_group(cycle_df):
    vals = cycle_df["capacity_Ah"].dropna()
    return float(vals.iloc[0]) if len(vals) > 0 else np.nan


def _draw_trend_cell(ax, df, hi_key):
    """단일 (세그먼트, HI) 산점도 + 셀별 궤적."""
    any_data = False
    for ds, color in DS_COLOR.items():
        sub = df[df["dataset"] == ds][["cell_id", hi_key, "capacity_Ah"]].dropna()
        if len(sub) == 0:
            continue
        any_data = True
        for _, grp in sub.groupby("cell_id"):
            grp_s = grp.sort_values("capacity_Ah", ascending=False)
            ax.plot(grp_s["capacity_Ah"], grp_s[hi_key],
                    color=color, lw=0.5, alpha=0.18)
        ax.scatter(sub["capacity_Ah"], sub[hi_key],
                   color=color, s=0.6, alpha=0.22)
    if not any_data:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, alpha=0.35)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: 세그먼트 분할 확인 (hi_segment_cuts.png) — qfrac 전용
# ─────────────────────────────────────────────────────────────────────────────

def _draw_seg_panel(ax, cell_df, cell_id, phase, seg_colors, seg_labels, n_cycles):
    rep  = _pick_cycles(cell_df["cycle"], n=n_cycles)
    cmap = matplotlib.colormaps["RdYlGn_r"].resampled(len(rep))

    for si, (x0, x1) in enumerate(zip(SEG_BOUNDS[:-1], SEG_BOUNDS[1:])):
        ax.axvspan(x0, x1, color=seg_colors[si], alpha=0.30, zorder=0)
        ax.text((x0 + x1) / 2, 0.98, seg_labels[si],
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color="dimgray", fontweight="bold", linespacing=1.3)
    for xb in SEG_BOUNDS[1:-1]:
        ax.axvline(xb, color="gray", lw=0.9, ls="--", zorder=1)

    for ci, cyc_num in enumerate(rep):
        cyd = cell_df[cell_df["cycle"] == cyc_num]
        q_frac, vc = _vq_frac(cyd, phase)
        if q_frac is None:
            continue
        cap = _cap_from_group(cyd)
        lbl = f"cycle {cyc_num}  ({cap:.3f} Ah)" if np.isfinite(cap) else f"cycle {cyc_num}"
        ax.plot(q_frac, vc, color=cmap(ci), lw=1.4, alpha=0.85, label=lbl, zorder=2)

    phase_kor = "방전" if phase == "discharge" else "충전"
    ax.set_title(f"{cell_id}  [{phase_kor}]", fontsize=9, fontweight="bold")
    ax.set_xlabel("q_frac  (누적Q / 총Q)", fontsize=8)
    ax.set_ylabel("Voltage (V)", fontsize=8)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7, loc="best", framealpha=0.7)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=7)


def plot_segment_cuts(mit_pkls, hust_pkls, out_path: Path, n_cycles: int = 4):
    selected = []
    for pkls, ds in [(mit_pkls, "MIT"), (hust_pkls, "HUST")]:
        if not pkls:
            continue
        picks = [pkls[0], pkls[-1]] if len(pkls) >= 2 else [pkls[0]]
        for p in picks:
            selected.append((p, ds))

    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.2 * n), squeeze=False)
    fig.suptitle(
        "세그먼트 분할 확인 -V vs q_frac  ( 배경색: SoC 구간 / 선: early→late 사이클 )",
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
# Figure 2: Global HI 열화 추이 (hi_trend.png)
# ─────────────────────────────────────────────────────────────────────────────

def plot_hi_trend(df: pd.DataFrame, out_path: Path):
    """Global HI 15종 전체 -용량 열화 추이."""
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")

    avail = [k for k in GLOBAL_HI_KEYS if k in df.columns]
    ncols = 5
    nrows = (len(avail) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.8, nrows * 3.2),
                              squeeze=False)
    fig.suptitle(
        "Global HI 15종 -열화 추이  (x=Capacity Ah, 선=셀별 궤적)\n"
        "■ 파란 계열=MIT   ■ 주황 계열=HUST",
        fontsize=11, fontweight="bold",
    )

    legend_done = False
    for ai, hi_key in enumerate(avail):
        ax = axes[ai // ncols][ai % ncols]
        for ds, color in DS_COLOR.items():
            sub = df[df["dataset"] == ds][["cell_id", hi_key, "capacity_Ah"]].dropna()
            if len(sub) == 0:
                continue
            for _, grp in sub.groupby("cell_id"):
                grp_s = grp.sort_values("capacity_Ah", ascending=False)
                ax.plot(grp_s["capacity_Ah"], grp_s[hi_key],
                        color=color, lw=0.7, alpha=0.22)
            ax.scatter(sub["capacity_Ah"], sub[hi_key],
                       color=color, s=1.2, alpha=0.28,
                       label=ds if not legend_done else None)

        ax.set_xlabel("Capacity (Ah)", fontsize=8)
        ax.set_ylabel(_plain_label(hi_key), fontsize=8)
        ax.set_title(hi_key, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.4)
        if not legend_done:
            ax.legend(fontsize=8, markerscale=4, loc="best")
            legend_done = True

    for ai in range(len(avail), nrows * ncols):
        axes[ai // ncols][ai % ncols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: 세그먼트별 HI 열화 추이 -카테고리별
# ─────────────────────────────────────────────────────────────────────────────

def plot_segment_hi_trend(df: pd.DataFrame, out_path: Path,
                          category: str, cat_title: str):
    """N구간 × M HI 그리드 -한 카테고리(Stat/Diff/LFP/Morph).

    세그먼트 이름은 HI_GROUPS에서 동적 추출하므로 qfrac/protocol/vwindow/rcs 모두 호환.
    """
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")
    is_morph = (category == "Morph")

    seg_order, seg_row_bg, seg_row_label, _, _ = _build_seg_meta(HI_GROUPS, category)
    seg_keys_list = [
        (seg, HI_GROUPS.get(f"{seg} — {category}", []))
        for seg in seg_order
    ]
    seg_keys_list = [(s, ks) for s, ks in seg_keys_list if ks]
    if not seg_keys_list:
        print(f"  [trend] {category} 카테고리 HI 없음, 건너뜀")
        return

    n_segs = len(seg_keys_list)
    n_his  = len(seg_keys_list[0][1])
    col_labels = [_plain_label(k) for k in seg_keys_list[0][1]]

    cell_w = 5.5 if is_morph else 3.2
    cell_h = 3.8 if is_morph else 3.0
    fig, axes = plt.subplots(
        n_segs, n_his,
        figsize=(n_his * cell_w, n_segs * cell_h),
        squeeze=False,
    )

    morph_note = (
        "\n( y=0: BOL 기준곡선과 동일,  열화 진행 → 거리 증가 )"
        if is_morph else ""
    )
    fig.suptitle(
        f"세그먼트별 HI 열화 추이 -{cat_title}{morph_note}\n"
        "( 행=세그먼트,  열=HI 종류,  x=Capacity Ah )\n"
        "■ 파란 계열=MIT   ■ 주황 계열=HUST",
        fontsize=13, fontweight="bold",
    )

    for ci, lbl in enumerate(col_labels):
        axes[0, ci].set_title(lbl, fontsize=10, fontweight="bold", pad=4)

    for ri, (seg, hi_keys) in enumerate(seg_keys_list):
        bg      = seg_row_bg[ri]    if ri < len(seg_row_bg)    else "#f0f0f0"
        row_lbl = seg_row_label[ri] if ri < len(seg_row_label) else seg

        for ci, hi_key in enumerate(hi_keys):
            ax = axes[ri, ci]
            ax.set_facecolor(bg)
            _draw_trend_cell(ax, df, hi_key)
            ax.set_xlabel("Cap (Ah)", fontsize=8)
            if is_morph:
                ax.set_ylim(bottom=0)
                ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.55, zorder=0)

        y_unit = "dist." if is_morph else _plain_label(hi_keys[0])
        axes[ri, 0].set_ylabel(f"{row_lbl}\n{y_unit}", fontsize=9, labelpad=4)
        for ci in range(1, n_his):
            axes[ri, ci].set_ylabel(_plain_label(hi_keys[ci]), fontsize=8)

    handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=ds)
        for ds, c in DS_COLOR.items()
    ]
    fig.legend(handles=handles, loc="lower right",
               fontsize=9, framealpha=0.85,
               bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: 시나리오 오버레이 -한 서브플랏에 1 HI × N 시나리오
# ─────────────────────────────────────────────────────────────────────────────

def plot_segment_hi_overlay(df: pd.DataFrame, out_path: Path,
                             category: str, cat_title: str) -> None:
    """한 서브플랏에 1 HI의 N개 시나리오 열화 추이를 동시 표시.

    세그먼트 이름은 HI_GROUPS에서 동적 추출하므로 qfrac/protocol/vwindow/rcs 모두 호환.
    dis 계열: 파란 계열 / chg 계열: 주황 계열 (팔레트 자동 배정)
    MIT = 실선 / HUST = 점선
    """
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")

    seg_order, _, _, scen_colors, scen_labels = _build_seg_meta(HI_GROUPS, category)
    if not seg_order:
        print(f"  [overlay] {category} 카테고리 세그먼트 없음, 건너뜀")
        return

    ref_seg   = seg_order[0]
    ref_group = HI_GROUPS.get(f"{ref_seg} — {category}", [])
    if not ref_group:
        print(f"  [overlay] {category} 그룹 키 없음, 건너뜀")
        return

    seg_suffix = f"_{ref_seg}"
    base_names = [k[: -len(seg_suffix)] for k in ref_group if k.endswith(seg_suffix)]
    n_his = len(base_names)
    if n_his == 0:
        return

    is_morph = (category == "Morph")
    ncols    = 3 if is_morph else 5
    nrows    = (n_his + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.0, nrows * 3.2),
        squeeze=False,
    )
    fig.patch.set_facecolor("#f5f5f5")
    fig.suptitle(
        f"시나리오 오버레이 -{cat_title}\n"
        "서브플랏 = 1 HI,  N개 시나리오 동시 표시  (x = Capacity Ah)\n"
        "dis: 파란 계열 ●  |  chg: 주황 계열 ●  |  실선 = MIT  /  점선 = HUST",
        fontsize=12, fontweight="bold",
    )

    cap_all = df["capacity_Ah"].dropna()
    cap_lo  = float(cap_all.quantile(0.01)) if len(cap_all) else 0.0
    cap_hi  = float(cap_all.quantile(0.99)) if len(cap_all) else 2.0
    n_bins  = 40

    def _median_trend(sub_df, key):
        sub = sub_df[["capacity_Ah", key]].dropna()
        if len(sub) < 5:
            return np.array([]), np.array([])
        bins = np.linspace(cap_lo, cap_hi, n_bins + 1)
        mids = (bins[:-1] + bins[1:]) / 2
        meds = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            seg_vals = sub.loc[(sub["capacity_Ah"] >= lo) & (sub["capacity_Ah"] < hi), key]
            meds.append(np.nanmedian(seg_vals) if len(seg_vals) >= 3 else np.nan)
        meds  = np.array(meds, dtype=float)
        valid = np.isfinite(meds)
        return mids[valid], meds[valid]

    for ai, base in enumerate(base_names):
        ax = axes[ai // ncols][ai % ncols]
        ax.set_facecolor("white")
        has_data = False

        for scen in seg_order:
            full_key = f"{base}_{scen}"
            if full_key not in df.columns:
                continue
            color    = scen_colors.get(scen, "#888888")
            scen_lbl = scen_labels.get(scen, scen)

            for ds, ls in [("MIT", "-"), ("HUST", "--")]:
                sub = df[df["dataset"] == ds][
                    ["cell_id", full_key, "capacity_Ah"]
                ].dropna()
                if len(sub) == 0:
                    continue
                has_data = True

                for _, grp in sub.groupby("cell_id"):
                    grp_s = grp.sort_values("capacity_Ah", ascending=False)
                    ax.plot(grp_s["capacity_Ah"], grp_s[full_key],
                            color=color, lw=0.7, alpha=0.18, ls=ls)

                mx, my = _median_trend(sub, full_key)
                if len(mx) >= 2:
                    lbl = (f"{scen_lbl} / {ds}"
                           if (ds == "MIT" and ls == "-") else None)
                    ax.plot(mx, my, color=color, lw=2.0, alpha=0.85,
                            ls=ls, label=lbl, zorder=3)

        if not has_data:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

        plain = _plain_label(f"{base}_{ref_seg}")
        ax.set_title(plain, fontsize=8.5, fontweight="bold", pad=3)
        ax.set_xlabel("Cap (Ah)", fontsize=7)
        ax.set_ylabel(plain, fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, lw=0.3, alpha=0.35)
        ax.set_xlim(cap_lo, cap_hi)

        if is_morph:
            ax.set_ylim(bottom=0)
            ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=0)

    for ai in range(n_his, nrows * ncols):
        axes[ai // ncols][ai % ncols].set_visible(False)

    handles = []
    for scen in seg_order:
        handles.append(
            plt.Line2D([0], [0], color=scen_colors.get(scen, "#888888"), lw=2.2, ls="-",
                       label=scen_labels.get(scen, scen))
        )
    handles += [
        plt.Line2D([0], [0], color="dimgray", lw=2.0, ls="-",  label="MIT (실선)"),
        plt.Line2D([0], [0], color="dimgray", lw=2.0, ls="--", label="HUST (점선)"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=8.5,
               framealpha=0.90, bbox_to_anchor=(1.0, 0.0), ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="세그먼트 분할 시각화 + HI 열화 추이 (카테고리 A–D, 다축 호환)")
    parser.add_argument("--workers",      type=int, default=4,
                        help="HI 추출 병렬 워커 수 (기본: 4)")
    parser.add_argument("--n-cycles",     type=int, default=4,
                        help="세그먼트 cuts 플롯 대표 사이클 수 (기본: 4)")
    parser.add_argument("--force",        action="store_true",
                        help="캐시 무시하고 HI 재추출")
    parser.add_argument("--seg-axis",     type=str, default="qfrac",
                        help="세그멘테이션 축: qfrac|protocol|vwindow|rcs|cluster (기본: qfrac)")
    parser.add_argument("--axis-config",  type=str, default="{}",
                        help="축 파라미터 JSON (예: '{\"max_steps\": 3}')")
    args = parser.parse_args()

    _axis = args.seg_axis
    try:
        _axis_cfg: dict = json.loads(args.axis_config)
    except json.JSONDecodeError as e:
        print(f"[ERROR] --axis-config JSON 파싱 실패: {e}"); return

    # qfrac 이외 축은 HI_GROUPS 재빌드 (모듈 전역 갱신)
    if _axis != "qfrac":
        import sys as _sys
        _sys.path.insert(0, str(PROJECT_ROOT.parent))
        from common.scenario import get_segmenter as _get_seg
        from hi_correlation import _build_hi_groups
        _seg_names = _get_seg(_axis, {_axis: _axis_cfg}).get_spec().scenario_names
        _new_groups, _, _ = _build_hi_groups(_seg_names)
        global HI_GROUPS
        HI_GROUPS = _new_groups
        print(f"[hi_viz] HI_GROUPS 재빌드: {_seg_names}")

    _dir_suffix = f"_{_axis}" if _axis != "qfrac" else ""
    hi_plot_dir = PROJECT_ROOT / "hi_plot" / (date.today().strftime("%m%d") + _dir_suffix)
    hi_plot_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: 세그먼트 분할 확인 (qfrac 전용 — 경계가 q_frac 기준)
    if _axis == "qfrac":
        print("=== 세그먼트 분할 시각화 ===")
        mit_pkls  = sorted(MIT_DIR.glob("*.pkl"))
        hust_pkls = sorted(HUST_DIR.glob("*.pkl"))
        plot_segment_cuts(mit_pkls, hust_pkls,
                          hi_plot_dir / "hi_segment_cuts.png",
                          n_cycles=args.n_cycles)

    # HI 로드
    print("\n=== HI 로드/추출 ===")
    df = load_or_extract(n_workers=args.workers, force=args.force,
                         axis=_axis, axis_cfg=_axis_cfg)
    print(f"  총 사이클: {len(df):,}")

    # Figure 2: Global HI 열화 추이
    print("\n=== Global HI 열화 추이 (15종) ===")
    plot_hi_trend(df, hi_plot_dir / "hi_trend.png")

    # Figure 3-A/B/C/D: 카테고리별 세그먼트 HI 열화 추이
    for cat, cat_title, fname in CATEGORIES:
        print(f"\n=== 세그먼트 HI 추이 -{cat_title} ===")
        plot_segment_hi_trend(df, hi_plot_dir / fname, cat, cat_title)

    # Figure 4-A/B/C/D: 시나리오 오버레이
    for cat, cat_title, fname in OVERLAY_CATEGORIES:
        print(f"\n=== 시나리오 오버레이 -{cat_title} ===")
        plot_segment_hi_overlay(df, hi_plot_dir / fname, cat, cat_title)

    print("\n완료!")


if __name__ == "__main__":
    main()
