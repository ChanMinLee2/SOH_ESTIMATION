"""
hi_segment_viz.py

출력 파일:
  hi_segment_cuts.png           — V vs q_frac 세그먼트 경계 확인
  hi_trend.png                  — Global HI 15종 열화 추이
  hi_segment_trend_stat.png     — 6구간 × 15 통계 HI 열화 추이 (카테고리 A)
  hi_segment_trend_diff.png     — 6구간 × 15 미분 HI 열화 추이 (카테고리 B)
  hi_segment_trend_lfp.png      — 6구간 × 15 LFP HI 열화 추이 (카테고리 C)
  hi_segment_trend_morph.png    — 6구간 × 6 형태학적 거리 HI 열화 추이 (카테고리 D)

사용:
  python hi_segment_viz.py
  python hi_segment_viz.py --workers 8 --n-cycles 4
"""

import argparse
import pickle
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hi_correlation import (
    ALL_SEGS,
    GLOBAL_HI_KEYS,
    HI_GROUPS,
    HI_LABELS,
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

# ── 세그먼트 경계선/색상 (hi_segment_cuts용) ──────────────────────────────────
SEG_BOUNDS     = [0.0, 0.4, 0.7, 1.0]
DIS_SEG_COLORS = ["#aed6f1", "#a9dfbf", "#f9e79f"]
DIS_SEG_LABELS = ["SoC 60~100%\n(초반·고전압)", "SoC 30~60%\n(플래토)", "SoC 0~30%\n(후반·저전압)"]
CHG_SEG_COLORS = ["#f9e79f", "#a9dfbf", "#aed6f1"]
CHG_SEG_LABELS = ["SoC 0~30%\n(초반·저전압)", "SoC 30~60%\n(플래토)", "SoC 60~100%\n(후반·CV)"]

# ── 행 배경색 (방전=파랑, 충전=주황) ──────────────────────────────────────────
SEG_ROW_BG = ["#eaf4fb"] * 3 + ["#fef5eb"] * 3   # dis_hi/mid/lo, chg_lo/mid/hi
SEG_ROW_LABEL = [
    "dis_hi\n(SoC 60–100%)",
    "dis_mid\n(SoC 30–60%)",
    "dis_lo\n(SoC 0–30%)",
    "chg_lo\n(SoC 0–40%)",
    "chg_mid\n(SoC 40–70%)",
    "chg_hi\n(SoC 70–100%)",
]

DS_COLOR = {"MIT": "#1f77b4", "HUST": "#d55e00"}

# ── 카테고리 메타 ──────────────────────────────────────────────────────────────
CATEGORIES = [
    ("Stat",  "카테고리 A: 통계 기반 (S01–S15)",          "hi_segment_trend_stat.png"),
    ("Diff",  "카테고리 B: 미분 기반 (D01–D15)",          "hi_segment_trend_diff.png"),
    ("LFP",   "카테고리 C: LFP 특징 기반 (L01–L15)",     "hi_segment_trend_lfp.png"),
    ("Morph", "카테고리 D: 형태학적 거리 (M01–M06)",     "hi_segment_trend_morph.png"),
]


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
    grp = cycle_df[cycle_df["phase"] == phase]
    if len(grp) < 10:
        return None, None
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
    dis = cycle_df[cycle_df["phase"] == "discharge"]
    return float(dis["capacity_Ah"].iloc[0]) if len(dis) > 0 else np.nan


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
# Figure 1: 세그먼트 분할 확인 (hi_segment_cuts.png)
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
        "세그먼트 분할 확인 — V vs q_frac  ( 배경색: SoC 구간 / 선: early→late 사이클 )",
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
    """Global HI 15종 전체 — 용량 열화 추이."""
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")

    avail = [k for k in GLOBAL_HI_KEYS if k in df.columns]
    ncols = 5
    nrows = (len(avail) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.8, nrows * 3.2),
                              squeeze=False)
    fig.suptitle(
        "Global HI 15종 — 열화 추이  (x=Capacity Ah, 선=셀별 궤적)\n"
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

        lbl = HI_LABELS.get(hi_key, hi_key)
        ax.set_xlabel("Capacity (Ah)", fontsize=8)
        ax.set_ylabel(lbl, fontsize=8)
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
# Figure 3: 세그먼트별 HI 열화 추이 — 카테고리별 (6 seg × 15 HI 그리드)
# ─────────────────────────────────────────────────────────────────────────────

def plot_segment_hi_trend(df: pd.DataFrame, out_path: Path,
                          category: str, cat_title: str):
    """6구간 × N HI 그리드 — 한 카테고리(Stat/Diff/LFP/Morph).

    category : "Stat" | "Diff" | "LFP" | "Morph"
    cat_title: 플롯 제목에 표시할 카테고리 이름
    N        : 카테고리별 HI 수 (Stat/Diff/LFP=15, Morph=6)
    """
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")

    is_morph = (category == "Morph")

    # 세그먼트 순서: ALL_SEGS 그대로 (dis_hi/mid/lo, chg_lo/mid/hi)
    seg_keys_list = [
        (seg, HI_GROUPS[f"{seg} — {category}"])
        for _, _, seg, _ in ALL_SEGS
    ]
    n_segs = len(seg_keys_list)        # 6
    n_his  = len(seg_keys_list[0][1])  # Stat/Diff/LFP=15, Morph=6

    # 열 헤더: 첫 번째 세그먼트(dis_hi) 기준 HI 레이블
    col_labels = [HI_LABELS.get(k, k) for k in seg_keys_list[0][1]]

    # Morph는 피처 수가 적으므로 셀 크기 확대
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
        f"세그먼트별 HI 열화 추이 — {cat_title}{morph_note}\n"
        "( 행=SoC 구간,  열=HI 종류,  x=Capacity Ah )\n"
        "■ 파란 계열=MIT   ■ 주황 계열=HUST",
        fontsize=13, fontweight="bold",
    )

    # 열 헤더
    for ci, lbl in enumerate(col_labels):
        axes[0, ci].set_title(lbl, fontsize=10, fontweight="bold", pad=4)

    for ri, (seg, hi_keys) in enumerate(seg_keys_list):
        bg = SEG_ROW_BG[ri]
        row_lbl = SEG_ROW_LABEL[ri]

        for ci, hi_key in enumerate(hi_keys):
            ax = axes[ri, ci]
            ax.set_facecolor(bg)
            _draw_trend_cell(ax, df, hi_key)
            ax.set_xlabel("Cap (Ah)", fontsize=8)
            if is_morph:
                # 거리값은 항상 ≥0 (BOL=0, 열화→증가)
                ax.set_ylim(bottom=0)
                ax.axhline(0, color="gray", lw=0.8, ls="--",
                           alpha=0.55, zorder=0)

        # 행 레이블: 첫 번째 열 y축
        y_unit = "dist." if is_morph else HI_LABELS.get(hi_keys[0], hi_keys[0])
        axes[ri, 0].set_ylabel(
            f"{row_lbl}\n{y_unit}",
            fontsize=9, labelpad=4,
        )
        # 나머지 열: HI 레이블만
        for ci in range(1, n_his):
            axes[ri, ci].set_ylabel(
                HI_LABELS.get(hi_keys[ci], hi_keys[ci]), fontsize=8)

    # 공통 범례
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
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="세그먼트 분할 시각화 + HI 열화 추이 (321-HI 전 카테고리 A–D)")
    parser.add_argument("--workers",  type=int, default=4,
                        help="HI 추출 병렬 워커 수 (기본: 4)")
    parser.add_argument("--n-cycles", type=int, default=4,
                        help="세그먼트 cuts 플롯 대표 사이클 수 (기본: 4)")
    parser.add_argument("--force",    action="store_true",
                        help="캐시 무시하고 HI 재추출")
    args = parser.parse_args()

    mit_pkls  = sorted(MIT_DIR.glob("*.pkl"))
    hust_pkls = sorted(HUST_DIR.glob("*.pkl"))

    hi_plot_dir = PROJECT_ROOT / "hi_plot" / date.today().strftime("%m%d")
    hi_plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: 세그먼트 분할 확인 ─────────────────────────────────────
    print("=== 세그먼트 분할 시각화 ===")
    plot_segment_cuts(mit_pkls, hust_pkls,
                      hi_plot_dir / "hi_segment_cuts.png",
                      n_cycles=args.n_cycles)

    # ── HI 로드 ───────────────────────────────────────────────────────────
    print("\n=== HI 로드/추출 ===")
    df = load_or_extract(n_workers=args.workers, force=args.force)
    print(f"  총 사이클: {len(df):,}")

    # ── Figure 2: Global HI 열화 추이 ────────────────────────────────────
    print("\n=== Global HI 열화 추이 (15종) ===")
    plot_hi_trend(df, hi_plot_dir / "hi_trend.png")

    # ── Figure 3-A/B/C/D: 카테고리별 세그먼트 HI 열화 추이 ───────────────
    for cat, cat_title, fname in CATEGORIES:
        print(f"\n=== 세그먼트 HI 추이 — {cat_title} ===")
        plot_segment_hi_trend(df, hi_plot_dir / fname, cat, cat_title)

    print("\n완료!")


if __name__ == "__main__":
    main()
