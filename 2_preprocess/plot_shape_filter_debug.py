"""
plot_shape_filter_debug.py

[진단 전용 — 데이터 수정 없음]

필터7 (V-q_frac 형상 편차) 이 특정 셀에서 어떻게 작동하는지 4패널로 시각화.

패널 구성:
  [0,0] 방전 V-q_frac 오버레이  — 전 사이클, 색상=z-score, 기준곡선 6시점
  [0,1] 충전 V-q_frac 오버레이  — 동일
  [1,0] 방전 z-score 시계열     — z/z_max 추이, 임계선, 이상 구간
  [1,1] 충전 z-score 시계열     — 동일
  [2, :] z-score 분포 히스토그램 — 방전·충전 합산

사용:
  python 2_preprocess/plot_shape_filter_debug.py
  python 2_preprocess/plot_shape_filter_debug.py --cell b1c23 --sigma 7.0 --window 21
  python 2_preprocess/plot_shape_filter_debug.py --sigma 5.0 6.0 7.0   # 여러 임계값 비교

출력:
  2_preprocess/outputs/shape_debug/
    shape_filter_debug_{cell}_s{sigma}_w{window}.png
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

# ── 경로 ────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MIT_DIR      = PROJECT_ROOT / "data_unified" / "MIT"
OUT_DIR      = HERE / "outputs" / "shape_debug"

# ── 한글 폰트 ────────────────────────────────────────────────────────────────
for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────────────────────────────────────
# preprocess.py 와 동일한 shape filter 계산 로직
# ─────────────────────────────────────────────────────────────────────────────

def _qfrac_v(phase_df: pd.DataFrame):
    """phase DataFrame → (q_frac array, voltage array). 데이터 부족 시 (None, None)."""
    if len(phase_df) < 10:
        return None, None
    t  = phase_df["time_s"].values.astype(float)
    v  = phase_df["voltage_V"].values.astype(float)
    i  = np.abs(phase_df["current_A"].values.astype(float))
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    q_cum = np.cumsum(i * dt) / 3600.0
    q_tot = float(q_cum[-1])
    if q_tot < 0.05:
        return None, None
    return q_cum / q_tot, v


def _robust_z(x: np.ndarray) -> np.ndarray:
    """MAD 기반 robust z-score. preprocess.py 와 동일."""
    med   = np.median(x)
    mad   = np.median(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-9 else (np.std(x) or 1.0)
    return (x - med) / scale


def compute_shape_stats(df: pd.DataFrame, phase: str,
                        grid: np.ndarray, window: int) -> dict | None:
    """
    한 phase 의 전 사이클에 대해 형상 편차 지표 계산.

    Returns dict with keys:
      M        : DataFrame (rows=cycle, cols=grid) — 보간된 V 곡선
      ref      : DataFrame — rolling median 기준 곡선
      diff     : DataFrame — M - ref (편차)
      dev      : Series   — 사이클별 RMSE
      max_dev  : Series   — 사이클별 max|ΔV|
      z        : Series   — RMSE 의 robust z
      z_max    : Series   — max_dev 의 robust z
      cycles   : ndarray  — 사이클 번호
    """
    rows, mat = [], []
    for cyc, cyc_df in df[df["phase"] == phase].groupby("cycle"):
        q_frac, v = _qfrac_v(cyc_df)
        if q_frac is None:
            continue
        order    = np.argsort(q_frac)
        v_interp = np.interp(grid, q_frac[order], v[order])
        rows.append(int(cyc))
        mat.append(v_interp)

    if len(rows) < max(window, 5):
        return None

    M       = pd.DataFrame(mat, index=rows)
    ref     = M.rolling(window=window, center=True, min_periods=3).median()
    diff    = M - ref
    dev     = np.sqrt((diff ** 2).mean(axis=1))
    max_dev = diff.abs().max(axis=1)
    z       = pd.Series(_robust_z(dev.values),     index=rows, name="z")
    z_max   = pd.Series(_robust_z(max_dev.values), index=rows, name="z_max")

    return dict(M=M, ref=ref, diff=diff,
                dev=dev, max_dev=max_dev,
                z=z, z_max=z_max,
                cycles=np.array(rows))


# ─────────────────────────────────────────────────────────────────────────────
# 패널 1: V-q_frac 오버레이 (사이클 인덱스 그라데이션)
# ─────────────────────────────────────────────────────────────────────────────

def draw_vqfrac_panel(ax, res: dict, phase: str, sigma: float,
                      grid: np.ndarray, show_outliers: bool = True):
    """
    show_outliers=True  : 원본 — 정상(얇음) + 이상치(두꺼움) 함께 표시
    show_outliers=False : 제거 후 — 정상 사이클만 표시
    """
    M      = res["M"]
    z      = res["z"].values
    z_max  = res["z_max"].values
    cycles = res["cycles"]
    n      = len(cycles)

    z_comb  = np.maximum(z, z_max)
    flagged = z_comb > sigma

    # ── 사이클 위치(시간순) 기반 컬러맵 ─────────────────────────────────
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=n - 1)
    pos_colors = [cmap(norm(i)) for i in range(n)]

    M_vals  = M.values
    ok_idx  = np.where(~flagged)[0]
    bad_idx = np.where(flagged)[0]

    # 정상 사이클 (항상 표시)
    segs_ok = [np.column_stack([grid, M_vals[i]]) for i in ok_idx]
    cols_ok = [pos_colors[i] for i in ok_idx]
    if segs_ok:
        ax.add_collection(LineCollection(
            segs_ok, colors=cols_ok, linewidths=0.4, alpha=0.25, zorder=1))

    # 이상치 사이클 — show_outliers 일 때만 두껍게 추가
    if show_outliers and len(bad_idx):
        segs_bad = [np.column_stack([grid, M_vals[i]]) for i in bad_idx]
        cols_bad = [pos_colors[i] for i in bad_idx]
        ax.add_collection(LineCollection(
            segs_bad, colors=cols_bad, linewidths=1.8, alpha=0.85, zorder=2))

    # ── 컬러바: 사이클 위치 → 실제 사이클 번호 ──────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("사이클 순서 (이른→늦음)", fontsize=9)
    tick_pos = np.linspace(0, n - 1, 5, dtype=int)
    cb.set_ticks(tick_pos)
    cb.set_ticklabels([str(cycles[i]) for i in tick_pos], fontsize=8)

    # y축 범위 (LineCollection 은 autoscale 안 됨)
    all_v = M_vals.ravel()
    ax.set_xlim(0, 1)
    ax.set_ylim(np.nanmin(all_v) - 0.01, np.nanmax(all_v) + 0.01)

    # ── 제거된 사이클 top-10 범례 ─────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    n_bad  = int(flagged.sum())
    n_kept = n - n_bad
    if n_bad > 0:
        sort_idx  = np.argsort(z_comb[flagged])[::-1][:10]
        top_cycs  = cycles[flagged][sort_idx]
        top_zvals = z_comb[flagged][sort_idx]
        n_show    = len(top_cycs)
        row1 = ", ".join(f"{c}(z={v:.1f})" for c, v in
                         zip(top_cycs[:5], top_zvals[:5]))
        row2 = ", ".join(f"{c}(z={v:.1f})" for c, v in
                         zip(top_cycs[5:], top_zvals[5:]))
        lbl  = f"제거 top{n_show} (z>{sigma:.1f})\n{row1}"
        if row2:
            lbl += f"\n{row2}"
        proxy = mlines.Line2D([], [], color="red", lw=2.0, alpha=0.8,
                              ls="--", label=lbl)
        handles.append(proxy)
        labels.append(lbl)

    ax.legend(handles, labels, fontsize=7.0, loc="lower left",
              ncol=1, framealpha=0.88)

    phase_kor = "방전" if phase == "discharge" else "충전"
    if show_outliers:
        subtitle = f"원본  |  전체 {n}개  |  이상치(z>{sigma:.1f}) {n_bad}개 (두꺼운 선)"
    else:
        subtitle = f"제거 후  |  잔존 {n_kept}개 / 전체 {n}개  |  제거 {n_bad}개"
    ax.set_title(
        f"{phase_kor}  V-q_frac  — {subtitle}",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("q_frac", fontsize=10)
    ax.set_ylabel("Voltage (V)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, lw=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 패널 2: z-score 시계열
# ─────────────────────────────────────────────────────────────────────────────

def draw_zscore_panel(ax, res: dict, phase: str, sigma: float):
    cycles  = res["cycles"]
    z       = res["z"].values
    z_max   = res["z_max"].values
    dev     = res["dev"].values * 1000      # → mV
    max_dev = res["max_dev"].values * 1000  # → mV
    flagged = (z > sigma) | (z_max > sigma)

    # ── 주 y축: z-score ──────────────────────────────────────────────────
    ax.plot(cycles, z,     color="#1f77b4", lw=0.9, alpha=0.85,
            label="z  (RMSE 기반)")
    ax.plot(cycles, z_max, color="#ff7f0e", lw=0.9, alpha=0.85, ls="--",
            label="z_max  (max|ΔV| 기반)")
    ax.axhline(sigma, color="red", lw=1.8, ls="--",
               label=f"임계값 σ = {sigma:.1f}")

    # 이상 구간 배경 음영
    ax.fill_between(cycles,
                    sigma, np.maximum(z, z_max),
                    where=flagged,
                    color="red", alpha=0.15, zorder=0)

    # 이상 판정 사이클 포인트
    z_comb = np.maximum(z, z_max)
    bad_z  = z_comb[flagged]
    ax.scatter(cycles[flagged], bad_z,
               color="red", s=18, zorder=5, alpha=0.75,
               label=f"이상 판정 {flagged.sum()}개")

    # ── top-10 이상치: 수직선 + 사이클 번호 ──────────────────────────────
    if flagged.sum() > 0:
        sort_idx = np.argsort(z_comb[flagged])[::-1][:10]
        top_cycs = cycles[flagged][sort_idx]
        xform    = ax.get_xaxis_transform()   # x=데이터좌표, y=axes 비율(0~1)
        for tc in top_cycs:
            ax.axvline(tc, color="darkred", lw=0.9, alpha=0.35,
                       ls=":", zorder=3)
            ax.text(tc, 0.97, str(tc),
                    transform=xform,
                    rotation=90, va="top", ha="center",
                    fontsize=6.5, color="darkred", fontweight="bold")

    # ── 부 y축: 실제 편차 (mV 단위) ───────────────────────────────────────
    ax2 = ax.twinx()
    ax2.plot(cycles, dev,     color="#2ca02c", lw=0.7, alpha=0.45,
             label="RMSE (mV)")
    ax2.plot(cycles, max_dev, color="#9467bd", lw=0.7, alpha=0.45, ls=":",
             label="max|ΔV| (mV)")
    ax2.set_ylabel("편차 (mV)", fontsize=9, color="gray")
    ax2.tick_params(labelsize=8, colors="gray")
    ax2.yaxis.label.set_color("gray")

    # ── 장식 ─────────────────────────────────────────────────────────────
    phase_kor = "방전" if phase == "discharge" else "충전"
    ax.set_title(f"{phase_kor}  z-score 시계열", fontsize=10, fontweight="bold")
    ax.set_xlabel("Cycle", fontsize=10)
    ax.set_ylabel("robust z-score", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, lw=0.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left",
              framealpha=0.85, ncol=2)


# ─────────────────────────────────────────────────────────────────────────────
# 패널 3: z-score 분포 히스토그램
# ─────────────────────────────────────────────────────────────────────────────

def draw_hist_panel(ax, results: dict, sigma: float):
    style = {
        "discharge": ("#1f77b4", "#aec7e8", "방전"),
        "charge":    ("#d62728", "#f5a4a4", "충전"),
    }

    for phase, (c1, c2, kor) in style.items():
        res = results.get(phase)
        if res is None:
            continue
        z_comb = np.maximum(res["z"].values, res["z_max"].values)
        clip   = sigma * 2.5

        ax.hist(np.clip(res["z"].values,     -2, clip),
                bins=80, color=c1, alpha=0.55, label=f"{kor} z")
        ax.hist(np.clip(res["z_max"].values, -2, clip),
                bins=80, color=c2, alpha=0.40, histtype="step",
                lw=1.5, ls="--", label=f"{kor} z_max")

        # sigma 이상 개수 표기
        n_bad = int((z_comb > sigma).sum())
        ax.axvline(sigma, color="red", lw=0, alpha=0)  # dummy for spacing
        ax.text(sigma + 0.05, ax.get_ylim()[1] * 0.7 if ax.get_ylim()[1] > 1 else 1,
                f"{kor} 이상: {n_bad}개", color=c1, fontsize=8)

    ax.axvline(sigma, color="red", lw=2.0, ls="--",
               label=f"임계값 σ={sigma:.1f}")
    ax.set_yscale("log")
    ax.set_xlabel(f"robust z-score  (clip 범위: -2 ~ σ×2.5 = {sigma*2.5:.1f})",
                  fontsize=10)
    ax.set_ylabel("사이클 수 (log scale)", fontsize=10)
    ax.set_title("z-score 분포 — 방전(파랑) / 충전(빨강)", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8.5, framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def _plot_one_cell(cell: str, grid: np.ndarray, sigma: float,
                   window: int) -> dict | None:
    """단일 셀에 대해 PKL 로드 → 계산 → 4행 플롯 저장.

    Returns:
        {"cell": str, "n_total": int,
         "bad_dis": int, "bad_chg": int, "bad_union": int}
        또는 파일 없으면 None
    """
    pkl_path = MIT_DIR / f"{cell}.pkl"
    if not pkl_path.exists():
        print(f"  [SKIP] 파일 없음: {pkl_path}")
        return None

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    df = raw["cycles"]
    n_cyc = df["cycle"].nunique()
    print(f"  로드: {cell}  ({n_cyc}사이클)")

    results = {}
    for phase in ("discharge", "charge"):
        res = compute_shape_stats(df, phase, grid, window)
        results[phase] = res

    # ── 이상치 집계 (OR 합산, preprocess.py 와 동일 로직) ────────────────
    bad_sets = {}
    for phase, res in results.items():
        if res is None:
            bad_sets[phase] = set()
            continue
        z_comb = np.maximum(res["z"].values, res["z_max"].values)
        flagged = z_comb > sigma
        bad_sets[phase] = set(res["cycles"][flagged].tolist())
        n_bad = len(bad_sets[phase])
        n_tot = len(res["cycles"])
        kor   = "방전" if phase == "discharge" else "충전"
        print(f"    {kor}: 이상치 {n_bad}개 / {n_tot}개 ({100*n_bad/n_tot:.1f}%)")

    union_bad = bad_sets["discharge"] | bad_sets["charge"]
    stats = {
        "cell":      cell,
        "n_total":   n_cyc,
        "bad_dis":   len(bad_sets["discharge"]),
        "bad_chg":   len(bad_sets["charge"]),
        "bad_union": len(union_bad),
    }

    fig = plt.figure(figsize=(24, 28))
    fig.suptitle(
        f"필터7 (V-q_frac 형상 편차) 진단 — MIT  {cell}\n"
        f"sigma = {sigma}  |  rolling window = {window}  |  grid = {len(grid)}\n"
        "Row1: 원본(이상치 포함, 두꺼운 선)  |  Row2: 이상치 제거 후  |  Row3: z-score 시계열  |  Row4: 분포",
        fontsize=11, fontweight="bold",
    )

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           height_ratios=[4, 4, 3, 2.5],
                           hspace=0.38, wspace=0.30)

    for ci, phase in enumerate(("discharge", "charge")):
        res = results.get(phase)
        if res is None:
            continue

        ax_before = fig.add_subplot(gs[0, ci])
        draw_vqfrac_panel(ax_before, res, phase, sigma, grid,
                          show_outliers=True)

        ax_after = fig.add_subplot(gs[1, ci])
        draw_vqfrac_panel(ax_after, res, phase, sigma, grid,
                          show_outliers=False)

        ax_z = fig.add_subplot(gs[2, ci])
        draw_zscore_panel(ax_z, res, phase, sigma)

    ax_hist = fig.add_subplot(gs[3, :])
    draw_hist_panel(ax_hist, results, sigma)

    fname    = f"shape_filter_debug_{cell}_s{sigma}_w{window}.png"
    out_path = OUT_DIR / fname
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"    저장: {out_path}")
    plt.close(fig)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="필터7 V-q_frac 형상 편차 진단 시각화")
    parser.add_argument("--cell",   nargs="+", default=None,
                        help="MIT 셀 ID (미지정 시 전체 셀 자동 처리)")
    parser.add_argument("--sigma",  type=float, nargs="+", default=[5.0],
                        help="z-score 임계값(들) (기본: 5.0)")
    parser.add_argument("--window", type=int, default=11,
                        help="rolling median 윈도우 (기본: 11)")
    parser.add_argument("--grid",   type=int, default=100,
                        help="q_frac 보간 격자 수 (기본: 100)")
    args = parser.parse_args()

    # 대상 셀 목록 결정
    if args.cell:
        cells = args.cell
    else:
        cells = sorted(p.stem for p in MIT_DIR.glob("*.pkl"))
        if not cells:
            print(f"[ERR] PKL 파일 없음: {MIT_DIR}")
            return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, args.grid)

    print(f"=== 대상 셀: {len(cells)}개  |  sigma={args.sigma}  |  window={args.window} ===")

    for sigma in args.sigma:
        all_stats: list[dict] = []

        for i, cell in enumerate(cells, 1):
            print(f"\n[{i}/{len(cells)}] {cell}  (sigma={sigma})")
            st = _plot_one_cell(cell, grid, sigma, args.window)
            if st is not None:
                all_stats.append(st)

        # ── 요약 TXT 저장 ────────────────────────────────────────────────
        _write_summary(all_stats, sigma, args.window, args.grid)

    print("\n완료!")


def _write_summary(stats: list[dict], sigma: float,
                   window: int, grid_n: int) -> None:
    """셀별 제거율 및 전체 합산을 TXT 파일로 저장."""
    import datetime

    total_cyc   = sum(s["n_total"]   for s in stats)
    total_dis   = sum(s["bad_dis"]   for s in stats)
    total_chg   = sum(s["bad_chg"]   for s in stats)
    total_union = sum(s["bad_union"] for s in stats)

    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        "필터7 (V-q_frac 형상 편차) 이상치 제거 요약",
        f"sigma={sigma}  |  rolling window={window}  |  grid={grid_n}",
        f"생성: {now}",
        "=" * 70,
        "",
        "[전체 요약]",
        f"  대상 셀       : {len(stats):>6}개",
        f"  전체 사이클   : {total_cyc:>8,}개",
        f"  방전 제거     : {total_dis:>8,}개  ({100*total_dis/total_cyc:.2f}%)",
        f"  충전 제거     : {total_chg:>8,}개  ({100*total_chg/total_cyc:.2f}%)",
        f"  합산 제거(OR) : {total_union:>8,}개  ({100*total_union/total_cyc:.2f}%)",
        f"  잔존 사이클   : {total_cyc - total_union:>8,}개  ({100*(total_cyc-total_union)/total_cyc:.2f}%)",
        "",
        "[셀별 상세]",
        f"{'셀ID':<12} {'전체':>7} {'방전제거':>8} {'충전제거':>8} {'합산제거':>8} {'제거율':>7}",
        "-" * 57,
    ]

    for s in sorted(stats, key=lambda x: x["bad_union"] / max(x["n_total"], 1),
                    reverse=True):
        pct = 100 * s["bad_union"] / s["n_total"] if s["n_total"] else 0.0
        lines.append(
            f"{s['cell']:<12} {s['n_total']:>7,} "
            f"{s['bad_dis']:>8,} {s['bad_chg']:>8,} "
            f"{s['bad_union']:>8,} {pct:>6.2f}%"
        )

    txt_path = OUT_DIR / f"removal_summary_s{sigma}_w{window}.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  [요약 저장] {txt_path}")


if __name__ == "__main__":
    main()
