"""
seg_diagnose.py

시나리오 분리 축 진단 도구.

  1. 세그먼트 통계 (지정 데이터셋 전체 스캔)
       - 시나리오별 세그먼트 갯수
       - 세그먼트 당 행 갯수 평균
       - 세그먼트 당 시간 길이 평균 (초)

  2. 사이클 시각화 --mode segment (기본)
       [상단] V vs time_s  (충전+방전 연속, 세그먼트 색상 밴드)
       [좌하] V vs Q       (방전 세그먼트 색상 오버레이)
       [우하] V vs Q       (충전 세그먼트 색상 오버레이)

  3. IC 커브 시각화  --mode ic
       [상단좌] 방전 V-Q  멀티사이클 + 창 가로밴드
       [상단우] 충전 V-Q  멀티사이클 + 창 가로밴드
       [하단좌] 방전 dQ/dV vs V  멀티사이클 + 창 세로밴드
       [하단우] 충전 dQ/dV vs V  멀티사이클 + 창 세로밴드
       → vwindow 외 축도 IC 커브만 표시 (경계선 없음)

  --mode all : segment + ic 둘 다 생성

사용 예시:
  python 4_hi_analysis/seg_diagnose.py                        # 자동: 모든 hi_features*.pkl 축
  python 4_hi_analysis/seg_diagnose.py --seg-axis protocol    # 특정 축 지정
  python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --dataset HUST
  python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --mode ic
  python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --mode ic --n-cycles 8
  python 4_hi_analysis/seg_diagnose.py --seg-axis qfrac --cell b1c0 --cycle 10
  python 4_hi_analysis/seg_diagnose.py --seg-axis protocol --no-plot
  python 4_hi_analysis/seg_diagnose.py --seg-axis cluster --no-stats
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP_DIR     = Path(__file__).resolve().parent
MIT_DIR      = PROJECT_ROOT / "_4_data_hi" / "clean" / "MIT"
HUST_DIR     = PROJECT_ROOT / "_4_data_hi" / "clean" / "HUST"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _font
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

# 최대 12개 시나리오를 커버하는 팔레트 (색조 구분 최대화)
_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#2980b9",
    "#27ae60", "#c0392b", "#8e44ad", "#16a085",
]


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _add_phase(df):
    df = df.copy()
    cur = df["current_A"]
    df["phase"] = "rest"
    df.loc[cur >  0.01, "phase"] = "charge"
    df.loc[cur < -0.01, "phase"] = "discharge"
    return df


def _build_arrays(rows):
    """(v, i_mag, t, dt, q_cum) 반환."""
    v  = rows["voltage_V"].values.astype(float)
    i  = np.abs(rows["current_A"].values.astype(float))
    t  = rows["time_s"].values.astype(float)
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    q  = np.cumsum(i * dt) / 3600.0
    return v, i, t, dt, q


def _t_range_from_q(t_full: np.ndarray, q_full: np.ndarray, q_seg: np.ndarray):
    """세그먼트 Q 범위 → 원본 t 배열에서 시작/끝 시각 추출.

    q_full이 단조 증가인 경우 searchsorted로 빠르게 찾는다.
    """
    if len(q_seg) == 0 or len(q_full) == 0:
        return None, None
    q0, q1 = float(q_seg[0]), float(q_seg[-1])
    idx0 = int(np.searchsorted(q_full, q0, side="left"))
    idx1 = int(np.searchsorted(q_full, q1, side="right")) - 1
    idx0 = np.clip(idx0, 0, len(t_full) - 1)
    idx1 = np.clip(idx1, idx0, len(t_full) - 1)
    return float(t_full[idx0]), float(t_full[idx1])


# ─────────────────────────────────────────────────────────────────────────────
# 1. 세그먼트 통계 수집
# ─────────────────────────────────────────────────────────────────────────────

def collect_stats(pkl_dir: Path, segmenter, spec_names: list) -> dict:
    """pkl_dir 전체 셀 스캔 → 시나리오별 통계 dict."""
    files = sorted(pkl_dir.glob("*.pkl"))
    if not files:
        print(f"  [경고] {pkl_dir} 에 pkl 파일 없음")
        return {n: {"count": 0, "rows": [], "dur_s": []} for n in spec_names}

    stats = {n: {"count": 0, "rows": [], "dur_s": []} for n in spec_names}
    _empty = np.empty(0, dtype=float)

    for f in tqdm(files, desc=f"  [{pkl_dir.name}] 통계"):
        try:
            with open(f, "rb") as fh:
                raw = pickle.load(fh)
        except Exception:
            continue

        df_all = raw.get("cycles")
        if df_all is None:
            continue
        if "phase" not in df_all.columns:
            df_all = _add_phase(df_all)

        cell_id = raw.get("meta", {}).get("cell_id", f.stem)

        for cyc, grp in df_all.groupby("cycle"):
            if int(cyc) == 0:
                continue
            dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
            chg = grp[grp["phase"] == "charge"].sort_values("time_s")
            if len(dis) < 30:
                continue

            v_d, i_d, _, dt_d, q_d = _build_arrays(dis)
            for rec in segmenter.iter_segments(cell_id, int(cyc), v_d, i_d, dt_d, q_d):
                sn = rec.meta.get("seg_name") or spec_names[rec.scenario_id]
                if sn in stats:
                    stats[sn]["count"] += 1
                    stats[sn]["rows"].append(len(rec.v))
                    stats[sn]["dur_s"].append(float(np.sum(rec.dt)))

            if len(chg) >= 20:
                v_c, i_c, _, dt_c, q_c = _build_arrays(chg)
                for rec in segmenter.iter_segments(
                    cell_id, int(cyc),
                    _empty, _empty, _empty, _empty,
                    v_c, i_c, dt_c, q_c,
                ):
                    sn = rec.meta.get("seg_name") or spec_names[rec.scenario_id]
                    if sn in stats:
                        stats[sn]["count"] += 1
                        stats[sn]["rows"].append(len(rec.v))
                        stats[sn]["dur_s"].append(float(np.sum(rec.dt)))

    return stats


def print_stats(stats: dict, spec_names: list, axis: str, dataset: str,
                out_path: Path | None = None):
    w = 60
    lines = [
        f"\n{'─' * w}",
        f"  [{axis.upper()}]  {dataset}  세그먼트 통계",
        f"{'─' * w}",
        f"  {'시나리오':<22} {'세그먼트 수':>10} {'평균 행 수':>11} {'평균 시간(s)':>13}",
        f"  {'─' * (w - 2)}",
    ]
    total = 0
    for name in spec_names:
        s   = stats.get(name, {})
        cnt = s.get("count", 0)
        total += cnt
        if cnt == 0:
            lines.append(f"  {name:<22} {'0':>10} {'—':>11} {'—':>13}")
        else:
            avg_r = np.mean(s["rows"])
            avg_d = np.mean(s["dur_s"])
            lines.append(f"  {name:<22} {cnt:>10,} {avg_r:>11.1f} {avg_d:>13.1f}")
    lines += [
        f"  {'─' * (w - 2)}",
        f"  {'합계':<22} {total:>10,}",
        f"{'─' * w}",
        "",
    ]
    text = "\n".join(lines)
    print(text)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"  통계 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 사이클 시각화  (--mode segment)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cycle_segments(
    pkl_path: Path,
    segmenter,
    spec_names: list,
    cycle_id: int,
    axis: str,
    out_path: Path,
):
    """특정 셀 사이클을 세그먼트별 색으로 분리해 하나의 사이클로 시각화.

    레이아웃
    ─────────────────────────────────────────────────────
    [상단 전체] V vs time_s — 충전+방전 연속 (세그먼트 색상 밴드)
    [좌하]     V vs Q      — 방전 세그먼트 색상 오버레이
    [우하]     V vs Q      — 충전 세그먼트 색상 오버레이

    iter_segments는 rcs처럼 무작위 축도 있으므로 각 방향당 1회만 호출해
    결과를 리스트로 캐싱 후 여러 패널에서 재사용한다.
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    df_all  = raw.get("cycles")
    cell_id = raw.get("meta", {}).get("cell_id", pkl_path.stem)

    if df_all is None:
        print(f"  [경고] {pkl_path.name}: cycles 없음")
        return

    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    # ── 사이클 선택 ───────────────────────────────────────────────────────────
    valid_cycs = sorted(c for c in df_all["cycle"].unique() if c != 0)
    if not valid_cycs:
        print("  [경고] 유효 사이클 없음")
        return
    if cycle_id == 0 or cycle_id not in valid_cycs:
        cycle_id = valid_cycs[0]
        print(f"  → 첫 번째 유효 사이클 {cycle_id} 사용")

    grp = df_all[df_all["cycle"] == cycle_id]
    dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
    chg = grp[grp["phase"] == "charge"].sort_values("time_s")

    # ── 세그먼트 한 번씩 수집 (iter_segments 재호출 방지) ──────────────────────
    _empty = np.empty(0, dtype=float)

    dis_arrays: "tuple | None" = None
    dis_recs:   list = []
    if len(dis) >= 30:
        v_d, i_d, t_d, dt_d, q_d = _build_arrays(dis)
        dis_arrays = (v_d, i_d, t_d, dt_d, q_d)
        dis_recs   = [
            (rec.meta.get("seg_name") or spec_names[rec.scenario_id], rec)
            for rec in segmenter.iter_segments(cell_id, cycle_id, v_d, i_d, dt_d, q_d)
        ]

    chg_arrays: "tuple | None" = None
    chg_recs:   list = []
    if len(chg) >= 20:
        v_c, i_c, t_c, dt_c, q_c = _build_arrays(chg)
        chg_arrays = (v_c, i_c, t_c, dt_c, q_c)
        chg_recs   = [
            (rec.meta.get("seg_name") or spec_names[rec.scenario_id], rec)
            for rec in segmenter.iter_segments(
                cell_id, cycle_id, _empty, _empty, _empty, _empty, v_c, i_c, dt_c, q_c
            )
        ]

    # ── 색상 매핑 ─────────────────────────────────────────────────────────────
    color_map = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(spec_names)}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.3, 1.0],
        hspace=0.48, wspace=0.28,
    )
    ax_t  = fig.add_subplot(gs[0, :])   # V vs time_s (전체 사이클)
    ax_dq = fig.add_subplot(gs[1, 0])   # V vs Q (방전)
    ax_cq = fig.add_subplot(gs[1, 1])   # V vs Q (충전)

    cap_label = ""
    if len(dis) > 0 and "capacity_Ah" in dis.columns:
        dis_cap = dis["capacity_Ah"].iloc[0]
        if np.isfinite(dis_cap):
            cap_label = f"  │  Q = {dis_cap:.4f} Ah"

    fig.suptitle(
        f"[{axis}]  {cell_id}  Cycle {cycle_id}{cap_label}",
        fontsize=12, fontweight="bold",
    )

    legend_added: set = set()
    top_handles: "list[mpatches.Patch]" = []

    # ═════════════════════════════════════════════════════════════════════════
    # 패널 0: V vs time_s  (충전 + 방전 연속)
    # ═════════════════════════════════════════════════════════════════════════
    ax_t.set_facecolor("#f8f9fa")
    ax_t.set_xlabel("Time [s]", fontsize=9)
    ax_t.set_ylabel("Voltage [V]", fontsize=9)
    ax_t.tick_params(labelsize=8)
    ax_t.set_title("전체 사이클 V-t  (세그먼트 색상 밴드)", fontsize=9, pad=4)

    # 배경 전체 곡선 (회색)
    for phase_grp, fill_col, line_col, lbl in [
        (chg, "#d6eaf8", "#2980b9", "Charge"),
        (dis, "#fdf2e9", "#ca6f1e", "Discharge"),
    ]:
        if len(phase_grp) < 10:
            continue
        t_bg = phase_grp["time_s"].values.astype(float)
        v_bg = phase_grp["voltage_V"].values.astype(float)
        ax_t.fill_between(t_bg, v_bg.min() - 0.02, v_bg, color=fill_col, alpha=0.2, zorder=0)
        ax_t.plot(t_bg, v_bg, color="#cccccc", lw=0.8, zorder=1, alpha=0.9)
        ax_t.text(
            (t_bg[0] + t_bg[-1]) / 2, 0.03, lbl,
            transform=ax_t.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color=line_col, alpha=0.65,
        )

    if len(chg) >= 10 and len(dis) >= 10:
        ax_t.axvline(float(chg["time_s"].iloc[-1]),
                     color="#555555", lw=1.2, ls="--", zorder=4, alpha=0.6)

    # 세그먼트 색상 밴드 + 오버레이 (방전)
    if dis_arrays is not None:
        _, _, t_d, _, q_d = dis_arrays
        for sn, rec in dis_recs:
            c = color_map.get(sn, "black")
            t0, t1 = _t_range_from_q(t_d, q_d, rec.q)
            if t0 is not None:
                ax_t.axvspan(t0, t1, color=c, alpha=0.35, zorder=2)
            idx0 = int(np.searchsorted(q_d, rec.q[0], "left"))
            idx1 = int(np.searchsorted(q_d, rec.q[-1], "right"))
            t_seg = t_d[idx0:idx1]
            n = min(len(t_seg), len(rec.v))
            ax_t.plot(t_seg[:n], rec.v[:n], color=c, lw=2.3, alpha=0.9, zorder=3)
            if sn not in legend_added:
                top_handles.append(mpatches.Patch(color=c, label=sn))
                legend_added.add(sn)

    # 세그먼트 색상 밴드 + 오버레이 (충전)
    if chg_arrays is not None:
        _, _, t_c, _, q_c = chg_arrays
        for sn, rec in chg_recs:
            c = color_map.get(sn, "black")
            t0, t1 = _t_range_from_q(t_c, q_c, rec.q)
            if t0 is not None:
                ax_t.axvspan(t0, t1, color=c, alpha=0.35, zorder=2)
            idx0 = int(np.searchsorted(q_c, rec.q[0], "left"))
            idx1 = int(np.searchsorted(q_c, rec.q[-1], "right"))
            t_seg = t_c[idx0:idx1]
            n = min(len(t_seg), len(rec.v))
            ax_t.plot(t_seg[:n], rec.v[:n], color=c, lw=2.3, alpha=0.9, zorder=3)
            if sn not in legend_added:
                top_handles.append(mpatches.Patch(color=c, label=sn))
                legend_added.add(sn)

    if top_handles:
        ax_t.legend(handles=top_handles, fontsize=7.5,
                    loc="upper right", framealpha=0.85,
                    ncol=min(len(top_handles), 4))

    # ═════════════════════════════════════════════════════════════════════════
    # 패널 1: V vs Q — 방전
    # ═════════════════════════════════════════════════════════════════════════
    ax_dq.set_facecolor("#fdfbf7")
    ax_dq.set_xlabel("Q_cum [Ah]", fontsize=8)
    ax_dq.set_ylabel("Voltage [V]", fontsize=8)
    ax_dq.set_title("방전  V-Q", fontsize=9, pad=4)
    ax_dq.tick_params(labelsize=7)

    if dis_arrays is not None:
        v_d, _, _, _, q_d = dis_arrays
        ax_dq.plot(q_d, v_d, color="#cccccc", lw=1.0, zorder=1)
        for sn, rec in dis_recs:
            c = color_map.get(sn, "black")
            ax_dq.plot(rec.q, rec.v, color=c, lw=2.5, alpha=0.9, zorder=2)
            ax_dq.scatter([rec.q[0]], [rec.v[0]],
                          color=c, s=40, zorder=4, edgecolors="white", linewidths=0.6)
    else:
        ax_dq.text(0.5, 0.5, "방전 데이터 없음", ha="center", va="center",
                   transform=ax_dq.transAxes, color="gray", fontsize=9)

    # ═════════════════════════════════════════════════════════════════════════
    # 패널 2: V vs Q — 충전
    # ═════════════════════════════════════════════════════════════════════════
    ax_cq.set_facecolor("#f7fbfd")
    ax_cq.set_xlabel("Q_cum [Ah]", fontsize=8)
    ax_cq.set_ylabel("Voltage [V]", fontsize=8)
    ax_cq.set_title("충전  V-Q", fontsize=9, pad=4)
    ax_cq.tick_params(labelsize=7)

    if chg_arrays is not None:
        v_c, _, _, _, q_c = chg_arrays
        ax_cq.plot(q_c, v_c, color="#cccccc", lw=1.0, zorder=1)
        for sn, rec in chg_recs:
            c = color_map.get(sn, "black")
            ax_cq.plot(rec.q, rec.v, color=c, lw=2.5, alpha=0.9, zorder=2)
            ax_cq.scatter([rec.q[0]], [rec.v[0]],
                          color=c, s=40, zorder=4, edgecolors="white", linewidths=0.6)
    else:
        ax_cq.text(0.5, 0.5, "충전 데이터 없음", ha="center", va="center",
                   transform=ax_cq.transAxes, color="gray", fontsize=9)

    # ── 공유 범례 (하단) ──────────────────────────────────────────────────────
    all_handles = [mpatches.Patch(color=color_map[n], label=n) for n in spec_names]
    fig.legend(
        handles=all_handles, fontsize=8.5,
        loc="lower center", ncol=min(len(spec_names), 6),
        bbox_to_anchor=(0.5, -0.01), framealpha=0.92,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  플롯 저장: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. IC 커브 시각화  (--mode ic)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ic(
    v: np.ndarray,
    q: np.ndarray,
    n_points: int = 600,
    smooth_sigma: float = 4.0,
) -> "tuple[np.ndarray, np.ndarray] | tuple[None, None]":
    """dQ/dV vs V (증분용량 커브) 계산.

    1. V 기준 오름차순 정렬 + 중복 제거
    2. 균등 V 격자에 Q 보간
    3. Gaussian 스무딩 (sigma: V 격자 포인트 단위)
    4. 수치 미분 → dQ/dV

    방전은 V 오름차순 시 Q가 내림차순이므로 dQ/dV < 0.
    호출 측에서 `-dqdv`로 부호를 반전해 피크를 양수로 표시.
    """
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        return None, None

    if len(v) < 30:
        return None, None

    sort_idx = np.argsort(v)
    v_s = v[sort_idx]
    q_s = q[sort_idx]

    # 중복 V 제거 (같은 전압값이 연속하면 미분 불안정)
    _, ui = np.unique(v_s, return_index=True)
    v_u, q_u = v_s[ui], q_s[ui]
    if len(v_u) < 20:
        return None, None

    v_grid   = np.linspace(v_u[0], v_u[-1], n_points)
    q_interp = np.interp(v_grid, v_u, q_u)
    q_smooth = gaussian_filter1d(q_interp, sigma=smooth_sigma)
    dqdv     = np.gradient(q_smooth, v_grid)
    return v_grid, dqdv


def _win_colors(n: int) -> list[str]:
    """창 개수에 맞는 구분 가능한 색상 목록."""
    palette = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#2980b9",
        "#27ae60", "#c0392b", "#8e44ad", "#16a085",
    ]
    return [palette[i % len(palette)] for i in range(n)]


def _draw_vq_bands(ax, edges: list[float], colors: list[str]):
    """V-Q 패널: V 경계 → 가로 밴드 + 파선."""
    for k, (v0, v1) in enumerate(zip(edges[:-1], edges[1:])):
        c = colors[k]
        ax.axhspan(v0, v1, color=c, alpha=0.13, zorder=0)
        ax.axhline(v0, color=c, lw=0.9, ls="--", alpha=0.55, zorder=1)
    ax.axhline(edges[-1], color=colors[-1], lw=0.9, ls="--", alpha=0.55, zorder=1)


def _draw_ic_bands(ax, edges: list[float], colors: list[str]):
    """IC 패널: V 경계 → 세로 밴드 + 파선 + 창 번호 레이블."""
    for k, (v0, v1) in enumerate(zip(edges[:-1], edges[1:])):
        c = colors[k]
        ax.axvspan(v0, v1, color=c, alpha=0.13, zorder=0)
        ax.axvline(v0, color=c, lw=1.1, ls="--", alpha=0.65, zorder=1)
        ax.text(
            (v0 + v1) / 2, 0.98, f"win{k}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5,
            color=c, fontweight="bold", alpha=0.9,
        )
    ax.axvline(edges[-1], color=colors[-1], lw=1.1, ls="--", alpha=0.65, zorder=1)


def plot_ic_windows(
    pkl_path: Path,
    segmenter,
    spec_names: list,
    axis: str,
    out_path: Path,
    n_cycles: int = 6,
):
    """IC 커브 (dQ/dV vs V) + 멀티사이클 V-Q — 창 경계 오버레이.

    레이아웃 (2행 × 2열)
    ┌──────────────────────┬──────────────────────┐
    │  방전 V-Q 멀티사이클  │  충전 V-Q 멀티사이클  │  ← 창=가로밴드
    ├──────────────────────┼──────────────────────┤
    │  방전 dQ/dV vs V     │  충전 dQ/dV vs V     │  ← 창=세로밴드
    └──────────────────────┴──────────────────────┘

    vwindow 축: 전압 경계 밴드 표시.
    그 외 축  : IC 커브만 표시 (경계 없음).
    사이클 색상: 초록(초기) → 빨강(말기).
    """
    with open(pkl_path, "rb") as fh:
        raw = pickle.load(fh)

    df_all  = raw.get("cycles")
    cell_id = raw.get("meta", {}).get("cell_id", pkl_path.stem)
    if df_all is None:
        print(f"  [경고] {pkl_path.name}: cycles 없음"); return
    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    # ── 대표 사이클 선택 ──────────────────────────────────────────────────────
    valid_cycs = sorted(c for c in df_all["cycle"].unique() if c != 0)
    if not valid_cycs:
        print("  [경고] 유효 사이클 없음"); return
    n     = min(n_cycles, len(valid_cycs))
    picks = [valid_cycs[int(round(i))]
             for i in np.linspace(0, len(valid_cycs) - 1, n)]

    # 사이클 나이 색상: 초록(early) → 빨강(late)
    _cmap = matplotlib.colormaps["RdYlGn_r"].resampled(256)
    cyc_color = [_cmap(ci / max(n - 1, 1)) for ci in range(n)]

    # ── 창 경계 추출 (vwindow 전용) ──────────────────────────────────────────
    dis_edges: list[float] | None = None
    chg_edges: list[float] | None = None
    if hasattr(segmenter, "_get_dis_edges"):
        try:
            dis_edges = list(segmenter._get_dis_edges())
            chg_edges = list(segmenter._get_chg_edges())
        except Exception:
            pass

    n_dis_wins = (len(dis_edges) - 1) if dis_edges else 0
    n_chg_wins = (len(chg_edges) - 1) if chg_edges else 0
    dis_wcolors = _win_colors(max(n_dis_wins, 1))
    chg_wcolors = _win_colors(max(n_chg_wins, 1))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 2, figsize=(16, 11),
        gridspec_kw={"hspace": 0.45, "wspace": 0.30},
    )
    ax_dvq, ax_cvq = axes[0, 0], axes[0, 1]
    ax_dic, ax_cic = axes[1, 0], axes[1, 1]

    has_edges = dis_edges is not None
    boundary_note = "세로선 = 창 경계" if has_edges else "경계 없음 (vwindow 외 축)"
    fig.suptitle(
        f"[{axis.upper()}]  {cell_id}  ·  IC 커브 & V-Q 멀티사이클 분석\n"
        f"색상: 초록(초기) → 빨강(말기)  │  {boundary_note}",
        fontsize=12, fontweight="bold",
    )

    for ax, title in [
        (ax_dvq, "방전  V-Q  (멀티사이클)  — 창=가로밴드"),
        (ax_cvq, "충전  V-Q  (멀티사이클)  — 창=가로밴드"),
        (ax_dic, "방전  IC  dQ/dV  vs  V  — 창=세로밴드"),
        (ax_cic, "충전  IC  dQ/dV  vs  V  — 창=세로밴드"),
    ]:
        ax.set_facecolor("#f8f9fa")
        ax.tick_params(labelsize=8)
        ax.grid(True, lw=0.4, alpha=0.35)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)

    ax_dvq.set_xlabel("Q_cum [Ah]", fontsize=8)
    ax_dvq.set_ylabel("Voltage [V]", fontsize=8)
    ax_cvq.set_xlabel("Q_cum [Ah]", fontsize=8)
    ax_cvq.set_ylabel("Voltage [V]", fontsize=8)
    ax_dic.set_xlabel("Voltage [V]", fontsize=8)
    ax_dic.set_ylabel("dQ/dV  [Ah/V]  (방전: 부호 반전)", fontsize=8)
    ax_cic.set_xlabel("Voltage [V]", fontsize=8)
    ax_cic.set_ylabel("dQ/dV  [Ah/V]", fontsize=8)

    # ── 창 밴드 ───────────────────────────────────────────────────────────────
    if dis_edges:
        _draw_vq_bands(ax_dvq, dis_edges, dis_wcolors)
        _draw_ic_bands(ax_dic, dis_edges, dis_wcolors)
    if chg_edges:
        _draw_vq_bands(ax_cvq, chg_edges, chg_wcolors)
        _draw_ic_bands(ax_cic, chg_edges, chg_wcolors)

    # ── 사이클별 커브 ─────────────────────────────────────────────────────────
    ic_d_vals: list[float] = []
    ic_c_vals: list[float] = []

    for ci, cyc in enumerate(picks):
        col   = cyc_color[ci]
        lw    = 1.2 + ci * 0.15
        alpha = 0.50 + ci * 0.07
        lbl   = f"Cyc {cyc}"

        grp = df_all[df_all["cycle"] == cyc]
        dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
        chg = grp[grp["phase"] == "charge"].sort_values("time_s")

        # 방전
        if len(dis) >= 30:
            v_d, _, _, dt_d, q_d = _build_arrays(dis)
            ax_dvq.plot(q_d, v_d, color=col, lw=lw, alpha=alpha, label=lbl)
            v_ic, dqdv = _compute_ic(v_d, q_d)
            if v_ic is not None:
                # 방전 IC: 부호 반전 → 피크를 양수로
                ax_dic.plot(v_ic, -dqdv, color=col, lw=lw, alpha=alpha)
                ic_d_vals.extend((-dqdv).tolist())

        # 충전
        if len(chg) >= 20:
            v_c, _, _, dt_c, q_c = _build_arrays(chg)
            ax_cvq.plot(q_c, v_c, color=col, lw=lw, alpha=alpha, label=lbl)
            v_ic, dqdv = _compute_ic(v_c, q_c)
            if v_ic is not None:
                ax_cic.plot(v_ic, dqdv, color=col, lw=lw, alpha=alpha)
                ic_c_vals.extend(dqdv.tolist())

    # IC y축: 2~98 percentile 기반 클리핑 (스파이크 이상치 제거)
    for ax, vals in [(ax_dic, ic_d_vals), (ax_cic, ic_c_vals)]:
        if len(vals) > 10:
            p2, p98 = np.percentile(vals, 2), np.percentile(vals, 98)
            span = max(p98 - p2, 1e-6)
            ax.set_ylim(p2 - span * 0.08, p98 + span * 0.12)
        ax.axhline(0, color="#aaaaaa", lw=0.8, ls=":", zorder=0)

    # ── 사이클 범례 (오른쪽에 세로 나열) ─────────────────────────────────────
    from matplotlib.lines import Line2D
    cyc_handles = [
        Line2D([0], [0], color=cyc_color[ci], lw=2.2, label=f"Cyc {picks[ci]}")
        for ci in range(n)
    ]
    fig.legend(
        handles=cyc_handles,
        loc="center right",
        fontsize=8, framealpha=0.88,
        bbox_to_anchor=(1.01, 0.5),
        title="사이클",
    )

    # ── 창 범례 (하단) ────────────────────────────────────────────────────────
    win_handles = []
    if dis_edges:
        for k in range(len(dis_edges) - 1):
            win_handles.append(
                mpatches.Patch(
                    color=dis_wcolors[k], alpha=0.55,
                    label=f"dis win{k}  [{dis_edges[k]:.2f}–{dis_edges[k+1]:.2f} V]",
                )
            )
    if chg_edges:
        for k in range(len(chg_edges) - 1):
            win_handles.append(
                mpatches.Patch(
                    color=chg_wcolors[k], alpha=0.55,
                    label=f"chg win{k}  [{chg_edges[k]:.2f}–{chg_edges[k+1]:.2f} V]",
                )
            )
    if win_handles:
        fig.legend(
            handles=win_handles,
            loc="lower center",
            ncol=min(len(win_handles), 6),
            fontsize=7.5, framealpha=0.90,
            bbox_to_anchor=(0.5, -0.04),
            title="창 구간",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  IC 플롯 저장: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def _discover_axes() -> list[str]:
    """4_hi_analysis/hi_features*.pkl 파일을 스캔해 축 이름 목록 반환.

    hi_features.pkl        → "qfrac"
    hi_features_rcs.pkl    → "rcs"
    hi_features_protocol.pkl → "protocol"
    hi_features_vwindow.pkl  → "vwindow"
    """
    axes = []
    for pkl in sorted(STEP_DIR.glob("hi_features*.pkl")):
        suffix = pkl.stem.replace("hi_features", "").lstrip("_")
        axes.append(suffix if suffix else "qfrac")
    return axes


def _run_for_axis(axis: str, axis_cfg: dict, args) -> None:
    from common.scenario import get_segmenter

    try:
        seg = get_segmenter(axis, {axis: axis_cfg})
    except Exception as e:
        print(f"[경고] 축 '{axis}' segmenter 로드 실패: {e}")
        return

    spec  = seg.get_spec()
    names = spec.scenario_names
    print(f"\n축: {axis}  |  시나리오 ({len(names)}개): {names}")

    ds      = args.dataset.upper()
    root    = MIT_DIR if ds == "MIT" else HUST_DIR
    out_dir = STEP_DIR / "outputs" / "seg_diagnose" / axis
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 통계 ──────────────────────────────────────────────────────────────────
    if not args.no_stats:
        print(f"\n=== 통계 수집: {root} ===")
        stats      = collect_stats(root, seg, names)
        stats_path = out_dir / f"stats_{ds}.txt"
        print_stats(stats, names, axis, ds, out_path=stats_path)

    # ── 사이클 플롯 ───────────────────────────────────────────────────────────
    if not args.no_plot:
        pkls = sorted(root.glob("*.pkl"))
        if not pkls:
            print(f"[경고] {root} 에 pkl 파일 없음 — 플롯 생략")
            return

        rng = np.random.default_rng(args.seed)

        if args.cell:
            # ── 지정 셀: 기존 동작 (단일 셀, 단일 사이클) ──────────────────
            cell_pkl = root / f"{args.cell}.pkl"
            if not cell_pkl.exists():
                print(f"[경고] {cell_pkl} 없음 — 첫 번째 셀로 대체")
                cell_pkl = pkls[0]
            cell_stem = cell_pkl.stem

            if args.mode in ("segment", "all"):
                out_path = out_dir / f"{ds}_{cell_stem}_cyc{args.cycle or 'auto'}.png"
                print(f"\n=== 사이클 시각화 (segment): {cell_stem}  cycle={args.cycle or 'auto'} ===")
                plot_cycle_segments(cell_pkl, seg, names, args.cycle, axis, out_path)

            if args.mode in ("ic", "all"):
                ic_path = out_dir / f"{ds}_{cell_stem}_ic.png"
                print(f"\n=== IC 커브 시각화: {cell_stem}  n_cycles={args.n_cycles} ===")
                plot_ic_windows(cell_pkl, seg, names, axis, ic_path,
                                n_cycles=args.n_cycles)

        else:
            # ── 미지정: 랜덤 N셀 × 랜덤 사이클 ────────────────────────────
            n_rand = min(args.n_random, len(pkls))
            sampled = rng.choice(len(pkls), size=n_rand, replace=False)
            sampled_pkls = [pkls[i] for i in sorted(sampled)]

            if args.mode in ("segment", "all"):
                print(f"\n=== 사이클 시각화 (segment): 랜덤 {n_rand}셀 × 랜덤 사이클 "
                      f"(seed={args.seed}) ===")
                for cell_pkl in sampled_pkls:
                    try:
                        with open(cell_pkl, "rb") as fh:
                            raw_meta = pickle.load(fh)
                        df_tmp = raw_meta.get("cycles")
                        if df_tmp is None:
                            continue
                        if "phase" not in df_tmp.columns:
                            df_tmp = _add_phase(df_tmp)
                        dis_tmp = df_tmp[df_tmp["phase"] == "discharge"]
                        valid_cycs = [
                            int(c) for c in df_tmp["cycle"].unique()
                            if c != 0
                            and int((dis_tmp["cycle"] == c).sum()) >= 30
                        ]
                        if not valid_cycs:
                            continue
                        chosen_cyc = int(rng.choice(valid_cycs))
                    except Exception as e:
                        print(f"  [건너뜀] {cell_pkl.stem}: {e}")
                        continue

                    cell_stem = cell_pkl.stem
                    out_path  = out_dir / f"{ds}_{cell_stem}_cyc{chosen_cyc}.png"
                    print(f"  {cell_stem}  cycle={chosen_cyc}")
                    plot_cycle_segments(cell_pkl, seg, names, chosen_cyc, axis, out_path)

            if args.mode in ("ic", "all"):
                # IC 모드: 랜덤 셀 중 첫 번째 사용
                cell_pkl  = sampled_pkls[0]
                cell_stem = cell_pkl.stem
                ic_path   = out_dir / f"{ds}_{cell_stem}_ic.png"
                print(f"\n=== IC 커브 시각화: {cell_stem}  n_cycles={args.n_cycles} ===")
                plot_ic_windows(cell_pkl, seg, names, axis, ic_path,
                                n_cycles=args.n_cycles)


def main():
    parser = argparse.ArgumentParser(description="시나리오 세그먼트 진단 (통계 + 시각화)")
    parser.add_argument("--seg-axis",    type=str, default=None,
                        help="세그멘테이션 축 (미지정 시 hi_features*.pkl 자동 탐색)")
    parser.add_argument("--axis-config", type=str, default="{}",
                        help="축 파라미터 JSON 문자열 (예: '{\"assign\": \"none\"}')")
    parser.add_argument("--dataset",     type=str, default="MIT",
                        choices=["MIT", "HUST", "mit", "hust"],
                        help="통계 스캔 대상 데이터셋 (기본: MIT)")
    parser.add_argument("--cell",        type=str, default="",
                        help="사이클 플롯 대상 셀 ID (미지정 시 첫 번째 셀 사용)")
    parser.add_argument("--cycle",       type=int, default=0,
                        help="사이클 플롯 대상 사이클 번호 (0이면 첫 번째 유효 사이클)")
    parser.add_argument("--mode",         type=str, default="segment",
                        choices=["segment", "ic", "all"],
                        help="시각화 모드: segment(기본)|ic|all")
    parser.add_argument("--n-cycles",    type=int, default=6,
                        help="IC 모드에서 표시할 대표 사이클 수 (기본: 6)")
    parser.add_argument("--n-random",    type=int, default=10,
                        help="--cell 미지정 시 segment 시각화에 사용할 랜덤 셀 수 (기본: 10)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="랜덤 셀/사이클 선택 재현성 seed (기본: 42)")
    parser.add_argument("--no-stats",    action="store_true",
                        help="통계 수집·출력 생략")
    parser.add_argument("--no-plot",     action="store_true",
                        help="사이클 시각화 생략")
    args = parser.parse_args()

    try:
        axis_cfg: dict = json.loads(args.axis_config)
    except json.JSONDecodeError as e:
        print(f"[ERROR] --axis-config JSON 파싱 실패: {e}")
        return

    if args.seg_axis is not None:
        # ── 단일 축 모드 (기존 동작) ──────────────────────────────────────
        _run_for_axis(args.seg_axis, axis_cfg, args)
    else:
        # ── 자동 탐색 모드: hi_features*.pkl → 모든 축 순회 ──────────────
        axes = _discover_axes()
        if not axes:
            print("[ERROR] 4_hi_analysis/ 에서 hi_features*.pkl 파일을 찾을 수 없습니다.")
            print("  --seg-axis 로 축을 직접 지정하거나 hi_correlation.py를 먼저 실행하세요.")
            return
        print(f"\n[자동 발견] hi_features*.pkl → 축: {axes}")
        for axis in axes:
            print(f"\n{'=' * 60}")
            print(f"  처리 중: {axis}")
            print(f"{'=' * 60}")
            _run_for_axis(axis, axis_cfg, args)

    print("\n완료!")


if __name__ == "__main__":
    main()
