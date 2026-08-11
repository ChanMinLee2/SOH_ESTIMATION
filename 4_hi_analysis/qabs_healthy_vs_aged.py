"""
Q_abs(정격용량 기반 절대용량) 세그멘테이션을 검토하기 전 사전 탐색 스크립트.

대상: MIT 9셀(배치 b1/b2/b3 당 3개) + HUST 10셀(그룹 1~10 당 1개).
각 셀에 대해, **충전·방전 둘 다** 한 파일에:
  1. 첫 사이클(=가장 건강)과 마지막 사이클(=가장 노화)의 용량(방향별 각자 기준)
  2. V-Q, V-t, DVA(dV/dQ), ICA(dQ/dV) 커브 (두 사이클 오버레이, 충/방전 각각)
  3. 그 방향 첫 사이클 용량(=정격 용량 근사)의 10%~90% 절대 용량 지점을 두 커브 위에
     동시에 표시 — 노화된 사이클이 같은 "절대 용량 마크"를 얼마나 못 따라가는지 시각화

추가(플래토 vs mid-zone 분석, 충전 기준):
  4. 셀별로 건강~노화 5단계 사이클을 샘플링해서, 실제 플래토(|dV/dQ|<THETA_FLAT) Q범위가
     기준안 mid zone(정격용량 20~50%) 안에 몇 % 들어있는지(=플래토 캡처율) 표+히트맵으로 정리
  5. mid zone 시작/끝 경계(정격용량 %) 후보를 그리드 서치해서, 노화 전 구간에서 캡처율이
     가장 높고(평균) 가장 강건한(최소=최악단계 기준) 경계를 탐색
"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"c:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION")
sys.path.insert(0, str(ROOT))
from common.scenario._curves import _build_vq_curve, _build_ica_seg, THETA_FLAT  # noqa: E402
from common.scenario import get_segmenter  # noqa: E402  (실제 QAbsSegmenter로 세그먼트 추출)
from data_directories import DATA_4_HI_ROOT  # noqa: E402

for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _font
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

MIT_DIR  = DATA_4_HI_ROOT / "clean" / "MIT"
HUST_DIR = DATA_4_HI_ROOT / "clean" / "HUST"
OUT_DIR  = ROOT / "4_hi_analysis" / "outputs" / "seg_diagnose" / "qabs_healthy_vs_aged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_PHASE_POS = 0.01
_PHASE_NEG = -0.01
MARK_FRACS = np.round(np.arange(0.1, 1.0, 0.1), 2)  # 10%~90%, 9개

N_STAGE_SAMPLES = 5           # 셀당 사이클 샘플 수(건강~노화)
STAGE_LABELS = ["건강(0%)", "25%", "50%", "75%", "노화(100%)"]

# 기준안(사용자 제안 20/30/50): mid = [20%, 50%] (정격용량=그 셀 첫 사이클 충전용량 기준)
BASELINE_MID = (0.20, 0.50)

# mid 경계 그리드 서치 후보 범위
MID_START_GRID = np.round(np.arange(0.10, 0.45, 0.05), 2)   # 10~40%
MID_END_GRID   = np.round(np.arange(0.45, 0.95, 0.05), 2)   # 45~90%
MIN_MID_WIDTH  = 0.10                                        # 최소 폭 10%p

# ── 2목적 트레이드오프 분석 파라미터 ──────────────────────────────────────────
# 목적1: plateau를 mid에 최대 포함(노화 전구간, 최악단계 기준)
# 목적2: 시나리오 간 개형 구분 + 시나리오 내 개형 유사 (분리도)
N_DESC = 20                      # 세그먼트 개형 descriptor 길이(로컬 q-frac 격자에 V 리샘플)
TRADEOFF_N_SAMPLES = 4           # 이 분석에서 존당 세그먼트 수(고정)
TRADEOFF_DIRECTION = "charge"    # 충전 기준 (방전은 phase만 바꿔 재실행)
TRADEOFF_SEG_LENS = [0.10, 0.15] # seg_len(정격용량 비율) 스윕 값
# (mid_start, mid_end) 후보 격자
TO_MID_START_GRID = np.round(np.arange(0.10, 0.35, 0.05), 2)   # 10~30%
TO_MID_END_GRID   = np.round(np.arange(0.45, 0.85, 0.05), 2)   # 45~80%


def _add_phase(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cur = df["current_A"]
    df["phase"] = "rest"
    df.loc[cur > _PHASE_POS, "phase"] = "charge"
    df.loc[cur < _PHASE_NEG, "phase"] = "discharge"
    return df


def _select_mit_cells(n_per_batch: int = 3) -> list[tuple[str, Path]]:
    files = sorted(MIT_DIR.glob("b*c*.pkl"))
    by_batch: dict[str, list[Path]] = {}
    for f in files:
        m = re.match(r"^(b\d+)c\d+$", f.stem)
        if m:
            by_batch.setdefault(m.group(1), []).append(f)

    selected = []
    for batch in sorted(by_batch):
        picked = 0
        for f in sorted(by_batch[batch], key=lambda p: p.stem):
            if picked >= n_per_batch:
                break
            try:
                with open(f, "rb") as fh:
                    raw = pickle.load(fh)
            except Exception:
                continue
            df = raw.get("cycles")
            if df is None or df["cycle"].nunique() < 20:
                continue
            selected.append((f.stem, f))
            picked += 1
    return selected


def _select_hust_cells() -> list[tuple[str, Path]]:
    files = sorted(HUST_DIR.glob("*-*.pkl"))
    by_group: dict[int, list[tuple[int, Path]]] = {}
    for f in files:
        m = re.match(r"^(\d+)-(\d+)$", f.stem)
        if m:
            by_group.setdefault(int(m.group(1)), []).append((int(m.group(2)), f))

    selected = []
    for g in sorted(by_group):
        for _, f in sorted(by_group[g], key=lambda t: t[0]):
            try:
                with open(f, "rb") as fh:
                    raw = pickle.load(fh)
            except Exception:
                continue
            df = raw.get("cycles")
            if df is None or df["cycle"].nunique() < 20:
                continue
            selected.append((f.stem, f))
            break
    return selected


_DIRECTIONS = ("charge", "discharge")
_DIR_LABEL = {"charge": "충전", "discharge": "방전"}


def _direction_arrays(df_cyc: pd.DataFrame, phase: str):
    sub = df_cyc[df_cyc["phase"] == phase].sort_values("time_s")
    if len(sub) < 20:
        return None
    v = sub["voltage_V"].values.astype(float)
    i = np.abs(sub["current_A"].values.astype(float))
    t = sub["time_s"].values.astype(float)
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    q = np.cumsum(i * dt) / 3600.0
    return v, i, t, dt, q


def _plateau_q_range(qm: np.ndarray, dvdq: np.ndarray, theta: float = THETA_FLAT):
    """|dV/dQ| < theta 인 Q구간 중 가장 넓은 연속 밴드를 (q_lo, q_hi)로 반환.
    없으면 None (플래토 미검출 — vqslope의 '실패 모드'와 동일 현상)."""
    valid = np.isfinite(qm) & np.isfinite(dvdq)
    if valid.sum() < 3:
        return None
    mask = valid & (np.abs(dvdq) < theta)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    best = max(runs, key=lambda r: qm[r[-1]] - qm[r[0]])
    return float(qm[best[0]]), float(qm[best[-1]])


def _overlap_pct(plateau_range, zone_range) -> float:
    """plateau_range(그 사이클의 실제 플래토 Q범위, Ah)가 zone_range(후보 mid zone, Ah)
    안에 몇 % 들어있는지 — 분모는 플래토 자신의 폭(요청하신 '플래토가 mid 구간 내에
    몇 % 존재하는지')."""
    if plateau_range is None:
        return np.nan
    p_lo, p_hi = plateau_range
    z_lo, z_hi = zone_range
    p_width = p_hi - p_lo
    if p_width <= 1e-9:
        return np.nan
    inter = max(0.0, min(p_hi, z_hi) - max(p_lo, z_lo))
    return 100.0 * inter / p_width


def _sample_stage_cycles(cycles: list[int], n: int = N_STAGE_SAMPLES) -> list[int]:
    """건강(첫)~노화(끝) 사이를 n단계로 균등 인덱싱해서 대표 사이클 n개 선택."""
    cycles = sorted(cycles)
    if len(cycles) <= n:
        return cycles
    pos = np.linspace(0, len(cycles) - 1, n).round().astype(int)
    pos = sorted(dict.fromkeys(pos.tolist()))  # 중복 제거, 순서 유지
    return [cycles[p] for p in pos]


def _mid_zone_candidates():
    """(start, end) 후보 목록 — start < end, 폭 >= MIN_MID_WIDTH."""
    cands = []
    for s in MID_START_GRID:
        for e in MID_END_GRID:
            if e - s >= MIN_MID_WIDTH:
                cands.append((round(float(s), 2), round(float(e), 2)))
    return cands


def plateau_capture_for_cell(cell_id: str, pkl_path: Path, dataset_label: str,
                              zone_candidates: list[tuple[float, float]]):
    """이 셀의 5단계 샘플 사이클마다: 플래토 Q범위 검출 + 후보 mid-zone들과의 캡처율(%) 계산.

    반환: (baseline_rows, grid_rows)
      baseline_rows : 기준안(BASELINE_MID)의 단계별 캡처율 — 요청 1번용
      grid_rows     : 모든 후보 zone × 단계의 캡처율 — 요청 2번(그리드 서치)용
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    df = raw.get("cycles")
    if df is None:
        return [], []
    if "phase" not in df.columns:
        df = _add_phase(df)

    all_cycles = sorted(c for c in df["cycle"].unique() if c != 0)
    if len(all_cycles) < 2:
        return [], []
    stage_cycles = _sample_stage_cycles(all_cycles, N_STAGE_SAMPLES)

    # 정격 용량 근사 = 첫(건강) 사이클 충전 용량 (analyze_cell 과 동일 컨벤션)
    arr0 = _direction_arrays(df[df["cycle"] == all_cycles[0]], "charge")
    if arr0 is None:
        return [], []
    cap_rated = float(arr0[4][-1])

    baseline_rows, grid_rows = [], []
    for stage_i, cyc in enumerate(stage_cycles):
        stage_label = STAGE_LABELS[stage_i] if len(stage_cycles) == N_STAGE_SAMPLES else f"stage{stage_i}"
        arr = _direction_arrays(df[df["cycle"] == cyc], "charge")
        if arr is None:
            continue
        v, i, t, dt, q = arr
        qm, _, dvdq, _ = _build_vq_curve(v, i, dt)
        plateau = _plateau_q_range(qm, dvdq)

        base_zone = (BASELINE_MID[0] * cap_rated, BASELINE_MID[1] * cap_rated)
        baseline_rows.append({
            "dataset": dataset_label, "cell_id": cell_id, "cycle": cyc,
            "stage": stage_label, "stage_i": stage_i,
            "cap_rated_Ah": cap_rated, "cap_now_Ah": float(q[-1]),
            "plateau_lo_Ah": None if plateau is None else plateau[0],
            "plateau_hi_Ah": None if plateau is None else plateau[1],
            "mid_start_pct": BASELINE_MID[0] * 100, "mid_end_pct": BASELINE_MID[1] * 100,
            "capture_pct": _overlap_pct(plateau, base_zone),
        })

        for (s, e) in zone_candidates:
            zone = (s * cap_rated, e * cap_rated)
            grid_rows.append({
                "dataset": dataset_label, "cell_id": cell_id, "cycle": cyc,
                "stage_i": stage_i, "mid_start_pct": s * 100, "mid_end_pct": e * 100,
                "capture_pct": _overlap_pct(plateau, zone),
            })

    return baseline_rows, grid_rows


def _mark_points(v, t, q, q_marks):
    """절대 용량 마크(Ah)가 처음 도달되는 (x, v) 좌표. 그 사이클 총 용량을 넘으면 None."""
    q_tot = float(q[-1]) if len(q) else 0.0
    pts_vt, pts_vq = [], []
    for qm in q_marks:
        if qm > q_tot:
            pts_vt.append(None)
            pts_vq.append(None)
            continue
        idx = min(int(np.searchsorted(q, qm)), len(v) - 1)
        pts_vt.append((t[idx], v[idx]))
        pts_vq.append((qm, v[idx]))
    return pts_vt, pts_vq


def analyze_cell(cell_id: str, pkl_path: Path, dataset_label: str) -> dict | None:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    df = raw.get("cycles")
    if df is None:
        return None
    if "phase" not in df.columns:
        df = _add_phase(df)

    cycles = sorted(c for c in df["cycle"].unique() if c != 0)
    if len(cycles) < 2:
        return None
    cyc_first, cyc_last = cycles[0], cycles[-1]
    grp_first = df[df["cycle"] == cyc_first]
    grp_last  = df[df["cycle"] == cyc_last]

    # 방향별로 독립 처리 (충전은 충전 자기 자신의 첫 사이클 용량을, 방전은 방전
    # 자기 자신의 첫 사이클 용량을 "정격 용량" 근사로 사용 — q_frac_wide/vwindow가
    # 충/방전을 별개 시나리오로 다루는 것과 같은 컨벤션).
    per_dir: dict[str, dict] = {}
    for phase in _DIRECTIONS:
        arr_first = _direction_arrays(grp_first, phase)
        arr_last  = _direction_arrays(grp_last, phase)
        if arr_first is None or arr_last is None:
            per_dir[phase] = None
            continue
        v1, i1, t1, dt1, q1 = arr_first
        v2, i2, t2, dt2, q2 = arr_last
        cap_first = float(q1[-1])
        cap_last  = float(q2[-1])
        q_marks   = MARK_FRACS * cap_first
        pts_vt_1, pts_vq_1 = _mark_points(v1, t1, q1, q_marks)
        pts_vt_2, pts_vq_2 = _mark_points(v2, t2, q2, q_marks)
        qm1, _, dvdq1, _ = _build_vq_curve(v1, i1, dt1)
        qm2, _, dvdq2, _ = _build_vq_curve(v2, i2, dt2)
        vm1, dqdv1 = _build_ica_seg(v1, i1, dt1)
        vm2, dqdv2 = _build_ica_seg(v2, i2, dt2)
        per_dir[phase] = dict(
            v1=v1, t1=t1, q1=q1, v2=v2, t2=t2, q2=q2,
            cap_first=cap_first, cap_last=cap_last, q_marks=q_marks,
            pts_vt_1=pts_vt_1, pts_vq_1=pts_vq_1, pts_vt_2=pts_vt_2, pts_vq_2=pts_vq_2,
            qm1=qm1, dvdq1=dvdq1, qm2=qm2, dvdq2=dvdq2,
            vm1=vm1, dqdv1=dqdv1, vm2=vm2, dqdv2=dqdv2,
        )

    if per_dir["charge"] is None:
        return None  # 충전 데이터가 없으면 이 셀은 스킵(방전만 있는 경우는 프로젝트 관례상 무의미)

    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    cap_txt = "   ".join(
        f"{_DIR_LABEL[p]} cap_first={d['cap_first']:.4f}Ah cap_last={d['cap_last']:.4f}Ah "
        f"fade={100 * (1 - d['cap_last'] / d['cap_first']):.1f}%"
        for p, d in per_dir.items() if d is not None
    )
    fig.suptitle(
        f"{dataset_label} {cell_id}   cycle {cyc_first}(건강) vs cycle {cyc_last}(노화)\n{cap_txt}",
        fontsize=11,
    )

    for col, phase in enumerate(_DIRECTIONS):
        d = per_dir[phase]
        label = _DIR_LABEL[phase]
        if d is None:
            for row in range(4):
                axes[row, col].text(0.5, 0.5, f"{label} 데이터 없음",
                                     ha="center", va="center", transform=axes[row, col].transAxes)
                axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            continue

        ax = axes[0, col]
        ax.plot(d["q1"], d["v1"], color="tab:blue", lw=1.3, label=f"cycle {cyc_first} (건강)")
        ax.plot(d["q2"], d["v2"], color="tab:red", lw=1.3, label=f"cycle {cyc_last} (노화)")
        for qm in d["q_marks"]:
            ax.axvline(qm, color="gray", ls=":", lw=0.5, alpha=0.5)
        for p1, p2 in zip(d["pts_vq_1"], d["pts_vq_2"]):
            if p1 is not None:
                ax.scatter(*p1, color="tab:blue", s=32, zorder=5, marker="o")
            if p2 is not None:
                ax.scatter(*p2, color="tab:red", s=40, zorder=5, marker="x")
        ax.set_xlabel("Q [Ah]"); ax.set_ylabel("V [V]")
        ax.set_title(f"[{label}] V-Q curve (점선=정격용량 10%단위, ●건강 ×노화)")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.plot(d["t1"], d["v1"], color="tab:blue", lw=1.3, label=f"cycle {cyc_first} (건강)")
        ax.plot(d["t2"], d["v2"], color="tab:red", lw=1.3, label=f"cycle {cyc_last} (노화)")
        for p1, p2 in zip(d["pts_vt_1"], d["pts_vt_2"]):
            if p1 is not None:
                ax.scatter(*p1, color="tab:blue", s=32, zorder=5, marker="o")
            if p2 is not None:
                ax.scatter(*p2, color="tab:red", s=40, zorder=5, marker="x")
        ax.set_xlabel("t [s]"); ax.set_ylabel("V [V]")
        ax.set_title(f"[{label}] V-t curve (같은 절대용량 마크의 도달 시각 비교)")
        ax.legend(fontsize=8)

        ax = axes[2, col]
        ax.plot(d["qm1"], d["dvdq1"], color="tab:blue", lw=1.1, label=f"cycle {cyc_first}")
        ax.plot(d["qm2"], d["dvdq2"], color="tab:red", lw=1.1, label=f"cycle {cyc_last}")
        ax.set_xlabel("Q [Ah]"); ax.set_ylabel("dV/dQ [V/Ah]")
        ax.set_title(f"[{label}] DVA (dV/dQ)")
        ax.legend(fontsize=8)

        ax = axes[3, col]
        if len(d["vm1"]):
            ax.plot(d["vm1"], d["dqdv1"], color="tab:blue", lw=1.1, label=f"cycle {cyc_first}")
        if len(d["vm2"]):
            ax.plot(d["vm2"], d["dqdv2"], color="tab:red", lw=1.1, label=f"cycle {cyc_last}")
        ax.set_xlabel("V [V]"); ax.set_ylabel("dQ/dV [Ah/V]")
        ax.set_title(f"[{label}] ICA (dQ/dV)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / f"{dataset_label}_{cell_id}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    row: dict = {
        "dataset": dataset_label,
        "cell_id": cell_id,
        "cyc_first": cyc_first,
        "cyc_last": cyc_last,
        "out_png": str(out),
    }
    for phase in _DIRECTIONS:
        d = per_dir[phase]
        pfx = "chg" if phase == "charge" else "dis"
        if d is None:
            row.update({f"{pfx}_cap_first_Ah": np.nan, f"{pfx}_cap_last_Ah": np.nan,
                        f"{pfx}_fade_pct": np.nan, f"{pfx}_marks_reached_healthy": np.nan,
                        f"{pfx}_marks_reached_aged": np.nan})
            continue
        row.update({
            f"{pfx}_cap_first_Ah": d["cap_first"],
            f"{pfx}_cap_last_Ah": d["cap_last"],
            f"{pfx}_fade_pct": 100 * (1 - d["cap_last"] / d["cap_first"]),
            f"{pfx}_marks_reached_healthy": sum(p is not None for p in d["pts_vq_1"]),
            f"{pfx}_marks_reached_aged": sum(p is not None for p in d["pts_vq_2"]),
        })
    return row


def run_plateau_analysis(cell_list: list[tuple[str, Path, str]]):
    """요청 1(기준안 mid 캡처율 리포트) + 요청 2(mid 경계 그리드 서치)."""
    zone_candidates = _mid_zone_candidates()
    print(f"\n[plateau] mid-zone 후보 {len(zone_candidates)}개 "
          f"(start {MID_START_GRID.min()*100:.0f}~{MID_START_GRID.max()*100:.0f}%, "
          f"end {MID_END_GRID.min()*100:.0f}~{MID_END_GRID.max()*100:.0f}%, "
          f"최소폭 {MIN_MID_WIDTH*100:.0f}%p)")

    baseline_all, grid_all = [], []
    for cell_id, pkl_path, dataset_label in cell_list:
        b_rows, g_rows = plateau_capture_for_cell(cell_id, pkl_path, dataset_label, zone_candidates)
        baseline_all.extend(b_rows)
        grid_all.extend(g_rows)

    bdf = pd.DataFrame(baseline_all)
    gdf = pd.DataFrame(grid_all)
    bdf.to_csv(OUT_DIR / "plateau_baseline_mid20_50.csv", index=False, encoding="utf-8-sig")
    gdf.to_csv(OUT_DIR / "plateau_grid_search.csv", index=False, encoding="utf-8-sig")
    print(f"[saved] {OUT_DIR / 'plateau_baseline_mid20_50.csv'}  ({len(bdf)}행)")
    print(f"[saved] {OUT_DIR / 'plateau_grid_search.csv'}  ({len(gdf)}행)")

    # ── 요청 1: 기준안(mid=20~50%) 셀×단계 캡처율 히트맵 ──────────────────────
    pivot = bdf.pivot_table(index=["dataset", "cell_id"], columns="stage",
                             values="capture_pct", aggfunc="first")
    pivot = pivot[[s for s in STAGE_LABELS if s in pivot.columns]]
    fig, ax = plt.subplots(figsize=(8, max(6, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{d} {c}" for d, c in pivot.index], fontsize=7)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if np.isfinite(val):
                ax.text(c, r, f"{val:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="플래토가 mid구간 내에 존재하는 비율 (%)")
    ax.set_title(f"기준안 mid=[{BASELINE_MID[0]*100:.0f}%,{BASELINE_MID[1]*100:.0f}%](정격용량 기준)\n"
                 "셀 × 노화단계(건강→노화)별 플래토 캡처율")
    fig.tight_layout()
    out1 = OUT_DIR / "_plateau_baseline_heatmap.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out1}")
    print("\n[baseline mid=20~50%] 단계별 평균 캡처율(%):")
    print(bdf.groupby("stage", sort=False)["capture_pct"].mean()
          .reindex([s for s in STAGE_LABELS if s in bdf["stage"].unique()])
          .to_string(float_format=lambda x: f"{x:.1f}"))

    # ── 요청 2: mid 경계 그리드 서치 — 후보별 평균/최소 캡처율 ─────────────────
    agg = gdf.groupby(["mid_start_pct", "mid_end_pct"])["capture_pct"].agg(
        mean_pct="mean", min_pct="min", std_pct="std", n="count").reset_index()
    agg.to_csv(OUT_DIR / "plateau_grid_search_agg.csv", index=False, encoding="utf-8-sig")

    best_by_mean = agg.loc[agg["mean_pct"].idxmax()]
    best_by_min = agg.loc[agg["min_pct"].idxmax()]
    print("\n[grid search][참고— 폭 무제약] 평균/최소 캡처율 최대 후보 (폭이 넓을수록 유리한 "
          "자명한 지표라 항상 가장 넓은 후보가 뽑힘 — 아래 '폭 고정 스윕'이 실질적인 답):")
    print(f"  평균 최대: mid=[{best_by_mean.mid_start_pct:.0f}%,{best_by_mean.mid_end_pct:.0f}%]  "
          f"mean={best_by_mean.mean_pct:.1f}%  min={best_by_mean.min_pct:.1f}%")
    print(f"  최소 최대: mid=[{best_by_min.mid_start_pct:.0f}%,{best_by_min.mid_end_pct:.0f}%]  "
          f"mean={best_by_min.mean_pct:.1f}%  min={best_by_min.min_pct:.1f}%")

    base_row = agg[(np.isclose(agg.mid_start_pct, BASELINE_MID[0] * 100)) &
                   (np.isclose(agg.mid_end_pct, BASELINE_MID[1] * 100))]
    if len(base_row):
        br = base_row.iloc[0]
        print(f"[grid search] 기준안(20~50%) 비교: mean={br.mean_pct:.1f}%  min={br.min_pct:.1f}%")

    # 히트맵: x=mid_start, y=mid_end, 색=min_pct(노화 강건성 기준)
    piv_min = agg.pivot(index="mid_end_pct", columns="mid_start_pct", values="min_pct").sort_index(ascending=False)
    piv_mean = agg.pivot(index="mid_end_pct", columns="mid_start_pct", values="mean_pct").sort_index(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, piv, title in ((axes[0], piv_mean, "평균 캡처율(%)"), (axes[1], piv_min, "최소(최악단계) 캡처율(%) — 강건성")):
        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{v:.0f}" for v in piv.columns], rotation=45)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{v:.0f}" for v in piv.index])
        ax.set_xlabel("mid 시작 (% 정격용량)"); ax.set_ylabel("mid 끝 (% 정격용량)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        # 기준안 위치 표시
        if BASELINE_MID[0] * 100 in piv.columns and BASELINE_MID[1] * 100 in piv.index:
            xi = list(piv.columns).index(BASELINE_MID[0] * 100)
            yi = list(piv.index).index(BASELINE_MID[1] * 100)
            ax.scatter([xi], [yi], marker="*", s=250, color="blue", edgecolors="black",
                       label="기준안(20~50%)", zorder=5)
            ax.legend(fontsize=8)
    fig.suptitle("mid zone 경계(정격용량 %) 그리드 서치 — 19셀×5단계(95표본) 기준")
    fig.tight_layout()
    out2 = OUT_DIR / "_plateau_grid_search_heatmap.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out2}")

    # ── 요청 2 본론: 폭을 고정하고(원안과 동일 비교 조건) 시작 위치만 슬라이딩 ──────
    # "넓을수록 유리"한 자명한 효과를 제거하고, "같은 크기 창을 어디에 두는 게
    # 노화에 가장 강건한가"라는 원래 질문에 바로 답한다.
    fixed_widths = [0.20, 0.30, 0.40]  # 20/30/40%p (원안 폭=30%p 포함)
    fig, axes = plt.subplots(1, len(fixed_widths), figsize=(6 * len(fixed_widths), 5), sharey=True)
    print("\n[grid search][본론 — 폭 고정 스윕] 같은 폭에서 위치만 바꿨을 때 가장 강건한 시작점:")
    for ax, width in zip(np.atleast_1d(axes), fixed_widths):
        w_pct = round(width * 100)
        sub = agg[np.isclose(agg["mid_end_pct"] - agg["mid_start_pct"], w_pct)].sort_values("mid_start_pct")
        if len(sub) == 0:
            continue
        ax.plot(sub["mid_start_pct"], sub["mean_pct"], marker="o", label="평균 캡처율")
        ax.plot(sub["mid_start_pct"], sub["min_pct"], marker="s", label="최소(최악단계) 캡처율")
        if w_pct == round((BASELINE_MID[1] - BASELINE_MID[0]) * 100):
            ax.axvline(BASELINE_MID[0] * 100, color="gray", ls=":", label="기준안 시작(20%)")
        best_row = sub.loc[sub["min_pct"].idxmax()]
        ax.axvline(best_row["mid_start_pct"], color="green", ls="--", alpha=0.6,
                   label=f"최소기준 최적 시작({best_row['mid_start_pct']:.0f}%)")
        ax.set_title(f"폭={w_pct}%p 고정")
        ax.set_xlabel("mid 시작점 (% 정격용량)")
        ax.legend(fontsize=8)
        print(f"  폭={w_pct}%p: 최소기준 최적 시작={best_row['mid_start_pct']:.0f}% "
              f"(끝={best_row['mid_end_pct']:.0f}%)  mean={best_row['mean_pct']:.1f}%  "
              f"min={best_row['min_pct']:.1f}%")
    axes[0].set_ylabel("캡처율 (%)")
    fig.suptitle("폭 고정 스윕 — 같은 크기 mid 창을 어디에 둬야 노화 전 구간에서 플래토를 "
                 "가장 강건하게(=최소 캡처율 최대) 담는가")
    fig.tight_layout()
    out3 = OUT_DIR / "_plateau_fixed_width_sweep.png"
    fig.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out3}")


# =============================================================================
# 2목적 트레이드오프 분석: (1) plateau→mid 포함  vs  (2) 시나리오 개형 분리도
# =============================================================================

def _seg_descriptor(v: np.ndarray, q: np.ndarray, n_desc: int = N_DESC):
    """세그먼트 개형 descriptor = 로컬 q-fraction 격자에 리샘플한 V 벡터(길이 n_desc).
    V를 정규화하지 않으므로 절대 전압 레벨(=곡선상 위치)과 형상이 모두 보존된다."""
    if len(q) < 3:
        return None
    q0, q1 = float(q[0]), float(q[-1])
    if q1 - q0 < 1e-9:
        return None
    qf = (q - q0) / (q1 - q0)
    # q가 단조 증가가 아닐 수 있으므로 정렬 후 보간
    order = np.argsort(qf)
    return np.interp(np.linspace(0, 1, n_desc), qf[order], v[order]).astype(float)


def _precompute_corpus(cell_list):
    """각 (셀, 노화단계)의 충전 배열 + BOL 기준용량(cap_ref) + plateau Q범위를 1회 계산.
    plateau/cap_ref는 mid 경계와 무관하므로 후보 루프 밖에서 미리 구해 재사용한다."""
    corpus = []  # {cell_id, stage_i, cap_ref, plateau, v,i,dt,q}
    phase = TRADEOFF_DIRECTION
    for cell_id, pkl_path, ds in cell_list:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        df = raw.get("cycles")
        if df is None:
            continue
        if "phase" not in df.columns:
            df = _add_phase(df)
        cycles = sorted(c for c in df["cycle"].unique() if c != 0)
        if len(cycles) < 2:
            continue
        stage_cycles = _sample_stage_cycles(cycles, N_STAGE_SAMPLES)
        arr0 = _direction_arrays(df[df["cycle"] == cycles[0]], phase)
        if arr0 is None:
            continue
        cap_ref = float(arr0[4][-1])   # BOL 첫 사이클 그 방향 총 용량
        for si, cy in enumerate(stage_cycles):
            arr = _direction_arrays(df[df["cycle"] == cy], phase)
            if arr is None:
                continue
            v, i, t, dt, q = arr
            qm, _, dvdq, _ = _build_vq_curve(v, i, dt)
            corpus.append(dict(cell_id=cell_id, ds=ds, stage_i=si, cap_ref=cap_ref,
                               plateau=_plateau_q_range(qm, dvdq),
                               v=v, i=i, dt=dt, q=q))
    return corpus


def _objective1_capture(corpus, mid_start, mid_end):
    """목적1: plateau가 mid=[mid_start,mid_end]*cap_ref 안에 존재하는 비율.
    반환 (capture_min=최악 노화단계, capture_mean)."""
    caps = []
    for e in corpus:
        if e["plateau"] is None:
            caps.append(0.0); continue
        zone = (mid_start * e["cap_ref"], mid_end * e["cap_ref"])
        caps.append(_overlap_pct(e["plateau"], zone))
    caps = [c for c in caps if np.isfinite(c)]
    if not caps:
        return np.nan, np.nan
    return float(np.min(caps)), float(np.mean(caps))


def _objective2_separability(corpus, mid_start, mid_end, seg_len, n_samples=TRADEOFF_N_SAMPLES):
    """목적2: 실제 QAbsSegmenter로 세그먼트를 추출해 개형 descriptor를 만들고,
    scenario 라벨(low/mid/high)에 대한 분리도(silhouette, Fisher ratio)를 계산.
    노화단계를 전부 한 풀에 넣어 계산하므로 intra 거리에 노화변동이 포함된다
    (=plateau 드리프트가 intra 유사성을 깎으면 정직하게 페널티됨)."""
    from sklearn.metrics import silhouette_samples
    seg = get_segmenter("q_abs", {"q_abs": {
        "mid_start": float(mid_start), "mid_end": float(mid_end),
        "seg_len": float(seg_len), "n_samples": int(n_samples)}})
    empty = np.empty(0, dtype=float)
    X, y = [], []
    # 셀별로 stage 오름차순(건강 먼저)으로 넣어야 cap_ref가 BOL로 잡힌다
    for e in sorted(corpus, key=lambda e: (e["cell_id"], e["stage_i"])):
        recs = seg.iter_segments(e["cell_id"], int(e["stage_i"]),
                                 empty, empty, empty, empty,
                                 e["v"], e["i"], e["dt"], e["q"])
        for r in recs:
            d = _seg_descriptor(r.v, r.q)
            if d is None:
                continue
            X.append(d); y.append(int(r.latent_class))
    _NAN = dict(silhouette=np.nan, fisher=np.nan, n_seg=len(X),
                sil_lo=np.nan, sil_mid=np.nan, sil_hi=np.nan,
                intra_lo=np.nan, intra_mid=np.nan, intra_hi=np.nan,
                n_lo=0, n_mid=0, n_hi=0)
    if len(set(y)) < 2 or len(X) < 10:
        return _NAN
    X = np.asarray(X); y = np.asarray(y)
    # 차원별 z-score (단일 V-포인트가 거리를 지배하지 않도록)
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-9] = 1.0
    Xz = (X - mu) / sd

    # 시나리오별 실루엣 = silhouette_samples 를 클래스별 평균 (전역평균이 아님)
    samp = silhouette_samples(Xz, y)
    sil = float(np.mean(samp))   # 전역 평균(참고용)
    out = dict(silhouette=sil, n_seg=len(X))
    _LAB = {0: "lo", 1: "mid", 2: "hi"}
    for k, name in _LAB.items():
        Xk = Xz[y == k]
        out[f"sil_{name}"]   = float(np.mean(samp[y == k])) if (y == k).any() else np.nan
        # intra 동질성 = 클래스 내 centroid 까지의 평균 거리(작을수록 개형 유사)
        out[f"intra_{name}"] = float(np.mean(np.linalg.norm(Xk - Xk.mean(0), axis=1))) if len(Xk) else np.nan
        out[f"n_{name}"]     = int(len(Xk))
    # Fisher ratio = between / within (trace 기반), 높을수록 분리 좋음
    g_mu = Xz.mean(0); Sb = Sw = 0.0
    for k in np.unique(y):
        Xk = Xz[y == k]
        Sb += len(Xk) * float(np.sum((Xk.mean(0) - g_mu) ** 2))
        Sw += float(np.sum((Xk - Xk.mean(0)) ** 2))
    out["fisher"] = float(Sb / Sw) if Sw > 1e-9 else np.nan
    return out


def _pareto_front(pts):
    """(x=capture_min↑, y=silhouette↑) 최대화 2목적 Pareto front 인덱스 반환."""
    idx = []
    for i, (xi, yi) in enumerate(pts):
        dominated = any((xj >= xi and yj >= yi and (xj > xi or yj > yi))
                        for j, (xj, yj) in enumerate(pts) if j != i)
        if not dominated:
            idx.append(i)
    return idx


def run_tradeoff_analysis(cell_list):
    """목적1(plateau→mid) vs 목적2(개형 분리도)의 2목적 그리드 탐색 + Pareto."""
    print("\n" + "=" * 60)
    print("  2목적 트레이드오프 분석 (충전 기준)")
    print("  목적1: plateau를 mid에 최대 포함 (capture_min = 최악 노화단계)")
    print("  목적2: 시나리오 개형 분리 (silhouette / Fisher, 노화 풀링)")
    print("=" * 60)
    corpus = _precompute_corpus(cell_list)
    n_cells = len({e["cell_id"] for e in corpus})
    print(f"[tradeoff] corpus: {n_cells}셀 × 최대 {N_STAGE_SAMPLES}단계 = {len(corpus)}개 (셀,단계)")

    rows = []
    for seg_len in TRADEOFF_SEG_LENS:
        for ms in TO_MID_START_GRID:
            for me in TO_MID_END_GRID:
                if me - ms < MIN_MID_WIDTH:
                    continue
                # seg_len 이 최소 존 폭 이하가 아니면 경계 비침범 세그먼트 불가 → 스킵
                narrowest = min(ms, me - ms, 1.0 - me)
                if seg_len > narrowest + 1e-9:
                    continue
                cmin, cmean = _objective1_capture(corpus, ms, me)
                sep = _objective2_separability(corpus, ms, me, seg_len)
                rows.append({
                    "seg_len": round(float(seg_len), 2),
                    "mid_start_pct": round(float(ms) * 100),
                    "mid_end_pct": round(float(me) * 100),
                    "capture_min": cmin, "capture_mean": cmean,
                    "silhouette": sep["silhouette"], "fisher": sep["fisher"],
                    # 시나리오별 분리도(실루엣)와 내부 동질성(intra centroid 거리)
                    "sil_lo": sep["sil_lo"], "sil_mid": sep["sil_mid"], "sil_hi": sep["sil_hi"],
                    # sil_worst = 세 시나리오 중 최악 실루엣 — 전역평균보다 정직한 목적2 스칼라
                    # ("모든 시나리오가 구분되는가" — 비대한 mid가 mid만 뭉개도 여기서 드러남)
                    "sil_worst": float(np.nanmin([sep["sil_lo"], sep["sil_mid"], sep["sil_hi"]])),
                    "intra_lo": sep["intra_lo"], "intra_mid": sep["intra_mid"], "intra_hi": sep["intra_hi"],
                    "n_lo": sep["n_lo"], "n_mid": sep["n_mid"], "n_hi": sep["n_hi"],
                    "n_seg": sep["n_seg"],
                })
    tdf = pd.DataFrame(rows)
    tdf.to_csv(OUT_DIR / "tradeoff_grid.csv", index=False, encoding="utf-8-sig")
    print(f"[saved] {OUT_DIR / 'tradeoff_grid.csv'}  ({len(tdf)}개 후보)")

    # ── Pareto 산점도 (seg_len 별) ────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(TRADEOFF_SEG_LENS), figsize=(7 * len(TRADEOFF_SEG_LENS), 6),
                             squeeze=False)
    for ax, seg_len in zip(axes[0], TRADEOFF_SEG_LENS):
        sub = tdf[np.isclose(tdf["seg_len"], seg_len)].reset_index(drop=True)
        sub = sub.dropna(subset=["capture_min", "silhouette"])
        if len(sub) == 0:
            continue
        pts = list(zip(sub["capture_min"], sub["silhouette"]))
        front = _pareto_front(pts)
        sc = ax.scatter(sub["capture_min"], sub["silhouette"],
                        c=sub["mid_end_pct"] - sub["mid_start_pct"], cmap="viridis", s=60)
        fig.colorbar(sc, ax=ax, label="mid 폭(%p)")
        fsub = sub.iloc[front].sort_values("capture_min")
        ax.plot(fsub["capture_min"], fsub["silhouette"], "r--o", lw=1.5,
                label="Pareto front", zorder=5)
        for _, r in fsub.iterrows():
            ax.annotate(f"[{r.mid_start_pct:.0f},{r.mid_end_pct:.0f}]",
                        (r.capture_min, r.silhouette), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("목적1: plateau capture_min (%, 최악 노화단계) →")
        ax.set_ylabel("목적2: silhouette (개형 분리도) →")
        ax.set_title(f"seg_len={seg_len} (정격용량 비율)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Q_abs 2목적 트레이드오프 — 우상단이 이상적(plateau 잘 담고 + 개형 잘 구분).\n"
                 "빨간 점선=Pareto front (지배당하지 않는 후보). 라벨=[mid_start%,mid_end%]")
    fig.tight_layout()
    out = OUT_DIR / "_tradeoff_pareto.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    # ── 단일목적 히트맵 (대표 seg_len) ────────────────────────────────────────
    seg_ref = TRADEOFF_SEG_LENS[len(TRADEOFF_SEG_LENS) // 2]
    sub = tdf[np.isclose(tdf["seg_len"], seg_ref)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, col, title, vmax in ((axes[0], "capture_min", "목적1: plateau capture_min(%)", 100),
                                 (axes[1], "silhouette", "목적2: silhouette", None)):
        piv = sub.pivot(index="mid_end_pct", columns="mid_start_pct", values=col).sort_index(ascending=False)
        im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto",
                       vmin=0 if col == "capture_min" else None, vmax=vmax)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{v:.0f}" for v in piv.columns])
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{v:.0f}" for v in piv.index])
        ax.set_xlabel("mid 시작(% 정격용량)"); ax.set_ylabel("mid 끝(% 정격용량)")
        ax.set_title(title)
        for r in range(piv.shape[0]):
            for c in range(piv.shape[1]):
                val = piv.values[r, c]
                if np.isfinite(val):
                    ax.text(c, r, f"{val:.2f}" if col == "silhouette" else f"{val:.0f}",
                            ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"단일목적 히트맵 (seg_len={seg_ref}) — 두 지표의 최적 위치가 어긋나는 정도를 확인")
    fig.tight_layout()
    out2 = OUT_DIR / "_tradeoff_heatmaps.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out2}")

    # ── 시나리오별 실루엣 히트맵 (대표 seg_len) — mid만 뭉개지는지 직접 확인 ──────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _vmin = float(np.nanmin(sub[["sil_lo", "sil_mid", "sil_hi"]].values))
    _vmax = float(np.nanmax(sub[["sil_lo", "sil_mid", "sil_hi"]].values))
    for ax, col, title in ((axes[0], "sil_lo", "low 시나리오 실루엣"),
                           (axes[1], "sil_mid", "mid 시나리오 실루엣"),
                           (axes[2], "sil_hi", "high 시나리오 실루엣")):
        piv = sub.pivot(index="mid_end_pct", columns="mid_start_pct", values=col).sort_index(ascending=False)
        im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=_vmin, vmax=_vmax)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{v:.0f}" for v in piv.columns])
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{v:.0f}" for v in piv.index])
        ax.set_xlabel("mid 시작(%)"); ax.set_ylabel("mid 끝(%)"); ax.set_title(title)
        for r in range(piv.shape[0]):
            for c in range(piv.shape[1]):
                val = piv.values[r, c]
                if np.isfinite(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"시나리오별 실루엣 (seg_len={seg_ref}) — 전역평균이 가리던 '어느 시나리오가 약한가'를 분해")
    fig.tight_layout()
    out3 = OUT_DIR / "_tradeoff_silhouette_per_scenario.png"
    fig.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out3}")

    # ── 요약 출력 ─────────────────────────────────────────────────────────────
    clean = tdf.dropna(subset=["capture_min", "silhouette", "sil_worst"])
    if len(clean):
        def _line(tag, r):
            print(f"  {tag}: mid=[{r.mid_start_pct:.0f},{r.mid_end_pct:.0f}] seg_len={r.seg_len}  "
                  f"cap_min={r.capture_min:.1f}%  sil(전역)={r.silhouette:.3f}  "
                  f"sil[lo/mid/hi]={r.sil_lo:.2f}/{r.sil_mid:.2f}/{r.sil_hi:.2f}  "
                  f"sil_worst={r.sil_worst:.3f}")
        best_c = clean.loc[clean["capture_min"].idxmax()]
        best_s = clean.loc[clean["silhouette"].idxmax()]
        best_w = clean.loc[clean["sil_worst"].idxmax()]
        # knee: capture_min 과 sil_worst(정직한 목적2)를 정규화 후 (1,1)까지 거리 최소
        cn = (clean["capture_min"] - clean["capture_min"].min()) / (np.ptp(clean["capture_min"]) + 1e-9)
        wn = (clean["sil_worst"] - clean["sil_worst"].min()) / (np.ptp(clean["sil_worst"]) + 1e-9)
        knee = clean.loc[(np.sqrt((1 - cn) ** 2 + (1 - wn) ** 2)).idxmin()]
        print("\n[요약] (sil_worst = 세 시나리오 중 최악 실루엣 = 정직한 목적2 지표)")
        _line("목적1 최고(cap_min)   ", best_c)
        _line("목적2 최고(전역 sil)  ", best_s)
        _line("목적2 최고(sil_worst) ", best_w)
        _line("knee(cap_min+worst)  ", knee)


def main():
    rows = []
    cell_list: list[tuple[str, Path, str]] = []
    for cell_id, f in _select_mit_cells(3):
        cell_list.append((cell_id, f, "MIT"))
        r = analyze_cell(cell_id, f, "MIT")
        if r:
            rows.append(r)
    for cell_id, f in _select_hust_cells():
        cell_list.append((cell_id, f, "HUST"))
        r = analyze_cell(cell_id, f, "HUST")
        if r:
            rows.append(r)

    df = pd.DataFrame(rows)
    csv_out = OUT_DIR / "summary.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df.drop(columns=["out_png"]).to_string(index=False))
    print(f"\n[info] 총 {len(df)}개 셀 처리 (MIT {sum(df.dataset == 'MIT')} / "
          f"HUST {sum(df.dataset == 'HUST')})")
    print(f"[saved] {csv_out}")
    print(f"[saved] 셀별 플랏 {len(df)}장 -> {OUT_DIR}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))
    x = np.arange(len(df))
    xticklabels = [f"{d}\n{c}" for d, c in zip(df["dataset"], df["cell_id"])]
    for ax, pfx, label in ((axes[0], "chg", "충전"), (axes[1], "dis", "방전")):
        ax.bar(x - 0.2, df[f"{pfx}_marks_reached_healthy"], width=0.4,
               label="건강(첫 사이클) 도달 마크 수", color="tab:blue")
        ax.bar(x + 0.2, df[f"{pfx}_marks_reached_aged"], width=0.4,
               label="노화(마지막 사이클) 도달 마크 수", color="tab:red")
        ax.axhline(len(MARK_FRACS), color="gray", ls="--", lw=1,
                   label=f"전체 마크 수({len(MARK_FRACS)}개=10~90%)")
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=7)
        ax.set_ylabel("도달한 절대 마크 개수")
        ax.set_title(f"[{label}] 건강 vs 노화 시점 정격용량(Q_abs) 마크 도달 개수 비교")
        ax.legend(fontsize=8)
    fig.suptitle("건강=첫 사이클 자기 자신 기준이라 항상 만점 — 실제 정보는 빨간 막대", fontsize=10)
    fig.tight_layout()
    summary_png = OUT_DIR / "_summary_marks_reached.png"
    fig.savefig(summary_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {summary_png}")

    run_plateau_analysis(cell_list)
    run_tradeoff_analysis(cell_list)


if __name__ == "__main__":
    main()
