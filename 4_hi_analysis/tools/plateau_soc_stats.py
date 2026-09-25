"""
플래토 진입/탈출 SOC(q_frac) 통계 — MIT/HUST(LFP) 전체 사이클 기준.

hi_compute.py의 _plateau() / _build_vq_curve()와 동일한 방법론(dV/dQ, Savitzky-
Golay 스무딩, THETA_FLAT=0.25 V/Ah 임계값)을 재사용하되, 부분 충방전 세그먼트가
아니라 **전체 사이클**의 충전/방전 곡선에 적용한다 — 세그먼트 로컬 q_frac이
아니라 사이클 전체 기준 SOC(q_frac) 절대 위치를 구하기 위함.

dV/dt가 아니라 dV/dQ를 쓰는 이유: MIT는 사이클 내 multi-step 급속충전(전류가
여러 번 바뀜) 프로토콜이라, dV/dt는 전류가 바뀌는 지점에서 전기화학적으로
무의미한 가짜 꺾임을 만든다. dV/dQ는 전류로 나눠 정규화되므로 이 문제에서
자유롭다(전류 부호만으로 충/방전을 나누고, 그 안에서는 rate-step 무관하게
누적 Q로 통합).

입력: _4_data_hi/clean/{MIT,HUST}/*.pkl (dict: {"meta", "cycles"}, cycles는
      cell_id/cycle/segment_id/time_s/voltage_V/current_A/capacity_Ah/chg_gap_seg)
출력: 4_hi_analysis/outputs/plateau_soc_stats/per_cycle.csv (원자료)
      4_hi_analysis/outputs/plateau_soc_stats/summary.csv (데이터셋×방향 집계)

Run: python 4_hi_analysis/tools/plateau_soc_stats.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data_directories import DATA_4_HI_ROOT  # noqa: E402

THETA_FLAT = 0.25   # V/Ah -- hi_compute.py와 동일 임계값(1차 시도, 전체 사이클 스케일에서도 유효한지는 결과 보고 재검토)
N_BINS = 60         # 전체 사이클이라 세그먼트(8~30)보다 해상도를 높임
MIN_RUN_BINS = 3    # 플래토로 인정할 최소 연속 bin 수(노이즈성 단발 crossing 배제)
MIN_CURRENT_A = 0.02  # 이 미만은 rest/전환 구간으로 보고 제외
MAX_CYCLES_PER_CELL = 40  # 셀당 균등 샘플링 상한(런타임 제어)

CLEAN_ROOT = DATA_4_HI_ROOT / "clean"
OUT_DIR = PROJECT_ROOT / "4_hi_analysis" / "outputs" / "plateau_soc_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_vq(v: np.ndarray, i_abs: np.ndarray, dt: np.ndarray, n_bins: int = N_BINS):
    """Q-bin V-Q 곡선 + Savgol 스무딩 dV/dQ. hi_compute.py::_build_vq_curve와 동일 로직."""
    q_rel = np.cumsum(i_abs * dt) / 3600.0
    q_tot = float(q_rel[-1]) if len(q_rel) else 0.0
    if q_tot < 0.05 or len(v) < 20:
        return None
    q_e = np.linspace(0.0, q_tot, n_bins + 1)
    qm = (q_e[:-1] + q_e[1:]) / 2
    v_av = np.full(n_bins, np.nan)
    for j in range(n_bins):
        m = (q_rel >= q_e[j]) & (q_rel < q_e[j + 1])
        if m.sum() > 0:
            v_av[j] = float(np.mean(v[m]))
    vld = np.isfinite(v_av)
    if vld.sum() < 10:
        return None
    v_sm = np.interp(qm, qm[vld], v_av[vld])
    ws = min(15, n_bins - (1 - n_bins % 2))
    ws = max(5, ws if ws % 2 == 1 else ws - 1)
    try:
        v_sm = savgol_filter(v_sm, ws, 3)
    except Exception:
        pass
    dq_b = q_tot / n_bins
    dvdq = np.gradient(v_sm, dq_b)
    return qm, v_sm, dvdq, q_tot


def find_plateau_bounds(dvdq: np.ndarray, min_run: int = MIN_RUN_BINS):
    """|dV/dQ| < THETA_FLAT인 가장 긴 연속 구간의 (start_idx, end_idx)."""
    flat = np.abs(dvdq) < THETA_FLAT
    runs, start = [], None
    for idx, f in enumerate(flat):
        if f and start is None:
            start = idx
        elif not f and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(flat) - 1))
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_run]
    if not runs:
        return None
    return max(runs, key=lambda r: r[1] - r[0])


def process_cell(pkl_path: Path, dataset: str) -> list[dict]:
    d = pd.read_pickle(pkl_path)
    cell_id = d["meta"]["cell_id"]
    df = d["cycles"]
    cycles_all = sorted(df["cycle"].unique())
    if len(cycles_all) > MAX_CYCLES_PER_CELL:
        idx = np.linspace(0, len(cycles_all) - 1, MAX_CYCLES_PER_CELL).astype(int)
        cycles = [cycles_all[k] for k in sorted(set(idx))]
    else:
        cycles = cycles_all

    rows = []
    for cyc in cycles:
        c = df[df["cycle"] == cyc]
        for direction, sel in (("charge", c.current_A > MIN_CURRENT_A),
                                ("discharge", c.current_A < -MIN_CURRENT_A)):
            sub = c[sel].sort_values("time_s")
            if len(sub) < 20:
                continue
            v = sub["voltage_V"].to_numpy()
            i_abs = np.abs(sub["current_A"].to_numpy())
            t = sub["time_s"].to_numpy()
            dt = np.diff(t, prepend=t[0] - (t[1] - t[0] if len(t) > 1 else 1.0))
            dt = np.clip(dt, 1e-6, None)

            res = build_vq(v, i_abs, dt)
            if res is None:
                continue
            qm, v_sm, dvdq, q_tot = res
            bounds = find_plateau_bounds(dvdq)
            if bounds is None:
                continue
            lo, hi = bounds
            rows.append(dict(
                dataset=dataset, cell_id=cell_id, cycle=int(cyc), direction=direction,
                entry_qfrac=float(qm[lo] / q_tot), exit_qfrac=float(qm[hi] / q_tot),
                plateau_width_qfrac=float((qm[hi] - qm[lo]) / q_tot),
                q_tot_Ah=q_tot,
            ))
    return rows


def main():
    t0 = time.time()
    all_rows = []
    for dataset in ("MIT", "HUST"):
        cell_dir = CLEAN_ROOT / dataset
        files = sorted(cell_dir.glob("*.pkl"))
        print(f"[plateau_soc_stats] {dataset}: {len(files)}개 셀")
        for fi, f in enumerate(files):
            rows = process_cell(f, dataset)
            all_rows.extend(rows)
            if (fi + 1) % 20 == 0:
                print(f"[plateau_soc_stats] {dataset} {fi+1}/{len(files)} 셀 처리, "
                      f"누적 {len(all_rows)}행, {time.time()-t0:.0f}s 경과")

    out = pd.DataFrame(all_rows)
    per_cycle_path = OUT_DIR / "per_cycle.csv"
    out.to_csv(per_cycle_path, index=False)
    print(f"[plateau_soc_stats] 저장: {per_cycle_path} ({len(out)}행)")

    summary = out.groupby(["dataset", "direction"]).agg(
        n=("entry_qfrac", "size"),
        entry_mean=("entry_qfrac", "mean"), entry_std=("entry_qfrac", "std"),
        entry_median=("entry_qfrac", "median"),
        exit_mean=("exit_qfrac", "mean"), exit_std=("exit_qfrac", "std"),
        exit_median=("exit_qfrac", "median"),
        width_mean=("plateau_width_qfrac", "mean"), width_std=("plateau_width_qfrac", "std"),
    ).reset_index()
    summary_path = OUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[plateau_soc_stats] 저장: {summary_path}")
    print(summary.to_string(index=False))
    print(f"[plateau_soc_stats] 총 소요 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
