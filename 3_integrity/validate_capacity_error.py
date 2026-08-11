"""
validate_capacity_error.py — docs/SOC.md §6 Phase 1: capacity_Ah(등록값) vs 전류적산값 오차 분석.

배경: q_frac_ref 축을 설계하려면 "전류적산으로 얻는 레퍼런스 용량이 등록된 capacity_Ah
피처와 얼마나 다른가"부터 알아야 한다(docs/SOC.md §6 Phase 0 — 이 둘은 원래 별개 값이다:
capacity_Ah는 MIT QDischarge/HUST dq 필드로 데이터셋이 "등록"한 값이고, 세그멘터의
q_tot(=q_frac_wide 분모)은 그 사이클 방전 phase를 우리가 직접 전류적산한 값이다).

**Step 0(단위 검증)을 반드시 먼저 통과해야 Step 1(오차 분석)이 의미가 있다** — MIT/HUST는
원본 소스가 다르고(1_convert/convert_unified.py 확인 결과 HUST는 mA/mAh→A/Ah 변환이
명시적으로 들어있는 반면 MIT는 원본이 이미 A/Ah 스케일이라 변환이 없다), 이 변환이 조용히
깨지면 "오차율"이 실제로는 단위 버그를 측정하는 꼴이 된다. Step 0는 두 데이터셋의 실측
스케일이 물리적으로 타당한 범위(LFP ~1.1Ah 셀 기준)에 있는지, 그리고 서로 같은 자릿수인지
확인하고 — 실패하면 Step 1을 아예 돌리지 않고 즉시 중단한다.

입력: _4_data_hi/clean/{MIT,HUST}/*.pkl  (2_preprocess/preprocess.py 출력, check_integrity.py
      와 동일한 표준 스키마: cell_id, cycle, segment_id, time_s, voltage_V, current_A,
      capacity_Ah, chg_gap_seg — "phase" 컬럼은 없으므로 current_A 부호로 직접 판별한다.
      1_convert/convert_unified.py:92-96의 assign_phase()와 동일한 임계값을 그대로 쓴다.)

출력:
  3_integrity/outputs/capacity_error_raw.csv        — 셀·사이클별 원시 오차 기록
  3_integrity/outputs/capacity_error_cell_summary.csv — 셀별 집계 통계
  3_integrity/outputs/capacity_error_group_summary.csv — 데이터셋×세기그룹 집계 통계
  3_integrity/outputs/capacity_error_by_dataset.png  — (a) 데이터셋별 오차율 분포(방전 기준)
  3_integrity/outputs/capacity_error_by_cell.png     — (b) 셀별 오차율(정렬, 방전 기준)
  3_integrity/outputs/capacity_error_by_intensity.png — (c) 충방전 세기 4군별 오차율(방전 기준)
  3_integrity/outputs/capacity_error_by_current.png  — (a') 데이터셋×충/방전 방향별 오차율 비교
  3_integrity/outputs/capacity_error_curve_example_{MIT,HUST}.png — 데이터셋별 대표 셀 3개
      (MIT: b1/b2/b3, HUST: 1-/4-/7-) 사이클별 등록용량 vs 충/방전 전류적산 곡선

사용:
  python 3_integrity/validate_capacity_error.py                    # 전체 셀
  python 3_integrity/validate_capacity_error.py --n-cells 20        # 데이터셋별 20셀만(빠른 확인)
  python 3_integrity/validate_capacity_error.py --workers 4
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Windows 콘솔(cp949 등)에서 한글 출력 시 깨지는 걸 방지 — 4_hi_analysis/hi_correlation.py에서
# 겪은 것과 동일한 종류의 콘솔 인코딩 문제. 실제 계산/저장과는 무관하지만 스크립트 실행 도중
# print()에서 죽는 걸 막기 위해 stdout/stderr을 UTF-8로 재설정한다.
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# 플랏 라벨이 한글이라 기본 폰트(DejaVu Sans)로는 글리프가 깨진다(□□□로 렌더링).
# Windows에 흔히 있는 한글 폰트를 찾아서 지정 — 없으면 경고만 남기고 기본 폰트로 진행.
for _font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"):
    if _font_name in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _font_name
        break
else:
    print("[경고] 한글 폰트를 찾지 못했습니다 — 플랏의 한글 라벨이 깨질 수 있습니다.")
matplotlib.rcParams["axes.unicode_minus"] = False  # 한글 폰트 사용 시 '-' 기호 깨짐 방지

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_directories import DATA_4_HI_ROOT  # noqa: E402
CLEAN_DIR    = DATA_4_HI_ROOT / "clean"
OUT_DIR      = Path(__file__).resolve().parent / "outputs"

DATASETS = ["MIT", "HUST"]

# 1_convert/convert_unified.py:57-58 assign_phase()와 동일한 임계값.
# 정리된 pkl에는 "phase" 컬럼이 없으므로(check_integrity.py EXPECTED_COLS 참조) 이
# 부호 임계값으로 직접 재현한다 — current_A 부호 규약: 양수=충전, 음수=방전
# (docs/DATASET_ANOMALIES.md "방전 전류 부호 규약: 음수" 확인).
PHASE_POS = 0.01
PHASE_NEG = -0.01

MIN_DIS_PTS = 5  # 방전 phase 포인트가 이 미만이면 그 사이클은 스킵(적산 신뢰 불가)

# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — 단위(A/Ah) 일치성 검증. 여기서 실패하면 Step 1을 돌리지 않는다.
# 근거값은 LFP ~1.1Ah 셀(docs/DATASET_ANOMALIES.md: "LFP 공칭 1.1 Ah", 실측 초기용량
# 1.159~1.232Ah) 기준 — mAh/mA 단위가 안 바뀌고 남아 있으면 자릿수(약 1000배)가 완전히
# 어긋나므로 이 넓은 범위로도 충분히 잡아낼 수 있다.
# ─────────────────────────────────────────────────────────────────────────────
CAP_AH_PLAUSIBLE_RANGE  = (0.3, 3.0)     # Ah — 이 범위 밖이면 mAh 단위 잔존 의심
CUR_A_MAX_PLAUSIBLE     = (0.05, 20.0)   # A(절대값 최댓값) — 이 범위 밖이면 mA 단위 잔존 의심
CROSS_DATASET_RATIO_MAX = 10.0           # 두 데이터셋 capacity_Ah 중앙값 비율 허용 상한(배)
# 전류적산/등록용량 비의 대략적(órder-of-magnitude) 스모크테스트 범위 — 완벽히 일치할
# 필요는 없다(둘은 애초에 별개 값, Step 1이 정밀 분석 담당). 단위가 어긋나면 이 비율이
# ~1000배 또는 ~0.001배로 튀므로, 그 정도만 걸러내는 넉넉한 범위로 잡는다.
COULOMB_RATIO_SMOKE_RANGE = (0.2, 5.0)


def _iter_cell_paths(dataset: str, n_cells: int | None) -> list[Path]:
    d = CLEAN_DIR / dataset
    paths = sorted(d.glob("*.pkl"))
    if n_cells is not None and n_cells < len(paths):
        # 앞에서부터 자르지 않고 데이터셋 전체에서 골고루 샘플링(결정론적).
        idx = sorted(set(np.linspace(0, len(paths) - 1, n_cells).round().astype(int)))
        paths = [paths[i] for i in idx]
    return paths


def _load_cycles(pkl_path: Path) -> pd.DataFrame:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    return raw["cycles"] if isinstance(raw, dict) else raw


def _discharge_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["current_A"] < PHASE_NEG]


def step0_scan_cell(pkl_path: Path) -> dict | None:
    """셀 1개에서 Step 0 스케일 통계 추출(빠른 스캔 — 첫 유효 사이클 1개로 쿨롱비만 확인)."""
    try:
        df = _load_cycles(pkl_path)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None

    cap = df["capacity_Ah"].to_numpy(dtype=float)
    cur_abs = df["current_A"].to_numpy(dtype=float)
    cur_abs = np.abs(cur_abs[np.isfinite(cur_abs)])

    coulomb_ratio = np.nan
    dis_all = _discharge_rows(df)
    if len(dis_all) > 0:
        first_cyc = int(dis_all["cycle"].min())
        sub = dis_all[dis_all["cycle"] == first_cyc].sort_values("time_s")
        if len(sub) >= MIN_DIS_PTS:
            t = sub["time_s"].to_numpy(dtype=float)
            i_mag = np.abs(sub["current_A"].to_numpy(dtype=float))
            dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
            integrated = float(np.cumsum(i_mag * dt)[-1] / 3600.0)
            registered = float(sub["capacity_Ah"].iloc[0])
            if registered > 0 and np.isfinite(integrated):
                coulomb_ratio = integrated / registered

    return {
        "cap_values": cap[np.isfinite(cap)],
        "cur_abs_values": cur_abs,
        "coulomb_ratio": coulomb_ratio,
    }


def step0_unit_check(n_cells: int | None) -> dict[str, dict]:
    """데이터셋별 스케일 리포트. {dataset: {cap_median, cap_p99, cur_max_abs, coulomb_ratio_median, n_cells}}."""
    report = {}
    for ds in DATASETS:
        paths = _iter_cell_paths(ds, n_cells)
        cap_chunks, cur_chunks, ratios = [], [], []
        for p in tqdm(paths, desc=f"[Step0] {ds} 스캔"):
            r = step0_scan_cell(p)
            if r is None:
                continue
            cap_chunks.append(r["cap_values"])
            cur_chunks.append(r["cur_abs_values"])
            if np.isfinite(r["coulomb_ratio"]):
                ratios.append(r["coulomb_ratio"])
        if not cap_chunks:
            report[ds] = {"n_cells": 0}
            continue
        cap_arr = np.concatenate(cap_chunks)
        cur_arr = np.concatenate(cur_chunks)
        report[ds] = {
            "n_cells":              len(cap_chunks),
            "cap_median":           float(np.median(cap_arr)),
            "cap_p01":              float(np.percentile(cap_arr, 1)),
            "cap_p99":              float(np.percentile(cap_arr, 99)),
            "cur_max_abs":          float(np.max(cur_arr)) if len(cur_arr) else float("nan"),
            "cur_median_abs":       float(np.median(cur_arr[cur_arr > PHASE_POS])) if (cur_arr > PHASE_POS).any() else float("nan"),
            "coulomb_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
            "coulomb_ratio_n":      len(ratios),
        }
    return report


def _print_step0_report(report: dict[str, dict]) -> None:
    print("\n" + "=" * 72)
    print("  Step 0 — 데이터 단위(A/Ah) 스케일 리포트")
    print("=" * 72)
    for ds, r in report.items():
        if r.get("n_cells", 0) == 0:
            print(f"  [{ds}] 셀 없음 — 경로 확인 필요: {CLEAN_DIR / ds}")
            continue
        print(f"  [{ds}] n_cells={r['n_cells']}")
        print(f"    capacity_Ah   : median={r['cap_median']:.4f}  "
              f"p1~p99=[{r['cap_p01']:.4f}, {r['cap_p99']:.4f}]  (기대범위 {CAP_AH_PLAUSIBLE_RANGE} Ah)")
        print(f"    |current_A|   : median={r['cur_median_abs']:.4f}  max={r['cur_max_abs']:.4f}  "
              f"(기대상한 {CUR_A_MAX_PLAUSIBLE[1]} A)")
        print(f"    전류적산/등록 비: median={r['coulomb_ratio_median']:.4f}  "
              f"(n={r['coulomb_ratio_n']}, 기대범위 {COULOMB_RATIO_SMOKE_RANGE} — 정밀분석은 Step 1)")
    print("=" * 72)


def assert_units_consistent(report: dict[str, dict]) -> None:
    """단위 불일치 의심 시 RuntimeError로 즉시 중단 — Step 1로 넘어가지 않는다."""
    problems: list[str] = []
    for ds, r in report.items():
        if r.get("n_cells", 0) == 0:
            problems.append(f"[{ds}] 유효 셀 0개 — 경로/스키마 확인 필요")
            continue
        lo, hi = CAP_AH_PLAUSIBLE_RANGE
        if not (lo <= r["cap_median"] <= hi):
            problems.append(
                f"[{ds}] capacity_Ah 중앙값={r['cap_median']:.4g} — 기대범위 [{lo}, {hi}] Ah 밖. "
                f"mAh 단위가 안 바뀌고 남아있을 가능성(예상 배율 ~1000배)."
            )
        clo, chi = CUR_A_MAX_PLAUSIBLE
        if not (clo <= r["cur_max_abs"] <= chi):
            problems.append(
                f"[{ds}] |current_A| 최댓값={r['cur_max_abs']:.4g} — 기대범위 [{clo}, {chi}] A 밖. "
                f"mA 단위가 안 바뀌고 남아있을 가능성."
            )
        if r["coulomb_ratio_n"] > 0:
            rlo, rhi = COULOMB_RATIO_SMOKE_RANGE
            ratio = r["coulomb_ratio_median"]
            if not (rlo <= ratio <= rhi):
                problems.append(
                    f"[{ds}] 전류적산/등록용량 비 중앙값={ratio:.4g} — 기대범위 [{rlo}, {rhi}] 밖. "
                    f"단위 불일치 시 이 비율은 통상 ~1000배 또는 ~0.001배로 튄다(순수 추정 오차라면 "
                    f"이 범위를 벗어날 정도로 크지 않아야 함)."
                )

    valid = {ds: r for ds, r in report.items() if r.get("n_cells", 0) > 0}
    if len(valid) == 2:
        ds_list = list(valid.keys())
        m0, m1 = valid[ds_list[0]]["cap_median"], valid[ds_list[1]]["cap_median"]
        ratio = max(m0, m1) / max(min(m0, m1), 1e-9)
        if ratio > CROSS_DATASET_RATIO_MAX:
            problems.append(
                f"{ds_list} 간 capacity_Ah 중앙값 비율={ratio:.1f}배 — "
                f"허용상한 {CROSS_DATASET_RATIO_MAX}배 초과. 두 데이터셋 간 단위 불일치 의심"
                f"(예: 한쪽만 mAh→Ah 변환이 누락됨)."
            )

    if problems:
        msg = (
            "\n\n[중단] Step 0 단위 검증 실패 — Step 1(오차 분석)을 진행하지 않는다.\n"
            "아래 문제를 해결하지 않고 오차율을 계산하면, 그 '오차율'이 실제로는 단위 버그를\n"
            "측정하는 것일 수 있다(docs/SOC.md §6 Phase 1 요구사항 참조).\n  - "
            + "\n  - ".join(problems)
        )
        raise RuntimeError(msg)

    print("\n[Step 0 통과] MIT/HUST 모두 A/Ah 단위로 일관되고 물리적으로 타당한 범위입니다. "
          "Step 1을 진행합니다.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — 셀·사이클별 오차 분석
# ─────────────────────────────────────────────────────────────────────────────

def _integrate_phase(grp: pd.DataFrame) -> tuple[float, float]:
    """time_s 오름차순 정렬 후 cumsum(|I|*dt)/3600 마지막 값과 |I| 평균을 반환.

    hi_correlation.py의 q_cum 계산과 동일 방식(= q_frac_wide가 실제로 쓰는 q_tot 정의).
    """
    grp = grp.sort_values("time_s")
    t = grp["time_s"].to_numpy(dtype=float)
    i_mag = np.abs(grp["current_A"].to_numpy(dtype=float))
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    integrated = float(np.cumsum(i_mag * dt)[-1] / 3600.0)
    return integrated, float(i_mag.mean())


def compute_cell_errors(pkl_path: Path, dataset: str) -> pd.DataFrame:
    """셀 1개의 사이클별 등록용량 vs 충/방전 전류적산값 오차 — 둘 다 계산.

    등록용량 = capacity_Ah 컬럼 값(그 사이클 내 상수 — MIT QDischarge/HUST dq 필드 유래,
              방전 기준 값이다 — HI_ANALYSIS.md: "q_dis = capacity_Ah 레이블 그 자체").
    방전적산(dis_integrated_Ah) = 방전 phase(current_A < PHASE_NEG) 적산.
    충전적산(chg_integrated_Ah) = 충전 phase(current_A > PHASE_POS) 적산 — **등록용량과
              같은 물리량이 아니다**(쿨롱효율<100%로 충전이 방전보다 원래 더 큼). 여기선
              "등록(방전) 대비 충전적산이 얼마나 벌어지는지"를 참고용으로 같이 낸다 —
              방전오차(dis_error_pct)와 나란히 봐서 "충/방전 모두에서 같은 방향/크기의
              괴리가 나는지"를 확인하는 용도(§6 Phase1 사용자 질의: MIT ~9% 괴리가 방전만의
              현상인지 확인).
              `chg_gap_seg=True`인 사이클(2_preprocess의 충전 phase 타임갭으로 일부 행이
              이미 제거된 사이클, docs/DATASET_ANOMALIES.md 필터4)은 남은 충전 행만
              적산하면 인위적으로 작게 나오므로 **충전 쪽만 스킵**(NaN) — 방전은 그대로
              살린다(hi_correlation.py의 "충전 phase 행만 제거, 방전 HI 보존"과 동일 원칙).
    """
    cell_id = pkl_path.stem
    try:
        df = _load_cycles(pkl_path)
    except Exception:
        return pd.DataFrame()
    if df is None or len(df) == 0:
        return pd.DataFrame()

    has_gap_col = "chg_gap_seg" in df.columns
    dis_all = _discharge_rows(df)
    chg_all = df[df["current_A"] > PHASE_POS]

    dis_by_cycle = {cyc: g for cyc, g in dis_all.groupby("cycle", sort=True)}
    chg_by_cycle = {cyc: g for cyc, g in chg_all.groupby("cycle", sort=True)}

    rows = []
    for cyc, dgrp in dis_by_cycle.items():
        if len(dgrp) < MIN_DIS_PTS:
            continue
        registered = float(dgrp["capacity_Ah"].iloc[0])
        if not (np.isfinite(registered) and registered > 0):
            continue
        dis_integrated, dis_mean_I = _integrate_phase(dgrp)
        if not np.isfinite(dis_integrated):
            continue

        chg_integrated, chg_mean_I = np.nan, np.nan
        cgrp = chg_by_cycle.get(cyc)
        if cgrp is not None and len(cgrp) >= MIN_DIS_PTS:
            gapped = bool(cgrp["chg_gap_seg"].any()) if has_gap_col else False
            if not gapped:
                chg_integrated, chg_mean_I = _integrate_phase(cgrp)

        rows.append({
            "dataset":         dataset,
            "cell_id":         cell_id,
            "cycle":           int(cyc),
            "registered_Ah":     registered,
            "dis_integrated_Ah": dis_integrated,
            "dis_error_pct":     (dis_integrated - registered) / registered * 100.0,
            "mean_abs_I_dis":    dis_mean_I,
            "chg_integrated_Ah": chg_integrated,
            "chg_error_pct":     ((chg_integrated - registered) / registered * 100.0
                                   if np.isfinite(chg_integrated) else np.nan),
            "mean_abs_I_chg":    chg_mean_I,
        })
    return pd.DataFrame(rows)


def _step1_worker(args: tuple) -> tuple:
    pkl_path_str, dataset = args
    try:
        df = compute_cell_errors(Path(pkl_path_str), dataset)
        return ("ok", df)
    except Exception:
        return ("err", Path(pkl_path_str).stem + ":\n" + traceback.format_exc())


def step1_collect(n_cells: int | None, workers: int) -> pd.DataFrame:
    jobs: list[tuple[str, str]] = []
    for ds in DATASETS:
        for p in _iter_cell_paths(ds, n_cells):
            jobs.append((str(p), ds))

    frames: list[pd.DataFrame] = []
    if workers <= 1:
        for job in tqdm(jobs, desc="[Step1] 셀별 오차 계산"):
            status, payload = _step1_worker(job)
            if status == "ok":
                frames.append(payload)
            else:
                print(f"\n  [ERR] {payload}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_step1_worker, job): job for job in jobs}
            with tqdm(total=len(jobs), desc="[Step1] 셀별 오차 계산") as pbar:
                for fut in as_completed(futures):
                    status, payload = fut.result()
                    if status == "ok":
                        frames.append(payload)
                    else:
                        print(f"\n  [ERR] {payload}")
                    pbar.update(1)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── 충방전 세기 4군 분할 ──────────────────────────────────────────────────────

def assign_intensity_group(raw_df: pd.DataFrame) -> pd.DataFrame:
    """셀별 평균 방전전류(mean_abs_I_dis)로 4분위 그룹(intensity_group: Q1~Q4) 부여.

    요구사항: "평균 충방전 세기 4개 군" — 방전 오차를 다루므로 방전전류 평균을 기준으로
    잡는다(MIT는 72종 충전 프로토콜, HUST는 77종 방전 프로토콜로 세기 편차가 커서
    이 분할이 의미 있다, docs/SCENARIO_STRATEGY.md §1). 두 데이터셋을 합쳐 분위를
    계산한다 — 이래야 "세기" 자체의 절대적 강도로 비교 가능(데이터셋 내부 상대분위가 아님).
    """
    cell_mean_I = raw_df.groupby("cell_id")["mean_abs_I_dis"].mean()
    try:
        labels = pd.qcut(cell_mean_I, q=4, labels=["Q1(약)", "Q2", "Q3", "Q4(강)"])
    except ValueError:
        # 동일값이 많아 qcut이 4구간을 못 만들 경우(rank 기반으로 폴백)
        labels = pd.qcut(cell_mean_I.rank(method="first"), q=4,
                          labels=["Q1(약)", "Q2", "Q3", "Q4(강)"])
    group_map = labels.to_dict()
    out = raw_df.copy()
    out["intensity_group"] = out["cell_id"].map(group_map)
    return out


# ── 요약 통계 ─────────────────────────────────────────────────────────────────

def summarize(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_summary = raw_df.groupby(["dataset", "cell_id"]).agg(
        n_cycles=("dis_error_pct", "size"),
        error_mean=("dis_error_pct", "mean"),
        error_std=("dis_error_pct", "std"),
        error_median=("dis_error_pct", "median"),
        error_abs_mean=("dis_error_pct", lambda s: s.abs().mean()),
        mean_abs_I_dis=("mean_abs_I_dis", "mean"),
        chg_error_mean=("chg_error_pct", "mean"),
    ).reset_index()

    group_summary = raw_df.groupby(["dataset", "intensity_group"], observed=True).agg(
        n_cycles=("dis_error_pct", "size"),
        n_cells=("cell_id", "nunique"),
        error_mean=("dis_error_pct", "mean"),
        error_std=("dis_error_pct", "std"),
        error_p05=("dis_error_pct", lambda s: s.quantile(0.05)),
        error_p50=("dis_error_pct", "median"),
        error_p95=("dis_error_pct", lambda s: s.quantile(0.95)),
        chg_error_mean=("chg_error_pct", "mean"),
    ).reset_index()

    return cell_summary, group_summary


# ── 플랏 ──────────────────────────────────────────────────────────────────────

def plot_by_dataset(raw_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [raw_df.loc[raw_df["dataset"] == ds, "dis_error_pct"].values for ds in DATASETS]
    bp = ax.boxplot(data, labels=DATASETS, showfliers=False, patch_artist=True)  # matplotlib<3.9(프로젝트 conda env=3.7.5) 호환
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("오차율 (%) = (방전전류적산 - 등록capacity_Ah) / 등록capacity_Ah × 100")
    ax.set_title("(a) 데이터셋별 오차율 분포 (방전 기준)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_by_current_direction(raw_df: pd.DataFrame, out_path: Path) -> None:
    """(a')  데이터셋 × 충/방전 방향별 오차율 분포 — MIT/HUST 각각 충전·방전을 나란히 비교.

    충전 쪽은 등록capacity_Ah(방전 기준 라벨)와 물리적으로 같은 양이 아니므로(쿨롱효율
    <100%로 원래도 충전이 더 큼) 절대적 "오차"라기보다 "등록 방전용량 대비 충전적산이
    얼마나 큰가"로 해석해야 한다 — 다만 여기서 보려는 건 그 절대수준이 아니라 **MIT의
    방전 쪽 큰 괴리(~9%)가 충전 쪽에서도 비슷한 크기·방향으로 나타나는지**다.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    directions = ["방전", "충전"]
    positions = np.arange(len(DATASETS))
    width = 0.35
    colors = {"방전": "#4C72B0", "충전": "#55A868"}
    for offset, direction in zip((-1, 1), directions):
        col = "dis_error_pct" if direction == "방전" else "chg_error_pct"
        vals = [raw_df.loc[raw_df["dataset"] == ds, col].dropna().values for ds in DATASETS]
        bp = ax.boxplot(
            vals, positions=positions + offset * width / 2, widths=width * 0.9,
            showfliers=False, patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[direction])
            patch.set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("오차율 (%) = (전류적산 - 등록capacity_Ah) / 등록capacity_Ah × 100")
    ax.set_title("데이터셋별 충/방전 전류적산 오차율 비교")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[d]) for d in directions]
    ax.legend(handles, directions, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── 셀 예시 곡선용 대표 셀 선정 ─────────────────────────────────────────────────

# MIT: 배치(b1/b2/b3)별 1개, HUST: 셀 번호 그룹(1-/4-/7-)별 1개.
EXAMPLE_CELL_PATTERNS: dict[str, list[str]] = {
    "MIT":  [r"^b1c", r"^b2c", r"^b3c"],
    "HUST": [r"^1-",  r"^4-",  r"^7-"],
}


def select_example_cells(dataset: str, available_cell_ids: list[str]) -> list[str]:
    """요구사항: MIT는 b1/b2/b3 배치별 1개, HUST는 1-/4-/7- 그룹별 1개 — 그룹 내
    셀 번호가 가장 작은 셀을 결정론적으로 고른다(재현성). 그룹에 후보가 없으면 건너뛴다
    (예: --n-cells로 샘플링해서 특정 배치가 비어 있는 경우)."""
    def _num_key(cid: str) -> int:
        m = re.search(r"(\d+)$", cid)
        return int(m.group(1)) if m else 0

    selected = []
    for pat in EXAMPLE_CELL_PATTERNS.get(dataset, []):
        candidates = [c for c in available_cell_ids if re.match(pat, c)]
        if not candidates:
            print(f"  [경고] {dataset} 패턴 '{pat}'에 맞는 셀이 샘플에 없어 대표 셀 선정에서 제외")
            continue
        candidates.sort(key=_num_key)
        selected.append(candidates[0])
    return selected


def plot_by_cell(cell_summary: pd.DataFrame, out_path: Path) -> None:
    df = cell_summary.sort_values(["dataset", "error_mean"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.12), 6))
    colors = df["dataset"].map({"MIT": "#4C72B0", "HUST": "#DD8452"})
    ax.bar(range(len(df)), df["error_mean"], yerr=df["error_std"],
           color=colors, width=0.9, error_kw={"linewidth": 0.4, "alpha": 0.5})
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks([])
    ax.set_xlabel(f"셀 (n={len(df)}, 데이터셋 내 오차 평균 오름차순 정렬)")
    ax.set_ylabel("셀 평균 오차율 (%) ± 표준편차")
    ax.set_title("(b) 셀별 오차율 (방전 기준)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#4C72B0", "#DD8452"]]
    ax.legend(handles, DATASETS, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_capacity_curve_example(raw_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """데이터셋별 대표 셀 3개(MIT: b1/b2/b3 배치별 1개, HUST: 1-/4-/7- 그룹별 1개):
    x=사이클, y=용량(Ah). 실선=등록capacity_Ah(정답 피처), 점선(주황)=방전 전류적산,
    점선(초록)=충전 전류적산 — 데이터셋마다 3개 서브플롯을 세로로 쌓은 그림 1장."""
    out_paths = []
    for ds in DATASETS:
        sub = raw_df[raw_df["dataset"] == ds]
        if sub.empty:
            continue
        example_cells = select_example_cells(ds, sorted(sub["cell_id"].unique()))
        if not example_cells:
            continue

        fig, axes = plt.subplots(len(example_cells), 1, figsize=(9, 4 * len(example_cells)), squeeze=False)
        for ax, example_cell in zip(axes[:, 0], example_cells):
            cell_df = sub[sub["cell_id"] == example_cell].sort_values("cycle")
            ax.plot(cell_df["cycle"], cell_df["registered_Ah"], "-",
                    label="등록 capacity_Ah (정답 피처)", color="#4C72B0", linewidth=1.5)
            ax.plot(cell_df["cycle"], cell_df["dis_integrated_Ah"], "--",
                    label="방전 전류적산", color="#DD8452", linewidth=1.5)
            chg_sub = cell_df.dropna(subset=["chg_integrated_Ah"])
            if not chg_sub.empty:
                ax.plot(chg_sub["cycle"], chg_sub["chg_integrated_Ah"], "--",
                        label="충전 전류적산", color="#55A868", linewidth=1.5)
            ax.set_xlabel("사이클")
            ax.set_ylabel("용량 (Ah)")
            ax.set_title(f"[{ds}] 셀 {example_cell} — 등록 capacity_Ah vs 전류적산값")
            ax.legend()
        fig.tight_layout()

        out_path = out_dir / f"capacity_error_curve_example_{ds}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  대표 셀 곡선 플랏 저장: {out_path}  (셀={example_cells})")
        out_paths.append(out_path)
    return out_paths


def plot_by_intensity(raw_df: pd.DataFrame, out_path: Path) -> None:
    groups = ["Q1(약)", "Q2", "Q3", "Q4(강)"]
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(groups))
    width = 0.35
    for offset, ds, color in zip((-1, 1), DATASETS, ["#4C72B0", "#DD8452"]):
        vals = [
            raw_df.loc[(raw_df["dataset"] == ds) & (raw_df["intensity_group"] == g), "dis_error_pct"].values
            for g in groups
        ]
        bp = ax.boxplot(
            vals, positions=positions + offset * width / 2, widths=width * 0.9,
            showfliers=False, patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_xlabel("평균 방전전류 4분위 그룹 (전체 셀 pooled 분위, 약→강)")
    ax.set_ylabel("오차율 (%)")
    ax.set_title("(c) 충방전 세기 4군별 오차율 (방전 기준)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#4C72B0", "#DD8452"]]
    ax.legend(handles, DATASETS, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── 요구사항 1-(2): 등록용량의 비단조/점프(=캘리브레이션 의심 지점) 요약 ─────────
# 주의: 입력이 _4_data_hi/clean(2_preprocess의 7단계 필터 완료본)이므로, 이미 문서화된
# RPT/HPPC성 이상치(docs/DATASET_ANOMALIES.md §1-E, 23개/0.02%)는 이 시점엔 대부분
# 제거된 뒤다. 즉 여기서는 "정리된 데이터에 남은 잔여 비단조성"만 잡을 수 있고,
# 원본 캘리브레이션 시점·주기 자체는 §4.2에서 이미 분석·문서화된 결론(체계적 RPT
# 없음, 산발적)을 그대로 인용한다 — 이 스크립트가 새로 그 존재를 재탐색하지는 않는다.

def summarize_monotonicity(raw_df: pd.DataFrame) -> pd.DataFrame:
    """셀별로 등록용량(registered_Ah)이 직전 사이클보다 '유의하게' 증가하는 비율.

    캘리브레이션/RPT라면 등록용량이 갑자기 위로 튀는 형태로 나타난다(§4.2 사례:
    Batch3 마지막 사이클 0.88Ah → RPT 1.76Ah). 노이즈성 미세 증가와 구분하기 위해
    "직전 대비 +1% 초과 증가"만 카운트한다(임계값은 대략치 — 정밀 RPT 판별이 아니라
    Phase 2-(a) 단조성 검증을 위한 빠른 스크리닝용).
    """
    rows = []
    for (ds, cid), grp in raw_df.sort_values("cycle").groupby(["dataset", "cell_id"]):
        reg = grp["registered_Ah"].to_numpy()
        if len(reg) < 3:
            continue
        rel_delta = np.diff(reg) / np.clip(reg[:-1], 1e-9, None)
        n_jump_up = int((rel_delta > 0.01).sum())
        rows.append({
            "dataset": ds, "cell_id": cid, "n_cycles": len(reg),
            "n_jump_up_gt1pct": n_jump_up,
            "jump_up_ratio": n_jump_up / max(len(reg) - 1, 1),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cells", type=int, default=None,
                        help="데이터셋별 샘플링할 셀 수(기본: 전체). 빠른 확인용으로 20~30 권장.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Step 1 병렬 프로세스 수(기본: 1 = 순차 실행)")
    parser.add_argument("--skip-unit-check", action="store_true",
                        help="[위험] Step 0 단위 검증을 건너뛴다 — 디버그 전용, 기본적으로 쓰지 말 것.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 0 ──────────────────────────────────────────────────────────────
    report = step0_unit_check(args.n_cells)
    _print_step0_report(report)
    if not args.skip_unit_check:
        assert_units_consistent(report)
    else:
        print("\n[경고] --skip-unit-check 지정됨 — 단위 검증을 건너뛰고 진행합니다.")

    # ── Step 1 ──────────────────────────────────────────────────────────────
    raw_df = step1_collect(args.n_cells, args.workers)
    if raw_df.empty:
        raise RuntimeError("Step 1: 유효한 오차 기록이 하나도 없습니다 — 입력 경로/스키마를 확인하세요.")

    raw_df = assign_intensity_group(raw_df)
    raw_df.to_csv(OUT_DIR / "capacity_error_raw.csv", index=False)

    cell_summary, group_summary = summarize(raw_df)
    cell_summary.to_csv(OUT_DIR / "capacity_error_cell_summary.csv", index=False)
    group_summary.to_csv(OUT_DIR / "capacity_error_group_summary.csv", index=False)

    mono_df = summarize_monotonicity(raw_df)
    mono_df.to_csv(OUT_DIR / "capacity_error_monotonicity.csv", index=False)

    plot_by_dataset(raw_df, OUT_DIR / "capacity_error_by_dataset.png")
    plot_by_cell(cell_summary, OUT_DIR / "capacity_error_by_cell.png")
    plot_by_intensity(raw_df, OUT_DIR / "capacity_error_by_intensity.png")
    plot_by_current_direction(raw_df, OUT_DIR / "capacity_error_by_current.png")
    plot_capacity_curve_example(raw_df, OUT_DIR)

    # ── 콘솔 요약 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Step 1 — 등록capacity_Ah vs 전류적산 오차율 요약")
    print("=" * 72)
    for ds in DATASETS:
        sub = raw_df[raw_df["dataset"] == ds]["dis_error_pct"]
        if len(sub) == 0:
            continue
        print(f"  [{ds}] n_cycles={len(sub)}  mean={sub.mean():+.3f}%  std={sub.std():.3f}%  "
              f"median={sub.median():+.3f}%  |mean|abs={sub.abs().mean():.3f}%")
    print("-" * 72)
    print(group_summary.to_string(index=False))
    print("-" * 72)
    print(f"  등록용량 비단조(+1%초과 점프) 비율 — 전체 평균: "
          f"{mono_df['jump_up_ratio'].mean() * 100:.3f}%  "
          f"(§4.2 참고: 원본 RPT/HPPC성 이상치는 0.02%, 이미 클리닝 단계에서 대부분 제거됨)")
    print("=" * 72)
    print(f"\n[완료] 출력 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
