"""
convert_unified.py

MIT(FastCharge) / HUST 데이터를 배터리 ID별 통일 구조로 변환.
MIT는 .mat (HDF5) 파일 직접 파싱, HUST는 원본 pkl 변환.
병렬 처리 지원 (--workers N, 기본 3).

파일 포맷:
  MIT  : data/FastCharge/*.mat  (HDF5 배치 파일 3개)
  HUST : data/our_data/our_data/{cell_id}.pkl

출력:
  data_raw/MIT/{bNcN}.pkl  + .csv       (이상치 제거 없는 파싱 원본, DELETE_CELLS 포함)
  data_raw/HUST/{cell_id}.pkl + .csv
  data_unified/MIT/{bNcN}.pkl  + .csv   (b1c0 ~ b3c46, 필터 적용)
  data_unified/HUST/{cell_id}.pkl + .csv
  docs/mit_conversion_summary.csv
  docs/hust_conversion_summary.csv

cycles DataFrame 컬럼:
  cycle, time_s, voltage_V, current_A, temperature_C, capacity_Ah, phase

사용:
  python convert_unified.py --dataset all --workers 3
  python convert_unified.py --dataset mit
  python convert_unified.py --dataset hust --workers 6
  python convert_unified.py --dataset hust --cell 1-1   # HUST 단일 셀
  python convert_unified.py --no-cache                  # 캐시 무시, 원본부터 재변환

캐시 동작:
  data_raw/MIT/ 또는 data_raw/HUST/ 에 PKL 파일이 있으면 MAT/원본 파싱을 건너뛰고
  캐시에서 로드 후 동일한 이상치 제거 파이프라인을 적용해 data_unified/ 를 재생성.
  캐시 PKL에는 누적 시간이 미적용된 상태(사이클별 상대 시간)로 저장됨.
"""

import argparse
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MIT_MAT_DIR   = PROJECT_ROOT / "data_raw" / "FastCharge"
HUST_PKL_DIR  = PROJECT_ROOT / "data_raw" / "our_data" / "our_data"
OUTPUT_ROOT     = PROJECT_ROOT / "data_unified"
RAW_OUTPUT_ROOT = PROJECT_ROOT / "data_raw"

PHASE_POS =  0.01   # A 초과 → charge
PHASE_NEG = -0.01   # A 미만 → discharge


# ---------------------------------------------------------------------------
# MIT MAT 상수
# ---------------------------------------------------------------------------

MAT_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

DELETE_CELLS = {
    "batch1": {"b1c8", "b1c10", "b1c12", "b1c13", "b1c22",
               "b1c18"},   # rest 구간 전압 오염(cycle 34~53), 측정 이상
    "batch3": {"b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"},
}

# (batch1_key, batch2_key) — Batch2 연속 셀 → Batch1에 병합
CONTINUING = [
    ("b1c0", "b2c7"),
    ("b1c1", "b2c8"),
    ("b1c2", "b2c9"),
    ("b1c3", "b2c15"),
    ("b1c4", "b2c16"),
]
CONTINUING_B2_KEYS = {b2k for _, b2k in CONTINUING}


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def assign_phase(current_A: np.ndarray) -> list:
    return [
        "charge" if i > PHASE_POS else ("discharge" if i < PHASE_NEG else "rest")
        for i in current_A
    ]


def save_cell(out_dir: Path, cell_id: str, meta: dict, df: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = cell_id.replace("/", "-")
    with open(out_dir / f"{safe_id}.pkl", "wb") as f:
        pickle.dump({"meta": meta, "cycles": df}, f)
    df.to_csv(out_dir / f"{safe_id}.csv", index=False)


def _remove_outlier_cycles(df: pd.DataFrame,
                           window: int = 11,
                           sigma: float = 2.5,
                           min_std: float = 0.01) -> tuple:
    """Rolling-median 기반 이상 사이클 제거 (MIT·HUST 공통).

    window 내 중앙값에서 sigma × std 이상 벗어난 사이클을 제거.
    - RPT  (고용량, ~+70% 급상승) : 상단 이상치
    - HPPC (저용량, 펄스 진단)    : 하단 이상치
    min_std: 정상 열화 구간 std 플로어 — 너무 작은 std로 인한 오탐 방지.
    """
    dis   = df[df["phase"] == "discharge"]
    cap_s = dis.groupby("cycle")["capacity_Ah"].first().dropna().sort_index()

    if len(cap_s) < window:
        return df, 0

    roll  = cap_s.rolling(window=window, center=True, min_periods=3)
    r_med = roll.median()
    r_std = roll.std().fillna(cap_s.std()).clip(lower=min_std)

    outlier_cycles = set(cap_s[(cap_s - r_med).abs() > sigma * r_std].index)
    if not outlier_cycles:
        return df, 0

    df_clean = df[~df["cycle"].isin(outlier_cycles)].copy().reset_index(drop=True)
    return df_clean, len(outlier_cycles)


def _remove_empty_cycles(df: pd.DataFrame, min_active_rows: int = 5) -> tuple:
    """charge/discharge 행이 min_active_rows 미만인 사이클 제거.

    MIT cycle 1처럼 rest 2행만 존재하는 초기화 아티팩트를 제거.
    """
    active = df[df["phase"].isin(["charge", "discharge"])]
    active_counts = active.groupby("cycle").size()
    bad_cycles = set(active_counts[active_counts < min_active_rows].index)
    # active 행이 아예 없는 사이클도 포함
    all_cycles = set(df["cycle"].unique())
    active_cycles = set(active_counts.index)
    bad_cycles |= (all_cycles - active_cycles)

    if not bad_cycles:
        return df, 0
    df_clean = df[~df["cycle"].isin(bad_cycles)].copy().reset_index(drop=True)
    return df_clean, len(bad_cycles)


def _remove_zero_current_rest(df: pd.DataFrame) -> tuple:
    """rest 행 중 current_A == 0.0인 행 제거.

    _fix_time_monotonicity 이후에 호출.
    """
    mask = (df["phase"] == "rest") & (df["current_A"] == 0.0)
    n = int(mask.sum())
    if n == 0:
        return df, 0
    return df[~mask].copy().reset_index(drop=True), n


def _fix_time_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    """사이클 내 time_s 역방향 점프를 단조 증가로 보정.

    사이클 간 누적 오프셋은 적용하지 않으며 사이클마다 시간이 초기화됨.
    MIT 원본 타임스탬프의 분 단위 정밀도 문제로 발생하는 사이클 내 역방향 점프를
    np.maximum.accumulate로 제거.
    """
    df = df.copy()
    for cyc in sorted(df["cycle"].unique()):
        mask = df["cycle"] == cyc
        t = df.loc[mask, "time_s"].values.astype(float)
        t = np.maximum.accumulate(t)          # 사이클 내 단조성 보정
        df.loc[mask, "time_s"] = t
    return df


# ---------------------------------------------------------------------------
# MIT MAT 파싱 유틸 (top-level — Windows ProcessPoolExecutor 호환)
# ---------------------------------------------------------------------------

def _read_ref(f, ref):
    """HDF5 Reference → 1D numpy float array."""
    data = f[ref][()]
    return np.atleast_1d(data.flatten()).astype(float)


def _load_summary_field(f, s_grp, key):
    """
    summary group의 한 필드를 1D float array로 반환.

    MATLAB HDF5 저장 방식:
      1) Cell array → (1, N) HDF5 Reference 배열
      2) Numeric array → 직접 float 값 배열
    """
    raw  = s_grp[key][()]
    flat = raw.flatten()
    if len(flat) == 0:
        return np.array([], dtype=float)
    if isinstance(flat[0], h5py.Reference):
        return np.hstack([_read_ref(f, r) for r in flat]).astype(float)
    return flat.astype(float)


def _load_cycle_field(f, c_grp, field, j):
    """
    cycles group의 j번째 사이클 한 필드를 1D float array로 반환.

    저장 방식:
      1) (n_cycles, 1) HDF5 Reference 배열 → 역참조
      2) (n_cycles, n_points) 직접 float 배열 → j번째 행
    """
    raw  = c_grp[field][()]
    elem = raw[j, 0]
    if isinstance(elem, h5py.Reference):
        return _read_ref(f, elem)
    return np.atleast_1d(raw[j]).flatten().astype(float)


def _build_cell_df(cycle_nums, QD_arr, cycles_raw, policy):
    """사이클 raw 데이터 리스트 → DataFrame 변환."""
    rows = []
    for cyc_num, QD, raw in zip(cycle_nums, QD_arr, cycles_raw):
        cyc = int(cyc_num)
        if cyc == 0:   # 출고 진단 사이클 제외
            continue

        I, V, T = raw["I"], raw["V"], raw["T"]
        t = raw["t"]

        n = len(V)
        if n == 0:
            continue

        phase = assign_phase(I)

        rows.append(pd.DataFrame({
            "cycle":         cyc,
            "time_s":        (t - t[0]) * 60.0,   # minutes → seconds, 상대 시간
            "voltage_V":     V,
            "current_A":     I,
            "temperature_C": T,
            "capacity_Ah":   float(QD) if np.isfinite(QD) else np.nan,
            "phase":         phase,
        }))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _parse_batch(args):
    """
    MAT 파일 1개를 파싱해 셀별 raw 데이터 dict를 반환.
    top-level 함수 — Windows ProcessPoolExecutor 필수.

    반환: (batch_key, {cell_key: {cycle_nums, QD_arr, cycle_life, policy, cycles_raw}})
    """
    batch_key, mat_path_str, prefix = args
    mat_path = Path(mat_path_str)
    print(f"  [{batch_key}] 파싱 시작: {mat_path.name}", flush=True)

    result = {}
    with h5py.File(mat_path, "r") as f:
        batch_grp = f["batch"]
        num_cells = int(batch_grp["summary"].shape[0])
        print(f"  [{batch_key}] 총 셀 수: {num_cells}", flush=True)

        for i in range(num_cells):
            cell_key = f"{prefix}c{i}"
            try:
                cl_raw = float(_read_ref(f, batch_grp["cycle_life"][i, 0]).flat[0])
                cl     = int(cl_raw) if np.isfinite(cl_raw) else None

                pol_raw = f[batch_grp["policy_readable"][i, 0]][()]
                policy  = pol_raw.tobytes()[::2].decode("utf-8", errors="replace").strip("\x00")

                s_grp      = f[batch_grp["summary"][i, 0]]
                cycle_nums = _load_summary_field(f, s_grp, "cycle")
                QD_arr     = _load_summary_field(f, s_grp, "QDischarge")

                c_grp    = f[batch_grp["cycles"][i, 0]]
                n_cycles = int(c_grp["I"].shape[0])

                cycles_raw = [
                    {
                        "I":  _load_cycle_field(f, c_grp, "I",  j),
                        "V":  _load_cycle_field(f, c_grp, "V",  j),
                        "T":  _load_cycle_field(f, c_grp, "T",  j),
                        "Qd": _load_cycle_field(f, c_grp, "Qd", j),
                        "Qc": _load_cycle_field(f, c_grp, "Qc", j),
                        "t":  _load_cycle_field(f, c_grp, "t",  j),
                    }
                    for j in range(n_cycles)
                ]

                result[cell_key] = {
                    "cycle_nums":  cycle_nums,
                    "QD_arr":      QD_arr,
                    "cycle_life":  cl,
                    "policy":      policy,
                    "cycles_raw":  cycles_raw,
                }

                if (i + 1) % 10 == 0 or i + 1 == num_cells:
                    print(f"  [{batch_key}] {i+1}/{num_cells} 셀 완료", flush=True)

            except Exception as exc:
                print(f"  [WARN] {cell_key} 파싱 실패: {exc}", flush=True)
                traceback.print_exc()

    print(f"  [{batch_key}] 파싱 완료: {len(result)}개 셀", flush=True)
    return batch_key, result


def _process_and_save_mit(cell_key: str, cell_data: dict, batch_num: int,
                          out_dir: Path, raw_out_dir: Path,
                          is_deleted: bool = False,
                          df_extra: pd.DataFrame = None) -> dict:
    """셀 raw 데이터 → raw 저장 → (불량 셀 제외) 이상치 제거 → unified 저장."""
    df = _build_cell_df(cell_data["cycle_nums"], cell_data["QD_arr"],
                        cell_data["cycles_raw"], cell_data["policy"])
    if df.empty:
        return {}

    # 연속 셀(batch2 → batch1 병합)
    if df_extra is not None and not df_extra.empty:
        df_extra  = df_extra.copy()
        b1_last   = df["cycle"].max()
        b2_start  = df_extra["cycle"].min()
        df_extra["cycle"] = df_extra["cycle"] + (b1_last + 1 - b2_start)
        df = pd.concat([df, df_extra], ignore_index=True)

    # ── raw 저장 (이상치 제거 없음, 누적 시간 미적용, DELETE_CELLS 포함) ─────
    # 누적 시간 미적용: 캐시 로드 시 _remove_empty_cycles → _fix_time_monotonicity
    # 순서를 unified 파이프라인과 동일하게 유지하기 위함
    save_cell(raw_out_dir, cell_key, {
        "cell_id":       cell_key,
        "dataset":       "MIT",
        "batch":         batch_num,
        "charge_policy": cell_data["policy"],
        "n_cycles":      df["cycle"].nunique(),
    }, df)

    # 불량 셀은 data_unified에 저장하지 않음
    if is_deleted:
        return {}

    # rest-only 빈 사이클 제거 (MIT cycle 1 초기화 아티팩트 등)
    df, n_empty = _remove_empty_cycles(df)
    if df.empty:
        return {}

    # time_s 단조 누적 보정 (사이클 간 불연속 + 사이클 내 역방향 점프)
    df = _fix_time_monotonicity(df)

    # rest 행 중 전류 0.0A 제거 (_fix_time_monotonicity 이후)
    df, n_rest_removed = _remove_zero_current_rest(df)

    df, n_outliers = _remove_outlier_cycles(df)
    n_actual = df["cycle"].nunique()

    dis_caps  = (df[df["phase"] == "discharge"]
                 .groupby("cycle")["capacity_Ah"].first()
                 .dropna().sort_index())
    init_cap  = float(dis_caps.iloc[0])  if len(dis_caps) > 0 else np.nan
    final_cap = float(dis_caps.iloc[-1]) if len(dis_caps) > 0 else np.nan

    meta = {
        "cell_id":            cell_key,
        "dataset":            "MIT",
        "batch":              batch_num,
        "charge_policy":      cell_data["policy"],
        "cycle_life_raw":     cell_data["cycle_life"],
        "n_cycles":             n_actual,
        "n_empty_removed":      n_empty,
        "n_rest_removed":       n_rest_removed,
        "n_outliers_removed":   n_outliers,
        "init_cap_Ah":        round(init_cap,  4) if np.isfinite(init_cap)  else None,
        "final_cap_Ah":       round(final_cap, 4) if np.isfinite(final_cap) else None,
    }
    save_cell(out_dir, cell_key, meta, df)

    return {
        "cell_id":        cell_key,
        "n_cycles":       n_actual,
        "n_empty":        n_empty,
        "n_rest_removed": n_rest_removed,
        "n_outliers":     n_outliers,
        "init_cap_Ah":    round(init_cap,  4) if np.isfinite(init_cap)  else "",
        "final_cap_Ah":   round(final_cap, 4) if np.isfinite(final_cap) else "",
        "total_rows":     len(df),
    }


# ---------------------------------------------------------------------------
# MIT 변환 진입점
# ---------------------------------------------------------------------------

def convert_mit(out_root: Path, target_cell: str = None, n_workers: int = 3,
                no_cache: bool = False):
    out_dir = out_root / "MIT"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_dir = RAW_OUTPUT_ROOT / "MIT"

    if target_cell:
        print("[MIT] --cell 옵션은 MAT 변환에서 지원하지 않습니다 (전체 변환 진행)")

    # ── 캐시 확인 ─────────────────────────────────────────────────────────────
    raw_pkls  = sorted(raw_out_dir.glob("*.pkl")) if raw_out_dir.exists() else []
    use_cache = bool(raw_pkls) and not no_cache
    t0        = time.time()
    all_stats = []

    if use_cache:
        # ── 캐시 모드: data_raw/MIT/ PKL → 이상치 제거 → data_unified ────────
        print(f"\n=== MIT 캐시 모드: {len(raw_pkls)}개 셀  ({raw_out_dir}) ===")
        all_delete = {k for keys in DELETE_CELLS.values() for k in keys}
        for key in sorted(all_delete):
            for ext in (".pkl", ".csv"):
                p = out_dir / f"{key}{ext}"
                if p.exists():
                    p.unlink()
        args_list = [
            (pkl.stem, str(out_dir), str(raw_out_dir), pkl.stem in all_delete)
            for pkl in raw_pkls
        ]
        records, _ = _run(args_list, _mit_cache_worker, "MIT cells (cache)", n_workers)
        all_stats  = [r for r in records if r]

    else:
        # ── 전체 변환 모드: MAT 파일 파싱 ────────────────────────────────────
        print("=== MAT 파일 확인 ===")
        tasks = []
        for batch_key, filename in MAT_FILES.items():
            mat_path = MIT_MAT_DIR / filename
            if not mat_path.exists():
                print(f"[ERROR] 파일 없음: {mat_path}")
                return
            size_gb = mat_path.stat().st_size / 1e9
            prefix  = "b" + batch_key[-1]
            print(f"  {batch_key}: {filename}  ({size_gb:.2f} GB)")
            tasks.append((batch_key, str(mat_path), prefix))

        n_parse = min(n_workers, len(tasks))
        print(f"\n=== 병렬 파싱 (workers={n_parse}) ===\n")

        raw_results = {}
        with ProcessPoolExecutor(max_workers=n_parse) as pool:
            futures = {pool.submit(_parse_batch, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                bk, cell_dict = fut.result()
                raw_results[bk] = cell_dict
                print(f"  ✓ [{bk}] {len(cell_dict)}개 셀  ({time.time()-t0:.0f}s)", flush=True)

        batch1_raw = raw_results["batch1"]
        batch2_raw = raw_results["batch2"]
        batch3_raw = raw_results["batch3"]

        raw_out_dir.mkdir(parents=True, exist_ok=True)

        all_delete = {k for keys in DELETE_CELLS.values() for k in keys}
        print("\n=== data_unified 제외 셀 (data_raw에는 포함) ===")
        for key in sorted(all_delete):
            print(f"  {key}")
        for key in sorted(all_delete):
            for ext in (".pkl", ".csv"):
                p = out_dir / f"{key}{ext}"
                if p.exists():
                    p.unlink()
                    print(f"  기존 파일 삭제: {p.name}")

        print("\n=== Batch2 연속 셀 DataFrame 변환 ===")
        cont_dfs = {}
        for b1k, b2k in CONTINUING:
            if b2k not in batch2_raw:
                continue
            d = batch2_raw[b2k]
            df_b2 = _build_cell_df(d["cycle_nums"], d["QD_arr"],
                                    d["cycles_raw"], d["policy"])
            cont_dfs[b2k] = df_b2
            print(f"  {b2k}: {df_b2['cycle'].nunique()}개 사이클")
            del batch2_raw[b2k]

        print(f"\n=== DataFrame 변환 + 저장  (workers={n_workers}) ===")
        print(f"  raw     → {raw_out_dir}")
        print(f"  unified → {out_dir}\n")

        for batch_label, batch_raw, batch_num in [
            ("batch1", batch1_raw, 1),
            ("batch2", batch2_raw, 2),
            ("batch3", batch3_raw, 3),
        ]:
            print(f"--- {batch_label} ({len(batch_raw)}개 셀) ---")
            args_list = []
            for cell_key in sorted(batch_raw, key=lambda k: int(k[3:])):
                is_deleted = cell_key in all_delete
                df_extra = None
                if batch_num == 1:
                    df_extra = next((cont_dfs[b2k] for b1k, b2k in CONTINUING
                                     if cell_key == b1k and b2k in cont_dfs), None)
                args_list.append((cell_key, batch_raw[cell_key],
                                  batch_num, str(out_dir), str(raw_out_dir),
                                  is_deleted, df_extra))
            records, _ = _run(args_list, _mit_cell_worker, batch_label, n_workers)
            all_stats.extend(r for r in records if r)

    # ── 요약 (공통) ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    b1 = sum(1 for s in all_stats if s["cell_id"].startswith("b1"))
    b2 = sum(1 for s in all_stats if s["cell_id"].startswith("b2"))
    b3 = sum(1 for s in all_stats if s["cell_id"].startswith("b3"))
    mode = "캐시" if use_cache else "변환"

    print(f"\n=== MIT {mode} 완료 ===")
    print(f"  batch1: {b1}개  batch2: {b2}개  batch3: {b3}개  합계: {len(all_stats)}개")
    print(f"  총 소요시간: {elapsed//60:.0f}분 {elapsed%60:.0f}초")
    print(f"  저장 위치: {out_dir}")

    if all_stats:
        docs_dir = PROJECT_ROOT / "docs"
        docs_dir.mkdir(exist_ok=True)
        csv_path = docs_dir / "mit_conversion_summary.csv"
        pd.DataFrame(all_stats).sort_values("cell_id").to_csv(csv_path, index=False)
        print(f"  요약 CSV: {csv_path}")


# ---------------------------------------------------------------------------
# HUST 변환
# ---------------------------------------------------------------------------

def convert_hust_cell(pkl_path: Path, out_dir: Path, raw_out_dir: Path) -> dict:
    """변환 후 stats dict 반환. 스킵 시 빈 dict."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    cell_id = pkl_path.stem
    inner   = raw[cell_id]
    dq_map  = {int(k): float(v) for k, v in inner["dq"].items()}
    data    = {int(k): v        for k, v in inner["data"].items()}

    rows        = []
    n_charge    = 0
    n_discharge = 0

    for cyc in sorted(data.keys()):
        df_cyc = data[cyc]
        if df_cyc is None or len(df_cyc) == 0:
            continue

        t_arr  = df_cyc["Time (s)"].values.astype(float)     if "Time (s)"     in df_cyc.columns else np.arange(len(df_cyc), dtype=float)
        v_arr  = df_cyc["Voltage (V)"].values.astype(float)  if "Voltage (V)"  in df_cyc.columns else np.full(len(df_cyc), np.nan)
        i_arr  = df_cyc["Current (mA)"].values.astype(float) / 1000.0 if "Current (mA)" in df_cyc.columns else np.zeros(len(df_cyc))
        cap_Ah = dq_map.get(cyc, np.nan) / 1000.0  # mAh → Ah

        # "discharge"가 "charge"를 포함하므로 discharge 먼저 검사
        if "Status" in df_cyc.columns:
            phase = [
                "discharge" if "discharge" in str(s).lower() else
                "charge"    if "charge"    in str(s).lower() else "rest"
                for s in df_cyc["Status"]
            ]
        else:
            phase = assign_phase(i_arr)

        if any(p == "charge"    for p in phase): n_charge    += 1
        if any(p == "discharge" for p in phase): n_discharge += 1

        rows.append(pd.DataFrame({
            "cycle":         cyc,
            "time_s":        t_arr,
            "voltage_V":     v_arr,
            "current_A":     i_arr,
            "temperature_C": 30.0,
            "capacity_Ah":   cap_Ah,
            "phase":         phase,
        }))

    if not rows:
        return {}

    df = pd.concat(rows, ignore_index=True)

    # ── raw 저장 (이상치 제거 없음, 누적 시간 미적용) ────────────────────────
    save_cell(raw_out_dir, cell_id, {
        "cell_id":       cell_id,
        "dataset":       "HUST",
        "batch_id":      cell_id.split("-")[0],
        "n_cycles":      df["cycle"].nunique(),
        "temperature_C": 30.0,
    }, df)

    # ── 이상치 제거 후 data_unified에 저장 ───────────────────────────────────
    df = _fix_time_monotonicity(df)
    df, n_rest_removed = _remove_zero_current_rest(df)
    df, n_outliers = _remove_outlier_cycles(df)

    meta = {
        "cell_id":            cell_id,
        "dataset":            "HUST",
        "batch_id":           cell_id.split("-")[0],
        "n_cycles":           df["cycle"].nunique(),
        "temperature_C":      30.0,
        "n_rest_removed":     n_rest_removed,
        "n_outliers_removed": n_outliers,
    }
    save_cell(out_dir, cell_id, meta, df)

    sorted_cycs = sorted(dq_map.keys())
    init_cap  = dq_map[sorted_cycs[0]]  / 1000.0 if sorted_cycs else np.nan
    final_cap = dq_map[sorted_cycs[-1]] / 1000.0 if sorted_cycs else np.nan

    return {
        "cell_id":          cell_id,
        "batch_id":         cell_id.split("-")[0],
        "total_cycles":     len(data),
        "charge_cycles":    n_charge,
        "discharge_cycles": n_discharge,
        "total_rows":       len(df),
        "init_cap_Ah":      round(init_cap,  4) if np.isfinite(init_cap)  else "",
        "final_cap_Ah":     round(final_cap, 4) if np.isfinite(final_cap) else "",
    }


def _mit_cell_worker(args):
    """top-level 함수 — Windows ProcessPoolExecutor 필수."""
    cell_key, cell_data, batch_num, out_dir_str, raw_out_dir_str, is_deleted, df_extra = args
    try:
        stats = _process_and_save_mit(cell_key, cell_data, batch_num,
                                      Path(out_dir_str), Path(raw_out_dir_str),
                                      is_deleted=is_deleted, df_extra=df_extra)
        return ("ok", stats)
    except Exception:
        return ("err", f"{cell_key}:\n{traceback.format_exc()}")


def _hust_worker(args):
    """top-level 함수 — Windows ProcessPoolExecutor 필수."""
    pkl_path_str, out_dir_str, raw_out_dir_str = args
    try:
        stats = convert_hust_cell(Path(pkl_path_str), Path(out_dir_str), Path(raw_out_dir_str))
        return ("ok", stats)
    except Exception:
        return ("err", f"{Path(pkl_path_str).stem}:\n{traceback.format_exc()}")


def _mit_cache_worker(args):
    """top-level 함수 — data_raw/MIT/ PKL → 이상치 제거 → data_unified/MIT/ 저장."""
    cell_key, out_dir_str, raw_out_dir_str, is_deleted = args
    try:
        if is_deleted:
            return ("ok", {})
        raw_pkl = Path(raw_out_dir_str) / f"{cell_key}.pkl"
        with open(raw_pkl, "rb") as f:
            raw = pickle.load(f)
        df       = raw["cycles"]
        meta_raw = raw["meta"]

        df, n_empty = _remove_empty_cycles(df)
        if df.empty:
            return ("ok", {})
        df = _fix_time_monotonicity(df)
        df, n_rest_removed = _remove_zero_current_rest(df)
        df, n_outliers = _remove_outlier_cycles(df)
        n_actual = df["cycle"].nunique()

        dis_caps = (df[df["phase"] == "discharge"]
                    .groupby("cycle")["capacity_Ah"].first()
                    .dropna().sort_index())
        init_cap  = float(dis_caps.iloc[0])  if len(dis_caps) > 0 else np.nan
        final_cap = float(dis_caps.iloc[-1]) if len(dis_caps) > 0 else np.nan

        meta = {
            "cell_id":            cell_key,
            "dataset":            "MIT",
            "batch":              meta_raw.get("batch"),
            "charge_policy":      meta_raw.get("charge_policy", ""),
            "cycle_life_raw":     None,
            "n_cycles":             n_actual,
            "n_empty_removed":      n_empty,
            "n_rest_removed":       n_rest_removed,
            "n_outliers_removed":   n_outliers,
            "init_cap_Ah":        round(init_cap,  4) if np.isfinite(init_cap)  else None,
            "final_cap_Ah":       round(final_cap, 4) if np.isfinite(final_cap) else None,
        }
        save_cell(Path(out_dir_str), cell_key, meta, df)
        return ("ok", {
            "cell_id":        cell_key,
            "n_cycles":       n_actual,
            "n_empty":        n_empty,
            "n_rest_removed": n_rest_removed,
            "n_outliers":     n_outliers,
            "init_cap_Ah":    round(init_cap,  4) if np.isfinite(init_cap)  else "",
            "final_cap_Ah":   round(final_cap, 4) if np.isfinite(final_cap) else "",
            "total_rows":     len(df),
        })
    except Exception:
        return ("err", f"{cell_key}:\n{traceback.format_exc()}")


def _hust_cache_worker(args):
    """top-level 함수 — data_raw/HUST/ PKL → 이상치 제거 → data_unified/HUST/ 저장."""
    cell_key, out_dir_str, raw_out_dir_str = args
    try:
        raw_pkl = Path(raw_out_dir_str) / f"{cell_key}.pkl"
        with open(raw_pkl, "rb") as f:
            raw = pickle.load(f)
        df       = raw["cycles"]
        meta_raw = raw["meta"]

        df = _fix_time_monotonicity(df)
        df, n_rest_removed = _remove_zero_current_rest(df)
        df, n_outliers = _remove_outlier_cycles(df)

        cyc_phases  = df.groupby("cycle")["phase"].apply(set)
        n_charge    = int(sum(1 for p in cyc_phases if "charge"    in p))
        n_discharge = int(sum(1 for p in cyc_phases if "discharge" in p))

        dis_caps = (df[df["phase"] == "discharge"]
                    .groupby("cycle")["capacity_Ah"].first()
                    .dropna().sort_index())
        init_cap  = float(dis_caps.iloc[0])  if len(dis_caps) > 0 else np.nan
        final_cap = float(dis_caps.iloc[-1]) if len(dis_caps) > 0 else np.nan

        meta = {
            "cell_id":            cell_key,
            "dataset":            "HUST",
            "batch_id":           meta_raw.get("batch_id", cell_key.split("-")[0]),
            "n_cycles":           df["cycle"].nunique(),
            "temperature_C":      30.0,
            "n_rest_removed":     n_rest_removed,
            "n_outliers_removed": n_outliers,
        }
        save_cell(Path(out_dir_str), cell_key, meta, df)
        return ("ok", {
            "cell_id":          cell_key,
            "batch_id":         meta_raw.get("batch_id", cell_key.split("-")[0]),
            "total_cycles":     df["cycle"].nunique(),
            "charge_cycles":    n_charge,
            "discharge_cycles": n_discharge,
            "total_rows":       len(df),
            "init_cap_Ah":      round(init_cap,  4) if np.isfinite(init_cap)  else "",
            "final_cap_Ah":     round(final_cap, 4) if np.isfinite(final_cap) else "",
        })
    except Exception:
        return ("err", f"{cell_key}:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 공통 실행 헬퍼
# ---------------------------------------------------------------------------

def _run(args_list: list, worker_fn, desc: str, n_workers: int):
    """순차 or 병렬 실행 → (records, error_count)."""
    records = []
    errors  = 0

    if n_workers <= 1:
        for args in tqdm(args_list, desc=desc):
            status, payload = worker_fn(args)
            if status == "ok":
                if payload: records.append(payload)
            else:
                errors += 1
                print(f"\n  [ERR] {payload}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(worker_fn, a): a for a in args_list}
            with tqdm(total=len(args_list), desc=desc) as pbar:
                for fut in as_completed(futures):
                    status, payload = fut.result()
                    if status == "ok":
                        if payload: records.append(payload)
                    else:
                        errors += 1
                        print(f"\n  [ERR] {payload}")
                    pbar.update(1)

    return records, errors




def convert_hust(out_root: Path, target_cell: str = None, n_workers: int = 4,
                 no_cache: bool = False):
    out_dir     = out_root / "HUST"
    raw_out_dir = RAW_OUTPUT_ROOT / "HUST"

    # ── 캐시 확인 ─────────────────────────────────────────────────────────────
    raw_pkls  = sorted(raw_out_dir.glob("*.pkl")) if raw_out_dir.exists() else []
    use_cache = bool(raw_pkls) and not no_cache

    if use_cache:
        if target_cell:
            raw_pkls = [p for p in raw_pkls if p.stem == target_cell]
            if not raw_pkls:
                print(f"[HUST] 캐시에서 cell '{target_cell}' 없음")
                return
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[HUST] 캐시 모드: {len(raw_pkls)}개 셀  ({raw_out_dir})")
        print(f"  unified → {out_dir}")
        args_list = [(pkl.stem, str(out_dir), str(raw_out_dir)) for pkl in raw_pkls]
        records, errors = _run(args_list, _hust_cache_worker, "HUST cells (cache)", n_workers)

    else:
        pkl_files = sorted(
            HUST_PKL_DIR.glob("*.pkl"),
            key=lambda p: [int(x) for x in p.stem.split("-")]
        )
        if not pkl_files:
            print(f"[HUST] pkl 파일 없음: {HUST_PKL_DIR}")
            return
        if target_cell:
            pkl_files = [p for p in pkl_files if p.stem == target_cell]
            if not pkl_files:
                print(f"[HUST] cell '{target_cell}' 없음")
                return

        out_dir.mkdir(parents=True, exist_ok=True)
        raw_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[HUST] {len(pkl_files)}개 셀 변환  (workers={n_workers})")
        print(f"  raw     → {raw_out_dir}")
        print(f"  unified → {out_dir}")

        args_list = [(str(p), str(out_dir), str(raw_out_dir)) for p in pkl_files]
        records, errors = _run(args_list, _hust_worker, "HUST cells", n_workers)

    print(f"  완료: {len(records)} 성공, {errors} 실패")
    if records:
        docs_dir = PROJECT_ROOT / "docs"
        docs_dir.mkdir(exist_ok=True)
        csv_path = docs_dir / "hust_conversion_summary.csv"
        pd.DataFrame(records).sort_values("cell_id").to_csv(csv_path, index=False)
        print(f"  요약 CSV: {csv_path}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MIT/HUST → 통일 포맷 변환 (병렬 지원)")
    parser.add_argument("--dataset",      default="all", choices=["mit", "hust", "all"])
    parser.add_argument("--output-root",  default=str(OUTPUT_ROOT))
    parser.add_argument("--cell",         default=None,
                        help="HUST 단일 셀 ID (예: 1-1). MIT는 항상 전체 변환.")
    parser.add_argument("--workers",      type=int, default=3,
                        help="병렬 프로세스 수 (기본: 3)")
    parser.add_argument("--no-cache",     action="store_true",
                        help="캐시 무시 — data_raw/ 가 있어도 원본 파일부터 재변환")
    args = parser.parse_args()

    out_root = Path(args.output_root)

    if args.dataset in ("mit", "all"):
        convert_mit(out_root, target_cell=args.cell, n_workers=args.workers,
                    no_cache=args.no_cache)

    if args.dataset in ("hust", "all"):
        convert_hust(out_root, target_cell=args.cell, n_workers=args.workers,
                     no_cache=args.no_cache)

    _plot_sample_cells(
        Path(__file__).resolve().parent / "outputs",
        out_root / "MIT",
        out_root / "HUST",
    )


def _plot_sample_cells(out_dir: Path, mit_dir: Path, hust_dir: Path) -> None:
    """변환 완료 후 대표 셀(MIT b1c0, HUST 1-1) 시각화 → 1_convert/outputs/"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
    except ImportError:
        print("  [SKIP] matplotlib 없음")
        return

    for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = _f; break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    samples = []
    for ds, d, cell in [("MIT", mit_dir, "b1c0"), ("HUST", hust_dir, "1-1")]:
        p = d / f"{cell}.pkl"
        if not p.exists():
            p_list = sorted(d.glob("*.pkl"))
            if p_list:
                p = p_list[0]
            else:
                continue
        samples.append((ds, p))

    if not samples:
        print("  [SKIP] 대표 셀 PKL 없음")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = cm.RdYlGn_r

    for ds, pkl_path in samples:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        meta = raw["meta"]
        df   = raw["cycles"]
        cell_id = meta.get("cell_id", pkl_path.stem)
        cycles  = sorted(df["cycle"].unique())
        n       = len(cycles)
        norm    = mcolors.Normalize(vmin=0, vmax=max(n - 1, 1))

        # 대표 사이클 5개 균등 선택
        n_show = min(5, n)
        rep    = [cycles[i] for i in np.linspace(0, n - 1, n_show, dtype=int)]

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
        fig.suptitle(f"[Step 1 변환 결과]  {ds}: {cell_id}  ({n} cycles)",
                     fontsize=11, fontweight="bold")

        for ax, col, ylabel, title in [
            (axes[0], "voltage_V",  "Voltage (V)",  "전압 프로파일"),
            (axes[1], "current_A",  "Current (A)",  "전류 프로파일"),
        ]:
            for cyc in rep:
                rank = cycles.index(cyc)
                cdf  = df[df["cycle"] == cyc].sort_values("time_s")
                t_rel = (cdf["time_s"] - cdf["time_s"].iloc[0]) / 3600
                ax.plot(t_rel, cdf[col],
                        color=cmap(norm(rank)), lw=0.8, alpha=0.85,
                        label=f"cycle {cyc}")
            ax.set_ylabel(ylabel); ax.set_xlabel("Time within cycle (h)")
            ax.set_title(title); ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=8, loc="best")

        ax = axes[2]
        dis = df[df["phase"] == "discharge"]
        cap = dis.groupby("cycle")["capacity_Ah"].first().reindex(cycles)
        ax.scatter(cycles, cap.values,
                   c=[cmap(norm(i)) for i in range(n)], s=6, zorder=2)
        ax.plot(cycles, cap.values, color="gray", lw=0.5, alpha=0.5, zorder=1)
        ax.set_ylabel("Capacity (Ah)"); ax.set_xlabel("Cycle")
        ax.set_title("사이클별 방전 용량 (열화 곡선)"); ax.grid(True, alpha=0.3)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
        cbar.set_label("Cycle rank", fontsize=9)
        cbar.set_ticks([0, n - 1])
        cbar.set_ticklabels([str(cycles[0]), str(cycles[-1])])

        out = out_dir / f"sample_{ds.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  대표 셀 플롯: {out}")


if __name__ == "__main__":
    main()
