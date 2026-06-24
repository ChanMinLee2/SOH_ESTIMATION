"""
preprocess.py

MIT / HUST data_unified PKL 에 이상 사이클·행 제거를 적용.
1_convert/convert_unified.py 변환 이후, 4_hi_analysis/ HI 추출 전에 실행.

필터 적용 순서:
  [필터1] _remove_empty_cycles       — charge/discharge 5행 미만 사이클 제거
  [필터2] _fix_time_monotonicity      — 사이클 내 time_s 역방향 점프 단조 보정
  [필터3] _remove_zero_current_rest   — rest 행 중 current_A == 0.0 제거
  [필터4] _remove_dt_gap_cycles       — 방전/충전 구간 내 시간 단절 사이클 처리
              방전 단절 → 해당 사이클 전체 제거
              충전 단절 → 해당 사이클 충전 phase 행만 제거 (방전 HI 보존)
  [필터5] _remove_outlier_cycles      — Rolling Median 이상 사이클 제거 (window=11, sigma=2.0)
  [필터6] _remove_bad_vend_cycles     — 방전 종지전압 하한 필터 (vend_min=1.8V)
              HUST 비정상 종료 ~4,214건(2.88%) 제거
  [필터7] _remove_shape_outlier_cycles — V-q_frac 형상 편차 기반 이상 사이클 제거
              방전·충전 각각 rolling median 기준 곡선과의 RMSE/max|ΔV| 편차를
              MAD robust z-score로 평가 → z 또는 z_max > sigma(기본 5.0) 시 제거

  ※ DELETE_CELLS (완전 불량 셀 제외) 는 1_convert/convert_unified.py 에서 유지.
     셀 단위 제외이므로 이 스크립트에서는 다루지 않음.

입력:  data_unified/MIT/*.pkl, data_unified/HUST/*.pkl
출력:  data_postprocess/MIT/*.pkl, data_postprocess/HUST/*.pkl
       2_preprocess/outputs/cleaning_report.csv

사용:
  python preprocess.py                            # MIT + HUST 전체 (기본값)
  python preprocess.py --dataset mit
  python preprocess.py --dataset hust
  python preprocess.py --window 15 --sigma 3.0   # Rolling Median 파라미터 조정
  python preprocess.py --vend-min 1.85            # 종지전압 임계값 조정
  python preprocess.py --dis-gap-s 300            # 방전 단절 기준 (초)
  python preprocess.py --shape-sigma 6.0          # 형상 필터 엄격도 완화
  python preprocess.py --workers 4                # 병렬 처리 (기본: 4)
"""

import argparse
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT     = Path(__file__).resolve().parent.parent
_OUTPUTS_ROOT    = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR       = _OUTPUTS_ROOT / date.today().strftime("%m%d")
POSTPROCESS_ROOT = PROJECT_ROOT / "data_postprocess"


# ─────────────────────────────────────────────────────────────────────────────
# 필터1: 빈 사이클 제거  (from 1_convert/convert_unified.py)
# ─────────────────────────────────────────────────────────────────────────────

def _remove_empty_cycles(df: pd.DataFrame, min_active_rows: int = 5) -> tuple:
    """charge/discharge 행이 min_active_rows 미만인 사이클 제거.

    MIT cycle 1처럼 rest 2행만 존재하는 초기화 아티팩트를 제거.

    Returns:
        (cleaned_df, n_removed, removed_cycles_set)
    """
    active        = df[df["phase"].isin(["charge", "discharge"])]
    active_counts = active.groupby("cycle").size()
    bad_cycles    = set(active_counts[active_counts < min_active_rows].index)
    all_cycles    = set(df["cycle"].unique())
    bad_cycles   |= all_cycles - set(active_counts.index)   # active 행 전혀 없는 사이클

    if not bad_cycles:
        return df, 0, set()
    df_clean = df[~df["cycle"].isin(bad_cycles)].copy().reset_index(drop=True)
    return df_clean, len(bad_cycles), bad_cycles


# ─────────────────────────────────────────────────────────────────────────────
# 필터2: time_s 단조 보정  (from 1_convert/convert_unified.py)
# ─────────────────────────────────────────────────────────────────────────────

def _fix_time_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    """사이클 내 time_s 역방향 점프를 단조 증가로 보정.

    MIT 원본 타임스탬프의 분 단위 정밀도 문제로 발생하는 역방향 점프를
    np.maximum.accumulate 로 제거. 사이클 간 누적 오프셋은 적용하지 않음.
    """
    df = df.copy()
    for cyc in sorted(df["cycle"].unique()):
        mask = df["cycle"] == cyc
        t    = df.loc[mask, "time_s"].values.astype(float)
        t    = np.maximum.accumulate(t)
        df.loc[mask, "time_s"] = t
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 필터3: rest 0전류 행 제거  (from 1_convert/convert_unified.py)
# ─────────────────────────────────────────────────────────────────────────────

def _remove_zero_current_rest(df: pd.DataFrame) -> tuple:
    """rest 행 중 current_A == 0.0 인 행 제거.

    필터2(_fix_time_monotonicity) 이후에 호출해야 time_s 불연속이 생기지 않음.

    Returns:
        (cleaned_df, n_rows_removed)
    """
    mask = (df["phase"] == "rest") & (df["current_A"] == 0.0)
    n    = int(mask.sum())
    if n == 0:
        return df, 0
    return df[~mask].copy().reset_index(drop=True), n


# ─────────────────────────────────────────────────────────────────────────────
# 필터4: 시간 단절 사이클 처리  (from 4_hi_analysis/hi_correlation.py 인라인)
# ─────────────────────────────────────────────────────────────────────────────

def _remove_dt_gap_cycles(df: pd.DataFrame,
                           dis_gap_s:          float = 600.0,
                           dis_gap_factor:     float = 50.0,
                           chg_gap_s:          float = 600.0,
                           chg_gap_factor:     float = 50.0,
                           chg_seg_gap_s:      float = 120.0,
                           chg_seg_gap_factor: float = 30.0) -> tuple:
    """방전/충전 구간 내 시간 단절이 있는 사이클을 처리.

    판정:
      dt     = clip(diff(t, prepend=t[0]), 0, None)
      dt_med = median(dt[dt > 0])
      단절   = dt.max() > max(gap_s, dt_med × gap_factor)

    방전 단절 (>dis_gap_s/dis_gap_factor)
      → 해당 사이클 전체 제거.

    충전 완전 중단 (>chg_gap_s/chg_gap_factor, 기본 600s/50×)
      → 충전 phase 행만 제거, 방전 HI 보존.

    충전 CC 프로토콜 전환 갭 (>chg_seg_gap_s/chg_seg_gap_factor, 기본 120s/30×)
      → 행 유지, `chg_gap_seg = True` 플래그 기록.
      → hi_correlation 에서 이 컬럼을 읽어 세그먼트 HI만 NaN 처리.
      → 전역 HI(에너지, 전압평균 등)는 추세 일관성이 있으므로 계산.

    Returns:
        (df_clean,
         n_dis_removed,    dis_removed_cycles,
         n_chg_rows,       chg_all_cycles,
         n_chg_seg_marked, chg_seg_cycles)
    """
    dis_bad:     set = set()
    chg_bad_all: set = set()   # 완전 중단 → 행 삭제
    chg_bad_seg: set = set()   # CC 전환 갭 → 플래그만

    for cyc, grp in df.groupby("cycle"):
        # ── 방전 단절 검사 ───────────────────────────────────────────────
        dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
        if len(dis) > 5:
            t      = dis["time_s"].values.astype(float)
            dt     = np.clip(np.diff(t, prepend=t[0]), 0, None)
            dt_pos = dt[dt > 0]
            if len(dt_pos) > 0:
                dt_med = float(np.median(dt_pos))
                if float(dt.max()) > max(dis_gap_s, dt_med * dis_gap_factor):
                    dis_bad.add(cyc)
                    continue          # 방전 불량 → 충전 검사 불필요

        # ── 충전 단절 검사 (두 단계) ─────────────────────────────────────
        chg = grp[grp["phase"] == "charge"].sort_values("time_s")
        if len(chg) > 5:
            tc      = chg["time_s"].values.astype(float)
            dtc     = np.clip(np.diff(tc, prepend=tc[0]), 0, None)
            dtc_pos = dtc[dtc > 0]
            if len(dtc_pos) > 0:
                dtc_med = float(np.median(dtc_pos))
                dtc_max = float(dtc.max())
                if dtc_max > max(chg_gap_s, dtc_med * chg_gap_factor):
                    chg_bad_all.add(cyc)          # 완전 중단
                elif dtc_max > max(chg_seg_gap_s, dtc_med * chg_seg_gap_factor):
                    chg_bad_seg.add(cyc)          # CC 전환 갭

    # 방전 단절: 전체 사이클 제거
    df_clean = df[~df["cycle"].isin(dis_bad)].copy()

    # 충전 완전 중단: 충전 phase 행 제거
    chg_mask   = df_clean["cycle"].isin(chg_bad_all) & (df_clean["phase"] == "charge")
    n_chg_rows = int(chg_mask.sum())
    if n_chg_rows > 0:
        df_clean = df_clean[~chg_mask].copy()

    # CC 전환 갭: chg_gap_seg 플래그 컬럼 기록 (hi_correlation 에서 읽음)
    df_clean["chg_gap_seg"] = False
    if chg_bad_seg:
        df_clean.loc[df_clean["cycle"].isin(chg_bad_seg), "chg_gap_seg"] = True

    df_clean = df_clean.reset_index(drop=True)
    return (df_clean,
            len(dis_bad), dis_bad,
            n_chg_rows, chg_bad_all,
            len(chg_bad_seg), chg_bad_seg)


# ─────────────────────────────────────────────────────────────────────────────
# 필터5: Rolling Median 이상 사이클 제거
# ─────────────────────────────────────────────────────────────────────────────

def _remove_outlier_cycles(df: pd.DataFrame,
                            window: int   = 11,
                            sigma:  float = 2.5,
                            min_std: float = 0.01,
                            window2: int  = 31,
                            sigma2: float = 2.0) -> tuple:
    """방전 용량 시계열에서 Rolling Median 기반 이상 사이클 제거 (2-pass).

    Pass 1 (window=11, sigma=2.5): 고립 이상치 제거.
    Pass 2 (window=31, sigma=2.0): Pass 1 이후 클러스터 이상치 제거.
      - MIT b1c0처럼 불완전 충전 이벤트가 연속 3-5사이클 묶여 나타나는 경우,
        window=11 rolling median이 이상치 쪽으로 당겨져 개별 이상치가 통과됨.
        더 넓은 window로 재검사해 잔여 이상치를 제거.

    Returns:
        (cleaned_df, n_removed, removed_cycles_set)
    """
    dis   = df[df["phase"] == "discharge"]
    cap_s = dis.groupby("cycle")["capacity_Ah"].first().dropna().sort_index()

    if len(cap_s) < window:
        return df, 0, set()

    # Pass 1
    roll  = cap_s.rolling(window=window, center=True, min_periods=3)
    r_med = roll.median()
    r_std = roll.std().fillna(cap_s.std()).clip(lower=min_std)
    outlier_cycles = set(cap_s[(cap_s - r_med).abs() > sigma * r_std].index)

    # Pass 2: Pass 1 제거 후 남은 데이터에서 넓은 윈도우로 재검사
    if len(cap_s) - len(outlier_cycles) >= window2:
        cap2   = cap_s.drop(index=outlier_cycles)
        roll2  = cap2.rolling(window=window2, center=True, min_periods=5)
        r_med2 = roll2.median().reindex(cap_s.index).ffill().bfill()
        r_std2 = roll2.std().reindex(cap_s.index).ffill().bfill().clip(lower=min_std)
        outlier_cycles |= set(
            cap_s[(cap_s - r_med2).abs() > sigma2 * r_std2].index
        ) - outlier_cycles

    if not outlier_cycles:
        return df, 0, set()

    df_clean = df[~df["cycle"].isin(outlier_cycles)].copy().reset_index(drop=True)
    return df_clean, len(outlier_cycles), outlier_cycles


# ─────────────────────────────────────────────────────────────────────────────
# 필터6: 방전 종지전압 하한 필터
# ─────────────────────────────────────────────────────────────────────────────

def _remove_bad_vend_cycles(df: pd.DataFrame, vend_min: float) -> tuple:
    """방전 종지전압이 vend_min 미만인 비정상 종료 사이클 제거.

    HUST 진단 결과 v_end < 1.9V 사이클이 2.88% 존재함.
    MIT 방전 컷오프는 2.0V이므로 기본값(1.8V)에서 MIT에는 영향 없음.

    Returns:
        (cleaned_df, n_removed, removed_cycles_set)
    """
    dis        = df[df["phase"] == "discharge"]
    last_v     = dis.groupby("cycle")["voltage_V"].last()
    bad_cycles = set(last_v[last_v < vend_min].index)
    if not bad_cycles:
        return df, 0, set()
    df_clean = df[~df["cycle"].isin(bad_cycles)].copy().reset_index(drop=True)
    return df_clean, len(bad_cycles), bad_cycles


# ─────────────────────────────────────────────────────────────────────────────
# 필터7: V-q_frac 형상 이상 사이클 제거  (diagnose_shape_outliers.py 로직 통합)
# ─────────────────────────────────────────────────────────────────────────────

# 자동 필터(robust z 임계값)를 미달하지만 진단으로 확인된 known 이상치.
# 키: (dataset_upper, cell_id) / 값: 제거할 사이클 번호 집합
# dataset은 "MIT_MAT" 등 원본 값에서 "_MAT" 제거 후 대문자로 정규화해 비교.
KNOWN_SHAPE_ANOMALIES: dict = {
    ("MIT", "b1c23"): {1003},   # charge: 3.6V 장구간 유지 — 말기 밴드 중 이 사이클만 제거
    ("MIT", "b1c36"): {73},     # discharge: 초기 사이클 위 국소 돌출 (max_dev 0.23, 임계 미달)
}

def _qfrac_interp(phase_df: pd.DataFrame, grid: np.ndarray):
    """phase_df → 공통 q_frac 격자에 보간된 V 배열. 데이터 부족 시 None."""
    if len(phase_df) < 10:
        return None
    t  = phase_df["time_s"].values.astype(float)
    v  = phase_df["voltage_V"].values.astype(float)
    i  = np.abs(phase_df["current_A"].values.astype(float))
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    q_cum = np.cumsum(i * dt) / 3600.0
    if float(q_cum[-1]) < 0.05:
        return None
    q_frac = q_cum / float(q_cum[-1])
    order  = np.argsort(q_frac)
    return np.interp(grid, q_frac[order], v[order])


def _robust_z(x: np.ndarray) -> np.ndarray:
    """MAD 기반 robust z-score. MAD≈0 시 std 대체."""
    med   = np.median(x)
    mad   = np.median(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-9 else (float(np.std(x)) or 1.0)
    return (x - med) / scale


def _remove_shape_outlier_cycles(df: pd.DataFrame,
                                  sigma:   float = 5.0,
                                  window:  int   = 11,
                                  grid_n:  int   = 100,
                                  cell_id: str   = "",
                                  dataset: str   = "") -> tuple:
    """V-q_frac 형상 편차 기반 이상 사이클 제거.

    방전·충전 각각 다음 두 지표로 이상 판정 (둘 중 하나라도 초과하면 제거):
      z     : 사이클별 RMSE(V, V_ref)의 셀 내부 MAD robust z → 전체 형상 붕괴 감지
      z_max : 사이클별 max|V - V_ref|의 셀 내부 MAD robust z → 국소 돌출/글리치 감지

    V_ref = q_frac 격자에 보간 후 rolling(window, center=True) median 곡선.
    열화에 따른 정상적 개형 변화는 기준이 따라가므로 통과,
    갑자기 튀는 사이클(충전 인터럽트, 측정 글리치 등)만 이상 판정.

    어느 phase(방전 또는 충전)에서 이상 판정되더라도 해당 사이클 전체 제거.
    자동 임계값을 미달하지만 진단으로 확인된 사이클은 KNOWN_SHAPE_ANOMALIES에 등록.

    Returns:
        (cleaned_df, n_removed, removed_cycles_set)
    """
    grid           = np.linspace(0.0, 1.0, grid_n)
    outlier_cycles: set = set()

    for phase in ("discharge", "charge"):
        rows: list = []
        mat:  list = []
        for cyc, cyc_df in df[df["phase"] == phase].groupby("cycle"):
            v_interp = _qfrac_interp(cyc_df, grid)
            if v_interp is None:
                continue
            rows.append(cyc)
            mat.append(v_interp)

        if len(rows) < max(window, 5):
            continue

        M    = pd.DataFrame(mat, index=rows)
        ref  = M.rolling(window=window, center=True, min_periods=3).median()
        diff = M - ref

        dev     = np.sqrt((diff ** 2).mean(axis=1)).values
        max_dev = diff.abs().max(axis=1).values
        z       = _robust_z(dev)
        z_max   = _robust_z(max_dev)

        bad = set(np.array(rows)[(z > sigma) | (z_max > sigma)])
        outlier_cycles |= bad

    # KNOWN_SHAPE_ANOMALIES: 자동 임계값 미달 known 이상치 추가 제거
    ds_key = dataset.upper().replace("_MAT", "")
    known  = KNOWN_SHAPE_ANOMALIES.get((ds_key, cell_id), set())
    existing_cycles = set(df["cycle"].unique())
    outlier_cycles |= (known & existing_cycles)

    if not outlier_cycles:
        return df, 0, set()

    df_clean = df[~df["cycle"].isin(outlier_cycles)].copy().reset_index(drop=True)
    return df_clean, len(outlier_cycles), outlier_cycles


# ─────────────────────────────────────────────────────────────────────────────
# 워커
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_worker(args) -> tuple:
    """top-level 함수 — Windows ProcessPoolExecutor 필수."""
    (src_path_str, dst_path_str,
     window, sigma, min_std, vend_min,
     dis_gap_s, dis_gap_factor,
     chg_gap_s, chg_gap_factor,
     chg_seg_gap_s, chg_seg_gap_factor,
     shape_sigma, shape_window, shape_grid) = args
    src = Path(src_path_str)
    dst = Path(dst_path_str)
    try:
        with open(src, "rb") as f:
            raw = pickle.load(f)

        meta     = raw["meta"]
        df       = raw["cycles"]
        n_before = df["cycle"].nunique()

        # ── 필터 적용 (순서 고정) ─────────────────────────────────────────
        df, n_empty,   empty_cycles                                        = _remove_empty_cycles(df)
        df                                                                  = _fix_time_monotonicity(df)
        df, n_rest                                                          = _remove_zero_current_rest(df)
        (df,
         n_dt_dis, dt_dis_cycles,
         n_chg_rows, dt_chg_cycles,
         n_chg_seg, dt_chg_seg_cycles)                                     = \
            _remove_dt_gap_cycles(df,
                                  dis_gap_s, dis_gap_factor,
                                  chg_gap_s, chg_gap_factor,
                                  chg_seg_gap_s, chg_seg_gap_factor)
        df, n_rolling, rolling_cycles                                      = _remove_outlier_cycles(df, window, sigma, min_std)
        df, n_vend,    vend_cycles                                         = _remove_bad_vend_cycles(df, vend_min)
        _cell_id = meta.get("cell_id", src.stem)
        _dataset = meta.get("dataset", "")
        df, n_shape,   shape_cycles                                        = _remove_shape_outlier_cycles(
            df, shape_sigma, shape_window, shape_grid,
            cell_id=_cell_id, dataset=_dataset)

        n_cycles_removed = n_empty + n_dt_dis + n_rolling + n_vend + n_shape
        n_after          = df["cycle"].nunique()

        any_change = n_cycles_removed > 0 or n_rest > 0 or n_chg_rows > 0 or n_chg_seg > 0
        if any_change:
            meta["n_cycles"]          = n_after
            meta["n_outliers_removed"] = meta.get("n_outliers_removed", 0) + n_cycles_removed

        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            pickle.dump({"meta": meta, "cycles": df}, f)
        if any_change:
            df.to_csv(dst.with_suffix(".csv"), index=False)

        return ("ok", {
            "cell_id":                  meta.get("cell_id", src.stem),
            "dataset":                  meta.get("dataset", ""),
            "n_cycles_before":          n_before,
            "n_cycles_after":           n_after,
            "n_removed_empty":          n_empty,
            "n_rows_removed_rest":      n_rest,
            "n_removed_dt_dis":         n_dt_dis,
            "n_chg_rows_cleaned":       n_chg_rows,
            "n_chg_seg_flagged":        n_chg_seg,
            "n_removed_rolling":        n_rolling,
            "n_removed_vend":           n_vend,
            "n_removed_shape":          n_shape,
            "n_removed":                n_cycles_removed,
            "removed_empty_cycles":     sorted(empty_cycles),
            "removed_dt_dis_cycles":    sorted(dt_dis_cycles),
            "dt_chg_cycles_cleaned":    sorted(dt_chg_cycles),
            "dt_chg_seg_cycles":        sorted(dt_chg_seg_cycles),
            "removed_rolling_cycles":   sorted(rolling_cycles),
            "removed_vend_cycles":      sorted(vend_cycles),
            "removed_shape_cycles":     sorted(shape_cycles),
        })
    except Exception:
        return ("err", f"{src.stem}:\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# 디렉토리 처리
# ─────────────────────────────────────────────────────────────────────────────

def process_dir(src_dir: Path, dst_dir: Path,
                window: int, sigma: float, min_std: float, vend_min: float,
                dis_gap_s: float, dis_gap_factor: float,
                chg_gap_s: float, chg_gap_factor: float,
                chg_seg_gap_s: float, chg_seg_gap_factor: float,
                shape_sigma: float, shape_window: int, shape_grid: int,
                n_workers: int = 1) -> list:
    """src_dir PKL → 이상 사이클 제거 → dst_dir 저장 → records 반환."""
    pkls = sorted(p for p in src_dir.glob("*.pkl")
                  if p.stem not in ("README",))
    if not pkls:
        print(f"  PKL 파일 없음: {src_dir}")
        return []

    dst_dir.mkdir(parents=True, exist_ok=True)
    args_list = [
        (str(p), str(dst_dir / p.name),
         window, sigma, min_std, vend_min,
         dis_gap_s, dis_gap_factor,
         chg_gap_s, chg_gap_factor,
         chg_seg_gap_s, chg_seg_gap_factor,
         shape_sigma, shape_window, shape_grid)
        for p in pkls
    ]
    records = []

    if n_workers <= 1:
        for a in tqdm(args_list, desc=src_dir.name):
            status, payload = _preprocess_worker(a)
            if status == "ok":
                records.append(payload)
            else:
                print(f"\n  [ERR] {payload}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_preprocess_worker, a): a for a in args_list}
            with tqdm(total=len(args_list), desc=src_dir.name) as pbar:
                for fut in as_completed(futures):
                    status, payload = fut.result()
                    if status == "ok":
                        records.append(payload)
                    else:
                        print(f"\n  [ERR] {payload}")
                    pbar.update(1)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import os
    parser = argparse.ArgumentParser(
        description="unified PKL → 7단계 이상 사이클·행 제거 → postprocess 저장"
    )
    parser.add_argument("--dataset",  default="all", choices=["mit", "hust", "all"],
                        help="처리할 데이터셋 (기본: all)")
    # 필터4 파라미터
    parser.add_argument("--dis-gap-s",      type=float, default=600.0,
                        help="[필터4] 방전 단절 절대 기준 초 (기본: 600)")
    parser.add_argument("--dis-gap-factor", type=float, default=50.0,
                        help="[필터4] 방전 단절 배율 기준 median×N (기본: 50)")
    parser.add_argument("--chg-gap-s",          type=float, default=600.0,
                        help="[필터4] 충전 완전 중단 절대 기준 초 (기본: 600) — 행 삭제")
    parser.add_argument("--chg-gap-factor",     type=float, default=50.0,
                        help="[필터4] 충전 완전 중단 배율 기준 median×N (기본: 50) — 행 삭제")
    parser.add_argument("--chg-seg-gap-s",      type=float, default=120.0,
                        help="[필터4] 충전 CC 전환 갭 절대 기준 초 (기본: 120) — chg_gap_seg 플래그")
    parser.add_argument("--chg-seg-gap-factor", type=float, default=30.0,
                        help="[필터4] 충전 CC 전환 갭 배율 기준 median×N (기본: 30) — chg_gap_seg 플래그")
    # 필터5 파라미터
    parser.add_argument("--window",   type=int,   default=11,
                        help="[필터5] Rolling Median 윈도우 (기본: 11)")
    parser.add_argument("--sigma",    type=float, default=2.0,
                        help="[필터5] 이상치 σ 임계값 (기본: 2.5)")
    parser.add_argument("--min-std",  type=float, default=0.01,
                        help="[필터5] std 플로어 Ah (기본: 0.01)")
    # 필터6 파라미터
    parser.add_argument("--vend-min", type=float, default=1.8,
                        help="[필터6] 방전 종지전압 하한 V (기본: 1.8)")
    # 필터7 파라미터
    parser.add_argument("--shape-sigma",  type=float, default=5.0,
                        help="[필터7] 형상 편차 robust z 임계값 (기본: 5.0). 낮을수록 더 많이 제거")
    parser.add_argument("--shape-window", type=int,   default=11,
                        help="[필터7] 기준 곡선 rolling median 윈도우 (기본: 11)")
    parser.add_argument("--shape-grid",   type=int,   default=100,
                        help="[필터7] q_frac 보간 격자 점 수 (기본: 100)")
    # 공통
    parser.add_argument("--workers",  type=int, default=min(4, os.cpu_count() or 1),
                        help="병렬 프로세스 수 (기본: 4)")
    args = parser.parse_args()

    w, s, m, vm     = args.window, args.sigma, args.min_std, args.vend_min
    dgs, dgf         = args.dis_gap_s, args.dis_gap_factor
    cgs, cgf         = args.chg_gap_s, args.chg_gap_factor
    csgs, csgf       = args.chg_seg_gap_s, args.chg_seg_gap_factor
    ss, sw, sg       = args.shape_sigma, args.shape_window, args.shape_grid

    print("=== 이상 사이클·행 제거 (7단계) ===")
    print(f"  [필터1] 빈 사이클     : min_active_rows=5")
    print(f"  [필터2] time 단조 보정: (자동)")
    print(f"  [필터3] rest 0전류    : current_A==0.0")
    print(f"  [필터4] 시간 단절     : dis_gap={dgs}s×{dgf}")
    print(f"            충전 완전중단: {cgs}s×{cgf}  (행 삭제)")
    print(f"            충전 CC전환갭: {csgs}s×{csgf}  (chg_gap_seg 플래그)")
    print(f"  [필터5] Rolling Median: window={w}, sigma={s}, min_std={m}")
    print(f"  [필터6] 종지전압 하한 : vend_min={vm}V")
    print(f"  [필터7] 형상 편차     : shape_sigma={ss}, window={sw}, grid={sg}")

    all_records = []
    SAMPLE_IDS  = {"MIT": "b1c0", "HUST": "1-1"}
    before_caps: dict = {}

    if args.dataset in ("mit", "all"):
        mit_src = PROJECT_ROOT / "data_unified" / "MIT"
        mit_dst = POSTPROCESS_ROOT / "MIT"
        if not mit_src.exists():
            print(f"\n[SKIP] MIT 폴더 없음: {mit_src}")
        else:
            before_caps["MIT"] = (SAMPLE_IDS["MIT"],
                                  *_cap_series(mit_src / f"{SAMPLE_IDS['MIT']}.pkl"))
            print(f"\n[MIT] {mit_src} → {mit_dst}")
            all_records.extend(
                process_dir(mit_src, mit_dst, w, s, m, vm,
                            dgs, dgf, cgs, cgf, csgs, csgf,
                            ss, sw, sg, args.workers))

    if args.dataset in ("hust", "all"):
        hust_src = PROJECT_ROOT / "data_unified" / "HUST"
        hust_dst = POSTPROCESS_ROOT / "HUST"
        if not hust_src.exists():
            print(f"\n[SKIP] HUST 폴더 없음: {hust_src}")
        else:
            before_caps["HUST"] = (SAMPLE_IDS["HUST"],
                                   *_cap_series(hust_src / f"{SAMPLE_IDS['HUST']}.pkl"))
            print(f"\n[HUST] {hust_src} → {hust_dst}")
            all_records.extend(
                process_dir(hust_src, hust_dst, w, s, m, vm,
                            dgs, dgf, cgs, cgf, csgs, csgf,
                            ss, sw, sg, args.workers))

    if not all_records:
        print("\n처리할 파일이 없습니다.")
        return

    df_report = pd.DataFrame(all_records)

    total_empty     = int(df_report["n_removed_empty"].sum())
    total_rest_rows = int(df_report["n_rows_removed_rest"].sum())
    total_dt_dis    = int(df_report["n_removed_dt_dis"].sum())
    total_chg_clean = int(df_report["n_chg_rows_cleaned"].sum())
    total_chg_seg   = int(df_report["n_chg_seg_flagged"].sum())
    total_rolling   = int(df_report["n_removed_rolling"].sum())
    total_vend      = int(df_report["n_removed_vend"].sum())
    total_shape     = int(df_report["n_removed_shape"].sum())
    total_removed   = int(df_report["n_removed"].sum())
    n_affected      = int((df_report["n_removed"] > 0).sum())

    print(f"\n=== 완료 ===")
    print(f"  처리 셀 수               : {len(df_report)}")
    print(f"  변경 발생 셀 수          : {n_affected}")
    print(f"  [필터1] 빈 사이클 제거   : {total_empty} 사이클")
    print(f"  [필터2] time 단조 보정   : (행 수 유지)")
    print(f"  [필터3] rest 0전류 제거  : {total_rest_rows} 행")
    print(f"  [필터4] 방전 단절 제거   : {total_dt_dis} 사이클")
    print(f"  [필터4] 충전 완전중단    : {total_chg_clean} 행 삭제 (사이클 유지)")
    print(f"  [필터4] 충전 CC전환갭    : {total_chg_seg} 사이클 → chg_gap_seg 플래그")
    print(f"  [필터5] Rolling 제거     : {total_rolling} 사이클")
    print(f"  [필터6] v_end 제거       : {total_vend} 사이클")
    print(f"  [필터7] 형상 편차 제거   : {total_shape} 사이클")
    print(f"  합계 사이클 제거         : {total_removed}")

    if total_removed > 0:
        print("\n  [제거 발생 셀 상세]")
        for _, row in df_report[df_report["n_removed"] > 0].iterrows():
            parts = []
            if row.n_removed_empty:
                parts.append(f"empty={row.n_removed_empty} {list(row.removed_empty_cycles)}")
            if row.n_removed_dt_dis:
                parts.append(f"dt_dis={row.n_removed_dt_dis} {list(row.removed_dt_dis_cycles)}")
            if row.n_removed_rolling:
                parts.append(f"rolling={row.n_removed_rolling} {list(row.removed_rolling_cycles)}")
            if row.n_removed_vend:
                parts.append(f"vend={row.n_removed_vend} {list(row.removed_vend_cycles)}")
            if row.n_removed_shape:
                parts.append(f"shape={row.n_removed_shape} {list(row.removed_shape_cycles)}")
            print(f"    {row.cell_id} ({row.dataset}): "
                  f"{row.n_cycles_before} → {row.n_cycles_after}  "
                  f"(-{row.n_removed}  [{',  '.join(parts)}])")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = _OUTPUTS_ROOT / "cleaning_report.csv"
    df_report.to_csv(report_path, index=False)
    print(f"\n  리포트: {report_path}")

    # 대표 셀 before/after 플롯
    after_caps: dict = {}
    for ds, (cell_id, *_) in before_caps.items():
        after_caps[ds] = _cap_series(POSTPROCESS_ROOT / ds / f"{cell_id}.pkl")
    _plot_preprocess_samples(OUTPUT_DIR, before_caps, after_caps, df_report)


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _cap_series(pkl_path: Path):
    """PKL에서 방전 용량 시리즈 (cycles, caps) 반환."""
    if not pkl_path.exists():
        return np.array([]), np.array([])
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    dis = raw["cycles"][raw["cycles"]["phase"] == "discharge"]
    cap = dis.groupby("cycle")["capacity_Ah"].first().dropna().sort_index()
    return cap.index.to_numpy(), cap.values


def _plot_preprocess_samples(out_dir: Path,
                              before: dict, after: dict,
                              df_report: pd.DataFrame) -> None:
    """Step 2 전처리 before/after 대표 셀 시각화 → outputs/"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib 없음")
        return

    for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = _f; break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    items = [(ds, v) for ds, v in before.items() if ds in after and len(v[0]) > 0]
    if not items:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for ds, (cell_id, cyc_b, cap_b) in items:
        cyc_a, cap_a = after[ds]
        removed = sorted(set(cyc_b) - set(cyc_a))
        rm_mask = np.isin(cyc_b, removed)

        row   = df_report[df_report["cell_id"] == cell_id]
        stats = {}
        for col in ["n_removed_empty", "n_removed_dt_dis",
                    "n_removed_rolling", "n_removed_vend", "n_removed_shape"]:
            stats[col] = int(row[col].iloc[0]) if len(row) else 0

        fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        fig.suptitle(f"[Step 2 전처리 결과]  {ds}: {cell_id}",
                     fontsize=11, fontweight="bold")

        ax.plot(cyc_b, cap_b, color="lightgray", lw=0.8, zorder=1, label="before")
        ax.scatter(cyc_b[~rm_mask], cap_b[~rm_mask],
                   color="steelblue", s=5, zorder=2, label="유지 사이클")
        if rm_mask.any():
            ax.scatter(cyc_b[rm_mask], cap_b[rm_mask],
                       color="red", s=18, marker="x", zorder=3,
                       label=f"제거 사이클 ({len(removed)}건)")
        ax.plot(cyc_a, cap_a, color="steelblue", lw=1.0, alpha=0.7, zorder=2)

        ann = (f"제거: {len(removed)}건  "
               f"(empty={stats['n_removed_empty']}, "
               f"dt_dis={stats['n_removed_dt_dis']}, "
               f"rolling={stats['n_removed_rolling']}, "
               f"v_end={stats['n_removed_vend']}, "
               f"shape={stats['n_removed_shape']})")
        ax.annotate(ann, xy=(0.02, 0.05), xycoords="axes fraction", fontsize=9,
                    color="red", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_xlabel("Cycle"); ax.set_ylabel("Capacity (Ah)")
        ax.set_title("사이클별 방전 용량 — before/after 이상 사이클 제거")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        out = out_dir / f"sample_{ds.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  대표 셀 플롯: {out}")


if __name__ == "__main__":
    main()
