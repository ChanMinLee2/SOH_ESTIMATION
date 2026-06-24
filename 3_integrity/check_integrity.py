"""
check_integrity.py

data_unified/MIT/, data_unified/HUST/ 전체 pkl 무결성 검사.

검사 항목:
  [셀 수준]
    1. 파일 수 (MIT=123, HUST=77)
    2. pkl 로드 가능 여부
    3. 스키마 컬럼 존재
    4. phase 값이 charge/discharge/rest 만 존재하는지
    5. 전류 방향 (MIT: 방전=양수 / HUST: 방전=음수)
    6. capacity_Ah 유효성 (방전 사이클)
    7. NaN 비율
    8. meta.n_cycles vs 실제 cycle 수 일치

  [사이클 수준]
    9.  전압 범위 이상 (V < 1.5V 또는 V > 4.5V)
    10. rest 행 비율 과다 (>80% — 오염/비정상 프로토콜)
    11. time_s 단조 증가 위반

출력:
  3_integrity/outputs/integrity_report.csv   — 셀 요약 통계
  3_integrity/outputs/integrity_issues.csv   — 셀·사이클별 이상 목록
"""

import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIT_DIR  = PROJECT_ROOT / "data_unified" / "MIT"
HUST_DIR = PROJECT_ROOT / "data_unified" / "HUST"
OUT_DIR  = Path(__file__).resolve().parent / "outputs"

EXPECTED_COLS = {"cycle", "time_s", "voltage_V", "current_A",
                 "temperature_C", "capacity_Ah", "phase"}
VALID_PHASES  = {"charge", "discharge", "rest"}
VALID_STAGES  = {"D0", "D1", "D2", "D3", ""}

MIT_DATASET_LABELS = {"MIT", "MIT_MAT"}

V_MIN, V_MAX     = 1.5, 4.5
CAP_MIN          = 0.1
REST_RATIO_WARN  = 0.80   # 사이클 내 rest 행 비율 > 이 값이면 경고


# ── 셀 단위 검사 ──────────────────────────────────────────────────────────────

def check_cell(pkl_path: Path) -> tuple:
    """Returns (record_dict, issues_list). 이상 없으면 issues_list=[]."""
    cell_id = pkl_path.stem
    issues  = []

    def _flag(severity, criterion, detail, cycle=None, dataset=""):
        issues.append({
            "severity":  severity,
            "dataset":   dataset,
            "cell_id":   cell_id,
            "cycle":     cycle,
            "criterion": criterion,
            "detail":    detail,
        })

    try:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
    except Exception as e:
        _flag("ERROR", "load_fail", str(e))
        return {}, issues

    meta    = raw.get("meta", {})
    df      = raw.get("cycles")
    dataset = meta.get("dataset", "")

    if df is None or not isinstance(df, pd.DataFrame):
        _flag("ERROR", "no_cycles_df", "cycles DataFrame 없음", dataset=dataset)
        return {}, issues

    # ── 1. 스키마 컬럼 ────────────────────────────────────────────────────────
    missing_cols = EXPECTED_COLS - set(df.columns)
    if missing_cols:
        _flag("ERROR", "missing_cols", f"컬럼 누락: {sorted(missing_cols)}", dataset=dataset)

    # ── 2. phase 값 ───────────────────────────────────────────────────────────
    bad_phases = set(df["phase"].unique()) - VALID_PHASES
    if bad_phases:
        _flag("ERROR", "invalid_phase", f"비정상 phase: {bad_phases}", dataset=dataset)

    phase_counts = df["phase"].value_counts().to_dict()
    n_charge    = phase_counts.get("charge", 0)
    n_discharge = phase_counts.get("discharge", 0)
    n_rest      = phase_counts.get("rest", 0)

    if n_discharge == 0:
        _flag("WARN", "no_discharge", "discharge 행 없음", dataset=dataset)
    if n_charge == 0:
        _flag("WARN", "no_charge", "charge 행 없음", dataset=dataset)

    # ── 3. 전류 방향 ──────────────────────────────────────────────────────────
    dis_df = df[df["phase"] == "discharge"]
    if dataset in MIT_DATASET_LABELS and len(dis_df) > 0:
        mean_i = dis_df["current_A"].mean()
        if mean_i > 0:
            _flag("ERROR", "current_direction",
                  f"MIT 방전 전류 평균 양수({mean_i:.3f}A) — phase 오류 의심", dataset=dataset)
    if dataset == "HUST" and len(dis_df) > 0:
        mean_i = dis_df["current_A"].mean()
        if mean_i > 0:
            _flag("ERROR", "current_direction",
                  f"HUST 방전 전류 평균 양수({mean_i:.3f}A) — phase 오류 의심", dataset=dataset)

    # ── 4. capacity_Ah 유효성 ─────────────────────────────────────────────────
    dis_cycles = dis_df["cycle"].unique() if len(dis_df) > 0 else []
    cap_by_cyc = (df[df["cycle"].isin(dis_cycles)]
                  .groupby("cycle")["capacity_Ah"].first())
    valid_caps = cap_by_cyc.dropna()

    if len(dis_cycles) > 0 and len(valid_caps) == 0:
        _flag("ERROR", "no_capacity", "방전 사이클에 capacity_Ah 없음", dataset=dataset)
    elif len(valid_caps) > 10:
        first_q = valid_caps.iloc[:len(valid_caps)//4].mean()
        last_q  = valid_caps.iloc[-len(valid_caps)//4:].mean()
        if last_q > first_q * 1.05:
            _flag("WARN", "capacity_increasing",
                  f"용량 증가 추세: 초기 {first_q:.4f} → 말기 {last_q:.4f} Ah", dataset=dataset)

    # ── 5. NaN 비율 ───────────────────────────────────────────────────────────
    nan_ratio = df.isnull().mean()
    for col, ratio in nan_ratio.items():
        if col == "temperature_C":
            continue
        if ratio > 0.5:
            _flag("WARN", "high_nan", f"{col} NaN {ratio:.1%}", dataset=dataset)

    # ── 6. meta.n_cycles 불일치 ───────────────────────────────────────────────
    meta_n = meta.get("n_cycles")
    real_n = df["cycle"].nunique()
    if meta_n is not None and abs(meta_n - real_n) > 0:
        _flag("WARN", "cycle_count_mismatch",
              f"meta.n_cycles={meta_n} ≠ 실제 {real_n}", dataset=dataset)

    # ── 7~9. 사이클 단위 검사 ─────────────────────────────────────────────────
    for cyc, grp in df.groupby("cycle"):
        n_cyc = len(grp)

        # 7. 전압 범위
        v_cyc = grp["voltage_V"].dropna()
        if len(v_cyc) > 0:
            if v_cyc.max() > V_MAX:
                bad_ph = sorted(grp.loc[grp["voltage_V"] > V_MAX, "phase"].unique())
                _flag("WARN", "voltage_high",
                      f"v_max={v_cyc.max():.3f}V > {V_MAX}V  phase={bad_ph}",
                      cycle=int(cyc), dataset=dataset)
            if v_cyc.min() < V_MIN:
                bad_ph = sorted(grp.loc[grp["voltage_V"] < V_MIN, "phase"].unique())
                _flag("WARN", "voltage_low",
                      f"v_min={v_cyc.min():.3f}V < {V_MIN}V  phase={bad_ph}",
                      cycle=int(cyc), dataset=dataset)

        # 8. rest 행 비율 과다
        n_rest_cyc = (grp["phase"] == "rest").sum()
        rest_ratio = n_rest_cyc / n_cyc if n_cyc > 0 else 0
        if rest_ratio > REST_RATIO_WARN:
            _flag("WARN", "rest_dominant",
                  f"rest 비율 {rest_ratio:.1%} ({n_rest_cyc}/{n_cyc}행)",
                  cycle=int(cyc), dataset=dataset)

        # 9. time_s 단조 증가 위반
        t = grp["time_s"].values
        if len(t) > 1 and np.any(np.diff(t) < 0):
            _flag("WARN", "time_nonmono", "time_s 단조 증가 위반",
                  cycle=int(cyc), dataset=dataset)

    v_all = df["voltage_V"].dropna()
    return {
        "cell_id":     cell_id,
        "dataset":     dataset,
        "total_rows":  len(df),
        "n_cycles":    real_n,
        "n_charge":    n_charge,
        "n_discharge": n_discharge,
        "n_rest":      n_rest,
        "v_min":       float(v_all.min()) if len(v_all) else np.nan,
        "v_max":       float(v_all.max()) if len(v_all) else np.nan,
        "cap_init":    float(valid_caps.iloc[0])  if len(valid_caps) > 0 else np.nan,
        "cap_final":   float(valid_caps.iloc[-1]) if len(valid_caps) > 0 else np.nan,
    }, issues


# ── top-level worker (Windows ProcessPoolExecutor 호환) ──────────────────────

def _check_worker(pkl_path_str: str) -> tuple:
    try:
        record, issues = check_cell(Path(pkl_path_str))
        return ("ok", (record, issues))
    except Exception:
        return ("err", Path(pkl_path_str).stem + ":\n" + traceback.format_exc())


# ── 디렉토리 단위 실행 ────────────────────────────────────────────────────────

def run_check(pkl_dir: Path, expected: int, label: str,
              n_workers: int = 1) -> tuple:
    """Returns (report_df, issues_list)."""
    files = sorted(pkl_dir.glob("*.pkl"))
    n = len(files)
    print(f"\n{'='*60}")
    print(f"  {label}  |  파일 수: {n}  (기대: {expected})")
    print(f"{'='*60}")

    all_issues = []
    if n != expected:
        all_issues.append({
            "severity": "WARN", "dataset": label, "cell_id": f"{label}_COUNT",
            "cycle": None, "criterion": "file_count_mismatch",
            "detail": f"파일 수 불일치: {n} ≠ {expected}",
        })

    records = []
    if n_workers <= 1:
        for p in tqdm(files, desc=label):
            r, iss = check_cell(p)
            if r: records.append(r)
            all_issues.extend(iss)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_check_worker, str(p)): p for p in files}
            with tqdm(total=len(files), desc=label) as pbar:
                for fut in as_completed(futures):
                    status, payload = fut.result()
                    if status == "ok":
                        r, iss = payload
                        if r: records.append(r)
                        all_issues.extend(iss)
                    else:
                        print(f"\n  [ERR] {payload}")
                    pbar.update(1)

    if not records:
        return pd.DataFrame(), all_issues

    df = pd.DataFrame(records)
    print(f"\n  총 데이터 행    : {df.total_rows.sum():,}")
    print(f"  셀별 사이클 수  : {df.n_cycles.min()} ~ {df.n_cycles.max()} (평균 {df.n_cycles.mean():.1f})")
    print(f"  전압 전체 범위  : {df.v_min.min():.3f} ~ {df.v_max.max():.3f} V")
    cap_v = df[df.cap_init.notna()]
    if len(cap_v) > 0:
        print(f"  초기 용량 범위  : {cap_v.cap_init.min():.4f} ~ {cap_v.cap_init.max():.4f} Ah")
        print(f"  최종 용량 범위  : {cap_v.cap_final.min():.4f} ~ {cap_v.cap_final.max():.4f} Ah")
    return df, all_issues


# ── 메인 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="data_unified PKL 무결성 검사")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="병렬 프로세스 수 (기본: 4)")
    args = parser.parse_args()

    df_mit,  issues_mit  = run_check(MIT_DIR,  123, "MIT",  args.workers)
    df_hust, issues_hust = run_check(HUST_DIR,  77, "HUST", args.workers)
    all_issues = issues_mit + issues_hust

    # ── 콘솔 요약 ─────────────────────────────────────────────────────────────
    issues_df = pd.DataFrame(all_issues) if all_issues else pd.DataFrame()
    errors    = issues_df[issues_df["severity"] == "ERROR"] if len(issues_df) else pd.DataFrame()
    warns     = issues_df[issues_df["severity"] == "WARN"]  if len(issues_df) else pd.DataFrame()

    print(f"\n{'='*60}")
    print("  이상 탐지 결과")
    print(f"{'='*60}")

    if len(errors) > 0:
        print(f"\n  [ERROR] {len(errors)}건")
        for _, row in errors.iterrows():
            cyc_str = f"  cycle={row.cycle}" if pd.notna(row.cycle) else ""
            print(f"    ✗ {row.cell_id}{cyc_str}  [{row.criterion}]  {row.detail}")
    else:
        print("\n  [ERROR] 없음 ✓")

    if len(warns) > 0:
        print(f"\n  [WARN] {len(warns)}건")
        for criterion, grp in warns.groupby("criterion"):
            n_cells  = grp["cell_id"].nunique()
            n_cycles = grp["cycle"].notna().sum()
            print(f"    △ {criterion}: 셀 {n_cells}개  사이클 {n_cycles}건")
    else:
        print("  [WARN] 없음 ✓")

    print(f"\n  총합: ERROR {len(errors)}건 / WARN {len(warns)}건")

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report_df = pd.concat([df_mit, df_hust], ignore_index=True)
    report_path = OUT_DIR / "integrity_report.csv"
    report_df.to_csv(report_path, index=False)

    if len(issues_df) > 0:
        issues_sorted = issues_df.sort_values(
            ["severity", "dataset", "cell_id", "cycle", "criterion"],
            na_position="first"
        )
        issues_path = OUT_DIR / "integrity_issues.csv"
        issues_sorted.to_csv(issues_path, index=False)
        print(f"\n  저장: {report_path}")
        print(f"  저장: {issues_path}")
    else:
        print(f"\n  저장: {report_path}")
        print("  이상 없음 — integrity_issues.csv 미생성")
