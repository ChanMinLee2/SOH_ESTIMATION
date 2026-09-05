"""
1_convert/convert_tju.py — Tongji(TJU) NCM 데이터셋 전용 변환기.

MIT/HUST(convert_unified.py)와 완전히 분리된 파일 — 공용 유틸(assign_phase, save_cell,
_fix_time_monotonicity, _remove_zero_current_rest, _remove_outlier_cycles, _run,
PROJECT_ROOT, RAW_OUTPUT_ROOT)만 convert_unified에서 가져다 쓰고, TJU 고유 파싱 로직은
전부 이 파일 안에 둔다.

출처: Zhu, J. et al. "Data-driven capacity estimation of commercial lithium-ion batteries
from voltage relaxation." Nature Communications (2022). DOI: 10.5281/zenodo.6405084
(Dataset_2_NCM_battery.zip — 130셀 중 순수 NCM 55셀만).

원본 파일 형식: 셀 하나 = CSV 파일 하나(그 셀의 전체 수명, 날짜별로 안 쪼개짐).
컬럼: time/s, control/V/mA, Ecell/V, <I>/mA, Q discharge/mA.h, Q charge/mA.h,
      control/V, control/mA, cycle number
(2026-09-05 실측 확인, ncm/CY25-05_1-#1.csv 샘플 기준)

- `cycle number`가 파일 전체에 걸쳐 신뢰 가능(1부터 끝까지 연속 증가, 리셋 없음)
  → CALCE CS2 raw txt와 달리 사이클 경계 재구성이 불필요하다.
- `<I>/mA` 부호가 충전(+)/방전(-)으로 정확히 갈린다 → 기존 assign_phase() 그대로 재사용.
- `time/s`는 파일(=셀 전체 수명) 기준 누적 시간이다 — MIT/HUST의 "사이클별 상대 시간"
  관례와 다르므로, 사이클마다 첫 값을 빼서 상대 시간으로 변환한다(_make_time_relative).
- `capacity_Ah`는 그 사이클의 Discharge_Capacity(=Q discharge/mA.h) 최댓값을 사이클 전체
  행에 broadcast — HUST/MIT과 동일 관례(그 사이클의 "실측 방전 용량" 단일값).

셀 이름: CY{온도}-{충전C}_{방전C}-#{번호} (예: CY25-05_1-#1 = 25°C, 0.5C 충전, 1C 방전, 1번 셀).

사용:
  python convert_tju.py --workers 4
  python convert_tju.py --cell "CY25-05_1-#1"
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from convert_unified import (  # noqa: E402
    PROJECT_ROOT, RAW_OUTPUT_ROOT, OUTPUT_ROOT,
    assign_phase, save_cell, _make_time_relative, _fix_time_monotonicity,
    _remove_zero_current_rest, _remove_outlier_cycles, _run,
)
import sys as _sys  # noqa: E402
if str(PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT))
from data_directories import EXTERNAL_DATA_ROOT  # noqa: E402

# 2026-09-05: 디스크 공간 때문에 D 드라이브로 이전(data_directories.py의 기존
# _4_data_hi 관례와 동일 원칙 — 이 저장소가 그 유일한 진입점이므로 여기서도 재사용).
TJU_DATA_DIR = EXTERNAL_DATA_ROOT / "_0_data_raw" / "TJU" / "Dataset_2_NCM_battery"


def convert_tju_cell(csv_path: Path, out_dir: Path, raw_out_dir: Path) -> dict:
    """변환 후 stats dict 반환. 스킵 시 빈 dict."""
    cell_id = csv_path.stem  # 예: "CY25-05_1-#1"

    raw = pd.read_csv(csv_path)
    if raw.empty:
        return {}

    cycle = raw["cycle number"].round().astype(int).values
    time_s = raw["time/s"].values.astype(float)
    voltage_V = raw["Ecell/V"].values.astype(float)
    current_A = raw["<I>/mA"].values.astype(float) / 1000.0
    q_dis_mAh = raw["Q discharge/mA.h"].values.astype(float)

    phase = assign_phase(current_A)

    df = pd.DataFrame({
        "cycle":     cycle,
        "time_s":    time_s,
        "voltage_V": voltage_V,
        "current_A": current_A,
        "phase":     phase,
        "_q_dis_mAh": q_dis_mAh,
    })
    # 사이클별 실측 방전 용량 = 그 사이클 내 Q discharge 최댓값 (mAh -> Ah), 전체 행에 broadcast
    cap_per_cycle = df.groupby("cycle")["_q_dis_mAh"].transform("max") / 1000.0
    df["capacity_Ah"] = cap_per_cycle
    df = df.drop(columns=["_q_dis_mAh"])

    # ── raw 저장 (이상치 제거 없음) ──────────────────────────────────────────
    save_cell(raw_out_dir, cell_id, {
        "cell_id": cell_id, "dataset": "TJU",
        "n_cycles": df["cycle"].nunique(),
    }, df)

    # ── 이상치 제거 후 _1_data_unified에 저장 (MIT/HUST와 동일 파이프라인 재사용) ──
    df = _make_time_relative(df)
    df = _fix_time_monotonicity(df)
    df, n_rest_removed = _remove_zero_current_rest(df)
    df, n_outliers = _remove_outlier_cycles(df)
    if df.empty:
        return {}

    dis_caps = (df[df["phase"] == "discharge"]
                .groupby("cycle")["capacity_Ah"].first()
                .dropna().sort_index())
    init_cap  = float(dis_caps.iloc[0])  if len(dis_caps) > 0 else np.nan
    final_cap = float(dis_caps.iloc[-1]) if len(dis_caps) > 0 else np.nan

    meta = {
        "cell_id":            cell_id,
        "dataset":            "TJU",
        "n_cycles":           df["cycle"].nunique(),
        "n_rest_removed":     n_rest_removed,
        "n_outliers_removed": n_outliers,
    }
    save_cell(out_dir, cell_id, meta, df)

    return {
        "cell_id":      cell_id,
        "total_cycles": df["cycle"].nunique(),
        "total_rows":   len(df),
        "init_cap_Ah":  round(init_cap,  4) if np.isfinite(init_cap)  else "",
        "final_cap_Ah": round(final_cap, 4) if np.isfinite(final_cap) else "",
    }


def _tju_worker(args):
    """top-level 함수 — Windows ProcessPoolExecutor 필수."""
    csv_path_str, out_dir_str, raw_out_dir_str = args
    try:
        stats = convert_tju_cell(Path(csv_path_str), Path(out_dir_str), Path(raw_out_dir_str))
        return ("ok", stats)
    except Exception:
        return ("err", f"{Path(csv_path_str).stem}:\n{traceback.format_exc()}")


def convert_tju(out_root: Path, target_cell: str | None = None, n_workers: int = 4) -> None:
    out_dir     = out_root / "TJU"
    raw_out_dir = RAW_OUTPUT_ROOT / "TJU"

    csv_files = sorted(TJU_DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[TJU] csv 파일 없음: {TJU_DATA_DIR}")
        return
    if target_cell:
        csv_files = [p for p in csv_files if p.stem == target_cell]
        if not csv_files:
            print(f"[TJU] cell '{target_cell}' 없음")
            return

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[TJU] {len(csv_files)}개 셀 변환  (workers={n_workers})")
    print(f"  raw     → {raw_out_dir}")
    print(f"  unified → {out_dir}")

    args_list = [(str(p), str(out_dir), str(raw_out_dir)) for p in csv_files]
    records, errors = _run(args_list, _tju_worker, "TJU cells", n_workers)

    print(f"  완료: {len(records)} 성공, {errors} 실패")
    if records:
        docs_dir = PROJECT_ROOT / "docs"
        docs_dir.mkdir(exist_ok=True)
        csv_path = docs_dir / "tju_conversion_summary.csv"
        pd.DataFrame(records).sort_values("cell_id").to_csv(csv_path, index=False)
        print(f"  요약 CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TJU(Tongji NCM) → 통일 포맷 변환")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cell", default=None, help="단일 셀만 변환 (예: 'CY25-05_1-#1')")
    args = parser.parse_args()
    convert_tju(Path(args.output_root), target_cell=args.cell, n_workers=args.workers)


if __name__ == "__main__":
    main()
