"""
charge_gap_census.py — 충전 phase `chg_gap_seg` 플래그(Step2 필터4 결과)의 배치별 전수조사.

배경: validate_capacity_error.py로 MIT b2c0의 대표 곡선을 그렸을 때 충전 전류적산 점선이
전혀 안 보이는 걸 발견 — b2c0 전체 사이클의 충전 phase 행이 100% chg_gap_seg=True였다
(docs/DATASET_ANOMALIES.md 필터4의 "충전 단절 → 충전 phase 행만 제거" 로직 때문).
이게 b2c0만의 문제인지 배치 전체 문제인지 확인하기 위해 123 MIT + 77 HUST 전 셀을 스캔한다.

입력: _4_data_hi/clean/{MIT,HUST}/*.pkl
출력: 3_integrity/outputs/charge_gap_census.csv (셀별) + 콘솔에 배치별 집계 출력
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import pandas as pd

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_directories import DATA_4_HI_ROOT  # noqa: E402
CLEAN_DIR = DATA_4_HI_ROOT / "clean"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

PHASE_POS = 0.01  # 1_convert/convert_unified.py assign_phase()와 동일 임계값


def _batch_of(dataset: str, cell_id: str) -> str:
    if dataset == "MIT":
        m = re.match(r"(b\d+)c", cell_id)
        return m.group(1) if m else "unknown"
    m = re.match(r"(\d+)-", cell_id)
    return f"{m.group(1)}-" if m else "unknown"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in ("MIT", "HUST"):
        for p in sorted((CLEAN_DIR / ds).glob("*.pkl")):
            with open(p, "rb") as f:
                raw = pickle.load(f)
            df = raw["cycles"] if isinstance(raw, dict) else raw
            chg = df[df["current_A"] > PHASE_POS]
            n_chg_rows = len(chg)
            gap_ratio = float(chg["chg_gap_seg"].mean()) if n_chg_rows else float("nan")
            n_cycles = df["cycle"].nunique()
            n_cycles_all_gapped = 0
            if n_chg_rows:
                per_cyc = chg.groupby("cycle")["chg_gap_seg"].mean()
                n_cycles_all_gapped = int((per_cyc == 1.0).sum())
            cell_id = p.stem
            rows.append({
                "dataset": ds,
                "batch": _batch_of(ds, cell_id),
                "cell_id": cell_id,
                "n_cycles": n_cycles,
                "n_chg_rows": n_chg_rows,
                "gap_ratio": gap_ratio,
                "n_cycles_all_gapped": n_cycles_all_gapped,
                "frac_cycles_all_gapped": n_cycles_all_gapped / n_cycles if n_cycles else float("nan"),
            })

    census = pd.DataFrame(rows)
    out_path = OUT_DIR / "charge_gap_census.csv"
    census.to_csv(out_path, index=False)

    summary = census.groupby(["dataset", "batch"]).agg(
        n_cells=("cell_id", "nunique"),
        gap_ratio_mean=("gap_ratio", "mean"),
        gap_ratio_min=("gap_ratio", "min"),
        gap_ratio_max=("gap_ratio", "max"),
        n_cells_fully_gapped=("gap_ratio", lambda s: (s == 1.0).sum()),
        frac_cycles_all_gapped_mean=("frac_cycles_all_gapped", "mean"),
    )
    print(summary.to_string())
    print(f"\n[완료] 셀별 원시 결과: {out_path}")


if __name__ == "__main__":
    main()
