"""
audit_zero_info_hi_segments.py — 학습에 실제로 들어가는 "정보량 0" 충전 세그먼트 비율을
MIT 123셀 개별로, 실제 학습 입력 데이터(q_frac_wide seg pkl) 기준으로 직접 센다.

배경: chg_gap_seg=True 사이클은 hi_correlation.py가 충전 세그먼트(chg_lo/chg_mid/chg_hi)
HI 계산을 건너뛰어 그 세그먼트의 HI 66개 컬럼이 전부 NaN이 되고, 이후
5_model/datasets/segment_dataset.py의 SegmentNormalizer가 NaN→z-score 평균(0)으로
채워서 "정보량 0" 입력으로 학습/평가에 그대로 들어간다(라벨 capacity_Ah는 방전
기준이라 정상값 유지). 이 스크립트는 그 정보량 0 비율을 셀별로 직접 측정한다 —
preprocess.py 필터4의 플래그 여부를 다시 계산하는 게 아니라, **실제 학습 입력
파일(_4_data_hi/q_frac_wide/.../seg/MIT/*.pkl)을 그대로 읽어서** 몇 %가 정보량
0인지 센다.

기준 데이터셋: 5_model/config/exp_qfw_mlp_basefix.yaml (0804_basefix 베이스라인)의
axis_config={n1:0.35, n2:0.2} → _4_data_hi/q_frac_wide/n1-35%_n2-20%_N-4/seg/

zone(segment_id) 매핑 (common/scenario/q_frac_wide.py _SCENARIO_NAMES):
  0=chg_lo, 1=chg_mid, 2=chg_hi (충전) | 3=dis_hi, 4=dis_mid, 5=dis_lo (방전)

출력:
  3_integrity/outputs/zero_info_hi_per_cell.csv   — MIT 123셀 개별 결과
  3_integrity/outputs/zero_info_hi_per_cell.png   — 셀별 막대 플랏(배치순 정렬)

사용:
  python 3_integrity/audit_zero_info_hi_segments.py
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

for _font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"):
    if _font_name in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _font_name
        break
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_directories import DATA_4_HI_ROOT  # noqa: E402
SEG_DIR = DATA_4_HI_ROOT / "q_frac_wide" / "n1-35%_n2-20%_N-4" / "seg" / "MIT"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

META_COLS = {"cell_id", "cycle", "segment_id", "capacity_Ah", "scen", "raw_v", "raw_i", "raw_t"}
CHARGE_SEGMENT_IDS = {0, 1, 2}  # chg_lo, chg_mid, chg_hi

BATCH_COLORS = {"b1": "#4C72B0", "b2": "#DD8452", "b3": "#55A868"}


def _batch_of(cell_id: str) -> str:
    m = re.match(r"(b\d+)c", cell_id)
    return m.group(1) if m else "unknown"


def scan_cell(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    df = raw["cycles"] if isinstance(raw, dict) else raw
    cell_id = pkl_path.stem
    batch = _batch_of(cell_id)

    hi_cols = [c for c in df.columns if c not in META_COLS]
    all_nan = df[hi_cols].isna().all(axis=1)

    is_charge = df["segment_id"].isin(CHARGE_SEGMENT_IDS)

    n_total_rows = len(df)
    n_charge_rows = int(is_charge.sum())
    n_zero_info_charge_rows = int((is_charge & all_nan).sum())
    n_zero_info_discharge_rows = int((~is_charge & all_nan).sum())

    return {
        "cell_id": cell_id,
        "batch": batch,
        "n_total_seg_rows": n_total_rows,
        "n_charge_seg_rows": n_charge_rows,
        "n_zero_info_charge_rows": n_zero_info_charge_rows,
        "zero_info_ratio_of_charge": (n_zero_info_charge_rows / n_charge_rows
                                       if n_charge_rows else 0.0),
        "zero_info_ratio_of_total": (n_zero_info_charge_rows / n_total_rows
                                      if n_total_rows else 0.0),
        "n_zero_info_discharge_rows": n_zero_info_discharge_rows,  # sanity check, 항상 ~0 기대
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(SEG_DIR.glob("*.pkl"))
    if not paths:
        print(f"[중단] 세그먼트 pkl 없음: {SEG_DIR}")
        return

    records = [scan_cell(p) for p in tqdm(paths, desc="[감사] 셀별 정보량0 충전세그먼트 비율")]
    df = pd.DataFrame(records).sort_values(["batch", "cell_id"]).reset_index(drop=True)

    csv_path = OUT_DIR / "zero_info_hi_per_cell.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 92)
    print("  학습 입력 기준 — MIT 셀별 '정보량 0'(HI 전부 NaN→0) 충전 세그먼트 비율")
    print(f"  데이터: {SEG_DIR}")
    print("=" * 92)
    summary = df.groupby("batch").agg(
        n_cells=("cell_id", "nunique"),
        total_seg_rows=("n_total_seg_rows", "sum"),
        zero_info_charge_rows=("n_zero_info_charge_rows", "sum"),
        ratio_of_charge_mean=("zero_info_ratio_of_charge", "mean"),
        ratio_of_total_mean=("zero_info_ratio_of_total", "mean"),
        sanity_discharge_leak=("n_zero_info_discharge_rows", "sum"),
    )
    print(summary.to_string())

    grand_total_rows = int(df["n_total_seg_rows"].sum())
    grand_zero_info = int(df["n_zero_info_charge_rows"].sum())
    print("-" * 92)
    print(f"  전체 MIT 세그먼트 행 수: {grand_total_rows:,}  |  "
          f"정보량0 충전 세그먼트: {grand_zero_info:,}  "
          f"({grand_zero_info/grand_total_rows*100:.2f}% of MIT 전체 세그먼트)")
    print(f"  셀별 원시 CSV(123행): {csv_path}")
    print("=" * 92)

    # ── 플랏: 셀별 zero_info_ratio_of_total (배치순 정렬) ──────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(df))
    colors = df["batch"].map(BATCH_COLORS)
    ax.bar(x, df["zero_info_ratio_of_total"] * 100, color=colors, width=0.9)
    ax.set_xlabel("MIT 셀 (배치순 정렬: b1 → b2 → b3)")
    ax.set_ylabel("정보량0 충전 세그먼트 비율 (%)\n(그 셀의 전체 세그먼트 행 중, HI 전부 NaN→0)")
    ax.set_title("MIT 123셀 개별 — 학습 입력에 실제로 들어가는 '정보량 0' 충전 세그먼트 비율")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BATCH_COLORS.values()]
    ax.legend(handles, BATCH_COLORS.keys(), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / "zero_info_hi_per_cell.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[완료] 플랏: {out_path}")


if __name__ == "__main__":
    main()
