"""
diagnose_chg_removal_per_cell.py — 필터4의 두 충전 메커니즘을 셀별로 명확히 분리해서 확인.

배경: docs/DATASET_ANOMALIES.md에 있던 "chg_gap_s=120, chg_gap_factor=30 → 충전
phase 행만 제거"라는 서술이 실제 preprocess.py 코드와 맞지 않는다는 게 확인됐다.
실제로는 두 개의 별개 메커니즘이 있다:

  ① 충전 완전중단 (chg_gap_s=600s, chg_gap_factor=50×) → 충전 phase 행을 **실제로 삭제**
  ② 충전 CC전환갭  (chg_seg_gap_s=120s, chg_seg_gap_factor=30×) → 행은 유지, `chg_gap_seg=True`
     플래그만 기록 (hi_correlation.py가 이 플래그를 보고 충전 세그먼트 HI 계산을 건너뜀)

이 스크립트는 `preprocess.py`의 실제 `_remove_dt_gap_cycles()`(기본 파라미터, 재구현
아님)를 그대로 셀 하나하나에 호출해 ①/② 각각이 MIT 123셀 개별로 어느 정도 영향을
주는지(충전 행 삭제 비율, 사이클 플래그 비율) 정확히 분리해서 보여준다.

입력: _1_data_unified/MIT/*.pkl (없으면 D:\chanminLee\LFP_SOH_prediction_v2\_1_data_unified\MIT)
출력:
  2_preprocess/outputs/chg_gap_dt_diagnosis/chg_removal_per_cell.csv
  2_preprocess/outputs/chg_gap_dt_diagnosis/chg_removal_per_cell.png

사용:
  python 2_preprocess/diagnose_chg_removal_per_cell.py --workers 16
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import (  # noqa: E402  (실제 파이프라인 함수 그대로 재사용)
    _remove_empty_cycles,
    _fix_time_monotonicity,
    _remove_zero_current_rest,
    _remove_dt_gap_cycles,
)

for _font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"):
    if _font_name in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _font_name
        break
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_MIT_SRC = PROJECT_ROOT / "_1_data_unified" / "MIT"
_EXTERNAL_MIT_SRC = Path(r"D:\chanminLee\LFP_SOH_prediction_v2\_1_data_unified\MIT")
MIT_SRC = _LOCAL_MIT_SRC if _LOCAL_MIT_SRC.exists() else _EXTERNAL_MIT_SRC
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "chg_gap_dt_diagnosis"

BATCH_COLORS = {"b1": "#4C72B0", "b2": "#DD8452", "b3": "#55A868"}


def _batch_of(cell_id: str) -> str:
    m = re.match(r"(b\d+)c", cell_id)
    return m.group(1) if m else "unknown"


def scan_cell(pkl_path: Path) -> dict | None:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    meta = raw["meta"]
    df = raw["cycles"]
    cell_id = meta.get("cell_id", pkl_path.stem)
    batch = _batch_of(cell_id)

    df, _, _ = _remove_empty_cycles(df)
    df = _fix_time_monotonicity(df)
    df, _ = _remove_zero_current_rest(df)

    n_cycles_total = df["cycle"].nunique()
    n_chg_rows_before = int((df["phase"] == "charge").sum())

    (_df_clean,
     n_dis_removed, _dis_cycles,
     n_chg_rows_removed, chg_all_cycles,   # ① 완전중단: 행 삭제
     n_chg_seg, _chg_seg_cycles) = _remove_dt_gap_cycles(df)  # 기본 파라미터 그대로

    return {
        "cell_id": cell_id,
        "batch": batch,
        "n_cycles_total": n_cycles_total,
        "n_chg_rows_before": n_chg_rows_before,
        "n_chg_rows_removed": n_chg_rows_removed,          # ① 실제 삭제된 충전 행 수
        "chg_rows_removed_ratio": (n_chg_rows_removed / n_chg_rows_before
                                    if n_chg_rows_before else 0.0),
        "n_cycles_full_removed": len(chg_all_cycles),       # ① 완전중단으로 걸린 사이클 수
        "cycles_full_removed_ratio": (len(chg_all_cycles) / n_cycles_total
                                       if n_cycles_total else 0.0),
        "n_cycles_seg_flagged": n_chg_seg,                  # ② 플래그만 찍힌 사이클 수
        "cycles_seg_flagged_ratio": (n_chg_seg / n_cycles_total
                                      if n_cycles_total else 0.0),
        "n_dis_removed_cycles": n_dis_removed,
    }


def _scan_cell_worker(path_str: str) -> tuple:
    try:
        return ("ok", scan_cell(Path(path_str)))
    except Exception:
        return ("err", Path(path_str).stem + ":\n" + traceback.format_exc())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(MIT_SRC.glob("*.pkl"))
    if not paths:
        print(f"[중단] MIT 원본 폴더 없음: {MIT_SRC}")
        return

    records = []
    if args.workers <= 1:
        for p in tqdm(paths, desc="[진단] 셀별 필터4 충전 삭제/플래그 비율"):
            status, payload = _scan_cell_worker(str(p))
            if status == "ok":
                records.append(payload)
            else:
                print(f"\n  [ERR] {payload}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_scan_cell_worker, str(p)): p for p in paths}
            with tqdm(total=len(paths), desc=f"[진단] 셀별 필터4 충전 삭제/플래그 비율 (workers={args.workers})") as pbar:
                for fut in as_completed(futures):
                    status, payload = fut.result()
                    if status == "ok":
                        records.append(payload)
                    else:
                        print(f"\n  [ERR] {payload}")
                    pbar.update(1)

    df = pd.DataFrame(records).sort_values(["batch", "cell_id"]).reset_index(drop=True)
    csv_path = OUT_DIR / "chg_removal_per_cell.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 92)
    print("  필터4 — 셀별 ① 충전 행 실제삭제(600s×50) vs ② CC전환갭 플래그(120s×30) 비교")
    print("=" * 92)
    summary = df.groupby("batch").agg(
        n_cells=("cell_id", "nunique"),
        chg_rows_removed_ratio_mean=("chg_rows_removed_ratio", "mean"),
        chg_rows_removed_ratio_max=("chg_rows_removed_ratio", "max"),
        cycles_full_removed_ratio_mean=("cycles_full_removed_ratio", "mean"),
        cycles_seg_flagged_ratio_mean=("cycles_seg_flagged_ratio", "mean"),
    )
    print(summary.to_string())
    print("-" * 92)
    print(f"  셀별 원시 CSV(123행): {csv_path}")
    print("=" * 92)

    # ── 플랏: 123셀 각각의 ①행삭제 비율 vs ②플래그 비율 막대 (배치별 정렬) ──
    df_sorted = df.sort_values(["batch", "cell_id"]).reset_index(drop=True)
    x = np.arange(len(df_sorted))
    colors = df_sorted["batch"].map(BATCH_COLORS)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axes[0].bar(x, df_sorted["chg_rows_removed_ratio"] * 100, color=colors, width=0.9)
    axes[0].set_ylabel("① 충전 행 실제삭제 비율 (%)\n(600s×50, 완전중단)")
    axes[0].set_title("MIT 123셀 개별 — 필터4 충전 메커니즘 ①실제삭제 vs ②플래그만")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, df_sorted["cycles_seg_flagged_ratio"] * 100, color=colors, width=0.9)
    axes[1].set_ylabel("② 사이클 플래그 비율 (%)\n(120s×30, chg_gap_seg=True, 행 미삭제)")
    axes[1].set_xlabel("MIT 셀 (배치순 정렬: b1 → b2 → b3)")
    axes[1].grid(axis="y", alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BATCH_COLORS.values()]
    axes[0].legend(handles, BATCH_COLORS.keys(), loc="upper right")

    fig.tight_layout()
    out_path = OUT_DIR / "chg_removal_per_cell.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[완료] 플랏: {out_path}")


if __name__ == "__main__":
    main()
