"""
diagnose_chg_gap_dt.py — 필터4(_remove_dt_gap_cycles) 충전 단절 기준 재설정을 위한 진단.

배경: docs/DATASET_ANOMALIES.md "충전 단절의 배치별 편중(전수조사)" — MIT batch2
43셀 전부가 충전 phase 사이클의 99%+에서 `chg_gap_seg=True`로 찍혀, 그 사이클들의
충전 세그먼트 HI가 통째로 NaN→0 대체된 채 학습/평가에 들어가는 문제가 확인됐다
(오염 규모: 전체 세그먼트 코퍼스의 약 10~11%, 3_integrity/validate_capacity_error.py
capacity_error_by_current.png 조사 과정에서 발견).

현재 기준(preprocess.py `_remove_dt_gap_cycles` 기본값):
  단절 판정 = dt.max() > max(gap_s, dt_med × gap_factor)
  - 충전 완전중단(행 삭제):     chg_gap_s=600s,     chg_gap_factor=50×
  - 충전 CC전환갭(플래그만):    chg_seg_gap_s=120s, chg_seg_gap_factor=30×

배율 기준(dt_med × factor)이 batch2처럼 dt_med 자체가 큰(=원래 샘플링 간격이
성긴) 셀에서는 "진짜 단절"이 아니라 "그냥 샘플링 간격이 넓다"만으로도 쉽게
트리거될 수 있다 — 이걸 실제 데이터로 확인하는 게 이 스크립트의 목적이다
(사용자 요구사항: "그냥 샘플링 타임이 늘어난 거 말고 진짜 단절된 것을 분리").

방법: `_1_data_unified/MIT/*.pkl`(필터4 이전, phase 컬럼 있음)에 preprocess.py의
필터1~3(빈 사이클 제거, time 단조 보정, rest 0전류 제거)까지 **동일 함수를 그대로
재사용**해 적용한 뒤(필터4가 실제로 보는 것과 같은 df 상태), 사이클별 충전 phase
인접행 간격(dt)을 전수 계산한다. 필터4 자체는 적용하지 않는다 — 그 판정 없이
"있는 그대로의 dt 분포"를 보기 위함.

주의: 로컬 프로젝트 폴더( PROJECT_ROOT/_1_data_unified )에는 원본이 없다(대용량이라
별도 드라이브에 보관 중) — 사용자 확인 결과 실제 위치는 D:\chanminLee\
LFP_SOH_prediction_v2\_1_data_unified 이며, 아래 `MIT_SRC`가 그 경로를 가리킨다.

출력:
  2_preprocess/outputs/chg_gap_dt_diagnosis/chg_dt_per_cycle.csv   — 사이클별 요약
  2_preprocess/outputs/chg_gap_dt_diagnosis/chg_dt_dist_by_batch.png   — dt_max 분포(배치별)
  2_preprocess/outputs/chg_gap_dt_diagnosis/chg_dt_med_vs_max.png     — dt_med vs dt_max 산점도
  2_preprocess/outputs/chg_gap_dt_diagnosis/chg_dt_all_values_by_batch.png — 개별 dt값 전체 분포

사용:
  python 2_preprocess/diagnose_chg_gap_dt.py
  python 2_preprocess/diagnose_chg_gap_dt.py --workers 16   # 셀 단위 병렬화(D드라이브 I/O 병목 완화)
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
from preprocess import (  # noqa: E402  (같은 필터1~3을 그대로 재사용 — 로직 중복 방지)
    _remove_empty_cycles,
    _fix_time_monotonicity,
    _remove_zero_current_rest,
)

for _font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"):
    if _font_name in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _font_name
        break
matplotlib.rcParams["axes.unicode_minus"] = False


def _use_plain_log_ticks(ax, x: bool = False, y: bool = False) -> None:
    """로그축 기본 포매터(LogFormatterSciNotation)는 지수(예: 10^-1)를 mathtext로
    그려 axes.unicode_minus 설정을 타지 않는다 — 한글 폰트엔 그 마이너스 글리프가
    없어 "10^x1"처럼 깨진다. 지수 표기 대신 그냥 숫자(0.1, 1, 10 ...)로 표시해 우회."""
    fmt = matplotlib.ticker.FuncFormatter(
        lambda v, _pos: f"{v:g}" if v != 0 else "0")
    if x:
        ax.xaxis.set_major_formatter(fmt)
    if y:
        ax.yaxis.set_major_formatter(fmt)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_MIT_SRC   = PROJECT_ROOT / "_1_data_unified" / "MIT"
_EXTERNAL_MIT_SRC = Path(r"D:\chanminLee\LFP_SOH_prediction_v2\_1_data_unified\MIT")
MIT_SRC = _LOCAL_MIT_SRC if _LOCAL_MIT_SRC.exists() else _EXTERNAL_MIT_SRC
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "chg_gap_dt_diagnosis"

# 현재 preprocess.py 기본값 (참고선으로 플랏에 표시)
CHG_GAP_S, CHG_GAP_FACTOR = 600.0, 50.0          # 완전중단(행 삭제)
CHG_SEG_GAP_S, CHG_SEG_GAP_FACTOR = 120.0, 30.0  # CC전환갭(플래그만)

BATCH_COLORS = {"b1": "#4C72B0", "b2": "#DD8452", "b3": "#55A868"}


def _batch_of(cell_id: str) -> str:
    m = re.match(r"(b\d+)c", cell_id)
    return m.group(1) if m else "unknown"


def _apply_filters_1_3(df: pd.DataFrame) -> pd.DataFrame:
    """필터4가 실제로 보는 df 상태를 재현 — preprocess.py와 동일한 순서로 필터1~3 적용."""
    df, _, _ = _remove_empty_cycles(df)
    df = _fix_time_monotonicity(df)
    df, _ = _remove_zero_current_rest(df)
    return df


def scan_cell(pkl_path: Path) -> tuple[pd.DataFrame, list[float]]:
    """셀 1개(`_1_data_unified` 기준, 필터1~3 적용) → (사이클별 충전 dt 요약, 개별 dt 값 리스트)."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    meta = raw["meta"]
    df = raw["cycles"]
    cell_id = meta.get("cell_id", pkl_path.stem)
    batch = _batch_of(cell_id)

    df = _apply_filters_1_3(df)

    rows = []
    all_dt: list[float] = []
    for cyc, grp in df.groupby("cycle"):
        chg = grp[grp["phase"] == "charge"].sort_values("time_s")
        if len(chg) <= 5:
            continue
        t = chg["time_s"].to_numpy(dtype=float)
        dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
        dt_pos = dt[dt > 0]
        if len(dt_pos) == 0:
            continue
        dt_med = float(np.median(dt_pos))
        dt_max = float(dt.max())
        rows.append({
            "cell_id": cell_id, "batch": batch, "cycle": int(cyc),
            "n_rows": len(chg),
            "dt_med": dt_med, "dt_max": dt_max,
            "dt_p90": float(np.percentile(dt_pos, 90)),
            "dt_p99": float(np.percentile(dt_pos, 99)),
            "ratio_max_over_med": dt_max / dt_med if dt_med > 0 else np.nan,
            "flagged_seg_current":  dt_max > max(CHG_SEG_GAP_S, dt_med * CHG_SEG_GAP_FACTOR),
            "flagged_full_current": dt_max > max(CHG_GAP_S, dt_med * CHG_GAP_FACTOR),
        })
        all_dt.extend(dt_pos.tolist())

    return pd.DataFrame(rows), all_dt


def _scan_cell_worker(path_str: str) -> tuple:
    """top-level 함수 — Windows ProcessPoolExecutor 필수(preprocess.py 관례와 동일)."""
    try:
        cyc_df, dt_list = scan_cell(Path(path_str))
        return ("ok", cyc_df, dt_list)
    except Exception:
        return ("err", Path(path_str).stem + ":\n" + traceback.format_exc(), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=1,
                        help="셀 단위 병렬 프로세스 수(기본: 1 = 순차 실행). D드라이브처럼 "
                             "느린 원본 경로를 읽을 때 늘리면 유리.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(MIT_SRC.glob("*.pkl"))
    if not paths:
        print(f"[중단] MIT 원본 폴더 없음: {MIT_SRC}")
        return

    per_cycle_frames = []
    all_dt_by_batch: dict[str, list[float]] = {"b1": [], "b2": [], "b3": []}

    if args.workers <= 1:
        for p in tqdm(paths, desc="[진단] MIT 셀별 충전 dt 스캔"):
            status, cyc_df, dt_list = _scan_cell_worker(str(p))
            if status == "ok" and not cyc_df.empty:
                per_cycle_frames.append(cyc_df)
                batch = cyc_df["batch"].iloc[0]
                all_dt_by_batch.setdefault(batch, []).extend(dt_list)
            elif status == "err":
                print(f"\n  [ERR] {cyc_df}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_scan_cell_worker, str(p)): p for p in paths}
            with tqdm(total=len(paths), desc=f"[진단] MIT 셀별 충전 dt 스캔 (workers={args.workers})") as pbar:
                for fut in as_completed(futures):
                    status, cyc_df, dt_list = fut.result()
                    if status == "ok" and not cyc_df.empty:
                        per_cycle_frames.append(cyc_df)
                        batch = cyc_df["batch"].iloc[0]
                        all_dt_by_batch.setdefault(batch, []).extend(dt_list)
                    elif status == "err":
                        print(f"\n  [ERR] {cyc_df}")
                    pbar.update(1)

    per_cycle = pd.concat(per_cycle_frames, ignore_index=True) if per_cycle_frames else pd.DataFrame()
    if per_cycle.empty:
        print("[중단] 유효한 충전 사이클이 하나도 없습니다.")
        return

    csv_path = OUT_DIR / "chg_dt_per_cycle.csv"
    per_cycle.to_csv(csv_path, index=False)

    # ── 콘솔 요약 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  충전 phase 인접행 간격(dt) — 배치별 요약 (필터1~3 적용 후, 필터4 미적용)")
    print("=" * 78)
    summary = per_cycle.groupby("batch").agg(
        n_cells=("cell_id", "nunique"),
        n_cycles=("cycle", "size"),
        dt_med_median=("dt_med", "median"),
        dt_med_p90=("dt_med", lambda s: s.quantile(0.9)),
        dt_max_median=("dt_max", "median"),
        dt_max_p90=("dt_max", lambda s: s.quantile(0.9)),
        flagged_seg_ratio=("flagged_seg_current", "mean"),
        flagged_full_ratio=("flagged_full_current", "mean"),
    )
    print(summary.to_string())
    print("-" * 78)
    print(f"  현재 기준: 완전중단 {CHG_GAP_S}s×{CHG_GAP_FACTOR}, "
          f"CC전환갭(플래그) {CHG_SEG_GAP_S}s×{CHG_SEG_GAP_FACTOR}")
    print(f"  전체 CSV: {csv_path}")
    print("=" * 78)

    # ── 플랏 A: dt_max 분포 (배치별, log-x) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(np.log10(max(per_cycle["dt_max"].min(), 0.1)),
                        np.log10(per_cycle["dt_max"].max()), 60)
    for batch, color in BATCH_COLORS.items():
        vals = per_cycle.loc[per_cycle["batch"] == batch, "dt_max"]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.5, color=color, label=f"{batch} (n={len(vals)})")
    ax.axvline(CHG_SEG_GAP_S, color="black", linestyle="--", linewidth=1,
               label=f"chg_seg_gap_s={CHG_SEG_GAP_S:.0f}s (플래그 절대기준)")
    ax.axvline(CHG_GAP_S, color="red", linestyle="--", linewidth=1,
               label=f"chg_gap_s={CHG_GAP_S:.0f}s (완전중단 절대기준)")
    ax.set_xscale("log")
    _use_plain_log_ticks(ax, x=True)
    ax.set_xlabel("사이클별 충전 dt 최댓값 (초, log scale)")
    ax.set_ylabel("사이클 수")
    ax.set_title("배치별 충전 dt_max 분포 (필터1~3 적용 후)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chg_dt_dist_by_batch.png", dpi=150)
    plt.close(fig)

    # ── 플랏 B: dt_med vs dt_max 산점도 (배율 기준선 포함) ──────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for batch, color in BATCH_COLORS.items():
        sub = per_cycle[per_cycle["batch"] == batch]
        if sub.empty:
            continue
        ax.scatter(sub["dt_med"], sub["dt_max"], s=6, alpha=0.35, color=color, label=batch)
    x_ref = np.logspace(np.log10(max(per_cycle["dt_med"].min(), 0.1)),
                         np.log10(per_cycle["dt_med"].max()), 50)
    ax.plot(x_ref, x_ref * CHG_SEG_GAP_FACTOR, "k--", linewidth=1,
            label=f"dt_med × {CHG_SEG_GAP_FACTOR:.0f} (플래그 배율기준)")
    ax.plot(x_ref, x_ref * CHG_GAP_FACTOR, "r--", linewidth=1,
            label=f"dt_med × {CHG_GAP_FACTOR:.0f} (완전중단 배율기준)")
    ax.axhline(CHG_SEG_GAP_S, color="black", linestyle=":", linewidth=1)
    ax.axhline(CHG_GAP_S, color="red", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _use_plain_log_ticks(ax, x=True, y=True)
    ax.set_xlabel("사이클 내 충전 dt 중앙값 (dt_med, 초, log)")
    ax.set_ylabel("사이클 내 충전 dt 최댓값 (dt_max, 초, log)")
    ax.set_title("dt_med vs dt_max — 점이 배율 기준선 위면 현재 기준으로 '단절' 판정됨")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chg_dt_med_vs_max.png", dpi=150)
    plt.close(fig)

    # ── 플랏 C: 개별 dt 값 전체 분포 (배치별, 정상 샘플링 간격 자체 비교) ───
    fig, ax = plt.subplots(figsize=(9, 5))
    all_vals_concat = np.concatenate([v for v in all_dt_by_batch.values() if v]) if any(all_dt_by_batch.values()) else np.array([1.0])
    bins_c = np.logspace(np.log10(max(np.min(all_vals_concat), 0.05)),
                          np.log10(np.max(all_vals_concat)), 80)
    for batch, color in BATCH_COLORS.items():
        vals = all_dt_by_batch.get(batch, [])
        if not vals:
            continue
        ax.hist(vals, bins=bins_c, alpha=0.5, color=color, density=True,
                label=f"{batch} (n={len(vals):,})")
    ax.axvline(CHG_SEG_GAP_S, color="black", linestyle="--", linewidth=1)
    ax.axvline(CHG_GAP_S, color="red", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    _use_plain_log_ticks(ax, x=True)
    ax.set_xlabel("개별 인접행 간격 dt (초, log scale) — 전체 충전 행 대상")
    ax.set_ylabel("밀도")
    ax.set_title("배치별 충전 행 개별 dt 값 분포 (정상 샘플링 간격 자체 비교)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chg_dt_all_values_by_batch.png", dpi=150)
    plt.close(fig)

    print(f"\n[완료] 출력 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
