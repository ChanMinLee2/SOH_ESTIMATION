"""
shaping_features.py
===================
4_hi_analysis/hi_features.pkl (DataFrame, per-cycle HI features)를
5_train/train.py 의 BatterySOHDataset이 기대하는
list-of-dict 형식으로 변환하여 pkl로 저장합니다.

출력 포맷 (per item):
    {
        "cell"      : str         -- 셀 ID
        "cyc"       : int         -- 사이클 번호
        "x"         : np.ndarray  -- HI 피처 벡터 (64-D or 67-D with --add-meta)
        "soc_label" : int         -- -2(High), -1(Mid), 0(Low)
        "mode_label": int         -- 1(Charge), 0(Discharge)
        "length_p"  : float       -- 세그먼트 길이 비율 (hi_features.pkl에 없으므로 1.0 고정)
        "y"         : float       -- capacity_Ah (RUL 정보 없으므로 capacity로 대체)
        "capacity"  : float       -- 방전 용량 [Ah]
    }

변환 전략
---------
hi_features.pkl의 각 행(사이클)에는 방전 + 충전 HI가 모두 포함됩니다.
preprocess.ipynb Phase 3 방식을 모사해 6-scenario로 expand합니다:

  방전(mode_label=0): soc_label -2, -1, 0
  충전(mode_label=1): soc_label -2, -1, 0

전체 64개 HI 컬럼을 x로 구성하되, 시나리오에 무관한 HI는 0으로 마스킹합니다.

사용법
------
  python shaping_features.py [옵션]

옵션
----
  --src   : hi_features.pkl 경로 (기본: ../4_hi_analysis/hi_features.pkl)
  --dst   : 출력 pkl 경로       (기본: ./hi_shaped.pkl)
  --add-meta : x 끝에 [soc_label, mode_label, length_p] 3개를 추가 (64->67-D)
  --dataset  : 사용할 dataset 필터 ('MIT', 'HUST', 'all')  기본: 'all'
  --verbose  : 진행 상황을 더 자세히 출력
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# HI 컬럼 정의 (hi_correlation.py HI_META 64개와 동일한 순서)
# ---------------------------------------------------------------------------

# 방전 Global (22개)
DIS_GLOBAL_COLS = [
    "v_mean", "v_std", "v_skew", "v_kurt", "v_end", "v_drop",
    "energy_Wh", "v_energy",
    "v_at_q20", "v_at_q50", "v_at_q80",
    "q_high_v", "q_tail", "q_plateau_ratio",
    "t_discharge",
    "ica_peak_h", "ica_peak_v", "ica_peak_area", "dvdq_min",
    "ce",
    "temp_mean", "temp_max",
]

# 방전 SoC 구간별 HI (6개 x 3구간 = 18개)
DIS_SOC_COLS = {
    -2: ["v_mean_s_hi",  "v_std_s_hi",  "energy_Wh_s_hi",  "q_abs_s_hi",  "ica_peak_h_s_hi",  "dvdq_min_s_hi"],
    -1: ["v_mean_s_mid", "v_std_s_mid", "energy_Wh_s_mid", "q_abs_s_mid", "ica_peak_h_s_mid", "dvdq_min_s_mid"],
     0: ["v_mean_s_lo",  "v_std_s_lo",  "energy_Wh_s_lo",  "q_abs_s_lo",  "ica_peak_h_s_lo",  "dvdq_min_s_lo"],
}

# 충전 Global (6개)
CHG_GLOBAL_COLS = [
    "q_cc_ratio", "chg_v_energy",
    "chg_ica_peak_h", "chg_ica_peak_v", "chg_ica_peak_area", "chg_dvdq_min",
]

# 충전 SoC 구간별 HI (6개 x 3구간 = 18개)
CHG_SOC_COLS = {
    -2: ["v_mean_chg_s_hi",  "v_std_chg_s_hi",  "energy_Wh_chg_s_hi",  "q_abs_chg_s_hi",  "ica_peak_h_chg_s_hi",  "dvdq_min_chg_s_hi"],
    -1: ["v_mean_chg_s_mid", "v_std_chg_s_mid", "energy_Wh_chg_s_mid", "q_abs_chg_s_mid", "ica_peak_h_chg_s_mid", "dvdq_min_chg_s_mid"],
     0: ["v_mean_chg_s_lo",  "v_std_chg_s_lo",  "energy_Wh_chg_s_lo",  "q_abs_chg_s_lo",  "ica_peak_h_chg_s_lo",  "dvdq_min_chg_s_lo"],
}

# 전체 HI 컬럼 순서 (64개 = 방전22 + 방전SoC18 + 충전6 + 충전SoC18)
ALL_HI_COLS = (
    DIS_GLOBAL_COLS
    + DIS_SOC_COLS[-2] + DIS_SOC_COLS[-1] + DIS_SOC_COLS[0]
    + CHG_GLOBAL_COLS
    + CHG_SOC_COLS[-2] + CHG_SOC_COLS[-1] + CHG_SOC_COLS[0]
)

# 시나리오별 활성 컬럼 집합
DIS_SOC_ACTIVE = {
    -2: set(DIS_GLOBAL_COLS + DIS_SOC_COLS[-2]),
    -1: set(DIS_GLOBAL_COLS + DIS_SOC_COLS[-1]),
     0: set(DIS_GLOBAL_COLS + DIS_SOC_COLS[0]),
}
CHG_SOC_ACTIVE = {
    -2: set(CHG_GLOBAL_COLS + CHG_SOC_COLS[-2]),
    -1: set(CHG_GLOBAL_COLS + CHG_SOC_COLS[-1]),
     0: set(CHG_GLOBAL_COLS + CHG_SOC_COLS[0]),
}


# ---------------------------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------------------------

def build_scenario_mask(soc_label: int, mode_label: int) -> np.ndarray:
    """시나리오에 해당하지 않는 HI 피처를 0으로 마스킹하는 binary 벡터를 반환합니다."""
    if mode_label == 0:
        active = DIS_SOC_ACTIVE[soc_label]
    else:
        active = CHG_SOC_ACTIVE[soc_label]
    return np.array([1.0 if c in active else 0.0 for c in ALL_HI_COLS], dtype=np.float32)


def row_to_x_vector(row: pd.Series) -> np.ndarray:
    """DataFrame 행에서 64-D HI 벡터를 추출합니다. 누락/NaN 컬럼은 0으로 채웁니다."""
    vals = []
    for col in ALL_HI_COLS:
        v = row.get(col, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            vals.append(0.0)
        else:
            vals.append(float(v))
    return np.array(vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# 메인 변환 함수
# ---------------------------------------------------------------------------

def convert(
    src_path: Path,
    dst_path: Path,
    add_meta: bool = False,
    dataset_filter: str = "all",
    verbose: bool = False,
) -> None:
    """
    hi_features.pkl → final_data 형식 pkl 변환

    Parameters
    ----------
    src_path      : 입력 pkl 경로 (hi_features.pkl)
    dst_path      : 출력 pkl 경로
    add_meta      : True이면 x 끝에 [soc_label, mode_label, length_p] 3개 추가
    dataset_filter: 'MIT', 'HUST', 또는 'all'
    verbose       : 상세 로그 출력 여부
    """
    print(f"[shaping_features] Loading: {src_path}")
    sys.setrecursionlimit(5000)
    warnings.filterwarnings("ignore")

    with open(src_path, "rb") as f:
        df: pd.DataFrame = pickle.load(f)

    print(f"  -> DataFrame shape : {df.shape}")
    print(f"  -> Datasets found  : {df['dataset'].unique().tolist()}")
    print(f"  -> Unique cells    : {df['cell_id'].nunique()}")
    print(f"  -> Cycle range     : {int(df['cycle'].min())} ~ {int(df['cycle'].max())}")

    # 데이터셋 필터
    if dataset_filter.lower() != "all":
        df = df[df["dataset"].str.upper() == dataset_filter.upper()].reset_index(drop=True)
        print(f"  -> After filter '{dataset_filter}': {len(df)} rows")

    # HI 컬럼 존재 여부 확인
    missing_cols = [c for c in ALL_HI_COLS if c not in df.columns]
    if missing_cols:
        print(f"  [Warning] {len(missing_cols)} HI columns not found in source pkl (will be 0):")
        for mc in missing_cols:
            print(f"    - {mc}")

    # 6 시나리오: (mode_label, soc_label)
    scenarios = [
        (0, -2),  # Discharge-High  (SoC 60~100%)
        (0, -1),  # Discharge-Mid   (SoC 30~60%)
        (0,  0),  # Discharge-Low   (SoC  0~30%)
        (1, -2),  # Charge-High     (SoC 60~100%)
        (1, -1),  # Charge-Mid      (SoC 30~60%)
        (1,  0),  # Charge-Low      (SoC  0~30%)
    ]

    # 시나리오별 마스크 사전 계산
    masks = {(m, s): build_scenario_mask(s, m) for m, s in scenarios}

    final_data = []

    print(f"\n[shaping_features] Expanding {len(df)} cycles x {len(scenarios)} scenarios ...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="rows"):
        cell_id  = str(row["cell_id"])
        cyc      = int(row["cycle"])
        cap_raw  = row.get("capacity_Ah", np.nan)
        capacity = 0.0 if (cap_raw is None or (isinstance(cap_raw, float) and np.isnan(cap_raw))) else float(cap_raw)
        y_val    = capacity  # y = capacity (RUL 정보 없음)

        # 64-D 기본 벡터 (시나리오 마스킹 전)
        x_base = row_to_x_vector(row)

        for mode_label, soc_label in scenarios:
            x_val = x_base * masks[(mode_label, soc_label)]

            if add_meta:
                meta = np.array([float(soc_label), float(mode_label), 1.0], dtype=np.float32)
                x_val = np.concatenate([x_val, meta])

            final_data.append({
                "cell"       : cell_id,
                "cyc"        : cyc,
                "x"          : x_val,
                "soc_label"  : soc_label,
                "mode_label" : mode_label,
                "length_p"   : 1.0,
                "y"          : y_val,
                "capacity"   : capacity,
            })

        if verbose and len(final_data) % 100000 == 0 and len(final_data) > 0:
            print(f"  ... {len(final_data):,} items generated")

    print(f"\n[shaping_features] Conversion complete:")
    print(f"  Total items   : {len(final_data):,}  ({len(df)} cycles x {len(scenarios)} scenarios)")
    print(f"  x dimension   : {final_data[0]['x'].shape[0]}-D")
    print(f"  Unique cells  : {len(set(d['cell'] for d in final_data))}")
    cap_vals = [d['capacity'] for d in final_data if d['capacity'] > 0]
    if cap_vals:
        print(f"  Capacity range: {min(cap_vals):.4f} ~ {max(cap_vals):.4f} Ah")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[shaping_features] Saving to: {dst_path}")
    with open(dst_path, "wb") as f:
        pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[shaping_features] Done!")

    # 간략 샘플 출력
    sample = final_data[0]
    print("\n[Sample item]")
    print(f"  cell={sample['cell']}, cyc={sample['cyc']}, "
          f"mode_label={sample['mode_label']}, soc_label={sample['soc_label']}")
    print(f"  x shape={sample['x'].shape}, x[:5]={sample['x'][:5]}")
    print(f"  y={sample['y']:.4f}, capacity={sample['capacity']:.4f}, length_p={sample['length_p']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hi_features.pkl -> train.py 입력 포맷(list-of-dict) 변환"
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "4_hi_analysis" / "hi_features.pkl",
        help="입력 hi_features.pkl 경로 (기본: ../4_hi_analysis/hi_features.pkl)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(__file__).resolve().parent / "hi_shaped.pkl",
        help="출력 pkl 경로 (기본: ./hi_shaped.pkl)",
    )
    parser.add_argument(
        "--add-meta",
        action="store_true",
        default=False,
        help="x 벡터 끝에 [soc_label, mode_label, length_p] 3개를 추가 (64->67-D)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "MIT", "HUST"],
        help="사용할 dataset 필터 (기본: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="상세 로그 출력",
    )
    args = parser.parse_args()

    convert(
        src_path=args.src,
        dst_path=args.dst,
        add_meta=args.add_meta,
        dataset_filter=args.dataset,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
