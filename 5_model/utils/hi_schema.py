"""
64-HI schema for SCR.

STAT(20) + DIFF(20) + LFP(20) + MORPH(6) = 66 per segment.
Excluded from model input:
  stat_q_abs    — capacity_Ah와 1:1 동일 (leakage)
  stat_energy_seg — q_abs × 평균전압으로 종속 변수, 정보 중복
=> 64 input HIs per segment.
"""

from __future__ import annotations

STAT_KEYS: list[str] = [
    "v_mean_cw", "v_std", "v_skew", "v_kurt", "v_ent",
    "i_mean", "i_std", "v_med", "corr_qi", "corr_vi",
    "q_abs", "energy_seg", "v_iqr", "v_range", "v_p10",
    "v_p90", "v_samp_ent", "corr_vt", "i_q_slope", "v_detrended_std",
]

DIFF_KEYS: list[str] = [
    "dvdq_mean", "dvdq_std", "dvdq_max_abs", "dvdq_min", "dvdq_area",
    "dqdv_peak_h", "dqdv_peak_v", "dqdv_peak_w", "dqdv_area", "v_trend_slope",
    "dqdv_peak_asym", "d2vdq2_rms", "dvdq_skew", "dvdq_ent", "dv_di_seg",
    "dqdv_valley_h", "dqdv_valley_v", "dvdq_peak_q", "dvdq_flat_q", "dqdv_area_asym",
]

LFP_KEYS: list[str] = [
    "plateau_frac", "plateau_v_mean", "plateau_v_std", "plateau_dvdq_std",
    "nonlin_idx", "v_dev_mid", "v_flatness", "delta_v_rms",
    "vq_slope_mid", "inflect_v", "inflect_q_frac", "v_concavity",
    "phase_entry_dvdq", "v_q_pearson", "ica_peak_cnt",
    "plateau_v_slope", "v_gradient_exit", "plateau_q_onset", "dv_dt_plateau", "v_ent_plateau",
]

MORPH_KEYS: list[str] = [
    "vt_dtw", "vq_dtw", "ve_dtw", "vt_frec", "vq_frec", "ve_frec",
]

# excluded from model inputs
LEAK_COLS: set[str] = {
    "stat_q_abs",       # capacity_Ah와 1:1 동일 (leakage)
    "stat_energy_seg",  # q_abs × 평균전압 → 종속 변수, 정보 중복
}


# ── 원시 세그먼트 곡선 (CNN 입력) ────────────────────────────────────────────
# 세그먼트의 V/I 시계열을 q_frac [0,1] 그리드에 리샘플한 고정 길이.
# hi_correlation.py(데이터 생성)와 segment_dataset.py(로더)가 공유하는 단일 상수.
RAW_N: int = 48       # 리샘플 포인트 수
RAW_CH: int = 2       # 채널 수: [V, |I|]


# Computation cost by category (used in L0 loss)
CATEGORY_COSTS: dict[str, float] = {
    "stat":  1.0,
    "diff":  1.5,
    "lfp":   2.0,
    "morph": 3.0,
}


def _make_seg_prefix(seg: str) -> dict[str, str]:
    """stat_{key}_{seg} naming used in wide pkl format."""
    return {
        "stat":  f"stat_{{key}}_{seg}",
        "diff":  f"diff_{{key}}_{seg}",
        "lfp":   f"lfp_{{key}}_{seg}",
        "morph": f"morph_{{key}}_{seg}",
    }


def get_hi_cols_for_seg(seg: str) -> list[str]:
    """
    Returns ordered list of 64 HI column names for the wide pkl format
    for a given segment (e.g. 'dis_hi').
    stat_q_abs and stat_energy_seg are excluded.
    """
    cols: list[str] = []
    _STAT_EXCLUDE = {"q_abs", "energy_seg"}
    for key in STAT_KEYS:
        if key in _STAT_EXCLUDE:
            continue
        cols.append(f"stat_{key}_{seg}")
    for key in DIFF_KEYS:
        cols.append(f"diff_{key}_{seg}")
    for key in LFP_KEYS:
        cols.append(f"lfp_{key}_{seg}")
    for key in MORPH_KEYS:
        cols.append(f"morph_{key}_{seg}")
    return cols


def get_hi_cost_vector(seg: str) -> list[float]:
    """
    Returns cost value for each of the 64 HIs (same ordering as get_hi_cols_for_seg).
    Used to weight the L0 penalty.
    """
    _STAT_EXCLUDE = {"q_abs", "energy_seg"}
    costs: list[float] = []
    for key in STAT_KEYS:
        if key in _STAT_EXCLUDE:
            continue
        costs.append(CATEGORY_COSTS["stat"])
    for _ in DIFF_KEYS:
        costs.append(CATEGORY_COSTS["diff"])
    for _ in LFP_KEYS:
        costs.append(CATEGORY_COSTS["lfp"])
    for _ in MORPH_KEYS:
        costs.append(CATEGORY_COSTS["morph"])
    return costs


N_HI: int = len(get_hi_cols_for_seg("dis_hi"))  # == 64


def spec_from_qfrac():
    """Default qfrac ScenarioSpec (backward-compat, avoids hardcoding N_SEGS/N_LEVELS)."""
    import sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from common.scenario.qfrac import QFracSegmenter
    return QFracSegmenter().get_spec()
