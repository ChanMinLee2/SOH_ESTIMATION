"""
66-HI schema for SCR.

STAT(20) + DIFF(20) + LFP(20) + MORPH(6) = 66 per segment.

2026-08-07: stat_q_abs/stat_energy_seg를 다시 모델 입력에 포함(N_HI 64→66, 기본값).
q_frac_wide에서는 세그멘테이션 분모가 그 사이클 자신의 실제 용량이라
stat_q_abs≈n2×현재용량으로 SOH를 거의 직접 역산할 수 있어(leakage) 제외했었다.
q_frac_ref(§SOC.md)에서는 분모가 q_ref(과거 lag + 노이즈 반영 추정치)로 바뀌어
"현재 진짜 용량"과의 직결이 끊긴다 — ref_lag가 커질수록 stat_q_abs는 "현재 SOH"가
아니라 "그 시점 추정치 기준 실측값"만 반영하므로 정당한 입력 피처가 된다.
**주의**: ref_lag=0일 때는 q_ref≈현재용량(±노이즈)이라 여전히 leakage에 가깝다 —
lag를 올려가며 이 피처의 기여도가 어떻게 변하는지 같이 볼 것(lag 스윕 실험 참고).
q_frac_wide/q_abs 등 "분모=현재 실제 용량"인 축으로 되돌아가면 다시 제외를
고려해야 한다.

2026-08-08: 위 재포함을 실험별로 끌 수 있는 토글을 환경변수 SOH_EXCLUDE_STAT_LEAK=1로
추가했다(docs/260808_RESULTS.md 실험4). N_HI가 cap_heads.py 등 여러 모델 모듈에서
**모듈 임포트 시점 상수**로 nn.Linear/게이트 크기를 정하는 데 쓰이므로(N_HI=64와 66은
서로 다른 체크포인트 구조), CLI 인자(argparse, main() 실행 시점)로는 이미 임포트된 뒤라
너무 늦다 — 프로세스 시작 전 환경변수로 줘야 hi_schema 임포트 시점에 반영된다:
  SOH_EXCLUDE_STAT_LEAK=1 python 5_model/train_scr.py --phase 1 ...
학습·분류기·평가(train_scr.py/train_classifier.py/test_scr.py) 전부 N_HI를 모듈 상단에서
임포트하므로, 한 run으로 만든 체크포인트를 나중에 평가할 때도 **그때 썼던 환경변수 값을
그대로 다시 지정**해야 한다(안 그러면 N_HI 불일치로 로드가 깨지거나 조용히 다른 차원으로
동작함). `train_scr.py`가 이 값을 저장되는 config.yaml과 run 디렉터리명(`_noleak` 접미사)에
기록해 사후 추적 가능하게 한다.
"""

from __future__ import annotations

import os

# 2026-08-08: SOH_EXCLUDE_STAT_LEAK=1 이면 stat_q_abs/stat_energy_seg를 다시 제외
# (N_HI 66→64, 2026-08-07 이전 동작으로 복귀). 프로세스 시작 시 1회만 읽는다 — 위 docstring
# 참고, 런타임 중 변경 불가(모든 소비자가 모듈 임포트 시점 상수로 N_HI를 쓰기 때문).
EXCLUDE_STAT_LEAK: bool = os.environ.get("SOH_EXCLUDE_STAT_LEAK", "0") == "1"

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

# 2026-08-07: 더 이상 자동 제외하지 않음(위 모듈 docstring 참고) — 실제 제외 로직은
# get_hi_cols_for_seg/get_hi_cost_vector의 _STAT_EXCLUDE였는데 그것도 비웠다.
# 이 세트는 과거 이력 참고용으로만 남겨둔다(현재 어디서도 참조 안 함).
LEAK_COLS: set[str] = set()


# ── 원시 세그먼트 곡선 (CNN 입력) ────────────────────────────────────────────
# 세그먼트의 V/I 시계열을 q_frac [0,1] 그리드에 리샘플한 고정 길이.
# hi_correlation.py(데이터 생성)와 segment_dataset.py(로더)가 공유하는 단일 상수.
RAW_N: int = 48       # 리샘플 포인트 수
RAW_CH: int = 3       # 채널 수: [V, I(signed), t_rel] — docs/260803_RESULTS.md §10.1/§10.10


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
    Returns ordered list of 66 HI column names for the wide pkl format
    for a given segment (e.g. 'dis_hi').
    stat_q_abs/stat_energy_seg 포함(2026-08-07, 모듈 docstring 참고) — 단
    SOH_EXCLUDE_STAT_LEAK=1이면 제외(2026-08-08).
    """
    cols: list[str] = []
    _STAT_EXCLUDE: set[str] = {"q_abs", "energy_seg"} if EXCLUDE_STAT_LEAK else set()
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
    Returns cost value for each of the 66 HIs (same ordering as get_hi_cols_for_seg).
    Used to weight the L0 penalty.
    """
    _STAT_EXCLUDE: set[str] = {"q_abs", "energy_seg"} if EXCLUDE_STAT_LEAK else set()
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


N_HI: int = len(get_hi_cols_for_seg("dis_hi"))  # == 66 기본, SOH_EXCLUDE_STAT_LEAK=1이면 64


def spec_from_qfrac():
    """Default qfrac ScenarioSpec (backward-compat, avoids hardcoding N_SEGS/N_LEVELS)."""
    import sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from common.scenario.qfrac import QFracSegmenter
    return QFracSegmenter().get_spec()
