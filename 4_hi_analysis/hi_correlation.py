"""
hi_correlation.py

MIT + HUST data_postprocess 에서 HI 148종을 사이클별로 추출하고
방전 용량(capacity_Ah)과의 Spearman 상관계수를 계산·시각화.

입력: data_postprocess/MIT/*.pkl, data_postprocess/HUST/*.pkl
  (이상치 제거는 2_preprocess/preprocess.py 에서 완료된 상태를 전제함)

HI 그룹
  [방전 Global]       22종 (전압 통계·에너지·SOC별 전압·부분 적산·ICA·DVA·CE·온도)
  [방전 SoC 구간]     6종 × 3구간 = 18종  (SoC 60~100% / 30~60% / 0~30%)
  [충전 Global]        6종  (CC ratio·에너지 전압·ICA·DVA)
  [충전 SoC 구간]     6종 × 3구간 = 18종  (SoC 0~30% / 30~60% / 60~100%)

SoC 구간 → 누적 Q 비율로 정의  (방전/충전 공통)
  방전: SoC 60~100% ↔ q_frac  0~40%  (방전 초반, 고전압)
        SoC 30~60%  ↔ q_frac 40~70%  (플래토 중심)
        SoC 0~30%   ↔ q_frac 70~100% (방전 후반, 저전압)
  충전: SoC 0~30%   ↔ q_frac  0~40%  (충전 초반, 저전압)
        SoC 30~60%  ↔ q_frac 40~70%  (플래토 중심)
        SoC 60~100% ↔ q_frac 70~100% (충전 후반, CV 포함)

출력 : hi_correlation.png
       data_HI/MIT/{cell_id}.pkl   (셀별 HI 특성, 사이클×64 HI)
       data_HI/HUST/{cell_id}.pkl
사용 : python hi_correlation.py [--workers N] [--n-top N] [--force]
"""

import argparse
import os
import pickle
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew
from scipy.stats import spearmanr
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP_DIR     = Path(__file__).resolve().parent
MIT_DIR      = PROJECT_ROOT / "data_postprocess" / "MIT"
HUST_DIR     = PROJECT_ROOT / "data_postprocess" / "HUST"
CACHE_PATH   = STEP_DIR / "hi_features.pkl"
HI_ROOT      = PROJECT_ROOT / "data_HI"

# 방전 SoC 세그먼트 (q_frac_lo, q_frac_hi, 내부키, 표시명)
SEG_DEFS = [
    (0.0, 0.4, "s_hi",  "SoC 60~100%\n(초반·고전압)"),
    (0.4, 0.7, "s_mid", "SoC 30~60%\n(플래토 중심)"),
    (0.7, 1.0, "s_lo",  "SoC 0~30%\n(후반·저전압)"),
]

# 충전 SoC 세그먼트 (충전은 SoC가 낮→높, q_frac 0~40% = SoC 0~30%)
CHG_SEG_DEFS = [
    (0.0, 0.4, "chg_s_lo",  "Chg SoC 0~30%\n(초반·저전압)"),
    (0.4, 0.7, "chg_s_mid", "Chg SoC 30~60%\n(플래토 중심)"),
    (0.7, 1.0, "chg_s_hi",  "Chg SoC 60~100%\n(후반·CV 포함)"),
]

# ── HI 메타 (내부 키, 짧은 레이블) ──────────────────────────────────────────
HI_META = [
    # 방전 Global ─────────────────────────────────────────────────────────
    ("v_mean",           "V mean"),
    ("v_std",            "V std"),
    ("v_skew",           "V skew"),
    ("v_kurt",           "V kurt"),
    ("v_end",            "V end"),
    ("v_drop",           "V drop"),
    ("energy_Wh",        "Energy[Wh]"),
    ("v_energy",         "V energy"),
    ("v_at_q20",         "V@Q20%"),
    ("v_at_q50",         "V@Q50%"),
    ("v_at_q80",         "V@Q80%"),
    ("q_high_v",         "Q>3.2V"),
    ("q_tail",           "Q<3.2V"),
    ("q_plateau_ratio",  "Q_plt ratio"),
    ("t_discharge",      "t_dis[s]"),
    ("ica_peak_h",       "ICA h"),
    ("ica_peak_v",       "ICA V"),
    ("ica_peak_area",    "ICA area"),
    ("dvdq_min",         "DVA min"),
    ("ce",               "CE"),
    ("temp_mean",        "T mean"),
    ("temp_max",         "T max"),
    # 방전 SoC 60~100% (q_frac 0~40%) ────────────────────────────────────
    ("v_mean_s_hi",           "V mean"),
    ("v_std_s_hi",            "V std"),
    ("energy_Wh_s_hi",        "Energy[Wh]"),
    ("q_abs_s_hi",            "Q[Ah]"),
    ("ica_peak_h_s_hi",       "ICA h"),
    ("dvdq_min_s_hi",         "DVA min"),
    ("ent_v_s_hi",            "ent V"),
    ("corr_vi_s_hi",          "r(V,I)"),
    ("power_var_s_hi",        "P var"),
    ("mean_dvdt_s_hi",        "dV/dt μ"),
    ("var_dvdt_s_hi",         "dV/dt σ²"),
    ("v_retention_s_hi",      "V ret."),
    ("v_total_var_s_hi",      "ΣΔV"),
    ("de_dq_s_hi",            "dE/dQ"),
    ("ocv_slope_s_hi",        "OCV slp"),
    ("dvdq_p5_s_hi",          "DVA p5"),
    ("nonlinear_idx_s_hi",    "NL idx"),
    ("temp_rise_per_ah_s_hi", "ΔT/Q"),
    ("dtdt_s_hi",             "dT/dt"),
    ("corr_vt_s_hi",          "r(V,T)"),
    # 방전 SoC 30~60% (q_frac 40~70%) ────────────────────────────────────
    ("v_mean_s_mid",           "V mean"),
    ("v_std_s_mid",            "V std"),
    ("energy_Wh_s_mid",        "Energy[Wh]"),
    ("q_abs_s_mid",            "Q[Ah]"),
    ("ica_peak_h_s_mid",       "ICA h"),
    ("dvdq_min_s_mid",         "DVA min"),
    ("ent_v_s_mid",            "ent V"),
    ("corr_vi_s_mid",          "r(V,I)"),
    ("power_var_s_mid",        "P var"),
    ("mean_dvdt_s_mid",        "dV/dt μ"),
    ("var_dvdt_s_mid",         "dV/dt σ²"),
    ("v_retention_s_mid",      "V ret."),
    ("v_total_var_s_mid",      "ΣΔV"),
    ("de_dq_s_mid",            "dE/dQ"),
    ("ocv_slope_s_mid",        "OCV slp"),
    ("dvdq_p5_s_mid",          "DVA p5"),
    ("nonlinear_idx_s_mid",    "NL idx"),
    ("temp_rise_per_ah_s_mid", "ΔT/Q"),
    ("dtdt_s_mid",             "dT/dt"),
    ("corr_vt_s_mid",          "r(V,T)"),
    # 방전 SoC 0~30% (q_frac 70~100%) ────────────────────────────────────
    ("v_mean_s_lo",            "V mean"),
    ("v_std_s_lo",             "V std"),
    ("energy_Wh_s_lo",         "Energy[Wh]"),
    ("q_abs_s_lo",             "Q[Ah]"),
    ("ica_peak_h_s_lo",        "ICA h"),
    ("dvdq_min_s_lo",          "DVA min"),
    ("ent_v_s_lo",             "ent V"),
    ("corr_vi_s_lo",           "r(V,I)"),
    ("power_var_s_lo",         "P var"),
    ("mean_dvdt_s_lo",         "dV/dt μ"),
    ("var_dvdt_s_lo",          "dV/dt σ²"),
    ("v_retention_s_lo",       "V ret."),
    ("v_total_var_s_lo",       "ΣΔV"),
    ("de_dq_s_lo",             "dE/dQ"),
    ("ocv_slope_s_lo",         "OCV slp"),
    ("dvdq_p5_s_lo",           "DVA p5"),
    ("nonlinear_idx_s_lo",     "NL idx"),
    ("temp_rise_per_ah_s_lo",  "ΔT/Q"),
    ("dtdt_s_lo",              "dT/dt"),
    ("corr_vt_s_lo",           "r(V,T)"),
    # 충전 Global ─────────────────────────────────────────────────────────
    ("q_cc_ratio",        "CC ratio"),
    ("chg_v_energy",      "V energy"),
    ("chg_ica_peak_h",    "ICA h"),
    ("chg_ica_peak_v",    "ICA V"),
    ("chg_ica_peak_area", "ICA area"),
    ("chg_dvdq_min",      "DVA min"),
    # 충전 SoC 0~30% (q_frac 0~40%, 초반·저전압) ─────────────────────────
    ("v_mean_chg_s_lo",           "V mean"),
    ("v_std_chg_s_lo",            "V std"),
    ("energy_Wh_chg_s_lo",        "Energy[Wh]"),
    ("q_abs_chg_s_lo",            "Q[Ah]"),
    ("ica_peak_h_chg_s_lo",       "ICA h"),
    ("dvdq_min_chg_s_lo",         "DVA min"),
    ("ent_v_chg_s_lo",            "ent V"),
    ("corr_vi_chg_s_lo",          "r(V,I)"),
    ("power_var_chg_s_lo",        "P var"),
    ("mean_dvdt_chg_s_lo",        "dV/dt μ"),
    ("var_dvdt_chg_s_lo",         "dV/dt σ²"),
    ("v_retention_chg_s_lo",      "V ret."),
    ("v_total_var_chg_s_lo",      "ΣΔV"),
    ("de_dq_chg_s_lo",            "dE/dQ"),
    ("ocv_slope_chg_s_lo",        "OCV slp"),
    ("dvdq_p5_chg_s_lo",          "DVA p5"),
    ("nonlinear_idx_chg_s_lo",    "NL idx"),
    ("temp_rise_per_ah_chg_s_lo", "ΔT/Q"),
    ("dtdt_chg_s_lo",             "dT/dt"),
    ("corr_vt_chg_s_lo",          "r(V,T)"),
    # 충전 SoC 30~60% (q_frac 40~70%, 플래토) ────────────────────────────
    ("v_mean_chg_s_mid",           "V mean"),
    ("v_std_chg_s_mid",            "V std"),
    ("energy_Wh_chg_s_mid",        "Energy[Wh]"),
    ("q_abs_chg_s_mid",            "Q[Ah]"),
    ("ica_peak_h_chg_s_mid",       "ICA h"),
    ("dvdq_min_chg_s_mid",         "DVA min"),
    ("ent_v_chg_s_mid",            "ent V"),
    ("corr_vi_chg_s_mid",          "r(V,I)"),
    ("power_var_chg_s_mid",        "P var"),
    ("mean_dvdt_chg_s_mid",        "dV/dt μ"),
    ("var_dvdt_chg_s_mid",         "dV/dt σ²"),
    ("v_retention_chg_s_mid",      "V ret."),
    ("v_total_var_chg_s_mid",      "ΣΔV"),
    ("de_dq_chg_s_mid",            "dE/dQ"),
    ("ocv_slope_chg_s_mid",        "OCV slp"),
    ("dvdq_p5_chg_s_mid",          "DVA p5"),
    ("nonlinear_idx_chg_s_mid",    "NL idx"),
    ("temp_rise_per_ah_chg_s_mid", "ΔT/Q"),
    ("dtdt_chg_s_mid",             "dT/dt"),
    ("corr_vt_chg_s_mid",          "r(V,T)"),
    # 충전 SoC 60~100% (q_frac 70~100%, 후반·CV 포함) ────────────────────
    ("v_mean_chg_s_hi",           "V mean"),
    ("v_std_chg_s_hi",            "V std"),
    ("energy_Wh_chg_s_hi",        "Energy[Wh]"),
    ("q_abs_chg_s_hi",            "Q[Ah]"),
    ("ica_peak_h_chg_s_hi",       "ICA h"),
    ("dvdq_min_chg_s_hi",         "DVA min"),
    ("ent_v_chg_s_hi",            "ent V"),
    ("corr_vi_chg_s_hi",          "r(V,I)"),
    ("power_var_chg_s_hi",        "P var"),
    ("mean_dvdt_chg_s_hi",        "dV/dt μ"),
    ("var_dvdt_chg_s_hi",         "dV/dt σ²"),
    ("v_retention_chg_s_hi",      "V ret."),
    ("v_total_var_chg_s_hi",      "ΣΔV"),
    ("de_dq_chg_s_hi",            "dE/dQ"),
    ("ocv_slope_chg_s_hi",        "OCV slp"),
    ("dvdq_p5_chg_s_hi",          "DVA p5"),
    ("nonlinear_idx_chg_s_hi",    "NL idx"),
    ("temp_rise_per_ah_chg_s_hi", "ΔT/Q"),
    ("dtdt_chg_s_hi",             "dT/dt"),
    ("corr_vt_chg_s_hi",          "r(V,T)"),
]

HI_KEYS   = [k for k, _ in HI_META]
HI_LABELS = {k: lbl for k, lbl in HI_META}

# 세그먼트별로 추가되는 신규 HI 14종 (방전·충전 공통)
NEW_SEG_KEYS = [
    "ent_v",            # 전압 히스토그램 엔트로피
    "corr_vi",          # V-|I| Pearson r
    "power_var",        # var(V · |I|)
    "mean_dvdt",        # mean(dV/dt)
    "var_dvdt",         # var(dV/dt)
    "v_retention",      # V_min / V_max
    "v_total_var",      # Σ|ΔV| (전압 총 변동)
    "de_dq",            # Σ(V·|I|·dt) / Σ(|I|·dt)  (에너지 가중 전압)
    "ocv_slope",        # polyfit(Q_rel, V, 1)[0]  (선형 V-Q 기울기)
    "dvdq_p5",          # 5th-percentile |dV/dQ| (binned curve)
    "nonlinear_idx",    # cubic poly V-Q 3차 계수
    "temp_rise_per_ah", # (T_last − T_first) / Q_Ah
    "dtdt",             # mean(ΔT/Δt)
    "corr_vt",          # V-T Pearson r
]

# 시각화 그룹  ── 슬라이스: 22 + 20×3 + 6 + 20×3 = 148 HIs
HI_GROUPS = OrderedDict([
    ("Discharge — Global",   [k for k, _ in HI_META[:22]]),
    ("Dis. SoC 60~100%",     [k for k, _ in HI_META[22:42]]),
    ("Dis. SoC 30~60%",      [k for k, _ in HI_META[42:62]]),
    ("Dis. SoC 0~30%",       [k for k, _ in HI_META[62:82]]),
    ("Charge — Global",      [k for k, _ in HI_META[82:88]]),
    ("Chg. SoC 0~30%",       [k for k, _ in HI_META[88:108]]),
    ("Chg. SoC 30~60%",      [k for k, _ in HI_META[108:128]]),
    ("Chg. SoC 60~100%",     [k for k, _ in HI_META[128:]]),
])

# HI 키 → 산점도 제목용 단축 그룹 태그
_GROUP_SHORT = {
    "Discharge — Global": "Dis·Global",
    "Dis. SoC 60~100%":   "Dis·SoC 60~100%",
    "Dis. SoC 30~60%":    "Dis·SoC 30~60%",
    "Dis. SoC 0~30%":     "Dis·SoC 0~30%",
    "Charge — Global":    "Chg·Global",
    "Chg. SoC 0~30%":     "Chg·SoC 0~30%",
    "Chg. SoC 30~60%":    "Chg·SoC 30~60%",
    "Chg. SoC 60~100%":   "Chg·SoC 60~100%",
}
HI_GROUP_TAG = {
    k: _GROUP_SHORT[gname]
    for gname, keys in HI_GROUPS.items()
    for k in keys
}


# ─────────────────────────────────────────────────────────────────────────────
# 세그먼트 ICA/DVA 헬퍼 (top-level 필요 없음 — _extract_one_cell 내부에서만 사용)
# ─────────────────────────────────────────────────────────────────────────────

def _seg_ica_peak(v_arr, i_arr, dt_arr):
    """세그먼트 전압 범위 내 dQ/dV 최댓값 반환."""
    vr = v_arr.max() - v_arr.min()
    if vr < 0.02 or len(v_arr) < 15:
        return np.nan
    v_lo = float(v_arr.min()) - 0.005
    v_hi = float(v_arr.max()) + 0.005
    n_b  = max(8, min(30, int(vr / 0.01)))
    edges = np.linspace(v_lo, v_hi, n_b + 1)
    dv    = edges[1] - edges[0]
    dqdv  = np.zeros(n_b)
    for j in range(n_b):
        m = (v_arr >= edges[j]) & (v_arr < edges[j + 1])
        if m.sum() > 0:
            dqdv[j] = np.sum(i_arr[m] * dt_arr[m]) / 3600.0 / dv
    ws = min(7, n_b - (1 - n_b % 2))
    ws = max(3, ws if ws % 2 == 1 else ws - 1)
    if n_b < ws:
        return float(np.max(dqdv))
    try:
        return float(np.max(savgol_filter(dqdv, ws, min(2, ws - 1))))
    except Exception:
        return float(np.max(dqdv))


def _seg_dvdq_min(v_arr, i_arr, dt_arr, q_cum_arr):
    """세그먼트 내 dV/dQ 최솟값 반환."""
    q_rel  = q_cum_arr - q_cum_arr[0]
    q_tot  = float(q_rel[-1]) if len(q_rel) > 1 else 0.0
    if q_tot < 0.005 or len(q_rel) < 10:
        return np.nan
    dq_b  = max(q_tot / 25.0, 0.002)
    q_e   = np.arange(0.0, q_tot + dq_b, dq_b)
    n_se  = len(q_e) - 1
    v_av  = np.full(n_se, np.nan)
    for j in range(n_se):
        m = (q_rel >= q_e[j]) & (q_rel < q_e[j + 1])
        if m.sum() > 0:
            v_av[j] = float(np.mean(v_arr[m]))
    vld = np.isfinite(v_av)
    if vld.sum() < 4:
        return np.nan
    qm  = (q_e[:-1] + q_e[1:]) / 2
    vf  = np.interp(qm, qm[vld], v_av[vld])
    wd  = min(5, n_se - (1 - n_se % 2))
    wd  = max(3, wd if wd % 2 == 1 else wd - 1)
    if n_se < wd:
        return float(np.min(np.gradient(vf, dq_b)))
    try:
        return float(np.min(np.gradient(savgol_filter(vf, wd, min(2, wd - 1)), dq_b)))
    except Exception:
        return np.nan


def _seg_extra_his(vs: np.ndarray, ims: np.ndarray, dts: np.ndarray,
                   qcs: np.ndarray, tmps: np.ndarray, seg: str) -> dict:
    """세그먼트별 신규 HI 14종 계산 (방전·충전 공통).

    vs   : 전압 [V]
    ims  : |전류| [A]
    dts  : Δt [s] (각 포인트의 이전 포인트와의 시간 차)
    qcs  : 누적 방전/충전 Q [Ah]
    tmps : 온도 [°C] (NaN 허용)
    seg  : 세그먼트 접미어 (e.g. 's_hi', 'chg_s_lo')
    """
    out = {f"{k}_{seg}": np.nan for k in NEW_SEG_KEYS}
    n = len(vs)
    if n < 10:
        return out

    # ── 전압 엔트로피 ────────────────────────────────────────────────────
    # density=True는 PDF(합≠1)를 반환하므로 entropy 계산에 부적합;
    # CV 구간처럼 전압 변화폭이 좁으면 밀도값>>1 → log(p)>0 → 거대 음수 발생.
    _counts = np.histogram(vs, bins=10)[0].astype(float)
    _total  = _counts.sum()
    if _total > 0:
        p = _counts[_counts > 0] / _total      # 확률질량; 합 = 1
        out[f"ent_v_{seg}"] = float(-np.sum(p * np.log(p)))

    # ── V-|I| 상관 ────────────────────────────────────────────────────────
    if np.std(vs) > 1e-6 and np.std(ims) > 1e-6:
        out[f"corr_vi_{seg}"] = float(np.corrcoef(vs, ims)[0, 1])

    # ── 전력 분산 ─────────────────────────────────────────────────────────
    out[f"power_var_{seg}"] = float(np.var(vs * ims))

    # ── dV/dt 통계 ────────────────────────────────────────────────────────
    # mean_dvdt: 세그먼트 전체 기울기 (V_end-V_start)/total_time.
    #   per-step 평균 대신 총 기울기를 사용하는 이유:
    #   MIT 로거는 전환 구간(~0.01s)과 플래토(~4s)를 혼합 기록하며,
    #   dt 기준 필터 적용 시 선택된 구간에 따라 값이 크게 왜곡됨.
    # var_dvdt: dt >= 1s인 행(플래토 구간)의 분산만 측정.
    #   7e-12s 수준 FP 오차 행과 전환 구간 transient를 모두 제외.
    dt_total = float(np.sum(dts))
    if n >= 2 and dt_total >= 1.0:
        out[f"mean_dvdt_{seg}"] = float(vs[-1] - vs[0]) / dt_total

    if n > 2:
        dt_seg = dts[1:]
        dv_seg = np.diff(vs)
        valid  = dt_seg >= 1.0
        if valid.sum() >= 2:
            dvdt = dv_seg[valid] / dt_seg[valid]
            out[f"var_dvdt_{seg}"] = float(np.var(dvdt))

    # ── V_min / V_max 유지율 ─────────────────────────────────────────────
    v_max = float(vs.max())
    if v_max > 1e-3:
        out[f"v_retention_{seg}"] = float(vs.min()) / v_max

    # ── 전압 총 변동 ─────────────────────────────────────────────────────
    # dt >= 1s 행(플래토)만 사용: 빠른 로깅 행의 측정 노이즈 누적 방지.
    slow_mask = dts >= 1.0
    vs_slow   = vs[slow_mask]
    if len(vs_slow) > 1:
        out[f"v_total_var_{seg}"] = float(np.sum(np.abs(np.diff(vs_slow))))

    # ── 에너지 가중 전압 ─────────────────────────────────────────────────
    denom = float(np.sum(ims * dts))
    if denom > 1e-9:
        out[f"de_dq_{seg}"] = float(np.sum(vs * ims * dts)) / denom

    # ── Q 상대 좌표 ──────────────────────────────────────────────────────
    q_rel = qcs - qcs[0]           # Ah (상대)
    q_tot = float(q_rel[-1]) if len(q_rel) > 0 else 0.0

    # ── 선형 V-Q 기울기 ──────────────────────────────────────────────────
    if q_tot > 0.005 and n > 3:
        try:
            out[f"ocv_slope_{seg}"] = float(np.polyfit(q_rel, vs, 1)[0])
        except Exception:
            pass

    # ── 5th-percentile |dV/dQ| (binned) ─────────────────────────────────
    if q_tot > 0.005 and n > 10:
        dq_b = max(q_tot / 25.0, 0.002)
        q_e  = np.arange(0.0, q_tot + dq_b, dq_b)
        n_se = len(q_e) - 1
        if n_se >= 4:
            v_av = np.full(n_se, np.nan)
            for j in range(n_se):
                m = (q_rel >= q_e[j]) & (q_rel < q_e[j + 1])
                if m.sum() > 0:
                    v_av[j] = float(np.mean(vs[m]))
            vld = np.isfinite(v_av)
            if vld.sum() >= 4:
                qm = (q_e[:-1] + q_e[1:]) / 2
                vf = np.interp(qm, qm[vld], v_av[vld])
                dv_dq_arr = np.gradient(vf, dq_b)
                out[f"dvdq_p5_{seg}"] = float(
                    np.percentile(np.abs(dv_dq_arr), 5))

    # ── 비선형 V-Q 지수 (3차 다항식 최고차 계수) ─────────────────────────
    if q_tot > 0.01 and n > 4:
        try:
            q_norm = q_rel / q_tot
            out[f"nonlinear_idx_{seg}"] = float(np.polyfit(q_norm, vs, 3)[0])
        except Exception:
            pass

    # ── 온도 기반 특성 ───────────────────────────────────────────────────
    fin = np.isfinite(tmps)
    if fin.sum() > 2:
        first_i = int(np.where(fin)[0][0])
        last_i  = int(np.where(fin)[0][-1])

        # ΔT / Q — q_tot > 0.05 Ah 조건으로 강화 (0.005는 너무 느슨 → ±35°C/Ah 이상치)
        if q_tot > 0.05:
            out[f"temp_rise_per_ah_{seg}"] = (
                float(tmps[last_i] - tmps[first_i]) / q_tot
            )

        # mean dT/dt — 총 기울기 방식: np.maximum(dt,1e-6) 버그 + 온도 양자화 노이즈 제거
        # 이전: np.diff(T) / max(dt, 1e-6) → p50=35°C/s (불가능한 값)
        if dt_total >= 1.0:
            out[f"dtdt_{seg}"] = float(tmps[last_i] - tmps[first_i]) / dt_total

        # V-T 상관 — 온도 std > 50mK 조건 추가 (노이즈 지배 구간 제외)
        if fin.sum() > 3:
            vs_ft   = vs[fin]
            tmps_ft = tmps[fin]
            if np.std(vs_ft) > 1e-6 and np.std(tmps_ft) > 0.05:
                out[f"corr_vt_{seg}"] = float(
                    np.corrcoef(vs_ft, tmps_ft)[0, 1])

    return out


# ─────────────────────────────────────────────────────────────────────────────
# HI 추출 (top-level — multiprocessing 호환)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_one_cell(pkl_path_str: str) -> list:
    path = Path(pkl_path_str)
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except Exception:
        return []

    meta   = raw.get("meta", {})
    df_all = raw.get("cycles")
    if df_all is None or not isinstance(df_all, pd.DataFrame):
        return []

    dataset = meta.get("dataset", "")
    cell_id = meta.get("cell_id", path.stem)
    records = []

    for cyc, grp in df_all.groupby("cycle"):
        if int(cyc) == 0:
            continue
        dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
        if len(dis) < 30:
            continue

        cap = float(dis["capacity_Ah"].iloc[0])
        if not np.isfinite(cap) or cap < 0.05:
            continue

        v   = dis["voltage_V"].values.astype(float)
        i   = dis["current_A"].values.astype(float)
        t   = dis["time_s"].values.astype(float)
        tmp = dis["temperature_C"].values.astype(float)

        dt    = np.clip(np.diff(t, prepend=t[0]), 0, None)
        i_mag = np.abs(i)


        # ── 전압 통계 ─────────────────────────────────────────────────────
        v_mean = float(np.mean(v))
        v_std  = float(np.std(v))
        v_skew = float(sp_skew(v))
        v_kurt = float(sp_kurtosis(v))
        v_end  = float(v[-1])
        n5     = max(1, len(v) // 20)
        v_drop = float(v[0] - np.mean(v[:n5]))

        # ── 방전 시간 ─────────────────────────────────────────────────────
        t_dis = float(t[-1] - t[0]) if len(t) > 1 else np.nan

        # ── 에너지 기반 ───────────────────────────────────────────────────
        energy_Wh = float(np.sum(v * i_mag * dt) / 3600.0)
        v_energy  = energy_Wh / cap if cap > 0.01 else np.nan

        # ── 누적 방전 Q (SOC 프록시) ──────────────────────────────────────
        q_cum   = np.cumsum(i_mag * dt) / 3600.0
        q_local = float(q_cum[-1]) if len(q_cum) > 0 else 0.0

        # 실제 방전량이 등록 용량의 30% 미만 → 셀 완전방전 상태에서 기록된 아티팩트
        # (MIT b2c27 cyc119: q_local=0.024Ah vs cap=1.067Ah, ratio=0.023)
        if q_local < cap * 0.30:
            continue

        # V at Q=20%, 50%, 80%
        v_at_q20 = v_at_q50 = v_at_q80 = np.nan
        if q_local > 0.05:
            for frac, slot in [(0.20, 0), (0.50, 1), (0.80, 2)]:
                idx = min(int(np.searchsorted(q_cum, frac * q_local)), len(v) - 1)
                if   slot == 0: v_at_q20 = float(v[idx])
                elif slot == 1: v_at_q50 = float(v[idx])
                else:           v_at_q80 = float(v[idx])

        # ── 전압 구간별 부분 적산 ─────────────────────────────────────────
        mask_hv  = v > 3.20
        mask_tail = v < 3.20
        mask_plt  = (v >= 3.20) & (v <= 3.45)
        q_high_v       = float(np.sum(i_mag[mask_hv]   * dt[mask_hv])   / 3600.0)
        q_tail         = float(np.sum(i_mag[mask_tail]  * dt[mask_tail]) / 3600.0)
        q_plt_Ah       = float(np.sum(i_mag[mask_plt]   * dt[mask_plt])  / 3600.0)
        q_plateau_ratio = q_plt_Ah / cap if cap > 0.01 else np.nan

        # ── 온도 ──────────────────────────────────────────────────────────
        fin    = np.isfinite(tmp)
        t_mean = float(np.mean(tmp[fin])) if fin.any() else np.nan
        t_max  = float(np.max(tmp[fin]))  if fin.any() else np.nan

        # ── ICA (dQ/dV, 전체 방전) ────────────────────────────────────────
        v_lo, v_hi, n_bins = 2.8, 3.65, 60
        edges  = np.linspace(v_lo, v_hi, n_bins + 1)
        dv     = edges[1] - edges[0]
        vmids  = (edges[:-1] + edges[1:]) / 2
        dqdv   = np.zeros(n_bins)
        for j in range(n_bins):
            m = (v >= edges[j]) & (v < edges[j + 1])
            if m.sum() > 0:
                dqdv[j] = np.sum(i_mag[m] * dt[m]) / 3600.0 / dv
        win = 9 if n_bins >= 9 else max(3, n_bins - (1 - n_bins % 2))
        try:
            dqdv_s = savgol_filter(dqdv, win, 3)
        except Exception:
            dqdv_s = dqdv
        lfp_m = (vmids >= 3.2) & (vmids <= 3.5)
        if lfp_m.sum() > 0:
            sub      = dqdv_s[lfp_m]; subv = vmids[lfp_m]
            pk       = np.argmax(sub)
            ica_h    = float(sub[pk])
            ica_v    = float(subv[pk])
            ica_area = float(np.trapz(np.maximum(sub, 0), subv))
        else:
            ica_h = ica_v = ica_area = np.nan

        # ── DVA (dV/dQ, 전체 방전) ────────────────────────────────────────
        dvdq_min = np.nan
        if q_local > 0.10 and len(q_cum) > 20:
            dq_bin  = max(q_local / 50.0, 0.005)
            q_edges = np.arange(0.0, q_local + dq_bin, dq_bin)
            n_seg   = len(q_edges) - 1
            v_avg   = np.full(n_seg, np.nan)
            for j in range(n_seg):
                m = (q_cum >= q_edges[j]) & (q_cum < q_edges[j + 1])
                if m.sum() > 0:
                    v_avg[j] = float(np.mean(v[m]))
            valid = np.isfinite(v_avg)
            if valid.sum() > 5:
                qm     = (q_edges[:-1] + q_edges[1:]) / 2
                v_fill = np.interp(qm, qm[valid], v_avg[valid])
                wd     = min(9, n_seg - (1 - n_seg % 2))
                wd     = max(3, wd if wd % 2 == 1 else wd - 1)
                if n_seg >= wd:
                    v_s    = savgol_filter(v_fill, wd, 2)
                    dvdqa  = np.gradient(v_s, dq_bin)
                    plt_m  = (v_fill >= 3.15) & (v_fill <= 3.50)
                    if plt_m.sum() > 0:
                        dvdq_min = float(np.min(dvdqa[plt_m]))

        # ── 쿨롱 효율 ─────────────────────────────────────────────────────
        ce = np.nan
        chg_grp = grp[grp["phase"] == "charge"].sort_values("time_s")
        if len(chg_grp) > 10:
            tc_  = chg_grp["time_s"].values.astype(float)
            ic_  = np.abs(chg_grp["current_A"].values.astype(float))
            dtc_ = np.clip(np.diff(tc_, prepend=tc_[0]), 0, None)
            qc_  = float(np.sum(ic_ * dtc_) / 3600.0)
            if qc_ > 0.05:
                ce = cap / qc_

        # ── SoC 세그먼트별 HI ─────────────────────────────────────────────
        seg_rec = {}
        for q_lo_f, q_hi_f, seg, _ in SEG_DEFS:
            defaults = {f"v_mean_{seg}": np.nan, f"v_std_{seg}": np.nan,
                        f"energy_Wh_{seg}": np.nan, f"q_abs_{seg}": np.nan,
                        f"ica_peak_h_{seg}": np.nan, f"dvdq_min_{seg}": np.nan,
                        **{f"{k}_{seg}": np.nan for k in NEW_SEG_KEYS}}
            if q_local < 0.05:
                seg_rec.update(defaults); continue

            lo   = q_lo_f * q_local
            hi   = q_hi_f * q_local
            m_s  = (q_cum >= lo) & (q_cum < hi)
            if m_s.sum() < 10:
                seg_rec.update(defaults); continue

            vs   = v[m_s];   ims = i_mag[m_s];  dts = dt[m_s]
            qcs  = q_cum[m_s]; tmps = tmp[m_s]

            seg_rec[f"v_mean_{seg}"]    = float(np.mean(vs))
            seg_rec[f"v_std_{seg}"]     = float(np.std(vs))
            seg_rec[f"energy_Wh_{seg}"] = float(np.sum(vs * ims * dts) / 3600.0)
            seg_rec[f"q_abs_{seg}"]     = float(np.sum(ims * dts) / 3600.0)
            seg_rec[f"ica_peak_h_{seg}"] = _seg_ica_peak(vs, ims, dts)
            seg_rec[f"dvdq_min_{seg}"]   = _seg_dvdq_min(vs, ims, dts, qcs)
            seg_rec.update(_seg_extra_his(vs, ims, dts, qcs, tmps, seg))

        # ── 충전 HI ───────────────────────────────────────────────────────
        _chg_seg_keys = []
        for _, _, _seg, _ in CHG_SEG_DEFS:
            _chg_seg_keys += [
                f"v_mean_{_seg}", f"v_std_{_seg}", f"energy_Wh_{_seg}",
                f"q_abs_{_seg}",  f"ica_peak_h_{_seg}", f"dvdq_min_{_seg}",
            ] + [f"{k}_{_seg}" for k in NEW_SEG_KEYS]
        chg_rec = {k: np.nan for k in [
            "q_cc_ratio", "chg_v_energy",
            "chg_ica_peak_h", "chg_ica_peak_v", "chg_ica_peak_area",
            "chg_dvdq_min",
        ] + _chg_seg_keys}

        if len(chg_grp) >= 20:
            tc   = chg_grp["time_s"].values.astype(float)
            vc   = chg_grp["voltage_V"].values.astype(float)
            ic   = np.abs(chg_grp["current_A"].values.astype(float))
            tmpc = chg_grp["temperature_C"].values.astype(float)
            dtc  = np.clip(np.diff(tc, prepend=tc[0]), 0, None)
            q_tc = float(np.sum(ic * dtc) / 3600.0)

            # 불완전 충전 사이클 제외 (총 충전량 < 방전 용량의 60%)
            _chg_incomplete = q_tc < cap * 0.60

            # CC 프로토콜 전환 갭 플래그 — preprocess.py 가 chg_gap_seg 컬럼으로 기록
            # True 이면 세그먼트 HI 스킵, 전역 HI는 계산
            _chg_gap_seg = bool(chg_grp["chg_gap_seg"].any()) \
                if "chg_gap_seg" in chg_grp.columns else False

            if q_tc > 0.05 and not _chg_incomplete:
                # CC ratio: |I| ≥ 80% of max → CC 구간
                i_mx = float(np.max(ic))
                if i_mx > 0:
                    cc_m = ic >= 0.80 * i_mx
                    chg_rec["q_cc_ratio"] = (
                        float(np.sum(ic[cc_m] * dtc[cc_m]) / 3600.0) / q_tc
                    )

                # 에너지 가중 전압
                ec = float(np.sum(vc * ic * dtc) / 3600.0)
                chg_rec["chg_v_energy"] = ec / q_tc

                # 충전 ICA
                n_bc    = 60
                edges_c = np.linspace(2.8, 3.65, n_bc + 1)
                dv_c    = edges_c[1] - edges_c[0]
                vmids_c = (edges_c[:-1] + edges_c[1:]) / 2
                dqdvc   = np.zeros(n_bc)
                for j in range(n_bc):
                    m = (vc >= edges_c[j]) & (vc < edges_c[j + 1])
                    if m.sum() > 0:
                        dqdvc[j] = np.sum(ic[m] * dtc[m]) / 3600.0 / dv_c
                try:
                    dqdvcs = savgol_filter(dqdvc, 9, 3)
                except Exception:
                    dqdvcs = dqdvc
                lfp_mc = (vmids_c >= 3.2) & (vmids_c <= 3.5)
                if lfp_mc.sum() > 0:
                    sub_c = dqdvcs[lfp_mc]; subv_c = vmids_c[lfp_mc]
                    pk_c  = np.argmax(sub_c)
                    chg_rec["chg_ica_peak_h"]    = float(sub_c[pk_c])
                    chg_rec["chg_ica_peak_v"]    = float(subv_c[pk_c])
                    chg_rec["chg_ica_peak_area"] = float(
                        np.trapz(np.maximum(sub_c, 0), subv_c))

                # 충전 DVA
                qcc  = np.cumsum(ic * dtc) / 3600.0
                qtc2 = float(qcc[-1])
                if qtc2 > 0.1:
                    dqbc  = max(qtc2 / 50.0, 0.005)
                    qec   = np.arange(0.0, qtc2 + dqbc, dqbc)
                    nsc   = len(qec) - 1
                    vavc  = np.full(nsc, np.nan)
                    for j in range(nsc):
                        m = (qcc >= qec[j]) & (qcc < qec[j + 1])
                        if m.sum() > 0:
                            vavc[j] = float(np.mean(vc[m]))
                    vldc = np.isfinite(vavc)
                    if vldc.sum() > 5:
                        qmc   = (qec[:-1] + qec[1:]) / 2
                        vfc   = np.interp(qmc, qmc[vldc], vavc[vldc])
                        wdc   = min(9, nsc - (1 - nsc % 2))
                        wdc   = max(3, wdc if wdc % 2 == 1 else wdc - 1)
                        if nsc >= wdc:
                            vsc   = savgol_filter(vfc, wdc, 2)
                            dvdqc = np.gradient(vsc, dqbc)
                            plmc  = (vfc >= 3.15) & (vfc <= 3.50)
                            if plmc.sum() > 0:
                                chg_rec["chg_dvdq_min"] = float(
                                    np.min(dvdqc[plmc]))

                # ── 충전 SoC 세그먼트 ─────────────────────────────────────
                # CC 전환 갭(_chg_gap_seg)이면 q_frac 축이 어긋나므로 세그먼트 HI 스킵.
                # 전역 HI(위)는 갭 유무와 무관하게 계산됨.
                if not _chg_gap_seg:
                    qcc_local = float(qcc[-1]) if len(qcc) > 0 else 0.0
                    for q_lo_f, q_hi_f, seg, _ in CHG_SEG_DEFS:
                        if qcc_local < 0.05:
                            break
                        lo_s = q_lo_f * qcc_local
                        hi_s = q_hi_f * qcc_local
                        m_s  = (qcc >= lo_s) & (qcc < hi_s)
                        if m_s.sum() < 10:
                            continue
                        vs_c   = vc[m_s];  ims_c  = ic[m_s];   dts_c  = dtc[m_s]
                        qcs_c  = qcc[m_s]; tmps_c = tmpc[m_s]
                        chg_rec[f"v_mean_{seg}"]     = float(np.mean(vs_c))
                        chg_rec[f"v_std_{seg}"]      = float(np.std(vs_c))
                        chg_rec[f"energy_Wh_{seg}"]  = float(
                            np.sum(vs_c * ims_c * dts_c) / 3600.0)
                        chg_rec[f"q_abs_{seg}"]      = float(
                            np.sum(ims_c * dts_c) / 3600.0)
                        chg_rec[f"ica_peak_h_{seg}"] = _seg_ica_peak(vs_c, ims_c, dts_c)
                        chg_rec[f"dvdq_min_{seg}"]   = _seg_dvdq_min(vs_c, ims_c,
                                                                       dts_c, qcs_c)
                        chg_rec.update(
                            _seg_extra_his(vs_c, ims_c, dts_c, qcs_c, tmps_c, seg))

        # ── 레코드 조립 ───────────────────────────────────────────────────
        records.append({
            "dataset": dataset, "cell_id": cell_id,
            "cycle": int(cyc), "capacity_Ah": cap,
            # 방전 Global
            "v_mean": v_mean, "v_std": v_std, "v_skew": v_skew,
            "v_kurt": v_kurt, "v_end": v_end, "v_drop": v_drop,
            "energy_Wh": energy_Wh, "v_energy": v_energy,
            "v_at_q20": v_at_q20, "v_at_q50": v_at_q50, "v_at_q80": v_at_q80,
            "q_high_v": q_high_v, "q_tail": q_tail,
            "q_plateau_ratio": q_plateau_ratio,
            "t_discharge": t_dis,
            "ica_peak_h": ica_h, "ica_peak_v": ica_v, "ica_peak_area": ica_area,
            "dvdq_min": dvdq_min,
            "ce": ce,
            "temp_mean": t_mean, "temp_max": t_max,
            # SoC 세그먼트
            **seg_rec,
            # 충전
            **chg_rec,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────

def load_all(pkl_dir: Path, n_workers: int = 4) -> pd.DataFrame:
    files = sorted(pkl_dir.glob("*.pkl"))
    all_rec = []
    if n_workers <= 1:
        for f in tqdm(files, desc=pkl_dir.name):
            all_rec.extend(_extract_one_cell(str(f)))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_extract_one_cell, str(f)): f for f in files}
            with tqdm(total=len(files), desc=pkl_dir.name) as pbar:
                for fut in as_completed(futs):
                    all_rec.extend(fut.result())
                    pbar.update(1)
    return pd.DataFrame(all_rec) if all_rec else pd.DataFrame()


def _save_per_cell_hi(df: pd.DataFrame, dataset: str) -> None:
    """셀별 HI 특성을 data_HI/{dataset}/{cell_id}.pkl 로 저장."""
    if df.empty:
        return
    out_dir = HI_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell_id, grp in df.groupby("cell_id"):
        grp.to_pickle(out_dir / f"{cell_id}.pkl")
    print(f"  셀별 HI 저장: {out_dir}  ({df['cell_id'].nunique()}개 셀)")


def load_or_extract(cache_path: Path = CACHE_PATH,
                    n_workers: int = 4,
                    force: bool = False) -> pd.DataFrame:
    """캐시(pkl)가 있으면 로드, 없으면 전체 추출 후 저장.

    전체 추출 시 data_HI/MIT/{cell_id}.pkl, data_HI/HUST/{cell_id}.pkl 도 저장.
    """
    if not force and cache_path.exists():
        print(f"  캐시 로드: {cache_path}")
        return pd.read_pickle(cache_path)

    print("=== MIT HI 추출 ===")
    df_mit  = load_all(MIT_DIR,  n_workers=n_workers)
    _save_per_cell_hi(df_mit, "MIT")
    print("=== HUST HI 추출 ===")
    df_hust = load_all(HUST_DIR, n_workers=n_workers)
    _save_per_cell_hi(df_hust, "HUST")
    df = pd.concat([df_mit, df_hust], ignore_index=True)
    print(f"  총 방전 사이클: MIT {len(df_mit):,}  /  HUST {len(df_hust):,}")

    df.to_pickle(cache_path)
    print(f"  캐시 저장: {cache_path}")
    return df


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ(HI, capacity_Ah) — MIT / HUST 각각."""
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")
    result = {}
    for ds in ["MIT", "HUST"]:
        sub  = df[df["dataset"] == ds]
        rhos = {}
        for hi in HI_KEYS:
            if hi not in sub.columns:
                rhos[hi] = np.nan; continue
            valid = sub[[hi, "capacity_Ah"]].dropna()
            rhos[hi] = spearmanr(valid[hi], valid["capacity_Ah"])[0] \
                if len(valid) > 30 else np.nan
        result[ds] = rhos
    return pd.DataFrame(result, index=HI_KEYS)


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def _draw_heatmap(ax, keys, title, corr_df, datasets=("MIT", "HUST")):
    """단일 히트맵 그리기. |ρ| 평균 내림차순 정렬."""
    avail  = [k for k in keys if k in corr_df.index]
    order  = (corr_df.loc[avail].abs().mean(axis=1)
               .fillna(0).sort_values(ascending=False).index.tolist())
    hm     = corr_df.loc[order, list(datasets)].values   # (n_hi, 2)

    im = ax.imshow(hm.T, aspect="auto", cmap="RdYlGn",
                   vmin=-1, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([HI_LABELS.get(k, k) for k in order],
                       rotation=38, ha="right", fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(datasets, fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=9, pad=5, fontweight="bold")

    for xi, k in enumerate(order):
        for yi, ds in enumerate(datasets):
            val = hm[xi, yi]
            txt = f"{val:.2f}" if np.isfinite(val) else "N/A"
            ax.text(xi, yi, txt, ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > 0.65 else "black",
                    fontweight="bold")
    return im, order


def plot_correlation(corr_df: pd.DataFrame, df: pd.DataFrame,
                     out_path: Path, n_top: int = 4):
    datasets = ["MIT", "HUST"]
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")

    for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = font; break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    # ── 레이아웃 ─────────────────────────────────────────────────────────
    # 행 0: Discharge Global (22 HIs)
    # 행 1: Dis SoC 3 segments 나란히
    # 행 2: Charge Global (6 HIs)
    # 행 3: Chg SoC 3 segments 나란히
    # 행 4: 상위 HI 산점도
    fig = plt.figure(figsize=(34, 30))
    fig.suptitle(
        "Health Indicator — Spearman ρ"
        "  [Discharge Global / Dis SoC / Charge Global / Chg SoC]",
        fontsize=13, fontweight="bold", y=0.998,
    )
    gs_main = gridspec.GridSpec(5, 1, figure=fig,
                                height_ratios=[1.1, 0.9, 0.55, 0.9, 1.7],
                                hspace=0.55)

    # ── 행 0: Discharge Global ────────────────────────────────────────────
    ax0 = fig.add_subplot(gs_main[0])
    im0, _ = _draw_heatmap(ax0, HI_GROUPS["Discharge — Global"],
                           "Discharge — Global  (22 HIs)", corr_df)

    # ── 행 1: Dis SoC 3 segments ─────────────────────────────────────────
    dis_seg_specs = [
        ("Dis. SoC 60~100%", "Dis SoC 60~100%  (초반·고전압,  q_frac 0~40%)"),
        ("Dis. SoC 30~60%",  "Dis SoC 30~60%   (플래토 중심,  q_frac 40~70%)"),
        ("Dis. SoC 0~30%",   "Dis SoC 0~30%    (후반·저전압,  q_frac 70~100%)"),
    ]
    gs_dis_seg = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[1], wspace=0.06)
    for ci, (gname, title) in enumerate(dis_seg_specs):
        ax_s = fig.add_subplot(gs_dis_seg[ci])
        _draw_heatmap(ax_s, HI_GROUPS[gname], title, corr_df)

    # ── 행 2: Charge Global ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs_main[2])
    im2, _ = _draw_heatmap(ax2, HI_GROUPS["Charge — Global"],
                           "Charge — Global  (6 HIs)", corr_df)

    # ── 행 3: Chg SoC 3 segments ─────────────────────────────────────────
    chg_seg_specs = [
        ("Chg. SoC 0~30%",   "Chg SoC 0~30%   (초반·저전압,  q_frac 0~40%)"),
        ("Chg. SoC 30~60%",  "Chg SoC 30~60%  (플래토 중심,  q_frac 40~70%)"),
        ("Chg. SoC 60~100%", "Chg SoC 60~100% (후반·CV,       q_frac 70~100%)"),
    ]
    gs_chg_seg = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[3], wspace=0.06)
    for ci, (gname, title) in enumerate(chg_seg_specs):
        ax_s = fig.add_subplot(gs_chg_seg[ci])
        _draw_heatmap(ax_s, HI_GROUPS[gname], title, corr_df)

    # ── 공유 컬러바 ──────────────────────────────────────────────────────
    cbar = plt.colorbar(im0, ax=[ax0, ax2], shrink=0.40, pad=0.01)
    cbar.set_label("Spearman ρ", fontsize=10)

    # ── 행 4: 상위 HI 산점도 ─────────────────────────────────────────────
    abs_mean = corr_df.abs().mean(axis=1).fillna(0).sort_values(ascending=False)
    top_his  = abs_mean.index[:n_top].tolist()

    gs_sc = gridspec.GridSpecFromSubplotSpec(
        2, n_top, subplot_spec=gs_main[4], hspace=0.52, wspace=0.30)
    cmaps  = {"MIT": "Blues",   "HUST": "Oranges"}
    colors = {"MIT": "#1f77b4", "HUST": "#d55e00"}

    for ci, hi_key in enumerate(top_his):
        for ri, ds in enumerate(datasets):
            ax = fig.add_subplot(gs_sc[ri, ci])
            sub = df[df["dataset"] == ds][[hi_key, "capacity_Ah", "cycle"]].dropna()

            if len(sub) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                ax.set_title(f"{HI_LABELS.get(hi_key, hi_key)}  [{ds}]", fontsize=8)
                continue

            cyc_n = ((sub["cycle"] - sub["cycle"].min()) /
                     max(sub["cycle"].max() - sub["cycle"].min(), 1))
            ax.scatter(sub[hi_key], sub["capacity_Ah"],
                       c=cyc_n, cmap=cmaps[ds],
                       s=1.5, alpha=0.35, linewidths=0, rasterized=True)

            if len(sub) > 20:
                coef  = np.polyfit(sub[hi_key], sub["capacity_Ah"], 1)
                x_lin = np.linspace(sub[hi_key].min(), sub[hi_key].max(), 200)
                ax.plot(x_lin, np.polyval(coef, x_lin),
                        "-", color=colors[ds], lw=1.8, alpha=0.9)

            rho     = corr_df.loc[hi_key, ds] if hi_key in corr_df.index else np.nan
            rho_str = f"ρ={rho:.3f}" if np.isfinite(rho) else "ρ=N/A"
            lbl     = HI_LABELS.get(hi_key, hi_key)
            tag     = HI_GROUP_TAG.get(hi_key, "")
            ax.set_title(f"{lbl}  [{ds}]\n[{tag}]  {rho_str}", fontsize=8, pad=3)
            ax.set_xlabel(lbl, fontsize=7)
            ax.set_ylabel("Capacity (Ah)", fontsize=7)
            ax.tick_params(labelsize=6)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    cpu = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description="HI 64종 추출 및 Spearman 상관 시각화")
    parser.add_argument("--workers", type=int, default=min(4, cpu))
    parser.add_argument("--n-top",   type=int, default=4,
                        help="산점도 표시 상위 HI 수 (기본: 4)")
    parser.add_argument("--force",   action="store_true",
                        help="캐시 무시하고 HI 재추출")
    args = parser.parse_args()

    df = load_or_extract(n_workers=args.workers, force=args.force)
    print(f"\n총 방전 사이클: {len(df):,}")

    print("\n=== Spearman ρ 계산 ===")
    corr = compute_correlations(df)

    # 그룹별 콘솔 출력
    for gname, gkeys in HI_GROUPS.items():
        avail = [k for k in gkeys if k in corr.index]
        sub   = corr.loc[avail].copy()
        sub["|ρ| avg"] = sub.abs().mean(axis=1)
        sub   = sub.sort_values("|ρ| avg", ascending=False)
        print(f"\n── {gname} ──")
        print(sub.to_string(float_format=lambda x: f"{x:+.3f}"))

    hi_plot_dir = STEP_DIR / "hi_plot" / date.today().strftime("%m%d")
    hi_plot_dir.mkdir(parents=True, exist_ok=True)
    out = hi_plot_dir / "hi_correlation.png"
    print(f"\n=== Plot 저장: {out} ===")
    plot_correlation(corr, df, out, n_top=args.n_top)

    print("\n=== 대표 셀 HI 플롯 ===")
    out_dir = STEP_DIR / "outputs" / date.today().strftime("%m%d")
    _plot_sample_hi(df, corr, out_dir)
    print("완료!")


def _plot_sample_hi(df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path) -> None:
    """대표 셀(MIT b1c0, HUST 1-1)의 상위 HI 사이클별 추이 → 4_hi_analysis/outputs/"""
    SAMPLES = {"MIT": "b1c0", "HUST": "1-1"}
    CMAPS   = {"MIT": "Blues", "HUST": "Oranges"}

    df_p = df.copy()
    df_p["dataset"] = df_p["dataset"].replace("MIT_MAT", "MIT")

    # |ρ| 평균 상위 4개 HI
    abs_mean = corr_df.abs().mean(axis=1).fillna(0).sort_values(ascending=False)
    top4 = abs_mean.index[:4].tolist()

    n_ds = len(SAMPLES)
    fig, axes = plt.subplots(n_ds, 4, figsize=(16, n_ds * 3.5),
                              squeeze=False, constrained_layout=True)
    fig.suptitle("[Step 4 HI 추출 결과]  대표 셀 상위 HI 사이클 추이",
                 fontsize=11, fontweight="bold")

    for ri, (ds, cell) in enumerate(SAMPLES.items()):
        sub = df_p[(df_p["dataset"] == ds) & (df_p["cell_id"] == cell)].sort_values("cycle")

        for ci, hi_key in enumerate(top4):
            ax = axes[ri, ci]
            if len(sub) == 0 or hi_key not in sub.columns:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                continue

            valid = sub[["cycle", hi_key, "capacity_Ah"]].dropna()
            if len(valid) < 3:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                continue

            cap_range = valid["capacity_Ah"].max() - valid["capacity_Ah"].min()
            c_norm = (valid["capacity_Ah"] - valid["capacity_Ah"].min()) / max(cap_range, 1e-9)
            sc = ax.scatter(valid["cycle"], valid[hi_key],
                            c=c_norm, cmap=CMAPS[ds], s=8, alpha=0.8)

            rho = corr_df.loc[hi_key, ds] if (hi_key in corr_df.index and ds in corr_df.columns) else np.nan
            rho_str = f"ρ={rho:.3f}" if np.isfinite(rho) else "ρ=N/A"
            lbl = HI_LABELS.get(hi_key, hi_key)
            tag = HI_GROUP_TAG.get(hi_key, "")

            title = f"{lbl}  [{tag}]\n{rho_str}" if ri == 0 else f"{lbl}  [{tag}]"
            ax.set_title(title, fontsize=8, fontweight="bold")
            ax.set_xlabel("Cycle", fontsize=7)
            ax.set_ylabel(lbl, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

            if ci == 3:
                cbar = fig.colorbar(sc, ax=ax, fraction=0.06, pad=0.02)
                cbar.set_label("Cap (norm)", fontsize=7)

        axes[ri, 0].set_ylabel(f"{ds}: {cell}\n{HI_LABELS.get(top4[0], top4[0])}", fontsize=7)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sample_hi.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  대표 셀 HI 플롯: {out}")


if __name__ == "__main__":
    main()
