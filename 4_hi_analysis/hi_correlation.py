"""
hi_correlation.py

MIT + HUST _2_data_clean 에서 HI 411종을 사이클별로 추출하고
방전 용량(capacity_Ah)과의 Spearman 상관계수를 계산·시각화.

입력 : _2_data_clean/MIT/*.pkl, _2_data_clean/HUST/*.pkl
출력 : hi_correlation.png
       _4_data_hi/MIT/{cell_id}.pkl
       _4_data_hi/HUST/{cell_id}.pkl
사용 : python hi_correlation.py [--workers N] [--n-top N] [--force]

HI 구조 (docs/NEW_HIS.md 참조)
  Global  (15):  G01–G15
  Segment (396): 6구간 × 66  (통계 S01–S20 / 미분 D01–D20 / LFP L01–L20 / Morph M01–M06)
  세그먼트: dis_hi / dis_mid / dis_lo / chg_lo / chg_mid / chg_hi
  키 명명: stat_{k}_{seg} / diff_{k}_{seg} / lfp_{k}_{seg} / morph_{k}_{seg}
"""

import argparse
import os
import pickle
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew
from scipy.stats import spearmanr
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP_DIR     = Path(__file__).resolve().parent
MIT_DIR      = PROJECT_ROOT / "_2_data_clean" / "MIT"
HUST_DIR     = PROJECT_ROOT / "_2_data_clean" / "HUST"
CACHE_PATH   = STEP_DIR / "hi_features.pkl"
HI_ROOT      = PROJECT_ROOT / "_4_data_hi"

# ─────────────────────────────────────────────────────────────────────────────
# HI 키 상수 정의
# ─────────────────────────────────────────────────────────────────────────────
THETA_FLAT = 0.05  # V/Ah — LFP 플래토 판별 임계값 (|dV/dQ| < θ_flat)

GLOBAL_HI_KEYS = [
    "q_dis", "energy_dis", "v_mean_dis", "r_dc_est", "q_plateau_frac",
    "ica_peak1_v", "ica_peak1_h", "ica_peak1_area",
    "dva_valley_q", "dva_valley_depth",
    "ce", "cv_q_frac", "cv_time_frac", "chg_ica_peak1_h", "ica_peak1_asym",
]
_GLOBAL_LABELS = {
    "q_dis":           "Q_dis",
    "energy_dis":      "E_dis",
    "v_mean_dis":      "μ(V)_dis",
    "r_dc_est":        "R_dc",
    "q_plateau_frac":  "Q_plat/Q",
    "ica_peak1_v":     "V @ peak(dQ/dV)",
    "ica_peak1_h":     "max(dQ/dV)",
    "ica_peak1_area":  "∫(dQ/dV)dV",
    "dva_valley_q":    "Q @ min(dV/dQ)",
    "dva_valley_depth":"min(dV/dQ)",
    "ce":              "CE",
    "cv_q_frac":       "Q_CV/Q",
    "cv_time_frac":    "t_CV/t",
    "chg_ica_peak1_h": "max(dQ/dV) [chg]",
    "ica_peak1_asym":  "ICA asym",
}

STAT_KEYS = [
    "v_mean", "v_std", "v_skew", "v_kurt", "v_ent",
    "i_mean", "i_std", "v_med", "corr_qi", "corr_vi",
    "q_abs", "energy_seg", "v_iqr", "v_range", "v_p10",
    "v_p90", "v_samp_ent", "corr_vt", "i_q_slope", "v_detrended_std",
]
_STAT_LABELS = {
    "v_mean":           "μ(V)",
    "v_std":            "σ(V)",
    "v_skew":           "skew(V)",
    "v_kurt":           "kurt(V)",
    "v_ent":            "H(V)",
    "i_mean":           "μ(|I|)",
    "i_std":            "σ(|I|)",
    "v_med":            "med(V)",
    "corr_qi":          "corr(Q,|I|)",
    "corr_vi":          "corr(V,|I|)",
    "q_abs":            "Q_seg",
    "energy_seg":       "E_seg",
    "v_iqr":            "IQR(V)",
    "v_range":          "Vmax−Vmin",
    "v_p10":            "V(10th pct)",
    "v_p90":            "V(90th pct)",
    "v_samp_ent":       "SampEn(V)",
    "corr_vt":          "corr(V,t)",
    "i_q_slope":        "slope(|I|/Q)",
    "v_detrended_std":  "σ(V detrend)",
}

DIFF_KEYS = [
    "dvdq_mean", "dvdq_std", "dvdq_max_abs", "dvdq_min", "dvdq_area",
    "dqdv_peak_h", "dqdv_peak_v", "dqdv_peak_w", "dqdv_area", "dvdt_slope",
    "dqdv_peak_asym", "d2vdq2_rms", "dvdq_skew", "dvdq_ent", "r_dyn_seg",
    "dqdv_valley_h", "dqdv_valley_v", "dvdq_peak_q", "dvdq_valley_q", "dqdv_area_asym",
]
_DIFF_LABELS = {
    "dvdq_mean":       "μ(dV/dQ)",
    "dvdq_std":        "σ(dV/dQ)",
    "dvdq_max_abs":    "max|dV/dQ|",
    "dvdq_min":        "min(dV/dQ)",
    "dvdq_area":       "∫|dV/dQ|dQ",
    "dqdv_peak_h":     "max(dQ/dV)",
    "dqdv_peak_v":     "V @ peak(dQ/dV)",
    "dqdv_peak_w":     "FWHM(dQ/dV)",
    "dqdv_area":       "∫(dQ/dV)dV",
    "dvdt_slope":      "ΔV/Δt",
    "dqdv_peak_asym":  "ICA asym",
    "d2vdq2_rms":      "rms(d²V/dQ²)",
    "dvdq_skew":       "skew(dV/dQ)",
    "dvdq_ent":        "H(dV/dQ)",
    "r_dyn_seg":       "R_dyn",
    "dqdv_valley_h":   "min(dQ/dV)",
    "dqdv_valley_v":   "V @ valley(dQ/dV)",
    "dvdq_peak_q":     "Q @ max|dV/dQ|",
    "dvdq_valley_q":   "Q @ min|dV/dQ|",
    "dqdv_area_asym":  "ICA area asym",
}

LFP_KEYS = [
    "plateau_frac", "plateau_v_mean", "plateau_v_std", "plateau_q_frac",
    "nonlin_idx", "v_sag_mid", "v_flatness", "delta_v_rms",
    "ocv_slope", "knee_v", "knee_q_frac", "v_concavity",
    "phase_entry_dvdq", "v_q_pearson", "ica_peak_cnt",
    "plateau_v_slope", "v_gradient_exit", "plateau_q_onset", "dv_dt_plateau", "v_ent_plateau",
]
_LFP_LABELS = {
    "plateau_frac":      "plat. frac.",
    "plateau_v_mean":    "μ(V)|plat",
    "plateau_v_std":     "σ(V)|plat",
    "plateau_q_frac":    "Q_plat/Q_seg",
    "nonlin_idx":        "NL index",
    "v_sag_mid":         "V sag(mid)",
    "v_flatness":        "V flatness",
    "delta_v_rms":       "rms(ΔV)",
    "ocv_slope":         "dV/dQ|mid",
    "knee_v":            "knee V",
    "knee_q_frac":       "knee q_frac",
    "v_concavity":       "V concav.",
    "phase_entry_dvdq":  "|dV/dQ|_entry",
    "v_q_pearson":       "corr(V,Q)",
    "ica_peak_cnt":      "# ICA peaks",
    "plateau_v_slope":   "slope(V)|plat",
    "v_gradient_exit":   "|dV/dQ|_exit",
    "plateau_q_onset":   "q_onset|plat",
    "dv_dt_plateau":     "dV/dt|plat",
    "v_ent_plateau":     "H(V)|plat",
}

# 카테고리 D: 형태학적 거리 (BOL 대비 DTW / 이산 Fréchet) × 3곡선 = 6종
MORPH_KEYS = [
    "vt_dtw", "vq_dtw", "ve_dtw",
    "vt_frec", "vq_frec", "ve_frec",
]
_MORPH_LABELS = {
    "vt_dtw":  "DTW(V-t)",       "vq_dtw":  "DTW(V-Q)",       "ve_dtw":  "DTW(V-E)",
    "vt_frec": "Fréchet(V-t)",   "vq_frec": "Fréchet(V-Q)",   "ve_frec": "Fréchet(V-E)",
}

DIS_SEGS = [
    (0.0, 0.4, "dis_hi",  "dis_hi (SoC 60–100%)"),
    (0.4, 0.7, "dis_mid", "dis_mid (SoC 30–60%)"),
    (0.7, 1.0, "dis_lo",  "dis_lo (SoC 0–30%)"),
]
CHG_SEGS = [
    (0.0, 0.4, "chg_lo",  "chg_lo (SoC 0–40%)"),
    (0.4, 0.7, "chg_mid", "chg_mid (SoC 40–70%)"),
    (0.7, 1.0, "chg_hi",  "chg_hi (SoC 70–100%)"),
]
ALL_SEGS = DIS_SEGS + CHG_SEGS

# scen 코드 및 segment_id (0-indexed, 시간 순서: 충전 먼저 → 방전)
_SEG_SCEN: "dict[str, tuple[int, int]]" = {
    "chg_lo":  ( 1, 0),
    "chg_mid": ( 2, 1),
    "chg_hi":  ( 3, 2),
    "dis_hi":  (-3, 3),
    "dis_mid": (-2, 4),
    "dis_lo":  (-1, 5),
}

# 세그먼트 HI 기본 이름 (접미사 제외) — 66개/구간 순서 고정
_SEG_HI_BASES: list = (
    [f"stat_{k}"  for k in STAT_KEYS]  +
    [f"diff_{k}"  for k in DIFF_KEYS]  +
    [f"lfp_{k}"   for k in LFP_KEYS]   +
    [f"morph_{k}" for k in MORPH_KEYS]
)

# ── 전체 HI 키 / 레이블 / 그룹 자동 빌드 ────────────────────────────────────
_HI_META: list = []
for _k in GLOBAL_HI_KEYS:
    _HI_META.append((_k, _GLOBAL_LABELS[_k]))
for _, _, _seg, _ in ALL_SEGS:
    for _k in STAT_KEYS:
        _HI_META.append((f"stat_{_k}_{_seg}", _STAT_LABELS[_k]))
    for _k in DIFF_KEYS:
        _HI_META.append((f"diff_{_k}_{_seg}", _DIFF_LABELS[_k]))
    for _k in LFP_KEYS:
        _HI_META.append((f"lfp_{_k}_{_seg}", _LFP_LABELS[_k]))
    for _k in MORPH_KEYS:
        _HI_META.append((f"morph_{_k}_{_seg}", _MORPH_LABELS[_k]))

ALL_HI_KEYS = [k for k, _ in _HI_META]   # 15 + 6×66 = 411
HI_LABELS   = {k: lbl for k, lbl in _HI_META}

HI_GROUPS: "OrderedDict[str, list[str]]" = OrderedDict()
HI_GROUPS["Global"] = GLOBAL_HI_KEYS[:]
for _, _, _seg, _seg_lbl in ALL_SEGS:
    HI_GROUPS[f"{_seg} — Stat"]  = [f"stat_{k}_{_seg}"  for k in STAT_KEYS]
    HI_GROUPS[f"{_seg} — Diff"]  = [f"diff_{k}_{_seg}"  for k in DIFF_KEYS]
    HI_GROUPS[f"{_seg} — LFP"]   = [f"lfp_{k}_{_seg}"   for k in LFP_KEYS]
    HI_GROUPS[f"{_seg} — Morph"] = [f"morph_{k}_{_seg}" for k in MORPH_KEYS]

HI_GROUP_TAG = {k: gname for gname, keys in HI_GROUPS.items() for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# 카테고리 D: 형태학적 거리 헬퍼 (top-level — multiprocessing 호환)
# ─────────────────────────────────────────────────────────────────────────────

_MORPH_GRID = 50   # 보간 그리드 해상도 (속도-정밀도 균형)
_DTW_BAND   = 5    # Sakoe-Chiba 밴드 (그리드의 10% = 위상 이동 허용폭)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Sakoe-Chiba banded DTW (정규화: / n)."""
    n = len(a)
    d = np.abs(a[:, None] - b[None, :])      # n×n 거리행렬 (vectorized)
    dtw = np.full((n, n), np.inf)
    dtw[0, 0] = d[0, 0]
    for j in range(1, min(_DTW_BAND + 1, n)):
        dtw[0, j] = dtw[0, j - 1] + d[0, j]
    for i in range(1, n):
        dtw[i, 0] = dtw[i - 1, 0] + d[i, 0]
    for i in range(1, n):
        j_lo = max(1, i - _DTW_BAND)
        j_hi = min(n, i + _DTW_BAND + 1)
        for j in range(j_lo, j_hi):
            best = dtw[i - 1, j]
            if dtw[i, j - 1] < best:
                best = dtw[i, j - 1]
            if dtw[i - 1, j - 1] < best:
                best = dtw[i - 1, j - 1]
            dtw[i, j] = d[i, j] + best
    return float(dtw[n - 1, n - 1]) / n


def _frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """이산 Fréchet 거리.

    고정 x-그리드에 보간된 1D 곡선에서는 대각선 경로가 최적이므로
    max|a[i]-b[i]| 와 동치 — O(n), numpy 연산.
    """
    return float(np.max(np.abs(a - b)))


def _seg_morph_curves(vs: np.ndarray, ims: np.ndarray, dts: np.ndarray):
    """세그먼트 → (V-t, V-Q, V-E) 3곡선을 [0,1] 정규화 그리드로 보간.

    Returns: (vt, vq, ve) — 계산 불가 시 None
    """
    if len(vs) < 8:
        return None, None, None

    t_cum = np.cumsum(dts)
    q_cum = np.cumsum(np.abs(ims) * dts) / 3600.0
    e_cum = np.cumsum(vs * np.abs(ims) * dts) / 3600.0

    grid = np.linspace(0.0, 1.0, _MORPH_GRID)

    def _interp(x_raw, min_val=1e-9):
        xf = float(x_raw[-1])
        if xf < min_val:
            return None
        return np.interp(grid, x_raw / xf, vs)

    vt = _interp(t_cum)
    vq = _interp(q_cum, min_val=1e-4)
    ve = _interp(e_cum, min_val=1e-7)
    return vt, vq, ve


# ─────────────────────────────────────────────────────────────────────────────
# 공유 헬퍼 함수 (top-level — multiprocessing 호환)
# ─────────────────────────────────────────────────────────────────────────────

def _build_vq_curve(vs, ims, dts, n_bins=None):
    """Q-빈 V-Q 곡선 + dV/dQ 스무딩 (SG window=15).

    Returns (qm, v_sm, dvdq_sm, q_tot)
      qm      : Q 빈 중점 [Ah], 크기 n_bins
      v_sm    : SG 스무딩된 V [V]
      dvdq_sm : dV/dQ [V/Ah]
      q_tot   : 구간 총 전하량 [Ah]
    """
    q_rel = np.cumsum(ims * dts) / 3600.0
    q_tot = float(q_rel[-1]) if len(q_rel) > 0 else 0.0
    n = len(vs)
    if n_bins is None:
        n_bins = max(8, min(30, n // 3))
    if q_tot < 0.005 or n < 8 or n_bins < 4:
        empty = np.full(max(n_bins, 1), np.nan)
        return empty, empty, empty, q_tot

    dq_b  = q_tot / n_bins
    q_e   = np.linspace(0.0, q_tot, n_bins + 1)
    qm    = (q_e[:-1] + q_e[1:]) / 2
    v_av  = np.full(n_bins, np.nan)
    for j in range(n_bins):
        m = (q_rel >= q_e[j]) & (q_rel < q_e[j + 1])
        if m.sum() > 0:
            v_av[j] = float(np.mean(vs[m]))
    vld = np.isfinite(v_av)
    if vld.sum() < 4:
        return qm, v_av, np.full(n_bins, np.nan), q_tot
    v_sm = np.interp(qm, qm[vld], v_av[vld])
    ws = min(15, n_bins - (1 - n_bins % 2))
    ws = max(3, ws if ws % 2 == 1 else ws - 1)
    try:
        v_sm = savgol_filter(v_sm, ws, min(3, ws - 1))
    except Exception:
        pass
    dvdq_sm = np.gradient(v_sm, dq_b)
    return qm, v_sm, dvdq_sm, q_tot


def _build_ica_seg(vs, ims, dts):
    """V-빈 dQ/dV 곡선 (ICA) 스무딩 (SG window=15).

    Returns (vmids, dqdv_sm) — 비어있으면 (array([]), array([]))
    """
    vr = float(vs.max() - vs.min()) if len(vs) > 1 else 0.0
    if vr < 0.01 or len(vs) < 8:
        return np.array([]), np.array([])
    v_lo  = float(vs.min()) - 0.002
    v_hi  = float(vs.max()) + 0.002
    n_b   = max(8, min(30, int(vr / 0.01)))
    edges = np.linspace(v_lo, v_hi, n_b + 1)
    dv    = edges[1] - edges[0]
    vmids = (edges[:-1] + edges[1:]) / 2
    dqdv  = np.zeros(n_b)
    for j in range(n_b):
        m = (vs >= edges[j]) & (vs < edges[j + 1])
        if m.sum() > 0:
            dqdv[j] = np.sum(ims[m] * dts[m]) / 3600.0 / dv
    ws = min(15, n_b - (1 - n_b % 2))
    ws = max(3, ws if ws % 2 == 1 else ws - 1)
    try:
        dqdv_sm = savgol_filter(dqdv, ws, min(3, ws - 1))
    except Exception:
        dqdv_sm = dqdv
    return vmids, dqdv_sm


def _peak_fwhm_asym(arr, pk_idx, x_arr):
    """ICA 피크의 FWHM 과 비대칭도 (left_hw / right_hw).

    Returns (fwhm, asym) — 계산 불가 시 (nan, nan)
    """
    h = float(arr[pk_idx])
    if h <= 0:
        return np.nan, np.nan
    half = h / 2.0
    left_idx = 0
    for j in range(pk_idx, -1, -1):
        if arr[j] <= half:
            left_idx = j
            break
    right_idx = len(arr) - 1
    for j in range(pk_idx, len(arr)):
        if arr[j] <= half:
            right_idx = j
            break
    if left_idx == pk_idx or right_idx == pk_idx:
        return np.nan, np.nan
    fwhm     = float(x_arr[right_idx] - x_arr[left_idx])
    x_peak   = float(x_arr[pk_idx])
    left_hw  = x_peak - float(x_arr[left_idx])
    right_hw = float(x_arr[right_idx]) - x_peak
    if right_hw < 1e-9:
        return fwhm, np.nan
    return fwhm, left_hw / right_hw


def _global_ica(v, i_mag, dt, v_lo=2.8, v_hi=3.65, n_bins=80):
    """전체 방전/충전 ICA (dQ/dV). SG window=21.

    Returns (peak_v, peak_h, peak_area, asym) — LFP 범위 [3.1, 3.5] V 내 1차 피크
    """
    edges  = np.linspace(v_lo, v_hi, n_bins + 1)
    dv     = edges[1] - edges[0]
    vmids  = (edges[:-1] + edges[1:]) / 2
    dqdv   = np.zeros(n_bins)
    for j in range(n_bins):
        m = (v >= edges[j]) & (v < edges[j + 1])
        if m.sum() > 0:
            dqdv[j] = np.sum(i_mag[m] * dt[m]) / 3600.0 / dv
    ws = min(21, n_bins - (1 - n_bins % 2))
    ws = max(3, ws if ws % 2 == 1 else ws - 1)
    try:
        dqdv_s = savgol_filter(dqdv, ws, min(3, ws - 1))
    except Exception:
        dqdv_s = dqdv
    lfp_m = (vmids >= 3.1) & (vmids <= 3.5)
    if lfp_m.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan
    sub   = dqdv_s[lfp_m]; subv = vmids[lfp_m]
    pk    = int(np.argmax(sub))
    peak_v    = float(subv[pk])
    peak_h    = float(sub[pk])
    peak_area = float(np.trapezoid(np.maximum(sub, 0), subv))
    full_pk   = int(np.where(lfp_m)[0][0]) + pk
    _, asym   = _peak_fwhm_asym(dqdv_s, full_pk, vmids)
    return peak_v, peak_h, peak_area, asym


def _global_dva(v, i_mag, dt, q_local):
    """전체 방전 DVA (dV/dQ). SG window=21.

    Returns (valley_q, valley_depth) — LFP 플래토 범위 [3.15, 3.50] V 내 최솟값
    """
    if q_local < 0.1 or len(v) < 20:
        return np.nan, np.nan
    dq_bin  = max(q_local / 50.0, 0.005)
    q_cum   = np.cumsum(i_mag * dt) / 3600.0
    q_edges = np.arange(0.0, q_local + dq_bin, dq_bin)
    n_seg   = len(q_edges) - 1
    v_avg   = np.full(n_seg, np.nan)
    for j in range(n_seg):
        m = (q_cum >= q_edges[j]) & (q_cum < q_edges[j + 1])
        if m.sum() > 0:
            v_avg[j] = float(np.mean(v[m]))
    valid = np.isfinite(v_avg)
    if valid.sum() < 5:
        return np.nan, np.nan
    qm     = (q_edges[:-1] + q_edges[1:]) / 2
    v_fill = np.interp(qm, qm[valid], v_avg[valid])
    ws = min(21, n_seg - (1 - n_seg % 2))
    ws = max(3, ws if ws % 2 == 1 else ws - 1)
    try:
        v_s = savgol_filter(v_fill, ws, min(3, ws - 1))
    except Exception:
        v_s = v_fill
    dvdqa = np.gradient(v_s, dq_bin)
    plt_m = (v_fill >= 3.15) & (v_fill <= 3.50)
    if plt_m.sum() < 2:
        return np.nan, np.nan
    sub_q   = qm[plt_m]; sub_d = dvdqa[plt_m]
    vi      = int(np.argmin(sub_d))
    return float(sub_q[vi]), float(sub_d[vi])


def _r_dc_from_chg(vc, ic, dtc):
    """CC→CV 전환 시 ΔV/ΔI 로 직류 내부저항 추정 [mΩ].

    전환 전후 각 5샘플 평균으로 안정화. 전환 없으면 NaN.
    """
    if len(ic) < 15:
        return np.nan
    i_mx = float(np.max(ic))
    if i_mx < 0.01:
        return np.nan
    cc_mask = ic >= 0.80 * i_mx
    trans_idx = None
    for j in range(1, len(cc_mask)):
        if cc_mask[j - 1] and not cc_mask[j]:
            trans_idx = j
            break
    if trans_idx is None or trans_idx < 3:
        return np.nan
    pre  = max(0, trans_idx - 5)
    post = min(len(ic), trans_idx + 5)
    v_pre  = float(np.mean(vc[pre:trans_idx]))
    v_post = float(np.mean(vc[trans_idx:post]))
    i_pre  = float(np.mean(ic[pre:trans_idx]))
    i_post = float(np.mean(ic[trans_idx:post]))
    di = abs(i_pre - i_post)
    dv = abs(v_pre - v_post)
    if di < 0.01:
        return np.nan
    r = dv / di * 1000.0   # mΩ
    return float(r) if 0.0 < r < 1000.0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 세그먼트 HI 계산 함수 (top-level — multiprocessing 호환)
# ─────────────────────────────────────────────────────────────────────────────

def _seg_stat(vs, ims, dts, qcs, seg):
    """카테고리 A: 통계 기반 15종 (S01–S15)."""
    out = {f"stat_{k}_{seg}": np.nan for k in STAT_KEYS}
    n = len(vs)
    if n < 5:
        return out
    q_rel = qcs - qcs[0]

    # S01 v_mean (용량 가중 평균)
    denom = float(np.sum(ims * dts))
    out[f"stat_v_mean_{seg}"] = (
        float(np.sum(vs * ims * dts)) / denom if denom > 1e-9 else float(np.mean(vs))
    )
    # S02–S04
    out[f"stat_v_std_{seg}"]  = float(np.std(vs))
    if n >= 3:
        out[f"stat_v_skew_{seg}"] = float(sp_skew(vs))
    if n >= 4:
        out[f"stat_v_kurt_{seg}"] = float(sp_kurtosis(vs))
    # S05 v_ent (PMF, 20-bin)
    _cnt = np.histogram(vs, bins=20)[0].astype(float)
    _tot = _cnt.sum()
    if _tot > 0:
        p = _cnt[_cnt > 0] / _tot
        out[f"stat_v_ent_{seg}"] = float(-np.sum(p * np.log(p)))
    # S06–S07
    out[f"stat_i_mean_{seg}"] = float(np.mean(ims))
    out[f"stat_i_std_{seg}"]  = float(np.std(ims))
    # S08
    out[f"stat_v_med_{seg}"]  = float(np.median(vs))
    # S09 corr_qi
    if np.std(q_rel) > 1e-9 and np.std(ims) > 1e-9:
        out[f"stat_corr_qi_{seg}"] = float(np.corrcoef(q_rel, ims)[0, 1])
    # S10 corr_vi
    if np.std(vs) > 1e-6 and np.std(ims) > 1e-9:
        out[f"stat_corr_vi_{seg}"] = float(np.corrcoef(vs, ims)[0, 1])
    # S11–S12
    out[f"stat_q_abs_{seg}"]      = float(np.sum(ims * dts) / 3600.0)
    out[f"stat_energy_seg_{seg}"] = float(np.sum(vs * ims * dts) / 3600.0)
    # S13–S15
    out[f"stat_v_iqr_{seg}"]  = float(np.percentile(vs, 75) - np.percentile(vs, 25))
    out[f"stat_v_range_{seg}"] = float(vs.max() - vs.min())
    out[f"stat_v_p10_{seg}"]   = float(np.percentile(vs, 10))

    # S16 v_p90
    out[f"stat_v_p90_{seg}"] = float(np.percentile(vs, 90))

    # S17 v_samp_ent (SampEn, m=2, r=0.2·std) — vectorized, subsampled ≤200 pts
    if n >= 10:
        r_tol = 0.2 * float(np.std(vs))
        if r_tol > 0:
            xs = vs[::max(1, n // 200)] if n > 200 else vs
            ns = len(xs)
            if ns >= 10:
                w2 = np.column_stack([xs[:-1], xs[1:]])
                w3 = np.column_stack([xs[:-2], xs[1:-1], xs[2:]])
                c2 = np.max(np.abs(w2[:, None, :] - w2[None, :, :]), axis=2)
                c3 = np.max(np.abs(w3[:, None, :] - w3[None, :, :]), axis=2)
                np.fill_diagonal(c2, np.inf)
                np.fill_diagonal(c3, np.inf)
                B_se = int(np.sum(c2 <= r_tol))
                A_se = int(np.sum(c3 <= r_tol))
                if B_se > 0 and A_se > 0:
                    out[f"stat_v_samp_ent_{seg}"] = float(-np.log(A_se / B_se))

    # S18 corr_vt and S20 v_detrended_std — shared t_norm
    t_seg = np.zeros(n)
    t_seg[1:] = np.cumsum(dts[1:])
    t_tot_seg = float(t_seg[-1])
    if t_tot_seg > 0:
        t_norm_s = t_seg / t_tot_seg
        # S18 corr_vt
        if np.std(vs) > 1e-6:
            out[f"stat_corr_vt_{seg}"] = float(np.corrcoef(vs, t_norm_s)[0, 1])
        # S20 v_detrended_std
        A20 = np.column_stack([t_norm_s, np.ones(n)])
        coef20 = np.linalg.lstsq(A20, vs, rcond=None)[0]
        out[f"stat_v_detrended_std_{seg}"] = float(np.std(vs - A20 @ coef20))

    # S19 i_q_slope (OLS slope of |I| vs Q_cum)
    if np.std(q_rel) > 1e-9:
        A19 = np.column_stack([q_rel, np.ones(n)])
        coef19 = np.linalg.lstsq(A19, ims, rcond=None)[0]
        out[f"stat_i_q_slope_{seg}"] = float(coef19[0])

    return out


def _seg_diff(vs, ims, dts, qcs, seg):
    """카테고리 B: 미분 기반 15종 (D01–D15)."""
    out = {f"diff_{k}_{seg}": np.nan for k in DIFF_KEYS}
    n = len(vs)
    if n < 8:
        return out

    # V-Q 곡선 (dV/dQ)
    n_bins = max(8, min(30, n // 3))
    qm, v_sm, dvdq_sm, q_tot = _build_vq_curve(vs, ims, dts, n_bins=n_bins)
    if q_tot > 0.005 and np.any(np.isfinite(dvdq_sm)):
        fin = np.isfinite(dvdq_sm)
        vd  = dvdq_sm[fin]
        if len(vd) >= 3:
            # D01–D05
            out[f"diff_dvdq_mean_{seg}"]    = float(np.mean(vd))
            out[f"diff_dvdq_std_{seg}"]     = float(np.std(vd))
            out[f"diff_dvdq_max_abs_{seg}"] = float(np.max(np.abs(vd)))
            out[f"diff_dvdq_min_{seg}"]     = float(np.min(vd))
            out[f"diff_dvdq_area_{seg}"]    = float(np.trapezoid(np.abs(dvdq_sm[fin]), qm[fin]))
            # D12 d²V/dQ² RMS
            dq_b = float(qm[1] - qm[0]) if len(qm) > 1 else 1.0
            d2   = np.gradient(dvdq_sm, dq_b)
            fin2 = np.isfinite(d2)
            if fin2.sum() > 0:
                out[f"diff_d2vdq2_rms_{seg}"] = float(np.sqrt(np.mean(d2[fin2] ** 2)))
            # D13 skew, D14 entropy
            out[f"diff_dvdq_skew_{seg}"] = float(sp_skew(vd))
            _cnt = np.histogram(np.abs(vd), bins=10)[0].astype(float)
            _tot = _cnt.sum()
            if _tot > 0:
                p = _cnt[_cnt > 0] / _tot
                out[f"diff_dvdq_ent_{seg}"] = float(-np.sum(p * np.log(p)))

    # dQ/dV (ICA in segment)
    vmids, dqdv_sm = _build_ica_seg(vs, ims, dts)
    if len(vmids) >= 4:
        # D09 dqdv_area
        out[f"diff_dqdv_area_{seg}"] = float(np.trapezoid(np.maximum(dqdv_sm, 0), vmids))
        pk = int(np.argmax(dqdv_sm))
        if dqdv_sm[pk] > 0:
            # D06–D08
            out[f"diff_dqdv_peak_h_{seg}"] = float(dqdv_sm[pk])
            out[f"diff_dqdv_peak_v_{seg}"] = float(vmids[pk])
            fwhm, asym = _peak_fwhm_asym(dqdv_sm, pk, vmids)
            out[f"diff_dqdv_peak_w_{seg}"]    = fwhm
            out[f"diff_dqdv_peak_asym_{seg}"] = asym  # D11

    # D10 dvdt_slope: 총 기울기 (FP 아티팩트 방지)
    dt_tot = float(np.sum(dts))
    if dt_tot >= 1.0:
        out[f"diff_dvdt_slope_{seg}"] = float(vs[-1] - vs[0]) / dt_tot

    # D15 r_dyn_seg: |ΔV/ΔI| where ΔI≠0, Δt<2s
    if n > 1:
        dv_a = np.diff(vs); di_a = np.diff(ims); dt_a = dts[1:]
        valid = (np.abs(di_a) > 0.01) & (dt_a < 2.0) & (dt_a > 0)
        if valid.sum() > 0:
            r_dyn = np.abs(dv_a[valid] / di_a[valid])
            r_dyn = r_dyn[r_dyn < 1000.0]
            if len(r_dyn) > 0:
                out[f"diff_r_dyn_seg_{seg}"] = float(np.mean(r_dyn))

    # D16–D17: IC curve valley (min of dQ/dV, relative to peak — uses ICA vmids/dqdv_sm)
    if len(vmids) >= 6:
        pk16 = int(np.argmax(dqdv_sm))
        pk16_h = float(dqdv_sm[pk16])
        if pk16_h > 0 and pk16 >= 2 and pk16 <= len(dqdv_sm) - 3:
            li = int(np.argmin(dqdv_sm[:pk16]))
            ri = pk16 + 1 + int(np.argmin(dqdv_sm[pk16 + 1:]))
            lh, rh = float(dqdv_sm[li]), float(dqdv_sm[ri])
            lv, rv = float(vmids[li]), float(vmids[ri])
            lval = lh <= 0.2 * pk16_h
            rval = rh <= 0.2 * pk16_h
            if lval or rval:
                vpk = float(vmids[pk16])
                if lval and rval:
                    if (vpk - lv) >= (rv - vpk):
                        vh, vv = lh, lv
                    else:
                        vh, vv = rh, rv
                elif lval:
                    vh, vv = lh, lv
                else:
                    vh, vv = rh, rv
                out[f"diff_dqdv_valley_h_{seg}"] = vh
                out[f"diff_dqdv_valley_v_{seg}"] = vv

    # D18–D19: V-Q curve peak/valley Q positions
    fin18 = np.isfinite(dvdq_sm)
    if q_tot > 0.005 and fin18.sum() >= 3:
        qm_f18 = qm[fin18]
        dv_f18 = dvdq_sm[fin18]
        out[f"diff_dvdq_peak_q_{seg}"]   = float(qm_f18[int(np.argmax(np.abs(dv_f18)))])
        out[f"diff_dvdq_valley_q_{seg}"] = float(qm_f18[int(np.argmin(np.abs(dv_f18)))])

    # D20: IC area asymmetry (left / right of peak)
    if len(vmids) >= 4:
        pk20 = int(np.argmax(dqdv_sm))
        if float(dqdv_sm[pk20]) > 0 and pk20 >= 1 and pk20 <= len(dqdv_sm) - 2:
            al = float(np.trapezoid(np.maximum(dqdv_sm[:pk20 + 1], 0), vmids[:pk20 + 1]))
            ar = float(np.trapezoid(np.maximum(dqdv_sm[pk20:],     0), vmids[pk20:]))
            if al > 1e-9 and ar > 1e-9:
                out[f"diff_dqdv_area_asym_{seg}"] = float(al / ar)

    return out


def _seg_lfp(vs, ims, dts, qcs, seg):
    """카테고리 C: LFP 특징 기반 15종 (L01–L15)."""
    out = {f"lfp_{k}_{seg}": np.nan for k in LFP_KEYS}
    n = len(vs)
    if n < 8:
        return out

    n_bins = max(8, min(30, n // 3))
    qm, v_sm, dvdq_sm, q_tot = _build_vq_curve(vs, ims, dts, n_bins=n_bins)
    dq_b = float(qm[1] - qm[0]) if len(qm) > 1 else 1.0

    if q_tot < 0.005:
        return out

    fin_b = np.isfinite(dvdq_sm) & np.isfinite(v_sm)

    # L01–L04: 플래토 기반
    plt_mask = fin_b & (np.abs(dvdq_sm) < THETA_FLAT)
    n_b = len(qm)
    plt_frac = float(plt_mask.sum()) / n_b if n_b > 0 else 0.0
    out[f"lfp_plateau_frac_{seg}"] = plt_frac
    min_plt_bins = max(2, int(0.05 * n_b))
    if plt_mask.sum() >= min_plt_bins:
        plt_vs = v_sm[plt_mask]
        out[f"lfp_plateau_v_mean_{seg}"] = float(np.mean(plt_vs))
        out[f"lfp_plateau_v_std_{seg}"]  = float(np.std(plt_vs))
        q_plt = float(plt_mask.sum() * dq_b)
        out[f"lfp_plateau_q_frac_{seg}"] = q_plt / q_tot if q_tot > 0 else np.nan

    # L05 nonlin_idx: RMSE(V, V_linear) / V_range
    if fin_b.sum() >= 4:
        v_lin = np.interp(qm, [qm[0], qm[-1]], [v_sm[0], v_sm[-1]])
        v_rng = float(v_sm[fin_b].max() - v_sm[fin_b].min())
        if v_rng > 1e-4:
            rmse = float(np.sqrt(np.mean((v_sm[fin_b] - v_lin[fin_b]) ** 2)))
            out[f"lfp_nonlin_idx_{seg}"] = rmse / v_rng

    # L06 v_sag_mid
    q_mid = q_tot / 2.0
    if fin_b.any():
        v_mid     = float(np.interp(q_mid, qm, v_sm))
        v_lin_mid = float(np.interp(q_mid, [qm[0], qm[-1]], [v_sm[0], v_sm[-1]]))
        out[f"lfp_v_sag_mid_{seg}"] = v_mid - v_lin_mid

    # L07 v_flatness
    v_rng_raw = float(vs.max() - vs.min())
    if v_rng_raw > 1e-4:
        out[f"lfp_v_flatness_{seg}"] = 1.0 - float(np.std(vs)) / v_rng_raw

    # L08 delta_v_rms (dt >= 1s のみ)
    if n > 1:
        dt_pairs = dts[1:]
        slow = dt_pairs >= 1.0
        if slow.sum() > 0:
            dv_arr = np.diff(vs)[slow]
            out[f"lfp_delta_v_rms_{seg}"] = float(np.sqrt(np.mean(dv_arr ** 2)))

    # L09 ocv_slope: dV/dQ at q_mid
    if fin_b.any():
        out[f"lfp_ocv_slope_{seg}"] = float(np.interp(q_mid, qm, dvdq_sm))

    # L10–L11 knee (V-Q 변곡점)
    if fin_b.sum() >= 6 and n_b >= 6:
        d2 = np.gradient(dvdq_sm, dq_b)
        ws11 = min(11, n_b - (1 - n_b % 2))
        ws11 = max(3, ws11 if ws11 % 2 == 1 else ws11 - 1)
        try:
            d2_sm = savgol_filter(d2, ws11, min(2, ws11 - 1))
        except Exception:
            d2_sm = d2
        sc = np.where(np.diff(np.sign(d2_sm)) != 0)[0]
        if len(sc) > 0:
            best = sc[int(np.argmax(np.abs(d2_sm[sc])))]
            out[f"lfp_knee_v_{seg}"]      = float(v_sm[best])
            out[f"lfp_knee_q_frac_{seg}"] = float(qm[best]) / q_tot

    # L12 v_concavity
    if n >= 10:
        denom_cw = float(np.sum(ims * dts))
        v_mean_cw = (
            float(np.sum(vs * ims * dts)) / denom_cw if denom_cw > 1e-9
            else float(np.mean(vs))
        )
        out[f"lfp_v_concavity_{seg}"] = v_mean_cw - (float(vs[0]) + float(vs[-1])) / 2.0

    # L13 phase_entry_dvdq: |dV/dQ| 구간 첫 5%
    n5 = max(1, int(0.05 * n_b))
    if fin_b[:n5].sum() > 0:
        out[f"lfp_phase_entry_dvdq_{seg}"] = float(
            np.mean(np.abs(dvdq_sm[:n5][fin_b[:n5]]))
        )

    # L14 v_q_pearson
    q_rel = qcs - qcs[0]
    if np.std(vs) > 1e-6 and np.std(q_rel) > 1e-9:
        out[f"lfp_v_q_pearson_{seg}"] = float(np.corrcoef(vs, q_rel)[0, 1])

    # L15 ica_peak_cnt
    vmids_ica, dqdv_ica = _build_ica_seg(vs, ims, dts)
    if len(vmids_ica) >= 4:
        try:
            pks, _ = find_peaks(dqdv_ica, height=0)
            out[f"lfp_ica_peak_cnt_{seg}"] = float(len(pks))
        except Exception:
            pass

    # L16 plateau_v_slope (OLS slope of V vs Q_cum within plateau mask)
    if plt_mask.sum() >= 5:
        qp16 = qm[plt_mask]
        vp16 = v_sm[plt_mask]
        if float(qp16[-1] - qp16[0]) > 1e-9:
            A16 = np.column_stack([qp16, np.ones(len(qp16))])
            out[f"lfp_plateau_v_slope_{seg}"] = float(
                np.linalg.lstsq(A16, vp16, rcond=None)[0][0]
            )

    # L17 v_gradient_exit (mean |dV/dQ| at final 5% of seg)
    n5e = max(1, int(0.05 * n_b))
    exit_mask = np.zeros(n_b, dtype=bool)
    exit_mask[max(0, n_b - n5e):] = True
    valid_exit = exit_mask & fin_b
    if valid_exit.sum() >= 1:
        out[f"lfp_v_gradient_exit_{seg}"] = float(np.mean(np.abs(dvdq_sm[valid_exit])))

    # L18 plateau_q_onset (q_frac of first plateau sample in seg)
    plt_idx18 = np.where(plt_mask)[0]
    if len(plt_idx18) > 0 and q_tot > 0:
        out[f"lfp_plateau_q_onset_{seg}"] = float(qm[plt_idx18[0]]) / q_tot

    # L19 dv_dt_plateau (mean |dV/dt| in plateau region, dt>=1s only) [mV/s]
    if plt_mask.sum() >= 2 and q_tot > 0 and n > 1:
        q_plt_lo = float(qm[plt_mask][0])  - dq_b / 2
        q_plt_hi = float(qm[plt_mask][-1]) + dq_b / 2
        raw_in_plt = (q_rel >= q_plt_lo) & (q_rel <= q_plt_hi)
        if raw_in_plt.sum() >= 3:
            vs_p  = vs[raw_in_plt]
            dts_p = dts[raw_in_plt]
            slow_p = dts_p[1:] >= 1.0
            if slow_p.sum() >= 3:
                dvdt_p = np.abs(np.diff(vs_p)[slow_p] / dts_p[1:][slow_p])
                out[f"lfp_dv_dt_plateau_{seg}"] = float(np.mean(dvdt_p)) * 1000.0

    # L20 v_ent_plateau (Shannon entropy of V within plateau mask, 10-bin PMF)
    if plt_mask.sum() >= 10:
        _cnt20 = np.histogram(v_sm[plt_mask], bins=10)[0].astype(float)
        _tot20 = _cnt20.sum()
        if _tot20 > 0:
            p20 = _cnt20[_cnt20 > 0] / _tot20
            out[f"lfp_v_ent_plateau_{seg}"] = float(-np.sum(p20 * np.log(p20)))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 플래토 판정 디버그 시각화
# ─────────────────────────────────────────────────────────────────────────────

def plot_plateau_debug(
    df_cycle: pd.DataFrame,
    cycle_id: int = 0,
    cell_id: str = "",
    out_path=None,
) -> None:
    """6개 세그먼트별 플래토 판정 시각화 + 전체 사이클 V-Q 개요.

    레이아웃 (5행 × 3열, height_ratios=[2.5, 3, 2, 3, 2])
      행 0 (3열 전체) : 전체 사이클 V-Q 개요 (좌=방전, 우=충전)
                        세그먼트 구간 음영 + 플래토 빈 초록 마커
                        + 전체 대비 플래토 비율 표시
      행 1 : V-Q 곡선 — chg_lo / chg_mid / chg_hi
      행 2 : |dV/dQ|  — 동일 3 세그먼트
      행 3 : V-Q 곡선 — dis_hi / dis_mid / dis_lo
      행 4 : |dV/dQ|  — 동일 3 세그먼트
    """
    for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        try:
            import matplotlib
            matplotlib.rcParams["font.family"] = _f
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue
    # ── 방전 / 충전 분리 ───────────────────────────────────────────────────
    if "phase" not in df_cycle.columns:
        df_cycle = _add_phase(df_cycle)
    dis = df_cycle[df_cycle["phase"] == "discharge"].sort_values("time_s")
    chg = df_cycle[df_cycle["phase"] == "charge"].sort_values("time_s")

    if len(dis) < 30:
        print("[plateau_debug] 방전 데이터 부족 (< 30 row) — 스킵")
        return

    def _build_arrays(rows):
        v   = rows["voltage_V"].values.astype(float)
        i   = np.abs(rows["current_A"].values.astype(float))
        t   = rows["time_s"].values.astype(float)
        dt  = np.clip(np.diff(t, prepend=t[0]), 0, None)
        qc  = np.cumsum(i * dt) / 3600.0
        return v, i, dt, qc

    v_d, i_d, dt_d, q_d = _build_arrays(dis)
    q_tot_d = float(q_d[-1])

    has_chg = len(chg) >= 30
    if has_chg:
        v_c, i_c, dt_c, q_c = _build_arrays(chg)
        q_tot_c = float(q_c[-1])

    # ── 세그먼트별 색상 ────────────────────────────────────────────────────
    _SEG_COLORS = {
        "dis_hi":  ("#d6eaf8", "#2874a6"),
        "dis_mid": ("#e8daef", "#7d3c98"),
        "dis_lo":  ("#d5f5e3", "#1e8449"),
        "chg_lo":  ("#fef9e7", "#d4ac0d"),
        "chg_mid": ("#fdebd0", "#ca6f1e"),
        "chg_hi":  ("#fadbd8", "#cb4335"),
    }

    # ── 6 세그먼트 데이터 구성 ─────────────────────────────────────────────
    # 각 원소: None  또는
    #   (vs_raw, q_rel_mAh, qm_mAh(seg-rel), v_sm, dvdq_sm,
    #    q_tot_seg, plt_mask, fin_b, seg_lbl, q_abs_lo_mAh)
    seg_data = []

    for q_lo_f, q_hi_f, seg_name, seg_lbl in DIS_SEGS:
        lo = q_lo_f * q_tot_d
        hi = q_hi_f * q_tot_d
        m  = (q_d >= lo) & (q_d < hi)
        if m.sum() < 8:
            seg_data.append(None)
            continue
        vs_s  = v_d[m]; ims_s = i_d[m]; dts_s = dt_d[m]
        q_rel_raw = (q_d[m] - float(q_d[m][0])) * 1000
        n_bins = max(8, min(30, int(m.sum()) // 3))
        qm, v_sm, dvdq_sm, q_tot = _build_vq_curve(vs_s, ims_s, dts_s, n_bins=n_bins)
        fin_b    = np.isfinite(dvdq_sm) & np.isfinite(v_sm)
        plt_mask = fin_b & (np.abs(dvdq_sm) < THETA_FLAT)
        seg_data.append((vs_s, q_rel_raw, qm * 1000, v_sm, dvdq_sm,
                         q_tot, plt_mask, fin_b, seg_lbl, lo * 1000))

    for q_lo_f, q_hi_f, seg_name, seg_lbl in CHG_SEGS:
        if not has_chg:
            seg_data.append(None)
            continue
        lo = q_lo_f * q_tot_c
        hi = q_hi_f * q_tot_c
        m  = (q_c >= lo) & (q_c < hi)
        if m.sum() < 8:
            seg_data.append(None)
            continue
        vs_s  = v_c[m]; ims_s = i_c[m]; dts_s = dt_c[m]
        q_rel_raw = (q_c[m] - float(q_c[m][0])) * 1000
        n_bins = max(8, min(30, int(m.sum()) // 3))
        qm, v_sm, dvdq_sm, q_tot = _build_vq_curve(vs_s, ims_s, dts_s, n_bins=n_bins)
        fin_b    = np.isfinite(dvdq_sm) & np.isfinite(v_sm)
        plt_mask = fin_b & (np.abs(dvdq_sm) < THETA_FLAT)
        seg_data.append((vs_s, q_rel_raw, qm * 1000, v_sm, dvdq_sm,
                         q_tot, plt_mask, fin_b, seg_lbl, lo * 1000))

    import matplotlib.transforms as mtrans
    import matplotlib.patches as mpatches

    mode_colors = {"dis": "#2471a3", "chg": "#d35400"}
    mode_bg     = {1: "#eaf2fb", 2: "#eaf2fb", 3: "#fef9e7", 4: "#fef9e7"}

    # ── 그림: 5행 × 3열 ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 17))
    gs  = gridspec.GridSpec(
        5, 3, figure=fig,
        hspace=0.65, wspace=0.35,
        height_ratios=[2.5, 3, 2, 3, 2],
    )

    # ══════════════════════════════════════════════════════════════════════
    # 행 0: 전체 사이클 V-Q 개요 (좌=방전, 우=충전)
    # ══════════════════════════════════════════════════════════════════════
    gs_ov   = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :], wspace=0.25)
    ax_ov_d = fig.add_subplot(gs_ov[0])
    ax_ov_c = fig.add_subplot(gs_ov[1])
    ax_ov_d.set_facecolor("#f2f8fc")
    ax_ov_c.set_facecolor("#fdfaf0")

    def _draw_cycle_overview(ax_ov, q_full, v_full, q_tot_full,
                             segs_list, seg_offset, mode_lbl):
        """전체 사이클 V-Q 개요 패널 그리기."""
        if q_full is None:
            ax_ov.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                       transform=ax_ov.transAxes, fontsize=11, color="gray")
            ax_ov.set_title(f"{mode_lbl} — N/A", fontsize=9)
            return

        ax_ov.scatter(q_full * 1000, v_full,
                      s=1.5, color="lightgray", alpha=0.5, zorder=1)

        trans = mtrans.blended_transform_factory(ax_ov.transData, ax_ov.transAxes)
        total_plt, total_bins = 0, 0

        for s_idx, (q_lo_f, q_hi_f, seg_name, _) in enumerate(segs_list):
            sd      = seg_data[seg_offset + s_idx]
            bg_c, ln_c = _SEG_COLORS[seg_name]
            lo_mAh  = q_lo_f * q_tot_full * 1000
            hi_mAh  = q_hi_f * q_tot_full * 1000

            ax_ov.axvspan(lo_mAh, hi_mAh, alpha=0.22, color=bg_c, zorder=0)
            ax_ov.axvline(lo_mAh, color="gray", ls="--", lw=0.7, alpha=0.4, zorder=1)
            ax_ov.text((lo_mAh + hi_mAh) / 2, 0.97,
                       seg_name.split("_")[-1],
                       ha="center", va="top", fontsize=7.5, color=ln_c,
                       fontweight="bold", transform=trans)

            if sd is None:
                continue
            _vs, _qr, qm_rel, v_sm, _dv, _qt, plt_mask, _fb, _sl, q_abs_lo = sd
            qm_abs = qm_rel + q_abs_lo
            ax_ov.plot(qm_abs, v_sm, color=ln_c, lw=1.6, zorder=2)
            if plt_mask.any():
                ax_ov.scatter(qm_abs[plt_mask], v_sm[plt_mask],
                              s=28, color="limegreen", edgecolors="darkgreen",
                              linewidths=0.4, zorder=3)
                total_plt += int(plt_mask.sum())
            total_bins += len(qm_rel)

        pf = total_plt / max(1, total_bins)
        ax_ov.set_title(
            f"{mode_lbl}  |  전체 플래토 비율: {pf:.1%}  "
            f"({total_plt}/{total_bins} bins, θ={THETA_FLAT} V/Ah)",
            fontsize=9, pad=4,
        )
        ax_ov.set_xlabel("Q_cumulative [mAh]", fontsize=8)
        ax_ov.set_ylabel("V [V]", fontsize=8)
        ax_ov.tick_params(labelsize=7)

        leg_handles = [
            mpatches.Patch(fc=_SEG_COLORS[s][0], ec=_SEG_COLORS[s][1],
                           label=s.split("_")[-1])
            for _, _, s, _ in segs_list
        ] + [plt.Line2D([0], [0], marker="o", color="w",
                        markerfacecolor="limegreen", markersize=6,
                        markeredgecolor="darkgreen", label="plateau")]
        ax_ov.legend(handles=leg_handles, fontsize=6.5,
                     loc="lower right", framealpha=0.8)

    _draw_cycle_overview(ax_ov_d, q_d,   v_d,   q_tot_d, DIS_SEGS, 0, "DISCHARGE")
    _draw_cycle_overview(ax_ov_c,
                         q_c   if has_chg else None,
                         v_c   if has_chg else None,
                         q_tot_c if has_chg else 0.0,
                         CHG_SEGS, 3, "CHARGE")

    # ══════════════════════════════════════════════════════════════════════
    # 행 1–4: 세그먼트 상세 패널 (row_offsets 이 0,2 → 1,3 으로 이동)
    # ══════════════════════════════════════════════════════════════════════
    row_offsets = [1, 3]

    for mode_idx, offset in enumerate(row_offsets):
        segs_mode = seg_data[mode_idx * 3 : mode_idx * 3 + 3]
        clr_main  = mode_colors["dis"] if mode_idx == 0 else mode_colors["chg"]

        for col, sd in enumerate(segs_mode):
            ax_v  = fig.add_subplot(gs[offset,     col])
            ax_dv = fig.add_subplot(gs[offset + 1, col])
            ax_v.set_facecolor(mode_bg[offset])
            ax_dv.set_facecolor(mode_bg[offset + 1])

            if sd is None:
                for ax, txt in [(ax_v, "데이터 없음\n(샘플 < 8)"), (ax_dv, "—")]:
                    ax.text(0.5, 0.5, txt, ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="gray")
                    ax.set_xticks([]); ax.set_yticks([])
                continue

            vs_s, q_raw_mAh, qm_mAh, v_sm, dvdq_sm, q_tot, plt_mask, fin_b, seg_lbl, *_ = sd
            dq_bin_mAh   = float(qm_mAh[1] - qm_mAh[0]) if len(qm_mAh) > 1 else 1.0
            n_bins_total = len(qm_mAh)
            n_plt        = int(plt_mask.sum())
            plt_frac     = n_plt / max(1, n_bins_total)

            # ── V-Q 패널 ────────────────────────────────────────────────
            ax_v.scatter(q_raw_mAh, vs_s,
                         s=3, color=clr_main, alpha=0.25, zorder=1, label="raw")
            ax_v.plot(qm_mAh, v_sm, color=clr_main, lw=1.8, zorder=2, label="smoothed")
            if plt_mask.any():
                q_plt_e = qm_mAh[plt_mask]
                ax_v.axvspan(q_plt_e[0] - dq_bin_mAh / 2,
                             q_plt_e[-1] + dq_bin_mAh / 2,
                             alpha=0.18, color="limegreen", zorder=0)
                ax_v.scatter(qm_mAh[plt_mask], v_sm[plt_mask],
                             s=55, color="limegreen", edgecolors="darkgreen",
                             linewidths=0.6, zorder=3,
                             label=f"plateau ({n_plt}/{n_bins_total} bins)")
            ax_v.set_title(
                f"{seg_lbl}\nplateau_frac = {plt_frac:.1%}  (θ = {THETA_FLAT} V/Ah)",
                fontsize=8.5, pad=3,
            )
            ax_v.set_ylabel("V [V]", fontsize=8)
            ax_v.set_xlabel("Q_seg [mAh]", fontsize=8)
            ax_v.tick_params(labelsize=7)
            ax_v.legend(fontsize=6.5, loc="best", markerscale=1.2, framealpha=0.7)

            # ── |dV/dQ| 패널 ───────────────────────────────────────────
            abs_dvdq = np.abs(dvdq_sm)
            ax_dv.plot(qm_mAh, abs_dvdq, color="darkorange", lw=1.4, zorder=2,
                       label="|dV/dQ|")
            ax_dv.axhline(THETA_FLAT, color="red", ls="--", lw=1.3, zorder=3,
                          label=f"θ = {THETA_FLAT}")
            ax_dv.fill_between(
                qm_mAh, 0,
                np.where(abs_dvdq < THETA_FLAT, abs_dvdq, np.nan),
                color="limegreen", alpha=0.45, zorder=1, label="plateau zone",
            )
            p90 = float(np.nanpercentile(abs_dvdq[fin_b], 90)) if fin_b.any() else THETA_FLAT
            ax_dv.set_ylim(0, max(3.0 * THETA_FLAT, p90 * 1.2))
            ax_dv.set_ylabel("|dV/dQ| [V/Ah]", fontsize=8)
            ax_dv.set_xlabel("Q_seg [mAh]", fontsize=8)
            ax_dv.tick_params(labelsize=7)
            ax_dv.legend(fontsize=6.5, loc="upper right", framealpha=0.7)

    fig.text(0.004, 0.60, "DISCHARGE", fontsize=10, fontweight="bold",
             color=mode_colors["dis"], rotation=90, va="center")
    fig.text(0.004, 0.22, "CHARGE",    fontsize=10, fontweight="bold",
             color=mode_colors["chg"], rotation=90, va="center")

    fig.suptitle(
        f"Plateau Detection Debug  |  cell: {cell_id}  cycle: {cycle_id}  "
        f"|  θ_flat = {THETA_FLAT} V/Ah",
        fontsize=11, fontweight="bold", y=1.005,
    )

    if out_path is None:
        plt.show()
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  [plateau_debug] 저장: {out_path}")
    plt.close(fig)


def plot_plateau_fraction_summary(df: pd.DataFrame, out_path=None) -> None:
    """모든 셀/사이클의 plateau_frac 요약 플롯.

    2행 × 3열 그리드 (행=모드, 열=시나리오)
      행 0 : chg_lo / chg_mid / chg_hi
      행 1 : dis_hi / dis_mid / dis_lo

    각 서브플롯:
      - 얇은 선: 셀별 plateau_frac vs cycle (데이터셋 색조)
      - 굵은 선: 데이터셋 rolling 중앙값 (window=15)
      - 제목 통계: 데이터셋별 μ ± σ
    """
    import matplotlib
    import matplotlib.patches as mpatches
    for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        try:
            matplotlib.rcParams["font.family"] = _f
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    SEG_ORDER = [
        ("dis_hi",  "dis_hi  (SoC 60–100%)"),
        ("dis_mid", "dis_mid (SoC 30–60%)"),
        ("dis_lo",  "dis_lo  (SoC 0–30%)"),
        ("chg_lo",  "chg_lo  (SoC 0–40%)"),
        ("chg_mid", "chg_mid (SoC 40–70%)"),
        ("chg_hi",  "chg_hi  (SoC 70–100%)"),
    ]
    DS_CFG = {
        "MIT":  {"color": "#1a5276", "cmap": plt.cm.Blues},
        "HUST": {"color": "#784212", "cmap": plt.cm.Oranges},
    }

    df2 = df.copy()
    df2["dataset"] = df2["dataset"].replace("MIT_MAT", "MIT")
    datasets       = [d for d in ("MIT", "HUST") if d in df2["dataset"].values]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    fig.patch.set_facecolor("#fafafa")

    for idx, (seg, seg_lbl) in enumerate(SEG_ORDER):
        row, col = divmod(idx, 3)
        ax       = axes[row, col]
        col_name = f"lfp_plateau_frac_{seg}"
        is_dis   = seg.startswith("dis")
        ax.set_facecolor("#eef4fb" if is_dis else "#fdf6e3")

        if col_name not in df2.columns:
            ax.text(0.5, 0.5, f"컬럼 없음\n({col_name})",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_title(seg_lbl, fontsize=9)
            continue

        stats_parts = []

        for ds in datasets:
            cfg   = DS_CFG.get(ds, {"color": "gray", "cmap": plt.cm.Greys})
            ds_df = df2[df2["dataset"] == ds]
            cells = sorted(ds_df["cell_id"].unique())
            n_c   = len(cells)
            if n_c == 0:
                continue

            # 셀별 색조 (colormap)
            clrs = cfg["cmap"](np.linspace(0.35, 0.85, max(1, n_c)))
            for ci, cid in enumerate(cells):
                valid = (ds_df[ds_df["cell_id"] == cid]
                         .sort_values("cycle")[["cycle", col_name]]
                         .dropna())
                if len(valid) < 2:
                    continue
                ax.plot(valid["cycle"], valid[col_name],
                        color=clrs[ci], alpha=0.2, lw=0.8, zorder=1)

            # 데이터셋 rolling 중앙값
            by_cyc = (ds_df.groupby("cycle")[col_name]
                      .median().sort_index().dropna())
            if len(by_cyc) >= 5:
                roll = by_cyc.rolling(15, min_periods=3, center=True).median()
                ax.plot(roll.index, roll.values,
                        color=cfg["color"], lw=2.5, zorder=5,
                        label=f"{ds} (rolling median)")

            # 통계 요약
            vals = ds_df[col_name].dropna()
            if len(vals):
                n_cells_with_data = ds_df[ds_df[col_name].notna()]["cell_id"].nunique()
                stats_parts.append(
                    f"{ds}: μ={vals.mean():.3f} σ={vals.std():.3f} "
                    f"[{n_cells_with_data}cells/{len(vals):,}cyc]"
                )

        ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlim(left=0)
        ax.set_title(f"{seg_lbl}\n" + "    ".join(stats_parts), fontsize=8.5, pad=4)
        ax.set_xlabel("Cycle", fontsize=8)
        ax.set_ylabel("plateau_frac", fontsize=8)
        ax.tick_params(labelsize=7)

        leg = [mpatches.Patch(color=DS_CFG.get(d, {"color": "gray"})["color"],
                              label=f"{d} rolling median")
               for d in datasets]
        ax.legend(handles=leg, fontsize=7.5, loc="best", framealpha=0.8)

    fig.text(0.005, 0.75, "DISCHARGE", fontsize=11, fontweight="bold",
             color="#1a5276", rotation=90, va="center")
    fig.text(0.005, 0.27, "CHARGE",    fontsize=11, fontweight="bold",
             color="#784212", rotation=90, va="center")

    n_cyc  = len(df2)
    n_cell = df2["cell_id"].nunique()
    fig.suptitle(
        f"Plateau Fraction Summary  |  θ_flat = {THETA_FLAT} V/Ah  "
        f"|  {n_cyc:,} cycles  /  {n_cell} cells",
        fontsize=12, fontweight="bold",
    )

    if out_path is None:
        plt.show()
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  [plateau_summary] 저장: {out_path}")
    plt.close(fig)


def _run_plateau_debug(dataset: str, cell_id: str, cycle: int) -> None:
    """CLI --plateau-debug 진입점: pkl 로드 → plot_plateau_debug() 호출."""
    root = MIT_DIR if dataset.upper().startswith("MIT") else HUST_DIR
    matches = list(root.glob(f"{cell_id}*.pkl"))
    if not matches:
        # 파일명이 정확하지 않으면 부분 매치 시도
        matches = [p for p in root.glob("*.pkl") if cell_id in p.stem]
    if not matches:
        print(f"[plateau_debug] 파일 없음: {root}/{cell_id}*.pkl")
        return

    pkl_path = matches[0]
    print(f"[plateau_debug] 로드: {pkl_path}")
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    df_all = raw.get("cycles")
    if df_all is None:
        print("[plateau_debug] 'cycles' 키 없음")
        return

    cycles_avail = sorted(df_all["cycle"].unique())
    if cycle not in cycles_avail:
        print(f"[plateau_debug] 사이클 {cycle} 없음. 가용: {cycles_avail[:10]}…")
        cycle = cycles_avail[len(cycles_avail) // 2]   # 중간 사이클로 대체
        print(f"[plateau_debug] 중간 사이클 {cycle} 사용")

    df_cyc = df_all[df_all["cycle"] == cycle]

    out_dir = STEP_DIR / "outputs" / "plateau_debug"
    out_path = out_dir / f"plateau_debug_{cell_id}_cyc{cycle:04d}.png"
    plot_plateau_debug(df_cyc, cycle_id=cycle, cell_id=cell_id, out_path=out_path)


# ─────────────────────────────────────────────────────────────────────────────
# HI 추출 (top-level — multiprocessing 호환)
# ─────────────────────────────────────────────────────────────────────────────

_PHASE_POS =  0.01   # A 초과 → charge
_PHASE_NEG = -0.01   # A 미만 → discharge


def _add_phase(df: pd.DataFrame) -> pd.DataFrame:
    """_2_data_clean 스키마(phase 컬럼 없음)에 phase 컬럼을 current_A 부호로 재구성."""
    df = df.copy()
    cur = df["current_A"]
    df["phase"] = "rest"
    df.loc[cur > _PHASE_POS, "phase"] = "charge"
    df.loc[cur < _PHASE_NEG, "phase"] = "discharge"
    return df


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

    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    dataset = meta.get("dataset", "")
    cell_id = meta.get("cell_id", path.stem)
    records = []

    # 카테고리 D: BOL(최초 유효 사이클) 곡선 참조값 {seg: {"vt": arr, "vq": arr, "ve": arr}}
    bol_curves: dict = {}

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
        dt  = np.clip(np.diff(t, prepend=t[0]), 0, None)
        i_mag = np.abs(i)

        q_cum   = np.cumsum(i_mag * dt) / 3600.0
        q_local = float(q_cum[-1]) if len(q_cum) > 0 else 0.0

        # 실제 방전량 < 등록 용량 30% → 불완전 사이클
        if q_local < cap * 0.30:
            continue

        # ── 전체 HI 키 NaN 초기화 ────────────────────────────────────────
        row: dict = {k: np.nan for k in ALL_HI_KEYS}
        row.update({"dataset": dataset, "cell_id": cell_id,
                    "cycle": int(cyc), "capacity_Ah": cap})

        # ── G01–G03 방전 기본 ─────────────────────────────────────────────
        row["q_dis"]      = q_local
        row["energy_dis"] = float(np.sum(v * i_mag * dt) / 3600.0)
        denom = float(np.sum(i_mag * dt))
        if denom > 1e-9:
            row["v_mean_dis"] = float(np.sum(v * i_mag * dt)) / denom

        # ── G05 q_plateau_frac ────────────────────────────────────────────
        mask_plt = (v >= 3.10) & (v <= 3.45)
        if q_local > 0:
            row["q_plateau_frac"] = (
                float(np.sum(i_mag[mask_plt] * dt[mask_plt]) / 3600.0) / q_local
            )

        # ── G06–G08, G15: ICA ─────────────────────────────────────────────
        p1v, p1h, p1ar, p1asy = _global_ica(v, i_mag, dt)
        row["ica_peak1_v"]    = p1v
        row["ica_peak1_h"]    = p1h
        row["ica_peak1_area"] = p1ar
        row["ica_peak1_asym"] = p1asy

        # ── G09–G10: DVA ──────────────────────────────────────────────────
        row["dva_valley_q"], row["dva_valley_depth"] = _global_dva(
            v, i_mag, dt, q_local
        )

        # ── 방전 세그먼트 HI ──────────────────────────────────────────────
        if q_local >= 0.05:
            for q_lo_f, q_hi_f, seg, _ in DIS_SEGS:
                lo  = q_lo_f * q_local
                hi  = q_hi_f * q_local
                m_s = (q_cum >= lo) & (q_cum < hi)
                if m_s.sum() < 10:
                    continue
                vs_s = v[m_s]; ims_s = i_mag[m_s]; dts_s = dt[m_s]; qcs_s = q_cum[m_s]
                row.update(_seg_stat(vs_s, ims_s, dts_s, qcs_s, seg))
                row.update(_seg_diff(vs_s, ims_s, dts_s, qcs_s, seg))
                row.update(_seg_lfp(vs_s, ims_s, dts_s, qcs_s, seg))
                # D: 형태학적 거리
                _mc = _seg_morph_curves(vs_s, ims_s, dts_s)
                if seg not in bol_curves and all(c is not None for c in _mc):
                    bol_curves[seg] = dict(zip(("vt", "vq", "ve"), _mc))
                if seg in bol_curves:
                    for _ct, _arr in zip(("vt", "vq", "ve"), _mc):
                        _bol = bol_curves[seg].get(_ct)
                        if _arr is not None and _bol is not None:
                            try:
                                row[f"morph_{_ct}_dtw_{seg}"]  = _dtw_distance(_arr, _bol)
                                row[f"morph_{_ct}_frec_{seg}"] = _frechet_distance(_arr, _bol)
                            except Exception:
                                pass

        # ── 충전 HI ───────────────────────────────────────────────────────
        chg_grp = grp[grp["phase"] == "charge"].sort_values("time_s")
        if len(chg_grp) >= 20:
            tc  = chg_grp["time_s"].values.astype(float)
            vc  = chg_grp["voltage_V"].values.astype(float)
            ic  = np.abs(chg_grp["current_A"].values.astype(float))
            dtc = np.clip(np.diff(tc, prepend=tc[0]), 0, None)
            qcc = np.cumsum(ic * dtc) / 3600.0
            q_tc = float(qcc[-1])

            _chg_incomplete = q_tc < cap * 0.60
            _chg_gap_seg = (
                bool(chg_grp["chg_gap_seg"].any())
                if "chg_gap_seg" in chg_grp.columns else False
            )

            if q_tc > 0.05 and not _chg_incomplete:
                # G04 r_dc_est
                row["r_dc_est"] = _r_dc_from_chg(vc, ic, dtc)

                # G11 CE
                row["ce"] = cap / q_tc

                # G12–G13 CV 거동
                i_mx = float(np.max(ic))
                if i_mx > 0:
                    cv_mask = ic < 0.80 * i_mx
                    q_cv  = float(np.sum(ic[cv_mask] * dtc[cv_mask]) / 3600.0)
                    t_cv  = float(np.sum(dtc[cv_mask]))
                    t_tot = float(np.sum(dtc))
                    row["cv_q_frac"]   = q_cv / q_tc if q_tc > 0 else np.nan
                    row["cv_time_frac"] = t_cv / t_tot if t_tot > 0 else np.nan

                # G14 chg_ica_peak1_h
                _, c_pk_h, _, _ = _global_ica(vc, ic, dtc)
                row["chg_ica_peak1_h"] = c_pk_h

                # 충전 세그먼트 HI (CC 전환 갭 없는 경우만)
                if not _chg_gap_seg and q_tc >= 0.05:
                    for q_lo_f, q_hi_f, seg, _ in CHG_SEGS:
                        lo  = q_lo_f * q_tc
                        hi  = q_hi_f * q_tc
                        m_c = (qcc >= lo) & (qcc < hi)
                        if m_c.sum() < 10:
                            continue
                        vs_c = vc[m_c]; ims_c = ic[m_c]; dts_c = dtc[m_c]; qcs_c = qcc[m_c]
                        row.update(_seg_stat(vs_c, ims_c, dts_c, qcs_c, seg))
                        row.update(_seg_diff(vs_c, ims_c, dts_c, qcs_c, seg))
                        row.update(_seg_lfp(vs_c, ims_c, dts_c, qcs_c, seg))
                        # D: 형태학적 거리
                        _mc_c = _seg_morph_curves(vs_c, ims_c, dts_c)
                        if seg not in bol_curves and all(c is not None for c in _mc_c):
                            bol_curves[seg] = dict(zip(("vt", "vq", "ve"), _mc_c))
                        if seg in bol_curves:
                            for _ct, _arr in zip(("vt", "vq", "ve"), _mc_c):
                                _bol = bol_curves[seg].get(_ct)
                                if _arr is not None and _bol is not None:
                                    try:
                                        row[f"morph_{_ct}_dtw_{seg}"]  = _dtw_distance(_arr, _bol)
                                        row[f"morph_{_ct}_frec_{seg}"] = _frechet_distance(_arr, _bol)
                                    except Exception:
                                        pass

        records.append(row)

    return records


# ─────────────────────────────────────────────────────────────────────────────

def load_all(pkl_dir: Path, n_workers: int = 4) -> pd.DataFrame:
    files = sorted(pkl_dir.glob("*.pkl"))
    all_rec: list = []
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


def _to_cycle_df(df: pd.DataFrame) -> pd.DataFrame:
    """평탄 HI DataFrame → 사이클별 글로벌 HI 테이블.

    출력: [cell_id, cycle, capacity_Ah, <글로벌 HI 15개>]
    """
    cols = ["cell_id", "cycle", "capacity_Ah"] + GLOBAL_HI_KEYS
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


def _to_seg_df(df: pd.DataFrame) -> pd.DataFrame:
    """평탄 HI DataFrame → 세그먼트별 HI 테이블 (long format).

    출력: [cell_id, cycle, segment_id, capacity_Ah, scen, stat_v_mean, ..., morph_ve_frec]
    - segment_id: 0(chg_lo) / 1(chg_mid) / 2(chg_hi) / 3(dis_hi) / 4(dis_mid) / 5(dis_lo)
    - scen: -3 / -2 / -1 / 1 / 2 / 3
    - capacity_Ah: stat_q_abs_{seg} (구간 누적 용량 Ah)
    - HI 컬럼: _{seg} 접미사 제거 (66개/구간, 순서 고정)
    """
    parts = []
    for seg, (scen_val, seg_id) in _SEG_SCEN.items():
        suffix    = f"_{seg}"
        # 현재 df에 존재하는 세그먼트 HI 컬럼 → base 이름 매핑
        col_map   = {f"{b}{suffix}": b for b in _SEG_HI_BASES
                     if f"{b}{suffix}" in df.columns}
        if not col_map:
            continue

        sub = df[["cell_id", "cycle"] + list(col_map.keys())].copy()
        sub = sub.rename(columns=col_map)

        # capacity_Ah = 구간 누적 용량 (stat_q_abs_{seg})
        q_abs_col = f"stat_q_abs{suffix}"
        sub["capacity_Ah"] = df[q_abs_col].values if q_abs_col in df.columns else np.nan

        sub["segment_id"] = seg_id
        sub["scen"]       = scen_val

        hi_present = [b for b in _SEG_HI_BASES if b in sub.columns]
        sub = sub[["cell_id", "cycle", "segment_id", "capacity_Ah", "scen"] + hi_present]
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    return (pd.concat(parts, ignore_index=True)
              .sort_values(["cell_id", "cycle", "segment_id"])
              .reset_index(drop=True))


def _save_sample_csvs(df_mit: pd.DataFrame, df_hust: pd.DataFrame) -> None:
    """데이터셋별 대표 셀 첫 번째 사이클을 cycle/seg 형식으로 CSV 저장."""
    sample_dir = HI_ROOT / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for ds_tag, df_full in [("mit", df_mit), ("hust", df_hust)]:
        if df_full.empty:
            continue
        first_cell = df_full["cell_id"].iloc[0]
        first_cyc  = int(df_full[df_full["cell_id"] == first_cell]["cycle"].min())
        mask       = (df_full["cell_id"] == first_cell) & (df_full["cycle"] == first_cyc)
        sample_row = df_full[mask]

        _to_cycle_df(sample_row).to_csv(
            sample_dir / f"{ds_tag}_hi_cycle{first_cyc}.csv", index=False)
        _to_seg_df(sample_row).to_csv(
            sample_dir / f"{ds_tag}_hi_seg{first_cyc}.csv",   index=False)

    print(f"  샘플 CSV: {sample_dir}")


def _save_per_cell_hi(df: pd.DataFrame, dataset: str) -> tuple:
    """평탄 HI DataFrame → cycle / seg 두 가지 형식으로 셀별 pkl 저장.

    Returns:
        (df_cycle, df_seg) — 이후 샘플 CSV 생성에 사용
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_cycle = _to_cycle_df(df)
    df_seg   = _to_seg_df(df)

    cycle_dir = HI_ROOT / "cycle" / dataset
    seg_dir   = HI_ROOT / "seg"   / dataset
    cycle_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    for cell_id, grp in df_cycle.groupby("cell_id"):
        grp.reset_index(drop=True).to_pickle(cycle_dir / f"{cell_id}.pkl")
    for cell_id, grp in df_seg.groupby("cell_id"):
        grp.reset_index(drop=True).to_pickle(seg_dir / f"{cell_id}.pkl")

    n = df["cell_id"].nunique()
    print(f"  사이클 HI 저장: {cycle_dir}  ({n}개 셀)")
    print(f"  세그먼트 HI 저장: {seg_dir}  ({n}개 셀)")
    return df_cycle, df_seg


def load_or_extract(cache_path: Path = CACHE_PATH,
                    n_workers: int = 4,
                    force: bool = False) -> pd.DataFrame:
    """캐시가 있으면 로드, 없으면 전체 추출 후 저장."""
    if not force and cache_path.exists():
        print(f"  캐시 로드: {cache_path}")
        return pd.read_pickle(cache_path)

    print("=== MIT HI 추출 ===")
    df_mit  = load_all(MIT_DIR,  n_workers=n_workers)
    dc_mit,  ds_mit  = _save_per_cell_hi(df_mit,  "MIT")
    print("=== HUST HI 추출 ===")
    df_hust = load_all(HUST_DIR, n_workers=n_workers)
    dc_hust, ds_hust = _save_per_cell_hi(df_hust, "HUST")
    _save_sample_csvs(df_mit, df_hust)
    df = pd.concat([df_mit, df_hust], ignore_index=True)
    print(f"  총 사이클: MIT {len(df_mit):,}  /  HUST {len(df_hust):,}")
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
        for hi in ALL_HI_KEYS:
            if hi not in sub.columns:
                rhos[hi] = np.nan; continue
            valid = sub[[hi, "capacity_Ah"]].dropna()
            rhos[hi] = (
                spearmanr(valid[hi], valid["capacity_Ah"])[0]
                if len(valid) > 30 else np.nan
            )
        result[ds] = rhos
    return pd.DataFrame(result, index=ALL_HI_KEYS)


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def _draw_heatmap(ax, keys, title, corr_df, datasets=("MIT", "HUST")):
    """단일 히트맵. |ρ| 평균 내림차순 정렬."""
    avail = [k for k in keys if k in corr_df.index]
    if not avail:
        ax.set_title(title, fontsize=8); ax.axis("off"); return None, []
    order = (
        corr_df.loc[avail].abs().mean(axis=1)
        .fillna(0).sort_values(ascending=False).index.tolist()
    )
    hm = corr_df.loc[order, list(datasets)].values

    im = ax.imshow(hm.T, aspect="auto", cmap="RdYlGn",
                   vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([HI_LABELS.get(k, k) for k in order],
                       rotation=38, ha="right", fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(datasets, fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=8, pad=4, fontweight="bold")
    for xi, k in enumerate(order):
        for yi, ds in enumerate(datasets):
            val = hm[xi, yi]
            txt = f"{val:.2f}" if np.isfinite(val) else "N/A"
            ax.text(xi, yi, txt, ha="center", va="center",
                    fontsize=6,
                    color="white" if abs(val) > 0.65 else "black",
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

    # ── 레이아웃: Global + 6 segment rows + scatter ────────────────────────
    # 각 세그먼트 행: [Stat | Diff | LFP] 3 sub-panels
    fig = plt.figure(figsize=(44, 56))
    fig.suptitle(
        "Health Indicator Spearman ρ  ─  285 HIs  (Global 15 + Segment 270)",
        fontsize=13, fontweight="bold", y=0.999,
    )
    gs_main = gridspec.GridSpec(
        8, 1, figure=fig,
        height_ratios=[1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        hspace=0.60,
    )

    # ── 행 0: Global ──────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs_main[0])
    im0, _ = _draw_heatmap(ax0, HI_GROUPS["Global"],
                           "Global  (15 HIs)", corr_df)

    # ── 행 1–6: 6 세그먼트 각각 3 sub-panels ─────────────────────────────
    seg_rows = [
        ("dis_hi",  "dis_hi — SoC 60–100%  (방전 초반)",   1),
        ("dis_mid", "dis_mid — SoC 30–60%  (플래토 중심)", 2),
        ("dis_lo",  "dis_lo — SoC 0–30%   (방전 후반)",    3),
        ("chg_lo",  "chg_lo — SoC 0–40%   (충전 초반)",    4),
        ("chg_mid", "chg_mid — SoC 40–70% (플래토 중심)",  5),
        ("chg_hi",  "chg_hi — SoC 70–100% (충전 후반·CV)", 6),
    ]
    ref_im = im0
    for seg, seg_title, row_idx in seg_rows:
        gs_seg = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_main[row_idx], wspace=0.06)
        for ci, cat in enumerate(["Stat", "Diff", "LFP"]):
            ax_s = fig.add_subplot(gs_seg[ci])
            im_s, _ = _draw_heatmap(
                ax_s,
                HI_GROUPS[f"{seg} — {cat}"],
                f"{seg_title}  [{cat}]",
                corr_df,
            )
            if im_s is not None and ref_im is None:
                ref_im = im_s

    # ── 공유 컬러바 ──────────────────────────────────────────────────────
    if ref_im is not None:
        cbar = plt.colorbar(ref_im, ax=fig.get_axes()[:7], shrink=0.25, pad=0.01)
        cbar.set_label("Spearman ρ", fontsize=10)

    # ── 행 7: 상위 HI 산점도 ─────────────────────────────────────────────
    abs_mean = corr_df.abs().mean(axis=1).fillna(0).sort_values(ascending=False)
    top_his  = abs_mean.index[:n_top].tolist()

    gs_sc = gridspec.GridSpecFromSubplotSpec(
        2, n_top, subplot_spec=gs_main[7], hspace=0.52, wspace=0.30)
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
            ax.set_title(f"{lbl}  [{ds}]\n[{tag}]  {rho_str}", fontsize=7, pad=3)
            ax.set_xlabel(lbl, fontsize=6)
            ax.set_ylabel("Capacity (Ah)", fontsize=6)
            ax.tick_params(labelsize=5)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


def _plot_sample_hi(df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path) -> None:
    """대표 셀 상위 HI 사이클 추이."""
    SAMPLES = {"MIT": "b1c0", "HUST": "1-1"}
    CMAPS   = {"MIT": "Blues", "HUST": "Oranges"}

    df_p = df.copy()
    df_p["dataset"] = df_p["dataset"].replace("MIT_MAT", "MIT")

    abs_mean = corr_df.abs().mean(axis=1).fillna(0).sort_values(ascending=False)
    top4     = abs_mean.index[:4].tolist()
    n_ds     = len(SAMPLES)

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
                        transform=ax.transAxes, fontsize=9); continue
            valid = sub[["cycle", hi_key, "capacity_Ah"]].dropna()
            if len(valid) < 3:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9); continue
            cap_range = valid["capacity_Ah"].max() - valid["capacity_Ah"].min()
            c_norm = (valid["capacity_Ah"] - valid["capacity_Ah"].min()) / max(cap_range, 1e-9)
            ax.scatter(valid["cycle"], valid[hi_key],
                       c=c_norm, cmap=CMAPS[ds], s=8, alpha=0.8)
            rho = corr_df.loc[hi_key, ds] if (
                hi_key in corr_df.index and ds in corr_df.columns) else np.nan
            rho_str = f"ρ={rho:.3f}" if np.isfinite(rho) else "ρ=N/A"
            lbl = HI_LABELS.get(hi_key, hi_key)
            tag = HI_GROUP_TAG.get(hi_key, "")
            title = f"{lbl}  [{tag}]\n{rho_str}" if ri == 0 else f"{lbl}  [{tag}]"
            ax.set_title(title, fontsize=8, fontweight="bold")
            ax.set_xlabel("Cycle", fontsize=7)
            ax.set_ylabel(lbl, fontsize=7)
            ax.tick_params(labelsize=6)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_hi_trend.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    cpu = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description="HI 411종 추출 및 Spearman 상관 시각화")
    parser.add_argument("--workers", type=int, default=min(4, cpu))
    parser.add_argument("--n-top",   type=int, default=4,
                        help="산점도 표시 상위 HI 수 (기본: 4)")
    parser.add_argument("--force",   action="store_true",
                        help="캐시 무시하고 HI 재추출")
    # ── 플래토 디버그 모드 ──────────────────────────────────────────────────
    parser.add_argument("--plateau-debug", action="store_true",
                        help="단일 사이클 플래토 판정 디버그 플롯 생성 후 종료")
    parser.add_argument("--plateau-summary", action="store_true",
                        help="전체 데이터 plateau_frac 요약 플롯 생성 후 종료")
    parser.add_argument("--dataset", type=str, default="MIT",
                        help="데이터셋 (MIT 또는 HUST, 기본: MIT)")
    parser.add_argument("--cell",    type=str, default="",
                        help="셀 ID (pkl 파일명 기준, 예: CH-Bat-000)")
    parser.add_argument("--cycle",   type=int, default=0,
                        help="시각화할 사이클 번호 (0이면 첫 유효 사이클)")
    args = parser.parse_args()

    # ── 플래토 디버그 단독 실행 ─────────────────────────────────────────────
    if args.plateau_debug:
        if not args.cell:
            root = MIT_DIR if args.dataset.upper().startswith("MIT") else HUST_DIR
            pkls = sorted(root.glob("*.pkl"))
            if not pkls:
                print(f"[plateau_debug] {root} 에 pkl 파일 없음"); return
            args.cell = pkls[0].stem
            print(f"[plateau_debug] --cell 미지정 → 첫 셀 사용: {args.cell}")
        _run_plateau_debug(args.dataset, args.cell, args.cycle)
        return

    # ── plateau_frac 전체 요약 플롯 ────────────────────────────────────────
    if args.plateau_summary:
        print("\n=== Plateau fraction 전체 요약 플롯 ===")
        df_s = load_or_extract(n_workers=args.workers, force=args.force)
        out_sum = STEP_DIR / "outputs" / "plateau_summary.png"
        plot_plateau_fraction_summary(df_s, out_path=out_sum)
        print("완료!")
        return

    df = load_or_extract(n_workers=args.workers, force=args.force)
    print(f"\n총 사이클: {len(df):,}")

    print("\n=== Spearman ρ 계산 ===")
    corr = compute_correlations(df)

    for gname, gkeys in HI_GROUPS.items():
        avail = [k for k in gkeys if k in corr.index]
        if not avail:
            continue
        sub = corr.loc[avail].copy()
        sub["|ρ| avg"] = sub.abs().mean(axis=1)
        sub = sub.sort_values("|ρ| avg", ascending=False)
        print(f"\n── {gname} ──")
        print(sub.to_string(float_format=lambda x: f"{x:+.3f}"))

    hi_plot_dir = STEP_DIR / "hi_plot" / date.today().strftime("%m%d")
    hi_plot_dir.mkdir(parents=True, exist_ok=True)
    out = hi_plot_dir / "hi_correlation.png"
    print(f"\n=== Plot 저장: {out} ===")
    plot_correlation(corr, df, out, n_top=args.n_top)

    out_dir = STEP_DIR / "outputs" / date.today().strftime("%m%d")
    print("\n=== 대표 셀 HI 플롯 ===")
    _plot_sample_hi(df, corr, out_dir)
    print("완료!")


if __name__ == "__main__":
    main()
