"""
hi_correlation.py

MIT + HUST data_postprocess 에서 HI 285종을 사이클별로 추출하고
방전 용량(capacity_Ah)과의 Spearman 상관계수를 계산·시각화.

입력 : data_postprocess/MIT/*.pkl, data_postprocess/HUST/*.pkl
출력 : hi_correlation.png
       data_HI/MIT/{cell_id}.pkl
       data_HI/HUST/{cell_id}.pkl
사용 : python hi_correlation.py [--workers N] [--n-top N] [--force]

HI 구조 (docs/NEW_HIS.md 참조)
  Global  (15):  G01–G15
  Segment (270): 6구간 × 45  (통계 S01–S15 / 미분 D01–D15 / LFP L01–L15)
  세그먼트: dis_hi / dis_mid / dis_lo / chg_lo / chg_mid / chg_hi
  키 명명: stat_{k}_{seg} / diff_{k}_{seg} / lfp_{k}_{seg}
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
MIT_DIR      = PROJECT_ROOT / "data_postprocess" / "MIT"
HUST_DIR     = PROJECT_ROOT / "data_postprocess" / "HUST"
CACHE_PATH   = STEP_DIR / "hi_features.pkl"
HI_ROOT      = PROJECT_ROOT / "data_HI"

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
    "q_dis": "Q_dis",     "energy_dis": "E_dis",    "v_mean_dis": "V_dis",
    "r_dc_est": "R_dc",   "q_plateau_frac": "Qplt%",
    "ica_peak1_v": "ICA_V", "ica_peak1_h": "ICA_h", "ica_peak1_area": "ICA_ar",
    "dva_valley_q": "DVA_Q", "dva_valley_depth": "DVA_dp",
    "ce": "CE",           "cv_q_frac": "CV_Q%",      "cv_time_frac": "CV_t%",
    "chg_ica_peak1_h": "cICA_h", "ica_peak1_asym": "ICA_asy",
}

STAT_KEYS = [
    "v_mean", "v_std", "v_skew", "v_kurt", "v_ent",
    "i_mean", "i_std", "v_med", "corr_qi", "corr_vi",
    "q_abs", "energy_seg", "v_iqr", "v_range", "v_p10",
]
_STAT_LABELS = {
    "v_mean": "V_mn",  "v_std": "V_sd",   "v_skew": "V_sk",  "v_kurt": "V_kt",
    "v_ent": "V_en",   "i_mean": "I_mn",  "i_std": "I_sd",   "v_med": "V_md",
    "corr_qi": "r(Q,I)", "corr_vi": "r(V,I)",
    "q_abs": "Q_abs",  "energy_seg": "E_seg",
    "v_iqr": "V_iqr",  "v_range": "V_rng", "v_p10": "V_p10",
}

DIFF_KEYS = [
    "dvdq_mean", "dvdq_std", "dvdq_max_abs", "dvdq_min", "dvdq_area",
    "dqdv_peak_h", "dqdv_peak_v", "dqdv_peak_w", "dqdv_area", "dvdt_slope",
    "dqdv_peak_asym", "d2vdq2_rms", "dvdq_skew", "dvdq_ent", "r_dyn_seg",
]
_DIFF_LABELS = {
    "dvdq_mean": "dVQ_mn", "dvdq_std": "dVQ_sd", "dvdq_max_abs": "|dVQ|mx",
    "dvdq_min": "dVQ_mn",  "dvdq_area": "dVQ_ar",
    "dqdv_peak_h": "ICA_h", "dqdv_peak_v": "ICA_V", "dqdv_peak_w": "ICA_w",
    "dqdv_area": "ICA_ar",  "dvdt_slope": "dVdt",
    "dqdv_peak_asym": "ICA_asy", "d2vdq2_rms": "d2V_rm",
    "dvdq_skew": "dVQ_sk",  "dvdq_ent": "dVQ_en",  "r_dyn_seg": "R_dyn",
}

LFP_KEYS = [
    "plateau_frac", "plateau_v_mean", "plateau_v_std", "plateau_q_frac",
    "nonlin_idx", "v_sag_mid", "v_flatness", "delta_v_rms",
    "ocv_slope", "knee_v", "knee_q_frac", "v_concavity",
    "phase_entry_dvdq", "v_q_pearson", "ica_peak_cnt",
]
_LFP_LABELS = {
    "plateau_frac": "plt%",   "plateau_v_mean": "pltVm",  "plateau_v_std": "pltVs",
    "plateau_q_frac": "pltQ%","nonlin_idx": "NL_idx",     "v_sag_mid": "V_sag",
    "v_flatness": "V_flt",    "delta_v_rms": "dV_rms",    "ocv_slope": "OCV_sl",
    "knee_v": "knee_V",       "knee_q_frac": "knee_Q",    "v_concavity": "V_cnc",
    "phase_entry_dvdq": "ent_dVQ", "v_q_pearson": "r(V,Q)", "ica_peak_cnt": "ICA_n",
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

ALL_HI_KEYS = [k for k, _ in _HI_META]   # 15 + 6×45 = 285
HI_LABELS   = {k: lbl for k, lbl in _HI_META}

HI_GROUPS: "OrderedDict[str, list[str]]" = OrderedDict()
HI_GROUPS["Global"] = GLOBAL_HI_KEYS[:]
for _, _, _seg, _seg_lbl in ALL_SEGS:
    HI_GROUPS[f"{_seg} — Stat"] = [f"stat_{k}_{_seg}" for k in STAT_KEYS]
    HI_GROUPS[f"{_seg} — Diff"] = [f"diff_{k}_{_seg}" for k in DIFF_KEYS]
    HI_GROUPS[f"{_seg} — LFP"]  = [f"lfp_{k}_{_seg}"  for k in LFP_KEYS]

HI_GROUP_TAG = {k: gname for gname, keys in HI_GROUPS.items() for k in keys}


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
    peak_area = float(np.trapz(np.maximum(sub, 0), subv))
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
            out[f"diff_dvdq_area_{seg}"]    = float(np.trapz(np.abs(dvdq_sm[fin]), qm[fin]))
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
        out[f"diff_dqdv_area_{seg}"] = float(np.trapz(np.maximum(dqdv_sm, 0), vmids))
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


def _save_per_cell_hi(df: pd.DataFrame, dataset: str) -> None:
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
    """캐시가 있으면 로드, 없으면 전체 추출 후 저장."""
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
    parser = argparse.ArgumentParser(description="HI 285종 추출 및 Spearman 상관 시각화")
    parser.add_argument("--workers", type=int, default=min(4, cpu))
    parser.add_argument("--n-top",   type=int, default=4,
                        help="산점도 표시 상위 HI 수 (기본: 4)")
    parser.add_argument("--force",   action="store_true",
                        help="캐시 무시하고 HI 재추출")
    args = parser.parse_args()

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
