"""
profile_hi_timing.py  —  HI 카테고리·피처별 계산 시간 프로파일링

출력:
  4_hi_analysis/hi_profile/hi_timing_category.png  — 카테고리별 평균 소요시간
  4_hi_analysis/hi_profile/hi_timing_feature.png   — 피처별 평균 소요시간 (컨셉 단위)

사용:
  python 4_hi_analysis/profile_hi_timing.py
  python 4_hi_analysis/profile_hi_timing.py --n-cells 10 --n-cycles 40
"""

import argparse
import pickle
import random
import sys
from pathlib import Path
from time import perf_counter as _pc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from hi_correlation import (
    ALL_SEGS, CHG_SEGS, DIS_SEGS,
    DIFF_KEYS, LFP_KEYS, MORPH_KEYS, STAT_KEYS,
    HI_LABELS,
    _build_ica_seg, _build_vq_curve,
    _dtw_distance, _frechet_distance,
    _global_dva, _global_ica,
    _peak_fwhm_asym, _r_dc_from_chg,
    _seg_diff, _seg_lfp, _seg_morph_curves, _seg_stat,
    HUST_DIR, MIT_DIR,
)

THETA_FLAT = 0.05  # _seg_lfp 와 동일한 플래토 임계값

# 카테고리 색상
CAT_COLORS = {
    "Global": "#555555",
    "Stat":   "#2980b9",
    "Diff":   "#e67e22",
    "LFP":    "#27ae60",
    "Morph":  "#8e44ad",
}

# ── 컨셉→레이블 매핑 (세그먼트 무관, dis_hi 기준) ─────────────────────────────
_REF_SEG = "dis_hi"
CONCEPT_LABEL: dict[str, str] = {}
for _k in STAT_KEYS:
    CONCEPT_LABEL[f"stat_{_k}"] = HI_LABELS.get(f"stat_{_k}_{_REF_SEG}", f"stat_{_k}")
for _k in DIFF_KEYS:
    CONCEPT_LABEL[f"diff_{_k}"] = HI_LABELS.get(f"diff_{_k}_{_REF_SEG}", f"diff_{_k}")
for _k in LFP_KEYS:
    CONCEPT_LABEL[f"lfp_{_k}"]  = HI_LABELS.get(f"lfp_{_k}_{_REF_SEG}",  f"lfp_{_k}")
for _k in MORPH_KEYS:
    CONCEPT_LABEL[f"morph_{_k}"] = HI_LABELS.get(f"morph_{_k}_{_REF_SEG}", f"morph_{_k}")


def _cat(concept: str) -> str:
    p = concept.split("_")[0]
    return {"stat": "Stat", "diff": "Diff", "lfp": "LFP", "morph": "Morph"}.get(p, "Global")


# ─────────────────────────────────────────────────────────────────────────────
# 개별 피처 타이밍 함수
# (공유 전처리: _build_vq_curve / _build_ica_seg 결과를 받아
#  각 HI 계산만 측정 → 마진 비용(marginal cost) 측정)
# ─────────────────────────────────────────────────────────────────────────────

def _time_stat(vs, ims, dts, qcs, seg) -> dict[str, float]:
    """카테고리 A — S01~S15 개별 타이밍 [초]."""
    T: dict[str, float] = {}
    n = len(vs)
    q_rel = qcs - qcs[0]

    t = _pc(); denom = float(np.sum(ims * dts))
    _ = float(np.sum(vs * ims * dts)) / denom if denom > 1e-9 else float(np.mean(vs))
    T[f"stat_v_mean_{seg}"] = _pc() - t

    t = _pc(); _ = float(np.std(vs));               T[f"stat_v_std_{seg}"]  = _pc() - t
    t = _pc()
    if n >= 3: _ = float(sp_skew(vs))
    T[f"stat_v_skew_{seg}"] = _pc() - t

    t = _pc()
    if n >= 4: _ = float(sp_kurtosis(vs))
    T[f"stat_v_kurt_{seg}"] = _pc() - t

    t = _pc()
    _cnt = np.histogram(vs, bins=20)[0].astype(float)
    _tot = _cnt.sum()
    if _tot > 0:
        p = _cnt[_cnt > 0] / _tot
        _ = float(-np.sum(p * np.log(p)))
    T[f"stat_v_ent_{seg}"] = _pc() - t

    t = _pc(); _ = float(np.mean(ims));              T[f"stat_i_mean_{seg}"] = _pc() - t
    t = _pc(); _ = float(np.std(ims));               T[f"stat_i_std_{seg}"]  = _pc() - t
    t = _pc(); _ = float(np.median(vs));             T[f"stat_v_med_{seg}"]  = _pc() - t

    t = _pc()
    if np.std(q_rel) > 1e-9 and np.std(ims) > 1e-9:
        _ = float(np.corrcoef(q_rel, ims)[0, 1])
    T[f"stat_corr_qi_{seg}"] = _pc() - t

    t = _pc()
    if np.std(vs) > 1e-6 and np.std(ims) > 1e-9:
        _ = float(np.corrcoef(vs, ims)[0, 1])
    T[f"stat_corr_vi_{seg}"] = _pc() - t

    t = _pc(); _ = float(np.sum(ims * dts) / 3600.0);    T[f"stat_q_abs_{seg}"]      = _pc() - t
    t = _pc(); _ = float(np.sum(vs * ims * dts) / 3600.0); T[f"stat_energy_seg_{seg}"] = _pc() - t
    t = _pc(); _ = float(np.percentile(vs, 75) - np.percentile(vs, 25)); T[f"stat_v_iqr_{seg}"]   = _pc() - t
    t = _pc(); _ = float(vs.max() - vs.min());            T[f"stat_v_range_{seg}"]    = _pc() - t
    t = _pc(); _ = float(np.percentile(vs, 10));          T[f"stat_v_p10_{seg}"]      = _pc() - t

    # S16 v_p90
    t = _pc(); _ = float(np.percentile(vs, 90));          T[f"stat_v_p90_{seg}"]      = _pc() - t

    # S17 v_samp_ent (SampEn vectorized)
    t = _pc()
    if n >= 10:
        r_t = 0.2 * float(np.std(vs))
        if r_t > 0:
            xs = vs[::max(1, n // 200)] if n > 200 else vs
            ns = len(xs)
            if ns >= 10:
                w2 = np.column_stack([xs[:-1], xs[1:]])
                w3 = np.column_stack([xs[:-2], xs[1:-1], xs[2:]])
                c2 = np.max(np.abs(w2[:, None, :] - w2[None, :, :]), axis=2)
                c3 = np.max(np.abs(w3[:, None, :] - w3[None, :, :]), axis=2)
                np.fill_diagonal(c2, np.inf)
                np.fill_diagonal(c3, np.inf)
                B_se = int(np.sum(c2 <= r_t))
                A_se = int(np.sum(c3 <= r_t))
    T[f"stat_v_samp_ent_{seg}"] = _pc() - t

    # S18 corr_vt, S20 v_detrended_std — shared t_seg
    t = _pc()
    t_s = np.zeros(n); t_s[1:] = np.cumsum(dts[1:])
    t_tot = float(t_s[-1])
    if t_tot > 0 and np.std(vs) > 1e-6:
        _ = float(np.corrcoef(vs, t_s / t_tot)[0, 1])
    T[f"stat_corr_vt_{seg}"] = _pc() - t

    t = _pc()
    if t_tot > 0:
        A_ = np.column_stack([t_s / t_tot, np.ones(n)])
        coef_ = np.linalg.lstsq(A_, vs, rcond=None)[0]
        _ = float(np.std(vs - A_ @ coef_))
    T[f"stat_v_detrended_std_{seg}"] = _pc() - t

    # S19 i_q_slope
    t = _pc()
    if np.std(q_rel) > 1e-9:
        A_ = np.column_stack([q_rel, np.ones(n)])
        _ = np.linalg.lstsq(A_, ims, rcond=None)[0][0]
    T[f"stat_i_q_slope_{seg}"] = _pc() - t

    return T


def _time_diff(vs, ims, dts, qcs, seg,
               qm, v_sm, dvdq_sm, q_tot,
               vmids_ica, dqdv_sm_ica) -> dict[str, float]:
    """카테고리 B — D01~D15 개별 타이밍 (전처리 완료 후 마진 비용)."""
    T: dict[str, float] = {}
    n = len(vs)
    fin = np.isfinite(dvdq_sm)
    vd  = dvdq_sm[fin] if fin.any() else np.array([])

    # D01–D05
    t = _pc()
    if len(vd) >= 3: _ = float(np.mean(vd))
    T[f"diff_dvdq_mean_{seg}"] = _pc() - t

    t = _pc()
    if len(vd) >= 3: _ = float(np.std(vd))
    T[f"diff_dvdq_std_{seg}"] = _pc() - t

    t = _pc()
    if len(vd) >= 3: _ = float(np.max(np.abs(vd)))
    T[f"diff_dvdq_max_abs_{seg}"] = _pc() - t

    t = _pc()
    if len(vd) >= 3: _ = float(np.min(vd))
    T[f"diff_dvdq_min_{seg}"] = _pc() - t

    t = _pc()
    if len(vd) >= 3: _ = float(np.trapezoid(np.abs(dvdq_sm[fin]), qm[fin]))
    T[f"diff_dvdq_area_{seg}"] = _pc() - t

    # D06–D09, D11
    t = _pc()
    if len(vmids_ica) >= 4: _ = float(np.trapezoid(np.maximum(dqdv_sm_ica, 0), vmids_ica))
    T[f"diff_dqdv_area_{seg}"] = _pc() - t

    t = _pc()
    if len(vmids_ica) >= 4:
        pk = int(np.argmax(dqdv_sm_ica))
        if dqdv_sm_ica[pk] > 0:
            _ = float(dqdv_sm_ica[pk])
    T[f"diff_dqdv_peak_h_{seg}"] = _pc() - t

    t = _pc()
    if len(vmids_ica) >= 4:
        pk = int(np.argmax(dqdv_sm_ica))
        if dqdv_sm_ica[pk] > 0:
            _ = float(vmids_ica[pk])
    T[f"diff_dqdv_peak_v_{seg}"] = _pc() - t

    t = _pc()
    if len(vmids_ica) >= 4:
        pk = int(np.argmax(dqdv_sm_ica))
        if dqdv_sm_ica[pk] > 0:
            fwhm, _ = _peak_fwhm_asym(dqdv_sm_ica, pk, vmids_ica)
    T[f"diff_dqdv_peak_w_{seg}"] = _pc() - t

    t = _pc()
    if len(vmids_ica) >= 4:
        pk = int(np.argmax(dqdv_sm_ica))
        if dqdv_sm_ica[pk] > 0:
            _, asym = _peak_fwhm_asym(dqdv_sm_ica, pk, vmids_ica)
    T[f"diff_dqdv_peak_asym_{seg}"] = _pc() - t

    # D10
    t = _pc()
    dt_tot = float(np.sum(dts))
    if dt_tot >= 1.0: _ = float(vs[-1] - vs[0]) / dt_tot
    T[f"diff_dvdt_slope_{seg}"] = _pc() - t

    # D12 d²V/dQ²
    t = _pc()
    if len(vd) >= 3:
        dq_b = float(qm[1] - qm[0]) if len(qm) > 1 else 1.0
        d2   = np.gradient(dvdq_sm, dq_b)
        fin2 = np.isfinite(d2)
        if fin2.sum() > 0: _ = float(np.sqrt(np.mean(d2[fin2] ** 2)))
    T[f"diff_d2vdq2_rms_{seg}"] = _pc() - t

    # D13 skew
    t = _pc()
    if len(vd) >= 3: _ = float(sp_skew(vd))
    T[f"diff_dvdq_skew_{seg}"] = _pc() - t

    # D14 entropy
    t = _pc()
    if len(vd) >= 3:
        _cnt = np.histogram(np.abs(vd), bins=10)[0].astype(float)
        _tot = _cnt.sum()
        if _tot > 0:
            p = _cnt[_cnt > 0] / _tot
            _ = float(-np.sum(p * np.log(p)))
    T[f"diff_dvdq_ent_{seg}"] = _pc() - t

    # D15 r_dyn
    t = _pc()
    if n > 1:
        dv_a = np.diff(vs); di_a = np.diff(ims); dt_a = dts[1:]
        valid = (np.abs(di_a) > 0.01) & (dt_a < 2.0) & (dt_a > 0)
        if valid.sum() > 0:
            r_d = np.abs(dv_a[valid] / di_a[valid])
            r_d = r_d[r_d < 1000.0]
            if len(r_d) > 0: _ = float(np.mean(r_d))
    T[f"diff_r_dyn_seg_{seg}"] = _pc() - t

    # D16–D17 IC valley
    t = _pc()
    if len(vmids_ica) >= 6:
        pk16 = int(np.argmax(dqdv_sm_ica))
        pk16_h = float(dqdv_sm_ica[pk16])
        if pk16_h > 0 and pk16 >= 2 and pk16 <= len(dqdv_sm_ica) - 3:
            li = int(np.argmin(dqdv_sm_ica[:pk16]))
            ri = pk16 + 1 + int(np.argmin(dqdv_sm_ica[pk16 + 1:]))
    T[f"diff_dqdv_valley_h_{seg}"] = _pc() - t

    t = _pc()
    T[f"diff_dqdv_valley_v_{seg}"] = _pc() - t  # same ops as valley_h

    # D18–D19 VQ peak/valley Q positions
    t = _pc()
    fin18 = np.isfinite(dvdq_sm)
    if q_tot > 0.005 and fin18.sum() >= 3:
        qmf18 = qm[fin18]; dvf18 = dvdq_sm[fin18]
        _ = float(qmf18[int(np.argmax(np.abs(dvf18)))])
    T[f"diff_dvdq_peak_q_{seg}"] = _pc() - t

    t = _pc()
    if q_tot > 0.005 and fin18.sum() >= 3:
        _ = float(qmf18[int(np.argmin(dvf18))])
    T[f"diff_dvdq_valley_q_{seg}"] = _pc() - t

    # D20 IC area asymmetry
    t = _pc()
    if len(vmids_ica) >= 4:
        pk20 = int(np.argmax(dqdv_sm_ica))
        if float(dqdv_sm_ica[pk20]) > 0 and pk20 >= 1 and pk20 <= len(dqdv_sm_ica) - 2:
            al = float(np.trapezoid(np.maximum(dqdv_sm_ica[:pk20 + 1], 0), vmids_ica[:pk20 + 1]))
            ar = float(np.trapezoid(np.maximum(dqdv_sm_ica[pk20:],     0), vmids_ica[pk20:]))
    T[f"diff_dqdv_area_asym_{seg}"] = _pc() - t

    return T


def _time_lfp(vs, ims, dts, qcs, seg,
              qm, v_sm, dvdq_sm, q_tot) -> dict[str, float]:
    """카테고리 C — L01~L15 개별 타이밍 (전처리 완료 후 마진 비용)."""
    T: dict[str, float] = {}
    n  = len(vs)
    n_b = len(qm)
    dq_b = float(qm[1] - qm[0]) if len(qm) > 1 else 1.0
    fin_b = np.isfinite(dvdq_sm) & np.isfinite(v_sm)
    q_mid = q_tot / 2.0
    q_rel = qcs - qcs[0]

    # L01–L04 플래토
    t = _pc()
    plt_mask = fin_b & (np.abs(dvdq_sm) < THETA_FLAT)
    _ = float(plt_mask.sum()) / n_b if n_b > 0 else 0.0
    T[f"lfp_plateau_frac_{seg}"] = _pc() - t

    t = _pc()
    min_plt = max(2, int(0.05 * n_b))
    if plt_mask.sum() >= min_plt:
        _ = float(np.mean(v_sm[plt_mask]))
    T[f"lfp_plateau_v_mean_{seg}"] = _pc() - t

    t = _pc()
    if plt_mask.sum() >= min_plt:
        _ = float(np.std(v_sm[plt_mask]))
    T[f"lfp_plateau_v_std_{seg}"] = _pc() - t

    t = _pc()
    if plt_mask.sum() >= min_plt:
        _ = (float(plt_mask.sum() * dq_b) / q_tot) if q_tot > 0 else np.nan
    T[f"lfp_plateau_q_frac_{seg}"] = _pc() - t

    # L05 nonlin_idx
    t = _pc()
    if fin_b.sum() >= 4:
        v_lin = np.interp(qm, [qm[0], qm[-1]], [v_sm[0], v_sm[-1]])
        v_rng = float(v_sm[fin_b].max() - v_sm[fin_b].min())
        if v_rng > 1e-4:
            _ = float(np.sqrt(np.mean((v_sm[fin_b] - v_lin[fin_b]) ** 2))) / v_rng
    T[f"lfp_nonlin_idx_{seg}"] = _pc() - t

    # L06 v_sag_mid
    t = _pc()
    if fin_b.any():
        v_mid     = float(np.interp(q_mid, qm, v_sm))
        v_lin_mid = float(np.interp(q_mid, [qm[0], qm[-1]], [v_sm[0], v_sm[-1]]))
        _ = v_mid - v_lin_mid
    T[f"lfp_v_sag_mid_{seg}"] = _pc() - t

    # L07 v_flatness
    t = _pc()
    v_rng_raw = float(vs.max() - vs.min())
    if v_rng_raw > 1e-4: _ = 1.0 - float(np.std(vs)) / v_rng_raw
    T[f"lfp_v_flatness_{seg}"] = _pc() - t

    # L08 delta_v_rms
    t = _pc()
    if n > 1:
        slow = dts[1:] >= 1.0
        if slow.sum() > 0:
            _ = float(np.sqrt(np.mean(np.diff(vs)[slow] ** 2)))
    T[f"lfp_delta_v_rms_{seg}"] = _pc() - t

    # L09 ocv_slope
    t = _pc()
    if fin_b.any(): _ = float(np.interp(q_mid, qm, dvdq_sm))
    T[f"lfp_ocv_slope_{seg}"] = _pc() - t

    # L10–L11 knee
    t = _pc()
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
    T[f"lfp_knee_v_{seg}"] = _pc() - t

    t = _pc()
    # (knee_q_frac는 knee_v와 동일 연산, 별도 타이밍)
    if fin_b.sum() >= 6 and n_b >= 6:
        d2 = np.gradient(dvdq_sm, dq_b)
        ws11 = min(11, n_b - (1 - n_b % 2))
        ws11 = max(3, ws11 if ws11 % 2 == 1 else ws11 - 1)
        try:
            d2_sm = savgol_filter(d2, ws11, min(2, ws11 - 1))
        except Exception:
            d2_sm = d2
        sc = np.where(np.diff(np.sign(d2_sm)) != 0)[0]
    T[f"lfp_knee_q_frac_{seg}"] = _pc() - t

    # L12 v_concavity
    t = _pc()
    if n >= 10:
        denom_cw = float(np.sum(ims * dts))
        _ = (float(np.sum(vs * ims * dts)) / denom_cw if denom_cw > 1e-9 else float(np.mean(vs)))
        _ = _ - (float(vs[0]) + float(vs[-1])) / 2.0
    T[f"lfp_v_concavity_{seg}"] = _pc() - t

    # L13 phase_entry_dvdq
    t = _pc()
    n5 = max(1, int(0.05 * n_b))
    if fin_b[:n5].sum() > 0:
        _ = float(np.mean(np.abs(dvdq_sm[:n5][fin_b[:n5]])))
    T[f"lfp_phase_entry_dvdq_{seg}"] = _pc() - t

    # L14 v_q_pearson
    t = _pc()
    if np.std(vs) > 1e-6 and np.std(q_rel) > 1e-9:
        _ = float(np.corrcoef(vs, q_rel)[0, 1])
    T[f"lfp_v_q_pearson_{seg}"] = _pc() - t

    # L15 ica_peak_cnt (ICA 재계산 포함)
    t = _pc()
    vmids_ica2, dqdv_ica2 = _build_ica_seg(vs, ims, dts)
    if len(vmids_ica2) >= 4:
        try:
            pks, _ = find_peaks(dqdv_ica2, height=0)
        except Exception:
            pass
    T[f"lfp_ica_peak_cnt_{seg}"] = _pc() - t

    # L16 plateau_v_slope
    t = _pc()
    if plt_mask.sum() >= 5:
        qp16 = qm[plt_mask]; vp16 = v_sm[plt_mask]
        if len(qp16) > 1 and float(qp16[-1] - qp16[0]) > 1e-9:
            A16 = np.column_stack([qp16, np.ones(len(qp16))])
            _ = np.linalg.lstsq(A16, vp16, rcond=None)[0][0]
    T[f"lfp_plateau_v_slope_{seg}"] = _pc() - t

    # L17 v_gradient_exit
    t = _pc()
    n5e = max(1, int(0.05 * n_b))
    exit_mask = np.zeros(n_b, dtype=bool)
    exit_mask[max(0, n_b - n5e):] = True
    valid_exit = exit_mask & fin_b
    if valid_exit.sum() >= 1:
        _ = float(np.mean(np.abs(dvdq_sm[valid_exit])))
    T[f"lfp_v_gradient_exit_{seg}"] = _pc() - t

    # L18 plateau_q_onset
    t = _pc()
    plt_idx18 = np.where(plt_mask)[0]
    if len(plt_idx18) > 0 and q_tot > 0:
        _ = float(qm[plt_idx18[0]]) / q_tot
    T[f"lfp_plateau_q_onset_{seg}"] = _pc() - t

    # L19 dv_dt_plateau
    t = _pc()
    if plt_mask.sum() >= 2 and q_tot > 0 and n > 1:
        q_lo19 = float(qm[plt_mask][0]) - dq_b / 2
        q_hi19 = float(qm[plt_mask][-1]) + dq_b / 2
        raw19  = (q_rel >= q_lo19) & (q_rel <= q_hi19)
        if raw19.sum() >= 3:
            dts_p19 = dts[raw19]
            slow19  = dts_p19[1:] >= 1.0
            if slow19.sum() >= 3:
                _ = float(np.mean(
                    np.abs(np.diff(vs[raw19])[slow19] / dts_p19[1:][slow19])
                )) * 1000.0
    T[f"lfp_dv_dt_plateau_{seg}"] = _pc() - t

    # L20 v_ent_plateau
    t = _pc()
    if plt_mask.sum() >= 10:
        _cnt_p = np.histogram(v_sm[plt_mask], bins=10)[0].astype(float)
        _tot_p = _cnt_p.sum()
        if _tot_p > 0:
            p_v = _cnt_p[_cnt_p > 0] / _tot_p
            _ = float(-np.sum(p_v * np.log(p_v)))
    T[f"lfp_v_ent_plateau_{seg}"] = _pc() - t

    return T


def _time_morph(mc, bol_ref, seg) -> dict[str, float]:
    """카테고리 D — M01~M06 개별 타이밍 (곡선 사전 계산된 상태에서 마진 비용만 측정)."""
    T: dict[str, float] = {}

    for ci, curve_type in enumerate(("vt", "vq", "ve")):
        arr  = mc[ci]
        bol  = bol_ref.get(curve_type) if bol_ref else None

        t = _pc()
        if arr is not None and bol is not None:
            _ = _dtw_distance(arr, bol)
        T[f"morph_{curve_type}_dtw_{seg}"] = _pc() - t

        t = _pc()
        if arr is not None and bol is not None:
            _ = _frechet_distance(arr, bol)
        T[f"morph_{curve_type}_frec_{seg}"] = _pc() - t

    return T


def _time_global(v, i_mag, dt, q_local, vc=None, ic=None, dtc=None) -> dict[str, float]:
    """Global HI 블록별 타이밍."""
    T: dict[str, float] = {}

    # G01–G03
    t = _pc()
    _ = float(np.sum(i_mag * dt) / 3600.0)
    T["q_dis"] = _pc() - t

    t = _pc()
    _ = float(np.sum(v * i_mag * dt) / 3600.0)
    T["energy_dis"] = _pc() - t

    t = _pc()
    denom = float(np.sum(i_mag * dt))
    if denom > 1e-9: _ = float(np.sum(v * i_mag * dt)) / denom
    T["v_mean_dis"] = _pc() - t

    # G05 q_plateau_frac
    t = _pc()
    mask_plt = (v >= 3.10) & (v <= 3.45)
    if q_local > 0:
        _ = float(np.sum(i_mag[mask_plt] * dt[mask_plt]) / 3600.0) / q_local
    T["q_plateau_frac"] = _pc() - t

    # G06–G08, G15 ICA
    t = _pc()
    _ = _global_ica(v, i_mag, dt)
    T["ica_peak1_[v+h+area+asym]"] = _pc() - t

    # G09–G10 DVA
    t = _pc()
    _ = _global_dva(v, i_mag, dt, q_local)
    T["dva_valley_[q+depth]"] = _pc() - t

    # G04, G11–G14 (충전 필요)
    if vc is not None and ic is not None and dtc is not None:
        t = _pc()
        _ = _r_dc_from_chg(vc, ic, dtc)
        T["r_dc_est"] = _pc() - t

        q_tc = float(np.sum(ic * dtc) / 3600.0)

        t = _pc()
        _ = _global_ica(vc, ic, dtc)
        T["chg_ica_peak1_h"] = _pc() - t

        t = _pc()
        i_mx = float(np.max(ic))
        if i_mx > 0:
            cv_mask = ic < 0.80 * i_mx
            _ = float(np.sum(ic[cv_mask] * dtc[cv_mask]) / 3600.0)
        T["cv_q_frac"] = _pc() - t

        t = _pc()
        if i_mx > 0:
            cv_mask = ic < 0.80 * i_mx
            _ = float(np.sum(dtc[cv_mask]))
        T["cv_time_frac"] = _pc() - t

        t = _pc()
        _ = q_local / q_tc if q_tc > 0 else np.nan
        T["ce"] = _pc() - t

    return T


# ─────────────────────────────────────────────────────────────────────────────
# 전처리 공유 비용 타이밍
# ─────────────────────────────────────────────────────────────────────────────

def _time_preproc(vs, ims, dts) -> dict[str, float]:
    n_bins = max(8, min(30, len(vs) // 3))
    t = _pc(); _ = _build_vq_curve(vs, ims, dts, n_bins=n_bins)
    t_vq = _pc() - t
    t = _pc(); _ = _build_ica_seg(vs, ims, dts)
    t_ica = _pc() - t
    t = _pc(); _ = _seg_morph_curves(vs, ims, dts)
    t_mc = _pc() - t
    return {"preproc_vq_curve": t_vq, "preproc_ica_seg": t_ica, "preproc_morph_curves": t_mc}


# ─────────────────────────────────────────────────────────────────────────────
# 셀 1개 프로파일링 (top-level → ProcessPoolExecutor 호환)
# ─────────────────────────────────────────────────────────────────────────────

def _profile_one_cell(args: tuple) -> list:
    """셀 1개분 타이밍 레코드를 반환. args = (pkl_path_str, n_cycles, show_cycle_bar)."""
    pkl_path_str, n_cycles_per_cell, show_cycle_bar = args
    pkl_path = Path(pkl_path_str)

    try:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
    except Exception:
        return []

    df_all = raw.get("cycles")
    if df_all is None:
        return []

    cycs = sorted(df_all["cycle"].unique())
    cycs = [c for c in cycs if c > 0]
    if len(cycs) == 0:
        return []

    idxs = np.linspace(0, len(cycs) - 1, min(n_cycles_per_cell, len(cycs)), dtype=int)
    sampled = [cycs[i] for i in idxs]

    if show_cycle_bar:
        from tqdm.auto import tqdm as _tqdm
        cyc_iter = _tqdm(sampled, desc=f"  {pkl_path.stem}", unit="cyc",
                         leave=False, dynamic_ncols=True)
    else:
        cyc_iter = sampled

    records = []
    bol_curves: dict = {}

    for cyc in cyc_iter:
            grp = df_all[df_all["cycle"] == cyc]
            dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
            if len(dis) < 30:
                continue

            v   = dis["voltage_V"].values.astype(float)
            i   = dis["current_A"].values.astype(float)
            t_s = dis["time_s"].values.astype(float)
            dt  = np.clip(np.diff(t_s, prepend=t_s[0]), 0, None)
            i_mag = np.abs(i)
            q_cum = np.cumsum(i_mag * dt) / 3600.0
            q_local = float(q_cum[-1])

            if q_local < 0.05:
                continue

            # Global HI 타이밍
            chg = grp[grp["phase"] == "charge"].sort_values("time_s")
            vc = ic = dtc = None
            if len(chg) >= 20:
                tc_arr  = chg["time_s"].values.astype(float)
                vc      = chg["voltage_V"].values.astype(float)
                ic      = np.abs(chg["current_A"].values.astype(float))
                dtc     = np.clip(np.diff(tc_arr, prepend=tc_arr[0]), 0, None)

            g_times = _time_global(v, i_mag, dt, q_local, vc, ic, dtc)
            for feat, elapsed in g_times.items():
                records.append({"feat_key": feat, "concept": feat,
                                 "category": "Global", "time_us": elapsed * 1e6})

            # 세그먼트별 타이밍
            for q_lo_f, q_hi_f, seg, _ in DIS_SEGS:
                lo   = q_lo_f * q_local
                hi   = q_hi_f * q_local
                m_s  = (q_cum >= lo) & (q_cum < hi)
                if m_s.sum() < 10:
                    continue
                vs_s = v[m_s]; ims_s = i_mag[m_s]; dts_s = dt[m_s]; qcs_s = q_cum[m_s]

                # 전처리 타이밍
                pp = _time_preproc(vs_s, ims_s, dts_s)
                n_bins = max(8, min(30, len(vs_s) // 3))
                qm, v_sm, dvdq_sm, q_tot = _build_vq_curve(vs_s, ims_s, dts_s, n_bins=n_bins)
                vmids_ica, dqdv_ica = _build_ica_seg(vs_s, ims_s, dts_s)
                for pk, elapsed in pp.items():
                    records.append({"feat_key": f"{pk}_{seg}", "concept": pk,
                                     "category": "Preproc", "time_us": elapsed * 1e6})

                # Stat
                for feat, elapsed in _time_stat(vs_s, ims_s, dts_s, qcs_s, seg).items():
                    concept = "_".join(feat.split("_")[:-2])   # strip seg suffix (2 tokens)
                    records.append({"feat_key": feat, "concept": concept,
                                     "category": "Stat", "time_us": elapsed * 1e6})

                # Diff
                for feat, elapsed in _time_diff(
                        vs_s, ims_s, dts_s, qcs_s, seg,
                        qm, v_sm, dvdq_sm, q_tot, vmids_ica, dqdv_ica).items():
                    concept = "_".join(feat.split("_")[:-2])
                    records.append({"feat_key": feat, "concept": concept,
                                     "category": "Diff", "time_us": elapsed * 1e6})

                # LFP
                for feat, elapsed in _time_lfp(
                        vs_s, ims_s, dts_s, qcs_s, seg,
                        qm, v_sm, dvdq_sm, q_tot).items():
                    concept = "_".join(feat.split("_")[:-2])
                    records.append({"feat_key": feat, "concept": concept,
                                     "category": "LFP", "time_us": elapsed * 1e6})

                # Morph (mc는 _time_preproc 에서 이미 계산; 재사용)
                mc_s = _seg_morph_curves(vs_s, ims_s, dts_s)
                if all(c is not None for c in mc_s) and seg not in bol_curves:
                    bol_curves[seg] = dict(zip(("vt", "vq", "ve"), mc_s))
                for feat, elapsed in _time_morph(mc_s, bol_curves.get(seg), seg).items():
                    concept = "_".join(feat.split("_")[:-2])
                    records.append({"feat_key": feat, "concept": concept,
                                     "category": "Morph", "time_us": elapsed * 1e6})

            # 충전 세그먼트
            if vc is not None:
                qcc = np.cumsum(ic * dtc) / 3600.0
                q_tc = float(qcc[-1])
                if q_tc < 0.05:
                    continue
                for q_lo_f, q_hi_f, seg, _ in CHG_SEGS:
                    lo  = q_lo_f * q_tc; hi = q_hi_f * q_tc
                    m_c = (qcc >= lo) & (qcc < hi)
                    if m_c.sum() < 10:
                        continue
                    vs_c = vc[m_c]; ims_c = ic[m_c]; dts_c = dtc[m_c]; qcs_c = qcc[m_c]

                    pp = _time_preproc(vs_c, ims_c, dts_c)
                    n_bins = max(8, min(30, len(vs_c) // 3))
                    qm, v_sm, dvdq_sm, q_tot = _build_vq_curve(vs_c, ims_c, dts_c, n_bins=n_bins)
                    vmids_ica, dqdv_ica = _build_ica_seg(vs_c, ims_c, dts_c)
                    for pk, elapsed in pp.items():
                        records.append({"feat_key": f"{pk}_{seg}", "concept": pk,
                                         "category": "Preproc", "time_us": elapsed * 1e6})

                    for feat, elapsed in _time_stat(vs_c, ims_c, dts_c, qcs_c, seg).items():
                        concept = "_".join(feat.split("_")[:-2])
                        records.append({"feat_key": feat, "concept": concept,
                                         "category": "Stat", "time_us": elapsed * 1e6})
                    for feat, elapsed in _time_diff(
                            vs_c, ims_c, dts_c, qcs_c, seg,
                            qm, v_sm, dvdq_sm, q_tot, vmids_ica, dqdv_ica).items():
                        concept = "_".join(feat.split("_")[:-2])
                        records.append({"feat_key": feat, "concept": concept,
                                         "category": "Diff", "time_us": elapsed * 1e6})
                    for feat, elapsed in _time_lfp(
                            vs_c, ims_c, dts_c, qcs_c, seg,
                            qm, v_sm, dvdq_sm, q_tot).items():
                        concept = "_".join(feat.split("_")[:-2])
                        records.append({"feat_key": feat, "concept": concept,
                                         "category": "LFP", "time_us": elapsed * 1e6})
                    mc_c = _seg_morph_curves(vs_c, ims_c, dts_c)
                    if all(c is not None for c in mc_c) and seg not in bol_curves:
                        bol_curves[seg] = dict(zip(("vt", "vq", "ve"), mc_c))
                    for feat, elapsed in _time_morph(mc_c, bol_curves.get(seg), seg).items():
                        concept = "_".join(feat.split("_")[:-2])
                        records.append({"feat_key": feat, "concept": concept,
                                         "category": "Morph", "time_us": elapsed * 1e6})

    return records   # list[dict]


# ─────────────────────────────────────────────────────────────────────────────
# 프로파일링 진입점 (순차 / 병렬 전환)
# ─────────────────────────────────────────────────────────────────────────────

def profile_cells(pkl_paths: list[Path], n_cycles_per_cell: int,
                  workers: int = 1) -> pd.DataFrame:
    """셀 목록 전체 프로파일링.

    workers=1 : 순차 처리 (사이클 진행바 포함)
    workers>1 : ProcessPoolExecutor 병렬 처리 (셀 단위 진행바)
    """
    from tqdm.auto import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if workers == 1:
        all_records = []
        cell_bar = tqdm(pkl_paths, desc="셀", unit="cell", dynamic_ncols=True)
        for p in cell_bar:
            cell_bar.set_postfix(cell=p.stem)
            all_records.extend(_profile_one_cell((str(p), n_cycles_per_cell, True)))
        return pd.DataFrame(all_records)

    # 병렬 모드
    task_args = [(str(p), n_cycles_per_cell, False) for p in pkl_paths]
    all_records = []
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futs = {exe.submit(_profile_one_cell, a): Path(a[0]).stem for a in task_args}
        bar = tqdm(as_completed(futs), total=len(futs),
                   desc=f"셀 (workers={workers})", unit="cell", dynamic_ncols=True)
        for fut in bar:
            bar.set_postfix(cell=futs[fut])
            try:
                all_records.extend(fut.result())
            except Exception as e:
                print(f"\n[WARN] {futs[fut]}: {e}")
    return pd.DataFrame(all_records)


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def plot_category(df: pd.DataFrame, out_path: Path):
    """(1) 카테고리별 피처 1개당 평균 계산 시간.

    전처리 비용은 해당 카테고리의 피처 수로 나눠 각 피처에 균등 배분.
      Diff/LFP: (vq_curve + ica_seg) / 15 per feature
      Morph:    morph_curves / 6          per feature
      Stat/Global: 전처리 없음
    """
    # ── 컨셉별 평균 시간 ────────────────────────────────────────────────────
    concept_avg = (df[df["category"] != "Preproc"]
                   .groupby(["concept", "category"])["time_us"].mean()
                   .reset_index()
                   .rename(columns={"time_us": "mean_us"}))

    # ── 카테고리별 피처 수·피처 평균 ─────────────────────────────────────────
    n_feat = concept_avg.groupby("category")["concept"].count().rename("n_feat")
    cat_mean = (concept_avg.groupby("category")["mean_us"].mean()
                .reset_index().rename(columns={"mean_us": "mean_us"}))
    cat_mean = cat_mean.join(n_feat, on="category")

    # ── 전처리 오버헤드 (피처 수로 나눠 배분) ────────────────────────────────
    prep_avg = (df[df["category"] == "Preproc"]
                .groupby("concept")["time_us"].mean())
    prep_vq  = float(prep_avg.get("preproc_vq_curve",  0.0))
    prep_ica = float(prep_avg.get("preproc_ica_seg",   0.0))
    prep_mc  = float(prep_avg.get("preproc_morph_curves", 0.0))

    # 피처 1개당 전처리 배분 = 전처리 비용 / 피처 수
    oh_per_feat = {"Diff":  (prep_vq + prep_ica) / 15,
                   "LFP":   (prep_vq + prep_ica) / 15,
                   "Morph": prep_mc / 6}
    cat_mean["prep_us"] = cat_mean["category"].map(oh_per_feat).fillna(0.0)
    cat_mean["total_us"] = cat_mean["mean_us"] + cat_mean["prep_us"]
    cat_mean = cat_mean.sort_values("total_us", ascending=True).reset_index(drop=True)

    colors = [CAT_COLORS.get(c, "#888") for c in cat_mean["category"]]

    # ── 누적 막대: 마진 + 전처리 배분 ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    y = cat_mean["category"]
    bars_m = ax.barh(y, cat_mean["mean_us"],
                     color=colors, alpha=0.85, height=0.55, label="마진 비용")
    bars_p = ax.barh(y, cat_mean["prep_us"], left=cat_mean["mean_us"],
                     color=colors, alpha=0.35, height=0.55, hatch="///",
                     label="전처리 배분 (피처 수 균등)")

    for bar, total, n in zip(bars_m, cat_mean["total_us"], cat_mean["n_feat"]):
        ax.text(total + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{total:.0f} us  (n={n})",
                va="center", ha="left", fontsize=8.5)

    ax.set_xlabel("피처 1개당 평균 계산 시간 (us)", fontsize=10)
    ax.set_title(
        "HI 카테고리별 피처 1개당 평균 계산 시간\n"
        "[마진 비용 + 공유 전처리를 피처 수로 균등 배분]",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8.5, loc="lower right")
    ax.xaxis.grid(True, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


def plot_feature(df: pd.DataFrame, out_path: Path):
    """(2) 피처(컨셉)별 평균 소요시간 순위 (상위 N, 전처리 별도 표시)."""
    # 컨셉별 집계 (Preproc 포함)
    feat_df = df.copy()
    feat_df["label"] = feat_df["concept"].map(
        lambda c: CONCEPT_LABEL.get(c, c.replace("preproc_", "[PRE] "))
    )
    # Preproc은 라벨 교체
    feat_df.loc[feat_df["category"] == "Preproc", "label"] = (
        feat_df.loc[feat_df["category"] == "Preproc", "concept"]
        .map({"preproc_vq_curve": "[PRE] build_vq_curve",
              "preproc_ica_seg":  "[PRE] build_ica_seg",
              "preproc_morph_curves": "[PRE] seg_morph_curves"})
    )

    agg = (feat_df.groupby(["concept", "category", "label"])["time_us"]
           .agg(["mean", "std"])
           .reset_index()
           .sort_values("mean", ascending=False))

    n_show = min(60, len(agg))
    agg = agg.head(n_show).iloc[::-1]   # 역순 → 위가 빠른 것

    colors = [CAT_COLORS.get(c, "#888") if c != "Preproc" else "#aaaaaa"
              for c in agg["category"]]

    fig_h = max(10, n_show * 0.28)
    fig, ax = plt.subplots(figsize=(11, fig_h), constrained_layout=True)
    y_pos = np.arange(len(agg))

    bars = ax.barh(y_pos, agg["mean"].values, xerr=agg["std"].values,
                   color=colors, alpha=0.85, height=0.72,
                   error_kw=dict(ecolor="black", capsize=3, lw=0.8))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg["label"].values, fontsize=8.5)

    # 값 레이블
    std_vals = agg["std"].values
    for i, (bar, val) in enumerate(zip(bars, agg["mean"].values)):
        if val > 0.05:
            ax.text(bar.get_width() + std_vals[i],
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f} us",
                    va="center", ha="left", fontsize=7.5)

    # 범례
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=v, label=k, alpha=0.85)
                      for k, v in CAT_COLORS.items()]
    legend_handles.append(Patch(color="#aaaaaa", label="Preproc (공유 전처리)", alpha=0.85))
    ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right", framealpha=0.85)

    ax.set_xlabel("평균 소요시간 / 세그먼트 (μs)", fontsize=10)
    ax.set_title(
        f"피처(컨셉)별 평균 계산 시간  [마진 비용 기준, 상위 {n_show}개]\n"
        "(에러바=std, 세그먼트·사이클 평균, [PRE]=공유 전처리 비용)",
        fontsize=11, fontweight="bold",
    )
    ax.xaxis.grid(True, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8.5)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 한글 + 특수문자 fallback 체인
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    parser = argparse.ArgumentParser(description="HI 카테고리·피처별 계산 시간 프로파일링")
    parser.add_argument("--n-cells",  type=int, default=8,
                        help="프로파일링할 셀 수 (MIT·HUST 합계, 기본 8)")
    parser.add_argument("--n-cycles", type=int, default=30,
                        help="셀당 샘플링 사이클 수 (기본 30)")
    parser.add_argument("--workers",  type=int, default=1,
                        help="병렬 프로세스 수 (기본 1 = 순차). 2 이상으로 설정 시 ProcessPoolExecutor 사용")
    args = parser.parse_args()

    # PKL 경로 수집
    mit_pkls  = sorted(MIT_DIR.glob("*.pkl"))
    hust_pkls = sorted(HUST_DIR.glob("*.pkl"))

    random.seed(42)
    n_half = args.n_cells // 2
    selected = (
        random.sample(mit_pkls,  min(n_half, len(mit_pkls))) +
        random.sample(hust_pkls, min(args.n_cells - n_half, len(hust_pkls)))
    )
    if not selected:
        print("[ERROR] PKL 파일을 찾을 수 없습니다. _2_data_clean/ 경로를 확인하세요.")
        return

    print(f"프로파일링 대상: {len(selected)}개 셀, 셀당 최대 {args.n_cycles}사이클, workers={args.workers}")

    # 프로파일링
    df = profile_cells(selected, args.n_cycles, workers=args.workers)
    print(f"총 측정 레코드: {len(df):,}개")

    if df.empty:
        print("[ERROR] 측정 결과 없음. 데이터를 확인하세요.")
        return

    # 저장 경로
    out_dir = _HERE / "hi_profile"
    out_dir.mkdir(exist_ok=True)

    # 플롯
    print("\n=== 카테고리별 소요시간 플롯 ===")
    plot_category(df, out_dir / "hi_timing_category.png")

    print("\n=== 피처별 소요시간 플롯 ===")
    plot_feature(df, out_dir / "hi_timing_feature.png")

    # 간단 요약 출력 (플롯과 동일한 기준: 피처 1개당 평균)
    print("\n── 카테고리 요약 (피처 1개당 평균 us, 마진 비용만) ──")
    summary = (df[df["category"] != "Preproc"]
               .groupby(["concept", "category"])["time_us"].mean()
               .reset_index()
               .groupby("category")["time_us"]
               .agg(["mean", "std", "count"])
               .rename(columns={"mean": "mean_us", "std": "std_us", "count": "n_concepts"}))
    print(summary.sort_values("mean_us", ascending=False).to_string())

    print("\n── Top-10 느린 피처 ──")
    top10 = (df[df["category"] != "Preproc"]
             .groupby(["concept", "category"])["time_us"]
             .mean()
             .sort_values(ascending=False)
             .head(10))
    print(top10.to_string())

    print("\n완료!")


if __name__ == "__main__":
    main()
