"""
common/scenario/_curves.py — dV/dQ(DVA) · dQ/dV(ICA) 곡선 계산 헬퍼 (단일 소스).

vqslope 세그멘터(기울기 기반 존 분할)와 hi_correlation.py(세그먼트 HI 추출)가
동일한 곡선 계산을 공유하기 위해 이 모듈로 추출했다. 이전에는 hi_correlation.py에만
있었으나, common/scenario 계층이 이를 참조해야 하므로 4단계 → common 으로 이동했다.

  _build_vq_curve : Q-빈 V-Q 곡선 + dV/dQ(DVA) 스무딩
  _build_ica_seg  : V-빈 dQ/dV(ICA) 스무딩
  _peak_fwhm_asym : ICA 피크 FWHM/비대칭
  THETA_FLAT      : LFP 플래토 판별 임계값 |dV/dQ| < θ_flat

hi_correlation.py 는 하위호환을 위해 이 모듈을 import 해 같은 이름으로 노출한다.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

# LFP 플래토 판별 임계값 [V/Ah] — hi_correlation.THETA_FLAT 과 동일 값(단일 소스).
THETA_FLAT: float = 0.25


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
        # 피크가 배열 끝에 있어 한 쪽 반폭만 측정 가능 → 2×단측 추정
        if left_idx != pk_idx:
            return 2.0 * float(x_arr[pk_idx] - x_arr[left_idx]), np.nan
        if right_idx != pk_idx:
            return 2.0 * float(x_arr[right_idx] - x_arr[pk_idx]), np.nan
        return np.nan, np.nan
    fwhm     = float(x_arr[right_idx] - x_arr[left_idx])
    x_peak   = float(x_arr[pk_idx])
    left_hw  = x_peak - float(x_arr[left_idx])
    right_hw = float(x_arr[right_idx]) - x_peak
    if right_hw < 1e-9:
        return fwhm, np.nan
    return fwhm, left_hw / right_hw


# ---------------------------------------------------------------------------
# CC→CV 전환 검출 — 원래 vwindow.py에 있었으나, 2026-09-24 비-정식 축(q_frac_ref
# 제외) 일괄 삭제 때 vwindow.py 자체는 지우면서 hi_correlation.py --exclude-cv
# 옵션이 축과 무관하게 계속 쓰는 이 함수만 이 공용 모듈로 옮겼다.
# ---------------------------------------------------------------------------

_DEFAULT_CV_V    = 3.60   # CC→CV 전환 전압 임계값 [V]
_DEFAULT_CV_FRAC = 0.80   # CC→CV: I < cv_cc_frac × I_max


def _detect_cv_start(
    v: np.ndarray,
    i: np.ndarray,
    v_thresh: float = _DEFAULT_CV_V,
    cc_frac: float = _DEFAULT_CV_FRAC,
) -> int:
    """CC→CV 전환 시작 인덱스 반환.

    V ≥ v_thresh 이고 I < cc_frac × I_max 를 동시에 만족하는 첫 샘플.
    조건을 만족하는 샘플이 없으면 len(v) 반환 (CV 구간 없음).

    근거: LFP 충전 CV 구간은 (1) 전압이 컷오프에 근접하고 (2) 충전기가
    정전압으로 전환해 전류가 감소하기 시작할 때 진입한다. cc_frac=0.80은
    전류 노이즈에 둔감하면서도 CC→CV 전환을 놓치지 않는 실용적 임계값이다.
    """
    if i is None or len(i) == 0:
        return len(v)
    i_max = float(np.max(i))
    if i_max < 1e-6:
        return len(v)
    cv_mask = (v >= v_thresh) & (i < cc_frac * i_max)
    if not cv_mask.any():
        return len(v)
    return int(np.argmax(cv_mask))
