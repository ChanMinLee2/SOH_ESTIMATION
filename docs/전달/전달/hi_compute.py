"""hi_compute.py — HI computation registry (single source, individual functions).

Every HI is an individual ``@hi`` function taking a ``W`` (the window + its cached
shared terms). Add ``@hi def my_feature(w): ...`` and it joins automatically —
registered, computed, cost-measured, and picked up by the model (n_hi from the data).

Shared heavy intermediates are cached ONCE per window on ``W`` (``W.vq`` = V-Q curve,
``W.ica`` = dQ/dV curve, ``W.q_rel``, ``W.t_norm``) so splitting into per-HI functions
costs no recomputation. This module is self-contained (the curve-building helpers live
here too); hi_correlation.py's segment analysis now IMPORTS these — this is the single
source. ``_seg_stat/_seg_diff/_seg_lfp`` adapters dispatch the individual functions on a
``W.from_segment`` for that consumer.

Groups: stat (20) + diff (20) + lfp (20) + morph (6) = 66. Morph = DTW/Fréchet of the
window's V-t/V-Q/V-E shape vs the cell's fresh-state (BOL) curve; needs a per-cell BOL
reference passed as ``compute_his(..., bol=...)`` (factory data, not a leak). Without it
the 6 morph HIs are NaN (masked).
"""

from __future__ import annotations
import time
from functools import cached_property
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew

if not hasattr(np, "trapezoid"):        # numpy<2.0 호환 (구버전은 np.trapz)
    np.trapezoid = np.trapz

# ── 상수 ──────────────────────────────────────────────────────────────────────
THETA_FLAT = 0.25   # V/Ah — LFP 플래토 판별 임계값 (|dV/dQ| < θ_flat)
_MORPH_GRID = 50    # morph 곡선 보간 그리드 해상도
_DTW_BAND = 5       # Sakoe-Chiba 밴드 (위상 이동 허용폭)
MORPH_KEYS = ["vt_dtw", "vq_dtw", "ve_dtw", "vt_frec", "vq_frec", "ve_frec"]

_HI: dict = {}                     # ordered registry: name → function (definition order)


# ── 공유 곡선 헬퍼 (HI 라이브러리 인프라) ─────────────────────────────────────
def _build_vq_curve(vs, ims, dts, n_bins=None):
    """Q-빈 V-Q 곡선 + dV/dQ 스무딩 (SG window=15). Returns (qm, v_sm, dvdq_sm, q_tot)."""
    q_rel = np.cumsum(ims * dts) / 3600.0
    q_tot = float(q_rel[-1]) if len(q_rel) > 0 else 0.0
    n = len(vs)
    if n_bins is None:
        n_bins = max(8, min(30, n // 3))
    if q_tot < 0.005 or n < 8 or n_bins < 4:
        empty = np.full(max(n_bins, 1), np.nan)
        return empty, empty, empty, q_tot
    dq_b = q_tot / n_bins
    q_e = np.linspace(0.0, q_tot, n_bins + 1)
    qm = (q_e[:-1] + q_e[1:]) / 2
    v_av = np.full(n_bins, np.nan)
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
    """V-빈 dQ/dV 곡선 (ICA) 스무딩. Returns (vmids, dqdv_sm)."""
    vr = float(vs.max() - vs.min()) if len(vs) > 1 else 0.0
    if vr < 0.01 or len(vs) < 8:
        return np.array([]), np.array([])
    v_lo = float(vs.min()) - 0.002
    v_hi = float(vs.max()) + 0.002
    n_b = max(8, min(30, int(vr / 0.01)))
    edges = np.linspace(v_lo, v_hi, n_b + 1)
    dv = edges[1] - edges[0]
    vmids = (edges[:-1] + edges[1:]) / 2
    dqdv = np.zeros(n_b)
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
    """ICA 피크의 FWHM 과 비대칭도 (left_hw / right_hw). Returns (fwhm, asym)."""
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
    fwhm = float(x_arr[right_idx] - x_arr[left_idx])
    x_peak = float(x_arr[pk_idx])
    left_hw = x_peak - float(x_arr[left_idx])
    right_hw = float(x_arr[right_idx]) - x_peak
    if right_hw < 1e-9:
        return fwhm, np.nan
    return fwhm, left_hw / right_hw


def _seg_morph_curves(vs, ims, dts):
    """세그먼트 → (V-t, V-Q, V-E) 3곡선을 [0,1] 정규화 그리드로 보간. Returns (vt, vq, ve)."""
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

    return _interp(t_cum), _interp(q_cum, 1e-4), _interp(e_cum, 1e-7)


def _dtw_distance(a, b):
    """Sakoe-Chiba banded DTW (정규화: / n)."""
    n = len(a)
    d = np.abs(a[:, None] - b[None, :])
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


def _frechet_distance(a, b):
    """이산 Fréchet 거리 — 고정 x-그리드 1D 곡선에서는 max|a-b| 와 동치."""
    return float(np.max(np.abs(a - b)))


def hi(fn):
    """Register an HI. Function name = HI name. Takes a W, returns a float."""
    _HI[fn.__name__] = fn
    return fn


class W:
    """A window's raw rows + lazily-cached shared intermediates (computed once)."""
    def __init__(self, v, i, t):
        self.v = np.asarray(v, float); self.i = np.asarray(i, float); self.t = np.asarray(t, float)

    @cached_property
    def ai(self): return np.abs(self.i)

    @cached_property
    def dt(self): return np.clip(np.diff(self.t, prepend=self.t[0]), 0.0, None)

    @cached_property
    def q(self): return np.cumsum(self.ai * self.dt) / 3600.0

    @cached_property
    def q_rel(self): return self.q - self.q[0]                 # window-local cumulative charge

    @cached_property
    def dQ(self): return float(self.q[-1])

    @cached_property
    def vrange(self): return float(self.v.max() - self.v.min())

    @cached_property
    def t_norm(self):                                          # normalized time [0,1] (S18/S20)
        ts = np.zeros(len(self.dt)); ts[1:] = np.cumsum(self.dt[1:])   # len(dt), not t (from_segment: t=None)
        tt = float(ts[-1])
        return ts / tt if tt > 0 else None

    @cached_property
    def vq(self):                                             # V-Q curve: (qm, v_sm, dvdq_sm, q_tot)
        return _build_vq_curve(self.v, self.ai, self.dt)

    @cached_property
    def ica(self):                                           # dQ/dV curve: (vmids, dqdv_sm)
        return _build_ica_seg(self.v, self.ai, self.dt)

    @classmethod
    def from_segment(cls, vs, ims, dts, qcs):
        """Build a W from pre-computed segment arrays (hi_correlation's segment analysis):
        injects the provided |i|, dt, cumulative-q directly — real dt[0], segment offset."""
        w = cls.__new__(cls)
        w.v = np.asarray(vs, float); w.i = None; w.t = None
        q = np.asarray(qcs, float)
        w.__dict__["ai"] = np.asarray(ims, float)
        w.__dict__["dt"] = np.asarray(dts, float)
        w.__dict__["q"] = q
        w.__dict__["q_rel"] = q - q[0]
        return w


# ═══════════════════════════ A. stat (통계) — 20 ═══════════════════════════

@hi
def stat_v_mean(w):
    if len(w.v) < 5: return np.nan
    den = float(np.sum(w.ai * w.dt))
    return float(np.sum(w.v * w.ai * w.dt)) / den if den > 1e-9 else float(np.mean(w.v))

@hi
def stat_v_std(w):
    return float(np.std(w.v)) if len(w.v) >= 5 else np.nan

@hi
def stat_v_skew(w):
    return float(sp_skew(w.v)) if len(w.v) >= 5 else np.nan

@hi
def stat_v_kurt(w):
    return float(sp_kurtosis(w.v)) if len(w.v) >= 5 else np.nan

@hi
def stat_v_ent(w):
    if len(w.v) < 5: return np.nan
    cnt = np.histogram(w.v, bins=20)[0].astype(float); tot = cnt.sum()
    if tot <= 0: return np.nan
    p = cnt[cnt > 0] / tot
    return float(-np.sum(p * np.log(p)))

@hi
def stat_i_mean(w):
    return float(np.mean(w.ai)) if len(w.v) >= 5 else np.nan

@hi
def stat_i_std(w):
    return float(np.std(w.ai)) if len(w.v) >= 5 else np.nan

@hi
def stat_v_med(w):
    return float(np.median(w.v)) if len(w.v) >= 5 else np.nan

@hi
def stat_corr_qi(w):
    if len(w.v) < 5: return np.nan
    if np.std(w.q_rel) > 1e-9 and np.std(w.ai) > 1e-9:
        return float(np.corrcoef(w.q_rel, w.ai)[0, 1])
    return np.nan

@hi
def stat_corr_vi(w):
    if len(w.v) < 5: return np.nan
    if np.std(w.v) > 1e-6 and np.std(w.ai) > 1e-9:
        return float(np.corrcoef(w.v, w.ai)[0, 1])
    return np.nan

@hi
def stat_q_abs(w):
    return float(np.sum(w.ai * w.dt) / 3600.0) if len(w.v) >= 5 else np.nan

@hi
def stat_energy_seg(w):
    return float(np.sum(w.v * w.ai * w.dt) / 3600.0) if len(w.v) >= 5 else np.nan

@hi
def stat_v_iqr(w):
    return float(np.percentile(w.v, 75) - np.percentile(w.v, 25)) if len(w.v) >= 5 else np.nan

@hi
def stat_v_range(w):
    return float(w.v.max() - w.v.min()) if len(w.v) >= 5 else np.nan

@hi
def stat_v_p10(w):
    return float(np.percentile(w.v, 10)) if len(w.v) >= 5 else np.nan

@hi
def stat_v_p90(w):
    return float(np.percentile(w.v, 90)) if len(w.v) >= 5 else np.nan

@hi
def stat_v_samp_ent(w):
    v = w.v; n = len(v)
    if n < 10: return np.nan
    r_tol = 0.2 * float(np.std(v))
    if r_tol <= 0: return np.nan
    xs = v[::max(1, n // 200)] if n > 200 else v
    ns = len(xs)
    if ns < 10: return np.nan
    w2 = np.column_stack([xs[:-1], xs[1:]])
    w3 = np.column_stack([xs[:-2], xs[1:-1], xs[2:]])
    c2 = np.max(np.abs(w2[:, None, :] - w2[None, :, :]), axis=2)
    c3 = np.max(np.abs(w3[:, None, :] - w3[None, :, :]), axis=2)
    np.fill_diagonal(c2, np.inf); np.fill_diagonal(c3, np.inf)
    B_se = int(np.sum(c2 <= r_tol)); A_se = int(np.sum(c3 <= r_tol))
    return float(-np.log(A_se / B_se)) if (B_se > 0 and A_se > 0) else np.nan

@hi
def stat_corr_vt(w):
    if len(w.v) < 5 or w.t_norm is None or np.std(w.v) <= 1e-6: return np.nan
    return float(np.corrcoef(w.v, w.t_norm)[0, 1])

@hi
def stat_v_detrended_std(w):
    if len(w.v) < 5 or w.t_norm is None: return np.nan
    A = np.column_stack([w.t_norm, np.ones(len(w.v))])
    coef = np.linalg.lstsq(A, w.v, rcond=None)[0]
    return float(np.std(w.v - A @ coef))

@hi
def stat_i_q_slope(w):
    if len(w.v) < 5 or np.std(w.q_rel) <= 1e-9: return np.nan
    A = np.column_stack([w.q_rel, np.ones(len(w.v))])
    return float(np.linalg.lstsq(A, w.ai, rcond=None)[0][0])


# ═══════════════════════════ B. diff (미분) — 20 ═══════════════════════════
# 공유: w.vq = (qm, v_sm, dvdq_sm, q_tot) / w.ica = (vmids, dqdv_sm)

def _vd(w):
    """유효 dV/dQ 값 (q_tot>0.005 & finite, len>=3) 또는 None."""
    qm, v_sm, dvdq_sm, q_tot = w.vq
    if len(w.v) < 8 or q_tot <= 0.005 or not np.any(np.isfinite(dvdq_sm)):
        return None
    fin = np.isfinite(dvdq_sm); vd = dvdq_sm[fin]
    return (qm, dvdq_sm, fin, vd) if len(vd) >= 3 else None

@hi
def diff_dvdq_mean(w):
    r = _vd(w); return float(np.mean(r[3])) if r else np.nan

@hi
def diff_dvdq_std(w):
    r = _vd(w); return float(np.std(r[3])) if r else np.nan

@hi
def diff_dvdq_max_abs(w):
    r = _vd(w); return float(np.max(np.abs(r[3]))) if r else np.nan

@hi
def diff_dvdq_min(w):
    r = _vd(w); return float(np.min(r[3])) if r else np.nan

@hi
def diff_dvdq_area(w):
    r = _vd(w)
    if not r: return np.nan
    qm, dvdq_sm, fin, _ = r
    return float(np.trapezoid(np.abs(dvdq_sm[fin]), qm[fin]))

@hi
def diff_d2vdq2_rms(w):
    r = _vd(w)
    if not r: return np.nan
    qm, dvdq_sm, _, _ = r
    dq_b = float(qm[1] - qm[0]) if len(qm) > 1 else 1.0
    d2 = np.gradient(dvdq_sm, dq_b); fin2 = np.isfinite(d2)
    return float(np.sqrt(np.mean(d2[fin2] ** 2))) if fin2.sum() > 0 else np.nan

@hi
def diff_dvdq_skew(w):
    r = _vd(w); return float(sp_skew(r[3])) if r else np.nan

@hi
def diff_dvdq_ent(w):
    r = _vd(w)
    if not r: return np.nan
    cnt = np.histogram(np.abs(r[3]), bins=10)[0].astype(float); tot = cnt.sum()
    if tot <= 0: return np.nan
    p = cnt[cnt > 0] / tot
    return float(-np.sum(p * np.log(p)))

@hi
def diff_dqdv_area(w):
    vmids, dqdv_sm = w.ica
    return float(np.trapezoid(np.maximum(dqdv_sm, 0), vmids)) if len(vmids) >= 4 else np.nan

@hi
def diff_dqdv_peak_h(w):
    vmids, dqdv_sm = w.ica
    if len(vmids) < 4: return np.nan
    pk = int(np.argmax(dqdv_sm))
    return float(dqdv_sm[pk]) if dqdv_sm[pk] > 0 else np.nan

@hi
def diff_dqdv_peak_v(w):
    vmids, dqdv_sm = w.ica
    if len(vmids) < 4: return np.nan
    pk = int(np.argmax(dqdv_sm))
    return float(vmids[pk]) if dqdv_sm[pk] > 0 else np.nan

@hi
def diff_dqdv_peak_w(w):
    vmids, dqdv_sm = w.ica
    if len(vmids) < 4: return np.nan
    pk = int(np.argmax(dqdv_sm))
    if dqdv_sm[pk] <= 0: return np.nan
    return _peak_fwhm_asym(dqdv_sm, pk, vmids)[0]

@hi
def diff_dqdv_peak_asym(w):
    vmids, dqdv_sm = w.ica
    if len(vmids) < 4: return np.nan
    pk = int(np.argmax(dqdv_sm))
    if dqdv_sm[pk] <= 0: return np.nan
    return _peak_fwhm_asym(dqdv_sm, pk, vmids)[1]

@hi
def diff_dvdt_slope(w):
    dt_tot = float(np.sum(w.dt))
    return float(w.v[-1] - w.v[0]) / dt_tot if (len(w.v) >= 8 and dt_tot >= 1.0) else np.nan

@hi
def diff_r_dyn_seg(w):
    if len(w.v) < 8 or len(w.v) <= 1: return np.nan
    dv = np.diff(w.v); di = np.diff(w.ai); dta = w.dt[1:]
    valid = (np.abs(di) > 0.01) & (dta < 2.0) & (dta > 0)
    if valid.sum() == 0: return np.nan
    r = np.abs(dv[valid] / di[valid]); r = r[r < 1000.0]
    return float(np.mean(r)) if len(r) > 0 else np.nan

@hi
def diff_dqdv_valley_h(w):
    return _dqdv_valley(w)[0]

@hi
def diff_dqdv_valley_v(w):
    return _dqdv_valley(w)[1]

def _dqdv_valley(w):
    """D16–D17: IC 곡선 valley (피크 대비 ≤20%). (h, v) 또는 (nan, nan)."""
    vmids, dqdv_sm = w.ica
    if len(vmids) < 6: return np.nan, np.nan
    pk = int(np.argmax(dqdv_sm)); pkh = float(dqdv_sm[pk])
    if not (pkh > 0 and pk >= 2 and pk <= len(dqdv_sm) - 3): return np.nan, np.nan
    li = int(np.argmin(dqdv_sm[:pk])); ri = pk + 1 + int(np.argmin(dqdv_sm[pk + 1:]))
    lh, rh = float(dqdv_sm[li]), float(dqdv_sm[ri])
    lv, rv = float(vmids[li]), float(vmids[ri])
    lval = lh <= 0.2 * pkh; rval = rh <= 0.2 * pkh
    if not (lval or rval): return np.nan, np.nan
    vpk = float(vmids[pk])
    if lval and rval:
        vh, vv = (lh, lv) if (vpk - lv) >= (rv - vpk) else (rh, rv)
    elif lval:
        vh, vv = lh, lv
    else:
        vh, vv = rh, rv
    return vh, vv

@hi
def diff_dvdq_peak_q(w):
    qm, v_sm, dvdq_sm, q_tot = w.vq
    fin = np.isfinite(dvdq_sm)
    if len(w.v) < 8 or q_tot <= 0.005 or fin.sum() < 3: return np.nan
    return float(qm[fin][int(np.argmax(np.abs(dvdq_sm[fin])))])

@hi
def diff_dvdq_valley_q(w):
    qm, v_sm, dvdq_sm, q_tot = w.vq
    fin = np.isfinite(dvdq_sm)
    if len(w.v) < 8 or q_tot <= 0.005 or fin.sum() < 3: return np.nan
    return float(qm[fin][int(np.argmin(np.abs(dvdq_sm[fin])))])

@hi
def diff_dqdv_area_asym(w):
    vmids, dqdv_sm = w.ica
    if len(vmids) < 4: return np.nan
    pk = int(np.argmax(dqdv_sm))
    if not (float(dqdv_sm[pk]) > 0 and pk >= 1 and pk <= len(dqdv_sm) - 2): return np.nan
    al = float(np.trapezoid(np.maximum(dqdv_sm[:pk + 1], 0), vmids[:pk + 1]))
    ar = float(np.trapezoid(np.maximum(dqdv_sm[pk:], 0), vmids[pk:]))
    return float(al / ar) if (al > 1e-9 and ar > 1e-9) else np.nan


# ═══════════════════════════ C. lfp (LFP 특징) — 20 ═══════════════════════════

def _plateau(w):
    """공유: (qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b) 또는 None (n<8 / q_tot<0.005)."""
    if len(w.v) < 8: return None
    qm, v_sm, dvdq_sm, q_tot = w.vq
    if q_tot < 0.005: return None
    dq_b = float(qm[1] - qm[0]) if len(qm) > 1 else 1.0
    fin_b = np.isfinite(dvdq_sm) & np.isfinite(v_sm)
    plt_mask = fin_b & (np.abs(dvdq_sm) < THETA_FLAT)
    return qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, len(qm)

@hi
def lfp_plateau_frac(w):
    p = _plateau(w)
    if p is None: return np.nan
    *_, plt_mask, dq_b, n_b = p
    return float(plt_mask.sum()) / n_b if n_b > 0 else 0.0

@hi
def lfp_plateau_v_mean(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    if plt_mask.sum() < max(2, int(0.05 * n_b)): return np.nan
    return float(np.mean(v_sm[plt_mask]))

@hi
def lfp_plateau_v_std(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    if plt_mask.sum() < max(2, int(0.05 * n_b)): return np.nan
    return float(np.std(v_sm[plt_mask]))

@hi
def lfp_plateau_q_frac(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    if plt_mask.sum() < max(2, int(0.05 * n_b)): return np.nan
    q_plt = float(plt_mask.sum() * dq_b)
    return q_plt / q_tot if q_tot > 0 else np.nan

@hi
def lfp_nonlin_idx(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    if fin_b.sum() < 4: return np.nan
    v_lin = np.interp(qm, [qm[0], qm[-1]], [v_sm[0], v_sm[-1]])
    v_rng = float(v_sm[fin_b].max() - v_sm[fin_b].min())
    if v_rng <= 1e-4: return np.nan
    return float(np.sqrt(np.mean((v_sm[fin_b] - v_lin[fin_b]) ** 2))) / v_rng

@hi
def lfp_v_sag_mid(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    if not fin_b.any(): return np.nan
    q_mid = q_tot / 2.0
    v_mid = float(np.interp(q_mid, qm, v_sm))
    v_lin_mid = float(np.interp(q_mid, [qm[0], qm[-1]], [v_sm[0], v_sm[-1]]))
    return v_mid - v_lin_mid

@hi
def lfp_v_flatness(w):
    if len(w.v) < 8: return np.nan
    if w.vq[3] < 0.005: return np.nan                          # q_tot guard (group returns early)
    v_rng_raw = float(w.v.max() - w.v.min())
    return 1.0 - float(np.std(w.v)) / v_rng_raw if v_rng_raw > 1e-4 else np.nan

@hi
def lfp_delta_v_rms(w):
    if len(w.v) < 8 or w.vq[3] < 0.005 or len(w.v) <= 1: return np.nan
    slow = w.dt[1:] >= 1.0
    if slow.sum() == 0: return np.nan
    dv = np.diff(w.v)[slow]
    return float(np.sqrt(np.mean(dv ** 2)))

@hi
def lfp_ocv_slope(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    if not fin_b.any(): return np.nan
    return float(np.interp(q_tot / 2.0, qm, dvdq_sm))

@hi
def lfp_knee_v(w):
    return _knee(w)[0]

@hi
def lfp_knee_q_frac(w):
    return _knee(w)[1]

def _knee(w):
    """L10–L11: V-Q 변곡점. (knee_v, knee_q_frac) 또는 (nan, nan)."""
    p = _plateau(w)
    if p is None: return np.nan, np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    if not (fin_b.sum() >= 6 and n_b >= 6): return np.nan, np.nan
    d2 = np.gradient(dvdq_sm, dq_b)
    ws = min(11, n_b - (1 - n_b % 2)); ws = max(3, ws if ws % 2 == 1 else ws - 1)
    try:
        d2_sm = savgol_filter(d2, ws, min(2, ws - 1))
    except Exception:
        d2_sm = d2
    sc = np.where(np.diff(np.sign(d2_sm)) != 0)[0]
    if len(sc) == 0: return np.nan, np.nan
    best = sc[int(np.argmax(np.abs(d2_sm[sc])))]
    return float(v_sm[best]), float(qm[best]) / q_tot

@hi
def lfp_v_concavity(w):
    if len(w.v) < 8 or w.vq[3] < 0.005 or len(w.v) < 10: return np.nan
    den = float(np.sum(w.ai * w.dt))
    v_mean = float(np.sum(w.v * w.ai * w.dt)) / den if den > 1e-9 else float(np.mean(w.v))
    return v_mean - (float(w.v[0]) + float(w.v[-1])) / 2.0

@hi
def lfp_phase_entry_dvdq(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    n5 = max(1, int(0.05 * n_b))
    if fin_b[:n5].sum() == 0: return np.nan
    return float(np.mean(np.abs(dvdq_sm[:n5][fin_b[:n5]])))

@hi
def lfp_v_q_pearson(w):
    if len(w.v) < 8 or w.vq[3] < 0.005: return np.nan
    if np.std(w.v) > 1e-6 and np.std(w.q_rel) > 1e-9:
        return float(np.corrcoef(w.v, w.q_rel)[0, 1])
    return np.nan

@hi
def lfp_ica_peak_cnt(w):
    if len(w.v) < 8 or w.vq[3] < 0.005: return np.nan
    vmids, dqdv = w.ica
    if len(vmids) < 4: return np.nan
    try:
        pks, _ = find_peaks(dqdv, height=0)
        return float(len(pks))
    except Exception:
        return np.nan

@hi
def lfp_plateau_v_slope(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    if plt_mask.sum() < 5: return np.nan
    qp = qm[plt_mask]; vp = v_sm[plt_mask]
    if float(qp[-1] - qp[0]) <= 1e-9: return np.nan
    A = np.column_stack([qp, np.ones(len(qp))])
    return float(np.linalg.lstsq(A, vp, rcond=None)[0][0])

@hi
def lfp_v_gradient_exit(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    n5e = max(1, int(0.05 * n_b))
    exit_mask = np.zeros(n_b, dtype=bool); exit_mask[max(0, n_b - n5e):] = True
    ve = exit_mask & fin_b
    return float(np.mean(np.abs(dvdq_sm[ve]))) if ve.sum() >= 1 else np.nan

@hi
def lfp_plateau_q_onset(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    idx = np.where(plt_mask)[0]
    return float(qm[idx[0]]) / q_tot if (len(idx) > 0 and q_tot > 0) else np.nan

@hi
def lfp_dv_dt_plateau(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    if not (plt_mask.sum() >= 2 and q_tot > 0 and len(w.v) > 1): return np.nan
    q_lo = float(qm[plt_mask][0]) - dq_b / 2; q_hi = float(qm[plt_mask][-1]) + dq_b / 2
    raw_in = (w.q_rel >= q_lo) & (w.q_rel <= q_hi)
    if raw_in.sum() < 3: return np.nan
    vs_p = w.v[raw_in]; dts_p = w.dt[raw_in]; slow_p = dts_p[1:] >= 1.0
    if slow_p.sum() < 3: return np.nan
    dvdt = np.abs(np.diff(vs_p)[slow_p] / dts_p[1:][slow_p])
    return float(np.mean(dvdt)) * 1000.0

@hi
def lfp_v_ent_plateau(w):
    p = _plateau(w)
    if p is None: return np.nan
    qm, v_sm, dvdq_sm, q_tot, fin_b, plt_mask, dq_b, n_b = p
    if plt_mask.sum() < 10: return np.nan
    cnt = np.histogram(v_sm[plt_mask], bins=10)[0].astype(float); tot = cnt.sum()
    if tot <= 0: return np.nan
    p20 = cnt[cnt > 0] / tot
    return float(-np.sum(p20 * np.log(p20)))


# ═══════════════════════════ D. morph (형태 거리) — 6 ═══════════════════════════
# 창 곡선 vs 셀 신품(BOL) 곡선의 DTW/Fréchet. BOL 없으면 NaN.

_MORPH_NAMES = [f"morph_{k}" for k in MORPH_KEYS]        # morph_vt_dtw … morph_ve_frec


def morph_curves(v, i, t) -> dict:
    """A window's (V-t, V-Q, V-E) shape curves on a [0,1] grid — for the BOL reference
    or for distance-to-BOL. Direction-agnostic (uses |i|). None entries if uncomputable."""
    w = v if isinstance(v, W) else W(v, i, t)
    vt, vq, ve = _seg_morph_curves(w.v, w.ai, w.dt)
    return {"vt": vt, "vq": vq, "ve": ve}


def _morph_dict(w, bol) -> dict:
    out = {n: np.nan for n in _MORPH_NAMES}
    if bol is None:
        return out
    cur = morph_curves(w, None, None)
    for ct in ("vt", "vq", "ve"):
        a = cur.get(ct); b = bol.get(ct)
        if a is not None and b is not None and len(a) == len(b):
            try:
                out[f"morph_{ct}_dtw"] = float(_dtw_distance(a, b))
                out[f"morph_{ct}_frec"] = float(_frechet_distance(a, b))
            except Exception:
                pass
    return out


def morph_reference(samples) -> dict:
    """Build a BOL reference curve set ({vt,vq,ve}) by averaging fresh-window shape
    curves — pass fresh-state (early-cycle) windows of ONE direction."""
    acc = {"vt": [], "vq": [], "ve": []}
    for v, i, t in samples:
        cur = morph_curves(v, i, t)
        for ct in acc:
            if cur[ct] is not None:
                acc[ct].append(cur[ct])
    return {ct: (np.mean(arrs, axis=0) if arrs else None) for ct, arrs in acc.items()}


# ═══════════════════════════ 세그먼트 어댑터 (hi_correlation 분석용) ═══════════════════════════
# 개별 @hi 함수를 세그먼트 배열에 적용해 seg-접미사 키로 반환 (옛 그룹 함수 대체).

def _seg_group(prefix, vs, ims, dts, qcs, seg):
    w = W.from_segment(vs, ims, dts, qcs)
    out = {}
    for name, fn in _HI.items():
        if name.startswith(prefix):
            try:
                out[f"{name}_{seg}"] = float(fn(w))
            except Exception:
                out[f"{name}_{seg}"] = np.nan
    return out


def _seg_stat(vs, ims, dts, qcs, seg): return _seg_group("stat_", vs, ims, dts, qcs, seg)
def _seg_diff(vs, ims, dts, qcs, seg): return _seg_group("diff_", vs, ims, dts, qcs, seg)
def _seg_lfp(vs, ims, dts, qcs, seg):  return _seg_group("lfp_",  vs, ims, dts, qcs, seg)


# ═══════════════════════════ 엔트리포인트 ═══════════════════════════

def compute_his(v, i, t, bol=None) -> dict:
    """All HIs (stat+diff+lfp+custom via @hi, + morph) from raw window samples.
    ``bol`` = the cell's fresh-state shape curves ({vt,vq,ve}); None → morph = NaN."""
    w = W(v, i, t)
    d = {}
    for name, fn in _HI.items():
        try:
            d[name] = float(fn(w))
        except Exception:
            d[name] = np.nan
    d.update(_morph_dict(w, bol))
    return d


def hi_names(*_) -> list:
    return list(_HI) + _MORPH_NAMES


def benchmark_cost(samples, reps: int = 20) -> tuple:
    """Measured per-HI compute time (μs) = marginal cost given the shared curves exist.
    Curves are pre-warmed on each W OUTSIDE timing (they're shared infra, cost-attributed
    to none); morph timed as a group over its 6."""
    names = hi_names()
    ws = []                                                    # 예열된 W (공유 곡선 캐시) — 비계측
    for v, i, t in samples:
        w = W(v, i, t); _ = (w.vq, w.ica, w.q_rel, w.t_norm); ws.append(w)
    cost = {}
    for name, fn in _HI.items():
        t0 = time.perf_counter()
        for _ in range(reps):
            for w in ws:
                fn(w)
        cost[name] = (time.perf_counter() - t0) / (reps * len(ws)) * 1e6
    ref = morph_reference(samples)
    t0 = time.perf_counter()
    for _ in range(reps):
        for w in ws:
            _morph_dict(w, ref)
    per = (time.perf_counter() - t0) / (reps * len(ws)) / max(1, len(_MORPH_NAMES)) * 1e6
    for n in _MORPH_NAMES:
        cost[n] = per
    return names, np.array([cost[n] for n in names])


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("_4_data_hi/clean/MIT/b1c34.csv")
    c = df[df.cycle == df.cycle.unique()[9]]; ph = c[c.current_A > 0.1].sort_values("time_s")
    samples = [(ph.iloc[a:a + 120].voltage_V.values, ph.iloc[a:a + 120].current_A.values,
                ph.iloc[a:a + 120].time_s.values) for a in range(0, 400, 40)]
    samples = [s for s in samples if len(s[0]) > 15]
    names, cost = benchmark_cost(samples)
    print(f"등록된 HI: {len(names)}개 (@hi {len(_HI)} + morph {len(_MORPH_NAMES)})")
    o = np.argsort(-cost)
    print("가장 비싼 5 / 가장 싼 5 (μs):")
    for j in list(o[:5]) + list(o[-5:]):
        print(f"  {names[j]:22s} {cost[j]:8.2f}")
