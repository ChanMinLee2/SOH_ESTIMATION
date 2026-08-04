"""
hi_correlation.py

MIT + HUST _4_data_hi/clean 에서 HI를 사이클별로 추출하고
방전 용량(capacity_Ah)과의 Spearman 상관계수를 계산·시각화.

입력 : _4_data_hi/clean/MIT/*.pkl, _4_data_hi/clean/HUST/*.pkl
출력 : hi_correlation.png
       _4_data_hi/{axis}/cycle/{DS}/{cell_id}.pkl
       _4_data_hi/{axis}/seg/{DS}/{cell_id}.pkl   (세그먼트 포맷)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 예시 — 세그멘테이션 축(--seg-axis)별
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① qfrac (기본, SOC 구간 균등 분할)
    python 4_hi_analysis/hi_correlation.py
    python 4_hi_analysis/hi_correlation.py --dataset HUST --force

② protocol (CC 단계 전환 경계)
    python 4_hi_analysis/hi_correlation.py --seg-axis protocol
    # 파라미터 조정: max_steps(단계 수), nom_cap(정격 용량 Ah), i_step_thresh_c(C-rate 임계값)
    python 4_hi_analysis/hi_correlation.py --seg-axis protocol \
        --axis-config '{"protocol": {"max_steps": 3, "nom_cap": 1.1, "i_step_thresh_c": 0.5}}'

③ vwindow (전압 구간 균등 분할)
    python 4_hi_analysis/hi_correlation.py --seg-axis vwindow
    # 파라미터 조정: n_windows(분할 수, 기본 3)
    python 4_hi_analysis/hi_correlation.py --seg-axis vwindow \
        --axis-config '{"vwindow": {"n_windows": 4}}'

④ rcs (랜덤 구간 샘플링)
    python 4_hi_analysis/hi_correlation.py --seg-axis rcs
    # 파라미터 조정: n_samples(샘플 수), window(구간 폭 qfrac), seed(재현성)
    python 4_hi_analysis/hi_correlation.py --seg-axis rcs \
        --axis-config '{"rcs": {"n_samples": 6, "window": 0.3, "seed": 42}}'

⑤ cluster (K-means 클러스터)
    python 4_hi_analysis/hi_correlation.py --seg-axis cluster
    # [경고] fit() 없이 실행 시 모든 세그먼트가 cluster 0으로 분류됨
    # 파라미터 조정: n_fine(미세분할 수), split_direction(방향별 분리 여부)
    python 4_hi_analysis/hi_correlation.py --seg-axis cluster \
        --axis-config '{"cluster": {"n_fine": 20, "split_direction": true}}'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
공통 옵션
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --dataset  MIT|HUST         (기본: MIT)
  --workers  N                (병렬 프로세스 수, 기본: min(4, cpu))
  --force                     캐시 무시하고 HI 재추출
  --n-top    N                상관계수 산점도 표시 상위 HI 수 (기본: 4)
  --cell     CELL_ID          curve-debug / plateau-debug 대상 셀
  --cycle    N                시각화 대상 사이클 번호 (0 = 첫 유효 사이클)
  --curve-debug               세그먼트×커브 시각화 (HI 유효성 검증)
  --cycles   1,100,300        curve-debug 대상 사이클 목록 (쉼표 구분)
  --n-cycles N                curve-debug 자동 선택 사이클 수 (기본: 5)
  --plateau-debug             단일 사이클 플래토 판정 디버그 플롯
  --plateau-summary           전체 데이터 plateau_frac 요약 플롯

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HI 구조 (docs/NEW_HIS.md 참조)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Global  (15):  G01–G15
  Segment (n_seg × 66): 통계 S01–S20 / 미분 D01–D20 / LFP L01–L20 / Morph M01–M06
  세그먼트 이름: 축마다 다름 (qfrac: dis_hi/dis_mid/dis_lo/chg_lo/chg_mid/chg_hi)
  키 명명: stat_{k}_{seg} / diff_{k}_{seg} / lfp_{k}_{seg} / morph_{k}_{seg}
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
MIT_DIR      = PROJECT_ROOT / "_4_data_hi" / "clean" / "MIT"
HUST_DIR     = PROJECT_ROOT / "_4_data_hi" / "clean" / "HUST"
CACHE_PATH   = STEP_DIR / "hi_features.pkl"
HI_ROOT      = PROJECT_ROOT / "_4_data_hi"

# common 패키지를 subprocess에서도 import 가능하게
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# dV/dQ(DVA)·dQ/dV(ICA) 곡선 계산 헬퍼 — common/scenario/_curves.py 단일 소스.
# (vqslope 세그멘터와 공유. 과거 이 파일에 로컬 정의돼 있던 것을 이동함)
from common.scenario._curves import (  # noqa: E402
    _build_vq_curve, _build_ica_seg, _peak_fwhm_asym,
)
# CC→CV 전환 검출 (--exclude-cv 옵션에서 재사용; vwindow 모듈의 기존 함수)
from common.scenario.vwindow import _detect_cv_start  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# HI 키 상수 정의
# ─────────────────────────────────────────────────────────────────────────────
THETA_FLAT = 0.25  # V/Ah — LFP 플래토 판별 임계값 (|dV/dQ| < θ_flat) — _curves.THETA_FLAT 과 동일

GLOBAL_HI_KEYS = [
    "q_dis", "energy_dis", "v_mean_cw_dis", "r_trans_est", "q_plateau_frac",
    "ica_peak1_v", "ica_peak1_h", "ica_peak1_area",
    "dva_valley_q", "dva_valley_depth",
    "ce", "cv_q_frac", "cv_time_frac", "chg_ica_peak1_h", "ica_peak1_asym",
]
_GLOBAL_LABELS = {
    "q_dis":           "Q_dis",
    "energy_dis":      "E_dis",
    "v_mean_cw_dis":   "μ_cw(V)_dis",
    "r_trans_est":     "R_trans",
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
    "v_mean_cw", "v_std", "v_skew", "v_kurt", "v_ent",
    "i_mean", "i_std", "v_med", "corr_qi", "corr_vi",
    "q_abs", "energy_seg", "v_iqr", "v_range", "v_p10",
    "v_p90", "v_samp_ent", "corr_vt", "i_q_slope", "v_detrended_std",
]
_STAT_LABELS = {
    "v_mean_cw":        "μ_cw(V)",
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
    "dqdv_peak_h", "dqdv_peak_v", "dqdv_peak_w", "dqdv_area", "v_trend_slope",
    "dqdv_peak_asym", "d2vdq2_rms", "dvdq_skew", "dvdq_ent", "dv_di_seg",
    "dqdv_valley_h", "dqdv_valley_v", "dvdq_peak_q", "dvdq_flat_q", "dqdv_area_asym",
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
    "v_trend_slope":   "ΔV/Δt(seg)",
    "dqdv_peak_asym":  "ICA asym",
    "d2vdq2_rms":      "rms(d²V/dQ²)",
    "dvdq_skew":       "skew(dV/dQ)",
    "dvdq_ent":        "H(dV/dQ)",
    "dv_di_seg":       "|ΔV/ΔI|_seg",
    "dqdv_valley_h":   "min(dQ/dV)",
    "dqdv_valley_v":   "V @ valley(dQ/dV)",
    "dvdq_peak_q":     "Q @ max|dV/dQ|",
    "dvdq_flat_q":     "Q @ min|dV/dQ|",
    "dqdv_area_asym":  "ICA area asym",
}

LFP_KEYS = [
    "plateau_frac", "plateau_v_mean", "plateau_v_std", "plateau_dvdq_std",
    "nonlin_idx", "v_dev_mid", "v_flatness", "delta_v_rms",
    "vq_slope_mid", "inflect_v", "inflect_q_frac", "v_concavity",
    "phase_entry_dvdq", "v_q_pearson", "ica_peak_cnt",
    "plateau_v_slope", "v_gradient_exit", "plateau_q_onset", "dv_dt_plateau", "v_ent_plateau",
]
_LFP_LABELS = {
    "plateau_frac":      "plat. frac.",
    "plateau_v_mean":    "μ(V)|plat",
    "plateau_v_std":     "σ(V)|plat",
    "plateau_dvdq_std":  "σ(dV/dQ)|plat",
    "nonlin_idx":        "NL index",
    "v_dev_mid":         "V dev(mid)",
    "v_flatness":        "V flatness",
    "delta_v_rms":       "rms(ΔV)",
    "vq_slope_mid":      "dV/dQ|mid(meas)",
    "inflect_v":         "inflect V",
    "inflect_q_frac":    "inflect q_frac",
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


def _build_hi_groups(seg_names: list) -> tuple:
    """HI_GROUPS, ALL_HI_KEYS, HI_LABELS을 임의 세그먼트 이름 목록으로 재빌드.

    qfrac 이외 축(protocol, vwindow, rcs, cluster)은 세그먼트 이름이 달라
    모듈 레벨 상수가 맞지 않는다. main()에서 축이 결정된 뒤 이 함수로 교체한다.
    """
    labels: dict = {k: _GLOBAL_LABELS[k] for k in GLOBAL_HI_KEYS}
    groups: "OrderedDict[str, list[str]]" = OrderedDict()
    groups["Global"] = GLOBAL_HI_KEYS[:]
    for seg in seg_names:
        for k in STAT_KEYS:
            labels[f"stat_{k}_{seg}"] = _STAT_LABELS[k]
        for k in DIFF_KEYS:
            labels[f"diff_{k}_{seg}"] = _DIFF_LABELS[k]
        for k in LFP_KEYS:
            labels[f"lfp_{k}_{seg}"] = _LFP_LABELS[k]
        for k in MORPH_KEYS:
            labels[f"morph_{k}_{seg}"] = _MORPH_LABELS[k]
        groups[f"{seg} — Stat"]  = [f"stat_{k}_{seg}"  for k in STAT_KEYS]
        groups[f"{seg} — Diff"]  = [f"diff_{k}_{seg}"  for k in DIFF_KEYS]
        groups[f"{seg} — LFP"]   = [f"lfp_{k}_{seg}"   for k in LFP_KEYS]
        groups[f"{seg} — Morph"] = [f"morph_{k}_{seg}" for k in MORPH_KEYS]
    return groups, list(labels.keys()), labels


# ─────────────────────────────────────────────────────────────────────────────
# 카테고리 D: 형태학적 거리 헬퍼 (top-level — multiprocessing 호환)
# ─────────────────────────────────────────────────────────────────────────────

_MORPH_GRID = 50   # 보간 그리드 해상도 (속도-정밀도 균형)
_DTW_BAND   = 5    # Sakoe-Chiba 밴드 (그리드의 10% = 위상 이동 허용폭)

# ── 원시 세그먼트 곡선 리샘플 (CNN 입력용) ──────────────────────────────────
# 5_model/utils/hi_schema.py 의 RAW_N 과 동일해야 함 (단일 소스: 값 48).
RAW_N = 48


def _resample_segment(vs: np.ndarray, ims: np.ndarray, qcs: np.ndarray, dts: np.ndarray):
    """세그먼트 V / |I| / 상대시간 시계열 → q_frac [0,1] 그리드에 RAW_N 포인트로 보간.

    build_dataset.py 의 _resample 규칙과 동일: 세그먼트 내 상대 누적전하(q_rel)를
    [0,1]로 정규화한 그리드에 V, |I|, t_rel 을 선형보간한다. 세그먼트 길이(포인트 수)가
    셀·사이클·데이터셋마다 달라도 동일 해상도로 CNN에 입력할 수 있게 한다.

    raw_i는 절댓값이 아니라 부호가 남는다(양수=충전/음수=방전, DATASET_HUST_README.md의
    current_A 부호 컨벤션과 동일) — 부호 자체는 호출부에서 direction을 곱해 사후 적용한다
    (여기서는 절댓값 보간까지만 수행; docs/260803_RESULTS.md §10.1 참고).

    raw_t는 (t-t[0])/(t[-1]-t[0]) 로 세그먼트 내부에서 시프팅+스케일링한 상대시간이다
    (docs/REGRESSION_UPGRADE.md §8.1 / docs/260803_RESULTS.md §10.10 의 안전 공식과 동일
    — 절대 지속시간이 아니라 상대값이라 SOH 대리 변수 누출을 피한다). 좌표축은 V/I와
    동일하게 q_rel 기반 x를 그대로 쓴다.

    Returns:
        (raw_v, raw_i, raw_t) — 각 (RAW_N,) float32. 전하량이 0에 가까우면 zero 배열.
    """
    grid = np.linspace(0.0, 1.0, RAW_N)
    if len(vs) < 2:
        z = np.zeros(RAW_N, np.float32)
        return z, z.copy(), z.copy()
    q_rel = np.asarray(qcs, float) - float(qcs[0])
    q_tot = float(q_rel[-1])
    if q_tot < 1e-6:
        # 전하 진행이 없는 세그먼트: 시간 순서(인덱스) 기준 균등 보간으로 대체
        x = np.linspace(0.0, 1.0, len(vs))
    else:
        x = q_rel / q_tot
    rv = np.interp(grid, x, np.asarray(vs, float)).astype(np.float32)
    ri = np.interp(grid, x, np.abs(np.asarray(ims, float))).astype(np.float32)
    t = np.cumsum(np.asarray(dts, float))
    t = t - t[0]
    t_span = float(t[-1])
    t_rel = (t / t_span) if t_span > 1e-9 else np.zeros_like(t)
    rt = np.interp(grid, x, t_rel).astype(np.float32)
    return rv, ri, rt


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


def _dtw_batch(queries: np.ndarray, bol: np.ndarray) -> np.ndarray:
    """N개 쿼리 곡선을 단일 참조 곡선에 대해 배치 DTW 계산.

    queries: (N, n)  bol: (n,)  → (N,) 정규화 DTW 거리
    _dtw_distance와 동일한 banded DP이지만 N 차원을 numpy 배열 연산으로 처리.
    Python 루프는 n(=50) 행에 대해서만 돌므로 호출 오버헤드가 N배 절감됨.
    """
    N, n = queries.shape
    band = _DTW_BAND
    d = np.abs(queries[:, :, None] - bol[None, None, :]).astype(np.float32)  # (N,n,n)
    dtw = np.full((N, n, n), np.inf, dtype=np.float32)
    dtw[:, 0, 0] = d[:, 0, 0]
    for j in range(1, min(band + 1, n)):
        dtw[:, 0, j] = dtw[:, 0, j - 1] + d[:, 0, j]
    for i in range(1, min(band + 1, n)):
        dtw[:, i, 0] = dtw[:, i - 1, 0] + d[:, i, 0]
    for i in range(1, n):
        j_lo = max(1, i - band)
        j_hi = min(n, i + band + 1)
        for j in range(j_lo, j_hi):
            np.minimum(dtw[:, i - 1, j], dtw[:, i, j - 1], out=dtw[:, i, j])
            np.minimum(dtw[:, i, j], dtw[:, i - 1, j - 1], out=dtw[:, i, j])
            dtw[:, i, j] += d[:, i, j]
    return (dtw[:, n - 1, n - 1] / n).astype(np.float64)


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
    """카테고리 A: 통계 기반 20종 (S01–S20)."""
    out = {f"stat_{k}_{seg}": np.nan for k in STAT_KEYS}
    n = len(vs)
    if n < 5:
        return out
    q_rel = qcs - qcs[0]

    # S01 v_mean_cw (전하 가중 평균 전압 = Σ(V·I·dt)/Σ(I·dt))
    denom = float(np.sum(ims * dts))
    out[f"stat_v_mean_cw_{seg}"] = (
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
    else:
        out[f"stat_corr_qi_{seg}"] = 0.0
    # S10 corr_vi
    if np.std(vs) > 1e-6 and np.std(ims) > 1e-9:
        out[f"stat_corr_vi_{seg}"] = float(np.corrcoef(vs, ims)[0, 1])
    else:
        out[f"stat_corr_vi_{seg}"] = 0.0
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
            xs = vs[::max(1, (n + 199) // 200)]  # ceil-div: ns ≤ 200
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
    """카테고리 B: 미분 기반 20종 (D01–D20)."""
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

    # D10 v_trend_slope: 구간 시작→끝 선형 기울기 ΔV/Δt_total
    dt_tot = float(np.sum(dts))
    if dt_tot >= 1.0:
        out[f"diff_v_trend_slope_{seg}"] = float(vs[-1] - vs[0]) / dt_tot

    # D15 dv_di_seg: |ΔV/ΔI| 비율 (연속 샘플, ΔI≠0, Δt<2s); CC 구간(ΔI≈0) → 0.0
    if n > 1:
        dv_a = np.diff(vs); di_a = np.diff(ims); dt_a = dts[1:]
        valid = (np.abs(di_a) > 0.01) & (dt_a < 2.0) & (dt_a > 0)
        if valid.sum() > 0:
            r_dyn = np.abs(dv_a[valid] / di_a[valid])
            r_dyn = r_dyn[r_dyn < 1000.0]
            if len(r_dyn) > 0:
                out[f"diff_dv_di_seg_{seg}"] = float(np.mean(r_dyn))
        else:
            out[f"diff_dv_di_seg_{seg}"] = 0.0

    # D16–D17: IC curve valley (min of dQ/dV, relative to peak — uses ICA vmids/dqdv_sm)
    # 밸리 미발견 시 0.0 폴백 (단일 피크 또는 짧은 세그먼트)
    if len(vmids) >= 6:
        pk16 = int(np.argmax(dqdv_sm))
        pk16_h = float(dqdv_sm[pk16])
        _valley_found = False
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
                _valley_found = True
        if not _valley_found:
            out[f"diff_dqdv_valley_h_{seg}"] = 0.0
            out[f"diff_dqdv_valley_v_{seg}"] = 0.0

    # D18–D19: V-Q curve peak/flat Q positions
    fin18 = np.isfinite(dvdq_sm)
    if q_tot > 0.005 and fin18.sum() >= 3:
        qm_f18 = qm[fin18]
        dv_f18 = dvdq_sm[fin18]
        out[f"diff_dvdq_peak_q_{seg}"] = float(qm_f18[int(np.argmax(np.abs(dv_f18)))]) / q_tot
        out[f"diff_dvdq_flat_q_{seg}"] = float(qm_f18[int(np.argmin(np.abs(dv_f18)))]) / q_tot

    # D20: IC area asymmetry (left / right of peak)
    if len(vmids) >= 4:
        pk20 = int(np.argmax(dqdv_sm))
        if float(dqdv_sm[pk20]) > 0 and pk20 >= 1 and pk20 <= len(dqdv_sm) - 2:
            al = float(np.trapz(np.maximum(dqdv_sm[:pk20 + 1], 0), vmids[:pk20 + 1]))
            ar = float(np.trapz(np.maximum(dqdv_sm[pk20:],     0), vmids[pk20:]))
            if al > 1e-9 and ar > 1e-9:
                out[f"diff_dqdv_area_asym_{seg}"] = float(al / ar)

    return out


def _seg_lfp(vs, ims, dts, qcs, seg):
    """카테고리 C: LFP 특징 기반 20종 (L01–L20)."""
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
        plt_dv = dvdq_sm[plt_mask]
        fin_pd = np.isfinite(plt_dv)
        if fin_pd.sum() >= 3:
            out[f"lfp_plateau_dvdq_std_{seg}"] = float(np.std(plt_dv[fin_pd]))

    # L05 nonlin_idx: RMSE(V, V_linear) / V_range
    if fin_b.sum() >= 4:
        v_lin = np.interp(qm, [qm[0], qm[-1]], [v_sm[0], v_sm[-1]])
        v_rng = float(v_sm[fin_b].max() - v_sm[fin_b].min())
        if v_rng > 1e-4:
            rmse = float(np.sqrt(np.mean((v_sm[fin_b] - v_lin[fin_b]) ** 2)))
            out[f"lfp_nonlin_idx_{seg}"] = rmse / v_rng

    # L06 v_dev_mid: V(q_mid) - V_linear(q_mid), 선형 대비 중간점 편차 (방전: 음, 충전: 양)
    q_mid = q_tot / 2.0
    if fin_b.any():
        v_mid     = float(np.interp(q_mid, qm, v_sm))
        v_lin_mid = float(np.interp(q_mid, [qm[0], qm[-1]], [v_sm[0], v_sm[-1]]))
        out[f"lfp_v_dev_mid_{seg}"] = v_mid - v_lin_mid

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

    # L09 vq_slope_mid: dV/dQ at q_mid (측정 전압 기반, OCV 아님)
    if fin_b.any():
        out[f"lfp_vq_slope_mid_{seg}"] = float(np.interp(q_mid, qm, dvdq_sm))

    # L10–L11 inflect (V-Q 변곡점: d²(dV/dQ)/dQ² 영교차 중 최대 곡률 지점)
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
            out[f"lfp_inflect_v_{seg}"]      = float(v_sm[best])
            out[f"lfp_inflect_q_frac_{seg}"] = float(qm[best]) / q_tot

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
    if plt_mask.sum() >= 3:
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
    if plt_mask.sum() >= 3:
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
    theta_flat: float = THETA_FLAT,
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
        plt_mask = fin_b & (np.abs(dvdq_sm) < theta_flat)
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
        plt_mask = fin_b & (np.abs(dvdq_sm) < theta_flat)
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
            f"({total_plt}/{total_bins} bins, θ={theta_flat} V/Ah)",
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
                f"{seg_lbl}\nplateau_frac = {plt_frac:.1%}  (θ = {theta_flat} V/Ah)",
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
            ax_dv.axhline(theta_flat, color="red", ls="--", lw=1.3, zorder=3,
                          label=f"θ = {theta_flat}")
            ax_dv.fill_between(
                qm_mAh, 0,
                np.where(abs_dvdq < theta_flat, abs_dvdq, np.nan),
                color="limegreen", alpha=0.45, zorder=1, label="plateau zone",
            )
            p90 = float(np.nanpercentile(abs_dvdq[fin_b], 90)) if fin_b.any() else theta_flat
            ax_dv.set_ylim(0, max(3.0 * theta_flat, p90 * 1.2))
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
        f"|  θ_flat = {theta_flat} V/Ah",
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


_DEBUG_THETAS = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25)
_SEGS_ORDER   = ["dis_hi", "dis_mid", "dis_lo", "chg_lo", "chg_mid", "chg_hi"]


def _compute_plateau_fracs(df_cyc: pd.DataFrame, theta_flat: float) -> dict:
    """단일 사이클에서 6개 세그먼트별 플래토 비율 계산.

    Returns
    -------
    dict: {'overall': float, 'dis_hi': float, ..., 'chg_hi': float}
          데이터 부족 시 해당 키 = np.nan
    """
    result = {"overall": np.nan, **{s: np.nan for s in _SEGS_ORDER}}

    df_cyc = df_cyc.copy()
    if "phase" not in df_cyc.columns:
        df_cyc = _add_phase(df_cyc)

    dis = df_cyc[df_cyc["phase"] == "discharge"].sort_values("time_s")
    chg = df_cyc[df_cyc["phase"] == "charge"].sort_values("time_s")
    if len(dis) < 30:
        return result

    def _build(rows):
        v  = rows["voltage_V"].values.astype(float)
        i  = np.abs(rows["current_A"].values.astype(float))
        t  = rows["time_s"].values.astype(float)
        dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
        return v, i, dt, np.cumsum(i * dt) / 3600.0

    v_d, i_d, dt_d, q_d = _build(dis)
    q_tot_d = float(q_d[-1])
    has_chg = len(chg) >= 30
    if has_chg:
        v_c, i_c, dt_c, q_c = _build(chg)
        q_tot_c = float(q_c[-1])

    total_plt = total_bins = 0

    for q_lo_f, q_hi_f, seg_name, _ in DIS_SEGS:
        lo, hi = q_lo_f * q_tot_d, q_hi_f * q_tot_d
        m = (q_d >= lo) & (q_d < hi)
        if m.sum() < 8:
            continue
        n_bins = max(8, min(30, int(m.sum()) // 3))
        qm, v_sm, dvdq_sm, _ = _build_vq_curve(
            v_d[m], i_d[m], dt_d[m], n_bins=n_bins)
        fin_b    = np.isfinite(dvdq_sm) & np.isfinite(v_sm)
        plt_mask = fin_b & (np.abs(dvdq_sm) < theta_flat)
        n_plt = int(plt_mask.sum()); n_tot = len(qm)
        result[seg_name] = n_plt / max(1, n_tot)
        total_plt += n_plt; total_bins += n_tot

    if has_chg:
        for q_lo_f, q_hi_f, seg_name, _ in CHG_SEGS:
            lo, hi = q_lo_f * q_tot_c, q_hi_f * q_tot_c
            m = (q_c >= lo) & (q_c < hi)
            if m.sum() < 8:
                continue
            n_bins = max(8, min(30, int(m.sum()) // 3))
            qm, v_sm, dvdq_sm, _ = _build_vq_curve(
                v_c[m], i_c[m], dt_c[m], n_bins=n_bins)
            fin_b    = np.isfinite(dvdq_sm) & np.isfinite(v_sm)
            plt_mask = fin_b & (np.abs(dvdq_sm) < theta_flat)
            n_plt = int(plt_mask.sum()); n_tot = len(qm)
            result[seg_name] = n_plt / max(1, n_tot)
            total_plt += n_plt; total_bins += n_tot

    result["overall"] = total_plt / max(1, total_bins)
    return result


def _cell_plateau_worker(args):
    """병렬 워커: 셀 1개의 first/last/avg 플래토 비율을 모든 theta에 대해 계산.

    Returns
    -------
    (cell_id, results_by_theta, error_msg)
    results_by_theta: {theta: (cyc_first, fracs_first, cyc_last, fracs_last,
                                n_cycs, fracs_avg)}
    """
    pkl_path, thetas = args
    cid = pkl_path.stem
    try:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        df2 = raw.get("cycles")
        if df2 is None:
            return cid, None, "cycles 키 없음"
        avail = sorted(df2["cycle"].unique())
        if not avail:
            return cid, None, "사이클 없음"

        cyc_first = avail[0]
        cyc_last  = avail[-1]
        df_first  = df2[df2["cycle"] == cyc_first]
        df_last   = df2[df2["cycle"] == cyc_last]
        keys      = ["overall"] + _SEGS_ORDER

        results = {}
        for theta in thetas:
            fracs_first = _compute_plateau_fracs(df_first, theta)
            fracs_last  = _compute_plateau_fracs(df_last,  theta)

            acc = {k: [] for k in keys}
            for cyc in avail:
                fr = _compute_plateau_fracs(df2[df2["cycle"] == cyc], theta)
                for k in keys:
                    v = fr.get(k, np.nan)
                    if np.isfinite(v):
                        acc[k].append(v)
            fracs_avg = {k: (float(np.mean(v)) if v else np.nan)
                         for k, v in acc.items()}

            results[theta] = (cyc_first, fracs_first,
                               cyc_last,  fracs_last,
                               len(avail), fracs_avg)
        return cid, results, None
    except Exception as e:
        return cid, None, str(e)


def _run_plateau_debug(dataset: str, cell_id: str, cycle: int,
                       workers: int = 4) -> None:
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
        nearest = min(cycles_avail, key=lambda c: abs(c - cycle))
        print(f"[plateau_debug] 사이클 {cycle} 없음 → 가장 가까운 사이클 {nearest} 사용")
        cycle = nearest

    df_cyc = df_all[df_all["cycle"] == cycle]

    out_dir = STEP_DIR / "outputs" / "plateau_debug" / f"{cell_id}_cyc{cycle:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. PNG: 지정 셀 × 6 theta ────────────────────────────────────────
    for theta in _DEBUG_THETAS:
        theta_tag = f"{theta:.2f}".replace(".", "p")
        out_path  = out_dir / f"plateau_debug_{cell_id}_cyc{cycle:04d}_th{theta_tag}.png"
        plot_plateau_debug(df_cyc, cycle_id=cycle, cell_id=cell_id,
                           out_path=out_path, theta_flat=theta)



# ─────────────────────────────────────────────────────────────────────────────
# Curve Debug: 6-세그먼트 × 5-커브 시각화 (HI 유효성 검증용)
# ─────────────────────────────────────────────────────────────────────────────

# 시간 순 세그먼트 정의: (q_lo_f, q_hi_f, seg_name, label, phase, bg_color)
_CURVE_SEG_ORDER = [
    (0.0, 0.4, "chg_lo",  "chg_lo\n(SoC 0–40%)",   "charge",    "#fff8f0"),
    (0.4, 0.7, "chg_mid", "chg_mid\n(SoC 40–70%)",  "charge",    "#fff0eb"),
    (0.7, 1.0, "chg_hi",  "chg_hi\n(SoC 70–100%)",  "charge",    "#ffeef4"),
    (0.0, 0.4, "dis_hi",  "dis_hi\n(SoC 60–100%)",  "discharge", "#eef5ff"),
    (0.4, 0.7, "dis_mid", "dis_mid\n(SoC 30–60%)",  "discharge", "#eeeeff"),
    (0.7, 1.0, "dis_lo",  "dis_lo\n(SoC 0–30%)",    "discharge", "#eef7ee"),
]


def plot_curve_debug(
    df_all: pd.DataFrame,
    cycles_to_show: "list[int] | None" = None,
    cell_id: str = "",
    n_cycles: int = 5,
    out_path=None,
) -> None:
    """6-세그먼트 × 5-커브 시각화로 HI 계산 기반 곡선 유효성 검증.

    열 순서 (시간 순): chg_lo / chg_mid / chg_hi / dis_hi / dis_mid / dis_lo
    행 0: V-t      — V vs 세그먼트 상대 시간 [s]
    행 1: V-Q      — V vs 누적 Q [mAh], 스무딩 + 플래토 삼각 마커
    행 2: dV/dQ    — Q 축 [mAh], SG 스무딩, θ_flat 기준선 + 피크 위치
    행 3: dQ/dV    — V 축 [V], SG 스무딩 (ICA), 피크 위치
    행 4: Morph    — [0,1] 정규화 V-t(파랑)/V-Q(빨강)/V-E(초록) 오버레이

    복수 사이클을 동일 서브플롯에 오버레이: 용량 高(BOL)=파랑, 低(EOL)=빨강.
    첫 번째(BOL) 사이클의 Morph 곡선은 회색 점선으로 별도 표시.
    """
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines

    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    # ── 사이클 선택 ───────────────────────────────────────────────────────────
    all_cycs = sorted(c for c in df_all["cycle"].unique() if c > 0)
    if not all_cycs:
        print("[curve_debug] 유효 사이클 없음")
        return

    if cycles_to_show is None:
        idx = np.linspace(0, len(all_cycs) - 1, min(n_cycles, len(all_cycs))).astype(int)
        cycles_to_show = [all_cycs[i] for i in idx]

    cycles_to_show = sorted(c for c in cycles_to_show if c in all_cycs)
    if not cycles_to_show:
        print("[curve_debug] 지정된 사이클이 데이터에 없음")
        return

    # BOL = 선택 사이클 중 가장 이른 사이클
    bol_cyc = cycles_to_show[0]

    # ── 용량 기준 colormap ────────────────────────────────────────────────────
    caps: dict = {}
    for cyc in cycles_to_show:
        grp = df_all[df_all["cycle"] == cyc]
        dis = grp[grp["phase"] == "discharge"]
        if len(dis) > 0 and "capacity_Ah" in dis.columns:
            caps[cyc] = float(dis["capacity_Ah"].iloc[0])
        else:
            caps[cyc] = np.nan

    valid_caps = [v for v in caps.values() if np.isfinite(v)]
    cap_lo = min(valid_caps) if valid_caps else 0.0
    cap_hi = max(valid_caps) if valid_caps else 1.0
    cmap = mcm.get_cmap("RdYlBu")

    def _cyc_color(cyc):
        c = caps.get(cyc, np.nan)
        if not np.isfinite(c) or cap_hi == cap_lo:
            return "gray"
        return cmap((c - cap_lo) / (cap_hi - cap_lo))

    # ── Figure 레이아웃 ───────────────────────────────────────────────────────
    N_SEG  = len(_CURVE_SEG_ORDER)
    N_ROWS = 5
    ROW_YLABELS = [
        "V [V]\n(V-t)",
        "V [V]\n(V-Q smooth)",
        "dV/dQ [V/Ah]\n(DVA)",
        "dQ/dV [Ah/V]\n(ICA)",
        "V [V]\n(Morph norm.)",
    ]

    fig, axes = plt.subplots(
        N_ROWS, N_SEG,
        figsize=(N_SEG * 3.4, N_ROWS * 2.6),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#f5f5f5")

    for col, (_, _, _, lbl, _, bg) in enumerate(_CURVE_SEG_ORDER):
        axes[0, col].set_title(lbl, fontsize=8, fontweight="bold", pad=3)
    for row, yl in enumerate(ROW_YLABELS):
        axes[row, 0].set_ylabel(yl, fontsize=7, labelpad=3)
    for row in range(N_ROWS):
        for col, (*_, bg) in enumerate(_CURVE_SEG_ORDER):
            axes[row, col].set_facecolor(bg)
            axes[row, col].tick_params(labelsize=5.5)

    # BOL morph 참조선 저장: seg → {"vt": arr, "vq": arr, "ve": arr}
    bol_morph: dict = {}

    # ── 사이클별 그리기 ───────────────────────────────────────────────────────
    alpha_base = max(0.30, min(0.90, 2.5 / len(cycles_to_show)))
    lw = 1.1

    for cyc in cycles_to_show:
        grp    = df_all[df_all["cycle"] == cyc]
        clr    = _cyc_color(cyc)
        is_bol = (cyc == bol_cyc)

        for col, (q_lo_f, q_hi_f, seg, _, mode, _) in enumerate(_CURVE_SEG_ORDER):
            phase_grp = grp[grp["phase"] == mode].sort_values("time_s")
            if len(phase_grp) < 15:
                continue

            tv  = phase_grp["time_s"].values.astype(float)
            vv  = phase_grp["voltage_V"].values.astype(float)
            iv  = np.abs(phase_grp["current_A"].values.astype(float))
            dtv = np.clip(np.diff(tv, prepend=tv[0]), 0.0, None)

            q_cum = np.cumsum(iv * dtv) / 3600.0
            q_tot = float(q_cum[-1])
            if q_tot < 0.01:
                continue

            lo = q_lo_f * q_tot
            hi = q_hi_f * q_tot
            m_s = (q_cum >= lo) & (q_cum < hi)
            if m_s.sum() < 8:
                continue

            vs_s  = vv[m_s]
            ims_s = iv[m_s]
            dts_s = dtv[m_s]
            ts_s  = tv[m_s] - tv[m_s][0]

            # ─ Row 0: V-t ──────────────────────────────────────────────────
            ax = axes[0, col]
            ax.plot(ts_s, vs_s, color=clr, lw=lw, alpha=alpha_base)
            ax.set_xlabel("t [s]", fontsize=6)

            # ─ Row 1: V-Q (빈 평균 스무딩 + 플래토 마커) ────────────────────
            ax = axes[1, col]
            qm, v_sm, dvdq_sm, _ = _build_vq_curve(vs_s, ims_s, dts_s)
            qm_mAh = qm * 1000.0
            fin_b  = np.isfinite(v_sm) & np.isfinite(dvdq_sm)
            if fin_b.any():
                ax.plot(qm_mAh[fin_b], v_sm[fin_b], color=clr, lw=lw, alpha=alpha_base)
                plt_mask = fin_b & (np.abs(dvdq_sm) < THETA_FLAT)
                if plt_mask.any():
                    ax.scatter(
                        qm_mAh[plt_mask], v_sm[plt_mask],
                        s=14, color=clr, marker="^",
                        alpha=min(1.0, alpha_base * 1.6),
                        linewidths=0, zorder=4,
                    )
            ax.set_xlabel("Q [mAh]", fontsize=6)

            # ─ Row 2: dV/dQ (DVA) ──────────────────────────────────────────
            ax = axes[2, col]
            if fin_b.any():
                ax.plot(qm_mAh[fin_b], dvdq_sm[fin_b], color=clr, lw=lw, alpha=alpha_base)
                ax.axhline( THETA_FLAT, color="dimgray", ls=":", lw=0.7, alpha=0.5)
                ax.axhline(-THETA_FLAT, color="dimgray", ls=":", lw=0.7, alpha=0.5)
                ax.axhline(0.0,         color="dimgray", ls="-", lw=0.5, alpha=0.25)
                # |dV/dQ| 최대 위치 표시 (D18: dvdq_peak_q)
                abs_dv = np.abs(dvdq_sm[fin_b])
                if abs_dv.size > 0:
                    pk_q = qm_mAh[fin_b][int(np.argmax(abs_dv))]
                    ax.axvline(pk_q, color=clr, ls="--", lw=0.7, alpha=alpha_base * 0.7)
            if col == 0:
                ax.text(0.02, 0.97, f"θ=±{THETA_FLAT}", transform=ax.transAxes,
                        fontsize=5.5, va="top", color="dimgray")
            ax.set_xlabel("Q [mAh]", fontsize=6)

            # ─ Row 3: dQ/dV ICA ────────────────────────────────────────────
            ax = axes[3, col]
            vmids, dqdv_sm = _build_ica_seg(vs_s, ims_s, dts_s)
            if len(vmids) >= 4:
                ax.plot(vmids, dqdv_sm, color=clr, lw=lw, alpha=alpha_base)
                ax.axhline(0.0, color="dimgray", ls="-", lw=0.5, alpha=0.25)
                # ICA 피크 위치 표시 (D06: dqdv_peak_v)
                if dqdv_sm.max() > 0:
                    pk_idx = int(np.argmax(dqdv_sm))
                    ax.axvline(vmids[pk_idx], color=clr, ls="--",
                               lw=0.7, alpha=alpha_base * 0.7)
            ax.set_xlabel("V [V]", fontsize=6)

            # ─ Row 4: Morph 정규화 V-t / V-Q / V-E ────────────────────────
            ax = axes[4, col]
            vt_n, vq_n, ve_n = _seg_morph_curves(vs_s, ims_s, dts_s)
            grid_x = np.linspace(0.0, 1.0, _MORPH_GRID)
            for arr_n, mc in zip([vt_n, vq_n, ve_n],
                                 ["#5c7aff", "#e64040", "#3dad6b"]):
                if arr_n is not None:
                    ax.plot(grid_x, arr_n, color=mc, lw=lw * 0.9, alpha=alpha_base)

            # BOL 참조선 저장
            if is_bol:
                bol_morph.setdefault(seg, {})
                for arr_n, key in zip([vt_n, vq_n, ve_n], ["vt", "vq", "ve"]):
                    if arr_n is not None:
                        bol_morph[seg][key] = arr_n

            ax.set_xlabel("norm. fraction", fontsize=6)

    # ── BOL 참조선 오버레이 (행 4) ────────────────────────────────────────────
    grid_x = np.linspace(0.0, 1.0, _MORPH_GRID)
    for col, (_, _, seg, *_) in enumerate(_CURVE_SEG_ORDER):
        ax = axes[4, col]
        for key, mc in zip(["vt", "vq", "ve"],
                           ["#5c7aff", "#e64040", "#3dad6b"]):
            arr = bol_morph.get(seg, {}).get(key)
            if arr is not None:
                ax.plot(grid_x, arr, color=mc, lw=2.2, alpha=0.35,
                        ls="--", zorder=2)

    # ── 범례 ─────────────────────────────────────────────────────────────────
    # 행 1: 플래토 마커 범례
    axes[1, N_SEG - 1].scatter([], [], s=14, color="gray", marker="^",
                               label="plateau bin")
    axes[1, N_SEG - 1].legend(fontsize=6, loc="best", framealpha=0.7)

    # 행 4: Morph 곡선 범례
    morph_handles = [
        mlines.Line2D([], [], color="#5c7aff", lw=1.5, label="V-t"),
        mlines.Line2D([], [], color="#e64040", lw=1.5, label="V-Q"),
        mlines.Line2D([], [], color="#3dad6b", lw=1.5, label="V-E"),
        mlines.Line2D([], [], color="gray",    lw=1.8, ls="--",
                      alpha=0.6, label=f"BOL (cyc {bol_cyc})"),
    ]
    axes[4, N_SEG - 1].legend(handles=morph_handles, fontsize=6,
                               loc="best", framealpha=0.75)

    # ── Colorbar (용량) ───────────────────────────────────────────────────────
    if valid_caps and cap_hi > cap_lo:
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mcolors.Normalize(vmin=cap_lo, vmax=cap_hi),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:, -1], shrink=0.55, pad=0.02)
        cbar.set_label("Capacity (Ah)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # ── 제목 ─────────────────────────────────────────────────────────────────
    n_cyc_str  = len(cycles_to_show)
    cyc_range  = f"cyc {cycles_to_show[0]}–{cycles_to_show[-1]}"
    cap_range  = (f"{cap_hi:.3f}→{cap_lo:.3f} Ah (BOL→EOL)"
                  if valid_caps else "capacity n/a")
    fig.suptitle(
        f"Curve Debug  |  cell: {cell_id}  |  {n_cyc_str} cycles ({cyc_range})"
        f"  |  {cap_range}  |  High cap=Blue, Low cap=Red",
        fontsize=9.5, fontweight="bold",
    )

    if out_path is None:
        plt.show()
    else:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=140, bbox_inches="tight")
        print(f"  [curve_debug] 저장 완료: {out_p}")
    plt.close(fig)


def _run_curve_debug(
    dataset: str,
    cell_id: str,
    cycles_str: str = "",
    n_cycles: int = 5,
) -> None:
    """CLI --curve-debug 진입점."""
    root = MIT_DIR if dataset.upper().startswith("MIT") else HUST_DIR
    matches = sorted(root.glob(f"{cell_id}*.pkl"))
    if not matches:
        matches = [p for p in sorted(root.glob("*.pkl")) if cell_id in p.stem]
    if not matches:
        print(f"[curve_debug] 파일 없음: {root}/{cell_id}*.pkl")
        return

    pkl_path = matches[0]
    print(f"[curve_debug] 로드: {pkl_path}")
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    df_all = raw if isinstance(raw, pd.DataFrame) else raw.get("cycles")
    if df_all is None:
        print("[curve_debug] DataFrame 로드 실패 (키: 'cycles' 없음)")
        return

    meta    = raw.get("meta", {}) if isinstance(raw, dict) else {}
    cell_lbl = meta.get("cell_id", pkl_path.stem)

    cycles_to_show = None
    if cycles_str:
        try:
            cycles_to_show = [int(c.strip()) for c in cycles_str.split(",") if c.strip()]
        except ValueError:
            print(f"[curve_debug] --cycles 파싱 오류: {cycles_str!r}")
            return

    suffix   = (f"_cyc{cycles_str.replace(',', '_')}" if cycles_str
                else f"_n{n_cycles}")
    out_path = STEP_DIR / "outputs" / "curve_debug" / f"curve_debug_{cell_lbl}{suffix}.png"

    plot_curve_debug(
        df_all,
        cycles_to_show=cycles_to_show,
        cell_id=cell_lbl,
        n_cycles=n_cycles,
        out_path=out_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HI 추출 (top-level — multiprocessing 호환)
# ─────────────────────────────────────────────────────────────────────────────

_PHASE_POS =  0.01   # A 초과 → charge
_PHASE_NEG = -0.01   # A 미만 → discharge


def _add_phase(df: pd.DataFrame) -> pd.DataFrame:
    """_4_data_hi/clean 스키마(phase 컬럼 없음)에 phase 컬럼을 current_A 부호로 재구성."""
    df = df.copy()
    cur = df["current_A"]
    df["phase"] = "rest"
    df.loc[cur > _PHASE_POS, "phase"] = "charge"
    df.loc[cur < _PHASE_NEG, "phase"] = "discharge"
    return df


_PROGRESS_BATCH = 20   # 사이클 N개마다 진행률 큐에 보고 (IPC 오버헤드 절감)


def _extract_one_cell(args) -> tuple:
    """반환: (cycle_rows: list[dict], coverage: dict[scenario, [covered, total]]).

    coverage는 random_segment 세그먼터에서만 채워지고, 그 외에는 빈 dict.
    (기존 list 반환 → 튜플로 확장; 호출부 load_all이 언팩 처리)
    """
    _progress_q = None
    _exclude_cv = False
    if isinstance(args, tuple) and len(args) == 5:
        pkl_path_str, _axis, _axis_cfg_json, _exclude_cv, _progress_q = args
    elif isinstance(args, tuple) and len(args) == 4:
        pkl_path_str, _axis, _axis_cfg_json, _exclude_cv = args
    elif isinstance(args, tuple):
        pkl_path_str, _axis, _axis_cfg_json = args
    else:
        pkl_path_str, _axis, _axis_cfg_json = str(args), "qfrac", "{}"

    from common.scenario import get_segmenter as _get_seg
    _axis_cfg = json.loads(_axis_cfg_json) if isinstance(_axis_cfg_json, str) else _axis_cfg_json
    _segmenter = _get_seg(_axis, {_axis: _axis_cfg})
    _spec_names = _segmenter.get_spec().scenario_names

    # subprocess는 모듈 레벨 ALL_HI_KEYS(qfrac 고정)를 그대로 읽으므로 재빌드
    if _axis != "qfrac":
        _, _cell_hi_keys, _ = _build_hi_groups(_spec_names)
    else:
        _cell_hi_keys = ALL_HI_KEYS

    path = Path(pkl_path_str)
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except Exception:
        return [], {}

    meta   = raw.get("meta", {})
    df_all = raw.get("cycles")
    if df_all is None or not isinstance(df_all, pd.DataFrame):
        return [], {}

    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    dataset = meta.get("dataset", "")
    cell_id = meta.get("cell_id", path.stem)

    # {cycle: row_dict} — 배치 morph 결과를 루프 후에 채워 넣기 위해 dict 사용
    cycle_rows: dict[int, dict] = {}
    # {seg_name: {curve_type: [(cyc, arr), ...]}} — 배치 DTW용 곡선 버퍼
    _curve_buf: dict[str, dict[str, list]] = {}

    _progress_local = 0
    for cyc, grp in df_all.groupby("cycle"):
        if _progress_q is not None:
            _progress_local += 1
            if _progress_local >= _PROGRESS_BATCH:
                _progress_q.put(_progress_local)
                _progress_local = 0

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
        row: dict = {k: np.nan for k in _cell_hi_keys}
        row.update({"dataset": dataset, "cell_id": cell_id,
                    "cycle": int(cyc), "capacity_Ah": cap})

        # ── G01–G03 방전 기본 ─────────────────────────────────────────────
        row["q_dis"]          = q_local
        row["energy_dis"]     = float(np.sum(v * i_mag * dt) / 3600.0)
        denom = float(np.sum(i_mag * dt))
        if denom > 1e-9:
            row["v_mean_cw_dis"] = float(np.sum(v * i_mag * dt)) / denom

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

        # ── 방전 세그먼트 HI (segmenter 기반) ───────────────────────────────
        if q_local >= 0.05:
            for _rec in _segmenter.iter_segments(
                cell_id, int(cyc), v, i_mag, dt, q_cum
            ):
                seg = _rec.meta.get("seg_name") or _spec_names[_rec.scenario_id]
                vs_s = _rec.v; ims_s = _rec.i; dts_s = _rec.dt; qcs_s = _rec.q
                row.update(_seg_stat(vs_s, ims_s, dts_s, qcs_s, seg))
                row.update(_seg_diff(vs_s, ims_s, dts_s, qcs_s, seg))
                row.update(_seg_lfp(vs_s, ims_s, dts_s, qcs_s, seg))
                _rv, _ri, _rt = _resample_segment(vs_s, ims_s, qcs_s, dts_s)   # CNN 원시 곡선
                _sign = 1.0 if _rec.direction > 0 else -1.0
                row[f"raw_v_{seg}"] = _rv.tolist()
                row[f"raw_i_{seg}"] = (_ri * _sign).tolist()   # 부호 있는 전류 (docs/260803_RESULTS.md §10.1)
                row[f"raw_t_{seg}"] = _rt.tolist()
                _mc = _seg_morph_curves(vs_s, ims_s, dts_s)
                for _ct, _arr in zip(("vt", "vq", "ve"), _mc):
                    if _arr is not None:
                        _curve_buf.setdefault(seg, {}).setdefault(_ct, []).append((int(cyc), _arr))

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
                # G04 r_trans_est: CC→CV 전환 시점 ΔV/ΔI [mΩ]
                row["r_trans_est"] = _r_dc_from_chg(vc, ic, dtc)

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

                # 충전 세그먼트 HI (CC 전환 갭 없는 경우만, segmenter 기반)
                if not _chg_gap_seg and q_tc >= 0.05:
                    _empty = np.empty(0, dtype=float)
                    # --exclude-cv: 세그먼터에 넘기는 사본만 CV 시작 지점에서 절단.
                    # ce/cv_q_frac/cv_time_frac 등 위의 전역 충전 HI는 원본(vc/ic/dtc/qcc)을
                    # 그대로 써야 하므로 여기서는 별도 변수(vc_s 등)로만 대체한다.
                    if _exclude_cv:
                        _cv_i = _detect_cv_start(vc, ic)
                        vc_s, ic_s, dtc_s, qcc_s = vc[:_cv_i], ic[:_cv_i], dtc[:_cv_i], qcc[:_cv_i]
                    else:
                        vc_s, ic_s, dtc_s, qcc_s = vc, ic, dtc, qcc
                    for _rec in _segmenter.iter_segments(
                        cell_id, int(cyc), _empty, _empty, _empty, _empty,
                        vc_s, ic_s, dtc_s, qcc_s,
                    ):
                        seg = _rec.meta.get("seg_name") or _spec_names[_rec.scenario_id]
                        vs_c = _rec.v; ims_c = _rec.i; dts_c = _rec.dt; qcs_c = _rec.q
                        row.update(_seg_stat(vs_c, ims_c, dts_c, qcs_c, seg))
                        row.update(_seg_diff(vs_c, ims_c, dts_c, qcs_c, seg))
                        row.update(_seg_lfp(vs_c, ims_c, dts_c, qcs_c, seg))
                        _rv_c, _ri_c, _rt_c = _resample_segment(vs_c, ims_c, qcs_c, dts_c)   # CNN 원시 곡선
                        _sign_c = 1.0 if _rec.direction > 0 else -1.0
                        row[f"raw_v_{seg}"] = _rv_c.tolist()
                        row[f"raw_i_{seg}"] = (_ri_c * _sign_c).tolist()   # 부호 있는 전류
                        row[f"raw_t_{seg}"] = _rt_c.tolist()
                        _mc_c = _seg_morph_curves(vs_c, ims_c, dts_c)
                        for _ct, _arr in zip(("vt", "vq", "ve"), _mc_c):
                            if _arr is not None:
                                _curve_buf.setdefault(seg, {}).setdefault(_ct, []).append((int(cyc), _arr))

        cycle_rows[int(cyc)] = row

    # ── 배치 DTW / Fréchet (곡선 버퍼 → 루프 종료 후 일괄 처리) ──────────────
    for _seg, _ct_dict in _curve_buf.items():
        for _ct, _pairs in _ct_dict.items():
            if not _pairs:
                continue
            _bol_arr = _pairs[0][1]                             # 첫 유효 사이클 = BOL
            _queries  = np.array([p[1] for p in _pairs])       # (N, n)
            _dtw_vals = _dtw_batch(_queries, _bol_arr)          # (N,)
            _frec_vals = np.max(np.abs(_queries - _bol_arr), axis=1)  # (N,)
            for (_c, _), _dv, _fv in zip(_pairs, _dtw_vals, _frec_vals):
                if _c in cycle_rows:
                    cycle_rows[_c][f"morph_{_ct}_dtw_{_seg}"]  = float(_dv)
                    cycle_rows[_c][f"morph_{_ct}_frec_{_seg}"] = float(_fv)

    if _progress_q is not None and _progress_local > 0:
        _progress_q.put(_progress_local)

    return list(cycle_rows.values()), dict(getattr(_segmenter, "coverage", {}) or {})


# ─────────────────────────────────────────────────────────────────────────────

def _merge_coverage(dst: dict, src: dict) -> None:
    """coverage 딕트 병합: scenario -> [covered, total] 합산."""
    for k, v in (src or {}).items():
        c = dst.setdefault(k, [0, 0])
        c[0] += v[0]; c[1] += v[1]


def load_all(
    pkl_dir: Path,
    n_workers: int = 4,
    axis: str = "qfrac",
    axis_cfg: dict | None = None,
    exclude_cv: bool = False,
) -> tuple:
    """반환: (df, coverage). coverage는 random_segment 시에만 채워짐(그 외 빈 dict).

    exclude_cv=True: 충전 세그먼트 HI 추출 시 CC→CV 전환 이후 구간을 제외
    (segmenter 자체는 수정 없음 — _extract_one_cell에서 세그먼터에 넘기는
    배열만 절단, 전역 충전 HI는 영향 없음).
    """
    # 사이클 수가 많은 셀(파일 크기 큰 순) 먼저 배정 → 워커 간 부하 균형 개선
    files = sorted(pkl_dir.glob("*.pkl"), key=lambda f: f.stat().st_size, reverse=True)
    cfg_json = json.dumps(axis_cfg or {})
    all_rec: list = []
    coverage: dict = {}
    if n_workers <= 1:
        for f in tqdm(files, desc=pkl_dir.name):
            rows, cov = _extract_one_cell((str(f), axis, cfg_json, exclude_cv))
            all_rec.extend(rows); _merge_coverage(coverage, cov)
    else:
        # 파일 완료 단위 tqdm(pbar)만으로는 큰 파일이 많이 배정된 초반에 진행률이
        # 한참 안 움직이는 것처럼 보임 → Manager 큐로 워커의 사이클 처리량을
        # 실시간으로 받아 별도 진행률 바(총량 미상, 카운트+속도만 표시)를 갱신.
        import multiprocessing as mp
        import queue as _queue_mod
        import threading

        _mgr = mp.Manager()
        _progress_q = _mgr.Queue()
        _stop_evt = threading.Event()
        _cyc_pbar = tqdm(desc=f"{pkl_dir.name} 사이클 처리량", unit="cyc", position=1, leave=False)

        def _drain_progress():
            while not _stop_evt.is_set():
                try:
                    n = _progress_q.get(timeout=0.2)
                    _cyc_pbar.update(n)
                except _queue_mod.Empty:
                    continue

        _drain_thread = threading.Thread(target=_drain_progress, daemon=True)
        _drain_thread.start()

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_extract_one_cell, (str(f), axis, cfg_json, exclude_cv, _progress_q)): f
                    for f in files}
            with tqdm(total=len(files), desc=pkl_dir.name, position=0) as pbar:
                for fut in as_completed(futs):
                    rows, cov = fut.result()
                    all_rec.extend(rows); _merge_coverage(coverage, cov)
                    pbar.update(1)

        _stop_evt.set()
        _drain_thread.join(timeout=1.0)
        _cyc_pbar.close()
        _mgr.shutdown()
    return (pd.DataFrame(all_rec) if all_rec else pd.DataFrame()), coverage


def _to_cycle_df(df: pd.DataFrame) -> pd.DataFrame:
    """평탄 HI DataFrame → 사이클별 글로벌 HI 테이블.

    출력: [cell_id, cycle, capacity_Ah, <글로벌 HI 15개>]
    """
    cols = ["cell_id", "cycle", "capacity_Ah"] + GLOBAL_HI_KEYS
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


def _to_seg_df(df: pd.DataFrame) -> pd.DataFrame:
    """평탄 HI DataFrame → 세그먼트별 HI 테이블 (long format).

    출력: [cell_id, cycle, segment_id, capacity_Ah, scen, stat_v_mean_cw, ..., morph_ve_frec]
    - segment_id: 세그먼트 인덱스 (ScenarioSpec scenario_id 기준)
    - scen: chg 구간 양수(1-based), dis 구간 음수(1-based); qfrac은 _SEG_SCEN 그대로
    - capacity_Ah: stat_q_abs_{seg} (구간 누적 용량 Ah)
    - HI 컬럼: _{seg} 접미사 제거 (66개/구간, 순서 고정)
    """
    # HI_GROUPS에서 세그먼트 이름 순서 추출 (main()에서 이미 재빌드됨)
    _seg_order = list(dict.fromkeys(
        g.split(" — ")[0] for g in HI_GROUPS if " — " in g
    ))

    # qfrac 이면 _SEG_SCEN 그대로 사용; 아니면 방향+위치로 scen 계산
    if all(s in _SEG_SCEN for s in _seg_order):
        _scen_map = {s: _SEG_SCEN[s] for s in _seg_order}
    else:
        _chg = [s for s in _seg_order if s.startswith("chg")]
        _dis  = [s for s in _seg_order if s not in _chg]
        _scen_map = {}
        for idx, seg in enumerate(_seg_order):
            if seg in _chg:
                sv = _chg.index(seg) + 1
            else:
                sv = -(_dis.index(seg) + 1)
            _scen_map[seg] = (sv, idx)

    # 원시 곡선 컬럼 (CNN 입력): 스칼라 HI 와 동일하게 _{seg} 접미사로 저장됨
    _RAW_BASES = ["raw_v", "raw_i", "raw_t"]

    parts = []
    for seg, (scen_val, seg_id) in _scen_map.items():
        suffix    = f"_{seg}"
        # 현재 df에 존재하는 세그먼트 HI 컬럼 → base 이름 매핑
        col_map   = {f"{b}{suffix}": b for b in _SEG_HI_BASES
                     if f"{b}{suffix}" in df.columns}
        if not col_map:
            continue
        # 원시 곡선 컬럼도 존재하면 함께 매핑 (raw_v_{seg} → raw_v)
        col_map.update({f"{b}{suffix}": b for b in _RAW_BASES
                        if f"{b}{suffix}" in df.columns})

        sub = df[["cell_id", "cycle"] + list(col_map.keys())].copy()
        sub = sub.rename(columns=col_map)

        # capacity_Ah = 구간 누적 용량 (stat_q_abs_{seg})
        q_abs_col = f"stat_q_abs{suffix}"
        sub["capacity_Ah"] = df[q_abs_col].values if q_abs_col in df.columns else np.nan

        sub["segment_id"] = seg_id
        sub["scen"]       = scen_val

        hi_present  = [b for b in _SEG_HI_BASES if b in sub.columns]
        raw_present = [b for b in _RAW_BASES    if b in sub.columns]
        sub = sub[["cell_id", "cycle", "segment_id", "capacity_Ah", "scen"]
                  + hi_present + raw_present]
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


def _save_per_cell_hi(
    df: pd.DataFrame,
    dataset: str,
    axis: str = "qfrac",
) -> tuple:
    """평탄 HI DataFrame → cycle / seg 두 가지 형식으로 셀별 pkl 저장.

    Returns:
        (df_cycle, df_seg)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_cycle = _to_cycle_df(df)
    df_seg   = _to_seg_df(df)

    cycle_dir = HI_ROOT / axis / "cycle" / dataset
    seg_dir   = HI_ROOT / axis / "seg"   / dataset
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


def _rand_suffix(axis_cfg: dict) -> str:
    """random_segment=True 면 경로 태그에 붙일 suffix (_random-L{seg_len_pts}), 아니면 빈 문자열.
    train_scr/train_classifier._axis_dir_from_spec 와 동일 규칙."""
    if not axis_cfg.get("random_segment", False):
        return ""
    return f"_random-L{int(axis_cfg.get('seg_len_pts', 20))}"


def _qfw_tag(axis_cfg: dict) -> str:
    """q_frac_wide 파라미터 → 파일/디렉터리 식별 태그."""
    n1 = int(round(axis_cfg.get("n1", 0.4) * 100))
    n2 = int(round(axis_cfg.get("n2", 0.2) * 100))
    ns = int(axis_cfg.get("n_samples", 4))
    return f"n1-{n1}%_n2-{n2}%_N-{ns}{_rand_suffix(axis_cfg)}"


def _qabs_tag(axis_cfg: dict) -> str:
    """q_abs 파라미터 → 파일/디렉터리 식별 태그.
    (train_scr/train_classifier/visualize_results 와 동일 규칙)"""
    ms = int(round(axis_cfg.get("mid_start", 0.20) * 100))
    me = int(round(axis_cfg.get("mid_end", 0.50) * 100))
    sl = int(round(axis_cfg.get("seg_len", 0.15) * 100))
    ns = int(axis_cfg.get("n_samples", 4))
    return f"ms-{ms}%_me-{me}%_sl-{sl}%_N-{ns}{_rand_suffix(axis_cfg)}"


def _vqslope_tag(axis_cfg: dict) -> str:
    """vqslope 파라미터 → 파일/디렉터리 식별 태그. (train_scr._axis_dir_from_spec 와 동일 규칙)"""
    mode = str(axis_cfg.get("mode", "dva")).lower()
    ns   = int(axis_cfg.get("n_samples", 1))
    return f"{mode}_N-{ns}{_rand_suffix(axis_cfg)}"


def _save_coverage_stats(path: Path, per_ds_cov: dict, axis: str, axis_cfg: dict) -> None:
    """random_segment 누락 비율(샘플링되지 못한 존 포인트 비율)을 텍스트로 저장."""
    order = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
    lines = [
        "=" * 72,
        f"  random_segment 누락 비율 통계   (axis={axis}, cfg={json.dumps(axis_cfg, ensure_ascii=False)})",
        "=" * 72,
        "  누락 비율 = 1 - (어느 창에든 포함된 존 포인트 수 / 존 전체 포인트 수)",
        "  (창 겹침은 covered 1회 집계; 랜덤 추출로 샘플링되지 못한 부분)",
        "",
    ]
    for ds, cov in per_ds_cov.items():
        if not cov:
            continue
        lines.append(f"[{ds}]")
        lines.append(f"  {'시나리오':<10}{'covered':>12}{'total':>12}{'커버율':>9}{'누락율':>9}")
        lines.append("  " + "-" * 52)
        tot_c = tot_t = 0
        for name in order:
            if name not in cov:
                continue
            c, t = cov[name]
            tot_c += c; tot_t += t
            covr = 100 * c / t if t else float("nan")
            lines.append(f"  {name:<10}{c:>12,}{t:>12,}{covr:>8.1f}%{100-covr:>8.1f}%")
        if tot_t:
            allcov = 100 * tot_c / tot_t
            lines.append("  " + "-" * 52)
            lines.append(f"  {'전체':<10}{tot_c:>12,}{tot_t:>12,}{allcov:>8.1f}%{100-allcov:>8.1f}%")
        lines.append("")
    lines.append("=" * 72)
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"  누락 비율 저장: {path}")


def load_or_extract(
    cache_path: Path = CACHE_PATH,
    n_workers: int = 4,
    force: bool = False,
    axis: str = "qfrac",
    axis_cfg: dict | None = None,
    exclude_cv: bool = False,
) -> pd.DataFrame:
    """캐시가 있으면 로드, 없으면 전체 추출 후 저장.

    exclude_cv=True: 결과 캐시/저장 경로에 '_ccOnly' 접미사를 붙여 CV 포함
    버전과 별도로 저장한다 (segmenter/axis_cfg 자체는 변경 없음, load_all 참고).
    """
    axis_cfg = dict(axis_cfg or {})

    # q_frac_wide: 파라미터별 고유 경로 사용
    if axis == "q_frac_wide":
        _tag      = _qfw_tag(axis_cfg)
        _cache    = cache_path.parent / f"hi_features_{_tag}.pkl"
        _axis_dir = f"q_frac_wide/{_tag}"
    elif axis == "q_abs":
        _tag      = _qabs_tag(axis_cfg)
        _cache    = cache_path.parent / f"hi_features_qabs_{_tag}.pkl"
        _axis_dir = f"q_abs/{_tag}"
    elif axis == "vqslope":
        # vqslope: mode(dva/ica)·n_samples 별 고유 경로
        _tag      = _vqslope_tag(axis_cfg)
        _cache    = cache_path.parent / f"hi_features_vqslope_{_tag}.pkl"
        _axis_dir = f"vqslope/{_tag}"
    elif axis == "qfrac":
        _cache    = cache_path
        _axis_dir = axis
    else:
        _cache    = cache_path.parent / f"hi_features_{axis}.pkl"
        _axis_dir = axis

    if exclude_cv:
        _cache    = _cache.with_name(_cache.stem + "_ccOnly.pkl")
        _axis_dir = f"{_axis_dir}_ccOnly"

    if not force and _cache.exists():
        print(f"  캐시 로드: {_cache}")
        return pd.read_pickle(_cache)

    # vwindow: dis_edges/chg_edges 없으면 LFP 물리 기반 고정 경계 사용
    if axis == "vwindow" and "dis_edges" not in axis_cfg:
        from common.scenario.vwindow import VWindowSegmenter as _VW
        _n_win = axis_cfg.get("n_windows", 3)
        _tmp = _VW.from_lfp(n_windows=_n_win)
        axis_cfg.setdefault("dis_edges", _tmp._dis_edges)
        axis_cfg.setdefault("chg_edges", _tmp._chg_edges)
        print(f"[vwindow] LFP 고정 전압 경계  dis={_tmp._dis_edges}  chg={_tmp._chg_edges}")

    from common.scenario import get_segmenter as _get_seg
    _segmenter = _get_seg(axis, {axis: axis_cfg or {}})
    if axis == "cluster":
        print("[경고] cluster 축은 fit() 없이 실행 시 모든 세그먼트가 cluster 0으로 분류됩니다. "
              "HI는 추출되지만 시나리오 라우팅이 무의미합니다.")

    print(f"=== MIT HI 추출 (axis={axis}, exclude_cv={exclude_cv}) ===")
    df_mit,  cov_mit  = load_all(MIT_DIR,  n_workers=n_workers, axis=axis, axis_cfg=axis_cfg,
                                  exclude_cv=exclude_cv)
    dc_mit,  ds_mit  = _save_per_cell_hi(df_mit,  "MIT",  axis=_axis_dir)
    print(f"=== HUST HI 추출 (axis={axis}, exclude_cv={exclude_cv}) ===")
    df_hust, cov_hust = load_all(HUST_DIR, n_workers=n_workers, axis=axis, axis_cfg=axis_cfg,
                                  exclude_cv=exclude_cv)
    dc_hust, ds_hust = _save_per_cell_hi(df_hust, "HUST", axis=_axis_dir)
    _save_sample_csvs(df_mit, df_hust)

    # ScenarioSpec 저장
    _spec_dir = HI_ROOT / _axis_dir
    _spec_dir.mkdir(parents=True, exist_ok=True)
    _segmenter.save_artifacts(_spec_dir)
    print(f"  ScenarioSpec 저장: {_spec_dir / 'scenario_spec.json'}")

    # random_segment 누락 비율 텍스트 저장 (coverage가 있을 때만)
    if cov_mit or cov_hust:
        _save_coverage_stats(_spec_dir / "coverage_stats.txt",
                             {"MIT": cov_mit, "HUST": cov_hust}, axis, axis_cfg)

    df = pd.concat([df_mit, df_hust], ignore_index=True)
    print(f"  총 사이클: MIT {len(df_mit):,}  /  HUST {len(df_hust):,}")
    df.to_pickle(_cache)
    print(f"  캐시 저장: {_cache}")
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

    # ── 레이아웃: Global + N segment rows + scatter ───────────────────────
    # 각 세그먼트 행: [Stat | Diff | LFP] 3 sub-panels
    _segs_for_plot = [k.split(" — ")[0] for k in HI_GROUPS if k.endswith("— Stat")]
    n_segs = len(_segs_for_plot)
    n_seg_hi = n_segs * (len(STAT_KEYS) + len(DIFF_KEYS) + len(LFP_KEYS) + len(MORPH_KEYS))
    fig = plt.figure(figsize=(44, 14 + 7 * n_segs))
    fig.suptitle(
        f"Health Indicator Spearman ρ  ─  {len(GLOBAL_HI_KEYS) + n_seg_hi} HIs"
        f"  (Global {len(GLOBAL_HI_KEYS)} + Segment {n_seg_hi})",
        fontsize=13, fontweight="bold", y=0.999,
    )
    gs_main = gridspec.GridSpec(
        n_segs + 2, 1, figure=fig,
        height_ratios=[1.1] + [1.0] * n_segs + [2.0],
        hspace=0.60,
    )

    # ── 행 0: Global ──────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs_main[0])
    im0, _ = _draw_heatmap(ax0, HI_GROUPS["Global"],
                           "Global  (15 HIs)", corr_df)

    # ── 행 1–N: 세그먼트별 3 sub-panels (HI_GROUPS 기반 동적 생성) ─────────
    seg_rows = [
        (seg, seg, row_idx + 1)
        for row_idx, seg in enumerate(_segs_for_plot)
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

def _print_run_config(axis: str, axis_cfg: dict, args) -> None:
    """HI 추출 실행 조건(축·파라미터·경로·랜덤옵션)을 터미널에 요약 출력."""
    from common.scenario import get_segmenter as _gs
    try:
        _spec = _gs(axis, {axis: axis_cfg}).get_spec()
        _params = _spec.params or {}
        _n_scen = _spec.n_scenarios
    except Exception:
        _params, _n_scen = dict(axis_cfg), "?"

    # 데이터 저장 경로 태그 (load_or_extract 와 동일 규칙)
    if axis == "q_frac_wide":
        _axis_dir = f"q_frac_wide/{_qfw_tag(axis_cfg)}"
    elif axis == "q_abs":
        _axis_dir = f"q_abs/{_qabs_tag(axis_cfg)}"
    elif axis == "vqslope":
        _axis_dir = f"vqslope/{_vqslope_tag(axis_cfg)}"
    elif axis == "qfrac":
        _axis_dir = axis
    else:
        _axis_dir = axis
    _exclude_cv = bool(getattr(args, "exclude_cv", False))
    if _exclude_cv:
        _axis_dir = f"{_axis_dir}_ccOnly"

    _is_rand = bool(axis_cfg.get("random_segment", False))
    w = 64
    print("\n" + "=" * w)
    print("  HI 추출 실행 조건")
    print("=" * w)
    print(f"  세그먼트 축      : {axis}   (시나리오 {_n_scen}개)")
    if _params:
        print(f"  축 파라미터      : {json.dumps(_params, ensure_ascii=False)}")
    print(f"  random_segment  : {_is_rand}"
          + (f"  (seg_len_pts={int(axis_cfg.get('seg_len_pts', 20))}, "
             f"seed={int(axis_cfg.get('random_seed', 42))})" if _is_rand else "  (기존 격자 방식)"))
    print(f"  워커 수          : {getattr(args, 'workers', '?')}")
    print(f"  force 재추출     : {getattr(args, 'force', False)}")
    print(f"  exclude_cv      : {_exclude_cv}"
          + ("  (충전 세그먼트 HI는 CC 구간만 사용)" if _exclude_cv else ""))
    print(f"  데이터 저장 경로 : _4_data_hi/{_axis_dir}/{{seg,cycle}}/")
    if _is_rand:
        print(f"  누락비율 저장    : _4_data_hi/{_axis_dir}/coverage_stats.txt")
    print("=" * w + "\n")


def main():
    cpu = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description="HI 411종 추출 및 Spearman 상관 시각화")
    parser.add_argument("--workers", type=int, default=max(1, cpu - 2),
                        help=f"병렬 프로세스 수 (기본: CPU수-2 = {max(1, cpu - 2)})")
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
    parser.add_argument("--curve-debug", action="store_true",
                        help="6-세그먼트 × 5-커브 시각화 (HI 유효성 검증)")
    parser.add_argument("--cycles", type=str, default="",
                        help="curve-debug 대상 사이클 (쉼표 구분, 예: 1,100,300,500). "
                             "미지정 시 --n-cycles 개수만큼 자동 선택")
    parser.add_argument("--n-cycles", type=int, default=5,
                        help="curve-debug 자동 선택 사이클 수 (기본: 5)")
    # ── 시나리오 축 ──────────────────────────────────────────────────────────
    parser.add_argument("--seg-axis", type=str, default="qfrac",
                        help="세그멘테이션 축: qfrac|protocol|vwindow|rcs|cluster|q_frac_wide|q_abs|vqslope|"
                             "full_cycle(부분 사이클 대비 베이스라인, 방향당 전체 curve 1개) (기본: qfrac)")
    parser.add_argument("--axis-config", type=str, default="{}",
                        help="축 파라미터 JSON 문자열 (예: '{\"n_windows\": 4}'). "
                             "PowerShell에서는 --axis-config=$cfg 형태 또는 --n1/--n2/--n-samples 사용")
    # q_frac_wide / vqslope 전용 단축 인자 — JSON 없이 파라미터 직접 지정 (PowerShell 호환)
    parser.add_argument("--n1",       type=float, default=None,
                        help="q_frac_wide: 구간 크기 (기본 0.4). --axis-config 대체")
    parser.add_argument("--n2",       type=float, default=None,
                        help="q_frac_wide: 세그먼트 길이 (기본 0.2). --axis-config 대체")
    parser.add_argument("--n-samples", type=int, default=None, dest="n_samples",
                        help="q_frac_wide/vqslope: 구간당 세그먼트 수. --axis-config 대체")
    parser.add_argument("--mode",     type=str, default=None,
                        help="vqslope: 플래토 검출 모드 dva|ica (기본 dva). --axis-config 대체")
    parser.add_argument("--random-segment", action="store_true", dest="random_segment",
                        help="q_frac_wide/vqslope/q_abs: 구간 내 고정길이 랜덤 창 추출. --axis-config 대체")
    parser.add_argument("--seg-len-pts", type=int, default=None, dest="seg_len_pts",
                        help="random_segment 시 창의 고정 관측 포인트 수 (기본 20)")
    # q_abs 전용 단축 인자 (정격용량 비율)
    parser.add_argument("--mid-start", type=float, default=None, dest="mid_start",
                        help="q_abs: mid 존 시작 (정격용량 비율, 기본 0.2). --axis-config 대체")
    parser.add_argument("--mid-end",   type=float, default=None, dest="mid_end",
                        help="q_abs: mid 존 끝 (정격용량 비율, 기본 0.5). --axis-config 대체")
    parser.add_argument("--seg-len",   type=float, default=None, dest="seg_len",
                        help="q_abs: 세그먼트 길이 (정격용량 비율, 기본 0.15). --axis-config 대체")
    parser.add_argument("--exclude-cv", action="store_true", dest="exclude_cv",
                        help="충전 세그먼트 HI 추출 시 CC→CV 전환 이후 구간 제외 "
                             "(segmenter는 무수정, 세그먼터에 넘기는 충전 배열만 CV 시작 지점에서 절단; "
                             "결과는 '_ccOnly' 접미사 경로에 별도 저장)")
    args = parser.parse_args()

    # 단축 인자 → axis_config 자동 구성 (PowerShell JSON 우회)
    if (args.n1 is not None or args.n2 is not None or args.n_samples is not None
            or args.mode is not None or args.random_segment or args.seg_len_pts is not None
            or args.mid_start is not None or args.mid_end is not None or args.seg_len is not None):
        _quick: dict = {}
        if args.n1        is not None: _quick["n1"]        = args.n1
        if args.n2        is not None: _quick["n2"]        = args.n2
        if args.n_samples is not None: _quick["n_samples"] = args.n_samples
        if args.mode      is not None: _quick["mode"]      = args.mode
        if args.random_segment:        _quick["random_segment"] = True
        if args.seg_len_pts is not None: _quick["seg_len_pts"] = args.seg_len_pts
        if args.mid_start is not None: _quick["mid_start"] = args.mid_start
        if args.mid_end   is not None: _quick["mid_end"]   = args.mid_end
        if args.seg_len   is not None: _quick["seg_len"]   = args.seg_len
        args.axis_config = json.dumps(_quick)

    _axis = args.seg_axis
    try:
        _axis_cfg: dict = json.loads(args.axis_config)
    except json.JSONDecodeError as e:
        print(f"[ERROR] --axis-config JSON 파싱 실패: {e}"); return

    # qfrac 이외 축은 세그먼트 이름이 달라 모듈 레벨 HI 상수를 재빌드
    if _axis != "qfrac":
        from common.scenario import get_segmenter as _get_seg_hi
        _seg_names_hi = _get_seg_hi(_axis, {_axis: _axis_cfg}).get_spec().scenario_names
        global HI_GROUPS, ALL_HI_KEYS, HI_LABELS, HI_GROUP_TAG
        HI_GROUPS, ALL_HI_KEYS, HI_LABELS = _build_hi_groups(_seg_names_hi)
        HI_GROUP_TAG = {k: g for g, ks in HI_GROUPS.items() for k in ks}
        print(f"[hi] 세그먼트 이름 재빌드: {_seg_names_hi}")

    # ── 실행 조건 요약 출력 (추출 진입 전) ──────────────────────────────────
    _print_run_config(_axis, _axis_cfg, args)

    # ── curve debug 단독 실행 ──────────────────────────────────────────────
    if args.curve_debug:
        if not args.cell:
            root = MIT_DIR if args.dataset.upper().startswith("MIT") else HUST_DIR
            pkls = sorted(root.glob("*.pkl"))
            if not pkls:
                print(f"[curve_debug] {root} 에 pkl 파일 없음"); return
            args.cell = pkls[0].stem
            print(f"[curve_debug] --cell 미지정 → 첫 셀 사용: {args.cell}")
        _run_curve_debug(args.dataset, args.cell,
                         cycles_str=args.cycles, n_cycles=args.n_cycles)
        return

    # ── 플래토 디버그 단독 실행 ─────────────────────────────────────────────
    if args.plateau_debug:
        if not args.cell:
            root = MIT_DIR if args.dataset.upper().startswith("MIT") else HUST_DIR
            pkls = sorted(root.glob("*.pkl"))
            if not pkls:
                print(f"[plateau_debug] {root} 에 pkl 파일 없음"); return
            args.cell = pkls[0].stem
            print(f"[plateau_debug] --cell 미지정 → 첫 셀 사용: {args.cell}")
        _run_plateau_debug(args.dataset, args.cell, args.cycle, args.workers)
        return

    # ── plateau_frac 전체 요약 플롯 ────────────────────────────────────────
    if args.plateau_summary:
        print("\n=== Plateau fraction 전체 요약 플롯 ===")
        df_s = load_or_extract(n_workers=args.workers, force=args.force,
                               axis=_axis, axis_cfg=_axis_cfg, exclude_cv=args.exclude_cv)
        out_sum = STEP_DIR / "outputs" / "plateau_summary.png"
        plot_plateau_fraction_summary(df_s, out_path=out_sum)
        print("완료!")
        return

    df = load_or_extract(n_workers=args.workers, force=args.force,
                         axis=_axis, axis_cfg=_axis_cfg, exclude_cv=args.exclude_cv)
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

    if _axis == "q_frac_wide":
        _dir_suffix = f"_qfw_{_qfw_tag(_axis_cfg)}"          # random suffix(_qfw_tag) 포함
    elif _axis == "q_abs":
        _dir_suffix = f"_qabs_{_qabs_tag(_axis_cfg)}"
    elif _axis == "vqslope":
        _dir_suffix = f"_vqslope_{_vqslope_tag(_axis_cfg)}"  # mode·random suffix 포함
    elif _axis != "qfrac":
        _dir_suffix = f"_{_axis}"
    else:
        _dir_suffix = ""
    if args.exclude_cv:
        _dir_suffix += "_ccOnly"
    hi_plot_dir = STEP_DIR / "hi_plot" / (date.today().strftime("%m%d") + _dir_suffix)
    hi_plot_dir.mkdir(parents=True, exist_ok=True)
    out = hi_plot_dir / "hi_correlation.png"
    print(f"\n=== Plot 저장: {out} ===")
    plot_correlation(corr, df, out, n_top=args.n_top)

    out_dir = STEP_DIR / "outputs" / (date.today().strftime("%m%d") + _dir_suffix)
    print("\n=== 대표 셀 HI 플롯 ===")
    _plot_sample_hi(df, corr, out_dir)

    # ── hi_segment_viz.py 플롯 (trend + overlay) ─────────────────────────────
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("_hi_viz", STEP_DIR / "hi_segment_viz.py")
        _viz = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_viz)
        print(f"\n=== Global HI 열화 추이 ===")
        _viz.plot_hi_trend(df, hi_plot_dir / "hi_trend.png")
        for _cat, _cat_title, _fname in _viz.CATEGORIES:
            print(f"\n=== 세그먼트 HI 추이 ({_cat}) ===")
            _viz.plot_segment_hi_trend(df, hi_plot_dir / _fname, _cat, _cat_title)
        for _cat, _cat_title, _fname in _viz.OVERLAY_CATEGORIES:
            print(f"\n=== 시나리오 오버레이 ({_cat}) ===")
            _viz.plot_segment_hi_overlay(df, hi_plot_dir / _fname, _cat, _cat_title)
    except Exception as _e:
        print(f"[경고] trend/overlay 플롯 생성 실패: {_e}")

    print("완료!")


# ── hi_compute 위임 ──────────────────────────────────────────────────────────
# HI 계산 로직의 단일 소스는 5_model/hi_compute.py.
# 아래 import가 이 파일 내 동명 함수 정의를 덮어써, 새 @hi 함수가
# _seg_stat/_seg_diff/_seg_lfp 를 통해 자동으로 포함된다.
import sys as _hc_sys
from pathlib import Path as _HCPath
_hc_root = str(_HCPath(__file__).resolve().parent.parent / "5_model")
if _hc_root not in _hc_sys.path:
    _hc_sys.path.insert(0, _hc_root)
from hi_compute import (    # noqa: E402
    _seg_stat,
    _seg_diff,
    _seg_lfp,
    _build_vq_curve,
    _build_ica_seg,
    _peak_fwhm_asym,
    _seg_morph_curves,
    _dtw_distance,
    _frechet_distance,
)

if __name__ == "__main__":
    main()
