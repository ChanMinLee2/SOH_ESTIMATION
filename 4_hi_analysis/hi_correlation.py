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
        --axis-config '{"max_steps": 3, "nom_cap": 1.1, "i_step_thresh_c": 0.5}'

③ vwindow (전압 구간 균등 분할)
    python 4_hi_analysis/hi_correlation.py --seg-axis vwindow
    # 파라미터 조정: n_windows(분할 수, 기본 3)
    python 4_hi_analysis/hi_correlation.py --seg-axis vwindow \
        --axis-config '{"n_windows": 4}'

④ rcs (랜덤 구간 샘플링)
    python 4_hi_analysis/hi_correlation.py --seg-axis rcs
    # 파라미터 조정: n_samples(샘플 수), window(구간 폭 qfrac), seed(재현성)
    python 4_hi_analysis/hi_correlation.py --seg-axis rcs \
        --axis-config '{"n_samples": 6, "window": 0.3, "seed": 42}'

⑤ cluster (K-means 클러스터)
    python 4_hi_analysis/hi_correlation.py --seg-axis cluster
    # [경고] fit() 없이 실행 시 모든 세그먼트가 cluster 0으로 분류됨
    # 파라미터 조정: n_fine(미세분할 수), split_direction(방향별 분리 여부)
    python 4_hi_analysis/hi_correlation.py --seg-axis cluster \
        --axis-config '{"n_fine": 20, "split_direction": true}'

  [주의] --axis-config는 축 이름으로 감싸지 않은 "맨" 파라미터 dict를 받는다
  (예: {"n_samples": 6, ...} — {"rcs": {"n_samples": 6, ...}}가 아님). main()이
  내부에서 이미 {axis: axis_cfg}로 한 번 감싸 get_segmenter()에 넘기므로, 여기서
  축 이름으로 한 번 더 감싸면 이중 래핑되어 axis_kwargs에 축 이름 자체가
  키로 들어가 TypeError가 난다(2026-08-11 실제 재현: RCSSegmenter.__init__()
  got an unexpected keyword argument 'random').

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
# 2026-08-08: pkl 데이터(_4_data_hi 전체, 4_hi_analysis의 캐시 pkl)만 D로 이동 — STEP_DIR은
# hi_segment_viz.py 등 실제 코드 파일 위치 조회에도 쓰이므로(아래 spec_from_file_location)
# 그대로 두고, data_directories.py의 공유 상수를 쓴다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_directories import DATA_4_HI_ROOT, PKL_CACHE_ROOT  # noqa: E402
MIT_DIR      = DATA_4_HI_ROOT / "clean" / "MIT"
HUST_DIR     = DATA_4_HI_ROOT / "clean" / "HUST"
CACHE_PATH   = PKL_CACHE_ROOT / "hi_features.pkl"
HI_ROOT      = DATA_4_HI_ROOT

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
# _dtw_batch 청크 크기 — 장수명 셀(사이클 수 많음) × n_samples>1 조합에서 (N,50,50)
# 배열을 한 번에 만들면 N이 수천~수만까지 커져 메모리 부족이 날 수 있다(2026-08-17
# 실측: HUST 1782사이클×4샘플=7128로 136MiB 임시 배열 할당 실패, --workers 다중 프로세스
# 동시 실행 시 압박 가중). N을 이 크기로 잘라 처리해 피크 메모리를 셀 크기와 무관하게
# 상한선 이하로 유지한다.
_DTW_CHUNK  = 2000

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

    입력을 미리 float32로 캐스팅해 뺄셈 단계에서 float64 임시 배열이 안 생기게 하고,
    N을 _DTW_CHUNK 단위로 나눠 처리해 (N,n,n) 배열의 피크 메모리가 N(=그 셀·시나리오의
    곡선 인스턴스 총합, 장수명 셀 × n_samples면 수천 단위까지 커질 수 있음)에 비례해
    무한정 커지지 않게 한다(2026-08-17, 실제 HUST 장수명 셀에서 메모리 부족 실측).
    """
    N, n = queries.shape
    band = _DTW_BAND
    queries = queries.astype(np.float32, copy=False)
    bol = bol.astype(np.float32, copy=False)

    out = np.empty(N, dtype=np.float64)
    chunk = max(1, min(N, _DTW_CHUNK))
    for start in range(0, N, chunk):
        end = start + chunk
        q = queries[start:end]
        m = q.shape[0]
        d = np.abs(q[:, :, None] - bol[None, None, :])  # (m,n,n), 이미 float32
        dtw = np.full((m, n, n), np.inf, dtype=np.float32)
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
        out[start:end] = dtw[:, n - 1, n - 1] / n
    return out


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


def _strip_seg_suffix(d: dict, seg: str) -> dict:
    """{"stat_v_mean_cw_chg_hi": v, ...} -> {"stat_v_mean_cw": v, ...}.

    호출부가 _seg_stat/_seg_diff/_seg_lfp(..., seg)로 직접 만든 접미사만 제거하므로
    (seg 문자열 자체에 언더스코어가 있어도) 항상 정확히 그 seg만 떼어낸다.
    """
    suf = f"_{seg}"
    n = len(suf)
    return {(k[:-n] if k.endswith(suf) else k): v for k, v in d.items()}


def _extract_one_cell(args) -> tuple:
    """반환: (seg_rows, cycle_rows, coverage).

    seg_rows: 세그먼트 인스턴스 1개당 행 1개(native seg 포맷, HI 컬럼 접미사 없음) —
      모델 학습 입력(5_model)이 실제로 읽는 데이터. 한 (사이클,시나리오)에 n_samples개면
      n_samples개 행이 그대로 남는다(2026-08-16 이전엔 row.update() 덮어쓰기로 마지막
      1개만 남았음 — docs/260816_RESULTS.md 참고).
    cycle_rows: 사이클 1개당 행 1개, 글로벌 HI(G01~G15) + capacity_Ah만 포함 — 세그먼트와
      무관하게 사이클당 한 번만 계산되므로 그대로 1행/사이클 유지.
    coverage: random_segment 세그먼터에서만 채워지고, 그 외에는 빈 dict.
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

    # scen 코드 사전 계산 (_SEG_SCEN에 있으면 그대로, 없으면 방향+등장순서 기반 — 기존
    # _to_seg_df와 동일 규칙). segment_id는 이제 SegmentRecord.scenario_id를 그대로 쓴다
    # (spec.routing으로 이미 계산된 값이라 재계산 불필요 — qfrac류에서 _SEG_SCEN 순서와도 일치).
    if all(s in _SEG_SCEN for s in _spec_names):
        _scen_lookup = {s: _SEG_SCEN[s][0] for s in _spec_names}
    else:
        _chg_names = [s for s in _spec_names if s.startswith("chg")]
        _dis_names = [s for s in _spec_names if s not in _chg_names]
        _scen_lookup = {}
        for s in _spec_names:
            _scen_lookup[s] = ((_chg_names.index(s) + 1) if s in _chg_names
                                else -(_dis_names.index(s) + 1))

    path = Path(pkl_path_str)
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except Exception:
        return [], [], {}

    meta   = raw.get("meta", {})
    df_all = raw.get("cycles")
    if df_all is None or not isinstance(df_all, pd.DataFrame):
        return [], [], {}

    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    dataset = meta.get("dataset", "")
    cell_id = meta.get("cell_id", path.stem)

    seg_rows: list[dict] = []      # 세그먼트 인스턴스별 행 (native seg 포맷)
    cycle_rows: list[dict] = []    # 사이클별 글로벌 HI 행 (cycle 포맷)
    # {seg_name: {curve_type: [(그 세그먼트 행 dict, arr), ...]}} — 배치 DTW용 곡선 버퍼.
    # 키를 사이클 번호가 아니라 "그 세그먼트 행 자체"로 잡아, 배치 처리 후 바로 그 행에
    # 대입한다(사이클 단위 딕셔너리를 거치지 않으므로 여러 세그먼트가 같은 사이클번호를
    # 공유해도 서로 안 덮어씀).
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

        # ── 사이클 글로벌 HI (세그먼트와 무관, 사이클당 한 번만 계산) ────────────
        grow: dict = {k: np.nan for k in GLOBAL_HI_KEYS}
        grow.update({"dataset": dataset, "cell_id": cell_id,
                     "cycle": int(cyc), "capacity_Ah": cap})

        # ── G01–G03 방전 기본 ─────────────────────────────────────────────
        grow["q_dis"]          = q_local
        grow["energy_dis"]     = float(np.sum(v * i_mag * dt) / 3600.0)
        denom = float(np.sum(i_mag * dt))
        if denom > 1e-9:
            grow["v_mean_cw_dis"] = float(np.sum(v * i_mag * dt)) / denom

        # ── G05 q_plateau_frac ────────────────────────────────────────────
        mask_plt = (v >= 3.10) & (v <= 3.45)
        if q_local > 0:
            grow["q_plateau_frac"] = (
                float(np.sum(i_mag[mask_plt] * dt[mask_plt]) / 3600.0) / q_local
            )

        # ── G06–G08, G15: ICA ─────────────────────────────────────────────
        p1v, p1h, p1ar, p1asy = _global_ica(v, i_mag, dt)
        grow["ica_peak1_v"]    = p1v
        grow["ica_peak1_h"]    = p1h
        grow["ica_peak1_area"] = p1ar
        grow["ica_peak1_asym"] = p1asy

        # ── G09–G10: DVA ──────────────────────────────────────────────────
        grow["dva_valley_q"], grow["dva_valley_depth"] = _global_dva(
            v, i_mag, dt, q_local
        )

        # ── 방전 세그먼트 HI (segmenter 기반) — 세그먼트 인스턴스마다 독립 행 ──────
        if q_local >= 0.05:
            for _rec in _segmenter.iter_segments(
                cell_id, int(cyc), v, i_mag, dt, q_cum
            ):
                seg = _rec.meta.get("seg_name") or _spec_names[_rec.scenario_id]
                vs_s = _rec.v; ims_s = _rec.i; dts_s = _rec.dt; qcs_s = _rec.q
                _srow: dict = {
                    "dataset": dataset, "cell_id": cell_id, "cycle": int(cyc),
                    "capacity_Ah": cap,
                    "segment_id": int(_rec.scenario_id),
                    "seg_name": seg,
                    "scen": _scen_lookup.get(seg, 0),
                    # assign="none"(no_scen 대조군)이라도 사후 존별 재분리가 가능하도록
                    # 원본 존/위치 정보를 그대로 보존한다(docs/260816_RESULTS.md §5-4).
                    "zone": _rec.meta.get("zone"),
                    "q_frac_lo": _rec.meta.get("q_frac_lo"),
                    "q_frac_hi": _rec.meta.get("q_frac_hi"),
                }
                _srow.update(_strip_seg_suffix(_seg_stat(vs_s, ims_s, dts_s, qcs_s, seg), seg))
                _srow.update(_strip_seg_suffix(_seg_diff(vs_s, ims_s, dts_s, qcs_s, seg), seg))
                _srow.update(_strip_seg_suffix(_seg_lfp(vs_s, ims_s, dts_s, qcs_s, seg), seg))
                _rv, _ri, _rt = _resample_segment(vs_s, ims_s, qcs_s, dts_s)   # CNN 원시 곡선
                _sign = 1.0 if _rec.direction > 0 else -1.0
                _srow["raw_v"] = _rv.tolist()
                _srow["raw_i"] = (_ri * _sign).tolist()   # 부호 있는 전류 (docs/260803_RESULTS.md §10.1)
                _srow["raw_t"] = _rt.tolist()
                _mc = _seg_morph_curves(vs_s, ims_s, dts_s)
                for _ct, _arr in zip(("vt", "vq", "ve"), _mc):
                    if _arr is not None:
                        _curve_buf.setdefault(seg, {}).setdefault(_ct, []).append((_srow, _arr))
                seg_rows.append(_srow)

        # ── 충전 HI ───────────────────────────────────────────────────────
        chg_grp = grp[grp["phase"] == "charge"].sort_values("time_s")
        if len(chg_grp) >= 20:
            tc  = chg_grp["time_s"].values.astype(float)
            vc  = chg_grp["voltage_V"].values.astype(float)
            ic  = np.abs(chg_grp["current_A"].values.astype(float))
            dtc = np.clip(np.diff(tc, prepend=tc[0]), 0, None)

            # chg_gap_seg=True인 행은 preprocess.py 필터4가 그 지점의 dt가
            # 비정상적으로 크다고(CC 전환 갭 등) 판정한 곳이다(2026-08-05부터
            # 행 단위 판정 — 사이클 전체가 아니라 그 행 하나만 플래그된다).
            # 그 큰 dt를 누적적분(qcc)에 그대로 넣으면 그 지점 "이후" 값까지
            # 전부 오염되므로, 이 행의 dt만 정상 구간 중앙값으로 대체한다 —
            # V/I 값 자체는 그대로 쓰므로 정보 손실은 이 한 행의 시간정보로
            # 국한된다(예전엔 chg_gap_seg가 하나라도 있으면 세그먼트 HI 계산
            # 전체를 스킵했다 — MIT batch2 99.94% 사이클이 이렇게 날아갔었다).
            if "chg_gap_seg" in chg_grp.columns:
                _gap_mask = chg_grp["chg_gap_seg"].to_numpy(dtype=bool)
                if _gap_mask.any():
                    _dtc_pos = dtc[dtc > 0]
                    _dtc_med = float(np.median(_dtc_pos)) if len(_dtc_pos) else 0.0
                    dtc = np.where(_gap_mask, _dtc_med, dtc)

            qcc = np.cumsum(ic * dtc) / 3600.0
            q_tc = float(qcc[-1])

            _chg_incomplete = q_tc < cap * 0.60

            if q_tc > 0.05 and not _chg_incomplete:
                # G04 r_trans_est: CC→CV 전환 시점 ΔV/ΔI [mΩ]
                grow["r_trans_est"] = _r_dc_from_chg(vc, ic, dtc)

                # G11 CE
                grow["ce"] = cap / q_tc

                # G12–G13 CV 거동
                i_mx = float(np.max(ic))
                if i_mx > 0:
                    cv_mask = ic < 0.80 * i_mx
                    q_cv  = float(np.sum(ic[cv_mask] * dtc[cv_mask]) / 3600.0)
                    t_cv  = float(np.sum(dtc[cv_mask]))
                    t_tot = float(np.sum(dtc))
                    grow["cv_q_frac"]   = q_cv / q_tc if q_tc > 0 else np.nan
                    grow["cv_time_frac"] = t_cv / t_tot if t_tot > 0 else np.nan

                # G14 chg_ica_peak1_h
                _, c_pk_h, _, _ = _global_ica(vc, ic, dtc)
                grow["chg_ica_peak1_h"] = c_pk_h

                # 충전 세그먼트 HI (segmenter 기반) — CC 전환 갭이 있던 행은 위에서
                # 이미 dtc를 정상값으로 대체해뒀으므로 더 이상 전체 스킵할 필요 없음.
                if q_tc >= 0.05:
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
                        _srow_c: dict = {
                            "dataset": dataset, "cell_id": cell_id, "cycle": int(cyc),
                            "capacity_Ah": cap,
                            "segment_id": int(_rec.scenario_id),
                            "seg_name": seg,
                            "scen": _scen_lookup.get(seg, 0),
                            "zone": _rec.meta.get("zone"),
                            "q_frac_lo": _rec.meta.get("q_frac_lo"),
                            "q_frac_hi": _rec.meta.get("q_frac_hi"),
                        }
                        _srow_c.update(_strip_seg_suffix(_seg_stat(vs_c, ims_c, dts_c, qcs_c, seg), seg))
                        _srow_c.update(_strip_seg_suffix(_seg_diff(vs_c, ims_c, dts_c, qcs_c, seg), seg))
                        _srow_c.update(_strip_seg_suffix(_seg_lfp(vs_c, ims_c, dts_c, qcs_c, seg), seg))
                        _rv_c, _ri_c, _rt_c = _resample_segment(vs_c, ims_c, qcs_c, dts_c)   # CNN 원시 곡선
                        _sign_c = 1.0 if _rec.direction > 0 else -1.0
                        _srow_c["raw_v"] = _rv_c.tolist()
                        _srow_c["raw_i"] = (_ri_c * _sign_c).tolist()   # 부호 있는 전류
                        _srow_c["raw_t"] = _rt_c.tolist()
                        _mc_c = _seg_morph_curves(vs_c, ims_c, dts_c)
                        for _ct, _arr in zip(("vt", "vq", "ve"), _mc_c):
                            if _arr is not None:
                                _curve_buf.setdefault(seg, {}).setdefault(_ct, []).append((_srow_c, _arr))
                        seg_rows.append(_srow_c)

        cycle_rows.append(grow)

    # ── 배치 DTW / Fréchet (곡선 버퍼 → 루프 종료 후 일괄 처리) ──────────────
    # 각 pair가 "그 세그먼트 행" 자체를 들고 있으므로 결과를 바로 그 행에 대입한다 —
    # 예전엔 사이클 번호로 cycle_rows[_c]를 다시 찾아 대입해서 같은 사이클의 여러
    # 세그먼트가 서로 덮어썼다(2026-08-16 이전 버그, docs/260816_RESULTS.md 참고).
    for _seg, _ct_dict in _curve_buf.items():
        for _ct, _pairs in _ct_dict.items():
            if not _pairs:
                continue
            _bol_arr = _pairs[0][1]                             # 그 시나리오의 첫 유효 세그먼트 = BOL
            _queries  = np.array([p[1] for p in _pairs])       # (N, n)
            _dtw_vals = _dtw_batch(_queries, _bol_arr)          # (N,)
            _frec_vals = np.max(np.abs(_queries - _bol_arr), axis=1)  # (N,)
            for (_srow_ref, _), _dv, _fv in zip(_pairs, _dtw_vals, _frec_vals):
                _srow_ref[f"morph_{_ct}_dtw"]  = float(_dv)
                _srow_ref[f"morph_{_ct}_frec"] = float(_fv)

    if _progress_q is not None and _progress_local > 0:
        _progress_q.put(_progress_local)

    return seg_rows, cycle_rows, dict(getattr(_segmenter, "coverage", {}) or {})


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
    """반환: (df_seg, df_cycle, coverage). coverage는 random_segment 시에만 채워짐(그 외 빈 dict).

    df_seg: 세그먼트 인스턴스별 HI(native seg 포맷, 모델 학습 입력). df_cycle: 사이클별
    글로벌 HI(cycle 포맷). 둘 다 이 디렉터리(MIT 또는 HUST)의 전체 셀을 이어붙인 것.

    exclude_cv=True: 충전 세그먼트 HI 추출 시 CC→CV 전환 이후 구간을 제외
    (segmenter 자체는 수정 없음 — _extract_one_cell에서 세그먼터에 넘기는
    배열만 절단, 전역 충전 HI는 영향 없음).
    """
    # 사이클 수가 많은 셀(파일 크기 큰 순) 먼저 배정 → 워커 간 부하 균형 개선
    files = sorted(pkl_dir.glob("*.pkl"), key=lambda f: f.stat().st_size, reverse=True)
    cfg_json = json.dumps(axis_cfg or {})
    all_seg: list = []
    all_cyc: list = []
    coverage: dict = {}
    if n_workers <= 1:
        for f in tqdm(files, desc=pkl_dir.name):
            seg_rows, cyc_rows, cov = _extract_one_cell((str(f), axis, cfg_json, exclude_cv))
            all_seg.extend(seg_rows); all_cyc.extend(cyc_rows); _merge_coverage(coverage, cov)
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
                    seg_rows, cyc_rows, cov = fut.result()
                    all_seg.extend(seg_rows); all_cyc.extend(cyc_rows); _merge_coverage(coverage, cov)
                    pbar.update(1)

        _stop_evt.set()
        _drain_thread.join(timeout=1.0)
        _cyc_pbar.close()
        _mgr.shutdown()
    return (
        pd.DataFrame(all_seg) if all_seg else pd.DataFrame(),
        pd.DataFrame(all_cyc) if all_cyc else pd.DataFrame(),
        coverage,
    )


def _build_flat_correlation_df(df_seg: pd.DataFrame, df_cycle: pd.DataFrame) -> pd.DataFrame:
    """세그먼트 인스턴스 df + 사이클 글로벌 df → Step4 자체 상관분석/플롯 전용 wide df.

    2026-08-16: 모델 학습(5_model)은 이제 df_seg를 그대로 읽으므로(세그먼트당 1행,
    docs/260816_RESULTS.md) 이 함수의 출력은 **학습에 쓰이지 않는다** — compute_correlations/
    plot_correlation/_plot_sample_hi가 기대하는 "사이클당 1행, 시나리오별 _{seg} 접미사
    컬럼" 형태를 맞춰주기 위한 순수 시각화·진단용 재구성이다. 같은 (사이클,시나리오)의
    n_samples개 세그먼트는 여기서 평균만 낸다 — 이 평균이 모델 입력에 영향을 주지 않으므로
    "세그먼트 단위로 독립 학습"이라는 목표와 충돌하지 않는다.
    """
    if df_cycle.empty:
        return pd.DataFrame()

    if df_seg.empty:
        return df_cycle.reset_index(drop=True)

    _hi_cols = [c for c in df_seg.columns if c in _SEG_HI_BASES]
    agg = (df_seg.groupby(["cell_id", "cycle", "seg_name"])[_hi_cols]
                 .mean()
                 .reset_index())

    wide_parts = []
    for seg_name, g in agg.groupby("seg_name"):
        g = g.drop(columns="seg_name").rename(
            columns={c: f"{c}_{seg_name}" for c in _hi_cols})
        wide_parts.append(g.set_index(["cell_id", "cycle"]))

    if not wide_parts:
        return df_cycle.reset_index(drop=True)

    wide = pd.concat(wide_parts, axis=1).reset_index()
    return df_cycle.merge(wide, on=["cell_id", "cycle"], how="left").reset_index(drop=True)


def _save_sample_csvs(
    seg_mit: pd.DataFrame, cyc_mit: pd.DataFrame,
    seg_hust: pd.DataFrame, cyc_hust: pd.DataFrame,
) -> None:
    """데이터셋별 대표 셀 첫 번째 사이클을 cycle/seg 형식으로 CSV 저장.

    seg CSV엔 이제 그 사이클의 세그먼트 인스턴스가 (n_samples개면) 여러 행으로 그대로
    남는다 — 예전엔 시나리오당 1행으로 뭉개진 걸 저장했었다.
    """
    sample_dir = HI_ROOT / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for ds_tag, df_cyc, df_seg in [("mit", cyc_mit, seg_mit), ("hust", cyc_hust, seg_hust)]:
        if df_cyc.empty:
            continue
        first_cell = df_cyc["cell_id"].iloc[0]
        first_cyc  = int(df_cyc[df_cyc["cell_id"] == first_cell]["cycle"].min())

        cyc_row = df_cyc[(df_cyc["cell_id"] == first_cell) & (df_cyc["cycle"] == first_cyc)]
        seg_row = (df_seg[(df_seg["cell_id"] == first_cell) & (df_seg["cycle"] == first_cyc)]
                   if not df_seg.empty else df_seg)

        cyc_row.to_csv(sample_dir / f"{ds_tag}_hi_cycle{first_cyc}.csv", index=False)
        seg_row.to_csv(sample_dir / f"{ds_tag}_hi_seg{first_cyc}.csv",   index=False)

    print(f"  샘플 CSV: {sample_dir}")


def _save_per_cell_hi(
    df_seg: pd.DataFrame,
    df_cycle: pd.DataFrame,
    dataset: str,
    axis: str = "qfrac",
) -> tuple:
    """이미 cycle/seg 형식으로 나뉜 DataFrame을 셀별 pkl로 저장.

    Returns:
        (df_cycle, df_seg)
    """
    cycle_dir = HI_ROOT / axis / "cycle" / dataset
    seg_dir   = HI_ROOT / axis / "seg"   / dataset
    cycle_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    if not df_cycle.empty:
        for cell_id, grp in df_cycle.groupby("cell_id"):
            grp.reset_index(drop=True).to_pickle(cycle_dir / f"{cell_id}.pkl")
    if not df_seg.empty:
        for cell_id, grp in df_seg.groupby("cell_id"):
            grp.reset_index(drop=True).to_pickle(seg_dir / f"{cell_id}.pkl")

    n = df_cycle["cell_id"].nunique() if not df_cycle.empty else 0
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
    # 2026-08-10: min_pts 기본값(10)이 아니면 접미사 — 세그먼트 최소 포인트 임계값이
    # 달라지면 완전히 다른 데이터이므로 반드시 다른 경로에 저장(confound 방지, §4.6).
    min_pts = int(axis_cfg.get("min_pts", 10))
    minpts_sfx = f"_minpts{min_pts}" if min_pts != 10 else ""
    # assign="none"(시나리오-only 대조군, docs/260816_RESULTS.md §5 no_scen)이면
    # 반드시 다른 경로에 저장 — position_bin(6시나리오)과 confound 방지(§4.6과 동일 원칙).
    assign_sfx = "" if axis_cfg.get("assign", "position_bin") == "position_bin" else "_noscen"
    return f"n1-{n1}%_n2-{n2}%_N-{ns}{_rand_suffix(axis_cfg)}{minpts_sfx}{assign_sfx}"


def _qfref_tag(axis_cfg: dict) -> str:
    """q_frac_ref 파라미터 → 파일/디렉터리 식별 태그.

    n1/n2/N은 q_frac_wide와 동일 규칙(부모 클래스 파라미터 상속) + ref_lag/noise_amp/
    noise_period를 덧붙여 lag·노이즈 파라미터가 다르면 반드시 다른 경로에 저장되게 한다
    (docs/SOC.md §6 Phase 3 "데이터 경로 분리 확인" — §4.6 confound 방지)."""
    base = _qfw_tag(axis_cfg)
    lag = int(axis_cfg.get("ref_lag", 0))
    noise_pct = int(round(axis_cfg.get("noise_amp", 0.03) * 100))
    mode = str(axis_cfg.get("noise_mode", "ou"))
    period = int(round(axis_cfg.get("noise_period_cycles", 200.0)))
    return f"{base}_lag-{lag}_noise-{noise_pct}%_{mode}-{period}"


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
    no_shape: bool = False,
) -> pd.DataFrame:
    """캐시가 있으면 로드, 없으면 전체 추출 후 저장.

    exclude_cv=True: 결과 캐시/저장 경로에 '_ccOnly' 접미사를 붙여 CV 포함
    버전과 별도로 저장한다 (segmenter/axis_cfg 자체는 변경 없음, load_all 참고).
    no_shape=True: preprocess.py --skip-shape로 만든 _4_data_hi/clean_noshape/를
    입력으로 쓰고(main()에서 MIT_DIR/HUST_DIR을 그쪽으로 재지정), 결과 캐시/저장
    경로에 '_noshape' 접미사를 붙여 필터7 있는 기본 버전과 절대 안 겹치게 한다.
    """
    axis_cfg = dict(axis_cfg or {})

    # q_frac_wide: 파라미터별 고유 경로 사용
    if axis == "q_frac_wide":
        _tag      = _qfw_tag(axis_cfg)
        _cache    = cache_path.parent / f"hi_features_{_tag}.pkl"
        _axis_dir = f"q_frac_wide/{_tag}"
    elif axis == "q_frac_ref":
        _tag      = _qfref_tag(axis_cfg)
        _cache    = cache_path.parent / f"hi_features_qfref_{_tag}.pkl"
        _axis_dir = f"q_frac_ref/{_tag}"
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

    if no_shape:
        _cache    = _cache.with_name(_cache.stem + "_noshape.pkl")
        _axis_dir = f"{_axis_dir}_noshape"

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
    seg_mit,  cyc_mit,  cov_mit  = load_all(MIT_DIR,  n_workers=n_workers, axis=axis,
                                             axis_cfg=axis_cfg, exclude_cv=exclude_cv)
    _save_per_cell_hi(seg_mit, cyc_mit, "MIT", axis=_axis_dir)
    print(f"=== HUST HI 추출 (axis={axis}, exclude_cv={exclude_cv}) ===")
    seg_hust, cyc_hust, cov_hust = load_all(HUST_DIR, n_workers=n_workers, axis=axis,
                                             axis_cfg=axis_cfg, exclude_cv=exclude_cv)
    _save_per_cell_hi(seg_hust, cyc_hust, "HUST", axis=_axis_dir)
    _save_sample_csvs(seg_mit, cyc_mit, seg_hust, cyc_hust)

    # ScenarioSpec 저장
    _spec_dir = HI_ROOT / _axis_dir
    _spec_dir.mkdir(parents=True, exist_ok=True)
    _segmenter.save_artifacts(_spec_dir)
    print(f"  ScenarioSpec 저장: {_spec_dir / 'scenario_spec.json'}")

    # random_segment 누락 비율 텍스트 저장 (coverage가 있을 때만)
    if cov_mit or cov_hust:
        _save_coverage_stats(_spec_dir / "coverage_stats.txt",
                             {"MIT": cov_mit, "HUST": cov_hust}, axis, axis_cfg)

    seg_all = pd.concat([seg_mit, seg_hust], ignore_index=True)
    cyc_all = pd.concat([cyc_mit, cyc_hust], ignore_index=True)
    print(f"  총 사이클: MIT {len(cyc_mit):,}  /  HUST {len(cyc_hust):,}"
          f"  (세그먼트 인스턴스: MIT {len(seg_mit):,} / HUST {len(seg_hust):,})")

    # Step4 자체 상관분석/플롯 전용 wide df — 모델 학습(5_model)은 seg pkl(native
    # 포맷, 세그먼트당 1행)을 직접 읽으므로 이 df는 학습에 안 쓰인다(_build_flat_correlation_df
    # 참고, docs/260816_RESULTS.md).
    df = _build_flat_correlation_df(seg_all, cyc_all)
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

    # ── 공유 컬러바 ── 이 시점의 fig.get_axes()는 전부 Global+세그먼트 히트맵
    # (산점도 axes는 아직 생성 전) — 예전엔 n_segs=6(qfrac 계열) 기준 [:7]로
    # 하드코딩돼 있었는데, random/random_grid(assign="none")처럼 n_segs=2인
    # 축에서는 무해했지만 우연히 맞았을 뿐이라 명시적으로 전체를 쓰도록 고침.
    if ref_im is not None:
        cbar = plt.colorbar(ref_im, ax=fig.get_axes(), shrink=0.25, pad=0.01)
        cbar.set_label("Spearman ρ", fontsize=10)

    # ── 마지막 행: 상위 HI 산점도 ── gs_main은 n_segs+2행(0=Global, 1..n_segs=세그먼트,
    # 마지막=산점도)이라 마지막 행 인덱스는 n_segs+1. 예전엔 이 값이 항상 7(=n_segs=6인
    # qfrac/q_frac_wide/q_frac_ref 표준 6-시나리오 축 기준)로 하드코딩돼 있어서
    # n_segs가 다른 축(random/random_grid의 assign="none"=2시나리오, protocol/vwindow/
    # cluster/full_cycle 등)에서 GridSpec 범위를 벗어나 IndexError가 났다(2026-08-15).
    abs_mean = corr_df.abs().mean(axis=1).fillna(0).sort_values(ascending=False)
    top_his  = abs_mean.index[:n_top].tolist()

    gs_sc = gridspec.GridSpecFromSubplotSpec(
        2, n_top, subplot_spec=gs_main[n_segs + 1], hspace=0.52, wspace=0.30)
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
    elif axis == "q_frac_ref":
        _axis_dir = f"q_frac_ref/{_qfref_tag(axis_cfg)}"
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
    if bool(getattr(args, "skip_shape", False)):
        _axis_dir = f"{_axis_dir}_noshape"

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
    parser.add_argument("--force",   action="store_true",
                        help="캐시 무시하고 HI 재추출")
    parser.add_argument("--dataset", type=str, default="MIT",
                        help="데이터셋 (MIT 또는 HUST, 기본: MIT)")
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
    # q_frac_ref 전용 단축 인자 (n1/n2/n_samples는 q_frac_wide와 공유해 위 인자 그대로 씀)
    parser.add_argument("--ref-lag",   type=int, default=None, dest="ref_lag",
                        help="q_frac_ref: 레퍼런스 지연 사이클 수 (기본 0=q_frac_wide와 동등). --axis-config 대체")
    parser.add_argument("--noise-amp", type=float, default=None, dest="noise_amp",
                        help="q_frac_ref: 레퍼런스 노이즈 최대 진폭, 분수 (기본 0.03=±3%%). --axis-config 대체")
    parser.add_argument("--noise-mode", type=str, default=None, dest="noise_mode",
                        choices=["ou", "sine"],
                        help="q_frac_ref: 노이즈 드리프트 방식 ou(기본, bounded random walk)|"
                             "sine(구버전, 결정론적). --axis-config 대체")
    parser.add_argument("--noise-period", type=float, default=None, dest="noise_period_cycles",
                        help="q_frac_ref: 노이즈 평균회귀 특성시간/파장(사이클 수, 기본 200). --axis-config 대체")
    parser.add_argument("--min-pts", type=int, default=None, dest="min_pts",
                        help="q_frac_wide/q_frac_ref: 세그먼트 최소 포인트 수(기본 10). "
                             "기본값과 다르면 '_minptsN' 접미사 경로에 별도 저장(§4.6 confound 방지). "
                             "--axis-config 대체")
    parser.add_argument("--exclude-cv", action="store_true", dest="exclude_cv",
                        help="충전 세그먼트 HI 추출 시 CC→CV 전환 이후 구간 제외 "
                             "(segmenter는 무수정, 세그먼터에 넘기는 충전 배열만 CV 시작 지점에서 절단; "
                             "결과는 '_ccOnly' 접미사 경로에 별도 저장)")
    parser.add_argument("--skip-shape", action="store_true", dest="skip_shape",
                        help="preprocess.py --skip-shape로 만든 _4_data_hi/clean_noshape/를 "
                             "입력으로 사용 (MIT_DIR/HUST_DIR을 그쪽으로 재지정). 결과는 "
                             "'_noshape' 접미사 경로에 별도 저장 — 필터7 있는 기본 데이터와 "
                             "절대 안 겹침")
    args = parser.parse_args()

    if args.skip_shape:
        global MIT_DIR, HUST_DIR
        MIT_DIR  = DATA_4_HI_ROOT / "clean_noshape" / "MIT"
        HUST_DIR = DATA_4_HI_ROOT / "clean_noshape" / "HUST"
        print(f"[--skip-shape] 입력 경로 재지정: MIT_DIR={MIT_DIR}  HUST_DIR={HUST_DIR}")

    # 단축 인자 → axis_config 자동 구성 (PowerShell JSON 우회)
    if (args.n1 is not None or args.n2 is not None or args.n_samples is not None
            or args.ref_lag is not None or args.noise_amp is not None
            or args.noise_mode is not None or args.noise_period_cycles is not None
            or args.min_pts is not None):
        _quick: dict = {}
        if args.n1        is not None: _quick["n1"]        = args.n1
        if args.n2        is not None: _quick["n2"]        = args.n2
        if args.n_samples is not None: _quick["n_samples"] = args.n_samples
        if args.ref_lag   is not None: _quick["ref_lag"]   = args.ref_lag
        if args.noise_amp is not None: _quick["noise_amp"] = args.noise_amp
        if args.noise_mode is not None: _quick["noise_mode"] = args.noise_mode
        if args.noise_period_cycles is not None: _quick["noise_period_cycles"] = args.noise_period_cycles
        if args.min_pts is not None: _quick["min_pts"] = args.min_pts
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

    df = load_or_extract(n_workers=args.workers, force=args.force,
                         axis=_axis, axis_cfg=_axis_cfg, exclude_cv=args.exclude_cv,
                         no_shape=args.skip_shape)
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
    elif _axis == "q_frac_ref":
        _dir_suffix = f"_qfref_{_qfref_tag(_axis_cfg)}"
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
    if args.skip_shape:
        _dir_suffix += "_noshape"
    hi_plot_dir = STEP_DIR / "hi_plot" / (date.today().strftime("%m%d") + _dir_suffix)
    hi_plot_dir.mkdir(parents=True, exist_ok=True)
    out = hi_plot_dir / "hi_correlation.png"
    print(f"\n=== Plot 저장: {out} ===")
    # HI 캐시(load_or_extract)는 이 시점에 이미 디스크에 저장 완료된 상태 — 아래는
    # 순수 시각화라 실패해도 캐시/추출 결과에는 영향 없다. 시나리오 개수(n_segs)가
    # 다른 축(random/random_grid 등)에서 레이아웃 가정이 깨지는 경우가 실제로
    # 있었으므로(2026-08-15, gs_main 인덱스 하드코딩 버그), 트렌드/오버레이 플롯과
    # 동일하게 예외를 잡아 경고만 출력하고 계속 진행한다 — --to-step 4처럼 캐시
    # 빌드만 필요한 실행이 순전히 플롯 문제로 실패 처리(exit!=0)되지 않게 하기 위함.
    try:
        plot_correlation(corr, df, out, n_top=4)
    except Exception as _e:
        print(f"[경고] hi_correlation.png 생성 실패(캐시는 정상 저장됨): {_e}")

    out_dir = STEP_DIR / "outputs" / (date.today().strftime("%m%d") + _dir_suffix)
    print("\n=== 대표 셀 HI 플롯 ===")
    try:
        _plot_sample_hi(df, corr, out_dir)
    except Exception as _e:
        print(f"[경고] 대표 셀 HI 플롯 생성 실패(캐시는 정상 저장됨): {_e}")

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
