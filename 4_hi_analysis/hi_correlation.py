"""
hi_correlation.py

_4_data_hi/clean 에서 HI(Health Indicator)를 사이클별로 추출하고 방전 용량(capacity_Ah)과의
Spearman 상관계수를 계산·시각화 — run_pipeline.py Step 4.

입력 : _4_data_hi/clean/{MIT,HUST,TJU,CALCE}/*.pkl
출력 : hi_correlation.png
       _4_data_hi/{axis}/cycle/{DS}/{cell_id}.pkl (dataset/cell_id/cycle/capacity_Ah만 —
         2026-09-28부로 완전 사이클 Global HI는 계산하지 않는다, 아래 "HI 구조" 참고.
         capacity_Ah 컬럼만은 계속 필요 — model_lib/datasets/segment_dataset.py가 세그먼트별
         부분 capacity_Ah를 이 사이클 총량으로 대체하는 데 씀.)
       _4_data_hi/{axis}/seg/{DS}/{cell_id}.pkl   (세그먼트 포맷 — Step 5 이후가 실제로 읽는 것)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 예시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2026-09-24: q_frac_ref가 유일한 세그멘테이션 축이다(과거 실험용 축 protocol/vwindow/
rcs/cluster/q_abs/vqslope/full_cycle/test_rs는 common/scenario/에서 삭제 — git
히스토리에 남아있으니 필요하면 그쪽에서 복원).

2026-09-29: 이 스크립트는 CLI 인자를 전혀 받지 않는다 — 워커 수/캐시 강제 재추출
여부/데이터셋 그룹/세그멘테이션 축/축 파라미터(n1/n2/ref_lag/noise_amp/...)/CC-only
여부/shape 필터 비활성화 여부, 전부 parameters.py(ACTIVE_WORKERS/ACTIVE_FORCE_EXTRACT/
FIXED_DATASET_GROUP/FIXED_SEG_AXIS/ACTIVE_AXIS_CONFIG/FIXED_EXCLUDE_CV/
FIXED_SKIP_SHAPE)에서만 읽는다(run_pipeline.py도 동일한 값을 읽어 쓴다 — 단일 소스).
실행은 그냥:
    python 4_hi_analysis/hi_correlation.py

실험 조건을 바꾸려면 parameters.py를 직접 고친 뒤 실행한다. 여러 조건을 순차로
돌리려면(예: n1을 바꿔가며 5케이스), 매 케이스 전에 parameters.py를 패치하고
실행 후 원래 값으로 복원하는 드라이버 스크립트를 쓴다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HI 구조 (docs/NEW_HIS.md 참조)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Segment (n_seg × 66): 통계 S01–S20 / 미분 D01–D20 / LFP L01–L20 / Morph M01–M06
  (완전 사이클 Global HI, 옛 G01–G15는 2026-09-28부로 계산하지 않음 — 진단/시각화
  전용이었고 모델 학습 입력도 아니었다. git 히스토리에서 복원 가능.)
  세그먼트 이름: 축마다 다름 (q_frac_ref: dis_hi/dis_mid/dis_lo/chg_lo/chg_mid/chg_hi)
  키 명명: stat_{k}_{seg} / diff_{k}_{seg} / lfp_{k}_{seg} / morph_{k}_{seg}
"""

import json
import os
import pickle
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
# 2026-08-08: pkl 데이터(_4_data_hi 전체, 4_hi_analysis의 캐시 pkl)만 D로 이동 — STEP_DIR은
# hi_segment_viz.py 등 실제 코드 파일 위치 조회에도 쓰이므로(아래 spec_from_file_location)
# 그대로 두고, data_directories.py의 공유 상수를 쓴다.
# 2026-09-28: 저장소가 pip install -e .로 editable install돼 있어(pyproject.toml),
# data_directories/parameters/common 전부 PROJECT_ROOT를 sys.path에 넣지 않아도
# 바로 import된다 — ProcessPoolExecutor로 spawn되는 subprocess에서도 동일.
from data_directories import DATA_4_HI_ROOT, PKL_CACHE_ROOT
import parameters as P  # 축 설정 단일 소스(P.ACTIVE_AXIS_CONFIG)
MIT_DIR      = DATA_4_HI_ROOT / "clean" / "MIT"
HUST_DIR     = DATA_4_HI_ROOT / "clean" / "HUST"
TJU_DIR      = DATA_4_HI_ROOT / "clean" / "TJU"
CALCE_DIR    = DATA_4_HI_ROOT / "clean" / "CALCE"
CACHE_PATH   = PKL_CACHE_ROOT / "hi_features.pkl"
HI_ROOT      = DATA_4_HI_ROOT

# 데이터셋 그룹 — "lfp"(MIT+HUST, 기존 기본값·캐시 경로 그대로 유지)와
# "ncm"(TJU+CALCE, NCM/LCO 화학종 신규 통합, 2026-09-05) 두 축을 독립적으로
# 돌린다. 특징 추출 함수(_extract_one_cell 등)는 화학종을 전혀 구분하지 않고
# 그대로 재사용 — LFP 카테고리(L01–L20, plateau 물리 기반)도 포함해 두 그룹 모두
# 동일한 코드로 계산한다(화학종별 특화 로직을 넣지 않는다는 명시적 결정).
DATASET_GROUPS = {
    "lfp": ["MIT", "HUST"],
    "ncm": ["TJU", "CALCE"],
    "all": ["MIT", "HUST", "TJU", "CALCE"],
}
# 대표 셀 ID/데이터셋별 플롯 스타일(cmap/색상)은 plot.py 소유(2026-09-29 분리) —
# 이 파일에서 상관분석 결과를 그릴 때 plot.py의 plot_correlation/_plot_sample_hi로
# 넘기기만 하고, 여기서는 값을 다시 정의하지 않는다.


def _ds_dir(name: str) -> Path:
    """데이터셋 이름 -> 현재 디렉터리 (MIT_DIR/HUST_DIR은 FIXED_SKIP_SHAPE=True 시
    런타임에 재할당되므로 모듈 전역을 매번 새로 조회해야 한다)."""
    return {"MIT": MIT_DIR, "HUST": HUST_DIR, "TJU": TJU_DIR, "CALCE": CALCE_DIR}[name]

# CC→CV 전환 검출 (FIXED_EXCLUDE_CV=True 시 재사용)
from common.scenario._curves import _detect_cv_start  # noqa: E402

# 세그먼트 HI 계산 로직의 단일 소스는 hi_compute.py(같은 디렉터리, 2026-09-23까지는
# 5_model/에 있었다가 Step4 전용이라 여기로 이동) — stat/diff/lfp/morph 66개 HI 전부
# 여기서 가져온다(같은 이름의 로컬 정의를 두지 않는다, 2026-09-22 정리).
from hi_compute import (  # noqa: E402
    _seg_stat,
    _seg_diff,
    _seg_lfp,
    _seg_morph_curves,
    _peak_fwhm_asym,  # 이 파일 자신은 더 이상 안 씀(완전 사이클 Global HI 계산
                      # 제거, 2026-09-29) — tools/profile_hi_timing.py가
                      # `from hi_correlation import _peak_fwhm_asym`으로 재수출
                      # 받아 쓰므로 그대로 유지.
)

# STAT/DIFF/LFP/MORPH 키 목록의 단일 소스는 model_lib/utils/hi_schema.py("66-HI
# schema") — 2026-09-28: 저장소가 editable install(pyproject.toml)돼 있어 이제
# Step4에서도 바로 import된다. 예전엔 이 목록을 여기 그대로 복사해뒀었다(hi_schema.py와
# 66개 키 전부 동일했음 — 진짜 중복, 같은 이름의 로컬 정의를 두지 않는다는 위 hi_compute.py
# 원칙과 동일하게 적용).
from utils.hi_schema import STAT_KEYS, DIFF_KEYS, LFP_KEYS, MORPH_KEYS  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# HI 키 상수 정의
# ─────────────────────────────────────────────────────────────────────────────

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
for _, _, _seg, _ in ALL_SEGS:
    for _k in STAT_KEYS:
        _HI_META.append((f"stat_{_k}_{_seg}", _STAT_LABELS[_k]))
    for _k in DIFF_KEYS:
        _HI_META.append((f"diff_{_k}_{_seg}", _DIFF_LABELS[_k]))
    for _k in LFP_KEYS:
        _HI_META.append((f"lfp_{_k}_{_seg}", _LFP_LABELS[_k]))
    for _k in MORPH_KEYS:
        _HI_META.append((f"morph_{_k}_{_seg}", _MORPH_LABELS[_k]))

ALL_HI_KEYS = [k for k, _ in _HI_META]   # 6×66 = 396 (완전 사이클 Global HI 15개는 더 이상 계산 안 함)
HI_LABELS   = {k: lbl for k, lbl in _HI_META}

HI_GROUPS: "OrderedDict[str, list[str]]" = OrderedDict()
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
    labels: dict = {}
    groups: "OrderedDict[str, list[str]]" = OrderedDict()
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


def _build_scen_lookup(spec_names: list) -> dict:
    """세그먼트 이름 → scen 코드(방향 부호 있는 정수). _SEG_SCEN(qfrac류 표준
    6-시나리오)에 전부 있으면 그대로, 아니면 이름 접두사(chg/dis)+등장순서로 계산."""
    if all(s in _SEG_SCEN for s in spec_names):
        return {s: _SEG_SCEN[s][0] for s in spec_names}
    _chg_names = [s for s in spec_names if s.startswith("chg")]
    _dis_names = [s for s in spec_names if s not in _chg_names]
    return {
        s: ((_chg_names.index(s) + 1) if s in _chg_names else -(_dis_names.index(s) + 1))
        for s in spec_names
    }


def _load_cell_pkl(path: Path):
    """셀 pkl 로드 + 최소 검증. 반환: (df_all, dataset, cell_id) 또는 실패 시 None."""
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except Exception:
        return None

    meta   = raw.get("meta", {})
    df_all = raw.get("cycles")
    if df_all is None or not isinstance(df_all, pd.DataFrame):
        return None

    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    dataset = meta.get("dataset", "")
    cell_id = meta.get("cell_id", path.stem)
    return df_all, dataset, cell_id


def _extract_segment_rows(rec_iter, spec_names: list, dataset: str, cell_id: str,
                           cyc, cap: float, scen_lookup: dict,
                           curve_buf: dict) -> list:
    """세그먼터가 내놓는 SegmentRecord들을 HI 행(dict) 리스트로 변환.

    방전/충전 양쪽에서 동일하게 쓴다(예전엔 두 곳에 거의 같은 코드가 복사돼 있었다 —
    2026-09-29 통합, docs/REFACTORING.md 참고). Morph HI는 여기서 값을 확정하지
    않고 곡선만 curve_buf에 버퍼링한다 — 실제 거리 계산은 그 사이클 루프가 전부 끝난
    뒤 _apply_batched_morph_distances()가 세그먼트/곡선타입별로 배치 처리한다
    (BOL 참조 곡선을 알려면 같은 시나리오의 첫 세그먼트가 먼저 나와야 하므로).
    """
    rows = []
    for _rec in rec_iter:
        seg = _rec.meta.get("seg_name") or spec_names[_rec.scenario_id]
        vs = _rec.v; ims = _rec.i; dts = _rec.dt; qcs = _rec.q
        _srow: dict = {
            "dataset": dataset, "cell_id": cell_id, "cycle": int(cyc),
            "capacity_Ah": cap,
            "segment_id": int(_rec.scenario_id),
            "seg_name": seg,
            "scen": scen_lookup.get(seg, 0),
            # assign="none"(no_scen 대조군)이라도 사후 존별 재분리가 가능하도록
            # 원본 존/위치 정보를 그대로 보존한다(docs/260816_RESULTS.md §5-4).
            "zone": _rec.meta.get("zone"),
            "q_frac_lo": _rec.meta.get("q_frac_lo"),
            "q_frac_hi": _rec.meta.get("q_frac_hi"),
        }
        _srow.update(_strip_seg_suffix(_seg_stat(vs, ims, dts, qcs, seg), seg))
        _srow.update(_strip_seg_suffix(_seg_diff(vs, ims, dts, qcs, seg), seg))
        _srow.update(_strip_seg_suffix(_seg_lfp(vs, ims, dts, qcs, seg), seg))
        _mc = _seg_morph_curves(vs, ims, dts)
        for _ct, _arr in zip(("vt", "vq", "ve"), _mc):
            if _arr is not None:
                curve_buf.setdefault(seg, {}).setdefault(_ct, []).append((_srow, _arr))
        rows.append(_srow)
    return rows


def _prepare_charge_arrays(chg_grp: pd.DataFrame, cap: float, exclude_cv: bool):
    """충전 phase 원시 배열 준비 + gap 보정 + 완전성 게이트.

    len(chg_grp)<20, 또는 완전 충전 전하량(q_tc)이 0.05Ah 이하거나 등록 용량의
    60% 미만(불완전 충전)이면 None — 이 사이클은 충전 세그먼트 HI를 추출하지
    않는다. 반환값은 (vc, ic, dtc, qcc) — exclude_cv=True면 CV 시작 지점에서
    절단된 사본(세그먼터 입력 전용)이고, 완전성 게이트 자체는 항상 절단 전
    원본 q_tc로 판정한다(CV 구간을 잘라내면 q_tc가 줄어 게이트 판정이 달라지면
    안 되므로).
    """
    if len(chg_grp) < 20:
        return None
    tc  = chg_grp["time_s"].values.astype(float)
    vc  = chg_grp["voltage_V"].values.astype(float)
    ic  = np.abs(chg_grp["current_A"].values.astype(float))
    dtc = np.clip(np.diff(tc, prepend=tc[0]), 0, None)

    # chg_gap_seg=True인 행은 preprocess.py 필터4가 그 지점의 dt가 비정상적으로
    # 크다고(CC 전환 갭 등) 판정한 곳이다(2026-08-05부터 행 단위 판정 — 사이클
    # 전체가 아니라 그 행 하나만 플래그된다). 그 큰 dt를 누적적분(qcc)에 그대로
    # 넣으면 그 지점 "이후" 값까지 전부 오염되므로, 이 행의 dt만 정상 구간
    # 중앙값으로 대체한다 — V/I 값 자체는 그대로 쓰므로 정보 손실은 이 한 행의
    # 시간정보로 국한된다(예전엔 chg_gap_seg가 하나라도 있으면 세그먼트 HI 계산
    # 전체를 스킵했다 — MIT batch2 99.94% 사이클이 이렇게 날아갔었다).
    if "chg_gap_seg" in chg_grp.columns:
        _gap_mask = chg_grp["chg_gap_seg"].to_numpy(dtype=bool)
        if _gap_mask.any():
            _dtc_pos = dtc[dtc > 0]
            _dtc_med = float(np.median(_dtc_pos)) if len(_dtc_pos) else 0.0
            dtc = np.where(_gap_mask, _dtc_med, dtc)

    qcc = np.cumsum(ic * dtc) / 3600.0
    q_tc = float(qcc[-1])
    if q_tc <= 0.05 or q_tc < cap * 0.60:
        return None

    # exclude_cv: 세그먼터에 넘기는 사본만 CV 시작 지점에서 절단.
    if exclude_cv:
        _cv_i = _detect_cv_start(vc, ic)
        return vc[:_cv_i], ic[:_cv_i], dtc[:_cv_i], qcc[:_cv_i]
    return vc, ic, dtc, qcc


def _apply_batched_morph_distances(curve_buf: dict) -> None:
    """곡선 버퍼(세그먼트/곡선타입별) → 배치 DTW/Fréchet 거리를 계산해 각 세그먼트
    행(dict)에 직접 써넣는다(반환값 없음, in-place). 그 시나리오의 첫 유효
    세그먼트를 BOL(fresh-state) 참조로 쓴다.

    각 pair가 "그 세그먼트 행" 자체를 들고 있으므로 결과를 바로 그 행에 대입한다 —
    예전엔 사이클 번호로 cycle_rows[_c]를 다시 찾아 대입해서 같은 사이클의 여러
    세그먼트가 서로 덮어썼다(2026-08-16 이전 버그, docs/260816_RESULTS.md 참고).
    """
    for _seg, _ct_dict in curve_buf.items():
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


def _extract_one_cell(args) -> tuple:
    """반환: (seg_rows, cycle_rows, coverage).

    seg_rows: 세그먼트 인스턴스 1개당 행 1개(native seg 포맷, HI 컬럼 접미사 없음) —
      모델 학습 입력(8_train)이 실제로 읽는 데이터. 한 (사이클,시나리오)에 n_samples개면
      n_samples개 행이 그대로 남는다(2026-08-16 이전엔 row.update() 덮어쓰기로 마지막
      1개만 남았음 — docs/260816_RESULTS.md 참고).
    cycle_rows: 사이클 1개당 행 1개, dataset/cell_id/cycle/capacity_Ah만 포함(2026-09-28
      이전엔 완전 사이클 Global HI(G01~G15)도 여기 같이 있었으나 진단/시각화 전용이라
      계산 자체를 없앴다 — capacity_Ah 컬럼만 여전히 실사용, 아래 주석 참고).
    coverage: random_segment 세그먼터에서만 채워지고, 그 외에는 빈 dict.

    단계별로 헬퍼에 위임하는 얇은 오케스트레이터다(2026-09-29 분리 —
    예전엔 5가지 관심사(인자 해석/pkl 로드/방전 추출/충전 추출/배치 거리계산)가
    231줄짜리 함수 하나에 다 섞여 있었고, 방전·충전 추출 블록도 거의 같은 코드가
    두 번 복사돼 있었다. docs/REFACTORING.md 참고):
      _build_scen_lookup      — 세그먼트 이름 → scen 코드
      _load_cell_pkl          — pkl 로드 + 검증
      _extract_segment_rows   — SegmentRecord → HI 행(방전/충전 공용)
      _prepare_charge_arrays  — 충전 배열 준비 + gap 보정 + 완전성 게이트
      _apply_batched_morph_distances — 루프 후 배치 DTW/Fréchet
    """
    # load_all()의 두 호출부(직렬/ProcessPoolExecutor)만이 실제 호출자다 — 둘 다 항상
    # (path, axis, axis_cfg, exclude_cv) 4-tuple이고, 병렬 실행 시에만 진행률 큐가 5번째로
    # 붙는다. axis_cfg는 dict를 그대로 넘긴다(ProcessPoolExecutor도 pickle로 dict를 그대로
    # 옮기므로 JSON 문자열로 왕복 인코딩할 이유가 없다 — 2026-09-29, 예전엔 3-tuple/
    # 비-tuple("qfrac" 기본값 포함) 분기까지 있었으나 둘 다 실제 호출자가 전혀 없었고,
    # "qfrac"은 이미 REGISTRY에서 삭제된 축이라 그 분기가 실행됐다면 바로 에러였다 —
    # 죽은 방어 코드였음을 grep으로 확인 후 정리).
    if len(args) == 5:
        pkl_path_str, _axis, _axis_cfg, _exclude_cv, _progress_q = args
    else:
        pkl_path_str, _axis, _axis_cfg, _exclude_cv = args
        _progress_q = None

    from common.scenario import get_segmenter as _get_seg
    _segmenter = _get_seg(_axis, {_axis: _axis_cfg})
    _spec_names = _segmenter.get_spec().scenario_names
    _scen_lookup = _build_scen_lookup(_spec_names)

    _loaded = _load_cell_pkl(Path(pkl_path_str))
    if _loaded is None:
        return [], [], {}
    df_all, dataset, cell_id = _loaded

    seg_rows: list[dict] = []      # 세그먼트 인스턴스별 행 (native seg 포맷)
    cycle_rows: list[dict] = []    # 사이클별 행 (cycle 포맷, dataset/cell_id/cycle/capacity_Ah)
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

        # ── 사이클 행(cycle 포맷) — 완전 사이클 Global HI(G01~G15)는 더 이상 계산하지
        # 않는다(2026-09-28, 진단/시각화 전용이었고 학습 입력도 아니었음 —
        # _build_flat_correlation_df 참고). 다만 이 행 자체(dataset/cell_id/cycle/
        # capacity_Ah)는 계속 필요하다 — model_lib/datasets/segment_dataset.py의
        # load_dataset_native_seg()가 이 "cycle" pkl에서 사이클 진짜 총 용량
        # (capacity_Ah)을 읽어 세그먼트별 capacity_Ah(부분 충방전량이라 그 자체로는
        # SOH 타깃이 될 수 없음)를 대체하는 데 실제로 쓰기 때문이다.
        grow: dict = {"dataset": dataset, "cell_id": cell_id,
                      "cycle": int(cyc), "capacity_Ah": cap}

        # ── 방전 세그먼트 HI (segmenter 기반) ──────────────────────────────
        if q_local >= 0.05:
            seg_rows.extend(_extract_segment_rows(
                _segmenter.iter_segments(cell_id, int(cyc), v, i_mag, dt, q_cum),
                _spec_names, dataset, cell_id, cyc, cap, _scen_lookup, _curve_buf,
            ))

        # ── 충전 세그먼트 HI (segmenter 기반) — gap 보정된 배열 + 완전성 게이트는
        # _prepare_charge_arrays가 전담(None이면 이 사이클은 충전 HI 스킵) ──────
        chg_grp = grp[grp["phase"] == "charge"].sort_values("time_s")
        _chg_arrays = _prepare_charge_arrays(chg_grp, cap, exclude_cv=_exclude_cv)
        if _chg_arrays is not None:
            vc_s, ic_s, dtc_s, qcc_s = _chg_arrays
            _empty = np.empty(0, dtype=float)
            seg_rows.extend(_extract_segment_rows(
                _segmenter.iter_segments(
                    cell_id, int(cyc), _empty, _empty, _empty, _empty,
                    vc_s, ic_s, dtc_s, qcc_s,
                ),
                _spec_names, dataset, cell_id, cyc, cap, _scen_lookup, _curve_buf,
            ))

        cycle_rows.append(grow)

    _apply_batched_morph_distances(_curve_buf)

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
    dataset/cell_id/cycle/capacity_Ah 행(cycle 포맷, 완전 사이클 HI 없음). 둘 다 이
    디렉터리(MIT 또는 HUST)의 전체 셀을 이어붙인 것.

    exclude_cv=True: 충전 세그먼트 HI 추출 시 CC→CV 전환 이후 구간을 제외
    (segmenter 자체는 수정 없음 — _extract_one_cell에서 세그먼터에 넘기는
    배열만 절단, 전역 충전 HI는 영향 없음).
    """
    # 사이클 수가 많은 셀(파일 크기 큰 순) 먼저 배정 → 워커 간 부하 균형 개선
    files = sorted(pkl_dir.glob("*.pkl"), key=lambda f: f.stat().st_size, reverse=True)
    _axis_cfg = axis_cfg or {}
    all_seg: list = []
    all_cyc: list = []
    coverage: dict = {}
    if n_workers <= 1:
        for f in tqdm(files, desc=pkl_dir.name):
            seg_rows, cyc_rows, cov = _extract_one_cell((str(f), axis, _axis_cfg, exclude_cv))
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
            futs = {ex.submit(_extract_one_cell, (str(f), axis, _axis_cfg, exclude_cv, _progress_q)): f
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

    2026-08-16: 모델 학습(8_train)은 이제 df_seg를 그대로 읽으므로(세그먼트당 1행,
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


def _save_sample_csvs(per_ds: dict) -> None:
    """데이터셋별 대표 셀 첫 번째 사이클을 cycle/seg 형식으로 CSV 저장.

    per_ds: {dataset_name: (df_cycle, df_seg)}.

    seg CSV엔 이제 그 사이클의 세그먼트 인스턴스가 (n_samples개면) 여러 행으로 그대로
    남는다 — 예전엔 시나리오당 1행으로 뭉개진 걸 저장했었다.
    """
    sample_dir = HI_ROOT / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, (df_cyc, df_seg) in per_ds.items():
        ds_tag = ds_name.lower()
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
    from common.scenario.q_frac_ref import n2_path_tag
    n1 = int(round(axis_cfg.get("n1", 0.4) * 100))
    # n2 조각: 고정 n2면 "n2-20%", q_frac_ref n2 범위 모드면 "n2-10~30%s10"
    # (네 곳 공통 규칙 — common/scenario/q_frac_ref.n2_path_tag 참고)
    n2_frag = n2_path_tag(axis_cfg)
    ns = int(axis_cfg.get("n_samples", 4))
    # 2026-08-10: min_pts 기본값(10)이 아니면 접미사 — 세그먼트 최소 포인트 임계값이
    # 달라지면 완전히 다른 데이터이므로 반드시 다른 경로에 저장(confound 방지, §4.6).
    min_pts = int(axis_cfg.get("min_pts", 10))
    minpts_sfx = f"_minpts{min_pts}" if min_pts != 10 else ""
    # assign="none"(시나리오-only 대조군, docs/260816_RESULTS.md §5 no_scen)이나
    # "mid_nmid"(4시나리오, 게이트 파편화 dose-response 중간점, 2026-09-16 추가)면
    # 반드시 다른 경로에 저장 — position_bin(6시나리오)과 confound 방지(§4.6과 동일
    # 원칙). 예전엔 position_bin이 아닌 값을 전부 "_noscen"으로 뭉뚱그려서, mid_nmid를
    # 추가하면 assign="none" 캐시와 충돌했을 것 — 값별로 별도 접미사를 쓴다.
    _assign = axis_cfg.get("assign", "position_bin")
    assign_sfx = {"position_bin": "", "none": "_noscen", "mid_nmid": "_midnmid"}.get(_assign, f"_{_assign}")
    # 2026-09-09: tile_scope("zone"|"full")를 assign과 분리(라벨 유무 vs 배치 방식,
    # docs/0909_RESULTS.md) — 명시적으로 준 경우만 접미사(미지정 시 기존 assign 연동
    # 동작과 100% 동일한 경로를 써야 하므로 빈 문자열 유지, 하위호환).
    tile_scope = axis_cfg.get("tile_scope")
    tile_scope_sfx = f"_tile{tile_scope}" if tile_scope is not None else ""
    return f"n1-{n1}%_{n2_frag}_N-{ns}{_rand_suffix(axis_cfg)}{minpts_sfx}{assign_sfx}{tile_scope_sfx}"


def _qfref_tag(axis_cfg: dict) -> str:
    """q_frac_ref 파라미터 → 파일/디렉터리 식별 태그.

    n1/n2/N은 q_frac_wide와 동일 규칙(부모 클래스 파라미터 상속) + ref_lag/noise_amp/
    noise_period를 덧붙여 lag·노이즈 파라미터가 다르면 반드시 다른 경로에 저장되게 한다
    (docs/SOC.md §6 Phase 3 "데이터 경로 분리 확인" — §4.6 confound 방지). calibration_*
    (docs/260903_RESULTS.md §1)이 있으면 calib_path_tag로 추가 접미사를 붙인다 —
    미설정(기본)이면 빈 문자열이라 기존 경로와 100% 동일하게 유지된다. offset_amp
    (센서 offset 오차, common/scenario/q_frac_ref.py 모듈 docstring 참고)도 같은
    원칙으로 offset_path_tag를 붙인다."""
    from common.scenario.q_frac_ref import calib_path_tag, offset_path_tag
    base = _qfw_tag(axis_cfg)
    lag = int(axis_cfg.get("ref_lag", 0))
    noise_pct = int(round(axis_cfg.get("noise_amp", 0.03) * 100))
    mode = str(axis_cfg.get("noise_mode", "ou"))
    period = int(round(axis_cfg.get("noise_period_cycles", 200.0)))
    return (f"{base}_lag-{lag}_noise-{noise_pct}%_{mode}-{period}"
            f"{calib_path_tag(axis_cfg)}{offset_path_tag(axis_cfg)}")


def _save_coverage_stats(path: Path, per_ds_cov: dict, axis: str, axis_cfg: dict) -> None:
    """존 포인트 커버리지/누락 비율을 텍스트로 저장.

    두 모드에서 채워진다 —
      - random_segment=True: 랜덤 배치라 설계상 누락이 생김(그 비율을 재는 게 목적).
      - q_frac_ref n2 범위 모드: 100% 커버리지가 **설계 조건**이므로, 여기 100% 미만이
        찍히면 그건 "존 전체 포인트 수 < min_pts라 세그먼트를 못 만든 사이클"뿐이어야
        한다(common/scenario/q_frac_ref.py 모듈 docstring 참고).
    """
    _n2_range = axis_cfg.get("n2_start") is not None and axis_cfg.get("n2_end") is not None
    order = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
    lines = [
        "=" * 72,
        f"  존 포인트 커버리지 통계   (axis={axis}, cfg={json.dumps(axis_cfg, ensure_ascii=False)})",
        "=" * 72,
        "  누락 비율 = 1 - (어느 세그먼트에든 포함된 존 포인트 수 / 존 전체 포인트 수)",
        ("  (n2 범위 모드: 랜덤 길이 타일링이라 커버율 100%가 조건 — 미달분은 "
         "존 포인트 수 < min_pts인 사이클)"
         if _n2_range else
         "  (창 겹침은 covered 1회 집계; 랜덤 추출로 샘플링되지 못한 부분)"),
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
    dataset_group: str = "lfp",
) -> pd.DataFrame:
    """캐시가 있으면 로드, 없으면 전체 추출 후 저장.

    exclude_cv=True: 결과 캐시/저장 경로에 '_ccOnly' 접미사를 붙여 CV 포함
    버전과 별도로 저장한다 (segmenter/axis_cfg 자체는 변경 없음, load_all 참고).
    no_shape=True: preprocess.py --skip-shape로 만든 _4_data_hi/clean_noshape/를
    입력으로 쓰고(main()에서 MIT_DIR/HUST_DIR을 그쪽으로 재지정), 결과 캐시/저장
    경로에 '_noshape' 접미사를 붙여 필터7 있는 기본 버전과 절대 안 겹치게 한다.
    dataset_group="lfp"(기본, MIT+HUST): 기존 캐시 경로 그대로 유지(하위호환).
    dataset_group="ncm"(TJU+CALCE) / "all": 캐시/저장 경로에 '_{group}' 접미사를
    붙여 기존 lfp 결과와 절대 안 겹치게 한다(2026-09-05 신규).
    """
    axis_cfg = dict(axis_cfg or {})
    if dataset_group not in DATASET_GROUPS:
        raise ValueError(f"알 수 없는 dataset_group: {dataset_group!r} (선택: {list(DATASET_GROUPS)})")
    _ds_names = DATASET_GROUPS[dataset_group]

    # 정식(v4) 축은 q_frac_ref 하나뿐(2026-09-24 비-정식 축 일괄 삭제) — 파라미터별
    # 고유 경로 태그(_qfref_tag가 내부적으로 _qfw_tag를 재사용, n1/n2/N은 부모 클래스
    # q_frac_wide와 동일 규칙).
    if axis == "q_frac_ref":
        _tag      = _qfref_tag(axis_cfg)
        _cache    = cache_path.parent / f"hi_features_qfref_{_tag}.pkl"
        _axis_dir = f"q_frac_ref/{_tag}"
    else:
        _cache    = cache_path.parent / f"hi_features_{axis}.pkl"
        _axis_dir = axis

    if exclude_cv:
        _cache    = _cache.with_name(_cache.stem + "_ccOnly.pkl")
        _axis_dir = f"{_axis_dir}_ccOnly"

    if no_shape:
        _cache    = _cache.with_name(_cache.stem + "_noshape.pkl")
        _axis_dir = f"{_axis_dir}_noshape"

    if dataset_group != "lfp":
        _cache    = _cache.with_name(_cache.stem + f"_{dataset_group}.pkl")
        _axis_dir = f"{_axis_dir}_{dataset_group}"

    if not force and _cache.exists():
        print(f"  캐시 로드: {_cache}")
        return pd.read_pickle(_cache)

    from common.scenario import get_segmenter as _get_seg
    _segmenter = _get_seg(axis, {axis: axis_cfg or {}})

    seg_per_ds: dict = {}
    cyc_per_ds: dict = {}
    cov_per_ds: dict = {}
    for ds_name in _ds_names:
        print(f"=== {ds_name} HI 추출 (axis={axis}, exclude_cv={exclude_cv}) ===")
        seg_i, cyc_i, cov_i = load_all(_ds_dir(ds_name), n_workers=n_workers, axis=axis,
                                        axis_cfg=axis_cfg, exclude_cv=exclude_cv)
        _save_per_cell_hi(seg_i, cyc_i, ds_name, axis=_axis_dir)
        seg_per_ds[ds_name] = seg_i
        cyc_per_ds[ds_name] = cyc_i
        cov_per_ds[ds_name] = cov_i
    _save_sample_csvs({ds: (cyc_per_ds[ds], seg_per_ds[ds]) for ds in _ds_names})

    # ScenarioSpec 저장
    _spec_dir = HI_ROOT / _axis_dir
    _spec_dir.mkdir(parents=True, exist_ok=True)
    _segmenter.save_artifacts(_spec_dir)
    print(f"  ScenarioSpec 저장: {_spec_dir / 'scenario_spec.json'}")

    # random_segment 누락 비율 텍스트 저장 (coverage가 있을 때만)
    if any(cov_per_ds.values()):
        _save_coverage_stats(_spec_dir / "coverage_stats.txt", cov_per_ds, axis, axis_cfg)

    seg_all = pd.concat([seg_per_ds[ds] for ds in _ds_names], ignore_index=True)
    cyc_all = pd.concat([cyc_per_ds[ds] for ds in _ds_names], ignore_index=True)
    print("  총 사이클: " + " / ".join(f"{ds} {len(cyc_per_ds[ds]):,}" for ds in _ds_names)
          + "  (세그먼트 인스턴스: "
          + " / ".join(f"{ds} {len(seg_per_ds[ds]):,}" for ds in _ds_names) + ")")

    # Step4 자체 상관분석/플롯 전용 wide df — 모델 학습(8_train)은 seg pkl(native
    # 포맷, 세그먼트당 1행)을 직접 읽으므로 이 df는 학습에 안 쓰인다(_build_flat_correlation_df
    # 참고, docs/260816_RESULTS.md).
    df = _build_flat_correlation_df(seg_all, cyc_all)
    df.to_pickle(_cache)
    print(f"  캐시 저장: {_cache}")
    return df


def compute_correlations(df: pd.DataFrame, datasets: list | None = None) -> pd.DataFrame:
    """Spearman ρ(HI, capacity_Ah) — 데이터셋별(기본 MIT/HUST, ncm 그룹은 TJU/CALCE)."""
    datasets = list(datasets) if datasets is not None else ["MIT", "HUST"]
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")
    result = {}
    for ds in datasets:
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
# 시각화 — 히트맵/산점도/대표 셀 추이는 plot.py 소유(2026-09-29 분리, 이 파일은
# 추출+상관분석만 남긴다). HI_GROUPS/HI_LABELS/HI_GROUP_TAG는 축에 따라 이 모듈이
# 런타임에 재빌드하는 값이라(아래 main() 참고) plot.py가 자체 import하지 않고
# 매 호출 시 인자로 받는다 — 재빌드 전 값을 계속 캐싱하는 걸 막기 위함.
# ─────────────────────────────────────────────────────────────────────────────
from plot import plot_correlation, _plot_sample_hi  # noqa: E402


def _print_run_config(axis: str, axis_cfg: dict, workers: int, force: bool,
                       exclude_cv: bool, skip_shape: bool) -> None:
    """HI 추출 실행 조건(축·파라미터·경로·랜덤옵션)을 터미널에 요약 출력."""
    from common.scenario import get_segmenter as _gs
    try:
        _spec = _gs(axis, {axis: axis_cfg}).get_spec()
        _params = _spec.params or {}
        _n_scen = _spec.n_scenarios
    except Exception:
        _params, _n_scen = dict(axis_cfg), "?"

    # 데이터 저장 경로 태그 (load_or_extract 와 동일 규칙)
    if axis == "q_frac_ref":
        _axis_dir = f"q_frac_ref/{_qfref_tag(axis_cfg)}"
    else:
        _axis_dir = axis
    _exclude_cv = exclude_cv
    if _exclude_cv:
        _axis_dir = f"{_axis_dir}_ccOnly"
    if skip_shape:
        _axis_dir = f"{_axis_dir}_noshape"

    _is_rand = bool(axis_cfg.get("random_segment", False))
    # q_frac_ref n2 범위 모드 — 세그먼트 길이가 고정이 아니라 격자에서 랜덤 추첨되고,
    # 존을 100% 덮도록 타일링된다(common/scenario/q_frac_ref.py 모듈 docstring).
    _n2_range = (axis_cfg.get("n2_start") is not None
                 and axis_cfg.get("n2_end") is not None)
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
    if _n2_range:
        _step = float(axis_cfg.get("n2_step", 0.1))
        _grid = [round(float(axis_cfg["n2_start"]) + j * _step, 10)
                 for j in range(int(round((float(axis_cfg["n2_end"])
                                           - float(axis_cfg["n2_start"])) / _step)) + 1)]
        print(f"  n2 범위 모드     : True   길이 격자={_grid}  "
              f"seed={int(axis_cfg.get('n2_seed', 20260903))}")
        print(f"                    (존을 랜덤 길이로 타일링 — 커버리지 100% 보장, "
              f"n_samples 무시)")
    print(f"  워커 수          : {workers}")
    print(f"  force 재추출     : {force}")
    print(f"  exclude_cv      : {_exclude_cv}"
          + ("  (충전 세그먼트 HI는 CC 구간만 사용)" if _exclude_cv else ""))
    print(f"  데이터 저장 경로 : _4_data_hi/{_axis_dir}/{{seg,cycle}}/")
    if _is_rand:
        print(f"  누락비율 저장    : _4_data_hi/{_axis_dir}/coverage_stats.txt")
    print("=" * w + "\n")


def main():
    # 2026-09-29: 이 스크립트의 실행 파라미터는 parameters.py에서만 읽는다 — 예전엔
    # --n1/--n2/--axis-config/--dataset-group/--exclude-cv/--skip-shape 등 CLI로
    # parameters.py를 우회해 값을 바꿀 수 있었지만(대부분 PowerShell JSON 인용 우회
    # 목적), 실험 조건을 바꾸려면 이제 parameters.py 자체를 고치고 실행한다. 여러
    # 조건을 순차 실행하려면 매 케이스 전에 parameters.py를 패치하고 실행 후 원복하는
    # 드라이버 스크립트를 쓴다(docs/REFACTORING.md 2026-09-29 항목).
    workers = min(P.ACTIVE_WORKERS, os.cpu_count() or 1)
    force = P.ACTIVE_FORCE_EXTRACT
    dataset_group = P.FIXED_DATASET_GROUP
    exclude_cv = P.FIXED_EXCLUDE_CV
    skip_shape = P.FIXED_SKIP_SHAPE
    _axis = P.FIXED_SEG_AXIS
    _axis_cfg: dict = dict(P.ACTIVE_AXIS_CONFIG)

    if skip_shape:
        if dataset_group != "lfp":
            print(f"[ERROR] FIXED_SKIP_SHAPE는 clean_noshape/{{MIT,HUST}}만 있음 — "
                  f"FIXED_DATASET_GROUP={dataset_group!r}(TJU/CALCE 포함)와 함께 쓸 수 없음")
            return
        global MIT_DIR, HUST_DIR
        MIT_DIR  = DATA_4_HI_ROOT / "clean_noshape" / "MIT"
        HUST_DIR = DATA_4_HI_ROOT / "clean_noshape" / "HUST"
        print(f"[skip_shape] 입력 경로 재지정: MIT_DIR={MIT_DIR}  HUST_DIR={HUST_DIR}")

    # qfrac 이외 축은 세그먼트 이름이 달라 모듈 레벨 HI 상수를 재빌드
    if _axis != "qfrac":
        from common.scenario import get_segmenter as _get_seg_hi
        _seg_names_hi = _get_seg_hi(_axis, {_axis: _axis_cfg}).get_spec().scenario_names
        global HI_GROUPS, ALL_HI_KEYS, HI_LABELS, HI_GROUP_TAG
        HI_GROUPS, ALL_HI_KEYS, HI_LABELS = _build_hi_groups(_seg_names_hi)
        HI_GROUP_TAG = {k: g for g, ks in HI_GROUPS.items() for k in ks}
        print(f"[hi] 세그먼트 이름 재빌드: {_seg_names_hi}")

    # ── 실행 조건 요약 출력 (추출 진입 전) ──────────────────────────────────
    _print_run_config(_axis, _axis_cfg, workers, force, exclude_cv, skip_shape)

    _ds_names = DATASET_GROUPS[dataset_group]

    df = load_or_extract(n_workers=workers, force=force,
                         axis=_axis, axis_cfg=_axis_cfg, exclude_cv=exclude_cv,
                         no_shape=skip_shape, dataset_group=dataset_group)
    print(f"\n총 사이클: {len(df):,}")

    print("\n=== Spearman ρ 계산 ===")
    corr = compute_correlations(df, datasets=_ds_names)

    for gname, gkeys in HI_GROUPS.items():
        avail = [k for k in gkeys if k in corr.index]
        if not avail:
            continue
        sub = corr.loc[avail].copy()
        sub["|ρ| avg"] = sub.abs().mean(axis=1)
        sub = sub.sort_values("|ρ| avg", ascending=False)
        print(f"\n── {gname} ──")
        print(sub.to_string(float_format=lambda x: f"{x:+.3f}"))

    if _axis == "q_frac_ref":
        _dir_suffix = f"_qfref_{_qfref_tag(_axis_cfg)}"
    else:
        _dir_suffix = f"_{_axis}"
    if exclude_cv:
        _dir_suffix += "_ccOnly"
    if skip_shape:
        _dir_suffix += "_noshape"
    if dataset_group != "lfp":
        _dir_suffix += f"_{dataset_group}"
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
        plot_correlation(corr, df, out, HI_GROUPS, HI_LABELS, HI_GROUP_TAG,
                         n_top=4, datasets=_ds_names)
    except Exception as _e:
        print(f"[경고] hi_correlation.png 생성 실패(캐시는 정상 저장됨): {_e}")

    out_dir = STEP_DIR / "outputs" / (date.today().strftime("%m%d") + _dir_suffix)
    print("\n=== 대표 셀 HI 플롯 ===")
    try:
        _plot_sample_hi(df, corr, out_dir, HI_LABELS, HI_GROUP_TAG, datasets=_ds_names)
    except Exception as _e:
        print(f"[경고] 대표 셀 HI 플롯 생성 실패(캐시는 정상 저장됨): {_e}")

    # ── hi_segment_viz.py 플롯 (trend + overlay) ─────────────────────────────
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("_hi_viz", STEP_DIR / "hi_segment_viz.py")
        _viz = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_viz)
        for _cat, _cat_title, _fname in _viz.CATEGORIES:
            print(f"\n=== 세그먼트 HI 추이 ({_cat}) ===")
            _viz.plot_segment_hi_trend(df, hi_plot_dir / _fname, _cat, _cat_title)
        for _cat, _cat_title, _fname in _viz.OVERLAY_CATEGORIES:
            print(f"\n=== 시나리오 오버레이 ({_cat}) ===")
            _viz.plot_segment_hi_overlay(df, hi_plot_dir / _fname, _cat, _cat_title)
    except Exception as _e:
        print(f"[경고] trend/overlay 플롯 생성 실패: {_e}")

    print("완료!")


if __name__ == "__main__":
    main()
