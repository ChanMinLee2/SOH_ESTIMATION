"""
4_hi_analysis/constants.py

hi_correlation.py(Step 4)가 쓰는 순수 데이터 정의 — 경로, 데이터셋 그룹, HI
라벨/그룹 메타, 숫자 상수. 함수는 두지 않는다(2026-09-29 main/plot/logics/
constants 4파일 분리 — 로직은 logics.py, 그림은 plot.py, main()은
hi_correlation.py가 이 모듈을 값 소스로 참조).

HI_GROUPS/ALL_HI_KEYS/HI_LABELS/HI_GROUP_TAG는 여기 아래에서 q_frac_ref 표준
6-시나리오(dis_hi/dis_mid/dis_lo/chg_lo/chg_mid/chg_hi) 기준으로 한 번 빌드해두지만,
hi_correlation.py::main()은 실행마다 logics.build_hi_groups()로 다시 빌드한 값을
지역 변수로 써서 plot.py에 명시적으로 넘긴다(축 파라미터에 따라 세그먼트 이름이
갈릴 수 있어 이 모듈의 초기값을 그대로 신뢰하면 안 됨) — 여기 있는 값은
tools/ 스크립트가 `from hi_correlation import HI_GROUPS` 식으로 참조할 때 쓰는
기본값 용도로만 남긴다.
"""

from collections import OrderedDict
from pathlib import Path

from data_directories import DATA_4_HI_ROOT, PKL_CACHE_ROOT
from utils.hi_schema import STAT_KEYS, DIFF_KEYS, LFP_KEYS, MORPH_KEYS

STEP_DIR     = Path(__file__).resolve().parent
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

# ── 시간/진행률 상수 ─────────────────────────────────────────────────────────
_PHASE_POS = 0.01        # A 초과 → charge
_PHASE_NEG = -0.01       # A 미만 → discharge
_PROGRESS_BATCH = 20     # 사이클 N개마다 진행률 큐에 보고 (IPC 오버헤드 절감)

# ── 형태학적 거리(DTW/Fréchet) 상수 ──────────────────────────────────────────
_MORPH_GRID = 50   # 보간 그리드 해상도 (속도-정밀도 균형)
_DTW_BAND   = 5    # Sakoe-Chiba 밴드 (그리드의 10% = 위상 이동 허용폭)
# _dtw_batch 청크 크기 — 장수명 셀(사이클 수 많음) × n_samples>1 조합에서 (N,50,50)
# 배열을 한 번에 만들면 N이 수천~수만까지 커져 메모리 부족이 날 수 있다(2026-08-17
# 실측: HUST 1782사이클×4샘플=7128로 136MiB 임시 배열 할당 실패, --workers 다중 프로세스
# 동시 실행 시 압박 가중). N을 이 크기로 잘라 처리해 피크 메모리를 셀 크기와 무관하게
# 상한선 이하로 유지한다.
_DTW_CHUNK  = 2000

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

# ── 전체 HI 키 / 레이블 / 그룹 초기값(q_frac_ref 표준 6-시나리오 기준) ─────────
# hi_correlation.py::main()이 실행마다 logics.build_hi_groups()로 다시 빌드해
# 지역 변수로 쓰므로, 여기 값은 tools/ 스크립트의 기본 참조용일 뿐이다.
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
