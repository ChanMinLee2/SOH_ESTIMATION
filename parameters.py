"""
parameters.py

run_pipeline.py의 파라미터 중앙 관리 파일. 2026-09-21 리팩토링 — 실제로 자주 바꿔가며
쓰는 파라미터와, 거의 항상 기본값 그대로 쓰던 파라미터를 분리했다(이 세션 내내 실행한
noscen/scen/HI63/64/66/멀티시드 명령어들을 기준으로 실측 사용 빈도를 반영).

- 위쪽(ACTIVE_*): run_pipeline.py가 이 값들을 argparse 기본값으로 참조한다 — CLI로
  계속 오버라이드 가능. 실험마다 바뀌는 값들.
- 아래쪽(FIXED_*): run_pipeline.py에 CLI 플래그 자체가 없다 — 이 파일의 값이 그대로
  하위 스크립트에 전달된다. 바꾸려면 이 파일을 직접 수정할 것.

각 상수 옆 주석의 "(구 --xxx)"는 리팩토링 전 run_pipeline.py에서 쓰던 CLI 플래그 이름 —
과거 문서/기억 속 명령어를 새 구조로 옮길 때 대응 관계를 찾기 위한 참고용.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# 자주 바꿔가며 쓰는 파라미터 — run_pipeline.py의 CLI 플래그로 계속 노출됨
# ═══════════════════════════════════════════════════════════════════════════

# ── 스텝 범위 ────────────────────────────────────────────────────────────
ACTIVE_FROM_STEP = 1                    # 위치 인자 기본값(구 from_step 위치인자)

# ── 실행 환경 ────────────────────────────────────────────────────────────
ACTIVE_WORKERS = 8                      # 데이터 스텝(1~5) 병렬 프로세스 수 상한 — Step 1~5
                                         # 스크립트 전부 이 값을 min(P.ACTIVE_WORKERS,
                                         # os.cpu_count())로 참조한다(2026-09-23 통일 —
                                         # 예전엔 스크립트마다 3/4/CPU-2 등 제각각이었음).
ACTIVE_FORCE_EXTRACT = False            # Step 4 캐시 무시 강제 재추출(구 --force-extract)
ACTIVE_DATASET = "all"                  # Step 1~2(convert_unified.py/preprocess.py)의
                                         # --dataset (mit|hust|tju|calce|all). Step 4의
                                         # --dataset-group(hi_correlation.py, lfp|ncm|all —
                                         # 값 체계가 달라 별도 상수, 아래 FIXED_DATASET_GROUP)과
                                         # 다른 상수다.
FIXED_DATASET_GROUP = "lfp"             # 구 hi_correlation.py --dataset-group 하드코딩
                                         # 기본값(lfp|ncm|all) — run_pipeline.py는 이 플래그를
                                         # 아직 CLI로 노출하지 않음(2026-09-24 단일화).

# ── 커널/다중공선성 파이프라인 설정(Step 6~8 산출물, Step 9가 소비) ───────
ACTIVE_KERNEL_FEATURES_PKL = None       # 구 --kernel-features-pkl (None=자동 경로/fallback)
ACTIVE_INTERACTION_JSON = None          # 구 --interaction-json (None=자동 경로/fallback)
ACTIVE_COMBINED_REDUNDANCY_JSON = None  # 구 --combined-redundancy-json (None=자동 경로)
ACTIVE_MAX_GROUP_SIZE = 4               # 구 --max-group-size (시너지 그룹 크기 상한)
ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD = 0.9  # 구 --synergy-redundancy-threshold (1차 배제)

# ── 학습(Step 9) ─────────────────────────────────────────────────────────
ACTIVE_MAX_EPOCHS = None                # 구 --max-epochs (None=yaml training.epochs)
ACTIVE_PATIENCE = None                  # 구 --patience (None=train.py 기본 60)
ACTIVE_BATCH_SIZE = None                # 구 --batch-size (None=yaml training.batch_size)
ACTIVE_HI_COST_WEIGHTED_L0 = False      # 구 --hi-cost-weighted-l0
ACTIVE_LAMBDA_L0_OVERRIDE = 0.000237    # 구 --lambda-l0-override — 2026-09-21부터 기본
                                         # 활성 고정값으로 전환(이 세션 모든 run이 이 값을
                                         # 명시했음, 더는 "옵션"이 아니라 표준 설정).
                                         # 다른 값을 스윕하려면 이 상수를 바꿀 것.

# ── N_HI 토글(raw HI leakage 제외 범위, docs/MODEL_FLOW.md §13 N_HI 토글 참고) ──
ACTIVE_N_HI = 64                        # 구 --include-stat-leak/--exclude-dqdv-leak 두
                                         # 불리언 플래그를 이 단일 선택으로 통합
                                         # (2026-09-21). 63=q_abs/energy/dqdv_area 전부
                                         # 제외, 64=q_abs/energy만 제외(기본), 66=전부 포함.
N_HI_CHOICES = (63, 64, 66)
N_HI_TO_ENV = {
    # N_HI: (SOH_EXCLUDE_STAT_LEAK, SOH_EXCLUDE_DQDV_LEAK)
    66: ("0", "0"),
    64: ("1", "0"),
    63: ("1", "1"),
}

# ── 태그/경로 ────────────────────────────────────────────────────────────
ACTIVE_P1_TAG = "p1v4_full"             # 구 --p1-tag
ACTIVE_REP_CELLS = None                 # 구 --rep-cells (None=데이터셋별 5개 자동 선정)
ACTIVE_DATA_DIR = None                  # 구 --data-dir — noscen/scen/HI63.. 축마다 값이
                                         # 다 달라서 고정 불가, CLI로 계속 필요. None이면
                                         # run_pipeline.py가 --data-dir 자체를 전달하지 않고,
                                         # 하위 스크립트가 FIXED_CANONICAL_DATA_DIR(아래)로
                                         # 최종 폴백한다.
ACTIVE_SEG_DATA_DIR = None              # 구 --seg-data-dir (위와 동일 이유)

# ── 축 설정(Step 4~9 공통, q_frac_ref) — 이 딕셔너리가 유일한 소스 ──────────
# 2026-09-23: --n1/--n2/--n-samples 등 개별 단축 플래그를 없애고 이 딕셔너리 하나로
# 통합했다(파라미터 운용의 경우의 수를 줄여 일관성을 확보하기 위한 리팩토링 —
# 이전엔 run_pipeline.py 자체 단축 인자와 hi_correlation.py 자체 단축 인자가 서로
# 다른 키 집합을 알고 있어, 어느 경로로 실행하느냐에 따라 같은 "옵션 없음"이 다른
# 결과를 냈다). run_pipeline.py는 이 값을 json.dumps()해서 --axis-config로 그대로
# 하위 스크립트(Step 4~9)에 전달하는 것 외에 축 파라미터를 다루지 않는다. 값을
# 바꾸려면 이 딕셔너리를 직접 수정하거나, run_pipeline.py --axis-config로 통째로
# 오버라이드할 것(부분 오버라이드 없음 — 그것도 "경우의 수"이므로).
ACTIVE_AXIS_CONFIG: dict = {            # model_lib/config/main_qfref_S.yaml: scenario.axis_config
    "n1": 0.35,                         # 와 100% 동일(단일 소스) — 이 값이 곧 train.py의
    "n2": 0.20,                         # 하드코딩 fallback data_dir(n1-35%_n2-20%_N-2_minpts5_lag-1_
    "n_samples": 2,                     # noise-3%_ou-200_calib-100_offA-5mA)을 만든 실제 축 설정이다.
    "ref_lag": 1,                       # ⚠️ "scen_lag1zone"류 실험 변형(tile_scope=zone, min_pts/
    "noise_amp": 0.03,                  # calibration/offset 없음)과 혼동 금지 — 그건 별도 실험이고
    "noise_mode": "ou",                 # 이게 정식(canonical) 설정이다(2026-09-23 정정).
    "noise_period_cycles": 200.0,
    "min_pts": 5,
    "calibration_period": 100,
    "offset_amp": 0.005,
}

# ── 모델/재현성 ──────────────────────────────────────────────────────────
ACTIVE_REGRESSION_MODEL = None          # 구 --regression-model (None=train.py 기본 mlp)
ACTIVE_SEED = None                      # 구 --seed (None→내부적으로 42로 해석)
ACTIVE_SPLIT_SEED = None                # 구 --split-seed (None→내부적으로 42로 해석)


# ═══════════════════════════════════════════════════════════════════════════
# 거의 안 바꾸는 파라미터 — CLI 노출 없음, 이 값 그대로 하위 스크립트에 전달됨
# ═══════════════════════════════════════════════════════════════════════════

# ── 공통 모델 설정 파일 ──────────────────────────────────────────────────
FIXED_PHASE1_MODEL_CONFIG = "model_lib/config/main_qfref_S.yaml"   # 구 --phase1-model-config

# ── 세그멘테이션 축 종류(q_frac_ref 고정 — 이번 세션 전부 이 축만 사용) ────
FIXED_SEG_AXIS = "q_frac_ref"           # 구 --seg-axis. vwindow 등 다른 축을 쓰려면
                                         # 이 상수를 바꿀 것(--axis-config만으로는 축
                                         # 종류 자체를 못 바꿈).

# ── ACTIVE_AXIS_CONFIG로 Step4가 실제 추출한 데이터의 저장 경로 ────────────
# train.py 등이 --data-dir/--seg-data-dir도 없고 --model-config yaml에도
# 없을 때 쓰는 최종 폴백(2026-09-23 — 예전엔 이 값이 train.py 안에 별도로
# 하드코딩돼 있다가 ACTIVE_AXIS_CONFIG와 조용히 어긋난 적이 있었다). ACTIVE_AXIS_CONFIG를
# 바꾸면 Step4가 만드는 실제 경로도 바뀌므로 이 상수도 반드시 같이 갱신할 것.
FIXED_CANONICAL_DATA_DIR = (
    "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/"
    "n1-35%_n2-20%_N-2_minpts5_lag-1_noise-3%_ou-200_calib-100_offA-5mA/cycle"
)
FIXED_CANONICAL_SEG_DATA_DIR = (
    "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/"
    "n1-35%_n2-20%_N-2_minpts5_lag-1_noise-3%_ou-200_calib-100_offA-5mA/seg"
)

# ── Step 6(HI-시나리오 상호작용 검정) ────────────────────────────────────
FIXED_INTERACTION_TAG = None            # 구 --interaction-tag (None=--p1-tag에서 자동 파생)
FIXED_INTERACTION_ALPHA = 0.05          # 구 --interaction-alpha
FIXED_INTERACTION_MIN_EFFECT_SIZE = 0.1 # 구 --interaction-min-effect-size

# ── Step 7(HI 시너지 그룹 구성) ──────────────────────────────────────────
FIXED_MIN_PARTIAL_CORR = 0.02           # 구 --min-partial-corr
FIXED_PREFILTER_TOP_M = 15              # 구 --prefilter-top-m
FIXED_GLOBAL_DEDUP = False              # 구 --global-dedup — ⚠️ docs 상 "v4 정식 레시피"는
                                         # True를 썼다고 돼 있으나, 이 세션에 실제로 돌린
                                         # noscen/scen/HI63/64/66 run은 전부 이 플래그를
                                         # 안 줘서 False로 실행됐다 — 실측 동작을 그대로
                                         # 기본값으로 고정(2026-09-21).
FIXED_SHUFFLE_SEED = 42                 # 구 --shuffle-seed (v-ctrl 무작위 대조군 전용 —
                                         # synergy.py/interaction.py 공용,
                                         # 2026-09-24 하드코딩 기본값에서 이전)
FIXED_SYNERGY_TAG = None                # 구 --synergy-tag (None=--p1-tag에서 자동 파생)

# ── Step 8(커널 HI 피처 생성) ────────────────────────────────────────────
FIXED_KERNEL_SYNERGY_GROUPS_JSON = None # 구 --kernel-synergy-groups-json (None=Step 7 자동 경로)
FIXED_KERNEL_ALPHA = 1.0                # 구 --kernel-alpha (Ridge 정규화 강도)
FIXED_KERNEL_GAMMA = None               # 구 --kernel-gamma (None=sklearn 기본 1/n_features)
FIXED_KERNEL_N_COMPONENTS = 100         # 구 --kernel-n-components (Nystroem 랜드마크 수)
FIXED_KERNEL_REDUNDANCY_THRESHOLD = 0.9 # 구 --kernel-redundancy-threshold (2차 배제, 커널끼리 pooled)
FIXED_KERNEL_MAX_FEATURES = None        # 구 --kernel-max-features (None=무제한)
FIXED_MIN_RAW_PARTIAL_CORR = None       # 구 --min-raw-partial-corr (None=비활성)
FIXED_COMBINED_REDUNDANCY_THRESHOLD = 0.95  # 구 --combined-redundancy-threshold (3차 배제, 시나리오별)
FIXED_KERNEL_TAG = None                 # 구 --kernel-tag (None=--p1-tag에서 자동 파생)

# ── Step 9(SCR Phase 1 학습) ─────────────────────────────────────────────
FIXED_TRAIN_CYCLE_FRAC = None           # 구 --train-cycle-frac (None=1.0, 전체 사용)
FIXED_BETA_MIN = None                   # 구 --beta-min (None=train.py 기본 0.1)
FIXED_L0_WARMUP_EPOCHS_OVERRIDE = None  # 구 --l0-warmup-epochs-override (None=yaml 값)
FIXED_L0_NORM_CONSTANT = None           # 구 --l0-norm-constant (None=n_scenarios로 나눔)
FIXED_VAL_RMSE_EPSILON = 0.0005         # 구 --val-rmse-epsilon (체크포인트 선택 기준, 2026-09-18)
FIXED_DEVICE = None                     # 구 --device (None=auto) — Step 9/10 공용

# ── 데이터 변형(Step 2/4 전용, 이번 세션 미사용) ─────────────────────────
FIXED_EXCLUDE_CV = False                # 구 --exclude-cv (CC→CV 이후 구간 제외)
FIXED_SKIP_SHAPE = False                # 구 --skip-shape (형상 이상치 필터 비활성화)


# ═══════════════════════════════════════════════════════════════════════════
# Step 9(train.py) 모델/학습 설정 — yaml 흡수
# ═══════════════════════════════════════════════════════════════════════════
# model_lib/config/fixed.yaml + main_qfref_S.yaml을 유틸리티(utils/io_utils.py:
# load_config, _deep_merge)로 실제로 병합한 결과를 그대로 옮긴 것(2026-09-23,
# 실 환경에서 load_config() 호출 출력을 그대로 사용 — 손으로 옮기며 생길 수 있는
# 오탈자 방지). train.py는 --model-config를 안 주면 이 딕셔너리를
# deepcopy해서 cfg로 쓴다. all4/hust_only/noscen/p60/transformerL 등 다른 프리셋을
# 재현하려면 지금처럼 --model-config model_lib/config/<preset>.yaml을 명시하면 된다
# (기존 yaml 로딩 경로 그대로 유지 — 이스케이프 해치).
#
# scenario.axis/axis_config는 별도로 안 쓰고 FIXED_SEG_AXIS/ACTIVE_AXIS_CONFIG를
# 그대로 참조한다 — 축 설정이 여기 또 복사되면 두 값이 다시 어긋날 수 있기 때문
# (이번 정리를 시작하게 만든 바로 그 문제). data.data_dir/seg_data_dir도 마찬가지로
# FIXED_CANONICAL_DATA_DIR/SEG_DATA_DIR을 참조한다.
P1_MODEL_CONFIG: dict = {
    "data": {
        "is_real_input": False,
        "data_dir": FIXED_CANONICAL_DATA_DIR,
        "seg_data_dir": FIXED_CANONICAL_SEG_DATA_DIR,
        "output_dir": "_5_data_model_scr",
        "is_cross_dataset_evaluate": False,
        "split_seed": 42,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "min_cycles_per_cell": 10,
        "io_workers": 16,
        "use_initial_capacity": True,
        "nominal_capacities": {"MIT": 1.1, "HUST": 1.2},
        "gates_from": None,
        "datasets": ["MIT", "HUST"],
    },
    "classifier": {
        # 2026-09-25: type/is_auto_mk_selection/probe_m_count 삭제 — 전부 구 시나리오
        # 분류기 프레임워크(model_lib/models/scenario_classifier.py, 삭제됨) 전용
        # 키였고 어디서도 읽지 않았다. charge_probe_m/discharge_probe_m만 실제로
        # train.py의 probe gate top-m 선정에 쓰인다.
        "charge_probe_m": 10,
        "discharge_probe_m": 10,
    },
    "model": {
        "d_probe": 64,
        "d_head": 128,
        "dropout": 0.2,
        "tr_n_heads": 4,
        "tr_n_layers": 2,
        "tr_d_ff": 512,
        "resnet_n_blocks": 4,
        "resnet_d_hidden_factor": 2.0,
        "regression_model": "mlp",
        "mlp_hidden_dims": [128, 64],
        "with_raw_flat": False,
        # 2026-09-25: with_raw_cnn/raw_cnn_pretrained_from 삭제 — 의존하는
        # models/raw_cnn.py 자체가 repo에 없었고(있었어도 train.py/test.py가 v4에서
        # 항상 강제 False), scr_model.py의 RawCNN 로딩 분기도 같이 제거했다.
    },
    "loss": {
        "lambda_scen": 0.01,
        "lambda_l0": 0.01,
        "lambda_l0_auto": True,
        "lambda_l0_schedule": "delayed_warmup",
        "lambda_l0_warmup_epochs": 50,
        "lambda_l0_ramp_epochs": 100,
        "leak_cols": ["stat_q_abs", "stat_energy_seg"],
    },
    "training": {
        "epochs": 500,
        "batch_size": 2048,
        "scheduler": "cosine",
        "warmup_epochs": 10,
        "grad_clip": 1.0,
        "log_interval": 10,
        "run_overfit_test": False,
        "overfit_test_samples": 1024,
        "overfit_test_epochs": 300,
        "lr": 2.0e-4,
        "weight_decay": 1.0e-3,
        "early_stop_patience": 100,
    },
    "evaluation": {
        "metrics": ["rmse", "mae", "r2", "mape"],
        "rep_cells_per_dataset": 5,
    },
    "uq": {
        "enabled": True,
        "prior_precision": 1.0,
        "optimize_prior": True,
        "noise_std": None,
    },
    "scenario": {
        "axis": FIXED_SEG_AXIS,
        "axis_config": ACTIVE_AXIS_CONFIG.copy(),
    },
    "regression": {
        "scen_k_count": 25,
    },
}
