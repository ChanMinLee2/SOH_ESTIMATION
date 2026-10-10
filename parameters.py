"""
parameters.py — run_pipeline.py 및 하위 스크립트(Step 1~9)의 파라미터 중앙 관리 파일.

- ACTIVE_*: run_pipeline.py의 CLI 플래그 기본값. 실행 시 CLI로 오버라이드 가능.
- FIXED_*: CLI로 노출되지 않음 — 값을 바꾸려면 이 파일을 직접 수정.
"""

from __future__ import annotations

# 대용량 중간 산출물 드라이브 루트(실제 경로는 data_directories.py가 관리)
from data_directories import DATA_4_HI_ROOT

# ═══════════════════════════════════════════════════════════════════════════
# 자주 바꿔가며 쓰는 파라미터 — run_pipeline.py의 CLI 플래그로 노출됨
# ═══════════════════════════════════════════════════════════════════════════

# ── 스텝 범위 ────────────────────────────────────────────────────────────
ACTIVE_FROM_STEP = 1                    # 파이프라인 시작 스텝 번호(1~9)

# ── 실행 환경 ────────────────────────────────────────────────────────────
ACTIVE_WORKERS = 8                      # Step 1~5 데이터 처리 병렬 프로세스 수 상한
ACTIVE_FORCE_EXTRACT = False            # Step 4 캐시 무시하고 강제 재추출 여부
ACTIVE_DATASET = "all"                  # Step 1~2 대상: mit|hust|tju|calce|all
FIXED_DATASET_GROUP = "lfp"             # Step 4 데이터셋 그룹: lfp|ncm|all
FIXED_CANONICAL_DATASETS = ["MIT", "HUST"]  # Step 5~9에서 실제 로드할 데이터셋 목록

# ── 커널/다중공선성 파이프라인 설정(Step 5~8 산출물, Step 9가 소비) ───────
ACTIVE_KERNEL_FEATURES_PKL = None       # kernel.py 산출물 경로. None=자동탐색
ACTIVE_INTERACTION_JSON = None          # interaction.py 산출물 경로. None=자동탐색
ACTIVE_COMBINED_REDUNDANCY_JSON = None  # 결합 다중공선성 배제 결과 경로. None=자동탐색
ACTIVE_MAX_GROUP_SIZE = 4               # 시너지 그룹 최대 크기(HI 개수)
ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD = 0.9  # 다중공선성 판정 상관계수 임계값(0~1)

# ── 학습(Step 8) ─────────────────────────────────────────────────────────
ACTIVE_MAX_EPOCHS = None                # 최대 학습 에폭. None=yaml 기본값(500)
ACTIVE_PATIENCE = None                  # 조기종료 patience(에폭 수). None=60
ACTIVE_BATCH_SIZE = None                # 배치 크기. None=yaml 기본값(2048)
ACTIVE_HI_COST_WEIGHTED_L0 = False      # L0 페널티에 HI 카테고리별 비용 가중 적용 여부
ACTIVE_USE_PROBE_KERNEL = False         # 분류기(Stage A)에 커널 HI 후보 사용 여부.
                                         # True면 커널 HI 다중공선성도 함께 적용됨
ACTIVE_WARMSTART_BRANCH_EPOCH = None     # 웜스타트 분기 커리큘럼 — None=비활성(기본,
                                         # scen_gates=시나리오별 n개로 바로 학습). 정수
                                         # T를 주면 epoch<T 동안 scen_gates가 방향별
                                         # 공유 게이트 2개(충전 3개 시나리오가 1개,
                                         # 방전 3개가 다른 1개를 공유)로 학습되다가
                                         # epoch==T에 시나리오별 n개로 분기함.
                                         # kernel_features_pkl/interaction_json과
                                         # 함께 사용 가능.
ACTIVE_LAMBDA_L0_OVERRIDE = 0.000237    # L0 페널티 가중치 고정값. None=yaml/auto 계산

# ── N_HI 토글(raw HI leakage 제외 범위) ──────────────────────────────────
ACTIVE_N_HI = 64                        # raw HI 개수 — 63(leak 전부 제외) |
                                         # 64(일부만 제외, 기본) | 66(전부 포함)
N_HI_CHOICES = (63, 64, 66)
N_HI_TO_ENV = {
    # N_HI: (SOH_EXCLUDE_STAT_LEAK, SOH_EXCLUDE_DQDV_LEAK)
    66: ("0", "0"),
    64: ("1", "0"),
    63: ("1", "1"),
}

# ── 태그/경로 ────────────────────────────────────────────────────────────
ACTIVE_P1_TAG = "refact"                # 산출물 파일명에 붙는 실행 태그
ACTIVE_REP_CELLS = None                 # 대표 셀 목록. None=데이터셋별 5개 자동 선정
ACTIVE_DATA_DIR = None                  # Step 4 산출 데이터 경로. None=아래
                                         # FIXED_CANONICAL_DATA_DIR로 폴백
ACTIVE_SEG_DATA_DIR = None              # 세그먼트 데이터 경로. None=아래
                                         # FIXED_CANONICAL_SEG_DATA_DIR로 폴백

# ── 축 설정(Step 4~9 공통, q_frac_ref) — 이 딕셔너리가 유일한 소스 ──────────
# 바꾸려면 이 딕셔너리를 직접 수정하거나 run_pipeline.py --axis-config로 통째로
# 오버라이드(부분 오버라이드는 지원 안 함).
ACTIVE_AXIS_CONFIG: dict = {
    "n1": 0.35,                 # 저/중 SOC 구간 경계 비율
    "n2": 0.20,                 # 세그먼트 길이 비율(전체 구간 대비)
    "n_samples": 2,             # 존(zone)당 샘플링할 세그먼트 수
    "ref_lag": 1,                # SOC 레퍼런스로 쓸 과거 사이클 수(0=현재 사이클 자신)
    "noise_amp": 0.03,          # 레퍼런스 SOC 노이즈 진폭
    "noise_mode": "ou",         # 노이즈 모델: ou(평균회귀 랜덤워크) | sine(사인파)
    "noise_period_cycles": 200.0,  # noise_mode="sine"일 때의 주기(사이클 수)
    "min_pts": 5,                # 세그먼트 최소 포인트 수(미만이면 병합/제외)
    "calibration_period": 100,  # 센서 재보정 주기(사이클). None=재보정 없음
    "offset_amp": 0.005,        # 센서 오프셋 노이즈 진폭
}

# ── 모델/재현성 ──────────────────────────────────────────────────────────
FIXED_DEFAULT_SEED = 42                 # ACTIVE_SEED/ACTIVE_SPLIT_SEED가 None일 때의
                                         # 공통 폴백값(모델 초기화/셀 분할/셔플 시드는
                                         # 서로 독립적으로 다른 값 지정 가능)
ACTIVE_SEED = None                      # 모델 초기화 시드. None→FIXED_DEFAULT_SEED
ACTIVE_SPLIT_SEED = None                # 셀 train/val/test 분할 시드. None→FIXED_DEFAULT_SEED


# ═══════════════════════════════════════════════════════════════════════════
# 거의 안 바꾸는 파라미터 — CLI 노출 없음, 이 값 그대로 하위 스크립트에 전달됨
# ═══════════════════════════════════════════════════════════════════════════

# ── 세그멘테이션 축 종류 ─────────────────────────────────────────────────
FIXED_SEG_AXIS = "q_frac_ref"           # 세그멘테이션 축 종류(다른 축을 쓰려면
                                         # 이 상수 자체를 변경해야 함 — axis_config만
                                         # 바꿔서는 축 종류를 못 바꿈)

# ── ACTIVE_AXIS_CONFIG로 Step 4가 실제 추출한 데이터의 저장 경로 ──────────
_CANONICAL_AXIS_DIR = (
    DATA_4_HI_ROOT / "q_frac_ref"
    / "n1-35%_n2-20%_N-2_minpts5_lag-1_noise-3%_ou-200_calib-100_offA-5mA"
)
FIXED_CANONICAL_DATA_DIR = str(_CANONICAL_AXIS_DIR / "cycle").replace("\\", "/")
FIXED_CANONICAL_SEG_DATA_DIR = str(_CANONICAL_AXIS_DIR / "seg").replace("\\", "/")

# ── Step 5(HI-시나리오 상호작용 검정) ────────────────────────────────────
FIXED_INTERACTION_TAG = None            # 산출물 태그. None=ACTIVE_P1_TAG에서 자동 파생
FIXED_INTERACTION_MIN_SEGS_PER_CELL = 10  # 셀·시나리오별 최소 세그먼트 수(미만이면
                                         # 그 셀 전체를 통계 계산에서 제외)
FIXED_INTERACTION_MIN_EFFECT_SIZE = 0.2 # 시나리오 간 유의성 판정 문턱(Cohen's small
                                         # effect 관례, 0~1)
FIXED_INTERACTION_SHUFFLE_FROM = None   # v-ctrl 무작위 대조군 전용 참조 경로.
                                         # None=일반 통계검정 모드(정식 레시피)

# ── Step 6(HI 시너지 그룹 구성) ──────────────────────────────────────────
FIXED_MIN_PARTIAL_CORR = 0.02           # 그룹 구성 시 최소 편상관계수 문턱
FIXED_PREFILTER_TOP_M = 15              # 그룹 후보 사전 필터링 시 상위 M개
FIXED_SHUFFLE_SEED = FIXED_DEFAULT_SEED # v-ctrl 무작위 대조군 전용 시드
FIXED_SYNERGY_TAG = None                # 산출물 태그. None=ACTIVE_P1_TAG에서 자동 파생
FIXED_SYNERGY_SHUFFLE_FROM = None       # v-ctrl 무작위 대조군 전용 참조 경로.
                                         # None=일반 그리디 알고리즘 모드(정식 레시피)
FIXED_SYNERGY_PLOT_MIN_EDGE_CORR = 0.5  # 그룹 구조 시각화에 그릴 간선의 최소 |상관계수|

# ── Step 7(커널 HI 피처 생성) ────────────────────────────────────────────
FIXED_KERNEL_SYNERGY_GROUPS_JSON = None # synergy.py 산출물 경로. None=자동탐색
FIXED_KERNEL_ALPHA = 1.0                # Ridge 회귀 정규화 강도
FIXED_KERNEL_GAMMA = None               # RBF 커널 폭. None=sklearn 기본(1/n_features)
FIXED_KERNEL_N_COMPONENTS = 100         # Nystroem 근사 랜드마크 포인트 수
FIXED_KERNEL_MAX_FEATURES = None        # 시나리오당 최대 커널 HI 개수. None=무제한
FIXED_MIN_RAW_PARTIAL_CORR = 0.1        # 커널 HI가 "진짜 비선형 시너지"로 인정받기
                                         # 위한 최소 편상관계수 문턱(0~1)
FIXED_COMBINED_REDUNDANCY_THRESHOLD = 0.95  # raw+kernel 결합 다중공선성 판정
                                         # 상관계수 임계값(0~1)
FIXED_KERNEL_TAG = None                 # 산출물 태그. None=ACTIVE_P1_TAG에서 자동 파생

# ── Step 8(SCR Phase 1 학습) ─────────────────────────────────────────────
FIXED_TRAIN_CYCLE_FRAC = None           # 학습에 쓸 사이클 비율(0~1). None=1.0(전체)
FIXED_BETA_MIN = None                   # L0 게이트 온도(temperature) 최소값.
                                         # None=train.py 기본 0.1
FIXED_L0_WARMUP_EPOCHS_OVERRIDE = None  # L0 페널티 warmup 에폭 수 오버라이드.
                                         # None=yaml 값 사용
FIXED_L0_NORM_CONSTANT = None           # L0 페널티 정규화 상수. None=n_scenarios로 나눔
FIXED_VAL_RMSE_EPSILON = 0.00005        # 체크포인트 선택 시 val_rmse 개선 판정 최소폭
FIXED_DEVICE = None                     # 연산 장치. None=auto(cuda 우선) — Step 8/9 공용
FIXED_CHARGE_PROBE_M = None             # 충전 probe 게이트 top-m 오버라이드(수동
                                         # ablation 전용). None=cfg 기본값(10)
FIXED_DISCHARGE_PROBE_M = None          # 방전 probe 게이트 top-m 오버라이드.
                                         # None=cfg 기본값(10)
FIXED_SCEN_K_COUNT = None               # 시나리오별 회귀 게이트 top-k 오버라이드.
                                         # None=cfg 기본값(25)
# kernel_features_pkl/interaction_json을 둘 다 못 찾았을 때(Step 5~7을 건너뛴 경우)
# 쓰는 최종 폴백 — 리팩토링 이전 레거시 고정 산출물.
FIXED_LEGACY_V4_KERNEL_FEATURES_PKL = (
    "legacy_results/experiments/phase1_lab/results/"
    "kernel_group_features_k25_full_N2_kernel_v3.pkl"
)
FIXED_LEGACY_V4_INTERACTION_JSON = (
    "legacy_results/experiments/phase1_lab/results/"
    "hi_scenario_interaction_k25_full_N2.json"
)

# ── Step 9(평가) ──────────────────────────────────────────────────────────
FIXED_TEST_CHECKPOINT_OVERRIDE = None   # 평가에 쓸 체크포인트 경로 수동 오버라이드.
                                         # None=<run-dir>/checkpoints/best_by_saturation.pt

# ── 데이터 변형(Step 2 전용) ──────────────────────────────────────────────
FIXED_SKIP_SHAPE = False                # 형상 이상치 필터 비활성화 여부

# ── Step 4 진단 플롯(4_hi_analysis/plot.py 단독 실행 전용) ────────────────
# 파이프라인 스텝이 아니라 수동 진단 도구 — `python 4_hi_analysis/plot.py <번호>`로
# 실행, 번호만 CLI로 받고 나머지는 아래 값을 그대로 사용.
FIXED_STEP4_DIAG_DATASET = "MIT"        # 대상 데이터셋
FIXED_STEP4_DIAG_CELL = "b1c0"          # 대상 셀 ID
FIXED_STEP4_DIAG_CYCLE = 2              # 대상 사이클 번호
FIXED_STEP4_DIAG_WORKERS = 4            # 병렬 워커 수
FIXED_STEP4_DIAG_Z_THRESH = 6.0         # 이상치 판정 z-score 문턱
FIXED_STEP4_DIAG_DEV_THRESH = {"discharge": 0.25, "charge": 0.07}  # 편차 문턱(방향별)
FIXED_STEP4_DIAG_MAXDEV_THRESH = {"discharge": 0.40, "charge": 0.25}  # 최대편차 문턱(방향별)


# ═══════════════════════════════════════════════════════════════════════════
# Step 5~9(interaction.py/synergy.py/kernel.py/train.py) 공통 모델/학습 설정
# ═══════════════════════════════════════════════════════════════════════════
# scenario.axis/axis_config·data.data_dir/seg_data_dir은 위 FIXED_SEG_AXIS/
# ACTIVE_AXIS_CONFIG/FIXED_CANONICAL_DATA_DIR/SEG_DATA_DIR을 그대로 참조한다
# (값이 여러 곳에 중복되지 않도록).
P1_MODEL_CONFIG: dict = {
    "data": {
        "data_dir": FIXED_CANONICAL_DATA_DIR,
        "seg_data_dir": FIXED_CANONICAL_SEG_DATA_DIR,
        "is_cross_dataset_evaluate": False,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "min_cycles_per_cell": 10,       # 셀당 최소 사이클 수(미만이면 제외)
        "io_workers": 16,                # 데이터 로딩 병렬 워커 수
        "use_initial_capacity": True,    # 초기 용량 기준 SOH 정규화 여부
        "nominal_capacities": {"MIT": 1.1, "HUST": 1.2},  # 데이터셋별 공칭 용량(Ah)
        "datasets": FIXED_CANONICAL_DATASETS.copy(),
    },
    "classifier": {
        "charge_probe_m": 10,            # 충전 probe 게이트 top-m
        "discharge_probe_m": 10,         # 방전 probe 게이트 top-m
    },
    "model": {
        "d_probe": 64,                   # 분류기 은닉 차원
        "d_head": 128,                   # 회귀 헤드 은닉 차원
        "dropout": 0.2,
        "tr_n_heads": 4,                 # transformer 헤드 수(regression_model="transformer"류 전용)
        "tr_n_layers": 2,
        "tr_d_ff": 512,
        "resnet_n_blocks": 4,            # resnet_tab 전용
        "resnet_d_hidden_factor": 2.0,
        "regression_model": "mlp",       # 회귀 헤드 종류: mlp|transformer|i_transformer|resnet_tab|ft_transformer
        "mlp_hidden_dims": [128, 64],
    },
    "loss": {
        "lambda_scen": 0.01,             # 분류 CE 손실 가중치
        "lambda_l0": 0.01,               # L0 희소화 페널티 가중치(기본값 — 보통 ACTIVE_LAMBDA_L0_OVERRIDE로 덮어씀)
        "lambda_l0_auto": True,          # lambda_l0 자동 계산 여부
        "lambda_l0_schedule": "delayed_warmup",  # none|delayed_warmup|exp_ramp|cyclic
        "lambda_l0_warmup_epochs": 50,
        "lambda_l0_ramp_epochs": 100,
    },
    "training": {
        "epochs": 500,
        "batch_size": 2048,
        "warmup_epochs": 10,             # 학습률 warmup 에폭 수
        "grad_clip": 1.0,
        "lr": 2.0e-4,
        "weight_decay": 1.0e-3,
    },
    "evaluation": {
        "metrics": ["rmse", "mae", "r2", "mape"],
        "rep_cells_per_dataset": 5,      # 플롯에 쓸 데이터셋별 대표 셀 수
    },
    "scenario": {
        "axis": FIXED_SEG_AXIS,
        "axis_config": ACTIVE_AXIS_CONFIG.copy(),
    },
    "regression": {
        "scen_k_count": 25,              # 시나리오별 회귀 게이트 top-k
    },
}
