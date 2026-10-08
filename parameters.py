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

# 대용량 중간 산출물 드라이브 루트(data_directories.py의 _D_ROOT) — 아래
# FIXED_CANONICAL_DATA_DIR/SEG_DATA_DIR이 이 모듈만 거쳐서 참조하게 하면,
# 드라이브를 다시 옮길 때 data_directories.py 한 곳만 고치면 된다(그 모듈
# docstring의 원칙 그대로 — 전에는 이 파일에 "D:/chanminLee/..."가 별도로
# 하드코딩돼 있어서 두 군데를 따로 고쳐야 했다, 2026-10-02 정정).
from data_directories import DATA_4_HI_ROOT

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
FIXED_CANONICAL_DATASETS = ["MIT", "HUST"]  # 구 interaction.py/synergy.py/kernel.py
                                         # --datasets 하드코딩(2026-09-25 통합) +
                                         # P1_MODEL_CONFIG["data"]["datasets"] 단일 소스.
                                         # FIXED_DATASET_GROUP="lfp"가 뜻하는 것과 같은
                                         # 조합(MIT+HUST)이지만 값 형식(그룹 이름 문자열 vs
                                         # 실제 데이터셋 리스트)이 달라 Step 4는 통합 대상이
                                         # 아님 — Step 5~9(정식 q_frac_ref 축)에서만 쓰는
                                         # "실제 로드할 데이터셋 리스트".

# ── 커널/다중공선성 파이프라인 설정(Step 6~8 산출물, Step 9가 소비) ───────
ACTIVE_KERNEL_FEATURES_PKL = None       # 구 --kernel-features-pkl (None=자동 경로/fallback)
ACTIVE_INTERACTION_JSON = None          # 구 --interaction-json (None=자동 경로/fallback)
ACTIVE_COMBINED_REDUNDANCY_JSON = None  # 구 --combined-redundancy-json (None=자동 경로)
ACTIVE_MAX_GROUP_SIZE = 4               # 구 --max-group-size (시너지 그룹 크기 상한)
ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD = 0.9  # 구 --synergy-redundancy-threshold — synergy.py
                                         # 1차 배제(그룹 내부)와 kernel.py 2차 배제(커널HI끼리,
                                         # 구 FIXED_KERNEL_REDUNDANCY_THRESHOLD)가 "동일 임계값
                                         # 재사용"이라고 서로의 독스트링에 명시하고 있었고 실제
                                         # RESULTS_LOG.md 36회 호출 전부 0.9로 일치해 2026-09-25
                                         # 하나로 통합 — kernel.py도 이제 이 상수를 그대로 쓴다.

# ── 학습(Step 8) ─────────────────────────────────────────────────────────
# (2026-10-02: 절 제목의 "Step 9"는 stale 주석이었다 — train.py는 Step 8이다.)
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
ACTIVE_P1_TAG = "refact"                # 구 --p1-tag
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
ACTIVE_AXIS_CONFIG: dict = {            # 이 값이 곧 train.py의 하드코딩 fallback
    "n1": 0.35,                         # data_dir(n1-35%_n2-20%_N-2_minpts5_lag-1_
    "n2": 0.20,                         # noise-3%_ou-200_calib-100_offA-5mA)을 만든
    "n_samples": 2,                     # 실제 축 설정이다.
    "ref_lag": 1,                       # ⚠️ "scen_lag1zone"류 실험 변형(tile_scope=zone, min_pts/
    "noise_amp": 0.03,                  # calibration/offset 없음)과 혼동 금지 — 그건 별도 실험이고
    "noise_mode": "ou",                 # 이게 정식(canonical) 설정이다(2026-09-23 정정).
    "noise_period_cycles": 200.0,
    "min_pts": 5,
    "calibration_period": 100,
    "offset_amp": 0.005,
}

# ── 모델/재현성 ──────────────────────────────────────────────────────────
FIXED_DEFAULT_SEED = 42                 # 2026-09-25 통합 — ACTIVE_SEED/ACTIVE_SPLIT_SEED가
                                         # None일 때의 폴백 값과 FIXED_SHUFFLE_SEED(아래)가
                                         # 전부 여기저기 흩어진 리터럴 42였던 걸 하나로 모음.
                                         # ⚠️ 이 세 시드는 서로 다른 무작위성을 제어한다(모델
                                         # 초기화 RNG / 셀 분할 / v-ctrl 대조군 재배정) —
                                         # 값이 같은 건 관례일 뿐, 멀티시드 실험에서는 여전히
                                         # --seed/--split-seed/--shuffle-seed를 각자 다른
                                         # 값으로 독립적으로 줄 수 있다(이 상수는 "아무것도 "
                                         # "안 줬을 때"의 공통 기본값일 뿐).
ACTIVE_SEED = None                      # 구 --seed (None→FIXED_DEFAULT_SEED로 해석)
ACTIVE_SPLIT_SEED = None                # 구 --split-seed (None→FIXED_DEFAULT_SEED로 해석)


# ═══════════════════════════════════════════════════════════════════════════
# 거의 안 바꾸는 파라미터 — CLI 노출 없음, 이 값 그대로 하위 스크립트에 전달됨
# ═══════════════════════════════════════════════════════════════════════════

# ── 세그멘테이션 축 종류(q_frac_ref 고정 — 이번 세션 전부 이 축만 사용) ────
FIXED_SEG_AXIS = "q_frac_ref"           # 구 --seg-axis. vwindow 등 다른 축을 쓰려면
                                         # 이 상수를 바꿀 것(--axis-config만으로는 축
                                         # 종류 자체를 못 바꿈).

# ── ACTIVE_AXIS_CONFIG로 Step4가 실제 추출한 데이터의 저장 경로 ────────────
# train.py 등이 --data-dir/--seg-data-dir이 없을 때 쓰는 최종 폴백(2026-09-23 —
# 예전엔 이 값이 train.py 안에 별도로 하드코딩돼 있다가 ACTIVE_AXIS_CONFIG와 조용히
# 어긋난 적이 있었다). ACTIVE_AXIS_CONFIG를 바꾸면 Step4가 만드는 실제 경로도
# 바뀌므로 이 상수도 반드시 같이 갱신할 것.
_CANONICAL_AXIS_DIR = (
    DATA_4_HI_ROOT / "q_frac_ref"
    / "n1-35%_n2-20%_N-2_minpts5_lag-1_noise-3%_ou-200_calib-100_offA-5mA"
)
FIXED_CANONICAL_DATA_DIR = str(_CANONICAL_AXIS_DIR / "cycle").replace("\\", "/")
FIXED_CANONICAL_SEG_DATA_DIR = str(_CANONICAL_AXIS_DIR / "seg").replace("\\", "/")

# ── Step 5(HI-시나리오 상호작용 검정) ────────────────────────────────────
FIXED_INTERACTION_TAG = None            # 구 --interaction-tag (None=--p1-tag에서 자동 파생)
FIXED_INTERACTION_MIN_SEGS_PER_CELL = 10 # 셀 단위 std_r/시나리오별 평균 계산 시
                                         # "이 셀, 이 시나리오" 세그먼트 수가 이 값
                                         # 미만이면 그 셀 전체를 제외(추정이 불안정해짐
                                         # 방지).
FIXED_INTERACTION_MIN_EFFECT_SIZE = 0.2 # 구 --interaction-min-effect-size — Cohen(1988)
                                         # 상관계수 효과크기 관행 중 "작음(small)" 문턱.
                                         # 2026-09-30엔 전체 풀링 std_r_across_scenarios에
                                         # 비교해서 근거가 약했다(풀링이 셀 간 베이스라인
                                         # 차이 때문에 값을 체계적으로 깎아, 66개 HI 전부
                                         # 풀링값 < 셀 단위 평균, 평균 2.6배 — 즉 "0.1이
                                         # 근거 없다"가 아니라 "0.1을 잘못된 깎인 값에
                                         # 비교했다"는 게 진짜 문제였음). 2026-10-01:
                                         # cell_level_std_r_mean(편향 없는 셀 단위 평균)에
                                         # 그대로 재적용 — 같은 관행값이지만 이제 올바른
                                         # 통계치와 비교한다. 중간에 "순열(permutation)
                                         # 귀무기준선"으로 대체를 시도했으나(셀의 세그먼트를
                                         # 섞어 가짜 시나리오를 만든 std_r과 비교), 셀 수가
                                         # 120개로 충분히 많아 거의 모든 HI가 통과해버려
                                         # (최소 7.7배) "우연이 아니다"는 답은 됐지만
                                         # "크기가 충분하다"는 답이 못 돼 폐기했다 —
                                         # docs/REFACTORING.md 참고.
FIXED_INTERACTION_SHUFFLE_FROM = None   # 구 --shuffle-from (v4-ctrl 전용 — 이 경로의
                                         # hi_scenario_interaction_*.json을 참조해 공유/
                                         # 특이 개수는 유지하고 배정만 무작위 재배정,
                                         # None=일반 통계검정 모드. 2026-09-30 CLI 제거와
                                         # 함께 이전, 정식 레시피는 미사용)

# ── Step 7(HI 시너지 그룹 구성) ──────────────────────────────────────────
FIXED_MIN_PARTIAL_CORR = 0.02           # 구 --min-partial-corr
FIXED_PREFILTER_TOP_M = 15              # 구 --prefilter-top-m
# FIXED_GLOBAL_DEDUP(구 --global-dedup)은 2026-10-02에 제거했다 — 예전엔 False가
# 기본값이라 raw HI 사전 가지치기(_prune_redundant_raw)가 꺼져 있었는데(docs상
# "v4 정식 레시피"는 True를 쓴다고 돼 있었지만 이 세션에 실제로 돌린 run은 전부
# False로 실행됐었음, 2026-09-21 그 실측을 기본값으로 고정했던 이력), 이번에
# 그 괴리를 "문서가 맞다"는 방향으로 해소하고 사전 가지치기를 상시 적용으로
# synergy.py에 고정했다(6_synergy/synergy.py 모듈 docstring 참고) — 더 이상
# 끄고 켤 수 있는 옵션이 아니므로 파라미터 자체를 삭제.
FIXED_SHUFFLE_SEED = FIXED_DEFAULT_SEED # 구 --shuffle-seed (v-ctrl 무작위 대조군 전용 —
                                         # synergy.py/interaction.py 공용,
                                         # 2026-09-24 하드코딩 기본값에서 이전, 2026-09-25
                                         # FIXED_DEFAULT_SEED 참조로 통합)
FIXED_SYNERGY_TAG = None                # 구 --synergy-tag (None=--p1-tag에서 자동 파생,
                                         # run_pipeline.py 기준 파생 규칙은 f"{p1_tag}_groups")
FIXED_SYNERGY_SHUFFLE_FROM = None       # 2026-10-01 신설, 구 synergy.py --shuffle-from
                                         # (v-ctrl 전용 — 이 경로의 synergy_groups_*.json이
                                         # 가진 시나리오별 그룹 크기 분포는 그대로 두고 멤버만
                                         # 무작위 재배정. interaction.py의
                                         # FIXED_INTERACTION_SHUFFLE_FROM과 짝이지만 서로
                                         # 다른 스텝의 v-ctrl 기능이라 상수를 공유하지 않음.
                                         # None=일반 그리디 알고리즘(build_groups) 모드.
                                         # 정식 레시피 미사용.)
FIXED_SYNERGY_PLOT_MIN_EDGE_CORR = 0.5   # 2026-10-01 신설, 6_synergy/plot.py 단독 —
                                         # 구 plot_cluster_structure.py --min-edge-corr.
                                         # 그룹 구조 네트워크 그림에 그릴 간선의 최소
                                         # |raw corr| — 순수 시각화 문턱이라 build_groups
                                         # 알고리즘(위 redundancy_threshold)과는 무관.

# ── Step 7(커널 HI 피처 생성) ────────────────────────────────────────────
FIXED_KERNEL_SYNERGY_GROUPS_JSON = None # 구 --kernel-synergy-groups-json (None=Step 7 자동 경로)
FIXED_KERNEL_ALPHA = 1.0                # 구 --kernel-alpha (Ridge 정규화 강도)
FIXED_KERNEL_GAMMA = None               # 구 --kernel-gamma (None=sklearn 기본 1/n_features)
FIXED_KERNEL_N_COMPONENTS = 100         # 구 --kernel-n-components (Nystroem 랜드마크 수)
# 2026-10-02: 구 "2차 배제(커널끼리 pooled)" 단계 자체가 삭제됐다(synergy.py의
# global_dedup 상시화 + 커널 측 own-scenario zero-masking으로 무의미해짐 —
# docs/REFACTORING.md 2026-10-02 "kernel.py 2차 다중공선성 배제 단계 삭제" 참고).
# 그 단계가 쓰던 ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD 재사용도 함께 없어짐.
FIXED_KERNEL_MAX_FEATURES = None        # 구 --kernel-max-features (None=무제한)
FIXED_MIN_RAW_PARTIAL_CORR = 0.1        # 구 --min-raw-partial-corr — 2026-10-04부로 활성화
    # (이전 기본값 None=비활성). 커널 예측값이 "자기 그룹 raw 멤버로 이미 선형
    # 설명되는 부분"을 빼고도 SOH와 남는 관계가 있는지(raw-conditioned partial
    # corr) 검사 — 이게 없으면 train_r2가 높아도 그게 진짜 "비선형 시너지"인지
    # "raw 멤버 선형결합을 재탕한 것"인지 구분이 안 된다. 과거 유일하게 실측 쓰인
    # 값은 0.02였는데(synergy.py의 min_partial_corr에서 그냥 복붙된 값, RESULTS_LOG.md
    # 확인 결과 6개 run 전부 "후보 N개 -> 최종 N개"로 단 한 건도 안 걸러진 사실상
    # no-op 문턱이었음) — 이번에 interaction.py의 FIXED_INTERACTION_MIN_EFFECT_SIZE
    # (0.1, Cohen's small-effect 관례)와 맞춰 실제로 의미 있게 작동할 값으로 교체.
FIXED_COMBINED_REDUNDANCY_THRESHOLD = 0.95  # 구 --combined-redundancy-threshold (3차 배제, 시나리오별)
FIXED_KERNEL_TAG = None                 # 구 --kernel-tag (None=--p1-tag에서 자동 파생)

# ── Step 8(SCR Phase 1 학습) ─────────────────────────────────────────────
# (2026-10-02: 절 제목의 "Step 9"는 stale 주석이었다 — train.py는 Step 8이다.
# kernel.py 쪽 "Step 8(커널 HI 피처 생성)"→"Step 7" 오타 수정과 동일한 종류의 실수.)
FIXED_TRAIN_CYCLE_FRAC = None           # 구 --train-cycle-frac (None=1.0, 전체 사용)
FIXED_BETA_MIN = None                   # 구 --beta-min (None=train.py 기본 0.1)
FIXED_L0_WARMUP_EPOCHS_OVERRIDE = None  # 구 --l0-warmup-epochs-override (None=yaml 값)
FIXED_L0_NORM_CONSTANT = None           # 구 --l0-norm-constant (None=n_scenarios로 나눔)
FIXED_VAL_RMSE_EPSILON = 0.00005         # 구 --val-rmse-epsilon (체크포인트 선택 기준, 2026-09-18)
FIXED_DEVICE = None                     # 구 --device (None=auto) — Step 8/9 공용
FIXED_CHARGE_PROBE_M = None             # 2026-10-02 신설, 구 train.py --charge-m —
                                         # run_pipeline.py가 넘긴 적 없는 수동 ablation
                                         # 전용 오버라이드(None=cfg의 classifier.charge_probe_m
                                         # 그대로, 기본 10)
FIXED_DISCHARGE_PROBE_M = None          # 구 --discharge-m (위와 동일, discharge_probe_m)
FIXED_SCEN_K_COUNT = None               # 구 --scen-k (위와 동일, regression.scen_k_count)
# train.py가 --kernel-features-pkl/--interaction-json 둘 다 없을 때(= Step 5~7을 이번
# run_pipeline.py 실행 범위에서 뺐을 때) 쓰는 최종 fallback — v4 정식 학습 레시피
# (docs/260827_RESULTS.md "v4 정식 학습 결과" 절) 고정 산출물. 2026-10-02: 원래
# run_pipeline.py에 P1V4_KERNEL_FEATURES_PKL/P1V4_INTERACTION_JSON으로 있던 걸
# train.py도 똑같이 알아야 해서(CLI 제거로 이 해석 로직 자체가 train.py 안으로
# 들어감) parameters.py로 옮겨 단일 소스화 — run_pipeline.py도 이제 이 상수를 참조.
FIXED_LEGACY_V4_KERNEL_FEATURES_PKL = (
    "legacy_results/experiments/phase1_lab/results/"
    "kernel_group_features_k25_full_N2_kernel_v3.pkl"
)
FIXED_LEGACY_V4_INTERACTION_JSON = (
    "legacy_results/experiments/phase1_lab/results/"
    "hi_scenario_interaction_k25_full_N2.json"
)

# ── Step 9(평가) ──────────────────────────────────────────────────────────
FIXED_TEST_CHECKPOINT_OVERRIDE = None   # 2026-10-02 신설, 구 test.py --checkpoint —
                                         # run_pipeline.py가 넘긴 적 없는 수동 오버라이드
                                         # (None=<run-dir>/checkpoints/best_by_saturation.pt)

# ── 데이터 변형(Step 2 전용, 이번 세션 미사용) ───────────────────────────
# 2026-09-29: FIXED_EXCLUDE_CV(구 hi_correlation.py --exclude-cv, Step4 CC-only
# 옵션)는 Step4가 그 옵션을 완전히 제거하면서 아무 데서도 안 쓰여 함께 삭제.
FIXED_SKIP_SHAPE = False                # 구 --skip-shape (형상 이상치 필터 비활성화,
                                         # 2_preprocess/preprocess.py 전용 — Step4는
                                         # 2026-09-29부로 이 옵션을 안 읽음)

# ── Step4 진단 플롯(4_hi_analysis/plot.py 단독 실행 전용, 2026-10-01 신설) ──
# run_pipeline.py 파이프라인 스텝이 아니라 수동 진단 도구라 run_pipeline.py가
# 안 읽는다 — tools/plot_cell_cycles.py 등 옛 argparse CLI 기본값을 그대로 옮김
# (동작 변화 없음). `python 4_hi_analysis/plot.py <번호>`로 실행, 번호만 CLI.
FIXED_STEP4_DIAG_DATASET = "MIT"        # 구 --dataset
FIXED_STEP4_DIAG_CELL = "b1c0"          # 구 --cell
FIXED_STEP4_DIAG_CYCLE = 2              # 구 --cycle
FIXED_STEP4_DIAG_WORKERS = 4            # 구 plot_all_mit_cells.py --workers
FIXED_STEP4_DIAG_Z_THRESH = 6.0         # 구 --z-thresh
FIXED_STEP4_DIAG_DEV_THRESH = {"discharge": 0.25, "charge": 0.07}
                                         # 구 --dev-thresh-discharge/-charge
FIXED_STEP4_DIAG_MAXDEV_THRESH = {"discharge": 0.40, "charge": 0.25}
                                         # 구 --maxdev-thresh-discharge/-charge


# ═══════════════════════════════════════════════════════════════════════════
# Step 5~9(interaction.py/synergy.py/kernel.py/train.py) 공통 모델/학습 설정
# ═══════════════════════════════════════════════════════════════════════════
# 유일한 소스 — 2026-09-27부로 yaml 프리셋(model_lib/config/*.yaml, --model-config
# 플래그)을 전부 폐기하고 이 딕셔너리 하나로 통일했다. main_qfref_S_all4/hust_only/
# noscen/p60/transformerL 등 과거 프리셋들은 이번 세션의 폴더 재편·train.py 개명
# 이전 코드를 전제로 한 채 한 번도 재검증되지 않았고, 그중 하나(--regression-model)는
# 이미 run_pipeline.py -> train.py 전달 경로가 끊겨 있었다(train.py가 그 플래그
# 자체를 안 받은 지 오래) — 재현이 필요해지면 그때 다시 설계할 것
# (docs/REFACTORING.md 2026-09-27 항목 참고).
#
# scenario.axis/axis_config는 별도로 안 두고 FIXED_SEG_AXIS/ACTIVE_AXIS_CONFIG를
# 그대로 참조한다 — 축 설정이 여기 또 복사되면 두 값이 다시 어긋날 수 있기 때문
# (예전에 실제로 있었던 문제). data.data_dir/seg_data_dir도 마찬가지로
# FIXED_CANONICAL_DATA_DIR/SEG_DATA_DIR을 참조한다.
P1_MODEL_CONFIG: dict = {
    "data": {
        "data_dir": FIXED_CANONICAL_DATA_DIR,
        "seg_data_dir": FIXED_CANONICAL_SEG_DATA_DIR,
        "is_cross_dataset_evaluate": False,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "min_cycles_per_cell": 10,
        "io_workers": 16,
        "use_initial_capacity": True,
        "nominal_capacities": {"MIT": 1.1, "HUST": 1.2},
        "datasets": FIXED_CANONICAL_DATASETS.copy(),
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
        # 2026-09-25: with_raw_cnn/raw_cnn_pretrained_from 삭제 — 의존하는
        # models/raw_cnn.py 자체가 repo에 없었고(있었어도 train.py/test.py가 v4에서
        # 항상 강제 False), scr_model.py의 RawCNN 로딩 분기도 같이 제거했다.
        # 2026-09-27: with_raw_flat도 완전히 죽어있어(train.py/test.py가 항상 강제
        # False) scr_model.py/cap_heads.py의 스캐폴딩과 함께 이 키도 삭제.
    },
    "loss": {
        "lambda_scen": 0.01,
        "lambda_l0": 0.01,
        "lambda_l0_auto": True,
        "lambda_l0_schedule": "delayed_warmup",
        "lambda_l0_warmup_epochs": 50,
        "lambda_l0_ramp_epochs": 100,
        # 2026-09-27: leak_cols 삭제 — train.py/scr_loss.py 어디서도 안 읽는다. 실제
        # leakage 제외는 SOH_EXCLUDE_STAT_LEAK 환경변수(ACTIVE_N_HI 토글, hi_schema.py)
        # 하나로만 이뤄진다 — 이 키는 그 이전 방식의 흔적이었다.
    },
    "training": {
        # 2026-09-25: scheduler/log_interval/run_overfit_test/overfit_test_samples/
        # overfit_test_epochs/early_stop_patience 삭제 — 전부 구 SCRTrainer.fit()
        # (model_lib/training/scr_trainer.py, 삭제됨) 전용 키였고 train.py 자체
        # 학습 루프는 어디서도 읽지 않았다(scheduler는 실제로 항상
        # CosineAnnealingLR 하나만 씀 — 값 분기 자체가 없었음).
        "epochs": 500,
        "batch_size": 2048,
        "warmup_epochs": 10,
        "grad_clip": 1.0,
        "lr": 2.0e-4,
        "weight_decay": 1.0e-3,
    },
    "evaluation": {
        "metrics": ["rmse", "mae", "r2", "mape"],
        "rep_cells_per_dataset": 5,
    },
    # 2026-09-25: "uq" 섹션(Laplace UQ) 삭제 — train.py가 UQ를 fit한 적이 없고
    # (유일한 경로였던 SCRTrainer.fit_laplace()가 삭제됨) test.py도 UQ 예측/
    # 캘리브레이션을 소비하지 않는다(model_lib/utils/uncertainty.py 전체 삭제,
    # scr_evaluator.py의 predict_dataset_uq/save_uq_metrics/plot_uq도 같이 삭제).
    "scenario": {
        "axis": FIXED_SEG_AXIS,
        "axis_config": ACTIVE_AXIS_CONFIG.copy(),
    },
    "regression": {
        "scen_k_count": 25,
    },
}
