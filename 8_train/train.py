"""
5_model/experiments/phase1_lab/phase1_trainer_v2.py

Stage1(체크포인트 선택 기준 변경) + Stage2(temperature annealing)를 적용한
독립 Phase1 트레이너.

기존 5_model/train_scr.py, 5_model/training/scr_trainer.py,
5_model/models/hard_concrete.py는 단 한 줄도 수정하지 않는다 — 전부 그대로
IMPORT해서 재사용(SCRModel/SCRLoss/build_datasets/L0LambdaScheduler/게이트 JSON
저장 함수)하고, "에폭을 몇 번 돌고 어느 시점을 최종본으로 저장할지"를 결정하는
학습 루프만 이 파일이 독자적으로 새로 짠다.

기존 SCRTrainer.fit() 대비 차이점 (docs/PHASE1_REDESIGN.md §3 Stage1/2):
  1. 체크포인트 선택 기준: val RMSE 최소가 아니라 "게이트 포화도"
     (gate_prob이 애매한 [0.1, 0.9] 구간에 남아있는 비율)가 가장 낮은 시점을
     저장한다. 단, L0 램프(warmup+ramp)가 완전히 끝난 이후 에폭만 후보로 삼는다
     — 이래야 "페널티가 실제로 적용된 상태"에서 고른 게 보장된다(기존 버그의
     직접적인 원인이었던 "페널티 걸리기 전에 저장" 문제를 구조적으로 차단).
  2. Temperature annealing: HardConcreteGate.BETA는 클래스 상수지만
     self.BETA로 접근하므로 인스턴스별 오버라이드가 가능하다(hard_concrete.py
     무변경). L0 램프 구간과 맞물려 2/3 -> --beta-min으로 선형으로 낮춰
     게이트를 더 확실하게 0/1로 밀어붙인다.

출력 레이아웃은 기존 Phase1 run과 100% 동일하게 맞춘다(gates/classification_HIs.json,
gates/regression_HIs.json, scenario_spec.json, logs/train_log_v2.csv) — 그래야
analyze_convergence.py / materialize_ensemble_gates.py를 무수정으로 재사용할 수 있다.

사용 예(--seg-axis/--axis-config는 표준 조합(q_frac_ref, n1=0.35/n2=0.20/n_samples=2)이면
생략 가능 — 다른 조합이면 직접 지정, 이때 --data-dir/--seg-data-dir도 같이 줘야 함):
  python 5_model/experiments/phase1_lab/phase1_trainer_v2.py \
      --model-config 5_model/config/main_qfref_S.yaml \
      --scen-k 25 --seed 42 --split-seed 42 --beta-min 0.1 --tag stage12_k25
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 콘솔이 cp949일 때 em-dash 등 특수문자 print가 UnicodeEncodeError로 죽는 문제
# 방지(lambda_sweep.py/plot_lambda_sweep.py와 동일 패턴) — 짧은 스모크런처럼 체크포인트가
# 한 번도 안 뽑히는 예외 경로의 경고 메시지에서 실제로 이 문제로 죽는 걸 확인해서 추가.
for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import numpy as np
import torch

from utils.io_utils import load_config, save_config  # noqa: E402
from utils.hi_schema import N_HI, get_hi_cols_for_seg, EXCLUDE_STAT_LEAK, EXCLUDE_DQDV_LEAK  # noqa: E402
from utils.metrics import rmse as _rmse, r2 as _r2  # noqa: E402
from utils.tqdm_utils import trange, write as tqdm_write  # noqa: E402
from datasets.segment_dataset import build_datasets, FastTensorLoader  # noqa: E402
from models.scr_model import SCRModel  # noqa: E402
from training.scr_loss import SCRLoss  # noqa: E402
from training.scr_trainer import L0LambdaScheduler  # noqa: E402
from common.scenario import get_segmenter  # noqa: E402

# train_scr.py는 스크립트지만 __main__ 가드가 있어 import해도 안전 — JSON 저장
# 함수만 재사용(중복 구현 금지 원칙, docs/PHASE1_REDESIGN.md 참고).
import train_scr as _base  # noqa: E402

# seg-axis/axis-config/data-dir/seg-data-dir 전부 이번 v3/v4/v-ctrl 검증 전체에서 한 번도
# 안 바뀐 고정 조합(q_frac_ref, n1=0.35/n2=0.20/n_samples=2) — build_synergy_groups.py 등
# 나머지 phase1_lab 스크립트와 동일한 기본값을 준다. 이 스크립트는 train_scr.py의 자동
# 경로계산(_axis_dir)을 재사용하지 않아서 --data-dir/--seg-data-dir을 안 주면(yaml도
# null) 즉시 RuntimeError였다 — lambda_sweep.py가 이미 겪은 문제와 동일(그쪽 주석 참고).
# 네 값은 항상 세트로 움직이므로, 표준 조합이 아니면 넷 다 함께 오버라이드해야 한다.
from data_directories import DATA_4_HI_ROOT_STR  # noqa: E402

DEFAULT_SEG_AXIS = "q_frac_ref"
DEFAULT_AXIS_CONFIG = json.dumps({
    "n1": 0.35, "n2": 0.20, "ref_lag": 0, "noise_amp": 0.03,
    "noise_mode": "ou", "noise_period_cycles": 200, "n_samples": 2,
})
_DATA_ROOT = f"{DATA_4_HI_ROOT_STR}/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200"
DEFAULT_DATA_DIR = f"{_DATA_ROOT}/cycle"
DEFAULT_SEG_DATA_DIR = f"{_DATA_ROOT}/seg"

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "p1v2_runs"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase1 v2 — 체크포인트 기준 변경 + temperature annealing")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", default=None,
                   help="미지정 시 --model-config yaml의 scenario.axis를 쓰고, "
                        "그마저 없으면 DEFAULT_SEG_AXIS(q_frac_ref)로 폴백")
    p.add_argument("--axis-config", default=None,
                   help="미지정 시 --model-config yaml의 scenario.axis_config를 쓰고, "
                        "그마저 없으면 DEFAULT_AXIS_CONFIG로 폴백 — 예전엔 CLI 기본값이 "
                        "항상 DEFAULT_AXIS_CONFIG라 yaml에 뭘 적어놔도 조용히 무시됐다 "
                        "(2026-09-04 수정 — yaml만으로 축 설정을 완결시킬 수 있게 함).")
    p.add_argument("--charge-m", type=int, default=None)
    p.add_argument("--discharge-m", type=int, default=None)
    p.add_argument("--scen-k", type=int, default=None)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--split-seed", type=int, required=True)
    p.add_argument("--train-cycle-frac", type=float, default=1.0,
                   help="진단용 — train split만 cell당 cycle 단위로 이 비율만큼 "
                        "서브샘플링(zone 다양성은 유지, val/test는 그대로). "
                        "docs/260909_RESULTS.md §6-5(e) 진단 실험. 기본 1.0=미적용.")
    p.add_argument("--data-dir", default=None,
                   help="cycle pkl 경로. 미지정 시 yaml의 data.data_dir, 그마저 없으면 "
                        "DEFAULT_DATA_DIR(캐논 경로)로 폴백 — 예전엔 CLI 기본값이 항상 "
                        "DEFAULT_DATA_DIR라 yaml 값이 있어도 조용히 무시됐다(2026-09-04 수정).")
    p.add_argument("--seg-data-dir", default=None,
                   help="seg pkl 경로. 미지정 시 yaml의 data.seg_data_dir, 그마저 없으면 "
                        "DEFAULT_SEG_DATA_DIR로 폴백(위와 동일 수정).")
    p.add_argument("--beta-min", type=float, default=0.1, help="annealing 종착 BETA(기본 0.1, 원래값 2/3)")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="cfg의 training.epochs(기본 500) 대신 쓸 상한. 시간 단축용 — "
                        "--patience 조기종료가 그 전에 걸리면 이 값까지 안 감")
    p.add_argument("--patience", type=int, default=60,
                   help="L0 완전 램프 이후, 게이트 포화도가 이 에폭 수만큼 연속 개선 없으면 "
                        "조기 종료(기본 60 — main_qfref_S_p60.yaml과 동일 근거). "
                        "베스트 체크포인트 자체는 patience 길이와 무관하게 항상 보존되므로 "
                        "결과에는 영향 없고 학습 시간만 줄어듦. 0이면 조기종료 비활성(항상 --max-epochs까지)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="cfg의 training.batch_size 오버라이드. 배치가 클수록 에폭당 배치 수가 "
                        "줄어 Stage A/B 게이트 루프(방향×시나리오별 서브포워드)의 배치당 "
                        "오버헤드 총합이 줄어든다 — 값 자체이 학습 결과를 바꿀 수 있으니 "
                        "(선형 학습이 아니라 완전 무해하지는 않음) 처음 켤 때는 baseline과 "
                        "1개 seed로 비교 검증 권장")
    p.add_argument("--regression-model", default="mlp",
                   choices=["mlp", "transformer", "i_transformer", "resnet_tab", "ft_transformer"],
                   help="Phase1 cap_head 종류. 원래 train_scr.py는 이걸 항상 'mlp'로 강제한다"
                        "(design intent: 게이트 선택이 최종 헤드 아키텍처와 무관해야 함) — "
                        "이 오버라이드는 그 전제가 실제로 맞는지 검증하기 위한 sanity check 전용. "
                        "기본값(mlp)이면 기존 train_scr.py Phase1과 동일하게 동작.")
    p.add_argument("--synergy-groups-json", default=None, dest="synergy_groups_json",
                   help="build_synergy_groups.py 산출물(synergy_groups_*.json) 경로. 주어지면 "
                        "scen_gates가 시나리오별 그룹 계층 게이트(GroupedHardConcreteGate)로 "
                        "학습됨 — train_scr.py의 --synergy-groups-json과 동일 로더 재사용. "
                        "--kernel-features-pkl과 동시 사용 불가(그룹이 이미 피처 레벨에서 "
                        "융합되면 게이트 레벨 그룹핑은 의미가 겹침)")
    p.add_argument("--kernel-features-pkl", default=None, dest="kernel_features_pkl",
                   help="build_kernel_group_features.py 산출물(kernel_group_features_*.pkl) 경로. "
                        "주어지면 raw HI(x_hi)는 그대로 두고, 그룹당 1개 RBF 커널 융합값을 "
                        "별도 게이트(scen_kernel_gates)로 추가해서 학습(대체 아님, 추가) — "
                        "--synergy-groups-json과 동시 사용 불가. 2026-09-18부터 커널 HI 컬럼은 "
                        "own-scenario 행에만 값이 채워짐(_apply_kernel_features 참고, v4 요구사항1)")
    p.add_argument("--combined-redundancy-json", default=None, dest="combined_redundancy_json",
                   help="build_kernel_group_features.py의 3차(결합 raw+kernel, 시나리오별) 다중공선성 "
                        "배제 산출물(kernel_group_features_*_combined_redundancy.json) 경로(v4 "
                        "요구사항2). 주어지면 시나리오별로 raw HI(nan_mask 0-강제)와 커널 HI(항상 0) "
                        "중 그 시나리오에서 |r|>=0.95인 쌍에 걸린 HI를 전부 배제한다. "
                        "--kernel-features-pkl 없이는 줄 수 없음(커널 쪽 산출물이 이 파일을 만드는 "
                        "선행 단계라 raw 전용으로만 쓰는 경우는 아직 미지원).")
    p.add_argument("--tag", required=True)
    p.add_argument("--output-dir", default=None, dest="output_dir",
                   help="run 산출물(gates/checkpoints/logs/config.yaml 등)을 저장할 폴더를 "
                        "직접 지정 — 주어지면 이 경로를 그대로 쓰고, RESULTS_DIR 밑에 "
                        "{timestamp}_p1v2_{tag}_seed{seed} 폴더를 새로 만들지 않는다. "
                        "run_pipeline.py가 Step 6~8(상호작용/시너지/커널) 산출물과 같은 "
                        "폴더에 학습 결과를 모으려고 씀(2026-09-19) — 단독 실행 시 기본"
                        "(미지정)이면 기존과 100% 동일하게 타임스탬프 폴더를 새로 만든다.")
    p.add_argument("--lambda-l0-override", type=float, default=None, dest="lambda_l0_override",
                   help="loss.lambda_l0을 이 값으로 강제 고정(lambda_l0_auto/yaml 값 무시, 최우선순위). "
                        "lambda_l0 정규화 경로 스윕 실험(lambda_sweep.py) 전용 — 평소 실행에서는 "
                        "주지 않으면 기존 동작(auto 또는 yaml 값)과 100%% 동일.")
    p.add_argument("--interaction-json", default=None, dest="interaction_json",
                   help="test_hi_scenario_interaction.py 산출물 경로(v4). significant=True인 "
                        "HI만 기존 scen_gates(시나리오별)로 남기고, 나머지는 새 shared_gate "
                        "1개로 통합한다. --synergy-groups-json(v0의 64폭 그룹)과는 동시 사용 불가.")
    p.add_argument("--l0-warmup-epochs-override", type=int, default=None,
                   dest="l0_warmup_epochs_override",
                   help="loss.lambda_l0_warmup_epochs을 이 값으로 강제 고정(yaml 값 무시). "
                        "warmup epoch 스위핑 실험(docs/260825_RESULTS.md) 전용 — 평소 실행에서는 "
                        "주지 않으면 기존 동작(yaml 값, 기본 50)과 100%% 동일. training.warmup_epochs"
                        "(learning rate warmup, 별개 값)에는 영향 없음.")
    p.add_argument("--l0-norm-constant", type=int, default=None, dest="l0_norm_constant",
                   help="docs/260915_RESULTS.md — scr_loss.py의 _l0_penalty가 시나리오별 "
                        "페널티 합을 n_scenarios로 나누는 것을, n_scenarios 대신 이 고정값으로 "
                        "나누게 바꿈. 라벨ON(n_scenarios=6)/라벨OFF(n_scenarios=2) 조건 간 "
                        "정규화 강도 confound를 분리하는 대조군 전용 — 예: rawonly(라벨ON, "
                        "원래 6으로 나눔)에 --l0-norm-constant 2를 주면 noscen(라벨OFF, "
                        "원래 2로 나눔)과 동일한 정규화 강도로 맞출 수 있음. 미지정(기본)이면 "
                        "기존과 100%% 동일 동작.")
    p.add_argument("--hi-cost-weighted-l0", action="store_true", dest="hi_cost_weighted_l0",
                   help="2026-09-19: L0 페널티에 HI 카테고리별 계산비용(stat/diff/lfp/morph, "
                        "utils/hi_schema.py::CATEGORY_COSTS)을 가중치로 곱한다. 꺼져 있으면(기본) "
                        "raw/커널 HI 전부 균일 비용 1.0 — 순수 '활성 게이트 개수'만 페널티가 된다. "
                        "raw HI 카테고리 비용을 4_hi_analysis/hi_profile/hi_timing_cost.json 실측치로 "
                        "바꾸면서(stat이 diff/lfp보다 훨씬 비싸게 나옴) 이 항의 영향을 검증하려고 "
                        "잠깐 끄는 토글로 분리했다 — 예전엔(2026-09-19 이전) raw HI 비용 가중치가 "
                        "스위치 없이 항상 켜져 있었다(프로젝트 최초 커밋부터).")
    p.add_argument("--val-rmse-epsilon", type=float, default=0.0005, dest="val_rmse_epsilon",
                   help="2026-09-18(체크포인트 선택 기준 변경): val_rmse가 best보다 이 값 "
                        "'이상' 좋아지면 그 epoch을 '진짜 개선'으로 채택한다. 그게 아니면(개선폭이 "
                        "epsilon 미만이거나 오히려 나빠져도) gate_saturation(sat)이 best보다 "
                        "낮은 epoch을 대신 채택한다 — sat 비교에 val_rmse 쪽 상한(동률 밴드)은 "
                        "따로 없음, 즉 val_rmse가 다소 나빠져도 sat이 새로 낮아졌으면 채택된다"
                        "(예전 기준은 반대로 sat이 1순위, val_rmse가 동률 tie-break였음 — "
                        "2026-09-04 결정을 뒤집음). 기본값 0.0005는 "
                        "p1v4_noscen_gatefix_l0fix_seed42 run의 L0 완전 램프 후 구간"
                        "(epoch 107~206) val_rmse 표준편차 실측치(~0.000246)의 약 2배 — "
                        "epoch-to-epoch 노이즈보다 뚜렷하게 큰 개선만 '진짜 개선'으로 "
                        "인정하려는 값. 실행 조건이 다르면(다른 축/lambda_l0) 노이즈 스케일도 "
                        "다를 수 있으니 log_path의 val_rmse 표준편차를 보고 필요시 조정할 것.")
    p.add_argument("--scen-gate-direction-only", action="store_true",
                   dest="scen_gate_direction_only",
                   help="docs/260917_REPORT.md 안건2 — scen_gates(+scen_kernel_gates)의 "
                        "게이트 뱅크 폭을 n_scenarios 대신 방향 수(2)로 줄이고, 각 scenario_id를 "
                        "spec.scenario_to_dir_class(sid)[0]로 묶어서 라우팅한다(scr_model.py의 "
                        "n_gate_groups). rawonly(라벨 6개)를 noscen(라벨 2개)과 동일한 게이트 "
                        "파편화 프로필로 맞추면서, --scenario-onehot-input과 짝지어 라벨 정보를 "
                        "게이트가 아니라 원샷 입력으로만 전달하는 실험에 사용. --synergy-groups-json"
                        "(전역 scenario_idx로 키된 그룹 게이팅)과는 동시 사용 불가"
                        "(scr_model.py에서 검증). 기본 False면 기존과 동일.")
    p.add_argument("--scenario-onehot-input", action="store_true",
                   dest="scenario_onehot_input",
                   help="docs/260917_REPORT.md 안건2 — 게이트 라우팅과 완전히 별개로, "
                        "batch['level'](진짜 zone/latent_class, position_bin의 실제 라벨) "
                        "원-핫을 cap_head 입력에 direction/cap_init처럼 그냥 이어붙인다 "
                        "(scr_model.py의 scenario_onehot, mlp 헤드만 지원). "
                        "--scen-gate-direction-only와 함께 쓰면 '게이트는 noscen처럼 안 쪼개고, "
                        "라벨은 원샷으로만 알려주기' 조합이 됨. 기본 False면 기존과 동일(head_in 불변).")
    p.add_argument("--warmstart-branch-epoch", type=int, default=None,
                   dest="warmstart_branch_epoch",
                   help="docs/260917_REPORT.md 안건2 '웜스타트 후 분기' 커리큘럼(T-스윕 전용, "
                        "권장 후보 {30,75,150}). 지정하면: (1) SCRModel을 n_gate_groups=1(전체 "
                        "시나리오 공유 단일 게이트)로 만들고, epoch<T 동안은 L0 페널티=0·BETA="
                        "beta_default로 고정(이산화/게이트 압력 없이 전체 데이터로 그 하나의 "
                        "게이트만 학습) — (2) epoch==T에 model.branch_scen_gates()로 그 게이트의 "
                        "log_alpha를 n_scenarios개로 복제해 분기하고, 옵티마이저의 scen_gates 관련 "
                        "Adam 모멘텀을 버리고 새 파라미터로 리셋(다른 모든 파라미터의 옵티마이저 "
                        "상태는 그대로 이어짐) — (3) epoch>=T부터는 L0 warmup/ramp와 BETA anneal이 "
                        "'T를 새 0으로' 삼아 처음부터 다시 시작한다(체크포인트 후보 판정 기준 "
                        "epoch>=l0_fully_ramped_ep도 T만큼 밀림). --scen-gate-direction-only/"
                        "--interaction-json/--synergy-groups-json과는 동시 사용 불가"
                        "(branch_scen_gates()가 shared_hi_mask 없는 일반 HardConcreteGate 단일 "
                        "게이트를 가정). 기본 None이면 기존과 100%% 동일 동작.")
    args = p.parse_args()
    if args.synergy_groups_json and args.kernel_features_pkl:
        p.error("--synergy-groups-json과 --kernel-features-pkl은 동시에 줄 수 없습니다 "
                "(커널 융합을 쓰면 그룹 정보가 이미 피처에 반영되어 게이트 레벨 그룹핑이 불필요함)")
    if args.combined_redundancy_json and not args.kernel_features_pkl:
        p.error("--combined-redundancy-json은 --kernel-features-pkl 없이 줄 수 없습니다 "
                "(그 산출물이 커널 features pkl을 만든 뒤 이어서 생성되는 후속 산출물이라)")
    if args.interaction_json and args.synergy_groups_json:
        p.error("--interaction-json과 --synergy-groups-json은 동시에 줄 수 없습니다 "
                "(--synergy-groups-json은 64폭 전체 기준이라 --interaction-json이 좁히는 "
                "specific 폭과 맞지 않음)")
    if args.scen_gate_direction_only and args.synergy_groups_json:
        p.error("--scen-gate-direction-only와 --synergy-groups-json은 동시에 줄 수 없습니다 "
                "(scr_model.py의 n_gate_groups/scen_group_ids 상호배타 검증과 동일)")
    if args.scen_gate_direction_only and args.kernel_features_pkl:
        p.error("--scen-gate-direction-only와 --kernel-features-pkl은 동시에 줄 수 없습니다 "
                "(2026-09-18부터 커널 게이트가 시나리오별 K_s 폭으로 고정돼 n_gate_groups의 "
                "방향 축소와 개념이 충돌 — scr_model.py에서도 검증)")
    if args.warmstart_branch_epoch is not None:
        if args.warmstart_branch_epoch <= 0:
            p.error("--warmstart-branch-epoch은 1 이상이어야 합니다.")
        if args.kernel_features_pkl:
            p.error("--warmstart-branch-epoch과 --kernel-features-pkl은 동시에 줄 수 없습니다 "
                    "(위와 동일 이유 — n_gate_groups=1로 시작하는 웜스타트와 커널 게이트의 "
                    "시나리오별 K_s 고정폭이 충돌)")
        if args.scen_gate_direction_only:
            p.error("--warmstart-branch-epoch과 --scen-gate-direction-only는 동시에 줄 수 없습니다 "
                    "(전자는 n_gate_groups=1로 시작해 도중에 6으로 분기, 후자는 계속 n_gate_groups=2 고정)")
        if args.interaction_json or args.synergy_groups_json:
            p.error("--warmstart-branch-epoch은 --interaction-json/--synergy-groups-json과 "
                    "동시에 줄 수 없습니다 "
                    "(branch_scen_gates()가 shared_hi_mask 없는 일반 HardConcreteGate 단일 게이트를 가정)")
    return args


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _apply_kernel_features(
    datasets: list, pkl_path: Path, spec, combined_redundancy: dict | None = None,
) -> tuple[dict[int, list[str]], list[int], dict[int, list[float]]]:
    """build_kernel_group_features.py 산출물을 로드해 각 dataset에 x_kernel(정규화된
    RBF 커널 융합값)을 새로 붙인다 — x_hi(raw HI)는 건드리지 않는다(v2: 대체가 아니라
    추가). FastTensorLoader가 ds.x_kernel 존재 여부를 보고 자동으로 배치에 포함시킨다
    (datasets/segment_dataset.py 참고).

    2026-09-18(v4 로직 수정 2차, 요구사항1을 "데이터 0-패딩"이 아니라 **게이트 구조 자체**로
    강제): x_kernel은 이제 시나리오별 **로컬** 인덱스 공간이다 — 폭은 max(K_s)(K_s=그
    시나리오가 own으로 가진 커널 개수)로 고정되고, 각 행은 자기 시나리오의 own 커널만
    [0:K_s) 구간에 채워지며 나머지는 0-패딩이다. SCRModel(kernel_hi_counts=counts)에
    이 K_s 리스트를 넘기면 scen_kernel_gates[s]의 게이트 폭 자체가 K_s로 좁아져서, 그
    시나리오 게이트에는 애초에 다른 시나리오 커널을 위한 슬롯이 존재하지 않는다(이전
    버전은 슬롯은 있되 입력이 상수라 "고를 수는 있지만 의미 없는" 상태였는데, 실측 결과
    실제로 gate_prob>=0.9로 다수 선택되는 게 확인돼 게이트 구조 레벨로 교체함).

    combined_redundancy: build_kernel_group_features.py의 3차(결합 raw+kernel, 시나리오별)
    다중공선성 배제 결과 dict({seg_name: {"removed_kernel_names": [...], ...}}, 요구사항2,
    --combined-redundancy-json). 주어지면 그 시나리오의 로컬 목록에서 해당 커널을 아예
    제외한다(그만큼 K_s가 줄어듦 — raw HI 쪽 배제는 이 함수가 아니라 _build_redundancy_mask()가
    만드는 게이트 레벨 redundancy_mask가 담당, 2026-09-21부터 입력 레벨 nan_mask 이중
    마스킹은 분류기 오염 버그로 제거됨).

    train으로만 fit된 모델을 val/test에도 그대로 적용(predict만)하고, 정규화도 train
    mean/std를 val/test에 그대로 적용(fit 안 함)하므로 누수는 없다.
    반환값이 v1(위 커밋 이전)과 다르다 — (시나리오별 이름 목록 dict, 시나리오별 개수
    리스트, 시나리오별 L0 비용 목록 dict) 3-튜플. 호출자는 둘째를 SCRModel(kernel_hi_counts=...)
    로, 첫째를 게이트 JSON 저장(gates/regression_kernel_HIs.json)에, 셋째를
    SCRModel(kernel_hi_costs=...)로 넘겨 scr_loss.py의 커널 L0 페널티에 쓴다.

    2026-09-18(L0 비용 가중치 추가): 각 커널 HI의 비용은 그 커널을 만든 멤버 raw HI들의
    카테고리 비용(hi_schema.CATEGORY_COSTS)의 평균이다 — build_kernel_group_features.py가
    pkl 저장 시점에 f["cost"]로 미리 계산해둔 값을 그대로 읽는다(구 pkl 호환: 없으면 1.0).
    """
    import pickle
    with open(pkl_path, "rb") as fh:
        artifact = pickle.load(fh)

    features = artifact["features"]
    seg_name_to_idx = {n: i for i, n in enumerate(spec.scenario_names)}

    removed_kernel_by_scen: dict[str, set] = {}
    if combined_redundancy:
        for seg_name, info in combined_redundancy.get("by_scenario", combined_redundancy).items():
            removed_kernel_by_scen[seg_name] = set(info.get("removed_kernel_names", []))

    # 시나리오별 own 커널만 모아 로컬 순서를 고정(원래 pkl 순서 유지) -- 결합 다중공선성
    # 배제로 탈락한 건 여기서 그냥 제외한다(그만큼 K_s가 줄어듦, 요구사항2).
    feats_by_scen: dict[int, list[dict]] = {s: [] for s in range(spec.n_scenarios)}
    n_excluded = 0
    for f in features:
        if f["name"] in removed_kernel_by_scen.get(f["scenario"], ()):
            n_excluded += 1
            continue
        feats_by_scen[seg_name_to_idx[f["scenario"]]].append(f)

    kernel_hi_counts = [len(feats_by_scen[s]) for s in range(spec.n_scenarios)]
    max_k = max(kernel_hi_counts) if kernel_hi_counts else 0
    names_by_scen = {s: [f["name"] for f in feats_by_scen[s]] for s in range(spec.n_scenarios)}
    costs_by_scen = {
        s: [float(f.get("cost", 1.0)) for f in feats_by_scen[s]]
        for s in range(spec.n_scenarios)
    }

    for ds in datasets:
        x = (ds.x_hi * ds.nan_mask).numpy()  # NaN 위치 0으로 (fit 시점과 동일 처리)
        scen_idx_np = ds.scen_idx.numpy()
        x_kernel = np.zeros((x.shape[0], max_k), dtype=np.float32)
        for s in range(spec.n_scenarios):
            row_mask = scen_idx_np == s
            if not row_mask.any():
                continue
            x_scen = x[row_mask]
            for local_j, f in enumerate(feats_by_scen[s]):
                pred = f["model"].predict(x_scen[:, f["members"]])
                x_kernel[row_mask, local_j] = (pred - f["mean"]) / f["std"]  # train 통계로 z-score
        ds.x_kernel = torch.from_numpy(x_kernel.astype(np.float32))

    avg_r2 = float(np.mean([f["train_r2"] for f in features])) if features else 0.0
    print(f"[p1v2] kernel-features-pkl 적용: {pkl_path} "
          f"(커널 HI 시나리오별 폭 {kernel_hi_counts}(max={max_k}, x_hi {N_HI}개와 별도 추가) "
          f"-- 게이트 구조 자체가 own-scenario로 제한됨, 평균 train R^2={avg_r2:.4f})"
          + (f" [combined-redundancy: {n_excluded}개 컬럼 로컬 목록에서 사전 제외]"
             if combined_redundancy else ""))
    return names_by_scen, kernel_hi_counts, costs_by_scen


def _build_redundancy_mask(combined_redundancy: dict, spec) -> torch.Tensor:
    """build_kernel_group_features.py의 3차 결합(raw+kernel) 다중공선성 배제 결과 중 raw HI
    쪽을 SCRModel(redundancy_mask=...)용 bool 텐서 (n_scenarios, N_HI)로 만든다(2026-09-18,
    요구사항2를 raw HI에도 게이트 구조로 강제 — scr_model.py의 _apply_scen_gate가 이 마스크를
    scen_gates 출력에 곱해 False 위치는 log_alpha와 무관하게 항상 0으로 만든다). True=허용,
    False=그 시나리오에서 배제된 raw HI.

    2026-09-21: 이전엔 입력 레벨에서도 nan_mask를 0으로 이중 강제하는 함수
    (_apply_combined_redundancy_raw)를 같이 썼는데, forward()가 그 nan_mask로 마스킹한
    x를 probe_x(분류기 입력)에도 그대로 재사용해서 분류기 정확도가 붕괴하는 버그였다
    (실측: 이 로직 추가 전 98.65% → 추가 후 36~67%). 이 게이트 레벨 마스크 하나만으로
    scen_x 쪽 정확성은 로그 알파와 무관하게 이미 완전히 보장되므로, 입력 레벨 이중
    마스킹은 제거하고 이 함수만 남겼다."""
    by_scenario = combined_redundancy.get("by_scenario", combined_redundancy)
    seg_name_to_idx = {n: i for i, n in enumerate(spec.scenario_names)}
    mask = torch.ones(spec.n_scenarios, N_HI, dtype=torch.bool)
    for seg_name, info in by_scenario.items():
        removed_raw_idx = info.get("removed_raw_idx", [])
        if not removed_raw_idx:
            continue
        s = seg_name_to_idx[seg_name]
        mask[s, removed_raw_idx] = False
    return mask


def _gate_saturation_fraction(model: SCRModel) -> float:
    """게이트 확률이 애매한 [0.1,0.9] 구간에 있는 비율 — 낮을수록 더 확실하게 이산화됨."""
    gates = [model.charge_probe_gate, model.discharge_probe_gate, *model.scen_gates]
    if model.scen_kernel_gates is not None:
        gates += list(model.scen_kernel_gates)
    if model.shared_gate is not None:
        gates.append(model.shared_gate)  # v4: scen_gates가 shared 몫만큼 좁아진 대신
            # shared_gate가 그 몫을 담당하므로, 얘를 빼면 포화도가 실제보다 낮게(더 좋게)
            # 잘못 나온다 — 전체 게이트 파라미터 집합에 반드시 포함해야 함.
    probs = [gate.gate_prob().detach().cpu() for gate in gates]
    p = torch.cat(probs)
    return float(((p > 0.1) & (p < 0.9)).float().mean().item())


def _save_scen_masks_with_shared(model: SCRModel, json_path, hi_cols_by_seg: dict[int, list[str]]) -> None:
    """v4 전용: model.shared_gate가 있으면 scen_gates[s](specific 폭)와 shared_gate를
    원래 컬럼 순서로 재조립해서, 기존과 동일한 seg_{s}_ranked/names/probs/seg_name
    스키마로 저장한다(하위 분석 스크립트가 무수정으로 읽을 수 있게). shared_gate가
    없으면(shared_hi_mask 미지정) train_scr.py의 원본 함수로 그대로 위임."""
    if model.shared_gate is None:
        _base._save_scen_masks_to_json(model, json_path, hi_cols_by_seg)
        return

    n_hi = len(next(iter(hi_cols_by_seg.values())))
    shared_idx = model._shared_idx.detach().cpu()
    specific_idx = model._specific_idx.detach().cpu()
    shared_prob = model.shared_gate.gate_prob().detach().cpu()

    out = {}
    seg_names = model.spec.scenario_names
    for s in range(model.n_scenarios):
        full_prob = torch.zeros(n_hi)
        full_prob[shared_idx] = shared_prob
        if len(specific_idx) > 0:
            full_prob[specific_idx] = model.scen_gates[s].gate_prob().detach().cpu()
        ranked = full_prob.argsort(descending=True).tolist()
        probs = [round(float(full_prob[i]), 6) for i in ranked]
        out[f"seg_{s}_ranked"] = ranked
        out[f"seg_{s}_names"] = [hi_cols_by_seg[s][i] for i in ranked]
        out[f"seg_{s}_probs"] = probs
        out[f"seg_{s}_seg_name"] = seg_names[s]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[p1v2] Saved scen HI ranking(shared_gate 반영) -> {json_path} (시나리오별 {n_hi}개 랭킹)")


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(str(PROJECT_ROOT / args.model_config))

    # 우선순위: CLI 명시 > yaml 값 > 이 파일의 DEFAULT_* 상수. 예전엔 --seg-axis/
    # --axis-config/--data-dir/--seg-data-dir 전부 CLI 기본값이 DEFAULT_*로 고정돼
    # 있어서(None이 아님) yaml에 뭘 적어놔도 매번 조용히 덮어썼다 — 이 4개를 한 번에
    # 고쳐 yaml만으로 축 설정을 완결시킬 수 있게 한다(2026-09-04).
    scenario_cfg = cfg.get("scenario", {}) or {}
    seg_axis = args.seg_axis if args.seg_axis is not None else scenario_cfg.get("axis", DEFAULT_SEG_AXIS)
    if args.axis_config is not None:
        axis_cfg = json.loads(args.axis_config)
    else:
        axis_cfg = scenario_cfg.get("axis_config") or json.loads(DEFAULT_AXIS_CONFIG)
    cfg.setdefault("scenario", {})["axis"] = seg_axis
    cfg["scenario"]["axis_config"] = axis_cfg
    cfg.setdefault("data", {})["split_seed"] = args.split_seed
    cfg["data"]["train_cycle_frac"] = args.train_cycle_frac
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    elif not cfg["data"].get("data_dir"):
        cfg["data"]["data_dir"] = DEFAULT_DATA_DIR
    if args.seg_data_dir is not None:
        cfg["data"]["seg_data_dir"] = args.seg_data_dir
    elif not cfg["data"].get("seg_data_dir"):
        cfg["data"]["seg_data_dir"] = DEFAULT_SEG_DATA_DIR

    cls_cfg = cfg.setdefault("classifier", {})
    reg_cfg = cfg.setdefault("regression", {})
    if args.charge_m is not None: cls_cfg["charge_probe_m"] = args.charge_m
    if args.discharge_m is not None: cls_cfg["discharge_probe_m"] = args.discharge_m
    if args.scen_k is not None: reg_cfg["scen_k_count"] = args.scen_k
    charge_m = cls_cfg.get("charge_probe_m", 10)
    discharge_m = cls_cfg.get("discharge_probe_m", 10)
    scen_k = reg_cfg.get("scen_k_count", 5)

    spec = get_segmenter(seg_axis, {seg_axis: axis_cfg}).get_spec()

    # 데이터 경로: train_scr.py의 자동 경로계산(_axis_dir) 로직을 재사용하지 않으므로,
    # yaml에 없으면 --data-dir/--seg-data-dir로 직접 줘야 한다.
    if not cfg["data"].get("data_dir") or not cfg["data"].get("seg_data_dir"):
        raise RuntimeError(
            "cfg['data']['data_dir']/['seg_data_dir']가 비어 있습니다 — "
            "이 v2 트레이너는 train_scr.py의 자동 경로계산 로직을 재사용하지 않으므로, "
            "--model-config에 이미 박혀있지 않다면 --data-dir/--seg-data-dir을 직접 넘겨주세요."
        )

    train_ds, val_ds, _test_ds, norm = build_datasets(cfg, spec=spec)

    kernel_names_by_scen = None
    kernel_hi_counts = None
    kernel_costs_by_scen = None
    combined_redundancy = None
    redundancy_mask = None
    if args.combined_redundancy_json:
        combined_redundancy = json.loads(Path(args.combined_redundancy_json).read_text(encoding="utf-8"))
    if args.kernel_features_pkl:
        kernel_names_by_scen, kernel_hi_counts, kernel_costs_by_scen = _apply_kernel_features(
            [train_ds, val_ds, _test_ds], Path(args.kernel_features_pkl), spec,
            combined_redundancy=combined_redundancy,
        )
    if combined_redundancy is not None:
        # 2026-09-21: 입력 레벨 nan_mask 이중 마스킹(_apply_combined_redundancy_raw)은
        # 삭제 — forward()가 x = x_hi * nan_mask를 probe_x/scen_x 양쪽에 공유해서 쓰는데,
        # 이 함수가 회귀 전용으로 만든 nan_mask 제로잉이 그대로 probe_x(분류기 입력)까지
        # 오염시키고 있었다(실측: 이 로직 추가 전 run은 분류기 정확도 98.65%, 추가 후
        # 36~67%로 붕괴). redundancy_mask(게이트 출력 강제)만으로 scen_x 쪽 정확성은
        # 이미 완전히 보장되므로(로그 알파와 무관하게 항상 0), 입력 단은 건드리지 않는다.
        redundancy_mask = _build_redundancy_mask(combined_redundancy, spec)
        print(f"[p1v2] combined-redundancy-json 적용: {args.combined_redundancy_json} "
              f"(게이트 출력 0-강제만 적용 — 입력 레벨 마스킹은 분류기 오염 버그로 제거됨, "
              f"{int((~redundancy_mask).sum().item())}개 (시나리오,HI) 조합 배제)")

    tr_cfg = cfg["training"]
    if args.batch_size is not None:
        print(f"[p1v2] batch_size 오버라이드: {tr_cfg['batch_size']} -> {args.batch_size}")
        tr_cfg["batch_size"] = args.batch_size
    train_loader = FastTensorLoader(train_ds, tr_cfg["batch_size"], shuffle=True)
    val_loader = FastTensorLoader(val_ds, tr_cfg["batch_size"], shuffle=False)

    lambda_scen = cfg.get("loss", {}).get("lambda_scen", 0.0)
    with_probe_mlp = lambda_scen > 0
    # 원래 train_scr.py는 여기를 항상 "mlp"로 강제한다(design intent) — 이 v2 트레이너는
    # --regression-model로 그 전제를 sanity-check할 수 있게 열어둔다(기본값은 "mlp"라
    # 오버라이드 안 주면 기존과 100% 동일 동작).
    p1_model_cfg = {**cfg["model"], "regression_model": args.regression_model,
                     "with_raw_cnn": False, "with_raw_flat": False}

    scen_group_ids = None
    if args.synergy_groups_json:
        scen_group_ids = _base._load_synergy_group_ids(
            Path(args.synergy_groups_json), spec.n_scenarios, spec.scenario_names,
        )
        n_groups_total = sum(len(set(g)) for g in scen_group_ids.values())
        print(f"[p1v2] synergy-groups-json 적용: {args.synergy_groups_json} "
              f"({len(scen_group_ids)}/{spec.n_scenarios}개 시나리오, 총 그룹 {n_groups_total}개)")

    shared_hi_mask = None
    if args.interaction_json:
        interaction_data = json.loads(Path(args.interaction_json).read_text(encoding="utf-8"))
        ref_seg_name = spec.scenario_names[0]
        ref_cols = get_hi_cols_for_seg(ref_seg_name)
        suffix = f"_{ref_seg_name}"
        concepts_in_order = [c[: -len(suffix)] if c.endswith(suffix) else c for c in ref_cols]
        per_hi = interaction_data["per_hi"]
        shared_hi_mask = torch.tensor(
            [not per_hi.get(c, {"significant": False})["significant"] for c in concepts_in_order],
            dtype=torch.bool,
        )
        n_shared = int(shared_hi_mask.sum().item())
        print(f"[p1v2] interaction-json 적용: {args.interaction_json} "
              f"({n_shared}/{len(shared_hi_mask)}개 HI -> shared_gate, "
              f"{len(shared_hi_mask) - n_shared}개 -> 기존 scen_gates)")

    model = SCRModel(
        d_probe=cfg["model"]["d_probe"], d_head=cfg["model"]["d_head"], dropout=cfg["model"]["dropout"],
        spec=spec, with_probe_mlp=with_probe_mlp, model_cfg=p1_model_cfg,
        scen_group_ids=scen_group_ids,
        shared_hi_mask=shared_hi_mask,
        kernel_hi_counts=kernel_hi_counts,
        kernel_hi_costs=kernel_costs_by_scen,
        redundancy_mask=redundancy_mask,
        n_gate_groups=(1 if args.warmstart_branch_epoch is not None else
                       2 if args.scen_gate_direction_only else None),
        scenario_onehot=args.scenario_onehot_input,
    ).to(device)
    if args.scen_gate_direction_only:
        print(f"[p1v2] scen-gate-direction-only 적용: 게이트 뱅크 폭 {spec.n_scenarios} -> 2(방향)")
    if args.scenario_onehot_input:
        print(f"[p1v2] scenario-onehot-input 적용: cap_head 입력에 level 원-핫({spec.n_classes}폭) 추가")
    if args.warmstart_branch_epoch is not None:
        print(f"[p1v2] warmstart-branch-epoch 적용: epoch<{args.warmstart_branch_epoch}까지 "
              f"scen_gates 뱅크 폭 1(전체 공유, L0=0/BETA=default) -> epoch {args.warmstart_branch_epoch}에 "
              f"{spec.n_scenarios}개로 분기 후 L0/BETA 스케줄 재시작")

    loss_cfg = cfg["loss"]
    loss_fn = SCRLoss(lambda_scen=lambda_scen, lambda_l0=loss_cfg["lambda_l0"],
                       l0_norm_constant=args.l0_norm_constant,
                       hi_cost_weighted=args.hi_cost_weighted_l0).to(device)
    if args.l0_norm_constant is not None:
        print(f"[p1v2] l0-norm-constant 적용: _l0_penalty를 n_scenarios 대신 "
              f"{args.l0_norm_constant}로 나눔")
    print(f"[p1v2] hi-cost-weighted-l0: {args.hi_cost_weighted_l0} "
          f"({'CATEGORY_COSTS 가중' if args.hi_cost_weighted_l0 else '균일 비용 1.0(기본)'})")

    if args.lambda_l0_override is not None:
        loss_cfg["lambda_l0"] = args.lambda_l0_override
        print(f"[p1v2] lambda_l0_override: {args.lambda_l0_override} (lambda_l0_auto/yaml 값 무시)")
    elif loss_cfg.get("lambda_l0_auto", False):
        avg_m = (charge_m + discharge_m) / 2
        probe_scale = 10 / max(avg_m, 1)
        scen_scale = 10 / max(scen_k, 1)
        auto_lambda = round(max(1e-4, min(0.01 * math.sqrt(probe_scale * scen_scale), 0.5)), 5)
        loss_cfg["lambda_l0"] = auto_lambda
        print(f"[p1v2] lambda_l0_auto: charge_m={charge_m} discharge_m={discharge_m} scen_k={scen_k} -> {auto_lambda}")

    epochs = args.max_epochs if args.max_epochs is not None else tr_cfg["epochs"]
    if args.max_epochs is not None:
        print(f"[p1v2] epochs 상한 오버라이드: {tr_cfg['epochs']} -> {epochs}")
    warmup_ep = tr_cfg.get("warmup_epochs", 10)
    if args.l0_warmup_epochs_override is not None:
        loss_cfg["lambda_l0_warmup_epochs"] = args.l0_warmup_epochs_override
        print(f"[p1v2] l0_warmup_epochs_override: {args.l0_warmup_epochs_override} (yaml 값 무시)")
    l0_scheduler = L0LambdaScheduler(target=loss_cfg["lambda_l0"], loss_cfg=loss_cfg, total_epochs=epochs)
    l0_warmup_ep = loss_cfg.get("lambda_l0_warmup_epochs", 50)
    l0_ramp_ep = loss_cfg.get("lambda_l0_ramp_epochs", 50)
    l0_fully_ramped_ep = l0_warmup_ep + l0_ramp_ep  # 이 에폭부터 체크포인트 후보로 인정

    optimizer = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_ep, 1), eta_min=1e-6)

    beta_default = 2.0 / 3.0
    beta_min = args.beta_min
    warmstart_T = args.warmstart_branch_epoch  # None(기본)이면 아래 로직이 전부 기존과 동일

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = RESULTS_DIR / f"{timestamp}_p1v2_{args.tag}_seed{args.seed}"
    (output_dir / "gates").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoints" / "best_by_saturation.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    spec.save(output_dir / "scenario_spec.json")

    log_path = output_dir / "logs" / "train_log_v2.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lambda_l0,beta,tr_rmse,tr_r2,val_rmse,val_r2,gate_saturation,is_selected\n")

    best_sat = float("inf")  # 2026-09-18부터 1순위 아님 — val_rmse가 epsilon 이내 동률일 때만 tie-break
    best_val_rmse = float("inf")  # 2026-09-18부터 체크포인트 선택 1순위(--val-rmse-epsilon)
    best_epoch = -1
    no_improve = 0  # L0 완전 램프 이후, best 갱신 없이 지난 에폭 수

    # 2026-09-18: config.yaml/p1v2_summary.json을 여기(학습 루프 시작 전)에서도 한 번 써둔다
    # — 원래는 학습이 끝난 뒤(맨 아래)에만 썼는데, 그러면 오래 걸리는 학습을 중간에
    # 중단해도 되는지 확인하다가(예: warmstart T-스윕처럼 500에폭 다 안 돌려도 되는 실험)
    # test_phase1_checkpoint.py가 필수로 요구하는 run_dir/config.yaml이 아직 없어서
    # best_by_saturation.pt가 이미 저장돼 있어도 테스트가 즉시 FileNotFoundError로 막히는
    # 문제가 있었다. 여기서 미리 써두면 checkpoints/best_by_saturation.pt가 한 번이라도
    # 저장된 시점부터는(=epoch>=l0_fully_ramped_ep_eff 도달) 언제 중단해도 바로 테스트 가능
    # — 맨 아래의 최종 write가 best_epoch/gate_saturation을 채워 그대로 덮어쓰므로 정상
    # 종료 시 동작은 100% 기존과 동일.
    cfg.setdefault("data", {})["exclude_stat_leak"] = EXCLUDE_STAT_LEAK
    cfg["data"]["exclude_dqdv_leak"] = EXCLUDE_DQDV_LEAK
    save_config(cfg, output_dir / "config.yaml")
    _early_summary = {
        "tag": args.tag, "seed": args.seed, "split_seed": args.split_seed,
        "lambda_l0_used": loss_cfg["lambda_l0"],
        "lambda_l0_warmup_epochs_used": l0_warmup_ep,
        "selected_epoch": None, "gate_saturation": None,  # 아직 모름 -- 학습 끝나면 채워짐
        "beta_min": beta_min, "l0_fully_ramped_epoch": l0_fully_ramped_ep,
        "output_dir": str(output_dir),
        "n_hi": N_HI, "exclude_stat_leak": EXCLUDE_STAT_LEAK, "exclude_dqdv_leak": EXCLUDE_DQDV_LEAK,
        "synergy_groups_json": args.synergy_groups_json,
        "synergy_n_groups": ({s: max(g) + 1 for s, g in scen_group_ids.items()}
                              if scen_group_ids else None),
        "kernel_features_pkl": args.kernel_features_pkl,
        "combined_redundancy_json": args.combined_redundancy_json,
        "interaction_json": args.interaction_json,
        "l0_norm_constant": args.l0_norm_constant,
        "hi_cost_weighted_l0": args.hi_cost_weighted_l0,
        "scen_gate_direction_only": args.scen_gate_direction_only,
        "scenario_onehot_input": args.scenario_onehot_input,
        "warmstart_branch_epoch": args.warmstart_branch_epoch,
        "regression_model_used": args.regression_model,
        "status": "in_progress",  # 최종 write에서는 이 키가 아예 빠짐(=완료) -- get()으로만 읽는
            # 기존 소비자(test_phase1_checkpoint.py 등)에는 영향 없음
    }
    (output_dir / "p1v2_summary.json").write_text(
        json.dumps(_early_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for epoch in trange(epochs, desc=f"[p1v2:{args.tag}] seed={args.seed}"):
        if epoch < warmup_ep:
            lr = tr_cfg["lr"] * (epoch + 1) / warmup_ep
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # 웜스타트 후 분기: epoch==T에 단일 공유 게이트(scen_gates[0])를 n_scenarios개로
        # 복제해 분기하고, 그 파라미터만 옵티마이저 모멘텀을 리셋한다(다른 모든 파라미터의
        # 옵티마이저 상태는 그대로 이어짐 — docs/260917_REPORT.md "이어서 학습" 결정).
        if warmstart_T is not None and epoch == warmstart_T:
            old_gate_params, new_gate_params = model.branch_scen_gates()
            old_ids = {id(p) for p in old_gate_params}
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if id(p) not in old_ids]
            for p in list(optimizer.state.keys()):
                if id(p) in old_ids:
                    del optimizer.state[p]
            optimizer.param_groups[0]["params"].extend(new_gate_params)
            tqdm_write(f"[p1v2] warmstart-branch: epoch {epoch}에 scen_gates 1 -> "
                       f"{model.n_scenarios}개로 분기, 옵티마이저 모멘텀 리셋")

        # 웜스타트 후 분기가 켜져 있으면 epoch<T 동안은 L0=0/BETA=beta_default로 고정하고,
        # epoch>=T부터는 T를 새 0으로 삼아 L0 warmup/ramp와 BETA anneal을 처음부터
        # 재시작한다(docs/260917_REPORT.md "리셋" 결정). warmstart_T=None이면 eff_epoch=epoch
        # 이라 아래 로직 전부 기존과 100% 동일.
        if warmstart_T is not None and epoch < warmstart_T:
            eff_l0 = 0.0
            beta_now = beta_default
        else:
            eff_epoch = epoch - warmstart_T if warmstart_T is not None else epoch
            eff_l0 = l0_scheduler.get(eff_epoch)
            if eff_epoch < l0_warmup_ep:
                beta_now = beta_default
            elif eff_epoch < l0_fully_ramped_ep:
                frac = (eff_epoch - l0_warmup_ep) / max(l0_ramp_ep, 1)
                beta_now = beta_default + (beta_min - beta_default) * frac
            else:
                beta_now = beta_min
        loss_fn.lambda_l0 = eff_l0
        for gate in [model.charge_probe_gate, model.discharge_probe_gate, *model.scen_gates]:
            gate.BETA = beta_now

        # ---- train epoch ----
        model.train()
        tr_preds, tr_targets = [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch)
            losses = loss_fn(out, batch, model)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg.get("grad_clip", 1.0))
            optimizer.step()
            tr_preds.append(out["cap_pred"].detach().cpu())
            tr_targets.append(batch["target"].cpu())
        tr_p, tr_t = torch.cat(tr_preds).numpy(), torch.cat(tr_targets).numpy()
        tr_rmse_v, tr_r2_v = float(_rmse(tr_t, tr_p)), float(_r2(tr_t, tr_p))

        if epoch >= warmup_ep:
            scheduler.step()

        # ---- val epoch ----
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch)
                val_preds.append(out["cap_pred"].cpu())
                val_targets.append(batch["target"].cpu())
        val_p, val_t = torch.cat(val_preds).numpy(), torch.cat(val_targets).numpy()
        val_rmse_v, val_r2_v = float(_rmse(val_t, val_p)), float(_r2(val_t, val_p))

        sat = _gate_saturation_fraction(model)

        # Stage1: 체크포인트 선택 = "L0가 완전히 램프된 이후" 구간에서 val_rmse 우선,
        # sat은 fallback(2026-09-18, 기준 변경 — 기존 sat-1순위 방식은 docs 2026-09-04
        # 결정 참고). val_rmse가 best보다 --val-rmse-epsilon **이상** 좋아지면 그 epoch을
        # "진짜 개선"으로 채택하고, 그게 아니면(개선폭이 epsilon 미만이거나 오히려 나빠져도)
        # sat이 best보다 낮은 epoch을 채택한다 — epoch마다 val_rmse가 수백 분의 1 수준으로
        # 출렁이는데(실측 표준편차 ~0.00025) 매번 그 노이즈만으로 best가 계속 갈아치워지는
        # 걸 막으면서도, sat이 실제로 더 낮아진(게이트가 더 이산화된) epoch은 val_rmse가
        # 다소 나빠도 놓치지 않기 위함.
        is_selected = False
        _l0_fully_ramped_ep_eff = (warmstart_T or 0) + l0_fully_ramped_ep
        if epoch >= _l0_fully_ramped_ep_eff:
            rmse_diff = val_rmse_v - best_val_rmse
            is_better = (rmse_diff <= -args.val_rmse_epsilon) or (sat < best_sat)
            if is_better:
                best_sat = sat
                best_val_rmse = val_rmse_v
                best_epoch = epoch
                is_selected = True
                no_improve = 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "gate_saturation": sat,
                            "val_rmse": val_rmse_v}, ckpt_path)
            else:
                no_improve += 1

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{eff_l0:.6f},{beta_now:.4f},{tr_rmse_v:.6f},{tr_r2_v:.6f},"
                    f"{val_rmse_v:.6f},{val_r2_v:.6f},{sat:.6f},{int(is_selected)}\n")

        if (epoch + 1) % 10 == 0 or is_selected:
            _msg = (f"epoch {epoch+1:4d}  lambda_l0={eff_l0:.4f}  beta={beta_now:.3f}  "
                    f"tr_r2={tr_r2_v:.4f}  val_r2={val_r2_v:.4f}  sat={sat:.3f}")
            tqdm_write(_msg + (" *selected*" if is_selected else ""))

        # 조기종료: best 갱신 없이 --patience 에폭이 지나면 중단. 이후 남은 에폭을
        # 더 돌아도 이미 저장된 best 체크포인트가 바뀌지 않으므로(항상 진짜 best만
        # 저장) 결과에는 영향 없이 시간만 절약된다 — SCRTrainer의 patience와 동일 원리.
        if args.patience > 0 and no_improve >= args.patience:
            best_ep_str = str(best_epoch + 1) if best_epoch >= 0 else "없음(한 번도 개선 안 됨)"
            tqdm_write(f"[p1v2] 조기종료: epoch {epoch+1} (best epoch={best_ep_str}, "
                       f"{args.patience}에폭 연속 val_rmse/sat 개선 없음)")
            break

    if best_epoch < 0:
        tqdm_write("[p1v2] 경고: L0 완전 램프 이후 구간에서 val_rmse/sat이 한 번도 개선되지 않음 — "
                   "마지막으로 돈 에폭을 그대로 채택합니다(epochs를 늘리거나 beta_min을 더 낮춰보세요).")
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "gate_saturation": sat,
                    "val_rmse": val_rmse_v}, ckpt_path)
        best_epoch = epoch

    # ---- 최종 선택 체크포인트 복원 후 게이트 JSON 저장 (기존 함수 그대로 재사용) ----
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # raw HI(x_hi) 랭킹은 커널 피처 사용 여부와 무관하게 항상 같은 방식으로 저장(v2:
    # 커널은 raw를 대체하지 않고 추가하므로).
    hi_cols_ref = get_hi_cols_for_seg("dis_hi")
    hi_cols_by_seg = {s: get_hi_cols_for_seg(spec.scenario_names[s]) for s in range(spec.n_scenarios)}
    _base._save_probe_masks_to_json(model, output_dir / "gates" / "classification_HIs.json", hi_cols_ref)
    _save_scen_masks_with_shared(model, output_dir / "gates" / "regression_HIs.json", hi_cols_by_seg)

    if kernel_names_by_scen is not None:
        # 2026-09-18부터 커널 피처는 시나리오별 로컬 인덱스 공간(폭 K_s, own 커널만) —
        # kernel_names_by_scen[s]가 이미 그 시나리오 전용 이름 목록(길이 K_s)이다.
        # K_s=0인 시나리오는 scr_model.py가 폭1 더미 게이트를 만들어두므로 이름도 1개
        # 채워준다(실제로 선택될 일은 없음 — 입력이 존재하지 않는 슬롯).
        kernel_cols_by_seg = {
            s: (names if names else ["_unused_slot"])
            for s, names in kernel_names_by_scen.items()
        }
        _base._save_scen_masks_to_json(
            model, output_dir / "gates" / "regression_kernel_HIs.json", kernel_cols_by_seg,
            gates=model.scen_kernel_gates,
        )

    _base._plot_gate_probs(
        model, output_dir / "gates" / "gate_probs.png", hi_cols_ref,
        charge_m, discharge_m, scen_k,
    )

    # config.yaml은 학습 루프 시작 전에 이미 써둠(위 _early_summary 주석 참고) — cfg가 그
    # 이후 안 바뀌므로 여기서 다시 쓸 필요 없음.
    summary = {
        "tag": args.tag, "seed": args.seed, "split_seed": args.split_seed,
        "lambda_l0_used": loss_cfg["lambda_l0"],
        "lambda_l0_warmup_epochs_used": l0_warmup_ep,
        "selected_epoch": best_epoch, "gate_saturation": best_sat,
        "beta_min": beta_min, "l0_fully_ramped_epoch": l0_fully_ramped_ep,
        "output_dir": str(output_dir),
        "n_hi": N_HI, "exclude_stat_leak": EXCLUDE_STAT_LEAK, "exclude_dqdv_leak": EXCLUDE_DQDV_LEAK,
        "synergy_groups_json": args.synergy_groups_json,
        "synergy_n_groups": ({s: max(g) + 1 for s, g in scen_group_ids.items()}
                              if scen_group_ids else None),
        "kernel_features_pkl": args.kernel_features_pkl,
        "combined_redundancy_json": args.combined_redundancy_json,
        "interaction_json": args.interaction_json,
        "l0_norm_constant": args.l0_norm_constant,
        "hi_cost_weighted_l0": args.hi_cost_weighted_l0,
        "scen_gate_direction_only": args.scen_gate_direction_only,
        "scenario_onehot_input": args.scenario_onehot_input,
        "warmstart_branch_epoch": args.warmstart_branch_epoch,  # 정보용 — 최종 체크포인트는
            # 분기 이후(표준 6-게이트) 형태이므로 test_phase1_checkpoint.py 재구성에는 영향 없음
        "regression_model_used": args.regression_model,  # config.yaml에는 CLI 오버라이드 전
            # 원본 yaml 값이 저장돼 실제 학습된 아키텍처와 다를 수 있음이 확인됨
            # (docs/260909_RESULTS.md §7-4) — p1v2_summary.json에 실제 사용값을 남겨
            # 앞으로 같은 함정을 피한다.
    }
    (output_dir / "p1v2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[p1v2] 선택된 epoch={best_epoch} (gate_saturation={best_sat:.4f})")
    print(f"[p1v2] run dir: {output_dir}")


if __name__ == "__main__":
    main()
