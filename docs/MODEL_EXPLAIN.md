# 5_model 코드 상세 설명

LFP 배터리 SOH(State of Health) 예측을 위한 **SCR(Scenario-Conditioned Routing)** 모델의 전체 코드를 파일별로 설명한다.

---

## 전체 흐름 요약

배터리 1**세그먼트**(사이클 내 구간 단위) 데이터 → **64개 HI**(Health Indicator) → **Stage A: 방향별 독립 probe gate로 시나리오(Low/Mid/High) 분류** → **Stage B: 시나리오 조건부 HI 선택 + 용량 예측**

SCR의 핵심 아이디어: "충전과 방전은 서로 다른 HI가 중요하다(클러스터링 결과). 따라서 분류용 probe gate를 방향별로 분리해 각자 최적 HI를 독립적으로 학습한다(Stage A). 그 위에 시나리오별로도 독립적인 회귀용 gate를 둬 용량 예측에 필요한 HI를 추가 선정한다(Stage B)."

---

## 운용 워크플로우 (핵심)

```
┌──────────────────────────────────────────────────────────────────┐
│  Phase 1 — HI 서브셋 선정     python train_scr.py --phase 1      │
│                                                                  │
│  L0 페널티로 최적 HI 탐색 (방향 적응):                              │
│    · charge_probe_gate  : 충전용 probe m1개 (chg_lo/mid/hi 공통) │
│    · discharge_probe_gate: 방전용 probe m2개 (dis_hi/mid/lo 공통) │
│    · scen_gates[0..5]  : 시나리오별 k개 독립 선정 (회귀용)         │
│                                                                  │
│  lambda_l0 스케줄 (delayed_warmup):                               │
│    epoch 0~49    : λ_l0 = 0  (표현 학습 우선, 프루닝 없음)         │
│    epoch 50~149  : λ_l0 = 0 → target 선형 증가 (점진적 압축)      │
│    epoch 150+    : λ_l0 = target 고정 (수렴 안정화)               │
│                                                                  │
│  결과 저장:                                                        │
│    gates/classification_HIs.json  ← charge/discharge 랭킹 분리    │
│    gates/regression_HIs.json      ← 시나리오 6개 × 64개 랭킹      │
│    gates/gate_probs.png           ← gate_prob 시각화 (8 서브플롯) │
└──────────────────────────────────┬───────────────────────────────┘
                                   │  JSON 완성 → gate 학습 종료
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Phase 2 — 분류·회귀 정밀 학습  python train_scr.py --phase 2     │
│                                                                  │
│  JSON 로드 → 고정 마스크 모델 재구성:                                │
│    · _charge_probe_mask_buf    (N_HI,) bool  — 충전용 고정         │
│    · _discharge_probe_mask_buf (N_HI,) bool  — 방전용 고정         │
│    · _scen_masks_buf (N_SEGS, N_HI) bool     — 시나리오별 고정     │
│                                                                  │
│  학습 목표: MSE + λ_scen·CE만 (L0=0 자동)                        │
│    probe_mlp + cap_head 가중치만 업데이트                           │
│    결과: checkpoints/ + figures/ + metrics/ + routing/           │
└──────────────────────────────────┬───────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  추론 — test_scr.py                                              │
│    입력 세그먼트 → 방향 판별                                        │
│    → 충전이면 charge probe m1개 HI로 Lo/Mid/Hi 분류                │
│    → 방전이면 discharge probe m2개 HI로 Lo/Mid/Hi 분류             │
│    → 해당 시나리오의 scen k개 HI로 용량(Ah) 예측                   │
│    (m + k HIs 재사용, gate 학습 없음, 순수 forward pass)           │
└──────────────────────────────────────────────────────────────────┘
```

### 실행 명령 요약

```bash
# Phase 1: 방향별 최적 HI 탐색 (L0 학습)
python 5_model/train_scr.py --phase 1
python 5_model/train_scr.py --phase 1 --charge-m 3 --discharge-m 1 --scen-k 5

# Phase 2: 고정 게이트로 MLP/Head 정밀 학습
python 5_model/train_scr.py --phase 2
python 5_model/train_scr.py --phase 2 --gates-from _5_data_model_scr/0709_0221

# 평가 (Phase 2 결과 체크포인트 기준)
python 5_model/test_scr.py
python 5_model/test_scr.py --checkpoint _5_data_model_scr/0709_0221/checkpoints/best.pt

# [legacy] --no-gates는 --phase 1, yaml gates_from 설정은 --phase 2와 동일하게 동작
```

---

## 학습 목표 및 최적화 과정

SCR 학습은 단일 optimizer(AdamW)가 아래 목표를 **동시에** 추구한다.  
모든 파라미터(gate log_alpha, probe_mlp, scen_gates, cap_head)가 매 step마다 함께 업데이트된다.

---

### 1. 분류에 사용할 HI 선정 — 방향별 독립 probe gate (Stage A)

**관련 파라미터**:
- `charge_probe_gate.log_alpha` — 64차원, 충전 세그먼트 전용
- `discharge_probe_gate.log_alpha` — 64차원, 방전 세그먼트 전용

**동작 원리**:
- 충전 샘플은 `charge_probe_gate`, 방전 샘플은 `discharge_probe_gate`로 분기 (direction > 0 조건)
- 각 gate는 독립적으로 `log_alpha_i`를 학습하며 서로 다른 HI 조합에 수렴 가능
- `gate_prob_i = σ(log_alpha_i − β·log(−γ/ζ))` = 이 방향에서 HI_i가 선택될 확률
- L0 손실이 각 방향별 gate를 스파스하게 당기고, CE 손실이 Lo/Mid/Hi 분류에 유용한 HI를 살려두는 방향으로 당겨 균형이 맞는 지점에서 수렴
- 추론 시 hard binary gate: `(σ(log_alpha_i) > 0.5)`

**Phase 1 종료 후**: gate_prob 내림차순으로 전체 64개 랭킹을 charge/discharge 분리해 JSON 저장.  
Phase 2 로드 시 yaml의 `charge_probe_m`개 / `discharge_probe_m`개만 고정 마스크로 절삭.

---

### 2. 회귀에 사용할 HI 선정 (Stage B gate 학습)

**관련 파라미터**: `scen_gates[0..5].log_alpha` — 시나리오 6개 × 64차원, 총 384개

**동작 원리**:
- 시나리오(충방전 × SOH 레벨)별로 별도의 `HardConcreteGate(64)`가 있어 각자 최적 HI 조합을 학습
- L0 페널티에서 HI i의 "실제 사용 확률"은 probe와 scen gate를 합산:
  ```
  P(active_i) = 1 − (1−p_probe_i)(1−p_scen_i)
  ```
  probe가 이미 HI를 쓴다면 scen gate는 추가 비용 없이 활용 가능
- 충전 시나리오(seg 0-2)는 `p_probe_ch`, 방전 시나리오(seg 3-5)는 `p_probe_dis` 사용
- MSE 손실이 시나리오별 scen HI 선택을 유도하고, L0 페널티가 각 시나리오의 HI 개수를 압축

**Phase 1 종료 후**: `scen_k_count`개 per 시나리오 랭킹이 `regression_HIs.json`에 저장됨.

---

### 3. 분류기 가중치 갱신 (Stage A MLP 학습)

**관련 파라미터**: `probe_mlp` — Linear(64→d_probe→d_probe/2→3), 기본 약 6.2K 파라미터

**학습 신호**:
```
L_ce = CrossEntropy(probe_mlp(probe_x), level_true)
probe_x = x_hi * z_probe   (게이트로 필터된 HI 입력)
```

- `lambda_scen × L_ce`의 gradient가 probe_mlp를 업데이트
- 동일한 gradient가 `probe_gate.log_alpha`까지 전파 → gate 학습과 MLP 학습이 **하나의 backward pass에서 동시에** 수행
- Phase 2(고정 마스크)에서는 게이트 파라미터가 없으므로 MLP 가중치만 업데이트됨

**수렴 방향**: probe_x가 Low/Mid/High를 잘 구분하는 방향으로 MLP 가중치 조정

---

### 4. 회귀기 가중치 갱신 (Capacity Head 학습)

**관련 파라미터**: `cap_head` — Linear(129→d_head→d_head/2→1), 기본 약 24.7K 파라미터

**학습 신호**:
```
L_mse = MSE(cap_head([probe_x ∥ scen_x ∥ direction]), target_norm)
```

- MSE gradient가 cap_head → scen_gate → probe_gate 순으로 전파
- 입력 `129 = 64(probe) + 64(scen) + 1(direction)`:
  - `probe_x`: Stage A gate로 필터된 HI (분류용 HI가 용량 예측에도 기여 — m+k 재사용)
  - `scen_x`: 해당 시나리오 gate로 필터된 HI
  - `direction`: 충전(+1) / 방전(−1) 스칼라
- `target_norm`은 z-score 정규화된 capacity_Ah (추론 후 역변환으로 실제 Ah 복원)
- probe_x는 m개 비-zero, scen_x는 k개 비-zero → 수학적으로 m+k+1개 HI로 예측하는 것과 동일

**수렴 방향**: 선택된 HI들로 SOH(용량) 예측 오차를 최소화

---

### 5. 그 외 학습 메커니즘

#### 복합 손실 구조

```
L_total = L_mse + λ_scen × L_ce + λ_l0(epoch) × L_l0
```

| 항목 | 기본값 | 영향 |
|------|--------|------|
| `L_mse` | — | 용량 예측 정확도 |
| `λ_scen = 0.5` | `loss.lambda_scen` | 시나리오 분류 정확도 |
| `λ_l0(epoch)` | **스케줄러로 동적 변화** | HI 개수 최소화 (계산 비용 절감) |

세 항목이 하나의 `total.backward()`로 모든 파라미터를 동시에 업데이트한다.

---

#### lambda_l0 스케줄러 (Phase 1 전용)

**문제**: λ_l0를 처음부터 크게 주면 gate가 수렴하기 전에 무작위로 닫혀버려 의미 있는 HI를 놓칠 수 있다.

**해결**: `lambda_l0_schedule: "delayed_warmup"` — 먼저 표현을 자유롭게 학습시킨 뒤 점진적으로 압축

```
epoch 0 → warmup_epochs (기본 50):
    λ_l0 = 0
    → gate 없이 MSE + CE 학습. 모든 HI가 probe_mlp / cap_head 학습에 기여.
    → log_alpha_i가 정보량에 따라 어느 정도 방향성을 잡음

epoch warmup_epochs → warmup_epochs + ramp_epochs (기본 50~150):
    λ_l0 = target × (epoch − warmup_epochs) / ramp_epochs  (선형 증가)
    → L0 압력이 서서히 올라오면서 불필요한 HI의 log_alpha가 음수로 이동
    → 유용한 HI는 CE/MSE gradient가 지탱해 살아남음

epoch warmup_epochs + ramp_epochs 이후:
    λ_l0 = target  (고정)
    → gate_prob 수렴 안정화 (cliff 형태로 bimodal 분포)
```

4가지 `lambda_l0_schedule` 옵션:

| 옵션 | 동작 | 용도 |
|------|------|------|
| `none` | 매 에폭 target 고정 (기존 동작) | 빠른 실험 |
| `delayed_warmup` | 0 유지 후 선형 증가 (**권장**) | Phase 1 기본 전략 |
| `exp_ramp` | epoch²에 비례 증가 (느린 초반) | 더 긴 warmup이 필요할 때 |
| `cyclic` | 0 ↔ target 코사인 사이클 | HI 수를 주기적으로 재탐색할 때 |

**Phase 2에서는 L0 페널티 항 자체가 0이므로 스케줄러 설정 무관.**

---

#### 동적 lambda_l0 기준값 (`lambda_l0_auto: true`, Phase 1 전용)

스케줄러의 "target" 값을 yaml의 `lambda_l0`로 고정하거나, `lambda_l0_auto: true`로 자동 계산한다.

자동 계산 시 목표 HI 수가 적을수록 L0 페널티를 강하게 줘서 gate가 그 근방에서 수렴하도록 유도:

```
λ_target = 0.01 × √(10/m × 10/k)
  기준: λ=0.01 → 경험적으로 ~10 probe HI, ~10 scen HI 선택
```

| m / k | λ_target |
|-------|---------|
| m=1,  k=5  | 0.04472 |
| m=3,  k=5  | 0.02582 |
| m=5,  k=5  | 0.02000 |
| m=5,  k=10 | 0.01414 |
| m=10, k=10 | 0.01000 (기준) |

이 값이 `delayed_warmup` 스케줄러의 `target`이 된다.

---

#### 학습률 스케줄

- **Warmup**: 처음 10 epoch은 lr을 0 → `lr_max`(5e-4)로 선형 증가 (초기 불안정 방지)
- **Cosine Annealing**: warmup 이후 `lr_max → 1e-6`으로 서서히 감소

---

#### 조기 종료 (Early Stopping)

- **기준**: val RMSE가 개선될 때마다 `checkpoints/best.pt` 갱신
- **patience=20**: 20 epoch 연속 비개선 시 학습 중단 (총 epoch 상한 500)
- **복원**: 학습 종료 후 best 체크포인트를 자동 복원

---

#### Phase 2에서의 변화

| 항목 | Phase 1 (L0) | Phase 2 (고정 마스크) |
|------|-------------|----------------------|
| charge_probe_gate.log_alpha | 학습됨 | **없음** |
| discharge_probe_gate.log_alpha | 학습됨 | **없음** |
| scen_gates[*].log_alpha | 학습됨 | **없음** |
| L0 손실 항 | 활성 (스케줄러로 제어) | **0 (자동 비활성)** |
| lambda_l0 스케줄러 | 동작 | **무관** |
| probe_mlp | 학습됨 | 학습됨 |
| cap_head | 학습됨 | 학습됨 |
| 선택된 HI | L0가 결정 | JSON 고정 |

Phase 2는 Phase 1에서 찾은 HI로 고정한 뒤, MLP/Head만 재학습해서 성능을 끌어올리는 용도다.

---

## 디렉토리 구조

```
5_model/
  config/
    scr.yaml                SCR 하이퍼파라미터 전체 정의
  utils/
    compat.py               numpy 버전 호환성 패치 (pkl 로드 전 필수)
    tqdm_utils.py           tqdm 안전 래퍼 (미설치 환경 fallback)
    hi_schema.py            64-HI 스키마·비용·SCEN_MAP·누수 배제
    metrics.py              회귀 평가 지표 (RMSE, MAE, R², MAPE)
    io_utils.py             체크포인트·설정·JSON 저장/로드
  datasets/
    segment_dataset.py      세그먼트 단위 로드·셀분할·정규화·DataLoader
  models/
    hard_concrete.py        Hard-Concrete L0 게이트 (Louizos 2018)
    scr_model.py            SCRModel (Stage A/B + Capacity Head)
  training/
    scr_loss.py             SCRLoss (MSE + CE + L0 방향별 비용가중 페널티)
    scr_trainer.py          SCRTrainer (학습 루프·L0 스케줄러·LR 스케줄러·early stopping)
  evaluation/
    scr_evaluator.py        SCREvaluator (메트릭·scatter·capacity curve)
  train_scr.py              학습 진입점 (--phase 1 / --phase 2)
  test_scr.py               평가 진입점
```

---

## 1. config/scr.yaml — 하이퍼파라미터 설정

모델 구조, 데이터, 손실, 학습, 평가에 필요한 모든 숫자값을 한 곳에 모아 관리한다.

### 주요 섹션

| 섹션 | 핵심 항목 | 설명 |
|------|-----------|------|
| `data` | `is_real_input` | `false`=사전계산 HI / `true`=raw V·I·t |
| `data` | `data_dir` | wide pkl 루트 (`_4_data_hi`) |
| `data` | `seg_data_dir` | 네이티브 seg 포맷 루트 (있으면 우선) |
| `data` | `output_dir` | run 결과 저장 루트 (`_5_data_model_scr`) |
| `data` | `gates_from` | 이전 run 폴더 경로 — 해당 run의 gates JSON 재사용 (`null`이면 L0 학습) |
| `data` | `gates_ignore` | `true`=`gates_from` 무시하고 L0부터 재학습 |
| `classifier` | `charge_probe_m` | 충전용 probe HI 수 (Phase 2 고정 마스크 절삭 기준) |
| `classifier` | `discharge_probe_m` | 방전용 probe HI 수 |
| `classifier` | `probe_m_count` | legacy fallback (charge/discharge_probe_m 미설정 시) |
| `regression` | `scen_k_count` | Phase 2/test 시 시나리오별 HI 수 |
| `model` | `d_probe` | Stage A MLP hidden dim |
| `model` | `d_head` | Capacity Head hidden dim |
| `loss` | `lambda_scen` | CE 손실 가중치 |
| `loss` | `lambda_l0_auto` | `true` = m/k 기반 자동 계산 (권장), `false` = `lambda_l0` 직접 사용 |
| `loss` | `lambda_l0` | `lambda_l0_auto=false` 시 직접 지정값 (스케줄러 target) |
| `loss` | `lambda_l0_schedule` | `none` / `delayed_warmup` / `exp_ramp` / `cyclic` |
| `loss` | `lambda_l0_warmup_epochs` | delayed_warmup: λ=0 유지 에폭 수 (기본 50) |
| `loss` | `lambda_l0_ramp_epochs` | delayed_warmup: target까지 선형 증가 에폭 수 (기본 100) |
| `loss` | `lambda_l0_cycle_epochs` | cyclic: 사이클 주기 (기본 100) |
| `training` | `lr` | 초기 학습률 (기본 5e-4) |
| `training` | `warmup_epochs` | LR warmup 구간 epoch 수 (기본 10) |
| `training` | `scheduler` | `cosine` / `step` |
| `training` | `early_stop_patience` | 조기 종료 patience (기본 20) |
| `evaluation` | `rep_cells_per_dataset` | 각 데이터셋에서 자동 선택할 대표 셀 수 (기본 3) |
| `evaluation` | `rep_cells` | 대표 셀 직접 지정 (auto-pick 대신 사용) |

---

## 2. utils/compat.py — numpy 버전 호환성 패치

### 문제 배경
- Step 4 HI pkl 생성 환경: numpy 2.1.3
- Step 5 학습 환경(`LFP_SOH_ESTIMATION`): numpy 1.24.4
- numpy 2.0에서 내부 모듈이 `numpy.core.*` → `numpy._core.*`로 이동
- numpy 1.x 환경에서 numpy 2.x pkl을 열면 `ModuleNotFoundError` 발생

### 해결 (`sys.modules` 패치)

```python
if not hasattr(np, "_core"):
    fake = types.ModuleType("numpy._core")
    fake.__dict__.update(numpy.core.__dict__)
    sys.modules["numpy._core"] = fake  # pickle unpickler 경로에 등록
```

`install_numpy2_shim()`을 pkl 로드 전에 한 번만 호출하면 된다. `train_scr.py`·`test_scr.py` 상단에서 자동 실행된다.

---

## 3. utils/tqdm_utils.py — tqdm 안전 래퍼

`tqdm` 미설치 환경에서도 코드가 조용히 동작하도록 fallback을 제공한다.

```python
try:
    from tqdm import tqdm as _tqdm
    def tqdm(iterable, **kwargs): return _tqdm(iterable, **kwargs)
except ImportError:
    def tqdm(iterable=None, **kwargs): return iterable
```

---

## 4. utils/hi_schema.py — 64-HI 스키마

SCR 전체에서 HI 컬럼명, 카테고리 비용, 세그먼트 메타데이터를 단일 진실 소스로 제공한다.

### 핵심 상수

```python
STAT_KEYS  = [...]  # 20개 정의, 2개 제외 → 18개 사용
DIFF_KEYS  = [...]  # 20개
LFP_KEYS   = [...]  # 20개
MORPH_KEYS = [...]  #  6개

LEAK_COLS = {"stat_q_abs", "stat_energy_seg"}  # capacity_Ah 누수 HI → 입력 제외

SEGMENTS = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]

SCEN_MAP = {
    "chg_lo": (1, 0), "chg_mid": (2, 1), "chg_hi": (3, 2),
    "dis_hi": (-3, 3), "dis_mid": (-2, 4), "dis_lo": (-1, 5),
}  # seg_name → (scen_code, seg_idx)

CATEGORY_COSTS = {"stat": 1.0, "diff": 1.5, "lfp": 2.0, "morph": 3.0}
N_HI   = 64   # stat 18 + diff 20 + lfp 20 + morph 6
N_SEGS = 6
```

### N_HI = 64 계산

```
STAT  20개 − stat_q_abs 1개 − stat_energy_seg 1개 = 18개
DIFF  20개                                         = 20개
LFP   20개                                         = 20개
MORPH  6개                                         =  6개
합계                                               = 64개
```

`stat_q_abs`와 `stat_energy_seg`는 모두 capacity_Ah와 강한 선형 상관관계를 가져 입력으로 사용하면 누수(label leakage)가 발생하므로 제외한다.

### 주요 함수

```python
get_hi_cols_for_seg("dis_hi")
# → ["stat_v_mean_dis_hi", ..., "morph_ve_frec_dis_hi"]  (64개)

get_hi_cost_vector("dis_hi")
# → [1.0, 1.0, ..., 1.5, ..., 2.0, ..., 3.0, 3.0]  (64개)
```

---

## 5. utils/metrics.py — 평가 지표

| 지표 | 수식 | 의미 |
|------|------|------|
| RMSE | √(Σ(ŷ−y)²/N) | 오차 크기 (Ah 단위), 이상치에 민감 |
| MAE  | Σ\|ŷ−y\|/N | 오차 크기 (Ah 단위), 이상치에 덜 민감 |
| R²   | 1 − SS\_res/SS\_tot | 설명력. 1.0에 가까울수록 좋음 |
| MAPE | Σ\|ŷ−y\|/y × 100 | 상대 오차 (%). 실제 용량 대비 몇 % 틀렸나 |

`compute_metrics(y_true, y_pred)` → `{"rmse": ..., "mae": ..., "r2": ..., "mape": ...}`

---

## 6. utils/io_utils.py — 파일 입출력

| 함수 | 역할 |
|------|------|
| `load_config` / `save_config` | YAML 설정 읽기/쓰기 |
| `save_checkpoint` / `load_checkpoint` | PyTorch 모델·옵티마이저 상태 저장/로드 |
| `save_json` | 지표·이력 JSON 저장 |
| `save_pickle` / `load_pickle` | 객체 저장/로드 |

`load_checkpoint`에서 `map_location=device`를 지정해 GPU 체크포인트를 CPU에서도 로드 가능.

---

## 7. datasets/segment_dataset.py — 세그먼트 데이터셋

### 로딩 우선순위

```
1순위: _4_data_hi/seg/{MIT,HUST}/*.pkl  (네이티브 seg 포맷)
2순위: _4_data_hi/{MIT,HUST}/*.pkl      (기존 wide 포맷, reshape)
```

### Wide → Segment Reshape (`_wide_to_segments`)

```
wide pkl 1행 (사이클 1개, ~415 컬럼)
  ↓  세그먼트 6개 × 반복
  각 행: hi_00..hi_63 (64개 HI), direction, level, scen, capacity_Ah
  capacity_Ah = 사이클 레벨 전체 용량 (SOH 타깃)
```

### SegmentNormalizer

- HI 64개 개별 z-score (훈련셋으로만 fit)
- NaN → z-score 0.0 (평균으로 대체). nan\_mask로 위치 별도 보존
- `capacity_Ah` 별도 z-score (역변환 `inverse_target` 제공)

### SegmentDataset 텐서

```python
x_hi      : (N_HI,)   정규화된 HI           [float32]
nan_mask  : (N_HI,)   1.0=유효, 0.0=NaN     [float32]
direction : scalar     +1.0(충전) / -1.0(방전)
level     : scalar     0/1/2 정답 레벨       [int64]
seg_idx   : scalar     0~5 세그먼트 인덱스   [int64]
target    : scalar     정규화된 capacity_Ah  [float32]
```

### build_datasets()

진입점에서 호출하는 최상위 빌더. wide/native 자동 선택 → split → 정규화 → SegmentDataset 3개 반환.

```python
train_ds, val_ds, test_ds, norm = build_datasets(cfg)
```

---

## 8. models/hard_concrete.py — Hard-Concrete L0 게이트

### 왜 Hard-Concrete인가

| 방식 | 미분 가능 | 정확히 0 | 배포 시 이진 |
|------|-----------|-----------|-------------|
| Gumbel-Sigmoid | O | X (≈ 0) | 임계값 처리 필요 |
| Hard-Concrete | O | **O** | **자연스럽게 0/1** |

Hard-Concrete는 샘플링된 z가 정확히 0이 될 수 있어 "이 HI를 계산하지 않는다"는 결정이 수학적으로 명확하다.

### 수식 (β=2/3, γ=−0.1, ζ=1.1)

```
학습 시:
  u ~ Uniform(0, 1)
  s = sigmoid((log u − log(1−u) + log_α) / β)
  z = clamp(s·(ζ−γ) + γ, 0, 1)

추론 시 (hard):
  s = sigmoid(log_α)
  z = clamp(s·(ζ−γ) + γ, 0, 1)
  active = (z > 0)   ← 정확히 이진 마스크
```

### L0 비용 확률 (`gate_prob`)

```python
P(z > 0) ≈ sigmoid(log_α − β·log(−γ/ζ))
# 학습 중 이 확률에 category cost를 곱해 L0 페널티 계산
```

### gate_prob의 Bimodal 수렴 (cliff 현상)

Phase 1 학습이 진행되면서 `gate_prob` 분포는 자연스럽게 **bimodal**(이중봉)로 수렴한다:
- 유용한 HI: MSE/CE gradient가 log_alpha를 밀어올려 gate_prob → 1
- 불필요한 HI: L0 페널티가 log_alpha를 끌어내려 gate_prob → 0

이것이 regression_HIs.json에서 상위 2~3개만 gate_prob ≈ 0.999이고 나머지는 ≈ 0.00004인 "cliff" 패턴의 원인이다. `delayed_warmup` 스케줄러는 λ_l0가 0인 동안 log_alpha가 의미 있는 방향으로 배치된 후에야 L0 압력을 가해, cliff 발생 전 충분한 표현 학습을 보장한다.

---

## 9. models/scr_model.py — SCRModel

### 아키텍처

```
입력: x_hi (B,64), nan_mask (B,64), direction (B,), seg_idx (B,)

[Stage A — 방향별 probe gate + 시나리오 분류기]
  direction > 0  → charge_probe_gate(64)    → probe_x (B,64) ← 충전 m1개 활성
  direction <= 0 → discharge_probe_gate(64) → probe_x (B,64) ← 방전 m2개 활성
  공유 MLP(64 → d_probe → N_LEVELS) → level_logits (B,3)

[Stage B — 시나리오 조건부 게이트]
  HardConcreteGate × 6 (시나리오별 독립, 각 64개)
  seg_idx로 해당 gate 선택 → scen_x (B,64)

[Capacity Head — m+k HI 재사용]
  concat(probe_x, scen_x, direction) → (B, 64+64+1=129)
  MLP(129 → d_head → d_head//2 → 1) → cap_pred (B,)
  ← probe_x (m개 비-zero) + scen_x (k개 비-zero) = 유효 m+k+1 입력
```

**`_CHARGE_SEGS = frozenset({0, 1, 2})`** (chg_lo, chg_mid, chg_hi)  
**`_DISCHARGE_SEGS = frozenset({3, 4, 5})`** (dis_hi, dis_mid, dis_lo)

### 두 가지 동작 모드

#### Phase 1 모드 — L0 게이트 학습

```python
model = SCRModel()  # 마스크 없음 → HardConcreteGate 활성화
# charge_probe_gate    = HardConcreteGate(64)   ← log_alpha 64개 학습
# discharge_probe_gate = HardConcreteGate(64)   ← log_alpha 64개 학습
# scen_gates = ModuleList([HardConcreteGate(64)] × 6)
# state_dict 키: "charge_probe_gate.log_alpha", "discharge_probe_gate.log_alpha",
#                "scen_gates.N.log_alpha"
```

#### Phase 2 모드 — 고정 마스크 재학습 / 추론

```python
model = SCRModel(
    charge_probe_mask=ch_mask,      # (N_HI,) bool
    discharge_probe_mask=dis_mask,  # (N_HI,) bool
    scen_masks=masks,               # (N_SEGS, N_HI) bool
)
# charge_probe_gate/discharge_probe_gate = None
# _charge_probe_mask_buf / _discharge_probe_mask_buf (버퍼, 학습 안 됨)
# _scen_masks_buf (버퍼, 학습 안 됨)
# L0 손실 항 자동 0, probe_mlp + cap_head만 학습
```

#### test_scr.py의 자동 판별

```python
ckpt_has_l0 = ("charge_probe_gate.log_alpha" in state_keys or
               "probe_gate.log_alpha" in state_keys)  # legacy compat
# Phase 1 ckpt → JSON 로드 후 고정 마스크로 모델 재구성 (strict=False)
# Phase 2 ckpt → 이미 마스크 모드이므로 JSON으로 일치
```

### 파라미터 수 (기본 설정 d_probe=64, d_head=128)

```
charge_probe_gate   : 64개 log_alpha     ≈   0.06K
discharge_probe_gate: 64개 log_alpha     ≈   0.06K
Stage A MLP (공유)  : 64×64+64×32+32×3  ≈   6.2K
Stage B gates (6×64): 6×64 log_alpha    ≈   0.4K
Capacity Head       : 129×128+128×64+64 ≈  24.7K
합계                                     ≈  31.5K
```

---

## 10. training/scr_loss.py — SCRLoss

### 손실 수식

```
L = MSE(cap_pred, cap_target)
  + λ_scen × CE(level_logits, level)
  + λ_l0(epoch) × Σ_i  cost_i × P(gate_i active)

P(gate_i active) = 1 − (1−p_probe_i)(1−p_scen_i)
```

`cost_i` 는 HI 카테고리별 계산 비용 (stat=1.0 / diff=1.5 / lfp=2.0 / morph=3.0).

**L0 페널티의 방향별 분기**:
- 충전 시나리오(seg 0,1,2): `p_probe = charge_probe_gate.gate_prob()`
- 방전 시나리오(seg 3,4,5): `p_probe = discharge_probe_gate.gate_prob()`

**Phase 2(고정 마스크)에서는 `_fixed_probe=True, _fixed_scen=True`이므로 `_l0_penalty()`가 즉시 0을 반환한다. 손실 = MSE + λ_scen·CE 만 남는다.**

### λ 튜닝 가이드

| 항목 | 기본값 | 크게 하면 | 작게 하면 |
|------|--------|-----------|-----------|
| `lambda_scen` | 0.5 | 시나리오 분류 우선 | 용량 예측 우선 |
| `lambda_l0` (target) | **자동 계산** | HI 수 최소화 (희소) | 많은 HI 허용 (정확) |
| `lambda_l0_warmup_epochs` | 50 | 표현 학습 기회 증가 | 초기부터 압축 |
| `lambda_l0_ramp_epochs` | 100 | 점진적 수렴 | 급격한 pruning |

---

## 11. training/scr_trainer.py — SCRTrainer

### L0LambdaScheduler

```python
class L0LambdaScheduler:
    def get(self, epoch: int) -> float:
        if schedule == "none":
            return target
        if schedule == "delayed_warmup":
            if epoch < warmup_ep: return 0.0
            progress = (epoch - warmup_ep) / ramp_ep
            return target * min(progress, 1.0)
        if schedule == "exp_ramp":
            return target * (epoch / total_epochs) ** 2
        if schedule == "cyclic":
            phase = (epoch % cycle_ep) / cycle_ep
            return target * 0.5 * (1 - cos(π * phase))
```

`SCRTrainer.__init__`에서 `L0LambdaScheduler` 인스턴스를 생성하고, `fit()` 루프 매 에폭 시작 시:

```python
eff_l0 = self.l0_scheduler.get(epoch)
self.loss_fn.lambda_l0 = eff_l0  # loss_fn에 직접 주입
```

Phase 2에서는 loss_fn이 내부적으로 L0=0을 반환하므로 `eff_l0` 값 자체는 무관.

### 학습 스케줄

#### LR Cosine (Warmup 포함)

```
워밍업 (epoch 0~10):
  lr = 5e-4 × (epoch+1) / 10  (선형 증가)
코사인 감쇠 (이후):
  CosineAnnealingLR(T_max=500−10=490, eta_min=1e-6)
```

#### Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
```

MSE + CE + L0 합산 손실의 그래디언트 스케일 불안정을 방지한다.

### 조기 종료 (Early Stopping)

val RMSE가 개선될 때마다 **full checkpoint**를 `checkpoints/best.pt`에 저장한다.

```python
if val_rmse < best_val_rmse:
    best_val_rmse = val_rmse
    no_improve = 0
    _save_model(model, output_dir / "checkpoints" / "best.pt", cfg, normalizer)
else:
    no_improve += 1
if no_improve >= patience:
    break
_load_model(model, ckpt_path)  # 완료 후 best 가중치 복원
```

### 체크포인트 포맷

```python
{
    "model_state": model.state_dict(),   # 모든 파라미터
    "cfg":         cfg,                  # scr.yaml 전체 (재현용)
    "norm_mean":   normalizer.mean_,     # HI z-score 평균
    "norm_std":    normalizer.std_,      #          표준편차
    "norm_target_mean": ...,             # capacity z-score 평균
    "norm_target_std":  ...,             #                  표준편차
}
```

normalizer 통계가 `.pt` 안에 포함되므로 별도 `normalizer.pkl` 없이 평가 가능하다.

### CSV 학습 로그 (`logs/train_log.csv`)

매 epoch마다 한 행씩 추가된다.

| 컬럼 | 설명 |
|------|------|
| `epoch` | 에폭 번호 |
| `tr_loss` / `tr_mse` / `tr_ce` / `tr_l0` | 훈련 손실 분해 |
| `lambda_l0` | **해당 에폭의 effective λ_l0** (스케줄러 출력, Phase 1에서 모니터링용) |
| `val_loss` / `val_mse` / `val_rmse` / `val_mae` / `val_r2` | 검증 지표 |
| `lr` | 현재 학습률 |
| `elapsed_s` | 에폭 소요 시간(초) |

### 에폭 콘솔 출력 형식

```
epoch   10 | tr_loss=1.2345 mse=0.9876 ce=0.3456 l0=102.34 λ_l0=0.0000 | val_loss=0.5123 rmse=0.0234 r2=0.9812 | lr=9.80e-04 t=51.0s
epoch   51 | tr_loss=1.1234 mse=0.9123 ce=0.3200 l0=98.12  λ_l0=0.0003 | val_loss=0.4987 rmse=0.0221 r2=0.9834 | lr=9.60e-04 t=51.5s *
```

`*` 표시는 val RMSE 갱신(best) 에폭. `λ_l0`가 0이면 warmup 중, 증가하면 ramp 중임을 바로 확인 가능하다.

---

## 12. evaluation/scr_evaluator.py — SCREvaluator

### 생성자 파라미터

```python
SCREvaluator(
    model,
    normalizer,
    device,
    figures_dir: Path,   # 그래프 저장 디렉토리
    rep_cells: list[str],
)
```

### 메서드 목록

| 메서드 | 역할 |
|--------|------|
| `predict_dataset(ds, batch_size)` | 배치 추론 → 정규화 역변환 → 예측 dict 반환 |
| `evaluate(train_ds, val_ds, test_ds)` | 전체 split 메트릭 계산 + 그래프 저장 |
| `save_metrics(results, metrics_dir)` | `metrics_dir/metrics.json` 저장 (중첩 구조) |
| `save_predictions(pred_dict, predictions_dir)` | `predictions_dir/test_predictions.csv` 저장 |
| `plot_routing_heatmap(routing_dir, probe_sel, scen_sel)` | 7×64 활성화 히트맵 + CSV 저장 |
| `_plot_scatter(pred_dict, tag)` | `figures_dir/scatter_{tag}.png` 저장 |
| `_plot_capacity_curves(pred_dict)` | 대표 셀별 3-패널 `figures_dir/capacity_curve_{cell}.png` |
| `_plot_confusion_matrix(pred_dict, tag)` | `figures_dir/confusion_matrix_{tag}.png` 저장 |

### predict_dataset 반환 dict

```python
{
    "cap_pred_raw":  (N,) Ah      # 역정규화된 예측 용량
    "cap_true_raw":  (N,) Ah      # 역정규화된 실측 용량
    "cap_pred_norm": (N,)         # 정규화 공간 예측값
    "cap_true_norm": (N,)         # 정규화 공간 실측값
    "level_pred":    (N,) int     # 예측 시나리오 레벨 (0/1/2)
    "level_true":    (N,) int     # 정답 시나리오 레벨
    "seg_idx":       (N,) int     # 세그먼트 인덱스 (0~5)
    "direction":     (N,) float   # +1.0(충전) / -1.0(방전)
    "probe_z":       (N, 64)      # probe gate 활성화값 (0 또는 1)
    "scen_z":        (N, 64)      # scen gate 활성화값
    "cell_ids":      list[str]
    "cycles":        list[int]
    "seg_names":     list[str]
    "cap_raw":       (N,) Ah      # 원본 capacity (정규화 전)
}
```

### Capacity Curve (대표 셀)

```python
# 세그먼트 → 사이클 집계
pred_per_cyc = [cap_pred[cycles == cy].mean() for cy in unique_cycles]
true_per_cyc = [cap_true[cycles == cy].mean() for cy in unique_cycles]
```

3-패널 플롯 (figsize 17×4):
- **패널 1**: 사이클 vs capacity Ah (실측 실선 + 예측 점선)
- **패널 2**: 사이클 vs |절대 오차| Ah
- **패널 3**: 사이클 vs 상대 오차 % (`|err|/true × 100`)

### metrics.json 포맷

```json
{
  "train": {
    "capacity":   {"rmse": 0.0165, "mae": 0.0079, "r2": 0.9560, "mape": 0.77},
    "breakdown":  {"charge": {...}, "discharge": {...}, "level_low": {...}, ...},
    "scenario":   {"accuracy": 0.92, "confusion_matrix": [[...]]},
    "efficiency": {"avg_probe_his": 2.0, "avg_scen_his": 5.0}
  },
  "val":  { ... },
  "test": { ... }
}
```

### test_predictions.csv 컬럼

`cell_id, cycle, seg_name, cap_true_Ah, cap_pred_Ah, error_Ah, level_true, level_pred, probe_active_n, scen_active_n`

---

## 13. 진입점 — train_scr.py

### 역할

1. CLI `--phase` 옵션 또는 yaml 설정으로 Phase 1 / Phase 2 자동 판별
2. 타임스탬프 run 폴더 생성 (`_5_data_model_scr/MMDD_HHMM/`)
3. `gates_from` / `--gates-from` 경로에서 gate JSON 탐색
4. 데이터 빌드 (native seg/ 우선 → wide reshape fallback)
5. JSON 없으면(Phase 1): L0 게이트 포함 학습 → JSON + gate_probs.png 저장
6. JSON 있으면(Phase 2): 고정 마스크 모델 재구성 → MLP/Head만 학습
7. **`lambda_l0_auto=true`이면 m/k 기반 `lambda_l0` 동적 계산** (Phase 1 스케줄러 target)
8. `checkpoints/final.pt` 저장

### Phase 판별 로직

```python
def _resolve_phase(args, cfg) -> int:
    if args.phase is not None:                    return args.phase  # CLI 최우선
    if args.no_gates:                             return 1           # legacy
    if cfg.get("data", {}).get("gates_from"):     return 2           # legacy
    return 1                                                         # 기본값
```

### 옵션 전체

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--config PATH` | `5_model/config/scr.yaml` | YAML 설정 파일 경로 |
| `--phase {1,2}` | auto | 1=L0 학습, 2=고정 마스크 재학습 |
| `--charge-m INT` | config 값 | 충전용 probe HI 수 |
| `--discharge-m INT` | config 값 | 방전용 probe HI 수 |
| `--scen-k INT` | config 값 | 시나리오별 HI 수 |
| `--gates-from PATH` | config의 `gates_from` | 이전 run 폴더 경로 (CLI > yaml) |
| `--no-gates` | `false` | gates_from 무시, L0 재학습 (legacy) |
| `--device STR` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` |

### 실행 예시

```bash
# ── Phase 1: L0 게이트 학습 (최초 1회) ─────────────────────────────
python 5_model/train_scr.py --phase 1
# → L0로 최적 HI 선정 → gates/*.json + gate_probs.png 저장

# m/k 수동 지정
python 5_model/train_scr.py --phase 1 --charge-m 3 --discharge-m 1 --scen-k 5

# ── Phase 2: 게이트 고정, MLP/Head 정밀 학습 ───────────────────────
# yaml의 gates_from에 이전 run 경로 설정 후 실행
python 5_model/train_scr.py --phase 2

# 다른 run의 gates 명시
python 5_model/train_scr.py --phase 2 --gates-from _5_data_model_scr/0709_0221

# GPU 지정
python 5_model/train_scr.py --phase 1 --device cuda:0

# conda run (환경 미활성화 상태에서)
conda run -n LFP_SOH_ESTIMATION python 5_model/train_scr.py --phase 1
```

### Gate JSON 탐색·저장 동작

```
우선순위: --no-gates / gates_ignore=true → 항상 L0 학습 (JSON 무시)
         --gates-from CLI > yaml의 gates_from > None

탐색 시 파일명 폴백:
  {gates_dir}/gates/classification_HIs.json   (신버전 구조)
  {gates_dir}/classification_HIs.json         (없으면)
  {gates_dir}/scenario_classification_HIs.json (구버전 폴백)

저장:
  gates 캐시에서 로드한 경우 → 원본 JSON을 새 run의 gates/ 에 복사
  L0 학습한 경우              → 모델에서 추출해 gates/ 에 저장
  (항상 새 run 폴더에 gates JSON 있음 → test_scr.py가 신뢰성 있게 탐색 가능)
```

### Phase 1 추가 출력

```
_5_data_model_scr/MMDD_HHMM/
  gates/
    classification_HIs.json    ← charge/discharge 분리 랭킹
    regression_HIs.json        ← 시나리오 6개 × 64개 랭킹
    gate_probs.png             ← 8 서브플롯 bar chart (charge probe, discharge probe, seg0-5)
                                  각 subplot: gate_prob 내림차순, 빨간 점선=top-m/k 임계선, 상위 5개 이름 표기
```

### 출력 구조 (전체)

```
_5_data_model_scr/
  MMDD_HHMM/
    checkpoints/
      best.pt                  val RMSE 최솟값 시점 full checkpoint
      final.pt                 학습 완료 후 best 가중치 복원 full checkpoint
    gates/
      classification_HIs.json
      regression_HIs.json
      gate_probs.png           (Phase 1만)
    logs/
      train_log.csv            epoch별 손실·lambda_l0·지표·LR·시간
    config.yaml                이 run에 사용한 scr.yaml 스냅샷
```

---

## 14. HI 랭킹 생성 메커니즘 — classification_HIs.json / regression_HIs.json

### 두 파일이 하는 일

| 파일 | 용도 | 내용 |
|------|------|------|
| `classification_HIs.json` | Stage A (시나리오 분류기) 입력 HI 선정 | charge/discharge probe gate의 gate_prob 내림차순으로 **방향별 분리** 64개 전체 랭킹 |
| `regression_HIs.json` | Stage B (용량 예측기) 입력 HI 선정 | 6개 시나리오 각각의 scen gate gate_prob로 독립 랭킹 |

Phase 1 학습이 끝나면 이 두 파일에 **전체 64개 HI의 중요도 순위**가 저장된다. Phase 2·추론에서는 yaml의 `charge_probe_m`/`discharge_probe_m`개 / `scen_k_count`개만 잘라 사용한다.

---

### gate_prob란 무엇인가

Hard-Concrete 게이트는 HI마다 학습 파라미터 `log_alpha_i` 하나를 갖는다.

```
gate_prob_i = σ(log_alpha_i − β·log(−γ/ζ))
            ≈ P(z_i > 0)  (이 HI가 켜질 확률)

β=2/3, γ=−0.1, ζ=1.1 (하이퍼파라미터 고정)
```

Phase 1 학습 동안 `log_alpha_i`는 **두 힘의 줄다리기**로 수렴한다.

| 방향 | 힘 | 원인 |
|------|-----|------|
| log_alpha_i ↑ (HI 켜기) | CE / MSE 손실 | 이 HI가 있으면 분류·예측 오차 감소 |
| log_alpha_i ↓ (HI 끄기) | L0 페널티 | 켜진 HI 수(× 비용) 만큼 손실 증가 |

`delayed_warmup` 스케줄러로 warmup 기간(epoch 0~49) 동안 L0 페널티 = 0으로 유지하면, 모든 log_alpha가 먼저 **실제 정보량에 따라 자유롭게 배치**된 뒤 ramp 기간(epoch 50~149)부터 L0 압력이 서서히 가해진다. 결과적으로 bimodal 수렴이 더 안정적으로 이루어진다.

---

### classification_HIs.json 생성 흐름

charge_probe_gate와 discharge_probe_gate 각각의 랭킹이 **분리 저장**된다.

```
Phase 1 학습 완료
    │
    ▼
_ranked_indices(model.charge_probe_gate)    → ch_ranked, ch_probs
_ranked_indices(model.discharge_probe_gate) → dis_ranked, dis_probs
    │
    ▼
JSON 저장
{
  "charge_ranked":    [idx, ...]  ← 충전 gate_prob 내림차순
  "charge_names":     [...]
  "charge_probs":     [0.9999, 0.9998, 0.9990, ...]
  "discharge_ranked": [idx, ...]  ← 방전 gate_prob 내림차순
  "discharge_names":  [...]
  "discharge_probs":  [...]
}
```

신구 구조에서는 충전/방전 gate가 독립 수렴하므로, 클러스터링 결과처럼 충전은 더 많은 HI(charge_probe_m=3), 방전은 더 적은 HI(discharge_probe_m=1)에 자연스럽게 수렴할 수 있다.

---

### regression_HIs.json 생성 흐름

```
Phase 1 학습 완료
    │
    ▼
시나리오 6개(chg_lo, chg_mid, chg_hi, dis_hi, dis_mid, dis_lo)마다 독립 처리
    │
    ├─ _ranked_indices(model.scen_gates[0])  → seg_0_ranked, seg_0_probs
    ├─ _ranked_indices(model.scen_gates[1])  → seg_1_ranked, seg_1_probs
    │  ...
    └─ _ranked_indices(model.scen_gates[5])  → seg_5_ranked, seg_5_probs
    │
    ▼
JSON 저장
{
  "seg_0_ranked":   [23, 26, 14, ...]   ← chg_lo gate의 gate_prob 내림차순
  "seg_0_names":    [...]
  "seg_0_probs":    [0.9999, 0.9999, 0.9999, 0.00004, ...]
  "seg_0_seg_name": "chg_lo",
  "seg_1_ranked":   [...],
  ...
}
```

각 시나리오 gate는 **완전히 독립적으로 학습**된다. L0 페널티에서 probe gate와 scen gate의 합산 확률이 사용되기 때문에 probe가 이미 켠 HI라면 scen gate에는 추가 페널티가 줄어든다.

```
P(active_i) = 1 − (1−p_probe_i)(1−p_scen_i)
```

즉 probe가 이미 HI i를 켜고 있으면(`p_probe_i ≈ 1`), scen gate가 같은 HI를 켜도 L0 비용이 거의 늘지 않는다. 반대로 probe가 끈 HI를 scen gate가 켜면 비용이 온전히 부과된다.

`scen_k_count=5` (yaml 설정)이므로 추론에서는 **시나리오별 상위 5개 HI**만 사용한다.

---

### Phase 2·추론에서의 사용

```python
# classification_HIs.json → charge/discharge probe_mask (N_HI,) bool
ch_ranked  = data["charge_ranked"]          # 64개 전체 랭킹
dis_ranked = data["discharge_ranked"]
ch_mask[ch_ranked[:charge_probe_m]]   = True   # 상위 m1개 충전 마스크
dis_mask[dis_ranked[:discharge_probe_m]] = True # 상위 m2개 방전 마스크

# regression_HIs.json → scen_masks (N_SEGS, N_HI) bool
for seg in range(6):
    ranked = data[f"seg_{seg}_ranked"]   # 세그먼트별 64개 랭킹
    scen_masks[seg, ranked[:scen_k_count]] = True
```

L0 게이트(파라미터 없음)가 고정 bool 마스크로 교체되면:
- `charge_probe_gate / discharge_probe_gate = None` → `_charge_probe_mask_buf / _discharge_probe_mask_buf` (버퍼)
- `scen_gates = None` → `_scen_masks_buf` (6 × 64 bool 버퍼)
- L0 손실 항 자동 0
- probe_mlp + cap_head만 학습 (Phase 2) 또는 추론만 (test)

---

### 요약

```
Phase 1 (L0 학습, delayed_warmup 스케줄)
  epoch 0~49:   λ_l0=0  →  자유로운 표현 학습
  epoch 50~149: λ_l0 선형 증가  →  bimodal 수렴 시작
  epoch 150+:   λ_l0=target  →  gate_prob ≈1 or ≈0 수렴
               ↓ gate_prob 내림차순 정렬
  classification_HIs.json  [charge_ranked / discharge_ranked 분리, 64개씩]
  regression_HIs.json      [시나리오 6개 × 64개 scen 랭킹]
  gate_probs.png           [8 서브플롯 시각화]

Phase 2 / 추론
  charge_ranked[:charge_probe_m]  → 충전 probe 고정 마스크
  discharge_ranked[:discharge_probe_m] → 방전 probe 고정 마스크
  seg_N_ranked[:scen_k_count]     → 시나리오별 scen 고정 마스크
  probe_mlp + cap_head 재학습 (Phase 2) 또는 forward-only (추론)
```

---

## 15. 진입점 — test_scr.py

**Phase 2 추론 진입점.** 게이트 학습 없음, 선정된 HI 조합으로 순수 forward pass.

### 역할

1. 체크포인트 자동 탐색 또는 `--checkpoint` 명시
2. 저장된 config·normalizer 복원
3. 동일 split seed로 데이터 재구성 (train/val/test 동일 셀 보장)
4. **gates JSON 우선 로드** → 항상 JSON 기준으로 HI 고정
   - JSON 있음 → 고정 마스크 모델 재구성. Phase 1 체크포인트라도 `strict=False`로 MLP/Head만 로드
   - JSON 없음 → L0 모델 그대로 재구성 (fallback)
5. 전체 split 메트릭 계산 + 그래프·CSV·routing 저장
6. 활성 HI 통계 출력

### 옵션 전체

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--config PATH` | `5_model/config/scr.yaml` | YAML 설정 파일 경로 |
| `--checkpoint PATH` | 자동 탐색 | 체크포인트 경로 |
| `--rep-cells STR [...]` | auto-pick | capacity curve 대표 셀 직접 지정 |
| `--device STR` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` |

### 대표 셀 자동 선택 (`_pick_rep_cells`)

`eval_cfg.get("rep_cells_per_dataset", 1)` 값으로 각 데이터셋에서 셀을 자동 선택한다.  
yaml `evaluation.rep_cells_per_dataset: 3`이면 MIT 3개 + HUST 3개 = 총 6개 자동 선택.

```python
def _pick_rep_cells(test_ds, cfg, n_per_dataset: int = 1) -> list[str]:
    for ds_name in datasets:
        candidates = sorted cells in test split belonging to ds_name
        picked.extend(candidates[:n_per_dataset])
```

평가 config는 **현재 yaml**(cfg)에서 읽어온다 — 체크포인트 저장 당시의 cfg_saved가 아니라, 실행 시점 yaml의 evaluation 섹션 기준이다.

### 콘솔 출력 예시

```
[test] device=cuda
[test] checkpoint: _5_data_model_scr/0709_0221/checkpoints/best.pt
[test] probe gate JSON: .../gates/classification_HIs.json
[test] scen  gate JSON: .../gates/regression_HIs.json
[test] rep_cells: ['b1c0', 'b1c1', 'b1c2', '10-7', '10-8', '10-9']

=== SCR Evaluation Summary ===
split         RMSE       MAE        R2      MAPE
------------------------------------------------
train       0.0165    0.0079    0.9560      0.77
val         0.0102    0.0063    0.9845      0.60
test        0.0158    0.0080    0.9548      0.78

Active charge probe HIs    : 3/64
Active discharge probe HIs : 1/64
Avg scen HIs/seg           : 5.0/64
  chg_lo    : 5 HIs selected
  chg_mid   : 5 HIs selected
  chg_hi    : 5 HIs selected
  dis_hi    : 5 HIs selected
  dis_mid   : 5 HIs selected
  dis_lo    : 5 HIs selected
```

### 출력 파일

```
_5_data_model_scr/MMDD_HHMM/
  figures/
    scatter_test.png
    capacity_curve_{cell}.png      (rep_cells 수만큼 생성)
    confusion_matrix_test.png
  metrics/
    metrics.json
  predictions/
    test_predictions.csv
  routing/
    routing_heatmap.png            7×64 히트맵 (charge probe ∪ discharge probe + 6 시나리오)
    routing_table.csv
```

---

## 파라미터 수 분석

기본 설정 (d_probe=64, d_head=128) 기준:

| 서브모듈 | 파라미터 수 |
|---------|------------|
| charge_probe_gate (HardConcreteGate) | 64 ≈ 0.06K |
| discharge_probe_gate (HardConcreteGate) | 64 ≈ 0.06K |
| Stage A MLP (공유) | 64×64+64×32+32×3 ≈ 6.2K |
| HardConcreteGate (6×scen) | 6×64 ≈ 0.4K |
| Capacity Head | 129×128+128×64+64×1 ≈ 24.7K |
| **합계** | **≈ 31.5K** |

배터리 데이터 특성상 셀 수(~200개)와 전체 세그먼트 수(~60만)로 과적합 방지가 중요하기 때문에 소형 모델로 설계했다.

---

## 데이터 흐름 요약

```
_4_data_hi/seg/{MIT,HUST}/*.pkl  (네이티브 seg 포맷, 우선)
_4_data_hi/{MIT,HUST}/*.pkl      (wide 포맷, fallback → reshape)
        │
        ▼
segment_dataset.py
  load_dataset_native_seg()    native: HI 컬럼 hi_00..hi_63 매핑
  (또는 load_dataset_wide())   wide: 사이클당 6행 reshape
  split_cells()                셀 단위 60/20/20 분할 (seed 고정)
  SegmentNormalizer.fit()      훈련셋으로만 z-score 피팅 (HI 64개 + target)
  SegmentDataset()             텐서 사전 빌드
  DataLoader                   배치 생성
        │
        ▼  (한 배치)
  x_hi      (B, 64)            정규화된 HI (stat_q_abs, stat_energy_seg 제외)
  nan_mask  (B, 64)            NaN 유효 마스크
  direction (B,)               +1.0(충전) / -1.0(방전)
  seg_idx   (B,)               0~5 세그먼트 인덱스
  level     (B,)               0/1/2 정답 레벨
  target    (B,)               정규화된 capacity_Ah
        │
        ▼
SCRModel.forward()
  x = x_hi * nan_mask                                  NaN 위치 제거
  Stage A:
    direction > 0  → charge_probe_gate(x)    → probe_x (B,64)  [m1개 비-zero]
    direction <= 0 → discharge_probe_gate(x) → probe_x (B,64)  [m2개 비-zero]
    level_logits   = probe_mlp(probe_x)                (B, 3)
  Stage B:
    scen_x = scen_gates[seg_idx](x)                    (B,64)  [k개 비-zero]
  Head:
    feat     = concat(probe_x, scen_x, direction)      (B, 129)
    cap_pred = cap_head(feat)                          (B,)
        │
        ▼
SCRLoss(outputs, batch, model)
  MSE(cap_pred, target)
  + λ_scen × CE(level_logits, level)
  + λ_l0(epoch) × Σ cost_i × P(gate_i active)   ← delayed_warmup으로 epoch 제어
    P(active_i) = 1 − (1 − p_probe_i)(1 − p_scen_i)
    충전 시나리오: p_probe = charge_probe_gate.gate_prob()
    방전 시나리오: p_probe = discharge_probe_gate.gate_prob()
        │
        ▼
optimizer.step()               파라미터 업데이트
        │  Phase 1 학습 완료 후
        ▼
gates/ 저장:
  classification_HIs.json  {charge_ranked, discharge_ranked} ← 방향별 분리
  regression_HIs.json      {seg_0_ranked, ..., seg_5_ranked} ← 시나리오별
  gate_probs.png           8 서브플롯 bar chart
checkpoints/best.pt
checkpoints/final.pt
logs/train_log.csv         (lambda_l0 컬럼 포함)
config.yaml

        │  Phase 2 / test_scr.py 실행
        ▼
classification_HIs.json 탐색
  charge_ranked[:charge_probe_m]    → _charge_probe_mask_buf
  discharge_ranked[:discharge_probe_m] → _discharge_probe_mask_buf
regression_HIs.json 탐색
  seg_N_ranked[:scen_k_count]       → _scen_masks_buf
SCRModel(masks) 재구성 → Phase 2 재학습 or 추론만

        │  test 완료
        ▼
figures/scatter_test.png
figures/capacity_curve_*.png        (rep_cells_per_dataset × n_datasets 개)
metrics/metrics.json
predictions/test_predictions.csv
routing/routing_heatmap.png         7×64 (charge∪discharge probe + 6 scen)
```
