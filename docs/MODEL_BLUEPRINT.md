# SCR — Scenario-Conditioned HI Routing (설계 문서)

> 2단계 캐스케이드로 **① 소수 HI로 시나리오(Low/Mid/High)를 식별**하고,
> **② 시나리오별로 필요한 최소 HI만 추가 계산**해 **세그먼트 용량**을 예측하는
> 적응적 최소연산(conditional computation) 프레임워크.

이 문서는 `5_model`의 **SCR(Scenario-Conditioned Routing)** 아키텍처의 설계·구현 명세다.

---

## 1. 설계 목표

| 항목 | 목표 |
|------|------|
| 선택 단위 | **개별 HI 65개** 단위 — 카테고리를 넘나드는 최소 조합 표현 |
| 희소화 | Hard-Concrete **L0** — 게이트 개수를 직접 최소화 |
| 시나리오 | **명시적 3-way 분류** — "몇 개 HI로 상황을 아는가"를 정량화 |
| 처리 단위 | **세그먼트** — 사이클당 6개, 시나리오별 구조 활용 |

핵심 서사: "먼저 소수의 저렴한 HI로 배터리 운영 시나리오를 식별하고, 그 시나리오에 필요한 HI만 추가로 계산해 용량을 추정한다. 무엇을 계산할지를 학습한다."

---

## 2. 문제 정의

- **입력 단위 = "세그먼트"** (사이클 아님). 한 사이클은 전류 적산(Coulomb counting)으로 6개 세그먼트로 분할된다. → 사이클당 예측 6개.
- 전류 적산 분할은 `4_hi_analysis/hi_correlation.py::_extract_one_cell`가 `q_cum = cumsum(|i|·dt)`를 구간(Low/Mid/High)으로 자르는 **결정론적·초저비용** 연산이다. HI 계산이 필요 없다.
- **모델 가정**: 모델은 세그먼트 행리스트만 받고 **그 세그먼트가 어떤 시나리오(Low/Mid/High)인지 모른다.** 스스로 최소 HI로 맞춰야 한다.
- **충/방전 방향**은 전류 부호로 이미 알 수 있으므로 **공짜 입력**으로 준다. 모델이 분류하는 것은 **레벨(Low/Mid/High) 3-way**뿐.
- **타깃**: 각 세그먼트의 용량 `capacity_Ah` (정답지 그대로, 세그먼트당 1개).

### 시나리오(scen) 라벨 규약

정답지 `_4_data_hi/seg/` 각 행의 `scen ∈ {-3,-2,-1,+1,+2,+3}`:

- `direction = sign(scen)` → 충/방전 (공짜 입력)
- `level = |scen| - 1 ∈ {0,1,2}` → Low/Mid/High (분류 대상, CE 지도학습)
- `context = direction × level` → 6개 시나리오 (Stage B 조건화 단위)

---

## 3. 아키텍처

```
익명 세그먼트 (정규화된 HI 65개 + nan_mask + 알려진 direction)
        │
        ▼  Stage A ── 시나리오 식별
   probe HI 선택 (Hard-Concrete L0, 전역 고정 마스크)
        x · mask · g_probe , direction ─▶ MLP ─▶ level_logits (3)
        │                                   ⟵ CE(level 정답)로 지도학습
        ▼  level_probs (soft)
   context_probs (B,6) = known direction ⊗ predicted level
        │
        ▼  Stage B ── 시나리오 조건부 추가 HI 획득
   log_alpha_B = context_probs @ ScenGateLogits(6,65)
   g_B = Hard-Concrete(log_alpha_B)          ← 시나리오별 다른 마스크
        │
        ▼  선택된 HI 통합 (soft-OR)
   g_tot = g_probe ⊕ g_B ;  x_sel = x · mask · g_tot
        │
        ▼  Capacity Head
   MLP([x_sel, direction, level_probs]) ─▶ capacity_Ah (세그먼트당 1개)
```

### 손실

```
L = MSE(cap_pred, cap_target)                     # 용량 정확도 (타깃 표준화)
  + λ_scen · CrossEntropy(level_logits, level)    # 시나리오 식별
  + λ0    · Σ_i cost_i · P(computed_i)            # 계산할 HI 총개수 최소화(L0, 비용가중)
```

- `P(computed_i) = 1 − (1−p_probe_i)(1−p_B_i)` — probe·StageB **합집합**을 세어 "실제 계산되는 HI 개수"를 직접 페널티.
- `cost_i`: 카테고리별 상대비용 stat 1.0 / diff 1.5 / lfp 2.0 / morph 3.0.
- λ0 하나로 "HI 개수 vs 정확도" Pareto가 자동 결정된다(사용자 선택: λ 자유방식).

### Hard-Concrete L0 게이트 (Louizos et al. 2018)

미분 가능하면서 **정확히 0**이 될 수 있는 게이트. 파라미터 `log_alpha`로부터:

- 학습: `s = σ((logU−log(1−U)+log_alpha)/β)`, `z = clamp(s(ζ−γ)+γ, 0, 1)`
- 추론: `s = σ(log_alpha)` (노이즈 없음) → 결정론적 게이트, `z>0`이면 "계산함"
- L0 페널티: `P(gate>0) = σ(log_alpha − β·log(−γ/ζ))`
- 상수: `β=2/3, γ=−0.1, ζ=1.1`

Stage A는 **전역 파라미터**(`probe_log_alpha (65,)`, 입력 무관 고정 probe 집합),
Stage B는 **시나리오 조건부 로짓**(`ScenGateLogits (6,65)` → 문맥가중 합)을 Hard-Concrete에 통과.

---

## 4. ⚠️ 데이터 누수 주의 (중요)

정답지에서 **`stat_q_abs == capacity_Ah` (일치율 100%)**. 즉 `stat_q_abs`는 타깃 그 자체다.
→ **입력 HI에서 `stat_q_abs`를 제외**한다(`LEAK_COLS`). 제외 후 HI = **65개**.

추가 유의: `stat_energy_seg = ∫V·|i|dt ≈ V̄·Q`로 타깃과 강한 상관(near-leak)일 수 있다.
기본은 포함하되 `LEAK_COLS`에 넣어 배제 가능하도록 설정으로 노출한다. (실험에서 이 HI가 지나치게 지배적이면 배제 권장.)

---

## 5. 학습 ↔ 배포 분리

- **학습/검증**: 정답지 `seg/`에 HI 65개가 이미 계산돼 있으므로, 게이트로 **마스킹만**(계산했다고 가정)하고 L0로 개수를 최소화. 게이트가 미분 가능해 end-to-end 학습.
- **배포**: raw(`clean/`) → 전류 적산 분할 → **게이트가 켠 HI만** `_seg_stat/_seg_diff/...` 선택 호출 → 실제 연산 절감. (별도 선택적 HI 계산기로 후속 구현)

---

## 6. 파일 구성

```
5_model/
  config/scr.yaml                 SCR 하이퍼파라미터
  utils/
    compat.py                     numpy 버전 호환성 패치 (pkl 로드 전 필수)
    tqdm_utils.py                 tqdm 안전 래퍼
    hi_schema.py                  65-HI 스키마·비용·scen→(level,direction)·누수배제
    metrics.py                    RMSE / MAE / R² / MAPE
    io_utils.py                   체크포인트·설정·JSON 저장/로드
  datasets/
    segment_dataset.py            세그먼트 단위 로드/셀분할/정규화/타깃표준화
  models/
    hard_concrete.py              Hard-Concrete L0 게이트
    scr_model.py                  SCRModel (Stage A/B + head)
  training/
    scr_loss.py                   SCRLoss (MSE + CE + L0)
    scr_trainer.py                SCRTrainer (학습 루프)
  evaluation/
    scr_evaluator.py              SCREvaluator (용량+시나리오+HI사용 통계)
  train_scr.py                    학습 진입점
  test_scr.py                     평가 진입점
```

---

## 7. 산출물 / 평가지표

`_5_data_model_scr/<run>/` 아래:

- **capacity**: RMSE / MAE / R² / MAPE (세그먼트 단위, 필요시 direction·level별 분해)
- **scenario**: level 분류 정확도, confusion matrix
- **efficiency**: 평균 계산 HI 개수, 평균 비용, probe/StageB 사용률
- **routing/**: `scen(6) × HI(65)` 활성화 히트맵 → "시나리오별 최적 HI 표" (핵심 그림)
- `predictions/*.csv`: 세그먼트별 실측·예측·예측레벨·게이트 상태

---

## 8. 논문 서사

> "모든 HI를 항상 계산하지 않는다. 몇 개의 저렴한 HI만으로 배터리 운영 시나리오를 식별하고,
> 그 시나리오에 필요한 HI만 추가로 계산해 용량을 추정한다. 무엇을 계산할지를 학습한다."

SCR의 기여: **개별 HI 단위 L0 선택 + 명시적 시나리오 식별**로
"적응적 최소연산"을 정량화(계산 HI 개수)하고 해석 가능(시나리오별 HI 표)하게 만든다.

---

## 9. 데이터 입력 모드 (`is_real_input`)

프레임워크는 두 가지 입력 모드를 지원한다. config 또는 CLI에서 `is_real_input` 플래그로 전환한다.

### `is_real_input = true` — 원시 신호 입력

```
입력 경로: _4_data_hi/clean/{MIT,HUST}/*.pkl
스키마:    [cell_id, cycle, segment_id, time_s, voltage_V, current_A, capacity_Ah]
```

각 행은 시계열 측정값 한 포인트다. 프레임워크가 **내부적으로** 세그먼트 분할 및 HI 계산을 수행한다.

```
방향(direction) 추출: sign(current_A)  < 0 → discharge / > 0 → charge
세그먼트 분할:        cumsum(|I|·dt) 구간 비율로 Low/Mid/High 분할  (수식, 초저비용)
HI 계산:              분할된 세그먼트 각각에 hi_schema.py 정의 함수 호출
```

### `is_real_input = false` — 사전 계산 HI 입력

```
입력 경로: _4_data_hi/seg/{MIT,HUST}/*.pkl
스키마:    [cell_id, cycle, segment_id, scen, capacity_Ah, HI_1, ..., HI_N]
```

HI가 이미 계산되어 있으므로 세그먼트 분할·HI 계산 단계를 건너뛴다.

> **⚠️ `scen` / `phase` 컬럼 사용 규칙**
>
> - `scen` : **정답지(ground truth)로만 사용** — Stage A CE 손실 계산 시 `level = |scen| - 1` 추출
> - `direction = sign(current_A)` : **모델 입력 허용** (전류 부호는 HI 계산 없이 즉시 알 수 있는 공짜 정보)
> - `scen` 또는 `phase` 자체를 모델 입력 피처로 넣으면 **치팅(leakage)** — 절대 금지

`scen` 열 규약 (`_4_data_hi/seg/` 정답지):

| scen | direction | level |
|------|-----------|-------|
| -3   | discharge | High  |
| -2   | discharge | Mid   |
| -1   | discharge | Low   |
| +1   | charge    | Low   |
| +2   | charge    | Mid   |
| +3   | charge    | High  |

---

## 10. 처리 파이프라인 상세

### 10-1. 전체 흐름

```
[입력]
  is_real_input=true  → _4_data_hi/clean/{MIT,HUST}/*.pkl
                        → (A) 세그먼트 분할  (B) HI 계산
  is_real_input=false → _4_data_hi/seg/{MIT,HUST}/*.pkl
                        → 그대로 사용 (HI 이미 완비)
        │
        ▼
[Stage A: 시나리오 분류기]  ── scenario_classification_HIs.json 캐시 참조
        │
        ▼
[Stage B: 용량 예측기]      ── scenario_regression_HIs.json 캐시 참조
        │
        ▼
[출력] capacity_Ah 예측 + 라우팅 통계
```

---

### 10-2. Stage A — 시나리오 분류기

#### JSON 캐시 존재 시 (`5_model/scenario_classification_HIs.json`)

파일을 읽어 기록된 probe HI 목록을 고정 마스크로 사용한다. **학습 없이 즉시 추론 진입.**

```json
{
  "probe_m": 5,
  "selected_his": ["stat_v_mean_dis_hi", "ica_peak1_v", "stat_v_range_dis_hi", ...]
}
```

#### JSON 캐시 없을 시

Stage A HI 선택 학습 페이즈를 실행한다:

1. Hard-Concrete L0 probe 게이트(`probe_log_alpha`, 65차원)를 학습
2. `probe_m_fixed` 설정에 따라 probe HI 수 결정:

| `probe_m_fixed` | 동작 |
|-----------------|------|
| `false` (기본)  | L0 페널티가 개수를 자동 결정 — λ0 조정으로 Pareto 탐색 |
| `true`          | 학습 후 `probe_m_count` 개(1~10)로 hard top-k 절삭 |

3. 선택된 HI 목록을 `5_model/scenario_classification_HIs.json`에 저장
4. 저장 후 위 "캐시 존재" 경로로 이어서 수행

**하이퍼파라미터 (config: `classifier` 섹션)**

```yaml
classifier:
  probe_m_fixed: false      # false = L0 자동 | true = 아래 개수로 강제
  probe_m_count: 5          # probe_m_fixed=true일 때만 유효 (선택: 1~10)
  json_path: "5_model/scenario_classification_HIs.json"
```

---

### 10-3. Stage B — 용량 예측기

#### JSON 캐시 존재 시 (`5_model/scenario_regression_HIs.json`)

파일을 읽어 6개 시나리오별 HI 목록을 고정 마스크로 사용한다. **학습 없이 즉시 추론 진입.**

```json
{
  "scen_k": 10,
  "scenarios": {
    "-3": ["stat_v_mean_dis_hi", "lfp_plateau_frac_dis_hi", ...],
    "-2": ["diff_dvdq_area_dis_mid", "stat_v_std_dis_mid", ...],
    "-1": [...],
    "+1": [...],
    "+2": [...],
    "+3": [...]
  }
}
```

#### JSON 캐시 없을 시

Stage B HI 선택 학습 페이즈를 실행한다:

1. 시나리오 조건부 게이트 로짓(`ScenGateLogits`, 6×65)을 Stage A와 함께 end-to-end 학습
2. `scen_k_fixed` 설정에 따라 시나리오별 HI 수 결정:

| `scen_k_fixed` | 동작 |
|----------------|------|
| `false` (기본) | L0 페널티가 시나리오별 개수를 자동 결정 |
| `true`         | 학습 후 각 시나리오별로 `scen_k_count` 개(5/10/15/20)로 hard top-k 절삭 |

3. 시나리오별 선택 HI 목록을 `5_model/scenario_regression_HIs.json`에 저장
4. 저장 후 위 "캐시 존재" 경로로 이어서 수행

**하이퍼파라미터 (config: `regression` 섹션)**

```yaml
regression:
  scen_k_fixed: false       # false = L0 자동 | true = 아래 개수로 강제
  scen_k_count: 10          # scen_k_fixed=true일 때만 유효 (선택: 5/10/15/20)
  json_path: "5_model/scenario_regression_HIs.json"
```

---

### 10-4. `scr.yaml` 전체 설계

```yaml
data:
  is_real_input: false                    # true = raw V/I/t | false = 사전계산 HI
  raw_data_dir: "_4_data_hi/clean"        # is_real_input=true 경로
  seg_data_dir: "_4_data_hi/seg"          # is_real_input=false 경로
  datasets: ["MIT", "HUST"]
  split_seed: 42
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  min_cycles_per_cell: 10
  io_workers: 8

classifier:
  probe_m_fixed: false      # false = L0 자동 | true = probe_m_count 강제
  probe_m_count: 5          # 유효범위: 1~10
  json_path: "5_model/scenario_classification_HIs.json"

regression:
  scen_k_fixed: false       # false = L0 자동 | true = scen_k_count 강제
  scen_k_count: 10          # 유효값: 5 / 10 / 15 / 20
  json_path: "5_model/scenario_regression_HIs.json"

model:
  d_probe: 64               # Stage A MLP hidden
  d_head: 128               # Capacity Head hidden
  dropout: 0.1

loss:
  lambda_scen: 0.5          # CE 가중치
  lambda_l0: 0.01           # L0 비용 가중치

training:
  epochs: 200
  batch_size: 512
  lr: 1.0e-3
  weight_decay: 1.0e-4
  warmup_epochs: 5
  early_stop_patience: 30
  grad_clip: 1.0
```

---

## 11. 결과 플롯 (`test.py` 추가 사항)

`test.py`는 기존 scatter plot 외에 대표 셀 용량 곡선을 추가로 출력한다.

### 기존 플롯 (유지)

- **Scatter**: x = 실측 `capacity_Ah`, y = 예측 `capacity_Ah` (세그먼트 단위)

### 추가 플롯 — 대표 셀 용량 열화 곡선

대표 셀 `b1c0` (MIT), `1-1` (HUST) 각각에 대해:

- **x축**: cycle 번호
- **y축**: capacity_Ah
- **실선**: 실측값
- **점선**: 모델 예측값 (세그먼트 예측값을 사이클 단위로 집계)
- 저장 경로:
  - `_5_data_model_scr/<run>/figures/capacity_curve_b1c0.png`
  - `_5_data_model_scr/<run>/figures/capacity_curve_1-1.png`

> 구현 노트: 대표 셀이 test split에 없을 경우, 해당 셀 pkl을 `_4_data_hi/seg/` 또는
> `_4_data_hi/clean/`에서 직접 로드해 **시각화 전용**으로 추론 실행.
> 이 데이터는 평가 지표(RMSE 등) 계산에서 제외한다.

세그먼트 → 사이클 집계 방법:

```python
# 한 사이클의 예측 용량 = 해당 사이클 내 전 세그먼트 예측값의 평균
cycle_pred = segment_df.groupby("cycle")["cap_pred"].mean()
cycle_real = segment_df.groupby("cycle")["capacity_Ah"].mean()
```

---

## 12. 구현 변경 이력

### 2026-07-06 — SCR 초기 구현 완료

블루프린트(§1~§11) 기반으로 `5_model/` 하위에 SCR 모듈 전체를 신규 작성.
DFR 관련 코드 전체 제거 — `5_model/`은 SCR 전용으로 정리됨.

#### 신규 파일

| 파일 | 내용 |
|------|------|
| `config/scr.yaml` | SCR 전용 하이퍼파라미터 (§10-4 스펙 반영) |
| `utils/hi_schema.py` | 65-HI 스키마, 카테고리 비용, SCEN_MAP, `get_hi_cols_for_seg()` |
| `datasets/segment_dataset.py` | 와이드 pkl reshape → 세그먼트 행, 셀분할(`split_cells`), `SegmentNormalizer`, `SegmentDataset`, `build_datasets()` |
| `models/hard_concrete.py` | Louizos 2018 Hard-Concrete L0 게이트 (β=2/3, γ=-0.1, ζ=1.1) |
| `models/scr_model.py` | `SCRModel`: Stage A (probe gate + MLP → 3-class level logits) + Stage B (6×N_HI 시나리오 조건부 게이트 + capacity head) |
| `training/scr_loss.py` | `SCRLoss`: MSE + λ\_scen·CE + λ\_l0·비용가중 L0 페널티 |
| `training/scr_trainer.py` | `SCRTrainer`: cosine LR warmup, early stopping, best checkpoint 저장 |
| `evaluation/scr_evaluator.py` | `SCREvaluator`: scatter plot + 대표셀 capacity curve (§11 스펙) |
| `train_scr.py` | 학습 진입점: JSON 캐시 확인 → 학습 → top-k 절삭(fixed 모드) → JSON 저장 |
| `test_scr.py` | 평가 진입점: 체크포인트 로드 → 3-split 메트릭 + 그래프 + 활성 HI 통계 |

#### 설계 결정 및 구현 세부사항

**데이터 로딩 (`segment_dataset.py`)**
- 네이티브 `_4_data_hi/seg/` 포맷 우선 시도, 없으면 기존 와이드 pkl(`_4_data_hi/{MIT,HUST}/*.pkl`) reshape 자동 적용
- reshape 시 cycle-level `capacity_Ah`를 각 세그먼트의 SOH 타깃으로 사용 (segment partial Ah 아님)
- `stat_q_abs` 완전 배제 → 실제 N_HI=65 런타임 검증 완료

**JSON 캐시 흐름**
- `scenario_classification_HIs.json` 또는 `scenario_regression_HIs.json` 존재 시 해당 gate 학습 스킵, 고정 마스크로 즉시 추론 진입
- `probe_m_fixed=true` 시: 학습 후 `gate_prob()` 기준 top-k hard 절삭 → JSON 저장
- `scen_k_fixed=true` 시: 각 시나리오별 top-k hard 절삭 → JSON 저장

**버그 수정 사항**
- `r2(y_true, y_pred)` 인수 순서 오류 수정: trainer/evaluator에서 `(pred, true)` → `(true, pred)`
- `tensor.expand(sel.sum(), -1)` 에서 0-dim 텐서 전달 오류 수정: `int(sel.sum().item())` 명시 변환

**검증**
- 10개 신규 파일 전체 `py_compile` 통과
- `SCRModel` forward/backward, `HardConcreteGate` train/eval, `SCRLoss` 계산 동작 확인 (N_HI=65 assert 포함)
