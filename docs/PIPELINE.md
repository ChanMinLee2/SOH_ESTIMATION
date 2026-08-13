# SOH_ESTIMATION — 프레임워크 개요

**작성 기준**: 2026-08-13 시점의 실제 코드(`common/scenario/`, `4_hi_analysis/hi_correlation.py`,
`5_model/`, `run_pipeline.py`). 이전 버전 문서는 전면 폐기하고 이 코드 상태만 근거로 다시 썼다.

---

## 1. 현재 프레임워크의 목적

LFP 배터리(MIT, HUST 두 공개 데이터셋)의 SOH(State of Health)를 **부분 충방전 곡선만으로**
추정하는 프레임워크다. 세 가지 설계 목표를 유지한다.

1. **부분 사이클 사용** — 완주하지 않은 세그먼트(사이클 전체가 아니라 그 일부 구간)만으로
   SOH를 추정한다. 실제 BMS가 전체 충방전 곡선을 항상 관측할 수 없다는 전제.
2. **시나리오별 최적화된 HI 서브셋 구성** — 충전/방전 × SOC 위치(존)마다 다른 hand-crafted
   HI(Health Indicator) 조합이 최적일 수 있다는 전제 아래, 이를 데이터로부터 자동으로 찾는다.
3. **미지 세그먼트에 대한 시나리오 추정** — 실배포 시 "이 세그먼트가 어느 시나리오(존)인지"를
   라벨 없이 스스로 판단해야 한다(hard/soft 라우팅).

두 데이터셋(MIT 123셀·HUST 77셀, 정격용량 1.1Ah/1.2Ah)을 풀링해 학습하고, cell 단위로
train:val:test = 6:2:2(`split_seed` 고정)로 나눠 평가한다.

---

## 2. 파이프라인 구조 — 단계별 수행사항과 결과물

`run_pipeline.py`가 Step 1~9를 순서대로 실행한다(`python run_pipeline.py <from> --to-step <to>`).

| Step | 스크립트 | 수행 내용 | 결과물 |
|---|---|---|---|
| 1 | `1_convert/convert_unified.py` | MIT/HUST 원본 포맷 → 통일 스키마 변환 | `_0_data_raw/` |
| 2 | `2_preprocess/preprocess.py` | 이상 사이클 제거, 필터7(형태 이상치) 적용 | `_1_data_clean/` |
| 3 | `3_integrity/check_integrity.py` | 데이터 무결성 검사 | 로그(리포트) |
| 4 | `4_hi_analysis/hi_correlation.py` | **세그멘테이션**(`--seg-axis`로 선택) + **HI 계산**(66개, `stat`/`diff`/`lfp`/`morph` 4카테고리) | `_4_data_hi/{axis}/{cycle,seg}/*.pkl`, `scenario_spec.json` |
| 5 | `4_hi_analysis/hi_segment_viz.py` | HI 세그먼트 시각화(선택) | 플롯 |
| 6 | `5_model/train_scr.py --phase 1` | **L0 게이트 학습** — probe/시나리오 게이트가 HI 랭킹을 학습 | `_5_data_model_scr/{run}/gates/*.json`, `checkpoints/best.pt` |
| 7 | `5_model/train_classifier.py` | **시나리오 분류기 학습** — "이 세그먼트가 어느 존인가" 예측(hard 라우팅용) | `classifier/clf_best.pt` |
| 8 | `5_model/train_scr.py --phase 2` | 게이트·분류기 **고정** 후 회귀 헤드만 재학습 | `_5_data_model_scr/{run}/checkpoints/best.pt` |
| 9 | `5_model/test_scr.py` | oracle/hard/soft 3가지 라우팅으로 평가 | `metrics/metrics.json`, `figures/`, `routing/`, `predictions/` |

**세그멘테이션 축**(`--seg-axis`, `common/scenario/__init__.py`의 `REGISTRY`): `qfrac`, `protocol`,
`vwindow`, `rcs`/`random`(같은 클래스, 저장 경로만 분리), `cluster`, `q_frac_wide`, `q_frac_ref`,
`vqslope`, `q_abs`, `full_cycle`, `test_rs`. 각 축은 `ScenarioSpec`(시나리오 개수·라우팅 테이블·
분류기 필요 여부)을 생성해 Step 4 이후 전 단계가 이를 그대로 따른다 — 축을 바꿔도 Step 6~9
코드는 대부분 수정 없이 동작한다("그 외 축" 폴백 규칙, `train_scr.py`/`train_classifier.py`).

**HI 66개 구성**: `stat`(통계, 20) + `diff`(미분/dV·dQ 기반, 20) + `lfp`(LFP 특징, 20) +
`morph`(BOL 대비 DTW/Fréchet, 6). `stat_q_abs`/`stat_energy_seg` 2개는 라벨 누수 위험이 있어
`SOH_EXCLUDE_STAT_LEAK=1` 환경변수로 제외 가능(`N_HI`가 66→64로 자동 조정, `5_model/utils/hi_schema.py`).

---

## 3. 모델 구조와 하이퍼파라미터의 의미

`5_model/models/scr_model.py`의 `SCRModel`은 두 단계 게이트 + 회귀 헤드로 구성된다.

```
x_hi(N_HI) ──► Stage A: charge/discharge probe gate(HardConcreteGate ×2)  ── m개 HI 선택
           ──► Stage B: 시나리오별 gate(HardConcreteGate × n_scenarios)   ── 시나리오당 k개 HI 선택
                                    │
                    [probe_x | scen_x (| raw CNN 임베딩 3D) | direction | cap_init]
                                    ▼
                              cap_head(회귀 헤드) ──► SOH 예측
```

- **`charge_probe_m` / `discharge_probe_m`**(기본 10) — Stage A 게이트가 최종적으로 남기는
  HI 개수(방향별). "이 세그먼트가 대략 어떤 상태인지"를 보는 저비용 사전 신호.
- **`scen_k_count`**(=`--scen-k`) — Stage B, **시나리오별** 게이트가 남기는 HI 개수. `N_HI`보다
  크면 사실상 no-op(전체 HI 사용, 파이썬 리스트 슬라이싱 특성).
- **`HardConcreteGate`**(`models/hard_concrete.py`, Louizos et al. 2018 L0 정규화) — 학습 중
  (`self.training=True`)엔 매 forward마다 새로 샘플링되는 확률적 마스크, 평가 시엔 결정론적
  0/1 마스크(`_hard_infer`)로 전환된다.
- **`regression_model`**(`mlp`/`transformer`/`i_transformer`/`resnet_tab`/`ft_transformer`/
  `raw_mlp`) — `cap_head`의 아키텍처. `raw_mlp`는 HI/게이트를 완전히 우회하고 raw V/I/t만
  쓰는 별도 베이스라인(§5-1 참고).
- **`mlp_hidden_dims`** — `regression_model="mlp"`일 때 `cap_head`의 은닉층 크기 리스트.
- **`d_probe`/`d_head`** — Stage A MLP·capacity head의 은닉 차원.
- **`with_raw_cnn`/`with_raw_flat`** — 회귀 헤드에 raw V/I/t CNN 임베딩(3D, `[h_scen,
  h_intensity, h_soh]`)을 추가 융합할지 여부(Phase 2 전용 옵션).
- **`lambda_scen`** — 시나리오 분류 보조손실(CE) 가중치(Phase 1 dual-objective 모드에서만).
- **`lambda_l0` / `lambda_l0_auto`** — L0 희소성 페널티 가중치. `auto=true`면
  `λ = clip(0.01·√((10/m̄)·(10/k)), 1e-4, 0.5)`로 자동 계산(`train_scr.py:_compute_lambda_l0`,
  `m̄=(charge_probe_m+discharge_probe_m)/2`) — m·k가 작을수록(=예산이 빠듯할수록) 더 세게
  압력을 준다.
- **`lambda_l0_schedule`**(`delayed_warmup`) / **`lambda_l0_warmup_epochs`**(50) /
  **`lambda_l0_ramp_epochs`**(100) — §4에서 상술.
- **`lr` / `warmup_epochs`(training) / `scheduler`(cosine)** — 옵티마이저 LR 스케줄, §4에서 상술.
- **`weight_decay`** — AdamW 가중치 감쇠(L2 정규화).
- **`early_stop_patience`** — val RMSE가 이 에폭 수만큼 갱신 안 되면 학습 종료.
- **`--seed` / `--split-seed`** — 각각 모델 초기화(torch/numpy/random RNG)와 train/val/test
  셀 분할 시드. 둘 다 독립적으로 CLI에서 오버라이드 가능(2026-08-12부터 — 그 전엔 `--seed`가
  실제로 아무 RNG에도 전달되지 않는 사실상 no-op이었다, §5-2 참고).

---

## 4. 학습 설계

**3단계 분리 학습**(Step 6→7→8, `run_pipeline.py`의 `STEPS`):

1. **Phase 1**(`train_scr.py --phase 1`): `SCRModel`을 마스크 없이(=게이트 활성) 생성,
   `regression_model`을 **강제로 `"mlp"`**로 고정한 프록시 헤드와 함께 `MSE(+CE)+L0` 손실로
   학습한다. 결과로 시나리오별 HI 랭킹(`gates/regression_HIs.json`: `seg_{s}_names`가 중요도
   내림차순)과 probe 랭킹(`classification_HIs.json`)을 저장한다.
2. **분류기**(`train_classifier.py`): Phase 1 run에 붙여 "이 세그먼트가 어느 존인가"를 예측하는
   `CNNProbeClassifier`/`MLPProbeClassifier`를 별도 CE 손실로 학습한다.
3. **Phase 2**(`train_scr.py --phase 2`): Phase 1의 게이트를 `ranked[:k]`로 잘라 **고정 마스크**로
   변환하고, `regression_model`을 yaml이 지정한 값 그대로(Phase 1과 다를 수 있음) 써서 회귀
   헤드를 **처음부터 새로 학습**한다. `lambda_l0=0`(게이트가 이미 고정이라 페널티 불필요).

**Phase 1 한 에폭의 스케줄**(`5_model/training/scr_trainer.py`):

- **LR**: 처음 `warmup_epochs`(기본 10)에폭 동안 0→base_lr 선형 증가, 이후
  `CosineAnnealingLR(T_max=epochs-warmup_epochs, eta_min=1e-6)`로 감쇠. 실제로는 early stopping이
  이 T_max보다 훨씬 일찍 걸려 코사인 감쇠가 끝까지 완주하는 경우는 드물다.
- **L0 lambda**(`delayed_warmup`): 에폭 0~49는 `λ_l0=0`(희소성 압력 없음, 전체 HI로 회귀만
  학습), 에폭 50~149는 target까지 **선형 증가**, 이후 target 고정.
- **Early stopping**: val RMSE 기준, 위 두 스케줄과 무관하게 독립적으로 patience를 센다 —
  L0 압력이 본격화되기 전(에폭 50 이전)에 val RMSE가 우연히 최고점을 찍고 그 뒤로 patience
  동안 갱신되지 않으면, **L0 정규화가 사실상 한 번도 적용되지 않은 상태의 체크포인트가
  "best"로 저장**될 수 있다(§5-2에서 실측 비율 제시).

**평가**(`test_scr.py`, Step 9): 같은 체크포인트를 세 가지 라우팅으로 평가한다 — `oracle`(정답
시나리오 라벨 사용, 상한선), `hard`(분류기 argmax), `soft`(분류기 확률 가중 평균). 분류기 체크
포인트가 없으면(`--skip-classifier` 등) 자동으로 `oracle`만 평가한다.

---

## 5. Contribution & Limitation

### 5-1. Contribution

- **해석 가능한 스파스 피처 선택**: hand-crafted, 물리적으로 이름 붙은 HI 66개에서
  L0 정규화로 시나리오별 서브셋을 자동 선택 — "어떤 HI를 왜 썼는지"를 raw-신호 기반 모델보다
  직접 제시할 수 있다.
- **부분 관측 능력을 직접 검증하는 대조 축**을 이미 갖추고 있다 — `full_cycle`(구간 분할 없는
  전체 곡선, 정보 손실 없는 상한 참조용)과 `random`/`rcs`(시나리오 타입 없이 균일 샘플링,
  구조화의 순수 기여도 측정용)이 코드 변경 없이 바로 실행 가능하다(§2의 "그 외 축" 폴백).
- **다축(multi-axis) 세그멘테이션 프레임워크**: 세그먼트 경계 정의 방식(`qfrac`/`q_frac_wide`/
  `q_frac_ref`/`vwindow`/`rcs`/`q_abs`/`vqslope`/`full_cycle` 등)을 손쉽게 교체·비교할 수 있어
  설계 선택 각각의 기여도를 어블레이션으로 분리할 수 있다.
- **부분 관측이 정확도를 반드시 희생하지 않는다는 실측 근거** — 통제된 비교에서 부분 관측
  baseline이 전체 관측(`full_cycle`)보다 나은 사례가 관측됐다(원인은 미확정, 5-2 참고).

### 5-2. Limitation

- **Phase 1/Phase 2 분리가 정규화 손실을 유발할 수 있다**: Phase 1의 확률적 게이트 마스킹이
  주는 암묵적 정규화가 Phase 2(고정 마스크)엔 없다 — 데이터가 적은 조건(`full_cycle`)에서
  Phase 2가 대용량 헤드로 수 에폭 만에 과적합한 사례를 실측했다. 모델 축소·`weight_decay` 인상
  등으로 완화 가능하나 근본 구조는 그대로다.
- **Phase 1의 "best" 체크포인트가 L0 압력이 적용되기 전 상태로 저장되는 경우가 많다**: 12개
  실측 run 중 8개(67%)가 `lambda_l0=0`인 시점(에폭 49~51)의 체크포인트를 최종 채택했다 —
  `q_frac_ref` 축은 6/6(100%) 전부 이 경우였다. 즉 다운스트림에 저장되는 HI 랭킹 중 상당수가
  "L0로 학습된 스파스 구조"가 아니라 "정규화 압력이 걸리기 전 gradient 신호"를 반영한다.
- **회귀 아키텍처 스윕 시 게이트-헤드 불일치**: Phase 1은 `regression_model`을 항상 `"mlp"`로
  고정하는데 Phase 2는 yaml이 지정한 값(`transformer`/`resnet_tab` 등)을 쓴다 — 랭킹을 만든
  아키텍처와 실제로 그 랭킹을 소비하는 아키텍처가 다를 수 있다(`regression_model="mlp"`인
  대다수 실험에는 해당 없음, 아키텍처 스윕 실험에만 국한).
- **시나리오별 HI 랭킹이 상위 순위 밖에서 불안정하다**: 같은 조건을 독립 seed로 재학습하면
  상위 5개 HI는 대체로 재현되지만, 6~15위권은 재현마다 상당수(최대 1/3) 바뀐다.
- **도메인 정렬 없이 MIT+HUST를 풀링한다**: 두 데이터셋 간 분포 차이를 보정하는 항(MMD 등)이
  없다.
- **`morph` HI(6개)는 그 셀의 BOL(신품) 곡선이 저장돼 있어야 계산된다** — 배포 시 이력 없는
  셀은 이 6개가 통째로 NaN이 될 수 있다.
- **대부분의 결과가 아직 single-seed다**: 모델 초기화 시드가 실제로 RNG에 반영되기 시작한 건
  최근이며, 복수 seed로 재확인된 비교는 일부(`random` vs `q_frac_ref`, k=5/15)뿐이다 — 나머지
  축·하이퍼파라미터 스윕 결과는 seed 1개에 기반해 신뢰구간이 없다.
- **`full_cycle`(이론적 상한)이 부분 관측 baseline을 이기지 못하는 조건이 있다**: scen_k를
  baseline과 동일하게 맞춘 지점에서 `full_cycle`이 오히려 `q_frac_ref`보다 낮은 R²를 보였다 —
  가설(세그먼트 수 차이, HI 신호 희석)은 있으나 검증되지 않았다.
