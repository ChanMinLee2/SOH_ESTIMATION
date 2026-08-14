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

---

## 6. 실행 스크립트 레퍼런스

> **2026-08-14 CLI 정리**: 아래는 정리(`docs/params.md` — 실사용 이력 없는 디버그·상수형 옵션
> 53개 제거) 이후의 현재 CLI다. 옵션 수가 예전보다 줄어서 이 문서에 나온 것이 곧 전체 목록이다.

### 6-0. 처음 쓰는 사람을 위해 — `run_pipeline.py` 파라미터 전달법

**이 프로젝트 구조를 몰라도 알아야 할 것은 딱 세 가지다.**

1. **`run_pipeline.py`는 실행기일 뿐, 실제 작업은 안 한다.** Step 1~9는 각각 독립된 파이썬
   스크립트(`1_convert/convert_unified.py`, `2_preprocess/preprocess.py`, ... `5_model/test_scr.py`)이고,
   `run_pipeline.py`는 이걸 순서대로 `subprocess`로 실행하는 오케스트레이터다. 그래서
   `run_pipeline.py --help`에 안 나오는 옵션이 하위 스크립트엔 있을 수 있다(예: `test_scr.py`의
   `--classifier-ckpt`) — 그럴 땐 `run_pipeline.py`를 거치지 말고 그 스크립트를 직접 실행하면 된다
   (§6-9 참고).
2. **`run_pipeline.py`에 준 옵션은 "관련 있는 Step에만" 자동으로 골라 전달된다.** 예를 들어
   `--seg-axis`는 세그멘테이션이 실제로 일어나는 Step 4/5/6/8에만 전달되고, 분류기 학습(Step 7)에는
   안 간다(분류기는 자기가 붙는 Phase 1 run의 `scenario_spec.json`에서 축을 읽어오기 때문).
   아래 표가 "이 옵션을 주면 어느 Step에 자동으로 흘러가는지"를 정리한 것이다 — 이 매핑이
   `run_pipeline.py`의 `main()` 안에 그대로 코드로 있다(주석 `# ── ... 주입 ──` 블록들).

   | 옵션 | 자동으로 전달되는 Step |
   |---|---|
   | `--seg-axis`, `--axis-config`(+단축 인자) | 4, 5, 6, 8 |
   | `--exclude-cv` | 4, 6, 7, 8 |
   | `--skip-shape` | 2, 4, 6, 7, 8 |
   | `--charge-m`, `--discharge-m`, `--scen-k`, `--seed`, `--split-seed` | 6, 8 |
   | `--with-raw-cnn` | 8만 (+ Step 7에서 학습된 분류기 체크포인트를 자동으로 찾아 같이 넘김) |
   | `--workers` | 1, 2, 3, 4, 5(데이터 처리 스텝만 — 6~9는 GPU 학습이라 워커 개념이 없음) |
   | `--model-config` | 6, 7, 8, 9(`--config`라는 이름으로 그대로 전달) |

3. **CLI로 안 주면 yaml 값이 그대로 쓰인다.** `run_pipeline.py`의 옵션은 전부 기본값이 `None`이고,
   `None`이면 그 옵션을 하위 스크립트에 아예 넘기지 않는다 — 그러면 각 스크립트가 `--model-config`로
   지정한 yaml(§4의 `main.yaml`/`fixed.yaml` 개편 이전엔 `5_model/config/*.yaml`)의 값을 그대로 쓴다.
   즉 **"이번 실험에서 바꾸고 싶은 값만" CLI로 주면 되고, 나머지는 yaml에 맡기면 된다** — 모든
   파라미터를 CLI로 나열할 필요가 없다.

**처음 실행해볼 때 최소 커맨드**(전체 파이프라인, 옵션 없이 yaml 기본값 그대로):
```powershell
python run_pipeline.py
```

**실험 하나 돌릴 때 흔한 패턴**(학습+평가만, 축·k·시드만 바꿈 — 나머지는 yaml 그대로):
```powershell
$env:SOH_EXCLUDE_STAT_LEAK="1"
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_S.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --scen-k 25 --seed 42 --split-seed 42 --workers 40
```

**PowerShell 주의**: `--axis-config`에 큰따옴표가 포함된 JSON을 직접 줄 때는
`'{\"key\": ...}'`처럼 전체를 작은따옴표로 감싸고 내부 큰따옴표를 `\"`로 이스케이프해야 한다 —
안 그러면 네이티브 실행파일 인자 전달 과정에서 따옴표가 삭제된다(PowerShell 5.1 특성). 헷갈리면
아래 §6-1 표의 `--n1`/`--n2`/`--n-samples` 같은 단축 인자를 대신 쓰면 이 문제 자체가 없다.

### 6-1. `run_pipeline.py` — Step 1~9 오케스트레이터

```
python run_pipeline.py [FROM_STEP] [--to-step N] [옵션들]
```

`FROM_STEP`(위치 인자, 기본 1)부터 `--to-step`(기본: 마지막 Step 9)까지 §2의 Step을 순서대로
실행한다.

**예시**:
```powershell
# 전체 파이프라인(Step1~9)
python run_pipeline.py

# 학습+평가만(Step6~9), q_frac_ref 축, k=25, 시드 42
$env:SOH_EXCLUDE_STAT_LEAK="1"
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_S.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --scen-k 25 --seed 42 --split-seed 42 --workers 40

# Phase1→Phase2만(평가 제외), 분류기 스킵
python run_pipeline.py 6 --to-step 8 --skip-classifier
```

| 플래그 | 타입 | 기본값 | 의미 | 예시 |
|---|---|---|---|---|
| `FROM_STEP`(위치인자) | int | 1 | 시작 Step 번호(1~9) | `python run_pipeline.py 6` (Step6부터) |
| `--to-step` | int | 9(끝까지) | 종료 Step 번호(포함) | `--to-step 8` (평가 Step9 제외하고 학습까지만) |
| `--skip-classifier` | flag | off | Step 7 건너뜀 — `test_scr.py`가 분류기 체크포인트 없으면 자동으로 oracle만 평가 | hard/soft 라우팅 비교가 필요 없을 때 학습 시간 절약 |
| `--workers` | int | `min(8, cpu_count)` | Step 1~5(데이터) 병렬 프로세스 수 | `--workers 40` |
| `--model-config` | str | `5_model/config/scr.yaml` | 학습/평가 yaml 경로 | `--model-config 5_model/config/exp_qfw_mlp_S.yaml` |
| `--seg-axis` | str | yaml 값 | 세그멘테이션 축(§2 REGISTRY 참고) | `--seg-axis q_frac_ref` |
| `--axis-config` | JSON str | `{}` | 축 파라미터, "맨" dict(`{"n1":0.4,...}`이지 `{"q_frac_wide":{...}}`가 아님) | `--axis-config '{\"n_samples\": 12, \"window\": 0.2}'` |
| `--n1` / `--n2` / `--n-samples` | float/float/int | None | `q_frac_wide`/`q_frac_ref`/`vqslope` 구간 크기·길이·개수(`--axis-config` 대체, PowerShell 호환) | `--n1 0.35 --n2 0.20` |
| `--ref-lag` / `--noise-amp` / `--noise-mode` / `--noise-period` | int/float/str/float | None | `q_frac_ref` 전용: 레퍼런스 지연·노이즈 진폭·드리프트 방식(ou\|sine)·특성시간 | `--ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200` |
| `--min-pts` | int | None | 세그먼트 최소 포인트 수(기본과 다르면 `_minptsN` 경로로 분리 저장) | `--min-pts 10` |
| `--exclude-cv` | flag | off | 충전 세그먼트에서 CC→CV 전환 이후 구간 제외 | `random`/`q_frac_ref` 등 축 간 CV 포함 여부를 통일할 때 |
| `--skip-shape` | flag | off | 전처리 필터7(형상 이상치 제거) 비활성화 | 필터7 영향도를 어블레이션할 때 |
| `--charge-m` / `--discharge-m` | int | yaml 값 | Stage A probe 게이트 예산(방향별) | `--charge-m 10 --discharge-m 10` |
| `--scen-k` | int | yaml 값 | Stage B 시나리오 게이트 예산(k) | `--scen-k 25` |
| `--seed` | int | yaml 값 | 모델 초기화 RNG 시드(torch/numpy/random) | `--seed 42` |
| `--split-seed` | int | yaml 값(42) | train/val/test 셀 분할 시드 | `--split-seed 0` |
| `--with-raw-cnn` | flag | off | Phase 2(Step 8)에 raw CNN 임베딩 융합, Step 7 분류기의 RawCNN을 자동으로 얼려 재사용 | Step 6→9를 한 번에 돌릴 때만 자동 연결 |

**과거 실행에서 특정 run을 다시 잇고 싶을 때(자동탐색이 안 통하는 경우)**: `run_pipeline.py`
자체엔 `--gates-from`/`--checkpoint` 같은 수동 오버라이드가 없다(자동탐색만 함 — §6-7/§6-9의
`--gates-from`/`--checkpoint`는 하위 스크립트를 **직접** 실행할 때만 쓰는 옵션이다). 예전 Phase 1
run을 다시 붙이거나 특정 체크포인트만 평가하고 싶으면 `run_pipeline.py`를 거치지 말고
`5_model/train_scr.py --phase 2 --gates-from <run_dir>` 또는 `5_model/test_scr.py --checkpoint <path>`를
직접 실행한다.

### 6-2. Step 1 — `1_convert/convert_unified.py`

```powershell
python 1_convert/convert_unified.py --dataset all --workers 3
```

| 플래그 | 타입 | 기본값 | 의미 | 예시 |
|---|---|---|---|---|
| `--dataset` | choice | `all` | 변환 대상: `mit`\|`hust`\|`all` | `--dataset mit` |
| `--output-root` | str | `OUTPUT_ROOT` | 출력 루트 경로 | 기본값 그대로 두는 게 보통 |
| `--workers` | int | 3 | 병렬 프로세스 수 | `--workers 8` |
| `--no-cache` | flag | off | 캐시 무시 — `_0_data_raw/`가 있어도 원본부터 재변환 | 원본 파일이 바뀌었을 때 |

### 6-3. Step 2 — `2_preprocess/preprocess.py`

```powershell
python 2_preprocess/preprocess.py --dataset all --workers 4
```

7단계 이상치 필터(필터1~7) 중 필터4~7의 임계값은 튜닝이 끝난 뒤 CLI에서 바꾼 이력이 없어
코드 상수로 승격했다(`docs/params.md` §CLI 정리 참고). 남은 CLI는 아래 3개뿐이다.

| 플래그 | 타입 | 기본값 | 의미 | 예시 |
|---|---|---|---|---|
| `--dataset` | choice | `all` | 처리할 데이터셋: `mit`\|`hust`\|`all` | `--dataset hust` |
| `--skip-shape` | flag | off | 필터7(형상 이상치) 완전 비활성화 — `_4_data_hi/clean_noshape/`에 별도 저장 | 필터7 있는/없는 데이터를 나란히 비교할 때 |
| `--workers` | int | `min(4, cpu_count)` | 병렬 프로세스 수 | `--workers 8` |

### 6-4. Step 3 — `3_integrity/check_integrity.py`

```powershell
python 3_integrity/check_integrity.py --workers 4
```

| 플래그 | 타입 | 기본값 | 의미 | 예시 |
|---|---|---|---|---|
| `--workers` | int | `min(4, cpu_count)` | 병렬 프로세스 수 | `--workers 8` |

옵션이 이거 하나뿐이다 — 데이터 무결성 검사는 조정할 임계값이 따로 없다.

### 6-5. Step 4 — `4_hi_analysis/hi_correlation.py`

```powershell
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --force --workers 40
```

세그멘테이션 + HI 66종 계산을 실제로 수행하는 스크립트. 축 관련 옵션은 `run_pipeline.py`와
이름·의미가 동일하다(§6-1 참고, 아래 표에선 축 옵션은 요약만).

| 플래그 | 타입 | 기본값 | 의미 | 예시 |
|---|---|---|---|---|
| `--workers` | int | `CPU수-2` | 병렬 프로세스 수 | `--workers 40` |
| `--force` | flag | off | 캐시 무시하고 HI 재추출 | 코드나 축 파라미터를 바꾼 뒤 재추출할 때(`run_pipeline.py`는 Step4에 항상 이 플래그를 자동으로 붙인다) |
| `--dataset` | str | `MIT` | 통계 스캔 대상: `MIT`\|`HUST` | `--dataset HUST` |
| `--seg-axis` | str | `qfrac` | 세그멘테이션 축 | `--seg-axis q_frac_ref` |
| `--axis-config` | JSON str | `{}` | 축 파라미터(맨 dict) | `--axis-config '{\"n1\": 0.35, \"n2\": 0.2}'` |
| `--n1`/`--n2`/`--n-samples` | — | None | 축 파라미터 단축 인자(§6-1과 동일) | `--n1 0.35 --n2 0.20` |
| `--ref-lag`/`--noise-amp`/`--noise-mode`/`--noise-period` | — | None | `q_frac_ref` 전용 단축 인자(§6-1과 동일) | `--ref-lag 0 --noise-amp 0.03` |
| `--min-pts` | int | None | 세그먼트 최소 포인트 수 | `--min-pts 10` |
| `--exclude-cv` | flag | off | CC→CV 전환 이후 구간 제외, `_ccOnly` 경로에 저장 | 축 간 CV 포함 여부 통일 |
| `--skip-shape` | flag | off | `clean_noshape/`(Step2 `--skip-shape` 결과)를 입력으로 사용 | 필터7 영향도 어블레이션 |

### 6-6. Step 5 — `4_hi_analysis/hi_segment_viz.py`

```powershell
python 4_hi_analysis/hi_segment_viz.py --seg-axis q_frac_ref --axis-config '{\"n1\": 0.35, \"n2\": 0.2}'
```

세그먼트 분할·HI 추이 시각화 전용 — 이 Step의 산출물(플롯)은 Step 6 이후 어디서도 다시 읽지
않는다(순수 사람이 보는 용도). 그래서 옵션도 축 지정 + 캐시 제어뿐이다.

| 플래그 | 타입 | 기본값 | 의미 | 예시 |
|---|---|---|---|---|
| `--workers` | int | 4 | HI 추출 병렬 워커 수 | `--workers 8` |
| `--force` | flag | off | 캐시 무시하고 HI 재추출 | Step4를 새로 돌린 직후 |
| `--seg-axis` | str | `qfrac` | 세그멘테이션 축 | `--seg-axis q_frac_ref` |
| `--axis-config` | JSON str | `{}` | 축 파라미터(맨 dict) | `--axis-config '{\"n1\": 0.35, \"n2\": 0.2}'` |

### 6-7. Step 6 & 8 — `5_model/train_scr.py`

```powershell
python 5_model/train_scr.py --phase 1 --config 5_model/config/scr.yaml
python 5_model/train_scr.py --phase 2 --gates-from _5_data_model_scr/0708_1533
```

Phase 1(Step6)/Phase 2(Step8)를 둘 다 이 스크립트 하나가 담당하며 `--phase`로 구분한다.
`run_pipeline.py`로 Step6→8을 이어 돌리면 `--gates-from`은 자동으로 채워지므로 직접 지정할
필요가 거의 없다 — **아래 표는 이 스크립트를 단독 실행할 때** 쓰는 전체 옵션이다.

| 플래그 | 타입 | 기본값 | 의미 | 예시 |
|---|---|---|---|---|
| `--config` | str | `5_model/config/scr.yaml` | 모델 설정 yaml | `--config 5_model/config/exp_qfw_mlp_S.yaml` |
| `--phase` | int | None | 1(Gate 학습) \| 2(분류·회귀 정밀 학습) | `--phase 1` |
| `--charge-m` / `--discharge-m` | int | yaml 값 | Stage A probe 게이트 예산(방향별) | `--charge-m 10 --discharge-m 10` |
| `--scen-k` | int | yaml 값 | Stage B 시나리오 게이트 예산(k) | `--scen-k 25` |
| `--gates-from` | str | yaml `data.gates_from` | Phase 2가 쓸 Phase 1 run 폴더(gates JSON 자동 탐색) | `--gates-from _5_data_model_scr/0708_1533` |
| `--device` | str | `auto` | 연산 장치 | `--device cuda:0` |
| `--seg-axis` | str | yaml `scenario.axis` | 세그멘테이션 축 | `--seg-axis q_frac_ref` |
| `--axis-config` | JSON str | yaml 값 | 축 파라미터(맨 dict) | `--axis-config '{\"n1\": 0.35}'` |
| `--seed` | int | yaml `training.seed` | 모델 초기화 RNG 시드 | `--seed 42` |
| `--split-seed` | int | yaml `data.split_seed` | train/val/test 셀 분할 시드 | `--split-seed 0` |
| `--exclude-cv` | flag | off | `_ccOnly` 데이터 경로 사용 | Step4를 `--exclude-cv`로 돌렸을 때 짝을 맞춤 |
| `--skip-shape` | flag | off | `_noshape` 데이터 경로 사용 | Step2/4를 `--skip-shape`로 돌렸을 때 짝을 맞춤 |
| `--with-raw-cnn` | flag | off | Phase 2 전용, raw CNN 임베딩 융합 | REGRESSION_UPGRADE.md §5/§8 실험 |
| `--raw-cnn-pretrained-from` | str | None | `--with-raw-cnn`과 함께: 분류기 체크포인트 경로(RawCNN 재사용) | `--raw-cnn-pretrained-from <run>/classifier/clf_best.pt` |

### 6-8. Step 7 — `5_model/train_classifier.py`

```powershell
python 5_model/train_classifier.py --run-dir _5_data_model_scr/0716_1200_p2_mlp_prot
```

| 플래그 | 타입 | 기본값 | 의미 | 예시 |
|---|---|---|---|---|
| `--config` | str | `5_model/config/scr.yaml` | 모델 설정 yaml | — |
| `--run-dir` | str | 최신 폴더 자동탐색 | Phase 1 run 폴더 — 분류기를 `{run_dir}/classifier/`에 저장 | `--run-dir _5_data_model_scr/0716_1200_p2_mlp_prot` |
| `--device` | str | `auto` | 연산 장치 | `--device cuda:0` |
| `--exclude-cv` | flag | off | `_ccOnly` 데이터 경로 사용 | Step4/6과 짝을 맞춤 |
| `--skip-shape` | flag | off | `_noshape` 데이터 경로 사용 | Step2/4/6과 짝을 맞춤 |

### 6-9. Step 9 — `5_model/test_scr.py`

```powershell
python 5_model/test_scr.py --checkpoint _5_data_model_scr/0811_1612_p2_mlp_full/checkpoints/best.pt
```

| 플래그 | 타입 | 기본값 | 의미 | 예시 |
|---|---|---|---|---|
| `--config` | str | `5_model/config/scr.yaml` | 모델 설정 yaml | — |
| `--checkpoint` | str | None | 평가 체크포인트 경로 | `--checkpoint <run>/checkpoints/best.pt` |
| `--classifier-ckpt` | str | `run_dir/classifier/clf_best.pt` 자동 탐색 | 시나리오 분류기 체크포인트(hard/soft 라우팅용) | 다른 run의 분류기를 재사용하고 싶을 때 |
| `--device` | str | `auto` | 연산 장치 | `--device cuda:0` |
| `--charge-m` / `--discharge-m` | int | yaml 값 | probe m 오버라이드 — 체크포인트 학습 당시 값과 일치해야 함 | `--charge-m 10 --discharge-m 10` |
| `--scen-k` | int | yaml 값 | scen k 오버라이드 — 체크포인트 학습 당시 값과 일치해야 함 | `--scen-k 25` |

### 6-10. (부록) `5_model/visualize_results.py` — 여러 run 비교 시각화("result_comparison")

```
python 5_model/visualize_results.py --runs <run_dir> <run_dir> [... 2개 이상] [옵션들]
```

Step 8(Phase 2) 출력 run 폴더 2개 이상을 받아 `metrics.json`/`predictions/test_predictions.csv`/
`routing/routing_table.csv`/`checkpoints/*.pt`를 읽고, 하나의 PNG로 다축 비교한다. 파일명은
`result_comparison.py`가 아니라 `visualize_results.py`지만 산출물 폴더/파일명이
`result_comparison`이라 그렇게 불린다.

**결과 저장 위치**: `_5_data_model_scr/comparison/<MMDD_HHMM>_<title 또는 result_comparison>/`
— `result_comparison.png`, `summary.json`.

**PNG 구성(GridSpec)**:
| 행 | 내용 |
|---|---|
| 1~3 | RMSE/MAE/MAPE — hard 라우팅 기준, Overall + 시나리오별 |
| 4 | 스칼라 지표 7종(R², 분류정확도, 인퍼런스 시간, 파라미터 수, oracle→hard 저하율, 평균 HI 비용, random-seg RMSE) |
| 5 | 선정 HI 자카드 유사도 히트맵(이진, top-k 컷 후) — 시나리오 간 HI 선택이 얼마나 다른지 |
| 6 | 확률 가중 Jaccard(Ruzicka) 히트맵 — top-k로 자르기 전 원본 게이트 확률 기준 |
| 7 | (`--with-jacobian` 지정 시) 선정 HI의 gradient 코사인 유사도 히트맵 |

같은 시나리오 축을 공유하는 run들끼리는 시나리오별 서브플롯까지 다 나오고, 축이 다르면
자동으로 "축 간 비교 모드"(Overall만, 행 5~7 생략)로 전환된다.

**예시**:
```powershell
python 5_model/visualize_results.py --runs "_5_data_model_scr/0811_1612_p2_mlp_full_noleak" "_5_data_model_scr/0811_1628_p2_mlp_full_noleak" --labels k15 k25 --title full_scenk_sweep
```

| 플래그 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--runs` | str 리스트 | (필수) | 비교할 run 디렉터리 경로 2개 이상 |
| `--labels` | str 리스트 | 폴더명 | run별 표시 이름(`--runs`와 개수 일치 필요) |
| `--with-jacobian` | flag | off | 행 7(gradient 코사인 유사도) 추가 — 느림, 데이터셋 재구축 필요 |
| `--checkpoint-name` | str | `best.pt` | run별 사용할 체크포인트 파일명(없으면 `final.pt` 폴백) |
| `--infer-batch-size` | int | 256 | 인퍼런스 시간 측정용 배치 크기 |
| `--infer-warmup` | int | 10 | 인퍼런스 시간 측정 전 워밍업 반복 수 |
| `--infer-reps` | int | 50 | 인퍼런스 시간 측정 반복 수 |
| `--jacobian-max-samples` | int | 300 | 시나리오/전체당 gradient 계산에 쓸 최대 샘플 수 |
| `--device` | str | `auto` | 연산 디바이스 |
| `--out-name` | str | None | 결과 폴더명 전체 오버라이드(`--title`보다 우선) |
| `--title` | str | `result_comparison` | 결과 폴더명의 접미사 부분만 대체(`<MMDD_HHMM>_<title>`) |
| `--rep-cells` | str 리스트 | 자동선정(MIT/HUST 각 최대 3개) | 용량곡선 비교 플롯에 쓸 셀 ID 직접 지정 |
