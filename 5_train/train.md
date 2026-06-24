# Battery SOH Prediction Training Pipeline Guide

이 문서는 `src/train.py`를 중심으로 한 딥러닝/머신러닝 기반 배터리 SOH 예측 시스템의 전체 아키텍처와 상세 흐름을 설명합니다.

---

## 1. 입력 데이터 구조 (Input Data Structure)

학습에 사용되는 데이터는 `optimized_tensors.pkl` 형태의 고도로 정제된 피처 셋입니다.

### 1.1. 피처 구성 (Feature Dimensions: 48D)
각 세그먼트 데이터는 총 48차원의 벡터로 구성됩니다.
- **[0:15] Charge HI**: 충전 전압 프로파일 기반 HI (Health Indicator).
- **[15:30] Discharge HI**: 방전 전압 프로파일 기반 HI.
- **[30:45] Common HI & IC**: 충/방전 공통 피처 및 Incremental Capacity 분석 피처.
- **[45:48] Meta Data**: 

### 1.2. 타겟 및 라벨 (Target & Labels)
- **Target**: `capacity` (Ah 단위, SOH 예측의 정답값).
- **t (Time)**: `cyc` (Cycle 번호). PINN 학습을 위해 `0~1` 범위로 정규화되어 사용됨.
- **mode_label**: `1` (Charge), `0` (Discharge). 피처 마스킹의 기준이 됨.

---

## 2. 데이터셋 및 로더 (Dataset & DataLoader)

### 2.1. 데이터 분리 전략 (Cell-wise Split)
데이터 누수(Data Leakage)를 방지하기 위해 **Cell 단위**로 데이터를 분리합니다.
- **비율**: Train(60%), Val(20%), Test(20%)
- 동일한 배터리 셀의 데이터는 절대로 훈련 셋과 테스트 셋에 동시에 존재할 수 없습니다.

### 2.2. BatterySOHDataset 특징
- **Normalization**: Cycle 데이터를 `2500.0`으로 나누어 수치적 안정성을 확보.
- **NaN Handling**: 로딩 시점에 `NaN` 값을 `0.0`으로 자동 치환하여 훈련 중단 방지.
- **Return Value**: `(x, t, m, y)` 4가지 요소를 반환 (Feature, Time, Mode, Target).

---

## 3. 하이퍼파라미터 (Hyperparameters)

`src/hyperparams.py`에서 통합 관리됩니다.
- **DL Params**: `batch_size: 512`, `LR: 5e-4`, `patience: 40`.
- **PINN Params**: 
    - `use_pi`: PINN 활성화 여부.
    - `alpha`: PDE Residual Loss 가중치 (데이터 손실과의 균형 조절).
- **Optimizer**: `AdamW` (Weight Decay 적용으로 과적합 방지).
- **Scheduler**: `ReduceLROnPlateau` (검증 손실 정체 시 LR 감소).

---

## 4. 모델 구성 (Model Architectures)

### 4.1. Deep Learning Models
- **SimpleMLP**: 다층 퍼셉트론 기반의 기본 회귀 모델.
- **VanillaLSTM**: 시계열 의존성을 파악하기 위한 순환 신경망.
- **InvertedTransformer**: 변수 간의 상관관계를 Attention 메커니즘으로 파악.

### 4.2. Physics-Informed (PI) Wrapper
기존 모델을 감싸는 **PhysicsInformedWrapper**가 핵심 역할을 수행합니다.
- **Feature Masking**: `mode_label`에 따라 충전/방전 피처를 실시간으로 가림.
- **Dynamics Network**: 시간($t$), 상태($x$), 예측값($u$)을 입력받아 열화율($du/dt$)을 예측하는 별도의 소형 네트워크 포함.
- **PDE Residual**: 자동 미분을 통해 $\text{Loss}_{\text{PDE}} = \| \frac{du}{dt} - G(t, x, u, \dots) \|^2$를 계산.

### 4.3. Machine Learning Models
- `RF`, `SVR`, `GPR` 등 Scikit-Learn 기반 모델 지원.
- DL 모델과 동일한 데이터 로더를 사용하되, 내부적으로 NumPy 변환 과정을 거쳐 학습.

---

## 5. 학습 프로세스 흐름 (Training Workflow)

1.  **Seed & Device Setup**: 결과 재현성을 위해 시드를 고정하고 GPU(CUDA) 가용 여부 확인.
2.  **Data Loading**: 지정된 경로에서 최적화된 텐서를 읽어와 Cell 단위 분할 및 로더 생성.
3.  **Model Loop**: `MODELS_TO_RUN` 리스트에 정의된 모델 순차 실행.
4.  **DL Pipeline (`fit`)**:
    - 매 에폭마다 `train_epoch`와 `validate_epoch` 실행.
    - **NaN/Finite Check**: 손실이나 가중치가 폭주하면 즉시 중단.
    - **Gradient Clipping**: PINN 미분 경로의 안정성을 위해 그래디언트 크기 제한.
5.  **Validation & EarlyStopping**: 검증 성능이 개선되지 않으면 조기 종료 및 최적 모델 저장.
6.  **Experiment Tracking**:
    - `experiments/v{major}.{minor}.{patch}/` 경로에 설정값(`config.json`), 로그(`train_log.txt`), 모델(`best_model.pth`), 손실 곡선 저장.

---

## 7. Training Quick Start

모델 학습을 시작하려면 프로젝트 루트에서 다음 명령어를 실행하십시오:
```bash
python ./src/train.py
```

### 주요 체크포인트:
1.  **Device**: `Using device: cuda` 메시지가 나오는지 확인 (속도 차이가 큼).
2.  **Experiment Version**: `v1.1.1` 등 현재 설정된 버전이 로그에 맞게 표시되는지 확인.
3.  **NaN Loss**: 만약 학습 중 `Loss is nan` 메시지가 발생하면, `train.md`의 **6. 수치적 안정성 확보** 섹션을 참고하여 `alpha` 값을 조절하거나 데이터 범위를 점검하십시오.
