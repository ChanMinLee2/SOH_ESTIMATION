# 5_model 코드 상세 설명

LFP 배터리 SOH(State of Health) 예측을 위한 Dynamic Feature Routing(DFR) 모델의 전체 코드를 파일별로 설명한다.

---

## 전체 흐름 요약

배터리 1사이클의 측정 데이터 → 411개 HI(Health Indicator) 피처 추출(Step 4) → **DFR 모델이 어떤 피처 그룹을 쓸지 스스로 선택하면서 용량(Ah)을 예측(Step 5)**

DFR의 핵심 아이디어는 "모든 피처를 항상 쓰지 말고, 현재 사이클 상태에 따라 필요한 그룹만 골라 쓰자"는 것이다. 24개 피처 그룹(6 구간 × 4 카테고리)에 각각 on/off 게이트를 달아, 학습 중에 게이트를 함께 최적화한다.

---

## 디렉토리 구조

```
5_model/
  config/
    default.yaml            하이퍼파라미터 전체 정의
  utils/
    compat.py               numpy 버전 호환성 패치
    tqdm_utils.py           tqdm 안전 래퍼
    hi_groups.py            HI 피처 그룹 메타데이터
    metrics.py              회귀/라우팅 평가 지표
    visualization.py        학습 곡선·산점도·히트맵 저장
    io_utils.py             체크포인트·설정·JSON 저장/로드
  datasets/
    battery_dataset.py      데이터 로드·분할·DataLoader 생성
    normalization.py        StandardScaler (훈련셋 기준 피팅)
  models/
    encoder.py              InitialEncoder, CategoryEncoder
    router.py               FeatureRouter (Gumbel-Sigmoid 게이트)
    feature_selector.py     FeatureSelector (게이트 × NaN 마스크)
    feature_fusion.py       FeatureFusion (concat → MLP)
    capacity_head.py        CapacityHead (회귀 헤드)
    dfr_model.py            DFRModel (6개 서브모듈 통합)
  training/
    loss.py                 DFRLoss (MSE + 희소성 정규화)
    trainer.py              Trainer (학습 루프 전체 관리)
  evaluation/
    evaluator.py            Evaluator (평가·저장·시각화)
  train.py                  학습 진입점
  test.py                   평가 진입점
  predict.py                추론 진입점
```

---

## 1. config/default.yaml — 하이퍼파라미터 설정

### 하는 일
모델 구조, 라우터, 학습, 평가에 필요한 모든 숫자값을 한 곳에 모아 관리한다. 코드를 수정하지 않고 이 파일만 바꿔서 실험을 반복할 수 있다.

### 주요 섹션

| 섹션 | 핵심 항목 | 설명 |
|------|-----------|------|
| `data` | `io_workers: 8` | pkl 병렬 로딩 스레드 수 |
| `data` | `num_workers: 0` | DataLoader worker 수 (Windows 안전값) |
| `model` | `d_enc: 64` | InitialEncoder 출력 차원 |
| `model` | `d_grp: 32` | 그룹 인코더 출력 차원 |
| `model` | `d_fus: 128` | FeatureFusion 출력 차원 |
| `router` | `gumbel_temp_start: 2.0` | 학습 초반 게이트 부드러움 정도 |
| `router` | `gumbel_temp_end: 0.5` | 학습 후반 게이트 이진화 정도 |
| `training` | `lambda_sparse: 0.01` | 희소성 패널티 강도 |
| `training` | `cost_weighted: true` | 비싼 피처 그룹에 더 강한 패널티 |

### 왜 YAML인가
Python dict로 관리하면 실험 설정이 코드 안에 묻혀 재현성이 떨어진다. YAML로 분리하면 실험마다 `config.yaml` 스냅샷을 결과 폴더에 저장할 수 있어, 나중에 어떤 설정으로 돌렸는지 추적 가능하다.

---

## 2. utils/compat.py — numpy 버전 호환성 패치

### 문제 배경
- Step 4에서 HI pkl 파일을 만들 때 사용한 numpy 버전: **2.1.3**
- Step 5 학습 환경(`LFP_SOH_ESTIMATION` conda 환경)의 numpy 버전: **1.24.4**
- numpy 2.0에서 내부 모듈이 `numpy.core.*` → `numpy._core.*`로 이동했다.
- numpy 1.x 환경에서 numpy 2.x가 만든 pkl을 열면 `ModuleNotFoundError: No module named 'numpy._core'` 에러가 발생한다.

### 해결 방법 (sys.modules 패치)

```python
# numpy 1.x 환경에서만 실행 (numpy 2.x는 _core가 이미 있음)
if not hasattr(np, "_core"):
    fake = types.ModuleType("numpy._core")
    fake.__dict__.update(numpy.core.__dict__)
    sys.modules["numpy._core"] = fake        # pickle unpickler가 찾는 경로에 등록
    # 하위 서브모듈도 동일하게 처리
    for submod in pkgutil.iter_modules(numpy.core.__path__, ...):
        sys.modules["numpy._core.xxx"] = 실제_numpy.core.xxx
```

파이썬의 `pickle`은 객체를 역직렬화할 때 `sys.modules`에서 모듈을 찾는다. numpy 2.x가 `numpy._core.multiarray`에 저장한 클래스를 numpy 1.x에서 열면, `sys.modules`에 `numpy._core`가 없어서 실패한다. 이 shim은 import 시점에 `sys.modules`에 alias를 등록해 두는 방식으로 해결한다.

### 특징
- 이 파일을 **import하기만 하면** 패치가 자동 적용된다 (`install_numpy2_shim()`이 모듈 로드 시 즉시 실행).
- `battery_dataset.py`가 제일 먼저 `import utils.compat`를 해서, 어떤 pkl 로딩보다 먼저 패치가 걸린다.

---

## 3. utils/tqdm_utils.py — tqdm 안전 래퍼

### 하는 일
`tqdm` 라이브러리가 설치되지 않은 환경에서도 코드가 조용히 동작하도록 fallback을 제공한다.

```python
try:
    from tqdm import tqdm as _tqdm
    def tqdm(iterable, **kwargs): return _tqdm(iterable, **kwargs)
except ImportError:
    def tqdm(iterable=None, **kwargs): return iterable  # 그냥 통과
```

### 왜 필요한가
VS Code Python 확장이 base conda 환경을 가리키고 있어 `tqdm`을 못 찾아 linter 경고가 발생했다. 각 파일에서 직접 `from tqdm import tqdm`을 쓰면 환경마다 에러가 날 수 있기 때문에, 한 곳에서 try/except로 처리하고 나머지 파일은 이 래퍼만 쓴다.

---

## 4. utils/hi_groups.py — HI 피처 그룹 메타데이터

### 하는 일
실제 데이터 파일(pkl)의 컬럼 목록을 스캔해서, DFR 모델이 다룰 피처 그룹 구조 전체를 자동으로 파악한다.

### 핵심 개념

**피처의 분류 체계**

| 종류 | 컬럼 수 | 예시 |
|------|---------|------|
| Global HI (항상 계산) | 15개 | `q_dis`, `r_dc_est`, `ica_peak1_h`, ... |
| Segment-Category HI (선택적) | 396개 (6구간 × 66) | `stat_v_mean_dis_hi`, `diff_dvdq_mean_chg_lo`, ... |

**6개 구간(segment)**
- 방전 구간: `dis_hi` (고SoC), `dis_mid`, `dis_lo` (저SoC)
- 충전 구간: `chg_lo` (저SoC), `chg_mid`, `chg_hi` (고SoC)

**4개 카테고리(category)**
- `stat` : 평균·분산 등 기본 통계
- `diff` : dV/dQ, dQ/dV 등 미분 기반 피처
- `lfp`  : LFP 고원(plateau) 특화 피처
- `morph`: DTW·Fréchet 형태 거리 (가장 계산 비용이 큰)

**24개 그룹** = 6구간 × 4카테고리 → 각 그룹이 DFR의 게이트 1개에 대응

### HIGroupInfo 데이터클래스

```python
@dataclass
class HIGroupInfo:
    global_his: List[str]              # 15개 Global HI 이름
    group_keys: List[Tuple[str, str]]  # 24개 (seg, cat) 쌍
    groups: Dict[str, List[str]]       # 그룹이름 → 컬럼 목록
    cat_sizes: Dict[str, int]          # 카테고리별 피처 수
    group_costs: List[float]           # 그룹별 계산 비용
```

### 자동 탐지 로직
컬럼 이름 규칙이 `{카테고리}_{HI이름}_{구간}` (예: `stat_v_mean_dis_hi`)이므로, prefix/suffix 매칭만으로 그룹을 자동 분류한다. 데이터가 바뀌어 컬럼이 추가/삭제되어도 코드 수정 없이 자동 반영된다.

### 계산 비용 가중치 (CATEGORY_COSTS)

```
stat: 1.0   diff: 1.5   lfp: 2.0   morph: 3.0
```

이 값은 희소성 손실 계산에 쓰인다. 비용이 비싼 `morph` 그룹을 선택하면 더 큰 패널티를 받아, 모델이 저렴한 피처를 선호하도록 유도한다.

---

## 5. datasets/battery_dataset.py — 데이터 로드·분할·DataLoader

### 하는 일
`_4_data_hi/MIT/*.pkl`, `_4_data_hi/HUST/*.pkl` 파일들을 읽어 PyTorch DataLoader로 만드는 전 과정을 담당한다.

### 핵심 설계 결정들

#### 병렬 pkl 로딩 (ThreadPoolExecutor)
```python
with ThreadPoolExecutor(max_workers=io_workers) as executor:
    future_map = {executor.submit(_load_cell, path, ds, min_cycles): (path, ds)
                  for path, ds in tasks}
    for future in tqdm(as_completed(future_map), total=len(future_map)):
        df = future.result()
```

- **왜 Thread, Process 아님?**: pkl 로딩은 디스크 읽기(I/O-bound). CPU 연산이 병목이 아니라 디스크 대기가 병목이므로 스레드로 충분하다. 또한 Windows에서는 `ProcessPoolExecutor`가 `conda run` 환경에서 spawn 문제를 일으킨다.
- **as_completed**: 먼저 끝난 파일부터 처리해 메모리에 쌓는다.
- **tqdm**: 123개 셀이 로딩되는 과정을 실시간 진행 바로 표시.

#### 셀 단위 분할 (데이터 누수 방지)
```python
# 잘못된 방법: 사이클 단위 랜덤 분할
# → 같은 셀의 사이클 50이 train에, 사이클 100이 val에 들어갈 수 있음
# → 모델이 해당 셀의 열화 패턴을 외워버림 (누수)

# 올바른 방법: 셀 단위 분할
for dataset in ["MIT", "HUST"]:
    cells = df[df["dataset"] == dataset]["cell_id"].unique()
    train_cells, val_cells, test_cells = _split_group(cells)
```

셀 b1c0 전체(사이클 1~1000)가 train에 들어가면, b1c0의 어떤 사이클도 val/test에 들어가지 않는다. 데이터셋별(MIT/HUST)로 층화 분할해서 각 셋에 두 데이터셋이 골고루 포함되게 한다.

#### BatteryHIDataset (모든 텐서 사전 빌드)
```python
class BatteryHIDataset(Dataset):
    def __init__(self, df, hi_info, normalizer):
        # 초기화 때 모든 텐서를 미리 만들어 둠
        self._x_global = torch.tensor(...)         # (N, 15)
        self._group_features = [torch.tensor(...)] # 24개 × (N, n_k)
        self._nan_masks = [torch.tensor(...)]       # 24개 × (N, n_k)
        self._targets = torch.tensor(...)          # (N,)

    def __getitem__(self, idx):
        return {"x_global": self._x_global[idx], ...}  # 순수 인덱스
```

`__getitem__`이 인덱스 참조만 하므로 DataLoader `num_workers=0`(메인 프로세스)으로도 병목 없이 빠르다. Windows에서 multiprocessing worker를 쓰면 각 worker가 데이터셋 전체를 복사해야 해서 오히려 느리고 불안정하다.

#### Windows 안전 num_workers
```python
def _safe_num_workers(requested: int) -> int:
    if sys.platform == "win32" and requested > 0:
        if not any("train.py" in a for a in sys.argv):
            return 0
    return requested
```

`train.py`로 직접 실행하는 경우(프로세스 가드 `if __name__ == "__main__"` 보장)에만 `num_workers > 0`을 허용한다.

---

## 6. datasets/normalization.py — StandardScaler

### 하는 일
각 피처를 평균 0, 표준편차 1로 정규화한다. **반드시 훈련 데이터로만 피팅**하고, val/test에는 훈련 통계를 적용한다.

### 설계 포인트

#### NaN 처리 두 단계
1. **fit 시**: `np.nanmean`, `np.nanstd`로 NaN을 제외하고 통계 계산
2. **transform 시**: NaN 위치를 별도로 기록(마스크)하고, NaN 값은 0.0으로 채움

```python
mask = (~np.isnan(vals)).astype(np.float32)   # 1=유효, 0=NaN이었음
vals_z = (vals - mean) / std
vals_z = np.nan_to_num(vals_z, nan=0.0)       # NaN → 평균(=정규화 후 0)
```

NaN을 0으로 채우는 이유: z-score 정규화 후 평균이 0이므로, NaN을 0으로 채우는 것은 "이 피처값을 모른다 → 평균으로 간주"를 의미한다. 그런 다음 마스크를 모델에 함께 전달해서 "이건 실제 0이 아니라 결측"임을 알린다.

#### 모든-NaN 컬럼 처리
```python
means = np.where(np.isnan(means), 0.0, means)
stds = np.where(np.isnan(stds) | (stds < 1e-8), 1.0, stds)
```
어떤 피처가 훈련셋 전체에서 NaN인 경우 mean=0, std=1로 fall back해서 0으로 채워진다.

#### RuntimeWarning 억제
```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    means = np.nanmean(vals, axis=0)
```
numpy 1.24.4에서 전체 NaN 컬럼에 `np.nanmean`을 쓰면 RuntimeWarning이 발생하는데, `np.errstate`로는 억제가 안 되고 `warnings` 모듈로 억제해야 한다.

---

## 7. models/encoder.py — InitialEncoder & CategoryEncoder

두 인코더가 한 파일에 있다. 모두 MLP 기반이며 구조가 유사해서 같이 관리한다.

### InitialEncoder

**역할**: 항상 계산 가능한 Global HI 15개 → 잠재 벡터 z (d_enc 차원)

```
x_global (B, 15) → [Linear → LayerNorm → GELU → Dropout] × (n_layers-1) → Linear → LayerNorm → z (B, 64)
```

- **LayerNorm**: Batch Normalization 대신 LayerNorm을 쓴다. 배치 크기에 무관하게 각 샘플을 독립적으로 정규화하므로, 작은 배치나 추론 시 1개 샘플에서도 안정적이다.
- **GELU**: ReLU 대신 GELU(Gaussian Error Linear Unit). 부드러운 활성화 함수로 Transformer 계열에서 표준적으로 쓰인다. 음수 입력에도 작은 그래디언트를 허용한다.
- z는 **라우터의 입력**이자 **FeatureFusion의 입력**으로 두 곳에서 재사용된다.

### CategoryEncoder

**역할**: 하나의 (구간, 카테고리) 그룹 피처 → d_grp 차원 임베딩

핵심은 **카테고리 공유(category-shared)** 설계다.

```python
self.group_encoders = nn.ModuleDict({
    "stat":  CategoryEncoder(n_stat_features, d_grp),   # stat 인코더 1개
    "diff":  CategoryEncoder(n_diff_features, d_grp),   # diff 인코더 1개
    "lfp":   CategoryEncoder(n_lfp_features,  d_grp),   # lfp  인코더 1개
    "morph": CategoryEncoder(n_morph_features, d_grp),  # morph인코더 1개
})
```

`dis_hi_stat`, `dis_mid_stat`, ..., `chg_hi_stat` — 6개 구간의 stat 피처가 **모두 같은 stat 인코더**를 통과한다. 24개 그룹을 위해 24개 인코더를 만드는 것이 아니라 4개만 만든다.

이유: 같은 카테고리 피처는 구간이 달라도 피처의 **의미**와 **스케일**이 비슷하다. 공유 인코더를 쓰면 파라미터 수를 줄이고, 구간 간 일반화 능력을 높인다.

---

## 8. models/router.py — FeatureRouter (Gumbel-Sigmoid)

### 역할
현재 사이클의 전반적 상태(z)를 보고, 24개 HI 그룹 각각에 대해 "이 그룹을 쓸 것인가?"를 결정한다.

```
z (B, 64) → [Linear → LayerNorm → GELU → Dropout → Linear] → logits (B, 24) → gates (B, 24)
```

### 핵심 기법: Gumbel-Sigmoid (Concrete Bernoulli)

라우팅 게이트를 on/off 이진값으로 만들고 싶지만, 이진 결정은 미분 불가능해서 역전파가 안 된다.

**해결**: Gumbel-Sigmoid는 이진 결정의 미분 가능한 연속 근사다.

```python
# 학습 중: Gumbel 노이즈 추가 → soft gate (0~1 사이 연속값)
u = torch.zeros_like(logits).uniform_().clamp(1e-6, 1-1e-6)
gumbel = -torch.log(-torch.log(u))           # Gumbel 노이즈
gates = torch.sigmoid((logits + gumbel) / temperature)

# 추론 시: 노이즈 없이 시그모이드 → hard thresholding
gates = torch.sigmoid(logits)
if hard:
    gates = (gates > 0.5).float()            # 실제 이진 게이트
```

**Gumbel 노이즈의 역할**: 학습 초반에 로짓이 불확실할 때 탐색(exploration)을 유도한다. 노이즈가 있어야 게이트가 0이나 1 근처에 고착되지 않고 다양한 조합을 시도한다.

### Temperature Annealing (온도 감쇠)

```
학습 초반 temperature = 2.0 → 게이트가 0.3, 0.7처럼 부드럽게 퍼짐
학습 후반 temperature = 0.5 → 게이트가 0.05, 0.95처럼 이진에 가깝게 수렴
```

온도를 에폭마다 `T = max(T_end, T_start × decay^epoch)`로 낮춘다. 처음에는 많은 그룹을 탐색하고, 나중에는 실제로 유용한 그룹만 남기도록 유도한다.

---

## 9. models/feature_selector.py — FeatureSelector

### 역할
라우터의 게이트와 normalization 단계의 NaN 마스크를 결합해서, 실제로 모델이 볼 피처를 결정한다.

```python
for k, (x_k, mask_k) in enumerate(zip(group_features, nan_masks)):
    g_k = gates[:, k:k+1]         # (B, 1) — 그룹 k의 게이트
    x_out = x_k * mask_k * g_k    # (B, n_k) — NaN 제거 + 게이트 적용
```

### NaN 마스크와 라우팅 게이트의 분리

두 종류의 "0"이 있다:
- **NaN 마스크 = 0**: 이 피처는 원래 계산이 불가능했음 (예: 충전 중단으로 세그먼트 HI가 없음)
- **라우팅 게이트 = 0**: 이 피처 그룹은 계산 가능하지만 이번 사이클에서는 선택하지 않기로 결정

이 두 가지를 곱셈으로 결합하면 처리가 단순해지고, FeatureSelector 자체는 학습 파라미터가 없는 순수 연산 모듈이다.

### 학습 vs 추론의 차이
- **학습**: gate ∈ (0, 1) 연속값 → 그래디언트가 역전파됨
- **추론**: gate ∈ {0, 1} 이진값 → gate=0인 그룹은 실제로 계산 건너뜀 (연산량 절감)

---

## 10. models/feature_fusion.py — FeatureFusion

### 역할
24개 그룹 인코딩 + Global 인코딩 z를 하나의 벡터로 합친다.

```
[z (64), h_0 (32), h_1 (32), ..., h_23 (32)] → concat (64 + 24×32 = 832) → MLP → h_fused (128)
```

### Concat 방식의 장점

Attention이나 Pooling 대신 단순 concat을 택한 이유:

- **고정 차원 보장**: gate=0으로 비활성화된 그룹의 h_k는 0 벡터가 되는데, concat이면 입력 차원이 항상 고정이다. Attention처럼 가변 길이 시퀀스를 다룰 필요 없다.
- **모든 위치 정보 보존**: 그룹 k가 fusion 벡터의 특정 위치에 항상 대응하므로, MLP가 "어떤 그룹이 활성화됐는지"도 암묵적으로 학습할 수 있다.
- **단순함**: 파라미터와 연산 복잡도가 낮아 162K 파라미터 모델로도 충분한 표현력을 갖는다.

---

## 11. models/capacity_head.py — CapacityHead

### 역할
h_fused → 배터리 용량(Ah) 스칼라값 회귀

```
h_fused (B, 128) → Linear(128, 64) → LayerNorm → GELU → Dropout → Linear(64, 1) → squeeze → cap (B,)
```

### 설계 선택
- 출력 활성화 함수 없음: 배터리 용량은 양수지만, 학습 데이터 범위 내에서 MSE 로스가 자연스럽게 올바른 범위로 유도한다. Softplus나 ReLU를 달면 그래디언트 흐름을 방해할 수 있어 생략했다.
- `.squeeze(-1)`: Linear 출력이 `(B, 1)`이므로 마지막 차원을 제거해 `(B,)`로 만든다.

---

## 12. models/dfr_model.py — DFRModel (전체 통합)

### 역할
위 6개 서브모듈을 조립해서 end-to-end 모델을 구성한다. 외부에서는 이 클래스만 보면 된다.

### 전체 Forward 흐름

```
입력 배치:
  x_global      (B, 15)          → 항상 계산 가능한 Global HI
  group_features [(B, n_k)] × 24 → 24개 그룹 피처 (일부 NaN 포함)
  nan_masks      [(B, n_k)] × 24 → 각 피처의 유효 여부

Step 1 — InitialEncoder
  x_global (B, 15) → z (B, 64)

Step 2 — FeatureRouter
  z (B, 64) → gates (B, 24)      ← Gumbel-Sigmoid

Step 3 — FeatureSelector
  group_features, nan_masks, gates → gated_features [(B, n_k)] × 24

Step 4 — CategoryEncoder (공유)
  gated_features[k] (B, n_k) → h_k (B, 32)   (같은 카테고리는 같은 인코더)
  결과: h_groups [(B, 32)] × 24

Step 5 — FeatureFusion
  z, h_groups → concat (B, 832) → MLP → h_fused (B, 128)

Step 6 — CapacityHead
  h_fused (B, 128) → capacity_Ah (B,)

반환: (cap, gates)
```

### 가중치 초기화 (Kaiming Normal)

```python
for m in self.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)
```

GELU 활성화를 쓰지만 ReLU 기준으로 초기화해도 실용적으로 잘 동작한다. 바이어스는 0으로 시작해서 학습 초반 활성화가 고르게 분포되도록 한다.

### predict() 메서드

`forward()`와 별도로 추론 전용 메서드가 있다.

```python
@torch.no_grad()
def predict(self, batch, device="cpu"):
    self.eval()
    # 배치를 device로 이동
    cap, gates = self.forward(batch_dev, hard=True)   # hard=True: 이진 게이트
    return cap.cpu(), gates.cpu()
```

`@torch.no_grad()`로 그래디언트 계산을 끄고, `hard=True`로 실제 이진 게이트를 써서 선택된 그룹의 수를 정확히 알 수 있다.

---

## 13. training/loss.py — DFRLoss

### 역할
배터리 용량 예측 정확도와 피처 선택 희소성을 동시에 최적화하는 복합 손실함수.

```
Total Loss = MSE(pred, true) + λ × SparsityLoss
```

### SparsityLoss (비용 가중 희소성 패널티)

```python
if cost_weighted:
    costs_norm = group_costs / group_costs.max()   # [0, 1] 정규화
    sparse = (gates * costs_norm).mean()           # 게이트×비용 평균
else:
    sparse = gates.mean()                          # 게이트 평균 (균일 패널티)
```

- `gates`는 학습 중 (0~1) 연속값 → 미분 가능
- 게이트가 클수록(그룹을 쓸수록) 패널티가 커짐
- `morph` 그룹(cost=3.0)은 `stat` 그룹(cost=1.0)보다 3배 패널티 → 모델이 `morph`보다 `stat`을 선호하도록 유도

### λ(lambda_sparse)의 의미

| λ 값 | 결과 |
|------|------|
| 0.0 | 희소성 패널티 없음 → 24개 그룹 모두 활성화 |
| 0.01 (기본) | 정확도와 희소성 균형 → 평균 12개 내외 그룹 |
| 0.1 | 강한 희소성 강제 → 매우 적은 그룹, 정확도 저하 가능 |

### register_buffer

```python
self.register_buffer("group_costs", torch.tensor(group_costs))
```

`group_costs`를 모델 파라미터가 아닌 버퍼로 등록한다. `.to(device)` 시 자동으로 같은 device로 이동하지만, optimizer가 업데이트하지는 않는다.

---

## 14. training/trainer.py — Trainer

### 역할
학습 루프 전체를 관리한다. epoch마다 train→val→로깅→체크포인트→조기종료를 순서대로 실행한다.

### 학습 스케줄

#### Cosine LR (Warmup 포함)

```
워밍업 (epoch 0~4):    lr = base_lr × (epoch+1) / warmup_epochs  (선형 증가)
코사인 감쇠 (5~200):   lr = min_lr + 0.5 × (base_lr - min_lr) × (1 + cos(π × progress))
```

훈련 초반에 lr을 급격히 올리면 loss가 발산할 수 있어, 처음 5 epoch는 천천히 올린다. 이후 코사인 곡선으로 부드럽게 감쇠시켜 지역 최솟값에 더 잘 수렴하게 한다.

#### Temperature Annealing

```python
temperature = max(T_end, T_start × decay^epoch)
# 기본: max(0.5, 2.0 × 0.99^epoch)
# epoch 0: T=2.0, epoch 50: T≈1.21, epoch 138: T≈0.50
```

epoch마다 라우터에 전달되는 Gumbel temperature를 줄여, 학습이 진행될수록 게이트가 이진에 가까워진다.

#### Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

역전파 후 파라미터 업데이트 전에 그래디언트 L2 노름을 1.0으로 제한한다. 희소성 손실과 MSE 손실이 합산될 때 그래디언트 스케일이 불안정해질 수 있어 클리핑으로 안정화한다.

### tqdm 진행 바 구조

```
Training  30%|███       | 60/200 [05:43, best=0.0321]
  train ep  60: 100%|██| 312/312 [loss=0.0482]           (완료 후 사라짐)
    val ep  60: 100%|██| 39/39   [loss=0.0321]            (완료 후 사라짐)
```

- **outer bar**: 전체 epoch 진행. `tr_rmse`, `val_rmse`, `active` 그룹 수, Gumbel temperature, 최적 여부를 postfix로 표시. 항상 유지됨.
- **inner bar**: 배치 진행. `leave=False`로 epoch 완료 후 자동 삭제돼 화면을 정리한다.

### 조기 종료 (Early Stopping)

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    no_improve = 0
    save_checkpoint(...)   # best.pth 저장
else:
    no_improve += 1

if no_improve >= patience:  # 기본 patience=30
    break
```

val loss가 30 epoch 동안 개선되지 않으면 학습을 중단한다. 과적합을 방지하고 불필요한 계산을 절약한다.

### 체크포인트 저장 전략

- `best.pth`: val loss 최솟값을 달성할 때마다 덮어씀
- `epoch_XXXX.pth`: 10 epoch마다 스냅샷 저장

체크포인트에는 `model_state`, `optimizer_state`, `epoch`, `val_loss`, `temperature`가 저장된다. 중단된 학습을 이어받거나, 특정 epoch 상태로 복원할 수 있다.

---

## 15. evaluation/evaluator.py — Evaluator

### 역할
저장된 모델로 전체 test/val셋을 추론하고, 결과를 파일로 저장·시각화한다.

### 수행 순서

1. **배치 추론** (tqdm 진행 바 표시)
   ```python
   pred, gates = model(batch, hard=True)   # hard=True: 이진 게이트
   ```

2. **지표 계산**: RMSE, MAE, R², MAPE

3. **라우팅 통계**: 그룹별 평균 게이트 활성화율, 전체 희소도

4. **파일 저장**
   ```
   metrics/{split}_metrics.json       → 지표 수치
   predictions/{split}_predictions.csv → 사이클별 예측값 + 게이트 상태
   routing/{split}_routing_stats.csv  → 그룹별 활성화율
   figures/{split}_scatter.png        → 예측 vs 실측 산점도
   figures/{split}_routing_heatmap.png→ 6구간×4카테고리 게이트 히트맵
   ```

### predictions CSV의 gate 컬럼

```csv
cell_id, cycle, capacity_true, capacity_pred, gate_dis_hi_stat, gate_dis_hi_diff, ...
b1c0,    1,     1.1,           1.08,          1.0,              0.0, ...
```

사이클마다 어떤 그룹이 선택됐는지 기록된다. 이를 통해 "셀이 열화될수록 어떤 그룹의 활성화 패턴이 변하는가"를 사후 분석할 수 있다.

---

## 16. utils/metrics.py — 평가 지표

4가지 회귀 지표를 구현한다.

| 지표 | 수식 | 의미 |
|------|------|------|
| RMSE | √(Σ(ŷ-y)²/N) | 오차 크기 (Ah 단위). 이상치에 민감 |
| MAE | Σ\|ŷ-y\|/N | 오차 크기 (Ah 단위). 이상치에 덜 민감 |
| R² | 1 - SS_res/SS_tot | 설명력. 1.0에 가까울수록 좋음 |
| MAPE | Σ\|ŷ-y\|/y × 100 | 상대 오차(%). 실제 용량 대비 몇 % 틀렸나 |

`routing_stats(gates, group_names)`: 게이트 행렬 `(N, 24)`에서 그룹별 평균 활성화율과 전체 희소도를 계산한다.

---

## 17. utils/visualization.py — 시각화

4종의 그래프를 `_5_data_model/<run>/figures/`에 저장한다. 모두 `matplotlib.use("Agg")`로 화면 없이 파일로만 저장한다 (서버/conda run 환경 호환).

| 함수 | 저장 파일 | 내용 |
|------|-----------|------|
| `plot_training_curves` | `training_curves.png` | 학습/검증 loss·RMSE 곡선 (2패널) |
| `plot_temperature_schedule` | `temperature_schedule.png` | epoch별 Gumbel temperature 감쇠 |
| `plot_prediction_scatter` | `{split}_scatter.png` | 실측 vs 예측 산점도 (MIT/HUST 색 구분) |
| `plot_routing_heatmap` | `{split}_routing_heatmap.png` | 6구간 × 4카테고리 게이트 평균 히트맵 |

라우팅 히트맵은 DFR의 핵심 해석 도구다. 어떤 구간/카테고리 조합이 용량 예측에 중요한지 한눈에 볼 수 있다.

---

## 18. utils/io_utils.py — 파일 입출력

체크포인트, 설정, JSON, pickle 저장/로드를 담당하는 유틸리티 함수 모음.

| 함수 | 역할 |
|------|------|
| `load_config` / `save_config` | YAML 설정 읽기/쓰기 |
| `save_checkpoint` / `load_checkpoint` | PyTorch 모델·옵티마이저 상태 저장/로드 |
| `save_json` | 지표·이력 JSON 저장 |
| `save_pickle` / `load_pickle` | normalizer 저장/로드 |

`load_checkpoint`에서 `map_location=device`를 지정해, GPU로 저장한 체크포인트를 CPU 환경에서도 로드할 수 있다.

---

## 19. 진입점 3종

### train.py
전체 파이프라인을 순서대로 실행한다:
1. config 로드 + CLI 오버라이드
2. Device 결정 (CUDA 자동 감지)
3. HI 그룹 정보 빌드 (`build_hi_group_info_from_data`)
4. DataLoader 생성 (`make_dataloaders`)
5. 모델·손실·옵티마이저 생성
6. `Trainer.train()` 실행
7. best.pth 로드 후 test/val 평가 (`Evaluator.evaluate()`)
8. 결과를 `_5_data_model/<run_name>/`에 저장

### test.py
저장된 run 디렉토리를 받아 특정 split(train/val/test)을 재평가한다. 훈련 때 저장한 `normalizer.pkl`과 `config.yaml`을 읽어 동일한 전처리를 재현한다.

### predict.py
새로운 pkl 파일(훈련에 사용하지 않은 셀)에 대해 용량을 추론한다. 단일 파일 또는 디렉토리 전체를 한 번에 처리할 수 있다. `--save-routing` 옵션을 주면 각 사이클에서 어떤 그룹이 선택됐는지도 CSV에 기록된다.

---

## 전체 파라미터 수 분석

기본 설정(d_enc=64, d_grp=32, d_fus=128, d_head=64) 기준:

| 서브모듈 | 파라미터 수 |
|---------|------------|
| InitialEncoder | 15×128 + 128×64 ≈ 10K |
| FeatureRouter | 64×64 + 64×24 ≈ 5.6K |
| FeatureSelector | 0 (파라미터 없음) |
| CategoryEncoder × 4 | 4 × (n_k×64 + 64×32) ≈ 50K |
| FeatureFusion | 832×128 + 128×128 ≈ 123K |
| CapacityHead | 128×64 + 64×1 ≈ 8.3K |
| **합계** | **≈ 162K** |

162K는 딥러닝 모델 중 매우 작은 편이다. 배터리 데이터의 특성상 셀 수(~200개)와 전체 사이클 수(~10만)가 많지 않아, 과적합 방지를 위해 의도적으로 작게 설계했다.

---

## 데이터 흐름 요약

```
_4_data_hi/MIT/*.pkl          (셀별 DataFrame: 415 컬럼)
_4_data_hi/HUST/*.pkl
        │
        ▼
battery_dataset.py
  load_all_cells()            병렬 로딩 (ThreadPoolExecutor)
  split_cells()               셀 단위 80/10/10 분할
  FeatureNormalizer.fit()     훈련셋으로만 StandardScaler 피팅
  BatteryHIDataset()          텐서 사전 빌드
  DataLoader                  배치 생성
        │
        ▼  (한 배치)
  x_global (B, 15)            Global HI
  group_features [(B,n_k)]×24 Segment-Category HI
  nan_masks      [(B,n_k)]×24 유효 여부 마스크
  target         (B,)          실측 capacity_Ah
        │
        ▼
DFRModel.forward()
  z = encoder(x_global)               (B, 64)
  gates = router(z, T, hard)          (B, 24)  ← Gumbel-Sigmoid
  gated = selector(features, masks, gates)
  h_k = group_encoders[cat](gated[k]) (B, 32)  ← 카테고리 공유
  h_fused = fusion(z, h_groups)       (B, 128)
  cap = head(h_fused)                 (B,)
        │
        ▼
DFRLoss(cap, target, gates)
  MSE + λ × (gates × costs).mean()
        │
        ▼
optimizer.step()               파라미터 업데이트
```
