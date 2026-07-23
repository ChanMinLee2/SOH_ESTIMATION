# MODEL_SPECS — 원시 데이터 통합 및 CNN 분류기 설계안

> 작성일: 2026-07-22  
> 참조: `docs/전달/전달/build_dataset.py`, `hi_compute.py`, `hi_cost.json`, `MODEL_WINDOW.md`  
> 미반영 항목: 시간 단위 창 시나리오 개념 (보류)

---

## 1. 개요

현재 파이프라인은 세그먼트별 HI(Health Index) 64개만을 모델 입력으로 사용한다.  
이 설계안은 두 가지를 추가한다.

| # | 요구사항 | 핵심 변경 |
|---|---------|---------|
| 1 | **입력 토큰화** | 세그먼트 하나 = HI 64D + 원시 V/I 시계열(각 48pt) + 메타데이터를 하나의 토큰으로 구성 |
| 2 | **CNN 분류기** | `MLPProbeClassifier` → CNN(raw) + HI 64D 융합 후 분류기 통과 |

기존 SCR 회귀 파이프라인(Phase 1/2/3 축/세그먼트/평가)과 **완전 호환**되어야 한다.  
원시 데이터가 없는 기존 pkl(raw 컬럼 없음)에 대해서도 fallback 동작이 보장된다.

---

## 2. 토큰 구조 설계

### 2.1 토큰 정의

하나의 세그먼트 토큰은 세 가지 컴포넌트로 구성된다.

```
SegmentToken = {
    hi:   float32 (64,)    — HI 피처 (leakage 제외, z-score 정규화)
    raw:  float32 (2, 48)  — [raw_v; raw_i]  q-정규화 그리드, 채널-first
    meta: {
        cell_id:   str       — 저장용, 모델 미사용 (현행 유지)
        dataset:   str
        cycle:     int
        seg_idx:   int       — 시나리오 인덱스 (0-5)
        direction: float     — +1.0(충전) / -1.0(방전)
        level:     int       — 잠재 클래스 (0/1/2)
    }
}
```

### 2.2 원시 곡선 리샘플링 규칙

`build_dataset.py`의 `_resample` 함수와 동일한 방식을 사용한다:

```python
RAW_N = 48          # 리샘플 포인트 수 (참조 파일과 동일)

def _resample_segment(v, ai, q):
    """세그먼트 배열 → q-정규화 48pt 곡선.
    q: 누적 전하량 (세그먼트 내 상대값, 시작=0)
    반환: raw_v (48,), raw_i (48,) — float32"""
    q_tot = q[-1]
    if q_tot < 1e-6:
        return np.zeros(RAW_N, np.float32), np.zeros(RAW_N, np.float32)
    q_norm = q / q_tot                           # [0, 1] 정규화
    grid   = np.linspace(0.0, 1.0, RAW_N)
    rv     = np.interp(grid, q_norm, v).astype(np.float32)
    ri     = np.interp(grid, q_norm, ai).astype(np.float32)
    return rv, ri
```

**근거**: q_frac 정규화 그리드를 사용하면 세그먼트 길이(n2)가 달라도 동일 해상도로 비교 가능하다.  
시간 축(t)은 전류 적분으로 파생되므로 별도 저장하지 않는다.

### 2.3 메타데이터 처리 방침

현행과 동일: 모델에 직접 입력하지 않고 데이터셋에 저장만 한다.  
추후 cross-dataset 전이학습, 셀별 특성 분석 등에 활용 가능.

---

## 3. 데이터 파이프라인 변경

### 3.1 세그먼트 생성 단계 (`4_hi_analysis/hi_correlation.py`)

**현재**: 세그먼트 자르기 → HI 계산 → `{hi_feat, cell_id, cycle, segment_id, capacity_Ah, scen}` 저장  
**변경**: 세그먼트 자르기 → HI 계산 + **원시 리샘플** → `raw_v`, `raw_i` 컬럼 추가 저장

#### 변경 위치

`hi_correlation.py`의 각 세그먼트를 처리하는 루프 (HI dict를 조립하는 지점):

```python
# 추가할 코드 (기존 HI dict 조립 직후)
from hi_schema import RAW_N   # 상수 import 추가

rv, ri = _resample_segment(seg_v, seg_ai, seg_q_rel)   # (48,), (48,)
row["raw_v"] = rv.tolist()     # JSON-serializable → pkl 저장 시 list 유지
row["raw_i"] = ri.tolist()
```

PKL 파일 컬럼 구조 (native seg format 신버전):

| 그룹 | 컬럼명 | shape | dtype |
|------|--------|-------|-------|
| 메타 | cell_id, cycle, segment_id, capacity_Ah, scen | scalar | 기존 유지 |
| HI | stat_*, diff_*, lfp_*, morph_* | scalar × 66 | float32 |
| **원시 (신규)** | **raw_v, raw_i** | list[48] | float32 |

**하위 호환**: `raw_v`/`raw_i` 컬럼이 없는 구 pkl은 로더에서 자동 fallback(zero 패딩).  
기존 pkl 재생성 없이 모델의 raw=off 모드로 계속 실험 가능.

#### 수정 파일 요약

- `4_hi_analysis/hi_correlation.py` — 세그먼트 루프에 리샘플 호출 + pkl 저장 컬럼 추가
- `4_hi_analysis/hi_segment_viz.py` — (변경 없음, raw 컬럼은 무시)
- `5_model/utils/hi_schema.py` — `RAW_N = 48` 상수 추가

### 3.2 데이터셋 로더 (`5_model/datasets/segment_dataset.py`)

#### `load_dataset_native_seg` 수정

```python
# raw_v / raw_i 로드 (없으면 zero-fallback)
RAW_N = 48
has_raw = "raw_v" in df.columns and "raw_i" in df.columns
if has_raw:
    raw_v = np.stack(df["raw_v"].apply(
        lambda x: np.array(x, np.float32) if isinstance(x, (list,np.ndarray))
                  else np.zeros(RAW_N, np.float32)
    ).values)   # (N, 48)
    raw_i = np.stack(df["raw_i"].apply(...).values)
else:
    raw_v = np.zeros((len(df), RAW_N), np.float32)
    raw_i = np.zeros((len(df), RAW_N), np.float32)
df["_raw_v"] = list(raw_v)
df["_raw_i"] = list(raw_i)
```

#### `SegmentDataset.__init__` 수정

기존 텐서에 `x_raw` 추가:

```python
# 기존 텐서 (변경 없음)
self.x_hi      = torch.from_numpy(x)             # (N, 64)
self.nan_mask  = torch.from_numpy(mask)           # (N, 64)
self.direction = ...
...

# 신규 텐서
raw_v = np.stack(df["_raw_v"].values)  # (N, 48)
raw_i = np.stack(df["_raw_i"].values)  # (N, 48)
raw   = np.stack([raw_v, raw_i], axis=1)           # (N, 2, 48)
self.x_raw = torch.from_numpy(raw)                 # (N, 2, 48) float32
```

#### `__getitem__` 수정

```python
def __getitem__(self, idx):
    return {
        "x_hi":      self.x_hi[idx],       # (64,)
        "x_raw":     self.x_raw[idx],       # (2, 48)  ← 신규
        "nan_mask":  self.nan_mask[idx],    # (64,)
        "direction": self.direction[idx],   # scalar
        "level":     self.level[idx],       # scalar
        "seg_idx":   self.seg_idx[idx],     # scalar
        "target":    self.target[idx],      # scalar
        "cap_init":  self.cap_init[idx],    # scalar
    }
```

#### `_subset_dataset` 수정

```python
new_ds.x_raw = ds.x_raw[indices]   # 추가
```

#### `collate_fn`

변경 없음 — `{k: torch.stack([b[k] for b in batch]) for k in keys}` 가 `x_raw`도 자동 처리.

---

## 4. CNN 분류기 설계

### 4.1 아키텍처 결정: Residual 1D CNN + Attention Pooling

단순 conv → avg-pool 대신 **잔차 블록 + Attention Pooling**을 제안한다.

**근거**:
- LFP 세그먼트의 V-Q 곡선은 국부적 특징(플래토 진입/이탈 기울기, knee point)이 SOH에 민감하다
- 단순 avg-pool은 이 국부 특징을 희석시킨다
- Attention Pooling은 어느 q_frac 위치가 중요한지 학습한다 → HI와 상보적

#### `RawCNN` 모듈 (`5_model/models/raw_cnn.py` 신규)

```
입력: (B, 2, 48)  — 채널=[V, |I|], 길이=48

Conv1d(2→32, k=7, pad=3, groups=1) → BatchNorm → GELU
MaxPool1d(2) → (B, 32, 24)

ResBlock1:
  Conv1d(32→32, k=3, pad=1) → BN → GELU → Conv1d(32→32, k=3, pad=1) → BN
  + skip → GELU
  → (B, 32, 24)

Conv1d(32→64, k=3, pad=1, stride=2) → BN → GELU → (B, 64, 12)

ResBlock2:
  Conv1d(64→64, k=3, pad=1) → BN → GELU → Conv1d(64→64, k=3, pad=1) → BN
  + skip → GELU
  → (B, 64, 12)

AttentionPool1d:
  score = Linear(64, 1)(각 위치) → softmax over 12
  output = weighted sum → (B, 64)

Linear(64, D_CNN=64) → GELU → (B, 64)
```

```python
class AttentionPool1d(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)
    def forward(self, x):   # x: (B, C, L)
        x_t = x.transpose(1, 2)                  # (B, L, C)
        w   = self.score(x_t).softmax(dim=1)     # (B, L, 1)
        return (x_t * w).sum(dim=1)              # (B, C)
```

출력 차원: **D_CNN = 64** (HI 64D와 동일하게 맞춤)

#### 대안 고려 (멀티스케일 CNN)

```
Multi-scale parallel branches:
  branch_k3: Conv1d(2→16, k=3) → BN → GELU → GlobalAvgPool → (B, 16)
  branch_k7: Conv1d(2→16, k=7) → BN → GELU → GlobalAvgPool → (B, 16)
  branch_k15: Conv1d(2→16, k=15) → BN → GELU → GlobalAvgPool → (B, 16)
  dual: Conv1d_V + Conv1d_I (separate branches for V and I channels)
concat → Linear(64, 64)
```

멀티스케일은 구현이 단순하고 병렬화가 쉽다. 단, 각 브랜치가 독립적으로 학습되어 세그먼트 내 위치 관계(연속성)를 무시한다는 단점이 있다. **Residual + Attention Pooling을 기본 설계안으로 권장**하되, 실험 구현 시 멀티스케일을 baseline으로 비교 가능.

### 4.2 `CNNProbeClassifier` (`5_model/models/scenario_classifier.py` 수정)

```python
class CNNProbeClassifier(ScenarioClassifier, nn.Module):
    """
    CNN(raw V/I) + HI probe → n_classes 분류기.
    batch["x_raw"] (B, 2, 48) 필수.
    """
    def __init__(self, n_hi, n_classes, d_cnn=64, d_hidden=128, dropout=0.1):
        nn.Module.__init__(self)
        self.cnn = RawCNN(d_out=d_cnn)
        # 융합: CNN_emb(64) + probe_x(64) + direction(1) = 129D
        self.head = nn.Sequential(
            nn.Linear(d_cnn + n_hi + 1, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, n_classes),
        )

    def forward(self, x):        # for nn.Module interface
        return x                 # not used directly

    def classify(self, probe_x, batch):
        """
        probe_x : (B, 64)
        batch   : must contain x_raw (B, 2, 48) and direction (B,)
        """
        cnn_emb   = self.cnn(batch["x_raw"])           # (B, 64)
        direction = batch["direction"].unsqueeze(1)    # (B, 1)
        fused     = torch.cat([cnn_emb, probe_x, direction], dim=1)  # (B, 129)
        return self.head(fused)                         # (B, n_classes)
```

#### 팩토리 등록 (`build_classifier` 함수 수정)

```python
_REGISTRY["cnn"] = CNNProbeClassifier

# build_classifier 분기 추가:
if key == "cnn":
    return CNNProbeClassifier(n_hi, n_classes, d_cnn=64, d_hidden=d_hidden, dropout=dropout)
```

#### `train_classifier.py` 수정

```python
# MLPProbeClassifier 학습 루프 내 inp 구성 부분 교체
if isinstance(clf, CNNProbeClassifier):
    logits = clf.classify(probe, batch)   # batch에 x_raw 포함
else:
    inp    = torch.cat([probe, dir_t.unsqueeze(1)], dim=1)
    logits = clf(inp)
```

### 4.3 평가 시 분류기 활용 — hard / soft 라우팅 (2026-07 추가)

`classifier.type`(mlp/cnn)이 **분류기 자체의 아키텍처**를 고르는 것과 달리,
**hard/soft는 학습이 끝난 뒤 평가 시점에 그 분류기 출력을 어떻게 쓸지** 결정하는
축이다 — 학습 로직(`train_classifier.py`)에는 hard/soft 개념이 전혀 없고,
`CrossEntropyLoss`로 분류기 1개만 학습한다. hard/soft는 `test_scr.py` 평가
단계에서만 등장한다 (§ PIPELINE.md 5-10/5-11).

SCRModel의 회귀망(cap_head)은 **단 하나**뿐이다. 시나리오마다 다른 신경망이
있는 게 아니라, 시나리오별로 **어떤 HI를 통과시킬지 고르는 게이트(scen_gate)**가
다르다:

```
x_hi (64개 HI)
   │
   ├─ scen_gates[0] (chg_lo용 게이트) ─┐
   ├─ scen_gates[1] (chg_mid용 게이트) ─┤
   ├─ ...                              ├─→ 선택된 seg_idx의 게이트만 통과
   └─ scen_gates[5] (dis_lo용 게이트) ─┘
                    │
              x_scen (게이트 통과 후 HI)
                    │
        [x_probe | x_scen | dir | cap_init]
                    │
              cap_head (공유, 단 하나)  ← 이 신경망 자체는 hard/soft 동일
                    │
                cap_pred
```

hard/soft가 결정하는 것은 **"어느 시나리오 게이트로 회귀 입력을 구성할지"**다:

| | 어떤 scen_gate를 쓰나 | cap_head는 몇 번 도나 |
|---|---|---|
| **hard** | 분류기 argmax로 고른 시나리오 게이트 **1개**만 적용 | 1회 (선택된 게이트 결과로만 예측) |
| **soft** | **n_classes개**(lo/mid/hi 3개) 시나리오 게이트를 전부 적용 | 3회 (각각 예측 후 softmax 확률로 가중평균) |
| **oracle** | 정답 seg_idx로 게이트 **1개** 고정 (분류기 우회) | 1회 |

즉 "어느 회귀 헤드를 쓸지"보다는 **"어느 시나리오의 HI 서브셋(게이트)으로 회귀
입력을 구성할지"** 결정이라고 보는 게 더 정확하다. `cap_head` 신경망 가중치는
세 모드 무관하게 완전히 동일한 것을 쓴다 — hard/oracle은 그 신경망에 입력 1개만
넣고, soft는 후보 입력 3개를 넣어 나온 3개 예측을 분류기 확신도(softmax
확률)로 블렌딩한다.

**분류 판정(`level_pred`)은 hard/soft에서 항상 동일** — 둘 다
`clf_logits.argmax(1)`을 분류 판정으로 쓴다. 차이는 회귀(`cap_pred`) 계산
방식에만 있다. 따라서 `confusion_matrix_test_hard.png`와
`confusion_matrix_test_soft.png`가 완전히 같은 건 정상이며(분류 판정이 같으므로),
회귀 산포도(`scatter_test_hard.png` vs `scatter_test_soft.png`)는 서로 달라야
정상이다.

---

## 5. 회귀 모델 CNN 적용 검토

### 5.1 적용 타당성

| 항목 | 분석 |
|------|------|
| **잠재 이득** | HUST chg_hi/mid처럼 HI만으로는 신호가 약한 구간에서 원시 V-Q 형상이 직접 SOH 정보를 제공할 가능성 |
| **중복성** | HI 64개가 이미 V/I에서 유도된 통계량 — 완전 중복은 아니나 상당 부분 겹침 |
| **연산 비용** | Phase 2 학습 시 배치마다 CNN forward 추가 (RAW_N=48로 경량) |
| **Phase 분리** | Phase 1(probe gate 학습)과 Phase 2(scenario gate + cap_head)를 모두 수정해야 함 |

### 5.2 권장 적용 방식 — 공유 CNN 백본

분류기와 회귀 헤드가 **동일한 `RawCNN` 인스턴스를 공유**하는 방식이 가장 효율적이다.

```
[x_raw (B,2,48)]
       │
   RawCNN (shared)
       │
  cnn_emb (B,64)
       ├──────────────────────────────────┐
       │                                  │
 [probe_x | scen_x | dir | cap_init]   [probe_x | dir]
  (B, 64+64+1+1 = 130)  + cnn_emb(64)   (B, 64+64+1)
  → cap_head (B, 194→SOH)               → classifier (B, n_classes)
```

`SCRModel` 수정 포인트:

```python
# __init__ 에 추가
if with_raw_cnn:
    self.raw_cnn = RawCNN(d_out=64)      # 공유 백본
else:
    self.raw_cnn = None

# forward 에서
if self.raw_cnn is not None:
    cnn_emb = self.raw_cnn(batch["x_raw"])   # (B, 64)
    feat = torch.cat([probe_x, scen_x, cnn_emb,
                      direction.unsqueeze(1),
                      batch["cap_init"].unsqueeze(1)], dim=1)   # (B, 194)
else:
    feat = torch.cat([probe_x, scen_x,
                      direction.unsqueeze(1),
                      batch["cap_init"].unsqueeze(1)], dim=1)   # (B, 130)
```

`cap_heads.py`의 `_HEAD_IN` 처리:

```python
_HEAD_IN_NO_CNN  = N_HI + N_HI + 1 + 1   # 130 (기존)
_HEAD_IN_WITH_CNN = N_HI + N_HI + 64 + 1 + 1  # 194 (CNN 적용)

def build_cap_head(model_cfg, d_head=128, dropout=0.1, with_raw_cnn=False):
    in_d = _HEAD_IN_WITH_CNN if with_raw_cnn else _HEAD_IN_NO_CNN
    ...
```

### 5.3 권장 결론

**1단계**: 분류기에만 CNN 적용 (요구사항 2)  
**2단계**: HUST chg_hi/mid 등 HI 신호 약한 시나리오 개선이 필요하다면 회귀 헤드에도 공유 CNN 확장  

HUST 진단(현재 분석 중)에서 chg_hi/mid의 평탄 예측 원인이 원시 곡선 신호 부재가 아닌 학습 수렴 문제로 밝혀질 경우, 회귀 CNN 적용의 효과가 제한될 수 있다. 따라서 분류기 CNN부터 적용하고 실험 결과를 보고 회귀 CNN 여부를 결정한다.

---

## 6. 파일별 수정 목록

### 신규 생성

| 파일 | 내용 |
|------|------|
| `5_model/models/raw_cnn.py` | `RawCNN` 클래스 (ResBlock + AttentionPool) |

### 수정 파일

| 파일 | 수정 내용 |
|------|---------|
| `5_model/utils/hi_schema.py` | `RAW_N = 48` 상수 추가 |
| `4_hi_analysis/hi_correlation.py` | 세그먼트 루프에서 `raw_v`/`raw_i` 계산 후 pkl에 저장 |
| `5_model/datasets/segment_dataset.py` | `raw_v`/`raw_i` 로드 → `x_raw (2,48)` 텐서, `__getitem__`/`_subset_dataset` 업데이트 |
| `5_model/models/scenario_classifier.py` | `CNNProbeClassifier` 클래스 추가, `build_classifier` 팩토리에 `"cnn"` 등록 |
| `5_model/train_classifier.py` | `CNNProbeClassifier` 분기 처리, `batch["x_raw"]` 전달 |
| `5_model/models/scr_model.py` | (선택) `with_raw_cnn` 플래그, `RawCNN` 통합 |
| `5_model/models/cap_heads.py` | (선택) `with_raw_cnn` 파라미터, `_HEAD_IN` 동적 계산 |
| `5_model/train_scr.py` | (선택) `with_raw_cnn` yaml 파라미터 반영 |
| `5_model/config/scr.yaml` | `classifier: cnn` 옵션 추가, (선택) `with_raw_cnn: false` |

### 변경 없음 (호환 유지)

| 파일 | 이유 |
|------|------|
| `common/scenario/*.py` | 시나리오 축 정의 — 토큰 구조와 무관 |
| `5_model/training/scr_trainer.py` | 배치 dict 키 추가에 자동 대응 |
| `5_model/training/scr_loss.py` | 손실 계산은 cap_pred / level_logits 기반 |
| `5_model/evaluation/scr_evaluator.py` | 평가 지표는 출력 기반 |
| `run_pipeline.py` | 파이프라인 진입점 — 내부 변경 불필요 |

---

## 7. 단계별 구현 순서

```
[Step A] 상수 정의
  hi_schema.py에 RAW_N = 48 추가

[Step B] 원시 곡선 저장 (데이터 재생성 필요)
  hi_correlation.py: _resample_segment 함수 추가 + row에 raw_v/raw_i 저장
  → 재실행: python run_pipeline.py 4 (hi_correlation step만)

[Step C] 데이터셋 로더 업데이트
  segment_dataset.py: x_raw 텐서 추가
  → 기존 pkl(raw 없음): zero-fallback으로 기존 실험 재현 가능

[Step D] RawCNN 모듈 생성
  5_model/models/raw_cnn.py 신규 작성
  → 단독 테스트: python -c "from models.raw_cnn import RawCNN; ..."

[Step E] CNNProbeClassifier 추가
  scenario_classifier.py에 클래스 추가 및 팩토리 등록
  train_classifier.py 분기 처리

[Step F] (선택) 회귀 헤드 CNN 통합
  scr_model.py + cap_heads.py 수정
  scr.yaml에 with_raw_cnn: true 설정

[Step G] 실험 검증
  기존 run으로 MLP baseline 재현 → CNN 분류기 비교
```

---

## 8. 호환성 체크리스트

| 체크 | 내용 | 방법 |
|------|------|------|
| ✅ 구 pkl 호환 | raw_v/raw_i 없는 기존 pkl → zero x_raw | `has_raw = "raw_v" in df.columns` fallback |
| ✅ 기존 MLP 분류기 유지 | `build_classifier("mlp", ...)` 변경 없음 | 기존 yaml classifier: mlp |
| ✅ 기존 Phase 1/2 학습 | x_raw가 batch에 있어도 SCRModel이 무시 | `if self.raw_cnn is not None:` 조건 |
| ✅ collate_fn | `torch.stack` 자동 처리 | x_raw shape (2,48) → 배치 (B,2,48) |
| ✅ _subset_dataset | x_raw 인덱싱 추가 | `new_ds.x_raw = ds.x_raw[indices]` |
| ✅ scr_evaluator | 출력 기반 평가 — 입력 변경 투명 | 변경 없음 |
| ⚠️ test_scr.py | x_raw를 배치에 포함해 모델에 전달하는지 확인 필요 | `SegmentDataset.__getitem__` 변경 자동 반영 |

---

## 9. 구현 결과 (2026-07-22 반영)

설계안 §1 요구사항 1·2를 반영 완료. 회귀 헤드 CNN(§5, Step F)은 실험 결과를 보고
결정하도록 **미적용(선택 사항)**으로 남겨둠. 시간 단위 창 시나리오 개념도 보류(설계 유지).

### 9.1 변경된 파일

| 파일 | 변경 내용 | 상태 |
|------|---------|------|
| `5_model/utils/hi_schema.py` | `RAW_N=48`, `RAW_CH=2` 상수 추가 | ✅ |
| `4_hi_analysis/hi_correlation.py` | `_resample_segment()` 추가 + 충/방전 세그먼트 루프에서 `raw_v_{seg}`/`raw_i_{seg}` 계산·저장, `_to_seg_df`가 raw 컬럼을 seg 포맷으로 전달 | ✅ |
| `5_model/datasets/segment_dataset.py` | `_stack_raw_col`/`_build_raw_tensor` 헬퍼, `SegmentDataset.x_raw (N,2,48)`, `__getitem__`·`_subset_dataset`·native 로더 `keep`에 raw 반영 (zero-fallback) | ✅ |
| `5_model/models/raw_cnn.py` | **신규** — `RawCNN`(stem→ResBlock×2→AttentionPool→Linear, 64D, 42.4K params) | ✅ |
| `5_model/models/scenario_classifier.py` | `CNNProbeClassifier` 추가, 팩토리 `"cnn"` 등록 | ✅ |
| `5_model/train_classifier.py` | `classifier.type`(mlp/cnn) 분기, 체크포인트에 `clf_type` 저장 | ✅ |
| `5_model/test_scr.py` | `clf_type` 기반 분류기 복원 분기 | ✅ |
| `5_model/evaluation/scr_evaluator.py` | routing 시 `isinstance(CNNProbeClassifier)` 분기로 `classify(probe_x, batch)` 호출 | ✅ |
| `5_model/config/scr.yaml` | `classifier.type: mlp` 키 추가 | ✅ |

### 9.2 최종 설계와 달라진 점

- **토큰 raw 저장 방식**: 설계에선 로더에서 `_raw_v`/`_raw_i` 임시 컬럼을 언급했으나,
  실제로는 `_build_raw_tensor(df)`가 `raw_v`/`raw_i` object 컬럼을 직접 `(N,2,48)` 텐서로
  변환하도록 단순화. 임시 컬럼 불필요.
- **CNN 분류기 `n_hi` 의미**: MLP는 `[probe_x∥direction]=N_HI+1` 입력이라 체크포인트
  `n_hi=65`, CNN은 direction을 내부에서 concat하므로 `n_hi=N_HI=64`. 체크포인트 `clf_type`으로
  구분해 복원.
- **구 pkl degrade 경로**: raw 컬럼 없는 기존 pkl은 `x_raw`가 전부 0 → CNN 임베딩이 상수에
  수렴, 분류는 `probe_x`만으로 동작(무해). 예외처리 없이 자연 degrade 확인.

### 9.3 검증 결과 (env: `LFP_SOH_ESTIMATION`, torch 1.12.1)

| 테스트 | 결과 |
|--------|------|
| `_resample_segment` 가변 길이(2·137·2801) → 항상 (48,) | ✅ |
| RawCNN forward (B,2,48)→(B,64), 42,401 params | ✅ |
| `build_classifier("cnn"/"mlp")` + `classify()` → (B,n_classes) | ✅ |
| **구 pkl(raw 없음)** `build_datasets` → x_raw 전부 0, 1.46M seg 로드, collate (B,2,48), CNN forward finite | ✅ |
| 실제 clean 셀 세그먼트(길이 33~79) → 모두 (48,) 리샘플, `_to_seg_df` raw 컬럼 보존(6 seg) | ✅ |
| SCRModel forward가 `x_raw` 키 무시 (회귀 경로 불변) | ✅ |

### 9.4 사용 방법

```yaml
# 1) 데이터 재생성 (raw_v/raw_i 컬럼 포함된 seg pkl 생성)
#    python run_pipeline.py 4 --seg-axis q_frac_wide --axis-config '{...}'
# 2) CNN 분류기 사용
classifier:
  type: cnn      # mlp → cnn 으로 변경
```

기존 pkl 그대로 `type: cnn`을 써도 동작하나(x_raw=0), CNN 이점을 얻으려면 Step 4를
재실행해 raw 컬럼이 포함된 seg pkl을 생성해야 함.

### 9.5 남은 작업 (선택)

- 회귀 헤드 공유 CNN 통합(§5.2): `SCRModel.with_raw_cnn` + `cap_heads._HEAD_IN` 동적화.
  HUST chg_hi/mid 평탄 예측 진단 결과에 따라 착수 여부 결정.
- CNN 분류기 학습 후 E3 routing=hard 실험으로 MLP 대비 분류 정확도·SOH RMSE 비교.
