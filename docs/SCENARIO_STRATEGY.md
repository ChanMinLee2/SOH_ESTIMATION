# SCENARIO_STRATEGY — 세그먼트 경계 축 설계

## 0. 배경: 세그먼트 경계 기준 선택 문제

SOH 예측을 위한 사이클 분할에서 경계 기준이 중요한 이유는 두 가지다.

**[1] 배포(inference) 시 seg_idx 취득 방법**  
q_frac 기반 분할은 학습 시점에 ground truth capacity를 사용해 `q_frac = q_cum / q_total`로 경계를 계산한다 — 이것은 순환성이 아니다. 다만 실제 배포에서는 q_total이 미지수이므로 Stage A 분류기가 HI 피처로부터 "이 세그먼트가 어느 구간인가"를 추론해야 한다. 전압·프로토콜 기반 경계는 이 추론 없이 rule-based로 seg_idx를 결정한다.

**[2] 경계의 전기화학적 정렬**  
경계가 CC/CV 전환점이나 LFP 플래토 전이 전압과 일치하면 동일 시나리오 내 HI 분포가 균질해지고, 시나리오 게이트가 더 일관된 회귀 신호를 학습한다.

| 후보 | 특성 | 리뷰어 관점 |
|------|------|------------|
| `q_frac` (공칭용량 대비 적산 전하량) | 구현 단순, 경계 고정 | 경계 선택 근거 없음 (임의 40%/30%/30%) |
| `protocol` (CC 전환점) | 프로토콜 구조와 1:1 대응 | 데이터셋 간 전환 C-rate가 다름 → 크로스 DS 한계 |
| `vwindow` (고정 전압 경계) | 전기화학 상전이 기반, 열화 불변 | LFP 화학계에 특화됨 |
| `rcs` (랜덤 윈도우) | 경계 가정 없음 | Deng 계열 재현, 물리 해석 불가 |

**현재 기본값**: `qfrac` 축, 경계 `[0.0, 0.4, 0.7, 1.0]`

이 문서는 기본 qfrac 분할과 함께 제공되는 4가지 대체 축(protocol / vwindow / rcs / cluster)의 설계 근거와 실행 방법을 정리한다.

---

## 1. 데이터셋 구조 요약

### MIT (Severson 2019)

**충전** — 셀마다 상이 (72가지 프로토콜):

| q_frac 구간 | SOC 구간 | 프로토콜 특성 | 시나리오 정보 |
|-------------|----------|--------------|--------------|
| [0.0, 0.8) | 0–80% | 1-step 또는 2-step CC (72가지 조합) | **프로토콜 다양 → 시나리오 분류 유효** |
| [0.8, 1.0] | 80–100% | 전 셀 동일: 1C CC → 3.6V → CV | 균일 → 시나리오 분류 불필요 |

**방전** — 전 셀 동일:

| q_frac 구간 | SOC 구간 | 프로토콜 특성 | 시나리오 정보 |
|-------------|----------|--------------|--------------|
| [0.0, 1.0] | 100%→2V | 전 셀 동일: 4C CC → 2.0V | 균일 → 시나리오 분류 불필요 |

### HUST (Ma et al. 2023, Table S1 / Fig. S2)

**충전** — 전 셀 동일 (Fig. S2):

| q_frac 구간 | SOC 구간 | 프로토콜 특성 | 시나리오 정보 |
|-------------|----------|--------------|--------------|
| [0.0, ~0.8) | 0–80% | 전 셀 동일: 5C CC | 균일 → 시나리오 분류 불필요 |
| [~0.8, 1.0] | 80–100% | 전 셀 동일: 1C CC → 3.6V → CV | 균일 → 시나리오 분류 불필요 |

**방전** — 77개 프로토콜 다양 (Table S1 확인, C4는 예외):

| q_frac 구간 | SOC 구간 | 프로토콜 단계 | 프로토콜 특성 | 시나리오 정보 |
|-------------|----------|--------------|--------------|--------------|
| [0.0, 0.4) | 100%→60% | **C1 단계** | 2C / 3C / 4C / 5C 중 하나 | **프로토콜 다양 → 시나리오 분류 유효** |
| [0.4, 0.6) | 60%→40% | **C2 단계** | 1C–5C 다양 | **프로토콜 다양 → 시나리오 분류 유효** |
| [0.6, 0.8) | 40%→20% | **C3 단계** | 1C–5C 다양 | **프로토콜 다양 → 시나리오 분류 유효** |
| [0.8, 1.0] | 20%→2V | **C4 단계** | **전 셀 동일: 1C 고정** | 균일 → 시나리오 분류 불필요 |

> **핵심**: Table S1 기준 77개 프로토콜 모두 C4=1C로 고정됨. C1/C2/C3만 변함.  
> HUST 방전 경계 [0.0, **0.40**, **0.60**, **0.80**, 1.0]은 논문에 직접 명시된 물리적 경계.

---

## 2. 균일 세그먼트 처리 원칙

> 시나리오 분류 헤드는 프로토콜 차이를 감지하는 것이 목적이다.  
> 전 셀이 동일한 프로토콜인 구간은 분류 학습에 노이즈만 추가한다.

### 균일 vs 정보 세그먼트 정리

| 세그먼트 | MIT | HUST | 시나리오 분류 | 용량 회귀 |
|----------|-----|------|--------------|----------|
| chg_lo (0–40%) | 정보 | 균일 | MIT만 유효 | 양쪽 유지 |
| chg_mid (40–70%) | 정보 | 균일 | MIT만 유효 | 양쪽 유지 |
| chg_hi (70–100%) | 균일 | 균일 | **공통 균일** | 양쪽 유지 |
| dis_hi (0–40%) | 균일 | 정보 | HUST만 유효 | 양쪽 유지 |
| dis_mid (40–70%) | 균일 | 정보 | HUST만 유효 | 양쪽 유지 |
| dis_lo (70–100%) | 균일 | 정보 | HUST만 유효 | 양쪽 유지 |

> 현재 구현에서 "균일 세그먼트 분류 손실 마스킹"은 명시적으로 분리되지 않는다.  
> 프로토콜 기반 축(`protocol`)을 사용하면 균일 CC 구간이 자연스럽게 통합되어 단일 세그먼트로 처리된다.

---

## 3. 세분화 수준 가이드라인

### 세분화 깊이의 트레이드오프

| 분할 수 | 장점 | 단점 |
|---------|------|------|
| 너무 적음 (≤2구간) | 구현 단순, 오버피팅 위험 낮음 | 균일/정보 구간 분리 불가, 물리 해석 불가 |
| **최적 범위 (3–4구간)** | 프로토콜 단계·전기화학 phase 1:1 대응, 데이터 충분 | — |
| 너무 많음 (≥6구간) | 이론상 세밀한 정보 | 짧은 구간 → HI 통계 불안정, 모델 복잡도 급증 |

### 최소 구간 크기 기준

A123 APR18650M1A 셀 기준 (공칭 용량 1.1–1.2Ah, 5mAh bin):

| 구간 비율 | 최소 빈 수 | 안정성 |
|----------|-----------|--------|
| 10% (110–120mAh) | ~22 bins | 마지노선 — 일부 HI 통계 불안정 |
| **20% (220–240mAh)** | **~44 bins** | **실용적 최소 (권장)** |
| 30%+ | 66+ bins | 안정적 |

→ qfrac 기본 경계 30% 구간 → 충분.  
→ 10% 미만 구간 분할은 HI 안정성 보장 불가 — 실험적 검증 없이는 사용 금지.

---

## 4. 구현된 5가지 축

모든 축은 `common/scenario/` 패키지 내 `Segmenter` ABC를 구현한다.

### 공통 인터페이스

```python
from common.scenario import get_segmenter

seg = get_segmenter("qfrac")                        # 기본값
seg = get_segmenter("protocol", cfg={"protocol": {"max_steps": 3}})
seg = get_segmenter("vwindow",  cfg={"vwindow":  {"n_windows": 3}})
seg = get_segmenter("rcs",      cfg={"rcs":      {"n_samples": 6, "window": 0.3}})
seg = get_segmenter("cluster",  cfg={"cluster":  {"n_fine": 10, "k_range": [2, 8]}})

spec: ScenarioSpec = seg.get_spec()   # ScenarioSpec 반환
```

**`ScenarioSpec`** — Step 4 출력 / Step 5 입력 계약:
```python
@dataclass
class ScenarioSpec:
    axis: str               # "qfrac" | "protocol" | "vwindow" | "rcs" | "cluster"
    n_scenarios: int        # Stage B gate 개수
    scenario_names: list    # 길이 = n_scenarios
    n_classes: int          # Stage A 분류 클래스 수
    class_names: list       # 길이 = n_classes  (예: ["lo", "mid", "hi"])
    routing: list[list[int]]# routing[dir_idx][latent_class] = scenario_id
    classifier_default: str # "mlp_probe" | "rule" | "centroid" | "none"
    params: dict            # 축별 재현 파라미터 (bounds, n_windows 등)
```

---

### 축 0: `qfrac` — q_frac 3분할 (현행 기본값)

> 구현 파일: `common/scenario/qfrac.py`

#### 경계 및 시나리오 구성

| 세그먼트 | 방향 | q_frac 구간 | scenario_id | latent_class |
|---------|------|------------|-------------|-------------|
| `chg_lo`  | 충전 | [0.0, 0.4) | 0 | lo(0) |
| `chg_mid` | 충전 | [0.4, 0.7) | 1 | mid(1) |
| `chg_hi`  | 충전 | [0.7, 1.0] | 2 | hi(2) |
| `dis_hi`  | 방전 | [0.0, 0.4) | 3 | hi(2) |
| `dis_mid` | 방전 | [0.4, 0.7) | 4 | mid(1) |
| `dis_lo`  | 방전 | [0.7, 1.0] | 5 | lo(0) |

라우팅 테이블: `routing = [[0, 1, 2], [5, 4, 3]]`  
→ 충전(dir_idx=0): lo→chg_lo(0), mid→chg_mid(1), hi→chg_hi(2)  
→ 방전(dir_idx=1): lo→dis_lo(5), mid→dis_mid(4), hi→dis_hi(3)

#### 파라미터 커스터마이즈

```powershell
# 기본값 그대로 실행
python 4_hi_analysis/hi_correlation.py --seg-axis qfrac

# 방전 분할점 변경 (기본: [0.0, 0.4, 0.7, 1.0])
python 4_hi_analysis/hi_correlation.py --seg-axis qfrac \
  --axis-config '{"dis_bounds":[0.0,0.4,0.7,1.0],"chg_bounds":[0.0,0.4,0.7,1.0]}'
```

#### ScenarioSpec 예시

```json
{
  "axis": "qfrac",
  "n_scenarios": 6,
  "scenario_names": ["chg_lo","chg_mid","chg_hi","dis_hi","dis_mid","dis_lo"],
  "n_classes": 3,
  "class_names": ["lo","mid","hi"],
  "routing": [[0,1,2],[5,4,3]],
  "classifier_default": "mlp_probe",
  "params": {"dis_bounds":[0.0,0.4,0.7,1.0],"chg_bounds":[0.0,0.4,0.7,1.0]}
}
```

#### 장단점

| 장점 | 단점 |
|------|------|
| 현행 동작 완전 재현, 기준선(baseline) 역할 | 경계에 물리적 근거 없음 (임의 30%/30%/40% 분할) |
| 가장 빠른 구현·실험 | 열화로 실제 SOC가 변해도 q_frac 경계는 고정 |

---

### 축 1: `protocol` — CC 단계 전환 기반 세그멘터

> 구현 파일: `common/scenario/protocol.py`

MIT 2단 충전(`C1(SOC1%)-C2`) + CV 꼬리, HUST 다단 방전을 **`|ΔI| > i_step_thresh` 검출**로 자동 분해.

#### 핵심 특성

- CC 단계 전환점: 인접 전류 차이 `|ΔI| > i_step_thresh_c × nom_cap`
- CV 구간 판별: V 포화(`v_mean/v_max > 0.98`) & I 단조 감소
- `max_steps` 파라미터로 n_scenarios 고정 (기본 3 → n_scenarios=6, qfrac과 동일)
- 분류기 기본값: `rule` (CC 단계 내 전류 통계로 결정론적 판별, 학습 불필요)

#### 데이터셋별 대응 및 실측 세그먼트 통계

**MIT (stats_protocol_MIT.txt)**:

| 시나리오 | 세그먼트 수 | 평균 행 수 | 물리적 의미 |
|---------|-----------|-----------|-----------|
| chg_step0 | 97,366 | 188.6 | C1 CC 구간 (SOC1 지점까지) |
| chg_step1 | 96,947 | 163.4 | C2 CC 구간 (SOC1→80%) |
| chg_step2 | 76,453 | 245.6 | 추가 CC 또는 CV 꼬리 |
| dis_step0 | 98,005 | 300.8 | 4C CC 방전 전반 |
| dis_step1 | 98,002 | 82.8 | 4C CC 방전 후반 (전류 프로파일 미세 변화) |
| **dis_step2** | **1** | 92.0 | ⚠️ 사실상 비어있음 — MIT 방전은 단일 CC |
| **합계** | **466,774** | — | — |

**HUST (stats_HUST.txt)**:

| 시나리오 | 세그먼트 수 | 평균 행 수 | 물리적 의미 |
|---------|-----------|-----------|-----------|
| chg_step0 | 146,039 | 109.9 | 5C CC 충전 구간 |
| chg_step1 | 146,039 | 234.2 | 1C CC→CV 구간 |
| **chg_step2** | **0** | — | ⚠️ HUST 충전은 2단 CC뿐 — 3번째 step 없음 |
| dis_step0 | 146,039 | 87.7 | C1 단계 (2/3/4/5C 중 하나) |
| dis_step1 | 145,981 | 88.4 | C2 단계 (다양) |
| dis_step2 | 128,287 | 79.7 | C3 단계 (일부 셀 없음) |
| **합계** | **712,385** | — | — |

> **핵심 구조 불일치**: MIT는 dis_step2가 1개(= 단일 CC 방전이라 step이 나뉘지 않음), HUST는 chg_step2가 0개(= 2단 CC만 존재). 즉 protocol 축의 `chg_step2`·`dis_step2` 게이트는 각 데이터셋에서 유효 데이터가 거의 없어 의미 없는 학습이 됨.

> ¹ MIT 충전 프로토콜 표기 `C1(SOC1%)-C2`: SOC1은 "1%"가 아니라 **첫 번째 CC 전환 SOC 값(%)** 을 뜻하는 첨자다.  
> 예) `6C(60%)-3C` → step0=6C로 0→60% 충전, step1=3C로 60→80% 충전.  
> 예) `8C(15%)-3.6C` → step0=8C로 0→15%만 충전, step1=3.6C로 15→80% 충전.  
> `ProtocolSegmenter`는 `|ΔI| > i_step_thresh` 로 이 전환점을 자동 감지한다.

#### 실행 방법

```powershell
# 기본값 (max_steps=3, i_step_thresh_c=0.1)
python 4_hi_analysis/hi_correlation.py --seg-axis protocol

# CC 전환 감도 조정
python 4_hi_analysis/hi_correlation.py --seg-axis protocol \
  --axis-config '{"max_steps":3,"i_step_thresh_c":0.15}'

# Step 5 학습 시 동일 축 지정
python 5_model/train_scr.py --phase 1 --seg-axis protocol
```

#### 장단점

| 장점 | 단점 |
|------|------|
| HUST C4(1C) 균일 구간 자연 분리 | 동일 C-rate에서 전류값 미세 차이로 오탐 가능 |
| MIT 2단 CC 전환점이 논문 프로토콜과 1:1 대응 | 1-step CC 셀(단일 구간)은 step0만 생성 → 세그먼트 수 불균일 |
| 레이블이 프로토콜 구조 직결 → 리뷰어 설득력 높음 | `max_steps` 초과 시 나머지 구간 병합 |

---

### 축 2: `vwindow` — 전압 윈도우 세그멘터

> 구현 파일: `common/scenario/vwindow.py`

방향별 고정 전압 경계(LFP 상전이 전압 기반)로 사이클을 분할하고, 충전 CC→CV 전환 이후는 독립 시나리오로 분리.

#### 시나리오 구조 (n_windows=3 기준, 총 7개)

| idx | 이름 | 방향 | 전압 범위 | 전기화학 의미 |
|-----|------|------|----------|--------------|
| 0 | `chg_win0` | 충전 | 2.80–3.38V | 사전-플래토 급경사 구간 |
| 1 | `chg_win1` | 충전 | 3.38–3.47V | 핵심 플래토 (~70% 용량) |
| 2 | `chg_win2` | 충전 | 3.47–3.60V | 플래토 탈출 구간 |
| 3 | `chg_cv`  | 충전 | 3.60V 이상 | CC→CV 전환 이후 (kinetic 체제) |
| 4 | `dis_win0` | 방전 | 2.50–3.10V | 플래토 종료 + 급강하 |
| 5 | `dis_win1` | 방전 | 3.10–3.35V | 하부 플래토 중심 |
| 6 | `dis_win2` | 방전 | 3.35–3.65V | 상부 플래토 + 숄더 |

`n_scenarios = 2·n_windows + 1 = 7` (chg_cv 포함), `n_classes = 4` (win0/win1/win2/cv)  
방전에는 CV 구간이 없으므로 방전 routing은 3개 entry만: `[4, 5, 6]`

#### 핵심 특성

- **LFP 고정 경계** (`from_lfp()`): `dis_edges=[2.50, 3.10, 3.35, 3.65]`, `chg_edges=[2.80, 3.38, 3.47, 3.60]`
- **auto-fit 경계** (`auto_range=True`, 구버전): train split 실측 V percentile(1–99)로 equal-Ah 그리드 자동 계산
  - ⚠️ auto-fit은 다단계 CC 프로토콜에서 V-Q 비단조성으로 경계가 왜곡됨. 현재 권장하지 않음.
- 분류기 기본값: `rule` (전압 경계 결정론적 판별, 학습 불필요)
- CC→CV 감지: V ≥ 3.60V & I < 0.80×I_max 동시 만족 첫 샘플

#### 실험에서 관찰된 데이터셋별 세그먼트 분포

**MIT (LFP-fixed 경계 기준, seg_diagnose stats_MIT.txt)**:

| 시나리오 | 세그먼트 수 | 평균 행 수 | 비고 |
|---------|-----------|-----------|------|
| chg_win0 | 97,366 | 28.0 | ⚠️ 극히 짧음 — HI 통계 불안정 위험 |
| chg_win1 | 97,366 | 31.7 | ⚠️ 짧음 |
| chg_win2 | 97,366 | 246.6 | 안정 |
| chg_cv  | 97,366 | 201.6 | 안정 |
| dis_win0 | 98,005 | 93.2 | 안정 |
| dis_win1 | 98,004 | 123.4 | 안정 |
| dis_win2 | 98,004 | 117.9 | 안정 |
| **합계** | **683,477** | — | — |

**HUST (auto-fit 경계 기준, vwindow_v1 stats_HUST.txt — 참고용)**:

| 시나리오 | 세그먼트 수 | 비고 |
|---------|-----------|------|
| chg_win0 | **0** | HUST 5C 충전은 낮은 전압 구간을 통과하지 않음 |
| chg_win1 | **0** | 동일 — auto-fit에서 HUST 충전이 이 윈도우 미도달 |
| chg_win2 | 146,039 | HUST 전체 충전이 하나의 윈도우에 집중 |
| dis_win0 | 179 | 거의 빈 구간 |
| dis_win1 | 146,039 | 안정 |
| dis_win2 | 146,039 | 안정 |

> **핵심 문제**: auto-fit 경계는 MIT 전압 분포 기준으로 계산되므로 HUST 충전 구간(전압 프로파일 상이)에서 chg_win0/win1이 공백이 됨. LFP 고정 경계 사용으로 이 문제를 완화.

#### 실행 방법

```powershell
# LFP 고정 경계 (권장)
python 4_hi_analysis/hi_correlation.py --seg-axis vwindow

# Step 5
python 5_model/train_scr.py --phase 1 --seg-axis vwindow
```

#### 장단점

| 장점 | 단점 |
|------|------|
| 열화해도 경계 전압 불변 (q_frac 대비 안정적) | chg_win0/win1은 평균 행 수 ~30 → HI 통계 불안정 가능 |
| MIT/HUST 동일 화학조성 → 동일 LFP 경계 적용 | HUST 충전 전압 프로파일이 MIT와 달라 일부 구간 희소 |
| IC 선행연구 직접 인용 가능 (Dubarry 2012, Hu 2020) | chg_cv R² ≈ 0.20–0.35 — CV 구간 예측 어려움 |
| chg_cv 독립 분리로 CC/CV 혼재 방지 | n_scenarios=7로 Stage B 파라미터 수 증가 |

---

### 축 3: `rcs` — 랜덤 세그먼트 샘플러

> 구현 파일: `common/scenario/rcs.py`  
> 참고: Deng et al. 2022 계열 (임의 윈도우 강건성 테스트)

사이클당 `n_samples`개 랜덤 시작점 + 고정 폭 `window` (q_frac 기준)로 부분 세그먼트를 샘플링.

#### 핵심 특성

- `assign="position_bin"` (기본): 세그먼트 중심 q_frac → lo/mid/hi 사후 binning  
  → n_scenarios=6 (qfrac과 동일, 교차 비교 가능)
- `assign="none"`: n_classes=1, 분류 라우팅 무력화 (Deng 원안 재현)  
  → n_scenarios=2 (방향만 구분)
- `seed` 고정으로 재현성 보장 (기본: 42)
- CV 구간 자동 제외: 충전 CC→CV 전환 이후는 샘플링에서 제외 (`_detect_cv_start` 적용)
- 이중 용도: 학습 축 또는 `test_scr.py --eval-axis rcs`에서 강건성 샘플러로 재사용

#### 실험에서 관찰된 세그먼트 통계 (stats_MIT.txt, n_samples=6, window=0.3)

| 시나리오 | 세그먼트 수 | 평균 행 수 | 비고 |
|---------|-----------|-----------|------|
| chg_lo | 152,760 | 59.8 | — |
| chg_mid | 276,594 | 57.7 | 중간 q_frac 구간이 가장 많음 (정규분포 효과) |
| chg_hi | 151,908 | 87.6 | — |
| dis_hi | 154,151 | 92.1 | — |
| dis_mid | 279,982 | 62.3 | — |
| dis_lo | 153,891 | 63.2 | — |
| **합계** | **1,169,286** | — | vwindow(683k)·protocol(466k)보다 훨씬 많음 |

> 랜덤 샘플링 특성상 mid 구간(0.33–0.67)이 lo/hi보다 약 1.8× 많다.  
> 단 평균 행 수 ~60은 HI 통계 신뢰성 관점에서 vwindow chg_win0/win1과 유사하게 짧음.

#### 실행 방법

```powershell
# 기본값 (n_samples=6, window=0.3, seed=42)
python 4_hi_analysis/hi_correlation.py --seg-axis rcs

# Deng 원안 (분류 없음)
python 4_hi_analysis/hi_correlation.py --seg-axis rcs \
  --axis-config '{"assign":"none","n_samples":6,"window":0.25}'

# Step 5
python 5_model/train_scr.py --phase 1 --seg-axis rcs --seed 42
```

#### 장단점

| 장점 | 단점 |
|------|------|
| 특정 경계 가정 없음 → 경계 선택 편향 제거 | 랜덤성 → 반복 실험 시 시드 관리 필수 |
| Deng baseline 완전 재현 + 직접 비교 가능 | 짧은 윈도우 → HI 통계 불안정 가능 |
| 강건성 검증 샘플러로도 활용 | 물리적 해석 불가 (레이블 = 위치 bin) |

---

### 축 4: `cluster` — K-means 클러스터 세그멘터

> 구현 파일: `common/scenario/cluster.py`  
> 참고: Ke 2026 재현

방향별 q_frac `n_fine` 균등분할 → 미세 세그먼트별 Δq 통계벡터(mean/std/skew) → K-means 클러스터링.

#### 핵심 특성

- `fit()`: train split 셀의 통계벡터 → K-means → Gap statistic + 1-SE rule로 k 자동 선정
- `k_range=(2, 8)`: 탐색 범위; Gap statistic이 최적 k 결정
- 분류기 기본값: `centroid` (최근접 센트로이드 배정, 학습 불필요)
- `mlp_probe`로 바꾸면 "학습형 라우팅 vs 고정 클러스터" 직접 비교 실험 가능

#### 실행 방법

```powershell
# 기본값 (n_fine=10, k_range=[2,8])
python 4_hi_analysis/hi_correlation.py --seg-axis cluster

# 세밀도 조정
python 4_hi_analysis/hi_correlation.py --seg-axis cluster \
  --axis-config '{"n_fine":10,"k_range":[2,6]}'

# Step 5 (centroid 분류기 → fit 단계 필요)
python 5_model/train_scr.py --phase 1 --seg-axis cluster
```

#### 장단점

| 장점 | 단점 |
|------|------|
| 데이터 기반 "자연 경계" 발견 | k 자동선정 결과가 실행마다 다를 수 있음 |
| Ke 2026 직접 재현 가능 | `fit()` 필수 (train split 확정 후 실행) |
| 경계가 프로토콜·전기화학과 일치하는지 독립 검증 가능 | 결과 해석이 주관적일 수 있음 |

---

## 5. 축 선택 및 실행 방법

### CLI 인터페이스

Step 4와 Step 5에 동일한 `--seg-axis`를 지정해야 한다.

```powershell
# Step 4: HI 추출
python 4_hi_analysis/hi_correlation.py --seg-axis <axis> [--axis-config '<JSON>']

# Step 5 Phase 1
python 5_model/train_scr.py --phase 1 --seg-axis <axis> [--seed N]

# Step 5 Phase 2
python 5_model/train_scr.py --phase 2 --seg-axis <axis> \
  --gates-from _5_data_model_scr/MMDD_HHMM

# 평가 (spec은 run_dir/scenario_spec.json에서 자동 로드)
python 5_model/test_scr.py
```

### `scr.yaml` 경로 동기화

축 변경 시 yaml의 두 경로를 함께 수정해야 한다:

```yaml
data:
  data_dir:     "_4_data_hi/<axis>/cycle"   # 예: "_4_data_hi/vwindow/cycle"
  seg_data_dir: "_4_data_hi/<axis>/seg"     # 예: "_4_data_hi/vwindow/seg"
```

### 출력 구조

```
_4_data_hi/
  qfrac/
    cycle/{MIT,HUST}/*.pkl    # wide-cycle HI (한 행=한 사이클)
    seg/{MIT,HUST}/*.pkl      # native-seg HI (한 행=한 세그먼트)
    scenario_spec.json
  protocol/
    cycle/ seg/ scenario_spec.json
  vwindow/
    cycle/ seg/ scenario_spec.json
  rcs/
    cycle/ seg/ scenario_spec.json
  cluster/
    cycle/ seg/ scenario_spec.json
```

---

## 6. 축 비교 요약

| 항목 | `qfrac` | `protocol` | `vwindow` | `rcs` | `cluster` |
|------|---------|------------|-----------|-------|-----------|
| 경계 근거 | 임의 q_frac 분할 | 논문 프로토콜 CC 전환 | IC 전압 상전이 | 랜덤 샘플링 | K-means 클러스터 |
| n_scenarios | 6 | 6 (max_steps=3) | **7** (n_windows=3+cv) | 6 또는 2 | 가변 (k_range) |
| 분류기 기본 | mlp_probe | rule | rule | mlp_probe | centroid |
| `fit()` 필요 | ✗ | ✗ | ✗ (LFP 고정 경계) | ✗ | ✓ |
| 열화 불변 경계 | ✗ | ✓ (프로토콜 고정) | ✓ (전압 고정) | ✓ | △ |
| MIT/HUST 경계 동일 | ✓ | 방향별 상이¹ | ✓ (동일 V범위) | ✓ | 데이터 의존 |
| 리뷰어 설득력 | 기준선 | 높음 (논문 인용) | 높음 (물리 해석) | 중간 (Deng 재현) | 높음 (if 일치) |
| 구현 리스크 | **완료** | **완료** | **완료** | **완료** | **완료** |

> ¹ protocol 축: MIT는 chg_step1이 있고 dis_step2는 1개, HUST는 chg_step2=0. 두 데이터셋에서 시나리오 인덱스가 가리키는 물리 체계가 달라 크로스 DS 학습 시 step 레이블 혼선 위험.

---

## 7. 권장 실험 순서

1. **`qfrac` (기준선)**: 현행 동작 그대로 성능 측정 → 모든 축 비교의 기준값 확보.

2. **`protocol` 비교**: HUST C4(균일) 자동 분리 효과 측정. MIT 2-step 프로토콜 전환 분리 효과 측정. qfrac 대비 분류 정확도 & SOH RMSE 변화 확인.

3. **`vwindow` 비교**: IC 전압 경계 기반 vs q_frac 경계 기반. cross-dataset 전이에서 어느 축이 더 안정적인지 측정.

4. **`cluster` 검증**: 자동 탐색 경계 vs protocol/vwindow 경계 일치도 확인.  
   일치 시 삼중 근거(프로토콜·전기화학·데이터) → 논문 contribution 강화.  
   불일치 시 원인 분석 후 cross-dataset 한계로 명시.

5. **`rcs` 강건성 테스트**: 경계 가정 없는 임의 윈도우 방식 대비 성능 하한 측정.  
   qfrac/protocol/vwindow보다 낮은 성능 확인 → 경계 설계 가치 입증.

---

## 8. 축별 예상 시나리오 분포
---

### `vwindow` 축 주의사항

vwindow 분류기의 시나리오 레이블은 **전압 윈도우 위치**로 결정되며 프로토콜과 무관하다.  
현재 구현에서는 CE loss가 비활성화되어 있고 분류기가 `rule` 기반이므로, 분류 로짓이 SOH 회귀에 영향을 주지 않는다.

**chg_cv 예측 어려움**:  
실험에서 chg_cv 시나리오의 R² ≈ 0.20–0.35로 다른 시나리오 대비 매우 낮음.  
원인: CV 구간은 셀 내부저항 및 노화 상태에 따라 길이·HI 분포가 크게 달라지며,  
단순 MSE로는 이 높은 가변성을 학습하기 어렵다.

**대응 방안**:  
- chg_cv 전용 loss 가중치 조정 또는 해당 시나리오 게이트 학습 강도 분리
- chg_cv R² 낮음은 모델 한계로 논문에 명시 필요

---

### `cluster` 축 검증 결과 해석 기준

| 결과 | 해석 | 후속 조치 |
|------|------|---------|
| 경계 ≈ protocol 분할점 ±0.05 | 프로토콜 구조가 데이터 신호와 일치 | 삼중 근거 확보 → 논문 기여 |
| 경계 ≈ vwindow V_b1/V_b2 대응 q_frac | 전기화학 구조가 데이터 신호 지배 | vwindow 우선 선택 권장 |
| 경계 MIT ≠ HUST | 데이터셋 구조 차이 정량화 | 원인 분석 후 cross-dataset 한계 명시 |
| k 불안정 (실행마다 다름) | 클러스터 구조 약함 | rcs 강건성 테스트로 대체 |

---

## 9. 실험 결과 — 축별 예측 성능 비교

> 학습 조건: MIT+HUST 혼합, train/val/test = 120/40/40 셀 (seed=42 고정 분할)  
> 모델: SCRModel (d_probe=64, d_head=128, MLP cap_head), Phase 1+2, CE loss 비활성

### 9-1. 전체 성능 요약

| 실행 폴더 | 축 | 경계 방식 | test RMSE | test MAE | test R² | test MAPE |
|---------|---|---------|----------|---------|--------|---------|
| `0713_2357` | **protocol** | CC 전환 자동 감지 | 0.0353 | 0.0220 | 0.743 | 2.46% |
| `0714_2010` | **vwindow** | auto-fit (equal-Ah) | 0.0422 | 0.0284 | 0.633 | 3.17% |
| `0715_0026` | **vwindow** | LFP 고정 경계 | **0.0337** | **0.0201** | **0.766** | **2.25%** |
| rcs | (진행 중) | — | — | — | — | — |
| qfrac | (미실행) | — | — | — | — | — |

**현재 결과 기준 순위**: vwindow-LFP > protocol > vwindow-auto > (rcs, qfrac 미측정)

### 9-2. 시나리오별 세부 성능

**protocol (0713_2357)**:

| 시나리오 | test RMSE | test R² | 비고 |
|---------|---------|--------|------|
| step0 | 0.0311 | 0.801 | — |
| step1 | 0.0256 | 0.865 | 최고 성능 구간 |
| step2 | 0.0461 | 0.563 | 최저 — HUST chg=0, MIT dis=1 |
| (충전 전체) | 0.0408 | 0.657 | — |
| (방전 전체) | 0.0288 | 0.829 | — |

**vwindow LFP-fixed (0715_0026)**:

| 시나리오 | test RMSE | test R² | 비고 |
|---------|---------|--------|------|
| win0 | 0.0258 | 0.863 | — |
| win1 | 0.0258 | 0.863 | — |
| win2 | 0.0316 | 0.795 | — |
| cv   | 0.0562 | 0.348 | ⚠️ CV 구간 예측 어려움 |
| (충전 전체) | 0.0379 | 0.705 | — |
| (방전 전체) | 0.0234 | 0.887 | 방전 성능 우수 |

### 9-3. 관찰된 패턴

1. **vwindow LFP-fixed > vwindow auto-fit**: 고정 전기화학 경계가 equal-Ah 자동피팅보다 RMSE 기준 20% 우수. auto-fit은 HUST 충전에서 chg_win0/win1이 공백이 되는 구조적 문제가 있었음.

2. **방전 성능 > 충전 성능 (vwindow)**: dis_win0–2의 R² ≈ 0.86–0.89, chg 전체 R² ≈ 0.70. 방전은 단일 CC 프로토콜(MIT 4C, HUST C1/C2/C3)이라 HI 분포가 일관되며 시나리오 신호가 명확함.

3. **protocol step2 ≈ 빈 시나리오**: MIT dis_step2=1건, HUST chg_step2=0건으로 해당 게이트는 사실상 학습 불가. test R²=0.563으로 전체 성능을 끌어내림. `max_steps=2`로 줄이면 이 문제 완화 가능.

4. **chg_cv 항상 저성능**: vwindow 양 실행 모두 level_cv R²≈0.20–0.35. CV 구간 HI의 높은 가변성이 원인으로 보이며, 이 시나리오를 별도 처리하거나 제외하는 방향 검토 필요.

5. **HI 효율**: 모든 실행에서 avg_probe_his ≈ 2.5–2.7 (64개 중), avg_scen_his ≈ 55.0. Stage A 게이트가 극히 소수(2~3개)의 HI로 방향 분류를 수행하는 반면 Stage B는 55개를 사용 — Stage B가 실질적인 회귀 부담을 담당.

### 9-4. HI plot 디렉토리 참조

| 축 | HI plot 경로 | seg_diagnose 경로 |
|----|------------|-----------------|
| protocol | `4_hi_analysis/hi_plot/0713_protocol/` | `4_hi_analysis/outputs/seg_diagnose/protocol/` |
| vwindow (현행) | `4_hi_analysis/hi_plot/0714_vwindow/` | `4_hi_analysis/outputs/seg_diagnose/vwindow/` |
| rcs | `4_hi_analysis/hi_plot/0714_rcs/` | `4_hi_analysis/outputs/seg_diagnose/rcs/` |

각 디렉토리 내 `hi_trend.png` (SOH vs HI scatter), `hi_segment_trend_*.png` (시나리오별 HI 추이),  
`hi_correlation.png` (HI 간 상관), `stats_*.txt` (세그먼트 수·크기 통계) 포함.
