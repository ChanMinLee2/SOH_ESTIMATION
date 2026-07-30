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

이 문서는 기본 qfrac 분할과 함께 제공되는 5가지 대체 축(protocol / vwindow / rcs / cluster / q_frac_wide)의 설계 근거와 실행 방법을 정리한다.

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

## 4. 구현된 7가지 축

모든 축은 `common/scenario/` 패키지 내 `Segmenter` ABC를 구현한다.

### 공통 인터페이스

```python
from common.scenario import get_segmenter

seg = get_segmenter("qfrac")                        # 기본값
seg = get_segmenter("protocol",     cfg={"protocol":     {"max_steps": 3}})
seg = get_segmenter("vwindow",      cfg={"vwindow":      {"n_windows": 3}})
seg = get_segmenter("rcs",          cfg={"rcs":          {"n_samples": 6, "window": 0.3}})
seg = get_segmenter("cluster",      cfg={"cluster":      {"n_fine": 10, "k_range": [2, 8]}})
seg = get_segmenter("q_frac_wide",  cfg={"q_frac_wide":  {"n1": 0.4, "n2": 0.2, "n_samples": 4}})
seg = get_segmenter("vqslope",      cfg={"vqslope":      {"mode": "dva", "n_samples": 1}})

spec: ScenarioSpec = seg.get_spec()   # ScenarioSpec 반환
```

**`ScenarioSpec`** — Step 4 출력 / Step 5 입력 계약:
```python
@dataclass
class ScenarioSpec:
    axis: str               # "qfrac" | "protocol" | "vwindow" | "rcs" | "cluster" | "q_frac_wide" | "vqslope"
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

### 축 5: `q_frac_wide` — 파라미터 구간 균등격자 세그멘터

> 구현 파일: `common/scenario/q_frac_wide.py`

rcs처럼 폭 기반 윈도우를 사용하지만, **세 고정 구간(hi/mid/lo)**을 사전 정의하고 그 내부에서 균등 격자 시작점을 배치한다. 구간 크기(n1)와 세그먼트 길이(n2)를 각각 파라미터로 지정할 수 있으며, n1을 크게 설정하면 구간 간 겹침이 발생한다.

#### 구간 정의

`n1` = 구간 크기 (q_frac 비율), `n2` = 세그먼트 길이 (q_frac 비율)

| 구간명 | q_frac 절대 범위 | 설명 |
|--------|----------------|------|
| `hi`  | `[0.0, n1]` | 0% 끝점 → 50% 방향 |
| `mid` | `[0.5 - n1/2, 0.5 + n1/2]` | 50% 중심 → 양방향 |
| `lo`  | `[1.0 - n1, 1.0]` | 100% 끝점 → 50% 방향 |

세그먼트 배치 규칙:
- 유효 시작 범위: `[zone_start, zone_end - n2]` (세그먼트가 구간을 벗어나지 않음)
- 균등 격자: `np.linspace(zone_start, zone_end - n2, n_samples)` 로 `n_samples`개 시작점 배치
- `n_samples = 1` → 유효 범위 중앙점 1개

#### 구간 커버리지 및 겹침 특성

**n1 × 3 < 100% (n1 < 1/3 ≈ 33.3%) — 겹침 없음, 구간 사이 gap 존재:**

```
n1=0.20:
  hi  : [0.00──0.20]
  mid :              [0.40──0.60]
  lo  :                           [0.80──1.00]
         ←── gap 0.20 ──→ ←── gap 0.20 ──→
  커버리지: 3 × 0.20 = 60%  |  레이블 충돌: 없음
```

**n1 × 3 = 100% (n1 = 1/3 ≈ 33.3%) — 겹침 없음, gap 없음 (최소 전체 커버):**

```
n1=0.33:
  hi  : [0.00──────0.33]
  mid :             [0.33──────0.67]
  lo  :                          [0.67──────1.00]
  커버리지: 3 × 0.33 = 100%  |  레이블 충돌: 없음
```

**n1 × 3 > 100% (n1 > 1/3 ≈ 33.3%) — 겹침 있음, 레이블 충돌 발생:**

```
n1=0.40:
  hi  : [0.00────────────0.40]
  mid :       [0.30────────────0.70]   ← hi∩mid = [0.30, 0.40] (10% 중복)
  lo  :             [0.60────────────1.00]   ← mid∩lo = [0.60, 0.70] (10% 중복)
  커버리지: 3 × 0.40 = 120%  |  레이블 충돌: 있음
```

겹침 구간에서 생성된 세그먼트는 두 구간 모두에서 추출되어 **서로 다른 scenario_id로 훈련 데이터에 중복 포함**된다. 동일 입력 HI에 상충되는 레이블이 붙어 학습 노이즈로 작용한다.

- 겹침 폭 공식 (n1 > 1/3일 때): `1.5·n1 − 0.5`
  - n1=0.35 → 겹침 2.5%,  n1=0.40 → 겹침 10.0%,  n1=0.44 → 겹침 16.0%

각 세그먼트는 자기 구간 내에서만 생성 (`s + n2 ≤ zone_end` 보장).

**구간 내 세그먼트 간격** = `(n1 − n2) / (n_samples − 1)`:

| 조건 | 구간 내 상태 |
|------|------------|
| 간격 > n2  즉  `n2 < n1 / n_samples` | 세그먼트 사이 gap 존재 (구간 일부 미커버) |
| 간격 = n2  즉  `n2 = n1 / n_samples` | 갭·겹침 없이 구간 전체를 정확히 타일링 |
| 간격 < n2  즉  `n2 > n1 / n_samples` | 세그먼트끼리 겹침, 구간 전체 커버 (중복 있음) |

예시 (n1=0.35, n_samples=4, 경계 n2 = 0.35/4 ≈ 0.088):
- n2=0.05 → 간격 0.10 > n2 → gap O, 커버리지 57% (4×0.05 / 0.35)
- n2=0.20 → 간격 0.05 < n2 → 세그먼트 내 겹침 O, 커버리지 100%

#### 파라미터 유효 범위

| 파라미터 | 코드 허용 범위 | 권장 실험 범위 | 핵심 제약 |
|---------|-------------|-------------|---------|
| **n1** | `[0.05, 0.45)` | `[0.10, 0.40]` | 현재 코드 최솟값 `0.35` — n1 민감도 실험 전 완화 필요 |
| **n2** | `(0, n1)` | `[0.05, 0.30]` | `n2 ≥ n1`이면 해당 구간 세그먼트 생성 불가(skip) |
| **n_samples** | `≥ 1` | `2 ~ 6` | `1`이면 유효 범위 중앙점 1개만 배치 |

**n1 구조 분류 (n_samples, n2 무관):**

| 조건 | 커버리지 | 레이블 충돌 | 비고 |
|------|---------|-----------|------|
| `n1 × 3 < 100%`  (n1 < 33.3%) | 부분 (gap 존재) | 없음 | 구간 민감도 분석에 적합 |
| `n1 × 3 = 100%`  (n1 = 1/3)   | 완전 (gap 없음) | 없음 | 최소 충돌 전체 커버 |
| `n1 × 3 > 100%`  (n1 > 33.3%) | 초과 (overlap)  | 있음 | 고밀도 커버, 충돌 감안 |

> **코드 제약 주의**: 현재 `q_frac_wide.py`의 n1 하한이 `0.35`로 설정되어 있어 n1 < 33.3% 실험이 불가하다. n1 민감도 실험(Phase B) 전에 `0.05 ≤ n1 < 0.45`로 완화해야 한다.

#### 시나리오 매핑 (qfrac/rcs와 동일 컨벤션)

| 세그먼트 | 방향 | 구간 | scenario_id | latent_class |
|---------|------|------|-------------|-------------|
| `chg_lo`  | 충전 | lo zone | 0 | lo(0) |
| `chg_mid` | 충전 | mid zone | 1 | mid(1) |
| `chg_hi`  | 충전 | hi zone | 2 | hi(2) |
| `dis_hi`  | 방전 | hi zone | 3 | hi(2) |
| `dis_mid` | 방전 | mid zone | 4 | mid(1) |
| `dis_lo`  | 방전 | lo zone | 5 | lo(0) |

라우팅 테이블: `routing = [[0, 1, 2], [5, 4, 3]]` — qfrac/rcs 와 동일

#### rcs와의 차이점

| 항목 | `rcs` | `q_frac_wide` |
|-----|-------|--------------|
| 시작점 선정 | 완전 랜덤 (`rng.uniform`) | 균등 격자 (`linspace`) |
| 구간 경계 | 없음 (전체 [0, 1-window]) | 3구간 고정 (hi/mid/lo) |
| 시나리오 판별 | 사후 중심 binning | 사전 구간 배정 |
| 구간 내 침범 | 없음 (단순 윈도우) | 수학적으로 보장 없음 |
| 세그먼트 수 | 사이클당 n_samples개 | 사이클당 3 × n_samples개 |

#### 실행 방법

```powershell
# ── Step 4: HI 추출 ────────────────────────────────────────────────────────
# 기본값 (n1=0.4, n2=0.2, n_samples=4) — 사이클당 12개 세그먼트
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide

# 파라미터 지정 (--axis-config 는 축 이름 없이 flat dict 로 전달)
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --axis-config '{"n1": 0.4, "n2": 0.2, "n_samples": 4}'

# 구간 겹침 활성화 (n1=0.45 미만 제약 있음)
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --axis-config '{"n1": 0.44, "n2": 0.2, "n_samples": 6}'

# ── Step 5: 학습 (--axis-config 불필요) ──────────────────────────────────
# n1/n2/n_samples는 모델 구조(n_scenarios=6, routing)에 영향을 주지 않으므로
# --seg-axis 만 지정하면 PKL에서 데이터를 자동 로드한다.
python 5_model/train_scr.py --phase 1 --seg-axis q_frac_wide
```

> **`--axis-config` 형식 주의 (`hi_correlation.py`)**:  
> - 스크립트 내부에서 `{axis: cfg}` 로 자동 래핑하므로, 축 이름 없이 flat dict 를 넘겨야 한다.  
>   `'{"n1": 0.4}'` ✓ / `'{"q_frac_wide": {"n1": 0.4}}'` ✗  
> - **PowerShell에서 `{"..."}` 는 스크립트블록으로 해석**되어 JSON이 잘린다.  
>   변수로 먼저 받은 뒤 전달할 것:  
>   ```powershell
>   $cfg = '{"n1": 0.4, "n2": 0.2, "n_samples": 4}'
>   python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --axis-config $cfg
>   ```

#### ScenarioSpec 예시

```json
{
  "axis": "q_frac_wide",
  "n_scenarios": 6,
  "scenario_names": ["chg_lo","chg_mid","chg_hi","dis_hi","dis_mid","dis_lo"],
  "n_classes": 3,
  "class_names": ["lo","mid","hi"],
  "routing": [[0,1,2],[5,4,3]],
  "classifier_default": "mlp_probe",
  "params": {"n1": 0.4, "n2": 0.2, "n_samples": 4}
}
```

#### 파라미터 설정 가이드

| 목적 | 권장 설정 | 비고 |
|------|---------|------|
| qfrac과 동일 경계 (비교 기준) | n1=1/3≈0.333, n2=0.133, n_samples=1 | 구간당 1개, 경계 일치 |
| rcs 대체 (균등 커버) | n1=0.4, n2=0.2, n_samples=4 | 구간당 4개, 겹침 없음 |
| 고밀도 데이터 수집 | n1=0.6, n2=0.15, n_samples=8 | 구간 겹침 + 세밀 격자 |
| 세그먼트 최소화 (빠른 실험) | n1=0.4, n2=0.3, n_samples=2 | 구간당 2개 |

> **n2 ≥ n1이면 해당 구간에서 세그먼트 생성 불가.** `_start_positions()`가 빈 배열 반환 → 해당 구간 skip.

#### 장단점

| 장점 | 단점 |
|------|------|
| 세그먼트가 구간을 벗어나지 않아 레이블 일관성 보장 | n1, n2, n_samples 조합 탐색 필요 |
| 균등 격자로 각 구간 내 q_frac 분포를 고르게 커버 | n2 ≥ n1이면 해당 구간 세그먼트 생성 불가 |
| qfrac/rcs와 동일 n_scenarios=6, 교차 비교 가능 | 구간 겹침 시 동일 q_frac에 다중 시나리오 세그먼트 공존 |
| n1 크기로 구간 겹침을 연속적으로 조절 가능 | 물리적 해석은 rcs와 마찬가지로 제한적 |

---

### 축 6: `vqslope` — 기울기(dV/dQ · dQ/dV) 형상 기반 세그멘터

`common/scenario/vqslope.py` — `q_frac_wide`(누적 전하량 기반)의 **라벨 얽힘 한계**를 극복하기 위해 설계된 축. 존 경계를 "얼마나 진행됐는가(누적 Q)"가 아니라 **"지금 곡선 형상이 어떤 상태인가(순간 기울기)"**로 정의한다.

#### 설계 동기 — 왜 누적량이 아니라 순간량인가

`q_frac_wide`는 존 경계를 `q_frac = q / q_tot`로 정의한다. 그런데 이 세그먼트를 만들려면 이미 `q_tot`(해당 사이클의 실측 총 전하량 ≈ SOH와 1:1 대응)을 알아야 한다. 즉 **존 라벨 자체가 라벨(SOH)의 함수**로 만들어진다. 이 경우 Stage A 분류기가 형상으로 존을 잘 맞혀도, "형상에서 위치를 복원했다"인지 "형상에서 SOH 단서를 감지해 존을 우회 추정했다"인지 구분할 수 없다.

세그멘테이션 축의 요건은 두 가지다:
- **(i) 라벨의 함수가 아닐 것** — `q_frac`은 여기서 탈락
- **(ii) 스니펫 내용만으로 존을 복원 가능할 것** — 그 복원 가능성을 측정하는 것이 정확히 Stage A 분류 정확도

`vqslope`는 경계를 **플래토 진입/이탈이라는 상전이 이벤트의 위치**로 정의한다. `vwindow`(고정 전압 경계)가 노화에 강건한 것과 같은 이유로, LFP 플래토 진입/이탈 전압은 노화에서 거의 변하지 않는다. `vqslope`는 그 랜드마크를 절대 전압값이 아니라 **"기울기가 평탄한가/급한가"라는 형상 판정**으로 잡으므로, 미세한 전압 시프트에도 더 안정적일 것으로 기대된다.

#### 존 정의 (곡선 진행 순서 = q_local 증가 방향)

LFP 방전/충전 곡선은 **급경사 → 평탄(플래토) → 급경사** 순서다. 이 3상태로 3분할한다:

| 존 | 정의 | 물리 | latent_class |
|----|------|------|:---:|
| **head** | 곡선 시작 ~ 플래토 진입 | 급경사 초입 (상단 숄더/니 진입) | 2 (hi) |
| **plateau** | 플래토 진입 ~ 이탈 (`\|dV/dQ\| < θ_flat`) | LFP 2상 공존 평탄 구간 | 1 (mid) |
| **tail** | 플래토 이탈 ~ 곡선 끝 | 급경사 말단 (하단 급강하/CV) | 0 (lo) |

이 `(head=hi, plateau=mid, tail=lo)` 매핑은 `q_frac_wide`의 `(hi=[0,n1], mid=중앙, lo=[1-n1,1])` 컨벤션과 정확히 정렬되어, **모델 구조(n_scenarios=6, routing=[[0,1,2],[5,4,3]])를 그대로 재사용**한다. 하류(회귀기·분류기·평가) 코드 무수정.

#### 모드 — DVA(dV/dQ) vs ICA(dQ/dV)

두 모드 모두 "플래토의 q_local 범위 `[q_entry, q_exit]`"를 구하고, 이후 3분할은 공통이다. 플래토 범위를 찾는 방법만 다르다:

| 모드 | 플래토 검출 방법 | 재사용 함수 |
|------|-----------------|-----------|
| **dva** (기본) | `\|dV/dQ\| < θ_flat`인 Q-빈들의 q 범위 | `_build_vq_curve` |
| **ica** | dQ/dV 주 피크의 FWHM(v_left..v_right) 안에 드는 원시 포인트의 q 범위 | `_build_ica_seg`, `_peak_fwhm_asym` |

> `θ_flat`는 `q_frac_wide`/`lfp_plateau_frac`이 쓰는 값과 동일한 `THETA_FLAT = 0.25 V/Ah`를 기본으로 하며, `common/scenario/_curves.py`에 단일 소스로 정의된다.

#### 실측 검증 결과 (MIT 5셀 + HUST 5셀, n_samples=1)

**DVA 모드 — 플래토 검출 실패 0건, 전 시나리오 생존율 99.5~100%.** `q_frac_wide`가 HUST `chg_mid`/`chg_hi`를 (짧은 n2에서) 완전히 누락(0%)시킨 문제가 `vqslope-dva`에서는 발생하지 않는다.

| 모드 | MIT | HUST | 비고 |
|------|-----|------|------|
| **dva** | 전 시나리오 100% | 99.5~100% | 안정적 — **권장 기본 모드** |
| **ica** | 99.8~100% | `dis_hi` **0%** (pts_median=2) | 방전 dis_hi에서 피크 FWHM이 과도하게 좁게 잡혀 plateau 존 붕괴 |

→ **`mode: dva`를 기본으로 권장.** `ica`는 옵션으로 유지하되 위 약점(방향/구간별 피크 검출 불안정)을 감안해야 한다.

#### 실패 모드 — 반드시 진단할 것

플래토 미검출(짧은 창, 노이즈) 시 그 방향의 3존이 **통째로 스킵**된다. 이는 `q_frac_wide`의 `min_pts` 절벽과 형태만 다른 같은 성격의 표본 손실이다. `VQSlopeSegmenter`는 이를 위해 `n_plateau_fail`(방향별 검출 실패 수)과 `candidate_n_points`(q_frac_wide와 동일 인터페이스) 카운터를 제공한다.

#### 파라미터

| 파라미터 | 기본값 | 의미 |
|---------|--------|------|
| `mode` | `"dva"` | `"dva"`(dV/dQ) \| `"ica"`(dQ/dV) |
| `n_samples` | `1` | 존당 세그먼트 수 (존 내부 q_local 균등 등분; 1이면 존 전체 1개) |
| `theta_flat` | `0.25` | dva 플래토 임계값 `\|dV/dQ\| < θ` |
| `min_pts` | `10` | 세그먼트 최소 원시 포인트 수 |

#### 실행 방법

```powershell
# Step 4: HI 추출 (mode·n_samples 별 고유 경로: _4_data_hi/vqslope/{mode}_N-{ns}/)
python 4_hi_analysis/hi_correlation.py --seg-axis vqslope --axis-config '{"vqslope": {"mode": "dva", "n_samples": 1}}'

# Step 5: 학습 (scr.yaml → scenario.axis: vqslope, axis_config: {mode: dva, n_samples: 1})
python 5_model/train_scr.py --phase 1
python 5_model/train_scr.py --phase 2 --gates-from _5_data_model_scr/MMDD_HHMM_p1_vqs_dva
```

> 데이터 경로 태그: `_4_data_hi/vqslope/{mode}_N-{n_samples}/{seg,cycle}` (예: `dva_N-1`). `q_frac_wide`와 동일하게 파라미터가 데이터 내용을 바꾸므로 파라미터별로 분리 저장된다.

#### ScenarioSpec 예시

```json
{
  "axis": "vqslope",
  "n_scenarios": 6,
  "scenario_names": ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"],
  "n_classes": 3,
  "class_names": ["lo", "mid", "hi"],
  "routing": [[0, 1, 2], [5, 4, 3]],
  "classifier_default": "mlp_probe",
  "params": {"mode": "dva", "n_samples": 1, "theta_flat": 0.25}
}
```

#### 장단점

| 장점 | 단점 |
|------|------|
| **존 경계가 라벨(SOH)의 함수가 아님** — q_frac_wide의 라벨 얽힘 한계 극복 | 플래토 미검출 시 그 방향 3존 통째 스킵 (진단 필수) |
| DVA 모드는 HUST 포함 전 시나리오 99.5%+ 생존 (q_frac_wide 절벽 없음) | ica 모드는 방향/구간별 피크 검출 불안정 (dis_hi 붕괴) |
| n_scenarios=6, routing 동일 → 모델 하류 코드 무수정 재사용 | 존 길이가 데이터로 결정되어 사이클마다 가변 (고정 폭 아님) |
| 순간 기울기 기반 → 배포 시 스니펫만으로 존 복원 가능성 측정(Stage A) | dV/dQ 계산이 짧은 창에서 노이즈 민감 (min_pts로 방어) |

---

### 공통 옵션: `random_segment` — 구간 내 고정길이 랜덤 창 (q_frac_wide · vqslope)

`q_frac_wide`와 `vqslope` 두 축에 공통으로 제공되는 **옵션**이다. 기본값은 `False`(기존 격자/등분 방식 그대로)이며, 켜면 각 시나리오 구간(존)을 벗어나지 않는 범위에서 **랜덤 위치의 고정길이 세그먼트**를 추출한다. `common/scenario/_random_seg.py`의 `sample_random_windows`를 두 축이 공유한다.

#### 설계 동기 — 세분화(n_samples↑) 시 짧은 존 붕괴 문제

`n_samples`를 키워 존을 등분하면 짧은 존(특히 vqslope의 head, q_frac_wide의 좁은 구간)이 `min_pts`를 못 넘겨 붕괴한다. 실측(vqslope dva, MIT+HUST 전량)에서 n_samples=4부터 HUST `chg_hi`가 0%로 소멸했다. 랜덤 모드는 등분 대신 **고정 관측 포인트 길이 창**을 랜덤 위치로 뽑고, 존이 창보다 짧으면 존 전체를 1개로 쓰는 fallback을 취해 이 붕괴를 완화한다.

#### 라벨(SOH) 독립성을 지키는 3대 원칙

랜덤 모드가 vqslope의 라벨-독립 강점(및 q_frac_wide의 구조)을 훼손하지 않으려면 세 원칙을 지켜야 하며, 구현이 이를 강제한다:

1. **창 길이 = 고정 관측 포인트 수(`seg_len_pts`)** — 그 사이클의 실측 총 용량 `q_tot`(≈SOH)을 참조하지 않는다. 스니펫 내용만으로 계산 가능하고 배포 시에도 동일 획득 가능하므로 라벨의 함수가 아니다.
2. **클리핑 절대 금지** — 존이 창보다 짧아도 창 길이를 존에 맞춰 줄이지 않는다(줄이면 세그먼트 길이·위치가 SOH의 결정론적 함수가 되어 가짜 HI–SOH 상관 유입). 대신 존 전체 1개 fallback.
3. **경계는 각 축 원래 방식 유지** — 존/구간 경계는 vqslope=형상 랜드마크, q_frac_wide=q_frac로 결정(랜덤 모드가 바꾸지 않음). 창 위치만 존 안에서 랜덤.

> **리뷰어 방어 요지**: 유일한 약점은 "고정 포인트 수가 데이터셋 샘플링 밀도에 의존"하는 것인데, 이는 라벨 누수가 아니라 **데이터셋 편향**이고 창 위치가 아니라 창 폭에만 영향을 준다. 아래 누락 비율 공개로 투명하게 방어한다.

#### 누락 비율(coverage) 기록

랜덤 추출은 존의 일부 포인트를 어느 창에도 포함시키지 않을 수 있다. 이 **샘플링되지 못한 포인트 비율**을 시나리오별로 집계해 `_4_data_hi/{axis_dir}/coverage_stats.txt`에 저장한다(재현성·투명성). 형식:

```
[HUST]
  시나리오        covered       total     커버율    누락율
  chg_hi           41,536      54,892    75.7%    24.3%
  chg_mid          61,263     219,472    27.9%    72.1%
  ...
```

#### 파라미터

| 파라미터 | 기본값 | 의미 |
|---------|--------|------|
| `random_segment` | `False` | True면 랜덤 창 모드 (False면 기존 격자/등분) |
| `seg_len_pts` | `20` | 랜덤 창의 고정 관측 포인트 수 (`>= min_pts` 필수, q_tot 무관) |
| `random_seed` | `42` | 재현성 시드 (`(seed, crc32(cell_id), cycle, dir)` 결정론적 — 멀티프로세싱 안전) |
| `n_samples` | (축 기본) | 존당 뽑을 랜덤 창 개수 |

#### 재현성

시드는 `(random_seed, crc32(cell_id), cycle, direction)`로 결정된다. `crc32`를 쓰므로 Python `hash()`의 `PYTHONHASHSEED` 의존성이 없어 **멀티프로세싱 워커 간에도 동일 결과**를 보장한다.

#### 저장 경로 구분

`random_segment=True`면 데이터 경로 태그에 `_random-L{seg_len_pts}` suffix가 붙어 non-random 데이터와 분리 저장된다. 이 규칙은 `hi_correlation.py`·`train_scr.py`·`train_classifier.py` 세 곳에서 일관 적용된다.

| | 데이터 경로 |
|---|---|
| vqslope non-random | `_4_data_hi/vqslope/dva_N-1/` |
| vqslope random | `_4_data_hi/vqslope/dva_N-2_random-L20/` |
| q_frac_wide random | `_4_data_hi/q_frac_wide/n1-45%_n2-20%_N-2_random-L30/` |

#### 실행 방법 (PowerShell 단축 인자)

```powershell
# vqslope + random (따옴표 없이) — 데이터 → _4_data_hi/vqslope/dva_N-2_random-L20/
python run_pipeline.py 4 --seg-axis vqslope --mode dva --n-samples 2 --random-segment --seg-len-pts 20

# Step 4만 단독
python 4_hi_analysis/hi_correlation.py --force --seg-axis vqslope --mode dva --n-samples 2 --random-segment --seg-len-pts 20 --workers 8
```

> ⚠️ **Step 4와 Step 5(학습)의 axis_config는 반드시 일치해야 한다.** Step 4를 옵션 없이 돌리면 non-random 기본 경로에 데이터가 생기는데, scr.yaml이 `random_segment: true`면 학습이 `_random-L{n}` 경로를 찾다가 "No data loaded"로 실패한다. `run_pipeline.py`로 단축 인자와 함께 한 번에 돌리면 Step 4~7에 동일 config가 전달되어 이 불일치가 원천 차단된다. (Step 4 시작 시 출력되는 "HI 추출 실행 조건"의 `데이터 저장 경로`가 scr.yaml 기대값과 같은지 눈으로 확인 가능)

#### 장단점

| 장점 | 단점 |
|------|------|
| 짧은 존 붕괴 완화 (fallback으로 최소 1개 보장) | 랜덤이라 존 일부가 학습에 미사용 (누락 비율 발생) |
| 창 길이 고정 → 라벨(SOH) 독립성 유지 | 고정 포인트 수가 데이터셋 샘플링 밀도에 의존 (편향, 누수 아님) |
| 데이터 증강 효과 (한 존에서 여러 랜덤 창) | seg_len_pts·n_samples 조합 탐색 필요 |
| 두 축 공통 + 기존 구조 무손상(기본 False) | 누락 비율을 리뷰어에 반드시 보고해야 함 |

---

## 5. 축 선택 및 실행 방법

### CLI 인터페이스

Step 4와 Step 5에 동일한 `--seg-axis`를 지정해야 한다.

```powershell
# Step 4: HI 추출 (--axis-config 로 세그먼터 파라미터 지정)
python 4_hi_analysis/hi_correlation.py --seg-axis <axis> [--axis-config '<JSON>']

# Step 5 Phase 1 (학습은 PKL 파일을 읽으므로 --axis-config 는 대부분 불필요)
python 5_model/train_scr.py --phase 1 --seg-axis <axis> [--seed N]

# Step 5 Phase 2
python 5_model/train_scr.py --phase 2 --seg-axis <axis> \
  --gates-from _5_data_model_scr/MMDD_HHMM

# 평가 (spec은 run_dir/scenario_spec.json에서 자동 로드)
python 5_model/test_scr.py
```

> **`--axis-config`가 학습 시에도 필요한 경우**: 축 파라미터가 모델 구조(n_scenarios·routing)를 바꿀 때만.  
> - `protocol --axis-config '{"max_steps":2}'` → n_scenarios 변경  
> - `vwindow --axis-config '{"n_windows":4}'` → n_scenarios 변경  
> - `rcs --axis-config '{"assign":"none"}'` → n_scenarios 6→2  
> - `qfrac`, `q_frac_wide`, `cluster` → 구조 파라미터 고정, 학습 시 불필요
>
> **PowerShell 주의**: `--axis-config '{"key": val}'` 에서 `{...}` 가 스크립트블록으로 해석될 수 있음.  
> 변수로 분리해서 전달하면 안전하다:  
> ```powershell
> $cfg = '{"max_steps": 2}'
> python 5_model/train_scr.py --phase 1 --seg-axis protocol --axis-config $cfg
> ```

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
  q_frac_wide/
    cycle/ seg/ scenario_spec.json
```

---

## 6. 축 비교 요약

| 항목 | `qfrac` | `protocol` | `vwindow` | `rcs` | `cluster` | `q_frac_wide` | `vqslope` |
|------|---------|------------|-----------|-------|-----------|--------------|----------|
| 경계 근거 | 임의 q_frac 분할 | 논문 프로토콜 CC 전환 | IC 전압 상전이 | 랜덤 샘플링 | K-means 클러스터 | 파라미터 구간 균등격자 | dV/dQ 플래토 진입/이탈 |
| 경계 기준량 | 누적(Q) | 순간(I 전환) | 순간(V) | 누적(Q) | 형상 통계 | 누적(Q) | **순간(기울기)** |
| n_scenarios | 6 | 6 (max_steps=3) | **7** (n_windows=3+cv) | 6 또는 2 | 가변 (k_range) | 6 | 6 |
| 분류기 기본 | mlp_probe | rule | rule | mlp_probe | centroid | mlp_probe | mlp_probe |
| `fit()` 필요 | ✗ | ✗ | ✗ (LFP 고정 경계) | ✗ | ✓ | ✗ | ✗ |
| 열화 불변 경계 | ✗ | ✓ (프로토콜 고정) | ✓ (전압 고정) | ✓ | △ | ✓ | ✓ (플래토 전이) |
| 라벨(SOH) 독립 경계² | ✗ | ✓ | ✓ | ✗ | △ | ✗ | ✓ |
| MIT/HUST 경계 동일 | ✓ | 방향별 상이¹ | ✓ (동일 V범위) | ✓ | 데이터 의존 | ✓ (n1,n2 동일) | ✓ (형상 기준) |
| 세그먼트 수/사이클 | 3–6 | max_steps×2 | (2n+1)×1 | n_samples | n_fine | 3×n_samples | 3×n_samples |
| 리뷰어 설득력 | 기준선 | 높음 (논문 인용) | 높음 (물리 해석) | 중간 (Deng 재현) | 높음 (if 일치) | 중간 (rcs 개선) | 높음 (라벨 독립) |
| 구현 리스크 | **완료** | **완료** | **완료** | **완료** | **완료** | **완료** | **완료** |

> ¹ protocol 축: MIT는 chg_step1이 있고 dis_step2는 1개, HUST는 chg_step2=0. 두 데이터셋에서 시나리오 인덱스가 가리키는 물리 체계가 달라 크로스 DS 학습 시 step 레이블 혼선 위험.
>
> ² **라벨(SOH) 독립 경계**: 존 경계가 SOH의 함수인지 여부. 누적량(Q) 기반 축은 `q_tot`(≈SOH)을 알아야 세그먼트를 만들 수 있어 라벨 얽힘이 있고(Stage A가 위치 복원인지 SOH 우회추정인지 구분 불가), 순간량 기반 축(vwindow/vqslope/protocol)은 이 문제에서 자유롭다. §4 축 6 참조.
>
> ³ **`random_segment` 옵션**: `q_frac_wide`·`vqslope` 두 축은 구간 내 고정길이 랜덤 창 추출 옵션을 지원한다(기본 False). 세분화 시 짧은 존 붕괴 완화 + 데이터 증강용이며, 라벨 독립성을 지키는 3원칙(고정 포인트 창·클리핑 금지·경계 불변)과 누락 비율 공개를 갖춘다. §4 "공통 옵션: random_segment" 참조.

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

---

## 10. q_frac_wide 파라미터 민감도 실험 계획

### 실험 목적

"LFP 배터리에서 SOH 정보가 어느 SOC 구간(n1)에, 어느 해상도(n2)로 측정할 때 가장 잘 나타나는가"를 체계적으로 정량화한다. 결과는 단순 성능 숫자에 그치지 않고 **어느 구간이 왜 중요한가**에 대한 물리적 근거를 제공한다.

n_samples=4, split_seed=42 고정 — 모델 구조(n_scenarios=6, routing)와 데이터 분할이 동일하므로 실험 간 공정한 비교가 가능하다.

---

### Phase A — n2 민감도 (1순위, n1=0.35 고정)

**n1=0.35를 선택한 이유**: n1×3 = 105% (전체 용량 커버 보장) + 겹침 폭 2.5% (레이블 충돌 최소).  
n1×3 ≥ 100% 조건을 만족하는 최솟값으로, 최소한의 충돌로 전체 구간을 커버한다.

| 실험 | n1 | n2 | n2/n1 | 특성 |
|------|----|----|-------|------|
| A1 | 0.35 | 0.05 | 0.14 | 매우 짧은 윈도우 — 국소 정보 최대화 |
| A2 | 0.35 | 0.10 | 0.29 | — |
| A3 | 0.35 | 0.15 | 0.43 | — |
| A4 | 0.35 | 0.20 | 0.57 | — |
| A5 | 0.35 | 0.25 | 0.71 | — |
| A6 | 0.35 | 0.30 | 0.86 | 구간 거의 전체 커버 |

**예상 결과 패턴별 해석:**

| 패턴 | 물리적 의미 |
|------|-----------|
| n2 작을수록 성능 증가 | 국소 열화 신호가 지배적 — 짧은 측정으로도 SOH 추정 가능 |
| n2 중간값에서 최적 (U자형) | 국소 신호 + 통계 안정성 균형점 존재 |
| n2 클수록 성능 증가 | 긴 윈도우의 통계적 안정성이 국소 정보보다 우위 |

---

### Phase B — n1 민감도 (2순위, best_n2 고정)

**선행 조건**: `common/scenario/q_frac_wide.py` n1 하한을 `0.35 → 0.05`로 완화해야 한다.

| 실험 | n1 | n1×3 | 커버리지 구조 | 특성 |
|------|----|----|------------|------|
| B1 | 0.10 | 30% | gap 40% (중간 구간 미커버) | 극단부만 측정 |
| B2 | 0.20 | 60% | gap 20% | — |
| B3 | 0.30 | 90% | gap 5% (거의 전체) | — |
| B4 | 0.35 | 105% | overlap 2.5% (Phase A 기준) | 기준점 |
| B5 | 0.40 | 120% | overlap 10% (현재 기본값) | — |

**gap이 있는 구간(B1~B3)의 해석**: hi/mid/lo 사이의 gap 영역(예: B1에서 [0.10, 0.40])은 모델이 활용하지 못한다. n1이 작아도 성능이 유지되면 gap 구간이 SOH에 기여하지 않는다는 증거가 된다.

**예상 결과 패턴별 해석:**

| 패턴 | 물리적 의미 |
|------|-----------|
| n1 클수록 성능 증가 (B1→B5 단조) | 넓은 커버리지 자체가 유리 — gap 구간도 정보 기여 |
| 특정 n1에서 최적 | 해당 폭이 열화 신호와 가장 잘 매칭 |
| n1 무관 (B1≈B5) | gap 구간이 SOH 예측에 무의미 — 극단부(hi/lo zone)만으로 충분 |

---

### 베이스라인

| 실험 | 축 | 목적 |
|------|---|------|
| `baseline-qfrac` | qfrac | 전체 비교 기준 (동일 n_scenarios=6) |
| `baseline-qfw-A4` | q_frac_wide (n1=0.35, n2=0.20) | Phase A 중간값 — 기존 기본값과 유사 |

---

### 실행 명령어

```powershell
# ── Phase A (n2 민감도, 코드 수정 불필요) ────────────────────────────────────
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --n1 0.35 --n2 0.05 --n-samples 4
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --n1 0.35 --n2 0.10 --n-samples 4
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --n1 0.35 --n2 0.15 --n-samples 4
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --n1 0.35 --n2 0.20 --n-samples 4
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --n1 0.35 --n2 0.25 --n-samples 4
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --n1 0.35 --n2 0.30 --n-samples 4

# 각 A* 실험에 대해 학습 (scr.yaml: axis: q_frac_wide, axis_config: {n1: 0.35, n2: X})
python run_pipeline.py 6 --seg-axis q_frac_wide --n2 0.05   # Phase A1
python run_pipeline.py 6 --seg-axis q_frac_wide --n2 0.10   # Phase A2
# ... 동일 패턴

# ── Phase B (n1 민감도, q_frac_wide.py n1 하한 0.05로 완화 후) ──────────────
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --n1 0.10 --n2 <best_n2> --n-samples 4
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --n1 0.20 --n2 <best_n2> --n-samples 4
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --n1 0.30 --n2 <best_n2> --n-samples 4
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --n1 0.35 --n2 <best_n2> --n-samples 4
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --n1 0.40 --n2 <best_n2> --n-samples 4
```

> `run_pipeline.py`에 `--n1 / --n2 / --n-samples` 인자가 없으므로, `scr.yaml`의 `axis_config`를 실험별로 갱신하거나 `--axis-config $cfg` 형태를 사용한다. (§5 CLI 주의사항 참조)

---

### 결과 기록 양식

| 실험 ID | n1 | n2 | n2/n1 | n1×3 | test RMSE | test R² | 비고 |
|--------|----|----|-------|------|-----------|---------|------|
| A1 | 0.35 | 0.05 | 0.14 | 105% | — | — | — |
| A2 | 0.35 | 0.10 | 0.29 | 105% | — | — | — |
| A3 | 0.35 | 0.15 | 0.43 | 105% | — | — | — |
| A4 | 0.35 | 0.20 | 0.57 | 105% | — | — | — |
| A5 | 0.35 | 0.25 | 0.71 | 105% | — | — | — |
| A6 | 0.35 | 0.30 | 0.86 | 105% | — | — | — |
| B1 | 0.10 | best | — | 30% | — | — | — |
| B2 | 0.20 | best | — | 60% | — | — | — |
| B3 | 0.30 | best | — | 90% | — | — | — |
| B4 | 0.35 | best | — | 105% | — | — | = A best |
| B5 | 0.40 | best | — | 120% | — | — | overlap 10% |
| baseline-qfrac | — | — | — | — | — | — | 비교 기준 |

---

## 11. 정보 누수(Data Leakage) 체크리스트 — 시나리오/세그먼트 분할 공통 원칙

새 축을 설계하거나 기존 축의 파라미터를 정할 때마다 재확인해야 하는 원칙을 모았다. 이
문서 곳곳(§0, §4 축6 `vqslope`, §4 "공통 옵션 random_segment")에 흩어져 있던 라벨-독립성
논의를 하나로 통합하고, `q_abs` 축 도입 과정에서 새로 드러난 두 가지 실패 모드를 추가한다.

### 11.0 세 가지 서로 다른 실패 모드

같은 "결과가 실제보다 좋아 보인다"는 증상이라도 원인이 다르면 대응이 다르다.

| 유형 | 정의 | 실배포 시 재현 가능한가 |
|---|---|---|
| **(A) 라벨 누수** | 그 샘플 자신의 라벨(또는 라벨과 거의 1:1 대응하는 값)이 그 샘플의 경계/피처 정의에 직접 쓰임 | ✗ (원리적으로 계산 불가) |
| **(B) 설계-단계 train/test 경계 위반** | 축 파라미터(zone 경계, seg_len 등)를 정할 때 test로 쓸 셀의 데이터를 미리 참조 | 배포 자체는 가능하나, 보고된 test 성능이 부풀려짐 |
| **(C) 선택 편향(생존자 편향)** | 라벨이 새지는 않지만, "이 시나리오에 데이터가 존재/얼마나 존재하는가" 자체가 라벨과 상관되어 학습 분포가 왜곡됨 | 배포는 가능하나, 희귀 케이스(노화 극단)에서 일반화 실패 |

### 11.1 시나리오(존) 분할에서 조심할 점

1. **경계 정의에 그 샘플 자신의 라벨-상관 값을 쓰지 않는다 — 원인(A).**
   `q_frac_wide`의 `q_frac = q/q_tot`에서 `q_tot`(그 사이클 실측 총 용량)은 SOH와 사실상
   동일한 값이다. 실배포 시 그 세션이 끝나야 `q_tot`을 알 수 있으므로 원리적으로 계산
   불가능하다(§0, §4 축6 참조).
   → 대안: `q_abs`처럼 그 셀의 **BOL(첫 사이클, 배포 전 1회 계측 가능)** 값을 고정
   기준으로 쓰거나, `vwindow`/`vqslope`처럼 순간 전압·형상으로 경계를 잡는다.

2. **경계·구간폭 파라미터를 튜닝할 때 test 셀의 데이터를 들여다보지 않는다 — 원인(B).**
   예: "test에 포함될 셀 중 가장 노화된 사이클이 zone을 얼마나 채우는지 보고 `mid_end`를
   정한다"는 발상 자체는 합리적이지만(보수적 여유값을 두겠다는 것), **그 관찰에 test 셀이
   포함되면 안 된다.** train/calibration 전용 부분집합에서만 파라미터를 정하고 고정한 뒤
   train+val+test 전체에 동일 적용해야 한다. 위험도는 "몇 개 셀의 통계를 종합했는지"에
   반비례한다 — 수백 개 셀의 평균적 fade 분포로 general한 경계를 정하는 것과 특정 test
   셀 하나의 값을 직접 반영하는 것은 위험도가 다르지만, 원칙은 항상 "가능하면 train만"이다.

3. **"이 시나리오가 존재하는가/얼마나 채워졌는가" 자체가 라벨과 상관되는 경우 — 원인(C).**
   `q_abs`에서 노화가 심할수록 특정 존(예: 정격용량 상단부 존)의 데이터가 통째로
   사라지는 경우(`4_hi_analysis/qabs_healthy_vs_aged.py` 실측: MIT 일부 셀 plateau
   capture 0%), 그 존 전용 헤드는 "내가 호출됐다는 사실 자체가 이미 상대적으로 건강한
   셀일 가능성이 높다"는 신호를 은연중 학습에 활용할 수 있다(생존자 편향). 진짜 라벨
   누수는 아니지만 그 헤드가 노화 극단에서 신뢰할 수 없게 된다.
   → 대응: 취약 존의 세그먼트 샘플링 시작 위치를 zone 저변(항상 채워지는 쪽)에
   고정(anchor)해서 존 자체가 사라지는 빈도를 최소화하거나, 최소한 그 존의 결측률을
   진단 지표로 공개한다(`random_segment`의 누락 비율 공개와 같은 정신, §4 참조).

4. **~~시나리오 이름·순서 관례가 축마다 반대일 수 있음에 주의~~ — 2026-07-30 버그로
   판명, 수정 완료.** 원래 이 항목은 "축마다 명명 관례가 다르니 매번 확인하라"는
   회피형 경고였지만, 실제로는 축들이 `q_frac_wide`의 `_ROUTING=[[0,1,2],[5,4,3]]`을
   무비판적으로 재사용하면서 생긴 **실제 라벨링 버그**였다: `chg_lo`/`chg_hi`/
   `dis_hi`/`dis_lo`라는 이름은 항상 실측 SOC(전압) 수준을 가리켜야 하는데(`qfrac.py`
   원조는 이를 정확히 구현: 충전=lo→mid→hi, 방전=hi→mid→lo, 둘 다 SOC 순서와 일치),
   `q_frac_wide`/`vqslope`(존 시작="hi" 컨벤션)는 **충전** 쪽이 반대로 붙어있었고
   (`chg_hi`가 실제로는 SOC 낮은 충전 초반), `q_abs`/`rcs`(존 시작="low" 컨벤션)는
   **방전** 쪽이 반대였다(`dis_lo`가 실제로는 SOC 높은 방전 초반) — 두 그룹이 정확히
   반대 방향으로 틀려 있었다. 4개 축 모두 `_ROUTING`을 SOC 정합적으로 고쳤다(각 파일
   상단 `[SOC 정합성 — 2026-07-30 수정]` 참조) — 이제 모든 축에서 `chg_lo`/`dis_hi`=
   SOC 낮은/높은 쪽을 일관되게 가리킨다. **주의**: 이 버그는 존 경계·latent_class·
   실제 라우팅되는 회귀 헤드에는 전혀 영향이 없었다(순수 표시 문자열 문제) — 기존
   런의 RMSE/R² 등 수치는 그대로 유효하다. 다만 기존 런 폴더(`scenario_spec.json`
   등)엔 옛 이름이 박혀있으므로, 수정 전/후 런을 `visualize_results.py`로 비교하면
   `scenario_names`가 달라 축 간 비교 모드(§ 코드 참조)로 처리된다.

### 11.2 세그먼트(윈도우) 분할에서 조심할 점

1. **세그먼트 HI에 그 세그먼트/사이클의 실측 용량 자체를 특징으로 넣지 않는다.** 이미
   `loss.leak_cols: ["stat_q_abs", "stat_energy_seg"]`로 명시적으로 제외되고 있다 — 이
   목록을 축소하거나 우회하지 않는다.

2. **세그먼트 길이(`n2`/`seg_len`)를 test 셀의 관측 가능 범위에 맞춰 사후적으로 정하지
   않는다** — §11.1-②와 같은 원인(B)의 세그먼트 레벨 버전이다. "가장 노화된 사이클도 이
   길이는 만족하도록 정한다"는 원칙은 맞지만, 그 관찰은 train/calibration 셀로만 한다.
   고정된 값을 이후 전체에 동일 적용하는 것은 `q_frac_wide`/`q_abs`의 `seg_len ≤ 최소
   zone폭` 제약(코드로 강제됨)과 같은 정신이다.

3. **세그먼트가 너무 길면 그 자체로 총 용량 정보에 근접해 간접 누출이 된다.**
   극단적으로 `seg_len`이 zone 폭 전체에 가까워지면, 그 zone의 "총 델타 용량"과 사실상
   동일해지고 이는 이미 SOH와 강하게 상관된 축약 통계가 될 수 있다 — 특히 `q_frac_wide`류는
   원래도 `q_tot` 문제가 있어 이중으로 위험하다. 반대로 너무 짧으면(`q_abs` 실전 학습:
   `seg_len=0.05`에서 시나리오 분류 자체가 붕괴, `docs/260730_RESULTS.md` §2 참조)
   정보 부족으로 회귀·분류 둘 다 무너진다. "누수 없이, 그러나 정보는 충분히" 사이의
   균형은 반드시 실측(oracle routing 성능 vs hard routing 성능 비교)으로 확인해야 한다.

4. **경계 비침범(segment ⊂ zone) 제약은 라벨 얽힘을 막는 안전장치이지, 그 자체로 정보
   누수가 없다는 뜻은 아니다.** `q_abs`가 이 제약을 100% 만족해도(실측 경계침범=0건)
   §11.1-③의 선택 편향은 별개로 남는다 — "경계를 안 넘었다"와 "라벨과 무관하다"는
   다른 말이다.

5. **랜덤 세그먼트(`random_segment`)에서 창 길이를 zone 폭에 맞춰 클리핑하지 않는다.**
   §4 "공통 옵션" 3대 원칙과 동일 — 클리핑하면 세그먼트 길이 자체가 그 사이클 상태
   (`q_tot` 등)의 함수가 되어 원인(A)가 재발한다. 존이 창보다 짧으면 존 전체를 1개
   세그먼트로 대체(fallback)하는 것이 맞다.

6. **"노화 시 도달 못 하는 구간을 배치 범위에서 아예 빼는" 방식으로 §11.1-③ 선택
   편향을 고치려 하지 않는다 — 다른 종류의 편향을 만든다.** `q_abs`의 `adaptive_samples`
   설계 중 실제로 시도했다가 폐기한 접근: high 존의 배치 상한을 노화 셀 평균 EOL
   잔존용량(예: BOL 대비 80%)으로 clip하면, 그 clip 범위 안에는 **노화 사이클의
   CV(그 사이클 자신의 충전 끝자락 = 낮은 BOL 비율대)만 남고, 건강 사이클의 CV(더
   높은 BOL 비율대)는 통째로 배제**된다. CV 구간은 항상 "그 사이클 자신의" 충전
   끝에서 발생하므로 BOL 비율 기준 clip과는 위치가 안 맞는다 — 결과적으로 그
   시나리오 헤드 안에서 "CV 형상으로 보인다 ↔ 노화된 샘플이다"라는, clip이 인위적으로
   만들어낸 상관관계가 생기고, 동시에 건강 사이클의 실측 데이터를 버리므로 요구조건
   "노화 정도와 무관하게 데이터를 최대한 활용" 에도 역행한다. → 대안: 배치 **개수**만
   존 폭에 맞춰 줄이고(목표 겹침률 기반, 본문 요구조건3), 배치 **위치**는 zone 전체
   범위를 그대로 쓴다. 노화 사이클이 상단 세그먼트에서 `min_pts` 미달로 자연 스킵되는
   현상 자체는 그대로 남지만(§11.1-③은 미해결), 이를 세그멘터 내부에서 숨기지 말고
   `n_attempted`/`n_yielded` 카운터로 노출한 뒤 `mid_end`/`seg_len`을 존별 생존율
   실측(`seg_diagnose`)에 맞춰 보수적으로 고르는 쪽으로 대응한다.

### 11.3 축별 누수 원인 대응표

| 축 | 원인(A) 라벨 누수 | 원인(B) 설계 튜닝 시 test 참조 위험 | 원인(C) 선택 편향 |
|---|:---:|:---:|:---:|
| `qfrac` / `q_frac_wide` | **있음**(`q_frac=q/q_tot`) | n1/n2를 넓은 셀 집합으로 정하면 낮음 | zone 겹침(n1>1/3)으로 인한 라벨 충돌(§4 축5)과는 별개 이슈 |
| `vwindow` / `vqslope` | 없음(순간 전압·형상 기준) | 낮음(전압 경계는 화학종 고정값) | `vqslope` 플래토 미검출 시 존 전체 스킵 — 노화·노이즈와 상관 가능 |
| `q_abs` | 없음(BOL 고정 기준) | **있음** — `mid_start`/`mid_end`/`seg_len`을 관측된 fade 범위로 정할 때 주의 | **있음** — 고정 경계가 노화 시 plateau를 놓치는 zone에서 존재 자체가 결측(`docs/260730_RESULTS.md`) |
| `rcs` | 없음(창 길이 고정, 위치 랜덤) | 낮음 | 낮음(랜덤 위치라 특정 zone 편향 적음) |
