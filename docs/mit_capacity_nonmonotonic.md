# MIT 데이터 capacity 단조성 위반 — 원인 분석 및 처리

대상 파일: `2_preprocess/preprocess.py` — `_remove_outlier_cycles()`

---

## 현상

MIT 배터리 데이터의 방전 용량(`capacity_Ah`) 시계열은 LFP 열화 메커니즘상 단조 감소해야 한다.
그러나 b1c0처럼 일부 셀에서 **특정 사이클만 2–3% 급강하한 뒤 다음 사이클에 즉시 회복**하는
비단조 패턴이 관측된다.

```
cycle 600:  1.0548 Ah   ← 정상
cycle 610:  1.0291 Ah   ← -2.6% 급강하
cycle 611:  [삭제됨]
cycle 612:  1.0330 Ah   ← 여전히 낮음
cycle 613:  1.0574 Ah   ← 즉시 회복
...
cycle 652:  1.0168 Ah   ← 최대 -2.7% 급강하 (b1c0 최악)
cycle 653:  1.0458 Ah   ← 즉시 회복
```

b1c0 기준 1841사이클 중 19개 사이클이 rolling median 대비 2% 이상 낮게 측정됐다.

---

## capacity_Ah 컬럼의 실체

`capacity_Ah`는 전류 적산값이 아니다.

```python
# 1_convert/convert_unified.py
QD_arr = _load_summary_field(f, s_grp, "QDischarge")   # MIT 배터리 요약 테이블
...
"capacity_Ah": float(QD) if np.isfinite(QD) else np.nan,  # 사이클 전 행에 상수로 저장
```

MIT 원본 HDF5의 `summary.QDischarge`는 **BMS 하드웨어 Coulomb 카운터로 측정한 해당 사이클의 방전 용량**이다.
이 값은 사이클 내 모든 행(충전·방전·휴지)에 동일하게 저장되며, 방전 행 첫 번째 값으로 읽힌다.

```python
# hi_correlation.py — SOH 레이블로 사용
cap = float(dis["capacity_Ah"].iloc[0])
```

---

## 원인: 충전 인터럽트 이벤트

LFP 배터리의 방전 용량은 **직전 충전에서 실제로 저장된 에너지의 양**에 의존한다.
정상 사이클에서는 CC1 → (66 s 휴지) → CC2/CV 순서로 완전 충전되지만,
실험 중 아래와 같은 이벤트가 발생하면 충전이 중단된다.

- 전원 순간 차단 (전압 강하 → 충전기 재시작)
- 프로토콜 리셋 (실험 소프트웨어 재시작)
- 장비 트리거 오류 (전류 클램프 이상 감지 후 자동 중단)

충전이 중단되면 배터리에 저장된 에너지가 줄어들어 그 사이클의 방전 용량이 낮게 기록된다.
이는 **실제 SOH와 무관한 측정 아티팩트**다.

### 패턴 특징

| 특징 | 설명 |
|------|------|
| 급강하 → 즉시 회복 | 인접 사이클은 정상 → 실제 열화가 아님 |
| 결측 사이클이 인접 | 충전 인터럽트가 심하면 해당 사이클 자체가 삭제됨 (F4B 필터) |
| 산발적 등장 | 약 10–50 사이클 간격으로 불규칙 발생 |
| 클러스터 등장 | 같은 인터럽트 이벤트가 연속 2–5 사이클에 영향을 줌 |

```
b1c0 케이스: 결측 사이클 직전의 용량
  Missing 611 → 직전 cycle 610: 1.0291 (정상 대비 -2.6%)
  Missing 639 → 직전 cycle 638: 1.0296 (정상 대비 -2.4%)
  Missing 832 → 직전 cycle 831: 1.0199 (정상 대비 -2.3%)
```

결측 사이클(F4B: 충전 완전 중단으로 삭제)이 생길 만큼 심각한 인터럽트는 앞 사이클에도 영향을 준다.

---

## 기존 필터가 놓친 이유

`_remove_outlier_cycles` Pass 1 (window=11, σ=2.5) 기준:

- **클러스터 마스킹**: 비정상 사이클이 3–5개씩 묶이면 window=11 rolling median이 그 쪽으로 당겨짐
  → 개별 이상치의 residual이 σ 임계값 내로 들어옴
- **σ 대역 확장**: 클러스터가 포함된 11개 윈도우의 std가 올라가 ±2.5σ 대역이 넓어짐

```
cycle 652 예시:
  rolling median(window=11) ≈ 1.0440  (주변 저용량 사이클에 당겨짐)
  rolling std   ≈ 0.013
  하한 임계값   = 1.0440 - 2.5×0.013 = 1.0115
  cycle 652 값  = 1.0168  →  1.0168 > 1.0115  →  통과 (미제거)
```

---

## 수정: 2-pass 필터

`preprocess.py`의 `_remove_outlier_cycles` 함수에 Pass 2를 추가.

```python
# Pass 1 (기존): window=11, σ=2.5 — 고립 이상치 제거
roll1  = cap_s.rolling(window=11, center=True, min_periods=3)
r_med1 = roll1.median()
r_std1 = roll1.std().fillna(cap_s.std()).clip(lower=min_std)
rm1    = set(cap_s[(cap_s - r_med1).abs() > 2.5 * r_std1].index)

# Pass 2 (신규): window=31, σ=2.0 — 클러스터 이상치 제거
cap2   = cap_s.drop(index=rm1)                          # Pass 1 제거 후 데이터
roll2  = cap2.rolling(window=31, center=True, min_periods=5)
r_med2 = roll2.median().reindex(cap_s.index).ffill().bfill()
r_std2 = roll2.std().reindex(cap_s.index).ffill().bfill().clip(lower=min_std)
rm2    = set(cap_s[(cap_s - r_med2).abs() > 2.0 * r_std2].index) - rm1
```

Pass 1 제거 후 남은 데이터에서 더 넓은 window(31)로 rolling median을 재계산하면,
클러스터 사이클들이 median을 당기는 효과가 줄어 잔여 이상치를 잡아낸다.

---

## 수정 효과 (MIT 전체)

| 셀 | 기존 제거 | 수정 후 제거 | 추가 |
|----|----------|------------|------|
| b1c0 | 1건 | 19건 | +18건 |
| b2c13 | 0건 | 1건 | +1건 |
| b2c17 | 0건 | 1건 | +1건 |
| **HUST 전체** | 0건 | **0건** | 영향 없음 |

b1c0에서 제거된 19개 사이클 위치:
`[408, 433, 512, 527, 610, 612, 638, 652, 701, 705, 706, 710, 748, 757, 758, 831, 848, 1717, 1727]`

사이클 1717, 1727은 수명 말기(EoL) 구간에서 같은 패턴(급강하 → 즉시 회복)으로 동일 원인.

---

## 주의사항

- 이 필터는 **실험 아티팩트 제거** 목적이며, 실제 SOH 급락(knee point, 갑작스러운 가속 열화)과 구분이 필요하다.
  - 아티팩트: 1–3사이클 드롭 후 즉시 원래 수준 회복
  - 실제 열화 가속: 회복 없이 지속 감소
  2-pass 필터는 "즉시 회복" 패턴만 제거하므로 실제 열화 가속에는 영향을 주지 않는다.

- `capacity_Ah`와 전류 적산값(`q_local`)의 차이 (~10%)는 별개 이슈다. 자세한 내용은 [[correlation_comparison]] 참조.
