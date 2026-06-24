# DATASET_ANOMALIES.md

데이터 파이프라인 각 단계에서 발생하는 이상 유형과 처리 방법을 단계별로 정리.

---

## 데이터셋 기본 정보

| 항목 | MIT FastCharge | HUST |
|------|---------------|------|
| 원본 셀 수 | 128 (batch1 25 + batch2 45 + batch3 44, CONTINUING 5셀은 batch1에 병합) | 77 |
| 변환 후 (`data_unified`) | **123** (12셀 DELETE_CELLS 제외) | **77** |
| 배터리 화학계 | LFP 18650 | LFP (공칭 1.1 Ah) |
| 방전 전류 부호 규약 | 음수 (−) | 음수 (−) |

---

## 단계별 제거 통계 요약

> 기준: `data_raw` 원본 사이클 수 (MIT **114,738** / HUST **146,122**)

| 단계 | 필터 | 단위 | MIT 제거 | MIT % | HUST 제거 | HUST % |
|------|------|------|---------|-------|---------|-------|
| Step 1-A | DELETE_CELLS | 셀 | 12 셀 / **14,927 cycle** | **13.0%** | 0 | — |
| Step 1-B | `_remove_empty_cycles` | 사이클 | **40** | 0.03% | 0 | — |
| Step 1-D | `_remove_zero_current_rest` | **행(row)** | **5,126,620 rows** | — | 0 | — |
| Step 1-E | `_remove_outlier_cycles` | 사이클 | **23** | 0.02% | 0 | — |
| **Step 1 후** | data_unified | — | **123셀 / 99,748 cycles** | 잔존 **87.0%** | **77셀 / 146,122 cycles** | 잔존 **100%** |
| Step 2-1 | `_remove_empty_cycles` | 사이클 | (Step 1 이후 잔존분) | — | 0 | — |
| Step 2-2 | `_fix_time_monotonicity` | 행 | (행 수 유지) | — | — | — |
| Step 2-3 | `_remove_zero_current_rest` | **행(row)** | (Step 1 이후 잔존분) | — | 0 | — |
| Step 2-4 | 방전 시간 단절 → 사이클 전체 제거 | 사이클 | **~46** | 0.04% | **~1** | 0.001% |
| Step 2-4 | 충전 시간 단절 → 충전 phase 행 제거 | 행 | **~수천 행** | — | 0 | — |
| Step 2-5 | Rolling Median | 사이클 | **1** | 0.001% | 0 | — |
| Step 2-6 | `vend_min` 1.8V | 사이클 | 0 | — | **~4,214** | **2.88%** |
| **Step 2 후** | data_postprocess | — | **~99,701 cycles** | 잔존 **86.9%** | **~141,907 cycles** | 잔존 **97.1%** |

**주요 포인트**:
- MIT 제거의 대부분(**13.0%p**)은 Step 1-A DELETE_CELLS — 필터 로직이 아닌 수동 제외 (Step 1에서만 처리)
- HUST 제거는 거의 전량 Step 2-6 `vend_min` 비정상 종료 사이클 (**2.88%**)
- Step 2-1~3은 data_unified 에 이미 적용됐으나 Step 2에서 재적용해 안전망 역할
- Step 2-4 충전 단절: 사이클 전체 제거 대신 **충전 phase 행만 제거** — 방전 HI 계산 보존
- `hi_correlation.py` 는 `data_postprocess/` 를 읽으므로 인라인 dt-gap 감지 불필요

---

## Step 1 — `convert_unified.py` 변환 단계

원본 파일(.mat / .pkl)을 파싱해 `data_unified/` 에 저장하는 과정에서 아래 처리가 순서대로 적용됨.  
`data_raw/` 에는 이 처리들이 **적용되지 않은** 원본 파싱 결과가 저장됨.

### 1-A. DELETE_CELLS — 완전 불량 셀 제외

`data_unified/` 저장 자체를 건너뜀. `data_raw/` 에는 포함.

| 셀 | 배치 | 제외 이유 |
|----|------|----------|
| `b1c8`, `b1c10`, `b1c12`, `b1c13`, `b1c22` | batch1 | 측정 오류 / 비정상 프로토콜 |
| `b1c18` | batch1 | rest 구간 전압 오염 — cycle 34~39: 최대 4.6V / cycle 50~53: 0.7~6.6V 무작위값 |
| `b3c2`, `b3c23`, `b3c32`, `b3c37`, `b3c42`, `b3c43` | batch3 | 측정 오류 / 비정상 프로토콜 |

DELETE_CELLS 합계: **12셀** → `data_unified/MIT/` 에는 **123셀** 저장

---

### 1-B. `_remove_empty_cycles()` — 빈 사이클 제거

charge/discharge 행이 5행 미만인 사이클을 제거.

| 대상 | 증상 | 원인 |
|------|------|------|
| MIT Batch1/2 전 셀 — **cycle 1** | rest 2행만 존재, 전압 0.0V, 전류 0.0A | 실험 시작 직후 센서 초기화 아티팩트 |

HUST는 빈 사이클 없음 — 이 단계는 MIT에만 적용.

---

### 1-C. `_make_time_cumulative()` — time_s 누적 보정

사이클별로 0부터 리셋되는 time_s를 셀 전체 연속 시간으로 변환. 두 가지 문제를 동시 보정:

| 유형 | 원인 | 보정 방법 |
|------|------|----------|
| 사이클 내 역방향 점프 | MIT 원본 타임스탬프가 분(min) 단위 저장 → 정밀도 손실 | `np.maximum.accumulate` 로 단조 증가 강제 |
| 사이클 간 time_s 리셋 | 각 사이클 time_s가 0부터 시작 | 이전 사이클 끝 시각을 누적 오프셋으로 가산 |

영향 셀: MIT 79셀 / 209사이클 (주로 Batch1/2). HUST는 영향 없음.

> `data_raw/` 에는 누적 시간이 미적용된 상태(사이클별 상대 시간)로 저장.  
> 캐시 모드(--no-cache 없이 재실행)에서는 `data_raw/` 를 읽은 뒤 이 단계부터 동일하게 적용.

---

### 1-D. `_remove_zero_current_rest()` — rest 0전류 행 제거

phase가 `rest` 이고 `current_A == 0.0` 인 행을 제거.  
`_make_time_cumulative` **이후** 적용 — 누적 시간 확정 후 제거해 time_s 불연속 없음.

---

### 1-E. `_remove_outlier_cycles()` — Rolling Median 이상 사이클 제거

방전 용량 시계열에서 이상 사이클을 탐지·제거.

```
파라미터: window=11, sigma=2.5, min_std=0.01 Ah
판정: |capacity_Ah − rolling_median| > 2.5 × rolling_std → 제거
```

| 탐지 대상 | 방향 | 주요 사례 |
|-----------|------|----------|
| RPT 저전류 측정 사이클 | 상단 (+) | Batch3 마지막 사이클: 정규 ~0.88 Ah → RPT ~1.76 Ah |
| HPPC·펄스 진단 사이클 | 하단 (−) | MIT 일부 셀 중간: 전류 양/음 반전, 용량 비정상 저하 |
| 일반 측정 오류 스파이크 | 양방향 | HUST 랜덤 이상치 |

MIT/HUST 양쪽 모두 적용.

---

### Step 1 처리 순서 요약

```
원본 파싱 (MAT / PKL)
  │
  ├─▶ data_raw/ 저장 (필터 없음, DELETE_CELLS 포함, 누적 시간 미적용)
  │
  ▼ DELETE_CELLS → data_unified 저장 건너뜀
  │
  ▼ [MIT만] _remove_empty_cycles()    — cycle 1 아티팩트 제거
  │
  ▼ _make_time_cumulative()            — time_s 누적 보정
  │
  ▼ _remove_zero_current_rest()        — rest 0전류 행 제거
  │
  ▼ _remove_outlier_cycles()           — RPT / HPPC / 스파이크 제거
  │
  └─▶ data_unified/ 저장
```

---

## Step 2 — `preprocess.py` 전처리 단계

`data_unified/` 를 읽어 6단계 이상 사이클·행 제거를 수행한 뒤 `data_postprocess/` 에 저장.  
원본 `data_unified/` 는 변경하지 않음. `hi_correlation.py` 는 `data_postprocess/` 를 입력으로 사용.

### 필터1 — `_remove_empty_cycles()` — 빈 사이클 제거

charge/discharge 행이 5행 미만인 사이클 제거. MIT cycle 1 아티팩트 방어.

### 필터2 — `_fix_time_monotonicity()` — time_s 단조 보정

사이클 내 time_s 역방향 점프를 `np.maximum.accumulate` 로 제거.  
MIT 원본 타임스탬프가 분(min) 단위 저장 → 정밀도 손실로 역방향 점프 발생.  
사이클 간 오프셋 누적은 적용하지 않음 (사이클마다 시간 초기화).

### 필터3 — `_remove_zero_current_rest()` — rest 0전류 행 제거

phase = `rest` 이고 `current_A == 0.0` 인 행 제거. 필터2 이후 적용.

### 필터4 — `_remove_dt_gap_cycles()` — 시간 단절 사이클 처리

방전/충전 phase 내에서 인접 행 간격(dt)이 비정상적으로 큰 사이클을 처리.

**원인**: 사이클러가 실험 중 일시 중단됐다가 재개될 때, 중단/재시작이 사이클 경계가 아닌 phase 내부 시간 공백으로 기록됨. 이 갭이 적분에 반영되면 `energy_Wh`, `q_abs` 등이 수십~수백 배 과대추정됨.

**판정 기준**:
```python
dt     = clip(diff(t, prepend=t[0]), 0, None)
dt_med = median(dt[dt > 0])
단절   = dt.max() > max(gap_s, dt_med × gap_factor)
```

| phase | 기본 파라미터 | 결과 |
|-------|-------------|------|
| 방전 | `dis_gap_s=600`, `dis_gap_factor=50` | 해당 사이클 **전체 제거** |
| 충전 | `chg_gap_s=120`, `chg_gap_factor=30` | 해당 사이클 **충전 phase 행만 제거** (방전 HI 보존) |

**발생 현황**:

| phase | 데이터셋 | 이상 사이클 수 | 영향 셀 수 | 최대 갭 |
|-------|----------|--------------|-----------|---------|
| 방전 | HUST | 1건 | 1셀 (`1-8` cycle 168) | 36,923s (~10.3h) |
| 방전 | MIT  | 45건 | 24셀 | 29,453s (~8.2h) |
| 충전 | MIT  | ~644건 | ~43셀 | 178,555s (~49.6h) |

```powershell
python 2_preprocess/preprocess.py --dis-gap-s 300 --dis-gap-factor 30  # 더 엄격하게
```

### 필터5 — `_remove_outlier_cycles()` — Rolling Median 이상 사이클 제거

방전 용량 시계열에서 이상 사이클 탐지·제거.

```
파라미터: window=11, sigma=2.5, min_std=0.01 Ah
판정: |capacity_Ah − rolling_median| > 2.5 × rolling_std → 제거
```

| 탐지 대상 | 방향 | 주요 사례 |
|-----------|------|----------|
| RPT 저전류 측정 사이클 | 상단 (+) | Batch3 마지막 사이클: ~0.88 Ah → RPT ~1.76 Ah |
| HPPC·펄스 진단 사이클 | 하단 (−) | MIT 일부 셀 중간: 전류 양/음 반전 |

```powershell
python 2_preprocess/preprocess.py --sigma 2.0   # 더 엄격하게
python 2_preprocess/preprocess.py --window 15   # 더 넓은 윈도우
```

### 필터6 — `_remove_bad_vend_cycles()` — 방전 종지전압 하한

방전 사이클의 마지막 전압이 `vend_min` 미만이면 비정상 종료로 판정하고 제거.

| 항목 | 내용 |
|------|------|
| 기본값 | **1.8 V** (`--vend-min` 으로 조정 가능) |
| 주 대상 | HUST — MIT는 정상 컷오프 2.0V이므로 1.8V 기준에서 영향 없음 |
| HUST 진단 결과 | **4,214건 (2.88%)** — 77셀 중 **41셀** 해당 |
| 최다 발생 셀 | 10-3 (13.2%), 7-1 (11.5%), 7-6 (10.6%) |
| 원인 | 방전이 정상 컷오프 도달 전 비정상 중단 → `v_end`, `v_drop`, `energy_Wh` HI 오염 |

```powershell
python 2_preprocess/preprocess.py --vend-min 1.9  # 더 엄격하게
```

### Step 2 처리 순서 요약

```
data_unified/
  │
  ▼ [필터1] _remove_empty_cycles()         — charge/discharge 5행 미만 사이클 제거
  │
  ▼ [필터2] _fix_time_monotonicity()        — 사이클 내 time_s 역방향 점프 단조 보정
  │
  ▼ [필터3] _remove_zero_current_rest()     — rest 행 중 current_A == 0.0 제거
  │
  ▼ [필터4] _remove_dt_gap_cycles()         — 시간 단절 사이클 처리
  │            방전 단절 → 전체 사이클 제거
  │            충전 단절 → 충전 phase 행만 제거
  │
  ▼ [필터5] _remove_outlier_cycles()        — Rolling Median 이상 사이클 제거
  │
  ▼ [필터6] _remove_bad_vend_cycles()       — 비정상 종료 사이클 (v_end < 1.8V)
  │
  └─▶ data_postprocess/ 저장
        PKL: 전 셀 / CSV: 제거 발생 셀만
        outputs/cleaning_report.csv (필터별 제거 수 + 사이클 번호)
```

---

## Step 3 — `check_integrity.py` 무결성 검사

`data_unified/` 를 대상으로 이상 여부를 검사. 이상치 제거는 수행하지 않음.

| 수준 | 검사 항목 | 설명 |
|------|-----------|------|
| 셀 | 파일 수 | MIT=123, HUST=77 일치 여부 |
| 셀 | `missing_cols` | 필수 컬럼(cycle, time_s, voltage_V …) 누락 |
| 셀 | `invalid_phase` | charge/discharge/rest 외 값 |
| 셀 | `current_direction` | 방전 전류 평균이 양수이면 phase 오류 의심 |
| 셀 | `capacity_increasing` | 말기 용량 > 초기 용량 (병합 오류 의심) |
| 셀 | `high_nan` | 컬럼 NaN 비율 50% 초과 |
| 셀 | `cycle_count_mismatch` | meta.n_cycles ≠ 실제 사이클 수 |
| 사이클 | `voltage_high` / `voltage_low` | V > 4.5V 또는 V < 1.5V |
| 사이클 | `rest_dominant` | rest 행 비율 > 80% |
| 사이클 | `time_nonmono` | time_s 단조 증가 위반 |

---

## Step 4 — `hi_correlation.py` HI 추출 단계

`data_postprocess/` 를 읽어 HI 148종을 사이클별로 계산.  
시간 단절 등 이상 처리는 Step 2에서 완료된 상태이므로 HI 추출 단계에서는 별도 필터링 없음.

> **구버전 Step 4-A (방전/충전 시간 단절 인라인 감지)** 는 **Step 2 [필터4]** 로 통합됨.  
> `hi_correlation.py` 의 `_extract_one_cell` 내 `_dt_med / _chg_gap` 인라인 코드는 제거됨.  
> 상세 내용은 위 Step 2 — 필터4 참조.

---

## 데이터셋별 잔존 주의사항

### MIT

| 항목 | 내용 |
|------|------|
| Batch3 열화율 | SOH 92~95% 수준에서 실험 조기 종료 — 측정 오류 아님 |
| CONTINUING 셀 | b2c7→b1c0 등 5셀이 batch1+batch2 데이터 병합 저장, 사이클 번호 연속 처리 |
| 방전 전류 | 4.0A 단일값으로 균일 (i_max p5~p95: 4.0004~4.0019A) |
| 정상 전압 범위 | 1.983 ~ 3.646V (체크 후 이상 없음) |
| 방전 구간 내 시간 단절 | 45건 / 24셀 — Step 2 [필터4] 에서 사이클 전체 제거 (상세 내용은 Step 2 — 필터4 참조) |

### HUST

| 항목 | 내용 |
|------|------|
| 방전 전류 바이모달 | 3.3A 그룹 (약 33%) / 5.5A 그룹 (약 67%) — 배치별 실험 프로토콜 차이 |
| HI 분석 권장 | 두 전류 그룹을 분리해 상관 분석 고려 |
| 초기 용량 범위 | 1.159 ~ 1.232 Ah (공칭 1.1 Ah 대비 정상) |
| SOH 열화 | 71.7 ~ 75.9% (전 셀 EOL 기준 도달 확인) |
| 정상 전압 범위 | 1.829 ~ 3.602V (체크 후 이상 없음) |
| 방전 구간 내 시간 단절 | 1건 (`1-8` cycle 168) — Step 2 [필터4] 에서 사이클 전체 제거 (상세 내용은 Step 2 — 필터4 참조) |
