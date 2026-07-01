# HI 피처 이슈 & 수정 기록

대상 파일: `4_hi_analysis/hi_correlation.py`  
HI 풀: Global 15 + 6구간 × 66 = **411D**

---

## 전체 이슈 요약

| 등급 | ID | 피처 | 문제 | 코드 수정 |
|-----|-----|------|------|---------|
| 🔴 Critical | Bug-1/2 | D18/D19 `dvdq_peak_q` / `dvdq_valley_q` | 방전 구간에서 두 피처가 수학적으로 동일 | **완료** |
| 🔴 Critical | Fix-1 | `ent_v` | `density=True` → PDF 밀도값, log 음수 | **완료** |
| 🔴 Critical | Fix-2 | `mean_dvdt` / `var_dvdt` | FP 아티팩트(7ps) 통과 → 1×10⁷ V/s | **완료** |
| 🔴 Critical | Fix-3 | `dtdt` | 동일 FP 아티팩트 → 1638 °C/s | **완료** |
| 🔴 Critical | Fix-4 | `v_total_var` | 빠른 로깅 노이즈 누적 | **완료** |
| 🟠 High | Fix-5 | `temp_rise_per_ah` | q_tot 임계값 부족 | **완료** |
| 🟠 High | Fix-6 | `corr_vt` | 온도 std 최소값 없음 | **완료** |
| 🟠 High | Risk-1 | S11 `q_abs`, S12 `energy_seg` | SOH 타겟 누수 (구조적) | 훈련 단계 필터링 필요 |
| 🟠 High | Risk-2 | S17 `v_samp_ent` | 로깅 주파수 의존 → 데이터셋 간 비교 불가 | 설계 변경 권장 (보류) |
| 🟠 High | Risk-3 | 전 구간 공통 | HUST 부분 사이클에서 q_frac SoC 매핑 오류 | 구조적 한계, 문서화 |
| 🟡 Medium | Des-1 | L16/L09 | `plateau_v_slope` ≈ `ocv_slope` 중복 가능성 | ε² 분석 후 결정 |
| 🟡 Medium | Des-2 | L20 `v_ent_plateau` | 10-bin 고정 + 수mV 범위 → 양자화 노이즈 지배 | 설계 변경 권장 (보류) |
| 🟡 Medium | Des-3 | D16/D17 | ICA valley 20% threshold 문헌 근거 없음 | 문서화 |
| 🟡 Medium | Des-4 | D15 `r_dyn_seg` | MIT 정전류 방전에서 구조적 NaN | 문서화 |
| 🟢 Low | Low-1 | 5곳 | `np.trapz` → `np.trapezoid` NumPy 2.0 경고 | **완료** |
| 🟢 Low | Low-2 | Morph BOL | 사이클 1 이상 시 기준값 오염 가능성 | 설계 변경 권장 (보류) |

---

## 배경: MIT 가변 로깅 속도

MIT 배터리 데이터는 **가변 로깅 속도**를 사용한다.

| 구간 | 로깅 간격 | 비율 |
|------|-----------|------|
| 전환/과도 구간 (CC 시작, 페이즈 전환) | ~0.01–0.1 s | ~40% 행 |
| 정상 플래토 구간 | ~3–4 s | ~60% 행 |

이 혼합 구조에서 단순 row-by-row 계산은 두 가지 문제를 일으킨다.

1. **FP 아티팩트**: 동일 타임스탬프가 부동소수점 오차로 `dt = 7e-12 s` 수준의 극소값을 가짐
2. **빠른 로깅 노이즈**: `dt ~ 0.01 s` 구간에서 측정 노이즈 또는 양자화 오차가 rate로 증폭됨

---

## 🔴 Critical / 🟠 High — 코드 수정 완료

### Bug-1 / Bug-2: D18 ≡ D19 for discharge (방전 구간 수식 동치 오류)

**위치**: `_seg_diff()` 내 D18–D19 블록

**문제**:
```python
# 수정 전
out[f"diff_dvdq_peak_q_{seg}"]   = float(qm_f18[int(np.argmax(np.abs(dv_f18)))])
out[f"diff_dvdq_valley_q_{seg}"] = float(qm_f18[int(np.argmin(dv_f18))])
```

방전 구간 전체에서 `dvdq_sm < 0` (V↓, Q↑이므로 dV/dQ < 0).  
이때 다음 등식이 성립:

```
argmax(|dV/dQ|) = argmax(−dV/dQ) = argmin(dV/dQ)
```

결과: `dis_hi / dis_mid / dis_lo` 3개 구간 모두에서 D18 ≡ D19 (완전히 동일한 숫자 출력).  
충전 구간(dV/dQ > 0)에서만 두 피처가 구분됨.

**D19 설명 불일치**: "V-Q 기울기가 **가장 평탄한** 지점의 Q (플래토 중심 Q 위치)"이지만,  
`argmin(dV/dQ)`는 방전에서 기울기가 **가장 가파른** 지점(전압 급락부)을 반환.

**수정**:
```python
# 수정 후
out[f"diff_dvdq_peak_q_{seg}"]   = float(qm_f18[int(np.argmax(np.abs(dv_f18)))])   # D18: 변경 없음
out[f"diff_dvdq_valley_q_{seg}"] = float(qm_f18[int(np.argmin(np.abs(dv_f18)))])   # D19: abs 추가
```

`argmin(|dV/dQ|)` = 절대 기울기가 가장 작은 점 = 방전/충전 모두 "가장 평탄한 Q 위치".

> **주의**: Global `G09 dva_valley_q`는 `_global_dva()`에서 `argmin(dV/dQ)` 사용.  
> 이는 전체 DVA 곡선에서 Berecibar 2016 DVVL 플래토 진입 경계를 찾는 것으로 올바름.  
> 세그먼트 수준의 D19만 오류이며 G09는 수정 불필요.

---

### Fix-1: `ent_v` — histogram density 버그로 인한 거대 음수 엔트로피

**증상**: `ent_v_chg_s_hi` 등에서 대규모 음수값 발생 (엔트로피는 0 이상이어야 함)

**원인**: `np.histogram(..., density=True)` 사용 시 PDF 밀도(합≠1)를 반환.  
CV 구간처럼 전압 변화폭이 좁으면 특정 bin의 밀도값 >> 1 → `log(p) > 0` → `-p·log(p) < 0` → 음수 엔트로피.

```python
# 이전 (buggy)
counts, _ = np.histogram(vs, bins=10, density=True)
p = counts / counts.sum()
```

**수정**:
```python
# PMF 직접 계산: 합 = 1 보장
_counts = np.histogram(vs, bins=10)[0].astype(float)
_total  = _counts.sum()
if _total > 0:
    p = _counts[_counts > 0] / _total
    out[f"ent_v_{seg}"] = float(-np.sum(p * np.log(p)))
```

**효과**: MIT 전체 `n_negative = 0`, 범위 [0.01, 2.30] (bits)

---

### Fix-2: `mean_dvdt` / `var_dvdt` — FP 아티팩트 및 가변 로깅 편향

**증상**: `mean_dvdt_s_hi` 에서 `-10,485,846 V/s` (기댓값 `-0.001 V/s`)

**원인 (3단계 진단)**:

| 단계 | 코드 | 문제 |
|------|------|------|
| 초기 | `dt_pos = np.maximum(dts[1:], 1e-6)` | dt=7e-12s → 1e-6s 바닥 적용 → `0.01V/1e-6s = 10,000 V/s` |
| 1차 수정 | `valid = dt_seg > 0` | 7e-12s는 0보다 크므로 여전히 통과 → 이상치 잔존 |
| 2차 수정 | `valid = dt_seg >= 1.0` | 빠른 로깅 행(40%) 전체 제거 → 플래토만 선택 → 16배 과소평가 편향 |

b1c9 c592: `dt = 7.275958e-12 s`, `dV = -0.01 V` → `dvdt = -1.35×10⁹ V/s`  
b1c31 c865 chg_s_lo: 총 기울기 +0.0075 V/s, `dt>=1s` 방식 +0.000137 V/s (16배 과소)

**최종 수정**:
```python
# mean_dvdt: (V_end - V_start) / 세그먼트 총 시간 — dt 필터 없이 편향 없음
dt_total = float(np.sum(dts))
if n >= 2 and dt_total >= 1.0:
    out[f"mean_dvdt_{seg}"] = float(vs[-1] - vs[0]) / dt_total

# var_dvdt: dt >= 1s 행(플래토)만 사용 — transient 제외, FP 아티팩트 제외
dt_seg = dts[1:]
dv_seg = np.diff(vs)
valid  = dt_seg >= 1.0
if valid.sum() >= 2:
    dvdt = dv_seg[valid] / dt_seg[valid]
    out[f"var_dvdt_{seg}"] = float(np.var(dvdt))
```

**효과**: `n_extreme(|v|>10) = 23 → 0`, 범위 [-0.003, +0.008] V/s

---

### Fix-3: `dtdt` — 동일한 FP 버그 수정

**증상**: `dtdt_s_mid` p50 = 35.5 °C/s, max = 1638 °C/s

**원인**: `np.maximum(dts[1:], 1e-6)` — Fix-2와 동일한 버그

**수정**:
```python
# dtdt: (T_last - T_first) / 세그먼트 총 시간 (총 기울기 방식)
if dt_total >= 1.0:
    out[f"dtdt_{seg}"] = float(tmps[last_i] - tmps[first_i]) / dt_total
```

**효과**: 물리적 범위 ±0.05 °C/s로 수렴

---

### Fix-4: `v_total_var` — 느린 로깅 행만 사용

**증상**: `v_total_var_s_hi` max = 4.21 V, median = 0.48 V

**원인**: `np.sum(np.abs(np.diff(vs)))` — 빠른 로깅 행의 측정 노이즈(±0.001 V)가 N행 × 0.001 V로 누적.

**수정**:
```python
slow_mask = dts >= 1.0
vs_slow   = vs[slow_mask]
if len(vs_slow) > 1:
    out[f"v_total_var_{seg}"] = float(np.sum(np.abs(np.diff(vs_slow))))
```

---

### Fix-5: `temp_rise_per_ah` — q_tot 최소값 강화

**증상**: max = ±35 °C/Ah (q_tot이 작은 짧은 세그먼트에서 발생)

**수정**: `q_tot > 0.005` → `q_tot > 0.05` (10배 강화)

---

### Fix-6: `corr_vt` — 온도 std 최소값 추가

**증상**: 온도가 거의 변하지 않는 세그먼트(std ≈ 0.01°C)에서 무의미한 ±1

**수정**:
```python
if np.std(vs_ft) > 1e-6 and np.std(tmps_ft) > 0.05:  # 50mK 최소 온도 변동
    out[f"corr_vt_{seg}"] = float(np.corrcoef(vs_ft, tmps_ft)[0, 1])
```

---

### Low-1: `np.trapz` → `np.trapezoid` (NumPy 2.0 deprecation)

5개 위치에서 교체 완료:

| 위치 | 피처 |
|------|------|
| Global ICA | `peak_area` |
| D05 | `dvdq_area` |
| D09 | `dqdv_area` |
| D20 (left/right) | `dqdv_area_asym` |

---

## 🟠 High — 미수정 (파이프라인/설계 레벨 대응 필요)

### Risk-1: S11 `q_abs`, S12 `energy_seg` — SOH 타겟 누수

```
q_abs_dis_mid = ∫|I|dt / 3600  ∝ Q_total  (= SOH 예측 타겟)
```

within-cell Spearman ρ가 높은 것은 전기화학 정보가 아닌 수식의 대수적 구조 때문.  
`q_abs_s_hi / capacity_Ah = 0.363 ± 0.005 (변동 1.3%)` — 사실상 capacity의 상수 배율.

**대응**: `5_train/` 피처 필터링 단계에서 제거:
```python
EXCLUDE_STRUCTURAL = [c for c in feat_cols if c.startswith("stat_q_abs_") or c.startswith("stat_energy_seg_")]
```

---

### Risk-2: S17 `v_samp_ent` — 로깅 주파수 의존성

| 데이터셋 | 로깅 Δt | 유효 Δt_eff | SampEn 템플릿 길이(m=2) |
|---------|---------|-----------|----------------------|
| MIT (10s) | 10s | 10s | **20s** |
| HUST (1s) | 1s | 3s | **6s** |

같은 배터리라도 두 데이터셋에서 서로 다른 주파수 성분을 측정 → 공동 학습 시 데이터셋 식별자처럼 작동 위험.

**권장 수정 방향** (보류):
```python
# 고정 시간 간격(10s)으로 보간 후 계산
t_grid = np.arange(0, t_tot_seg, 10.0)
if len(t_grid) >= 10:
    xs = np.interp(t_grid, t_seg, vs)
```

---

### Risk-3: HUST 부분 사이클 — q_frac SoC 매핑 오류

`q_frac = Q_cumsum / Q_segment_total` — Q_segment_total은 해당 사이클 실제 방전량.

| 사이클 유형 | dis_mid 실제 SoC 범위 |
|-----------|-------------------|
| MIT 완전 방전 (100→0%) | 약 60→30% (의도된 범위) |
| HUST 부분 방전 (80→40%) | 약 56→32% (±10% 미끄러짐) |
| HUST 부분 방전 (60→20%) | 약 44→20% (완전히 다른 범위) |

같은 `dis_mid` 레이블이 데이터셋별로 다른 절대 SoC 구간을 대표. 구조적 한계.

---

## 🟡 Medium — 설계 주의 사항

### Des-1: L16 `plateau_v_slope` ≈ L09 `ocv_slope` 중복 가능성

- **L09 `ocv_slope`**: V-Q 곡선에서 q_mid 단일 빈의 도함수 `dvdq_sm[q_mid_idx]`
- **L16 `plateau_v_slope`**: 플래토 마스크 전체의 OLS slope (V vs Q)

LFP 플래토에서 V-Q가 선형에 가까우면 Pearson r(L09, L16) > 0.95 예상.  
`seg_corr_analysis.py`의 ε² 분석 결과 확인 후 L16 제거 검토.

---

### Des-2: L20 `v_ent_plateau` 양자화 노이즈 지배

건강한 LFP 방전 플래토의 전압 범위: **1~5 mV**.  
`np.histogram(v_sm[plt_mask], bins=10)` → 빈 폭 ≈ 0.1~0.5 mV.  
ADC 분해능과 로깅 노이즈가 이 스케일에서 분포를 결정 → 엔트로피가 물리적 의미보다 측정 불확실도를 반영.

**권장 수정 방향** (보류): 고정 빈 폭(1mV) 사용.

---

### Des-3: D16/D17 ICA valley 20% threshold 미검증

`lh <= 0.2 * pk16_h` — 피크 높이 대비 20% 이하만 유효 밸리로 인정.  
문헌 근거 없음; LFP ICA 피크가 넓고 완만한 구간에서 어깨값이 25~30%이면 NaN 반환.  
향후 0.2 → 0.3 감도 실험 권장.

---

### Des-4: D15 `r_dyn_seg` MIT 정전류 방전 구조적 NaN

```python
valid = (|ΔI| > 0.01 A) & (Δt < 2.0s)
```

MIT 방전: 정전류(CC) → ΔI ≈ 0 → 방전 3구간 × D15 = 3개 열 전체 NaN.  
충전 측 또는 HUST 다단계 프로토콜 전용 피처.

---

## 🟢 Low — 설계 개선 권장

### Low-2: Morph BOL 기준 사이클 이상 취약성

현재: `q_local >= cap * 0.30` 조건을 만족하는 첫 번째 사이클 = BOL 참조.  
사이클 1이 컨디셔닝(conditioning) 사이클이면 V-Q 곡선 형태가 정상과 다를 수 있음.

**권장**: 첫 3개 유효 사이클 V-Q 곡선 평균을 BOL 참조로 사용.

---

## 수정 전후 비교 (MIT 데이터)

| 피처 | 수정 전 | 수정 후 | 방법 |
|------|---------|---------|------|
| `ent_v_*` | 음수값 다수 | [0.01, 2.30] bits | PMF 방식으로 교체 |
| `mean_dvdt_*` | 최대 1.05×10⁷ V/s | 최대 0.008 V/s | 총 기울기 방식 |
| `var_dvdt_*` | 최대 1.41×10¹⁶ | 최대 3.8×10⁻⁴ | dt≥1s 행만 사용 |
| `dtdt_*` | p50=35°C/s, 최대 1638°C/s | 최대 ~0.05°C/s | 총 기울기 방식 |
| `v_total_var_*` | 최대 4.21 V | 최대 ~0.5 V | dt≥1s 행만 사용 |
| `temp_rise_per_ah_*` | 최대 ±35°C/Ah | 최대 ±15°C/Ah | q_tot > 0.05 Ah 강화 |
| `corr_vt_*` | 노이즈 지배 ±1 | 유의미한 ±1 | std(T) > 0.05°C 조건 추가 |
| `diff_dvdq_valley_q_*` (D19, 방전) | D18과 동일 | min\|dV/dQ\|의 Q 위치 | `argmin(abs(dv))` 수정 |

---

## 주의사항

- 가변 로깅 수정(Fix-1~6)은 **HUST 데이터에도 동일하게 적용**된다. HUST의 로깅 간격이 다를 경우 (`dt >= 1s` 조건에서 유효 행이 없을 수 있음) → NaN 반환으로 안전하게 처리됨.
- `dt >= 1s` 기준은 MIT의 정상 플래토 로깅(~4s)을 기준으로 설정. 향후 다른 데이터셋 추가 시 재검토 필요.

---

## 변경 이력

| 날짜 | 파일 | 변경 내용 |
|------|------|---------|
| 2026-06-29 | `hi_correlation.py` | Fix-1~6: 가변 로깅 관련 이상치 수정 (ent_v, dvdt, dtdt, v_total_var, temp_rise, corr_vt) |
| 2026-06-30 | `hi_correlation.py` | D19 `argmin(dv)` → `argmin(abs(dv))` (Bug-1/2 수정) |
| 2026-06-30 | `hi_correlation.py` | `np.trapz` → `np.trapezoid` 5곳 (Low-1 수정) |
