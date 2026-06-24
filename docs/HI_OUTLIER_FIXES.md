# HI Feature 이상치 제거 — 원인 분석 및 수정 기록

대상 파일: `4_hi_analysis/hi_correlation.py` — `_seg_extra_his()` 함수

---

## 배경

MIT 배터리 데이터는 **가변 로깅 속도**를 사용한다.

| 구간 | 로깅 간격 | 비율 |
|------|-----------|------|
| 전환/과도 구간 (CC 시작, 페이즈 전환) | ~0.01–0.1 s | ~40% 행 |
| 정상 플래토 구간 | ~3–4 s | ~60% 행 |

이 혼합 구조에서 단순 row-by-row 계산은 두 가지 문제를 일으킨다.

1. **FP 아티팩트**: 동일 타임스탬프가 부동소수점 오차로 `dt = 7e-12 s` 수준의 극소값을 가짐
2. **빠른 로깅 노이즈**: `dt ~ 0.01 s` 구간에서 측정 노이즈 또는 양자화 오차가 rate로 증폭됨

---

## 수정 내역

### 1. `ent_v` — histogram density 버그로 인한 거대 음수 엔트로피

**증상**: `ent_v_chg_s_hi` 등에서 대규모 음수값 발생 (엔트로피는 0 이상이어야 함)

**원인**: `np.histogram(..., density=True)` 사용 시 PDF 밀도(합≠1)를 반환.  
CV 구간처럼 전압 변화폭이 좁으면 특정 bin의 밀도값 >> 1 → `log(p) > 0` → `-p·log(p) < 0` → 음수 엔트로피.

```python
# 이전 (buggy): density=True → bin 밀도, 합 = bin_width × n_bins ≠ 1
counts, _ = np.histogram(vs, bins=10, density=True)
p = counts / counts.sum()   # 이 경우에도 counts 자체가 밀도값이라 log 오염
```

**수정**:
```python
# 확률질량함수(PMF) 직접 계산: 합 = 1 보장
_counts = np.histogram(vs, bins=10)[0].astype(float)
_total  = _counts.sum()
if _total > 0:
    p = _counts[_counts > 0] / _total   # PMF; 0 < p ≤ 1 → log(p) ≤ 0 → 엔트로피 ≥ 0
    out[f"ent_v_{seg}"] = float(-np.sum(p * np.log(p)))
```

**효과**: 현재 PKL 기준 MIT 전체 `n_negative = 0`, 범위 [0.01, 2.30] (bits)

---

### 2. `mean_dvdt` / `var_dvdt` — FP 아티팩트 및 가변 로깅 편향

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

### 3. `dtdt` — 동일한 FP 버그 수정

**증상**: `dtdt_s_mid` p50 = 35.5 °C/s, max = 1638 °C/s (배터리에서 물리적으로 불가능)

**원인**: `np.maximum(dts[1:], 1e-6)` — dvdt와 동일한 버그
```python
dt_pos_t = np.maximum(dts[1:], 1e-6)   # 1e-6s 바닥
dtdt_arr = np.diff(tmps) / dt_pos_t    # ΔT(0.01°C) / 0.01s = 1°C/s (양자화 노이즈)
                                        # ΔT(0.001°C) / 1e-6s = 1000°C/s (FP 아티팩트)
```

MIT 온도계 분해능 ~0.01°C: 빠른 로깅 행(0.01s)에서 연속 측정값이 한 단계(0.01°C) 달라지면
`dtdt = 0.01 / 0.01 = 1°C/s`. p50 = 35.5°C/s → 절반 이상이 노이즈 지배.

**수정**:
```python
# dtdt: (T_last - T_first) / 세그먼트 총 시간 (총 기울기 방식)
if dt_total >= 1.0:
    out[f"dtdt_{seg}"] = float(tmps[last_i] - tmps[first_i]) / dt_total
```

**효과**: 물리적 범위 ±0.05 °C/s로 수렴

---

### 4. `v_total_var` — 느린 로깅 행만 사용

**증상**: `v_total_var_s_hi` max = 4.21 V, median = 0.48 V

**원인**: `np.sum(np.abs(np.diff(vs)))` — 단조 감소라면 `|V_start - V_end|`이어야 하지만,
빠른 로깅 행의 측정 노이즈(±0.001 V)가 N행 × 0.001 V로 누적.
예) 1000개 빠른 로깅 행 × 0.001V = +1.0V 추가

**수정**:
```python
# v_total_var: dt >= 1s 행(플래토)만 사용하여 노이즈 누적 방지
slow_mask = dts >= 1.0
vs_slow   = vs[slow_mask]
if len(vs_slow) > 1:
    out[f"v_total_var_{seg}"] = float(np.sum(np.abs(np.diff(vs_slow))))
```

---

### 5. `temp_rise_per_ah` — q_tot 최소값 강화

**증상**: max = ±35 °C/Ah (q_tot이 작은 짧은 세그먼트에서 발생)

**원인**: `q_tot > 0.005 Ah` 조건이 너무 느슨함.
`q_tot = 0.01 Ah` + `ΔT = 0.3°C` → `30°C/Ah`

**수정**:
```python
if q_tot > 0.05:   # 0.005 → 0.05 Ah (10배 강화)
    out[f"temp_rise_per_ah_{seg}"] = float(tmps[last_i] - tmps[first_i]) / q_tot
```

---

### 6. `corr_vt` — 온도 std 최소값 추가

**증상**: 온도가 거의 변하지 않는 세그먼트(std ≈ 0.01°C)에서 상관계수가 노이즈에 지배
→ 무의미한 ±1에 가까운 값

**원인**: `np.std(tmps_ft) > 1e-6` 조건이 사실상 비어 있음

**수정**:
```python
if np.std(vs_ft) > 1e-6 and np.std(tmps_ft) > 0.05:  # 50mK 최소 온도 변동
    out[f"corr_vt_{seg}"] = float(np.corrcoef(vs_ft, tmps_ft)[0, 1])
```

---

## 수정 전후 비교 (MIT 데이터)

| Feature | 수정 전 | 수정 후 | 비고 |
|---------|---------|---------|------|
| `ent_v_*` | 음수값 다수 | 범위 [0.01, 2.30] | PMF 방식으로 교체 |
| `mean_dvdt_*` | 최대 1.05×10⁷ V/s | 최대 0.008 V/s | 총 기울기 방식으로 교체 |
| `var_dvdt_*` | 최대 1.41×10¹⁶ | 최대 3.8×10⁻⁴ | 플래토 행(dt≥1s)만 사용 |
| `dtdt_*` | p50=35°C/s, 최대 1638°C/s | 최대 ~0.05°C/s | 총 기울기 방식으로 교체 |
| `v_total_var_*` | 최대 4.21 V | 최대 ~0.5 V | 플래토 행(dt≥1s)만 사용 |
| `temp_rise_per_ah_*` | 최대 ±35°C/Ah | 최대 ±15°C/Ah | q_tot > 0.05 Ah 강화 |
| `corr_vt_*` | 노이즈 지배 ±1 | 유의미한 ±1 | std(T) > 0.05°C 조건 추가 |

---

## 주의사항

- 위 수정은 **HUST 데이터에도 동일하게 적용**된다. HUST의 로깅 간격이 다를 경우 (`dt >= 1s` 조건에서 유효 행이 없을 수 있음) → NaN 반환으로 안전하게 처리됨.
- `dt >= 1s` 기준은 MIT의 정상 플래토 로깅(~4s)을 기준으로 설정. 향후 다른 데이터셋 추가 시 재검토 필요.
