# HI 중복 분석: hi_correlation.py (64D) vs preprocess.ipynb (75D)

두 HI 집합을 통합할 때 의미가 겹치는 피처를 정리한 문서.

> **중요 전제**: 75D 피처는 슬라이딩 윈도우 **segment** 단위로 계산되고,  
> 64D 피처는 **사이클 전체** 또는 **q_frac 구간 전체** 단위로 계산된다.  
> 수식이 동일하더라도 입력 데이터 범위가 달라 수치는 다를 수 있음.

---

## 1. 직접 중복 — 수식이 사실상 동일

| 75D 피처 | 64D 피처 | 수식 비교 |
|---------|---------|---------|
| `c1_1_mean_v` | `v_mean` | 둘 다 `mean(V)` |
| `c1_2_var_v` | `v_std` | `var(V)` vs `std(V)` — 단조 변환(std² = var), 정보량 동일 |
| `c1_3_skew_v` | `v_skew` | 둘 다 `skewness(V)` |
| `c1_4_kurt_v` | `v_kurt` | 둘 다 `kurtosis(V)` |
| `c1_5_mean_t` | `temp_mean` | 둘 다 `mean(T)` |
| `c2_dis_6_discharge_energy` | `energy_Wh` | 둘 다 `∫V·\|I\|·dt / 3600` |
| `c3_cha_14_ic_peak` | `chg_ica_peak_h` | 둘 다 충전 구간 `max(dQ/dV)` |
| `n1_ic_peak_pos` | `ica_peak_v` | 둘 다 `argmax(dQ/dV)`의 전압 위치 |

**처리 권장**: 통합 시 64D 피처 이름을 기준으로 삼고 75D 중복 피처는 제거.  
단, `c1_2_var_v`는 `v_std`의 제곱이므로 어느 쪽을 쓸지 통일 필요.

---

## 2. 개념적 중복 — 같은 물리 현상, 다른 수식

수치는 다르지만 열화와의 상관 방향·의미가 겹쳐 모델에 중복 정보를 줄 가능성이 있는 쌍.

### 2-A. 에너지 가중 전압 계산

| 75D | 64D | 차이 |
|-----|-----|------|
| `c1_14_de_dq` = `Σ(V·\|I\|·dt) / Σ(\|I\|·dt)` | `v_energy` = `energy_Wh / capacity_Ah` | 수식 동일 (에너지/전하량 = 가중평균 전압). 분모가 `Σ(I·dt)` vs `capacity_Ah`로 표기만 다름 |
| `n27_energy_eff_ratio` = `Σ(V·dQ) / (Q_total · 3.2)` | `v_energy` | 3.2V 기준 정규화만 추가. 단조 변환에 가까움 |

### 2-B. 방전 초기 전압 강하 (내부 저항 대리 지표)

| 75D | 64D | 차이 |
|-----|-----|------|
| `n20_imp_proxy` = `mean((V - 3.2) / I)` | `v_drop` = `V[0] − mean(V[:5%])` | 둘 다 I×R 분극을 측정하나 n20은 전 구간 평균, v_drop은 초반 5% 전압 강하 |
| `n24_trans_slope` = `(V[1]-V[0]) / (V[-1]-V[-2])` | `v_drop` | 초기/말기 기울기 비율 vs 초반 절대 강하량 — 물리적 의미 유사 |
| `n29_time_to_step` = V가 0.1V 변하는 데 걸리는 시간 | `v_drop` | 초기 분극 속도 반영, 같은 현상의 다른 표현 |
| `n30_boundary_shift` = `V[-1] - V[0]` (부호 반대의 v_drop과 같음) | `v_drop` | `n30 ≈ −v_drop`. 정보량 동일 |

### 2-C. 플래토 용량 관련

| 75D | 64D | 차이 |
|-----|-----|------|
| `n13_dv_area_thresh` = `Σ dQ` where `\|dV/dQ\| < 0.1` | `q_plateau_ratio`, `q_high_v` | 모두 플래토 구간 용량 측정. 임계값 기준만 다름 |
| `n15_plateau_len` = `Σ dQ` where `\|dV/dQ\| < 0.2` | `q_plateau_ratio`, `q_high_v` | n13과 동일 개념, 임계값만 더 느슨함 |

### 2-D. DVA(dV/dQ) 최솟값 계열

| 75D | 64D | 차이 |
|-----|-----|------|
| `n10_dv_min` = 5th percentile of `\|dV/dQ\|` | `dvdq_min` = `min(dV/dQ)` in plateau V range | 5번째 백분위 vs 절댓값 최솟값 — 플래토 평탄도를 측정하는 지표로 방향 동일 |
| `c2_dis_8_mean_dvdq` = `mean(dV/dQ)` | `dvdq_min` | 평균 vs 최솟값 — 상관은 높지만 정보가 다소 다름 |

### 2-E. ICA 피크 위치

| 75D | 64D | 차이 |
|-----|-----|------|
| `n9_dv_center_v` = `V` at `min(\|dV/dQ\|)` | `ica_peak_v` = `argmax(dQ/dV)`의 전압 | min dV/dQ ↔ max dQ/dV는 역수 관계이므로 이론상 동일 위치. 스무딩 유무로 미세한 차이 |

### 2-F. 에너지 면적(∫V·dt)

| 75D | 64D | 차이 |
|-----|-----|------|
| `n22_v_auc_time` = `trapz(V - V_min, t)` | `energy_Wh` = `∫V·\|I\|·dt / 3600` | 정전류 방전 시 `\|I\|`가 상수이므로 `n22 ∝ energy_Wh`. HUST 이전류 혼재 시 차이 발생 |

### 2-G. 방전 말기 전압

| 75D | 64D | 차이 |
|-----|-----|------|
| `n28_v_relax_rate` = `\|V[-1] - V[-5]\| / Δt` (말기 전압 변화 속도) | `v_end` = `V[-1]` | 둘 다 방전 종료 직전 전압 거동 측정. v_end는 절댓값, n28은 변화 속도 |

### 2-H. 방전 초반/후반 절대 전압

| 75D | 64D | 차이 |
|-----|-----|------|
| `n18_plateau_start_v` = 플래토 시작 전압 (첫 번째 `\|dV/dQ\|<0.2` 지점) | `v_at_q20` = Q=20% 시점 전압 | 둘 다 방전 초반 고전압 영역 전압을 포착. 기준점 정의가 다름 |
| `n19_plateau_end_v` = 플래토 종료 전압 (마지막 `\|dV/dQ\|<0.2` 지점) | `v_at_q80` = Q=80% 시점 전압 | 둘 다 방전 후반 플래토 이탈 전압 포착 |

### 2-I. 온도 변동 폭

| 75D | 64D | 차이 |
|-----|-----|------|
| `c1_6_delta_t` = `max(T) - min(T)` | `temp_max` | delta_t = temp_max - temp_min 이므로 temp_min이 일정하면 단조 변환 |

### 2-J. ICA 피크 면적/형태

| 75D | 64D | 차이 |
|-----|-----|------|
| `n3_ic_width_half` = std(V where `\|IC\| > 0.5·max`) | `ica_peak_area` = `∫max(dQ/dV, 0)dV` in 3.2~3.5V | 둘 다 ICA 피크의 "넓이" 측정. n3은 반치 너비, ica_peak_area는 적분 면적 |

---

## 3. 신규 정보 — 64D에 대응 없는 75D 피처

아래 피처들은 64D에 의미적으로 유사한 HI가 없으며, 통합 시 실질적으로 새로운 정보를 추가함.

| 피처 | 설명 | 물리적 의의 |
|------|------|------------|
| `c1_7_dtdt` | mean(dT/dt) | 온도 상승 속도 — I²R 발열 동적 변화 |
| `c1_8_mean_i` | mean(\|I\|) | 평균 전류 크기 (C-rate 정보) |
| `c1_9_var_i` | var(\|I\|) | 전류 변동성 (CC/CV 전환 패턴) |
| `c1_10_ent_v` | 전압 히스토그램 엔트로피 | 전압 분포의 무질서도 — 플래토 선명도 대리 지표 |
| `c1_11_ent_t` | 온도 히스토그램 엔트로피 | 온도 분포 균일성 |
| `c1_12_corr_vi` | pearsonr(V, \|I\|) | V-I 상관 — 내부 저항 특성 반영 |
| `c1_13_corr_vt` | pearsonr(V, T) | V-T 상관 — 발열-전압 결합 거동 |
| `c1_15_power_var` | var(V·I) | 순시 전력 변동 — 임피던스 비선형성 |
| `c2_dis_1_mean_dvdt` | mean(dV/dt) | 전압 감소 평균 속도 |
| `c2_dis_2_var_dvdt` | var(dV/dt) | 전압 감소 변동성 |
| `c2_dis_3_mean_d2vdt2` | mean(d²V/dt²) | 전압 곡률 — 플래토 이탈 민감도 |
| `c2_dis_5_dynamic_resistance` | mean(\|dV/dI\|) when ΔI > 0.005 | 동적 내부 저항 직접 추정 |
| `c2_dis_9_var_dvdq` | var(dV/dQ) | DVA 곡선 변동성 |
| `c2_dis_10_voltage_retention` | V_min / V_max | 전압 유지율 |
| `c2_dis_11_temp_rise_per_ah` | ΔT / Q | Ah당 온도 상승 — 효율 열화 지표 |
| `c2_dis_12_voltage_fluctuation` | Σ\|Δ(dV/dt)\| | 전압 동역학 거칠기 |
| `c2_dis_14_ocv_slope` | polyfit(Q_cum, V, 1)[0] | V-Q 선형 기울기 — OCV 경사 |
| `n2_ic_grad_max` | max\|d(IC)/dt\| | ICA 피크 선명도(급격한 변화 여부) |
| `n4_ic_skew` | skewness(IC curve) | ICA 피크 비대칭 — 열화 메커니즘 구분 |
| `n5_ic_kurt` | kurtosis(IC curve) | ICA 피크 첨도 |
| `n6_ic_valley` | 5th percentile(\|IC\|) | ICA 평탄부 최솟값 |
| `n7_ic_t_corr` | pearsonr(IC, dT) | ICA와 온도 변화 상관 |
| `n8_ic_peak_count` | IC 극값 개수 | 다중 반응 피크 유무 |
| `n11_dv_peak_gap` | Q range where \|dV/dQ\| > 90th pct | 상전이 구간 Q 폭 |
| `n12_dv_curvature` | mean\|d²V/dQ²\| | DVA 곡률 — 비선형 열화 |
| `n14_dv_symmetry` | std(dV/dQ 전반)/std(dV/dQ 후반) | 방전 전반/후반 DVA 비대칭 |
| `n16_v_sum_abs_diff` | Σ\|ΔV\| | 전압 총 변동량 |
| `n21_v_roughness` | std(dV/dt) | 전압 노이즈 수준 |
| `n23_nonlinear_idx` | cubic polyfit(Q_cum, V)[0] | V-Q 비선형성 — 고차 열화 거동 |
| `n25_thermal_stability` | std(dT/dt) | 온도 변화율 안정성 |
| `n26_dq_dt_plateau` | mean(dQ/\|dT\|) at plateau | 플래토 열 효율 |
| `c3_cha_1~c3_cha_13`, `c3_cha_15` | 충전 방향 dV/dt, dV/dQ 통계 다수 | 64D 충전 Global(6개)에 없는 충전 dynamics |

---

## 4. 요약 테이블

| 구분 | 피처 수 | 해당 피처 |
|------|---------|----------|
| **직접 중복** (수식 동일) | 8쌍 | c1_1, c1_2, c1_3, c1_4, c1_5, c2_dis_6, c3_cha_14, n1 |
| **개념적 중복** (정보 중복 가능성) | ~18쌍 | c1_14, n27 / n20, n24, n29, n30 / n13, n15 / n10, c2_dis_8 / n9 / n22 / n28 / n18, n19 / c1_6 / n3 |
| **신규 정보** | ~34개 | c1_7~c1_15 일부, c2_dis_1~5, c2_dis_9~15, n2~n8, n11~n12, n14, n16, n21, n23, n25~n26, c3_cha_1~15 대부분 |

---

## 5. 통합 시 권장 처리 방향

1. **직접 중복 8쌍**: 64D 이름 기준으로 통일하고 75D 쪽 제거. `c1_2_var_v`는 `v_std²`이므로 std/var 중 하나로 통일.

2. **개념적 중복 쌍**: 상관계수를 계산해 ρ > 0.95이면 제거 후보로 분류. 특히 `{n20, n24, n29, n30}` vs `v_drop`, `{n13, n15}` vs `q_plateau_ratio`는 높은 상관 예상.

3. **계산 컨텍스트 차이 주의**: 75D는 슬라이딩 윈도우 segment 기반, 64D는 전체 구간 기반이므로 수식이 같아도 값 범위가 다를 수 있음. 동일 모델에 혼용 시 스케일 정규화 필수.

4. **ICA/DVA 스무딩 차이**: 64D는 Savitzky-Golay 스무딩 후 피크 추출, 75D의 `n` 계열은 raw 미분. 혼용 시 noise sensitivity 차이 발생 → 75D의 ICA 관련 피처는 스무딩 추가 여부 재검토 필요.

5. **온도 피처 (HUST 제외)**: `c1_5~c1_7`, `c1_11`, `c1_13`, `n25`, `n26` 등 온도 관련 피처는 HUST에서 항온조 30°C 고정이므로 상수에 가깝고 정보량이 없음. HUST 학습 시 해당 피처 마스킹 필요.
