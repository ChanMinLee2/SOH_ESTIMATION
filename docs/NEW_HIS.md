# NEW HI 설계안

LFP 배터리 전문가 관점에서 재설계한 Health Indicator 구조.  
**총 285개** = Global 15 + Segment (3카테고리 × 15 × 6구간) 270.

> **LFP 열화 메커니즘 요약**
>
> | 메커니즘 | 약어 | 주요 영향 | 포착 가능한 HI 카테고리 |
> |---------|------|----------|----------------------|
> | 리튬 재고 감소 | LLI | 가용 용량 감소, 전극 전위 슬라이딩 | ICA/DVA 피크 이동, q_plateau_frac |
> | 양극 활물질 손실 | LAM_pe | 플래토 단축, ICA 피크 감소 | ica_peak_h, plateau_frac, ica_peak_area |
> | 음극 활물질 손실 | LAM_ne | 전위 미스매치, ICA 피크 이동 | dva_valley_q, dvdq_min |
> | 내부 저항 증가 | R↑ | 분극 심화, 전압 곡률 증가 | r_dc_est, v_sag_mid, v_concavity |
>
> LFP 특징: 플래토(~3.2–3.35 V)가 매우 평탄하여 전압 통계 단독으로는 열화 신호 약함.  
> **플래토 형태 변화**(ICA/DVA), **구간별 V-Q 비선형성**, **발열 거동**이 핵심 신호.

---

## 1. Global HI (15개) — 전체 사이클 기반

전체 방전/충전 사이클에서 추출하는 전역 지표. 세그먼트 분할 없이 계산.

| # | 키 | 상 | 수식 / 추출법 | 물리적 의미 | 열화 방향 | 포착 메커니즘 |
|---|----|----|--------------|------------|-----------|--------------|
| G01 | `q_dis` | dis | `∫\|I\|·dt / 3600` [Ah] | 총 방전 용량 | ↓ 열화 | LLI, LAM_pe, LAM_ne |
| G02 | `energy_dis` | dis | `∫V·\|I\|·dt / 3600` [Wh] | 총 방전 에너지 | ↓ | 위와 동일, 전압 저하도 이중 반영 |
| G03 | `v_mean_dis` | dis | `∫V·\|I\|·dt / ∫\|I\|·dt` [V] | 용량 가중 평균 전압 | ↓ 약 | R↑ 에 의한 분극 |
| G04 | `r_dc_est` | dis/chg | `ΔV / ΔI` at current step [mΩ] | 직류 내부 저항 | ↑ | SEI 성장, 전해질 열화 |
| G05 | `q_plateau_frac` | dis | `Q[3.10V≤V≤3.45V] / Q_total` | 플래토 구간 용량 비율 | ↓ | R↑(전압 강하 가속), LAM |
| G06 | `ica_peak1_v` | dis | `argmax(dQ/dV)` in 3.1–3.5 V [V] | 1차 ICA 피크 전압 (상전이 위치) | 이동 | LLI (전극 전위 슬라이딩) |
| G07 | `ica_peak1_h` | dis | `max(dQ/dV)` [Ah/V] | 1차 ICA 피크 높이 | ↓ | LAM_pe (상전이 가역성 저하) |
| G08 | `ica_peak1_area` | dis | `∫dQ/dV dV` in 3.1–3.5 V [Ah] | ICA 피크 적분 (플래토 용량) | ↓ | LAM, LLI |
| G09 | `dva_valley_q` | dis | `argmin(dV/dQ)` Q 위치 [Ah] | DVA 골짜기 Q 위치 (상전이 경계) | 이동 | LAM_ne/pe (전극 용량 비율 변화) |
| G10 | `dva_valley_depth` | dis | `min(dV/dQ)` [V/Ah] | DVA 골짜기 깊이 (0에 가까울수록 건강) | ↑음수 | R↑, LAM |
| G11 | `ce` | dis+chg | `Q_dis / Q_chg` | 쿨롱 효율 | ↓ | SEI 성장, Li 석출 (10–50 사이클 이동평균 권장) |
| G12 | `cv_q_frac` | chg | `Q_CV / Q_chg` | CV 구간 용량 비율 | ↑ | R↑ (동력학 저하 → CC→CV 전환 빠름) |
| G13 | `cv_time_frac` | chg | `t_CV / t_total_chg` | CV 구간 시간 비율 | ↑ | R↑, Li 확산 저하 |
| G14 | `chg_ica_peak1_h` | chg | `max(dQ/dV)` during charge [Ah/V] | 충전 방향 ICA 피크 높이 | ↓ | LAM_pe (충전 플래토 가역성) |
| G15 | `ica_peak1_asym` | dis | `left_hw / right_hw` of ICA peak [0–∞] | ICA 1차 피크 비대칭도 (좌/우 반폭 비율) | 이동 | LLI (전극 전위 슬라이딩) / LAM_ne vs LAM_pe 불균형 — 1에 가까울수록 대칭 |

> **계산 주의사항**
> - `r_dc_est`: CC→CV 전환 시 전류 계단 이용. CC 단독 방전 시 방전 시작 순간(OCV→방전 첫 샘플) 이용.
> - ICA/DVA: Savitzky-Golay (window=21, poly=3) 스무딩 후 추출.
> - `ce`: HUST는 충전 종료 전류 정의 불명확하므로 MIT 전용 or careful 처리 필요.
> - `dva_valley_depth`: 방전 방향이므로 음수값. 더 음수일수록 플래토 기울기 증가 = 열화.
> - `ica_peak1_asym`: Savitzky-Golay 스무딩 후 scipy.signal.peak_widths로 반폭 추출. `left_hw / right_hw` (피크 높이 50% 기준). 피크 없으면 NaN. 1 = 완전 대칭; >1 = 우측 편중; <1 = 좌측 편중.

---

## 2. Segment HI (270개) — 6구간 × 45개

### 세그먼트 정의

q_frac = cumsum(|I|·dt) / Q_total 기준으로 구간 분할.

| 세그먼트 키 | q_frac 범위 | 근사 SoC | 물리적 위치 |
|------------|------------|---------|------------|
| `dis_hi` | 0.00–0.40 | SoC 60–100% | 방전 초반: 고전압 → 플래토 진입 |
| `dis_mid` | 0.40–0.70 | SoC 30–60% | 방전 중반: LFP 플래토 중심부 |
| `dis_lo` | 0.70–1.00 | SoC 0–30% | 방전 후반: 플래토 이탈 → 저전압 급강하 |
| `chg_lo` | 0.00–0.40 | SoC 0–40% | 충전 초반: 저전압 → 플래토 진입 |
| `chg_mid` | 0.40–0.70 | SoC 40–70% | 충전 중반: LFP 플래토 중심부 |
| `chg_hi` | 0.70–1.00 | SoC 70–100% | 충전 후반: 플래토 이탈 → CV 포화 |

각 세그먼트에 3개 카테고리 × 15개 = 45개 HI 적용.  
키 명명 예시: `stat_v_mean_dis_hi`, `diff_dvdq_min_chg_mid`, `lfp_plateau_frac_dis_mid`

---

### 카테고리 A: 통계 기반 (Statistical, 15개/구간)

원시 신호(V, I, T, Q)의 분포 통계량. 계산이 단순하고 노이즈 강인함.

| # | 키 | 입력 신호 | 수식 | 물리적 의미 | 예상 열화 방향 | 비고 |
|---|----|---------|----|------------|--------------|------|
| S01 | `v_mean` | V | `∫V·\|I\|·dt / ∫\|I\|·dt` [V] | 구간 내 용량 가중 평균 전압 | ↓ (분극 증가) | 플래토 구간에서 신호 약 |
| S02 | `v_std` | V | `std(V)` [V] | 구간 내 전압 변동성 | ↑ (플래토 붕괴) 또는 ↓ | 방향 데이터 확인 필요 |
| S03 | `v_skew` | V | `skewness(V)` | 전압 분포 비대칭성 | 이동 | 플래토 기울기 방향 반영 |
| S04 | `v_kurt` | V | `kurtosis(V)` | 전압 분포 첨도 | ↓ (분포 넓어짐) | 플래토 구간에서 초기값 높음 |
| S05 | `v_ent` | V | `−Σp·log(p)`, PMF 20-bin [nats] | 전압 분포 엔트로피 | ↑ (불규칙성 증가) | density=False, PMF 직접 계산 |
| S06 | `i_mean` | I | `mean(\|I\|)` [A] | 구간 내 평균 전류 절댓값 | ↓ chg_hi (CV 전환 가속) | CC 구간은 상수, CV 전환 구간에서 감소; HUST 다단계 C-rate 포착 |
| S07 | `i_std` | I | `std(\|I\|)` [A] | 구간 내 전류 표준편차 | ↑ chg_hi (조기 CV 전환) | CC: ≈0, CC→CV 전환 비중 커질수록 증가 |
| S08 | `v_med` | V | `median(V)` [V] | 구간 내 전압 중앙값 | ↓ (분극 증가) | v_mean보다 이상치 강인; 분포 비대칭 변화도 반영 |
| S09 | `corr_qi` | Q, I | `Pearson(Q_cum, \|I\|)` | 누적 전하-전류 상관 | ↑ 음수 chg_hi | CC: Q↑ I 일정 → ≈0; CV: Q↑ I↓ → 음수; 열화로 CV 전환 빠를수록 음수 강화 |
| S10 | `corr_vi` | V, I | `Pearson(V, \|I\|)` | V-전류 상관 | 이동 | CC 구간에서 상수, CV에서 변화 |
| S11 | `q_abs` | I | `∫\|I\|·dt / 3600` [Ah] | 구간 내 전하량 | ↓ | SOH와 준선형 — leakage 주의 |
| S12 | `energy_seg` | V, I | `∫V·\|I\|·dt / 3600` [Wh] | 구간 내 에너지 | ↓ | q_abs × v_mean — 이중 반영 |
| S13 | `v_iqr` | V | `V_75th − V_25th` [V] | 전압 로버스트 범위 (이상치 영향 적음) | 이동 | v_range보다 이상치 강인 |
| S14 | `v_range` | V | `V_max − V_min` [V] | 구간 내 전압 스윙 | ↑ (분극 심화) | 플래토 구간은 작음; 전환 구간(dis_hi/lo, chg_lo/hi)에서 크고 열화 민감 |
| S15 | `v_p10` | V | `10th-percentile(V)` [V] | 구간 하위 10% 전압 (저전압 꼬리) | ↓ | 분극 극단값 추적 |

> **주의**: `q_abs`, `energy_seg`는 capacity_Ah와 구조적 leakage 가능성 있음 (구간 경계가 q_frac 기반이므로 q_abs ∝ Q_total). 상관 분석 시 높은 ρ가 예상되나 ML feature로서는 유용.

---

### 카테고리 B: 미분 기반 (Differential, 15개/구간)

V-Q, V-t 미분 및 ICA/DVA 변환 기반 피처. LFP 상전이 직접 포착.

| # | 키 | 수식 / 추출법 | 물리적 의미 | 예상 열화 방향 | 적용 구간 주의 |
|---|---|--------------|------------|--------------|--------------|
| D01 | `dvdq_mean` | `mean(dV/dQ)` in seg [V/Ah] | V-Q 곡선 평균 기울기 | ↑ 음수 커짐 (dis) | 플래토(`mid`)에서 ≈ 0 |
| D02 | `dvdq_std` | `std(dV/dQ)` [V/Ah] | V-Q 기울기 변동성 | ↑ (곡선 불규칙) | 플래토에서 작음, 전환 구간에서 큼 |
| D03 | `dvdq_max_abs` | `max(\|dV/dQ\|)` [V/Ah] | 가장 급격한 V-Q 변화 | ↑ | dis_hi, dis_lo에서 큼 |
| D04 | `dvdq_min` | `min(dV/dQ)` [V/Ah] | V-Q 곡선 최저 기울기 (dis 플래토 깊이) | ↑ 0에서 멀어짐 | `dis_mid`에서 핵심 지표 |
| D05 | `dvdq_area` | `∫\|dV/dQ\|·dQ` in seg | V-Q 비선형성 적분 | ↑ | 모든 구간 유효 |
| D06 | `dqdv_peak_h` | `max(dQ/dV)` in seg [Ah/V] | 구간 내 ICA 피크 높이 | ↓ | `dis_mid`, `chg_mid`에서 의미 큼 |
| D07 | `dqdv_peak_v` | `V at max(dQ/dV)` [V] | 구간 내 ICA 피크 위치 | 이동 | LLI 진행 시 이동; 구간에 피크 없으면 NaN |
| D08 | `dqdv_peak_w` | FWHM of dQ/dV peak [V] | ICA 피크 반폭 (넓을수록 상전이 저하) | ↑ | `dis_mid`, `chg_mid` 전용 |
| D09 | `dqdv_area` | `∫dQ/dV·dV` in seg [Ah] | 구간 내 ICA 적분 (구간 방출 용량) | ↓ | 모든 구간 유효 |
| D10 | `dvdt_slope` | `(V_end − V_start) / Δt_seg` [V/s] | 구간 전체 V-t 기울기 | 이동 | FP 아티팩트 방지: 총 기울기 사용 |
| D11 | `dqdv_peak_asym` | `left_hw / right_hw` of dQ/dV peak in seg | 구간 ICA 피크 비대칭도 | 이동 | `dis_mid`, `chg_mid` 유의미; 피크 없으면 NaN; LLI 진행 시 비대칭 증가 |
| D12 | `d2vdq2_rms` | `rms(d²V/dQ²)` in seg [V/Ah²] | V-Q 2차 미분 RMS (곡률) | ↑ | 플래토→전환 구간 경계에서 큼 |
| D13 | `dvdq_skew` | `skewness(dV/dQ)` in seg | V-Q 기울기 분포 비대칭 | 이동 | 플래토 기울기 방향 변화 반영 |
| D14 | `dvdq_ent` | `−Σp·log(p)` of \|dV/dQ\| PMF [nats] | V-Q 기울기 분포 엔트로피 | ↑ | 기울기가 균일할수록 낮음 = 건강 |
| D15 | `r_dyn_seg` | `mean(\|ΔV/ΔI\|)` where ΔI≠0, Δt<2s [Ω] | 동적 내부 저항 (순간 전류 변화시) | ↑ | CC 구간에서는 ΔI≈0이라 희소; CV 전환부 유효 |

> **계산 주의사항**
> - dQ/dV, dV/dQ: Savitzky-Golay 스무딩(window=15, poly=3) 후 추출.
> - `dqdv_peak_w`: scipy.signal.peak_widths 또는 반폭 직접 계산. 피크 없으면 NaN.
> - `dvdt_slope`: 구간 총 기울기 사용 (개별 dt-step 평균 X → FP 아티팩트 방지).
> - `dqdv_peak_asym`: Global G15 `ica_peak1_asym`과 동일 방식이나 구간 내 ICA에 적용. `dis_mid`, `chg_mid` 외 구간은 피크 없어 대부분 NaN.
> - `d2vdq2_rms`: dV/dQ 스무딩 후 2차 미분. 경계부 edge effect 주의.
> - `r_dyn_seg`: CC→CV 전환이 구간 내에 없으면 NaN. HUST 다단계 방전은 단계 전환 시 계산.

---

### 카테고리 C: LFP 특징 기반 (LFP-specific, 15개/구간)

LFP 이상전이(two-phase reaction) 및 플래토 거동에 특화된 지표.

| # | 키 | 수식 / 추출법 | 물리적 의미 | 예상 열화 방향 | 비고 |
|---|---|--------------|------------|--------------|------|
| L01 | `plateau_frac` | `len(samples where \|dV/dQ\|<θ_flat) / N_seg` | 구간 내 플래토 비율 (θ_flat=0.05 V/Ah) | ↓ | `dis_mid`, `chg_mid`에서 크고 다른 구간은 작음; 열화 시 감소 |
| L02 | `plateau_v_mean` | `mean(V)` where `\|dV/dQ\|<θ_flat` [V] | 플래토 평균 전압 | ↓ (분극) | 플래토 서브구간 없으면 NaN |
| L03 | `plateau_v_std` | `std(V)` where `\|dV/dQ\|<θ_flat` [V] | 플래토 전압 균일성 | ↑ (플래토 기울어짐) | LFP 건강 지표: 작을수록 좋음 |
| L04 | `plateau_q_frac` | `Q_plateau / Q_seg` where plateau 정의 동일 | 구간 내 플래토 용량 비율 | ↓ | L01과 상관 높지만 시간 vs 용량 기준 차이 |
| L05 | `nonlin_idx` | `RMSE(V_actual, V_linear) / V_range_seg` | V-Q 곡선 선형 기준 편차 비율 | ↑ | V_linear: (Q_start, V_start)→(Q_end, V_end) 직선 |
| L06 | `v_sag_mid` | `V_actual(q_mid) − V_linear(q_mid)` [V] | 구간 중점에서 V-Q 처짐 | ↑ 음수 (dis) | 분극 의존; 방전은 음수(처짐), 충전은 양수(부풀음) |
| L07 | `v_flatness` | `1 − std(V) / (V_max − V_min)` in seg | 전압 평탄도 지수 [0–1] | ↓ (플래토 붕괴) | 1에 가까울수록 완벽한 플래토 |
| L08 | `delta_v_rms` | `rms(ΔV)` where dt≥1s [V] | 샘플 간 전압 변화 RMS (roughness) | ↑ | FP 아티팩트 방지: dt<1s 제외 |
| L09 | `ocv_slope` | `smoothed dV/dQ at q_mid of seg` [V/Ah] | 구간 중점 OCV 기울기 근사 | 이동 | dV/dQ 스무딩 후 구간 중점값 |
| L10 | `knee_v` | `V at inflection(V-Q): argzero(d²V/dQ²)` [V] | V-Q 변곡점 전압 (플래토 진입/이탈 경계) | 이동 | d²V/dQ²의 부호 전환 위치; 없으면 NaN |
| L11 | `knee_q_frac` | `q_frac at inflection` in seg | 구간 내 변곡점 Q 위치 | ↓ (earlier transition) | 열화 시 플래토 이탈이 더 일찍 일어남 |
| L12 | `v_concavity` | `V_mean − (V_start + V_end) / 2` [V] | V-Q 곡선 오목볼록 지수 | ↑ 절댓값 (곡률 증가) | 방전: 음수(오목), 충전: 양수(볼록). 열화 시 V-Q 비선형성 증가 → 절댓값 커짐; R↑, LAM 반영 |
| L13 | `phase_entry_dvdq` | `\|dV/dQ\|` at first 5% of seg | 구간 진입부 V-Q 기울기 절댓값 | ↑ (경계 더 날카로워짐) | 상전이 경계 선명도; 열화 시 전이 구간 이동 |
| L14 | `v_q_pearson` | `Pearson(V, Q_cum)` in seg | V-Q 선형성 (플래토: 낮음, 전환: 높음) | 이동 | 플래토에서 \|r\|≈0, 전환 구간에서 \|r\|≈1; 열화 시 dis_mid에서 \|r\| 증가 |
| L15 | `ica_peak_cnt` | `count(local maxima of dQ/dV)` in seg | 구간 내 ICA 피크 개수 | 이동 | 정상: dis_mid=1, 나머지=0; 비정상 피크 발생 시 LAM 분리 시사 |

> **θ_flat 설정 근거**: LFP 플래토에서 실측 dV/dQ ≈ 0.01–0.03 V/Ah. θ_flat=0.05 V/Ah로 여유 있게 설정하여 측정 노이즈 포함.
>
> **계산 주의사항**
> - `plateau_frac`, `plateau_v_mean`, `plateau_v_std`, `plateau_q_frac`: dV/dQ 스무딩 후 적용. 플래토 서브구간이 전체 구간의 5% 미만이면 신뢰도 낮음 → NaN 처리 권장.
> - `v_sag_mid`: V_linear는 구간 양 끝점을 잇는 직선. 음수(방전 처짐) / 양수(충전 부풀음)로 부호 있음.
> - `knee_v`, `knee_q_frac`: d²V/dQ² smoothed(window=11) 후 부호 전환 탐색. 전환점이 없거나 복수이면 primary (최대 절댓값) 선택.
> - `v_concavity`: V_start = 구간 첫 샘플 전압, V_end = 마지막 샘플 전압, V_mean = capacity-weighted mean. 구간 길이 < 10 샘플이면 NaN.
> - `v_q_pearson`: Pearson r의 절댓값이 아닌 부호 포함값 사용 (방향 정보 중요).

---

## 3. 전체 구조 요약

```
Global HI (15)
├── Discharge Global (12): G01–G11, G15
│   └── 용량·에너지·전압·ICA·DVA·쿨롱효율·온도
└── Charge Global (3): G12–G14
    └── CV 거동·충전 ICA

Segment HI (270 = 6 × 45)
├── dis_hi (SoC 60–100%, q_frac 0–0.4)
│   ├── Statistical:   S01–S15 (15개)
│   ├── Differential:  D01–D15 (15개)
│   └── LFP-specific:  L01–L15 (15개)
├── dis_mid (SoC 30–60%, q_frac 0.4–0.7)  ← 플래토 중심부, 신호 가장 강
│   ├── ...
├── dis_lo (SoC 0–30%, q_frac 0.7–1.0)
│   ├── ...
├── chg_lo (SoC 0–40%, q_frac 0–0.4)
│   ├── ...
├── chg_mid (SoC 40–70%, q_frac 0.4–0.7)  ← 충전 플래토 중심부
│   ├── ...
└── chg_hi (SoC 70–100%, q_frac 0.7–1.0)  ⚠️ CV 포화 구간 — ICA/DVA 주의
    ├── ...

Total: 15 + 270 = 285
```

---

## 4. 구간별 기대 신호 강도

신호 강도: ● 강 / ◐ 중 / ○ 약 / ⚠ 불안정

| HI 카테고리 | dis_hi | dis_mid | dis_lo | chg_lo | chg_mid | chg_hi |
|------------|--------|---------|--------|--------|---------|--------|
| 통계 — v_mean/std | ○ | ○ | ◐ | ○ | ○ | ◐ |
| 통계 — v_ent/skew/kurt | ○ | ○ | ◐ | ○ | ○ | ◐ |
| 통계 — q_abs/energy | ● | ● | ● | ● | ● | ● |
| 통계 — i_mean/std | ◐ | ○ | ◐ | ◐ | ○ | ● |
| 미분 — dvdq_mean/std/min | ◐ | ● | ◐ | ◐ | ● | ⚠ |
| 미분 — dqdv_peak_h/v/w | ○ | ● | ○ | ○ | ● | ⚠ |
| 미분 — dvdt_slope/peak_asym | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ |
| 미분 — d2vdq2_rms | ◐ | ○ | ◐ | ◐ | ○ | ⚠ |
| LFP — plateau_frac/v_std | ◐ | ● | ○ | ○ | ● | ⚠ |
| LFP — nonlin_idx/v_sag | ● | ◐ | ● | ● | ◐ | ◐ |
| LFP — v_concavity | ● | ◐ | ● | ● | ◐ | ◐ |
| LFP — knee_v/q_frac | ● | ○ | ● | ● | ○ | ◐ |
| LFP — ica_peak_cnt | ○ | ◐ | ○ | ○ | ◐ | ⚠ |

> ⚠ `chg_hi`: CV 포화 구간에서 dV→0, I→0이므로 dQ/dV→∞, dV/dQ→0 발산 가능.  
> 해당 구간 미분 기반 HI는 CV 구간 데이터 제외 후 CC 부분만 사용하거나 NaN 처리.

---

## 5. 기존 HI(148D)와의 비교

| 항목 | 기존 (148D) | 신규 (285D) |
|-----|-----------|-----------|
| Global | 28개 (dis 22 + chg 6) | **15개** (축약·재정의) |
| Segment | 120개 (6seg × 20) | **270개** (6seg × 45) |
| 세그먼트 카테고리 | 혼합 | **통계/미분/LFP 3분류** |
| ICA/DVA | Global만 | **Global + 구간별** |
| 플래토 특화 | 일부 (q_plateau_ratio 등) | **LFP 카테고리 15종 전용** |
| 변곡점/경계 선명도 | 없음 | **L10(knee_v), L13(phase_entry)** 신규 |
| 전류 거동 | 없음 | **i_mean, i_std, corr_qi** (신규) |
| V-Q 곡률 | 일부 (nonlin_idx) | **v_concavity, v_range** (신규) |
| 이상치 취약 HI | ent_v (density 버그), dvdt (FP 아티팩트) | 설계 단계에서 제거: PMF 강제, 총 기울기 사용 |

---

## 6. 구현 우선순위 제안

```
Phase 1 (핵심, 즉시 구현)
  Global: G01–G10 (ICA/DVA 포함 기본 10종)
  Segment Statistical: S01–S12 (기본 통계 + q_abs/energy)
  Segment LFP: L05(nonlin_idx), L07(v_flatness), L08(delta_v_rms)

Phase 2 (신호 검증 후)
  Global: G11–G15 (ce, CV 거동, ica_peak1_asym)
  Segment Differential: D01–D09 (ICA/DVA 구간별)
  Segment LFP: L01–L04(plateau), L06(v_sag_mid)

Phase 3 (선택적)
  Segment Differential: D10–D15 (dvdt, dqdv_asym, d2v, r_dyn)
  Segment LFP: L09–L15 (knee, v_concavity, pearson)
```
