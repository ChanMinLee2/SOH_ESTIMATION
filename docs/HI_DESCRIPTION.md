# Health Indicator (HI) 설명 — 411D 설계 기준

LFP 배터리 SOH 예측 파이프라인의 HI 구조, 수식, 물리적 의미, 논문 계통 분석을 통합한 문서.  
**총 411개** = Global 15 + Segment (4카테고리 × 6구간) 396.

> **LFP 열화 메커니즘 요약**
>
> | 메커니즘 | 약어 | 주요 영향 | 포착 가능한 HI 카테고리 |
> |---------|------|----------|----------------------|
> | 리튬 재고 감소 | LLI | 가용 용량 감소, 전극 전위 슬라이딩 | ICA/DVA 피크 이동, q_plateau_frac |
> | 양극 활물질 손실 | LAM_pe | 플래토 단축, ICA 피크 감소 | ica_peak_h, plateau_frac, ica_peak_area |
> | 음극 활물질 손실 | LAM_ne | 전위 미스매치, ICA 피크 이동 | dva_valley_q, dvdq_min |
> | 내부 저항 증가 | R↑ | 분극 심화, 전압 곡률 증가 | r_dc_est, v_sag_mid, v_concavity |
> | 곡선 형상 변화 (종합) | 형태 열화 | 위 메커니즘들의 복합 결과 — V-t/V-Q/V-E 곡선 전체 변형 | **카테고리 D: 형태학적 거리** |
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

## 2. Segment HI (396개) — 6구간 × 66개

### 세그먼트 정의

q_frac = cumsum(|I|·dt) / Q_total 기준으로 구간 분할.

| 세그먼트 키 | q_frac 범위 | 근사 SoC | 물리적 위치 |
|------------|------------|---------|------------|
| `chg_lo` | 0.00–0.40 | SoC 0–40% | 충전 초반: 저전압 → 플래토 진입 |
| `chg_mid` | 0.40–0.70 | SoC 40–70% | 충전 중반: LFP 플래토 중심부 |
| `chg_hi` | 0.70–1.00 | SoC 70–100% | 충전 후반: 플래토 이탈 → CV 포화 |
| `dis_hi` | 0.00–0.40 | SoC 60–100% | 방전 초반: 고전압 → 플래토 진입 |
| `dis_mid` | 0.40–0.70 | SoC 30–60% | 방전 중반: LFP 플래토 중심부 |
| `dis_lo` | 0.70–1.00 | SoC 0–30% | 방전 후반: 플래토 이탈 → 저전압 급강하 |

각 세그먼트에 4개 카테고리 66개 HI 적용 (A·B·C 각 20개 + D 6개).  
키 명명 예시: `stat_v_mean_dis_hi`, `diff_dvdq_min_chg_mid`, `lfp_plateau_frac_dis_mid`, `morph_vq_dtw_chg_mid`

---

### 카테고리 A: 통계 기반 (Statistical, 20개/구간)

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
| S16 | `v_p90` | V | `90th-percentile(V)` [V] | 구간 상위 10% 전압 (고전압 꼬리) | ↓ dis / ↑ chg | v_p10과 쌍으로 10th–90th 범위 포착; 플래토 상단 경계 이동 반영 |
| S17 | `v_samp_ent` | V | `SampEn(V, m=2, r=0.2·std(V))` [nats] | 전압 시계열 패턴 복잡도 (sample entropy) | ↑ (플래토 불규칙성 증가) | 히스토그램 엔트로피(S05)보다 시간적 패턴 구조에 민감; Hu 2015 (IEEE TIE) 참조 |
| S18 | `corr_vt` | V, t | `Pearson(V, t_norm)` in seg | V-t 선형 연관도 (플래토 기울기 전반) | ↑ \|corr\| (플래토 기울어짐) | dvdt_slope(D10)은 끝점 2개만 사용; corr_vt는 전체 시계열 기반 — 플래토 중간 굴곡도 반영 |
| S19 | `i_q_slope` | I, Q | `OLS slope of \|I\| vs Q_cum` [A/Ah] | Q 증가에 따른 전류 감쇠율 (CC→CV 전환 속도) | ↑ \|slope\| chg_hi (열화 → 조기 CV 전환) | CC 구간: ≈0; CV 구간: 음수 기울기; 열화 시 chg_hi에서 더 급격한 전류 감쇠 |
| S20 | `v_detrended_std` | V, t | `std(V − OLS_fit(t_norm))` [V] | 선형 트렌드 제거 후 비선형 전압 변동 | ↑ (불규칙 플래토 패턴) | v_std(S02)는 트렌드 포함; v_detrended_std는 순수 비선형 성분만 — 플래토 파형 요동 포착 |

> **주의**:
> - `q_abs`, `energy_seg`는 capacity_Ah와 구조적 leakage 가능성 있음 (구간 경계가 q_frac 기반이므로 q_abs ∝ Q_total). 상관 분석 시 높은 ρ가 예상되나 ML feature로서는 유용.
> - `v_samp_ent`: scipy 없이 직접 구현 권장 (m=2 탬플릿 매칭, r=0.2×std(V); 샘플 수 < 10이면 NaN).
> - `i_q_slope`: CC 구간에서 ΔI≈0이므로 slope≈0, SE 크다 → 수치적으로 안정하나 정보 약함. chg_hi(CV) 구간에서 가장 의미 있음.
> - `v_detrended_std`: OLS는 t_norm=[0,1] 기준. 구간 내 샘플 < 5이면 NaN.

---

### 카테고리 B: 미분 기반 (Differential, 20개/구간)

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
| D16 | `dqdv_valley_h` | `min(dQ/dV)` in seg [Ah/V] | 구간 내 IC 밸리 깊이 (IC 곡선 최솟값) | 이동 | Hu2021 ICV/ICHV 대응; LFP 충전 중 피크 전후 골짜기; 피크 이전/이후 최솟값 중 피크에서 더 멀리 있는 쪽; 명확한 밸리 없으면 NaN |
| D17 | `dqdv_valley_v` | `V at min(dQ/dV)` in seg [V] | IC 밸리 전압 위치 | 이동 (LLI 진행 시 전위 슬라이딩) | Hu2021 ICVL 대응; D16과 동시 추출 |
| D18 | `dvdq_peak_q` | `Q at max\|dV/dQ\|` in seg [Ah] | V-Q 곡선이 가장 가파른 지점의 누적 전하량 (상전이 경계 Q 위치) | 이동 (LAM 시 플래토 범위 변화) | Hu2021 DVPL 대응; Global dva_valley_q는 전체 방전 기준, 이것은 세그먼트 내 위치 |
| D19 | `dvdq_valley_q` | `Q at min\|dV/dQ\|` in seg [Ah] | V-Q 기울기가 가장 평탄한 지점의 Q (플래토 중심 Q 위치) | 이동 (플래토 이동 반영) | Hu2021 DVVL 대응; dis_mid/chg_mid에서 핵심 — 플래토 무게중심 추적; 방전에서 argmin(dV/dQ)는 D18과 동일하므로 argmin(|dV/dQ|)로 수정 |
| D20 | `dqdv_area_asym` | `∫dQ/dV·dV [V≤V_peak] / ∫dQ/dV·dV [V>V_peak]` | IC 곡선 피크 좌우 면적 비율 (리튬 삽탈 비대칭도) | 1에서 멀어짐 (전극 불균형 심화) | **신선한 시도**: 전극 양측 리튬 이용 비대칭성을 직접 정량화; LAM_pe/ne 불균형 진행 시 비대칭 증가; 피크 없으면 NaN |

> **계산 주의사항**
> - dQ/dV, dV/dQ: Savitzky-Golay 스무딩(window=15, poly=3) 후 추출.
> - `dqdv_peak_w`: scipy.signal.peak_widths 또는 반폭 직접 계산. 피크 없으면 NaN.
> - `dvdt_slope`: 구간 총 기울기 사용 (개별 dt-step 평균 X → FP 아티팩트 방지).
> - `dqdv_peak_asym`: Global G15 `ica_peak1_asym`과 동일 방식이나 구간 내 ICA에 적용. `dis_mid`, `chg_mid` 외 구간은 피크 없어 대부분 NaN.
> - `d2vdq2_rms`: dV/dQ 스무딩 후 2차 미분. 경계부 edge effect 주의.
> - `r_dyn_seg`: CC→CV 전환이 구간 내에 없으면 NaN. HUST 다단계 방전은 단계 전환 시 계산.
> - `dqdv_valley_h`, `dqdv_valley_v`: 스무딩된 dQ/dV 곡선에서 피크 좌측 및 우측 각각의 최솟값 탐색 → 피크에서 더 먼 쪽의 값 사용. 피크 대비 20% 이하 상대 높이인 경우에만 유효 밸리로 인정.
> - `dvdq_peak_q`, `dvdq_valley_q`: 스무딩 후 각각 argmax(|dV/dQ|), argmin(|dV/dQ|) → qcs 배열에서 해당 인덱스 Q값 읽기. 방전(dV/dQ<0)에서 argmin(dV/dQ)=argmax(|dV/dQ|)이므로 D19는 반드시 argmin(|dV/dQ|)를 사용해야 D18과 구분됨.
> - `dqdv_area_asym`: 피크 전압 V_peak 기준으로 좌우 분리 적분. 두 면적 모두 0에 가까우면 NaN (피크 미발견 처리와 동일).

---

### 카테고리 C: LFP 특징 기반 (LFP-specific, 20개/구간)

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
| L16 | `plateau_v_slope` | `OLS slope of V vs Q_cum` within plateau mask [V/Ah] | 플래토 내부 전압 기울기 (분극 유발 플래토 경사) | ↑ \|slope\| (방전: 더 음수, 충전: 더 양수) | ocv_slope(L09)은 구간 중점 1개 값; L16은 플래토 전 구간 OLS → 평균 분극 경사 더 안정적 |
| L17 | `v_gradient_exit` | `\|dV/dQ\|` at final 5% of seg [V/Ah] | 구간 이탈부 V-Q 기울기 절댓값 (상전이 출구 선명도) | ↑ (LAM 진행 시 경계 더 날카로워짐) | L13 phase_entry_dvdq의 대칭 쌍 (진입/이탈 경계 선명도 비교 가능); 두 값 차이 = 경계 비대칭성 |
| L18 | `plateau_q_onset` | `q_frac` within seg of first plateau sample | 구간 내 플래토 시작 위치 (정규화 q_frac [0,1]) | ↓ (플래토 진입이 늦어짐 = 압축) | 플래토 없으면 NaN; L11(knee_q_frac)은 변곡점 기반, L18은 θ_flat 기반 — 접근법 상보적 |
| L19 | `dv_dt_plateau` | `mean(\|dV/dt\|)` within plateau mask [mV/s] | 플래토 내 전압 드리프트 속도 | ↑ (R↑ → CC 전류에 의한 전압 크리프 증가) | plateau_v_std(L03)은 공간적 분산; dv_dt_plateau는 시간적 드리프트 속도 — 두 정보 비직교 |
| L20 | `v_ent_plateau` | `−Σp·log(p)` of V within plateau mask, 10-bin PMF [nats] | 플래토 서브구간 전압 분포 엔트로피 | ↑ (플래토 붕괴 → 전압 분포 분산) | **신선한 시도**: 플래토 마스크 내부만의 엔트로피 → v_ent(S05, 전구간) 대비 플래토 품질 집중 측정; 건강 플래토: 전압 집중 → 낮은 엔트로피 |

> **θ_flat 설정 근거**: LFP 플래토에서 실측 dV/dQ ≈ 0.01–0.03 V/Ah. θ_flat=0.05 V/Ah로 여유 있게 설정하여 측정 노이즈 포함.
>
> **계산 주의사항**
> - `plateau_frac`, `plateau_v_mean`, `plateau_v_std`, `plateau_q_frac`: dV/dQ 스무딩 후 적용. 플래토 서브구간이 전체 구간의 5% 미만이면 신뢰도 낮음 → NaN 처리 권장.
> - `v_sag_mid`: V_linear는 구간 양 끝점을 잇는 직선. 음수(방전 처짐) / 양수(충전 부풀음)로 부호 있음.
> - `knee_v`, `knee_q_frac`: d²V/dQ² smoothed(window=11) 후 부호 전환 탐색. 전환점이 없거나 복수이면 primary (최대 절댓값) 선택.
> - `v_concavity`: V_start = 구간 첫 샘플 전압, V_end = 마지막 샘플 전압, V_mean = capacity-weighted mean. 구간 길이 < 10 샘플이면 NaN.
> - `v_q_pearson`: Pearson r의 절댓값이 아닌 부호 포함값 사용 (방향 정보 중요).
> - `plateau_v_slope`: plateau_v_mean(L02)과 동일한 플래토 마스크 사용; 플래토 샘플 < 5이면 NaN.
> - `v_gradient_exit`: 구간 마지막 5% 샘플(최소 3개)의 dV/dQ 절댓값 평균; L13과 동일 스무딩 적용.
> - `plateau_q_onset`: 플래토 마스크에서 첫 번째 True 인덱스 / (N_seg−1). 플래토가 전혀 없으면 NaN.
> - `dv_dt_plateau`: Δt ≥ 1s 조건 적용 (FP 아티팩트 방지); 유효 샘플 < 3이면 NaN.
> - `v_ent_plateau`: 플래토 서브구간 샘플 < 10이면 NaN (10-bin histogram 의미 없음).

---

### 카테고리 D: 형태학적 거리 (Morphological Distance, 6개/구간)

**개념**: Point-based HI(A·B·C)가 특정 지점의 통계/미분값을 추출하는 것과 달리, 카테고리 D는  
전체 곡선의 **형상 자체**가 최초 건강 곡선(BOL: Beginning of Life, 사이클 1)으로부터  
얼마나 달라졌는지를 거리 지표로 정량화한다.

**3종 곡선 × 2종 거리 = 6개 HI**

| # | 키 | 곡선 유형 | 거리 지표 | 수식 / 방법 | 물리적 의미 | 예상 열화 방향 |
|---|---|---------|---------|-----------|------------|-------------|
| M01 | `vt_dtw` | V-t | DTW | Sakoe-Chiba banded DTW (band=5/50점=10%) | V-t 곡선의 위상 이동 허용 누적 형상 거리 | ↑ (열화 → BOL 곡선과 멀어짐) |
| M02 | `vq_dtw` | V-Q | DTW | 동일, q_frac 축 [0,1] 50점 보간 | V-Q 곡선 누적 형상 거리 (충전량 기준) | ↑ |
| M03 | `ve_dtw` | V-E | DTW | 동일, e_frac 축 [0,1] 50점 보간 | V-E 곡선 누적 형상 거리 (에너지 기준) | ↑ |
| M04 | `vt_frec` | V-t | Fréchet | `max\|a[i]−b[i]\|` (고정 그리드) | V-t 최대 순간 편차 (worst-case) | ↑ |
| M05 | `vq_frec` | V-Q | Fréchet | `max\|a[i]−b[i]\|` (고정 그리드) | V-Q 최대 순간 편차 | ↑ |
| M06 | `ve_frec` | V-E | Fréchet | `max\|a[i]−b[i]\|` (고정 그리드) | V-E 최대 순간 편차 | ↑ |

**3종 곡선 정의**

| 곡선 | x축 | y축 | 정규화 방법 | 포착하는 변화 |
|------|-----|-----|-----------|-------------|
| V-t | 정규화 시간 `t_frac = t_rel / t_total` ∈ [0,1] | V [V] | 구간 내 경과 시간 기준 | 동일 시간 위치에서의 전압 변화 → 동력학(kinetics) 변화 민감 |
| V-Q | 정규화 전하 `q_frac = q_cum / q_total` ∈ [0,1] | V [V] | 누적 전하량 기준 | 동일 SoC에서의 전압 변화 → 열역학(thermodynamics) 변화 민감 |
| V-E | 정규화 에너지 `e_frac = e_cum / e_total` ∈ [0,1] | V [V] | 누적 에너지 기준 | V-Q와 유사하나 에너지 가중 → 고전압 구간(높은 V) 변화에 더 민감 |

**2종 거리 지표 비교**

| 지표 | 수식 | 특성 | LFP에서 강점 |
|------|------|------|------------|
| DTW (Dynamic Time Warping) | 밴드 내 최적 정렬 DP, 결과 / n 정규화 | x축 위상 이동 ±10% 허용 → 누적 형상 차이 | 플래토 위치가 이동해도 형상 변화를 올바르게 측정 |
| 이산 Fréchet | `max\|a[i]−b[i]\|` (고정 그리드에서 대각 경로가 최적) | 최악 순간 편차 (leash 거리) | 전압 급변 구간(dis_hi, dis_lo)의 단일 큰 편차 포착 |

**BOL 참조 방식**

- **BOL 기준**: 셀별 최초 유효 사이클(cycle 1)의 곡선을 각 세그먼트·곡선 유형별로 저장
- **사이클 N의 거리**: 사이클 N의 [0,1] 보간 곡선 vs BOL 보간 곡선 간 거리
- **BOL 사이클 자체**: 거리 = 0 (또는 근사 0) — 이 점이 열화 추이의 기준점
- **독립성**: 카테고리 A·B·C와 달리 절대값이 아닌 **상대적 변화량**이므로 셀 간 제조 편차(baseline)가 제거됨

> **계산 주의사항**
> - 보간 그리드: `_MORPH_GRID = 50` 포인트, x축 균일 배치 [0, 1]
> - DTW 밴드: `_DTW_BAND = 5` (50포인트의 10%), n×n 거리행렬 numpy 사전계산 후 DP
> - Fréchet: 고정 그리드에서 이산 Fréchet = `max|a-b|` (O(n), numpy 1줄) — DP 불필요
> - V-E 계산: `e_cum = cumsum(V × |I| × dt) / 3600` [Wh]; `e_frac = e_cum / e_total`
> - 최소 샘플 수: 구간 내 8점 미만이면 곡선 추출 불가 → NaN
> - BOL 미등록 상태(최초 유효 구간 없음)이면 해당 세그먼트 전체 NaN
> - `chg_gap_seg` 플래그 구간(CC 프로토콜 전환 갭)은 충전 HI 자체가 NaN → morph도 NaN

---

## 3. 전체 구조 요약

```
Global HI (15)
├── Discharge Global (12): G01–G11, G15
│   └── 용량·에너지·전압·ICA·DVA·쿨롱효율
└── Charge Global (3): G12–G14
    └── CV 거동·충전 ICA

Segment HI (396 = 6 × 66)
├── chg_lo (SoC 0–40%, q_frac 0–0.4)    ← 충전 초반
│   ├── Statistical:   S01–S20 (20개)
│   ├── Differential:  D01–D20 (20개)
│   ├── LFP-specific:  L01–L20 (20개)
│   └── Morphological: M01–M06 (6개)
├── chg_mid (SoC 40–70%, q_frac 0.4–0.7) ← 충전 플래토 중심부
│   ├── ...
├── chg_hi (SoC 70–100%, q_frac 0.7–1.0) ⚠ CV 포화 구간 — ICA/DVA 주의
│   ├── ...
├── dis_hi (SoC 60–100%, q_frac 0–0.4)  ← 방전 초반, 고전압
│   ├── ...
├── dis_mid (SoC 30–60%, q_frac 0.4–0.7) ← 플래토 중심부, 신호 가장 강
│   ├── ...
└── dis_lo (SoC 0–30%, q_frac 0.7–1.0)
    ├── ...

Total: 15 + 396 = 411
키 명명:  stat_{k}_{seg}   /  diff_{k}_{seg}
          lfp_{k}_{seg}    /  morph_{k}_{seg}
```

---

## 4. 구간별 기대 신호 강도

신호 강도: ● 강 / ◐ 중 / ○ 약 / ⚠ 불안정

| HI 카테고리 | chg_lo | chg_mid | chg_hi | dis_hi | dis_mid | dis_lo |
|------------|--------|---------|--------|--------|---------|--------|
| 통계 — v_mean/std/p10/p90 | ○ | ○ | ◐ | ○ | ○ | ◐ |
| 통계 — v_ent/samp_ent/skew/kurt | ○ | ◐ | ◐ | ○ | ◐ | ◐ |
| 통계 — q_abs/energy | ● | ● | ● | ● | ● | ● |
| 통계 — i_mean/std/i_q_slope | ◐ | ○ | ● | ◐ | ○ | ◐ |
| **통계 — corr_vt / v_detrended_std** | ○ | ◐ | ◐ | ○ | ◐ | ◐ |
| 미분 — dvdq_mean/std/min/peak_q/valley_q | ◐ | ● | ⚠ | ◐ | ● | ◐ |
| 미분 — dqdv_peak_h/v/w / valley_h/v | ○ | ● | ⚠ | ○ | ● | ○ |
| **미분 — dqdv_area_asym** | ○ | ● | ⚠ | ○ | ● | ○ |
| 미분 — dvdt_slope/peak_asym/d2vdq2 | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ |
| LFP — plateau_frac/v_std/v_slope | ○ | ● | ⚠ | ◐ | ● | ○ |
| LFP — nonlin_idx/v_sag/v_concavity | ● | ◐ | ◐ | ● | ◐ | ● |
| LFP — knee_v/q_frac / plateau_q_onset | ● | ○ | ◐ | ● | ○ | ● |
| **LFP — dv_dt_plateau / v_ent_plateau** | ○ | ● | ⚠ | ◐ | ● | ○ |
| **LFP — v_gradient_exit** | ● | ○ | ◐ | ● | ○ | ● |
| 형태학 — DTW (vt/vq/ve) | ◐ | ● | ◐ | ◐ | ● | ◐ |
| 형태학 — Fréchet (vt/vq/ve) | ● | ◐ | ◐ | ● | ◐ | ● |

> ⚠ `chg_hi`: CV 포화 구간에서 dV→0, I→0이므로 dQ/dV→∞, dV/dQ→0 발산 가능.  
> 해당 구간 미분 기반 HI는 CV 구간 데이터 제외 후 CC 부분만 사용하거나 NaN 처리.

---

## 5. 기존 HI(148D)와의 비교

| 항목 | 기존 (148D) | 신규 v1 (321D) | 신규 v2 (411D) |
|-----|-----------|--------------|--------------|
| Global | 28개 (dis 22 + chg 6) | **15개** (축약·재정의) | **15개** (동일) |
| Segment | 120개 (6seg × 20) | **306개** (6seg × 51) | **396개** (6seg × 66) |
| 세그먼트 카테고리 | 혼합 | 통계/미분/LFP/형태학 4분류 | **동일** |
| 카테고리당 HI 수 | 혼합 | A/B/C 각 15, D 6 | **A/B/C 각 20, D 6** |
| ICA/DVA 위치 정보 | Global만 | Global + 구간별(높이·전압) | **+ 밸리 위치 D16–D19 추가** |
| IC 곡선 비대칭 | 없음 | peak_asym (좌우 반폭 비율) | **+ dqdv_area_asym (면적 비율, D20)** |
| 전압 복잡도 | 없음 | v_ent (히스토그램) | **+ v_samp_ent (시계열 패턴, S17)** |
| 플래토 기울기 | 없음 | ocv_slope (중점) | **+ plateau_v_slope (플래토 전구간 OLS, L16)** |
| 플래토 경계 | 일부 | phase_entry_dvdq (진입) | **+ v_gradient_exit (이탈, L17) — 진입·이탈 쌍** |
| 플래토 품질 | plateau_frac | plateau_frac + plateau_v_std | **+ v_ent_plateau (엔트로피, L20) + dv_dt_plateau (드리프트, L19)** |
| 논문 갭 DVPL/DVVL | 없음 | 없음 | **dvdq_peak_q / dvdq_valley_q (D18/D19)** |
| 논문 갭 ICV/ICVL | 없음 | 없음 | **dqdv_valley_h / dqdv_valley_v (D16/D17)** |
| 곡선 형상 거리 | 없음 | 카테고리 D 6종 × 6구간 = 36개 | **동일** |
| 이상치 취약 HI | ent_v (density 버그), dvdt (FP 아티팩트) | PMF 강제, 총 기울기 사용 | **동일** |

---

## 6. 구현 우선순위

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

Phase 4 (형태학적 거리 — 카테고리 D)
  morph_vq_dtw_{seg}   : V-Q DTW  ← 우선 검증 (LFP 열화 핵심 곡선)
  morph_vq_frec_{seg}  : V-Q Fréchet
  morph_vt_dtw_{seg}   : V-t DTW  (동력학 변화)
  morph_ve_dtw_{seg}   : V-E DTW  (에너지 가중 형상)
  morph_vt_frec_{seg}  : V-t Fréchet
  morph_ve_frec_{seg}  : V-E Fréchet
  → seg_corr_analysis.py 결과로 카테고리 D 상관계수 확인 후 Phase 2 승격 여부 결정
```

---

## 7. 논문 계통 대비 갭 분석

`docs/HI_refferences.md` 에 정리된 4대 논문 계통과 현재 411D HI 설계를 대조한 분석.

### 7-1. 논문 계통별 커버리지 요약

| 계통 | 대표 논문 | 핵심 아이디어 | 현재 커버 | 평가 |
|------|----------|-------------|----------|------|
| 1. 미분 기법 (ICA/DVA) | Berecibar 2016 (Energy) | DVA 두 피크 위치·거리·깊이 | Global G06–G10, Segment B카테고리 D01–D15 | **70%** — DVA 2차 피크 미포함 |
| 1. 미분 기법 (ICA/DVA) | Zheng 2018 (JPS) | 부분 CC 구간 IC 피크 추출 | Segment D06·D07 (구간별 ICA) | **85%** — 부분 곡선 외삽 미구현 |
| 2. 형태학적 특징 | Hu 2021 (IEEE Trans. Transp. Electr.) | V-Q·V-T 곡선 기하 피처 + 선택 | 카테고리 D (DTW·Fréchet × 3곡선 × 6구간) | **95%** — V-T 제외(temperature 데이터 없음) |
| 2. 형태학적 특징 | DTW + GPR (JES 2020) | DTW 거리를 단일 HI 스칼라로 | 카테고리 D `morph_v*_dtw_*` | **100%** |
| 3. 시간/에너지 기반 | Ref-based Time (IEEE Trans. PD 2023) | 고정 전압 [V₁,V₂] 구간 충전 시간 | 없음 (dvdt_slope은 용량 경계 기반) | **40%** — 핵심 갭 |
| 3. 시간/에너지 기반 | Energy-based GPR (JES 2022) | V-Q 적분 = 에너지 HI | G02 `energy_dis`, S12 `E_seg` | **100%** |
| 4. 딥러닝 잠재 특성 | Tian 2022 (ESM) | CNN 인코더 → latent vector HI | 의도적 제외 (hand-crafted 프레임워크) | N/A |

---

### 7-2. 갭 상세 분석

#### Gap 1: DVA 2차 피크 특성 — Berecibar 2016

**논문의 핵심**: LFP 방전 dV/dQ 곡선에는 플래토 양 경계(진입·이탈부)에 두 개의 골짜기(valley)가 존재한다.  
두 골짜기의 Q-거리 = 플래토 용량 = SOH 직결 지표.

**현재 설계 상황**:

| Berecibar HI | 현재 대응 | 비고 |
|---|---|---|
| DVA 1차 골짜기 Q 위치 | G09 `dva_valley_q` ✅ | |
| DVA 1차 골짜기 깊이 | G10 `dva_valley_depth` ✅ | |
| DVA **2차 골짜기** Q 위치 | 없음 ❌ | |
| DVA **2차 골짜기** 깊이 | 없음 ❌ | |
| 두 골짜기 간 Q 거리 | G05 `q_plateau_frac` (전압 경계 기반 근사) △ | DVA 피크 직접 측정 아님 |

**부분 방전 시 가시성 분석**:

```
dV/dQ 형태 (방전)
  |
0 |.......____________________________......... → Q
  |  Valley 1         plateau ≈ 0        Valley 2
  |  (플래토 진입)                       (플래토 이탈)
  |
  q_frac: 0   ~0.05~0.15            ~0.85~0.95   1.0
  SoC:  100%      85%                   15%        0%
```

| 방전 시나리오 | Valley 1 | Valley 2 |
|---|---|---|
| 완전 방전 (SoC 100→0%) — MIT | ✅ | ✅ |
| 부분 방전 (SoC 100→50%) — HUST 흔한 패턴 | ✅ | ❌ (미도달) |
| 부분 방전 (SoC 70→30%) — mid 구간만 | ❌ | ❌ |

**결론**: Valley 2는 심방전(SoC 0% 근접) 시에만 존재.  
HUST 부분 방전 데이터에서는 NaN 비율이 높아 **실용성 낮음**.  
MIT 전체 방전에서는 추출 가능하나, 현재 `dis_lo` 세그먼트의 `min(dV/dQ)` (D04)가 사실상 이 영역을 이미 커버.

**우선순위: 낮음** — 추가 효과 대비 NaN 비용이 큼.

---

#### Gap 2: 고정 전압 경계 충전 플래토 HI — 2023 Ref-based Time ★

**논문의 핵심**:  
충전 시작 SoC가 매 사이클 달라도, 고정 전압 [V₁, V₂] 내 데이터가 조금이라도 있으면 유효한 HI 추출 가능.  
→ **부분 충전 시나리오에서 가장 강건한 HI 중 하나**.

```
현재 q_frac 기반 세그먼트의 한계 (부분 충전 시)

cycle A (SoC 30→90% 충전):  chg_lo = SoC 30~52%, chg_mid = SoC 52~72%
cycle B (SoC 10→90% 충전):  chg_lo = SoC 10~42%, chg_mid = SoC 42~66%
→ 같은 q_frac 경계가 다른 절대 SoC를 가리킴

고정 전압 경계 [3.10V, 3.45V] 기반:
cycle A, B 모두 이 전압 범위 안 데이터 → 직접 비교 가능
```

**현재 설계 대응**:

| 논문 HI | 현재 대응 | 차이 |
|---|---|---|
| ΔT(V₁→V₂) — 충전 시간 | `dvdt_slope` (ΔV/Δt) 역수 유사 | q_frac 경계 기반, 전압 경계 아님 |
| Q(V₁→V₂) — 충전 용량 | G05 `q_plateau_frac` (방전 전용) | 충전 방향 없음 |
| 충전 플래토 용량 | 없음 | 완전 누락 |
| 충전 플래토 시간 | G13 `cv_time_frac` (전체 CV 비율) | 전압 구간 특정 아님 |

**신규 HI 제안**:

| 제안 키 | 수식 | 물리적 의미 | 열화 방향 |
|---|---|---|---|
| `q_chg_plateau` | `Q(CC phase ∩ V ∈ [3.10, 3.45V])` [Ah] | CC 충전 중 LFP 플래토 구간 충전 용량 | ↓ (R↑ → 빠른 전압 상승 → 구간 단축) |
| `t_chg_plateau` | `t(CC phase ∩ V ∈ [3.10, 3.45V])` [s] | 동 구간 경과 시간 | ↓ |

구현 예시:
```python
# hi_correlation.py 내 _extract_one_cell() 충전 처리 블록에 추가
chg_rows = cyc[cyc["phase"] == "charge"]
cc_plateau = chg_rows[
    (chg_rows["voltage_V"] >= 3.10) &
    (chg_rows["voltage_V"] <= 3.45) &
    (chg_rows["current_A"].abs() > CC_THRESHOLD)   # CC 상태
]
row["q_chg_plateau"] = float(np.trapz(np.abs(cc_plateau["current_A"]),
                                       cc_plateau["time_s"]) / 3600)
row["t_chg_plateau"] = float(cc_plateau["time_s"].iloc[-1]
                              - cc_plateau["time_s"].iloc[0]) if len(cc_plateau) > 1 else np.nan
```

Global HI로 추가 시 총 **411 → 413D**.

**우선순위: 높음** — 구현 난이도 낮음 (전압 마스킹 후 시간·전하 적산), HUST 부분 충전에 즉시 적용 가능.

---

#### Gap 3: 부분 IC 곡선 외삽 — Zheng 2018 (부분 구현)

**논문의 핵심**:  
부분 CC 충전 구간에서 ICA 피크 전체가 보이지 않을 때,  
현재 보이는 IC 곡선의 **기울기 추세**로 피크 위치를 추정할 수 있다.

**현재 설계 상황**:  
`dqdv_peak_v` (D07), `dqdv_peak_h` (D06) 는 피크가 존재할 때만 유효 → 피크 없으면 NaN.  
부분 충전 구간에서 NaN 비율이 높아 이 HI들의 정보 밀도가 낮음.

**보강 아이디어**:

| 제안 키 | 수식 | 물리적 의미 |
|---|---|---|
| `dqdv_slope_at_entry` | dQ/dV 곡선에서 구간 진입 5% 지점의 기울기 | IC 피크를 향해 오르고 있는 속도 → 피크 높이 간접 추정 |
| `dqdv_at_exit` | 구간 마지막 5% 지점의 dQ/dV 값 | 피크 이후 하강 속도 |

**우선순위: 중간** — 효과 불확실, NaN 채우기 이상의 새 정보를 주는지 검증 필요.

---

#### Gap 4: V-T 곡선 기반 HI — Hu 2021 (의도적 제외)

**논문의 핵심**: V-T(Voltage vs Temperature) 곡선의 기하 피처가 강력한 HI.

**제외 이유**: `_1_data_unified` 포맷에서 temperature_C 컬럼 제거됨.  
(`_0_data_raw` 에는 있으나 통합 파이프라인에서 제외된 상태)

**향후 복원 시 추가 가능한 HI**:
- `T_rise_seg`: 구간 내 온도 상승 [°C]
- `T_peak_seg`: 구간 최고 온도
- `corr_VT_seg`: V-T 상관계수 (발열과 전압 관계)

---

### 7-3. 신규 HI 보강 우선순위

| 순위 | 추가 HI | 관련 논문 | 기대 효과 | 구현 난이도 |
|------|--------|---------|---------|-----------|
| ★★★ | `q_chg_plateau`, `t_chg_plateau` | 2023 Ref-based Time | HUST 부분 충전 강건성 확보, 논문 직접 인용 가능 | 낮음 |
| ★★☆ | `dva_valley2_q/h`, `dva_valley_spacing` | Berecibar 2016 | MIT 전체 방전에서 유효, 논문 인용 | 중간 (MIT only) |
| ★☆☆ | `dqdv_slope_at_entry` | Zheng 2018 | NaN 채우기 보조, 효과 불확실 | 낮음 |
| — | V-T curve HI | Hu 2021 | temperature 데이터 복원 시 추가 | 높음 (파이프라인 변경) |

---

### 7-4. 논문 인용 전략

```
서론 논리 흐름:

단일 계통 HI의 한계
  └── ICA/DVA (계통1): 전체 사이클 필요, 부분 충전 불가 (Berecibar 2016)
  └── 기준 전압 시간 (계통3): 부분 충전 강건하나 형상 정보 없음 (2023 Ref-based Time)
  └── 형태학적 거리 (계통2): 곡선 전체 정보 활용하나 절대 기준 필요 (DTW: JES 2020)
          ↓
  "본 연구는 4대 계통 방법론을 통합한 411D 피처 풀을 구성하고,
   동적 부분 충전 시나리오에서 SOH 추적 능력을 구간별(segment-wise)
   within-cell Spearman 상관계수로 검증한다"

인용 위치:
  카테고리 A (통계): Zheng 2018 §2.1 "partial CC segments"
  카테고리 B (미분): Berecibar 2016 (DVA), Zheng 2018 (partial IC)
  카테고리 C (LFP): Berecibar 2016 §3.2 "plateau capacity"
  카테고리 D (형태학): Hu 2021 (fusion features), JES 2020 (DTW)
  q_chg_plateau:    IEEE Trans. PD 2023 (Ref-based Time)
```

---

## 부록: SoC 구간 분할 설계 근거

### 전압 기준 vs 누적 전하량 비율(q_frac) 기준

HI 추출 시 사이클을 SoC 구간별로 나눌 때 **전압 임계값**을 경계로 삼는 방법과  
**누적 전하량 비율(q_frac)** 을 경계로 삼는 방법 중 후자를 채택했다.

```
q_frac = cumsum(|I| · dt) / Q_total   (0 ~ 1)
```

#### LFP 플래토 문제

LFP 양극 특성상 3.2 ~ 3.45 V 구간(플래토)에 전체 용량의 70 ~ 80 %가 집중된다.  
전압 임계값으로 구간을 나누면:

| 전압 구간 | 실제 포함 용량 | 문제 |
|-----------|---------------|------|
| V < 3.2 V | ~15 % | 데이터 극소수 |
| 3.2 ~ 3.4 V | ~70 % | 모든 데이터 쏠림 |
| V > 3.4 V | ~15 % | 데이터 극소수 |

플래토에 데이터가 몰려 구간 분할이 사실상 의미 없어진다.

#### q_frac 기준의 장점

| 항목 | 전압 기준 | q_frac 기준 |
|------|-----------|-------------|
| 구간 내 데이터 수 | 불균등 (플래토 쏠림) | 균등 (항상 전체의 일정 비율) |
| 열화 영향 | 전압 곡선 이동 → 경계 틀어짐 | 총 Q 대비 비율이므로 열화와 무관 |
| C-rate 영향 | IR drop으로 전압 왜곡 | 상대적으로 작음 |
| SoC 근사 정확도 | LFP 비선형 OCV → 부정확 | Q ∝ SoC이므로 선형 근사 |

#### q_frac 기준의 한계

- **절대 SoC 이동**: 열화가 진행될수록 Q_total이 감소하므로 q_frac 40 %의 절대 Ah가 달라진다. 사이클 초반과 말기에 동일한 q_frac 구간이 약간 다른 SoC 위치를 가리킬 수 있다.
- **부분 충방전 민감**: Q_total이 작은 부분 사이클이 섞이면 q_frac 의미가 희석된다.

#### 구간 경계 설정 (0.4 / 0.7)

SoC 경계를 0.3 / 0.6 대신 q_frac 0.4 / 0.7 로 설정한 이유:  
LFP 플래토(SoC 약 20 ~ 80 %)가 실제 용량 기준으로 넓게 걸쳐 있어  
중간 구간이 플래토 핵심을 충분히 포함하도록 경계를 약간 바깥으로 잡았다.

| 세그먼트 | q_frac | 근사 SoC | 물리적 의미 |
|----------|--------|----------|-------------|
| `chg_lo` (충전) / `dis_hi` (방전) | 0 ~ 40 % | 0~40% / 60~100% | 충전 초반(저전압) / 방전 초반(고전압) |
| `chg_mid` / `dis_mid` | 40 ~ 70 % | 40~70% / 30~60% | 플래토 중심 |
| `chg_hi` (충전) / `dis_lo` (방전) | 70 ~ 100 % | 70~100% / 0~30% | 충전 후반(CV 포함) / 방전 후반(저전압) |

방전과 충전은 진행 방향이 반대(SoC 감소 vs 증가)이나 q_frac 계산 로직은 동일하며 SoC 라벨만 반전된다.
