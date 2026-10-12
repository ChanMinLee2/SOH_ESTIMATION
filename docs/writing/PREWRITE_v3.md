# PREWRITE_v3.md — 논문 초안 섹션별 콘텐츠 개조식 스토리보드 (구조 재편판)

> v2 대비 변경: §4(데이터셋/전처리/다중공선성)를 §3 Methodology로 이동(모델이
> 소비하는 입력을 모델보다 먼저 설명 — forward-reference 해소). §4는
> "Experiment"로 개편해 학습·평가 프로토콜+파라미터 테이블만 남기고, 옛 §5
> Results 전체를 흡수. §6 Conclusion → §5로 재번호.

---

## 핵심 기여 (3대 축)

1. **①** 부분구간 위치 식별로 시나리오 서브셋 학습
2. **②** 위치별 열화 반영한 카테고리형 HI 설계
3. **③** 시너지 HI 그룹화 및 커널 압축으로 다중공선성 해소

## 제목(후보)

- Category-Based Health Indicators and Synergy-Driven Kernel Fusion for Scenario-Conditioned Battery SOH Estimation
- (대안) An LFP-Centric Framework for Partial-Cycle Battery SOH Estimation with Cross-Chemistry Transferability to NCM

## Abstract

- 문제: 실사용은 SOC 일부 구간만 관측 가능

<!-- -->

- 기여①: 부분구간 위치 식별해 대응하는 구조
- 기여②: 위치별 열화 반영 카테고리형 HI 설계
- 기여③: 시너지 HI 그룹화 및 커널 압축으로 다중공선성 해소

<!-- -->

- 구조: L0 게이트로 시나리오별 HI 서브셋 선택
- 데이터: LFP·NCM 3개 데이터셋 사용
- 결과: hard 라우팅 기준 R² 0.94~0.95대

---

## 1. Introduction

### 1.1 문제 정의

- [REF] SOH 추정, 배터리 관리에서 중요도 상승

<!-- -->

- **(1-1)** 기존 연구 다수는 완전충전 데이터로 실험 [REF]
- **(1-2)** 부분 데이터도 구간 모르면 유의성 판단 어려움
- **(1-3)** 범용 설계는 LFP 특이 거동 반영 못함 [REF]

<!-- -->

- LFP 특화하되 타 화학종 재적용 가능성 확인 필요

### 1.2 컨트리뷰션

- ① 부분구간 위치를 식별하는 구조 제안
- ② 위치별 열화 정보 반영한 카테고리 HI 설계
- ③ 시너지 HI 그룹화 및 커널 압축으로 다중공선성 해소

---

## 2. Related Work

### (A) 물리기반/모델기반 방법론

- ECM 등 물리기반 접근도 활발히 연구됨 [REF]
- 전기화학 해석엔 별도 실험·장비 필요 [REF]

<!-- -->

- 본 연구는 데이터 기반+해석 가능 HI로 보완

### (B) 완전충전 가정

- Tian(2022): 임의 구간 쓰나 화학종 비교 제한적
- GPR 계열: 완충곡선 전체 입력 전제가 많음

<!-- -->

- 완주 가정 설계는 보정 구간 다루지 않는 편
- 실배포는 완주 전 총충전량 알기 어려움 [REF]

### (C) HI 설계의 범위

- Soon(리뷰): HI 분류하나 통합 적용 사례는 없음
- 통계·IC 외 범주는 taxonomy에 빠진 편
- 모든 특성 반영한 연구는 흔치 않음

<!-- -->

- Gong(2026): 멀티데이터셋 학습하나 설계 HI는 없음

### 정리

- 세 갈래 문제의식을 하나의 구조로 함께 고려
- (1-3) 뒷받침할 Related Work 보강 필요

---

## 3. Methodology

### 3.1 데이터셋

<!-- Fig(신규): 3개 데이터셋 비교 플랏. -->

**Table 1.** 데이터셋 개요(정확 숫자 미기입)

| 데이터셋 | 화학종 | 셀 수 | 사이클 수(대략) | 프로토콜 특징 |
|---|---|---|---|---|
| MIT(Severson) | LFP | - | - | 고속충전 프로토콜 다양 |
| HUST | LFP | - | - | 방전 프로토콜 다양 |
| TJU | NCM | - | - | - |

- 3개 데이터셋 출처·공개 경로 각각 명시
- 화학종별 전압창·공칭용량 차이 수치 명시
- 충방전 프로토콜 차이를 데이터셋별 요약

<!-- -->

- 셀 단위 분할로 재현(동일 split_seed)

### 3.2 시나리오 정의와 동기 — Fig 1

<img src="../figures/fig1_problem_scenario_definition.png" width="820" alt="Figure 1">

- Fig1(a): 충/방전 V-q_frac 곡선, SOC 6구간 표시
- Fig1(b): 화학종별 dV/dq 피크 모양·위치가 서로 다름

<!-- -->

- 화학종마다 전압-용량 곡선 형태 다름(§3.1)
- 시나리오 = SOC 6구간(충/방전×저/중/고)
- SOC 기준(q_frac)은 그 사이클 q_tot로 정규화

<!-- -->

- [수식1] q_frac 정의식
- 레퍼런스엔 센서 바이어스·드리프트 반영 [REF]
- [수식2] 레퍼런스 노이즈 모델(OU/AR(1))
- 완주 전제·재보정 부재 문제는 §2(B)에서 다룸

<!-- -->

- 이 정의 기준으로 3.3~3.6 방법 전개
- <u>의미: 화학종 차이와 시나리오 구간을 실데이터로 제시</u>

### 3.3 카테고리별 HI 설계 — Fig 2, Table 2

**Table 2.** HI 카테고리별 정의

| 카테고리 | n | 물리량 |
|---|---|---|
| stat | 18 | 전압/전류 분포 통계 |
| diff(dQ/dV) | 20 | 미분전압 피크·곡률 |
| lfp-specific | 20 | LFP 2상전이 평탄구간 |
| morphology | 6 | 기준곡선 형태 유사도 |

<img src="../figures/fig4_hi_scenario_deviation.png" width="460" alt="Figure 2">

- 열화 메커니즘별 HI 4개 카테고리로 분류
- Fig2: 카테고리별 시나리오 간 편차 비교

<!-- -->

- 카테고리 간 상관 그대로 쓰면 다중공선성 발생
- 카테고리 내부도 민감도 제각각(§4.4 재확인)

### 3.4 전처리 — 상관성 → 시너지 그룹화 → 커널 융합 — Fig 3

<img src="../figures/synergy+kernel.png" width="600" alt="Figure 3">

- (1) interaction: 시나리오별 HI 유의성 탐지
- (2) synergy: 중복 정리 후 편상관으로 그룹화
- (3) kernel: Nystroem+Ridge로 그룹→커널 HI 1개

<!-- -->

- [수식5] 상관 임계값 기준 중복 판정식
- [수식6] 커널 융합(Nystroem+Ridge)

<!-- -->

- 3단계가 HI 전환과 다중공선성 정리를 겸함
- <u>의미: 전처리 알고리즘을 재현 가능하게 명시</u>

### 3.5 다중공선성 제거(결합 raw+kernel)

- §3.3의 다중공선성을 이 단계에서 실제 해소
- 시나리오별 결합(raw+kernel) 상관으로 HI 배제

<!-- -->

- 분류기 쪽 확장은 §5 향후과제로 이동

### 3.6 시나리오 조건 서브셋 학습 구조(메인 모델) — Fig 4

<img src="../figures/architecture.png" width="600" alt="Figure 4">

- 분류기가 방향+구간 정보로 시나리오 추정
- 회귀기는 시나리오별 전용 L0 게이트로 HI 선택
- raw HI 게이트는 공유+전용 혼합 구조
- 커널 HI 게이트는 시나리오 전용 구조
- Fig4: 라우팅-게이팅 전체 forward 구조도

<!-- -->

- [수식3] HardConcrete 게이트 확률식
- [수식4] 전체 손실함수

<!-- -->

- 학습된 게이트 확률은 §4.4에서 실험 제시
- "조건화 없음" 비교군은 §4.3 ablation 조건

---

## 4. Experiment

### 4.1 학습·평가 프로토콜

- 지표: RMSE/MAE/R², hard 라우팅 기준
- 구현: 9단계 파이프라인, 설정 파일 하나로 통일
- 체크포인트: val_rmse+gate_saturation 기준

**Table 3.** 주요 하이퍼파라미터

| 파라미터 | 값 |
|---|---|
| λ_l0(L0 희소화 가중치) | 0.000237 |
| λ_scen(분류 CE 가중치) | 0.01 |
| learning rate | 2×10⁻⁴ |
| weight decay | 1×10⁻³ |
| batch size | 2048 |
| max epochs(patience) | 500(60) |
| d_probe / d_head | 64 / 128 |
| dropout | 0.2 |
| seed | 42 |

### 4.2 예측 진단 및 SOH 예측 궤적 — Fig 5, Fig 6

<img src="../figures/fig3_diagnostics.png" width="900" alt="Figure 5">

- Fig5(a): 예측-실측 SOH 산점도(y=x 대비 분산 확인)
- Fig5(b): 오차(%) 히트맵, cycle×capacity 기준
- Fig5(c): 오차(%) 히트맵, scenario×capacity 기준

<!-- -->

- 저용량(열화 말기) 구간일수록 오차가 커짐
- 시나리오별로도 오차 분포가 고르지 않음(dis_mid 등)

<!-- -->

- <u>의미: 구간별 신뢰도 차이를 정량적으로 제시</u>

<img src="../figures/fig10_v4_example.png" width="460" alt="Figure 6">

- 대표 셀 1개의 예측-실측 궤적을 색으로 비교

<!-- -->

- 열화 전구간에서 예측이 실측을 잘 따라감
- 저SOH 구간일수록 오차가 커지는 경향

<!-- -->

- 정량 성능 표는 필요시 부록으로
- <u>의미: 제안 구조가 실제로 동작함을 입증</u>

### 4.3 시너지·커널 융합 효과(ablation) — Fig 7

<img src="../figures/fig4_ablation_synergy.png" width="820" alt="Figure 7">

- 4조건 비교: no-scenario/group/kernel/제안구조
- 시너지 빼면 성능 나빠짐(그룹화 효과)
- 커널 빼면 성능 더 나빠짐(커널 효과 더 큼)
- "조건화 없음" 비교군이 4조건 중 하나

<!-- -->

- Fig7(b): 진짜 시너지 그룹에 게이트 사용 집중
- <u>의미: 게이트 선택이 이론적 시너지와 일치</u>

<!-- -->

- (향후과제) 대표 baseline 2~3개 인용·적용
- 예: Ridge/RF를 동일 HI·split에 재현

### 4.4 시나리오별 서브셋 차이 실증 — Fig 8, Fig 9

<img src="../figures/fig5_hi_selection_trimmed.png" width="480" alt="Figure 8">

- Fig8: 학습된 게이트 확률 히트맵(선택 HI만)

<!-- -->

- 학습된 서브셋이 시나리오마다 실제로 다름
- <u>의미: 고정 HI 세트 대신 적응형 설계가 필요함</u>

<img src="../figures/fig9_jaccard_cross_chemistry.png" width="460" alt="Figure 9">

- Fig9(a): 화학종 간 선택 HI 자카드 비교
- Fig9(b): 커널 HI 자카드(raw 경유) 비교

<!-- -->

- 시나리오·화학종마다 선택 HI가 달라짐
- "고정 HI 세트" 하나로는 부족함을 시사
- <u>의미: 화학종 넘나드는 범용 HI 세트는 비현실적</u>

### 4.5 NCM 재적용 검증(보조 결과) — Fig 10

<img src="../figures/fig8_single_dataset.png" width="460" alt="Figure 10">

- MIT/HUST/TJU 독립 학습·평가 결과 비교
- LFP 설계 구조, NCM 재적용해도 성능 유지

<!-- -->

- 검증 엄격도 가장 가까운 선례는 Gong(§2(C))
- <u>의미: LFP 특화 설계도 NCM에 전이 가능함을 입증</u>

---

## 5. Conclusion

- ① 검증: 분류 정확도로 위치 식별 확인(Fig5)
- ② 검증: 카테고리·서브셋 차이로 확인(Fig2,8,9)
- ③ 검증: ablation으로 그룹화·압축 효과 확인(Fig7)

<!-- -->

- 한계: 강건성 데이터 공백, multi-seed 미실시

<!-- -->

- 향후: 분류기 커널+다중공선성 확장
- 향후: baseline 비교, 문헌 7~12편 보강

---

## 부록 — 그림 배치 요약표 (11개, 신규 1개 포함)

| Fig | 위치 | 상태 | 비고 |
|---|---|---|---|
| 1 | 3.2 시나리오 정의와 동기 | ✅ | `fig1_problem_scenario_definition.png` |
| 2 | 3.3 HI 설계 | ✅ | `fig4_hi_scenario_deviation.png` |
| 3 | 3.4 전처리 파이프라인 | ✅ | `synergy+kernel.png` |
| 4 | 3.6 아키텍처(메인 모델) | ✅ | `architecture.png`(디자인 외주) |
| 5 | 4.2 예측 진단(산점도+오차 히트맵×2) | ✅ | `fig3_diagnostics.png`(1×3) |
| 6 | 4.2 SOH 예측 궤적 | ✅ | `fig10_v4_example.png` |
| 7 | 4.3 Ablation(시너지+커널) | ✅ | `fig4_ablation_synergy.png`(1×2) |
| 8 | 4.4 학습된 서브셋 | ✅ | `fig5_hi_selection_trimmed.png` |
| 9 | 4.4 자카드 | ✅ | `fig9_jaccard_cross_chemistry.png` |
| 10 | 4.5 NCM 재적용 검증 | ✅ | `fig8_single_dataset.png` |
| 신규A | 3.1 데이터셋 비교 | 신규 제작 | 용량열화 overlay 등 |

## 부록 — 후보 그림군 (메인 흐름에서 제외, 보조자료용)

| 후보 | 원본 파일 | 제외 이유 | 예상 위치(필요시) |
|---|---|---|---|
| (구)Fig2 시나리오 정의 | `fig2_scenario_spec.png` | Fig1과 의미 중복 | 3.2 보완용 |
| (구)커널 개수/R² 막대 | `fig5_kernel_expansion.png` | 파이프라인 도식이 더 적합 | 3.4 보조 패널 |
| 강건성(3D 플랏) | `fig11_robustness_3d.png` | synthetic, 데이터 공백 | 범위 밖(§5 향후과제) |
| HI 64개 상관 매트릭스 | `fig3_hi_design_rationale.png` | Fig2(b)로 충분 | 필요시 §3.5 보조 패널 |
| 서브셋 선택(raw+kernel 전체) | `fig7_subset_selection.png` | trimmed 버전이 더 적합 | 참고용 |
| 전처리 파이프라인(구버전) | `kernel_fusion_template.png` | synergy+kernel.png로 교체 | 참고용 |
| 아키텍처(자동생성판) | `fig6_arch_scr_model.png` | architecture.png로 대체 | 참고용 |

## 부록 — 수식 배치 계획

| 번호 | 수식 | 위치 | 비고 |
|---|---|---|---|
| [수식1] | q_frac 정의(세그먼트 전류적산/q_tot) | §3.2 | SOC 기준 정의 |
| [수식2] | 레퍼런스 노이즈 모델(OU/AR(1)) | §3.2 | 세부는 구현 문서로 대체 가능 |
| [수식3] | HardConcrete 게이트 확률식 | §3.6 | 메인 모델 핵심 수식 |
| [수식4] | 전체 손실함수(MSE+λ_scen·CE+λ_l0·L0) | §3.6 | 메인 모델과 함께 설명 |
| [수식5] | 다중공선성 임계값(|r|≥0.95) | §3.4 | 전처리 단계 |
| [수식6] | 커널 융합(Nystroem+Ridge) | §3.4 | 전처리 단계 |
| — | 평가지표(RMSE/MAE/R²) | §4.1 | 표준식, 각주로 처리 가능(미작성) |

## 부록 — 수식 전체 모음

**[수식1]** q_frac 정의식(§3.2):
$$
q(t) = \int_0^t I(\tau)\, d\tau, \qquad
q_{\mathrm{frac}}(t) = \frac{q(t)}{q_{\mathrm{tot}}}
$$

**[수식2]** 레퍼런스 노이즈 모델(§3.2, OU/AR(1) 드리프트, 사이클 인덱스 $c$ 기준):
$$
\varepsilon_c = \phi\, \varepsilon_{c-1} + \eta_c, \quad
\eta_c \sim \mathcal{N}(0, \sigma_\eta^2), \quad
\phi = \exp\!\left(-\tfrac{1}{T_{\mathrm{period}}}\right)
$$
$$
\varepsilon_c \leftarrow \mathrm{clip}(\varepsilon_c,\, -A,\, A), \qquad
q_{\mathrm{ref}}(t) = q_{\mathrm{cum,ref}}^{(c-L)}(t) + b + \varepsilon_c
$$
($b$: 셀·방향별 고정 바이어스, $L$: ref_lag, $A$=noise_amp, $T_{\mathrm{period}}$=noise_period_cycles)

**[수식3]** HardConcrete 게이트 확률식(§3.6, 학습 시 — $U \sim \mathrm{Uniform}(0,1)$, $\beta=2/3$, $\gamma=-0.1$, $\zeta=1.1$):
$$
s = \sigma\!\left(\frac{\log U - \log(1-U) + \log\alpha}{\beta}\right), \qquad
z = \mathrm{clip}\big(s(\zeta - \gamma) + \gamma,\ 0,\ 1\big)
$$
추론 시 게이트 확률(선택 여부 판정):
$$
P(z \neq 0) \approx \sigma\!\left(\log\alpha - \beta \log\frac{-\gamma}{\zeta}\right)
$$

**[수식4]** 전체 손실함수(§3.6):
$$
\mathcal{L} = \underbrace{\frac{1}{N}\sum_{n=1}^N (\hat{y}_n - y_n)^2}_{\text{MSE}}
\;+\; \lambda_{\mathrm{scen}} \underbrace{\left(-\frac{1}{N}\sum_{n=1}^N \log \hat{p}_{n, c_n}\right)}_{\text{CE}}
\;+\; \lambda_{l0} \underbrace{\sum_{i} P(z_i \neq 0)}_{\text{L0 penalty}}
$$

**[수식5]** 상관 임계값 기준 중복 판정식(§3.4, $\tau=0.95$):
$$
\rho_{ij} = \mathrm{corr}(x_i, x_j), \qquad
\text{redundant}(i,j) \iff |\rho_{ij}| \geq \tau
$$

**[수식6]** 커널 융합(§3.4, RBF 커널의 Nystroem 근사 + Ridge 회귀, 랜드마크 집합 $Z$, $|Z|=m$):
$$
K(x, x') = \exp\!\left(-\gamma \lVert x - x' \rVert^2\right), \qquad
\phi(x) = K(x, Z)\, K(Z, Z)^{-1/2} \in \mathbb{R}^m
$$
$$
h_{\mathrm{kernel}}(x) = w^\top \phi(x), \qquad
w^\ast = \operatorname*{arg\,min}_w \lVert y - \Phi w \rVert_2^2 + \alpha \lVert w \rVert_2^2
$$

## 부록 — 페이지 예산 추적 (목표 10~15p, 15p 절대 금지)

| 섹션 | 목표 분량 | 누적 |
|---|---|---|
| 1 Introduction | ~1p | 1p |
| 2 Related Work | ~2p | 3p |
| 3.1(데이터셋) | ~0.5p | 3.5p |
| 3.2~3.3(시나리오·HI) | ~1.5p | 5p |
| 3.4~3.5(전처리·다중공선성) | ~1.5p(추정) | ~6.5p |
| 3.6(메인 모델) | ~1p(추정) | ~7.5p |
| 4.1(프로토콜+파라미터 표) | ~0.5p | ~8p |
| 4.2~4.5(구 Results) | ~3~4p(그림 6개) | ~11~12p |
| 5 Conclusion | ~0.5p | ~11.5~12.5p |
| Abstract/References/여백 | 나머지 | 10~15p 범위 내 |

> §3이 데이터셋+전처리+다중공선성까지 흡수하면서 구 Methodology 대비
> 약 2p 늘어남 — 총합 자체는 §4/§5가 합쳐지며 상쇄돼 v2와 비슷한
> 11.5~13p대로 추정(실측 전까지는 추정치).

## 부록 — 미해결 항목

- Table 1 셀/사이클 수 채우기 — 현재 미기입
- 3.1 데이터셋 비교 플랏 — 미제작

<!-- -->

- Related Work 추가 문헌 7~12편 탐색
- SOC 레퍼런스 재보정 가정 외부 인용 1편 이상
- ECM 등 물리기반 방법론 인용 1~2편(§2(A))
- (1-3) LFP 성능 저하 뒷받침 문헌 신규 확보 [REF]

<!-- -->

- baseline(Ridge/RF) 재현 비교(§4.3) — 미착수
- 강건성·multi-seed·실차 검증 — 범위 밖

<!-- -->

- 분류기 커널+다중공선성 확장 — 후속 논문 후보
- 15페이지 조판 테스트 — 실제 확인 필요
