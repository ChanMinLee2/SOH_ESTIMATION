# Related Work — 논문 요약

작성일: 2026-09-01(2026-09-02 5편 추가, 총 7편). 전부 원문 그대로 읽고 정리함(초록/제목
검색 아님) — 2026-09-02분은 서브에이전트로 병렬로 원문을 읽되, 종합 판단(차별성/랭킹)은
직접 함.

---

## 1. Soon et al., "A review of selection of health indicators for predicting battery health: Analysis, challenges and future directions"

**저널**: *e-Prime – Nexus of Electrical, Electronic, and Intelligent Engineering* 17 (2026) 201180, Elsevier (오픈 액세스, CC BY-NC-ND). ScienceDirect에서 검색했을 때 "Chemical Engineering Journal"로 잘못 추정했었는데, 원문 확인 결과 정정함 — 실제 저널은 e-Prime이다.

### Contribution
- 논문 자체의 실험은 없는 **리뷰(taxonomy) 논문**. 기존 리뷰들이 "어떤 모델을 쓸지"(model-centric)에 집중한 데 반해, 이 논문은 "어떤 HI를 입력으로 쓸지"(feature-centric)에 집중한다는 게 핵심 차별점.
- HI를 **Direct HI**(Capacity/SOH/내부저항/Capacitance — 배터리 상태를 직접 측정)와 **Indirect HI**(충방전 데이터에서 간접적으로 뽑아낸 파생 피처)로 크게 나누고, Indirect HI를 다시 7개 하위군(Incremental Capacity, Battery Voltage, Battery Current, Battery Temperature, Charging Time, Discharging Time, Statistical Measures)으로 세분화한 taxonomy를 제시(Table A1, 최근 5년 문헌 정리).
- HI 선정의 문제를 5개 영역(데이터 관련, 피처 엔지니어링, 용량 재생(capacity regeneration), 셀 편차, 운용 조건)으로 나눠 도전과제를 분석하고, 각각에 대한 향후 방향을 제시.

### 실험 결과
해당 없음(리뷰 논문) — 대신 부록 Table A1에 최근 5년 문헌 109편의 HI/모델/데이터셋/성능지표를 정리한 메타 테이블 제공.

### 내 논문에서 인용할 필요성: **높음**
- 이 프로젝트가 "HI를 체계적으로 4개 카테고리(STAT/DIFF/LFP/MORPH)로 나눠 SOH 입력에 쓴다"는 접근 자체를 정당화하는 데 가장 직접적으로 쓸 수 있는 근거 논문 — "HI 선택이 모델 성능을 좌우하는 병목"이라는 이 논문의 핵심 주장이 이 프로젝트의 전체 v0~v5 게이트 안정성 투자(어떤 HI를 고를지가 왜 중요한가)의 동기와 정확히 일치함.
- 다만 아래 "HI 정의 커버리지 분석"에서 보듯, 이 논문의 taxonomy를 이 프로젝트가 얼마나 반영하는지는 **부분적**이라 — "이 리뷰의 분류를 따랐다"가 아니라 "이 리뷰가 정리한 것 중 일부(IC/통계)를 취하고, 나머지(시간 기반/전압 임계값 기반)는 의도적으로 배제했으며, 리뷰에 없는 카테고리(LFP 플래토, 곡선형태 유사도)를 추가했다"는 식으로 인용하는 게 정확함.

---

## 2. Tian et al., "Flexible battery state of health and state of charge estimation using partial charging data and deep learning"

**저널**: *Energy Storage Materials* 51 (2022) 372–381, Elsevier. Beijing Institute of Technology.

### Contribution
- 부분 충전 데이터(임의 시작 전압, 400초 분량 등)만으로 SOH(최대용량)와 SOC(잔존용량)를 **동시에** 추정하는 end-to-end CNN 제안 — 기존 연구처럼 "고정된 전압 구간에서 손으로 피처를 뽑는" 전제를 없애고, raw 전압 시퀀스를 그대로 CNN에 넣음(피처 엔지니어링 자체를 생략).
- 두 태스크(SOH, SOC)를 동시에 학습(multi-task learning)하면 각각 따로 학습하는 것보다 오히려 더 정확하다는 걸 실증 — loss 가중치 γ(식 14)로 두 태스크 기여도를 조절하며 검증.
- 입력 윈도우 길이(100~1000초)를 바꿔가며 성능 변화를 정량적으로 분석(Fig. 7, Table 4) — 이 프로젝트의 "세그먼트 분할 길이별 성능 변화" 목적(3번)과 방법론적으로 가장 가까운 논문.

### 실험 결과
- Oxford 열화 데이터셋(0.74 Ah × 8셀) 기준: 400초 충전 데이터로 SOH/SOC(최대/잔존 용량) RMSE 각각 **12.68 / 11.71 mAh**(공칭용량 대비 1.6%대).
- γ=0.4~0.8 구간에서 동시추정이 단일추정보다 더 정확(멀티태스크 학습 이득 실증, Table 1/2).
- 입력 윈도우 100초→1000초로 늘리면 RMSE가 28.62→9.26 mAh로 개선되지만, **400초 이후로는 개선 폭이 급격히 줄어듦**(수확체감) — "세그먼트 길이 대비 성능"을 정량적으로 보여주는 핵심 실험(Fig. 7).
- CNN이 RF/SVR/GPR/MLP/EN 등 5개 전통 ML 방법보다 전부 우수(Table 3, RMSE 12.68 vs 19~157 mAh).
- LFP 화학(LR1865SZ, 고속충전) 및 NASA 데이터셋(방전 데이터, 저샘플링)에도 일반화 확인 — 화학종/충방전 방향 무관하게 재현됨.

### 내 논문에서 인용할 필요성: **중간~높음**
- **세그먼트 분할 길이에 따른 성능 변화(목적 3)**를 다루는 데 가장 구체적인 정량적 선례 — 이 프로젝트가 세그먼트 길이(n1/n2) 스윕을 할 때 "몇 초/몇 % 이상부터 수확체감이 오는가"를 비교할 벤치마크로 인용 가능.
- 다만 이 논문은 **HI를 아예 안 쓰고 raw 시계열을 CNN에 직접 넣는 접근**이라, 이 프로젝트의 핵심 방향(HI 기반 게이트 선택으로 해석가능성 확보)과는 철학적으로 반대에 가깝다 — "왜 raw 대신 HI를 쓰는가"를 정당화하는 대조군(counterpoint)으로 인용하거나, "raw 기반 접근도 있지만 우리는 해석가능성을 위해 HI 경로를 택했다"는 식으로 위치시키는 게 적절함. 직접적인 방법론 비교(CNN vs 우리 게이팅)로 인용하기엔 목적이 다름.

---

## 3. Ke et al., "A sparse SOH estimation framework based on clustered voltage segments and multi-head CNN under arbitrary charging condition"

**저널**: *Energy* 344 (2026) 140057, Elsevier. Fujian Institute of Research on the Structure of Matter(CAS) / Xi'an Jiaotong Univ. / Hefei Univ. of Technology.

### Contribution
- 임의의 짧은 부분 충전 구간에서 IC(dQ/dV) 곡선 10점 벡터를 뽑고, **K-means**(Gap statistic으로 K=5 선택)로 구간을 형태별로 클러스터링 → 클러스터마다 별도 CNN head(Multi-head 1D-CNN) → attention으로 head 출력을 융합하는 구조.
- "sparse"는 학습된 피처 선택(L0 게이팅 등)이 아니라 **입력 데이터 자체가 짧은 부분구간**이라는 뜻 + 경량 모델(137.9K 파라미터)이라는 뜻 — 이 프로젝트의 "sparse gating"과는 용어만 같고 메커니즘은 다름.
- 클러스터→CNN head 라우팅은 **오프라인에서 한 번 확정한 뒤 고정된 룩업 테이블**로 배포 시 그대로 씀(학습된 분류기가 아님) — 이 프로젝트의 oracle/hard/soft 구분과 유사한 문제의식이지만 훨씬 단순한 해법.
- 세그먼트 개수(1~20)·전압 윈도우 폭/위치를 광범위하게 스윕(Fig. 7-9, 100회 랜덤 샘플링 포함) — 이 프로젝트의 목적3(세그먼트 길이 민감도)과 방법론적으로 가장 가까운 선례.

### 실험 결과
- 3개 데이터셋(자체 NMC 4셀, Oxford Artemis NCA/NMC, MIT/Severson LFP 124셀). 최적 조건 MAE 0.95%, R²≈96.9%. 전압 윈도우 0.11V→0.275V로 넓히면 RMSE 1.7%→0.8%. Ensemble-RVM/CNN-LSTM/Informer/Transformer 등보다 우수.

### 내 논문에서 인용할 필요성: **높음**
- 목적2(시나리오)의 가장 근접한 선행연구 — 다만 충전측만 다루고(방전 없음), K-means 클러스터링은 순수 데이터 기반이라 LFP 2상 플래토라는 물리적 근거로 시나리오를 나눈 이 프로젝트와 결이 다름. 목적3(세그먼트 길이) 재설계 시 직접 비교/인용 대상 1순위.
- 시드 안정성(feature/cluster selection 재현성), 누수 감사, shared/specific HI 분리 전부 없음 — 이 세 축에서는 이 프로젝트가 명확히 더 엄격함.

---

## 4. Su, Tao et al., "Data sufficiency for transferable lithium-ion battery periodical SOH estimation under resource constraints"

**저널**: *Cell Reports Physical Science* 6 (2025) 102901, Cell Press(오픈 액세스). Tsinghua/UC Berkeley 등.

### Contribution
- "얼마나 적은 초기 사이클 데이터로 새 운용조건에 전이(fine-tune)할 수 있는가"를 정량화하는 **Data Sufficiency(DS)** 지표 2종(이론값 TDS, 관측값 ODS) 제안. LSTM을 소스 도메인에서 사전학습 후 FC층만 타깃 도메인 초기 d사이클로 파인튜닝.
- 7개 데이터셋·310셀·6화학종 기준 **수명의 약 1.2~8%**(대표적으로 80사이클 근방) 데이터만으로 median MAPE&lt;1% 달성을 주장.

### 실험 결과
- 데이터셋별 DS%: THU 6.0, NASA 5.9, TJU 2.5, CALCE 1.2, MIT 8.0, XJTU 2.5, HUST 3.0. EOL 예측오차 12~42사이클.

### 내 논문에서 인용할 필요성: **낮음~중간**
- 이 프로젝트와 **완전히 다른 축**(피처 선택이 아니라 "몇 사이클이면 충분한가"라는 데이터량 축)이라 직접 비교 대상은 아니지만, "리소스 제약 하 SOH 추정"이라는 문제의식 자체는 배경 인용으로 쓸 수 있음. 누수 감사·시드 안정성·시나리오 분할·레퍼런스 노이즈 모델링 전부 없음 — 이 프로젝트가 다루는 축과 겹치지 않아 우열 비교 자체가 성립하지 않는 카테고리.

---

## 5. Gong et al., "Generalized foundation model for lithium-ion battery state-of-health prediction with distribution metric learning"

**저널**: *Journal of Energy Storage* 150 (2026) 120566, Elsevier. Xi'an Jiaotong Univ. 등.

### Contribution
- ResNet(6블록)→Transformer(4층)→KAN 백본을 10개 공개 데이터셋에 **동시(single-stage) 공동학습**, MMD(Maximum Mean Discrepancy) 손실로 소스-타깃 분포를 정렬. 입력은 원시 V/I/t 곡선을 256점으로 리샘플한 것 — 손수 설계한 피처 전혀 없음.
- "foundation model"이라 칭하지만 8.37M 파라미터·단일단계 학습으로, 저자 스스로도 "전통적 foundation model 워크플로우(사전학습→파인튜닝)와 다르다"고 인정 — 용어 사용이 다소 과장됨을 짚어둘 만함.

### 실험 결과
- 3개 leave-one-dataset-out 태스크에서 MAE 0.41~1.22%대, 단일소스/CNN/CNN-Transformer 베이스라인 대비 우수. 8.9ms 추론(CPU).

### 내 논문에서 인용할 필요성: **중간**
- 이 프로젝트가 명시적으로 반대편에 서는 철학(raw 곡선 + 블랙박스 대형 백본 vs 손수 설계 HI + 해석가능 sparse 게이팅)을 대표하는 논문이라 "왜 HI 기반 해석가능 경로를 택했는가"의 대조군으로 인용 가치가 높음. 누수 감사·피처 선택 안정성·시나리오 분할·해석가능성 검증 전부 없음(블랙박스 자체를 인정).

---

## 6. Huang et al., "iMOE: prediction of second-life battery degradation trajectory using interpretable mixture of experts"

**저널**: *Nature Communications* 17 (2026) 2549(오픈 액세스). Tsinghua SIGS/UC Berkeley/Chalmers/Stanford/PKU 컨소시엄 — 7편 중 가장 저명한 학술지.

### Contribution
- AMDP(Shazeer식 noisy top-k 게이트 MoE, E=5, top-2)로 부분충전곡선+30분 이완전압에서 뽑은 물리기반 피처 12개를 라우터 입력으로 써서 5개 "전문가"를 게이팅 → Trend 임베딩을 FORNN(LSTM)에 넣어 장기 궤적을 예측.
- **라우터가 실제로 학습되는 신경망**이라 학습·배포 시점 동작이 동일함(이 프로젝트의 oracle/hard/soft 같은 배포 갭이 원천적으로 없음) — 이 지점에서는 이 프로젝트보다 오히려 "즉시 배포가능"이라는 면에서 앞섬.
- "interpretable"의 근거는 전문가 가중치의 t-SNE 군집·SOH 기반 사후분류(91.5~96%)뿐 — 저자 스스로 "통계적 상관일 뿐 물리적 인과가 아니다"라고 명시.
- 2차 사용(재사용) 배터리 대상 — 이력 데이터가 없고 미래 운용조건이 불확실한 문제 세팅이라, 1차 수명 SOH를 다루는 이 프로젝트와 문제 자체가 다름.

### 실험 결과
- 3개 데이터셋(UL/LSD/TPSL) 295셀 84,213사이클. 평균 MAPE 0.95%, 0.43ms 추론 — PatchTST/Informer보다 빠르고 정확.

### 내 논문에서 인용할 필요성: **높음**
- 이 프로젝트의 "시나리오별 게이트"와 구조적으로 가장 가까운 비교대상(둘 다 라우팅+전문가/게이트 조합) — 다만 iMOE는 라우터가 실제 학습되는 연속 MoE라 배포 갭이 없고, 이 프로젝트는 6개 시나리오를 물리적 근거(충방전×전압대)로 미리 정의한 뒤 formula 라벨(oracle)과 학습된 분류기(hard/soft)를 명시적으로 분리·정량화한다는 점이 다름 — "배포 갭이 아예 없는 설계" vs "배포 갭을 인지하고 그 크기를 정직하게 측정하는 설계"로 대비해서 인용 가능. 시드 안정성/누수 감사/레퍼런스 노이즈 모델링은 iMOE도 없음.

---

## 7. Shi et al., "Multistage state of health estimation for lithium-ion battery based on partial charging curves and an optimized heterogeneous stacking model"

**저널**: *Journal of Energy Storage* 141 (2026) 119420, Elsevier. Hunan Univ.

### Contribution
- "Multistage"는 시나리오 그리드가 아니라 **이진 전압 임계값(3.9V) 분기**일 뿐 — 시작전압&lt;3.9V면 IC피크+충전시간+유사도 피처(IMKSVREL2), ≥3.9V(IC피크 관측 불가)면 IC피크를 뺀 나머지만 쓰는(IMKSVREL1) 규칙 기반 스위치.
- 4종 커널(linear/poly/RBF/sigmoid) SVR을 스태킹하고 선형회귀로 메타학습, HHO(Harris Hawks Optimization)로 하이퍼파라미터 자동탐색.
- 피처는 단 6개(IC피크 값/위치, CC/CV 충전시간, 초기사이클 대비 DTW/MMD거리) — 상관계수(PCC) 필터링만, 학습된 sparse 선택 없음.

### 실험 결과
- CALCE 4셀 + NASA 4셀. &lt;3.9V 구간 RMSE 0.5~1.7%, 앙상블 베이스라인 대비 RMSE 54~73% 개선.

### 내 논문에서 인용할 필요성: **낮음**
- "multistage"라는 이름과 달리 시나리오 개념이 극히 단순(이진 스위치, 충전측만)해서 목적2와 직접 비교하기엔 체급 차이가 큼. 다만 "전압 임계값에 따라 쓸 수 있는 피처가 달라진다"는 문제의식과 DTW 기반 유사도 피처는 이 프로젝트의 MORPH 카테고리·q_frac 설계 정당화에 참고할 만함. 시드 안정성·누수 감사·레퍼런스 노이즈 전부 없음.

---

## 8. HI 정의 커버리지 분석 — 리뷰 논문의 taxonomy 대비 이 프로젝트의 4개 카테고리

리뷰 논문(1번)의 Indirect HI taxonomy(Table 3–7, Fig. 2)를 기준으로, 이 프로젝트의 STAT/DIFF/LFP/MORPH(`5_model/utils/hi_schema.py`)가 각각 얼마나 겹치는지 대조함.

| 리뷰의 Indirect HI 하위 카테고리 | 리뷰가 예시로 든 것들 | 이 프로젝트 대응 | 커버 여부 |
|---|---|---|---|
| **Incremental Capacity (IC)** | 피크 높이/위치/면적, IC 곡선 형태 | `DIFF_KEYS`: dqdv_peak_h/v/w/area, dvdq_*, d2vdq2_rms 등 20개 | ✅ **잘 커버됨** — 사실상 DIFF 카테고리 전체가 이 항목의 확장 |
| **Statistical Measures** | 평균/분산/표준편차/왜도/첨도 등 | `STAT_KEYS`: v_mean/std/skew/kurt/ent/iqr/range/percentile 등 20개 | ✅ **잘 커버됨** — STAT 카테고리 전체가 이 항목과 거의 1:1 |
| **Battery Voltage**(OCV, DTV, Q-V matrix, pDV, MVF, 등전압충전량) | 개회로전압, 특정 전압에서의 충전량 등 | 없음 | ❌ **미커버** — 이 프로젝트는 절대 전압 임계값 기반 피처를 안 씀(세그먼트 상대좌표 q_frac 기준이라 절대 전압 임계값 개념 자체가 없음) |
| **Battery Current**(등시간 전류강하, Current Rate) | — | 일부만(`i_mean`, `i_std`, `i_q_slope`는 STAT에 있지만 리뷰가 말하는 "등시간/등전압 구간 전류강하량" 같은 정의는 없음) | 🟡 **부분 커버** |
| **Charging Time**(CCCT, 등전압 충전시간 구간 등 8종) | 완충시간, CC/CV 충전시간 등 | 없음 | ❌ **미커버** — 이 프로젝트는 시간 축이 아니라 q_frac(용량 비율) 축으로 세그먼트를 정의해서, "몇 초 걸렸는가" 개념 자체가 설계에서 배제됨 |
| **Discharging Time**(등전압강하 방전시간 등) | — | 없음 | ❌ **미커버**(위와 동일 이유) |
| **Battery Temperature** | 온도 변화/최대온도 도달시간 | 없음 | ❌ **미커버** — 이 프로젝트 데이터셋에 온도 채널 자체가 없음 |

**리뷰에 없는데 이 프로젝트에만 있는 것**:
- **LFP 카테고리**(plateau_frac, plateau_v_*, inflect_v/q_frac, ica_peak_cnt 등 20개) — 리뷰는 "voltage plateau"를 Table 4의 Charge-discharge voltage 설명 중에 딱 한 문장으로만 언급("Analyzing features of voltage plateaus... provide valuable insights")할 뿐, 별도 HI 타입으로 formalize하지 않음. LFP의 2상(two-phase) 플래토 거동에 특화된 이 카테고리는 리뷰의 taxonomy보다 더 세밀하다.
- **MORPH 카테고리**(vt_dtw, vq_dtw, ve_dtw, vt_frec, vq_frec, ve_frec — DTW/Fréchet 곡선유사도) — 리뷰의 7개 Indirect HI 하위군 어디에도 곡선 형태 유사도(DTW/Fréchet) 개념이 없음. 완전히 리뷰 taxonomy 밖의 피처군.

### 요약
- 이 프로젝트의 4개 카테고리 중 **STAT/DIFF는 리뷰의 taxonomy와 잘 정렬**된다(Statistical Measures, Incremental Capacity에 각각 대응).
- **LFP/MORPH는 리뷰의 taxonomy에 없는 독자적 카테고리**다 — LFP는 리뷰가 지나가듯 언급만 한 개념을 정식 카테고리로 승격시킨 것이고, MORPH는 리뷰에 아예 없는 새로운 축이다. 이건 논문에서 "우리 HI 설계가 기존 리뷰의 분류를 얼마나 따르고 얼마나 확장하는가"를 설명할 때 강점으로 쓸 수 있는 지점이다.
- 반대로 **리뷰가 강조하는 Time-based/전압임계값 기반 HI(Charging/Discharging Time, Battery Voltage 하위군 대부분)는 이 프로젝트에 전혀 없다** — q_frac 상대좌표 기반 세그먼트 설계의 자연스러운 귀결이지만, 이건 논문에서 "왜 시간 기반 HI를 배제했는가"에 대한 설명(또는 한계 인정)이 필요한 지점이기도 하다.
