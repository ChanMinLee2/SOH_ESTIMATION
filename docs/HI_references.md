LFP 배터리 HI(Health Indicator) 추출 방법론의 4대 계통도 및 핵심 논문

LFP 배터리의 평탄한 전압 특성과 '부분 충방전(Partial Cycle)' 환경을 극복하기 위해 학계에서 고안한 HI 추출 방법론은 크게 4가지 갈래로 나뉩니다. 본인의 285개 HI가 어디에 속하는지 매핑하며 논문에 인용하시기 바랍니다.

1. 미분 기법 계통 (Differential Analysis: IC / DVA)

LFP 배터리의 평탄부(Flat Plateau)를 피크(Peak)로 변환하여 상전이(Phase Transition)의 이동을 추적하는, LFP SOH 연구의 가장 정통적이고 강력한 갈래입니다.

[LFP DVA의 교과서] * 논문명: State of health estimation algorithm of LiFePO4 battery packs based on differential voltage curves for battery management system application (Energy, 2016 - M. Berecibar)

핵심 기여: LFP 배터리 팩 단위에서 DVA(dV/dQ) 곡선의 피크 간 거리(Distance between peaks)와 면적이 SOH와 어떻게 완벽하게 선형적으로 비례하는지를 실험적으로 증명한 전설적인 논문입니다. (인용수 500회 이상)

[부분 충전 IC 분석의 정립]

논문명: Online capacity estimation of lithium-ion batteries based on novel feature extraction and adaptive importance sampling (Journal of Power Sources, 2018 - X. Zheng)

핵심 기여: 전체 충전 곡선이 아닌, 아주 짧은 '부분 정전류(Partial CC) 충전 구간'에서 추출한 불완전한 IC 곡선만으로도 메인 피크의 위치(Position)와 높이(Height)를 기반으로 우수한 HI를 만들 수 있음을 증명했습니다.

2. 기하학적 & 형태학적 특징 계통 (Geometric & Morphological)

전압/전류 곡선을 하나의 '이미지'나 '궤적(Trajectory)'으로 보고, 초기(BOL) 사이클 곡선과의 형태학적 거리나 왜곡 정도를 수학적으로 측정하는 갈래입니다. 질문자님의 DTW distance, Fréchet distance 아이디어가 여기에 속합니다.

[형태학적 융합 HI의 최고봉]

논문명: Battery Health Prediction Using Fusion-Based Feature Selection and Machine Learning (IEEE Trans. on Transportation Electrification, 2021 - X. Hu)

핵심 기여: 질문자님께서 이미 보유하고 계신 논문으로, V-Q, V-T 곡선 등에서 면적, 기울기, 피크 등 수십 개의 기하학적 형태(Morphological) 피처를 추출한 뒤 최적의 Subset을 골라내는 프레임워크의 교과서입니다.

[시계열 궤도 매칭 (DTW 활용)]

논문명: Lithium-ion battery state of health estimation based on dynamic time warping and Gaussian process regression (Journal of Energy Storage, 2020)

핵심 기여: 전압 곡선이 노화됨에 따라 x축(시간/용량)과 y축(전압)으로 모두 틀어지는 현상을 동적 시간 워핑(DTW, Dynamic Time Warping)을 통해 하나의 스칼라 거리값(HI)으로 완벽하게 압축해 냈습니다.

3. 시간 및 에너지 기반 직관적 지표 계통 (Time / Energy Tracking)

복잡한 미분이나 곡선 피팅(Fitting) 없이, BMS 마이크로컨트롤러(MCU)에서 실시간으로 연산하기 가장 가벼운 스칼라 값들을 추출하는 갈래입니다.

[특정 전압 구간 충전 시간 (Ref-based Time)]

논문명: A Data-Driven State-of-Health Estimation Model for Lithium-Ion Batteries Using Referenced-Based Charging Time (IEEE Trans. on Power Delivery, 2023)

핵심 기여: 충전이 시작되는 전압은 매번 다르지만, '특정 기준 전압 $V_1$에서 $V_2$까지 도달하는 데 걸린 시간(Time interval $\Delta t$)'은 노화될수록 급격히 짧아진다는 직관적인 사실을 수학적 HI로 규격화했습니다. 부분 충방전에 가장 강건한 HI 추출법 중 하나입니다.

[전압 곡선 하부 면적 (Area Under Curve)]

논문명: State of health estimation for lithium-ion battery based on energy features and Gaussian process regression (Journal of Energy Storage, 2022)

핵심 기여: 질문자님의 설계안 중 energy_dis나 v_auc_time과 정확히 일치하는 접근입니다. V-Q 곡선의 적분값(에너지) 자체가 훌륭한 복합 노화 지표임을 증명합니다.

4. 딥러닝 기반 잠재 특성 자동 추출 (Deep Latent Feature Extraction)

사람이 수식을 짜서(Hand-crafted) HI를 추출하는 것을 넘어, CNN이나 Autoencoder가 원본 시계열(V, I, T)에서 직접 보이지 않는 특성(Latent Vector)을 뽑아내게 하는 최신 트렌드입니다.

[부분 충전 데이터 + 딥러닝 융합]

논문명: Flexible battery state of health and state of charge estimation using partial charging data and deep learning (Energy Storage Materials, 2022 - J. Tian)

핵심 기여: 시작점과 끝점이 랜덤한 부분 충전 시계열(V, I, T) 데이터를 컨볼루션 신경망(CNN) 기반의 인코더에 넣어 유연하게 HI(Latent features)를 자동 추출하는 프레임워크를 제시했습니다. HUST 데이터셋과 같은 복잡한 프로토콜 분석에 필수적인 참고 문헌입니다.

💡 질문자님의 논문 작성 시 시사점

질문자님께서 새롭게 설계하신 285개의 HI Pool은 위 4가지 계통의 최상위 방법론들을 하나로 합친 '종합선물세트(Comprehensive Feature Engineering Pool)'입니다.

논문 작성 시 *"단일 계통(예: IC 피크만 쓰거나 Time만 쓰는 것)의 기존 연구들은 특정 부분 충전 시나리오(예: High SOC vs Low SOC 구간)에서 취약성을 보일 수밖에 없다. 따라서 우리는 위의 1~3번 계통을 모두 아우르는 거대한 Feature Pool을 생성하고, 동적인 부분 충전 시나리오에 맞춰 최적의 갈래(Subset)를 동적으로 선택하는 프레임워크를 제안한다"*라고 논리를 전개하시면 완벽한 서론(Introduction)이 완성됩니다.