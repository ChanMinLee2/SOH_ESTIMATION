Phase 1 파이프라인은 세그먼트 경계 계산에 쓰이는 SOC 레퍼런스 용량값에 인위적
노이즈를 주입해, 실제 BMS가 겪는 불완전한 용량 추정 상황을 재현한다. 현재 채택된
설정(noise_mode=ou, noise_amp=3%, offset_amp=5mA, ref_lag=1)의 근거는 다음과 같다.

**노이즈 방식(OU)**: 사인파 등 결정론적 드리프트 대신 Ornstein-Uhlenbeck(평균회귀
랜덤워크)을 채택했다. 실제 센서 드리프트는 고정 주기로 오르내리지 않고 불규칙하게
표류하다 서서히 평균으로 회귀하는 패턴을 보이므로, 물리적 근거가 더 명확하다.

**noise_amp=3%**: 이 값은 편측 클리핑 상한이며, 정상상태 실효 오차(1σ)는
noise_amp/4로 약 0.75%다. Kampik et al.(2025, Energies 18, 2850)은 상용 BMS
제조사가 광고하는 ±1% 정확도가 "신품·이상조건·수명 초기에서만" 성립하며
실사용·노화 조건에서는 이보다 나쁘다고 지적한다. 셀 수명 전체(BOL~EOL)를 다루는
본 프로젝트 맥락에서는 이 실사용 기준에 맞추는 게 타당하며, 3%는 과도하게
낙관적이지도 비관적이지도 않은 적정 수준이다.

**offset_amp=5mA**: Movassagh et al.(2021, Energies 14, 4074)은 쿨롱카운팅
오차를 전류측정(gain)오차·적분근사오차·용량불확실성·타이밍오차 4가지로 formal
분해했다. 이 중 gain형(비례오차)은 적분 길이와 무관하게 동일 비율로 나타나 기존
bias+drift(OU) 곱셈형 노이즈와 수학적으로 동치이므로 추가 구현이 불필요했다.
반면 offset형(전류 크기와 무관한 고정오차)은 사이클 소요시간에 비례하는
절대오차를 만들어 곱셈형 모델로는 표현 불가능했다 — 이를 새 항으로 추가했다.
5mA는 Movassagh et al. Table II의 예시값(1.5Ah 셀 기준 10mA)을 본 데이터셋의
평균 셀 용량(약 1Ah)에 비례해 조정한 값이다. 타이밍오차는 같은 논문에서 무시
가능(69×10⁻⁶) 수준으로 확인돼 모델링에서 제외했고, 용량불확실성은 기존
bias+drift+ref_lag 메커니즘 전체가 이미 대표하는 개념이라 별도 항을 두지
않았다.

**ref_lag=1**: ref_lag=0(그 사이클 자신의 실측값을 레퍼런스로 삼는 것)은
라벨과 사실상 동일한 값을 참조해 리뷰어에게 누수로 오인될 소지가 있다. lag=1은
이 동일-사이클 참조를 기계적으로 끊으면서도, 실측 데이터상 인접 사이클 간 용량
변화가 무시할 수준(중앙값 0.05%대)이라 노이즈 통계 자체에는 실질적 영향이 없다
— 안전장치 성격의 최소 변경이다. lag를 더 키우지 않은 이유는, 그 변화가 실질적
효과를 내려면 수십 사이클 이상이 필요한데(같은 실측 기준 lag=50에서도 0.4%대에
불과), 그 정도 지연은 별도 축인 `calibration_period`(재보정 주기)가 이미 더
적절한 방식으로 담당하는 개념과 겹쳐 혼란을 준다고 판단했기 때문이다 — 이번
설정에는 calibration을 켜지 않았다.

**보강 근거**: Ng et al.(2009, Applied Energy 86, 1506–1511)은 쿨롱효율 보정을
더한 enhanced coulomb counting을 제안한 고전 논문으로, "단순 곱셈형 오차만으로는
불충분하며 보정항이 필요하다"는 방향성 자체를 독립적으로 뒷받침한다(MDPI 계열이
아닌 Elsevier 저널이라는 점에서 위 두 인용과 출판사가 겹치지 않는 근거이기도
하다).

**실측과의 정합성**: 이 설정(offset_amp=5mA 추가)을 기존(offset 없음) 대비
실제로 측정해보면 셀 평균 노이즈가 mean +0.42%p→+0.43%p, absmax 2.32%p→2.37%p
수준으로 거의 변하지 않는다. 이는 "현실성을 높였더니 결과가 바뀌었다"가 아니라
"현실성을 높여도 결과가 안정적"이라는 뜻이며, 노이즈 진폭·재보정 주기를 바꿔도
`q_abs`류 오차가 거의 안 움직인다는 기존 관측(세그먼트 이산화가 오차의 주원인)과
같은 방향의 결과다. 즉 본 설정은 성능 개선이 아니라 face-validity(모델링 근거의
방어 가능성) 확보를 목적으로 채택되었다.

---

**인용 논문**

| 논문 | 인용 목적 | 저널 | 연도 |
|---|---|---|---|
| Movassagh, K., Raihan, A., Balasingam, B., Pattipati, K. — "A Critical Look at Coulomb Counting Approach for State of Charge Estimation in Batteries" | 쿨롱카운팅 오차의 gain/offset formal 분해 — `offset_amp` 항 도입의 직접 근거 | Energies (14, 4074) | 2021 |
| Kampik, M., Fice, M., Sztymelski, K., Oliwa, W., Wieczorek, G. — "Examples of Problems with Estimating the State of Charge of Batteries for Micro Energy Systems" | 상용 BMS 광고 정확도(±1%)가 이상조건(신품·BOL) 한정임을 지적 — `noise_amp=3%` 적정성(과도한 낙관 아님) 근거 | Energies (18(11), 2850) | 2025 |
| Ng, K.S., Moo, C.-S., Chen, Y.-P., Hsieh, Y.-C. — "Enhanced coulomb counting method for estimating state-of-charge and state-of-health of lithium-ion batteries" | 곱셈형 오차 단독으로는 불충분하다는 방향성의 보강 근거(비-MDPI, 위 두 인용과 출판사 분산) | Applied Energy (86(9), 1506–1511) | 2009 |
| Texas Instruments — "Impedance Track™ Based Fuel Gauging"(백서, SLPY002) / BQ34Z100 데이터시트 | 상용 쿨롱카운팅 IC(Impedance Track)가 "다양한 동작조건에서 최대 1% 오차"를 표방한다는 업계 스펙 근거 — 학술논문 아닌 벤더 공식문서로 별도 구분 | TI 공식 문서(비-저널) | 미상(제품 문서) |
