# 260701

* 1. 수식 오류(완벽하게 - 내 기준, 아래 과정은 md파일에 계속 반영하기, 다되면 말씀드리고 넘어가기)
    - 1-1. 각 HI들의 상관관계 방향/수치정도를 예측해달라고 리스트로 작성
      - 이 리스트를 만들 때, 검증용으로 실제 HI 몇 개 셀 정도(가장 깔끔한거 4개를 내가 선정)를 만들어서 보라고 프롬프팅
      - b1c9, b3c13, 1-1, 7-5 
  
    - 1-2. 리스트를 실제 trend와 비교해서 예상과 너무 다른 케이스 찾기 
    - 1-3. HI별 의미에 맞게 수식이 작성되었는 지

* 2. 수식 수정(우선순위 낮음)
    - HI 이상치(수염, 범위 밖 값) 제거 -> 포인트(+너무 긴 수염)만 심한 거 있으면 제거하는 방향으로
    - 수식은 ok, 실제로 돌려보니까 그림이 이상하면 수식을 수정

* 3. 최종 전처리 데이터 저장(형태 : (1)셀별 데이터 (2)세그먼트별 데이터)
(1) 셀별 데이터 : 각 row하나(전압/전류/시간값만)마다 어떤 셀/사이클/세그먼트에 해당하는 지 ID 메타 데이터 추가로 달기
-> [V, I, t, cid, cycle_id, segment_id] 
(2) 세그먼트별 데이터 : 각 unique한 세그먼트 id별로 HI 20개 + 메타 데이터(셀/사이클) 추가로 달기
-> [segment_id, HI_1, … , HI_20, cid, cycle_id]

===============
2_preprocess 결과(_2_data_clean)로 아래 형태가 나오도록 수정해줘

1. 셀별 데이터 
각 row하나(전압/전류/시간값만)마다 어떤 셀/사이클/세그먼트에 해당하는 지 ID 메타 데이터 추가로 달기
data_clean에서 chg_gap_seg, phase는 빼고, segment_id 붙임
이 때 segment_id는 사이클마다 초기화되고 0부터 시작해야 해
[cell_id, cycle, segment_id, time_s,voltage_V,current_A,capacity_Ah]

3_integrity 과정도 위 수정사항을 반영할 수 있도록 수정


_4_data_hi/ 경로에 저장되는 셀별 HI 데이터들을 아래와 같은 두 가지 형식으로 저장하도록 수정하고싶어. 데이터들은 모두 pkl 파일로 생성하고 각 sample 배터리셋 종류별 한 cycle(ex-mit_hi_cycle1.csv)만 csv로 저장. 각 HI를 칼럼으로 지정할때는 지금처럼 feature 명으로 네이밍되게 해.

1. 사이클별 데이터 (글로벌만)
   [cell_id, cycle, capacity_Ah, global HI 15개]

2. 세그먼트별 데이터 (글로벌 외 전부)
각 unique한 세그먼트 id별로 HI 20개 + 메타 데이터(셀/사이클) 추가로 달기
capacity_Ah : segment 내의 구간 capacity
scen 칼럼 정의는 아래와 같아
-3 : discharge - high
-2 : discharge - mid
-1 : discahrge - low
1 : charge - low
2 : charge - mid 
3 : charge - high
-> [cell_id, cycle, segment_id, capacity_Ah, scen, HI_1, … , HI_20]
