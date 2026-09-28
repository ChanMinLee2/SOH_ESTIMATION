# SOH_ESTIMATION — 프레임워크 개요

**작성 기준**: 2026-09-27 시점의 실제 코드(`1_convert/`~`9_eval/`, `common/scenario/`,
`model_lib/`, `run_pipeline.py`, `parameters.py`). 이전 버전 문서(2026-08-13 기준,
`5_model/` 등 옛 폴더 구조를 전제로 함)는 이번 대규모 리팩토링(폴더 번호 재정렬,
`model_lib/` 공유 인프라 분리, CNN/UQ/SCRTrainer 등 죽은 코드 제거)으로 완전히
달라졌으므로 전면 폐기하고 이 코드 상태만 근거로 다시 썼다. 코드를 직접 몰라도
따라올 수 있도록 각 단계의 입력·처리·출력을 구체적인 컬럼명/수식/파일 경로로 적는다.
같은 내용을 그림으로 요약한 인포그래픽이 함께 제공된다.

---

## 1. 이 프레임워크가 하는 일

LFP(리튬인산철) 배터리의 **State of Health(SOH = 현재 용량 / 초기 용량)** 를,
**충방전 사이클 전체가 아니라 그 일부 구간(세그먼트)만 보고** 추정한다. 세 가지
설계 목표를 유지한다.

1. **부분 사이클 사용** — 완주하지 않은 세그먼트만으로 SOH를 추정한다. 실제 BMS가
   항상 완전한 충방전 곡선을 관측할 수 있는 건 아니라는 전제.
2. **시나리오별 최적 특징 자동 탐색** — 충전/방전 × SOC 위치(존)마다 유용한
   HI(Health Indicator) 조합이 다를 수 있다는 전제 아래, 이를 사람이 정하지 않고
   데이터로부터 게이트가 자동으로 고른다.
3. **미지 세그먼트의 시나리오 추정** — 실배포 시 "이 세그먼트가 어느 시나리오(존)인지"
   정답 라벨 없이 모델 스스로 판단해야 한다(hard/soft 라우팅, §11).

MIT(FastCharge, 123셀)와 HUST(77셀), 두 공개 데이터셋을 하나로 합쳐 학습하고,
**셀 단위**로 train : val : test = 6 : 2 : 2 (`split_seed` 고정)로 나눠 평가한다 —
같은 셀의 사이클이 train과 test에 동시에 들어가는 일이 없다.

---

## 2. 전체 파이프라인 — 단계별 한눈에 보기

`run_pipeline.py`가 Step 1~9를 순서대로 실행한다(`python run_pipeline.py <from> --to-step <to>`).

| Step | 폴더/스크립트 | 무엇을 하는가 | 핵심 산출물 |
|---|---|---|---|
| 1 | `1_convert/convert_unified.py` | 원본 실험 로그를 공통 스키마로 변환 | `_1_data_unified/{MIT,HUST}/*.pkl` |
| 2 | `2_preprocess/preprocess.py` | 7단계 이상치·오염 필터 | `_4_data_hi/clean/{dataset}/*.pkl` |
| 3 | `3_integrity/check_integrity.py` | 스키마·품질 검증(데이터 변형 없음) | `integrity_report.csv` |
| 4 | `4_hi_analysis/hi_correlation.py` | 사이클 → 6개 시나리오 세그먼트 분할 + HI 66(64)개 계산 | `_4_data_hi/{axis}/seg,cycle/{dataset}/*.pkl` |
| 5 | `5_interaction/interaction.py` | HI별 "시나리오 공유 vs 전용" 판정 | `hi_scenario_interaction_*.json` |
| 6 | `6_synergy/synergy.py` | 시나리오별 HI 그룹화(중복 배제 + 시너지) | `synergy_groups_*.json` |
| 7 | `7_kernel/kernel.py` | 그룹 → 비선형 융합 "커널 HI" 생성 | `kernel_group_features_*.pkl` (+ redundancy json) |
| 8 | `8_train/train.py` | SCR 모델(게이트 라우팅) 학습 | `checkpoints/best_by_saturation.pt`, `config.yaml` |
| 9 | `9_eval/test.py` | oracle/hard/soft 평가 + 플롯 | `metrics/metrics.json`, `predictions/*.csv` |

Step 5~7의 산출물은 파일 경로로 Step 8에 전달될 뿐, 서로 강하게 결합되어 있지
않다 — 예를 들어 `python run_pipeline.py 8`만 실행하면 이미 만들어둔 고정 경로의
산출물을 그대로 재사용해 학습만 다시 돌릴 수 있다.

---

## 3. Step 1 — 데이터 변환 (`1_convert/convert_unified.py`)

**입력**
- MIT: `_0_data_raw/FastCharge/*.mat` — HDF5 포맷 MATLAB 파일 3개(batch1/2/3).
  `h5py`로 직접 파싱하며 `batch/summary`, `batch/cycles`, `batch/cycle_life`,
  `batch/policy_readable` 그룹을 읽는다.
- HUST: `_0_data_raw/our_data/our_data/{cell_id}.pkl` — 이미 pkl로 정리된 원본.
  내부에 `data[cyc]` DataFrame(컬럼 `Time (s)`, `Voltage (V)`, `Current (mA)`, `Status`).

**처리**
- 사이클 0(출고 진단 사이클) 제외.
- 전류 부호로 phase 판정: `current_A > 0.01` → charge, `< -0.01` → discharge, 그 외 rest.
- MIT는 Batch2의 연속 셀 5쌍을 Batch1과 사이클 번호로 이어붙이고(`CONTINUING`),
  전압 오염 등으로 완전 불량인 셀 11개(`DELETE_CELLS`)를 제외.
- HUST는 전류(mA→A, ÷1000), 용량(dq, mAh→Ah, ÷1000) 단위 변환. temperature는
  측정값이 없어 고정 30.0°C.
- 공통 후처리: 활성 행 5개 미만 사이클 제거, 사이클 내 시간 역행 보정, rest 중
  전류 0A인 행 제거, rolling median(window=11, sigma=2.5) 기준 방전용량 이상
  사이클 제거.

**출력**: `_1_data_unified/{MIT,HUST}/{cell_id}.pkl` — `{"meta": {...}, "cycles": DataFrame}`.
DataFrame 컬럼: `cycle, time_s, voltage_V, current_A, capacity_Ah, phase`
(phase ∈ {charge, discharge, rest}). meta에는 `cell_id, dataset, batch,
charge_policy, n_cycles, init_cap_Ah, final_cap_Ah` 등.

---

## 4. Step 2 — 전처리 (`2_preprocess/preprocess.py`)

**입력**: `_1_data_unified/{dataset}/*.pkl`.

**처리**: 고정 순서의 7단계 필터.

1. 빈 사이클 제거(활성 행 < 5, 또는 discharge 구간 자체가 없음)
2. `time_s` 단조 보정
3. rest 구간의 0전류 행 제거
4. 방전 구간 단절 사이클 제거(간격 > 600s 또는 중앙값×50배) — 충전 CC→구간 전환
   갭(> 120s 또는 중앙값×30배)은 사이클 전체가 아니라 "행 단위"로 `chg_gap_seg`
   플래그만 남긴다
5. rolling median 2-pass 이상 사이클 제거(window=11/sigma=2.0, window=31/sigma=2.0)
6. 방전 종지전압 하한 필터(`vend_min=1.8V` 미만 사이클 제거)
7. V–q_frac 형상 이상치 제거 — 방전/충전 각각 q_frac 격자로 보간한 곡선을 그
   셀의 rolling median 기준곡선과 비교, RMSE/최대편차의 robust z-score가
   sigma=30.0을 넘으면 제거

**출력**: `_4_data_hi/clean/{dataset}/*.pkl` (`--skip-shape` 옵션 시 `clean_noshape/`).
컬럼: `cell_id, cycle, segment_id, time_s, voltage_V, current_A, capacity_Ah, chg_gap_seg`
— `segment_id`는 사이클 내 phase 연속 구간 번호(0부터, 사이클마다 리셋)로 이 단계에서
새로 부여된다. 저장 루트는 `data_directories.py`의 `DATA_4_HI_ROOT`.

---

## 5. Step 3 — 무결성 검사 (`3_integrity/check_integrity.py`)

**입력**: `_4_data_hi/clean/{MIT,HUST,TJU,CALCE}/*.pkl`.

**처리(데이터를 바꾸지 않고 확인만)**: pkl 로드 가능 여부, 스키마 컬럼 존재,
`cell_id` 일치, `segment_id`가 사이클마다 0에서 시작하고 비감소인지, 용량 추세
이상(WARN), 컬럼별 NaN 비율(WARN), `meta.n_cycles`와 실제 개수 불일치, 전압 범위
이상(1.5V 미만/4.5V 초과, WARN), 시간 단조 위반(WARN).

**출력**: `integrity_report.csv`(셀별 요약), `integrity_issues.csv`(ERROR/WARN 목록).

---

## 6. Step 4 — 시나리오 분할 & HI 추출 (`4_hi_analysis/`)

이 단계가 전체 파이프라인에서 가장 중요한 특징공학 단계다. 사이클 하나를 여러
개의 "세그먼트"로 자르고, 세그먼트마다 66개(기본)의 손수 설계한 특징을 계산한다.

### 6.1 시나리오 축 — `common/scenario/q_frac_ref.py`

각 사이클의 충전 구간과 방전 구간을 각각 **q_frac**(그 구간의 누적 전하량이
전체 대비 차지하는 비율, 0~1)을 기준으로 3개 존(zone)으로 나눈다.

- `hi` 존: q_frac ∈ (0, n1) — 구간 시작 근처
- `mid` 존: q_frac ∈ (0.5-n1/2, 0.5+n1/2) — 구간 중앙
- `lo` 존: q_frac ∈ (1-n1, 1) — 구간 끝 근처

여기에 방향(충전/방전)을 곱해 **6개 시나리오**가 나온다:
`chg_lo, chg_mid, chg_hi, dis_hi, dis_mid, dis_lo`. 존 하나 안에서 길이 `n2`
(q_frac 비율)짜리 세그먼트를 `n_samples`개 균등 배치로 뽑는다. 정식 학습에
쓰는 값(canonical recipe)은 `n1=0.35, n2=0.20, n_samples=2`.

**왜 "q_frac_ref"인가 — 라벨 누수(leakage) 회피**: q_frac의 분모(그 구간의
100%가 몇 Ah인지)를 "그 사이클 자신의 실제 완주 용량"으로 두면, 그 값 자체가
사실상 SOH의 다른 표현이라 세그먼트를 자르는 시점에 이미 정답을 흘리게 된다.
대신 `q_frac_ref`는 분모를 **그 (셀, 방향)의 과거 사이클 값 + 작은 드리프트
노이즈**(`ref_lag`만큼 과거, OU 프로세스 노이즈, 진폭 `noise_amp`)로 대체해
"현재 진짜 용량"과의 직결을 끊는다. 정식 레시피는
`ref_lag=1, noise_amp=0.03, noise_mode="ou", noise_period_cycles=200, tile_scope="zone"`.

### 6.2 HI(Health Indicator) — 4개 카테고리, 66개(기본)

세그먼트 하나(원시 V/I/t 배열)마다 `4_hi_analysis/hi_correlation.py`가 아래
4개 카테고리 특징을 계산한다(`model_lib/utils/hi_schema.py`의 `STAT_KEYS` /
`DIFF_KEYS` / `LFP_KEYS` / `MORPH_KEYS`).

| 카테고리 | 개수 | 예시 | 의미 |
|---|---|---|---|
| **STAT** | 20 | `v_mean_cw`, `v_std`, `v_skew`, `v_kurt`, `q_abs`, `energy_seg`, `v_samp_ent` | 전압/전류의 통계량(평균, 분산, 왜도/첨도, 샘플엔트로피 등) |
| **DIFF** | 20 | `dvdq_mean`, `dqdv_peak_h/v/w`, `dqdv_valley_h/v`, `dv_di_seg` | dV/dQ, dQ/dV(ICA) 피크 높이·전압·폭 등 미분 기반 특징 |
| **LFP** | 20 | `plateau_frac`, `nonlin_idx`, `inflect_v`, `inflect_q_frac`, `ica_peak_cnt` | LFP 특유의 완만한 전압 플래토(|dV/dQ| < 0.25 구간)의 형태 |
| **MORPH** | 6 | `vt_dtw`, `vq_dtw`, `ve_dtw`, `vt_frec`, `vq_frec`, `ve_frec` | 그 셀의 BOL(첫 유효 세그먼트) 곡선과의 DTW/이산 Fréchet 형태 거리(V–t, V–Q, V–Energy) |

기본 66개(20+20+20+6)이지만, `stat_q_abs`/`stat_energy_seg`는 (분모 설계에
따라) 용량과 거의 직결돼 leakage 우려가 있어 `SOH_EXCLUDE_STAT_LEAK=1`이면
64개로, 추가로 `SOH_EXCLUDE_DQDV_LEAK=1`이면 `diff_dqdv_area`도 빠져 63개가
된다. **이 프로젝트의 정식 검증 레시피는 64개**(`SOH_EXCLUDE_STAT_LEAK=1`)를
쓴다 — `q_frac_ref`는 분모가 과거+노이즈 참조값이라 `q_abs`의 leakage가
완화되지만, 보수적으로 계속 제외한다.

이 두 환경변수는 **프로세스 시작 전에 지정해야** 한다 — `N_HI`가 모델 여러
모듈에서 모듈 임포트 시점 상수로 신경망 크기를 정하는 데 쓰이기 때문이다.

**출력**: 세그먼트 1개 = 1행. `_4_data_hi/{axis}/seg/{dataset}/{cell_id}.pkl`
(Step 5 이후가 실제로 읽는 파일)의 컬럼은 `dataset, cell_id, cycle,
capacity_Ah, segment_id, seg_name(chg_lo 등), scen, zone, q_frac_lo/hi` +
`stat_{key}, diff_{key}, lfp_{key}, morph_{key}`(원래 이름, 세그 접미사 없음).
사이클당 1행짜리 Global HI(진단·시각화 전용, 15개)는 `_4_data_hi/{axis}/cycle/`에
따로 저장된다.

> **주의**: `hi_00`..`hi_63`(또는 65)이라는 익명 컬럼명, `scen_idx`/`direction`/
> `level`은 이 원본 pkl에는 없다. 이 이름들은 로더인
> `model_lib/datasets/segment_dataset.py`가 원본 이름을 고정 순서
> (stat→diff→lfp→morph)로 매핑해 만드는 것이다(§10.1).

---

## 7. Step 5 — Interaction test (`5_interaction/interaction.py`)

**목적**: raw HI 개념 하나하나가 "6개 시나리오 어디서나 비슷하게 유용한 특징"인지,
"시나리오마다 유용성이 크게 다른 특징"인지 판정해서, 모델의 게이트 구조(공유
게이트 vs 시나리오별 게이트, §10.2)를 미리 정한다.

**처리**: (train split만 사용, val/test 누수 방지)
1. HI 개념마다 6개 시나리오 각각에서 SOH와의 상관계수 r을 구한다.
2. 6개 중 2개씩 짝지은 15쌍(6C2) 전부에 Fisher z-변환 상관계수 동일성 검정(귀무가설:
   두 시나리오의 r이 같다)을 수행하고, 15쌍 중 최소 p-value를 그 HI의 "상호작용
   증거"로 삼아 Benjamini–Hochberg로 다중비교 보정한다.
3. 표본이 매우 커서(수십만~백만 행) p-value만으로는 변별력이 없어, **최종 판정은
   시나리오 간 상관계수의 표준편차가 임계값(`min_effect_size`, 기본 0.1) 이상인가**로
   내린다.
4. 임계값을 넘으면(`significant=True`) 그 HI는 시나리오별 게이트(`scen_gates`)로,
   못 넘으면 시나리오 공유 게이트(`shared_gate`)로 분류된다.

**출력**: `hi_scenario_interaction_{tag}.json` — HI 개념마다 `r_by_scenario`,
`std_r_across_scenarios`, `p_adj_bh`, `significant` 등을 담은 `per_hi` 딕셔너리.
검증에 쓰인 실제 실행 결과 예: raw HI 64개 중 24개 → `shared_gate`, 40개 →
기존 `scen_gates`.

---

## 8. Step 6 — Synergy grouping (`6_synergy/synergy.py`)

**목적**: 시나리오별로 "서로 중복되지 않으면서 함께 쓰면 시너지가 있는" HI들을
미리 그룹으로 묶어, Step 7의 커널 융합 대상으로 넘긴다.

**처리** (시나리오별 독립 수행):
1. 전체 HI를 SOH와의 단순상관 절대값 내림차순 정렬(시드 순서로 사용).
2. 미배정 HI 하나로 새 그룹을 시작해 `--max-group-size`(기본 4)까지 그리디하게
   확장한다.
3. 매 확장 단계: (a) 현재 그룹 멤버 전부와 raw 상관 절대값이
   `--redundancy-threshold`(기본 0.9) 미만인 후보만 남겨 다중공선성을 배제하고,
   (b) 그중 SOH 단순상관 상위 `--prefilter-top-m`개만 저비용으로 걸러낸 뒤,
   (c) 그 소수에 대해서만 **편상관계수**(그룹 전체로 SOH/후보를 회귀한 잔차 간
   상관)를 정밀 계산해 최댓값을 채택한다. 채택값이 `--min-partial-corr` 미만이면
   그룹 성장을 멈춘다.
4. 모든 HI가 배정될 때까지 반복 — 약한 HI는 크기 1짜리 그룹으로 남고, 이런
   그룹은 Step 7의 커널 융합 대상에서 제외된다.

**출력**: `synergy_groups_{tag}.json` — 시나리오별 그룹 멤버 인덱스/이름/점수.

---

## 9. Step 7 — Kernel HI 생성 (`7_kernel/kernel.py`)

**목적**: 편상관계수 같은 선형 지표로는 못 잡는 **비선형** 시너지를 명시적으로
캡처하는 새 특징을 만든다.

**처리**: Step 6에서 만든 그룹 중 크기 2 이상인 것만 대상으로,
**Nystroem(RBF 커널 근사) + Ridge 회귀** 파이프라인을 "그룹 멤버 HI들 →
SOH"로 직접 학습시키고, 그 **예측값 자체를 새 "커널 HI" 특징**으로 쓴다(PCA나
단순 가중합이 아니다). Raw HI를 대체하지 않고 옆에 "추가"한다. 3단계로
다중공선성을 관리한다: (1) Step 6이 이미 그룹 내부 raw 중복 배제, (2) 커널 HI끼리
(다른 시나리오 그룹 간에도) 상관이 높으면 train R²가 낮은 쪽 제거, (3) raw+kernel을
합쳐 시나리오별로 상관 절대값이 매우 높은(기본 0.95) 쌍마다 한쪽을 제거 대상으로
기록(실제 적용은 Step 8의 `train.py`가 담당).

**출력**:
- `kernel_group_features_{tag}.pkl` — 학습된 sklearn 파이프라인을 포함한 pickle.
  각 커널 피처의 `name, scenario, members(원본 HI), train_r2, model, mean, std`
  (정규화 통계는 own-scenario 데이터로 계산). 검증 실행 예: 시나리오별 커널
  HI 폭 `[16, 15, 16, 16, 15, 15]`, 평균 train R²=0.39.
- `kernel_group_features_{tag}_combined_redundancy.json` — Step 8이 게이트
  출력에 적용할 (시나리오, raw HI) 배제 목록.

---

## 10. Step 8 — 모델 학습 (`8_train/train.py`, `model_lib/models/scr_model.py`)

모델 이름은 **SCR (Scenario-Conditioned Routing)**. "어떤 HI를 볼지"를 사람이
고정하지 않고, 학습 가능한 희소 게이트(HardConcreteGate, L0 정규화)가
방향·시나리오별로 직접 고르게(select) 만든다.

### 10.1 모델 입력이 만들어지는 과정 — `model_lib/datasets/segment_dataset.py`

1. Step 4가 저장해 둔 raw HI 값을 그대로 로드한다(이 단계에서 새로 계산하는 값
   없음).
2. 원본 컬럼명(`stat_v_mean_cw_chg_lo` 등)을 `hi_00..hi_{N_HI-1}`로 익명화한다
   — 컬럼 순서는 stat→diff→lfp→morph 카테고리 순으로 고정.
3. `SegmentNormalizer`가 **train split만으로** 컬럼별 평균/표준편차를 `fit`하고
   (val/test는 fit하지 않음 — 누수 방지), `(x-mean)/std`로 z-score한 뒤 원래
   NaN이었던 자리를 정확히 0.0으로 채운다. 그 결과가 `x_hi`(정규화값)와
   `nan_mask`(1.0=유효했음, 0.0=원래 NaN)다.
4. Step 4가 이미 부여해 둔 `segment_id`로부터 `scen_idx`(0~5), `direction`
   (+1 충전/-1 방전), `level`(zone 0/1/2, 분류 타깃)을 역산한다 — 모델이 추론하는
   게 아니라 데이터 생성 시점에 이미 정해진 라벨이다.
5. Step 7의 커널 HI(`x_kernel`)와 Step 5/6/7의 산출물(shared/scen 배정, 커널
   게이트 폭, redundancy 마스크)이 `train.py`에서 `SCRModel` 생성자 인자로
   그대로 꽂힌다.

배치 딕셔너리 최종 형태: `{x_hi, nan_mask, direction, scen_idx, level, cap_init,
target, [x_kernel]}`. `cap_init`은 그 셀의 초기 용량을 z-score한 값(셀 크기
조건부화용), `target`은 SOH 비율(정답).

### 10.2 아키텍처 — `SCRModel.forward()`

```
입력: x_hi(64) · nan_mask · direction · scen_idx · cap_init
                    │
   ┌────────────────┼────────────────────┐
   ▼                ▼                    ▼
Stage A          Stage B               Stage B′
방향별 probe 게이트   시나리오별 게이트        커널 HI 게이트
(charge/discharge   (scen_gates 6개 +      (scen_kernel_gates,
 probe_gate,        공유 필요 HI는          커널 HI가 있을 때만)
 MSE+CE 동시 학습)   shared_gate 1개,
                    MSE만)
   │                │                    │
 probe_x           scen_x               kernel_x
   └────────────────┴────────────────────┘
                    ▼
     concat[probe_x, scen_x, kernel_x, direction, cap_init]
                    ▼
              Capacity Head (MLP)
                    ▼
              cap_pred (SOH 비율)
```

- **Stage A (방향 인지 probe 게이트)**: 세그먼트가 충전이면
  `charge_probe_gate`, 방전이면 `discharge_probe_gate`를 통과시킨다. 이 게이트는
  회귀(MSE) 그래디언트뿐 아니라, `probe_x + direction`을 입력으로 받는 별도
  분류 헤드(`probe_mlp`, level 3클래스 CE)로부터도 그래디언트를 받는다
  (dual-objective) — "회귀에도 분류에도 다 쓸모 있는 HI"를 고르도록 유도.
- **Stage B (시나리오별 게이트)**: `scen_idx`로 라우팅되는 6개(또는 공유가
  있으면 그만큼 좁아진) `HardConcreteGate`. Interaction test(§7)에서 "공유"로
  판정된 HI는 여기서 빠지고 시나리오 무관 단일 `shared_gate`가 대신 담당한다.
  MSE 그래디언트만 받는다 — 순수하게 "이 시나리오에서 회귀에 유용한가"만 본다.
- **Stage B′ (커널 게이트)**: Step 7의 커널 HI(`x_kernel`)에 동일한 방식의
  시나리오별 게이트를 적용한다. `n_kernel_hi=0`이면(커널 블록 미사용) 이 단계
  전체가 비활성.
- **HardConcreteGate**: L0 정규화의 연속 완화(concrete relaxation) — 학습 중엔
  거의 이진(0/1)에 가까운 확률적 마스크를 곱하고, 비활성 상태로 수렴한 입력은
  기여가 0이 된다. "몇 개의 HI를 실제로 쓰는가"가 loss의 세 번째 항(§10.3)으로
  직접 제어된다.
- **Capacity Head**: `[probe_x(64) + scen_x(64) + kernel_x(≤16) + direction(1)
  + cap_init(1)]`을 이어붙인 벡터를 2-hidden-layer MLP(`d_head=128`)에 넣어
  스칼라 SOH 비율 하나를 낸다. (다른 회귀 헤드 아키텍처 — Transformer,
  ResNet-tabular, FT-Transformer 등 — 도 `model_lib/models/cap_heads.py`에
  구현되어 있으나, 정식 학습 레시피는 항상 `mlp`를 쓴다. 이들은 논문의
  ablation 실험 인프라다.)

### 10.3 손실 함수 — `model_lib/training/scr_loss.py`

```
L = MSE(cap_pred, SOH 정답)
  + λ_scen · CE(level_logits, level 정답)     ← Stage A의 probe_mlp만
  + λ_l0   · L0_penalty                        ← 위 세 게이트 전부
```

`L0_penalty`는 시나리오마다 "활성 HI들의 비용 기댓값"을 계산한다(기본은 균일
비용 1.0 — 즉 활성 게이트 "개수"만 페널티가 된다; `--hi-cost-weighted-l0`로
카테고리별 실측 계산비용 가중치를 켤 수 있다). 이 항 덕분에 학습이 끝나면
시나리오마다 실제로 쓰는 HI 목록이 좁게 수렴한다 — 검증 실행에서
`gate_saturation`(게이트가 얼마나 극단값 0/1에 붙었는지) 0.004 수준까지 수렴함을
확인했다.

### 10.4 학습 루프

`8_train/train.py`가 500 epoch을 상한으로 학습하며, `lambda_l0`를 warmup 구간
동안 서서히 올리고(`delayed_warmup`), val 성능과 gate saturation이 일정 기간
개선되지 않으면 조기 종료한다. 체크포인트는 `checkpoints/best_by_saturation.pt`
(saturation 기준 최적 epoch)에 저장되고, `config.yaml`/`p1v2_summary.json`에
그 run의 완전한 설정이 기록된다. **seed 지정 시 전체 파이프라인이 바이트 단위로
재현 가능** — 이 프로젝트의 모든 리팩토링은 동일 seed로 재학습한 체크포인트의
MD5가 리팩토링 전후로 완전히 일치하는지로 검증되어 왔다.

---

## 11. Step 9 — 평가 (`9_eval/test.py`)

실배포 상황에서는 "지금 보는 세그먼트가 어느 시나리오인지" 정답 라벨이 없다.
그래서 세 가지 라우팅 조건을 모두 평가한다.

| 모드 | 라우팅 방식 | 의미 |
|---|---|---|
| **oracle** | 정답 `scen_idx`로 직접 라우팅 | 분류가 100%라고 가정한 회귀 성능의 상한선 |
| **hard** | `probe_mlp` argmax(최고 확률 시나리오 하나)로 라우팅 | 실배포와 가장 가까운 조건 |
| **soft** | `probe_mlp` 확률로 시나리오별 예측을 가중 평균 | 분류 경계 근처 샘플의 오차를 완화 |

체크포인트에 `probe_mlp`가 있으면(즉 학습 시 `lambda_scen > 0`) 세 모드 모두
평가하고, 없으면 oracle만 평가한다. 측정 지표는 SOH에 대한 RMSE/MAE/R²/MAPE,
시나리오별 breakdown, (hard/soft에서는) 분류 정확도이며, `metrics/metrics.json`에
저장된다. 대표 셀 몇 개를 골라 시간에 따른 실제 용량 대 예측 용량 곡선도
`figures/`에 함께 그린다.

---

## 12. 최종 출력이 의미하는 것

모델이 세그먼트 하나에 대해 실제로 내놓는 값은 **SOH 비율 하나**(`cap_pred`,
0~1 사이 실수 = 현재 용량 / 초기 용량)다. 여기에 그 셀이 속한 데이터셋의 공칭
용량(`nominal_capacities`: MIT 1.1 Ah, HUST 1.2 Ah, `parameters.py`)을 곱하면
추정 잔존 용량(Ah)이 된다. 배터리를 끝까지 충방전하지 않고, 짧은 구간 하나만
관측해서 얻어내는 숫자라는 점이 이 프레임워크의 핵심 가치다.

---

## 13. 용어 정리

| 용어 | 의미 |
|---|---|
| **세그먼트(segment)** | 한 사이클의 충전 또는 방전 구간 중, q_frac 기준으로 잘라낸 짧은 구간 하나. 모델의 최소 입력 단위. |
| **시나리오(scenario)** | 세그먼트가 속한 (방향 × SOC 존) 조합. 6개: chg_lo/chg_mid/chg_hi/dis_hi/dis_mid/dis_lo. |
| **HI (Health Indicator)** | 세그먼트 하나의 V/I/t 곡선으로부터 계산한 손수 설계 특징. 4 카테고리(stat/diff/lfp/morph), 기본 66개. |
| **게이트(gate)** | 학습 가능한 이진에 가까운 마스크(HardConcreteGate). "이 HI를 쓸지 말지"를 데이터로부터 학습. |
| **라우팅(routing)** | 세그먼트를 어느 시나리오 게이트로 보낼지 결정하는 것. oracle=정답 라벨, hard/soft=분류기 기반. |
| **q_frac** | 충전 또는 방전 구간 안에서, 지금까지 누적된 전하량이 전체 대비 차지하는 비율(0~1). |
| **SOH** | State of Health = 현재 용량 / 초기(신품) 용량. 모델의 회귀 타깃. |
| **BOL** | Beginning of Life — 그 셀의 첫 유효 세그먼트. MORPH 특징의 형태 비교 기준. |
| **L0 정규화** | 활성(0이 아닌) 파라미터 개수 자체에 페널티를 주는 희소화 기법. HardConcreteGate로 미분 가능하게 완화해서 사용. |

---

## 14. 실행 방법 요약

```bash
python run_pipeline.py                 # 전체 파이프라인, Step 1부터
python run_pipeline.py 4 --to-step 4   # Step 4(시나리오 분할+HI 추출)만
python run_pipeline.py 8               # 학습+평가만 (Step 5~7 산출물 재사용)
python run_pipeline.py 8 --to-step 8   # 학습만(평가 제외)
python run_pipeline.py 9               # 평가만(직전 run 자동 탐색)
```

자주 바꾸는 파라미터(`ACTIVE_*`)는 CLI 플래그로 노출되고, 거의 안 바꾸는
파라미터(`FIXED_*`)는 `parameters.py`를 직접 수정해야 한다(§`docs/REFATORING.md`
참고 — 이 구분과 세부 변경 이력은 그 문서가 계속 추적한다).
