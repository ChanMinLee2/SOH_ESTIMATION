# `preprocess.ipynb` vs `1_convert/ ~ 4_hi_analysis/` 단계별 차이 분석

---

## 1단계: 데이터 로딩 및 어댑터

| 항목 | 파이프라인 (`convert_unified.py`) | `preprocess.ipynb` (Phase 0/0-1) |
|------|----------------------------------|----------------------------------|
| 입력 형식 | `.mat` (MIT) / `.pkl` (HUST) 원본 파일 | `load_hust()` / `load_mit()` 커스텀 함수로 로딩 |
| 내부 자료구조 | 셀별 단일 long-format DataFrame (모든 사이클 행 스택) | `cell.data[cyc]` 계층형 딕셔너리 (사이클 → DataFrame) |
| 통합 스키마 | `cycle, time_s, voltage_V, current_A, temperature_C, capacity_Ah, phase` | `COL_V, COL_I, COL_TIME, COL_T, capacity` (컬럼명 매핑만 수행) |
| phase 구분 | `phase` 컬럼으로 charge/discharge/rest 명시 분리 | 현재 전류 부호 (`>0.01` / `<-0.01`)로 런타임에 분리 |
| 저장 | `data_raw/`, `data_unified/` 에 PKL/CSV | 메모리에만 유지, 배치 캐시(`batch_clean_*.pkl`)로만 디스크 저장 |

**핵심 차이**: 파이프라인은 사이클 경계를 `phase` 컬럼으로 정적으로 확정한 뒤 저장하지만, 노트북은 전류 부호 필터를 매번 실행 시 동적으로 적용합니다. 이로 인해 CV 충전 구간 말단(전류 ≈ 0)에서 양쪽이 다르게 처리될 수 있습니다.

---

## 2단계: 이상 사이클/포인트 제거 (클리닝)

| 항목 | 파이프라인 (Step 1+2) | `preprocess.ipynb` (Phase 1/1.1) |
|------|----------------------|----------------------------------|
| 불량 셀 수동 제거 | `DELETE_CELLS`: MIT 12셀 완전 제외 (Step 1에서 처리) | `removal_index.csv` 로드 — 셀 단위가 아닌 **사이클 단위** 수동 목록 |
| 빈 사이클 제거 | `_remove_empty_cycles()`: 5행 미만 | 별도 없음 (슬라이싱 후 5행 미만이면 segment 미포함) |
| 시간 단조성 보정 | `_fix_time_monotonicity()`: 사이클 내 역방향 점프 보정 (MIT 분 단위 타임스탬프 대응) | 없음 |
| rest 0전류 행 제거 | `_remove_zero_current_rest()`: phase=rest && I==0인 **행** 제거 | `clean_physical_data()` 에서 `I.abs() == 0` 행 제거 (phase 구분 없이) |
| 시간 단절 제거 | `_remove_dt_gap_cycles()`: 방전 단절 → 전체 사이클 제거, 충전 단절 → 충전 행만 제거 (기본: dis ≥ max(600s, 50×median) / chg ≥ max(120s, 30×median)) | **없음** |
| 이상 사이클 자동 제거 | **Rolling Median** (window=11, σ=2.5): 용량 시계열 이상치 사이클 제거 → MIT 24건, HUST 0건 | **없음** — 롤링 메디안 자동 제거 로직 미구현 |
| 비정상 종료 제거 | `vend_min < 1.8V`: HUST **4,214건(2.88%)** 제거 | **없음** — v_end 기반 필터 없음 |
| 단위 표준화 | Phase 1에서 완전히 완료 | `clean_physical_data()` 내 mA→A 변환 (mean>50이면 /1000) |
| NaN 처리 | 개별 컬럼별 rolling 보간 없이, HI 추출 시 NaN → 0 | NaN 위치 리포트만 기록, 보간 없음 |
| Z-score 포인트 제거 | 없음 | 코드 주석에 "Z-score 로직 삭제됨" 명시 — 예전에 있었다가 제거 |

**핵심 차이**: 파이프라인은 **시간 단절 감지**, **롤링 메디안 자동 사이클 제거**, **종지전압 하한 필터** 등 6단계 필터를 Step 2에서 모두 수행하여 `data_postprocess/`에 저장합니다. 노트북은 이 필터들이 없어 HUST 비정상 종료 4,214개 사이클과 시간 단절 사이클이 HI 계산에 그대로 노출됩니다.

---

## 3단계: 슬라이싱 전략 (SoC 구간 분할)

이 항목이 가장 근본적인 설계 차이입니다.

| 항목 | 파이프라인 (Step 4 `hi_correlation.py`) | `preprocess.ipynb` (Phase 2) |
|------|----------------------------------------|------------------------------|
| 분할 단위 | 사이클 1개 → HI scalar 1세트 | 사이클 1개 → **다수의 슬라이스 segment** |
| SoC 구간 수 | 3구간 고정 (s_hi/s_mid/s_lo) | 3구간 고정 (H/M/L) |
| 구간 경계 | **q_frac 0.4/0.7** (비대칭 — 플래토 중심 확장) | **q_frac 0.3/0.7** (준대칭) |
| 윈도우 크기 | 없음 — 구간 전체 | `lengths = [0.2, 0.3, 0.4]` — **3가지 크기** |
| 슬라이딩 스텝 | 없음 | `step = 0.1` — **슬라이딩 윈도우** |
| 모드별 분리 | phase 컬럼으로 사전 분리 | 전류 부호 필터 런타임 적용 |
| 데이터 증강 | 없음 | 1사이클에서 `3(SoC) × 3(len) × n(step) × 2(mode)` 다수 segment 생성 |
| 실제 데이터량 증폭 | 1 cycle → 1 HI row | 1 cycle → 수십 개 segment |
| 세그먼트 최소 길이 | 없음 (전체 구간 사용) | 5포인트 미만 segment 폐기 |
| 사이클 샘플링 | 전체 사이클 처리 | `cycle_step` 적용 (매 N번째 사이클만, cycle 1 제외) |

**q_frac 경계 비교**:

```
파이프라인:  [---s_hi(0~0.4)---][--s_mid(0.4~0.7)--][---s_lo(0.7~1.0)---]
노트북:      [--H(0~0.3)--]    [-----M(0.3~0.7)-----]    [--L(0.7~1.0)--]
```

파이프라인의 s_hi 구간이 더 넓고(40%), 노트북의 H 구간이 더 좁습니다(30%).

---

## 4단계: HI 피처 추출

이 항목이 두 번째 핵심 차이입니다.

| 항목 | 파이프라인 (`hi_correlation.py`) | `preprocess.ipynb` (Phase 3 `extract_75d_hi`) |
|------|----------------------------------|-----------------------------------------------|
| 입력 경로 | `data_postprocess/` (Step 2 산출물) | 메모리 내 클린 데이터 |
| 피처 차원 | **148D** (22 전체 + 20×3 방전구간 + 6 충방전 공통 + 20×3 충전구간) | **75D** (45D base + 30D new) |
| 피처 단위 | **사이클 전체** 또는 **SoC 구간 전체** | **슬라이스 segment** (슬라이딩 윈도우 부분) |
| 피처 철학 | 물리 메커니즘 직결 (ICA, DVA, 에너지, 플래토 용량 등) | 통계적 특성 + 물리 특성 혼합 |
| ICA 처리 | Savitzky-Golay 스무딩 후 peak 추출 (`ica_peak_h/v/area`) | 스무딩 없이 raw `dQ/dV`, peak 통계 (`n1~n8`) |
| DVA 처리 | SoC 구간 내 `dV/dQ` 최솟값 (`dvdq_min`) | `dV/dQ` 통계 다수 (`n9~n14`) + 곡률/비대칭 |
| 엔트로피 피처 | 없음 | `c1_10_ent_v`, `c1_11_ent_t`, `c3_cha_13_dvdt_entropy` |
| 상관 피처 | 없음 | `c1_12_corr_vi` (V-I 상관), `c1_13_corr_vt` (V-T 상관) |
| 플래토 분석 | `q_high_v`, `q_tail`, `q_plateau_ratio` | `n15_plateau_len`, `n18/19_plateau_start/end_v` |
| 온도 피처 | `temp_mean`, `temp_max` | `c1_5_mean_t`, `c1_6_delta_t`, `c1_7_dtdt` (dT/dt), `n25_thermal_stability`, `n26_dq_dt_plateau` |
| 쿨롱 효율 | `ce` (방전Q/충전Q) | 없음 |
| 에너지 가중 전압 | `v_energy = energy_Wh / capacity_Ah` | `c1_14_de_dq` (에너지/전하 비율, 유사 개념) |
| SOC별 포인트 전압 | `v_at_q20/50/80` | 없음 |
| 피처 가독성 | 물리 명칭 (e.g., `ica_peak_h`, `q_plateau_ratio`) | 코드형 명칭 (e.g., `c1_3_skew_v`, `n23_nonlinear_idx`) |

**새로운 30D 피처(`n1~n30`) 특징**:

```
n1~n8:  ICA (dQ/dV) 기반 — peak 위치/기울기/반폭/skew/kurt/valley/IC-T 상관/peak 개수
n9~n14: DVA (dV/dQ) 기반 — center/min/gap/곡률/면적/비대칭
n15~n30: 플래토/전압 형태 — 플래토 길이, 시작/끝 전압, 내부저항 근사, 비선형 지수 등
```

---

## 5단계: 정규화/스케일링

| 항목 | 파이프라인 | `preprocess.ipynb` (Phase 3.1) |
|------|-----------|-------------------------------|
| 스케일링 | **없음** — raw HI 값 그대로 | `StandardScaler` **partial_fit** → 전체 배치 transform |
| 스케일러 저장 | 없음 | `{DATASET_TYPE}_scaler.pkl` 로 저장 |
| 마스킹 | `shaping_features.py` 에서 시나리오 무관 HI → 0 | Phase 3.1에서 즉시 적용: discharge면 c3(15~29) → 0, charge면 c2(15~29) → 0 |
| 메타데이터 임베딩 | `--add-meta` 옵션 시 `[soc_label, mode_label, 1.0]` 3D 추가 (148→151D) | 항상 추가: `[soc_label, mode_label, length_p]` → **75→78D** |

**주의**: 노트북은 `length_p`가 실제 슬라이딩 윈도우 길이(0.2/0.3/0.4)를 담습니다. `shaping_features.py`는 `length_p = 1.0` 고정 (전체 구간이라 의미 없음).

---

## 6단계: 출력 포맷 및 레이블

| 항목 | 파이프라인 (`hi_features.pkl` → `shaping_features.py`) | `preprocess.ipynb` (Phase 4) |
|------|-------------------------------------------------------|------------------------------|
| 중간 저장 | `data_HI/{dataset}/{cell_id}.pkl` (셀별) + `hi_features.pkl` (전체 통합 DataFrame) | `{DATASET_TYPE}_feature_batches/final_*.pkl` (배치 분할) |
| 최종 출력 | `hi_shaped.pkl` — `list[dict]` | `{DATASET_TYPE}_optimized_tensors.pkl` — `list[dict]` |
| dict 구조 | `{cell, cyc, x(148D/151D), soc_label, mode_label, length_p, y, capacity}` | `{cell, cyc, x(75D/78D), soc_label, mode_label, length_p, y, capacity}` |
| y 레이블 | `capacity_Ah` (RUL 없음, 명시) | `get_cell_labels()`가 반환하는 `rul_label` 또는 `cap_label` |
| 아이템 수 | `총 사이클 수 × 6 시나리오` (고정) | `총 사이클 수 × cycle_step ÷ N × (다수 segment)` (슬라이딩으로 더 많음) |
| 시나리오 확장 방식 | `shaping_features.py`에서 후처리로 6시나리오로 **복사 + 마스킹** | Phase 2 슬라이싱에서 각 segment가 자체 `soc_label` 보유 |

---

## 전체 요약: 설계 철학의 차이

```
파이프라인 (1~4단계)                    preprocess.ipynb
─────────────────────────────────       ─────────────────────────────────
목적: 탐색적 분석 (EDA) + HI 계산      목적: 모델 학습 데이터 생성
단위: 사이클 1개 = HI scalar 1세트      단위: 슬라이딩 윈도우 segment
피처: 148D, 물리 해석 중심              피처: 75D, 통계+물리 혼합, StandardScaling 포함
클리닝: 6단계 자동 필터 (data_postprocess/)   클리닝: 수동 목록 + 최소 처리
SoC 경계: q_frac 0.4/0.7               SoC 경계: q_frac 0.3/0.7
데이터 증강: 없음 (1→1)                 데이터 증강: 있음 (슬라이딩, 1→N)
저장: 셀별 PKL → 통합 DataFrame         저장: 배치 PKL → 최종 tensor list
스케일링: 없음                          스케일링: StandardScaler
length_p: 항상 1.0 고정                 length_p: 실제 윈도우 크기 (0.2/0.3/0.4)
```

---

## 가장 주목할 불일치 포인트 3가지

1. **HUST 비정상 종료 4,214사이클**: 파이프라인은 `vend_min < 1.8V`로 제거하지만 노트북은 이를 제거하지 않아 오염된 segment가 학습 데이터에 포함될 수 있습니다.

2. **SoC 구간 경계 불일치 (0.4/0.7 vs 0.3/0.7)**: 파이프라인의 s_hi(0~0.4)는 노트북의 H(0~0.3)보다 넓어, 같은 "방전 초반 고전압" 구간이 서로 다르게 정의됩니다. 두 시스템이 생성한 피처를 혼용할 때 의미 불일치가 발생합니다.

3. **피처 차원/집합 불일치 (148D vs 75D)**: 파이프라인의 `hi_features.pkl`을 `shaping_features.py`로 변환한 결과(148/151D)와 노트북이 직접 생성한 tensor(75/78D)는 **완전히 다른 피처 공간**입니다. `train.py`의 `input_dim=78` 설정은 노트북 경로를 전제한 것입니다.
