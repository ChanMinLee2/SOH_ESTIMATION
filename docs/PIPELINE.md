# LFP SOH Prediction — 데이터 파이프라인

원본 데이터(.mat / .pkl)를 모델 학습까지 처리하는 전체 흐름.

---

## 파이프라인 흐름

```
원본 데이터
  ├── data_raw/FastCharge/*.mat          MIT (FastCharge, HDF5 포맷)
  └── data_raw/data_raw/our_data/our_data/*.pkl   HUST (our_data 원본)
             │
             ▼
  ┌─────────────────────────────────────────────────┐
  │  Step 1: 데이터 변환          convert_unified.py  │
  │    MIT  → mit_batch_parser/build_batch_pkl.py    │
  │    HUST → convert_unified.py 내 _hust_worker()   │
  └─────────────────────────────────────────────────┘
             │
             ├──────────────────────────────────────────┐
             ▼                                          ▼
  data_raw/                                  data_unified/
    ├── MIT/  b1c0.pkl, b1c0.csv, ...          ├── MIT/  b1c0.pkl, b1c0.csv, ...
    │         (이상치 제거 없음,                │         (필터 적용, 123셀)
    │          DELETE_CELLS 포함)               └── HUST/ 1-1.pkl,  1-1.csv, ...
    └── HUST/ 1-1.pkl,  1-1.csv, ...                     (필터 적용,  77셀)
              (이상치 제거 없음)
    ★ data_raw/FastCharge/ (MIT 원본 .mat) 와 공존

  PKL 스키마: {"meta": dict, "cycles": DataFrame}
  cycles 컬럼: cycle, time_s, voltage_V, current_A,
               temperature_C, capacity_Ah, phase
             │
             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Step 2: 이상 사이클·행 제거 (7단계)     preprocess.py     │
  │    [필터1] _remove_empty_cycles          빈 사이클 제거     │
  │    [필터2] _fix_time_monotonicity         time_s 단조 보정  │
  │    [필터3] _remove_zero_current_rest      rest 0전류 행 제거 │
  │    [필터4] _remove_dt_gap_cycles          시간 단절 처리    │
  │              방전 단절 → 사이클 전체 제거                   │
  │              충전 단절 → 충전 phase 행만 제거               │
  │    [필터5] _remove_outlier_cycles  Rolling Median 2-pass    │
  │              Pass 1: w=11, σ=--sigma(기본2.0) — 고립 이상치 │
  │              Pass 2: w=31, σ2=2.0  — 클러스터 이상치       │
  │    [필터6] _remove_bad_vend_cycles vend_min=1.8V            │
  │              HUST 비정상 종료 사이클 ~4,214건 제거           │
  │    [필터7] _remove_shape_outlier_cycles  V-q_frac 형상 편차  │
  │              방전·충전 각각 rolling median 기준 곡선 대비    │
  │              RMSE·max|ΔV|의 MAD robust z > σ(기본5.0) 제거 │
  │              + KNOWN_SHAPE_ANOMALIES 수동 지정 사이클 추가   │
  │                MIT b1c23 charge #1003, b1c36 discharge #73  │
  │    data_unified 원본 유지 → data_postprocess에 저장          │
  │    PKL: 전체 셀 / CSV: 제거 발생 셀만                       │
  │    → outputs/cleaning_report.csv (필터별 통계 + 사이클 번호) │
  │    → outputs/<MMDD>/*.png  (plot_cleaning_report.py 시각화) │
  │    병렬 처리: --workers N (기본 4)                           │
  └──────────────────────────────────────────────────────────────┘
             │
             ▼
  data_postprocess/
    ├── MIT/  b1c0.pkl, [b1c0.csv], ...   (전처리 적용)
    └── HUST/ 1-1.pkl,  [1-1.csv],  ...   (CSV는 제거 셀만)
             │
             ▼
  ┌──────────────────────────────────────────────────┐
  │  Step 3: 무결성 검사         check_integrity.py  │
  │    파일 수 / 스키마 / phase / 전압범위 / NaN 등  │
  │    병렬 처리: --workers N (기본 4)               │
  └──────────────────────────────────────────────────┘
             │
             ▼
  ┌───────────────────────────────────────────────────────────┐
  │  Step 4: HI 분석 (탐색)                                  │
  │    hi_correlation.py   → hi_plot/<MMDD>/                 │
  │                            hi_correlation.png            │
  │    hi_segment_viz.py   → hi_plot/<MMDD>/                 │
  │                            hi_segment_cuts.png           │
  │                            hi_trend.png                  │
  │                            hi_segment_trend.png          │
  │    seg_corr_analysis.py → seg_corr/<MMDD>/              │
  │                            corr_rank_<ds>.png            │
  │                            corr_matrix_<ds>.png          │
  │                            top7_cross_<ds>.png           │
  │                                                          │
  │  HI 64종  (방전 Global 22 + 방전 SoC 세그먼트 18         │
  │           + 충전 Global 6 + 충전 SoC 세그먼트 18)        │
  │    방전 Global: v_mean, v_std, v_skew, v_kurt, v_end,   │
  │                 v_drop, energy_Wh, v_energy,             │
  │                 v_at_q20/50/80, q_high_v, q_tail,        │
  │                 q_plateau_ratio, t_discharge,            │
  │                 ica_peak_h/v/area, dvdq_min, ce,         │
  │                 temp_mean, temp_max                      │
  │    방전 SoC 3구간: v_mean, v_std, energy_Wh, q_abs,     │
  │                    ica_peak_h, dvdq_min  × 3구간         │
  │    충전 Global: q_cc_ratio, chg_v_energy,               │
  │                 chg_ica_peak_h/v/area, chg_dvdq_min      │
  │    충전 SoC 3구간: v_mean, v_std, energy_Wh, q_abs,     │
  │                    ica_peak_h, dvdq_min  × 3구간         │
  └───────────────────────────────────────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────────────┐
  │  Step 5~6: 텐서 생성 / 모델 학습  (미구현) │
  └──────────────────────────────────────────────┘
```

---

## 소스 파일 역할

| 파일 | 단계 | 역할 |
|------|------|------|
| `1_convert/convert_unified.py` | 1 | MIT MAT + HUST PKL → unified PKL 변환 (전체 통합) |
| `2_preprocess/preprocess.py` | 2 | 7단계 이상 사이클·행 제거 (빈 사이클, time 보정, 0전류 rest, 시간 단절, Rolling Median 2-pass, vend_min, 형상 편차) |
| `2_preprocess/plot_cleaning_report.py` | 2 | cleaning_report.csv 읽어 필터별 시각화 플롯 생성 |
| `3_integrity/check_integrity.py` | 3 | unified PKL 무결성 검사 → 이상 목록 CSV 저장 |
| `4_hi_analysis/hi_correlation.py` | 4 | HI 64종 추출 및 풀링 Spearman 상관 시각화 |
| `4_hi_analysis/hi_segment_viz.py` | 4 | 세그먼트 분할 확인 + HI 열화 추이 시각화 |
| `4_hi_analysis/seg_corr_analysis.py` | 4 | 세그먼트별 within-cell Spearman 상관계수 랭킹·히트맵·Top-7 비교 |

**분석 도구** (파이프라인 외부):

| 파일 | 역할 |
|------|------|
| `4_hi_analysis/plot_cell_cycles.py` | 특정 셀의 전체 사이클 용량 곡선 + V-q_frac 오버레이 시각화 |
| `4_hi_analysis/plot_cycle_segments.py` | 단일 셀·사이클의 방전/충전 세그먼트 V-q_frac 시각화 |

---

## 실행 명령어

> 모든 명령은 프로젝트 루트(`LFP_SOH_prediction/`)에서 실행.

---

### Step 1 — 데이터 변환 (`1_convert/convert_unified.py`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--dataset` | `all` | 변환할 데이터셋: `mit` / `hust` / `all` |
| `--workers` | `3` | 병렬 프로세스 수. CPU 코어 수 이하로 설정 권장 |
| `--cell` | (없음) | HUST 단일 셀만 변환 (예: `1-1`). MIT는 항상 전체 변환 |
| `--output-root` | `data_unified/` | 출력 디렉토리 (변경 불필요) |
| `--no-cache` | `False` | `data_raw/` 캐시 무시 — 원본 MAT/PKL부터 재파싱 |

```powershell
# 권장 — MIT + HUST 전체 변환 (data_raw/ 캐시 있으면 자동 사용)
python 1_convert/convert_unified.py --dataset all --workers 3

# 캐시 무시하고 원본부터 재파싱
python 1_convert/convert_unified.py --dataset all --no-cache

# 데이터셋 개별 변환
python 1_convert/convert_unified.py --dataset mit
python 1_convert/convert_unified.py --dataset hust

# 빠른 재변환 (CPU 많을 때)
python 1_convert/convert_unified.py --dataset all --workers 8

# HUST 단일 셀 테스트
python 1_convert/convert_unified.py --dataset hust --cell 1-1
```

출력:
- `data_raw/MIT/*.pkl` / `data_raw/HUST/*.pkl` — 원본 파싱 결과 (필터 없음, 캐시로 재사용)
- `data_unified/MIT/*.pkl` / `data_unified/HUST/*.pkl` — 필터 적용 결과 (셀당 .csv도 생성)

---

### Step 2 — 이상 사이클·행 제거 (`2_preprocess/preprocess.py`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--dataset` | `all` | 처리할 데이터셋: `mit` / `hust` / `all` |
| `--window` | `11` | [필터5] Pass 1 Rolling Median 윈도우 크기 |
| `--sigma` | `2.0` | [필터5] Pass 1 이상치 σ 임계값. 낮을수록 더 많이 제거 (Pass 2는 w=31, σ=2.0 고정) |
| `--min-std` | `0.01` | [필터5] Rolling std 최솟값 플로어 (Ah). 이 이하로 내려가지 않도록 클리핑 |
| `--vend-min` | `1.8` | [필터6] 방전 종지전압 하한 (V). HUST 2.88% 해당 |
| `--shape-sigma` | `5.0` | [필터7] 형상 편차 MAD robust z 임계값. 낮을수록 더 많이 제거 |
| `--shape-window` | `11` | [필터7] 기준 곡선 rolling median 윈도우 |
| `--shape-grid` | `100` | [필터7] q_frac 보간 격자 점 수 |
| `--dis-gap-s` | `600` | [필터4] 방전 단절 절대 기준 (초) |
| `--dis-gap-factor` | `50` | [필터4] 방전 단절 배율 기준 (median × N) |
| `--chg-gap-s` | `600` | [필터4] 충전 완전 중단 절대 기준 (초). CC 프로토콜 전환 갭(120~600s)은 hi_correlation에서 세그먼트 HI만 NaN |
| `--chg-gap-factor` | `50` | [필터4] 충전 완전 중단 배율 기준 (median × N) |
| `--workers` | `4` | 병렬 프로세스 수 |

```powershell
# 권장 — 기본 파라미터 전체 실행
python 2_preprocess/preprocess.py

# 병렬 처리 (CPU 많을 때)
python 2_preprocess/preprocess.py --workers 8

# HUST만 재처리
python 2_preprocess/preprocess.py --dataset hust

# Rolling Median 더 엄격하게
python 2_preprocess/preprocess.py --sigma 2.0 --window 15

# 종지전압 임계값 조정
python 2_preprocess/preprocess.py --vend-min 1.9

# 형상 필터 임계값 완화 (false positive 우려 시)
python 2_preprocess/preprocess.py --shape-sigma 6.0

# 형상 필터 강화 (더 많이 제거)
python 2_preprocess/preprocess.py --shape-sigma 4.0

# 시간 단절 기준 완화 (정상 갭 보존)
python 2_preprocess/preprocess.py --dis-gap-s 1200 --dis-gap-factor 100
```

출력:
- `data_postprocess/MIT/*.pkl` / `data_postprocess/HUST/*.pkl` — 전체 셀 저장
- `data_postprocess/MIT/*.csv` / `data_postprocess/HUST/*.csv` — 제거 발생 셀만
- `2_preprocess/outputs/cleaning_report.csv` — 셀별 제거 사이클 수 + 번호 리포트

> **주의**: Step 1 재실행 후에는 반드시 Step 2도 재실행해야 필터가 적용됨.  
> `data_unified/` 원본은 변경되지 않음 — 후속 분석은 `data_postprocess/` 사용.

#### 2-B. 전처리 리포트 시각화 (`2_preprocess/plot_cleaning_report.py`)

`cleaning_report.csv`를 읽어 필터별 제거 결과를 그래프로 출력하는 보조 스크립트.  
`preprocess.py`와 별도로 실행 (PKL 재처리 없이 리포트만 재시각화 가능).

```powershell
python 2_preprocess/plot_cleaning_report.py
```

출력: `2_preprocess/outputs/<MMDD>/` 아래 필터별 PNG 파일
- `F4A_dis_gap.png` — 방전 단절 사이클
- `F4B_chg_stop.png` — 충전 완전 중단 행
- `F4C_chg_seg.png` — 충전 CC 전환 갭 플래그
- `F5_rolling.png` — Rolling Median 이상치 제거 전·후 용량 곡선

---

### Step 3 — 무결성 검사 (`3_integrity/check_integrity.py`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--workers` | `4` | 병렬 프로세스 수. CPU 코어 수 이하로 설정 권장 |

```powershell
# 기본 실행
python 3_integrity/check_integrity.py

# 병렬 처리
python 3_integrity/check_integrity.py --workers 8
```

검사 항목:

| 수준 | criterion | 설명 |
|------|-----------|------|
| 셀 | `missing_cols` | 필수 컬럼 누락 |
| 셀 | `invalid_phase` | charge/discharge/rest 외 phase 값 |
| 셀 | `current_direction` | MIT 방전 전류 음수 / HUST 방전 전류 양수 |
| 셀 | `capacity_increasing` | 용량 증가 추세 (병합 오류 의심) |
| 셀 | `high_nan` | 컬럼 NaN 비율 50% 초과 |
| 셀 | `cycle_count_mismatch` | meta.n_cycles ≠ 실제 사이클 수 |
| 사이클 | `voltage_high` | v_max > 4.5V |
| 사이클 | `voltage_low` | v_min < 1.5V |
| 사이클 | `rest_dominant` | rest 행 비율 > 80% (비정상 프로토콜) |
| 사이클 | `time_nonmono` | time_s 단조 증가 위반 |

출력:
- `3_integrity/outputs/integrity_report.csv` — 셀별 요약 통계
- `3_integrity/outputs/integrity_issues.csv` — 셀·사이클별 이상 목록 (`severity`, `dataset`, `cell_id`, `cycle`, `criterion`, `detail`)

---

### Step 4 — HI 추출 및 시각화

#### 4-A. Spearman 상관 분석 (`4_hi_analysis/hi_correlation.py`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--workers` | CPU 수 (최대 4) | HI 추출 병렬 워커 수 |
| `--n-top` | `4` | 상관계수 플롯 하단 산점도에 표시할 상위 HI 개수 |
| `--force` | `False` | 캐시(`hi_features.pkl`) 무시하고 HI 재추출 |

```powershell
# 권장 (첫 실행 — HI 추출 후 캐시 저장)
python 4_hi_analysis/hi_correlation.py

# 산점도 상위 HI를 더 많이 보고 싶을 때
python 4_hi_analysis/hi_correlation.py --n-top 8

# convert 재실행 후 캐시 갱신
python 4_hi_analysis/hi_correlation.py --force
```

출력:
- `4_hi_analysis/hi_plot/<MMDD>/hi_correlation.png` — HI×용량 풀링 Spearman ρ 히트맵 + 산점도
- `4_hi_analysis/hi_features.pkl` — 전체 HI 추출 결과 캐시 (4-B·4-C와 공유)
- `data_HI/MIT/{cell_id}.pkl` / `data_HI/HUST/{cell_id}.pkl` — 셀별 HI 특성 (첫 실행 또는 `--force` 시)

#### 4-B. 세그먼트 시각화 (`4_hi_analysis/hi_segment_viz.py`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--workers` | `4` | HI 추출 병렬 워커 수 (캐시 없을 때만 사용) |
| `--n-cycles` | `4` | 세그먼트 분할 확인 플롯에서 보여줄 대표 사이클 수 |
| `--force` | `False` | 캐시 무시하고 HI 재추출 |

```powershell
# 권장 (4-A 실행 후 캐시에서 즉시 로드)
python 4_hi_analysis/hi_segment_viz.py

# 대표 사이클 더 많이 표시
python 4_hi_analysis/hi_segment_viz.py --n-cycles 6
```

출력:
- `4_hi_analysis/hi_plot/<MMDD>/hi_segment_cuts.png` — V-q_frac 곡선에 세그먼트 경계 표시
- `4_hi_analysis/hi_plot/<MMDD>/hi_trend.png` — 선별 HI별 용량 열화 추이 (21종)
- `4_hi_analysis/hi_plot/<MMDD>/hi_segment_trend.png` — 6 SoC 구간 × 6 HI 격자 플롯 (전체 셀·사이클, NaN 제외)

#### 4-C. 세그먼트별 상관분석 (`4_hi_analysis/seg_corr_analysis.py`)

6개 SoC 세그먼트 시나리오별로 within-cell Spearman 상관계수를 계산·시각화.  
`hi_correlation.py`의 풀링 방식과 달리 셀 내부 상관계수를 평균내므로 SOH 추적 능력을 더 정확히 반영한다 (→ `docs/correlation_comparison.md` §6 참조).

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--pkl` | `hi_features.pkl` | HI 캐시 경로 (날짜 붙은 최신 파일 자동 탐색) |
| `--dataset` | `all` | 분석 대상: `mit` / `hust` / `all` |
| `--min-cycles` | `5` | 셀당 최소 유효 사이클 수 |
| `--workers` | `8` | 병렬 스레드 수 |

```powershell
# 권장 (4-A 실행 후 캐시에서 로드)
python 4_hi_analysis/seg_corr_analysis.py

# MIT만 분석
python 4_hi_analysis/seg_corr_analysis.py --dataset mit

# 다른 PKL 지정
python 4_hi_analysis/seg_corr_analysis.py --pkl 4_hi_analysis/0622_1154_hi_features.pkl
```

출력 (`4_hi_analysis/seg_corr/<MMDD>/`):
- `corr_rank_<ds>.png` — 6 시나리오 × |ρ| 랭킹 바 차트 (부호 (+)/(-) 병기)
- `corr_matrix_<ds>.png` — 6 시나리오 × feature × feature Spearman 히트맵
- `top7_cross_<ds>.png` — 시나리오 간 Top-7 HI 비교 (HI 아이덴티티 20색 고정)

---

### 분석 도구 — 셀 전체 사이클 시각화 (`4_hi_analysis/plot_cell_cycles.py`)

특정 셀의 전체 사이클을 한 번에 확인하는 스크립트. 이상 셀 조사나 열화 패턴 검토 시 사용.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--dataset` | `mit` | 데이터셋: `mit` / `hust` |
| `--cell` | `b1c0` | 셀 ID |
| `--phase` | `discharge` | 오버레이할 phase: `discharge` / `charge` |

```powershell
# 기본 (MIT b1c0 방전)
python 4_hi_analysis/plot_cell_cycles.py --cell b1c0

# 이상 셀 확인
python 4_hi_analysis/plot_cell_cycles.py --cell b1c18

# 충전 곡선으로 보기
python 4_hi_analysis/plot_cell_cycles.py --cell b1c0 --phase charge

# HUST 셀
python 4_hi_analysis/plot_cell_cycles.py --dataset hust --cell 1-1
```

출력: `4_hi_analysis/cell/cell_cycles_{dataset}_{cell}.png`

2개 패널:
- **위**: 사이클별 방전 용량 (열화 곡선), 색상으로 사이클 진행 표시
- **아래**: 전체 사이클 V-q_frac 오버레이, 초록(초기) → 빨강(말기) 컬러맵

---

### 분석 도구 — 단일 사이클 세그먼트 확인 (`4_hi_analysis/plot_cycle_segments.py`)

특정 셀·사이클 1개를 골라 방전/충전 세그먼트가 V-q_frac 축에서 어떻게 잘리는지 확인하는 **임시 디버그용** 스크립트.  
`hi_segment_viz.py`가 전체 셀 평균 경향을 보여준다면, 이 스크립트는 **개별 사이클 한 장**을 정밀하게 검토할 때 사용.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--dataset` | `mit` | 데이터셋: `mit` / `hust` |
| `--cell` | `b1c0` | 셀 ID |
| `--cycle` | `2` | 사이클 번호 |

```powershell
# 기본 (MIT b1c0, cycle 2)
python 4_hi_analysis/plot_cycle_segments.py

# 다른 셀·사이클 지정
python 4_hi_analysis/plot_cycle_segments.py --cell b1c0 --cycle 50
python 4_hi_analysis/plot_cycle_segments.py --dataset hust --cell 1-1 --cycle 5
```

출력: `4_hi_analysis/segment/segment_{dataset}_{cell}_cycle{N}.png`  
좌측 패널(방전) / 우측 패널(충전) 2칸. 각 패널에 q_frac 0.4 / 0.7 경계선과 SoC 구간 색상 밴드 표시.

---

### 전체 파이프라인 한 번에 실행

`run_pipeline.py`를 사용하면 번호 순서대로 자동 실행.  
스텝 실패 시 계속 진행 여부를 물어봄.

```powershell
# 전체 실행 (Step 1부터)
python run_pipeline.py

# 특정 스텝부터 재실행
python run_pipeline.py 2         # Step 2(이상 사이클 제거)부터
python run_pipeline.py 3         # Step 3(무결성 검사)부터

# 병렬 워커 수 지정
python run_pipeline.py --workers 8
python run_pipeline.py 2 --workers 8
```

각 스텝을 직접 실행하려면:

```powershell
python 1_convert/convert_unified.py --dataset all --workers 3
python 2_preprocess/preprocess.py --workers 4
python 2_preprocess/plot_cleaning_report.py        # 전처리 결과 시각화 (선택)
python 3_integrity/check_integrity.py --workers 4
python 4_hi_analysis/hi_correlation.py             # HI 추출 + hi_features.pkl 캐시 생성
python 4_hi_analysis/hi_segment_viz.py             # 캐시에서 로드
python 4_hi_analysis/seg_corr_analysis.py          # 캐시에서 로드
```

---

## 데이터 디렉토리

```
LFP_SOH_prediction/
  data_raw/
    FastCharge/                    MIT 원본 .mat 파일 (입력)
    our_data/our_data/             HUST 원본 .pkl 파일 (입력)
    MIT/                           파싱 원본 .pkl / .csv  (이상치 제거 없음, DELETE_CELLS 포함)
    HUST/                          파싱 원본 .pkl / .csv  (이상치 제거 없음)
  data_unified/
    MIT/                           변환된 셀별 .pkl / .csv  (필터 적용, Step 1 출력)
    HUST/                          변환된 셀별 .pkl / .csv  (필터 적용, Step 1 출력)
  data_postprocess/
    MIT/                           전처리된 셀별 .pkl + .csv (제거 셀만)
    HUST/                          전처리된 셀별 .pkl + .csv (제거 셀만)
  1_convert/
    convert_unified.py
    mit_batch_parser/
  2_preprocess/
    preprocess.py
    plot_cleaning_report.py        cleaning_report.csv → 필터별 시각화
    outputs/
      cleaning_report.csv          이상치 제거 리포트 (날짜 무관, 항상 덮어씀)
      <MMDD>/                      실행 날짜별 서브폴더
        F4A_dis_gap.png
        F4B_chg_stop.png
        F4C_chg_seg.png
        F5_rolling.png
  3_integrity/
    check_integrity.py
  4_hi_analysis/
    hi_correlation.py
    hi_segment_viz.py
    seg_corr_analysis.py           세그먼트별 within-cell 상관분석
    plot_cell_cycles.py
    plot_cycle_segments.py
    hi_features.pkl                HI 추출 캐시 (4-A/4-B/4-C 공유)
    hi_plot/
      <MMDD>/                      실행 날짜별 서브폴더
        hi_correlation.png         풀링 Spearman 히트맵
        hi_segment_cuts.png        세그먼트 분할 확인
        hi_trend.png               HI 열화 추이
        hi_segment_trend.png       세그먼트별 HI 추이 (전체 데이터)
    seg_corr/
      <MMDD>/                      실행 날짜별 서브폴더
        corr_rank_mit.png          within-cell |ρ| 랭킹
        corr_matrix_mit.png        feature × feature 히트맵
        top7_cross_mit.png         시나리오 간 Top-7 비교
    cell/
      cell_cycles_*.png            셀별 전체 사이클 시각화
    segment/
      segment_*.png                단일 사이클 세그먼트 시각화
  data_HI/
    MIT/                           셀별 HI 특성 .pkl (hi_correlation.py 출력)
    HUST/
  docs/
    PIPELINE.md
    HI_DESCRIPTION.md
    DATASET_ANOMALIES.md           파이프라인 단계별 이상치 처리 정리
    correlation_comparison.md      preprocess.ipynb vs 현재 / 풀링 vs within-cell 비교
    mit_capacity_nonmonotonic.md   MIT capacity 단조성 위반 원인 및 2-pass 필터 설명
    differences.md                 구현 변경점 상세 기록
    HI_OUTLIER_FIXES.md            HI 피처 버그 수정 이력
    hi_overlap_analysis.md         HI 중복·leakage 분석
    DATASET_MIT_README.md          MIT 데이터셋 설명
    DATASET_HUST_README.md         HUST 데이터셋 설명
```
