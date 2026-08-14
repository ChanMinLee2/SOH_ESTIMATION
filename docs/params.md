# 파라미터 정리 (params.md)

`run_pipeline.py`와 `1_convert/`, `2_preprocess/`, `3_integrity/`, `4_hi_analysis/`, `5_model/`
아래 모든 `.py` 스크립트의 CLI 옵션(argparse `add_argument`) 전수 목록. 스크립트별 표로 정리하고,
마지막 열에 **제거 가능 여부를 O(제거 가능)/X(유지 필요)** 로 표시했다.

**판단 기준**
- **X (유지)**: 이 세션 또는 문서화된 실험(`docs/260811_RESULTS.md` 등)에서 실제로 값이 바뀌어 쓰인 이력이 있거나,
  재현성·정합성에 직결되는 파라미터(seed, exclude-cv, skip-shape, seg-axis 등).
- **O (제거 후보)**: (a) `--axis-config` JSON으로 완전히 대체 가능한 단축 인자인데 실사용 이력이 없거나,
  (b) 디버그/일회성 시각화 전용이라 파이프라인 흐름과 무관하거나, (c) 튜닝 후 값이 고정된 채 한 번도
  CLI로 오버라이드된 적 없는 전처리 임계값(상수화 대상)인 경우.
- 스크립트가 `run_pipeline.py`의 Step 1~9에 실제로 연결되어 있는지도 각 절 제목에 표시했다
  (연결 안 된 것은 독립 진단/디버그 도구).

> 참고: `--axis-config` 단축 인자들(`--n1/--n2/--n-samples/--mode/...`)이 원래 존재한 이유는 PowerShell이
> 중첩 따옴표 JSON을 인자로 넘길 때 벗겨먹는 버그 때문이었다. 이 세션에서 `'{\"key\": ...}'` 패턴(외부
> 홑따옴표 + 내부 `\"` 이스케이프)으로 JSON을 직접 넘기는 것이 확실히 동작함을 검증했으므로, **JSON 직접
> 전달이 필요할 때마다 가능**하다. 다만 `q_frac_ref` 계열 커맨드는 이 세션 내내 단축 인자
> (`--n1 --n2 --ref-lag --noise-amp --noise-mode --noise-period`)를 실제로 계속 써왔으므로 그 6개는 X로
> 남겼다 — "대체 가능"과 "실제로 안 쓴다"는 다른 문제.

---

## run_pipeline.py (오케스트레이터, Step 1~9 총괄)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `from_step` (위치인자) | 1 | 시작 스텝 번호 | X |
| `--to-step` | 끝까지 | 종료 스텝 번호(포함) | X |
| `--skip-classifier` | False | Step 7(분류기) 건너뜀 — 이 세션 내내 실사용 | X |
| `--workers` | min(8,CPU) | 데이터 스텝(1~5) 병렬 워커 수 | X |
| `--model-config` | scr.yaml | 모델 학습/평가 yaml 경로 | X |
| `--gates-from` | 자동탐색 | Phase1 run 폴더 수동 지정 (자동탐색 fallback 있음) | O (자동탐색이 대부분 커버, 실패시만 필요) |
| `--checkpoint` | 자동탐색 | 평가용 체크포인트 수동 지정 (fallback 있음) | O (자동탐색이 대부분 커버) |
| `--seg-axis` | None | 세그멘테이션 축 (Step 4~6,8 전달) | X |
| `--axis-config` | None | 축 파라미터 JSON | X |
| `--n1` | None | q_frac_wide/q_frac_ref 구간 크기 단축 | X (q_frac_ref 표준 커맨드에서 실사용) |
| `--n2` | None | q_frac_wide/q_frac_ref 세그먼트 길이 단축 | X (동일) |
| `--n-samples` | None | 구간당 세그먼트 수 단축 | X (random/random_grid/q_frac_ref 전부 실사용) |
| `--mode` | None | vqslope 플래토 검출 모드(dva\|ica) | O (이 세션 내 실사용 이력 없음) |
| `--random-segment` | False | q_frac_wide/vqslope/q_abs 구간 내 고정길이 랜덤 창 | O (실사용 이력 없음) |
| `--seg-len-pts` | None | random-segment 종속 옵션 | O (`--random-segment`와 함께 제거) |
| `--mid-start` | None | q_abs 존 시작 단축 | O (이 세션 내 실사용 이력 없음, q_abs는 축 자체가 별도 실험 단계) |
| `--mid-end` | None | q_abs 존 끝 단축 | O (동일) |
| `--seg-len` | None | q_abs 세그먼트 길이 단축 | O (동일) |
| `--ref-lag` | None | q_frac_ref 레퍼런스 지연 사이클 | X (표준 커맨드 실사용) |
| `--noise-amp` | None | q_frac_ref 노이즈 진폭 | X (동일) |
| `--noise-mode` | None | q_frac_ref 노이즈 방식(ou\|sine) | X (동일) |
| `--noise-period` | None | q_frac_ref 노이즈 특성시간 | X (동일) |
| `--min-pts` | None | 세그먼트 최소 포인트 수 | X (모든 축 실험에서 실질적 통제 변수) |
| `--exclude-cv` | False | CC→CV 전환 후 구간 제외 | X (이번 세션 핵심 이슈, 축 간 일관성 확보용) |
| `--skip-shape` | False | 필터7(형상 이상치) 비활성 데이터 사용 | X (실험 축으로 문서화됨) |
| `--charge-m` / `--discharge-m` | None | probe 상위 m개 오버라이드 | X (m/k 실험 매트릭스 핵심) |
| `--scen-k` | None | 시나리오별 scen HI 수 | X (k=5/15/25/65 실험 전부 이걸로 스윕) |
| `--seed` | None | 모델 초기화 RNG 시드 | X (시드 실험 §10~13 핵심) |
| `--split-seed` | None | train/val/test 분할 시드 | X (동일) |
| `--phase1-lr` | None | Phase1 peak LR 오버라이드 | O (이 세션은 전용 yaml 파일(`exp_*_S.yaml`)로 LR을 관리 — CLI 오버라이드 미사용) |
| `--phase2-lr` | None | Phase2 peak LR 오버라이드 | O (동일 이유) |
| `--with-raw-cnn` | False | Phase2 raw CNN 융합 (REGRESSION_UPGRADE.md §8) | X (문서화된 아키텍처 옵션, 드물어도 구조적으로 유지 필요) |

**33개 중 O 후보 11개.** 특히 `--mode/--random-segment/--seg-len-pts/--mid-start/--mid-end/--seg-len` 6개는
한 세트로 묶여 있어(q_abs·vqslope 전용) 같이 제거하면 깔끔하다.

---

## 1_convert/convert_unified.py (Step 1)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--dataset` | all | mit\|hust\|all | X |
| `--output-root` | OUTPUT_ROOT | 출력 경로 | X (드물지만 경로 재정의 필요시 유일한 수단) |
| `--cell` | None | HUST 단일 셀 ID (디버그용) | O (단일 셀 재변환은 드문 디버그 상황) |
| `--workers` | 3 | 병렬 프로세스 수 | X |
| `--no-cache` | False | 캐시 무시 재변환 | X (원본 재처리 필요할 때 유일한 스위치) |

---

## 2_preprocess/preprocess.py (Step 2)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--dataset` | all | mit\|hust\|all | X |
| `--dis-gap-s` | 600.0 | [필터4] 방전 단절 절대 기준(초) | O (튜닝 완료 후 고정값, CLI 오버라이드 이력 없음) |
| `--dis-gap-factor` | 50.0 | [필터4] 방전 단절 배율 기준 | O (동일) |
| `--chg-gap-s` | 600.0 | [필터4] 충전 중단 절대 기준(초) | O (동일) |
| `--chg-gap-factor` | 50.0 | [필터4] 충전 중단 배율 기준 | O (동일) |
| `--chg-seg-gap-s` | 120.0 | [필터4] CC 전환 갭 절대 기준 | O (동일) |
| `--chg-seg-gap-factor` | 30.0 | [필터4] CC 전환 갭 배율 기준 | O (동일) |
| `--window` | 11 | [필터5] Rolling Median 윈도우 | O (동일) |
| `--sigma` | 2.0 | [필터5] 이상치 σ 임계값 | O (동일) |
| `--min-std` | 0.01 | [필터5] std 플로어 | O (동일) |
| `--vend-min` | 1.8 | [필터6] 방전 종지전압 하한 | O (동일) |
| `--shape-sigma` | 30.0 | [필터7] 형상 편차 임계값 | O (동일 — 다만 완전 삭제보다 코드 상수 승격 권장) |
| `--shape-window` | 11 | [필터7] 기준곡선 윈도우 | O (동일) |
| `--shape-grid` | 100 | [필터7] q_frac 보간 격자 수 | O (동일) |
| `--skip-shape` | False | 필터7 완전 비활성화 | X (문서화된 실험 축, `_noshape` 데이터 생성 스위치) |
| `--workers` | min(4,CPU) | 병렬 프로세스 수 | X |

**16개 중 O 후보 12개** — 필터4/5/6/7 임계값 12개가 전부 "한 번 튜닝하고 다시 안 건드린" 상수형
파라미터. `--skip-shape`만 실험 축으로 실사용되므로 남기고, 나머지는 코드 상단 상수로 승격해
CLI에서 빼는 걸 권장.

---

## 2_preprocess/plot_cleaning_report.py (독립 진단 도구, 파이프라인 미연결)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--window` | 11 | F5 rolling median 윈도우(리포트용) | O (preprocess.py와 같은 값 사용, 별도 오버라이드 이력 없음) |
| `--sigma` | 2.5 | F5 이상치 σ | O (동일) |
| `--min-std` | 0.01 | F5 std 플로어 | O (동일) |
| `--top-rolling` | 9 | Rolling 상세 표시 셀 수 | O (cosmetic) |
| `--workers` | min(16,CPU) | PKL 병렬 스캔 스레드 수 | X |

---

## 2_preprocess/plot_shape_filter_debug.py (독립 진단 도구)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--cell` | None | MIT 셀 ID (미지정시 전체) | X (디버그 도구 본연의 대상 지정) |
| `--sigma` | [5.0] | z-score 임계값(복수 가능) | O (필터7 튜닝 종료 후 고정) |
| `--window` | 11 | rolling median 윈도우 | O (동일) |
| `--grid` | 100 | q_frac 보간 격자 수 | O (동일) |

---

## 3_integrity/check_integrity.py (Step 3)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--workers` | min(4,CPU) | 병렬 프로세스 수 | X |

가장 군더더기 없는 스크립트 — 그대로 유지.

---

## 4_hi_analysis/hi_correlation.py (Step 4) — 이 세션 최대 파라미터 보유(28개)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--workers` | CPU-2 | 병렬 프로세스 수 | X |
| `--n-top` | 4 | 산점도 표시 상위 HI 수 | O (cosmetic) |
| `--force` | False | 캐시 무시 재추출 | X (run_pipeline이 Step4에 항상 주입, 독립 실행시도 필요) |
| `--plateau-debug` | False | 단일 사이클 플래토 디버그 플롯 | O (일회성 디버그) |
| `--plateau-summary` | False | 전체 plateau_frac 요약 플롯 | O (일회성 디버그) |
| `--dataset` | MIT | MIT\|HUST | X |
| `--cell` | "" | 셀 ID(디버그 플롯 대상) | O (디버그 전용) |
| `--cycle` | 0 | 사이클 번호(디버그 플롯 대상) | O (디버그 전용) |
| `--curve-debug` | False | 6세그먼트×5커브 시각화 | O (일회성 디버그) |
| `--cycles` | "" | curve-debug 대상 사이클 목록 | O (`--curve-debug` 종속) |
| `--n-cycles` | 5 | curve-debug 자동 선택 개수 | O (`--curve-debug` 종속) |
| `--seg-axis` | qfrac | 세그멘테이션 축 | X |
| `--axis-config` | {} | 축 파라미터 JSON | X |
| `--n1` | None | q_frac_wide/ref 구간 크기 | X (실사용) |
| `--n2` | None | q_frac_wide/ref 세그먼트 길이 | X (실사용) |
| `--n-samples` | None | 구간당 세그먼트 수 | X (실사용) |
| `--mode` | None | vqslope dva\|ica | O (실사용 이력 없음) |
| `--random-segment` | False | 구간 내 고정길이 랜덤 창 | O (실사용 이력 없음) |
| `--seg-len-pts` | None | random-segment 종속 | O (동일) |
| `--mid-start` | None | q_abs 존 시작 | O (실사용 이력 없음) |
| `--mid-end` | None | q_abs 존 끝 | O (동일) |
| `--seg-len` | None | q_abs 세그먼트 길이 | O (동일) |
| `--ref-lag` | None | q_frac_ref 레퍼런스 지연 | X (실사용) |
| `--noise-amp` | None | q_frac_ref 노이즈 진폭 | X (실사용) |
| `--noise-mode` | None | q_frac_ref 노이즈 방식 | X (실사용) |
| `--noise-period` | None | q_frac_ref 노이즈 특성시간 | X (실사용) |
| `--min-pts` | None | 세그먼트 최소 포인트 수 | X (모든 축 실험 통제 변수) |
| `--exclude-cv` | False | CV 구간 제외 | X (핵심) |
| `--skip-shape` | False | `_noshape` 데이터 사용 | X (핵심) |

**28개 중 O 후보 12개** — `--plateau-debug/--plateau-summary/--cell/--cycle/--curve-debug/--cycles/
--n-cycles` 7개는 디버그 전용 묶음, `--mode/--random-segment/--seg-len-pts/--mid-start/--mid-end/
--seg-len` 6개는 vqslope/q_abs 전용 묶음(run_pipeline.py와 동일 사유). 디버그 묶음은 별도
`hi_correlation_debug.py`로 분리하거나 완전히 `seg_diagnose.py`/`hi_segment_viz.py` 쪽으로 이관하는
것도 방법.

---

## 4_hi_analysis/hi_segment_viz.py (Step 5)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--workers` | 4 | HI 추출 병렬 워커 수 | X |
| `--n-cycles` | 4 | 세그먼트 cuts 플롯 대표 사이클 수 | O (cosmetic) |
| `--force` | False | 캐시 무시 재추출 | X |
| `--seg-axis` | qfrac | 세그멘테이션 축 | X |
| `--axis-config` | {} | 축 파라미터 JSON | X |
| `--n1` | None | 구간 크기 단축 | O (Step5는 시각화 전용, run_pipeline이 axis-config로만 전달) |
| `--n2` | None | 세그먼트 길이 단축 | O (동일) |
| `--n-samples` | None | 세그먼트 수 단축 | O (동일) |
| `--mode` | None | vqslope dva\|ica | O (실사용 이력 없음) |
| `--random-segment` | False | 고정길이 랜덤 창 | O (실사용 이력 없음) |
| `--seg-len-pts` | None | random-segment 종속 | O (동일) |

참고: Step 5는 순수 시각화 산출물만 만들고 Step 6 이후가 그 결과물을 읽지 않는다(이전 답변 참고).
파라미터를 줄이기보다 **스텝 자체를 선택적으로 건너뛰는 것**(`run_pipeline.py`에 `--skip-viz` 같은
플래그 추가)이 더 효과적일 수 있다.

---

## 4_hi_analysis/seg_diagnose.py (독립 진단 도구 — 이 세션에서 실사용)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--seg-axis` | qfrac | 세그멘테이션 축 | X |
| `--axis-config` | {} | 축 파라미터 JSON | X |
| `--dataset` | MIT | 통계 스캔 대상 | X |
| `--cell` | "" | 사이클 플롯 대상 셀 | X (`--no-stats` 단일 셀 점검에 실사용) |
| `--cycle` | 0 | 사이클 플롯 대상 번호 | X (동일) |
| `--mode` | segment | segment\|ic\|vqzone\|compare\|all | X (`compare` 모드 이 세션에서 실사용) |
| `--compare-config` | None | compare 모드 조건 JSON 경로 | X (`compare_random_vs_qfref.json` 실사용) |
| `--n-cycles` | 6 | IC 모드 대표 사이클 수 | O (cosmetic) |
| `--n-random` | 10 | segment 시각화용 랜덤 셀 수 | O (기본값 그대로 사용) |
| `--seed` | 42 | 랜덤 선택 재현성 시드 | X (재현성 원칙상 유지) |
| `--no-stats` | False | 통계 수집 생략 | X (무거운 전수 스캔 회피용으로 이 세션에서 실사용) |
| `--no-plot` | False | 사이클 시각화 생략 | O (단독 사용 이력 없음) |
| `--survival-stats` | False | q_frac_wide 생존율 통계 | O (니치한 일회성 분석) |
| `--n1`/`--n2`/`--n-samples` | None | `--survival-stats` 전용 단축 | O (`--survival-stats`와 함께 제거) |

---

## 4_hi_analysis/seg_corr_analysis.py (독립 진단 도구)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--pkl` | "" | PKL 파일 경로 수동 지정 | O (axis 기반 자동 선택으로 대부분 충분) |
| `--seg-axis` | qfrac | 세그멘테이션 축 | X |
| `--axis-config` | {} | 축 파라미터 JSON | X |
| `--dataset` | all | all\|mit\|hust | X |
| `--min-cycles` | 5 | 셀당 최소 사이클 수 | O (고정값 사용) |
| `--workers` | 8 | 병렬 프로세스 수 | X |
| `--top-n` | 5 | top_cross 플롯 상위 HI 개수 | O (cosmetic) |

---

## 4_hi_analysis/plot_cycle_segments.py (독립 진단 도구, 미니멀)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--dataset` | mit | mit\|hust | X |
| `--cell` | b1c0 | 셀 ID | X |
| `--cycle` | 2 | 사이클 번호 | X |

3개뿐 — 불필요한 옵션 없음, 손댈 필요 없음.

---

## 4_hi_analysis/plot_cell_cycles.py (독립 진단 도구 — 임계값 과다)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--dataset` | mit | mit\|hust | X |
| `--cell` | b1c0 | 셀 ID | X |
| `--z-thresh` | 6.0 | 제거 후보 z 임계값 | O (튜닝 종료 후 고정) |
| `--dev-thresh-discharge` | 0.25 | 방전 전압 RMSE 임계값 | O (동일) |
| `--dev-thresh-charge` | 0.07 | 충전 전압 RMSE 임계값 | O (동일) |
| `--maxdev-thresh-discharge` | 0.40 | 방전 국소 최대편차 임계값 | O (동일) |
| `--maxdev-thresh-charge` | 0.25 | 충전 국소 최대편차 임계값 | O (동일) |
| `--flat-thresh-discharge` | 0.0(끔) | 방전 평탄선 제거 하한 | O (기본 비활성) |
| `--flat-thresh-charge` | 0.0(끔) | 충전 평탄선 제거 하한 | O (동일) |
| `--fhigh-thresh-discharge` | 0.0(끔) | 방전 고전압 체류 상한 | O (동일) |
| `--fhigh-thresh-charge` | 0.0(끔) | 충전 고전압 체류 상한 | O (기본 비활성, 코드 주석에도 "개별 사이클은 수동 목록 사용" 명시) |

**11개 중 O 후보 9개** — 이미 코드 주석에서 절반은 "기본 비활성/수동 목록 권장"이라 밝히고 있다.
임계값 9개를 스크립트 상단 상수로 옮기고 CLI에서는 `--dataset/--cell`만 남기는 걸 권장.

---

## 4_hi_analysis/plot_all_mit_cells.py (독립 진단 도구, 미니멀)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--workers` | 1 | 동시 실행 프로세스 수 | X |

1개뿐 — 손댈 필요 없음.

---

## 4_hi_analysis/profile_hi_timing.py (독립 진단 도구, 미니멀)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--n-cells` | 8 | 프로파일링 대상 셀 수 | X |
| `--n-cycles` | 30 | 셀당 샘플링 사이클 수 | X |
| `--workers` | 1 | 병렬 프로세스 수 | X |

3개 전부 프로파일링 본연의 파라미터 — 유지.

---

## 5_model/train_scr.py (Step 6, 8) — 두 번째로 파라미터 많음(18개)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--config` | scr.yaml | 모델 설정 yaml | X |
| `--phase` | None | 1(Gate 학습)\|2(정밀 학습) | X |
| `--charge-m` | None | 충전 probe 상위 m개 | X |
| `--discharge-m` | None | 방전 probe 상위 m개 | X |
| `--scen-k` | None | 시나리오별 scen HI 수 | X |
| `--gates-from` | None | Phase2용 Phase1 run 경로 | X (자동탐색 실패시 필수 수단) |
| `--no-gates` | False | [legacy] `--phase 1`과 동일 | O (코드 주석에 이미 `[legacy]`로 명시, `--phase 1`로 완전 대체됨) |
| `--device` | auto | 연산 장치 | X |
| `--seg-axis` | None | 세그멘테이션 축 | X |
| `--axis-config` | None | 축 파라미터 JSON | X |
| `--seed` | None | 모델 초기화 RNG 시드 | X |
| `--split-seed` | None | 데이터 분할 시드 | X |
| `--lr` | None | peak LR 오버라이드 | O (이 세션은 전용 yaml로 LR 관리, CLI 오버라이드 미사용) |
| `--exclude-cv` | False | `_ccOnly` 데이터 사용 | X |
| `--skip-shape` | False | `_noshape` 데이터 사용 | X |
| `--with-raw-cnn` | False | Phase2 raw CNN 융합 | X (문서화된 구조적 옵션) |
| `--raw-cnn-pretrained-from` | None | 분류기 CNN 재사용 경로 | X (`--with-raw-cnn`의 방안 a/b 분기에 필요) |
| `--with-raw-flat` | False | Phase2 raw flatten 융합 | O (REGRESSION_UPGRADE.md에 대안으로만 언급, 이 세션 실사용/문서화된 결과 없음) |

**18개 중 O 후보 3개** — `--no-gates`는 코드 스스로 legacy라고 밝히고 있어 가장 확실한 제거 대상.

---

## 5_model/train_classifier.py (Step 7)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--config` | scr.yaml | 모델 설정 yaml | X |
| `--run-dir` | 자동탐색 | Phase1 run 폴더(분류기 저장 위치) | X |
| `--device` | auto | 연산 장치 | X |
| `--epochs` | None | 최대 에폭 오버라이드 | O (yaml로 충분, CLI 오버라이드 이력 없음) |
| `--lr` | None | 학습률 오버라이드 | O (동일) |
| `--d-hidden` | 64 | MLPProbeClassifier hidden dim | O (아키텍처 상수에 가까움, 튜닝 이력 없음) |
| `--exclude-cv` | False | `_ccOnly` 데이터 사용 | X |
| `--skip-shape` | False | `_noshape` 데이터 사용 | X |

---

## 5_model/test_scr.py (Step 9)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--config` | scr.yaml | 모델 설정 yaml | X |
| `--checkpoint` | None | 평가 체크포인트 경로 | X |
| `--classifier-ckpt` | 자동탐색 | 분류기 체크포인트(hard routing용) | X |
| `--rep-cells` | None | 대표 셀 지정(곡선 비교용) | O (미지정시 자동 선정 로직 존재, 자동값으로 대부분 충분) |
| `--device` | auto | 연산 장치 | X |
| `--charge-m`/`--discharge-m` | None | probe m 오버라이드(체크포인트와 일치 필요) | X |
| `--scen-k` | None | scen k 오버라이드 | X |
| `--cat-top-k` | None | HI 카테고리 히트맵 상위 랭크 수 | O (cosmetic) |

---

## 5_model/visualize_results.py (`result_comparison`)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--runs` | 필수 | 비교할 run 폴더 목록(2개 이상) | X |
| `--labels` | None | run별 표시 이름 | O (폴더명으로 대체 가능) |
| `--with-jacobian` | False | Jacobian 코사인 유사도 패널 | O (느려서 거의 안 씀 — 코드 주석도 "느림" 명시) |
| `--checkpoint-name` | best.pt | 사용할 체크포인트 파일명 | O (사실상 항상 best.pt) |
| `--infer-batch-size` | 256 | 추론 벤치마크 배치 크기 | O (벤치마크 세부값, 튜닝 이력 없음) |
| `--infer-warmup` | 10 | 추론 벤치마크 워밍업 반복 | O (동일) |
| `--infer-reps` | 50 | 추론 벤치마크 반복 횟수 | O (동일) |
| `--jacobian-max-samples` | 300 | Jacobian 계산 최대 샘플 | O (`--with-jacobian` 종속) |
| `--device` | auto | 연산 장치 | X |
| `--out-name` | None | 결과 폴더명 전체 오버라이드 | O (`--title`과 기능 중복) |
| `--title` | None | 결과 폴더명 일부 오버라이드 | X (`--out-name`보다 가볍게 자주 씀) |
| `--rep-cells` | None | SOH 곡선 비교용 셀 직접 지정 | O (미지정시 자동 선정으로 대부분 충분) |

**12개 중 O 후보 8개** — 추론 벤치마크 세부 파라미터(`--infer-*`) 4개와 Jacobian 관련 2개가 특히
거의 안 건드리는 값들.

---

## 5_model/clustering.py (독립 진단 도구)

| 옵션 | 기본값 | 설명 | 제거가능 |
|---|---|---|---|
| `--k-max` | 9 | K-Means 최대 k | O (고정값 사용) |
| `--n-init` | 5 | MiniBatchKMeans n_init | O (동일) |
| `--knn` | 10 | KNN Purity 이웃 수 | O (동일) |
| `--mrmr-m` | 20 | mRMR 선택 HI 최대 수 | O (동일) |
| `--no-plot` | False | 플롯 생략 | X (배치 실행시 유용) |
| `--n-jobs` | -1 | joblib 병렬 수 | X |
| `--out` | docs/_clustering_results.json | 결과 JSON 경로 | X |

---

## 5_model/hi_compute.py

CLI 인자 없음 — 다른 스크립트가 임포트해서 쓰는 모듈. 해당 없음.

---

## 요약

| 폴더/스크립트 | 총 파라미터 | O(제거후보) | 비고 |
|---|---:|---:|---|
| run_pipeline.py | 33 | 11 | q_abs/vqslope 단축 인자 6개 + LR 오버라이드 2개 + gates-from/checkpoint 2개 + mode 1개 |
| 1_convert/convert_unified.py | 5 | 1 | |
| 2_preprocess/preprocess.py | 16 | 12 | 필터4~7 임계값 상수화 권장 |
| 2_preprocess/plot_cleaning_report.py | 5 | 4 | |
| 2_preprocess/plot_shape_filter_debug.py | 4 | 3 | |
| 3_integrity/check_integrity.py | 1 | 0 | 모범 사례 |
| 4_hi_analysis/hi_correlation.py | 28 | 12 | 최다 보유, 디버그 묶음 분리 권장 |
| 4_hi_analysis/hi_segment_viz.py | 11 | 5 | Step 자체를 skip 가능하게 하는 게 더 효과적 |
| 4_hi_analysis/seg_diagnose.py | 16 | 5 | |
| 4_hi_analysis/seg_corr_analysis.py | 7 | 3 | |
| 4_hi_analysis/plot_cycle_segments.py | 3 | 0 | 모범 사례 |
| 4_hi_analysis/plot_cell_cycles.py | 11 | 9 | 임계값 상수화 권장(코드 주석도 이미 그렇게 말함) |
| 4_hi_analysis/plot_all_mit_cells.py | 1 | 0 | 모범 사례 |
| 4_hi_analysis/profile_hi_timing.py | 3 | 0 | |
| 5_model/train_scr.py | 18 | 3 | `--no-gates`는 코드 스스로 legacy 명시 |
| 5_model/train_classifier.py | 8 | 3 | |
| 5_model/test_scr.py | 8 | 2 | |
| 5_model/visualize_results.py | 12 | 8 | 벤치마크/Jacobian 세부값 다수 |
| 5_model/clustering.py | 7 | 4 | |
| 5_model/hi_compute.py | 0 | — | CLI 없음(모듈) |
| **합계** | **197** | **85** | |

**우선 제거 추천 (확실도 높은 순)**
1. `train_scr.py --no-gates` — 코드에 `[legacy]`로 명시, `--phase 1`로 완전 대체됨.
2. `preprocess.py`/`plot_cell_cycles.py`의 임계값류 21개 — 튜닝 이후 고정값, CLI로 바꾼 이력 전무.
   코드 상단 상수로 승격.
3. `hi_correlation.py`/`run_pipeline.py`의 vqslope·q_abs 전용 단축 인자 12개(`--mode/--random-segment/
   --seg-len-pts/--mid-start/--mid-end/--seg-len` × 2곳) — `--axis-config` JSON 하나로 통일.
4. `visualize_results.py`의 추론 벤치마크/Jacobian 세부 파라미터 6개 — 코드 내부 상수로.

---

# config.yaml 구조 개편안: main.yaml / fixed.yaml 분리

CLI 파라미터 정리와 별개로, `5_model/config/*.yaml` 쪽도 같은 문제(불필요하게 넓은 표면적)를
안고 있다. 현재 `5_model/config/`에는 15개 yaml 파일(`scr.yaml` + `exp_*.yaml` 14개)이 있는데,
`scr.yaml`과 나머지를 실제로 `diff` 떠보면 **파일마다 진짜 다른 값은 5~8개 필드뿐**이고 나머지
60줄 이상은 토씨 하나 안 틀리고 동일하다(주석 유무 정도만 다름). 즉 실험 하나 새로 팔 때마다
127줄짜리 파일을 통째로 복사해서 그 중 5줄만 고치는 식으로 굴러가고 있다 — 이게 "프로젝트 구조를
몰라도 실행 옵션을 편하게 줄 수 있게" 하고 싶다는 요구와 정반대 방향이다.

## 분리 기준 (실제 diff 근거)

`exp_qfw_mlp_basefix/cnn/flat/mit_only`, `exp_full_cycle`, `exp_random_none`을 `scr.yaml`과 비교한
결과, 실제로 실험마다 바뀌는 필드는 다음과 같다:

| 바뀐 필드 | 관찰된 예시 |
|---|---|
| `scenario.axis` / `scenario.axis_config` | q_frac_wide → full_cycle / random / random_grid |
| `data.gates_from` | 이전 run 경로 재사용 vs `null`(자체 Phase1 새로 학습) |
| `data.datasets` (+ `data.forced_test_cells`) | `["MIT","HUST"]` → `["MIT"]`만(mit_only 실험, 이때 기존 baseline의 test 셀 40개 중 MIT 20개를 `forced_test_cells`로 고정해 비교 가능하게 함) |
| `classifier.charge_probe_m` / `discharge_probe_m` | 5/5 → 10/10 (m 실험) |
| `regression.scen_k_count` | 15 → 55 → 65 (k 실험) |
| `model.regression_model` | mlp → transformer/resnet_tab/cnn 등 |
| `model.mlp_hidden_dims` | `[512,256,128,64]`(MLP_L) → `[128,64]`(MLP_S) |
| `model.with_raw_cnn` / `with_raw_flat` | false → true (raw 융합 실험) |
| `training.lr` / `weight_decay` / `early_stop_patience` | 5.0e-4/1e-4/30(baseline) → 2e-4/1e-3/100(MLP_S 튜닝) |

이 필드들이 그대로 **`main.yaml`(실험별 가변 요소)** 후보다. 반대로 `data.is_real_input`,
`data.split_ratio`류, `classifier.type`/`is_auto_mk_selection`, `model.d_probe`/`d_head`/`dropout`/
`tr_*`/`resnet_*`, `loss.*`(L0 스케줄), `training.epochs`/`batch_size`/`scheduler`/`grad_clip`,
`evaluation.*`, `uq.*`, `test.*`는 15개 yaml 전부에서 **단 한 글자도 다르지 않았다** — `fixed.yaml`
후보.

## 제안 구조

**`5_model/config/fixed.yaml`** (공통 1벌, 실험 간 거의 안 바뀜)
```yaml
data:
  is_real_input: false
  data_dir: null
  seg_data_dir: null
  output_dir: "_5_data_model_scr"
  is_cross_dataset_evaluate: false
  split_seed: 42
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  min_cycles_per_cell: 10
  io_workers: 16
  use_initial_capacity: true
  nominal_capacities: { MIT: 1.1, HUST: 1.2 }
classifier:
  type: cnn
  is_auto_mk_selection: false
  probe_m_count: 1
model:
  d_probe: 64
  d_head: 128
  dropout: 0.2
  tr_n_heads: 4
  tr_n_layers: 2
  tr_d_ff: 512
  resnet_n_blocks: 4
  resnet_d_hidden_factor: 2.0
  raw_cnn_pretrained_from: null
loss:
  lambda_scen: 0.01
  lambda_l0: 0.01
  lambda_l0_auto: true
  lambda_l0_schedule: "delayed_warmup"
  lambda_l0_warmup_epochs: 50
  lambda_l0_ramp_epochs: 100
  leak_cols: ["stat_q_abs", "stat_energy_seg"]
training:
  epochs: 500
  batch_size: 2048
  scheduler: "cosine"
  warmup_epochs: 10
  grad_clip: 1.0
  log_interval: 10
  run_overfit_test: false
  overfit_test_samples: 1024
  overfit_test_epochs: 300
evaluation:
  metrics: ["rmse", "mae", "r2", "mape"]
  rep_cells_per_dataset: 5
uq:
  enabled: true
  prior_precision: 1.0
  optimize_prior: true
  noise_std: null
test:
  random_segment_test: false
  random_seg_data_dir: "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/test_rs/seg"
  random_seg_datasets: ["MIT", "HUST"]
```

**`5_model/config/main_<실험명>.yaml`** (실험별 8~10줄, 사람이 매번 손으로 채우는 부분)
```yaml
scenario:
  axis: random_grid
  axis_config: { n_samples: 12, window: 0.2, assign: "none", seed: 42, min_pts: 10, axis_name: "random_grid", placement: "grid" }
data:
  gates_from: null
  datasets: ["MIT", "HUST"]
classifier:
  charge_probe_m: 10
  discharge_probe_m: 10
regression:
  scen_k_count: 25
model:
  regression_model: "mlp"
  mlp_hidden_dims: [128, 64]
  with_raw_cnn: false
  with_raw_flat: false
training:
  lr: 2.0e-4
  weight_decay: 1.0e-3
  early_stop_patience: 100
```

## 구현 지점 (코드 변경 범위는 생각보다 작다)

yaml 로딩이 이미 한 곳으로 모여 있다 — [5_model/utils/io_utils.py:12](5_model/utils/io_utils.py#L12)의
`load_config()` 하나뿐이고, `train_scr.py`(413행) / `train_classifier.py` / `test_scr.py` 세 스크립트가
전부 이 함수만 거쳐서 yaml을 읽는다. 즉:

1. `load_config(main_path, fixed_path=None)`으로 시그니처 확장 — `fixed_path` 생략 시 지금처럼 단일
   파일 그대로 읽는 기존 동작 100% 유지(하위호환). 지정 시 `fixed.yaml`을 베이스로 로드하고
   `main.yaml`을 얕은 딥머지로 덮어씀(섹션 단위 dict merge — `scenario`/`data`처럼 한쪽에만 있는
   키는 합집합, 겹치는 키는 main이 승리).
2. `run_pipeline.py --model-config`를 `--main-config`로 이름을 바꾸거나(또는 그대로 두고) 새
   `--fixed-config`(기본값 `5_model/config/fixed.yaml`)를 추가해 Step 6/7/8/9로 그대로 전달.
3. `save_config()`(train_scr.py가 run_dir에 저장하는 스냅샷)는 **merge된 결과**를 그대로 저장해야
   재현성이 깨지지 않는다 — `visualize_results.py:165`가 `run_dir/config.yaml` 하나만 읽어 실험을
   복원하므로, 이 스냅샷은 지금처럼 계속 완전한 단일 파일이어야 한다.

## 마이그레이션

기존 `exp_*.yaml` 14개 → 공통부는 `fixed.yaml` 하나로 합치고, 각 파일은 위 표의 가변 필드만 남긴
`main_*.yaml`로 축소(127줄 → 8~15줄). 이번 세션에서 만든 `exp_random_S.yaml` /
`exp_qfw_mlp_S.yaml` / `exp_random_grid_S.yaml` 3개가 사실상 이 구조 전환의 첫 후보 — 세 파일 다
바뀐 필드가 정확히 위 "main.yaml 후보" 표에 들어맞는다.

**제거가능 표기는 보류** — 이 구조 개편은 "필드를 없앤다"가 아니라 "같은 필드를 어느 파일에
두느냐"의 재배치라서 O/X 이분법이 안 맞는다.

**2026-08-14 구현 완료.** `5_model/utils/io_utils.py`의 `load_config()`가 `fixed_path` 인자를
받도록 확장됐고(생략 시 `config_path`와 같은 폴더의 `fixed.yaml`을 자동 탐색해 병합, 없으면
기존과 동일하게 단독 사용), `5_model/config/fixed.yaml` + `5_model/config/main.yaml`을 실제로
만들었다. `main.yaml`은 q_frac_ref 표준 베이스라인(n1=35%, n2=20%, k=65, m=10/10, MLP_L,
seed=42)을 그대로 담고 있다. 검증 내용:

- 기존 15개 yaml(`scr.yaml` + `exp_*.yaml` 14개) 전부에 대해 "이 병합이 기존 값을 하나라도
  바꾸는지" 재귀 비교 — 전부 무손실 확인(새로 채워지는 키는 원래 코드에서도 `.get()`으로만
  읽혀 `None`과 동치였던 것들뿐).
- `train_scr.py`/`train_classifier.py`/`test_scr.py`의 `cfg["..."]["..."]` 형태 bracket 접근을
  전수 grep — 전부 `fixed.yaml`+`main.yaml` 병합 결과에 존재하는 키였음(구조적 호환 확인).
- `main.yaml` 병합 결과가 사용자가 준 재현 커맨드(`--seg-axis q_frac_ref --n1 0.35 --n2 0.20
  --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65
  --seed 42 --split-seed 42`)와 정확히 일치하는지 JSON dump로 육안 대조.
- `save_config()` round-trip(로드→저장→재로드) 일치 확인 — run_dir에 저장되는 config.yaml
  스냅샷은 항상 병합 후 완결형이라 재현성에 문제 없음.
- 실제 GPU 학습(Phase1/2)까지는 돌리지 않았다 — 필요하면 `python run_pipeline.py 6
  --model-config 5_model/config/main.yaml --workers 40 --skip-classifier`로 짧게 확인 가능.

---

# CLI 파라미터 제거 결과 (1차: run_pipeline.py + Step 1~9 요소 스크립트)

위 표에서 O로 표시된 항목 중 `run_pipeline.py`와 그 Step 1~9 요소 스크립트(`1_convert/convert_unified.py`,
`2_preprocess/preprocess.py`, `3_integrity/check_integrity.py`, `4_hi_analysis/hi_correlation.py`,
`4_hi_analysis/hi_segment_viz.py`, `5_model/train_scr.py`, `5_model/train_classifier.py`,
`5_model/test_scr.py`) 8개 파일분을 전부 제거했다. 독립 진단 도구(`seg_diagnose.py`,
`visualize_results.py`, `clustering.py`, `plot_*.py` 등)는 이번 1차 작업 범위 밖이라 손대지 않았다.

**총 53개 CLI 옵션 제거.** (원래 표의 스크립트별 O 개수는 집계 과정에서 일부 오차가 있었다 —
실제 행 단위로 다시 세어 아래 숫자로 정정)

| 스크립트 | 제거한 옵션 | 처리 방식 |
|---|---|---|
| `run_pipeline.py` | `--gates-from` `--checkpoint` `--mode` `--random-segment` `--seg-len-pts` `--mid-start` `--mid-end` `--seg-len` `--phase1-lr` `--phase2-lr` (10개) | gates-from/checkpoint는 자동탐색 로직만 남기고 `args.xxx or` 분기 제거(코드 단순화). 나머지는 그대로 삭제 |
| `1_convert/convert_unified.py` | `--cell` (1개) | 삭제. `convert_mit/convert_hust`의 `target_cell` 파라미터는 원래도 기본값 `None`이라 함수 시그니처는 그대로 둠(프로그램적 재사용은 유지) |
| `2_preprocess/preprocess.py` | 필터4~7 임계값 13개(`--dis-gap-s/-factor` `--chg-gap-s/-factor` `--chg-seg-gap-s/-factor` `--window` `--sigma` `--min-std` `--vend-min` `--shape-sigma/-window/-grid`) | 삭제 후 `main()` 안에 튜닝 완료 값으로 지역 상수화(`w, s, m, vm = 11, 2.0, 0.01, 1.8` 등) — 동작 100% 동일 |
| `4_hi_analysis/hi_correlation.py` | `--n-top` `--plateau-debug` `--plateau-summary` `--cell` `--cycle` `--curve-debug` `--cycles` `--n-cycles` `--mode` `--random-segment` `--seg-len-pts` `--mid-start` `--mid-end` `--seg-len` (14개) | CLI만 벗겨내지 않고, 어디서도 참조되지 않게 된 디버그 전용 코드(`plot_plateau_debug`, `plot_plateau_fraction_summary`, `plot_curve_debug`, `_run_plateau_debug`, `_run_curve_debug`, `_compute_plateau_fracs`, `_cell_plateau_worker` 등) 약 900줄(구 941~1844행)을 통째로 삭제. 다른 파일에서 이 함수들을 import하는 곳이 없음을 grep으로 먼저 확인 후 진행 — 파일이 2922→2018줄로 줄었다 |
| `4_hi_analysis/hi_segment_viz.py` | `--n-cycles` `--n1` `--n2` `--n-samples` `--mode` `--random-segment` `--seg-len-pts` (7개) | 삭제. `n_cycles`는 호출부에서 기존 기본값 4로 하드코딩 |
| `5_model/train_scr.py` | `--no-gates` `--lr` `--with-raw-flat` (3개) | 삭제. `--no-gates`는 `_resolve_phase()`의 legacy 분기까지 함께 제거(`--phase 1`로 완전 대체됐다고 코드 스스로 명시했던 부분) |
| `5_model/train_classifier.py` | `--epochs` `--lr` `--d-hidden` (3개) | 삭제 후 각각 yaml 기본값 사용 또는 상수화(`d_hidden = 64`) |
| `5_model/test_scr.py` | `--rep-cells` `--cat-top-k` (2개) | 삭제. `rep_cells`는 yaml `evaluation.rep_cells` → 자동 선정(`_pick_rep_cells`) 순서로 폴백, `cat_top_k`는 `None`(전체 표시)으로 고정 |

**검증**: 8개 파일 전부 `py_compile` 통과 + 프로젝트 conda 환경(`LFP_SOH_ESTIMATION`)에서
`--help` 정상 출력 확인. 제거된 각 옵션에 대해 `args.<옵션명>` 잔여 참조가 없는지 grep으로
재확인했다.

**설계 판단 메모**
- `run_pipeline.py`의 `--gates-from`/`--checkpoint`는 최상위 CLI에서만 제거했다. `train_scr.py
  --gates-from`, `train_classifier.py --run-dir`, `test_scr.py --checkpoint`처럼 하위 스크립트를
  **직접** 실행할 때 쓰는 동명의 옵션은 그대로 유지된다(그쪽은 자동탐색 fallback이 없는 경우도
  있어 X로 남겨둔 항목). `run_pipeline.py`를 거칠 때만 자동탐색이 항상 우선한다.
- `preprocess.py`/`train_classifier.py`의 임계값·하이퍼파라미터는 **상수화**했지 기능을 없애지
  않았다 — 값을 또 바꿔야 하는 상황이 오면 코드 상단 상수를 고치면 된다(재도입 비용이 한 줄
  수준으로 낮음).
- `hi_correlation.py`만 예외적으로 코드 자체를 삭제했다. 다른 파일들의 O 항목은 대부분 "값
  하나"였지만, 이 파일의 디버그 플래그들은 뒤에 200~900줄짜리 플로팅 함수가 딸려 있어 플래그만
  없애면 죽은 코드가 그대로 남아 오히려 직관성을 해쳤기 때문. 필요해지면 git 이력에서 복구 가능.
