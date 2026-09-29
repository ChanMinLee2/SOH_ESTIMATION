# REFACTORING.md

2026-09-22부터 진행 중인 코드베이스 리팩토링 기록. **계속 업데이트되는 살아있는 문서** —
새 작업을 마칠 때마다 "진행 이력"에 절을 추가한다. 과거 절은 그 시점 기준 사실이므로
이후 작업이 내용을 뒤집어도 수정하지 않고, 새 절에 "N월N일 항목 정정" 식으로 남긴다.

## 결론 요약 (TL;DR)

아래 "진행 이력"(2026-09-22~09-27, 총 18개 라운드)을 한 문장씩으로 압축하면:

1. **파라미터 단일화** — 9개 스크립트에 흩어져 있던 CLI 기본값·yaml 프리셋
   (`model_lib/config/*.yaml`, `--model-config` 플래그)을 전부 폐기하고
   `parameters.py` 하나로 모았다. 그 과정에서 발견한 의미 중복/완전히 죽은 값
   (시드 3종, `--datasets`, redundancy threshold, `leak_cols`, `is_real_input` 등)도
   전부 통합·삭제.
2. **폴더 구조 = 파이프라인 스텝 번호** — `5_model/`이라는 단일 폴더에 모든 스텝이
   뒤섞여 있던 구조를 `1_convert/`~`9_eval/` + 공유 라이브러리 `model_lib/`로
   재배치해, 폴더 번호만 보면 실행 순서를 알 수 있게 만들었다.
3. **죽은 코드 전량 제거** — v0~v3 옛 버전, 미채택 v5 실험, CNN 분류기/임베딩,
   `SCRTrainer`/Laplace UQ 체인, 비-정식 시나리오 축 8종, "Bucket A" 죽은 코드
   15항목 + yaml 설정 체계 전체까지 — "지금 정식 학습 레시피가 실제로 타는 경로"
   기준으로 안 쓰는 코드를 전부 걷어냈다. 반대로 논문의 실제 ablation 실험
   경로("Bucket B" — 대체 회귀 헤드, 축 파라미터 변형 등)는 의도적으로 보존.
4. **모델 로직 설명 문서 신규 작성** — `docs/PIPELINE.md`(데이터 변환부터 SOH
   출력까지 9단계 전체 설명) + 인포그래픽, 처음 보는 사람 기준으로 새로 작성.

**코드 절감**(git 기준, 리팩토링 시작 직전 커밋 `b5ae262` 대비 현재 작업트리,
`.py` 파일만 집계):

| 지표 | 리팩토링 전 | 현재 | 변화 |
|---|---|---|---|
| Python 파일 수 | 110개 | 72개 | **-38개 (-35%)** |
| Python 코드 총 줄 수 | 38,791줄 | 26,247줄 | **-12,544줄 (-32%)** |
| 완전 삭제된 파일 | — | — | **46개** (그 외 다수 파일이 이름 바뀌며 내용도 축소) |

**검증 신뢰도** — 학습 루프(`model_lib/models/scr_model.py`, `8_train/train.py`
등)에 손댈 때마다 동일 조건(seed=0/split-seed=0) 전체 재학습을 돌려 체크포인트
MD5를 비교했고, **지금까지 6회 전부 원본(`32c2cd84216510655ce3c9f15072ee70`)과
바이트 단위로 완전히 일치**했다 — 리팩토링이 실제 학습 결과에 아무 영향도 주지
않았음을 반복적으로 확인.

**남은 일**: `1_convert/`~`3_integrity/`는 아직 이번 정리 대상에 포함되지 않음
(아래 "진행 상황" 표 참고).

## 최종 목적

논문과 함께 코드를 공개했을 때, 처음 보는 사람도 구조만 보고 바로 이해하고 실행해볼 수
있는 상태를 만드는 것. 세 가지 축으로 진행한다:

1. **파라미터 단일화** — 어떤 스크립트도 자기 자신의 하드코딩된 기본값을 갖지 않는다.
   모든 실행 파라미터(축 설정, 실행 환경, 모델/학습 하이퍼파라미터 포함)는
   `parameters.py` 하나에서만 관리하고, 각 스크립트는 그 값을 참조만 한다. CLI 단축
   플래그·yaml 설정 파일 등 파편화된 기본값 소스를 없애 "같은 옵션 없음이 실행 경로에
   따라 다른 결과를 내는" 상황을 원천 차단한다.
2. **폴더 구조 = `run_pipeline.py` 스텝 순서** — 프로젝트 최상위 디렉터리 이름만 보고도
   전체 파이프라인 순서(1~10)를 알 수 있게 만든다. 스텝 전용 코드는 번호 폴더에,
   2개 이상 스텝이 공유하는 코드는 별도 공유 라이브러리 위치에 모은다.
3. **코드 크기 축소** — 메인 로직(현재 실제로 쓰이는 v4 파이프라인)은 유지하면서, 더는
   쓰지 않는 과거 버전(v0~v3)·미채택 실험(v5) 경로를 제거한다. 뒷 스텝(10_eval)부터
   앞 스텝(1_convert) 방향으로 거꾸로 진행 — 뒤쪽이 앞쪽 산출물에 의존하므로, 이 순서가
   호환성 깨질 위험이 가장 적다.

## 검증 원칙

구조를 옮기거나 코드를 지울 때마다, 가능하면 **실제 데이터로 돌려서 수치로 확인**한다 —
컴파일/임포트 성공만으로는 "로직이 똑같이 동작하는지"를 보증하지 못하기 때문. 이번
리팩토링 내내 쓴 방법:

- 실제 학습된 v4 체크포인트(`5_model/experiments/phase1_lab/results/p1v2_runs/
  0921_1953_p1v2_p1v4_scen_lag1zone_redunfix_seed0`)를 기준점으로 삼는다.
- 구조/코드를 바꾸기 전후로 `10_eval/test.py --run-dir <그 경로>`를 각각 돌려 test
  set(66만+ 세그먼트) 전체 재평가 metrics(oracle/hard/soft RMSE·R²)를 비교한다.
- 소수점까지 완전히 일치해야 통과 — 지금까지 두 차례(폴더 재배치 직후, 10_eval 코드
  축소 직후) 전부 완전 일치 확인됨.
- 학습 루프 자체를 고치는 경우(예: 9_train)는 처음엔 스모크 테스트(짧게 몇 에폭만 돌려
  코드 경로가 안 끊기는지만 확인)로 충분하다고 봤는데, 2026-09-24에 실제로 **원본과
  완전히 동일한 조건(seed/split-seed/axis-config/data-dir 등)으로 전체 재학습을
  돌려봤더니 최종 체크포인트 파일의 MD5 해시가 원본과 완전히 같았다** — 즉 이 학습
  파이프라인은 (적어도 이 환경/GPU에서는) 완전히 결정론적이다. 그래서 지금은: 학습
  루프를 고쳤을 때도 **시간이 허락하면 동일 조건 전체 재학습 → 체크포인트 MD5 비교**가
  1순위 검증이고(비용이 크면 스모크 테스트로 대체), 에폭 수 자체가 크게 달라지는 경우
  (max-epochs 축소 등 의도된 변경)만 스모크 테스트로 내려간다.

## 진행 상황

| 대상 | 상태 |
|---|---|
| `parameters.py` 단일화 (축 설정 + 9개 스크립트 파라미터 + Step9 yaml 흡수) | ✅ 완료 |
| 폴더 재배치 (`5_model/` → `6_interaction/`~`10_eval/` + `model_lib/`) | ✅ 완료 (이후 아래 renumbering으로 번호가 한 번 더 바뀜) |
| 번호 renumbering (`5_model/`→`legacy_results/`, `6~10`→`5~9`) | ✅ 완료 — 아래 "진행 이력" 참고 |
| `9_eval/`(구 `10_eval/`) 코드 축소 (`test_phase1_checkpoint.py` → `test.py`) | ✅ 완료 |
| `8_train/`(구 `9_train/`) 코드 축소 (`phase1_trainer_v2.py` → `train.py`) | ✅ 완료 — 전체 재학습 체크포인트 MD5 완전 일치로 검증 |
| `7_kernel/`(구 `8_kernel/`) 코드 축소 (`build_kernel_group_features.py` → `kernel.py`) | ✅ 완료 — 산출물(pkl/json) 완전 일치로 검증 |
| `6_synergy/`(구 `7_synergy/`) 코드 축소 (`build_synergy_groups.py` → `synergy.py`) | ✅ 완료 — 산출물 JSON 완전 일치로 검증 |
| `5_interaction/`(구 `6_interaction/`) 코드 축소 (`test_hi_scenario_interaction.py` → `interaction.py`) | ✅ 완료 — 산출물 JSON 완전 일치로 검증 |
| `4_hi_analysis/` 코드 정리 | ✅ 완료 — 파라미터 기본값 버그 2건 수정 + 독스트링 갱신, 아래 참고 |
| Stage0(`model_lib/legacy/`) 삭제 + 비-정식 시나리오 축(`common/scenario/`) 삭제 | ✅ 완료 — 전체 재학습 체크포인트 MD5 완전 일치로 검증 |
| `1_convert/`~`3_integrity/` | ⬜ 미착수 |

## 진행 이력

### 2026-09-22 — `4_hi_analysis/` 구조 파일럿

폴더 구조 목표를 처음 정할 때 가장 지저분했던 `4_hi_analysis/`(14개 파일 중 파이프라인이
실제로 쓰는 건 2개뿐)를 시범 대상으로 골랐다.

- **`hi_correlation.py` 죽은 코드 제거**: 2231줄 → 1792줄. `_seg_stat`/`_seg_diff`/
  `_seg_lfp`/`_dtw_distance`/`_frechet_distance`/`_seg_morph_curves`가 파일 앞쪽에 로컬
  정의돼 있었는데, 파일 맨 끝에서 `hi_compute.py`(당시 `5_model/`)로부터 동명 함수를
  다시 import해 조용히 덮어쓰고 있었다 — 즉 앞쪽 정의는 한 번도 실행되지 않는 도달
  불가 코드였다. 로컬 정의를 삭제하고 파일 상단에서 정상적으로 import하도록 정리.
  이 과정에서 `_peak_fwhm_asym`가 `common/scenario/_curves.py`가 아니라
  `hi_compute.py` 버전(경계 폴백 로직이 다름)으로 실제 운영되고 있었다는 것도
  발견 — 계산값이 바뀌지 않도록 그 버전을 그대로 유지.
- **비파이프라인 스크립트 12개**를 `4_hi_analysis/tools/`로 이동(`axis_comparison.py`,
  `cluster_scatter.py`, `k2_boundary.py`, `k_selection_diagnostics.py`,
  `plateau_soc_stats.py`, `plot_all_mit_cells.py`, `plot_cell_cycles.py`,
  `plot_cycle_segments.py`, `profile_hi_timing.py`, `redundancy_gate_resolution.py`,
  `seg_corr_analysis.py`, `seg_diagnose.py`) — 각 파일의 `PROJECT_ROOT`/`sys.path`
  깊이를 이동한 만큼 재계산.

### 2026-09-23 — `parameters.py` 단일화

- `hi_correlation.py`의 `--axis-config` 기본값을 하드코딩 대신 참조하도록 시작 →
  범위가 커져 `run_pipeline.py`의 `--n1`/`--n2`/`--n2-start`/`--n2-end`/`--n2-step`/
  `--n-samples` 개별 단축 플래그를 전부 없애고 `parameters.py: ACTIVE_AXIS_CONFIG`
  딕셔너리 하나로 통합. `hi_correlation.py` 자신의 15개 단축 플래그는 `default=None`
  센티넬을 유지하되(명시 여부 구분 필요), 병합 로직을 "단축 인자 있으면 axis-config
  전체를 새로 만들기"에서 "이미 정해진 axis-config 위에 명시된 값만 patch"로 재설계 —
  raw `--axis-config` JSON과 단축 플래그를 같이 써도 나머지 키가 안 사라지게 됨.
- **`ACTIVE_AXIS_CONFIG` 오류 발견/정정**: 처음엔 "마지막 실행값"을 `RESULTS_LOG.md`의
  최신 항목에서 가져왔는데, 그게 사실 `scen_lag1zone`이라는 별도 실험 변형이었다 —
  실제 정식(canonical) 값은 `model_lib/config/main_qfref_S.yaml: scenario.axis_config`
  (`min_pts`/`calibration_period`/`offset_amp` 포함, `tile_scope` 없음)였음. 이 값으로
  정정하고 `FIXED_CANONICAL_DATA_DIR`/`FIXED_CANONICAL_SEG_DATA_DIR`를 신설.
- 나머지 8개 스크립트(`1_convert/convert_unified.py`, `2_preprocess/preprocess.py`,
  `3_integrity/check_integrity.py`, `hi_segment_viz.py`,
  `test_hi_scenario_interaction.py`, `build_synergy_groups.py`,
  `build_kernel_group_features.py`, `test_phase1_checkpoint.py`)의 모든 파라미터를
  `parameters.py` 참조로 전환. `--workers` 기본값이 스크립트마다 3/4/CPU-2로 제각각
  이던 것도 `P.ACTIVE_WORKERS` 하나로 통일.
- 이 과정에서 **`DEFAULT_AXIS_CONFIG` 지뢰**(`ref_lag=0`, `min_pts`/`calibration`/
  `offset` 누락 — `ACTIVE_AXIS_CONFIG`와 조용히 어긋나 있던 하드코딩 폴백)를
  `phase1_trainer_v2.py`·`test_hi_scenario_interaction.py`·`build_synergy_groups.py`
  ·`build_kernel_group_features.py` 4개 파일에서 전부 발견해 `parameters.py` 참조로
  교체 — 지금까지는 yaml/축-설정이 항상 명시돼 있어 드러나지 않던 죽은 폴백이었다.
- **`phase1_trainer_v2.py`의 yaml 흡수**: `fixed.yaml` + `main_qfref_S.yaml` 병합
  결과(`data`/`classifier`/`regression`/`model`/`loss`/`training`/`evaluation`/`uq`/
  `test` 9개 섹션)를 `parameters.py: P1_MODEL_CONFIG` 딕셔너리로 그대로 옮겼다 — 실제
  `load_config()` 호출 결과와 JSON 정규화 후 diff해서 완전히 동일함을 확인 후 반영.
  `--model-config`는 필수에서 선택으로 바뀌었고, 미지정 시 이 딕셔너리를 `deepcopy`해서
  쓴다(명시하면 기존처럼 그 yaml을 읽는 이스케이프 해치는 유지 — `all4`/`hust_only`/
  `noscen`/`p60`/`transformerL` 등 다른 프리셋 재현용). `run_pipeline.py`는 Step9에
  더는 `--model-config`를 자동 주입하지 않음(Step6~8은 아직 필수라 그대로 유지).

### 2026-09-23 — 폴더 구조 전면 재배치

`5_model/`을 분해해 프로젝트 최상위에 스텝 번호 폴더를 신설:

```
6_interaction/  ← test_hi_scenario_interaction.py
7_synergy/      ← build_synergy_groups.py
8_kernel/       ← build_kernel_group_features.py
9_train/        ← phase1_trainer_v2.py
10_eval/        ← test_phase1_checkpoint.py, plot_hi_selection_matrix.py
model_lib/      ← models/ datasets/ training/ evaluation/ utils/ config/(yaml)
                  legacy/(train_scr.py, test_scr.py) tools/(visualize_results.py)
                  log_utils.py results/(새 run 출력 위치)
5_model/        ← experiments/phase1_lab/results/(기존 run 데이터, 대용량이라 안 옮김)
```

- `models`/`datasets`/`training`/`evaluation`/`utils` 5개 패키지는 서로
  `from models.xxx import`처럼 절대경로 스타일로 import하고 있어서 **패키지 내부 코드는
  한 줄도 안 건드리고** 폴더째 옮기는 것만으로 충분했다(사전 조사로 확인 후 진행).
  대신 모든 소비자 스크립트의 `sys.path` 삽입 지점(`5_model`→`model_lib`)과
  `PROJECT_ROOT` 상대 깊이(폴더 단계 수 변화만큼)를 스크립트별로 재계산해서 고쳤다.
- `hi_compute.py`(Step4 전용 HI 계산 로직인데 엉뚱하게 `5_model/`에 있던 것)를
  `4_hi_analysis/`로 이동 — 오히려 import가 단순해짐.
- `run_pipeline.py`의 `STEPS` 리스트 10개 경로, `P1V2_RUNS_DIR`(새 위치:
  `model_lib/results/p1v2_runs`), `LEGACY_P1V2_RUNS_DIR`(기존 위치, `_latest_p1v2_run_dir()`가
  재배치 직후에도 둘 다 뒤져서 최신 run을 찾도록), `plot_script` 경로 전부 갱신.
- `docs/figures/*.py`(논문 그림 스크립트 8개)는 전부 안 옮긴 `results/` 경로만 참조해서
  **변경 불필요** — 사전 조사로 확인 후 건드리지 않았다.
- **검증**: `run_pipeline.py 1 --to-step 10` 드라이런으로 10개 스텝 전부 새 경로 resolve
  확인 + 실제 v4 체크포인트 재평가(oracle RMSE=0.017441369593143463,
  R²=0.9322315847790643 등)로 전체 파이프라인 동작 재현성 확인.

### 2026-09-24 — `10_eval/` 코드 축소 (뒤에서부터 첫 스텝)

`test_phase1_checkpoint.py` → **`test.py`**로 개명. v4 아키텍처에 더는 쓰이지 않는
과거/실험 경로 제거:

- **`scen_group_ids`/`synergy_groups_json`** 처리 전체 삭제(+ `train_scr` import 제거).
  v0~v2용 `GroupedHardConcreteGate` 그룹 게이트 메커니즘 — v4는 kernel HI +
  interaction_json(shared_gate)로 완전히 대체됨. 실제 v4 run의 `p1v2_summary.json`에
  `"synergy_groups_json": null`인 것으로 확인.
- **`scen_gate_direction_only`/`scenario_onehot_input`** 처리 삭제. `parameters.py`에
  이미 "v5 실험 전용, 미사용"으로 표시돼 있던 죽은 분기 — `SCRModel` 생성자 기본값과
  동일해서 인자를 아예 안 넘겨도 동작 무변화.
- **`--regression-model`** CLI 플래그 삭제(transformer/resnet_tab 등 4개 대안 아키텍처
  선택지 포함). 원래 설계의도가 "항상 mlp 강제, sanity-check 전용 오버라이드"였던
  옵션 — `cfg["model"]`에 이미 저장된 값을 그대로 쓰도록 단순화.
- **`--export-for-visualize`** CLI 플래그 삭제. 코드 추적 결과 완전한 no-op이었음
  (관련 저장 함수가 이 플래그 값과 무관하게 항상 실행되고 있었음) — `run_pipeline.py`
  주입부와 `parameters.py: FIXED_EXPORT_FOR_VISUALIZE`도 같이 제거.
- 안 읽히던 `kernel_names_by_scen` 변수 삭제.
- 남은 파라미터(`--interaction-json`/`--kernel-features-pkl`/`--combined-redundancy-json`)
  기본값을 기존에 있던 `P.ACTIVE_INTERACTION_JSON`/`ACTIVE_KERNEL_FEATURES_PKL`/
  `ACTIVE_COMBINED_REDUNDANCY_JSON`에 연결. `--run-dir`/`--checkpoint`는 실행마다 고유한
  필수값이라 parameters.py 이전 대상에서 제외(값을 가질 수 없는 성격이라 판단).
- 674줄 → 623줄(7.6% 축소).
- **검증**: 같은 v4 체크포인트로 리팩토링 전/후 재평가 — oracle/hard/soft RMSE·R² 전부
  소수점까지 완전 일치. 제거한 로직이 실측으로 100% 죽은 코드였음이 확인됨.

### 2026-09-24 — `9_train/` 코드 축소 (`phase1_trainer_v2.py`)

10_eval과 같은 기준(실제 v4 run의 `p1v2_summary.json`에 값이 `null`/`false`로 찍혀있는
것 = 확인된 죽은 경로)으로 과거/미채택 실험 로직 제거:

- **`--synergy-groups-json`/`scen_group_ids`** 처리 전체 삭제(v0~v2 그룹 게이트,
  10_eval과 동일 근거). `train_scr._load_synergy_group_ids` 호출부만 제거 — JSON/plot
  저장 함수(`_save_probe_masks_to_json` 등)는 여전히 써서 `train_scr` import는 유지.
- **`--scen-gate-direction-only`/`--scenario-onehot-input`** 처리 삭제(v5 실험,
  `parameters.py`에 이미 "미사용"으로 표시돼 있던 것).
- **`--warmstart-branch-epoch`**("웜스타트 후 분기" 커리큘럼, docs/260917_REPORT.md
  안건2) 처리 전체 삭제 — 실제 v4 run에 `"warmstart_branch_epoch": null` 확인. 학습
  루프의 `eff_l0`/`beta_now` 계산에서 `warmstart_T` 분기 조건을 없애고 기존 "T=0"
  경로(원래도 이게 100% 동일 동작이라고 주석에 명시돼 있었음)로 통일. `model.
  branch_scen_gates()` 호출부도 같이 삭제 — 정의 자체(`model_lib/models/scr_model.py`)는
  이번 스코프 밖이라 그대로 둠(다음 `model_lib/` 정리 때 같이 볼 후보로 남김).
- **`--regression-model`** CLI 플래그 삭제 — 10_eval과 동일 판단(항상 mlp, sanity-check
  전용 오버라이드였음). `p1_model_cfg`가 `cfg["model"]`의 저장값을 그대로 씀.
- `p1v2_summary.json` 스키마에서 위 항목들의 키(`synergy_n_groups`,
  `scen_gate_direction_only`, `scenario_onehot_input`, `warmstart_branch_epoch`,
  `regression_model_used`)를 제거. 단, `synergy_groups_json` 키는 `None` 고정값으로
  남김 — `model_lib/tools/visualize_results.py`가 과거(v0~v3) run과 나란히 읽는 스키마라
  키 자체를 없애면 그쪽이 깨질 수 있어 하위호환 목적으로 유지.
- `--seed`/`--split-seed`/`--tag`를 `required=True`에서 `parameters.py`(`ACTIVE_SEED`/
  `ACTIVE_SPLIT_SEED`/`ACTIVE_P1_TAG`) 기본값으로 전환. `--train-cycle-frac`/`--beta-min`/
  `--device`/`--max-epochs`/`--patience`/`--batch-size`/`--kernel-features-pkl`/
  `--combined-redundancy-json`/`--interaction-json`/`--lambda-l0-override`/
  `--l0-warmup-epochs-override`/`--l0-norm-constant`/`--hi-cost-weighted-l0`도 각각
  대응하는 `parameters.py` 상수를 참조하도록 연결(전부 이미 존재했지만 이 스크립트
  자신의 argparse 기본값에는 안 걸려 있던 것들). `--val-rmse-epsilon`은 대응 상수가
  아예 없어서 `FIXED_VAL_RMSE_EPSILON = 0.0005`를 신설.
- `run_pipeline.py`에서도 이제 사라진 `--warmstart-branch-epoch`/`--scen-gate-
  direction-only`/`--scenario-onehot-input` 관련 CLI 정의·주입 로직·프리뷰 출력을 같이
  제거, `parameters.py`의 대응 상수(`ACTIVE_WARMSTART_BRANCH_EPOCH`,
  `FIXED_SCEN_GATE_DIRECTION_ONLY`, `FIXED_SCENARIO_ONEHOT_INPUT`)도 삭제.
- 895줄 → 765줄(14.5% 축소).
- **검증(1차, 스모크 테스트)**: `parameters.py` 기본값만으로(축/모델 설정 플래그 일체
  없이) 3에폭 실제 학습을 돌려 데이터 로딩→모델 구성→학습 루프→체크포인트/config.yaml/
  p1v2_summary.json 저장까지 정상 완료 확인 후, 그 체크포인트를 `10_eval/test.py`에
  그대로 넘겨 oracle/hard/soft 평가와 figures/metrics/predictions/routing 저장까지
  끝까지 도는 것을 확인(수치는 3에폭이라 당연히 나쁨 — 코드 경로 검증이 목적). 스모크
  테스트 산출물은 검증 후 삭제.

### 2026-09-24 — `phase1_trainer_v2.py` → `train.py` 개명 + 전체 재학습 검증

`9_train/phase1_trainer_v2.py` → **`train.py`**로 개명(`10_eval/test.py`가 이제
`from train import (...)`로 참조 — `test.py`/`train.py` 이름 대칭 완성). 스크립트
경로를 언급하는 모든 파일(`run_pipeline.py`, `10_eval/test.py`,
`model_lib/models/scr_model.py`, `model_lib/legacy/train_scr.py`,
`model_lib/tools/visualize_results.py`, `docs/figures/*.py` 등)의 주석/독스트링도
일괄 갱신. 이 김에 지난 10_eval 개명 때 놓쳤던 `test_phase1_checkpoint.py` 잔여
참조 4곳(`plot_hi_selection_matrix.py`, `docs/figures/fig8_single_dataset.py`,
`fig10_v4_example.py`, `visualize_results.py`)도 같이 정리.

**파라미터 분리 재감사**: `train.py`의 argparse 인자 22개를 전수 점검 — 전부 (a)
`parameters.py` 상수를 직접 참조하거나(`P.ACTIVE_*`/`P.FIXED_*`, 필요시 `if ... is
not None else 기본값` 패턴), (b) 애초에 "단일 소스 값"이 존재할 수 없는 순수
오버라이드(`--charge-m`/`--discharge-m`/`--scen-k`는 이미 `P1_MODEL_CONFIG`에서 온
`cfg["classifier"]`/`cfg["regression"]` 위에 얹는 값, `--output-dir`/`--tag`는
실행마다 고유한 경로/식별자)로 확인됨. argparse 기본값에 하드코딩된 매직 넘버 없음.

**검증(2차, 전체 재학습)**: 스모크 테스트로는 "학습 루프를 고친" 변경(웜스타트 분기
로직 삭제 등)이 수치적으로 안전한지까지는 못 보여줘서, 원본 run
(`0921_1953_p1v2_p1v4_scen_lag1zone_redunfix_seed0`)과 **완전히 동일한 조건**
(seed=0, split-seed=0, 동일 axis-config/data-dir/seg-data-dir/kernel-features-pkl/
combined-redundancy-json/interaction-json)으로 전체 재학습을 처음부터 끝까지 돌렸다.
결과:

| | 원본 | 재현 학습 |
|---|---|---|
| 조기종료 총 에폭 | 416 | **416** |
| 선택된 epoch | 355 | **355** |
| gate_saturation | 0.0041 | **0.0041** |
| 체크포인트 파일 MD5 | `32c2cd84...` | **`32c2cd84...`(완전 동일)** |
| test oracle RMSE/R² | 0.017441.../0.93223... | **완전 동일(전체 자릿수 일치)** |
| test hard/soft RMSE/R² | 0.017462.../0.017464... | **완전 동일** |

체크포인트 파일 자체가 바이트 단위로 원본과 동일했다 — 이 학습 파이프라인이(적어도
이 환경에서) 완전히 결정론적이라는 뜻이고, 이번 세션에 제거한 로직(웜스타트 분기,
v0~v2 그룹 게이트, v5 direction-only/onehot, regression-model 대안 아키텍처)이 이
run에서 정말로 한 번도 실행되지 않는 순수 죽은 코드였다는 걸 가장 강력한 방식으로
증명한다. 재현 학습 산출물(`model_lib/results/p1v4_scen_lag1zone_redunfix_verify_seed0/`)은
검증 기록으로 보존.

### 2026-09-24 — `8_kernel/` 코드 축소 (`build_kernel_group_features.py` → `kernel.py`)

10_eval/9_train과 같은 절차를 적용했지만, 이 스크립트는 감사 결과 **v0~v3/v5용 대체
아키텍처 분기 자체가 없었다** — synergy 그룹→커널 HI 융합이라는 단일 v4 로직만 존재하고,
과거 다른 스크립트들에서 발견됐던 "이제 안 쓰는 모델 구조" 류의 죽은 코드가 없었다(이미
Phase 3에서 `DEFAULT_AXIS_CONFIG` 지뢰 수정 등으로 상당 부분 정리돼 있었던 상태).
그래서 이번 라운드는 개명 + 파라미터 재감사 + 산출물 검증이 중심이었다:

- **개명**: `build_kernel_group_features.py` → **`kernel.py`**(`test.py`/`train.py`와
  이름 대칭). 스크립트 경로를 언급하는 모든 파일(`run_pipeline.py`의 `STEPS` 리스트 포함,
  `9_train/train.py`, `7_synergy/build_synergy_groups.py`, `model_lib/log_utils.py`,
  `model_lib/training/scr_loss.py`, `model_lib/legacy/train_scr.py`,
  `model_lib/datasets/segment_dataset.py`, `model_lib/models/scr_model.py`,
  `model_lib/models/cap_heads.py`)의 주석/독스트링을 일괄 갱신. `docs/phase1_lab/
  RESULTS_LOG.md`(append-only 실험 로그)와 논문 초안 계열 문서(`docs/DRAFT.md`,
  `docs/260913_REPORT.md`, `docs/MODEL_FLOW.md`, `docs/PLOTS.md`)는 각 시점의 실제
  실행 기록/서술이라 손대지 않음(10_eval/9_train 때와 동일 원칙).
- **파라미터 재감사**: 17개 인자 전수 점검 — 전부 이미(Phase 3 때) `parameters.py`
  상수(`P.FIXED_KERNEL_*`, `P.FIXED_MIN_RAW_PARTIAL_CORR`, `P.FIXED_COMBINED_REDUNDANCY_
  THRESHOLD`, `P.ACTIVE_SPLIT_SEED`)에 연결돼 있었음을 확인. 새로 옮길 하드코딩된 기본값
  없음. `--min-raw-partial-corr`는 자기 help 문구에 "v3 전용"이라 적혀 있어 얼핏
  10_eval/9_train에서 지운 v0~v2/v5 분기들과 비슷해 보이지만, 실제로는 **대체 모델
  아키텍처가 아니라 같은 v4 로직 안의 선택적 추가 필터**이고(기본값 None=비활성일 때
  "기존 v1/v2 동작과 100% 동일"이라고 스스로 명시), `run_pipeline.py`가 이미
  `P.FIXED_MIN_RAW_PARTIAL_CORR`로 정식 노출해두고 있어 — 삭제 대상이 아니라고 판단하고
  유지.
- **참고(수정 안 함)**: 검증 실행 중 Windows cp949 콘솔에서 마지막 요약 print문(em-dash
  `—` 문자 포함)이 `UnicodeEncodeError`로 죽는 걸 발견 — 두 산출물 파일(pkl/json)은
  이미 그 전에 저장이 끝난 뒤라 결과에는 영향 없고, 콘솔 요약 한 줄과 `append_log_entry()`
  호출만 스킵된다. 과거 이 스크립트가 성공 로그를 남긴 걸 보면 그때는 UTF-8 콘솔에서
  실행됐던 것으로 보임 — 코드 자체의 회귀가 아니라 환경(콘솔 코드페이지) 의존적인
  기존 버그라 이번 스코프에서는 고치지 않고 기록만 남김(향후 `model_lib/`나 공통
  유틸리티 정리 때 `print` 전반에 UTF-8 강제나 ASCII 대체를 검토할 후보).
- **검증**: 원본 run(`0921_1953_p1v2_p1v4_scen_lag1zone_redunfix_seed0`)이 실제로 썼던
  synergy-groups-json과 **완전히 동일한 입력**(seed=0, split-seed=0, alpha=1.0,
  n-components=100, redundancy-threshold=0.9, combined-redundancy-threshold=0.95,
  동일 axis-config/data-dir/seg-data-dir)으로 새 `kernel.py`를 재실행해 별도
  디렉터리(`model_lib/results/kernel_verify_seed0/`)에 저장한 뒤, 원본 산출물과
  프로그램적으로 비교:
  - `kernel_group_features_*.pkl`: 후보 94개 → 최종 93개(원본과 동일한 수치), 93개
    피처 전부에 대해 이름/시나리오/멤버 인덱스/`train_r2`/정규화 `mean`·`std`가 완전
    일치, 각 피처의 `Nystroem`(`components_`/`component_indices_`/`normalization_`)과
    `Ridge`(`coef_`/`intercept_`) 내부 배열도 `np.array_equal`로 **비트 단위 완전
    일치**(불일치 0/93).
  - `*_combined_redundancy.json`: 6개 시나리오 전부(`removed_raw_idx`/
    `removed_raw_names`/`removed_kernel_names`/`n_total_checked`/`n_edges`) 완전
    일치(불일치 0/6).
  - 콘솔에 찍힌 핵심 수치(평균 train R²=0.3887, 시나리오별 개수
    `{chg_lo:16, chg_mid:15, chg_hi:16, dis_hi:16, dis_mid:15, dis_lo:15}`, 결합
    다중공선성 배제 raw 77개/kernel 0개)도 `RESULTS_LOG.md`에 기록된 원본 값과 전부
    동일.
  - 이 RBF 커널 피팅 파이프라인도(Nystroem 랜덤 샘플링에 `random_state=split_seed`
    고정 + Ridge는 closed-form) 학습 루프와 마찬가지로 이 환경에서 완전히 결정론적임을
    추가로 확인. 검증 산출물은 `model_lib/results/kernel_verify_seed0/`에 기록으로 보존.

### 2026-09-24 — `7_synergy/` 코드 축소 (`build_synergy_groups.py` → `synergy.py`)

이 스크립트도 8_kernel과 같은 결론 — **v0~v3/v5용 대체 아키텍처 분기가 없다**. 다만 이번엔
얼핏 "v3 전용"/"실험용"이라고 적힌 두 플래그(`--global-dedup`, `--shuffle-from`)가 실제로는
삭제 대상이 아니라는 걸 확인하는 과정이 핵심이었다:

- **`--global-dedup`**: `parameters.py: FIXED_GLOBAL_DEDUP`에 이미 "docs 상 v4 정식
  레시피는 True라 돼 있지만 이 세션에 실제로 돌린 run은 전부 False로 실행됐다"는 경고
  주석이 있었음(2026-09-21에 실측 기준으로 이미 정리된 사안) — 대체 모델 구조가 아니라
  그룹 구성 *전처리 알고리즘*의 선택지(사전 union-find 다중공선성 정리 vs 순차 그리디)라
  8_kernel의 `--min-raw-partial-corr`와 같은 이유로 유지.
- **`--shuffle-from`/`build_groups_shuffled()`("v-ctrl 전용")**: 처음엔 9_train에서 지운
  `--warmstart-branch-epoch` 같은 폐기된 실험으로 보였으나, `docs/260825_RESULTS.md`·
  `docs/260827_RESULTS.md`·`docs/260901_REPORT.md`·`docs/260901_V5_DESIGN.md`를 확인한
  결과 이건 **논문 어블레이션(무작위 그룹 대조군, "시너지가 진짜인지 우연인지" 검증)에
  실제로 쓰인 방법론**이고 `docs/260901_V5_DESIGN.md`엔 향후 재사용 계획까지 적혀
  있어 — 폐기된 게 아니라 현재도 유효한 재현성 인프라라고 판단해 유지. `6_interaction/
  test_hi_scenario_interaction.py`가 이 파일의 `_load_all_scenarios`를 직접 `import`해서
  재사용하고 있다는 것도 이번에 확인(단순 주석 언급이 아니라 실제 함수 의존) — 개명 시
  `from build_synergy_groups import` → `from synergy import`로 같이 갱신, 실제 로딩까지
  재현해 정상 동작 확인.
- **파라미터 재감사**: 14개 인자 중 13개는 이미 `parameters.py` 연결 확인. `--shuffle-seed`
  하나가 하드코딩 기본값(`42`)으로 남아있던 걸 발견해 `parameters.py: FIXED_SHUFFLE_SEED = 42`
  신설 후 연결(`6_interaction/test_hi_scenario_interaction.py`에도 동일한 하드코딩
  `42`가 있지만, 그 폴더 차례가 아니라 지금은 안 건드림 — 다음 `6_interaction` 라운드
  후보로 남김).
- **개명**: `build_synergy_groups.py` → **`synergy.py`**. 참조하는 모든 파일
  (`run_pipeline.py`의 `STEPS`, `9_train/train.py`, `8_kernel/kernel.py`,
  `6_interaction/test_hi_scenario_interaction.py`의 실제 `import`문 + 주석,
  `model_lib/log_utils.py`, `model_lib/legacy/train_scr.py`,
  `model_lib/models/scr_model.py`, `model_lib/models/hard_concrete.py`)의 주석/코드
  갱신. `docs/phase1_lab/RESULTS_LOG.md`·논문 초안 계열 문서는 기존 원칙대로 안 건드림.
- **검증**: 원본 run이 실제로 썼던 입력과 완전히 동일한 조건(seed=0, split-seed=0,
  max-group-size=4, redundancy-threshold=0.9, min-partial-corr=0.02, prefilter-top-m=15,
  `--global-dedup`/`--shuffle-from` 둘 다 미지정 — 원본과 동일)으로 새 `synergy.py`를
  재실행해 별도 디렉터리(`model_lib/results/synergy_verify_seed0/`)에 저장한 뒤 원본
  `synergy_groups_*.json`과 프로그램적으로 비교: `tag`를 제외한 43개 키(6개 시나리오 ×
  7개 필드: `seg_name`/`groups`/`group_names`/`group_scores`/`group_attached`/
  `group_attached_names` + 최상위 설정 필드) **전부 완전 일치**(불일치 0/43) — 그룹 멤버
  인덱스·이름·점수·attached 목록까지 값 단위로 하나도 다르지 않음. 이 그룹 구성 알고리즘엔
  애초에 난수가 전혀 없어(순수 numpy 상관/최소자승) 결정론이 당연히 보장되지만, 리팩토링
  전후로 실제 재현까지 확인. 검증 산출물은 `model_lib/results/synergy_verify_seed0/`에
  기록으로 보존.

### 2026-09-24 — `6_interaction/` 코드 축소 (`test_hi_scenario_interaction.py` → `interaction.py`)

7_synergy와 같은 패턴 — 대체 아키텍처 분기 없음, `--shuffle-from`("v4-ctrl 전용")도
`synergy.py`의 `--shuffle-from`과 동일한 논문 어블레이션 인프라라 유지(지난 라운드에서
이미 확인한 원칙 재적용, 새로 조사할 것 없었음).

- **파라미터 재감사**: 지난 7_synergy 라운드에서 "다음 6_interaction 차례로 남김"이라고
  적어뒀던 `--shuffle-seed` 하드코딩(`42`)을 이번에 `parameters.py: FIXED_SHUFFLE_SEED`로
  연결 — 이제 `synergy.py`/`interaction.py` 둘 다 같은 상수를 공유. 나머지 11개 인자는
  이미 `parameters.py` 참조로 확인.
- **개명**: `test_hi_scenario_interaction.py` → **`interaction.py`**. `run_pipeline.py`의
  `STEPS`(+ 관련 주석 2곳), `9_train/train.py`, `parameters.py`,
  `model_lib/models/scr_model.py`의 참조를 일괄 갱신. 이 파일을 직접 `import`하는
  다른 코드는 없음(확인 완료). `docs/260903_REPORT.md`는 기존 원칙대로 안 건드림.
- **검증**: 이 스크립트는 `append_log_entry()`를 호출하지 않아 `RESULTS_LOG.md`에 원본
  실행 커맨드 기록이 없었음 — 대신 원본 run 폴더에 저장된
  `hi_scenario_interaction_p1v4_scen_lag1zone_redunfix_interaction.json` 자체에서
  `alpha=0.05`/`min_effect_size=0.1`(둘 다 `parameters.py` 기본값과 동일)을 역으로
  확인하고, 나머지 입력(axis-config/split-seed=0/model-config)은 형제 스크립트들과
  동일 조건으로 맞춰 재실행. 결과를 원본과 비교: 최상위 필드(`n_significant=40`
  등 포함)와 `per_hi`의 HI 64개 전부(`r_by_scenario`/`std_r_across_scenarios`/
  `min_p_raw`/`p_adj_bh`/`worst_pair`/`significant` 등) **완전 일치**(불일치 0/64).
  검증 산출물은 `model_lib/results/interaction_verify_seed0/`에 기록으로 보존.

### 2026-09-24 — 폴더 번호 renumbering (`5_model/` → `legacy_results/`, `6~10` → `5~9`)

Step6~10 코드 축소를 전부 마친 뒤, 사용자가 "`5_model/`은 이제 코드가 없는데 그대로 둬도
되냐"는 질문을 던진 게 계기. 확인해보니 `5_model/`엔 정말 코드가 하나도 없고
`experiments/phase1_lab/results/`(2829개 파일 — 이번 세션 내내 검증 기준점으로 쓴 과거
run 데이터 포함, 대부분 gitignore돼 git 추적 대상은 아님)만 아카이브로 남아있었다.

**당초 계획 대비 바뀐 점**: 처음엔 "`5_model` 이름만 바꾸고 `6_interaction`~`10_eval`을
그대로 `5_interaction`~`9_eval`로 한 칸씩 당기면 된다"고 생각했는데, 실제로 STEPS
리스트를 보니 **스텝 번호 5가 이미 `4_hi_analysis/hi_segment_viz.py`로 존재**했다(전용
폴더 없이 `4_hi_analysis/hi_correlation.py`와 폴더만 같이 씀). 그대로 당기면 스텝 번호
5가 두 개(hi_segment_viz.py, interaction.py)가 돼 `run_pipeline.py 5` 같은 CLI가
애매해지는 문제를 사용자에게 물어봐서 확인: **hi_segment_viz.py도 같이 당기기로
결정** — HI 세그먼트 시각화를 HI 상관 분석과 같은 **Step 4**로 합치고(폴더 하나에
스크립트 두 개, `run_step()`의 `num==4` 분기가 둘 다에 적용됨 — `--force-extract`가
이제 hi_segment_viz.py의 캐시도 같이 무시하게 되는데, 이 스크립트도 원래
`--force`(캐시 무시 재추출)를 지원해서 문제없음), 그 뒤 상호작용 검정→시너지 그룹→
커널 HI→학습→평가를 Step 5~9로 한 칸씩 당겼다.

**실행**:
- `git mv 5_model legacy_results`, `git mv 6_interaction 5_interaction`, ... `git mv
  10_eval 9_eval` — 폴더 6개 개명.
- **`run_pipeline.py`가 핵심**: STEPS 리스트를 새 번호/경로로 재작성(`hi_segment_viz.py`
  항목의 번호를 5→4로 변경, 더는 2개 항목이 서로 다른 번호가 아니게 됨). `len(STEPS)`가
  이제 실제 최대 스텝 번호(9)와 다르므로(=10, 항목 수는 그대로 10개, 번호만 겹침) 그
  전부를 `N_STEPS = max(s[0] for s in STEPS)`로 교체(help 텍스트의 "범위: 1~N" 및
  from_step/to_step 검증 로직 포함) — 이걸 놓치면 `--to-step 10`처럼 더는 존재하지 않는
  번호도 조용히 통과되는 버그가 생겼을 것. 그 외 `if num == X`/`s[0] in (...)` 형태의
  스텝별 분기 로직 15곳 이상을 전부 새 번호로 수정(예: 상호작용검정 6→5, 시너지 7→6,
  커널 8→7, 학습 9→8, 평가 10→9). "Step N" 텍스트(도움말/주석/print, 83곳)는 정규식
  일괄 치환으로 먼저 처리했는데, **이 정규식은 `if num == 9:` 같은 코드의 raw 숫자
  리터럴은 안 건드려서**(패턴이 "Step " 접두사를 요구) 텍스트와 실제 분기 로직이 따로
  놀 뻔한 게 이번 작업에서 가장 까다로운 부분이었다 — grep으로 raw 숫자 비교
  (`s[0]`/`num ==`/`num in`)를 전부 다시 훑어서 잡아냈다.
- `LEGACY_P1V2_RUNS_DIR`/`P1V4_KERNEL_FEATURES_PKL`/`P1V4_INTERACTION_JSON`의
  `5_model/...` 하드코딩 경로도 `legacy_results/...`로 교체.
- 각 스크립트 자신의 docstring 헤더/사용 예시, 서로 참조하는 `sys.path.insert`(가장
  중요: `interaction.py`가 synergy를 찾는 경로 `7_synergy`→`6_synergy`, `test.py`가
  train을 찾는 경로 `9_train`→`8_train`)와 `from synergy import`/`from train import`가
  실제로 깨지지 않는지 재현 임포트로 확인.
- 논문 그림 스크립트(`docs/figures/fig3/5/6/7/8/9/10/12_*.py`)의 `5_model/experiments/
  phase1_lab/...` 하드코딩 경로도 전부 `legacy_results/...`로 교체 — 이 김에
  `fig6_arch_scr_model.py`의 `5_model/models/scr_model.py`(Phase 5 폴더 재배치 때
  놓쳤던 잔재 — 실제로는 이미 그때 `model_lib/models/scr_model.py`로 옮겨져 있었음)도
  같이 수정.
- `docs/phase1_lab/RESULTS_LOG.md`·`docs/DRAFT.md`·`docs/260903_REPORT.md` 등 날짜가
  박힌 과거 기록/논문 초안 문서와, `run_pipeline.py`/`hi_correlation.py` 자체의
  "그 시점엔 `5_model`이었다"는 역사적 서술은 기존 원칙대로 그대로 둠.
- **검증**: (a) `python run_pipeline.py --help` 실제 실행 — "범위: 1~9", 스텝 목록에
  `4 HI 상관 분석`/`4 HI 세그먼트 시각화`가 나란히 나오는 것, 나머지 스텝이 5~9로
  깔끔하게 나오는 것 확인. (b) `run_pipeline` 모듈을 임포트해 `STEPS`/`N_STEPS`를 직접
  불러와 `from_step<=s[0]<=to_step` 필터링을 여러 범위(1~9, 5~9, 8~8, 4~4)로 재현 —
  4~4가 올바르게 두 스크립트를 함께 선택하는 것, 나머지 범위도 의도한 스크립트만
  고르는 것 확인. (c) 모든 STEPS 경로의 실제 파일 존재 확인. (d) `synergy`/`train`
  모듈이 새 폴더 경로로 정상 임포트되는 것 재확인. 계산 로직 자체는 전혀 안 건드린
  순수 경로/번호 재배치라 실제 파이프라인 재실행(수 시간 소요)까지는 하지 않음 —
  10_eval/9_train 라운드 때처럼 산출물을 재현해 수치로 비교할 대상 자체가 없는 종류의
  변경(로직 무변경, 껍데기만 재배치).

### 2026-09-24 — `4_hi_analysis/` 코드 정리 (전면 축소 라운드)

Step 6~10 renumbering을 마친 뒤 이어서 진행. Phase 2 파일럿(2026-09-22, `hi_correlation.py`
도달불가 코드 제거 + tools/ 12개 이동)의 후속으로, `hi_correlation.py`(1808줄)/
`hi_segment_viz.py`(648줄)/`hi_compute.py`(867줄) 3개 파일을 전수 점검했다(Explore
서브에이전트로 정찰 후 직접 검증·수정).

- **버그 발견 및 수정: `--seg-axis` 기본값이 `parameters.py`와 조용히 어긋나 있었음.**
  `hi_correlation.py`의 `--seg-axis` 기본값이 하드코딩 `"qfrac"`(구식 축)이었는데
  `--axis-config` 기본값은 `P.ACTIVE_AXIS_CONFIG`(`q_frac_ref` 축 스키마: n1/n2/ref_lag/
  noise_amp/...)라, 인자 없이 그냥 실행하면 `QFracSegmenter.__init__() got an
  unexpected keyword argument 'n1'`로 **즉시 크래시**했다(실제 재현 확인). 이전 세션에
  `--axis-config` 쪽은 이미 고쳤지만(Phase 3, "DEFAULT_AXIS_CONFIG 지뢰") `--seg-axis`
  이름 자체의 기본값은 그때 안 잡혔던 것 — `P.FIXED_SEG_AXIS`로 교체해 수정, 재현
  테스트로 더 이상 크래시 안 하는 것 확인. `run_pipeline.py`를 통해 실행할 때는 Step 4가
  `--seg-axis`를 항상 명시적으로 주입해서 이 버그가 실전에서 드러난 적은 없었다(직접
  스크립트를 인자 없이 돌릴 때만 재현).
- **`--workers` 기본값도 같은 부류의 버그**: 하드코딩 `max(1, cpu-2)`였는데
  `parameters.py`(`ACTIVE_WORKERS` 주석)엔 "Step 1~5 스크립트 전부 `min(P.ACTIVE_WORKERS,
  os.cpu_count())`로 통일했다(2026-09-23)"고 돼 있었음 — 같은 폴더의
  `hi_segment_viz.py`는 이미 올바르게 `min(...)` 패턴이었는데 `hi_correlation.py`만
  누락. `min(P.ACTIVE_WORKERS, os.cpu_count() or 1)`로 교체.
- **`--dataset-group` 기본값 갭 해소**: `parameters.py`에 이미 "`--dataset-group`은 값
  체계가 달라 별도 상수 필요"라는 주석이 있었지만 그 상수가 실제로는 없었음(설계
  의도만 있고 미구현) — `FIXED_DATASET_GROUP = "lfp"` 신설 후 연결.
- **독스트링 전면 재작성**: 모듈 상단 독스트링이 실제로 존재하지 않는 CLI 플래그
  (`--dataset`, `--n-top`, `--cell`, `--cycle`, `--curve-debug`, `--cycles`,
  `--n-cycles`, `--plateau-debug`, `--plateau-summary` — grep으로 전수 확인, 코드
  어디에도 없음)를 문서화하고 있었고, 실행 예시 축 5개(qfrac/protocol/vwindow/rcs/
  cluster) 중 정작 **정식(v4) 축인 `q_frac_ref`는 언급조차 없었다** — 실제 레지스트리엔
  13개 축(`common.scenario.REGISTRY`)이 있는데 그중 정식 축이 빠진 예시만 있었던 것.
  실제 CLI 표면에 맞춰 다시 쓰고, `q_frac_ref`를 "정식/v4 프로덕션 축"으로 명시, 나머지는
  "과거 실험/대조군용"으로 정리. `--seg-axis` 도움말도 13개 축 전체로 갱신.
- **`cluster` 축 미완성 상태 문서화**: 코드 자체가 `fit()` 없이 쓰면 전부 cluster 0으로
  분류된다는 `[경고]` 로그를 이미 내고 있었고, 이 파이프라인 어디에도 `fit()`을 호출하는
  경로가 없어 실질적으로 미완성 기능임을 확인(`4_hi_analysis/tools/cluster_scatter.py`는
  이름만 비슷할 뿐 완전히 별개의 독립 KMeans 진단 도구라 무관). 코드는 그대로 두고
  독스트링에 "실사용은 q_frac_ref만"이라고 명시.
- **삭제하지 않고 유지한 것**: protocol/vwindow/rcs/cluster/q_frac_wide/q_abs/vqslope/
  full_cycle 등 8개 비-정식 축 지원 코드 전체 — `common/scenario/`에 전부 정식 등록돼
  있고, `RESULTS_LOG.md`엔 실제 호출 기록이 없지만(직접 스크립트 호출은 애초에 로그를
  안 남김) parameters.py 자체가 "이번 세션은 q_frac_ref만 썼다"고 명시하는 걸 보면 *과거*
  세션/어블레이션에서 쓰였을 가능성이 높다 — 7_synergy의 `--shuffle-from`과 같은
  "죽은 것처럼 보이지만 재현성 인프라" 함정으로 보고 보수적으로 유지.
  `hi_correlation.py`/`hi_compute.py` 간 코드 중복은 이미 이전 파일럿에서 해소됐음을
  재확인(HI 계산 함수는 전부 `hi_compute.py`에서 import, `hi_correlation.py`는 자기
  전용 최적화 배치 DTW(`_dtw_batch`)와 Global HI(G01–G15)만 로컬 정의 — 의도된 비중복).
- **개명 없음**: 세 파일 다 이미 짧고 명확한 이름이라(`hi_correlation.py`/
  `hi_segment_viz.py`/`hi_compute.py`) `test.py`/`train.py`류 개명 대상이 아님.
- **검증**: `py_compile` + `PYTHONIOENCODING=utf-8`로 `--help` 실제 실행해 새 도움말
  텍스트가 의도대로 나오는 것 확인, `get_segmenter(P.FIXED_SEG_AXIS, ...)` 재현 호출로
  크래시가 사라진 것 확인. 이번 라운드는 인자 기본값·문서만 고쳤고 HI 계산 함수 자체는
  전혀 안 건드려서(전부 `hi_compute.py`에서 import, 무변경) 수치 비교 검증은 필요
  없다고 판단(변경 자체가 "설정 기본값이 맞았는지"라 재현할 산출물이 따로 없음) —
  `--help`의 cp949 콘솔 크래시는 이 파일에 이미 있던 em-dash 20곳 때문에 내 편집과
  무관하게도 재현되는 것 확인(8_kernel 라운드 때와 동일한 기존 환경 이슈, 그때처럼
  스코프 밖으로 둠).

### 2026-09-24 — Stage0(`model_lib/legacy/`) + 비-정식 시나리오 축 일괄 삭제

전체 코드베이스 대상 "리팩토링 전문가 관점" 죽은 코드 감사(Explore 서브에이전트 2개
병렬)를 거친 뒤, 사용자가 "Stage0는 더는 안 쓸 것이고 git 히스토리에 남아있으니
최대한 깔끔히 지우자, scenario axis도 q_frac_ref 제외 전부 지우자"고 명시적으로
지시해 실행. 둘 다 git 히스토리로 복원 가능하다는 전제로 작업 트리에서 완전히
삭제(마킹/주석 처리가 아님).

**비-정식 시나리오 축 삭제(`common/scenario/`)**:
- 삭제 파일(8개): `protocol.py`/`vwindow.py`/`rcs.py`/`cluster.py`/`test_rs.py`/
  `vqslope.py`/`q_abs.py`/`full_cycle.py`.
- **삭제 전 의존성 그래프 확인이 핵심이었음** — `QFracRefSegmenter`(정식 축)가
  `QFracWideSegmenter`를 상속하고(`q_frac_ref.py: class QFracRefSegmenter
  (QFracWideSegmenter)`), `QFracWideSegmenter`는 `_random_seg.py`의
  `sample_random_windows`를 쓴다 — 그래서 `q_frac_wide.py`/`qfrac.py`(아래 참고)/
  `_random_seg.py`/`_curves.py`/`base.py`는 "비-정식 축"이지만 파일은 그대로 뒀다.
  삭제 전 `from .` 상대 임포트 전수 grep으로 확인 후 실행 — 안 그랬으면 q_frac_ref
  자체가 깨질 뻔했다.
- **`qfrac.py`는 특히 위험했다**: `--seg-axis qfrac`으로 직접 선택하는 용도는 없앴지만
  (REGISTRY에서 뺌), `QFracSegmenter` 클래스 자체는 `model_lib/datasets/
  segment_dataset.py`(`_DEFAULT_SPEC = spec_from_qfrac()`, **모듈 import 시점에 실행**)
  와 `model_lib/models/scr_model.py`(생성자 fallback)가 내부적으로 기본 ScenarioSpec
  용도로 계속 쓰고 있었다 — grep 없이 지웠으면 모델/데이터셋 모듈 자체가 import부터
  깨졌을 것.
- **`_detect_cv_start`(CC→CV 전환 검출) 구조: vwindow.py에 있었지만 축과 무관하게
  `hi_correlation.py --exclude-cv`가 재사용** — `common/scenario/_curves.py`(살아남는
  공용 헬퍼 모듈, `seg_diagnose.py` 도구가 `_build_vq_curve`로 이미 씀)로 옮기고 나서
  vwindow.py 삭제.
- `common/scenario/__init__.py`를 `REGISTRY = {"q_frac_ref": QFracRefSegmenter}`
  하나만 남도록 재작성.
- `4_hi_analysis/hi_correlation.py`: `_qabs_tag`/`_vqslope_tag` 삭제(`_qfw_tag`는
  `_qfref_tag`가 내부적으로 호출해서 유지), q_frac_wide/q_abs/vqslope/qfrac별
  캐시경로 분기 3곳을 q_frac_ref 단일 분기로 단순화, `vwindow`/`cluster` 축 전용
  특수 처리 블록(이제 영원히 도달 불가) 제거, 독스트링/`--seg-axis` 도움말을
  "q_frac_ref만 지원"으로 재작성.
- `4_hi_analysis/hi_segment_viz.py`: 삭제된 `_vqslope_tag`를 import하던 분기 제거,
  독스트링/도움말 갱신.
- `4_hi_analysis/tools/axis_comparison.py`/`k2_boundary.py` **삭제** —
  vqslope/vwindow/cluster를 직접 import해서 축 삭제로 즉시 ImportError가 나는
  상태였고, 두 스크립트의 존재 목적 자체가 "지금 지운 축들을 비교/분석"이라 남겨둘
  이유가 없었음(`cluster_scatter.py`/`k_selection_diagnostics.py`는 CSV만 소비해
  import 의존이 없어 그대로 둠).
- `5_interaction/interaction.py`, `6_synergy/synergy.py`, `7_kernel/kernel.py`,
  `8_train/train.py`, `9_eval/test.py`는 애초에 축 이름을 하드코딩하지 않고
  `get_segmenter()`에 그대로 넘기는 방식이라 **코드 변경 없음**(q_frac_ref 외 축을
  주면 자연히 `ValueError`).

**Stage0(`model_lib/legacy/`) 삭제**:
- `train_scr.py`(883줄)/`test_scr.py`(693줄) 전체와 `model_lib/legacy/` 디렉터리
  자체를 삭제. 두 파일 다 실제로는 각각 4개/2개 함수만 v4 파이프라인이 쓰고
  나머지(85~90%)는 이미 각 파일 자신의 `--phase 1/2` 독립 CLI로만 닿는 죽은
  코드였음(이전 세션 Explore 감사 결과) — 이번에 파일째 정리.
- **함수 이전**: 여러 소비자가 쓰는 5개 함수(`_ranked_indices`/
  `_save_probe_masks_to_json`/`_save_scen_masks_to_json`/`_plot_gate_probs`/
  `_load_synergy_group_ids`)는 신설 `model_lib/utils/gate_io.py`로 옮김
  (`8_train/train.py`가 저장 3종을, `model_lib/tools/visualize_results.py`가
  `_load_synergy_group_ids`를 재사용 — model_lib/ 공유 규칙 그대로 적용). 단일
  소비자뿐인 2개 함수(`_resolve_device`/`_pick_rep_cells`, `9_eval/test.py`
  전용)는 공유 모듈을 새로 안 만들고 `test.py` 로컬 함수로 인라인.
- `8_train/train.py`/`9_eval/test.py`/`model_lib/tools/visualize_results.py`의
  `sys.path.insert(..., "model_lib" / "legacy")` 및 `import train_scr as _base`/
  `import test_scr as _tbase` 전부 제거, 호출부(`_base.X`/`_tbase.X`) 일괄 치환.
- **연쇄로 드러난 추가 죽은 코드**: `model_lib/datasets/segment_dataset.py`의
  `build_random_seg_dataset()`(랜덤 세그먼트/`test_rs` 평가용, 26줄)가
  `test_scr.py`의 `_run_random_segment_test`가 유일한 소비자였다는 걸 grep으로
  발견 — 같이 삭제. `parameters.py: P1_MODEL_CONFIG`의 `"test": {random_segment_test,
  random_seg_data_dir, random_seg_datasets}` 섹션도 이제 아무도 안 읽어서 같이 제거
  (`cfg["test"]`를 읽는 코드가 정말 하나도 없는지 전체 grep으로 확인 후 실행).
- `docs/phase1_lab/RESULTS_LOG.md`, 각 파일의 과거 사실을 서술하는 주석/독스트링
  (예: "train_scr.py --phase 2로 생성된...")은 기존 원칙대로 안 건드림 — 실제
  `import train_scr`/`import test_scr` 구문만 전부 제거했는지 grep으로 재확인.

**검증**:
- `py_compile` 전체 + `common.scenario.get_segmenter("q_frac_ref", ...)` 실제 호출로
  기존 canonical axis-config가 여전히 작동하는 것, `get_segmenter("vwindow", ...)`가
  의도대로 `ValueError`를 내는 것 확인.
- `utils.hi_schema.spec_from_qfrac()` 재현 호출로 qfrac.py 내부 의존성이 안 깨진 것
  확인.
- `train.py`/`test.py`/`visualize_results.py` 세 파일을 `importlib`로 직접
  모듈 전체 실행(모듈 레벨 import 전부 통과 확인) — 세 파일 다 새 `gate_io.py`/
  로컬 함수 경로로 문제없이 로드됨.
- **가장 강한 검증**: 원본 기준 run(`0921_1953_..._seed0`)과 완전히 동일한 조건
  (seed=0, split-seed=0, 동일 kernel-pkl/combined-redundancy-json/interaction-json/
  axis-config/data-dir/seg-data-dir, `--model-config` 미지정 → `parameters.py:
  P1_MODEL_CONFIG` 기본값 사용, 단 `"test"` 섹션은 뺀 상태)으로 전체 재학습을 처음부터
  끝까지 다시 실행했다(`model_lib/results/p1v4_scen_lag1zone_redunfix_verify2_seed0/`).
  **결과**: 조기종료 416에폭·선택 epoch 355·gate_saturation=0.0041 전부 원본과 동일,
  `p1v2_summary.json`의 경로 필드(태그/kernel-pkl 등, legacy_results 개명으로 값 자체가
  바뀌는 게 당연한 필드)를 뺀 모든 필드가 완전 일치(불일치 0개), **체크포인트
  `best_by_saturation.pt`의 MD5가 원본과 완전히 동일**(`32c2cd84216510655ce3c9f15072ee70`).
  Stage0/비-정식 축 삭제와 `P1_MODEL_CONFIG`에서 `"test"` 섹션을 뺀 것 둘 다 v4 학습
  결과에 정말로 아무 영향이 없었다는 걸 가장 강한 방식(바이트 단위 재현)으로 증명.

### 2026-09-25 — CNN 분류기/CNN 임베딩 잔재 삭제

사용자가 "cnn 로직 모두 제거됐나?"라고 물어봐서 재감사한 결과, 서로 다른 두 개의
CNN 잔재가 아직 남아있었음을 확인하고 정리:

- **`model_lib/models/scenario_classifier.py`(350줄) 전체 삭제**
  (`RuleClassifier`/`CentroidClassifier`/`OracleClassifier`/`NoneClassifier`/
  `CNNProbeClassifier` + `build_classifier()`). 구 2단계 아키텍처("시나리오 분류기
  학습 → 그 결과로 회귀 헤드 미세조정")의 잔재 — 그 분류기 학습 스텝 자체는 이
  세션보다 훨씬 전에 이미 파이프라인에서 빠졌고(`run_pipeline.py` 자체 히스토리
  주석에 그 사실이 남아있음), v4는 `SCRModel.probe_mlp`로 완전히 대체했다.
  `build_classifier()` 호출자 repo 전체 0곳, `SCREvaluator.set_classifier()`도
  `test.py`에서 `model.probe_mlp`만 넘기는 것으로 grep 재확인 후 삭제.
- **`model_lib/evaluation/scr_evaluator.py`**: `CNNProbeClassifier` isinstance 분기
  제거 — `self._classifier`가 실제로 `CNNProbeClassifier`였던 적이 없어(항상
  `probe_mlp`) 그 분기는 한 번도 안 탄 죽은 코드였다. else 경로(`probe_x+direction
  concat`)만 남김.
- **`model_lib/models/scr_model.py`**: `with_raw_cnn=True`일 때 `models/raw_cnn.py`
  에서 `RawCNN`을 로드하는 블록 삭제. **이 파일은 repo 어디에도 실제로 존재하지
  않았다** — `with_raw_cnn=True`일 때만 import되는 경로라 지금까지 한 번도 실행된
  적 없이 방치된, 켜면 즉시 `ModuleNotFoundError`가 나는 깨진 코드였다. v4는
  `train.py`/`test.py`가 어차피 `with_raw_cnn`을 항상 강제 `False`로 덮어써서
  실사용 경로도 아니었음. `self.with_raw_cnn`/`self._raw_cnn_frozen`/`self.raw_cnn`은
  forward()/`train()` 오버라이드/`visualize_results.py`가 여전히 참조하므로 "항상
  꺼짐" 상수로만 남김(다른 파일은 무수정) — `with_raw_flat`(별개의, 안 깨진 기능)은
  그대로 유지.
- **`parameters.py: P1_MODEL_CONFIG`**: `classifier.type`("cnn")/
  `classifier.is_auto_mk_selection`/`classifier.probe_m_count`(전부 어디서도 안
  읽음, 구 분류기 프레임워크 전용) 및 `model.with_raw_cnn`/
  `model.raw_cnn_pretrained_from`(위와 동일 이유) 삭제. `classifier.charge_probe_m`/
  `discharge_probe_m`(train.py의 probe gate top-m 선정에 실제로 쓰임)과
  `model.with_raw_flat`은 유지.
- **검증**: `py_compile`/`compileall` 전체 통과, `SCRModel`/`SCREvaluator` 실제
  import 재현, `P1_MODEL_CONFIG['classifier']`가 `{charge_probe_m, discharge_probe_m}`
  둘만 남은 것 확인. 그 뒤 원본과 완전히 동일한 조건으로 세 번째 전체 재학습을
  실행(`p1v4_scen_lag1zone_redunfix_verify3_seed0`) — 조기종료 416에폭·선택
  epoch 355·gate_saturation=0.0041 전부 동일, `p1v2_summary.json` 전 필드 완전
  일치(불일치 0개), **체크포인트 MD5 완전 일치**(`32c2cd84216510655ce3c9f15072ee70`,
  9월 24일 검증본·원본 셋 다 동일). CNN 분류기/CNN 임베딩 삭제도 학습 결과에
  실제로 아무 영향이 없었음을 바이트 단위로 확인.

### 2026-09-25 — `--datasets` 단일화 + `parameters.py` 의미 중복 파라미터 통합

사용자가 "각 스텝이 자기만의 파라미터를 안 가지는 게 맞냐"고 물어서 전체 9개
스텝의 argparse 기본값을 재점검한 결과 `--datasets nargs="+" default=["MIT",
"HUST"]`가 `interaction.py`/`synergy.py`/`kernel.py` 세 곳에 하드코딩으로 남아있던
걸 발견(그 자리에서 정리 예고) — 이어서 "parameters.py 안에서 의미가 같아 통합할
수 있는 파라미터가 있는지" 감사 요청을 받아 같이 처리:

- **`FIXED_CANONICAL_DATASETS = ["MIT", "HUST"]` 신설**: `interaction.py`/
  `synergy.py`/`kernel.py`의 `--datasets` 기본값 + `P1_MODEL_CONFIG["data"]
  ["datasets"]`(기존엔 별도로 `["MIT","HUST"]` 리터럴 반복) 4곳을 이 상수 하나로
  통합. `P1_MODEL_CONFIG` 쪽은 값 변형 방지를 위해 `.copy()`로 참조(다른 축
  설정 필드와 동일 관례). `FIXED_DATASET_GROUP="lfp"`(Step 4 전용, 값 형식이
  "그룹 이름"이라 다름)는 의미상 겹치지만 통합 대상에서 제외 — 근거를 주석으로
  남김.
- **`ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD`/`FIXED_KERNEL_REDUNDANCY_THRESHOLD`
  통합**: 둘 다 값이 `0.9`였고, `kernel.py`/`synergy.py` 양쪽 독스트링에 이미
  "동일 임계값 재사용"이라고 명시돼 있던 걸 발견 — `docs/phase1_lab/
  RESULTS_LOG.md`의 실제 호출 36회 전부 `--redundancy-threshold 0.9`로 일치하는
  것까지 확인 후 `FIXED_KERNEL_REDUNDANCY_THRESHOLD` 삭제, `kernel.py`가
  `ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD`를 직접 참조하도록 변경.
  `run_pipeline.py`의 Step 7(kernel.py) 주입부도 상수 직접 참조 대신
  `args.synergy_redundancy_threshold`(Step 6 synergy.py와 동일 CLI 플래그로 이미
  해석된 값)를 쓰도록 바꿔서, 이제 `--synergy-redundancy-threshold` CLI 플래그
  하나가 두 스텝을 동시에 제어한다(전에는 우연히 같은 값일 뿐 서로 독립적으로
  바꿀 수 있어 드리프트 위험이 있었음).
- **통합하지 않기로 한 것들(검토 후 기각, 근거)**: `ACTIVE_SEED`/`ACTIVE_SPLIT_SEED`/
  `FIXED_SHUFFLE_SEED`가 셋 다 기본값 42로 같지만, 서로 다른 무작위성(모델 초기화
  RNG / 셀 분할 / v-ctrl 대조군 재배정)을 제어해 통합하면 "시드 3개를 독립적으로
  바꾸는 멀티시드 실험"이 불가능해짐 — 값이 같은 건 관례일 뿐 의미가 다름.
  `ACTIVE_MAX_EPOCHS`/`ACTIVE_PATIENCE`/`ACTIVE_BATCH_SIZE`(모두 `None`=오버라이드
  없음) vs `P1_MODEL_CONFIG["training"]`의 `epochs`/`batch_size` 등은 "CLI
  오버라이드값 vs 실제 기본값"의 의도된 계층 구조라 통합 대상 아님.
- **부수 발견(이번엔 안 건드림)**: 같은 감사 중에 `model_lib/training/scr_trainer.py`의
  `SCRTrainer` 클래스 자체가 이제 아무 데서도 인스턴스화되지 않는다는 걸 확인
  (`train.py`는 그 파일의 `L0LambdaScheduler`만 import) — Stage0(`train_scr.py`)가
  삭제되면서 유일한 호출자가 사라진 것. 그래서 `P1_MODEL_CONFIG["training"]`의
  `early_stop_patience`/`log_interval`/`run_overfit_test`/`overfit_test_samples`/
  `overfit_test_epochs`(전부 grep 0회, `SCRTrainer.__init__`에서만 읽힘)와
  `"scheduler": "cosine"`(값은 기록되지만 `train.py`가 실제로는 분기 없이 항상
  `CosineAnnealingLR`만 씀)도 죽은 값으로 보인다 — 이번 "파라미터 통합" 스코프는
  아니라 정리 안 하고 기록만 남김(다음 라운드 후보).
- **검증**: `py_compile`/실제 import 재현 + 세 스크립트 `--help` 실행으로 새 기본값이
  반영된 것 확인. 순수 인자 배선 변경(어느 실제 실행에서도 값 자체는 안 바뀜 —
  `--datasets`는 항상 `["MIT","HUST"]`였고 `--redundancy-threshold`도 항상 `0.9`였음)
  이라 전체 재학습 재검증은 하지 않음.

### 2026-09-25 — 시드 3종 통합 + `SCRTrainer`/Laplace UQ 체인 전체 삭제

사용자가 "시드도 그냥 통합하자"(위 라운드에서 의미가 달라 기각했던 것)와 "SCR
트레이너/SCR 테스트도 이제 안 쓰는데 다 제거"를 명시적으로 지시.

**시드 통합**: `ACTIVE_SEED`/`ACTIVE_SPLIT_SEED`가 `None`일 때의 인라인 폴백 `42`와
`FIXED_SHUFFLE_SEED = 42`가 6개 파일(`interaction.py`/`synergy.py`/`kernel.py`/
`train.py`/`run_pipeline.py` 2곳)에 리터럴로 반복되던 걸 `FIXED_DEFAULT_SEED = 42`
하나로 통합 — 세 시드가 서로 다른 무작위성(모델 초기화/셀 분할/v-ctrl 재배정)을
제어한다는 사실 자체는 그대로 두고(각 CLI 플래그는 계속 독립적으로 오버라이드
가능, 멀티시드 실험 영향 없음), "아무것도 안 줬을 때 쓰는 공통 기본값"만 한
군데로 모음.

**`SCRTrainer`/Laplace UQ 체인 삭제**: `SCRTrainer`(옛 `fit()`/`fit_laplace()`/
`run_overfit_test()` 루프, `model_lib/training/scr_trainer.py`)의 유일한 호출자가
이미 삭제된 `train_scr.py`(Stage0)였다는 걸 grep으로 재확인(`train.py`는 같은
파일의 `L0LambdaScheduler`만 import) — 삭제 과정에서 딸려있던 죽은 사슬을 끝까지
추적:
- `SCRTrainer` 클래스 전체(`fit`/`fit_laplace`/`run_overfit_test`/`_train_epoch`/
  `_val_epoch`/`_build_scheduler`/`_lr_warmup`/`_collect_small_batch` + 모듈
  헬퍼 `_save_model`/`_load_model`/`_init_log_csv`/`_append_log_csv`) 삭제,
  `L0LambdaScheduler`만 남김. `model_lib/training/__init__.py`의
  `from training.scr_trainer import SCRTrainer`도 같이 고침(어차피 아무도
  `training.SCRTrainer`로 접근 안 했지만 그대로 두면 이제 ImportError).
- `fit_laplace()`의 유일한 소비자가 사라지면서 `model_lib/utils/uncertainty.py`
  (`LaplaceUQ`/`calibration_metrics`/`plot_calibration`, 444줄) 전체가 고아가 된
  것도 grep으로 확인(`train.py`/`test.py` 어디에도 `uq` 관련 코드 0줄) — 파일
  통째로 삭제.
- `model_lib/evaluation/scr_evaluator.py`의 `predict_dataset_uq`/
  `save_uq_metrics`/`plot_uq` 3개 메서드도 같은 이유로 삭제(`test.py`가 호출한
  적 없음 — `SCREvaluator` 나머지 부분은 `test.py`가 실제로 쓰는 살아있는
  클래스라 그대로 유지).
- `parameters.py: P1_MODEL_CONFIG`에서 `"uq"` 섹션 전체와 `"training"`의
  `scheduler`/`log_interval`/`run_overfit_test`/`overfit_test_samples`/
  `overfit_test_epochs`/`early_stop_patience`(전부 `SCRTrainer.__init__`
  에서만 읽히던 값, `train.py` 자체 루프는 0회 참조 — `scheduler`는 특히
  "cosine"이라고 적혀만 있을 뿐 분기 코드 자체가 없었음, 항상
  `CosineAnnealingLR` 고정) 삭제.
- **`model_lib/legacy/test_scr.py`는 이미 이전 라운드(Stage0 삭제)에서 지워진
  상태** — "SCR 테스트"는 그 재확인으로 처리(추가로 지울 파일 없음).
- **삭제하지 않고 확인만 한 것**: `SCREvaluator`(`scr_evaluator.py`)와
  `SCRLoss`(`scr_loss.py`)는 `test.py`/`train.py`가 각각 실제로 쓰는 살아있는
  클래스라 전혀 안 건드림. `hard_concrete.py`도 무관.
- **검증**: `py_compile`/`compileall` 전체 통과, `train.py`/`test.py`/
  `visualize_results.py` 세 파일 전부 `importlib`로 모듈 전체 실행 재현(모듈
  레벨 import 경로 전부 통과), `SCREvaluator`에 `predict_dataset_uq` 속성이
  없어진 것 확인. 그 뒤 원본과 동일 조건으로 네 번째 전체 재학습 실행
  (`p1v4_scen_lag1zone_redunfix_verify4_seed0`) — 조기종료 416에폭·선택 epoch
  355·gate_saturation=0.0041 전부 동일, `p1v2_summary.json` 전 필드 완전
  일치(불일치 0개), **체크포인트 MD5 완전 일치**(`32c2cd84216510655ce3c9f15072ee70`,
  이번까지 네 번째 검증본 전부 원본과 동일). 시드 통합과 SCRTrainer/UQ 체인
  삭제 둘 다 학습 결과에 실제로 아무 영향이 없었음을 바이트 단위로 확인.

### 2026-09-26 — `4_hi_analysis/`(hi_correlation/hi_compute) 죽은 코드 재감사

사용자가 "4_hi_correlation부터 엄밀하게" 지금 안 쓰는 로직이 있는지 재검사해달라고
요청 — 지금까지의 CNN/UQ/SCRTrainer 삭제 사슬이 hi_correlation.py 쪽까지 이어지는지
확인. 이번엔 함수 단위 reachability(전체 repo grep)로 찾아서 보고 후 사용자가 3개
전부 승인해 실행:

1. **`hi_correlation.py` 안 쓰는 import 4개 삭제**: `ThreadPoolExecutor`,
   `find_peaks`, `sp_kurtosis`(`scipy.stats.kurtosis`), `sp_skew`(`scipy.stats.skew`)
   — import문 외엔 파일 어디서도 안 쓰임(아마 예전에 실제 계산 로직을
   `hi_compute.py`로 옮기면서 남은 흔적).
2. **`_resample_segment()` + `raw_v`/`raw_i`/`raw_t` 컬럼 삭제(CNN/UQ 삭제 사슬의
   연장)**: `_extract_one_cell` 안 2곳에서 모든 세그먼트마다 실제로 계산·저장하던
   원시 곡선 데이터인데, 소비 경로가 `with_raw_cnn`(이미 삭제)과 `with_raw_flat`
   뿐이었고 **`train.py:492`/`test.py:271`가 `with_raw_flat`도 `with_raw_cnn`과
   똑같이 항상 강제 `False`로 덮어쓰고 있었다**(지난 CNN 정리 때는 놓쳤던 부분,
   이번에 발견). `RAW_N` 상수도 이 함수 전용이라 같이 삭제.
3. **`hi_compute.py`의 `compute_his()`/`benchmark_cost()`/`morph_reference()` +
   연쇄로 고아가 된 `morph_curves()`/`_morph_dict()` 삭제**: `hi_correlation.py`는
   이 "전체를 한 번에 계산하는 편의 API"를 애초에 쓴 적이 없고 — 자기 자신의
   `_curve_buf`/`_dtw_batch()`(배치 최적화 버전)로 DTW/Fréchet morph HI를 독립적으로
   계산한다. 5개 함수 전부 서로만 호출하는 닫힌 죽은 루프였고(`hi_names()`만
   `__main__` 자체 테스트에서 살아있어 유지), 외부 호출자 0곳을 grep으로 확인.
   `benchmark_cost()`만 쓰던 `import time`도 같이 삭제. 모듈 독스트링의 stale
   `compute_his(..., bol=...)` 예시도 실제 구조(hi_correlation.py의 자체 배치
   구현)를 설명하도록 갱신.
- **검증**: `py_compile` 전체 통과. `hi_compute` 모듈 재현 import로 삭제된 3개
  함수가 실제로 없어진 것, `hi_names()`는 여전히 66개를 정확히 반환하는 것 확인.
  **가장 중요한 검증**: 이 변경은 추출(Step 4) 로직 자체를 건드려서, 캐시된
  기존 데이터로 도는 학습 재현 검증(MD5)으로는 못 잡는 종류라 — MIT `b1c0` 셀
  하나를 실제로 처음부터 재추출해서(`_extract_one_cell` 직접 호출) 기존 캐시된
  결과와 비교: 세그먼트 행 수 완전 일치(21828건), `raw_v` 컬럼이 새 추출본에는
  의도대로 없어진 것 확인, `stat_*` 8개 컬럼 전부 완전 일치(불일치 0/8) — 남은
  계산 로직은 전혀 안 건드렸다는 걸 실측으로 증명.

### 2026-09-27 — "현재 검증 버전 조건" 기준 Bucket A 죽은 코드 전량 삭제

사용자가 "지금 검증수행하는 버전 조건대로 실행하는데 있어서 실행되지 않는 코드
모두 찾아서 보고" 요청 → repo 전체를 두 개 병렬 Explore 서브에이전트로 감사해
**Bucket A**(현재 config로는 어떤 경로로도 절대 안 타는 코드 — 삭제 안전)와
**Bucket B**(non-canonical yaml/CLI 오버라이드로만 닿는, 이 프로젝트의 실제
ablation 실험 인프라 — 논문용이라 보존 필수, 예: `TransformerHead` 계열
회귀 헤드, `assign="none"`/`tile_scope`/`random_segment` 등 축 파라미터,
`--hi-cost-weighted-l0`)로 분리해 보고 → 사용자가 Bucket A 전량 삭제 승인.

**`model_lib/models/cap_heads.py`**: `with_raw_flat` 스캐폴딩 전체 삭제 —
`_D_RAW_FLAT`/`_HEAD_IN_WITH_RAW_FLAT` 상수, `TransformerHead`의
`with_raw_flat` 파라미터/`raw_flat_embed`/forward 분기, `build_cap_head()`의
`n_scen_onehot`/`with_raw_flat` 파라미터와 관련 검증 로직 전부 삭제.
`with_raw_cnn`은 이미 지난 라운드에 `train.py:492`/`test.py:271`가 항상 강제
`False`로 덮어써서 죽어있는 게 확인됐었는데, **`with_raw_flat`도 정확히 같은
이유로 100% 죽어있었다**(이번 라운드에서 새로 발견).

**`model_lib/models/scr_model.py`**: 3개 항목 삭제.
- `n_gate_groups`/`scenario_onehot` 생성자 파라미터와 관련 초기화 로직(게이트
  뱅크 축소 + "웜스타트 후 분기" 커리큘럼 메커니즘) 전체 삭제 — 이 기능을 켜는
  CLI 플래그가 애초에 존재한 적이 없었음(grep으로 외부 호출자 0곳 확인).
  `build_cap_head()` 호출의 `n_scen_onehot` 인자, `train()` 메서드 오버라이드
  (raw_cnn 동결용이었는데 조건이 영구 `False`라 상속 기본 동작과 100% 동일),
  `with_raw_flat` 스캐폴딩(`raw_flat_norm`/forward elif 분기)도 함께 삭제.
- `_apply_scen_gate()`/`_apply_scen_kernel_gate()` 안의
  `if self._gate_group_map is not None: scen_idx = self._gate_group_map[scen_idx]`
  가드 2곳 삭제 — 위 항목에서 `_gate_group_map`을 대입하는 코드를 지웠는데 이
  가드를 놓치면 첫 forward 호출에서 `AttributeError`가 나는 걸 자체 재현
  테스트로 잡아서 즉시 수정.
- `branch_scen_gates()` 메서드 전체 삭제(위 게이트-그룹 메커니즘의 일부, 외부
  호출자 0곳).
- **연쇄 발견**: `model_lib/utils/gate_io.py`의 `_save_scen_masks_to_json`/
  `_plot_gate_probs` 2곳이 `getattr(model, "_gate_group_map", None)`으로 이제
  영구 `None`인 속성을 방어적으로 읽고 있던 것도 함께 단순화(`g = s` 직접 대입).
  `8_train/train.py`/`9_eval/test.py`의
  `p1_model_cfg = {**cfg["model"], "with_raw_cnn": False, "with_raw_flat": False}`
  강제 오버라이드도 — `SCRModel`이 이제 두 키를 `model_cfg`에서 아예 안 읽으므로
  완전한 no-op이 됨 — `dict(cfg["model"])`로 단순화. `parameters.py`의
  `P1_MODEL_CONFIG["model"]["with_raw_flat"]` 키도 같이 삭제.

**`model_lib/datasets/segment_dataset.py`**: `aux_scen_target`/
`aux_intensity_target`(h_scen/h_intensity 보조손실 타깃 — 소비하던 CNN 분류기
학습 경로가 이미 삭제됨) 계산·저장·`__getitem__` 반환·`FastTensorLoader`의
`include_aux` 옵션까지 전부 삭제. `filter_dataset_by_cells`/`_subset_dataset`
(외부 호출자 0곳), 모듈 레벨 `collate_fn`(표준 `DataLoader` 미사용 —
`FastTensorLoader`가 대체)도 삭제. `datasets/__init__.py`의 재수출도 정리.

**`model_lib/evaluation/scr_evaluator.py`**: `evaluate()`/`plot_for_dataset()`
(둘 다 `test.py`가 직접 `evaluate_modes()`/`_plot_*` private 메서드를 쓰지
이 래퍼들은 호출한 적 없음), `save_predictions()`/`plot_routing_heatmap()`
(`test.py`가 자체 `_write_predictions_csv`/routing_table.csv 저장 로직을
따로 갖고 있어서 완전히 중복·미사용), `plot_hi_category_heatmap()`(전용 색상
상수 `_CAT_PREFIX`/`_CAT_COLOR`/`_CAT_LABELS`와 함께) 삭제. `predict_dataset()`의
`direction_routing`/`_dir_t` 분기(예전 `test_rs` 축의 scen_idx 0/1 체계 전용,
그 축 자체가 이미 삭제됨)와 `evaluate_modes()`의 `direction_routing_for_oracle`
파라미터(호출부 `test.py`가 넘긴 적 없어 항상 `False`)도 삭제. 안 쓰는
`import csv`/`get_hi_cols_for_seg` 정리, 모듈 독스트링을 실제 산출 책임
기준으로 갱신.

**기타 단일 함수/상수 삭제**: `model_lib/utils/metrics.py`의 `routing_stats()`
(외부 호출자 0곳), `model_lib/utils/hi_schema.py`의 `LEAK_COLS`(주석에 이미
"현재 어디서도 참조 안 함"이라고 적혀 있던 세트), `model_lib/utils/io_utils.py`의
`save_checkpoint`/`load_checkpoint`/`save_json`/`save_pickle`/`load_pickle`
(전부 `utils/__init__.py` 재수출 외 실제 호출자 0곳 — `torch.save`/`json.dump`류를
직접 쓰는 `train.py`/`test.py`가 실제 체크포인트 저장을 담당), `1_convert/
convert_unified.py`의 `Qd`/`Qc` HDF5 필드 파싱(`_build_cell_df`가 애초에
안 읽음), `common/scenario/q_frac_wide.py`의 `cv_v_thresh`/`cv_cc_frac`
생성자 파라미터(자기 파일 주석에 "더 이상 쓰이지 않음"이라고 이미 적혀있던 것)
삭제.

- **검증**: 위 모든 파일 `py_compile` 통과 + 실제 가상환경(`LFP_SOH_ESTIMATION`
  conda env)에서 `importlib`로 `models.scr_model`/`models.cap_heads`/
  `datasets.segment_dataset`/`evaluation.scr_evaluator`/`utils.*` 전체 모듈
  재현 로드 성공. `SCRModel`을 실제로 생성해 forward(train/eval 둘 다)·
  backward·`get_selected_scen_his()`/`get_selected_probe_his()`까지 직접
  실행해 `_gate_group_map` 관련 `AttributeError`가 없는 것을 실측으로 확인(수정
  전 재현 시도에서 실제로 해당 에러가 났던 것도 확인 후 수정). 그 뒤 원본과
  동일 조건으로 다섯 번째 전체 재학습(`p1v4_scen_lag1zone_redunfix_verify5_seed0`)
  실행 — 조기종료 416에폭·선택 epoch 355·gate_saturation=0.0041 전부 이전
  검증본들과 동일, `p1v2_summary.json` 전 필드 완전 일치(run tag/output_dir
  문자열 자체 차이 제외 불일치 0개), **체크포인트 MD5 완전 일치**
  (`32c2cd84216510655ce3c9f15072ee70`) — 이번까지 다섯 번째 검증본 전부 원본과
  바이트 단위로 동일. Bucket A 15개 항목 전량 삭제 + 2개 연쇄 발견(gate_io.py,
  train.py/test.py/parameters.py의 no-op `with_raw_cnn`/`with_raw_flat`
  오버라이드) 모두 학습 결과에 실제로 아무 영향이 없었음을 확인.

### 2026-09-27 — yaml 설정 프리셋(`--model-config`) 전면 폐기 + `parameters.py` 잔여 중복 정리

사용자 질문("parameters.py 중복 다 제거했나? yaml을 파라미터로 받는 것도 이제
안 해도 되지 않아?")을 계기로 재감사.

**`parameters.py` 잔여 중복/죽은 값 4건 발견 및 정리**(전부 grep으로 "실제로 어디서도
안 읽음"을 먼저 확인 후 삭제):
- `P1_MODEL_CONFIG["data"]["split_seed"] = 42` — 하드코딩 리터럴. `train.py`/
  `synergy.py`/`kernel.py` 전부 cfg 생성 직후 `args.split_seed`로 무조건 덮어써서
  이 초기값은 애초에 한 번도 실제로 쓰인 적이 없었다(시드 통합 라운드에서
  `FIXED_DEFAULT_SEED`를 만들 때 이 중첩 딕셔너리 안쪽 값은 놓쳤던 것) — 키 자체를
  삭제(덮어쓰기 로직이 `cfg.setdefault`/직접 대입이라 키가 없어도 안전).
- `P1_MODEL_CONFIG["data"]["is_real_input"/"output_dir"/"gates_from"]` — `model_lib/`
  어디서도 읽지 않는 완전히 죽은 키. 삭제.
- `P1_MODEL_CONFIG["loss"]["leak_cols"]` — `train.py`/`scr_loss.py` 어디서도 안 읽음.
  실제 leakage 제외는 `SOH_EXCLUDE_STAT_LEAK` 환경변수(`ACTIVE_N_HI` 토글,
  `hi_schema.py`) 하나로만 이뤄지는데, 이 키는 그 이전 방식의 흔적이었다. 삭제.
- `ACTIVE_REGRESSION_MODEL`/`FIXED_PHASE1_MODEL_CONFIG` — 아래 yaml 폐기와 함께 삭제
  (사용처가 아래에서 같이 없어짐).

**yaml 설정 프리셋(`model_lib/config/*.yaml`, `--model-config` CLI 플래그) 전면 폐기**.
재감사 결과, "다른 프리셋 재현용 이스케이프 해치"라는 기존 설명이 더 이상 사실이
아니었다:
- `main_qfref_S_all4/mit_only/hust_only/calce_tju/tju_only/p60/transformerL.yaml`
  전부 파일 내부에 `5_model/config/...`, `phase1_trainer_v2.py`, `run_pipeline.py
  Step 6`, `SCRTrainer.fit()` 같은 이번 세션 리팩토링(폴더 재편 · `train.py` 개명 ·
  `SCRTrainer` 삭제) **이전** 용어가 그대로 남아 있었다 — 리팩토링 이후 한 번도
  재검증되지 않은 프리셋들이었다는 뜻.
- 결정적으로 `run_pipeline.py`가 `--regression-model`을 받아 `train.py` 호출에
  그대로 전달하고 있었는데, `train.py` 자신의 `--regression-model` 플래그는 이미
  예전 라운드에 삭제돼 있었다(`regression_model`은 이제 항상 cfg 값 그대로 사용,
  주석: "다른 아키텍처는 sanity-check용 오버라이드라 제거함") — 즉
  `main_qfref_S_transformerL.yaml`을 CLI로 재현하는 경로는 **이미 조용히 깨져
  있었다**(`run_pipeline.py 8 --regression-model transformer`를 실행하면 `train.py`가
  "unrecognized arguments"로 즉시 죽는 상태). 아무도 최근에 이 경로를 실제로
  실행한 적이 없다는 직접 증거.
- 사용자 확인: 이 ablation 프리셋들(데이터셋 subset 조합, 회귀 헤드 교체)을 논문에
  다시 쓸 계획 없음 → 전량 폐기 승인.

**삭제/변경 내역**:
- `model_lib/config/` 디렉터리 전체 삭제(34개 파일 — `*.yaml`/`*.txt`라 애초에
  `.gitignore`로 추적 대상이 아니었음, `git rm` 불필요).
- `model_lib/utils/io_utils.py`: `_deep_merge`/`_substitute_data_root`(yaml
  프리셋 병합 + `${DATA_4_HI_ROOT}` 토큰 치환 전용) 삭제, `load_config()`를
  "그 run 자신이 저장한 `config.yaml`을 그대로 읽기"만 하는 1줄짜리로 단순화
  (`save_config()`로 남기는 run 기록용 — `9_eval/test.py`가 유일한 소비자 — 은
  전혀 다른 용도라 그대로 유지).
- `8_train/train.py`: `--model-config` 플래그 삭제, `cfg = load_config(...) if
  ... else deepcopy(P.P1_MODEL_CONFIG)` 분기를 무조건 `deepcopy(P.P1_MODEL_CONFIG)`로
  단순화. yaml 유무에 따라 갈리던 `--seg-axis`/`--axis-config`/`--data-dir`/
  `--seg-data-dir` 폴백 체인(`scenario_cfg.get(...)`)도 CLI-vs-DEFAULT_* 2단
  우선순위로 단순화 — yaml 경로가 사라져 cfg 자신의 값이 항상 DEFAULT_*와 이미
  동일했으므로 동작 변화 없음. 이제 항상 통과하는 `data_dir`/`seg_data_dir` 빈 값
  체크(RuntimeError)도 같이 삭제.
- `5_interaction/interaction.py`: `--model-config`가 `required=True`였는데
  **코드 어디서도 `args.model_config`를 쓰지 않는 완전한 죽은 인자**였음을 확인 —
  삭제.
- `6_synergy/synergy.py`, `7_kernel/kernel.py`: `--model-config`(각각 `required=True`)
  삭제, `cfg = load_config(args.model_config)` → `cfg =
  copy.deepcopy(P.P1_MODEL_CONFIG)`로 교체(train.py와 동일 패턴). `interaction.py`가
  `synergy.py`의 `_load_all_scenarios()`를 그대로 재사용하므로 이 함수 하나만
  고치면 두 스크립트 모두 해결됐다.
- `run_pipeline.py`: `--regression-model` 플래그(이미 깨져 있던 죽은 경로) 삭제,
  Step 8 호출부의 pass-through 삭제. Step 5~7 전용 `--model-config` 주입 로직
  (`_config_flag`/`_step_model_config`, `P.FIXED_PHASE1_MODEL_CONFIG` 참조)
  삭제 — `run_step()` 함수 시그니처에서 이제 아무도 안 쓰는 `model_config`/
  `config_flag` 파라미터도 제거.
- **부수 발견**: 위 변경들을 `--help` 출력으로 검증하다가 `train.py`의
  `--output-dir` 도움말 문자열에 이스케이프 안 된 리터럴 `%`(`"100% 동일하게..."`)가
  있어 `argparse`가 도움말을 조합할 때마다 `ValueError: unsupported format
  character`로 죽는 잠재 버그를 발견(이번 변경과 무관한 기존 버그, 우연히
  검증 과정에서 발견) — `%%`로 이스케이프해 같이 수정.
- **검증**: 변경된 전 파일 `py_compile` 통과, `--help` 전부 정상 출력(플래그
  삭제 확인). `5_interaction/interaction.py`를 실제 canonical 데이터로 재실행해
  기존 검증 레퍼런스 run(`0921_1953_..._seed0`)의
  `hi_scenario_interaction_*.json`과 `tag` 필드를 제외한 전체 내용을
  완전히 동일하게 재현함을 확인(`synergy.py`의 `_load_all_scenarios()`를 그대로
  재사용하므로 이 결과가 `synergy.py` 쪽 데이터 로딩 경로도 함께 검증한다).
  `7_kernel/kernel.py`도 같은 데이터·같은 시너지 그룹으로 실행해 시나리오별 커널
  HI 폭(`[16,15,16,16,15,15]`)과 평균 train R²(0.3887) 전부 레퍼런스와 정확히
  일치함을 확인(끝에 붙는 진단 print 한 줄이 기존부터 있던 cp949 em-dash
  인코딩 문제로 죽지만, 두 산출물 파일은 그 전에 이미 정상 저장 완료 — 이번
  변경과 무관한 기존 환경 이슈라 이번에도 손대지 않음). 마지막으로 원본과 동일
  조건으로 여섯 번째 전체 재학습(`p1v4_scen_lag1zone_redunfix_verify6_seed0`) 실행 —
  조기종료 416에폭·선택 epoch 355·gate_saturation=0.0041 전부 이전 검증본들과
  동일, `p1v2_summary.json` 전 필드 완전 일치(run tag 문자열 제외), **체크포인트
  MD5 완전 일치**(`32c2cd84216510655ce3c9f15072ee70`) — 이번까지 여섯 번째
  검증본 전부 원본과 바이트 단위로 동일. yaml 설정 프리셋 전면 폐기가 학습
  결과에 실제로 아무 영향이 없었음을 확인.

### 2026-09-28 — `pyproject.toml` 도입: 진입점 스크립트의 sys.path 부트스트랩 제거

사용자 질문("`hi_correlation.py`의 `PROJECT_ROOT`/`sys.path.insert`, `data_directories`에
있는데 그냥 가져다 쓰면 안 되나?")을 계기로 확인 — `data_directories.py`가 저장소
루트에 있는 이상, 그걸 import하려면 저장소 루트가 먼저 `sys.path`에 들어가 있어야
하는 "닭과 달걀" 문제라 그 자체로는 못 없앤다고 답했으나, 이어진 "이 구조가
어색한데 방법이 없냐"는 재질문에 근본 해결책(패키지화)을 검토해 적용.

**해결책**: 저장소를 `pip install -e .`로 설치 가능한 패키지로 만들었다
(`pyproject.toml` 신설, setuptools 백엔드). 이후 `data_directories`/`parameters`/
`log_utils`(모듈)과 `common`/`models`/`datasets`/`training`/`evaluation`/`utils`
(패키지, 뒤 5개는 실제로는 `model_lib/` 밑에 있지만 그 내부 코드가 이미
`from models.xxx import` 같은 절대경로 스타일로 서로를 import하고 있어 그 관례를
그대로 유지하도록 `package-dir`만 재매핑) 전부 **실행 위치(cwd)나 스크립트 파일
위치와 무관하게** 어디서든 바로 import된다.

- `model_lib/log_utils.py` → 저장소 루트 `log_utils.py`로 이동(다른 top-level
  모듈들과 같은 위치에 있어야 동일한 방식으로 잡힘). 내부 `PROJECT_ROOT` 계산도
  이동한 깊이에 맞게 `.parent`로 수정.
- 파이프라인 진입점 스크립트 전부(`1_convert/`~`9_eval/`, `model_lib/tools/
  visualize_results.py`, `model_lib/datasets/segment_dataset.py`,
  `model_lib/utils/hi_schema.py`, `docs/figures/_style.py`/`fig1`/`fig2` 등
  22개 파일)에서 "저장소 루트를 계산해 sys.path에 넣는" 부트스트랩 코드를
  제거. `hi_correlation.py`에 중복으로 남아있던 완전히 동일한
  `sys.path.insert` 블록 2개도 이 김에 정리.
- **건드리지 않은 것**: 같은 폴더가 아닌 다른 스크립트 폴더의 형제 스크립트를
  가져다 쓰는 패턴(예: `interaction.py`가 `6_synergy/synergy.py`를,
  `test.py`가 `8_train/train.py`를, `profile_hi_timing.py`가 부모 폴더의
  `hi_correlation.py`를 import하는 것)은 그대로 유지 — 이 폴더들은 의도적으로
  패키지가 아니라 "직접 실행하는 스크립트" 취급이라 패키지화 대상이 아니고,
  이런 사이트 지정 `sys.path.insert`는 여전히 필요하다.
- **검증**: 22개 파일 전부 `py_compile` 통과. `pip install -e . --no-deps`로
  실제 설치 후, 저장소와 전혀 무관한 디렉터리(`/tmp`)에서 `data_directories`/
  `parameters`/`common.scenario`/`models.hard_concrete`/`datasets.segment_dataset`/
  `training.scr_loss`/`evaluation.scr_evaluator`/`utils.hi_schema` import가 전부
  성공하는 것을 직접 실행해 확인. 8개 핵심 스크립트(`convert_unified`/
  `preprocess`/`check_integrity`/`interaction`/`synergy`/`kernel`/`train`/`test`)
  전부 `/tmp`를 cwd로 두고 `--help`를 실행해도 정상 종료함을 확인(전부 exit 0)
  — 더는 어디서 실행하든 저장소 루트 sys.path 하드코딩에 의존하지 않는다는 뜻.
  일곱 번째 전체 재학습(`p1v4_scen_lag1zone_redunfix_verify7_seed0`)도 원본과
  동일 조건으로 시작해 12에폭까지 정상 진행 중이었으나(설정 로그 완전 일치),
  사용자 요청으로 완료 전에 중단 — import 경로만 바뀌고 `segment_dataset.py`/
  `hi_schema.py`의 실제 계산 로직은 한 줄도 안 건드렸으므로 위험은 낮다고 판단,
  MD5 재검증은 다음 기회로 미룸(아래 표에 미완료로 표기).
- `.gitignore`에 `*.egg-info/`/`build/` 추가(editable install이 만드는 빌드
  메타데이터, 커밋 대상 아님).

### 2026-09-28 — `hi_correlation.py`의 STAT/DIFF/LFP/MORPH_KEYS·THETA_FLAT 중복 제거

사용자 질문("117~300행은 parameters.py로 보내도 되지 않아?")을 계기로 확인 —
`parameters.py`(실행 파라미터 전용)로 보낼 내용은 아니었지만, 대신 진짜 중복
2건을 발견했다: 위 editable install 덕분에 이제서야 발견/해결 가능해진 것들이다.

- **`STAT_KEYS`/`DIFF_KEYS`/`LFP_KEYS`/`MORPH_KEYS`**: `model_lib/utils/hi_schema.py`
  ("66-HI schema" 단일 소스 자처)에 있는 것과 66개 키 전부 완전히 동일한 로컬
  복사본이 `hi_correlation.py`에도 있었다 — 예전엔 Step4가 `model_lib`를 import할
  경로가 없어서 어쩔 수 없이 복사했던 것으로 추정. `from utils.hi_schema import
  STAT_KEYS, DIFF_KEYS, LFP_KEYS, MORPH_KEYS`로 교체(로컬 정의 4개 삭제, 옆의
  `_STAT_LABELS`/`_DIFF_LABELS`/`_LFP_LABELS`/`_MORPH_LABELS`는 hi_schema.py에
  없는 플롯 전용 레이블이라 그대로 둠). `hi_segment_viz.py`/`tools/
  profile_hi_timing.py`/`tools/seg_corr_analysis.py`가 `from hi_correlation
  import STAT_KEYS, ...`로 재수출을 받아 쓰고 있어 이 셋의 import 문은 무수정으로
  계속 동작(hi_correlation 모듈 자신이 이제 import로 이 이름들을 갖고 있으므로).
- **`THETA_FLAT = 0.25`**: `hi_correlation.py`는 정의만 해두고 실제로는 어디서도
  안 읽고 있었다(정작 쓰는 곳은 `hi_compute.py`/`common/scenario/_curves.py`).
  4곳에 흩어진 중복 중 하나였고(나머지 `_curves.py`/`tools/plateau_soc_stats.py`는
  이번 정리 범위 밖 — `tools/profile_hi_timing.py`는 값 자체가 0.05로 이미
  어긋나 있어 더더욱 손대지 않음, 별도 확인 필요) — 안 쓰는 이 정의만 삭제,
  외부 재수출 소비자 없음을 grep으로 확인 후 진행.
- **`DIS_SEGS`/`CHG_SEGS`의 SoC 경계값·레이블은 그대로 유지**: 얼핏 죽은 것처럼
  보였지만(`hi_correlation.py` 자신은 `for _, _, _seg, _ in ALL_SEGS`로 앞 2개
  버림) `tools/profile_hi_timing.py`가 `for q_lo_f, q_hi_f, seg, _ in DIS_SEGS`로
  경계값 자체를 실제 사용 중임을 확인 — 건드리지 않음(사용자 승인 범위 밖이기도
  했음).
- **검증**: `py_compile` 통과. 실제 import로 `hi_correlation.STAT_KEYS is
  utils.hi_schema.STAT_KEYS`(object identity)까지 확인, `ALL_HI_KEYS` 개수
  15+6×66 그대로, `THETA_FLAT` 속성 완전히 사라짐, 외부 3개 소비자의 `from
  hi_correlation import STAT_KEYS, ...` 재수출 import도 무수정으로 정상 동작
  확인. 순수 상수 중복 제거(값 변경 없음, object identity만 바뀜)라 재학습
  검증은 불필요하다고 판단.

### 2026-09-28 — 완전 사이클 Global HI(G01~G15) 계산 제거

사용자 요청("완전 사이클 HI는 구하지 않도록 수정하고 싶어"). `_extract_one_cell`이
사이클마다 세그먼트와 무관하게 한 번씩 계산하던 15개 "Global HI"(q_dis,
energy_dis, ica_peak1_*, dva_valley_*, r_trans_est, ce, cv_q_frac/time_frac,
chg_ica_peak1_h 등, `_build_flat_correlation_df`가 "학습엔 안 쓰임, 진단/시각화
전용"이라고 이미 명시하고 있던 값들)의 계산 자체를 없앴다.

- **`_extract_one_cell` 안 계산 로직 삭제**: 방전측(G01-03/05-10, `_global_ica`/
  `_global_dva` 호출) + 충전측(G04/11-14, `_r_dc_from_chg`/`_global_ica` 호출)
  전부 삭제. `grow` 딕셔너리는 이제 `{dataset, cell_id, cycle, capacity_Ah}`
  4개 필드만 담는다.
- **⚠️ "cycle" pkl 파일 자체는 안 없앴다** — 처음엔 완전 삭제를 검토했으나,
  `model_lib/datasets/segment_dataset.py`의 `load_dataset_native_seg()`가 이
  파일에서 `cycle`→`capacity_Ah` 매핑을 실제로 읽어, native seg pkl의
  `capacity_Ah`(세그먼트 부분 충방전량이라 SOH 타깃이 될 수 없음)를 사이클
  전체 총 용량으로 교체하는 데 쓰고 있음을 확인했다 — 이 파일을 없애면 학습
  타깃 자체가 깨진다. 그래서 cycle pkl은 유지하되 내용만 4개 필수 컬럼으로
  축소했다.
- **`GLOBAL_HI_KEYS`/`_GLOBAL_LABELS` 상수, `HI_GROUPS["Global"]`, `_HI_META`/
  `_build_hi_groups()`의 Global 루프 삭제**. `plot_correlation()`의 그림
  레이아웃(GridSpec)에서 Global 히트맵 행 자체를 제거하고 세그먼트 행 인덱스를
  한 칸씩 당겼다(`n_segs+2`→`n_segs+1`행, 마지막 산점도 행 인덱스도 같이 보정).
- **`hi_segment_viz.py`의 `plot_hi_trend()`(Global HI 15종 열화 추이 플롯) 삭제**
  — 데이터 소스 자체가 없어져 항상 빈 그림만 그리게 될 함수라 두 호출부(자체
  `main()`, `hi_correlation.py`가 `exec_module`로 동적 호출하는 지점) 모두
  같이 정리.
- **건드리지 않은 것**: `_global_ica`/`_global_dva`/`_r_dc_from_chg` 함수 정의는
  삭제하지 않았다 — `4_hi_analysis/tools/profile_hi_timing.py`가 이 셋을
  `from hi_correlation import ...`로 가져다 자체 타이밍 프로파일링에 쓰고
  있어(hi_correlation.py 안에서는 이제 안 쓰이지만) 그대로 유지.
- **검증**: `py_compile` 통과. 실제 MIT `b1c0` 셀을 `_extract_one_cell()`
  직접 호출로 재추출해 기존 캐시와 비교 — `cycle_rows`(1,820행)는 의도한 대로
  `dataset/cell_id/cycle/capacity_Ah` 4개 컬럼만 남음, `seg_rows`는 21,828건
  (기존 캐시와 행 수 완전 일치, `raw_v`/`raw_i`/`raw_t` 이미 제거된 이전
  라운드 반영해 컬럼 수만 79→76) + 샘플 `stat_*` 5개 컬럼 값 전부 완전 일치 —
  세그먼트 HI 계산 로직은 이번 변경으로 전혀 안 건드렸다는 것을 실측으로 확인.

### 2026-09-29 — `profile_hi_timing.py`의 Global HI 타이밍 + `_global_ica`/`_global_dva`/`_r_dc_from_chg` 제거

사용자 질문("profile_hi_timing.py에도 Global HI 로직이 남아있는 것 같은데, 제거하면
이 3개 함수도 이제 안 쓰게 되는 거 아니야?") — 맞았다. 위 2026-09-28 라운드에서
`_extract_one_cell`의 Global HI 계산은 지웠지만, `_global_ica`/`_global_dva`/
`_r_dc_from_chg` 함수 정의 자체는 `profile_hi_timing.py`가 자기 타이밍 프로파일링에
가져다 쓰고 있어서 남겨뒀었다. 이번에 그 소비처까지 마저 정리.

- **`profile_hi_timing.py`**: `_time_global()`(Global HI 블록별 소요시간 측정
  함수) 전체 삭제 + 호출부("Global HI 타이밍" 레코드 추가 블록) 삭제. 단
  충전 구간(`chg`/`vc`/`ic`/`dtc`) 계산 자체는 유지 — 이후 `CHG_SEGS` 세그먼트
  타이밍 루프가 그대로 쓰기 때문(완전 사이클 HI가 아니라 세그먼트 추출
  전제조건). `CAT_COLORS`의 `"Global"` 항목, `from hi_correlation import (...)`
  의 `_global_dva`/`_global_ica`/`_r_dc_from_chg` 삭제. 호출되는 곳이 아예
  없던 죽은 함수 `_cat()`(concept→category 분류, `"Global"` 폴백이 존재
  이유였음)도 같이 삭제.
- **`hi_correlation.py`**: 이제 `profile_hi_timing.py`가 안 가져다 쓰므로
  `_global_ica`/`_global_dva`/`_r_dc_from_chg` 정의 자체를 삭제(전수 grep으로
  이 3개를 참조하는 곳이 정의 외엔 전혀 없음을 확인). 그 결과 이 파일에서만
  쓰던 `from scipy.signal import savgol_filter`도 같이 삭제. `_peak_fwhm_asym`
  (hi_compute.py에서 import)은 이 파일 자신은 이제 안 쓰지만
  `profile_hi_timing.py`가 `from hi_correlation import _peak_fwhm_asym`로
  재수출받아 쓰고 있어 그대로 유지 — 이유를 주석으로 남김.
- **검증**: `py_compile` 전체 통과. 실제 import로 `hi_correlation`/
  `profile_hi_timing`에 삭제 대상 이름이 전부 없어졌음을 `hasattr()`로 확인,
  `CAT_COLORS`에 `"Global"` 키 없음 확인. MIT `b1c0` 셀 재추출로 `seg_rows`
  21,828건/`cycle_rows` 1,820건 그대로임을 재확인(이번 변경은 죽은 함수
  제거뿐이라 추출 결과에 영향 없어야 정상 — 실측으로 확인).

### 2026-09-29 — `_extract_one_cell(args)`의 죽은 인자 분기 + 불필요한 JSON 왕복 제거

사용자 질문 2건: (1) "파라미터는 parameters.py에서만 수집하기로 하지 않았냐 —
`_extract_one_cell`의 `else` 분기가 `"qfrac"`/`"{}"`를 하드코딩하고 있다", (2) "왜
`_axis_cfg`를 JSON으로 인코딩해서 넘기냐".

- **(1) 확인 결과**: `_extract_one_cell`을 실제로 부르는 곳은 `load_all()` 안
  2곳(직렬 루프, `ProcessPoolExecutor.submit`)뿐이고 둘 다 항상 4-tuple 또는
  5-tuple만 넘긴다 — 3-tuple/비-tuple 분기(`"qfrac"`/`"{}"` 하드코딩 포함)는
  repo 전체 grep으로 호출자가 전무함을 확인. 게다가 `"qfrac"`은 이미
  `common/scenario/__init__.py`의 `REGISTRY`에서 삭제된 축이라(현재
  `q_frac_ref`만 등록) 저 분기가 실행됐다면 `ValueError: Unknown scenario
  axis 'qfrac'`로 즉시 죽었을 것 — "기본값"이 아니라 죽어서 검증조차 안 된
  방어 코드였다. 두 분기 삭제, 실제 쓰이는 4/5-tuple 처리만 남김.
- **(2) 확인 결과**: `load_all()`이 `axis_cfg`(dict)를 `json.dumps()`로 문자열화해
  `ProcessPoolExecutor` 워커에 넘기고, 각 워커가 `_extract_one_cell` 안에서
  다시 `json.loads()`로 되돌리고 있었다 — 같은 튜플에 `_progress_q`
  (`multiprocessing.Manager().Queue()` 프록시, dict보다 훨씬 복잡한 객체)는
  아무 변환 없이 그대로 넘어가는 걸로 보아 `ProcessPoolExecutor`가 pickle로
  뭐든 그대로 직렬화한다는 게 이미 증명돼 있었다 — `axis_cfg`만 JSON으로
  왕복시킬 기술적 이유가 없었다(레거시 흔적으로 추정). `load_all()`의 두
  호출부와 `_extract_one_cell` 양쪽에서 `json.dumps`/`json.loads` 제거,
  dict를 그대로 전달하도록 변경.
- **검증**: `py_compile` 통과. MIT `b1c0` 셀로 새 4-tuple/5-tuple 시그니처
  둘 다 직접 호출해 재추출 — `seg_rows` 21,828건/`cycle_rows` 1,820건 그대로
  (기존 검증과 완전 동일), 두 경로(직렬/병렬 시그니처) 모두 정상 동작 확인.

### 2026-09-29 — `run_pipeline.py`의 CLI 파라미터 전량 제거(스텝 선택만 남김)

사용자 지적: "`run_pipeline.py`가 `parser.add_argument`로 실험 파라미터를 CLI
옵션으로 받는 건, 어차피 이유가 커맨드라인 오버라이드를 위해서였는데, 이제
`run_pipeline.py`의 커맨드라인은 스텝 선택(`from_step`/`--to-step`)만
유일하게 관리하도록 하자" — `parameters.py`를 "단일 소스"로 삼기로 한
원칙(§ 진행 이력 다수)을 오케스트레이터 자신에게도 적용한 것.

- **문제**: `run_pipeline.py`는 `--workers`/`--force-extract`/
  `--kernel-features-pkl`/`--interaction-json`/`--combined-redundancy-json`/
  `--max-group-size`/`--synergy-redundancy-threshold`/`--max-epochs`/
  `--patience`/`--batch-size`/`--hi-cost-weighted-l0`/`--n-hi`/`--p1-tag`/
  `--rep-cells`/`--axis-config`/`--data-dir`/`--seg-data-dir`/
  `--lambda-l0-override`/`--seed`/`--split-seed` 19개 CLI 플래그를 갖고
  있었는데, 전부 기본값이 `parameters.py`의 `ACTIVE_*` 값이라 이미 "`parameters.py`가
  단일 소스"였다 — CLI 오버라이드 경로는 `parameters.py`를 우회해 값을 바꿀
  수 있는 **또 하나의 숨은 입력 경로**였을 뿐, 실제로 쓰이지도 않았다.
  각 하위 스텝 스크립트(`train.py`, `interaction.py`, `synergy.py`,
  `kernel.py`, `test.py` 등) 자신도 이미 동일하게 `parameters.py` 기본값을
  갖는 자기 CLI를 유지하므로, 한 번만 다르게 돌려보고 싶으면 그 스크립트를
  직접 호출하면 되고 오케스트레이터 레이어에 중복 CLI가 있을 이유가 없었다.
- **수정**: `main()`의 `argparse.ArgumentParser`에서 `from_step`(위치 인자)과
  `--to-step`만 남기고 나머지 19개 `add_argument` 호출을 전부 삭제, 그
  자리를 `parameters.py`에서 직접 읽는 지역 변수 19개(`workers`,
  `force_extract`, `kernel_features_pkl`, `interaction_json`,
  `combined_redundancy_json`, `max_group_size`,
  `synergy_redundancy_threshold`, `p1_max_epochs`, `p1_patience`,
  `p1_batch_size`, `hi_cost_weighted_l0`, `n_hi`, `p1_tag`, `rep_cells`,
  `axis_config`, `p1_data_dir`, `p1_seg_data_dir`, `lambda_l0_override`,
  `seed`, `split_seed`)로 대체. `main()` 본문 전체(스텝별 인자 조립 루프,
  미리보기 `print` 블록)에서 `args.X` 참조를 전부 대응하는 지역 변수로
  치환(`args.from_step`/`args.to_step`만 예외로 유지 — 유일하게 남은 실제
  CLI). 시그니처가 `args` 전체를 받던 `_resolve_interaction_path()`/
  `_resolve_kernel_paths()` 두 헬퍼도 필요한 값만 명시적으로 받도록
  변경(`(interaction_json, interaction_out, default_json)` /
  `(kernel_features_pkl, combined_redundancy_json, kernel_pkl_out,
  kernel_redundancy_out, default_pkl)`), 두 헬퍼의 호출부 4곳(미리보기 1회 +
  Step 8 블록 내 재해석 2회 세트)도 새 시그니처에 맞게 수정.
  덤으로 발견한 죽은 코드: `axis_config`가 이제 항상
  `json.dumps(P.ACTIVE_AXIS_CONFIG)`로 고정되어(CLI로 바꿀 방법이 없으므로)
  `_non_canon = axis_config != json.dumps(P.ACTIVE_AXIS_CONFIG)`가 항상
  `False`가 되는 캐논-축 불일치 경고 블록도 함께 삭제. 모듈 docstring에
  2026-09-29 항목을 추가해 변경 배경과 "실험값을 바꾸려면 이제
  `parameters.py`를 고칠 것, 1회성 오버라이드는 해당 스텝 스크립트를 직접
  실행할 것"이라는 새 사용 원칙을 명시하고, "사용:" 예시 블록에서
  삭제된 플래그를 쓰던 예시들을 제거.
- **검증**: `py_compile` 통과. `python run_pipeline.py --help` 출력이
  `[-h] [--to-step TO_STEP] [FROM_STEP]`만 노출함을 확인(19개 옵션 전부
  제거됨). 실제 서브프로세스 디스패치 동작 확인을 위해
  `python run_pipeline.py 3 --to-step 3`(Step 3=무결성 검사, 빠르고
  부작용 없음)을 실제 실행 — 4개 데이터셋(MIT/HUST/TJU/CALCE) 전부 정상
  처리, ERROR/WARN 0건, 미리보기 출력의 `axis-config` 값이
  `parameters.py: ACTIVE_AXIS_CONFIG`와 일치, 종료 코드 0으로 정상 완료.

### 2026-09-29 — `hi_correlation.py`의 CLI 파라미터 전량 제거(parameters.py 직접 참조)

`run_pipeline.py` CLI 정리(바로 위 항목) 직후 사용자 질문: "hi_correlation.py의
main()도 parameters.py를 직접 import하면 되니까 parser는 전부 사라져도 되는거
아니야?" — 처음엔 이 스크립트의 `--n1`/`--n2`/`--axis-config`/`--force` 등 단축
CLI가 "PowerShell JSON 인용 우회용" ad-hoc 실험 오버라이드(run_pipeline.py의
경우와 달리 실제로 쓰이는 기능)라 다르다고 답했으나, 후속 논의에서 사용자가
그 오버라이드 자체를 CLI가 아니라 "매 실험 전 `parameters.py`를 정규식/AST로
패치 → 실행 → 원복"하는 드라이버 스크립트로 대체하겠다는 워크플로를 확정 —
이 전제하에서는 hi_correlation.py의 CLI도 run_pipeline.py와 동일하게 전량 제거
가능해졌다(런타임 프로세스 경계상 "파일 수정 없이 메모리로만 오버라이드"는
subprocess 기반 구조에서 불가능함을 먼저 확인 — 환경변수 방식도 검토했으나
사용자가 최종적으로 파일 패치 방식을 선택).

- **수정**: `main()`의 `argparse.ArgumentParser`와 19개 `add_argument`
  (`--workers`, `--force`, `--dataset-group`, `--seg-axis`, `--axis-config`,
  `--n1`/`--n2`/`--n-samples`/`--n2-start`/`--n2-end`/`--n2-step`/`--n2-seed`/
  `--ref-lag`/`--noise-amp`/`--noise-mode`/`--noise-period`/`--min-pts`/
  `--calibration-period`/`--calibration-mode`/`--calibration-jitter`/
  `--offset-amp`/`--exclude-cv`/`--skip-shape`) 전체 삭제, `parameters.py`
  (`ACTIVE_WORKERS`, `ACTIVE_FORCE_EXTRACT`, `FIXED_DATASET_GROUP`,
  `FIXED_SEG_AXIS`, `ACTIVE_AXIS_CONFIG`, `FIXED_EXCLUDE_CV`,
  `FIXED_SKIP_SHAPE` — 전부 기존에 "구 --옵션명" 주석과 함께 이미 준비돼 있던
  상수들)에서 직접 읽는 지역 변수로 대체. 덤으로 `--axis-config` JSON
  문자열 왕복(`json.dumps`/`json.loads`)과 "단축 인자 patch" 병합 로직도
  전부 사라짐 — `ACTIVE_AXIS_CONFIG` dict를 그대로 사용. `_print_run_config()`
  시그니처를 `args` 객체 대신 `(axis, axis_cfg, workers, force, exclude_cv,
  skip_shape)`로 변경. 모듈 docstring/일부 주석(`_ds_dir`, CC→CV 검출 import,
  `_extract_one_cell` 내부)에서 삭제된 CLI 플래그를 가리키던 문구를 대응
  parameters.py 변수명으로 정리. 이제 안 쓰는 `import argparse` 제거.
- **`run_pipeline.py` 동반 수정**: `hi_correlation.py`가 CLI를 전혀 안 받게
  되어, Step 4 두 엔트리 중 `hi_correlation.py` 쪽 `use_workers`를 `True`→
  `False`로 바꾸고, `--seg-axis`/`--axis-config`/`--force` 주입 조건에서
  `script.endswith("hi_segment_viz.py")`로 분기해 `hi_correlation.py`는
  제외(같은 Step 4의 `hi_segment_viz.py`는 아직 자체 CLI를 유지하므로 그대로
  유지).
- **검증**: `py_compile` 통과. `python run_pipeline.py 4 --to-step 4`를
  15초 제한으로 실행해 디스패치되는 실제 커맨드를 확인 — `$ ...python.exe
  .../hi_correlation.py`로 인자가 전혀 없음(기존엔 `--seg-axis`/
  `--axis-config`/`--force`/`--workers`가 붙었음). 전체 HI 추출은 원래도
  수 분 걸리는 무거운 작업이라 끝까지 기다리는 대신, Step4 전용 검증
  관행대로 MIT `b1c0` 셀을 `_extract_one_cell()`로 직접 호출(axis_cfg는
  `P.ACTIVE_AXIS_CONFIG`를 dict 그대로 전달) — `seg_rows` 21,828건/
  `cycle_rows` 1,820건, 기존 검증과 완전 동일.

### 2026-09-29 — `DEFAULT_SEG_AXIS`/`DEFAULT_AXIS_CONFIG`/`DEFAULT_DATA_DIR`/`DEFAULT_SEG_DATA_DIR` 4중 재선언 제거

코드베이스 전체 중복 조사(Explore 에이전트) 결과 중 "파라미터성" 항목 — 사용자
지적대로 이건 함수/로직이 아니라 순수 값 재선언이라 parameters.py 참조로
바로 고칠 수 있는 항목이었다.

- **문제**: `5_interaction/interaction.py`/`6_synergy/synergy.py`/
  `7_kernel/kernel.py`/`8_train/train.py` 4개 파일이 각자
  `DEFAULT_SEG_AXIS = P.FIXED_SEG_AXIS`, `DEFAULT_AXIS_CONFIG =
  json.dumps(P.ACTIVE_AXIS_CONFIG)`, `DEFAULT_DATA_DIR =
  P.FIXED_CANONICAL_DATA_DIR`, `DEFAULT_SEG_DATA_DIR =
  P.FIXED_CANONICAL_SEG_DATA_DIR` 4줄을 바이트 단위로 동일하게 재선언하고
  있었다 — 이미 parameters.py가 단일 소스인 값을 스크립트마다 지역 별칭으로
  다시 감싼 것뿐이라, 값 자체의 불일치 위험은 없었지만 4곳에 똑같은 코드가
  중복돼 가독성만 떨어뜨리고 있었다.
- **수정**: 4개 파일 전부에서 이 4줄 블록을 삭제하고, 그 이름을 쓰던 자리
  (interaction.py/synergy.py/kernel.py의 argparse `default=`, train.py의
  argparse `default=`/help 텍스트 4곳 + `main()`의 None-폴백 로직 4곳)를
  전부 `P.FIXED_SEG_AXIS`/`json.dumps(P.ACTIVE_AXIS_CONFIG)`/
  `P.FIXED_CANONICAL_DATA_DIR`/`P.FIXED_CANONICAL_SEG_DATA_DIR` 직접 참조로
  치환. train.py의 `axis_cfg` 폴백은 `json.loads(DEFAULT_AXIS_CONFIG)`
  대신 `dict(P.ACTIVE_AXIS_CONFIG)`로 바꿔 불필요한 JSON 왕복도 하나 더
  제거(hi_correlation.py 정리 때와 동일한 이유).
- **검증**: `py_compile` 4개 파일 전부 통과. `--help` 4개 전부 exit 0 확인.
  train.py는 실제 `_parse_args()` 호출 + 폴백 로직을 직접 실행해
  `seg_axis == "q_frac_ref"`, `axis_cfg == P.ACTIVE_AXIS_CONFIG`,
  `data_dir`가 정식 캐논 경로로 해석됨을 확인 — 기존 동작과 동일.

**남은 항목**: 같은 조사에서 나온 나머지 중복(축 설정→데이터스펙 해석 로직,
tqdm 폴백 shim 재구현 — 이미 `model_lib/utils/tqdm_utils.py`에 있음에도
안 쓰고 있었음, `--axis-config` JSON 파싱 try/except, JSON 저장 한 줄,
퍼센트 포맷터)은 값이 아니라 함수/로직이라 parameters.py가 아닌 별도
utils 모듈로 빼야 함 — 아직 미착수. 별도로 진행한 행렬/벡터/텐서 연산
중복 조사에서는 `4_hi_analysis/hi_compute.py`/`common/scenario/_curves.py`/
`4_hi_analysis/tools/plateau_soc_stats.py` 3곳에 사실상 동일한 DVA/ICA
곡선 생성 로직(V-Q binning, Savitzky-Golay, dV/dQ)이 있는데 **최소 샘플
가드값이 다르다**(hi_compute.py는 `n<6`/`len(vs)<6`, `_curves.py`는 `n<8`/
`len(vs)<8`) — 경계 길이 세그먼트에서 두 "단일 소스"가 서로 다른 결과를
낼 수 있는 잠재 버그로, 아직 미수정. `_peak_fwhm_asym`도 피크가 배열
경계에 걸릴 때 hi_compute.py는 `(nan, nan)`을 반환하는데 `_curves.py`는
"2배 편측 추정값"을 반환해 동작이 다르다. `synergy.py`/`kernel.py`의
`_residualize`/partial-corr 헬퍼도 바이트 단위 중복(kernel.py 주석에
"중복 재구현" 명시, 의도적).

### 2026-09-29 — `hi_correlation.py`의 시각화 함수를 `4_hi_analysis/plot.py`로 분리

사용자가 IDE에서 994~1243행(히트맵/산점도/대표 셀 추이 플롯 + `_print_run_config`)을
선택하고 "hi_correlation이 아니라 plot.py로 옮겨야 하는거 아닌가"라고 물어 진행.
`_print_run_config`는 matplotlib을 전혀 안 쓰는 터미널 요약 출력이라 "plot"이라는
이름에 안 맞아 이동 대상에서 제외하고, 실제 matplotlib 렌더링 함수 3개
(`_draw_heatmap`/`plot_correlation`/`_plot_sample_hi`)만 옮겼다.

- **순환 import 회피가 핵심 설계 포인트**: `HI_GROUPS`/`HI_LABELS`/`HI_GROUP_TAG`는
  `main()`이 축에 따라 런타임에 재빌드하는 전역(`_build_hi_groups()` 호출 후
  `global` 재할당, 1275행 부근)이다. `plot.py`가 `from hi_correlation import
  HI_GROUPS`처럼 값을 직접 import하면, 그 바인딩은 plot.py가 처음 import되는
  시점(= hi_correlation.py가 `from plot import ...`를 실행하는 시점, 즉 main()의
  재빌드보다 먼저)의 스냅샷이라 재빌드 이후에도 갱신되지 않는 낡은 값을 계속 들고
  있게 된다. 이를 피하려고 `_draw_heatmap`/`plot_correlation`/`_plot_sample_hi`
  전부 이 세 값을 모듈 전역으로 import하지 않고 매 호출 시 명시적 인자로 받도록
  시그니처를 바꿨다(`plot_correlation(corr_df, df, out_path, hi_groups, hi_labels,
  hi_group_tag, n_top=4, datasets=None)`, `_plot_sample_hi(df, corr_df, out_dir,
  hi_labels, hi_group_tag, datasets=None)`). 결과적으로 `plot.py`는
  `hi_correlation.py`를 전혀 import하지 않는 단방향 의존만 남았다
  (`hi_correlation.py → plot.py`).
- **함께 옮긴 것**: `SAMPLE_CELL_IDS`/`DATASET_CMAPS`/`DATASET_COLORS`(대표 셀
  ID·데이터셋별 cmap/색상) — hi_correlation.py 안에서 이 세 상수는 옮겨진
  플롯 함수 밖에서 전혀 안 쓰였음을 grep으로 확인 후 plot.py로 완전히 이전(값
  중복 없음). `matplotlib`/`gridspec`/`pyplot` import와 Agg 백엔드 설정도
  플롯 함수 밖에서 안 쓰여서 hi_correlation.py에서 제거하고 plot.py로 이전.
  `STAT_KEYS`/`DIFF_KEYS`/`LFP_KEYS`/`MORPH_KEYS`는 재빌드 대상이 아닌 고정
  값(`utils.hi_schema`가 단일 소스)이라 plot.py가 직접 import(인자로 안 받음).
- **`main()` 호출부 수정**: `plot_correlation(corr, df, out, n_top=4,
  datasets=_ds_names)` → `plot_correlation(corr, df, out, HI_GROUPS, HI_LABELS,
  HI_GROUP_TAG, n_top=4, datasets=_ds_names)`, `_plot_sample_hi(df, corr,
  out_dir, datasets=_ds_names)` → `_plot_sample_hi(df, corr, out_dir, HI_LABELS,
  HI_GROUP_TAG, datasets=_ds_names)`.
- **검증**: `py_compile` 둘 다 통과. `import hi_correlation`/`import plot` 둘 다
  순환 없이 정상 로드되고 `plot.plot_correlation is hi_correlation.plot_correlation`
  확인(재-export가 실제로 같은 객체). MIT 셀 2개를 `_extract_one_cell()`로 직접
  추출 → `_build_flat_correlation_df` → `compute_correlations`(corr shape
  (396, 1), 66 HI × 6세그먼트와 일치) → `plot_correlation()`/`_plot_sample_hi()`를
  실제 데이터로 끝까지 실행 — `hi_correlation.png`(1.4MB)/`sample_hi_trend.png`
  (200KB) 둘 다 정상 생성 확인(빈 파일이나 크래시 없음).

### 2026-09-29 — `_extract_one_cell()` 5개 헬퍼로 분리(방전/충전 내부 중복 통합)

사용자가 "함수 단위로 동작을 나눠놓고 함수명으로 로직을 추측할 수 있어야 하는 거
아니냐"고 물어, hi_correlation.py 전체를 훑어 평가한 결과 `_extract_one_cell`
하나만 231줄짜리 단일 블록으로 남아있었다(인자 해석/scen 룩업/pkl 로드/방전 HI
추출/충전 HI 추출/배치 거리계산 5가지 관심사가 섞여 있었고, 방전·충전 HI 추출
블록은 phase만 다를 뿐 거의 같은 코드가 두 번 복사돼 있었음).

- **분리한 헬퍼 5개**(전부 `_extract_one_cell` 바로 앞에 배치, 이 함수는 이제
  이들을 순서대로 호출하는 얇은 오케스트레이터):
  - `_build_scen_lookup(spec_names)` — 세그먼트 이름→scen 코드 룩업 테이블
  - `_load_cell_pkl(path)` — pkl 로드+검증, 실패 시 `None`
  - `_extract_segment_rows(rec_iter, spec_names, dataset, cell_id, cyc, cap,
    scen_lookup, curve_buf)` — SegmentRecord→HI 행 변환. **방전과 충전 양쪽에서
    동일하게 호출**해 기존에 두 번 복사돼 있던 코드를 하나로 통합.
  - `_prepare_charge_arrays(chg_grp, cap, exclude_cv)` — 충전 배열 준비+gap
    보정+완전성 게이트, 스킵 대상이면 `None`. 원래 `if q_tc > 0.05 and not
    _chg_incomplete:` 밖에 `if q_tc >= 0.05:`가 한 번 더 있었는데(첫 조건에
    이미 포함되는 항상-참 중복 체크 — Global HI 제거 리팩토링 때 남은 흔적으로
    추정) 단일 게이트(`if q_tc <= 0.05 or q_tc < cap * 0.60: return None`)로
    정리. 의미상 동치라 동작 변화는 없음.
  - `_apply_batched_morph_distances(curve_buf)` — 루프 종료 후 배치 DTW/Fréchet,
    결과를 행(dict)에 in-place로 써넣음.
- **`_extract_one_cell`은 이제** 인자 언패킹 → segmenter/scen_lookup 준비 →
  pkl 로드 → 사이클 루프(방전 게이트 통과 시 `_extract_segment_rows` 호출,
  충전은 `_prepare_charge_arrays`가 `None`이 아니면 같은 함수 재호출) →
  `_apply_batched_morph_distances` 순서로만 읽힌다 — 각 단계가 함수명으로
  바로 식별됨.
- **검증**: `py_compile` 통과. MIT `b1c0` 셀을 4-tuple/5-tuple(병렬 경로,
  `multiprocessing.Manager().Queue()` 포함) 양쪽 시그니처로 직접 호출 —
  둘 다 `seg_rows` 21,828건/`cycle_rows` 1,820건으로 기존 검증값과 완전
  동일. 두 호출 결과의 행 dict 키 집합도 일치, 샘플 행(`seg_name='dis_hi'`,
  `cycle=2`, `capacity_Ah=1.0706892`)도 정상 값.
