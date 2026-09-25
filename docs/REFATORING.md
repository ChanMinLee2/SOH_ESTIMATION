# REFATORING.md

2026-09-22부터 진행 중인 코드베이스 리팩토링 기록. **계속 업데이트되는 살아있는 문서** —
새 작업을 마칠 때마다 "진행 이력"에 절을 추가한다. 과거 절은 그 시점 기준 사실이므로
이후 작업이 내용을 뒤집어도 수정하지 않고, 새 절에 "N월N일 항목 정정" 식으로 남긴다.

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
