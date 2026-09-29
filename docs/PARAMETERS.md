# PARAMETERS.md — 파라미터 구조 정리

**작성 기준**: 2026-09-28 시점의 실제 코드(`parameters.py`, `run_pipeline.py`,
`1_convert/`~`9_eval/`). 이전 버전 문서(2026-08-14 기준, `5_model/`·`scr.yaml`·
`train_scr.py` 등 리팩토링 이전 폴더 구조를 전제로 CLI 옵션 197개를 O/X로 감사하고
yaml `main`/`fixed` 분리를 제안하던 문서)는 그 제안이 실제로 구현됐다가(2026-08-14)
이후 아예 더 단순한 구조(yaml 완전 폐기, `parameters.py` 단일화)로 대체되어 완전히
달라졌으므로 전면 폐기하고 이 코드 상태만 근거로 다시 썼다. 옛 문서의 판단 근거·구현
로그가 필요하면 git 이력에 남아 있다.

---

## 핵심 원칙

1. **`parameters.py`가 유일한 소스** — 파이프라인의 모든 실행 파라미터(축 설정, 실행
   환경, 모델/학습 하이퍼파라미터)는 이 파일 하나에서만 관리한다. 다른 스크립트는
   이 값을 참조만 하고, 자기 자신의 하드코딩된 기본값을 갖지 않는다.
2. **`ACTIVE_*` vs `FIXED_*`** — 실제로 실험마다 바꿔가며 쓰는 값(`ACTIVE_*`)만
   `run_pipeline.py`의 CLI 플래그로 노출된다. 거의 안 바꾸는 값(`FIXED_*`)은 CLI
   플래그 자체가 없다 — 바꾸려면 `parameters.py`를 직접 수정해야 한다. 이 구분은
   "CLI 표면적을 줄인다"는 목적이지 "기능을 없앤다"는 뜻이 아니다.
3. **yaml 설정 프리셋 없음** — 과거엔 `model_lib/config/*.yaml` + `--model-config`
   플래그로 여러 프리셋(다른 회귀 헤드, 다른 데이터셋 조합 등)을 골라 쓸 수 있었으나,
   2026-09-27부로 **전면 폐기**했다(재검토 결과 그 프리셋들이 폴더 재편 이전 코드를
   전제로 한 채 한 번도 재검증되지 않았고, 그중 하나는 CLI 전달 경로 자체가 이미
   조용히 끊겨 있었음 — `docs/REFACTORING.md` 2026-09-27 항목 참고). 지금은
   `parameters.py: P1_MODEL_CONFIG` 딕셔너리 하나가 Step 5~9(상호작용/시너지/커널/
   학습)가 공유하는 모델·학습 설정의 유일한 소스다.
4. **`parameters.py`는 editable install로 import된다** — 2026-09-28부로 저장소가
   `pip install -e .`(`pyproject.toml`)로 설치되는 구조라, 각 스크립트는 실행 위치와
   무관하게 `import parameters as P`만 하면 된다. 예전처럼 `PROJECT_ROOT`를 계산해
   `sys.path`에 넣는 부트스트랩 코드는 전부 제거했다 — 새로 클론했으면 `pip install -e .`를
   먼저 실행해야 한다.
5. **폴더 = 파이프라인 스텝** — 파라미터도 이 원칙을 따른다. 1개 스텝 전용 파라미터는
   그 스텝 CLI에만, 여러 스텝이 공유하는 파라미터(축 설정, 데이터셋, 시드 등)는
   `run_pipeline.py`가 한 번만 받아 해당 스텝들에 동일하게 뿌린다.

---

## `parameters.py` 구조

### `ACTIVE_*` — `run_pipeline.py` CLI로 계속 노출됨 (18개)

| 상수 | 기본값 | 대응 CLI 플래그 | 적용 스텝 |
|---|---|---|---|
| `ACTIVE_FROM_STEP` | 1 | `from_step`(위치인자) | 오케스트레이터 |
| `ACTIVE_WORKERS` | 8 | `--workers` | 1~4 |
| `ACTIVE_FORCE_EXTRACT` | False | `--force-extract` | 4 |
| `ACTIVE_DATASET` | "all" | `--dataset`(1_convert/2_preprocess 자체 CLI) | 1~2 |
| `ACTIVE_KERNEL_FEATURES_PKL` | None(자동) | `--kernel-features-pkl` | 8~9 |
| `ACTIVE_INTERACTION_JSON` | None(자동) | `--interaction-json` | 8~9 |
| `ACTIVE_COMBINED_REDUNDANCY_JSON` | None(자동) | `--combined-redundancy-json` | 8~9 |
| `ACTIVE_MAX_GROUP_SIZE` | 4 | `--max-group-size` | 6 |
| `ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD` | 0.9 | `--synergy-redundancy-threshold` | 6~7 |
| `ACTIVE_MAX_EPOCHS` | None | `--max-epochs` | 8 |
| `ACTIVE_PATIENCE` | None | `--patience` | 8 |
| `ACTIVE_BATCH_SIZE` | None | `--batch-size` | 8 |
| `ACTIVE_HI_COST_WEIGHTED_L0` | False | `--hi-cost-weighted-l0` | 8 |
| `ACTIVE_LAMBDA_L0_OVERRIDE` | 0.000237 | `--lambda-l0-override` | 8 |
| `ACTIVE_N_HI` | 64 | `--n-hi`(63\|64\|66) | 5~9 |
| `ACTIVE_P1_TAG` | "p1v4_full" | `--p1-tag` | 8 |
| `ACTIVE_REP_CELLS` | None(자동선정) | `--rep-cells` | 9 |
| `ACTIVE_DATA_DIR` / `ACTIVE_SEG_DATA_DIR` | None | `--data-dir` / `--seg-data-dir` | 5~8 |
| `ACTIVE_AXIS_CONFIG` | (아래) | `--axis-config` | 4~8 |
| `ACTIVE_SEED` / `ACTIVE_SPLIT_SEED` | None→`FIXED_DEFAULT_SEED` | `--seed` / `--split-seed` | 5~8 |

`ACTIVE_AXIS_CONFIG`(정식 q_frac_ref 축 설정, 통째로 교체만 가능 — 부분 오버라이드 없음):
`n1=0.35, n2=0.20, n_samples=2, ref_lag=1, noise_amp=0.03, noise_mode="ou", noise_period_cycles=200, min_pts=5, calibration_period=100, offset_amp=0.005`

### `FIXED_*` — CLI 노출 없음, 값 자체를 바꾸려면 파일 직접 수정 (18개)

| 상수 | 값 | 의미 |
|---|---|---|
| `FIXED_DATASET_GROUP` | "lfp" | `hi_correlation.py --dataset-group`(Step4 전용, MIT+HUST) |
| `FIXED_CANONICAL_DATASETS` | `["MIT","HUST"]` | Step 5~9 `--datasets`/`P1_MODEL_CONFIG["data"]["datasets"]` 단일 소스 |
| `FIXED_DEFAULT_SEED` | 42 | 시드 3종(`ACTIVE_SEED`/`ACTIVE_SPLIT_SEED`/`FIXED_SHUFFLE_SEED`)의 공통 폴백값 |
| `FIXED_SEG_AXIS` | "q_frac_ref" | 세그멘테이션 축 종류(축 자체를 바꾸려면 이 값을 수정) |
| `FIXED_CANONICAL_DATA_DIR` / `_SEG_DATA_DIR` | (경로) | `ACTIVE_AXIS_CONFIG`로 Step4가 실제 추출한 데이터 위치 |
| `FIXED_INTERACTION_ALPHA` | 0.05 | Step5 BH 보정 유의수준 |
| `FIXED_INTERACTION_MIN_EFFECT_SIZE` | 0.1 | Step5 effect-size 임계값 |
| `FIXED_MIN_PARTIAL_CORR` | 0.02 | Step6 그룹 성장 중단 기준 |
| `FIXED_PREFILTER_TOP_M` | 15 | Step6 편상관 정밀계산 전 사전필터 폭 |
| `FIXED_GLOBAL_DEDUP` | False | Step6 `--global-dedup`(실측 기본 동작 고정) |
| `FIXED_SHUFFLE_SEED` | `FIXED_DEFAULT_SEED` | Step5~6 `--shuffle-from` 대조군 전용 시드 |
| `FIXED_KERNEL_ALPHA` / `_GAMMA` / `_N_COMPONENTS` | 1.0 / None / 100 | Step7 Ridge/Nystroem 하이퍼파라미터 |
| `FIXED_KERNEL_MAX_FEATURES` | None(무제한) | Step7 시나리오별 커널 피처 쿼터 |
| `FIXED_MIN_RAW_PARTIAL_CORR` | None(비활성) | Step7 raw HI 추가 필터 |
| `FIXED_COMBINED_REDUNDANCY_THRESHOLD` | 0.95 | Step7 3차(raw+kernel) 배제 임계값 |
| `FIXED_TRAIN_CYCLE_FRAC` | None(=1.0) | Step8 train split cycle 서브샘플 비율(진단용) |
| `FIXED_BETA_MIN` | None(=0.1) | Step8 temperature annealing 하한 |
| `FIXED_L0_NORM_CONSTANT` | None(=n_scenarios) | Step8 L0 정규화 상수 |
| `FIXED_VAL_RMSE_EPSILON` | 0.0005 | Step8 체크포인트 선택 tie-break 기준 |
| `FIXED_DEVICE` | None(auto) | Step8~9 연산 장치 |
| `FIXED_EXCLUDE_CV` / `FIXED_SKIP_SHAPE` | False / False | Step2/4 데이터 변형 옵션(이번 정식 레시피 미사용) |

### `P1_MODEL_CONFIG` — Step 5~9 공통 모델/학습 설정 딕셔너리

`interaction.py`/`synergy.py`/`kernel.py`/`train.py` 네 스크립트가 전부 이 딕셔너리를
`copy.deepcopy`해서 시작한다(과거엔 `--model-config` yaml을 로드했던 자리).

| 섹션 | 주요 필드 |
|---|---|
| `data` | `data_dir`/`seg_data_dir`(`FIXED_CANONICAL_*` 참조), `train_ratio=0.6`, `val_ratio=0.2`, `test_ratio=0.2`, `min_cycles_per_cell=10`, `nominal_capacities={MIT:1.1, HUST:1.2}`, `datasets`(`FIXED_CANONICAL_DATASETS` 참조) |
| `classifier` | `charge_probe_m=10`, `discharge_probe_m=10` |
| `model` | `d_probe=64`, `d_head=128`, `dropout=0.2`, `regression_model="mlp"`, `mlp_hidden_dims=[128,64]` (+ Transformer/ResNet 계열 ablation용 예비 필드) |
| `loss` | `lambda_scen=0.01`, `lambda_l0=0.01`(→ `ACTIVE_LAMBDA_L0_OVERRIDE`로 덮임), `lambda_l0_schedule="delayed_warmup"` |
| `training` | `epochs=500`, `batch_size=2048`, `lr=2e-4`, `weight_decay=1e-3` |
| `evaluation` | `metrics=[rmse,mae,r2,mape]`, `rep_cells_per_dataset=5` |
| `scenario` | `axis`/`axis_config` — `FIXED_SEG_AXIS`/`ACTIVE_AXIS_CONFIG`를 그대로 참조(중복 저장 금지) |
| `regression` | `scen_k_count=25` |

`scenario.axis`/`scenario.axis_config`와 `data.data_dir`/`data.seg_data_dir`은 이
딕셔너리 안에 값을 또 넣지 않고 위 `ACTIVE_*`/`FIXED_*` 상수를 참조만 한다 — 두 곳에
같은 정보가 중복 저장되면 다시 어긋날 수 있기 때문(실제로 있었던 문제).

---

## 스크립트별 현재 CLI 옵션

### `run_pipeline.py` (오케스트레이터, 22개)
`from_step`(위치인자), `--to-step`, `--workers`, `--force-extract`,
`--kernel-features-pkl`, `--interaction-json`, `--combined-redundancy-json`,
`--max-group-size`, `--synergy-redundancy-threshold`, `--max-epochs`, `--patience`,
`--batch-size`, `--hi-cost-weighted-l0`, `--n-hi`, `--p1-tag`, `--rep-cells`,
`--axis-config`, `--data-dir`, `--seg-data-dir`, `--lambda-l0-override`, `--seed`,
`--split-seed` — 전부 `parameters.py: ACTIVE_*`를 기본값으로 참조.

### `1_convert/convert_unified.py` (Step 1, 4개)
`--dataset`(`P.ACTIVE_DATASET`), `--output-root`, `--workers`(`P.ACTIVE_WORKERS`),
`--no-cache`.

### `2_preprocess/preprocess.py` (Step 2, 3개)
`--dataset`(`P.ACTIVE_DATASET`), `--skip-shape`(`P.FIXED_SKIP_SHAPE`),
`--workers`(`P.ACTIVE_WORKERS`). 필터4~7의 임계값(윈도우/시그마/gap 기준 등)은
2026-08-14에 이미 코드 상수로 승격되어 CLI에서 빠졌다.

### `3_integrity/check_integrity.py` (Step 3, 1개)
`--workers`(`P.ACTIVE_WORKERS`) — 가장 군더더기 없는 스크립트, 그대로 유지.

### `4_hi_analysis/hi_correlation.py` (Step 4, 23개)
공통: `--workers`, `--force`, `--dataset-group`(`P.FIXED_DATASET_GROUP`), `--seg-axis`,
`--axis-config`, `--exclude-cv`, `--skip-shape`.
축 파라미터 단축 인자(`--axis-config` JSON을 CLI 한 줄로 편하게 쓰기 위한 통로):
`--n1`/`--n2`/`--n-samples`/`--ref-lag`/`--noise-amp`/`--noise-mode`/`--noise-period`/
`--min-pts`/`--calibration-period`/`--offset-amp` (정식 레시피에서 실사용) +
`--n2-start`/`--n2-end`/`--n2-step`/`--n2-seed`/`--calibration-mode`/
`--calibration-jitter`(세그먼트 길이 랜덤화 등 **ablation 실험 전용**, 정식 레시피
미사용 — Bucket B, 삭제하지 않고 유지).

### `5_interaction/interaction.py` (Step 5, 12개)
`--seg-axis`, `--axis-config`, `--data-dir`, `--seg-data-dir`, `--datasets`,
`--split-seed`, `--alpha`(`P.FIXED_INTERACTION_ALPHA`),
`--min-effect-size`(`P.FIXED_INTERACTION_MIN_EFFECT_SIZE`), `--tag`(필수),
`--out-dir` + `--shuffle-from`/`--shuffle-seed`(무작위 대조군 전용, ablation).

### `6_synergy/synergy.py` (Step 6, 15개)
`--seg-axis`, `--axis-config`, `--data-dir`, `--seg-data-dir`, `--datasets`,
`--split-seed`, `--max-group-size`, `--redundancy-threshold`, `--min-partial-corr`,
`--prefilter-top-m`, `--global-dedup`, `--tag`(필수), `--out-dir` +
`--shuffle-from`/`--shuffle-seed`(대조군 전용, ablation).

### `7_kernel/kernel.py` (Step 7, 16개)
`--seg-axis`, `--axis-config`, `--data-dir`, `--seg-data-dir`, `--datasets`,
`--split-seed`, `--synergy-groups-json`(필수), `--alpha`, `--gamma`, `--n-components`,
`--redundancy-threshold`, `--max-features`, `--min-raw-partial-corr`,
`--combined-redundancy-threshold`, `--tag`(필수), `--out-dir`.

### `8_train/train.py` (Step 8, 25개)
`--seg-axis`, `--axis-config`, `--charge-m`, `--discharge-m`, `--scen-k`, `--seed`,
`--split-seed`, `--train-cycle-frac`, `--data-dir`, `--seg-data-dir`, `--beta-min`,
`--device`, `--max-epochs`, `--patience`, `--batch-size`, `--kernel-features-pkl`,
`--combined-redundancy-json`, `--interaction-json`, `--tag`, `--output-dir`,
`--lambda-l0-override`, `--l0-warmup-epochs-override`, `--l0-norm-constant`,
`--hi-cost-weighted-l0`, `--val-rmse-epsilon`. **2026-09-27부로 `--model-config`/
`--regression-model` 삭제** — 전자는 yaml 폐기, 후자는 이미 오래전부터 실제
전달 경로가 끊겨 있던 죽은 플래그였음(`docs/REFACTORING.md` 참고). 대안 회귀
헤드(Transformer 등)는 이제 `P1_MODEL_CONFIG["model"]["regression_model"]`을
코드에서 직접 바꿔야 활성화된다.

### `9_eval/test.py` (Step 9, 9개)
`--run-dir`(필수), `--checkpoint`, `--interaction-json`, `--kernel-features-pkl`,
`--combined-redundancy-json`, `--rep-cells`, `--data-dir`, `--seg-data-dir`,
`--device`. `--run-dir`가 가리키는 폴더의 `config.yaml`(그 run 자신이 저장한
스냅샷)을 읽어 학습 당시 설정을 복원한다 — 이건 프리셋 선택이 아니라 기록
읽기라 yaml 폐기와 무관하게 유지된다.

---

## 연혁 요약

- **2026-08-14**: 1차 CLI 전수 감사(당시 20개 스크립트, 197개 옵션 중 85개를
  "제거 후보"로 표시) + `scr.yaml`을 `fixed.yaml`/`main_*.yaml`로 분리하는 구조 제안.
  이후 실제로 8개 핵심 스크립트에서 53개 옵션을 1차 제거(임계값 상수화, legacy
  플래그 삭제 등).
- **2026-09-23**: `parameters.py` 단일화 — 위 `ACTIVE_*`/`FIXED_*` 체계 확립,
  축 설정을 `--n1`/`--n2`같은 개별 단축 플래그에서 `--axis-config` JSON 하나로 통합.
  yaml은 이 시점까진 "지정하면 읽고, 안 하면 `parameters.py` 기본값" 방식의
  이스케이프 해치로 축소돼 있었다.
- **2026-09-27**: yaml 설정 프리셋(`model_lib/config/*.yaml`, `--model-config` 플래그)
  전면 폐기 — `interaction.py`/`synergy.py`/`kernel.py`/`train.py` 전부
  `P1_MODEL_CONFIG`를 직접 참조하도록 통일. 이 문서의 옛 버전에 있던
  "config.yaml 구조 개편안" 절은 이 폐기로 완전히 대체되었다(자세한 경위는
  `docs/REFACTORING.md` 2026-09-27 항목).
