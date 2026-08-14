# 260812 — k=5 시드 재현성 검증 실험 (`random` vs `q_frac_ref`, 3-seed)

## 배경

`docs/260811_RESULTS.md` §9의 k=5/15/25 비교가 전부 single-seed라, 관측된
비단조(non-monotonic) 격차(-0.0010 → +0.0174 → +0.0003)가 실제 패턴인지
학습 노이즈인지 판단할 수 없었다. 이를 확인하기 위해 k=5 지점에서
`random`/`q_frac_ref` 각각 3개 seed로 반복 실행하기로 했다.

## 발견 — `--seed`가 지금까지 사실상 무효였다 (2026-08-12)

- `train_scr.py`의 `--seed`는 `cfg["training"]["seed"]`에 값을 저장만 하고,
  `torch.manual_seed()`/`np.random.seed()`/`random.seed()`를 호출하는 코드가
  어디에도 없었다 — 모델 가중치 초기화(HardConcreteGate `log_alpha`, 회귀
  헤드 등)가 실제로는 매 실행마다 통제되지 않은 랜덤이었다.
- `data.split_seed`(train/val/test 셀 분할)는 정상 동작했지만 CLI 오버라이드가
  없어 모든 yaml이 `42`로 고정돼 있었다 — 그래서 지금까지의 모든 run은
  "동일한 데이터 분할 + 비제어 모델 초기화" 조합이었다(테스트 셀 구성이
  항상 같았던 이유).

## 조치 — 코드 수정

- **`5_model/train_scr.py`**: `main()` 시작부에 실제 시딩 로직 추가
  (`random.seed`/`np.random.seed`/`torch.manual_seed`/`torch.cuda.manual_seed_all`,
  `args.seed` 또는 yaml `training.seed` 사용). `--split-seed` CLI 인자 신규
  추가(`data.split_seed` 오버라이드).
- **`run_pipeline.py`**: `--seed`/`--split-seed`를 Step 6/8에 전달하도록 배선
  (`--scen-k`와 동일 패턴). `--skip-classifier` 플래그 신규 추가 — Step 7(분류기)을
  건너뛴다. `test_scr.py`는 분류기 체크포인트가 없으면 자동으로 oracle 모드만
  평가하도록 이미 구현돼 있어(`has_clf` 분기) 에러 없이 안전하게 스킵된다 —
  지금까지의 모든 실험에서 oracle≈hard≈soft였으므로 결과 비교엔 영향 없음.

## 실험 설계

- 고정: `scen_k=5`, **MLP_S로 통일**(`mlp_hidden_dims=[128,64]`, `lr=2e-4`,
  `weight_decay=1e-3`, `patience=100` — `full_cycle`에 적용했던 축소 설정과
  동일, 2026-08-12에 두 축 다 여기 맞춤). 공유 baseline 파일
  (`exp_qfw_mlp_basefix.yaml`/`exp_random_none.yaml`)은 과거 문서들이 참조하는
  값이라 직접 고치지 않고, 전용 파일 `exp_qfw_mlp_S.yaml`/`exp_random_S.yaml`을
  새로 만들어 적용했다. `--skip-classifier`(시간 절약, oracle만 평가)도 고정.
- 변수: `--seg-axis`(`random` vs `q_frac_ref`) × `seed`(=`split_seed`, 42/0/123) → 총 6 run.

## 예상 소요 시간 (기존 로그 실측 기반)

| | 에폭당 시간 | 1 run(Phase1+Phase2) |
|---|---|---|
| `random` | 1.9~3.8초 | ~11분 |
| `q_frac_ref` | 7~16초(원인 미상 — `random`보다 세그먼트가 1/4인데도 4~8배 느림, 미조사) | ~43분 |

6 run 총합 예상 **~2.5~3시간**.

## 실행 명령어

```powershell
$env:SOH_EXCLUDE_STAT_LEAK="1"

# ── random, k=5, seed 3개(MLP_S) ──
python run_pipeline.py 6 --model-config 5_model/config/exp_random_S.yaml --seg-axis random --axis-config '{\"n_samples\": 12, \"window\": 0.2, \"assign\": \"none\", \"seed\": 42, \"min_pts\": 10, \"axis_name\": \"random\"}' --scen-k 5 --seed 42 --split-seed 42 --skip-classifier --workers 40

python run_pipeline.py 6 --model-config 5_model/config/exp_random_S.yaml --seg-axis random --axis-config '{\"n_samples\": 12, \"window\": 0.2, \"assign\": \"none\", \"seed\": 42, \"min_pts\": 10, \"axis_name\": \"random\"}' --scen-k 5 --seed 0 --split-seed 0 --skip-classifier --workers 40

python run_pipeline.py 6 --model-config 5_model/config/exp_random_S.yaml --seg-axis random --axis-config '{\"n_samples\": 12, \"window\": 0.2, \"assign\": \"none\", \"seed\": 42, \"min_pts\": 10, \"axis_name\": \"random\"}' --scen-k 5 --seed 123 --split-seed 123 --skip-classifier --workers 40

# ── q_frac_ref, k=5, seed 3개(MLP_S) ──
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_S.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 5 --seed 42 --split-seed 42 --skip-classifier --workers 40

python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_S.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 5 --seed 0 --split-seed 0 --skip-classifier --workers 40

python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_S.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 5 --seed 123 --split-seed 123 --skip-classifier --workers 40
```

`--axis-config` 내부의 `"seed": 42`(random 축의 윈도우 위치 샘플링용 시드)는
모델/분할 시드와 무관하므로 항상 42로 고정 — 건드리면 세그먼트 정의 자체가
바뀐다.

## 다음 단계

6개 run의 R²(oracle) 평균·표준편차를 비교해, §9에서 관측한 비단조 격차가
seed에 따른 노이즈 범위 안에 있는지, 아니면 `random`/`q_frac_ref` 간 실제
차이인지 판정한다.
