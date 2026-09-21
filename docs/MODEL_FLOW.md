# MODEL_FLOW — SCR 모델 학습과정 코드 흐름

`phase1_trainer_v2.py main()` 실행 순서를 코드 문단 단위로 정리. 각 항목: 2줄 설명 + 예시.
관련 파일: `models/scr_model.py`, `models/hard_concrete.py`, `training/scr_loss.py`.

## 0. 모델 입력(batch) 구성 — `x_hi`/`nan_mask`가 어디서 오는가

아래 §1부터는 `x_hi`(N_HI개 raw HI)가 이미 배치에 있다고 전제하고 시작한다 — 그 전
단계(`datasets/segment_dataset.py`)가 복잡해서 한 번 짚는다. `build_datasets()`가
만드는 `SegmentDataset`이 SCRModel의 `forward(batch)`가 받는 딕셔너리 전부를 만든다.

1. **raw HI 값 자체는 여기서 계산 안 한다** — Step 4/5(`4_hi_analysis/hi_correlation.py`
   + `5_model/hi_compute.py`)가 세그먼트마다 미리 계산해 pkl에 저장해 둔 값을
   `load_dataset_native_seg()`가 그대로 로드한다. 세그먼트 1개 = "그 사이클의 시나리오
   1개"이고, 컬럼 순서는 `get_hi_cols_for_seg(seg)`(카테고리 stat→diff→lfp→morph 순,
   N_HI=63/64/66 — §13 N_HI 토글 참고).
2. **컬럼 익명화**: 원본 컬럼명(`stat_v_mean_cw_chg_lo` 등, `_NATIVE_HI_COLS`)을
   `hi_00..hi_{N_HI-1}`로 rename한다 — 이후 전 파이프라인(게이트, 코스트 벡터,
   redundancy_mask 등)은 이 익명 인덱스로만 다루고, 원래 이름이 필요하면 아무
   시나리오나 골라 `get_hi_cols_for_seg(ref_seg)`로 역산한다(모든 시나리오가 같은
   순서를 공유하므로 어느 시나리오를 기준으로 삼아도 인덱스는 같다).
3. **정규화 + 진짜 결측치 처리**(`SegmentNormalizer`): train split으로만 컬럼별
   mean/std를 `fit()`하고(val/test는 fit 안 함 — 누수 방지), `transform_x()`가
   `(x-mean)/std`로 z-score한 뒤 원래 NaN이었던 자리를 정확히 0.0으로 채운다. NaN은
   "세그먼트가 너무 짧아 그 통계량을 못 낸 경우" 등 진짜 결측치다 — 결과로 `x_hi`
   (z-score 값, NaN 자리는 0.0)와 `nan_mask`(1.0=유효했음, 0.0=원래 NaN)를 같이 반환.
4. **`scen_idx`/`direction`/`level`**: Step 4/5가 축 설정(q_frac_ref/vwindow 등)에 따라
   이미 부여해 둔 `segment_id`를 그대로 가져와, `spec.scenario_to_dir_class()`로
   `direction`(+1 충전/-1 방전)과 `level`(zone 0/1/2, CE 분류 타깃)을 역산한다 — 셋 다
   **모델이 추론하는 게 아니라 데이터 생성 시점에 이미 정해진 정답 라벨**이고,
   `scen_idx`는 forward()에서 어느 게이트로 라우팅할지 그대로 쓰인다(= "oracle" 라우팅).
5. **`cap_init`**: 그 셀의 초기/정격 용량(Ah) — SOH 타깃(`capacity_Ah/cap_init`)과는
   완전히 별도로 자체 mean/std로 z-score해서(`transform_cap_init`) conditioning
   입력으로 추가한다(셀마다 용량 스케일이 다른 걸 모델에 알려주는 용도).
6. **`x_kernel`은 여기 없다** — 원본 pkl/`SegmentDataset`엔 없고, 학습 스크립트가
   `_apply_kernel_features()`(바로 아래 §1)로 나중에 별도로 덧붙인다.

`forward(batch)`가 실제로 받는 텐서 요약:

| key | shape | 내용 |
|---|---|---|
| `x_hi` | (B, N_HI) | z-score된 raw HI, 원래 NaN이던 자리는 0.0 |
| `nan_mask` | (B, N_HI) | 1.0=유효, 0.0=원래 NaN — 2026-09-21부터 다중공선성 배제는 이 값을 더 이상 건드리지 않음(§2) |
| `direction` | (B,) | +1=충전 / -1=방전 (정답, 데이터 생성 시점에 확정) |
| `scen_idx` | (B,) | 0..n_scenarios-1 (정답, "oracle" 라우팅에 그대로 쓰임) |
| `level` | (B,) | CE 분류 타깃(zone) |
| `cap_init` | (B,) | z-score된 초기/정격 용량 |
| `x_kernel` | (B, max_k) | §1에서 학습 스크립트가 나중에 붙임 — 원본 배치엔 없음 |

## 1. 커널 HI 준비 — `_apply_kernel_features()`

`--kernel-features-pkl`이 주어지면, 시나리오별로 그 시나리오가 만든 RBF 커널 HI만
[0:K_s) 구간에 채우고 나머지는 0-패딩한 `x_kernel` 텐서를 각 데이터셋에 붙인다.
각 커널 피처는 멤버 raw HI 카테고리 비용의 평균(`cost`)도 함께 실어온다.

예: `dis_hi` 시나리오가 커널 5개(K_5=5)를 만들었다면 `x_kernel[:, 0:5]`엔 실값,
`chg_lo`(K=3)면 `x_kernel[:, 3:5]`는 항상 0. `kernel_hi_counts=[..,5,..]`로 전달됨.

## 2. 결합 다중공선성 마스킹 — `_build_redundancy_mask()`

`--combined-redundancy-json`이 주어지면, raw+커널을 시나리오별로 합쳐 |r|≥0.95인 쌍 중
degree/타깃상관 기준으로 진 쪽을 게이트 레벨에서 차단한다(`redundancy_mask`,
`_apply_scen_gate`가 `scen_gates` 출력에 곱함).

예: `chg_lo`에서 `stat_v_std`가 `diff_dvdq_std`와 |r|=0.97이고 degree가 더 높으면
`redundancy_mask[chg_lo_idx, v_std_idx]=False` → 그 자리는 log_alpha와 무관하게 항상 0.

⚠️ **2026-09-21 버그 수정**: 예전엔 `_apply_combined_redundancy_raw()`가 raw HI를
입력 레벨(`nan_mask`)에서도 이중으로 0-강제했었다(지금은 삭제됨). §0에서 보듯
`x = x_hi * nan_mask`가 `probe_x`(분류기 입력)와 `scen_x`(회귀 입력) 양쪽의 공통
입력이라, 이 입력 레벨 마스킹이 "이 시나리오 회귀엔 필요 없는 HI"를 분류기 입력에서도
지워버려 분류기 정확도를 붕괴시키고 있었다(실측: 98.65%→36~67%). 게이트 레벨
`redundancy_mask` 하나만으로 `scen_x`의 정확성은 로그 알파와 무관하게 이미 완전히
보장되므로, 입력 레벨 마스킹은 제거하고 게이트 레벨만 남겼다 — `probe_x`는 이제
원본 값을 그대로 본다.

## 3. SCRModel 생성 — `SCRModel.__init__`

Stage A(방향별 probe gate) + Stage B(시나리오별 raw HI gate) + Stage B'(시나리오별
커널 HI gate) + cap_head(MLP)로 구성된 3단 게이팅 회귀 모델을 만든다.

예: `SCRModel(spec=..., kernel_hi_counts=[14,15,8,...], kernel_hi_costs={...},
redundancy_mask=mask, shrinkage_gate=False)` — 옵션 인자는 전부 `None`이면 원래 구조와 동일.

## 4. Forward Stage A — `_apply_probe_gate()`

충전/방전 방향별로 별도의 `HardConcreteGate(N_HI)`를 두어, MSE(회귀) + CE(분류, Phase1만)
두 목적함수의 그래디언트를 동시에 받는 "범용으로 쓸모있는 HI"를 고른다.

예: 방전 구간 배치는 `discharge_probe_gate`를 통과, `probe_x = x * z_probe` (B,N_HI).

## 5. Forward Stage B — `_apply_scen_gate()`

시나리오 인덱스(`scen_idx`)로 그 시나리오 전용 `HardConcreteGate`를 라우팅해 MSE
그래디언트만으로 "그 시나리오에서 쓸모있는 HI"를 고르고, 끝에서 `redundancy_mask`를 곱한다.

예: `scen_idx==dis_hi`인 행은 `scen_gates[dis_hi]`를 통과, 다중공선성으로 배제된
`v_std` 자리는 `masked[:, v_std_idx] *= 0` (log_alpha 값과 무관하게 항상 0).

## 6. Forward Stage B' — `_apply_scen_kernel_gate()`

`kernel_hi_counts`가 주어졌으면 시나리오별로 자기 폭 K_s만큼만 own 게이트를 적용하고
결과를 `max_k` 폭 텐서의 앞쪽에 써넣는다 — 다른 시나리오 커널을 위한 슬롯 자체가 없다.

예: `dis_hi`(K=15) 배치는 `x_kernel[:, :15]`만 게이트 통과, `[15:max_k)`는 항상 0
(폭이 더 넓은 다른 시나리오가 있어도 구조적으로 침범 불가).

## 7. Forward — 회귀/분류 입력이 서로 다르다

회귀(`cap_head`)는 `[probe_x, scen_x, kernel_x(선택), raw_cnn/raw_flat(선택),
scenario_onehot(선택), direction, cap_init]`을 이어붙여 SOH 예측값 하나를 낸다.
분류(`probe_mlp`, Phase1 dual-objective)는 `scen_x`/`kernel_x`/`cap_init` 없이
**`probe_x`+`direction`만** 쓴다 — zone 분류엔 "이전 용량 기준점"(`cap_init`)이나
시나리오 전용 게이트가 필요 없기 때문이다(scr_model.py forward 참고).

예: 기본 구성(N_HI=64, `SOH_EXCLUDE_STAT_LEAK=1`이 run_pipeline.py 기본)이면
회귀 `feat = [probe_x(64) | scen_x(64) | kernel_x(K) | direction(1) | cap_init(1)]`
→ `cap_head(feat) -> cap_pred (B,)`. 분류는 `probe_x_dir = [probe_x(64) | direction(1)]`
→ `probe_mlp(probe_x_dir) -> level_logits`. N_HI는 63/64/66 중 선택 가능(§13 N_HI 토글).

## 8. HardConcreteGate 메커니즘 — `hard_concrete.py`

학습 중엔 Gumbel-등가 연속 완화(`s = sigmoid((logU-log(1-U)+log_alpha)/BETA)`)로
미분 가능한 게이트를 쓰고, 추론 시엔 `sigmoid(log_alpha)` 기준 하드 0/1 마스크로 바뀐다.

예: `log_alpha`가 커질수록 `gate_prob()`(L0 페널티에 쓰이는 P(z>0))가 1에 가까워짐 —
학습이 끝나면 `active_indices()`로 "이 HI는 실제로 선택됐다" 확정.

## 9. SCRLoss — `training/scr_loss.py`

`total = MSE + lambda_scen*CE + lambda_l0*L0penalty`(2026-09-18부터 shrinkage-gate
경로 자체가 삭제돼 lambda_shrink 항 없음).
L0penalty는 probe/scen(raw) + kernel(멤버 raw 평균)을 합산 — 카테고리 비용 가중은
`hi_cost_weighted`(기본 **False**)가 켜져 있을 때만 적용되고, 꺼져 있으면(현재 기본)
전부 균일 비용 1.0이다(§13 카테고리 비용).

예(`hi_cost_weighted=True`일 때만): `morph` 카테고리 raw HI는 비용 1.0(실측 최고가),
`diff`는 0.1477(실측 최저가)이라 같은 gate_prob이어도 `morph` HI가 훨씬 더 강하게
억제됨(§13 카테고리 비용 표 참고). 커널 HI도 동일 원리로 멤버 raw HI 카테고리 평균 비용을
받는다(예: stat 2개+diff 2개 멤버 → 비용 (0.8084+0.8084+0.1477+0.1477)/4≈0.4781).
기본값(`hi_cost_weighted=False`)에서는 이 비용들이 전부 1.0으로 대체돼 카테고리와
무관하게 "활성 게이트 개수"만 페널티가 된다.

## 10. 학습 루프 — `main()`의 `for epoch in trange(epochs)`

매 epoch마다 lambda_l0를 웜업/램프 스케줄로 올리고 BETA(gate 온도)를 anneal하며,
배치마다 forward→loss→backward→clip→step 후 train/val RMSE·R²를 기록한다.

예: `l0_warmup_ep=50` 동안 `eff_l0=0`(게이트 자유 학습) → 이후 50 epoch에 걸쳐
목표 `lambda_l0`까지 선형 램프, 동시에 `BETA`가 2/3→0.1로 내려가며 게이트가 점점 이산화.

## 11. 체크포인트 선택 — val_rmse 우선, gate saturation은 fallback

L0가 완전히 램프된 이후 구간에서 val_rmse가 best보다 `--val-rmse-epsilon`(기본 0.0005)
**이상** 좋아지면 그 epoch을 "진짜 개선"으로 채택한다. 그게 아니면(개선폭이 epsilon
미만이거나 오히려 나빠져도) gate saturation(sat, 애매구간 [0.1,0.9] 게이트 비율)이
best보다 낮은 epoch을 대신 채택한다 — val_rmse 쪽에 "동률 밴드" 같은 상한은 없어서,
val_rmse가 다소 나빠져도 sat이 새로 낮아졌으면 채택된다(2026-09-18, 기존 sat-1순위
기준을 뒤집음).

예: epoch 187(val_rmse=0.015276, sat=0.0105)이 best인 상태에서 epoch 204(val_rmse=
0.015712, sat=0.0035)가 오면, rmse는 오히려 0.000436 나빠졌지만 epsilon(0.0005)만큼
개선한 것도 아니라 첫 조건은 불성립 → sat이 0.0105→0.0035로 낮아졌으니 두 번째 조건으로
204를 베스트로 채택(`p1v4_noscen_gatefix_l0fix_seed42` 실측 로그 재현으로 검증).

## 12. 산출물 저장 — `train_scr.py`의 `_save_scen_masks_to_json` / `_plot_gate_probs`

베스트 체크포인트 기준으로 게이트별 `gate_prob()`을 읽어 `regression_HIs.json`
(raw)/`regression_kernel_HIs.json`(kernel)과 `hi_selection_matrix.png`를 만든다.

예: `chg_lo`의 `regression_kernel_HIs.json`엔 그 시나리오 own 커널 이름 목록만
(다른 시나리오 것은 구조적으로 존재하지 않음). ⚠️ raw HI 쪽은 `redundancy_mask` 적용
**전** `gate_prob()`을 읽으므로, 결합 다중공선성으로 실제 기여가 0인 HI도 "선택됨"으로
보일 수 있다(`docs/260917_RESULTS.md` "남은 이슈" 참고, 아직 미수정).

## 13. 주요 파라미터 현재값 (2026-09-21 기준)

이후 실험이 바뀌면 이 표부터 갱신할 것 — 코드 기본값(`fixed.yaml`/`main_qfref_S.yaml`/
CLI default)과 최근 noscen/scen 재실행에 쓰는 override 값을 모아둔다.

**N_HI 토글** (`utils/hi_schema.py`, 프로세스 시작 시 환경변수로만 지정 가능 — 런타임 변경 불가):

| 환경변수 | 기본 | 켜면 제외되는 것 | N_HI |
|---|---|---|---|
| `SOH_EXCLUDE_STAT_LEAK` | run_pipeline.py가 자동 `1`(`--include-stat-leak`로 끔) | `stat_q_abs`/`stat_energy_seg` | 66→64 |
| `SOH_EXCLUDE_DQDV_LEAK` | `0`(run_pipeline.py `--exclude-dqdv-leak`로 켬) | `diff_dqdv_area` | -1(다른 토글과 독립 적용) |

즉 기본(N_HI=64)이고, 위 둘 다 켜면 63(HI63), 둘 다 끄면 66(HI66) — `docs/260920_REPORT.md`
안건2-3 참고. 학습·평가·상호작용검정·시너지·커널생성 전부 같은 값으로 맞춰야 하며
(`run_pipeline.py`가 Step 6~10에 매번 명시 주입), 어긋나면 shape 불일치로 즉시 죽는다.

**axis-config** (`--seg-axis q_frac_ref`) — noscen/scen을 lag=1로 맞추고 tile_scope를
분리한 최신 버전(§2 참고, `4_hi_analysis/hi_correlation.py`의 `_qfref_tag`가 경로명 생성):

| 항목 | noscen | scen |
|---|---|---|
| `assign` | `"none"`(라우팅 없음, 방향만 구분) | `"position_bin"`(기본, 존별 6시나리오) |
| `tile_scope` | `"full"`([0,1] 전체 고르게 배치) | `"zone"`(존 3개 안에서만 배치) |
| `ref_lag` | 1 | 1(0→1로 변경, noscen과 confound 제거) |
| `n1`/`n2`/`n_samples` | 0.35 / 0.20 / 2 | 0.35 / 0.20 / 2 |
| `min_pts` | 5 | 10(기본값, 명시 안 함) |
| `noise_amp`/`noise_mode`/`noise_period_cycles` | 0.03 / `"ou"` / 200 | 0.03 / `"ou"` / 200 |
| `calibration_period`/`offset_amp` | 100 / 0.005 | 미지정(비활성) |

⚠️ noscen과 scen이 `min_pts`/`calibration_period`/`offset_amp`에서 여전히 다르다(위 표) —
lag/tile_scope만 이번에 맞췄고 이 셋은 예전 설정 그대로 남아있는 추가 confound 후보.

**min_pts 기본값**: `q_frac_wide.__init__`의 클래스 기본값은 10 — `axis_config`에
`min_pts` 키가 없으면(현재 scen이 이 경우) 10이 그대로 쓰인다. 5 미만이면 세그먼트가
너무 짧아 통계 HI가 불안정해질 수 있어 하한 취급.

**raw HI 카테고리 비용** (`utils/hi_schema.py::CATEGORY_COSTS`, L0 페널티 가중치):

2026-09-19부터 `4_hi_analysis/hi_profile/hi_timing_cost.json`의 실측 세그먼트당 평균
처리 시간(µs)을 `morph=1.0` 기준으로 정규화한 값 — 예전엔 실측 없이 정한 추정치
(stat=1.0/diff=1.5/lfp=2.0/morph=3.0)였고 순서가 실측과 어긋났다(특히 stat을 가장
싸다고 가정했는데 실제론 고차 통계량 때문에 morph 다음으로 비쌈).

| 카테고리 | diff | lfp | stat | morph |
|---|---|---|---|---|
| 실측(µs/세그먼트) | 46.39 | 94.08 | 253.93 | 314.12 |
| 비용(morph=1.0 정규화) | 0.1477 | 0.2995 | 0.8084 | 1.0 |

커널 HI 비용은 멤버 raw HI 카테고리 비용의 산술평균(`build_kernel_group_features.py`가
pkl 저장 시점에 `f["cost"]`로 미리 계산, §9 예시 참고) — 구 pkl(cost 필드 없음)은 1.0 폴백.

⚠️ **2026-09-19부터 `--hi-cost-weighted-l0`(기본 False)로 토글** — 꺼져 있으면(현재
기본) 이 표는 raw/커널 둘 다 실제로는 적용 안 되고 전부 균일 비용 1.0이다. stat이
diff/lfp보다 훨씬 비싸게 나온 이유(개별 알고리즘 복잡도 — `stat_v_samp_ent`가 진짜
O(n²), `stat_v_skew`/`stat_v_kurt`가 scipy.stats 오버헤드로 비쌈, "카테고리" 자체의
문제는 아님)를 검증하는 동안 잠깐 끈 상태 — §13 전체가 "켰을 때의 값"이라는 점에 유의.

**loss** (`training/scr_loss.py` + `config/fixed.yaml`):

| 항목 | 기본값 | 비고 |
|---|---|---|
| `lambda_scen` | 0.01 | CE(분류) 가중치, probe_mlp 있을 때만 작동 |
| `lambda_l0` | `fixed.yaml`의 `lambda_l0_auto=true`면 자동 계산 | `0.01*sqrt(probe_scale*scen_scale)`을 `[1e-4,0.5]`로 clip, `probe_scale=10/avg(charge_probe_m,discharge_probe_m)`, `scen_scale=10/scen_k_count` |
| `--lambda-l0-override` | 없음(yaml/auto 무시하고 고정) | 지정하면 `charge_probe_m`/`discharge_probe_m`/`scen_k_count`는 완전히 무시됨(auto 분기 자체가 안 돎) — 이 세션의 noscen/scen/HI63/64/66 run은 전부 0.000237로 고정, `main_qfref_S.yaml`의 m=10/scen_k=25는 summary.json에 기록만 되고 학습엔 영향 없음 |
| `lambda_l0_warmup_epochs` / `lambda_l0_ramp_epochs` | 50 / 100 | 이후 `l0_fully_ramped_epoch=150`부터 체크포인트 후보 |
| `l0_norm_constant` | `None`(→ n_scenarios로 나눔) | scen=6, noscen=2 — 시나리오당 평균 페널티로 정규화(`docs/260915_RESULTS.md`, 파편화 confound 원인 아님이 이미 확인됨) |
| HardConcreteGate `BETA`/`GAMMA`/`ZETA` | 2/3(anneal 시작) / -0.1 / 1.1 | `--beta-min`(기본 0.1)까지 L0 warmup~ramp 구간에서 선형 anneal |

**학습 루프** (`config/fixed.yaml` + `main_qfref_S.yaml` + CLI):

| 항목 | 기본값 |
|---|---|
| `lr` / `weight_decay` | 2.0e-4 / 1.0e-3 (MLP_S 튜닝값) |
| `epochs` / `batch_size` | 500 / 2048 (`--max-epochs`/`--batch-size`로 오버라이드 가능) |
| optimizer / scheduler | AdamW / CosineAnnealingLR(`warmup_epochs=10`) |
| `grad_clip` | 1.0 |
| `--patience` | 60 (연속 no-improve epoch 수, 0이면 비활성) |

**체크포인트 선택** (§11 참고, 2026-09-18 변경):

| 항목 | 기본값 |
|---|---|
| 1순위 | val_rmse가 best보다 `--val-rmse-epsilon`(기본 **0.0005**) 이상 좋아지면 채택 |
| fallback | 위 조건 불성립(개선폭 < epsilon 또는 악화)이면 sat이 best보다 낮은 쪽 채택(val_rmse 상한 없음) |
| epsilon 근거 | `p1v4_noscen_gatefix_l0fix_seed42` 실측 val_rmse 표준편차(≈0.000246, L0 완전 램프 후 100 epoch)의 약 2배 |
| 체크포인트 후보 시작 | `epoch >= l0_fully_ramped_epoch`(웜스타트 없으면 150) |
