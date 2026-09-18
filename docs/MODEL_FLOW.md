# MODEL_FLOW — SCR 모델 학습과정 코드 흐름

`phase1_trainer_v2.py main()` 실행 순서를 코드 문단 단위로 정리. 각 항목: 2줄 설명 + 예시.
관련 파일: `models/scr_model.py`, `models/hard_concrete.py`, `training/scr_loss.py`.

## 1. 커널 HI 준비 — `_apply_kernel_features()`

`--kernel-features-pkl`이 주어지면, 시나리오별로 그 시나리오가 만든 RBF 커널 HI만
[0:K_s) 구간에 채우고 나머지는 0-패딩한 `x_kernel` 텐서를 각 데이터셋에 붙인다.
각 커널 피처는 멤버 raw HI 카테고리 비용의 평균(`cost`)도 함께 실어온다.

예: `dis_hi` 시나리오가 커널 5개(K_5=5)를 만들었다면 `x_kernel[:, 0:5]`엔 실값,
`chg_lo`(K=3)면 `x_kernel[:, 3:5]`는 항상 0. `kernel_hi_counts=[..,5,..]`로 전달됨.

## 2. 결합 다중공선성 마스킹 — `_apply_combined_redundancy_raw()` / `_build_redundancy_mask()`

`--combined-redundancy-json`이 주어지면, raw+커널을 시나리오별로 합쳐 |r|≥0.95인 쌍 중
degree/타깃상관 기준으로 진 쪽을 raw는 `nan_mask`(입력 0-강제)로, 게이트 레벨에서도
`redundancy_mask`로 이중 차단한다.

예: `chg_lo`에서 `stat_v_std`가 `diff_dvdq_std`와 |r|=0.97이고 degree가 더 높으면
`redundancy_mask[chg_lo_idx, v_std_idx]=False` → 그 자리는 log_alpha와 무관하게 항상 0.

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

## 7. Forward — cap_head 입력 결합

`[probe_x, scen_x, kernel_x(선택), raw_cnn/raw_flat(선택), scenario_onehot(선택),
direction, cap_init]`을 이어붙여 `cap_head` MLP에 넣어 SOH 예측값 하나를 낸다.

예: 기본 구성이면 `feat = [probe_x(66) | scen_x(66) | kernel_x(K) | direction(1) |
cap_init(1)]` → `cap_head(feat) -> cap_pred (B,)`.

## 8. HardConcreteGate 메커니즘 — `hard_concrete.py`

학습 중엔 Gumbel-등가 연속 완화(`s = sigmoid((logU-log(1-U)+log_alpha)/BETA)`)로
미분 가능한 게이트를 쓰고, 추론 시엔 `sigmoid(log_alpha)` 기준 하드 0/1 마스크로 바뀐다.

예: `log_alpha`가 커질수록 `gate_prob()`(L0 페널티에 쓰이는 P(z>0))가 1에 가까워짐 —
학습이 끝나면 `active_indices()`로 "이 HI는 실제로 선택됐다" 확정.

## 9. SCRLoss — `training/scr_loss.py`

`total = MSE + lambda_scen*CE + lambda_l0*L0penalty + lambda_shrink*shrink`.
L0penalty는 probe/scen(raw, 카테고리 비용 가중) + kernel(멤버 카테고리 비용 평균 가중)을 합산.

예: `diff` 카테고리 raw HI는 비용 1.5, `morph`는 3.0이라 도입 시 페널티가 더 커
같은 gate_prob이어도 `morph` HI가 더 강하게 억제됨. 커널 HI도 이제 동일 원리로
멤버 raw HI 카테고리 평균 비용을 받는다(예: stat 2개+diff 2개 멤버 → 비용 (1.0+1.0+1.5+1.5)/4=1.25).

## 10. 학습 루프 — `main()`의 `for epoch in trange(epochs)`

매 epoch마다 lambda_l0를 웜업/램프 스케줄로 올리고 BETA(gate 온도)를 anneal하며,
배치마다 forward→loss→backward→clip→step 후 train/val RMSE·R²를 기록한다.

예: `l0_warmup_ep=50` 동안 `eff_l0=0`(게이트 자유 학습) → 이후 50 epoch에 걸쳐
목표 `lambda_l0`까지 선형 램프, 동시에 `BETA`가 2/3→0.1로 내려가며 게이트가 점점 이산화.

## 11. 체크포인트 선택 — val_rmse 우선, gate saturation은 동률 tie-break

L0가 완전히 램프된 이후 구간에서 val_rmse가 `--val-rmse-epsilon`(기본 0.0005)보다 뚜렷하게
낮아야 "진짜 개선"으로 채택하고, 그 이내(노이즈 수준 동률)면 gate saturation(sat, 애매구간
[0.1,0.9] 게이트 비율)이 더 낮은 쪽을 택한다(2026-09-18, 기존 sat-1순위 기준을 뒤집음).

예: epoch 187(val_rmse=0.015276, 최솟값)과 204(val_rmse=0.015712, sat=0.0035로 더 이산화)가
있으면 차이(0.000436)가 epsilon 이내라 동률 취급 → sat이 더 낮은 204를 베스트로 채택
(`p1v4_noscen_gatefix_l0fix_seed42` 실측 로그로 검증).

## 12. 산출물 저장 — `train_scr.py`의 `_save_scen_masks_to_json` / `_plot_gate_probs`

베스트 체크포인트 기준으로 게이트별 `gate_prob()`을 읽어 `regression_HIs.json`
(raw)/`regression_kernel_HIs.json`(kernel)과 `hi_selection_matrix.png`를 만든다.

예: `chg_lo`의 `regression_kernel_HIs.json`엔 그 시나리오 own 커널 이름 목록만
(다른 시나리오 것은 구조적으로 존재하지 않음). ⚠️ raw HI 쪽은 `redundancy_mask` 적용
**전** `gate_prob()`을 읽으므로, 결합 다중공선성으로 실제 기여가 0인 HI도 "선택됨"으로
보일 수 있다(`docs/260917_RESULTS.md` "남은 이슈" 참고, 아직 미수정).

## 13. 주요 파라미터 현재값 (2026-09-18 기준)

이후 실험이 바뀌면 이 표부터 갱신할 것 — 코드 기본값(`fixed.yaml`/`main_qfref_S.yaml`/
CLI default)과 최근 noscen/scen 재실행에 쓰는 override 값을 모아둔다.

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

| 카테고리 | stat | diff | lfp | morph |
|---|---|---|---|---|
| 비용 | 1.0 | 1.5 | 2.0 | 3.0 |

커널 HI 비용은 멤버 raw HI 카테고리 비용의 산술평균(`build_kernel_group_features.py`가
pkl 저장 시점에 `f["cost"]`로 미리 계산, §9 예시 참고) — 구 pkl(cost 필드 없음)은 1.0 폴백.

**loss** (`training/scr_loss.py` + `config/fixed.yaml`):

| 항목 | 기본값 | 비고 |
|---|---|---|
| `lambda_scen` | 0.01 | CE(분류) 가중치, probe_mlp 있을 때만 작동 |
| `lambda_l0` | `lambda_l0_auto=true`면 자동 계산 | `0.01*sqrt(probe_scale*scen_scale)`을 `[1e-4,0.5]`로 clip, `probe_scale=10/avg(m)`, `scen_scale=10/scen_k` — 기본 m=10/scen_k=25면 ≈0.00632 |
| `--lambda-l0-override` | 없음(yaml/auto 무시하고 고정) | 최근 noscen/scen 재실행은 0.000237로 고정(과거 스윕 최적값, §MODEL_FLOW 커널 페널티 도입 전 값이라 재보정 필요 가능성 있음) |
| `lambda_l0_warmup_epochs` / `lambda_l0_ramp_epochs` | 50 / 100 | 이후 `l0_fully_ramped_epoch=150`부터 체크포인트 후보 |
| `lambda_shrink` | 0.0(비활성) | `--shrinkage-gate` 켜야 의미 있음 |
| `l0_norm_constant` | `None`(→ n_scenarios로 나눔) | scen=6, noscen=2 — 시나리오당 평균 페널티로 정규화(§MODEL_FLOW 12 "재실행 전 리뷰" 참고, 이 정규화 자체는 파편화 confound 원인 아님이 이미 확인됨) |
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
| 1순위 | val_rmse가 더 낮으면 채택 |
| tie-break | `\|Δval_rmse\| ≤` `--val-rmse-epsilon`(기본 **0.0005**)이면 sat이 더 낮은 쪽 |
| epsilon 근거 | `p1v4_noscen_gatefix_l0fix_seed42` 실측 val_rmse 표준편차(≈0.000246, L0 완전 램프 후 100 epoch)의 약 2배 |
| 체크포인트 후보 시작 | `epoch >= l0_fully_ramped_epoch`(웜스타트 없으면 150) |
