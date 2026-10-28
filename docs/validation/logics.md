# logics.md — 분류기용 다중공선성 제거 로직 설계·구현·검증 기록

2026-10-09. `9_eval/plot_hi_usage_matrix.py`로 분류기의 커널 HI 사용 실태를 까본 것에서
시작해, 분류기 전용 다중공선성 제거 로직을 설계·구현·수정·검증한 전체 과정 기록.

---

## 1. 배경

- `1007_1328`/`1008_0144` run의 `gates/classification_kernel_HIs.json`을 까보니,
  분류기(Stage A)는 커널 HI 후보 58~59개 중 **단 2개**(`kernel_chg_mid_g2`,
  `kernel_dis_mid_g6`)만 쓰고 있었다 — 두 run에서 재현성 있게 동일했다. 회귀기는
  같은 run에서 13~15개를 씀(완전히 다른 패턴).
- `1002_1531`(분류기 커널 확장 코드가 없던 구 run)과 `1008_0144`(확장 적용 run)
  성능 비교에서도 차이가 거의 없었다(oracle R² 0.9379 vs 0.9363) — 확장이
  회귀 성능에 실질적 영향이 없다는 뜻.
- **결정**: 분류기는 커널 HI를 더 이상 쓰지 않는다(`with_probe_kernel=False`
  기본값으로 롤백, 완전 삭제가 아니라 토글). 다만 raw HI 다중공선성 제거는
  분류기에도 필요하다(회귀는 이미 하고 있었음) — 어떻게 적용할지가 이 문서의 본론.
- 핵심 난제: **분류기는 세그먼트의 "시나리오"를 아직 모른다**(그게 바로 분류기가
  예측해야 하는 대상). 반면 다중공선성 배제는 원래 시나리오별로 계산된다 —
  이 둘을 어떻게 연결할지가 설계의 핵심.

## 2. 설계 결정 (반복적으로 수정됨)

### 2.1 회귀기 — 변경 없음

- `7_kernel/kernel.py::k6_build_combined_redundancy`가 시나리오별로 raw+kernel
  결합 상관행렬에서 `|r|≥0.95` 쌍을 찾고, **degree(1순위) → SOH상관(2순위) →
  인덱스(3순위)** 규칙으로 패자를 정한다. 이 부분은 2026-09-18부터 있던 로직
  그대로, 이번 작업에서 손대지 않았다.

### 2.2 분류기 1차 시도 — 폐기

- 처음엔 회귀용과 **같은 결합(raw+kernel) 그래프**를 재사용하고 2순위 기준만
  "그 시나리오 안에서의 raw 값 분산"으로 바꿨다.
- **문제 발견**(사용자 지적): 분류기는 `with_probe_kernel=False`라 커널 HI를
  아예 안 받는데, "어떤 raw HI가 커널 HI와 중복"이라는 이유로 그 raw HI가
  제거되는 모순이 있었다 — 대체재(그 커널) 없이 정보만 사라짐.

### 2.3 분류기 2차 수정 — raw-only 독립 그래프

- 결합 그래프 대신 **raw HI만으로 새 상관행렬·edges·degree**를 계산(`corr_raw`/
  `edges_raw`/`degree_raw`). 커널 HI는 애초에 이 그래프에 안 들어가므로, 분류기가
  안 쓰는 피처와의 상관 때문에 raw HI가 빠지는 일이 사라짐.
- 합성 데이터로 검증: "커널과만 고상관인 raw HI"가 더는 분류기 목록에서 제거되지
  않음을 확인(`n_edges_classifier`가 `n_edges`보다 작게 나오는 케이스로 직접 확인).

### 2.4 분류기 3차 수정(최종) — tiebreak 기준 교체

- 2차 수정까지도 2순위 기준은 여전히 "raw 값 자체의 분산"이었는데, 사용자가 의도한
  건 **"시나리오 간 SOH-상관계수의 분산"**이었음이 재확인됨 — 전혀 다른 값.
- 이 지표는 이미 `5_interaction/interaction.py`가 계산해서 저장해 둔 값과
  정확히 같다: `per_hi[concept]["cell_level_std_r_mean"]`(셀 단위로 계산한 뒤
  평균 — pooled 방식의 편향을 걷어낸 정식 지표, 2026-07-23부터 채택).
- **직접 재계산하지 않고 재사용**하기로 함(pooled-bias 재도입 위험 회피) —
  `kernel.py`가 Step 5(interaction.py) 산출물을 새로 읽도록 의존성 추가.

### 2.5 시나리오 → 방향 레벨 합집합

- 커널.py는 여전히 시나리오 6개 각각에 대해 `removed_raw_idx_classifier`를
  따로 계산한다(분류기가 어느 시나리오인지 모른다고 해서 시나리오별 계산 자체를
  생략할 이유는 없음 — 모르는 건 "추론 시점에 어느 걸 쓸지"뿐).
- `train.py`가 그 방향(충전/방전)에 속한 3개 시나리오의 결과를 **합집합**
  (하나라도 허용하면 허용)해서 `(2, N_HI)`로 축소 — 기존 회귀 마스크를
  union하던 2026-10-07 방식과 동일한 보수적 원칙, 소스만 독립 계산으로 교체.

### 2.6 `with_probe_kernel` 토글

- §1의 결정(커널 미사용)을 `SCRModel(with_probe_kernel=False)`(기본값)로
  구현 — 완전 삭제가 아니라 토글이라 `True`로 주면 2026-10-07 동작(분류기도
  커널 후보 사용)을 그대로 재현할 수 있다.
- raw HI 다중공선성 제거(§2.1~2.5)는 이 토글과 **무관하게 항상 적용**된다.

## 3. 구현 파일

| 파일 | 함수/위치 | 역할 |
|---|---|---|
| `7_kernel/kernel.py` | `_resolve_redundancy_losers()` | degree→tiebreak→인덱스 공용 판정 로직 |
| | `k6_build_combined_redundancy()` | 회귀용(결합 그래프)+분류기용(raw-only 그래프) 둘 다 계산, JSON으로 저장 |
| | `k1_resolve_params_and_paths()` | interaction_json 자동 경로 해석 추가(§2.4) |
| | `k2_load_data()` | interaction.py의 `cell_level_std_r_mean`을 raw HI 인덱스 순서로 로드(`data.cross_scen_std_r`) |
| `8_train/train.py` | `_build_redundancy_mask(..., key=...)` | JSON → `(n_scenarios, N_HI)` bool 텐서(회귀/분류기 공용, key로 분기) |
| | `_build_probe_redundancy_mask()` | 분류기용 — 시나리오별 마스크를 방향별 합집합 |
| | `_apply_kernel_features(..., use_probe_kernel=)` | False(기본)면 분류기용 전체-시나리오 predict(비용 ~6배) 생략 |
| | `t3_build_tensor_masks()` | 위 함수들을 호출해 `masks.redundancy_mask`/`masks.probe_redundancy_mask` 구성 |
| `model_lib/models/scr_model.py` | `__init__(probe_redundancy_mask=, with_probe_kernel=)` | 외부에서 직접 마스크를 받음(내부 union 유도 제거) |
| | `_build_kernel_gates(..., with_probe_kernel)` | `with_probe_kernel=False`면 `probe_kernel_gates=None` |
| | `_apply_scen_gate()` / `_apply_probe_gate()` | 회귀/분류기 각각 자기 마스크를 게이트 출력에 곱함(마지막 적용 지점) |
| `parameters.py` | `ACTIVE_USE_PROBE_KERNEL = False` | 토글(§2.6) |
| `model_lib/training/scr_loss.py` | `_probe_kernel_hi_penalty()` | **수정 없음** — 이미 `probe_kernel_gates is None` 가드가 있어 안전 |

## 4. 검증

### 4.1 단위/합성 데이터 테스트 (스크래치, 재현 가능)

1. `_resolve_redundancy_losers` — degree/tiebreak 분기 단위 검증
2. `_build_redundancy_mask`/`_build_probe_redundancy_mask` — 합집합("하나라도 허용하면 허용") 설계가 분류기 필드에도 정확히 적용됨
3. `SCRModel(with_probe_kernel=False)`(기본) — `probe_kernel_gates is None`, forward 정상
4. `SCRModel(with_probe_kernel=True)` — 2026-10-07 동작 재현 가능 확인
5. `_apply_kernel_features(use_probe_kernel=False)` — `x_kernel`만 생성, `x_kernel_probe` 없음
6. 합성 데이터로 `k6_build_combined_redundancy` 직접 호출 — "raw-raw 진짜 중복"은 회귀/분류기 둘 다 잡고, "raw-kernel만 중복"은 분류기가 보존함을 확인(§2.3)
7. 합성 데이터로 cross_scen_std_r 기준 적용 확인 — 변별력 낮은(값이 작은) HI가 제거되고, 커널과만 중복인 HI는 여전히 보존됨(§2.4)
8. 실제 `hi_scenario_interaction_refact_interaction.json`으로 concept 이름 매칭 검증 — 66개 중 64개 정상 매칭, 2개(STAT leak 컬럼)는 interaction.py 쪽에 없어 fallback 0.0(train.py의 `shared_hi_mask`와 동일한 graceful fallback 패턴)

### 4.2 실제 데이터 검증 — `1009_1342_p1v2_refact_seed42` run

**시나리오별 원본 배제 개수**(`removed_raw_idx` vs `removed_raw_idx_classifier`):

| 시나리오 | 회귀(결합 그래프) | 분류기(raw-only 그래프) |
|---|---|---|
| chg_lo | 20 | 20 |
| chg_mid | 10 | 10 |
| chg_hi | 4 | 4 |
| dis_hi | 24 | 24 |
| dis_mid | 8 | 8 |
| dis_lo | 12 | 12 |

회귀/분류기 개수가 완전히 동일 — 확인해보니 이번 run의 `|r|≥0.95` 쌍이 전부
raw-raw 쌍이고, raw-kernel/kernel-kernel 쌍은 하나도 임계값을 안 넘었다
(`removed_kernel_names`도 6개 시나리오 전부 0개). 커널 HI는 그룹 전체를
Nystroem+Ridge로 비선형 결합한 값이라 특정 raw 멤버 하나와 거의 동일해지는
경우가 드문 게 정상 — §2.3의 raw-only 분리 수정이 "이 run에서는" 숫자상 차이를
안 만들었지만, 구조적으로는 여전히 필요한 수정(다른 축/run에서는 달라질 수 있음).

**방향별 합집합 후 분류기 최종 배제**(`probe_redundancy_mask`):

| 방향 | 배제 개수 | 배제된 raw HI 인덱스 |
|---|---|---|
| charge | 2/66 | [59, 63] |
| discharge | 6/66 | [15, 19, 37, 59, 62, 63] |

시나리오별 4~24개씩 지워지던 게, "3개 시나리오 전부가 동의해야 지운다"는
합집합 규칙으로 최종 2~6개만 실제로 꺼짐.

## 5. 현재 상태 / 남은 일

- `1009_1342` run 학습 완료 — 성능 비교는 §6 참고.
- 커널 재사용(`with_probe_kernel=True`) 관련 추가 lambda_l0 분리 조정(분류기
  전용 희소화 압력 분리) — 아직 미착수, 필요성 자체가 낮다고 판단돼 보류 중.
- raw-kernel 임계값 미초과가 이번 run 한정 현상인지, 다른 축/시드에서도
  일관된지는 추가 run으로 더 봐야 함.
- §6에서 제기된 "probe_redundancy_mask 단독 효과"를 깔끔히 분리하는 대조군
  run — 아직 미착수(필요성 논의 중).
- §7에서 제안한 `drop_probe_from_regression` 플래그 구현 + 재학습 비교 —
  아직 미착수. probe_x/scen_x 겹침이 다른 시드/run에서도 일관되는지 먼저
  확인 필요.

## 6. 성능 비교 — `1002_1531`(마스크 없음) vs `1009_1342`(분류기용 마스크 적용)

`1002_1531`은 §1에서 언급한 "분류기 커널 확장 코드가 없던 구 run"이자, 분류기
쪽 raw HI 다중공선성 마스크(`probe_redundancy_mask`) 자체가 없던 시절의 run이다
— 지금 막 완료된 `1009_1342`(§4.2의 그 run)와 성능을 직접 비교해봤다.

### 6.1 수치

| | oracle RMSE | oracle R² | hard RMSE | hard R² | selected_epoch | gate_saturation |
|---|---|---|---|---|---|---|
| `1002_1531`(마스크 없음) | 0.01736 | 0.9379 | 0.01736 | 0.9378 | 226 | 0.0178 |
| `1009_1342`(마스크 적용) | 0.01632 | 0.9451 | 0.01634 | 0.9450 | 300 | 0.0059 |

oracle R² 기준 **+0.7%p** — §4.1(a vs c, 0.16%p 차이)보다 훨씬 큰 차이라 원인을
따로 파봤다.

### 6.2 원인 — `probe_x`가 회귀기(`cap_head`) 입력에도 직접 들어감

- `config.yaml`을 두 run 간 `diff` — **완전히 동일**(출력 없음)
- `p1v2_summary.json` 전체 필드 비교 — 다른 건 경로 문자열(자기 폴더 경로)과
  `selected_epoch`/`gate_saturation`(학습 **결과**, 원인 아님)뿐
- 회귀 쪽 산출물(`kernel_group_features_refact_kernel.pkl`, `combined_redundancy`의
  `removed_raw_idx`)도 두 run이 전부 바이트 단위로 동일(커널 피처 58개, 시나리오별
  개수까지 일치)
- 즉 **통제된 입력 차원에서 유일한 차이는 `probe_redundancy_mask` 유무**
  (`1002_1531`=없음, `1009_1342`=charge 2/66·discharge 6/66 강제 0)
- oracle은 분류기 예측과 무관하게 진짜 시나리오로 라우팅되는 지표인데도 수치가
  바뀐 이유: `scr_model.py:557`
  ```python
  feat_parts = [probe_x, scen_x]   # cap_head(회귀) 입력
  ```
  `probe_x`는 분류기 전용이 아니라 **분류기+회귀기 공용 입력**이다 — 분류기용으로
  설계한 마스크가 회귀기 입력의 일부도 같이 바꾸는 구조라서, "분류기 전용" 변경인데도
  oracle 성능까지 움직인다.

### 6.3 해석 및 유보

- `cross_scen_std_r` 기준으로 제거된 2~6개 raw HI는 "시나리오마다 SOH상관이
  거의 안 변하는" HI라, 애초에 회귀적으로도 변별력이 크지 않았을 가능성이 있다
  — 빠지면서 약간의 정규화 효과(과적합 감소)가 났을 수 있음(가설, 미검증).
- 다만 `selected_epoch`(226→300)이 꽤 차이 나는 등 학습 자체의 확률적 변동폭도
  있어, 0.7%p 전부를 이 메커니즘 하나로 단정하긴 이르다. **"probe_redundancy_mask만
  끄고 나머지는 1009_1342와 동일"한 대조군 run**이 있어야 인과관계를 깔끔하게
  분리할 수 있음(§5 남은 일).

## 7. `probe_x`/`scen_x` 중복도 분석 — probe_x를 회귀기에서 분리하는 근거(2026-10-14)

§6.2가 지적한 "`probe_x`가 분류기 전용이 아니라 회귀기(`cap_head`) 입력에도
공용으로 들어간다"는 구조적 문제를, 마스크 유무가 아니라 **애초에 `probe_x`를
`feat_parts = [probe_x, scen_x]`(`scr_model.py` forward())에서 빼야 하는 것
아니냐**는 더 근본적인 질문으로 이어서 조사한 기록.

### 7.1 배경 — 왜 이게 문제가 될 수 있는가

- `probe_gate`(Stage A, `charge_probe_gate`/`discharge_probe_gate`)는 MSE(회귀)
  와 CE(분류) 양쪽 그래디언트를 동시에 받는 구조다(`scr_model.py` 상단
  docstring: "probe_gate receives gradients from BOTH MSE (regression) and
  CE (classification)").
- `noscen`(시나리오 라벨 없음, `assign="none"`) 조건에서는 분류 과제가
  충전/방전 2-클래스로 퇴화하고 `direction`이 `probe_mlp`에 별도 피처로
  직접 들어가므로, CE가 `probe_x`에 거의 의존하지 않고 0에 가깝게
  떨어진다 — 실제로 `1011_1710_p1v2_noscen_full_seed42`의
  `train_log_v2.csv`에서 `tr_ce`가 **전 에폭에 걸쳐 0.000000**으로 찍혀
  있음을 확인(반면 `1009_1342_p1v2_baseline_seed42`는 ep1=1.008→
  late=0.00068로 정상적으로 학습됨).
- 즉 `scen`(정식 6-시나리오) run에서만 `probe_gate`가 CE로부터 실질적
  그래디언트를 받고, 이게 같은 파라미터를 통해 회귀 입력(`probe_x`)에도
  반영된다 — `scen` vs `noscen` 비교가 "시나리오 조건화 자체의 효과"뿐
  아니라 이 공유-게이트 간섭까지 섞여 있을 가능성.

### 7.2 실측 — probe_x와 scen_x가 고르는 HI가 얼마나 겹치는가

`1009_1342_p1v2_baseline_seed42`의 `gates/classification_HIs.json`
(probe_x, Stage A)과 `gates/regression_HIs.json`(scen_x, Stage B)을
`gate_prob≥0.9`(기존 `docs/figures/fig5_hi_selection_trimmed.py`와 동일
threshold) 기준으로 "선택됨" 판정 후, **HI 인덱스(0~63, `hi_cols_ref`
라벨이 아니라 `_ranked_indices()`가 반환하는 실제 인덱스)** 기준으로 교집합 계산.
(`hi_cols_ref`는 probe/scen이 공유하는 라벨용 이름표일 뿐이라 이름 문자열
매칭은 부정확 — 인덱스 비교가 맞는 방법.)

**활성 HI 개수**(threshold≥0.9):

| | 활성 개수 |
|---|---|
| probe_charge | 16/64 |
| probe_discharge | 17/64 |
| chg_lo / chg_mid / chg_hi | 15 / 12 / 15 |
| dis_hi / dis_mid / dis_lo | 13 / 13 / 13 |

**probe(방향) ∩ scen(그 방향 소속 시나리오) 겹침**:

| | 겹침/전체 |
|---|---|
| probe_charge(16) ∩ chg_lo(15) | 14 |
| probe_charge(16) ∩ chg_mid(12) | 11 |
| probe_charge(16) ∩ chg_hi(15) | 14 |
| probe_discharge(17) ∩ dis_hi(13) | **13 (100%)** |
| probe_discharge(17) ∩ dis_mid(13) | **13 (100%)** |
| probe_discharge(17) ∩ dis_lo(13) | 11 |

discharge 쪽은 `dis_hi`/`dis_mid`에서 scen_x가 고른 HI가 **전부** probe_x에도
들어있고, charge 쪽도 87~93% 수준으로 거의 포개진다.

### 7.3 해석 — probe_x 분리 제안의 근거

- `probe_x`와 `scen_x`가 거의 같은 HI 조합을 고르고 있다면, 지금
  `cap_head` 입력 `[probe_x, scen_x, ...]`은 새 정보를 더하기보다 **거의
  같은 정보를 두 번 중복 투입**하고 있을 가능성이 높다.
- 겹침이 이렇게 크면 **probe_x를 회귀 입력에서 빼도 정보 손실은 적을
  것으로 예상**되는 반면, `probe_gate`는 CE 전용 그래디언트만 받게 되어
  §7.1의 "공유 게이트 간섭" 경로가 구조적으로 차단된다.
- **제안하는 변경**(2026-10-14 구현 완료): `SCRModel`에 `drop_probe_from_regression`
  플래그를 추가해 `feat_parts = [scen_x, ...]`(probe_x 제외)로 cap_head
  입력을 구성 — `probe_mlp`(분류기)는 그대로 `probe_x`를 받으므로 CE
  학습 자체엔 영향 없음. Phase 1 기본 헤드(`MLPHead`)/`resnet_tab`만 지원
  (`cap_heads.py`의 `_HEAD_IN` 상수 및 Transformer/i_transformer 헤드의
  `x[:, :N_HI]`/`x[:, N_HI:2*N_HI]` 슬라이싱은 다른 헤드 타입 전용이라
  이 ablation 범위 밖 — 조합 시 `NotImplementedError`). 합성 데이터
  smoke test로 forward+backward 정상 동작 확인(`cap_head` 입력 차원이
  의도대로 `N_HI`만큼 줄어듦, `probe_gate`는 여전히 0 아닌 그래디언트
  받음 — CE 경로는 살아있고 MSE 경로만 끊김).
- **유보**: 겹침 분석은 `1009_1342` run(seed42) 하나에 대한 스냅샷이다 —
  다른 시드/run에서도 겹침 정도가 일관되는지, 그리고 실제로
  `drop_probe_from_regression=True`로 재학습했을 때 oracle/hard 성능이
  어떻게 변하는지는 아직 미검증(§5 "남은 일"에 추가할 항목, §7.4도 참고).

### 7.4 결론 — probe_x는 회귀기에서 빼는 게 맞다(2026-10-16)

§7.2/7.3의 겹침 분석과 §6.2의 0.7%p 성능 이동 관찰을 종합한 최종 판단.

- **구조적 근거**: `probe_x`/`scen_x` 겹침이 방향에 따라 65~100%에 달해,
  `probe_x`를 회귀 입력에 추가해도 **새 정보가 거의 없다** — 거의 같은
  정보를 두 번 넣는 것에 가깝다.
- **실측 근거**: §6.2에서 "분류기 전용"으로 설계한 `probe_redundancy_mask`
  변경이 **oracle 회귀 성능(분류기 예측과 무관한 지표)을 0.7%p 움직였다**
  — 분류기와 회귀기가 의도와 달리 결합돼 있다는 직접 증거.
- **리뷰어 방어 관점**: `probe_x`를 유지하면 논문이 (1) 공유 구조의
  정당성, (2) 중복 선택의 의미, (3) §7.1의 그래디언트 간섭 가능성까지
  **세 가지를 동시에 방어**해야 한다. 제거하면 (1)(2)가 질문 대상에서
  사라지고, "분류기=배포 라우팅 전용, 회귀기=순수 회귀 목적 게이트"라는
  더 간결한 설명으로 대체된다.
- **아직 확정 아님**: 위 근거는 모두 간접적(인덱스 겹침, 성능 이동의
  상관관계)이다. `drop_probe_from_regression=True`로 실제 수렴까지
  학습해 RMSE/R²가 유지되는지 확인해야 이 권고가 "결과로 뒷받침된
  설계 변경"이 된다 — 확인되면 ablation이 아니라 **메인 아키텍처**로
  승격하는 것을 권고.

## 8. q_ref(SOC reference)가 분류기/회귀기 입력에 안 들어가는지 검증 (2026-10-21)

`docs/writing/#2_METHODOLOGY.md` 2.2절의 주장("이 노이즈 모델은 시나리오 축[라벨]을
정의하는 데만 쓰이고, 분류기나 회귀기(§2.6)의 입력으로는 절대 들어가지 않는다")을
코드로 직접 추적 검증한 기록.

### 8.1 결론 — 기본 설정에서는 참, 단 명시적 예외(토글) 하나 존재

**부분 확인(PARTIALLY CONFIRMED WITH CAVEAT)**: 기본 파이프라인
(`ACTIVE_SOC_REF_TO_REGRESSION=False`)에서는 claim이 정확하다. 그런데
`model_lib/models/scr_model.py`에 **`soc_ref_to_regression` 토글이 있어 켜면
q_ref 값 자체가 회귀기(`cap_head`) 입력에 직접 추가**된다 — 죽은 코드가 아니라
`run_socref_sweep.py`가 실제로 스윕하는 활성 ablation 경로다(마침 지금 진행 중인
`1020_1220_p1v2_socref_scen_seed42` run이 바로 이 토글을 켠 실행으로 보임).
"never"라는 절대적 표현은 코드 전체 기준으로는 부정확 — 본문에서는 "기본
설정에서는"이라는 조건을 달아야 한다.

### 8.2 세부 검증(파일:라인)

- **Step 4 — 존 경계는 고정 상수, noisy q_ref는 멤버십 판정에만 사용**:
  `common/scenario/q_frac_ref.py:511-534`(`_normalizer()`)이 OU 노이즈를 주입한
  `q_ref`(`q_ref_raw * (1+noise_frac)`)를 반환하고, `q_frac_wide.py:398-416`
  (`_extract()`)이 이 값을 분모로 세그먼트가 어느 존에 속하는지만 판정한다.
  저장되는 `q_frac_lo`/`q_frac_hi` 컬럼(`q_frac_wide.py:412-413`)은 0.0/0.4/0.7/1.0
  같은 **고정 경계 상수**이고, noisy 분모로 나눈 값이 아니다. 노이즈가 섞인
  분모 자체는 `q_ref`라는 별도 컬럼으로 메타데이터에만 남는다
  (`q_frac_wide.py:415`, `4_hi_analysis/logics.py:208`).
- **raw HI 값 자체는 q_ref 노이즈와 무관**: `lfp_inflect_q_frac`/
  `lfp_plateau_q_onset`(`4_hi_analysis/tools/hi_compute.py:643`/`702`)처럼
  이름에 "q_frac"이 들어간 HI도, 전부 그 세그먼트 **자신의** raw v/|i|/dt
  배열로 만든 `_build_vq_curve`(같은 파일 220행) 기반이라 q_frac_ref의
  전역 noisy 분모와는 독립적이다. (`stat_q_abs`/`stat_energy_seg`는 별개의
  누수 우려 — 현재/과거 용량 비교 문제이고 q_ref 노이즈와는 무관.)
- **`segment_dataset.py` — x_hi에서 q_frac_lo/q_ref 제외**: `_NATIVE_HI_COLS`
  (119행)는 stat_/diff_/lfp_/morph_ 접두사만 포함(`hi_schema.py:145-149`
  `get_native_hi_cols()`). `q_frac_lo`/`q_ref`는 184-194행에서 `_hi_cols`가
  아니라 205행의 `meta_cols`로 분리돼 애초에 텐서에 안 들어간다.
  `scen_idx`/`direction`/`level`도 `q_frac_lo`가 아니라 `segment_id` →
  `spec.scenario_to_dir_class`(179-182행)로 역산된다.
- **`scr_model.py::forward()` — 기본은 제외, 토글로 포함 가능**: 기본
  `feat_parts = [probe_x, scen_x, (kernel_x,) direction, cap_init]`
  (616-631행)에 q_ref는 없다. `cap_init`은 셀의 실측 첫 사이클 용량(또는
  고정 공칭용량)으로, q_ref의 lag/노이즈 메커니즘과 무관하다. 다만
  625-626행: `if self.soc_ref_to_regression: feat_parts.append(batch["q_ref"].unsqueeze(1))`
  — 플래그가 켜지면 z-score된 q_ref를 그대로 추가한다.
- **토글 배선**: `parameters.py:58` `ACTIVE_SOC_REF_TO_REGRESSION = False`(기본)
  → `model_cfg["soc_ref_to_regression"]`(`parameters.py:280`,
  `cap_heads.py:429-436`) → `segment_dataset.py`가 `q_ref_mean_`/`q_ref_std_`로
  z-score해 `SegmentDataset.__getitem__`(475행)에서 배치에 포함
  (`"q_ref": self.q_ref[idx]`). `run_socref_sweep.py:60-64,88`이 이 플래그를
  `True`로 놓고 실제로 스윕한다.

### 8.3 논문 서술 수정 권고

"it is never an input to the classifier or the regressor"는 **기본 설정 한정**으로
완화해야 한다 — 예: "...it is never an input to the classifier or the regressor
**in the default configuration reported here**; §3.x reports a controlled ablation
(`soc_ref_to_regression`) that instead gives the regressor direct access to $q_\text{ref}$ and
causes [결과]." 지금 문장 뒤에 이미 "an ablation that instead gave the regressor
direct access to $q_\text{ref}$ caused severe validation overfitting..."라는 절이
붙어 있어 이 ablation 자체는 이미 언급돼 있다 — 다만 앞 절의 "never"가 그
ablation과 모순되는 절대적 표현이라는 점만 손보면 된다.

## 9. 평가 코드(`9_eval/test.py`)가 `probe_redundancy_mask`를 빠뜨린 버그 (2026-10-25)

### 9.1 증상
`drop_probe_from_regression=False`(probe_x를 회귀 입력에도 사용)인 noscen run들이 test에서
R²가 음수까지 붕괴(`1022_0416` -5.2, `1024_0347` -3.4). 그런데 학습 로그의 val은 정상(R²≈0.94)이고,
같은 체크포인트를 `test.py` 경로로 train/val split에 돌려도 똑같이 붕괴 — test 셀 분포 문제가 아니라
"평가 때 재구성한 모델이 학습된 모델과 다름"이 원인.

### 9.2 원인
- `train.py::t3_build_tensor_masks`는 §2.5의 분류기용 마스크(`probe_redundancy_mask`, (2, N_HI))를
  만들어 `SCRModel`에 넘기는데, `test.py::_build_and_load_model`은 이를 넘기지 않았다.
- `SCRModel`은 이 마스크를 `register_buffer(..., persistent=False)`로 등록 → 체크포인트에 저장되지 않고,
  `load_state_dict(strict=True)`도 이 누락을 잡지 못한다.

### 9.3 왜 일부 run만 망가지나 — 마스크된 칸의 게이트가 열려 있을 때만
`probe_x[i] = x[i] × z[i](게이트) × m[i](마스크)`, 마스크된 HI는 학습 내내 `m[i]=0`:
1. 출력이 항상 0이라 손실이 게이트 `z[i]`에 그래디언트를 주지 않는다 — 닫는 힘은 L0 패널티뿐이라
   학습이 끝나도 열린 채(`log_alpha > ln(1/11)`, 평가 시 z=1) 남을 수 있다.
2. 이 칸을 받는 다음 층 가중치도 입력이 늘 0이라 그래디언트가 0 → 초기 랜덤값 그대로
   (AdamW lr 2e-4·wd 1e-3로는 수십만 step 동안 약 3%만 줄어듦).

평가 때 마스크가 빠지면 게이트가 닫힌 칸은 여전히 0이라 무해하지만, 열린 칸에는 학습 때 한 번도 본
적 없는 실제 HI 값이 랜덤 가중치를 거쳐 들어가 예측이 체계적으로 밀린다. scen은 마스크가 방향 내
시나리오 3개 **교집합**이라 2~6개뿐이고 게이트도 전부 닫혀 있어 영향이 0, noscen은 방향당
13/12개가 마스크되고 그중 일부 게이트가 열린 채 남아 누수가 생겼다. `drop_probe_from_regression=True`
run에서는 probe_x가 분류기에만 들어가므로 hard/soft만 영향(noscen은 hard=oracle이라 무영향).

### 9.4 조치
- `test.py`가 `train.py::_build_probe_redundancy_mask`를 그대로 import해 모델에 넘기도록 수정.
  `9_eval/scenario_attribution.py`, `docs/figures/_extract_embedding.py`는 `test.py` 로더를 재사용하므로
  함께 고쳐짐. `model_lib/tools/visualize_results.py`는 별도 구현이라 두 마스크 모두 여전히 누락(미수정).
- 검증(`1024_0347`): 마스크 복원 시 train/val/test RMSE 0.1725/0.1616/0.1595 → 0.0140/0.0198/0.0217
  (test R² -3.43 → 0.9238).
- 체크포인트에서 "마스크된 HI 중 평가 시 게이트가 열리는 수"를 전 run 스캔 → 0이 아닌 8개 run만
  재평가 대상(재학습 불필요, Step 9만): 회귀까지 오염 `1010_1118`, `1011_1035`, `1022_0416`,
  `1022_1626`, `1024_0347`, `1024_1522` / 분류기만 오염 `1016_1211`, `1018_0755`.
  기존 산출물은 각 run의 `_backup_pre_probe_mask_fix/`에 백업.

### 9.5 재평가 중 발견한 두 번째 문제 — 평가 데이터 경로
- `test.py`는 `P.ACTIVE_DATA_DIR`/`P.ACTIVE_SEG_DATA_DIR`이 None이 아니면 run의 `config.yaml` 데이터
  경로를 덮어썼는데, 이 값은 현재 `ACTIVE_AXIS_CONFIG`에서 항상 계산돼 None이 될 수 없다. 파이프라인
  안(학습 직후 Step 9)에서는 둘이 같아 무해하지만, 단독 재평가에서는 **지금 parameters.py의 축**으로
  평가된다. 1차 재평가 때 scen run(`1011_1035`, `1016_1211`, `1018_0755`)이 noscen·n2=20% 캐시로 평가돼
  oracle까지 붕괴(예: `1018_0755` 0.843 → -10.35)한 것이 이 때문 — 마스크 수정과 무관.
- 조치: `test.py`가 데이터 경로는 항상 run의 `config.yaml` 값을 쓰도록 수정. 같은 함수에서, 학습 후 run
  폴더 이름이 바뀌어 `p1v2_summary.json`의 절대경로가 깨진 경우(`1010_1118`) 같은 파일명을 run 폴더
  안에서 찾도록 보강.

### 9.6 재평가 결과 (oracle R² / hard R² / hard 분류정확도, 수정 전 → 후)

| run | 수정 전 | 수정 후 |
|---|---|---|
| `1010_1118` warmstart150 (probe 회귀 포함) | 0.9429 / 0.9429 / 100% | 0.9453 / 0.9453 / 100% |
| `1011_1035` warmstart200 (probe 회귀 포함) | 0.9115 / 0.9074 / 98.25% | **0.9474 / 0.9474 / 99.98%** |
| `1016_1211` no_gate (probe 회귀 제외) | 0.9455 / 0.8009 / 94.72% | 0.9455 / **0.9455** / 100% |
| `1018_0755` n2=5% scen (probe 회귀 제외) | 0.8430 / 0.7383 / 82.25% | 0.8430 / **0.8419** / 99.33% |
| `1022_0416` noscen mlp probe ws100 | -5.2152 | **0.9288** |
| `1022_1626` noscen transformer probe ws100 | 0.7711 | **0.9392** |
| `1024_0347` noscen mlp probe nowarm | -3.4323 | **0.9278** |
| `1024_1522` noscen transformer probe nowarm | 0.9388 | 0.9388 (변화 없음) |

probe 회귀 제외 run(`1016_1211`, `1018_0755`)은 oracle이 그대로이고 hard만 바뀌어 9.3의 메커니즘과 일치.
이 결과로 뒤집히는 기존 해석: warmstart=200 "붕괴"(분기 시점 이후 그래디언트 소실 가설)와 n2=5%의
"분류기 정확도 붕괴/`dis_mid` 30.9%"는 둘 다 이 버그의 산물이었다.
