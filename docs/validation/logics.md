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
