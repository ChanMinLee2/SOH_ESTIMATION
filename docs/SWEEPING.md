# 논문 리포트용 하이퍼파라미터 스위핑 계획

> 2026-10-11 정리. `PREWRITE_v1.md`의 3대 기여(① HI 설계/부분구간, ② 시나리오
> 조건 게이팅, ③ 시너지/커널 융합)에 맞춰 "어떤 파라미터를 스윕해서 리포트해야
> 하는가"를 정리함. 전부 본문 Figure로 넣기엔 10~15p 예산(PREWRITE_v1.md 상단
> 메타노트)을 넘기므로, 우선순위를 나눠뒀음 — 아래 "우선순위" 열 참고.

## 우선순위 기준

- **A(본문 Figure)**: 논문의 핵심 주장과 직결 — 민감도 곡선/표를 본문에 넣어야 함
- **B(부록 표)**: 보고는 필요하지만 본문 Figure까지는 과함 — 부록에 요약 표 1개로 압축
- **C(한 줄 보고)**: "이 범위에서 안정적이었음"처럼 텍스트 한 줄로 충분

---

## 기여① HI 설계·부분구간 대응 관련

| 파라미터 | 위치 | 현재값 | 왜 필요한가 | 우선순위 |
|---|---|---|---|---|
| `n2`(세그먼트 길이) | `ACTIVE_AXIS_CONFIG["n2"]` | 0.20 | "얼마나 짧은 구간까지 버티는가"가 기여②(부분구간 대응)의 핵심 주장 — 성능 곡선 자체가 사실상 그 주장의 증거임 | **A** |
| `n1`(존 경계) / `n_samples` | `ACTIVE_AXIS_CONFIG` | 0.35 / 2 | n2보다 우선순위 낮음, 축 정의 민감도 확인용 | B |
| `ref_lag`(SOC 레퍼런스 지연) | `ACTIVE_AXIS_CONFIG["ref_lag"]` | 1 | "배포 현실성"(SOC reference) 주장과 직결 — 지연이 커질수록 성능이 어떻게 떨어지는지 | **A** |
| `noise_amp` / `calibration_period` | `ACTIVE_AXIS_CONFIG` | 0.03 / 100 | 센서 바이어스·재보정 주기 강건성 — ref_lag와 묶어서 같은 Figure(다축 곡선)로 가능 | B |
| `ACTIVE_N_HI`(leakage 제외 범위) | `parameters.py` | 64 (63\|64\|66 중) | 63→64→66 성능 비교로 "리키지 HI 제외가 실제로 타당했다"를 보여줌 — 하나의 ablation 표 | B |

## 기여② 시나리오 조건 게이팅 관련

| 파라미터 | 위치 | 현재값 | 왜 필요한가 | 우선순위 |
|---|---|---|---|---|
| `ACTIVE_LAMBDA_L0_OVERRIDE`(L0 페널티 강도) | `parameters.py` | 0.000237 | 희소화 압력 자체 — 서브셋 크기·성능 트레이드오프의 1축 | **A** |
| `scen_k_count` / `charge_probe_m` / `discharge_probe_m`(게이트 top-k 예산) | `P1_MODEL_CONFIG["regression"/"classifier"]` (오버라이드: `FIXED_SCEN_K_COUNT`/`FIXED_CHARGE_PROBE_M`/`FIXED_DISCHARGE_PROBE_M`) | 25 / 10 / 10 | "서브셋 크기"를 직접 결정 — l0_lambda보다도 더 직접적인 예산 파라미터 | **A** |
| `lambda_scen`(분류 CE 손실 가중치) | `P1_MODEL_CONFIG["loss"]["lambda_scen"]` | 0.01 | probe_gate가 MSE+CE 동시에 받는 구조라, 이 비율이 "분류 정확도 vs 회귀 유용성" 중 뭘 우선할지 결정 | B |
| `lambda_l0_schedule` + `warmup_epochs`/`ramp_epochs`(L0 스케줄 형태) | `P1_MODEL_CONFIG["loss"]` | delayed_warmup / 50 / 100 | 이번 세션에 직접 확인 — 스케줄을 branch 시점에 리셋하느냐만으로 noscen 대비 결과가 바뀜(`docs/validation/logics.md` §6 연장선) | **A** |
| `ACTIVE_WARMSTART_BRANCH_EPOCH`(T) | `parameters.py` | 200 | 이 메커니즘을 최종적으로 쓴다면 T 자체가 스윕 대상(현재 150→200 등 테스트 중) | B |

## 기여③ 시너지·커널 융합 관련

| 파라미터 | 위치 | 현재값 | 왜 필요한가 | 우선순위 |
|---|---|---|---|---|
| `FIXED_COMBINED_REDUNDANCY_THRESHOLD` / `ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD`(다중공선성 컷오프) | `parameters.py` | 0.95 / 0.9 | "왜 하필 0.95냐"는 리뷰어 질문이 거의 확실 — 민감도 곡선 1개 필요 | **A** |
| `FIXED_MIN_PARTIAL_CORR`(시너지 그룹 구성 최소 편상관) | `parameters.py` | 0.02 | 그룹화 자체의 엄격도 | B |
| `ACTIVE_MAX_GROUP_SIZE`(시너지 그룹 최대 크기) | `parameters.py` | 4 | 압축 강도 vs 정보손실 트레이드오프 | B |
| `FIXED_KERNEL_ALPHA`(Ridge 정규화) / `FIXED_KERNEL_GAMMA`(RBF 폭) | `parameters.py` | 1.0 / None(sklearn 기본) | 커널 융합 품질 축 — n_components와 묶어서 표 하나로 | B |
| `FIXED_KERNEL_N_COMPONENTS`(Nystroem 랜드마크 수) | `parameters.py` | 100 | 근사 품질 vs 연산비용 | B |

## 모델 용량/구조 (기여 축과 독립 — "구조 선택에 안 종속적이다"를 보여주는 용도)

| 파라미터 | 위치 | 현재값 | 왜 필요한가 | 우선순위 |
|---|---|---|---|---|
| `regression_model`(헤드 종류) | `P1_MODEL_CONFIG["model"]` | mlp (mlp\|transformer\|i_transformer\|resnet_tab\|ft_transformer 중) | 게이팅+커널 구조가 특정 회귀 헤드에 종속되지 않음을 보여주는 표 1개 | **A**(표 1개면 충분) |
| `d_probe`/`d_head`/`mlp_hidden_dims` 등 용량 | `P1_MODEL_CONFIG["model"]` | 64/128/[128,64] | 과적합 민감도 | C |
| `dropout` | `P1_MODEL_CONFIG["model"]["dropout"]` | 0.2 | 과적합 민감도 | C |

## 파라미터는 아니지만 반드시 같이 보고해야 하는 것

- **multi-seed 분산**: `ACTIVE_SEED`/`ACTIVE_SPLIT_SEED` — 지금까지(`noscen` vs `scen` 비교 등) 전부 단일 시드(seed42)로만 비교함. Conclusion의 "한계" 항목에 이미 명시돼 있음(`docs/writing/PREWRITE_v1.md` §6) — 스윕 표가 아니라 "모든 핵심 비교 수치는 평균±표준편차로 보고" 원칙으로 별도 체크.

---

## 실행 메모

- 본문 Figure(A) 후보: n2 민감도, ref_lag/noise 강건성, l0_lambda 또는 scen_k 민감도, L0 스케줄 형태 비교, redundancy_threshold 민감도, regression_model 비종속성 — 총 6개 안팎. 실제로 몇 개를 넣을지는 최종 페이지 예산(`PREWRITE_v1.md` 부록 "페이지 예산 추적") 보고 조정.
- 부록 표(B) 후보는 한 표로 묶어서(파라미터·범위·성능 변화폭) 제시 — 개별 Figure로 늘리지 않음.
- `ACTIVE_WARMSTART_BRANCH_EPOCH`은 현재 세션에서 150→200으로 바꿔가며 noscen 대비 성능을 재검증 중(디커플링된 L0 스케줄 적용 후) — 결론 나면 이 문서의 B 항목 값/우선순위를 업데이트할 것.
