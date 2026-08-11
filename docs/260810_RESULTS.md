# 260808 — `q_frac_ref` 5변수 OFAT 실험 설계 (n2 / noise_amp / k / HI포함 / shape filter)

**2026-08-08 갱신**: §11의 다음 액션 1~3(가정 확정, §3.1/§3.2 구현)을 완료했다. §2의
가정 4개는 그대로 확정해 진행한다. **실행(액션 4)은 아직 하지 않았다** — 아래 §4~§8을
실제 `_4_data_hi` 데이터 상태 기준으로 스킵 가능한 Step을 표시한 skip-aware 명령어로
갱신했으니, 이 문서를 검토한 뒤 실행 지시하면 된다.

**2026-08-08 2차 갱신 — baseline noise_amp 5%→3%로 변경.** 기존 `q_frac_ref` 추출이
전부 `noise=3%` 기준이라(§3.3) 3%를 baseline으로 쓰면 baseline 자체·실험1의 n2=9%·
실험5의 shape 미적용까지 **3개 데이터 포인트를 그대로 재사용**할 수 있어 신규 Step 4가
6개 조합→4개로 줄어든다. 이론적 리크 상한(§7)도 노이즈가 작을수록 살짝 올라가지만
(3%→상한≈0.972, 5%→상한≈0.922) 정당 성능(0.9558)과의 간극이 여전히 ~1.6%p 남아
실험4의 판별력은 유지된다고 판단해 절충했다. 아래 §0~§8 전부 3% 기준으로 갱신함.

---

## 0. 설계 요청 원문 요약

- 고정 조건(전 실험 공통): `q_frac_ref` 축, `n1=35%`
- 변수 5개 — 각 실험은 **자신의 변수만 스윕하고 나머지 4개는 "다른 실험 중 고정값"으로 통일**:

| 변수 | 스윕 값 | 다른 실험 중 고정값 |
|---|---|---|
| n2 | 5%, 9%, 20% | **20%** |
| noise_amp | 1%, 3%, 5%, 10% | **3%**(2차 갱신, 원래 5% — 아래 참고) |
| k(`scen_k_count`) | 15, 25, 65 | **65** |
| HI(`stat_q_abs`/`stat_energy_seg`) | 포함 / 미포함 | **미포함** |
| shape filter(전처리 필터7) | 적용 / 미적용 | **적용** |

- 실험 1: n2 스윕 / 실험 2: noise_amp 스윕 / 실험 3: k 스윕 / 실험 4: HI 적용 테스트 /
  실험 5: shape filter 테스트

**2026-08-10 추가**: 원래 5변수 설계에는 없었지만, §14의 n2=5% 시나리오 소멸 진단 과정에서
`min_pts`(세그먼트 최소 포인트 수)가 실질적인 6번째 변수로 떠올라 **실험 6**을 추가한다
(§8-2). 원문 설계와 달리 이 변수는 **n2=20%(baseline)에서는 효과가 없다는 게 이미 확인돼
있어**(§14.2 — baseline은 min_pts=3~15 전 구간 100% 생존) 고정값을 n2=20%가 아니라
**n2=5%**로 둔다 — 이 변수만 원문 5개 변수와 고정 기준점이 다르다는 점에 주의.

---

## 1. 설계 원칙 — OFAT(One-Factor-At-a-Time), baseline 공유

위 "고정값" 5개를 동시에 적용한 지점을 **baseline**으로 한 번만 실행하고, 각 실험은
baseline에서 자기 변수 하나만 바꾼 지점들을 추가로 실행한다. baseline 자체가 각 실험의
스윕 값 중 하나(n2=20%, noise=**3%**, k=65, HI=미포함, shape=적용)와 정확히 겹치므로
**중복 실행 없이 재사용**한다(학습 run 기준 — Step 4 데이터 재사용은 §3.3 별도).

| 실험 | 스윕 값 | baseline과 겹치는 값(재사용) | 신규 학습 실행 필요 |
|---|---|---|---|
| 1. n2 | 5/9/20% | 20% | 5%, 9% (2건) |
| 2. noise_amp | 1/3/5/10% | 3% | 1%, 5%, 10% (3건) |
| 3. k | 15/25/65 | 65 | 15, 25 (2건) |
| 4. HI | 포함/미포함 | 미포함 | 포함 (1건) |
| 5. shape filter | 적용/미적용 | 적용 | 미적용 (1건) |
| 6. min_pts(§8-2, 2026-08-10 신설) | 6/10 | 10(단, n2=5% 고정 — 다른 실험들의 baseline인 n2=20%가 아니라 **실험1의 n2=5% 지점**을 재사용) | 6 (1건) |
| 7. raw 융합(§8-3, 2026-08-10 신설) | 미융합/방안1/신규(a) | 미융합(baseline) | 방안1, 신규(a) (2건) |
| 8. regression_model(§8-4, 2026-08-10 신설) | mlp/transformer/resnet_tab | mlp(baseline) | transformer, resnet_tab (2건) |
| 9. datasets 구성(§8-5, 2026-08-10 신설) | 풀링(MIT+HUST)/MIT-only/HUST-only | 풀링(baseline) | MIT-only, HUST-only (2건) |

**총 학습 실행 수 = baseline 1건 + 신규 16건 = 17건**(실험1~5는 원안 그대로 신규 9건,
2026-08-10에 실험6~9가 추가되며 신규 7건이 더해졌다 — §10.3의 17개 행과 정확히 일치).
실험6만 예외적으로 baseline(n2=20%)이 아니라 실험1의 n2=5% 지점을 자기 기준점으로
삼는다(§8-2에 이유 명시) — 나머지 8개 실험은 전부 공통 baseline 1개를 공유한다.

---

## 2. Baseline 정의

| 항목 | 값 | 비고 |
|---|---|---|
| `--seg-axis` | `q_frac_ref` | 고정(요청) |
| `--n1` | `0.35` | 고정(요청) |
| `--n2` | `0.20` | 실험1의 고정값 |
| `--noise-amp` | `0.03`(2차 갱신, 원래 0.05) | 실험2의 고정값 — 기존 추출 데이터 재사용 극대화 목적으로 변경(§3.3, 위 2차 갱신 메모) |
| `--scen-k` | `65` | 실험3의 고정값 |
| HI(`stat_q_abs`/`stat_energy_seg`) | **미포함** | 실험4의 고정값 — ⚠️ §3.1 참고 |
| shape filter | **적용**(기본, `--skip-shape` 없음) | 실험5의 고정값 |
| `--ref-lag` | `0` | **가정** — 5변수에 없어 값 미지정. `docs/SOC.md` §6 Phase4가 "초기 구현은 lag=0"으로 명시한 값을 그대로 사용. 다른 값을 원하면 지정 요망 |
| `--noise-mode` | `ou`(기본) | **가정** — 현재 코드 기본값 그대로 |
| `--noise-period` | `200`(기본) | **가정** — 현재 코드 기본값 그대로 |
| `--n-samples`(N) | `4` | **가정** — 5변수에 없어 값 미지정. `q_frac_wide` 계열 코드베이스 기본값 사용 |
| `--model-config` | `5_model/config/exp_qfw_mlp_basefix.yaml` | **가정** — 최근 0805/0806 비교에 쓰인 검증된 baseline config(mlp, `split_seed=42`, `lr=5e-4`, `epochs=500`). CLI `--seg-axis q_frac_ref`가 yaml의 `scenario.axis: q_frac_wide`를 덮어쓰므로 그대로 재사용 가능(`train_scr.py:435-442`에서 확인) |
| 데이터셋/평가 | MIT+HUST, oracle/hard/soft 표준 평가 | 기존 파이프라인 그대로 |

**가정 4개(ref_lag, noise_mode/period, n_samples, model-config)는 요청 원문에 없어 제가
정한 기본값이었고, 2026-08-08에 그대로 확정해 진행합니다.**

---

## 3. 구현 완료 사항 (2026-08-08)

### 3.1 [구현 완료] HI(`stat_q_abs`/`stat_energy_seg`) 포함/미포함 — 환경변수 토글

**구현 방식(단순 config 대신 환경변수를 선택한 이유)**: `N_HI`가 `cap_heads.py` 등
13개 파일에서 **모듈 임포트 시점 상수**로 `nn.Linear`/`HardConcreteGate` 크기를 정하는 데
쓰인다(예: `cap_heads.py:22 _HEAD_IN = N_HI+N_HI+1+1`). `train_scr.py`는 `hi_schema`를
파일 최상단에서 임포트하므로, `main()`의 argparse가 실행되는 시점엔 이미 늦다 — CLI
플래그로는 불가능하고, **프로세스 시작 전 환경변수**로 줘야 한다.

- `5_model/utils/hi_schema.py`: `EXCLUDE_STAT_LEAK = os.environ.get("SOH_EXCLUDE_STAT_LEAK", "0") == "1"`
  모듈 최상단에 추가, `get_hi_cols_for_seg`/`get_hi_cost_vector`의 `_STAT_EXCLUDE`가 이 값을
  따르도록 수정. `N_HI`는 그대로 66(기본)/64(`SOH_EXCLUDE_STAT_LEAK=1`)로 자동 계산됨.
- `5_model/datasets/segment_dataset.py`: `_get_native_hi_cols()`도 동일하게 수정(중복
  로직이라 둘 다 고쳐야 함, 지난 세션에 이미 확인된 패턴).
- `5_model/train_scr.py`: run 디렉터리명에 `_noleak` 접미사 추가(체크포인트 구조가
  N_HI=64/66으로 달라지므로 나중에 잘못 섞어 로드하는 사고 방지), 저장되는 `config.yaml`에
  `data.exclude_stat_leak` 기록(환경변수는 저장된 cfg만 봐선 안 보이므로 감사 추적용),
  실행 로그에 `[train] N_HI=.. (SOH_EXCLUDE_STAT_LEAK=..)` 출력.
- **단위 테스트 통과 확인**: `SOH_EXCLUDE_STAT_LEAK` 미지정 시 `N_HI=66`(포함, 기본 유지),
  `=1` 지정 시 `N_HI=64`(미포함) — `hi_schema.py`·`segment_dataset.py` 양쪽 다 정확히
  전환됨을 실제 임포트로 확인.
- **사용법**(모든 학습·평가 스크립트에 동일하게 앞에 붙임):
  ```
  SOH_EXCLUDE_STAT_LEAK=1 python run_pipeline.py 6 ...
  ```
  **주의**: 한 run으로 만든 체크포인트를 나중에(`test_scr.py --checkpoint ...`) 평가할 때도
  그때 썼던 환경변수 값을 그대로 다시 지정해야 한다(안 그러면 N_HI 불일치).
- 데이터 재추출은 불필요(기존 확인대로 pkl에는 항상 두 컬럼이 저장돼 있고 필터링은 로드
  시점에만 일어남).

### 3.2 [구현 완료] shape filter(`--skip-shape`) 학습 파이프라인 배선

- `run_pipeline.py`에 `--skip-shape` 인자 추가, Step 2/4/6/7/8에 전달(Step 2=전처리
  자체가 필터7 스킵, Step 4=`_noshape` 경로로 추출, Step 6/7/8=`_noshape` 경로로 학습).
- `train_scr.py`에 `--skip-shape` 인자 추가, `_axis_dir`에 hi_correlation.py와 동일한
  순서(`_ccOnly` 다음 `_noshape`)로 접미사 적용, 저장 cfg에 `data.skip_shape` 기록.
- `train_classifier.py`에도 `--skip-shape` 인자 추가(`_axis_dir_from_spec` 재구성 시
  `--exclude-cv`와 동일하게 spec.params에 없는 정보라 CLI로 별도 전달 필요).
- **`--help` 출력으로 3개 스크립트 전부 정상 파싱 확인**(run_pipeline.py/train_scr.py/
  train_classifier.py).
---

> **공통 표기**: HI=미포함이 고정값인 baseline·실험1·2·3·5는 학습 커맨드 앞에
> `SOH_EXCLUDE_STAT_LEAK=1`을 반드시 붙인다(§3.1, 안 붙이면 기본값인 "포함"으로 돈다).
> `--model-config`는 전부 `5_model/config/exp_qfw_mlp_basefix.yaml` 고정. Step 4는
> `--force`가 있어야 기존 캐시를 덮어쓴다(신규 조합은 캐시가 없으니 있으나 없으나 무방하되
> 명시).

## 4. 실험 1 — n2 스윕

고정: noise_amp=**3%**, k=65, HI=미포함, shape=적용, n1=35%. **9%/20%는 기존 데이터
재사용(§3.3), 5%만 N=4로 신규 추출 필요**(기존 n2=5% 폴더는 N=7이라 재사용 불가).

| n2 | Step 4 | Step 6 |
|---|---|---|
| 5% | `python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_ref --n1 0.35 --n2 0.05 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --force --workers 40` | `SOH_EXCLUDE_STAT_LEAK=1 python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.05 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16` |
| 9% | **스킵 — 기존 데이터 재사용** | `SOH_EXCLUDE_STAT_LEAK=1 python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.09 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16` |
| **20%(=baseline)** | **스킵 — 기존 데이터 재사용** | 아래 baseline 커맨드와 동일(공유) |

**baseline 커맨드(실험1·2·3·4·7의 "미포함" 쪽·5의 "적용" 쪽 전부 공유)**:
```
SOH_EXCLUDE_STAT_LEAK=1 python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16
```

---

## 5. 실험 2 — noise_amp 스윕

고정: n2=20%, k=65, HI=미포함, shape=적용, n1=35%. **3%(=baseline)만 재사용, 나머지
(1%/5%/10%)는 신규 추출 필요.**

| noise_amp | Step 4 | Step 6 |
|---|---|---|
| 1% | `python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.01 --noise-mode ou --noise-period 200 --n-samples 4 --force --workers 40` | `SOH_EXCLUDE_STAT_LEAK=1 python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.01 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16` |
| **3%(=baseline)** | 스킵 — 기존 데이터 재사용 | 위 §4의 baseline 커맨드와 동일(공유) |
| 5% | `python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.05 --noise-mode ou --noise-period 200 --n-samples 4 --force --workers 40` | `SOH_EXCLUDE_STAT_LEAK=1 python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.05 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16` |
| 10% | 위 1% 행과 동일, `--noise-amp 0.10` | 위 1% 행과 동일, `--noise-amp 0.10` |

---

## 6. 실험 3 — k(`scen_k_count`) 스윕

고정: n2=20%, noise_amp=3%, HI=미포함, shape=적용, n1=35%. **k는 학습 파라미터라 Step 4
불필요** — baseline 추출 데이터(기존 재사용, §3.3)를 그대로 쓰고 Step 6만 돈다.

| k | Step 4 | Step 6 |
|---|---|---|
| 15 | 스킵(baseline 데이터 재사용) | `SOH_EXCLUDE_STAT_LEAK=1 python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 15 --workers 16` |
| 25 | 스킵 | 위와 동일, `--scen-k 25` |
| **65(=baseline)** | — | 위 §4의 baseline 커맨드와 동일(공유) |

---

## 7. 실험 4 — HI(`stat_q_abs`/`stat_energy_seg`) 포함 테스트

고정: n2=20%, noise_amp=3%, k=65, shape=적용, n1=35%. **Step 4 불필요**(HI 포함 여부는
학습 시점 필터링, §3.1). baseline("미포함")은 `SOH_EXCLUDE_STAT_LEAK=1`, "포함"은 환경변수
없이(기본값) 실행한다 — 둘은 **N_HI가 64/66으로 달라 서로 다른 구조의 체크포인트**가 되고
run 디렉터리명도 `_noleak` 접미사로 구분된다(§3.1).

| HI | Step 4 | Step 6 |
|---|---|---|
| **미포함(=baseline)** | 스킵 | 위 §4의 baseline 커맨드와 동일(공유) |
| 포함 | 스킵 | `python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16` (환경변수 없음 = 기본 포함) |

**참고(2차 갱신)**: `docs/SOC.md` §6 계산 방식(`combined_std≈noise_amp×0.382`, 타겟
std=6.85%p)으로 `noise_amp=3%`의 이론적 리크 상한을 다시 구하면 σ≈1.15%p →
`1-(1.15/6.85)²≈0.972`. 정당 성능(0.9558)과의 간극은 ~1.6%p — "포함" 쪽이 이 방향으로
튀는지가 리크 재발의 실증 근거가 된다(5%였을 때 상한 0.922보다 간극이 좁아져 판별력이
약간 떨어지지만, 여전히 관측 가능한 수준으로 판단해 절충함 — 위 2026-08-08 2차 갱신
메모 참고).

---

## 8. 실험 5 — shape filter(전처리 필터7) 테스트

고정: n2=20%, noise_amp=3%, k=65, HI=미포함, n1=35%. **Step 2(§3.2에서 이미 완료 확인)와
Step 4(§3.3, noshape 데이터도 이미 존재) 둘 다 스킵** — 이 실험은 Step 6만 돌리면 된다.

| shape filter | Step 2 | Step 4 | Step 6 |
|---|---|---|---|
| **적용(=baseline)** | — | 스킵(기존 재사용) | 위 §4의 baseline 커맨드와 동일(공유) |
| 미적용 | 스킵 — `_4_data_hi/clean_noshape/` 이미 완료 | 스킵 — `n1-35%_n2-20%_N-4_lag-0_noise-3%_ou-200_noshape` 이미 존재 | `SOH_EXCLUDE_STAT_LEAK=1 python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --skip-shape --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16` |

---

## 8-2. 실험 6 — min_pts 스윕 (2026-08-10 신설)

**§0 참고**: 이 실험만 고정값 기준점이 다르다 — n2=**5%**(다른 실험들의 n2=20%가 아님).
이유: baseline(n2=20%)에서는 `min_pts`를 3~15 어떤 값으로 둬도 전 존 100% 생존이라
(§14.2에서 이미 실측 확인) min_pts 변화가 관측되지 않는다. n2=5%일 때만 `chg_lo`/
`chg_mid`/`dis_hi` 3개 존이 min_pts=10에서 100% 탈락하는 현상(§14.1)이 있어, 그 지점에서
비교해야 효과가 보인다.

고정: **n2=5%**, noise_amp=3%, k=65, HI=미포함, shape=적용, n1=35%(다른 실험들과 동일한
표준 고정값).

| min_pts | Step 4 | Step 6 | 비고 |
|---|---|---|---|
| **10** | 스킵 — 기존 데이터 재사용(`n1-35%_n2-5%_N-4_lag-0_noise-3%_ou-200`) | 스킵 — 이미 실행 완료(`0809_1229_p2_mlp_q_fr_35%_5%_noleak`, §14.1에서 시나리오 소멸을 처음 발견한 그 run) | §10.3 표의 **2번 행**과 동일 데이터 — 실험1의 n2=5% 지점을 그대로 재사용 |
| 6 | `n1-35%_n2-5%_N-4_minpts6_lag-0_noise-3%_ou-200`(§14.4 접미사 규칙, 기존 데이터와 안 겹침 확인됨) | 실행 필요 | §10.3 표의 **11번 행** |

**min_pts=6 커맨드**(§14.5와 동일, Step4부터 한 번에):
```powershell
$env:SOH_EXCLUDE_STAT_LEAK="1"
python run_pipeline.py 4 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.05 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --min-pts 6 --scen-k 65 --workers 40
```

**비교 판정 기준**: 전체 R²/MAPE 개선폭보다 **존별(zone-wise) 회복 여부**가 핵심이다.
min_pts=10 쪽은 `chg_lo`/`chg_mid`/`dis_hi` 3개 존이 HUST 77/77셀에서 사실상 예측 자체가
없다(hard 라우팅 시 분류기가 그 클래스를 아예 고르지 않음 — `capacity_curve_1-7.png`의
`Pred-mid`/`Pred-hi` 누락이 그 증거). min_pts=6에서 이 3개 존의 `capacity_curve_*.png`
(HUST 대표 셀 우선 확인, 예: `1-7`)에 `Pred-mid`/`Pred-hi` 라인이 실제로 나타나는지,
그리고 그 존들의 oracle 존별 R²/RMSE가 0(또는 결측)에서 유의미한 값으로 바뀌는지를
1차 판정 기준으로 삼는다.

---

## 8-3. 실험 7 — raw CNN 융합(방안1 / 신규(a)) (2026-08-10 신설)

`docs/260804_RESULTS.md` §2.1의 raw 융합 4방식 종합 랭킹(방안1 1위, 신규(a) 2위,
구(b) 3위, 구(a) 4위) 중 상위 2개만 이 프로젝트의 현재 q_frac_ref baseline(n2=20%,
noise=3%, k=65, HI=미포함, shape=적용, min_pts=10)에 적용한다. 구(a)/구(b)는 제외.

**핵심 절약 포인트**: raw 융합은 Phase 2(회귀) 전용 변경이라 Phase 1(게이트)·분류기
학습은 baseline과 완전히 동일하게 재사용 가능하다(`docs/260804_RESULTS.md` §0 "Phase 1은
`x_hi`만 쓰므로 이번 CNN 재설계와 무관해 재사용해도 안전" 그대로 적용). baseline의 Phase1
run(`0809_1010_p1_q_fr_35%_20%_noleak`)이 이미 `classifier.type: cnn`으로 학습돼
`classifier/clf_best.pt`에 RawCNN이 들어있으므로, 신규(a)도 **분류기 재학습 없이** 그
체크포인트를 얼려서 그대로 재사용한다. `run_pipeline.py 8`(Step 8=Phase2부터 시작)로
Step 4~7을 전부 건너뛴다.

고정: n1=35%, n2=20%, n_samples=4, ref_lag=0, noise_amp=3%, noise_mode=ou,
noise_period=200, scen_k=65, HI=미포함(exclude_stat_leak), shape=적용, min_pts=10 —
baseline(1번 행)과 완전히 동일. **regression_model은 mlp로 고정**(모델 종류를 바꾸는
비교는 실험 8에서 별도로 함) — 이번 실험의 유일한 변수는 raw 융합 방식이다.

| 방식 | model-config | 추가 플래그 | 비고 |
|---|---|---|---|
| 방안1(flatten) | `5_model/config/exp_qfw_mlp_flat.yaml` | (없음, yaml에 `with_raw_flat: true` 내장) | `RAW_CH`가 260804 당시 2채널(`V,\|I\|`)에서 지금은 3채널(`V, signed I, t_rel`)로 바뀌어 있어(§14.3 참고), flatten 차원도 96D가 아니라 144D(=48×3)로 자동 변경됨 — 260804의 "1위" 수치와 완전히 같은 조건 재현은 아니고 "현재 baseline 데이터에 방안1 방식을 적용"하는 것 |
| 신규(a)(RawCNN, frozen) | `5_model/config/exp_qfw_mlp_basefix.yaml` | `--with-raw-cnn` | `run_pipeline.py`가 `--gates-from`으로 지정한 baseline classifier의 `clf_best.pt`를 자동으로 찾아 `--raw-cnn-pretrained-from`에 연결(run_pipeline.py:401-412) — 수동 지정 불필요 |

**실행 명령(PowerShell)**:
```powershell
$env:SOH_EXCLUDE_STAT_LEAK="1"

# 방안1
python run_pipeline.py 8 --model-config 5_model/config/exp_qfw_mlp_flat.yaml --gates-from "_5_data_model_scr/0809_1010_p1_q_fr_35%_20%_noleak" --seg-axis q_frac_ref --n1 0.35 --n2 0.2 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 40

# 신규(a)
python run_pipeline.py 8 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --gates-from "_5_data_model_scr/0809_1010_p1_q_fr_35%_20%_noleak" --seg-axis q_frac_ref --n1 0.35 --n2 0.2 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --with-raw-cnn --workers 40
```

**비교 판정 기준**: baseline(raw 미융합, R²=0.9524)과 R²/RMSE/MAPE 3지표로 직접 비교.
`docs/260804_RESULTS.md`의 결론(raw 융합이 "공정하게 통제된 baseline"을 확실히 이기는
조합은 드물었음, qfw+mlp 조합 자체는 baseline이 방안1·신규(a) 모두를 이겼음 — §1.3의
qfw+mlp 행 참고)이 이 프로젝트의 q_frac_ref 축·현재 baseline 조건에서도 재현되는지가
핵심 질문이다.

## 8-4. 실험 8 — 회귀 모델 교체(transformer / resnet_tab) (2026-08-10 신설)

`regression_model`만 `mlp`→`transformer`/`resnet_tab`으로 바꾸고 나머지는 baseline과
전부 동일하게 고정한다(raw 융합 없음 — 실험 7과는 독립된 변수). **게이트는 회귀 모델별로
새로 학습해야 한다** — `docs/260804_RESULTS.md`(exp_qfw_transformer_basefix.yaml vs
exp_qfw_mlp_basefix.yaml)의 `gates_from`이 모델마다 다른 별도 run을 가리키는 데서 확인되듯,
Phase 1의 HardConcrete 게이트 학습이 회귀 헤드를 통과하는 gradient로 이뤄져 모델 종류에
따라 결과가 달라지기 때문이다(raw 융합과 달리 재사용 불가). `run_pipeline.py 6`(Step
6=Phase1부터)으로 Step4(추출)만 기존 baseline 데이터를 그대로 재사용하고 Step6~9를 새로
돌린다.

고정: n1=35%, n2=20%, n_samples=4, ref_lag=0, noise_amp=3%, noise_mode=ou,
noise_period=200, scen_k=65, HI=미포함(exclude_stat_leak), shape=적용, min_pts=10,
raw 융합 없음 — baseline(1번 행)과 완전히 동일. 유일한 변수는 `regression_model`.

**실행 명령(PowerShell)**:
```powershell
$env:SOH_EXCLUDE_STAT_LEAK="1"

# transformer
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_transformer_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.2 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 40

# resnet_tab
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_resnet_tab_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.2 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 40
```

두 yaml 모두 `exp_qfw_mlp_basefix.yaml`과 `regression_model`/`gates_from`(무시됨, 새
Phase1이 자동 대체)만 다르고 나머지 설정은 동일함을 diff로 확인했다(`lr=5.0e-4`,
`scen_k_count`는 CLI `--scen-k 65`로 yaml의 15를 덮어씀).

**비교 판정 기준**: baseline(mlp, R²=0.9524)과 R²/RMSE/MAPE 3지표로 직접 비교.
`docs/260804_RESULTS.md` §1.3(qfw축, k=15 통제 조건)에서는 baseline 대비
transformer가 R² 근소 우위(0.9606 vs 0.9463 — 단 이건 mlp baseline과의 비교이며
"통제된 transformer baseline" 자체 R²=0.9606였음), resnet_tab이 근소 열위였다 — 이
프로젝트의 q_frac_ref·k=65 조건에서도 같은 순서(transformer ≥ mlp ≥ resnet_tab 또는
유사)가 나오는지 확인하는 것이 이 실험의 목적이다.

---

## 8-5. 실험 9 — MIT-only / HUST-only 단일 데이터셋 학습·평가 (2026-08-10 신설)

### 배경 — MMD 없는 MIT+HUST 풀링이 실제로 문제인지 검증

`docs/Generalized foundation model for lithium-ion battery state-of-health prediction with
distribution metric learning.pdf`(Gong 외, *J. Energy Storage* 150:120566, 2026) 검토
과정에서, 이 논문은 MIT+HUST류의 이종 데이터셋을 섞을 때 **MMD(Maximum Mean Discrepancy)로
피처 분포를 정렬**한다 — 정렬 없이 그냥 풀링하는 baseline은 아예 비교 대상에 넣지도 않는다.
이 프로젝트는 지금 MMD 없이 MIT+HUST를 그냥 풀링해서 학습하고 있어, "이게 실제로 성능을
깎아먹는 negative transfer인가"를 직접 검증할 필요가 제기됐다.

| | 전체(공식, §10.3 1번 행) | MIT만(20셀, 직접 집계) | HUST만(20셀, 직접 집계) |
|---|---|---|---|
| R² | 0.9524 | 0.9099 | 0.9508 |
| RMSE | 1.520% | 1.315% | 1.609% |
| MAE | 1.133% | 0.986% | 1.204% |
| MAPE | 1.251% | 1.030% | 1.357% |

파국적인 격차(예: 논문 Table 2의 target-only 대비 4~6배 MAE 차이)는 없다 — 다만 이건
**"풀링된 모델이 각 데이터셋에서 얼마나 잘하는가"**만 보여줄 뿐, **"MIT만/HUST만 따로
학습했으면 더 잘했을까(=풀링이 오히려 해가 됐는가)"**는 별도 실험 없이는 알 수 없다.
실험 9는 바로 이 비교를 위한 것이다.

### 설계

MIT-only, HUST-only 두 케이스를 baseline과 동일한 축·전처리 조건에서 각각 처음부터
학습·평가한다. 유일한 변수는 `data.datasets`(풀링 여부) — 나머지는 baseline과 완전히
고정한다: n1=35%, n2=20%, n_samples=4, ref_lag=0, noise_amp=3%, noise_mode=ou,
noise_period=200, scen_k=65, HI=미포함(exclude_stat_leak), shape=적용, min_pts=10,
regression_model=mlp, raw 융합 없음.

`mit_only.yaml`/`hust_only.yaml` 둘 다에 baseline의 실제 test 셀 40개(MIT 20+HUST 20)를
그대로 넣어뒀다 — 각 run은 자기 데이터셋에 있는 셀만 교집합으로 자동 필터링되므로 파일을
따로 나눌 필요가 없다. **실제 코드 로직을 그대로 재현해 검증 완료**: MIT-only 테스트셋 =
baseline MIT 테스트 20셀과 정확히 일치, HUST-only 테스트셋 = baseline HUST 테스트 20셀과
정확히 일치, 두 경우 다 train/val로 새는 셀 없음, `forced_test_cells` 없는 기존 방식은
원래의 40셀 테스트셋을 그대로 재현(영향 없음) — 4가지 전부 확인됨. 이제 세 run(풀링
baseline/MIT-only/HUST-only) 모두 완전히 동일한 테스트 셀에 대한 짝지어진(paired)
비교가 가능하다.

**실행 명령**:
```powershell
$env:SOH_EXCLUDE_STAT_LEAK="1"

# MIT-only
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_mit_only.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.2 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 40

# HUST-only
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_hust_only.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.2 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 40
```

Step 4(추출) 재실행은 불필요 — `data.datasets`는 이미 추출된 `_4_data_hi/q_frac_ref/...` pkl
중 어느 셀 파일을 로드할지만 필터링하는 데이터-로딩 단계 설정이라, 기존 baseline 추출
데이터를 그대로 재사용할 수 있다(추출 자체는 MIT/HUST 둘 다 이미 포함돼 있음).

### 비교 판정 기준

| 비교 | 의미 |
|---|---|
| MIT-only R²/RMSE vs 위 표의 "MIT만(직접 집계, 풀링 모델)" | MIT-only가 더 좋으면 → 풀링이 MIT을 깎아먹음(negative transfer) / 더 나쁘면 → 풀링이 MIT에 도움 |
| HUST-only R²/RMSE vs 위 표의 "HUST만(직접 집계, 풀링 모델)" | 동일 논리를 HUST에 적용 |

두 비교 다 "풀링이 도움"으로 나오면 MMD 없는 현재 방식을 그대로 유지해도 근거가 충분하고,
한쪽이라도 "풀링이 해"로 나오면 MMD류 도메인 정렬 도입을 §10.5(게재 가능성 평가)에
개선 과제로 추가해야 한다.

§10.3에 결과 기록용 행(16, 17번)을 미리 만들어둔다.

### 실행 후 발견 — `forced_test_cells`를 쓰면 train/val/test 비율이 6:2:2로 유지되지 않는다 (2026-08-11)

`build_datasets()`의 `forced_test_cells` 분기(§5-2 참고)는 지정된 40셀(MIT 20+HUST 20)을
**먼저 test로 고정**해두고, **남은 셀만** `train_ratio:val_ratio`(=0.6:0.2 → 상대비 3:1)로
재분배한다 — 이 때문에 다음 두 가지가 원래의 6:2:2 설계에서 벗어난다:

| | 전체 셀 | test | train | val | 비율(train:val:test) |
|---|---|---|---|---|---|
| 풀링 baseline(강제 없음) | 200 | 40(20.0%) | 120(60.0%) | 40(20.0%) | 정확히 6:2:2 |
| MIT-only | 123 | 20(**16.3%**) | 77(62.6%) | 25(20.3%) | 62.6:20.3:16.3 |
| HUST-only | 77 | 20(**26.0%**) | 42(54.5%) | 14(18.2%) | 54.5:18.2:26.0 |

1. **test 비율이 데이터셋 크기에 좌우된다** — 40셀을 절대 개수로 고정하다 보니, 셀이 많은
   MIT는 test 비중이 20%보다 작아지고(16.3%), 셀이 적은 HUST는 커진다(26.0%). train:val은
   **서로 간의 상대 비율(3:1)만** 유지되고, 전체 대비 절대 비율은 아니다.
2. **반올림으로 데이터셋당 셀 1개가 어디에도 안 들어간다.** `remaining`(강제 test 제외 후
   남은 셀)을 `int()`로 자르는 과정에서 MIT는 103개 중 77+25=102(1개 누락), HUST는 57개
   중 42+14=56(1개 누락) — `split_cells()`가 내부적으로 반환하는 세 번째 값(나머지)을
   강제-test 분기가 버리기 때문이다. 전체의 <1.5% 수준이라 결과에 미치는 영향은 미미하지만,
   의도된 설계는 아니고 실측으로 확인된 코드 동작이다(수정하지 않고 이 상태로 실험 9를
   진행·기록함).

---

## 9. 비교 통제 재확인 (`docs/SOC.md` §5 Phase5 baseline 통제 원칙 준용)

## 10. 실행 명령 전체 목록 + 결과 기록 템플릿

**실행 순서: Step 4(선행 추출, 4건) 먼저 → Step 6(학습, 10건) → 아래 표에 결과 채우기.**
Step 6은 서로 다른 GPU 학습이라 순차 실행 권장. Step 4는 각 커맨드가 이미 `--workers 40`
(48코어 중 40개, §12.2 실측 기반)을 쓰므로 **4건을 동시에 여러 개 띄우지 말고 하나씩
순차 실행**할 것 — 두 개를 동시에 돌리면 80워커가 48코어를 두고 경합해 오히려 느려진다.

### 10.1 Step 4 — 선행 추출 (4건, baseline·n2=9%·shape미적용은 기존 데이터라 불필요)

```
# [P1] 실험1 n2=5%
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_ref --n1 0.35 --n2 0.05 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --force --workers 40

# [P2] 실험2 noise=1%
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.01 --noise-mode ou --noise-period 200 --n-samples 4 --force --workers 40

# [P3] 실험2 noise=5%
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.05 --noise-mode ou --noise-period 200 --n-samples 4 --force --workers 40

# [P4] 실험2 noise=10%
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.10 --noise-mode ou --noise-period 200 --n-samples 4 --force --workers 40
```

### 10.2 Step 6 — 학습 10건 (Phase1→분류기→Phase2→평가 자동 체이닝)

**PowerShell 전용 문법**(사용자 환경 기본 셸). PowerShell은 `VAR=값 명령` 같은 bash식
인라인 환경변수 대입을 지원하지 않으므로 `$env:SOH_EXCLUDE_STAT_LEAK=...`를 별도 줄로
먼저 실행해야 한다 — 한 번 설정하면 값을 바꾸기 전까지 그 세션의 이후 모든 명령에 계속
적용되므로, 값이 바뀌는 지점([9] 앞, [10] 앞)에만 다시 설정하면 된다.

```powershell
$env:SOH_EXCLUDE_STAT_LEAK="1"

# [1] baseline (n2=20%, noise=3%, k=65, HI=미포함, shape=적용)
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16

# [2] 실험1 n2=5%  (P1 먼저 완료돼 있어야 함)
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.05 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16

# [3] 실험1 n2=9%  (기존 데이터 재사용, 선행 추출 불필요)
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.09 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16

# [4] 실험2 noise=1%  (P2 먼저 완료돼 있어야 함)
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.01 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16

# [5] 실험2 noise=5%  (P3 먼저 완료돼 있어야 함)
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.05 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16

# [6] 실험2 noise=10%  (P4 먼저 완료돼 있어야 함)
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.10 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16

# [7] 실험3 k=15  (기존 데이터 재사용)
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 15 --workers 16

# [8] 실험3 k=25  (기존 데이터 재사용)
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 25 --workers 16

# [9] 실험4 HI=포함  (기존 데이터 재사용, 환경변수 반드시 "0"으로)
$env:SOH_EXCLUDE_STAT_LEAK="0"
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16

# [10] 실험5 shape=미적용  (기존 noshape 데이터 재사용, --skip-shape 필수, 환경변수 다시 "1")
$env:SOH_EXCLUDE_STAT_LEAK="1"
python run_pipeline.py 6 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --skip-shape --n1 0.35 --n2 0.20 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --scen-k 65 --workers 16
```

**주의**: [1]~[8], [10]은 `SOH_EXCLUDE_STAT_LEAK="1"`이어야 HI=미포함이 맞는다([9]만
예외 — 실험4의 "포함" 지점). 하나라도 빠뜨리면 그 run만 조용히 N_HI=66(포함)으로 돌아
§3.1에서 경고한 confound가 된다. 위 블록 전체를 PowerShell 터미널에 그대로 붙여넣으면
`python` 줄은 블로킹 실행이라 하나 끝나야 다음 줄로 넘어가므로 자동으로 순차 실행된다.

### 10.3 결과 기록 (실행 후 채울 것)

| # | 실험 | 변수값 | 결과 run 디렉터리 | R²(oracle) | RMSE | MAPE |
|---|---|---|---|---|---|---|
| 1 | baseline | n2=20%,noise=3%,k=65,HI=미포함,shape=적용 | `0809_1118_p2_mlp_q_fr_35%_20%_noleak` | 0.9524 | 1.520%p | 1.251% |
| 2 | 1 / 6(min_pts=10) | n2=5%(§8-2 실험6의 min_pts=10 지점과 동일 데이터) | `0809_1229_p2_mlp_q_fr_35%_5%_noleak` | **0.7646** | **3.198%p** | **2.124%** |
| 3 | 1 | n2=9% | `0809_1342_p2_mlp_q_fr_35%_9%_noleak` | 0.8853 | 2.359%p | 1.869% |
| 4 | 2 | noise=1% | `0809_1441_p2_mlp_q_fr_35%_20%_noleak` | 0.9652 | 1.300%p | 1.079% |
| 5 | 2 | noise=5% | `0809_1620_p2_mlp_q_fr_35%_20%_noleak` | 0.9429 | 1.665%p | 1.284% |
| 6 | 2 | noise=10% | `0809_1718_p2_mlp_q_fr_35%_20%_noleak` | 0.9014 | 2.187%p | 1.784% |
| 7 | 3 | k=15 | `0809_1821_p2_mlp_q_fr_35%_20%_noleak` | 0.9475 | 1.596%p | 1.329% |
| 8 | 3 | k=25 | `0809_1932_p2_mlp_q_fr_35%_20%_noleak` | 0.9493 | 1.569%p | 1.280% |
| 9 | 4 | HI=포함 | `0809_2037_p2_mlp_q_fr_35%_20%` | 0.9599 | 1.394%p | 1.148% |
| 10 | 5 | shape=미적용 | `0809_2153_p2_mlp_q_fr_35%_20%_noleak` | 0.9431 | 1.664%p | 1.415% |
| 11 | 6 | n2=5%, min_pts=6(§8-2, 나머지는 표준 고정값과 동일) | `0810_1232_p2_mlp_q_fr_35%_5%_noleak` | 0.8159 | 2.989%p | 2.278% |
| 12 | 7 | raw 융합 방안1(flatten, §8-3) | `0810_1435_p2_mlp_q_fr_35%_20%_noleak` | 0.9583 | 1.423%p | 1.158% |
| 13 | 7 | raw 융합 신규(a)(RawCNN frozen, §8-3) | `0810_1500_p2_mlp_q_fr_35%_20%_noleak` | 0.9506 | 1.548%p | 1.197% |
| 14 | 8 | regression_model=transformer(§8-4) | `0810_1615_p2_tr_q_fr_35%_20%_noleak` | 0.9611 | 1.373%p | 1.047% |
| 15 | 8 | regression_model=resnet_tab(§8-4) | `0810_1735_p2_res_q_fr_35%_20%_noleak` | 0.9571 | 1.442%p | 1.110% |
| 16 | 9 | MIT-only 학습·평가(§8-5) | `0810_1842_p2_mlp_q_fr_35%_20%_noleak` | **-0.2038** | 4.809%p | 4.191% |
| 17 | 9 | HUST-only 학습·평가(§8-5) | `0810_1913_p2_mlp_q_fr_35%_20%_noleak` | 0.9477 | 1.658%p | 1.420% |

**표 채우면서 확인한 것(2026-08-10)**: 각 폴더명만으론 `noise_amp`/`scen_k`가 구분 안 돼서
(예: 실험2·3·5의 결과 폴더가 전부 `q_fr_35%_20%_noleak`로 이름이 같음) `config.yaml`을
직접 읽어 `n2`/`noise_amp`/`scen_k`/`skip_shape`/`exclude_stat_leak` 값으로 각 run을
확정 매칭했다(추측 아님). 9번(HI=포함) run만 `config.yaml`의 `exclude_stat_leak=False`로
정확히 구분되고, 10번(shape=미적용) run만 `seg_data_dir`에 `_noshape`가 붙어 구분된다.
11번(min_pts=6)도 `config.yaml`에서 `min_pts=6`, `seg_data_dir`에 `_minpts6` 접미사가
붙어 있는 걸 확인했다.

**실험1(§4) 결과 — n2 스윕 (2026-08-10)**:

| n2 | R²(oracle) | RMSE | MAPE |
|---|---|---|---|
| 20%(baseline, 1번 행) | 0.9524 | 1.520%p | 1.251% |
| 9%(3번 행) | 0.8853(-0.0671) | 2.359%p | 1.869% |
| 5%(2번 행) | 0.7646(-0.1207) | 3.198%p | 2.124% |

세그먼트가 짧아질수록 R²가 단조 감소하고, **20%→9%보다 9%→5% 구간의 하락폭이 더
가파르다**(-0.067 vs -0.121) — 선형이 아니라 가속적으로 나빠진다. §14에서 나중에
밝혀졌듯 n2=5%에서는 `chg_lo`/`chg_mid`/`dis_hi` 존이 `min_pts` 미달로 통째로
사라지는 문제가 겹쳐 있어(§8-2/§14.1), 이 가속 하락의 상당 부분은 순수 "정보량 감소"가
아니라 **존 소멸이라는 별도 병목**이 섞인 결과로 봐야 한다(실험6이 이 중 일부만 완화).

**실험2(§5) 결과 — noise_amp 스윕 (2026-08-10)**:

| noise_amp | R²(oracle) | RMSE | MAPE |
|---|---|---|---|
| 1%(4번 행) | 0.9652(+0.0128) | 1.300%p | 1.079% |
| 3%(baseline, 1번 행) | 0.9524 | 1.520%p | 1.251% |
| 5%(5번 행) | 0.9429(-0.0095) | 1.665%p | 1.284% |
| 10%(6번 행) | 0.9014(-0.0510) | 2.187%p | 1.784% |

노이즈가 커질수록 단조 하락하지만, **10%(업계 "Max Error" 트리거인 8%에 근접한
값, `docs/SOC.md` §7 참고)에서도 R²=0.9014로 0.90선을 지킨다** — 레퍼런스 노이즈에
대한 내성 자체는 나쁘지 않다는 뜻이다. 다만 하락 기울기는 noise_amp가 커질수록
가팔라진다(1%→3% 구간 -0.013, 5%→10% 구간 -0.051) — 노이즈가 특정 임계 이상으로
커지면 손해가 비선형으로 커질 수 있음을 시사한다.

**실험3(§6) 결과 — k(scen_k_count) 스윕 (2026-08-10)**:

| k | R²(oracle) | RMSE | MAPE |
|---|---|---|---|
| 15(7번 행) | 0.9475(-0.0049) | 1.596%p | 1.329% |
| 25(8번 행) | 0.9493(-0.0031) | 1.569%p | 1.280% |
| 65(baseline, 1번 행) | 0.9524 | 1.520%p | 1.251% |

k=65 > k=25 > k=15로 순서는 baseline이 최선이지만, **격차 자체가 n2·noise 스윕
대비 훨씬 작다**(R² 범위 0.9475~0.9524, 폭 0.005) — 시나리오별로 허용하는 HI 개수
상한(15~65)은 이 구간 안에서는 성능에 상대적으로 덜 민감한 하이퍼파라미터로 보인다.
Phase 1의 L0 게이트가 k 상한을 다 채우지 않고도 비슷한 효과를 내는 HI 부분집합을
찾아내고 있을 가능성이 있다(§8-4의 routing_table.csv로 실제 선택 개수를 확인하면
더 명확해질 것).

**실험4(§7) 결과 — HI(`stat_q_abs`/`stat_energy_seg`) 포함 여부 (2026-08-10)**:

| HI 포함 여부 | R²(oracle) | RMSE | MAPE |
|---|---|---|---|
| 미포함(baseline, 1번 행) | 0.9524 | 1.520%p | 1.251% |
| 포함(9번 행) | **0.9599(+0.0075)** | 1.394%p | 1.148% |

포함 쪽이 3지표 전부 개선된다 — `docs/SOC.md`의 leak-ceiling 계산(noise_amp=3%
기준 이론 상한 R²≈0.95~0.96대)과 맞아떨어지는 수준의 개선폭이라, **"완전히 라벨을
역산하는 수준의 심각한 누수"는 아니지만 어느 정도의 지름길 효과는 있다**는 이전
결론(§4.1.1 leak-ceiling 논의)과 일치한다. 이게 baseline에서 `SOH_EXCLUDE_STAT_LEAK=1`로
이 두 HI를 기본 제외해두는 근거다 — 성능은 살짝 손해 보더라도 방법론적으로 더
안전한 쪽을 baseline으로 삼은 것.

**실험5(§8) 결과 — shape filter(전처리 필터7) 적용 여부 (2026-08-10)**:

| shape filter | R²(oracle) | RMSE | MAPE |
|---|---|---|---|
| 적용(baseline, 1번 행) | 0.9524 | 1.520%p | 1.251% |
| 미적용(10번 행) | 0.9431(-0.0093) | 1.664%p | 1.415% |

필터7(형상 이상치 제거)을 빼면 3지표 전부 악화된다 — 이 필터가 단순히 데이터를
줄이기만 하는 게 아니라 **실제로 학습에 해로운 이상 사이클을 걸러내는 순기능을
하고 있음**을 확인시켜준다. baseline이 이 필터를 기본 적용 상태로 유지하는 게
타당하다는 근거.

**실험6(§8-2) 결과 — min_pts=10 vs 6, n2=5% 고정 (2026-08-10)**:

| min_pts | R²(oracle) | RMSE | MAPE |
|---|---|---|---|
| 10(2번 행) | 0.7646 | 3.198%p | 2.124% |
| 6(11번 행) | **0.8159**(+0.0513) | **2.989%p**(개선) | 2.278%(**악화**) |

R²·RMSE는 개선됐지만(§14.2에서 실측한 대로 죽어있던 존이 일부 살아난 결과로 보임)
MAPE는 오히려 소폭 나빠졌다 — 그리고 무엇보다 baseline(n2=20%, R²=0.9524)과의 격차는
여전히 크다(0.8159 vs 0.9524). min_pts=6이 n2=5%의 시나리오 소멸 문제를 **완전히
해결하진 못했고 부분적으로만 완화**한 것으로 보인다. §8-2에서 계획한 대로
`capacity_curve_*.png`(HUST `1-7` 등)에서 `Pred-mid`/`Pred-hi`가 실제로 나타나는지,
그리고 zone별 R²/RMSE가 얼마나 회복됐는지 다음으로 확인이 필요하다.

**실험7(§8-3) 결과 — raw 융합, baseline(R²=0.9524) 대비 (2026-08-10)**:

| 방식 | R²(oracle) | RMSE | MAPE | baseline 대비 |
|---|---|---|---|---|
| 방안1(flatten, 12번 행) | **0.9583** | **1.423%p** | 1.158% | **개선**(R²+0.0059, RMSE-0.097%p) |
| 신규(a)(RawCNN frozen, 13번 행) | 0.9506 | 1.548%p | 1.197% | 근소 열세(R²-0.0018) |

`docs/260804_RESULTS.md`(qfw축, k=15 통제)에서는 baseline이 방안1·신규(a) 둘 다 이겼는데,
이 프로젝트의 q_frac_ref·k=65 조건에서는 **방안1이 baseline을 넘어선다** — 축·k값이
다르면 raw 융합의 유불리가 뒤집힐 수 있음을 보여준다. 신규(a)는 이번에도 baseline에
근소하게 못 미쳐(§14.3 RAW_CH 3채널 재설계 이후 기준) 260804의 경향과 방향은 같다.

**실험8(§8-4) 결과 — 회귀 모델 교체, baseline(mlp, R²=0.9524) 대비 (2026-08-10)**:

| regression_model | R²(oracle) | RMSE | MAPE | baseline 대비 |
|---|---|---|---|---|
| transformer(14번 행) | **0.9611** | **1.373%p** | **1.047%** | **개선**(전 지표) |
| resnet_tab(15번 행) | 0.9571 | 1.442%p | 1.110% | **개선**(전 지표) |

두 모델 다 baseline(mlp)을 3지표 전부에서 이긴다 — `docs/260804_RESULTS.md` §1.3(qfw축,
k=15)에서 transformer가 mlp baseline보다 근소 우위였던 경향과 방향이 같고, 이번엔
resnet_tab도 함께 우위로 나왔다. q_frac_ref·k=65 조건에서는 mlp보다 transformer/resnet_tab
쪽이 전반적으로 더 나은 회귀 헤드로 보인다 — 최종 baseline 모델 선택 시 transformer로
교체를 검토할 근거가 된다.

**실험9(§8-5) 결과 — MIT-only/HUST-only, §8-5에서 미리 계산해둔 풀링 모델의 데이터셋별
성능과 비교 (2026-08-10, 이번 세션 MMD 필요성 검증의 핵심 결과)**:

| | 풀링 모델의 해당 서브셋(§8-5 1차 점검, 직접 집계) | 단일 데이터셋 학습(16·17번 행) | 차이 |
|---|---|---|---|
| MIT | R²=0.9099, RMSE=1.315%p, MAPE=1.030% | R²=**-0.2038**, RMSE=4.809%p, MAPE=4.191% | **풀링이 압도적으로 유리**(R² 1.11 이상 차이) |
| HUST | R²=0.9508, RMSE=1.609%p, MAPE=1.357% | R²=0.9477, RMSE=1.658%p, MAPE=1.420% | 풀링이 근소 유리(R² -0.0031) |

**MIT-only 결과가 예상보다 훨씬 나쁘다** — R²가 음수라는 건 "그냥 평균값으로 찍는 것보다도
못 맞힌다"는 뜻이다. 원인 조사를 위해 학습 로그·예측값을 직접 확인했다: `train_log.csv`
마지막 epoch(33, `early_stop_patience=30`로 조기종료)의 `val_r2`가 -0.40~-0.72 사이를
오갔고, `test_predictions.csv`의 `soh_pred`가 최대 **1.858**까지 나온다(SOH가 물리적으로
1.0을 넘을 수 없는데도) — 이는 스크립트 버그가 아니라(**test 셀은 정확히 baseline의 MIT
20셀과 일치함을 확인함**, forced_test_cells 정상 작동) **MIT 123셀만으로는(train 약 77셀)
새로 학습하는 Phase 1 게이트 + 회귀 헤드가 제대로 수렴하지 못하고 발산에 가까운 상태로
조기종료된 것**으로 보인다.

**이게 §7(`docs/SOC.md`가 아니라 이 문서 §8-5)의 질문(MMD 없는 풀링이 문제인가)에 주는
답은 정반대다** — 최소한 MIT에 대해서는, **풀링이 negative transfer를 일으키기는커녕
MIT 단독으로는 불가능했던 안정적 수렴 자체를 가능하게 해주는 필수 요소**로 보인다(HUST가
데이터 다양성·양을 보태준 덕에 Phase 1 게이트 학습이 안정화됐을 가능성). HUST는 원래도
데이터가 더 크고 프로토콜이 다양해(§8-5 배경 참고) 단독 학습으로도 준수했지만(R²=0.948),
그래도 풀링 쪽이 근소하게 더 나았다. **결론: 이 프로젝트의 현재 규모(MIT 123셀·HUST
77셀)에서는 MMD 없는 단순 풀링이 두 데이터셋 모두에 순이익이며, 도메인 정렬을 추가로
투입해야 할 긴급한 필요는 실측상 확인되지 않는다** — Gong 외의 few-shot(target 3~4셀)
시나리오와 달리, 이 프로젝트는 애초에 "데이터가 부족한 쪽"이 있다면 그건 MIT-only처럼
쪼갰을 때뿐이고, 풀링 자체가 그 문제를 해결해주고 있다.

### 10.4 다른 논문과의 성능 비교 (2026-08-10, 2차 갱신)

**주의(스케일/단위)**: 표의 RMSE/MAE는 전부 "%p"(SOH를 0~100 스케일로 놓았을 때의
percentage-point) 기준으로 통일했다. 내 파이프라인의 원시 저장값(`metrics.json`)은
0~1 SOH 비율 스케일이라 **×100**해서 옮겼다(예: `rmse=0.01520`→`1.520%p`). 논문이
Ah 단위로 보고한 경우(iScience)는 원 단위 그대로 병기했다 — 정격/기준용량을 모르면
%로 환산이 안 되므로 임의 변환하지 않았다. **확인수준** 열은 원문 PDF를 직접 읽고 수치를
뽑은 것(`docs/SOC.md` §2에서 이미 검증했거나 이번에 `docs/*.pdf`를 직접 재확인한 것)과
웹서치 요약만으로 확인한 것을 구분한다 — 후자는 AI 요약이 단위를 잘못 옮기는 사고가
실제로 있었으므로(§ 아래 참고) 그대로 신뢰하지 말 것. **부분사이클 사용방법** 열은 해당
논문이 "한 사이클 안의 일부 구간"을 쓰는지, 쓴다면 어떤 방식(고정 위치/임의 위치/전압
윈도 등)인지를 요약한다 — 없으면 `-`.

| # | 방법/논문 | 저널·연도 | 화학종/규모 | 부분사이클 사용방법 | SOC/SOH 레퍼런스 값 사용 방법 | RMSE | MAE | MAPE | R² | 확인수준 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **내 결과(baseline, 실험1행)** | — | MIT+HUST, LFP, 200+셀·24만+cyc | q_frac_ref 존별 고정 위치 세그먼트(부분사이클) | 갱신형 최근 레퍼런스(coulomb-counting 기반 q_ref + 노이즈 시뮬레이션) | 1.520%p | 1.133%p | 1.251% | 0.9524 | 직접 측정(§10.3) |
| 1 | 고속충전 SOH(Zhao, Li 외) | *iScience*, 2025(IF 4.5/4.1, Q1) | LFP, 222셀·146,074cyc | SOC 80~97% 고정 구간 | 고정 2종 병용 — 정격(공칭)용량 분모 + 10번째 사이클(BOL) 대비 편차 | 0.00479Ah(=4.79mAh) | — | 0.39% | 0.977 | **원문 PDF 확인**(`docs/SOC.md` §2.1) |
| 2 | Multistage SOH(Shi 외, 최선 케이스) | *J. Energy Storage*, 2026(Q1) | LCO/NCA, partial charge | 부분 충전곡선(IC-peak+DTW/MMD, 초기전압<3.9V 조건에서 최선) | BOL 고정 — 초기(BOL) 사이클 IC곡선과 DTW/MMD 비교 | 0.933% | 0.656% | 0.877% | — | **원문 PDF 확인**(`docs/SOC.md` §2.3) |
| 3 | CV충전 SQI/ΔQ+BPNN(Xian, Li 외) | *PLOS ONE*, 2025(Q1~Q2) | NCM/NCA, CV충전단계 일부 | CV 충전단계 말단 일부(고정 전류 임계값 트리거) | SOC/레퍼런스 용량 미사용(고정 전류 임계값 트리거만) | <0.16% | <0.10% | — | >0.98 | **원문 PDF 확인**(`docs/SOC.md` §2.2) |
| 4 | 임피던스 전이주파수(Al-Smadi & Abu Qahouq) | *Batteries*(MDPI) 11(4):133, 2025 | 상용셀, SOH 96~60% 범위 | `-`(EIS 임피던스 스펙트럼 기반, 충전곡선 부분윈도 아님) | 해당 없음(임피던스 스펙트럼 기반, SOC/레퍼런스 용량 미사용) | ≤1.39%(최악값) | ≤1.25%(최악값) | ≤1.55%(최악값) | ≥0.983(최소값) | 웹서치 요약(원문 미대조) |
| 5 | Cycle-based DL, MLP(Bairwa, Pareek, Jadoun) | *Scientific Reports*, 2025-10-23 | NCA(NASA B0005), 3셀 초소형 벤치마크 | `-`(전체 방전곡선 사용, 부분사이클 아님) | 미확인(원문 미대조) | 0.69%p | 0.49%p | — | 0.9955 | 웹서치 요약(원문 미대조) |
| 6 | Multi-Task SOH/RUL 프레임워크 | **arXiv 프리프린트**, 2026-03(동료심사 안 됨) | NASA/CALCE 소형 벤치마크 | 미확인 | 미확인 | 0.92~1.05% | 0.65~0.78% | 0.90~1.07% | — | 웹서치 요약, **미검증 프리프린트** |
| 7 | iMOE(Huang, Tao 외) | *Nature Communications*, 2026(IF 18.1/15.7, Q1) | NCA/NCM/LCO(**LFP 미포함**), 295셀·93조건·84,213cyc | 임의 시작전압(무작위 초기SOC)~컷오프 부분충전곡선 + 완충 후 30분 이완전압(OCV relaxation) 곡선 | 레퍼런스 불필요(임의/미상 initial SOC에서도 강건 — history-free 설계) | TPSL 평균 0.05(Ah 추정, 원문 단위 불명확) | — | 전체평균 0.95%(데이터셋별 0.52~2.96%) | 수치 미제시(그래프만) | **원문 PDF 확인**(2026-08-10 재확인) |
| 8 | 클러스터 기반 sparse SOH(Ke, Lin 외) | *Energy*(Elsevier) 344:140057, 2026(IF 10.1/10.95, Q1) | 전체는 NMC(자체 4셀)+LC(Oxford)+MIT LFP 124셀 3종, **단 Table 6(케이스 a~h)은 NMC 4셀 단일 데이터셋으로 추정**(§10.4 해석 참고, MIT LFP는 Fig.7/8 일반화 검증에만 등장) | Table 6: 세그먼트 1/2/4/8개(=SOC구간 3.75/7.5/15/30%) 위치 2군데씩 비교 | 레퍼런스 용량 미사용(SOC/총용량 무관 임의 위치·길이 윈도 클러스터링) | 1.15~2.99%(케이스 a~h) | 0.95~2.60%(케이스 a~h) | — | 77.34~96.94%(케이스 a~h) | **원문 PDF 확인**(2026-08-10, PyMuPDF 직접 추출·Fig.6 이미지 렌더링으로 케이스 정의 재확인) |
| 9 | Data sufficiency(Su, Tao 외) | *Cell Reports Physical Science* 6:102901, 2025(IF 6.9/7.25, Q1) | LFP 포함 6화학종, 310셀·7데이터셋·30만+cyc | `-`(부분사이클 아님, 전체 CCCV 곡선) — 대신 **전체 수명 중 초반 부분만** 사용(평균 8%) | SOC/용량 정규화 없음(단, 직전 3사이클 피처를 autoregressive LSTM 입력으로 사용) | — | — | 중앙값 1%(수명 8%만으로 달성) | — | **원문 PDF 확인**(2026-08-10, PyMuPDF 직접 추출) |

**#1·#8 추가 확인 사항(2026-08-10)**:
- **#1(iScience)**: 웹서치로 원문 확인 결과 "Trained on fast-charging protocols (3.6C–8C charge, 4C discharge)" — 충전 3.6~8C/방전 4C 고정의 **단일 fast-charging 프로토콜 계열 1종만** 사용한 결과다(`docs/DATASET_MIT_README.md`의 MIT `C1(Q1)-C2` 프로토콜 설명과 사실상 동일 계열 — 내 MIT 124셀과 같은 실험설계 계열로 추정).
- **#8(클러스터 기반 sparse SOH)**: Table 6(케이스 a~h)의 정량 수치는 Dataset 1(NMC)만 대상이며, **LFP(Dataset 3) 전용 케이스별(a~h) 수치는 논문에 없다.** LFP는 §4.4(Fig.9, 세그먼트 개수 1~20 × 100회 랜덤 샘플링)에서만 확인되는데, 이마저 정확한 수치표가 아니라 바이올린플롯(이미지)이고, 텍스트로 확인 가능한 수치는 "1개 세그먼트일 때 LFP 최대 RMSE 2.3~2.9%"와 "10개 이상 세그먼트일 때 median RMSE 약 0.8~1.2%(Dataset 1~3 통합값, LFP만 분리한 수치 아님)" 두 가지뿐이다.

**해석**:
- 화학종까지 같은(LFP) 논문은 #1·#8 두 편이다. #1(MAPE 0.39%)은 SOC 80~97% 고정구간+
  정격용량 정규화 방식이라(이 세션에서 계속 짚은 SOC 의존성 이슈) 방법론이 다르다.
- **#8 재검증(2026-08-10 정정)**: 이전 버전에서는 "#8 최선케이스 RMSE≈1.7%로 내
  RMSE(1.52%p)와 비슷한 수준 — 가장 방법론적으로 유사하고 직접 비교 가치가 있는 대상"
  이라고 썼는데, 이 "1.7%"는 세그먼트 개수 ablation인 Table 6이 아니라 논문의 **별도
  실험**(§4.3, 전압폭 0.11V 윈도)에서 나온 수치를 잘못 대응시킨 것이었다 — 정정한다.
  Table 6은 세그먼트 1/2/4/8개(=SOC 구간 3.75/7.5/15/30%)를 위치 2군데씩(초반/후반)
  비교하는데, 이 개수를 내 `n2`(세그먼트 1개의 길이)와 동일 기준(총 용량길이)으로
  매칭하면:

  | 내 실험 | 길이 | R²/RMSE/MAE | 논문 대응 케이스 | 길이 | R²/RMSE/MAE |
  |---|---|---|---|---|---|
  | n2=5%(min_pts=10, 2번행) | 5% | 76.46% / 3.198%p / 1.946% | (a)/(b) 1개 | 3.75% | 77.34~78.79% / 2.89~2.99% / 2.23~2.60% |
  | n2=5%(min_pts=6, 11번행) | 5% | 81.59% / 2.989%p / 2.070% | (a)/(b) 1개 | 3.75% | 〃 |
  | n2=9%(3번행) | 9% | 88.53% / 2.359%p / 1.722% | (c)/(d) 2개 | 7.5% | **92.21~93.07% / 1.65~1.75% / 1.37~1.49%** |
  | n2=20%(baseline, 1번행) | 20% | 95.24% / 1.520%p / 1.133% | (e)/(f) 4개 | 15% | **95.98~96.31% / 1.26~1.34% / 1.02~1.11%** |
  | — | — | — | (g)/(h) 8개 | 30% | 96.19~96.94% / 1.15~1.22% / 0.95~0.98% |

  **추가 정정(2026-08-10, Table 6 데이터셋 규모 확인)**: Table 6(케이스 a~h)이 실제로
  어느 데이터셋인지 원문에 명시 문장은 없지만, 정황상 **Dataset 1(자체 수집 NMC, 4셀,
  25℃, ~600cyc)만 쓴 것으로 추정된다** — (1) 방법론 예시 그림(Fig.2/3)이 전부 "in
  Dataset 1"로 명시, (2) 바로 다음 §4.2 소제목이 "...across battery types"라 그때
  처음으로 3개 데이터셋을 함께 쓴다는 뜻이 되므로 §4.1(Table 6)은 그 이전 단일
  데이터셋일 수밖에 없음, (3) Fig.6의 SOH 궤적이 cycle 0~600 구간의 매끄러운 단일
  궤적으로 Table 1의 "Dataset 1, 4셀, 거의 1년 사이클" 설명과 부합. MIT LFP 124셀은
  §4.2(Fig.7)·§4.3(Fig.8)의 "화학종 간 일반화" 검증에만 등장하고 Table 6에는 안 쓰인
  것으로 보인다.
- #4·#5는 "최악값"·"3셀 초소형" 조건이라 내 200+셀·24만+cyc 규모의 결과와 난이도 자체가
  다르다 — 숫자만 보고 순위를 매기면 오해의 소지가 크다.
- #6은 **동료심사를 거치지 않은 프리프린트**라 다른 행과 같은 신뢰도로 취급하면 안 된다.
- #7(iMOE)은 부분사이클+이완전압을 쓰는 "이력 불필요(history-free)" 설계라 이번 세션
  내내 논의한 "레퍼런스 순환성" 문제의 실제 해법 사례로서 `docs/SOC.md` §2.1의 논지와
  직결된다 — 다만 LFP가 없고, 타겟이 "현재 SOH"가 아니라 "미래 열화 궤적"이라 태스크
  자체가 다르므로 수치를 직접 순위 비교하기는 어렵다. RMSE는 TPSL 데이터셋 기준 "0.05"
  로만 원문에 나오고 단위가 명시돼 있지 않아(Ah로 추정) 그대로 병기했다.
- #9는 RMSE/R²를 아예 보고하지 않고 MAPE만 쓰는 논문이라 다른 지표는 공란으로 뒀다 —
  "부분사이클"이 아니라 "부분 수명(early-cycle)"이라는 다른 축의 데이터 절약 전략이라는
  점에 유의(내 실험들의 "부분사이클 세그먼트" 절약과는 다른 문제를 푼다).
- "열/기계 특징 융합(LFP)" 논문(MAE 1.786%→0.822%로 개선됐다던 것)은 이번 재검색으로
  정확한 출처를 다시 특정하지 못해 **표에서 제외**했다 — 이전 턴의 웹서치 요약을 근거로
  없는 출처를 확정 인용하는 것보다는 빼는 쪽을 택했다.
- 웹서치 요약 자체의 오류 사례: iScience 논문을 처음 검색했을 때 AI 요약이 "MAPE of 3.89
  mAh"라고 표현했는데 실제로는 **MAE≈3.89mAh, MAPE=0.39%**(무단위 %)였다 — 위 표의
  확인수준이 "웹서치 요약"인 행들은 이런 식의 단위 혼동이 원문 대조 없이는 남아있을 수
  있다는 뜻이다.
- #7~#9는 `docs/` 폴더의 로컬 PDF에서 직접 읽었다 — #8·#9는 파일 용량이 20MB를 넘거나
  PDF 렌더링 도구(`pdftoppm`)가 환경에 없어 `Read` 툴의 기본 페이지 렌더링이 실패했고,
  대신 PyMuPDF(`fitz`)로 텍스트만 직접 추출해서 읽었다.

### 10.5 게재 가능성 평가 (2026-08-10 신설, 2026-08-11 실험7/8/9 결과 반영 재평가)

**결론(갱신)**: 실험7·8(구조 탐색)로 baseline보다 확실히 나은 모델이 나왔고, 실험9(풀링
없이 단일 데이터셋만 쓴 대조군)로 "MMD 없는 풀링이 정당한가"라는 가장 취약했던 지점에
**직접적인 실증 근거**가 생겼다. 정확도 자체는 여전히 이 비교표의 상위권이 아니지만,
**"체계적 실패모드·설계선택 검증" 쪽 근거는 이번 갱신으로 눈에 띄게 두꺼워졌다.**
Q1급 응용에너지 저널 투고 시도는 이전보다 더 현실적이 됐고, "SOTA 갱신"이 아니라
"체계적 검증+대규모 교차데이터셋 일반화"로 포지셔닝하는 전략은 그대로 유효하다.

**1) 정확도 재평가 — 이제는 mlp baseline이 아니라 transformer가 "현재 최고"다.**

| | mlp baseline(1번 행, 기존 비교 기준) | **transformer(14번 행, 현재 최고)** |
|---|---|---|
| R² | 0.9524 | **0.9611** |
| RMSE | 1.520%p | **1.373%p** |
| MAPE | 1.251% | **1.047%** |

이 새 수치로 §10.4 비교를 다시 보면:

| 비교 대상 | 비교 결과(transformer 기준) |
|---|---|
| #8(클러스터, 길이-매칭 15%(e/f) 케이스: R²=95.98~96.31%, RMSE=1.26~1.34%p) | 우리 20%-길이 R²(96.11%)가 **이제 이 구간 안/근접**(기존 mlp 95.24%는 이 구간보다 낮았음) — RMSE는 여전히 근소 열세(1.373 vs 1.26~1.34)지만 격차가 확 줄었다. 단, 우리가 여전히 더 긴 길이(20% vs 15%)를 쓰고 있다는 비대칭은 남는다. |
| #1(iScience, MAPE=0.39%) | 여전히 2.7배 격차(1.047% vs 0.39%) — 다만 §10.4에서 이미 확인했듯 iScience 222셀이 사실상 우리 MIT와 같은 단일 프로토콜 계열이라는 점이 이 격차의 상당 부분을 설명한다는 논지는 그대로 유효, 오히려 우리 쪽 수치가 개선되며 그 논지가 조금 더 설득력을 얻었다. |
| #2/#3(0.16~0.93% 대) | 여전히 더 낮은 값들이나, 화학종·조건이 달라(§10.4 해석 참고) 직접 순위 비교는 여전히 부적절. |

**결론적으로 transformer로의 교체는 "SOTA 역전"까지는 아니지만, 가장 방법론적으로
유사한 비교 대상(#8)과의 격차를 실질적으로 좁혔다** — §10.4의 baseline 행을 앞으로
transformer 결과로 갱신하는 것을 검토할 만하다(단, 이 문서의 실험 1~6·9는 전부 mlp
baseline 기준으로 통제된 것이므로, transformer로 교체 시 그 비교들을 다시 통제해야
한다는 점은 유의).

**2) 데이터 규모·다양성** — 기존 평가와 동일, 여전히 강점(MIT+HUST 200+셀·24만+cyc,
비교 대상 대부분이 단일 데이터셋).

**3) 방법론적 엄밀성 — 실험9로 가장 크게 보강됐다.**

기존 평가(2026-08-10)는 "MMD 없이 풀링해도 괜찮은지"를 **논리적 추론**(few-shot
시나리오가 아니므로 Gong 외의 극적 격차가 그대로 적용되지 않는다)으로만 방어했다.
지금은 **직접 대조군 실험(실험9)이 있다**: MIT만 단독 학습하면 R²=**-0.2038**(발산에
가까운 실패)로 무너지는데, MIT+HUST를 풀링하면 그 서브셋에서만도 R²=0.9099로 정상
작동한다 — **"MMD 없는 풀링이 negative transfer는커녕, 이 데이터 규모에서는 안정적
수렴 자체를 가능하게 하는 필수 요소"라는 걸 사후 추론이 아니라 통제된 ablation으로
보여준다.** 이건 리뷰어가 "왜 도메인 정렬을 안 썼냐"고 물었을 때 논리 대신 숫자로
답할 수 있는 근거이고, 비교 논문(Gong 외)이 제기하는 우려에 **정면으로 답하는 실험**이라는
점에서 이 프로젝트만의 독자적 기여로 포지셔닝할 수 있다. 추가로 실험7(raw 융합 2종)·
실험8(회귀헤드 3종) 역시 baseline이 임의로 고른 게 아니라 체계적 탐색 후 정당화된
선택이라는 걸 보여준다(§8-3/§8-4).

**4) 종합 판단(갱신)**

- **"정확도 1등" 주장은 여전히 어렵다** — #1/#2/#3 등 여러 논문이 여전히 더 낮은
  절대 오차를 보고한다.
- **"체계적 검증 + 대규모 교차데이터셋 일반화 + MMD 없는 풀링의 실증적 정당화"로
  포지셔닝하는 전략은 이전보다 더 근거가 탄탄해졌다** — 특히 실험9는 이 표의 어떤
  비교 논문도 갖고 있지 않은 유형의 결과(단일/풀링 데이터셋 직접 대조)라 차별화
  포인트로 쓸 만하다.
- **남은 과제 갱신**:
  1. ~~동일 %길이에서 교차데이터셋 일반화 격차 vs 단일데이터셋 논문들~~ — **실험9로
     해결**(위 §3 참고). 다만 이 결과가 다른 축(n2 값, min_pts 등)에서도 재현되는지는
     추가 검증 여지가 있음.
  2. `min_pts=6`도 완전 해결이 아니라 "부분 완화"에 그친 상태(§10.3 실험6 해석) —
     여전히 미해결.
  3. §8-2에서 계획했던 zone별 `capacity_curve_*.png` 회복 확인 — 여전히 미완료.
  4. (신규) transformer가 새 최고 성능이므로, §10.4 baseline 행 및 실험 1~6·9의
     "baseline 대비 비교"를 transformer 기준으로 다시 통제할지 결정 필요 — 지금은
     전부 mlp 기준으로 비교돼 있어 일관성 있게 유지할지, 최고 성능만 별도로 보고할지
     정책 결정이 필요하다.

---

## 11. 진행 상황 (2026-08-08)

1. ~~§2의 가정 4개 확정~~ — **완료**, 그대로 확정
2. ~~§3.1(HI 토글) 구현~~ — **완료**(환경변수 `SOH_EXCLUDE_STAT_LEAK`, 단위테스트 통과)
3. ~~§3.2(shape filter 배선)~~ — **완료**(코드 배선 + `--help` 검증). 전체 재전처리는 예상과
   달리 **이미 끝나 있어서**(§3.2) 추가 실행 불필요
4. ~~baseline noise_amp 5%→3% 조정~~ — **완료**(2026-08-08 2차 갱신, §3.3 데이터 재사용
   극대화 목적). §0/§2/§4~§8 전부 3% 기준으로 재작성함.
5. **실행(아직 안 함)** — §4~§8에 Step4/Step6 커맨드 전부 정리 완료. 실제 필요한 작업량:
   - Step 4(추출) 신규 **4개 조합**: n2=5%(N=4,noise=3%), noise=1%/5%/10%(n2=20%,N=4).
     baseline·n2=9%·shape미적용(noshape)은 기존 데이터 재사용(§3.3)이라 제외.
   - Step 6(학습, Phase1→분류기→Phase2→평가) 총 **10건**(실험별 데이터 조합에 학습 파라미터만
     바꿔 도는 것 포함)
   - 순서 제안: Step 4 신규 4건 먼저 추출 → baseline Step 6 실행 → 나머지 9건 Step 6 실행
     → §10 템플릿 채워 결과 정리
6. 다음 지시를 기다림 — 이 문서 검토 후 실행 여부/순서를 알려주면 시작한다.

---

## 12. 실행 인프라 벤치마크 (2026-08-08) — 드라이브 I/O·워커 수

실제 실행(§10) 전에 두 가지 인프라 레버를 점검했다: (a) 중간 pkl 저장 위치를 C→D
드라이브로 옮기면 이득이 있는지, (b) `--workers` 수를 지금 커맨드(16)보다 올리면 이득이
있는지. 둘 다 **실제 코드·실제 데이터로 측정**했다(가정이 아님).

### 12.1 C 드라이브 vs D 드라이브 pkl I/O

실제 `q_frac_ref`(n2=20%, noise=3%) MIT 데이터(seg 대용량 123개 + cycle 소용량 123개,
총 1,108.5MB)를 C(프로젝트 루트와 같은 물리 드라이브, 스크래치패드 경로)와 D(`D:\`,
별도 물리 드라이브, 여유 6.5TB)에 각각 복사(쓰기)하고 `pd.read_pickle`로 두 번씩
읽었다(읽기 순서는 매번 랜덤화).

| 항목 | C 드라이브 | D 드라이브 | 차이 |
|---|---|---|---|
| 쓰기(복사) 1108.5MB | 1.20s (925 MB/s) | 1.18s (940 MB/s) | -1.6% |
| 읽기(`pd.read_pickle`) 1차 | 5.97s (186 MB/s) | 5.96s (186 MB/s) | -0.2% |
| 읽기 2차 | 5.70s (195 MB/s) | 5.65s (196 MB/s) | -0.8% |

**결론: 차이 없음(전부 2% 이내, 측정 노이즈 수준).** 두 드라이브 다 빠른 SSD이고,
읽기 처리량(~190MB/s)이 순수 복사 처리량(~930MB/s)보다 훨씬 낮다는 것 자체가 병목이
**디스크가 아니라 pandas 언피클링(CPU 역직렬화)**이라는 뜻이다 — 이게 바로 아래 §12.2에서
워커 수가 훨씬 중요한 레버로 확인되는 이유와 직결된다. **D 드라이브로 옮길 속도상 이유는
없다** — C 드라이브 여유공간(~442GB)도 충분해 공간 문제로 옮길 필요도 당장은 없다고
판단, 이 축은 보류.

### 12.2 CPU 워커 수 — 실측 결과 "16보다 훨씬 올릴 여지가 크다"

**배경**: 이 머신은 48코어다. `hi_correlation.py` 자체 기본값은 `max(1, cpu-2)=46`인데,
§4~§8·§10의 커맨드는 전부 `--workers 16`으로 적어놨다 — 벤치마크 없이 이전 세션 관행을
그대로 따른 값이라 실측이 필요했다.

**측정 과정(안전하게 축소)**: 처음엔 MIT 전체(123셀, noise=1%/16%/46% 3점) 풀 스케일
벤치마크를 시도했으나 workers=4에서 10분간 전체의 ~15%(11,379/약 245,000cyc)만 처리될
정도로 무거워 **중단하고 프로세스를 강제 종료**(orphan 프로세스 6개 확인 후 kill,
실제 실험 데이터 폴더는 전혀 안 건드려짐 — n2=0.11은 실험 계획에 없는 처분용 값이라
경로 충돌 없음). 이후 **`hi_correlation._extract_one_cell()`을 디스크 저장 없이 직접
호출**하는 방식으로 축소 재설계: `_4_data_hi/clean/MIT/`에서 가장 작은 셀 2개
(`b2c1.pkl`=170cyc, `b2c0.pkl`=326cyc)만 읽기 전용으로 처리해 비교했다.

| 실행 방식 | 결과 |
|---|---|
| 단일 프로세스(workers=1), 순차 처리 | b2c1: 38.36s, b2c0: 77.89s → **합계 116.25s** (≈0.226~0.239s/cycle, 두 셀 일관) |
| `ProcessPoolExecutor(max_workers=2)`, 병렬 | **총 wall-clock 80.66s** |

**속도 향상 ≈1.44배**(이론상 최대는 2배지만 두 셀의 작업량이 안 맞아서(170 vs 326cyc)
느린 쪽 하나(77.89s)가 병목이 됨 — 실제 80.66s가 그 병목값에 거의 정확히 붙어 있어
**병렬화 자체는 오버헤드 없이 거의 이상적으로 동작**한다는 뜻이다(작업 분배가
균등했다면 2배에 더 가까웠을 것).

**함의**: 사이클당 비용(~0.23s)과 전체 MIT+HUST 사이클 수(약 245,802개, §3.2 노이즈
전처리 요약 기준)로 전체 추출의 순수 CPU 총량을 추산하면 약 245,802×0.23s
≈ 56,500 CPU-sec ≈ **15.7 CPU-시간**. 워커 수별 이상적(오버헤드 0) wall-clock 추정:

| `--workers` | 이상적 wall-clock(추정) |
|---|---|
| 8 (`run_pipeline.py` 자체 기본값) | ≈117분 |
| 16 (지금 §4~§10 커맨드에 쓴 값) | ≈59분 |
| 32 | ≈29분 |
| 46 (`hi_correlation.py` 자체 기본값, cpu-2) | ≈20분 |

실측(2코어, 불균등 작업)에서 병렬 오버헤드가 거의 0에 가까웠던 걸 감안하면, 코어 포화
전까지는 이 표가 꽤 근접한 추정일 가능성이 높다 — 단, **46코어 규모까지 실제로 검증한
건 아니라서**(디스크 I/O는 §12.1에서 병목이 아님을 확인했지만, 메모리 대역폭·OS
스케줄링 경합은 소규모 테스트로는 안 보일 수 있음) 이론 추정이라는 점은 명시한다.

**권장**: §10.1의 Step 4 선행 추출 4건(및 향후 Step 4 실행 전부)에서 `--workers 16`을
**`--workers 40`** 정도로 올릴 것을 권한다(46 전부를 안 쓰는 이유: OS·기타 프로세스용
여유 확보, `hi_correlation.py` 자체 기본값 46에서 좀 더 보수적으로). 아래 §10.1 커맨드에
반영했다. §10.2의 Step 6(학습) 커맨드는 `--workers`가 Step 1~5(데이터) 전용 인자라
Step 6+에는 영향이 없으므로 그대로 둬도 무방하다.

---

## 13. pkl 데이터 저장 경로 C→D 드라이브 이전 (2026-08-08)

§12.1에서 C/D 드라이브 간 I/O 속도 차이가 없음을 확인했지만, **공간 확보 목적**으로
중간 pkl 저장 경로 세 곳을 D 드라이브로 옮기기로 했다. **실제 파일 이동은 사용자가 직접
수행하고, 이 세션에서는 코드가 참조하는 경로만 수정**했다(파일 이동 전에는 코드가 D:의
빈 경로를 가리키게 되므로, 실제 실행은 파일 이동 완료 후에 해야 한다).

### 13.1 대상 경로

| 이전 위치(C, 현재) | 이후 위치(D) | 크기 | 상태 |
|---|---|---|---|
| `_2_data_clean/` | `D:\chanminLee\LFP_SOH_prediction_v2\_2_data_clean` | 21GB | **코드 어디서도 참조 안 함(실사조사 확인) — 코드 수정 불필요, 파일만 옮기면 끝** |
| `_4_data_hi/` | `D:\chanminLee\LFP_SOH_prediction_v2\_4_data_hi` | 42GB | 26개 파일 + yaml 25개에서 참조 — 전부 수정 완료(§13.2) |
| `4_hi_analysis/`의 pkl 캐시(`hi_features*.pkl`, `outputs/**/*.pkl`) | `D:\chanminLee\LFP_SOH_prediction_v2\4_hi_analysis` | 13GB 중 pkl 부분 | `.py` 코드 파일은 그대로 C에 둠(임포트·실행 위치라 이동 불가) — pkl 캐시 상수만 별도 분리해 수정(§13.2) |

**`_2_data_clean`이 미사용으로 확인된 근거**: `2_preprocess/preprocess.py`의 실제 출력
경로는 `POSTPROCESS_ROOT = PROJECT_ROOT / "_4_data_hi" / "clean"`이고(이미 `_4_data_hi`
밑으로 통합돼 있음), `1_convert/convert_unified.py`는 `_1_data_unified`/`_0_data_raw`에
쓴다 — `_2_data_clean`을 가리키는 코드가 처음부터 없다. mtime(2026-06-24~07-01)도 최근
활동(08-08)보다 한 달 이상 오래돼 이전 파이프라인 구조의 잔재로 판단된다.

### 13.2 `data_directories.py` — 단일 경로 진입점으로 중앙화

처음에는 17개 `.py` 파일(+ yaml 25개)에 D: 절대경로 문자열을 각자 인라인으로 박아
넣었는데, 이러면 "공유 상수 모듈이 없어 26개 파일이 각자 하드코딩"이라는 §13 도입부의
지적을 그대로 반복하는 셈이었다 — 사용자가 이 점을 짚어줘서, 프로젝트 루트에
**`data_directories.py`** 하나를 만들고 모든 `.py` 파일이 여기서 import하도록
리팩터링했다(yaml은 파이썬 import가 불가능해 문자열 리터럴 유지 — 25개 전부 값이
동일해 유지보수 부담은 낮음).

```python
# data_directories.py — 이 프로젝트 pkl 데이터 경로의 유일한 진입점
_D_ROOT = Path(r"D:\chanminLee\LFP_SOH_prediction_v2")
DATA_2_CLEAN_ROOT  = _D_ROOT / "_2_data_clean"   # 미사용(레거시), 문서화 목적
DATA_4_HI_ROOT     = _D_ROOT / "_4_data_hi"
DATA_4_HI_ROOT_STR = str(DATA_4_HI_ROOT).replace("\\", "/")  # PROJECT_ROOT join용
PKL_CACHE_ROOT      = _D_ROOT / "4_hi_analysis"  # hi_features*.pkl, outputs/**/*.pkl
```

**나중에 드라이브를 또 옮길 일이 생기면 이 파일의 `_D_ROOT` 한 줄만 고치면 된다** —
26개 파일을 다시 뒤질 필요가 없다.

| 파일 | 수정 내용 |
|---|---|
| `2_preprocess/preprocess.py` | `POSTPROCESS_ROOT`(→`_4_data_hi/clean`, `clean_noshape`도 자동 추종) |
| `3_integrity/audit_zero_info_hi_segments.py` | `SEG_DIR` |
| `3_integrity/charge_gap_census.py` | `CLEAN_DIR` |
| `3_integrity/check_integrity.py` | `MIT_DIR`/`HUST_DIR` |
| `3_integrity/phase2_dataset_features.py` | `CLEAN_DIR` |
| `3_integrity/plot_qfref_noise.py` | `CYCLE_DIR` |
| `3_integrity/validate_capacity_error.py` | `CLEAN_DIR` |
| `4_hi_analysis/hi_correlation.py` | `MIT_DIR`/`HUST_DIR`/`HI_ROOT`(기본+`--skip-shape` 재지정 두 세트) + `CACHE_PATH`(`PKL_CACHE_ROOT` 사용 — `STEP_DIR`은 `hi_segment_viz.py` 등 코드 파일 조회에도 쓰여 그대로 둠) |
| `4_hi_analysis/qabs_healthy_vs_aged.py` | `MIT_DIR`/`HUST_DIR`(입력만; `OUT_DIR`은 플롯 전용이라 미변경) |
| `4_hi_analysis/seg_diagnose.py` | `MIT_DIR`/`HUST_DIR` + `hi_features*.pkl` glob 대상 + `survival_stats_*.pkl` 저장 `out_dir`(`PKL_CACHE_ROOT`, hi_correlation.py와 공유) |
| `5_model/clustering.py` | `SEG_DIRS` |
| `5_model/hi_compute.py` | `__main__` 디버그 블록의 1회성 pkl 읽기 |
| `5_model/train_scr.py` | `seg_data_dir`/`data_dir` 자동경로(`DATA_4_HI_ROOT_STR` f-string 베이스) |
| `5_model/train_classifier.py` | 동일(두 분기 — spec 기반/저장된 config fallback) |
| `5_model/visualize_results.py` | 동일 |
| `5_model/test_scr.py` | `random_seg_data_dir` 기본값(fallback) |
| `tmp_make_test_rs.py` | `out_dir`/`clean_dir`(`test_rs` 데이터 생성 스크립트 — 계속 쓰이므로 수정. `tmp_standalone_overfit.py`는 이미 죽은 `leak_cols` 패턴을 쓰는 완전 disposable 스크립트라 미변경) |
| `5_model/config/*.yaml` (25개) | `random_seg_data_dir: "_4_data_hi/test_rs/seg"` → D 절대경로로 일괄 치환(yaml이라 상수 재사용 불가, 리터럴 유지) |

**임포트 위치 주의**: 각 파일이 `sys.path`에 프로젝트 루트를 넣는 시점이 제각각이라
(일부는 `PROJECT_ROOT / "5_model"`만 넣고 루트 자체는 안 넣는 경우도 있었음), `from
data_directories import ...`는 반드시 **그 파일에서 `sys.path`에 프로젝트 루트가 이미
들어간 뒤**에 와야 한다 — 파일별로 위치를 맞춰 넣었고, 8개 대표 파일(`data_directories`
단독 임포트, `check_integrity.py`/`seg_diagnose.py`/`hi_correlation.py`/`clustering.py`의
임포트 체인)을 실제로 실행해 정상 로드를 확인했다.

**경로 해석 원리**: `5_model/*.py`의 상대경로(`f"_4_data_hi/{axis_dir}/seg"`)와
`*.yaml`의 `random_seg_data_dir`는 나중에 `segment_dataset.py`에서
`PROJECT_ROOT / data_cfg["seg_data_dir"]`로 join된다. pathlib는 join 우변이 절대경로면
좌변(`PROJECT_ROOT`, C:)을 무시하고 우변을 그대로 쓰는 표준 동작을 하므로(직접 확인함),
이 경로들을 D: **절대경로 문자열**(`DATA_4_HI_ROOT_STR`)로 바꾸는 것만으로
`PROJECT_ROOT`를 안 건드리고도 정확히 D:로 리다이렉트된다.

**미변경(의도적으로 남긴 것)**: 주석·docstring·argparse description 안의 `_4_data_hi`
언급(`check_integrity.py`/`hi_correlation.py`/`common/split.py` 등), 커밋된 과거 run들의
`_5_data_model_scr/*/config.yaml`(그 시점 실행 기록이라 손대면 안 됨), `docs/전달/` 하위
전달용 스냅샷 스크립트, 주석 처리된(`#`) 죽은 코드 라인(`plot_all_mit_cells.py`,
`plot_cell_cycles.py`).

### 13.3 실행 전 체크리스트

- [ ] 사용자가 `_4_data_hi/`, `4_hi_analysis/`의 pkl 캐시(`hi_features*.pkl`,
  `outputs/`)를 `D:\chanminLee\LFP_SOH_prediction_v2\` 밑 동일 이름으로 이동
- [ ] `_2_data_clean/`은 원하면 그냥 D로 이동(미사용 확인됨, 코드 수정 불필요)
- [ ] 이동 후 §10의 Step 4/Step 6 커맨드를 실행해 정상 동작 확인(특히 baseline 1건으로
  먼저 스모크 테스트 후 나머지 9건 진행 권장)

---

## 14. n2=5% 시나리오 소멸 진단 → `min_pts` 하향 실험 11 신설 (2026-08-10)

### 14.1 문제 발견

실험2(`0809_1229_p2_mlp_q_fr_35%_5%_noleak`) 결과 플랏(`capacity_curve_1-7.png`)에서
HUST 셀의 charge가 `Pred-lo`만, discharge가 `Pred-lo`/`Pred-mid`만 나오고 나머지
시나리오가 통째로 빠진 걸 발견했다. 원인 조사 결과:

- `seg_idx`는 hard 라우팅 시 **분류기가 예측한** 시나리오(진짜 라벨 아님) — 즉 분류기가
  그 셀의 세그먼트를 해당 클래스로 아예 예측하지 않는다는 뜻.
- 그런데 원인은 분류기가 아니라 **데이터 자체**였다: n2=5% 추출본에서 HUST 셀 1-7의
  `scen∈{1,2,-3}`(chg_lo/chg_mid/dis_hi 추정)이 **100% all-NaN**(=사실상 세그먼트 0개).
  baseline(n2=20%)에서는 같은 셀·같은 존이 all-NaN 0%로 완전히 정상.
- HUST 전체(77/77셀)가 n2=5%에서 하나 이상의 존이 100%-NaN. MIT는 123개 중 36개(29%)만
  영향받음(그마저 실패 존도 다름 — 주로 `dis_mid`).
- `seg_diagnose.py --mode compare`로 HUST `1-7` cyc400을 baseline·n2=5%·noise=1/5/10%
  5조건 비교 플랏(`HUST_1-7_cyc400_compare.png`)으로 시각 확인 — noise 조건 3개는 baseline과
  세그먼트 개수·구성이 완전히 동일(24개: dis=12,chg=12), n2=5%만 11개(dis=8,chg=3)로 붕괴.
  **원인은 noise가 아니라 n2다.**
- 메커니즘: baseline에서 `chg_lo`/`chg_mid`/`dis_hi`의 세그먼트당 실측 포인트 수가
  이미 가장 적음(n=30~31, 다른 존은 67~213) — n2를 4분의 1로 줄이면 이 얇은 존들의
  포인트 수가 `min_pts`(기본 10) 밑으로 떨어져 통째로 탈락한다.

### 14.2 `min_pts` 하향 효과 실측

`hi_correlation._extract_one_cell()`을 HI 계산 없이(비싼 `_seg_stat`/`_seg_diff`/
`_seg_lfp` 등을 no-op으로 치환) 세그멘테이션만 반복해, `candidate_n_points`(min_pts
필터링 전 원시 포인트 수 분포)를 HUST 16셀 샘플로 수집:

| 존 | 중앙값 포인트수(n2=5%) | th=10(현재) | th=8 | th=7 | **th=6** |
|---|---|---|---|---|---|
| chg_lo | 7.0 | 0.0% | 43.4% | 87.1% | **99.7%** |
| chg_mid | 7.0 | 0.0% | 32.7% | 82.9% | **99.5%** |
| dis_hi | 10.0 | 52.4% | 82.9% | 96.5% | **99.9%** |

`min_pts=10→6`만으로 죽어있던 존이 거의 전부(99.5~99.9%) 살아난다. baseline(n2=20%)은
th=3~15 전 구간에서 모든 존 100% 생존(median 29~120pt) — 대조군 확인.

### 14.3 `min_pts=6` 적용 시 필요했던 추가 수정 — hi_correlation.py의 8-포인트 게이트는 **죽은 코드**였다

`hi_correlation.py`에 `_seg_diff`/`_seg_lfp`/`_seg_morph_curves` 등 이름으로 `n<8`
게이트가 있었지만, 파일 맨 끝에 이런 위임 구문이 있었다:

```python
# HI 계산 로직의 단일 소스는 5_model/hi_compute.py.
# 아래 import가 이 파일 내 동명 함수 정의를 덮어써...
from hi_compute import (_seg_stat, _seg_diff, _seg_lfp, _build_vq_curve,
                         _build_ica_seg, _peak_fwhm_asym, _seg_morph_curves,
                         _dtw_distance, _frechet_distance)
```

즉 실제로 살아있는 게이트는 **`5_model/hi_compute.py`**에 있었다(hi_correlation.py의
동명 함수는 임포트로 덮어써지는 죽은 정의). 전수조사 결과 `<8` 게이트가 14곳(`_build_vq_curve`,
`_build_ica_seg`, `_seg_morph_curves`, `_vd`, `diff_v_trend_slope`, `diff_dv_di_seg`,
`diff_dvdq_peak_q`, `diff_dvdq_flat_q`, `_plateau`, `lfp_v_flatness`, `lfp_delta_v_rms`,
`lfp_v_concavity`(이건 `<8`과 별개로 `<10`도 있어 둘 다), `lfp_v_q_pearson`,
`lfp_ica_peak_cnt`) — 전부 `min_pts=6`에 맞춰 `<6`으로 낮췄다(`stat_*` 20종은 이미
`len>=5`라 손 안 댐, `stat_v_samp_ent`만 알고리즘 특성상 `n<10`을 그대로 둠 — 표본엔트로피는
O(n²) 페어매칭이라 6~9점에서는 통계적으로 무의미해질 위험이 커서 min_pts 정렬 대상에서
제외).

**보간 대신 `min_pts` 하향을 택한 이유**(사용자 확정): 실측값 그대로 쓰는 게 왜곡이 없고,
이미 `candidate_n_points` 기반 스윕 인프라가 있어 즉시 검증 가능했다. 보간은 `v_std`/
엔트로피처럼 실측 변동성에 의존하는 피처를 인위적으로 매끈하게 왜곡할 위험이 있고 구현·
유지보수 부담도 크다고 판단해 배제했다.

### 14.4 경로 태깅 전파 — `_minptsN` 접미사 신설

`min_pts`가 기본값(10)이 아니면 기존 `_ccOnly`/`_noshape`와 같은 패턴으로 `_minptsN`
접미사를 붙이도록 4곳을 동기화했다(하나라도 놓치면 §4.6급 confound 재발):

- `common/scenario/q_frac_wide.py`: `get_spec()`의 `params`에 `min_pts` 추가(이게 없으면
  `scenario_spec.json`에 안 남아서 분류기 쪽 경로 재구성이 불가능했음)
- `4_hi_analysis/hi_correlation.py`: `_qfw_tag()`(→`_qfref_tag()`도 자동 상속)에 접미사 추가
- `5_model/train_scr.py`, `train_classifier.py`, `visualize_results.py`: 각자 독립
  구현된 `_axis_dir`/`_axis_dir_from_spec()`에 동일 접미사 추가
- `4_hi_analysis/hi_correlation.py`, `run_pipeline.py`에 **`--min-pts` 단축 인자 신설**
  (PowerShell JSON 우회 목적, 기존 `--ref-lag`/`--noise-amp`와 동일 패턴)

4개 파일이 동일 입력(`n1=0.35,n2=0.05,...,min_pts=6`)에 대해 정확히 같은 경로
`q_frac_ref/n1-35%_n2-5%_N-4_minpts6_lag-0_noise-3%_ou-200`를 생성하는지 직접
실행해 교차검증했다.

### 14.5 실험 11 — n2=5% + min_pts=6 (baseline 1~10과 나머지 조건 동일)

**2026-08-10 갱신 — Step 4부터 한 번에**: 원래 Step4(추출)와 Step6(학습)을 따로 실행하는
2단계 커맨드로 적어뒀는데, `run_pipeline.py`를 **Step 4부터** 시작하면 Step4→5(시각화)
→6(Phase1)→7(분류기)→8(Phase2)→9(평가)가 자동으로 이어서 돈다는 걸 확인해서 한 커맨드로
합쳤다. 근거: STEPS 정의상 Step4는 `--force`가 항상 기본 포함되고, `--seg-axis`/축
파라미터(`--min-pts` 포함)는 Step4~6·8에, `--scen-k`는 Step6·8에, `--workers`는
Step1~5에 자동 전달된다(run_pipeline.py 실조사 완료). 중간 Step이 실패하면 `input()`으로
"계속 진행하시겠습니까? [y/N]" 확인을 받으므로(비대화형에서는 자동 중단) 안전하다.

**PowerShell (Step 4부터 한 번에):**
```powershell
$env:SOH_EXCLUDE_STAT_LEAK="1"
python run_pipeline.py 4 --model-config 5_model/config/exp_qfw_mlp_basefix.yaml --seg-axis q_frac_ref --n1 0.35 --n2 0.05 --ref-lag 0 --noise-amp 0.03 --noise-mode ou --noise-period 200 --n-samples 4 --min-pts 6 --scen-k 65 --workers 40
```

결과는 §10.3 표의 11번 행에 기록. 1~10번은 이미 실행 완료된 상태이므로 이번엔 11번만
새로 돈다.
