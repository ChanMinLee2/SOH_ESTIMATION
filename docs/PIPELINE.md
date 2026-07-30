# SOH Estimation Framework — 메인 파이프라인 정리

> 플롯 생성 파일(`plot_*.py`, 시각화 전용) 제외. 데이터 처리·학습·평가 메인 로직만 포함.

---

## 전체 흐름 요약

```
raw data
  ↓ [1_convert]       MIT(.mat HDF5) / HUST(.pkl) → _1_data_unified/
  ↓ [2_preprocess]    7단계 필터링                 → _4_data_hi/clean/
  ↓ [3_integrity]     10개 항목 무결성 검사         → reports/
  ↓ [4_hi_analysis]   HI 추출 (Global 15 + Seg 6×66) + 원시 곡선 리샘플(raw_v/raw_i, 48pt)
                        → _4_data_hi/{axis}/cycle/{ds}/*.pkl  (wide-cycle)
                        → _4_data_hi/{axis}/seg/{ds}/*.pkl    (native-seg + raw_v/raw_i)
                        → _4_data_hi/{axis}/scenario_spec.json
  ↓ [5_model]
      Phase 1 : L0 HardConcrete gate → probe/scen HI 선택 → gates/*.json
      Phase 2 : 고정 gate + 회귀 전용 → checkpoints/best.pt
      Clf     : 독립 시나리오 분류기 학습 → classifier/clf_best.pt  ← B안 추가
                classifier.type = mlp(HI) | cnn(원시 V/I CNN + HI 융합)  ← CNN 옵션
  ↓ [test_scr]   통합 평가 (분류 → 라우팅 → 회귀, 분류기 있으면 자동 all-mode):
      oracle : 정답 seg_idx 라우팅 → 회귀 상한선 (분류기 없어도 항상 실행)
      hard   : 분류기 argmax 라우팅 → 실배포 시나리오
      soft   : 분류기 확률 가중 평균
      (test.random_segment_test=true 시 test_rs로 동일 3-모드 추가 평가)
                → metrics/ + figures/ + routing/ + predictions/
```

오케스트레이터: **`run_pipeline.py`** (Step 1~9)

| Step | 이름 | 스크립트 |
|------|------|---------|
| 1 | 데이터 변환 | `1_convert/convert_unified.py` |
| 2 | 이상 사이클 제거 | `2_preprocess/preprocess.py` |
| 3 | 무결성 검사 | `3_integrity/check_integrity.py` |
| 4 | HI 상관 분석 | `4_hi_analysis/hi_correlation.py` |
| 5 | HI 세그먼트 시각화 | `4_hi_analysis/hi_segment_viz.py` |
| 6 | SCR Phase 1 학습 | `5_model/train_scr.py --phase 1` |
| 7 | SCR Phase 2 학습 | `5_model/train_scr.py --phase 2` |
| **8** | **시나리오 분류기 학습** | **`5_model/train_classifier.py`** |
| 9 | SCR 평가 | `5_model/test_scr.py` |

```powershell
# 전체 파이프라인 (Step 1~9)
python run_pipeline.py

# 학습+평가만 (Step 6~9)
python run_pipeline.py 6

# Phase 1~2만 (분류기·평가 제외)
python run_pipeline.py 6 --to-step 7

# 분류기 학습만
python run_pipeline.py 8 --to-step 8

# 평가만 (체크포인트 직접 지정)
python run_pipeline.py 9 --checkpoint _5_data_model_scr/0716_1200_p2_mlp_prot/checkpoints/best.pt

# 병렬 프로세스 수 지정 (Step 1~5 전용)
python run_pipeline.py --workers 4
```

**Phase 간 자동 핸드오프:**
- Step 6 완료 → 신규 run dir 자동 감지 → Step 7에 `--gates-from` 주입
- Step 7 완료 → 신규 run dir 자동 감지 → Step 8에 `--run-dir` 주입, Step 9에 `--checkpoint` 주입
- **yaml의 `data.gates_from`보다 자동 감지된 CLI `--gates-from`이 우선**

세그멘테이션 축 (`--seg-axis`): `qfrac`(기본) / `protocol` / `vwindow` / `rcs` / `cluster` / `q_frac_wide` / `vqslope` / `full_cycle`  
축 변경 시 Step 4~7이 모두 동일한 `--seg-axis`(및 `--axis-config`)를 사용해야 데이터 경로가 일치한다.

---

## 스크립트별 CLI 옵션 전체 목록

> 이 절은 Step 4 이후 스크립트와 보조 진단/비교 도구의 **모든** 커맨드라인 옵션을 한 곳에 모은 빠른 참조다.
> Step 1~3(`1_convert`/`2_preprocess`/`3_integrity`)의 옵션은 각 절(§1~§3)에 이미 정리되어 있으므로 생략한다.

### 공통: 축 단축 인자 (`--n1`/`--n2`/`--n-samples`/`--mode`/`--random-segment`/`--seg-len-pts`)

`run_pipeline.py`, `hi_correlation.py`, `hi_segment_viz.py` 세 스크립트가 동일한 패턴을 공유한다 — PowerShell이
`--axis-config '{"n1":0.4,...}'`의 따옴표를 벗겨버려 JSON 파싱이 깨지는 문제를 우회하기 위해, 자주 쓰는
q_frac_wide/vqslope 파라미터를 개별 플래그로도 받을 수 있게 했다. 아래 플래그 중 하나라도 지정되면
내부적으로 JSON으로 합쳐져 `--axis-config`를 대체한다(둘 다 주면 이 단축 인자들이 우선).

| 플래그 | 타입 | 대상 축 | 의미 |
|---|---|---|---|
| `--n1` | float | q_frac_wide | 구간 크기(q_frac 비율, `[0.35, 0.45]`) — hi=`[0,n1]`, mid=중앙 n1폭, lo=`[1-n1,1]` |
| `--n2` | float | q_frac_wide | 세그먼트 길이(q_frac 비율, `0 < n2 < n1`) |
| `--n-samples` | int | q_frac_wide / vqslope | 구간(존)당 세그먼트 수 |
| `--mode` | str | vqslope | 플래토 검출 모드 `dva`\|`ica` (기본 `dva`) |
| `--random-segment` | flag | q_frac_wide / vqslope | 구간 내 고정길이 랜덤 창 추출 옵션 켜기 (기본 `False`) |
| `--seg-len-pts` | int | q_frac_wide / vqslope | `--random-segment` 시 창의 고정 관측 포인트 수 (기본 20) |

`--random-segment`를 켜면 각 존(hi/mid/lo 또는 head/plateau/tail) 경계 자체는 그대로 두고, 그 안에서
`seg_len_pts` 길이의 창을 `n_samples`개 랜덤 위치에서 뽑는다(클리핑 금지 — 존이 창보다 짧으면 존 전체를
1개 세그먼트로 사용). 존 전체 대비 어느 창에도 안 뽑힌 포인트 비율은 `_4_data_hi/{axis}/{tag}/coverage_stats.txt`에
저장된다. 자세한 설계 근거는 `docs/SCENARIO_STRATEGY.md`의 "공통 옵션: random_segment" 절 참조.

```powershell
# 예: q_frac_wide, n1=45%, n2=20%, 구간당 4개, 랜덤 창(20포인트 고정) 사용
python run_pipeline.py 4 --seg-axis q_frac_wide --n1 0.45 --n2 0.2 --n-samples 4 --random-segment --seg-len-pts 20

# 예: vqslope, ica 모드, 구간당 2개
python 4_hi_analysis/hi_correlation.py --seg-axis vqslope --mode ica --n-samples 2
```

### `run_pipeline.py` (오케스트레이터)

위치 인자:

| 인자 | 타입/기본값 | 설명 |
|---|---|---|
| `from_step` | int, 기본 1 | 시작 스텝 번호 (1~9) |

옵션:

| 플래그 | 타입/기본값 | 설명 |
|---|---|---|
| `--to-step` | int, 기본 None(끝까지) | 종료 스텝 번호(포함) |
| `--workers` | int, 기본 `min(8, cpu_count)` | 데이터 스텝(1~5)에 전달할 병렬 프로세스 수 |
| `--model-config` | str, 기본 `5_model/config/scr.yaml` | 모델 학습/평가 설정 파일 |
| `--gates-from` | str, 기본 None | Phase 2(Step 7) gates 디렉터리 직접 지정 (미지정 시 Step 6 출력 자동 탐색 — **7부터 단독 실행 시 반드시 명시**, 안 그러면 `_5_data_model_scr` 전체에서 최근 수정 폴더를 잘못 집을 수 있음) |
| `--checkpoint` | str, 기본 None | 평가(Step 9) 체크포인트 직접 지정 (미지정 시 Step 7 출력 자동 탐색) |
| `--seg-axis` | str, 기본 None | 세그멘테이션 축, Step 4~7에 전달 |
| `--axis-config` | str(JSON), 기본 None | 축 파라미터 JSON, Step 4~7에 전달 |
| `--n1`/`--n2`/`--n-samples`/`--mode`/`--random-segment`/`--seg-len-pts` | 위 "공통: 축 단축 인자" 참고 | `--axis-config` 대체(PowerShell 호환) |

```powershell
python run_pipeline.py                      # 전체(Step 1~9)
python run_pipeline.py 6                    # 학습+평가만(Step 6~9)
python run_pipeline.py 6 --to-step 7         # Phase 1~2만
python run_pipeline.py 8 --to-step 8         # 분류기 학습만
python run_pipeline.py 9 --checkpoint _5_data_model_scr/.../checkpoints/best.pt   # 평가만
python run_pipeline.py --workers 4           # 병렬 수 지정(Step 1~5 전용)
```

### Step 4 — `4_hi_analysis/hi_correlation.py`

| 플래그 | 타입/기본값 | 설명 |
|---|---|---|
| `--seg-axis` | str, 기본 `qfrac` | `qfrac`\|`protocol`\|`vwindow`\|`rcs`\|`cluster`\|`q_frac_wide`\|`vqslope`\|`full_cycle`(부분 사이클 대비 베이스라인, 방향당 전체 curve 1개) |
| `--axis-config` | str(JSON), 기본 `"{}"` | 축 파라미터 JSON 문자열. PowerShell에서는 `--axis-config=$cfg` 형태 또는 아래 단축 인자 사용 |
| `--n1`/`--n2`/`--n-samples`/`--mode`/`--random-segment`/`--seg-len-pts` | 위 "공통: 축 단축 인자" 참고 | `--axis-config` 대체 |
| `--workers` | int, 기본 `cpu_count-2` | 병렬 프로세스 수 |
| `--force` | flag | 캐시 무시하고 HI 재추출 |
| `--dataset` | str, 기본 `MIT` | 디버그/시각화 대상 데이터셋(`MIT`\|`HUST`) |
| `--cell` | str, 기본 `""` | 디버그 시각화 대상 셀 ID |
| `--cycle` | int, 기본 0 | 디버그 시각화 대상 사이클(0=첫 유효 사이클) |
| `--n-top` | int, 기본 4 | 산점도에 표시할 상위 HI 수 |
| `--plateau-debug` | flag | 단일 사이클 플래토 판정 디버그 플롯 생성 후 종료 |
| `--plateau-summary` | flag | 전체 데이터 plateau_frac 요약 플롯 생성 후 종료 |
| `--curve-debug` | flag | 6-세그먼트 × 5-커브 시각화(HI 유효성 검증) |
| `--cycles` | str, 기본 `""` | `--curve-debug` 대상 사이클(쉼표 구분, 예: `1,100,300,500`) — 미지정 시 `--n-cycles` 개수만큼 자동 선택 |
| `--n-cycles` | int, 기본 5 | `--curve-debug` 자동 선택 사이클 수 |

```powershell
python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide --force --workers 8
python 4_hi_analysis/hi_correlation.py --seg-axis full_cycle --force --workers 8
python 4_hi_analysis/hi_correlation.py --seg-axis vqslope --mode dva --n-samples 1
```

### Step 5 — `4_hi_analysis/hi_segment_viz.py`

| 플래그 | 타입/기본값 | 설명 |
|---|---|---|
| `--seg-axis` | str, 기본 `qfrac` | `qfrac`\|`protocol`\|`vwindow`\|`rcs`\|`cluster`(q_frac_wide/vqslope도 단축 인자로 지원) |
| `--axis-config` | str(JSON), 기본 `"{}"` | 축 파라미터 JSON |
| `--n1`/`--n2`/`--n-samples`/`--mode`/`--random-segment`/`--seg-len-pts` | 위 "공통: 축 단축 인자" 참고 | `--axis-config` 대체 |
| `--workers` | int, 기본 4 | HI 추출 병렬 워커 수 |
| `--n-cycles` | int, 기본 4 | 세그먼트 cuts 플롯 대표 사이클 수 |
| `--force` | flag | 캐시 무시하고 HI 재추출 |

### Step 6/7 — `5_model/train_scr.py`

| 플래그 | 타입/기본값 | 설명 |
|---|---|---|
| `--config` | str, 기본 `5_model/config/scr.yaml` | 설정 파일 |
| `--phase` | int, `{1,2}`, 기본 None | 1=Gate 학습(HI 서브셋 선정), 2=분류·회귀 정밀 학습. 미지정 시 `--no-gates`/yaml `gates_from` 유무로 자동 결정 |
| `--charge-m` | int, 기본 None | Phase 2 충전 probe 상위 m개 (yaml `charge_probe_m` 오버라이드) |
| `--discharge-m` | int, 기본 None | Phase 2 방전 probe 상위 m개 (yaml `discharge_probe_m` 오버라이드) |
| `--scen-k` | int, 기본 None | 시나리오별 scen HI 수 오버라이드 (yaml `scen_k_count`) |
| `--gates-from` | str, 기본 None | Phase 2 시 이전 run 폴더 경로(gates JSON 자동 탐색). 미지정 시 yaml `data.gates_from` 사용 |
| `--no-gates` | flag | [legacy] `--phase 1`과 동일 |
| `--seg-axis` | str, 기본 None | 세그멘테이션 축. 미지정 시 yaml `scenario.axis` 또는 `qfrac` |
| `--axis-config` | str(JSON), 기본 None | 축 파라미터 JSON |
| `--seed` | int, 기본 None | 재현성 시드(yaml 오버라이드) |
| `--device` | str, 기본 `auto` | `auto`\|`cpu`\|`cuda` |

`model.regression_model` 값에 따라 회귀 헤드가 바뀐다(yaml `model:` 섹션, CLI 옵션 아님):
`mlp`(기본) / `transformer` / `i_transformer` / `resnet_tab` / `ft_transformer` / `raw_mlp`(HI 없이 raw_v/raw_i만
사용하는 베이스라인 — §"베이스라인 비교" 1-B 참조).

```powershell
python 5_model/train_scr.py --phase 1
python 5_model/train_scr.py --phase 2 --gates-from _5_data_model_scr/0708_1533
python 5_model/train_scr.py --phase 1 --charge-m 3 --discharge-m 1 --scen-k 5
```

### Step 8 — `5_model/train_classifier.py`

| 플래그 | 타입/기본값 | 설명 |
|---|---|---|
| `--config` | str, 기본 `5_model/config/scr.yaml` | 설정 파일 |
| `--run-dir` | str, 기본 None | Phase 2 run 폴더 경로(분류기를 `{run_dir}/classifier/`에 저장). 미지정 시 `output_dir` 내 최신 폴더 |
| `--epochs` | int, 기본 None | 최대 에폭(yaml `training.epochs` 오버라이드) |
| `--lr` | float, 기본 None | 학습률(yaml `training.lr` 오버라이드) |
| `--d-hidden` | int, 기본 64 | `MLPProbeClassifier` hidden dim |
| `--device` | str, 기본 `auto` | `auto`\|`cpu`\|`cuda` |

`raw_mlp` 모드 run(`config.yaml`의 `model.regression_model == "raw_mlp"`)이면 라우팅 자체가 무의미해
자동으로 학습을 건너뛴다(§"베이스라인 비교" 1-B 참조).

### Step 9 — `5_model/test_scr.py`

| 플래그 | 타입/기본값 | 설명 |
|---|---|---|
| `--config` | str, 기본 `5_model/config/scr.yaml` | 설정 파일 |
| `--checkpoint` | str, 기본 None | 평가할 체크포인트(미지정 시 최신 run 자동 탐색) |
| `--classifier-ckpt` | str, 기본 None | 시나리오 분류기 체크포인트. 미지정 시 `run_dir/classifier/clf_best.pt` 자동 탐색 |
| `--rep-cells` | str 여러 개(`nargs="+"`), 기본 None | 용량 곡선 플롯용 대표 셀 ID 목록 |
| `--device` | str, 기본 `auto` | `auto`\|`cpu`\|`cuda` |

분류기 체크포인트가 있으면 oracle/hard/soft 3모드를 모두 평가하고, 없으면(또는 `raw_mlp` run이면) 자동으로
oracle 모드만 실행한다. `test.random_segment_test: true`(yaml)면 test_rs 데이터로 동일 3모드를 추가 평가한다.

### 보조 도구 — `4_hi_analysis/seg_diagnose.py` (세그먼트 진단·비교)

| 플래그 | 타입/기본값 | 설명 |
|---|---|---|
| `--seg-axis` | str, 기본 None | 세그멘테이션 축(미지정 시 `hi_features*.pkl` 자동 탐색해 모든 축 순회) |
| `--axis-config` | str(JSON), 기본 `"{}"` | 축 파라미터 JSON |
| `--dataset` | str, 기본 `MIT`, `choices=[MIT,HUST,mit,hust]` | 통계 스캔 대상 데이터셋 |
| `--cell` | str, 기본 `""` | 사이클 플롯 대상 셀 ID(미지정 시 첫 번째 셀) |
| `--cycle` | int, 기본 0 | 사이클 플롯 대상 사이클(0=첫 유효 사이클) |
| `--mode` | str, 기본 `segment`, `choices=[segment,ic,vqzone,compare,all]` | `segment`(기본, V-t/V-Q 세그먼트 밴드) \| `ic`(IC 커브+창 경계) \| `vqzone`(vqslope 존 분리 근거, vqslope 전용) \| `compare`(여러 축/파라미터 조건 비교, `--cell` 필수) \| `all` |
| `--compare-config` | str, 기본 None | `--mode compare` 전용: 비교 조건 목록 JSON 경로(기본 `4_hi_analysis/compare_conditions.json`) — `[{"axis":..., "axis_config":{...}, "label":"(선택)"}]` 형식 |
| `--n-cycles` | int, 기본 6 | `ic` 모드 대표 사이클 수 |
| `--n-random` | int, 기본 10 | `--cell` 미지정 시 사용할 랜덤 셀 수 |
| `--seed` | int, 기본 42 | 랜덤 셀/사이클 선택 재현성 시드 |
| `--no-stats` | flag | 통계 수집·출력 생략 |
| `--no-plot` | flag | 사이클 시각화 생략 |
| `--survival-stats` | flag | q_frac_wide 전용: MIT+HUST 생존율/시간길이/전압길이 비교 통계(`--seg-axis q_frac_wide` 필수, `--dataset`/`--no-plot` 무시) |
| `--n1`/`--n2`/`--n-samples` | float/float/int, 기본 None | `--survival-stats`용 q_frac_wide 파라미터 |

```powershell
# 동일 셀·사이클을 여러 조건(축/파라미터)으로 나란히 비교
python 4_hi_analysis/seg_diagnose.py --mode compare --dataset MIT --cell b1c11 --cycle 400

# vqslope의 head/plateau/tail 분리 근거 시각화
python 4_hi_analysis/seg_diagnose.py --seg-axis vqslope --mode vqzone --cell 10-1 --cycle 500
```

### 보조 도구 — `5_model/visualize_results.py` (다중 run 비교)

| 플래그 | 타입/기본값 | 설명 |
|---|---|---|
| `--runs` | str 여러 개(`nargs="+"`), 필수 | 비교할 run 폴더 경로 2개 이상(동일 축·`scenario_names`이어야 함) |
| `--labels` | str 여러 개, 기본 None | run별 표시 이름(미지정 시 폴더명, `--runs`와 개수 일치 필요) |
| `--with-jacobian` | flag | 실 데이터 기반 Jacobian(gradient) 코사인 유사도 패널 추가(느림 — 데이터셋 재구축 필요) |
| `--checkpoint-name` | str, 기본 `best.pt` | run별 사용할 체크포인트 파일명(없으면 `final.pt` 폴백) |
| `--infer-batch-size` | int, 기본 256 | 인퍼런스 벤치마크 배치 크기 |
| `--infer-warmup` | int, 기본 10 | 인퍼런스 벤치마크 워밍업 반복 수 |
| `--infer-reps` | int, 기본 50 | 인퍼런스 벤치마크 측정 반복 수 |
| `--jacobian-max-samples` | int, 기본 300 | 시나리오/전체당 gradient 계산 최대 샘플 수 |
| `--device` | str, 기본 `auto` | `auto`\|`cpu`\|`cuda` |
| `--out-name` | str, 기본 None | 결과 폴더명 오버라이드(기본 `<MMDD_HHMM>_result_comparison`) |

결과 PNG는 RMSE/MAE/MAPE(시나리오별) + 스칼라 지표 7종 + 선정 HI 자카드 유사도(이진/확률 가중 2종,
run별 시나리오×시나리오 행렬) [+ `--with-jacobian` 시 Jacobian 코사인 유사도]로 구성된다.

```powershell
python 5_model/visualize_results.py --runs _5_data_model_scr/RUN1 _5_data_model_scr/RUN2 --labels mlp resnet
```

### 보조 도구 — `5_model/finetune_scr.py` (Few-shot fine-tuning)

| 플래그 | 타입/기본값 | 설명 |
|---|---|---|
| `--checkpoint` | str, 필수 | Phase 2 체크포인트 경로(`best.pt`) |
| `--finetune-dataset` | str, 기본 `MIT` | fine-tuning 대상 데이터셋 |
| `--finetune-cells` | int, 기본 5 | few-shot 학습에 사용할 셀 수(N-shot) |
| `--val-split` | float, 기본 0.2 | 나머지 셀 중 검증용 비율 |
| `--epochs` | int, 기본 100 | fine-tuning 에폭 수 |
| `--lr` | float, 기본 1e-4 | 학습률 |
| `--seed` | int, 기본 42 | 재현성 시드 |
| `--config` | str, 기본 None | yaml 오버라이드(미지정 시 체크포인트에 저장된 config 사용) |
| `--device` | str, 기본 `auto` | `auto`\|`cpu`\|`cuda` |

### 보조 도구 — `5_model/clustering.py` (HI 클러스터링·mRMR 분석)

| 플래그 | 타입/기본값 | 설명 |
|---|---|---|
| `--k-max` | int, 기본 9 | K-Means 최대 k |
| `--n-init` | int, 기본 5 | `MiniBatchKMeans` n_init |
| `--knn` | int, 기본 10 | KNN Purity 이웃 수 |
| `--mrmr-m` | int, 기본 20 | mRMR로 선택할 최대 HI 수 |
| `--no-plot` | flag | 플롯 생성 생략 |
| `--n-jobs` | int, 기본 -1 | joblib 병렬 수(-1=전체 코어) |
| `--out` | str, 기본 `docs/_clustering_results.json` | 결과 JSON 경로 |

---

## 1. 데이터 변환 — `1_convert/convert_unified.py`

| 항목 | 내용 |
|---|---|
| 입력 | MIT: `.mat` (HDF5, MATLAB v7.3) / HUST: `.pkl` |
| 출력 | `_0_data_raw/` (원본 복사) + `_1_data_unified/{MIT,HUST}/*.pkl` |
| 형식 | wide DataFrame: 한 행 = 한 사이클. 컬럼: `cycle`, `capacity_Ah`, 원시 V/I/t 시계열 |
| 주요 로직 | MIT HDF5 그룹 구조 파싱 (`b1cXX`, `b2cXX`, `b3cXX`), HUST 피클 직접 로드 |

```powershell
# 권장 — MIT + HUST 전체 변환 (_0_data_raw/ 캐시 있으면 자동 사용)
python 1_convert/convert_unified.py --dataset all --workers 3

# 캐시 무시하고 원본부터 재파싱
python 1_convert/convert_unified.py --dataset all --no-cache

# 데이터셋 개별 변환
python 1_convert/convert_unified.py --dataset mit
python 1_convert/convert_unified.py --dataset hust

# HUST 단일 셀 테스트
python 1_convert/convert_unified.py --dataset hust --cell 1-1
```

---

## 2. 전처리 — `2_preprocess/preprocess.py`

7단계 순차 필터 (사이클 레벨):

| 단계 | 필터 |
|---|---|
| F1 | 빈 사이클 제거 (시계열 길이 0) |
| F2 | 시간 단조성 검사 (역전 제거) |
| F3 | Rest 구간 0전류 행 제거 |
| F4 | 단절 사이클 제거 (방전 단절 → 사이클 전체 / 충전 단절 → 충전 행만) |
| F5 | Rolling Median 2-pass 용량 이상치 제거 (w=11/31, σ=2.0) |
| F6 | 종지전압 하한 검사 (vend_min=1.8 V) |
| F7 | V-q_frac 형상 편차 필터 (MAD robust z > σ, KNOWN_SHAPE_ANOMALIES 수동 지정 포함) |

출력: `_4_data_hi/clean/{MIT,HUST}/*.pkl` (V/I/t 시계열 보존)

```powershell
# 권장 — MIT + HUST 전체 전처리 (병렬 4 프로세스)
python 2_preprocess/preprocess.py

# 데이터셋 개별 처리
python 2_preprocess/preprocess.py --dataset mit
python 2_preprocess/preprocess.py --dataset hust

# 이상치 필터 완화 (σ 높이기) — 데이터 보존 우선
python 2_preprocess/preprocess.py --sigma 3.0 --shape-sigma 50.0

# 단절 기준 조정 — 짧은 갭 허용
python 2_preprocess/preprocess.py --dis-gap-s 300 --chg-gap-s 300

# 프로세스 수 지정
python 2_preprocess/preprocess.py --workers 2
```

---

## 3. 무결성 검사 — `3_integrity/check_integrity.py`

10개 항목 자동 검사 후 CSV 저장:

| 수준 | criterion |
|---|---|
| 셀 | `missing_cols`, `invalid_phase`, `current_direction`, `capacity_increasing`, `high_nan`, `cycle_count_mismatch` |
| 사이클 | `voltage_high` (>4.5V), `voltage_low` (<1.5V), `rest_dominant` (>80%), `time_nonmono` |

출력: `3_integrity/outputs/integrity_report.csv` / `integrity_issues.csv`

```powershell
# 권장 — 병렬 4 프로세스 (기본값)
python 3_integrity/check_integrity.py

# 프로세스 수 지정
python 3_integrity/check_integrity.py --workers 2
```

---

## 4. HI 추출 — `4_hi_analysis/hi_correlation.py`

### 세그멘테이션 — `common/scenario/`

Step 4의 세그먼트 분할 로직은 `common/scenario/` 패키지의 `Segmenter` ABC로 추상화되어 있다.  
`--seg-axis` 인수로 축을 선택하며, `get_segmenter(name, cfg)` 팩토리로 인스턴스를 생성한다.

| 축 | 클래스 | 기본 시나리오 수 | 특징 |
|---|---|---|---|
| `qfrac` | `QFracSegmenter` | 6 | q_frac 3분할 × 방향. 현행 DIS_SEGS/CHG_SEGS 동작을 완전히 재현 |
| `protocol` | `ProtocolSegmenter` | 가변 | CC 전환점 감지, 프로토콜 단계별 분할 |
| `vwindow` | `VWindowSegmenter` | **2·n+1** | **LFP 고정 전압 경계** + 충전 CC/CV 분리. `from_lfp()` 기본 사용; `fit()` deprecated |
| `rcs` | `RCSSegmenter` | 가변 | 랜덤 부분 관측 샘플러 (Deng 2022). **충전 CC 구간만** 샘플링, CV 분리 적용 |
| `cluster` | `ClusterSegmenter` | 가변 | 사전 학습 클러스터 ID 기반 |

**`ScenarioSpec`** — Step 4 출력 / Step 5 입력 계약 아티팩트 (`scenario_spec.json`):
```python
@dataclass
class ScenarioSpec:
    axis: str               # "qfrac" | "protocol" | ...
    n_scenarios: int        # Stage B gate 개수
    scenario_names: list    # 길이 = n_scenarios
    n_classes: int          # Stage A 분류 클래스 수
    class_names: list       # 길이 = n_classes
    routing: list[list[int]]# routing[dir_idx][latent_class] = scenario_id
    classifier_default: str # "mlp_probe" | "rule" | "centroid" | "none"
```

qfrac 기본 라우팅:
```
routing = [[0, 1, 2], [5, 4, 3]]
  dir_idx=0(충전): Low→chg_lo(0), Mid→chg_mid(1), Hi→chg_hi(2)
  dir_idx=1(방전): Low→dis_lo(5), Mid→dis_mid(4), Hi→dis_hi(3)
```

**`SegmentRecord`** — Segmenter → HI 추출기 사이 중간 표현:
```python
@dataclass
class SegmentRecord:
    scenario_id: int     # Stage B gate 라우팅 타깃
    latent_class: int    # Stage A 분류 타깃
    direction: int       # +1=충전 / -1=방전
    v, i, dt, q: np.ndarray  # 슬라이스된 배열
    meta: dict           # {"seg_name": "dis_hi", "q_frac_lo": 0.0, ...}
```

---

### 4-0. VWindow 세그멘테이션 — `common/scenario/vwindow.py`

#### 설계 원칙

| 구간 | 방법 | 근거 |
|---|---|---|
| **방전 / 충전 CC** | **LFP 고정 전압 경계** | 플래토 전이 전압 기준 고정 → 충전 프로토콜(단일/다단계 CC)과 무관. 노화 사이클에서 동일 전압 구간 내 Ah 감소 자체가 SOH 신호 (Dubarry et al. 2012, Hu et al. 2020) |
| **충전 CV** | 독립 시나리오 `chg_cv` | CC/CV는 thermodynamic vs kinetic 체제로 물리가 다름. CV 지속시간·Q_CV/Q_total은 노화 지표 (Ma et al. 2018, Plett 2015) |

> **이전 방식(equal-Ah 피팅) 폐기 이유**: BOL 첫 사이클 V-Q 곡선에서 분위수 보간으로 경계를 계산하는 방식은 MIT 6C→1C→CV 같은 다단계 CC 프로토콜에서 V-Q 비단조성이 발생해 충전 경계가 프로토콜 구조에 강하게 의존하는 문제가 있었다. `fit()` 메서드는 단일-CC 데이터셋 호환성을 위해 유지되지만 `UserWarning`을 발생시킨다.

#### LFP 기본 전압 경계 (n_windows=3)

| 구간 | 윈도우 | 범위 [V] | 전기화학적 의미 |
|---|---|---|---|
| **충전 CC** | chg_win0 | 2.80 – 3.38 | 사전-플래토 급경사. 다단계 CC 전류 전환 포함 |
| | chg_win1 | 3.38 – 3.47 | 핵심 플래토. 총 용량 ~70% 집중 |
| | chg_win2 | 3.47 – 3.60 | 플래토 탈출 + CV 진입 직전 |
| **방전** | dis_win0 | 2.50 – 3.10 | 플래토 종료 + 급강하. 용량 페이드 끝단 |
| | dis_win1 | 3.10 – 3.35 | 하부 플래토 중심. 2상 공존 영역 |
| | dis_win2 | 3.35 – 3.65 | 상부 플래토 + 숄더. 노화 초기 특징 |

#### 시나리오 구조 (n_windows=3 기준, 총 7개)

```
인덱스  이름       방향   물리 의미
─────────────────────────────────────────
0       chg_win0   +1     충전 CC 저전압 구간   [2.80–3.38 V]
1       chg_win1   +1     충전 CC 중전압 구간   [3.38–3.47 V]
2       chg_win2   +1     충전 CC 고전압 구간   [3.47–3.60 V]
3       chg_cv     +1     충전 CV 구간 (정전압 테이퍼)
4       dis_win0   -1     방전 저전압 구간      [2.50–3.10 V]
5       dis_win1   -1     방전 중전압 구간      [3.10–3.35 V]
6       dis_win2   -1     방전 고전압 구간      [3.35–3.65 V]
─────────────────────────────────────────
n_scenarios = 2·n_windows + 1
n_classes   = n_windows + 1   (CC 클래스 0..n-1, CV 클래스 n)
```

#### CC→CV 전환 감지 (`_detect_cv_start`)

```python
# V ≥ cv_v_thresh(3.60 V) AND I < cv_cc_frac(0.80) × I_max 를 동시에 만족하는 첫 인덱스
cv_start = _detect_cv_start(chg_v, chg_i, cv_v_thresh=3.60, cv_cc_frac=0.80)
# CC 구간: [:cv_start] → 고정 전압 윈도우
# CV 구간: [cv_start:] → chg_cv 시나리오 (min_pts 미만이면 건너뜀)
```

#### 고정 경계 주입 — `hi_correlation.py`

`load_or_extract()` 내부에서 `axis == "vwindow"` 이고 `dis_edges`가 `axis_cfg`에 없을 때
`VWindowSegmenter.from_lfp()`로 LFP 기본 경계를 `axis_cfg`에 주입한다.

```python
_tmp = VWindowSegmenter.from_lfp(n_windows=n_win)
axis_cfg.setdefault("dis_edges", _tmp._dis_edges)   # [2.50, 3.10, 3.35, 3.65]
axis_cfg.setdefault("chg_edges", _tmp._chg_edges)   # [2.80, 3.38, 3.47, 3.60]
# → VWindowSegmenter(**axis_cfg) 로 인스턴스 생성
# → ScenarioSpec.save() → scenario_spec.json 에 edges 포함 저장
```

#### VWindowSegmenter API

```python
# 권장: LFP 고정 경계 (프로토콜 무관)
seg = VWindowSegmenter.from_lfp(n_windows=3, cv_v_thresh=3.60, cv_cc_frac=0.80)

# 커스텀 경계 직접 지정
seg = VWindowSegmenter(dis_edges=[2.5, 3.1, 3.35, 3.65],
                       chg_edges=[2.8, 3.38, 3.47, 3.60])

# 레거시: BOL 데이터 기반 equal-Ah 피팅 (단일-CC 전용, UserWarning 발생)
seg = VWindowSegmenter(n_windows=3)
seg.fit(first_cycle_data)   # UserWarning: 다단계 CC에서 V-Q 비단조성 위험
```

---

### 4-1a. RCS 세그멘테이션 — `common/scenario/rcs.py`

Deng et al. (2022, *Nature Communications*) 방식의 랜덤 부분 관측 샘플러.

**핵심 파라미터:**
```python
RCSSegmenter(
    n_samples=6,      # 사이클당 랜덤 윈도우 수
    window=0.3,       # 윈도우 폭 (q_frac 기준, 0.3 = 전체 용량의 30%)
    assign="position_bin",  # "position_bin": center q_frac → lo/mid/hi 분류
                            # "none": 방향만 구분 (Deng 원안)
    cv_v_thresh=3.60, # CC→CV 전환 전압 임계값 [V]
    cv_cc_frac=0.80,  # CC→CV: I < cv_cc_frac × I_max
)
```

**CC/CV 분리 (충전):**  
`iter_segments()` 내부에서 `_detect_cv_start()`(vwindow 공유 함수)로 CV 시작점을 감지하고 **CC 구간만** `_sample_segments()`에 전달한다.  
근거: CC에서 q_frac ≈ t_frac ≈ SOC_frac (정전류)이므로 0.3 q_frac 윈도우가 균일한 물리적 의미를 가짐. CV 포함 시 동일 q_frac 윈도우의 시간 길이가 10배 이상 달라져 HI 분포 오염.

**윈도우 중심 분포 제약:**  
`start_qf ∈ [0.0, 0.7]` → `center_qf ∈ [0.15, 0.85]`.  
q_frac [0.00, 0.15] 및 [0.85, 1.00] 구간은 윈도우 중심이 될 수 없음 (구조적 특성).

---

### 4-1b. 랜덤세그먼트 평가 데이터 — `tmp_make_test_rs.py`

`hi_correlation.py` 파이프라인을 우회해 완전 랜덤 구간·랜덤 길이 세그먼트를 직접
추출하는 평가 전용 데이터 생성기. 실배포 시(정확한 q_frac 경계를 보장할 수 없는
상황)를 시뮬레이션하기 위한 것으로, `test.random_segment_test: true` 설정 시
`test_scr.py`가 소비한다 (§5-10/5-11).

> **2026-07 변경**: 통합 평가(oracle/hard/soft)에서 분류 성능을 실제로 측정하려면
> 정답 레짐 라벨과 CNN 분류기 입력(raw_v/raw_i)이 필요하다는 게 드러나
> (§5-11 참조), 스크립트에 두 가지를 추가했다. **코드만 수정했고 재생성은 아직
> 실행하지 않았다** — 기존 `_4_data_hi/test_rs/` 캐시는 구 형식(2-시나리오,
> 라벨 없음) 그대로다.

**추가 1 — 위치기반 레짐 라벨 (`_assign_regime`)**

세그먼트 중심 q_frac을 3분위(tercile)로 나눠 q_frac_wide 6-시나리오 모델공간과
동일한 `segment_id`(0~5)·`scen` 값을 배정한다 (기존은 충/방전 2종류만 구분,
`segment_id ∈ {0,1}`이라 레짐 정답이 없었다):

```python
def _assign_regime(center_qf, is_charge):
    b = 0 if center_qf < 1/3 else (1 if center_qf < 2/3 else 2)
    if is_charge:
        return b, b + 1                 # chg_lo(0) / chg_mid(1) / chg_hi(2)
    return 3 + b, -(3 - b)              # dis_hi(3) / dis_mid(4) / dis_lo(5)
```

`_build_rs_spec()`이 q_frac_wide와 동일한 `routing=[[0,1,2],[5,4,3]]`의
6-시나리오 `ScenarioSpec`을 생성 — 재생성 후 `test_scr.py`가 `direction_routing`
보정 없이 바로 oracle 라우팅 가능 (`rs_spec.n_scenarios == model.n_scenarios`로
자동 감지, §5-11).

**추가 2 — 원시 곡선 (`raw_v`/`raw_i`)**

각 세그먼트에서 `hi_correlation._resample_segment()`(§4 참조, RAW_N=48)를 호출해
CNN 분류기 입력을 함께 저장한다. 이전에는 HI 64개만 저장했다.

```python
rv, ri = _WORKER_HIC._resample_segment(v[m], i[m], q[m])
seg_rows.append({..., "raw_v": rv.tolist(), "raw_i": ri.tolist(), ...})
```

**재생성 방법 (실행 시):**
```bash
python tmp_make_test_rs.py --force
# → _4_data_hi/test_rs/seg/{MIT,HUST}/*.pkl  (raw_v/raw_i + segment_id 0..5 포함)
# → _4_data_hi/test_rs/scenario_spec.json    (n_scenarios=6, q_frac_wide 라우팅과 동일)
```
재생성 전까지는 `random_seg_test`의 hard/soft 분류 정확도가 `classification_metrics`의
degenerate 가드에 걸려 `{"note": "..."}`로만 보고된다 (회귀 지표는 정상).

---

### 4-2. 세그멘테이션 축별 분할 비교

> **공통 기준 사이클**: LFP 방전 1C (3.65 V→2.50 V) / 충전 CC 1C (2.80 V→3.60 V)  
> MIT 충전은 6C→1C→CV 다단 프로토콜, HUST는 1C 단일 CC→CV.

#### LFP 기준 사이클 형상

```
[방전] V vs q_frac (1C)              [충전 CC] V vs q_frac
V(V)                                 V(V)
3.65 ┤╮ ← 고전압 숄더                 3.60 ┤           ╭─ (CV 시작)
3.40 ┤╰──╮                           3.47 ┤      ╭────╯
     │   ╰──────────────────╮  플래토 3.38 ┤─────╯  ← 플래토 (~3.38–3.47V)
3.10 ┤                      ╰──╮     3.00 ┤╭╯  ← MIT 6C→1C 전환 dip
2.50 ┤                          ╰─    2.80 ┤╯
     └────────────────────────────         └───────────────────────────
     0%      15%           75% 100%        0%   25%          80%  100%
     │← 숄더 │← 플래토(~60%) →│← 말단 │   │CC1 6C│←── CC2 1C ──│← CV │
```

---

#### 축 1. qfrac — 누적 용량(Ah) 등분할

경계: q_frac = 0 / **0.4** / **0.7** / 1.0

```
[방전] q_frac 기준
 0%         40%        70%      100%
 ├──────────┼──────────┼─────────┤
    dis_hi     dis_mid    dis_lo
  (≈3.65→    (≈3.15→   (≈3.15→
   3.20V)     3.10V)    2.50V)

[충전] q_frac 기준
 0%         40%        70%      100%
 ├──────────┼──────────┼─────────┤
    chg_lo     chg_mid    chg_hi
  (≈2.80→    (≈3.35→   (≈3.45V+
   3.35V)     3.45V)    CV 포함)
```

**특징 / 주의사항:**
- 어떤 프로토콜·화학계에도 적용 가능한 가장 범용적인 분할
- 정전류(CC) 환경에서 q_frac ≈ t_frac → 세그먼트 간 행 수 균형 보장
- LFP 플래토(전체 용량의 ~70%)가 특정 세그먼트 하나에 집중 → 전기화학적으로 이질적인 구간이 묶임
- 충전 chg_hi에 CV 구간이 섞여 들어갈 수 있음 (CV 미분리)

---

#### 축 2. protocol — 전류 단계(CC→CC→CV) 경계

경계: |ΔI| > 0.5C 급변점 자동 감지, max_steps=3 캡

```
[MIT 충전: 6C → 1C → CV]
│←CC1 6C (~25% Ah)→│←──CC2 1C (~55% Ah)──→│←CV (~20%)→│
      chg_step0            chg_step1            chg_step2
     (≈2.80→3.40V,        (≈3.00→3.60V,       (V≈3.60V,
      고전류 고편차)        1C 플래토 통과)       I 감쇠)

[HUST 충전: 1C → CV]
│←────────── CC 1C (~85% Ah) ──────────────→│←CV (~15%)→│
                  chg_step0                    chg_step1

[방전: 단일 CC]
│←─────────────── CC 1C or 2C ─────────────────────────→│
                          dis_step0
```

**특징 / 주의사항:**
- 프로토콜 구조를 직접 반영 → 동일 step 내 전류 균일 → HI 해석 명확
- 크로스 데이터셋 위험: MIT chg_step1(1C CC2) ≠ HUST chg_step0(1C CC) — 시나리오 ID가 같아도 물리 구간이 다를 수 있음
- HUST는 step이 2개뿐이어서 chg_step2가 항상 비어 NaN → 모델이 미사용 gate를 학습해야 함
- n_scenarios = 6 (max_steps=3 × 방향 2) → qfrac과 동일 크기로 공정 비교 가능

---

#### 축 3. vwindow — LFP 고정 전압 경계

경계: 방전 [2.50 / 3.10 / **3.35** / 3.65 V], 충전 CC [2.80 / 3.38 / **3.47** / 3.60 V]

```
[방전] V 기준 분할
 V(V)
 3.65 ┤─────────────── dis_win2 ──────────────── (숄더 + 플래토 상부)
 3.35 ┤
      ├─────────────── dis_win1 ──────────────── (플래토 중심, ~50% Ah)
 3.10 ┤
      ├─────────────── dis_win0 ──────────────── (플래토 말단 + 급강하)
 2.50 ┤
      └──── (Ah 분포 불균일: win1 ≈ 50-60%, win0/win2 ≈ 20-25% 각각)

[충전 CC] V 기준 분할                         [충전 CV]
 3.60 ┤──────── chg_win2 ────────              chg_cv (별도 시나리오)
 3.47 ┤
      ├──────── chg_win1 ────────  ← 핵심 플래토 (~70% Ah 집중)
 3.38 ┤
      ├──────── chg_win0 ────────  ← MIT CC1(6C) 포함
 2.80 ┤
```

**특징 / 주의사항:**
- 동일 V 구간 = 동일 화학 상태 → 노화 시 해당 구간의 Ah 감소 = SOH 직접 신호 (Dubarry 2012)
- 충전 프로토콜 독립: CC1(6C)은 chg_win0에 편입되어 win1/win2를 오염하지 않음
- Ah 불균형: chg_win1에 전체 Ah의 ~70%가 집중 → win1 세그먼트가 압도적으로 많은 행 보유
- CC/CV 분리로 chg_cv가 독립 추출됨 (vwindow만의 특성)
- 크로스 데이터셋 이식성 최고: 전압값은 LFP 화학계로 고정 → MIT·HUST 동일 물리 구간

---

#### 축 4. rcs — 랜덤 부분 관측 샘플링 (CC 전용)

파라미터: n_samples=6, window=0.3, CC 구간만 (CV 제거)

```
[방전 또는 충전 CC q_frac 0~100%]  ← CV는 제거됨
 0%                                        100%
 │                                               │
 │  Sample 1:  [━━━ 0.3 ━━━]                   │  → center=0.25 → lo
 │  Sample 2:        [━━━ 0.3 ━━━]              │  → center=0.45 → mid
 │  Sample 3:               [━━━ 0.3 ━━━]       │  → center=0.65 → mid
 │  Sample 4:  [━━━ 0.3 ━━━]                   │  → center=0.22 → lo
 │  Sample 5:                    [━━━ 0.3 ━━━]  │  → center=0.79 → hi
 │  Sample 6:     [━━━ 0.3 ━━━]                │  → center=0.35 → lo
 │                                    ↑          │
 │                           max start=0.70      │
 │                                               │
  [0.00~0.15]: 윈도우 중심 불가    [0.85~1.00]: 윈도우 중심 불가
```

**특징 / 주의사항:**
- 1 사이클 → 6개 독립 훈련 샘플 (데이터 증강 효과, 타 축 대비 6배)
- 부분 관측(partial charge) 시뮬레이션 → EV 실제 환경 모델링 (Deng et al. 2022)
- CC에서 q_frac ≈ SOC_frac (정전류) → 윈도우 위치가 전기화학적 의미 가짐
- 양끝 q_frac [0.00-0.15], [0.85-1.00] 구간은 윈도우 중심 불가 → 해당 구간 HI 추정에 불리
- 동일 사이클 내 샘플 간 데이터 중복 가능 → 독립성 가정 위반 (미니배치 구성 시 주의)
- seed 고정 필수 (seed 다르면 경계 위치 달라짐 → 재현성 깨짐)

---

#### 축 5. cluster — 데이터 기반 K-means 군집

파라미터: n_fine=10 (10% 등분), k_range=(2,8), Gap statistic + 1-SE rule

```
[방전 q_frac 10분할 → 각 bin에서 (dQ_mean, dQ_std, dQ_skew) 추출 → K-means (예시 K=4)]

 q_frac: 0%  10% 20% 30% 40% 50% 60% 70% 80% 90% 100%
         [S0][S1][S2][S3][S4][S5][S6][S7][S8][S9]
 cluster: c0  c1  c2  c2  c3  c3  c3  c2  c1  c0
          ↑숄더           ↑플래토 중심            ↑말단
  → 숄더·말단(빠른 V변화)이 같은 군집(c0), 플래토(완만한 V변화)가 c3

 시나리오 배정 (예시):
  q_frac bin S0(c0) → scenario dis_c0
  q_frac bin S4(c3) → scenario dis_c3
  (10개 bin이 K개 시나리오로 병합됨)
```

**특징 / 주의사항:**
- 인간이 정의한 경계 없이 데이터 패턴으로 전기화학 유사 구간을 자동 군집화 (Ke 2026)
- K는 Gap statistic으로 자동 결정 → 과적합 방지, 단 실험마다 K가 달라질 수 있음
- `fit()`은 반드시 train 셀만 사용 (test 정보 누출 방지)
- 크로스 데이터셋 약점: train(MIT) 클러스터 패턴이 test(HUST)에 일반화되는지 별도 검증 필요
- n_scenarios = 2K → K 결정 전까지 모델 아키텍처 미확정 (동적 spec)

---

#### 비교 요약

| 축 | 경계 기준 | 시나리오 수 | 데이터 증강 | 크로스-DS 이식성 | Ah 균형 | 주요 약점 |
|---|---|---|---|---|---|---|
| qfrac | q_frac 등분 | 6 | ✗ | 높음 | 균일 | 플래토 이질성 미반영 |
| protocol | 전류 전환점 감지 | 6 (max_steps×2) | ✗ | 중간 | 불균일 | step 수 불일치 시 공 시나리오 |
| vwindow | LFP 고정 전압 | 7 | ✗ | **최고** | 불균일 | win1에 Ah 과집중 |
| rcs | 랜덤 q_frac 창 | 6 | ✓(×6) | 높음 | 균일(per window) | 양끝 커버리지 낮음 |
| cluster | K-means 자동 | 2K | ✗ | 낮음 | 불균일 | 크로스-DS 군집 일반화 미검증 |

---

### HI 구조 (총 411개 컬럼, 모델 입력 64개/세그먼트)

**Global HI (15개)** — 사이클 전체 특성:
```
q_dis, energy_dis, v_mean_dis, r_dc_est, q_plateau_frac,
ica_peak1_v/h/area/asym, dva_valley_q/depth,
ce, cv_q_frac, cv_time_frac, chg_ica_peak1_h
```

**Segment HI (qfrac 기준 6구간 × 66개 = 396개)**:

| 카테고리 | 수 | 컬럼 접두사 | 예시 |
|---|---|---|---|
| STAT | 20 | `stat_{k}_{seg}` | `v_mean`, `v_std`, `corr_qi`, `v_samp_ent` |
| DIFF | 20 | `diff_{k}_{seg}` | `dqdv_peak_h/v/w`, `dvdq_mean`, `d2vdq2_rms` |
| LFP  | 20 | `lfp_{k}_{seg}`  | `plateau_frac`, `plateau_dvdq_std`, `v_flatness`, `knee_v` |
| MORPH|  6 | `morph_{k}_{seg}`| `vt_dtw`, `vq_dtw`, `ve_dtw`, `*_frec` × 3 |

모델 입력: `stat_q_abs` + `stat_energy_seg` 누설 변수 제외 → **64 HI/세그먼트**

> **LFP L04 변경**: `plateau_q_frac` 제거 (`plateau_frac`과 수학적으로 동일 — `n_plt/n_b` 증명).  
> → `plateau_dvdq_std` 대체: 플래토 마스크 내 dV/dQ 표준편차. 플래토 평탄도를 직접 측정하며 다른 피처와 비중복.

---

### 추출 흐름 (`_extract_one_cell`)

```
1. phase 분리: dis = grp[phase=="discharge"], chg = grp[phase=="charge"]
2. q_cum 계산: 방전 / 충전 각각 독립 적산 (Ah)
3. Segmenter.iter_segments() 호출:
     방전: iter_segments(cell_id, cyc, v, i_mag, dt, q_cum)
     충전: iter_segments(..., np.empty(0)×4, vc, ic, dtc, qcc)
           ← 방전 인자를 np.empty(0)로 전달 → min_pts 체크로 자동 skip
4. SegmentRecord.meta["seg_name"] 또는 spec.scenario_names[scenario_id]
5. _seg_stat / _seg_diff / _seg_lfp / _seg_morph_curves 호출
6. _resample_segment(v, i, q) → raw_v_{seg}/raw_i_{seg} (q_frac [0,1] 그리드 48pt) 저장
7. row 딕셔너리 누적 → wide DataFrame (한 행 = 한 사이클)
```

> **원시 곡선 리샘플 (`_resample_segment`, RAW_N=48)**: 세그먼트 V/|I| 시계열을 세그먼트
> 내 상대 누적전하 q_rel을 [0,1]로 정규화한 그리드에 선형보간한다. 세그먼트 길이(포인트
> 수)가 셀·사이클·데이터셋마다 달라도 동일 해상도로 CNN에 입력 가능하게 한다.
> `_to_seg_df`가 `raw_v_{seg}`/`raw_i_{seg}`를 native-seg의 `raw_v`/`raw_i` 컬럼으로 전달.

multiprocessing 호환: `_extract_one_cell((pkl_path_str, axis, cfg_json))` 튜플 형태로 전달 (pickle 가능).

---

### 출력

```
_4_data_hi/{axis}/
  cycle/{ds}/{cell}.pkl    wide-cycle HI (한 행=한 사이클, stat/diff/lfp/morph × 세그먼트)
  seg/{ds}/{cell}.pkl      native-seg HI (한 행=한 세그먼트, hi_00..hi_63 + raw_v/raw_i[48])
  scenario_spec.json       해당 축의 ScenarioSpec
hi_features.pkl            qfrac 추출 캐시 (축 변경 시 hi_features_{axis}.pkl)
```

```powershell
# 권장 — qfrac 기본 축, 캐시 사용
python 4_hi_analysis/hi_correlation.py

# 캐시 무시하고 HI 재추출
python 4_hi_analysis/hi_correlation.py --force

# 다른 세그멘테이션 축
python 4_hi_analysis/hi_correlation.py --seg-axis vwindow
python 4_hi_analysis/hi_correlation.py --seg-axis protocol

# 축 파라미터 커스터마이즈 (qfrac q_frac 분할점 변경)
python 4_hi_analysis/hi_correlation.py --seg-axis qfrac --axis-config '{"dis_bounds":[0,0.3,0.6,1.0]}'

# 병렬 프로세스 수 지정
python 4_hi_analysis/hi_correlation.py --workers 8

# 단일 사이클 플래토 디버그 플롯 (시각화)
python 4_hi_analysis/hi_correlation.py --plateau-debug --dataset MIT --cell b1c0

# 6-세그먼트 × 5-커브 유효성 검증
python 4_hi_analysis/hi_correlation.py --curve-debug --dataset MIT --cell b1c0
```

---

### 4-1. 세그멘테이션 진단 도구 — `4_hi_analysis/seg_diagnose.py`

세그멘테이션 축의 균형·시각적 품질을 빠르게 진단하는 독립 스크립트.

**3가지 출력 모드:**

| 모드 | 내용 |
|---|---|
| `--no-plot` | 시나리오별 세그먼트 수·평균 행 수·평균 시간(s) 통계 출력 |
| `--mode segment` (기본) | 사이클 V-t + V-Q(방전) + V-Q(충전) 세그먼트 색상 오버레이 |
| `--mode ic` | 멀티사이클 V-Q + dQ/dV 커브 (vwindow: 창 경계 밴드 표시) |
| `--mode all` | segment + ic 둘 다 생성 |

**랜덤 셀 × 랜덤 사이클 시각화 (기본 동작):**

`--cell` 미지정 시 데이터셋에서 `--n-random`(기본 10)개 셀을 무작위로 뽑고,
각 셀에서 유효 사이클(방전 ≥ 30 행) 중 하나를 무작위 선택 → 셀당 PNG 1개 생성.  
`--seed`로 재현 가능.

```powershell
# 기본: 랜덤 10셀 × 랜덤 사이클 (segment 모드)
python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow

# 재현 가능한 샘플링
python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --seed 42

# 셀 수 지정
python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --n-random 5

# 특정 셀·사이클 고정 (기존 동작)
python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --cell b1c0 --cycle 10

# IC 커브 + 창 경계 밴드 (vwindow 축에서 경계 확인용)
python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --mode ic --n-cycles 8

# 통계만 (플롯 생략)
python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --no-plot
python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --no-plot --dataset HUST
```

출력: `4_hi_analysis/outputs/seg_diagnose/{axis}/`

---

## 5. 모델 — `5_model/`

### 5-0. 설정 — `5_model/config/scr.yaml`

핵심 설정:
```yaml
scenario:
  axis: protocol        # 세그멘테이션 축: qfrac | protocol | vwindow | rcs | cluster
  axis_config: {}       # 축별 파라미터 (예: vwindow: {n_windows: 3})

data:
  data_dir: null        # null → scenario.axis 기반 자동 결정 (_4_data_hi/{axis}/cycle)
  seg_data_dir: null    # null → scenario.axis 기반 자동 결정 (_4_data_hi/{axis}/seg)
  datasets: ["MIT", "HUST"]
  is_cross_dataset_evaluate: false
  gates_from: null | "_5_data_model_scr/MMDD_HHMM_p1_prot"  # Phase 2 시 지정
  use_initial_capacity: true

classifier:             # ── Phase 1/2: SCRModel probe gate + 분류기 설정 ──────
  type: mlp             # 분류기 타입: mlp(HI probe) | cnn(원시 V/I CNN + HI 융합)
  is_auto_mk_selection: false
  charge_probe_m:   3   # Phase 2에서 probe JSON 상위 m개 사용 (분류기와 별개)
  discharge_probe_m: 2

regression:
  scen_k_count: 55      # 시나리오별 regression HI 수 (Phase 2) — 64→55: L0 sparsity 유효화

model:
  regression_model: "mlp"       # mlp / transformer / i_transformer / resnet_tab
  mlp_hidden_dims: [512, 256, 128, 64]

loss:
  lambda_scen: 0.01     # CE 손실 활성 (MSE 스케일과 균형; 0이면 CE 비활성)
  lambda_l0_auto: true

training:
  run_overfit_test: false  # 별도 tmp_standalone_overfit.py 사용

classifier:
  type: mlp              # mlp | cnn (원시 V/I RawCNN + HI 융합, §5-5)

test:
  random_segment_test: false    # true = 랜덤세그먼트 통합 평가 추가 실행 (test_rs 사용)
  random_seg_data_dir: "_4_data_hi/test_rs/seg"
  random_seg_datasets: ["MIT", "HUST"]
  # random_seg_routing 은 더 이상 단일 모드를 고르지 않는다 — 통합 평가가
  # 분류기 유무에 따라 oracle(+hard+soft)를 자동으로 모두 실행한다 (§5-10/5-11).
```

**run dir 명명 규칙** (`train_scr.py` 자동 생성):
```
_5_data_model_scr/
  MMDD_HHMM_p1_{axis}               ← Phase 1 (예: 0716_1149_p1_prot)
  MMDD_HHMM_p2_{model}_{axis}       ← Phase 2 (예: 0716_1200_p2_mlp_prot)
    ├── checkpoints/best.pt
    ├── gates/classification_HIs.json  (Phase 1 출력)
    ├── gates/regression_HIs.json      (Phase 1 출력)
    ├── classifier/clf_best.pt         (Step 8 출력 — B안)
    ├── scenario_spec.json
    └── config.yaml
```

> `data_dir` / `seg_data_dir` 는 `null` 로 두면 `scenario.axis` 기준으로 자동 결정된다.  
> `--seg-axis` CLI 인수가 yaml보다 우선.  
> `run_pipeline.py`로 실행 시 `gates_from` / `--run-dir` / `--checkpoint` 는 자동 주입된다.

---

### 5-0. 데이터 흐름 및 모델 입출력 구조

전체 흐름: **Wide pkl → Segment DataFrame → SegmentDataset → Batch → SCRModel → 출력 dict**

---

#### (A) Wide pkl 포맷 (`_4_data_hi/{axis}/{MIT,HUST}/*.pkl`)

한 파일 = 한 셀, 한 행 = 한 사이클.

```
columns:
  cycle          : int — 사이클 번호
  capacity_Ah    : float — 해당 사이클 실측 방전 용량 [Ah] = SOH 타겟 원본
  [세그먼트별 HI × n_scenarios]:
    stat_{key}_{seg}   18개 × n_seg   → e.g. stat_v_mean_dis_hi
    diff_{key}_{seg}   20개 × n_seg
    lfp_{key}_{seg}    20개 × n_seg
    morph_{key}_{seg}   6개 × n_seg
    (stat_q_abs, stat_energy_seg는 제외 → 세그먼트당 64 HI)
  총 컬럼 수 ≈ 1 + 1 + 64×n_scenarios  (qfrac: 64×6=384+2=386)
```

---

#### (B) Segment DataFrame (`_wide_to_segments()` 또는 native seg pkl)

`pd.DataFrame` — 한 행 = 한 (셀, 사이클, 세그먼트) 트리플.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `cell_id` | str | 셀 식별자 (e.g. `b1c0`, `1-1`) |
| `cycle` | int | 사이클 번호 |
| `seg_name` | str | 시나리오 이름 (e.g. `dis_hi`, `chg_win1`) |
| `seg_idx` | int | 시나리오 인덱스 0..n_scenarios-1 |
| `direction` | float | +1.0=충전, -1.0=방전 |
| `level` | int | latent_class 0/1/2 (lo/mid/hi) |
| `capacity_Ah` | float | 사이클 실측 용량 — SOH 타겟 원본 |
| `hi_00`..`hi_63` | float32 | 64개 HI (세그먼트별, NaN 허용) |
| `raw_v` / `raw_i` | list[48] | 원시 V/|I| 곡선 (q_frac 정규화 48pt, CNN 입력용). native-seg에만 존재; 없으면 로더가 zero-fallback |
| `dataset` | str | `MIT` 또는 `HUST` |

> Wide 포맷: `_wide_to_segments()` 로 reshape (raw 컬럼 없음 → x_raw=0).  
> Native seg 포맷: `load_dataset_native_seg()` 직접 로드 (raw_v/raw_i 있으면 CNN 입력으로 사용).

---

#### (C) SegmentNormalizer (train 셀만 fit)

```python
mean_  : (64,) float64  # 각 HI의 train 집합 mean (전부 NaN → 0.0)
std_   : (64,) float64  # 각 HI의 train 집합 std  (constant 또는 전부 NaN → 1.0)

# transform_x():
x_norm = (x_raw - mean_) / std_   # NaN 위치는 0.0으로 마스킹
nan_mask = ~isnan(x_raw)          # 1.0=유효, 0.0=결측

# target_mean_ / target_std_: cap_init z-scoring 전용
# SOH ratio(target) = capacity_Ah / cap_init_Ah 는 정규화 불필요
```

---

#### (D) SegmentDataset `__getitem__` 반환 (단일 샘플)

```python
{
  "x_hi":      Tensor (64,)    float32  # z-scored HI, NaN→0
  "x_raw":     Tensor (2, 48)  float32  # 원시 [V; |I|] 곡선 (CNN 입력). raw 없는 pkl → 전부 0
  "nan_mask":  Tensor (64,)    float32  # 1.0=유효, 0.0=결측
  "direction": Tensor ()       float32  # +1.0 or -1.0
  "level":     Tensor ()       int64    # latent class (0/1/2)
  "seg_idx":   Tensor ()       int64    # 시나리오 인덱스 0..n-1
  "target":    Tensor ()       float32  # SOH ratio = cap_Ah / cap_init_Ah ∈ (0, 1]
  "cap_init":  Tensor ()       float32  # z-scored 초기/정격 용량 Ah (크기 conditioning)
}
```

`collate_fn` 후 배치:

```python
batch = {
  "x_hi":      (B, 64)    float32
  "x_raw":     (B, 2, 48) float32   # RAW_CH=2, RAW_N=48 (CNN 분류기 입력)
  "nan_mask":  (B, 64)    float32
  "direction": (B,)       float32
  "level":     (B,)       int64
  "seg_idx":   (B,)       int64
  "target":    (B,)       float32
  "cap_init":  (B,)       float32
}
```

> `x_raw`는 CNN 분류기(`classifier.type: cnn`)에서만 사용. SCRModel 회귀 forward는 이 키를
> 읽지 않고 무시하므로 기존 회귀 경로는 완전히 불변.

---

#### (E) SCRModel 내부 연산 흐름

```
입력 batch
  x_hi    (B, 64)         — z-scored HI
  nan_mask(B, 64)         — 유효 비트
  direction (B,)
  seg_idx   (B,)
  cap_init  (B,)

─── 전처리 ───────────────────────────────────────────────
x = x_hi × nan_mask                          (B, 64)  [NaN 위치 → 0]

─── Stage A: 방향별 probe gate ─────────────────────────
  충전 샘플 (direction > 0) → charge_probe_gate
  방전 샘플 (direction ≤ 0) → discharge_probe_gate
  Gate 타입:
    Phase 1: HardConcreteGate (학습 가능, L0 정규화 대상)
    Phase 2: buffer mask (고정 bool (64,))

probe_x (B, 64)   — 선택된 HI만 비0 (m개 활성)
probe_z (B, 64)   — gate 확률 벡터 (0/1 혼합 또는 이진)

─── Stage B: 시나리오별 gate ────────────────────────────
  seg_idx 기준으로 각 샘플에 해당 시나리오 gate 적용
  scen_gates[s] (64,) — 시나리오 s 전용 gate (k개 활성)

scen_x (B, 64)   — 시나리오 gate 출력
scen_z (B, 64)   — gate 확률 벡터

─── Capacity head 입력 구성 ──────────────────────────────
feat = concat(probe_x, scen_x, direction.unsqueeze(1), cap_init.unsqueeze(1))
     = (B, 64+64+1+1) = (B, 130)

─── Capacity head (선택 가능) ────────────────────────────
  mlp          : Linear(130→d)→ReLU→Linear(d→d/2)→ReLU→Linear(d/2→1)
  transformer  : probe/scen/meta 3토큰 → TransformerEncoder(Pre-LN) → mean pool → Linear→1
  i_transformer: 130 스칼라 각각이 토큰 → feature-wise attention → mean pool → Linear→1

cap_pred (B,)   — SOH ratio 예측값 [정규화 없음, 직접 SOH ratio 단위]
```

---

#### (F) SCRModel `forward()` 반환 (출력 dict)

```python
{
  "cap_pred":     Tensor (B,)           float32  # SOH ratio 예측 ∈ (0, ~1]
  "level_logits": Tensor (B, n_classes) float32  # probe_mlp 출력 (lambda_scen>0) or all-zero
  "probe_x":      Tensor (B, 64)        float32  # Stage A gate 출력 (상위 m개 활성, 나머지 0)
  "probe_z":      Tensor (B, 64)        float32  # Stage A gate 확률
  "scen_z":       Tensor (B, 64)        float32  # Stage B gate 확률
}
```

**손실 계산 (`scr_loss.py`):**
```
L = MSE(cap_pred, batch["target"])       ← SOH ratio vs SOH ratio
  + λ_l0 × L0_penalty(probe_z, scen_z)  ← Phase 1만 비0
```

---

#### (G) 평가 시 후처리 (`scr_evaluator.py`)

```python
# SOH ratio → Ah 변환 (논문 지표 및 capacity curve 플롯용)
cap_pred_Ah = cap_pred_raw × cap_init_raw   # (N,) float32
cap_true_Ah = cap_true_raw × cap_init_raw   # (N,) float32

# 지표 계산 (SOH 단위 기준)
RMSE = sqrt(mean((cap_pred_raw - cap_true_raw)²))   # SOH 단위
MAE  = mean(|cap_pred_raw - cap_true_raw|)
R²   = 1 - SS_res / SS_tot
MAPE = mean(|Δ| / cap_true_raw) × 100  [%]
```

**저장 텐서 크기 요약:**

| 텐서/배열 | 형태 | 단위 | 어디서 생성 |
|---|---|---|---|
| `x_hi` | `(B, 64)` | z-score | SegmentDataset |
| `nan_mask` | `(B, 64)` | 0/1 | SegmentDataset |
| `probe_z`, `scen_z` | `(B, 64)` | 확률 | SCRModel |
| `cap_pred` | `(B,)` | SOH ratio | SCRModel |
| `cap_pred_Ah` | `(N_test,)` | Ah | SCREvaluator |
| `norm.mean_`, `norm.std_` | `(64,)` | 원본 단위 | SegmentNormalizer |

---

### 5-1. HI 스키마 — `5_model/utils/hi_schema.py`

```python
N_HI = 64   # STAT 18 + DIFF 20 + LFP 20 + MORPH 6 (q_abs, energy_seg 제외)

# 카테고리별 컬럼 키 목록 (순서 고정)
STAT_KEYS   # 20개
DIFF_KEYS   # 20개
LFP_KEYS    # 20개
MORPH_KEYS  # 6개

CATEGORY_COSTS = {"stat": 1.0, "diff": 1.5, "lfp": 2.0, "morph": 3.0}
LEAK_COLS = {"stat_q_abs", "stat_energy_seg"}   # 모델 입력 제외

RAW_N  = 48   # 원시 세그먼트 곡선 리샘플 포인트 수 (CNN 입력)
RAW_CH = 2    # 원시 곡선 채널 수: [V, |I|]
# hi_correlation.py(데이터 생성)와 segment_dataset.py(로더)가 공유하는 단일 소스

def get_hi_cols_for_seg(seg: str) -> list[str]:  # 64개 컬럼명 반환 (순서 고정)
def get_hi_cost_vector(seg: str) -> list[float]: # 64개 L0 비용 벡터
def spec_from_qfrac() -> ScenarioSpec:           # backward-compat: qfrac 기본 spec 반환
```

**제거됨** (시나리오 하드코딩 → ScenarioSpec으로 이관):
- `N_SEGS`, `N_LEVELS`, `SEGMENTS`, `SCEN_MAP`, `SEG_LEVEL`, `SEG_DIRECTION`

---

### 5-2. 데이터셋 — `5_model/datasets/segment_dataset.py`

**로딩 경로 (우선순위 순):**

```
1순위: native-seg pkl (_4_data_hi/{axis}/seg/{ds}/*.pkl)
       → load_dataset_native_seg(seg_dir, datasets, wide_dir, spec)
       → 한 행 = 한 세그먼트 (이미 reshape됨)
       → wide pkl에서 cycle-level capacity_Ah 덮어쓰기 (segment q_abs 대신)

2순위: wide-cycle pkl (_4_data_hi/{axis}/cycle/{ds}/*.pkl)
       → load_dataset_wide(data_dir, datasets, spec=spec)
       → _wide_to_segments(df, cell_id, spec) 로 reshape
       → 한 행 = 한 사이클 → n_scenarios 행으로 분해
```

**ScenarioSpec 연동:**
```python
_DEFAULT_SPEC: ScenarioSpec = spec_from_qfrac()   # 모듈 레벨 기본값

def _wide_to_segments(df, cell_id, spec=None):
    _spec = spec or _DEFAULT_SPEC
    for seg_idx, seg in enumerate(_spec.scenario_names):
        hi_cols = get_hi_cols_for_seg(seg)
        dir_idx, latent_class = _spec.scenario_to_dir_class(seg_idx)
        direction = 1 if dir_idx == 0 else -1  # +1=충전, -1=방전
        level = latent_class                    # 0=lo, 1=mid, 2=hi

def build_datasets(cfg, spec=None) -> (train_ds, val_ds, test_ds, norm):
    ...  # spec=None 이면 _DEFAULT_SPEC(qfrac) 사용
```

**SegmentNormalizer:**
- `fit()`: train 분할로 HI별 z-score 파라미터 계산
  - 전-NaN 컬럼 (예: RCS/vwindow CC 전용 → `corr_qi`, `r_dyn_seg` 구조적 NaN): mean=0, std=1 (no-op)
  - `warnings.catch_warnings()` + `warnings.simplefilter("ignore", RuntimeWarning)` 로 empty-slice 경고 억제
    - (이전 `np.errstate(all="ignore")`는 Python 레벨 RuntimeWarning을 잡지 못함)
- `transform_x()`: NaN 위치를 `nan_mask`로 기록 후 0.0으로 대체

**SegmentDataset (PyTorch Dataset):**

| 텐서 (`__getitem__` 반환) | 형상 | 설명 |
|---|---|---|
| `x_hi` | (64,) | z-score 정규화된 HI 피처 |
| `x_raw` | (2, 48) | 원시 [V; |I|] 곡선 (CNN 입력). raw 컬럼 없는 pkl → zero-fallback (`_build_raw_tensor`) |
| `nan_mask` | (64,) | 1=유효값, 0=원래NaN |
| `direction` | scalar | +1=충전, -1=방전 |
| `level` | scalar(int64) | 0=lo, 1=mid, 2=hi (spec.class_names 인덱스) |
| `seg_idx` | scalar(int64) | 0~(n_scenarios-1) scenario_id |
| `cap_init` | scalar | z-score 정규화된 초기/정격 용량 Ah (모델 conditioning) |
| `target` | scalar | **SOH ratio** = `capacity_Ah / cap_init_Ah` ∈ (0, 1] |

| 메타 속성 (평가용, `__getitem__` 미반환) | 형상 | 설명 |
|---|---|---|
| `cap_init_raw` | (N,) float32 | 초기/정격 용량 Ah (SOH→Ah 변환: `soh × cap_init_raw`) |
| `capacity_raw` | (N,) float32 | 실측 `capacity_Ah` |
| `cell_ids` / `cycles` / `seg_names` | list[str/int] | 평가 메타데이터 |

> **SOH ratio 채택 이유**: HUST(1.2 Ah)와 MIT(1.1 Ah)의 nominal capacity 차이로 인한  
> 스케일 불일치를 제거한다. `target ∈ (0, 1]` 이므로 정규화기가 불필요하다.

**유틸리티:**
```python
from datasets.segment_dataset import filter_dataset_by_cells

# cell_id 리스트로 Dataset 서브셋 생성 (fine-tuning split 등에 사용)
sub_ds = filter_dataset_by_cells(ds, ["b1c0", "b1c1", "1-1"])
```

---

### 5-3. L0 게이트 — `5_model/models/hard_concrete.py`

Louizos et al. (2018) Hard-Concrete:
```
Train:  z = clamp(sigmoid((log U - log(1-U) + log_alpha) / β) × (ζ-γ) + γ, 0, 1)
Infer:  z = (sigmoid(log_alpha) × (ζ-γ) + γ > 0).float()  ← 이진 마스크

β=2/3, γ=-0.1, ζ=1.1
P(z>0) = sigmoid(log_alpha - β×log(-γ/ζ))  ← L0 패널티 계산에 사용
```
`active_indices()`: 추론 시 활성(z>0) HI 인덱스 반환

---

### 5-4. 모델 아키텍처 — `5_model/models/scr_model.py`

```
입력: x_hi(64) + nan_mask(64) + direction(±1) + cap_init(z-scored Ah)

Stage A — direction-aware probe gate (HI 서브셋 선택)
  direction 기반 라우팅:
    충전 → charge_probe_gate    (HardConcreteGate, 64→64)
    방전 → discharge_probe_gate (HardConcreteGate, 64→64)
  probe_x (B, 64)  — 게이트 적용 결과 (상위 m개 활성)
  ※ lambda_scen > 0 시 probe_mlp 활성:
       probe_x_dir = [probe_x | direction]  (B, 65)
       level_logits = probe_mlp(probe_x_dir)  → CE 손실 입력
     lambda_scen = 0 시: level_logits = zeros(B, n_classes)  (호환용)

Stage B — scenario-conditioned regression
  scen_gates[n_scenarios]: 각 시나리오별 HardConcreteGate (64→64)
                           n_scenarios = spec.n_scenarios
  x_probe = probe_gate(x_hi)               # (B, 64), direction별 게이트
  x_scen  = scen_gates[seg_idx](x_hi)      # (B, 64), seg_idx 하드 라우팅
  feat    = [x_probe | x_scen | direction | cap_init]  # (B, 130)
  cap_pred = cap_head(feat)                # (B,) — SOH ratio ∈ (0,1]

출력 dict:
  cap_pred      : (B,) SOH ratio 예측값
  level_logits  : (B, n_classes) 항상 0 (CE 비활성)
  probe_z       : (B, 64) probe gate 활성값
  scen_z        : (B, 64) scenario gate 활성값
```

**SCRModel 생성:**
```python
model = SCRModel(
    d_probe=64, d_head=128, dropout=0.1,
    charge_probe_mask=None,    # Phase 1: None → HardConcreteGate (학습 가능)
    discharge_probe_mask=None, # Phase 2/FT: (64,) bool tensor → 고정 buffer
    scen_masks=None,           # Phase 2/FT: (n_scenarios, 64) bool tensor → 고정 buffer
    model_cfg={"regression_model": "mlp"},
    spec=spec,                 # ScenarioSpec — None이면 qfrac 기본값
)
# model.spec, model.n_scenarios, model.n_classes 로 접근
```

**Phase 전환:**
- **Phase 1**: gate 학습, L0 패널티 활성. gate_prob 랭킹 JSON 저장
- **Phase 2/Fine-tune**: JSON 로드 → 고정 buffer. **cap_head만 학습**  
  (gates가 buffer이므로 `model.parameters()` = cap_head만 → 명시적 freeze 불필요)

**cap_head 옵션** (`5_model/models/cap_heads.py`):

| 모드 | 구조 | 특징 |
|---|---|---|
| `mlp` | Linear(130→d)→ReLU→…→1 | 기본, 빠름 |
| `transformer` | probe/scen/meta 3토큰 → TransformerEncoder | semantic 토큰 분리 |
| `i_transformer` | 130 피처 각각이 토큰 → feature-wise attention | feature-wise |

---

### 5-5. 시나리오 분류기 플러그인 — `5_model/models/scenario_classifier.py`

`ScenarioClassifier` ABC: `classify(probe_x, batch) → (B, n_classes) logits`

| 클래스 | 학습 방식 | 레이블 필요 | 용도 |
|---|---|---|---|
| `MLPProbeClassifier` | CE (독립) | ✗ | **B안 기본 분류기** — `train_classifier.py`로 별도 학습 |
| `CNNProbeClassifier` | CE (독립) | ✗ | **원시 V/I 곡선 CNN + HI probe 융합** — `classifier.type: cnn` |
| `RuleClassifier` | 없음 (규칙) | ✗ | `spec.scenario_to_dir_class(seg_idx)` 결정론적 배정 |
| `CentroidClassifier` | fit 단계 | ✗ | L2 최근접 센트로이드 (`update()+finalize()` 또는 `fit()`) |
| `OracleClassifier` | 없음 (GT) | ✓ | `batch["level"]` 직접 사용 → 상한선 측정 |
| `NoneClassifier` | 없음 | ✗ | 전-0 로짓 반환 (routing=none 동작) |

```python
from models.scenario_classifier import build_classifier
clf = build_classifier("mlp",  n_hi=64, n_classes=3, d_hidden=64)   # B안 (HI만)
clf = build_classifier("cnn",  n_hi=64, n_classes=3, d_hidden=128)  # 원시 곡선 CNN 융합
clf = build_classifier("rule", n_hi=64, n_classes=3, spec=spec)     # 규칙 기반
```

**CNNProbeClassifier 구조** (`models/raw_cnn.py`의 `RawCNN` 사용):
```
x_raw (B,2,48) ─ RawCNN ─→ cnn_emb (B,64)
   RawCNN: Conv stem → ResBlock×2 → AttentionPool → Linear  (≈42K params)
[cnn_emb(64) ‖ probe_x(64) ‖ direction(1)] = 129D → MLP → n_classes logits

classify(probe_x, batch): batch["x_raw"]·batch["direction"]를 내부에서 융합
구 pkl(x_raw 전부 0) → CNN 임베딩이 상수로 수렴 → probe_x만으로 동작(무해 degrade)
```

> **B안 설계 원칙**: 분류기는 회귀 모델(SCRModel)과 완전히 분리.  
> 배터리 운영 레짐(CC1/CC2/CV 등)은 물리적으로 실재하는 상태이므로 독립 분류가 타당하며,  
> 분류 정확도를 회귀 성능과 별도로 평가할 수 있어 논문 방어에 유리하다.  
> `SCRModel`에는 분류기가 없다 — `level_logits = zeros` 반환 (evaluator 호환용).

---

### 5-6. 손실 — `5_model/training/scr_loss.py`

```
SCRLoss (이중 목적, Phase 1 전용):
  L = MSE(cap_pred, target)                         ← target = SOH ratio ∈ (0,1]
    + λ_scen × CE(level_logits, batch["level"])     ← lambda_scen > 0 시 활성
    + λ_l0 × L0_penalty

  # CE 항 (lambda_scen): probe_gate를 분류에 유리한 HI로 유도 (이중 목적 Phase 1)
  #   λ_scen = 0.01  (MSE ~0.001-0.005 vs CE ~0.3-0.8 → λ≈0.01로 균형)
  #   lambda_scen = 0 시 CE 항 완전 비활성
  # Phase 2 / Fine-tune: gate가 buffer → L0_penalty = 0 자동, λ_scen = 0 강제

L0_penalty = (1 / model.n_scenarios) × Σ_s Σ_i cost_i × P(z_i≠0 in scenario s)
  P(z_i≠0) = 1 - (1 - p_probe_i)(1 - p_scen_i)
  Phase 2 / Fine-tune: gate가 buffer → L0_penalty = 0 자동
```

```
ClfLoss (분류기 전용, train_classifier.py):
  L = CrossEntropyLoss(logits, batch["level"])
  scr_loss.py와 완전히 분리된 독립 손실
```

---

### 5-7. 학습 루프 — `5_model/training/scr_trainer.py`

**Phase 1 — L0 게이트 학습:**
```
λ_l0 스케줄 (loss.lambda_l0_schedule):
  delayed_warmup : λ_l0=0 (warmup_epochs) → 선형 증가 (ramp_epochs)
  exp_ramp       : 지수 증가
  cyclic         : 주기적 증감
  none           : 고정 lambda_l0

Early stopping (patience=20, val RMSE 기준)
Gate JSON 저장:
  gates/classification_HIs.json  : charge_ranked + discharge_ranked + probs
  gates/regression_HIs.json      : seg_{s}_ranked + probs (s = 0..n_scenarios-1)
  gates/gate_probs.png           : gate 확률 bar chart
```

**Phase 2 — 회귀 정밀 학습:**
```
gates_from 경로에서 JSON 로드 → 고정 buffer 마스크 (SCRModel에 주입)
λ_l0 = 0, λ_scen = 0 (L0·CE 비활성)
학습 파라미터: cap_head만 (probe/scen gates는 buffer → requires_grad=False)
Early stopping (patience=20, val SOH RMSE 기준)
checkpoints/best.pt 저장 (norm_mean/std/target_mean/target_std 포함)
```

공통: AdamW + CosineAnnealingLR (warmup 10 epoch) + grad_clip 1.0

---

### 5-8. 학습 진입점 — `5_model/train_scr.py`

**Phase / Spec 결정 흐름:**
```
--phase > --no-gates(legacy) > yaml gates_from 유무 순서로 Phase 결정
--seg-axis > yaml scenario.axis > "qfrac" 순서로 축 결정
spec = get_segmenter(axis, cfg).get_spec()
spec.save(output_dir / "scenario_spec.json")       ← test_scr.py 재사용
build_datasets(cfg, spec=spec)                     ← spec 명시 전달 필수
  # spec 미전달 시 _DEFAULT_SPEC(qfrac, n_scenarios=6)으로 fallback →
  # vwindow(n_scenarios=7) 데이터에서 segment_id=6 매핑 실패(IntCastingNaNError)
```

```bash
python 5_model/train_scr.py --phase 1
python 5_model/train_scr.py --phase 2
python 5_model/train_scr.py --phase 2 --gates-from _5_data_model_scr/0711_1538

# 세그멘테이션 축 지정 (Step 4와 동일하게)
python 5_model/train_scr.py --phase 1 --seg-axis vwindow
python 5_model/train_scr.py --phase 1 --seg-axis qfrac \
  --axis-config '{"dis_bounds":[0,0.3,0.6,1.0]}'

# probe/scen HI 수 오버라이드
python 5_model/train_scr.py --phase 2 --charge-m 3 --discharge-m 1 --scen-k 10

# 재현성 시드
python 5_model/train_scr.py --phase 1 --seed 42
```

run_dir 자동 생성 규칙:
```
Phase 1: MMDD_HHMM_p1_{axis}          (예: 0716_1149_p1_prot)
Phase 2: MMDD_HHMM_p2_{model}_{axis}  (예: 0716_1200_p2_mlp_prot)
  axis 약어 : qfrac→qfr / protocol→prot / vwindow→vwin / rcs→rcs / cluster→clst
  model 약어: mlp→mlp / transformer→tr / i_transformer→itr / resnet_tab→res
```

---

### 5-9. 시나리오 분류기 학습 (B안) — `5_model/train_classifier.py`

SCRModel Phase 1/2와 완전히 독립적으로 HI 특성만으로 시나리오 레짐(level)을 학습한다.

**학습 구조:**
```
타입   : scr.yaml classifier.type = mlp(기본) | cnn
마스크 : classification_HIs.json 로드 → charge/discharge 방향별 상위 m개 probe HI 마스크
입력   : (mlp) probe_x (N_HI=64, 마스크 적용) + direction (1) = 65차원
         (cnn) probe_x (64) + RawCNN(x_raw (2,48))=64 + direction (1) = 129차원 내부 융합
         (마스크 없으면 x_hi 전체 fallback)
타겟   : batch["level"]  (latent_class: lo=0 / mid=1 / hi=2)
모델   : (mlp) MLPProbeClassifier  Linear(65→d_hidden)→ReLU→Dropout→Linear(→d_hidden//2)→ReLU→Linear(→n_classes)
         (cnn) CNNProbeClassifier  RawCNN + Linear(129→d_hidden)→GELU→Dropout→Linear(→d_hidden//2)→GELU→Linear(→n_classes)
손실   : CrossEntropyLoss  (regression과 완전 분리)
정규화 : 동일 config로 build_datasets 호출 → Phase 2와 동일 정규화기 자동 재현
저장   : {run_dir}/classifier/clf_best.pt
```

```bash
# Phase 2 run_dir 지정 (분류기를 해당 run_dir 내에 저장)
python 5_model/train_classifier.py --run-dir _5_data_model_scr/0716_1200_p2_mlp_prot

# 하이퍼파라미터 오버라이드
python 5_model/train_classifier.py \
    --run-dir _5_data_model_scr/0716_1200_p2_mlp_prot \
    --epochs 300 --lr 5e-4 --d-hidden 128

# run_pipeline.py Step 8로 자동 실행 (--run-dir 자동 주입)
python run_pipeline.py 8 --to-step 8
```

**저장 포맷:**
```python
{
  "clf_state": ...,     # MLPProbeClassifier | CNNProbeClassifier state_dict
  "clf_type":  "mlp",   # "mlp" | "cnn" — test_scr.py 복원 시 분기 기준
  "n_hi":      65,      # mlp: N_HI(64)+direction(1)=65 | cnn: N_HI=64 (direction 내부 concat)
  "n_classes": 3,       # 출력 클래스 수
  "d_hidden":  64,      # 히든 차원
  "val_acc":   0.xxx,   # 최고 검증 정확도
  "epoch":     int,     # 저장 시점 에폭
  "axis":      "prot",  # 학습 축
}
```

> `type: cnn`으로 학습하려면 Step 4를 재실행해 raw_v/raw_i 컬럼이 포함된 seg pkl을 먼저
> 생성해야 한다. raw 없는 기존 pkl로 `cnn`을 써도 동작하나 x_raw=0이라 CNN 이점이 없다.

---

### 5-10. 평가 진입점 — `5_model/test_scr.py`

> **2026-07 재구조**: 기존 E1(qfrac→qfrac)/E2(qfrac→random)/E3(routing) 3축 체계를
> 폐기하고, **분류→회귀 통합 평가**로 단일화했다. 학습된 분류기가 테스트셋을 실제로
> 분류하고, 그 결과로 라우팅한 회귀 성능까지 한 번에 확인할 수 있어야 한다는 요구
> (routing=none으로 분류기를 우회하는 기존 E1은 분류 성능을 전혀 보여주지 못했음)를
> 반영했다.

**평가 모드 (분류기 유무에 따라 자동 결정):**

| 모드 | 라우팅 | 의미 |
|---|---|---|
| `oracle` | 정답 seg_idx | 분류 100% 가정 시 회귀 상한선. 분류기 없어도 항상 실행 |
| `hard` | 분류기 argmax | 실배포 시나리오 (분류기 체크포인트 있을 때만) |
| `soft` | 분류기 확률 가중 평균 | 분류 불확실성을 반영한 회귀 (분류기 있을 때만) |

각 모드마다 **회귀(capacity/breakdown) + 분류(classification) 지표를 함께** 계산한다
(`SCREvaluator.evaluate_modes()`, §5-11).

```bash
# 기본: run_dir 내 classifier/clf_best.pt 자동 탐색 → oracle+hard+soft 모두 실행
python 5_model/test_scr.py --checkpoint .../best.pt

# 분류기 체크포인트 직접 지정
python 5_model/test_scr.py \
    --checkpoint _5_data_model_scr/0716_1200_p2_mlp_prot/checkpoints/best.pt \
    --classifier-ckpt _5_data_model_scr/0716_1200_p2_mlp_prot/classifier/clf_best.pt

# 분류기 없으면 oracle만 자동 실행 (hard/soft 건너뜀)
python 5_model/test_scr.py --checkpoint .../best.pt   # classifier/clf_best.pt 없는 run_dir

python 5_model/test_scr.py --rep-cells b1c0 b1c1 1-1
```

랜덤세그먼트 평가(`test.random_segment_test: true`)도 동일한 oracle/hard/soft
플로우로 실행된다 (§5-11 랜덤세그먼트 통합 평가 참조).

**Spec 로드 순서 (build_datasets 이전에 수행):**
```python
run_dir    = ckpt_path.parent.parent
_spec_path = run_dir / "scenario_spec.json"
spec = ScenarioSpec.load(_spec_path) if _spec_path.exists() else spec_from_qfrac()
# spec 로드 후 build_datasets(cfg_saved, spec=spec) 호출
# → SCRModel(spec=spec) 에 전달 → n_scenarios / n_classes 복원
# ※ spec 로드 전에 build_datasets 호출 시 _DEFAULT_SPEC(qfrac) fallback → segment_id 매핑 오류
```

---

### 5-11. 평가 모듈 — `5_model/evaluation/scr_evaluator.py`

**SCREvaluator 생성 시 spec 정보 추출:**
```python
self._n_scenarios = model.n_scenarios
self._n_classes   = model.n_classes
self._seg_names   = model.spec.scenario_names
self._class_names = model.spec.class_names
self._spec        = model.spec          # routing table 접근용 (B안 라우팅)
self._classifier  = None               # set_classifier()로 주입
```

**B안 라우팅 지원:**
```python
evaluator.set_classifier(clf)           # MLPProbeClassifier | CNNProbeClassifier 주입
# predict_dataset(ds, routing_mode="hard")  → 분류기 argmax → seg_idx 오버라이드
# predict_dataset(ds, routing_mode="soft")  → 분류기 확률 가중 평균
# predict_dataset(ds, routing_mode="none")  → 기존 동작 (seg_idx 그대로, oracle 라우팅에 사용)

# 분류기 입력 구성 (train_classifier.py와 동일, 타입별 분기):
#   probe_x_clf = model.get_probe_x(x_hi, direction, seg_idx)   # (B, 64) probe 마스크 적용
#   MLP: clf_logits = classifier([probe_x_clf | direction])      # (B, 65)
#   CNN: clf_logits = classifier.classify(probe_x_clf, batch_d)  # batch_d에 x_raw 포함
#        (isinstance(classifier, CNNProbeClassifier) 로 분기)

# routing table (spec.routing)이 jagged list일 경우 padding 후 수동 채움:
#   _routing_t = zeros(n_dir, max_n_cls)  ← torch.tensor() 대신 안전한 방식
```

**통합 평가 — `evaluate_modes()` (2026-07 신규):**
```python
results = evaluator.evaluate_modes(
    ds, modes=("oracle", "hard", "soft"), batch_size=512,
)
# results[mode] = {
#   "capacity":       compute_metrics 결과 (rmse/mae/r2/mape)
#   "breakdown":      charge/discharge + level_lo/mid/hi 별 capacity 지표
#   "classification": classification_metrics() 결과 (아래) 또는 oracle이면 {"accuracy":1.0,...}
#   "_pred":          predict_dataset() 원본 반환값 (플롯/CSV 저장용, JSON 직렬화 전 제거)
# }
clean = evaluator.strip_modes_for_json(results, efficiency=eff_dict)  # _pred 제거 + efficiency 부착
```

**분류 성능 — `classification_metrics(pred_dict)` (2026-07 신규):**
```python
{
  "accuracy":               float,   # 전체 평균 정확도
  "per_class_accuracy":     {"lo": .., "mid": .., "hi": ..},   # 클래스(레벨)별 recall
  "per_direction_accuracy": {"charge": .., "discharge": ..},
  "per_scenario_accuracy":  {"chg_lo": .., "dis_hi": .., ...}, # routing[dir][level] 기준 시나리오별
  "confusion_matrix":       [[...]],  # n_classes × n_classes
  "n_samples":              int,
}
# 반환 None 조건: level_pred 가 전부 동일값 (분류기 미사용/미학습)
# degenerate 가드: level_true 가 전부 동일값(placeholder, 예: 구 test_rs n_classes=1)이면
#   accuracy 대신 {"note": "...", "predicted_distribution": {...}} 반환 — 무의미한 수치 노출 방지
```

**데이터셋별 플롯 저장 (random_seg_test):**
```python
evaluator.plot_for_dataset(rs_pred, out_dir, rep_cells, tag="random_seg_hard")
# → out_dir/scatter_random_seg_hard.png     (정답/오답 색상 산포도, 아래 참조)
# → out_dir/capacity_curve_{cell}.png       (대표 셀 용량 곡선)
# → out_dir/confusion_matrix_random_seg_hard.png  (level 혼동 행렬)
```

하드코딩 제거: `N_SEGS`, `N_LEVELS`, `SEGMENTS`, `_LEVEL_NAMES` 상수 없음.

**scatter — 분류 정확도 색상 표시 (2026-07 추가):** 분류(level_pred)가 활성이면
정답(초록)/오답(빨강) 색상으로 산포도를 그리고 제목에 `level acc=` 를 표시한다.
분류 비활성(oracle, 또는 분류기 없음)이면 기존 단색 산포도로 자동 폴백.

**predict_dataset 반환 dict:**
```python
{
  "cap_pred_raw":  np.ndarray,  # SOH ratio 예측값 — inverse_target 불필요
  "cap_true_raw":  np.ndarray,  # SOH ratio 실측값
  "cap_init_raw":  np.ndarray,  # Ah per sample (SOH→Ah 변환: pred × cap_init_raw)
  "level_pred":    np.ndarray,  # 분류기 argmax (routing=none 이면 항상 0 — 회귀모델 더미값)
  "level_true":    np.ndarray,  # 정답 latent class
  ...
}
```
> `level_pred`가 전부 0인 건 `routing_mode="none"`(oracle 라우팅 시 seg_idx를 정답으로
> 고정)일 때의 정상 동작이다 — SCRModel의 `level_logits`가 항상 zeros이기 때문. 분류기가
> 실제로 호출되는 건 `hard`/`soft` 모드뿐이다 (evaluator 내부에서 `set_classifier()`된
> 분류기의 `classify()`를 호출).

**저장 파일 (통합 평가 이후 구조):**

| 경로 | 내용 |
|---|---|
| `figures/scatter_test_{mode}.png` | mode ∈ {oracle,hard,soft}. hard/soft는 정답/오답 색상 |
| `figures/confusion_matrix_test_{mode}.png` | hard/soft만 생성 (oracle은 자명하므로 생략) |
| `figures/capacity_curve_{cell}.png` | 대표 셀 용량 곡선 (분류기 있으면 hard 기준, 없으면 oracle) |
| `metrics/metrics.json` | `{split: {mode: {capacity, breakdown, classification}, efficiency}}` — split ∈ {train(oracle만),val(oracle만),test(oracle+hard+soft)} |
| `routing/routing_heatmap.png` | probe(1행) + scen(n_scenarios행) × HI(64열) 활성 히트맵 |
| `routing/routing_table.csv` | gate별 선택 HI 목록 |
| `predictions/test_predictions.csv` | 실배포(hard) 예측 저장 (분류기 없으면 oracle) — `soh_true`, `soh_pred`, `soh_error`, `cap_true_Ah`, `cap_pred_Ah`, `cap_init_Ah`, level_true/pred, active HI 수 |

**metrics.json 예시 구조:**
```jsonc
{
  "test": {
    "oracle": {"capacity": {...}, "breakdown": {...}, "classification": {"accuracy": 1.0, "note": "oracle: ..."}},
    "hard":   {"capacity": {...}, "breakdown": {...}, "classification": {"accuracy": 0.822, "per_class_accuracy": {...}, "per_scenario_accuracy": {...}, "confusion_matrix": [...]}},
    "soft":   {"capacity": {...}, "breakdown": {...}, "classification": {"accuracy": 0.822, ...}},
    "efficiency": {"avg_probe_his": 10.0, "avg_scen_his": 55.0, ...}
  },
  "train": {"oracle": {...}},
  "val":   {"oracle": {...}}
}
```

**랜덤세그먼트 통합 평가 (`_run_random_segment_test`):** 메인 테스트와 동일하게
`evaluate_modes()`를 호출해 `random_seg_test/metrics.json`에 oracle/hard/soft +
efficiency를 저장한다. 구 `test_rs`(2-시나리오, `_assign_regime` 이전 형식)는 정답
레짐 라벨이 placeholder(단일값)라 `classification_metrics`의 degenerate 가드가
`{"note": "...", "predicted_distribution": {...}}`를 반환한다 — **회귀(RMSE 등)는
정상적으로 보고되지만 분류 정확도는 위치기반 레짐 라벨 포함해 `test_rs`를
재생성(`tmp_make_test_rs.py`, §4-1b)하기 전까지 무의미**하다.

```python
_is_model_space = (rs_spec.n_scenarios == model.n_scenarios)   # 6-시나리오 재생성 여부 감지
evaluator.evaluate_modes(
    rs_ds, modes=rs_modes,
    direction_routing_for_oracle=not _is_model_space,  # 구 형식(0/1)만 방향 보정
)
```

---

### 5-12. Few-shot Fine-tuning — `5_model/finetune_scr.py`

Phase 2 checkpoint를 소수의 target dataset 셀로 재학습하는 도메인 적응 스크립트.

**프로토콜:**
```
1. Phase-2 checkpoint 로드 (probe + scen gates = 고정 buffer)
2. target dataset 전체 셀을 checkpoint normalizer로 로드 (HI z-score 유지)
   → SOH ratio target은 dataset-agnostic이므로 정규화기 재학습 불필요
3. 셀 split:
     N 셀 (--finetune-cells)  → few-shot train
     20% of remainder         → val (early stopping)
     나머지                   → test (최종 평가)
4. gates가 buffer이므로 optimizer = cap_head 파라미터만 자동 포함
5. before/after 비교 리포트 저장
```

```bash
# MIT 5셀로 fine-tune (HUST→MIT cross-dataset)
python 5_model/finetune_scr.py \
    --checkpoint _5_data_model_scr/0711_1538/checkpoints/best.pt \
    --finetune-dataset MIT \
    --finetune-cells 5

# 하이퍼파라미터 조정
python 5_model/finetune_scr.py \
    --checkpoint ...  --finetune-dataset MIT \
    --finetune-cells 5 --epochs 100 --lr 1e-4 --seed 42
```

**출력 구조:**
```
_5_data_model_scr/{run_id}/ft_{dataset}_{N}shot_{timestamp}/
  before_metrics.json      # fine-tune 전 test RMSE/MAE/R2
  finetune_report.json     # before / after / delta 비교
  checkpoints/best.pt      # fine-tuned checkpoint
  logs/train_log.csv
  figures/scatter_test.png
  metrics/metrics.json
```

---

### 5-13. HI 클러스터링 분석 — `5_model/clustering.py`

Phase 1 학습 전 probe HI 수(m) 결정 보조 도구:

```bash
python 5_model/clustering.py
python 5_model/clustering.py --k-max 9 --mrmr-m 20 --knn 10
```

- **mRMR**: MI(HI_i; level) - 선택된 HI와의 Pearson 중복 → 최적 HI 순위 도출
- **Silhouette + KNN Purity** (k=2..k_max): K-Means 군집 품질 평가
- **LogReg 5-fold CV**: m개 HI로 n_classes-class 분류 정확도 vs m 곡선 → plateau_m 결정

출력: `charge_probe_m`, `discharge_probe_m` 권장값 → `scr.yaml`에 수동 반영

---

## 데이터 디렉토리 구조

```
_0_data_raw/              # 원본 복사본 (MIT .mat / HUST .pkl)
_1_data_unified/          # 변환된 통합 포맷
  MIT/ HUST/              # 셀별 .pkl + .csv
_4_data_hi/               # HI 추출 결과
  clean/                  # hi_correlation.py 입력 (전처리 완료 시계열)
    MIT/ HUST/
  {axis}/                 # 세그멘테이션 축별 서브트리 (예: qfrac/)
    cycle/                # wide-cycle HI (한 행=한 사이클)
      MIT/ HUST/          # b1c0.pkl, 1-1.pkl, ...
    seg/                  # native-seg HI (한 행=한 세그먼트)
      MIT/ HUST/
    scenario_spec.json    # ScenarioSpec (n_scenarios, routing, class_names ...)
_5_data_model_scr/        # 학습 결과
  MMDD_HHMM/              # run_dir (Phase 1 또는 Phase 2)
    scenario_spec.json    # 사용된 ScenarioSpec 복사본
    config.yaml           # 학습 시 yaml 복사본
    gates/
      classification_HIs.json   # probe gate 랭킹 (charge/discharge)
      regression_HIs.json       # scen gate 랭킹 (시나리오별)
      gate_probs.png
    checkpoints/
      best.pt             # val SOH RMSE 최솟값 (normalizer 포함)
      final.pt            # 최종 (model + normalizer)
    figures/ metrics/ routing/ predictions/
    ft_{dataset}_{N}shot_{timestamp}/   # finetune_scr.py 출력
      before_metrics.json
      finetune_report.json
      checkpoints/best.pt
      logs/ figures/ metrics/
common/
  scenario/
    __init__.py           # get_segmenter() 레지스트리
    base.py               # ScenarioSpec, SegmentRecord, Segmenter ABC
    qfrac.py              # QFracSegmenter (기본, 6 시나리오)
    protocol.py           # ProtocolSegmenter
    vwindow.py            # VWindowSegmenter
    rcs.py                # RCSSegmenter
    cluster.py            # ClusterSegmenter
```

---

## Phase 1 → Phase 2 → Fine-tuning 실행 순서

```bash
# 0. (선택) clustering.py로 probe_m 결정
python 5_model/clustering.py

# 1. Step 4: HI 추출 (축 변경 시 --seg-axis 지정)
python 4_hi_analysis/hi_correlation.py               # qfrac 기본
python 4_hi_analysis/hi_correlation.py --seg-axis vwindow
# → _4_data_hi/{axis}/cycle/, seg/, scenario_spec.json 생성

# 2. scr.yaml 확인
#    scenario.axis: qfrac   (또는 사용할 축)
#    data_dir / seg_data_dir: null   ← 자동 결정
#    data.datasets:    ["HUST", "MIT"]
#    data.is_cross_dataset_evaluate: true   ← cross-dataset 실험 시
#    data.gates_from:  null   ← Phase 1

# 3. Phase 1 (L0 gate 학습 → HI 서브셋 탐색)
conda activate LFP_SOH_ESTIMATION
python 5_model/train_scr.py --phase 1
# → _5_data_model_scr/MMDD_HHMM/gates/*.json 생성

# 4. Phase 2 (고정 gate + 회귀 정밀 학습, cap_head만 업데이트)
python 5_model/train_scr.py --phase 2 --gates-from _5_data_model_scr/MMDD_HHMM

# 5. Zero-shot 평가 (논문 Table 메인 결과 — Phase 2 모델, 적응 없음)
python 5_model/test_scr.py --checkpoint _5_data_model_scr/MMDD_HHMM2/checkpoints/best.pt

# 6. Few-shot Fine-tuning (논문 Ablation — N셀 적응 후 성능)
#    내부적으로 before(zero-shot 부분집합) / after(N셀 적응) 동시 비교
python 5_model/finetune_scr.py \
    --checkpoint _5_data_model_scr/MMDD_HHMM2/checkpoints/best.pt \
    --finetune-dataset MIT \
    --finetune-cells 5
# → finetune_report.json: before RMSE(MIT 부분집합) / after RMSE 비교
# → Step 5의 test_scr.py 결과(MIT 전체)가 논문 Table 메인 zero-shot 수치

# (선택) Fine-tuned 모델 전체 재평가 (scatter/커브 플롯이 필요할 때)
# python 5_model/test_scr.py \
#     --checkpoint _5_data_model_scr/MMDD_HHMM2/ft_MIT_5shot_.../checkpoints/best.pt
```

**축별 전체 실행 예시 (vwindow):**

```bash
python 4_hi_analysis/hi_correlation.py --seg-axis vwindow
python 5_model/train_scr.py --phase 1 --seg-axis vwindow
python 5_model/train_scr.py --phase 2 --seg-axis vwindow --gates-from _5_data_model_scr/MMDD_HHMM
python 5_model/test_scr.py
python 5_model/finetune_scr.py --checkpoint ... --finetune-dataset MIT --finetune-cells 5
```

---

## 논문 완성을 위한 향후 수행 목록 (리뷰어 관점)

> **최종 목표**: LFP 배터리 SOH 추정 방법론(SCR 프레임워크)을 크로스-데이터셋 일반화 관점에서 학술 발표.  
> 아래 항목들은 IEEE TII / Journal of Power Sources / Applied Energy 수준 리뷰어가 Accept 판단 전에 요구하는 필수·강화·선택 과제.

---

### I. 실험 필수 항목 (없으면 Reject 가능성 높음)

#### 1. 베이스라인 비교

리뷰어가 가장 먼저 묻는 것: "기존 방법 대비 얼마나 좋은가?"

| 베이스라인 | 유형 | 구현 방법 | 비고 |
|---|---|---|---|
| Severson 2019 feature set | 피처 기반 | 논문 Table S1 피처 → LinearReg / ElasticNet | NATURE ENERGY, 과학계 기준점 |
| XGBoost (hand-crafted HI) | 트리 앙상블 | `data_HI/{axis}/*.pkl` → xgb.train() | 피처 중요도 vs SCR gate 비교 가능 |
| LSTM (raw V-I-t 시계열) | Deep | PyTorch LSTM, 전체 방전 곡선 입력 | raw signal 모델 상한선 |
| Single-segment (probe-only) | SCR 어블레이션 | scen gate 비활성, probe만 → MLP head | 라우팅 효용 검증 |

> 위 표는 아직 미구현 계획 목록이다. 아래 두 베이스라인(`full_cycle`, `raw_mlp`)은 **이미 구현·검증 완료**된 상태이며,
> 각각 "부분 사이클 관측"과 "HI 추출" — 이 프레임워크의 두 핵심 설계 선택 — 이 실제로 기여하는지를 직접 검증한다.

##### 1-A. `full_cycle` 베이스라인 — 부분 관측의 비용(손실폭) 측정

**검증 대상**: q_frac_wide/vqslope 등은 사이클의 일부(예: n2=9~20%)만 관측하는 "부분 사이클" 조건이다.
"그럼 전체 사이클을 다 보면 얼마나 더 잘 맞히는가"를 재는 **상한선(ceiling) 베이스라인**이 없으면,
"부분 사이클로 충분하다"는 논문 핵심 주장의 손실폭 자체를 아무도 검증할 수 없다.

**구현**: `common/scenario/full_cycle.py`의 `FullCycleSegmenter` — 구간 분할을 전혀 하지 않고 방향(충전/방전)당
전체 curve 1개를 그대로 세그먼트로 사용한다. 시나리오는 `chg_full`(0)/`dis_full`(1) 2개뿐이고, 시나리오가
방향만으로 100% 결정되므로 분류 자체가 무의미해 `n_classes=1`, `classifier_default="none"`으로 등록했다
(`common/scenario/test_rs.py`와 동일한 패턴). HI 계산 코드(`5_model/hi_compute.py`)는 그대로 재사용 —
seg가 훨씬 길어져도 morph(DTW/Fréchet) 계열 HI는 고정 50포인트 그리드로 보간 후 계산하므로 연산량이
늘지 않는다(실측: MIT 1개 셀 778 세그먼트, 28초).

**실행**:
```bash
# Step 4: HI 추출 (전체 MIT+HUST, 방향당 curve 전체를 1개 세그먼트로 HI 계산)
python 4_hi_analysis/hi_correlation.py --seg-axis full_cycle --force --workers 8

# Step 6~9: 학습~평가 (run_pipeline로 한 번에, 다른 축과 동일하게 동작)
python run_pipeline.py 6 --seg-axis full_cycle --to-step 9
```
데이터 경로: `_4_data_hi/full_cycle/{seg,cycle}/` — 축 파라미터가 없어 q_frac_wide/vqslope처럼 하위
태그 폴더가 붙지 않고 axis 이름 그대로 쓰인다(`hi_correlation.py`/`train_scr.py`/`train_classifier.py`
모두 "그 외 축" 폴백 규칙을 그대로 타므로 코드 수정이 필요 없었다).

**해석 방법**: `full_cycle`의 test RMSE/R²를 같은 조건(축 설정 외 동일 config)의 q_frac_wide 결과와
나란히 비교한다(`visualize_results.py --runs full_cycle_run q_frac_wide_run`). 그 차이가 "부분 관측으로
인한 정확도 손실"이며, 작을수록 "부분 사이클 관측만으로 충분하다"는 주장이 강해진다.

**주의**: 세그먼트 길이가 훨씬 길어(전체 충/방전 curve) `stat`/`diff`/`lfp` HI의 절대 스케일이 부분
세그먼트와 다를 수 있다 — HI 값 자체를 직접 비교하지 말고 최종 SOH 예측 성능(RMSE/R²)만 비교할 것.

##### 1-B. `raw_mlp` 베이스라인 — HI 추출 자체의 기여도 측정

**검증 대상**: "수작업 HI 64개를 뽑는 게 실제로 가치가 있는가, 아니면 raw 곡선을 그냥 학습기에 넣어도
비슷한가?"라는 질문. CNN(B-4/B-5 계열)을 쓰면 "raw 입력"과 "CNN 아키텍처"가 뒤섞여 원인 구분이
안 되므로, **같은 MLP 구조**에 입력만 HI(게이트로 고른 64차원) 대신 raw로 바꿔 비교한다.

**구현**: `5_model/models/raw_mlp_model.py`의 `RawMLPModel` — HI/게이트를 완전히 우회하고, 이미
`SegmentDataset`이 만들어 두는 `x_raw`(raw_v/raw_i 48포인트 리샘플, 2채널=96차원)를 flatten해
direction/cap_init과 concat한 뒤 평범한 2-hidden-layer MLP로 SOH를 직접 회귀한다. `SCRModel`과 동일한
batch/출력 dict 계약(`cap_pred`/`level_logits`/`probe_z`/`scen_z`)을 따르고, `_fixed_probe=True`/
`_fixed_scen=True`/`probe_mlp=None`으로 선언해 `SCRLoss`의 L0·CE 항이 자동으로 비활성화되므로
**`SCRTrainer`/`SCREvaluator`를 코드 수정 없이 그대로 재사용**한다.

**실행**: `scr.yaml`에 한 줄만 바꾸면 어떤 축(q_frac_wide/vqslope/full_cycle 무엇이든)에도 적용된다.
```yaml
model:
  regression_model: "raw_mlp"   # 기본값 "mlp" 대신
```
```bash
python run_pipeline.py 6 --seg-axis <원하는 축> --model-config 5_model/config/scr.yaml --to-step 9
```
- Step 6/7(Phase 1/2): raw_mlp는 게이트가 없어 phase 구분이 무의미 — 두 phase 모두 동일하게
  `RawMLPModel`을 처음부터 학습한다(모델이 작아 재학습 비용이 낮음, `train_scr.py`).
- Step 8(분류기 학습): `train_classifier.py`가 run의 `config.yaml`에서 `regression_model: raw_mlp`를
  감지하면 "라우팅이 필요 없다"는 메시지와 함께 자동으로 건너뛴다.
- Step 9(평가): `test_scr.py`가 raw_mlp 체크포인트를 감지하면 `RawMLPModel`로 재구성하고, 분류기
  로딩·HI 게이트 로딩·routing heatmap을 모두 스킵한 뒤 **oracle 전용**으로 자동 평가한다(hard/soft
  라우팅은 HI 기반 시나리오 분류 개념이 있어야 의미가 있는데 raw_mlp는 seg_idx를 직접 입력받으므로
  해당 없음).

**해석 방법**: raw_mlp 결과를 같은 축의 HI 기반 SCR 결과와 비교한다. HI 기반이 확연히 좋으면
"수작업 HI 추출이 실제로 정보를 압축·정제하는 가치가 있다"는 근거가 되고, raw_mlp가 비슷하거나
더 좋으면 HI 설계(선택된 카테고리/통계량)를 재검토해야 한다는 신호다.

#### 2. 양방향 크로스-데이터셋 평가

현재 train(MIT)→test(HUST)만 확인된 것으로 보임. 리뷰어는 반드시 **양방향**을 요구한다.

```bash
# 방향 A: MIT 학습 → HUST 테스트 (현재 구현)
# scr.yaml: datasets: [MIT, HUST], cross_dataset_evaluate: true
python 5_model/train_scr.py --phase 1 && python 5_model/train_scr.py --phase 2

# 방향 B: HUST 학습 → MIT 테스트
# scr.yaml: datasets: [HUST, MIT], cross_dataset_evaluate: true
python 5_model/train_scr.py --phase 1 && python 5_model/train_scr.py --phase 2
```

- **왜 중요한가**: 결과가 비대칭(한 방향만 좋음)이면 방법론 자체의 일반화 한계 노출  
- **예상 난점**: MIT 충전 6C→1C는 HUST 1C와 프로토콜이 다름 → qfrac/vwindow 축이 protocol 축보다 유리할 가능성

#### 3. 통계적 유의성 (복수 seed 반복)

단일 seed 결과는 리뷰어가 "운이 좋았다"고 기각할 수 있음.

```bash
for seed in 0 1 2 3 4; do
    python 5_model/train_scr.py --phase 1 --seed $seed
    python 5_model/train_scr.py --phase 2 --seed $seed --gates-from ...
    python 5_model/test_scr.py --checkpoint ...
done
```

보고 형식: **mean ± std (5-seed)** — 표 아래 footnote로 기재

---

### II. 분석 강화 항목 (없으면 Major Revision 확정)

#### 4. 어블레이션 연구 (Ablation Study)

각 설계 결정의 기여도를 정량화해야 한다.

| 변형 실험 | 목적 | 기대 결과 |
|---|---|---|
| 축 비교 (5개 전체) | 세그멘테이션 축 선택의 영향 | vwindow > qfrac > protocol > rcs > cluster 예상 |
| HI 카테고리 기여도 (전기화학/통계/dVdQ/용량) | 어떤 물리량이 핵심인가 | dVdQ·전기화학 계열이 높을 것으로 예상 |
| Gate 비활성화 (probe 제거 / scen 제거 / 둘 다 제거) | L0 라우팅의 효용 | Gate 포함 > probe-only > scen-only > 없음 |
| n_windows(K) 민감도 (K=2,3,4) | 파라미터 튜닝 불필요성 검증 | K=3에서 plateau 확인 |

#### 5. 선택된 HI의 물리 해석

리뷰어: "gate가 선택한 HI가 전기화학적으로 의미 있는가?"

- `routing/routing_table.csv`에서 top-HI 추출
- `docs/HI_DESCRIPTION.md`와 대조 → 물리 의미 연결
- 논문 Table 또는 Figure: "selected HI list + physical interpretation + degradation mechanism"
- 예시 연결: `dvdq_peak_v` 선택 → "Li plating 전위 변화 (Dahn 2012)" 인용

#### 6. dvdq_peak_q / dvdq_valley_q 절대량 → 정규화 변환

현재 `dvdq_peak_q`, `dvdq_valley_q`는 절대 Ah 단위 → 셀 용량이 다른 크로스-데이터셋에서 도메인 갭 직접 유발.

```python
# hi_extractor.py 수정 대상
hi["dvdq_peak_q"]   = peak_q / q_total   # q_frac 단위 (0~1)
hi["dvdq_valley_q"] = valley_q / q_total
```

- MIT (3Ah) vs HUST (2Ah): peak_q 절대값이 1.5배 차이 → 같은 피처 축에서 군집 분리 발생
- 수정 후 게이트 재학습 → 크로스-데이터셋 RMSE 변화 측정 (기대: 감소)

---

### III. 모델 개선 항목 (있으면 논문 강화, Accept 확률 ↑)

#### 7. Few-shot 적응 커브 (N-shot curve)

현재 finetune_scr.py로 단일 N 실험만 가능. 논문 그림으로 필요한 것:

```
RMSE
  |▓▓▓ Zero-shot (N=0)
  |▓▓  
  |▓   N=1
  |    N=3   N=5   N=10
  └──────────────────────→ N (# adaptation cells)
```

```bash
for n in 1 3 5 10; do
    python 5_model/finetune_scr.py --finetune-cells $n ...
done
```

참고 문헌: [Deng 2022 MAML 기반 배터리 few-shot], [Liu 2023 meta-learning SOH]

#### 8. 도메인 적응 (선택적, 논문 차별화)

Zero-shot 결과가 경쟁력 없을 경우:

| 방법 | 적용 위치 | 참고 |
|---|---|---|
| MMD (Maximum Mean Discrepancy) | HI 공간에서 소스/타겟 분포 정렬 | Pan 2010 |
| DANN (Domain-Adversarial NN) | Probe 뒤에 domain discriminator 추가 | Ganin 2016 |
| BOL 기준 상대화 피처 | HI 자체를 1사이클 기준으로 정규화 | Severson 2019 변형 |

#### 9. 불확실성 정량화 (UQ, Uncertainty Quantification)

실용 배포 관점에서 confidence interval이 없으면 신뢰성 주장 약화.

- **MC Dropout**: cap_head dropout 유지, inference 시 T=50 forward pass → mean/std
- **Deep Ensemble**: 3-5 seed 모델의 예측 ensemble → 분산 = epistemic uncertainty
- 보고 지표: ECE (Expected Calibration Error), PICP (Prediction Interval Coverage)

---

### IV. 미해결 코드 과제 (논문 제출 전 수정 필요)

| 항목 | 위치 | 현재 상태 | 수정 방향 |
|---|---|---|---|
| `THETA_FLAT` 문서 불일치 | `docs/HI_DESCRIPTION.md` | 0.05V로 기재, 코드는 0.25V | 코드 재확인 후 문서 수정 |
| `dvdq_peak_q` 단위 | `hi_extractor.py` | 절대 Ah → 크로스-DS 편향 | `peak_q / q_total` q_frac 변환 |
| cluster 축 크로스-DS 검증 | `cluster.py` | train K-means의 test 일반화 미검증 | HUST train 클러스터 → MIT test 적용 실험 |
| 재현성 문서화 | 전체 | seed=42 기본값 있으나 FRAMEWORK 미기재 | Phase 1/2/finetune seed 전파 정책 명시 |

---

### V. 논문 구조 제안 (IEEE TII 기준 6섹션)

```
§1 Introduction       — LFP 배터리 노화, 크로스-DS 일반화 문제
§2 Related Work       — 피처 기반(Severson), DL(LSTM/Transformer), 부분 충전(Deng)
§3 Methodology        — 5단계 파이프라인, SCR 모델, 세그멘테이션 축 5종
§4 Experimental Setup — MIT/HUST 데이터셋 사양, 양방향 cross-DS 분할
§5 Results            — 베이스라인 비교 / 어블레이션 / 물리 해석 / Few-shot
§6 Conclusion
Appendix              — HI 전체 목록 (148D), 축별 ASCII 분할 다이어그램
```

---

> **체크리스트 (제출 직전)**
> - [ ] 5-seed 반복 실험 완료 (mean ± std)
> - [ ] 양방향 크로스-DS 결과 표 완성
> - [ ] dvdq_peak_q q_frac 변환 후 gate 재학습
> - [ ] THETA_FLAT 문서-코드 일치 확인
> - [ ] 베이스라인 4종 RMSE/MAE 표 완성
> - [ ] 어블레이션 5종 실험 완료
> - [ ] 선택 HI 물리 해석 표 작성
> - [ ] Few-shot 커브 그림 (N=0/1/3/5/10)
