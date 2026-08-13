# 260811 — 프레임워크 설계 요소별 기여도 정량화 실험

실험 1: `full_cycle` 베이스라인으로 partial cycle 성능 하락 정량화.
실험 2: `random`(= `rcs.py`의 `RCSSegmenter`, `assign="none"` 모드) 베이스라인으로 시나리오(존 타이핑) 개념의 기여도 정량화.

---

# 실험 1: `full_cycle` 베이스라인으로 partial cycle 성능 하락 정량화

## 0. 요청 원문 요약

실험 1: 현재 프레임워크에 대해 "partial cycle 사용"이 성능에 미치는 손실을 측정한다.
방법은 (충/방전만 나눈) **full cycle**을 입력으로 HI를 추출하고, 나머지는
`docs/260810_RESULTS.md`의 실험 설계(OFAT, baseline 비교)를 그대로 따른다. 단
"이 경우는 시나리오 축 개념이 없으므로 적절히 관련 코드를 수정해야 함"이 사용자가
명시한 전제였다.

**이번 문서는 실행이 아니라 설계 문서다.** 코드는 아직 아무것도 바꾸지 않았고,
아래는 "실험 1을 수행하려면 어떤 파일을 어떻게 바꿔야 하는지"를 실제 코드를 읽고
검증한 결과다.

---

## 1. 핵심 결론 먼저

**`.py` 코드 파일은 한 줄도 바꿀 필요가 없다.** `full_cycle` 축은 이미
`common/scenario/full_cycle.py`에 완전히 구현되어 있고, 하류 파이프라인
(`hi_correlation.py`/`train_scr.py`/`train_classifier.py`/`scr_evaluator.py`/
`test_scr.py`/`segment_dataset.py`)은 시나리오 축의 `n_scenarios`/`n_classes`/
`routing`을 전부 `ScenarioSpec`에서 동적으로 읽으므로 "시나리오 축 개념이 없는"
축(2시나리오·1클래스)도 이미 일반화된 경로로 처리된다. 이는 `docs/PIPELINE.md`
"1-A. `full_cycle` 베이스라인" 절이 "이미 구현·검증 완료"라 적어둔 것과 일치한다.

**단, 코드가 아니라 데이터가 낡았다.** `_4_data_hi/full_cycle/`에 이미 추출된
캐시(`D:\...\full_cycle\seg\{MIT,HUST}\*.pkl`, MIT 123 + HUST 77 파일)가 있지만,
파일 생성 시각이 **2026-07-27**(커밋 `b722bf4` 직후)로, 이후 `hi_correlation.py`에
들어간 **RAW_CH 2→3 재설계(부호 있는 전류 + 상대시간 채널, 커밋 `a5b8ac3`,
2026-08-04)보다 앞선다**. 아래 §3에서 실제 pkl을 열어 확인한 대로, 이 캐시는
`raw_t` 컬럼이 아예 없고 `raw_i`가 전부 양수(부호 없음, 절댓값)다 — 260810
baseline이 쓰는 현재 스키마와 다르다. 따라서 **Step 4(HI 추출)를 `--force`로
재실행**해야 260810 baseline과 동등한 조건으로 비교할 수 있다. 이건 "파일 변경"이
아니라 "데이터 재생성"이다.

---

## 2. 배경 — 왜 `full_cycle`이 partial cycle의 성능 손실을 측정하는 척도가 되는가

이 프로젝트의 표준 프레임워크(`q_frac_ref`/`q_frac_wide` 등)는 한 사이클의
충전·방전 곡선을 여러 개의 부분 구간(zone)으로 쪼갠 뒤, 그중 일부(`n1`~`n2` 구간
등)만 관측했다고 가정하고 HI를 뽑는다 — 실제 배터리 관리 시스템(BMS)이 전체
충방전 곡선을 항상 볼 수 없고 부분 관측만 가능한 상황을 모사하기 위해서다. 이
설계가 "관측 가능한 정보량을 줄여서 얻는 대가"가 곧 이 프로젝트가 감수하는
성능 손실이다.

`full_cycle` 축은 이 손실의 **상한선(ceiling)** 을 재는 대조군이다: 구간 분할 없이
충전 전체 1개 + 방전 전체 1개(`n_scenarios=2`)를 그대로 하나의 세그먼트로 취급해
HI를 뽑는다. 이 축의 성능이 곧 "정보 손실이 전혀 없을 때"의 이론적 상한이고,
`q_frac_ref` baseline(R²=0.9524, RMSE=1.520%, `docs/260810_RESULTS.md` §10.3
1번 행)과의 격차가 partial cycle 사용이 초래하는 순수 비용이다.

---

## 3. 코드 검증 결과 — 파일별 점검

### 3.1 `common/scenario/full_cycle.py` — 변경 불필요 (이미 구현됨)

```python
_SCENARIO_NAMES = ["chg_full", "dis_full"]
_ROUTING = [[0], [1]]

class FullCycleSegmenter(Segmenter):
    name = "full_cycle"
    def __init__(self, min_pts: int = 10):
        self.min_pts = min_pts
    def get_spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            axis="full_cycle", n_scenarios=2, scenario_names=_SCENARIO_NAMES,
            n_classes=1, class_names=["full"], routing=_ROUTING,
            classifier_default="none", params={"min_pts": self.min_pts},
        )
    # iter_segments(): 사이클당 충전 1개(direction=+1) + 방전 1개(direction=-1)만
    # SegmentRecord로 반환 — 존(zone) 분할 없음, q_frac_lo/hi는 항상 0.0/1.0
```

`test_rs.py`(과거 E1/E2/E3 테스트 시나리오, `n_classes=1`/`classifier_default="none"`)와
정확히 같은 패턴이라 이미 하류 코드가 이 패턴을 한 번 검증한 전례가 있다.
`min_pts=10`은 baseline과 동일한 기본값이라 별도 태그도 안 붙는다.

### 3.2 `common/scenario/__init__.py` — 변경 불필요 (이미 등록됨)

```python
from .full_cycle import FullCycleSegmenter
REGISTRY["full_cycle"] = FullCycleSegmenter
```

`--seg-axis full_cycle`이 모든 스크립트(`hi_correlation.py`/`run_pipeline.py`/
`train_scr.py`)에서 즉시 인식된다.

### 3.3 `4_hi_analysis/hi_correlation.py` — 코드 변경 불필요, **데이터 재추출 필요**

HI 계산 함수(`_seg_stat`/`_seg_diff`/`lfp_*`/`morph_*`)는 세그먼트의 `v/i/dt/q`
배열만 받아 축과 무관하게 동작한다 — 실제로 기존 캐시의 컬럼을 열어보면 66개
HI(stat 20 + diff 20 + lfp 20 + morph 6, `SOH_EXCLUDE_STAT_LEAK` 미지정 시 기준)가
이미 현재 스키마와 이름이 정확히 일치한다. 즉 **HI 피처 정의 자체는 낡지 않았다.**

문제는 raw 채널이다. 실제로 캐시를 열어 확인한 결과:

```python
>>> df = pd.read_pickle(".../full_cycle/seg/MIT/b1c0.pkl")
>>> "raw_t" in df.columns
False
>>> np.asarray(df.iloc[0]["raw_i"]).min(), np.asarray(df.iloc[0]["raw_i"]).max()
(0.0479, 3.6004)   # 전부 양수 — 부호 없음
```

`segment_dataset.py:_build_raw_tensor`의 주석("`raw_t` 컬럼만 없는 구 pkl
(RAW_CH=2 시절 캐시)은 `raw_t` 채널만 0으로 채운다")이 정확히 이 상태를 가리킨다.
즉 지금 이 캐시로 그대로 학습해도 **에러는 나지 않는다** — 다만 RawCNN 입력의
상대시간 채널이 전부 0으로 죽고, 전류 채널이 방향 정보 없는 절댓값으로 들어가
260810 baseline과 입력 표현이 달라진다(baseline 실험들은 전부 `a5b8ac3`
이후 재추출된 데이터를 쓴다). 공정한 비교가 아니므로:

```powershell
python 4_hi_analysis/hi_correlation.py --seg-axis full_cycle --force --workers 40
```

로 재추출해야 한다. `docs/PIPELINE.md`가 측정해둔 대로 셀 1개당 약 28초
(778 세그먼트) 수준이라 MIT+HUST 200셀 전체를 재추출해도 시간 비용은 크지 않다.
Step 4(추출) 단계에는 `n1`/`n2`/`noise_amp` 같은 존 파라미터가 아예 없으므로
(`params`엔 `min_pts`만 존재) 이 플래그들은 주지 않는다 — 주더라도
`FullCycleSegmenter`가 받는 인자가 아니라 무시된다. (`scen_k`는 여기 Step 4가
아니라 Step 6 학습 시점 파라미터라 이 목록과 무관 — §4에서 baseline과 동일하게
65로 고정함.)

### 3.4 `5_model/train_scr.py`, `5_model/train_classifier.py` — 변경 불필요

`train_classifier.py`는 `spec.n_classes` 개수만큼 로짓을 내는 분류기를 그대로
만든다(`n_classes=1`이면 `CrossEntropyLoss`가 항상 정답인 단일 클래스를 예측하는
퇴화 케이스가 되지만 에러는 나지 않는다 — softmax(1클래스)=1.0, loss=0). 다만
`full_cycle`은 이미 `direction`만으로 시나리오가 100% 결정되므로(존 분할이 없어
분류할 대상 자체가 없음) 분류기 학습(Step 7)이 정보량 0인 형식적 스텝이 된다 —
스킵해도 되지만, `docs/260810_RESULTS.md`가 baseline에서 실제로 사용한 절차
(`classifier.type: cnn` 포함 표준 Step 4~9 흐름)와 최대한 동일하게 두기 위해
**그대로 실행**해도 무방하다(비용도 작다).

`classifier_default` 필드는 `ScenarioSpec`에 메타데이터로만 저장되고 어떤
스크립트도 이를 읽어 Step을 자동으로 스킵하지 않는다(전수 grep 확인, `run_pipeline.py`
에 `classifier_default` 참조 없음) — 즉 "none"이라고 자동으로 뭔가 달라지는
동작은 없다. `train_scr.py`의 Phase1(HardConcrete 게이트)도 `model.n_scenarios`를
동적으로 읽어 게이트 텐서 크기를 정하므로(§260810 문서에서 이미 확인된 패턴)
`n_scenarios=2`에서도 그대로 동작한다.

### 3.5 `5_model/evaluation/scr_evaluator.py`, `5_model/test_scr.py` — 변경 불필요

`_LV_COLOR`/`_LV_LS`는 `_spec.n_classes`로부터 동적으로 만들어지고(하드코딩된
`3`/`6` 없음, 전수 grep 확인), `classification_metrics()`엔 이미 **"정답 레짐
라벨이 단일값(placeholder)이면 정확도 무의미"** 가드가 있다(구 `test_rs`
`n_classes=1` 사례를 위해 만들어진 코드, `scr_evaluator.py:405-413`) — `full_cycle`도
`level`이 항상 0이라 정확히 이 가드에 걸려 "note" 메시지로 대체되고 에러 없이
넘어간다. hard/soft 라우팅은 `routing=[[0],[1]]`이 사실상 `direction`과 동일한
정보이므로 oracle/hard/soft 세 라우팅이 전부 같은 결과로 수렴하는 특수 케이스가
된다 — 이 자체가 실험 1에서는 의미 있는 관찰(라우팅 방식 간 차이가 원천적으로
없어짐)이라 별도 처리가 필요 없다.

### 3.6 `5_model/datasets/segment_dataset.py` — 변경 불필요

`direction`/`level`/`seg_idx`는 pkl에 저장된 `scen`(=scenario_id) 컬럼과
`_spec.scenario_to_dir_class(seg_idx)`로 로드 시점에 동적으로 파생된다
(`segment_dataset.py:114-123, 255-261`) — 축마다 새로 구현할 필요 없이 이미
범용이다. `full_cycle`의 `routing=[[0],[1]]`을 넣으면 `scen=0→direction=+1,
level=0`, `scen=1→direction=-1, level=0`이 자동으로 나온다(실제 캐시에서
`scen` 컬럼 존재 확인함, §3.3).

### 3.7 `run_pipeline.py` — 변경 불필요, **`--axis-config "{}"` 명시 필수** (2026-08-11 실행 중 발견)

`--n1`/`--n2`/`--noise-amp` 등 CLI 플래그는 전부 `default=None`이라 생략해도
안전하다(단 `--scen-k`는 예외 — §4에서 바로잡듯 이건 축 파라미터가 아니라
시나리오별 HI 게이트 예산이라 `full_cycle`에도 그대로 적용되므로 baseline과
맞춰 65를 명시해야 한다). 하지만 **`--axis-config`를 생략하면 안전하지 않다** — 이건
실제로 `run_pipeline.py 6 --seg-axis full_cycle --model-config
exp_qfw_mlp_basefix.yaml`을 실행해보고서야 드러난 문제로, §3.7 원래 서술
("생략해도 에러 없음")은 틀렸다.

**원인**: `train_scr.py:446`이 `_axis_cfg`를 `args.axis_config or
cfg["scenario"]["axis_config"]`(즉 CLI로 안 주면 **모델 yaml의 값으로 폴백**)로
정한다. `exp_qfw_mlp_basefix.yaml`엔 원래 축(q_frac_wide)용으로
`axis_config: {n1: 0.35, n2: 0.2, n_samples: 4}`가 박혀 있는데, `--seg-axis
full_cycle`은 축 **이름**만 바꿀 뿐 이 axis_config를 비우지 않는다. 그 결과
`get_segmenter("full_cycle", {"full_cycle": {n1:..,n2:..,n_samples:..}})` →
`FullCycleSegmenter(**{n1:..,n2:..,n_samples:..})`가 호출되는데
`FullCycleSegmenter.__init__`은 `min_pts` 하나만 받으므로
`TypeError: got an unexpected keyword argument 'n1'`이 발생한다(실제 재현 로그,
Step 6 3초 만에 실패).

**해결**(코드 수정 아님, 실행 커맨드에 플래그 추가): `run_pipeline.py`에
`--axis-config "{}"`를 명시하면 `args.axis_config`가 truthy해져 yaml 폴백을
막고, `_axis_cfg = json.loads("{}") = {}` → `FullCycleSegmenter(**{})` →
`min_pts` 기본값(10)으로 정상 생성됨을 직접 재현해 확인했다. `run_pipeline.py`는
`--axis-config`가 있으면 Step 6~9 전 스텝의 `step_extra`에 동일하게 실어 보내고
(`run_pipeline.py:342-343`), `train_classifier.py`/`test_scr.py`는 애초에
`get_segmenter`를 직접 호출하지 않고 Step 6이 저장한 `config.yaml`의
`scenario.axis_config`를 읽으므로(각각 271/287, 526/544행), Step 6에서 한 번만
바로잡으면 Step 7~9까지 전부 올바른 빈 axis_config로 전파된다 — 스텝별로
플래그를 반복할 필요는 없다.

---

## 4. 실험 설계 (260810 baseline과의 단일 변수 비교)

### 고정값 (260810 baseline과 동일하게 유지)

| 항목 | 값 | 근거 |
|---|---|---|
| HI(`stat_q_abs`/`stat_energy_seg`) | 미포함(`SOH_EXCLUDE_STAT_LEAK=1`) | baseline과 동일 조건 유지 |
| shape filter(전처리 필터7) | 적용(기본) | baseline과 동일, Step 2 단계라 축과 무관 |
| `min_pts` | 10(기본) | baseline과 동일, 기존 캐시도 이미 이 값으로 추출됨 |
| `scen_k`(`--scen-k`) | **65** | baseline과 동일. **주의**: 이건 존 경계 파라미터가 아니라 `model.scen_gates[s]`(시나리오별 HardConcrete 게이트, `scr_model.py:94`)가 `N_HI`개 HI 중 몇 개를 살릴지 정하는 예산이라, 시나리오 수가 6개(q_frac_ref)든 2개(full_cycle)든 게이트 자체는 항상 존재한다 — "시나리오 축 개념이 없어 적용 불가"인 n1/n2/noise류와는 성격이 다르다. `N_HI=64`(HI 미포함 기준)에 `scen_k=65`를 주면 이전에 이미 확인된 no-op(파이썬 리스트 슬라이싱 `ranked[:k]`가 `k>len`이면 전체 반환)이 여기서도 그대로 재현되어 게이트가 사실상 무제한이 된다 — baseline과 동일한 동작이므로 그대로 맞춰야 비교가 성립한다 |
| `regression_model` | mlp | baseline과 동일(모델 종류 비교는 실험 8의 영역) |
| raw 융합 | 없음 | baseline과 동일(raw 융합 비교는 실험 7의 영역) |
| 데이터셋 | MIT+HUST 풀링, oracle/hard/soft 표준 평가 | baseline과 동일 |
| `--model-config` | `5_model/config/exp_full_cycle.yaml`(신설, 2026-08-11) | `exp_qfw_mlp_basefix.yaml` 복사 + `scenario.axis=full_cycle`/`axis_config={}`/`regression.scen_k_count=65`/**`training.lr=2.0e-4`**(baseline 5.0e-4에서 하향, 아래 근거)만 변경 |

### 유일한 변수(의도된 것)

`--seg-axis`: `q_frac_ref`(baseline, 존 분할 있음) vs `full_cycle`(존 분할 없음,
충전 전체/방전 전체 각 1세그먼트).

### peak LR 하향(2.0e-4, baseline 5.0e-4) — 2026-08-11 추가 조정

이건 "다른 실험 중 고정값"을 어기는 변경이라 원칙적으로는 실험 오염 요인이다.
다만 `full_cycle`은 사이클당 세그먼트가 최대 2개(충전 1+방전 1)뿐이라 — §2 대비
`q_frac_ref`는 `n1`~`n2` 구간을 `n_samples=4`로 반복 샘플링해 사이클당 세그먼트가
훨씬 많다 — 동일 `batch_size=2048` 하에서 에폭당 배치 수·gradient 통계가 근본적으로
달라진다. `docs/260810_RESULTS.md` §8-5에서 이미 확인된 전례(MIT-only처럼 표본
구성이 baseline과 달라지면 Phase2가 warmup 구간에서 조기 발산하는 패턴, R²=-0.2038
사례)를 감안해, baseline과 동일한 `lr=5.0e-4`로 먼저 시도하지 않고 **선제적으로**
`lr=2.0e-4`로 낮춰 시작한다. **주의**: 이 때문에 `full_cycle` vs baseline 비교는
엄밀히는 "존 분할 유무"와 "peak LR" 두 변수가 동시에 바뀐 비교가 된다 — 결과
해석 시 이 점을 명시해야 하고, 여유가 되면 `lr=5.0e-4`로도 한 번 더 돌려 발산
여부 자체를 별도로 확인하는 편이 인과관계를 깨끗하게 분리할 수 있다.

### full_cycle 쪽에는 진짜로 존재하지 않는(=적용 불가) 260810 변수들

`n1`/`n2`(존 경계 자체가 없음), `noise_amp`/`noise_mode`/`noise_period`
(q_frac_ref 전용 레퍼런스 노이즈 주입 — full_cycle은 레퍼런스 사이클 개념이
없음)만 `full_cycle`의 `params`에 없는 진짜 "해당 없음"이다. **`scen_k`는 여기서
제외** — 위 표대로 65로 고정해서 넘겨야 한다(2026-08-11, 사용자 지적으로 정정).

---

## 5. 실행 명령

```powershell
# Step 4 — HI 재추출 (구 캐시가 RAW_CH=2 시절 스키마라 재추출 필수, §3.3)
python 4_hi_analysis/hi_correlation.py --seg-axis full_cycle --force --workers 40

# Step 6~9 — Phase1(게이트)+분류기+Phase2(회귀)+평가
$env:SOH_EXCLUDE_STAT_LEAK="1"
python run_pipeline.py 6 --model-config 5_model/config/exp_full_cycle.yaml --seg-axis full_cycle --axis-config "{}" --scen-k 65 --workers 40
```

`--model-config`를 baseline의 `exp_qfw_mlp_basefix.yaml`이 아니라 전용으로 새로
만든 `5_model/config/exp_full_cycle.yaml`로 바꿨다(`scenario.axis=full_cycle`,
`axis_config={}`, `regression.scen_k_count=65`, **`training.lr=2.0e-4`** 를 이미
yaml 안에 반영해둠 — 위 "peak LR 하향" 절 참고). CLI의 `--axis-config "{}"`/
`--scen-k 65`는 yaml에 이미 같은 값이 들어있어 사실 없어도 되지만, §3.7의
`TypeError` 재발 방지를 위해 이중으로 명시해 안전하게 둔다(CLI가 항상 yaml을
덮어쓰므로 값이 어긋날 위험은 없음). `--n1`/`--n2`/`--ref-lag`/`--noise-amp`/
`--noise-mode`/`--noise-period`/`--n-samples`는 애초에 full_cycle에 없는
파라미터라(§4) 아예 주지 않는다.

---

## 6. 비교 판정 기준

| 비교 | 의미 |
|---|---|
| `full_cycle` R²/RMSE/MAPE vs `q_frac_ref` baseline(R²=0.9524, RMSE=1.520%, MAE=1.133%, MAPE=1.251%) | 격차가 partial cycle 사용의 순수 비용(상한 대비 손실) |
| `full_cycle`의 hard/soft/oracle 라우팅 결과가 서로 동일한지 | §3.5에서 예측한 대로 라우팅 방식이 무의미해지는지 확인(예측이 맞으면 세 값이 정확히 같아야 함 — 다르면 코드 어딘가 가정이 깨진 것) |
| 학습/추론 속도 비교 | 존 분할이 없어 세그먼트 수 자체가 줄어드므로(사이클당 최대 2개) epoch당 학습 시간도 함께 기록해 "정보량-속도" 트레이드오프까지 같이 보고 |

결과가 나오면 `docs/260810_RESULTS.md` §10.3 스타일의 표에 `full_cycle` 행을
추가하고, 격차 크기에 따라 §10.5(게재 가능성 평가)의 "partial cycle observation"
주장에 정량적 근거를 붙인다.

---

## 7. 남은 리스크 / 확인 필요 사항

- **HI 컬럼 정의가 66개로 일치함은 확인했지만, `_seg_stat`/`_seg_diff` 내부 계산
  로직 자체가 07-27 이후 미세하게 바뀌었을 가능성은 완전히 배제하지 못했다**
  (git diff에 함수 본문 변경이 일부 섞여 있었으나 전수 검토는 하지 않음) — 어차피
  §3.3에서 재추출을 권고하므로 실질적 영향은 없다(재추출하면 현재 로직으로
  통일됨).
- Step 7(분류기 학습)이 `n_classes=1`로 형식적으로만 도는 것을 그대로 둘지,
  아예 스킵(`run_pipeline.py 8`로 Step 4~7 건너뛰고 Step 8부터, 단 이 경우
  `--gates-from`으로 Phase1 게이트 run을 명시해야 함 — 실험 7의 전례)할지는
  실행 전 결정 필요. 결과에는 영향 없고 실행 시간만 다르다.
- `full_cycle`은 세그먼트 수가 사이클당 최대 2개(충전 1 + 방전 1)라 전체 학습
  샘플 수가 `q_frac_ref`보다 훨씬 적다 — 이는 실험의 의도된 특성(정보량 상한
  측정)이지만, 샘플 수 차이 자체가 일반화 성능에 미치는 별도 효과와 "정보량
  증가 효과"가 섞일 수 있음을 결과 해석 시 유의해야 한다.

---

## 8. 실행 결과 — Phase2 재조정 후 `scen_k` 스윕 (2026-08-11)

앞서 §Phase1/Phase2 분석 대화에서 baseline 설정(`mlp_hidden_dims=[512,256,128,64]`,
`lr=2e-4`, `weight_decay=1e-4`, `patience=30`) 그대로는 Phase2가 5 에폭 만에
과적합 붕괴(test R²=0.11)한다는 걸 확인했고, 데이터량(baseline의 1/3)에 맞춰
모델을 축소·정규화 강화한 뒤(`mlp_hidden_dims=[128,64]`, `weight_decay=1e-3`,
`patience=100`) 재학습했다. 이 조정된 설정을 고정하고 `scen_k`만 스윕한 4개
run의 실제 결과:

| k(`scen_k`) | run 디렉터리 | test R²(oracle) | RMSE | MAE | MAPE | best epoch | avg_scen_his | `q_frac_ref` baseline R²/RMSE/MAE/MAPE(같은 k) |
|---|---|---|---|---|---|---|---|---|
| 15 | `0811_1612_p2_mlp_full_noleak` | **0.9818** | 0.939%p | 0.647%p | 0.732% | 75/175 | 15.0 | 0.9572 / 1.441%p / 1.075%p / 1.178%(MLP_S) |
| 25 | `0811_1628_p2_mlp_full_noleak` | 0.9811 | 0.957%p | 0.589%p | 0.667% | 78/178 | 25.0 | 0.9493 / 1.569%p / 1.151%p / 1.280% |
| 35 | `0811_1620_p2_mlp_full_noleak` | **0.9819** | 0.936%p | 0.575%p | 0.659% | 93/193 | 35.0 | (없음) |
| 65(전체 HI) | `0811_1605_p2_mlp_full_noleak` | 0.9146 | 2.035%p | 1.731%p | 1.864% | 77/177 | 64.0 | 0.9524 / 1.520%p / 1.133%p / 1.251% |

(baseline 출처: `docs/260810_RESULTS.md` §10.3 — k=25는 `0809_1932_p2_mlp_q_fr_35%_20%_noleak`,
k=65는 baseline 자신 `0809_1118_p2_mlp_q_fr_35%_20%_noleak`(둘 다
`mlp_hidden_dims=[512,256,128,64]`, 원래 설정 그대로). k=35는 baseline 쪽에
해당 스윕 지점이 없어 비교 불가. **주의**: k=25/65 baseline은 `full_cycle`
(`mlp_hidden_dims=[128,64]`)과 **MLP 크기가 다른 상태의 비교**다(재조정하지
않은 이유·영향 평가는 하지 않음). **k=15만 예외**: baseline도
`mlp_hidden_dims=[128,64]`(MLP_S)로 새로 돌린 `0812_1210_p2_mlp_q_fr_35%_20%_noleak`
(`gates_from=0812_1115_p1_q_fr_35%_20%_noleak`)로 교체해서, 이 행만은 MLP
크기가 양쪽 다 같은(=`full_cycle`과 대칭인) 비교다. 4개 `full_cycle` run
전부 `gates_from=0811_1536_p1_full_noleak`로 Phase1 게이트 공유,
`lr=2e-4`/`weight_decay=1e-3`/`patience=100`/`mlp_hidden_dims=[128,64]` 동일 —
유일한 변수는 `scen_k`. 4개 run 전부 oracle=hard=soft 정확히 동일함을 직접
확인함, §2 표에서 예측한 대로.)

### 해석

과적합 문제(§Phase1/Phase2 분석)가 모델 축소+정규화 강화로 실제로 해결됐다 —
k=15/25/35에서 test R²가 **0.98 안팎**까지 올라왔다.

**§4의 OFAT 원칙(고정값: `scen_k=65`, baseline과 동일)대로 비교해야 할 지점은
"scen_k=65" 행이다 — 여기서 `full_cycle`(R²=0.9146)이 baseline(R²=0.9524)보다
못하다.** §2에서 `full_cycle`을 "정보 손실이 없는 이론적 상한(ceiling)"이라고
정의했으므로, 원래 기대는 `full_cycle` ≥ baseline이어야 한다 — **k=65 조건의
이 결과는 그 기대와 반대다.** (이전 버전 문서에서 이걸 "원래 가설과 방향이
일치한다"고 적었던 건 서술 오류였다 — ceiling 가설과는 반대 방향이 맞다.)

가장 그럴듯한 설명은 §7에서 이미 지적했던 표본 수 차이다: `full_cycle`은
사이클당 세그먼트가 2개뿐인 반면, baseline은 사이클당 6개다(zone 3개 × 방향
2개 — `n_samples=4`가 zone당 4개 후보를 만들지만 `hi_correlation.py:1979-1997`의
`row[f"raw_v_{seg}"] = ...` 같은 `seg_name` 키 덮어쓰기 때문에 실제로는 zone당
마지막 1개만 최종 데이터에 남는다는 걸 실제 pkl로 직접 확인함 — 즉 baseline도
"24개 후보 중 6개만 생존"이라, `full_cycle`보다 여전히 3배 더 많고 더 국소화된
학습 신호를 갖는다). 또한 `stat_*`/`diff_*` 등 다수의 HI가 구간 전체를
평균·집계하는 방식이라, `full_cycle`처럼 여러 전기화학적 구간(플래토 진입/이탈,
CC/CV 전환 등)이 뒤섞인 훨씬 긴 창에서 계산하면 baseline의 zone-국소화된
세그먼트보다 노화 신호가 희석될 가능성이 있다 — "더 넓은 구간을 본다"가
자동으로 "더 유용한 HI"로 이어지지 않는다는 뜻이다. 둘 다 사후 해석이라 확정은
아니다.

`scen_k=15/25/35`에서 baseline보다도 높은 R²가 나온 것은 별도로 흥미로운
관찰이다 — **"scen_k=65(전체 64개 HI 사용)"이 오히려 가장 나쁘고**, HI를
15~35개로 솎아내는 쪽이 분명히 낫다. `avg_scen_his`가 k값과 정확히 일치하는
것으로 보아(65만 예외 — `N_HI=64`라 65는 여전히 no-op) `scen_k` 게이트가 실제로
작동해 상위 k개 HI만 선택하고 있고, 이 스파스화 자체가 (표본 수가 적은
`full_cycle`에서) 강력한 정규화 역할을 하는 것으로 보인다 — Phase2의 모델
용량·`weight_decay`를 줄인 것과 같은 방향의 효과(과적합 방지)를 `scen_k` 축소도
추가로 제공하는 셈이다. 다만 이건 **"축을 바꾼 효과"가 아니라 "scen_k를
낮춘 효과"라 §4의 "유일한 변수는 `--seg-axis`" 원칙에서 벗어난 별도 관찰**이다
— `full_cycle` vs baseline을 엄밀히 비교하려면 scen_k=65 행(§4 원칙 그대로,
`full_cycle`이 baseline보다 못함)을 봐야 하고, k=15/25/35 결과는 "scen_k
자체가 `full_cycle`에 유효한 정규화 수단이더라"는 부가 발견으로 별도 기록한다.
k=15/25/35 사이엔 뚜렷한 단조 경향 없이 비슷한 수준(0.981~0.982)이라, 정확한
최적 k를 더 좁혀보려면 k=5/10 같은 더 낮은 값도 추가로 시도해볼 만하다.

---

# 실험 2: `random`(`rcs.py`, assign="none")로 시나리오(존 타이핑) 개념의 기여도 정량화

## 0. 요청 원문 요약

실험 2: 시나리오 개념의 이득 정량화. 그냥 20% 길이의 세그먼트 24개(충전 12개 +
방전 12개)로 사이클을 분할(분류기는 필요 없음), k=65, `docs/260810_RESULTS.md`의
baseline 모델과 비교.

## 1. 핵심 결론 먼저

`common/scenario/rcs.py`의 `RCSSegmenter`가 `assign="none"` 모드를 이미
지원하는데, 이게 정확히 "충전/방전 방향 말고는 세그먼트에 아무 타입(시나리오/
클래스)도 부여하지 않는" 모드다 — `n_scenarios=2`(`chg`/`dis`),
`n_classes=1`(`["any"]`), `classifier_default="none"` — 실험 1의 `full_cycle`과
완전히 같은 패턴이고, 하류 코드(`hi_correlation.py`/`train_scr.py`/
`train_classifier.py`/`scr_evaluator.py`/`test_scr.py`/`segment_dataset.py`)가
`ScenarioSpec`을 동적으로 읽는다는 것도 실험 1에서 이미 검증했으므로 그대로
적용된다.

**단, 저장 경로가 문제였다 — `_4_data_hi/rcs/`에 이미 다른 설정
(`n_samples=6, window=0.3, assign="position_bin"`)으로 추출된 캐시가 있고, 그걸
실제로 쓴 기존 run(`_5_data_model_scr/0717_1847_p1_rcs`,
`0717_2003_p2_mlp_rcs`)까지 있어서, 이번 실험 설정으로 재추출하면 그 자리를
덮어써 기존 run을 재현할 수 없게 됐다.** 이건 **작은 코드 수정**으로
해결했다(2026-08-11 적용, 아래 §3.2) — `RCSSegmenter`에 `axis_name` 파라미터를
추가하고 `common/scenario/__init__.py`에 같은 클래스를 `"random"`이라는 별칭으로
추가 등록해서, `--seg-axis random`으로 실행하면 `_4_data_hi/random/`이라는
완전히 별도 경로에 저장되도록 만들었다 — 기존 `rcs` 캐시는 전혀 건드리지 않는다.

## 2. 배경 — `random`(`rcs.py`, assign="none")이 왜 "시나리오 개념의 기여도"를 재는 척도가 되는가

`q_frac_ref` baseline은 사이클을 6개 시나리오(`chg_lo/mid/hi`, `dis_hi/mid/lo`)로
나누고, 각 세그먼트가 "어느 시나리오인지"를 분류기(Step 7)가 예측해 시나리오별로
분리 학습된 게이트/HI 서브셋을 적용한다(hard/soft 라우팅). 이 구조 자체가
성능에 얼마나 기여하는지 분리해서 재려면, **세그먼트 길이·개수는 비슷하게
유지하면서 시나리오 타입 구분만 없앤** 대조군이 필요하다.

`rcs.py`의 `assign="none"` 모드가 정확히 이걸 제공한다 — `window`(세그먼트 길이,
q_frac 기준)와 `n_samples`(방향당 세그먼트 개수)는 그대로 조절 가능하면서,
`latent_class`가 항상 0으로 고정되어(`rcs.py:143` `else: latent_class = 0`)
시나리오가 방향(충전/방전) 하나로만 갈린다 — `q_frac_wide`/`q_frac_ref`가
가진 "존 위치별 타입 구분+라우팅" 기능만 정확히 빠진 상태다. 즉:

| | `q_frac_ref` baseline | `random`(assign="none") — 실험 2 |
|---|---|---|
| 세그먼트 길이 | 존 경계(`n1`~`n2`) 기준, 가변 | 고정 20%(q_frac 기준) |
| 세그먼트 개수(사이클당) | 6(충전 3 + 방전 3, 실측) | 24(충전 12 + 방전 12) |
| 시나리오/존 타입 구분 | 있음(6종, 분류기로 예측) | 없음(방향 2종만) |
| 분류기(Step 7) | `mlp_probe`/`cnn` 필요 | `classifier_default="none"` — 형식적 스텝 |
| 라우팅(oracle/hard/soft) | 3가지가 서로 다른 결과 가능 | 3가지 전부 동일(실험 1의 `full_cycle`과 동일 이유) |

**baseline의 "6개" 관련 주의**: `q_frac_ref`의 `n_samples=4`는 zone당 4개의
후보 시작점을 만들지만(`q_frac_wide.py:_start_positions`), 실제로는 zone당
마지막 1개만 최종 데이터에 남는다 — `hi_correlation.py:1979-1997`이 `iter_segments()`
결과를 `row[f"raw_v_{seg}"] = ...`처럼 `seg_name`을 키로 덮어쓰기 때문에, 같은
zone의 4개 후보가 순서대로 같은 키를 3번 덮어써 마지막 것만 남는다(2026-08-11,
실제 pkl에서 사이클당 정확히 6행·6개 `scen` 값만 있음을 직접 확인해 검증).
즉 baseline도 "24개 후보 중 6개만 최종 생존"인 셈이라, 위 표의 "6"은 실제
학습에 쓰이는 최종 세그먼트 수가 맞다.

baseline과 실험 2의 R²/RMSE 격차가 곧 "존 위치를 알고 그에 맞게 라우팅하는 것"의
순수 기여도다. 세그먼트 개수(24)가 baseline(6)보다 많아 정보량 자체는 실험 2
쪽이 더 많다는 점은 유의해야 한다 — 즉 이 비교는 "정보량 손실"이 아니라
"동일하거나 더 많은 정보를 주고도 시나리오 구조화 없이 얼마나 스스로 학습할 수
있는가"를 재는 것에 가깝다.

## 3. 코드 검증 결과

### 3.1 `common/scenario/rcs.py` — 변경 불필요(이미 구현됨)

```python
class RCSSegmenter(Segmenter):
    name = "rcs"
    def __init__(self, n_samples=6, window=0.3, unit="qfrac",
                 assign="position_bin", seed=42, min_pts=10,
                 cv_v_thresh=3.60, cv_cc_frac=0.80):
        ...
    def get_spec(self) -> ScenarioSpec:
        if self.assign == "position_bin":
            ...
        else:  # "none"
            n_classes = 1
            class_names = ["any"]
            routing = [[0], [1]]
            n_scenarios = 2
            scenario_names = ["chg", "dis"]
        return ScenarioSpec(..., classifier_default="mlp_probe" if position_bin else "none", ...)
```

`assign="none"`으로 두면 `n_samples=12, window=0.2`만 지정해도 원하는 스펙
(`n_scenarios=2`, `n_classes=1`, `classifier_default="none"`)이 정확히 나온다 —
실험 1의 `full_cycle`(`n_scenarios=2, n_classes=1, classifier_default="none"`)과
스펙 형태가 완전히 동일해서, 실험 1에서 검증한 하류 코드 호환성
(§3.4~§3.6, 특히 `scr_evaluator.py`의 "level_true 단일값 placeholder" 가드,
`segment_dataset.py`의 `scen`→`direction`/`level` 동적 파생)이 코드 변경 없이
그대로 적용된다. **단, 이 모드는 방향당 세그먼트가 매 (cell, cycle)마다 독립
시드로 랜덤 배치된다는 점이 다르다** — §5 참고.

### 3.2 [구현 완료, 2026-08-11] `rcs` → `random` 별칭 — 저장 경로 분리

**원인 재확인**: `_4_data_hi/rcs/`에 이미 캐시가 있고(`scenario_spec.json`
직접 확인: `n_samples=6, window=0.3, assign="position_bin"`), 이걸 쓴 기존 run
(`_5_data_model_scr/0717_1847_p1_rcs`, `0717_2003_p2_mlp_rcs`)도 존재한다.
`rcs` 축은 `full_cycle`처럼 파라미터별 하위 폴더 태깅이 없어(`train_scr.py`의
`_axis_dir` 분기가 `q_frac_wide`/`q_frac_ref`/`q_abs`/`vqslope`만 특수 처리하고
`rcs`는 `else: _axis_dir = _axis_name` 일반 경로를 탐 — 실험 1 §3.3과 동일
패턴) 실험 2 설정으로 재추출하면 이 자리를 그대로 덮어쓴다.

처음엔 "재추출 전 폴더 백업"을 권장 대응으로 적었으나, `--seg-axis` 별칭을
새로 하나 등록하는 정도면 코드 변경도 작다고 판단해 실제로 적용했다. 단순히
`common/scenario/__init__.py`에 `REGISTRY["random"] = RCSSegmenter`만 추가하는
걸로는 부족했다 — 실제로 확인해보니 `train_classifier.py`(Step 7)의
`_axis_dir_from_spec()`(`train_classifier.py:118`)은 CLI의 `--seg-axis` 문자열이
아니라 **`spec.axis`**로 저장 경로를 다시 찾는데, `RCSSegmenter.get_spec()`이
`axis="rcs"`를 하드코딩하고 있어서 REGISTRY 별칭만 추가하면 Step 4/6/8/9는
`_4_data_hi/random/`을 보는데 **Step 7만 여전히 `_4_data_hi/rcs/`(기존
`position_bin` 캐시)를 봐서 조용히 잘못된 데이터로 분류기를 학습하는 버그**가
생겼을 것이다.

**실제 변경 내용**:
```python
# common/scenario/rcs.py — RCSSegmenter.__init__에 axis_name 파라미터 추가
def __init__(self, ..., axis_name: str = "rcs"):
    ...
    self.axis_name = axis_name

def get_spec(self) -> ScenarioSpec:
    ...
    return ScenarioSpec(axis=self.axis_name, ...)   # 기존엔 axis="rcs" 하드코딩

# common/scenario/__init__.py
REGISTRY["rcs"] = RCSSegmenter
REGISTRY["random"] = RCSSegmenter   # 같은 클래스, 별도 axis_name으로 경로만 분리
```

기본값이 `axis_name="rcs"`라 기존 `rcs` 축 사용은 전혀 영향받지 않는다(직접
재현 확인: `get_segmenter("rcs", {...})` → `spec.axis == "rcs"` 그대로).
`"random"`으로 실행할 땐 **axis-config JSON에 `"axis_name": "random"`을
반드시 포함**해야 한다 — 빠뜨리면 `RCSSegmenter`가 기본값 `axis_name="rcs"`로
조용히 되돌아가 다시 `_4_data_hi/rcs/`를 가리킨다(직접 재현해서 이 실패 케이스도
확인함). 이 실수를 방지할 파라미터 검증 로직은 넣지 않았으니 명령 작성 시
주의할 것.

### 3.3 `run_pipeline.py`/`train_scr.py` — `--axis-config` 필수(실험 1과 동일 이유)

실험 1(§3.7)에서 확인한 대로 `--axis-config`를 안 주면 model-config yaml에 남은
이전 축의 axis_config가 그대로 전달된다. `full_cycle`은 생성자가 `min_pts` 외
다른 kwarg를 안 받아 `TypeError`로 바로 드러났지만, `RCSSegmenter.__init__`은
모든 인자에 기본값이 있어 엉뚱한 이전 축의 파라미터가 섞여 들어와도 타입만
맞으면 조용히 받아들인다 — 즉 에러 없이 **의도와 다른 설정으로 조용히
실행**될 위험이 더 크다. 반드시
```
--axis-config '{\"n_samples\": 12, \"window\": 0.2, \"assign\": \"none\", \"seed\": 42, \"min_pts\": 10, \"axis_name\": \"random\"}'
```
전체를 명시한다.

**[2026-08-11 실행 중 추가로 발견한 버그] `--axis-config` JSON은 축 이름으로
감싸면 안 된다.** `hi_correlation.py`의 자체 docstring(②~⑤ 예시)과 처음 이
문서에 적었던 명령어 전부 `--axis-config '{"random": {...}}'`처럼 축 이름으로
한 번 더 감싼 형태였는데, 이러면 실제로 `TypeError:
RCSSegmenter.__init__() got an unexpected keyword argument 'random'`이 난다.
원인: `hi_correlation.py:1878`/`2859`, `train_scr.py:451`이 전부
`get_segmenter(_axis, {_axis: _axis_cfg})` 형태로 **이미 한 번 감싸서**
호출한다 — 즉 `_axis_cfg` 자체는 축 이름 키가 없는 "맨" 파라미터 dict여야
하는데, 축 이름으로 한 번 더 감싸면 이중 래핑되어 `axis_kwargs`에 축 이름
자체가 엉뚱한 키워드 인자로 들어간다. 이건 `q_frac_ref`/`q_frac_wide`
실험들이 전부 `--n1`/`--n2` 같은 전용 단축 플래그(내부적으로 이미 "맨" 형태로
`json.dumps`됨, `hi_correlation.py:2848`)만 써왔고 raw `--axis-config` JSON을
실제로 실행해본 적이 없어서 지금까지 발견되지 않은, `rcs`/`protocol`/
`vwindow`/`cluster` 축 한정의 기존 버그였다(코드 자체는 이번에 고치지 않고
`hi_correlation.py`의 docstring만 올바른 형태로 수정함, §5 명령어도 정정함).
**최종적으로 필요한 형태는 "축 이름 래핑 없음" + "PowerShell `\"` 이스케이프"
둘 다 적용된 형태다** — 아래 §5 명령어 참고.

### 3.4 세그먼트 배치가 결정론적이지 않다는 점(설계상 선택)

`_sample_segments`가 `rng.uniform(0.0, max_start, size=n_samples)`로 시작점을
뽑는다 — "24개로 균등 타일링"이 아니라 "고정 시드로 재현 가능한 12개 랜덤 위치
샘플링"이다. `rng = np.random.default_rng([seed, hash(cell_id)%2**31, cycle])`라
(cell, cycle) 조합마다 항상 같은 위치가 나오므로 **추출은 재현 가능**하지만,
"20% 폭 세그먼트가 사이클을 고르게 덮는다"는 보장은 없다(같은 영역에 여러 세그먼트가
겹칠 수 있음). 완전히 균등한 타일링(예: `np.linspace(0, 1-window, n_samples)`로
결정론적 등간격 배치)을 원하면 `_sample_segments`의 `rng.uniform(...)` 한 줄을
바꾸는 작은 코드 수정이 필요하다 — 이건 "시나리오 개념 유무"와 무관한 별도
설계 선택이라, 우선 기존 랜덤 모드로 실행해보고 결과가 이상하면(예: 특정 구간
과다 샘플링으로 편향) 그때 등간격으로 바꾸는 쪽을 권장한다.

## 4. 실험 설계 (260810 baseline과의 비교)

### 고정값(260810 baseline과 동일하게 유지)

| 항목 | 값 | 근거 |
|---|---|---|
| HI(`stat_q_abs`/`stat_energy_seg`) | 미포함(`SOH_EXCLUDE_STAT_LEAK=1`) | baseline과 동일 조건 |
| shape filter | 적용(기본) | baseline과 동일 |
| `min_pts` | 10 | baseline과 동일 |
| `scen_k`(`--scen-k`) | **65** | 요청대로 고정. `random`도 `model.scen_gates[s]`가 `n_scenarios=2`개 존재하므로(실험 1 §4와 동일 원리) 여전히 유효한 파라미터 — `N_HI=64`(HI 미포함)라 65는 여기서도 no-op(전 HI 사용) |
| `regression_model` | mlp | baseline과 동일 |
| raw 융합 | 없음 | baseline과 동일 |
| 데이터셋 | MIT+HUST 풀링, oracle/hard/soft 표준 평가 | baseline과 동일(단 §2 표대로 셋 다 값이 같게 나올 것) |

### 유일한 변수(의도된 것)

`--seg-axis`: `q_frac_ref`(baseline, 시나리오 타이핑 있음) vs `random`+`assign=none`
(시나리오 타이핑 없음, 20% 고정폭 세그먼트 24개).

### `model-config`는 실험 1의 `exp_full_cycle.yaml`을 재사용하지 않는다

실험 1 도중 `mlp_hidden_dims`를 `[128,64]`로, `weight_decay`를 `1e-3`으로,
`early_stop_patience`를 `100`으로, `lr`을 `1e-4`로 낮췄는데, 이건 **full_cycle의
작은 데이터량(baseline의 1/3)에 맞춘 조정**이었다(§Phase1/Phase2 분석 대화 참고).
`random`(assign=none)은 세그먼트가 사이클당 24개로 baseline(6개)보다 오히려
**4배 많다** — 데이터량 문제가 없으므로 baseline의 원래 하이퍼파라미터
(`exp_qfw_mlp_basefix.yaml`: `mlp_hidden_dims=[512,256,128,64]`, `lr=5e-4`,
`weight_decay=1e-4`, `patience=30`)를 그대로 쓰는 게 맞다 — 축소된 모델을 여기에
쓰면 오히려 "시나리오 구조 없음"과 "모델 용량 부족"이 뒤섞여 실험이 오염된다.
새 `exp_rcs_none.yaml`을 baseline(`exp_qfw_mlp_basefix.yaml`) 그대로 복사해
`scenario.axis`/`axis_config`만 바꿔서 만든다.

## 5. 실행 명령

```powershell
# Step 4 — HI 추출(assign=none, window=0.2, n_samples=12) → _4_data_hi/random/ (rcs/와 분리 저장)
python 4_hi_analysis/hi_correlation.py --seg-axis random --axis-config '{\"n_samples\": 12, \"window\": 0.2, \"assign\": \"none\", \"seed\": 42, \"min_pts\": 10, \"axis_name\": \"random\"}' --force --workers 40

# Step 6~9 — Phase1+분류기(형식적)+Phase2+평가
$env:SOH_EXCLUDE_STAT_LEAK="1"
python run_pipeline.py 6 --model-config 5_model/config/exp_random_none.yaml --seg-axis random --axis-config '{\"n_samples\": 12, \"window\": 0.2, \"assign\": \"none\", \"seed\": 42, \"min_pts\": 10, \"axis_name\": \"random\"}' --scen-k 65 --workers 40
```

`--seg-axis random`으로 바뀌었다(§3.2 — `rcs` 캐시와 완전히 분리된 경로를 쓰기
위해 별칭을 새로 등록함). axis-config JSON의 `"axis_name": "random"`을 빠뜨리면
안 된다 — 빠뜨리면 저장 경로가 조용히 다시 `rcs/`로 돌아간다(§3.2에서 실제
재현 확인). `exp_random_none.yaml`은 아직 생성 전이다(이번 턴은 설계 문서화까지만
— 실험 1 때와 동일하게 실행 전 확인 필요). 만들 때 `exp_qfw_mlp_basefix.yaml`을
그대로 복사해 `scenario.axis: random`, `scenario.axis_config: {n_samples: 12,
window: 0.2, assign: "none", seed: 42, min_pts: 10, axis_name: "random"}`만
바꾸면 된다.

## 6. 비교 판정 기준

| 비교 | 의미 |
|---|---|
| `random`(assign=none) R²/RMSE/MAPE vs `q_frac_ref` baseline(R²=0.9524, RMSE=1.520%) | 격차가 "시나리오 타이핑+라우팅" 구조의 순수 기여도. 세그먼트 개수가 더 많은데도 baseline보다 못하면 시나리오 구조의 가치가 명확히 입증됨 |
| oracle/hard/soft 3가지가 실제로 전부 동일하게 나오는지 | §2 표에서 예측한 대로 라우팅이 무의미해지는지 확인(실험 1과 같은 검증) |
| 세그먼트 개수 4배 증가에도 불구하고 얼마나 뒤처지는지 | "단순히 더 많은 조각을 주는 것"과 "정보를 시나리오로 구조화해서 주는 것" 중 어느 쪽이 더 중요한지에 대한 직접 증거 |

## 7. 남은 리스크 / 확인 필요 사항

- axis-config에 `"axis_name": "random"`을 빠뜨리면 조용히 `rcs/` 경로로
  되돌아간다(§3.2) — 명령 실행 전 재확인할 것. 이 실수를 막는 검증 로직은
  코드에 넣지 않았다.
- 랜덤 윈도우 배치(§3.4)가 특정 SOC 구간을 과다/과소 샘플링할 가능성 — 결과가
  이상하면 등간격 배치로 바꾸는 작은 코드 수정을 고려.
- `window=0.2`(20%)·`n_samples=12`가 사이클을 몇 번이나 덮는지(평균 커버리지
  배수) 미리 계산해보지 않았다 — 24개 × 20% = 480%/2(양방향 평균 240%)이므로
  각 지점이 평균 2.4회 정도 중복 샘플링되는 셈인데, 이 중복이 정보량을 실제로
  늘리는지 아니면 중복된 정보만 반복하는지는 결과 해석 시 함께 봐야 한다.

---

## 8. 실행 결과 — `scen_k=25` 통제 비교 (2026-08-11)

`_5_data_model_scr/0811_1826_p2_mlp_rand_noleak`(`--seg-axis random`, `scen_k=25`,
나머지는 baseline 하이퍼파라미터 그대로 — `gates_from=0811_1818_p1_rand_noleak`,
`lr=5e-4`, `weight_decay=1e-4`, `patience=30`, `mlp_hidden_dims=[512,256,128,64]`)
결과를, `docs/260810_RESULTS.md`의 실험3 k 스윕 중 **같은 `scen_k=25`** 지점
(`0809_1932_p2_mlp_q_fr_35%_20%_noleak`, `q_frac_ref` 축)과 비교한다 — 두 run
모두 `scen_k=25`로 맞춰뒀으므로 §4의 "유일한 변수는 `--seg-axis`" OFAT 원칙이
정확히 지켜진 비교다(실험 1에서 scen_k가 안 맞아 겪었던 문제가 여기선 없음).

| | `q_frac_ref` baseline(k=25) | `random`(assign=none, k=25) | 격차 |
|---|---|---|---|
| run 디렉터리 | `0809_1932_p2_mlp_q_fr_35%_20%_noleak` | `0811_1826_p2_mlp_rand_noleak` | |
| R²(oracle) | 0.94927 | 0.94898 | **-0.00029** |
| RMSE | 1.569%p | 1.573%p | +0.004%p |
| MAE | 1.151%p | 1.164%p | +0.012%p |
| MAPE | 1.280% | 1.289% | +0.009%p |
| oracle=hard=soft | 아니오(6종 시나리오, 라우팅 방식별 차이 존재) | **예**(§2에서 예측한 대로) | — |
| `avg_computed_his` | 25.67 | 25.00 | random이 더 적음 |
| `avg_cost` | 42.33 | 34.75 | random이 더 낮음(효율 ↑) |
| `cost_reduction_pct` | 60.06% | 67.22% | random이 더 높음 |

### 해석

**격차가 사실상 없다.** R² 차이는 0.0003(0.03%p 수준), RMSE 차이는 0.004%p로
측정오차 범위에 가깝다 — `scen_k=25`로 HI를 충분히 솎아낸 조건에서는, 시나리오
타입 구분·라우팅 구조(baseline)가 있든 없든(random) 정확도가 거의 동일하다.
이건 §1에서 정의한 "baseline과 실험 2의 R²/RMSE 격차 = 존 타이핑+라우팅 구조의
순수 기여도"라는 판정 기준을 그대로 적용하면, **이 조건에서 시나리오 구조의
기여도가 거의 0에 수렴한다**는 뜻이다 — 실험 1의 `full_cycle`(scen_k=65 기준
baseline 대비 R² -0.038, §8)과 비교하면 훨씬 작은 격차다.

두 실험을 나란히 보면 흥미로운 대비가 생긴다:
- 실험 1(`full_cycle`, 사이클당 2세그먼트, baseline의 1/3): baseline에 뚜렷이
  못 미침(R² -0.038) → **관측 범위(정보량) 자체가 부족하면 구조가 있어도
  못 메운다.**
- 실험 2(`random`, 사이클당 24세그먼트, baseline의 4배): baseline과 사실상
  동률(R² -0.0003) → **세그먼트 개수가 충분하면 시나리오 타입 구분·라우팅
  구조가 거의 필요 없어 보인다.**

즉 지금까지의 두 실행 결과를 종합하면, 이 프레임워크의 성능을 좌우하는 주된
요인은 "시나리오로 구조화했는가"보다는 **"충분히 많고 국소화된 세그먼트를
확보했는가"** 쪽에 더 가깝다는 잠정 결론이 선다 — 다만 각 실험이 단일 run
(seed 1개)이라 재현성 확인 없이 확정하기는 이르고, 시나리오 구조 자체의 가치는
scen_k가 작을 때(예: k=5~10, HI를 극단적으로 제한해 "어떤 HI가 어느 시나리오에
중요한지"가 더 결정적으로 작용할 조건)는 다시 커질 수 있다 — 이 조건에서
baseline과 random을 다시 비교해보는 게 자연스러운 다음 단계다.

`avg_cost`/`cost_reduction_pct`에서 random이 소폭 더 효율적인 건(34.75 vs
42.33) baseline이 `scen_k=25`를 요청해도 실제로는 시나리오별로 조금씩 다른
개수가 선택돼(`avg_computed_his=25.67`, 정확히 25가 아님 — 시나리오마다 게이트가
약간 다르게 수렴) 평균이 25를 넘는 반면, random은 `n_scenarios=2`뿐이라
두 게이트 다 정확히 25개로 수렴했기 때문으로 보인다 — 정확도와는 별개로,
시나리오 수가 적을수록(2 vs 6) 게이트 선택이 더 균일해지는 부수 효과로 해석된다.

---

## 9. 실행 결과 — `scen_k` 전 구간 비교(k=5/15/25/65) (2026-08-11)

§8의 "scen_k가 작을 때는 시나리오 구조의 가치가 다시 커질 수 있다"는 다음 단계
제안을 실제로 실행했다. k=5는 이번에 새로 돌렸고(`random`/`q_frac_ref` 각각),
k=15/25는 기존 run(§8, `docs/260810_RESULTS.md`)을 재사용한다.

| k | axis | run 디렉터리 | R²(oracle) | RMSE | MAE | MAPE | oracle=hard=soft |
|---|---|---|---|---|---|---|---|
| 5 | `q_frac_ref` | `0811_2005_p2_mlp_q_fr_35%_20%_noleak` | 0.9339 | 1.791%p | 1.214%p | 1.329% | 거의(Δ≈0.00002, 분류 정확도 99.97%) |
| 5 | `random` | `0811_2033_p2_mlp_rand_noleak` | **0.9349** | 1.777%p | 1.269%p | 1.398% | 예 |
| 15 | `q_frac_ref` | `0809_1821_p2_mlp_q_fr_35%_20%_noleak` | **0.9475** | 1.596%p | 1.208%p | 1.329% | 거의 |
| 15 | `random` | `0811_1851_p2_mlp_rand_noleak` | 0.9301 | 1.841%p | 1.439%p | 1.559% | 예 |
| 25 | `q_frac_ref` | `0809_1932_p2_mlp_q_fr_35%_20%_noleak` | 0.9493 | 1.569%p | 1.151%p | 1.280% | 거의 |
| 25 | `random` | `0811_1826_p2_mlp_rand_noleak` | 0.9490 | 1.573%p | 1.164%p | 1.289% | 예 |
| 65(baseline) | `q_frac_ref` | `0809_1118_p2_mlp_q_fr_35%_20%_noleak` | 0.9524 | 1.520%p | 1.133%p | 1.251% | 거의 |
| 65 | `random` | (미실행) | — | — | — | — | — |

(k=5 두 run 다 `gates_from`만 새로 학습되고 나머지는 baseline 하이퍼파라미터 —
`lr=5e-4`/`weight_decay=1e-4`/`patience=30`/`mlp_hidden_dims=[512,256,128,64]`
— 그대로. `q_frac_ref` k=5는 분류 정확도가 99.97%라 oracle/hard/soft 차이가
R² 기준 0.00002 수준으로 사실상 무시할 만하다.)

### 해석 — 예상과 다른, 비단조(non-monotonic) 패턴

§8에서 "k가 작을수록(HI가 귀할수록) 시나리오 구조의 가치가 커질 것"이라고
예상했는데, 실제로는 그렇지 않다:

- **k=5에서는 오히려 `random`이 `q_frac_ref`를 근소하게 이긴다**(0.9349 vs
  0.9339, ΔR²=+0.0010) — 구조가 없는 쪽이 HI가 가장 귀한 조건에서 더 나은,
  예상과 반대되는 결과다.
- **격차가 가장 크게 벌어지는 지점은 k=15다**(baseline이 +0.0174 우위) — k=5도
  k=25도 아닌 중간 지점에서 baseline의 우위가 가장 크다.
- k=25에서는 §8에서 이미 확인한 대로 거의 동률(+0.0003)이고, k=65는 `random`을
  아직 안 돌려 비교할 수 없다.

즉 "k가 작아질수록 구조의 가치가 단조롭게 커진다"는 가설은 이 데이터로는
기각된다 — k=5/15/25에 걸쳐 격차가 -0.0010 → +0.0174 → +0.0003로 뚜렷한
단조 경향 없이 오르내린다. 두 축 다 baseline 세팅을 그대로 쓴 single-seed
run이라, 이 정도 크기(±0.01~0.02 R²)의 등락은 학습 노이즈(초기화, `min_pts`
경계에서의 표본 변동 등)로 설명 가능한 범위일 수 있다 — 시나리오 구조의
"진짜" 기여도를 k별로 확정하려면 최소 seed 2~3개씩 반복해 신뢰구간을 봐야
한다는 게 지금 데이터가 주는 가장 정직한 결론이다. 현재 단일-seed 결과만
놓고 "k=15에서 구조가 중요하다"거나 "k=5에서 구조가 불필요하다"고 확정하는
건 과대해석이다.

k=5 `q_frac_ref`의 `avg_computed_his=10`(=probe와 동일, `avg_scen_his=5`인데
`avg_computed_his`가 10인 건 시나리오별로 겹치는 HI가 있어 union 크기가
scen_k보다 커지기 때문)은 §8의 "시나리오 수가 적을수록 게이트가 더 균일하게
수렴" 패턴과 일관되게, `random`도 k=5에서 정확히 `avg_scen_his=5,
avg_computed_his=10`으로 baseline과 동일한 union 크기를 보인다 — 이 지점만은
효율성 지표 자체는 두 축이 우연히 같다.

### k=15 `random`이 유독 나쁜 이유 — Phase1 게이트 랭킹 직접 대조(2026-08-12)

`random`의 k=5/15/25는 각각 독립적으로 새로 학습한 Phase1이다(`scen_k`가
`lambda_l0` 자동계산에도 들어가 Phase1 자체가 매번 다시 학습됨, §Phase1/Phase2
분석 대화 참고). 세 Phase1의 `gates/regression_HIs.json`(`seg_0_names`)을
직접 열어 비교했다:

- **상위 5개 HI는 세 run 모두 사실상 동일**하다 — `diff_dqdv_area_chg`,
  `diff_dqdv_peak_h_chg`, `diff_dqdv_peak_w_chg`, `lfp_v_ent_plateau_chg`,
  `stat_i_mean_chg`(순서만 미세하게 다름, k=5/25 run은 완전히 같은 순서,
  k=15 run만 2·4위가 바뀌어 있음). 즉 게이트가 "이 5개는 무조건 중요하다"고
  세 번 다 일관되게 수렴했다 — k=5가 baseline과 비슷한 성능을 낸 건 이
  안정적인 핵심 5개만으로 충분했기 때문으로 보인다.
- **6~15위는 run마다 흔들린다** — k=15 run의 top15와 k=25 run의 top15를
  비교하면 15개 중 **10개만 겹친다**(2/3). k=15 run에만 있는 5개
  (`lfp_v_gradient_exit_chg`, `diff_dvdq_skew_chg`, `lfp_v_flatness_chg`,
  `lfp_plateau_v_slope_chg`, `morph_vq_dtw_chg`)는 k=25 run의 top15엔 없다.

즉 핵심 신호(top5)는 게이트가 아주 안정적으로 재현하는데, 그다음 순위(6~15위)는
Phase1을 독립적으로 다시 학습할 때마다 상당히 다른 HI가 걸린다. k=15 run이
유독 이 "흔들리는 구간"에서 덜 유용한 조합을 뽑았을 가능성이 커 보인다 — 위
"해석"에서 이미 추측했던 "single-seed 노이즈"라는 설명을, 실제 게이트 랭킹
불안정성이라는 구체적 근거로 뒷받침한다. 다만 이게 k=15 run의 R²를 실제로
깎아먹었는지(인과관계) 확인하려면 그 5개 HI를 빼고 재학습해보는 ablation이
필요한데, 이번 턴에서는 하지 않았다.

---

## 10. 실행 결과 — MLP_S 통일 + 3-seed 재현성 검증 (k=5) (2026-08-12)

§9까지의 모든 비교가 single-seed였다는 한계(§9 해석 마지막 문단)를 보완하기 위해,
`docs/260812_RESULTS.md`에서 설계한 대로 k=5에서 `random`/`q_frac_ref` 각각
seed(=`split_seed`) 42/0/123으로 3회씩 반복 실행했다. §9와 달리 이번엔 **두 축
다 MLP_S**(`mlp_hidden_dims=[128,64]`, `lr=2e-4`, `weight_decay=1e-3`,
`patience=100`, `exp_random_S.yaml`/`exp_qfw_mlp_S.yaml`)로 통일해서, MLP 크기
불일치 문제(§8/§9에서 반복 지적된) 없이 순수하게 `--seg-axis`만 다른 대칭
비교다. `--skip-classifier`를 썼으므로 아래는 전부 oracle 지표다.

| axis | seed | run 디렉터리 | R²(oracle) | RMSE | MAE | MAPE |
|---|---|---|---|---|---|---|
| `random` | 42 | `0812_1506_p2_mlp_rand_noleak` | 0.9269 | 1.884%p | 1.399%p | 1.544% |
| `random` | 0 | `0812_1525_p2_mlp_rand_noleak` | 0.9299 | 1.774%p | 1.308%p | 1.434% |
| `random` | 123 | `0812_1549_p2_mlp_rand_noleak` | 0.9354 | 1.789%p | 1.341%p | 1.475% |
| **`random` 평균±표준편차** | | | **0.9307 ± 0.0043** | **1.815 ± 0.060%p** | **1.349 ± 0.046%p** | **1.484 ± 0.056%** |
| `q_frac_ref` | 42 | `0812_1642_p2_mlp_q_fr_35%_20%_noleak` | 0.9433 | 1.658%p | 1.187%p | 1.306% |
| `q_frac_ref` | 0 | `0812_1745_p2_mlp_q_fr_35%_20%_noleak` | 0.9454 | 1.565%p | 1.134%p | 1.239% |
| `q_frac_ref` | 123 | `0812_1858_p2_mlp_q_fr_35%_20%_noleak` | 0.9471 | 1.619%p | 1.180%p | 1.302% |
| **`q_frac_ref` 평균±표준편차** | | | **0.9453 ± 0.0019** | **1.614 ± 0.046%p** | **1.167 ± 0.029%p** | **1.282 ± 0.037%** |

(표준편차는 표본표준편차, n=3. 세 run 전부 `gates_from` 없이 매번 Phase1부터
새로 학습, `scen_k=5`, `--skip-classifier`로 oracle만 평가.)

### 해석 — 이번엔 노이즈가 아니라 실제 격차로 보인다

**두 축 평균의 격차(ΔR²=+0.0146, q_frac_ref 우위)가 각 축 자체의 seed-to-seed
표준편차(random 0.0043, q_frac_ref 0.0019)보다 3배 이상 크다.** §9에서 겪었던
"single-seed라 노이즈인지 실제 패턴인지 모른다"는 문제가 여기서는 해소된다 —
`q_frac_ref`(시나리오 구조 있음)의 3개 seed(0.9433~0.9471) 구간과
`random`(구조 없음)의 3개 seed(0.9269~0.9354) 구간이 **전혀 겹치지 않는다.**
이 조건(k=5, MLP_S)에서는 시나리오 구조가 재현 가능한 실질적 이득을 준다고
볼 수 있다.

**주의 — §9의 k=5(MLP-L, single-seed) 결과와 방향이 다르다.** §9에서는 원래
큰 MLP로 k=5를 한 번씩만 돌렸을 때 `random`(0.9349)이 `q_frac_ref`(0.9339)를
근소하게 이겼었다. 지금 이 3-seed 결과(MLP_S)는 정반대(`q_frac_ref`가 뚜렷한
우위)다. 두 결과가 서로 모순되는 게 아니라 **다른 조건**이다 — §9는 큰 MLP·
seed 1개, §10은 작은 MLP(MLP_S)·seed 3개 평균. 이걸로 "MLP_S가 정답이다"
또는 "§9가 틀렸다"라고 결론 내릴 수는 없고, **"single-seed 비교는 결론을
뒤집을 수 있을 만큼 신뢰구간이 넓다"**는 게 오히려 이번 실험이 확인해주는
핵심 교훈이다 — §9의 k=15/25 비교(같은 문제를 겪었을 가능성이 있는 지점들)도
3-seed로 다시 확인해볼 가치가 있다.

**부수 발견**: `random`의 표준편차(0.0043)가 `q_frac_ref`(0.0019)보다 **2배
이상 크다** — 시나리오 구조가 없는 축이 seed에 더 민감하다는 신호로 보인다.
`avg_computed_his`도 `q_frac_ref`가 11.8~12.5, `random`이 10.0~11.0으로
`q_frac_ref` 쪽이 약간 더 많은 HI를 쓰지만(6개 시나리오 union이라 겹침이
더 클 수 있음), 이 차이가 성능 격차의 주된 원인인지는 확인하지 않았다.

---

## 11. 실행 결과 — MLP_S 통일 + 3-seed 재현성 검증 (k=15) (2026-08-13)

§10과 완전히 동일한 설계(`exp_random_S.yaml`/`exp_qfw_mlp_S.yaml`, MLP_S 통일,
`--skip-classifier`, seed=`split_seed`=42/0/123)를 `scen_k=15`에서 반복했다.

| axis | seed | run 디렉터리 | R²(oracle) | RMSE | MAE | MAPE |
|---|---|---|---|---|---|---|
| `random` | 42 | `0812_2010_p2_mlp_rand_noleak` | 0.9321 | 1.815%p | 1.347%p | 1.486% |
| `random` | 0 | `0812_2028_p2_mlp_rand_noleak` | 0.9303 | 1.768%p | 1.299%p | 1.427% |
| `random` | 123 | `0812_2048_p2_mlp_rand_noleak` | 0.9395 | 1.732%p | 1.291%p | 1.415% |
| **`random` 평균±표준편차** | | | **0.9340 ± 0.0049** | **1.772 ± 0.042%p** | **1.312 ± 0.030%p** | **1.443 ± 0.037%** |
| `q_frac_ref` | 42 | `0812_2137_p2_mlp_q_fr_35%_20%_noleak` | 0.9512 | 1.538%p | 1.104%p | 1.219% |
| `q_frac_ref` | 0 | `0812_2244_p2_mlp_q_fr_35%_20%_noleak` | 0.9493 | 1.508%p | 1.094%p | 1.196% |
| `q_frac_ref` | 123 | `0812_2348_p2_mlp_q_fr_35%_20%_noleak` | 0.9555 | 1.486%p | 1.066%p | 1.180% |
| **`q_frac_ref` 평균±표준편차** | | | **0.9520 ± 0.0031** | **1.511 ± 0.026%p** | **1.088 ± 0.020%p** | **1.198 ± 0.020%** |

(표준편차는 표본표준편차, n=3. `avg_computed_his`는 `random` 16.0~16.5,
`q_frac_ref` 19.8~20.5 — 둘 다 §10의 k=5보다 커졌고 격차도 비슷하게 유지된다.)

### 해석 — k=5와 같은 방향, 더 큰 격차

k=15에서도 `q_frac_ref`가 `random`을 뚜렷하게 앞선다(ΔR²=+0.0180, k=5의
+0.0146보다 더 벌어짐). 이번에도 두 축의 3-seed 구간이 겹치지 않고
(`q_frac_ref` 0.9493~0.9555 vs `random` 0.9303~0.9395), 격차(0.0180)가 각 축
표준편차(0.0049/0.0031)의 4~6배라 §10과 마찬가지로 노이즈로 보기 어렵다.

k=5(§10, ΔR²=+0.0146)와 k=15(§11, ΔR²=+0.0180)를 나란히 보면, **k가 커질수록
(HI를 더 많이 쓸 수 있을수록) `q_frac_ref`의 우위가 오히려 더 벌어지는 경향**이
보인다 — 이전 single-seed 관찰(§9)에서 나왔던 "k가 작을수록 구조의 가치가
커질 것"이라는 가설과는 다시 반대 방향이다. 다만 2개 k 지점만으로 추세를
확정하기엔 이르다 — k=25/65도 같은 3-seed 방식으로 재확인하면 더 명확한 그림이
나올 것이다. 두 축 모두 `random`의 표준편차가 `q_frac_ref`보다 크다는 §10의
패턴도 k=15에서 그대로 재현된다(0.0049 vs 0.0031).
