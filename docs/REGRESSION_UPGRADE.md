# REGRESSION_UPGRADE — 회귀 헤드(+분류기) 입력에 원본 곡선 데이터 반영

**작성일**: 2026-07-30
**배경**: `docs/MODEL_SPECS.md` §5(회귀 모델 CNN 적용 검토)에서 "2단계"로 미뤄뒀던 작업을
지금 착수한다. 그 문서는 "HUST chg_hi/mid 평탄 예측 진단 결과에 따라 착수 여부 결정"이라는
조건부 보류였고, §9.5(남은 작업)에 미구현으로 남아있다 — 이 문서는 그 이후 관점에서
다시 설계하고, 이번엔 보류 없이 실행까지 간다.

---

## 0. 사전 정정 — 사용자 제안의 두 가지 전제 확인

작업을 시작하기 전에 요청 문구의 두 전제를 코드로 직접 검증했다.

1. **"V, I, t의 원본 곡선"** → 현재 파이프라인이 실제로 저장하는 원본 채널은
   **V, |I| 두 개뿐**이다(`RAW_CH=2`, `5_model/utils/hi_schema.py:50`). t(시간)는 별도
   채널로 저장되지 않는다 — 세그먼트 내부 상대 위치는 시간이 아니라 **q_frac(그
   세그먼트 자신의 상대 누적전하, [0,1])** 그리드로 인코딩된다(`_resample_segment`,
   `4_hi_analysis/hi_correlation.py:336`). t 채널을 추가하려면 별도 작업이 필요하다 —
   이 문서의 두 방안 모두 "V, |I|" 2채널을 전제로 비교하고, t 추가는 §7 확장 여지로만
   다룬다.

2. **"패딩 및 정규화 수행하여 크기 고정 필요"** → **이미 돼 있다.** 세그먼트 길이가
   셀·사이클마다 달라도(실측 33~2801 포인트 확인됨, `MODEL_SPECS.md §9.3`), 제로패딩이
   아니라 **q_frac [0,1] 그리드에 선형보간**해 항상 정확히 `RAW_N=48` 포인트로 맞춘다
   (`_resample_segment`). 이 방식은 제로패딩보다 유리하다 — 패딩 마스크가 필요 없고,
   짧은 세그먼트도 정보 밀도가 희석되지 않는다. **또한 이 정규화는 그 세그먼트 자신의
   상대 위치만 쓰므로 q_frac_wide의 `q_tot` 라벨 누수 문제(SCENARIO_STRATEGY.md §11)와
   무관하게 라벨-독립적이다** — 절대 크기(얼마나 길었는지)를 버리고 형태만 남기기
   때문. 이 파이프라인은 `common/scenario/q_abs.py`를 포함해 이번 세션에서 다룬 모든
   축에 이미 축-불문(axis-agnostic)으로 연결돼 있다(세그먼트 추출 루프 안에서 호출됨).

즉 두 방안 모두 **데이터 준비 단계는 이미 끝나 있다** — 이 문서의 실질적인 논점은
"이미 준비된 `x_raw (B,2,48)`를 회귀 헤드에서 어떻게 소비할 것인가"로 좁혀진다.

---

## 1. 현재 구조 정밀 검증

### 1.1 회귀 헤드 — 현재 원본 데이터를 전혀 쓰지 않는다

`5_model/models/scr_model.py`의 `SCRModel.forward`:

```python
feat = torch.cat([probe_x, scen_x, direction.unsqueeze(1), batch["cap_init"].unsqueeze(1)], dim=1)
# (B, 2*N_HI + 2) = (B, 130)
cap_pred = self.cap_head(feat)
```

`probe_x`/`scen_x`는 HardConcreteGate로 게이팅된 HI 벡터(N_HI=64)다 — 즉 회귀 헤드
입력은 **HI만** 쓰고, `batch["x_raw"]`는 `SegmentDataset.__getitem__`이 만들어는 두지만
`SCRModel.forward`가 아예 읽지 않는다. `cap_heads.py`의 5개 헤드(MLP/Transformer/
iTransformer/ResNetTab/FTTransformer) 전부 `_HEAD_IN = N_HI+N_HI+1+1 = 130`을
하드코딩하고, 그 130차원을 각자 내부에서 슬라이싱한다(`x[:, :N_HI]`, `x[:, N_HI:2*N_HI]`
등). **5개 헤드 전부가 입력 레이아웃을 하드코딩**하고 있다는 점이 두 방안의 구현
비용에 공통으로 걸린다.

### 1.2 분류기 — 사용자 설명대로 CNN+HI 융합이 이미 구현·활성화돼 있다 (검증 완료)

`common/scenario/scenario_classifier.py`의 `CNNProbeClassifier`:

```python
cnn_emb = self.cnn(x_raw)                                  # RawCNN(B,2,48) → (B,64)
feat = torch.cat([cnn_emb, probe_x, direction], dim=1)      # (B, 64+64+1=129)
logits = self.head(feat)                                    # MLP → n_classes
```

`probe_x`는 `HardConcreteGate`로 선택된 **N_HI=64개 중 활성 m개만 0이 아닌** 벡터다
(charge_probe_m=5, discharge_probe_m=5가 현재 `scr.yaml` 기본값) — 사용자가 물은
"HI 64개 중 m개를 입력으로 쓰는지"는 **정확히 맞다**, 검증됨. 그리고 이게 지금
**dormant 코드가 아니라 실제 프로덕션 기본값**이다:

```yaml
# 5_model/config/scr.yaml
classifier:
  type: cnn   # ← 현재 활성 기본값. mlp가 아님.
```

이번 세션 전체에서 언급된 프로덕션 clf_acc(예: q_abs Run A 98.36%, 260729 결과의
93.21% 등)는 전부 **이 CNN 분류기로 나온 수치**다 — MLP 분류기 대비 순수 개선폭을
분리 측정한 A/B 비교는 이 프로젝트 문서 어디에도 없다(직접 grep 확인) — 즉 "CNN 융합이
분류에 얼마나 기여했는가"조차 아직 정량화된 적이 없다. 이는 §3(예상 성능 향상)에서
정직하게 반영해야 할 공백이다.

### 1.3 이미 있는(그러나 다른 목적의) raw-only 베이스라인 — 혼동 주의

`5_model/models/raw_mlp_model.py`(`regression_model: raw_mlp`)가 이미 존재한다.
`MODEL_DIRECTION.md`의 **B-4(원본 곡선 직접 입력 CNN/Transformer)** 베이스라인 구현체다.
**이건 이 작업과 다른 것이다** — `RawMLPModel`은 HI를 완전히 버리고 raw만 쓰는
**대체(ablation)** 모델(HI 추출 자체의 기여도를 측정하는 것이 목적)이고, 이 문서의
두 방안은 HI를 유지하면서 raw를 **더하는(fusion)** 모델이다. 이름이 비슷해 헷갈릴
여지가 크므로, 구현 시 이 둘을 같은 실험 표에 나란히 두고 "HI-only → HI+raw fusion →
raw-only" 스펙트럼으로 보고하면 오히려 논문 스토리가 강해진다(§5 참조).

### 1.4 UQ(Laplace)는 입력 차원에 무관 — 두 방안 모두 리스크 없음

`5_model/utils/uncertainty.py`는 `cap_head`의 **마지막 `nn.Linear`**만 찾아 last-layer
Laplace를 적용한다(`_get_last_linear`) — 입력 차원(130이든 194든 226이든)에 의존하지
않는다. 두 방안 모두 UQ 파이프라인 수정 불필요.

---

## 2. 두 방안의 정확한 아키텍처

### 방안 1 — 원본 곡선 직접 결합(flatten concat)

```
feat = [probe_x(64) ‖ scen_x(64) ‖ direction(1) ‖ cap_init(1) ‖ raw_v(48) ‖ raw_i(48)]
     = (B, 226)
cap_pred = cap_head(feat)
```

`RawMLPModel`이 이미 쓰는 `x_raw.reshape(B,-1)` flatten 패턴을 그대로 재사용하되,
HI 항을 제거하지 않고 **추가**한다는 점만 다르다.

### 방안 2 — CNN 임베딩 결합 (MODEL_SPECS.md §5.2 재사용)

```
cnn_emb = RawCNN(x_raw)                                    # (B, 2, 48) → (B, 64)
feat = [probe_x(64) ‖ scen_x(64) ‖ cnn_emb(64) ‖ direction(1) ‖ cap_init(1)]
     = (B, 194)
cap_pred = cap_head(feat)
```

`_HEAD_IN_WITH_CNN = N_HI+N_HI+64+1+1 = 194`, `MODEL_SPECS.md`가 이미 pseudo-code까지
써둔 것과 정확히 같다. **단, 원 설계는 "분류기와 회귀 헤드가 RawCNN 인스턴스를
공유"를 전제했는데, 실제로 구현된 파이프라인은 그 전제와 어긋난다**: `train_scr.py`
(Phase 1/2, 회귀)와 `train_classifier.py`(분류기)가 **완전히 분리된 스크립트 +
분리된 시점**(Phase1→Phase2→분류기 순차)에서 각자 독립적으로 모델을 만든다
(`train_classifier.py:275`가 `CNNProbeClassifier`를 자체 생성). 가중치 공유는
이 시점에 이미 학습이 끝난 분류기 RawCNN을 불러와 Phase 2 cap_head 학습 시 얼리거나
동시 학습 흐름을 다시 짜야 하는 **더 큰 구조 변경**이다. 그러므로 이 문서는
현실적인 범위로 **회귀 헤드 전용 RawCNN 인스턴스를 새로 하나 더 두는 것**을
기본안으로 하고, 가중치 공유는 §7의 후속 과제로 분리한다(파라미터 42K 두 벌은
비용상 무시할 만함).

---

## 3. 비교

### 3.1 예상 성능 향상 수치

**정직한 전제**: 이 프로젝트에는 "raw 융합이 실제로 얼마나 이득인지"를 보여주는
직접 측정치가 하나도 없다(분류기 CNN조차 MLP 대비 A/B가 없음, §1.2). 아래는 그
공백을 메우는 근거 있는 **추정**이지 실측이 아니다.

| 근거 | 시사점 |
|---|---|
| 현재 오라클 R²가 이미 0.86~0.95 (q_frac_wide 0.9501, q_abs 0.8595) | HI 64종이 이미 대부분의 신호를 포착 — raw 추가로 인한 **전체 평균** 개선폭은 작을 가능성 높음(한 자릿수 %) |
| `MODEL_SPECS.md §5.1`이 명시한 잠재 이득 지점: "HUST chg_hi/mid처럼 HI만으로 신호가 약한 구간" | 개선은 **균일하지 않고 국소적**일 것 — 약한 시나리오에서 상대적으로 크고(추정 10~30%), 강한 시나리오에서는 0에 수렴하거나 과적합으로 오히려 악화 가능 |
| HI 64종 자체가 V/I/Q에서 유도된 통계량(§5.1 "중복성") | raw가 주는 **순증분** 정보는 "HI가 요약하며 버린 것"(예: 국소 굴곡의 미세 형태, HI가 캡처 못하는 비선형 패턴)에 국한 — 정보 상한이 있음 |
| 방안1은 파라미터가 96차원 그대로 늘어나 과적합 위험(특히 좁은 존, n_samples 적은 시나리오)이 방안2보다 큼 | 방안1은 오히려 **일부 시나리오에서 방안2보다 성능이 떨어질** 가능성 있음 |

**결론**: "전체 RMSE가 X% 개선된다"고 미리 숫자를 박는 건 이 시점에 근거가 없다.
대신 **국소 개선 가설**(HUST chg_hi/mid 등 약한 시나리오)을 검증 목표로 삼고,
구현 후 시나리오별 R² 비교표로 실측해야 한다 — 이게 바로 `MODEL_SPECS.md`가 미뤄온
그 진단이다. 두 방안 사이의 상대 비교로는: **방안2가 방안1보다 과적합에 강해 기대
성능이 더 안정적**이라고 볼 근거(파라미터 수, CNN의 귀납적 편향)는 있다.

### 3.2 구현 복잡도

| | 방안 1 (flatten concat) | 방안 2 (CNN 임베딩) |
|---|---|---|
| 데이터 파이프라인 | **이미 완료** (양쪽 공통, §0) | **이미 완료** (양쪽 공통, §0) |
| 새 모듈 | 없음 — `RawMLPModel`의 flatten 패턴 재사용 | 없음 — `RawCNN`(이미 구현·검증됨, 42,401 파라미터) 재사용, 신규 인스턴스만 |
| `cap_heads.py` | `_HEAD_IN` 130→226, 5개 헤드 전부 슬라이싱 로직 수정 | `_HEAD_IN` 130→194, 5개 헤드 전부 슬라이싱 로직 수정 (**두 방안 모두 동일한 5곳을 건드림**) |
| `scr_model.py` | `feat` 조립에 `x_raw.reshape(B,-1)` 추가 | `self.raw_cnn = RawCNN(...)` 추가 + forward에 CNN forward pass 추가 |
| `train_scr.py` | Phase 2에서만 `FastTensorLoader(include_raw=True)` — Phase 1은 불필요(§1.2, `_p1_model_cfg`가 강제로 `regression_model="mlp"`이므로 CNN/raw 미사용) | 동일 |
| Transformer 계열 헤드(3종) | 226차원을 토큰화하는 방식을 다시 정해야 함(현재 "probe/scen/meta 3토큰" 또는 "피처별 토큰" 구조에 raw 96개를 어떻게 태울지 설계 필요 — 토큰 수가 96개 늘어나면 어텐션 비용도 커짐) | **192차원 그대로 "4토큰"(probe/scen/cnn_emb/meta)으로 자연스럽게 확장** — 기존 3-토큰 구조에 토큰 하나만 추가하면 됨, 설계 고민 거의 없음 |
| `iTransformer`/`FTTransformer`(피처별 토큰 96/194개) | raw 96차원이 전부 "개별 스칼라 토큰"이 되어 시퀀스 길이 130→226(+74%), 어텐션 비용 O(n²)로 급증 | 시퀀스 길이 130→194(+49%), CNN이 이미 압축해준 임베딩이라 개별 스칼라 토큰화가 의미상 어색함(64차원 임베딩 벡터를 다시 64개 스칼라 토큰으로 쪼개는 건 부자연스러움 — 별도 토큰 1개로 넣는 게 맞음, 즉 방안2는 오히려 iTransformer류에 대해 설계를 다시 정해야 함) |
| 체크포인트/설정 | `model_cfg`에 `with_raw_concat` 플래그, yaml에 옵션 추가 | `model_cfg`에 `with_raw_cnn` 플래그, yaml에 옵션 추가 (**MODEL_SPECS.md §5.2가 이미 pseudo-code로 설계해둠** — 그대로 구현) |
| 추론 비용 | 96차원 추가 concat, 사실상 무시 가능 | 세그먼트당 RawCNN forward 1회 추가(42K 파라미터, RAW_N=48 — 가볍지만 방안1보다는 무겁다) |

**요약**: MLP/ResNet 계열 헤드는 두 방안 다 거의 동등한 노력. **Transformer 계열
3종(현재 `scr.yaml`에서 선택 가능한 5개 헤드 중 3개)에서는 방안2가 명확히 쉽다** —
기존 토큰 구조에 자연스럽게 얹히기 때문. 방안1은 96개의 원시 스칼라를 토큰화하는
설계를 새로 고민해야 해서 실질적으로 더 손이 간다. 그리고 방안2는 **이미 pseudo-code
수준까지 설계돼 있는(§5.2) 문서가 있어 설계 리스크가 낮다.**

### 3.3 리뷰어 설득력

| | 방안 1 | 방안 2 |
|---|---|---|
| 서사 일관성 | 이 프로젝트 전체가 "raw에서 손수 설계한 HI로 압축 → L0로 희소화"라는 축을 따라왔는데(HI_DESCRIPTION.md, SCENARIO_STRATEGY.md 전체), 방안1은 그 축과 반대로 "raw를 압축 없이 그대로 넣는다"—설계 철학상 이질적 | **분류기가 이미 정확히 이 패턴(CNN 압축 후 HI와 융합)을 쓰고 있다**(§1.2, 활성 프로덕션 설정) — "회귀 헤드도 분류기와 같은 원칙을 따른다"는 **구조적 일관성 주장**을 그대로 할 수 있음 |
| 해석 가능성 | 96개의 개별 raw 스칼라가 각각 무슨 의미인지 사후 설명하기 어려움(어느 q_frac 위치의 전압/전류값이 중요했는지는 attribution 기법 없이는 불투명) | `AttentionPool1d`(RawCNN 내부)가 "어느 q_frac 위치가 중요한지" 학습하므로 어텐션 가중치를 직접 시각화해 해석 가능 — 이미 `raw_cnn.py` 설계 의도에 명시됨 |
| 파라미터 효율성 주장 | HI 64개를 어렵게 L0로 희소화해놓고 raw 96개를 무압축으로 얹으면, "우리는 sparse/효율적 설계를 지향한다"는 논문의 핵심 주장(비용 인지 L0 게이트, Ablation A-2)과 정면으로 부딪힘 — 리뷰어가 "그럼 애초에 HI 설계가 왜 필요했나"라고 물을 위험 | CNN 임베딩도 64차원으로 **HI와 동일한 크기로 맞춰뒀다**(`raw_cnn.py`: "출력 차원: D_CNN=64 — HI 64D와 동일하게 맞춤", 의도적 설계) — "동등한 예산 안에서 압축 방식만 다른 두 정보원을 공정 비교한다"는 주장이 가능 |
| 기존 baseline과의 관계 | B-4(raw-only)와 구분이 모호해질 위험(§1.3) — "그냥 B-4에 HI를 얹은 것" 정도로 축소 해석될 수 있음 | B-4(raw-only, CNN)와 자연스러운 스펙트럼을 이룸: HI-only → **HI+CNN(이 작업)** → raw-only(B-4, `raw_mlp`와 별개로 CNN 버전도 이미 구조상 거의 존재) — ablation 스토리가 깔끔함 |

### 3.4 모델 학습 용이성 / 전문가 관점에서 봐야 할 지표

두 방안 모두 구현 후 반드시 아래를 측정해야 한다(이 프로젝트가 이미 갖춘 인프라로
거의 무료로 얻을 수 있음):

1. **시나리오별 R²/RMSE 분해** (기존 `visualize_results.py`가 이미 Overall+시나리오별
   분해를 그림) — HUST `chg_hi`/`chg_mid`처럼 HI만으로 약했던 시나리오가 실제로
   개선되는지가 이 작업 전체의 존재 이유(`MODEL_SPECS.md` §5.1)이므로, **이 분해 없이
   "전체 RMSE"만 보면 판단 자체가 무의미**하다.
2. **oracle vs hard 라우팅 gap(routing_gap_pct)** — raw 신호가 회귀에 새로 들어가면
   분류기 라우팅 오류의 대가(hard routing 시 엉뚱한 헤드로 보내졌을 때의 손실)도
   달라질 수 있다. 이미 `visualize_results.py`가 자동 계산.
3. **avg_cost / 인퍼런스 시간(ms/sample)** — L0 비용 인지 설계(Ablation A-2)와의 정합성.
   방안2는 CNN forward 비용이 avg_cost 지표에 안 잡히는 구조적 맹점이 있음(HI 비용은
   `get_hi_cost_vector`로 정량화되지만 CNN 연산 비용은 별도 트래킹이 없다) — **CNN
   FLOPs/추론시간을 avg_cost 표에 별도 행으로 추가하는 것을 권장**(안 하면 "공짜로
   성능이 올랐다"는 잘못된 인상을 줄 위험).
4. **과적합 신호** (train/val gap, 특히 n_samples 적은 좁은 존) — 방안1은 파라미터가
   96차원 통째로 늘어 HI 대비 원시 신호 비중이 커지므로, 특히 세그먼트 수가 적은
   시나리오(q_abs의 `low` 존처럼 n_k=1~2)에서 과적합 위험이 더 크다 — **존별 학습
   샘플 수 대비 파라미터 증가율**을 학습 곡선과 함께 봐야 한다.
5. **classifier CNN vs MLP 격리 실험** (§1.2의 공백) — 이 작업과 별개로, 거의 무료로
   실행 가능한 선행 실험: `classifier.type: mlp`로 한 번 더 돌려 현재 CNN 분류기
   대비 정확도 격차를 처음으로 측정한다. 이 숫자가 나오면 "CNN이 이 데이터에서
   실제로 얼마나 가치 있는가"에 대한 **이 프로젝트 최초의 직접 증거**가 되고, 회귀
   헤드 CNN의 기대 이득을 훨씬 덜 추측에 의존해 예측할 수 있다.

---

## 4. 추천: **방안 2 (CNN 임베딩 융합)**

### 근거 요약

1. **이미 검증된 패턴의 확장이다.** 분류기가 이 정확한 아키텍처(CNN(raw) + HI
   융합)를 프로덕션 기본값으로 이미 쓰고 있다(§1.2) — 새 설계 리스크가 아니라
   기존 설계를 한 곳 더 적용하는 문제로 축소된다.
2. **MODEL_SPECS.md §5.2가 이미 pseudo-code까지 설계해뒀다.** `_HEAD_IN_WITH_CNN=194`,
   `SCRModel.with_raw_cnn` 플래그 구조 그대로 구현하면 된다 — 이 문서에서 유일하게
   바꾼 건 "가중치 공유"를 "독립 인스턴스"로 낮춘 것뿐(§2, 파이프라인이 분리 학습
   구조로 굳어졌기 때문).
3. **Transformer 계열 3개 헤드에 자연스럽게 확장된다**(§3.2) — 방안1은 96개 스칼라
   토큰화를 새로 설계해야 하는데, 이는 5개 헤드 중 3개에서 방안1의 구현 비용을
   실질적으로 방안2보다 높게 만든다.
4. **파라미터 예산이 HI와 대등해(64=64) 리뷰어에게 "공정한 비교"라는 주장을 할 수
   있다**(§3.3) — 이 프로젝트의 핵심 주장(비용 인지 희소 선택)과 방안1보다 덜
   충돌한다.
5. **해석 가능성**(AttentionPool 시각화)이 방안1보다 명확히 좋다 — 논문에서 "raw가
   기여한 부분이 무엇인가"를 보여줄 구체적 그림을 만들 수 있다.

방안1이 이기는 유일한 축은 "추론 비용"(CNN forward 없음)과 "가장 단순한 MLP/ResNet
헤드에서의 구현 난이도"(둘 다 비슷하게 쉬움)뿐이다 — 둘 다 방안2의 근본적 약점이라기
보다 사소한 트레이드오프다.

---

## 5. 구현 계획 (방안 2)

`MODEL_SPECS.md §5.2`의 설계를 아래처럼 구체화한다. Phase 1은 건드리지 않는다(§1.1의
`_p1_model_cfg`가 이미 `regression_model="mlp"`로 강제 — CNN은 Phase 2 전용 옵션이
되어야 함).

| 파일 | 변경 |
|---|---|
| `5_model/models/scr_model.py` | `SCRModel.__init__`에 `with_raw_cnn: bool=False` 인자 추가 → `True`면 `self.raw_cnn = RawCNN(d_out=64)` 생성. `forward`에서 `with_raw_cnn`이면 `cnn_emb = self.raw_cnn(batch["x_raw"])`를 `feat` 조립에 추가(194차원) |
| `5_model/models/cap_heads.py` | `build_cap_head(model_cfg, ..., with_raw_cnn=False)` 인자 추가. `_HEAD_IN`을 `with_raw_cnn` 여부에 따라 130/194로 동적 결정. **5개 헤드 클래스 전부**(MLPHead/TransformerHead/ITransformerHead/ResNetTabHead/FTTransformerHead)의 `__init__`이 `_HEAD_IN`을 받아 슬라이싱 로직(`x[:, :N_HI]` 등)에 CNN 토큰 위치를 반영하도록 수정 — TransformerHead/FTTransformerHead는 토큰 3개→4개(cnn_emb 토큰 추가), iTransformer/FTTransformer는 194개 피처 중 CNN 임베딩 64개를 **개별 스칼라 토큰이 아니라 하나의 토큰**으로 넣도록 설계 변경(§3.2 지적사항 그대로 반영) |
| `5_model/train_scr.py` | Phase 2 분기에서 `model_cfg.get("with_raw_cnn", False)`를 읽어 `FastTensorLoader(..., include_raw=with_raw_cnn)`로 전환. `build_cap_head`/`SCRModel` 생성 호출에 `with_raw_cnn` 전달 |
| `5_model/config/scr.yaml` | `model:` 섹션에 `with_raw_cnn: false` 키 추가(주석: "회귀 헤드에 CNN(raw V/|I|) 임베딩 융합 — MODEL_SPECS.md §5.2 / REGRESSION_UPGRADE.md") |
| `5_model/test_scr.py`, `5_model/evaluation/scr_evaluator.py` | 체크포인트에서 `with_raw_cnn` 플래그 복원 후 모델 재구성 시 반영(§9.1의 `clf_type` 저장 패턴과 동일한 방식으로 `with_raw_cnn`도 체크포인트 메타에 저장) |
| `5_model/utils/uncertainty.py` | **수정 불필요**(§1.4) |

구현 순서 제안:
1. `RawCNN` 재사용 확인(이미 됨) → `cap_heads.py`의 `MLPHead`만 우선 194차원 지원(가장
   단순, 리스크 낮음)
2. `scr_model.py`/`train_scr.py` 연결 → MLP 헤드로 end-to-end 스모크 테스트
3. 시나리오별 R² 분해로 HUST `chg_hi`/`chg_mid` 개선 여부 1차 확인(§3.4-1, 이게
   `MODEL_SPECS.md`가 미뤄온 진단에 대한 답)
4. 개선이 확인되면 ResNetTab/Transformer 계열로 확장(§3.2 토큰화 설계 반영)
5. §3.4-5의 classifier mlp-vs-cnn 격리 실험을 병행해 "CNN 자체의 가치"에 대한 첫
   직접 증거 확보

---

## 6. 리스크 / 한계

- **가중치 비공유로 인한 파라미터 중복**(§2): 분류기·회귀 헤드가 각자 RawCNN을
  학습 — 42K×2 파라미터는 무시할 수준이지만, 두 CNN이 서로 다른 raw 표현을 학습할
  수 있어(한쪽은 CE, 한쪽은 MSE 그래디언트) "같은 raw 곡선인데 왜 다르게 해석하는가"라는
  리뷰어 질문에 대한 대비가 필요(답: 두 태스크의 최적 표현이 다를 수 있음은 정상 —
  실제로 공유가 항상 이득이라는 보장도 없음).
- **HUST chg_hi/mid 문제의 원인이 수렴 문제일 가능성**(`MODEL_SPECS.md` §5.3의 원래
  우려) — 여전히 미해결. 이 작업 자체가 그 진단 실험이 되므로, 구현 후 개선이 없다면
  "raw 신호 부재가 원인이 아니었다"는 것도 유효한 결론이며 실패로 볼 필요 없음.
- **avg_cost 지표의 맹점**(§3.4-3) — 대응 전까지는 "비용 대비 성능"을 왜곡되게
  보고할 위험.

---

## 8. 추가 검토 (2026-07-30 후속) — t 채널, latent 재사용, 3자 비교

### 8.1 t(시간) 채널 추가 — 보류, 우선순위 낮음

q_frac 그리드에서 `dq=I·dt`이므로 `dt/dq_frac ≈ q_tot/(48·I(q_frac))` — 세그먼트
내부의 상대적 시간 페이싱은 이미 저장된 `|I|` 곡선에서 근사 도출 가능하다(CNN이
conv로 암묵 학습 가능한 관계). 순증분 정보는 제한적일 가능성이 높다.

**리스크는 사실 두 가지이고, 각각 다른 처리가 필요하다** (하나만 하면 안 된다):

1. **절대 위치 누출** — 원시 t를 시프팅 없이 그대로 넣으면, 세그먼트 시작 시각
   자체가 "이 사이클이 전체 테스트 중 얼마나 진행됐는가"(≈대략적인 노화 정도)를
   거의 곧바로 알려준다. → **`t - t[0]`로 세그먼트 시작점을 0으로 시프팅하면 이
   문제는 해결된다.**
2. **지속시간(스케일) 누출** — 시프팅은 시작점만 0으로 옮길 뿐 길이는 그대로
   보존한다(`t_shifted[-1] = t[-1]-t[0]` = 원래 지속시간 그 자체). CC 구간에서
   세그먼트 지속시간은 그 세그먼트의 절대 Q(≈SOH 상관)와 거의 선형이므로, 시프팅만
   하면 `stat_q_abs`/`stat_energy_seg`를 leak_cols로 배제한 것과 **같은 성격의
   간접 누출**이 스케일 형태로 재발한다 — 예: 건강한 셀 세그먼트는 시프팅 후
   `[0,...,200]`초, 같은 q_frac 구간을 도는 노화 셀 세그먼트는 정전류 하에 더 오래
   걸려 `[0,...,800]`초 — CNN이 "48개 그리드가 끝에 가서 얼마나 큰 값까지
   올라가는가"만 봐도 지속시간(≈절대 Q)을 그대로 읽어낼 수 있다. → **지속시간으로
   한 번 더 나눠야 한다.**

즉 시프팅과 정규화는 서로 다른 문제를 없애므로 **둘 다** 필요하다 —
`t_rel = (t-t[0]) / (t[-1]-t[0])` (q_frac 그리드에 보간). 시프팅만 하고 나누기를
생략하면 문제 2번이 그대로 남는다.

**"V, I와 같은 방식으로"가 아니라 "Q와 같은 방식으로" 정규화해야 한다는 점에
주의.** `_resample_segment(vs, ims, qcs)`가 지금 하는 일을 정확히 나누면:

```python
q_rel = qcs - qcs[0]        # ① Q를 시프팅
x     = q_rel / q_rel[-1]   # ② Q를 [0,1]로 스케일링 → 좌표축(도메인)으로만 사용
rv = np.interp(grid, x, vs)              # V는 원본 물리값(Volt) 그대로 — 값 자체는 정규화 안 함
ri = np.interp(grid, x, np.abs(ims))     # I도 원본 물리값(Amp) 그대로 — 값 자체는 정규화 안 함
```

**실제로 시프팅+스케일링(정규화)이 적용되는 건 Q 하나뿐**이고, 그 결과(`x`)는
V/I를 리샘플할 좌표축으로만 쓰인다. V/I 값 자체는 정규화 없이 원본 단위 그대로
저장된다 — 가능한 이유는 전압은 화학종이 정하는 범위에 묶여 있고 전류는 프로토콜이
명령한 값이라, 절대 스케일 자체가 SOH를 직접 누설하지 않기 때문이다. 반대로 Q는
절대 스케일 자체가 SOH와 직접 상관되므로(그 세그먼트가 전체 용량 중 얼마나 큰
덩어리인지) 시프팅+스케일링으로 그 정보를 일부러 지운다.

T는 성격상 V/I가 아니라 **Q 쪽 부류**다(절대 지속시간 자체가 SOH와 상관되는
변수라는 점에서 Q와 같다, 위 문제 2번) — 그러니 V/I처럼 "원본 그대로 보간"하면 안
되고, Q가 받는 것과 같은 시프팅+스케일링 처리(`t_rel`)를 거쳐야 한다. 좌표축(도메인)
자체는 그대로 Q 기반 `x`를 쓰고, T는 그 위에 얹히는 세 번째 "값" 채널이지만
V/I와 달리 자기 자신도 정규화가 필요하다는 것이 핵심 차이다:

```python
t_rel = (t - t[0]) / (t[-1] - t[0])   # Q(x)가 받는 것과 동일한 시프팅+스케일링
rt    = np.interp(grid, x, t_rel)     # 좌표축은 그대로 x(Q 기반), 보간 대상 값만 t_rel
```

**넣을 근거가 되는 유일한 지점**: `HI_DESCRIPTION.md`가 명시한 "HUST 다단계 C-rate
포착" — 48점 리샘플 `|I|`가 뭉갤 수 있는 급격한 전류 계단 전환을, `t_rel(q_frac)`
매핑이 더 날카롭게 보여줄 가능성. 우선순위는 낮음 — **본 계획(§8.2~8.6)이 raw
fusion 자체의 가치를 먼저 확인한 뒤**, V,I만 대비 V,I,t_rel 3채널의 ablation으로
별도 실험할 항목으로 남긴다.

### 8.2 분류기 CNN latent 재사용 — 유용하나 표현 품질 트레이드오프 있음

`train_classifier.py`는 자기 docstring에 "B안 — 회귀와 완전 분리"로 명시돼 있고,
실제 의존성은 **Phase 1의 probe mask뿐**(`_load_probe_masks`) — Phase 2의 `cap_head`
가중치는 전혀 읽지 않는다. 따라서 "분류기 CNN을 먼저 확정하고 Phase 2가 그 고정
임베딩을 가져다 쓰는" 순서 재배치는 구조적으로 어렵지 않다.

**우려**: CE로 학습된 CNN은 클래스 간 마진을 벌리는 방향으로만 그래디언트를 받고,
일단 올바르게 분류된 샘플에서는 그래디언트가 saturate된다 — 같은 클래스 내부의
연속적 SOH 차이를 보존할 유인이 없다(neural collapse류 현상). 회귀가 정확히 그
연속적 변화를 필요로 하므로, 판별 학습 임베딩이 회귀에 최적이라는 보장이 없다.

### 8.3 세 가지 방식 비교 및 실행 결정

(a) 그대로 재사용(frozen) / (b) 헤드별 독립 학습 / (c) 헤드별 가중치 공유(joint
multi-task) 세 가지를 구현 복잡도·학습 비용·표현 품질·파이프라인 영향으로 비교한
결과(상세 비교표는 대화 로그 참조, 요지만 재기록):

- (c)는 `SCRModel`과 `CNNProbeClassifier`가 서로 다른 `nn.Module`·다른 스크립트·다른
  옵티마이저로 완전히 분리된 현재 구조("B안 — 완전 분리"라는 명시적 설계 의도)를
  뒤집는 가장 비용이 큰 선택지이고, (b)조차 아직 검증 안 된 가설(raw fusion이
  회귀에 도움되는가) 위에 negative-transfer 리스크까지 얹는 것이라 **현재 시점에서
  기각**.
- **(a)를 초저비용 사전 검증(sanity check)으로 먼저 실행하고, 신호가 확인되면
  §5의 (b)로 간다.** (a)는 이미 학습된 분류기 CNN을 얼려서 추론만 태우면 되므로
  새 학습 루프가 필요 없다 — "raw 정보가 회귀에 조금이라도 도움이 되는가"라는
  가장 근본적인 질문에 (b) 전체 구현보다 훨씬 싸게 답할 수 있다.

### 8.4 (a)/(b) 공용 구현 설계 — 하나의 스위치로 두 옵션을 다 지원

(a)와 (b)는 사실 같은 코드 경로의 두 설정일 뿐이다 — 이렇게 설계하면 사전 검증
코드가 버려지지 않고 §5의 (b) 구현으로 그대로 이어진다.

```yaml
# scr.yaml, model: 섹션에 추가
with_raw_cnn: false                 # true → cap_head 입력에 CNN(raw) 임베딩 融합 (194차원)
raw_cnn_pretrained_from: null        # null → (b) RawCNN 랜덤 초기화, Phase2에서 함께 학습
                                      # 경로 지정 → (a) 그 classifier 체크포인트의 RawCNN을 얼려서 재사용
```

`scr_model.py` 의사코드:

```python
if with_raw_cnn:
    self.raw_cnn = RawCNN(d_out=64)
    if raw_cnn_pretrained_from:
        ckpt = torch.load(raw_cnn_pretrained_from, map_location="cpu")
        cnn_state = {k[len("cnn."):]: v for k, v in ckpt["clf_state"].items() if k.startswith("cnn.")}
        self.raw_cnn.load_state_dict(cnn_state)
        for p in self.raw_cnn.parameters():
            p.requires_grad_(False)   # (a): 얼림 — Phase2 MSE 그래디언트가 CNN에 안 흐름
        self.raw_cnn.eval()            # BatchNorm 등 항상 추론 모드로 고정
else:
    self.raw_cnn = None
```

`forward`는 얼렸든 학습하든 동일하게 `cnn_emb = self.raw_cnn(batch["x_raw"])`를
호출하면 되므로 분기 불필요 — `requires_grad_(False)` + `eval()`만으로 (a)/(b)가
같은 forward 코드를 공유한다. `cap_heads.py`의 194차원 지원(§5, MLP/Transformer/
ResNetTab 3종만 이번 검증에 필요)도 (a)/(b) 공용이다.

**분류기 체크포인트 확인 완료**: 아래 5개 baseline run 모두 `classifier/clf_best.pt`
보유, 전부 `clf_type="cnn", n_hi=64, d_hidden=64` — (a)에 바로 재사용 가능(추가 분류기
학습 불필요).

### 8.5 검증 1 — 회귀 모델 불문 일관된 개선 (axis 고정: q_frac_wide n1=35%, n2=20%, N=4)

목적: **어떤 회귀 헤드를 쓰든** CNN(raw) 임베딩을 같이 넣는 것이 이득인지 확인.
세 헤드 모두 같은 축·같은 데이터·같은 Phase 1 게이트·같은 frozen CNN을 공유해
"헤드 아키텍처"만 변수로 남긴다.

| # | 회귀 헤드 | Baseline (기존) | +CNN(신규, 이 실험에서 생성) | frozen CNN 출처 |
|---|---|---|---|---|
| 1 | mlp | `0723_1633_p2_mlp_qfw_35%_20%` | `<날짜>_p2_mlp_qfw_35%_20%_cnnfrozen` | `0723_1633_p2_mlp_qfw_35%_20%/classifier/clf_best.pt` |
| 2 | transformer | `0723_2356_p2_tr_qfw_35%_20%` | `<날짜>_p2_tr_qfw_35%_20%_cnnfrozen` | 위와 동일(축·파라미터 동일하므로 공유) |
| 3 | resnet_tab | `0724_0111_p2_res_qfw_35%_20%` | `<날짜>_p2_res_qfw_35%_20%_cnnfrozen` | 위와 동일 |

세 baseline 모두 `scenario_spec.json` 확인 결과 `axis=q_frac_wide,
n1=0.35, n2=0.2, n_samples=4`로 동일 — frozen CNN을 세 실험이 공유해도 "같은 raw
표현으로 여러 헤드를 비교한다"는 취지에 어긋나지 않는다(오히려 헤드 비교의
교란 변수를 하나 줄여준다).

실행 스케치(예: #1):
```bash
python 5_model/train_scr.py --config 5_model/config/scr.yaml \
  --phase 2 --seg-axis q_frac_wide --axis-config '{"n1":0.35,"n2":0.2,"n_samples":4}' \
  --gates-from "_5_data_model_scr/0723_1633_p2_mlp_qfw_35%_20%"
# config.yaml: model.regression_model=mlp, model.with_raw_cnn=true,
#              model.raw_cnn_pretrained_from="_5_data_model_scr/0723_1633_p2_mlp_qfw_35%_20%/classifier/clf_best.pt"
```
\#2/#3은 `regression_model`만 `transformer`/`resnet_tab`으로 바꾸고 나머지 동일.
`--gates-from`으로 baseline이 이미 확정한 Phase 1 게이트를 그대로 재사용해
"입력에 CNN이 추가된 것" 외의 차이를 없앤다.

### 8.6 검증 2 — 시나리오 축 불문 일관된 개선 (모델 고정: mlp)

목적: **어떤 세그멘테이션 축을 쓰든** CNN(raw) 임베딩이 이득인지 확인. 축마다
세그먼트 경계·raw 곡선이 다르므로, frozen CNN은 **그 축 자신의 classifier**에서만
가져온다(축 간 공유 불가 — §8.5와의 차이).

| # | 축 | Baseline (기존) | +CNN(신규) | frozen CNN 출처 |
|---|---|---|---|---|
| 1 | q_frac_wide (n1=35%,n2=20%,N=4) | `0723_1633_p2_mlp_qfw_35%_20%` | §8.5 #1과 동일 run 재사용 | `0723_1633.../classifier/clf_best.pt` |
| 2 | vqslope (mode=dva, N=1) | `0725_1348_p2_mlp_vqs_dva` | `<날짜>_p2_mlp_vqs_dva_cnnfrozen` | `0725_1348_p2_mlp_vqs_dva/classifier/clf_best.pt` |
| 3 | q_abs (mid_start=20%,mid_end=50%,seg_len=15%,N=4) | `0729_1908_p2_mlp_qabs_20-50%` | `<날짜>_p2_mlp_qabs_20-50%_cnnfrozen` | `0729_1908_p2_mlp_qabs_20-50%/classifier/clf_best.pt` |

q_frac_wide+mlp 조합은 §8.5 #1과 완전히 같은 실험이므로 **중복 실행 불필요** —
신규 run은 vqslope·q_abs 2개만 추가하면 된다(§8.5의 1개 + §8.6의 2개 = 총 **신규
3개 run**으로 두 검증 모두 커버).

실행 스케치(vqslope):
```bash
python 5_model/train_scr.py --config 5_model/config/scr.yaml \
  --phase 2 --seg-axis vqslope --axis-config '{"mode":"dva","n_samples":1}' \
  --gates-from "_5_data_model_scr/0725_1348_p2_mlp_vqs_dva"
# config.yaml: model.regression_model=mlp, model.with_raw_cnn=true,
#              model.raw_cnn_pretrained_from="_5_data_model_scr/0725_1348_p2_mlp_vqs_dva/classifier/clf_best.pt"
```
q_abs는 `--axis-config '{"mid_start":0.2,"mid_end":0.5,"seg_len":0.15,"n_samples":4}'`
+ 해당 run의 classifier 경로로 동일하게 구성.

### 8.7 평가 지표 및 판정 기준

- **주 지표**: oracle 라우팅 기준 전체 RMSE/MAE/R²/MAPE — 분류 라우팅 오차를
  배제하고 "CNN 입력이 회귀 헤드 자체를 개선하는가"만 순수하게 본다.
- **부 지표**: 시나리오별 R² 분해(`visualize_results.py`가 이미 자동 생성) — 특히
  `MODEL_SPECS.md §5.1`이 지목한 HI 신호 약한 시나리오(축마다 다름: q_frac_wide는
  `chg_hi`/`chg_mid` 경계 겹침 구간, q_abs는 `high` 존)에서 개선이 **불균일하게(더
  크게)** 나타나는지 확인 — 균일하지 않고 약한 곳에서 크게 나타나야 "raw가 HI의
  빈틈을 메운다"는 가설과 부합.
- **판정 기준**: "일관된 개선" = 5개 신규 run 전부에서 baseline 대비 oracle RMSE가
  개선(또는 최소 유의미한 악화 없음) — 방향성이 핵심이지 개선 폭의 크기가
  아니다(폭은 §3.1에서 이미 불확실하다고 명시함).
- **한계**: 단일 seed 비교다(비용을 최소화하는 sanity check이므로). 여기서 신호가
  보이면 §5 (b) 본 구현 단계에서 다중 seed로 통계적으로 재확인한다 — 이 단계에서
  seed 흔들림까지 통제하려 하지 않는다.

### 8.8 결과에 따른 후속 조치

| 결과 | 조치 |
|---|---|
| 5개 run 전부 일관 개선 | §5의 (b) — 헤드별 독립 학습 CNN을 정식 구현. 개선이 컸던 축/헤드부터 우선순위 부여 |
| 일부만 개선(예: mlp/resnet_tab만, transformer는 악화) | (b) 적용 범위를 개선이 확인된 조합으로 좁힘 — 나머지는 보류 |
| 축별로 갈림(예: q_abs만 개선, q_frac_wide/vqslope는 무변화) | "HI가 이미 잘 커버하는 축에서는 raw가 무용하다"는 가설로 재해석 — q_abs처럼 존재 자체가 상대적으로 새로 도입된 축에서만 (b) 적용 |
| 전부 무효과 또는 악화 | raw fusion 자체를 보류. `MODEL_SPECS.md §5.3`이 우려했던 "HUST chg_hi/mid 문제가 raw 신호 부재가 아니라 수렴 문제"라는 가설이 유력해짐 — 그 방향(학습 스케줄/lambda_l0 등)으로 재조사 전환 |

---

## 9. 향후 확장 (범위 밖, 참고용)

- t(시간) 채널 추가(RAW_CH 2→3) — CV 구간처럼 시간 의존적 동역학이 중요한 신호를
  q_frac 정규화가 지워버릴 가능성에 대응.
- 분류기·회귀 헤드 간 진짜 가중치 공유 — Phase1/2/classifier 학습 순서를 재설계해야
  하는 더 큰 작업, 이 문서 범위 밖.
- 방안1(flatten)을 "저비용 sanity check"로 먼저 돌려 raw 정보의 존재 자체가 도움이
  되는지만 빠르게 확인한 뒤, 방안2로 넘어가는 2단계 접근도 가능(방안1이 MLP 헤드
  기준으로는 방안2와 구현 비용이 비슷하므로 완전히 버리는 옵션은 아님).

---

## 10. 구현 완료 (2026-07-30) — with_raw_cnn/raw_cnn_pretrained_from 스위치

§8.4에서 설계한 (a)/(b) 공용 스위치를 실제로 구현했다. 변경 파일:

| 파일 | 변경 |
|---|---|
| `5_model/models/scr_model.py` | `with_raw_cnn`/`raw_cnn_pretrained_from`를 `model_cfg`에서 읽어 `self.raw_cnn`(RawCNN) 구성. 경로 지정 시 그 classifier 체크포인트의 `cnn.*` 서브모듈만 로드 후 `requires_grad_(False)`+`eval()`로 고정(방안 a). 미지정 시 랜덤 초기화 후 Phase2와 함께 학습(방안 b). `forward()`가 `cnn_emb`을 `feat`에 concat. **`train()` 오버라이드 추가** — 얼린 raw_cnn은 부모 `model.train()` 호출 후에도 항상 `eval()`로 되돌려, BatchNorm 러닝 통계가 그래디언트와 무관하게 오염되는 걸 막음(흔한 freezing 함정). 상대경로(`raw_cnn_pretrained_from`)는 repo root 기준으로 자동 해석(`_PROJECT_ROOT`) — `train_scr.py`/`test_scr.py` 둘 다 별도 처리 없이 동일하게 동작. |
| `5_model/models/cap_heads.py` | `_HEAD_IN_WITH_CNN=194` 추가. `MLPHead`/`ResNetTabHead`는 `head_in` 파라미터로 130/194 동적 지원. `TransformerHead`는 `with_raw_cnn=True` 시 토큰 3개→4개(cnn_emb을 64개 스칼라로 안 쪼개고 통째로 1개 토큰, §3.2 설계 그대로). `ITransformerHead`/`FTTransformerHead`는 `with_raw_cnn=True` 시 `NotImplementedError`로 명시적 보류(토큰화 방식 재설계 필요 — 이번 검증엔 불필요). `build_cap_head`가 `model_cfg["with_raw_cnn"]`을 읽어 위 전부에 전파. |
| `5_model/config/scr.yaml` | `model.with_raw_cnn: false`, `model.raw_cnn_pretrained_from: null` 기본값 추가(하위 호환 — 미지정 시 기존과 완전히 동일 동작). |
| `5_model/train_scr.py` | `_use_raw_cnn` 플래그로 `FastTensorLoader(..., include_raw=...)`를 raw_mlp 모드와 함께 트리거. **Phase 1은 `_p1_model_cfg`에서 `with_raw_cnn: False`를 강제**(게이트 선정과 무관한 Phase 2 전용 옵션이므로 우연히 흘러들어가는 것 방지). |
| `5_model/test_scr.py`, `5_model/evaluation/scr_evaluator.py` | **코드 변경 없음** — 검증 결과 두 파일 모두 `model_cfg`(체크포인트의 `cfg["model"]`)를 이미 제네릭하게 `SCRModel(...)`에 그대로 전달하고, `SegmentDataset.__getitem__`/`_collate`가 `x_raw`를 항상 포함해 배치를 만들므로 별도 분기 없이 자동으로 작동함을 확인. |

**검증(스모크 테스트, 실측 완료)**:
- 더미 텐서로 5개 헤드 × with_raw_cnn=False(하위호환) 전부 정상 동작 확인, i_transformer/ft_transformer + with_raw_cnn=True는 의도한 대로 `NotImplementedError` 발생 확인.
- mlp/transformer/resnet_tab × (a)frozen/(b)trainable 조합 전부 forward+backward 그래디언트 흐름 확인 — (a)는 raw_cnn 그래디언트 0, (b)는 0 아님. `model.train()` 후에도 (a)의 `raw_cnn.training`이 `False` 유지되는 것 확인.
- **실제 데이터로 Phase 2 학습(2 epoch) + `test_scr.py` 평가까지 전체 파이프라인 실행** — `q_frac_wide n1=35%/n2=20%/N=4` 축, (a) frozen 모드: `0723_1633_p2_mlp_qfw_35%_20%/classifier/clf_best.pt`를 정상 로드, oracle R²=0.8735(2 epoch만으로도 정상 수렴 궤적). (b) trainable 모드도 동일 데이터로 정상 학습 확인(2 epoch는 CNN이 처음부터 학습돼야 해서 아직 수렴 전 — 예상된 동작, 실제 실험은 500 epoch 설정 그대로 사용). 스모크 테스트 산출물은 정리 완료.

### 10.1 실험 설정 파일 (10개 신설 — (a) 5개 + (b) 5개)

§8.5/§8.6 계획을 실행하는 데 필요한 신규 run은 조합당 **정확히 5개**다(앞선 대화의 "3개"/"4개" 요약은 계산 착오 — 검증1이 qfw 축에서 mlp/transformer/resnet_tab 3개, 검증2가 mlp 모델에서 qfw/vqslope/q_abs 3개를 필요로 하는데 qfw+mlp 조합이 겹치므로 3+3−1=5). `regression_model`/`with_raw_cnn`/`raw_cnn_pretrained_from`은 `train_scr.py`/`run_pipeline.py`에 CLI 단축 옵션이 없어(둘 다 `--model-config` yaml로만 설정 가능), 실험마다 별도 yaml을 만들었다. 그 외 설정(`training`/`loss`/`uq` 등)은 기존 `scr.yaml`과 동일해 baseline과의 유일한 차이가 "+CNN 융합"이 되도록 통제했다.

**(a) 사전 검증 — 분류기 CNN 재사용(frozen)**: `raw_cnn_pretrained_from`이 기존 baseline의 `classifier/clf_best.pt`를 가리킨다. 초저비용(§8.3) — 새 CNN 학습이 전혀 없다.

| config | 축 | 회귀 헤드 | gates_from / raw_cnn_pretrained_from (동일 run) | 대응 검증 |
|---|---|---|---|---|
| `5_model/config/exp_qfw_mlp_cnn.yaml` | q_frac_wide (n1=35%,n2=20%,N=4) | mlp | `0723_1633_p2_mlp_qfw_35%_20%` | 검증1 #1 · 검증2 #1 (공유) |
| `5_model/config/exp_qfw_transformer_cnn.yaml` | q_frac_wide (동일) | transformer | `0723_2356_p2_tr_qfw_35%_20%`(gates) / `0723_1633.../clf_best.pt`(cnn, mlp와 공유) | 검증1 #2 |
| `5_model/config/exp_qfw_resnet_tab_cnn.yaml` | q_frac_wide (동일) | resnet_tab | `0724_0111_p2_res_qfw_35%_20%`(gates) / `0723_1633.../clf_best.pt`(cnn, mlp와 공유) | 검증1 #3 |
| `5_model/config/exp_vqslope_mlp_cnn.yaml` | vqslope (mode=dva,N=1) | mlp | `0725_1348_p2_mlp_vqs_dva` | 검증2 #2 |
| `5_model/config/exp_qabs_mlp_cnn.yaml` | q_abs (ms=20%,me=50%,sl=15%,N=4) | mlp | `0729_1908_p2_mlp_qabs_20-50%` | 검증2 #3 |

**(b) 본 구현 — 회귀 헤드 전용 RawCNN을 처음부터 학습**: `raw_cnn_pretrained_from: null` — 분류기의 CNN과 완전히 독립된 자기만의 RawCNN을 Phase2와 함께 랜덤 초기화부터 학습한다(§2/§3.3의 (b) 방안 그대로). **(a)와 파일명 끝의 `_b` 외에는 `raw_cnn_pretrained_from` 한 줄만 다르다** — 같은 `gates_from`(Phase 1 게이트)을 그대로 재사용해 "HI 서브셋은 baseline과 동일, CNN 학습 방식만 다름"이라는 통제된 비교가 유지된다.

| config | 축 | 회귀 헤드 | gates_from (동일) | 대응 검증 |
|---|---|---|---|---|
| `5_model/config/exp_qfw_mlp_cnn_b.yaml` | q_frac_wide (동일) | mlp | `0723_1633_p2_mlp_qfw_35%_20%` | 검증1 #1 · 검증2 #1 (공유) |
| `5_model/config/exp_qfw_transformer_cnn_b.yaml` | q_frac_wide (동일) | transformer | `0723_2356_p2_tr_qfw_35%_20%` | 검증1 #2 |
| `5_model/config/exp_qfw_resnet_tab_cnn_b.yaml` | q_frac_wide (동일) | resnet_tab | `0724_0111_p2_res_qfw_35%_20%` | 검증1 #3 |
| `5_model/config/exp_vqslope_mlp_cnn_b.yaml` | vqslope (동일) | mlp | `0725_1348_p2_mlp_vqs_dva` | 검증2 #2 |
| `5_model/config/exp_qabs_mlp_cnn_b.yaml` | q_abs (동일) | mlp | `0729_1908_p2_mlp_qabs_20-50%` | 검증2 #3 |

### 10.2 실행 명령어 (2026-07-30 파이프라인 재배치 이후 — §10.3 참조)

`run_pipeline.py`의 스텝 순서를 **6=Phase1 → 7=분류기 → 8=Phase2 → 9=평가**로
재배치했다(§10.3). 이 5개 검증 run은 기존 baseline의 Phase 1 게이트와 분류기를
그대로 재사용하므로(Phase1·분류기를 새로 학습하지 않음) **Step 8(Phase2)→9(평가)가
연속 구간이 되어 한 명령으로 끝난다** — 예전처럼 "Phase2 실행 → 콘솔에 출력된 run
dir를 수동으로 복사해 평가 명령에 붙여넣기"가 더 이상 필요 없다(중간에 분류기 Step이
끼어있던 옛 번호에서는 Phase2·평가만 골라 연속 구간으로 선택할 수 없어 불가피했던
불편이었음). `--gates-from`으로 기존 baseline의 Phase 1 폴더를 지정하면, Step 8이
그 폴더의 게이트를 로드하고, Step 9는 Step 8이 방금 만든 Phase 2 run을 같은
프로세스 안에서 자동으로 찾아 `--checkpoint`를 스스로 채운다. `with_raw_cnn`/
`raw_cnn_pretrained_from`은 각 yaml에 이미 박혀 있으므로 CLI로 다시 줄 필요 없다.

**(a) 사전 검증 — 분류기 CNN 재사용(frozen)**:

```bash
# ── 검증1 #1 / 검증2 #1 (공유): q_frac_wide + mlp + frozen CNN ──────────────
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qfw_mlp_cnn.yaml \
    --gates-from "_5_data_model_scr/0723_1633_p2_mlp_qfw_35%_20%"

# ── 검증1 #2: q_frac_wide + transformer + frozen CNN ─────────────────────────
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qfw_transformer_cnn.yaml \
    --gates-from "_5_data_model_scr/0723_2356_p2_tr_qfw_35%_20%"

# ── 검증1 #3: q_frac_wide + resnet_tab + frozen CNN ──────────────────────────
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qfw_resnet_tab_cnn.yaml \
    --gates-from "_5_data_model_scr/0724_0111_p2_res_qfw_35%_20%"

# ── 검증2 #2: vqslope(dva) + mlp + frozen CNN ────────────────────────────────
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_vqslope_mlp_cnn.yaml \
    --gates-from "_5_data_model_scr/0725_1348_p2_mlp_vqs_dva"

# ── 검증2 #3: q_abs(20-50%) + mlp + frozen CNN ───────────────────────────────
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qabs_mlp_cnn.yaml \
    --gates-from "_5_data_model_scr/0729_1908_p2_mlp_qabs_20-50%"
```

**(b) 본 구현 — 회귀 헤드 전용 RawCNN 처음부터 학습** (분류기 CNN과 독립, §2/§3.3 (b) 방안):
`raw_cnn_pretrained_from`이 없으므로 분류기가 먼저 학습돼 있을 필요가 없다 — `--gates-from`은
여전히 Phase 1 게이트 재사용을 위해 필요하지만, 그 폴더에 분류기가 없어도(또는 있어도 무시)
무방하다. `training.epochs=500`으로 CNN을 처음부터 학습하므로 (a)보다 학습 시간이 길다(§3.2).

```bash
# ── 검증1 #1 / 검증2 #1 (공유): q_frac_wide + mlp + CNN 처음부터 학습 ────────
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qfw_mlp_cnn_b.yaml \
    --gates-from "_5_data_model_scr/0723_1633_p2_mlp_qfw_35%_20%"

# ── 검증1 #2: q_frac_wide + transformer + CNN 처음부터 학습 ──────────────────
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qfw_transformer_cnn_b.yaml \
    --gates-from "_5_data_model_scr/0723_2356_p2_tr_qfw_35%_20%"

# ── 검증1 #3: q_frac_wide + resnet_tab + CNN 처음부터 학습 ───────────────────
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qfw_resnet_tab_cnn_b.yaml \
    --gates-from "_5_data_model_scr/0724_0111_p2_res_qfw_35%_20%"

# ── 검증2 #2: vqslope(dva) + mlp + CNN 처음부터 학습 ──────────────────────────
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_vqslope_mlp_cnn_b.yaml \
    --gates-from "_5_data_model_scr/0725_1348_p2_mlp_vqs_dva"

# ── 검증2 #3: q_abs(20-50%) + mlp + CNN 처음부터 학습 ─────────────────────────
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qabs_mlp_cnn_b.yaml \
    --gates-from "_5_data_model_scr/0729_1908_p2_mlp_qabs_20-50%"
```

**주의**: 각 yaml의 `data.output_dir`는 반드시 기본값 `"_5_data_model_scr"`(최상위) 그대로
둬야 한다 — `run_pipeline.py`의 새 run 폴더 자동탐지가 `_5_data_model_scr` 최상위만
스캔하도록 돼 있어서(§10.3의 사고 원인), 하위 폴더를 output_dir로 지정하면 엉뚱한
(과거의 무관한) run을 잘못 골라 그 폴더에 실수로 덮어쓸 수 있다.

실행 후 각 run의 `metrics/metrics.json`(`test.oracle.capacity`)을 해당 baseline(`0723_1633_p2_mlp_qfw_35%_20%` 등)과, 그리고 (a) vs (b) 같은 조합끼리도 비교하면 §8.7의 판정 기준(oracle RMSE/R² 개선 여부, 시나리오별 분해)에 더해 "분류기 CNN 재사용이 처음부터 학습보다 나은가"(§8.3 표현 품질 우려, §4.2)까지 함께 검증할 수 있다.

### 10.3 파이프라인 재배치 — 분류기(Step 7) ↔ Phase 2(Step 8)

**배경**: 원래 순서(Phase1=6, Phase2=7, 분류기=8)는 분류기 체크포인트를 항상
`{Phase2 run_dir}/classifier/clf_best.pt`에 저장하는 관례 때문이었다. 그런데
`train_classifier.py`는 실제로는 Phase 1 게이트(`run_dir/gates/*.json`)와
`scenario_spec.json`만 있으면 동작하고 Phase 2 가중치는 전혀 안 쓴다(코드로 확인
완료) — 그래서 with_raw_cnn(§5/§8)으로 "그 분류기의 CNN을 재사용"하려면 Phase 2를
**두 번**(베이스라인 학습 → 분류기 학습 → CNN 포함 재학습) 돌려야 하는 게 불필요하게
어색했다. `train_classifier.py` 자체는 코드 변경 없이 Phase 1 폴더를 `--run-dir`로
받아도 그대로 동작하므로, 분류기를 Phase 1 직후로 옮기고 Phase 2가 그 결과를 바로
재사용하도록 순서를 바꿨다 — **Phase 2를 두 번 돌릴 필요가 완전히 없어진다.**

**변경 파일**:
| 파일 | 변경 |
|---|---|
| `run_pipeline.py` | `STEPS` 재배치(6=Phase1, 7=분류기, 8=Phase2, 9=평가). 분류기(7) 주입은 `p2_run_dir` 대신 `p1_run_dir`(또는 `--gates-from`) 사용. Phase2(8) 주입에 `--gates-from` 유지 + **`--with-raw-cnn` 지정 시 Step 7이 방금 저장한 `{p1_run_dir}/classifier/clf_best.pt`를 `--raw-cnn-pretrained-from`으로 자동 연결**. `--seg-axis`/`--axis-config`/`--charge-m`/`--discharge-m`/`--scen-k`/`--phase2-lr` 주입 조건을 `num==7`(분류기, 해당 CLI 없음)에서 `num==8`(Phase2)로 이동. 신규 `--with-raw-cnn` CLI 추가. 모듈 docstring의 스텝 번호 예시도 갱신. |
| `5_model/train_scr.py` | `--with-raw-cnn`/`--raw-cnn-pretrained-from` CLI 추가(`cfg["model"]`에 오버라이드) — run_pipeline.py가 파이프라인 한 번에 값을 이어붙일 수 있게 함. |
| `5_model/test_scr.py` | `_resolve_classifier_ckpt()` 추가 — **레거시 위치(`{run_dir}/classifier/`, 기존 run 전부 이 위치) 우선 탐색, 없으면 신규 위치(`gates_from` 폴더의 `classifier/`)로 폴백**. 기존 run 전부 그대로 호환. |
| `5_model/train_classifier.py` | **변경 없음** — `run_dir`이 Phase 1 폴더든 Phase 2 폴더든 이미 동일하게 동작함을 코드로 확인. |

**기존 결과와의 충돌**: 없음. 위 표의 유일한 동작 변화는 새로 실행하는 파이프라인의
분류기 저장 위치뿐이고(`{p1_run_dir}/classifier/` 신규 관례), 기존에 이미
`{p2_run_dir}/classifier/`에 저장된 모든 run(이 문서의 5개 baseline 포함)은
`test_scr.py`가 레거시 경로를 여전히 먼저 찾으므로 그대로 유효하다.

**검증(스모크 테스트, 실측 완료)**: `run_pipeline.py 6 --to-step 9 --with-raw-cnn`
한 번으로 Phase1(신규 게이트 학습)→분류기(신규 학습, Phase1 폴더에 저장)→Phase2
(`--raw-cnn-pretrained-from`이 자동으로 그 분류기를 가리킴, `[scr_model] raw_cnn:
frozen, loaded from ...` 로그로 확인)→평가(hard 라우팅에 그 분류기 사용, `분류기
있음` 로그로 확인)까지 전부 한 프로세스 흐름으로 정상 완료됨을 확인.

**⚠️ 스모크 테스트 중 발생한 사고 및 교훈**: 테스트에 `output_dir:
"_5_data_model_scr/_smoke_test2"`(하위 폴더)를 썼는데, `run_pipeline.py`의 신규
run 폴더 자동탐지(`_find_new_run_dir`)가 `_5_data_model_scr` **최상위만** 스캔하도록
하드코딩돼 있어(`MODEL_OUTPUT_DIR`가 yaml의 `output_dir`을 안 따라감 — 이 재배치와
무관한 기존 결함), 하위 폴더에 새로 생긴 진짜 run을 "새로 생김"으로 못 잡고 "최상위에서
가장 최근 수정된 run"이라는 폴백이 발동해 **무관한 기존 run 2개**(`0730_1244_p1_
qabs_20-50%`, `0730_1956_p2_mlp_qfw_35%_20%` — 후자는 이 문서 §10.2의 검증1#1/
검증2#1 run으로 이미 실제 실행돼 있던 것)를 잘못 골랐다. 전자에는 저품질(2 epoch)
분류기가 실수로 추가 저장됐다가 즉시 삭제해 원상복구했고, 후자는 평가 산출물
(`metrics/figures/predictions/routing`)이 재생성됐는데 확인 결과 체크포인트는
그대로였고 재생성된 수치도 정상 범위(oracle R²=0.9363 등, config도 `exp_qfw_mlp_
cnn.yaml`과 정확히 일치)라 문제없음을 확인했다. **교훈**: `output_dir`은 항상 기본값
그대로 두고(§10.2에 경고 추가), 어쩔 수 없이 바꿀 경우 `run_pipeline.py`가 아니라
`train_scr.py`/`test_scr.py`를 직접 호출해 경로를 수동 지정할 것.

---

## 11. 방안 1 구현 및 실측 (2026-08-01) — 방안 2(CNN) 대비 명확히 우세

§4에서 방안2(CNN 임베딩)를 추천하고 실측까지 마쳤는데(§10, `docs/260731_RESULTS.md`),
그 결과가 **양쪽 방식(frozen 재사용/separate 학습) 다 baseline 대비 뚜렷한 개선이
아니었다**(평균 oracle Δ: frozen -0.0125, separate +0.0006 — 거의 중립~소폭 손해).
이에 따라 §3.1에서 "방안2가 방안1보다 과적합에 강해 기대 성능이 더 안정적"이라고
추정했던 판단을 실측으로 검증하기 위해 방안1도 마저 구현해 같은 5개 조합(검증1: qfw
축에서 mlp/transformer/resnet_tab, 검증2: mlp에서 qfw/vqslope/q_abs)으로 돌렸다.

### 11.1 구현 — `with_raw_flat` 스위치

§2 방안1 pseudocode(`feat = probe_x‖scen_x‖direction‖cap_init‖raw_v‖raw_i`)를 거의
그대로 구현하되, 한 가지 **의도적으로 원안과 다르게** 처리했다: HI는 이미 z-score
(mean0/std1)인데 raw_v(~3~4V)/raw_i(~0~5A)는 스케일이 전혀 달라 그대로 concat하면
raw 블록이 그래디언트를 불공정하게 지배할 위험이 있다 — `RawCNN`이 stem에서
`BatchNorm1d`로 채널 스케일을 흡수하는 것과 동등한 처리를 방안1에도 줘야 공정한
비교가 된다. 그래서 `SCRModel`에 `nn.BatchNorm1d(RAW_CH*RAW_N)`을 하나 추가해
flatten 직후·concat 직전에 정규화한다.

| 파일 | 변경 |
|---|---|
| `5_model/models/cap_heads.py` | `_D_RAW_FLAT=96`, `_HEAD_IN_WITH_RAW_FLAT=226` 상수 추가. `TransformerHead`에 `with_raw_flat` 파라미터 + 4번째 토큰(`raw_flat_embed: Linear(96,d_model)`, cnn_emb 토큰과 동일하게 "압축 없는 96차원을 통째로 토큰 1개로 선형 투영" — 96개 개별 스칼라 토큰화는 어텐션 비용이 급증해(§3.2) 피함). `build_cap_head`에 `with_raw_cnn`/`with_raw_flat` 동시 사용 금지 검증 + i_transformer/ft_transformer는 `NotImplementedError`(방안2와 동일 지원 범위: mlp/transformer/resnet_tab만). |
| `5_model/models/scr_model.py` | `with_raw_flat` 읽어 `raw_flat_norm=BatchNorm1d(96)` 구성. `forward()`에서 `raw_cnn`과 상호 배타적으로 `x_raw.reshape(B,-1)` → `raw_flat_norm` → `feat_parts`에 추가. |
| `5_model/config/scr.yaml` | `model.with_raw_flat: false` 기본값. |
| `5_model/train_scr.py` | `--with-raw-flat` CLI, `_use_raw_flat` 플래그로 `FastTensorLoader(include_raw=...)` 조건에 합류, Phase 1은 `_p1_model_cfg`에서 강제 `False`(방안2와 동일 이유). |
| `5_model/test_scr.py` | **무수정** — `SegmentDataset.x_raw`가 `include_raw`와 무관하게 항상 존재해 `model_cfg`만 맞으면 자동으로 동작(방안2 때와 동일한 이유, §8.4). |
| `5_model/visualize_results.py` | `_benchmark_inference`/`_compute_jacobian_profiles`의 `getattr(model, "raw_cnn", None) is not None` 체크에 `or getattr(model, "with_raw_flat", False)`를 추가 — 방안2 구현 때 겪었던 것과 같은 `KeyError: x_raw` 버그를 미리 방지(§10.2 참고 사례를 학습해 이번엔 사전 조치). 추가로 `--rep-cells` CLI를 신설해(§11.3) 특정 셀 하나만 직접 지정해 곡선 비교를 그릴 수 있게 함(기존엔 데이터셋당 최대 3셀 자동선정만 가능했음). |

**검증 순서**(방안2 때의 교훈 반영 — 값비싼 본 학습 전에 저비용 검증부터):
1. 더미 텐서 유닛 테스트 — mlp/transformer/resnet_tab forward+backward 통과,
   i_transformer/ft_transformer가 `NotImplementedError`, `with_raw_cnn`+`with_raw_flat`
   동시 지정 시 `ValueError` 확인. 전부 통과.
2. 실제 데이터로 2-epoch 스모크 학습(`output_dir`을 프로젝트 바깥 임시 경로로 지정해
   §10.3 사고 재발 방지) → `test_scr.py` 평가까지 크래시 없이 완주, `clf_acc=0.9836`이
   baseline과 정확히 일치(=probe_m이 처음부터 올바르게 10으로 설정됐음을 확인 — §4.2
   버그를 이번엔 사전에 피함).
3. 5개 조합 본 학습(`run_pipeline.py 8 --to-step 9`, 순차 백그라운드) — 5/5 exit=0.

### 11.2 결과 — baseline / 방안2(a) / 방안2(b) / 방안1 4자 비교

> **2026-08-03 정정**: qfw+transformer/qfw+resnet_tab의 (a) 수치는 최초 작성 시
> `raw_cnn_pretrained_from`이 각자 자기 baseline이 아니라 qfw+mlp baseline의 분류기를
> 가리키는 버그(세 조합의 cnn_emb가 완전히 동일한 값이었음 — `analyze_raw_embeddings.py`
> 로 임베딩을 직접 비교하다 발각)가 있었다. 자기 baseline 분류기로 재학습해 아래 표를
> 정정했다(원래 값: qfw+transformer 0.9541, qfw+resnet_tab 0.9456 — 정정 후 둘 다 소폭
> 개선, 결론 방향은 안 바뀜). 상세 경위는 `docs/260801_RESULTS.md` 참고.

| 축/모델 | baseline R² | 방안2(a) frozen | 방안2(b) separate | **방안1 flat** |
|---|---:|---:|---:|---:|
| qfw+mlp | 0.9501 | 0.9350 (-0.0151) | 0.9321 (-0.0180) | **0.9502 (+0.0001)** |
| qfw+transformer | 0.9538 | 0.9555 (+0.0017) | 0.9532 (-0.0006) | **0.9545 (+0.0007)** |
| qfw+resnet_tab | 0.9464 | 0.9461 (-0.0003) | 0.9506 (+0.0042) | **0.9483 (+0.0019)** |
| vqslope+mlp | 0.9103 | 0.8655 (-0.0448) | 0.9160 (+0.0057) | 0.9095 (-0.0008) |
| q_abs+mlp | 0.8595 | 0.8572 (-0.0023) | 0.8714 (+0.0119) | **0.8696 (+0.0101)** |
| **평균 Δ** | | **-0.0122** | **+0.0006** | **+0.0024** |

(hard도 oracle과 거의 동일 — clf_acc가 5개 전부 baseline과 일치하는 0.9835~0.9836으로
나와 probe_m 문제 없음을 재확인했다.)

RMSE/MAPE 개선율(`(baseline-방안1)/baseline×100`, 양수=개선):

| 축/모델 | RMSE 개선율 | MAPE 개선율 |
|---|---:|---:|
| qfw+mlp | +0.12% | +1.97% |
| qfw+transformer | +0.76% | -0.74% |
| qfw+resnet_tab | +1.75% | +2.71% |
| vqslope+mlp | -0.44% | -1.04% |
| q_abs+mlp | +3.63% | -4.81% |
| **평균** | **+1.16%** | **-0.38%** |

파라미터 수(trainable): flat은 baseline+~49K(BatchNorm 192 + head_in 확대분)만
늘어 방안2(b)의 +75K보다 적게 늘면서도 성능은 더 낫다 — mlp 기준 baseline 239.6K →
flat 289.0K vs (b) 314.8K.

**결론**:
1. **방안1이 방안2보다 명확히 낫다** — 평균 Δ +0.0024로 baseline과 (b)보다도 좋고,
   (a)의 큰 손해(-0.0125)나 (a)의 vqslope 파국적 악화(-0.0448, MAPE -59%)에 해당하는
   사례가 방안1엔 전혀 없다. **§3.1/§4에서 "방안2가 과적합에 더 강할 것"이라 추정했던
   판단은 실측과 반대로 나왔다** — 이 문서에 정정 기록을 남긴다.
2. **왜 예상이 틀렸는지**: §3.1의 우려("방안1은 파라미터가 96차원 그대로 늘어나
   과적합 위험이 방안2보다 큼")는 raw를 **비정규화 상태로** concat한다고 가정한
   pseudocode 기준이었다. 실제 구현에서 BatchNorm1d로 raw 블록을 정규화하자 이
   문제가 상당 부분 해소된 것으로 보인다 — "raw를 압축 없이 쓰는 것" 자체보다 "raw를
   제대로 정규화하지 않는 것"이 진짜 위험 요인이었을 가능성이 크다.
3. **CNN 압축이 오히려 손해였다는 재해석**: 방안2(CNN)가 방안1(flatten)보다 나쁘다는
   건, "raw 정보를 압축해서 넣는 것"이 "압축 없이 넣는 것"보다 못하다는 뜻 —
   RawCNN이 회귀에 필요한 미세 신호를 압축 과정에서 버리고 있을 가능성을 시사한다
   (§8.2에서 이미 우려했던 "분류기용으로 최적화된 표현이 회귀엔 손해"라는 가설이,
   방안2(b)처럼 회귀 전용으로 새로 학습한 CNN에도 — 정도는 약하지만 — 적용되는
   것으로 보인다: (b)도 방안1보다는 약간 못하다).
4. **q_abs+mlp에서 가장 큰 개선**(+0.0101, RMSE -3.63%) — HI 설계가 상대적으로
   덜 성숙한 축에서 raw 정보의 한계효용이 가장 크다는 §5 대목(HI 결론)과 일치한다.
5. vqslope+mlp만 유일하게 근소하게 baseline보다 낮다(-0.0008) — 노이즈 범위로
   보인다(다른 4개 조합은 전부 baseline 이상).

### 11.3 셀·시나리오별 SOH 예측 곡선 비교 (baseline/frozen(a)/separate(b) 3-way)

방안2(a)가 실제로 baseline보다 예측 곡선이 얼마나 더 튀는지 시각적으로 확인하기 위해
`visualize_results.py --rep-cells`(§11.1에서 신설)로 5개 조합 각각 baseline/frozen(a)/
separate(b) 3-way를 같은 셀(`b1c0`) 기준으로 그렸다 —
`_5_data_model_scr/comparison/cellcmp_{qfw_mlp,qfw_transformer,qfw_resnet_tab,
vqslope_mlp,qabs_mlp}/capacity_curve_compare_b1c0.png`(시나리오 6개×run 3개 그리드).
`cellcmp_qfw_mlp` 기준: frozen_a(가운데 열)가 baseline/separated_b보다 예측 곡선의
진동 폭이 chg_mid/chg_hi/dis_mid에서 눈에 띄게 넓다 — §11.2의 수치 결론(방안2(a)가
가장 나쁨)과 시각적으로 일치.

### 11.4 실험 산출물

| 그룹 | run_dir | 폴더 |
|---|---|---|
| 방안1(flat) 5개 | `0801_1524/1543/1612_p2_*_qfw_*`, `0801_1643_p2_mlp_vqs_dva`, `0801_1701_p2_mlp_qabs_20-50%` | `_5_data_model_scr/comparison/raw_flat_result_comparison/` |
| 방안2(a) 5개 | §4.2(260731_RESULTS.md)의 재학습분(`0731_1039/1152/1301`, `0731_1349`, `0731_0049`) | `_5_data_model_scr/comparison/frozen_after_result_comparison/` |
| 방안2(b) 5개 | §4.2 재학습분(`0731_1103/1216/1320`, `0731_1403`, `0731_0339`) | `_5_data_model_scr/comparison/cnn_separated_result_comparison/` |
| baseline 5개 | `docs/260728_RESULTS.md`/`260729_RESULTS.md` | `_5_data_model_scr/comparison/frozen_base_result_comparison/` |
| 3-way 셀 비교 | 위 표 | `_5_data_model_scr/comparison/cellcmp_*/` |

신규 yaml 5개: `5_model/config/exp_{qfw_mlp,qfw_transformer,qfw_resnet_tab,vqslope_mlp,
qabs_mlp}_flat.yaml`.

**실행 명령**(baseline의 Phase 1 게이트 재사용, Step 8→9 연속):

```bash
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qfw_mlp_flat.yaml \
    --gates-from "_5_data_model_scr/0723_1633_p2_mlp_qfw_35%_20%"
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qfw_transformer_flat.yaml \
    --gates-from "_5_data_model_scr/0723_2356_p2_tr_qfw_35%_20%"
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qfw_resnet_tab_flat.yaml \
    --gates-from "_5_data_model_scr/0724_0111_p2_res_qfw_35%_20%"
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_vqslope_mlp_flat.yaml \
    --gates-from "_5_data_model_scr/0725_1348_p2_mlp_vqs_dva"
python run_pipeline.py 8 --to-step 9 --model-config 5_model/config/exp_qabs_mlp_flat.yaml \
    --gates-from "_5_data_model_scr/0729_1908_p2_mlp_qabs_20-50%"
```

### 11.5 남은 과제

- **§4의 추천을 방안1로 뒤집을지 결정 필요** — 이 실측 결과만 보면 방안1이 우세하나,
  §3.3(리뷰어 설득력)의 "이 프로젝트는 raw→HI 압축→L0 희소화라는 서사를 일관되게
  따라왔는데 방안1은 그 축과 이질적"이라는 우려는 여전히 유효하다. 성능은 방안1이
  낫고 서사 일관성은 방안2가 낫다는 상충을 어떻게 정리할지는 사용자 판단이 필요하다.
- avg_cost 해석 시 §4.2(260731_RESULTS.md §3)에서 지적한 것과 같은 주의가 방안1에도
  적용된다 — `avg_cost`는 raw 처리 자체의 비용(BatchNorm 통과 등, 매우 저렴하긴 함)을
  반영하지 않는다.
- i_transformer/ft_transformer 지원은 방안1/방안2 둘 다 미구현 상태로 남아있다.
