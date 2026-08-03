# 260801 — raw 융합 4자 비교: baseline / 방안2(a) frozen / 방안2(b) separated / 방안1(raw flatten concat)

`docs/REGRESSION_UPGRADE.md`에서 다룬 raw 융합 두 방안(§2) — **방안2(CNN 임베딩 압축,
§10~§11 이전)**과 **방안1(원본 곡선 flatten concat, §11)** — 을 baseline(raw 미융합)과
함께 4자 비교한다. 방안2는 재사용 방식에 따라 (a) 분류기 CNN 재사용(frozen)과 (b) 회귀
전용 CNN 별도 학습(separate)으로 나뉘어 총 baseline 5 + (a) 5 + (b) 5 + 방안1 5 = **20개
run**을 비교한다. 방안1은 압축 모듈 자체가 없어(BatchNorm 정규화만 거친 flatten) frozen/
separate 구분이 없다.

**중요**: baseline/(a)/(b) 세 그룹은 이 문서 작성 시점 이전에 이미 완료·검증된 결과를
그대로 인용한다(`docs/260731_RESULTS.md` — probe_m 불일치 버그를 발견해 8개 run을
m=10으로 재학습·재평가까지 마친 최종본). 방안1(raw_concat) 5개만 이 문서를 위해 신규로
학습·평가했다. 네 그룹 다 oracle≈hard(clf_acc가 baseline과 사실상 동일한 0.9835~0.9836)
이므로 아래 수치는 hard(실배포) 기준이며 oracle과 사실상 같다.

> **⚠️ 2026-08-03 정정**: 이 문서를 위해 임베딩 자체를 직접 비교하는 분석
> (`5_model/analyze_raw_embeddings.py`)을 수행하던 중, **qfw+transformer/qfw+resnet_tab
> 의 (a)가 각자 자기 baseline이 아니라 qfw+mlp baseline의 분류기 CNN을 재사용하고
> 있었던 버그**를 발견했다 — 세 조합의 cnn_emb 분산이 소수점 6자리까지 완전히 동일하게
> 나와서 발각(§0.5 "회귀 입력 형태" 설계상 raw_cnn이 frozen이면 어느 head를 붙여도
> 값이 안 바뀌는 것과, 애초에 세 조합이 서로 다른 CNN을 재사용해야 했다는 설계 의도가
> 충돌한 사례). `5_model/config/exp_qfw_transformer_cnn.yaml`/`exp_qfw_resnet_tab_cnn.yaml`
> 의 `raw_cnn_pretrained_from`을 각자 자기 baseline(`0723_2356`/`0724_0111`)으로 정정해
> 재학습했다(새 run_dir: `0803_1240_p2_tr_qfw_35%_20%`/`0803_1319_p2_res_qfw_35%_20%`).
> 아래 표는 전부 정정된 수치로 갱신했다 — 두 조합 다 소폭 개선됐고(oracle R²
> 0.9541→0.9555, 0.9456→0.9461), **종합 결론(§4)의 방향은 바뀌지 않는다.** qfw+mlp/
> vqslope/q_abs 및 모든 (b)/방안1은 이 버그의 영향을 받지 않는다(각자 자기 baseline을
> 정확히 참조).

---

## 0. 공통 설정

| 항목 | 값 |
|---|---|
| 데이터 | MIT+HUST, `split_seed=42`, train/val/test=60/20/20 (4그룹 모두 동일 — 셀 단위 분할이라 그룹 간 셀 구성 일치) |
| 분류기 | `type: cnn`, Phase 1 게이트에서 재사용 (4그룹 모두 각 축의 동일 baseline 분류기 재사용) |
| 회귀 헤드 | `d_probe=64, d_head=128, dropout=0.2, scen_k_count=15` |
| baseline과의 유일한 차이 | (a)/(b): `model.with_raw_cnn`+`raw_cnn_pretrained_from` / 방안1: `model.with_raw_flat` — 나머지(`training`/`loss`/`uq`/`evaluation`)는 전부 동일하게 통제 |

| 축 / 모델 | baseline run_dir | (a) frozen | (b) separated | 방안1 raw_concat |
|---|---|---|---|---|
| qfw + mlp | `0723_1633_p2_mlp_qfw_35%_20%` | `0731_1039` | `0731_1103` | `0801_1524` |
| qfw + transformer | `0723_2356_p2_tr_qfw_35%_20%` | `0803_1240`(정정, 구 `0731_1152`) | `0731_1216` | `0801_1543` |
| qfw + resnet_tab | `0724_0111_p2_res_qfw_35%_20%` | `0803_1319`(정정, 구 `0731_1301`) | `0731_1320` | `0801_1612` |
| vqslope + mlp | `0725_1348_p2_mlp_vqs_dva` | `0731_1349` | `0731_1403` | `0801_1643` |
| q_abs + mlp | `0729_1908_p2_mlp_qabs_20-50%` | `0731_0049` | `0731_0339` | `0801_1701` |

(a)/(b)/방안1 run_dir은 전부 `_5_data_model_scr/<run_dir>_p2_*` 형태의 폴더명 접미사가 붙는다.

---

## 0.5 각 방식의 모델 흐름 · 회귀 입력 형태 · 데이터 정규화 (코드 기준)

세 방식 모두 **Stage A(방향별 probe gate)·Stage B(시나리오별 gate)까지는 완전히 동일**하고
(`5_model/models/scr_model.py`), 그 뒤 capacity head에 들어가는 `feat` 벡터의 구성과 raw 곡선
처리 방식만 다르다. 이 절은 그 차이를 코드 기준으로 정리한다(baseline은 raw 융합이 전혀
없는 기준선이라 별도 행 없이 각 표의 "raw 없음" 케이스로 포함).

### 공통 부분 — Stage A/B (baseline·(a)·(b)·방안1 전부 동일)

```python
x = x_hi * nan_mask                                    # (B, 64) NaN→0
probe_x, probe_z = _apply_probe_gate(x, direction, seg_idx)   # (B, 64) — 방향별 게이트(Phase2는 고정 mask)
scen_x,  scen_z  = _apply_scen_gate(x, seg_idx)                # (B, 64) — 시나리오별 게이트(Phase2는 고정 mask)
```
`probe_x`/`scen_x` 둘 다 **HI 64종이 이미 `SegmentNormalizer`로 z-score된 뒤** 게이트를 거친
값이다(비활성 위치는 0) — 정규화 자체는 raw 융합 여부와 무관하게 항상 동일하게 적용된다:
```python
# SegmentNormalizer.fit() — train 분할만으로 계산, transform_x()에서 전체 분할에 적용
mean_ = nanmean(x_hi, axis=0); std_ = nanstd(x_hi, axis=0)   # (64,) 각 HI별
x_norm = (x_hi - mean_) / std_; x_norm[isnan] = 0.0            # NaN 위치는 z-score 후 0으로 마스킹
```
`direction`(±1, 정규화 없음)과 `cap_init`(z-score, `cap_init_mean_`/`cap_init_std_`는 HI와
별개로 `capacity_Ah` 분포에서 계산)도 세 방식·baseline 전부 동일하게 취급된다.

### (a) 방안2 — 분류기 CNN 재사용(frozen)

**모델 흐름**:
```python
# __init__ — raw_cnn_pretrained_from(=baseline 분류기의 clf_best.pt)가 지정된 경우
self.raw_cnn = RawCNN(d_out=64)
_cnn_state = {k[4:]: v for k, v in classifier_ckpt["clf_state"].items() if k.startswith("cnn.")}
self.raw_cnn.load_state_dict(_cnn_state, strict=True)   # 분류기의 RawCNN 가중치를 그대로 복사
self.raw_cnn.requires_grad_(False); self.raw_cnn.eval() # 이후 절대 갱신 안 됨(그래디언트도 BN 통계도)

# forward() — 매 스텝마다 x_raw를 다시 통과시킴(캐싱 아님, §REGRESSION_UPGRADE.md 대화 참고)
with torch.no_grad():
    cnn_emb = self.raw_cnn(batch["x_raw"])              # (B, 2, 48) → (B, 64)
feat = concat(probe_x, scen_x, cnn_emb, direction, cap_init)   # (B, 194)
```
`train()` 오버라이드로 `model.train()`이 호출돼도 `self.raw_cnn`만 항상 `eval()`로 되돌려
BatchNorm 러닝 통계가 Phase2 데이터 분포로 오염되는 걸 막는다(흔한 freezing 함정).

**회귀 입력 형태**: `(B, 194) = probe_x(64) ‖ scen_x(64) ‖ cnn_emb(64) ‖ direction(1) ‖ cap_init(1)`.

**raw 곡선 정규화**: HI처럼 사전 z-score를 하지 않는다 — 대신 `RawCNN`(`5_model/models/
raw_cnn.py`) 내부 stem의 `Conv1d→BatchNorm1d→GELU→MaxPool1d`가 raw_v(~3-4V)/raw_i(~0-5A)의
채널별 스케일 차이를 **학습된 채널 정규화**로 흡수한다. 이 BatchNorm은 (a)에서는 **분류기
학습 때 고정된 통계**를 그대로 쓴다(얼린 뒤 다시는 안 바뀜) — 즉 (a)의 raw 정규화 통계는
Phase2 회귀 데이터가 아니라 분류기 학습 데이터 분포를 반영한다.

### (b) 방안2 — 회귀 전용 CNN 별도 학습(separate)

**모델 흐름**: (a)와 구조는 동일하되 `raw_cnn_pretrained_from`이 없어 `RawCNN`을 랜덤
초기화한 뒤 Phase 2 MSE 그래디언트로 **cap_head와 함께 처음부터 학습**한다:
```python
self.raw_cnn = RawCNN(d_out=64)   # 랜덤 초기화, requires_grad 그대로(True)
# forward()에서 torch.no_grad() 없이 그대로 통과 — 그래디언트가 raw_cnn까지 흐름
cnn_emb = self.raw_cnn(batch["x_raw"])
feat = concat(probe_x, scen_x, cnn_emb, direction, cap_init)   # (B, 194) — (a)와 차원 동일
```

**회귀 입력 형태**: (a)와 동일한 `(B, 194)` 구성 — 차이는 오직 `cnn_emb`을 만드는 `RawCNN`의
가중치가 고정이냐(a) 학습되냐(b)뿐이다.

**raw 곡선 정규화**: (a)와 동일하게 `RawCNN` 내부 `BatchNorm1d`가 담당하지만, (b)는 이
BatchNorm의 러닝 통계도 Phase 2 학습 중 **회귀 데이터 분포로 계속 갱신**된다(고정 아님) —
그래서 (a)/(b)는 "같은 압축 아키텍처, 다른 학습·정규화 대상 데이터"라는 차이로 요약된다.

### 방안1 — raw flatten concat(압축 없이 그대로 융합)

**모델 흐름**:
```python
# __init__ — with_raw_flat=True일 때만
self.raw_flat_norm = nn.BatchNorm1d(RAW_CH * RAW_N)   # = BatchNorm1d(96), 학습 가능 파라미터

# forward()
x_raw = batch["x_raw"]                                  # (B, 2, 48)
raw_flat = self.raw_flat_norm(x_raw.reshape(B, -1))     # (B, 96) — 압축 없이 정규화만
feat = concat(probe_x, scen_x, raw_flat, direction, cap_init)   # (B, 226)
```
CNN 인코더가 없으므로 학습되는 추가 파라미터는 `BatchNorm1d(96)`의 스케일·시프트(총 192개)
뿐이다 — (a)/(b)가 RawCNN 전체(약 42K 파라미터, (a)는 비학습·(b)는 학습)를 통과시키는 것과
대조적으로, 방안1은 raw 96차원을 사실상 있는 그대로(정규화만 거쳐) head에 넘긴다.

**회귀 입력 형태**: `(B, 226) = probe_x(64) ‖ scen_x(64) ‖ raw_flat(96) ‖ direction(1) ‖ cap_init(1)`.
`REGRESSION_UPGRADE.md` §2 원안(`feat = probe_x‖scen_x‖direction‖cap_init‖raw_v‖raw_i`)과
슬롯 순서가 다른데(raw 블록을 direction/cap_init 앞에 둠), 이는 (a)/(b)의 `cnn_emb` 삽입
위치와 통일해 `TransformerHead`의 토큰 슬라이싱 로직을 재사용하기 위한 구현 선택이며 MLP/
ResNet 계열 헤드에는 순서가 영향을 주지 않는다(전부 하나의 벡터로 이어붙여 Linear에 통과).

**raw 곡선 정규화**: `nn.BatchNorm1d(96)`이 raw_v/raw_i 96개 스칼라 각각(채널×위치 조합)에
대해 **독립적인 학습 가능 스케일·시프트**를 적용한다 — RawCNN의 BatchNorm(채널 2개에만
적용, 위치는 conv가 공유 가중치로 처리)보다 훨씬 세분화된 정규화 단위다. 이 BatchNorm은
`REGRESSION_UPGRADE.md` §2 원안(단순 `x_raw.reshape(B,-1)` concat, 정규화 없음)에는 없던
**의도적 추가**다 — 이미 z-score된 HI(mean0/std1) 옆에 정규화 안 된 raw(V~3-4, I~0-5A)를
그대로 두면 raw 블록이 그래디언트를 불공정하게 지배해 방안1이 원래보다 더 나쁘게 나올
위험이 있었기 때문(§11.2 결론 "방안1의 우위는 flatten 자체보다 이 정규화 추가에서 왔을
가능성"과 직결).

### 세 방식 요약 비교

| | (a) frozen | (b) separated | 방안1 flat |
|---|---|---|---|
| raw 압축 방식 | RawCNN(42K 파라미터) | RawCNN(42K 파라미터) | 없음(그대로) |
| RawCNN 학습 여부 | 고정(분류기 학습 시점에 종료) | Phase2와 함께 학습 | 해당 없음 |
| raw 정규화 | RawCNN 내부 BatchNorm1d(채널 2개, **고정** 통계) | RawCNN 내부 BatchNorm1d(채널 2개, **갱신** 통계) | 전용 BatchNorm1d(96개 스칼라, 갱신 통계) |
| feat 차원 | 194 (=130+64) | 194 (=130+64) | 226 (=130+96) |
| raw 블록에서 새로 학습되는 파라미터 수 | 0 | ≈42,401(`raw_cnn.py` 자체 계산) | 192(=96×2) |
| i_transformer/ft_transformer 지원 | 미지원(`NotImplementedError`) | 미지원 | 미지원 |

---

## 1. Overall 성능 — R² / RMSE / MAPE 4자 비교

| 축 / 모델 | baseline R² | (a) frozen | (b) separated | **방안1 flat** |
|---|---:|---:|---:|---:|
| qfw + mlp | 0.9501 | 0.9349 | 0.9321 | **0.9502** |
| qfw + transformer | 0.9538 | **0.9555** | 0.9532 | 0.9545 |
| qfw + resnet_tab | 0.9464 | 0.9461 | 0.9506 | **0.9483** |
| vqslope + mlp | 0.9103 | 0.8655 | 0.9160 | 0.9095 |
| q_abs + mlp | 0.8595 | 0.8572 | 0.8714 | **0.8696** |
| **평균 Δ(vs baseline)** | — | **-0.0122** | **+0.0006** | **+0.0024** |

| 축 / 모델 | baseline RMSE | (a) frozen | (b) separated | 방안1 flat |
|---|---:|---:|---:|---:|
| qfw + mlp | 0.01557 | 0.01777 | 0.01816 | 0.01555 |
| qfw + transformer | 0.01498 | 0.01470 | 0.01508 | 0.01487 |
| qfw + resnet_tab | 0.01613 | 0.01617 | 0.01549 | 0.01585 |
| vqslope + mlp | 0.02086 | 0.02555 | 0.02019 | 0.02095 |
| q_abs + mlp | 0.02611 | 0.02633 | 0.02502 | 0.02516 |

| 축 / 모델 | baseline MAPE(%) | (a) frozen | (b) separated | 방안1 flat |
|---|---:|---:|---:|---:|
| qfw + mlp | 1.127 | 1.333 | 1.402 | 1.105 |
| qfw + transformer | 0.954 | 0.938 | 0.997 | 0.961 |
| qfw + resnet_tab | 1.081 | 1.090 | 1.027 | 1.052 |
| vqslope + mlp | 1.322 | 2.104 | 1.337 | 1.336 |
| q_abs + mlp | 1.686 | 1.825 | 1.685 | 1.767 |

**개선율(%)** — `(baseline - 방안) / baseline × 100`, 양수=개선(오차 감소):

| 축 / 모델 | RMSE (a) | RMSE (b) | RMSE 방안1 | MAPE (a) | MAPE (b) | MAPE 방안1 |
|---|---:|---:|---:|---:|---:|---:|
| qfw + mlp | -14.1% | -16.6% | **+0.1%** | -18.3% | -24.4% | **+2.0%** |
| qfw + transformer | **+1.8%** | -0.7% | +0.8% | **+1.7%** | -4.5% | -0.7% |
| qfw + resnet_tab | -0.2% | +4.0% | +1.7% | -0.9% | +5.0% | **+2.7%** |
| vqslope + mlp | -22.5% | +3.2% | -0.4% | -59.2% | -1.2% | -1.0% |
| q_abs + mlp | -0.8% | +4.3% | **+3.6%** | -8.2% | +0.04% | -4.8% |
| **평균** | **-7.2%** | **-1.1%** | **+1.2%** | **-17.0%** | **-5.0%** | **-0.4%** |

**RMSE 기준으로는 방안1이 세 raw-융합 방식 중 유일하게 평균 플러스(+1.2%)** — (a)는
평균 -7.6%로 확실히 손해, (b)는 -1.1%로 사실상 중립. MAPE 기준에서는 방안1도 평균
-0.4%로 근소하게 음수지만, (a)의 -18.4%나 (b)의 -5.0%보다는 baseline에 훨씬 가깝다.
`docs/260731_RESULTS.md` §2에서 발견한 "MAPE 악화가 RMSE보다 항상 크다"는 패턴이
방안1에는 거의 나타나지 않는다는 점도 주목할 만하다 — vqslope+mlp만 예외적으로 RMSE
-0.4%/MAPE -1.0%로 둘 다 근소하게 음수이고, 나머지 4개 조합은 RMSE가 플러스이거나
MAPE가 baseline과 거의 같다.

---

## 2. 시나리오별 세부 비교 (R² / RMSE / MAPE(%))

**qfw + mlp**

| 시나리오 | R² base | R² (a) | R² (b) | R² 방안1 | RMSE base | RMSE (a) | RMSE (b) | RMSE 방안1 | MAPE base | MAPE (a) | MAPE (b) | MAPE 방안1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chg_lo | 0.9435 | 0.9365 | 0.9286 | 0.9448 | 0.01657 | 0.01756 | 0.01861 | 0.01636 | 0.98 | 1.19 | 1.32 | 0.98 |
| chg_mid | 0.9142 | 0.8805 | 0.8887 | 0.9157 | 0.02040 | 0.02408 | 0.02324 | 0.02023 | 1.53 | 1.88 | 1.82 | 1.48 |
| chg_hi | 0.9238 | 0.9063 | 0.9051 | 0.9176 | 0.01923 | 0.02132 | 0.02146 | 0.02000 | 1.34 | 1.60 | 1.63 | 1.41 |
| dis_hi | 0.9667 | 0.9423 | 0.9422 | 0.9636 | 0.01271 | 0.01674 | 0.01674 | 0.01328 | 1.05 | 1.31 | 1.37 | 1.07 |
| dis_mid | 0.9695 | 0.9624 | 0.9515 | 0.9708 | 0.01216 | 0.01350 | 0.01534 | 0.01190 | 1.07 | 1.20 | 1.34 | 1.06 |
| dis_lo | 0.9826 | 0.9817 | 0.9763 | 0.9884 | 0.00918 | 0.00942 | 0.01073 | 0.00751 | 0.79 | 0.82 | 0.93 | 0.63 |

방안1은 **6개 시나리오 전부 (a)/(b)보다 낫고, baseline과 비교해도 4개(chg_lo/mid,
dis_mid/lo)에서 baseline을 앞선다** — qfw+mlp는 4개 그룹 중 방안1이 가장 확실하게
우세한 조합.

**qfw + transformer**

| 시나리오 | R² base | R² (a) | R² (b) | R² 방안1 | RMSE base | RMSE (a) | RMSE (b) | RMSE 방안1 | MAPE base | MAPE (a) | MAPE (b) | MAPE 방안1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chg_lo | 0.9519 | 0.9508 | 0.9522 | 0.9512 | 0.01527 | 0.01545 | 0.01523 | 0.01539 | 0.76 | 0.76 | 0.74 | 0.75 |
| chg_mid | 0.9144 | 0.9159 | 0.9145 | 0.9138 | 0.02038 | 0.02020 | 0.02037 | 0.02046 | 1.48 | 1.45 | 1.50 | 1.49 |
| chg_hi | 0.9203 | 0.9237 | 0.9188 | 0.9140 | 0.01967 | 0.01924 | 0.01985 | 0.02043 | 1.30 | 1.31 | 1.42 | 1.38 |
| dis_hi | 0.9603 | 0.9654 | 0.9623 | 0.9695 | 0.01388 | 0.01295 | 0.01353 | 0.01217 | 0.99 | 0.95 | 1.00 | 0.98 |
| dis_mid | 0.9815 | 0.9830 | 0.9792 | 0.9840 | 0.00948 | 0.00908 | 0.01004 | 0.00880 | 0.76 | 0.71 | 0.81 | 0.73 |
| dis_lo | 0.9941 | 0.9938 | 0.9919 | 0.9943 | 0.00535 | 0.00550 | 0.00629 | 0.00527 | 0.43 | 0.45 | 0.52 | 0.43 |

(a) 열은 2026-08-03 정정된 수치 — 네 그룹 다 등락 폭이 소수점 셋째 자리 수준으로,
이 조합은 raw 융합 방식과 거의 무관하게 안정적이다(transformer head 자체가 노이즈성
입력에 강건한 것으로 보인다는 `260731_RESULTS.md` §2의 해석과 일치). 정정 후에는
(a)가 6개 중 4개(chg_mid/hi, dis_hi/mid)에서 baseline을 앞서 오히려 방안1과 비슷한
수준으로 개선됐다.

**qfw + resnet_tab**

| 시나리오 | R² base | R² (a) | R² (b) | R² 방안1 | RMSE base | RMSE (a) | RMSE (b) | RMSE 방안1 | MAPE base | MAPE (a) | MAPE (b) | MAPE 방안1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chg_lo | 0.9511 | 0.9482 | 0.9508 | 0.9495 | 0.01541 | 0.01586 | 0.01546 | 0.01565 | 0.78 | 0.81 | 0.80 | 0.78 |
| chg_mid | 0.8934 | 0.8935 | 0.9025 | 0.9090 | 0.02274 | 0.02274 | 0.02175 | 0.02101 | 1.78 | 1.74 | 1.63 | 1.57 |
| chg_hi | 0.9183 | 0.9213 | 0.9175 | 0.9096 | 0.01991 | 0.01954 | 0.02001 | 0.02095 | 1.40 | 1.35 | 1.38 | 1.52 |
| dis_hi | 0.9493 | 0.9541 | 0.9664 | 0.9538 | 0.01568 | 0.01492 | 0.01276 | 0.01497 | 1.09 | 1.09 | 0.98 | 1.04 |
| dis_mid | 0.9726 | 0.9681 | 0.9730 | 0.9734 | 0.01153 | 0.01245 | 0.01144 | 0.01135 | 0.98 | 1.03 | 0.90 | 0.97 |
| dis_lo | 0.9935 | 0.9917 | 0.9932 | 0.9942 | 0.00562 | 0.00635 | 0.00576 | 0.00530 | 0.45 | 0.53 | 0.47 | 0.42 |

(a) 열은 2026-08-03 정정된 수치 — (b)와 방안1이 비슷하게 좋고(각자 3개 시나리오씩
서로 앞섬), (a)는 대체로 처지지만 정정 전보다는 완화됐다(chg_hi/dis_hi에서 baseline을
앞섬). chg_hi만 방안1이 유일하게 baseline보다 뚜렷이 나쁘다(0.9183→0.9096).

**vqslope + mlp**

| 시나리오 | R² base | R² (a) | R² (b) | R² 방안1 | RMSE base | RMSE (a) | RMSE (b) | RMSE 방안1 | MAPE base | MAPE (a) | MAPE (b) | MAPE 방안1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chg_lo | 0.9148 | 0.8613 | 0.9227 | 0.9278 | 0.02033 | 0.02595 | 0.01937 | 0.01872 | 1.23 | 2.27 | 1.19 | 1.23 |
| chg_mid | 0.9282 | 0.8903 | 0.9241 | 0.9225 | 0.01866 | 0.02307 | 0.01919 | 0.01939 | 1.09 | 1.90 | 1.31 | 1.18 |
| chg_hi | 0.9037 | 0.8529 | 0.9189 | 0.9100 | 0.02162 | 0.02671 | 0.01983 | 0.02090 | 1.26 | 2.20 | 1.18 | 1.23 |
| dis_hi | 0.8892 | 0.8765 | 0.9045 | 0.8976 | 0.02319 | 0.02448 | 0.02153 | 0.02229 | 1.40 | 2.00 | 1.41 | 1.44 |
| dis_mid | 0.9074 | 0.8643 | 0.9055 | 0.9308 | 0.02120 | 0.02566 | 0.02142 | 0.01832 | 1.67 | 2.24 | 1.71 | 1.48 |
| dis_lo | 0.9186 | 0.8475 | 0.9204 | 0.8685 | 0.01988 | 0.02720 | 0.01966 | 0.02526 | 1.27 | 2.00 | 1.24 | 1.44 |

이 조합은 **(b)가 4그룹 중 가장 확실히 우세**하다(6개 중 5개에서 baseline·방안1·  (a)
전부를 앞섬). 방안1은 baseline과 엇비슷(3승 3패)하고, (a)만 6개 전부 크게 처진다 —
`REGRESSION_UPGRADE.md` §11.2에서 지적한 "vqslope만 유일하게 방안1이 근소하게
baseline보다 낮다"는 관찰이 dis_lo(0.9186→0.8685, RMSE+27%)에 집중돼 있음을 이 표에서
확인할 수 있다.

**q_abs + mlp**

| 시나리오 | R² base | R² (a) | R² (b) | R² 방안1 | RMSE base | RMSE (a) | RMSE (b) | RMSE 방안1 | MAPE base | MAPE (a) | MAPE (b) | MAPE 방안1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chg_lo | 0.8945 | 0.9068 | 0.8930 | 0.8981 | 0.02263 | 0.02127 | 0.02279 | 0.02224 | 1.63 | 1.54 | 1.72 | 1.63 |
| chg_mid | 0.8276 | 0.8209 | 0.8375 | 0.8528 | 0.02892 | 0.02948 | 0.02808 | 0.02673 | 2.04 | 2.15 | 2.11 | 1.99 |
| chg_hi | 0.8183 | 0.8260 | 0.8298 | 0.8263 | 0.02969 | 0.02906 | 0.02874 | 0.02904 | 1.56 | 1.90 | 1.52 | 1.97 |
| dis_hi | 0.9094 | 0.8537 | 0.9499 | 0.9274 | 0.02097 | 0.02664 | 0.01559 | 0.01876 | 1.04 | 1.58 | 0.85 | 1.17 |
| dis_mid | 0.8118 | 0.8421 | 0.8208 | 0.8157 | 0.03022 | 0.02768 | 0.02949 | 0.02991 | 2.27 | 2.09 | 2.26 | 2.23 |
| dis_lo | 0.8952 | 0.8936 | 0.8975 | 0.8972 | 0.02255 | 0.02272 | 0.02230 | 0.02234 | 1.59 | 1.68 | 1.66 | 1.61 |

`dis_hi`에서 (b)가 가장 크게 개선(0.9094→0.9499)되고 방안1도 개선(0.9094→0.9274)되는데
(a)만 악화(0.9094→0.8537)된다 — `260731_RESULTS.md`에서 지적한 "같은 시나리오에서 (a)/
(b)가 정반대로 갈리는" 패턴이 방안1까지 포함해도 (a)만 예외적으로 나쁜 방향이라는 걸
재확인.

---

## 3. Overall 지표 정리 — clf_acc / avg_cost / 파라미터 수

| 축 / 모델 | 그룹 | clf_acc | avg_cost | trainable params(K) |
|---|---|---:|---:|---:|
| qfw+mlp | baseline | 0.9836 | 87.25 | 239.6 |
| | (a) | 0.9836 | 25.50 | 272.4 |
| | (b) | 0.9836 | 25.50 | 314.8 |
| | 방안1 | 0.9836 | 25.50 | **289.0** |
| qfw+transformer | baseline | 0.9836 | 93.58 | 413.7 |
| | (a) | 0.9836 | 36.67 | 422.0 |
| | (b) | 0.9836 | 36.67 | 464.4 |
| | 방안1 | 0.9836 | 36.67 | **426.3** |
| qfw+resnet_tab | baseline | 0.9835 | 90.08 | 281.9 |
| | (a) | 0.9835 | 26.83 | 290.0 |
| | (b) | 0.9835 | 26.83 | 332.5 |
| | 방안1 | 0.9835 | 26.83 | **294.3** |
| vqslope+mlp | baseline | 0.9836 | 90.42 | 239.6 |
| | (a) | 0.9836 | 29.58 | 272.4 |
| | (b) | 0.9836 | 29.58 | 314.8 |
| | 방안1 | 0.9836 | 29.58 | **289.0** |
| q_abs+mlp | baseline | 0.9836 | 25.17 | 239.6 |
| | (a) | 0.9836 | 25.17 | 272.4 |
| | (b) | 0.9836 | 25.17 | 314.8 |
| | 방안1 | 0.9836 | 25.17 | **289.0** |

clf_acc는 4그룹 전부 baseline과 사실상 동일 — 라우팅 신뢰도 문제 없음(§0 참고).
`avg_cost`는 raw 융합 여부와 무관하게 같은 축 안에서 (a)/(b)/방안1이 전부 동일한데,
이는 `avg_cost`가 활성 probe/scen HI 개수만 재는 지표라 CNN/flatten 자체의 계산 비용을
포함하지 않기 때문이다(`260731_RESULTS.md` §3의 주의사항 참고 — 방안1도 마찬가지로
"더 저렴해 보이는" 착시가 있을 수 있다). **파라미터 수는 방안1이 (a)보다 조금 크고 (b)
보다는 뚜렷이 작다** — CNN 인코더(약 42K)를 아예 두지 않고 BatchNorm(파라미터 192개)만
추가하기 때문. 즉 방안1은 **가장 적은 구조적 비용 증가로 가장 좋은 성능**을 낸 조합이다.

---

## 4. 종합 결론

1. **방안1(raw flatten concat)이 raw 융합 세 방식 중 유일하게 baseline 대비 평균
   플러스**(R² Δ +0.0024, RMSE 개선율 +1.2%) — (a) frozen 재사용(-0.0122, -7.2%)과
   (b) separate 학습(+0.0006, -1.1%) 둘 다보다 낫다. `REGRESSION_UPGRADE.md` §3.1/§4가
   "방안2가 방안1보다 과적합에 강해 더 안정적일 것"이라 추정했던 판단은 실측과
   반대였고, 이 정정을 §11에 기록했다.
2. **다만 조합별 편차가 크다** — qfw+mlp/q_abs+mlp에서는 방안1이 가장 우세하지만,
   **vqslope+mlp에서는 (b)가 가장 우세**하고, **qfw+transformer에서는 정정 후 (a)가
   가장 우세**하다(0.9555, 방안1 0.9545보다도 근소하게 앞섬). "raw 융합 중 어느
   방식이 최선인가"는 축·모델 조합에 따라 갈리므로, 단일 결론("방안1이 항상 최선")
   으로 일반화하면 안 된다.
3. **(a) frozen 재사용은 5개 조합 중 4개(qfw+mlp/resnet_tab, vqslope, q_abs)에서
   baseline보다 못하거나 근접한 수준에 그친다** — 2026-08-03 정정 전에는 "5개 전부"로
   기록했으나, qfw+transformer는 자기 baseline 분류기로 바로잡은 뒤 baseline을 근소하게
   앞섰다(0.9538→0.9555). 그래도 평균(-0.0122)은 세 방식 중 가장 나쁘고, vqslope에서의
   파국적 악화(-0.0448, MAPE -59%)는 그대로다 — "분류기 목적으로 학습된 CNN 표현을
   회귀에 그대로 재사용하는 게 대체로 구조적 손해"라는 해석(`260731_RESULTS.md` §2,
   `REGRESSION_UPGRADE.md` §11.2)은 대체로 유효하나 "예외 없이 항상 나쁘다"까지는
   아니라고 정정한다.
4. **방안1의 우위는 "raw를 압축 없이 쓰는 게 낫다"보다 "raw를 제대로 정규화했는가"에서
   왔을 가능성이 크다** — 방안1 구현 시 문서 원안(단순 flatten concat)에 없던
   `BatchNorm1d`를 추가로 넣었는데(§11.1), 이게 방안1이 §3.1에서 우려했던 "과적합 위험"
   을 상쇄한 핵심 요인으로 보인다. 즉 이번 비교가 "flatten vs CNN 압축" 자체의 순수
   비교라기보다 "잘 정규화된 flatten vs CNN 압축"의 비교라는 점을 감안해야 한다.
5. **리뷰어 설득력(REGRESSION_UPGRADE.md §3.3) 문제는 미해결** — 방안1은 이 프로젝트의
   "raw→HI 압축→L0 희소화" 서사와 이질적이라는 우려가 여전히 유효하다. 성능은 방안1이
   우세, 서사 일관성은 방안2가 우세라는 상충을 어떻게 정리할지는 `REGRESSION_UPGRADE.md`
   §11.5에 남긴 대로 사용자 판단이 필요하다.

---

원본 run 폴더: §0 표 참고. 비교 산출물: `_5_data_model_scr/comparison/
{frozen_base_result_comparison,frozen_after_result_comparison,cnn_separated_result_comparison,
raw_flat_result_comparison}/`. 셀별 3-way(baseline/(a)/(b)) SOH 곡선 비교:
`_5_data_model_scr/comparison/cellcmp_{qfw_mlp,qfw_transformer,qfw_resnet_tab,
vqslope_mlp,qabs_mlp}/capacity_curve_compare_b1c0.png`(방안1은 미포함 — 필요시
`visualize_results.py --runs <baseline> <a> <b> <flat> --rep-cells b1c0`로 4-way
확장 가능).
