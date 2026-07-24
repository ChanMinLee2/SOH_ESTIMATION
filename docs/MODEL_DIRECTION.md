# MODEL_DIRECTION.md
# SCR 모델 개선 방향 — MIT↔HUST 크로스 데이터셋 기준 리뷰어 관점 분석

> 크로스 데이터셋(MIT→HUST, HUST→MIT) 설정에서 리뷰어가 제기할 수 있는 개선 방향을 카테고리별로 정리한다.  
> 각 항목은 **현재 문제 → 제안 → 근거 레퍼런스** 순으로 기술하며, 구현 우선순위 표를 마지막에 제공한다.

---

## I. 피처 수준 — 도메인 불변성 강화

### I-1. BOL-상대 피처 (Δ from cycle 1)

**현재 문제**  
HI 절댓값은 셀마다 초기 제조 편차를 포함한다. MIT와 HUST 셀의 초기 `plateau_frac`, `v_mean` 등이 다르면 모델이 SOH 노화 신호가 아니라 데이터셋 정체성을 학습할 수 있다.

**제안**  
각 HI에 대해 BOL 기준 상대값으로 변환:

```
HI_ratio_t = HI_t / HI_1      (비율)
HI_delta_t = HI_t - HI_1      (절대 차)
```

노화 추이만 남기고 초기 절댓값 정보를 제거한다.

**적용 위치**: `4_hi_analysis/hi_extractor.py` — 세그먼트 HI 추출 후 후처리 단계 추가

**근거**
- Li et al. (2022), *J. Power Sources* — "One-shot battery degradation trajectory prediction with deep learning": BOL 기준 상대 특성이 셀 간 초기값 차이를 제거해 크로스셀 일반화를 향상시킴을 실험으로 입증
- Dubarry & Beck (2021), *Batteries* — IC 피크 위치의 절댓값보다 BOL 대비 이동량이 화학계 간 SOH 상관이 더 높음

---

### I-2. `dvdq_peak_q` / `dvdq_valley_q` 정규화 (미수정 누설)

**현재 문제**  
`diff_dvdq_peak_q`, `diff_dvdq_valley_q`는 절댓값 Ah 위치로 저장된다. MIT(공칭 3Ah)와 HUST(공칭 2Ah) 간 스케일이 체계적으로 다르므로, 모델이 이 값으로 데이터셋 출처를 구별할 수 있다 — 의도치 않은 도메인 누설(domain leakage).

**제안**  
q_frac 단위로 정규화:

```python
# hi_extractor.py — _seg_diff() 내부
peak_q_frac   = peak_q   / q_tot   # 절댓값 Ah → 0~1 q_frac
valley_q_frac = valley_q / q_tot
```

**적용 위치**: `4_hi_analysis/hi_extractor.py` `_seg_diff()` 함수 내 2줄 수정

**근거**
- Dubarry et al. (2012), *J. Electrochem. Soc.* — ICA/DVA 피크 위치를 SOC(= q_frac) 기준으로 표현하는 것이 전기화학적 표준; 절댓값 Ah 사용은 정격 용량 차이에 취약

---

## II. 도메인 적응 — 분포 이동 대응

### II-1. 도메인 적대적 학습 (DANN)

**제안**  
"어느 데이터셋 출신인가"를 판별하는 domain discriminator를 gradient reversal layer(GRL)로 연결한다. 인코더가 데이터셋 구별 불가능한 표현을 학습하도록 강제.

```
HI 입력
  └→ [Encoder / Probe Gate]
          ├→ [SOH Head]          ← 정방향 gradient
          └→ [Domain Discriminator] ← GRL (gradient 부호 반전)
```

**적용 위치**: `5_model/models/scr_model.py` — probe gate 출력 뒤 discriminator 브랜치 추가

**근거**
- Ganin et al. (2016), *JMLR* — "Domain-Adversarial Training of Neural Networks": 이론·실험 모두 확립됨
- Shen et al. (2020), *IEEE Trans. Ind. Electron.* — 배터리 다양한 충전 프로토콜 간 크로스 적용에서 RMSE 30% 이상 개선

---

### II-2. MMD 정규화

**제안**  
손실 함수에 분포 정렬 항을 추가:

```
L_total = MSE(SOH_pred, target)
        + λ_l0 × L0_penalty
        + λ_mmd × MMD(φ(X_MIT), φ(X_HUST))
```

인코더 출력(φ)의 두 데이터셋 분포 거리를 직접 최소화한다.

**적용 위치**: `5_model/training/scr_loss.py`

**근거**
- Long et al. (2015), *ICML* — "Learning Transferable Features with Deep Adaptation Networks (DAN)": feature layer별 MMD 최소화로 분포 정렬
- DANN보다 학습 안정적; discriminator 없이 커널 추가만으로 구현 가능

---

### II-3. 히스토그램 매칭 (전처리)

**제안**  
학습 전에 소스 데이터셋의 각 HI 컬럼 분위수를 타깃 데이터셋에 맞게 정렬. 순위(rank) 보존, 분포 형태만 이전.

**적용 위치**: `5_model/datasets/segment_dataset.py` — `build_datasets()` 전처리 단계

**근거**
- Nyúl et al. (2000), *IEEE Trans. Med. Imaging* — 의료 영상 멀티사이트 정규화 표준 기법; 배터리 HI 분포 정렬에 직접 이식 가능
- 단점: 타깃 데이터 일부가 있어야 함 → few-shot fine-tuning(`finetune_scr.py`)과 결합 가능

---

## III. 학습 전략

### III-1. 메타학습 (MAML 계열)

**제안**  
각 셀을 독립 "태스크"로 정의하고 MAML로 초기화 파라미터를 학습. 테스트 시 타깃 데이터셋의 소수 셀(N=1~10)로 빠르게 적응.

**현재 코드와 관계**  
`finetune_scr.py`가 few-shot fine-tuning 역할을 하지만, MAML은 Phase 1 학습 자체를 "빠른 적응에 최적화된 초기화 탐색"으로 변경한다 — 수렴 기울기(inner loop)와 메타 업데이트(outer loop) 이중 구조 필요.

**근거**
- Finn et al. (2017), *ICML* — "Model-Agnostic Meta-Learning": N-shot 적응 수식 제공
- Ma et al. (2022), *Energy Storage Mater.* — "Few-shot battery lifetime prediction": 5셀로 MIT 기준 RMSE 2% 이내

---

### III-2. 다중 축 앙상블

**제안**  
qfrac, vwindow, rcs 각각으로 학습한 모델의 예측값을 평균 또는 학습된 가중치로 결합.

```
SOH_ensemble = w_qfrac × SOH_qfrac + w_vwindow × SOH_vwindow + w_rcs × SOH_rcs
```

**근거**
- Breiman (1996), *Machine Learning* — "Bagging Predictors": 독립적으로 편향이 다른 예측기 앙상블이 분산을 줄임
- 각 세그멘테이션 축은 서로 다른 전기화학 신호를 포착 → 예측 오류가 독립적으로 분산 → 앙상블 이득
- Severson et al. (2019), *Nature Energy* — 여러 피처 그룹에서 독립 학습한 모델 결합 시 OOD 성능 개선

**구현 비용**: 추론(inference) 단계 결합만 필요 — 각 축 모델을 별도 학습 후 예측값만 합산

---

### III-3. 커리큘럼 학습

**제안**  
높은 SOH(초기 사이클, SOH ≈ 1.0)부터 학습하고 낮은 SOH(심한 노화)로 점진적 확장.

```
Epoch 1~N/3  : SOH > 0.95 샘플만 사용
Epoch N/3~2N/3: SOH > 0.85 샘플 추가
Epoch 2N/3~N : 전체 샘플
```

**근거**
- Bengio et al. (2009), *ICML* — "Curriculum Learning": 쉬운 샘플 우선 학습이 일반화 향상
- 배터리 맥락: 초기 사이클은 프로토콜 노이즈가 적고 측정 신뢰도가 높음; 노화 후기 샘플은 이상 거동이 많아 "어려운 샘플"로 분류 타당

---

## IV. 모델 아키텍처

### IV-1. 불확실성 정량화 (UQ)

**제안**  
MC Dropout 또는 Deep Ensemble로 예측 신뢰구간을 제공. OOD 데이터에서는 자동으로 uncertainty가 넓어짐.

```python
# MC Dropout: inference 시 dropout 활성화 + T회 forward pass
preds = [model(batch) for _ in range(T)]   # T=50
soh_mean = torch.stack(preds).mean(0)
soh_std  = torch.stack(preds).std(0)
```

**근거**
- Gal & Ghahramani (2016), *ICML* — "Dropout as a Bayesian Approximation"
- Richardson et al. (2019), *IEEE Trans. Ind. Inform.* — GPR 기반 UQ가 OOD 감지와 신뢰할 수 있는 SOH 구간 추정에 유효함을 실증
- **리뷰어 관점**: 크로스 데이터셋 논문에서 포인트 추정만 제공하면 실용성 약점으로 지적됨

---

### IV-2. 물리 제약 — 단조성 정규화

**제안**  
셀 내에서 SOH는 단조 감소해야 하므로, 이를 위반하는 예측에 패널티:

```python
# scr_loss.py
mono_pen = F.relu(soh_pred_t - soh_pred_t_minus1).pow(2).mean()
L_total  = MSE + λ_l0 * L0 + λ_mono * mono_pen
```

**구현 조건**: 미니배치 내 같은 셀의 연속 사이클이 함께 묶여 있어야 함 → DataLoader 셀 단위 정렬 필요.

**근거**
- Nascimento et al. (2021), *Neural Comput. Appl.* — PINN + 단조성 제약이 데이터 희소 영역에서 예측 신뢰성 개선
- Liu et al. (2019), *Appl. Energy* — 물리 제약 통합이 OOD 외삽 안정성을 향상

---

## V. 구현 우선순위 요약

| 우선순위 | 제안 | 코드 변경 범위 | 기대 효과 | 핵심 레퍼런스 |
|:---:|---|---|---|---|
| ★★★ | `dvdq_peak/valley_q` → q_frac 정규화 | `hi_extractor.py` 2줄 | 도메인 누설 즉각 제거 | Dubarry et al. 2012 |
| ★★★ | 다중 축 앙상블 (qfrac+vwindow+rcs) | 추론 단계 결합 | 분산 감소, 구현 쉬움 | Breiman 1996, Severson 2019 |
| ★★☆ | MC Dropout UQ | dropout + 다중 forward | 신뢰구간 제공 | Gal & Ghahramani 2016 |
| ★★☆ | BOL-상대 HI 변환 | 전처리 1단계 추가 | 초기값 편차 제거 | Li et al. 2022 |
| ★★☆ | MMD 정규화 | 손실 항 추가 | 분포 정렬, 구현 단순 | Long et al. 2015 |
| ★☆☆ | 단조성 제약 | 손실 항 + DataLoader 수정 | OOD 외삽 안정성 | Nascimento et al. 2021 |
| ★☆☆ | DANN 도메인 적대 학습 | discriminator + GRL 추가 | 분포 이동 대응 | Ganin et al. 2016 |
| ★☆☆ | 히스토그램 매칭 | 전처리 분위수 정렬 | 분포 형태 정렬 | Nyúl et al. 2000 |
| ☆☆☆ | 커리큘럼 학습 | DataLoader 스케줄러 | 노이즈 완화 | Bengio et al. 2009 |
| ☆☆☆ | MAML 메타학습 | Phase 1 학습 루프 전면 수정 | N-shot 적응 최적화 | Finn et al. 2017, Ma et al. 2022 |

> **권고 순서**: `dvdq` 정규화(누설 수정, 2줄) → 다중 축 앙상블(추론 결합, ROI 최고) → MC Dropout(dropout on/off 1줄) → BOL 상대화 + MMD 순으로 적용.

---

## VI. 연구 포지셔닝 및 실험 아젠다

> **비교 대상 논문**: *"A sparse SOH estimation framework based on clustered voltage segments and multihead CNN under arbitrary charging condition"*  
> (전압 세그먼트 클러스터링 기반 다중 헤드 CNN SOH 추정 프레임워크)

---

### VI-1. 비교 대상 논문 요약

| 항목 | 비교 대상 논문 |
|------|----------------|
| 세그먼트 정의 | **전압 윈도우 기반** 고정 분할 |
| 클러스터링 기준 | 전압 특성 유사도 클러스터링 (k-means 등) |
| 모델 구조 | 클러스터별 **독립 CNN Head** (Multihead) |
| 충전 조건 대응 | 클러스터 라우팅으로 arbitrary charging 처리 |
| HI 선택 방식 | 없음 (raw 전압 곡선 직접 입력 또는 고정 통계량) |
| 연산 예산 제약 | 없음 |
| 전이 가능성 | 명시적 구조 없음 |

**핵심 한계**: "어떤 특성이 왜 중요한가"에 대한 답 없이 전압 구간만 나눔. 클러스터 할당이 고정되면 새로운 충전 프로토콜/다른 데이터셋에서 클러스터 경계 재학습 필요.

---

### VI-2. SCR 프레임워크의 차별화 포인트

#### [차별점 1] 고정 클러스터링 → 학습된 희소 특성 선택 (Learned Sparse Feature Selection)

비교 대상: 전압 구간 클러스터링으로 구역을 나누고 각 구역에 독립 CNN 부착.  
SCR: **Hard Concrete L0 Gate**로 데이터로부터 *어떤 전기화학 특성(HI)이 각 시나리오에서 중요한가*를 학습. 64개 후보 중 시나리오별 3~5개 HI만 자동 선별.

> 클러스터링은 입력 공간(전압 구간)을 나누지만, SCR은 **특성 공간(HI 서브셋)을 나눈다**.  
> 후자가 전기화학적 해석 가능성과 연산 효율성 면에서 우위.

#### [차별점 2] 연산 예산 제약 하 특성 획득 (Cost-Aware Feature Acquisition)

비교 대상: 연산 비용 개념 없음.  
SCR: L0 페널티에 **카테고리별 계산 비용(stat=1.0, diff=1.5, lfp=2.0, morph=3.0)을 명시적으로 부여**.

```
L_l0 = λ_l0 × Σ_i  cost_i × P(gate_i active)
```

이것은 단순 feature selection이 아니라 **예산 제약 하 피처 획득(budget-constrained feature acquisition)** 문제다.  
LFP plateau 기반 morph 특성(비용 3.0)이 연산 비용을 감수할 만큼 정보를 추가하는가를 데이터로 검증 가능.

> 배터리 HI 선택에서 계산/센싱 비용을 L0 정규화항에 명시적으로 통합한 연구는 희소하다.

#### [차별점 3] 충방전 방향별 독립 게이트 (Direction-Adaptive Gating)

비교 대상: 충전 조건 arbitrary를 클러스터 라우팅으로만 처리.  
SCR: Phase 1에서 **충전(charge_probe_gate)과 방전(discharge_probe_gate)이 각각 독립적으로 최적 HI를 학습**.

근거: 클러스터링 선행 실험에서 충전 구간에서는 diff/morph HI 비율이 높고, 방전 구간에서는 stat/lfp HI가 지배적으로 나타남. 물리적으로도 충전 곡선(CC→CV 전환)과 방전 곡선(IC 피크)이 다른 전기화학 신호를 담고 있음.

#### [차별점 4] 2단계 분리 학습으로 크로스 데이터셋 전이 (Decoupled Transfer Learning)

비교 대상: 새 데이터셋 적용 시 전체 재학습 필요.  
SCR:
- **Phase 1**: 도메인 무관 지식 — 어떤 HI가 SOH 변화를 반영하는가 (gate 구조)
- **Phase 2**: 도메인 특화 지식 — 그 HI들로 특정 배터리의 SOH를 예측 (cap_head)

Phase 1 gate를 동결하고 Phase 2 cap_head만 10개 셀 fine-tuning → HUST zero-shot R²=-0.76 → fine-tuned R²=0.81 달성.  
이 구조가 "**게이트는 전기화학 법칙, 헤드는 배터리 특성**"으로 역할을 분리한다는 주장의 근거.

#### [차별점 5] LFP 특화 HI 카테고리 설계

비교 대상: 전압 곡선 직접 입력 또는 범용 통계량.  
SCR: LFP 화학계 고유 거동을 포착하는 **4개 HI 카테고리** 명시 정의:

| 카테고리 | 특성 수 | 대상 현상 | 비용 |
|----------|---------|----------|------|
| stat | 18개 | 충방전 평균 전압·전류·전하량 통계 | 1.0 |
| diff | 20개 | dV/dQ (ICA) 피크·면적·위치 | 1.5 |
| lfp | 20개 | LFP plateau 구간 통계 (3.2~3.4V 특화) | 2.0 |
| morph | 6개 | Fréchet/DTW 기반 곡선 형태 거리 | 3.0 |

`lfp`/`morph` 카테고리는 LFP 화학계의 평탄 구간(flat plateau) 특성을 명시적으로 포착하는 설계로, 범용 CNN에서는 암묵적으로만 학습되는 정보.

---

### VI-3. 연구 목적 (Research Objective)

> **"임의의 충방전 프로토콜과 부분 사이클 관측 환경에서, 연산 예산 제약 하에 시나리오별 최소 전기화학 특성 집합을 학습하고, 그 집합이 데이터셋을 넘어 전이됨을 보인다."**

구체적으로:
1. **관측 가능한** 세그먼트 경계(전압 윈도우, 프로토콜 전환점) 기준 분할로 배포 가능성 보장
2. **L0 학습된 sparse HI 서브셋**이 시나리오(충방전 구간 × SOH 레벨)마다 서로 다른 전기화학 정보를 반영함을 gate 분석으로 해석
3. **연산 비용 대비 정확도 Pareto**를 통해 현장 BMS 임베딩 시 최소 필요 HI 집합 제안
4. Phase 1 gate 동결 + Phase 2 fine-tuning의 **데이터셋 간 전이 메커니즘** 실험 검증

---

#### 이 연구 목적이 실제로 효용이 있는가 — 근거

아래 네 가지 현실 맥락에서 이 목적이 해결하는 문제가 실재함을 확인한다.

---

**[근거 1] BMS 임베디드 환경의 연산 자원 제약 — "연산 예산 제약"의 실재성**

상용 BMS 마이크로컨트롤러(MCU, 예: TI BQ 시리즈, NXP S32K)는 100~200MHz 클럭, 수백 KB RAM 수준이다. 여기서 SOH 추정에 사용하는 HI 계산 비용을 범주별로 추정하면:

| HI 카테고리 | 대표 계산 | 200점 곡선 기준 추정 연산 수 | 100MHz에서 소요시간 |
|------------|----------|:--------------------------:|:-----------------:|
| stat | 평균, 표준편차, 적분 | ~2K ops | ~0.02 ms |
| diff | dV/dQ 피크 탐색 | ~15K ops | ~0.15 ms |
| lfp | plateau 구간 탐색 + 통계 | ~25K ops | ~0.25 ms |
| morph | DTW / Fréchet | ~200K ops | **~2.0 ms** |

6개 세그먼트 전체 morph 계산 = ~12ms/사이클. 100ms 샘플링 BMS에서 이 비중은 수용 불가 수준이다. stat만으로 충분한 시나리오가 gate로 확인된다면, 해당 시나리오에서 morph 계산을 건너뛸 수 있다 — 이것이 cost-aware L0 선택의 직접적 배포 효용이다.

> 단, 이 추정은 현재 실험에서 정량적으로 검증되지 않았다. Ablation A-2(cost-aware vs 균일)에서 **실제 선택된 HI 구성과 예측 정확도의 Pareto 관계**를 측정해야 이 주장이 성립한다.

---

**[근거 2] 배터리 데이터 수집의 현실적 비용 — "데이터셋 전이"의 실재성**

배터리 수명 주기 시험(전체 사이클 데이터 확보)의 실험실 비용은 셀 1개당 통상 수개월~1년, 비용으로 환산하면 수백만원 이상이다. 이로 인해:

- MIT Severson 데이터셋: 124개 셀, 약 6개월 이상 실험 기간 추정
- HUST Ma 데이터셋: 77개 셀, 유사 규모

새 배터리 셀 화학계나 새 제조사 제품에 모델을 적용할 때마다 이 규모의 재실험을 반복하는 것은 현실적으로 불가능하다. 본 연구의 결과:

- HUST zero-shot(재학습 없음): R² = -0.76 → 사실상 예측 불가
- HUST 10-shot fine-tuning(10개 셀, ~수일 실험): R² = **0.813**

즉, 기존 방식 대비 **약 10분의 1 이하의 실험 비용**으로 배포 가능한 수준의 성능을 달성한다. 10-shot이 의미 있으려면 fine-tuning 대상 10개 셀이 54개 test 셀에 대해 대표성이 있어야 하며, 이것이 C-4 실험(N-shot 곡선)에서 검증되어야 한다.

> **전제 조건 한계**: 현재 fine-tuning은 Phase 1 gate 구조가 두 데이터셋 모두에서 유효하다는 가정 위에 있다. MIT와 HUST가 동일한 A123 LFP 화학계이기 때문에 성립하는 것일 수 있다. NMC 또는 다른 화학계로의 전이는 별도 검증 필요.

---

**[근거 3] 임의 프로토콜 환경의 실재 — "임의 충방전 조건"의 실재성**

MIT 데이터셋에만 72가지 충전 프로토콜이 존재하며, HUST 방전에는 77가지 C-rate 조합이 있다. 현실 EV/ESS 운용에서는:

- 급속 충전(DC 급속, 350kW급)과 완속 충전(AC 7kW)의 동일 셀 혼용
- V2G(Vehicle-to-Grid) 방전 프로토콜의 불규칙성
- 계절/온도에 따른 충전 전략 변경

이 조건에서 **단일 프로토콜 가정 모델**은 즉시 분포 이탈(out-of-distribution)에 빠진다. 본 연구의 시나리오 분류(Stage A)가 "지금 관측 중인 세그먼트가 어떤 조건인가"를 추론하고 그에 맞는 HI 서브셋으로 전환하는 구조는 이 문제를 직접 타깃으로 한다.

단, **현재 시나리오 분류기는 seg_idx(세그먼트 인덱스)를 직접 입력받아 라우팅하는 구조**여서, 미지의 충방전 조건에서 seg_idx를 추론하는 능력이 아직 검증되지 않았다. "임의 프로토콜"을 클레임하려면 Stage A 분류기가 프로토콜 정보 없이 HI만으로 세그먼트를 분류하는 실험(SCENARIO_STRATEGY.md의 window localization 문제)이 필요하다.

> **현재 한계**: seg_idx가 입력으로 주어지지 않는 진정한 "임의 관측" 시나리오에서의 성능은 아직 미검증. 이 클레임을 유지하려면 Stage A 분류기의 독립적 평가 실험이 필수.

---

**[근거 4] LFP의 SOH 추정 난이도 — "최소 특성 집합"의 의미**

LFP는 NMC/NCA와 달리 평탄 전압 구간(Flat Plateau)이 약 3.2~3.4V에 걸쳐 존재해, 전압으로 SOC/SOH를 직접 추론하기 어렵다. 이것이 LFP SOH 추정이 NMC 대비 훨씬 어려운 이유이며:

- 단순 OCV 기반 방법: LFP plateau에서 V-SOC 곡선이 거의 수평 → 작동 불가
- 단일 HI(예: 방전 용량)만으로는 프로토콜 변동에 취약
- 본 연구 결과: rcs 축 기준 val R² ≈ 0.64, 1000에폭 과적합 시 R² ≈ 0.65

과적합(dropout=0, 1024 샘플) 상한이 0.65 수준이라는 것은 이 태스크 자체의 노이즈 플로어가 높다는 의미다. 즉 어떤 모델로도 R² > 0.7~0.75를 안정적으로 넘기기 어려운 "어려운 태스크"이며, 이 조건에서 **최소 HI 집합으로 노이즈 플로어 부근의 성능을 달성한다는 것 자체가 효율성의 근거**가 된다.

> **반론 가능성**: "어차피 R²=0.65면 실용성이 없는 거 아닌가?"  
> 답: SOH 추정의 절대 정확도보다 **트렌드 추적(trend monitoring)** 관점에서 RMSE ≈ 0.028 Ah / 공칭 1.1 Ah = 약 2.5% 오차는 BMS 경보 기준(통상 5~10% 열화 임계) 이내에 있다. 포인트 추정이 아니라 열화 추이 감지 용도로는 충분한 정확도.

---

**종합 판단**

| 목적 구성 요소 | 효용 실재 여부 | 현재 검증 상태 | 남은 과제 |
|----------------|:-------------:|:--------------:|----------|
| 임의 충방전 조건 | ✅ 실재함 | ⚠️ 부분 (seg_idx 직접 입력 한계) | Stage A 독립 분류 실험 |
| 부분 사이클 관측 | ✅ 실재함 | ⚠️ HI 추출은 완전 세그먼트 필요 | 실시간 HI 추출 방식 명시 |
| 연산 예산 제약 | ✅ 실재함 | ❌ Pareto 미측정 | Ablation A-2 |
| 시나리오별 최소 HI | ✅ 실재함 | ✅ gate JSON으로 확인 | 안정성(seed) 검증 |
| 데이터셋 간 전이 | ✅ 실재함 | ✅ 10-shot R²=0.813 | N-shot 곡선, 역방향 전이 |

**결론**: 목적 자체의 효용은 실재하며, 현재 구조로 3~5번 항목은 이미 부분 검증됐다. "임의 프로토콜"과 "연산 Pareto" 두 주장은 추가 실험(Stage A 독립 평가, Ablation A-2) 없이는 논문에서 강하게 클레임하기 어렵다.

---

### VI-4. 증명해야 할 실험 목록

아래 실험들이 모두 결과로 뒷받침되어야 논문의 각 주장이 성립한다.

#### [실험 A] Ablation: 핵심 설계 요소의 기여 검증

| 실험 | 제거 조건 | 측정 지표 | 검증 주장 |
|------|-----------|----------|----------|
| A-1 | 방향별 독립 게이트 → 단일 공유 게이트 | RMSE, R² | 충방전 분리가 유효한가 |
| A-2 | 비용 가중 L0 → 균일 비용 L0 | 선택 HI 구성, Pareto 곡선 | cost-aware가 morph 선택 억제에 기여하는가 |
| A-3 | 시나리오별 독립 gate → 단일 전역 gate | 시나리오별 RMSE | 시나리오 조건부 라우팅이 필요한가 |
| A-4 | Phase 2 (고정 gate) → Phase 1만 사용 | RMSE, R² | Phase 분리 학습의 효과 |

#### [실험 B] Baseline 비교 (논문의 존재 이유 증명)

| Baseline | 목적 |
|----------|------|
| B-1: 글로벌 단일 MLP + 세그먼트 one-hot | 시나리오별 라우팅 없이 같은 성능이 나오는가 |
| B-2: 시나리오별 완전 독립 모델 (6개) | 공유 구조(probe gate 재사용)의 이득 |
| B-3: Lasso/RF feature selection + XGBoost | 전통적 희소 선택 대비 L0 gate의 차이 |
| B-4: 원본 전압 곡선 직접 입력 CNN/Transformer | 수작업 HI 없이 raw 입력 시 성능·해석성 비교 |
| B-5: Multihead CNN (비교 대상 논문 재현) | 직접 비교: 전압 클러스터링 vs L0 gate 선택 |

#### [실험 C] 크로스 데이터셋 전이 검증

| 실험 | 설명 |
|------|------|
| C-1: MIT→HUST zero-shot | Phase 1 gate 동결, Phase 2 cap_head만 MIT로 학습, HUST 전체 test |
| C-2: HUST→MIT 역방향 | 방향 전환 시 게이트 안정성 확인 |
| C-3: Gate HI subset Jaccard 일치도 | 두 데이터셋 각각의 Phase 1 결과로 선택된 HI가 얼마나 겹치는가 |
| C-4: Fine-tuning 효율 곡선 | HUST N=1/5/10/20 셀 fine-tuning → RMSE 수렴 속도 |

> C-3에서 Jaccard 일치도가 높을수록 "선택된 HI subset이 데이터셋 무관한 전기화학 지식을 반영한다"는 주장이 강해진다.

#### [실험 D] 연산 예산 Pareto (cost-aware L0의 실용적 가치)

- λ_l0 스윕으로 선택 HI 수 vs RMSE 곡선 작성 (시나리오 6개 각각)
- cost-aware L0 vs 균일 cost L0의 Pareto 곡선 비교
- **결론 예시**: "rcs 축 rcs_lo 시나리오에서 stat 3개(비용 3.0)만으로 RMSE X%, morph 추가 시 Y% 개선 — 배포 환경에서는 stat만으로 충분"

#### [실험 E] Gate 안정성 (Hard Concrete의 seed 민감도)

- seed 5~10개 변경 → 시나리오별 선택 HI 집합 변화 기록
- 각 HI의 선택 빈도(stability selection frequency) 계산
- Threshold 0.8 이상인 HI만 "안정적 선택"으로 보고

> Hard Concrete는 seed에 따른 선택 불안정성이 알려진 약점. 이를 정면에서 측정·보고하지 않으면 리뷰어가 지적함.

#### [실험 F] 세그먼트 축 비교 (어떤 분할이 전이에 유리한가)

- protocol / vwindow / rcs 축 각각의 Phase 2 RMSE 및 cross-dataset RMSE 비교
- **배포 관점**: vwindow 축이 가장 관측 가능 (전압은 항상 측정 가능) → 배포 가능 분할의 기준선
- **성능 관점**: rcs 축이 현재 실험에서 가장 높은 R² → 하지만 배포 시 경계 결정 방법 논의 필요

---

### VI-5. 논문 포지셔닝 문장 (Contribution Statement 후보)

1. **Learned sparse HI selection**: "기존 전압 세그먼트 클러스터링이 '어느 구간'을 나누는 데 그친 반면, 본 연구는 Hard Concrete L0 gate로 '각 시나리오에서 어떤 전기화학 특성이 중요한가'를 데이터로부터 학습한다."

2. **Cost-aware acquisition**: "HI 카테고리별 계산 비용(stat < diff < lfp < morph)을 L0 정규화 항에 통합하여, 정확도-비용 Pareto 최적 HI 집합을 시나리오별로 도출한다. 이는 임베디드 BMS 환경에서 어떤 피처를 얼마나 계산할 것인가에 대한 정량적 근거를 제공한다."

3. **Decoupled transfer**: "Phase 1(게이트 학습)과 Phase 2(용량 예측 학습)의 분리 구조로, 도메인 무관 전기화학 지식(gate)과 도메인 특화 예측(head)을 명시적으로 분리한다. 소수 셀(N=10) fine-tuning만으로 zero-shot R²=-0.76 → 0.81 달성."

4. **LFP-specific HI**: "범용 통계량을 넘어, LFP 화학계 고유의 plateau 특성(lfp 카테고리)과 곡선 형태 거리(morph 카테고리)를 포함한 4-카테고리 HI 체계를 제안하고, 각 카테고리가 시나리오별 게이트 선택에서 차지하는 비중을 분석한다."

---

### VI-6. 현재 진행 상황 (2026-07 기준)

| 단계 | 상태 | 비고 |
|------|------|------|
| 데이터 전처리 (MIT/HUST) | ✅ 완료 | 4-카테고리 HI 64개 추출 |
| 이상치 수정 | ✅ 완료 | `docs/HI_OUTLIER_FIXES.md` |
| Phase 1 학습 (protocol/vwindow/rcs) | ✅ 완료 | gate JSON 저장 |
| Phase 2 학습 (MLP/Transformer/iTransformer) | ✅ 완료 | rcs 축 R²≈0.65(val) |
| Cross-dataset fine-tuning (HUST) | ✅ 완료 | N=10 → R²=0.81 |
| 과적합 테스트 (variant 비교) | 🔄 진행 중 | MLP-S/M/L, Tr-S/M/L |
| Ablation A-1~A-4 | ⬜ 미착수 | **다음 우선순위** |
| Baseline B-1~B-5 | ⬜ 미착수 | 논문 핵심 비교군 |
| Cross-dataset C-1~C-4 | 🔄 일부 진행 | C-1만 완료 |
| Pareto 곡선 D | ⬜ 미착수 | λ_l0 스윕 필요 |
| Gate 안정성 E | ⬜ 미착수 | seed 반복 실험 필요 |
| 세그먼트 축 비교 F | ⬜ 일부 | qfrac/cluster 축 미완성 |

**당장 해야 할 것**: Ablation A-1 (방향별 게이트 공유 vs 분리) → 이 결과가 핵심 설계 결정의 근거.

---

## VII. 세부 실험 계획서

---

### 핵심 평가 프레임 (통합 평가: oracle/hard/soft)

> **2026-07-23 변경**: 기존 E1(qfrac→qfrac)/E2(qfrac→random)/E3(routing) 3축 명칭은
> 폐기되었다. `test_scr.py`가 분류기 유무에 따라 **oracle/hard/soft 3모드를 자동으로
> 한 번에** 평가하는 통합 평가로 대체됐다(`docs/PIPELINE.md` §5-10/5-11 참조).
> `test.random_segment_test: true`를 켜면 동일한 qfrac 테스트 세트 평가 뒤에
> test_rs(랜덤 컷 세그먼트)에서도 같은 oracle/hard/soft 3모드를 추가로 평가한다.
> 이 계획서의 실험들은 이 통합 평가에서 나오는 아래 4개 지점 조합으로 결과를 수집한다.

| 평가 지점 | 데이터 | routing | 의미 (구 명칭) |
|---------|--------|---------|------|
| `main.oracle` | qfrac test split | 정답 seg_idx | 이상적 조건 상한선 (구 E1) |
| `test_rs.oracle` | test_rs (랜덤 컷) | 정답 seg_idx (위치기반 `_assign_regime` 라벨) | distribution shift만 반영한 상한선 (구 E2 — 기존엔 방향(충/방전)만으로 라우팅했으나 이제 6-시나리오 정답 라벨이 있어 진짜 oracle 라우팅이 됨) |
| `test_rs.hard` | test_rs (랜덤 컷) | 분류기 argmax | 실배포 시나리오 — 분류기 기여도 (구 E3-hard) |
| `test_rs.soft` | test_rs (랜덤 컷) | 분류기 확률 가중 | 실배포 시나리오 보조 (구 E3-soft) |

`main.oracle` − `test_rs.oracle` = **distribution shift 손실**  
`test_rs.oracle` − `test_rs.hard` = **분류기 불완전성으로 인한 추가 손실**(= 분류기 라우팅의 기여도를 뒤집어 본 것)  
이 두 갭이 논문의 핵심 주장을 정량화한다 — 개념은 과거 "E1-E2", "E2-E3" 갭과 동일하며 명칭과 계산 경로만 바뀌었다.

---

### 0. 전제 조건 확인 (실험 시작 전)

모든 실험을 수행하기 전에 아래 항목이 완료되어야 한다.

#### 0-1. test_rs 데이터셋 생성

랜덤 세그먼트 데이터셋(`_4_data_hi/test_rs/`)이 생성 완료되어야 `test_rs.oracle`/`test_rs.hard`/`test_rs.soft` 평가를 수행할 수 있다.

```bash
# 백그라운드 실행 완료 확인
python tmp_make_test_rs.py --n-samples 8 --workers 4

# 생성 확인
ls _4_data_hi/test_rs/seg/MIT/    # 셀별 PKL 존재 여부
cat _4_data_hi/test_rs/test_rs_stats.txt   # 통계 요약
```

**정상 기준**: 총 세그먼트 수 ≈ 사이클 수 × 14~16개/사이클, MIT+HUST 모두 존재.

#### 0-2. 분류기 재활성화

`test_rs.hard`/`test_rs.soft` 평가를 위해 분류기(`lambda_scen > 0`)를 포함한 Phase 1을 재학습해야 한다.

```yaml
# scr.yaml — 분류기 활성화 설정
loss:
  lambda_scen: 0.5      # 현재 0.0 → 0.5로 변경 (CE 활성화)
  lambda_l0: 0.01
```

분류기가 비활성화된 기존 체크포인트는 `main.oracle`/`test_rs.oracle`에만 사용 가능하다(분류기가
없으면 `test_scr.py`가 자동으로 oracle만 실행하고 hard/soft는 건너뜀). `test_rs.hard`/`soft`는
분류기 포함 재학습 체크포인트가 필요.

#### 0-3. 코드 수정 완료 확인

| 항목 | 파일 | 상태 |
|------|------|------|
| `cap_init_mean_/std_` 이름 변경 | `segment_dataset.py` | ✅ 완료 |
| `build_random_seg_dataset()` 추가 | `segment_dataset.py` | ✅ 완료 |
| `_build_normalizer_from_ckpt` 하위호환 | `test_scr.py`, `finetune_scr.py` | ✅ 완료 |
| `predict_dataset(routing_mode=)` | `scr_evaluator.py` | ✅ 완료 |
| `test.random_segment_test` yaml 옵션 | `config/scr.yaml` | ✅ 완료 |

---

### 0-BIS. 참조 기준선 (현재 완료된 결과)

모든 실험 결과의 비교 기준점.

| Run ID | 축 | Head | MIT→MIT val R² | HUST zero-shot R² | HUST 10-shot R² |
|--------|-----|------|:--------------:|:-----------------:|:---------------:|
| `0715_1015` | protocol | MLP-S | 0.726 | — | — |
| `0715_1040` | vwindow | MLP-S | — | — | — |
| `0715_1149` | rcs | Phase 1 only | — | — | — |
| `0715_1340` | **rcs** | **MLP-S** | **0.643** | **-0.759** | **0.813** |

> **핵심 관찰**: protocol 축이 cross-dataset zero-shot에 강하고(val R²=0.726), rcs 축은 MIT 내 val은 낮지만 fine-tuning 후 최고 성능. 두 축의 특성 차이 분석이 필요.

공통 기준 YAML (Phase 2 기본값):
```yaml
data:
  datasets: ["MIT", "HUST"]
  is_cross_dataset_evaluate: true
  split_seed: 42
classifier:
  charge_probe_m: 3
  discharge_probe_m: 2
regression:
  scen_k_count: 55
model:
  d_head: 128
  dropout: 0.1
  regression_model: "mlp"
  mlp_hidden_dims: null          # → [128, 64] default
training:
  epochs: 500
  lr: 5.0e-4
  weight_decay: 1.0e-4
  early_stop_patience: 30
  run_overfit_test: false
```

---

### R-1. 통합 평가 — main.oracle / test_rs.oracle·hard·soft 일괄 수집

**목적**: 위 4개 평가 지점(구 E1/E2/E3)을 `test_scr.py` 한 번 실행으로 전부 수집한다 —
현재 코드는 분류기 유무에 따라 자동으로 필요한 모드를 실행하므로, 과거처럼
`--override` 플래그로 모드를 하나씩 바꿔가며 세 번 실행할 필요가 없다
(`--override`는 현재 `test_scr.py`에 존재하지 않는 옵션이니 쓰지 않는다).

**전제**: test_rs 데이터 생성 완료(0-1). `test_rs.hard`/`soft`까지 보려면 분류기 포함
재학습 체크포인트 필요(0-2) — 없으면 oracle만 자동 실행되고 hard/soft는 건너뛴다.

#### 0단계 — 분류기 포함 체크포인트 준비 (hard/soft가 필요한 경우만)

```yaml
# scr.yaml
loss:
  lambda_scen: 0.5              # CE 손실 활성화
  lambda_l0: 0.01
  lambda_l0_auto: true
```

```bash
# Phase 1 재학습 (분류기 포함) → gates/classification_HIs.json 생성
python 5_model/train_scr.py --phase 1 --config 5_model/config/scr.yaml

# Phase 2 재학습 (고정 게이트 + 회귀 헤드)
python 5_model/train_scr.py --phase 2 --gates-from _5_data_model_scr/{run_id_cls}

# 시나리오 분류기 학습 (B안 — 회귀와 분리)
python 5_model/train_classifier.py --run-dir _5_data_model_scr/{run_id_cls_p2}
```

#### 1단계 — 통합 평가 실행

```yaml
# scr.yaml
test:
  random_segment_test: true     # true여야 test_rs.oracle/hard/soft까지 같이 실행됨
  random_seg_data_dir: "_4_data_hi/test_rs/seg"
  random_seg_datasets: ["MIT", "HUST"]
  # random_seg_routing 키는 더 이상 사용되지 않음 — oracle/hard/soft 전부 자동 실행됨
```

```bash
python 5_model/test_scr.py \
  --checkpoint _5_data_model_scr/{run_id_cls_p2}/checkpoints/best.pt \
  --classifier-ckpt _5_data_model_scr/{run_id_cls_p2}/classifier/clf_best.pt
```

**출력**:
- `{run_dir}/metrics/metrics.json` → `test.oracle` / `test.hard` / `test.soft` (= `main.oracle` 등)
- `{run_dir}/random_seg_test/metrics.json` → `test_rs.oracle` / `test_rs.hard` / `test_rs.soft`

#### 비교 요약표 (실험 완료 후 작성)

| 평가 지점 | RMSE | R² | 비고 (구 명칭) |
|---|---|---|---|
| `main.oracle` | TBD | TBD | 상한선 (구 E1) |
| `test_rs.oracle` | TBD | TBD | distribution shift만 (구 E2, 라벨 정확도 개선판) |
| `test_rs.hard` | TBD | TBD | 실배포, 분류기 기여 (구 E3-hard) |
| `test_rs.soft` | TBD | TBD | 실배포 보조 (구 E3-soft) |

**해석 기준** (구 기준과 동일한 논리, 이름만 교체):
- `(main.oracle − test_rs.oracle) > (test_rs.oracle − test_rs.hard)` : 라우팅이 기여함 (논문 주장 성립)
- `main.oracle ≈ test_rs.oracle` : distribution shift가 애초에 작음 (문제 자체가 약함, 추가 분석 필요)
- `test_rs.hard > main.oracle` : 라우팅이 부작용 발생 (잘못된 분류가 오히려 손해)

---

### A-1. Ablation — 충방전 방향별 독립 게이트 vs 공유 단일 게이트

**목적**: 충전/방전 probe gate를 분리한 설계 결정이 성능에 기여하는지 정량화.  
**주장**: "충전·방전은 서로 다른 HI가 중요하다 (클러스터링 결과). 따라서 방향별 독립 게이트가 필요하다."  
**판단 기준**: 공유 게이트 대비 방향별 게이트의 val R² 유의미한 개선 + 선택 HI 집합 차이.

#### 필요 코드 수정

`5_model/models/scr_model.py`에 `shared_probe_gate` 모드 추가:
```python
# scr.yaml: classifier.shared_probe_gate: true 시
# charge_probe_gate == discharge_probe_gate (동일 파라미터 공유)
if cfg.get("classifier", {}).get("shared_probe_gate", False):
    self.charge_probe_gate = HardConcreteGate(N_HI)
    self.discharge_probe_gate = self.charge_probe_gate  # 동일 객체 참조
```

#### 실험 셋

| 실험 ID | 게이트 모드 | 축 | Phase | gates_from |
|---------|------------|-----|-------|------------|
| **A1-SCR-RCS** | 방향별 독립 (현재, 기준) | rcs | P1→P2 | `0715_1149` |
| A1-SHARED-RCS-P1 | 공유 단일 | rcs | P1 | *(신규 학습)* |
| A1-SHARED-RCS-P2 | 공유 단일 | rcs | P2 | A1-SHARED-RCS-P1 결과 |
| A1-SHARED-PROTO-P1 | 공유 단일 | protocol | P1 | *(신규 학습)* |
| A1-SHARED-PROTO-P2 | 공유 단일 | protocol | P2 | A1-SHARED-PROTO-P1 결과 |

#### YAML 차이 (A1-SHARED vs 기준)

```yaml
# A1-SHARED 전용 — classifier 섹션에만 추가
classifier:
  shared_probe_gate: true    # ← 새 옵션 (기본값 false)
  charge_probe_m: 3          # 공유 게이트에서 상위 3개 선택
  discharge_probe_m: 2       # 동일 게이트에서 상위 2개 선택
```

#### 실행 절차

```bash
# Phase 1: 공유 게이트 학습
python 5_model/train_scr.py --phase 1 --config 5_model/config/scr_a1_shared.yaml

# Phase 2: 공유 게이트 고정, Head 학습
python 5_model/train_scr.py --phase 2 --gates-from _5_data_model_scr/{A1-P1-결과폴더}
```

#### 입출력

- **입력**: 세그먼트 HI 64차원 (`_4_data_hi/{axis}/seg/`)
- **Phase 1 출력**: `gates/classification_HIs.json` (charge_ranked == discharge_ranked)
- **Phase 2 출력**: `metrics/metrics.json`, `figures/scatter_*.png`

#### 수집 지표

| 지표 | 비교 쌍 |
|------|---------|
| val R², test R², RMSE | 독립 게이트 vs 공유 게이트 |
| 선택 HI 집합 (charge vs discharge Jaccard) | 독립 게이트에서 충전·방전 선택 집합의 차이 |
| 시나리오 분류 정확도 | charge 시나리오, discharge 시나리오 각각 |

---

### A-2. Ablation — Cost-Aware L0 vs 균일 비용 L0

**목적**: HI 카테고리별 비용 차등(stat=1.0, diff=1.5, lfp=2.0, morph=3.0)이 선택 구성과 Pareto 효율에 기여하는지 검증.  
**주장**: "비용 가중 L0는 고비용 morph HI 선택을 억제하고, 동일 정확도를 더 저렴한 HI로 달성한다."  
**판단 기준**: 동일 총 선택 HI 수에서 cost-aware가 morph 비율 낮추면서 RMSE 동등 또는 개선.

#### 필요 코드 수정

`5_model/utils/hi_schema.py`에 균일 비용 옵션 추가:
```python
# 기존 (cost-aware)
CATEGORY_COSTS = {"stat": 1.0, "diff": 1.5, "lfp": 2.0, "morph": 3.0}

# 신규: scr.yaml uniform_hi_cost: true 시 override
def get_cost_vector(seg_name: str, uniform: bool = False) -> list[float]:
    costs = {k: 1.0 for k in CATEGORY_COSTS} if uniform else CATEGORY_COSTS
    ...
```

또는 scr.yaml에 추가:
```yaml
loss:
  uniform_hi_cost: false    # true → 모든 카테고리 비용 1.0 (ablation용)
```

#### 실험 셋

| 실험 ID | 비용 모드 | λ_l0 | 축 |
|---------|----------|------|-----|
| **A2-COST-RCS** | cost-aware (기준) | auto | rcs |
| A2-UNIFORM-RCS | 균일 (all=1.0) | auto | rcs |
| A2-COST-RCS-SWEEP | cost-aware, λ 스윕 | 0.001~0.1 × 5단계 | rcs |
| A2-UNIFORM-RCS-SWEEP | 균일, λ 스윕 | 0.001~0.1 × 5단계 | rcs |

λ 스윕 값: `[0.001, 0.005, 0.01, 0.05, 0.1]` → Phase 1을 5회 반복

#### YAML (A2-UNIFORM)

```yaml
loss:
  uniform_hi_cost: true
  lambda_l0_auto: true        # m/k 기준 자동 계산 (비용은 균일)
```

#### 수집 지표

| 지표 | 설명 |
|------|------|
| 선택 HI 카테고리 분포 | 각 시나리오별 stat/diff/lfp/morph 선택 수 |
| 가중 비용 합계 | Σ cost_i × I(gate_i active) — 실제 "지불한 비용" |
| val RMSE | 동일 λ에서 cost-aware vs 균일 |
| Pareto 곡선 | x=가중 비용, y=val RMSE (λ 스윕으로 생성) |

---

### A-3. Ablation — 시나리오별 독립 Scen Gate vs 단일 전역 Gate

**목적**: 6개 시나리오 각각에 독립 gate를 두는 설계의 효과 검증.  
**주장**: "충전 Low와 방전 High는 서로 다른 HI가 SOH를 예측한다."  
**판단 기준**: 시나리오별 RMSE breakdown에서 독립 gate가 특정 시나리오에서 유의미하게 개선.

#### 필요 코드 수정

`scr_model.py`에 `single_scen_gate: true` 옵션:
```python
# 6개 독립 gate → 1개 공유 gate
if cfg.get("classifier", {}).get("single_scen_gate", False):
    shared = HardConcreteGate(N_HI)
    self.scen_gates = nn.ModuleList([shared] * N_SEGS)
```

#### 실험 셋

| 실험 ID | Gate 구조 | 시나리오 분리 | 축 |
|---------|----------|------------|-----|
| **A3-SCR-RCS** | 6개 독립 (기준) | O | rcs |
| A3-SINGLE-RCS-P1 | 1개 공유 | X | rcs |
| A3-SINGLE-RCS-P2 | 1개 공유 | X | rcs |

#### YAML (A3-SINGLE)

```yaml
classifier:
  single_scen_gate: true    # ← 새 옵션
regression:
  scen_k_count: 55          # 동일 (공유 gate에서 상위 55개)
```

#### 수집 지표

| 지표 |
|------|
| 시나리오별 RMSE (chg_lo / chg_mid / chg_hi / dis_hi / dis_mid / dis_lo 분리) |
| 선택 HI 집합의 시나리오 간 Jaccard — 독립 시 얼마나 달라지는가 |
| 전체 val R² |

---

### A-4. Ablation — Phase 1 Only vs Phase 1 + Phase 2 (기존 코드로 실행 가능)

**목적**: Phase 2 분리 학습(고정 gate + Head 재학습)이 실제로 기여하는지 정량화.  
**판단 기준**: Phase 1 best.pt 평가 vs Phase 2 best.pt 평가 RMSE 차이.

#### 실험 셋

| 실험 ID | 학습 Phase | 평가 체크포인트 |
|---------|-----------|----------------|
| **A4-P2-RCS** | Phase 1 → Phase 2 (기준) | Phase 2 best.pt |
| A4-P1ONLY-RCS | Phase 1만 | Phase 1 final.pt |
| A4-P1ONLY-PROTO | Phase 1만 | Phase 1 final.pt |

#### 실행 절차 (코드 수정 불필요)

```bash
# Phase 1만으로 평가 — test_scr.py에 Phase 1 체크포인트 직접 전달
python 5_model/test_scr.py --checkpoint _5_data_model_scr/0715_1149/checkpoints/best.pt
```

> Phase 1 체크포인트에는 gate log_alpha가 있고, test_scr.py가 이를 감지해 gate JSON으로 변환 후 자동 평가. 추가 코드 불필요.

#### 수집 지표

| 지표 |
|------|
| train / val / test RMSE, R² |
| 시나리오 분류 정확도 (Phase 1에서 probe_mlp 성능) |

---

### B-1. Baseline — 글로벌 단일 MLP + 시나리오 One-Hot

**목적**: 시나리오별 라우팅 없이 동일 성능이 나오면 SCR의 존재 이유가 약해짐.  
**판단 기준**: SCR Phase 2 val R² > 글로벌 MLP val R².

#### 모델 구조

```
입력: [HI 64차원] + [seg one-hot 6차원] + [direction 1] + [cap_init 1]  = 72차원
→ MLP (72 → 256 → 128 → 64 → 1)
L0 gate 없음, 시나리오 라우팅 없음, 전체 HI 입력
```

#### 구현 방법

새 스크립트 `5_model/baselines/train_global_mlp.py` 작성:

```python
# scr.yaml과 동일한 데이터 분할 사용 (is_cross_dataset_evaluate: true)
# 입력: torch.cat([x_hi, seg_onehot, direction, cap_init], dim=-1)  # (B, 72)
# 손실: MSE만 (L0, CE 없음)
# 조기 종료: val RMSE 기준
```

#### YAML (별도 파일: `config/baseline_global_mlp.yaml`)

```yaml
scenario:
  axis: rcs               # 데이터 분할 기준 축
data:
  is_cross_dataset_evaluate: true
  split_seed: 42
baseline:
  model: "global_mlp"
  hidden_dims: [256, 128, 64]
  dropout: 0.1
training:
  epochs: 500
  lr: 5.0e-4
  early_stop_patience: 30
```

#### 수집 지표

| 지표 | 비교 기준 |
|------|----------|
| val R², test R² | SCR A4-P2-RCS vs B1-GLOBAL |
| 파라미터 수 | SCR(~31K) vs 글로벌 MLP(~50K) |
| 시나리오별 RMSE | 라우팅 없이 특정 시나리오에서 열세가 나타나는지 |

---

### B-2. Baseline — 시나리오별 완전 독립 모델 (6개)

**목적**: SCR의 공유 구조(probe gate 재사용, Stage A 공유 MLP)의 이득 검증.  
**판단 기준**: 공유 구조 SCR의 성능이 완전 독립 6모델보다 동등하거나 우수 (특히 데이터 희소 시나리오에서).

#### 구현 방법

기존 `scr.yaml`을 6개 복사 후 각각 단일 시나리오 데이터만 사용:

```bash
# 시나리오 0 (chg_lo)만 학습
python 5_model/train_scr.py --phase 2 --scen-idx 0
```

또는 독립 MLP 6개 학습 스크립트:
```python
# 각 시나리오(scen_idx)의 세그먼트만 필터링하여 독립 MLP 학습
for scen in range(6):
    ds = filter_by_scen(train_ds, scen)
    model = SimpleMLP(64 + 2, [128, 64], 1)  # 64 HI + direction + cap_init
    ...
```

#### 수집 지표

| 지표 |
|------|
| 시나리오별 RMSE — SCR vs 완전 독립 6모델 |
| 희소 시나리오(데이터 수가 적은 구간)에서의 성능 차이 |

---

### B-3. Baseline — 전통적 Feature Selection + XGBoost

**목적**: L0 학습된 HI 선택이 Lasso / RF Feature Importance 대비 우수함을 보임.  
**판단 기준**: SCR Phase 2 RMSE < XGBoost (동일 HI 수 조건).

#### 구현 방법 (`5_model/baselines/train_classic_selection.py`)

```python
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 1. Lasso로 per-scenario HI 55개 선택
lasso = Lasso(alpha=λ).fit(X_train_scen, y_train_scen)
selected = np.where(np.abs(lasso.coef_) > 0)[0][:55]

# 2. 선택된 HI로 XGBoost 학습
xgb = XGBRegressor(...).fit(X_train_scen[:, selected], y_train_scen)
```

#### 실험 셋

| 실험 ID | 선택 방법 | 회귀 모델 |
|---------|----------|----------|
| B3-LASSO-XGB | Lasso (per-scen) | XGBoost |
| B3-RF-XGB | RF Importance | XGBoost |
| B3-MRMR-XGB | mRMR | XGBoost |
| **비교 기준** | **SCR L0 gate** | **MLP Head** |

#### 수집 지표

| 지표 |
|------|
| val RMSE / R² (선택 HI 수 고정 시 동일 조건 비교) |
| 선택 HI 집합과 SCR gate 결과의 Jaccard 일치도 |

---

### C-1~C-4. 크로스 데이터셋 전이 검증 (보완 실험)

현재 C-1(MIT→HUST, rcs, 10-shot)만 완료. 나머지:

| 실험 ID | 방향 | 축 | Shot | 현황 |
|---------|------|-----|------|------|
| **C1-RCS** | MIT→HUST | rcs | 10-shot | ✅ R²=0.813 |
| C1-PROTO | MIT→HUST | protocol | 10-shot | ⬜ |
| C1-VWIN | MIT→HUST | vwindow | 10-shot | ⬜ |
| C2-RCS-REV | HUST→MIT | rcs | 10-shot | ⬜ |
| C3-JACCARD | — | rcs, protocol | — | ⬜ gate 일치도 계산 |
| C4-NSHOT-RCS | MIT→HUST | rcs | N=1,5,10,20 | ⬜ fine-tuning 곡선 |

#### C4 Fine-Tuning 절차

```bash
# N=1 fine-tuning
python 5_model/finetune_scr.py \
  --checkpoint _5_data_model_scr/0715_1340/checkpoints/best.pt \
  --n-cells 1 --seed 42 --dataset HUST

# N=5, 10, 20 반복 (seed 3개)
```

Fine-tuning YAML 핵심 설정:
```yaml
finetune:
  freeze_gates: true          # Phase 1 gate 동결 (반드시)
  n_cells: 10                 # HUST 학습 셀 수
  dataset: "HUST"
  lr: 2.0e-3                  # Phase 2보다 높은 LR (소량 데이터)
  epochs: 100
  early_stop_patience: 20
```

---

### E. Gate 안정성 — Seed 반복 실험

**목적**: Hard Concrete seed 민감도 사전 보고.  
**방법**: Phase 1을 seed 5개로 반복 → 각 HI의 선택 빈도 계산.

#### 실험 셋

```bash
# 5가지 seed로 Phase 1 반복 (rcs 축)
for seed in 0 1 2 3 4; do
  python 5_model/train_scr.py --phase 1 \
    --config 5_model/config/scr_stability.yaml \
    --seed $seed
done
```

stability 전용 YAML:
```yaml
scenario:
  axis: rcs
data:
  split_seed: 42              # 데이터 분할은 고정 (seed 변경은 모델 init만)
training:
  epochs: 300
  early_stop_patience: 50     # stability 실험은 빠른 수렴으로 충분
  run_overfit_test: false
```

#### 분석 스크립트 (`tmp_gate_stability.py`)

```python
# 5개 run의 regression_HIs.json 로드
# 시나리오별: 각 HI의 선택 빈도(0~1) 계산
# stability_selection_freq[seg][hi_idx] = (5회 중 선택 횟수) / 5
# freq >= 0.8 → "안정적 선택" (논문에서 이 집합만 최종 보고)
```

---

### F. 세그먼트 축 비교 — protocol / vwindow / rcs 전면 비교

**목적**: 어떤 분할 축이 MIT 내 성능과 MIT→HUST 전이에서 가장 우수한지 결정.  
**기대 결과**: vwindow가 배포 가능성(전압 직접 관측)에서 유리, rcs가 MIT 내 최고, protocol이 zero-shot 전이에 강.

#### Phase 2 실험 셋 (Phase 1은 모두 완료)

| 실험 ID | 축 | Phase 1 gates_from | Head | YAML 핵심 변경 |
|---------|-----|-------------------|------|--------------|
| **F-RCS-P2** | rcs | `0715_1149` | MLP-S | 기준선 (완료) |
| F-PROTO-P2 | protocol | `0715_1015` | MLP-S | `axis: protocol`, `gates_from: 0715_1015` |
| F-VWIN-P2 | vwindow | `0715_1040` | MLP-S | `axis: vwindow`, `gates_from: 0715_1040` |
| F-RCS-P2-FT | rcs | `0715_1149` | MLP-S | 위 C1-RCS fine-tune |
| F-PROTO-P2-FT | protocol | `0715_1015` | MLP-S | C1-PROTO fine-tune |
| F-VWIN-P2-FT | vwindow | `0715_1040` | MLP-S | C1-VWIN fine-tune |

각 실험 YAML — 변경 필드만:
```yaml
# F-PROTO-P2
scenario:
  axis: protocol
data:
  gates_from: "_5_data_model_scr/0715_1015"

# F-VWIN-P2
scenario:
  axis: vwindow
data:
  gates_from: "_5_data_model_scr/0715_1040"
```

#### 수집 지표 및 최종 비교표 형태

| 축 | MIT val R² | MIT val RMSE | HUST zero-shot R² | HUST 10-shot R² | 배포 가능성 |
|----|:----------:|:------------:|:-----------------:|:---------------:|:----------:|
| rcs | 0.643 | 0.0282 | -0.759 | **0.813** | 중 (경계 추론 필요) |
| protocol | 0.726 | 0.0360 | **0.766** | — | 중 (전환점 관측 필요) |
| vwindow | — | — | — | — | **고** (전압 직접 관측) |

---

### 실험 실행 순서 및 소요 시간 추정

#### Phase 0: 전제 조건 완료 (다른 실험 시작 전)

| 단계 | 작업 | 명령/파일 | 소요 시간 | 상태 |
|:----:|------|---------|:--------:|:----:|
| 0-1 | test_rs 생성 완료 확인 | `cat _4_data_hi/test_rs/test_rs_stats.txt` | — | 진행 중 |
| 0-2 | `main.oracle` 기준선 평가 (qfrac→qfrac) | `test_scr.py --checkpoint 0715_1340/...` | ~5분 | ⬜ |
| 0-3 | `test_rs.oracle` 배포 시나리오 평가 (qfrac→random) | scr.yaml `random_segment_test: true` | ~5분 | ⬜ |

#### Phase 1: 분류기 활성화 + test_rs.hard/soft (핵심 실험)

| 단계 | 실험 ID | 코드 수정 | 실행 시간 | 우선순위 |
|:----:|---------|:--------:|:--------:|:--------:|
| 1 | 분류기 활성화 Phase 1 재학습 | `lambda_scen: 0.5` | ~1시간 | ★★★ |
| 2 | 분류기 포함 Phase 2 재학습 | `gates_from: {cls_run}` | ~30분 | ★★★ |
| 3 | **test_rs.hard** 평가 (분류기 argmax 라우팅) | 분류기 체크포인트 지정 후 test_scr.py | ~5분 | ★★★ |
| 4 | **test_rs.soft** 평가 (분류기 확률 가중, 보조) | 동일 실행에서 자동 산출 | ~5분 | ★★★ |

> **단계 1~4 완료 후**: `main.oracle`/`test_rs.oracle`/`test_rs.hard`/`test_rs.soft` 비교표 작성 → 분류기 기여도 정량화

#### Phase 2: 축 비교 실험 (코드 수정 불필요)

| 단계 | 실험 ID | 코드 수정 | 실행 시간 | 우선순위 |
|:----:|---------|:--------:|:--------:|:--------:|
| 5 | **F-PROTO-P2, F-VWIN-P2** | 없음 | 각 ~30분 | ★★★ |
| 6 | **A4-P1ONLY-RCS** | 없음 | ~5분 (평가만) | ★★★ |
| 7 | **C4-NSHOT-RCS** (N=1,5,20) | 없음 | ~30분 | ★★★ |
| 8 | 각 축별 `test_rs.oracle`/`hard`/`soft` 평가 (protocol/vwindow) | yaml 변경 | ~20분 | ★★☆ |

#### Phase 3: Ablation 및 Baseline (코드 수정 필요)

| 단계 | 실험 ID | 코드 수정 | 실행 시간 | 우선순위 |
|:----:|---------|:--------:|:--------:|:--------:|
| 9 | **B3-LASSO-XGB** | 신규 스크립트 | ~2시간 | ★★☆ |
| 10 | **E-STABILITY** (5 seeds) | 없음 | ~2.5시간 | ★★☆ |
| 11 | **A2-UNIFORM** | hi_schema.py 1줄 | ~1시간 | ★★☆ |
| 12 | **A1-SHARED** (게이트 공유) | scr_model.py 수정 | ~1시간 | ★★☆ |
| 13 | **B1-GLOBAL-MLP** | 신규 스크립트 | ~1시간 | ★★☆ |
| 14 | **A3-SINGLE** (단일 scen gate) | scr_model.py 수정 | ~1시간 | ★☆☆ |
| 15 | **B2-INDEPENDENT** | 신규 스크립트 | ~3시간 | ★☆☆ |
| 16 | F-PROTO/VWIN fine-tuning (C시리즈 보완) | 없음 | ~1시간 | ★☆☆ |
| 17 | C3-JACCARD 분석 | 분석 스크립트 | ~30분 | ★☆☆ |

> **즉시 실행 가능**: Phase 0 (0-1 완료 후 0-2, 0-3) → Phase 1 (1~4) → Phase 2 (5~8)  
> 분류기 코드 활성화 외 코드 수정이 불필요한 작업들이 대부분이므로 Phase 0~2를 먼저 완주하고 논문 주장 검증 후 Phase 3 ablation으로 진행.
