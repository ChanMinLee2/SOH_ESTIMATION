# DFR 모델 동작 상세 설명

LFP 배터리 용량(Ah) 예측을 위한 Dynamic Feature Routing(DFR) 모델의 학습과 인퍼런스 과정을 단계별로 정리한다.

---

## 학습 과정 (Training)

### Phase 0. 사전 준비 (train.py 실행 시 1회)

**0-1. 데이터 로딩**
- `_4_data_hi/MIT/*.pkl`, `HUST/*.pkl` 파일들을 ThreadPoolExecutor 8스레드로 병렬 로드
- 전체를 하나의 DataFrame으로 합침 (행 = 사이클, 열 = 415개 컬럼)

**0-2. 셀 단위 분할**
- 셀 ID 목록을 MIT/HUST 각각 80/10/10으로 나눔
- 같은 셀의 사이클이 train/val/test에 걸쳐 나뉘지 않음 (누수 방지)

**0-3. 정규화기 피팅**
- train DataFrame만 사용해서 피처 그룹별 (mean, std) 계산
- NaN은 제외하고 계산, 완전 NaN 컬럼은 mean=0 std=1 fallback
- val/test에는 이 통계를 그대로 적용 (test time leakage 방지)

**0-4. 텐서 사전 빌드 (BatteryHIDataset)**

```
x_global:       (N_train, 15)       ← Global HI, 정규화 완료
group_features: (N_train, n_k) × 24 ← 그룹별 Segment HI, NaN→0 처리
nan_masks:      (N_train, n_k) × 24 ← 1=유효값, 0=원래 NaN이었음
targets:        (N_train,)          ← capacity_Ah
```

모든 텐서가 메모리에 올라간 상태로 `__getitem__`은 순수 인덱스 접근만 수행.

---

### Phase 1. Epoch 루프 (epochs=200 반복)

매 epoch 시작 시 두 값을 계산:

```
Temperature T = max(0.5,  2.0 × 0.99^epoch)
               epoch   0: T = 2.00
               epoch  50: T = 1.21
               epoch 138: T = 0.50

Learning Rate = Cosine with warmup
               epoch 0–4:   선형 증가  0 → 1e-3
               epoch 5–200: 코사인 감쇠  1e-3 → 1e-6
```

---

### Phase 2. _train_epoch (배치 루프)

배치 하나 `(B=256)`를 기준으로 단계별 설명.

**Step A. 배치 수신**

```
x_global       (256, 15)           Global HI 정규화값
group_features [(256, 20)] × 20    Segment-Category 피처 (stat/diff/lfp 각 20개)
               [(256,  6)] ×  4    morph 피처 (6개)
nan_masks      같은 구조            0=결측, 1=유효
target         (256,)              실측 capacity_Ah
```

**Step B. InitialEncoder 통과**

```
x_global (256, 15)
→ Linear(15→128) → LayerNorm → GELU → Dropout
→ Linear(128→64) → LayerNorm
→ z (256, 64)
```

z는 해당 사이클의 전반적 열화 상태를 압축한 잠재 벡터.  
이후 ① 라우터, ② 퓨전 두 곳에 동시에 공급됨.

**Step C. FeatureRouter 통과 — 핵심 단계**

```
z (256, 64)
→ Linear(64→64) → LayerNorm → GELU → Dropout
→ Linear(64→24)
→ logits (256, 24)
```

이후 Gumbel-Sigmoid 적용:

```python
u = Uniform(0,1).sample()                # (256, 24)
gumbel = -log(-log(u))                    # Gumbel 노이즈
gates = sigmoid((logits + gumbel) / T)    # (256, 24), 값은 (0, 1) 연속
```

- T=2.0일 때: gate값이 0.2~0.8처럼 중간에 퍼짐 → 다양한 그룹 탐색
- T=0.5일 때: gate값이 0.05 또는 0.95처럼 극단화 → 이진 결정에 수렴
- 같은 배치 내 샘플마다 Gumbel 노이즈가 달라서 각 사이클이 다른 gate를 가짐

**Step D. FeatureSelector 통과**

```
각 그룹 k에 대해:
  x_out_k = group_features[k] * nan_masks[k] * gates[:, k:k+1]
  (256, n_k) =     (256, n_k)  *   (256, n_k) *    (256, 1)
```

- `nan_masks=0`인 자리: 원래 계산 불가였던 피처 → 0으로 소거
- `gates≈0`인 자리: 라우터가 이 그룹 불필요 판단 → 0에 가깝게 소거
- 학습 중이라 gate가 연속값 → 역전파 가능

**Step E. CategoryEncoder 통과**

```
24개 그룹 각각:
  gated_k (256, n_k)
  → 해당 카테고리의 공유 인코더 (stat / diff / lfp / morph 중 하나)
  → h_k (256, 32)

결과: h_groups = [h_0, h_1, ..., h_23]  각각 (256, 32)
```

gate≈0이었던 그룹은 입력이 0에 가까우므로 h_k도 0에 가까운 벡터가 됨.  
같은 카테고리의 6개 구간은 동일 인코더 가중치를 공유.

**Step F. FeatureFusion 통과**

```
concat([z,  h_0, h_1, ..., h_23])
     = concat([(256, 64), (256, 32) × 24])
     = (256, 64 + 768) = (256, 832)

→ Linear(832→128) → LayerNorm → GELU → Dropout
→ Linear(128→128) → LayerNorm
→ h_fused (256, 128)
```

z의 skip connection: 라우터가 모든 그룹을 꺼도 z가 항상 퓨전에 참여하므로 Global 정보는 손실 없음.

**Step G. CapacityHead 통과**

```
h_fused (256, 128)
→ Linear(128→64) → LayerNorm → GELU → Dropout
→ Linear(64→1) → squeeze
→ cap (256,)   ← 예측 capacity_Ah
```

**Step H. 손실 계산**

```
mse = MSE(cap, target)

costs_norm = [1.0, 1.0, ..., 1.5, ..., 2.0, ..., 3.0] / 3.0   # (24,), 정규화
sparse = mean(gates * costs_norm)   # 활성 게이트 × 비용 평균

loss = mse + 0.01 × sparse
```

- `sparse` 항은 학습 내내 "게이트를 켜는 비용"을 부여
- morph 그룹(cost=3.0)은 stat(cost=1.0)보다 3배 비싸므로 자연스럽게 억제됨

**Step I. 역전파 및 파라미터 업데이트**

```
loss.backward()
→ CapacityHead → FeatureFusion → CategoryEncoders
   → FeatureSelector (gates의 미분이 흐름)
   → FeatureRouter (logits로 역전파)
   → InitialEncoder

clip_grad_norm_(모든 파라미터, max_norm=1.0)   # 그래디언트 폭발 방지
optimizer.step()   # Adam
```

Gumbel 노이즈는 상수이므로 미분에서 사라짐.  
`sigmoid((logit + G) / T)`의 `logit`에 대한 미분이 정상 흐름.

---

### Phase 3. _eval_epoch (val, 매 epoch)

train epoch와 동일하지만 세 가지 차이:

| 항목 | train | val |
|------|-------|-----|
| Gumbel 노이즈 | 추가됨 | **없음** |
| hard thresholding | 없음 | **없음** (soft gate) |
| 역전파 | 있음 | **없음** (`@torch.no_grad`) |
| Dropout | 활성 | **비활성** (`model.eval()`) |

val에서는 `gates = sigmoid(logits)` 그대로 사용.  
0.5 기준 활성 그룹 수를 계산해 `mean_active` 통계 추적.

---

### Phase 4. 체크포인트 및 조기종료

```
매 epoch 후:
  if val_loss < best_val_loss:
      → best.pth 저장  (model_state + optimizer_state + epoch + val_loss + T)
      no_improve = 0
  else:
      no_improve += 1
      if no_improve >= 30:   # patience
          학습 종료

매 10 epoch:
  → epoch_XXXX.pth 저장  (스냅샷)
```

---

## 인퍼런스 과정 (Inference)

`model.eval()` + `@torch.no_grad()` 상태에서 진행.

---

**Step 1. 전처리**
- 학습 때 저장한 `normalizer.pkl` 로드
- 동일한 정규화 통계(train 기준 mean/std)를 새 데이터에 적용

**Step 2. InitialEncoder**

```
x_global (B, 15) → z (B, 64)
Dropout 비활성, LayerNorm 정상 동작
```

**Step 3. FeatureRouter — 학습과 핵심 차이**

```python
logits = router_mlp(z)           # (B, 24)
gates  = sigmoid(logits)         # Gumbel 노이즈 없음
gates  = (gates > 0.5).float()   # hard=True: {0.0, 1.0}으로 이진화
```

- 노이즈 없이 결정론적(deterministic)으로 동작
- 0.5 이상이면 1, 미만이면 0 → 진짜 이진 스위치
- 같은 입력에 항상 같은 라우팅 결과

**Step 4. FeatureSelector**

```
x_out_k = x_k * mask_k * gate_k   (gate_k ∈ {0.0, 1.0})
```

`gate_k = 0.0`인 그룹: 입력이 완전히 0 → CategoryEncoder가 완전히 0 벡터 출력.

**Step 5–7. 나머지 (학습과 동일)**

```
CategoryEncoders → h_groups
FeatureFusion(z, h_groups) → h_fused
CapacityHead → capacity_Ah
```

---

## 학습 vs 인퍼런스 핵심 차이 요약

| 단계 | 학습 | 인퍼런스 |
|------|------|---------|
| FeatureRouter 출력 | `sigmoid((logit + Gumbel) / T)`, 연속값 | `sigmoid(logit) > 0.5`, 이진값 |
| gate의 성질 | 미분 가능한 soft gate | 실제 on/off 스위치 |
| gate=0 그룹 처리 | 0에 가까운 작은 값, 그래디언트 흐름 | 완전히 0, 연산 절감 가능 |
| Dropout | 활성 | 비활성 |
| 역전파 | 있음 | 없음 |
| 결과 | loss 최소화 + gate 최적화 | 결정론적 예측값 |

---

## 전체 흐름 한 줄 요약

**학습**: 매 배치마다 Gumbel 노이즈로 확률적 탐색을 하면서, MSE와 비용 가중 희소성을 동시에 줄이는 방향으로 InitialEncoder·FeatureRouter·CategoryEncoders·FeatureFusion·CapacityHead 162K개 파라미터 전체를 역전파로 업데이트.

**인퍼런스**: 학습된 FeatureRouter가 현재 사이클의 z를 보고 24개 그룹 중 필요한 것만 이진 선택, 나머지는 완전 소거한 후 용량 예측.

---

## 관련 문서

- [MODEL_BLUEPRINT.md](MODEL_BLUEPRINT.md) — DFR 설계 원칙
- [MODEL_EXPLAIN.md](MODEL_EXPLAIN.md) — 파일별 코드 상세 설명
- [PIPELINE.md](PIPELINE.md) — 전체 파이프라인 흐름
