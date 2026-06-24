# HI–Capacity 상관계수 비교: 이전 접근 vs 현재 접근

대상 파일:
- 이전: `5_train/preprocess.ipynb`
- 현재: `4_hi_analysis/hi_correlation.py` + `4_hi_analysis/seg_corr_analysis.py`

---

## 결론 요약

이전 접근에서 HI–capacity 상관계수가 최대 0.4 수준이었던 반면, 현재 접근에서는 0.7–0.9+가 나오는 주요 원인은 세 가지다.

1. **상관계수 계산 방식의 차이** (가장 큰 요인) — pooled scatter vs within-cell Spearman
2. **세그먼트 경계 정의 방식의 차이** — 고정 초기 용량 기준 vs 현재 사이클 실측 용량 기준
3. **피처 계산 버그** — `dtdt`, `ent_v` 일부 피처에 한정

---

## 1. 상관계수 계산 방식

### 이전 접근 (preprocess.ipynb Cell 44)

```python
sample_pool = random.sample(pool, min(5000, len(pool)))
df_features = pd.DataFrame([{
    "cyc": s["cyc"], "cap": s["capacity"],
    **{f"f{i}": val for i, val in enumerate(s["x"])}
} for s in sample_pool])

# feature-feature correlation matrix
corr = df_features[[f"f{i}" for i in range(num_features)]].corr()

# feature-capacity scatter plot (시각적 추정)
sns.scatterplot(data=df_features, x='cap', y='f0', alpha=0.3)
```

모든 셀 × 모든 사이클에서 5,000개를 랜덤 샘플링해 한꺼번에 섞은 뒤 scatter plot으로 시각적으로 상관계수를 추정했다. 명시적인 feature–capacity Spearman 계산 없이 scatter의 퍼짐 정도로 판단했다.

이 pooled 상관계수에는 두 신호가 섞여 있다:
- **원하는 신호**: 셀 내 열화에 따른 feature 변화 (동일 셀의 사이클 진행)
- **노이즈/교란 변수**: 셀 간 제조 편차 (초기 용량, 온도 특성, 충방전 프로토콜 차이)

### 현재 접근 (seg_corr_analysis.py)

```python
# 각 셀마다 독립적으로 Spearman 계산
for cell_id, grp in df.groupby("cell_id"):
    rho, _ = spearmanr(grp["feature"], grp["capacity_Ah"])
    cell_rhos.append(rho)

# 셀 전체 평균 집계
mean_rho = np.mean(cell_rhos)
std_rho  = np.std(cell_rhos)
```

셀마다 `Spearman(feature, capacity_Ah)`를 독립 계산한 뒤 셀 전체의 평균과 표준편차를 낸다.

### 왜 수치 차이가 크게 나는가

```
Pooled scatter:
  셀 A (초기 1.05 Ah) 사이클 1–500
  셀 B (초기 1.08 Ah) 사이클 1–300    → 모두 혼합
  셀 C (초기 1.10 Ah) 사이클 1–700
  → 셀 간 제조 편차가 scatter를 넓힘 → ρ ≈ 0.4

Within-cell Spearman 평균:
  셀 A: ρ(feature, cap) = 0.82
  셀 B: ρ(feature, cap) = 0.79
  셀 C: ρ(feature, cap) = 0.84
  → 평균 ≈ 0.82
```

동일 셀 내에서는 inter-cell 노이즈가 없어 feature의 단조성이 훨씬 강하게 드러난다.

---

## 2. 세그먼트 경계 정의 방식

### 이전 접근 (preprocess.ipynb Cell 23, 26)

```python
# Cell 26: 셀의 '첫 번째' 유효 사이클 용량을 nominal_q로 고정
nominal_q = 1.1  # Default
for test_cyc in sorted_all:
    _, test_cap = get_cell_labels(cell, test_cyc, DATASET_TYPE)
    if test_cap > 0:
        nominal_q = test_cap  # 초기 용량, 이후 모든 사이클에 고정됨
        break

# Cell 23: 분모가 고정된 nominal_q
q_ratio = q_acc / (nominal_q + 1e-9)  # 열화 후에도 분모는 초기 용량 그대로
mask = (q_ratio >= start_p) & (q_ratio <= end_p)
```

충전 s_hi = `q_ratio ∈ [0.7, 1.0]` = 초기 용량의 70–100% 구간 (CV 영역).

셀이 1.1 Ah → 0.75 Ah로 열화하면:
- `q_ratio` 최대값 = 0.75 / 1.1 = **0.68**
- 충전 s_hi (`q_ratio ≥ 0.7`) → **데이터 없음**
- 충전 s_mid (`q_ratio 0.3–0.7`) → 일부만 존재

결과: 열화가 진행된 사이클은 s_hi 세그먼트가 NaN → 상관계수 계산에서 제외 → 초기 사이클(용량 높음)만 남아 변동폭 감소 → ρ 낮아짐.

### 현재 접근 (hi_correlation.py)

```python
# 현재 사이클의 실측 용량을 분모로 사용
q_local = float(seg["capacity_Ah"])  # 이번 사이클의 실제 측정 용량
# chg_gap_seg 컬럼이 q_cum / q_local 비율로 미리 계산되어 있음
mask = df_seg["chg_gap_seg"] == seg_label  # s_hi, s_mid, s_lo
```

어느 사이클이든 s_hi = 실제 용량의 상위 40%를 항상 캡처. 열화가 진행돼도 세그먼트는 사라지지 않는다.

### 영향 비교

| 셀 상태 | 이전 (고정 nominal_q=1.1) | 현재 (current q_local) |
|---------|--------------------------|----------------------|
| 초기 (1.10 Ah) | chg s_hi: 정상 (70–100%) | chg s_hi: 정상 (60–100%) |
| 중간 열화 (0.85 Ah) | chg s_hi: 부분 (70–77%) | chg s_hi: 정상 (60–100%) |
| 심한 열화 (0.75 Ah) | chg s_hi: **비어 있음** | chg s_hi: 정상 (60–100%) |

---

## 3. 피처 계산 버그 (일부 피처에 한정)

상세 내용은 `docs/HI_OUTLIER_FIXES.md` 참조.

| 피처 | 이전 버그 | 영향 |
|------|-----------|------|
| `dtdt` | `np.maximum(dt, 1e-6)` 하한 → 노이즈 지배 (p50 = 35°C/s) | dtdt–capacity 상관 거의 0 |
| `ent_v` | `density=True` → 음수 엔트로피 발생 | ent_v 값 자체가 무의미 |

`mean_v`, `v_std`, `corr_vi` 같은 기본 통계 피처는 이전에도 버그가 없었으므로 이 피처들의 상관계수 차이는 원인 1·2가 지배적이다.

---

## 4. 연구적 관점: 어느 방식이 더 정확한가

**SOH 추정 목적에서는 within-cell Spearman이 더 정확하다.**

SOH의 정의 자체가 per-cell이기 때문이다:

```
SOH_i(t) = capacity_i(t) / capacity_i(0)
```

모델이 해야 할 일은 "이 셀이 열화할 때 feature가 얼마나 잘 따라가는가"이고, within-cell Spearman이 이것을 직접 측정한다. Pooled 방식은 제조 편차와 열화 신호가 혼재해 어느 것을 설명하는지 구분할 수 없다.

### 엄밀한 검증을 위해 필요한 metric

| Metric | 측정 내용 | 용도 |
|--------|-----------|------|
| Within-cell Spearman — **평균** (현재 구현) | 열화 추적 능력 | 피처 선택 |
| Within-cell Spearman — **표준편차** | 셀 간 일관성 | 일반화 검증 |
| Pooled Spearman on **SOH** (= cap/cap_initial) | 셀 간 절대 레벨 구분 | 다수 셀 동시 추론 시 |

std_ρ가 크면 일부 셀에서만 작동하는 피처라는 의미이므로, mean_ρ가 높더라도 채택을 보수적으로 검토해야 한다.

---

## 5. 피처 선택 시 주의: leakage 위험

Within-cell Spearman이 0.9+로 매우 높은 피처는 반드시 leakage 여부를 검토해야 한다.

현재 파이프라인에서 확인된 leakage 피처:

```
q_abs_s_hi / capacity_Ah = 0.3629 ± 0.0048  (변동계수 1.3%)
```

세그먼트가 `q_cum` 비율로 정의되므로 `q_abs` (세그먼트 내 누적 전하량)는 사실상 `capacity_Ah × 상수`다. `energy_Wh`도 동일한 이유로 leakage다.

이 두 피처를 제외한 후에도 현재 접근의 상관계수가 높은 것은 **원인 1 (within-cell metric)과 원인 2 (일관된 세그먼트 경계)**의 효과다.

---

## 6. 현재 파이프라인 내 두 플롯 간 수치 차이

### 현상

같은 `hi_features.pkl`을 읽어도 두 플롯의 charge-low(충전 SoC 0~30%) 상관계수가 크게 다르다.

| 플롯 | 파일 | charge-low 최고 비-leakage ρ |
|------|------|------------------------------|
| `hi_correlation.png` | `hi_correlation.py` | ≈ 0.42 (`ocv_slope`) |
| `corr_rank_mit.png` | `seg_corr_analysis.py` | ≥ 0.5 (다수 HI) |

### 원인: 계산 방식이 코드 수준에서 다르다

**`hi_correlation.py` — 풀링 방식 (line 857–865)**

```python
sub = df[df["dataset"] == ds]          # MIT 전체 셀을 한 덩어리로 합침
valid = sub[[hi, "capacity_Ah"]].dropna()
rhos[hi] = spearmanr(valid[hi], valid["capacity_Ah"])[0]
```

MIT 전 셀의 데이터가 하나의 pool로 섞인 후 Spearman을 계산한다.

**`seg_corr_analysis.py` — 셀 내부 방식 (line 100–113)**

```python
for (dataset, cell_id), sub in df.groupby(["dataset", "cell_id"]):
    rho = spearmanr(vals[mask], cap[mask])   # 셀마다 독립 계산
# 전체 셀에 걸쳐 mean / std 집계
mean_rho = cell_corr_df.mean(skipna=True)
```

각 셀 내부에서 독립적으로 Spearman을 구한 뒤 셀 전체 평균을 낸다.

### 왜 charge-low에서 차이가 특히 크게 나타나는가

충전 초반 저전압(SoC 0~30%) 구간 HI는 셀 간 초기값 차이(제조 편차)가 상대적으로 크다.
이 구간의 전압·전류 특성은 셀마다 양극재 로트나 전해질 충진량 차이에 민감하기 때문이다.

```
풀링 시:
  b1c0  HI값 범위: 0.27~0.31   (cap: 1.10→1.00 Ah)
  b2c13 HI값 범위: 0.23~0.26   (cap: 1.08→0.98 Ah)
  → 두 셀의 HI 범위가 겹치지 않음
  → Spearman이 "셀 식별자"를 구분하는 방향으로 결정됨
  → 열화 추이와 무관한 분산이 섞여 ρ↓ (≈ 0.42)

셀 내부 방식:
  b1c0  내부:  ρ(HI, cap) = +0.95
  b2c13 내부:  ρ(HI, cap) = +0.91
  b2c17 내부:  ρ(HI, cap) = +0.93
  → 평균 ρ ≈ 0.93  ← corr_rank_mit.png 값
```

셀 간 제조 편차가 풀에서는 "노이즈"로 작용하지만, 셀 내부 계산에서는 완전히 사라진다.

### 요약

| 구분 | 풀링 Spearman (`hi_correlation.py`) | 셀 내부 Spearman 평균 (`seg_corr_analysis.py`) |
|------|--------------------------------------|------------------------------------------------|
| 계산 단위 | MIT 전체 셀 혼합 | 셀마다 독립 → 전체 평균 |
| 포함되는 분산 | 열화 신호 + 셀 간 제조 편차 | 열화 신호만 |
| charge-low 결과 | ρ ≈ 0.42 (편차에 묻힘) | ρ > 0.5 (편차 제거됨) |
| SOH 추적 능력 반영 | 과소평가 | 정확한 반영 |

`corr_rank_mit.png`의 값이 charge-low HI의 SOH 추적 능력을 더 정확하게 반영한다.
`hi_correlation.png`의 낮은 값은 HI가 나쁜 게 아니라, 풀링으로 인해 셀 간 편차가 신호를 희석시킨 결과다.
