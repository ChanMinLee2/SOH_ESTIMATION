# HI 설계 방향성 검토: 논문 계통 대비 갭 분석

`docs/HI_refferences.md` 에 정리된 4대 논문 계통과 현재 321D HI 설계(NEW_HIS.md)를 대조하여  
커버되지 못한 영역과 보강 방향을 정리한 문서.

---

## 1. 논문 계통별 커버리지 요약

| 계통 | 대표 논문 | 핵심 아이디어 | 현재 커버 | 평가 |
|------|----------|-------------|----------|------|
| 1. 미분 기법 (ICA/DVA) | Berecibar 2016 (Energy) | DVA 두 피크 위치·거리·깊이 | Global G06–G10, Segment B카테고리 D01–D15 | **70%** — DVA 2차 피크 미포함 |
| 1. 미분 기법 (ICA/DVA) | Zheng 2018 (JPS) | 부분 CC 구간 IC 피크 추출 | Segment D06·D07 (구간별 ICA) | **85%** — 부분 곡선 외삽 미구현 |
| 2. 형태학적 특징 | Hu 2021 (IEEE Trans. Transp. Electr.) | V-Q·V-T 곡선 기하 피처 + 선택 | 카테고리 D (DTW·Fréchet × 3곡선 × 6구간) | **95%** — V-T 제외(temperature 데이터 없음) |
| 2. 형태학적 특징 | DTW + GPR (JES 2020) | DTW 거리를 단일 HI 스칼라로 | 카테고리 D `morph_v*_dtw_*` | **100%** |
| 3. 시간/에너지 기반 | Ref-based Time (IEEE Trans. PD 2023) | 고정 전압 [V₁,V₂] 구간 충전 시간 | 없음 (dvdt_slope은 용량 경계 기반) | **40%** — 핵심 갭 |
| 3. 시간/에너지 기반 | Energy-based GPR (JES 2022) | V-Q 적분 = 에너지 HI | G02 `energy_dis`, S12 `E_seg` | **100%** |
| 4. 딥러닝 잠재 특성 | Tian 2022 (ESM) | CNN 인코더 → latent vector HI | 의도적 제외 (hand-crafted 프레임워크) | N/A |

---

## 2. 갭 상세 분석

### Gap 1: DVA 2차 피크 특성 — Berecibar 2016

**논문의 핵심**: LFP 방전 dV/dQ 곡선에는 플래토 양 경계(진입·이탈부)에 두 개의 골짜기(valley)가 존재한다.  
두 골짜기의 Q-거리 = 플래토 용량 = SOH 직결 지표.

**현재 설계 상황**:

| Berecibar HI | 현재 대응 | 비고 |
|---|---|---|
| DVA 1차 골짜기 Q 위치 | G09 `dva_valley_q` ✅ | |
| DVA 1차 골짜기 깊이 | G10 `dva_valley_depth` ✅ | |
| DVA **2차 골짜기** Q 위치 | 없음 ❌ | |
| DVA **2차 골짜기** 깊이 | 없음 ❌ | |
| 두 골짜기 간 Q 거리 | G05 `q_plateau_frac` (전압 경계 기반 근사) △ | DVA 피크 직접 측정 아님 |

**부분 방전 시 가시성 분석**:

```
dV/dQ 형태 (방전)
  |
0 |.......____________________________......... → Q
  |  Valley 1         plateau ≈ 0        Valley 2
  |  (플래토 진입)                       (플래토 이탈)
  |
  q_frac: 0   ~0.05~0.15            ~0.85~0.95   1.0
  SoC:  100%      85%                   15%        0%
```

| 방전 시나리오 | Valley 1 | Valley 2 |
|---|---|---|
| 완전 방전 (SoC 100→0%) — MIT | ✅ | ✅ |
| 부분 방전 (SoC 100→50%) — HUST 흔한 패턴 | ✅ | ❌ (미도달) |
| 부분 방전 (SoC 70→30%) — mid 구간만 | ❌ | ❌ |

**결론**: Valley 2는 심방전(SoC 0% 근접) 시에만 존재.  
HUST 부분 방전 데이터에서는 NaN 비율이 높아 **실용성 낮음**.  
MIT 전체 방전에서는 추출 가능하나, 현재 `dis_lo` 세그먼트의 `min(dV/dQ)` (D04)가 사실상 이 영역을 이미 커버.

**우선순위: 낮음** — 추가 효과 대비 NaN 비용이 큼.

---

### Gap 2: 고정 전압 경계 충전 플래토 HI — 2023 Ref-based Time ★

**논문의 핵심**:  
충전 시작 SoC가 매 사이클 달라도, 고정 전압 [V₁, V₂] 내 데이터가 조금이라도 있으면 유효한 HI 추출 가능.  
→ **부분 충전 시나리오에서 가장 강건한 HI 중 하나**.

```
현재 q_frac 기반 세그먼트의 한계 (부분 충전 시)

cycle A (SoC 30→90% 충전):  chg_lo = SoC 30~52%, chg_mid = SoC 52~72%
cycle B (SoC 10→90% 충전):  chg_lo = SoC 10~42%, chg_mid = SoC 42~66%
→ 같은 q_frac 경계가 다른 절대 SoC를 가리킴

고정 전압 경계 [3.10V, 3.45V] 기반:
cycle A, B 모두 이 전압 범위 안 데이터 → 직접 비교 가능
```

**현재 설계 대응**:

| 논문 HI | 현재 대응 | 차이 |
|---|---|---|
| ΔT(V₁→V₂) — 충전 시간 | `dvdt_slope` (ΔV/Δt) 역수 유사 | q_frac 경계 기반, 전압 경계 아님 |
| Q(V₁→V₂) — 충전 용량 | G05 `q_plateau_frac` (방전 전용) | 충전 방향 없음 |
| 충전 플래토 용량 | 없음 | 완전 누락 |
| 충전 플래토 시간 | G13 `cv_time_frac` (전체 CV 비율) | 전압 구간 특정 아님 |

**신규 HI 제안**:

| 제안 키 | 수식 | 물리적 의미 | 열화 방향 |
|---|---|---|---|
| `q_chg_plateau` | `Q(CC phase ∩ V ∈ [3.10, 3.45V])` [Ah] | CC 충전 중 LFP 플래토 구간 충전 용량 | ↓ (R↑ → 빠른 전압 상승 → 구간 단축) |
| `t_chg_plateau` | `t(CC phase ∩ V ∈ [3.10, 3.45V])` [s] | 동 구간 경과 시간 | ↓ |

**우선순위: 높음** — 구현 난이도 낮음 (전압 마스킹 후 시간·전하 적산), HUST 부분 충전에 즉시 적용 가능.

---

### Gap 3: 부분 IC 곡선 외삽 — Zheng 2018 (부분 구현)

**논문의 핵심**:  
부분 CC 충전 구간에서 ICA 피크 전체가 보이지 않을 때,  
현재 보이는 IC 곡선의 **기울기 추세**로 피크 위치를 추정할 수 있다.

**현재 설계 상황**:  
`dqdv_peak_v` (D07), `dqdv_peak_h` (D06) 는 피크가 존재할 때만 유효 → 피크 없으면 NaN.  
부분 충전 구간에서 NaN 비율이 높아 이 HI들의 정보 밀도가 낮음.

**보강 아이디어**:

| 제안 키 | 수식 | 물리적 의미 |
|---|---|---|
| `dqdv_slope_at_entry` | dQ/dV 곡선에서 구간 진입 5% 지점의 기울기 | IC 피크를 향해 오르고 있는 속도 → 피크 높이 간접 추정 |
| `dqdv_at_exit` | 구간 마지막 5% 지점의 dQ/dV 값 | 피크 이후 하강 속도 |

**우선순위: 중간** — 효과 불확실, NaN 채우기 이상의 새 정보를 주는지 검증 필요.

---

### Gap 4: V-T 곡선 기반 HI — Hu 2021 (의도적 제외)

**논문의 핵심**: V-T(Voltage vs Temperature) 곡선의 기하 피처가 강력한 HI.

**제외 이유**: `data_unified` 포맷에서 temperature_C 컬럼 제거됨.  
(`data_raw` 에는 있으나 통합 파이프라인에서 제외된 상태)

**향후 복원 시 추가 가능한 HI**:
- `T_rise_seg`: 구간 내 온도 상승 [°C]
- `T_peak_seg`: 구간 최고 온도
- `corr_VT_seg`: V-T 상관계수 (발열과 전압 관계)

---

## 3. 신규 HI 보강 제안 종합

### Priority 1: `q_chg_plateau`, `t_chg_plateau` (즉시 추가 권장)

**근거**: 2023 Ref-based Time 논문의 직접 대응; HUST 부분 충전 강건성 확보; 구현 3줄 수준.

```python
# hi_correlation.py 내 _extract_one_cell() 충전 처리 블록에 추가
chg_rows = cyc[cyc["phase"] == "charge"]
cc_plateau = chg_rows[
    (chg_rows["voltage_V"] >= 3.10) &
    (chg_rows["voltage_V"] <= 3.45) &
    (chg_rows["current_A"].abs() > CC_THRESHOLD)   # CC 상태
]
row["q_chg_plateau"] = float(np.trapz(np.abs(cc_plateau["current_A"]),
                                       cc_plateau["time_s"]) / 3600)
row["t_chg_plateau"] = float(cc_plateau["time_s"].iloc[-1]
                              - cc_plateau["time_s"].iloc[0]) if len(cc_plateau) > 1 else np.nan
```

Global HI로 추가 시 총 **321 → 323D**.

---

### Priority 2: DVA 2차 골짜기 (MIT 전체 방전 전용, 선택적)

HUST에서는 대부분 NaN이므로 **MIT 전용 분석** 또는 **별도 피처 집합**으로 관리 권장.

```python
# dV/dQ 스무딩 후 2차 극솟값 탐색 (primary valley 오른쪽 탐색)
valley1_q_idx = np.argmin(dvdq_smooth)
search_range = dvdq_smooth[valley1_q_idx + plateau_width_samples:]
if len(search_range) > 5:
    valley2_local = np.argmin(search_range)
    row["dva_valley2_q"] = q_cum[valley1_q_idx + plateau_width_samples + valley2_local]
    row["dva_valley2_h"] = float(search_range[valley2_local])
    row["dva_valley_spacing"] = row["dva_valley2_q"] - row["dva_valley_q"]
```

Global HI로 추가 시 총 **323 → 326D** (또는 323D 유지, MIT only 메타 태그).

---

### Priority 3: 부분 IC 기울기 (선택적, 검증 후 결정)

`dqdv_slope_at_entry` 를 카테고리 B 16번째 피처로 추가하는 방안.  
피크 미가시 구간에서도 "얼마나 빨리 ICA 피크에 근접하는가"를 포착.

---

## 4. 논문 인용 전략 (논문 작성 시 참고)

```
서론 논리 흐름:

단일 계통 HI의 한계
  └── ICA/DVA (계통1): 전체 사이클 필요, 부분 충전 불가 (Berecibar 2016)
  └── 기준 전압 시간 (계통3): 부분 충전 강건하나 형상 정보 없음 (2023 Ref-based Time)
  └── 형태학적 거리 (계통2): 곡선 전체 정보 활용하나 절대 기준 필요 (DTW: JES 2020)
          ↓
  "본 연구는 4대 계통 방법론을 통합한 321D 피처 풀을 구성하고,
   동적 부분 충전 시나리오에서 SOH 추적 능력을 구간별(segment-wise)
   within-cell Spearman 상관계수로 검증한다"

인용 위치:
  카테고리 A (통계): Zheng 2018 §2.1 "partial CC segments"
  카테고리 B (미분): Berecibar 2016 (DVA), Zheng 2018 (partial IC)
  카테고리 C (LFP): Berecibar 2016 §3.2 "plateau capacity"
  카테고리 D (형태학): Hu 2021 (fusion features), JES 2020 (DTW)
  q_chg_plateau:    IEEE Trans. PD 2023 (Ref-based Time)
```

---

## 5. 결론: 우선순위 정리

| 순위 | 추가 HI | 관련 논문 | 기대 효과 | 구현 난이도 |
|------|--------|---------|---------|-----------|
| ★★★ | `q_chg_plateau`, `t_chg_plateau` | 2023 Ref-based Time | HUST 부분 충전 강건성 확보, 논문 직접 인용 가능 | 낮음 |
| ★★☆ | `dva_valley2_q/h`, `dva_valley_spacing` | Berecibar 2016 | MIT 전체 방전에서 유효, 논문 인용 | 중간 (MIT only) |
| ★☆☆ | `dqdv_slope_at_entry` | Zheng 2018 | NaN 채우기 보조, 효과 불확실 | 낮음 |
| — | V-T curve HI | Hu 2021 | temperature 데이터 복원 시 추가 | 높음 (파이프라인 변경) |
