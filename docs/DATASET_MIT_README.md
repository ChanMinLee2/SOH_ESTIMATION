# MIT FastCharge — MAT 파일 변환 데이터

소스: Severson et al. (2019), 원본 .mat (HDF5) 파일
변환: build_batch_pkl.py

## 셀 수
- batch1: 41개  (b1c0 ~ b1c44, b1c8/10/12/13/22 제외)
- batch2: 43개  (b2c0 ~ b2c47, b2c7/8/9/15/16 → b1c0-4에 병합)
- batch3: 40개  (b3c0 ~ b3c46, b3c2/23/32/37/42/43 제외)
- 합계:   124개

## 파일 형식
| 파일 | 내용 |
|------|------|
| bNcN.pkl | {"meta": dict, "cycles": DataFrame} |
| bNcN.csv | cycles DataFrame |
| conversion_summary.csv | 셀별 변환 통계 |

## DataFrame 컬럼
| 컬럼 | 단위 | 설명 |
|------|------|------|
| cycle | - | 사이클 번호 (0 제외) |
| time_s | s | 셀 전체 누적 경과 시간 |
| voltage_V | V | 전압 |
| current_A | A | 전류 (양수 = 충전, 음수 = 방전) |
| temperature_C | °C | 온도 |
| capacity_Ah | Ah | 해당 사이클 방전 용량 |
| phase | - | charge / discharge / rest |

## 이상치 처리
Rolling Median 필터 (window=11, σ=2.5): RPT·HPPC 진단 사이클 자동 제거

---

## 충전 프로토콜

### 기본 구조

`C1(Q1)-C2` 형식의 fast charging 후 1C CC-CV로 마무리.

| 단계 | 내용 |
|------|------|
| Step 1 | C1 전류로 CC 충전 → Q1% SOC 도달 시 C2로 전환 (일부 셀은 단일 CC) |
| Step 2 | C2 전류로 CC 충전 → **80% SOC** 도달까지 |
| Step 3 | **1C (≈1.1A) CC-CV** → 3.6V 도달 후 CV, 전류 컷오프까지 |
| 방전 | 4C CC, 2.0V 컷오프 |

상·하한 전압: 3.6V / 2.0V (전 단계 공통 적용).

**SOC 기준**: Arbin 장비가 직전 사이클 방전 용량을 FCC로 사용하는 쿨롱 카운팅 방식.  
건강한 초기 셀에서 실측 방전 용량 ≈ 1.00 Ah → 80% SOC 타겟 = **0.80 Ah**.  
(공칭 용량 1.1 Ah 기준 72.7% ≠ 80% — 공칭 용량 기준이 아님)

---

### 충전 중 두 개의 전환점

충전 1사이클 안에 서로 다른 두 전환이 존재한다.

```
전환 1: fast CC  →  1C CC   (컨트롤러 타겟: Q ≈ 0.80 Ah = 80% SOC)
전환 2: 1C CC   →  3.6V CV  (전압 도달 시점: 열화에 따라 크게 변동)
```

| 전환 | 기준 | 비고 |
|------|------|------|
| fast CC → 1C CC | Q ≈ 0.80 Ah (SOC 80%) | Arbin이 고정 타겟으로 제어 |
| 1C CC → 3.6V CV | 전압 cutoff 도달 시점 | 내부저항 R↑에 따라 사이클마다 달라짐 |

---

### 열화 단계별 충전 동작 (b3c44 실측 예시)

| 단계 | 사이클 | Q_total | 3.6V CV 시작 Q | q_frac | 구간 |
|------|--------|---------|--------------|--------|------|
| 초기 | 2 ~ 622 | ~0.99 Ah | ~0.95 Ah | 0.96 ~ 0.97 | chg_hi |
| 중기 | 684 ~ 870 | ~0.91 Ah | 0.71 ~ 0.80 Ah | 0.78 ~ 0.83 | chg_hi |
| 말기 | 932+ | ~0.875 Ah | ~0.45 Ah | ~0.51 | **chg_mid** |

**초기 (사이클 1~620)**
```
fast CC (5C) ─→ [Q=0.80Ah, 80% SOC] ─→ 1C CC ─→ [Q≈0.95Ah, 3.6V 도달] ─→ CV
                 (컨트롤러 전환)                    (실제 CV 진입, q_frac≈0.97)
```
- R이 낮아 1C CC 구간에서 약 0.15 Ah를 추가 충전한 뒤 3.6V에 도달
- CV 시작점이 chg_hi 후반부에 위치

**중기 (사이클 680~870)**
```
fast CC (5C) ─→ [Q=0.80Ah] ─→ 1C CC ─→ [거의 즉시 3.6V] ─→ CV
```
- R↑ → 1C CC 시작 직후 3.6V cutoff 도달
- CV 시작점이 chg_hi 초입(q_frac ≈ 0.80~0.83)으로 이동

**말기 (사이클 930+)**
```
fast CC (5C) ─→ [Q≈0.45Ah에서 3.6V cutoff — 0.80Ah 타겟 미달] ─→ CV
```
- R↑ → fast CC 구간 자체가 3.6V에 먼저 걸림
- 컨트롤러 타겟(0.80 Ah) 도달 전 강제 CV 진입
- CV 시작점이 **chg_mid** (q_frac ≈ 0.51)로 진입

> 이 현상이 `v_range_chg_mid` HI에서 열화 신호가 나타나는 직접 원인.  
> 데이터셋 설명 원문: *"after some cycling, the cells may hit the upper cutoff potential  
> during fast charging, leading to significant constant-voltage charging."*
