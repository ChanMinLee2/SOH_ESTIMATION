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
