# HI 계산·추가 가이드 (`5_model/hi_compute.py`)

`hi_compute.py`는 **자립형 단일 소스 HI 라이브러리**다. 모든 HI가 창 객체 `W`를 받아
스칼라를 반환하는 **독립 `@hi` 함수**이며, 함수 하나만 추가하면 계산·비용측정·모델 반영이
자동으로 이뤄진다. 스키마 파일도, 차원 지정도 없다.

- **66개 HI** = stat 20 + diff 20 + lfp 20 + morph 6
- 수식·곡선 헬퍼가 전부 이 파일에 있고, `4_hi_analysis/hi_correlation.py`는 이걸 **import해
  쓰는 소비자**다 (세그먼트 분석은 `_seg_stat/_seg_diff/_seg_lfp` 어댑터가 개별 함수를 호출).
  → **중복 없는 단일 소스**, 순환 참조 없음.

---

## 1. HI 함수 형태

각 HI는 `@hi`로 등록되는 개별 함수다. **함수 이름 = HI 이름.**

```python
@hi
def stat_v_mean(w):       # w.v/i/t + 캐시된 w.ai, w.dt, w.q, w.q_rel, w.dQ, w.vrange
    den = float(np.sum(w.ai * w.dt))
    return float(np.sum(w.v * w.ai * w.dt)) / den if den > 1e-9 else float(np.mean(w.v))

@hi
def diff_dqdv_area(w):    # 무거운 공유 곡선은 W가 1회만 계산 → 함수는 읽기만
    vmids, dqdv = w.ica   # w.ica = dQ/dV 곡선 (cached_property)
    return float(np.trapezoid(np.maximum(dqdv, 0), vmids)) if len(vmids) >= 4 else np.nan
```

- 값이 정의되지 않는 창에서는 `np.nan`을 반환한다 → 로더가 mask=0으로 처리해 그 창에선 무시.
- 예외가 나도 `compute_his`가 잡아 `np.nan` 처리하므로 방어 코드 없이 수식만 써도 된다.
- 등록 목록/개수는 `hi_names()`로 **동적** → 모델의 HI 수 = 데이터의 `hi_` 컬럼 수(자동).

---

## 2. `W` 객체와 공유 캐시 (핵심 설계)

여러 HI가 재사용하는 무거운 중간값은 `W`에 **`cached_property`로 한 번만 계산**된다. 개별
함수로 쪼개도 재계산이 0이다.

| 속성 | 내용 |
|------|------|
| `w.v` / `w.i` / `w.t` | 창의 전압 / 부호있는 전류 / 시간 배열 |
| `w.ai` | 전류 크기 `|i|` |
| `w.dt` | 시간 간격 `diff(t)` |
| `w.q` / `w.q_rel` | 누적 전하 [Ah] / 창 시작 기준 상대 전하 |
| `w.dQ` / `w.vrange` | 구간 총 전하 / 전압 범위 |
| **`w.vq`** | **V-Q 곡선** `(qm, v_sm, dvdq_sm, q_tot)` — diff·lfp 다수가 공유 |
| **`w.ica`** | **dQ/dV 곡선** `(vmids, dqdv_sm)` — dQ/dV 계열 HI가 공유 |
| `w.t_norm` | 정규화 시간 [0,1] |

> 새 HI가 dV/dQ·dQ/dV 곡선이 필요하면 **직접 계산하지 말고 `w.vq`/`w.ica`를 읽어라.**
> 공유 term이 하나 더 필요하면 `W`에 `@cached_property`로 추가하면 모든 HI가 재사용한다.

---

## 3. HI 그룹 (66개)

| 그룹 | 개수 | 내용 | 접두사 |
|------|------|------|--------|
| **stat** | 20 | 전압·전류 통계 (평균/표준편차/왜도/첨도/엔트로피/상관/샘플엔트로피 …) | `stat_` |
| **diff** | 20 | 미분 기반 (dV/dQ·dQ/dV 곡선의 피크/면적/기울기/비대칭 …) | `diff_` |
| **lfp** | 20 | LFP 특징 (플래토 비율/전압/기울기, knee, 비선형지수 …) | `lfp_` |
| **morph** | 6 | 형태 거리 (창 곡선 vs 셀 신품(BOL) 곡선의 DTW/Fréchet) | `morph_` |

`stat`/`diff`/`lfp` 60개는 창의 V/I/t만으로 계산되는 관측 가능 지표다.

---

## 4. HI 추가 방법

1. `5_model/hi_compute.py`에 `@hi` 함수 작성:
   ```python
   @hi
   def my_feature(w):
       # 예: V-Q 곡선의 3차 다항 계수
       return float(np.polyfit(w.q, w.v, 3)[0]) if w.dQ > 1e-6 else np.nan
   ```
2. 데이터 재빌드 (새 HI 계산·저장 + 실측 계산비용 측정):
   ```bash
   python 5_model/build_dataset.py
   ```
3. 학습 — **코드·스키마 수정 없이** 새 HI가 자동 포함된다 (`n_hi`가 컬럼 수로 자동 조정):
   ```bash
   python 5_model/train.py --run-name my_run
   ```

확인:
```bash
python 5_model/hi_compute.py        # 등록된 HI 개수·그룹·비용 상위/하위 5개
```

---

## 5. morph — 신품(BOL) 곡선 참조가 필요한 HI

morph 6개(`morph_vt_dtw` … `morph_ve_frec`)는 창 곡선을 **그 셀의 신품 상태(첫 30사이클) 곡선**과
DTW/Fréchet 거리로 비교한다. 따라서 셀별 BOL 곡선 참조가 필요하다:

- `compute_his(v, i, t, bol=참조곡선)` 로 전달 — `bol`이 없으면 morph 6개는 `NaN`(마스킹).
- `build_dataset.py`가 셀마다 `_build_bol_ref`로 방향별(충/방) BOL 곡선을 만들어 넘긴다.
- 신품 곡선은 공장 BOL 데이터라 정당한 참조다(정답 누출 아님).

> 참고: 이 창 과제에서 morph는 실험상 정확도 이득이 없었다(창-vs-전체BOL 거리가 창 위치에
> 좌우되는 노이즈). 코드는 유지하되, 필요 없으면 `bol` 없이 호출하면 자동 마스킹된다.

---

## 6. 실측 계산비용 측정

`benchmark_cost(samples)`가 각 HI 함수를 표본 창들에 여러 번 실행해 **한계 계산시간(μs)** 을
측정한다 (공유 곡선은 예열해 계측에서 제외 → 함수 자체의 한계 비용). `build_dataset.py`가 이를
`_4_data_hi/samples/hi_cost.json`에 저장한다.

- 예: `stat_v_samp_ent ≈ 1700μs` (O(n²) 샘플엔트로피) vs 단순 읽기 `≈ 3μs` → 최대 수백 배 차이.
- 이 실측 비용은 모델의 비용가중 희소화(L0)에서 "싸고 예측적인 HI"를 선호하는 데 쓰인다.

---

## 7. 파일 위치

```
5_model/hi_compute.py        HI 라이브러리(자립·단일소스): @hi 개별함수 66 + 곡선헬퍼
                               + W(공유캐시) + morph + benchmark_cost      ← HI 추가는 여기
5_model/build_dataset.py     각 창의 HI 계산·저장 + 셀별 BOL 참조 구성 + 계산비용 측정
4_hi_analysis/hi_correlation.py   hi_compute를 import하는 소비자(세그먼트 분석 어댑터)
_4_data_hi/samples/hi_cost.json   HI 함수별 실측 계산시간(μs)
```
