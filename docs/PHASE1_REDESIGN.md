# Phase1(HI 랭킹) 수정 설계안 — 쉽게 설명하는 버전

> 관련 브랜치: `exp/phase1-isolated-lab` (`main`은 안 건드림)
> 관련 코드: `5_model/experiments/phase1_lab/`
> 관련 로그: `docs/phase1_lab/RESULTS_LOG.md`

## 0. 한 줄 요약

**"어떤 HI(배터리 특징값)를 몇 개 골라 쓸지"를 정하는 Phase1이, 사실은 거의 아무 조건도
안 걸린 상태에서 순위를 정하고 있었다.** k(몇 개 고를지)를 5로 하든 15로 하든 25로 하든
**완전히 똑같은 순위**가 나왔고, 그 순위 자체도 시드(random seed)만 바꿔도 top-5 중
2~3개가 뒤바뀔 만큼 불안정했다. 이 문서는 왜 이렇게 됐는지, 그리고 어떻게 고칠지를
정리한다.

## 1. 무슨 문제가 있었나

Phase1은 "HardConcreteGate"라는 학습 가능한 스위치를 HI(64~66개)마다 하나씩 붙여서,
학습이 끝났을 때 스위치가 많이 켜진(=중요한) HI 순서대로 랭킹을 뽑는 구조다. 이때
"너무 많은 HI를 쓰지 마라"고 압박을 주는 페널티(L0 penalty, `lambda_l0`)가 있고, 이
페널티의 세기는 `scen_k`(최종적으로 몇 개를 쓸지)에 따라 자동으로 조절되도록 설계돼
있었다.

그런데 실제로 k=5/15/25로 각각 Phase1을 돌려서 나온 랭킹 JSON을 비교해보니:

```
seg_0 (chg_lo): k=5/15/25 top5 전부 [26, 23, 55, 45, 60]  ← 완전히 동일
seg_1 (chg_mid): k=5/15/25 top5 전부 [26, 57, 41, 42, 30]  ← 완전히 동일
... (6개 시나리오 전부 동일)
```

**k를 바꿨는데도 랭킹이 소수점 하나까지 안 바뀌었다.** 즉 "k에 따라 페널티 세기가
달라지고, 그래서 다른 학습 결과가 나온다"는 설계 의도가 실제로는 전혀 작동하지 않고
있었다.

## 2. 왜 이런 일이 생겼나 (원인)

세 가지가 맞물린 결과다.

1. **학습 초반 50 epoch은 페널티가 0이다.** `lambda_l0_schedule: delayed_warmup`,
   `warmup_epochs: 50` 설정 때문에, epoch 50이 되기 전까지는 "너무 많이 쓰지 마라"는
   압박이 아예 걸리지 않는다. 이 구간에서는 k값이 얼마든 학습이 완전히 똑같이
   진행된다(순수 정확도만 좇는 학습).
2. **"가장 성능 좋은 시점"을 저장하는 방식이라, 페널티가 걸리기 시작하면 그 이후는
   저장될 일이 없다.** 페널티는 "정확도를 조금 희생해서라도 적게 써라"는 압박이라,
   걸리는 순간부터 정의상 검증 성능(val RMSE)이 좋아질 수가 없다. 그래서 시스템은
   거의 항상 "페널티가 막 걸리기 시작한 epoch 50" 근처를 최종 결과로 저장한다.
3. **그 결과, 저장된 랭킹은 사실상 "페널티가 있기 전"의 스냅샷이다.** k가 페널티
   세기에 영향을 주게 설계돼 있어도, 그 페널티가 한 번도 제대로 작동할 기회를 못
   가지니 k별 차이가 안 생긴다.

비유하자면, "시험 난이도(k)에 따라 공부량을 조절해서 최적의 학생을 뽑겠다"고
설계해놓고, 정작 **난이도가 적용되기 전에 이미 합격자를 확정지어버리는** 상황과
비슷하다.

시드 불안정성도 같은 뿌리다 — 페널티가 없으니 게이트들이 확실하게 "켜짐/꺼짐"으로
갈리지 않고 애매한 중간값에 많이 몰려있는데, 이 상태에서는 비슷하게 유용한 HI들끼리
누가 근소하게 앞서는지가 학습 중의 무작위 노이즈(시드)에 따라 쉽게 뒤집힌다.

## 3. 개선 설계 — Phase1 독립검증 로드맵

Phase2와 섞기 전에, **Phase1 혼자서도 "안정적이고 믿을 만한 랭킹"을 내놓는지**부터
증명하기로 했다. 단계마다 통과 기준(숫자)을 두고, 그걸 넘어야 다음 단계로 간다.

| 단계 | 내용 | 확인 방법 |
|---|---|---|
| **0. 기준선 측정** | 지금 상태 그대로 시드 5개(42/0/123/7/2024) 반복 실행 | 랭킹이 시드마다 얼마나 다른지 수치화(아래 지표) — 이후 "개선됐다"고 말할 근거 |
| **1. 저장 기준 분리** | "검증 성능 최고점"이 아니라 "게이트가 실제로 켜짐/꺼짐으로 확실히 갈린 시점"을 저장 기준으로 변경 | 게이트 확률이 애매한 중간값(0.1~0.9)에 남은 비율 — 줄어들어야 함 |
| **2. 이산화 압력 강화** | 학습이 진행될수록 게이트를 더 확실하게 켜짐/꺼짐으로 밀어붙이는 장치(temperature annealing) 추가 | 위와 동일 지표로 추가 개선 확인 |
| **3. 다중 시드 앙상블** | 시드 하나만 믿지 않고 5~10개 시드 결과를 합쳐서 "여러 시드가 공통으로 뽑은 HI"를 최종 랭킹으로 채택 (통계학의 stability selection과 동일 아이디어) | 시드 간 랭킹 일치도(Jaccard/Kendall) — baseline 대비 확실히 개선돼야 함 |
| **4. 유사 HI 정리** | 서로 거의 같은 정보를 담은 HI들을 미리 묶어서, "둘 중 아무거나 뽑혀도 문제없다"는 걸 구분 | 랭킹이 흔들리는 게 진짜 불안정인지, 그냥 동급 HI끼리의 무해한 교체인지 재확인 |
| **5. HI 조합 시너지 검증** | "개별로는 별로여도 같이 쓰면 좋은 HI 쌍"을 찾는 별도 방법 도입(상호정보량 스크리닝 → 조합 인식형 순위 → 실제 시너지 검증) | 조합 인식형 랭킹이 기존 방식보다 val RMSE를 실제로 낮추는지 |
| **6. 최종 산출물 확정** | 위 검증을 통과한 랭킹 + 그 검증 수치를 함께 저장 | — |

## 4. Phase1 → Phase2 통합 전략

Phase1이 6단계를 통과해 "믿을 만하다"고 확인된 뒤에는, Phase2와 합치는 방식을
**감이 아니라 실험으로** 정한다. 세 가지 방식을 같은 조건(seed/split)으로 비교한다.

| 방식 | 설명 |
|---|---|
| **A. 완전 분리(현재 방식)** | Phase1이 고른 HI 목록을 그대로 고정(freeze)하고, Phase2는 그 위에서 예측만 다시 학습 |
| **B. 부분 결합** | "어떤 HI를 쓸지"는 A처럼 고정하되, 그 HI들 각각의 가중치는 Phase2에서도 계속 조정 허용 |
| **C. 완전 결합** | Phase2 학습 후반에 페널티를 약하게 다시 걸어서 게이트 자체도 재조정 허용(단, 너무 많이 안 늘어나게 상한 유지) |

**기본 방침은 A(완전 분리) 유지**다 — 이 프로젝트의 "몇 개의 HI로 몇 %의 성능을
낼 수 있는가"(비용-효율 비교, k=5/15/25/65 실험들)라는 전체 이야기가 "한번 고른
HI 세트는 안 바뀐다"는 전제 위에 있기 때문이다. 다만 B/C가 A보다 확실히 더 잘하는지는
실제로 측정해서, 이득이 크면 제한적으로 받아들이는 쪽으로 정한다.

```
Phase1 6단계 통과
      │
      ▼
A vs B vs C 3자 비교 (같은 seed/split)
      │
      ├─ A가 이기거나 비슷 → 지금처럼 완전 분리 유지
      └─ B/C가 확실히 이김 → 그 이득만큼만 제한적으로 결합 구조 도입
```

## 5. 지금 만들어진 실험 환경

기존 코드(`train_scr.py`, `run_pipeline.py`)는 전혀 건드리지 않고, 별도 브랜치와
독립 스크립트로 검증만 먼저 진행한다.

- **브랜치**: `exp/phase1-isolated-lab`
- **`5_model/experiments/phase1_lab/run_convergence_seeds.py`**: 여러 시드로 Phase1을
  반복 실행(`--trainer baseline`=기존 `train_scr.py` 그대로 / `--trainer v2`=
  아래 `phase1_trainer_v2.py`), `--parallel`/tqdm 지원
- **`phase1_trainer_v2.py`**: Stage1(체크포인트 기준=게이트 포화도, L0 완전 램프
  이후 구간에서만 후보 인정)+Stage2(temperature annealing) 적용한 독립 트레이너.
  `--max-epochs`/`--patience`(결과 영향 없이 시간만 절약)/`--batch-size`(영향
  가능성 있어 opt-in)/`--regression-model`(기본 mlp — §4-1 sanity check용 오버라이드)
- **`run_head_comparison.py`**: 같은 seed로 cap_head 종류만 바꿔가며
  `phase1_trainer_v2.py` 반복 실행 (§4-1 sanity check 전용)
- **`analyze_convergence.py`**: 여러 run의 랭킹 일치도(Jaccard/Kendall τ) 계산 +
  다중시드 앙상블 랭킹 산출 — seed 비교/헤드 비교 양쪽에 재사용
- **`analyze_hi_synergy.py`**: HI 조합 시너지 스크리닝(상호정보량 → 조합 인식형
  순위 → 초가법성 검증), `--workers` 병렬화
- **`analyze_hi_clusters.py`**: 상관 클러스터링으로 랭킹 불안정성 재해석(Stage4)
- **`materialize_ensemble_gates.py`**: 다중시드 앙상블을 Phase2 `--gates-from`이
  바로 먹는 합성 run 디렉터리로 저장(Stage6, `PROVENANCE.json`으로 출처 명시)
- **`run_all_stages.py`**: Stage0→1+2→3→4→5를 한 번에 실행하는 마스터
  오케스트레이터
- **`docs/phase1_lab/RESULTS_LOG.md`**: 실험 기록 누적 로그 — 위 분석 스크립트들이
  실행 끝에 자동으로 append

## 6. 진행 상태 / 다음 단계

- [x] 문제 진단 (k=5/15/25 랭킹 동일 문제 실증 확인)
- [x] 로드맵 설계 (§3, §4)
- [x] 독립 실험 환경 구축 (§5)
- [ ] §4-1 Phase1 cap_head 종류 sanity check (**다음 할 일** — §7 명령어 참고)
- [ ] Stage 0~6 순차 적용 및 검증 (§7 명령어 참고)
- [ ] Phase2 통합 방식 A/B/C 비교 실험

## 7. 실행 절차 — 헤드 sanity check → Stage0-6 통합 검증

세션마다 먼저 설정:
```powershell
$env:SOH_EXCLUDE_STAT_LEAK="1"
```

`n_samples=2`(n1=35%/n2=20%)로 진행하기로 함 — 이 조합의 seg pkl 캐시가 아직
없으므로(`n1-35%_n2-20%_N-4_...`만 존재) **§7-1 전에 Step4 추출이 먼저 필요**하다.

### 7-0. (사전 준비) n_samples=2 캐시 추출

```powershell
python run_pipeline.py 4 --to-step 4 --force-extract --seg-axis q_frac_ref `
  --axis-config '{\"n1\": 0.35, \"n2\": 0.20, \"ref_lag\": 0, \"noise_amp\": 0.03, \"noise_mode\": \"ou\", \"noise_period_cycles\": 200, \"n_samples\": 2}' `
  --workers 8
```
완료되면 `.../q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/{cycle,seg}/`가 생긴다
(`--data-dir`/`--seg-data-dir`을 아래처럼 그 경로로 맞춰야 함).

### 7-1. Phase1 cap_head 종류 sanity check (§4-1 근거)

```powershell
python 5_model/experiments/phase1_lab/run_head_comparison.py `
  --model-config 5_model/config/main_qfref_S.yaml `
  --seg-axis q_frac_ref `
  --axis-config '{\"n1\": 0.35, \"n2\": 0.20, \"ref_lag\": 0, \"noise_amp\": 0.03, \"noise_mode\": \"ou\", \"noise_period_cycles\": 200, \"n_samples\": 2}' `
  --data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle" `
  --seg-data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg" `
  --scen-k 25 --seed 42 --heads mlp resnet_tab transformer `
  --max-epochs 300 --patience 60 --beta-min 0.1 --tag head_sanity_k25_N2

python 5_model/experiments/phase1_lab/analyze_convergence.py `
  --manifest 5_model/experiments/phase1_lab/results/convergence_manifest_head_sanity_k25_N2.json `
  --k-values 5 15 25
```

**예상 시간**: epoch **개수** 자체는 n_samples와 무관(L0 스케줄이 좌우 — 최소
150+patience60=210, 최대 `--max-epochs`(300))하지만, **epoch당 시간은 데이터
행 수에 비례**한다. `n_samples=2`는 zone당 세그먼트가 절반(6zone×2=12개/cycle,
기존 24개/cycle의 절반)이라 train 행 수도 대략 절반(≈1.68M) — 실측
≈68초/epoch(N-4, N_HI=64 기준)의 **≈절반인 ≈34초/epoch**로 추정:

| | epoch 범위 | 헤드당 시간 |
|---|---|---|
| 최소 | 210 | ≈2.0시간 |
| 최대(cap) | 300 | ≈2.8시간 |

**3개 헤드 순차 합계 ≈ 6~8.5시간.**(N-4 기준이던 12~17시간의 절반)

### 7-2. Stage0-6 통합 검증 (7-1에서 헤드 영향이 미미하다고 확인된 뒤)

```powershell
python 5_model/experiments/phase1_lab/run_all_stages.py `
  --model-config 5_model/config/main_qfref_S_p60.yaml `
  --seg-axis q_frac_ref `
  --axis-config '{\"n1\": 0.35, \"n2\": 0.20, \"ref_lag\": 0, \"noise_amp\": 0.03, \"noise_mode\": \"ou\", \"noise_period_cycles\": 200, \"n_samples\": 2}' `
  --data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle" `
  --seg-data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg" `
  --scen-k 25 --seeds 42 0 123 7 2024 --beta-min 0.1 `
  --max-epochs 300 --v2-patience 60 `
  --k-values 5 15 25 --synergy-k 15 --tag k25_full_N2
```

**예상 시간** (≈34초/epoch, `n_samples=2` 기준, seed 5개 순차):

| 단계 | seed당 epoch | 계산 | 소요 시간 |
|---|---|---|---|
| Stage0 (baseline, `_p60.yaml`이라 patience=60) | best≈50 + patience60 ≈ 110 | 5 × 110 × 34s | ≈5.2시간 |
| Stage1+2 (v2, §7-1과 동일 210~300 범위, 평균≈250 가정) | ≈250 | 5 × 250 × 34s | ≈11.8시간 |
| Stage3 (앙상블 합성) | — | JSON 처리만 | ≈수초 |
| Stage4 (클러스터링, 6시나리오) | — | 데이터 로드+상관행렬 (행수 절반이라 더 빠름) | ≈3~7분 |
| Stage5 (시너지, 6시나리오) | — | 시나리오당 재로드+MI+greedy (행수 절반) | ≈20~40분 |
| **합계** | | | **≈17.5~18시간 (약 하루 이내)** |

7-1(≈6~8.5h) + 7-2(≈17.5~18h)를 순서대로 다 돌리면 총 **≈24~26.5시간(약 1일)**
— 7-0 추출 시간(수십 분~1시간 내외, `--workers 8` 기준)은 별도. `n_samples=4`
대비 대략 절반으로 줄었다. GPU가 1개뿐이라 두 단계를 동시에 돌리면 안 되고
(§5 `run_convergence_seeds.py` `--parallel` 경고 참고), 7-0→7-1→7-2 순서를 지킬 것.
