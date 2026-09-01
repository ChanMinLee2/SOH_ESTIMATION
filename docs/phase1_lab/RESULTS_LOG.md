# Phase1 독립검증 실험 로그

`exp/phase1-isolated-lab` 브랜치에서 진행하는 Phase1(HI 랭킹) 독립 개선·검증 기록.
설계 배경은 `docs/260816_RESULTS.md` §4-10-2 이슈①(체크포인트 선택 기준과 L0 스케줄의
구조적 충돌 — k=5/15/25 랭킹이 완전히 동일하게 나온 문제)에서 출발한다.

각 실험을 돌린 뒤 아래 형식으로 이 파일에 append하고, 그 커밋에 결과 json/md
(`5_model/experiments/phase1_lab/results/`)도 같이 포함시킨다.

---

## 실험 기록 템플릿 (복사해서 사용)

### YYYY-MM-DD — <실험 태그>

- **목적**:
- **명령어**:
  ```powershell

  ```
- **결과 파일**: `5_model/experiments/phase1_lab/results/...`
- **핵심 수치**:
- **해석 / 다음 액션**:

---

## Stage 0 — 현재 구조 baseline 수렴성 측정

(여기에 첫 실험 결과를 append)

### 2026-08-19 00:37 — convergence_k25_full_N2_baseline

- **목적**: Phase1 랭킹 시드-수렴성 측정 (5개 seed, k=[5, 15, 25])
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\analyze_convergence.py --manifest C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_manifest_k25_full_N2_baseline.json --k-values 5 15 25
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_report_k25_full_N2_baseline.json`
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_report_k25_full_N2_baseline.md`
- **핵심 수치**: 평균 Kendall τ=0.0685, 평균 Jaccard(k=5: 0.524, k=15: 0.599, k=25: 0.658)
- **해석 / 다음 액션**: ⚠ 불안정 — Kendall τ 또는 Jaccard가 0.5 미만인 항목 있음. Stage1(체크포인트 기준)/Stage2(temperature annealing) 적용 후 같은 명령으로 재측정 필요.

---

### 2026-08-19 10:24 — convergence_k25_full_N2_stage12

- **목적**: Phase1 랭킹 시드-수렴성 측정 (5개 seed, k=[5, 15, 25])
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\analyze_convergence.py --manifest C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_manifest_k25_full_N2_stage12.json --k-values 5 15 25
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_report_k25_full_N2_stage12.json`
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_report_k25_full_N2_stage12.md`
- **핵심 수치**: 평균 Kendall τ=0.1188, 평균 Jaccard(k=5: 0.683, k=15: 0.745, k=25: 0.723)
- **해석 / 다음 액션**: ⚠ 불안정 — Kendall τ 또는 Jaccard가 0.5 미만인 항목 있음. Stage1(체크포인트 기준)/Stage2(temperature annealing) 적용 후 같은 명령으로 재측정 필요.

---

### 2026-08-19 10:24 — ensemble_k25_full_N2_ensemble

- **목적**: 다중 시드 앙상블 gates 합성 (Stage6, 5개 seed, Borda count)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\materialize_ensemble_gates.py --manifest C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_manifest_k25_full_N2_stage12.json --out-tag k25_full_N2_ensemble
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\ensembles\k25_full_N2_ensemble\gates\regression_HIs.json`
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\ensembles\k25_full_N2_ensemble\gates\classification_HIs.json`
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\ensembles\k25_full_N2_ensemble\PROVENANCE.json`
- **핵심 수치**: source_seeds=['0', '7', '42', '123', '2024']
- **해석 / 다음 액션**: Phase2 --gates-from C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\ensembles\k25_full_N2_ensemble 로 바로 사용 가능(가짜 run 디렉터리 — PROVENANCE.json에 출처 기록됨).

---

### 2026-08-19 10:26 — clusters_k25_full_N2_clusters

- **목적**: HI 상관 클러스터링으로 불안정성 재해석 (threshold=|r|>=0.9)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\analyze_hi_clusters.py --manifest C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_manifest_k25_full_N2_stage12.json --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1": 0.35, "n2": 0.20, "ref_lag": 0, "noise_amp": 0.03, "noise_mode": "ou", "noise_period_cycles": 200, "n_samples": 2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --datasets MIT HUST --corr-threshold 0.9 --k-values 5 15 25 --split-seed 42 --tag k25_full_N2_clusters
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\cluster_report_k25_full_N2_clusters.json`
- **핵심 수치**: 평균 raw_jaccard vs cluster_aware_jaccard 격차 = +0.036
- **해석 / 다음 액션**: 평균 Δ(cluster_aware-raw)=+0.036 — 클러스터로 보정해도 격차가 안 줄어듦 — 진짜 불안정성, Stage1/2 재점검 필요.

---

### 2026-08-19 11:16 — full_pipeline_k25_full_N2

- **목적**: Stage0-6 통합 실행 (5개 seed: [42, 0, 123, 7, 2024])
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/run_all_stages.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1": 0.35, "n2": 0.20, "ref_lag": 0, "noise_amp": 0.03, "noise_mode": "ou", "noise_period_cycles": 200, "n_samples": 2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --scen-k 25 --seeds 42 0 123 7 2024 --beta-min 0.1 --max-epochs 300 --v2-patience 60 --k-values 5 15 25 --synergy-k 15 --tag k25_full_N2 --skip-stages 0 12 3 4
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_report_k25_full_N2_baseline.md`
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_report_k25_full_N2_stage12.md`
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\cluster_report_k25_full_N2_clusters.json`
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\ensembles\k25_full_N2_ensemble`
- **핵심 수치**: 개별 하위 스크립트 로그 항목 참고(위쪽에 각각 자동 기록됨)
- **해석 / 다음 액션**: Stage6 산출물(C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\ensembles\k25_full_N2_ensemble)을 --gates-from으로 Phase2와 비교(§4 A/B/C 실험)로 다음 단계 진행.

---

### 2026-08-19 11:27 — synergy_k25_full_N2_chg_lo

- **목적**: HI 조합 시너지 검증 (scenario=chg_lo, k=15)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/analyze_hi_synergy.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --datasets MIT HUST --scenario chg_lo --k 15 --split-seed 42 --tag k25_full_N2_chg_lo
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_report_k25_full_N2_chg_lo.json`
- **핵심 수치**: baseline_val_rmse=0.04457, greedy_val_rmse=0.02749, synergy_gain=+0.01708, 초가법쌍=0/8
- **해석 / 다음 액션**: 시너지 이득 확인(개선폭 +0.01708) — 초가법 쌍 0/8개. 실제 SCRModel(HardConcreteGate)로 재검증 권장.

---

### 2026-08-19 11:30 — synergy_k25_full_N2_chg_mid

- **목적**: HI 조합 시너지 검증 (scenario=chg_mid, k=15)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/analyze_hi_synergy.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --datasets MIT HUST --scenario chg_mid --k 15 --split-seed 42 --tag k25_full_N2_chg_mid
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_report_k25_full_N2_chg_mid.json`
- **핵심 수치**: baseline_val_rmse=0.04551, greedy_val_rmse=0.03983, synergy_gain=+0.00568, 초가법쌍=0/8
- **해석 / 다음 액션**: 시너지 이득 확인(개선폭 +0.00568) — 초가법 쌍 0/8개. 실제 SCRModel(HardConcreteGate)로 재검증 권장.

---

### 2026-08-19 11:32 — synergy_k25_full_N2_chg_hi

- **목적**: HI 조합 시너지 검증 (scenario=chg_hi, k=15)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/analyze_hi_synergy.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --datasets MIT HUST --scenario chg_hi --k 15 --split-seed 42 --tag k25_full_N2_chg_hi
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_report_k25_full_N2_chg_hi.json`
- **핵심 수치**: baseline_val_rmse=0.04382, greedy_val_rmse=0.03626, synergy_gain=+0.00756, 초가법쌍=0/8
- **해석 / 다음 액션**: 시너지 이득 확인(개선폭 +0.00756) — 초가법 쌍 0/8개. 실제 SCRModel(HardConcreteGate)로 재검증 권장.

---

### 2026-08-19 11:34 — synergy_k25_full_N2_dis_hi

- **목적**: HI 조합 시너지 검증 (scenario=dis_hi, k=15)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/analyze_hi_synergy.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --datasets MIT HUST --scenario dis_hi --k 15 --split-seed 42 --tag k25_full_N2_dis_hi
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_report_k25_full_N2_dis_hi.json`
- **핵심 수치**: baseline_val_rmse=0.04180, greedy_val_rmse=0.01684, synergy_gain=+0.02496, 초가법쌍=1/8
- **해석 / 다음 액션**: 시너지 이득 확인(개선폭 +0.02496) — 초가법 쌍 1/8개. 실제 SCRModel(HardConcreteGate)로 재검증 권장.

---

### 2026-08-19 11:37 — synergy_k25_full_N2_dis_mid

- **목적**: HI 조합 시너지 검증 (scenario=dis_mid, k=15)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/analyze_hi_synergy.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --datasets MIT HUST --scenario dis_mid --k 15 --split-seed 42 --tag k25_full_N2_dis_mid
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_report_k25_full_N2_dis_mid.json`
- **핵심 수치**: baseline_val_rmse=0.04315, greedy_val_rmse=0.03492, synergy_gain=+0.00823, 초가법쌍=1/8
- **해석 / 다음 액션**: 시너지 이득 확인(개선폭 +0.00823) — 초가법 쌍 1/8개. 실제 SCRModel(HardConcreteGate)로 재검증 권장.

---

### 2026-08-19 11:39 — synergy_k25_full_N2_dis_lo

- **목적**: HI 조합 시너지 검증 (scenario=dis_lo, k=15)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/analyze_hi_synergy.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --datasets MIT HUST --scenario dis_lo --k 15 --split-seed 42 --tag k25_full_N2_dis_lo
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_report_k25_full_N2_dis_lo.json`
- **핵심 수치**: baseline_val_rmse=0.04191, greedy_val_rmse=0.02856, synergy_gain=+0.01335, 초가법쌍=0/8
- **해석 / 다음 액션**: 시너지 이득 확인(개선폭 +0.01335) — 초가법 쌍 0/8개. 실제 SCRModel(HardConcreteGate)로 재검증 권장.

---

### 2026-08-19 20:40 — convergence_head_sanity_k25_N2

- **목적**: Phase1 랭킹 시드-수렴성 측정 (3개 seed, k=[5, 15, 25])
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/analyze_convergence.py --manifest 5_model/experiments/phase1_lab/results/p1v2_runs_reg_heads/convergence_manifest_head_sanity_k25_N2_fixed.json --k-values 5 15 25
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_report_head_sanity_k25_N2.json`
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\convergence_report_head_sanity_k25_N2.md`
- **핵심 수치**: 평균 Kendall τ=0.2030, 평균 Jaccard(k=5: 0.746, k=15: 0.652, k=25: 0.658)
- **해석 / 다음 액션**: ⚠ 불안정 — Kendall τ 또는 Jaccard가 0.5 미만인 항목 있음. Stage1(체크포인트 기준)/Stage2(temperature annealing) 적용 후 같은 명령으로 재측정 필요.

---

### 2026-08-20 15:50 — synergy_groups_k25_full_N2_groups_test

- **목적**: Phase1 이전 HI 시너지 그룹 사전 구성 (편상관계수 필터 = 다중공선성 배제 + 시너지 발굴 통합)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_synergy_groups.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --split-seed 42 --tag k25_full_N2_groups_test
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_groups_k25_full_N2_groups_test.json`
- **핵심 수치**: 전체 HI 126개 -> 그룹 126개, 평균 그룹 크기 3.14
- **해석 / 다음 액션**: 평균 그룹 크기가 1에 가까우면 대부분 HI가 독립적(다중공선성/시너지 둘 다 약함), 4에 가까우면 대부분 HI가 큰 시너지 그룹으로 묶임 — Stage4 클러스터 개수(39~55/64)와 함께 보면 이 그룹 구조가 타당한지 교차검증 가능.

---

### 2026-08-20 15:57 — synergy_groups_k25_full_N2_groups_test2

- **목적**: Phase1 이전 HI 시너지 그룹 사전 구성 (편상관계수 필터 = 다중공선성 배제 + 시너지 발굴 통합)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_synergy_groups.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --split-seed 42 --tag k25_full_N2_groups_test2
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_groups_k25_full_N2_groups_test2.json`
- **핵심 수치**: 전체 HI 126개 -> 그룹 126개, 평균 그룹 크기 3.14
- **해석 / 다음 액션**: 평균 그룹 크기가 1에 가까우면 대부분 HI가 독립적(다중공선성/시너지 둘 다 약함), 4에 가까우면 대부분 HI가 큰 시너지 그룹으로 묶임 — Stage4 클러스터 개수(39~55/64)와 함께 보면 이 그룹 구조가 타당한지 교차검증 가능.

---

### 2026-08-20 16:23 — synergy_groups_k25_full_N2_groups_noleak

- **목적**: Phase1 이전 HI 시너지 그룹 사전 구성 (편상관계수 필터 = 다중공선성 배제 + 시너지 발굴 통합)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_synergy_groups.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --split-seed 42 --tag k25_full_N2_groups_noleak
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_groups_k25_full_N2_groups_noleak.json`
- **핵심 수치**: 전체 HI 120개 -> 그룹 120개, 평균 그룹 크기 3.20
- **해석 / 다음 액션**: 평균 그룹 크기가 1에 가까우면 대부분 HI가 독립적(다중공선성/시너지 둘 다 약함), 4에 가까우면 대부분 HI가 큰 시너지 그룹으로 묶임 — Stage4 클러스터 개수(39~55/64)와 함께 보면 이 그룹 구조가 타당한지 교차검증 가능.

---

### 2026-08-20 17:40 — kernel_group_features_k25_full_N2_kernel

- **목적**: 시너지 그룹을 RBF 커널 릿지로 그룹당 1개 HI로 융합 + 2차 다중공선성 배제(비선형 시너지 캡처)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_kernel_group_features.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --synergy-groups-json 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_noleak.json --split-seed 42 --tag k25_full_N2_kernel
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\kernel_group_features_k25_full_N2_kernel.pkl`
- **핵심 수치**: 후보 120개 -> 최종 64개(+패딩 0) = 64개, 평균 train R^2=0.4546, 시나리오별 개수={'chg_lo': 11, 'chg_mid': 11, 'chg_hi': 11, 'dis_hi': 11, 'dis_mid': 10, 'dis_lo': 10}
- **해석 / 다음 액션**: 이 pkl은 phase1_trainer_v2.py --kernel-features-pkl로 넘기면 train/val/test의 x_hi를 커널 융합값으로 대체해서 학습한다(N_HI 폭은 그대로라 모델 구조 무변경). 평균 train R^2가 각 그룹 멤버 HI 개별 상관보다 뚜렷이 높다면 비선형 시너지가 실제로 존재한다는 신호.

---

### 2026-08-21 11:09 — kernel_group_features_k25_full_N2_kernel_v2

- **목적**: 시너지 그룹(크기2+)을 RBF 커널로 그룹당 1개 HI로 융합(raw HI는 유지, 추가) + 2차 다중공선성 배제 + 정규화 통계 저장
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_kernel_group_features.py --model-config 5_model/config/main_qfref_S_p60.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --synergy-groups-json 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_noleak.json --split-seed 42 --tag k25_full_N2_kernel_v2
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\kernel_group_features_k25_full_N2_kernel_v2.pkl`
- **핵심 수치**: 후보 91개 -> 최종 91개, 평균 train R^2=0.3869, 시나리오별 개수={'chg_lo': 16, 'chg_mid': 15, 'chg_hi': 15, 'dis_hi': 15, 'dis_mid': 15, 'dis_lo': 15}
- **해석 / 다음 액션**: 이 pkl은 phase1_trainer_v2.py --kernel-features-pkl로 넘기면 x_hi(raw HI)는 그대로 두고 x_kernel(정규화된 커널 융합값)을 별도 게이트(scen_kernel_gates)로 추가한다 — raw HI와 커널 HI를 동시에 쓰는 게 목적. 평균 train R^2가 각 그룹 멤버 HI 개별 상관보다 뚜렷이 높다면 비선형 시너지가 실제로 존재한다는 신호.

---

### 2026-08-26 15:58 — synergy_groups_k25_full_N2_groups_v3

- **목적**: Phase1 이전 HI 시너지 그룹 사전 구성 (편상관계수 필터 = 다중공선성 배제 + 시너지 발굴 통합)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_synergy_groups.py --model-config 5_model/config/main_qfref_S.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --split-seed 42 --global-dedup --tag k25_full_N2_groups_v3
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_groups_k25_full_N2_groups_v3.json`
- **핵심 수치**: 전체 HI 150개 -> 그룹 150개, 평균 그룹 크기 2.56
- **해석 / 다음 액션**: 평균 그룹 크기가 1에 가까우면 대부분 HI가 독립적(다중공선성/시너지 둘 다 약함), 4에 가까우면 대부분 HI가 큰 시너지 그룹으로 묶임 — Stage4 클러스터 개수(39~55/64)와 함께 보면 이 그룹 구조가 타당한지 교차검증 가능.

---

### 2026-08-26 16:00 — synergy_groups_k25_full_N2_groups_vctrl

- **목적**: Phase1 이전 HI 시너지 그룹 사전 구성 (편상관계수 필터 = 다중공선성 배제 + 시너지 발굴 통합)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_synergy_groups.py --model-config 5_model/config/main_qfref_S.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --split-seed 42 --shuffle-from 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_noleak.json --shuffle-seed 42 --tag k25_full_N2_groups_vctrl
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_groups_k25_full_N2_groups_vctrl.json`
- **핵심 수치**: 전체 HI 120개 -> 그룹 120개, 평균 그룹 크기 3.20
- **해석 / 다음 액션**: 평균 그룹 크기가 1에 가까우면 대부분 HI가 독립적(다중공선성/시너지 둘 다 약함), 4에 가까우면 대부분 HI가 큰 시너지 그룹으로 묶임 — Stage4 클러스터 개수(39~55/64)와 함께 보면 이 그룹 구조가 타당한지 교차검증 가능.

---

### 2026-08-26 16:10 — synergy_groups_k25_full_N2_groups_v3

- **목적**: Phase1 이전 HI 시너지 그룹 사전 구성 (편상관계수 필터 = 다중공선성 배제 + 시너지 발굴 통합)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_synergy_groups.py --model-config 5_model/config/main_qfref_S.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --split-seed 42 --global-dedup --tag k25_full_N2_groups_v3
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_groups_k25_full_N2_groups_v3.json`
- **핵심 수치**: 전체 HI 82개 -> 그룹 82개, 평균 그룹 크기 4.68
- **해석 / 다음 액션**: 평균 그룹 크기가 1에 가까우면 대부분 HI가 독립적(다중공선성/시너지 둘 다 약함), 4에 가까우면 대부분 HI가 큰 시너지 그룹으로 묶임 — Stage4 클러스터 개수(39~55/64)와 함께 보면 이 그룹 구조가 타당한지 교차검증 가능.

---

### 2026-08-26 22:37 — synergy_groups_k25_full_N2_groups_v3

- **목적**: Phase1 이전 HI 시너지 그룹 사전 구성 (편상관계수 필터 = 다중공선성 배제 + 시너지 발굴 통합)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_synergy_groups.py --model-config 5_model/config/main_qfref_S.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --split-seed 42 --global-dedup --tag k25_full_N2_groups_v3
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_groups_k25_full_N2_groups_v3.json`
- **핵심 수치**: 전체 HI 87개 -> 그룹 87개, 평균 그룹 크기 3.11
- **해석 / 다음 액션**: 평균 그룹 크기가 1에 가까우면 대부분 HI가 독립적(다중공선성/시너지 둘 다 약함), 4에 가까우면 대부분 HI가 큰 시너지 그룹으로 묶임 — Stage4 클러스터 개수(39~55/64)와 함께 보면 이 그룹 구조가 타당한지 교차검증 가능.

---

### 2026-08-26 22:47 — synergy_groups_k25_full_N2_groups_v3

- **목적**: Phase1 이전 HI 시너지 그룹 사전 구성 (편상관계수 필터 = 다중공선성 배제 + 시너지 발굴 통합)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_synergy_groups.py --model-config 5_model/config/main_qfref_S.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --split-seed 42 --global-dedup --tag k25_full_N2_groups_v3
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_groups_k25_full_N2_groups_v3.json`
- **핵심 수치**: 전체 HI 82개 -> 그룹 82개, 평균 그룹 크기 3.05
- **해석 / 다음 액션**: 평균 그룹 크기가 1에 가까우면 대부분 HI가 독립적(다중공선성/시너지 둘 다 약함), 4에 가까우면 대부분 HI가 큰 시너지 그룹으로 묶임 — Stage4 클러스터 개수(39~55/64)와 함께 보면 이 그룹 구조가 타당한지 교차검증 가능.

---

### 2026-08-27 02:09 — kernel_group_features_k25_full_N2_kernel_v3

- **목적**: 시너지 그룹(크기2+)을 RBF 커널로 그룹당 1개 HI로 융합(raw HI는 유지, 추가) + 2차 다중공선성 배제 + 정규화 통계 저장
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_kernel_group_features.py --model-config 5_model/config/main_qfref_S.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --split-seed 42 --synergy-groups-json 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_v3.json --min-raw-partial-corr 0.02 --tag k25_full_N2_kernel_v3
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\kernel_group_features_k25_full_N2_kernel_v3.pkl`
- **핵심 수치**: 후보 61개 -> 최종 61개, 평균 train R^2=0.3631, 시나리오별 개수={'chg_lo': 8, 'chg_mid': 11, 'chg_hi': 12, 'dis_hi': 8, 'dis_mid': 11, 'dis_lo': 11}
- **해석 / 다음 액션**: 이 pkl은 phase1_trainer_v2.py --kernel-features-pkl로 넘기면 x_hi(raw HI)는 그대로 두고 x_kernel(정규화된 커널 융합값)을 별도 게이트(scen_kernel_gates)로 추가한다 — raw HI와 커널 HI를 동시에 쓰는 게 목적. 평균 train R^2가 각 그룹 멤버 HI 개별 상관보다 뚜렷이 높다면 비선형 시너지가 실제로 존재한다는 신호.

---

### 2026-08-27 14:29 — kernel_group_features_k25_full_N2_kernel_v3

- **목적**: 시너지 그룹(크기2+)을 RBF 커널로 그룹당 1개 HI로 융합(raw HI는 유지, 추가) + 2차 다중공선성 배제 + 정규화 통계 저장
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_kernel_group_features.py --model-config 5_model/config/main_qfref_S.yaml --seg-axis q_frac_ref --axis-config {"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2} --data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle --seg-data-dir D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg --split-seed 42 --synergy-groups-json 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_v3.json --min-raw-partial-corr 0.02 --tag k25_full_N2_kernel_v3
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\kernel_group_features_k25_full_N2_kernel_v3.pkl`
- **핵심 수치**: 후보 59개 -> 최종 59개, 평균 train R^2=0.3444, 시나리오별 개수={'chg_lo': 8, 'chg_mid': 11, 'chg_hi': 12, 'dis_hi': 7, 'dis_mid': 10, 'dis_lo': 11}
- **해석 / 다음 액션**: 이 pkl은 phase1_trainer_v2.py --kernel-features-pkl로 넘기면 x_hi(raw HI)는 그대로 두고 x_kernel(정규화된 커널 융합값)을 별도 게이트(scen_kernel_gates)로 추가한다 — raw HI와 커널 HI를 동시에 쓰는 게 목적. 평균 train R^2가 각 그룹 멤버 HI 개별 상관보다 뚜렷이 높다면 비선형 시너지가 실제로 존재한다는 신호.

---

### 2026-08-31 08:46 — kernel_group_features_k25_full_N2_kernel_v2

- **목적**: 시너지 그룹(크기2+)을 RBF 커널로 그룹당 1개 HI로 융합(raw HI는 유지, 추가) + 2차 다중공선성 배제 + 정규화 통계 저장
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_kernel_group_features.py --model-config 5_model/config/main_qfref_S.yaml --synergy-groups-json 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_noleak.json --split-seed 42 --tag k25_full_N2_kernel_v2
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\kernel_group_features_k25_full_N2_kernel_v2.pkl`
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\kernel_group_features_k25_full_N2_kernel_v2_rejected.json`
- **핵심 수치**: 후보 91개 -> 최종 91개, 평균 train R^2=0.3869, 시나리오별 개수={'chg_lo': 16, 'chg_mid': 15, 'chg_hi': 15, 'dis_hi': 15, 'dis_mid': 15, 'dis_lo': 15}
- **해석 / 다음 액션**: 이 pkl은 phase1_trainer_v2.py --kernel-features-pkl로 넘기면 x_hi(raw HI)는 그대로 두고 x_kernel(정규화된 커널 융합값)을 별도 게이트(scen_kernel_gates)로 추가한다 — raw HI와 커널 HI를 동시에 쓰는 게 목적. 평균 train R^2가 각 그룹 멤버 HI 개별 상관보다 뚜렷이 높다면 비선형 시너지가 실제로 존재한다는 신호.

---

### 2026-08-31 16:05 — kernel_group_features_k25_full_N2_kernel_vctrl

- **목적**: 시너지 그룹(크기2+)을 RBF 커널로 그룹당 1개 HI로 융합(raw HI는 유지, 추가) + 2차 다중공선성 배제 + 정규화 통계 저장
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_kernel_group_features.py --model-config 5_model/config/main_qfref_S.yaml --split-seed 42 --synergy-groups-json 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_vctrl.json --tag k25_full_N2_kernel_vctrl
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\kernel_group_features_k25_full_N2_kernel_vctrl.pkl`
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\kernel_group_features_k25_full_N2_kernel_vctrl_rejected.json`
- **핵심 수치**: 후보 91개 -> 최종 91개, 평균 train R^2=0.3753, 시나리오별 개수={'chg_lo': 16, 'chg_mid': 15, 'chg_hi': 15, 'dis_hi': 15, 'dis_mid': 15, 'dis_lo': 15}
- **해석 / 다음 액션**: 이 pkl은 phase1_trainer_v2.py --kernel-features-pkl로 넘기면 x_hi(raw HI)는 그대로 두고 x_kernel(정규화된 커널 융합값)을 별도 게이트(scen_kernel_gates)로 추가한다 — raw HI와 커널 HI를 동시에 쓰는 게 목적. 평균 train R^2가 각 그룹 멤버 HI 개별 상관보다 뚜렷이 높다면 비선형 시너지가 실제로 존재한다는 신호.

---

### 2026-08-31 16:26 — synergy_groups_k25_full_N2_groups_v3_check

- **목적**: Phase1 이전 HI 시너지 그룹 사전 구성 (편상관계수 필터 = 다중공선성 배제 + 시너지 발굴 통합)
- **명령어**:
  ```powershell
  C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe 5_model/experiments/phase1_lab/build_synergy_groups.py --model-config 5_model/config/main_qfref_S.yaml --split-seed 42 --global-dedup --tag k25_full_N2_groups_v3_check
  ```
- **결과 파일**:
  - `C:\Users\ksshin\Desktop\ChanminLee\SOH_ESTIMATION\5_model\experiments\phase1_lab\results\synergy_groups_k25_full_N2_groups_v3_check.json`
- **핵심 수치**: 전체 HI 82개 -> 그룹 82개, 평균 그룹 크기 3.05
- **해석 / 다음 액션**: 평균 그룹 크기가 1에 가까우면 대부분 HI가 독립적(다중공선성/시너지 둘 다 약함), 4에 가까우면 대부분 HI가 큰 시너지 그룹으로 묶임 — Stage4 클러스터 개수(39~55/64)와 함께 보면 이 그룹 구조가 타당한지 교차검증 가능.

---
