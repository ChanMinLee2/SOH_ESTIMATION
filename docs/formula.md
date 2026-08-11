# 논문용 수식 정리

> 이 문서의 모든 수식은 **실제 코드를 직접 대조해 작성**했다(추정/일반화 아님). 코드가
> 바뀌면 이 문서도 같이 갱신해야 한다. 각 절 제목에 해당 소스 파일을 명시했다.

---

## 0. 표기법

| 기호 | 의미 |
|---|---|
| $t$ | 세그먼트/사이클 내 시간 인덱스 |
| $V(t)$, $I(t)$ | 전압[V], 전류[A] (부호: $I>0$=충전, $I<0$=방전) |
| $\Delta t$ | 샘플 간격[s] |
| $Q(t)$ | 누적 전하량(쿨롱카운팅)[Ah] |
| $N_{HI}$ | 세그먼트당 HI 피처 수 (기본 66, `SOH_EXCLUDE_STAT_LEAK=1`이면 64) |
| $s$ | 시나리오(존) 인덱스, $s \in \{$dis_lo, dis_mid, dis_hi, chg_lo, chg_mid, chg_hi$\}$ |

---

## 1. 전처리 — 쿨롱카운팅 (`5_model/hi_compute.py`, `common/scenario/*.py` 공통)

$$
Q(t) = \sum_{k=1}^{t} |I(t_k)| \cdot \Delta t_k \Big/ 3600
$$

세그먼트 내 상대 위치(분율):

$$
q_{frac}(t) = \frac{Q(t)}{Q_{tot}}, \qquad Q_{tot} = Q(t_{end})
$$

---

## 2. 세그멘테이션(시나리오) 축

### 2.1 `q_frac_wide` — 존 경계 (`common/scenario/q_frac_wide.py`)

존 폭 파라미터 $n_1 \in (0,1)$(zone 크기), $n_2 \in (0, n_1)$(세그먼트 길이):

$$
\text{hi} = [0, n_1], \quad
\text{mid} = \left[\tfrac{1-n_1}{2},\ \tfrac{1+n_1}{2}\right], \quad
\text{lo} = [1-n_1,\ 1]
$$

($n_1 > 1/3$이면 존끼리 겹칠 수 있음 — 설계상 허용)

세그먼트는 각 존 안에서 길이 $n_2$인 구간을 $n_{samples}$개 샘플링(구간 내 겹침 허용).
세그먼트 최소 관측 포인트 수 제약(2026-08-10 신설):

$$
n_{points}(\text{segment}) \geq \texttt{min\_pts} \quad (\text{기본 } 10,\ \text{미달 시 세그먼트 폐기})
$$

### 2.2 `q_frac_ref` — 과거 레퍼런스 + 노이즈 (`common/scenario/q_frac_ref.py`)

$q_{frac\_wide}$와 존/라우팅은 동일, 분모($Q_{tot}$)만 아래로 대체:

$$
Q_{ref}(t) = Q_{ref,raw}(t)\cdot\bigl(1+\eta(t)\bigr)
$$

$$
Q_{ref,raw}(t) =
\begin{cases}
Q_{cyc}(t) & \text{lag} \le 0 \text{ 또는 콜드스타트 (이력 부족)} \\[2pt]
Q_{cyc}(t-\text{lag}) & \text{lag} \ge 1
\end{cases}
$$

여기서 $Q_{cyc}(t)$는 **그 사이클 자신의(또는 lag 사이클 전의) 실측 누적 방전용량**이다 —
BOL(첫 사이클) 고정값이 **아니다**. 즉 노이즈 진폭(`noise_amp`)은 항상 "그 레퍼런스 시점의
실측값" 기준 상대 비율이며, 셀이 노화될수록 주입되는 절대 Ah 크기도 함께 줄어든다.

**노이즈 $\eta(t)$** — 셀·방향별 고정 바이어스 + 느린 드리프트, $[-a, a]$로 클리핑
($a=$ `noise_amp`):

$$
\eta(t) = \operatorname{clip}\bigl(b + d(t),\ -a,\ a\bigr), \qquad
b \sim \mathcal{U}(-0.5, 0.5)\cdot a \ \ (\text{셀·방향별 1회 고정})
$$

드리프트 $d(t)$ — OU(Ornstein-Uhlenbeck) 이산화(AR(1)), 기본 모드:

$$
d(t) = \phi \cdot d(t-1) + \sigma\sqrt{1-\phi^2}\cdot \varepsilon_t, \qquad
\varepsilon_t \sim \mathcal{N}(0,1)
$$

$$
\phi = e^{-1/\tau}, \qquad \sigma = a/4
$$

$\tau=$ `noise_period_cycles`(평균회귀 특성시간, 기본 200) — 클수록 이전 값을 오래
"기억"해 느리게 표류. 시드는 $\texttt{crc32}(\text{ref\_seed}:\text{cell\_id}:\text{direction})$로
셀·방향별 결정론적(재현성 보장, 멀티프로세싱 안전).

### 2.3 `q_abs` — BOL 고정 절대용량 분할 (`common/scenario/q_abs.py`)

$q_{frac\_wide}/q_{frac\_ref}$와 달리 분모가 **그 셀의 BOL(첫 사이클) 실측 용량 $Q_{BOL}$로
고정**:

$$
q_{abs}(t) = \frac{Q(t)}{Q_{BOL}}, \qquad Q_{BOL} = Q_{cell}(\text{cycle}=1)
$$

---

## 3. HI(Health Indicator) 피처 — 66종/세그먼트 (`5_model/hi_compute.py`)

카테고리: STAT(20) + DIFF(20) + LFP(20) + MORPH(6) = 66. 대표 정의만 발췌(전수는
`5_model/utils/hi_schema.py`의 `STAT_KEYS`/`DIFF_KEYS`/`LFP_KEYS`/`MORPH_KEYS` 참고).

### 3.1 STAT (예시)

전하가중 평균전압:

$$
\bar V_{cw} = \frac{\sum_t V(t)\,|I(t)|\,\Delta t}{\sum_t |I(t)|\,\Delta t}
$$

전압 표준편차: $\sigma_V = \operatorname{std}(V)$.
Q–I 피어슨 상관: $\rho_{QI} = \operatorname{corr}(Q_{rel}, |I|)$.
세그먼트 절대 전하량(2026-08-07 재포함, `SOH_EXCLUDE_STAT_LEAK`로 토글):

$$
q_{abs,seg} = \sum_t |I(t)|\,\Delta t \big/ 3600, \qquad
E_{seg} = \sum_t V(t)\,|I(t)|\,\Delta t \big/ 3600
$$

### 3.2 DIFF (예시 — 미분용량 IC/DV)

$$
\frac{dV}{dQ}(t) \approx \frac{V(t+h)-V(t)}{Q(t+h)-Q(t)}, \qquad
\text{IC}(V) = \frac{dQ}{dV}
$$

전압 추세 기울기: $\displaystyle \text{slope}_V = \frac{V(t_{end})-V(t_0)}{\sum \Delta t}$.
IC 피크 비대칭도(FWHM 기반): $\text{peak\_asym} = \dfrac{\text{half-width}_{left}}{\text{half-width}_{right}}$.

### 3.3 LFP (LFP 화학종 특이적, 예시)

플래토 구간 비율: $\displaystyle \text{plateau\_frac} = \frac{n(\text{plateau mask})}{n_{bins}}$.

전압 평탄도: $\displaystyle \text{v\_flatness} = 1 - \frac{\sigma_V}{V_{max}-V_{min}}$.

전압 오목도(전하가중 평균 대비 양끝 평균 편차):

$$
\text{v\_concavity} = \bar V_{cw} - \frac{V(t_0)+V(t_{end})}{2}
$$

### 3.4 MORPH — BOL 대비 형태 거리 (6종, `hi_correlation.py:2074-2086`)

세그먼트 곡선을 $[0,1]$ 그리드로 정규화한 3종(V–t, V–Q, V–E)에 대해, **그 셀에서 그
세그먼트가 처음 유효했던 사이클의 곡선을 BOL 기준선**으로 저장해두고 매 사이클 비교:

$$
\text{DTW}(x, x_{BOL}) = \frac{1}{n}\min_{\pi} \sum_{(i,j)\in\pi} |x_i - x_{BOL,j}|
\quad (\text{Sakoe–Chiba 밴드 제약})
$$

$$
\text{Fréchet}(x, x_{BOL}) = \max_i |x_i - x_{BOL,i}| \quad (\text{고정 격자에서는 이산 Fréchet} \equiv \max|\Delta|)
$$

BOL 곡선이 없으면(그 셀에서 그 세그먼트가 한 번도 안 뽑히면) NaN.

---

## 4. HardConcrete L0 게이트 (Louizos et al. 2018) — `5_model/models/hard_concrete.py`

파라미터: $\beta=2/3$, $\gamma=-0.1$, $\zeta=1.1$ (고정).

**학습 시(재매개변수화 샘플링)**:

$$
u \sim \mathcal{U}(0,1), \qquad
s = \sigma\!\left(\frac{\log u - \log(1-u) + \log\alpha}{\beta}\right)
$$

$$
z = \operatorname{clip}\bigl(s\cdot(\zeta-\gamma)+\gamma,\ 0,\ 1\bigr)
$$

**추론 시(결정론적)**: $s=\sigma(\log\alpha)$, $z=\operatorname{clip}(s(\zeta-\gamma)+\gamma,0,1)$,
활성 여부 $= \mathbb{1}[z>0]$.

**게이트 활성 확률(L0 페널티에 사용)**:

$$
P(z\neq 0) = \sigma\!\Bigl(\log\alpha - \beta\log(-\gamma/\zeta)\Bigr)
$$

각 HI 인덱스 $i$마다 학습 가능한 $\log\alpha_i$가 있고, 게이트는 두 단계로 존재한다:
① 방향별 probe gate($\text{charge\_probe\_gate}$, $\text{discharge\_probe\_gate}$, 각 $N_{HI}$개),
② 시나리오별 scen gate($\text{scen\_gates}[s]$, $s=1..6$, 각 $N_{HI}$개).

---

## 5. 모델 순전파 — `5_model/models/scr_model.py`

$$
x = x_{hi} \odot \text{nan\_mask} \in \mathbb{R}^{N_{HI}}
$$

**Stage A(방향별 probe gate)**:

$$
x_{probe} = x \odot z_{probe}^{(dir)}, \qquad
z_{probe}^{(dir)} =
\begin{cases}
z_{\text{charge\_probe}} & \text{direction}>0 \\
z_{\text{discharge\_probe}} & \text{direction}\le 0
\end{cases}
$$

(선택) $\lambda_{scen}>0$이면 보조 분류 로짓: $\hat{\ell} = \text{MLP}_{\text{probe}}([x_{probe}\,\|\,\text{dir}])$.

**Stage B(시나리오별 scen gate)**:

$$
x_{scen} = x \odot z_{scen}^{(s)}, \qquad s = \text{seg\_idx (하드 라우팅)}
$$

**특징 결합 + 회귀 헤드**:

$$
\text{feat} = \bigl[\,x_{probe}\ \|\ x_{scen}\ \|\ (\text{cnn\_emb 또는 raw\_flat})\ \|\ \text{direction}\ \|\ \text{cap\_init}\,\bigr]
$$

$$
\hat{y} = \text{cap\_head}(\text{feat}) \in (0,1] \quad (\text{SOH ratio})
$$

기본(raw 융합 없음) $\dim(\text{feat}) = 2N_{HI}+2$. `with_raw_cnn=$true$`면 $+3$
($[h_{scen}, h_{intensity}, h_{soh}]$, §7), `with_raw_flat=true`면 $+144$
($\text{RAW\_CH}\times\text{RAW\_N} = 3\times 48$).

---

## 6. 손실 함수 (Phase 1) — `5_model/training/scr_loss.py`

$$
\mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda_{scen}\,\mathcal{L}_{CE} + \lambda_{L0}(\text{epoch})\,\mathcal{L}_{L0}
$$

$$
\mathcal{L}_{MSE} = \frac{1}{B}\sum_{i=1}^B (\hat y_i - y_i)^2, \qquad
\mathcal{L}_{CE} = -\frac{1}{B}\sum_i \log \operatorname{softmax}(\hat{\ell}_i)_{level_i}
$$

**비용가중 L0 페널티** — 시나리오 $s$에서 HI $i$가 활성일 확률(직렬 결합):

$$
P_s(i \text{ active}) = 1-(1-p_{probe,i}^{(dir(s))})(1-p_{scen,i}^{(s)})
$$

$$
\mathcal{L}_{L0} = \frac{1}{6}\sum_{s=1}^{6}\sum_{i=1}^{N_{HI}} c_i \cdot P_s(i \text{ active})
$$

카테고리별 비용 $c_i$: STAT=1.0, DIFF=1.5, LFP=2.0, MORPH=3.0
(`CATEGORY_COSTS`, `5_model/utils/hi_schema.py`).

**$\lambda_{L0}$ 스케줄**(`delayed_warmup`, `5_model/training/scr_trainer.py`):

$$
\lambda_{L0}(e) =
\begin{cases}
0 & e < e_{warmup} \\[4pt]
\lambda_{target}\cdot\min\!\left(\dfrac{e-e_{warmup}}{e_{ramp}},\ 1\right) & e \ge e_{warmup}
\end{cases}
$$

기본 $e_{warmup}=50$, $e_{ramp}=100$.

**$\lambda_{L0}$ 자동 계산**(`lambda_l0_auto: true`, `5_model/train_scr.py:_compute_lambda_l0`):

$$
\lambda_{target} = \operatorname{clip}\!\left(0.01\cdot\sqrt{\frac{10}{\bar m}\cdot\frac{10}{k}},\ 10^{-4},\ 0.5\right),
\qquad \bar m = \frac{m_{charge}+m_{discharge}}{2}
$$

$k=$ `scen_k_count`(시나리오별 목표 활성 HI 수), $m$=probe 목표 활성 수.

**Phase 2**(게이트 고정): $\mathcal{L} = \mathcal{L}_{MSE}$만 사용 ($\mathcal{L}_{L0}$ 항은
게이트가 buffer라 항상 0).

---

## 7. RawCNN + 보조손실 (2026-08-03 재설계) — `5_model/models/raw_cnn.py`, `train_classifier.py`

$$
x_{raw} \in \mathbb{R}^{3\times 48} = [V,\ I_{signed},\ t_{rel}]
$$

$$
\text{cnn\_emb} = \text{RawCNN}(x_{raw}) = [h_{scen},\ h_{intensity},\ h_{soh}] \in \mathbb{R}^3
$$

(stem→ResBlock→AttentionPool×3 독립분기→각 Linear(64,1)→BatchNorm1d(3), 34.7K 파라미터)

**분류기 학습 시 보조손실**(`train_classifier.py`, `classifier.type: cnn` 전용):

$$
\mathcal{L}_{aux} = \lambda_1 \text{MSE}(h_{scen}, y_{scen}) + \lambda_2 \text{MSE}(h_{intensity}, y_{int})
+ \lambda_3 \text{MSE}(\text{probe}(h_{soh}), y_{SOH}) + \lambda_4 \mathcal{L}_{decorr}
$$

$\lambda_1=\lambda_2=\lambda_3=0.1$, $\lambda_4=0.05$. 보조 타깃: $y_{scen}=$
`lfp_plateau_frac`, $y_{int}=$ `stat_i_std`(둘 다 이미 계산된 HI 재사용), $y_{SOH}=$ 실제
SOH 라벨.

**Decorrelation penalty**(배치 내 $[h_{scen},h_{intensity},h_{soh}]$ 상호 비상관 유도):

$$
\mathcal{L}_{decorr} = \sum_{j\neq k} \rho_{jk}^2, \qquad
\rho_{jk} = \operatorname{corr}(h_j, h_k) \ \text{(배치 내 피어슨 상관계수)}
$$

**분류기 총 손실**: $\mathcal{L}_{clf} = \text{CE}(\text{logits}, \text{level}) + \mathcal{L}_{aux}$
(mlp 타입이면 $\mathcal{L}_{aux}=0$).

---

## 8. SOH 타깃 정의 — `5_model/datasets/segment_dataset.py`

$$
\text{SOH} = \frac{Q_{cap}(\text{cycle})}{Q_{init}} \in (0, 1]
$$

$Q_{init}$은 (a) `use_initial_capacity=true`면 그 셀의 **첫 사이클 실측값**(BOL), 또는
(b) 데이터셋별 정격용량(`nominal_capacities`: MIT=1.1Ah, HUST=1.2Ah). **주의**: 이
$Q_{cap}$/$Q_{init}$은 §2.2의 $Q_{ref}$(노이즈 낀 세그먼트 분모)와 **완전히 분리된
경로**다 — 학습 타깃은 노이즈가 섞이지 않은 실측값 그대로다.

---

## 9. 평가 지표 — `5_model/utils/metrics.py`

$$
\text{RMSE} = \sqrt{\frac{1}{N}\sum_i (y_i-\hat y_i)^2}, \qquad
\text{MAE} = \frac{1}{N}\sum_i |y_i-\hat y_i|
$$

$$
R^2 = 1 - \frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}, \qquad
\text{MAPE} = \frac{100}{N}\sum_i \left|\frac{y_i-\hat y_i}{y_i}\right| \ \ (|y_i|>10^{-6}\text{만})
$$

(이 프로젝트 표기 관례: 위 값들을 논문 %p 단위로 보고할 때는 SOH ratio 스케일(0–1)에
$\times 100$한다 — `docs/260810_RESULTS.md` §10.4 참고.)

---

## 10. 라우팅(배포 시나리오 추정) 3모드 — `5_model/test_scr.py`, `evaluation/scr_evaluator.py`

- **oracle**: $s = \text{seg\_idx}$(정답 존 라벨) — 회귀 상한선.
- **hard**: $s = \arg\max_c \text{softmax}(\text{clf\_logits})_c$ (분류기 argmax) → 그 시나리오
  게이트로 단일 forward.
- **soft**: 분류기 확률 가중 평균 —
$$
\hat y_{soft} = \sum_{c=1}^{n_{classes}} P(\text{class}=c)\cdot \hat y^{(c)}
$$
  ($\hat y^{(c)}$는 클래스 $c$에 대응하는 시나리오 게이트로 계산한 예측값)
