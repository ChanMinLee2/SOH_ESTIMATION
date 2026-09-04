"""
common/scenario/q_frac_ref.py — 축: q_frac_wide와 존/라우팅/spec 전부 동일, 분모(q_tot)만
"과거 레퍼런스 + 노이즈" 기반으로 바꾼 서브클래스.

배경: docs/SOC.md §6 Phase 3(구현 방식 결정, 방식 A 채택) / Phase 4(q_ref·노이즈 설계).
q_frac_wide는 세그먼트 경계를 그 세션(사이클) 자신의 전류적산 최종값(q[-1])으로 나눈다 —
이는 "완주해야 q_tot을 안다"는 세션-완료 요구를 갖는다(§6 Phase 0 재프레이밍). q_frac_ref는
q_frac_wide._extract의 `q_tot = q[-1]` 부분(이제 `_normalizer` 훅으로 분리됨,
q_frac_wide.py 참고)만 오버라이드해 과거 사이클의 값(+노이즈)을 대신 쓴다 — 나머지 로직
(존 정의, 라우팅, random_segment 모드 등)은 부모 그대로 재사용한다.

초기 구현은 lag=0 (2026-08-05 결정, docs/SOC.md §6 Phase 4 참고): `q_ref(t) =
q_cum_ref(t) + noise`, `q_cum_ref(t)`는 그 세그먼트가 포함된 사이클 자신의 전류적산값 —
q_frac_wide의 q_tot과 본질적으로 같고 라벨 누수에 가깝다는 §4.1.1의 함정을 **인지한 채로**
채택한 스캐폴딩 단계다. `ref_lag`를 1 이상으로 올리면 §4.1.1이 권장하는 "진짜 과거 시점
실측값 + 얇은 측정오차" 형태로 자연스럽게 전환된다 — lag 스윕(§4.1.2/§6 Phase 5)의
한쪽 끝점이 바로 이 lag=0(=q_frac_wide와 동등)이다.

lag 정의(§6 Phase 2-(b) 미해결 이슈에 대한 이번 구현의 선택): 원본 사이클 번호가 아니라
**그 (셀, 방향)에서 실제로 처리된 사이클 순서** 기준이다 — 결측 사이클이 있어도 매 호출마다
"몇 번째로 유효했던 사이클인가"만 센다. 사이클은 항상 오름차순으로 들어온다(§6 Phase 0).

노이즈(§6 Phase 4 요구사항 5, 2026-08-05 개정): 화이트노이즈가 아니라 셀·방향별 고정
바이어스 + 느린 드리프트. 드리프트는 두 가지 모드 지원(`noise_mode`):

  - "ou"(**기본값**) — bounded random walk(Ornstein-Uhlenbeck 이산화, AR(1)).
    실제 BMS 센서 드리프트처럼 불규칙하게 천천히 표류하고, 평균회귀(→0)로 무한정
    발산하지 않는다. "왜 하필 고정 주기로 오르내리는가"라는 사인파의 물리적 근거
    부족 문제를 해결한다 — 사용자 피드백(2026-08-05)으로 사인파에서 전환.
  - "sine" — 이전 기본값. 결정론적 사인파, 재현·해석이 더 단순해 비교용으로 남겨둠.

  두 모드 다 진폭 합이 [-noise_amp, +noise_amp]로 클리핑된다.
  `crc32(cell_id, direction, ref_seed)` 결정론적 시드로 재현성 보장(멀티프로세싱 안전,
  q_frac_wide.py `_rng_for`와 동일 관례) — OU는 추가로 그 시드의 RNG를 셀 내에서
  사이클 순서대로(=§6 Phase 0 오름차순 보장) 순차 소비해 매 사이클 새 증분을 뽑는다.

콜드스타트(§4.5): 그 (셀,방향)에서 아직 lag만큼 과거 이력이 쌓이지 않았으면(첫 lag개
사이클) 그 사이클 자신의 q_this로 폴백한다 — lag=0일 때는 이 폴백이 매번 발동하는
정상 동작이다.

──────────────────────────────────────────────────────────────────────────────
n2 범위 모드 (2026-09-03 추가)
──────────────────────────────────────────────────────────────────────────────
`n2_start`/`n2_end`를 같이 주면 세그먼트 길이가 고정 n2가 아니라 격자
{n2_start, n2_start+n2_step, ..., n2_end}(기본 n2_step=0.1)에서 **매번 랜덤으로**
뽑힌다. 배포 현실에서 관측 창 길이가 고정이라는 가정을 깨기 위한 축이다.

핵심 조건은 **커버리지 100%** — 한 존의 모든 포인트가 적어도 하나의 세그먼트에
반드시 들어가야 한다. 이를 위해 "랜덤 배치"가 아니라 **랜덤 길이 타일링**을 쓴다:

  1) 존 시작에서 출발해 격자에서 길이 L을 뽑아 [s, s+L]을 놓고 s += L 을 반복한다
     (조각 사이에 틈이 없다 = 이어붙이기).
  2) s+L이 존 끝을 넘어서는 순간, 마지막 조각은 [zone_end-L, zone_end]로 놓고 끝낸다.
     길이는 격자값 그대로 유지되고 끝점이 정확히 존 끝에 붙으므로, 앞 조각과 조금
     겹칠 수는 있어도 **틈은 생기지 않는다**(겹침 < 틈, 커버리지가 조건이므로).
  3) 마지막 조각만 상한 포함(q <= hi)으로 마스킹해 존 끝점 포인트도 빠지지 않게 한다.
  4) 포인트 수가 min_pts 미만인 조각은 그냥 버리면 그 구간이 커버리지 구멍이 되므로,
     **버리지 않고 이웃 조각에 병합**한다(마지막 조각이면 앞 조각에 병합). 이웃을
     흡수한 조각은 meta["merged"]=True로 표시된다 — 길이는 대개 격자를 벗어나지만
     0.1+0.1=0.2처럼 우연히 격자값과 같아질 수도 있으므로, "격자 위 조각만" 골라내려면
     seg_len이 아니라 이 플래그를 봐야 한다.

따라서 커버리지가 100% 미만이 되는 경우는 "존 전체 포인트 수 < min_pts"라 아무
세그먼트도 만들 수 없는 사이클뿐이다(알고리즘 한계가 아니라 데이터 한계) —
_4_data_hi/<axis_dir>/coverage_stats.txt 에 실측값이 남는다.

랜덤은 (ref_seed 계열과 분리된) n2_seed + crc32(cell, cycle, direction, zone)로
결정론적이라 멀티프로세싱/재실행에서 동일하게 재현된다. 이 모드에서 `n_samples`는
쓰이지 않는다(존당 조각 수가 타일링 결과로 정해짐).

사용 예:
  --seg-axis q_frac_ref --axis-config '{"n1": 0.4, "n2_start": 0.1, "n2_end": 0.3}'

──────────────────────────────────────────────────────────────────────────────
레퍼런스 calibration (2026-09-03 추가, docs/260903_RESULTS.md §1)
──────────────────────────────────────────────────────────────────────────────
기존 노이즈 모델은 (셀,방향) 하나가 셀 수명 전체(수백~2000+ 사이클) 동안 **한 번도
재보정되지 않는** BMS를 가정했다 — bias는 영원히 고정, OU drift는 평균회귀는 해도
0으로 돌아가는 데 수백 사이클이 걸린다. `calibration_period`를 주면 그 (셀,방향)에서
N사이클마다 드리프트가 리셋되는 "주기적으로 재보정하는 BMS"를 시뮬레이션한다.

무엇을 리셋하는가(`calibration_mode`):
  - "drift_only"(**기본값**) — OU 누적 상태만 0으로. bias(그 개체의 계통오차,
    예: 쿨롱효율 추정 편향)는 유지 — 재보정은 "그동안 쌓인 표류를 터는" 행위이지
    센서 자체의 고정 오차까지 없애주는 보장은 없다는 보수적 가정.
  - "full" — bias까지 0으로. 실제 OCV 기반 재보정은 절대 기준점을 다시 잡으므로
    계통오차까지 잡는 경우가 흔하다는 반대 근거도 있다 — **두 모드 다 정당화되므로
    실험에서 반드시 둘 다 스윕해서 비교할 것**(어느 쪽을 정답으로 미리 못 박지 않음).

주기는 `calibration_jitter`로 셀마다 살짝 흔들 수 있다(전체 셀이 똑같은 사이클에
동시 재보정되는 비현실성 완화) — (셀,방향)마다 결정론적으로 한 번 뽑은 고정 오프셋을
쓴다(재보정마다 다시 뽑지 않음 — "이 개체는 대략 T±jitter 주기로 재보정된다"는
단순화된 모델링 선택).

**sine 모드에는 적용 안 됨**: sine 드리프트는 애초에 상태가 없는(cycle의 순수
결정론적 함수) 모델이라 "리셋할 누적 상태"가 없다 — `calibration_period`를 줘도
noise_mode="sine"이면 조용히 no-op이다(에러 아님, 문서화만 해둠).

**권장 초기값**: `calibration_period=100`. 근거 — (1) OU 평균회귀 특성시간
`noise_period_cycles`의 기본값이 200이므로, 100은 그 절반이라 "자연 회귀보다 재보정이
먼저 개입"하는 조건을 만든다(그래야 두 메커니즘이 뚜렷이 구분되는 효과를 낸다).
(2) 실측 셀 수명이 MIT 530~2236, HUST 1143~2678사이클이므로, 100주기면 셀 하나당
5~27회 재보정이 일어나 궤적 그림(V1-b)에서 톱니 패턴이 여러 번 반복돼 통계적으로
안정된 판단이 가능하다. (3) 스윕은 `{None, 200, 100, 50, 25}`로 시작할 것 — 200은
noise_period_cycles와 같아 "재보정이 자연 회귀와 거의 같은 속도" 경계 조건, 25는
재보정이 지배적인 극단.

**주의**: `docs/260903_RESULTS.md`의 자체 실측(§4)에서 노이즈 진폭을 1%→10%로
올려도 `q_abs`류 HI의 오차가 4.48%→4.98%로 거의 안 변했다(주 오차원은 세그먼트
이산화, 레퍼런스 노이즈가 아니었음) — 이는 calibration으로 노이즈 "진폭"을 줄여도
성능이 크게 안 움직일 수 있다는 뜻이다. 다만 calibration이 바꾸는 건 진폭이 아니라
**시간축 상관구조**(오래 표류한 값 vs 방금 재보정된 값)이므로, 이 논리가 그대로
적용되지는 않는다 — 실험으로 확인해야 하는 이유가 이것이다(V1-d).
"""

from __future__ import annotations

import zlib

import numpy as np

from .base import ScenarioSpec
from .q_frac_wide import QFracWideSegmenter

_VALID_NOISE_MODES = ("ou", "sine")
_VALID_CALIB_MODES = ("drift_only", "full")
_N2_GRID_TOL = 1e-6


def calib_path_tag(params: dict) -> str:
    """calibration 관련 파라미터 → 경로 태그 조각 (예: "" | "_calib-100" | "_calib-100_full_j10").

    n2_path_tag와 동일한 이유로 존재한다 — hi_correlation._qfref_tag / train_scr /
    train_classifier / visualize_results 네 곳이 q_frac_ref 경로 태그를 각자 조립하므로,
    calibration 파라미터가 다르면 반드시 다른 경로에 저장되도록 이 함수 하나로 규칙을
    모았다(§4.6 confound 방지 — 복붙 규칙을 한 곳이라도 빠뜨리면 재보정 있는/없는 데이터가
    같은 캐시를 조용히 공유해버림). calibration_period 미설정(기본)이면 빈 문자열을
    돌려줘 기존 경로와 100% 동일하게 유지한다(하위 호환).
    """
    period = params.get("calibration_period")
    if not period:
        return ""
    mode = str(params.get("calibration_mode", "drift_only"))
    jitter = int(params.get("calibration_jitter", 0) or 0)
    mode_sfx = "" if mode == "drift_only" else f"_{mode}"
    jitter_sfx = f"_j{jitter}" if jitter else ""
    return f"_calib-{int(period)}{mode_sfx}{jitter_sfx}"


def n2_path_tag(params: dict) -> str:
    """n2 관련 파라미터 → 데이터 경로 태그 조각 (예: "n2-20%" | "n2-10~30%s10").

    hi_correlation._qfw_tag / train_scr / train_classifier / visualize_results 네 곳이
    각자 n2 태그를 만들던 걸 이 함수 하나로 모았다 — n2 범위 모드가 켜졌을 때
    **반드시 다른 경로에 저장**되게 하려면 네 곳이 동시에 같은 규칙을 써야 하고,
    복붙 규칙을 하나 빠뜨리면 고정 n2 캐시를 조용히 재사용해버리기 때문(§4.6 confound).
    범위 파라미터가 없으면 기존 문자열과 100% 동일한 값을 돌려준다(하위 호환).
    """
    n2_start = params.get("n2_start")
    n2_end = params.get("n2_end")
    if n2_start is None or n2_end is None:
        return f"n2-{int(round(params.get('n2', 0.2) * 100))}%"
    step = int(round(float(params.get("n2_step", 0.1)) * 100))
    return (f"n2-{int(round(float(n2_start) * 100))}~{int(round(float(n2_end) * 100))}%"
            f"s{step}")


class QFracRefSegmenter(QFracWideSegmenter):
    """q_frac_wide + 과거 레퍼런스 기반 분모(q_tot) — 존/라우팅/spec은 부모와 동일."""

    name = "q_frac_ref"

    def __init__(
        self,
        ref_lag: int = 0,                 # 0=q_frac_wide와 동등(스캐폴딩 초기값), >=1=진짜 과거 레퍼런스
        noise_amp: float = 0.03,          # 레퍼런스에 얹는 노이즈 최대 진폭 (분수, 0.03=±3%)
        noise_mode: str = "ou",           # "ou"(기본, bounded random walk) | "sine"(구버전, 결정론적)
        noise_period_cycles: float = 200.0,  # ou: 평균회귀 특성시간(사이클) | sine: 파장(사이클)
        ref_seed: int = 20260805,         # 노이즈 재현성 시드 (연-월-일, 다른 축 시드와 충돌 방지)
        # ── n2 범위 모드 (2026-09-03, 모듈 docstring 참고) ──────────────────
        n2_start: float | None = None,    # 세그먼트 길이 격자 하한 (None이면 고정 n2 모드)
        n2_end: float | None = None,      # 세그먼트 길이 격자 상한 (None이면 고정 n2 모드)
        n2_step: float = 0.1,             # 격자 간격 (기본 0.1 = 요구사항)
        n2_seed: int = 20260903,          # 길이 추첨 재현성 시드 (ref_seed와 분리)
        # ── 레퍼런스 calibration (2026-09-03, 모듈 docstring 참고) ───────────
        calibration_period: int | None = None,  # N사이클마다 재보정. None=재보정 없음(기존 동작)
        calibration_mode: str = "drift_only",   # "drift_only"(OU만 리셋) | "full"(bias까지 리셋)
        calibration_jitter: int = 0,            # 재보정 주기를 ±jitter 사이클 흔듦 (0=정확히 주기대로)
        **kwargs,
    ):
        # 범위 모드면 부모의 `0 < n2 < n1` 검증을 통과시키기 위해 n2를 격자 하한으로
        # 맞춰둔다 — 실제 조각 길이는 _segment_plan이 매번 다시 뽑으므로 self.n2는
        # "격자 하한"이라는 의미만 갖는다(spec.params에도 그대로 기록됨).
        self._n2_range = (n2_start is not None) and (n2_end is not None)
        if self._n2_range:
            kwargs["n2"] = float(n2_start)
        elif (n2_start is None) != (n2_end is None):
            raise ValueError(
                "q_frac_ref: n2_start와 n2_end는 둘 다 주거나 둘 다 생략해야 합니다. "
                f"현재 n2_start={n2_start}, n2_end={n2_end}"
            )
        super().__init__(**kwargs)
        if ref_lag < 0:
            raise ValueError(f"q_frac_ref: ref_lag은 0 이상이어야 합니다. 현재 ref_lag={ref_lag}")
        if not (0.0 <= noise_amp <= 0.10):
            raise ValueError(
                f"q_frac_ref: noise_amp은 [0, 0.10](0~10%) 범위 권장. 현재 noise_amp={noise_amp}"
            )
        if noise_mode not in _VALID_NOISE_MODES:
            raise ValueError(
                f"q_frac_ref: noise_mode은 {_VALID_NOISE_MODES} 중 하나여야 합니다. "
                f"현재 noise_mode={noise_mode!r}"
            )
        self.ref_lag = int(ref_lag)
        self.noise_amp = float(noise_amp)
        self.noise_mode = str(noise_mode)
        self.noise_period_cycles = float(noise_period_cycles)
        self.ref_seed = int(ref_seed)

        # ── calibration 검증 ─────────────────────────────────────────────────
        if calibration_mode not in _VALID_CALIB_MODES:
            raise ValueError(
                f"q_frac_ref: calibration_mode은 {_VALID_CALIB_MODES} 중 하나여야 합니다. "
                f"현재 calibration_mode={calibration_mode!r}"
            )
        if calibration_period is not None and int(calibration_period) <= 0:
            raise ValueError(
                f"q_frac_ref: calibration_period은 None 또는 양의 정수여야 합니다. "
                f"현재 calibration_period={calibration_period}"
            )
        if calibration_jitter < 0:
            raise ValueError(
                f"q_frac_ref: calibration_jitter는 0 이상이어야 합니다. "
                f"현재 calibration_jitter={calibration_jitter}"
            )
        if (calibration_period is not None and calibration_jitter > 0
                and calibration_jitter >= calibration_period):
            raise ValueError(
                f"q_frac_ref: calibration_jitter는 calibration_period보다 작아야 합니다 "
                f"(같거나 크면 유효 주기가 0 이하로 붕괴할 수 있음). 현재 "
                f"calibration_period={calibration_period}, calibration_jitter={calibration_jitter}"
            )
        self.calibration_period = int(calibration_period) if calibration_period is not None else None
        self.calibration_mode = str(calibration_mode)
        self.calibration_jitter = int(calibration_jitter)
        # (셀,방향) -> 마지막 재보정(또는 첫 관측) 사이클. 재보정 트리거 판정용.
        self._last_calib: dict[tuple[str, int], int] = {}
        # (셀,방향) -> 실제 적용 주기(jitter 반영, 한 번 뽑으면 그 키에서 고정).
        self._calib_period_cache: dict[tuple[str, int], int] = {}
        # 진단용 — (셀,방향)별 실제 재보정 발동 횟수.
        self.n_calibrations: dict[tuple[str, int], int] = {}

        # ── n2 범위 모드 검증 + 격자 생성 ────────────────────────────────────
        self.n2_start = float(n2_start) if n2_start is not None else None
        self.n2_end = float(n2_end) if n2_end is not None else None
        self.n2_step = float(n2_step)
        self.n2_seed = int(n2_seed)
        self._n2_grid: np.ndarray | None = None
        # 범위 모드 진단용 — 존 전체 포인트가 min_pts 미만이라 아무 세그먼트도 못 만든
        # (셀,사이클,존) 횟수(커버리지 100% 미달의 유일한 원인)와 min_pts 병합 발생 수.
        self.zone_too_small: dict[str, int] = {}
        self.n_merged: dict[str, int] = {}
        if self._n2_range:
            if self.n2_step <= 0:
                raise ValueError(f"q_frac_ref: n2_step은 0보다 커야 합니다. 현재 n2_step={self.n2_step}")
            if not (0 < self.n2_start <= self.n2_end):
                raise ValueError(
                    f"q_frac_ref: 0 < n2_start <= n2_end 필요. "
                    f"현재 n2_start={self.n2_start}, n2_end={self.n2_end}"
                )
            if self.n2_end >= self.n1:
                # 존 폭이 곧 n1이므로 n2_end >= n1이면 존 안에 안 들어가는 길이가 생긴다.
                raise ValueError(
                    f"q_frac_ref: n2_end < n1 필요(존 폭=n1). "
                    f"현재 n2_end={self.n2_end}, n1={self.n1}"
                )
            n_steps = (self.n2_end - self.n2_start) / self.n2_step
            if abs(n_steps - round(n_steps)) > _N2_GRID_TOL * max(1.0, abs(n_steps)):
                raise ValueError(
                    f"q_frac_ref: (n2_end - n2_start)가 n2_step의 정수배여야 합니다. "
                    f"현재 n2_start={self.n2_start}, n2_end={self.n2_end}, n2_step={self.n2_step} "
                    f"→ {n_steps}스텝(정수 아님)"
                )
            k = int(round(n_steps))
            # 부동소수점 오차 누적을 막으려고 곱셈 + 반올림으로 격자를 만든다
            # (0.1+0.1+0.1 != 0.3 문제 회피).
            self._n2_grid = np.array(
                [round(self.n2_start + j * self.n2_step, 10) for j in range(k + 1)],
                dtype=float,
            )

        # (cell_id, direction) -> [(cycle, q_this), ...] 유효 사이클 순서대로 누적.
        # §6 Phase 0: 세그멘터는 셀마다 워커 안에서 새로 생성되므로 이 상태는
        # 셀 간 오염 없이 자연스럽게 격리된다(q_abs._cap_ref와 동일 관례).
        self._q_hist: dict[tuple[str, int], list[tuple[int, float]]] = {}
        # (cell_id, direction) -> (bias, phase) — 고정 바이어스(+sine 위상), 첫 호출 시 확정.
        self._noise_params: dict[tuple[str, int], tuple[float, float]] = {}
        # ou 모드 전용 상태
        self._ou_rng: dict[tuple[str, int], np.random.Generator] = {}
        self._ou_state: dict[tuple[str, int], float] = {}
        # 진단용: 마지막으로 계산된 노이즈 값(플랏 스크립트 등에서 재계산 없이 조회, OU
        # 상태를 이중으로 전진시키지 않기 위함) — _noise_frac 호출 시마다 갱신.
        self._last_noise: dict[tuple[str, int], float] = {}

    # ── 노이즈 ───────────────────────────────────────────────────────────────

    def _noise_params_for(self, cell_id: str, direction: int) -> tuple[float, float]:
        """(bias, phase) — bias는 두 모드 공통(고정 오프셋), phase는 sine 모드 전용."""
        key = (str(cell_id), int(direction))
        params = self._noise_params.get(key)
        if params is None:
            seed = zlib.crc32(f"{self.ref_seed}:{cell_id}:{direction}".encode())
            rng = np.random.default_rng(seed)
            bias = float(rng.uniform(-0.5, 0.5)) * self.noise_amp
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            params = (bias, phase)
            self._noise_params[key] = params
        return params

    def _noise_frac_sine(self, cell_id: str, direction: int, cycle: int, bias: float) -> float:
        """결정론적 사인파(구버전 기본값) — 진폭 noise_amp/2, 물리적 근거는 약하나 재현·해석 단순."""
        _, phase = self._noise_params_for(cell_id, direction)
        return 0.5 * self.noise_amp * float(
            np.sin(2.0 * np.pi * cycle / self.noise_period_cycles + phase)
        )

    def _noise_frac_ou(self, cell_id: str, direction: int) -> float:
        """Bounded random walk(Ornstein-Uhlenbeck, AR(1) 이산화).

        noise_period_cycles를 평균회귀 특성시간으로 해석: phi=exp(-1/period) — period가
        클수록 이전 값을 더 오래 "기억"해 더 느리게 표류한다. 정상분포 표준편차를
        noise_amp/4로 고정해 exact AR(1) 재귀식(phi*prev + std*sqrt(1-phi^2)*eps)으로
        period에 무관하게 항상 같은 정상분산을 유지한다(표준 OU 이산화 공식).
        사이클 번호 간격이 아니라 **호출 순서**를 한 스텝으로 센다(§6 Phase 2-(b)
        "가용 사이클 기준" 결정과 동일 관례 — ref_lag 히스토리 색인과 일관성 유지).
        """
        key = (str(cell_id), int(direction))
        std = self.noise_amp / 4.0
        phi = float(np.exp(-1.0 / self.noise_period_cycles))

        rng = self._ou_rng.get(key)
        if rng is None:
            seed = zlib.crc32(f"{self.ref_seed}:ou:{cell_id}:{direction}".encode())
            rng = np.random.default_rng(seed)
            self._ou_rng[key] = rng
            # 정상분포에서 초기화 — "이미 오래 진행 중이던 과정을 지금 관측 시작"으로
            # 취급(첫 사이클부터 부자연스럽게 0에서 시작하지 않음).
            state = float(rng.normal(0.0, std))
        else:
            prev = self._ou_state[key]
            eps = float(rng.normal())
            state = phi * prev + std * float(np.sqrt(max(1.0 - phi ** 2, 0.0))) * eps

        self._ou_state[key] = state
        return state

    def _effective_calib_period(self, cell_id: str, direction: int) -> int:
        """jitter를 반영한 실제 재보정 주기. (셀,방향)마다 한 번만 뽑아 고정한다
        ("이 개체는 대략 T±jitter 주기로 재보정된다"는 단순화 — 재보정 이벤트마다
        다시 뽑지 않음, 모듈 docstring 참고)."""
        key = (str(cell_id), int(direction))
        period = self._calib_period_cache.get(key)
        if period is None:
            if self.calibration_jitter <= 0:
                period = self.calibration_period
            else:
                seed = zlib.crc32(f"{self.ref_seed}:calib:{cell_id}:{direction}".encode())
                rng = np.random.default_rng(seed)
                offset = int(rng.integers(-self.calibration_jitter, self.calibration_jitter + 1))
                period = max(1, self.calibration_period + offset)
            self._calib_period_cache[key] = period
        return period

    def _maybe_calibrate(self, cell_id: str, direction: int, cycle: int) -> None:
        """calibration_period가 설정돼 있고 OU 모드일 때, 재보정 주기가 찼으면
        드리프트 상태(+full 모드면 바이어스도) 리셋한다.

        sine 모드는 상태가 없어(cycle의 순수 함수) 적용 대상이 아니다 — 조용히 no-op.
        첫 관측 사이클에서는 리셋하지 않고 기준점만 잡는다 — 이 시점엔 아직
        `_noise_frac_ou`가 OU 상태를 초기화(`_ou_rng`에 키 등록)하기 전이라, 여기서
        `_ou_state`를 건드려도 `_noise_frac_ou`의 '최초 호출' 분기가 그 값을 무시하고
        정상분포에서 새로 초기화해버리기 때문(그래서 첫 호출은 기준점만 기록).
        """
        if not self.calibration_period or self.noise_mode != "ou":
            return
        key = (str(cell_id), int(direction))
        cycle = int(cycle)
        last = self._last_calib.get(key)
        if last is None:
            self._last_calib[key] = cycle
            return
        period = self._effective_calib_period(cell_id, direction)
        if cycle - last >= period:
            self._ou_state[key] = 0.0
            self._last_calib[key] = cycle
            self.n_calibrations[key] = self.n_calibrations.get(key, 0) + 1
            if self.calibration_mode == "full":
                _, phase = self._noise_params_for(cell_id, direction)
                self._noise_params[key] = (0.0, phase)

    def _noise_frac(self, cell_id: str, direction: int, cycle: int) -> float:
        """완만한(사이클축에 매끄러운) 노이즈 — [-noise_amp, +noise_amp]로 클리핑.

        구성: 셀·방향별 고정 바이어스(최대 ±noise_amp/2) + 느린 드리프트(ou 또는 sine).
        화이트노이즈(사이클마다 급변)가 아니라 §6 Phase 4 요구사항 5의 "완만한 fade" 형태.
        calibration_period가 설정돼 있으면 드리프트 상태를 주기적으로 리셋한다(모듈
        docstring 참고) — 이 리셋 판정은 드리프트를 실제로 계산하기 전에 먼저 한다.
        **부작용 있음(상태 전진)**: ou 모드는 호출마다 내부 상태가 한 스텝씩 진행되므로,
        같은 (cell_id, direction, cycle)이라도 두 번 호출하면 다른 값이 나온다 — 진단
        목적으로 노이즈 값만 다시 보고 싶으면 self._last_noise[(cell_id, direction)]를
        조회할 것(재호출 금지).
        """
        key = (str(cell_id), int(direction))
        self._maybe_calibrate(cell_id, direction, cycle)
        bias, _ = self._noise_params_for(cell_id, direction)
        if self.noise_mode == "ou":
            drift = self._noise_frac_ou(cell_id, direction)
        else:
            drift = self._noise_frac_sine(cell_id, direction, cycle, bias)
        total = float(np.clip(bias + drift, -self.noise_amp, self.noise_amp))
        self._last_noise[key] = total
        return total

    # ── q_tot 훅 오버라이드 ──────────────────────────────────────────────────

    def _normalizer(
        self,
        direction: int,
        cell_id: str,
        cycle: int,
        q: np.ndarray,
    ) -> float:
        q_this = float(q[-1]) if len(q) > 0 else 0.0

        key = (str(cell_id), int(direction))
        hist = self._q_hist.setdefault(key, [])
        hist.append((int(cycle), q_this))

        if self.ref_lag <= 0 or len(hist) <= self.ref_lag:
            # lag=0(초기 구현 기본값) 또는 콜드스타트(§4.5: 이 (셀,방향)에서 아직
            # ref_lag개만큼의 과거 이력이 없음) → 그 사이클 자신으로 폴백.
            q_ref_raw = q_this
        else:
            q_ref_raw = hist[-(self.ref_lag + 1)][1]

        noise_frac = self._noise_frac(cell_id, direction, cycle)
        return q_ref_raw * (1.0 + noise_frac)

    # ── n2 범위 모드: 랜덤 길이 타일링 ───────────────────────────────────────

    def _n2_rng(
        self, cell_id: str, cycle: int, direction: int, zone_name: str
    ) -> np.random.Generator:
        """(n2_seed, cell, cycle, direction, zone) 결정론적 RNG.

        q_frac_wide._rng_for와 같은 관례(crc32로 문자열 → 비음수 정수)를 따른다.
        존까지 시드에 넣는 이유: 같은 사이클 안에서도 hi/mid/lo가 서로 다른 길이
        조합을 갖게 해 한 사이클이 통째로 "짧은 세그먼트만 있는 사이클"이 되지
        않도록 하기 위함.
        """
        dir_key = 0 if direction > 0 else 1
        cell_key = zlib.crc32(str(cell_id).encode())
        zone_key = zlib.crc32(str(zone_name).encode())
        return np.random.default_rng(
            [self.n2_seed, cell_key, int(cycle), dir_key, zone_key]
        )

    def _tile_zone(
        self, zone_start: float, zone_end: float, rng: np.random.Generator
    ) -> list[tuple[float, float]]:
        """존을 격자 길이 랜덤 조각으로 **틈 없이** 덮는다 (모듈 docstring 1~2단계).

        마지막 조각은 길이를 격자값으로 유지한 채 끝점을 zone_end에 맞춰 놓기 때문에
        앞 조각과 겹칠 수 있다 — 커버리지가 조건이므로 "겹침 허용, 틈 금지"를 택했다.
        """
        grid = self._n2_grid
        bounds: list[tuple[float, float]] = []
        s = zone_start
        # 길이는 항상 n2_start(>0) 이상이라 s는 매 반복 최소 그만큼 전진한다 →
        # 최대 ceil((zone_end-zone_start)/n2_start)회면 끝난다. 방어적 상한만 둔다.
        max_tiles = int(np.ceil((zone_end - zone_start) / self.n2_start)) + 2
        for _ in range(max_tiles):
            length = float(grid[rng.integers(len(grid))])
            if s + length >= zone_end - 1e-12:
                bounds.append((zone_end - length, zone_end))
                return bounds
            bounds.append((s, s + length))
            s += length
        # 도달 불가(위 상한 계산상). 혹시 모를 경우에도 틈은 남기지 않는다.
        bounds.append((min(s, zone_end - float(grid[0])), zone_end))
        return bounds

    def _merge_below_min_pts(
        self,
        bounds: list[tuple[float, float]],
        q: np.ndarray,
        q_tot: float,
        zone_name: str,
    ) -> list[tuple[float, float, bool]]:
        """포인트 수 < min_pts 인 조각을 이웃에 병합 (모듈 docstring 4단계).

        그냥 두면 _extract의 min_pts 필터에서 탈락해 그 구간이 통째로 커버리지
        구멍이 된다. 뒤 조각으로 병합(마지막이면 앞 조각으로)해서 어떤 포인트도
        버려지지 않게 한다.
        반환: (start_qf, end_qf, merged) — merged=True면 이웃을 흡수한 조각.
        self.n_merged는 "병합 **연산** 횟수"라 한 조각이 두 번 흡수하면 2가 더해진다
        (merged=True 조각 수보다 클 수 있음).
        """
        def _count(s: float, e: float, is_last: bool) -> int:
            lo_q, hi_q = s * q_tot, e * q_tot
            # 마지막 조각만 상한 포함 — _extract의 마스킹 규칙과 반드시 일치해야 한다.
            if is_last:
                return int(((q >= lo_q) & (q <= hi_q)).sum())
            return int(((q >= lo_q) & (q < hi_q)).sum())

        out: list[list] = [[s, e, False] for s, e in bounds]
        i = 0
        while len(out) > 1 and i < len(out):
            s, e, _ = out[i]
            if _count(s, e, i == len(out) - 1) >= self.min_pts:
                i += 1
                continue
            if i + 1 < len(out):
                out[i + 1] = [min(s, out[i + 1][0]), out[i + 1][1], True]
                out.pop(i)          # i 유지 — 합쳐진 조각을 다시 검사
            else:
                out[i - 1] = [out[i - 1][0], max(e, out[i - 1][1]), True]
                out.pop(i)          # 앞 조각은 커지기만 하므로 재검사 불필요
            self.n_merged[zone_name] = self.n_merged.get(zone_name, 0) + 1
        return [(s, e, bool(mg)) for s, e, mg in out]

    def _segment_plan(
        self,
        zone_start: float,
        zone_end: float,
        cell_id: str,
        cycle: int,
        direction: int,
        zone_name: str,
        q: np.ndarray,
        q_tot: float,
    ) -> list[tuple[float, float, bool, dict | None]]:
        if not self._n2_range:
            return super()._segment_plan(
                zone_start, zone_end, cell_id, cycle, direction, zone_name, q, q_tot)

        rng = self._n2_rng(cell_id, cycle, direction, zone_name)
        tiles = self._tile_zone(zone_start, zone_end, rng)
        tiles = self._merge_below_min_pts(tiles, q, q_tot, zone_name)

        # 존 전체가 min_pts 미만이면 병합해도 살릴 수 없다 — 데이터 한계로 따로 기록.
        if len(tiles) == 1:
            lo_q, hi_q = tiles[0][0] * q_tot, tiles[0][1] * q_tot
            if int(((q >= lo_q) & (q <= hi_q)).sum()) < self.min_pts:
                self.zone_too_small[zone_name] = self.zone_too_small.get(zone_name, 0) + 1

        last = len(tiles) - 1
        # merged=True = 이웃을 흡수한 조각 — 사후 분석에서 "순수 격자 조각만" 걸러낼 때 사용.
        return [(s, e, idx == last, {"merged": True} if mg else None)
                for idx, (s, e, mg) in enumerate(tiles)]

    def _track_zone_coverage(self) -> bool:
        # 범위 모드는 100% 커버리지가 설계 조건이므로 항상 실측해서 남긴다
        # (_4_data_hi/<axis_dir>/coverage_stats.txt).
        return self._n2_range

    def reset_counters(self) -> None:
        super().reset_counters()
        self.zone_too_small = {}
        self.n_merged = {}
        self.n_calibrations = {}

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def get_spec(self) -> ScenarioSpec:
        spec = super().get_spec()
        spec.axis = "q_frac_ref"
        spec.params = dict(spec.params)
        spec.params.update({
            "ref_lag": self.ref_lag,
            "noise_amp": self.noise_amp,
            "noise_mode": self.noise_mode,
            "noise_period_cycles": self.noise_period_cycles,
            "ref_seed": self.ref_seed,
        })
        if self._n2_range:
            # 범위 모드일 때만 넣는다 — 고정 n2 런의 scenario_spec.json과 경로 태그가
            # 예전과 100% 동일하게 유지되도록(하위 호환, n2_path_tag 참고).
            spec.params.update({
                "n2_start": self.n2_start,
                "n2_end": self.n2_end,
                "n2_step": self.n2_step,
                "n2_seed": self.n2_seed,
            })
        if self.calibration_period is not None:
            # calibration 미사용 런의 scenario_spec.json/경로 태그는 예전과 100% 동일하게
            # 유지되도록 사용할 때만 넣는다(하위 호환, calib_path_tag와 동일 원칙).
            spec.params.update({
                "calibration_period": self.calibration_period,
                "calibration_mode": self.calibration_mode,
                "calibration_jitter": self.calibration_jitter,
            })
        return spec
