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
"""

from __future__ import annotations

import zlib

import numpy as np

from .base import ScenarioSpec
from .q_frac_wide import QFracWideSegmenter

_VALID_NOISE_MODES = ("ou", "sine")


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
        **kwargs,
    ):
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

    def _noise_frac(self, cell_id: str, direction: int, cycle: int) -> float:
        """완만한(사이클축에 매끄러운) 노이즈 — [-noise_amp, +noise_amp]로 클리핑.

        구성: 셀·방향별 고정 바이어스(최대 ±noise_amp/2) + 느린 드리프트(ou 또는 sine).
        화이트노이즈(사이클마다 급변)가 아니라 §6 Phase 4 요구사항 5의 "완만한 fade" 형태.
        **부작용 있음(상태 전진)**: ou 모드는 호출마다 내부 상태가 한 스텝씩 진행되므로,
        같은 (cell_id, direction, cycle)이라도 두 번 호출하면 다른 값이 나온다 — 진단
        목적으로 노이즈 값만 다시 보고 싶으면 self._last_noise[(cell_id, direction)]를
        조회할 것(재호출 금지).
        """
        key = (str(cell_id), int(direction))
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
        return spec
