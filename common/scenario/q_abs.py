"""
common/scenario/q_abs.py — 축: 절대용량(Q_abs) 기반 세그멘터.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
설계 근거 (리뷰어용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[q_frac_wide 와의 차이 — 별개의 축]
  q_frac_wide 는 존 경계를 q_frac = q/q_tot (그 사이클 실측 총 용량) 대비 비율로 정의한다.
  → 실배포 추론 시점에는 q_tot 을 알 수 없고(세션이 아직 안 끝남), q_tot 을 안다는 것 자체가
    사실상 SOH 를 아는 것이라 부분관측 문제와 모순된다.
  q_abs 는 존 경계를 **각 셀의 첫 사이클(가장 건강, BOL) 용량** 을 고정 기준으로 하는
  절대용량(Ah)으로 정의한다. 방향(충/방)별로 각자의 첫 사이클 용량을 기준으로 쓴다.
  → 배포 시에도 그 셀의 BOL 용량만 한 번 알아두면 되므로 q_tot 이 필요 없다.

[존 정의 — 정격용량(=첫 사이클 용량) 대비 비율]
  low  : [0.0,        mid_start]   — 저용량 구간            → latent_class 0 (lo)
  mid  : [mid_start,  mid_end]     — 중간 구간(플래토 겨냥) → latent_class 1 (mid)
  high : [mid_end,    1.0]         — 고용량 구간            → latent_class 2 (hi)

  mid_start / mid_end 두 파라미터로 3구간 경계를 지정한다.
  존은 [0,1]을 겹침 없이 타일링하므로 q_frac_wide(n1>1/3 시 겹침)의 mid/hi 경계 모호성이 없다.

[세그먼트 — seg_len(정격용량 비율)]
  각 존 안에서 유효 시작 범위 [zone_start, zone_end - seg_len] 를 n_samples 로 균등 배치.
  s + seg_len <= zone_end  (존 경계 침범 없음 — q_frac_wide 와 동일 제약).
  절대 위치(Ah) = start_frac * cap_ref, 여기서 cap_ref = 그 방향 첫 사이클 용량.
  seg_len <= 최소 존 폭 이어야 한다(생성자에서 검증).

  노화 사이클에서는 실제 도달 용량(q[-1])이 cap_ref 보다 작아지므로, high 존의 고용량
  끝쪽 세그먼트는 데이터가 없어 스킵될 수 있다(min_pts 미달). 이는 q_frac_wide 의 min_pts
  절벽과 같은 성격의 표본 손실이므로, seg_diagnose 로 존별 생존율을 먼저 측정해야 한다.

[adaptive_samples — 존 폭 반영 n_k 자동 산정 (배치 범위는 건드리지 않음)]
  존마다 유효 배치범위(zone폭 - seg_len)가 크게 다르면(예: low 5%p vs high 50%p)
  같은 n_samples를 균등 배치할 때 좁은 존에서 세그먼트끼리 심하게 겹친다(요구조건3).
  adaptive_samples=True 면 생성자에서 존별로 n_k(인접 세그먼트 목표 겹침률
  max_overlap(θ) 이하가 되도록 "존 폭(명목값, zone_end 그대로)" 기준으로 계산한 값과
  n_samples(상한) 중 작은 쪽)를 1회 계산해 고정한다 — 사이클마다 바뀌지 않으므로
  (라벨과 무관한 정적 계산) 원인(A) 라벨 누수는 없다.

  ※ 배치 범위를 "노화 셀 평균 EOL 잔존용량"으로 clip하는 방식은 시도했다가 폐기했다
  (원인: CV 구간은 그 사이클 "자신의" 충전 끝자락에서 발생하므로, BOL 비율로 clip하면
  clip 범위 안엔 노화 사이클의 CV만 남고 건강 사이클의 CV(더 높은 BOL 비율대)는
  통째로 제외된다 — high 존 안에서 "CV 형상 ↔ 노화됨"이라는 인위적 상관을 만들어내고
  건강 사이클의 실측 데이터도 버리므로 요구조건2 "노화 무관 최대 활용"에 오히려
  역행한다). 그래서 n_k(개수)만 조정하고 배치 위치(zone_start~zone_end 전체)는
  그대로 둔다 — 노화 사이클이 high 존 상단 세그먼트에서 min_pts 미달로 스킵되는
  현상(§11.1-③ 선택 편향) 자체는 남지만, 이는 `n_attempted`/`n_yielded` 카운터로
  이미 진단 가능하므로 mid_end/seg_len을 존별 생존율 실측(seg_diagnose)에 맞춰
  보수적으로 고르는 것으로 대응한다 — 세그멘터 내부에서 데이터를 숨기지 않는다.

[시나리오 구조 — q_frac_wide 와 동일(모델 구조 재사용), routing 은 다름]
  chg_lo(0) chg_mid(1) chg_hi(2) | dis_hi(3) dis_mid(4) dis_lo(5)
  routing = [[0, 1, 2], [3, 4, 5]]

  [SOC 정합성 — 2026-07-30 수정]
  이름의 lo/mid/hi 는 항상 실측 SOC(전압) 수준을 가리켜야 한다. q_abs 의 low 존
  (bound=[0, mid_start], latent_class=0)은 방향에 상관없이 항상 "그 방향 Q
  누적의 시작점"이다 — 충전이면 시작점=SOC 낮음(맞게 chg_lo 로 이름 붙음),
  방전이면 시작점=배터리가 아직 꽉 차있는 SOC 높은 시점(그런데 예전엔 dis_lo 로
  이름 붙어 있었다 — 반대).
  원인: q_abs 는 q_frac_wide 의 routing=[[0,1,2],[5,4,3]] 을 그대로 재사용했다.
  이 routing[1]=[5,4,3] 의 방전 반전은 q_frac_wide 자신의 _ZONES 컨벤션(존
  시작="hi", latent_class=2)에 맞춰 계산된 것이라 q_frac_wide 에서는 방전이
  맞게 나왔지만, q_abs 는 _ZONES 컨벤션이 반대(존 시작="low", latent_class=0)
  라 같은 routing[1]을 쓰면 충전은 맞고 방전이 틀어진다. routing[1]을 [3,4,5]로
  고쳐 low 존(latent0, 방전 시작=SOC 높음)→dis_hi(3), high 존(latent2, 방전
  끝=SOC 낮음)→dis_lo(5) 가 되도록 정정했다 — 충전(routing[0])은 원래도
  맞았으므로 그대로 둠. 이 수정으로 §11.1-④(축마다 이름 관례가 반대)로
  문서화됐던 q_frac_wide/q_abs 간 명명 불일치도 함께 해소된다 — 이제 둘 다
  "chg_lo/dis_hi=SOC 낮은/높은 쪽"을 일관되게 가리킨다.

사용 예:
  python 4_hi_analysis/hi_correlation.py --seg-axis q_abs \\
      --axis-config '{"q_abs": {"mid_start": 0.2, "mid_end": 0.5, "seg_len": 0.15, "n_samples": 4}}'
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter
from ._random_seg import sample_random_windows

_SCENARIO_NAMES = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
_ROUTING = [[0, 1, 2], [3, 4, 5]]  # SOC 정합성 수정 — 모듈 docstring 참조

# (zone_name, latent_class) — low=0, mid=1, high=2 (q_frac_wide 의 lo/mid/hi 컨벤션과 정렬)
_ZONES: list[tuple[str, int]] = [
    ("low",  0),
    ("mid",  1),
    ("high", 2),
]


class QAbsSegmenter(Segmenter):
    """절대용량(BOL 첫 사이클 용량 기준) 세그멘터."""

    name = "q_abs"

    def __init__(
        self,
        mid_start: float = 0.20,   # mid 존 시작 (정격용량 비율)
        mid_end: float = 0.50,     # mid 존 끝 (정격용량 비율)
        seg_len: float = 0.15,     # 세그먼트 길이 (정격용량 비율)
        n_samples: int = 4,        # 존당 세그먼트 수 (adaptive_samples=False일 때 그대로 사용)
        min_pts: int = 10,
        random_segment: bool = False,   # True: 존 내부 고정길이(seg_len_pts) 랜덤 창
        seg_len_pts: int = 20,          # 랜덤 창의 고정 관측 포인트 수 (cap 무관)
        random_seed: int = 42,
        adaptive_samples: bool = False,  # True: 존 폭(명목)에 맞춰 존별 n_k(개수)만 자동 산정 — 배치 범위는 그대로
        max_overlap: float = 0.50,       # 존 안 인접 세그먼트 목표 겹침률 상한(θ) — adaptive_samples 전용
    ):
        if not (0.0 < mid_start < mid_end < 1.0):
            raise ValueError(
                f"q_abs: 0 < mid_start < mid_end < 1 필요. "
                f"현재 mid_start={mid_start}, mid_end={mid_end}"
            )
        if seg_len <= 0:
            raise ValueError(f"q_abs: seg_len > 0 필요. 현재 seg_len={seg_len}")
        # seg_len 은 가장 좁은 존 폭 이하여야 한다(경계 비침범 세그먼트 생성 조건).
        zone_widths = {
            "low":  mid_start,
            "mid":  mid_end - mid_start,
            "high": 1.0 - mid_end,
        }
        narrowest = min(zone_widths.values())
        if not random_segment and seg_len > narrowest + 1e-9:
            raise ValueError(
                f"q_abs: seg_len({seg_len})이 가장 좁은 존 폭({narrowest:.3f}, "
                f"{min(zone_widths, key=zone_widths.get)})보다 큽니다 — 경계 비침범 "
                f"세그먼트를 만들 수 없습니다. seg_len 을 줄이거나 mid_start/mid_end 를 조정하세요."
            )
        if random_segment and seg_len_pts < min_pts:
            raise ValueError(
                f"q_abs: random_segment 시 seg_len_pts({seg_len_pts}) >= min_pts({min_pts}) 필요.")
        if not (0.0 <= max_overlap < 1.0):
            raise ValueError(f"q_abs: 0 <= max_overlap < 1 필요. 현재 max_overlap={max_overlap}")

        self.mid_start = float(mid_start)
        self.mid_end = float(mid_end)
        self.seg_len = float(seg_len)
        self.n_samples = int(n_samples)
        self.min_pts = int(min_pts)
        self.random_segment = bool(random_segment)
        self.seg_len_pts = int(seg_len_pts)
        self.random_seed = int(random_seed)
        self.adaptive_samples = bool(adaptive_samples)
        self.max_overlap = float(max_overlap)

        # BOL 기준 용량 캐시 — (cell_id, direction) → 그 방향 첫 사이클 q[-1].
        # 셀별로 첫 iter_segments 호출 시 설정되어 이후 모든 사이클에서 재사용된다.
        self._cap_ref: dict[tuple[str, int], float] = {}

        # 진단용 카운터 (q_frac_wide 와 동일 인터페이스) — seg_diagnose 재사용.
        self.n_attempted: dict[str, int] = {}
        self.n_yielded: dict[str, int] = {}
        self.candidate_n_points: dict[str, list] = {}
        self.coverage: dict[str, list] = {}

        # adaptive_samples: 존별 n_k(개수)를 한 번 계산해 캐시 — 사이클마다 재계산 불필요
        # (존 경계·seg_len·theta 모두 생성자에서 고정되므로 정적으로 결정됨). 배치 범위는
        # 그대로 [zone_start, zone_end]를 쓴다 — n_k만 바뀐다.
        self._zone_plan: dict[str, int] | None = (
            self._compute_adaptive_plan() if self.adaptive_samples and not self.random_segment else None
        )
        if self._zone_plan is not None:
            plan_str = ", ".join(f"{z}: n={n}" for z, n in self._zone_plan.items())
            print(f"[q_abs] adaptive_samples 적용 (max_overlap={self.max_overlap}): {plan_str}")

    def reset_counters(self) -> None:
        self.n_attempted = {}
        self.n_yielded = {}
        self.candidate_n_points = {}
        self.coverage = {}

    # ── 구간 정의 ────────────────────────────────────────────────────────────

    def _zone_bounds(self) -> dict[str, tuple[float, float]]:
        """zone_name → (시작 frac, 끝 frac) — 정격용량(BOL) 대비 비율."""
        return {
            "low":  (0.0,           self.mid_start),
            "mid":  (self.mid_start, self.mid_end),
            "high": (self.mid_end,  1.0),
        }

    def _start_positions(
        self, zone_start: float, zone_end: float, n_samples: "int | None" = None,
    ) -> np.ndarray:
        """유효 시작 범위 [zone_start, zone_end - seg_len] 균등 격자. n=1 → 중앙점.

        n_samples 를 넘기지 않으면 self.n_samples(고정 모드)를 쓴다 — adaptive_samples
        모드에서는 _extract 가 존별로 계산된 n_k 를 명시적으로 넘긴다.
        """
        n = self.n_samples if n_samples is None else n_samples
        lo = zone_start
        hi = zone_end - self.seg_len
        if hi < lo - 1e-9:
            return np.array([])
        if n == 1:
            return np.array([(lo + hi) / 2.0])
        return np.linspace(lo, hi, n)

    def _compute_adaptive_plan(self) -> dict[str, int]:
        """존별 n_k(세그먼트 개수)를 정적으로 계산한다 (생성자에서 1회 호출).

        배치 범위(zone_start~zone_end)는 절대 바꾸지 않는다 — "노화 셀 평균 도달범위로
        상한을 당기는" 방식은 시도했다가 폐기했다: BOL 비율로 clip하면 clip 범위 안엔
        노화 사이클의 CV(그 사이클 자신의 충전 끝자락 = 낮은 BOL 비율대)만 남고 건강
        사이클의 CV(더 높은 BOL 비율대)는 통째로 제외돼, "CV 형상 ↔ 노화됨"이라는
        인위적 상관을 만든다(요구조건1 정신 위반) + 건강 사이클의 실측 데이터를
        버린다(요구조건2 위반). 그래서 n_k(개수)만 존 폭에 맞춰 줄이고, 그 n_k개를
        원래 zone_start~zone_end 전 범위에 그대로 균등 배치한다 — 노화 사이클이 상단
        세그먼트에서 min_pts 미달로 자연 스킵되는 것은 그대로 두고(§11.1-③), 얼마나
        스킵되는지는 n_attempted/n_yielded로 진단한다.

        n_k 는 목표 겹침률 max_overlap(θ) 이하가 되도록 "존 폭(명목값)" 기준으로 계산한
        값과 사용자가 지정한 n_samples 중 작은 쪽 — n_samples 는 "이 이상은 늘리지
        않는다"는 상한 역할(요구조건3: 시나리오 내 과도한 중복 방지).
        """
        bounds = self._zone_bounds()
        plan: dict[str, int] = {}
        for zone_name, _latent in _ZONES:
            zone_start, zone_end = bounds[zone_name]
            r_k = max(0.0, zone_end - zone_start - self.seg_len)
            n_theta = 1.0 + r_k / ((1.0 - self.max_overlap) * self.seg_len)
            n_k = max(1, min(self.n_samples, int(np.floor(n_theta + 1e-9))))
            plan[zone_name] = n_k
        return plan

    # ── 한 방향 추출 ─────────────────────────────────────────────────────────

    def _extract(
        self,
        v: np.ndarray,
        i: np.ndarray,
        dt: np.ndarray,
        q: np.ndarray,
        direction: int,
        cell_id: str,
        cycle: int,
        seg_local_start: int,
        rand_rng: "np.random.Generator | None" = None,
    ) -> tuple[list[SegmentRecord], int]:
        q_this = float(q[-1]) if len(q) > 0 else 0.0
        if q_this < 0.05:
            return [], seg_local_start

        # BOL 기준 용량: 이 (셀, 방향)에 대한 첫 호출이면 이 사이클의 총 용량을 기준으로 고정.
        # (셀 처리 시 사이클이 오름차순으로 들어오므로 첫 유효 사이클 = 가장 건강한 시점.)
        # 단일 사이클 진단 호출 등 첫 사이클이 아닌 경우엔 그 사이클 자신을 기준으로 폴백.
        key = (str(cell_id), int(direction))
        cap_ref = self._cap_ref.get(key)
        if cap_ref is None or cap_ref < 0.05:
            cap_ref = q_this
            self._cap_ref[key] = cap_ref

        spec      = self.get_spec()
        dir_idx   = 0 if direction == 1 else 1
        bounds    = self._zone_bounds()
        records: list[SegmentRecord] = []
        seg_local = seg_local_start

        for zone_name, latent_class in _ZONES:
            zone_start, zone_end = bounds[zone_name]
            scenario_id = spec.routing[dir_idx][latent_class]
            sname       = _SCENARIO_NAMES[scenario_id]

            if self.random_segment:
                # ── 랜덤 모드: 존 안에서 고정 포인트수(seg_len_pts) 랜덤 창 ──
                is_last = (zone_name == "high")   # high=[mid_end, 1.0] → 상한 포함
                lo_q, hi_q = zone_start * cap_ref, zone_end * cap_ref
                zmask = ((q >= lo_q) & (q <= hi_q)) if is_last else ((q >= lo_q) & (q < hi_q))
                zone_idx = np.where(zmask)[0]
                windows, cov, tot = sample_random_windows(
                    zone_idx, self.seg_len_pts, self.n_samples, self.min_pts, rand_rng)
                c = self.coverage.setdefault(sname, [0, 0])
                c[0] += cov; c[1] += tot
                for w in windows:
                    n_pts = int(len(w))
                    self.n_attempted[sname] = self.n_attempted.get(sname, 0) + 1
                    self.candidate_n_points.setdefault(sname, []).append(n_pts)
                    if n_pts < self.min_pts:
                        continue
                    self.n_yielded[sname] = self.n_yielded.get(sname, 0) + 1
                    records.append(SegmentRecord(
                        cell_id=cell_id, cycle=cycle, seg_local_id=seg_local,
                        scenario_id=scenario_id, latent_class=latent_class,
                        direction=direction,
                        v=v[w], i=i[w], dt=dt[w], q=q[w],
                        meta={"zone": zone_name, "random": True,
                              "cap_ref": cap_ref,
                              "q_abs_lo": float(q[w[0]]),
                              "q_abs_hi": float(q[w[-1]])},
                    ))
                    seg_local += 1
                continue

            # ── 고정폭(seg_len, 정격용량 비율) 격자 모드 ──────────────────────────
            if self._zone_plan is not None:
                starts = self._start_positions(zone_start, zone_end, n_samples=self._zone_plan[zone_name])
            else:
                starts = self._start_positions(zone_start, zone_end)
            if len(starts) == 0:
                continue

            for start_fr in starts:
                end_fr = start_fr + self.seg_len
                lo_q   = start_fr * cap_ref
                hi_q   = end_fr   * cap_ref
                m      = (q >= lo_q) & (q < hi_q)
                n_pts  = int(m.sum())
                self.n_attempted[sname] = self.n_attempted.get(sname, 0) + 1
                self.candidate_n_points.setdefault(sname, []).append(n_pts)
                if n_pts < self.min_pts:
                    continue
                self.n_yielded[sname] = self.n_yielded.get(sname, 0) + 1

                records.append(SegmentRecord(
                    cell_id=cell_id,
                    cycle=cycle,
                    seg_local_id=seg_local,
                    scenario_id=scenario_id,
                    latent_class=latent_class,
                    direction=direction,
                    v=v[m], i=i[m], dt=dt[m], q=q[m],
                    meta={
                        "zone":      zone_name,
                        "cap_ref":   cap_ref,
                        "frac_lo":   float(start_fr),
                        "frac_hi":   float(end_fr),
                        "q_abs_lo":  float(lo_q),
                        "q_abs_hi":  float(hi_q),
                    },
                ))
                seg_local += 1

        return records, seg_local

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def get_spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            axis="q_abs",
            n_scenarios=6,
            scenario_names=_SCENARIO_NAMES,
            n_classes=3,
            class_names=["lo", "mid", "hi"],
            routing=_ROUTING,
            classifier_default="mlp_probe",
            params={"mid_start": self.mid_start, "mid_end": self.mid_end,
                    "seg_len": self.seg_len, "n_samples": self.n_samples,
                    "random_segment": self.random_segment, "seg_len_pts": self.seg_len_pts,
                    "adaptive_samples": self.adaptive_samples,
                    "max_overlap": self.max_overlap,
                    "zone_plan": dict(self._zone_plan) if self._zone_plan is not None else None},
        )

    def _rng_for(self, cell_id: str, cycle: int, direction: int) -> "np.random.Generator | None":
        if not self.random_segment:
            return None
        import zlib
        dir_key  = 0 if direction > 0 else 1
        cell_key = zlib.crc32(str(cell_id).encode())
        return np.random.default_rng([self.random_seed, cell_key, int(cycle), dir_key])

    def iter_segments(
        self,
        cell_id: str,
        cycle: int,
        dis_v: np.ndarray,
        dis_i: np.ndarray,
        dis_dt: np.ndarray,
        dis_q: np.ndarray,
        chg_v: np.ndarray | None = None,
        chg_i: np.ndarray | None = None,
        chg_dt: np.ndarray | None = None,
        chg_q: np.ndarray | None = None,
    ) -> Iterator[SegmentRecord]:
        seg_local = 0

        # 방전
        if len(dis_v) >= self.min_pts:
            recs, seg_local = self._extract(
                dis_v, dis_i, dis_dt, dis_q, -1, cell_id, cycle, seg_local,
                rand_rng=self._rng_for(cell_id, cycle, -1))
            yield from recs

        # 충전 (CC+CV 전체 사용 — cap_ref 도 첫 사이클 완충 용량 기준)
        if chg_v is not None and len(chg_v) >= self.min_pts:
            recs, seg_local = self._extract(
                chg_v, chg_i, chg_dt, chg_q, +1, cell_id, cycle, seg_local,
                rand_rng=self._rng_for(cell_id, cycle, +1))
            yield from recs
