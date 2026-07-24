"""
common/scenario/q_frac_wide.py — 축: 파라미터 구간 균등격자 세그먼트.

세 구간(hi/mid/lo) × 방향(충/방) = 6 시나리오.

구간 정의 (n1=구간 크기):
  hi  : [0.0,       n1]           — 0% 끝점 → 50% 방향
  mid : [0.5-n1/2,  0.5+n1/2]    — 50% 중심 → 양방향
  lo  : [1.0-n1,    1.0]          — 100% 끝점 → 50% 방향

  n1 > 0.5 이면 구간 간 겹침 발생 가능 (의도적 허용).

세그먼트 (n2=길이):
  구간 내 유효 시작 범위 [zone_start, zone_end-n2] 를
  np.linspace(..., n_samples) 로 균등 배치.
  s + n2 <= zone_end  (구간 경계 침범 없음).
  n_samples=1 이면 유효 범위의 중앙점 1개.

시나리오 이름·라우팅 — rcs/qfrac 와 동일 컨벤션:
  chg_lo(0) chg_mid(1) chg_hi(2) | dis_hi(3) dis_mid(4) dis_lo(5)
  routing = [[0, 1, 2], [5, 4, 3]]

사용 예:
  python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide \\
      --axis-config '{"q_frac_wide": {"n1": 0.4, "n2": 0.2, "n_samples": 4}}'
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter
# CV 구간 제거 비활성화로 더 이상 사용하지 않음 (iter_segments 참고)
# from .vwindow import _detect_cv_start

_SCENARIO_NAMES = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
_ROUTING = [[0, 1, 2], [5, 4, 3]]

# (zone_name, latent_class) — lo=0, mid=1, hi=2
_ZONES: list[tuple[str, int]] = [
    ("hi",  2),
    ("mid", 1),
    ("lo",  0),
]


class QFracWideSegmenter(Segmenter):
    """파라미터 구간 균등격자 세그멘터."""

    name = "q_frac_wide"

    def __init__(
        self,
        n1: float = 0.4,       # 구간 크기 (q_frac 비율, [0.35, 0.45] 범위)
        n2: float = 0.2,       # 세그먼트 길이 (q_frac 비율, 0 < n2 < n1)
        n_samples: int = 4,    # 구간당 세그먼트 수
        min_pts: int = 10,
        cv_v_thresh: float = 3.59,
        cv_cc_frac: float = 0.80,
    ):
        if not (0.35 <= n1 <= 0.45):
            raise ValueError(
                f"q_frac_wide: n1은 [0.35, 0.45] 범위여야 합니다. 현재 n1={n1}"
            )
        if not (0 < n2 < n1):
            raise ValueError(
                f"q_frac_wide: 0 < n2 < n1 필요. 현재 n1={n1}, n2={n2}"
            )
        self.n1 = n1
        self.n2 = n2
        self.n_samples = n_samples
        self.min_pts = min_pts
        self.cv_v_thresh = cv_v_thresh
        self.cv_cc_frac = cv_cc_frac

        # 진단용 카운터 (min_pts 생존율 계산) — iter_segments의 공개 동작에는 영향 없음.
        # scenario_name -> count. reset_counters()로 초기화 후 여러 셀에 걸쳐 누적 가능.
        self.n_attempted: dict[str, int] = {}
        self.n_yielded: dict[str, int] = {}
        # scenario_name -> [원시 포인트 수, ...] — 시도한 모든 후보의 실제 m.sum() 기록.
        # 이 분포만 있으면 재스캔 없이 임의의 min_pts 값에서 생존율을
        # (arr >= threshold).mean() 으로 즉시 계산할 수 있다.
        self.candidate_n_points: dict[str, list] = {}

    def reset_counters(self) -> None:
        """카운터 초기화 (seg_diagnose.py 등 진단 스크립트용)."""
        self.n_attempted = {}
        self.n_yielded = {}
        self.candidate_n_points = {}

    # ── 구간 정의 ────────────────────────────────────────────────────────────

    def _zone_bounds(self) -> dict[str, tuple[float, float]]:
        """zone_name → (절대 시작 q_frac, 절대 끝 q_frac)."""
        n1 = self.n1
        return {
            "hi":  (0.0,         n1),
            "mid": (0.5 - n1/2,  0.5 + n1/2),
            "lo":  (1.0 - n1,    1.0),
        }

    def _start_positions(self, zone_start: float, zone_end: float) -> np.ndarray:
        """유효 시작 범위 [zone_start, zone_end-n2] 의 균등 격자점.

        n_samples=1 → 범위 중앙점 반환.
        유효 범위 없음(n2 >= zone_end-zone_start) → 빈 배열.
        """
        lo = zone_start
        hi = zone_end - self.n2
        if hi < lo - 1e-9:
            return np.array([])
        if self.n_samples == 1:
            return np.array([(lo + hi) / 2.0])
        return np.linspace(lo, hi, self.n_samples)

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
    ) -> tuple[list[SegmentRecord], int]:
        q_tot = float(q[-1]) if len(q) > 0 else 0.0
        if q_tot < 0.05:
            return [], seg_local_start

        spec      = self.get_spec()
        dir_idx   = 0 if direction == 1 else 1
        bounds    = self._zone_bounds()
        records: list[SegmentRecord] = []
        seg_local = seg_local_start

        for zone_name, latent_class in _ZONES:
            zone_start, zone_end = bounds[zone_name]
            starts = self._start_positions(zone_start, zone_end)
            if len(starts) == 0:
                continue

            scenario_id = spec.routing[dir_idx][latent_class]
            sname       = _SCENARIO_NAMES[scenario_id]

            for start_qf in starts:
                end_qf = start_qf + self.n2
                lo_q   = start_qf * q_tot
                hi_q   = end_qf   * q_tot
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
                        "q_frac_lo": start_qf,
                        "q_frac_hi": end_qf,
                    },
                ))
                seg_local += 1

        return records, seg_local

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def get_spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            axis="q_frac_wide",
            n_scenarios=6,
            scenario_names=_SCENARIO_NAMES,
            n_classes=3,
            class_names=["lo", "mid", "hi"],
            routing=_ROUTING,
            classifier_default="mlp_probe",
            params={"n1": self.n1, "n2": self.n2, "n_samples": self.n_samples},
        )

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
                dis_v, dis_i, dis_dt, dis_q, -1, cell_id, cycle, seg_local)
            yield from recs

        # 충전 (CC+CV 전체 사용 — 100% = 해당 사이클 실제 완충 용량과 일치시키기 위해
        # CV 구간 제거 로직을 비활성화함. cv_v_thresh/cv_cc_frac는 더 이상 쓰이지 않음.)
        # cv_start = _detect_cv_start(chg_v, chg_i, self.cv_v_thresh, self.cv_cc_frac)
        # cc_v  = chg_v[:cv_start]
        # cc_i  = chg_i[:cv_start]
        # cc_dt = chg_dt[:cv_start]
        # cc_q  = chg_q[:cv_start]
        if chg_v is not None and len(chg_v) >= self.min_pts:
            recs, seg_local = self._extract(
                chg_v, chg_i, chg_dt, chg_q, +1, cell_id, cycle, seg_local)
            yield from recs
