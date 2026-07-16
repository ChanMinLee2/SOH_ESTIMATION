"""
common/scenario/qfrac.py — 축 0: q_frac 기반 3분할 × 방향 (현행 기본값).

현재 hi_correlation.py 의 DIS_SEGS / CHG_SEGS 동작을 완전히 재현한다.
  방전: [0.0, 0.4, 0.7, 1.0] → dis_hi / dis_mid / dis_lo
  충전: [0.0, 0.4, 0.7, 1.0] → chg_lo / chg_mid / chg_hi

M1 회귀 게이트: QFracSegmenter로 산출한 HI pkl이 기존 hi_correlation.py
산출물과 수치적으로 동일해야 한다 (segment 슬라이스 경계값이 동일함을 보장).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter

# ---------------------------------------------------------------------------
# 기본 경계 — 현행 hi_correlation.py DIS_SEGS / CHG_SEGS 와 동일
# ---------------------------------------------------------------------------
_DEFAULT_DIS_BOUNDS: list[float] = [0.0, 0.4, 0.7, 1.0]
_DEFAULT_CHG_BOUNDS: list[float] = [0.0, 0.4, 0.7, 1.0]

# 세그먼트 이름 — hi_schema.SEGMENTS 순서와 동일 (backward compat)
_SCENARIO_NAMES: list[str] = [
    "chg_lo", "chg_mid", "chg_hi",   # scenario_id 0, 1, 2
    "dis_hi", "dis_mid", "dis_lo",   # scenario_id 3, 4, 5
]

# 방전 세그먼트: (bounds index → segment name, latent_class)
# dis_hi = 첫 40%(high SOC) = High class(2)
# dis_mid = 30%(mid SOC)    = Mid  class(1)
# dis_lo  = 마지막 30%(low) = Low  class(0)
_DIS_SEGS: list[tuple[str, int]] = [
    ("dis_hi",  2),
    ("dis_mid", 1),
    ("dis_lo",  0),
]

# 충전 세그먼트: (bounds index → segment name, latent_class)
_CHG_SEGS: list[tuple[str, int]] = [
    ("chg_lo",  0),
    ("chg_mid", 1),
    ("chg_hi",  2),
]

# routing[dir_idx][latent_class] = scenario_id
# dir_idx: 0=충전, 1=방전
# 충전: Low(0)→chg_lo(0), Mid(1)→chg_mid(1), High(2)→chg_hi(2)
# 방전: Low(0)→dis_lo(5), Mid(1)→dis_mid(4), High(2)→dis_hi(3)
_ROUTING: list[list[int]] = [[0, 1, 2], [5, 4, 3]]


class QFracSegmenter(Segmenter):
    """q_frac 기반 세그멘터 — 현행 동작의 직접 래핑."""

    name = "qfrac"

    def __init__(
        self,
        dis_bounds: list[float] | None = None,
        chg_bounds: list[float] | None = None,
        min_pts: int = 10,
    ):
        self.dis_bounds = dis_bounds or _DEFAULT_DIS_BOUNDS
        self.chg_bounds = chg_bounds or _DEFAULT_CHG_BOUNDS
        self.min_pts = min_pts

    def get_spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            axis="qfrac",
            n_scenarios=len(_SCENARIO_NAMES),
            scenario_names=_SCENARIO_NAMES,
            n_classes=3,
            class_names=["lo", "mid", "hi"],
            routing=_ROUTING,
            classifier_default="mlp_probe",
            params={
                "dis_bounds": self.dis_bounds,
                "chg_bounds": self.chg_bounds,
            },
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
        q_tot_d = float(dis_q[-1]) if len(dis_q) > 0 else 0.0

        # ── 방전 세그먼트 ────────────────────────────────────────────────
        if q_tot_d >= 0.05:
            for i_seg, (seg_name, latent_class) in enumerate(_DIS_SEGS):
                lo_f = self.dis_bounds[i_seg]
                hi_f = self.dis_bounds[i_seg + 1]
                lo = lo_f * q_tot_d
                hi = hi_f * q_tot_d
                m = (dis_q >= lo) & (dis_q < hi)
                if m.sum() < self.min_pts:
                    continue
                yield SegmentRecord(
                    cell_id=cell_id,
                    cycle=cycle,
                    seg_local_id=seg_local,
                    scenario_id=_SCENARIO_NAMES.index(seg_name),
                    latent_class=latent_class,
                    direction=-1,
                    v=dis_v[m],
                    i=dis_i[m],
                    dt=dis_dt[m],
                    q=dis_q[m],
                    meta={"q_frac_lo": lo_f, "q_frac_hi": hi_f, "seg_name": seg_name},
                )
                seg_local += 1

        # ── 충전 세그먼트 ────────────────────────────────────────────────
        if chg_v is None or len(chg_v) < self.min_pts:
            return
        q_tot_c = float(chg_q[-1]) if len(chg_q) > 0 else 0.0
        if q_tot_c < 0.05:
            return

        for i_seg, (seg_name, latent_class) in enumerate(_CHG_SEGS):
            lo_f = self.chg_bounds[i_seg]
            hi_f = self.chg_bounds[i_seg + 1]
            lo = lo_f * q_tot_c
            hi = hi_f * q_tot_c
            m = (chg_q >= lo) & (chg_q < hi)
            if m.sum() < self.min_pts:
                continue
            yield SegmentRecord(
                cell_id=cell_id,
                cycle=cycle,
                seg_local_id=seg_local,
                scenario_id=_SCENARIO_NAMES.index(seg_name),
                latent_class=latent_class,
                direction=+1,
                v=chg_v[m],
                i=chg_i[m],
                dt=chg_dt[m],
                q=chg_q[m],
                meta={"q_frac_lo": lo_f, "q_frac_hi": hi_f, "seg_name": seg_name},
            )
            seg_local += 1
