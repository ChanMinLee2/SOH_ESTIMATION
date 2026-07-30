"""
common/scenario/full_cycle.py — 축: 전체 사이클 단일 세그먼트 (부분 관측 대비 베이스라인).

구간 분할을 전혀 하지 않고, 충전 전체 curve 1개 + 방전 전체 curve 1개를 그대로
세그먼트로 사용한다 — "부분 사이클(q_frac_wide/vqslope 등)로 관측했을 때 전체 사이클
대비 얼마나 손해를 보는가"를 재기 위한 상한선(ceiling) 베이스라인.

시나리오: chg_full(0) / dis_full(1)  — 방향만으로 결정되므로 분류가 필요 없다
routing = [[0], [1]]   # dir_idx: 0=충전, 1=방전 / latent_class는 항상 0

test_rs.py와 동일하게 n_classes=1(무의미한 단일 클래스), classifier_default="none"
컨벤션을 따른다.

사용 예:
  python 4_hi_analysis/hi_correlation.py --seg-axis full_cycle
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter

_SCENARIO_NAMES = ["chg_full", "dis_full"]
_ROUTING = [[0], [1]]


class FullCycleSegmenter(Segmenter):
    """방향별 전체 사이클을 그대로 1개 세그먼트로 사용하는 세그멘터."""

    name = "full_cycle"

    def __init__(self, min_pts: int = 10):
        self.min_pts = min_pts

    def get_spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            axis="full_cycle",
            n_scenarios=2,
            scenario_names=_SCENARIO_NAMES,
            n_classes=1,
            class_names=["full"],
            routing=_ROUTING,
            classifier_default="none",
            params={"min_pts": self.min_pts},
        )

    def _extract(
        self,
        v: np.ndarray, i: np.ndarray, dt: np.ndarray, q: np.ndarray,
        direction: int, cell_id: str, cycle: int, seg_local_start: int,
    ) -> tuple[list[SegmentRecord], int]:
        if len(v) < self.min_pts:
            return [], seg_local_start
        dir_idx = 0 if direction == 1 else 1
        scenario_id = _ROUTING[dir_idx][0]
        q_tot = float(q[-1]) if len(q) > 0 else 0.0
        rec = SegmentRecord(
            cell_id=cell_id, cycle=cycle, seg_local_id=seg_local_start,
            scenario_id=scenario_id, latent_class=0, direction=direction,
            v=v, i=i, dt=dt, q=q,
            meta={"zone": "full", "seg_name": _SCENARIO_NAMES[scenario_id],
                  "q_frac_lo": 0.0, "q_frac_hi": 1.0, "q_tot": q_tot},
        )
        return [rec], seg_local_start + 1

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

        if len(dis_v) >= self.min_pts:
            recs, seg_local = self._extract(
                dis_v, dis_i, dis_dt, dis_q, -1, cell_id, cycle, seg_local)
            yield from recs

        if chg_v is not None and len(chg_v) >= self.min_pts:
            recs, seg_local = self._extract(
                chg_v, chg_i, chg_dt, chg_q, +1, cell_id, cycle, seg_local)
            yield from recs
