"""
common/scenario/rcs.py — 축 3: 임의 랜덤 세그멘트 (Deng 2022 계열).

사이클당 n_samples개 랜덤 시작점 + 고정 윈도우 폭 window(q_frac 기준)로
부분 세그먼트를 샘플링. seed 고정으로 재현성 보장.

assign:
  "position_bin" (기본): 세그먼트 중심 q_frac을 lo/mid/hi로 사후 binning
                         → n_scenarios=6 (qfrac과 동일, 교차 비교 가능)
  "none"               : n_classes=1, 라우팅 무력화 (Deng 원안)
                         → n_scenarios=2 (방향만 구분)

이중 용도:
  - 학습 축으로 사용하거나
  - test_scr.py --eval-axis rcs --eval-n N 에서 임의 윈도우 강건성 샘플러로 재사용
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter
from .vwindow import _detect_cv_start

_BIN_EDGES = [0.0, 1 / 3, 2 / 3, 1.0]   # lo / mid / hi 경계 (q_frac 기준)


def _bin_class(center_qf: float) -> int:
    """q_frac 중심값 → lo(0)/mid(1)/hi(2) 클래스."""
    for k in range(len(_BIN_EDGES) - 1):
        if center_qf < _BIN_EDGES[k + 1]:
            return k
    return len(_BIN_EDGES) - 2


class RCSSegmenter(Segmenter):
    """랜덤 세그멘트 샘플러."""

    name = "rcs"

    def __init__(
        self,
        n_samples: int = 6,
        window: float = 0.3,
        unit: str = "qfrac",   # "qfrac" | "voltage" (qfrac만 구현)
        assign: str = "position_bin",   # "position_bin" | "none"
        seed: int = 42,
        min_pts: int = 10,
        cv_v_thresh: float = 3.60,   # CC→CV 전환 전압 임계값 [V]
        cv_cc_frac: float = 0.80,    # CC→CV: I < cv_cc_frac × I_max
    ):
        self.n_samples = n_samples
        self.window = window
        self.unit = unit
        self.assign = assign
        self.seed = seed
        self.min_pts = min_pts
        self.cv_v_thresh = cv_v_thresh
        self.cv_cc_frac = cv_cc_frac

    def get_spec(self) -> ScenarioSpec:
        if self.assign == "position_bin":
            n_classes = 3
            class_names = ["lo", "mid", "hi"]
            routing = [[0, 1, 2], [5, 4, 3]]
            n_scenarios = 6
            scenario_names = ["chg_lo", "chg_mid", "chg_hi",
                              "dis_hi", "dis_mid", "dis_lo"]
        else:  # "none"
            n_classes = 1
            class_names = ["any"]
            routing = [[0], [1]]
            n_scenarios = 2
            scenario_names = ["chg", "dis"]

        return ScenarioSpec(
            axis="rcs",
            n_scenarios=n_scenarios,
            scenario_names=scenario_names,
            n_classes=n_classes,
            class_names=class_names,
            routing=routing,
            classifier_default="mlp_probe" if self.assign == "position_bin" else "none",
            params={
                "n_samples": self.n_samples,
                "window":    self.window,
                "unit":      self.unit,
                "assign":    self.assign,
                "seed":      self.seed,
            },
        )

    def _sample_segments(
        self,
        v: np.ndarray,
        i: np.ndarray,
        dt: np.ndarray,
        q: np.ndarray,
        direction: int,
        cell_id: str,
        cycle: int,
        seg_local_start: int,
        rng: np.random.Generator,
    ) -> tuple[list[SegmentRecord], int]:
        q_tot = float(q[-1]) if len(q) > 0 else 0.0
        if q_tot < 0.05:
            return [], seg_local_start

        spec = self.get_spec()
        dir_idx = 0 if direction == 1 else 1
        records = []
        seg_local = seg_local_start

        max_start = max(0.0, 1.0 - self.window)
        starts_qf = rng.uniform(0.0, max_start, size=self.n_samples)

        for start_qf in starts_qf:
            end_qf = start_qf + self.window
            lo = start_qf * q_tot
            hi = end_qf * q_tot
            m = (q >= lo) & (q < hi)
            if m.sum() < self.min_pts:
                continue

            if self.assign == "position_bin":
                center_qf = (start_qf + end_qf) / 2.0
                latent_class = _bin_class(center_qf)
            else:
                latent_class = 0

            scenario_id = spec.routing[dir_idx][latent_class]

            records.append(SegmentRecord(
                cell_id=cell_id, cycle=cycle, seg_local_id=seg_local,
                scenario_id=scenario_id,
                latent_class=latent_class,
                direction=direction,
                v=v[m], i=i[m], dt=dt[m], q=q[m],
                meta={"q_frac_lo": start_qf, "q_frac_hi": end_qf},
            ))
            seg_local += 1

        return records, seg_local

    def iter_segments(
        self,
        cell_id, cycle,
        dis_v, dis_i, dis_dt, dis_q,
        chg_v=None, chg_i=None, chg_dt=None, chg_q=None,
    ) -> Iterator[SegmentRecord]:
        rng = np.random.default_rng([self.seed, hash(cell_id) % (2**31), cycle])
        seg_local = 0

        if len(dis_v) >= self.min_pts:
            recs, seg_local = self._sample_segments(
                dis_v, dis_i, dis_dt, dis_q, -1, cell_id, cycle, seg_local, rng)
            yield from recs

        if chg_v is not None and len(chg_v) >= self.min_pts:
            cv_start = _detect_cv_start(chg_v, chg_i, self.cv_v_thresh, self.cv_cc_frac)
            cc_v  = chg_v[:cv_start]
            cc_i  = chg_i[:cv_start]
            cc_dt = chg_dt[:cv_start]
            cc_q  = chg_q[:cv_start]
            if len(cc_v) >= self.min_pts:
                recs, seg_local = self._sample_segments(
                    cc_v, cc_i, cc_dt, cc_q, +1, cell_id, cycle, seg_local, rng)
                yield from recs
