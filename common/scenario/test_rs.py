"""
common/scenario/test_rs.py — 완전 랜덤 구간·랜덤 길이 세그먼트 (테스트 전용).

시작점: q_frac [0, 1-length] 균등분포
길이  : q_frac [min_len, max_len] 균등분포 (겹침 허용)
할당  : 충전(0) / 방전(1) 만 구분 — n_scenarios=2

실제 HI 추출은 tmp_make_test_rs.py 스탠드얼론 스크립트 사용.
이 클래스는 ScenarioSpec 등록 및 get_segmenter("test_rs") 호환용.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter
from .vwindow import _detect_cv_start


class TestRSSegmenter(Segmenter):
    """완전 랜덤 구간·랜덤 길이 세그먼트 샘플러."""

    name = "test_rs"

    def __init__(
        self,
        n_samples: int   = 8,     # 방향당 사이클당 샘플 수
        min_len:   float = 0.05,
        max_len:   float = 0.40,
        seed:      int   = 42,
        min_pts:   int   = 10,
        cv_v_thresh: float = 3.60,
        cv_cc_frac:  float = 0.80,
    ):
        self.n_samples   = n_samples
        self.min_len     = min_len
        self.max_len     = max_len
        self.seed        = seed
        self.min_pts     = min_pts
        self.cv_v_thresh = cv_v_thresh
        self.cv_cc_frac  = cv_cc_frac

    def get_spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            axis="test_rs",
            n_scenarios=2,
            scenario_names=["chg", "dis"],
            n_classes=1,
            class_names=["any"],
            routing=[[0], [1]],
            classifier_default="none",
            params={
                "n_samples": self.n_samples,
                "min_len":   self.min_len,
                "max_len":   self.max_len,
                "seed":      self.seed,
            },
        )

    def _sample_dir(
        self,
        v, i, dt, q,
        direction, cell_id, cycle, rng,
    ) -> list[SegmentRecord]:
        q_tot = float(q[-1]) if len(q) > 0 else 0.0
        if q_tot < 0.05:
            return []
        spec    = self.get_spec()
        dir_idx = 0 if direction == 1 else 1
        records = []
        lengths = rng.uniform(self.min_len, self.max_len, size=self.n_samples)
        for k, length in enumerate(lengths):
            max_start = max(0.0, 1.0 - length)
            start_qf  = rng.uniform(0.0, max_start)
            end_qf    = start_qf + length
            m = (q >= start_qf * q_tot) & (q < end_qf * q_tot)
            if int(m.sum()) < self.min_pts:
                continue
            records.append(SegmentRecord(
                cell_id=cell_id, cycle=cycle, seg_local_id=k,
                scenario_id=spec.routing[dir_idx][0],
                latent_class=0,
                direction=direction,
                v=v[m], i=i[m], dt=dt[m], q=q[m],
                meta={"q_frac_lo": float(start_qf), "q_frac_hi": float(end_qf)},
            ))
        return records

    def iter_segments(
        self,
        cell_id, cycle,
        dis_v, dis_i, dis_dt, dis_q,
        chg_v=None, chg_i=None, chg_dt=None, chg_q=None,
    ) -> Iterator[SegmentRecord]:
        rng = np.random.default_rng([self.seed, hash(cell_id) % (2**31), cycle])
        if len(dis_v) >= self.min_pts:
            yield from self._sample_dir(dis_v, dis_i, dis_dt, dis_q, -1, cell_id, cycle, rng)
        if chg_v is not None and len(chg_v) >= self.min_pts:
            cv_start = _detect_cv_start(chg_v, chg_i, self.cv_v_thresh, self.cv_cc_frac)
            cc_v = chg_v[:cv_start]; cc_i = chg_i[:cv_start]
            cc_dt = chg_dt[:cv_start]; cc_q = chg_q[:cv_start]
            if len(cc_v) >= self.min_pts:
                yield from self._sample_dir(cc_v, cc_i, cc_dt, cc_q, +1, cell_id, cycle, rng)
