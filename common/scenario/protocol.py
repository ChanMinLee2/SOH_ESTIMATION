"""
common/scenario/protocol.py — 축 1: 프로토콜 경계 기반 세그멘터.

충전 내 |ΔI| > i_step_thresh (C-rate 환산) 로 CC 단계 전환을 검출하고,
V≈V_max & I 감소 구간을 CC→CV 전환으로 판별한다.

MIT 2단 충전 (C1(SOC1%)-C2) + CV 꼬리, HUST 다단 프로토콜을 자동 분해.
단계 수를 max_steps로 캡하여 n_scenarios = 2 × max_steps 고정
(qfrac의 6 시나리오와 같은 크기 → 공정한 모델 크기 비교).

분류기: rule — 세그먼트 내 I 통계(i_std≈0 → CC, V포화 & i 감소 → CV)로
결정론적 판별, 학습 불필요.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter

_MIN_PTS = 10
_I_STEP_THRESH_C = 0.5  # 정격 용량 대비 C-rate 단위 전류 변화 임계값


def _detect_steps(
    i: np.ndarray,
    dt: np.ndarray,
    nom_cap: float = 1.1,
    i_step_thresh_c: float = _I_STEP_THRESH_C,
) -> list[int]:
    """
    전류 배열에서 CC 단계 전환점 인덱스 반환.
    |ΔI| > i_step_thresh_c × nom_cap 인 위치를 전환점으로 본다.
    반환값: [0, idx_switch1, idx_switch2, ..., len(i)] (경계 포함)
    """
    thresh = i_step_thresh_c * nom_cap
    di = np.abs(np.diff(i, prepend=i[0]))
    # 전환 후 짧은 과도 구간 무시: 5개 샘플 이내 재발 병합
    switch_raw = np.where(di > thresh)[0]
    if len(switch_raw) == 0:
        return [0, len(i)]
    # 클러스터링: 연속 전환점 → 첫 인덱스만 유지
    switches = [switch_raw[0]]
    for s in switch_raw[1:]:
        if s - switches[-1] > 5:
            switches.append(s)
    return [0] + switches + [len(i)]


def _is_cv(v: np.ndarray, i: np.ndarray, v_max_thresh: float = 0.98) -> bool:
    """세그먼트가 CV 구간인지 판별: V 포화 & I 단조 감소."""
    v_ratio = float(np.mean(v)) / (float(np.max(v)) + 1e-9)
    i_decreasing = float(np.polyfit(np.arange(len(i)), i, 1)[0]) < 0
    return v_ratio > v_max_thresh and i_decreasing


class ProtocolSegmenter(Segmenter):
    """프로토콜 경계(CC 단계 전환 / CC→CV) 기반 세그멘터."""

    name = "protocol"

    def __init__(
        self,
        max_steps: int = 3,
        i_step_thresh_c: float = _I_STEP_THRESH_C,
        nom_cap: float = 1.1,
        min_pts: int = _MIN_PTS,
    ):
        self.max_steps = max_steps
        self.i_step_thresh_c = i_step_thresh_c
        self.nom_cap = nom_cap
        self.min_pts = min_pts
        self._n_scen = 2 * max_steps
        # scenario_names: chg_step0..N-1, dis_step0..N-1
        self._scen_names = (
            [f"chg_step{k}" for k in range(max_steps)]
            + [f"dis_step{k}" for k in range(max_steps)]
        )
        # routing: charge→0..max_steps-1, discharge→max_steps..2*max_steps-1
        # class == step index (0-indexed)
        self._routing = [
            list(range(max_steps)),                            # charge: class k → scenario k
            list(range(max_steps, 2 * max_steps)),            # discharge: class k → scenario max_steps+k
        ]

    def get_spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            axis="protocol",
            n_scenarios=self._n_scen,
            scenario_names=self._scen_names,
            n_classes=self.max_steps,
            class_names=[f"step{k}" for k in range(self.max_steps)],
            routing=self._routing,
            classifier_default="rule",
            params={
                "max_steps":        self.max_steps,
                "i_step_thresh_c":  self.i_step_thresh_c,
                "nom_cap":          self.nom_cap,
            },
        )

    def _split_direction(
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
        """단방향(dis/chg) 배열을 CC 단계로 분할, SegmentRecord 목록 반환."""
        boundaries = _detect_steps(i, dt, self.nom_cap, self.i_step_thresh_c)
        dir_idx = 0 if direction == 1 else 1
        records: list[SegmentRecord] = []
        seg_local = seg_local_start

        valid_k = 0
        for k in range(len(boundaries) - 1):
            if valid_k >= self.max_steps:
                break
            lo, hi = boundaries[k], boundaries[k + 1]
            if hi - lo < self.min_pts:
                continue
            vs = v[lo:hi]; is_ = i[lo:hi]; dts = dt[lo:hi]; qs = q[lo:hi]
            latent_class = valid_k
            scenario_id = self._routing[dir_idx][latent_class]
            records.append(SegmentRecord(
                cell_id=cell_id,
                cycle=cycle,
                seg_local_id=seg_local,
                scenario_id=scenario_id,
                latent_class=latent_class,
                direction=direction,
                v=vs, i=is_, dt=dts, q=qs,
                meta={"step": valid_k, "is_cv": _is_cv(vs, is_)},
            ))
            seg_local += 1
            valid_k += 1

        return records, seg_local

    def iter_segments(
        self,
        cell_id: str,
        cycle: int,
        dis_v, dis_i, dis_dt, dis_q,
        chg_v=None, chg_i=None, chg_dt=None, chg_q=None,
    ) -> Iterator[SegmentRecord]:
        seg_local = 0

        # 방전
        if len(dis_v) >= self.min_pts:
            recs, seg_local = self._split_direction(
                dis_v, dis_i, dis_dt, dis_q, -1, cell_id, cycle, seg_local)
            yield from recs

        # 충전
        if chg_v is not None and len(chg_v) >= self.min_pts:
            recs, seg_local = self._split_direction(
                chg_v, chg_i, chg_dt, chg_q, +1, cell_id, cycle, seg_local)
            yield from recs
