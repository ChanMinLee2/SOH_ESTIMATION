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

[SOC 정합성 — 2026-07-30 수정]
  _bin_class 는 q_frac(그 방향 진행률) 중심값을 그대로 lo(0)/mid(1)/hi(2)에
  매핑한다 — q_frac≈0(그 방향 시작점)이 lo. 충전이면 시작점=SOC 낮음이라 이름이
  맞지만(chg_lo), 방전이면 시작점=배터리가 아직 꽉 찬 SOC 높은 시점이라 "lo"라는
  이름이 반대가 된다. 예전 routing=[[0,1,2],[5,4,3]] 은 q_frac_wide 에서 가져온
  것인데, q_frac_wide 는 정반대 컨벤션(q_frac≈0 이 hi)이라 그 routing 이 거기선
  방전에서 맞았을 뿐 여기 재사용할 근거가 아니었다. routing[1]을 [3,4,5]로 고쳐
  class0(lo, 방전 시작=SOC 높음)→dis_hi(3), class2(hi, 방전 끝=SOC 낮음)→dis_lo(5)
  가 되도록 정정했다 — 충전(routing[0])은 원래도 맞았으므로 그대로 둠.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter

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
        axis_name: str = "rcs",   # 저장 경로/식별용 axis 이름. REGISTRY에 별칭으로
                                   # 등록해 같은 클래스를 다른 파라미터 프리셋으로 쓸 때
                                   # (예: "random") --axis-config로 넘겨 기존 "rcs" 캐시와
                                   # 경로 충돌 없이 분리 저장한다(docs/260811_RESULTS.md 실험2 참고).
        placement: str = "random",   # "random"(기본, 매 사이클 독립 균등분포 시작점) |
                                      # "grid"(결정론적 등간격 시작점 — [0, 1-window] 구간에
                                      # n_samples개를 np.linspace로 균등 배치, 커버리지 100%
                                      # 보장). "random"은 겹침·경계 미커버로 평균 커버리지가
                                      # window/n_samples 조합에 따라 100%에 못 미칠 수 있다
                                      # (예: window=0.2,n_samples=12 → 실측 평균 86%,
                                      # docs/260811_RESULTS.md 2026-08-14 분석 참고).
    ):
        self.n_samples = n_samples
        self.window = window
        self.unit = unit
        self.assign = assign
        self.seed = seed
        self.min_pts = min_pts
        self.cv_v_thresh = cv_v_thresh
        self.cv_cc_frac = cv_cc_frac
        self.axis_name = axis_name
        if placement not in ("random", "grid"):
            raise ValueError(f"rcs: placement은 'random'|'grid' 중 하나여야 합니다. 현재={placement!r}")
        self.placement = placement

    def get_spec(self) -> ScenarioSpec:
        if self.assign == "position_bin":
            n_classes = 3
            class_names = ["lo", "mid", "hi"]
            routing = [[0, 1, 2], [3, 4, 5]]  # SOC 정합성 수정 — 모듈 docstring 참조
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
            axis=self.axis_name,
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
                "placement": self.placement,
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
        if self.placement == "grid":
            # 결정론적 등간격 배치 — window(예:0.2) > 간격(max_start/(n_samples-1))이면
            # 인접 세그먼트끼리 겹치면서 [0, 1-window] 전체를 빈틈없이 덮는다. n_samples=1이면
            # 중앙 시작점 하나만 사용(q_frac_wide._start_positions와 동일 관례).
            if self.n_samples <= 1:
                starts_qf = np.array([max_start / 2.0])
            else:
                starts_qf = np.linspace(0.0, max_start, self.n_samples)
        else:
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

        # 충전 (CC+CV 전체 사용 — q_frac_wide.py와 동일 원칙, 2026-08-14 정정).
        # 예전엔 여기서 자체적으로 _detect_cv_start로 CV 구간을 항상 잘라냈는데, 이러면
        # hi_correlation.py의 범용 --exclude-cv 플래그(기본 False=CV 포함)와 무관하게
        # random/random_grid 축만 CV가 항상 빠지는 축 간 불일치가 생겼다(q_frac_ref는
        # --exclude-cv 없이 돌린 모든 baseline에서 CV를 포함해왔음, docs/260811_RESULTS.md
        # 참고). CV 포함/제외는 이제 다른 축과 동일하게 hi_correlation.py 호출부의
        # --exclude-cv 하나로만 통제한다(cv_v_thresh/cv_cc_frac 파라미터는 하위호환을 위해
        # 시그니처에 남겨두되 더 이상 쓰이지 않음).
        if chg_v is not None and len(chg_v) >= self.min_pts:
            recs, seg_local = self._sample_segments(
                chg_v, chg_i, chg_dt, chg_q, +1, cell_id, cycle, seg_local, rng)
            yield from recs
