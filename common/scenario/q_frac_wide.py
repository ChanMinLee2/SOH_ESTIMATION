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

시나리오 이름·라우팅:
  chg_lo(0) chg_mid(1) chg_hi(2) | dis_hi(3) dis_mid(4) dis_lo(5)
  routing = [[2, 1, 0], [5, 4, 3]]

  [SOC 정합성 — 2026-07-30 수정]
  이름의 lo/mid/hi 는 항상 실측 SOC(전압) 수준을 가리켜야 한다: chg_lo=충전 초반
  (SOC 낮음) ... chg_hi=충전 막판(SOC 높음), dis_hi=방전 초반(SOC 높음) ...
  dis_lo=방전 막판(SOC 낮음) — 원조 qfrac.py 가 정확히 이렇게 구현돼 있다(방향별로
  _CHG_SEGS/_DIS_SEGS 를 따로 정의).
  본 파일은 방향 무관 단일 _ZONES(존 시작="hi", 존 끝="lo")를 쓰고 routing 반전으로
  방향차를 보정하는 방식으로 일반화됐는데, 예전 routing=[[0,1,2],[5,4,3]] 는 방전만
  올바르게 보정하고 충전은 못 채웠다(존 시작=q_frac≈0=충전 초반=SOC 낮음인데
  latent=2 라 구 routing 에서 scenario_id=2="chg_hi" 로 배정 — 이름과 실제 SOC가
  반대). 그 결과 이 파일로 생성된 모든 과거 런(예: `qfw_35%_20%` 등)의 chg_lo/chg_hi
  라벨이 뒤바뀌어 있었다 — 숫자(RMSE 등)는 존 자체 정의가 안 바뀌었으므로 영향
  없지만, "chg_hi 존"이라는 표현을 SOC 높은 쪽으로 해석했다면 그 해석만 틀렸다.
  routing[0]을 [2,1,0]으로 뒤집어 존 시작(latent=2)→scenario_id=0="chg_lo" 가
  되도록 고쳤다 — 방전(routing[1])은 원래도 맞았으므로 그대로 둠.

사용 예:
  python 4_hi_analysis/hi_correlation.py --seg-axis q_frac_wide \\
      --axis-config '{"q_frac_wide": {"n1": 0.4, "n2": 0.2, "n_samples": 4}}'
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter
from ._random_seg import sample_random_windows
# CV 구간 제거 비활성화로 더 이상 사용하지 않음 (iter_segments 참고)
# from .vwindow import _detect_cv_start

_SCENARIO_NAMES = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
# routing[dir_idx][latent_class] = scenario_id — dir_idx 0=충전, 1=방전.
# 충전: 존 시작(latent2)→chg_lo(0, SOC 낮음) ... 존 끝(latent0)→chg_hi(2, SOC 높음).
# 방전: 존 시작(latent2)→dis_hi(3, SOC 높음) ... 존 끝(latent0)→dis_lo(5, SOC 낮음).
_ROUTING = [[2, 1, 0], [5, 4, 3]]

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
        random_segment: bool = False,   # True: 구간 내 고정길이 랜덤 창 (설계 A)
        seg_len_pts: int = 20,          # 랜덤 창의 고정 관측 포인트 수 (q_tot 무관)
        random_seed: int = 42,          # 랜덤 재현성 시드
        assign: str = "position_bin",   # "position_bin"(기본, 존별 6시나리오 라우팅) |
                                         # "none"(시나리오 라우팅 없음, 방향만 구분해 2개) —
                                         # rcs.py와 동일 컨벤션. 존 경계(n1)·세그먼트 길이/개수
                                         # (n2/n_samples)·분모(q_tot, q_frac_ref면 q_ref+노이즈)는
                                         # 전혀 안 바뀜 — "이 세그먼트가 어느 존인지"를 모델에
                                         # 알려주는 시나리오 라우팅만 지운다. "시나리오 타이핑
                                         # 자체의 순수 기여도"를 재는 대조군 축 용도
                                         # (docs/260816_RESULTS.md §5 no_scen).
    ):
        if not (0.35 <= n1 <= 0.45):
            raise ValueError(
                f"q_frac_wide: n1은 [0.35, 0.45] 범위여야 합니다. 현재 n1={n1}"
            )
        if not (0 < n2 < n1):
            raise ValueError(
                f"q_frac_wide: 0 < n2 < n1 필요. 현재 n1={n1}, n2={n2}"
            )
        if random_segment and seg_len_pts < min_pts:
            raise ValueError(
                f"q_frac_wide: random_segment 시 seg_len_pts({seg_len_pts}) >= min_pts({min_pts}) 필요.")
        if assign not in ("position_bin", "none"):
            raise ValueError(
                f"q_frac_wide: assign은 'position_bin'|'none' 중 하나여야 합니다. 현재 assign={assign!r}")
        self.n1 = n1
        self.n2 = n2
        self.n_samples = n_samples
        self.min_pts = min_pts
        self.cv_v_thresh = cv_v_thresh
        self.cv_cc_frac = cv_cc_frac
        self.random_segment = bool(random_segment)
        self.seg_len_pts = int(seg_len_pts)
        self.random_seed = int(random_seed)
        self.assign = assign

        # 진단용 카운터 (min_pts 생존율 계산) — iter_segments의 공개 동작에는 영향 없음.
        # scenario_name -> count. reset_counters()로 초기화 후 여러 셀에 걸쳐 누적 가능.
        self.n_attempted: dict[str, int] = {}
        self.n_yielded: dict[str, int] = {}
        # scenario_name -> [원시 포인트 수, ...] — 시도한 모든 후보의 실제 m.sum() 기록.
        # 이 분포만 있으면 재스캔 없이 임의의 min_pts 값에서 생존율을
        # (arr >= threshold).mean() 으로 즉시 계산할 수 있다.
        self.candidate_n_points: dict[str, list] = {}
        # random_segment 누락 통계 — scenario_name -> [covered_pts, total_pts]
        self.coverage: dict[str, list] = {}

    def reset_counters(self) -> None:
        """카운터 초기화 (seg_diagnose.py 등 진단 스크립트용)."""
        self.n_attempted = {}
        self.n_yielded = {}
        self.candidate_n_points = {}
        self.coverage = {}

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

    # ── 세그먼트 배치 훅 (2026-09-03) ────────────────────────────────────────
    # 고정폭 격자 모드에서 "이 존을 어떤 (시작, 끝) 조각들로 자를지"를 결정하는 훅.
    # 기본구현 = 기존 동작 그대로(길이 n2 고정, _start_positions 균등격자, 상한 배타).
    # QFracRefSegmenter가 n2 범위 모드(n2_start/n2_end)에서 이 훅만 오버라이드해
    # 가변 길이 타일링을 반환한다 — _extract 본체는 건드리지 않는다.

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
        """(start_qf, end_qf, include_hi, extra_meta) 목록.

        include_hi=True면 상한을 포함(q <= hi). extra_meta는 그 세그먼트의 meta에
        덧붙일 dict(없으면 None).
        q/q_tot은 기본구현에선 안 쓰지만, 오버라이드 쪽이 "이 조각에 실제로 포인트가
        몇 개 들어가는지"를 보고 배치를 조정(min_pts 병합 등)할 수 있게 넘겨준다.
        """
        return [(float(s), float(s) + self.n2, False, None)
                for s in self._start_positions(zone_start, zone_end)]

    def _track_zone_coverage(self) -> bool:
        """True면 고정폭 격자 모드에서도 존 포인트 커버리지를 self.coverage에 집계.

        기본 False — 격자 모드는 설계상 존을 다 덮지 않으므로(균등격자 + 고정 n2)
        집계해봐야 의미가 없고, 기존 런의 출력(coverage_stats.txt 생성 여부)도
        바뀌지 않게 하기 위함. n2 범위 모드(100% 커버리지가 조건)에서만 켠다.
        """
        return False

    # ── 분모(q_tot) 산출 훅 ──────────────────────────────────────────────────
    # 기본구현: 이 세션(사이클) 자신의 전류적산 최종값 — q_frac_wide 원래 정의.
    # QFracRefSegmenter(q_frac_ref.py)가 이 메서드만 오버라이드해 과거 레퍼런스
    # 기반 분모를 반환한다(docs/SOC.md §6 Phase 3 — 서브클래스 + stateful 캐시).

    def _normalizer(
        self,
        direction: int,
        cell_id: str,
        cycle: int,
        q: np.ndarray,
    ) -> float:
        return float(q[-1]) if len(q) > 0 else 0.0

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
        q_tot = self._normalizer(direction, cell_id, cycle, q)
        if q_tot < 0.05:
            return [], seg_local_start

        spec      = self.get_spec()
        dir_idx   = 0 if direction == 1 else 1
        bounds    = self._zone_bounds()
        records: list[SegmentRecord] = []
        seg_local = seg_local_start

        for zone_name, latent_class in _ZONES:
            zone_start, zone_end = bounds[zone_name]
            # assign="none" 이면 latent_class(존)를 라우팅에 반영하지 않고 항상 0으로
            # 고정 — 모델에 노출되는 scenario_id/이름은 방향만 구분(chg/dis), 존 경계
            # 자체(zone_start/zone_end)는 그대로 사용해 세그먼트 위치는 안 바뀐다.
            _latent     = latent_class if self.assign == "position_bin" else 0
            scenario_id = spec.routing[dir_idx][_latent]
            sname       = spec.scenario_names[scenario_id]

            if self.random_segment:
                # ── 랜덤 모드: 구간 안에서 고정길이(seg_len_pts) 랜덤 창 (설계 A) ──
                # 구간 경계는 q_frac(기존 방식). 창 길이는 관측 포인트 고정 → q_tot 무관.
                is_last = (zone_name == "lo")   # lo=[1-n1, 1.0] → 상한 포함
                lo_q, hi_q = zone_start * q_tot, zone_end * q_tot
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
                        scenario_id=scenario_id, latent_class=_latent,
                        direction=direction,
                        v=v[w], i=i[w], dt=dt[w], q=q[w],
                        meta={"zone": zone_name, "random": True,
                              "q_frac_lo": float(q[w[0]] / q_tot),
                              "q_frac_hi": float(q[w[-1]] / q_tot)},
                    ))
                    seg_local += 1
                continue

            # ── 기존(고정폭 격자) 모드 ──────────────────────────────────────────
            plan = self._segment_plan(
                zone_start, zone_end, cell_id, cycle, direction, zone_name, q, q_tot)
            if not plan:
                continue

            track_cov = self._track_zone_coverage()
            if track_cov:
                # 존 전체 포인트(상·하한 모두 포함) 대비 실제로 방출된 세그먼트가
                # 덮은 포인트 비율 — 겹침은 1회만 집계.
                zmask   = (q >= zone_start * q_tot) & (q <= zone_end * q_tot)
                covered = np.zeros(len(q), dtype=bool)

            for start_qf, end_qf, include_hi, extra_meta in plan:
                lo_q   = start_qf * q_tot
                hi_q   = end_qf   * q_tot
                m      = ((q >= lo_q) & (q <= hi_q)) if include_hi else ((q >= lo_q) & (q < hi_q))
                n_pts  = int(m.sum())
                self.n_attempted[sname] = self.n_attempted.get(sname, 0) + 1
                self.candidate_n_points.setdefault(sname, []).append(n_pts)
                if n_pts < self.min_pts:
                    continue
                self.n_yielded[sname] = self.n_yielded.get(sname, 0) + 1
                if track_cov:
                    covered |= m

                _meta = {
                    "zone":      zone_name,
                    "q_frac_lo": start_qf,
                    "q_frac_hi": end_qf,
                    "seg_len":   float(end_qf - start_qf),
                }
                if extra_meta:
                    _meta.update(extra_meta)
                records.append(SegmentRecord(
                    cell_id=cell_id,
                    cycle=cycle,
                    seg_local_id=seg_local,
                    scenario_id=scenario_id,
                    latent_class=_latent,
                    direction=direction,
                    v=v[m], i=i[m], dt=dt[m], q=q[m],
                    meta=_meta,
                ))
                seg_local += 1

            if track_cov:
                c = self.coverage.setdefault(sname, [0, 0])
                c[0] += int((covered & zmask).sum())
                c[1] += int(zmask.sum())

        return records, seg_local

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def get_spec(self) -> ScenarioSpec:
        params = {"n1": self.n1, "n2": self.n2, "n_samples": self.n_samples,
                  "random_segment": self.random_segment, "seg_len_pts": self.seg_len_pts,
                  "min_pts": self.min_pts, "assign": self.assign}
        if self.assign == "none":
            # 시나리오 라우팅을 감춘 대조군: 존 구분 없이 방향만 2개.
            # (docs/260816_RESULTS.md §5 no_scen) — n1/n2/n_samples/분모는 그대로.
            return ScenarioSpec(
                axis="q_frac_wide",
                n_scenarios=2,
                scenario_names=["chg", "dis"],
                n_classes=1,
                class_names=["all"],
                routing=[[0], [1]],
                classifier_default="none",
                params=params,
            )
        return ScenarioSpec(
            axis="q_frac_wide",
            n_scenarios=6,
            scenario_names=_SCENARIO_NAMES,
            n_classes=3,
            class_names=["lo", "mid", "hi"],
            routing=_ROUTING,
            classifier_default="mlp_probe",
            params=params,
        )

    def _rng_for(self, cell_id: str, cycle: int, direction: int) -> "np.random.Generator | None":
        """random_segment 재현성: (seed, cell, cycle, direction) 기반 결정론적 rng."""
        if not self.random_segment:
            return None
        import zlib
        dir_key  = 0 if direction > 0 else 1   # 시드는 비음수여야 함
        cell_key = zlib.crc32(str(cell_id).encode())   # 프로세스 무관 결정론적 해시
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

        # 충전 (CC+CV 전체 사용 — 100% = 해당 사이클 실제 완충 용량과 일치시키기 위해
        # CV 구간 제거 로직을 비활성화함. cv_v_thresh/cv_cc_frac는 더 이상 쓰이지 않음.)
        if chg_v is not None and len(chg_v) >= self.min_pts:
            recs, seg_local = self._extract(
                chg_v, chg_i, chg_dt, chg_q, +1, cell_id, cycle, seg_local,
                rand_rng=self._rng_for(cell_id, cycle, +1))
            yield from recs
