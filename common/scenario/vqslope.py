"""
common/scenario/vqslope.py — 축: 기울기(dV/dQ · dQ/dV) 형상 기반 세그멘터.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
설계 근거 (리뷰어용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[문제] q_frac_wide(누적 전하량 기반)의 한계
  존 경계를 q_frac = q/q_tot 으로 정의하면, 그 세그먼트를 만들기 위해 이미
  q_tot(해당 사이클 실측 용량 ≈ SOH)을 알아야 한다. 즉 존 라벨이 라벨(SOH)의
  함수가 되어, "형상에서 위치를 복원했다"와 "형상에서 SOH 단서를 감지했다"를
  구분할 수 없다. 세그멘테이션 축의 요건은 (i) 라벨의 함수가 아닐 것,
  (ii) 스니펫 내용만으로 복원 가능할 것 — q_frac은 (i)에서 탈락한다.

[해법] 순간량(instantaneous) 랜드마크로 경계를 정의
  경계를 "얼마나 진행됐는가(누적)"가 아니라 "지금 곡선 형상이 어떤 상태인가"로
  정한다. LFP 방전/충전 곡선은 급경사 → 평탄(플래토) → 급경사 순서이므로,
  플래토 진입·이탈이라는 상전이 이벤트의 위치로 3구간을 나눈다.

  vwindow(고정 전압 경계)가 노화에 강건한 것과 같은 이유로, 플래토 진입/이탈
  전압은 LFP 노화에서 거의 변하지 않는다. vqslope는 그 랜드마크를 절대 전압값이
  아니라 "기울기가 평탄한가/급한가"라는 형상 판정으로 잡으므로, 미세한 전압
  시프트에도 더 안정적일 것으로 기대된다.

[존 정의 — 곡선 진행 순서(q_local 증가 방향)]
  head    : 곡선 시작 ~ 플래토 진입    (|dV/dQ| 큼, 급경사 초입)  → latent_class 2
  plateau : 플래토 진입 ~ 이탈          (|dV/dQ| < θ_flat, 평탄)   → latent_class 1 (mid)
  tail    : 플래토 이탈 ~ 곡선 끝        (|dV/dQ| 큼, 급경사 말단)  → latent_class 0

  이 (head, mid, tail) 매핑은 q_frac_wide 의 (hi=[0,n1], mid=중앙, lo=[1-n1,1])
  컨벤션과 정확히 정렬되어, 모델 구조(n_scenarios=6, routing)를 그대로 재사용한다
  — routing 도 q_frac_wide 와 동일하게 SOC 정합성 수정을 적용했다(아래 참조).

[모드] DVA(dV/dQ) vs ICA(dQ/dV)
  두 모드 모두 "플래토의 q_local 범위 [q_entry, q_exit]"를 구하고 이후 3분할은
  공통이다. 플래토 범위를 찾는 방법만 다르다:
    dva : |dV/dQ| < θ_flat 인 Q-빈들의 q 범위.
    ica : dQ/dV 주 피크의 FWHM(v_left..v_right) 안에 드는 원시 포인트의 q 범위.

[시나리오 구조 — q_frac_wide 와 동일 (2026-07-30 routing SOC 정합성 수정 포함)]
  chg_lo(0) chg_mid(1) chg_hi(2) | dis_hi(3) dis_mid(4) dis_lo(5)
  routing = [[2, 1, 0], [5, 4, 3]]
  head(곡선 시작=충전 초반/방전 초반)는 충전이면 SOC 낮음(→chg_lo), 방전이면
  SOC 높음(→dis_hi)이어야 한다. 예전 routing=[[0,1,2],[5,4,3]] 은 head(latent=2)를
  충전에서 scenario_id=2="chg_hi" 로 배정해 이름과 실제 SOC가 반대였다(방전은
  우연히 맞았음) — q_frac_wide 와 동일한 버그였다. routing[0]을 [2,1,0]으로
  고쳐 head(latent=2)→chg_lo(0) 가 되도록 정정했다.

[실패 모드 — 반드시 진단할 것]
  플래토 미검출(짧은 창, 노이즈) 시 그 방향의 3존이 통째로 스킵된다. 이는
  q_frac_wide 의 min_pts 절벽과 형태만 다른 같은 성격의 표본 손실이므로,
  seg_diagnose 로 플래토 검출 실패율과 존별 생존율을 먼저 측정해야 한다.

사용 예:
  python 4_hi_analysis/hi_correlation.py --seg-axis vqslope \\
      --axis-config '{"vqslope": {"mode": "dva", "n_samples": 1}}'
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter
from ._curves import _build_vq_curve, _build_ica_seg, _peak_fwhm_asym, THETA_FLAT
from ._random_seg import sample_random_windows

_SCENARIO_NAMES = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
_ROUTING = [[2, 1, 0], [5, 4, 3]]  # SOC 정합성 수정 — 모듈 docstring 참조

# (zone_name, latent_class) — 곡선 진행 순서(q_local 증가): head=hi, plateau=mid, tail=lo
_ZONES: list[tuple[str, int]] = [
    ("head",    2),   # 급경사 초입
    ("plateau", 1),   # 평탄
    ("tail",    0),   # 급경사 말단
]


class VQSlopeSegmenter(Segmenter):
    """기울기(dV/dQ·dQ/dV) 형상 기반 세그멘터 (플래토 진입/이탈 3분할)."""

    name = "vqslope"

    def __init__(
        self,
        mode: str = "dva",          # "dva"(dV/dQ) | "ica"(dQ/dV)
        n_samples: int = 1,         # 존당 세그먼트 수 (균등 등분 또는 랜덤 창 개수)
        theta_flat: float = THETA_FLAT,   # dva 플래토 임계값 |dV/dQ| < θ
        ica_peak_frac: float = 0.5,       # ica: 피크 높이 대비 이 비율 이상을 plateau로 (FWHM=0.5)
        min_pts: int = 10,
        random_segment: bool = False,     # True: 존 내부 고정길이 랜덤 창 (설계 A)
        seg_len_pts: int = 20,            # 랜덤 창의 고정 관측 포인트 수 (q_tot 무관)
        random_seed: int = 42,            # 랜덤 재현성 시드
    ):
        mode = str(mode).lower()
        if mode not in ("dva", "ica"):
            raise ValueError(f"vqslope: mode는 'dva'|'ica' 여야 합니다. 현재 mode={mode}")
        if n_samples < 1:
            raise ValueError(f"vqslope: n_samples >= 1 필요. 현재 {n_samples}")
        if random_segment and seg_len_pts < min_pts:
            raise ValueError(
                f"vqslope: random_segment 시 seg_len_pts({seg_len_pts}) >= min_pts({min_pts}) 필요.")
        self.mode = mode
        self.n_samples = int(n_samples)
        self.theta_flat = float(theta_flat)
        self.ica_peak_frac = float(ica_peak_frac)
        self.min_pts = int(min_pts)
        self.random_segment = bool(random_segment)
        self.seg_len_pts = int(seg_len_pts)
        self.random_seed = int(random_seed)

        # 진단용 카운터 (q_frac_wide 와 동일 인터페이스) — seg_diagnose 재사용.
        self.n_attempted: dict[str, int] = {}
        self.n_yielded: dict[str, int] = {}
        self.candidate_n_points: dict[str, list] = {}
        # 플래토 검출 실패(그 방향 3존 전부 스킵) 횟수 — vqslope 고유 진단.
        self.n_plateau_fail: dict[str, int] = {"charge": 0, "discharge": 0}
        # random_segment 누락 통계 — scenario_name -> [covered_pts, total_pts]
        self.coverage: dict[str, list] = {}

    def reset_counters(self) -> None:
        self.n_attempted = {}
        self.n_yielded = {}
        self.candidate_n_points = {}
        self.n_plateau_fail = {"charge": 0, "discharge": 0}
        self.coverage = {}

    # ── 플래토 q_local 범위 검출 ─────────────────────────────────────────────

    def _plateau_q_range(
        self, v: np.ndarray, ims: np.ndarray, dts: np.ndarray, q_local: np.ndarray, q_tot: float
    ) -> tuple[float, float] | None:
        """세그먼트 곡선에서 플래토의 q_local 범위 [q_entry, q_exit] 반환. 실패 시 None.

        dva: |dV/dQ| < θ_flat 인 Q-빈들의 qm 범위.
        ica: dQ/dV 주 피크의 FWHM(v_left..v_right) 안에 드는 원시 포인트의 q_local 범위.
        """
        if self.mode == "dva":
            qm, v_sm, dvdq_sm, q_tot_c = _build_vq_curve(v, ims, dts)
            fin = np.isfinite(dvdq_sm) & np.isfinite(qm)
            if fin.sum() < 3 or q_tot_c < 0.005:
                return None
            plt_mask = fin & (np.abs(dvdq_sm) < self.theta_flat)
            if plt_mask.sum() < 2:
                return None
            q_plt = qm[plt_mask]
            return float(q_plt.min()), float(q_plt.max())

        # ── ica ──────────────────────────────────────────────────────────
        vmids, dqdv_sm = _build_ica_seg(v, ims, dts)
        if len(vmids) < 4:
            return None
        pk = int(np.argmax(dqdv_sm))
        if dqdv_sm[pk] <= 0:
            return None
        # 피크 높이의 ica_peak_frac(=FWHM 기본 0.5) 이상 구간의 전압 경계
        thr = self.ica_peak_frac * float(dqdv_sm[pk])
        above = dqdv_sm >= thr
        # 피크를 포함하는 연속 True 구간의 좌우 경계 전압
        li = pk
        while li - 1 >= 0 and above[li - 1]:
            li -= 1
        ri = pk
        while ri + 1 < len(above) and above[ri + 1]:
            ri += 1
        v_left, v_right = float(vmids[li]), float(vmids[ri])
        # 원시 포인트 중 전압이 피크영역 [v_left, v_right] 안인 것 → q_local 범위
        in_peak = (v >= v_left) & (v <= v_right)
        if in_peak.sum() < 2:
            return None
        q_in = q_local[in_peak]
        return float(q_in.min()), float(q_in.max())

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
        rand_rng: np.random.Generator | None = None,
    ) -> tuple[list[SegmentRecord], int]:
        # 세그먼트 로컬 누적 전하 (곡선 진행축). _build_vq_curve 내부와 동일 정의.
        q_local = np.cumsum(np.abs(i) * dt) / 3600.0
        q_tot   = float(q_local[-1]) if len(q_local) > 0 else 0.0
        dir_key = "charge" if direction == 1 else "discharge"
        if q_tot < 0.05:
            return [], seg_local_start

        plt_rng = self._plateau_q_range(v, np.abs(i), dt, q_local, q_tot)
        if plt_rng is None:
            # 플래토 미검출 → 이 방향 3존 전부 스킵 (진단용으로 기록)
            self.n_plateau_fail[dir_key] += 1
            return [], seg_local_start
        q_entry, q_exit = plt_rng
        # 안전: entry <= exit 보장 및 곡선 범위로 클램프
        q_entry = max(0.0, min(q_entry, q_tot))
        q_exit  = max(q_entry, min(q_exit, q_tot))

        spec      = self.get_spec()
        dir_idx   = 0 if direction == 1 else 1
        # 존별 q_local 경계: [0, q_entry] / [q_entry, q_exit] / [q_exit, q_tot]
        zone_qbounds = {
            "head":    (0.0,     q_entry),
            "plateau": (q_entry, q_exit),
            "tail":    (q_exit,  q_tot),
        }

        records: list[SegmentRecord] = []
        seg_local = seg_local_start

        for zone_name, latent_class in _ZONES:
            z_lo, z_hi = zone_qbounds[zone_name]
            scenario_id = spec.routing[dir_idx][latent_class]
            sname       = _SCENARIO_NAMES[scenario_id]

            if self.random_segment:
                # ── 랜덤 모드: 존 안에서 고정길이(seg_len_pts) 랜덤 창 (설계 A) ──
                # 존 경계(형상 랜드마크)는 라벨 독립. 창 길이는 관측 포인트 고정 → q_tot 무관.
                is_last = (zone_name == "tail")
                zmask = ((q_local >= z_lo) & (q_local <= z_hi)) if is_last \
                    else ((q_local >= z_lo) & (q_local < z_hi))
                zone_idx = np.where(zmask)[0]
                windows, cov, tot = sample_random_windows(
                    zone_idx, self.seg_len_pts, self.n_samples, self.min_pts, rand_rng)
                # 누락 통계 누적
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
                        meta={"zone": zone_name, "mode": self.mode, "random": True,
                              "q_lo": float(q_local[w[0]]), "q_hi": float(q_local[w[-1]]),
                              "q_entry": q_entry, "q_exit": q_exit},
                    ))
                    seg_local += 1
                continue

            # ── 기존(등분) 모드: 존 내부를 n_samples 등분 (n_samples=1 → 존 전체 1개) ──
            edges = np.linspace(z_lo, z_hi, self.n_samples + 1)
            for s in range(self.n_samples):
                lo_q, hi_q = float(edges[s]), float(edges[s + 1])
                # 마지막 서브구간은 상한 포함
                if s == self.n_samples - 1:
                    m = (q_local >= lo_q) & (q_local <= hi_q)
                else:
                    m = (q_local >= lo_q) & (q_local < hi_q)
                n_pts = int(m.sum())
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
                        "q_lo":      lo_q,
                        "q_hi":      hi_q,
                        "q_entry":   q_entry,
                        "q_exit":    q_exit,
                        "mode":      self.mode,
                    },
                ))
                seg_local += 1

        return records, seg_local

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def get_spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            axis="vqslope",
            n_scenarios=6,
            scenario_names=_SCENARIO_NAMES,
            n_classes=3,
            class_names=["lo", "mid", "hi"],
            routing=_ROUTING,
            classifier_default="mlp_probe",
            params={
                "mode":       self.mode,
                "n_samples":  self.n_samples,
                "theta_flat": self.theta_flat,
                "random_segment": self.random_segment,
                "seg_len_pts":    self.seg_len_pts,
            },
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

        # 충전 (CC+CV 전체 — q_frac_wide 와 동일하게 CV 분리 안 함)
        if chg_v is not None and len(chg_v) >= self.min_pts:
            recs, seg_local = self._extract(
                chg_v, chg_i, chg_dt, chg_q, +1, cell_id, cycle, seg_local,
                rand_rng=self._rng_for(cell_id, cycle, +1))
            yield from recs
