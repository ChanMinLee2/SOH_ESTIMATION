"""
common/scenario/vwindow.py — 축 2: 전압 윈도우 세그멘터 (고정 전압 경계 + CC/CV 분리).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
설계 근거 (리뷰어용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[경계 결정 방식] 화학계 물리 기반 고정 전압 (LFP 기본값)
  이전 버전은 BOL(첫 사이클) equal-Ah 피팅으로 전압 경계를 계산했으나,
  다단계 CC 프로토콜(예: MIT 6C→1C→CV)에서 V-Q 곡선이 비단조적이 되어
  피팅 결과가 프로토콜 구조에 강하게 의존하는 문제가 있었다.
  고정 전압 경계는 LFP 전기화학적 상태 전이(플래토 진입/탈출)를 기준으로
  설정되므로 충전 프로토콜과 무관하게 동일한 전기화학 구간을 일관되게 추출한다.

  노화 사이클에서 동일 전압 구간 내 누적 Ah가 줄어드는 현상 자체가 SOH 신호가
  되므로, 고정 전압 경계는 피팅 기반보다 오히려 SOH 민감도가 명확하다
  (Dubarry et al., 2012; Hu et al., 2020).

[방전 경계 근거 — LFP 3-window]
  win0 [2.50–3.10V]: 플래토 종료 + 급강하 구간. 용량 페이드의 끝단 신호.
  win1 [3.10–3.35V]: 하부 플래토 중심. LFP 2상 공존 영역.
  win2 [3.35–3.65V]: 상부 플래토 + 숄더. 노화 초기 특징(숄더 붕괴 등) 포착.

[충전 CC 경계 근거 — LFP 3-window]
  win0 [2.80–3.38V]: 사전-플래토 급경사 구간. 다단계 CC 전류 전환 포함.
  win1 [3.38–3.47V]: 핵심 플래토. 총 용량의 ~70%가 이 구간에 집중.
  win2 [3.47–3.60V]: 플래토 탈출 + CV 진입 직전 구간.

[충전 CV] 독립 시나리오 chg_cv
  CC→CV 전환 이후 구간은 별도 시나리오로 분리한다.
  근거:
  1. 물리 체계 차이: CV 구간은 과전위가 고정되고 전류가 감쇠하는 kinetic 체제이며,
     CC 구간의 thermodynamic 체제(삽입 반응, 확산)와 본질적으로 다르다.
  2. SOH 지표: 노화 셀은 내부저항 증가로 CV 구간이 길어지고 Q_CV/Q_total 이
     증가한다. 이 현상은 Ma et al. (2018), Plett (2015) 등에서 검증된 SOH 지표이다.
  3. CC와 혼합하면 두 체제의 HI가 뒤섞여 모델이 혼동하는 것을 방지한다.

[충전 CV] 독립 시나리오 chg_cv
  CC→CV 전환 이후 구간은 별도 시나리오로 분리한다.
  근거:
  1. 물리 체계 차이: CV 구간은 과전위가 고정되고 전류가 감쇠하는 kinetic 체제이며,
     CC 구간의 thermodynamic 체제(삽입 반응, 확산)와 본질적으로 다르다.
  2. SOH 지표: 노화 셀은 내부저항 증가로 CV 구간이 길어지고 Q_CV/Q_total 이
     증가한다. 이 현상은 Ma et al. (2018), Plett (2015) 등에서 검증된 SOH 지표이다.
  3. CC와 혼합하면 두 체제의 HI가 뒤섞여 모델이 혼동하는 것을 방지한다.

CC→CV 전환 감지:
  V ≥ cv_v_thresh (기본 3.60 V) 이고 I < cv_cc_frac × I_max (기본 0.80)를
  동시에 만족하는 첫 번째 샘플 인덱스. HUST/MIT 데이터셋의 표준 CC-CV 프로토콜에서
  오감지 없이 동작한다. CV 구간이 min_pts 미만이면 chg_cv 시나리오를 건너뛴다.

시나리오 구조 (n_windows=3 기준, 총 7개):
  인덱스  이름       방향   물리 의미
  ─────────────────────────────────────────
  0       chg_win0   +1     충전 CC 저전압 구간
  1       chg_win1   +1     충전 CC 중전압 구간
  2       chg_win2   +1     충전 CC 고전압 구간
  3       chg_cv     +1     충전 CV 구간 (정전압 테이퍼)
  4       dis_win0   -1     방전 저전압 구간
  5       dis_win1   -1     방전 중전압 구간
  6       dis_win2   -1     방전 고전압 구간
  ─────────────────────────────────────────
  n_scenarios = 2·n_windows + 1
  n_classes   = n_windows + 1  (CC 클래스 0..n-1, CV 클래스 n)
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter

_DEFAULT_DIS_V   = (2.0, 3.65)
_DEFAULT_CHG_V   = (2.8, 3.65)
_DEFAULT_N_WIN   = 3
_DEFAULT_CV_V    = 3.60   # CC→CV 전환 전압 임계값 [V]
_DEFAULT_CV_FRAC = 0.80   # CC→CV: I < cv_cc_frac × I_max

# LFP 화학계 기본 전압 경계 (n_windows=3)
# 플래토 진입(~3.38V)·탈출(~3.47V chg / ~3.35V dis) 기준으로 설정
_LFP_CHG_EDGES_3: list[float] = [2.80, 3.38, 3.47, 3.60]
_LFP_DIS_EDGES_3: list[float] = [2.50, 3.10, 3.35, 3.65]


class VWindowSegmenter(Segmenter):
    """전압 윈도우 기반 세그멘터 (고정 전압 경계 + 충전 CC/CV 분리).

    일반 사용: VWindowSegmenter.from_lfp()
    커스텀 경계: VWindowSegmenter(dis_edges=[...], chg_edges=[...])
    데이터 기반 피팅(구버전): VWindowSegmenter(); .fit(bol_data)  — 다단계 CC 주의.
    """

    name = "vwindow"

    def __init__(
        self,
        n_windows: int = _DEFAULT_N_WIN,
        dis_v_range: tuple[float, float] | None = None,
        chg_v_range: tuple[float, float] | None = None,
        dis_edges: list[float] | None = None,
        chg_edges: list[float] | None = None,
        cv_v_thresh: float = _DEFAULT_CV_V,
        cv_cc_frac: float = _DEFAULT_CV_FRAC,
        auto_range: bool = True,
        min_pts: int = 10,
    ):
        self.n_windows   = n_windows
        self.dis_v_lo, self.dis_v_hi = dis_v_range or _DEFAULT_DIS_V
        self.chg_v_lo, self.chg_v_hi = chg_v_range or _DEFAULT_CHG_V
        self.cv_v_thresh = cv_v_thresh
        self.cv_cc_frac  = cv_cc_frac
        self.auto_range  = auto_range
        self.min_pts     = min_pts

        if dis_edges is not None and len(dis_edges) == n_windows + 1:
            self._dis_edges: list[float] | None = [float(v) for v in dis_edges]
        else:
            self._dis_edges = None

        if chg_edges is not None and len(chg_edges) == n_windows + 1:
            self._chg_edges: list[float] | None = [float(v) for v in chg_edges]
        else:
            self._chg_edges = None

    # ------------------------------------------------------------------
    @classmethod
    def from_lfp(
        cls,
        n_windows: int = _DEFAULT_N_WIN,
        cv_v_thresh: float = _DEFAULT_CV_V,
        cv_cc_frac: float = _DEFAULT_CV_FRAC,
        min_pts: int = 10,
    ) -> "VWindowSegmenter":
        """LFP 화학계 기본 전압 경계로 초기화된 세그멘터 반환.

        n_windows=3 전용. 경계는 LFP 플래토 전이 전압 기반으로 고정되어
        충전 프로토콜(1-step/다단계 CC 등)과 무관하게 동작한다.
        """
        if n_windows != 3:
            raise ValueError(
                f"LFP 사전 설정은 n_windows=3 전용입니다. (요청: {n_windows})"
            )
        return cls(
            n_windows=n_windows,
            dis_edges=_LFP_DIS_EDGES_3,
            chg_edges=_LFP_CHG_EDGES_3,
            cv_v_thresh=cv_v_thresh,
            cv_cc_frac=cv_cc_frac,
            min_pts=min_pts,
        )

    # ------------------------------------------------------------------
    # fit: BOL 기반 equal-Ah 전압 경계 계산 (레거시)
    #      다단계 CC 프로토콜(예: MIT 6C→1C)에서는 V-Q 비단조성으로 경계가
    #      왜곡될 수 있다. 단일-CC 데이터셋에서만 사용 권장.
    # ------------------------------------------------------------------
    def fit(self, first_cycle_data: list[dict]) -> None:
        """BOL 데이터에서 equal-Ah 전압 경계를 계산한다 (레거시).

        다단계 CC 프로토콜(예: 6C→1C→CV)에서는 V-Q 비단조성으로 경계가
        왜곡될 수 있다. LFP 데이터셋에는 from_lfp()를 권장한다.

        Args:
            first_cycle_data: [
                {
                  "cell_id": str,
                  "dis_v":   np.ndarray,
                  "dis_q":   np.ndarray,
                  "chg_v":   np.ndarray,  (없으면 키 없음)
                  "chg_q":   np.ndarray,
                  "chg_i":   np.ndarray,  (CC→CV 감지용)
                }, ...
            ]
        """
        import warnings
        warnings.warn(
            "VWindowSegmenter.fit()은 단일-CC 프로토콜에서만 안전합니다. "
            "LFP 다단계 CC 데이터셋에는 VWindowSegmenter.from_lfp()를 사용하세요.",
            UserWarning,
            stacklevel=2,
        )
        quantiles = np.linspace(0, 1, self.n_windows + 1)

        dis_cell_edges, chg_cell_edges = [], []
        for cell in first_cycle_data:
            # 방전: 전체 구간 사용
            if "dis_v" in cell and "dis_q" in cell:
                e = _q_to_v_edges(cell["dis_q"], cell["dis_v"], quantiles)
                if e is not None:
                    dis_cell_edges.append(e)

            # 충전: CC 구간만 사용
            if "chg_v" in cell and "chg_q" in cell:
                chg_i = cell.get("chg_i")
                if chg_i is not None:
                    cv_s = _detect_cv_start(
                        cell["chg_v"], chg_i, self.cv_v_thresh, self.cv_cc_frac
                    )
                else:
                    cv_s = len(cell["chg_v"])   # fallback: i 없으면 전체

                cc_q = cell["chg_q"][:cv_s]
                cc_v = cell["chg_v"][:cv_s]
                if len(cc_q) >= 5:
                    e = _q_to_v_edges(cc_q, cc_v, quantiles)
                    if e is not None:
                        chg_cell_edges.append(e)

        if dis_cell_edges:
            med = np.median(np.stack(dis_cell_edges), axis=0)
            self._dis_edges = sorted(float(v) for v in med)
        if chg_cell_edges:
            med = np.median(np.stack(chg_cell_edges), axis=0)
            self._chg_edges = sorted(float(v) for v in med)

    # ------------------------------------------------------------------
    def _get_dis_edges(self) -> list[float]:
        if self._dis_edges is None:
            self._dis_edges = list(
                np.linspace(self.dis_v_lo, self.dis_v_hi, self.n_windows + 1))
        return self._dis_edges

    def _get_chg_edges(self) -> list[float]:
        if self._chg_edges is None:
            self._chg_edges = list(
                np.linspace(self.chg_v_lo, self.chg_v_hi, self.n_windows + 1))
        return self._chg_edges

    # ------------------------------------------------------------------
    def get_spec(self) -> ScenarioSpec:
        n = self.n_windows
        # chg_win0..n-1, chg_cv, dis_win0..n-1
        scenario_names = (
            [f"chg_win{k}" for k in range(n)]
            + ["chg_cv"]
            + [f"dis_win{k}" for k in range(n)]
        )
        # routing[dir][class] = scenario_id
        # charge dir (0): CC class 0..n-1 → scen 0..n-1, CV class n → scen n
        # dis    dir (1): class 0..n-1 → scen n+1..2n
        routing = [
            list(range(n)) + [n],           # charge: n+1 classes
            list(range(n + 1, 2 * n + 1)),  # discharge: n classes
        ]
        # class_names: CC windows + CV
        class_names = [f"win{k}" for k in range(n)] + ["cv"]

        params: dict = {
            "n_windows":   self.n_windows,
            "dis_v_range": [self.dis_v_lo, self.dis_v_hi],
            "chg_v_range": [self.chg_v_lo, self.chg_v_hi],
            "cv_v_thresh": self.cv_v_thresh,
            "cv_cc_frac":  self.cv_cc_frac,
            "auto_range":  self.auto_range,
        }
        if self._dis_edges is not None:
            params["dis_edges"] = self._dis_edges
        if self._chg_edges is not None:
            params["chg_edges"] = self._chg_edges

        return ScenarioSpec(
            axis="vwindow",
            n_scenarios=2 * n + 1,
            scenario_names=scenario_names,
            n_classes=n + 1,
            class_names=class_names,
            routing=routing,
            classifier_default="rule",
            params=params,
        )

    # ------------------------------------------------------------------
    def _split_by_voltage(
        self,
        v: np.ndarray,
        i: np.ndarray,
        dt: np.ndarray,
        q: np.ndarray,
        edges: list[float],
        direction: int,
        cell_id: str,
        cycle: int,
        seg_local_start: int,
        scen_offset: int,        # 시나리오 ID 오프셋 (dir_idx 대신 명시)
    ) -> tuple[list[SegmentRecord], int]:
        records   = []
        seg_local = seg_local_start

        for k in range(self.n_windows):
            v_lo, v_hi = edges[k], edges[k + 1]
            m = (v >= v_lo) & (v < v_hi)
            if k == self.n_windows - 1:
                m = (v >= v_lo) & (v <= v_hi)   # 마지막 구간: 상한 포함
            if m.sum() < self.min_pts:
                continue
            records.append(SegmentRecord(
                cell_id=cell_id, cycle=cycle, seg_local_id=seg_local,
                scenario_id=scen_offset + k,
                latent_class=k,
                direction=direction,
                v=v[m], i=i[m], dt=dt[m], q=q[m],
                meta={"v_lo": v_lo, "v_hi": v_hi, "win": k},
            ))
            seg_local += 1
        return records, seg_local

    def iter_segments(
        self,
        cell_id, cycle,
        dis_v, dis_i, dis_dt, dis_q,
        chg_v=None, chg_i=None, chg_dt=None, chg_q=None,
    ) -> Iterator[SegmentRecord]:
        n = self.n_windows
        seg_local = 0

        # ── 방전: equal-Ah 전압 윈도우 (scen 오프셋 = n+1, chg_cv 다음) ──
        if len(dis_v) >= self.min_pts:
            recs, seg_local = self._split_by_voltage(
                dis_v, dis_i, dis_dt, dis_q,
                self._get_dis_edges(), -1, cell_id, cycle,
                seg_local, scen_offset=n + 1,
            )
            yield from recs

        # ── 충전: CC 구간 equal-Ah 윈도우 + CV 독립 시나리오 ──────────────
        if chg_v is not None and len(chg_v) >= self.min_pts:
            cv_start = _detect_cv_start(
                chg_v, chg_i, self.cv_v_thresh, self.cv_cc_frac
            )

            # CC 구간 → equal-Ah 전압 윈도우 (scen 오프셋 = 0)
            if cv_start >= self.min_pts:
                recs, seg_local = self._split_by_voltage(
                    chg_v[:cv_start], chg_i[:cv_start],
                    chg_dt[:cv_start], chg_q[:cv_start],
                    self._get_chg_edges(), +1, cell_id, cycle,
                    seg_local, scen_offset=0,
                )
                yield from recs

            # CV 구간 → chg_cv 시나리오 (scen_id = n, latent_class = n)
            cv_len = len(chg_v) - cv_start
            if cv_len >= self.min_pts:
                yield SegmentRecord(
                    cell_id=cell_id, cycle=cycle, seg_local_id=seg_local,
                    scenario_id=n,
                    latent_class=n,
                    direction=+1,
                    v=chg_v[cv_start:],
                    i=chg_i[cv_start:],
                    dt=chg_dt[cv_start:],
                    q=chg_q[cv_start:],
                    meta={"seg_name": "chg_cv", "cv_start_idx": int(cv_start)},
                )
                seg_local += 1


# ------------------------------------------------------------------
# 모듈 레벨 헬퍼
# ------------------------------------------------------------------

def _detect_cv_start(
    v: np.ndarray,
    i: np.ndarray,
    v_thresh: float = _DEFAULT_CV_V,
    cc_frac: float = _DEFAULT_CV_FRAC,
) -> int:
    """CC→CV 전환 시작 인덱스 반환.

    V ≥ v_thresh 이고 I < cc_frac × I_max 를 동시에 만족하는 첫 샘플.
    조건을 만족하는 샘플이 없으면 len(v) 반환 (CV 구간 없음).

    근거: LFP 충전 CV 구간은 (1) 전압이 컷오프에 근접하고 (2) 충전기가
    정전압으로 전환해 전류가 감소하기 시작할 때 진입한다. cc_frac=0.80은
    전류 노이즈에 둔감하면서도 CC→CV 전환을 놓치지 않는 실용적 임계값이다.
    """
    if i is None or len(i) == 0:
        return len(v)
    i_max = float(np.max(i))
    if i_max < 1e-6:
        return len(v)
    cv_mask = (v >= v_thresh) & (i < cc_frac * i_max)
    if not cv_mask.any():
        return len(v)
    return int(np.argmax(cv_mask))


def _q_to_v_edges(
    q: np.ndarray,
    v: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray | None:
    """누적 Ah(q)와 전압(v)에서 분위수 → 전압 경계 반환.

    q 단조증가 가정. 최소 5 포인트 미만이면 None 반환.
    """
    if len(q) < 5 or len(q) != len(v):
        return None
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    q_total = q[-1] - q[0]
    if q_total <= 0:
        return None
    q_norm = np.clip((q - q[0]) / q_total, 0.0, 1.0)
    return np.interp(quantiles, q_norm, v)
