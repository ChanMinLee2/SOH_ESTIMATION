"""
4_hi_analysis/logics.py

hi_correlation.py(Step 4)의 실행 로직 — HI 추출/상관분석/결과 저장. 값은
constants.py, 그림은 plot.py, 진입점(main)은 hi_correlation.py가 갖는다
(2026-09-29 4파일 분리).
"""

import json
import multiprocessing as mp
import pickle
import queue as _queue_mod
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm

import constants as C
from common.scenario import get_segmenter
from common.scenario.q_frac_ref import calib_path_tag, n2_path_tag, offset_path_tag
# 세그먼트 HI 계산 로직의 단일 소스는 tools/hi_compute.py(2026-09-29 tools/로 이동
# — hi_correlation.py 전용 오케스트레이션이 아니라 여러 tools/ 스크립트가 함께
# 참조하는 독립 계산 레지스트리라서 logics.py 소유로 두지 않는다) — stat/diff/
# lfp/morph 66개 HI 전부 여기서 가져온다(같은 이름의 로컬 정의를 두지 않는다,
# 2026-09-22 정리).
from tools.hi_compute import (
    _seg_stat,
    _seg_diff,
    _seg_lfp,
    _seg_morph_curves,
)


def _ds_dir(name: str) -> Path:
    """데이터셋 이름 -> 원본 clean pkl 디렉터리."""
    return {"MIT": C.MIT_DIR, "HUST": C.HUST_DIR, "TJU": C.TJU_DIR, "CALCE": C.CALCE_DIR}[name]


def build_hi_groups(seg_names: list) -> tuple:
    """HI_GROUPS/ALL_HI_KEYS/HI_LABELS를 임의 세그먼트 이름 목록으로 빌드.

    q_frac_ref 표준 6-시나리오가 아닌 세그먼트 이름 집합이 오더라도 동작한다.
    hi_correlation.py::main()이 실행마다 이 함수로 새로 빌드한 값을 지역 변수로
    받아 plot.py에 명시적으로 넘긴다 — constants.py의 초기값(정적으로 한 번
    빌드해둔 것)을 직접 쓰지 않는다.
    """
    labels: dict = {}
    groups: dict = {}
    for seg in seg_names:
        for k in C.STAT_KEYS:
            labels[f"stat_{k}_{seg}"] = C._STAT_LABELS[k]
        for k in C.DIFF_KEYS:
            labels[f"diff_{k}_{seg}"] = C._DIFF_LABELS[k]
        for k in C.LFP_KEYS:
            labels[f"lfp_{k}_{seg}"] = C._LFP_LABELS[k]
        for k in C.MORPH_KEYS:
            labels[f"morph_{k}_{seg}"] = C._MORPH_LABELS[k]
        groups[f"{seg} — Stat"]  = [f"stat_{k}_{seg}"  for k in C.STAT_KEYS]
        groups[f"{seg} — Diff"]  = [f"diff_{k}_{seg}"  for k in C.DIFF_KEYS]
        groups[f"{seg} — LFP"]   = [f"lfp_{k}_{seg}"   for k in C.LFP_KEYS]
        groups[f"{seg} — Morph"] = [f"morph_{k}_{seg}" for k in C.MORPH_KEYS]
    return groups, list(labels.keys()), labels


# 구 이름(_build_hi_groups) — tools/profile_hi_timing.py, tools/seg_corr_analysis.py가
# `from hi_correlation import _build_hi_groups`로 이 함수를 참조하므로 이름을 유지한다.
_build_hi_groups = build_hi_groups


# ─────────────────────────────────────────────────────────────────────────────
# 카테고리 D: 형태학적 거리 헬퍼 (top-level — multiprocessing 호환)
# ─────────────────────────────────────────────────────────────────────────────

def _dtw_batch(queries: np.ndarray, bol: np.ndarray) -> np.ndarray:
    """N개 쿼리 곡선을 단일 참조 곡선에 대해 배치 DTW 계산.

    queries: (N, n)  bol: (n,)  → (N,) 정규화 DTW 거리
    tools/hi_compute.py::_dtw_distance와 동일한 banded DP이지만 N 차원을 numpy
    배열 연산으로 처리. Python 루프는 n(=50) 행에 대해서만 돌므로 호출 오버헤드가
    N배 절감됨.

    입력을 미리 float32로 캐스팅해 뺄셈 단계에서 float64 임시 배열이 안 생기게 하고,
    N을 C._DTW_CHUNK 단위로 나눠 처리해 (N,n,n) 배열의 피크 메모리가 N(=그 셀·시나리오의
    곡선 인스턴스 총합, 장수명 셀 × n_samples면 수천 단위까지 커질 수 있음)에 비례해
    무한정 커지지 않게 한다(2026-08-17, 실제 HUST 장수명 셀에서 메모리 부족 실측).
    """
    N, n = queries.shape
    band = C._DTW_BAND
    queries = queries.astype(np.float32, copy=False)
    bol = bol.astype(np.float32, copy=False)

    out = np.empty(N, dtype=np.float64)
    chunk = max(1, min(N, C._DTW_CHUNK))
    for start in range(0, N, chunk):
        end = start + chunk
        q = queries[start:end]
        m = q.shape[0]
        d = np.abs(q[:, :, None] - bol[None, None, :])  # (m,n,n), 이미 float32
        dtw = np.full((m, n, n), np.inf, dtype=np.float32)
        dtw[:, 0, 0] = d[:, 0, 0]
        for j in range(1, min(band + 1, n)):
            dtw[:, 0, j] = dtw[:, 0, j - 1] + d[:, 0, j]
        for i in range(1, min(band + 1, n)):
            dtw[:, i, 0] = dtw[:, i - 1, 0] + d[:, i, 0]
        for i in range(1, n):
            j_lo = max(1, i - band)
            j_hi = min(n, i + band + 1)
            for j in range(j_lo, j_hi):
                np.minimum(dtw[:, i - 1, j], dtw[:, i, j - 1], out=dtw[:, i, j])
                np.minimum(dtw[:, i, j], dtw[:, i - 1, j - 1], out=dtw[:, i, j])
                dtw[:, i, j] += d[:, i, j]
        out[start:end] = dtw[:, n - 1, n - 1] / n
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HI 추출 (top-level — multiprocessing 호환)
# ─────────────────────────────────────────────────────────────────────────────

def _add_phase(df: pd.DataFrame) -> pd.DataFrame:
    """_4_data_hi/clean 스키마(phase 컬럼 없음)에 phase 컬럼을 current_A 부호로 재구성."""
    df = df.copy()
    cur = df["current_A"]
    df["phase"] = "rest"
    df.loc[cur > C._PHASE_POS, "phase"] = "charge"
    df.loc[cur < C._PHASE_NEG, "phase"] = "discharge"
    return df


def _strip_seg_suffix(d: dict, seg: str) -> dict:
    """{"stat_v_mean_cw_chg_hi": v, ...} -> {"stat_v_mean_cw": v, ...}.

    호출부가 _seg_stat/_seg_diff/_seg_lfp(..., seg)로 직접 만든 접미사만 제거하므로
    (seg 문자열 자체에 언더스코어가 있어도) 항상 정확히 그 seg만 떼어낸다.
    """
    suf = f"_{seg}"
    n = len(suf)
    return {(k[:-n] if k.endswith(suf) else k): v for k, v in d.items()}


def _build_scen_lookup(spec_names: list) -> dict:
    """세그먼트 이름 → scen 코드(방향 부호 있는 정수). C._SEG_SCEN(qfrac류 표준
    6-시나리오)에 전부 있으면 그대로, 아니면 이름 접두사(chg/dis)+등장순서로 계산."""
    if all(s in C._SEG_SCEN for s in spec_names):
        return {s: C._SEG_SCEN[s][0] for s in spec_names}
    _chg_names = [s for s in spec_names if s.startswith("chg")]
    _dis_names = [s for s in spec_names if s not in _chg_names]
    return {
        s: ((_chg_names.index(s) + 1) if s in _chg_names else -(_dis_names.index(s) + 1))
        for s in spec_names
    }


def _load_cell_pkl(path: Path):
    """셀 pkl 로드 + 최소 검증. 반환: (df_all, dataset, cell_id) 또는 실패 시 None."""
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except Exception:
        return None

    meta   = raw.get("meta", {})
    df_all = raw.get("cycles")
    if df_all is None or not isinstance(df_all, pd.DataFrame):
        return None

    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    dataset = meta.get("dataset", "")
    cell_id = meta.get("cell_id", path.stem)
    return df_all, dataset, cell_id


def _extract_segment_rows(rec_iter, spec_names: list, dataset: str, cell_id: str,
                           cyc, cap: float, scen_lookup: dict,
                           curve_buf: dict) -> list:
    """세그먼터가 내놓는 SegmentRecord들을 HI 행(dict) 리스트로 변환.

    방전/충전 양쪽에서 동일하게 쓴다(예전엔 두 곳에 거의 같은 코드가 복사돼 있었다 —
    2026-09-29 통합, docs/REFACTORING.md 참고). Morph HI는 여기서 값을 확정하지
    않고 곡선만 curve_buf에 버퍼링한다 — 실제 거리 계산은 그 사이클 루프가 전부 끝난
    뒤 _apply_batched_morph_distances()가 세그먼트/곡선타입별로 배치 처리한다
    (BOL 참조 곡선을 알려면 같은 시나리오의 첫 세그먼트가 먼저 나와야 하므로).
    """
    rows = []
    for _rec in rec_iter:
        seg = _rec.meta.get("seg_name") or spec_names[_rec.scenario_id]
        vs = _rec.v; ims = _rec.i; dts = _rec.dt; qcs = _rec.q
        _srow: dict = {
            "dataset": dataset, "cell_id": cell_id, "cycle": int(cyc),
            "capacity_Ah": cap,
            "segment_id": int(_rec.scenario_id),
            "seg_name": seg,
            "scen": scen_lookup.get(seg, 0),
            # assign="none"(no_scen 대조군)이라도 사후 존별 재분리가 가능하도록
            # 원본 존/위치 정보를 그대로 보존한다(docs/260816_RESULTS.md §5-4).
            "zone": _rec.meta.get("zone"),
            "q_frac_lo": _rec.meta.get("q_frac_lo"),
            "q_frac_hi": _rec.meta.get("q_frac_hi"),
        }
        _srow.update(_strip_seg_suffix(_seg_stat(vs, ims, dts, qcs, seg), seg))
        _srow.update(_strip_seg_suffix(_seg_diff(vs, ims, dts, qcs, seg), seg))
        _srow.update(_strip_seg_suffix(_seg_lfp(vs, ims, dts, qcs, seg), seg))
        _mc = _seg_morph_curves(vs, ims, dts)
        for _ct, _arr in zip(("vt", "vq", "ve"), _mc):
            if _arr is not None:
                curve_buf.setdefault(seg, {}).setdefault(_ct, []).append((_srow, _arr))
        rows.append(_srow)
    return rows


def _prepare_charge_arrays(chg_grp: pd.DataFrame, cap: float):
    """충전 phase 원시 배열 준비 + gap 보정 + 완전성 게이트.

    len(chg_grp)<20, 또는 완전 충전 전하량(q_tc)이 0.05Ah 이하거나 등록 용량의
    60% 미만(불완전 충전)이면 None — 이 사이클은 충전 세그먼트 HI를 추출하지
    않는다. 반환값은 (vc, ic, dtc, qcc).
    """
    if len(chg_grp) < 20:
        return None
    tc  = chg_grp["time_s"].values.astype(float)
    vc  = chg_grp["voltage_V"].values.astype(float)
    ic  = np.abs(chg_grp["current_A"].values.astype(float))
    dtc = np.clip(np.diff(tc, prepend=tc[0]), 0, None)

    # chg_gap_seg=True인 행은 preprocess.py 필터4가 그 지점의 dt가 비정상적으로
    # 크다고(CC 전환 갭 등) 판정한 곳이다(2026-08-05부터 행 단위 판정 — 사이클
    # 전체가 아니라 그 행 하나만 플래그된다). 그 큰 dt를 누적적분(qcc)에 그대로
    # 넣으면 그 지점 "이후" 값까지 전부 오염되므로, 이 행의 dt만 정상 구간
    # 중앙값으로 대체한다 — V/I 값 자체는 그대로 쓰므로 정보 손실은 이 한 행의
    # 시간정보로 국한된다(예전엔 chg_gap_seg가 하나라도 있으면 세그먼트 HI 계산
    # 전체를 스킵했다 — MIT batch2 99.94% 사이클이 이렇게 날아갔었다).
    if "chg_gap_seg" in chg_grp.columns:
        _gap_mask = chg_grp["chg_gap_seg"].to_numpy(dtype=bool)
        if _gap_mask.any():
            _dtc_pos = dtc[dtc > 0]
            _dtc_med = float(np.median(_dtc_pos)) if len(_dtc_pos) else 0.0
            dtc = np.where(_gap_mask, _dtc_med, dtc)

    qcc = np.cumsum(ic * dtc) / 3600.0
    q_tc = float(qcc[-1])
    if q_tc <= 0.05 or q_tc < cap * 0.60:
        return None

    return vc, ic, dtc, qcc


def _apply_batched_morph_distances(curve_buf: dict) -> None:
    """곡선 버퍼(세그먼트/곡선타입별) → 배치 DTW/Fréchet 거리를 계산해 각 세그먼트
    행(dict)에 직접 써넣는다(반환값 없음, in-place). 그 시나리오의 첫 유효
    세그먼트를 BOL(fresh-state) 참조로 쓴다.

    각 pair가 "그 세그먼트 행" 자체를 들고 있으므로 결과를 바로 그 행에 대입한다 —
    예전엔 사이클 번호로 cycle_rows[_c]를 다시 찾아 대입해서 같은 사이클의 여러
    세그먼트가 서로 덮어썼다(2026-08-16 이전 버그, docs/260816_RESULTS.md 참고).
    """
    for _seg, _ct_dict in curve_buf.items():
        for _ct, _pairs in _ct_dict.items():
            if not _pairs:
                continue
            _bol_arr = _pairs[0][1]                             # 그 시나리오의 첫 유효 세그먼트 = BOL
            _queries  = np.array([p[1] for p in _pairs])       # (N, n)
            _dtw_vals = _dtw_batch(_queries, _bol_arr)          # (N,)
            _frec_vals = np.max(np.abs(_queries - _bol_arr), axis=1)  # (N,)
            for (_srow_ref, _), _dv, _fv in zip(_pairs, _dtw_vals, _frec_vals):
                _srow_ref[f"morph_{_ct}_dtw"]  = float(_dv)
                _srow_ref[f"morph_{_ct}_frec"] = float(_fv)


def _extract_one_cell(args) -> tuple:
    """반환: (seg_rows, cycle_rows, coverage).

    seg_rows: 세그먼트 인스턴스 1개당 행 1개(native seg 포맷, HI 컬럼 접미사 없음) —
      모델 학습 입력(8_train)이 실제로 읽는 데이터. 한 (사이클,시나리오)에 n_samples개면
      n_samples개 행이 그대로 남는다(2026-08-16 이전엔 row.update() 덮어쓰기로 마지막
      1개만 남았음 — docs/260816_RESULTS.md 참고).
    cycle_rows: 사이클 1개당 행 1개, dataset/cell_id/cycle/capacity_Ah만 포함(2026-09-28
      이전엔 완전 사이클 Global HI(G01~G15)도 여기 같이 있었으나 진단/시각화 전용이라
      계산 자체를 없앴다 — capacity_Ah 컬럼만 여전히 실사용, 아래 주석 참고).
    coverage: random_segment 세그먼터에서만 채워지고, 그 외에는 빈 dict.

    단계별로 헬퍼에 위임하는 얇은 오케스트레이터다(2026-09-29 분리 —
    예전엔 5가지 관심사(인자 해석/pkl 로드/방전 추출/충전 추출/배치 거리계산)가
    231줄짜리 함수 하나에 다 섞여 있었고, 방전·충전 추출 블록도 거의 같은 코드가
    두 번 복사돼 있었다. docs/REFACTORING.md 참고):
      _build_scen_lookup      — 세그먼트 이름 → scen 코드
      _load_cell_pkl          — pkl 로드 + 검증
      _extract_segment_rows   — SegmentRecord → HI 행(방전/충전 공용)
      _prepare_charge_arrays  — 충전 배열 준비 + gap 보정 + 완전성 게이트
      _apply_batched_morph_distances — 루프 후 배치 DTW/Fréchet
    """
    # extract_dataset_cells()의 두 호출부(직렬/ProcessPoolExecutor)만이 실제 호출자다 — 둘 다 항상
    # (path, axis, axis_cfg) 3-tuple이고, 병렬 실행 시에만 진행률 큐가 4번째로 붙는다.
    # axis_cfg는 dict를 그대로 넘긴다(ProcessPoolExecutor도 pickle로 dict를 그대로
    # 옮기므로 JSON 문자열로 왕복 인코딩할 이유가 없다 — 2026-09-29).
    if len(args) == 4:
        pkl_path_str, _axis, _axis_cfg, _progress_q = args
    else:
        pkl_path_str, _axis, _axis_cfg = args
        _progress_q = None

    _segmenter = get_segmenter(_axis, {_axis: _axis_cfg})
    _spec_names = _segmenter.get_spec().scenario_names
    _scen_lookup = _build_scen_lookup(_spec_names)

    _loaded = _load_cell_pkl(Path(pkl_path_str))
    if _loaded is None:
        return [], [], {}
    df_all, dataset, cell_id = _loaded

    seg_rows: list[dict] = []      # 세그먼트 인스턴스별 행 (native seg 포맷)
    cycle_rows: list[dict] = []    # 사이클별 행 (cycle 포맷, dataset/cell_id/cycle/capacity_Ah)
    # {seg_name: {curve_type: [(그 세그먼트 행 dict, arr), ...]}} — 배치 DTW용 곡선 버퍼.
    # 키를 사이클 번호가 아니라 "그 세그먼트 행 자체"로 잡아, 배치 처리 후 바로 그 행에
    # 대입한다(사이클 단위 딕셔너리를 거치지 않으므로 여러 세그먼트가 같은 사이클번호를
    # 공유해도 서로 안 덮어씀).
    _curve_buf: dict[str, dict[str, list]] = {}

    _progress_local = 0
    for cyc, grp in df_all.groupby("cycle"):
        if _progress_q is not None:
            _progress_local += 1
            if _progress_local >= C._PROGRESS_BATCH:
                _progress_q.put(_progress_local)
                _progress_local = 0

        if int(cyc) == 0:
            continue
        dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
        if len(dis) < 30:
            continue

        cap = float(dis["capacity_Ah"].iloc[0])
        if not np.isfinite(cap) or cap < 0.05:
            continue

        v   = dis["voltage_V"].values.astype(float)
        i   = dis["current_A"].values.astype(float)
        t   = dis["time_s"].values.astype(float)
        dt  = np.clip(np.diff(t, prepend=t[0]), 0, None)
        i_mag = np.abs(i)

        q_cum   = np.cumsum(i_mag * dt) / 3600.0
        q_local = float(q_cum[-1]) if len(q_cum) > 0 else 0.0

        # 실제 방전량 < 등록 용량 30% → 불완전 사이클
        if q_local < cap * 0.30:
            continue

        # ── 사이클 행(cycle 포맷) — 완전 사이클 Global HI(G01~G15)는 더 이상 계산하지
        # 않는다(2026-09-28, 진단/시각화 전용이었고 학습 입력도 아니었음 —
        # build_flat_correlation_df 참고). 다만 이 행 자체(dataset/cell_id/cycle/
        # capacity_Ah)는 계속 필요하다 — model_lib/datasets/segment_dataset.py의
        # load_dataset_native_seg()가 이 "cycle" pkl에서 사이클 진짜 총 용량
        # (capacity_Ah)을 읽어 세그먼트별 capacity_Ah(부분 충방전량이라 그 자체로는
        # SOH 타깃이 될 수 없음)를 대체하는 데 실제로 쓰기 때문이다.
        grow: dict = {"dataset": dataset, "cell_id": cell_id,
                      "cycle": int(cyc), "capacity_Ah": cap}

        # ── 방전 세그먼트 HI (segmenter 기반) ──────────────────────────────
        if q_local >= 0.05:
            seg_rows.extend(_extract_segment_rows(
                _segmenter.iter_segments(cell_id, int(cyc), v, i_mag, dt, q_cum),
                _spec_names, dataset, cell_id, cyc, cap, _scen_lookup, _curve_buf,
            ))

        # ── 충전 세그먼트 HI (segmenter 기반) — gap 보정된 배열 + 완전성 게이트는
        # _prepare_charge_arrays가 전담(None이면 이 사이클은 충전 HI 스킵) ──────
        chg_grp = grp[grp["phase"] == "charge"].sort_values("time_s")
        _chg_arrays = _prepare_charge_arrays(chg_grp, cap)
        if _chg_arrays is not None:
            vc_s, ic_s, dtc_s, qcc_s = _chg_arrays
            _empty = np.empty(0, dtype=float)
            seg_rows.extend(_extract_segment_rows(
                _segmenter.iter_segments(
                    cell_id, int(cyc), _empty, _empty, _empty, _empty,
                    vc_s, ic_s, dtc_s, qcc_s,
                ),
                _spec_names, dataset, cell_id, cyc, cap, _scen_lookup, _curve_buf,
            ))

        cycle_rows.append(grow)

    _apply_batched_morph_distances(_curve_buf)

    if _progress_q is not None and _progress_local > 0:
        _progress_q.put(_progress_local)

    return seg_rows, cycle_rows, dict(getattr(_segmenter, "coverage", {}) or {})


# ─────────────────────────────────────────────────────────────────────────────

def _merge_coverage(dst: dict, src: dict) -> None:
    """coverage 딕트 병합: scenario -> [covered, total] 합산."""
    for k, v in (src or {}).items():
        c = dst.setdefault(k, [0, 0])
        c[0] += v[0]; c[1] += v[1]


def extract_dataset_cells(
    pkl_dir: Path,
    n_workers: int = 4,
    axis: str = "qfrac",
    axis_cfg: dict | None = None,
) -> tuple:
    """한 데이터셋 디렉터리(MIT 또는 HUST) 안의 전체 셀에 대해 _extract_one_cell을
    병렬(ProcessPoolExecutor) 또는 직렬로 돌리고 결과를 이어붙여 반환한다

    반환: (df_seg, df_cycle, coverage). coverage는 random_segment 시에만 채워짐(그 외 빈 dict).

    df_seg: 세그먼트 인스턴스별 HI(native seg 포맷, 모델 학습 입력). df_cycle: 사이클별
    dataset/cell_id/cycle/capacity_Ah 행(cycle 포맷, 완전 사이클 HI 없음). 둘 다 이
    디렉터리(MIT 또는 HUST)의 전체 셀을 이어붙인 것.
    """
    # 사이클 수가 많은 셀(파일 크기 큰 순) 먼저 배정 → 워커 간 부하 균형 개선
    files = sorted(pkl_dir.glob("*.pkl"), key=lambda f: f.stat().st_size, reverse=True)
    _axis_cfg = axis_cfg or {}
    all_seg: list = []
    all_cyc: list = []
    coverage: dict = {}
    if n_workers <= 1:
        for f in tqdm(files, desc=pkl_dir.name):
            seg_rows, cyc_rows, cov = _extract_one_cell((str(f), axis, _axis_cfg))
            all_seg.extend(seg_rows); all_cyc.extend(cyc_rows); _merge_coverage(coverage, cov)
    else:
        # 파일 완료 단위 tqdm(pbar)만으로는 큰 파일이 많이 배정된 초반에 진행률이
        # 한참 안 움직이는 것처럼 보임 → Manager 큐로 워커의 사이클 처리량을
        # 실시간으로 받아 별도 진행률 바(총량 미상, 카운트+속도만 표시)를 갱신.
        _mgr = mp.Manager()
        _progress_q = _mgr.Queue()
        _stop_evt = threading.Event()
        _cyc_pbar = tqdm(desc=f"{pkl_dir.name} 사이클 처리량", unit="cyc", position=1, leave=False)

        def _drain_progress():
            while not _stop_evt.is_set():
                try:
                    n = _progress_q.get(timeout=0.2)
                    _cyc_pbar.update(n)
                except _queue_mod.Empty:
                    continue

        _drain_thread = threading.Thread(target=_drain_progress, daemon=True)
        _drain_thread.start()

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_extract_one_cell, (str(f), axis, _axis_cfg, _progress_q)): f
                    for f in files}
            with tqdm(total=len(files), desc=pkl_dir.name, position=0) as pbar:
                for fut in as_completed(futs):
                    seg_rows, cyc_rows, cov = fut.result()
                    all_seg.extend(seg_rows); all_cyc.extend(cyc_rows); _merge_coverage(coverage, cov)
                    pbar.update(1)

        _stop_evt.set()
        _drain_thread.join(timeout=1.0)
        _cyc_pbar.close()
        _mgr.shutdown()
    return (
        pd.DataFrame(all_seg) if all_seg else pd.DataFrame(),
        pd.DataFrame(all_cyc) if all_cyc else pd.DataFrame(),
        coverage,
    )


def build_flat_correlation_df(df_seg: pd.DataFrame, df_cycle: pd.DataFrame) -> pd.DataFrame:
    """(d) 세그먼트 인스턴스 df + 사이클 글로벌 df → Step4 자체 상관분석/플롯 전용 wide df
    — 옛 load_or_extract()의 8). hi_correlation.py가 load_cache()/extract_all_datasets()/
    save_extraction_results() 다음으로 직접 호출한다(2026-10-01 분리 — 이전엔
    load_or_extract() 안에서만 쓰여 비공개(`_` 접두)였다).

    2026-08-16: 모델 학습(8_train)은 이제 df_seg를 그대로 읽으므로(세그먼트당 1행,
    docs/260816_RESULTS.md) 이 함수의 출력은 **학습에 쓰이지 않는다** — compute_correlations/
    plot_correlation/_plot_sample_hi가 기대하는 "사이클당 1행, 시나리오별 _{seg} 접미사
    컬럼" 형태를 맞춰주기 위한 순수 시각화·진단용 재구성이다. 같은 (사이클,시나리오)의
    n_samples개 세그먼트는 여기서 평균만 낸다 — 이 평균이 모델 입력에 영향을 주지 않으므로
    "세그먼트 단위로 독립 학습"이라는 목표와 충돌하지 않는다.
    """
    if df_cycle.empty:
        return pd.DataFrame()

    if df_seg.empty:
        return df_cycle.reset_index(drop=True)

    _hi_cols = [c for c in df_seg.columns if c in C._SEG_HI_BASES]
    agg = (df_seg.groupby(["cell_id", "cycle", "seg_name"])[_hi_cols]
                 .mean()
                 .reset_index())

    wide_parts = []
    for seg_name, g in agg.groupby("seg_name"):
        g = g.drop(columns="seg_name").rename(
            columns={c: f"{c}_{seg_name}" for c in _hi_cols})
        wide_parts.append(g.set_index(["cell_id", "cycle"]))

    if not wide_parts:
        return df_cycle.reset_index(drop=True)

    wide = pd.concat(wide_parts, axis=1).reset_index()
    return df_cycle.merge(wide, on=["cell_id", "cycle"], how="left").reset_index(drop=True)


def _save_sample_csvs(per_ds: dict) -> None:
    """데이터셋별 대표 셀 첫 번째 사이클을 cycle/seg 형식으로 CSV 저장.

    per_ds: {dataset_name: (df_cycle, df_seg)}.

    seg CSV엔 이제 그 사이클의 세그먼트 인스턴스가 (n_samples개면) 여러 행으로 그대로
    남는다 — 예전엔 시나리오당 1행으로 뭉개진 걸 저장했었다.
    """
    sample_dir = C.HI_ROOT / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, (df_cyc, df_seg) in per_ds.items():
        ds_tag = ds_name.lower()
        if df_cyc.empty:
            continue
        first_cell = df_cyc["cell_id"].iloc[0]
        first_cyc  = int(df_cyc[df_cyc["cell_id"] == first_cell]["cycle"].min())

        cyc_row = df_cyc[(df_cyc["cell_id"] == first_cell) & (df_cyc["cycle"] == first_cyc)]
        seg_row = (df_seg[(df_seg["cell_id"] == first_cell) & (df_seg["cycle"] == first_cyc)]
                   if not df_seg.empty else df_seg)

        cyc_row.to_csv(sample_dir / f"{ds_tag}_hi_cycle{first_cyc}.csv", index=False)
        seg_row.to_csv(sample_dir / f"{ds_tag}_hi_seg{first_cyc}.csv",   index=False)

    print(f"  샘플 CSV: {sample_dir}")


def _save_per_cell_hi(
    df_seg: pd.DataFrame,
    df_cycle: pd.DataFrame,
    dataset: str,
    axis: str = "qfrac",
) -> tuple:
    """이미 cycle/seg 형식으로 나뉜 DataFrame을 셀별 pkl로 저장.

    Returns:
        (df_cycle, df_seg)
    """
    cycle_dir = C.HI_ROOT / axis / "cycle" / dataset
    seg_dir   = C.HI_ROOT / axis / "seg"   / dataset
    cycle_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    if not df_cycle.empty:
        for cell_id, grp in df_cycle.groupby("cell_id"):
            grp.reset_index(drop=True).to_pickle(cycle_dir / f"{cell_id}.pkl")
    if not df_seg.empty:
        for cell_id, grp in df_seg.groupby("cell_id"):
            grp.reset_index(drop=True).to_pickle(seg_dir / f"{cell_id}.pkl")

    n = df_cycle["cell_id"].nunique() if not df_cycle.empty else 0
    print(f"  사이클 HI 저장: {cycle_dir}  ({n}개 셀)")
    print(f"  세그먼트 HI 저장: {seg_dir}  ({n}개 셀)")
    return df_cycle, df_seg


def _rand_suffix(axis_cfg: dict) -> str:
    """random_segment=True 면 경로 태그에 붙일 suffix (_random-L{seg_len_pts}), 아니면 빈 문자열.
    train_scr/train_classifier._axis_dir_from_spec 와 동일 규칙."""
    if not axis_cfg.get("random_segment", False):
        return ""
    return f"_random-L{int(axis_cfg.get('seg_len_pts', 20))}"


def _qfw_tag(axis_cfg: dict) -> str:
    """q_frac_wide 파라미터 → 파일/디렉터리 식별 태그."""
    n1 = int(round(axis_cfg.get("n1", 0.4) * 100))
    # n2 조각: 고정 n2면 "n2-20%", q_frac_ref n2 범위 모드면 "n2-10~30%s10"
    # (네 곳 공통 규칙 — common/scenario/q_frac_ref.n2_path_tag 참고)
    n2_frag = n2_path_tag(axis_cfg)
    ns = int(axis_cfg.get("n_samples", 4))
    # 2026-08-10: min_pts 기본값(10)이 아니면 접미사 — 세그먼트 최소 포인트 임계값이
    # 달라지면 완전히 다른 데이터이므로 반드시 다른 경로에 저장(confound 방지, §4.6).
    min_pts = int(axis_cfg.get("min_pts", 10))
    minpts_sfx = f"_minpts{min_pts}" if min_pts != 10 else ""
    # assign="none"(시나리오-only 대조군, docs/260816_RESULTS.md §5 no_scen)이나
    # "mid_nmid"(4시나리오, 게이트 파편화 dose-response 중간점, 2026-09-16 추가)면
    # 반드시 다른 경로에 저장 
    _assign = axis_cfg.get("assign", "position_bin")
    assign_sfx = {"position_bin": "", "none": "_noscen", "mid_nmid": "_midnmid"}.get(_assign, f"_{_assign}")
    # 2026-09-09: tile_scope("zone"|"full")를 assign과 분리(라벨 유무 vs 배치 방식,
    # docs/0909_RESULTS.md) — 명시적으로 준 경우만 접미사(미지정 시 기존 assign 연동
    # 동작과 100% 동일한 경로를 써야 하므로 빈 문자열 유지, 하위호환).
    tile_scope = axis_cfg.get("tile_scope")
    tile_scope_sfx = f"_tile{tile_scope}" if tile_scope is not None else ""
    return f"n1-{n1}%_{n2_frag}_N-{ns}{_rand_suffix(axis_cfg)}{minpts_sfx}{assign_sfx}{tile_scope_sfx}"


def _qfref_tag(axis_cfg: dict) -> str:
    """q_frac_ref 파라미터 → 파일/디렉터리 식별 태그.

    n1/n2/N은 q_frac_wide와 동일 규칙(부모 클래스 파라미터 상속) + ref_lag/noise_amp/
    noise_period를 덧붙여 lag·노이즈 파라미터가 다르면 반드시 다른 경로에 저장되게 한다
    """
    base = _qfw_tag(axis_cfg)
    lag = int(axis_cfg.get("ref_lag", 0))
    noise_pct = int(round(axis_cfg.get("noise_amp", 0.03) * 100))
    mode = str(axis_cfg.get("noise_mode", "ou"))
    period = int(round(axis_cfg.get("noise_period_cycles", 200.0)))
    return (f"{base}_lag-{lag}_noise-{noise_pct}%_{mode}-{period}"
            f"{calib_path_tag(axis_cfg)}{offset_path_tag(axis_cfg)}")


def _save_coverage_stats(path: Path, per_ds_cov: dict, axis: str, axis_cfg: dict) -> None:
    """존 포인트 커버리지/누락 비율을 텍스트로 저장.

    두 모드에서 채워진다 —
      - random_segment=True: 랜덤 배치라 설계상 누락이 생김(그 비율을 재는 게 목적).
      - q_frac_ref n2 범위 모드: 100% 커버리지가 **설계 조건**이므로, 여기 100% 미만이
        찍히면 그건 "존 전체 포인트 수 < min_pts라 세그먼트를 못 만든 사이클"뿐이어야
        한다(common/scenario/q_frac_ref.py 모듈 docstring 참고).
    """
    _n2_range = axis_cfg.get("n2_start") is not None and axis_cfg.get("n2_end") is not None
    order = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
    lines = [
        "=" * 72,
        f"  존 포인트 커버리지 통계   (axis={axis}, cfg={json.dumps(axis_cfg, ensure_ascii=False)})",
        "=" * 72,
        "  누락 비율 = 1 - (어느 세그먼트에든 포함된 존 포인트 수 / 존 전체 포인트 수)",
        ("  (n2 범위 모드: 랜덤 길이 타일링이라 커버율 100%가 조건 — 미달분은 "
         "존 포인트 수 < min_pts인 사이클)"
         if _n2_range else
         "  (창 겹침은 covered 1회 집계; 랜덤 추출로 샘플링되지 못한 부분)"),
        "",
    ]
    for ds, cov in per_ds_cov.items():
        if not cov:
            continue
        lines.append(f"[{ds}]")
        lines.append(f"  {'시나리오':<10}{'covered':>12}{'total':>12}{'커버율':>9}{'누락율':>9}")
        lines.append("  " + "-" * 52)
        tot_c = tot_t = 0
        for name in order:
            if name not in cov:
                continue
            c, t = cov[name]
            tot_c += c; tot_t += t
            covr = 100 * c / t if t else float("nan")
            lines.append(f"  {name:<10}{c:>12,}{t:>12,}{covr:>8.1f}%{100-covr:>8.1f}%")
        if tot_t:
            allcov = 100 * tot_c / tot_t
            lines.append("  " + "-" * 52)
            lines.append(f"  {'전체':<10}{tot_c:>12,}{tot_t:>12,}{allcov:>8.1f}%{100-allcov:>8.1f}%")
        lines.append("")
    lines.append("=" * 72)
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"  누락 비율 저장: {path}")


def resolve_cache_and_axis_dir(
    axis: str, axis_cfg: dict, dataset_group: str, cache_path: Path | None = None,
) -> tuple[Path, str]:
    """(캐시 파일 경로, axis_dir) 계산 — load_cache()도 이 함수를 호출해서
    쓴다(중복 정의 없이 태그 규칙 단일 소스). 실제 추출/로드 없이 경로만 알고
    싶을 때도 쓴다(예: Step4 결과 매니페스트 작성 — hi_correlation.py는
    load_cache()가 이미 반환한 axis_dir을 재사용하므로 이 함수를 두 번 안 부른다).

    dataset_group="lfp"(기본, MIT+HUST): 기존 캐시 경로 그대로 유지(하위호환).
    dataset_group="ncm"(TJU+CALCE) / "all": 캐시/저장 경로에 '_{group}' 접미사를
    붙여 기존 lfp 결과와 절대 안 겹치게 한다(2026-09-05 신규).
    """
    cache_path = cache_path if cache_path is not None else C.CACHE_PATH
    # 정식(v4) 축은 q_frac_ref 하나뿐(2026-09-24 비-정식 축 일괄 삭제) — 파라미터별
    # 고유 경로 태그(_qfref_tag가 내부적으로 _qfw_tag를 재사용, n1/n2/N은 부모 클래스
    # q_frac_wide와 동일 규칙).
    if axis == "q_frac_ref":
        _tag      = _qfref_tag(axis_cfg)
        _cache    = cache_path.parent / f"hi_features_qfref_{_tag}.pkl"
        _axis_dir = f"q_frac_ref/{_tag}"
    else:
        _cache    = cache_path.parent / f"hi_features_{axis}.pkl"
        _axis_dir = axis
    if dataset_group != "lfp":
        _cache    = _cache.with_name(_cache.stem + f"_{dataset_group}.pkl")
        _axis_dir = f"{_axis_dir}_{dataset_group}"
    return _cache, _axis_dir


def load_cache(
    axis: str, axis_cfg: dict, dataset_group: str,
    cache_path: Path | None = None, force: bool = False,
) -> tuple[Path, str, pd.DataFrame | None]:
    """(a) 캐시 경로 계산 + 캐시 hit면 즉시 로드 — 옛 load_or_extract()의 1)+2).

    반환: (cache_path, axis_dir, df). df는 force=True거나 캐시가 없으면 None —
    호출부가 None을 보고 extract_all_datasets() → save_extraction_results() →
    build_flat_correlation_df() 순으로 직접 이어가야 한다는 신호다. axis_dir은
    캐시 hit/miss와 무관하게 항상 반환하므로(이후 단계가 다 필요로 함), 호출부가
    resolve_cache_and_axis_dir()를 따로 또 부를 필요 없다.
    """
    if dataset_group not in C.DATASET_GROUPS:
        raise ValueError(f"알 수 없는 dataset_group: {dataset_group!r} (선택: {list(C.DATASET_GROUPS)})")
    cache, axis_dir = resolve_cache_and_axis_dir(axis, axis_cfg, dataset_group, cache_path)
    if not force and cache.exists():
        print(f"  캐시 로드: {cache}")
        return cache, axis_dir, pd.read_pickle(cache)
    return cache, axis_dir, None


def extract_all_datasets(
    dataset_group: str, n_workers: int = 4,
    axis: str = "qfrac", axis_cfg: dict | None = None,
) -> tuple[dict, dict, dict]:
    """(b) extract_dataset_cells을 데이터셋별로 돌려 전체 셀 HI를 추출(병렬) — 옛
    load_or_extract()의 3). 저장은 안 한다(save_extraction_results()가 담당) —
    "추출"과 "저장"을 분리해 한 데이터셋 끝날 때마다 바로 저장하던 예전 인터리브
    방식을 "전부 추출 → 전부 저장" 두 단계로 나눴다(최종 저장 결과는 동일).

    반환: (seg_per_ds, cyc_per_ds, cov_per_ds) — 전부 {dataset_name: ...} dict.
    """
    axis_cfg = dict(axis_cfg or {})
    ds_names = C.DATASET_GROUPS[dataset_group]
    seg_per_ds: dict = {}
    cyc_per_ds: dict = {}
    cov_per_ds: dict = {}
    for ds_name in ds_names:
        print(f"=== {ds_name} HI 추출 (axis={axis}) ===")
        seg_i, cyc_i, cov_i = extract_dataset_cells(_ds_dir(ds_name), n_workers=n_workers, axis=axis,
                                                     axis_cfg=axis_cfg)
        seg_per_ds[ds_name] = seg_i
        cyc_per_ds[ds_name] = cyc_i
        cov_per_ds[ds_name] = cov_i
    return seg_per_ds, cyc_per_ds, cov_per_ds


def save_extraction_results(
    seg_per_ds: dict, cyc_per_ds: dict, cov_per_ds: dict,
    axis: str, axis_cfg: dict, axis_dir: str, dataset_group: str,
) -> None:
    """(c) extract_all_datasets() 결과를 전부 저장 — 옛 load_or_extract()의 4)~7):
      4) _save_per_cell_hi — 데이터셋별 셀 단위 pkl
      5) _save_sample_csvs — 대표 셀 1개씩 사람이 보기 쉬운 CSV
      6) _segmenter.save_artifacts — 이번에 쓴 ScenarioSpec(축 파라미터) 저장
      7) _save_coverage_stats — random_segment일 때만 존 커버리지 텍스트 저장
    """
    ds_names = C.DATASET_GROUPS[dataset_group]

    # 4) 데이터셋별 셀 단위 pkl 저장
    for ds_name in ds_names:
        _save_per_cell_hi(seg_per_ds[ds_name], cyc_per_ds[ds_name], ds_name, axis=axis_dir)

    # 5) 대표 셀 CSV 저장
    _save_sample_csvs({ds: (cyc_per_ds[ds], seg_per_ds[ds]) for ds in ds_names})

    # 6) ScenarioSpec 저장
    segmenter = get_segmenter(axis, {axis: axis_cfg or {}})
    spec_dir = C.HI_ROOT / axis_dir
    spec_dir.mkdir(parents=True, exist_ok=True)
    segmenter.save_artifacts(spec_dir)
    print(f"  ScenarioSpec 저장: {spec_dir / 'scenario_spec.json'}")

    # 7) random_segment 누락 비율 텍스트 저장(coverage가 있을 때만)
    if any(cov_per_ds.values()):
        _save_coverage_stats(spec_dir / "coverage_stats.txt", cov_per_ds, axis, axis_cfg)


def compute_correlations(df: pd.DataFrame, all_hi_keys: list, datasets: list | None = None) -> pd.DataFrame:
    """Spearman ρ(HI, capacity_Ah) — 데이터셋별(기본 MIT/HUST, ncm 그룹은 TJU/CALCE).

    all_hi_keys는 호출자가 build_hi_groups()로 방금 빌드한 값을 넘긴다(모듈
    전역을 여기서 직접 읽지 않음 — 축에 따라 매번 재빌드되는 값이라 stale
    import를 피하기 위함, plot.py 분리 때와 동일 원칙)."""
    datasets = list(datasets) if datasets is not None else ["MIT", "HUST"]
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")
    result = {}
    for ds in datasets:
        sub  = df[df["dataset"] == ds]
        rhos = {}
        for hi in all_hi_keys:
            if hi not in sub.columns:
                rhos[hi] = np.nan; continue
            valid = sub[[hi, "capacity_Ah"]].dropna()
            rhos[hi] = (
                spearmanr(valid[hi], valid["capacity_Ah"])[0]
                if len(valid) > 30 else np.nan
            )
        result[ds] = rhos
    return pd.DataFrame(result, index=all_hi_keys)


def _print_run_config(axis: str, axis_cfg: dict, workers: int, force: bool) -> None:
    """HI 추출 실행 조건(축·파라미터·경로·랜덤옵션)을 터미널에 요약 출력."""
    try:
        _spec = get_segmenter(axis, {axis: axis_cfg}).get_spec()
        _params = _spec.params or {}
        _n_scen = _spec.n_scenarios
    except Exception:
        _params, _n_scen = dict(axis_cfg), "?"

    # 데이터 저장 경로 태그 (resolve_cache_and_axis_dir 와 동일 규칙)
    if axis == "q_frac_ref":
        _axis_dir = f"q_frac_ref/{_qfref_tag(axis_cfg)}"
    else:
        _axis_dir = axis

    _is_rand = bool(axis_cfg.get("random_segment", False))
    # q_frac_ref n2 범위 모드 — 세그먼트 길이가 고정이 아니라 격자에서 랜덤 추첨되고,
    # 존을 100% 덮도록 타일링된다(common/scenario/q_frac_ref.py 모듈 docstring).
    _n2_range = (axis_cfg.get("n2_start") is not None
                 and axis_cfg.get("n2_end") is not None)
    w = 64
    print("\n" + "=" * w)
    print("  HI 추출 실행 조건")
    print("=" * w)
    print(f"  세그먼트 축      : {axis}   (시나리오 {_n_scen}개)")
    if _params:
        print(f"  축 파라미터      : {json.dumps(_params, ensure_ascii=False)}")
    print(f"  random_segment  : {_is_rand}"
          + (f"  (seg_len_pts={int(axis_cfg.get('seg_len_pts', 20))}, "
             f"seed={int(axis_cfg.get('random_seed', 42))})" if _is_rand else "  (기존 격자 방식)"))
    if _n2_range:
        _step = float(axis_cfg.get("n2_step", 0.1))
        _grid = [round(float(axis_cfg["n2_start"]) + j * _step, 10)
                 for j in range(int(round((float(axis_cfg["n2_end"])
                                           - float(axis_cfg["n2_start"])) / _step)) + 1)]
        print(f"  n2 범위 모드     : True   길이 격자={_grid}  "
              f"seed={int(axis_cfg.get('n2_seed', 20260903))}")
        print(f"                    (존을 랜덤 길이로 타일링 — 커버리지 100% 보장, "
              f"n_samples 무시)")
    print(f"  워커 수          : {workers}")
    print(f"  force 재추출     : {force}")
    print(f"  데이터 저장 경로 : _4_data_hi/{_axis_dir}/{{seg,cycle}}/")
    if _is_rand:
        print(f"  누락비율 저장    : _4_data_hi/{_axis_dir}/coverage_stats.txt")
    print("=" * w + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 결과 폴더(step_4_result/) — 2026-09-29 신규
# ─────────────────────────────────────────────────────────────────────────────

def _series_stats(counts: list) -> dict:
    """정수 리스트 → {mean, median, total, min, max} (빈 리스트면 전부 NaN/0)."""
    if not counts:
        return {"mean": float("nan"), "median": float("nan"), "total": 0,
                "min": float("nan"), "max": float("nan")}
    arr = np.asarray(counts, dtype=float)
    return {
        "mean": float(arr.mean()), "median": float(np.median(arr)),
        "total": int(arr.sum()), "min": float(arr.min()), "max": float(arr.max()),
    }


def compute_extraction_stats(
    axis_dir: str, ds_names: list, cache_path: Path, total_rows: int,
) -> pd.DataFrame:
    """Step4 추출 결과 기초 통계 — 데이터셋별 셀 수/사이클 수/세그먼트 수 분포 +
    원본 대비 게이팅으로 제외된 사이클 수(cycles_removed_by_gate).

    HI_ROOT/axis_dir/{cycle,seg}에 이미 저장된 산출물(_save_per_cell_hi 출력)과
    원본 clean pkl만 읽는다 — 재추출은 하지 않는다. 캐시 히트로 이번 실행이
    추출을 아예 안 했어도(load_cache가 pd.read_pickle로 바로 반환한 경우),
    같은 axis_cfg로 예전에 한 번 추출됐을 때 저장된 per-cell pkl이 이미 있으므로
    이 함수는 항상 동작한다.
    """
    rows = []
    for ds in ds_names:
        cyc_dir = C.HI_ROOT / axis_dir / "cycle" / ds
        seg_dir = C.HI_ROOT / axis_dir / "seg" / ds
        raw_dir = _ds_dir(ds)
        cyc_files = sorted(cyc_dir.glob("*.pkl")) if cyc_dir.exists() else []

        cyc_counts: list = []
        seg_counts: list = []
        raw_cyc_counts: list = []
        for f in cyc_files:
            df_c = pd.read_pickle(f)
            cyc_counts.append(len(df_c))
            seg_f = seg_dir / f.name
            seg_counts.append(len(pd.read_pickle(seg_f)) if seg_f.exists() else 0)
            raw_f = raw_dir / f.name
            if raw_f.exists():
                try:
                    with open(raw_f, "rb") as fh:
                        raw = pickle.load(fh)
                    raw_cyc_counts.append(int(raw["cycles"]["cycle"].nunique()))
                except Exception:
                    pass

        cyc_stat = _series_stats(cyc_counts)
        seg_stat = _series_stats(seg_counts)
        removed = (sum(raw_cyc_counts) - cyc_stat["total"]) if raw_cyc_counts else float("nan")

        rows.append({
            "dataset": ds,
            "cache_path": str(cache_path),
            "total_rows_in_cache": total_rows,
            "n_cells": len(cyc_files),
            "cycle_count_mean": cyc_stat["mean"], "cycle_count_median": cyc_stat["median"],
            "cycle_count_total": cyc_stat["total"], "cycle_count_min": cyc_stat["min"],
            "cycle_count_max": cyc_stat["max"],
            "segment_count_mean": seg_stat["mean"], "segment_count_median": seg_stat["median"],
            "segment_count_total": seg_stat["total"], "segment_count_min": seg_stat["min"],
            "segment_count_max": seg_stat["max"],
            # 원본(clean) 사이클 수 - 추출된 사이클 수 = 완전성/최소길이 게이트로
            # 제외된 사이클 수(_extract_one_cell의 len(dis)<30 / cap 이상 / q_local
            # 게이트, docs/REFACTORING.md 참고). raw pkl을 못 찾으면 NaN.
            "cycles_removed_by_gate": removed,
        })
    return pd.DataFrame(rows)


def save_step4_result(
    step_dir: Path,
    corr: pd.DataFrame,
    stats_df: pd.DataFrame,
    manifest: dict,
) -> None:
    """Step4 산출물을 실험 결과 폴더(step_4_result/)에 저장 — correlation.csv/
    extraction_stats.csv/manifest.json. 큰 데이터(df, 셀별 seg/cycle pkl)는
    이미 axis-config로 키가 매겨진 전역 캐시(_4_data_hi/, PKL_CACHE_ROOT)에
    있으므로 여기 다시 복제하지 않는다 — manifest.json에 그 경로만 적어 둔다.
    plot.py가 이 폴더 경로 하나만 받아 다시 그릴 때 이 매니페스트로 df를 찾는다.
    """
    step_dir.mkdir(parents=True, exist_ok=True)
    corr.to_csv(step_dir / "correlation.csv", encoding="utf-8-sig")
    stats_df.to_csv(step_dir / "extraction_stats.csv", index=False, encoding="utf-8-sig")
    (step_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Step4 결과 저장: {step_dir}")
