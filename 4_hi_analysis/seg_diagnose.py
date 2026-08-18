"""
seg_diagnose.py

시나리오 분리 축 진단 도구.

  1. 세그먼트 통계 (지정 데이터셋 전체 스캔)
       - 시나리오별 세그먼트 갯수
       - 세그먼트 당 행 갯수 평균
       - 세그먼트 당 시간 길이 평균 (초)

  2. 사이클 시각화 --mode segment (기본)
       [상단] V vs time_s  (충전+방전 연속, 세그먼트 색상 밴드)
       [좌하] V vs Q       (방전 세그먼트 색상 오버레이)
       [우하] V vs Q       (충전 세그먼트 색상 오버레이)

  3. IC 커브 시각화  --mode ic
       [상단좌] 방전 V-Q  멀티사이클 + 창 가로밴드
       [상단우] 충전 V-Q  멀티사이클 + 창 가로밴드
       [하단좌] 방전 dQ/dV vs V  멀티사이클 + 창 세로밴드
       [하단우] 충전 dQ/dV vs V  멀티사이클 + 창 세로밴드
       → vwindow 외 축도 IC 커브만 표시 (경계선 없음)

  --mode all : segment + ic 둘 다 생성

  4. 조건 비교 시각화  --mode compare
       동일 셀·동일 사이클에 대해 서로 다른 축/파라미터 조건(q_frac_wide n1/n2/N,
       random_segment 유무, vqslope mode/N 등) 여러 개를 한 장에 세로로 쌓아
       "같은 데이터가 조건별로 어떻게 잘리는지"를 나란히 비교한다.
       각 행 = plot_cycle_segments의 [상단] V-t 밴드 패널과 동일한 그림.
       조건 목록은 JSON 설정 파일(--compare-config, 기본: compare_conditions.json)로 지정
       — PowerShell에서 여러 조건을 CLI로 전달할 때 생기는 따옴표 깨짐 문제를 회피.
       조건별 segmenter는 그 자리에서 새로 만들어 실행하므로, 해당 조건의 데이터가
       아직 _4_data_hi/{axis}/{tag}/ 에 추출되어 있지 않아도 즉시 비교할 수 있다.

사용 예시:
  python 4_hi_analysis/seg_diagnose.py                        # 자동: 모든 hi_features*.pkl 축
  python 4_hi_analysis/seg_diagnose.py --seg-axis protocol    # 특정 축 지정
  python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --dataset HUST
  python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --mode ic
  python 4_hi_analysis/seg_diagnose.py --seg-axis vwindow --mode ic --n-cycles 8
  python 4_hi_analysis/seg_diagnose.py --seg-axis qfrac --cell b1c0 --cycle 10
  python 4_hi_analysis/seg_diagnose.py --seg-axis protocol --no-plot
  python 4_hi_analysis/seg_diagnose.py --seg-axis cluster --no-stats
  python 4_hi_analysis/seg_diagnose.py --mode compare --dataset HUST --cell HUST_1-8 --cycle 1145
  python 4_hi_analysis/seg_diagnose.py --mode compare --dataset HUST --cell HUST_1-8 --cycle 1145 \
      --compare-config 4_hi_analysis/compare_conditions.json
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP_DIR     = Path(__file__).resolve().parent
# 2026-08-08: pkl 데이터(_4_data_hi 입력, 4_hi_analysis 캐시/outputs pkl)만 D로 이동 —
# STEP_DIR은 다른 곳(compare_conditions.json 등 실제 코드 자산)에도 쓰이므로 그대로 두고
# data_directories.py의 공유 상수를 쓴다(hi_correlation.py와 동일 패턴).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_directories import DATA_4_HI_ROOT, PKL_CACHE_ROOT  # noqa: E402
MIT_DIR      = DATA_4_HI_ROOT / "clean" / "MIT"
HUST_DIR     = DATA_4_HI_ROOT / "clean" / "HUST"

for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _font
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

# 최대 12개 시나리오를 커버하는 팔레트 (색조 구분 최대화)
_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#2980b9",
    "#27ae60", "#c0392b", "#8e44ad", "#16a085",
]


def _sample_shade(base_hex: str, sample_idx: int, n_samples: int) -> str:
    """zone의 기본 색조(hue)는 유지하고 명도(lightness)만 샘플 인덱스별로 바꾼다.

    같은 zone(같은 색조)에 속한 n_samples개 세그먼트가 "겹쳐진 하나"가 아니라
    "서로 다른 개별 인스턴스"임을 한눈에 구분하기 위한 용도(verify-fix Part D,
    docs/260816_RESULTS.md §2-6). sample_idx=0(zone 내 가장 이른 위치)일수록 밝고,
    마지막 샘플일수록 어둡다.
    """
    import colorsys
    r = int(base_hex[1:3], 16) / 255
    g = int(base_hex[3:5], 16) / 255
    b = int(base_hex[5:7], 16) / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l_new = (0.72 - 0.42 * (sample_idx / (n_samples - 1))) if n_samples > 1 else l
    r2, g2, b2 = colorsys.hls_to_rgb(h, l_new, min(s * 1.05, 1.0))
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r2 * 255)), int(round(g2 * 255)), int(round(b2 * 255)))

# q_frac_wide 세그먼트 파라미터 — CLI --axis-config 대신 여기서 직접 수정 가능.
# (--axis-config를 명시하면 그 값이 아래 값을 덮어씀)
QFRAC_WIDE_AXIS_CONFIG = {
    "n1": 0.45,
    "n2": 0.09,
    "n_samples": 4,
}


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _add_phase(df):
    df = df.copy()
    cur = df["current_A"]
    df["phase"] = "rest"
    df.loc[cur >  0.01, "phase"] = "charge"
    df.loc[cur < -0.01, "phase"] = "discharge"
    return df


def _build_arrays(rows):
    """(v, i_mag, t, dt, q_cum) 반환."""
    v  = rows["voltage_V"].values.astype(float)
    i  = np.abs(rows["current_A"].values.astype(float))
    t  = rows["time_s"].values.astype(float)
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    q  = np.cumsum(i * dt) / 3600.0
    return v, i, t, dt, q


def _t_range_from_q(t_full: np.ndarray, q_full: np.ndarray, q_seg: np.ndarray):
    """세그먼트 Q 범위 → 원본 t 배열에서 시작/끝 시각 추출.

    q_full이 단조 증가인 경우 searchsorted로 빠르게 찾는다.
    """
    if len(q_seg) == 0 or len(q_full) == 0:
        return None, None
    q0, q1 = float(q_seg[0]), float(q_seg[-1])
    idx0 = int(np.searchsorted(q_full, q0, side="left"))
    idx1 = int(np.searchsorted(q_full, q1, side="right")) - 1
    idx0 = np.clip(idx0, 0, len(t_full) - 1)
    idx1 = np.clip(idx1, idx0, len(t_full) - 1)
    return float(t_full[idx0]), float(t_full[idx1])


def _rand_suffix_tag(axis_cfg: dict) -> str:
    """random_segment 파라미터 → 태그 접미사. hi_correlation._rand_suffix와 동일 규칙(중복 정의)."""
    if not axis_cfg.get("random_segment", False):
        return ""
    return f"_random-L{int(axis_cfg.get('seg_len_pts', 20))}"


def _adaptive_suffix_tag(axis_cfg: dict) -> str:
    """adaptive_samples=True 면 태그 접미사(_adaptive-th{max_overlap%}), 아니면 빈 문자열.

    q_abs 전용 — max_overlap이 다르면 존별 n_k가 달라져 결과가 달라지므로, 켜고 끄거나
    max_overlap을 바꿔가며 비교할 때 출력 폴더가 서로 덮어쓰지 않도록 구분한다.
    (hi_correlation.py._qabs_tag 등 하위 파이프라인에는 아직 이 태그가 반영되지 않음 —
    adaptive_samples가 이번 세션에 새로 추가된 옵션이라 seg_diagnose 쪽만 우선 반영.)
    """
    if not axis_cfg.get("adaptive_samples", False):
        return ""
    ov = int(round(axis_cfg.get("max_overlap", 0.50) * 100))
    return f"_adaptive-th{ov}"


def _qabs_tag(axis_cfg: dict) -> str:
    """q_abs 파라미터 → 파일/디렉터리 식별 태그. hi_correlation._qabs_tag와 동일 규칙
    (+ adaptive_samples 접미사, 위 _adaptive_suffix_tag 참조)."""
    ms = int(round(axis_cfg.get("mid_start", 0.20) * 100))
    me = int(round(axis_cfg.get("mid_end", 0.50) * 100))
    sl = int(round(axis_cfg.get("seg_len", 0.15) * 100))
    ns = int(axis_cfg.get("n_samples", 4))
    return f"ms-{ms}%_me-{me}%_sl-{sl}%_N-{ns}{_rand_suffix_tag(axis_cfg)}{_adaptive_suffix_tag(axis_cfg)}"


def _condition_tag(axis: str, axis_cfg: dict) -> str:
    """--mode compare 조건 표시용 라벨.

    hi_correlation._qfw_tag/_vqslope_tag와 동일 규칙(다른 파일 참조 대신 중복 정의 —
    이 프로젝트에서 train_scr.py/train_classifier.py도 같은 방식으로 태그 로직을 각자 보유).
    """
    if axis == "q_frac_wide":
        n1 = int(round(axis_cfg.get("n1", 0.4) * 100))
        n2 = int(round(axis_cfg.get("n2", 0.2) * 100))
        ns = int(axis_cfg.get("n_samples", 4))
        return f"q_frac_wide/n1-{n1}%_n2-{n2}%_N-{ns}{_rand_suffix_tag(axis_cfg)}"
    if axis == "q_abs":
        return f"q_abs/{_qabs_tag(axis_cfg)}"
    if axis == "vqslope":
        mode = str(axis_cfg.get("mode", "dva")).lower()
        ns   = int(axis_cfg.get("n_samples", 1))
        return f"vqslope/{mode}_N-{ns}{_rand_suffix_tag(axis_cfg)}"
    return f"{axis}/{json.dumps(axis_cfg, ensure_ascii=False)}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. 세그먼트 통계 수집
# ─────────────────────────────────────────────────────────────────────────────

def collect_stats(pkl_dir: Path, segmenter, spec_names: list) -> dict:
    """pkl_dir 전체 셀 스캔 → 시나리오별 통계 dict."""
    files = sorted(pkl_dir.glob("*.pkl"))
    if not files:
        print(f"  [경고] {pkl_dir} 에 pkl 파일 없음")
        return {n: {"count": 0, "rows": [], "dur_s": []} for n in spec_names}

    stats = {n: {"count": 0, "rows": [], "dur_s": []} for n in spec_names}
    _empty = np.empty(0, dtype=float)

    for f in tqdm(files, desc=f"  [{pkl_dir.name}] 통계"):
        try:
            with open(f, "rb") as fh:
                raw = pickle.load(fh)
        except Exception:
            continue

        df_all = raw.get("cycles")
        if df_all is None:
            continue
        if "phase" not in df_all.columns:
            df_all = _add_phase(df_all)

        cell_id = raw.get("meta", {}).get("cell_id", f.stem)

        for cyc, grp in df_all.groupby("cycle"):
            if int(cyc) == 0:
                continue
            dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
            chg = grp[grp["phase"] == "charge"].sort_values("time_s")
            if len(dis) < 30:
                continue

            v_d, i_d, _, dt_d, q_d = _build_arrays(dis)
            for rec in segmenter.iter_segments(cell_id, int(cyc), v_d, i_d, dt_d, q_d):
                sn = rec.meta.get("seg_name") or spec_names[rec.scenario_id]
                if sn in stats:
                    stats[sn]["count"] += 1
                    stats[sn]["rows"].append(len(rec.v))
                    stats[sn]["dur_s"].append(float(np.sum(rec.dt)))

            if len(chg) >= 20:
                v_c, i_c, _, dt_c, q_c = _build_arrays(chg)
                for rec in segmenter.iter_segments(
                    cell_id, int(cyc),
                    _empty, _empty, _empty, _empty,
                    v_c, i_c, dt_c, q_c,
                ):
                    sn = rec.meta.get("seg_name") or spec_names[rec.scenario_id]
                    if sn in stats:
                        stats[sn]["count"] += 1
                        stats[sn]["rows"].append(len(rec.v))
                        stats[sn]["dur_s"].append(float(np.sum(rec.dt)))

    return stats


def print_stats(stats: dict, spec_names: list, axis: str, dataset: str,
                out_path: Path | None = None):
    w = 60
    lines = [
        f"\n{'─' * w}",
        f"  [{axis.upper()}]  {dataset}  세그먼트 통계",
        f"{'─' * w}",
        f"  {'시나리오':<22} {'세그먼트 수':>10} {'평균 행 수':>11} {'평균 시간(s)':>13}",
        f"  {'─' * (w - 2)}",
    ]
    total = 0
    for name in spec_names:
        s   = stats.get(name, {})
        cnt = s.get("count", 0)
        total += cnt
        if cnt == 0:
            lines.append(f"  {name:<22} {'0':>10} {'—':>11} {'—':>13}")
        else:
            avg_r = np.mean(s["rows"])
            avg_d = np.mean(s["dur_s"])
            lines.append(f"  {name:<22} {cnt:>10,} {avg_r:>11.1f} {avg_d:>13.1f}")
    lines += [
        f"  {'─' * (w - 2)}",
        f"  {'합계':<22} {total:>10,}",
        f"{'─' * w}",
        "",
    ]
    text = "\n".join(lines)
    print(text)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"  통계 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1b. q_frac_wide 전용 — 생존율 + 시간길이 + 전압길이 통계
#
# iter_segments()는 min_pts를 통과한 세그먼트만 yield하므로, 생존율(시도 대비
# 통과 비율)을 구하려면 세그먼터가 내부적으로 시도 횟수도 세야 한다.
# QFracWideSegmenter._extract()에 추가된 n_attempted/n_yielded 카운터를 사용.
# ─────────────────────────────────────────────────────────────────────────────

def collect_qfracwide_stats(pkl_dir: Path, segmenter, spec_names: list) -> dict:
    """pkl_dir(한 데이터셋) 전체 스캔 → 시나리오별 {attempted, yielded, dur_s, v_range}.

    segmenter.reset_counters()로 시작해 segmenter.n_attempted/n_yielded를 그대로
    읽어 생존율 분모로 사용한다 — q_frac_wide 세그먼터 전용(다른 축은 카운터 없음).
    """
    segmenter.reset_counters()
    files = sorted(pkl_dir.glob("*.pkl"))
    if not files:
        print(f"  [경고] {pkl_dir} 에 pkl 파일 없음")
        return {n: {"dur_s": [], "v_range": []} for n in spec_names}

    stats = {n: {"dur_s": [], "v_range": []} for n in spec_names}
    _empty = np.empty(0, dtype=float)

    for f in tqdm(files, desc=f"  [{pkl_dir.name}] 생존율/길이 통계"):
        try:
            with open(f, "rb") as fh:
                raw = pickle.load(fh)
        except Exception:
            continue
        df_all = raw.get("cycles")
        if df_all is None:
            continue
        if "phase" not in df_all.columns:
            df_all = _add_phase(df_all)
        cell_id = raw.get("meta", {}).get("cell_id", f.stem)

        for cyc, grp in df_all.groupby("cycle"):
            if int(cyc) == 0:
                continue
            dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
            chg = grp[grp["phase"] == "charge"].sort_values("time_s")
            if len(dis) < 30:
                continue

            v_d, i_d, _, dt_d, q_d = _build_arrays(dis)
            for rec in segmenter.iter_segments(cell_id, int(cyc), v_d, i_d, dt_d, q_d):
                sn = rec.meta.get("seg_name") or spec_names[rec.scenario_id]
                if sn in stats:
                    stats[sn]["dur_s"].append(float(np.sum(rec.dt)))
                    stats[sn]["v_range"].append(float(rec.v.max() - rec.v.min()))

            if len(chg) >= 20:
                v_c, i_c, _, dt_c, q_c = _build_arrays(chg)
                for rec in segmenter.iter_segments(
                    cell_id, int(cyc), _empty, _empty, _empty, _empty, v_c, i_c, dt_c, q_c,
                ):
                    sn = rec.meta.get("seg_name") or spec_names[rec.scenario_id]
                    if sn in stats:
                        stats[sn]["dur_s"].append(float(np.sum(rec.dt)))
                        stats[sn]["v_range"].append(float(rec.v.max() - rec.v.min()))

    return stats


def _collect_one_cell_qfw(args: tuple) -> tuple:
    """워커 프로세스: 셀 1개 처리 → (stats, n_attempted, n_yielded, candidate_n_points).

    ProcessPoolExecutor로 호출되므로 top-level 함수여야 pickle 가능하다.
    프로세스마다 독립된 QFracWideSegmenter를 새로 만들어 상태를 로컬로 누적한다
    (segmenter 인스턴스는 프로세스 간 공유 불가).
    """
    pkl_path_str, axis_cfg, spec_names = args
    from common.scenario import get_segmenter

    segmenter = get_segmenter("q_frac_wide", {"q_frac_wide": axis_cfg})
    stats = {n: {"dur_s": [], "v_range": []} for n in spec_names}
    _empty = np.empty(0, dtype=float)

    f = Path(pkl_path_str)
    try:
        with open(f, "rb") as fh:
            raw = pickle.load(fh)
    except Exception:
        return stats, {}, {}, {}

    df_all = raw.get("cycles")
    if df_all is None:
        return stats, {}, {}, {}
    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)
    cell_id = raw.get("meta", {}).get("cell_id", f.stem)

    for cyc, grp in df_all.groupby("cycle"):
        if int(cyc) == 0:
            continue
        dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
        chg = grp[grp["phase"] == "charge"].sort_values("time_s")
        if len(dis) < 30:
            continue

        v_d, i_d, _, dt_d, q_d = _build_arrays(dis)
        for rec in segmenter.iter_segments(cell_id, int(cyc), v_d, i_d, dt_d, q_d):
            sn = rec.meta.get("seg_name") or spec_names[rec.scenario_id]
            if sn in stats:
                stats[sn]["dur_s"].append(float(np.sum(rec.dt)))
                stats[sn]["v_range"].append(float(rec.v.max() - rec.v.min()))

        if len(chg) >= 20:
            v_c, i_c, _, dt_c, q_c = _build_arrays(chg)
            for rec in segmenter.iter_segments(
                cell_id, int(cyc), _empty, _empty, _empty, _empty, v_c, i_c, dt_c, q_c,
            ):
                sn = rec.meta.get("seg_name") or spec_names[rec.scenario_id]
                if sn in stats:
                    stats[sn]["dur_s"].append(float(np.sum(rec.dt)))
                    stats[sn]["v_range"].append(float(rec.v.max() - rec.v.min()))

    return stats, dict(segmenter.n_attempted), dict(segmenter.n_yielded), dict(segmenter.candidate_n_points)


def collect_qfracwide_stats_parallel(
    pkl_dir: Path, axis_cfg: dict, spec_names: list, n_workers: int = 4,
) -> tuple:
    """collect_qfracwide_stats의 병렬 버전. ProcessPoolExecutor로 셀 단위 분산 후 병합.

    Returns: (stats, n_attempted, n_yielded, candidate_n_points) — 병합된 결과.
    segmenter 객체가 아니라 dict 3종을 직접 반환한다(워커별 인스턴스는 버려지므로).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    files = sorted(pkl_dir.glob("*.pkl"))
    if not files:
        print(f"  [경고] {pkl_dir} 에 pkl 파일 없음")
        empty = {n: {"dur_s": [], "v_range": []} for n in spec_names}
        return empty, {}, {}, {}

    stats: dict = {n: {"dur_s": [], "v_range": []} for n in spec_names}
    n_attempted: dict = {}
    n_yielded: dict = {}
    candidate_n_points: dict = {}

    task_args = [(str(f), axis_cfg, spec_names) for f in files]

    if n_workers <= 1:
        results = [_collect_one_cell_qfw(a) for a in tqdm(task_args, desc=f"  [{pkl_dir.name}]")]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_collect_one_cell_qfw, a): a for a in task_args}
            with tqdm(total=len(futs), desc=f"  [{pkl_dir.name}] ({n_workers} workers)") as pbar:
                for fut in as_completed(futs):
                    results.append(fut.result())
                    pbar.update(1)

    for cell_stats, cell_att, cell_yld, cell_cnp in results:
        for sn in spec_names:
            stats[sn]["dur_s"].extend(cell_stats.get(sn, {}).get("dur_s", []))
            stats[sn]["v_range"].extend(cell_stats.get(sn, {}).get("v_range", []))
        for sn, c in cell_att.items():
            n_attempted[sn] = n_attempted.get(sn, 0) + c
        for sn, c in cell_yld.items():
            n_yielded[sn] = n_yielded.get(sn, 0) + c
        for sn, pts in cell_cnp.items():
            candidate_n_points.setdefault(sn, []).extend(pts)

    return stats, n_attempted, n_yielded, candidate_n_points


class _CounterProxy:
    """collect_qfracwide_stats_parallel의 병합 결과를 세그먼터 인스턴스처럼 보이게
    감싸는 얇은 래퍼 — print_qfracwide_stats/plot_*가 seg.n_attempted 등을 그대로
    읽을 수 있도록 인터페이스를 맞춘다 (실제 세그먼터는 아니지만 이 3개 속성만 사용됨)."""

    def __init__(self, n_attempted: dict, n_yielded: dict, candidate_n_points: dict):
        self.n_attempted = n_attempted
        self.n_yielded = n_yielded
        self.candidate_n_points = candidate_n_points


def print_qfracwide_stats(
    per_dataset_stats: dict,   # {"MIT": stats_dict, "HUST": stats_dict}
    per_dataset_segmenter,     # {"MIT": segmenter, "HUST": segmenter} (n_attempted/n_yielded 보유)
    spec_names: list,
    n1: float, n2: float, n_samples: int,
    out_path: Path | None = None,
) -> None:
    w = 100
    lines = [
        f"\n{'=' * w}",
        f"  q_frac_wide  생존율 / 시간길이 / 전압길이 통계   (n1={n1}, n2={n2}, n_samples={n_samples})",
        f"{'=' * w}",
    ]
    header = (f"  {'dataset':<6} {'시나리오':<10} {'시도':>8} {'생존':>8} {'생존율':>7}  "
              f"{'시간(s) mean':>12} {'median':>8}  {'전압폭(V) mean':>14} {'median':>8} {'max':>8}")
    for ds in per_dataset_stats:
        lines.append(f"\n{'─' * w}")
        lines.append(f"  [{ds}]")
        lines.append(header)
        lines.append(f"  {'─' * (w - 2)}")
        stats = per_dataset_stats[ds]
        seg = per_dataset_segmenter[ds]
        for name in spec_names:
            attempted = seg.n_attempted.get(name, 0)
            yielded   = seg.n_yielded.get(name, 0)
            surv_pct  = (100 * yielded / attempted) if attempted > 0 else float("nan")
            dur   = stats.get(name, {}).get("dur_s", [])
            vrng  = stats.get(name, {}).get("v_range", [])
            if dur:
                dur_mean, dur_med = np.mean(dur), np.median(dur)
            else:
                dur_mean = dur_med = float("nan")
            if vrng:
                v_mean, v_med, v_max = np.mean(vrng), np.median(vrng), np.max(vrng)
            else:
                v_mean = v_med = v_max = float("nan")
            lines.append(
                f"  {ds:<6} {name:<10} {attempted:>8,} {yielded:>8,} {surv_pct:>6.1f}%  "
                f"{dur_mean:>12.2f} {dur_med:>8.2f}  {v_mean:>14.4f} {v_med:>8.4f} {v_max:>8.4f}"
            )
    lines.append(f"\n{'=' * w}\n")
    text = "\n".join(lines)
    print(text)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"  통계 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1c. min_pts 임계값 스윕 — 재스캔 없이 segmenter.candidate_n_points 분포로 계산
# ─────────────────────────────────────────────────────────────────────────────

def print_min_pts_sweep(
    per_dataset_segmenter: dict,     # {"MIT": segmenter, "HUST": segmenter}
    spec_names: list,
    thresholds: list,                # 예: [5, 6, 8, 10]
    n1: float, n2: float, n_samples: int,
    out_path: Path | None = None,
) -> None:
    """candidate_n_points(모든 시도의 실제 원시 포인트 수) 분포로 min_pts별
    생존율을 재스캔 없이 즉시 계산해 표로 출력한다."""
    w = 14 + 10 + 8 * len(thresholds) + 2
    lines = [
        f"\n{'=' * w}",
        f"  q_frac_wide  min_pts 임계값별 생존율 스윕   (n1={n1}, n2={n2}, n_samples={n_samples})",
        f"{'=' * w}",
    ]
    th_header = "".join(f"{'min_pts='+str(t):>10}" for t in thresholds)
    header = f"  {'dataset':<6} {'시나리오':<10}{th_header}"
    for ds, seg in per_dataset_segmenter.items():
        lines.append(f"\n{'─' * w}")
        lines.append(f"  [{ds}]")
        lines.append(header)
        lines.append(f"  {'─' * (w - 2)}")
        for name in spec_names:
            counts = np.array(seg.candidate_n_points.get(name, []))
            if len(counts) == 0:
                row = "".join(f"{'—':>10}" for _ in thresholds)
            else:
                row = "".join(f"{100*(counts>=t).mean():>9.1f}%" for t in thresholds)
            n_att = len(counts)
            lines.append(f"  {ds:<6} {name:<10}{row}   (시도={n_att:,}, "
                        f"pts median={int(np.median(counts)) if n_att else '—'})")
    lines.append(f"\n{'=' * w}\n")
    text = "\n".join(lines)
    print(text)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"  통계 저장: {out_path}")


def plot_min_pts_sweep(
    per_dataset_segmenter: dict,
    spec_names: list,
    n1: float, n2: float, n_samples: int,
    out_path: Path,
    th_min: int = 3,
    th_max: int = 20,
) -> None:
    """min_pts를 th_min~th_max 전체 스윕한 생존율 곡선. 시나리오별 서브플롯,
    각 서브플롯에 MIT/HUST 두 선. x=min_pts, y=생존율(%)."""
    datasets  = list(per_dataset_segmenter.keys())
    colors    = {"MIT": "#3498db", "HUST": "#e74c3c"}
    n_scen    = len(spec_names)
    ncols     = 3
    nrows     = int(np.ceil(n_scen / ncols))
    ths       = np.arange(th_min, th_max + 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for idx, name in enumerate(spec_names):
        ax = axes[idx // ncols][idx % ncols]
        for ds in datasets:
            seg = per_dataset_segmenter[ds]
            counts = np.array(seg.candidate_n_points.get(name, []))
            if len(counts) == 0:
                continue
            surv = [100 * (counts >= t).mean() for t in ths]
            ax.plot(ths, surv, marker="o", markersize=3, label=ds,
                    color=colors.get(ds, None))
        ax.axvline(8, color="gray", ls="--", lw=1, alpha=0.6)
        ax.text(8.2, 5, "n≥8\n(diff/lfp 가능)", fontsize=7, color="gray")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("min_pts")
        ax.set_ylabel("생존율 (%)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(n_scen, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"min_pts 임계값 스윕 — 시나리오별 생존율 곡선   "
                f"n1={n1} n2={n2} n_samples={n_samples}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  플롯 저장: {out_path}")


def plot_qfracwide_stats(
    per_dataset_stats: dict,   # {"MIT": stats_dict, "HUST": stats_dict}
    per_dataset_segmenter: dict,
    spec_names: list,
    n1: float, n2: float, n_samples: int,
    out_path: Path,
) -> None:
    """3패널 플롯: 생존율(막대) / 시간길이(박스플롯) / 전압폭(박스플롯).

    각 패널은 시나리오(x축)별로 MIT·HUST를 나란히 비교한다.
    """
    datasets = list(per_dataset_stats.keys())          # ["MIT","HUST"]
    n_scen   = len(spec_names)
    colors   = {"MIT": "#3498db", "HUST": "#e74c3c"}

    fig, axes = plt.subplots(3, 1, figsize=(max(9, n_scen * 1.6), 13))

    # ── 패널 1: 생존율 막대그래프 ────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(n_scen)
    width = 0.35
    for di, ds in enumerate(datasets):
        seg = per_dataset_segmenter[ds]
        surv = []
        for name in spec_names:
            att = seg.n_attempted.get(name, 0)
            yld = seg.n_yielded.get(name, 0)
            surv.append(100 * yld / att if att > 0 else 0.0)
        offset = (di - (len(datasets) - 1) / 2) * width
        bars = ax.bar(x + offset, surv, width, label=ds,
                      color=colors.get(ds, None), edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, surv):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(spec_names)
    ax.set_ylabel("생존율 (%)  = yielded / attempted")
    ax.set_ylim(0, 108)
    ax.set_title(f"q_frac_wide 세그먼트 생존율 (min_pts 통과 비율)   "
                f"n1={n1}  n2={n2}  n_samples={n_samples}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ── 패널 2/3 공통: 시나리오별 [MIT박스, HUST박스] 나란히 배치 ─────────────
    def _grouped_boxplot(ax, key: str, ylabel: str, title: str):
        plot_data, positions, box_colors = [], [], []
        pos = 0.0
        group_gap = 0.6
        for name in spec_names:
            for ds in datasets:
                data = per_dataset_stats[ds].get(name, {}).get(key, [])
                plot_data.append(data if len(data) > 0 else [np.nan])
                positions.append(pos)
                box_colors.append(colors.get(ds, "#999999"))
                pos += 1.0
            pos += group_gap
        bp = ax.boxplot(plot_data, positions=positions, widths=0.8,
                        patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        group_centers = []
        step = len(datasets) + group_gap
        for gi in range(n_scen):
            start = gi * step
            group_centers.append(start + (len(datasets) - 1) / 2.0)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(spec_names)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        handles = [plt.matplotlib.patches.Patch(facecolor=colors[ds], alpha=0.7, label=ds)
                  for ds in datasets]
        ax.legend(handles=handles, loc="upper right")

    _grouped_boxplot(axes[1], "dur_s", "세그먼트 시간 길이 (s)",
                     "세그먼트별 시간 길이 분포 (이상치 제외)")
    _grouped_boxplot(axes[2], "v_range", "세그먼트 전압 폭 V_max-V_min (V)",
                     "세그먼트별 전압 폭 분포 (이상치 제외)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  플롯 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 사이클 시각화  (--mode segment)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cycle_segments(
    pkl_path: Path,
    segmenter,
    spec_names: list,
    cycle_id: int,
    axis: str,
    out_path: Path,
):
    """특정 셀 사이클을 세그먼트별 색으로 분리해 하나의 사이클로 시각화.

    레이아웃
    ─────────────────────────────────────────────────────
    [상단 전체] V vs time_s — 충전+방전 연속 (세그먼트 색상 밴드)
    [좌하]     V vs Q      — 방전 세그먼트 색상 오버레이
    [우하]     V vs Q      — 충전 세그먼트 색상 오버레이

    iter_segments는 rcs처럼 무작위 축도 있으므로 각 방향당 1회만 호출해
    결과를 리스트로 캐싱 후 여러 패널에서 재사용한다.
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    df_all  = raw.get("cycles")
    cell_id = raw.get("meta", {}).get("cell_id", pkl_path.stem)

    if df_all is None:
        print(f"  [경고] {pkl_path.name}: cycles 없음")
        return

    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    # ── 사이클 선택 ───────────────────────────────────────────────────────────
    valid_cycs = sorted(c for c in df_all["cycle"].unique() if c != 0)
    if not valid_cycs:
        print("  [경고] 유효 사이클 없음")
        return
    if cycle_id == 0 or cycle_id not in valid_cycs:
        cycle_id = valid_cycs[0]
        print(f"  → 첫 번째 유효 사이클 {cycle_id} 사용")

    grp = df_all[df_all["cycle"] == cycle_id]
    dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
    chg = grp[grp["phase"] == "charge"].sort_values("time_s")

    # ── 세그먼트 한 번씩 수집 (iter_segments 재호출 방지) ──────────────────────
    _empty = np.empty(0, dtype=float)

    dis_arrays: "tuple | None" = None
    dis_recs:   list = []
    if len(dis) >= 30:
        v_d, i_d, t_d, dt_d, q_d = _build_arrays(dis)
        dis_arrays = (v_d, i_d, t_d, dt_d, q_d)
        dis_recs   = [
            (rec.meta.get("seg_name") or spec_names[rec.scenario_id], rec)
            for rec in segmenter.iter_segments(cell_id, cycle_id, v_d, i_d, dt_d, q_d)
        ]

    chg_arrays: "tuple | None" = None
    chg_recs:   list = []
    if len(chg) >= 20:
        v_c, i_c, t_c, dt_c, q_c = _build_arrays(chg)
        chg_arrays = (v_c, i_c, t_c, dt_c, q_c)
        chg_recs   = [
            (rec.meta.get("seg_name") or spec_names[rec.scenario_id], rec)
            for rec in segmenter.iter_segments(
                cell_id, cycle_id, _empty, _empty, _empty, _empty, v_c, i_c, dt_c, q_c
            )
        ]

    # ── 색상 매핑 ─────────────────────────────────────────────────────────────
    color_map = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(spec_names)}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.3, 1.0],
        hspace=0.48, wspace=0.28,
    )
    ax_t  = fig.add_subplot(gs[0, :])   # V vs time_s (전체 사이클)
    ax_dq = fig.add_subplot(gs[1, 0])   # V vs Q (방전)
    ax_cq = fig.add_subplot(gs[1, 1])   # V vs Q (충전)

    cap_label = ""
    if len(dis) > 0 and "capacity_Ah" in dis.columns:
        dis_cap = dis["capacity_Ah"].iloc[0]
        if np.isfinite(dis_cap):
            cap_label = f"  │  Q = {dis_cap:.4f} Ah"

    fig.suptitle(
        f"[{axis}]  {cell_id}  Cycle {cycle_id}{cap_label}",
        fontsize=12, fontweight="bold",
    )

    legend_added: set = set()
    top_handles: "list[mpatches.Patch]" = []

    # ═════════════════════════════════════════════════════════════════════════
    # 패널 0: V vs time_s  (충전 + 방전 연속)
    # ═════════════════════════════════════════════════════════════════════════
    ax_t.set_facecolor("#f8f9fa")
    ax_t.set_xlabel("Time [s]", fontsize=9)
    ax_t.set_ylabel("Voltage [V]", fontsize=9)
    ax_t.tick_params(labelsize=8)
    ax_t.set_title("전체 사이클 V-t  (세그먼트 색상 밴드)", fontsize=9, pad=4)

    # 배경 전체 곡선 (회색)
    for phase_grp, fill_col, line_col, lbl in [
        (chg, "#d6eaf8", "#2980b9", "Charge"),
        (dis, "#fdf2e9", "#ca6f1e", "Discharge"),
    ]:
        if len(phase_grp) < 10:
            continue
        t_bg = phase_grp["time_s"].values.astype(float)
        v_bg = phase_grp["voltage_V"].values.astype(float)
        ax_t.fill_between(t_bg, v_bg.min() - 0.02, v_bg, color=fill_col, alpha=0.2, zorder=0)
        ax_t.plot(t_bg, v_bg, color="#cccccc", lw=0.8, zorder=1, alpha=0.9)
        ax_t.text(
            (t_bg[0] + t_bg[-1]) / 2, 0.03, lbl,
            transform=ax_t.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color=line_col, alpha=0.65,
        )

    if len(chg) >= 10 and len(dis) >= 10:
        ax_t.axvline(float(chg["time_s"].iloc[-1]),
                     color="#555555", lw=1.2, ls="--", zorder=4, alpha=0.6)

    # 세그먼트 색상 밴드 + 오버레이 (방전)
    if dis_arrays is not None:
        _, _, t_d, _, q_d = dis_arrays
        for sn, rec in dis_recs:
            c = color_map.get(sn, "black")
            t0, t1 = _t_range_from_q(t_d, q_d, rec.q)
            if t0 is not None:
                ax_t.axvspan(t0, t1, color=c, alpha=0.35, zorder=2)
            idx0 = int(np.searchsorted(q_d, rec.q[0], "left"))
            idx1 = int(np.searchsorted(q_d, rec.q[-1], "right"))
            t_seg = t_d[idx0:idx1]
            n = min(len(t_seg), len(rec.v))
            ax_t.plot(t_seg[:n], rec.v[:n], color=c, lw=2.3, alpha=0.9, zorder=3)
            if sn not in legend_added:
                top_handles.append(mpatches.Patch(color=c, label=sn))
                legend_added.add(sn)

    # 세그먼트 색상 밴드 + 오버레이 (충전)
    if chg_arrays is not None:
        _, _, t_c, _, q_c = chg_arrays
        for sn, rec in chg_recs:
            c = color_map.get(sn, "black")
            t0, t1 = _t_range_from_q(t_c, q_c, rec.q)
            if t0 is not None:
                ax_t.axvspan(t0, t1, color=c, alpha=0.35, zorder=2)
            idx0 = int(np.searchsorted(q_c, rec.q[0], "left"))
            idx1 = int(np.searchsorted(q_c, rec.q[-1], "right"))
            t_seg = t_c[idx0:idx1]
            n = min(len(t_seg), len(rec.v))
            ax_t.plot(t_seg[:n], rec.v[:n], color=c, lw=2.3, alpha=0.9, zorder=3)
            if sn not in legend_added:
                top_handles.append(mpatches.Patch(color=c, label=sn))
                legend_added.add(sn)

    if top_handles:
        ax_t.legend(handles=top_handles, fontsize=7.5,
                    loc="upper right", framealpha=0.85,
                    ncol=min(len(top_handles), 4))

    # ═════════════════════════════════════════════════════════════════════════
    # 패널 1: V vs Q — 방전
    # ═════════════════════════════════════════════════════════════════════════
    ax_dq.set_facecolor("#fdfbf7")
    ax_dq.set_xlabel("Q_cum [Ah]", fontsize=8)
    ax_dq.set_ylabel("Voltage [V]", fontsize=8)
    ax_dq.set_title("방전  V-Q", fontsize=9, pad=4)
    ax_dq.tick_params(labelsize=7)

    if dis_arrays is not None:
        v_d, _, _, _, q_d = dis_arrays
        ax_dq.plot(q_d, v_d, color="#cccccc", lw=1.0, zorder=1)
        for sn, rec in dis_recs:
            c = color_map.get(sn, "black")
            ax_dq.plot(rec.q, rec.v, color=c, lw=2.5, alpha=0.9, zorder=2)
            ax_dq.scatter([rec.q[0]], [rec.v[0]],
                          color=c, s=40, zorder=4, edgecolors="white", linewidths=0.6)
    else:
        ax_dq.text(0.5, 0.5, "방전 데이터 없음", ha="center", va="center",
                   transform=ax_dq.transAxes, color="gray", fontsize=9)

    # ═════════════════════════════════════════════════════════════════════════
    # 패널 2: V vs Q — 충전
    # ═════════════════════════════════════════════════════════════════════════
    ax_cq.set_facecolor("#f7fbfd")
    ax_cq.set_xlabel("Q_cum [Ah]", fontsize=8)
    ax_cq.set_ylabel("Voltage [V]", fontsize=8)
    ax_cq.set_title("충전  V-Q", fontsize=9, pad=4)
    ax_cq.tick_params(labelsize=7)

    if chg_arrays is not None:
        v_c, _, _, _, q_c = chg_arrays
        ax_cq.plot(q_c, v_c, color="#cccccc", lw=1.0, zorder=1)
        for sn, rec in chg_recs:
            c = color_map.get(sn, "black")
            ax_cq.plot(rec.q, rec.v, color=c, lw=2.5, alpha=0.9, zorder=2)
            ax_cq.scatter([rec.q[0]], [rec.v[0]],
                          color=c, s=40, zorder=4, edgecolors="white", linewidths=0.6)
    else:
        ax_cq.text(0.5, 0.5, "충전 데이터 없음", ha="center", va="center",
                   transform=ax_cq.transAxes, color="gray", fontsize=9)

    # ── 공유 범례 (하단) ──────────────────────────────────────────────────────
    all_handles = [mpatches.Patch(color=color_map[n], label=n) for n in spec_names]
    fig.legend(
        handles=all_handles, fontsize=8.5,
        loc="lower center", ncol=min(len(spec_names), 6),
        bbox_to_anchor=(0.5, -0.01), framealpha=0.92,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  플롯 저장: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. IC 커브 시각화  (--mode ic)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ic(
    v: np.ndarray,
    q: np.ndarray,
    n_points: int = 600,
    smooth_sigma: float = 4.0,
) -> "tuple[np.ndarray, np.ndarray] | tuple[None, None]":
    """dQ/dV vs V (증분용량 커브) 계산.

    1. V 기준 오름차순 정렬 + 중복 제거
    2. 균등 V 격자에 Q 보간
    3. Gaussian 스무딩 (sigma: V 격자 포인트 단위)
    4. 수치 미분 → dQ/dV

    방전은 V 오름차순 시 Q가 내림차순이므로 dQ/dV < 0.
    호출 측에서 `-dqdv`로 부호를 반전해 피크를 양수로 표시.
    """
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        return None, None

    if len(v) < 30:
        return None, None

    sort_idx = np.argsort(v)
    v_s = v[sort_idx]
    q_s = q[sort_idx]

    # 중복 V 제거 (같은 전압값이 연속하면 미분 불안정)
    _, ui = np.unique(v_s, return_index=True)
    v_u, q_u = v_s[ui], q_s[ui]
    if len(v_u) < 20:
        return None, None

    v_grid   = np.linspace(v_u[0], v_u[-1], n_points)
    q_interp = np.interp(v_grid, v_u, q_u)
    q_smooth = gaussian_filter1d(q_interp, sigma=smooth_sigma)
    dqdv     = np.gradient(q_smooth, v_grid)
    return v_grid, dqdv


def _win_colors(n: int) -> list[str]:
    """창 개수에 맞는 구분 가능한 색상 목록."""
    palette = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#2980b9",
        "#27ae60", "#c0392b", "#8e44ad", "#16a085",
    ]
    return [palette[i % len(palette)] for i in range(n)]


def _draw_vq_bands(ax, edges: list[float], colors: list[str]):
    """V-Q 패널: V 경계 → 가로 밴드 + 파선."""
    for k, (v0, v1) in enumerate(zip(edges[:-1], edges[1:])):
        c = colors[k]
        ax.axhspan(v0, v1, color=c, alpha=0.13, zorder=0)
        ax.axhline(v0, color=c, lw=0.9, ls="--", alpha=0.55, zorder=1)
    ax.axhline(edges[-1], color=colors[-1], lw=0.9, ls="--", alpha=0.55, zorder=1)


def _draw_ic_bands(ax, edges: list[float], colors: list[str]):
    """IC 패널: V 경계 → 세로 밴드 + 파선 + 창 번호 레이블."""
    for k, (v0, v1) in enumerate(zip(edges[:-1], edges[1:])):
        c = colors[k]
        ax.axvspan(v0, v1, color=c, alpha=0.13, zorder=0)
        ax.axvline(v0, color=c, lw=1.1, ls="--", alpha=0.65, zorder=1)
        ax.text(
            (v0 + v1) / 2, 0.98, f"win{k}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5,
            color=c, fontweight="bold", alpha=0.9,
        )
    ax.axvline(edges[-1], color=colors[-1], lw=1.1, ls="--", alpha=0.65, zorder=1)


def plot_ic_windows(
    pkl_path: Path,
    segmenter,
    spec_names: list,
    axis: str,
    out_path: Path,
    n_cycles: int = 6,
):
    """IC 커브 (dQ/dV vs V) + 멀티사이클 V-Q — 창 경계 오버레이.

    레이아웃 (2행 × 2열)
    ┌──────────────────────┬──────────────────────┐
    │  방전 V-Q 멀티사이클  │  충전 V-Q 멀티사이클  │  ← 창=가로밴드
    ├──────────────────────┼──────────────────────┤
    │  방전 dQ/dV vs V     │  충전 dQ/dV vs V     │  ← 창=세로밴드
    └──────────────────────┴──────────────────────┘

    vwindow 축: 전압 경계 밴드 표시.
    그 외 축  : IC 커브만 표시 (경계 없음).
    사이클 색상: 초록(초기) → 빨강(말기).
    """
    with open(pkl_path, "rb") as fh:
        raw = pickle.load(fh)

    df_all  = raw.get("cycles")
    cell_id = raw.get("meta", {}).get("cell_id", pkl_path.stem)
    if df_all is None:
        print(f"  [경고] {pkl_path.name}: cycles 없음"); return
    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    # ── 대표 사이클 선택 ──────────────────────────────────────────────────────
    valid_cycs = sorted(c for c in df_all["cycle"].unique() if c != 0)
    if not valid_cycs:
        print("  [경고] 유효 사이클 없음"); return
    n     = min(n_cycles, len(valid_cycs))
    picks = [valid_cycs[int(round(i))]
             for i in np.linspace(0, len(valid_cycs) - 1, n)]

    # 사이클 나이 색상: 초록(early) → 빨강(late)
    _cmap = matplotlib.colormaps["RdYlGn_r"].resampled(256)
    cyc_color = [_cmap(ci / max(n - 1, 1)) for ci in range(n)]

    # ── 창 경계 추출 (vwindow 전용) ──────────────────────────────────────────
    dis_edges: list[float] | None = None
    chg_edges: list[float] | None = None
    if hasattr(segmenter, "_get_dis_edges"):
        try:
            dis_edges = list(segmenter._get_dis_edges())
            chg_edges = list(segmenter._get_chg_edges())
        except Exception:
            pass

    n_dis_wins = (len(dis_edges) - 1) if dis_edges else 0
    n_chg_wins = (len(chg_edges) - 1) if chg_edges else 0
    dis_wcolors = _win_colors(max(n_dis_wins, 1))
    chg_wcolors = _win_colors(max(n_chg_wins, 1))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 2, figsize=(16, 11),
        gridspec_kw={"hspace": 0.45, "wspace": 0.30},
    )
    ax_dvq, ax_cvq = axes[0, 0], axes[0, 1]
    ax_dic, ax_cic = axes[1, 0], axes[1, 1]

    has_edges = dis_edges is not None
    boundary_note = "세로선 = 창 경계" if has_edges else "경계 없음 (vwindow 외 축)"
    fig.suptitle(
        f"[{axis.upper()}]  {cell_id}  ·  IC 커브 & V-Q 멀티사이클 분석\n"
        f"색상: 초록(초기) → 빨강(말기)  │  {boundary_note}",
        fontsize=12, fontweight="bold",
    )

    for ax, title in [
        (ax_dvq, "방전  V-Q  (멀티사이클)  — 창=가로밴드"),
        (ax_cvq, "충전  V-Q  (멀티사이클)  — 창=가로밴드"),
        (ax_dic, "방전  IC  dQ/dV  vs  V  — 창=세로밴드"),
        (ax_cic, "충전  IC  dQ/dV  vs  V  — 창=세로밴드"),
    ]:
        ax.set_facecolor("#f8f9fa")
        ax.tick_params(labelsize=8)
        ax.grid(True, lw=0.4, alpha=0.35)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)

    ax_dvq.set_xlabel("Q_cum [Ah]", fontsize=8)
    ax_dvq.set_ylabel("Voltage [V]", fontsize=8)
    ax_cvq.set_xlabel("Q_cum [Ah]", fontsize=8)
    ax_cvq.set_ylabel("Voltage [V]", fontsize=8)
    ax_dic.set_xlabel("Voltage [V]", fontsize=8)
    ax_dic.set_ylabel("dQ/dV  [Ah/V]  (방전: 부호 반전)", fontsize=8)
    ax_cic.set_xlabel("Voltage [V]", fontsize=8)
    ax_cic.set_ylabel("dQ/dV  [Ah/V]", fontsize=8)

    # ── 창 밴드 ───────────────────────────────────────────────────────────────
    if dis_edges:
        _draw_vq_bands(ax_dvq, dis_edges, dis_wcolors)
        _draw_ic_bands(ax_dic, dis_edges, dis_wcolors)
    if chg_edges:
        _draw_vq_bands(ax_cvq, chg_edges, chg_wcolors)
        _draw_ic_bands(ax_cic, chg_edges, chg_wcolors)

    # ── 사이클별 커브 ─────────────────────────────────────────────────────────
    ic_d_vals: list[float] = []
    ic_c_vals: list[float] = []

    for ci, cyc in enumerate(picks):
        col   = cyc_color[ci]
        lw    = 1.2 + ci * 0.15
        alpha = 0.50 + ci * 0.07
        lbl   = f"Cyc {cyc}"

        grp = df_all[df_all["cycle"] == cyc]
        dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
        chg = grp[grp["phase"] == "charge"].sort_values("time_s")

        # 방전
        if len(dis) >= 30:
            v_d, _, _, dt_d, q_d = _build_arrays(dis)
            ax_dvq.plot(q_d, v_d, color=col, lw=lw, alpha=alpha, label=lbl)
            v_ic, dqdv = _compute_ic(v_d, q_d)
            if v_ic is not None:
                # 방전 IC: 부호 반전 → 피크를 양수로
                ax_dic.plot(v_ic, -dqdv, color=col, lw=lw, alpha=alpha)
                ic_d_vals.extend((-dqdv).tolist())

        # 충전
        if len(chg) >= 20:
            v_c, _, _, dt_c, q_c = _build_arrays(chg)
            ax_cvq.plot(q_c, v_c, color=col, lw=lw, alpha=alpha, label=lbl)
            v_ic, dqdv = _compute_ic(v_c, q_c)
            if v_ic is not None:
                ax_cic.plot(v_ic, dqdv, color=col, lw=lw, alpha=alpha)
                ic_c_vals.extend(dqdv.tolist())

    # IC y축: 2~98 percentile 기반 클리핑 (스파이크 이상치 제거)
    for ax, vals in [(ax_dic, ic_d_vals), (ax_cic, ic_c_vals)]:
        if len(vals) > 10:
            p2, p98 = np.percentile(vals, 2), np.percentile(vals, 98)
            span = max(p98 - p2, 1e-6)
            ax.set_ylim(p2 - span * 0.08, p98 + span * 0.12)
        ax.axhline(0, color="#aaaaaa", lw=0.8, ls=":", zorder=0)

    # ── 사이클 범례 (오른쪽에 세로 나열) ─────────────────────────────────────
    from matplotlib.lines import Line2D
    cyc_handles = [
        Line2D([0], [0], color=cyc_color[ci], lw=2.2, label=f"Cyc {picks[ci]}")
        for ci in range(n)
    ]
    fig.legend(
        handles=cyc_handles,
        loc="center right",
        fontsize=8, framealpha=0.88,
        bbox_to_anchor=(1.01, 0.5),
        title="사이클",
    )

    # ── 창 범례 (하단) ────────────────────────────────────────────────────────
    win_handles = []
    if dis_edges:
        for k in range(len(dis_edges) - 1):
            win_handles.append(
                mpatches.Patch(
                    color=dis_wcolors[k], alpha=0.55,
                    label=f"dis win{k}  [{dis_edges[k]:.2f}–{dis_edges[k+1]:.2f} V]",
                )
            )
    if chg_edges:
        for k in range(len(chg_edges) - 1):
            win_handles.append(
                mpatches.Patch(
                    color=chg_wcolors[k], alpha=0.55,
                    label=f"chg win{k}  [{chg_edges[k]:.2f}–{chg_edges[k+1]:.2f} V]",
                )
            )
    if win_handles:
        fig.legend(
            handles=win_handles,
            loc="lower center",
            ncol=min(len(win_handles), 6),
            fontsize=7.5, framealpha=0.90,
            bbox_to_anchor=(0.5, -0.04),
            title="창 구간",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  IC 플롯 저장: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. vqslope 존 분리 시각화  (--mode vqzone)  — vqslope 전용
#    한 사이클에 대해 "왜 여기서 head/mid/tail을 잘랐는가"를 명시적으로 보여준다:
#      [열1] V vs t (원본)      — head/plateau/tail 색상 밴드
#      [열2] V vs Q (V-Q curve) — 플래토 진입/이탈(q_entry/q_exit) 세로선 + 존 음영
#      [열3] |dV/dQ| vs Q       — θ_flat 임계선 + 플래토(|dV/dQ|<θ) 음영
#    행: 방전 / 충전
# ─────────────────────────────────────────────────────────────────────────────

# head=hi(급경사초입) / plateau=mid / tail=lo(급경사말단)
_VQZONE_COLOR = {"head": "#e74c3c", "plateau": "#2ecc71", "tail": "#3498db"}
_VQZONE_LABEL = {"head": "head (hi, 급경사초입)",
                 "plateau": "plateau (mid, 평탄)",
                 "tail": "tail (lo, 급경사말단)"}


def _zone_bounds_from_recs(recs: list) -> "tuple[float, float] | None":
    """세그먼트 meta에서 q_entry/q_exit(플래토 경계) 추출. 없으면 None."""
    for _, rec in recs:
        m = rec.meta or {}
        if "q_entry" in m and "q_exit" in m:
            return float(m["q_entry"]), float(m["q_exit"])
    return None


def _plot_vqzone_direction(axes_row, arrays, recs, theta_flat: float, title_prefix: str):
    """한 방향(방전/충전)의 3-패널 그리기. axes_row = [ax_vt, ax_vq, ax_dvdq]."""
    from common.scenario._curves import _build_vq_curve

    ax_vt, ax_vq, ax_dvdq = axes_row
    if arrays is None:
        for ax in axes_row:
            ax.text(0.5, 0.5, f"{title_prefix}\n데이터 부족", transform=ax.transAxes,
                    ha="center", va="center"); ax.axis("off")
        return

    v, i_mag, t, dt, q = arrays          # q = 원본 누적전하 (q_local)
    t_rel = t - t[0]
    qm, v_sm, dvdq_sm, q_tot = _build_vq_curve(v, i_mag, dt)
    rng = _zone_bounds_from_recs(recs)

    # 존 경계를 q_local 축과 시간축 인덱스로 변환
    if rng is not None:
        q_entry, q_exit = rng
        i_entry = int(np.searchsorted(q, q_entry))
        i_exit  = int(np.searchsorted(q, q_exit))
        i_entry = np.clip(i_entry, 0, len(t) - 1)
        i_exit  = np.clip(i_exit, i_entry, len(t) - 1)
        zone_spans = [("head", 0, i_entry), ("plateau", i_entry, i_exit),
                      ("tail", i_exit, len(t) - 1)]
    else:
        q_entry = q_exit = None
        zone_spans = []

    # ── 패널 1: V vs t (원본) + 존 밴드 ──────────────────────────────────────
    ax_vt.plot(t_rel, v, "k-", lw=1.0, alpha=0.7, zorder=3)
    for zname, a, b in zone_spans:
        if b > a:
            ax_vt.axvspan(t_rel[a], t_rel[b], color=_VQZONE_COLOR[zname], alpha=0.22,
                          label=_VQZONE_LABEL[zname])
    ax_vt.set_xlabel("time (s, 구간 시작 기준)"); ax_vt.set_ylabel("V [V]")
    ax_vt.set_title(f"{title_prefix} — V-t (원본) + 존 분리")
    ax_vt.grid(alpha=0.3)
    if zone_spans:
        ax_vt.legend(fontsize=7, loc="best")

    # ── 패널 2: V vs Q (V-Q curve) + 존 경계 ─────────────────────────────────
    ax_vq.plot(q, v, color="0.6", lw=0.8, alpha=0.5, label="원본 V-Q")
    fin = np.isfinite(qm) & np.isfinite(v_sm)
    if fin.any():
        ax_vq.plot(qm[fin], v_sm[fin], "b-", lw=1.8, label="V-Q (SG 스무딩)")
    for zname, a, b in zone_spans:
        if b > a:
            ax_vq.axvspan(q[a], q[b], color=_VQZONE_COLOR[zname], alpha=0.18)
    for xq, lbl in [(q_entry, "플래토 진입 q_entry"), (q_exit, "플래토 이탈 q_exit")]:
        if xq is not None:
            ax_vq.axvline(xq, color="k", ls="--", lw=1.2, alpha=0.8)
            ax_vq.text(xq, ax_vq.get_ylim()[1], lbl, fontsize=6.5, rotation=90,
                       va="top", ha="right", color="k")
    ax_vq.set_xlabel("Q [Ah] (누적)"); ax_vq.set_ylabel("V [V]")
    ax_vq.set_title(f"{title_prefix} — V-Q curve + 존 경계")
    ax_vq.grid(alpha=0.3); ax_vq.legend(fontsize=7, loc="best")

    # ── 패널 3: |dV/dQ| vs Q + θ_flat 임계선 + 플래토 음영 ────────────────────
    if fin.any():
        adv = np.abs(dvdq_sm)
        ax_dvdq.plot(qm[fin], adv[fin], color="#8e44ad", lw=1.6, label="|dV/dQ|")
        # 플래토 판정: |dV/dQ| < θ_flat
        plt_mask = fin & (adv < theta_flat)
        if plt_mask.any():
            ax_dvdq.scatter(qm[plt_mask], adv[plt_mask], s=18, color="#2ecc71",
                            zorder=5, label=f"플래토 (|dV/dQ|<{theta_flat:.2f})")
        ax_dvdq.axhline(theta_flat, color="r", ls="--", lw=1.2,
                        label=f"θ_flat = {theta_flat:.2f}")
        for xq in (q_entry, q_exit):
            if xq is not None:
                ax_dvdq.axvline(xq, color="k", ls="--", lw=1.0, alpha=0.7)
    ax_dvdq.set_xlabel("Q [Ah] (누적)"); ax_dvdq.set_ylabel("|dV/dQ| [V/Ah]")
    ax_dvdq.set_title(f"{title_prefix} — |dV/dQ| + 플래토 판정(θ_flat)")
    ax_dvdq.grid(alpha=0.3); ax_dvdq.legend(fontsize=7, loc="best")


def plot_vqslope_zones(pkl_path: Path, segmenter, spec_names: list,
                       cycle_id: int, out_path: Path):
    """vqslope 전용: 한 사이클의 head/plateau/tail 분리 근거를 3-패널×2방향으로 시각화."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    df_all  = raw.get("cycles")
    cell_id = raw.get("meta", {}).get("cell_id", pkl_path.stem)
    if df_all is None:
        print(f"  [경고] {pkl_path.name}: cycles 없음"); return
    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    valid_cycs = sorted(c for c in df_all["cycle"].unique() if c != 0)
    if not valid_cycs:
        print("  [경고] 유효 사이클 없음"); return
    if cycle_id == 0 or cycle_id not in valid_cycs:
        cycle_id = valid_cycs[len(valid_cycs) // 2]
        print(f"  → 사이클 미지정/무효 → 중간 사이클 {cycle_id} 사용")

    grp = df_all[df_all["cycle"] == cycle_id]
    dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
    chg = grp[grp["phase"] == "charge"].sort_values("time_s")
    _empty = np.empty(0, dtype=float)

    dis_arrays = dis_recs = None
    if len(dis) >= 30:
        v_d, i_d, t_d, dt_d, q_d = _build_arrays(dis)
        dis_arrays = (v_d, i_d, t_d, dt_d, q_d)
        dis_recs = [(rec.meta.get("seg_name") or spec_names[rec.scenario_id], rec)
                    for rec in segmenter.iter_segments(cell_id, cycle_id, v_d, i_d, dt_d, q_d)]
    chg_arrays = chg_recs = None
    if len(chg) >= 20:
        v_c, i_c, t_c, dt_c, q_c = _build_arrays(chg)
        chg_arrays = (v_c, i_c, t_c, dt_c, q_c)
        chg_recs = [(rec.meta.get("seg_name") or spec_names[rec.scenario_id], rec)
                    for rec in segmenter.iter_segments(
                        cell_id, cycle_id, _empty, _empty, _empty, _empty, v_c, i_c, dt_c, q_c)]

    theta_flat = float(getattr(segmenter, "theta_flat", 0.25))
    mode = getattr(segmenter, "mode", "dva")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _plot_vqzone_direction(axes[0], dis_arrays, dis_recs or [], theta_flat, "방전(discharge)")
    _plot_vqzone_direction(axes[1], chg_arrays, chg_recs or [], theta_flat, "충전(charge)")
    fig.suptitle(f"vqslope 존 분리 근거  |  {cell_id}  cycle={cycle_id}  mode={mode}  "
                 f"θ_flat={theta_flat:.2f}\n"
                 f"head=hi(급경사초입) / plateau=mid(평탄) / tail=lo(급경사말단)  "
                 f"— |dV/dQ|가 θ_flat 아래로 떨어지는 구간이 플래토",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  vqzone 플롯 저장: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 조건 비교 시각화  (--mode compare)
#    동일 셀·동일 사이클에 대해 여러 축/파라미터 조건을 세로로 쌓아 비교.
#    각 행은 plot_cycle_segments()의 [상단] V-t 밴드 패널과 동일한 그림이다.
# ─────────────────────────────────────────────────────────────────────────────

def _seg_len_summary(recs: list) -> dict[str, str]:
    """시나리오 이름 → 관측 포인트 수 요약 문자열.

    같은 시나리오 이름이 여러 세그먼트(n_samples>1)로 나오면 개수·범위를 함께 표시.
    이 사이클·이 조건에서 "실제로" 몇 포인트짜리 세그먼트가 만들어졌는지 보여준다 —
    n2(q_frac_wide) 같은 파라미터는 비율이라 사이클마다 실제 포인트 수가 달라지므로,
    여기서 계산하는 값이 곧 "이 그림에서의 정의된 구간 길이"다.
    """
    by_name: dict[str, list[int]] = {}
    for sn, rec in recs:
        by_name.setdefault(sn, []).append(len(rec.v))
    out: dict[str, str] = {}
    for sn, lens in by_name.items():
        if len(lens) == 1:
            out[sn] = f"n={lens[0]}"
        elif min(lens) == max(lens):
            out[sn] = f"n={lens[0]}×{len(lens)}"
        else:
            out[sn] = f"n={min(lens)}–{max(lens)}(×{len(lens)})"
    return out


# hi/lo는 서로 겹치지 않으므로(mid과만 겹침) 같은 방향을 공유해도 무방 — hi·mid, mid·lo
# 인접(겹침 가능) 쌍만 서로 다른 빗금 방향이면 충분하다.
_ZONE_HATCH = {"hi": "//", "mid": "\\\\", "lo": "//"}


def _qfw_zone_frac_bounds(n1: float) -> dict[str, tuple[float, float]]:
    """q_frac_wide 존 경계 (q_frac 비율, 0~1). hi=[0,n1] / mid=중앙 n1폭 / lo=[1-n1,1].
    n1 > 1/3이면 존끼리 겹칠 수 있다(설계상 의도된 동작 — SCENARIO_STRATEGY.md 참조)."""
    return {
        "hi":  (0.0, n1),
        "mid": (0.5 - n1 / 2, 0.5 + n1 / 2),
        "lo":  (1.0 - n1, 1.0),
    }


def _vqslope_zone_q_bounds(recs: list, q_tot: float) -> "dict[str, tuple[float, float]] | None":
    """vqslope 존 경계 (절대 q, Ah). recs 중 아무 하나의 meta에서 q_entry/q_exit을 꺼낸다
    (같은 방향의 모든 세그먼트가 같은 플래토 경계를 공유하므로 어느 것이든 상관없다)."""
    for _, rec in recs:
        m = rec.meta or {}
        if "q_entry" in m and "q_exit" in m:
            qe, qx = float(m["q_entry"]), float(m["q_exit"])
            return {"hi": (0.0, qe), "mid": (qe, qx), "lo": (qx, q_tot)}
    return None


def _zone_bounds_t(axis: str, axis_cfg: dict, direction: str,
                    phase_arrays, recs: list) -> "dict[str, tuple[float, float]]":
    """시나리오별 "존"(세그먼트를 뽑아내는 전체 구간) 경계를 시간(t0,t1)으로 계산.

    개별 세그먼트 길이(_seg_len_summary)와 달리, 존 경계는 n_samples/random_segment와
    무관하게 n1(q_frac_wide) 또는 플래토 검출(vqslope)만으로 고정된다 — "이 시나리오가
    원래 정의된 범위가 얼마나 넓은가"를 보여준다.
    """
    if phase_arrays is None:
        return {}
    _, _, t_p, _, q_p = phase_arrays
    if len(q_p) == 0:
        return {}
    q_tot = float(q_p[-1])

    if axis == "q_frac_wide":
        n1 = float(axis_cfg.get("n1", 0.4))
        frac_bounds = _qfw_zone_frac_bounds(n1)
        q_bounds = {z: (f0 * q_tot, f1 * q_tot) for z, (f0, f1) in frac_bounds.items()}
    elif axis == "vqslope":
        q_bounds = _vqslope_zone_q_bounds(recs, q_tot)
        if q_bounds is None:
            return {}
    else:
        return {}

    out: dict[str, tuple[float, float]] = {}
    for zone, (q0, q1) in q_bounds.items():
        t0, t1 = _t_range_from_q(t_p, q_p, np.array([q0, q1]))
        if t0 is not None:
            out[f"{direction}_{zone}"] = (t0, t1)
    return out


def _draw_vt_panel(ax, dis_arrays, dis_recs, chg_arrays, chg_recs, color_map, title="",
                    len_summary: "dict[str, str] | None" = None,
                    zone_bounds: "dict[str, tuple[float, float]] | None" = None):
    """V vs time_s 패널: 세그먼트 색상 밴드 오버레이 (plot_condition_comparison 전용).

    plot_cycle_segments()의 [상단] 패널과 같은 시각 문법(회색 배경 곡선 + 색상 밴드 +
    색상 오버레이 곡선)을 쓰되, 여러 조건을 한 figure에 여러 행으로 쌓을 수 있도록
    독립 함수로 분리했다. len_summary가 주어지면 범례에 시나리오별 관측 포인트 수를 덧붙이고,
    zone_bounds가 주어지면 "존" 전체 범위를 옅은 배경 밴드로 먼저 깔아 세그먼트 밴드(진한 색)와
    구분해서 보여준다 — 진한 밴드=실제 뽑힌 세그먼트, 옅은 밴드=시나리오가 정의된 전체 구간.
    """
    ax.set_facecolor("#f8f9fa")
    ax.set_xlabel("Time [s]", fontsize=9)
    ax.set_ylabel("Voltage [V]", fontsize=9)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=9, pad=4)

    for phase_arrays, fill_col, line_col, lbl in [
        (chg_arrays, "#d6eaf8", "#2980b9", "Charge"),
        (dis_arrays, "#fdf2e9", "#ca6f1e", "Discharge"),
    ]:
        if phase_arrays is None:
            continue
        v_bg, _, t_bg, _, _ = phase_arrays
        if len(t_bg) < 10:
            continue
        ax.fill_between(t_bg, v_bg.min() - 0.02, v_bg, color=fill_col, alpha=0.2, zorder=0)
        ax.plot(t_bg, v_bg, color="#cccccc", lw=0.8, zorder=1, alpha=0.9)
        ax.text((t_bg[0] + t_bg[-1]) / 2, 0.03, lbl, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=line_col, alpha=0.65)

    if (chg_arrays is not None and dis_arrays is not None
            and len(chg_arrays[2]) >= 10 and len(dis_arrays[2]) >= 10):
        ax.axvline(float(chg_arrays[2][-1]), color="#555555", lw=1.2, ls="--", zorder=4, alpha=0.6)

    # ── 존(시나리오 정의 구간) 배경 밴드 — 세그먼트 밴드보다 먼저(아래) 그림 ──────
    if zone_bounds:
        for sn, (t0, t1) in zone_bounds.items():
            c = color_map.get(sn, "black")
            zone_suffix = sn.rsplit("_", 1)[-1]
            hatch = _ZONE_HATCH.get(zone_suffix, "//")
            ax.axvspan(t0, t1, facecolor=c, edgecolor=c, alpha=0.13, zorder=1,
                       hatch=hatch, linewidth=0)
            ax.text((t0 + t1) / 2, 0.97, sn, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=5.5, color=c, alpha=0.85, rotation=0)

    legend_added: set = set()
    handles: list = []
    for phase_arrays, recs in [(dis_arrays, dis_recs), (chg_arrays, chg_recs)]:
        if phase_arrays is None:
            continue
        _, _, t_p, _, q_p = phase_arrays
        for sn, rec in recs:
            c = color_map.get(sn, "black")
            t0, t1 = _t_range_from_q(t_p, q_p, rec.q)
            if t0 is not None:
                ax.axvspan(t0, t1, color=c, alpha=0.35, zorder=2)
            idx0 = int(np.searchsorted(q_p, rec.q[0], "left"))
            idx1 = int(np.searchsorted(q_p, rec.q[-1], "right"))
            t_seg = t_p[idx0:idx1]
            n = min(len(t_seg), len(rec.v))
            ax.plot(t_seg[:n], rec.v[:n], color=c, lw=2.3, alpha=0.9, zorder=3)
            if sn not in legend_added:
                extra = f"  ({len_summary[sn]})" if len_summary and sn in len_summary else ""
                handles.append(mpatches.Patch(color=c, label=f"{sn}{extra}"))
                legend_added.add(sn)

    if handles:
        ax.legend(handles=handles, fontsize=6.5, loc="upper right", framealpha=0.85,
                  ncol=min(len(handles), 3))


def plot_condition_comparison(
    pkl_path: Path,
    conditions: list,          # [{"axis": str, "axis_config": dict, "label": str(optional)}]
    cycle_id: int,
    out_path: Path,
):
    """동일 셀·동일 사이클의 원시 데이터에 조건별 segmenter를 새로 실행해
    "같은 데이터가 조건별로 어떻게 잘리는지"를 세로로 쌓아 한 장에 비교한다.

    조건마다 사전 추출된 _4_data_hi/{axis}/{tag}/ 데이터를 읽는 대신, 원시 v/i/t/dt/q
    배열(한 번만 계산)에 대해 그 자리에서 segmenter.iter_segments()를 실행한다 —
    아직 추출되지 않은 조건도 즉시 비교 가능하고, 데이터 추출 파이프라인과 완전히 독립적이다.
    """
    from common.scenario import get_segmenter

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    df_all  = raw.get("cycles")
    cell_id = raw.get("meta", {}).get("cell_id", pkl_path.stem)
    if df_all is None:
        print(f"  [경고] {pkl_path.name}: cycles 없음"); return
    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    valid_cycs = sorted(c for c in df_all["cycle"].unique() if c != 0)
    if not valid_cycs:
        print("  [경고] 유효 사이클 없음"); return
    if cycle_id == 0 or cycle_id not in valid_cycs:
        cycle_id = valid_cycs[len(valid_cycs) // 2]
        print(f"  → 사이클 미지정/무효 → 중간 사이클 {cycle_id} 사용")

    grp = df_all[df_all["cycle"] == cycle_id]
    dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
    chg = grp[grp["phase"] == "charge"].sort_values("time_s")
    _empty = np.empty(0, dtype=float)

    dis_arrays = _build_arrays(dis) if len(dis) >= 30 else None
    chg_arrays = _build_arrays(chg) if len(chg) >= 20 else None
    if dis_arrays is None and chg_arrays is None:
        print(f"  [경고] {cell_id} cyc{cycle_id}: 충/방전 데이터 부족"); return

    n_cond = len(conditions)
    fig, axes = plt.subplots(n_cond, 1, figsize=(13, 3.0 * n_cond), squeeze=False)

    for row, cond in enumerate(conditions):
        ax = axes[row][0]
        axis     = cond["axis"]
        axis_cfg = cond.get("axis_config", {})
        label    = cond.get("label") or _condition_tag(axis, axis_cfg)

        try:
            seg = get_segmenter(axis, {axis: axis_cfg})
        except Exception as e:
            ax.text(0.5, 0.5, f"[{label}]\nsegmenter 로드 실패: {e}",
                    transform=ax.transAxes, ha="center", va="center", color="red", fontsize=9)
            ax.axis("off")
            print(f"  [경고] 조건 '{label}' segmenter 로드 실패: {e}")
            continue

        names     = seg.get_spec().scenario_names
        color_map = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(names)}

        dis_recs = []
        if dis_arrays is not None:
            v_d, i_d, t_d, dt_d, q_d = dis_arrays
            dis_recs = [(rec.meta.get("seg_name") or names[rec.scenario_id], rec)
                        for rec in seg.iter_segments(cell_id, cycle_id, v_d, i_d, dt_d, q_d)]
        chg_recs = []
        if chg_arrays is not None:
            v_c, i_c, t_c, dt_c, q_c = chg_arrays
            chg_recs = [(rec.meta.get("seg_name") or names[rec.scenario_id], rec)
                        for rec in seg.iter_segments(
                            cell_id, cycle_id, _empty, _empty, _empty, _empty, v_c, i_c, dt_c, q_c)]

        n_total = len(dis_recs) + len(chg_recs)
        len_summary = _seg_len_summary(dis_recs + chg_recs)
        zone_bounds = {
            **_zone_bounds_t(axis, axis_cfg, "dis", dis_arrays, dis_recs),
            **_zone_bounds_t(axis, axis_cfg, "chg", chg_arrays, chg_recs),
        }
        _draw_vt_panel(
            ax, dis_arrays, dis_recs, chg_arrays, chg_recs, color_map,
            title=f"[{row}] {label}   (세그먼트 {n_total}개: dis={len(dis_recs)}, chg={len(chg_recs)})",
            len_summary=len_summary,
            zone_bounds=zone_bounds,
        )

    fig.suptitle(f"세그멘테이션 조건 비교  |  {cell_id}  cycle={cycle_id}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  조건 비교 플롯 저장: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def _discover_axes() -> list[str]:
    """4_hi_analysis/hi_features*.pkl 파일을 스캔해 축 이름 목록 반환.

    hi_features.pkl        → "qfrac"
    hi_features_rcs.pkl    → "rcs"
    hi_features_protocol.pkl → "protocol"
    hi_features_vwindow.pkl  → "vwindow"
    """
    axes = []
    for pkl in sorted(PKL_CACHE_ROOT.glob("hi_features*.pkl")):
        suffix = pkl.stem.replace("hi_features", "").lstrip("_")
        axes.append(suffix if suffix else "qfrac")
    return axes


def _run_for_axis(axis: str, axis_cfg: dict, args) -> None:
    from common.scenario import get_segmenter

    try:
        seg = get_segmenter(axis, {axis: axis_cfg})
    except Exception as e:
        print(f"[경고] 축 '{axis}' segmenter 로드 실패: {e}")
        return

    spec  = seg.get_spec()
    names = spec.scenario_names
    print(f"\n축: {axis}  |  시나리오 ({len(names)}개): {names}")

    ds      = args.dataset.upper()
    root    = MIT_DIR if ds == "MIT" else HUST_DIR
    if axis == "q_frac_wide":
        dir_name = (f"q_frac_wide_n1-{int(round(seg.n1*100))}%"
                    f"_n2-{int(round(seg.n2*100))}%_N-{seg.n_samples}")
    elif axis == "q_abs":
        # seg(실제 생성된 segmenter)의 해석된 값을 쓴다 — q_frac_wide와 같은 패턴으로,
        # axis_cfg에 값이 없을 때의 기본값이 _qabs_tag와 어긋날 위험을 없앤다.
        adaptive_cfg = {"adaptive_samples": seg.adaptive_samples, "max_overlap": seg.max_overlap}
        dir_name = (
            f"q_abs_ms-{int(round(seg.mid_start*100))}%"
            f"_me-{int(round(seg.mid_end*100))}%"
            f"_sl-{int(round(seg.seg_len*100))}%_N-{seg.n_samples}"
            f"{_adaptive_suffix_tag(adaptive_cfg)}"
        )
    else:
        dir_name = axis
    out_dir = PKL_CACHE_ROOT / "outputs" / "seg_diagnose" / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 통계 ──────────────────────────────────────────────────────────────────
    if not args.no_stats:
        print(f"\n=== 통계 수집: {root} ===")
        stats      = collect_stats(root, seg, names)
        stats_path = out_dir / f"stats_{ds}.txt"
        print_stats(stats, names, axis, ds, out_path=stats_path)

    # ── 사이클 플롯 ───────────────────────────────────────────────────────────
    if not args.no_plot:
        pkls = sorted(root.glob("*.pkl"))
        if not pkls:
            print(f"[경고] {root} 에 pkl 파일 없음 — 플롯 생략")
            return

        rng = np.random.default_rng(args.seed)

        if args.cell:
            # ── 지정 셀: 기존 동작 (단일 셀, 단일 사이클) ──────────────────
            cell_pkl = root / f"{args.cell}.pkl"
            if not cell_pkl.exists():
                print(f"[경고] {cell_pkl} 없음 — 첫 번째 셀로 대체")
                cell_pkl = pkls[0]
            cell_stem = cell_pkl.stem

            if args.mode in ("segment", "all"):
                out_path = out_dir / f"{ds}_{cell_stem}_cyc{args.cycle or 'auto'}.png"
                print(f"\n=== 사이클 시각화 (segment): {cell_stem}  cycle={args.cycle or 'auto'} ===")
                plot_cycle_segments(cell_pkl, seg, names, args.cycle, axis, out_path)

            if args.mode in ("ic", "all"):
                ic_path = out_dir / f"{ds}_{cell_stem}_ic.png"
                print(f"\n=== IC 커브 시각화: {cell_stem}  n_cycles={args.n_cycles} ===")
                plot_ic_windows(cell_pkl, seg, names, axis, ic_path,
                                n_cycles=args.n_cycles)

            if args.mode in ("vqzone", "all"):
                if axis != "vqslope":
                    print("  [경고] --mode vqzone은 --seg-axis vqslope 전용입니다. 건너뜀.")
                else:
                    vz_path = out_dir / f"{ds}_{cell_stem}_vqzone_cyc{args.cycle or 'auto'}.png"
                    print(f"\n=== vqzone 시각화: {cell_stem}  cycle={args.cycle or 'auto'} ===")
                    plot_vqslope_zones(cell_pkl, seg, names, args.cycle, vz_path)

        else:
            # ── 미지정: 랜덤 N셀 × 랜덤 사이클 ────────────────────────────
            n_rand = min(args.n_random, len(pkls))
            sampled = rng.choice(len(pkls), size=n_rand, replace=False)
            sampled_pkls = [pkls[i] for i in sorted(sampled)]

            if args.mode in ("segment", "all"):
                print(f"\n=== 사이클 시각화 (segment): 랜덤 {n_rand}셀 × 랜덤 사이클 "
                      f"(seed={args.seed}) ===")
                for cell_pkl in sampled_pkls:
                    try:
                        with open(cell_pkl, "rb") as fh:
                            raw_meta = pickle.load(fh)
                        df_tmp = raw_meta.get("cycles")
                        if df_tmp is None:
                            continue
                        if "phase" not in df_tmp.columns:
                            df_tmp = _add_phase(df_tmp)
                        dis_tmp = df_tmp[df_tmp["phase"] == "discharge"]
                        valid_cycs = [
                            int(c) for c in df_tmp["cycle"].unique()
                            if c != 0
                            and int((dis_tmp["cycle"] == c).sum()) >= 30
                        ]
                        if not valid_cycs:
                            continue
                        chosen_cyc = int(rng.choice(valid_cycs))
                    except Exception as e:
                        print(f"  [건너뜀] {cell_pkl.stem}: {e}")
                        continue

                    cell_stem = cell_pkl.stem
                    out_path  = out_dir / f"{ds}_{cell_stem}_cyc{chosen_cyc}.png"
                    print(f"  {cell_stem}  cycle={chosen_cyc}")
                    plot_cycle_segments(cell_pkl, seg, names, chosen_cyc, axis, out_path)

            if args.mode in ("ic", "all"):
                # IC 모드: 랜덤 셀 중 첫 번째 사용
                cell_pkl  = sampled_pkls[0]
                cell_stem = cell_pkl.stem
                ic_path   = out_dir / f"{ds}_{cell_stem}_ic.png"
                print(f"\n=== IC 커브 시각화: {cell_stem}  n_cycles={args.n_cycles} ===")
                plot_ic_windows(cell_pkl, seg, names, axis, ic_path,
                                n_cycles=args.n_cycles)

            if args.mode in ("vqzone", "all"):
                if axis != "vqslope":
                    print("  [경고] --mode vqzone은 --seg-axis vqslope 전용입니다. 건너뜀.")
                else:
                    print(f"\n=== vqzone 시각화: 랜덤 {n_rand}셀 × 랜덤 사이클 (seed={args.seed}) ===")
                    for cell_pkl in sampled_pkls:
                        try:
                            with open(cell_pkl, "rb") as fh:
                                raw_meta = pickle.load(fh)
                            df_tmp = raw_meta.get("cycles")
                            if df_tmp is None:
                                continue
                            if "phase" not in df_tmp.columns:
                                df_tmp = _add_phase(df_tmp)
                            dis_tmp = df_tmp[df_tmp["phase"] == "discharge"]
                            valid_cycs = [int(c) for c in df_tmp["cycle"].unique()
                                          if c != 0 and int((dis_tmp["cycle"] == c).sum()) >= 30]
                            if not valid_cycs:
                                continue
                            chosen_cyc = int(rng.choice(valid_cycs))
                        except Exception as e:
                            print(f"  [건너뜀] {cell_pkl.stem}: {e}"); continue
                        cell_stem = cell_pkl.stem
                        vz_path = out_dir / f"{ds}_{cell_stem}_vqzone_cyc{chosen_cyc}.png"
                        print(f"  {cell_stem}  cycle={chosen_cyc}")
                        plot_vqslope_zones(cell_pkl, seg, names, chosen_cyc, vz_path)


def _run_qfracwide_survival(axis_cfg: dict, n_workers: int = 1) -> None:
    """q_frac_wide 전용: MIT+HUST 각각 세그먼터를 만들어 생존율/시간길이/전압길이 통계 수집.

    n_workers > 1 이면 collect_qfracwide_stats_parallel(ProcessPoolExecutor)로 셀 단위 병렬 처리.
    """
    from common.scenario import get_segmenter

    n1 = axis_cfg.get("n1", 0.4)
    n2 = axis_cfg.get("n2", 0.2)
    n_samples = axis_cfg.get("n_samples", 4)

    # spec_names는 segmenter 없이도 결정 가능 (q_frac_wide는 항상 고정 6개)
    _tmp_seg = get_segmenter("q_frac_wide", {"q_frac_wide": axis_cfg})
    names = _tmp_seg.get_spec().scenario_names

    per_ds_stats = {}
    per_ds_seg = {}
    for ds, root in (("MIT", MIT_DIR), ("HUST", HUST_DIR)):
        print(f"\n=== [{ds}] 생존율/길이 통계 수집: {root}  (workers={n_workers}) ===")
        if n_workers <= 1:
            seg = get_segmenter("q_frac_wide", {"q_frac_wide": axis_cfg})
            per_ds_stats[ds] = collect_qfracwide_stats(root, seg, names)
            per_ds_seg[ds] = seg
        else:
            stats, att, yld, cnp = collect_qfracwide_stats_parallel(
                root, axis_cfg, names, n_workers=n_workers)
            per_ds_stats[ds] = stats
            per_ds_seg[ds] = _CounterProxy(att, yld, cnp)

    out_dir = PKL_CACHE_ROOT / "outputs" / "seg_diagnose" / "q_frac_wide"
    tag = f"n1-{int(round(n1*100))}%_n2-{int(round(n2*100))}%_N-{n_samples}"

    txt_path = out_dir / f"survival_stats_{tag}.txt"
    print_qfracwide_stats(per_ds_stats, per_ds_seg, names, n1, n2, n_samples, out_path=txt_path)

    plot_path = out_dir / f"survival_stats_{tag}.png"
    plot_qfracwide_stats(per_ds_stats, per_ds_seg, names, n1, n2, n_samples, out_path=plot_path)

    # min_pts 스윕 — candidate_n_points 분포로 재스캔 없이 5/6/8/10 등 임계값별 생존율 계산
    sweep_txt_path = out_dir / f"minpts_sweep_{tag}.txt"
    print_min_pts_sweep(per_ds_seg, names, [5, 6, 8, 10], n1, n2, n_samples, out_path=sweep_txt_path)

    sweep_plot_path = out_dir / f"minpts_sweep_{tag}.png"
    plot_min_pts_sweep(per_ds_seg, names, n1, n2, n_samples, out_path=sweep_plot_path)

    # 원시 통계 저장 (dur_s/v_range 리스트 + attempted/yielded 카운트 + 포인트수 분포)
    # → 재스캔 없이 다른 n1/n2 run과 비교하거나, 임의의 min_pts 값을 추가로 조사할 때 재사용 가능
    raw_path = out_dir / f"survival_stats_{tag}.pkl"
    raw_dump = {
        "n1": n1, "n2": n2, "n_samples": n_samples,
        "spec_names": names,
        "stats": per_ds_stats,
        "counters": {ds: {"n_attempted": dict(seg.n_attempted),
                          "n_yielded": dict(seg.n_yielded),
                          "candidate_n_points": dict(seg.candidate_n_points)}
                    for ds, seg in per_ds_seg.items()},
    }
    with open(raw_path, "wb") as f:
        pickle.dump(raw_dump, f)
    print(f"  원시 통계 저장(재사용용): {raw_path}")


def _run_compare(args) -> None:
    """--mode compare 진입점: JSON 설정 파일의 조건 목록을 읽어 plot_condition_comparison 호출."""
    cfg_path = Path(args.compare_config) if args.compare_config else (STEP_DIR / "compare_conditions.json")
    if not cfg_path.exists():
        print(f"[ERROR] 비교 조건 설정 파일 없음: {cfg_path}")
        print("  --compare-config PATH 로 조건 리스트 JSON 파일을 지정하거나,")
        print(f"  {STEP_DIR / 'compare_conditions.json'} 을 만들어 두세요.")
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        conditions = json.load(f)
    if not conditions:
        print(f"[ERROR] {cfg_path}: 조건이 비어 있음")
        return

    if not args.cell:
        print("[ERROR] --mode compare 는 --cell 지정이 필수입니다 (비교 대상 셀).")
        return

    ds       = args.dataset.upper()
    root     = MIT_DIR if ds == "MIT" else HUST_DIR
    cell_pkl = root / f"{args.cell}.pkl"
    if not cell_pkl.exists():
        print(f"[ERROR] {cell_pkl} 없음")
        return

    print(f"\n=== 조건 비교: {cell_pkl.stem}  cycle={args.cycle or 'auto'}  "
          f"(조건 {len(conditions)}개, 설정파일={cfg_path}) ===")
    for c in conditions:
        print(f"  - {c.get('label') or _condition_tag(c['axis'], c.get('axis_config', {}))}")

    # 설정파일 이름을 파일명에 포함 — compare-config를 바꿔가며 같은 셀/사이클을 여러 번
    # 비교할 때(예: exp1_n2 vs exp2_noise vs exp6_minpts) 서로 덮어쓰지 않도록 한다.
    _cfg_tag = cfg_path.stem
    if _cfg_tag.startswith("compare_"):
        _cfg_tag = _cfg_tag[len("compare_"):]
    _cfg_sfx = f"_{_cfg_tag}" if _cfg_tag and _cfg_tag != "conditions" else ""

    out_dir  = PKL_CACHE_ROOT / "outputs" / "seg_diagnose" / "compare"
    out_path = out_dir / f"{ds}_{cell_pkl.stem}_cyc{args.cycle or 'auto'}{_cfg_sfx}_compare.png"
    plot_condition_comparison(cell_pkl, conditions, args.cycle, out_path)


def _resolve_seg_dir(axis: str, axis_cfg: dict, dataset: str) -> Path:
    """hi_correlation.py와 동일한 태그 규칙으로 실제 저장된 seg pkl 디렉터리를 계산.

    _qfw_tag/_qfref_tag/_qabs_tag/_vqslope_tag를 새로 베끼지 않고 hi_correlation.py에서
    직접 import해서 쓴다 — 태그 규칙이 바뀌면 자동으로 같이 따라감(단일 소스 유지).
    """
    if str(STEP_DIR) not in sys.path:
        sys.path.insert(0, str(STEP_DIR))
    import hi_correlation as _hc

    if axis == "q_frac_wide":
        axis_dir = f"q_frac_wide/{_hc._qfw_tag(axis_cfg)}"
    elif axis == "q_frac_ref":
        axis_dir = f"q_frac_ref/{_hc._qfref_tag(axis_cfg)}"
    elif axis == "q_abs":
        axis_dir = f"q_abs/{_hc._qabs_tag(axis_cfg)}"
    elif axis == "vqslope":
        axis_dir = f"vqslope/{_hc._vqslope_tag(axis_cfg)}"
    else:
        axis_dir = axis
    return DATA_4_HI_ROOT / axis_dir / "seg" / dataset.upper()


def _recompute_hi(hc_module, rec, seg_name: str, hi_name: str) -> float:
    """rec(SegmentRecord)의 원시 v/i/dt/q로 hi_name 하나를 즉석 재계산.

    hi_correlation.py가 추출 시점에 쓰는 것과 동일한 함수(_seg_stat/_seg_diff/_seg_lfp —
    실제 계산 로직의 단일 소스는 5_model/hi_compute.py, hi_correlation.py는 그걸 그대로
    재노출)를 호출하므로 "저장 로직과 다른 별도 재구현"이 아니라 "그 세그먼트를 지금 다시
    계산하면 뭐가 나오는가"를 정직하게 확인한다."""
    merged: dict = {}
    merged.update(hc_module._strip_seg_suffix(hc_module._seg_stat(rec.v, rec.i, rec.dt, rec.q, seg_name), seg_name))
    merged.update(hc_module._strip_seg_suffix(hc_module._seg_diff(rec.v, rec.i, rec.dt, rec.q, seg_name), seg_name))
    merged.update(hc_module._strip_seg_suffix(hc_module._seg_lfp(rec.v, rec.i, rec.dt, rec.q, seg_name), seg_name))
    return float(merged.get(hi_name, float("nan")))


def plot_hi_overwrite_check(
    cell_pkl: Path,
    seg_df_cycle: "pd.DataFrame",
    segmenter,
    spec_names: list,
    cell_id: str,
    cycle_id: int,
    axis: str,
    out_path: Path,
    hi_name: str = "diff_dqdv_area",
) -> None:
    """한 (cell, cycle)에서 뽑힌 세그먼트 전부(예: 24개)가 seg pkl에 개별 행으로, 각자
    고유한 HI 값으로 저장됐는지 — "덮어쓰기 없음"을 눈으로 바로 확인하는 플랏
    (docs/260816_RESULTS.md §2-6 Part D).

    상단(좌: 방전 V-Q, 우: 충전 V-Q): 세그먼트 각각을 zone(색조)×zone 내 샘플순번(명도)으로
    구분해 표시 — 같은 zone이라도 서로 다른 밝기의 곡선 n_samples개가 보여야 정상(예전
    버그였다면 zone당 곡선이 1개만 있었을 것).
    하단(막대): x축 = 세그먼트(zone#샘플순번), 막대 높이 = seg pkl에 저장된 hi_name 값,
    검은 ◆ = 그 세그먼트의 원시 v/i/dt/q로 지금 즉석 재계산한 값. zone 내에서도 막대
    높이가 서로 달라야(=덮어쓰기로 전부 같은 값이 아님) 하고, 막대와 ◆가 정확히
    겹쳐야(=저장된 값이 바로 그 세그먼트의 값) 한다.
    """
    if str(STEP_DIR) not in sys.path:
        sys.path.insert(0, str(STEP_DIR))
    import hi_correlation as _hc

    with open(cell_pkl, "rb") as f:
        raw = pickle.load(f)
    df_all = raw.get("cycles")
    if df_all is None:
        print(f"  [경고] {cell_pkl.name}: cycles 없음 — HI 덮어쓰기 검증 플롯 스킵")
        return
    if "phase" not in df_all.columns:
        df_all = _add_phase(df_all)

    grp = df_all[df_all["cycle"] == cycle_id]
    dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
    chg = grp[grp["phase"] == "charge"].sort_values("time_s")
    _empty = np.empty(0, dtype=float)

    dis_arrays = _build_arrays(dis) if len(dis) >= 30 else None
    chg_arrays = _build_arrays(chg) if len(chg) >= 20 else None
    if dis_arrays is None and chg_arrays is None:
        print(f"  [경고] {cell_id} cyc{cycle_id}: 충/방전 데이터 부족 — HI 덮어쓰기 검증 플롯 스킵")
        return

    v_d, i_d, t_d, dt_d, q_d = dis_arrays if dis_arrays is not None else (_empty,) * 5
    v_c, i_c, t_c, dt_c, q_c = chg_arrays if chg_arrays is not None else (_empty,) * 5

    # 원본 추출(hi_correlation._extract_one_cell)과 동일한 순서로 재생성:
    # discharge 세그먼트 전부 → charge 세그먼트 전부. non-random(격자) 모드는 결정론적이라
    # 같은 axis_config로 다시 부르면 위치까지 완전히 동일한 세그먼트가 나온다.
    recs = list(segmenter.iter_segments(cell_id, cycle_id, v_d, i_d, dt_d, q_d, v_c, i_c, dt_c, q_c))
    if not recs:
        print(f"  [경고] {cell_id} cyc{cycle_id}: 세그먼트 0개 — HI 덮어쓰기 검증 플롯 스킵")
        return

    n_rows = len(seg_df_cycle)
    n = min(len(recs), n_rows)
    if len(recs) != n_rows:
        print(f"  [경고] 재생성 세그먼트 수({len(recs)}) != 저장된 행 수({n_rows}) — "
              f"앞 {n}개만 대조합니다(축/min_pts 설정이 캐시와 다를 수 있음).")

    # ── zone별 샘플 순번 부여 + 저장값/재계산값 매칭 ──────────────────────────
    zone_next_idx: dict = {}
    zone_of, sample_idx_of = [], []
    labels, stored_vals, recomp_vals = [], [], []
    color_map = {sn: _PALETTE[i % len(_PALETTE)] for i, sn in enumerate(spec_names)}

    for i in range(n):
        rec = recs[i]
        zone = rec.meta.get("seg_name") or spec_names[rec.scenario_id]
        s_idx = zone_next_idx.get(zone, 0)
        zone_next_idx[zone] = s_idx + 1
        zone_of.append(zone)
        sample_idx_of.append(s_idx)

        row = seg_df_cycle.iloc[i]
        stored_vals.append(float(row.get(hi_name, np.nan)))
        recomp_vals.append(_recompute_hi(_hc, rec, zone, hi_name))
        labels.append(f"{zone}\n#{s_idx}")

    bar_colors = [_sample_shade(color_map.get(zone_of[i], "#333333"), sample_idx_of[i],
                                 zone_next_idx[zone_of[i]]) for i in range(n)]

    diffs = np.abs(np.array(stored_vals) - np.array(recomp_vals))
    max_abs_diff = float(np.nanmax(diffs)) if n else float("nan")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(12, n * 0.55), 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.0], hspace=0.55, wspace=0.25)
    ax_dq  = fig.add_subplot(gs[0, 0])
    ax_cq  = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    ax_dq.set_title("방전  V-Q  (zone=색조, zone 내 샘플순번=명도)", fontsize=9)
    ax_dq.set_xlabel("Q_cum [Ah]", fontsize=8); ax_dq.set_ylabel("Voltage [V]", fontsize=8)
    if dis_arrays is not None:
        ax_dq.plot(q_d, v_d, color="#cccccc", lw=1.0, zorder=1)
    ax_cq.set_title("충전  V-Q  (zone=색조, zone 내 샘플순번=명도)", fontsize=9)
    ax_cq.set_xlabel("Q_cum [Ah]", fontsize=8); ax_cq.set_ylabel("Voltage [V]", fontsize=8)
    if chg_arrays is not None:
        ax_cq.plot(q_c, v_c, color="#cccccc", lw=1.0, zorder=1)

    for i in range(n):
        rec = recs[i]
        c = bar_colors[i]
        ax = ax_cq if rec.direction > 0 else ax_dq
        ax.plot(rec.q, rec.v, color=c, lw=2.4, alpha=0.95, zorder=2)
        ax.scatter([rec.q[0]], [rec.v[0]], color=c, s=30, zorder=4,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(str(sample_idx_of[i]), (rec.q[0], rec.v[0]), fontsize=6, color="black",
                    xytext=(2, 2), textcoords="offset points")

    # ── 하단: 세그먼트별 저장값(막대) + 재계산값(마커) ──────────────────────────
    x = np.arange(n)
    ax_bar.bar(x, stored_vals, color=bar_colors, edgecolor="black", linewidth=0.4,
               label="seg pkl에 저장된 값", zorder=2)
    ax_bar.scatter(x, recomp_vals, color="black", marker="D", s=26, zorder=3,
                   label="원시 v/i/dt/q에서 즉석 재계산한 값")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=6.5)
    ax_bar.set_ylabel(hi_name, fontsize=9)
    ax_bar.set_title(
        f"세그먼트 {n}개 개별 저장 검증 — 막대 높이가 zone 내에서도 서로 다르고(=덮어쓰기 없음), "
        f"막대와 ◆가 겹침(=그 세그먼트 고유 값이 정확히 보존됨)",
        fontsize=9,
    )
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.text(
        0.01, 0.97, f"max|저장값-재계산값| = {max_abs_diff:.3e}  (부동소수점 오차 수준 = 완전 일치)",
        transform=ax_bar.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="#fff9db", edgecolor="#e0c46c", alpha=0.9),
    )

    fig.suptitle(
        f"[{axis}]  {cell_id}  Cycle {cycle_id}  —  세그먼트당 HI 개별 저장 검증 (verify-fix Part D)",
        fontsize=12, fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  플롯 저장: {out_path}  (max|저장-재계산|={max_abs_diff:.3e})")


def _run_verify_fix(axis: str, axis_cfg: dict, args) -> None:
    """2026-08-16 세그먼트별-행 수정이 실제로 반영됐는지 검증 (docs/260816_RESULTS.md §2-6).

    A. 그룹당(cell,cycle,seg_name) 행 수 분포 — n_samples와 일치해야 함(예전엔 전부 1).
    C. 사이클 내 capacity_Ah 일관성 — 같은 사이클의 여러 행이 같은 타깃을 공유해야 함(std=0).
    둘 다 hi_correlation.py가 저장한 실제 seg pkl(_4_data_hi/{axis}/{tag}/seg/{dataset}/)을
    직접 읽는다 — Step4를 --force(-extract)로 재추출한 뒤 실행해야 의미 있는 결과가 나온다.
    """
    seg_dir = _resolve_seg_dir(axis, axis_cfg, args.dataset)
    pkls = sorted(seg_dir.glob("*.pkl"))
    if not pkls:
        print(f"[ERROR] {seg_dir} 에 seg pkl이 없습니다 — 먼저 Step4를 재추출하세요:")
        print(f"  python run_pipeline.py 4 --to-step 4 --force-extract --seg-axis {axis} "
              f"--axis-config '{json.dumps(axis_cfg)}' --workers 8")
        return

    print(f"=== seg pkl 로드: {seg_dir} ({len(pkls)}개 셀) ===")
    df = pd.concat([pd.read_pickle(p) for p in pkls], ignore_index=True)
    print(f"총 {len(df):,}행\n")

    print("=== A. 그룹당(cell,cycle,seg_name) 행 수 분포 ===")
    print("(예전 버그: 항상 1. 수정 후: n_samples와 일치해야 함 — 존 경계/min_pts 탈락으로")
    print(" 일부 소량 미달은 정상)")
    counts = df.groupby(["cell_id", "cycle", "seg_name"]).size()
    vc = counts.value_counts().sort_index()
    print(vc)

    print("\n=== C. 사이클 내 capacity_Ah(SOH 타깃) 일관성 ===")
    std_per_cycle = df.groupby(["cell_id", "cycle"])["capacity_Ah"].std()
    max_std = float(std_per_cycle.max()) if len(std_per_cycle) else float("nan")
    print(f"max std = {max_std:.3e}  (부동소수점 오차 이내여야 정상, 0이 아니면 캡용량 재대입 로직 확인 필요)")

    if not args.no_plot:
        out_dir = PKL_CACHE_ROOT / "outputs" / "seg_diagnose" / f"verify_fix_{axis}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([str(k) for k in vc.index], vc.values, color="#3498db")
        ax.set_xlabel("그룹(cell,cycle,seg_name)당 행 수")
        ax.set_ylabel("그룹 개수")
        ax.set_title(f"{axis} 세그먼트 집계 검증 — {args.dataset}\n(max capacity_Ah std={max_std:.1e})")
        out_path = out_dir / f"row_count_hist_{args.dataset}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n플롯 저장: {out_path}")

        # ── D. 세그먼트별 HI 개별 저장 검증 (덮어쓰기 없음을 값 수준에서 직접 확인) ──────
        print("\n=== D. 세그먼트별 HI 개별 저장 검증 (재계산 대조) ===")
        from common.scenario import get_segmenter
        segmenter = get_segmenter(axis, {axis: axis_cfg})
        spec_names = segmenter.get_spec().scenario_names

        cell_id, cycle_id = args.cell, args.cycle
        if not cell_id or not cycle_id:
            # "가장 흔한(=완전한) 행 수"를 모든 zone에서 다 채운 (cell,cycle)을 자동 선택
            # — 24개짜리 등 "정상적으로 꽉 찬" 사례를 보여주기 위함.
            full_n = int(vc.index[vc.values.argmax()])
            cnt_df = counts.reset_index(name="n")
            per_cyc = cnt_df.groupby(["cell_id", "cycle"]).agg(
                n_zones=("seg_name", "nunique"),
                all_full=("n", lambda s: bool((s == full_n).all())),
            )
            candidates = per_cyc[(per_cyc["n_zones"] == len(spec_names)) & (per_cyc["all_full"])]
            if len(candidates):
                cell_id, cycle_id = candidates.index[0]
            else:
                cell_id, cycle_id = df.iloc[0][["cell_id", "cycle"]]
            cell_id, cycle_id = str(cell_id), int(cycle_id)
            print(f"  자동 선택: cell={cell_id}  cycle={cycle_id}  (zone당 {full_n}개씩 꽉 찬 사례)")

        root = MIT_DIR if args.dataset.upper() == "MIT" else HUST_DIR
        cell_pkl = root / f"{cell_id}.pkl"
        if not cell_pkl.exists():
            print(f"  [경고] 원본 셀 pkl 없음: {cell_pkl} — Part D 스킵")
        else:
            seg_df_cycle = df[(df["cell_id"] == cell_id) & (df["cycle"] == cycle_id)]
            hi_out_path = out_dir / f"hi_overwrite_check_{args.dataset}_{cell_id}_cyc{cycle_id}.png"
            plot_hi_overwrite_check(
                cell_pkl, seg_df_cycle, segmenter, spec_names,
                cell_id, cycle_id, axis, hi_out_path,
            )


def main():
    parser = argparse.ArgumentParser(description="시나리오 세그먼트 진단 (통계 + 시각화)")
    parser.add_argument("--seg-axis",    type=str, default=None,
                        help="세그멘테이션 축 (미지정 시 hi_features*.pkl 자동 탐색)")
    parser.add_argument("--axis-config", type=str, default="{}",
                        help="축 파라미터 JSON 문자열 (예: '{\"assign\": \"none\"}')")
    parser.add_argument("--dataset",     type=str, default="MIT",
                        choices=["MIT", "HUST", "mit", "hust"],
                        help="통계 스캔 대상 데이터셋 (기본: MIT)")
    parser.add_argument("--cell",        type=str, default="",
                        help="사이클 플롯 대상 셀 ID (미지정 시 첫 번째 셀 사용)")
    parser.add_argument("--cycle",       type=int, default=0,
                        help="사이클 플롯 대상 사이클 번호 (0이면 첫 번째 유효 사이클)")
    parser.add_argument("--mode",         type=str, default="segment",
                        choices=["segment", "ic", "vqzone", "compare", "verify-fix", "all"],
                        help="시각화 모드: segment(기본)|ic|vqzone(vqslope 존 분리 근거)|"
                             "compare(여러 축/파라미터 조건 비교, --cell 필수)|"
                             "verify-fix(2026-08-16 세그먼트별-행 수정 검증, "
                             "docs/260816_RESULTS.md §2-6 A/C/D — D는 한 사이클의 세그먼트 "
                             "전부가 개별 HI 값으로 저장됐는지 재계산 대조로 확인)|all")
    parser.add_argument("--compare-config", type=str, default=None,
                        help="--mode compare 전용: 비교할 조건 목록 JSON 파일 경로 "
                             "(기본: 4_hi_analysis/compare_conditions.json). "
                             "형식: [{\"axis\":\"q_frac_wide\",\"axis_config\":{...},\"label\":\"(선택)\"}, ...]")
    parser.add_argument("--n-cycles",    type=int, default=6,
                        help="IC 모드에서 표시할 대표 사이클 수 (기본: 6)")
    parser.add_argument("--n-random",    type=int, default=10,
                        help="--cell 미지정 시 segment 시각화에 사용할 랜덤 셀 수 (기본: 10)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="랜덤 셀/사이클 선택 재현성 seed (기본: 42)")
    parser.add_argument("--no-stats",    action="store_true",
                        help="통계 수집·출력 생략")
    parser.add_argument("--no-plot",     action="store_true",
                        help="사이클 시각화 생략")
    parser.add_argument("--survival-stats", action="store_true",
                        help="q_frac_wide 전용: MIT+HUST 생존율/시간길이/전압길이 비교 통계. "
                             "--seg-axis q_frac_wide 필수. --n1/--n2/--n-samples로 파라미터 지정 "
                             "(--axis-config 로도 가능). --dataset/--no-plot 무시하고 시각화 생략, "
                             "MIT+HUST 둘 다 자동 수행.")
    parser.add_argument("--n1", type=float, default=None, help="q_frac_wide n1 (--survival-stats용)")
    parser.add_argument("--n2", type=float, default=None, help="q_frac_wide n2 (--survival-stats용)")
    parser.add_argument("--n-samples", type=int, default=None, dest="n_samples",
                        help="q_frac_wide n_samples (--survival-stats용)")
    args = parser.parse_args()

    if args.mode == "compare":
        _run_compare(args)
        print("\n완료!")
        return

    try:
        axis_cfg: dict = json.loads(args.axis_config)
    except json.JSONDecodeError as e:
        print(f"[ERROR] --axis-config JSON 파싱 실패: {e}")
        return

    if args.seg_axis == "q_frac_wide" or args.survival_stats:
        # 파일 상단 QFRAC_WIDE_AXIS_CONFIG를 기본값으로 쓰고, --axis-config로
        # 넘어온 값이 있으면 그것으로 덮어씀 (CLI 인자가 우선).
        axis_cfg = {**QFRAC_WIDE_AXIS_CONFIG, **axis_cfg}

    if args.mode == "verify-fix":
        if not args.seg_axis:
            print("[ERROR] --mode verify-fix 는 --seg-axis 지정이 필수입니다.")
            return
        _run_verify_fix(args.seg_axis, axis_cfg, args)
        print("\n완료!")
        return

    if args.survival_stats:
        if args.n1 is not None:
            axis_cfg["n1"] = args.n1
        if args.n2 is not None:
            axis_cfg["n2"] = args.n2
        if args.n_samples is not None:
            axis_cfg["n_samples"] = args.n_samples
        _run_qfracwide_survival(axis_cfg)
        print("\n완료!")
        return

    if args.seg_axis is not None:
        # ── 단일 축 모드 (기존 동작) ──────────────────────────────────────
        _run_for_axis(args.seg_axis, axis_cfg, args)
    else:
        # ── 자동 탐색 모드: hi_features*.pkl → 모든 축 순회 ──────────────
        axes = _discover_axes()
        if not axes:
            print("[ERROR] 4_hi_analysis/ 에서 hi_features*.pkl 파일을 찾을 수 없습니다.")
            print("  --seg-axis 로 축을 직접 지정하거나 hi_correlation.py를 먼저 실행하세요.")
            return
        print(f"\n[자동 발견] hi_features*.pkl → 축: {axes}")
        for axis in axes:
            print(f"\n{'=' * 60}")
            print(f"  처리 중: {axis}")
            print(f"{'=' * 60}")
            _run_for_axis(axis, axis_cfg, args)

    print("\n완료!")


if __name__ == "__main__":
    main()
