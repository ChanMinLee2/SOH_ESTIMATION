"""
tmp_make_test_rs.py — 완전 랜덤 구간·랜덤 길이 테스트 세그먼트 데이터셋 생성기.

[동작 방식]
  · hi_correlation.py 파이프라인을 우회해 직접 추출
  · 방향당 n_samples 개의 랜덤 세그먼트 (기본 8개, 총 ~16/사이클)
  · 시작점: q_frac ∈ [0, 1 - length]  균등분포 (완전 랜덤)
  · 길이   : q_frac ∈ [min_len, max_len] 균등분포 (겹침 허용)
  · 분류   : 충전(segment_id=0) / 방전(segment_id=1) 만 구분

[출력]
  _4_data_hi/test_rs/seg/{MIT,HUST}/{cell}.pkl   ← native long 형식 (행 = 세그먼트)
  _4_data_hi/test_rs/cycle/{MIT,HUST}/{cell}.pkl ← cycle 용량 (load_dataset_native_seg용)
  _4_data_hi/test_rs/scenario_spec.json
  _4_data_hi/test_rs/test_rs_stats.txt

[사용법]
  python tmp_make_test_rs.py
  python tmp_make_test_rs.py --n-samples 8 --min-len 0.05 --max-len 0.40 --workers 4
  python tmp_make_test_rs.py --force
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# HI 컬럼 순서 (segment_dataset._NATIVE_HI_COLS 와 동일해야 함)
_STAT_EXCL = {"q_abs", "energy_seg"}
_STAT_KEYS = [k for k in [
    "v_mean_cw", "v_std", "v_skew", "v_kurt", "v_ent",
    "i_mean", "i_std", "v_med", "corr_qi", "corr_vi",
    "q_abs", "energy_seg", "v_iqr", "v_range", "v_p10",
    "v_p90", "v_samp_ent", "corr_vt", "i_q_slope", "v_detrended_std",
] if k not in _STAT_EXCL]  # 18개
_DIFF_KEYS = [
    "dvdq_mean", "dvdq_std", "dvdq_max_abs", "dvdq_min", "dvdq_area",
    "dqdv_peak_h", "dqdv_peak_v", "dqdv_peak_w", "dqdv_area", "v_trend_slope",
    "dqdv_peak_asym", "d2vdq2_rms", "dvdq_skew", "dvdq_ent", "dv_di_seg",
    "dqdv_valley_h", "dqdv_valley_v", "dvdq_peak_q", "dvdq_flat_q", "dqdv_area_asym",
]  # 20개
_LFP_KEYS = [
    "plateau_frac", "plateau_v_mean", "plateau_v_std", "plateau_dvdq_std",
    "nonlin_idx", "v_dev_mid", "v_flatness", "delta_v_rms",
    "vq_slope_mid", "inflect_v", "inflect_q_frac", "v_concavity",
    "phase_entry_dvdq", "v_q_pearson", "ica_peak_cnt",
    "plateau_v_slope", "v_gradient_exit", "plateau_q_onset", "dv_dt_plateau", "v_ent_plateau",
]  # 20개
_MORPH_KEYS  = ["vt_dtw", "vq_dtw", "ve_dtw", "vt_frec", "vq_frec", "ve_frec"]  # 6개
_MORPH_GRID  = 50   # _seg_morph_curves 와 동일
_DTW_BAND    = 5    # Sakoe-Chiba band

_NATIVE_HI_COLS = (
    [f"stat_{k}" for k in _STAT_KEYS]
    + [f"diff_{k}" for k in _DIFF_KEYS]
    + [f"lfp_{k}"  for k in _LFP_KEYS]
    + [f"morph_{k}" for k in _MORPH_KEYS]
)  # 64 columns


def _load_ref_curves(pkl_path_str: str) -> dict:
    """clean PKL에서 cycle 1 (최초 유효) morph 기준 곡선 로드.

    Returns {"discharge": (vt, vq, ve), "charge": (vt, vq, ve)}
    각 tuple 원소는 _MORPH_GRID(=50)점 array 또는 None.
    """
    hic   = _WORKER_HIC
    none3 = (None, None, None)
    result = {"discharge": none3, "charge": none3}
    try:
        with open(pkl_path_str, "rb") as f:
            raw = pickle.load(f)
        df_all = raw.get("cycles")
        if df_all is None or not isinstance(df_all, pd.DataFrame):
            return result
        if "phase" not in df_all.columns:
            df_all = hic._add_phase(df_all)
        for direction, phase_name in [("discharge", "discharge"), ("charge", "charge")]:
            phase_df = df_all[df_all["phase"] == phase_name]
            for cyc in sorted(phase_df["cycle"].unique()):
                if int(cyc) == 0:
                    continue
                grp = phase_df[phase_df["cycle"] == cyc].sort_values("time_s")
                if len(grp) < 8:
                    continue
                vs  = grp["voltage_V"].values.astype(float)
                ims = np.abs(grp["current_A"].values.astype(float))
                t   = grp["time_s"].values.astype(float)
                dts = np.clip(np.diff(t, prepend=t[0]), 0, None)
                if phase_name == "charge":
                    cv    = _cv_start(vs, ims)
                    q_tmp = np.cumsum(ims * dts) / 3600.0
                    q_cc  = float(q_tmp[cv - 1]) if cv > 0 else 0.0
                    if cv < 8 or q_cc < 0.05:
                        cv = len(vs)
                    vs = vs[:cv]; ims = ims[:cv]; dts = dts[:cv]
                if len(vs) < 8:
                    continue
                curves = hic._seg_morph_curves(vs, ims, dts)
                if any(c is not None for c in curves):
                    result[direction] = curves
                    break
    except Exception:
        pass
    return result


def _compute_morph(
    v: np.ndarray, i: np.ndarray, dt: np.ndarray,
    ref_tuple: tuple, start_qf: float, end_qf: float,
) -> dict:
    """현재 세그먼트 vs cycle-1 기준 곡선 → morph 6개 계산.

    실패·fastdtw 없을 시 NaN dict 반환.
    """
    nan_dict = {f"morph_{mk}": float("nan") for mk in _MORPH_KEYS}
    if all(c is None for c in ref_tuple):
        return nan_dict
    try:
        from fastdtw import fastdtw as _fdt  # worker re-import 안정성
        hic = _WORKER_HIC
        if hic is None:
            return nan_dict
        seg_curves = hic._seg_morph_curves(v, i, dt)
        names = ["vt", "vq", "ve"]
        out = {}
        for name, seg_arr, ref_arr in zip(names, seg_curves, ref_tuple):
            if seg_arr is None or ref_arr is None:
                out[f"morph_{name}_dtw"]  = float("nan")
                out[f"morph_{name}_frec"] = float("nan")
                continue
            # ref에서 [start_qf, end_qf] 구간 추출 → MORPH_GRID점 재보간
            i_lo = max(0, int(start_qf * _MORPH_GRID))
            i_hi = min(_MORPH_GRID, max(i_lo + 2, int(end_qf * _MORPH_GRID)))
            ref_sub = ref_arr[i_lo:i_hi]
            if len(ref_sub) < 2:
                out[f"morph_{name}_dtw"]  = float("nan")
                out[f"morph_{name}_frec"] = float("nan")
                continue
            ref_interp = np.interp(
                np.linspace(0.0, 1.0, _MORPH_GRID),
                np.linspace(0.0, 1.0, len(ref_sub)),
                ref_sub,
            )
            dtw_dist, _ = _fdt(seg_arr, ref_interp, radius=_DTW_BAND)
            out[f"morph_{name}_dtw"]  = float(dtw_dist) / _MORPH_GRID
            out[f"morph_{name}_frec"] = float(np.max(np.abs(seg_arr - ref_interp)))
        return out
    except Exception:
        return nan_dict


def _cv_start(v: np.ndarray, i: np.ndarray,
              v_thresh: float = 3.60, cc_frac: float = 0.80) -> int:
    """CC→CV 전환 인덱스 (인덱스 이전까지가 CC 구간).

    전류 피크(argmax) 위치를 ramp-up 종료점으로 간주하고
    그 이후부터 CC→CV 전환을 탐색한다.
    MIT 등 ramp-up 프로토콜에서도 조기 오감지를 방지한다.
    """
    if len(i) == 0:
        return 0
    # ramp-up 종료 = 전류 피크 위치 (전반부 50% 내에서 탐색)
    half = max(1, len(i) // 2)
    peak_idx = int(np.argmax(i[:half]))
    i0 = float(i[peak_idx])
    if i0 <= 0:
        return len(i)
    # 피크 이후부터 CC→CV 전환 탐색
    for k in range(peak_idx, len(i)):
        if i[k] < cc_frac * i0 or v[k] > v_thresh:
            return k
    return len(i)


# ─────────────────────────────────────────────────────────────────────────────
# 멀티프로세싱 워커 (모듈 레벨 → pickling 안전)
# ─────────────────────────────────────────────────────────────────────────────

_WORKER_HIC = None  # 각 subprocess에서 _worker_init이 설정


def _worker_init(project_root_str: str) -> None:
    """Worker 초기화: hi_correlation을 일반 import로 로드 (프로세스당 1회)."""
    global _WORKER_HIC
    root = Path(project_root_str)
    for p in [str(root / "4_hi_analysis"), str(root)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    import hi_correlation as _hic  # 일반 import → sys.modules에 등록
    _WORKER_HIC = _hic


def _extract_segment_hi(v, i, dt, q) -> dict:
    """한 세그먼트에서 64 HI 추출 (morph는 NaN)."""
    hic = _WORKER_HIC
    SEG = "rs"
    raw: dict = {}
    try:
        raw.update(hic._seg_stat(v, i, dt, q, SEG))
    except Exception:
        pass
    try:
        raw.update(hic._seg_diff(v, i, dt, q, SEG))
    except Exception:
        pass
    try:
        raw.update(hic._seg_lfp(v, i, dt, q, SEG))
    except Exception:
        pass

    # suffix "_rs" 제거 (q_abs, energy_seg 포함해서 일단 전부 저장)
    stripped = {k[:-3]: v_val for k, v_val in raw.items() if k.endswith(f"_{SEG}")}

    # morph = NaN
    for mk in _MORPH_KEYS:
        stripped[f"morph_{mk}"] = float("nan")

    return stripped


def _extract_cell(args: tuple):
    """
    한 셀을 처리해 (cell_id, dataset, seg_rows, cycle_rows) 반환.

    seg_rows  : list[dict]  — 행 = 랜덤 세그먼트 하나
    cycle_rows: list[dict]  — 행 = cycle (cell_id, cycle, capacity_Ah)
    """
    pkl_path_str, n_samples, min_len, max_len, seed, min_pts = args
    path = Path(pkl_path_str)

    # 단일 프로세스 모드(--workers=1 없이도) 대비 — init 없이도 로드
    global _WORKER_HIC
    if _WORKER_HIC is None:
        root = path.parent.parent.parent
        for p in [str(root / "4_hi_analysis"), str(root)]:
            if p not in sys.path:
                sys.path.insert(0, p)
        import hi_correlation as _hic
        _WORKER_HIC = _hic

    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except Exception as e:
        return path.stem, "", [], []

    # cycle 1 기준 morph 곡선 (같은 clean PKL에서 로드)
    ref_curves = _load_ref_curves(pkl_path_str)

    meta   = raw.get("meta", {})
    df_all = raw.get("cycles")
    if df_all is None or not isinstance(df_all, pd.DataFrame):
        return path.stem, "", [], []

    cell_id = str(meta.get("cell_id", path.stem))
    dataset = str(meta.get("dataset", ""))

    if "phase" not in df_all.columns:
        try:
            df_all = _WORKER_HIC._add_phase(df_all)
        except Exception:
            return cell_id, dataset, [], []

    seg_rows:   list[dict] = []
    cycle_rows: list[dict] = []

    for cyc_val, grp in df_all.groupby("cycle"):
        cyc = int(cyc_val)
        if cyc == 0:
            continue

        cyc_rng = np.random.default_rng([seed, hash(cell_id) % (2 ** 31), cyc])

        dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
        chg = grp[grp["phase"] == "charge"].sort_values("time_s")

        # ── 방전 ─────────────────────────────────────────────────────────
        dis_cap = float("nan")
        if len(dis) >= min_pts and "capacity_Ah" in dis.columns:
            dis_cap = float(dis["capacity_Ah"].iloc[0])

        if len(dis) >= min_pts:
            v  = dis["voltage_V"].values.astype(float)
            i  = np.abs(dis["current_A"].values.astype(float))
            t  = dis["time_s"].values.astype(float)
            dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
            q  = np.cumsum(i * dt) / 3600.0
            q_tot = float(q[-1]) if len(q) > 0 else 0.0

            if q_tot >= 0.05:
                lengths = cyc_rng.uniform(min_len, max_len, size=n_samples)
                for length in lengths:
                    max_start = max(0.0, 1.0 - length)
                    start_qf  = float(cyc_rng.uniform(0.0, max_start))
                    end_qf    = min(1.0, start_qf + length)
                    lo, hi    = start_qf * q_tot, end_qf * q_tot
                    m = (q >= lo) & (q < hi)
                    if int(m.sum()) < min_pts:
                        continue
                    hi_dict = _extract_segment_hi(v[m], i[m], dt[m], q[m])
                    hi_dict.update(_compute_morph(
                        v[m], i[m], dt[m], ref_curves["discharge"], start_qf, end_qf,
                    ))
                    seg_rows.append({
                        "cell_id":    cell_id,
                        "dataset":    dataset,
                        "cycle":      cyc,
                        "segment_id": 1,         # discharge
                        "scen":       1,
                        "capacity_Ah": dis_cap,
                        "q_frac_lo":  start_qf,
                        "q_frac_hi":  end_qf,
                        **hi_dict,
                    })

        # ── 충전 (CC 구간만) ──────────────────────────────────────────────
        if len(chg) >= min_pts:
            vc  = chg["voltage_V"].values.astype(float)
            ic  = np.abs(chg["current_A"].values.astype(float))
            tc  = chg["time_s"].values.astype(float)
            dtc = np.clip(np.diff(tc, prepend=tc[0]), 0, None)
            qc  = np.cumsum(ic * dtc) / 3600.0

            cv = _cv_start(vc, ic)
            q_cc = float(qc[cv - 1]) if cv > 0 else 0.0
            if cv < min_pts or q_cc < 0.05:   # CC 짧거나 용량 부족 → 전체 충전
                cv = len(vc)
            vc = vc[:cv]; ic = ic[:cv]; dtc = dtc[:cv]; qc = qc[:cv]
            q_tot_c = float(qc[-1]) if len(qc) > 0 else 0.0

            if q_tot_c >= 0.05 and len(vc) >= min_pts:
                lengths_c = cyc_rng.uniform(min_len, max_len, size=n_samples)
                for length in lengths_c:
                    max_start = max(0.0, 1.0 - length)
                    start_qf  = float(cyc_rng.uniform(0.0, max_start))
                    end_qf    = min(1.0, start_qf + length)
                    lo, hi    = start_qf * q_tot_c, end_qf * q_tot_c
                    m = (qc >= lo) & (qc < hi)
                    if int(m.sum()) < min_pts:
                        continue
                    hi_dict = _extract_segment_hi(vc[m], ic[m], dtc[m], qc[m])
                    hi_dict.update(_compute_morph(
                        vc[m], ic[m], dtc[m], ref_curves["charge"], start_qf, end_qf,
                    ))
                    seg_rows.append({
                        "cell_id":    cell_id,
                        "dataset":    dataset,
                        "cycle":      cyc,
                        "segment_id": 0,         # charge
                        "scen":       0,
                        "capacity_Ah": dis_cap,  # cycle 용량 = 방전 기준
                        "q_frac_lo":  start_qf,
                        "q_frac_hi":  end_qf,
                        **hi_dict,
                    })

        # cycle 행 (capacity_Ah 로드에 사용)
        if np.isfinite(dis_cap):
            cycle_rows.append({
                "cell_id":    cell_id,
                "cycle":      cyc,
                "capacity_Ah": dis_cap,
            })

    return cell_id, dataset, seg_rows, cycle_rows


# ─────────────────────────────────────────────────────────────────────────────
# 통계 파일
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stats(out_dir: Path) -> str:
    seg_root = out_dir / "seg"
    all_dfs: list[pd.DataFrame] = []
    for ds_dir in sorted(seg_root.iterdir()) if seg_root.exists() else []:
        if not ds_dir.is_dir():
            continue
        for pkl_path in sorted(ds_dir.glob("*.pkl")):
            with open(pkl_path, "rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame):
                df["_ds"] = ds_dir.name
                all_dfs.append(df)

    if not all_dfs:
        return "(통계 계산 불가: seg PKL 없음)"

    combined = pd.concat(all_dfs, ignore_index=True)
    n_segs   = len(combined)
    n_cells  = combined["cell_id"].nunique()
    n_cycles = combined.groupby(["cell_id", "cycle"]).ngroups

    W = 62
    lines: list[str] = []

    def _h(t):    lines.append("=" * W); lines.append(f"  {t}"); lines.append("=" * W)
    def _r(l, v): lines.append(f"  {l:<38s} {v}")

    _h("test_rs 세그먼트 데이터셋 통계")
    lines.append("")
    lines.append("[기본 카운트]")
    _r("총 셀 수",       f"{n_cells:,}")
    _r("총 사이클 수",   f"{n_cycles:,}")
    _r("총 세그먼트 수", f"{n_segs:,}  ({n_segs/n_cycles:.1f}/사이클)")

    lines.append("")
    lines.append("[데이터셋별]")
    for ds in sorted(combined["_ds"].unique()):
        sub  = combined[combined["_ds"] == ds]
        nc   = sub["cell_id"].nunique()
        ncy  = sub.groupby(["cell_id", "cycle"]).ngroups
        ns   = len(sub)
        _r(ds, f"{nc} 셀 / {ncy:,} 사이클 / {ns:,} 세그먼트")

    lines.append("")
    lines.append("[시나리오 분포]")
    _r("segment_id", "카운트     비율")
    lines.append("  " + "-" * 42)
    _SCEN_NAME = {0: "chg (충전)", 1: "dis (방전)"}
    if "segment_id" in combined.columns:
        for sid, cnt in combined["segment_id"].value_counts().sort_index().items():
            name = _SCEN_NAME.get(int(sid), f"id={sid}")
            _r(f"  {name}", f"{cnt:>8,}  {100*cnt/n_segs:5.1f}%")

    lines.append("")
    lines.append("[세그먼트 길이 (q_frac)]")
    if "q_frac_lo" in combined.columns and "q_frac_hi" in combined.columns:
        seg_len = combined["q_frac_hi"] - combined["q_frac_lo"]
        _r("평균 길이",   f"{seg_len.mean():.3f}")
        _r("표준편차",    f"{seg_len.std():.3f}")
        _r("최소 / 최대", f"{seg_len.min():.3f} / {seg_len.max():.3f}")
        for pct in [10, 25, 50, 75, 90]:
            _r(f"  {pct}백분위수", f"{seg_len.quantile(pct/100):.3f}")

    lines.append("")
    lines.append("[cycle 총 용량 (capacity_Ah)]")
    if "capacity_Ah" in combined.columns:
        cap = combined.drop_duplicates(["cell_id", "cycle"])["capacity_Ah"].dropna()
        _r("평균",   f"{cap.mean():.4f} Ah")
        _r("std",    f"{cap.std():.4f} Ah")
        _r("min/max", f"{cap.min():.4f} / {cap.max():.4f} Ah")

    lines.append("")
    lines.append("[HI NaN 비율 (상위 10개)]")
    hi_cols = [c for c in _NATIVE_HI_COLS if c in combined.columns]
    if hi_cols:
        nan_rates = combined[hi_cols].isna().mean().sort_values(ascending=False)
        n_clean   = int((nan_rates == 0).sum())
        _r("전체 HI 수",    str(len(hi_cols)))
        _r("NaN 없는 HI 수", f"{n_clean}/{len(hi_cols)}")
        lines.append("")
        for col, rate in nan_rates.head(10).items():
            _r(f"  {col}", f"{rate*100:.1f}%")

    lines.append("")
    lines.append("=" * W)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 시각화 — 한 사이클에서 생성된 세그먼트 예시
# ─────────────────────────────────────────────────────────────────────────────

def _plot_demo_cycle(
    out_dir: Path,
    clean_dir: Path,
    n_samples: int,
    min_len: float,
    max_len: float,
    seed: int,
    min_pts: int,
    n_cells_per_ds: int = 3,
) -> None:
    """
    데이터셋별 n_cells_per_ds개 셀의 대표 사이클에서 랜덤 세그먼트를 시각화.

    레이아웃: n_cells_per_ds 행 × 4 열 (MIT 방전|충전, HUST 방전|충전).
    캐시 우선: seg PKL이 있으면 저장된 q_frac_lo/q_frac_hi 사용.
    폴백: 캐시에 없는 방향(충전 등)은 RNG로 세그먼트 생성.
    배경 V곡선은 항상 clean_dir에서 로드.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    # ── 0. hi_correlation import (phase 추가용) ──────────────────────────────
    _hic = None
    try:
        for _p in [str(PROJECT_ROOT / "4_hi_analysis"), str(PROJECT_ROOT)]:
            if _p not in sys.path:
                sys.path.insert(0, _p)
        import hi_correlation as _hic_mod
        _hic = _hic_mod
    except Exception:
        pass

    seg_root = out_dir / "seg"
    from_cache = seg_root.exists() and any(seg_root.rglob("*.pkl"))

    # ── 1. 데이터셋별 셀 선택 ─────────────────────────────────────────────────
    def _spread(pkls: list, n: int) -> list:
        """n개를 균등 분포로 선택 (첫/중/끝 방식)."""
        if len(pkls) <= n:
            return pkls
        step = (len(pkls) - 1) / (n - 1) if n > 1 else 0
        return [pkls[round(i * step)] for i in range(n)]

    DS_ORDER = ["MIT", "HUST"]
    # ds_name → list of (cell_id, seg_df|None, clean_pkl|None)
    ds_cells: dict[str, list] = {}

    for ds in DS_ORDER:
        ds_seg_dir   = seg_root / ds if seg_root.exists() else None
        ds_clean_dir = clean_dir / ds
        seg_pkls   = sorted(ds_seg_dir.glob("*.pkl"))   if (ds_seg_dir and ds_seg_dir.exists())  else []
        clean_pkls = sorted(ds_clean_dir.glob("*.pkl")) if ds_clean_dir.exists() else []
        ref_pkls   = seg_pkls if seg_pkls else clean_pkls
        if not ref_pkls:
            continue

        cells = []
        for pkl_path in _spread(ref_pkls, n_cells_per_ds):
            cell_id = pkl_path.stem
            seg_df  = None
            if seg_pkls:
                sp = ds_seg_dir / f"{cell_id}.pkl"
                if sp.exists():
                    try:
                        with open(sp, "rb") as f:
                            seg_df = pickle.load(f)
                        if not isinstance(seg_df, pd.DataFrame):
                            seg_df = None
                    except Exception:
                        seg_df = None
            cp = ds_clean_dir / f"{cell_id}.pkl"
            cells.append((cell_id, seg_df, cp if cp.exists() else None))
        if cells:
            ds_cells[ds] = cells

    if not ds_cells:
        print("[demo_plot] 사용 가능한 셀 없음 — 건너뜀")
        return

    # ── 2. 레이아웃 설정 ──────────────────────────────────────────────────────
    # 열: [DS0-방전 | DS0-충전 | DS1-방전 | DS1-충전]
    PANELS = [("방전", "discharge", 1), ("충전 CC", "charge", 0)]
    n_ds   = len(ds_cells)
    n_rows = max(len(v) for v in ds_cells.values())
    n_cols = n_ds * 2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols + 0.5, 4.2 * n_rows + 1.2),
        squeeze=False,
    )
    cmap = plt.colormaps.get_cmap("tab20")

    # 열별 헤더 레이블 (첫 행 위에 텍스트로 표시)
    ds_col_off: dict[str, int] = {ds: i * 2 for i, ds in enumerate(ds_cells)}

    # ── 3. 셀 × 패널 플롯 ────────────────────────────────────────────────────
    for ds, cell_list in ds_cells.items():
        col_off = ds_col_off[ds]

        for row_idx, (cell_id, seg_df, clean_pkl) in enumerate(cell_list):

            # clean PKL 로드
            df_all = None
            if clean_pkl:
                try:
                    with open(clean_pkl, "rb") as f:
                        raw = pickle.load(f)
                    df_all = raw.get("cycles") if isinstance(raw, dict) else None
                    if not isinstance(df_all, pd.DataFrame):
                        df_all = None
                except Exception:
                    pass

            if df_all is None:
                for pc in range(2):
                    ax = axes[row_idx, col_off + pc]
                    ax.text(0.5, 0.5, f"{cell_id}\n로드 실패",
                            transform=ax.transAxes, ha="center", va="center")
                    ax.axis("off")
                continue

            if "phase" not in df_all.columns and _hic is not None:
                try:
                    df_all = _hic._add_phase(df_all)
                except Exception:
                    pass

            # demo_cyc 결정: seg_df에서 양방향 모두 있는 중간 사이클 탐색
            demo_cyc = -1
            segs_by_dir: dict[int, list[tuple]] = {}

            if seg_df is not None and "cycle" in seg_df.columns:
                cycs = sorted(c for c in seg_df["cycle"].unique() if c > 0)
                # 25~75% 구간에서 양방향 공존 사이클 탐색
                search = cycs[len(cycs) // 4: 3 * len(cycs) // 4] or cycs
                for cyc in search:
                    sids = set(seg_df[seg_df["cycle"] == cyc]["segment_id"].unique())
                    if 0 in sids and 1 in sids:
                        demo_cyc = cyc
                        break
                if demo_cyc < 0 and cycs:
                    demo_cyc = cycs[len(cycs) // 2]

                if demo_cyc > 0:
                    cyc_df = seg_df[seg_df["cycle"] == demo_cyc]
                    for sid in cyc_df["segment_id"].unique():
                        rows = cyc_df[cyc_df["segment_id"] == sid]
                        segs_by_dir[int(sid)] = list(
                            zip(rows["q_frac_lo"].tolist(), rows["q_frac_hi"].tolist())
                        )

            if demo_cyc < 0 and "cycle" in df_all.columns:
                cycs = sorted(c for c in df_all["cycle"].unique() if c > 0)
                demo_cyc = cycs[len(cycs) // 2] if cycs else 1

            grp = (df_all[df_all["cycle"] == demo_cyc]
                   if demo_cyc > 0 and "cycle" in df_all.columns else pd.DataFrame())

            # 패널별 플롯
            for panel_idx, (label, phase_name, seg_id) in enumerate(PANELS):
                ax = axes[row_idx, col_off + panel_idx]

                pdf = (grp[grp["phase"] == phase_name].sort_values("time_s")
                       if "phase" in grp.columns and not grp.empty else pd.DataFrame())

                if len(pdf) < min_pts:
                    ax.text(0.5, 0.5, f"{label}\n데이터 부족",
                            transform=ax.transAxes, ha="center", va="center", fontsize=9)
                    ax.set_title(f"[{ds}] {cell_id}  {label}  cyc {demo_cyc}", fontsize=8)
                    ax.grid(True, alpha=0.3)
                    continue

                v  = pdf["voltage_V"].values.astype(float)
                i  = np.abs(pdf["current_A"].values.astype(float))
                t  = pdf["time_s"].values.astype(float)
                dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
                q  = np.cumsum(i * dt) / 3600.0

                actual_label = label
                if phase_name == "charge":
                    cv = _cv_start(v, i)
                    # CC 구간이 너무 짧거나 용량 부족 → 전체 충전 구간으로 폴백
                    q_cc = float(q[cv - 1]) if cv > 0 else 0.0
                    if cv < min_pts or q_cc < 0.05:
                        cv = len(v)
                        actual_label = "충전 전체"
                    v = v[:cv]; dt = dt[:cv]; q = q[:cv]

                q_tot = float(q[-1]) if len(q) > 0 else 0.0
                if q_tot < 0.05 or len(v) < min_pts:
                    ax.text(0.5, 0.5, "유효 구간 부족",
                            transform=ax.transAxes, ha="center", va="center", fontsize=9)
                    ax.set_title(f"[{ds}] {cell_id}  {actual_label}  cyc {demo_cyc}", fontsize=8)
                    ax.grid(True, alpha=0.3)
                    continue

                q_frac = q / q_tot
                ax.plot(q_frac, v, "k-", lw=1.2, alpha=0.35, zorder=1)

                # 세그먼트 범위: 캐시 → RNG 폴백
                if seg_id in segs_by_dir and segs_by_dir[seg_id]:
                    seg_ranges = segs_by_dir[seg_id]
                    src_tag = "캐시"
                else:
                    rng = np.random.default_rng(
                        [seed, hash(cell_id) % (2 ** 31), demo_cyc, panel_idx])
                    seg_ranges = []
                    for length in rng.uniform(min_len, max_len, size=n_samples):
                        max_s = max(0.0, 1.0 - length)
                        s = float(rng.uniform(0.0, max_s))
                        e = min(1.0, s + length)
                        if int(((q >= s * q_tot) & (q < e * q_tot)).sum()) >= min_pts:
                            seg_ranges.append((s, e))
                    src_tag = "RNG"

                # 유효 세그먼트 필터링 (음영 범위 계산용)
                valid_segs = [
                    (s, e) for s, e in seg_ranges
                    if int(((q >= s * q_tot) & (q < e * q_tot)).sum()) >= min_pts
                ]

                # ── 전체 샘플링 범위 음영 (min q_frac_lo ~ max q_frac_hi) ──────
                if valid_segs:
                    q_lo_min = min(s for s, _ in valid_segs)
                    q_hi_max = max(e for _, e in valid_segs)
                    ax.axvspan(
                        q_lo_min, q_hi_max,
                        alpha=0.10, color="steelblue", zorder=-1,
                    )
                    # 경계선
                    for xv in (q_lo_min, q_hi_max):
                        ax.axvline(xv, color="steelblue", lw=1.0, ls="--",
                                   alpha=0.6, zorder=-1)
                    # 범위 레이블 (상단)
                    ax.text(
                        (q_lo_min + q_hi_max) / 2, ax.get_ylim()[1],
                        f"샘플링 범위  [{q_lo_min:.2f} – {q_hi_max:.2f}]",
                        ha="center", va="bottom", fontsize=6.5,
                        color="steelblue", fontweight="bold",
                    )

                # ── 개별 세그먼트 ────────────────────────────────────────────────
                patches = []
                for k, (s, e) in enumerate(valid_segs):
                    m = (q >= s * q_tot) & (q < e * q_tot)
                    color = cmap(k / max(len(valid_segs) - 1, 1))
                    ax.axvspan(s, e, alpha=0.18, color=color, zorder=0)
                    ax.plot(q_frac[m], v[m], "-", color=color, lw=2.0, zorder=2)
                    patches.append(mpatches.Patch(
                        color=color, alpha=0.7,
                        label=f"#{k+1}: {s:.2f}–{e:.2f} (Δ={e-s:.2f})"
                    ))

                ax.set_xlabel("q_frac", fontsize=8)
                ax.set_ylabel("V [V]", fontsize=8)
                ax.set_title(
                    f"[{ds}] {cell_id} — {actual_label}  cyc {demo_cyc}  [{src_tag}]",
                    fontsize=8,
                )
                if patches:
                    ax.legend(handles=patches, fontsize=5.5, loc="best", ncol=2)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)

        # 빈 행(셀 수 부족) 숨기기
        for row_idx in range(len(cell_list), n_rows):
            for pc in range(2):
                axes[row_idx, col_off + pc].axis("off")

    # ── 4. 열 구분선: 데이터셋 경계에 수직선 ─────────────────────────────────
    if n_ds > 1:
        mid_x = 0.5   # figure 좌표 기준 중앙
        fig.add_artist(
            plt.matplotlib.lines.Line2D(
                [mid_x, mid_x], [0.02, 0.96],
                transform=fig.transFigure,
                color="gray", lw=1.0, ls="--",
            )
        )

    source_tag = "캐시 우선" if from_cache else "RNG 생성"
    ds_labels = " / ".join(
        f"{ds}({len(cells)}셀)" for ds, cells in ds_cells.items()
    )
    fig.suptitle(
        f"test_rs 랜덤 세그먼트 생성 예시  [{source_tag}]  {ds_labels}\n"
        f"n_samples={n_samples}  길이={min_len:.0%}~{max_len:.0%}  seed={seed}  "
        f"열순서: [DS-방전 | DS-충전] × {n_ds}개 데이터셋",
        fontsize=10, y=1.005,
    )
    plt.tight_layout()

    out_path = out_dir / "demo_segments.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[demo] 세그먼트 시각화 [{source_tag}] → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="test_rs 세그먼트 HI 생성 (완전 랜덤 구간·랜덤 길이, 충/방전 구분)"
    )
    p.add_argument("--n-samples", type=int,   default=8,
                   help="방향당 사이클당 샘플 수 (default: 8 → 총 ~16/사이클)")
    p.add_argument("--min-len",   type=float, default=0.15,
                   help="최소 창 폭 q_frac (default: 0.15 = 15%%)")
    p.add_argument("--max-len",   type=float, default=0.40,
                   help="최대 창 폭 q_frac (default: 0.40 = 40%%)")
    p.add_argument("--seed",      type=int,   default=42,
                   help="난수 시드 (default: 42)")
    p.add_argument("--workers",   type=int,   default=2,
                   help="병렬 프로세스 수 (default: 2)")
    p.add_argument("--min-pts",   type=int,   default=10,
                   help="세그먼트 최소 포인트 수 (default: 10)")
    p.add_argument("--force",     action="store_true",
                   help="기존 출력 삭제 후 재생성")
    return p.parse_args()


def _cache_exists(out_dir: Path) -> bool:
    """out_dir/seg/ 아래 PKL이 하나라도 있으면 캐시 존재로 판단."""
    seg_root = out_dir / "seg"
    if not seg_root.exists():
        return False
    return any(seg_root.rglob("*.pkl"))


def main() -> None:
    args = _parse_args()

    out_dir   = PROJECT_ROOT / "_4_data_hi" / "test_rs"
    clean_dir = PROJECT_ROOT / "_4_data_hi" / "clean"

    # ── --force: 기존 출력 삭제 ──────────────────────────────────────────────
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"[test_rs] --force: {out_dir} 삭제 완료")

    # ── 캐시 확인 ────────────────────────────────────────────────────────────
    if _cache_exists(out_dir):
        print("=" * 62)
        print("[test_rs] 캐시 발견 — 세그먼트 분할 건너뜀 (플랏만 생성)")
        print(f"  캐시 경로 : {out_dir / 'seg'}")
        print(f"  재생성하려면 --force 옵션을 사용하세요")
        print("=" * 62)
        print("\n[demo 플롯 생성 중...]")
        _plot_demo_cycle(
            out_dir=out_dir,
            clean_dir=clean_dir,
            n_samples=args.n_samples,
            min_len=args.min_len,
            max_len=args.max_len,
            seed=args.seed,
            min_pts=args.min_pts,
        )
        return

    # ── 신규 생성 ─────────────────────────────────────────────────────────────
    for ds in ("MIT", "HUST"):
        (out_dir / "seg"   / ds).mkdir(parents=True, exist_ok=True)
        (out_dir / "cycle" / ds).mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("[test_rs] 완전 랜덤 세그먼트 데이터셋 생성")
    print("=" * 62)
    print(f"  n_samples  : {args.n_samples}  (방향당, 총 ~{2*args.n_samples}/사이클)")
    print(f"  길이 범위  : {args.min_len:.0%} ~ {args.max_len:.0%}  (랜덤, 겹침 허용)")
    print(f"  시작점     : q_frac [0, 1-length]  완전 랜덤")
    print(f"  시나리오   : 충전(0) / 방전(1)  2가지")
    print(f"  seed       : {args.seed}")
    print(f"  workers    : {args.workers}")
    print(f"  입력       : {clean_dir}")
    print(f"  출력       : {out_dir}")
    print()

    # ── ScenarioSpec 저장 ─────────────────────────────────────────────────────
    from common.scenario import get_segmenter
    _segmenter = get_segmenter("test_rs")
    _segmenter.save_artifacts(out_dir)
    print(f"  scenario_spec.json 저장 → {out_dir / 'scenario_spec.json'}")

    # ── 입력 파일 수집 ─────────────────────────────────────────────────────────
    task_args: list[tuple] = []
    for ds in ("MIT", "HUST"):
        ds_clean = clean_dir / ds
        if not ds_clean.exists():
            print(f"  [skip] {ds_clean} 없음")
            continue
        for pkl in sorted(ds_clean.glob("*.pkl")):
            task_args.append((
                str(pkl),
                args.n_samples, args.min_len, args.max_len,
                args.seed, args.min_pts,
            ))
    print(f"  총 {len(task_args)}개 셀 처리 시작\n")

    # ── 병렬 처리 ─────────────────────────────────────────────────────────────
    total_seg   = 0
    total_cycle = 0
    errors      = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(str(PROJECT_ROOT),),
    ) as ex:
        futs = {ex.submit(_extract_cell, a): a for a in task_args}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="cells"):
            try:
                cell_id, dataset, seg_rows, cycle_rows = fut.result()
            except Exception as e:
                print(f"\n  ERROR: {e}")
                errors += 1
                continue

            if not seg_rows:
                continue

            seg_df = pd.DataFrame(seg_rows)
            cyc_df = pd.DataFrame(cycle_rows) if cycle_rows else pd.DataFrame()

            # dataset 폴더 결정 (HUST 셀은 dataset 필드에 있음)
            ds_name = dataset if dataset in ("MIT", "HUST") else (
                "HUST" if "HUST" in str(futs[fut][0]).upper() else "MIT"
            )

            seg_path = out_dir / "seg" / ds_name / f"{cell_id}.pkl"
            with open(seg_path, "wb") as f:
                pickle.dump(seg_df, f)

            if not cyc_df.empty:
                cyc_path = out_dir / "cycle" / ds_name / f"{cell_id}.pkl"
                with open(cyc_path, "wb") as f:
                    pickle.dump(cyc_df, f)

            total_seg   += len(seg_df)
            total_cycle += len(cyc_df)

    print(f"\n  완료: {total_seg:,} 세그먼트 / {total_cycle:,} 사이클 / 오류 {errors}건")

    # ── 통계 파일 ─────────────────────────────────────────────────────────────
    print("\n[통계 계산 중...]")
    stats_str  = _compute_stats(out_dir)
    stats_path = out_dir / "test_rs_stats.txt"
    stats_path.write_text(stats_str, encoding="utf-8")
    print(stats_str)
    print(f"\n통계 저장 → {stats_path}")

    # ── 세그먼트 생성 예시 시각화 ────────────────────────────────────────────
    print("\n[demo 플롯 생성 중...]")
    _plot_demo_cycle(
        out_dir=out_dir,
        clean_dir=clean_dir,
        n_samples=args.n_samples,
        min_len=args.min_len,
        max_len=args.max_len,
        seed=args.seed,
        min_pts=args.min_pts,
    )

    # ── 최종 요약 ─────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("[test_rs] 완료!")
    for sub in ("seg", "cycle"):
        sub_dir = out_dir / sub
        if sub_dir.exists():
            n = sum(1 for _ in sub_dir.rglob("*.pkl"))
            print(f"  {sub:5s} : {n} pkl  ({sub_dir})")
    print(f"  spec  : {out_dir / 'scenario_spec.json'}")
    print(f"  stats : {stats_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
