"""
plot_cleaning_report.py   (plot_dropped_segments.py 스타일 기반)

preprocess.py [필터1~6] 이 제거·처리한 사이클/행을 유형별로 시각화.
각 플롯 제목에 [사이클 제거] / [행 제거] / [chg_gap_seg 플래그] 를 명시.

필터 유형:
  F1  빈 사이클           [사이클 제거]       — charge/discharge < 5행
  F3  rest 0전류          [행 제거]           — phase=rest && current_A==0.0
  F4A 방전 단절           [사이클 제거]       — dt.max > max(600s, med×50)
  F4B 충전 완전중단       [행 제거]           — dtc.max > max(600s, med×50)
  F4C 충전 CC전환갭       [chg_gap_seg 플래그] — dtc.max > max(120s, med×30)
  F5  Rolling Median      [사이클 제거]       — 방전용량 시계열 이상치
  F6  v_end 하한          [사이클 제거]       — 방전 종지전압 < 1.8V

데이터 소스:
  F1/F3/F4A/F4B/F4C/F6 : _1_data_unified/ PKL (전처리 前 원본) 병렬 스캔
  F5                    : 2_preprocess/outputs/cleaning_report.csv +
                          _1_data_unified/ PKL (용량 시계열)

출력: 2_preprocess/outputs/
  F1_empty.png  F3_rest.png  F4A_dis_gap.png  F4B_chg_stop.png
  F4C_chg_seg.png  F5_rolling.png  F6_vend.png

사용:
  python 2_preprocess/plot_cleaning_report.py
  python 2_preprocess/plot_cleaning_report.py --window 15 --sigma 3.0 --workers 8
"""

import argparse
import ast
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# numpy 1.x / 2.x pickle 호환성 shim
try:
    import numpy._core  # noqa: F401
except ImportError:
    import numpy.core as _nc
    for _attr in ("multiarray", "numeric", "umath", "fromnumeric",
                  "shape_base", "function_base"):
        if hasattr(_nc, _attr):
            sys.modules.setdefault(f"numpy._core.{_attr}", getattr(_nc, _attr))
    sys.modules.setdefault("numpy._core", _nc)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# ── 경로 ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MIT_DIR      = PROJECT_ROOT / "_1_data_unified" / "MIT"
HUST_DIR     = PROJECT_ROOT / "_1_data_unified" / "HUST"
OUT_DIR      = HERE / "outputs" / date.today().strftime("%m%d")
REPORT_PATH  = HERE / "outputs" / "cleaning_report.csv"

# ── 임계값 (preprocess.py 와 동일) ──────────────────────────────────────────
DIS_GAP_S          = 600.0;  DIS_GAP_FACTOR      = 50.0
CHG_STOP_S         = 600.0;  CHG_STOP_FACTOR     = 50.0
CHG_SEG_GAP_S      = 120.0;  CHG_SEG_GAP_FACTOR  = 30.0
MIN_ACTIVE_ROWS    = 5
VEND_MIN           = 1.8

DS_COLOR = {"MIT": "#1f77b4", "HUST": "#d55e00"}

# ── 폰트 ─────────────────────────────────────────────────────────────────────
for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _f; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _iter_pkls():
    for ds, ddir in [("MIT", MIT_DIR), ("HUST", HUST_DIR)]:
        if not ddir.exists():
            continue
        for pkl in sorted(ddir.glob("*.pkl")):
            yield ds, pkl


def _load_pkl(pkl: Path) -> tuple:
    with open(pkl, "rb") as f:
        raw = pickle.load(f)
    return raw["meta"], raw["cycles"]


def _make_grid(n, ncols=6):
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.6),
                             constrained_layout=True)
    return fig, np.array(axes).reshape(nrows, ncols)


def _hide_empty(axes_2d, n_used):
    for ax in axes_2d.flatten()[n_used:]:
        ax.set_visible(False)


def _parse_list(val) -> list:
    if pd.isna(val) or str(val).strip() in ("", "[]"):
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


def _dis_tuple(grp):
    dis = grp[grp["phase"] == "discharge"].sort_values("time_s")
    if len(dis) < 5:
        return None
    t   = dis["time_s"].values.astype(float)
    v   = dis["voltage_V"].values.astype(float)
    im  = np.abs(dis["current_A"].values.astype(float))
    dt  = np.clip(np.diff(t, prepend=t[0]), 0, None)
    cap = float(dis["capacity_Ah"].iloc[0])
    return v, t, dt, im, cap


def _chg_tuple(grp):
    chg = grp[grp["phase"] == "charge"].sort_values("time_s")
    if len(chg) < 5:
        return None
    tc  = chg["time_s"].values.astype(float)
    vc  = chg["voltage_V"].values.astype(float)
    ic  = np.abs(chg["current_A"].values.astype(float))
    dtc = np.clip(np.diff(tc, prepend=tc[0]), 0, None)
    return vc, tc, dtc, ic


# ─────────────────────────────────────────────────────────────────────────────
# PKL 스캔 워커  (ThreadPoolExecutor — numpy/pandas GIL 해제 + I/O 병렬)
# ─────────────────────────────────────────────────────────────────────────────

def _scan_pkl(args: tuple) -> dict:
    """단일 PKL 스캔 → 필터별 케이스 dict 반환."""
    ds, pkl_str = args
    pkl = Path(pkl_str)
    out: dict = {"f1": [], "f3": {}, "f4a": [], "f4b": [], "f4c": [], "f6": [],
                 "error": None, "pkl": pkl.stem}
    try:
        meta, df = _load_pkl(pkl)
    except Exception as e:
        out["error"] = str(e)
        return out
    cell_id = meta.get("cell_id", pkl.stem)

    for cyc, grp in df.groupby("cycle"):
        cyc = int(cyc)
        if cyc == 0:
            continue

        # F1 빈 사이클
        active = grp[grp["phase"].isin(["charge", "discharge"])]
        if len(active) < MIN_ACTIVE_ROWS:
            out["f1"].append({"ds": ds, "cell_id": cell_id, "cycle": cyc,
                              "rows": len(active)})
            continue

        # F3 rest 0전류
        n_rest = int(((grp["phase"] == "rest") & (grp["current_A"] == 0.0)).sum())
        if n_rest > 0:
            key = (ds, cell_id)
            out["f3"][key] = out["f3"].get(key, 0) + n_rest

        # 방전 데이터
        dis_tup = _dis_tuple(grp)
        if dis_tup is None:
            continue
        v, t, dt, im, cap = dis_tup
        if not np.isfinite(cap) or cap < 0.05:
            continue

        dt_pos = dt[dt > 0]
        dt_med = float(np.median(dt_pos)) if len(dt_pos) > 0 else 1.0
        dt_max = float(dt.max())

        # F4A 방전 단절
        if dt_max > max(DIS_GAP_S, dt_med * DIS_GAP_FACTOR):
            out["f4a"].append({
                "ds": ds, "cell_id": cell_id, "cycle": cyc,
                "dt_max": dt_max, "dt_med": dt_med, "cap": cap,
                "t": t, "v": v, "i_mag": im,
                "gap_idx": int(np.argmax(dt)),
            })
            continue

        # F6 v_end 하한
        if len(v) > 0 and float(v[-1]) < VEND_MIN:
            out["f6"].append({
                "ds": ds, "cell_id": cell_id, "cycle": cyc, "cap": cap,
                "t": t, "v": v, "i_mag": im,
            })

        # 충전 데이터
        chg_tup = _chg_tuple(grp)
        if chg_tup is None:
            continue
        vc, tc, dtc, ic = chg_tup
        dtc_pos = dtc[dtc > 0]
        dtc_med = float(np.median(dtc_pos)) if len(dtc_pos) > 0 else 1.0
        dtc_max = float(dtc.max())

        # F4B 충전 완전중단
        if dtc_max > max(CHG_STOP_S, dtc_med * CHG_STOP_FACTOR):
            out["f4b"].append({
                "ds": ds, "cell_id": cell_id, "cycle": cyc,
                "dtc_max": dtc_max, "dtc_med": dtc_med,
                "tc": tc, "vc": vc, "ic": ic,
                "gap_idx": int(np.argmax(dtc)),
            })
            continue

        # F4C 충전 CC전환갭
        if dtc_max > max(CHG_SEG_GAP_S, dtc_med * CHG_SEG_GAP_FACTOR):
            out["f4c"].append({
                "ds": ds, "cell_id": cell_id, "cycle": cyc,
                "dtc_max": dtc_max, "dtc_med": dtc_med,
                "tc": tc, "vc": vc, "ic": ic,
                "gap_idx": int(np.argmax(dtc)),
            })
    return out


def _merge_results(results: list) -> tuple:
    f1: list = []; f3: dict = {}
    f4a: list = []; f4b: list = []; f4c: list = []; f6: list = []
    for r in results:
        if r["error"]:
            print(f"  [SKIP] {r['pkl']}: {r['error']}")
            continue
        f1.extend(r["f1"])
        for k, v in r["f3"].items():
            f3[k] = f3.get(k, 0) + v
        f4a.extend(r["f4a"])
        f4b.extend(r["f4b"])
        f4c.extend(r["f4c"])
        f6.extend(r["f6"])
    return f1, f3, f4a, f4b, f4c, f6


# ─────────────────────────────────────────────────────────────────────────────
# 패널 드로어
# ─────────────────────────────────────────────────────────────────────────────

def _panel_dis(ax, case, vend_line=None):
    t, v, im = case["t"], case["v"], case["i_mag"]
    ax2 = ax.twinx()
    ax.plot(t, v,   color="steelblue", lw=1.0, alpha=0.85)
    ax2.plot(t, im, color="tomato",    lw=0.7, alpha=0.5)
    if vend_line is not None:
        ax.axhline(vend_line, color="red", lw=1.0, ls="--", alpha=0.8)
    ax.set_ylabel("V (V)",   fontsize=5, color="steelblue")
    ax2.set_ylabel("|I| (A)", fontsize=5, color="tomato")
    ax.tick_params(labelsize=5); ax2.tick_params(labelsize=5)
    ax.grid(True, alpha=0.25)


def _panel_dis_gap(ax, case):
    t, v, im, gi = case["t"], case["v"], case["i_mag"], case["gap_idx"]
    ax2 = ax.twinx()
    ax.scatter(t, v,   c="steelblue", s=2, alpha=0.7, zorder=3)
    ax2.scatter(t, im, c="tomato",    s=1, alpha=0.4, zorder=2)
    if 0 < gi < len(t):
        ax.axvspan(t[gi - 1], t[gi], color="red", alpha=0.20)
        ax.axvline(t[gi - 1], color="red", lw=1.2, ls="--")
    ax.set_title(
        f"{case['ds']} {case['cell_id']} c{case['cycle']}\n"
        f"gap={case['dt_max']:.0f}s  med={case['dt_med']:.1f}s  cap={case['cap']:.3f}Ah",
        fontsize=6.5, pad=2)
    ax.set_ylabel("V (V)",   fontsize=5, color="steelblue")
    ax2.set_ylabel("|I| (A)", fontsize=5, color="tomato")
    ax.tick_params(labelsize=5); ax2.tick_params(labelsize=5)
    ax.grid(True, alpha=0.25)


def _panel_chg_gap(ax, case, label_kind):
    tc, vc, ic, gi = case["tc"], case["vc"], case["ic"], case["gap_idx"]
    ax2 = ax.twinx()
    ax.scatter(tc, vc, c="darkorange", s=2, alpha=0.7, zorder=3)
    ax2.scatter(tc, ic, c="purple",   s=1, alpha=0.4, zorder=2)
    if 0 < gi < len(tc):
        ax.axvspan(tc[gi - 1], tc[gi], color="red", alpha=0.20)
        ax.axvline(tc[gi - 1], color="red", lw=1.2, ls="--")
    ax.set_title(
        f"{case['ds']} {case['cell_id']} c{case['cycle']}\n"
        f"gap={case['dtc_max']:.0f}s  med={case['dtc_med']:.1f}s  [{label_kind}]",
        fontsize=6.5, pad=2)
    ax.set_ylabel("V (V)",   fontsize=5, color="darkorange")
    ax2.set_ylabel("|I| (A)", fontsize=5, color="purple")
    ax.tick_params(labelsize=5); ax2.tick_params(labelsize=5)
    ax.grid(True, alpha=0.25)


# ─────────────────────────────────────────────────────────────────────────────
# 플롯 함수 (ThreadPoolExecutor 로 병렬 실행)
# ─────────────────────────────────────────────────────────────────────────────

def plot_f1(cases):
    if not cases:
        print("  [F1] 빈 사이클 없음 (skip)"); return
    ncols = min(6, len(cases))
    fig, axes = _make_grid(len(cases), ncols=ncols)
    fig.suptitle(
        f"[F1] 빈 사이클  [사이클 제거]  ({len(cases)}건)\n"
        f"charge/discharge 행 수 < {MIN_ACTIVE_ROWS}",
        fontsize=9, fontweight="bold")
    for idx, c in enumerate(cases):
        r, col = divmod(idx, ncols)
        ax = axes[r, col]
        ax.text(0.5, 0.5,
                f"{c['ds']} {c['cell_id']}\ncycle={c['cycle']}\nactive rows={c['rows']}",
                transform=ax.transAxes, ha="center", va="center", fontsize=8,
                color="darkred",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdecea", alpha=0.9))
        ax.set_title(f"{c['ds']} {c['cell_id']} c{c['cycle']}", fontsize=7, pad=2)
        ax.axis("off")
    _hide_empty(axes, len(cases))
    out = OUT_DIR / "F1_empty.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [F1] 저장: {out}")


def plot_f3(rest_dict):
    if not rest_dict:
        print("  [F3] rest 0전류 행 없음 (skip)"); return
    rows   = sorted(rest_dict.items(), key=lambda x: -x[1])
    labels = [f"{ds}\n{cid}" for (ds, cid), _ in rows]
    counts = [n for _, n in rows]
    colors = [DS_COLOR.get(ds, "gray") for (ds, _), _ in rows]
    n_cells = len(rows)
    fig_w   = max(10, n_cells * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, 5), constrained_layout=True)
    fig.suptitle(
        f"[F3] rest 0전류 행  [행 제거]  —  총 {sum(counts):,}행  ({n_cells}셀)\n"
        "phase=rest & current_A==0.0 인 행 제거",
        fontsize=10, fontweight="bold")
    ax.bar(range(n_cells), counts, color=colors, alpha=0.85)
    ax.set_xticks(range(n_cells))
    ax.set_xticklabels(labels, rotation=90, fontsize=max(4, min(7, 200 // n_cells)))
    ax.set_ylabel("제거 행 수", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for ds, c in DS_COLOR.items():
        ax.bar([], [], color=c, alpha=0.85, label=ds)
    ax.legend(fontsize=9, loc="upper right")
    out = OUT_DIR / "F3_rest.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [F3] 저장: {out}")


def plot_f4a(cases):
    if not cases:
        print("  [F4A] 방전 단절 없음 (skip)"); return
    cases_s = sorted(cases, key=lambda x: -x["dt_max"])
    ncols   = min(6, len(cases_s))
    fig, axes = _make_grid(len(cases_s), ncols=ncols)
    fig.suptitle(
        f"[F4A] 방전 시간 단절  [사이클 제거]  ({len(cases_s)}건)\n"
        f"dt.max > max({DIS_GAP_S:.0f}s, med×{DIS_GAP_FACTOR:.0f})  |  "
        "빨간 영역=갭  |  파란=V, 주황=|I|",
        fontsize=9, fontweight="bold")
    for idx, c in enumerate(cases_s):
        r, col = divmod(idx, ncols)
        _panel_dis_gap(axes[r, col], c)
    _hide_empty(axes, len(cases_s))
    out = OUT_DIR / "F4A_dis_gap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [F4A] 저장: {out}")


def plot_f4b(cases):
    if not cases:
        print("  [F4B] 충전 완전중단 없음 (skip)"); return
    by_cell = {}
    for c in cases:
        key = (c["ds"], c["cell_id"])
        if key not in by_cell or c["dtc_max"] > by_cell[key]["dtc_max"]:
            by_cell[key] = c
    rep   = sorted(by_cell.values(), key=lambda x: -x["dtc_max"])
    ncols = min(6, len(rep))
    fig, axes = _make_grid(len(rep), ncols=ncols)
    fig.suptitle(
        f"[F4B] 충전 완전중단  [행 제거]  "
        f"(전체 {len(cases)}건, 셀당 최악 1건: {len(rep)}셀)\n"
        f"dtc.max > max({CHG_STOP_S:.0f}s, med×{CHG_STOP_FACTOR:.0f})  |  "
        "빨간 영역=갭  |  주황=V, 보라=|I|",
        fontsize=9, fontweight="bold")
    for idx, c in enumerate(rep):
        r, col = divmod(idx, ncols)
        _panel_chg_gap(axes[r, col], c, "행 제거")
    _hide_empty(axes, len(rep))
    out = OUT_DIR / "F4B_chg_stop.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [F4B] 저장: {out}  (전체 {len(cases)}건 중 셀 대표 {len(rep)}건)")


def plot_f4c(cases):
    if not cases:
        print("  [F4C] 충전 CC전환갭 없음 (skip)"); return
    by_cell = {}
    for c in cases:
        key = (c["ds"], c["cell_id"])
        if key not in by_cell or c["dtc_max"] > by_cell[key]["dtc_max"]:
            by_cell[key] = c
    rep   = sorted(by_cell.values(), key=lambda x: -x["dtc_max"])
    ncols = min(6, len(rep))
    fig, axes = _make_grid(len(rep), ncols=ncols)
    fig.suptitle(
        f"[F4C] 충전 CC 프로토콜 전환갭  [chg_gap_seg 플래그]  "
        f"(전체 {len(cases)}건, 셀당 최악 1건: {len(rep)}셀)\n"
        f"dtc.max > max({CHG_SEG_GAP_S:.0f}s, med×{CHG_SEG_GAP_FACTOR:.0f})  |  "
        "충전 세그먼트 HI만 NaN, 전역 HI 유지  |  주황=V, 보라=|I|",
        fontsize=9, fontweight="bold")
    for idx, c in enumerate(rep):
        r, col = divmod(idx, ncols)
        _panel_chg_gap(axes[r, col], c, "세그먼트 HI NaN")
    _hide_empty(axes, len(rep))
    out = OUT_DIR / "F4C_chg_seg.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [F4C] 저장: {out}  (전체 {len(cases)}건 중 셀 대표 {len(rep)}건)")


def plot_f5(window: int, sigma: float, min_std: float, top_n: int):
    if not REPORT_PATH.exists():
        print(f"  [F5] cleaning_report.csv 없음 (skip): {REPORT_PATH}"); return
    rpt   = pd.read_csv(REPORT_PATH)
    rpt["dataset"] = rpt["dataset"].str.replace("MIT_MAT", "MIT", regex=False)
    rm_df = rpt[rpt["n_removed_rolling"] > 0].copy()
    if len(rm_df) == 0:
        print("  [F5] Rolling Median 제거 없음 (skip)"); return
    rm_df = rm_df.sort_values("n_removed_rolling", ascending=False).head(top_n)
    ncols = min(3, len(rm_df))
    nrows = (len(rm_df) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 3.8),
                             constrained_layout=True)
    axes_flat = np.array(axes).reshape(-1)
    fig.suptitle(
        f"[F5] Rolling Median  [사이클 제거]  —  상위 {len(rm_df)}개 셀\n"
        f"window={window}, sigma={sigma}  음영=±{sigma}σ 밴드  |  X=제거 사이클",
        fontsize=11, fontweight="bold")

    for ai, (_, row) in enumerate(rm_df.iterrows()):
        ax      = axes_flat[ai]
        ds      = row["dataset"]
        cell_id = row["cell_id"]
        removed = set(_parse_list(row["removed_rolling_cycles"]))
        pkl = PROJECT_ROOT / "_1_data_unified" / ds / f"{cell_id}.pkl"
        if not pkl.exists():
            ax.text(0.5, 0.5, f"{cell_id}\n(PKL 없음)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9)
            continue
        try:
            _, df_raw = _load_pkl(pkl)
        except Exception as e:
            ax.text(0.5, 0.5, f"{cell_id}\n({e})",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8)
            continue
        dis   = df_raw[df_raw["phase"] == "discharge"]
        cap_s = dis.groupby("cycle")["capacity_Ah"].first().dropna().sort_index()
        cycs  = cap_s.index.to_numpy()
        caps  = cap_s.values
        s     = pd.Series(caps, index=cycs)
        roll  = s.rolling(window=window, center=True, min_periods=3)
        med   = roll.median().values
        std   = roll.std().fillna(s.std()).clip(lower=min_std).values
        color = DS_COLOR.get(ds, "gray")
        ax.fill_between(cycs, med - sigma * std, med + sigma * std,
                        alpha=0.18, color=color)
        ax.plot(cycs, med,  color=color, lw=1.2, ls="--", label="Rolling Median")
        ax.plot(cycs, caps, color="dimgray", lw=0.7, alpha=0.5)
        rm_mask = np.isin(cycs, list(removed))
        ax.scatter(cycs[~rm_mask], caps[~rm_mask], color=color, s=6, alpha=0.5, zorder=2)
        if rm_mask.any():
            ax.scatter(cycs[rm_mask], caps[rm_mask],
                       color="red", s=35, marker="x", lw=1.3, zorder=4,
                       label=f"제거 {rm_mask.sum()}건")
        ax.set_title(f"{ds}: {cell_id}  (−{len(removed)}건)  [사이클 제거]",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Cycle", fontsize=8); ax.set_ylabel("Capacity (Ah)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(rm_df):]:
        ax.set_visible(False)
    out = OUT_DIR / "F5_rolling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [F5] 저장: {out}")


def plot_f6(cases):
    if not cases:
        print("  [F6] v_end 하한 제거 없음 (skip)"); return
    by_cell = {}
    for c in cases:
        key  = (c["ds"], c["cell_id"])
        vend = float(c["v"][-1]) if len(c["v"]) > 0 else 9.9
        if key not in by_cell or vend < float(by_cell[key]["v"][-1]):
            by_cell[key] = c
    rep   = sorted(by_cell.values(),
                   key=lambda x: float(x["v"][-1]) if len(x["v"]) > 0 else 9.9)
    ncols = min(6, len(rep))
    fig, axes = _make_grid(len(rep), ncols=ncols)
    fig.suptitle(
        f"[F6] v_end 하한  [사이클 제거]  "
        f"(전체 {len(cases)}건, 셀당 최저 v_end 1건: {len(rep)}셀)\n"
        f"방전 종지전압 < {VEND_MIN}V  |  빨간 점선={VEND_MIN}V  |  파란=V, 주황=|I|",
        fontsize=9, fontweight="bold")
    for idx, c in enumerate(rep):
        r, col = divmod(idx, ncols)
        ax = axes[r, col]
        _panel_dis(ax, c, vend_line=VEND_MIN)
        vend = float(c["v"][-1]) if len(c["v"]) > 0 else float("nan")
        ax.set_title(
            f"{c['ds']} {c['cell_id']} c{c['cycle']}\n"
            f"v_end={vend:.3f}V  cap={c['cap']:.3f}Ah  [사이클 제거]",
            fontsize=6.5, pad=2)
    _hide_empty(axes, len(rep))
    out = OUT_DIR / "F6_vend.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [F6] 저장: {out}  (전체 {len(cases)}건 중 셀 대표 {len(rep)}건)")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="preprocess.py 필터별 제거 사이클/행 시각화")
    parser.add_argument("--window",      type=int,   default=11)
    parser.add_argument("--sigma",       type=float, default=2.5)
    parser.add_argument("--min-std",     type=float, default=0.01)
    parser.add_argument("--top-rolling", type=int,   default=9,
                        help="F5 Rolling 상세 표시 최대 셀 수 (기본: 9)")
    parser.add_argument("--workers",     type=int,
                        default=min(16, (os.cpu_count() or 4)),
                        help="PKL 병렬 스캔 스레드 수 (기본: min(16, CPU코어))")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1단계: PKL 병렬 스캔 ──────────────────────────────────────────────────
    pkl_list = [(ds, str(pkl)) for ds, pkl in _iter_pkls()]
    if not pkl_list:
        print("1_data_unified/ 에 PKL 파일 없음. 경로 확인 필요."); return

    print(f"=== PKL 병렬 스캔 ({len(pkl_list)}개, workers={args.workers}) ===")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_scan_pkl, a): a for a in pkl_list}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="scan"):
            results.append(fut.result())

    f1, f3, f4a, f4b, f4c, f6 = _merge_results(results)

    print(f"\n수집 완료:")
    print(f"  F1  빈 사이클     [사이클 제거]      : {len(f1):>6}건")
    print(f"  F3  rest 0전류    [행 제거]           : {sum(f3.values()):>6}행  ({len(f3)}셀)")
    print(f"  F4A 방전 단절     [사이클 제거]       : {len(f4a):>6}건")
    print(f"  F4B 충전 완전중단 [행 제거]           : {len(f4b):>6}건")
    print(f"  F4C 충전 CC전환갭 [chg_gap_seg 플래그]: {len(f4c):>6}건")
    print(f"  F6  v_end 하한    [사이클 제거]       : {len(f6):>6}건")

    # ── 2단계: 플롯 병렬 생성 ─────────────────────────────────────────────────
    print("\n=== 플롯 병렬 생성 (7개) ===")
    plot_tasks = {
        "F1":  (plot_f1,  (f1,)),
        "F3":  (plot_f3,  (f3,)),
        "F4A": (plot_f4a, (f4a,)),
        "F4B": (plot_f4b, (f4b,)),
        "F4C": (plot_f4c, (f4c,)),
        "F5":  (plot_f5,  (args.window, args.sigma, args.min_std, args.top_rolling)),
        "F6":  (plot_f6,  (f6,)),
    }
    with ThreadPoolExecutor(max_workers=len(plot_tasks)) as ex:
        fut_map = {ex.submit(fn, *fn_args): name
                   for name, (fn, fn_args) in plot_tasks.items()}
        for fut in as_completed(fut_map):
            name = fut_map[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"  [{name}] 오류: {e}")

    print(f"\n완료 -- {OUT_DIR}/")


if __name__ == "__main__":
    main()
