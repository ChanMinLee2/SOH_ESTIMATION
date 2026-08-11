"""
phase2_dataset_features.py — docs/SOC.md §6 Phase 2: "ref_SOC를 안다"는 가정이 성립하려면
먼저 확인해야 할 6가지 데이터셋 특징을 MIT/HUST 전 셀에 대해 엄밀하게 검증한다.

(a) 용량 페이드 단조성 — capacity_Ah가 사이클에 따라 단조 감소하는가?
(b) 결측 사이클 밀도 — 사이클 번호가 듬성한 정도(원본 번호 기준 lag 정의에 영향)
(c) 완전 방전/충전 도달 여부 — 컷오프 전압까지 실제로 가는가(부분 사이클이면 q_cum이
    완전 용량을 대표 못함) — hi_correlation.py의 기존 완화 필터(q_local>=cap*0.3,
    q_tc>=cap*0.6) 위에서 "진짜 완전 도달" 비율을 별도로 잰다.
(d) 충·방전 용량비(쿨롱효율, Qc/Qd) — 방전 레퍼런스를 충전에 그대로 쓸 수 있는지 근거.
(e) SOC 전이 범위 — 셀이 실제로 컷오프~만충 전압을 다 훑는가.
(f) knee-point 위치 — 셀별 급격 열화 전환점(거리-현곡선법, distance-from-chord).

입력: _4_data_hi/clean/{MIT,HUST}/*.pkl (phase 컬럼 없음 — current_A 부호로 재현,
      1_convert/convert_unified.py의 assign_phase()와 동일 임계값)

출력:
  3_integrity/outputs/phase2_dataset_features.csv        — 셀별 전체 지표
  3_integrity/outputs/phase2_a_monotonicity.png
  3_integrity/outputs/phase2_b_missing_cycles.png
  3_integrity/outputs/phase2_c_full_reach.png
  3_integrity/outputs/phase2_d_coulomb_efficiency.png
  3_integrity/outputs/phase2_e_soc_range.png
  3_integrity/outputs/phase2_f_knee_point.png
  3_integrity/outputs/phase2_f_knee_examples_{MIT,HUST}.png  — 대표 셀 knee 표시 곡선

사용:
  python 3_integrity/phase2_dataset_features.py --workers 16
"""

from __future__ import annotations

import argparse
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

for _font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"):
    if _font_name in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _font_name
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_directories import DATA_4_HI_ROOT  # noqa: E402
CLEAN_DIR    = DATA_4_HI_ROOT / "clean"
OUT_DIR      = Path(__file__).resolve().parent / "outputs"

DATASETS = ["MIT", "HUST"]
BATCH_COLORS = {"MIT": "#4C72B0", "HUST": "#DD8452"}

PHASE_POS = 0.01
PHASE_NEG = -0.01

# 2026-08-05 정정: 둘 다 정상 컷오프는 2.0V다(MIT/HUST 동일). 1.8V는
# preprocess.py 필터6 vend_min(비정상 조기종료 사이클 "제거"용 하한)일 뿐,
# HUST의 정상 컷오프가 아니다 — 실측 방전 종지전압 분포(중앙값 1.975V, p90=1.996V,
# p99≈2.0V)가 이를 확인해준다. 예전에 vend_min과 컷오프를 혼동해 1.80을 넣었던 건
# 오류였다(사용자 지적으로 발견, docs/DATASET_ANOMALIES.md "HUST 정상 컷오프 2.0V"
# 서술과도 애초에 모순이었음).
DIS_CUTOFF_V = {"MIT": 2.00, "HUST": 2.00}
CUTOFF_TOL_V = 0.05                 # 컷오프+이 값 이내면 "도달"로 인정
FULL_CHARGE_V = 3.55                # LFP 만충 근접 판정(실측 최대 3.60~3.65V)
NONMONO_SIG_THRESH = 0.01           # 직전 대비 +1% 초과 증가만 "유의미한" 비단조로 카운트
MIN_CYCLES_FOR_KNEE = 30            # knee 추정에 필요한 최소 사이클 수
KNEE_SMOOTH_WINDOW = 11             # 필터5(rolling median)와 동일 관례


def _batch_of(dataset: str, cell_id: str) -> str:
    if dataset == "MIT":
        import re
        m = re.match(r"(b\d+)c", cell_id)
        return m.group(1) if m else "unknown"
    import re
    m = re.match(r"(\d+)-", cell_id)
    return f"{m.group(1)}-" if m else "unknown"


def _load_cycles(pkl_path: Path) -> pd.DataFrame:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    return raw["cycles"] if isinstance(raw, dict) else raw


def _integrate(grp: pd.DataFrame) -> float:
    """cumsum(|I|*dt)/3600의 마지막 값(그 phase의 전류적산, Ah)."""
    g = grp.sort_values("time_s")
    t = g["time_s"].to_numpy(dtype=float)
    i_mag = np.abs(g["current_A"].to_numpy(dtype=float))
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    return float(np.cumsum(i_mag * dt)[-1] / 3600.0)


def _knee_cycle(cycles: np.ndarray, cap: np.ndarray) -> tuple[float, float]:
    """거리-현곡선(distance-from-chord)법으로 knee 사이클 추정.

    (rolling median으로 스무딩한) 정규화 곡선에서, 시작점-끝점을 잇는 직선(현)으로부터
    수직거리가 최대인 점을 knee로 판정 — 배터리 열화 knee-point 탐지의 표준 기법
    (Attia et al./Severson et al. 계열 문헌에서 흔히 쓰는 방식).

    Returns: (knee_cycle, knee_cycle_frac) — frac은 knee/(사이클 범위) 로 0~1 정규화.
    """
    if len(cycles) < MIN_CYCLES_FOR_KNEE:
        return float("nan"), float("nan")
    order = np.argsort(cycles)
    cyc = cycles[order].astype(float)
    c = pd.Series(cap[order]).rolling(
        window=KNEE_SMOOTH_WINDOW, center=True, min_periods=3
    ).median().to_numpy()
    valid = np.isfinite(c)
    cyc, c = cyc[valid], c[valid]
    if len(cyc) < MIN_CYCLES_FOR_KNEE:
        return float("nan"), float("nan")

    x = (cyc - cyc.min()) / max(cyc.max() - cyc.min(), 1e-9)
    y = (c - c.min()) / max(c.max() - c.min(), 1e-9)
    # 현(첫점-끝점) 직선까지의 수직거리 = |cross product| / |chord|
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = np.hypot(dx, dy) + 1e-12
    dist = np.abs(dx * (y0 - y) - (x0 - x) * dy) / denom
    k_idx = int(np.argmax(dist))
    knee_cyc = float(cyc[k_idx])
    knee_frac = float((knee_cyc - cyc.min()) / max(cyc.max() - cyc.min(), 1e-9))
    return knee_cyc, knee_frac


def scan_cell(pkl_path: Path, dataset: str) -> dict | None:
    cell_id = pkl_path.stem
    df = _load_cycles(pkl_path)
    if df is None or len(df) == 0:
        return None
    batch = _batch_of(dataset, cell_id)

    dis_cutoff = DIS_CUTOFF_V[dataset]

    cycles_all = sorted(df["cycle"].unique())
    n_cycles = len(cycles_all)
    if n_cycles < 5:
        return None

    # ── (a) 단조성 + (f) knee용 등록용량 시계열 ────────────────────────────
    dis_all = df[df["current_A"] < PHASE_NEG]
    reg_cap_by_cycle = dis_all.groupby("cycle")["capacity_Ah"].first().dropna().sort_index()
    reg_cap_cycles = reg_cap_by_cycle.index.to_numpy(dtype=float)
    reg_cap_vals = reg_cap_by_cycle.to_numpy(dtype=float)

    n_nonmono = 0
    n_nonmono_sig = 0
    if len(reg_cap_vals) >= 2:
        rel_delta = np.diff(reg_cap_vals) / np.clip(reg_cap_vals[:-1], 1e-9, None)
        n_nonmono = int((rel_delta > 0).sum())
        n_nonmono_sig = int((rel_delta > NONMONO_SIG_THRESH).sum())
    n_pairs = max(len(reg_cap_vals) - 1, 1)

    # ── (b) 결측 사이클 밀도 ────────────────────────────────────────────────
    cyc_arr = np.array(cycles_all, dtype=int)
    full_span = int(cyc_arr.max() - cyc_arr.min() + 1)
    missing_ratio = 1.0 - (len(cyc_arr) / full_span) if full_span > 0 else np.nan
    max_gap = int(np.diff(cyc_arr).max()) if len(cyc_arr) >= 2 else 0

    # ── (c)(d) 사이클별 완전도달/쿨롱효율 ───────────────────────────────────
    dis_by_cyc = {c: g for c, g in dis_all.groupby("cycle")}
    chg_all = df[df["current_A"] > PHASE_POS]
    chg_by_cyc = {c: g for c, g in chg_all.groupby("cycle")}

    n_dis_reach = n_dis_gate = n_dis_checked = 0
    n_chg_reach = n_chg_gate = n_chg_checked = 0
    ce_list: list[float] = []
    v_dis_span: list[float] = []
    v_chg_span: list[float] = []

    for cyc in cycles_all:
        dgrp = dis_by_cyc.get(cyc)
        cgrp = chg_by_cyc.get(cyc)
        cap = None
        if dgrp is not None and len(dgrp) >= 5:
            cap = float(dgrp["capacity_Ah"].iloc[0])
            n_dis_checked += 1
            dgrp_sorted = dgrp.sort_values("time_s")
            v_start = float(dgrp_sorted["voltage_V"].iloc[0])
            v_end   = float(dgrp_sorted["voltage_V"].iloc[-1])
            if v_end <= dis_cutoff + CUTOFF_TOL_V:
                n_dis_reach += 1
            q_local = _integrate(dgrp)
            if cap and cap > 0 and q_local / cap >= 0.30:
                n_dis_gate += 1
            v_dis_span.append(v_start - v_end)   # (e) 방전 전압 스윙

        if cgrp is not None and len(cgrp) >= 5:
            n_chg_checked += 1
            cgrp_sorted = cgrp.sort_values("time_s")
            v_start_c = float(cgrp_sorted["voltage_V"].iloc[0])
            v_max_c   = float(cgrp_sorted["voltage_V"].max())
            if v_max_c >= FULL_CHARGE_V:
                n_chg_reach += 1
            q_chg = _integrate(cgrp)
            if cap and cap > 0 and q_chg / cap >= 0.60:
                n_chg_gate += 1
            v_chg_span.append(v_max_c - v_start_c)  # (e) 충전 전압 스윙
            if dgrp is not None and len(dgrp) >= 5:
                q_local = _integrate(dgrp)
                if q_local > 1e-6:
                    ce_list.append(q_chg / q_local)  # (d) 쿨롱효율 Qc/Qd

    # ── (f) knee-point ──────────────────────────────────────────────────────
    knee_cycle, knee_frac = _knee_cycle(reg_cap_cycles, reg_cap_vals)

    return {
        "dataset": dataset, "batch": batch, "cell_id": cell_id,
        "n_cycles": n_cycles,
        # (a)
        "nonmono_ratio":     n_nonmono / n_pairs,
        "nonmono_sig_ratio": n_nonmono_sig / n_pairs,
        # (b)
        "missing_cycle_ratio": missing_ratio,
        "max_cycle_gap":       max_gap,
        # (c)
        "frac_dis_reach_cutoff": n_dis_reach / n_dis_checked if n_dis_checked else np.nan,
        "frac_dis_pass_hi_gate": n_dis_gate  / n_dis_checked if n_dis_checked else np.nan,
        "frac_chg_reach_full":   n_chg_reach / n_chg_checked if n_chg_checked else np.nan,
        "frac_chg_pass_hi_gate": n_chg_gate  / n_chg_checked if n_chg_checked else np.nan,
        # (d)
        "coulomb_eff_median": float(np.median(ce_list)) if ce_list else np.nan,
        "coulomb_eff_std":    float(np.std(ce_list))    if ce_list else np.nan,
        "coulomb_eff_n":      len(ce_list),
        # (e)
        "v_dis_span_median": float(np.median(v_dis_span)) if v_dis_span else np.nan,
        "v_chg_span_median": float(np.median(v_chg_span)) if v_chg_span else np.nan,
        # (f)
        "knee_cycle":      knee_cycle,
        "knee_cycle_frac": knee_frac,
        # 예시 플랏용 원자료(대표 셀만 나중에 재사용)
        "_reg_cap_cycles": reg_cap_cycles,
        "_reg_cap_vals":   reg_cap_vals,
    }


def _worker(args: tuple) -> tuple:
    pkl_path_str, dataset = args
    try:
        r = scan_cell(Path(pkl_path_str), dataset)
        return ("ok", r)
    except Exception:
        return ("err", Path(pkl_path_str).stem + ":\n" + traceback.format_exc())


def collect(n_cells: int | None, workers: int) -> list[dict]:
    jobs: list[tuple[str, str]] = []
    for ds in DATASETS:
        paths = sorted((CLEAN_DIR / ds).glob("*.pkl"))
        if n_cells is not None:
            idx = sorted(set(np.linspace(0, len(paths) - 1, min(n_cells, len(paths))).round().astype(int)))
            paths = [paths[i] for i in idx]
        jobs.extend((str(p), ds) for p in paths)

    results: list[dict] = []
    if workers <= 1:
        for job in tqdm(jobs, desc="[Phase2] 셀별 특징 검증"):
            status, payload = _worker(job)
            if status == "ok" and payload is not None:
                results.append(payload)
            elif status == "err":
                print(f"\n  [ERR] {payload}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, job): job for job in jobs}
            with tqdm(total=len(jobs), desc=f"[Phase2] 셀별 특징 검증 (workers={workers})") as pbar:
                for fut in as_completed(futs):
                    status, payload = fut.result()
                    if status == "ok" and payload is not None:
                        results.append(payload)
                    elif status == "err":
                        print(f"\n  [ERR] {payload}")
                    pbar.update(1)
    return results


# ── 플랏 ──────────────────────────────────────────────────────────────────────

def _box_by_dataset(df: pd.DataFrame, col: str, out_path: Path, title: str, ylabel: str,
                     pct: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [df.loc[df["dataset"] == ds, col].dropna().values * (100 if pct else 1) for ds in DATASETS]
    bp = ax.boxplot(data, labels=DATASETS, showfliers=True, patch_artist=True)
    for patch, ds in zip(bp["boxes"], DATASETS):
        patch.set_facecolor(BATCH_COLORS[ds]); patch.set_alpha(0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cells", type=int, default=None, help="데이터셋별 샘플링 셀 수(기본 전체)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = collect(args.n_cells, args.workers)
    if not results:
        raise RuntimeError("유효한 셀이 하나도 없습니다.")

    raw_curves = {(r["dataset"], r["cell_id"]): (r.pop("_reg_cap_cycles"), r.pop("_reg_cap_vals"))
                  for r in results}
    df = pd.DataFrame(results).sort_values(["dataset", "cell_id"]).reset_index(drop=True)
    csv_path = OUT_DIR / "phase2_dataset_features.csv"
    df.to_csv(csv_path, index=False)

    # ── 콘솔 요약 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("  docs/SOC.md §6 Phase 2 — 데이터셋 특징 검증 요약 (평균, 데이터셋별)")
    print("=" * 96)
    summary_cols = [
        "n_cycles", "nonmono_ratio", "nonmono_sig_ratio", "missing_cycle_ratio", "max_cycle_gap",
        "frac_dis_reach_cutoff", "frac_dis_pass_hi_gate", "frac_chg_reach_full", "frac_chg_pass_hi_gate",
        "coulomb_eff_median", "v_dis_span_median", "v_chg_span_median", "knee_cycle_frac",
    ]
    print(df.groupby("dataset")[summary_cols].mean().to_string(float_format=lambda x: f"{x:.4f}"))
    print("-" * 96)
    print(f"  셀별 원시 CSV: {csv_path}")
    print("=" * 96)

    # ── (a) 단조성 ───────────────────────────────────────────────────────────
    _box_by_dataset(df, "nonmono_sig_ratio", OUT_DIR / "phase2_a_monotonicity.png",
                     "(a) 등록용량 비단조 비율 (직전 대비 +1% 초과 증가)",
                     "비단조 사이클 비율 (%)", pct=True)

    # ── (b) 결측 사이클 밀도 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title, pct in [
        (axes[0], "missing_cycle_ratio", "결측 사이클 비율", True),
        (axes[1], "max_cycle_gap", "최대 단일 결측폭 (사이클 수)", False),
    ]:
        data = [df.loc[df["dataset"] == ds, col].dropna().values * (100 if pct else 1) for ds in DATASETS]
        bp = ax.boxplot(data, labels=DATASETS, patch_artist=True)
        for patch, ds in zip(bp["boxes"], DATASETS):
            patch.set_facecolor(BATCH_COLORS[ds]); patch.set_alpha(0.6)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("(b) 결측 사이클 밀도")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase2_b_missing_cycles.png", dpi=150)
    plt.close(fig)

    # ── (c) 완전 도달 여부 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cutoff_col, gate_col, title in [
        (axes[0], "frac_dis_reach_cutoff", "frac_dis_pass_hi_gate", "방전"),
        (axes[1], "frac_chg_reach_full",   "frac_chg_pass_hi_gate", "충전"),
    ]:
        positions = np.arange(len(DATASETS))
        width = 0.35
        for offset, col, lbl, alpha in [(-1, cutoff_col, "진짜 완전도달", 0.8), (1, gate_col, "기존 hi_correlation 게이트 통과", 0.4)]:
            data = [df.loc[df["dataset"] == ds, col].dropna().values * 100 for ds in DATASETS]
            bp = ax.boxplot(data, positions=positions + offset * width / 2, widths=width * 0.9,
                             patch_artist=True, showfliers=False)
            for patch, ds in zip(bp["boxes"], DATASETS):
                patch.set_facecolor(BATCH_COLORS[ds]); patch.set_alpha(alpha)
        ax.set_xticks(positions); ax.set_xticklabels(DATASETS)
        ax.set_title(f"{title} — 진한색=컷오프 실도달, 연한색=기존 완화게이트 통과")
        ax.set_ylabel("비율 (%)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("(c) 완전 방전/충전 도달 비율 vs 기존 완화 게이트 통과율")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase2_c_full_reach.png", dpi=150)
    plt.close(fig)

    # ── (d) 쿨롱효율 ─────────────────────────────────────────────────────────
    _box_by_dataset(df, "coulomb_eff_median", OUT_DIR / "phase2_d_coulomb_efficiency.png",
                     "(d) 셀별 쿨롱효율 중앙값 (Qc/Qd)", "Qc/Qd")

    # ── (e) SOC 전이 범위 ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [(axes[0], "v_dis_span_median", "방전 전압 스윙 (V)"),
                            (axes[1], "v_chg_span_median", "충전 전압 스윙 (V)")]:
        data = [df.loc[df["dataset"] == ds, col].dropna().values for ds in DATASETS]
        bp = ax.boxplot(data, labels=DATASETS, patch_artist=True)
        for patch, ds in zip(bp["boxes"], DATASETS):
            patch.set_facecolor(BATCH_COLORS[ds]); patch.set_alpha(0.6)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("(e) SOC 전이 범위(전압 스윙 대리지표) — 클수록 0~100% 가까이 훑음")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase2_e_soc_range.png", dpi=150)
    plt.close(fig)

    # ── (f) knee-point ───────────────────────────────────────────────────────
    _box_by_dataset(df, "knee_cycle_frac", OUT_DIR / "phase2_f_knee_point.png",
                     "(f) knee-point 위치 (수명 대비 비율, 0=BOL~1=EOL)",
                     "knee_cycle / 전체 사이클범위", pct=True)

    # (f) 대표 셀 knee 곡선 예시 (데이터셋별 3셀)
    for ds in DATASETS:
        sub = df[(df["dataset"] == ds) & df["knee_cycle"].notna()].sort_values("cell_id")
        if sub.empty:
            continue
        n_ex = min(3, len(sub))
        idx = np.linspace(0, len(sub) - 1, n_ex).round().astype(int)
        example_cells = sub.iloc[idx]["cell_id"].tolist()

        fig, axes = plt.subplots(len(example_cells), 1, figsize=(9, 4 * len(example_cells)), squeeze=False)
        for ax, cid in zip(axes[:, 0], example_cells):
            cyc, cap = raw_curves[(ds, cid)]
            row = sub[sub["cell_id"] == cid].iloc[0]
            ax.plot(cyc, cap, color=BATCH_COLORS[ds], lw=1.0)
            if np.isfinite(row["knee_cycle"]):
                ax.axvline(row["knee_cycle"], color="red", ls="--", lw=1.2,
                            label=f"knee~{row['knee_cycle']:.0f} (frac={row['knee_cycle_frac']:.2f})")
            ax.set_title(f"[{ds}] {cid} — 등록용량 vs knee-point")
            ax.set_xlabel("사이클"); ax.set_ylabel("Ah")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"phase2_f_knee_examples_{ds}.png", dpi=150)
        plt.close(fig)

    print(f"\n[완료] 출력 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
