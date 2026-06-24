"""
diagnose_shape_outliers.py

[진단 전용 — 데이터는 수정하지 않음]

모든 셀의 방전/충전 사이클에 대해 "기준 곡선 대비 형상 편차"를 계산하여
형상(개형)이 다른 이상 사이클이 얼마나 되는지 분포를 진단한다.

아이디어 (방법1: Rolling reference curve):
  1. 각 사이클의 V를 공통 q_frac 격자(GRID)에 보간
  2. 인접 사이클들의 중앙값 곡선(rolling median, window=W)을 국소 기준으로 삼음
       → 노화에 따른 정상적 개형 변화는 기준이 따라가므로 통과,
         갑자기 튀는 사이클만 편차가 커진다
  3. 각 사이클 편차 — 두 지표:
       dev     = RMSE(V_cyc - V_ref)        → 전체적 형상 붕괴에 민감
       max_dev = max|V_cyc - V_ref|         → 국소 돌출/글리치에 민감
  4. 셀 내부 robust z = (x - median) / (1.4826 * MAD)  를 각 지표에 적용 (z, z_max)
       → z 또는 z_max 가 큰 사이클 = 형상 이상 후보
         (RMSE만으로는 국소적으로만 튀는 사이클을 놓치므로 max_dev 병행)
  + 평탄/체류 지표(편차와 무관, 절대 기준):
       v_span    = max(V_cyc) - min(V_cyc)  → 곡선 전체가 한 전압에 평탄한 경우 작다.
       frac_high = mean(V_cyc >= v_high)    → 3.6V 부근에 q_frac 상당 구간을
         머무는("3.6 유지") 사이클을 잡는다. 초반엔 낮게 시작해 v_span 은 정상이라
         편차/ v_span 으로는 못 잡는 유형을 frac_high 의 "절대" 상한으로 검출한다.

출력:
  - 콘솔: σ 임계값별 이상 사이클 개수 (z 또는 z_max 기준, 방전/충전 각각)
          + 충전 평탄(v_span 작음) / 고전압 체류(frac_high 큼) 후보
  - outputs/shape_outlier_report.csv : cell, phase, cycle, dev, max_dev, v_span, frac_high, z, z_max
  - outputs/shape_outlier_hist.png   : z / z_max 분포 히스토그램

사용:
  python 2_preprocess/diagnose_shape_outliers.py                 # MIT 전체
  python 2_preprocess/diagnose_shape_outliers.py --dataset all   # MIT + HUST
  python 2_preprocess/diagnose_shape_outliers.py --window 15 --grid 100
  python 2_preprocess/diagnose_shape_outliers.py --top 40        # 상위 N개 콘솔 출력
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_ROOT = PROJECT_ROOT / "data_unified"
OUTPUT_DIR   = Path(__file__).resolve().parent / "outputs"

PHASES = ["discharge", "charge"]
# 진단에서 검토할 후보 임계값 (robust z)
SIGMA_LEVELS = [3.0, 4.0, 5.0, 6.0]


def compute_qfrac(phase_df: pd.DataFrame):
    """phase_df → (q_frac, v). plot_cell_cycles.py 와 동일 로직."""
    if len(phase_df) < 10:
        return None, None
    t  = phase_df["time_s"].values.astype(float)
    v  = phase_df["voltage_V"].values.astype(float)
    i  = np.abs(phase_df["current_A"].values.astype(float))
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    q_cum = np.cumsum(i * dt) / 3600.0
    q_tot = float(q_cum[-1])
    if q_tot < 0.05:
        return None, None
    return q_cum / q_tot, v


def _robust_z(x: np.ndarray) -> np.ndarray:
    """MAD 기반 robust z-score."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-9 else (np.std(x) or 1.0)
    return (x - med) / scale


def shape_deviation(df: pd.DataFrame, phase: str, grid: np.ndarray, window: int,
                    v_high: float = 3.5):
    """주어진 phase 의 사이클별 형상 편차 지표 계산.

    dev     : 기준 곡선 대비 RMSE (전체적 형상 붕괴에 민감)
    max_dev : 기준 곡선 대비 최대 점별 |편차| (국소 돌출/글리치에 민감)
    v_span  : 그 사이클 자체의 전압 변동폭 max(V)-min(V)
              → 곡선 "전체"가 한 전압에 평탄한 경우에 작아진다.
    frac_high : V >= v_high 인 q_frac 격자 비율 (고전압 체류 비율)
              → 초반엔 낮게 시작했다가 3.6V 부근에 도달해 q_frac 상당 구간을
                거기 머무는 사이클을 잡는다. 정상 충전은 마지막에 잠깐만 닿으므로
                작고, 3.6 체류형은 크다. (v_span 으로는 못 잡는 유형)
    z / z_max : dev / max_dev 의 셀 내부 robust z-score

    Returns:
        DataFrame(columns=[cycle, dev, max_dev, v_span, frac_high, z, z_max])
    """
    rows, mat = [], []
    # phase로 1회 필터 후 groupby — 사이클마다 전체 df 스캔 방지
    for cyc, cyc_df in df[df["phase"] == phase].groupby("cycle"):
        q_frac, v = compute_qfrac(cyc_df)
        if q_frac is None:
            continue
        # q_frac 단조 증가 보장 (보간 안정성)
        order = np.argsort(q_frac)
        v_interp = np.interp(grid, q_frac[order], v[order])
        rows.append(cyc)
        mat.append(v_interp)

    if len(rows) < max(window, 5):
        return pd.DataFrame(columns=["cycle", "dev", "max_dev", "v_span",
                                     "frac_high", "z", "z_max"])

    M = pd.DataFrame(mat, index=rows)                      # row=cycle, col=grid
    ref = M.rolling(window=window, center=True, min_periods=3).median()
    diff = (M - ref)
    dev      = np.sqrt((diff ** 2).mean(axis=1)).values   # 사이클별 RMSE
    max_dev  = diff.abs().max(axis=1).values              # 사이클별 최대 점별 편차
    v_span   = (M.max(axis=1) - M.min(axis=1)).values     # 사이클 자체 전압 변동폭
    frac_high = (M >= v_high).mean(axis=1).values         # 고전압 체류 q_frac 비율

    return pd.DataFrame({
        "cycle":     rows,
        "dev":       dev,
        "max_dev":   max_dev,
        "v_span":    v_span,
        "frac_high": frac_high,
        "z":         _robust_z(dev),
        "z_max":     _robust_z(max_dev),
    })


def iter_pkls(dataset: str):
    dirs = []
    if dataset in ("mit", "all"):
        dirs.append(UNIFIED_ROOT / "MIT")
    if dataset in ("hust", "all"):
        dirs.append(UNIFIED_ROOT / "HUST")
    for d in dirs:
        if not d.exists():
            print(f"  [SKIP] 폴더 없음: {d}")
            continue
        for p in sorted(d.glob("*.pkl")):
            if p.stem in ("README",):
                continue
            yield d.name, p


def main():
    parser = argparse.ArgumentParser(description="형상 이상 사이클 진단 (수정 없음)")
    parser.add_argument("--dataset", default="mit", choices=["mit", "hust", "all"])
    parser.add_argument("--window", type=int, default=11,
                        help="기준 곡선 rolling median 윈도우 (기본: 11)")
    parser.add_argument("--grid", type=int, default=100,
                        help="q_frac 보간 격자 점 수 (기본: 100)")
    parser.add_argument("--top", type=int, default=30,
                        help="콘솔에 출력할 편차 상위 후보 수 (기본: 30)")
    parser.add_argument("--v-high", type=float, default=3.5,
                        help="frac_high 계산용 고전압 기준 (기본: 3.5V, 3.6 체류 검출)")
    args = parser.parse_args()

    grid = np.linspace(0.0, 1.0, args.grid)
    records = []

    pkls = list(iter_pkls(args.dataset))
    print(f"=== 형상 이상 진단 (dataset={args.dataset}, "
          f"window={args.window}, grid={args.grid}) ===")
    print(f"  대상 셀: {len(pkls)}개\n")

    for idx, (ds, pkl) in enumerate(pkls, 1):
        try:
            with open(pkl, "rb") as f:
                raw = pickle.load(f)
            meta, df = raw["meta"], raw["cycles"]
            cell_id = meta.get("cell_id", pkl.stem)
        except Exception as e:
            print(f"  [ERR] {pkl.stem}: {e}")
            continue

        for phase in PHASES:
            res = shape_deviation(df, phase, grid, args.window, args.v_high)
            for _, r in res.iterrows():
                records.append({
                    "dataset": ds, "cell_id": cell_id, "phase": phase,
                    "cycle": int(r["cycle"]),
                    "dev": r["dev"], "max_dev": r["max_dev"],
                    "v_span": r["v_span"], "frac_high": r["frac_high"],
                    "z": r["z"], "z_max": r["z_max"],
                })
        if idx % 25 == 0:
            print(f"  ... {idx}/{len(pkls)} 처리")

    if not records:
        print("처리된 사이클이 없습니다.")
        return

    rep = pd.DataFrame(records)

    # ── 콘솔 요약: σ 임계값별 이상 사이클 개수 ──────────────────────────────
    print(f"\n=== 요약 (총 {len(rep)} 사이클 분석) ===")
    for phase in PHASES:
        sub = rep[rep["phase"] == phase]
        if sub.empty:
            continue
        print(f"\n[{phase}] 분석 사이클 {len(sub)}개")
        for sg in SIGMA_LEVELS:
            mask = (sub["z"] > sg) | (sub["z_max"] > sg)   # RMSE 또는 국소편차
            n_cyc = int(mask.sum())
            n_cell = sub.loc[mask, "cell_id"].nunique()
            n_only_max = int(((sub["z"] <= sg) & (sub["z_max"] > sg)).sum())
            print(f"   z>{sg:>3.1f} 또는 z_max>{sg:>3.1f} : {n_cyc:>5d} 사이클  "
                  f"({n_cell} 셀)  = {100*n_cyc/len(sub):.2f}%  "
                  f"[국소편차로만 추가: {n_only_max}]")

    # ── 편차 상위 후보 (RMSE 기준 / 국소편차 기준 각각) ────────────────────
    cols = ["dataset", "cell_id", "phase", "cycle", "dev", "max_dev",
            "v_span", "frac_high", "z", "z_max"]
    fmt = lambda x: f"{x:.4f}"
    print(f"\n=== RMSE(z) 상위 {args.top} 후보 ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(rep.sort_values("z", ascending=False).head(args.top)[cols]
              .to_string(index=False, float_format=fmt))
    print(f"\n=== 국소편차(z_max) 상위 {args.top} 후보 ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(rep.sort_values("z_max", ascending=False).head(args.top)[cols]
              .to_string(index=False, float_format=fmt))

    # ── 평탄(flat) 후보: 전압 변동폭 v_span 이 작은 충전 사이클 ─────────────
    chg = rep[rep["phase"] == "charge"]
    if not chg.empty:
        print(f"\n=== 충전 평탄(v_span 작음) 하위 {args.top} 후보 "
              f"= 곡선 전체 평탄선 의심 ===")
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(chg.sort_values("v_span", ascending=True).head(args.top)[cols]
                  .to_string(index=False, float_format=fmt))

        # ── 3.6V 체류형: frac_high 큰 충전 사이클 ─────────────────────────
        print(f"\n=== 충전 고전압(V≥{args.v_high}) 체류비율 frac_high 상위 {args.top} 후보 "
              f"= 3.6V 장구간 유지 의심 ===")
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(chg.sort_values("frac_high", ascending=False).head(args.top)[cols]
                  .to_string(index=False, float_format=fmt))

    # ── 저장 ───────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "shape_outlier_report.csv"
    rep.sort_values(["dataset", "cell_id", "phase", "cycle"]).to_csv(csv_path, index=False)
    print(f"\n  전체 리포트: {csv_path}")

    _plot_hist(rep)


def _plot_hist(rep: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib 없음")
        return

    # 한글 폰트 설정 — 반드시 플롯 생성 전에 적용
    for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = _f; break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    # 행: 지표(z=RMSE, z_max=국소편차) / 열: phase
    metrics = [("z", "RMSE"), ("z_max", "국소편차 max|ΔV|")]
    fig, axes = plt.subplots(len(metrics), len(PHASES),
                             figsize=(12, 8), constrained_layout=True, squeeze=False)
    for r, (zcol, mlabel) in enumerate(metrics):
        for c, phase in enumerate(PHASES):
            ax = axes[r][c]
            sub = rep[rep["phase"] == phase]
            if sub.empty:
                continue
            zc = np.clip(sub[zcol].values, -2, 12)
            ax.hist(zc, bins=80, color="steelblue", alpha=0.8)
            for sg in SIGMA_LEVELS:
                ax.axvline(sg, color="red", ls="--", lw=0.8)
                ax.text(sg, ax.get_ylim()[1]*0.9, f"{sg}", color="red", fontsize=8,
                        ha="center")
            ax.set_yscale("log")
            ax.set_title(f"{phase}  {mlabel}  robust-z 분포")
            ax.set_xlabel("robust z  (clip -2~12)")
            ax.set_ylabel("사이클 수 (log)")
            ax.grid(True, alpha=0.3)

    out = OUTPUT_DIR / "shape_outlier_hist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  분포 히스토그램: {out}")


if __name__ == "__main__":
    main()
