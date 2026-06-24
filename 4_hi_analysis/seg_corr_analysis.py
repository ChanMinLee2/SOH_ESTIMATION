"""
seg_corr_analysis.py

hi_features.pkl 에서 6개 세그먼트 시나리오별로
  1) features ↔ capacity_Ah (SOH) Spearman 상관계수 랭킹
  2) features 서로 간의 Spearman 상관행렬 (히트맵)
  3) 시나리오별 Top-7 HI 비교 — HI 종류에 20색 고정 할당 (시나리오 간 공통 feature 파악)
을 계산·시각화.

정답(target): capacity_Ah — 해당 사이클의 방전용량 (SOH 프록시).
              셀마다 initial capacity 가 다르므로 셀 내부 상관계수를 계산한 뒤 집계.

6 시나리오:
  discharge-high  (_s_hi)       방전 SoC 60~100%
  discharge-mid   (_s_mid)      방전 SoC 30~60%
  discharge-low   (_s_lo)       방전 SoC 0~30%
  charge-low      (_chg_s_lo)   충전 SoC 0~30%
  charge-mid      (_chg_s_mid)  충전 SoC 30~60%
  charge-high     (_chg_s_hi)   충전 SoC 60~100%

출력: 4_hi_analysis/seg_corr/<MMDD>/
  corr_rank.png        — 6 시나리오 × 상관계수 랭킹 바 차트
  corr_matrix.png      — 6 시나리오 × 20×20 feature 상관행렬 (히트맵)
  top7_cross.png       — 6 시나리오 × Top-7 HI 비교 (HI 아이덴티티 기준 20색)

사용:
  python 4_hi_analysis/seg_corr_analysis.py
  python 4_hi_analysis/seg_corr_analysis.py --pkl 4_hi_analysis/0622_1154_hi_features.pkl
  python 4_hi_analysis/seg_corr_analysis.py --dataset mit
  python 4_hi_analysis/seg_corr_analysis.py --min-cycles 10 --workers 8
"""

import argparse
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# ── 경로 ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
PKL_DEFAULT  = HERE / "hi_features.pkl"
OUT_BASE     = HERE / "seg_corr" / date.today().strftime("%m%d")

# ── 폰트 ─────────────────────────────────────────────────────────────────────
for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _f; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ── 시나리오 정의 ─────────────────────────────────────────────────────────────
SCENARIOS = [
    ("discharge-high", "_s_hi",      lambda c: c.endswith("_s_hi")      and "chg" not in c),
    ("discharge-mid",  "_s_mid",     lambda c: c.endswith("_s_mid")     and "chg" not in c),
    ("discharge-low",  "_s_lo",      lambda c: c.endswith("_s_lo")      and "chg" not in c),
    ("charge-low",     "_chg_s_lo",  lambda c: c.endswith("_chg_s_lo")),
    ("charge-mid",     "_chg_s_mid", lambda c: c.endswith("_chg_s_mid")),
    ("charge-high",    "_chg_s_hi",  lambda c: c.endswith("_chg_s_hi")),
]

SCENARIO_COLORS = {
    "discharge-high": "#1f77b4",
    "discharge-mid":  "#4daf4a",
    "discharge-low":  "#984ea3",
    "charge-high":    "#d62728",
    "charge-mid":     "#ff7f0e",
    "charge-low":     "#a65628",
}

# feature 이름에서 suffix 제거해 짧은 레이블 생성
def _short(col, suffix):
    return col.replace(suffix, "").replace("_chg", "")


# ─────────────────────────────────────────────────────────────────────────────
# 셀별 Spearman 상관계수 계산
# ─────────────────────────────────────────────────────────────────────────────

def _cell_corr(args):
    """워커: 단일 (cell_id, group_df) → feature별 Spearman ρ Series."""
    cell_key, sub, feat_cols, min_cycles = args
    cap = sub["capacity_Ah"].values
    if len(cap) < min_cycles:
        return cell_key, None
    result = {}
    for col in feat_cols:
        vals = sub[col].values
        mask = np.isfinite(vals) & np.isfinite(cap)
        if mask.sum() < min_cycles:
            result[col] = np.nan
        else:
            rho, _ = spearmanr(vals[mask], cap[mask])
            result[col] = float(rho)
    return cell_key, pd.Series(result)


def compute_corr_with_capacity(df: pd.DataFrame, feat_cols: list,
                                min_cycles: int = 5,
                                workers: int = 4) -> pd.DataFrame:
    """
    셀별 Spearman(feature, capacity_Ah) 계산.
    반환: DataFrame (index=cell_key, columns=feat_cols)
    """
    groups = list(df.groupby(["dataset", "cell_id"]))
    args   = [(key, sub[feat_cols + ["capacity_Ah"]], feat_cols, min_cycles)
              for key, sub in groups]

    rows = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_cell_corr, a) for a in args]
        for fut in futs:
            key, ser = fut.result()
            if ser is not None:
                rows[key] = ser

    return pd.DataFrame(rows).T  # shape: (n_cells, n_features)


def summarise(cell_corr_df: pd.DataFrame) -> pd.DataFrame:
    """mean / std / median / 유효셀수 집계."""
    return pd.DataFrame({
        "mean":   cell_corr_df.mean(skipna=True),
        "std":    cell_corr_df.std(skipna=True),
        "median": cell_corr_df.median(skipna=True),
        "n_valid":cell_corr_df.notna().sum(),
    }).sort_values("mean", key=abs, ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Feature ↔ Feature 상관행렬
# ─────────────────────────────────────────────────────────────────────────────

def feature_corr_matrix(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """전체 데이터 풀링 후 Spearman 상관행렬 (rank 변환 → Pearson)."""
    sub = df[feat_cols].dropna(how="all")
    # rank 변환 (spearman = pearson on ranks)
    ranked = sub.rank(method="average", na_option="keep")
    return ranked.corr(method="pearson", min_periods=20)


# ─────────────────────────────────────────────────────────────────────────────
# 플롯 1: 상관계수 랭킹 바 차트
# ─────────────────────────────────────────────────────────────────────────────

def plot_corr_rank(summaries: dict, out_path: Path, dataset_name: str = ""):
    n_sc   = len(summaries)
    n_cols = 3
    n_rows = (n_sc + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 7, n_rows * 9),
                             constrained_layout=True)
    ds_label = f"[{dataset_name}] " if dataset_name else ""
    fig.suptitle(
        f"{ds_label}세그먼트별 Feature ↔ 방전용량(capacity_Ah) |Spearman ρ| 랭킹\n"
        "(셀 내부 상관계수 절댓값 → 전체 셀 평균±std,  방향은 (+)/(-) 표기)",
        fontsize=13, fontweight="bold")

    axes_flat = np.array(axes).reshape(-1)
    for ax, (sc_name, summary_df, suffix) in zip(axes_flat, summaries.values()):
        color     = SCENARIO_COLORS.get(sc_name, "steelblue")
        feat_cols = summary_df.index.tolist()
        labels    = [_short(c, suffix) for c in feat_cols]
        means     = summary_df["mean"].values
        means_abs = np.abs(means)
        stds      = summary_df["std"].values

        ax.barh(labels[::-1], means_abs[::-1],
                xerr=stds[::-1], color=color,
                alpha=0.82, capsize=3, height=0.65,
                error_kw={"elinewidth": 1.0, "alpha": 0.6})

        ax.set_xlim(0, 1.15)
        ax.set_xlabel("|Spearman ρ| (mean across cells)", fontsize=9)
        ax.set_title(f"[{sc_name}]", fontsize=10, fontweight="bold",
                     color=color, pad=4)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.3)

        for i, (m, s) in enumerate(zip(means[::-1], stds[::-1])):
            sign_str = "(+)" if m >= 0 else "(-)"
            ax.text(abs(m) + 0.02, i, f"{abs(m):.3f} {sign_str}",
                    va="center", ha="left", fontsize=7, color="black")

    for ax in axes_flat[n_sc:]:
        ax.set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 플롯 2: Feature × Feature 상관행렬 히트맵
# ─────────────────────────────────────────────────────────────────────────────

def plot_corr_matrix(matrices: dict, out_path: Path, dataset_name: str = ""):
    n_sc   = len(matrices)
    n_cols = 3
    n_rows = (n_sc + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 7.5, n_rows * 7.5),
                             constrained_layout=True)
    ds_label = f"[{dataset_name}] " if dataset_name else ""
    fig.suptitle(
        f"{ds_label}세그먼트별 Feature × Feature Spearman 상관행렬\n"
        "(전체 데이터 풀링, rank 변환 후 Pearson 계산)",
        fontsize=13, fontweight="bold")

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = "RdBu_r"

    axes_flat = np.array(axes).reshape(-1)
    for ax, (sc_name, (corr_mat, suffix)) in zip(axes_flat, matrices.items()):
        labels = [_short(c, suffix) for c in corr_mat.columns]
        n      = len(labels)
        mat    = corr_mat.values

        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)

        # 셀 값 표시 (n<=15 일 때만)
        if n <= 20:
            for i in range(n):
                for j in range(n):
                    v = mat[i, j]
                    if not np.isfinite(v): continue
                    txt_c = "white" if abs(v) > 0.6 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5, color=txt_c)

        color = SCENARIO_COLORS.get(sc_name, "steelblue")
        ax.set_title(f"[{sc_name}]", fontsize=10, fontweight="bold",
                     color=color, pad=4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for ax in axes_flat[n_sc:]:
        ax.set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 플롯 3: 시나리오별 Top-N HI 비교 (HI 아이덴티티 20색 고정)
# ─────────────────────────────────────────────────────────────────────────────

def plot_top7_cross_scenario(summaries: dict, out_path: Path,
                              top_n: int = 7, dataset_name: str = ""):
    """
    6 시나리오 × Top-N HI 비교 플롯 — HI 7개를 x축에 배치.
    HI 종류(short name)별로 tab20 팔레트에서 고유 색상 할당 →
    같은 feature가 여러 시나리오에 등장할 때 동일 색으로 표시.
    """
    # ── 1. 전체 시나리오를 합쳐 고유 HI short-name 목록 수집 (등장 순서 유지)
    all_short: list = []
    seen: set = set()
    for _, (_, summ, suffix) in summaries.items():
        for feat in summ.index:
            sn = _short(feat, suffix)
            if sn not in seen:
                all_short.append(sn)
                seen.add(sn)

    # tab20: 20가지 고유색 — HI 하나당 색 하나 고정
    palette = matplotlib.colormaps["tab20"].colors
    hi_color = {sn: palette[i % 20] for i, sn in enumerate(all_short)}

    # ── 2. 서브플롯 구성 (HI x축, |ρ| y축 수직 바)
    n_sc   = len(summaries)
    n_cols = 3
    n_rows = (n_sc + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5.8, n_rows * 4.8),
        constrained_layout=True,
    )
    ds_label = f"[{dataset_name}] " if dataset_name else ""
    fig.suptitle(
        f"{ds_label}시나리오별 Top-{top_n} HI  (|ρ| 기준,  막대 위 수치·방향 표기)\n"
        "막대 색 = HI 종류 기준 — 같은 색 = 동일 feature",
        fontsize=13, fontweight="bold",
    )

    top_names_used: set = set()
    axes_flat = np.array(axes).reshape(-1)
    x_pos = np.arange(top_n)

    for ax, (sc_name, (_, summ, suffix)) in zip(axes_flat, summaries.items()):
        top       = summ.head(top_n)
        labels    = [_short(c, suffix) for c in top.index]
        means     = top["mean"].values
        means_abs = np.abs(means)
        stds      = top["std"].values
        colors    = [hi_color[lbl] for lbl in labels]
        top_names_used.update(labels)

        ax.bar(x_pos, means_abs, yerr=stds, color=colors,
               alpha=0.85, capsize=4, width=0.62,
               error_kw={"elinewidth": 1.2, "alpha": 0.55})

        # 막대 위: |ρ| 수치 + 방향 부호
        for xi, (m, m_abs, std) in enumerate(zip(means, means_abs, stds)):
            sign_str = "(+)" if m >= 0 else "(-)"
            ax.text(xi, m_abs + std + 0.03,
                    f"{m_abs:.3f}\n{sign_str}",
                    ha="center", va="bottom", fontsize=7,
                    fontweight="bold", linespacing=1.25)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("|Spearman ρ|", fontsize=8)
        sc_col = SCENARIO_COLORS.get(sc_name, "steelblue")
        ax.set_title(f"[{sc_name}]  Top-{top_n}",
                     fontsize=10, fontweight="bold", color=sc_col, pad=4)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.3, lw=0.7)
        ax.set_axisbelow(True)

    for ax in axes_flat[n_sc:]:
        ax.set_visible(False)

    # ── 3. 범례: Top-N에 등장한 HI만, all_short 순서 유지
    legend_handles = [
        Patch(facecolor=hi_color[sn], label=sn)
        for sn in all_short if sn in top_names_used
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 7),
        fontsize=8.5,
        framealpha=0.85,
        title="HI 아이덴티티 색상 범례",
        title_fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="세그먼트별 feature 상관분석")
    parser.add_argument("--pkl",        default=str(PKL_DEFAULT))
    parser.add_argument("--dataset",    default="all", choices=["all", "mit", "hust"],
                        help="분석 대상 데이터셋 (기본: all)")
    parser.add_argument("--min-cycles", type=int, default=5,
                        help="셀당 최소 유효 사이클 수 (기본: 5)")
    parser.add_argument("--workers",    type=int, default=8,
                        help="병렬 스레드 수 (기본: 8)")
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        # 날짜 붙은 최신 파일 자동 탐색
        candidates = sorted(HERE.glob("*hi_features*.pkl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"PKL 파일 없음: {pkl_path}"); return
        pkl_path = candidates[0]
        print(f"  자동 선택 PKL: {pkl_path.name}")

    print(f"=== 로드: {pkl_path} ===")
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    print(f"  shape: {df.shape}  |  셀 수: {df.groupby(['dataset','cell_id']).ngroups}")

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # 처리할 데이터셋 목록 결정
    if args.dataset == "all":
        datasets = sorted(df["dataset"].str.upper().unique().tolist())
    else:
        datasets = [args.dataset.upper()]

    for ds_name in datasets:
        df_ds = df[df["dataset"].str.upper() == ds_name].copy()
        if df_ds.empty:
            print(f"\n[SKIP] {ds_name}: 데이터 없음")
            continue

        print(f"\n{'='*60}")
        print(f"  데이터셋: {ds_name}  ({len(df_ds)}행, "
              f"{df_ds.groupby('cell_id').ngroups}셀)")
        print(f"{'='*60}")

        # ── 시나리오별 feature 컬럼 목록 추출 ───────────────────────────────
        scenarios_cols: dict[str, tuple] = {}
        for sc_name, suffix, fn in SCENARIOS:
            cols = [c for c in df_ds.columns if fn(c)]
            if not cols:
                continue
            scenarios_cols[sc_name] = (cols, suffix)
            print(f"  {sc_name}: {len(cols)}개 feature")

        # ── 1) Feature ↔ capacity_Ah 상관계수 ───────────────────────────────
        print(f"\n=== [{ds_name}] capacity_Ah 상관계수 계산 (workers={args.workers}) ===")
        summaries: dict = {}
        for sc_name, (cols, suffix) in tqdm(scenarios_cols.items(),
                                             desc=f"{ds_name} corr"):
            cell_corr = compute_corr_with_capacity(
                df_ds, cols, min_cycles=args.min_cycles, workers=args.workers)
            summ = summarise(cell_corr)
            summaries[sc_name] = (sc_name, summ, suffix)
            print(f"  {sc_name}: 유효셀={cell_corr.notna().all(axis=1).sum()}"
                  f"  top ρ={summ['mean'].iloc[0]:+.3f}({summ.index[0]})")

        rank_path = OUT_BASE / f"corr_rank_{ds_name.lower()}.png"
        print(f"\n=== [{ds_name}] Rank 플롯 저장 ===")
        plot_corr_rank(summaries, rank_path, dataset_name=ds_name)

        top7_path = OUT_BASE / f"top7_cross_{ds_name.lower()}.png"
        print(f"\n=== [{ds_name}] Top-7 Cross-Scenario 플롯 저장 ===")
        plot_top7_cross_scenario(summaries, top7_path, top_n=7, dataset_name=ds_name)

        # ── 2) Feature × Feature 상관행렬 ───────────────────────────────────
        print(f"\n=== [{ds_name}] Feature 상관행렬 계산 ===")
        matrices: dict = {}
        for sc_name, (cols, suffix) in tqdm(scenarios_cols.items(),
                                             desc=f"{ds_name} matrix"):
            corr_mat = feature_corr_matrix(df_ds, cols)
            matrices[sc_name] = (corr_mat, suffix)

        matrix_path = OUT_BASE / f"corr_matrix_{ds_name.lower()}.png"
        print(f"\n=== [{ds_name}] Matrix 플롯 저장 ===")
        plot_corr_matrix(matrices, matrix_path, dataset_name=ds_name)

        # ── 3) 텍스트 요약 출력 ─────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  [{ds_name}] Feature ↔ capacity_Ah 상관계수 요약 (상위 5개)")
        print(f"{'='*70}")
        for sc_name, (_, summ, suffix) in summaries.items():
            print(f"\n▶ [{sc_name}]")
            for feat, row in summ.head(5).iterrows():
                label = _short(feat, suffix)
                print(f"   {label:<22}  ρ_mean={row['mean']:+.4f}  "
                      f"std={row['std']:.4f}  median={row['median']:+.4f}  "
                      f"n={int(row['n_valid'])}")

    print(f"\n완료 — {OUT_BASE}/")


if __name__ == "__main__":
    main()
