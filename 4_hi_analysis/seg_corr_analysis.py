"""
seg_corr_analysis.py

285-HI 구조 기준 세그먼트 × 카테고리별 상관분석.

분석 단위: 6구간(dis_hi/mid/lo, chg_lo/mid/hi) × 3카테고리(Stat/Diff/LFP) = 18 그룹
정답(target): capacity_Ah (셀 내부 Spearman 상관계수 → 전체 셀 평균)

출력: 4_hi_analysis/seg_corr/<MMDD>/
  corr_rank_stat_{ds}.png    — 카테고리 A: 6구간 × 15 통계 HI 상관계수 랭킹
  corr_rank_diff_{ds}.png    — 카테고리 B: 6구간 × 15 미분 HI
  corr_rank_lfp_{ds}.png     — 카테고리 C: 6구간 × 15 LFP HI
  corr_matrix_stat_{ds}.png  — 카테고리 A: 6구간 × 15×15 feature 상관행렬
  corr_matrix_diff_{ds}.png  — 카테고리 B
  corr_matrix_lfp_{ds}.png   — 카테고리 C
  top_cross_{ds}.png         — 3카테고리 × 6구간 Top-5 HI 비교 (전체 요약)

사용:
  python 4_hi_analysis/seg_corr_analysis.py
  python 4_hi_analysis/seg_corr_analysis.py --dataset mit
  python 4_hi_analysis/seg_corr_analysis.py --min-cycles 5 --workers 8
"""

import argparse
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from hi_correlation import (
    ALL_SEGS,
    HI_GROUPS,
    HI_LABELS,
    STAT_KEYS,
    DIFF_KEYS,
    LFP_KEYS,
)

# ── 경로 ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
PKL_DEFAULT = HERE / "hi_features.pkl"
OUT_BASE    = HERE / "seg_corr" / date.today().strftime("%m%d")

# ── 폰트 ─────────────────────────────────────────────────────────────────────
for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ── 세그먼트 색상 (방전=파랑 계열, 충전=주황 계열) ──────────────────────────
SEG_COLORS = {
    "dis_hi":  "#1a5276",
    "dis_mid": "#2980b9",
    "dis_lo":  "#85c1e9",
    "chg_lo":  "#f0b27a",
    "chg_mid": "#e67e22",
    "chg_hi":  "#a04000",
}
SEG_LABELS = {
    "dis_hi":  "dis_hi\n(SoC 60–100%)",
    "dis_mid": "dis_mid\n(SoC 30–60%)",
    "dis_lo":  "dis_lo\n(SoC 0–30%)",
    "chg_lo":  "chg_lo\n(SoC 0–40%)",
    "chg_mid": "chg_mid\n(SoC 40–70%)",
    "chg_hi":  "chg_hi\n(SoC 70–100%)",
}

# ── 카테고리 메타 ─────────────────────────────────────────────────────────────
CATEGORIES = [
    ("Stat", "카테고리 A: 통계 기반",      "corr_rank_stat",   "corr_matrix_stat",   STAT_KEYS),
    ("Diff", "카테고리 B: 미분 기반",      "corr_rank_diff",   "corr_matrix_diff",   DIFF_KEYS),
    ("LFP",  "카테고리 C: LFP 특징 기반", "corr_rank_lfp",    "corr_matrix_lfp",    LFP_KEYS),
]

# tab20 팔레트 (15개 feature 색 고정)
_TAB20 = matplotlib.colormaps["tab20"].colors


def _feat_palette(keys: list) -> dict:
    """feature key → 고정 색상 (tab20)."""
    return {k: _TAB20[i % 20] for i, k in enumerate(keys)}


# ─────────────────────────────────────────────────────────────────────────────
# 상관계수 계산
# ─────────────────────────────────────────────────────────────────────────────

def _cell_corr(args):
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


def compute_corr(df: pd.DataFrame, feat_cols: list,
                 min_cycles: int = 5, workers: int = 4) -> pd.DataFrame:
    """셀별 Spearman(feature, capacity_Ah). 반환: (n_cells, n_feats)"""
    groups = list(df.groupby(["dataset", "cell_id"]))
    args   = [(key, sub[feat_cols + ["capacity_Ah"]], feat_cols, min_cycles)
              for key, sub in groups]
    rows = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for key, ser in ex.map(_cell_corr, args):
            if ser is not None:
                rows[key] = ser
    return pd.DataFrame(rows).T


def summarise(cell_corr_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "mean":    cell_corr_df.mean(skipna=True),
        "std":     cell_corr_df.std(skipna=True),
        "median":  cell_corr_df.median(skipna=True),
        "n_valid": cell_corr_df.notna().sum(),
    }).sort_values("mean", key=abs, ascending=False)


def feature_corr_matrix(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    sub    = df[feat_cols].dropna(how="all")
    ranked = sub.rank(method="average", na_option="keep")
    return ranked.corr(method="pearson", min_periods=20)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: 상관계수 랭킹 바 차트 (6 seg × 1 cat)
# ─────────────────────────────────────────────────────────────────────────────

def plot_corr_rank(seg_summaries: dict, out_path: Path,
                   cat_title: str, dataset_name: str = ""):
    """6 세그먼트 서브플롯 — feature ↔ capacity_Ah |ρ| 랭킹 바.

    seg_summaries: {seg: (summary_df, hi_keys)}
    """
    n_segs = len(seg_summaries)
    n_cols = 3
    n_rows = (n_segs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 7, n_rows * 9),
                              constrained_layout=True)
    ds_lbl = f"[{dataset_name}] " if dataset_name else ""
    fig.suptitle(
        f"{ds_lbl}{cat_title}\n"
        "Feature ↔ capacity_Ah  |Spearman ρ| 랭킹  (셀 내부 ρ 평균±std, 방향 부호 표기)",
        fontsize=13, fontweight="bold",
    )

    axes_flat = np.array(axes).reshape(-1)
    for ax, (seg, (summ, _hi_keys)) in zip(axes_flat, seg_summaries.items()):
        color     = SEG_COLORS.get(seg, "steelblue")
        feat_cols = summ.index.tolist()
        labels    = [HI_LABELS.get(k, k) for k in feat_cols]
        means     = summ["mean"].values
        means_abs = np.abs(means)
        stds      = summ["std"].values

        ax.barh(labels[::-1], means_abs[::-1],
                xerr=stds[::-1], color=color,
                alpha=0.82, capsize=3, height=0.65,
                error_kw={"elinewidth": 1.0, "alpha": 0.6})

        for i, (m, s) in enumerate(zip(means[::-1], stds[::-1])):
            ax.text(abs(m) + 0.02, i,
                    f"{abs(m):.3f} {'(+)' if m >= 0 else '(-)'}",
                    va="center", ha="left", fontsize=7)

        ax.set_xlim(0, 1.2)
        ax.set_xlabel("|Spearman ρ|", fontsize=9)
        ax.set_title(SEG_LABELS.get(seg, seg), fontsize=10,
                     fontweight="bold", color=color, pad=4)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.3)

    for ax in axes_flat[n_segs:]:
        ax.set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Feature × Feature 상관행렬 히트맵 (6 seg × 1 cat)
# ─────────────────────────────────────────────────────────────────────────────

def plot_corr_matrix(seg_matrices: dict, out_path: Path,
                     cat_title: str, dataset_name: str = ""):
    """6 세그먼트 서브플롯 — 15×15 feature 상관행렬.

    seg_matrices: {seg: (corr_df, hi_keys)}
    """
    n_segs = len(seg_matrices)
    n_cols = 3
    n_rows = (n_segs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 7.5, n_rows * 7.5),
                              constrained_layout=True)
    ds_lbl = f"[{dataset_name}] " if dataset_name else ""
    fig.suptitle(
        f"{ds_lbl}{cat_title}\n"
        "Feature × Feature Spearman 상관행렬  (전체 풀링, rank→Pearson)",
        fontsize=13, fontweight="bold",
    )

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = "RdBu_r"

    axes_flat = np.array(axes).reshape(-1)
    for ax, (seg, (corr_mat, _)) in zip(axes_flat, seg_matrices.items()):
        labels = [HI_LABELS.get(c, c) for c in corr_mat.columns]
        n      = len(labels)
        mat    = corr_mat.values

        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)

        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center", fontsize=5.5,
                        color="white" if abs(v) > 0.6 else "black")

        color = SEG_COLORS.get(seg, "steelblue")
        ax.set_title(SEG_LABELS.get(seg, seg), fontsize=10,
                     fontweight="bold", color=color, pad=4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for ax in axes_flat[n_segs:]:
        ax.set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: 3카테고리 × 6구간 Top-5 비교 (전체 요약)
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_cross(all_summaries: dict, out_path: Path,
                   top_n: int = 5, dataset_name: str = ""):
    """3 × 6 그리드 — rows=카테고리(Stat/Diff/LFP), cols=세그먼트.

    all_summaries: {cat: {seg: (summary_df, hi_keys)}}
    """
    cats     = list(all_summaries.keys())      # ["Stat","Diff","LFP"]
    segs     = [seg for _, _, seg, _ in ALL_SEGS]  # 6 segments
    n_cats   = len(cats)
    n_segs   = len(segs)

    # feature 색상: 카테고리별 15개 feature에 tab20 고정 매핑
    cat_palettes = {}
    for cat, _, _, _, base_keys in CATEGORIES:
        cat_palettes[cat] = {k: _TAB20[i % 20] for i, k in enumerate(base_keys)}

    fig, axes = plt.subplots(
        n_cats, n_segs,
        figsize=(n_segs * 4.2, n_cats * 4.0),
        constrained_layout=True,
    )
    ds_lbl = f"[{dataset_name}] " if dataset_name else ""
    fig.suptitle(
        f"{ds_lbl}카테고리 × 세그먼트 Top-{top_n} HI  (|ρ| 기준)\n"
        "행=카테고리, 열=세그먼트,  막대 색=feature 아이덴티티",
        fontsize=12, fontweight="bold",
    )

    x_pos = np.arange(top_n)
    for ri, (cat, cat_title_short, _, _, base_keys) in enumerate(CATEGORIES):
        # base_keys 인덱스로 색 결정 (stat_v_mean_* → v_mean → palette)
        palette = cat_palettes[cat]

        for ci, seg in enumerate(segs):
            ax = axes[ri, ci]
            summ, hi_keys = all_summaries.get(cat, {}).get(seg, (None, []))
            if summ is None or len(summ) == 0:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_xticks([])
                continue

            top       = summ.head(top_n)
            feat_keys = top.index.tolist()
            # strip prefix+suffix → base key for color lookup
            base      = [k.split("_", 1)[1].rsplit("_", 2)[0]
                         if k.count("_") >= 3 else k
                         for k in feat_keys]
            labels    = [HI_LABELS.get(k, k) for k in feat_keys]
            means     = top["mean"].values
            means_abs = np.abs(means)
            stds      = top["std"].values
            colors    = [palette.get(b, "#888888") for b in base]

            ax.bar(x_pos, means_abs, yerr=stds, color=colors,
                   alpha=0.85, capsize=3, width=0.62,
                   error_kw={"elinewidth": 1.0, "alpha": 0.5})

            for xi, (m, ma, std) in enumerate(zip(means, means_abs, stds)):
                ax.text(xi, ma + std + 0.04,
                        f"{ma:.2f}\n{'(+)' if m >= 0 else '(-)'}",
                        ha="center", va="bottom", fontsize=6,
                        fontweight="bold", linespacing=1.2)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
            ax.set_ylim(0, 1.3)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.3, lw=0.6)
            ax.set_axisbelow(True)

            # 열 제목 (첫 행에만)
            if ri == 0:
                ax.set_title(SEG_LABELS.get(seg, seg), fontsize=8.5,
                             fontweight="bold",
                             color=SEG_COLORS.get(seg, "black"), pad=3)
            # 행 레이블 (첫 열에만)
            if ci == 0:
                ax.set_ylabel(f"{cat_title_short}\n|ρ|", fontsize=8.5)

    # 카테고리별 범례 (각 카테고리의 feature 색상)
    legend_handles = []
    for cat, _, _, _, base_keys in CATEGORIES:
        legend_handles.append(Patch(color="white", label=f"─ {cat} ─"))
        for bk in base_keys:
            col = cat_palettes[cat][bk]
            lbl = HI_LABELS.get(f"stat_{bk}_dis_hi", bk)  # Stat 기준 레이블 조회
            legend_handles.append(Patch(facecolor=col, label=lbl))

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=16,
        fontsize=7,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="285-HI 세그먼트 × 카테고리 상관분석")
    parser.add_argument("--pkl",        default=str(PKL_DEFAULT))
    parser.add_argument("--dataset",    default="all", choices=["all", "mit", "hust"])
    parser.add_argument("--min-cycles", type=int, default=5)
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--top-n",      type=int, default=5,
                        help="top_cross 플롯 상위 HI 개수 (기본: 5)")
    args = parser.parse_args()

    # ── PKL 로드 ──────────────────────────────────────────────────────────────
    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        candidates = sorted(HERE.glob("*hi_features*.pkl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"PKL 파일 없음: {pkl_path}"); return
        pkl_path = candidates[0]
        print(f"  자동 선택 PKL: {pkl_path.name}")

    print(f"=== 로드: {pkl_path} ===")
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")
    print(f"  rows={len(df):,}  셀={df.groupby(['dataset','cell_id']).ngroups}")

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    datasets = (sorted(df["dataset"].unique().tolist())
                if args.dataset == "all" else [args.dataset.upper()])

    for ds_name in datasets:
        df_ds = df[df["dataset"] == ds_name].copy()
        if df_ds.empty:
            print(f"\n[SKIP] {ds_name}: 데이터 없음"); continue

        print(f"\n{'='*60}")
        print(f"  데이터셋: {ds_name}  ({len(df_ds):,}행, "
              f"{df_ds.groupby('cell_id').ngroups}셀)")
        print(f"{'='*60}")

        # all_summaries[cat][seg] = (summary_df, hi_keys)
        # all_matrices[cat][seg]  = (corr_df, hi_keys)
        all_summaries: dict = {cat: {} for cat, *_ in CATEGORIES}
        all_matrices:  dict = {cat: {} for cat, *_ in CATEGORIES}

        for cat, cat_title, rank_stem, matrix_stem, _base_keys in CATEGORIES:
            print(f"\n─── {cat_title} ───")
            for _, _, seg, seg_lbl in ALL_SEGS:
                group_key = f"{seg} — {cat}"
                hi_keys   = HI_GROUPS.get(group_key, [])
                avail     = [k for k in hi_keys if k in df_ds.columns]
                if not avail:
                    print(f"  [{seg}] 컬럼 없음, SKIP")
                    continue

                # 상관계수
                cell_corr = compute_corr(df_ds, avail,
                                         min_cycles=args.min_cycles,
                                         workers=args.workers)
                summ = summarise(cell_corr)
                all_summaries[cat][seg] = (summ, avail)
                n_valid = int(cell_corr.notna().all(axis=1).sum())
                top_lbl = HI_LABELS.get(summ.index[0], summ.index[0])
                print(f"  [{seg}] 유효셀={n_valid}  "
                      f"top ρ={summ['mean'].iloc[0]:+.3f} ({top_lbl})")

                # feature 상관행렬
                corr_mat = feature_corr_matrix(df_ds, avail)
                all_matrices[cat][seg] = (corr_mat, avail)

            # ── Plot 1: Rank ──────────────────────────────────────────────────
            if all_summaries[cat]:
                rank_path = OUT_BASE / f"{rank_stem}_{ds_name.lower()}.png"
                print(f"\n  [Plot] Rank  → {rank_path.name}")
                plot_corr_rank(all_summaries[cat], rank_path,
                               cat_title, dataset_name=ds_name)

            # ── Plot 2: Matrix ────────────────────────────────────────────────
            if all_matrices[cat]:
                mat_path = OUT_BASE / f"{matrix_stem}_{ds_name.lower()}.png"
                print(f"  [Plot] Matrix → {mat_path.name}")
                plot_corr_matrix(all_matrices[cat], mat_path,
                                 cat_title, dataset_name=ds_name)

        # ── Plot 3: Top-N Cross ───────────────────────────────────────────────
        top_path = OUT_BASE / f"top_cross_{ds_name.lower()}.png"
        print(f"\n  [Plot] Top-{args.top_n} Cross → {top_path.name}")
        plot_top_cross(all_summaries, top_path,
                       top_n=args.top_n, dataset_name=ds_name)

        # ── 텍스트 요약 ──────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  [{ds_name}] Top-3 HI 요약 (|ρ| 기준)")
        print(f"{'='*70}")
        for cat, cat_title, *_ in CATEGORIES:
            print(f"\n▶ {cat_title}")
            for seg, (summ, _) in all_summaries[cat].items():
                seg_lbl = SEG_LABELS.get(seg, seg).replace("\n", " ")
                top3 = summ.head(3)
                items = "  |  ".join(
                    f"{HI_LABELS.get(k,k)} {row['mean']:+.3f}"
                    for k, row in top3.iterrows()
                )
                print(f"  {seg_lbl:<22} {items}")

    print(f"\n완료 — {OUT_BASE}/")


if __name__ == "__main__":
    main()
