"""
seg_corr_analysis.py

HI 구조 기준 세그먼트 × 카테고리별 상관분석 + MI 분석 + Scenario 구분 능력 분석.

분석 단위: N구간(축별) × 4카테고리(Stat/Diff/LFP/Morph)
정답(target): capacity_Ah (셀 내부 Spearman 상관계수 → 전체 셀 평균)

출력: 4_hi_analysis/seg_corr/<MMDD>[_<axis>]/
  corr_rank_stat_{ds}.png    — 카테고리 A: N구간 × 통계 HI 상관계수 랭킹
  corr_rank_diff_{ds}.png    — 카테고리 B: N구간 × 미분 HI
  corr_rank_lfp_{ds}.png     — 카테고리 C: N구간 × LFP HI
  corr_matrix_stat_{ds}.png  — 카테고리 A: N구간 × feature 상관행렬
  corr_matrix_diff_{ds}.png  — 카테고리 B
  corr_matrix_lfp_{ds}.png   — 카테고리 C
  top_cross_{ds}.png         — 4카테고리 × N구간 Top-5 HI 비교 (전체 요약)
  scenario_eps2_{ds}.png     — HI 컨셉별 Scenario 구분 능력 ε²
  mi_rank_category_{ds}.png  — 4카테고리별 HI Mutual Information 랭킹
  mi_vs_rho_{ds}.png         — MI vs |ρ| 산점도 (⚠ 항목 식별)
  corr_3d_{ds}.png           — 시나리오별 |ρ| 3D 산점도

사용:
  python 4_hi_analysis/seg_corr_analysis.py
  python 4_hi_analysis/seg_corr_analysis.py --seg-axis protocol
  python 4_hi_analysis/seg_corr_analysis.py --seg-axis vwindow --axis-config '{"n_windows": 4}'
  python 4_hi_analysis/seg_corr_analysis.py --dataset mit --min-cycles 5 --workers 8
"""

import argparse
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import kruskal as _kruskal
from scipy.stats import spearmanr

try:
    from sklearn.feature_selection import mutual_info_regression as _mir
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print("[경고] scikit-learn 미설치 — MI 계산 건너뜀")

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
    HI_GROUPS,
    HI_LABELS,
    STAT_KEYS,
    DIFF_KEYS,
    LFP_KEYS,
    MORPH_KEYS,
)

# ── 경로 ─────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).resolve().parent

# ── 폰트 ─────────────────────────────────────────────────────────────────────
for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ── 카테고리 색상 ─────────────────────────────────────────────────────────────
CAT_COLORS = {
    "Stat":  "#2980b9",
    "Diff":  "#e67e22",
    "LFP":   "#27ae60",
    "Morph": "#8e44ad",
}

# ── 세그먼트 색상 / 레이블 — 기본값(qfrac). main()에서 동적 재빌드됨 ───────────
SEG_COLORS: dict[str, str] = {
    "dis_hi":  "#1a5276",
    "dis_mid": "#2980b9",
    "dis_lo":  "#85c1e9",
    "chg_lo":  "#f0b27a",
    "chg_mid": "#e67e22",
    "chg_hi":  "#a04000",
}
SEG_LABELS: dict[str, str] = {
    "dis_hi":  "dis_hi\n(SoC 60–100%)",
    "dis_mid": "dis_mid\n(SoC 30–60%)",
    "dis_lo":  "dis_lo\n(SoC 0–30%)",
    "chg_lo":  "chg_lo\n(SoC 0–40%)",
    "chg_mid": "chg_mid\n(SoC 40–70%)",
    "chg_hi":  "chg_hi\n(SoC 70–100%)",
}

# 활성 세그먼트 목록 (순서 유지) + 방향별 분류 — main()에서 재빌드
_ACTIVE_SEGS: list[str] = list(SEG_COLORS.keys())
_MODE_SEGS: dict[str, tuple] = {
    "discharge": ("dis_hi", "dis_mid", "dis_lo"),
    "charge":    ("chg_lo", "chg_mid", "chg_hi"),
}

# ── 카테고리 메타 ─────────────────────────────────────────────────────────────
CATEGORIES = [
    ("Stat",  "카테고리 A: 통계 기반",        "corr_rank_stat",   "corr_matrix_stat",   STAT_KEYS),
    ("Diff",  "카테고리 B: 미분 기반",        "corr_rank_diff",   "corr_matrix_diff",   DIFF_KEYS),
    ("LFP",   "카테고리 C: LFP 특징 기반",   "corr_rank_lfp",    "corr_matrix_lfp",    LFP_KEYS),
    ("Morph", "카테고리 D: 형태학적 거리",   "corr_rank_morph",  "corr_matrix_morph",  MORPH_KEYS),
]

# tab20 팔레트 (15개 feature 색 고정)
_TAB20 = matplotlib.colormaps["tab20"].colors

# ── 방향별 색상 팔레트 ────────────────────────────────────────────────────────
_DIS_PALETTE = ["#1a5276", "#2980b9", "#85c1e9", "#5dade2", "#2471a3", "#7fb3d3", "#154360"]
_CHG_PALETTE = ["#f0b27a", "#e67e22", "#a04000", "#dc7633", "#ca6f1e", "#fad7a0", "#784212"]


# ─────────────────────────────────────────────────────────────────────────────
# 세그먼트 메타 동적 생성
# ─────────────────────────────────────────────────────────────────────────────

def _derive_seg_meta(seg_names: list[str]) -> tuple[dict, dict, list, dict]:
    """세그먼트 이름 목록에서 시각화 메타 동적 생성.

    qfrac(dis_hi/mid/lo, chg_lo/mid/hi)뿐 아니라 protocol(chg_step0 …),
    vwindow, rcs 등 임의 축에 대해 작동.

    Returns
    -------
    seg_colors  : dict[str, str]        세그먼트 → 색상 hex
    seg_labels  : dict[str, str]        세그먼트 → 범례 레이블
    active_segs : list[str]             순서 있는 활성 세그먼트 목록
    mode_segs   : dict[str, tuple]      방향 → 세그먼트 튜플
    """
    dis_segs = [s for s in seg_names if s.startswith("dis")]
    chg_segs = [s for s in seg_names if s.startswith("chg")]

    seg_colors: dict[str, str] = {}
    seg_labels: dict[str, str] = {}
    for i, s in enumerate(dis_segs):
        seg_colors[s] = _DIS_PALETTE[i % len(_DIS_PALETTE)]
        seg_labels[s] = s
    for i, s in enumerate(chg_segs):
        seg_colors[s] = _CHG_PALETTE[i % len(_CHG_PALETTE)]
        seg_labels[s] = s

    active_segs = dis_segs + chg_segs
    mode_segs = {
        "discharge": tuple(dis_segs),
        "charge":    tuple(chg_segs),
    }
    return seg_colors, seg_labels, active_segs, mode_segs


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


# ─────────────────────────────────────────────────────────────────────────────
# Mutual Information 계산
# ─────────────────────────────────────────────────────────────────────────────

def _cell_mi(args):
    cell_key, sub, feat_cols, min_cycles = args
    cap = sub["capacity_Ah"].values
    if len(cap) < min_cycles or not _HAS_SKLEARN:
        return cell_key, None
    result = {}
    for col in feat_cols:
        vals = sub[col].values
        mask = np.isfinite(vals) & np.isfinite(cap)
        if mask.sum() < min_cycles:
            result[col] = np.nan
        else:
            mi = _mir(vals[mask].reshape(-1, 1), cap[mask],
                      discrete_features=False, random_state=42)[0]
            result[col] = float(mi)
    return cell_key, pd.Series(result)


def compute_mi(df: pd.DataFrame, feat_cols: list,
               min_cycles: int = 5, workers: int = 4) -> pd.DataFrame:
    """셀별 MI(feature, capacity_Ah). 반환: (n_cells, n_feats), 값 ≥ 0 (nats)."""
    groups = list(df.groupby(["dataset", "cell_id"]))
    args   = [(key, sub[feat_cols + ["capacity_Ah"]], feat_cols, min_cycles)
              for key, sub in groups]
    rows = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for key, ser in ex.map(_cell_mi, args):
            if ser is not None:
                rows[key] = ser
    return pd.DataFrame(rows).T


def summarise_mi(cell_mi_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "mean":    cell_mi_df.mean(skipna=True),
        "std":     cell_mi_df.std(skipna=True),
        "median":  cell_mi_df.median(skipna=True),
        "n_valid": cell_mi_df.notna().sum(),
    }).sort_values("mean", ascending=False)


def feature_corr_matrix(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    sub    = df[feat_cols].dropna(how="all")
    ranked = sub.rank(method="average", na_option="keep")
    return ranked.corr(method="pearson", min_periods=20)


# ─────────────────────────────────────────────────────────────────────────────
# 랭킹 집계용 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _concept_from_key(feat_key: str) -> str:
    """stat_v_mean_cw_dis_hi → stat_v_mean_cw  (세그먼트 suffix 2토큰 제거)."""
    return feat_key.rsplit("_", 2)[0]


def _cat_from_concept(concept: str) -> str:
    """stat_v_mean_cw → Stat,  morph_vt_dtw → Morph"""
    return {"stat": "Stat", "diff": "Diff", "lfp": "LFP", "morph": "Morph"}.get(
        concept.split("_")[0], "Stat"
    )


def _short_label(concept: str) -> str:
    """stat_v_mean_cw → v_mean_cw"""
    parts = concept.split("_", 1)
    return parts[1] if len(parts) > 1 else concept


def collect_flat(ds_summaries: dict, dataset_name: str) -> pd.DataFrame:
    """all_summaries[cat][seg] → 행당 (dataset, cat, seg, feat_key, concept, abs_rho)."""
    records = []
    for cat, cat_sums in ds_summaries.items():
        for seg, (summ, _hi_keys) in cat_sums.items():
            for feat_key, row in summ.iterrows():
                mean_rho = row["mean"]
                if pd.isna(mean_rho):
                    continue
                records.append({
                    "dataset":  dataset_name,
                    "cat":      cat,
                    "seg":      seg,
                    "feat_key": feat_key,
                    "concept":  _concept_from_key(feat_key),
                    "abs_rho":  abs(float(mean_rho)),
                    "mean_rho": float(mean_rho),
                })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: 상관계수 랭킹 바 차트 (N seg × 1 cat)
# ─────────────────────────────────────────────────────────────────────────────

def plot_corr_rank(seg_summaries: dict, out_path: Path,
                   cat_title: str, dataset_name: str = ""):
    """N 세그먼트 서브플롯 — feature ↔ capacity_Ah |ρ| 랭킹 바."""
    n_segs = len(seg_summaries)
    n_cols = 3
    n_rows = (n_segs + n_cols - 1) // n_cols

    n_his_max = max(
        (len(summ.dropna(subset=["mean"])) for summ, _ in seg_summaries.values()),
        default=15,
    )
    fig_h_row = max(4.0, min(10.0, n_his_max * 0.58))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 7, n_rows * fig_h_row),
                              constrained_layout=True)
    ds_lbl = f"[{dataset_name}] " if dataset_name else ""
    fig.suptitle(
        f"{ds_lbl}{cat_title}\n"
        "Feature ↔ capacity_Ah  |Spearman ρ| 랭킹  (셀 내부 ρ 평균±std, 방향 부호 표기)",
        fontsize=13, fontweight="bold",
    )

    axes_flat = np.array(axes).reshape(-1)
    for ax, (seg, (summ, _hi_keys)) in zip(axes_flat, seg_summaries.items()):
        color = SEG_COLORS.get(seg, "steelblue")

        summ_valid = summ.dropna(subset=["mean"])
        n_nan      = len(summ) - len(summ_valid)

        feat_cols = summ_valid.index.tolist()
        labels    = [HI_LABELS.get(k, k) for k in feat_cols]
        means     = summ_valid["mean"].values
        means_abs = np.abs(means)
        stds      = summ_valid["std"].fillna(0).values

        ax.barh(labels[::-1], means_abs[::-1],
                xerr=stds[::-1], color=color,
                alpha=0.82, capsize=3, height=0.65,
                error_kw={"elinewidth": 1.0, "alpha": 0.6})
        ax.set_ylim(-0.7, len(labels) - 0.3)

        for i, (m, s) in enumerate(zip(means[::-1], stds[::-1])):
            ax.text(abs(m) + 0.02, i,
                    f"{abs(m):.3f} {'(+)' if m >= 0 else '(-)'}",
                    va="center", ha="left", fontsize=7)

        ax.set_xlim(0, 1.2)
        ax.set_xlabel("|Spearman ρ|", fontsize=9)
        title_str = SEG_LABELS.get(seg, seg)
        if n_nan:
            nan_names = [HI_LABELS.get(k, k) for k in summ[summ["mean"].isna()].index]
            title_str += f"\n※ NaN 제외 {n_nan}개: {', '.join(nan_names)}"
        ax.set_title(title_str, fontsize=9, fontweight="bold", color=color, pad=4)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.3)

    for ax in axes_flat[n_segs:]:
        ax.set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Feature × Feature 상관행렬 히트맵 (N seg × 1 cat)
# ─────────────────────────────────────────────────────────────────────────────

def plot_corr_matrix(seg_matrices: dict, out_path: Path,
                     cat_title: str, dataset_name: str = ""):
    """N 세그먼트 서브플롯 — feature 상관행렬."""
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
# Plot 3: 카테고리 × 세그먼트 Top-5 비교 (전체 요약)
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_cross(all_summaries: dict, out_path: Path,
                   top_n: int = 5, dataset_name: str = ""):
    """4 × N 그리드 — rows=카테고리(Stat/Diff/LFP/Morph), cols=세그먼트."""
    cats    = list(all_summaries.keys())
    segs    = _ACTIVE_SEGS[:]          # 동적 세그먼트 목록 사용
    n_cats  = len(cats)
    n_segs  = len(segs)

    cat_palettes = {}
    for cat, _, _, _, base_keys in CATEGORIES:
        cat_palettes[cat] = {k: _TAB20[i % 20] for i, k in enumerate(base_keys)}

    fig, axes = plt.subplots(
        n_cats, n_segs,
        figsize=(n_segs * 4.2, n_cats * 4.0),
        constrained_layout=True,
    )
    if n_cats == 1:
        axes = axes[np.newaxis, :]
    if n_segs == 1:
        axes = axes[:, np.newaxis]

    ds_lbl = f"[{dataset_name}] " if dataset_name else ""
    fig.suptitle(
        f"{ds_lbl}카테고리 × 세그먼트 Top-{top_n} HI  (|ρ| 기준)\n"
        "행=카테고리, 열=세그먼트,  막대 색=feature 아이덴티티",
        fontsize=12, fontweight="bold",
    )

    x_pos = np.arange(top_n)
    for ri, (cat, cat_title_short, _, _, base_keys) in enumerate(CATEGORIES):
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

            for xi, (m, ma, std, b) in enumerate(zip(means, means_abs, stds, base)):
                feat_idx   = base_keys.index(b) + 1 if b in base_keys else xi + 1
                txt_color  = "white" if ma > 0.12 else "#333333"
                ax.text(xi, ma * 0.42, str(feat_idx),
                        ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color=txt_color, zorder=5)
                ax.text(xi, ma + std + 0.025,
                        f"{ma:.2f}{'(+)' if m >= 0 else '(-)'}",
                        ha="center", va="bottom", fontsize=5.5,
                        fontweight="bold")

            ax.set_xticks(x_pos)
            ax.set_xticklabels([""] * top_n)
            ax.set_ylim(0, 1.15)
            ax.axhline(1.0, color="red", lw=0.8, ls="--", alpha=0.5, zorder=0)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.3, lw=0.6)
            ax.set_axisbelow(True)

            if ri == 0:
                seg_color = SEG_COLORS.get(seg, "black")
                ax.set_title(SEG_LABELS.get(seg, seg), fontsize=8.5,
                             fontweight="bold", color=seg_color, pad=3)
            if ci == 0:
                ax.set_ylabel(f"{cat_title_short}\n|ρ|", fontsize=8.5)

    _prefix = {"Stat": "stat", "Diff": "diff", "LFP": "lfp", "Morph": "morph"}
    legend_handles = []
    for cat, _, _, _, base_keys in CATEGORIES:
        legend_handles.append(Patch(color="white", label=f"── {cat} ──"))
        pfx = _prefix.get(cat, "stat")
        for idx, bk in enumerate(base_keys, 1):
            col = cat_palettes[cat][bk]
            ref_seg = _ACTIVE_SEGS[0] if _ACTIVE_SEGS else "dis_hi"
            lbl = HI_LABELS.get(f"{pfx}_{bk}_{ref_seg}", bk)
            legend_handles.append(Patch(facecolor=col, label=f"{idx:2d}. {lbl}"))

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=16,
        fontsize=7.5,
        framealpha=0.88,
        bbox_to_anchor=(0.5, -0.07),
    )

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Feature 종합 랭킹 — 배터리별 (모든 구간×카테고리 Σ|ρ|)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_rank_battery(flat_df: pd.DataFrame, out_path: Path,
                               dataset_name: str):
    ranked = (
        flat_df.groupby("concept")["abs_rho"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    ranked["cat"]   = ranked["concept"].map(_cat_from_concept)
    ranked["label"] = ranked["concept"].map(_short_label)
    colors = [CAT_COLORS.get(c, "#888888") for c in ranked["cat"]]

    n = len(ranked)
    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.33)), constrained_layout=True)
    y_pos = np.arange(n)

    ax.barh(y_pos, ranked["abs_rho"], color=colors, alpha=0.85, height=0.72)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranked["label"], fontsize=7.5)
    ax.invert_yaxis()

    for i, (val, cat) in enumerate(zip(ranked["abs_rho"], ranked["cat"])):
        ax.text(val + 0.04, i, f"{val:.2f}", va="center", fontsize=6.5)
    for i in range(n):
        ax.text(-0.1, i, f"#{i+1}", va="center", ha="right",
                fontsize=6, color="#555555")

    n_segs = len(_ACTIVE_SEGS)
    ax.set_xlabel(f"Σ |Spearman ρ|  ({n_segs}구간 합산)", fontsize=9)
    ax.set_title(
        f"[{dataset_name}] Feature 종합 랭킹\n"
        f"모든 구간·카테고리에 걸쳐 |ρ| 합산 (구간 수 × 최대 1.0)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3, lw=0.7)

    legend_handles = [Patch(facecolor=CAT_COLORS[c], label=c)
                      for c in ["Stat", "Diff", "LFP"]]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Feature 종합 랭킹 — 구간별 (모든 배터리×카테고리 Σ|ρ|)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_rank_seg(all_flat_df: pd.DataFrame, out_path: Path):
    """구간 N개 서브플롯 — 모든 배터리·카테고리 합산 Σ|ρ| 랭킹 (공통 y-순서)."""
    segs = _ACTIVE_SEGS[:]            # 동적 세그먼트 목록 사용

    global_order = (
        all_flat_df.groupby("concept")["abs_rho"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    n = len(global_order)
    y_pos = np.arange(n)

    n_segs = len(segs)
    n_cols = min(3, n_segs)
    n_rows = (n_segs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 7, n_rows * max(5, n * 0.3)),
        constrained_layout=True,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = np.array(axes).reshape(n_rows, n_cols)

    datasets_str = " + ".join(sorted(all_flat_df["dataset"].unique()))
    fig.suptitle(
        f"[{datasets_str}] 구간별 Feature 랭킹\n"
        "모든 배터리·카테고리에 걸쳐 |ρ| 합산  (공통 y-순서: 전체 합산 기준)",
        fontsize=13, fontweight="bold",
    )

    bar_colors = [CAT_COLORS.get(_cat_from_concept(c), "#888888") for c in global_order]
    y_labels   = [_short_label(c) for c in global_order]

    axes_flat = axes.reshape(-1)
    for ax, seg in zip(axes_flat, segs):
        seg_df = all_flat_df[all_flat_df["seg"] == seg]
        if seg_df.empty:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            continue

        seg_sum = (
            seg_df.groupby("concept")["abs_rho"]
            .sum()
            .reindex(global_order)
            .fillna(0)
        )

        ax.barh(y_pos, seg_sum.values, color=bar_colors, alpha=0.85, height=0.72)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.invert_yaxis()

        for i, val in enumerate(seg_sum.values):
            if val > 0.01:
                ax.text(val + 0.015, i, f"{val:.2f}", va="center", fontsize=5.5)

        seg_color = SEG_COLORS.get(seg, "steelblue")
        ax.set_title(SEG_LABELS.get(seg, seg), fontsize=9,
                     fontweight="bold", color=seg_color)
        ax.set_xlabel("Σ |ρ|", fontsize=8)
        ax.grid(axis="x", alpha=0.3, lw=0.7)

    for ax in axes_flat[n_segs:]:
        ax.set_visible(False)

    legend_handles = [Patch(facecolor=CAT_COLORS[c], label=c)
                      for c in ["Stat", "Diff", "LFP"]]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.025))

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Scenario ε² (Kruskal-Wallis) — 세그먼트 레벨 구분 능력
# ─────────────────────────────────────────────────────────────────────────────

_CAT_MAP = {"stat": "Stat", "diff": "Diff", "lfp": "LFP", "morph": "Morph"}

_ALL_CONCEPTS = (
    [("stat",  k) for k in STAT_KEYS]
    + [("diff", k) for k in DIFF_KEYS]
    + [("lfp",  k) for k in LFP_KEYS]
    + [("morph", k) for k in MORPH_KEYS]
)


def compute_scenario_eps2(df: pd.DataFrame) -> pd.DataFrame:
    """각 HI 컨셉별 scenario(레벨) 구분 능력 ε² 계산.

    ε² = max(0, (H − k + 1) / (N − k))
      H = Kruskal-Wallis 통계량, k = 그룹 수, N = 전체 샘플 수

    _MODE_SEGS 모듈 전역을 사용하므로 축별 재빌드 후 자동 반영.
    """
    records = []
    for mode_name, segs_tuple in _MODE_SEGS.items():
        segs = list(segs_tuple)
        if len(segs) < 2:
            continue
        for prefix, key in _ALL_CONCEPTS:
            concept  = f"{prefix}_{key}"
            col_keys = [f"{concept}_{s}" for s in segs]
            avail    = [ck for ck in col_keys if ck in df.columns]
            if len(avail) < 2:
                continue

            groups = [df[ck].dropna().values for ck in avail]
            groups = [g for g in groups if len(g) >= 3]
            k      = len(groups)
            if k < 2:
                continue

            N = int(sum(len(g) for g in groups))
            try:
                H, pval = _kruskal(*groups)
                eps2 = max(0.0, (H - k + 1) / (N - k)) if N > k else np.nan
            except Exception:
                H = pval = eps2 = np.nan

            records.append({
                "concept":     concept,
                "cat":         _CAT_MAP.get(prefix, "Stat"),
                "short_label": _short_label(concept),
                "mode":        mode_name,
                "eps2":        float(eps2) if np.isfinite(eps2) else np.nan,
                "H":           float(H)    if np.isfinite(H)    else np.nan,
                "pvalue":      float(pval) if np.isfinite(pval) else np.nan,
                "n_samples":   N,
            })

    return pd.DataFrame(records)


def plot_scenario_eps2(eps2_df: pd.DataFrame, out_path: Path,
                       dataset_name: str = ""):
    """HI 컨셉별 scenario ε² 수평 바 차트.

    패널 수 = 방향 수 (_MODE_SEGS 기준, 최대 2개).
    방향별 세그먼트 이름은 동적으로 title에 표기.
    """
    modes = [m for m in ("discharge", "charge") if m in _MODE_SEGS and _MODE_SEGS[m]]
    if not modes:
        return

    n_modes    = len(modes)
    n_concepts = eps2_df["concept"].nunique()
    fig_h      = max(10, n_concepts * 0.30)

    fig, axes_arr = plt.subplots(1, n_modes, figsize=(12 * n_modes, fig_h),
                                  constrained_layout=True)
    if n_modes == 1:
        axes_arr = [axes_arr]

    ds_lbl = f"[{dataset_name}] " if dataset_name else ""
    fig.suptitle(
        f"{ds_lbl}HI 컨셉별 Scenario 구분 능력\n"
        r"$\varepsilon^2$ (Kruskal-Wallis)  — "
        r"높을수록 세그먼트 분포 차이 큼 → Classifier HI 후보",
        fontsize=13, fontweight="bold",
    )

    for ax, mode_name in zip(axes_arr, modes):
        segs_in_mode = " / ".join(_MODE_SEGS.get(mode_name, ())[:4])
        mode_title   = f"{'방전' if mode_name == 'discharge' else '충전'}  ({segs_in_mode})"

        sub = (
            eps2_df[eps2_df["mode"] == mode_name]
            .dropna(subset=["eps2"])
            .sort_values("eps2", ascending=False)
            .reset_index(drop=True)
        )
        if sub.empty:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            continue

        y_pos  = np.arange(len(sub))
        colors = [CAT_COLORS.get(c, "#888888") for c in sub["cat"]]
        x_max  = float(sub["eps2"].max())

        ax.barh(y_pos, sub["eps2"], color=colors, alpha=0.82, height=0.72)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["short_label"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, min(1.05, x_max * 1.18 + 0.04))
        ax.set_xlabel(r"$\varepsilon^2$", fontsize=10)
        ax.set_title(mode_title, fontsize=10, fontweight="bold")

        for i, (val, pv) in enumerate(zip(sub["eps2"], sub["pvalue"])):
            if not np.isfinite(val):
                continue
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            ax.text(val + x_max * 0.012, i,
                    f"{val:.3f}{sig}", va="center", fontsize=6.3)

        for thr, lbl, col in [(0.01, "small", "#aaaaaa"),
                               (0.06, "medium", "#e67e22"),
                               (0.14, "large",  "#c0392b")]:
            if thr <= x_max:
                ax.axvline(thr, color=col, lw=0.9, ls="--", alpha=0.55, zorder=0)
                ax.text(thr + x_max * 0.005, len(sub) - 0.5,
                        lbl, color=col, fontsize=6.5, va="top", alpha=0.8)

        ax.grid(axis="x", alpha=0.25, lw=0.7)

    cat_handles = [Patch(facecolor=CAT_COLORS[c], label=c)
                   for c in ["Stat", "Diff", "LFP", "Morph"]]
    sig_handles = [
        Patch(facecolor="none", edgecolor="none", label="유의성: * p<0.05"),
        Patch(facecolor="none", edgecolor="none", label="** p<0.01"),
        Patch(facecolor="none", edgecolor="none", label="*** p<0.001"),
    ]
    fig.legend(
        handles=cat_handles + sig_handles,
        loc="lower center", ncol=7, fontsize=8.5,
        bbox_to_anchor=(0.5, -0.025),
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7: MI 카테고리별 랭킹
# ─────────────────────────────────────────────────────────────────────────────

def plot_mi_rank_category(all_mi_summaries: dict, out_path: Path,
                          dataset_name: str = ""):
    """4 카테고리 서브플롯 — HI base 이름별 avg MI 랭킹."""
    n_cats = len(CATEGORIES)
    fig_h  = sum(max(3.5, len(bk) * 0.55) for _, _, _, _, bk in CATEGORIES)

    fig, axes = plt.subplots(n_cats, 1, figsize=(12, fig_h),
                              constrained_layout=True)
    if n_cats == 1:
        axes = [axes]

    ds_lbl = f"[{dataset_name}] " if dataset_name else ""
    fig.suptitle(
        f"{ds_lbl}카테고리별 HI Mutual Information 랭킹\n"
        "막대 = 세그먼트 평균 MI (nats)   점 = 세그먼트별 MI",
        fontsize=13, fontweight="bold",
    )

    for ax, (cat, cat_title, _, _, base_keys) in zip(axes, CATEGORIES):
        color    = CAT_COLORS.get(cat, "#888888")
        seg_sums = all_mi_summaries.get(cat, {})
        if not seg_sums:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        base_mi: dict = {}
        for seg, (mi_summ, _hi_keys) in seg_sums.items():
            for feat_key in mi_summ.index:
                concept = _concept_from_key(feat_key)
                base    = _short_label(concept)
                base_mi.setdefault(base, {})[seg] = mi_summ.loc[feat_key, "mean"]

        if not base_mi:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        avg_mi       = {b: np.nanmean(list(sv.values())) for b, sv in base_mi.items()}
        sorted_bases = sorted(avg_mi, key=lambda b: avg_mi[b], reverse=True)
        means        = [avg_mi[b] for b in sorted_bases]

        y_pos = np.arange(len(sorted_bases))
        ax.barh(y_pos, means[::-1], color=color, alpha=0.72, height=0.60)

        for yi, base in enumerate(sorted_bases[::-1]):
            for seg, seg_color in SEG_COLORS.items():
                mi_val = base_mi.get(base, {}).get(seg, np.nan)
                if np.isfinite(mi_val):
                    ax.scatter(mi_val, yi, color=seg_color, s=22, zorder=5,
                               edgecolors="white", linewidths=0.4, alpha=0.85)

        for yi, m in enumerate(means[::-1]):
            if np.isfinite(m) and m > 0:
                ax.text(m + 0.001, yi, f"{m:.3f}", va="center", fontsize=7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_bases[::-1], fontsize=8)
        ax.set_xlabel("MI (nats)", fontsize=9)
        ax.set_title(cat_title, fontsize=10, fontweight="bold", color=color)
        ax.grid(axis="x", alpha=0.3, lw=0.7)

    seg_handles = [Patch(facecolor=SEG_COLORS[s], label=SEG_LABELS.get(s, s))
                   for s in _ACTIVE_SEGS if s in SEG_COLORS]
    fig.legend(handles=seg_handles, loc="lower center", ncol=min(len(_ACTIVE_SEGS), 6),
               fontsize=8.5, bbox_to_anchor=(0.5, -0.03),
               title="점 = 세그먼트별 MI")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8: MI vs |ρ| 산점도
# ─────────────────────────────────────────────────────────────────────────────

def plot_mi_vs_rho(all_summaries: dict, all_mi_summaries: dict,
                   out_path: Path, dataset_name: str = ""):
    """MI vs |ρ| 산점도 — 각 점 = (HI concept, seg) 조합."""
    records = []
    for cat, _cat_title, _, _, _base_keys in CATEGORIES:
        for seg in _ACTIVE_SEGS:     # 동적 세그먼트 목록 사용
            corr_entry = all_summaries.get(cat, {}).get(seg)
            mi_entry   = all_mi_summaries.get(cat, {}).get(seg)
            if corr_entry is None or mi_entry is None:
                continue
            summ_rho, _ = corr_entry
            summ_mi,  _ = mi_entry
            for feat_key in summ_rho.index:
                if feat_key not in summ_mi.index:
                    continue
                rho_val = summ_rho.loc[feat_key, "mean"]
                mi_val  = summ_mi.loc[feat_key, "mean"]
                if pd.isna(rho_val) or pd.isna(mi_val):
                    continue
                records.append({
                    "cat":     cat,
                    "seg":     seg,
                    "base":    _short_label(_concept_from_key(feat_key)),
                    "abs_rho": abs(float(rho_val)),
                    "mi":      float(mi_val),
                })

    if not records:
        print("  [MI vs ρ] 데이터 없음, 건너뜀")
        return

    df_plot = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    ds_lbl = f"[{dataset_name}] " if dataset_name else ""
    ax.set_title(
        f"{ds_lbl}MI vs |ρ|  산점도\n"
        "각 점 = HI concept × 세그먼트  |  상단 좌측 = 비단조 관계(⚠ 후보)",
        fontsize=12, fontweight="bold",
    )

    rho_th = np.linspace(0.001, 0.999, 400)
    mi_th  = -0.5 * np.log(1.0 - rho_th ** 2)
    ax.plot(rho_th, mi_th, "--", color="gray", lw=1.3, alpha=0.55, zorder=1,
            label="Gaussian 이론: MI = −½ln(1−ρ²)")

    for cat in ["Stat", "Diff", "LFP", "Morph"]:
        sub = df_plot[df_plot["cat"] == cat]
        if sub.empty:
            continue
        ax.scatter(sub["abs_rho"], sub["mi"],
                   color=CAT_COLORS[cat], s=32, alpha=0.72,
                   edgecolors="white", linewidths=0.4,
                   label=cat, zorder=3)

    rho_thr = 0.3
    mi_thr  = float(df_plot["mi"].quantile(0.70))
    interesting = df_plot[
        (df_plot["abs_rho"] < rho_thr) & (df_plot["mi"] > mi_thr)
    ].sort_values("mi", ascending=False)
    for _, row in interesting.head(8).iterrows():
        ax.annotate(
            f"{row['base']} ({row['seg']})",
            (row["abs_rho"], row["mi"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=6.5, alpha=0.9,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#888888"),
        )

    ax.axvspan(0, rho_thr, alpha=0.06, color="orange", zorder=0)
    ax.axhline(mi_thr, color="#bbbbbb", lw=0.8, ls=":", zorder=0)
    ax.text(0.01, mi_thr + 0.003, f"MI 70th pctile ({mi_thr:.3f})",
            fontsize=7, color="#999999")

    ax.set_xlabel("|Spearman ρ|  (셀 평균)", fontsize=10)
    ax.set_ylabel("Mutual Information  (nats,  셀 평균)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.7)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 9: 시나리오별 |ρ| 3D 산점도
# ─────────────────────────────────────────────────────────────────────────────

def plot_corr_3d(all_summaries: dict, out_path: Path, dataset_name: str = "") -> None:
    """방향별 세그먼트 3개(또는 가능한 수)의 |ρ| 3D 산점도.

    _MODE_SEGS에서 방향별 세그먼트를 동적으로 추출하므로 qfrac/protocol/vwindow 모두 호환.
    방향당 세그먼트 < 2이면 해당 방향은 건너뜀.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _CAT_PREFIX = {"Stat": "stat", "Diff": "diff", "LFP": "lfp", "Morph": "morph"}
    _CAT_KEYS   = {"Stat": STAT_KEYS, "Diff": DIFF_KEYS, "LFP": LFP_KEYS, "Morph": MORPH_KEYS}

    # 방향별 최대 3개 세그먼트 선택
    mode_cfg = []
    for mode_name, segs_tuple in _MODE_SEGS.items():
        segs = list(segs_tuple)
        if len(segs) < 2:
            continue
        mode_label = "방전" if mode_name == "discharge" else "충전"
        # 3개 미만이면 마지막으로 패딩
        while len(segs) < 3:
            segs.append(segs[-1])
        seg_hi, seg_mid, seg_lo = segs[0], segs[1], segs[2]
        mode_cfg.append((mode_name, mode_label, seg_hi, seg_mid, seg_lo))

    if not mode_cfg:
        print("[plot_corr_3d] 유효한 방향 없음, 건너뜀")
        return

    rows = []
    for mode_name, mode_label, seg_hi, seg_mid, seg_lo in mode_cfg:
        for cat, *_ in CATEGORIES:
            s_hi  = all_summaries.get(cat, {}).get(seg_hi)
            s_mid = all_summaries.get(cat, {}).get(seg_mid)
            s_lo  = all_summaries.get(cat, {}).get(seg_lo)
            if s_hi is None or s_mid is None or s_lo is None:
                continue
            df_hi,  _ = s_hi
            df_mid, _ = s_mid
            df_lo,  _ = s_lo
            prefix    = _CAT_PREFIX[cat]
            for base in _CAT_KEYS[cat]:
                k_hi  = f"{prefix}_{base}_{seg_hi}"
                k_mid = f"{prefix}_{base}_{seg_mid}"
                k_lo  = f"{prefix}_{base}_{seg_lo}"
                if k_hi not in df_hi.index or k_mid not in df_mid.index or k_lo not in df_lo.index:
                    continue
                x = df_hi.at[k_hi,  "mean"]
                y = df_mid.at[k_mid, "mean"]
                z = df_lo.at[k_lo,  "mean"]
                if pd.isna(x) or pd.isna(y) or pd.isna(z):
                    continue
                rows.append({
                    "mode": mode_name, "mode_label": mode_label,
                    "cat": cat, "hi_key": base,
                    "label": HI_LABELS.get(k_hi, base),
                    "x": abs(float(x)), "y": abs(float(y)), "z": abs(float(z)),
                })

    if not rows:
        print("[plot_corr_3d] 데이터 없음, 건너뜀"); return

    data = pd.DataFrame(rows)
    cats    = ["Stat", "Diff", "LFP", "Morph"]
    markers = {"Stat": "o", "Diff": "s", "LFP": "^", "Morph": "D"}
    sizes   = {"Stat": 38, "Diff": 38, "LFP": 44, "Morph": 44}

    n_modes = len(mode_cfg)
    fig = plt.figure(figsize=(10 * n_modes, 8.5))
    fig.patch.set_facecolor("#f5f6fa")

    for col_idx, (mode_name, mode_label, seg_hi, seg_mid, seg_lo) in enumerate(mode_cfg):
        ax = fig.add_subplot(1, n_modes, col_idx + 1, projection="3d")
        ax.set_facecolor("#f5f6fa")
        sub = data[data["mode"] == mode_name]

        for cat in cats:
            s = sub[sub["cat"] == cat]
            if s.empty:
                continue
            ax.scatter(
                s["x"], s["y"], s["z"],
                c=CAT_COLORS.get(cat, "#888888"),
                marker=markers[cat],
                s=sizes[cat], alpha=0.78,
                edgecolors="white", linewidths=0.35,
                label=cat, depthshade=True,
            )

        diag = np.linspace(0, 1, 60)
        ax.plot(diag, diag, diag, color="#555555", lw=0.9, alpha=0.45,
                linestyle="--", zorder=0)
        ax.plot(diag, diag, np.zeros(60), color="#aaaaaa", lw=0.5, alpha=0.3)
        ax.plot(diag, np.zeros(60), diag, color="#aaaaaa", lw=0.5, alpha=0.3)
        ax.plot(np.zeros(60), diag, diag, color="#aaaaaa", lw=0.5, alpha=0.3)

        ax.set_xlabel(f"{seg_hi}  |ρ|", fontsize=9, labelpad=7)
        ax.set_ylabel(f"{seg_mid}  |ρ|", fontsize=9, labelpad=7)
        ax.set_zlabel(f"{seg_lo}  |ρ|", fontsize=9, labelpad=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(labelsize=7)
        ax.view_init(elev=22, azim=45)
        ax.set_title(
            f"{mode_label}  (X={seg_hi} / Y={seg_mid} / Z={seg_lo})",
            fontsize=10.5, fontweight="bold", pad=10,
        )
        ax.grid(True, lw=0.4, alpha=0.3)

    leg_handles = [Patch(facecolor=CAT_COLORS.get(c, "#888"), label=c) for c in cats]
    fig.legend(handles=leg_handles, loc="lower center", ncol=4,
               fontsize=9.5, framealpha=0.85, bbox_to_anchor=(0.5, 0.01))

    ds_label = f" — {dataset_name}" if dataset_name else ""
    fig.suptitle(f"시나리오(세그먼트별) |ρ| 3D 산점도{ds_label}",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HI 세그먼트 × 카테고리 상관분석 + Scenario ε² (다축 호환)")
    parser.add_argument("--pkl",        default="",
                        help="PKL 파일 경로 (미지정 시 axis 기반 자동 선택)")
    parser.add_argument("--seg-axis",   type=str, default="qfrac",
                        help="세그멘테이션 축: qfrac|protocol|vwindow|rcs|cluster (기본: qfrac)")
    parser.add_argument("--axis-config", type=str, default="{}",
                        help="축 파라미터 JSON (예: '{\"max_steps\": 3}')")
    parser.add_argument("--dataset",    default="all", choices=["all", "mit", "hust"])
    parser.add_argument("--min-cycles", type=int, default=5)
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--top-n",      type=int, default=5,
                        help="top_cross 플롯 상위 HI 개수 (기본: 5)")
    args = parser.parse_args()

    _axis = args.seg_axis
    try:
        _axis_cfg: dict = json.loads(args.axis_config)
    except json.JSONDecodeError as e:
        print(f"[ERROR] --axis-config JSON 파싱 실패: {e}"); return

    # ── 세그먼트 메타 재빌드 ──────────────────────────────────────────────────
    global SEG_COLORS, SEG_LABELS, _ACTIVE_SEGS, _MODE_SEGS, HI_GROUPS, HI_LABELS

    if _axis != "qfrac":
        import sys as _sys
        _sys.path.insert(0, str(HERE.parent))
        from common.scenario import get_segmenter as _get_seg
        from hi_correlation import _build_hi_groups
        _seg_names = _get_seg(_axis, {_axis: _axis_cfg}).get_spec().scenario_names
        SEG_COLORS, SEG_LABELS, _ACTIVE_SEGS, _MODE_SEGS = _derive_seg_meta(_seg_names)
        _new_groups, _, _new_labels = _build_hi_groups(_seg_names)
        HI_GROUPS = _new_groups
        HI_LABELS = _new_labels
        print(f"[corr] 축={_axis}  세그먼트: {_seg_names}")
    else:
        _ACTIVE_SEGS = list(SEG_COLORS.keys())
        _MODE_SEGS   = {
            "discharge": ("dis_hi", "dis_mid", "dis_lo"),
            "charge":    ("chg_lo", "chg_mid", "chg_hi"),
        }

    # ── 출력 디렉토리 ─────────────────────────────────────────────────────────
    _dir_suffix = f"_{_axis}" if _axis != "qfrac" else ""
    OUT_BASE = HERE / "seg_corr" / (date.today().strftime("%m%d") + _dir_suffix)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # ── PKL 로드 ──────────────────────────────────────────────────────────────
    if args.pkl:
        pkl_path = Path(args.pkl)
    else:
        # 축별 기본 캐시 파일
        default_name = "hi_features.pkl" if _axis == "qfrac" else f"hi_features_{_axis}.pkl"
        pkl_path = HERE / default_name

    if not pkl_path.exists():
        candidates = sorted(HERE.glob(f"*hi_features*{_axis}*.pkl" if _axis != "qfrac" else "*hi_features*.pkl"),
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

    datasets = (sorted(df["dataset"].unique().tolist())
                if args.dataset == "all" else [args.dataset.upper()])

    all_flat_records: list = []

    for ds_name in datasets:
        df_ds = df[df["dataset"] == ds_name].copy()
        if df_ds.empty:
            print(f"\n[SKIP] {ds_name}: 데이터 없음"); continue

        print(f"\n{'='*60}")
        print(f"  데이터셋: {ds_name}  ({len(df_ds):,}행, "
              f"{df_ds.groupby('cell_id').ngroups}셀)")
        print(f"{'='*60}")

        all_summaries:    dict = {cat: {} for cat, *_ in CATEGORIES}
        all_matrices:     dict = {cat: {} for cat, *_ in CATEGORIES}
        all_mi_summaries: dict = {cat: {} for cat, *_ in CATEGORIES}

        for cat, cat_title, rank_stem, matrix_stem, _base_keys in CATEGORIES:
            print(f"\n─── {cat_title} ───")
            for seg in _ACTIVE_SEGS:       # 동적 세그먼트 목록 사용
                group_key = f"{seg} — {cat}"
                hi_keys   = HI_GROUPS.get(group_key, [])
                avail     = [k for k in hi_keys if k in df_ds.columns]
                if not avail:
                    print(f"  [{seg}] 컬럼 없음, SKIP")
                    continue

                cell_corr = compute_corr(df_ds, avail,
                                         min_cycles=args.min_cycles,
                                         workers=args.workers)
                summ = summarise(cell_corr)
                all_summaries[cat][seg] = (summ, avail)
                n_valid = int(cell_corr.notna().all(axis=1).sum())
                top_lbl = HI_LABELS.get(summ.index[0], summ.index[0])
                print(f"  [{seg}] 유효셀={n_valid}  "
                      f"top ρ={summ['mean'].iloc[0]:+.3f} ({top_lbl})")

                if _HAS_SKLEARN:
                    cell_mi_df = compute_mi(df_ds, avail,
                                            min_cycles=args.min_cycles,
                                            workers=args.workers)
                    mi_summ = summarise_mi(cell_mi_df)
                    all_mi_summaries[cat][seg] = (mi_summ, avail)
                    top_mi_lbl = HI_LABELS.get(mi_summ.index[0], mi_summ.index[0])
                    print(f"  [{seg}] top MI={mi_summ['mean'].iloc[0]:.4f} ({top_mi_lbl})")

                corr_mat = feature_corr_matrix(df_ds, avail)
                all_matrices[cat][seg] = (corr_mat, avail)

            if all_summaries[cat]:
                rank_path = OUT_BASE / f"{rank_stem}_{ds_name.lower()}.png"
                print(f"\n  [Plot] Rank  → {rank_path.name}")
                plot_corr_rank(all_summaries[cat], rank_path,
                               cat_title, dataset_name=ds_name)

            if all_matrices[cat]:
                mat_path = OUT_BASE / f"{matrix_stem}_{ds_name.lower()}.png"
                print(f"  [Plot] Matrix → {mat_path.name}")
                plot_corr_matrix(all_matrices[cat], mat_path,
                                 cat_title, dataset_name=ds_name)

        top_path = OUT_BASE / f"top_cross_{ds_name.lower()}.png"
        print(f"\n  [Plot] Top-{args.top_n} Cross → {top_path.name}")
        plot_top_cross(all_summaries, top_path,
                       top_n=args.top_n, dataset_name=ds_name)

        flat_ds = collect_flat(all_summaries, ds_name)
        all_flat_records.append(flat_ds)
        rank_batt_path = OUT_BASE / f"feature_rank_battery_{ds_name.lower()}.png"
        print(f"\n  [Plot] Feature Rank (battery) → {rank_batt_path.name}")
        plot_feature_rank_battery(flat_ds, rank_batt_path, ds_name)

        print(f"\n  [Plot] Scenario ε² 계산 중...")
        eps2_df   = compute_scenario_eps2(df_ds)
        eps2_path = OUT_BASE / f"scenario_eps2_{ds_name.lower()}.png"
        plot_scenario_eps2(eps2_df, eps2_path, dataset_name=ds_name)

        if _HAS_SKLEARN and any(all_mi_summaries[c] for c, *_ in CATEGORIES):
            mi_rank_path = OUT_BASE / f"mi_rank_category_{ds_name.lower()}.png"
            print(f"\n  [Plot] MI Rank (category) → {mi_rank_path.name}")
            plot_mi_rank_category(all_mi_summaries, mi_rank_path,
                                  dataset_name=ds_name)

            mi_rho_path = OUT_BASE / f"mi_vs_rho_{ds_name.lower()}.png"
            print(f"  [Plot] MI vs |ρ|  → {mi_rho_path.name}")
            plot_mi_vs_rho(all_summaries, all_mi_summaries,
                           mi_rho_path, dataset_name=ds_name)

        corr3d_path = OUT_BASE / f"corr_3d_{ds_name.lower()}.png"
        print(f"\n  [Plot] 3D 산점도 → {corr3d_path.name}")
        plot_corr_3d(all_summaries, corr3d_path, dataset_name=ds_name)

        # ε² Top-10 텍스트 요약
        print(f"\n  {'─'*60}")
        for mode_name, segs_tuple in _MODE_SEGS.items():
            mode_lbl = f"{'방전' if mode_name == 'discharge' else '충전'} ({'/'.join(list(segs_tuple)[:3])})"
            top10 = (
                eps2_df[eps2_df["mode"] == mode_name]
                .dropna(subset=["eps2"])
                .nlargest(10, "eps2")
            )
            print(f"\n  Scenario ε² Top-10  [{mode_lbl}]")
            for _, r in top10.iterrows():
                sig = ("***" if r["pvalue"] < 0.001 else
                       "**"  if r["pvalue"] < 0.01  else
                       "*"   if r["pvalue"] < 0.05  else " ")
                print(f"    {r['short_label']:<32} ε²={r['eps2']:.4f} {sig}  [{r['cat']}]")

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

    if all_flat_records:
        all_flat_df = pd.concat(all_flat_records, ignore_index=True)
        rank_seg_path = OUT_BASE / "feature_rank_seg.png"
        print(f"\n[Plot] Feature Rank (segment) → {rank_seg_path.name}")
        plot_feature_rank_seg(all_flat_df, rank_seg_path)

    print(f"\n완료 — {OUT_BASE}/")


if __name__ == "__main__":
    main()
