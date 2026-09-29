"""
4_hi_analysis/plot.py

hi_correlation.py Step 4 산출물(상관계수 df, 사이클 df)을 그림으로 그리는 순수
시각화 함수 모음. 2026-09-29: hi_correlation.py 994~1243행(히트맵/산점도/대표
셀 추이 플롯)을 이 파일로 분리 — hi_correlation.py는 추출+상관분석 로직만
남기고, "그리는" 부분은 여기로 뺐다.

hi_correlation.py를 import하지 않는다(단방향 의존 — hi_correlation.py가 이
모듈을 import한다). HI_LABELS/HI_GROUPS/HI_GROUP_TAG는 hi_correlation.py의
main()이 축에 따라 런타임에 재빌드하는 값이라(비-qfrac 축이면 _build_hi_groups로
교체), 이 모듈이 자체적으로 import해서 캐싱하면 재빌드 전 값을 계속 들고 있는
문제가 생긴다 — 그래서 모듈 전역으로 두지 않고 매 호출마다 인자로 받는다.
STAT_KEYS/DIFF_KEYS/LFP_KEYS/MORPH_KEYS(utils.hi_schema, 세그먼트당 66개
HI의 고정 키 목록)와 SAMPLE_CELL_IDS/DATASET_CMAPS/DATASET_COLORS(데이터셋별
플롯 스타일)는 재빌드 대상이 아니라 이 모듈 안에 직접 둔다.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.hi_schema import STAT_KEYS, DIFF_KEYS, LFP_KEYS, MORPH_KEYS

# 대표 셀(플롯용) — 2_preprocess/preprocess.py의 SAMPLE_IDS와 동일 셀
SAMPLE_CELL_IDS = {"MIT": "b1c0", "HUST": "1-1", "TJU": "CY25-05_1-#1", "CALCE": "CS2_8"}
DATASET_CMAPS   = {"MIT": "Blues", "HUST": "Oranges", "TJU": "Greens", "CALCE": "Purples"}
DATASET_COLORS  = {"MIT": "#1f77b4", "HUST": "#d55e00", "TJU": "#2ca02c", "CALCE": "#9467bd"}


def _draw_heatmap(ax, keys, title, corr_df, hi_labels: dict, datasets=("MIT", "HUST")):
    """단일 히트맵. |ρ| 평균 내림차순 정렬."""
    avail = [k for k in keys if k in corr_df.index]
    if not avail:
        ax.set_title(title, fontsize=8); ax.axis("off"); return None, []
    order = (
        corr_df.loc[avail].abs().mean(axis=1)
        .fillna(0).sort_values(ascending=False).index.tolist()
    )
    datasets = list(datasets)
    hm = corr_df.loc[order, datasets].values

    im = ax.imshow(hm.T, aspect="auto", cmap="RdYlGn",
                   vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([hi_labels.get(k, k) for k in order],
                       rotation=38, ha="right", fontsize=7)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=8, pad=4, fontweight="bold")
    for xi, k in enumerate(order):
        for yi, ds in enumerate(datasets):
            val = hm[xi, yi]
            txt = f"{val:.2f}" if np.isfinite(val) else "N/A"
            ax.text(xi, yi, txt, ha="center", va="center",
                    fontsize=6,
                    color="white" if abs(val) > 0.65 else "black",
                    fontweight="bold")
    return im, order


def plot_correlation(corr_df: pd.DataFrame, df: pd.DataFrame,
                     out_path: Path, hi_groups: dict, hi_labels: dict,
                     hi_group_tag: dict, n_top: int = 4, datasets: list | None = None):
    datasets = list(datasets) if datasets is not None else ["MIT", "HUST"]
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")

    for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = font; break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    # ── 레이아웃: Global + N segment rows + scatter ───────────────────────
    # 각 세그먼트 행: [Stat | Diff | LFP] 3 sub-panels
    _segs_for_plot = [k.split(" — ")[0] for k in hi_groups if k.endswith("— Stat")]
    n_segs = len(_segs_for_plot)
    n_seg_hi = n_segs * (len(STAT_KEYS) + len(DIFF_KEYS) + len(LFP_KEYS) + len(MORPH_KEYS))
    fig = plt.figure(figsize=(44, 14 + 7 * n_segs))
    fig.suptitle(
        f"Health Indicator Spearman ρ  ─  {n_seg_hi} HIs (Segment)",
        fontsize=13, fontweight="bold", y=0.999,
    )
    gs_main = gridspec.GridSpec(
        n_segs + 1, 1, figure=fig,
        height_ratios=[1.0] * n_segs + [2.0],
        hspace=0.60,
    )

    # ── 행 0–(N-1): 세그먼트별 3 sub-panels (HI_GROUPS 기반 동적 생성, 완전 사이클
    # Global HI는 2026-09-28부로 계산 자체를 안 하므로 그 행도 같이 없앴다) ───────
    seg_rows = [
        (seg, seg, row_idx)
        for row_idx, seg in enumerate(_segs_for_plot)
    ]
    ref_im = None
    for seg, seg_title, row_idx in seg_rows:
        gs_seg = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_main[row_idx], wspace=0.06)
        for ci, cat in enumerate(["Stat", "Diff", "LFP"]):
            ax_s = fig.add_subplot(gs_seg[ci])
            im_s, _ = _draw_heatmap(
                ax_s,
                hi_groups[f"{seg} — {cat}"],
                f"{seg_title}  [{cat}]",
                corr_df,
                hi_labels,
                datasets=datasets,
            )
            if im_s is not None and ref_im is None:
                ref_im = im_s

    # ── 공유 컬러바 ── 이 시점의 fig.get_axes()는 전부 Global+세그먼트 히트맵
    # (산점도 axes는 아직 생성 전) — 예전엔 n_segs=6(qfrac 계열) 기준 [:7]로
    # 하드코딩돼 있었는데, random/random_grid(assign="none")처럼 n_segs=2인
    # 축에서는 무해했지만 우연히 맞았을 뿐이라 명시적으로 전체를 쓰도록 고침.
    if ref_im is not None:
        cbar = plt.colorbar(ref_im, ax=fig.get_axes(), shrink=0.25, pad=0.01)
        cbar.set_label("Spearman ρ", fontsize=10)

    # ── 마지막 행: 상위 HI 산점도 ── gs_main은 n_segs+1행(0..n_segs-1=세그먼트,
    # 마지막=산점도, 2026-09-28 Global 행 제거로 한 칸씩 당겨짐)이라 마지막 행
    # 인덱스는 n_segs. 예전엔 이 값이 항상 7(=n_segs=6인 qfrac/q_frac_wide/
    # q_frac_ref 표준 6-시나리오 축 기준)로 하드코딩돼 있어서 n_segs가 다른 축
    # (random/random_grid의 assign="none"=2시나리오, protocol/vwindow/cluster/
    # full_cycle 등)에서 GridSpec 범위를 벗어나 IndexError가 났다(2026-08-15).
    abs_mean = corr_df.abs().mean(axis=1).fillna(0).sort_values(ascending=False)
    top_his  = abs_mean.index[:n_top].tolist()

    gs_sc = gridspec.GridSpecFromSubplotSpec(
        len(datasets), n_top, subplot_spec=gs_main[n_segs], hspace=0.52, wspace=0.30)
    cmaps  = DATASET_CMAPS
    colors = DATASET_COLORS

    for ci, hi_key in enumerate(top_his):
        for ri, ds in enumerate(datasets):
            ax = fig.add_subplot(gs_sc[ri, ci])
            sub = df[df["dataset"] == ds][[hi_key, "capacity_Ah", "cycle"]].dropna()
            if len(sub) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                ax.set_title(f"{hi_labels.get(hi_key, hi_key)}  [{ds}]", fontsize=8)
                continue
            cyc_n = ((sub["cycle"] - sub["cycle"].min()) /
                     max(sub["cycle"].max() - sub["cycle"].min(), 1))
            ax.scatter(sub[hi_key], sub["capacity_Ah"],
                       c=cyc_n, cmap=cmaps[ds],
                       s=1.5, alpha=0.35, linewidths=0, rasterized=True)
            if len(sub) > 20:
                coef  = np.polyfit(sub[hi_key], sub["capacity_Ah"], 1)
                x_lin = np.linspace(sub[hi_key].min(), sub[hi_key].max(), 200)
                ax.plot(x_lin, np.polyval(coef, x_lin),
                        "-", color=colors[ds], lw=1.8, alpha=0.9)
            rho     = corr_df.loc[hi_key, ds] if hi_key in corr_df.index else np.nan
            rho_str = f"ρ={rho:.3f}" if np.isfinite(rho) else "ρ=N/A"
            lbl     = hi_labels.get(hi_key, hi_key)
            tag     = hi_group_tag.get(hi_key, "")
            ax.set_title(f"{lbl}  [{ds}]\n[{tag}]  {rho_str}", fontsize=7, pad=3)
            ax.set_xlabel(lbl, fontsize=6)
            ax.set_ylabel("Capacity (Ah)", fontsize=6)
            ax.tick_params(labelsize=5)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


def _plot_sample_hi(df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path,
                     hi_labels: dict, hi_group_tag: dict,
                     datasets: list | None = None) -> None:
    """대표 셀 상위 HI 사이클 추이."""
    datasets = list(datasets) if datasets is not None else ["MIT", "HUST"]
    SAMPLES = {ds: SAMPLE_CELL_IDS[ds] for ds in datasets}
    CMAPS   = {ds: DATASET_CMAPS[ds] for ds in datasets}

    df_p = df.copy()
    df_p["dataset"] = df_p["dataset"].replace("MIT_MAT", "MIT")

    abs_mean = corr_df.abs().mean(axis=1).fillna(0).sort_values(ascending=False)
    top4     = abs_mean.index[:4].tolist()
    n_ds     = len(SAMPLES)

    fig, axes = plt.subplots(n_ds, 4, figsize=(16, n_ds * 3.5),
                              squeeze=False, constrained_layout=True)
    fig.suptitle("[Step 4 HI 추출 결과]  대표 셀 상위 HI 사이클 추이",
                 fontsize=11, fontweight="bold")

    for ri, (ds, cell) in enumerate(SAMPLES.items()):
        sub = df_p[(df_p["dataset"] == ds) & (df_p["cell_id"] == cell)].sort_values("cycle")
        for ci, hi_key in enumerate(top4):
            ax = axes[ri, ci]
            if len(sub) == 0 or hi_key not in sub.columns:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9); continue
            valid = sub[["cycle", hi_key, "capacity_Ah"]].dropna()
            if len(valid) < 3:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9); continue
            cap_range = valid["capacity_Ah"].max() - valid["capacity_Ah"].min()
            c_norm = (valid["capacity_Ah"] - valid["capacity_Ah"].min()) / max(cap_range, 1e-9)
            ax.scatter(valid["cycle"], valid[hi_key],
                       c=c_norm, cmap=CMAPS[ds], s=8, alpha=0.8)
            rho = corr_df.loc[hi_key, ds] if (
                hi_key in corr_df.index and ds in corr_df.columns) else np.nan
            rho_str = f"ρ={rho:.3f}" if np.isfinite(rho) else "ρ=N/A"
            lbl = hi_labels.get(hi_key, hi_key)
            tag = hi_group_tag.get(hi_key, "")
            title = f"{lbl}  [{tag}]\n{rho_str}" if ri == 0 else f"{lbl}  [{tag}]"
            ax.set_title(title, fontsize=8, fontweight="bold")
            ax.set_xlabel("Cycle", fontsize=7)
            ax.set_ylabel(lbl, fontsize=7)
            ax.tick_params(labelsize=6)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_hi_trend.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()
