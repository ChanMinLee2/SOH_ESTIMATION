"""
4_hi_analysis/plot.py

hi_correlation.py Step 4 산출물(상관계수 df, 사이클 df)을 그림으로 그리는 순수
시각화 함수 모음. 2026-09-29: hi_correlation.py 994~1243행(히트맵/산점도/대표
셀 추이 플롯)을 이 파일로 분리했고, 같은 날 hi_segment_viz.py(세그먼트별 HI
열화 추이/시나리오 오버레이)도 여기로 합쳤다 — hi_correlation.py는 추출+상관분석
로직만 남기고, "그리는" 부분은 전부 여기로 뺐다(hi_segment_viz.py는 삭제).
합치면서 hi_segment_viz.py의 Figure 1(hi_segment_cuts.png, `plot_segment_cuts`)은
가져오지 않았다 — 트리거 조건이 `_axis == "qfrac"`이었는데 이 프로젝트의 실제
축 이름은 "q_frac_ref"라 2026-09-24 비-정식 축 정리 이후 이 조건이 항상 거짓이라
사실상 죽은 코드였다(원본 hi_segment_viz.py git 히스토리에서 복원 가능).

hi_correlation.py를 import하지 않는다(단방향 의존 — hi_correlation.py가 이
모듈을 import한다). logics.py는 import한다(plot_from_result_dir가
build_hi_groups를 씀) — logics.py는 plot.py를 import하지 않으므로 순환 없음.

HI_LABELS/HI_GROUPS/HI_GROUP_TAG는 hi_correlation.py의 main()이 축에 따라
런타임에 재빌드하는 값이라, 이 모듈이 자체적으로 import해서 캐싱하면 재빌드
전 값을 계속 들고 있는 문제가 생긴다 — 그래서 모듈 전역으로 두지 않고 매
호출마다 인자로 받는다. STAT_KEYS/DIFF_KEYS/LFP_KEYS/MORPH_KEYS(utils.hi_schema,
세그먼트당 66개 HI의 고정 키 목록)와 SAMPLE_CELL_IDS/DATASET_CMAPS/
DATASET_COLORS(데이터셋별 플롯 스타일)는 재빌드 대상이 아니라 이 모듈 안에
직접 둔다.

2026-09-29 신규: _plot_step4_from_result_dir(step_dir) — hi_correlation.py::main()이
저장한 step_4_result/ 폴더 경로 하나만 받아서(manifest.json+correlation.csv)
전체 플롯을 다시 그려 같은 폴더에 저장한다. 이상탐지(산출물 검증)는 아직 없다
— 보류(다음 라운드).

2026-10-01: tools/ 밑에 따로 있던 독립 실행용 진단 플롯 스크립트를 이 파일로
흡수하기 시작했다(1단계 — plot_cell_cycles.py/plot_cycle_segments.py/
plot_all_mit_cells.py, 총 3개 삭제). 각자 한국어 폰트 폴백·데이터셋 경로 딕셔너리를
따로 들고 있던 걸 이 파일 하나로 합쳤다. 규칙(이후 2단계=seg_corr_analysis.py,
3단계=seg_diagnose.py 흡수 때도 동일하게 적용할 것):
  - 모든 플롯 함수(최상위든 서브패널 헬퍼든)는 `_plot_step4_`로 시작 — 이름만
    보고 "Step4 진단 플롯이고 뭘 그리는지" 알 수 있게. 호출 depth는 최대
    2단계(예: 데이터셋 전체 셀 오버레이 → 셀 하나 오버레이).
  - 계산/로드 헬퍼(플롯 아님)는 `_compute_step4_*`/`_load_step4_*`.
  - 데이터셋 정체성 색상이 필요하면 **반드시 기존 DATASET_COLORS를 그대로
    참조** — 새 로컬 색상 딕셔너리를 만들지 않는다(흡수 전 seg_diagnose.py가
    `{"MIT":"#3498db","HUST":"#e74c3c"}`를 따로 갖고 있어 DATASET_COLORS와
    색이 어긋났던 문제를 재발시키지 않기 위함).
  - `STEP4_DIAG_PLOTS` 레지스트리에 번호를 등록하면 `python 4_hi_analysis/plot.py
    <번호>`로 독립 실행 가능 — 옛 tools/ 스크립트 각자의 argparse CLI를 대체한다.
    실제 파라미터(데이터셋/셀/사이클 등)는 번호 외엔 CLI로 안 받고
    parameters.py(FIXED_STEP4_DIAG_*)에서만 읽는다(run_pipeline.py와 동일 원칙).
"""

import argparse
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import parameters as P
from data_directories import EXTERNAL_DATA_ROOT
from utils.hi_schema import STAT_KEYS, DIFF_KEYS, LFP_KEYS, MORPH_KEYS
import logics as L

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent

# 한국어 폰트 폴백 — 모듈 로드 시 한 번만(2026-10-01, 예전엔 _plot_step4_correlation()
# 안에만 인라인으로 있었고 흡수해온 tools/ 스크립트마다 똑같은 블록이 또 있었다 —
# 전부 이 하나로 통합).
for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _font; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# 대표 셀(플롯용) — 2_preprocess/preprocess.py의 SAMPLE_IDS와 동일 셀
SAMPLE_CELL_IDS = {"MIT": "b1c0", "HUST": "1-1", "TJU": "CY25-05_1-#1", "CALCE": "CS2_8"}
DATASET_CMAPS   = {"MIT": "Blues", "HUST": "Oranges", "TJU": "Greens", "CALCE": "Purples"}
DATASET_COLORS  = {"MIT": "#1f77b4", "HUST": "#d55e00", "TJU": "#2ca02c", "CALCE": "#9467bd"}
# hi_segment_viz.py에서 합류 — MIT/HUST 실선, TJU/CALCE 점선(시나리오 오버레이 전용)
DATASET_LINESTYLE = {"MIT": "-", "HUST": "--", "TJU": "-", "CALCE": "--"}

# ─────────────────────────────────────────────────────────────────────────────
# Step4 진단 플롯 공용 — tools/plot_cell_cycles.py+plot_cycle_segments.py+
# plot_all_mit_cells.py 흡수(2026-10-01, 1단계). 세 스크립트 다 _4_data_hi/clean
# (전처리 후, constants.MIT_DIR 등)이 아니라 _1_data_unified(전처리 **전** 원본)를
# 읽는다 — plot_cell_cycles가 2_preprocess/outputs/shape_outlier_report.csv(어떤
# 사이클이 전처리에서 걸러졌는지)를 원본 위에 겹쳐 그리는 용도라 의도적으로 다른
# 디렉터리다. 각 파일이 따로 갖고 있던 하드코딩 경로 딕셔너리 3개를 함수 하나로
# 합치면서, 로컬에 없으면 EXTERNAL_DATA_ROOT(D:)로 폴백하는 기존 관례
# (2_preprocess/preprocess.py)도 같이 적용했다 — 옛 3개 스크립트는 로컬 전용
# 하드코딩이라 MIT/HUST가 D:로 옮겨간 뒤로는 애초에 못 찾고 있었다.
def _resolve_step4_dataset_dir(dataset: str) -> Path:
    """_1_data_unified 원본(대용량)이 로컬에 없을 수 있음 — 이 경우 외부 드라이브로
    폴백(2_preprocess/preprocess.py::_resolve_unified_src와 동일 관례, 2026-10-01
    plot.py에도 적용). 1단계 흡수 당시엔 옛 tools/ 3개 스크립트의 로컬 전용
    하드코딩을 그대로 옮겼는데, 실제로 MIT/HUST가 로컬(C:)엔 더 이상 없고
    D:(EXTERNAL_DATA_ROOT)에만 있어서 못 찾고 있었다(REFACTORING.md 2026-10-01
    "발견(수정 안 함)" 항목 — 이번에 바로 그 항목을 고친 것)."""
    local = PROJECT_ROOT / "_1_data_unified" / dataset
    if local.exists():
        return local
    return EXTERNAL_DATA_ROOT / "_1_data_unified" / dataset
STEP4_OUTLIER_CSV = PROJECT_ROOT / "2_preprocess" / "outputs" / "shape_outlier_report.csv"
STEP4_MANUAL_OUTLIER_CSV = PROJECT_ROOT / "2_preprocess" / "manual_outliers.csv"
STEP_DIR = _HERE  # 산출물 저장 위치(4_hi_analysis/cell/, 4_hi_analysis/segment/) —
                  # 옛 tools/ 스크립트는 한 단계 더 깊어서 parent.parent였지만
                  # 여기선 이 파일 자체가 이미 4_hi_analysis/ 안에 있다.

CYCLE_RANK_CMAP = "RdYlGn_r"  # 사이클 진행(초기=초록→말기=빨강) — plot_cell_cycles.py/
                              # plot_cycle_segments.py 둘 다 인라인으로 같은 값을 썼다.

# matplotlib의 pyplot 전역 상태(plt.figure/plt.savefig가 암묵적으로 참조하는
# "현재 figure")는 스레드 세이프하지 않다(공식 문서에 명시된 제약) —
# _plot_step4_dataset_cell_cycle_overlay가 ThreadPoolExecutor로 여러 셀을
# 동시에 그릴 때 이 락으로 "그림 그리기+저장" 구간만 직렬화한다(데이터 로드/
# 계산은 계속 병렬). 2026-10-01, 실측으로 레이스(RendererAgg 에러) 확인 후 추가.
_STEP4_PLOT_LOCK = threading.Lock()

# SOC-zone 배경 밴드 색상(plot_cycle_segments.py 전용) — constants.py의
# CHG_SEGS/DIS_SEGS(세그먼트 *이름* 리스트, q_frac_ref 6-시나리오)와 이름이
# 겹쳐서 혼동하지 않도록 STEP4_ 접두사를 붙인다. 옛 3구간(0~40/40~70/70~100%)
# 스킴 — 지금 파이프라인의 정식 q_frac_ref 6-시나리오와는 별개(이 진단 플롯
# 전용 간이 구간).
_STEP4_SEG_BOUNDS = [0.0, 0.4, 0.7, 1.0]
STEP4_CHG_ZONE_COLORS = [
    ("#f9e79f", "Chg SoC 0~30%"),
    ("#a9dfbf", "Chg SoC 30~60%"),
    ("#aed6f1", "Chg SoC 60~100%"),
]
STEP4_DIS_ZONE_COLORS = [
    ("#aed6f1", "Dis SoC 60~100%"),
    ("#a9dfbf", "Dis SoC 30~60%"),
    ("#f9e79f", "Dis SoC 0~30%"),
]

# _plot_step4_axis_aware_cycle_segments 전용(2026-10-05) — 시나리오마다 다른 색
# 계열(순서대로 순환)을 배정하고, 같은 시나리오 안의 세그먼트는 순서에 따라
# 옅음→진함 그라데이션을 준다(scenario_id % len(...)로 색 계열 선택).
_STEP4_SEG_CMAPS = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens,
                    plt.cm.Purples, plt.cm.Reds, plt.cm.YlOrBr]


def _plot_step4_correlation_heatmap_panel(ax, keys, title, corr_df, hi_labels: dict, datasets=("MIT", "HUST")):
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


def _plot_step4_correlation(corr_df: pd.DataFrame, df: pd.DataFrame,
                     out_path: Path, hi_groups: dict, hi_labels: dict,
                     hi_group_tag: dict, n_top: int = 4, datasets: list | None = None):
    datasets = list(datasets) if datasets is not None else ["MIT", "HUST"]
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")

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
            im_s, _ = _plot_step4_correlation_heatmap_panel(
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


def _plot_step4_sample_hi_trend(df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path,
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


# ─────────────────────────────────────────────────────────────────────────────
# 세그먼트별 HI 열화 추이 / 시나리오 오버레이 — hi_segment_viz.py에서 합류
# (2026-09-29, 원본 파일은 삭제). HI_GROUPS를 모듈 전역으로 import하지 않고
# hi_groups 인자로 받는 것도 위와 동일한 이유(stale-import 방지).
# ─────────────────────────────────────────────────────────────────────────────

_CAT_PREFIXES = ("stat_", "diff_", "lfp_", "morph_")

# 카테고리 메타
CATEGORIES = [
    ("Stat",  "카테고리 A: 통계 기반 (S01–S20)",         "hi_segment_trend_stat.png"),
    ("Diff",  "카테고리 B: 미분 기반 (D01–D20)",         "hi_segment_trend_diff.png"),
    ("LFP",   "카테고리 C: LFP 특징 기반 (L01–L20)",    "hi_segment_trend_lfp.png"),
    ("Morph", "카테고리 D: 형태학적 거리 (M01–M06)",    "hi_segment_trend_morph.png"),
]
OVERLAY_CATEGORIES = [
    ("Stat",  "카테고리 A: 통계 기반 (S01–S20)",         "hi_overlay_stat.png"),
    ("Diff",  "카테고리 B: 미분 기반 (D01–D20)",         "hi_overlay_diff.png"),
    ("LFP",   "카테고리 C: LFP 특징 기반 (L01–L20)",    "hi_overlay_lfp.png"),
    ("Morph", "카테고리 D: 형태학적 거리 (M01–M06)",    "hi_overlay_morph.png"),
]

# 방향별 색상 팔레트 (dis=파랑 계열, chg=주황 계열)
_DIS_SHADES = ["#AED6F1", "#5DADE2", "#2471A3", "#1A5276", "#0E3460",
               "#7FB3D3", "#3498DB", "#1F618D"]
_CHG_SHADES = ["#FAD7A0", "#F0A500", "#E67E22", "#CA6F1E", "#7D3C98",
               "#F5CBA7", "#DC7633", "#A04000"]


def _compute_step4_plain_label(key: str) -> str:
    """HI 컬럼 이름에서 직관적인 스네이크 표기 라벨 추출.

    stat_v_mean_cw_dis_hi → v_mean_cw
    lfp_inflect_v_chg_lo  → inflect_v
    """
    for prefix in _CAT_PREFIXES:
        if key.startswith(prefix):
            without_prefix = key[len(prefix):]
            parts = without_prefix.rsplit("_", 2)
            return parts[0] if len(parts) == 3 else without_prefix
    return key


def _compute_step4_seg_meta(hi_groups: dict, category: str) -> tuple:
    """hi_groups에서 활성 세그먼트 목록과 시각화 메타(색상/레이블) 동적 생성.

    q_frac_ref(dis_hi/mid/lo, chg_lo/mid/hi) 세그먼트 이름 기준으로 동작.

    Returns
    -------
    seg_order    : list[str]        활성 세그먼트 이름 목록 (dis 먼저, chg 다음)
    seg_row_bg   : list[str]        행 배경색 (dis=청, chg=주황)
    seg_row_label: list[str]        행 레이블 (세그먼트 이름 그대로)
    scen_colors  : dict[str, str]   세그먼트별 색상 hex
    scen_labels  : dict[str, str]   세그먼트별 범례 레이블
    """
    seg_order = list(dict.fromkeys(
        g.split(" — ")[0] for g in hi_groups if f" — {category}" in g
    ))
    if not seg_order:
        return [], [], [], {}, {}

    dis_segs = [s for s in seg_order if s.startswith("dis")]
    chg_segs = [s for s in seg_order if s.startswith("chg")]

    seg_row_bg    = ["#eaf4fb" if s.startswith("dis") else "#fef5eb" for s in seg_order]
    seg_row_label = seg_order[:]

    scen_colors: dict[str, str] = {}
    scen_labels: dict[str, str] = {}
    for i, s in enumerate(dis_segs):
        scen_colors[s] = _DIS_SHADES[i % len(_DIS_SHADES)]
        scen_labels[s] = s
    for i, s in enumerate(chg_segs):
        scen_colors[s] = _CHG_SHADES[i % len(_CHG_SHADES)]
        scen_labels[s] = s

    return seg_order, seg_row_bg, seg_row_label, scen_colors, scen_labels


def _plot_step4_segment_hi_trend_panel(ax, df, hi_key):
    """단일 (세그먼트, HI) 산점도 + 셀별 궤적."""
    any_data = False
    for ds, color in DATASET_COLORS.items():
        sub = df[df["dataset"] == ds][["cell_id", hi_key, "capacity_Ah"]].dropna()
        if len(sub) == 0:
            continue
        any_data = True
        for _, grp in sub.groupby("cell_id"):
            grp_s = grp.sort_values("capacity_Ah", ascending=False)
            ax.plot(grp_s["capacity_Ah"], grp_s[hi_key],
                    color=color, lw=0.5, alpha=0.18)
        ax.scatter(sub["capacity_Ah"], sub[hi_key],
                   color=color, s=0.6, alpha=0.22)
    if not any_data:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, alpha=0.35)


def _plot_step4_segment_hi_trend(df: pd.DataFrame, out_path: Path,
                          category: str, cat_title: str, hi_groups: dict):
    """N구간 × M HI 그리드 -한 카테고리(Stat/Diff/LFP/Morph).

    세그먼트 이름은 hi_groups에서 동적 추출한다.
    """
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")
    is_morph = (category == "Morph")

    seg_order, seg_row_bg, seg_row_label, _, _ = _compute_step4_seg_meta(hi_groups, category)
    seg_keys_list = [
        (seg, hi_groups.get(f"{seg} — {category}", []))
        for seg in seg_order
    ]
    seg_keys_list = [(s, ks) for s, ks in seg_keys_list if ks]
    if not seg_keys_list:
        print(f"  [trend] {category} 카테고리 HI 없음, 건너뜀")
        return

    n_segs = len(seg_keys_list)
    n_his  = len(seg_keys_list[0][1])
    col_labels = [_compute_step4_plain_label(k) for k in seg_keys_list[0][1]]

    cell_w = 5.5 if is_morph else 3.2
    cell_h = 3.8 if is_morph else 3.0
    fig, axes = plt.subplots(
        n_segs, n_his,
        figsize=(n_his * cell_w, n_segs * cell_h),
        squeeze=False,
    )

    morph_note = (
        "\n( y=0: BOL 기준곡선과 동일,  열화 진행 → 거리 증가 )"
        if is_morph else ""
    )
    fig.suptitle(
        f"세그먼트별 HI 열화 추이 -{cat_title}{morph_note}\n"
        "( 행=세그먼트,  열=HI 종류,  x=Capacity Ah )\n"
        "■ 파란 계열=MIT   ■ 주황 계열=HUST",
        fontsize=13, fontweight="bold",
    )

    for ci, lbl in enumerate(col_labels):
        axes[0, ci].set_title(lbl, fontsize=10, fontweight="bold", pad=4)

    for ri, (seg, hi_keys) in enumerate(seg_keys_list):
        bg      = seg_row_bg[ri]    if ri < len(seg_row_bg)    else "#f0f0f0"
        row_lbl = seg_row_label[ri] if ri < len(seg_row_label) else seg

        for ci, hi_key in enumerate(hi_keys):
            ax = axes[ri, ci]
            ax.set_facecolor(bg)
            _plot_step4_segment_hi_trend_panel(ax, df, hi_key)
            ax.set_xlabel("Cap (Ah)", fontsize=8)
            if is_morph:
                ax.set_ylim(bottom=0)
                ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.55, zorder=0)

        y_unit = "dist." if is_morph else _compute_step4_plain_label(hi_keys[0])
        axes[ri, 0].set_ylabel(f"{row_lbl}\n{y_unit}", fontsize=9, labelpad=4)
        for ci in range(1, n_his):
            axes[ri, ci].set_ylabel(_compute_step4_plain_label(hi_keys[ci]), fontsize=8)

    handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=ds)
        for ds, c in DATASET_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower right",
               fontsize=9, framealpha=0.85,
               bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


def _plot_step4_segment_hi_overlay(df: pd.DataFrame, out_path: Path,
                             category: str, cat_title: str, hi_groups: dict) -> None:
    """한 서브플랏에 1 HI의 N개 시나리오 열화 추이를 동시 표시.

    세그먼트 이름은 hi_groups에서 동적 추출한다.
    dis 계열: 파란 계열 / chg 계열: 주황 계열 (팔레트 자동 배정)
    MIT = 실선 / HUST = 점선
    """
    df = df.copy()
    df["dataset"] = df["dataset"].replace("MIT_MAT", "MIT")

    seg_order, _, _, scen_colors, scen_labels = _compute_step4_seg_meta(hi_groups, category)
    if not seg_order:
        print(f"  [overlay] {category} 카테고리 세그먼트 없음, 건너뜀")
        return

    ref_seg   = seg_order[0]
    ref_group = hi_groups.get(f"{ref_seg} — {category}", [])
    if not ref_group:
        print(f"  [overlay] {category} 그룹 키 없음, 건너뜀")
        return

    seg_suffix = f"_{ref_seg}"
    base_names = [k[: -len(seg_suffix)] for k in ref_group if k.endswith(seg_suffix)]
    n_his = len(base_names)
    if n_his == 0:
        return

    is_morph = (category == "Morph")
    ncols    = 3 if is_morph else 5
    nrows    = (n_his + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.0, nrows * 3.2),
        squeeze=False,
    )
    fig.patch.set_facecolor("#f5f5f5")
    fig.suptitle(
        f"시나리오 오버레이 -{cat_title}\n"
        "서브플랏 = 1 HI,  N개 시나리오 동시 표시  (x = Capacity Ah)\n"
        "dis: 파란 계열 ●  |  chg: 주황 계열 ●  |  실선 = MIT  /  점선 = HUST",
        fontsize=12, fontweight="bold",
    )

    cap_all = df["capacity_Ah"].dropna()
    cap_lo  = float(cap_all.quantile(0.01)) if len(cap_all) else 0.0
    cap_hi  = float(cap_all.quantile(0.99)) if len(cap_all) else 2.0
    n_bins  = 40

    def _median_trend(sub_df, key):
        sub = sub_df[["capacity_Ah", key]].dropna()
        if len(sub) < 5:
            return np.array([]), np.array([])
        bins = np.linspace(cap_lo, cap_hi, n_bins + 1)
        mids = (bins[:-1] + bins[1:]) / 2
        meds = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            seg_vals = sub.loc[(sub["capacity_Ah"] >= lo) & (sub["capacity_Ah"] < hi), key]
            meds.append(np.nanmedian(seg_vals) if len(seg_vals) >= 3 else np.nan)
        meds  = np.array(meds, dtype=float)
        valid = np.isfinite(meds)
        return mids[valid], meds[valid]

    for ai, base in enumerate(base_names):
        ax = axes[ai // ncols][ai % ncols]
        ax.set_facecolor("white")
        has_data = False

        for scen in seg_order:
            full_key = f"{base}_{scen}"
            if full_key not in df.columns:
                continue
            color    = scen_colors.get(scen, "#888888")
            scen_lbl = scen_labels.get(scen, scen)

            _first_ds_with_data = None
            for ds, ls in DATASET_LINESTYLE.items():
                sub = df[df["dataset"] == ds][
                    ["cell_id", full_key, "capacity_Ah"]
                ].dropna()
                if len(sub) == 0:
                    continue
                has_data = True
                if _first_ds_with_data is None:
                    _first_ds_with_data = ds

                for _, grp in sub.groupby("cell_id"):
                    grp_s = grp.sort_values("capacity_Ah", ascending=False)
                    ax.plot(grp_s["capacity_Ah"], grp_s[full_key],
                            color=color, lw=0.7, alpha=0.18, ls=ls)

                mx, my = _median_trend(sub, full_key)
                if len(mx) >= 2:
                    lbl = f"{scen_lbl} / {ds}" if ds == _first_ds_with_data else None
                    ax.plot(mx, my, color=color, lw=2.0, alpha=0.85,
                            ls=ls, label=lbl, zorder=3)

        if not has_data:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

        plain = _compute_step4_plain_label(f"{base}_{ref_seg}")
        ax.set_title(plain, fontsize=8.5, fontweight="bold", pad=3)
        ax.set_xlabel("Cap (Ah)", fontsize=7)
        ax.set_ylabel(plain, fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, lw=0.3, alpha=0.35)
        ax.set_xlim(cap_lo, cap_hi)

        if is_morph:
            ax.set_ylim(bottom=0)
            ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=0)

    for ai in range(n_his, nrows * ncols):
        axes[ai // ncols][ai % ncols].set_visible(False)

    handles = []
    for scen in seg_order:
        handles.append(
            plt.Line2D([0], [0], color=scen_colors.get(scen, "#888888"), lw=2.2, ls="-",
                       label=scen_labels.get(scen, scen))
        )
    handles += [
        plt.Line2D([0], [0], color="dimgray", lw=2.0, ls="-",  label="MIT (실선)"),
        plt.Line2D([0], [0], color="dimgray", lw=2.0, ls="--", label="HUST (점선)"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=8.5,
               framealpha=0.90, bbox_to_anchor=(1.0, 0.0), ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  저장: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Step4 진단 플롯 — tools/plot_cell_cycles.py 흡수(2026-10-01, 1단계)
# ─────────────────────────────────────────────────────────────────────────────

def _load_step4_cell_pkl(pkl_path: Path) -> tuple:
    """원본(전처리 전) 셀 pkl 로드. 반환: (meta, cycles_df)."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    return raw["meta"], raw["cycles"]


def _compute_step4_qfrac(phase_df: pd.DataFrame):
    """phase(충전 또는 방전) 구간의 누적 전하량 비율(q_frac) 계산 — 옛
    plot_cell_cycles.py::compute_qfrac와 plot_cycle_segments.py::compute_seg_times가
    각자 구현하던 같은 전하량 적분(2026-10-01, 하나로 통합). 포인트가 10개 미만이거나
    총 전하량이 0.05Ah 미만이면 None(두 원본 함수의 조기 종료 조건 그대로).

    반환: (t, q_frac, v, q_tot) — t/q_frac/v는 원본 포인트 배열, q_tot은 스칼라.
    오버레이 플롯(_plot_step4_cell_cycle_overlay_panel)은 q_frac/v만, 세그먼트
    밴드 플롯(_plot_step4_cycle_segments)은 t/q_tot까지 전부 쓴다.
    """
    if len(phase_df) < 10:
        return None
    t = phase_df["time_s"].values.astype(float)
    v = phase_df["voltage_V"].values.astype(float)
    i = np.abs(phase_df["current_A"].values.astype(float))
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    q_cum = np.cumsum(i * dt) / 3600.0
    q_tot = float(q_cum[-1])
    if q_tot < 0.05:
        return None
    return t, q_cum / q_tot, v, q_tot


def _load_step4_flagged_cycles(dataset: str, cell_id: str, z_thresh: float,
                                dev_thresh: dict, maxdev_thresh: dict,
                                flat_thresh: dict, fhigh_thresh: dict) -> tuple:
    """2_preprocess 형상-이상치 진단 CSV(STEP4_OUTLIER_CSV)에서 제거 후보 사이클을
    phase별 집합으로 반환. 조건(넷 중 하나라도 충족):
      - 전체 형상 붕괴 : z     > z_thresh AND dev     > dev_thresh[phase]
      - 국소 돌출/글리치: z_max > z_thresh AND max_dev > maxdev_thresh[phase]
      - 전체 평탄선     : v_span    < flat_thresh[phase]    (값>0 일 때만)
      - 3.6V 체류형     : frac_high > fhigh_thresh[phase]   (값>0 일 때만)
    반환: ({"discharge": set(...), "charge": set(...)}, csv_존재_여부).
    """
    empty = {"discharge": set(), "charge": set()}
    if not STEP4_OUTLIER_CSV.exists():
        return empty, False
    rep = pd.read_csv(STEP4_OUTLIER_CSV)
    has_max   = "z_max" in rep.columns and "max_dev" in rep.columns
    has_span  = "v_span" in rep.columns
    has_fhigh = "frac_high" in rep.columns
    base = rep[(rep["dataset"].str.upper() == dataset.upper())
               & (rep["cell_id"] == cell_id)]
    flagged = dict(empty)
    for phase in ("discharge", "charge"):
        ph = base[base["phase"] == phase]
        cond = (ph["z"] > z_thresh) & (ph["dev"] > dev_thresh[phase])
        if has_max:
            cond = cond | ((ph["z_max"] > z_thresh)
                           & (ph["max_dev"] > maxdev_thresh[phase]))
        if has_span and flat_thresh[phase] > 0:
            cond = cond | (ph["v_span"] < flat_thresh[phase])
        if has_fhigh and fhigh_thresh[phase] > 0:
            cond = cond | (ph["frac_high"] > fhigh_thresh[phase])
        flagged[phase] = set(ph[cond]["cycle"].astype(int))
    return flagged, True


def _load_step4_manual_flags(dataset: str, cell_id: str) -> dict:
    """사람이 손으로 적은 강제 제거 목록(STEP4_MANUAL_OUTLIER_CSV)을 phase별
    집합으로 반환 — 지표 임계로 못 잡거나 과하게 잡히는 개별 사이클을
    (dataset,cell_id,phase,cycle)로 지정하면 임계와 무관하게 항상 반영된다."""
    manual = {"discharge": set(), "charge": set()}
    if not STEP4_MANUAL_OUTLIER_CSV.exists():
        return manual
    with open(STEP4_MANUAL_OUTLIER_CSV, encoding="utf-8-sig") as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or ln == 0:
                continue
            parts = line.split(",", 4)
            if len(parts) < 4:
                continue
            ds, cid, phase, cyc = (p.strip() for p in parts[:4])
            if ds.upper() != dataset.upper() or cid != str(cell_id):
                continue
            if phase not in manual:
                continue
            try:
                manual[phase].add(int(cyc))
            except ValueError:
                continue
    return manual


def _plot_step4_cell_cycle_overlay_panel(ax, df, cycles, phase, cmap, norm,
                                          flagged=None) -> None:
    """한 phase(충전 또는 방전)의 전체 사이클 V-q_frac 오버레이를 ax 하나에 그림
    (2단계 — _plot_step4_cell_cycle_overlay가 호출). flagged(제거 후보 사이클
    집합)는 굵은 검은선으로 맨 위에 강조."""
    flagged = flagged or set()
    rank_of = {cyc: rank for rank, cyc in enumerate(cycles)}
    groups = df[df["phase"] == phase].groupby("cycle")

    n_flag = 0
    flag_curves = []
    for cyc, cyc_df in groups:
        result = _compute_step4_qfrac(cyc_df)
        if result is None:
            continue
        _, q_frac, v, _ = result
        if cyc in flagged:
            flag_curves.append((q_frac, v))
            continue
        ax.plot(q_frac, v, color=cmap(norm(rank_of.get(cyc, 0))), lw=0.6, alpha=0.6)

    for q_frac, v in flag_curves:
        n_flag += 1
        ax.plot(q_frac, v, color="black", lw=1.8, alpha=0.95, zorder=5,
                label="제거 후보" if n_flag == 1 else None)

    phase_label = "방전" if phase == "discharge" else "충전"
    title = f"전체 사이클 {phase_label} V-q_frac 오버레이"
    if flagged:
        title += f"  (제거 후보 {n_flag}건)"
    ax.set_xlabel("q_frac (누적 용량 비율)", fontsize=10)
    ax.set_ylabel("Voltage (V)", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    if n_flag:
        ax.legend(fontsize=8, loc="best")


def _plot_step4_cell_cycle_overlay(
    dataset: str, cell_id: str,
    z_thresh: float = 6.0,
    dev_thresh: dict | None = None,
    maxdev_thresh: dict | None = None,
    flat_thresh: dict | None = None,
    fhigh_thresh: dict | None = None,
) -> Path:
    """셀 하나의 전체 사이클을 한 그림(1단계)으로 — 위: 사이클별 방전 용량
    (열화 곡선, 제거 후보는 검은 X), 아래: 전체 사이클 방전/충전 V-q_frac
    오버레이(좌/우, 초기=초록→말기=빨강, 제거 후보는 굵은 검은선). 옛
    tools/plot_cell_cycles.py::main() — 구조·출력 파일명 그대로 유지.

    `z_thresh`/`dev_thresh`/`maxdev_thresh`는 2_preprocess 형상-이상치 진단과
    연동된 제거 후보 표시 임계값(parameters.py의 FIXED_STEP4_DIAG_*가 기본값
    공급원). `flat_thresh`/`fhigh_thresh`는 원본부터 "기본 비활성"(0.0=끔)이라
    parameters.py로 승격하지 않고 함수 기본값 그대로 둠 — 필요하면 호출부에서
    직접 넘긴다.
    """
    dev_thresh = dev_thresh or {"discharge": 0.25, "charge": 0.07}
    maxdev_thresh = maxdev_thresh or {"discharge": 0.40, "charge": 0.25}
    flat_thresh = flat_thresh or {"discharge": 0.0, "charge": 0.0}
    fhigh_thresh = fhigh_thresh or {"discharge": 0.0, "charge": 0.0}

    data_dir = _resolve_step4_dataset_dir(dataset)
    pkl_path = data_dir / f"{cell_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL 파일 없음: {pkl_path}")

    meta, df = _load_step4_cell_pkl(pkl_path)
    cell_id = meta.get("cell_id", cell_id)
    cycles = sorted(df["cycle"].unique())
    n_cycles = len(cycles)

    flagged, has_csv = _load_step4_flagged_cycles(
        dataset, cell_id, z_thresh, dev_thresh, maxdev_thresh, flat_thresh, fhigh_thresh)
    manual = _load_step4_manual_flags(dataset, cell_id)
    n_manual = len(manual["discharge"]) + len(manual["charge"])
    for phase in ("discharge", "charge"):
        flagged[phase] = flagged[phase] | manual[phase]
    flagged_any = flagged["discharge"] | flagged["charge"]

    dis_all = df[df["phase"] == "discharge"]
    cap_by_cyc = dis_all.groupby("cycle")["capacity_Ah"].first().reindex(cycles)

    cmap = matplotlib.colormaps[CYCLE_RANK_CMAP]
    norm = mcolors.Normalize(vmin=0, vmax=n_cycles - 1)

    out_dir = STEP_DIR / "cell"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"cell_cycles_{dataset.lower()}_{cell_id}.png"

    # matplotlib pyplot 전역 상태는 스레드 세이프하지 않다 — 이 구간(그림 생성~저장)만
    # _STEP4_PLOT_LOCK으로 직렬화한다(위 상수 docstring/_plot_step4_dataset_cell_cycle_overlay
    # 참고). 데이터 로드·필터링(위쪽)은 락 밖이라 계속 병렬로 돈다.
    with _STEP4_PLOT_LOCK:
        fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
        ax_cap = fig.add_subplot(gs[0, :])
        ax_dis = fig.add_subplot(gs[1, 0])
        ax_chg = fig.add_subplot(gs[1, 1])
        fig.suptitle(f"Cell: {cell_id}  |  전체 {n_cycles} 사이클", fontsize=12, fontweight="bold")

        colors_cap = [cmap(norm(i)) for i in range(n_cycles)]
        ax_cap.scatter(cycles, cap_by_cyc.values, c=colors_cap, s=8, zorder=2)
        ax_cap.plot(cycles, cap_by_cyc.values, color="gray", lw=0.6, alpha=0.5, zorder=1)
        if flagged_any:
            cap_map = cap_by_cyc.to_dict()
            fx = sorted(c for c in flagged_any if c in cap_map and not np.isnan(cap_map[c]))
            if fx:
                ax_cap.scatter(fx, [cap_map[c] for c in fx],
                               color="black", s=40, marker="x", lw=1.2, zorder=4,
                               label=f"제거 후보 ({len(fx)}건)")
                ax_cap.legend(fontsize=8, loc="best")
        ax_cap.set_ylabel("Capacity (Ah)", fontsize=10)
        ax_cap.set_xlabel("Cycle", fontsize=10)
        ax_cap.set_title("사이클별 방전 용량", fontsize=10)
        ax_cap.grid(True, alpha=0.3)
        ax_cap.tick_params(labelsize=9)

        _plot_step4_cell_cycle_overlay_panel(ax_dis, df, cycles, "discharge", cmap, norm,
                                              flagged["discharge"])
        _plot_step4_cell_cycle_overlay_panel(ax_chg, df, cycles, "charge", cmap, norm,
                                              flagged["charge"])

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_cap, ax_dis, ax_chg], fraction=0.02, pad=0.01)
        cbar.set_label("Cycle rank (초기→말기)", fontsize=9)
        cbar.set_ticks([0, n_cycles - 1])
        cbar.set_ticklabels([str(cycles[0]), str(cycles[-1])])

        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

    msg = f"  저장: {out}"
    if has_csv:
        msg += (f"  | 제거 후보 방전={len(flagged['discharge'])} "
                f"충전={len(flagged['charge'])} (수동={n_manual})")
    else:
        msg += "  | (CSV 없음 — 제거 후보 표시 생략)"
    print(msg)
    return out


def _plot_step4_dataset_cell_cycle_overlay(dataset: str, n_workers: int = 1) -> None:
    """데이터셋 전체 셀에 대해 `_plot_step4_cell_cycle_overlay`를 한 번씩 호출
    (1단계 — "데이터셋 전체 셀" → "셀 하나"의 2단계 구조 그 자체). 옛
    tools/plot_all_mit_cells.py — subprocess로 plot_cell_cycles.py를 셀마다 새
    프로세스로 띄우던 걸 프로세스 내 직접 호출(ThreadPoolExecutor)로 교체해
    훨씬 가볍다(프로세스 기동 비용 제거, matplotlib Agg 백엔드라 스레드로 안전).
    """
    data_dir = _resolve_step4_dataset_dir(dataset)
    cells = sorted(p.stem for p in data_dir.glob("*.pkl"))
    if not cells:
        raise FileNotFoundError(f"PKL 파일 없음: {data_dir}")

    n = len(cells)
    print(f"  총 {n}개 {dataset} 셀 시각화 시작 (workers={n_workers})")

    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_plot_step4_cell_cycle_overlay, dataset, cell): cell
                   for cell in cells}
        for future in as_completed(futures):
            cell = futures[future]
            done += 1
            try:
                future.result()
                print(f"  [{done}/{n}] 완료 {cell}")
            except Exception as e:
                failed.append(cell)
                print(f"  [{done}/{n}] 실패 {cell}: {e}")

    print(f"  완료: {n - len(failed)}/{n} 성공")
    if failed:
        print(f"  실패 셀: {', '.join(sorted(failed))}")


def _plot_step4_segment_bands_panel(ax, t, boundary_times, seg_defs,
                                     label_top: bool = False) -> None:
    """세그먼트 배경 밴드 + 경계선 하나를 ax에 그림(2단계 — SOC-zone 음영,
    `_plot_step4_cycle_segments`가 V/I 패널 각각에 대해 호출). boundary_times는
    [t_0.4, t_0.7](옛 3구간 스킴 경계)."""
    t_bounds = [t[0]] + list(boundary_times) + [t[-1]]
    for i, (color, label) in enumerate(seg_defs):
        t0, t1 = t_bounds[i], t_bounds[i + 1]
        ax.axvspan(t0, t1, color=color, alpha=0.28, zorder=0)
        if label_top:
            tc = (t0 + t1) / 2
            ax.text(tc, 0.98, label,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5,
                    color="#444444", fontweight="bold", linespacing=1.3)
    for tb in boundary_times:
        ax.axvline(tb, color="#999999", lw=1.0, ls="--", zorder=1)


def _plot_step4_cycle_segments(dataset: str, cell_id: str, cycle: int) -> Path:
    """사이클 하나의 충전→방전 세그먼트를 3행 그림(1단계)으로 — 위: V-vs-time_s
    (SOC-zone 밴드), 중간: I-vs-time_s(〃), 아래: 전 사이클 방전 용량 열화 곡선
    (선택 사이클 강조). 옛 tools/plot_cycle_segments.py::main() — 구조·출력
    파일명 그대로 유지."""
    data_dir = _resolve_step4_dataset_dir(dataset)
    pkl_path = data_dir / f"{cell_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL 파일 없음: {pkl_path}")

    meta, df_all = _load_step4_cell_pkl(pkl_path)
    cell_id = meta.get("cell_id", cell_id)

    cyc_df = df_all[df_all["cycle"] == cycle]
    if len(cyc_df) == 0:
        available = sorted(df_all["cycle"].unique())
        raise ValueError(f"cycle {cycle} 없음. 사용 가능: {available[:10]}...")

    dis_all = df_all[df_all["phase"] == "discharge"]
    cap_ser = dis_all.groupby("cycle")["capacity_Ah"].first().dropna().sort_index()
    all_cycs = cap_ser.index.to_numpy()
    all_caps = cap_ser.values
    sel_cap = float(cap_ser[cycle]) if cycle in cap_ser.index else np.nan

    chg_df = cyc_df[cyc_df["phase"] == "charge"]
    dis_df = cyc_df[cyc_df["phase"] == "discharge"]

    def _seg_times(phase_df):
        """_compute_step4_qfrac 결과에서 _STEP4_SEG_BOUNDS 경계 시각만 뽑아낸다
        — (t, q_tot, [t_0.4, t_0.7]) 형태로, 옛 compute_seg_times()의 반환과 동일."""
        result = _compute_step4_qfrac(phase_df)
        if result is None:
            return None
        t, q_frac, _, q_tot = result
        boundary_times = [
            float(t[int(np.clip(np.searchsorted(q_frac, b), 0, len(t) - 1))])
            for b in _STEP4_SEG_BOUNDS[1:-1]
        ]
        return t, q_tot, boundary_times

    chg_info = _seg_times(chg_df)
    dis_info = _seg_times(dis_df)

    fig, (ax_v, ax_i, ax_c) = plt.subplots(
        3, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [2, 1, 1.2], "hspace": 0.42},
        constrained_layout=False,
    )
    fig.subplots_adjust(top=0.93, bottom=0.10, left=0.08, right=0.97, hspace=0.42)
    fig.suptitle(
        f"Cell: {cell_id}  |  Cycle: {cycle}  "
        f"{'|  Q=' + f'{sel_cap:.4f} Ah' if np.isfinite(sel_cap) else ''}",
        fontsize=12, fontweight="bold",
    )

    for ax, label_top in [(ax_v, True), (ax_i, False)]:
        if chg_info is not None:
            t_chg, _, bt_chg = chg_info
            _plot_step4_segment_bands_panel(ax, t_chg, bt_chg, STEP4_CHG_ZONE_COLORS, label_top)
        if dis_info is not None:
            t_dis, _, bt_dis = dis_info
            _plot_step4_segment_bands_panel(ax, t_dis, bt_dis, STEP4_DIS_ZONE_COLORS, label_top)
        if chg_info is not None and dis_info is not None:
            t_sep = float(chg_info[0][-1])
            ax.axvline(t_sep, color="#c0392b", lw=1.8, zorder=3)

    if chg_info is not None:
        t_chg, q_tot_chg, _ = chg_info
        ax_v.scatter(t_chg, chg_df["voltage_V"].values, color="#2980b9",
                     s=2, label=f"Charge  Q={q_tot_chg:.4f} Ah")
        ax_i.scatter(t_chg, chg_df["current_A"].values, color="#2980b9", s=2)

    if dis_info is not None:
        t_dis, q_tot_dis, _ = dis_info
        ax_v.scatter(t_dis, dis_df["voltage_V"].values, color="#2c3e50",
                     s=2, label=f"Discharge  Q={q_tot_dis:.4f} Ah")
        ax_i.scatter(t_dis, dis_df["current_A"].values, color="#2c3e50", s=2)

    if chg_info is not None and dis_info is not None:
        t_chg_mid = (chg_info[0][0] + chg_info[0][-1]) / 2
        t_dis_mid = (dis_info[0][0] + dis_info[0][-1]) / 2
        for ax in (ax_v, ax_i):
            ax.text(t_chg_mid, 0.04, "← Charge →",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color="#2980b9", alpha=0.8)
            ax.text(t_dis_mid, 0.04, "← Discharge →",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color="#2c3e50", alpha=0.8)

    ax_i.axhline(0, color="black", lw=0.8, ls=":", zorder=1)

    ax_v.set_ylabel("Voltage (V)", fontsize=10)
    ax_v.legend(loc="lower right", fontsize=9)
    ax_v.grid(True, alpha=0.3)
    ax_v.tick_params(labelbottom=False, labelsize=9)

    ax_i.set_ylabel("Current (A)", fontsize=10)
    ax_i.set_xlabel("Time (s)", fontsize=10)
    ax_i.grid(True, alpha=0.3)
    ax_i.tick_params(labelsize=9)

    n_cycs = len(all_cycs)
    norm = plt.Normalize(vmin=0, vmax=max(n_cycs - 1, 1))
    cmap = matplotlib.colormaps[CYCLE_RANK_CMAP]

    ax_c.scatter(all_cycs, all_caps, c=[cmap(norm(i)) for i in range(n_cycs)],
                 s=4, zorder=2, alpha=0.7)
    ax_c.plot(all_cycs, all_caps, color="lightgray", lw=0.6, zorder=1)

    if np.isfinite(sel_cap):
        ax_c.scatter([cycle], [sel_cap], color="#e74c3c", s=120, zorder=5,
                     marker="*", label=f"Cycle {cycle}  ({sel_cap:.4f} Ah)")
        ax_c.axvline(cycle, color="#e74c3c", lw=1.2, ls="--", alpha=0.7, zorder=4)
        ax_c.annotate(
            f"  cycle {cycle}\n  {sel_cap:.4f} Ah",
            xy=(cycle, sel_cap),
            xytext=(cycle + max(n_cycs * 0.03, 5), sel_cap),
            fontsize=8.5, color="#c0392b", va="center",
            arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8),
        )

    if len(all_caps) > 0:
        cap_init = float(all_caps[0])
        eol_cap = cap_init * 0.80
        ax_c.axhline(eol_cap, color="#999999", lw=0.8, ls=":",
                     label=f"80% SOH ({eol_cap:.3f} Ah)")

    ax_c.set_xlabel("Cycle", fontsize=10)
    ax_c.set_ylabel("Discharge Capacity (Ah)", fontsize=10)
    ax_c.set_title("방전 용량 열화 곡선 (전 사이클)", fontsize=9, pad=4)
    ax_c.legend(fontsize=8.5, loc="upper right")
    ax_c.grid(True, alpha=0.3)
    ax_c.tick_params(labelsize=9)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_c, fraction=0.018, pad=0.01)
    cbar.set_label("Cycle rank", fontsize=8)
    if n_cycs > 1:
        cbar.set_ticks([0, n_cycs - 1])
        cbar.set_ticklabels([str(all_cycs[0]), str(all_cycs[-1])])

    patches = [mpatches.Patch(color=c, alpha=0.6, label=lbl)
               for c, lbl in STEP4_DIS_ZONE_COLORS]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.01))

    out_dir = STEP_DIR / "segment"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"segment_{dataset.lower()}_{cell_id}_cycle{cycle}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out}")
    return out


def _plot_step4_axis_aware_cycle_segments(dataset: str, cell_id: str, cycle: int) -> Path:
    """사이클 하나의 충전/방전 구간을, **지금 활성화된 `P.FIXED_SEG_AXIS` +
    `P.ACTIVE_AXIS_CONFIG`로 실제 세그멘터**(`get_segmenter(...)._extract`)를 돌려
    진짜 세그먼트 경계를 그린다. 옵션 3(`_plot_step4_cycle_segments`)은 하드코딩된
    `_STEP4_SEG_BOUNDS=[0,0.4,0.7,1.0]`(범용 3-zone 근사 스케치)를 쓰기 때문에
    `assign="none"`(no_scen)처럼 존 구조 자체가 달라지는 축 변형을 반영하지
    못한다 — 이 함수는 공식 생성 경로(`get_segmenter`)를 그대로 써서 그 문제를
    피한다(2026-10-05 신설, no_scen 검증 중 발견).

    세그먼트 경계(q_frac_lo/hi)는 `segmenter._extract()`가 실제로 반환하는
    `SegmentRecord.meta`에서 그대로 가져온다 — 축마다 몇 개/어떤 폭의 세그먼트가
    나오는지까지 정확하다. `scenario_id`별로 색을 칠해 범례에 시나리오 이름을
    표시한다(no_scen이면 `chg`/`dis` 2개, 정식 축이면 6개)."""
    from common.scenario import get_segmenter

    data_dir = _resolve_step4_dataset_dir(dataset)
    pkl_path = data_dir / f"{cell_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL 파일 없음: {pkl_path}")
    meta, df_all = _load_step4_cell_pkl(pkl_path)
    cell_id = meta.get("cell_id", cell_id)
    cyc_df = df_all[df_all["cycle"] == cycle]
    if len(cyc_df) == 0:
        available = sorted(df_all["cycle"].unique())
        raise ValueError(f"cycle {cycle} 없음. 사용 가능: {available[:10]}...")

    segmenter = get_segmenter(P.FIXED_SEG_AXIS, {P.FIXED_SEG_AXIS: P.ACTIVE_AXIS_CONFIG})
    spec = segmenter.get_spec()

    def _phase_arrays(phase_df):
        """_compute_step4_qfrac과 동일 적분(t/v/i/dt/q), q는 비율이 아니라
        누적 Ah(segmenter._extract가 받는 그대로의 단위)."""
        if len(phase_df) < 10:
            return None
        t = phase_df["time_s"].values.astype(float)
        v = phase_df["voltage_V"].values.astype(float)
        i = np.abs(phase_df["current_A"].values.astype(float))
        dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
        q = np.cumsum(i * dt) / 3600.0
        return t, v, i, dt, q

    chg_arr = _phase_arrays(cyc_df[cyc_df["phase"] == "charge"])
    dis_arr = _phase_arrays(cyc_df[cyc_df["phase"] == "discharge"])
    chg_records = (segmenter._extract(chg_arr[1], chg_arr[2], chg_arr[3], chg_arr[4],
                                       1, cell_id, cycle, 0)[0] if chg_arr else [])
    dis_records = (segmenter._extract(dis_arr[1], dis_arr[2], dis_arr[3], dis_arr[4],
                                       -1, cell_id, cycle, 0)[0] if dis_arr else [])

    def _segment_colors(records: list) -> dict:
        """레코드(파이썬 객체 id)별 색 — 시나리오마다 다른 색 계열(파랑/주황/...)을
        쓰고, 같은 시나리오 안에서는 세그먼트 순서(q_frac_lo 기준)에 따라 옅은색
        → 진한색으로 그라데이션을 준다. no_scen처럼 한 시나리오에 세그먼트가
        여러 개 몰려도(원래는 전부 동일 단색) 순서를 눈으로 구분할 수 있다."""
        by_scen: dict[int, list] = {}
        for rec in sorted(records, key=lambda r: r.meta["q_frac_lo"]):
            by_scen.setdefault(rec.scenario_id, []).append(rec)
        colors: dict[int, tuple] = {}
        for scenario_id, recs in by_scen.items():
            cmap = _STEP4_SEG_CMAPS[scenario_id % len(_STEP4_SEG_CMAPS)]
            shades = np.linspace(0.35, 0.85, len(recs)) if len(recs) > 1 else [0.6]
            for rec, shade in zip(recs, shades):
                colors[id(rec)] = cmap(shade)
        return colors

    fig, (ax_v, ax_vq, ax_i) = plt.subplots(
        3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1.6, 1.6, 1]})
    fig.suptitle(
        f"Cell: {cell_id} | Cycle: {cycle} | axis={P.FIXED_SEG_AXIS} "
        f"| assign={P.ACTIVE_AXIS_CONFIG.get('assign', 'position_bin')}",
        fontsize=11, fontweight="bold")

    def _draw_phase_time(arr, records, t_offset, label, line_color):
        """V-t/I-t 패널 — 세그먼트 구간(실제 겹침 포함)을 그라데이션 배경으로 칠하고
        순번을 상단에 적는다."""
        if arr is None:
            return t_offset
        t, v, i, dt, q = arr
        t_shift = t - t[0] + t_offset
        ax_v.plot(t_shift, v, color=line_color, lw=0.9, label=f"{label} V", zorder=3)
        ax_i.plot(t_shift, i, color=line_color, lw=0.9, zorder=3)
        q_frac = q / q[-1] if q[-1] > 0 else q

        seg_colors = _segment_colors(records)
        v_top = float(np.nanmax(v)) if len(v) else 0.0
        for seg_no, rec in enumerate(sorted(records, key=lambda r: r.meta["q_frac_lo"]), start=1):
            lo_q, hi_q = rec.meta["q_frac_lo"], rec.meta["q_frac_hi"]
            idx_lo = int(np.clip(np.searchsorted(q_frac, lo_q), 0, len(t) - 1))
            idx_hi = int(np.clip(np.searchsorted(q_frac, hi_q), 0, len(t) - 1))
            t_lo, t_hi = t_shift[idx_lo], t_shift[idx_hi]
            color = seg_colors[id(rec)]
            for ax in (ax_v, ax_i):
                ax.axvspan(t_lo, t_hi, color=color, alpha=0.55, lw=0, zorder=1)
            ax_v.text((t_lo + t_hi) / 2, v_top, str(seg_no), ha="center", va="bottom",
                      fontsize=7, zorder=4)
        return float(t_shift[-1]) if len(t_shift) else t_offset

    def _draw_phase_qfrac(arr, records, x_offset, label, line_color):
        """V-q_frac 패널("vq curve") — 세그먼트 경계가 q_frac_lo/hi 그 자체라
        searchsorted 없이 정확하고, 시간축 왜곡(CC/CV 구간별 체류시간 차이) 없이
        세그먼트 폭이 설계한 그대로(n2 기준 균등) 보인다."""
        if arr is None:
            return x_offset
        t, v, i, dt, q = arr
        q_frac = (q / q[-1] if q[-1] > 0 else q) + x_offset
        ax_vq.plot(q_frac, v, color=line_color, lw=0.9, label=f"{label} V", zorder=3)

        seg_colors = _segment_colors(records)
        v_top = float(np.nanmax(v)) if len(v) else 0.0
        for seg_no, rec in enumerate(sorted(records, key=lambda r: r.meta["q_frac_lo"]), start=1):
            lo_q = rec.meta["q_frac_lo"] + x_offset
            hi_q = rec.meta["q_frac_hi"] + x_offset
            color = seg_colors[id(rec)]
            ax_vq.axvspan(lo_q, hi_q, color=color, alpha=0.55, lw=0, zorder=1)
            ax_vq.text((lo_q + hi_q) / 2, v_top, str(seg_no), ha="center", va="bottom",
                       fontsize=7, zorder=4)
        return x_offset + 1.0

    t_end = _draw_phase_time(chg_arr, chg_records, 0.0, "Charge", "#1a1a1a")
    _draw_phase_time(dis_arr, dis_records, t_end + 1.0, "Discharge", "#555555")
    x_end = _draw_phase_qfrac(chg_arr, chg_records, 0.0, "Charge", "#1a1a1a")
    _draw_phase_qfrac(dis_arr, dis_records, x_end + 0.15, "Discharge", "#555555")

    scen_handles = [plt.Line2D([0], [0], color=_STEP4_SEG_CMAPS[sid % len(_STEP4_SEG_CMAPS)](0.6),
                                lw=7, alpha=0.6, label=name)
                     for sid, name in enumerate(spec.scenario_names)]
    ax_v.legend(handles=scen_handles, loc="upper right", fontsize=7, ncol=3, framealpha=0.9,
                title="시나리오(색 계열) — 같은 계열 내 옅음→진함 = 세그먼트 순서",
                title_fontsize=7)
    ax_v.set_ylabel("Voltage (V)", fontsize=10)
    ax_v.set_title("V-t curve (실제 시간 — CC/CV 등 체류시간 차이로 세그먼트 폭이 달라 보임)",
                    fontsize=9)
    ax_v.grid(True, alpha=0.3)
    ax_v.tick_params(labelbottom=False, labelsize=9)

    ax_vq.set_ylabel("Voltage (V)", fontsize=10)
    ax_vq.set_xlabel("q_frac (충전/방전 각자 0→1, 둘 사이 약간 띄움)", fontsize=10)
    ax_vq.set_title("V-Q curve (q_frac 기준 — 세그먼트 폭이 설계한 그대로 균등하게 보임)",
                     fontsize=9)
    ax_vq.grid(True, alpha=0.3)

    ax_i.set_ylabel("|Current| (A)", fontsize=10)
    ax_i.set_xlabel("time (s, phase별 재배치)", fontsize=10)
    ax_i.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = STEP_DIR / "segment"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"axis_segment_{dataset.lower()}_{cell_id}_cycle{cycle}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out}")
    return out


STEP4_DIAG_PLOTS = {
    1: ("셀 하나 사이클 오버레이+용량열화", _plot_step4_cell_cycle_overlay),
    2: ("데이터셋 전체 셀 사이클 오버레이", _plot_step4_dataset_cell_cycle_overlay),
    3: ("사이클 하나 SOC-zone 세그먼트(근사 3-zone 스케치)", _plot_step4_cycle_segments),
    4: ("사이클 하나 SOC-zone 세그먼트(현재 ACTIVE_AXIS_CONFIG 정확 반영)",
        _plot_step4_axis_aware_cycle_segments),
}


def main() -> None:
    """Step4 진단 플롯 번호 선택형 진입점 — 옛 tools/plot_cell_cycles.py/
    plot_cycle_segments.py/plot_all_mit_cells.py 각자의 argparse CLI를 대체한다.
    CLI는 "어떤 플롯이냐" 번호 하나만 받고, dataset/cell/cycle 등 실제 파라미터는
    parameters.py(FIXED_STEP4_DIAG_*)에서만 읽는다(run_pipeline.py와 동일 원칙).

    실행: python 4_hi_analysis/plot.py <번호>  (인자 없으면 선택 가능한 번호 목록 출력)
    """
    parser = argparse.ArgumentParser(description="Step4 진단 플롯 선택 실행")
    parser.add_argument("which", nargs="?", type=int, default=None,
                        help="실행할 플롯 번호 (생략 시 목록만 출력)")
    args = parser.parse_args()

    if args.which is None:
        print("[plot] 사용 가능한 Step4 진단 플롯:")
        for n, (desc, _) in STEP4_DIAG_PLOTS.items():
            print(f"  {n}) {desc}")
        print("실행: python 4_hi_analysis/plot.py <번호>")
        return

    if args.which not in STEP4_DIAG_PLOTS:
        raise SystemExit(f"알 수 없는 번호: {args.which} (선택: {list(STEP4_DIAG_PLOTS)})")

    desc, fn = STEP4_DIAG_PLOTS[args.which]
    print(f"[plot] {args.which}) {desc}")
    dataset = P.FIXED_STEP4_DIAG_DATASET
    if fn is _plot_step4_cell_cycle_overlay:
        fn(dataset, P.FIXED_STEP4_DIAG_CELL,
           z_thresh=P.FIXED_STEP4_DIAG_Z_THRESH,
           dev_thresh=P.FIXED_STEP4_DIAG_DEV_THRESH,
           maxdev_thresh=P.FIXED_STEP4_DIAG_MAXDEV_THRESH)
    elif fn is _plot_step4_dataset_cell_cycle_overlay:
        fn(dataset, n_workers=P.FIXED_STEP4_DIAG_WORKERS)
    elif fn in (_plot_step4_cycle_segments, _plot_step4_axis_aware_cycle_segments):
        fn(dataset, P.FIXED_STEP4_DIAG_CELL, P.FIXED_STEP4_DIAG_CYCLE)


# ─────────────────────────────────────────────────────────────────────────────
# 폴더 기반 진입점 — 2026-09-29 신규
# ─────────────────────────────────────────────────────────────────────────────

def _plot_step4_from_result_dir(step_dir: Path) -> None:
    """step_4_result/ 폴더 경로 하나만 받아서 그 안의 manifest.json/
    correlation.csv를 읽고, manifest가 가리키는 캐시에서 df를 로드해 전체
    플롯(히트맵/산점도/대표 셀 추이/세그먼트별 추이·오버레이)을 다시 그려
    같은 폴더에 저장한다.

    hi_correlation.py::main()이 추출 직후 이 함수를 자동으로 호출하지만,
    이 함수 자체는 그 실행과 완전히 독립적으로 폴더 경로만 있으면 언제든
    재실행할 수 있다(예: 플롯 스타일만 고쳐서 재생성). 산출물 이상탐지는
    아직 없다 — 보류(다음 라운드, docs/REFACTORING.md 참고).
    """
    manifest = json.loads((step_dir / "manifest.json").read_text(encoding="utf-8"))
    corr = pd.read_csv(step_dir / "correlation.csv", index_col=0)
    df = pd.read_pickle(manifest["cache_path"])

    hi_groups, _, hi_labels = L.build_hi_groups(manifest["seg_names"])
    hi_group_tag = {k: g for g, ks in hi_groups.items() for k in ks}
    datasets = manifest["datasets"]

    _plot_step4_correlation(corr, df, step_dir / "hi_correlation.png",
                     hi_groups, hi_labels, hi_group_tag, n_top=4, datasets=datasets)
    _plot_step4_sample_hi_trend(df, corr, step_dir, hi_labels, hi_group_tag, datasets=datasets)

    for cat, cat_title, fname in CATEGORIES:
        _plot_step4_segment_hi_trend(df, step_dir / fname, cat, cat_title, hi_groups)
    for cat, cat_title, fname in OVERLAY_CATEGORIES:
        _plot_step4_segment_hi_overlay(df, step_dir / fname, cat, cat_title, hi_groups)

    print(f"  Step4 플롯 전체 저장: {step_dir}")


if __name__ == "__main__":
    main()
