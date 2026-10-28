"""Figure 5 — hi_selection_matrix.png에서 "어느 시나리오에서도 선택 안 된" raw HI
행을 잘라낸 버전(2026-10-12). 64행 중 43행이 6개 시나리오 전부 gate_prob<threshold라
원본은 대부분 빈 칸 — 실제로 한 번이라도 선택된 HI(전형: ~21개)만 남겨서 "학습된
서브셋이 시나리오마다 실제로 달라짐"이라는 메시지를 바로 보이게 한다.

그리기 로직(_draw_panel 등)은 9_eval/plot_hi_selection_matrix.py를 그대로 재사용 —
행 필터링 한 단계만 추가.

2026-10-22: panel (b) 추가 — 시나리오별 선택된 raw HI를 카테고리(stat/diff/lfp/morph)
로 쌓은 bar chart. panel (a)의 raw HI 단위 매트릭스는 "어떤 HI가 선택됐는지"를 보여주지만
"카테고리 단위로 보면 시나리오마다 무엇에 의존하는지가 다른가"는 바로 안 보인다 — 같은
threshold(0.9) 선택 결과를 카테고리로 집계해 이를 직접 보여준다. NCM(TJU) 쪽도 같은 방식으로
확인했으나, 그 run(legacy_results의 사전-리팩토링 run)은 threshold=0.9에서 ever-selected가
단 1개뿐이라(LFP의 21개와 규모가 안 맞음, run family가 다름 — Fig5/6 TODO와 같은 문제) 의미
있는 카테고리 비교가 안 돼 LFP 단독으로만 그린다.

데이터 소스: <run-dir>/gates/regression_HIs.json.
실행: python docs/figures/fig5_hi_selection_trimmed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "9_eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_hi_selection_matrix import (  # noqa: E402
    _load_matrix, _strip_seg_suffix, _category_of, _sort_rows, _draw_panel, _CATEGORY_COLORS,
)

RUN_DIR = PROJECT_ROOT / "model_lib/results/p1v2_runs/1009_1342_p1v2_baseline_seed42"
THRESHOLD = 0.9
_CATEGORY_ORDER = ["stat", "diff", "lfp", "morph"]
_KERNEL_COLOR = "#555555"


def _category_composition(full_mat, full_cats, seg_names, threshold: float):
    """(n_scen, n_cat) 선택 개수 matrix — panel (a)의 trim과 무관하게 전체 raw HI
    matrix에서 바로 집계(trim에서 빠진 행은 어차피 전 시나리오 gate_prob<threshold라
    어느 열에도 기여하지 않으므로 결과는 trim 전/후 동일)."""
    counts = np.zeros((len(seg_names), len(_CATEGORY_ORDER)), dtype=int)
    for s in range(len(seg_names)):
        selected = full_mat[:, s] >= threshold
        for c, keep in zip(full_cats, selected):
            if keep:
                counts[s, _CATEGORY_ORDER.index(c)] += 1
    return counts


def _kernel_counts(kernel_path: Path, seg_names, threshold: float):
    """시나리오별 (선택된 커널 HI 수, 그 시나리오의 커널 후보 수 K_s).

    커널 게이트는 own-scenario 제한이라 시나리오마다 후보 집합 자체가 다르다
    (9_eval/plot_hi_selection_matrix.py docstring) — 시나리오x커널 매트릭스는
    무의미하지만 시나리오별 "선택 개수"는 그대로 비교 가능하다."""
    d = json.loads(kernel_path.read_text(encoding="utf-8"))
    by_name = {}
    s = 0
    while f"seg_{s}_names" in d:
        probs = np.asarray(d[f"seg_{s}_probs"], dtype=float)
        by_name[d[f"seg_{s}_seg_name"]] = (int((probs >= threshold).sum()), len(probs))
        s += 1
    return (np.array([by_name[n][0] for n in seg_names]),
            np.array([by_name[n][1] for n in seg_names]))


def _draw_category_bars(ax, counts, seg_names, threshold: float, kernel_sel=None, kernel_tot=None):
    bottoms = np.zeros(len(seg_names))
    for ci, cat in enumerate(_CATEGORY_ORDER):
        vals = counts[:, ci]
        ax.bar(range(len(seg_names)), vals, bottom=bottoms, color=_CATEGORY_COLORS[cat],
               edgecolor="white", linewidth=0.6, label=cat)
        for s, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(s, b + v / 2, str(int(v)), ha="center", va="center",
                        fontsize=7, color="white")
        bottoms += vals
    if kernel_sel is not None:
        ax.bar(range(len(seg_names)), kernel_sel, bottom=bottoms, color=_KERNEL_COLOR,
               edgecolor="white", linewidth=0.6, label="kernel")
        for s, (v, b, k) in enumerate(zip(kernel_sel, bottoms, kernel_tot)):
            if v > 0:
                ax.text(s, b + v / 2, f"{int(v)}/{int(k)}", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
        bottoms = bottoms + kernel_sel
    totals = bottoms
    for s, t in enumerate(totals):
        ax.text(s, t + 0.3, str(int(t)), ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(seg_names)))
    ax.set_xticklabels(seg_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"# HI selected (gate prob ≥ {threshold})")
    ax.set_title("Selected-HI composition per scenario\n(raw HI by category + kernel HI)",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0, totals.max() * 1.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(PROJECT_ROOT / "docs/figures"))
    from _style import setup_rcparams  # noqa: E402
    setup_rcparams()

    raw_path = RUN_DIR / "gates" / "regression_HIs.json"
    full_mat, raw_names, seg_names = _load_matrix(raw_path)
    full_base_names = [_strip_seg_suffix(n, seg_names[0]) for n in raw_names]
    full_cats = [_category_of(n) for n in full_base_names]

    # 어느 시나리오에서도 선택 안 된(gate_prob<threshold 전부) 행은 제거 (panel a만).
    ever_selected = (full_mat >= THRESHOLD).any(axis=1)
    n_total = len(full_base_names)
    mat = full_mat[ever_selected]
    base_names = [n for n, keep in zip(full_base_names, ever_selected) if keep]
    cats = [c for c, keep in zip(full_cats, ever_selected) if keep]
    n_kept = len(base_names)
    print(f"[trim] {n_total}개 중 {n_kept}개만 유지(어느 시나리오에서도 선택 안 된 "
          f"{n_total - n_kept}개 제거)")

    mat, base_names, cats = _sort_rows(mat, base_names, cats, "prob")
    counts = _category_composition(full_mat, full_cats, seg_names, THRESHOLD)
    kernel_sel, kernel_tot = _kernel_counts(RUN_DIR / "gates" / "regression_kernel_HIs.json",
                                            seg_names, THRESHOLD)
    print("[kernel] selected/candidates per scenario:",
          {n: f"{a}/{b}" for n, a, b in zip(seg_names, kernel_sel, kernel_tot)})

    fig_h = max(3.0, n_kept * 0.22)
    fig = plt.figure(figsize=(10.5, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    im = _draw_panel(ax_a, mat, base_names, cats, seg_names, THRESHOLD,
                      f"(a) Raw HI, ever selected only (N={n_kept}/{n_total})")
    _draw_category_bars(ax_b, counts, seg_names, THRESHOLD, kernel_sel, kernel_tot)
    ax_b.set_title(f"(b) {ax_b.get_title()}", fontsize=10, fontweight="bold")

    cat_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                               markersize=8, label=cat)
                   for cat, c in _CATEGORY_COLORS.items() if cat in set(cats)]
    cat_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=_KERNEL_COLOR,
                                     edgecolor="white", label="kernel (panel b)"))
    marker_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                                markersize=6, label=f"selected (gate_prob≥{THRESHOLD})")
    fig.legend(handles=cat_handles + [marker_handle], loc="upper center",
               ncol=min(len(cat_handles) + 1, 7), fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.colorbar(im, ax=ax_a, fraction=0.04, pad=0.02, label="gate_prob")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = Path(__file__).resolve().parent / "fig5_hi_selection_trimmed.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"saved: {out_png}")
