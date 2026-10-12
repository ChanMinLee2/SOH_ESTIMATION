"""Figure 5 — hi_selection_matrix.png에서 "어느 시나리오에서도 선택 안 된" raw HI
행을 잘라낸 버전(2026-10-12). 64행 중 43행이 6개 시나리오 전부 gate_prob<threshold라
원본은 대부분 빈 칸 — 실제로 한 번이라도 선택된 HI(전형: ~21개)만 남겨서 "학습된
서브셋이 시나리오마다 실제로 달라짐"이라는 메시지를 바로 보이게 한다.

그리기 로직(_draw_panel 등)은 9_eval/plot_hi_selection_matrix.py를 그대로 재사용 —
행 필터링 한 단계만 추가.

데이터 소스: <run-dir>/gates/regression_HIs.json.
실행: python docs/figures/fig5_hi_selection_trimmed.py
"""

from __future__ import annotations

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


def build_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(PROJECT_ROOT / "docs/figures"))
    from _style import setup_rcparams  # noqa: E402
    setup_rcparams()

    raw_path = RUN_DIR / "gates" / "regression_HIs.json"
    mat, raw_names, seg_names = _load_matrix(raw_path)
    base_names = [_strip_seg_suffix(n, seg_names[0]) for n in raw_names]
    cats = [_category_of(n) for n in base_names]

    # 어느 시나리오에서도 선택 안 된(gate_prob<threshold 전부) 행은 제거.
    ever_selected = (mat >= THRESHOLD).any(axis=1)
    n_total = len(base_names)
    mat = mat[ever_selected]
    base_names = [n for n, keep in zip(base_names, ever_selected) if keep]
    cats = [c for c, keep in zip(cats, ever_selected) if keep]
    n_kept = len(base_names)
    print(f"[trim] {n_total}개 중 {n_kept}개만 유지(어느 시나리오에서도 선택 안 된 "
          f"{n_total - n_kept}개 제거)")

    mat, base_names, cats = _sort_rows(mat, base_names, cats, "prob")

    fig_h = max(3.0, n_kept * 0.22)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, fig_h))
    im = _draw_panel(ax, mat, base_names, cats, seg_names, THRESHOLD,
                      f"Raw HI, ever selected only (N={n_kept}/{n_total})")

    cat_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                               markersize=8, label=cat)
                   for cat, c in _CATEGORY_COLORS.items() if cat in set(cats)]
    marker_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                                markersize=6, label=f"selected (gate_prob≥{THRESHOLD})")
    fig.legend(handles=cat_handles + [marker_handle], loc="upper center",
               ncol=min(len(cat_handles) + 1, 6), fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="gate_prob")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = Path(__file__).resolve().parent / "fig5_hi_selection_trimmed.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"saved: {out_png}")
