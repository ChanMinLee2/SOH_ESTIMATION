"""raw HI 선택 매트릭스 대안 시각화 — 8x8 그리드 + 시나리오 6겹 투명 오버레이.

기존 hi_selection_matrix.png(9_eval/plot_hi_selection_matrix.py, "HI 행 x 시나리오 열"
히트맵)와 같은 데이터를 다르게 배치: 64개 raw HI를 8x8 고정 위치에 두고(칸 위치는
모든 시나리오가 공유), 시나리오마다 자기 색으로 "선택된 HI 칸만" 반투명 사각형을
그려 6겹을 겹친다 — 여러 시나리오가 같은 HI를 선택할수록 알파가 누적돼 그 칸이
더 진하고 탁하게 보인다("어느 HI가 시나리오를 넘나들며 공통으로 중요한가"를 64행
리스트 대신 그리드의 "진한 덩어리"로 한눈에 보려는 목적, 2026-10-12).

오버레이만으로는 "몇 겹 겹쳤는지"는 보여도 "정확히 어느 시나리오들인지"는 색이
섞여 구분이 어려워서, 같은 그리드에 선택 시나리오 개수를 숫자+컬러맵으로 보여주는
대조 패널을 오른쪽에 같이 둔다.

데이터 소스: <run-dir>/gates/regression_HIs.json (plot_hi_selection_matrix.py와 동일
스키마). 실행: python docs/figures/gate_new_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import setup_rcparams, INK, GRID  # noqa: E402

RUN_DIR = PROJECT_ROOT / "model_lib/results/p1v2_runs/1009_1342_p1v2_baseline_seed42"
THRESHOLD = 0.9  # 프로젝트 관례(plot_hi_selection_matrix.py와 동일, GATE_THRESHOLD)

_CATEGORY_COLORS = {"stat": "#1f77b4", "diff": "#ff7f0e", "lfp": "#2ca02c", "morph": "#9467bd"}

# 시나리오 6색 — 포지션(저/중/고) 3색 계열 x 방향(충/방) 2음영, _style.py의
# Z_LEFT/MID/RIGHT와 같은 색상군(채도/명도만 방향별로 다르게 함). 서로 다른 HUE
# 6개를 겹치면 겹침 자체는 muddy한 갈색 계열로 수렴하는데, 그게 의도된 신호
# ("겹칠수록 진하고 탁해짐")다 — 정확히 어떤 조합인지는 오른쪽 개수 패널로 보완.
_SCEN_COLORS = {
    "chg_lo":  "#3A6178",
    "chg_mid": "#A9863F",
    "chg_hi":  "#96473A",
    "dis_hi":  "#8FB4C9",
    "dis_mid": "#D9BE85",
    "dis_lo":  "#D18F80",
}


def load_matrix(json_path: Path):
    d = json.loads(json_path.read_text(encoding="utf-8"))
    n_scen = 0
    while f"seg_{n_scen}_names" in d:
        n_scen += 1
    seg_names = [d[f"seg_{s}_seg_name"] for s in range(n_scen)]
    idx_to_name = dict(zip(d["seg_0_ranked"], d["seg_0_names"]))
    n_hi = len(idx_to_name)
    mat = np.zeros((n_hi, n_scen), dtype=float)
    for s in range(n_scen):
        idx_to_prob = dict(zip(d[f"seg_{s}_ranked"], d[f"seg_{s}_probs"]))
        for idx in range(n_hi):
            mat[idx, s] = idx_to_prob[idx]
    hi_names = [idx_to_name[i] for i in range(n_hi)]
    return mat, hi_names, seg_names


def strip_seg_suffix(name: str, seg_name: str) -> str:
    suffix = f"_{seg_name}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def category_of(base_name: str) -> str:
    return base_name.split("_", 1)[0] if "_" in base_name else base_name


def build_figure() -> plt.Figure:
    setup_rcparams()
    mat, raw_names, seg_names = load_matrix(RUN_DIR / "gates" / "regression_HIs.json")
    base_names = [strip_seg_suffix(n, seg_names[0]) for n in raw_names]
    cats = [category_of(n) for n in base_names]
    n_hi = len(base_names)
    assert n_hi == 64, f"8x8 그리드 가정이 깨짐(N_HI={n_hi})"

    # 카테고리 블록 순서(stat->diff->lfp->morph), 블록 내부는 전 시나리오 평균
    # gate_prob 내림차순 -- hi_selection_matrix.png의 --sort category와 동일 규칙.
    mean_prob = mat.mean(axis=1)
    cat_order = {"stat": 0, "diff": 1, "lfp": 2, "morph": 3}
    order = sorted(range(n_hi), key=lambda i: (cat_order[cats[i]], -mean_prob[i]))
    mat = mat[order]
    base_names = [base_names[i] for i in order]
    cats = [cats[i] for i in order]

    # 8x8 그리드 좌표(행 우선, 왼쪽->오른쪽/위->아래) -- 모든 시나리오가 공유하는 고정 위치.
    xs = np.array([i % 8 for i in range(n_hi)])
    ys = np.array([7 - i // 8 for i in range(n_hi)])

    fig, (ax, ax_cnt) = plt.subplots(1, 2, figsize=(11.5, 5.6),
                                      gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28})

    # ---- 좌: 6겹 투명 오버레이 ----
    sel = mat >= THRESHOLD  # (n_hi, n_scen) bool
    for s, sname in enumerate(seg_names):
        m = sel[:, s]
        if not m.any():
            continue
        ax.scatter(xs[m], ys[m], s=640, marker="s", color=_SCEN_COLORS[sname],
                   alpha=0.40, linewidths=0, zorder=2 + s)

    for i in range(n_hi):
        ax.text(xs[i], ys[i] - 0.37, base_names[i], ha="center", va="top",
                 fontsize=3.6, color=_CATEGORY_COLORS[cats[i]])
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.6)
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(-0.6, 7.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(f"6-scenario overlay (selected = gate_prob ≥ {THRESHOLD})",
                 loc="left", fontsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    scen_handles = [mpatches.Patch(color=_SCEN_COLORS[s], alpha=0.75, label=s) for s in seg_names]
    ax.legend(handles=scen_handles, loc="upper center", bbox_to_anchor=(0.5, -0.03),
              ncol=6, fontsize=6.6, frameon=False, columnspacing=1.0, handlelength=1.2)

    # ---- 우: 같은 그리드, "선택한 시나리오 개수" 숫자+컬러맵 (대조용) ----
    counts = sel.sum(axis=1)
    sc = ax_cnt.scatter(xs, ys, s=640, marker="s", c=counts, cmap="YlOrBr", vmin=0, vmax=6,
                         edgecolors=INK, linewidths=0.4)
    for i in range(n_hi):
        if counts[i] > 0:
            ax_cnt.text(xs[i], ys[i], str(counts[i]), ha="center", va="center",
                        fontsize=7, color="white" if counts[i] >= 3 else INK)
    ax_cnt.set_xlim(-0.6, 7.6)
    ax_cnt.set_ylim(-0.6, 7.6)
    ax_cnt.set_xticks([])
    ax_cnt.set_yticks([])
    ax_cnt.set_aspect("equal")
    ax_cnt.set_title("Same grid, count of scenarios selecting each HI (reference)",
                      loc="left", fontsize=9.5)
    for spine in ax_cnt.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(sc, ax=ax_cnt, fraction=0.046, pad=0.03)
    cbar.set_label("# scenarios selecting this HI", fontsize=7.6)

    fig.suptitle(f"Raw HI selection — 8x8 grid overlay ({RUN_DIR.name})", fontsize=11, y=0.985)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = Path(__file__).resolve().parent / "gate_new_matrix.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"saved: {out_png}")
