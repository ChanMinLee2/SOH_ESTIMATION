"""
4_hi_analysis/tools/redundancy_gate_resolution.py

Fig3(a, docs/figures/fig3_hi_design_rationale.py)가 |Pearson r| >= 0.95로 표시한
"극단적으로 중복된" raw-HI 쌍들이, 실제 학습된 게이트(canonical v4 run,
Fig7/fig7_subset_selection.py와 동일 소스)에서 얼마나 "그대로 남아있는지"(둘 다
gate_prob >= threshold로 살아남아 여전히 같이 쓰이는지) 확인한다.

Fig3(a)는 dis_lo 세그먼트 상관행렬 하나로 쌍을 뽑는다(그 축 자체가 그렇게 설계됨) --
이 스크립트는 그 "같은 쌍 목록"을 그대로 갖고, 6개 시나리오 각각의 게이트에서 두
멤버가 전부 선택됐는지를 센다. "남아있음" = 둘 다 gate_prob>=threshold(기본 0.9,
plot_hi_selection_matrix.py와 동일 관례) = 게이트가 이 중복을 해소하지 못함.

Run: python 4_hi_analysis/tools/redundancy_gate_resolution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "docs" / "figures"))

from _style import INK, SUBINK, setup_rcparams  # noqa: E402
from fig3_hi_design_rationale import (  # noqa: E402
    base_names_and_categories, load_hi_frame, CAT_COLOR,
)
from fig7_subset_selection import load_raw_matrix, SCEN_NAMES  # noqa: E402

setup_rcparams()

EXTREME_CORR = 0.95     # Fig3(a)와 동일
GATE_THRESHOLD = 0.9    # 프로젝트 관례(plot_hi_selection_matrix.py 기본값)
REF_SCEN = "dis_lo"     # Fig3(a)가 쌍을 뽑는 데 쓴 세그먼트


def find_redundant_pairs(df, order, cats):
    """Fig3(a)와 동일한 방식으로 |r|>=EXTREME_CORR 쌍(i,j) 목록을 뽑는다."""
    cols = [f"{n}_{REF_SCEN}" for n in order]
    sub = df[cols].dropna()
    keep = [c for c in cols if sub[c].std() > 1e-12]
    sub = sub[keep]
    order_r = [n for n, c in zip(order, cols) if c in keep]
    corr = sub.corr().values
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    n = len(order_r)
    ys, xs = np.where((np.abs(corr) >= EXTREME_CORR) & ~np.eye(n, dtype=bool))
    pairs = sorted({tuple(sorted((order_r[i], order_r[j]))) for i, j in zip(ys, xs)})
    return pairs


def build_figure():
    order, cats = base_names_and_categories()
    df = load_hi_frame(order)
    pairs = find_redundant_pairs(df, order, cats)
    print(f"[redundancy_gate_resolution] Fig3(a) 기준 |r|>={EXTREME_CORR} 쌍: {len(pairs)}개 "
          f"(세그먼트={REF_SCEN})")
    for a, b in pairs:
        print(f"  {a}  <->  {b}")

    mat, gate_order, _, _ = load_raw_matrix()
    name_to_row = {n: i for i, n in enumerate(gate_order)}

    n_pairs = len(pairs)
    remain_frac = np.zeros(len(SCEN_NAMES))
    remain_count = np.zeros(len(SCEN_NAMES), dtype=int)
    for s, sname in enumerate(SCEN_NAMES):
        both = 0
        for a, b in pairs:
            if mat[name_to_row[a], s] >= GATE_THRESHOLD and mat[name_to_row[b], s] >= GATE_THRESHOLD:
                both += 1
        remain_count[s] = both
        remain_frac[s] = both / n_pairs if n_pairs else np.nan

    overall_both = sum(
        1 for a, b in pairs
        if all(mat[name_to_row[a], s] >= GATE_THRESHOLD and mat[name_to_row[b], s] >= GATE_THRESHOLD
               for s in range(len(SCEN_NAMES)))
    )
    print(f"[redundancy_gate_resolution] 시나리오별 '둘 다 남음' 개수: "
          f"{dict(zip(SCEN_NAMES, remain_count.tolist()))}")
    print(f"[redundancy_gate_resolution] 6개 시나리오 전부에서 둘 다 남은 쌍: "
          f"{overall_both}/{n_pairs} ({overall_both / n_pairs * 100:.1f}%)")

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.13, right=0.97)
    colors = [CAT_COLOR.get(sn.split("_")[0], SUBINK) for sn in SCEN_NAMES]
    bars = ax.bar(range(len(SCEN_NAMES)), remain_frac * 100, color=INK, alpha=0.82, width=0.62)
    for i, (b, cnt) in enumerate(zip(bars, remain_count)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{cnt}/{n_pairs}",
                 ha="center", fontsize=7.4, color=SUBINK)
    ax.axhline(100, color=SUBINK, lw=0.6, ls=(0, (3, 2)), alpha=0.5)
    ax.set_xticks(range(len(SCEN_NAMES)))
    ax.set_xticklabels(SCEN_NAMES, fontsize=8.4)
    ax.set_ylabel("Redundant pairs still both selected (%)", fontsize=8.4)
    ax.set_ylim(0, max(remain_frac.max() * 100 + 15, 20))
    ax.set_title(f"Does gate selection resolve Fig3(a)'s |r|≥{EXTREME_CORR} redundant pairs?\n"
                  f"(n={n_pairs} pairs, gate_prob≥{GATE_THRESHOLD} = \"selected\")",
                  loc="left", fontsize=8.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(direction="out", length=2.8)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = str(PROJECT_ROOT / "4_hi_analysis" / "outputs" / "redundancy_gate_resolution.png")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"saved: {out_png}")
