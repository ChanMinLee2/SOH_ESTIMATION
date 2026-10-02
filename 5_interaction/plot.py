"""
5_interaction/plot.py

interaction.py(Step 5)의 "셀 단위 직접 계산" 판정 근거를 시각화한다 — 셀 각각에서
독립적으로 구한 std_r_across_scenarios 값의 분포를 HI 전체(66개)에 대해 작은
히스토그램 격자(small multiples)로 그리고, 그 분포의 평균/95% CI(하한·상한)를
세로선으로 표시한다. interaction.py가 저장하는 hi_scenario_interaction_{tag}.json
에는 집계된 평균/CI만 있고 셀별 원값은 없으므로, 이 스크립트가
_load_all_scenarios/_compute_per_cell_std_r/_cell_level_ci를 그대로 재사용해
(중복 구현 금지) 전체 HI를 다시 계산한다.

std_r_across_scenarios 내림차순으로 정렬해 배치(가장 시나리오 간 차이가 큰 HI가
왼쪽 위부터) — significant=True인 HI는 제목을 초록색/굵게, False는 회색으로 구분.

실행(interaction.py가 먼저 실행돼 hi_scenario_interaction_{tag}.json이 있어야 함):
    python 5_interaction/plot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import parameters as P
from interaction import _load_all_scenarios, _compute_per_cell_std_r, _cell_level_ci

for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _font; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
RESULTS_DIR = PROJECT_ROOT / "model_lib" / "results"

N_COLS = 6  # 격자 열 수 — HI 66개 기준 11행


def _draw_cell(ax, cell_values, mean: float, ci_lo: float, ci_hi: float,
               concept: str, std_r: float, significant: bool) -> None:
    """HI 하나의 셀별 std_r 분포 미니 히스토그램 + 평균/95% CI 세로선."""
    ax.hist(cell_values, bins=15, color="#4C72B0", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(mean, color="black", linewidth=1.0)
    ax.axvline(ci_lo, color="crimson", linestyle="--", linewidth=0.8)
    ax.axvline(ci_hi, color="crimson", linestyle="--", linewidth=0.8)
    ax.axvspan(ci_lo, ci_hi, color="crimson", alpha=0.08)
    color = "#2a9d2a" if significant else "#888888"
    ax.set_title(f"{concept}\nstd_r={std_r:.3f}", fontsize=6.5,
                 color=color, fontweight="bold" if significant else "normal")
    ax.tick_params(labelsize=5.5)
    ax.set_yticklabels([])


def _plot_ranking(ordered_concepts: list[str], concept_to_idx: dict, cell_mean: np.ndarray,
                  cell_ci_lower: np.ndarray, cell_ci_upper: np.ndarray, per_hi: dict,
                  min_effect_size: float, tag: str, n_cells_used: int, out_path: Path) -> None:
    """66개 HI의 cell_level_std_r_mean(± 95% CI)을 내림차순 가로 막대로 — 순위를
    한눈에 보기 위한 플롯(개별 분포는 _draw_cell의 격자 쪽에서 본다). 세로 점선은
    min_effect_size 문턱(기본 0.1) — 이 선보다 오른쪽이면 effect_size_meaningful."""
    n_hi = len(ordered_concepts)
    means = np.array([cell_mean[concept_to_idx[c]] for c in ordered_concepts])
    ci_lo = np.array([cell_ci_lower[concept_to_idx[c]] for c in ordered_concepts])
    ci_hi = np.array([cell_ci_upper[concept_to_idx[c]] for c in ordered_concepts])
    colors = ["#2a9d2a" if per_hi[c]["significant"] else "#888888" for c in ordered_concepts]

    fig, ax = plt.subplots(figsize=(8, max(6, n_hi * 0.22)))
    y = np.arange(n_hi)
    xerr = np.vstack([means - ci_lo, ci_hi - means])
    ax.barh(y, means, color=colors, height=0.65,
            xerr=xerr, error_kw=dict(ecolor="black", elinewidth=0.7, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels(ordered_concepts, fontsize=7)
    ax.invert_yaxis()  # 1위가 맨 위
    ax.axvline(min_effect_size, color="crimson", linestyle="--", linewidth=1.2,
               label=f"min_effect_size={min_effect_size}")
    ax.set_xlabel("cell_level_std_r_mean (± 95% CI)")
    ax.set_title(
        f"HI {n_hi}개 순위 — 셀 단위 std_r 평균(tag={tag}, 셀 {n_cells_used}개 사용)\n"
        f"초록=significant, 회색=비유의, 빨간 점선=효과크기 문턱({min_effect_size})",
        fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    tag = P.FIXED_INTERACTION_TAG or f"{P.ACTIVE_P1_TAG}_interaction"
    json_path = RESULTS_DIR / f"hi_scenario_interaction_{tag}.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    per_hi = payload["per_hi"]
    min_segs_per_cell = payload.get("min_segs_per_cell", P.FIXED_INTERACTION_MIN_SEGS_PER_CELL)
    n_hi = len(per_hi)
    print(f"[plot_interaction] HI {n_hi}개 전체를 그린다 "
          f"(significant={sum(1 for v in per_hi.values() if v['significant'])}개)")

    # _load_all_scenarios는 interaction.py 소유(중복 구현 금지) — main()과 동일하게
    # parameters.py에서 읽은 값만으로 SimpleNamespace를 구성해 넘긴다.
    split_seed = P.ACTIVE_SPLIT_SEED if P.ACTIVE_SPLIT_SEED is not None else P.FIXED_DEFAULT_SEED
    loader_args = SimpleNamespace(
        data_dir=P.FIXED_CANONICAL_DATA_DIR, seg_data_dir=P.FIXED_CANONICAL_SEG_DATA_DIR,
        datasets=P.FIXED_CANONICAL_DATASETS, split_seed=split_seed,
        axis_config=json.dumps(P.ACTIVE_AXIS_CONFIG), seg_axis=P.FIXED_SEG_AXIS,
    )
    print("[plot_interaction] 데이터 로드 중...")
    x_all, y_all, scen_idx_all, spec, names_by_seg, cell_ids = _load_all_scenarios(loader_args)
    n_scen = spec.n_scenarios

    seg0_names = names_by_seg[0]
    seg0_suffix = f"_{spec.scenario_names[0]}"
    concepts = [n[:-len(seg0_suffix)] if n.endswith(seg0_suffix) else n for n in seg0_names]
    concept_to_idx = {c: i for i, c in enumerate(concepts)}

    per_cell_std_r = _compute_per_cell_std_r(
        x_all, y_all, scen_idx_all, cell_ids, n_scen, min_segs_per_cell)
    cell_mean, cell_ci_lower, cell_ci_upper = _cell_level_ci(per_cell_std_r)

    # cell_mean(로컬에서 다시 계산한 셀 단위 std_r 평균) 내림차순 — 시나리오 간
    # 차이가 큰 HI부터 배치. JSON의 풀링 기준 std_r_across_scenarios 필드는
    # 체계적으로 깎인 값이라 2026-10-01에 제거됐으므로 여기서도 안 쓴다.
    ordered_concepts = sorted(concepts, key=lambda c: -cell_mean[concept_to_idx[c]])

    n_cols = N_COLS
    n_rows = math.ceil(n_hi / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 1.8))
    axes = axes.flatten()

    for i, concept in enumerate(ordered_concepts):
        idx = concept_to_idx[concept]
        _draw_cell(axes[i], per_cell_std_r[:, idx], cell_mean[idx],
                   cell_ci_lower[idx], cell_ci_upper[idx],
                   concept, cell_mean[idx],
                   per_hi[concept]["significant"])
    for ax in axes[n_hi:]:
        ax.axis("off")

    fig.suptitle(
        f"HI {n_hi}개 전체 — 셀 단위 std_r_across_scenarios 분포 + 95% CI\n"
        f"(tag={tag}, 셀 {per_cell_std_r.shape[0]}개 사용, 초록 제목=significant, "
        f"검은 세로선=평균, 빨간 점선=95% CI)",
        fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = RESULTS_DIR / f"hi_scenario_interaction_{tag}_cell_ci_all.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_interaction] 저장: {out_path}")

    # 순위 플롯 — cell_mean(± 95% CI) 내림차순 가로 막대, 이미 계산된 값 재사용(추가 비용 없음)
    ranking_out_path = RESULTS_DIR / f"hi_scenario_interaction_{tag}_ranking.png"
    _plot_ranking(ordered_concepts, concept_to_idx, cell_mean, cell_ci_lower, cell_ci_upper,
                  per_hi, payload["min_effect_size"], tag, per_cell_std_r.shape[0], ranking_out_path)
    print(f"[plot_interaction] 저장: {ranking_out_path}")


if __name__ == "__main__":
    main()
