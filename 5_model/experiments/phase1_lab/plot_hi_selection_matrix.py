"""
5_model/experiments/phase1_lab/plot_hi_selection_matrix.py

raw HI(N_HI개, 63/64/66 등 실행에 따라 다름)가 시나리오(chg_lo/chg_mid/chg_hi/dis_hi/
dis_mid/dis_lo 등)별로 얼마나/어떻게 선택되는지 한눈에 비교하는 "raw HI × 시나리오"
게이트 확률 히트맵.

기존 gate_probs.png(train_scr.py._plot_gate_probs)는 시나리오별 "정렬된 순위" bar
chart라 x축이 HI 순위일 뿐 HI 정체성이 아니다 — 이 스크립트는 HI 정체성을 행으로
고정하고 시나리오를 열로 둬서, 어떤 raw HI가 어느 시나리오에서(만) 선택되는지를
행 단위로 바로 비교할 수 있게 만든다.

2026-09-21: 커널 HI 패널은 제거했다. 2026-09-18 커널 own-scenario 제한
(docs/260917_RESULTS.md) 이후 scen_kernel_gates[s]의 폭 자체가 시나리오별 실제
개수(K_s)로 좁혀져서, 그 시나리오에 속하지 않는 커널은 애초에 게이트 슬롯이 없다 —
즉 "시나리오 × 커널" 매트릭스를 그리면 커널 하나가 항상 자기 시나리오 열에만 값을
갖고 나머지는 구조적으로 무조건 0이라(선택 여부와 무관하게 원천적으로 그럴 수밖에
없음), raw HI 패널과 달리 "시나리오 간 비교"라는 이 플랏의 목적 자체가 성립하지
않는다(raw HI는 진짜 전 시나리오 공유 카탈로그라 비교가 의미 있음). 커널 HI의
중요도/시나리오별 분포는 `plot_hi_importance_ranking`이 만드는
`figures/hi_importance_ranking.png`(test_phase1_checkpoint.py에 내장, 랭킹 막대 +
시나리오 색 구분)에서 확인할 것 — 그쪽은 애초에 "시나리오 간 매트릭스 비교"가 아니라
"전체 랭킹"이라 이 구조적 제약과 무관하게 의미가 있다.

입력은 학습이 이미 저장해 둔 산출물만 읽는다(재학습/체크포인트 로드 없음):
  <run-dir>/gates/regression_HIs.json (raw HI, phase1_trainer_v2.py/train_scr.py 저장)

"gate_prob 랭킹 전체"(seg_s_ranked/names/probs/seg_name)를 담고 있을 뿐 임계값으로
걸러진 "선택됨" 표시는 없다 — 이 스크립트가 --threshold(기본 0.9, 프로젝트 전체 관례인
GATE_THRESHOLD/--min-active-prob와 동일값)로 직접 이진화해서 히트맵 위에 점(●)
마커로 표시한다.

사용 예:
  python 5_model/experiments/phase1_lab/plot_hi_selection_matrix.py \
      --run-dir 5_model/experiments/phase1_lab/results/p1v2_runs/0827_1705_p1v2_p1v4_full_seed42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_CATEGORY_COLORS = {
    "stat":  "#1f77b4",
    "diff":  "#ff7f0e",
    "lfp":   "#2ca02c",
    "morph": "#9467bd",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="raw HI의 시나리오별 gate_prob 선택 매트릭스 히트맵"
    )
    p.add_argument("--run-dir", required=True, dest="run_dir",
                    help="results/p1v2_runs/<run> 디렉터리(gates/regression_HIs.json이 있는 곳)")
    p.add_argument("--threshold", type=float, default=0.9,
                    help="이 값 이상이면 '선택됨' 마커(●) 표시 (기본 0.9, "
                         "plot_kernel_gate_inactive.py --min-active-prob와 동일 관례)")
    p.add_argument("--sort", choices=["prob", "category", "index"], default="prob",
                    help="행(HI) 정렬 기준: prob=시나리오 평균 gate_prob 내림차순(기본), "
                         "category=stat/diff/lfp/morph 묶음 내 prob순, "
                         "index=학습 시 원래 인덱스 순")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                    help="기본: <run-dir>/gates/")
    return p.parse_args()


def _load_matrix(json_path: Path):
    """gates/*.json → (mat[n_hi, n_scen], hi_names, seg_names).

    seg_s_ranked[i]는 gate 내부 인덱스, seg_s_names[i]/seg_s_probs[i]가 그 인덱스의
    이름/확률(train_scr.py._save_scen_masks_to_json과 동일 스키마) — 인덱스 기준으로
    역정렬해 "HI 정체성(행) × 시나리오(열)" 매트릭스로 재구성한다.
    """
    d = json.loads(json_path.read_text(encoding="utf-8"))
    n_scen = 0
    while f"seg_{n_scen}_names" in d:
        n_scen += 1
    seg_names = [d[f"seg_{s}_seg_name"] for s in range(n_scen)]

    idx_to_name = dict(zip(d["seg_0_ranked"], d["seg_0_names"]))
    n_hi = len(idx_to_name)

    import numpy as np
    mat = np.zeros((n_hi, n_scen), dtype=float)
    for s in range(n_scen):
        idx_to_prob = dict(zip(d[f"seg_{s}_ranked"], d[f"seg_{s}_probs"]))
        for idx in range(n_hi):
            mat[idx, s] = idx_to_prob[idx]

    hi_names = [idx_to_name[i] for i in range(n_hi)]
    return mat, hi_names, seg_names


def _strip_seg_suffix(name: str, seg_name: str) -> str:
    suffix = f"_{seg_name}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _category_of(base_name: str) -> str:
    return base_name.split("_", 1)[0] if "_" in base_name else base_name


def _sort_rows(mat, labels, categories, mode: str):
    import numpy as np
    mean_prob = mat.mean(axis=1)
    if mode == "prob":
        order = np.argsort(-mean_prob)
    elif mode == "index":
        order = np.arange(len(labels))
    else:  # category
        cat_rank = {c: i for i, c in enumerate(sorted(set(categories)))}
        order = sorted(range(len(labels)),
                        key=lambda i: (cat_rank[categories[i]], -mean_prob[i]))
        order = list(order)
    return mat[order], [labels[i] for i in order], [categories[i] for i in order]


def _draw_panel(ax, mat, labels, categories, seg_names, threshold: float, title: str):
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    n_hi = len(labels)
    ax.set_xticks(range(len(seg_names)))
    ax.set_xticklabels(seg_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_hi))
    fontsize = 7.2 if n_hi <= 70 else max(4.0, 700 / n_hi / 10)
    ax.set_yticklabels(labels, fontsize=fontsize)
    for tick, cat in zip(ax.get_yticklabels(), categories):
        tick.set_color(_CATEGORY_COLORS.get(cat, "#000000"))

    for i in range(n_hi):
        for j in range(len(seg_names)):
            if mat[i, j] >= threshold:
                color = "white" if mat[i, j] < 0.55 else "black"
                ax.text(j, i, "●", ha="center", va="center",
                         fontsize=6, color=color)

    ax.set_title(title, fontsize=10, fontweight="bold")
    return im


def main() -> None:
    args = _parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 미설치 - 종료")
        return
    for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = _font
            break
    plt.rcParams["axes.unicode_minus"] = False

    run_dir = Path(args.run_dir)
    gates_dir = run_dir / "gates"
    raw_path = gates_dir / "regression_HIs.json"
    kernel_path = gates_dir / "regression_kernel_HIs.json"
    if not raw_path.exists():
        print(f"[plot] {raw_path} 없음 — run-dir이 맞는지 확인하세요")
        return

    raw_mat, raw_names, seg_names = _load_matrix(raw_path)
    raw_base = [_strip_seg_suffix(n, seg_names[0]) for n in raw_names]
    raw_cat = [_category_of(n) for n in raw_base]
    raw_mat, raw_base, raw_cat = _sort_rows(raw_mat, raw_base, raw_cat, args.sort)

    if kernel_path.exists():
        print(f"[plot] {kernel_path} 있음 — 참고: 커널 HI는 own-scenario 제한(2026-09-18)으로 "
              "시나리오 간 매트릭스 비교가 구조적으로 무의미해 이 스크립트에서 제외했습니다. "
              "figures/hi_importance_ranking.png(test_phase1_checkpoint.py 산출)를 참고하세요.")

    fig_h = max(6.0, len(raw_base) * 0.16)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, fig_h))

    im0 = _draw_panel(ax, raw_mat, raw_base, raw_cat, seg_names, args.threshold,
                       f"Raw HI (N={len(raw_base)})")

    cat_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                               markersize=8, label=cat)
                   for cat, c in _CATEGORY_COLORS.items() if cat in set(raw_cat)]
    marker_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                                markersize=6, label=f"선택됨(gate_prob≥{args.threshold})")
    fig.legend(handles=cat_handles + [marker_handle], loc="upper center",
               ncol=min(len(cat_handles) + 1, 6), fontsize=8, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"HI 선택 매트릭스 — {run_dir.name}", fontsize=12, fontweight="bold", y=1.06)
    fig.colorbar(im0, ax=ax, fraction=0.02, pad=0.02, label="gate_prob")
    out_dir = Path(args.out_dir) if args.out_dir else gates_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hi_selection_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")

    print(f"\n[요약] 시나리오별 raw HI 선택 개수(gate_prob≥{args.threshold}):")
    for s, sname in enumerate(seg_names):
        n_raw_sel = int((raw_mat[:, s] >= args.threshold).sum())
        print(f"  {sname:<8} raw {n_raw_sel:>3}/{len(raw_base)}")


if __name__ == "__main__":
    main()
