"""
5_model/experiments/phase1_lab/plot_hi_selection_matrix.py

64개 raw HI + N개(그룹 결과에 따라 다름, 보통 ~59개) 커널 HI가 6개 시나리오
(chg_lo/chg_mid/chg_hi/dis_hi/dis_mid/dis_lo)별로 각각 얼마나/어떻게 선택되는지
한눈에 비교하는 "HI × 시나리오" 게이트 확률 히트맵.

기존 gate_probs.png(train_scr.py._plot_gate_probs)는 시나리오별 "정렬된 순위" bar
chart라 x축이 HI 순위일 뿐 HI 정체성이 아니고, 커널 게이트(model.scen_kernel_gates)는
아예 안 그린다 — 이 스크립트는 그 공백을 메운다: raw/kernel 각각 HI 정체성을 행으로
고정하고 6개 시나리오를 열로 둬서, 어떤 HI가 어느 시나리오에서(만) 선택되는지를
행 단위로 바로 비교할 수 있게 만든다.

입력은 학습이 이미 저장해 둔 산출물만 읽는다(재학습/체크포인트 로드 없음):
  <run-dir>/gates/regression_HIs.json        (raw HI, phase1_trainer_v2.py/train_scr.py 저장)
  <run-dir>/gates/regression_kernel_HIs.json (kernel HI, v2/v3/v4 커널 피처 run에만 존재)

두 JSON 다 시나리오별 "gate_prob 랭킹 전체"(seg_s_ranked/names/probs/seg_name)를
담고 있을 뿐 임계값으로 걸러진 "선택됨" 표시는 없다 — 이 스크립트가 --threshold
(기본 0.9, 프로젝트 전체 관례인 GATE_THRESHOLD/--min-active-prob와 동일값)로
직접 이진화해서 히트맵 위에 점(●) 마커로 표시한다.

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
    "kernel": "#555555",
}
_RAW_CATEGORIES = ("stat", "diff", "lfp", "morph")  # 커널 구성비 스택바 고정 순서


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="raw HI + 커널 HI의 시나리오별 gate_prob 선택 매트릭스 히트맵"
    )
    p.add_argument("--run-dir", required=True, dest="run_dir",
                    help="results/p1v2_runs/<run> 디렉터리(gates/regression_HIs.json이 있는 곳)")
    p.add_argument("--threshold", type=float, default=0.9,
                    help="이 값 이상이면 '선택됨' 마커(●) 표시 (기본 0.9, "
                         "plot_kernel_gate_inactive.py --min-active-prob와 동일 관례)")
    p.add_argument("--sort", choices=["prob", "category", "index"], default="prob",
                    help="행(HI) 정렬 기준: prob=시나리오 평균 gate_prob 내림차순(기본), "
                         "category=raw는 stat/diff/lfp/morph 묶음 내 prob순, kernel은 이름순, "
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
    if base_name.startswith("kernel_"):
        return "kernel"
    return base_name.split("_", 1)[0] if "_" in base_name else base_name


def _resolve_kernel_pkl_path(run_dir: Path) -> Path | None:
    """<run-dir>/p1v2_summary.json의 kernel_features_pkl 경로를 읽어 PROJECT_ROOT
    기준으로 고정한다(학습 당시 cwd 기준 상대경로가 그대로 남아있을 수 있음 —
    test_phase1_checkpoint.py의 _resolve_summary_path와 동일 원칙)."""
    summary_path = run_dir / "p1v2_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    v = summary.get("kernel_features_pkl")
    if not v:
        return None
    p = Path(v)
    resolved = p if p.is_absolute() else PROJECT_ROOT / p
    return resolved if resolved.exists() else None


def _load_kernel_composition(pkl_path: Path) -> dict[str, list[str]]:
    """kernel_group_features_*.pkl → {커널명: [멤버 raw HI 카테고리, ...]}.

    build_kernel_group_features.py가 저장하는 스키마(features[i] = {"name",
    "member_names", "model"(sklearn Pipeline), ...})에서 이름 두 필드만 쓴다 —
    model(Pipeline)까지 통째로 언피클되긴 하지만(가벼운 진단 스크립트라 감수),
    sklearn 버전 경고는 메타데이터 추출과 무관해 조용히 무시한다."""
    import pickle
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(pkl_path, "rb") as f:
            artifact = pickle.load(f)
    return {
        feat["name"]: [_category_of(m) for m in feat["member_names"]]
        for feat in artifact["features"]
    }


def _draw_composition_panel(ax, ker_names: list[str], composition: dict[str, list[str]]):
    """커널 HI 행마다 멤버 raw HI의 카테고리 구성비를 100% 가로 스택바로 그린다
    (raw HI 패널과 같은 카테고리 색 재사용) — 이 커널이 "뭘로 만들어졌는지"를
    선택 히트맵과 나란히 보여줘 카테고리 지배도를 한눈에 파악하게 한다."""
    n = len(ker_names)
    for i, name in enumerate(ker_names):
        cats = composition.get(name)
        if not cats:
            ax.barh(i, 1.0, color="#dddddd", edgecolor="none")
            continue
        total = len(cats)
        left = 0.0
        for cat in _RAW_CATEGORIES:
            frac = cats.count(cat) / total
            if frac <= 0:
                continue
            ax.barh(i, frac, left=left, color=_CATEGORY_COLORS[cat],
                     edgecolor="white", linewidth=0.3, height=0.8)
            left += frac

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0%", "50%", "100%"], fontsize=7)
    ax.set_yticks([])
    ax.set_title("구성비", fontsize=9)


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

    have_kernel = kernel_path.exists()
    composition: dict[str, list[str]] = {}
    if have_kernel:
        ker_mat, ker_names, ker_seg_names = _load_matrix(kernel_path)
        ker_cat = [_category_of(n) for n in ker_names]
        ker_mat, ker_names, ker_cat = _sort_rows(ker_mat, ker_names, ker_cat, args.sort)

        kernel_pkl = _resolve_kernel_pkl_path(run_dir)
        if kernel_pkl:
            composition = _load_kernel_composition(kernel_pkl)
            print(f"[plot] 커널 구성비 로드: {kernel_pkl}")
        else:
            print("[plot] kernel_features_pkl을 못 찾음 — 구성비 패널 생략")
    else:
        print(f"[plot] {kernel_path} 없음 — 이 run은 커널 피처 미사용, raw HI만 그림")

    have_composition = have_kernel and bool(composition)
    n_panels = (3 if have_composition else 2) if have_kernel else 1
    max_rows = max(len(raw_base), len(ker_names) if have_kernel else 0)
    fig_h = max(6.0, max_rows * 0.16)
    width_ratios = [6.5, 6.5, 1.8][:n_panels]
    fig, axes = plt.subplots(1, n_panels, figsize=(sum(width_ratios), fig_h),
                              gridspec_kw={"wspace": 0.9, "width_ratios": width_ratios})
    axes = [axes] if n_panels == 1 else list(axes)

    im0 = _draw_panel(axes[0], raw_mat, raw_base, raw_cat, seg_names, args.threshold,
                       f"Raw HI (N={len(raw_base)})")
    if have_kernel:
        im1 = _draw_panel(axes[1], ker_mat, ker_names, ker_cat, ker_seg_names,
                           args.threshold, f"Kernel HI (N={len(ker_names)})")
        if have_composition:
            axes[2].sharey(axes[1])
            _draw_composition_panel(axes[2], ker_names, composition)

    cat_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                               markersize=8, label=cat)
                   for cat, c in _CATEGORY_COLORS.items()
                   if cat in set(raw_cat) | (set(ker_cat) if have_kernel else set())]
    marker_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                                markersize=6, label=f"선택됨(gate_prob≥{args.threshold})")
    fig.legend(handles=cat_handles + [marker_handle], loc="upper center",
               ncol=min(len(cat_handles) + 1, 6), fontsize=8, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"HI 선택 매트릭스 — {run_dir.name}", fontsize=12, fontweight="bold", y=1.06)
    fig.colorbar(im0, ax=axes[:2] if have_kernel else axes, fraction=0.02, pad=0.02, label="gate_prob")
    out_dir = Path(args.out_dir) if args.out_dir else gates_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hi_selection_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")

    print(f"\n[요약] 시나리오별 선택 개수(gate_prob≥{args.threshold}):")
    for s, sname in enumerate(seg_names):
        n_raw_sel = int((raw_mat[:, s] >= args.threshold).sum())
        line = f"  {sname:<8} raw {n_raw_sel:>3}/{len(raw_base)}"
        if have_kernel:
            n_ker_sel = int((ker_mat[:, s] >= args.threshold).sum())
            line += f"   kernel {n_ker_sel:>3}/{len(ker_names)}"
        print(line)


if __name__ == "__main__":
    main()
