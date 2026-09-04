"""
5_model/experiments/phase1_lab/plot_category_survival.py

v0/v2/v3/v4 3-seed 학습 결과를 놓고, 4개 HI 카테고리(STAT/DIFF/LFP/MORPH)별로 "실제로
gate_prob>0.9로 살아남은 비율"을 비교하는 막대그래프를 그린다. phase1_trainer_v2.py와
분리된 독립 스크립트 — 저장된 gates/*.json과 kernel pkl만 읽고 학습은 안 한다.

raw뿐 아니라 커널까지 반영: 커널 피처 하나가 활성(gate_prob>0.9)이면, 그 값 1.0을
재료가 된 raw HI(member_names, kernel_group_features_*.pkl)들의 카테고리에 1/len(members)씩
가중 분배한다 — "이 커널 1개는 몇 개 카테고리의 정보를 얼마씩 담고 있는가"를 반영하는
방식. raw 활성 HI는 그대로 자기 카테고리에 1.0을 준다. 그래서 raw로도 kernel 경유로도
카테고리당 최종 점수는 "그 카테고리가 최종 예측에 기여한 정도"의 근사치가 된다(정확한
그래디언트 기반 기여도는 아님 — 게이트 on/off + 커널 구성 비율 기반의 근사).

v0는 커널이 없어 raw만 집계된다(다른 variant와 직접 비교 시 이 점 감안).

사용 예:
  SOH_EXCLUDE_STAT_LEAK=1 python 5_model/experiments/phase1_lab/plot_category_survival.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from utils.hi_schema import get_hi_cols_for_seg  # noqa: E402

RESULTS_DIR = _HERE / "results"
RUNS_DIR = RESULTS_DIR / "p1v2_runs"
CATEGORIES = ["stat", "diff", "lfp", "morph"]
GATE_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# 3-seed run dir(auto-lambda) + variant별 커널 pkl(없으면 None) — 이 세션에서 실제로
# 완료된 run들 그대로(docs/260827_RESULTS.md 3-a 표와 동일 출처).
# ---------------------------------------------------------------------------
VARIANTS: dict[str, dict] = {
    "v0": {
        "runs": {
            "42": "0821_0917_p1v2_k25_full_N2_stage12_grouped_seed42/0821_0917_p1v2_k25_full_N2_stage12_grouped_seed42",
            "0":  "0828_0239_p1v2_p1v0_full_seed0_seed0",
            "123": "0828_0506_p1v2_p1v0_full_seed123_seed123",
        },
        "kernel_pkl": None,
    },
    "v2": {
        "runs": {
            "42": "0821_1120_p1v2_k25_full_N2_stage12_kernel_v2_seed42",
            "0":  "0828_0738_p1v2_p1v2_full_seed0_seed0",
            "123": "0828_1047_p1v2_p1v2_full_seed123_seed123",
        },
        "kernel_pkl": "kernel_group_features_k25_full_N2_kernel_v2.pkl",
    },
    "v3": {
        "runs": {
            "42": "0827_1636_p1v2_p1v3_full_seed42",
            "0":  "0828_1350_p1v2_p1v3_full_seed0_seed0",
            "123": "0828_1655_p1v2_p1v3_full_seed123_seed123",
        },
        "kernel_pkl": "kernel_group_features_k25_full_N2_kernel_v3.pkl",
    },
    "v4": {
        "runs": {
            "42": "0827_1705_p1v2_p1v4_full_seed42",
            "0":  "0828_1957_p1v2_p1v4_full_seed0_seed0",
            "123": "0828_2330_p1v2_p1v4_full_seed123_seed123",
        },
        "kernel_pkl": "kernel_group_features_k25_full_N2_kernel_v3.pkl",  # v4는 v3 커널 재사용
    },
}


def _category_of(concept: str) -> str | None:
    for cat in CATEGORIES:
        if concept.startswith(cat + "_"):
            return cat
    return None


def _find_run_dir(rel: str) -> Path:
    p = RUNS_DIR / rel
    if not (p / "gates" / "regression_HIs.json").exists():
        raise FileNotFoundError(f"gates/regression_HIs.json 없음: {p}")
    return p


def _category_totals(ref_seg: str) -> dict[str, int]:
    cols = get_hi_cols_for_seg(ref_seg)
    suffix = f"_{ref_seg}"
    concepts = [c[: -len(suffix)] if c.endswith(suffix) else c for c in cols]
    totals: dict[str, int] = defaultdict(int)
    for c in concepts:
        cat = _category_of(c)
        if cat:
            totals[cat] += 1
    return totals


def _kernel_member_categories(kernel_pkl_path: Path) -> dict[tuple[str, str], list[str]]:
    """(scenario, kernel_feature_name) -> [member concept(카테고리 판별용, suffix 제거)]"""
    import pickle
    artifact = pickle.load(open(kernel_pkl_path, "rb"))
    out: dict[tuple[str, str], list[str]] = {}
    for f in artifact["features"]:
        scen = f["scenario"]
        suffix = f"_{scen}"
        concepts = [n[: -len(suffix)] if n.endswith(suffix) else n for n in f["member_names"]]
        out[(scen, f["name"])] = concepts
    return out


def _score_one_run_by_scenario(run_dir: Path, kernel_pkl_path: Path | None) -> tuple[list[str], list[dict[str, float]]]:
    """이 run(1개 seed) 안에서 시나리오별 카테고리 점수를 반환(평균 내지 않음).
    반환: (seg_names, [{cat: score} per scenario])."""
    reg = json.loads((run_dir / "gates" / "regression_HIs.json").read_text(encoding="utf-8"))
    n_scen = 0
    while f"seg_{n_scen}_names" in reg:
        n_scen += 1

    kernel_reg = None
    member_cats = None
    if kernel_pkl_path is not None:
        kreg_path = run_dir / "gates" / "regression_kernel_HIs.json"
        if kreg_path.exists():
            kernel_reg = json.loads(kreg_path.read_text(encoding="utf-8"))
            member_cats = _kernel_member_categories(kernel_pkl_path)

    per_scen_scores: list[dict[str, float]] = []
    for s in range(n_scen):
        scores: dict[str, float] = defaultdict(float)
        seg_name = reg.get(f"seg_{s}_seg_name", f"seg_{s}")

        # raw HI: 활성(prob>0.9)이면 자기 카테고리에 1.0
        for name, prob in zip(reg[f"seg_{s}_names"], reg[f"seg_{s}_probs"]):
            if prob <= GATE_THRESHOLD:
                continue
            suffix = f"_{seg_name}"
            concept = name[: -len(suffix)] if name.endswith(suffix) else name
            cat = _category_of(concept)
            if cat:
                scores[cat] += 1.0

        # kernel: 활성이면 재료 raw HI들의 카테고리에 1/len(members)씩
        if kernel_reg is not None and f"seg_{s}_names" in kernel_reg:
            for kname, kprob in zip(kernel_reg[f"seg_{s}_names"], kernel_reg[f"seg_{s}_probs"]):
                if kprob <= GATE_THRESHOLD:
                    continue
                member_concepts = member_cats.get((seg_name, kname))
                if not member_concepts:
                    continue
                w = 1.0 / len(member_concepts)
                for concept in member_concepts:
                    cat = _category_of(concept)
                    if cat:
                        scores[cat] += w

        per_scen_scores.append({cat: scores.get(cat, 0.0) for cat in CATEGORIES})

    seg_names = [reg.get(f"seg_{s}_seg_name", f"seg_{s}") for s in range(n_scen)]
    return seg_names, per_scen_scores


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot] matplotlib 미설치 - 종료")
        return
    for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = _font
            break
    plt.rcParams["axes.unicode_minus"] = False

    totals = _category_totals("chg_lo")
    print(f"[plot] 카테고리별 전체 개수: {dict(totals)}")

    # variant -> seg_name -> [per-seed {cat: score}]
    variant_scen_seed: dict[str, dict[str, list[dict[str, float]]]] = {}
    seg_names_ref: list[str] | None = None
    for vname, spec in VARIANTS.items():
        kernel_pkl_path = (RESULTS_DIR / spec["kernel_pkl"]) if spec["kernel_pkl"] else None
        if kernel_pkl_path is not None and not kernel_pkl_path.exists():
            fallback = RESULTS_DIR / "outputs" / spec["kernel_pkl"]
            kernel_pkl_path = fallback if fallback.exists() else kernel_pkl_path

        by_scen: dict[str, list[dict[str, float]]] = defaultdict(list)
        for seed, rel in spec["runs"].items():
            run_dir = _find_run_dir(rel)
            seg_names, per_scen = _score_one_run_by_scenario(run_dir, kernel_pkl_path)
            seg_names_ref = seg_names
            for sname, sc in zip(seg_names, per_scen):
                by_scen[sname].append(sc)
        variant_scen_seed[vname] = dict(by_scen)

    variants = list(VARIANTS.keys())

    # 시나리오 평균(기존 플랏용) — 시드 평균 후 시나리오 평균
    variant_scores: dict[str, dict[str, float]] = {}
    variant_std: dict[str, dict[str, float]] = {}
    for v in variants:
        per_scen_seed_mean = {
            sname: {cat: float(np.mean([sc[cat] for sc in seeds])) for cat in CATEGORIES}
            for sname, seeds in variant_scen_seed[v].items()
        }
        variant_scores[v] = {
            cat: float(np.mean([per_scen_seed_mean[s][cat] for s in seg_names_ref])) for cat in CATEGORIES
        }
        # 시나리오 간 변동까지 반영한 std(참고용 — 기존 3-seed std와는 다른 축임에 주의)
        variant_std[v] = {
            cat: float(np.std([per_scen_seed_mean[s][cat] for s in seg_names_ref])) for cat in CATEGORIES
        }
        print(f"[plot] {v}(시나리오 평균): {variant_scores[v]}")

    # ---- 플랏 A: 카테고리별 그룹 막대(값=카테고리 내 생존 비율 %) ----
    fig, (ax_pct, ax_raw) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(CATEGORIES))
    width = 0.8 / len(variants)
    colors = plt.get_cmap("tab10")

    for i, v in enumerate(variants):
        pcts = [100 * variant_scores[v][cat] / totals[cat] for cat in CATEGORIES]
        errs = [100 * variant_std[v][cat] / totals[cat] for cat in CATEGORIES]
        xpos = x + (i - (len(variants) - 1) / 2) * width
        ax_pct.bar(xpos, pcts, width=width, yerr=errs, capsize=3, label=v, color=colors(i))

    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels([f"{c.upper()}\n(전체 {totals[c]}개)" for c in CATEGORIES])
    ax_pct.set_ylabel("카테고리 내 생존 비율(%) — gate_prob>0.9, 커널 경유 가중 포함")
    ax_pct.set_title("카테고리별 생존 비율 (3-seed 평균 ± std)")
    ax_pct.legend(fontsize=9)
    ax_pct.grid(axis="y", alpha=0.3)

    for i, v in enumerate(variants):
        vals = [variant_scores[v][cat] for cat in CATEGORIES]
        errs = [variant_std[v][cat] for cat in CATEGORIES]
        xpos = x + (i - (len(variants) - 1) / 2) * width
        ax_raw.bar(xpos, vals, width=width, yerr=errs, capsize=3, label=v, color=colors(i))

    ax_raw.set_xticks(x)
    ax_raw.set_xticklabels([c.upper() for c in CATEGORIES])
    ax_raw.set_ylabel("가중 점수(raw 1.0/HI + 커널 1/n_members 분배)")
    ax_raw.set_title("카테고리별 절대 점수 (3-seed 평균 ± std)")
    ax_raw.legend(fontsize=9)
    ax_raw.grid(axis="y", alpha=0.3)

    fig.suptitle("v0~v4 — 카테고리(STAT/DIFF/LFP/MORPH)별 생존 비교 (raw+커널 가중 포함, gate_prob>0.9)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path = RESULTS_DIR / "category_survival_v0_v2_v3_v4.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")

    # ---- 플랏 B: 시나리오별(6개) 그룹 막대 — 값=절대 개수(가중 점수), % 정규화 없음 ----
    fig2, axes = plt.subplots(2, 3, figsize=(19, 10), sharey=True)
    axes = axes.flatten()
    for ax, sname in zip(axes, seg_names_ref):
        for i, v in enumerate(variants):
            seeds = variant_scen_seed[v][sname]
            vals = [float(np.mean([sc[cat] for sc in seeds])) for cat in CATEGORIES]
            errs = [float(np.std([sc[cat] for sc in seeds])) for cat in CATEGORIES]
            xpos = x + (i - (len(variants) - 1) / 2) * width
            ax.bar(xpos, vals, width=width, yerr=errs, capsize=2, label=v, color=colors(i))
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in CATEGORIES], fontsize=8)
        ax.set_title(sname, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("가중 점수(개수) — raw 1.0/HI + 커널 1/n_members")
    axes[3].set_ylabel("가중 점수(개수) — raw 1.0/HI + 커널 1/n_members")
    axes[0].legend(fontsize=8)

    fig2.suptitle("v0~v4 — 시나리오별 카테고리 생존 개수 (raw+커널 가중, gate_prob>0.9, 3-seed 평균 ± std)",
                  fontsize=12, fontweight="bold")
    fig2.tight_layout()
    out_path2 = RESULTS_DIR / "category_survival_by_scenario_v0_v2_v3_v4.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"[plot] 저장: {out_path2}")

    by_scenario_summary = {
        v: {
            sname: {cat: float(np.mean([sc[cat] for sc in seeds])) for cat in CATEGORIES}
            for sname, seeds in variant_scen_seed[v].items()
        }
        for v in variants
    }

    summary = {
        "totals": dict(totals),
        "by_scenario": by_scenario_summary,
        "variant_scores": variant_scores,
        "variant_std": variant_std,
    }
    (RESULTS_DIR / "category_survival_v0_v2_v3_v4.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
