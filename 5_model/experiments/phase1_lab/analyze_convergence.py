"""
5_model/experiments/phase1_lab/analyze_convergence.py

Phase1 랭킹 시드-수렴성 검증 — 2단계(분석).

run_convergence_seeds.py가 만든 manifest(seed -> run_dir)를 읽어, 각 run의
gates/regression_HIs.json(시나리오별 HI 랭킹)을 비교한다.

지표:
  - top-k Jaccard overlap (seed쌍마다, k in --k-values)
  - Kendall's tau (전체 랭킹 순위상관, seed쌍마다)
  - 선택빈도(top-k 채택 빈도) 기반 앙상블 랭킹 (stability selection)

출력: results/convergence_report_{tag}.md + .json

사용 예:
  python 5_model/experiments/phase1_lab/analyze_convergence.py \
      --manifest 5_model/experiments/phase1_lab/results/convergence_manifest_k25_convergence.json \
      --k-values 5 15 25
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

from scipy.stats import kendalltau

from log_utils import append_log_entry, current_command_str

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_gates(run_dir: str) -> dict:
    p = Path(run_dir) / "gates" / "regression_HIs.json"
    if not p.exists():
        raise FileNotFoundError(f"gates json 없음: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _n_scenarios(gates: dict) -> int:
    n = 0
    while f"seg_{n}_ranked" in gates:
        n += 1
    return n


def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def analyze(manifest_path: Path, k_values: list[int]) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = list(manifest["runs"].keys())
    if len(seeds) < 2:
        raise RuntimeError(f"seed가 2개 미만입니다({len(seeds)}개) — 최소 2개 이상의 run이 필요합니다")

    gates_by_seed = {s: _load_gates(manifest["runs"][s]) for s in seeds}
    n_scen = _n_scenarios(gates_by_seed[seeds[0]])
    seg_names = {s: gates_by_seed[seeds[0]].get(f"seg_{s}_seg_name", f"seg_{s}") for s in range(n_scen)}

    report: dict = {"tag": manifest["tag"], "seeds": seeds, "n_scenarios": n_scen,
                     "k_values": k_values, "per_scenario": {}}

    for s in range(n_scen):
        full_rankings = {seed: gates_by_seed[seed][f"seg_{s}_ranked"] for seed in seeds}

        # Kendall's tau — 모든 seed쌍
        taus = []
        for a, b in itertools.combinations(seeds, 2):
            tau, _ = kendalltau(full_rankings[a], full_rankings[b])
            taus.append(tau)
        mean_tau = sum(taus) / len(taus)

        # k별 Jaccard + 선택빈도 기반 앙상블
        k_results = {}
        for k in k_values:
            jaccards = []
            for a, b in itertools.combinations(seeds, 2):
                jaccards.append(jaccard(full_rankings[a][:k], full_rankings[b][:k]))
            mean_jac = sum(jaccards) / len(jaccards)

            # 선택빈도(stability selection): 각 HI가 top-k에 뽑힌 seed 비율
            freq = Counter()
            for seed in seeds:
                freq.update(full_rankings[seed][:k])
            ensemble_ranked = [idx for idx, _ in freq.most_common()]
            ensemble_top_k = ensemble_ranked[:k]
            selection_rate = {idx: cnt / len(seeds) for idx, cnt in freq.items()}
            avg_selection_rate_of_ensemble = sum(
                selection_rate[idx] for idx in ensemble_top_k
            ) / len(ensemble_top_k)

            k_results[k] = {
                "mean_jaccard": round(mean_jac, 4),
                "ensemble_top_k": ensemble_top_k,
                "avg_selection_rate_of_ensemble": round(avg_selection_rate_of_ensemble, 4),
            }

        report["per_scenario"][s] = {
            "seg_name": seg_names[s],
            "mean_kendall_tau": round(mean_tau, 4),
            "by_k": k_results,
        }

    return report


def to_markdown(report: dict) -> str:
    lines = [
        f"# Phase1 랭킹 시드-수렴성 리포트 — `{report['tag']}`",
        "",
        f"- seed 수: {len(report['seeds'])} ({', '.join(report['seeds'])})",
        f"- 시나리오 수: {report['n_scenarios']}",
        "",
        "## 시나리오별 결과",
        "",
        "| 시나리오 | mean Kendall τ | " + " | ".join(f"k={k} Jaccard" for k in report["k_values"]) + " | " +
        " | ".join(f"k={k} 앙상블평균선택률" for k in report["k_values"]) + " |",
        "|---|---|" + "---|" * len(report["k_values"]) + "---|" * len(report["k_values"]),
    ]
    for s, data in report["per_scenario"].items():
        row = [data["seg_name"], f"{data['mean_kendall_tau']:.4f}"]
        row += [f"{data['by_k'][k]['mean_jaccard']:.4f}" for k in report["k_values"]]
        row += [f"{data['by_k'][k]['avg_selection_rate_of_ensemble']:.4f}" for k in report["k_values"]]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 해석 가이드", "",
              "- **Kendall τ**: 1에 가까울수록 seed 간 전체 순위가 일치(수렴). 0 근처면 사실상 무작위.",
              "- **Jaccard**: 1에 가까울수록 top-k 구성원이 seed 간 동일. 낮으면 상위권 자체가 seed마다 바뀜.",
              "- **앙상블평균선택률**: 선택빈도 기반 앙상블 top-k 구성원들이 평균적으로 몇 %의 seed에서 뽑혔는지.",
              "  1.0이면 모든 seed가 동의, 낮으면 앙상블도 소수 seed에 의존하는 불안정한 상태.",
              "",
              "## 시나리오별 앙상블 top-k (선택빈도 기준)", ""]
    for s, data in report["per_scenario"].items():
        lines.append(f"**{data['seg_name']}**")
        for k in report["k_values"]:
            lines.append(f"  - k={k}: {data['by_k'][k]['ensemble_top_k']}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--k-values", type=int, nargs="+", default=[5, 15, 25])
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    report = analyze(manifest_path, args.k_values)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_out = RESULTS_DIR / f"convergence_report_{report['tag']}.json"
    md_out = RESULTS_DIR / f"convergence_report_{report['tag']}.md"
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_out.write_text(to_markdown(report), encoding="utf-8")

    print(to_markdown(report))
    print(f"\n[analyze] 저장: {json_out}")
    print(f"[analyze] 저장: {md_out}")

    # ── 실험 로그 자동 기록 ──────────────────────────────────────────────────
    mean_tau_all = sum(d["mean_kendall_tau"] for d in report["per_scenario"].values()) / len(report["per_scenario"])
    jac_by_k = {
        k: sum(d["by_k"][k]["mean_jaccard"] for d in report["per_scenario"].values()) / len(report["per_scenario"])
        for k in args.k_values
    }
    jac_str = ", ".join(f"k={k}: {v:.3f}" for k, v in jac_by_k.items())
    unstable = mean_tau_all < 0.5 or any(v < 0.5 for v in jac_by_k.values())
    interpretation = (
        ("⚠ 불안정 — Kendall τ 또는 Jaccard가 0.5 미만인 항목 있음. "
         "Stage1(체크포인트 기준)/Stage2(temperature annealing) 적용 후 같은 명령으로 재측정 필요.")
        if unstable else
        "안정적 — 지표가 0.5 이상. 다음 seed 수를 늘려 재확인하거나 Stage4(상관클러스터링)로 넘어가도 됨."
    )
    append_log_entry(
        tag=f"convergence_{report['tag']}",
        purpose=f"Phase1 랭킹 시드-수렴성 측정 ({len(report['seeds'])}개 seed, k={args.k_values})",
        command=current_command_str(),
        result_files=[str(json_out), str(md_out)],
        key_metrics=f"평균 Kendall τ={mean_tau_all:.4f}, 평균 Jaccard({jac_str})",
        interpretation=interpretation,
    )


if __name__ == "__main__":
    main()
