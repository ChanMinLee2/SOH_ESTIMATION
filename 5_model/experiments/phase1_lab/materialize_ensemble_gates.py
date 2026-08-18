"""
5_model/experiments/phase1_lab/materialize_ensemble_gates.py

Stage6 — 다중 시드 앙상블 랭킹을 "가짜 Phase1 run 디렉터리" 형태로 저장한다.

배경(왜 필요한가): train_scr.py의 Phase2(--gates-from)는 진짜 Phase1 run 디렉터리
하나(gates/classification_HIs.json + gates/regression_HIs.json)를 기대한다.
다중 시드 앙상블은 K개 run의 결과를 합친 것이라 그런 폴더가 자연스럽게 존재하지
않는다 — 이 스크립트가 앙상블 결과를 "그 폴더처럼 보이는" 산출물로 물리적으로
만들어줘서, train_scr.py Phase2 코드를 단 한 줄도 안 고치고 --gates-from에
그대로 넘길 수 있게 한다.

랭킹 통합 방법: Borda count(각 seed의 전체 순위에서의 순위값을 평균) — 특정 k에서만
빈도를 세는 것보다 각 seed가 가진 전체 순위 정보를 다 활용하므로, 이후 Phase2에서
어떤 k를 골라도(k=5든 65든) 같은 앙상블 랭킹을 그대로 잘라 쓸 수 있다.

주의: 이 폴더는 "진짜 학습 run"이 아니다 — logs/checkpoints가 없다. 그래서
PROVENANCE.json에 어떤 seed/run들로 만들었는지 반드시 같이 남긴다.

사용 예:
  python 5_model/experiments/phase1_lab/materialize_ensemble_gates.py \
      --manifest 5_model/experiments/phase1_lab/results/convergence_manifest_k25_baseline.json \
      --out-tag k25_ensemble
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from log_utils import append_log_entry, current_command_str

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _borda_ranking(rankings: list[list[int]], n_items: int) -> list[int]:
    avg_rank = np.zeros(n_items)
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            avg_rank[idx] += pos
        # 혹시 랭킹에 없는 인덱스(다른 run과 N_HI가 다른 등 이상 케이스)는 최하위로 취급
        missing = set(range(n_items)) - set(ranking)
        for idx in missing:
            avg_rank[idx] += n_items
    avg_rank /= len(rankings)
    return [int(i) for i in np.argsort(avg_rank)]


def _pseudo_probs(order: list[int]) -> list[float]:
    n = len(order)
    return [round(1.0 - rank / n, 6) for rank in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True,
                   help="run_convergence_seeds.py 또는 동등한 {seed: run_dir} manifest json")
    p.add_argument("--out-tag", required=True)
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    seeds = list(manifest["runs"].keys())
    run_dirs = [Path(manifest["runs"][s]) for s in seeds]
    if len(run_dirs) < 2:
        raise RuntimeError("run이 2개 미만입니다 — 앙상블 의미가 없습니다")

    scen_jsons = [json.loads((d / "gates" / "regression_HIs.json").read_text(encoding="utf-8")) for d in run_dirs]
    probe_jsons = [json.loads((d / "gates" / "classification_HIs.json").read_text(encoding="utf-8")) for d in run_dirs]

    n_scen = 0
    while f"seg_{n_scen}_ranked" in scen_jsons[0]:
        n_scen += 1
    n_hi = len(scen_jsons[0]["seg_0_ranked"])

    out_dir = RESULTS_DIR / "ensembles" / args.out_tag
    (out_dir / "gates").mkdir(parents=True, exist_ok=True)

    # ── regression_HIs.json (시나리오별 앙상블 랭킹) ──────────────────────────
    scen_out: dict = {}
    for s in range(n_scen):
        rankings = [sj[f"seg_{s}_ranked"] for sj in scen_jsons]
        ensemble = _borda_ranking(rankings, n_hi)
        scen_out[f"seg_{s}_ranked"] = ensemble
        scen_out[f"seg_{s}_names"] = scen_jsons[0].get(f"seg_{s}_names", [])
        scen_out[f"seg_{s}_probs"] = _pseudo_probs(ensemble)
        scen_out[f"seg_{s}_seg_name"] = scen_jsons[0].get(f"seg_{s}_seg_name", f"seg_{s}")
    (out_dir / "gates" / "regression_HIs.json").write_text(
        json.dumps(scen_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── classification_HIs.json (충/방전 probe 앙상블 랭킹) ───────────────────
    ch_rankings = [pj["charge_ranked"] for pj in probe_jsons]
    dis_rankings = [pj["discharge_ranked"] for pj in probe_jsons]
    ch_ensemble = _borda_ranking(ch_rankings, n_hi)
    dis_ensemble = _borda_ranking(dis_rankings, n_hi)
    probe_out = {
        "charge_ranked": ch_ensemble,
        "charge_names": probe_jsons[0].get("charge_names", []),
        "charge_probs": _pseudo_probs(ch_ensemble),
        "discharge_ranked": dis_ensemble,
        "discharge_names": probe_jsons[0].get("discharge_names", []),
        "discharge_probs": _pseudo_probs(dis_ensemble),
    }
    (out_dir / "gates" / "classification_HIs.json").write_text(
        json.dumps(probe_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── scenario_spec.json (seed 무관, 축 설정만 같으면 동일 — 첫 run에서 복사) ──
    src_spec = run_dirs[0] / "scenario_spec.json"
    if src_spec.exists():
        shutil.copy(src_spec, out_dir / "scenario_spec.json")

    # ── PROVENANCE — 이 폴더가 "진짜 학습"이 아니라 앙상블 산출물임을 명시 ─────
    provenance = {
        "type": "ensemble_gates",
        "method": "borda_count_over_full_ranking",
        "source_manifest": str(args.manifest),
        "source_seeds": seeds,
        "source_run_dirs": [str(d) for d in run_dirs],
        "note": "이 디렉터리에는 실제 학습 로그/체크포인트가 없습니다 — "
                "gates/*.json만 --gates-from 대상으로 쓰기 위해 합성된 산출물입니다.",
    }
    (out_dir / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[materialize] 앙상블 gates 저장: {out_dir}")
    print(f"[materialize] Phase2에서 사용: python 5_model/train_scr.py --phase 2 "
          f"--gates-from {out_dir} ...")

    append_log_entry(
        tag=f"ensemble_{args.out_tag}",
        purpose=f"다중 시드 앙상블 gates 합성 (Stage6, {len(seeds)}개 seed, Borda count)",
        command=current_command_str(),
        result_files=[str(out_dir / "gates" / "regression_HIs.json"),
                      str(out_dir / "gates" / "classification_HIs.json"),
                      str(out_dir / "PROVENANCE.json")],
        key_metrics=f"source_seeds={seeds}",
        interpretation=f"Phase2 --gates-from {out_dir} 로 바로 사용 가능(가짜 run 디렉터리 — "
                        f"PROVENANCE.json에 출처 기록됨).",
    )


if __name__ == "__main__":
    main()
