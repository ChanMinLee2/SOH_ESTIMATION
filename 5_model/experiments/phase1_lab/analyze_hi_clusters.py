"""
5_model/experiments/phase1_lab/analyze_hi_clusters.py

Stage4 — 상관 HI 클러스터링으로 "불안정성이 진짜인지" 재해석.

analyze_convergence.py의 Jaccard/Kendall은 "HI 인덱스가 그대로 일치하는가"만
본다. 하지만 서로 정보량이 거의 같은 HI 두 개(상관계수 0.9 이상)가 시드마다
번갈아 뽑히는 건 사실 무해한 동률 교체다 — 이 스크립트는 HI들을 상관관계로
클러스터링한 뒤, "클러스터 단위로 봐도 랭킹이 안 흔들리는지"를 별도 지표로
계산해서 원래 Jaccard와 나란히 보여준다.

사용 예:
  python 5_model/experiments/phase1_lab/analyze_hi_clusters.py \
      --manifest 5_model/experiments/phase1_lab/results/convergence_manifest_k25_baseline.json \
      --model-config 5_model/config/main_qfref_S.yaml \
      --seg-axis q_frac_ref \
      --axis-config '{"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":4}' \
      --data-dir "D:/.../cycle" --seg-data-dir "D:/.../seg" \
      --corr-threshold 0.9 --k-values 5 15 25 --tag k25_clusters
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import load_config  # noqa: E402
from datasets.segment_dataset import build_datasets  # noqa: E402
from common.scenario import get_segmenter  # noqa: E402

from log_utils import append_log_entry, current_command_str  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HI 상관 클러스터링 기반 불안정성 재해석")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", required=True)
    p.add_argument("--axis-config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--seg-data-dir", required=True)
    p.add_argument("--datasets", nargs="+", default=["MIT", "HUST"])
    p.add_argument("--corr-threshold", type=float, default=0.9,
                   help="|상관계수| >= 이 값이면 같은 클러스터로 묶음 (기본 0.9)")
    p.add_argument("--k-values", type=int, nargs="+", default=[5, 15, 25])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--tag", required=True)
    return p.parse_args()


def _build_clusters(x: np.ndarray, threshold: float) -> np.ndarray:
    """HI 상관행렬 -> 계층적 클러스터링 -> 각 HI의 클러스터 라벨 배열(길이 N_HI)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # 부동소수점 비대칭 보정
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    return labels


def _cluster_aware_jaccard(a: list[int], b: list[int], cluster_of: np.ndarray) -> float:
    ca = {int(cluster_of[i]) for i in a}
    cb = {int(cluster_of[i]) for i in b}
    if not ca and not cb:
        return 1.0
    return len(ca & cb) / len(ca | cb)


def main() -> None:
    args = _parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    seeds = list(manifest["runs"].keys())

    cfg = load_config(args.model_config)
    cfg.setdefault("data", {})
    cfg["data"]["data_dir"] = args.data_dir
    cfg["data"]["seg_data_dir"] = args.seg_data_dir
    cfg["data"]["datasets"] = args.datasets
    cfg["data"]["split_seed"] = args.split_seed
    axis_cfg = json.loads(args.axis_config)
    spec = get_segmenter(args.seg_axis, {args.seg_axis: axis_cfg}).get_spec()
    train_ds, _val_ds, _test_ds, _norm = build_datasets(cfg, spec=spec)
    x_all = train_ds.x_hi.numpy()
    seg_idx_all = train_ds.seg_idx.numpy()

    gates_by_seed = {s: json.loads((Path(manifest["runs"][s]) / "gates" / "regression_HIs.json")
                                    .read_text(encoding="utf-8")) for s in seeds}
    n_scen = 0
    while f"seg_{n_scen}_ranked" in gates_by_seed[seeds[0]]:
        n_scen += 1

    report: dict = {"tag": args.tag, "corr_threshold": args.corr_threshold, "per_scenario": {}}

    for s in range(n_scen):
        seg_name = gates_by_seed[seeds[0]].get(f"seg_{s}_seg_name", f"seg_{s}")
        sel = seg_idx_all == s
        x_scen = x_all[sel]
        if x_scen.shape[0] < 10:
            print(f"[cluster] {seg_name}: 표본 부족({x_scen.shape[0]}) — 스킵")
            continue

        cluster_of = _build_clusters(x_scen, args.corr_threshold)
        n_clusters = len(set(cluster_of.tolist()))
        rankings = {seed: gates_by_seed[seed][f"seg_{s}_ranked"] for seed in seeds}

        by_k = {}
        for k in args.k_values:
            raw_jacs, cluster_jacs = [], []
            for a, b in itertools.combinations(seeds, 2):
                raw_jacs.append(len(set(rankings[a][:k]) & set(rankings[b][:k])) /
                                 len(set(rankings[a][:k]) | set(rankings[b][:k])))
                cluster_jacs.append(_cluster_aware_jaccard(rankings[a][:k], rankings[b][:k], cluster_of))
            by_k[k] = {
                "raw_jaccard": round(sum(raw_jacs) / len(raw_jacs), 4),
                "cluster_aware_jaccard": round(sum(cluster_jacs) / len(cluster_jacs), 4),
            }

        report["per_scenario"][s] = {
            "seg_name": seg_name, "n_hi": x_scen.shape[1], "n_clusters": n_clusters,
            "by_k": by_k,
        }
        print(f"[cluster] {seg_name}: N_HI={x_scen.shape[1]} -> {n_clusters}개 클러스터 "
              f"(threshold=|r|>={args.corr_threshold})")
        for k in args.k_values:
            r = by_k[k]
            gap = r["cluster_aware_jaccard"] - r["raw_jaccard"]
            note = "동급 HI 교체가 대부분(무해)" if gap > 0.15 else "클러스터 보정해도 여전히 불안정"
            print(f"    k={k:3d}: raw_jaccard={r['raw_jaccard']:.3f}  "
                  f"cluster_aware={r['cluster_aware_jaccard']:.3f}  (Δ={gap:+.3f}, {note})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"cluster_report_{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[cluster] 저장: {out_path}")

    mean_gap = np.mean([
        report["per_scenario"][s]["by_k"][k]["cluster_aware_jaccard"] - report["per_scenario"][s]["by_k"][k]["raw_jaccard"]
        for s in report["per_scenario"] for k in args.k_values
    ]) if report["per_scenario"] else 0.0
    interpretation = (
        f"평균 Δ(cluster_aware-raw)={mean_gap:+.3f} — "
        + ("불안정성 대부분이 동급 HI 간 무해한 교체로 설명됨." if mean_gap > 0.15 else
           "클러스터로 보정해도 격차가 안 줄어듦 — 진짜 불안정성, Stage1/2 재점검 필요.")
    )
    append_log_entry(
        tag=f"clusters_{args.tag}",
        purpose=f"HI 상관 클러스터링으로 불안정성 재해석 (threshold=|r|>={args.corr_threshold})",
        command=current_command_str(),
        result_files=[str(out_path)],
        key_metrics=f"평균 raw_jaccard vs cluster_aware_jaccard 격차 = {mean_gap:+.3f}",
        interpretation=interpretation,
    )


if __name__ == "__main__":
    main()
