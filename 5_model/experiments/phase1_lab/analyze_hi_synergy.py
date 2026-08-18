"""
5_model/experiments/phase1_lab/analyze_hi_synergy.py

다중 HI 시너지 검증 — 독립 실험 스크립트 (기존 5_model 코드 무변경, import만 재사용).

기존 SCRModel/HardConcreteGate 전체 학습 루프를 매 조합마다 돌리는 건 너무 비싸므로,
이 스크립트는 빠른 대리 모델(sklearn Ridge)로 "조합 인식형 랭킹"을 빠르게 스크리닝한다.
여기서 나온 유망 조합은 이후 실제 SCRModel(Phase1)로 재검증하는 걸 전제로 한다
(전략 로드맵 Stage5 참고) — 이 스크립트 자체가 최종 결론은 아니다.

단계:
  A. 개별 HI의 target과의 상관계수 랭킹 (현재 gate_prob 방식과 유사한 "개별 중요도" 베이스라인)
  B. 상위 후보 HI들 간 pairwise Interaction Information (binned MI 기반) — 시너지/중복 스크리닝
  C. Greedy Forward Selection (Ridge 대리모델, val RMSE 기준) — 조합 인식형 랭킹
  D. Ablation 초가법성 검증 (Δi, Δj, Δij) — B에서 나온 유망 쌍이 실제로 초가법적인지 확인
  E. 베이스라인(A의 top-k) vs 조합인식형(C의 top-k) val RMSE 비교 — "시너지 이득" 정량화

데이터 로딩은 5_model/datasets/segment_dataset.py의 build_datasets()를 그대로 재사용한다
(정규화/타깃/스플릿 로직을 이 스크립트에서 새로 만들지 않음 — 실제 학습 파이프라인과 100% 동일).

사용 예:
  python 5_model/experiments/phase1_lab/analyze_hi_synergy.py \
      --model-config 5_model/config/main_qfref_S.yaml \
      --seg-axis q_frac_ref \
      --axis-config '{"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":4}' \
      --data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-4_lag-0_noise-3%_ou-200/cycle" \
      --seg-data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-4_lag-0_noise-3%_ou-200/seg" \
      --scenario dis_hi --k 15 --tag disHi_k15
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import load_config  # noqa: E402
from utils.tqdm_utils import tqdm, write as tqdm_write  # noqa: E402
from datasets.segment_dataset import build_datasets  # noqa: E402
from common.scenario import get_segmenter  # noqa: E402

from log_utils import append_log_entry, current_command_str  # noqa: E402


# ---------------------------------------------------------------------------
# ProcessPoolExecutor 워커 — 큰 배열(x_tr/x_val 등)을 매 태스크마다 다시 pickle하지
# 않도록, 워커 프로세스당 한 번만 initializer로 전달해 전역에 캐싱한다.
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(x_tr, mask_tr, y_tr, x_val, mask_val, y_val, n_bins) -> None:
    _WORKER["x_tr"] = x_tr
    _WORKER["mask_tr"] = mask_tr
    _WORKER["y_tr"] = y_tr
    _WORKER["x_val"] = x_val
    _WORKER["mask_val"] = mask_val
    _WORKER["y_val"] = y_val
    _WORKER["n_bins"] = n_bins


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HI 조합(시너지) 스크리닝 — 독립 실험")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", required=True)
    p.add_argument("--axis-config", required=True)
    p.add_argument("--data-dir", required=True, help="cycle pkl 경로 (train_scr.py가 자동계산하는 것과 동일 경로를 직접 지정)")
    p.add_argument("--seg-data-dir", required=True, help="seg pkl 경로")
    p.add_argument("--datasets", nargs="+", default=["MIT", "HUST"])
    p.add_argument("--scenario", required=True, help="시나리오 이름(예: dis_hi) 또는 'pooled'(전체 풀링)")
    p.add_argument("--k", type=int, default=15, help="최종 비교할 top-k 크기")
    p.add_argument("--n-mi-candidates", type=int, default=20, help="pairwise MI 스크리닝 후보 수(상관계수 상위 N개)")
    p.add_argument("--n-ablation-pairs", type=int, default=8, help="초가법성 검증할 상위 시너지 쌍 개수")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                   help="Stage B/C 병렬 워커 수 (기본: CPU 코어수-1)")
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    return args


def _load_train_val(args) -> tuple:
    cfg = load_config(args.model_config)
    cfg.setdefault("data", {})
    cfg["data"]["data_dir"] = args.data_dir
    cfg["data"]["seg_data_dir"] = args.seg_data_dir
    cfg["data"]["datasets"] = args.datasets
    cfg["data"]["split_seed"] = args.split_seed

    axis_cfg = json.loads(args.axis_config)
    spec = get_segmenter(args.seg_axis, {args.seg_axis: axis_cfg}).get_spec()
    train_ds, val_ds, _test_ds, _norm = build_datasets(cfg, spec=spec)

    hi_names = [f"hi_{i:02d}" for i in range(train_ds.x_hi.shape[1])]

    def _extract(ds):
        x = ds.x_hi.numpy()
        mask = ds.nan_mask.numpy()
        y = ds.target.numpy()
        seg_idx = ds.seg_idx.numpy()
        if args.scenario != "pooled":
            names = spec.scenario_names
            if args.scenario not in names:
                raise ValueError(f"scenario '{args.scenario}' not in {names}")
            sid = names.index(args.scenario)
            sel = seg_idx == sid
            x, mask, y = x[sel], mask[sel], y[sel]
        return x, mask, y

    x_tr, mask_tr, y_tr = _extract(train_ds)
    x_val, mask_val, y_val = _extract(val_ds)
    print(f"[data] scenario={args.scenario}  train={len(y_tr):,}  val={len(y_val):,}  N_HI={x_tr.shape[1]}")
    return x_tr, mask_tr, y_tr, x_val, mask_val, y_val, hi_names


# ---------------------------------------------------------------------------
# A. 개별 상관계수 랭킹 (baseline)
# ---------------------------------------------------------------------------

def marginal_correlation_ranking(x: np.ndarray, y: np.ndarray) -> list[int]:
    corrs = []
    for i in range(x.shape[1]):
        col = x[:, i]
        if np.nanstd(col) < 1e-8:
            corrs.append(0.0)
            continue
        c = np.corrcoef(col, y)[0, 1]
        corrs.append(0.0 if np.isnan(c) else abs(c))
    return list(np.argsort(corrs)[::-1])


# ---------------------------------------------------------------------------
# B. Pairwise Interaction Information (binned MI)
# ---------------------------------------------------------------------------

def _bin(col: np.ndarray, n_bins: int = 8) -> np.ndarray:
    ranks = np.argsort(np.argsort(col))
    return np.minimum((ranks * n_bins) // len(col), n_bins - 1)


def _mi_pair_task(pair: tuple[int, int]) -> tuple[int, int, float]:
    """워커 프로세스에서 실행 — _WORKER 전역(initializer로 한 번만 세팅됨)을 읽는다."""
    from sklearn.metrics import mutual_info_score

    i, j = pair
    x_tr, y_tr, n_bins = _WORKER["x_tr"], _WORKER["y_tr"], _WORKER["n_bins"]
    y_bin = _bin(y_tr, n_bins)
    bi, bj = _bin(x_tr[:, i], n_bins), _bin(x_tr[:, j], n_bins)
    mi_i = mutual_info_score(bi, y_bin)
    mi_j = mutual_info_score(bj, y_bin)
    joint = bi * n_bins + bj
    i_joint = mutual_info_score(joint, y_bin)
    return i, j, float(i_joint - mi_i - mi_j)


def pairwise_interaction_info(executor: ProcessPoolExecutor,
                               candidates: list[int]) -> list[tuple[int, int, float]]:
    pairs = list(itertools.combinations(candidates, 2))
    results = list(tqdm(executor.map(_mi_pair_task, pairs), total=len(pairs),
                         desc="[B] pairwise interaction info", unit="pair"))
    results.sort(key=lambda t: -t[2])
    return results


# ---------------------------------------------------------------------------
# C. Greedy Forward Selection (Ridge 대리모델)
# ---------------------------------------------------------------------------

def _fit_eval(x_tr, mask_tr, y_tr, x_val, mask_val, y_val, indices: list[int]) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    xt = (x_tr[:, indices] * mask_tr[:, indices])
    xv = (x_val[:, indices] * mask_val[:, indices])
    model = Ridge(alpha=1.0)
    model.fit(xt, y_tr)
    pred = model.predict(xv)
    return float(np.sqrt(mean_squared_error(y_val, pred)))


def _eval_candidate_task(indices: list[int]) -> tuple[list[int], float]:
    """워커 프로세스에서 실행 — _WORKER 전역을 읽어 후보 HI 집합 하나를 평가."""
    d = _WORKER
    rmse = _fit_eval(d["x_tr"], d["mask_tr"], d["y_tr"], d["x_val"], d["mask_val"], d["y_val"], indices)
    return indices, rmse


def greedy_forward_selection(executor: ProcessPoolExecutor, n_hi: int,
                              k: int, pool: list[int] | None = None) -> list[int]:
    pool = pool if pool is not None else list(range(n_hi))
    selected: list[int] = []
    remaining = list(pool)

    for step in tqdm(range(k), desc="[C] greedy forward selection", unit="step"):
        candidate_sets = [selected + [c] for c in remaining]
        results = list(tqdm(executor.map(_eval_candidate_task, candidate_sets),
                             total=len(candidate_sets), desc=f"  step {step+1}/{k} 후보평가",
                             unit="cand", leave=False))
        best_indices, best_rmse = min(results, key=lambda r: r[1])
        best_idx = best_indices[-1]
        selected.append(best_idx)
        remaining.remove(best_idx)
        tqdm_write(f"  [greedy] step {step+1}/{k}: +HI_{best_idx:02d}  val_rmse={best_rmse:.5f}")

    return selected


# ---------------------------------------------------------------------------
# D. Ablation 초가법성 검증
# ---------------------------------------------------------------------------

def ablation_superadditivity(x_tr, mask_tr, y_tr, x_val, mask_val, y_val,
                              full_set: list[int], pairs: list[tuple[int, int, float]]) -> list[dict]:
    base_rmse = _fit_eval(x_tr, mask_tr, y_tr, x_val, mask_val, y_val, full_set)
    out = []
    for i, j, mi_synergy in pairs:
        set_wo_i = [h for h in full_set if h != i]
        set_wo_j = [h for h in full_set if h != j]
        set_wo_ij = [h for h in full_set if h not in (i, j)]
        rmse_wo_i = _fit_eval(x_tr, mask_tr, y_tr, x_val, mask_val, y_val, set_wo_i)
        rmse_wo_j = _fit_eval(x_tr, mask_tr, y_tr, x_val, mask_val, y_val, set_wo_j)
        rmse_wo_ij = _fit_eval(x_tr, mask_tr, y_tr, x_val, mask_val, y_val, set_wo_ij)
        d_i = rmse_wo_i - base_rmse
        d_j = rmse_wo_j - base_rmse
        d_ij = rmse_wo_ij - base_rmse
        out.append({
            "hi_i": i, "hi_j": j, "mi_interaction_info": mi_synergy,
            "delta_i": d_i, "delta_j": d_j, "delta_ij": d_ij,
            "superadditive_gap": d_ij - (d_i + d_j),  # >0 이면 초가법(시너지)
        })
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    x_tr, mask_tr, y_tr, x_val, mask_val, y_val, hi_names = _load_train_val(args)

    print("\n=== A. 개별 상관계수 랭킹 ===")
    corr_ranking = marginal_correlation_ranking(x_tr, y_tr)
    baseline_top_k = corr_ranking[:args.k]
    baseline_rmse = _fit_eval(x_tr, mask_tr, y_tr, x_val, mask_val, y_val, baseline_top_k)
    print(f"  top-{args.k} (상관계수 기준): {baseline_top_k}  val_rmse={baseline_rmse:.5f}")

    print(f"\n[병렬] Stage B/C용 워커 {args.workers}개 프로세스 풀 생성 중...")
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker,
        initargs=(x_tr, mask_tr, y_tr, x_val, mask_val, y_val, 8),
    ) as executor:

        print(f"\n=== B. Pairwise Interaction Information (후보 {args.n_mi_candidates}개) ===")
        mi_candidates = corr_ranking[:args.n_mi_candidates]
        interactions = pairwise_interaction_info(executor, mi_candidates)
        for i, j, syn in interactions[:10]:
            tag = "시너지" if syn > 0 else "중복"
            print(f"  HI_{i:02d} x HI_{j:02d}: interaction_info={syn:+.5f} ({tag})")

        print(f"\n=== C. Greedy Forward Selection (k={args.k}) ===")
        greedy_top_k = greedy_forward_selection(executor, x_tr.shape[1], args.k)

    greedy_rmse = _fit_eval(x_tr, mask_tr, y_tr, x_val, mask_val, y_val, greedy_top_k)

    print(f"\n=== D. Ablation 초가법성 검증 (상위 {args.n_ablation_pairs}쌍) ===")
    top_pairs = interactions[:args.n_ablation_pairs]
    ablation_results = ablation_superadditivity(x_tr, mask_tr, y_tr, x_val, mask_val, y_val,
                                                  greedy_top_k, top_pairs)
    for r in ablation_results:
        tag = "초가법(시너지 확인)" if r["superadditive_gap"] > 0 else "부분가법(중복 가능성)"
        print(f"  HI_{r['hi_i']:02d}+HI_{r['hi_j']:02d}: gap={r['superadditive_gap']:+.5f} ({tag})")

    print("\n=== E. 베이스라인 vs 조합인식형 비교 ===")
    gain = baseline_rmse - greedy_rmse
    print(f"  baseline(상관계수 top-{args.k}) val_rmse = {baseline_rmse:.5f}")
    print(f"  greedy(조합인식) top-{args.k}   val_rmse = {greedy_rmse:.5f}")
    print(f"  개선폭 = {gain:+.5f}  ({'시너지 이득 확인' if gain > 0 else '이득 없음 — 개별 랭킹으로 충분'})")

    report = {
        "tag": args.tag, "scenario": args.scenario, "k": args.k,
        "hi_names": hi_names,
        "baseline_top_k": baseline_top_k, "baseline_val_rmse": baseline_rmse,
        "greedy_top_k": greedy_top_k, "greedy_val_rmse": greedy_rmse,
        "synergy_gain": gain,
        "top_pairwise_interactions": [{"hi_i": i, "hi_j": j, "interaction_info": s} for i, j, s in interactions[:20]],
        "ablation_results": ablation_results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"synergy_report_{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[synergy] 저장: {out_path}")

    # ── 실험 로그 자동 기록 ──────────────────────────────────────────────────
    n_superadditive = sum(1 for r in ablation_results if r["superadditive_gap"] > 0)
    interpretation = (
        f"시너지 이득 확인(개선폭 {gain:+.5f}) — 초가법 쌍 {n_superadditive}/{len(ablation_results)}개. "
        "실제 SCRModel(HardConcreteGate)로 재검증 권장."
        if gain > 0 else
        f"이득 없음(개선폭 {gain:+.5f}) — 개별 상관계수 랭킹으로 충분, 조합탐색 불필요."
    )
    append_log_entry(
        tag=f"synergy_{args.tag}",
        purpose=f"HI 조합 시너지 검증 (scenario={args.scenario}, k={args.k})",
        command=current_command_str(),
        result_files=[str(out_path)],
        key_metrics=(f"baseline_val_rmse={baseline_rmse:.5f}, greedy_val_rmse={greedy_rmse:.5f}, "
                      f"synergy_gain={gain:+.5f}, 초가법쌍={n_superadditive}/{len(ablation_results)}"),
        interpretation=interpretation,
    )


if __name__ == "__main__":
    main()
